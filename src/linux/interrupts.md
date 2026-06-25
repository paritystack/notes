# Interrupts & Deferred Work

## Overview

This page covers how the kernel handles **hardware interrupts (IRQs)** and the machinery for
pushing work *out* of the interrupt path — the **bottom halves**: softirqs, tasklets, threaded
IRQs, and **workqueues**, plus **NAPI** for high-rate network devices. It is the subsystem-level
companion to the driver-author view in [Driver Development](driver_development.md) (its
*Interrupts* and *DMA* sections), and it is where the "can this code sleep?" rule from
[Synchronization](synchronization.md) comes from. The deferred-work primitives overlap with the
timed deferral in [Kernel Timers](kernel_timers.md), and the execution contexts referenced here
are defined in [Process Internals (Kernel)](process_internals.md). NAPI ties into the receive
path in [Networking](networking.md) and is a common observation point for [eBPF](ebpf.md).

An interrupt is a hardware signal that preempts whatever the CPU is doing to run a handler. The
golden rule: the handler runs with the CPU "stolen" from some unrelated task (and often with
interrupts disabled), so it must be **fast and non-blocking**. Anything slow gets *deferred* to
a context where the kernel is allowed to schedule.

```
   device raises IRQ line
          │
          ▼
   ┌──────────────────┐   hard IRQ (top half) — atomic, must be tiny
   │  interrupt handler│   ack device, grab data, schedule bottom half
   └────────┬─────────┘
            │ defer
            ▼
   ┌─────────────────────────────────────────────┐
   │  bottom half — runs later, lower priority    │
   │   softirq ─► tasklet      (atomic, no sleep)  │
   │   threaded IRQ / workqueue (process, may sleep)│
   └─────────────────────────────────────────────┘
```

## The top half / bottom half split

Historically Linux splits interrupt handling in two:

- **Top half (hard IRQ handler)** — runs immediately, in *interrupt context*, usually with that
  IRQ line masked. It cannot sleep, cannot call blocking allocators, and cannot be scheduled.
  Do the minimum: acknowledge the device, read out time-critical state, and schedule the rest.
- **Bottom half** — the deferred remainder, run later in a context where more is permitted.

Why split at all? While a hard IRQ runs, that line (and sometimes all local interrupts) is
masked, hurting latency for everything else. Keeping the top half short keeps the system
responsive.

## Registering a handler

```c
/* request_threaded_irq() is the modern primitive; request_irq() is a wrapper
 * with thread_fn == NULL. */
ret = request_irq(irq, my_handler, IRQF_SHARED, "mydev", dev);

irqreturn_t my_handler(int irq, void *dev_id)
{
    struct mydev *d = dev_id;
    u32 status = readl(d->regs + STATUS);
    if (!(status & MY_IRQ_BIT))
        return IRQ_NONE;          /* not ours — shared line */
    writel(status, d->regs + STATUS_CLEAR);   /* ack */
    schedule_work(&d->work);       /* defer the heavy lifting */
    return IRQ_HANDLED;
}
```

Key `IRQF_*` flags: `IRQF_SHARED` (line shared between devices — the handler *must* check it was
its device and return `IRQ_NONE` otherwise), `IRQF_ONESHOT` (keep the line masked until the
threaded handler finishes — required for threaded IRQs without a hard-IRQ part), and
`IRQF_NO_SUSPEND`/`IRQF_TRIGGER_*`. `dev_id` must be unique on shared lines and is what
`free_irq()` uses to find the right handler.

## Choosing a deferral mechanism

| Mechanism      | Context  | Can sleep? | Concurrency                              | Use when |
|----------------|----------|-----------|-------------------------------------------|----------|
| **softirq**    | atomic   | No        | same softirq runs concurrently on CPUs    | very high frequency, core subsystems only |
| **tasklet**    | atomic   | No        | same tasklet serialised across all CPUs   | simple "finish the IRQ" work, no sleeping |
| **threaded IRQ** | process | **Yes**  | one thread per IRQ                         | handler needs to sleep (I2C/SPI access, mutex) |
| **workqueue**  | process  | **Yes**  | configurable (per-CPU / unbound pool)      | general deferred work that may block |

```
 latency / priority           may sleep?
   high  softirq  ───────────────  no
    │    tasklet  (built on softirq) no
    │    threaded IRQ              yes  ◄─ per-IRQ kthread
   low   workqueue                yes  ◄─ kworker pool
```

### softirqs

Softirqs are a *fixed, compile-time* set (`NET_RX_SOFTIRQ`, `NET_TX_SOFTIRQ`, `TIMER_SOFTIRQ`,
`BLOCK_SOFTIRQ`, `TASKLET_SOFTIRQ`, `RCU_SOFTIRQ`, …). You don't add new ones from a driver —
they're reserved for the hottest paths. They run on the same CPU that raised them, after the
hard IRQ returns (in `irq_exit()`), or in the `ksoftirqd/N` kthread when the load is heavy.
Because the *same* softirq can run on multiple CPUs simultaneously, its data must be
per-CPU or properly locked.

### tasklets

A tasklet is a softirq specialisation (`TASKLET_SOFTIRQ`) that's easy to use from drivers and
guarantees a given tasklet **never runs on two CPUs at once** — so its callback needs less
locking against itself. Still atomic: no sleeping. (Tasklets are gradually being phased out in
favour of threaded IRQs and the BH workqueue, but remain widespread.)

```c
void my_tasklet_fn(struct tasklet_struct *t) { /* atomic work */ }
DECLARE_TASKLET(my_tasklet, my_tasklet_fn);
/* in the hard IRQ handler: */
tasklet_schedule(&my_tasklet);
```

### Threaded IRQs

Register both a quick *hard-IRQ* primary handler and a *threaded* secondary that runs in its own
kthread (process context, may sleep). The primary returns `IRQ_WAKE_THREAD` to hand off:

```c
request_threaded_irq(irq, primary_check, threaded_work,
                     IRQF_ONESHOT, "mydev", dev);

irqreturn_t primary_check(int irq, void *id) { return IRQ_WAKE_THREAD; }
irqreturn_t threaded_work(int irq, void *id) {
    /* runs in kthread: safe to take a mutex, do I2C/SPI, etc. */
    return IRQ_HANDLED;
}
```

This is the preferred model for devices behind a sleeping bus (I2C/SPI sensors, etc.) and it's
what `PREEMPT_RT` forces for almost all IRQs.

### Workqueues

Workqueues run callbacks in `kworker` kthreads — full process context, may sleep, may be
preempted. Use the shared `system_wq` via `schedule_work()`/`schedule_delayed_work()`, or create
a dedicated queue with `alloc_workqueue()` (e.g. `WQ_UNBOUND`, `WQ_HIGHPRI`, `WQ_MEM_RECLAIM`).
`delayed_work` adds a timer for "run this after N jiffies" (overlaps with
[Kernel Timers](kernel_timers.md)). This is the general-purpose deferral for anything that can
block.

## NAPI: interrupt mitigation for the network RX path

At high packet rates, one IRQ per packet melts the CPU. **NAPI** flips to polling under load: on
the first RX interrupt the driver *disables* RX interrupts and schedules a poll; the napi poll
function then drains the ring in batches (up to a `budget`) from softirq context, re-enabling
interrupts only when the queue empties.

```
 low traffic:   IRQ ─► process 1 packet ─► IRQ ─► …      (interrupt-driven)
 high traffic:  IRQ ─► disable IRQ ─► poll batch ─► poll batch ─► … ─► re-enable IRQ
```

This adaptive switch bounds interrupt rate while keeping latency low when idle. The poll runs in
`NET_RX_SOFTIRQ`, which is why softirq time shows up as network load.

## Observing and tuning

```bash
# Per-CPU interrupt counts by IRQ line
cat /proc/interrupts

# softirq counts per CPU
cat /proc/softirqs

# Pin an IRQ to specific CPUs (smp_affinity is a CPU bitmask)
echo 4 > /proc/irq/24/smp_affinity        # CPU2
cat   /proc/irq/24/smp_affinity_list

# Time spent in hard/soft IRQ (hi/si columns)
mpstat -P ALL 1
```

`irqbalance` spreads IRQs across CPUs automatically; pin manually for latency-sensitive or
NUMA-local workloads. softirq pressure ("ksoftirqd" eating a CPU) usually means a network or
timer storm.

## Where this connects

- [Driver Development](driver_development.md) — the driver-side recipe for `request_irq`, DMA
  completion, and bottom halves; this page is the subsystem context behind it.
- [Synchronization](synchronization.md) — IRQ/softirq handlers are *atomic context*, so shared
  data needs `spin_lock_irqsave()`/`spin_lock_bh()`, never a sleeping lock.
- [Kernel Timers](kernel_timers.md) — `delayed_work` and timer callbacks are the timed cousins
  of the deferral mechanisms here; timer callbacks also run in softirq context.
- [Process Internals (Kernel)](process_internals.md) — defines process vs interrupt context and
  why only process-context work (threaded IRQ, workqueue) may sleep.
- [Networking](networking.md) / [eBPF](ebpf.md) — NAPI drives the RX softirq path that XDP and
  many tracing probes hook into.

## Pitfalls

- **Doing too much in the hard IRQ.** Long top halves keep the line masked and wreck latency.
  Ack and defer; measure with `/proc/interrupts` deltas.
- **Sleeping in atomic context.** Calling a blocking allocator (`GFP_KERNEL`), taking a mutex,
  or `msleep()` inside a hard IRQ / softirq / tasklet panics or deadlocks. Use a threaded IRQ or
  workqueue instead.
- **Shared-IRQ handler that doesn't check ownership.** On `IRQF_SHARED` lines, returning
  `IRQ_HANDLED` when it wasn't your device hides others' interrupts; always test the status
  register and return `IRQ_NONE`.
- **Assuming a softirq runs on one CPU.** The same softirq can run concurrently on every CPU;
  only *tasklets* are serialised against themselves. Protect per-device state accordingly.
- **`free_irq` with the wrong `dev_id`.** On shared lines the cookie identifies your handler;
  passing the wrong (or NULL) `dev_id` frees the wrong one or fails.
- **Forgetting `IRQF_ONESHOT` on threaded-only IRQs.** Without it the line is unmasked before
  the threaded handler clears the source, causing an interrupt storm.
- **Long-running work items blocking the shared queue.** A blocking work item on `system_wq` can
  stall others; use a dedicated `WQ_UNBOUND`/`WQ_MEM_RECLAIM` queue for slow or reclaim-path work.
