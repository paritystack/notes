# CPU Scheduler

## Overview

This page covers how the kernel decides **which task runs on which CPU and for how long** — the
scheduling classes, the fair scheduler's evolution from **CFS to EEVDF**, real-time policies,
preemption, SMP load balancing, and the cgroup **cpu** controller. It expands the brief
*Scheduling* and *Context Switch* sections in
[Process Internals (Kernel)](process_internals.md) (which defines `task_struct`, task states,
and the mechanics of switching) — read that first for the data structures. The cgroup integration
ties into [Control Groups (cgroups)](cgroups.md); the userspace knobs (`nice`, `chrt`, affinity)
connect to [Process Management](process.md), and "uninterruptible sleep" / wakeups link back to
the locking rules in [Synchronization](synchronization.md).

The scheduler runs whenever a task blocks, a timer tick fires, a task is woken, or a syscall
returns — it picks the next runnable task per CPU and performs the context switch. The design
goal is conflicting: be *fair* to interactive tasks (low latency) while keeping throughput high
and honouring hard real-time deadlines.

```
   schedule()  ── ask each class, highest priority first ──►
   ┌────────────────────────────────────────────────────┐
   │ stop_sched_class      (migration/hotplug — absolute) │
   │ dl_sched_class        SCHED_DEADLINE  (EDF)          │
   │ rt_sched_class        SCHED_FIFO / SCHED_RR          │
   │ fair_sched_class      SCHED_NORMAL/BATCH  (CFS→EEVDF)│ ◄─ most tasks
   │ idle_sched_class      SCHED_IDLE / the idle task     │
   └────────────────────────────────────────────────────┘
   first class with a runnable task wins
```

## Scheduling classes and policies

Each runqueue is a stack of **scheduling classes** queried in fixed priority order. A lower class
only runs when all higher ones are empty. User-visible **policies** map onto these classes:

| Policy           | Class | Semantics |
|------------------|-------|-----------|
| `SCHED_DEADLINE` | dl    | EDF: each task has (runtime, deadline, period); the earliest deadline runs. Admission-controlled. |
| `SCHED_FIFO`     | rt    | Fixed priority 1–99, runs until it blocks/yields — *no* timeslice. |
| `SCHED_RR`       | rt    | Like FIFO but round-robins among equal priorities with a timeslice. |
| `SCHED_NORMAL`   | fair  | The default; weighted by `nice` (−20…+19). |
| `SCHED_BATCH`    | fair  | Like NORMAL but assumed non-interactive (no wakeup preemption boost). |
| `SCHED_IDLE`     | fair  | Lowest weight; runs only when nothing else wants the CPU. |

Real-time policies (FIFO/RR) always preempt fair tasks, so a runaway `SCHED_FIFO` task can
starve the whole system — `sched_rt_runtime_us` reserves a slice (default 95%) for RT to bound
this.

## CFS → EEVDF (the fair class)

For ~15 years the fair class was **CFS** (Completely Fair Scheduler): each task accrues
**virtual runtime** (`vruntime`) that advances faster for low-priority (high-nice) tasks via a
weight; the scheduler picks the task with the smallest `vruntime` from a red-black tree, so over
time every task gets a CPU share proportional to its weight.

Since **6.6**, the fair class is **EEVDF** (Earliest Eligible Virtual Deadline First). EEVDF adds
*latency* as a first-class notion:

- Each task has a **lag** — how much CPU it is owed vs its fair share. A task is **eligible** when
  its lag ≥ 0 (it hasn't run ahead of its entitlement).
- Each task is assigned a **virtual deadline** = eligible-time + its requested time slice.
- The scheduler runs the **eligible task with the earliest virtual deadline**.

The practical win: a task can ask for a *shorter slice* (lower latency, more frequent scheduling)
without claiming a larger CPU *share* — interactivity and fairness are decoupled. `nice` still
sets the weight (CPU proportion); the new `sched_attr` field `sched_runtime` hints the slice.

```
 weight from nice  ──►  CPU *share*   (how much)
 EEVDF deadline    ──►  scheduling *latency* (how often / how soon)
```

## Preemption and ticks

A task keeps running until: it blocks, it's preempted by a higher class waking, its slice expires
(checked at the timer tick / `task_tick`), or it returns to userspace with `need_resched` set.
Preemption models (build-time, plus the runtime `preempt=` boot/`debugfs` switch on recent
kernels):

- **`PREEMPT_NONE`** — server: no kernel-side preemption; reschedule only at well-defined points.
- **`PREEMPT_VOLUNTARY`** — desktop: adds explicit `might_resched()` points.
- **`PREEMPT` (full)** — low-latency: kernel code is preemptible except in atomic sections.
- **`PREEMPT_RT`** — makes almost everything preemptible (most spinlocks sleep; see
  [Synchronization](synchronization.md)).

`need_resched` is a per-task flag; the actual switch happens at the next safe point
(`preempt_enable()`, IRQ return, syscall return).

## SMP: load balancing and topology

On multiprocessors each CPU has its own runqueue. The scheduler keeps them balanced via
**scheduling domains** that mirror the hardware topology (SMT siblings → cores → LLC/cache →
NUMA nodes). Periodic and idle balancing pull tasks toward idle CPUs, but the cost rises with
domain distance — migrating across NUMA nodes is expensive (cold caches, remote memory).

```
 NUMA node ──┐
   LLC       ├─ balance cheaply within, reluctantly across
     core    │
       SMT ──┘
```

Wakeup placement (`select_task_rq`) tries to wake a task on a CPU sharing cache with the waker,
trading perfect balance for cache warmth. Pin tasks with `taskset`/`sched_setaffinity()` or
isolate CPUs with `isolcpus=`/cpusets for latency-critical workloads.

## The cgroup cpu controller

The fair class is hierarchical: cgroups are scheduler entities too, so CPU time is divided
*between groups* first, then among tasks within a group.

- **`cpu.weight`** (v2; `cpu.shares` in v1) — proportional share when CPUs are contended.
- **`cpu.max`** — hard bandwidth cap: `"quota period"` (e.g. `50000 100000` = 50ms per 100ms =
  half a core). This is how container CPU limits work — and the source of **throttling** stalls
  when a bursty app exceeds quota.

See [Control Groups (cgroups)](cgroups.md) for the controller hierarchy and
[Container Runtimes](container_runtimes.md) for how limits are applied to containers.

## Userspace knobs & observation

```bash
nice -n 10 ./batch_job            # start with lower priority
renice -n 5 -p 1234               # change a running task's nice
chrt -f 80 ./rt_task              # SCHED_FIFO priority 80
chrt -p 1234                      # show a task's policy/priority
taskset -c 2,3 ./pinned           # restrict to CPUs 2,3

cat /proc/<pid>/schedstat         # run time, wait time, timeslices
cat /proc/<pid>/sched             # detailed per-task scheduler stats
cat /sys/kernel/debug/sched/debug # per-CPU runqueues (root)
perf sched record -- sleep 5; perf sched latency   # wakeup→run latency
```

`schedstat`/`perf sched` expose the metric that usually matters: **scheduling latency**
(wakeup-to-run delay), not raw CPU%.

## Where this connects

- [Process Internals (Kernel)](process_internals.md) — `task_struct`, run states, and the
  context-switch mechanics this page builds on; start there.
- [Control Groups (cgroups)](cgroups.md) — the cpu controller's weight/quota hierarchy; CPU
  limits and throttling for services and containers.
- [Process Management](process.md) — the userspace view: `nice`, priorities, job control.
- [Synchronization](synchronization.md) — wakeups, `need_resched`, and why `PREEMPT_RT` turns
  spinlocks into sleeping locks.
- [Kernel Timers](kernel_timers.md) — the timer tick drives slice accounting and preemption checks.

## Pitfalls

- **Runaway `SCHED_FIFO`/`SCHED_RR` task** starves everything below it. RT tasks have no fairness;
  bound them with `sched_rt_runtime_us` and never busy-loop at RT priority.
- **Confusing `nice` with latency.** Under EEVDF `nice` sets CPU *share*, not how *soon* you run.
  For responsiveness, request a shorter slice rather than a better nice.
- **cgroup CPU throttling looks like a hang.** A container hitting `cpu.max` quota is stalled
  until the next period, even with idle CPUs — symptoms are periodic latency spikes; check
  `cpu.stat` `nr_throttled`/`throttled_usec`.
- **Fighting the load balancer.** Manually pinning some tasks while leaving others free can cause
  ping-ponging; either pin a coherent set (cpusets/`isolcpus`) or leave it to the scheduler.
- **NUMA-blind placement.** Migrating a memory-heavy task across NUMA nodes tanks performance via
  remote memory access; pair affinity with NUMA-aware allocation.
- **Priority inversion.** A low-priority task holding a lock can block a high-priority one;
  `SCHED_DEADLINE`/RT workloads need priority-inheritance mutexes (`PI-futex`/`rt_mutex`).
