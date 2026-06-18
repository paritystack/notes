# Kernel Timers and Time Management

## Overview

This page covers how to keep track of time and schedule deferred work from kernel code —
timekeeping units, busy-wait vs sleeping delays, the legacy timer wheel (`timer_list`),
high-resolution timers (`hrtimer`), and timed deferral via the workqueue. It is a companion
to [Kernel Development Patterns](kernel_patterns.md) (which introduces timers alongside
workqueues, wait queues, and kthreads) and is used heavily in [Driver Development](driver_development.md)
for timeouts, polling, and watchdogs. For *who* runs when, see the scheduler in
[Process Internals (Kernel)](process_internals.md); for the runtime-tunable side, see
[sysctl](sysctl.md).

The first decision is always the same: **do I need to wait, or to be called back later?**
And: **am I in a context where I'm allowed to sleep?** Those two questions pick the API.

```
   Need to pause execution NOW?
     ├─ very short, atomic context  →  busy-wait: udelay()/ndelay()/mdelay()
     └─ can sleep (process context)  →  sleep:    usleep_range()/msleep()

   Need a callback LATER?
     ├─ ms-ish, jiffy granularity ok →  timer_list (timer wheel)
     ├─ precise / sub-ms / periodic  →  hrtimer (high-resolution)
     └─ callback may sleep           →  delayed_work (workqueue)
```

## Timekeeping Units

```c
#include <linux/jiffies.h>
#include <linux/ktime.h>

extern unsigned long volatile jiffies;   /* ticks since boot (per CONFIG_HZ) */
```

- **`jiffies` / `HZ`.** `jiffies` is a counter incremented `HZ` times per second by the
  timer interrupt (`CONFIG_HZ`, commonly 100/250/1000). It is the unit of the legacy timer
  wheel. Convert with `msecs_to_jiffies()`, `jiffies_to_msecs()`, `usecs_to_jiffies()`.
- **Never compare jiffies with `<`/`>`** — it wraps. Use the macros that handle wraparound:
  `time_after(a, b)`, `time_before()`, `time_after_eq()`, `time_before_eq()`.
- **`ktime_t`.** A scalar nanosecond time used by hrtimers. Read the current time with
  `ktime_get()` (monotonic), `ktime_get_real()` (wall clock / realtime),
  `ktime_get_boottime()` (includes suspend). Build offsets with `ms_to_ktime()`,
  `ns_to_ktime()`, `ktime_set(sec, nsec)`.
- **Clock bases.** `CLOCK_MONOTONIC` never goes backwards and is the right choice for
  timeouts/intervals; `CLOCK_REALTIME` can jump (NTP, settimeofday) and is for wall-clock
  deadlines only.

```c
/* Safe deadline check */
unsigned long deadline = jiffies + msecs_to_jiffies(500);
if (time_after(jiffies, deadline))
        /* 500 ms elapsed (wraparound-safe) */;
```

Under the hood, hardware **clocksources** (TSC, HPET, arch timers) provide monotonic counts
for timekeeping, while **clockevent** devices generate the interrupts that drive both the
periodic tick and the next hrtimer expiry. On a `NO_HZ` (tickless) kernel the periodic tick
is suppressed when a CPU is idle or running a single task, so timers are programmed
on-demand rather than checked every jiffy.

## Delays: Busy-Wait vs Sleeping

Picking the wrong one is a classic bug. **Busy-wait** spins the CPU and is the *only* option
in atomic context (interrupt handlers, holding a spinlock). **Sleeping** delays yield the
CPU and may only be used in process context.

```c
#include <linux/delay.h>

/* Busy-wait (atomic-safe, burns the CPU) — keep SHORT (≤ ~ms) */
ndelay(100);     /* nanoseconds */
udelay(50);      /* microseconds */
mdelay(5);       /* milliseconds — frowned upon; that's a long spin */

/* Sleeping (process context only — must be allowed to sleep) */
usleep_range(50, 100);   /* µs: give a range so the kernel can batch */
msleep(20);              /* ms: ≥ jiffy granularity, may sleep longer */
msleep_interruptible(20);
```

Rules of thumb:

- For sub-millisecond sleeps prefer **`usleep_range(min, max)`** over `msleep(1)`; the range
  lets the kernel coalesce wakeups and avoids `msleep`'s rounding up to the next jiffy.
- `mdelay()` and large `udelay()` are red flags — you're holding a CPU hostage. Restructure
  to a timer/work callback if you can sleep.
- In atomic context you **must not** call any sleeping delay (`msleep`, `usleep_range`,
  `schedule_timeout`) — it can deadlock or panic.

## Legacy Timers: `timer_list` (the Timer Wheel)

`timer_list` schedules a callback at a future **jiffies** value. Granularity is one jiffy and
expiry may be slightly late (it's batched in the timer wheel), so it's right for
coarse, "roughly N ms from now" needs — heartbeats, retries, soft timeouts.

```c
#include <linux/timer.h>

static struct timer_list my_timer;

static void my_timer_cb(struct timer_list *t)
{
        struct my_dev *dev = from_timer(dev, t, timer); /* container_of helper */

        /* runs in SOFTIRQ context: atomic, must NOT sleep */

        mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000)); /* re-arm if periodic */
}

/* setup: callback + flags (e.g. TIMER_DEFERRABLE, TIMER_PINNED) */
timer_setup(&my_timer, my_timer_cb, 0);

/* arm / re-arm to an absolute jiffies value */
mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000));

/* cancel — pick the right one */
del_timer(&my_timer);        /* may return while the callback runs elsewhere */
timer_delete_sync(&my_timer); /* waits for any running callback (older: del_timer_sync) */
```

Important properties:

- The callback runs in **softirq (timer) context** — atomic. No sleeping, no `copy_to_user`,
  no mutex; use spinlocks and the workqueue if you need to do heavyweight or sleeping work.
- Always use **`timer_delete_sync()`** (formerly `del_timer_sync`) before freeing the
  structure or unloading the module, or the callback can fire into freed memory. Don't call
  it from within the timer's own callback.
- `TIMER_DEFERRABLE` lets an idle CPU skip the timer to save power (good for non-urgent
  housekeeping); `TIMER_PINNED` keeps it on the arming CPU.

## High-Resolution Timers: `hrtimer`

When jiffy granularity isn't enough — sub-millisecond, precise periodic sampling, audio,
PWM-ish timing — use `hrtimer`, which works in `ktime_t` nanoseconds and is backed directly
by clockevent hardware.

```c
#include <linux/hrtimer.h>

static struct hrtimer my_hrtimer;

static enum hrtimer_restart my_hrtimer_cb(struct hrtimer *timer)
{
        /* also atomic (hardirq/softirq) — do not sleep */

        /* periodic: advance by a fixed interval and keep going */
        hrtimer_forward_now(timer, ms_to_ktime(10));
        return HRTIMER_RESTART;     /* or HRTIMER_NORESTART for one-shot */
}

/* Modern API (v6.16+): one call sets clock, mode, and callback */
hrtimer_setup(&my_hrtimer, my_hrtimer_cb, CLOCK_MONOTONIC, HRTIMER_MODE_REL);

/* Older trees use hrtimer_init() + assign .function manually:
 *   hrtimer_init(&my_hrtimer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
 *   my_hrtimer.function = my_hrtimer_cb;
 */

hrtimer_start(&my_hrtimer, ms_to_ktime(10), HRTIMER_MODE_REL);

/* cancel (waits for the callback if it's running) */
hrtimer_cancel(&my_hrtimer);
```

Notes:

- Modes: `HRTIMER_MODE_REL` (relative to now) vs `HRTIMER_MODE_ABS` (absolute time);
  add `_SOFT`/`_HARD` to force softirq vs hardirq delivery.
- For periodic timers use **`hrtimer_forward_now()`** inside the callback and return
  `HRTIMER_RESTART` — this avoids drift that you'd get by re-reading "now" and adding.
- hrtimer callbacks are still atomic. To do work that may sleep, queue it (see below).

## Timed Deferral That May Sleep: `delayed_work`

Timer and hrtimer callbacks can't sleep. When the deferred work needs to sleep (I/O,
mutexes, allocations with `GFP_KERNEL`), schedule it on a workqueue with a delay; the
worker thread runs in **process context**.

```c
#include <linux/workqueue.h>

static void my_work_fn(struct work_struct *work)
{
        struct delayed_work *dw = to_delayed_work(work);
        /* process context: sleeping is allowed here */
}

static DECLARE_DELAYED_WORK(my_dwork, my_work_fn);

schedule_delayed_work(&my_dwork, msecs_to_jiffies(200)); /* run ~200 ms later */
mod_delayed_work(system_wq, &my_dwork, msecs_to_jiffies(500)); /* re-arm */
cancel_delayed_work_sync(&my_dwork);                     /* before freeing/unload */
```

This is essentially "a `timer_list` whose callback runs in a kthread," and is the idiomatic
way to get a *delayed, sleepable* callback. See workqueues and kthreads in
[Kernel Development Patterns](kernel_patterns.md).

## Choosing an API

| Need                                   | Use                                   | Context of callback |
|----------------------------------------|----------------------------------------|---------------------|
| Spin a few µs in atomic context        | `udelay()` / `ndelay()`               | caller's            |
| Sleep ms in process context            | `msleep()` / `usleep_range()`         | caller's            |
| Coarse (~ms) deferred callback         | `timer_list` + `mod_timer()`          | softirq (atomic)    |
| Precise / sub-ms / periodic callback   | `hrtimer`                             | hard/softirq (atomic)|
| Deferred callback that may sleep       | `delayed_work`                        | process (sleepable) |
| Sleep until a condition, with timeout  | `wait_event_timeout()`                | caller's            |

## Where this connects

- [Kernel Development Patterns](kernel_patterns.md) — timers sit alongside workqueues, wait
  queues, and kthreads; `wait_event_timeout()` is the timed-blocking counterpart.
- [Driver Development](driver_development.md) — timeouts, debounce, polling, and watchdogs in
  device drivers lean on these APIs.
- [Process Internals (Kernel)](process_internals.md) — the scheduler decides when a woken
  task actually runs; softirq timer callbacks preempt it.
- [sysctl](sysctl.md) — runtime knobs for timekeeping/scheduling behaviour.
- [Kernel Architecture](kernel.md) — where time management and interrupts sit in the kernel.

## Pitfalls

- **Sleeping in atomic context.** Calling `msleep`/`usleep_range`/`schedule_timeout` (or
  doing sleepable work) inside a `timer_list`/`hrtimer` callback, an IRQ handler, or under a
  spinlock can deadlock or panic ("scheduling while atomic"). Defer to `delayed_work`.
- **Comparing jiffies directly.** `if (jiffies > deadline)` breaks on wraparound. Always use
  `time_after()`/`time_before()`.
- **Forgetting `*_sync` cancel before free/unload.** `timer_delete()`/`cancel_delayed_work()`
  may return while the callback is still running; use `timer_delete_sync()` /
  `cancel_delayed_work_sync()` / `hrtimer_cancel()` before freeing the structure — but never
  the sync variant from inside that same callback.
- **`mdelay()` / long `udelay()`.** Busy-waiting milliseconds wastes a CPU and hurts latency;
  use a sleeping delay or a timer if you can sleep.
- **`msleep(1)` is imprecise.** It rounds up to the next jiffy and can sleep much longer than
  1 ms. Use `usleep_range()` for short, fairly precise sleeps.
- **Periodic drift.** Re-arming with `now + interval` accumulates error; use
  `hrtimer_forward_now()` (hrtimer) or arm against the previous expiry, not "now".
- **Wrong clock base.** Timeouts on `CLOCK_REALTIME` can fire early/late when the wall clock
  is adjusted; use `CLOCK_MONOTONIC` for intervals and timeouts.
- **Stale APIs.** Newer trees rename/replace some calls: `del_timer_sync()` →
  `timer_delete_sync()`, and `hrtimer_init()` + manual `.function` → `hrtimer_setup()`.
  Match what your target kernel version provides.
