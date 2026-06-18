# Kernel Synchronization

## Overview

This page covers the kernel's core mutual-exclusion primitives — **spinlocks, mutexes, and
semaphores** (plus their reader/writer variants) — and, crucially, *how to choose between
them*. It expands the brief snippets in [Kernel Development Patterns](kernel_patterns.md)
(which also holds atomics, RCU, and completions) and is applied throughout
[Driver Development](driver_development.md). The "can this code sleep?" question that drives
the choice ties directly to execution context in [Process Internals (Kernel)](process_internals.md)
and to the atomic-context delays in [Kernel Timers](kernel_timers.md).

The kernel is deeply concurrent: multiple CPUs, preemption, interrupts, and softirqs can all
touch the same data. A lock serialises access to a *critical section*. The single most
important decision is whether the lock may **sleep** when it can't be acquired — because that
determines which contexts you can use it in.

```
   Need mutual exclusion. Can this context sleep?
     ├─ NO  (IRQ handler, softirq/tasklet, holding a spinlock)
     │        →  spinlock_t   (busy-waits; keep the section tiny)
     │           + also taken in an IRQ?  → spin_lock_irqsave()
     │           + also taken in a softirq? → spin_lock_bh()
     │
     └─ YES (process context, e.g. syscall / kthread / work item)
              ├─ simple mutual exclusion        →  struct mutex
              ├─ count N resources / signalling  →  struct semaphore
              └─ read-mostly data                →  rwlock_t / rw_semaphore
```

## The Sleeping-vs-Atomic Rule

Kernel code runs in either **process context** (a syscall, a kthread, a workqueue item —
backed by a `task_struct` that the scheduler can sleep and resume) or **atomic context**
(interrupt handlers, softirqs/tasklets, or *any* code holding a spinlock or with preemption
disabled). The rule:

> **You must never sleep in atomic context.** A sleeping lock (`mutex`, `semaphore`,
> `rw_semaphore`) may block the caller, so it can only be taken in process context. A
> spinlock busy-waits without sleeping, so it is the only mutual-exclusion lock usable in
> atomic context.

Violating this ("scheduling while atomic"/deadlock) is one of the most common kernel bugs.
On contention each primitive behaves differently:

- **spinlock** → the CPU *spins* (busy-waits) until the lock is free. Cheap if held briefly,
  wasteful (and dangerous) if held long.
- **mutex / semaphore** → the task is put to sleep and woken when the lock is released. More
  overhead per acquisition, but no CPU is burned while waiting.

## Spinlocks

A spinlock protects short critical sections that may be entered from atomic context. The
waiter spins, so the section must be **short and must never sleep** while the lock is held
(no `kmalloc(GFP_KERNEL)`, no `mutex_lock`, no `copy_to_user`, no `msleep`).

```c
#include <linux/spinlock.h>

static DEFINE_SPINLOCK(my_lock);     /* or spinlock_t l; spin_lock_init(&l); */

/* Plain: data shared only between process-context paths (preemption disabled while held) */
spin_lock(&my_lock);
/* critical section — short, no sleeping */
spin_unlock(&my_lock);

/* Shared with an IRQ handler: disable IRQs on THIS cpu and save prior state */
unsigned long flags;
spin_lock_irqsave(&my_lock, flags);
/* critical section */
spin_unlock_irqrestore(&my_lock, flags);

/* Shared with a softirq/tasklet/timer bottom half */
spin_lock_bh(&my_lock);
/* critical section */
spin_unlock_bh(&my_lock);

/* Non-blocking attempt */
if (spin_trylock(&my_lock)) {
        /* got it */
        spin_unlock(&my_lock);
}
```

Choosing the variant by *who else* touches the data:

- Only process-context paths → `spin_lock()`.
- Also an **IRQ handler** → `spin_lock_irqsave()`. If the IRQ fired on the same CPU while you
  held a plain spinlock and then tried to take it, you'd self-deadlock; disabling IRQs
  prevents that. Use `irqsave`/`irqrestore` (not the bare `irq` variant) unless you *know*
  IRQs were already enabled.
- Also a **softirq/tasklet/timer callback** → `spin_lock_bh()`.

Under **`CONFIG_PREEMPT_RT`** most spinlocks become sleeping mutexes (only `raw_spinlock_t`
keeps the classic spin-and-disable-preemption behaviour) — so even "spinlock" code should
avoid assuming truly atomic execution unless it uses `raw_spinlock_t`.

## Mutex

`struct mutex` is the preferred primitive for plain mutual exclusion in process context. It
sleeps on contention, so it must **not** be used in atomic context (IRQ/softirq/while holding
a spinlock).

```c
#include <linux/mutex.h>

static DEFINE_MUTEX(my_mutex);       /* or struct mutex m; mutex_init(&m); */

mutex_lock(&my_mutex);
/* critical section — sleeping (kmalloc GFP_KERNEL, I/O, etc.) is allowed here */
mutex_unlock(&my_mutex);

/* Interruptible: returns -EINTR if a signal arrives while blocked (use in syscalls) */
if (mutex_lock_interruptible(&my_mutex))
        return -ERESTARTSYS;
mutex_unlock(&my_mutex);

/* Non-blocking attempt */
if (mutex_trylock(&my_mutex)) {
        mutex_unlock(&my_mutex);
}
```

Key semantics:

- **Owner-bound.** The task that locks a mutex must be the one that unlocks it. You cannot
  use a mutex as a cross-task signalling mechanism, and you cannot acquire it recursively.
- Optimised for the common case with adaptive spinning (it briefly spins if the owner is
  running on another CPU before sleeping), so it's nearly as fast as a spinlock when
  uncontended while still sleeping under real contention.
- Prefer `mutex_lock_interruptible()` on user-triggered paths so a blocked task can still be
  killed/interrupted.

## Semaphore

`struct semaphore` is a **counting** semaphore: it allows up to *N* holders at once
(`count` set at init). It also sleeps on contention → process context only. For plain mutual
exclusion (`N == 1`) a `mutex` is almost always the better choice today; reach for a
semaphore when you genuinely need to count resources or to signal between tasks.

```c
#include <linux/semaphore.h>

static struct semaphore sem;
sema_init(&sem, 3);          /* allow up to 3 concurrent holders */

down(&sem);                  /* acquire (sleep until count > 0), then count-- */
/* up to 3 tasks can be in here at once */
up(&sem);                    /* release, count++ — may wake a waiter */

/* Interruptible / non-blocking variants */
if (down_interruptible(&sem))
        return -ERESTARTSYS;
/* ... */
up(&sem);

if (down_trylock(&sem) == 0) {   /* 0 = acquired */
        up(&sem);
}
```

How it differs from a mutex:

- **Not owner-bound.** A semaphore can be `up()`'d by a *different* task than the one that
  `down()`'d it. That makes it suitable for producer/consumer signalling (producer `up()`s,
  consumer `down()`s) — but also means no ownership debugging/priority-inheritance support.
- **Counting.** `N > 1` bounds concurrent access to a pool of resources.
- The old "binary semaphore as a lock" pattern has been superseded by `mutex`, which is
  faster, owner-checked, and lockdep-friendly. For one-shot "wait until done" signalling,
  prefer `struct completion` (see [Kernel Development Patterns](kernel_patterns.md)).

## Reader/Writer Variants

When access is **read-mostly**, reader/writer locks let many readers proceed in parallel and
only serialise writers:

```c
/* Spinning version (atomic-safe, busy-waits) */
#include <linux/rwlock.h>
static DEFINE_RWLOCK(my_rwlock);
read_lock(&my_rwlock);   /* ... */   read_unlock(&my_rwlock);
write_lock(&my_rwlock);  /* ... */   write_unlock(&my_rwlock);

/* Sleeping version (process context only) */
#include <linux/rwsem.h>
static DECLARE_RWSEM(my_rwsem);
down_read(&my_rwsem);    /* ... */   up_read(&my_rwsem);
down_write(&my_rwsem);   /* ... */   up_write(&my_rwsem);
```

Use these only when there's a clear read/write imbalance and the critical sections are
non-trivial; otherwise the bookkeeping overhead outweighs a plain spinlock/mutex. Watch for
**writer starvation** under a heavy stream of readers. For very read-heavy, latency-critical
data, **RCU** (in [Kernel Development Patterns](kernel_patterns.md)) often beats rwlocks
because readers take no lock at all.

## Choosing a Primitive

| Primitive          | Sleeps on contention? | Usable context        | Owner-bound? | Counting? | Use when |
|--------------------|-----------------------|-----------------------|--------------|-----------|----------|
| `spinlock_t`       | No (busy-waits)       | any (incl. atomic)    | n/a          | No        | short section reachable from IRQ/softirq |
| `mutex`            | Yes                   | process only          | **Yes**      | No        | default mutual exclusion that may sleep |
| `semaphore`        | Yes                   | process only          | No           | **Yes**   | count N resources / cross-task signalling |
| `rwlock_t`         | No                    | any (incl. atomic)    | n/a          | No        | read-mostly, short, atomic context |
| `rw_semaphore`     | Yes                   | process only          | No           | No        | read-mostly, may sleep |

When a lock isn't the right tool at all, see [Kernel Development Patterns](kernel_patterns.md):
**atomic_t / bitops** for simple counters and flags, **RCU** for read-mostly lookups with
near-zero reader cost, and **completion** for "wait until event X happens."

## Where this connects

- [Kernel Development Patterns](kernel_patterns.md) — atomics, RCU, completions, and wait
  queues round out the synchronization toolbox; this page deepens its locking snippets.
- [Driver Development](driver_development.md) — drivers protect their private state with these
  locks; IRQ handlers force the `irqsave`/`bh` variants.
- [Process Internals (Kernel)](process_internals.md) — execution context (process vs atomic)
  and the scheduler are what make the sleeping-vs-spinning distinction matter.
- [Kernel Timers](kernel_timers.md) — timer/hrtimer callbacks run in atomic context, so they
  may only take spinlocks, never mutexes/semaphores.

## Pitfalls

- **Sleeping in atomic context.** Taking a `mutex`/`semaphore`, or doing sleepable work
  (`GFP_KERNEL` alloc, I/O, `msleep`), while holding a spinlock or inside an IRQ/softirq, can
  deadlock or panic. Use a spinlock there, or defer to a workqueue.
- **Deadlock from lock ordering (AB-BA).** Two paths that take locks A and B in opposite
  orders can deadlock. Establish and document a global lock order; enable `CONFIG_LOCKDEP` to
  catch violations and recursion automatically.
- **IRQ self-deadlock.** A plain `spin_lock()` on data also touched by an interrupt handler
  deadlocks if the IRQ fires on the same CPU while held — use `spin_lock_irqsave()`.
- **Holding a spinlock too long.** Spinlocks disable preemption (and burn the waiting CPU);
  long sections hurt latency. Keep them tiny; move heavy work outside the lock.
- **Forgetting to unlock on error paths.** Every early `return`/`goto` in a locked section
  must release the lock; use a single `goto unlock` cleanup pattern.
- **Unlocking a mutex from the wrong task.** Mutexes are owner-bound and non-recursive — if
  you need cross-task release or recursion, you've picked the wrong primitive (semaphore /
  completion).
- **`PREEMPT_RT` surprises.** On RT kernels most spinlocks sleep; code that assumed truly
  atomic execution must use `raw_spinlock_t` (and keep those sections genuinely tiny).
