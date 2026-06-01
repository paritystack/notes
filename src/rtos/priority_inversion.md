# Priority Inversion

## Overview

Priority inversion is the classic real-time bug where a **high-priority task is blocked
by a low-priority one** — inverting the priority order the [scheduler](scheduling.md) is
supposed to enforce. It happens whenever tasks share a resource guarded by a
[mutex](synchronization.md): the high-priority task needs a lock the low-priority task
holds, and while it waits, *medium*-priority tasks can run and starve the low-priority
holder, stretching the block without bound. This page explains the mechanism, the famous
[Mars Pathfinder](#case-study-mars-pathfinder) failure, and the two standard fixes —
**priority inheritance** and the **priority ceiling protocol** — that
[FreeRTOS](freertos.md), [ThreadX](threadx.md), and [Zephyr](zephyr.md) implement.

It's the missing "blocking term" from [response-time analysis](scheduling.md): without a
protocol, that term is unbounded, so the whole schedulability proof collapses.

## The mechanism

Three tasks — H (high), M (medium), L (low) — and one mutex shared by H and L:

```
prio
  H │            ┌╌╌╌╌╌╌ wants mutex (held by L) ╌╌╌╌╌╌┐  ┌─ runs ─
    │            ╎  BLOCKED                            ╎  │
  M │            ╎        ┌──────── M runs ────────┐   ╎  │  M preempts L,
    │            ╎        │ (doesn't touch mutex)  │   ╎  │  extending H's wait
  L │ ──lock──┐  ╎ ┌──────┘                        └───┘unlock
    │ holds   └──┴─┘ L can't make progress while M runs
    └──────────────────────────────────────────────────────────► time
        L takes      H ready        M ready      M done   L unlocks → H runs
        the lock     but blocked
```

H is delayed not just by L's short critical section (**bounded** inversion, unavoidable)
but by the *entire* run time of every medium task that preempts L (**unbounded**
inversion). H — the most important task — effectively runs at L's priority for an
arbitrarily long time.

## Case study: Mars Pathfinder

In 1997 the Pathfinder lander began resetting itself on Mars. A high-priority bus-management
task shared a mutex (a VxWorks pipe) with a low-priority meteorological
task; medium-priority communications tasks would preempt the low task while it held the
lock. The high task missed its deadline, the [watchdog](../embedded/watchdog.md) fired,
and the system reset. The fix was uploaded live: **enable priority inheritance** on that
mutex. It's the canonical real-world lesson that priority inversion is a *correctness*
bug, not a performance one.

## Fix 1: Priority inheritance

While a low-priority task holds a mutex that a higher-priority task is waiting on, it
**temporarily inherits the waiter's priority**. Now medium tasks can't preempt it, so it
finishes its critical section and releases the lock quickly.

```
prio
  H │            ┌╌╌ blocked ╌╌┐  ┌─ runs ─
    │            ╎             ╎  │
  M │            ╎             ╎  │   M stays ready but CANNOT preempt L:
    │            ╎             ╎  │   L now runs at H's priority
  L │ ──lock──┐  ╎  ┌─ L @H ──┐╎  │
    │         └──┴──┘ priority └┴──┘unlock → H runs immediately
    └────────────────────────────────────────────────► time
```

H's blocking is now bounded by L's critical-section length only — independent of how many
medium tasks exist. This is the default mutex behavior in FreeRTOS (mutexes, *not*
binary semaphores), ThreadX, and Zephyr. The inheritance can be *transitive* (L blocks on
another lock held by an even lower task, which also inherits).

## Fix 2: Priority ceiling protocol

Each mutex is given a **ceiling** = the priority of the highest task that can ever lock
it. A task that acquires the mutex is immediately raised to that ceiling (Immediate
Ceiling Priority Protocol, ICPP). Two strong properties fall out:

```
- A task can block at most ONCE, and only at the start of its execution.
- Deadlock between these mutexes is impossible (a task holding one is at the
  ceiling, so no other task that could take it can run).
- Worst-case blocking = the single longest critical section of any lower task.
```

The trade-off: ceilings must be configured ahead of time (you must know the task set),
which suits statically-configured safety systems like [OSEK/AUTOSAR](osek_autosar.md)
(which mandates ICPP) and [ARINC 653](arinc653.md). Inheritance needs no configuration,
which suits dynamic systems.

```
Priority inheritance : reactive, raises priority only when contention occurs.
                       No config. Default on most MCU RTOSes.
Priority ceiling      : proactive, raises priority on every lock acquire.
                       Needs static config. Prevents deadlock; bounds blocking
                       to one critical section. Required by OSEK (PCP).
```

## Where this connects

- [Synchronization](synchronization.md) — only *mutexes* carry these protocols; binary
  semaphores used as locks do **not**, which reintroduces inversion
- [Scheduling](scheduling.md) — bounded blocking is the term response-time analysis adds
  to make hard-real-time guarantees provable
- [Context switching](context_switching.md) — raising/restoring a task's priority drives
  extra switches; the cost is real
- [OSEK/AUTOSAR](osek_autosar.md) / [ARINC 653](arinc653.md) — safety standards mandate
  the priority ceiling protocol
- [Concurrency](../programming/concurrency.md) — locks, deadlock, and lock ordering in
  the general (non-real-time) setting
- [Watchdog](../embedded/watchdog.md) — what catches the symptom (a missed deadline)
  when inversion goes unhandled, as on Pathfinder

## Pitfalls

```
- Using a binary semaphore as a lock: it has NO priority inheritance. Use the RTOS's
  mutex type for mutual exclusion; reserve semaphores for ISR→task signalling.
- Assuming inheritance is free: it adds context switches and slightly complicates the
  scheduler; budget for it.
- Holding a mutex across a blocking call (delay, queue receive): you extend the inherited
  high-priority window and can chain into deadlock. Keep critical sections short.
- Misconfigured ceilings (set too low) break the ceiling protocol's guarantees silently.
- Nested locks with inconsistent ordering: inheritance bounds inversion but does NOT
  prevent deadlock — only the ceiling protocol does. Order your locks.
- Forgetting to account for blocking at all in schedulability analysis — the Pathfinder
  trap. Always add the worst-case blocking term.
```
