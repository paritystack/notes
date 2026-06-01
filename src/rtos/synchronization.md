# Inter-Task Communication & Synchronization

## Overview

Tasks in an RTOS are mostly independent threads of execution, but they constantly need to
**pass data** and **coordinate timing** — a sensor task hands samples to a processing
task, an ISR signals that a DMA transfer finished, two tasks take turns at a shared bus.
This page surveys the primitives every RTOS provides — queues/mailboxes, semaphores,
mutexes, event flags — at a concept level that maps onto [FreeRTOS](freertos.md),
[ThreadX](threadx.md), [Zephyr](zephyr.md), and [PREEMPT_RT Linux](rt_linux.md). Picking
the right primitive is a real-time decision: the wrong one reintroduces
[priority inversion](priority_inversion.md), races, or deadlock, and changes a task's
blocking behavior in [response-time analysis](scheduling.md).

These are the concepts the [RTOS overview](README.md) lists under "Inter-Task
Communication"; for the non-real-time treatment of the same ideas see
[Concurrency](../programming/concurrency.md).

## The two jobs: data transfer vs synchronization

```
Communication   move DATA between tasks (or ISR→task)        → queues, mailboxes, streams
Synchronization coordinate TIMING / access                    → semaphores, mutexes, events
```

Most real designs combine them: a queue both *transfers* a message and *wakes* the
receiver, which is why queues are the workhorse primitive.

## Queues and mailboxes

A queue is a thread-safe FIFO. A sender copies a message in; a receiver blocks until one
is available (or a timeout expires), then copies it out. Senders and receivers are
decoupled in time.

```
   producer task                 queue (depth N)               consumer task
   ─────────────                ┌──┬──┬──┬──┐               ─────────────
   send(msg) ──── copy in ────► │  │  │  │  │ ──── copy out ────► receive(&msg)
   blocks if full               └──┴──┴──┴──┘  blocks if empty
```

- **By copy, not reference** on MCUs — passing pointers means the data must outlive the
  send (don't queue a pointer to a stack variable).
- **ISR-safe variants** (`xQueueSendFromISR`) let an interrupt hand work to a task; this
  is the standard "deferred interrupt processing" pattern (see
  [interrupts](../embedded/interrupts.md)).
- **Mailbox** = a one-deep queue (latest message); **stream/message buffer** = byte- or
  variable-length variants for [UART](../embedded/uart.md)-style data.

## Semaphores: signalling and counting

A semaphore is a counter with `take` (wait, decrement) and `give` (signal, increment).

```
Binary semaphore   count ∈ {0,1}    — event flag / signalling. ISR gives, task takes.
Counting semaphore count 0..N        — manage N identical resources (e.g. 3 DMA channels)
                                       or count events that may arrive faster than served.
```

The dominant use is **ISR-to-task signalling**: the ISR does the minimum, `give`s a
semaphore, and a waiting task wakes to do the heavy work at task priority. A semaphore is
*not* owned by anyone — any task can give it.

## Mutexes: mutual exclusion (with inheritance)

A mutex protects a shared resource (a peripheral, a data structure). Crucially it differs
from a binary semaphore in two ways:

```
                  binary semaphore        mutex
ownership         none (anyone gives)      owned by the locker; only owner unlocks
priority protocol none                     priority inheritance / ceiling  ← critical
use for           signalling               mutual exclusion
```

**Use a mutex, not a semaphore, for locking** — only the mutex carries the
[priority-inversion](priority_inversion.md) protections. Recursive mutexes allow the
owner to re-lock; keep critical sections short and never block (delay, queue-receive)
while holding one.

## Event flags / event groups

A set of bits a task can wait on, with AND/OR conditions — useful when a task must wait
for *several* things or *any one* of several things.

```
   bits:  [ TX_DONE | RX_DONE | ERROR | TICK ]
   task waits for (RX_DONE AND TICK)  → unblocks only when both are set
   ISR sets RX_DONE; timer sets TICK  → one wait, multiple sources
```

More expressive than a single semaphore, and one wait can replace several nested ones.

## Choosing a primitive

```
Need                                              Use
──────────────────────────────────────────────   ───────────────────────────
Pass data items between tasks                      Queue
Latest-value handoff                               Mailbox (1-deep queue)
ISR tells a task "something happened"              Binary semaphore (give-from-ISR)
Count N identical resources / pooled buffers       Counting semaphore
Exclusive access to a shared resource              Mutex  (NOT a semaphore)
Wait on several conditions / any-of                Event group
Stream of bytes (UART, audio)                      Stream / message buffer
```

## Where this connects

- [Priority inversion](priority_inversion.md) — why locking must use a mutex (with
  inheritance/ceiling), never a plain semaphore
- [Scheduling](scheduling.md) — blocking on these primitives is what moves a task between
  ready and blocked; blocking time enters response-time analysis
- [Context switching](context_switching.md) — a `give`/send that unblocks a higher
  priority task triggers an immediate switch (e.g. `portYIELD_FROM_ISR`)
- [Interrupts](../embedded/interrupts.md) — `...FromISR` APIs implement deferred interrupt
  processing; only ISR-safe calls are legal in an ISR
- [Concurrency](../programming/concurrency.md) — the same primitives (mutexes, condition
  variables, channels) in general multithreading
- [FreeRTOS](freertos.md) / [ThreadX](threadx.md) / [Zephyr](zephyr.md) — concrete APIs
  for every primitive here

## Pitfalls

```
- Using a binary semaphore as a lock: no priority inheritance → priority inversion.
  Use the mutex type for mutual exclusion.
- Calling a non-ISR-safe API from an ISR (e.g. xQueueSend instead of ...FromISR):
  corrupts the kernel. Always use the FromISR variants and honour the woken-task flag.
- Queueing a pointer to a stack/short-lived buffer: it's freed before the receiver reads.
  Queue by value, or guarantee lifetime.
- Blocking (delay, queue-receive) while holding a mutex: extends inversion and risks
  deadlock. Keep critical sections short and non-blocking.
- Unbounded timeouts (wait forever) hide lockups; prefer a timeout + error handling so a
  stuck producer is detectable (often tied into the watchdog).
- Priority of the receiver lower than the sender's expectation → data piles up; size
  queues for the worst-case burst, not the average.
- Forgetting that giving a semaphore from an ISR may need an immediate yield to honour
  real-time response; drop the yield and the high-priority task waits a full tick.
```
