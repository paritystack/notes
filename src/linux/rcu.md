# RCU (Read-Copy-Update)

## Overview

**RCU** is the kernel's synchronization mechanism for **read-mostly data**: it lets readers run
with essentially *zero* overhead — no locks, no atomic operations, no cache-line bouncing — while
writers update data without ever blocking readers. This page is the canonical expansion of the
RCU snippets referenced in [Synchronization](synchronization.md) (the locking decision table) and
[Kernel Development Patterns](kernel_patterns.md) (atomics/completions/RCU toolbox) — link back to
those for where RCU sits among the alternatives. The "readers can't sleep" rules here are the same
atomic-context rules from [Process Internals (Kernel)](process_internals.md), and RCU underpins
hot lookup paths in [Networking](networking.md), VFS dentry caches in
[Filesystems](filesystems.md), and many list traversals across the kernel.

The core idea: instead of mutually excluding readers and writers, **let them run concurrently**,
and have the writer defer destroying the old data until it's guaranteed no reader can still see
it. Readers observe either the old or the new version — never a torn one.

```
   Writer: update by replacement, then defer the free
   ─────────────────────────────────────────────────
   old ──┐                          old (unlinked, still
         ▼                           visible to in-flight readers)
   [A]──[B]──[C]        ──►   [A]──[B']──[C]
                                     ▲
   reader traversing here keeps using old B until it finishes;
   free(old B) waits for a GRACE PERIOD (all readers done)
```

## Why not just a rwlock?

A reader/writer lock still has every reader take and release the lock — atomic writes to a shared
cache line that bounce between CPUs and serialise readers against each other under contention.
RCU read-side primitives compile to almost nothing (often just a compiler barrier and disabling
preemption), so read scaling is effectively perfect. The cost is moved entirely onto the *writer*,
which must wait out a grace period. RCU therefore wins precisely when **reads vastly outnumber
writes** and readers are short.

## The read side

```c
rcu_read_lock();                       /* enter RCU read-side critical section */
p = rcu_dereference(gp);               /* safe publish-aware load of the pointer */
if (p)
    do_something(p->field);            /* p stays valid until rcu_read_unlock() */
rcu_read_unlock();                     /* exit; must not sleep in between (classic RCU) */
```

- `rcu_read_lock()`/`rcu_read_unlock()` mark the critical section. In classic RCU they just
  disable preemption — extremely cheap — and you **must not block** inside.
- `rcu_dereference()` is the read-side counterpart of publish: it loads the pointer with the
  dependency-ordering barrier needed so you never see the new pointer but stale pointee data.

## The write side

```c
struct foo *new = kmalloc(sizeof(*new), GFP_KERNEL);
*new = *old;                           /* copy */
new->field = newval;                   /* modify the copy */
rcu_assign_pointer(gp, new);           /* publish: barrier + store */

synchronize_rcu();                     /* wait for all pre-existing readers to finish */
kfree(old);                            /* now safe to free */
```

This is the **"copy-update"** in the name: copy the object, mutate the copy, atomically swing the
pointer with `rcu_assign_pointer()` (which has the release barrier so a reader that sees the new
pointer also sees the fully-initialised object), then reclaim the old version once it's
unreachable.

## Grace periods: the heart of RCU

A **grace period** is an interval after which every read-side critical section that existed at its
start has completed. Reclaiming old data is safe only after a grace period elapses, because only
then is no reader holding a reference to it.

```
        grace period
   ├─────────────────────────┤
 reader1 ──────┘ (must finish)
 reader2 ──────────┘ (must finish)
              reader3 ├────────── (started after; sees NEW data — doesn't matter)
   removal ▲                      ▲ safe to free here
```

Two ways to wait:

- **`synchronize_rcu()`** — *blocks* the writer until a grace period passes. Simple; writer must
  be in process context (it may sleep).
- **`call_rcu(&p->rcu_head, callback)`** — *asynchronous*: registers a callback run after the next
  grace period, so the writer doesn't block. Use in atomic context or hot writer paths.
  `kfree_rcu(p, rcu_head)` is the common shorthand for "free after a grace period."

Classic RCU detects grace periods cheaply by observing **quiescent states** — points where a CPU
*cannot* be inside an RCU reader (a context switch, idle, or return to userspace). Once every CPU
has passed through one, all prior readers must be done.

## RCU-protected lists

The most common use is the `list`/`hlist` RCU variants, which pair publish/subscribe barriers
with the standard linked lists:

```c
/* writer */            list_add_rcu(&new->node, head);
                        list_del_rcu(&old->node);   /* then call_rcu/synchronize to free */
/* reader */            list_for_each_entry_rcu(e, head, node) { ... }
```

## Flavors

Modern kernels unified the classic flavors, but the concepts remain:

- **RCU (classic / "sched")** — readers disable preemption; can't sleep. The default.
- **SRCU (Sleepable RCU)** — readers *may* sleep inside the critical section, at the cost of a
  per-domain `srcu_struct` and slightly heavier read primitives (`srcu_read_lock()` returns an
  index). Use when a reader must block (e.g. take a mutex, do I/O).
- **Tasks RCU** — specialised grace periods used by tracing/`ftrace` trampolines.

## Where this connects

- [Synchronization](synchronization.md) — RCU is the read-mostly entry in the locking decision
  table; this page is its deep dive. Compare with `rwlock_t`/`rw_semaphore` for when writes are
  frequent or readers must sleep without SRCU.
- [Kernel Development Patterns](kernel_patterns.md) — RCU sits alongside atomics, completions, and
  wait queues in the concurrency toolbox.
- [Process Internals (Kernel)](process_internals.md) — quiescent states are context switches /
  idle / userspace transitions; the same atomic-context rule forbids sleeping in classic readers.
- [Filesystems](filesystems.md) / [Networking](networking.md) — dcache lookups and routing/socket
  hot paths use RCU to scale reads across CPUs.

## Pitfalls

- **Sleeping in a classic RCU reader.** Blocking between `rcu_read_lock()`/`unlock()` breaks grace-
  period detection (and triggers "illegal context switch" with `CONFIG_PROVE_RCU`). Use **SRCU** if
  the reader must sleep.
- **Dereferencing without `rcu_dereference()`.** A plain load may let the CPU/compiler reorder and
  read the pointee before the pointer, returning garbage on weakly-ordered architectures.
- **Publishing without `rcu_assign_pointer()`.** A plain store can make the new pointer visible
  before the object's initialising writes, so a reader sees an uninitialised object.
- **Freeing before the grace period.** `kfree(old)` immediately after unlinking — without
  `synchronize_rcu()`/`call_rcu()`/`kfree_rcu()` — is a use-after-free for in-flight readers.
- **`synchronize_rcu()` in atomic context or on a hot path.** It may sleep (illegal in atomic) and
  can take milliseconds; prefer `call_rcu`/`kfree_rcu` to avoid blocking the writer.
- **`call_rcu` callback floods.** A storm of `call_rcu()` without throttling can pile up pending
  callbacks and OOM; batch updates or use `synchronize_rcu()` to apply backpressure.
- **RCU stalls.** A CPU stuck in a long reader (or with preemption disabled) prevents grace periods
  from ending — the kernel logs "RCU CPU stall" warnings; the culprit is usually a too-long
  read-side section or a livelocked CPU.
