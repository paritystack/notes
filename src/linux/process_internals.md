# Linux Kernel Process Internals

## Overview

This page describes how the **kernel itself** represents and manages processes —
the data structures, creation/teardown paths, and the scheduler — as opposed to
the userspace-facing view in [Process Management](process.md) (PIDs, memory layout,
IPC, `/proc`, tooling). It complements [Kernel Architecture](kernel.md), draws on
[Memory Management](memory_management.md) for address spaces, and connects to
[Namespaces](namespace.md) and [cgroups](cgroups.md), which slice and bound what a
process can see and consume.

The core idea: to the Linux kernel there is no separate "process" and "thread" type.
Every schedulable entity is a `struct task_struct` (a *task*). What userspace calls a
multi-threaded process is just a group of tasks that share a thread group ID (`tgid`)
and resources like the address space and open files. The kernel-source touch points
referenced below are stable enough to navigate a recent tree (v6.x):
`include/linux/sched.h`, `kernel/fork.c`, `kernel/exit.c`, `kernel/exec.c`, and
`kernel/sched/`.

```
            userspace "process"  ==  thread group  ==  one tgid
            ┌───────────────────────────────────────────────┐
            │ task A (tgid=A, pid=A)  ← "main thread"        │
            │ task B (tgid=A, pid=B)  ← pthread              │  shared: mm, files,
            │ task C (tgid=A, pid=C)  ← pthread              │  signal handlers, ...
            └───────────────────────────────────────────────┘
   getpid() returns tgid;  gettid() returns the per-task pid.
```

## The Process Descriptor: `task_struct`

Every task is described by a `struct task_struct` defined in `include/linux/sched.h`.
It is large (kilobytes) because it aggregates everything the kernel needs to know
about a task. The fields cluster into a few groups:

```c
struct task_struct {
    /* state & scheduling */
    unsigned int        __state;        /* TASK_RUNNING, etc. */
    int                 prio, static_prio, normal_prio;
    const struct sched_class *sched_class;
    struct sched_entity se;             /* CFS/EEVDF bookkeeping */
    unsigned int        policy;         /* SCHED_NORMAL, SCHED_FIFO, ... */

    /* identity */
    pid_t               pid;            /* per-task id (== TID) */
    pid_t               tgid;           /* thread-group id (== userspace PID) */
    struct pid          *thread_pid;

    /* relationships */
    struct task_struct __rcu *real_parent;
    struct task_struct __rcu *parent;
    struct list_head    children;       /* list of my children */
    struct list_head    sibling;        /* link in parent->children */
    struct task_struct  *group_leader;

    /* resources (shared across a thread group) */
    struct mm_struct    *mm;            /* address space; NULL for kthreads */
    struct mm_struct    *active_mm;     /* borrowed mm while running */
    struct fs_struct    *fs;            /* cwd, root */
    struct files_struct *files;         /* open file table */
    struct signal_struct *signal;       /* shared signal state */
    struct sighand_struct *sighand;

    /* credentials & isolation */
    const struct cred __rcu *cred;      /* uid/gid/capabilities */
    struct nsproxy      *nsproxy;       /* pid/net/mnt/... namespaces */

    void                *stack;         /* kernel stack for this task */
    /* ... hundreds more fields ... */
};
```

Key distinctions:

- **`pid` vs `tgid`.** The `pid` field is the kernel's per-task identifier — what
  `gettid(2)` returns. The `tgid` is the thread-group id and is what userspace `getpid(2)`
  returns. For a single-threaded process `pid == tgid`. The `group_leader` points at the
  task whose `pid == tgid`.
- **`mm` vs `active_mm`.** `mm` is the task's own address space (`struct mm_struct`).
  Kernel threads have `mm == NULL` because they have no userspace; while running they
  *borrow* the previous task's page tables via `active_mm` (lazy TLB — see context switch).
- **Shared resource pointers.** Threads in the same group share `mm`, `files`, `signal`,
  and `sighand` (these are the resources `clone()` was asked to share via `CLONE_*`).

## `thread_info`, the Kernel Stack, and `current`

Each task has its own **kernel stack** (typically 8 KiB or 16 KiB, `THREAD_SIZE`,
config-dependent) used whenever the task runs in kernel mode (syscalls, traps, interrupts
on its behalf). Historically a small `struct thread_info` lived at the base of that stack,
letting the kernel find the current task by masking the stack pointer. On modern arches
with `CONFIG_THREAD_INFO_IN_TASK`, `thread_info` is embedded in `task_struct` instead, and
`current` is read from a per-CPU variable (e.g. `pcpu_hot.current_task` on x86-64).

```
   high addr ┌──────────────────────────┐
             │   kernel stack (grows ↓)  │   THREAD_SIZE (e.g. 16 KiB)
             │            ...            │
             │   [ guard page ]          │  VMAP_STACK: unmapped → overflow faults
   low  addr └──────────────────────────┘
   current ──► task_struct { ..., thread_info, *stack, ... }
```

`current` is a macro yielding the running task's `task_struct *`. With
`CONFIG_VMAP_STACK`, kernel stacks are vmalloc'd with a guard page so a stack overflow
faults cleanly instead of silently corrupting the neighbouring `task_struct` or another
task's data. See [Memory Management](memory_management.md) for the vmalloc area.

## PIDs and the Process Tree

A numeric PID is backed by a `struct pid` (`include/linux/pid.h`) that can map to several
numbers at once — one per [PID namespace](namespace.md) it is visible in. Lookups go
through an IDR per namespace rather than the old global hash table.

Two tasks anchor the tree:

- **PID 0 — the idle task** (`swapper`), one per CPU, run when nothing else is runnable.
- **PID 1 — `init`** (e.g. [systemd](systemd.md)), the first userspace task; it adopts
  orphaned children so they can be reaped.

```
           [0] swapper (idle, per-CPU)
                  │
           [1] init / systemd ──────────────┐
            ├── [930]  sshd                  │ real_parent / children / sibling
            │     └── [1450] bash            │ form the tree
            │            └── [1600] vim
            └── [410]  NetworkManager
```

Relationships are walked via `real_parent`, the `children` list, and the `sibling` link.
`real_parent` is the task that created it; `parent` may differ when a tracer (e.g. a
debugger using `ptrace`) is attached.

## Process States

`__state` (plus `exit_state` for the dying phase) tracks where a task is in its lifecycle
(`include/linux/sched.h`):

```c
TASK_RUNNING            /* runnable: on a runqueue or currently running */
TASK_INTERRUPTIBLE      /* sleeping; wakes on event OR signal      (ps: S) */
TASK_UNINTERRUPTIBLE    /* sleeping; wakes only on event           (ps: D) */
TASK_KILLABLE           /* uninterruptible but fatal signals wake it */
TASK_IDLE               /* uninterruptible sleep, not counted in load avg */
__TASK_STOPPED          /* stopped by SIGSTOP/SIGTSTP              (ps: T) */
__TASK_TRACED           /* stopped under a tracer                  (ps: t) */
/* exit_state: */
EXIT_ZOMBIE             /* dead, awaiting reap by parent           (ps: Z) */
EXIT_DEAD               /* being removed (transient) */
```

```
                wake_up()                  schedule()/preempt
   (new) ──► TASK_RUNNING ◄───────────────────────────────┐
              │  ▲   │ pick_next_task → on CPU             │
              │  │   └─────────────────────────────────────┘
   block on   │  │ event arrives / signal
   event ─────▼  │
        TASK_(UN)INTERRUPTIBLE
              │ do_exit()
              ▼
         EXIT_ZOMBIE ── parent wait() ──► EXIT_DEAD ──► freed
```

`TASK_UNINTERRUPTIBLE` (the **D** state) cannot be killed — a task stuck there (often on
broken I/O or NFS) is immune even to `SIGKILL`, which is why such hangs are notorious.
`TASK_KILLABLE` was introduced to get the same "don't wake on ordinary signals" behaviour
while still allowing fatal signals.

## Creation: `fork` / `vfork` / `clone` → `copy_process()`

`fork(2)`, `vfork(2)`, and `clone(2)` are thin wrappers; in the kernel they all funnel
through `kernel_clone()` → `copy_process()` in `kernel/fork.c`. The behavioural differences
are entirely a matter of which `CLONE_*` flags are passed:

| Flag             | Effect when set                                  |
|------------------|--------------------------------------------------|
| `CLONE_VM`       | Share the address space (`mm`) — makes a thread  |
| `CLONE_FILES`    | Share the open-file table                         |
| `CLONE_FS`       | Share cwd/root                                     |
| `CLONE_SIGHAND`  | Share signal handlers                             |
| `CLONE_THREAD`   | Same thread group (same `tgid`)                  |
| `CLONE_NEWPID` … | Create new [namespaces](namespace.md)            |

```
   pthread_create  →  clone(CLONE_VM|CLONE_FILES|CLONE_FS|
                            CLONE_SIGHAND|CLONE_THREAD, ...)
   fork            →  clone(SIGCHLD)            (copy everything, COW the mm)
   vfork           →  clone(CLONE_VM|CLONE_VFORK)  (share mm, parent blocks)
```

`copy_process()` does the heavy lifting:

1. `dup_task_struct()` allocates a new `task_struct` and a fresh kernel stack, copying the
   parent's descriptor.
2. Each resource is either **shared** (refcount bumped) or **copied**, per the flags:
   `copy_mm`, `copy_files`, `copy_fs`, `copy_sighand`, `copy_signal`, `copy_namespaces`, …
3. For a real `fork`, `copy_mm` does **not** duplicate physical pages. It clones the VMA
   layout and marks shared pages read-only so the first write triggers a
   **copy-on-write** fault — see COW in [Memory Management](memory_management.md).
4. A new `struct pid` is allocated and the task is wired into the tree, then made runnable
   via `wake_up_new_task()`.

**Kernel threads** are tasks with no userspace (`mm == NULL`). They are created with
`kthread_create()` / `kthread_run()`; the helper `kthreadd` (PID 2) is their common
ancestor. They run only in kernel mode and are visible in `ps` with bracketed names
(e.g. `[kworker/0:1]`).

## `execve()` Internals

`fork` and `exec` are deliberately separate. `execve(2)` (`kernel/exec.c`, via
`do_execveat_common` → `bprm_execve`) **replaces the program** running inside an *existing*
`task_struct` — the PID, parent relationship, and (mostly) open files survive; only the
program image changes:

1. Build a `struct linux_binprm` and pull argv/envp into a fresh, temporary stack.
2. `search_binary_handler()` asks each registered **binfmt** (e.g. `binfmt_elf`,
   `binfmt_script` for `#!`, `binfmt_misc`) to claim the file.
3. The chosen handler calls `begin_new_exec()`, which tears down the old address space and
   installs a brand-new `mm_struct`, then maps the ELF segments and the interpreter
   (`ld.so`) and sets the initial register state / entry point.
4. On success, control returns to userspace at the new program's entry; on failure the
   caller's old image is (where possible) left intact.

This is why the classic shell pattern is `fork()` in the parent, then `execve()` in the
child — see the userspace examples in [Process Management](process.md).

## Termination and Reaping

Exit flows through `do_exit()` in `kernel/exit.c` (group exits and fatal signals go via
`do_group_exit()` so all threads in the group die together):

1. Release resources: drop references to `mm`, `files`, `fs`, signal state, etc.
2. Notify the parent with `SIGCHLD` and set `exit_state = EXIT_ZOMBIE`. The task is now a
   **zombie**: nearly everything is freed, but the `task_struct` lingers so the parent can
   read the exit status.
3. The parent calls a `wait(2)`-family syscall; `wait_task_zombie()` collects `rusage`/exit
   code and `release_task()` finally frees the descriptor (`EXIT_DEAD`).
4. If the parent dies first, the children are **reparented** — to the nearest process marked
   as a *child subreaper* (`PR_SET_CHILD_SUBREAPER`) or otherwise to init (PID 1), which
   reaps them.

```
   child do_exit() ──► EXIT_ZOMBIE ──┐
                                      │ SIGCHLD to parent
   parent wait()  ◄──────────────────┘
        └─► wait_task_zombie() → release_task() → struct freed (EXIT_DEAD)
```

A parent that never reaps leaks **zombies** (they hold a PID slot but no other resources);
a parent that dies leaves **orphans**, which init reaps automatically.

## Scheduling

The scheduler decides which runnable task each CPU runs next. It is organised as a stack of
**scheduler classes** in priority order (`kernel/sched/`); the highest class with a runnable
task wins:

```
   stop_sched_class   (migration/CPU-stop — highest)
   dl_sched_class     (SCHED_DEADLINE — EDF)
   rt_sched_class     (SCHED_FIFO / SCHED_RR — fixed priority)
   fair_sched_class   (SCHED_NORMAL/BATCH/IDLE — EEVDF, the default)
   idle_sched_class   (the per-CPU idle task — lowest)
```

Each CPU owns a `struct rq` (runqueue) holding the runnable tasks for that CPU. The fair
class historically used **CFS**, which ordered tasks by `vruntime` (accumulated runtime
weighted by nice) in a red-black tree and always picked the smallest. Since **v6.6** the
fair class uses **EEVDF** (Earliest Eligible Virtual Deadline First), which adds a per-task
*lag* and *virtual deadline* so latency-sensitive tasks (e.g. interactive) get serviced
sooner while long-run fairness is preserved. (Much online material still describes plain
CFS; on a current kernel the picker is EEVDF.)

```
   CPU0 rq                         CPU1 rq
   ┌───────────────┐               ┌───────────────┐
   │ rt:  [vim]    │               │ rt:  ∅        │
   │ fair (EEVDF): │               │ fair (EEVDF): │
   │   bash, cc1   │  load balance │   make, sshd  │
   │ idle: swapper │ ◄───────────► │ idle: swapper │
   └───────────────┘               └───────────────┘
```

The entry point is `schedule()` → `__schedule()` → `pick_next_task()`, which walks the
classes and calls the winner's `pick_next_task`. A switch happens on:

- **Voluntary** scheduling — a task blocks (sleeps on I/O, a lock, a wait queue) and calls
  `schedule()` itself.
- **Preemption** — a higher-priority task becomes runnable, or the timer tick decides the
  current task has run long enough (`TIF_NEED_RESCHED` is set and acted on at the next safe
  point). `CONFIG_PREEMPT_*` controls whether the kernel can be preempted while running in
  kernel mode.

`nice` (−20..19) maps to a CPU **weight**: lower nice → more weight → larger share. Realtime
policies (`SCHED_FIFO`/`SCHED_RR`) always outrank normal tasks; `chrt` sets them.

## Context Switch

Once `pick_next_task()` selects the next task, `context_switch()` (`kernel/sched/core.c`)
swaps the CPU from `prev` to `next` in two steps:

1. **`switch_mm()`** — install `next`'s page tables (load `CR3` on x86, swap TTBR on ARM)
   and handle TLB/ASID bookkeeping. If `next` is a **kernel thread** (`mm == NULL`) the
   kernel performs a **lazy TLB switch**: it keeps `prev`'s page tables active and records
   them in `next->active_mm`, avoiding a pointless TLB flush since kernel threads never touch
   userspace addresses.
2. **`switch_to()`** — an arch-specific routine that saves `prev`'s callee-saved registers
   and kernel stack pointer and restores `next`'s, so execution resumes inside `next` exactly
   where it last left `__schedule()`.

```
   __schedule()
      ├─ pick_next_task() ─► next
      └─ context_switch(prev, next)
            ├─ switch_mm(prev->mm, next->mm)   # address space (or lazy/active_mm)
            └─ switch_to(prev, next)           # registers + kernel stack
                  └─► now running as `next`
```

After the switch, `current` points at `next`, and userspace (if any) resumes on `next`'s
restored register state. This pairs with [Memory Management](memory_management.md): the
address-space swap is what makes each process see its own virtual memory.

## Where this connects

- [Process Management](process.md) — the userspace counterpart: PIDs/PGIDs, memory layout,
  the `fork`/`exec`/`wait` API, IPC, `/proc`, and process tooling.
- [Kernel Architecture](kernel.md) — where the scheduler and process subsystem sit among the
  kernel's components.
- [Memory Management](memory_management.md) — `mm_struct`, page tables, COW, and the vmalloc
  area backing kernel stacks.
- [Namespaces](namespace.md) — `nsproxy`/`struct pid` give each PID namespace its own view of
  the process tree (init at PID 1 inside the namespace).
- [cgroups](cgroups.md) — bound and account the CPU/memory a group of tasks may use; the
  scheduler honours cgroup CPU weights and quotas.
- [systemd](systemd.md) — PID 1 in practice: it reaps orphans and supervises service tasks.
- [Kernel Development Patterns](kernel_patterns.md) — wait queues, completions, and work
  queues are how kernel code sleeps tasks and wakes them.

## Pitfalls

- **D-state hangs.** A task in `TASK_UNINTERRUPTIBLE` ignores every signal, including
  `SIGKILL`; `kill -9` can't clear an NFS/I/O wedge. Find them with `ps -eo pid,stat,comm | awk '$2 ~ /D/'`
  and check `/proc/<pid>/stack`.
- **Zombie leaks.** A long-lived parent that never `wait()`s accumulates zombies that pin PID
  slots. Reap children (handle `SIGCHLD`) or let them be reparented to init.
- **`mm` vs `active_mm`.** Kernel threads have `mm == NULL` and borrow `active_mm`. Code that
  blindly dereferences `current->mm` from kernel-thread context will fault — check for NULL.
- **Stack overflow.** The kernel stack is small (8–16 KiB). Deep recursion or large on-stack
  buffers overflow it; without `CONFIG_VMAP_STACK` guard pages this silently corrupts
  adjacent memory rather than faulting.
- **`pid` vs `tgid` confusion.** `getpid()` returns `tgid`; `gettid()` returns the kernel
  `pid`. Logging the wrong one makes multi-threaded traces impossible to follow.
- **Stale scheduler docs.** Material describing CFS's red-black tree and `vruntime`-only
  selection predates v6.6; current kernels schedule the fair class with **EEVDF**.
