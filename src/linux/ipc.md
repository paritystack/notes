# Inter-Process Communication

## Overview

This page surveys the kernel mechanisms processes use to **talk to each other and synchronise**:
pipes/FIFOs, signals, shared memory, semaphores, message queues, the `futex` behind userspace
locks, the `*fd` family (`eventfd`/`signalfd`/`memfd`), and **Unix domain sockets** with file-
descriptor passing. It complements [Process Management](process.md) (which covers signals and job
control from the shell side) and [Process Internals (Kernel)](process_internals.md) (the
`task_struct`/`fork`/`exec` mechanics). IPC objects are namespaced — see the **IPC namespace** in
[Namespace](namespace.md) — and the socket-based mechanisms share the stack described in
[Networking](networking.md).

There is no single "IPC API"; instead a toolbox, chosen by *what you're moving* (a byte stream, a
notification, a chunk of shared state) and *between whom* (related vs unrelated processes).

```
   notify only            ──►  signals, eventfd, pipes(empty writes)
   byte stream / datagrams ──►  pipes/FIFOs, Unix sockets, message queues
   shared state (zero-copy)──►  shared memory (+ a semaphore/futex to guard it)
   mutual exclusion        ──►  futex (the basis of pthread mutexes/semaphores)
   pass an fd / credentials ──►  Unix socket SCM_RIGHTS / SO_PEERCRED
```

## Pipes and FIFOs

A **pipe** is a unidirectional in-kernel byte buffer: `pipe(fd)` gives a read end and a write end,
inherited across `fork()` — the classic shell `cmd1 | cmd2` plumbing. A **FIFO** (named pipe,
`mkfifo`) is the same buffer exposed as a filesystem path, so *unrelated* processes can open it.

```c
int fd[2]; pipe(fd);
if (fork() == 0) { close(fd[1]); read(fd[0], buf, n); }   /* child reads */
else            { close(fd[0]); write(fd[1], data, n); }   /* parent writes */
```

Semantics worth knowing: writes up to `PIPE_BUF` (4096) are atomic; reading a pipe whose write
ends are all closed returns EOF; writing to a pipe with no readers raises `SIGPIPE`.

## Signals

Signals are asynchronous notifications (`SIGTERM`, `SIGINT`, `SIGCHLD`, `SIGKILL`/`SIGSTOP` —
the latter two uncatchable). A handler interrupts the target at the next safe point. Key rules:

- Handlers run in a constrained context — only **async-signal-safe** functions are legal inside
  (no `malloc`, no `printf`). The robust pattern is to set a `volatile sig_atomic_t` flag (or
  write a byte to a self-pipe / `signalfd`) and act in the main loop.
- Use `sigaction()` (not `signal()`) for portable, well-defined behaviour, and block signals with
  `sigprocmask()` around critical sections.
- **`signalfd()`** turns signals into readable file descriptors, so they integrate with
  `epoll`/`poll` event loops instead of needing handlers.

See [Process Management](process.md) for delivery, job control, and process-group signalling.

## System V vs POSIX IPC

Two parallel APIs exist for shared memory, semaphores, and message queues:

| Object        | System V                | POSIX                       | Notes |
|---------------|-------------------------|-----------------------------|-------|
| Shared memory | `shmget`/`shmat`        | `shm_open`+`mmap`           | POSIX is fd-based, nicer |
| Semaphore     | `semget`/`semop`        | `sem_open`/`sem_wait`       | POSIX has named + unnamed |
| Message queue | `msgget`/`msgsnd`       | `mq_open`/`mq_send`         | POSIX mq supports priorities + `mq_notify` |

POSIX IPC is generally preferred: it's file-descriptor / pathname based (`/dev/shm`, `/dev/mqueue`),
cleans up more predictably, and integrates with `poll`. System V IPC persists until explicitly
removed (`ipcrm`) or reboot — visible via `ipcs`.

### Shared memory + a guard

Shared memory is the fastest IPC (zero-copy: the same physical pages mapped into multiple address
spaces) but provides *no synchronisation* — you must pair it with a semaphore or futex:

```c
int fd = shm_open("/myseg", O_CREAT|O_RDWR, 0600);
ftruncate(fd, size);
void *p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
/* p is shared; protect concurrent access with a sem_t/futex placed inside it */
```

## futex: the basis of userspace locks

A **futex** ("fast userspace mutex") makes the *uncontended* case lock/unlock entirely in
userspace via an atomic compare-and-swap on a shared integer — only when a thread must **wait** or
**wake** does it call into the kernel (`futex(FUTEX_WAIT/FUTEX_WAKE)`).

```
   lock:  atomic CAS 0→1 in userspace        (no syscall if free)
          contended? ─► futex(FUTEX_WAIT)     (sleep in kernel)
   unlock: atomic set →0
          waiters?  ─► futex(FUTEX_WAKE)       (wake one)
```

You rarely call `futex()` directly — it's the engine under `pthread_mutex`, `pthread_cond`,
`sem_t`, and Go/Rust locks. Priority-inheritance futexes (`PI-futex`) avoid priority inversion for
RT workloads (see [CPU Scheduler](scheduler.md)).

## The fd family: eventfd, memfd

- **`eventfd()`** — a kernel-managed 64-bit counter you read/write as an fd; ideal for one process
  (or thread, or the kernel via AIO/KVM) to *notify* another inside an `epoll` loop.
- **`memfd_create()`** — an anonymous memory-backed file (no filesystem name). Combined with seals
  (`F_SEAL_*`) it's the modern way to share a read-only/​fixed-size buffer over a socket safely.

## Unix domain sockets

Unix sockets (`AF_UNIX`) are the most flexible local IPC: stream or datagram, bidirectional,
poll-able, with a filesystem or *abstract* (leading-NUL) address. Two superpowers over pipes:

- **Fd passing** via `SCM_RIGHTS` ancillary data — hand an open file/socket to another process
  (how a server passes a connection to a worker, or systemd passes listening sockets).
- **Peer credentials** via `SO_PEERCRED`/`SCM_CREDENTIALS` — the kernel attests the peer's
  PID/UID/GID, the basis of local authentication (polkit, D-Bus).

```c
/* send fd 'payload_fd' over unix socket 'sock' */
struct msghdr msg = {0}; char cbuf[CMSG_SPACE(sizeof(int))] = {0};
msg.msg_control = cbuf; msg.msg_controllen = sizeof cbuf;
struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS;
c->cmsg_len = CMSG_LEN(sizeof(int));
*(int *)CMSG_DATA(c) = payload_fd;
sendmsg(sock, &msg, 0);
```

Higher-level local IPC (D-Bus, Wayland, X11, container runtimes) is built on Unix sockets.

## Choosing a mechanism

| Need | Reach for |
|------|-----------|
| Shell-style stream between related procs | pipe |
| Stream between unrelated procs | FIFO or Unix socket |
| Notify an event loop | eventfd / signalfd |
| Zero-copy shared state | shared memory + semaphore/futex |
| Mutual exclusion (in-process) | futex (via pthread) |
| Pass fds / authenticate peer | Unix socket (SCM_RIGHTS / SO_PEERCRED) |
| Prioritised discrete messages | POSIX message queue |
| Cross-machine | TCP/UDP sockets — see [Networking](networking.md) |

## Where this connects

- [Process Management](process.md) — signals, job control, and process groups from the userspace
  side; this page adds the programmatic and shared-memory mechanisms.
- [Process Internals (Kernel)](process_internals.md) — `fork`/`exec` determine which fds and IPC
  objects are inherited; address spaces are what shared memory maps into.
- [Namespace](namespace.md) — the **IPC namespace** isolates System V IPC and POSIX mqueues;
  Unix-socket abstract addresses are network-namespaced.
- [Networking](networking.md) — Unix sockets share the socket API and stack; cross-host IPC uses
  TCP/UDP.
- [CPU Scheduler](scheduler.md) — futex waits put tasks to sleep; PI-futexes address priority
  inversion for RT.

## Pitfalls

- **Calling unsafe functions in a signal handler.** Only async-signal-safe functions are legal;
  use the self-pipe/`signalfd` pattern and a `sig_atomic_t` flag instead of doing work inline.
- **Ignoring `SIGPIPE`.** Writing to a pipe/socket with no reader kills the process by default;
  handle or ignore `SIGPIPE` (or use `MSG_NOSIGNAL`).
- **Shared memory without synchronisation.** Mapping the same pages buys you nothing safe without
  a semaphore/futex; expect torn reads and races otherwise.
- **Leaking System V IPC.** `shm`/`sem`/`msg` objects survive process exit; orphaned segments
  accumulate (`ipcs`/`ipcrm`). Prefer POSIX/​fd-based objects that close with the process.
- **Assuming pipe writes are atomic.** Only writes ≤ `PIPE_BUF` are; larger writes interleave with
  other writers.
- **Trusting a peer's claimed PID/UID.** Read it from the kernel via `SO_PEERCRED`, never from
  data the peer sends.
- **Fd leaks across `exec`.** Fds without `O_CLOEXEC`/`FD_CLOEXEC` leak into children (and over
  `SCM_RIGHTS`); set close-on-exec by default.
