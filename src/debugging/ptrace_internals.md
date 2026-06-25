# ptrace & Debugger Internals

## Overview

Every interactive debugger on Linux — [gdb](gdb.md), [lldb](lldb.md), the system-call
tracers in [tools.md](tools.md) (`strace`/`ltrace`), and the record/replay engine
[rr](rr_debugging.md) — is built on one kernel facility: the **`ptrace(2)`** system call.
`ptrace` lets one process (the *tracer*) observe and control the execution of another (the
*tracee*): stop it on signals and syscalls, read and write its memory and registers, and
single-step it instruction by instruction. The fancy source-level features you actually use
— breakpoints, watchpoints, `next`/`step`, variable inspection — are all built in userspace
*on top of* these few primitives, with [DWARF debug info](#source-level-mapping) bridging
machine addresses back to source. Where [perf](perf_profiling.md) samples without stopping
the target, `ptrace` is the **stop-the-world, full-control** path.

```
   TRACER (gdb)                       TRACEE (your program)
   -----------                        ---------------------
   ptrace(ATTACH/SEIZE)  ───────────▶ stopped
   ptrace(POKETEXT 0xCC) ───────────▶ breakpoint armed
   ptrace(CONT)          ───────────▶ runs ... hits 0xCC → SIGTRAP
   waitpid() ◀──────────────────────  stops, kernel notifies tracer
   ptrace(GETREGSET) read state
   ptrace(POKETEXT orig) restore byte, SINGLESTEP, re-arm
   ptrace(CONT)          ───────────▶ continues
```

## The ptrace model

A tracer attaches to a tracee in one of two ways:

- **`PTRACE_TRACEME`** — the *child* calls this right after `fork()` and before `exec()`,
  marking itself as traced by its parent. This is how a debugger launches a fresh process
  under its control.
- **`PTRACE_ATTACH`** / **`PTRACE_SEIZE`** — the tracer attaches to an *already-running*
  process by PID (`gdb -p`, `strace -p`). `SEIZE` is the modern variant: it does not send a
  spurious `SIGSTOP`, and it enables richer stop reporting.

Once attached, the tracee stops and the tracer drives it through a **wait loop**. Every time
the tracee stops, the kernel notifies the tracer via `waitpid()`, whose status word tells you
*why* it stopped:

```
   while ((pid = waitpid(child, &status, 0)) > 0) {
       if (WIFEXITED(status))   break;           // tracee finished
       if (WIFSTOPPED(status)) {
           int sig = WSTOPSIG(status);
           // signal-delivery-stop, syscall-stop, group-stop, or event-stop
           ptrace(PTRACE_CONT, child, 0, sig);   // optionally inject the signal
       }
   }
```

Key stop kinds:

- **Signal-delivery-stop** — the tracee was about to receive a signal; the tracer sees it
  *first* and decides whether to deliver, suppress, or substitute it.
- **Syscall-stop** — with `PTRACE_SYSCALL`, the tracee stops on syscall *entry* and again on
  *exit*. This is exactly how `strace` enumerates every call and its return value.
- **Group-stop / event-stop** — job-control stops, and `PTRACE_EVENT_*` notifications
  (fork/exec/clone/exit) enabled via `PTRACE_O_*` options like `PTRACE_O_TRACEEXEC` and
  `PTRACE_O_TRACECLONE` (how a debugger follows `fork`/new threads).

## Software breakpoints: the INT3 dance

A software breakpoint is a one-byte trick. The debugger overwrites the first byte of the
target instruction with **`0xCC`** — the x86 `INT3` "breakpoint trap" — saving the original
byte. When the CPU executes `0xCC`, it raises a trap that the kernel turns into a `SIGTRAP`,
stopping the tracee and notifying the tracer.

```
   original:  48 89 e5      mov %rsp,%rbp
   armed:     CC 89 e5      <int3> 89 e5         (orig 0x48 saved by gdb)

   hit:  CPU executes CC → trap → SIGTRAP → tracer wakes in waitpid()
         RIP now points just AFTER the 0xCC (one byte past breakpoint addr)

   to continue transparently:
     1. rewind RIP by 1 (back onto the breakpoint address)
     2. restore the saved original byte (0x48)
     3. PTRACE_SINGLESTEP over the now-real instruction
     4. re-write 0xCC to re-arm the breakpoint
     5. PTRACE_CONT
```

That restore → single-step → re-arm cycle is why a breakpoint that fires in a hot loop is
expensive: each hit is several `ptrace` round-trips and context switches between tracer and
tracee.

## Hardware breakpoints & watchpoints

The CPU also provides **debug registers** `DR0`–`DR7`. `DR0`–`DR3` hold up to four linear
addresses; `DR7` configures each as an instruction breakpoint or a **data watchpoint** that
traps on read/write/execute of 1/2/4/8 bytes. Hardware watchpoints are how gdb's
`watch expr` can be fast — instead of single-stepping the whole program and re-checking the
value (the slow "software watchpoint" fallback), the CPU traps only when that memory is
touched. The limit is the number of debug registers (typically 4), which is why gdb warns
when you exceed them.

```
   gdb:  watch counter      → DR0 = &counter, DR7: len=4, rw=write
         hardware traps the exact store, no per-instruction overhead
```

## Single-stepping & reading state

- **`PTRACE_SINGLESTEP`** sets the CPU's trap flag so exactly one instruction executes, then
  a `SIGTRAP` fires. Source-level `step`/`next` are built on this plus DWARF line tables: the
  debugger single-steps until the line number changes (and steps *over* calls for `next` by
  setting a temporary breakpoint at the return address).
- **`PTRACE_CONT`** resumes until the next stop; **`PTRACE_SYSCALL`** resumes until the next
  syscall boundary.
- **Reading/writing tracee memory**: the classic word-at-a-time `PTRACE_PEEKTEXT`/`POKETEXT`
  and `PTRACE_PEEKDATA`/`POKEDATA`, or — far faster for bulk transfers — reading/writing
  `/proc/<pid>/mem` directly, or `process_vm_readv`/`process_vm_writev`.
- **Registers**: `PTRACE_GETREGSET`/`SETREGSET` (the modern, arch-neutral form using
  `NT_PRSTATUS` etc.) fetch the general-purpose and FP/vector register banks.

## Source-level mapping

`ptrace` works purely in addresses and bytes. Turning `0x4011a6` into `main.c:42` and a stack
frame into named local variables is the job of **DWARF** debug info embedded in the binary
(`.debug_line`, `.debug_info`). The debugger reads DWARF to place breakpoints by file:line, to
unwind the stack, and to locate variables. This is also why stripped binaries debug poorly —
no DWARF, only raw addresses — and why `debuginfod` exists to fetch matching symbols on
demand. The same debug info feeds [core dump](core_dump.md) analysis and
[perf](perf_profiling.md) symbolication.

## ptrace_scope: why attach is denied

On most distros the Yama LSM restricts who may `PTRACE_ATTACH` to a running process via the
`kernel.yama.ptrace_scope` [sysctl](../linux/sysctl.md):

```
   0  classic     — any process may trace another with the same uid
   1  restricted  — only a parent may trace its child (default on Ubuntu)
   2  admin-only   — only CAP_SYS_PTRACE may trace
   3  no attach    — ptrace attach disabled entirely
```

This is why `gdb -p <pid>` or `strace -p <pid>` against an unrelated process often fails with
`Operation not permitted` until you lower the value or run with `CAP_SYS_PTRACE`. Containers
frequently drop this capability, which is why debugging inside one needs `--cap-add=SYS_PTRACE`.

## Where this connects

- **[gdb](gdb.md) / [lldb](lldb.md)** — breakpoints, watchpoints, and stepping are the
  userspace layer over the primitives here.
- **[Binary analysis tools](tools.md)** — `strace`/`ltrace` are thin loops over
  `PTRACE_SYSCALL` and library-call interception.
- **[rr](rr_debugging.md)** — records the non-deterministic inputs seen through ptrace +
  `seccomp`, then replays deterministically.
- **[Sanitizers](sanitizers.md)** — an *alternative* to ptrace: compile-time instrumentation
  instead of an external tracer, with much lower per-event cost.
- **[Core dumps](core_dump.md)** — the same DWARF mapping, applied post-mortem.

## Pitfalls

- **One tracer per tracee.** A process can be traced by only one tracer at a time — you can't
  attach gdb to a process already under strace.
- **`ptrace` is slow at scale.** Every breakpoint hit and every syscall stop is two context
  switches; tracing a syscall-heavy program with `strace` can slow it 10–100×. Prefer
  [ftrace](ftrace.md)/[eBPF](../linux/ebpf.md) for low-overhead, always-on observation.
- **RIP off-by-one.** After a software breakpoint the instruction pointer sits *past* the
  `0xCC`; forgetting to rewind it corrupts execution.
- **Detaching matters.** A tracer that dies without detaching can leave the tracee stopped;
  `PTRACE_O_EXITKILL` makes the tracee die with the tracer.
- **Hardware watchpoint scarcity.** Only ~4 debug registers; gdb silently falls back to
  glacially slow software watchpoints when you ask for more.
