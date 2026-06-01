# rr (Record & Replay)

## Overview

`rr` is a record-and-replay debugger from Mozilla. It records a program's execution
once — every non-deterministic input (syscalls, signals, thread scheduling) — into a
trace, then replays that trace *deterministically* as many times as you like under
[GDB](gdb.md). The killer feature is **reverse execution**: `reverse-continue` and
`reverse-step` run the program backwards to the instruction that corrupted your state.
It turns Heisenbugs and rare races into something you can reproduce on demand, which is
exactly the hard case [Core Dump Analysis](core_dump.md) and a live debugger struggle
with. GDB has built-in reverse debugging, but it is far slower; rr makes it practical.

```
Workflow:
  rr record ./prog args      →  trace saved to ~/.local/share/rr/<name>
  rr replay                  →  deterministic re-run inside GDB
       (gdb) continue        →  forward to a crash/breakpoint
       (gdb) reverse-continue→  backward to the previous hit
       (gdb) reverse-next    →  step backward over a line
```

## Recording

```bash
rr record ./prog arg1 arg2           # record one run
rr record -n ./prog                  # don't compress (faster record, bigger trace)
rr record --chaos ./prog             # randomise scheduling to surface rare races
rr ps                                # list recorded traces
```

Recording forces threads onto a single core and serialises them, so a race that needs
true parallelism may not reproduce — `--chaos` mode varies scheduling to flush these
out, and once recorded the race is captured forever.

## Replaying & reverse execution

```bash
rr replay                            # replay the latest trace under GDB
rr replay <trace-dir>                # a specific trace
```

```
(gdb) continue                 # run forward to the SIGSEGV
(gdb) watch -l some_global     # hardware watchpoint on a memory location
(gdb) reverse-continue         # run BACKWARD until the watchpoint last changed
(gdb) reverse-next / rn        # step backward over the current line
(gdb) reverse-step  / rs       # step backward into calls
(gdb) reverse-finish           # back out to the caller
```

The classic loop for "who set this variable to garbage?":

```
1. replay → continue to the crash
2. watch -l the bad variable / memory address
3. reverse-continue  → stops at the exact write that set the bad value
4. inspect the backtrace there; repeat to walk further back
```

## Why it's deterministic

```
Normal run:                      rr replay:
  scheduler timing varies          identical scheduling, recorded
  syscall return values vary        replayed from the trace
  ASLR addresses random             same addresses every replay
  → bug appears 1 run in 1000      → bug appears EVERY replay
```

Because every replay is byte-identical, watchpoints, conditional breakpoints, and
logging all behave reproducibly — and event numbers (`when`, `seek-ticks`) let you jump
to an exact point in time.

## Pitfalls

```
- Requires hardware performance counters → bare metal or a VM/cloud instance that exposes the PMU
  (many cloud VMs and containers don't; needs perf_event access).
- Single-core serialised recording: ~1.2–2× slowdown, and true-parallel races may not record
  without --chaos.
- x86/x86-64 and (newer) ARM64 only; CPU-feature mismatches between record and replay break it.
- Traces can be large; they capture inputs, so treat them as sensitive (may contain secrets).
- Some syscalls / GPU / ptrace-heavy programs aren't supported.
- It records ONE execution — if you didn't capture the bug, re-record (use --chaos for rare ones).
- Reverse execution is the point; plain forward debugging is better served by GDB/LLDB directly.
```

## Where this connects

- [GDB](gdb.md) — rr replays inside GDB; reverse-* commands are GDB's, made fast by rr
- [LLDB](lldb.md) — alternative debugger (limited reverse support)
- [Core Dump Analysis](core_dump.md) — post-mortem when you can't record live
- [Debugging Methodology](debugging_methodology.md) — reverse execution as scientific-method bisection in time
- [Concurrency](../programming/concurrency.md) — `--chaos` for surfacing data races
- [Sanitizers](sanitizers.md) — TSan to find the race, rr to replay and understand it
