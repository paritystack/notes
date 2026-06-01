# perf & Flame Graphs

## Overview

`perf` is the Linux kernel's profiling toolkit. It uses hardware performance counters
(PMUs) and kernel tracepoints to answer "where is the time/cache/branch budget going?"
with very low overhead — it *samples* rather than instruments, so it can profile a
whole machine in production. This is the opposite trade-off from
[Valgrind](valgrind.md) and [Sanitizers](sanitizers.md), which check correctness by
slowing the program down. Flame graphs turn perf's stack samples into a picture of the
hot paths. For tracing individual syscalls see [strace in Binary Analysis
Tools](tools.md); for programmable tracing see [eBPF](../linux/ebpf.md).

```
perf data sources:
  hardware events   → cycles, instructions, cache-misses, branch-misses (PMU)
  software events   → context-switches, page-faults, cpu-clock
  tracepoints       → kernel events (syscalls, scheduler, block I/O)
  dynamic probes    → kprobes (kernel) / uprobes (userspace) on any function
```

## perf stat — counters for a run

```bash
perf stat ./prog                     # summary: cycles, IPC, branches, misses
perf stat -d ./prog                  # detailed: cache levels too
perf stat -e cache-misses,branch-misses ./prog
perf stat -r 5 ./prog                # repeat 5×, show mean ± stddev
```

```
 1,234,567,890  cycles
   987,654,321  instructions     #  0.80  insn per cycle   ← IPC: <1 means stalls
     2,345,678  cache-misses     #  3.21% of all cache refs
       456,789  branch-misses    #  0.92% of all branches
```

Low IPC points at memory stalls or branch mispredicts; high cache-miss rate points at
poor data locality (see [Cache & TCM](../embedded/cache_tcm.md) and data layout).

## perf record / report — sampling profiler

```bash
perf record -g ./prog                # -g = capture call graphs (stacks)
perf record -g -F 999 ./prog         # sample at 999 Hz
perf record -g -p <pid> -- sleep 10  # attach to a running process for 10s
perf record -g -a -- sleep 10        # whole system

perf report                          # interactive TUI, sorted by self/children
perf report --stdio                  # plain text
```

For reliable stacks compile with `-fno-omit-frame-pointer`, or use `--call-graph dwarf`
(uses debug info, heavier) or `--call-graph lbr` (Intel Last Branch Record, cheap).

## Flame graphs

A flame graph collapses thousands of stack samples into one SVG: width = time spent,
y-axis = stack depth. The widest boxes at any level are where to optimise.

```bash
git clone https://github.com/brendangregg/FlameGraph
perf record -g -F 999 ./prog
perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > out.svg
```

```
        ┌───────────────────────────────────────────┐
        │ main                                        │  ← root, full width
        ├───────────────────────┬─────────────────────┤
        │ parse_input           │ compute             │
        ├───────────┬───────────┼──────────┬──────────┤
        │ tokenize  │ validate  │ matmul   │ normalize│  ← widest leaf = hottest
        └───────────┴───────────┴──────────┴──────────┘
        width = proportion of samples (≈ CPU time)
```

## On-CPU vs off-CPU

```
on-CPU   → why is the CPU busy?      perf record (the default; flame graph above)
off-CPU  → why is it waiting/blocked? sched tracepoints / eBPF offcputime
```

```bash
perf top                             # live, htop-style hottest functions
perf record -e sched:sched_switch -g -a -- sleep 5   # scheduling/off-CPU analysis
```

For off-CPU and latency analysis, [eBPF](../linux/ebpf.md) tools (`offcputime`,
`bpftrace`) are usually better than raw perf tracepoints.

## Other perf subcommands

```bash
perf list                            # all available events on this CPU
perf mem record / perf mem report    # memory load/store latency
perf c2c                             # cache-line false sharing between cores
perf annotate <symbol>               # per-instruction hotspots in disassembly
perf probe --add 'do_sys_open filename:string'   # add a dynamic kprobe
```

## Pitfalls

```
- Broken/short stacks → rebuild with -fno-omit-frame-pointer, or use --call-graph dwarf/lbr.
- Missing symbols → install -dbg/-debuginfo packages; strip removes them (see tools.md).
- perf_event_paranoid sysctl blocks unprivileged profiling: sysctl kernel.perf_event_paranoid=1.
- Sampling can miss short, infrequent functions — raise -F or use a tracer instead.
- Profiling a debug (-O0) build optimises the wrong thing; profile -O2 with -g.
- In containers/VMs, hardware PMU counters may be unavailable — fall back to software events.
- Flame graph width is sample count, not wall-clock — off-CPU time is invisible in an on-CPU graph.
```

## Where this connects

- [Valgrind](valgrind.md) — callgrind/cachegrind give deterministic *simulated* profiles
- [Sanitizers](sanitizers.md) — fix correctness first, then profile
- [eBPF](../linux/ebpf.md) — programmable tracing; off-CPU, latency histograms, bpftrace
- [Binary Analysis Tools](tools.md) — strace/ltrace for syscall/library-level tracing
- [Linux Kernel Debugging](linux_kernel.md) — ftrace and kernel tracepoints
- [Performance & Profiling](../android/performance_profiling.md) — simpleperf/systrace on Android
- [Cache & TCM](../embedded/cache_tcm.md) — what cache-miss counters are telling you
