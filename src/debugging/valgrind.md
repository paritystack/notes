# Valgrind

## Overview

Valgrind is a dynamic binary instrumentation framework: it runs your program on a
synthetic CPU and watches every memory access and instruction. That makes it the
go-to tool for catching memory errors and leaks that a normal run hides, and it needs
no recompilation (though `-g` gives line numbers). The cost is speed — programs run
10–50× slower. For a faster, compile-time alternative see [Sanitizers](sanitizers.md);
for profiling without correctness checks see [perf & Flame Graphs](perf_profiling.md).
Valgrind complements [GDB](gdb.md) (you can attach GDB to a running Valgrind session)
and pairs with [Core Dump Analysis](core_dump.md) when a crash is hard to reproduce.

```
Valgrind is a suite of tools (pick one with --tool=):
  memcheck   (default) → memory errors & leaks
  helgrind            → data races, lock-order violations (POSIX threads)
  drd                 → data races (lower memory, per-thread model)
  cachegrind          → cache + branch-prediction simulation
  callgrind           → call-graph profiler (cachegrind + call counts)
  massif              → heap profiler (memory-over-time)
  dhat                → heap access patterns / churn
```

## Memcheck — the default tool

Detects: reads/writes of unaddressable memory, use of uninitialised values,
use-after-free, double-free, mismatched `new`/`delete[]`, and memory leaks.

```bash
# Compile with -g (and -O0 ideally) for readable reports
gcc -g -O0 prog.c -o prog

valgrind ./prog                              # memcheck is the default
valgrind --leak-check=full ./prog            # per-leak stack traces
valgrind --leak-check=full --show-leak-kinds=all \
         --track-origins=yes ./prog          # where uninit values came from
valgrind --error-exitcode=1 ./prog           # non-zero exit on any error (CI)
```

```
Leak categories (memcheck):
  definitely lost  → no pointer to the block at all          (real leak, fix it)
  indirectly lost  → only reachable via a definitely-lost block
  possibly lost    → only an interior pointer remains        (often real)
  still reachable   → pointer exists at exit                  (usually benign)
```

`--track-origins=yes` is the single most useful flag for "use of uninitialised value"
errors — it tells you which allocation the garbage came from (at a memory cost).

## Helgrind & DRD — thread errors

For [Concurrency](../programming/concurrency.md) bugs: data races, misuse of pthreads
APIs, and lock-ordering problems that could deadlock.

```bash
valgrind --tool=helgrind ./threaded_prog     # races + lock-order graph
valgrind --tool=drd      ./threaded_prog     # races, lighter on memory
```

Helgrind builds a happens-before graph from synchronisation primitives; DRD models
each thread's segments. They report different false positives, so trying both helps.
For races, [TSan](sanitizers.md) (ThreadSanitizer) is usually faster and clearer.

## Cachegrind & Callgrind — profiling

```bash
valgrind --tool=cachegrind ./prog            # I1/D1/LL cache miss counts
cg_annotate cachegrind.out.<pid>             # annotate source with miss data

valgrind --tool=callgrind ./prog             # call graph + instruction counts
callgrind_annotate callgrind.out.<pid>
kcachegrind callgrind.out.<pid>              # GUI: call graph, hot paths
```

Because the CPU is simulated, results are deterministic and independent of machine
load — great for comparing two implementations, but they measure *simulated* cache
behaviour, not your real CPU. Use [perf](perf_profiling.md) for real hardware counters.

## Massif — heap profiling

```bash
valgrind --tool=massif ./prog                # snapshots of heap usage over time
ms_print massif.out.<pid>                    # ASCII graph + allocation tree
massif-visualizer massif.out.<pid>           # GUI
```

```
    MB
  8 ^                                  #
    |                              @@@ #
    |                          @@@@@@@ #::
    |                    :::::@@@@@@@@ #::
  0 +----------------------------------------------> instructions
        ↑ each column is a snapshot; # = detailed, @ = peak contributors
```

## Suppressions

Third-party libraries (glibc, GUI toolkits) generate known, benign reports. Suppress
them rather than ignore the noise.

```bash
# Generate suppression stubs from a clean run
valgrind --gen-suppressions=all --log-file=vg.log ./prog
# Reuse a suppression file
valgrind --suppressions=mylib.supp ./prog
```

## GDB integration (vgdb)

```bash
# Terminal 1: stop at the first error and wait for a debugger
valgrind --vgdb=yes --vgdb-error=0 ./prog
# Terminal 2:
gdb ./prog
(gdb) target remote | vgdb            # now inspect state at the error
(gdb) monitor leak_check              # run memcheck commands from GDB
```

## Pitfalls

```
- Speed: 10–50× slower; not for production or tight timing loops. Reduce the workload.
- Build with -g; avoid heavy -O2 (inlining/optimised-out vars muddy reports).
- "still reachable" is usually fine (e.g. global caches) — don't chase every one.
- Custom allocators/memory pools hide errors from memcheck unless annotated
  (VALGRIND_MALLOCLIKE_BLOCK) — wrap them or errors look like leaks.
- Memcheck and a sanitizer (ASan) at the same time conflict — pick one.
- helgrind has false positives with lock-free/atomic code; annotate or prefer TSan.
- It catches errors only on paths actually executed — coverage still matters.
- JIT'd / self-modifying code and some SIMD instructions may be unsupported.
```

## Where this connects

- [Sanitizers](sanitizers.md) — faster compile-time ASan/TSan/UBSan; the modern default
- [perf & Flame Graphs](perf_profiling.md) — real-hardware profiling vs simulated
- [GDB](gdb.md) — attach via `vgdb` to inspect program state at an error
- [Core Dump Analysis](core_dump.md) — post-mortem when you can't run under Valgrind
- [Memory Management](../programming/memory_management.md) — heap/stack model the errors map to
- [Concurrency](../programming/concurrency.md) — the races helgrind/DRD detect
- [Test Debugging](../testing/debugging.md) — running Valgrind in CI for memory regressions
