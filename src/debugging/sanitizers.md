# Sanitizers

## Overview

Sanitizers are compiler-instrumented runtime bug detectors built into GCC and Clang.
You add a `-fsanitize=` flag, recompile, and the compiler injects checks (shadow
memory, redzones, instrumented loads/stores) that catch bugs the moment they happen,
with a precise stack trace. They are the modern, fast alternative to
[Valgrind](valgrind.md) — typically 2–4× slowdown instead of 10–50× — at the cost of
needing a rebuild. They pair naturally with [GDB](gdb.md) (break on the report) and
belong in CI alongside the test suite ([Test Debugging](../testing/debugging.md)).

```
The major sanitizers:
  ASan   AddressSanitizer        → out-of-bounds, use-after-free, double-free
  LSan   LeakSanitizer           → memory leaks (bundled with ASan)
  UBSan  UndefinedBehaviorSan.   → signed overflow, bad shifts, misaligned ptrs, ...
  TSan   ThreadSanitizer         → data races, lock-order inversions
  MSan   MemorySanitizer         → use of uninitialised memory (Clang only)
```

## AddressSanitizer (ASan)

Catches spatial (out-of-bounds) and temporal (use-after-free) memory errors using
"redzones" around allocations and a quarantine for freed memory.

```bash
gcc -g -fsanitize=address -fno-omit-frame-pointer prog.c -o prog
./prog                          # aborts with a detailed report on first error
```

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
WRITE of size 4 at 0x602000000014 thread T0
    #0 0x... in store_value prog.c:17
    #1 0x... in main prog.c:30
0x602000000014 is located 0 bytes to the right of 20-byte region
  allocated by thread T0 here:
    #0 0x... in malloc
    #1 0x... in main prog.c:28
```

```bash
# Runtime options via env var
ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:abort_on_error=1 ./prog
ASAN_OPTIONS=detect_stack_use_after_return=1 ./prog
```

LeakSanitizer runs at exit when bundled with ASan (`detect_leaks=1`, default on Linux).

## UndefinedBehaviorSanitizer (UBSan)

Flags C/C++ undefined behaviour — the bugs that "work until they don't" under
optimisation (see [Compilers](../programming/compilers.md)).

```bash
gcc -g -fsanitize=undefined prog.c -o prog
# Make UB a hard failure with a trace:
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./prog

# Useful sub-checks (compose them):
-fsanitize=signed-integer-overflow,shift,null,bounds,alignment,return
```

## ThreadSanitizer (TSan)

Detects data races and lock-order inversions in threaded code
([Concurrency](../programming/concurrency.md)). Lower false positives and clearer
output than Valgrind's helgrind, but ~5–15× slower and higher memory use.

```bash
gcc -g -fsanitize=thread prog.c -o prog
./prog
```

```
WARNING: ThreadSanitizer: data race (pid=123)
  Write of size 4 by thread T2 at 0x... :
    #0 increment counter.c:8
  Previous read of size 4 by thread T1 at 0x... :
    #0 read_counter counter.c:14
  Location is global 'counter' of size 4
```

## MemorySanitizer (MSan)

Tracks uninitialised reads (Clang only). Requires *all* code — including libc++ — to
be instrumented to avoid false positives, so it's the hardest to deploy.

```bash
clang -g -fsanitize=memory -fno-omit-frame-pointer prog.c -o prog
```

## Compatibility & combining

```
Can combine in one build:           Mutually exclusive (separate builds):
  ASan + UBSan + LSan  ✓               ASan  ✗ TSan
  TSan + UBSan         ✓               ASan  ✗ MSan
                                       TSan  ✗ MSan
```

Always build with `-g` (line info) and `-fno-omit-frame-pointer` (readable stacks).
A typical CI matrix runs the test suite three times: ASan+UBSan, TSan, and a plain
optimised build.

## Sanitizers vs Valgrind

```
                     Sanitizers              Valgrind (memcheck)
  Needs recompile     yes                     no (runs any binary)
  Slowdown            2–4× (ASan)             10–50×
  Uninit. memory      MSan (Clang only)       yes (memcheck, easy)
  Use-after-return    yes (ASan opt-in)       no
  Coverage of libs    only instrumented code  all code, including 3rd-party
  Best for            CI on your own code     binaries you can't rebuild
```

## Pitfalls

```
- Sanitizer builds are for testing, NOT production (ASan ~2× memory; all are attack surface).
- Don't trust an un-instrumented dependency under MSan — you'll get false positives.
- ASan and the kernel/embedded targets: use KASAN in the kernel, not userspace ASan.
- Optimisation can elide the very UB you're hunting; keep -O1 and compile the suspect TU at -O0.
- detect_leaks needs a clean exit path; long-running daemons report "leaks" that are caches.
- Symbolized stacks need llvm-symbolizer/addr2line on PATH, else you get raw addresses.
- A green sanitizer run only covers executed paths — pair with good test coverage.
```

## Where this connects

- [Valgrind](valgrind.md) — no-recompile alternative; better for uninitialised-memory and 3rd-party code
- [ptrace & debugger internals](ptrace_internals.md) — the external-tracer approach sanitizers replace with compile-time instrumentation
- [GDB](gdb.md) — `break __asan_report_error` / `ASAN_OPTIONS=abort_on_error=1` to stop in the debugger
- [perf & Flame Graphs](perf_profiling.md) — performance once correctness is clean
- [Concurrency](../programming/concurrency.md) — the races TSan finds
- [Memory Management](../programming/memory_management.md) — heap/stack model behind ASan redzones
- [Compilers](../programming/compilers.md) — the UB UBSan formalises
- [Test Debugging](../testing/debugging.md) — wiring sanitizers into the CI matrix
