# Debugging Methodology

## Overview

Tools find bugs faster, but *method* is what makes debugging reliable instead of
lucky. This page is the technique layer above the tooling pages — [GDB](gdb.md),
[LLDB](lldb.md), [Valgrind](valgrind.md), [Sanitizers](sanitizers.md),
[perf](perf_profiling.md), [rr](rr_debugging.md) — describing how to reason your way
from "it's broken" to a verified fix. The core idea is to treat debugging as the
scientific method: form a hypothesis, design an experiment that can falsify it, and
narrow the search space systematically instead of guessing.

```
The loop:
  1. Reproduce        → a reliable, minimal trigger
  2. Observe          → gather facts (logs, state, traces) — don't assume
  3. Hypothesize      → a falsifiable theory of the cause
  4. Experiment       → one change that confirms or refutes it
  5. Narrow / fix     → bisect toward the cause, then fix
  6. Verify           → prove the fix; add a regression test
```

## 1. Reproduce reliably

A bug you can't trigger on demand is one you can't confirm fixed. Pin down the inputs,
environment, timing, and state that produce it. For rare/non-deterministic bugs, capture
the execution with [rr](rr_debugging.md) so every replay is identical, or stress
scheduling (`rr --chaos`, [TSan](sanitizers.md)) to make races reproducible.

## 2. Reduce the repro (delta debugging)

Shrink the failing case to the smallest input that still fails — smaller repros expose
the cause directly and make great regression tests.

```
Delta debugging (automate the shrink):
  given a large failing input, repeatedly:
    remove a chunk → still fails?  keep removing
                   → now passes?   put it back, try a smaller chunk
  → converges on a 1-minimal failing input

Tools: C-Reduce / cvise (C/C++ source), creduce, or a hand-rolled bisection
       on test data; for compilers/fuzzers this is the standard first step.
```

## 3. Bisect the search space

Halve the unknown each step instead of scanning linearly.

```
Dimensions you can bisect:
  in TIME      → git bisect (which commit introduced it)
  in CODE      → comment out / short-circuit halves (binary search by hand)
  in DATA      → delta debugging (above)
  in EXECUTION → rr reverse-continue + watchpoint (which write corrupted state)
  in CONFIG    → toggle features/flags by halves
```

```bash
git bisect start
git bisect bad                 # current commit is broken
git bisect good v1.2.0         # this old tag worked
# git checks out the midpoint; test and mark each:
git bisect good   # or: git bisect bad
# ... log2(N) steps later git names the first bad commit
git bisect run ./test.sh       # fully automated if you have a pass/fail script
git bisect reset               # return to where you started
```

`git bisect run` with a script that exits 0 (good) / non-zero (bad) automates the whole
thing — see [Git Commands](../git/commands.md).

## 4. Make the invisible visible

```
- Logging: structured, leveled; log the inputs and decisions, not just "got here".
- Assertions: encode invariants; a failed assert localises the bug to the line.
- Tracing: strace/ltrace (tools.md), perf, eBPF for what the program actually did.
- Diffing: compare a working run vs a failing run (env, inputs, versions, logs).
- Print debugging is legitimate — fast, no setup; just remove it after.
```

## 5. Reason, don't guess

```
- Question your assumptions: the bug is usually where you're SURE it can't be.
- Change one thing at a time — multiple simultaneous changes hide which mattered.
- Rubber-duck: explain the code line by line aloud; the gap reveals itself.
- Follow the data, not the code path: where did the bad value come from? (rr excels here)
- "It can't be the compiler/library/OS" — it almost never is; suspect your code first.
- Read the error message completely, including the stack trace and the second sentence.
```

## 6. Verify and lock it in

A fix isn't done until you've (a) confirmed the bug is gone *and* (b) confirmed you know
*why* the change fixed it — a fix you don't understand may just hide the symptom. Add a
regression test (ideally your minimal repro from step 2) so it can't silently return —
see [Test Debugging](../testing/debugging.md).

## Pitfalls

```
- Fixing the symptom, not the cause — the bug resurfaces elsewhere later.
- "Shotgun debugging": changing many things hoping one helps; you lose the causal thread.
- Trusting your mental model over observation — measure, don't assume.
- Non-reproducible "fixes": if you can't reproduce it, you can't prove you fixed it.
- Skipping the regression test — the same bug returns in six months.
- Debugging the optimised build's symptoms while reasoning about the source (inlining lies).
- Heisenbugs: adding a print/debugger changes timing and hides the race — use rr/TSan instead.
```

## Where this connects

- [GDB](gdb.md) / [LLDB](lldb.md) — interactive hypothesis testing
- [rr (Record & Replay)](rr_debugging.md) — bisection in time; deterministic reproduction
- [Valgrind](valgrind.md) / [Sanitizers](sanitizers.md) — make memory/race bugs reproducible and located
- [perf & Flame Graphs](perf_profiling.md) — the same method applied to performance bugs
- [Binary Analysis Tools](tools.md) — strace/ltrace/objdump for observation
- [Git Commands](../git/commands.md) — `git bisect` for time-bisection
- [Test Debugging](../testing/debugging.md) — turning repros into regression tests
