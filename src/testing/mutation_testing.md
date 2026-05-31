# Mutation Testing

## Overview

High code [coverage](coverage.md) tells you which lines *ran* during tests — not
whether your tests would *catch a bug* in those lines. A test suite can hit 100%
coverage while asserting almost nothing.

**Mutation testing** measures test *quality* directly: it introduces small bugs
("mutants") into your code and checks whether your tests fail. If a mutant
survives (tests still pass), you have a gap.

## How It Works

1. Make a small change to the source — a **mutant** (e.g. `+` → `-`, `>` → `>=`,
   `return x` → `return None`, delete a line).
2. Run the test suite against the mutated code.
3. Classify the result:
   - **Killed** — a test failed. Good: your tests detected the bug.
   - **Survived** — all tests passed. Bad: a real bug here would slip through.
4. Repeat for many mutants across the codebase.

```python
# Original
def is_adult(age):
    return age >= 18

# Mutant 1: >= becomes >    → is_adult(18) now False
# Mutant 2: 18 becomes 19   → boundary shifted
# Mutant 3: return age > 0  → logic changed
```

If no test asserts the exact boundary (`age == 18`), mutants 1 and 2 survive —
revealing a missing edge-case test.

## Mutation Score

```
mutation score = killed mutants / (total mutants − equivalent mutants)
```

Higher is better. Unlike coverage %, this reflects whether tests have real
assertions. A 90% line coverage with a 40% mutation score means lots of code
runs untested in any meaningful sense.

## Tools

```bash
# Python — mutmut
pip install mutmut
mutmut run                     # generate & test mutants
mutmut results                 # list survivors
mutmut show <id>               # see the surviving mutant's diff

# Python — cosmic-ray (config-driven, distributed)
cosmic-ray init config.toml session.sqlite
cosmic-ray exec config.toml session.sqlite

# JavaScript / TypeScript — Stryker
npm install --save-dev @stryker-mutator/core
npx stryker run                # produces an HTML report with the mutation score

# Java — Pitest (Maven)
mvn org.pitest:pitest-maven:mutationCoverage
```

## Equivalent Mutants

Some mutants change the code but **not its behavior** (e.g. `x < 10` → `x <= 9`
for integers). These can never be killed and must be excluded manually,
otherwise they drag the score down. Detecting them is undecidable in general —
review survivors and mark the equivalent ones as ignored.

## Cost & When to Run

Mutation testing is **slow** — it runs the suite once per mutant. Strategies:

- Run on **changed files only** (incremental mode in Stryker/mutmut) on PRs.
- Run the **full suite nightly** or weekly in CI, not on every commit.
- Scope to **critical modules** (auth, billing, parsing) rather than everything.
- Use a fast test subset and good test parallelism.

## Best Practices

1. **Use it to find weak tests, not as a gate** initially — chase survivors,
   not a target number.
2. **Pair with coverage**: coverage finds unrun code; mutation finds unverified
   behavior. See [Coverage](coverage.md).
3. **Fix survivors by adding assertions**, especially around boundaries and
   error paths.
4. **Exclude generated code, logging, and equivalent mutants** to keep the
   signal clean.
5. **Run incrementally in CI** to keep feedback fast.

## ELI10

Coverage checks that you *walked into* every room. Mutation testing sneaks in
and breaks something in each room, then checks whether your alarm goes off. If
the alarm stays silent, that room isn't really protected.

## Further Resources

- [mutmut docs](https://mutmut.readthedocs.io/)
- [Stryker Mutator](https://stryker-mutator.io/)
- [Pitest](https://pitest.org/)
