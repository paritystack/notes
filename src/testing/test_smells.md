# Test Smells

## Overview

Test code is real code — it has to be maintained. **Test smells** are recurring
patterns that make a suite slow, fragile, or untrustworthy. They don't
necessarily fail today, but they make tests painful to read, prone to false
alarms, or likely to hide real bugs. Spotting and refactoring them keeps the
suite an asset rather than a liability.

This complements the TDD [anti-patterns](tdd.md) (process smells) and
[Debugging](debugging.md) (diagnosing failures); here the focus is the *quality
of the test code itself*.

## Catalogue

### Fragile / brittle tests
Break on unrelated changes — typically because they assert on implementation
details (private internals, exact HTML, call order) instead of behavior.
**Fix:** assert observable outcomes; test through the public interface.

### Eager test
One test exercises many behaviors, so a failure doesn't tell you *what* broke.
**Fix:** one behavior per test; split into focused cases with clear names.

### Mystery guest
The test depends on external data not visible in the test — a shared fixture
file, a magic DB row, a global.
**Fix:** make inputs explicit with factories/builders (see
[Test Data Management](test_data.md)).

### Test interdependence / order coupling
Tests pass only in a certain order because they share mutable state; reorder or
run in isolation and they fail.
**Fix:** each test sets up and tears down its own state; no shared globals. See
isolation in [Test Data Management](test_data.md).

### Assertion roulette
Many bare assertions with no messages; when one fails you can't tell which.
**Fix:** fewer, well-named assertions; add messages or use descriptive
matchers.

### Slow tests
Hit real network/DB/sleep, so the suite is too slow to run often.
**Fix:** [mock](mocking.md) external boundaries; replace `sleep` with fake
timers/clocks; reserve real I/O for [integration](integration.md) tests.

### Conditional logic in tests
`if`/`for`/`try` inside a test means it might silently skip its assertions, and
the test itself can be buggy.
**Fix:** straight-line Arrange-Act-Assert; use parametrization for variations.

### Test code duplication
The same setup copy-pasted everywhere; a change means editing dozens of tests.
**Fix:** extract fixtures, factories, and helpers — but keep tests readable
(don't over-abstract into a maze).

### Over-mocking
Mocking so much that the test only verifies the mocks, not real behavior; it
passes even when the integration is broken.
**Fix:** mock only true boundaries (network, time, randomness); prefer real
objects for in-process collaborators ([classicist](tdd.md) style). See
[Contract Testing](contract_testing.md) for verifying boundaries you mock.

### Flaky tests
Pass sometimes, fail sometimes — usually timing, ordering, or shared state.
**Fix:** remove nondeterminism (seed randomness, freeze time, await properly);
quarantine chronic offenders in [CI](ci_testing.md).

### Obscure / unclear names
`test1`, `test_user` tell you nothing when they fail.
**Fix:** name the scenario and expectation, e.g.
`test_withdraw_more_than_balance_raises`.

### Testing the framework / trivial code
Tests for getters/setters or library behavior add noise without value.
**Fix:** focus on your logic, edge cases, and error paths.

## Refactoring Tests Safely

Tests have no tests of their own, so change them carefully:

1. **Verify before refactoring** — confirm the test currently *fails* when the
   code is wrong (mutate the code briefly, or recall it once failed for real).
2. **Refactor structure, not assertions** in one step; keep behavior identical.
3. **Run the suite after each change** — green throughout.
4. Extract helpers/fixtures incrementally; don't hide the test's intent behind
   too much indirection.

## Best Practices

1. **Tests should read like documentation** — Arrange-Act-Assert, clear names.
2. **One reason to fail** per test.
3. **Independent and deterministic** — order- and environment-agnostic.
4. **Mock boundaries, not everything.**
5. **DRY setup, but keep each test's intent obvious** — readability beats reuse.
6. **Treat flaky tests as bugs**, not noise to retry away.

## ELI10

A test is a smoke alarm. A good one is silent until there's real smoke and then
tells you exactly which room. A smelly one beeps when you make toast (fragile),
beeps with no light to show where (assertion roulette), or has a dead battery
that never beeps at all (asserts nothing) — and you stop trusting it.

## Further Resources

- [xUnit Test Patterns - Gerard Meszaros (test smells catalogue)](http://xunitpatterns.com/Test%20Smells.html)
- [Martin Fowler: Test Smells & Refactoring](https://martinfowler.com/)

## Where this connects

- [TDD](tdd.md) — the discipline that keeps tests clean
- [Unit testing](unit_testing.md) — where most smells appear
- [Mocking](mocking.md) — over-mocking as a common smell
- [Coverage](coverage.md) — coverage gaming versus real assertions
- [Code quality](code_quality.md) — the production-code parallel
- [Mutation testing](mutation_testing.md) — detecting assertion-free tests
