# CI/CD Test Automation

## Overview

Tests deliver the most value when they run **automatically** on every change.
CI/CD test automation is about making the suite **fast, reliable, and
informative** in the pipeline so it gates bad code without slowing the team
down. The two enemies are *slowness* (people stop waiting for it) and
*flakiness* (people stop trusting it).

## Pipeline Stages & Gating

Run cheap, broad checks first; fail fast before spending time on expensive ones:

```
lint + type-check  →  unit tests  →  integration tests  →  e2e tests  →  deploy
   (seconds)           (fast)          (medium)             (slow)
```

Each stage **gates** the next. Add gates for [coverage](coverage.md) thresholds
and [security](security_testing.md) scans. Block the merge when a required check
fails.

## Parallelization & Sharding

The single biggest speedup for large suites:

- **Parallel jobs**: run unit, integration, and e2e suites as separate
  concurrent jobs.
- **Sharding**: split one suite across N machines.

```bash
# pytest across workers (pytest-xdist)
pytest -n auto

# Jest sharding across CI machines
jest --shard=1/4      # machine 1 of 4

# Playwright sharding (see e2e_testing.md for the full e2e setup)
npx playwright test --shard=1/4
```

## Caching

Avoid reinstalling/recomputing on every run:

- Cache **dependencies** (`~/.cache/pip`, `node_modules`/npm cache) keyed on the
  lockfile hash.
- Cache **build artifacts** and tool caches where supported.
- Invalidate correctly — a stale cache causes confusing failures.

## Matrix Builds

Test across versions/OSes in one config:

```yaml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    os: [ubuntu-latest, macos-latest]
```

## Coverage Gating & Artifacts

Publish coverage reports and JUnit XML as artifacts so failures and trends are
visible. Optionally fail the build below a threshold (see [Coverage](coverage.md)) —
but prefer gating on coverage of the **diff** rather than the whole repo.

## Flaky-Test Handling

Flaky tests erode trust in the whole suite. Tactics:

- **Detect**: track tests that pass on retry; flag repeat offenders.
- **Retry sparingly**: auto-retry only known-flaky tests, never blanket-retry
  (it hides real bugs).
- **Quarantine**: move chronic offenders to a non-gating job and file a ticket
  to fix or delete them.
- **Fix the root cause**: most flakiness is timing/ordering/shared-state — see
  [Debugging](debugging.md) and [Test Smells](test_smells.md).

## Test Selection

On large repos, run only what's affected:

- Run tests for **changed files / modules** on PRs; run the full suite on merge
  to main and nightly.
- Tools: `pytest --picked`, Jest `--onlyChanged`, build-graph–aware runners
  (Nx, Bazel, Turborepo).

## Example: GitHub Actions

```yaml
name: tests
on: [push, pull_request]

jobs:
  python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip                       # dependency caching
      - run: pip install -r requirements.txt
      - run: pytest -n auto --cov=myapp --cov-report=xml --junitxml=junit.xml
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: junit-${{ matrix.python-version }}, path: junit.xml }

  js:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm test -- --coverage --shard=1/2
```

> The [E2E Testing](e2e_testing.md) doc has a dedicated Playwright CI workflow —
> reference that for browser-test pipelines rather than duplicating it here.

## Best Practices

1. **Fail fast**: cheap checks before expensive ones.
2. **Keep PR feedback under ~10 minutes** via parallelism, sharding, and caching.
3. **Make it deterministic** — pin versions, seed data, isolate state.
4. **Treat flaky as broken**: detect, quarantine, fix; never blanket-retry.
5. **Gate on what matters**: required checks, diff coverage, high-severity
   security findings.
6. **Surface results**: artifacts, JUnit reports, and clear status checks.

## ELI10

It's an assembly line with quality gates: a robot checks each part the moment
it's made. Quick checks come first so a broken part gets caught fast, several
robots work side by side to keep the line moving, and a part can't move forward
until it passes its gate.

## Further Resources

- [GitHub Actions docs](https://docs.github.com/en/actions)
- [pytest-xdist](https://pytest-xdist.readthedocs.io/) · [Jest CI guide](https://jestjs.io/docs/cli)
- [Google Testing Blog: Flaky Tests](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)

## Where this connects

- [TDD](tdd.md) — the fast tests that gate every commit
- [Coverage](coverage.md), [Mutation testing](mutation_testing.md) — quality gates in the pipeline
- [E2E testing](e2e_testing.md), [Performance testing](performance_testing.md) — slower pipeline stages
- [Test data](test_data.md) — provisioning data in CI
- [Security testing](security_testing.md) — SAST/DAST stages
- [Code quality](code_quality.md) — linting and static-analysis gates
