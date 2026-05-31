# Property-Based Testing

## Overview

Example-based tests check specific inputs you thought of. **Property-based
testing** flips this: you describe a *property* (an invariant that should hold
for **all** valid inputs), and the framework generates hundreds of random
inputs trying to break it.

```python
# Example-based: one case you picked
def test_reverse():
    assert reverse([1, 2, 3]) == [3, 2, 1]

# Property-based: a rule that must hold for ANY list
@given(st.lists(st.integers()))
def test_reverse_twice_is_identity(xs):
    assert reverse(reverse(xs)) == xs
```

The payoff: the framework finds the edge cases you forgot — empty inputs, huge
values, duplicates, Unicode, `NaN`, off-by-one boundaries.

## Core Idea: Properties

A property is a statement true for all inputs. You stop thinking about
individual cases and start thinking about *invariants*.

Common property patterns:

- **Round-trip**: `decode(encode(x)) == x` (serialization, parsing, compression)
- **Invariant**: `len(sorted(xs)) == len(xs)`; output always within bounds
- **Idempotence**: `f(f(x)) == f(x)` (e.g. `normalize`, `dedupe`)
- **Commutativity / associativity**: `add(a, b) == add(b, a)`
- **Oracle / model-based**: compare a fast implementation against a slow,
  obviously-correct reference

## Hypothesis (Python)

```python
from hypothesis import given, strategies as st, example, settings

@given(st.integers(), st.integers())
def test_addition_commutes(a, b):
    assert add(a, b) == add(b, a)

# Strategies build inputs:
st.integers(min_value=0, max_value=100)
st.text()
st.lists(st.floats(allow_nan=False), min_size=1)
st.dictionaries(st.text(), st.integers())
st.sampled_from(["GET", "POST", "PUT"])

# Compose / constrain
@given(st.lists(st.integers()).filter(lambda xs: len(xs) > 0))
def test_max_is_in_list(xs):
    assert max(xs) in xs

# Pin a known tricky case alongside generated ones
@given(st.integers())
@example(0)
def test_handles_zero(n):
    assert process(n) is not None
```

Hypothesis remembers previously-found failures in a local database and replays
them on the next run, so a flaky edge case won't slip away.

## fast-check (JavaScript / TypeScript)

```javascript
import fc from 'fast-check';

test('reverse twice is identity', () => {
  fc.assert(
    fc.property(fc.array(fc.integer()), (xs) => {
      expect(reverse(reverse(xs))).toEqual(xs);
    })
  );
});

// Arbitraries (input generators)
fc.integer({ min: 0, max: 100 })
fc.string()
fc.record({ name: fc.string(), age: fc.nat() })
fc.constantFrom('GET', 'POST', 'PUT')
```

## Shrinking

When a property fails, the framework **shrinks** the failing input to the
smallest/simplest case that still fails — so instead of a 500-element list with
huge numbers, you get reported `[0]` or `[1, 0]`. This is what makes the
failures actionable.

```
Falsifying example: test_sum_positive(
    xs=[0],   # shrunk from a large random list
)
```

## When to Use (and Not)

✅ Pure functions, parsers/serializers, data transforms, math, stateful models
✅ Code with clear invariants or a reference implementation
✅ Finding edge cases in "I think I covered everything" code

❌ Code dominated by side effects / external I/O (hard to state properties)
❌ When you can't articulate a property — a vague test is worse than a clear example
❌ As a *replacement* for example tests; use both — examples document intent,
   properties hunt for edge cases

## Best Practices

1. **Start with one strong property** (often round-trip) rather than many weak ones.
2. **Keep generated data realistic** — constrain strategies to valid domains.
3. **Make tests deterministic** to debug: Hypothesis replays failures from its DB;
   in CI, capture the seed.
4. **Don't reimplement the code in the test** — assert invariants, not the algorithm.
5. **Combine with example tests** (see [Unit Testing](unit_testing.md)) for
   documentation and regression pins.

## ELI10

Example tests are like checking a few specific doors are locked. Property-based
testing is like hiring a robot to rattle *every* door, all night, and then hand
you the exact one that was open — described as simply as possible.

## Further Resources

- [Hypothesis docs](https://hypothesis.readthedocs.io/)
- [fast-check docs](https://fast-check.dev/)
- [Choosing properties for property-based testing - F# for Fun and Profit](https://fsharpforfunandprofit.com/posts/property-based-testing-2/)
