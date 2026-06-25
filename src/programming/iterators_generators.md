# Iterators & Generators

## Overview

Iterators provide a uniform way to walk a sequence of values one at a time, decoupling
*how you traverse* from *what you traverse*. Generators are functions that *produce* such
sequences lazily by suspending and resuming. Together they enable composable, memory-frugal
data pipelines and are the backbone of looping in [Python](python.md), [Rust](rust.md),
[JavaScript](javascript.md), [Kotlin](kotlin.md), [Java](java.md) (streams) and
[Go](go.md) (range-over-func, 1.23+). They are the lazy, streaming face of
[functional programming](functional_programming.md) and close cousins of the coroutines in
[async programming](async_programming.md).

## The iterator protocol

An iterator exposes a single operation: "give me the next value, or signal that you're done."

```
        ┌─────────────┐  next()   ┌──────────┐
  for ──▶│  iterator   │──────────▶│  value   │
        │  (has state)│◀────done──┤ or end   │
        └─────────────┘           └──────────┘

Python: __iter__/__next__ + StopIteration   Rust: Iterator::next() -> Option<T>
JS:     [Symbol.iterator] + {value, done}    Java: hasNext()/next()
```

Because everything reduces to `next()`, the same `for` loop and the same combinators work
over arrays, files, ranges, trees, or infinite sequences.

## Generators and lazy evaluation

A generator function uses `yield` to emit a value and pause, resuming where it left off on
the next request. This makes **laziness** natural: values are computed on demand, so you can
model infinite or expensive streams without materialising them.

```python
def naturals():        # infinite, but lazy
    n = 0
    while True:
        yield n
        n += 1

from itertools import islice
list(islice(naturals(), 5))   # [0, 1, 2, 3, 4] — only 5 computed
```

A generator is essentially a coroutine specialised for producing values (see
[async programming](async_programming.md) for the general case, and async generators that
combine both).

## Lazy pipelines

Combinators (`map`, `filter`, `take`, `flat_map`) chain into pipelines that process one
element end-to-end before fetching the next — constant memory, early termination, no
intermediate collections.

```rust
let first_three_squares: Vec<i32> = (1..)        // infinite range
    .map(|x| x * x)
    .filter(|x| x % 2 == 1)
    .take(3)
    .collect();   // [1, 9, 25] — pull-driven, stops after 3
```

```
(1..) ─▶ map ─▶ filter ─▶ take(3) ─▶ collect
   one element flows all the way through, then the next  (pull / demand-driven)
```

Contrast with **eager** evaluation (building a full list at each step), which uses more
memory and can't handle infinite sources.

## Where this connects

- [Functional programming](functional_programming.md) — `map`/`filter`/`reduce` over iterators.
- [Async programming](async_programming.md) — generators are coroutines; async iterators stream
  awaited values.
- [Type systems](type_systems.md) — the iterator trait/interface, generic over element type.
- [Python](python.md) / [Rust](rust.md) / [JavaScript](javascript.md) for concrete protocols.
- [Closures & lexical scope](closures.md) — generator state lives in a captured frame resumed
  across yields.

## Pitfalls

- **Single-use exhaustion.** Most iterators/generators can be consumed once; iterating again
  yields nothing. Re-create or collect if you need multiple passes.
- **Hidden eager work.** A `sorted()`/`collect()` in the middle forces the whole stream into
  memory, defeating laziness.
- **Infinite sources without a bound.** Forgetting `take`/`break` on an infinite generator
  hangs forever.
- **Mutating while iterating.** Modifying the underlying collection during iteration causes
  `ConcurrentModificationException` / undefined behaviour.
- **Re-computation in lazy chains.** Re-traversing a lazy pipeline reruns all the work; cache
  if it's expensive.
