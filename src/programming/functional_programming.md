# Functional Programming

## Overview

Functional programming (FP) is a paradigm that treats computation as the evaluation of
functions over immutable data, avoiding shared mutable state and side effects. It is one of
the major [programming paradigms](paradigms.md) alongside imperative and
[object-oriented](oop_concepts.md) styles, and most languages in this section now support a
functional style to some degree — [Rust](rust.md), [Kotlin](kotlin.md),
[JavaScript](javascript.md)/[TypeScript](typescript.md), [Python](python.md), and even
[Go](go.md) and [Java](java.md) via lambdas and streams. FP leans heavily on the
[type system](type_systems.md) (especially sum types) and on
[pattern matching](pattern_matching.md).

## Core ideas

- **Pure functions** — output depends only on inputs, with no observable side effects. Same
  input → same output, always. Pure functions are trivially testable, cacheable
  (memoizable), and safe to run concurrently (see [concurrency](concurrency.md)).
- **Immutability** — data isn't mutated in place; transformations produce new values. This
  removes whole categories of aliasing and data-race bugs.
- **First-class & higher-order functions** — functions are values: passed as arguments,
  returned, stored. A higher-order function takes or returns functions (`map`, `filter`,
  `reduce`).
- **Referential transparency** — any expression can be replaced by its value without
  changing the program. This is what makes equational reasoning possible.

```python
# Imperative: mutate accumulator
total = 0
for x in nums:
    total += x * 2

# Functional: compose pure transformations
total = sum(map(lambda x: x * 2, nums))
```

## Closures and currying

A **closure** is a function bundled with the variables it captures from its defining scope.
Closures are how FP carries state around without mutation.

```javascript
const adder = (n) => (x) => x + n;   // returns a closure over n
const add5 = adder(5);
add5(10); // 15
```

**Currying** turns an n-argument function into a chain of one-argument functions, enabling
**partial application** — fixing some arguments now and the rest later. This is the basis of
point-free, highly composable code.

## Composition over control flow

FP builds programs by composing small functions rather than sequencing statements.

```
data ──▶ filter ──▶ map ──▶ reduce ──▶ result
        (pure)     (pure)   (pure)

f ∘ g  means  "apply g, then f"   →   compose(f, g)(x) === f(g(x))
```

Pipelines (`|>` in Elixir/F#, method chaining on iterators) read as a data-flow description
rather than step-by-step mutation. See [iterators & generators](iterators_generators.md)
for the lazy version of this.

## Handling absence and effects

Pure code can't just `throw` or return `null`, so FP encodes those in the type system:

- `Option`/`Maybe` for "maybe absent", `Result`/`Either` for "maybe failed" — see
  [error handling](error_handling.md).
- **Monads** generalise "a value in a context" plus a way to chain context-aware steps
  (`flatMap`/`bind`). `Option`, `Result`, lists, futures and `Promise.then` are all monadic.
  The practical takeaway: a monad lets you sequence computations that carry extra structure
  (failure, async, multiplicity) without manually unwrapping at each step.

## Where this connects

- [Paradigms](paradigms.md) — FP vs imperative/OOP, and where they mix.
- [Pattern matching](pattern_matching.md) and [type systems](type_systems.md) — sum types
  and exhaustive matching are FP's bread and butter.
- [Error handling](error_handling.md) — `Result`/`Option` instead of exceptions.
- [Concurrency](concurrency.md) — immutability removes data races; pure functions
  parallelise freely.
- [Iterators & generators](iterators_generators.md) — lazy functional pipelines.

## Pitfalls

- **Immutability has a cost.** Naive copy-on-write allocates heavily; rely on *persistent
  data structures* (structural sharing) or builder patterns in hot paths.
- **Deep recursion without tail calls** blows the stack in languages (Python, most JVM
  setups) that don't optimise tail calls — prefer iteration or explicit accumulators there.
- **Monad/abstraction overload.** Wrapping everything in clever combinators can hurt
  readability for teams unfamiliar with FP; reach for it where it pays.
- **"Pure" that isn't.** Hidden I/O, logging, clock, or RNG inside a "pure" function breaks
  referential transparency and the guarantees that follow.
- **Eager vs lazy confusion.** Lazy pipelines defer work; forgetting to force them (or
  forcing them twice) causes surprising performance or re-computation.
