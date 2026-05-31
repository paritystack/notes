# Programming Paradigms

## Overview

A programming paradigm is a style of structuring programs — a set of concepts and
constraints that shape how you express computation. Most languages in this section are
*multi-paradigm*: [Python](python.md), [JavaScript](javascript.md), [Rust](rust.md),
[Kotlin](kotlin.md), [C++](cpp.md), [Go](go.md) and [Java](java.md) all blend several. This
page is the hub that ties together [functional programming](functional_programming.md),
[object-oriented concepts](oop_concepts.md), [type systems](type_systems.md), and
[pattern matching](pattern_matching.md).

## The major families

```
IMPERATIVE  ─────────────────────────  DECLARATIVE
"how": step-by-step state changes      "what": describe the result, not the steps

 ├─ Procedural        (C, Go)           ├─ Functional   (Haskell, FP in Rust/JS)
 └─ Object-oriented   (Java, C++)       ├─ Logic        (Prolog, SQL constraints)
                                        └─ Dataflow / reactive (spreadsheets, RxJS)
```

- **Imperative** — the program is a sequence of statements that mutate state. Procedural
  groups them into procedures; [OOP](oop_concepts.md) groups state and behaviour into objects.
- **Declarative** — you describe the desired result and the runtime figures out execution.
  [Functional](functional_programming.md) (expressions over immutable data), logic
  (facts + rules), and query/dataflow ([SQL](sql.md), reactive streams) are the main members.

## Imperative vs declarative, concretely

```python
# Imperative: tell the machine each step
result = []
for x in nums:
    if x % 2 == 0:
        result.append(x * x)

# Declarative: describe the transformation
result = [x * x for x in nums if x % 2 == 0]
```

Both compute the same thing; the declarative form omits the mechanics (the accumulator, the
loop index) and is usually easier to reason about and parallelise.

## Why multi-paradigm wins

Real codebases mix paradigms because each fits different problems: OOP for modelling
stateful entities and boundaries, functional for transformations and concurrency-safe logic,
declarative queries for data access. The skill is choosing the right one per layer rather
than forcing everything into a single style.

## Where this connects

- [Functional programming](functional_programming.md) and [OOP concepts](oop_concepts.md) —
  the two paradigms you'll combine most.
- [Type systems](type_systems.md) — sum types lean functional, classes lean OOP.
- [Design patterns](design_patterns.md) — many patterns are OOP workarounds for things FP
  does natively (and vice versa).
- [SQL](sql.md) — a mainstream declarative language.

## Pitfalls

- **Paradigm dogma.** Insisting everything be "pure OOP" or "pure FP" produces awkward code;
  match the paradigm to the problem.
- **Hidden imperative state in declarative code.** Side effects inside a comprehension/stream
  break the declarative reasoning you were buying.
- **Paradigm-shaped over-engineering.** Deep class hierarchies or monad stacks where a plain
  function or struct would do.
