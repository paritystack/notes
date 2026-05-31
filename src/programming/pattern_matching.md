# Pattern Matching

## Overview

Pattern matching tests a value against a set of structural patterns, binding parts of it to
variables and branching on the first match. It is the natural way to consume the sum types
described in [type systems](type_systems.md) and a hallmark of
[functional programming](functional_programming.md). It is central to [Rust](rust.md)
(`match`), [Kotlin](kotlin.md) (`when`), [Scala], and modern [Python](python.md) (`match`,
3.10+) and [TypeScript](typescript.md) (discriminated unions), and is the functional dual of
the dynamic dispatch in [OOP](oop_concepts.md).

## Beyond the switch statement

A `switch` compares one value against constants. Pattern matching also *destructures* —
matching the shape of the data and binding its parts in one step.

```rust
match msg {
    Message::Quit                     => stop(),
    Message::Move { x, y }            => move_to(x, y),   // destructure struct
    Message::Write(text)              => print(text),     // bind payload
    Message::Color(r, g, b) if r > 0  => recolor(r,g,b),  // guard
}
```

Patterns can nest (`Some((a, b))`), match literals and ranges, bind with `@`, and ignore with
`_`.

## Exhaustiveness

The most valuable property: with sum types, the compiler checks that **every case is
handled**. Add a new variant and every non-exhaustive `match` becomes a compile error,
pointing you at exactly the code to update — a refactoring safety net that `if`/`switch`
chains can't provide.

```
enum Status { Active, Paused, Closed }
            │
            ▼  add `Archived`
match status {        // ❌ compile error: `Archived` not covered
    Active => ...,
    Paused => ...,
    Closed => ...,
}
```

This is why "make illegal states unrepresentable" (sum types) + exhaustive matching is such
a powerful combination — the [type system](type_systems.md) and the control flow reinforce
each other.

## Patterns elsewhere

Even languages without a full `match` expose pieces of it: **destructuring** assignment in
JS/Python (`const {x, y} = point`, `a, *rest = xs`), tuple unpacking, and TypeScript's
discriminated unions narrowed by a `kind` tag. These give much of the ergonomic benefit short
of compiler-checked exhaustiveness.

## Where this connects

- [Type systems](type_systems.md) — pattern matching consumes sum/product types.
- [Functional programming](functional_programming.md) — matching replaces conditional chains.
- [Error handling](error_handling.md) — matching on `Result`/`Option` is the idiomatic way to
  handle outcomes.
- [OOP concepts](oop_concepts.md) — dynamic dispatch is the OO alternative to matching on type.
- [Rust](rust.md) / [Kotlin](kotlin.md) / [Python](python.md) for concrete syntax.

## Pitfalls

- **Catch-all hides new cases.** A wildcard `_ =>` arm defeats exhaustiveness checking — new
  variants silently fall through. Use it sparingly.
- **Guard ordering bugs.** Arms match top-to-bottom; an early broad pattern can shadow a later
  specific one.
- **Expression vs statement confusion.** In languages where `match` returns a value, every arm
  must produce the same type.
- **Over-matching.** Deeply nested patterns can be harder to read than a couple of `if let`s.
- **Visitor vs match coupling.** In OOP, adding a *type* is easy but adding an *operation* is
  hard; with match it's reversed — pick the axis you expect to change.
