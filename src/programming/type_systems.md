# Type Systems

## Overview

A type system is the set of rules a language uses to assign *types* to expressions and to
reject programs that misuse them. It is the backbone that the rest of a language is built
on: it shapes [error handling](error_handling.md), enables [generics](generics.md),
underpins [pattern matching](pattern_matching.md), and decides how much the
[compiler](compilers.md) can prove before your code ever runs. The languages in this
section sit at very different points on the spectrum — [Python](python.md) and
[JavaScript](javascript.md) are dynamically typed, [TypeScript](typescript.md) bolts a
gradual static layer on top of JS, while [Rust](rust.md), [C++](cpp.md),
[Zig](zig.md), [Go](go.md), [Java](java.md) and [Kotlin](kotlin.md) check types ahead of
time.

## The main axes

Type systems are best understood as a set of independent axes, not a single ranking.

```
                static  <----------------->  dynamic
                (checked at compile time)    (checked at run time)
    Rust, C++, Java, Go        TypeScript        Python, JS, Lua

                strong  <----------------->  weak
                (few implicit coercions)    (many implicit coercions)
    Python, Rust               Java/TS           C, JavaScript

                nominal <----------------->  structural
                (compatible by name)        (compatible by shape)
    Java, C++, Rust            mixed             TypeScript, Go (interfaces)
```

- **Static vs dynamic** — *when* types are checked. Static catches whole classes of bugs
  before running and powers IDE tooling; dynamic trades that for flexibility and less
  ceremony.
- **Strong vs weak** — how willing the language is to *implicitly* convert between types.
  C's silent integer/pointer coercions are weak; `1 + "1"` is a `TypeError` in Python
  (strong) but `"11"` in JavaScript (weak).
- **Nominal vs structural** — two types are compatible because they share a *name/declaration*
  (nominal) or because they have the same *shape* (structural). Go interfaces and TypeScript
  objects are structural; a Java `class` is nominal.

## Type inference

Inference lets the compiler deduce types you don't write. Local inference (`var`/`auto`/
`let`) is now common; whole-program Hindley–Milner inference (ML, Haskell, and partly
Rust) can infer almost everything from usage.

```rust
let xs = vec![1, 2, 3];   // Rust infers Vec<i32>
let total = xs.iter().sum::<i32>();
```

Inference improves ergonomics without giving up static guarantees — but error messages get
worse the more the compiler has to guess, which is why annotating public function
signatures is a near-universal convention.

## Algebraic data types

Most modern static type systems model data as **products** (a struct/tuple holds an A *and*
a B) and **sums** (an enum holds an A *or* a B). Sum types make illegal states
unrepresentable and pair naturally with [pattern matching](pattern_matching.md).

```rust
enum Shape {                 // sum: a Shape is ONE of these
    Circle { r: f64 },       // product: Circle has an r
    Rect { w: f64, h: f64 }, // product: Rect has a w AND an h
}
```

`Option<T>` / `Result<T, E>` (Rust), `Optional<T>` (Java/Swift) and TS unions are the same
idea — encoding "maybe absent" or "maybe failed" in the type instead of via `null` or
exceptions.

## Generics and variance

Parametric polymorphism (see [generics](generics.md)) lets a type system describe
`List<T>` once for all `T`. **Variance** governs subtyping of those containers: if `Cat` is
a `Animal`, is `List<Cat>` a `List<Animal>`? Covariant yes, contravariant reversed,
invariant neither. Getting variance wrong is a classic source of unsoundness (Java's array
covariance throws `ArrayStoreException` at runtime for exactly this reason).

## Gradual and gradual-ish typing

Gradual typing mixes static and dynamic in one program: untyped code interoperates with
typed code, with checks inserted at the boundary. [TypeScript](typescript.md), Python type
hints (`mypy`/`pyright`), and PHP/Hack are the mainstream examples. The trade-off is that
the dynamic parts can still violate the static guarantees at runtime — types are erased and
not enforced unless a runtime check is added.

## Where this connects

- [Generics](generics.md) and [pattern matching](pattern_matching.md) are features the type
  system enables.
- [Error handling](error_handling.md) — `Result`/`Option` types are a type-system answer to
  failure.
- [Compilers](compilers.md) — type checking is a compiler phase; inference is constraint
  solving.
- [Memory management](memory_management.md) — Rust's ownership is enforced *by* its type
  system (the borrow checker).
- [TypeScript](typescript.md) / [Rust](rust.md) / [Kotlin](kotlin.md) pages show concrete
  type systems in practice.

## Pitfalls

- **`null` as a hole in the type system.** "The billion-dollar mistake" — a `null` that
  inhabits every reference type defeats static guarantees. Prefer `Option`/nullable types
  with compiler-enforced checks.
- **Confusing strong with static.** They are orthogonal: Python is strong + dynamic, C is
  weak + static.
- **Over-trusting gradual types.** TS/`mypy` types are erased; data crossing an API or
  `any`/`# type: ignore` boundary can be the wrong shape at runtime. Validate at trust
  boundaries.
- **Fighting inference.** Letting the compiler infer deeply nested types yields cryptic
  errors; annotate public signatures.
- **Unsound escape hatches.** `any` (TS), `unsafe` (Rust), `reflect`/casts (Java) bypass the
  checker — every use is a place the guarantees no longer hold.
