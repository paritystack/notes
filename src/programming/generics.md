# Generics

## Overview

Generics (parametric polymorphism) let you write code that works uniformly over many types
without giving up type safety — `List<T>`, `Map<K, V>`, `Option<T>` are written once and
reused for every element type. They are a core feature of the [type system](type_systems.md)
and appear in [Rust](rust.md), [C++](cpp.md) (templates), [Java](java.md),
[Kotlin](kotlin.md), [Go](go.md) (since 1.18), [TypeScript](typescript.md) and
[Zig](zig.md). They sit between fully dynamic code and hand-written per-type duplication, and
are the type-safe corner of [metaprogramming](metaprogramming.md).

## The problem they solve

```
Without generics, you either...
  duplicate:   IntStack, StringStack, UserStack ...   (no reuse)
  or erase:    Stack of Object + casts             (no type safety, runtime errors)

With generics:
  Stack<T>     one definition, fully type-checked per use
```

## Bounds and constraints

A bare `T` can't do much — you can store and move it but not compare, add, or print it.
**Bounds** constrain `T` to types that provide certain capabilities, so the body can use them
safely.

```rust
fn max<T: PartialOrd>(a: T, b: T) -> T {   // T must be orderable
    if a > b { a } else { b }
}
```

Equivalents: Java `<T extends Comparable<T>>`, Go `[T constraints.Ordered]`, TypeScript
`<T extends ...>`, C++20 `concept`s. Bounds are how generics stay safe: the compiler verifies
the constraint at the *definition*, so every instantiation is guaranteed to work.

## Two implementation strategies

```
MONOMORPHIZATION (C++, Rust)            TYPE ERASURE (Java, TS at runtime)
-----------------------------           ----------------------------------
generate a specialised copy             one shared copy; type args removed
per concrete type at compile time       after checking

+ zero-cost, fully inlinable            + small binary, fast compile
- code bloat, slower builds             - no runtime type info; casts/boxing,
                                          can't do `new T()` or `instanceof T`
```

Go takes a middle path (GC-shape stenciling + dictionaries). The strategy explains real
limitations: Java can't write `new T[]`, and Rust binaries grow with heavy generic use.

## Variance

When generics nest with subtyping, **variance** decides whether `List<Cat>` is a `List<Animal>`
— covariant (yes, for read-only/producers), contravariant (reversed, for write-only/consumers),
or invariant (neither). Declaration-site variance (Kotlin `out`/`in`, C# `out`/`in`) or
use-site wildcards (Java `? extends`/`? super`) express it. Getting it wrong is a classic
[type system](type_systems.md) unsoundness (Java's covariant arrays throw at runtime).

## Where this connects

- [Type systems](type_systems.md) — generics are parametric polymorphism; variance is a
  type-system rule.
- [OOP concepts](oop_concepts.md) — complements subtype polymorphism.
- [Metaprogramming](metaprogramming.md) — templates blur generics and compile-time codegen.
- [Pattern matching](pattern_matching.md) — generic sum types (`Result<T, E>`) are matched.
- [Rust](rust.md) / [Java](java.md) / [Go](go.md) pages for concrete generics.

## Pitfalls

- **Over-generalising.** Adding type parameters "just in case" hurts readability; introduce
  them when there's real reuse.
- **Erasure surprises (Java/TS).** No runtime type args — `instanceof List<String>`,
  reflection on `T`, and `new T()` don't work; types can lie after an unchecked cast.
- **Monomorphization bloat (C++/Rust).** Heavy generic + inline use balloons binary size and
  compile times.
- **Variance mistakes.** Treating a mutable `List<Cat>` as `List<Animal>` and writing a `Dog`
  into it — unsound; understand producer/consumer (PECS).
- **Constraint leakage.** Forgetting a bound forces awkward casts inside the generic body.
