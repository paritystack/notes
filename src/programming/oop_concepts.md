# Object-Oriented Concepts

## Overview

Object-oriented programming (OOP) structures programs as objects: bundles of state (fields)
and behaviour (methods) that interact by sending messages. It is one of the major
[programming paradigms](paradigms.md) and the dominant style in [Java](java.md),
[C++](cpp.md), [Kotlin](kotlin.md), [Python](python.md) and C#, with lighter
support in [Go](go.md) (structs + interfaces) and [Rust](rust.md) (structs + traits). This
page covers the *mechanics*; broader design guidance (SOLID, clean code) lives in
[code quality](../testing/code_quality.md), and reusable solutions in
[design patterns](design_patterns.md).

## The four pillars

```
Encapsulation   bundle state + behaviour, hide internals behind an interface
Abstraction     expose what an object does, not how it does it
Inheritance     a subclass reuses/extends a superclass ("is-a")
Polymorphism    one interface, many implementations (dispatch on actual type)
```

- **Encapsulation** keeps invariants safe by controlling access (`private`/`public`). Callers
  depend on the interface, not the representation, so internals can change freely.
- **Abstraction** models a concept at the right level — an interface/abstract class that
  hides implementation detail behind a contract.
- **Inheritance** shares code along an "is-a" relationship.
- **Polymorphism** lets code call the same method on different types and get type-appropriate
  behaviour.

## Polymorphism in detail

```
Subtype (runtime)    shape.area()  → dispatches to Circle.area or Rect.area  (vtable)
Parametric           List<T>       → see generics.md
Ad-hoc (overloading) print(int) vs print(str)
```

Subtype polymorphism via a vtable (dynamic dispatch) is the form people usually mean. It's
the OOP answer to the conditional dispatch that [pattern matching](pattern_matching.md)
provides in functional code — the two are dual approaches to the "many cases" problem.

## Composition over inheritance

Deep inheritance hierarchies are brittle: subclasses depend on superclass internals (the
*fragile base class* problem) and "is-a" often turns out to be wrong. The widely-favoured
alternative is **composition** — build behaviour by holding other objects and delegating —
plus interfaces/traits for polymorphism. Go and Rust deliberately omit class inheritance for
this reason.

```kotlin
// Inheritance: Car IS-A Engine?  No — Car HAS-A Engine.
class Car(private val engine: Engine) {     // composition
    fun start() = engine.start()            // delegation
}
```

## Interfaces, abstract classes, traits

An **interface/trait** declares a contract without (much) implementation; a class promises to
satisfy it. This decouples callers from concrete types and is the backbone of testable,
swappable designs and dependency injection. Traits (Rust) and Go interfaces add
[structural](type_systems.md) twists — conformance by shape rather than by explicit `implements`.

## Where this connects

- [Paradigms](paradigms.md) — OOP vs functional and where they mix.
- [Generics](generics.md) — parametric polymorphism complements subtype polymorphism.
- [Pattern matching](pattern_matching.md) — the functional dual of dynamic dispatch.
- [Design patterns](design_patterns.md) and [code quality (SOLID)](../testing/code_quality.md).
- [Java](java.md) / [C++](cpp.md) / [Kotlin](kotlin.md) for concrete OOP.

## Pitfalls

- **Inheritance for code reuse.** Subclassing just to grab methods couples unrelated classes;
  prefer composition.
- **Anemic objects.** Data-only classes with all logic in "manager"/"service" classes is
  procedural code wearing an OOP costume.
- **Leaky encapsulation.** Public mutable fields or getters/setters that expose internal
  collections let callers break invariants.
- **God objects / deep hierarchies.** Huge classes and tall trees are hard to change; keep
  responsibilities small (single-responsibility).
- **Equality/identity confusion.** Overriding `equals`/`hashCode` (or not) inconsistently
  breaks collections and comparisons.
