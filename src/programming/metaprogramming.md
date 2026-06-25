# Metaprogramming

## Overview

Metaprogramming is writing programs that treat other programs (or themselves) as data —
generating, inspecting, or transforming code. It spans compile-time techniques like macros
and templates and run-time techniques like reflection. It cuts across nearly every language
in this section: [C](c.md)'s preprocessor, [C++](cpp.md) templates and `constexpr`,
[Rust](rust.md) macros, [Zig](zig.md) `comptime`, [Java](java.md)/[Kotlin](kotlin.md)
annotations + reflection, [Python](python.md) decorators/metaclasses, and
[JavaScript](javascript.md) `Proxy`. It leans on the [type system](type_systems.md) and is,
in effect, programming the [compiler](compilers.md).

## Compile-time vs run-time

```
COMPILE TIME                              RUN TIME
------------                              --------
text macros   (C #define)                reflection      (Java, C#, Go)
syntactic macros (Rust, Lisp)            introspection   (Python dir/getattr)
templates / generics (C++)               dynamic proxies (JS Proxy, Java)
comptime (Zig), constexpr (C++)          eval / codegen  (Python, JS)
annotation processors (Java)             monkey-patching (Python, Ruby)

  + zero runtime cost, type-checked        + adapts to data unknown until runtime
  - harder to debug, slower builds         - slower, less safe, defeats static checks
```

The trade-off is fundamental: compile-time metaprogramming pays once at build time and keeps
runtime fast and type-safe; run-time metaprogramming is flexible and dynamic but costs
performance and erodes the guarantees of the [type system](type_systems.md).

## Macros

A macro transforms code before (or during) compilation. Two kinds:

- **Textual / preprocessor** (C `#define`) — blind token substitution. Powerful but unhygienic:
  no awareness of scope or types, infamous for surprising bugs.
- **Syntactic / hygienic** (Rust `macro_rules!` and proc-macros, Lisp, Scheme) — operate on
  the *syntax tree*, respect scoping, and can't accidentally capture variables.

```rust
// Rust declarative macro: variadic, type-aware, hygienic
macro_rules! max {
    ($x:expr) => { $x };
    ($x:expr, $($rest:expr),+) => { std::cmp::max($x, max!($($rest),+)) };
}
let m = max!(3, 7, 2); // expands at compile time
```

## Templates, generics, and comptime

[Generics](generics.md) are a constrained form of metaprogramming — the compiler generates a
specialised version per type. C++ templates push this further into full Turing-complete
*template metaprogramming*; [Zig](zig.md) replaces all of it with ordinary code that runs at
`comptime`, and C++ `constexpr`/`consteval` evaluate normal functions at compile time. These
let you compute types and constants, unroll loops, and validate invariants before the program
runs.

## Reflection and introspection

Reflection lets running code inspect and manipulate its own structure — list a class's
fields/methods, read [annotations](type_systems.md), instantiate by name, or call methods
dynamically. It powers serialization libraries, dependency injection, ORMs, and test
frameworks.

```python
@dataclass            # decorator: rewrites the class at definition time
class User:
    name: str
    age: int

for f in fields(User):          # introspection
    print(f.name, f.type)
```

Decorators (Python/TS), annotations (Java/Kotlin), and attributes (C#/Rust) are the common
*declarative* surface: you tag code, and a macro/processor or runtime acts on the tag.

## Code generation

When macros/reflection aren't enough, projects generate source files from a schema:
Protobuf/gRPC stubs (see [serialization](serialization.md)), `go generate`, build-time
codegen. It keeps generated code visible and debuggable at the cost of an extra build step.

## Where this connects

- [Generics](generics.md) — the type-safe, constrained subset of compile-time metaprogramming.
- [Type systems](type_systems.md) — annotations and `constexpr`/`comptime` interplay with types.
- [Compilers](compilers.md) — macros are syntax-tree transformations; you're scripting a compile phase.
- [Serialization](serialization.md) — reflection/codegen generate (de)serializers.
- [Design patterns](design_patterns.md) — reflection underpins DI and proxy-based patterns.
- [Closures & lexical scope](closures.md) — decorators and DSL builders are closures over the
  wrapped target.

## Pitfalls

- **Debuggability collapse.** Macro-expanded and generated code has no obvious source line;
  errors point at the expansion. Keep macros small and provide `cargo expand`-style tooling.
- **Hygiene bugs.** Unhygienic textual macros capture/shadow variables and mis-handle
  operator precedence — always parenthesise, prefer hygienic macros.
- **Reflection breaks static guarantees.** Dynamic field/method access can't be type-checked
  and silently breaks on renames/minification/obfuscation.
- **Performance cost of reflection.** Runtime introspection is slow; cache resolved
  metadata or move to compile-time codegen on hot paths.
- **Over-magic.** Heavy metaprogramming makes code unreadable to newcomers and tooling (jump-
  to-definition fails). Use the least powerful technique that works.
- **Template/comptime blow-up.** Excessive specialization inflates binary size and build times.
