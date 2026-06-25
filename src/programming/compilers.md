# Compilers

## Overview

A compiler translates source code into a lower-level form — machine code, bytecode, or
another language — checking that the program is well-formed along the way. It is where most
of the guarantees in this section are actually enforced: type checking is a compiler phase
(see [type systems](type_systems.md)), [generics](generics.md) are expanded or erased during
codegen, [metaprogramming](metaprogramming.md) is scripting the front end, and a language's
[memory management](memory_management.md) model decides what the back end must emit. The
compiled languages here exercise different points of the design space —
[Rust](rust.md), [C++](cpp.md), [Go](go.md) and [Zig](zig.md) compile ahead-of-time to native
code, [Java](java.md)/[Kotlin](kotlin.md) compile to bytecode that a JIT finishes at runtime,
and [TypeScript](typescript.md) is a transpiler that lowers to JavaScript.

## The pipeline

A compiler is a pipeline that narrows source text down to a target, one representation at a
time.

```
source ─▶ LEXER ─▶ tokens ─▶ PARSER ─▶ AST ─▶ SEMANTIC ─▶ typed AST
                                              (type-check,
                                               scope/name res)
                                                    │
                                                    ▼
target ◀─ LINKER ◀─ codegen ◀─ OPTIMIZER ◀───────── IR
          (back end)          (middle end)      (lowering)
```

Conventionally this splits into three parts. The **front end** (lex → parse → semantic
analysis) is language-specific: it turns text into a typed AST and rejects ill-formed
programs, leaning on the [type system](type_systems.md) for type checking and inference. The
**middle end** lowers the AST to an intermediate representation and optimizes it. The **back
end** is target-specific: instruction selection, register allocation, and emitting
machine/bytecode, then linking object files and libraries into the final artifact.

## IR and optimization

The middle works on an **intermediate representation** rather than the AST or raw assembly.
An IR decouples the *N* source languages from the *M* target architectures: each front end
lowers to the shared IR, each back end consumes it, so you write `N + M` pieces instead of
`N × M`. **SSA** (static single assignment — every variable assigned exactly once) is the
dominant IR form because it makes data flow explicit and most optimizations cheap to express.

Optimizations rewrite the IR to be faster or smaller without changing observable behavior:

```
constant folding:   x = 2 * 60 * 60     ─▶  x = 7200
dead-code elim:      if (false) { ... }  ─▶  (removed)
inlining:            y = square(a)       ─▶  y = a * a   (then fold/simplify)
```

Inlining is the keystone — once a call is inlined, constant folding, dead-code elimination,
and common-subexpression elimination cascade through the exposed code. Flags like `-O0`…`-O3`
/`-Os` trade compile time and binary size against runtime speed; `-O0` keeps a one-to-one
mapping to source for debugging, higher levels reorder and erase it.

## AOT vs JIT vs transpilers

*When* compilation happens shapes the whole trade-off.

```
AHEAD-OF-TIME (AOT)                  JUST-IN-TIME (JIT)
------------------                   ------------------
compile fully before running         compile hot paths during execution
C, C++, Rust, Go                     JVM (Java/Kotlin), V8 (JS), PyPy

+ fast, predictable startup          + optimizes on real runtime profiles
+ no runtime compiler shipped        + can speculate, deoptimize, recompile
- can't use runtime information      - warm-up cost; compiler runs in-process
```

A **transpiler** (source-to-source compiler) is just a compiler whose target is another
high-level language: [TypeScript](typescript.md) → JavaScript, or Babel down-leveling new JS
to older JS. Much of this machinery is shared infrastructure: **LLVM** provides a
language-agnostic IR and back ends (used by Clang, Rust, Swift, Zig), **GCC** is the long-
standing native suite, and the **JVM** pairs a bytecode interpreter with a profiling JIT.

## Where this connects

- [Type systems](type_systems.md) — type checking and inference are compiler phases; inference
  is constraint solving over the typed AST.
- [Metaprogramming](metaprogramming.md) — macros are syntax-tree transformations; you're
  scripting the front end.
- [Generics](generics.md) — monomorphization vs type erasure is a codegen strategy decided in
  the back end.
- [Memory management](memory_management.md) — the compiler emits allocation/free or GC barriers
  and (in Rust) enforces ownership during semantic analysis.
- [Build systems & linking](build_systems.md) — what happens after codegen: symbol resolution,
  linking object files into an artifact, and orchestrating which units to compile.
- [Regular expressions](regular_expressions.md) — the lexer specifies tokens as regexes and
  compiles them to a DFA; regex theory's original home.
- [Rust](rust.md) / [C++](cpp.md) / [Go](go.md) / [Zig](zig.md) / [Java](java.md) for concrete
  compilation models.

## Pitfalls

- **Optimization exploiting undefined behavior.** The optimizer assumes UB never happens and
  may delete checks or "impossible" branches; in C/C++ a signed overflow or null deref can make
  whole blocks vanish. Build with sanitizers, not just trust.
- **`-Ofast` breaks the rules.** `-ffast-math` reorders floating-point and assumes no NaN/Inf,
  silently changing numeric results — never enable it blindly in numerical code.
- **Debug vs release divergence.** Bugs that appear only at `-O2` (or only at `-O0`) usually
  point at UB, uninitialized memory, or a data race the optimizer exposed — not a compiler bug.
- **JIT warm-up.** Benchmarks that don't account for warm-up measure the interpreter, not the
  JIT-compiled steady state.
- **Trusting the optimizer over the algorithm.** No optimization level turns an O(n²) loop into
  O(n); fix algorithms and data layout first.
- **Miscompilation and "trusting trust".** Compilers have bugs, and a compromised compiler can
  inject code invisible in source — reproducible builds and cross-compiler checks matter for
  high-assurance work.
