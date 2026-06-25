# FFI & Language Interop

## Overview

A **foreign function interface (FFI)** lets code written in one language call code compiled
from another — Python calling a C math kernel, [Rust](rust.md) exposing a library to
[Go](go.md), a JVM app reaching native code through JNI. Because no two languages share a
runtime, they have to agree on a *binary* contract at the boundary: the calling convention and
data layout of the **ABI** (see [build systems & linking](build_systems.md), which covers how
those symbols get linked). C is the universal meeting point — almost every language can speak
the C ABI — so "FFI" in practice usually means "talk C". The hard parts are
[memory](memory_management.md) ownership across the boundary and marshalling data between
representations ([serialization](serialization.md)'s in-process cousin).

## Calling conventions & the ABI

Calling a function at the machine level means agreeing on *where arguments go* (which
registers, then the stack), *who cleans up the stack*, and *how the return value comes back*.
That agreement is the **calling convention**, and together with name mangling, struct layout,
and type sizes it forms the **ABI**.

```
API  (source contract)        ABI  (binary contract)
- function signatures         - args in rdi, rsi, rdx, ... then stack (SysV x86-64)
- type names                  - who pops the stack, callee-saved registers
- compiles or it doesn't      - struct padding/alignment, name mangling, type sizes
```

The **C ABI** is the lingua franca because it's stable and dead simple — no exceptions, no
mangling, no runtime. [C++](cpp.md) mangles names and has no stable ABI, so cross-language
exports are wrapped in `extern "C"`. [Rust](rust.md) and [Zig](zig.md) similarly expose C-ABI
functions (`extern "C"`, `export`) to be callable from anywhere.

## Marshalling across the boundary

The boundary is where rich language types meet C's flat ones, and where ownership gets
decided. Three recurring problems:

```
memory ownership   who frees this pointer — caller or callee? across a GC boundary?
data layout        struct field order/padding must match exactly; no hidden headers
strings & arrays   C uses NUL-terminated char* + length; most langs use (ptr,len) objects
```

You pass **plain old data** (numbers, pointers, C structs) directly; everything else is
*marshalled* — copied or wrapped into a C-compatible shape and back. Strings are the classic
trap: a `String`/`str` must become a `char*` (often a copy, with someone responsible for
freeing it), and lifetimes that the source language tracked automatically become manual at the
boundary.

## Tooling & the cost of crossing

Writing bindings by hand is error-prone, so generators do it: **bindgen** (C headers → Rust),
**cbindgen** (Rust → C headers), **SWIG** (C/C++ → many languages), **cffi**/**ctypes**
(Python), and **cgo** (Go). They read declarations and emit the glue and type shims.

Crossing the boundary is not free, and the cost depends on the runtimes involved:

```
thin FFI   (Rust↔C, Zig↔C)   ~a normal function call; same ABI, no runtime gap
cgo        (Go→C)            stack switch off the goroutine stack; pins an OS thread
JNI        (JVM→native)      handle bookkeeping, GC safepoints, copies; call overhead
ctypes     (Python→C)        per-call marshalling in the interpreter; batch to amortize
```

Runtime mismatches dominate: a garbage collector must not move or free objects a native call
still holds (pinning/handles), and green-threaded runtimes like Go must hand off to a real OS
thread before entering blocking C. The rule of thumb is to cross the boundary *coarsely* — one
call doing a lot of work beats thousands of chatty calls.

## Where this connects

- [Build systems & linking](build_systems.md) — FFI is an ABI problem; the linker resolves the
  foreign symbols and static/dynamic linking decides how the library ships.
- [C](c.md) / [C++](cpp.md) — the C ABI is the meeting point; `extern "C"` defeats C++ name
  mangling for export.
- [Rust](rust.md) / [Zig](zig.md) / [Go](go.md) — `extern "C"`/`export`/cgo expose or consume
  C-ABI functions.
- [Python](python.md) — `ctypes`/`cffi` and native extension modules are FFI; numerical stacks
  (NumPy) are thin wrappers over C/Fortran.
- [Memory management](memory_management.md) — ownership, pinning, and freeing across a GC
  boundary is the core difficulty.
- [Serialization](serialization.md) — marshalling is in-process serialization; the same
  layout/ownership questions apply.

## Pitfalls

- **Ownership ambiguity.** Unclear who frees a returned pointer → leaks or double-frees.
  Document and encode ownership at every boundary function.
- **Struct layout mismatch.** Different padding/alignment or field order between the two sides
  silently corrupts data; pin layout (`#[repr(C)]`, `packed`) and match it exactly.
- **String/encoding errors.** Forgetting the NUL terminator, mismatched length, or UTF-8 vs
  UTF-16 assumptions; copy and validate at the edge.
- **GC moving/collecting live objects.** Native code holding a pointer the collector moves or
  frees → use-after-free; pin or use handles for the duration of the call.
- **Exceptions/panics across the boundary.** Unwinding past an `extern "C"` frame is undefined;
  catch and convert to error codes before returning.
- **Chatty boundaries.** Per-element FFI calls dominate runtime; batch work into one call.
- **ABI drift.** A prebuilt foreign library compiled against a different ABI version links but
  crashes — pin and match toolchain/ABI (see [build systems](build_systems.md)).
