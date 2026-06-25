# Build Systems & Linking

## Overview

A [compiler](compilers.md) turns one source file into one object file; a **build system**
decides *which* files to compile and in what order, and a **linker** stitches the resulting
objects and libraries into a single runnable artifact. This is the step the compilers page
hands off to — it ends at "linking object files and libraries into the final artifact" and
this page picks up there. It sits between source code and the running process, so it touches
[memory management](memory_management.md) (where code and data land in the address space),
[serialization](serialization.md) (lockfiles and package manifests are just serialized
dependency graphs), and [security](../security/supply_chain_security.md) (every dependency you
pull is code you ship). For the bare-metal specialization of all this — cross-compilers,
`.ld` scripts, flashing — see [embedded build systems](../embedded/build_systems.md) and
[linker scripts](../embedded/linker_scripts.md).

## Separate compilation

Real programs aren't compiled as one giant file. Each **translation unit** (a source file
plus the headers it pulls in) is compiled independently into an **object file** containing
machine code plus an unfinished symbol table. Headers carry *declarations* (a promise that
`f` exists somewhere) while exactly one source file carries the *definition*.

```
main.c ─▶ cc ─▶ main.o ┐
util.c ─▶ cc ─▶ util.o ┼─▶ ld ─▶ a.out
math.c ─▶ cc ─▶ math.o ┘      ▲
                              │
                       libc.a / libm.so
```

Compiling units separately is what makes **incremental builds** possible: change `util.c`
and only `util.o` needs rebuilding, then a relink. It also means the compiler sees one unit
at a time — it cannot check that your declaration of `f` matches its definition in another
file. That mismatch is deferred to link time (or silently miscompiled), which is the root of
a whole class of bugs below.

## Linking: symbols & resolution

An object file exports **defined symbols** (functions/globals it provides) and lists
**undefined symbols** (things it calls but doesn't define). The linker's job is *symbol
resolution* — match every undefined symbol to exactly one definition — followed by
*relocation*, patching the placeholder addresses now that everything has a final location.

```
main.o   defines: main          undefined: printf, util_init
util.o   defines: util_init     undefined: malloc
libc     defines: printf, malloc

resolve → every undefined symbol bound to one definition → patch addresses
```

Two rules bite constantly. The **one-definition rule**: exactly one definition per symbol —
zero gives "undefined reference", two give "multiple definition". And **link order**: with
static libraries the traditional Unix linker scans left to right and only pulls in archive
members that satisfy a *currently* undefined symbol, so a library must appear *after* the
object that needs it. The embedded [linker scripts](../embedded/linker_scripts.md) page
covers the deep version of this — sections, relocation, and placing symbols at fixed
addresses.

## Static vs dynamic linking

Libraries come in two flavours, and the choice shapes deployment, size, and security.

```
STATIC  (.a / .lib)                    DYNAMIC  (.so / .dll / .dylib)
-------------------                    -----------------------------
copied into the executable at link     referenced; loaded at run time
self-contained, no runtime deps        smaller binary, shared in RAM across procs
larger binary, duplicated code         needs the lib present & ABI-compatible
fix a bug → relink & redeploy all      fix a bug → patch one .so, all users get it
faster startup, predictable            startup cost; "missing .so" failures
```

Static linking trades binary size for a self-contained artifact (the reason Go binaries and
many containers favour it). Dynamic linking trades a runtime dependency for shared memory and
central security patching — one `libssl` fix updates every program on the box.

## Dynamic loading at run time

With dynamic linking the executable only holds *references*; a **dynamic linker/loader**
(`ld.so` on Linux, the loader in the OS on Windows/macOS) resolves them when the process
starts. It finds libraries via a search path — `RPATH`/`RUNPATH` baked into the binary,
`LD_LIBRARY_PATH`, and system defaults — then maps them into the address space. Programs can
also load libraries explicitly at run time with `dlopen`/`LoadLibrary`, the mechanism behind
plugin systems.

The contract here is the **ABI** (application *binary* interface), not the API: struct
layout, calling convention, and symbol names/versions. Source that still compiles can break a
prebuilt binary if the ABI shifts — reordering struct fields or changing a type is an ABI
break even when the API looks identical. **Symbol versioning** (`GLIBC_2.34`) lets one `.so`
ship multiple incompatible versions of a symbol so old binaries keep working.

## Build orchestration

For anything beyond a handful of files you describe the work as a **dependency graph** — each
output depends on inputs — and let a tool rebuild only what's stale, in parallel where the
graph allows.

```
app  ← main.o ← main.c, util.h
     ← util.o ← util.c, util.h     # touch util.h → rebuild both .o, relink app
```

`make` is the archetype (targets, prerequisites, recipes) but tracks dependencies manually.
The common progression layers abstraction on top: **CMake**/Meson *generate* build files for a
faster engine like **Ninja**; **Bazel**/Buck add hermetic, cached, reproducible builds for
large monorepos. Most languages ship their own front end over the same ideas — `cargo`
(Rust), `go build`, Gradle/Maven (JVM), `npm`/webpack (JS) — bundling compilation, linking,
and the dependency management below into one command.

## Dependency management

Modern builds pull code from **package managers** (Cargo, npm, pip, Go modules, Maven). You
declare wanted versions, usually as ranges via **semantic versioning** (`^1.4` = "1.4 up to
but not including 2.0"); the resolver picks concrete versions and records them in a
**lockfile** so every machine and CI run builds the identical graph. The lockfile is just a
serialized resolution — see [serialization](serialization.md).

The hard part is **transitive dependencies**: your deps have deps. When two paths demand
incompatible versions of the same package you hit a **diamond/conflict** ("dependency hell").
Ecosystems differ — npm can nest multiple versions, Maven picks one nearest the root, Cargo
unifies within semver-compatible ranges. Every pulled package is also code you execute at
build and run time, which is why pinning, lockfiles, and provenance matter for
[supply-chain security](../security/supply_chain_security.md).

## Where this connects

- [Compilers](compilers.md) — produces the object files this page links; linking is the
  compiler's back-end hand-off.
- [Memory management](memory_management.md) — static vs dynamic linking decides what's mapped
  into the address space and where code/data sections live.
- [Serialization](serialization.md) — lockfiles and manifests are serialized dependency
  graphs; schema/version discipline mirrors dependency pinning.
- [C](c.md) / [C++](cpp.md) — the headers + `.o` + linker model in its rawest form (ODR, link
  order, `extern`).
- [Rust](rust.md) / [Go](go.md) / [Zig](zig.md) — language-native build+dependency tooling
  (`cargo`, `go build`, `zig build`) and a default lean toward static linking.
- [Embedded build systems](../embedded/build_systems.md) / [linker scripts](../embedded/linker_scripts.md)
  — the bare-metal specialization: cross-compilation, `.ld` placement, flash/RAM layout.
- [FFI & language interop](ffi.md) — calling foreign code is an ABI problem; the linker
  resolves the foreign symbols and linking decides how the library ships.
- [Supply-chain security](../security/supply_chain_security.md) — dependencies are untrusted
  code you ship; pinning, lockfiles, and provenance.

## Pitfalls

- **Multiple definition / undefined reference.** A symbol defined in a header included by
  several units, or a missing definition — the ODR violated in either direction. Declare in
  headers, define in exactly one source file.
- **Link order.** With classic static linking, a library listed before the object that uses
  it gets its symbols dropped → "undefined reference" despite the lib being present.
- **ABI breaks across shared libraries.** Rebuilding a `.so` after reordering a struct or
  changing a type breaks prebuilt callers even though the source still compiles. Bump the
  soname / use symbol versioning.
- **"Works on my machine" — missing or wrong `.so`.** Dynamic deps that exist on the dev box
  but not in production, or a different version on the load path (`LD_LIBRARY_PATH`/RPATH).
  Static linking or containers sidestep it.
- **Unpinned dependencies / dependency hell.** No lockfile → non-reproducible builds; diamond
  conflicts → unresolvable graphs. Commit lockfiles; pin transitive versions.
- **Link-time optimization surprises.** LTO and aggressive inlining across units can expose
  latent UB or change behaviour between debug and release — the same divergence the compilers
  page warns about, now at link scope.
- **Static-linking licensing.** Statically linking a copyleft (e.g. GPL) library pulls its
  obligations into your binary in ways dynamic linking may not — a legal, not technical, trap.
