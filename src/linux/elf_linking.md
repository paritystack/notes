# ELF, Linking & Loading

## Overview

This page explains what a Linux executable *is* and how it gets into memory and running: the
**ELF** format, **static vs dynamic linking**, the dynamic loader `ld.so`, the **PLT/GOT** lazy-
binding machinery, **relocations**, PIE/ASLR, and the **vDSO**/syscall entry boundary. It picks up
where [Process Management](process.md) (the userspace process memory layout) and
[Process Internals (Kernel)](process_internals.md) (`execve()` internals) leave off, and it's the
runtime counterpart to building those binaries in [Cross Compilation](cross_compilation.md)
(toolchains, sysroots, target ABIs).

From source to running process there are four stages; this page is mostly about the **link** and
**load** stages.

```
   .c ──cc──► .o (relocatable ELF)
                 │  ld  (static link)
                 ▼
   a.out / lib.so (ELF executable / shared object)
                 │  execve()  → kernel maps it, hands off to...
                 ▼
   ld.so (dynamic loader) → map libraries, relocate, run
                 ▼
   _start → __libc_start_main → main()
```

## ELF anatomy: two views of one file

An ELF file is described twice — by **sections** (link-time view) and **segments** (load-time
view). The ELF header points at a *section header table* and a *program header table*.

```
   ELF header
   ├─ Program headers (SEGMENTS) ── what the loader maps
   │    LOAD  r-x  .text + .rodata
   │    LOAD  rw-  .data + .bss
   │    DYNAMIC, INTERP, GNU_STACK, GNU_RELRO …
   └─ Section headers (SECTIONS) ── what the linker/debugger uses
        .text .rodata .data .bss .symtab .rela.* .dynamic .got .plt …
```

- **Sections** (`.text` code, `.rodata` constants, `.data` initialised globals, `.bss` zero-init
  globals, `.symtab`/`.dynsym` symbols, `.rela.*` relocations) — used by the linker and tools.
- **Segments** (program headers) — what `execve`/`ld.so` actually `mmap`s. The `INTERP` segment
  names the dynamic loader (`/lib64/ld-linux-x86-64.so.2`); `DYNAMIC` holds the linking metadata.

`.bss` occupies no file bytes (just a size) — it's zero-filled at load.

## Static vs dynamic linking

- **Static** (`-static`): all library code is copied into the binary at link time. One
  self-contained file, no runtime loader, fastest startup — but larger, and a libc security fix
  requires relinking. Common for containers (scratch images) and embedded.
- **Dynamic** (default): the binary lists *needed* shared objects (`DT_NEEDED`, e.g.
  `libc.so.6`); `ld.so` maps them at startup and resolves symbols. Smaller binaries, shared code
  pages across processes, and central library upgrades — at the cost of load-time work.

```bash
ldd ./app            # shared libs the loader will pull in
file ./app           # "dynamically linked, interpreter /lib64/ld-linux..."
readelf -d ./app     # .dynamic: NEEDED, RPATH/RUNPATH, SONAME
```

## How the dynamic loader finds libraries

At `execve`, the kernel sees the `INTERP` and instead loads `ld.so`, which then maps the
executable's dependencies. Search order for each `DT_NEEDED`:

1. `DT_RPATH` (legacy, before `LD_LIBRARY_PATH`),
2. `LD_LIBRARY_PATH` (env override — handy for dev, dangerous for setuid),
3. `DT_RUNPATH` (modern rpath, after the env),
4. `ld.so.cache` (`/etc/ld.so.conf` → `ldconfig`),
5. default `/lib`, `/usr/lib`.

**SONAME** versioning (`libfoo.so.1`) lets multiple ABI-incompatible versions coexist;
the dev symlink `libfoo.so` is link-time only.

## PLT/GOT and lazy binding

Dynamic calls go through two tables so code pages stay read-only and position-independent:

- **GOT** (Global Offset Table, `.got`/`.got.plt`) — writable table of resolved addresses.
- **PLT** (Procedure Linkage Table, `.plt`) — small stubs the code calls instead of the real
  function.

```
   call foo@plt ──► PLT stub ──► jmp *GOT[foo]
                                    │
              first call: GOT points back into PLT → ld.so resolver
                          resolver finds foo, writes its addr into GOT[foo]
              later calls: GOT[foo] already → jumps straight to foo
```

This is **lazy binding**: a symbol is resolved on first use, amortising startup. `LD_BIND_NOW=1`
(or full **RELRO**, `-Wl,-z,now`) resolves everything up front and then marks the GOT read-only —
slower start, but removes a classic GOT-overwrite exploit target.

## Relocations, PIE and ASLR

A **relocation** patches an address that wasn't known at compile time. **Position-independent**
code (`-fPIC` for libraries, **PIE** executables, default on modern distros) uses PC-relative
addressing plus the GOT so the image can load at *any* base — which is what lets **ASLR**
randomise the load address each run for security. The trade-off is slightly more indirection
than a fixed-address non-PIE binary.

## The vDSO and the syscall boundary

A handful of "syscalls" that only *read* kernel data (`gettimeofday`, `clock_gettime`, `getpid` on
some arches) are served without a real kernel transition: the kernel maps a tiny shared object,
the **vDSO**, into every process; libc calls into it and reads kernel-maintained data directly.

```bash
cat /proc/self/maps | grep vdso     # [vdso] mapping
```

Real syscalls cross to the kernel via the `syscall` instruction (the ABI: number in `rax`, args in
`rdi, rsi, rdx, r10, r8, r9` on x86-64) — see `execve` handling in
[Process Internals (Kernel)](process_internals.md).

## Toolbox

```bash
readelf -h a.out         # ELF header (type: EXEC/DYN, machine, entry)
readelf -l a.out         # program headers (segments) + section→segment map
readelf -S a.out         # section headers
readelf -d a.out         # .dynamic (NEEDED, SONAME, RUNPATH, FLAGS)
nm -D libfoo.so          # exported dynamic symbols
objdump -d -j .plt a.out # disassemble the PLT stubs
ldd a.out                # resolved shared-lib tree (don't run on untrusted binaries)
LD_DEBUG=libs,reloc ./app  # trace the loader: search + relocation processing
strace -e trace=openat ./app  # watch which .so files get opened
```

## Where this connects

- [Process Management](process.md) — the in-memory layout (text/data/bss/heap/stack/mmap) these
  ELF segments become; `/proc/<pid>/maps` shows the result.
- [Process Internals (Kernel)](process_internals.md) — `execve()` parses the ELF, sets up the
  address space, and transfers control to `ld.so`/`_start`.
- [Cross Compilation](cross_compilation.md) — toolchains, sysroots, target ABIs/SONAMEs, and
  static vs dynamic choices when building for another architecture.
- [Container Runtimes](container_runtimes.md) — static linking (or bundling libs) is why "scratch"
  container images work without a full userland.

## Pitfalls

- **`LD_LIBRARY_PATH` as a deployment fix.** It's a dev convenience; it's ignored for setuid
  binaries and breaks when the environment isn't set. Use `RUNPATH` (`-Wl,-rpath`) or install into
  the standard path + `ldconfig`.
- **Forgetting `ldconfig` after installing a library.** The new `.so` won't be in `ld.so.cache`,
  so dynamic links fail at startup with "cannot open shared object file."
- **SONAME / ABI mismatch.** Bumping a library's interface without bumping the SONAME silently
  breaks dependents; symbol-versioning and SONAME discipline prevent it.
- **Running `ldd` on an untrusted binary.** Historically `ldd` could execute the target; use
  `readelf -d` / `objdump` for untrusted files.
- **Assuming the vDSO exists.** It's an optimisation, not guaranteed for every call/arch; libc
  falls back to a real syscall — don't hand-roll vDSO access.
- **Static-linking glibc and using NSS/`dlopen`.** Statically linked glibc still `dlopen`s NSS
  modules at runtime, so "fully static" glibc binaries can fail `getpwnam`/DNS on other hosts;
  prefer musl (or `getent`) for truly static deployments.
