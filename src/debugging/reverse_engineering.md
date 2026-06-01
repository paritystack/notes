# Reverse Engineering

## Overview

Reverse engineering (RE) is debugging without source: recovering structure and
behaviour from a compiled binary. It builds directly on the [Binary Analysis
Tools](tools.md) (objdump, readelf, nm, strings) but adds interactive disassemblers and
*decompilers* that reconstruct C-like pseudocode, plus dynamic inspection with
[GDB](gdb.md)/[LLDB](lldb.md). It's used for malware analysis, vulnerability research,
interoperability, firmware work, and CTFs. This page surveys the major tools and the
static-vs-dynamic workflow; it assumes you are working on binaries you are authorised to
analyse.

```
The toolchain by job:
  triage / quick look   → file, strings, nm, readelf, objdump   (tools.md)
  static disassembly    → Ghidra, IDA Pro, radare2/rizin, Binary Ninja
  decompilation          → Ghidra (free), IDA + Hex-Rays, Binary Ninja
  dynamic analysis       → GDB/LLDB, rr, ltrace/strace, Frida, x64dbg (Windows)
  packing/format         → binwalk (firmware), upx -d, capa (capability ID)
```

## Static analysis — disassemblers & decompilers

```
 Tool          Cost     Decompiler   Notes
 ------------  -------  -----------  -----------------------------------------
 Ghidra        free     yes (good)   NSA-built; multi-arch; Java/Python scripting
 IDA Pro       $$$$     Hex-Rays $   industry standard; best-in-class analysis
 radare2/rizin free     (via plugin) CLI-first, scriptable; rizin is the fork w/ Cutter GUI
 Binary Ninja  $$       yes (BNIL)   clean API, multi-level IL
```

Static analysis recovers the control-flow graph, identifies functions, and (with a
decompiler) reconstructs readable pseudocode — without ever running the code, which is
the safe default for untrusted/malware samples.

```bash
# radare2 quick session
r2 ./binary
[0x00001060]> aaa          # analyse all (functions, refs, strings)
[0x00001060]> afl          # list functions
[0x00001060]> pdf @ main   # print disassembly of main
[0x00001060]> s sym.check_password; pdf   # seek + disassemble a function
[0x00001060]> izz          # all strings in the binary
```

```
Ghidra workflow:
  import binary → auto-analysis → browse the symbol tree
  → double-click a function → read the Decompiler pane (C-like pseudocode)
  → rename vars/funcs as you understand them → annotations persist in the project
```

## Dynamic analysis — run and observe

```
- GDB / LLDB     → breakpoints, watch values, dump memory at runtime (debugging pages)
- ltrace/strace  → library + syscall calls a stripped binary makes (tools.md)
- rr             → record once, replay deterministically, reverse-execute (rr_debugging.md)
- Frida          → inject JS to hook/patch functions in a live process (cross-platform)
- rizin/r2 + dbg → r2 -d ./bin for an integrated disasm + debugger
```

Static tells you what the code *can* do; dynamic tells you what it *does* with real
inputs. Malware that unpacks itself or is obfuscated often only reveals its logic at
runtime — but run untrusted code only in an isolated VM/sandbox.

## Common obstacles

```
- Stripped binaries      → no symbol names; rely on strings, xrefs, call patterns (nm shows none)
- Statically linked       → your code is buried among libc functions; use FLIRT/signatures to ID them
- Packed/compressed       → unpack first (upx -d) or dump from memory after it self-extracts
- Obfuscation/anti-debug   → ptrace checks, timing checks; patch them out or use Frida
- Position-independent     → addresses are relative; let the tool rebase (see tools.md PIE check)
```

## Pitfalls

```
- Decompiler output is a best-effort reconstruction, not the original source — verify against asm.
- Only analyse code you're authorised to (licensing, anti-circumvention law, malware handling).
- Run untrusted samples in an isolated VM with no network — malware detonates on execution.
- Stripped + optimised binaries hide intent; cross-reference strings/imports to anchor yourself.
- Don't trust a single view: confirm static hypotheses dynamically (and vice versa).
- AT&T vs Intel syntax mismatch causes misreads — pick one (objdump -M intel; tools.md).
```

## Where this connects

- [Binary Analysis Tools](tools.md) — objdump/readelf/nm/strings, the RE foundation
- [GDB](gdb.md) / [LLDB](lldb.md) — dynamic inspection and patching
- [rr (Record & Replay)](rr_debugging.md) — deterministic replay of obfuscated/self-modifying flows
- [Core Dump Analysis](core_dump.md) — recovering state from a crashed unknown binary
- [Debugging Methodology](debugging_methodology.md) — hypothesis-driven analysis without source
- [C](../programming/c.md) / [C++](../programming/cpp.md) — the languages most native binaries compile from
- [Compilers](../programming/compilers.md) — how source becomes the assembly you're reversing
