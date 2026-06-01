# LLDB

## Overview

LLDB is the debugger of the LLVM project — the [GDB](gdb.md) counterpart in the
Clang/LLVM world. It is the default debugger on macOS and iOS (driving Xcode), and is
widely used wherever Clang, Swift, or Rust toolchains live. It shares GDB's job
(breakpoints, stepping, inspecting state, [Core Dump Analysis](core_dump.md)) but has a
more regular, verb-noun command grammar and an embedded Python API. If you know GDB,
the command-mapping table below is most of what you need. For the binary-inspection
tooling LLDB sits on top of, see [Binary Analysis Tools](tools.md).

```
Architecture:
  liblldb  ← core engine, scriptable C++/Python API
    ├── command-line driver (lldb)
    ├── Xcode / VS Code (via lldb-dap, the Debug Adapter Protocol)
    └── debugserver / lldb-server  ← remote stub (cf. gdbserver)
```

## Starting and running

```bash
clang -g -O0 prog.c -o prog
lldb ./prog                          # load the program
(lldb) run                           # or: process launch
lldb -- ./prog arg1 arg2             # pass args
lldb -c core ./prog                  # post-mortem on a core file
lldb -p <pid>                        # attach to a running process
```

## GDB → LLDB command map

```
 task                    GDB                       LLDB
 ----------------------  ------------------------  --------------------------------
 set breakpoint (line)   break file.c:42           breakpoint set -f file.c -l 42
 set breakpoint (func)   break main                breakpoint set -n main
 conditional break       break f if x==5           breakpoint set -n f -c 'x==5'
 list breakpoints        info breakpoints          breakpoint list
 run / continue          run / continue            run / continue   (same)
 step / next             step / next               step / next  (or s / n; thread step-in)
 step instruction        stepi                     thread step-inst   (si)
 finish (step out)       finish                    finish   (or thread step-out)
 backtrace               bt                        thread backtrace  (bt)
 select frame            frame 2                   frame select 2
 print expression        print x  /  p x           expression x   (or p x)
 print all locals        info locals               frame variable   (fr v)
 examine memory          x/8xw &v                  memory read -c 8 -f x -s 4 &v
 watchpoint              watch var                 watchpoint set variable var
 list threads            info threads              thread list
 registers               info registers            register read
 disassemble             disassemble               disassemble   (di)
```

Most LLDB commands have short aliases (`b`, `r`, `c`, `n`, `s`, `p`, `bt`), and the
default settings ship with GDB-style aliases too, so muscle memory mostly transfers.

## Inspecting state

```
(lldb) frame variable               # all locals/args in the current frame
(lldb) p some_struct                # evaluate & pretty-print
(lldb) p/x flags                    # hex format
(lldb) memory read --format x --size 4 --count 8 ptr
(lldb) thread backtrace all         # stacks for every thread
(lldb) breakpoint set -n malloc -C 'bt' -G true   # log a backtrace, auto-continue
```

## Python scripting

LLDB embeds Python; the `lldb` module exposes the full API for automation and custom
data formatters.

```python
# my_cmds.py  →  load with: (lldb) command script import my_cmds.py
import lldb
def count_frames(debugger, command, result, internal_dict):
    thread = debugger.GetSelectedTarget().GetProcess().GetSelectedThread()
    result.PutCString(f"{thread.GetNumFrames()} frames")
def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f my_cmds.count_frames nframes')
```

```
(lldb) script print(lldb.frame.FindVariable("x").GetValue())   # one-off Python
(lldb) type summary add --summary-string "len=${var.len}" MyVec # custom formatter
```

## Remote debugging

```bash
# Target device:
lldb-server platform --server --listen "*:1234"
# Host:
lldb
(lldb) platform select remote-linux
(lldb) platform connect connect://device:1234
(lldb) file ./prog && run
```

## Pitfalls

```
- Command grammar differs from GDB: "print" is "expression"; use the aliases if unsure.
- Optimised builds: variables show as <unavailable>; debug at -O0 or use -Og.
- On macOS, attaching/launching needs code-signing entitlements or root (taskgated).
- LLDB and GDB read each other's core files imperfectly across platforms — match the toolchain.
- Mixing a GDB .gdbinit mindset: LLDB init file is ~/.lldbinit (run with --no-lldbinit to skip).
- Pretty-printers are Python data formatters here, not GDB pretty-printers — not interchangeable.
- Swift/Obj-C debugging needs the matching LLDB from that toolchain, not a generic build.
```

## Where this connects

- [GDB](gdb.md) — the GNU counterpart; the map above translates between them
- [Core Dump Analysis](core_dump.md) — `lldb -c core` for post-mortem
- [rr (Record & Replay)](rr_debugging.md) — reverse debugging (GDB-based; LLDB reverse support is limited)
- [Binary Analysis Tools](tools.md) — disassembly/symbols LLDB surfaces
- [C](../programming/c.md) / [C++](../programming/cpp.md) / [Rust](../programming/rust.md) — common debuggees
- [HardFault Debugging](../embedded/hardfault_debugging.md) — bare-metal debugging via lldb + a probe
