# GDB for Embedded

## Overview

`arm-none-eabi-gdb` is the same GDB you use on Linux, configured to drive a remote target instead of a local process. The "remote" is a GDB server provided by your debug probe (OpenOCD, pyOCD, J-Link GDBServer, Black Magic Probe, probe-rs) that translates GDB's remote protocol into SWD/JTAG transactions on the wire.

```
   arm-none-eabi-gdb                  Probe                    Target MCU
                                                              
   "load",                            ┌──────────┐            ┌──────────┐
   "break main",     RSP (GDB remote) │ OpenOCD/ │   SWD/JTAG │ Cortex-M │
   "continue"   ◄───────────────────► │ J-Link/  │ ◄────────► │   DAP    │
                       TCP 3333       │  pyOCD   │            │          │
                                       └──────────┘            └──────────┘
```

Once connected, GDB drives the chip: load firmware, set breakpoints, single-step, peek memory, watch variables, dump SCB. This doc covers the workflow, scripting, and the embedded-specific extensions.

## The Connection Step

Start the GDB server in one terminal:

```bash
# OpenOCD with ST-Link, targeting STM32F4
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

# Or pyOCD
pyocd gdbserver -t stm32f407vg

# Or J-Link
JLinkGDBServer -device STM32F407VG -if SWD -speed 4000

# Or probe-rs
probe-rs gdb --chip STM32F407VG
```

All listen on **TCP 3333** by default. Then in another terminal:

```bash
arm-none-eabi-gdb firmware.elf
(gdb) target extended-remote :3333
(gdb) monitor reset halt
(gdb) load
(gdb) break main
(gdb) continue
```

The `extended-remote` variant supports more probe commands than plain `remote` (notably `run` and `kill`).

## .gdbinit

Anything you'd type after connecting can live in `.gdbinit` — GDB sources it at startup.

```gdb
# project .gdbinit
set confirm off
set pagination off
set print pretty on
set print array on

target extended-remote :3333

# Reset every time we load
define hook-load
    monitor reset halt
end

# Convenience: 'r' = reset + reload + go
define r
    monitor reset halt
    load
    continue
end

# Decode a HardFault stack frame
define hardfault
    set $sp_real = ($lr & 4) ? $psp : $msp
    printf "PC   = 0x%08x\n", ((uint32_t*)$sp_real)[6]
    printf "LR   = 0x%08x\n", ((uint32_t*)$sp_real)[5]
    printf "PSR  = 0x%08x\n", ((uint32_t*)$sp_real)[7]
    printf "CFSR = 0x%08x\n", *(uint32_t*)0xE000ED28
    printf "HFSR = 0x%08x\n", *(uint32_t*)0xE000ED2C
    printf "MMFAR= 0x%08x\n", *(uint32_t*)0xE000ED34
    printf "BFAR = 0x%08x\n", *(uint32_t*)0xE000ED38
end

# Layout for visual mode
layout src
focus cmd
```

Add a `.gdbinit` to your project root and a global one in `~/.gdbinit` for things like `set history save on` and `add-auto-load-safe-path /`.

## Essential Commands

### Navigating Code

```gdb
list main                 # show source around symbol
list file.c:42            # show source at file:line
disassemble main          # show disassembly
disassemble /m main       # mixed source + asm
disassemble 0x08001234, +32  # disassemble a range
```

### Breakpoints

```gdb
break main                          # symbolic
break file.c:42                     # source location
break *0x080012ae                   # raw address
break my_func if x > 5              # conditional
break my_func thread 2              # thread-scoped
tbreak main                         # temporary (one-shot)
hbreak fn                           # hardware breakpoint (forced; useful in flash)
info breakpoints
disable 2
enable 2
delete 2
clear file.c:42
```

Cortex-M has 6-8 hardware breakpoint slots (FPB) for flash. GDB will use software BKPT instructions in RAM regions automatically.

### Watchpoints

```gdb
watch g_counter         # break on write
rwatch g_state          # break on read
awatch g_flag           # break on any access
watch *0x20000010       # by address
info watchpoints
```

Limited (4 DWT comparators on M3/M4). The killer feature for "what is overwriting this variable" bugs.

### Stepping

```gdb
continue / c            # run until break
step / s                # step into (one source line)
next / n                # step over
stepi / si              # single instruction
nexti / ni              # one instruction, over calls
finish                  # run until current function returns
until 50                # run until line 50 in current frame
```

### Inspecting State

```gdb
info registers           # all CPU regs
info reg lr pc           # specific regs
p/x $sp                  # one reg in hex
p variable               # print variable
p/x variable             # in hex
p/t variable             # in binary
p *ptr                   # dereference
p arr[5]
p/x *(uint32_t*)0xE000ED28    # peek memory at address
x/16wx 0x20000000        # dump 16 words from address
x/16bx buf               # 16 bytes hex
x/s str_ptr              # null-terminated string
x/i $pc                  # disassemble at PC
display/x $r0            # show on every step
```

Format suffixes: `x` hex, `d` decimal, `u` unsigned, `t` binary, `o` octal, `f` float, `c` char, `s` string, `i` instruction. Size: `b` byte, `h` halfword, `w` word, `g` giant (8 bytes).

### Backtrace

```gdb
bt                  # full backtrace
bt full             # backtrace with locals at each frame
frame 3 / f 3       # switch to frame 3
info locals
info args
up / down           # move up/down frames
```

### Modifying State

```gdb
set $r0 = 42
set var x = 0
set *0x20000010 = 0xDEADBEEF
return                   # force the current function to return (skip its body)
return 7                 # ... and return value 7
jump *0x08001234         # set PC and continue
call my_func(5)          # call a function from the prompt
```

`call` is dangerous in embedded — it pushes a fake return frame. Useful for triggering DFU mode, calling a test routine, etc.

## `monitor` Commands

`monitor <cmd>` forwards to the GDB server's native command language (OpenOCD's TCL, J-Link's command set, etc.). Examples:

```gdb
monitor reset                 # system reset
monitor reset halt            # reset and stop at vector
monitor reset init            # reset, then run init script (clocks, etc.)
monitor halt
monitor resume
monitor flash erase_sector 0 0 0    # erase sector 0
monitor mww 0x40020014 0x12345678   # memory write word
monitor mdw 0x40020000 16           # memory dump 16 words
monitor cortex_m maskisr on   # mask IRQs while stepping
monitor adapter speed 4000    # SWD clock in kHz
```

Set `set remote interrupt-on-connect on` to ensure Ctrl+C halts a running target reliably.

## CMSIS-SVD Integration

CMSIS-SVD files describe every peripheral register and bit. Two integrations:

### VSCode + Cortex-Debug

The VSCode `cortex-debug` extension reads SVD and shows peripherals as a tree. Live view, click to drill into bitfields, edit values.

### `gdb-multiarch` + `peripherals`

pyOCD's GDB server supports `monitor peripherals` if loaded with `--pack`:

```bash
pyocd gdbserver -t stm32f407vg --pack Keil.STM32F4xx_DFP
```

```gdb
(gdb) monitor peripherals --show all
(gdb) monitor peripheral RCC.APB1ENR
```

OpenOCD doesn't have first-class SVD support but VSCode bridges it.

## Watching FreeRTOS

OpenOCD ships with a FreeRTOS "RTOS" plugin. Enable in your TCL config:

```tcl
$_TARGETNAME configure -rtos FreeRTOS
```

Now `info threads` shows all FreeRTOS tasks. Switch with `thread N`. GDB will see per-task stacks and `bt` shows the right one.

```gdb
(gdb) info threads
  Id   Target Id                  Frame
* 1    Thread 0x20001234 (Name: IDLE) 0x080012ae in idle_task ()
  2    Thread 0x20001500 (Name: blink) 0x08001500 in blink_task ()
  3    Thread 0x200016ff (Name: net)  0x08001abc in tcp_recv ()
```

J-Link's RTOS plugin (`-rtos /path/to/GDBServer/RTOSPlugin_FreeRTOS.so`) does the same.

## Python Scripting

GDB has Python embedded. Useful for custom decoders.

```python
# decode_event_queue.py
class EventQueue(gdb.Command):
    def __init__(self):
        super().__init__("eventq", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        head = int(gdb.parse_and_eval("event_queue.head"))
        tail = int(gdb.parse_and_eval("event_queue.tail"))
        cap  = int(gdb.parse_and_eval("EVENT_QUEUE_CAP"))
        print(f"queue: head={head} tail={tail} depth={(head-tail) % cap}")
        for i in range(tail, head):
            ev = gdb.parse_and_eval(f"event_queue.items[{i % cap}]")
            print(f"  {i}: {ev}")

EventQueue()
```

Source it: `(gdb) source decode_event_queue.py` then `(gdb) eventq`.

Pretty-printers turn raw structs into readable text — write one for your driver state, FreeRTOS task struct, ring buffer, etc.

## Common Workflows

### "Why did my code crash here?"

```
(gdb) continue
(gdb) ...HardFault hit...
(gdb) hardfault         # custom command from .gdbinit
PC = 0x080012ae
(gdb) list *0x080012ae
file.c:142: *ptr = 0;
```

### "What's setting this variable to garbage?"

```
(gdb) watch g_state
(gdb) continue
Hardware watchpoint 2: g_state
Old value = 1
New value = 0xDEADBEEF
(gdb) bt
#0  buggy_func at file.c:42
#1  caller     at other.c:8
```

### "Is my ISR firing?"

```
(gdb) break USART1_IRQHandler
(gdb) continue
Breakpoint 1, USART1_IRQHandler () at uart.c:25
(gdb) p/x USART1->SR
$1 = 0x00C0
```

### Logging Function Entries Without Halting

```gdb
break my_func
commands
    silent
    printf "my_func called with x=%d\n", x
    continue
end
```

GDB stops, prints, resumes. Cheap "printf-via-debugger".

### Time-Travel-Lite With Reverse-Step

If your GDB server supports record/replay (most don't on embedded; some Lauterbach/Segger setups with ETM trace do):

```gdb
(gdb) record
(gdb) continue
... crash ...
(gdb) reverse-step
(gdb) reverse-next
```

In practice on Cortex-M, you usually need ETM-trace-capable probes (J-Trace, Lauterbach) — not standard. For most projects, plain breakpoints + watchpoints are your tools.

## VSCode + Cortex-Debug Setup

Most-used GUI front-end for GDB on embedded. `launch.json`:

```json
{
    "type": "cortex-debug",
    "request": "launch",
    "name": "Debug STM32",
    "servertype": "openocd",
    "configFiles": ["interface/stlink.cfg", "target/stm32f4x.cfg"],
    "executable": "build/firmware.elf",
    "svdFile": "stm32f407.svd",
    "rtos": "FreeRTOS",
    "showDevDebugOutput": "none"
}
```

Get peripheral tree, watch panels, RTT integration, all on top of GDB.

## Common Pitfalls

### Pitfall 1: Stepping Through an ISR Hangs

Lower-priority interrupts pile up while you're stepping. `monitor cortex_m maskisr on` masks IRQs during single-step, but then your code's interrupt-driven behavior doesn't run. Pick your poison; both are debugging artifacts.

### Pitfall 2: `load` Doesn't Reset State

GDB writes flash but the CPU is still in some arbitrary halted state. Always `monitor reset halt` after `load` (the `hook-load` trick above automates this).

### Pitfall 3: Stale `.elf` After Rebuild

Forgetting to reload the ELF after rebuilding → GDB shows source for old addresses, breakpoints land in wrong places. `file firmware.elf` to reload symbols.

### Pitfall 4: Hardware Breakpoints Exhausted

GDB silently fails to break. `set remote hardware-breakpoint-limit 4` so GDB warns when you're out. Or move target code to RAM where software BPs work.

### Pitfall 5: `info threads` Empty With FreeRTOS

OpenOCD's RTOS detection looks at specific symbols (`pxCurrentTCB`, etc.). Strip those out with `-fdata-sections` GC and threads disappear. Add `KEEP(pxCurrentTCB)` in your linker script, or build without aggressive GC for debug builds.

### Pitfall 6: SWD Clock Too Fast for Long Wires

GDB connect succeeds but `monitor reset` hangs or `info reg` returns garbage. `monitor adapter speed 1000` (1 MHz).

### Pitfall 7: Optimized Code Confuses GDB

`-O2` reorders, inlines, eliminates locals. "Variable 'x' is optimized out". Use `-Og` (optimize for debug) or `-O0` for debug builds. Or accept the limitation and read assembly.

### Pitfall 8: Auto-Loading `.gdbinit` Blocked

Recent GDB refuses to source per-project `.gdbinit` unless allowed. Add to `~/.gdbinit`:

```gdb
add-auto-load-safe-path /home/me/projects
```

## TUI Mode

```
gdb -tui firmware.elf
```

or in GDB:
```gdb
(gdb) layout src
(gdb) layout asm
(gdb) layout split
(gdb) layout regs
(gdb) focus cmd
Ctrl+L            redraw
Ctrl+X o          switch focused window
```

For light debugging, TUI beats VSCode. For heavy debugging, VSCode wins.

## Summary

1. **GDB drives the probe through a GDB server** (OpenOCD/pyOCD/J-Link/probe-rs).
2. **`monitor` commands** = pass-through to the probe's native command set.
3. **`.gdbinit`** captures your project's connect + helper commands.
4. **`watch` / `awatch`** is the killer feature for memory-corruption bugs.
5. **Hardware breakpoints** in flash; software in RAM.
6. **Build with `-Og -g3`** for debug; `-O2` confuses GDB.
7. **FreeRTOS thread awareness** needs an RTOS plugin in the GDB server.
8. **VSCode + Cortex-Debug** is the GUI option; `tui` is the terminal one.
9. **Reset after load**: `monitor reset halt` (hook to automate).
10. **`call`, `return`, `jump`** can patch the CPU state on the fly — handy but easy to break.

## See Also

- [JTAG/SWD](jtag_swd.md) — the wire layer underneath
- [HardFault Debugging](hardfault_debugging.md) — what to do once GDB stops you
- [RTT/Semihosting](rtt_semihosting.md) — logging that pairs with debugging
- [Debugging](debugging.md) — broader strategies

## Where this connects

- [JTAG/SWD](jtag_swd.md) — the wire protocols GDB drives via a probe
- [RTT, semihosting & SWO](rtt_semihosting.md) — alternative debug I/O channels
- [HardFault debugging](hardfault_debugging.md) — post-mortem analysis in GDB
- [Debugging](debugging.md) — broader embedded debugging techniques
- [CMSIS](cmsis.md) — SVD/register views GDB uses
- [Startup code](startup_code.md) — what to step through before main()
