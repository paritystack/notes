# JTAG and SWD

## Overview

JTAG and SWD are the two wire-level protocols used to talk to the **debug logic** baked into virtually every MCU. Through them, a debug probe (J-Link, ST-Link, CMSIS-DAP, Black Magic Probe) can:

- Halt the CPU, single-step, set hardware breakpoints
- Read/write CPU registers, memory, peripheral registers
- Program flash
- Stream trace data
- Reset the chip in various ways

```
         Host                            Target MCU
   ┌────────────────┐                ┌──────────────────┐
   │  GDB / IDE     │                │     Cortex-M     │
   │       │        │                │       │          │
   │       ▼        │  USB / Eth     │       ▼          │
   │  pyOCD/OpenOCD ├───────────────►│  ┌─────────┐     │
   │  / J-Link srv  │                │  │   DAP   │ ◄── 4-5 wires
   │                │                │  └─────────┘     │
   └────────────────┘                │       │          │
                                     │       ▼          │
                                     │   CPU + buses    │
                                     └──────────────────┘
```

JTAG is older, more wires (4-5), supports daisy-chaining. SWD is the ARM-specific 2-wire alternative used on nearly every Cortex-M chip.

## Pin Comparison

| Signal | JTAG | SWD | Direction |
|--------|------|-----|-----------|
| **Clock** | TCK | SWCLK | host → target |
| **Data** | TDI (in), TDO (out) | SWDIO (bidirectional) | both |
| **State** | TMS (state machine) | (multiplexed onto SWDIO) | host → target |
| **Reset** | nTRST (optional) | (none, uses nRST line) | host → target |
| **System reset** | nRESET | nRESET | host → target |
| **Trace** | TDO (BSCAN trace) | SWO (single-wire output) | target → host |

JTAG = TCK + TMS + TDI + TDO + (TRST). SWD = SWCLK + SWDIO + (SWO for trace) + GND + VTREF.

On a Cortex-M chip, the **same physical pins** can do either — the JTAG-DP (Debug Port) auto-detects the protocol from a magic sequence at the start of a session.

## JTAG: The TAP State Machine

JTAG was originally a boundary-scan protocol (test pins on a chip without removing it from the PCB). Every JTAG device has a **TAP controller** — a 16-state finite state machine driven by TMS edges sampled on TCK.

```
         ┌──────────────┐
   ┌────►│  Test-Logic- │
   │     │    Reset     │
   │     └──────┬───────┘
   │            │ TMS=0
   │            ▼
   │     ┌──────────────┐         ┌──────────────┐
   │     │  Run-Test/   │ TMS=1   │   Select-    │
   │     │     Idle     ├────────►│    DR-Scan   │
   │     └──────────────┘         └──────┬───────┘
   │                                     │ ...
   │           DR path: shift data       │
   │           IR path: shift instruction│
   └─────────────────────────────────────┘
```

Two paths matter: **DR-Scan** (Data Register, used to shift in/out data) and **IR-Scan** (Instruction Register, used to choose which DR is active). Higher-level tools take care of the state-machine walk; you usually don't poke TMS by hand.

**Daisy chaining**: multiple JTAG devices on one bus, TDO of one feeds TDI of the next. The host issues IRs sized to span all devices and uses BYPASS (a 1-bit DR) to skip those it doesn't want to talk to. Rarely used on single-MCU boards.

## SWD: The Two-Wire Alternative

ARM created SWD to save pins. Same debug logic underneath; different wire format.

### Frame Format

Each SWD operation is a **request** (8 bits, host → target), **ack** (3 bits, target → host), then **data** (33 bits including parity).

```
Bits 0-7 (host):     1  APnDP  RnW  A2  A3  parity  0  1
Bits 8-10 (target):  OK / WAIT / FAULT
Bits 11-44 (data):   32 data bits + parity
```

| Field | Meaning |
|-------|---------|
| APnDP | 0 = access DP register, 1 = access AP register |
| RnW   | 0 = write, 1 = read |
| A2,A3 | register address bits (4 registers per DP/AP) |
| Ack   | `001` OK, `010` WAIT (retry), `100` FAULT (error sticky) |

You almost never look at this byte-stream — the probe firmware does. But if you're writing a debug probe or bit-banging SWD on a spare MCU, this is what you implement.

### Protocol Switch (JTAG → SWD)

The DP starts in a "dormant" or JTAG state. To switch it to SWD, send a documented magic sequence: 50 SWCLK cycles with SWDIO high (line reset), then the 16-bit token `0xE79E`, then 50 more cycles high, then 1-cycle low to start operations.

OpenOCD/pyOCD do this transparently — but it's why you sometimes need to power-cycle the target after a failed connect.

## Debug Architecture: DAP → AP → MEM-AP

ARM CoreSight (the debug subsystem in Cortex-M) is layered:

```
              Host / probe
                   │
                   ▼  JTAG or SWD wires
              ┌─────────┐
              │   DP    │  Debug Port
              │ (SWJ-DP)│  (entry point — speaks SWD or JTAG)
              └────┬────┘
                   │  internal AP bus
        ┌──────────┼──────────┬─────────────┐
        ▼          ▼          ▼             ▼
    ┌───────┐  ┌───────┐  ┌───────┐   ┌──────────┐
    │ AHB-AP│  │ APB-AP│  │ JTAG- │   │  custom  │
    │       │  │       │  │  AP   │   │   APs    │
    └───┬───┘  └───┬───┘  └───────┘   └──────────┘
        │          │
        ▼          ▼
   System bus   Private peripheral bus
   (memory,     (DWT, ITM, FPB, ...)
    Flash,
    SRAM, ...)
```

- **DP (Debug Port)** is the protocol endpoint: SWJ-DP can switch between JTAG and SWD.
- **APs (Access Ports)** sit behind the DP. The most important is the **AHB-AP** (a.k.a. MEM-AP): it gives the host a memory-mapped view of the target — read/write any address as if you were the CPU.
- Other APs expose private buses (APB-AP for CoreSight components like ITM/DWT/FPB).

This is why a debugger can poke `SCB->CFSR` from your laptop without involving any code on the MCU: the AHB-AP just reads address `0xE000ED28` for you.

## CoreSight Components Worth Knowing

Inside Cortex-M, several CoreSight blocks support debug/trace:

| Block | Purpose |
|-------|---------|
| **FPB** (Flash Patch & Breakpoint) | Hardware breakpoints in flash (6-8 on M3/M4) |
| **DWT** (Data Watchpoint & Trace) | Hardware watchpoints, cycle counter, exception trace |
| **ITM** (Instrumentation Trace Macrocell) | Lightweight `printf` channel via SWO |
| **TPIU** (Trace Port Interface Unit) | Serializes ITM/ETM trace out SWO or parallel pins |
| **ETM** (Embedded Trace Macrocell) | Full instruction trace (premium, often only on dev boards) |

`DWT->CYCCNT` (cycle counter) is the cheapest profiler available — enable once, read at start/end of code section.

```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;
DWT->CYCCNT       = 0;
do_thing();
printf("cycles: %lu\n", DWT->CYCCNT);
```

## Common Debug Probes

| Probe | Source | Notes |
|-------|--------|-------|
| **J-Link** (SEGGER) | Commercial | Fastest, best supported, free for non-commercial use (EDU model). Has its own RTT, SystemView. |
| **ST-Link/V2 / V3** | ST | Free with every Nucleo / Disco board. Can be reflashed to J-Link with SEGGER's tool. |
| **CMSIS-DAP** | Open | ARM's reference design. Any CMSIS-DAP probe works with any CMSIS-DAP host (OpenOCD, pyOCD). DAPLink (mbed) is the most common implementation. |
| **Black Magic Probe** | Open | Embedded GDB server — connect GDB directly, no OpenOCD needed. |
| **PicoProbe** | Open | RP2040-based, free if you have a spare Pico. |

For SWD-only Cortex-M, any of them works. For JTAG with custom IR sequences (FPGA, non-ARM), J-Link or a dedicated tool.

## Host-Side Tools

### OpenOCD

Open-source GDB server. Speaks to most probes. Configured with TCL scripts.

```bash
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg
# In another terminal:
arm-none-eabi-gdb firmware.elf
(gdb) target extended-remote :3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

### pyOCD

Python-native, works with CMSIS-DAP and ST-Link out of the box. Simpler config than OpenOCD if your chip is supported.

```bash
pyocd flash -t stm32f407vg firmware.elf
pyocd gdbserver -t stm32f407vg
```

### J-Link tools

`JLinkExe`, `JLinkGDBServer`, `JLinkRTTClient`. Highest performance, especially for streaming RTT logs.

```bash
JLinkGDBServer -device STM32F407VG -if SWD -speed 4000
```

### Probe-RS

Rust-native, fast, good for CI flashing. Drop-in for pyOCD/OpenOCD in many cases.

## Reset Types

Three flavors, often confused:

| Type | What | When |
|------|------|------|
| **System reset** | Resets CPU and most peripherals (not debug). Equivalent to nRST pin. | Normal "reset and run". |
| **Core reset** | Resets only the CPU. Peripherals keep state. | Sometimes useful when peripherals shouldn't be re-initialized. |
| **Connect under reset** | Probe asserts nRST then connects to JTAG/SWD. | The only reliable way to attach to a chip running low-power code that disables debug pins. |

`monitor reset halt` in GDB performs a system reset and halts the CPU on the first instruction (before main).

## Debug-Survives-Sleep (or Doesn't)

In low-power modes, the debug interface can lose contact:

- STM32 has `DBGMCU_CR` bits to keep debug alive in STOP and STANDBY modes (`DBG_STOP`, `DBG_STANDBY`).
- nRF52 has similar `DBGCTRL` settings.
- Set these in debug builds only — they raise sleep current significantly.

```c
DBGMCU->CR |= DBGMCU_CR_DBG_SLEEP
           |  DBGMCU_CR_DBG_STOP
           |  DBGMCU_CR_DBG_STANDBY;
```

## Hardware Breakpoints vs Software Breakpoints

- **Hardware (FPB)**: 6-8 slots on Cortex-M3/M4, set by writing comparator registers. Required when running from read-only flash. GDB picks these automatically when the PC is in a non-RAM region.
- **Software**: Replace the instruction at the breakpoint address with `BKPT #0`. Only works in RAM (since flash is read-only). GDB prefers these when the target is in RAM.

Run out of hardware breakpoints? Symptom: GDB silently fails to stop. Use `info breakpoints`, free some, or move the code to RAM.

## Watchpoints (DWT)

Hardware data breakpoints — stop when an address is read, written, or accessed. Limited (typically 4 comparators on M3/M4).

```
(gdb) watch g_buffer        # write watchpoint
(gdb) rwatch g_flag         # read watchpoint
(gdb) awatch g_state        # any-access watchpoint
```

Brilliant for catching "who is overwriting this variable" bugs.

## Common Pitfalls

### Pitfall 1: SWD Pins Reused as GPIO

Application code reconfigures PA13/PA14 as regular GPIO, killing SWD. Connection dies after the app starts. Fix: **connect under reset**, then erase the app, or reserve SWD pins.

### Pitfall 2: VTREF Not Connected

The probe's level shifters need VTREF to know the target's I/O voltage. Without it, SWDIO floats. Symptom: probe reports "no target detected" even though it's wired correctly.

### Pitfall 3: Clock Too Fast for the Wires

SWD up to ~10 MHz is fine on a clean board. Long jumper wires, weak ground, or breadboards → drop to 1 MHz. OpenOCD: `adapter speed 1000`.

### Pitfall 4: Wrong Reset Type

Some chips need "connect under reset" because boot code disables SWD pins. If `monitor reset halt` fails, try the probe's "connect under reset" mode.

### Pitfall 5: Stale Flash After Erase

Flash programming has subtle race conditions with the cache. `monitor reset halt` after a flash load is the safe default; `monitor reset` alone may leave I-cache stale.

### Pitfall 6: Out of Hardware Breakpoints

GDB sets a hidden BP for `main`, `_start`, and step-over. Combined with user BPs, the 6 FPB slots fill quickly. `monitor cortex_m maskisr on` and `set remote hardware-breakpoint-limit 4` help.

### Pitfall 7: ST-Link Firmware Mismatch

ST-Link probe firmware and OpenOCD/STM32CubeProgrammer go out of sync. Update via `STLinkUpgrade` tool when "unknown device" or hang on connect.

## Quick Reference: GDB Cheats

```gdb
target extended-remote :3333    # connect to OpenOCD
monitor reset halt              # reset + halt at vector
load                            # flash the .elf
break main                      # set BP
continue / c
step / s                        # step one source line, into calls
next / n                        # step over calls
stepi / si                      # one instruction
info reg                        # all CPU registers
p/x *(uint32_t*)0xE000ED28      # peek any address
x/16wx 0x20000000               # dump 16 words from SRAM
watch *0x20000004               # watchpoint on write
monitor mww 0x40020014 0x1234   # poke a peripheral register
```

## Summary

1. **JTAG** = 4-5 wires, daisy-chainable, state-machine driven.
2. **SWD** = 2 wires (plus SWO for trace), Cortex-M default.
3. Both reach the same **CoreSight DAP** underneath.
4. **AHB-AP** is what lets the host read/write any memory address without target software help.
5. **FPB + DWT + ITM + TPIU** are the on-chip debug/trace blocks worth knowing.
6. **Connect under reset** when the app disables SWD pins.
7. **Cycle counter (`DWT->CYCCNT`)** is the cheapest profiler.
8. **Keep VTREF connected** and **drop clock speed on flaky wiring**.
9. **Watchpoints** are the killer feature for chasing memory corruption.

## See Also

- [Debugging](debugging.md) — broader debugging strategies
- [HardFault Debugging](hardfault_debugging.md) — using the debugger after a fault
- [Interrupts](interrupts.md) — vector table relocation and debug
