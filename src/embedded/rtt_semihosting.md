# RTT, Semihosting, and ITM/SWO

## Overview

When something goes wrong on an MCU, the first instinct is `printf`. But "regular" printf needs a UART, which means pins, a UART peripheral, baud config, and a host serial terminal. For interrupt-heavy or fault-state code, UART printf is also too slow and not safe (it busy-waits, often inside a critical section).

Cortex-M provides three lower-overhead alternatives that go through the debug probe instead of a UART:

| Mechanism | Bandwidth | Wires | Needs probe | Works after fault? |
|-----------|-----------|-------|-------------|---------------------|
| **Semihosting** | Slow (CPU halts every call) | SWD only | Yes | No (halts CPU) |
| **ITM / SWO** | Moderate (~1 MHz pin-rate) | SWD + SWO pin | Yes | Yes (one-way) |
| **RTT** | Fast (multi-MB/s) | SWD only | Yes | Yes |

Plus, of course, plain UART is still always available and survives without a probe.

## Semihosting

Semihosting is an ARM ABI where the target halts the CPU with a `BKPT #0xAB` instruction, and the host debugger handles the request (printf, file I/O, time, etc.) by reading registers and memory.

```c
#include <stdio.h>

int main(void) {
    printf("hello from semihosting\n");
}
```

To make stdio work via semihosting, link with `-specs=rdimon.specs` and call `initialise_monitor_handles()` once. Then printf to stdout shows up in the OpenOCD/GDB console.

**Pros**: trivially simple, no UART needed, works with the libc you already have.
**Cons**:
- **CPU halts on every call**. A printf in an ISR halts the system for milliseconds.
- **Useless without a debugger attached.** If the probe is disconnected, every semihosting call hangs the CPU forever.
- **Slow.** Single-digit Hz throughput practical.

**Use semihosting for**: one-shot debugging, never in production, never in code that ships unattended.

```c
// To make sure code doesn't hang if the probe disconnects:
#define BKPT_NOP_IF_NO_DEBUGGER \
    if (!(CoreDebug->DHCSR & CoreDebug_DHCSR_C_DEBUGEN_Msk)) return;
```

## ITM / SWO

ARM's **Instrumentation Trace Macrocell** is a CoreSight peripheral that streams data out the **SWO pin** as part of the TPIU (Trace Port Interface Unit). The probe captures SWO and forwards it to the host.

There are 32 "stimulus ports". Software writes to a memory-mapped register; the ITM serializes and emits.

```c
#define ITM_CHANNEL_PRINTF 0

void itm_putchar(uint8_t c) {
    if ((ITM->TCR & ITM_TCR_ITMENA_Msk) &&     // ITM enabled
        (ITM->TER & (1 << ITM_CHANNEL_PRINTF))) {
        while (ITM->PORT[ITM_CHANNEL_PRINTF].u32 == 0) {}  // wait for room
        ITM->PORT[ITM_CHANNEL_PRINTF].u8 = c;
    }
}

// Redirect printf
int _write(int fd, const char* buf, int len) {
    for (int i = 0; i < len; i++) itm_putchar(buf[i]);
    return len;
}
```

Enable beforehand:

```c
// Configure SWO pin output (varies by chip; STM32 needs DBGMCU register)
DBGMCU->CR |= DBGMCU_CR_TRACE_IOEN;     // enable trace pins

// Configure TPIU for SWO Manchester or NRZ (probe-dependent)
TPI->ACPR = (HCLK_HZ / 2000000) - 1;    // SWO baud = 2 MHz
TPI->SPPR = 2;                          // NRZ
TPI->FFCR = 0x100;                      // no formatter

// Enable ITM
ITM->LAR = 0xC5ACCE55;                  // unlock
ITM->TCR = ITM_TCR_TraceBusID_Msk | ITM_TCR_SWOENA_Msk
         | ITM_TCR_SYNCENA_Msk    | ITM_TCR_ITMENA_Msk;
ITM->TER = 0xFFFFFFFF;                  // enable all ports
ITM->TPR = 0;                           // privilege: all ports user
```

Bandwidth: the SWO pin runs at up to a few MHz, giving you a few hundred KB/s usable. Plenty for log lines.

**Probe support**:
- J-Link: SWO Viewer / SystemView read ITM transparently.
- ST-Link: STM32CubeIDE / OpenOCD `itm port 0 on`.
- pyOCD: `pyocd commander` with SWO.

**Pros**: doesn't halt CPU, works during normal execution and even in ISRs, multi-channel.
**Cons**: needs SWO pin routed, baud has to match HCLK config, target hardware setup is fiddly, only one-way (target → host).

## RTT (SEGGER's Real-Time Transfer)

RTT shoves data through **SWD memory reads** at the probe level — no SWO pin needed. The MCU writes to a ring buffer in RAM at a known address; the probe periodically reads that RAM via the debug interface (which doesn't halt the CPU on M3/M4/M7).

```
   Target RAM                          Host
   ┌──────────────────┐
   │ "SEGGER RTT"     │ ◄── known magic string at start of control block
   │ Up buffers (T→H) │
   │  ┌─────────────┐ │       SWD reads via DAP
   │  │ ring buffer │ │ ◄────────────────────── J-Link / probe
   │  └─────────────┘ │
   │ Down buffers     │       (host can also write via DAP)
   │ (H→T)            │ ────────────────────────► probe writes
   └──────────────────┘
```

The probe scans RAM at startup looking for the `"SEGGER RTT"` magic, then knows where the buffers are.

### Why It's Fast

- No CPU halts. The probe reads RAM in the background while the MCU runs.
- No special pins (just SWD).
- Multi-channel up/down (default: 16 buffers each way).
- Multi-MB/s in practice.

### Minimal Integration

SEGGER provides the source as a few drop-in C files. Add `SEGGER_RTT.c`, `SEGGER_RTT.h`, `SEGGER_RTT_Conf.h`.

```c
#include "SEGGER_RTT.h"

int main(void) {
    SEGGER_RTT_Init();
    SEGGER_RTT_printf(0, "boot ok @ %lu\n", HAL_GetTick());

    while (1) {
        SEGGER_RTT_printf(0, "tick %lu\n", HAL_GetTick());

        // Read input from host (e.g., "menu", "reset", ...)
        char ch;
        if (SEGGER_RTT_Read(0, &ch, 1) > 0) {
            handle_command(ch);
        }
        HAL_Delay(1000);
    }
}
```

Host side:
```bash
JLinkRTTClient                # text terminal
JLinkRTTLogger -t 0           # raw capture to file
```

OpenOCD also supports RTT (`rtt server start 9090 0`), as does probe-rs (`probe-rs rtt`).

### Multi-Channel Use

Channel 0 = main log. Channel 1 = high-rate telemetry. Channel 2 = trace events. Channels can be read independently by separate host tools.

```c
SEGGER_RTT_printf(0, "log: %s\n", msg);
SEGGER_RTT_Write(1, packet, packet_len);    // binary telemetry
SEGGER_RTT_printf(2, "tag:irq_in:%lu\n", DWT->CYCCNT);
```

### Limitation

RTT uses the probe's debug interface to read RAM **while the CPU is running**. On Cortex-M0/M0+ this is slow (every access stops the CPU briefly). On M3/M4/M7 the DAP can read AHB without stopping. So **RTT is fast on M3+, slower on M0/M0+**.

## SystemView

A SEGGER tool built on top of RTT. The target instruments key RTOS events (task switch, IRQ enter/exit, semaphore take/give, etc.). The host visualizes the timeline.

```
Task A  ──────████████──┐                ┌────████──
                         │                │
IRQ                      └──████──────────┘
                                  │
Task B  ──██──────────────────────┴████████──────
```

Combined with `DWT->CYCCNT` timestamps, this gives you a Tracealyzer-style view of system behavior. Free for non-commercial use.

Hook it up by replacing FreeRTOS trace macros with SystemView's:

```c
#define traceTASK_SWITCHED_IN()  SEGGER_SYSVIEW_OnTaskStartExec(...)
```

## When to Use Each

| Goal | Tool |
|------|------|
| Quick `printf("%d\n", x)` while debugging | RTT |
| Log lines without halting | ITM or RTT |
| Stream MB/s of binary telemetry | RTT |
| Visualize RTOS task scheduling | SystemView (RTT-based) |
| Read/write files on host | Semihosting |
| Production logging (no probe) | UART or storage |
| Logging after a fault | RTT (probe reads RAM regardless of CPU state) |

## Common Pitfalls

### Pitfall 1: Semihosting Hangs Without Debugger

Production code accidentally left with `printf` linked to semihosting. Without a probe, every `printf` is a `BKPT` → infinite halt. Always test the release build without the probe attached.

### Pitfall 2: SWO Baud Mismatch

You changed HCLK from 84 MHz to 168 MHz but `TPI->ACPR` still divides for the old clock. Result: garbled SWO output. Recompute when clocks change.

### Pitfall 3: SWO Pin Stolen

Application code reconfigures the SWO pin (often PB3) as GPIO. ITM still emits, but the data goes nowhere. Reserve PB3 in your pin config.

### Pitfall 4: RTT Buffer Too Small

Default config has small (~1 KB) up buffer. Bursty `printf` overflows it and characters drop silently. Bump `BUFFER_SIZE_UP` in `SEGGER_RTT_Conf.h` for high-rate logging.

### Pitfall 5: RTT in a Fault Handler

RTT writes are RAM writes — safe in a fault handler. But if you `printf`, the underlying libc may use locks or call `malloc`. Use `SEGGER_RTT_Write()` raw with no formatting in the fault path.

### Pitfall 6: ITM Locked After Reset

Some chips power-gate the ITM block. If `ITM->TCR` reads back zero after init, check that `CoreDebug->DEMCR & TRCENA` is set first.

```c
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
```

### Pitfall 7: SystemView Buffer Stall

SystemView is **lossy**: if the host can't keep up, events get dropped. For dense tracing, increase the buffer or sample-trace only when needed.

### Pitfall 8: Mixing RTT Channels

Reading channel 0 with `JLinkRTTClient` while a second instance reads channel 0 → conflict. Use one tool per channel, or use multi-channel-aware clients.

## Quick Setup Cheat Sheet

```c
// --- RTT (fastest path to printf) ---
#include "SEGGER_RTT.h"
SEGGER_RTT_printf(0, "%lu: %s\n", now(), msg);

// --- ITM (one-line printf via SWO) ---
ITM_SendChar(c);

// --- Semihosting (only when desperate) ---
#include <stdio.h>
printf("hello\n");   // link with -specs=rdimon.specs

// --- Production: UART fallback ---
HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100);
```

## Summary

1. **RTT = default modern choice**: fast, no extra pins, multi-channel, works in faults.
2. **ITM/SWO** is the ARM standard; useful when you can't use RTT (probe that doesn't speak it).
3. **Semihosting** is small but halts the CPU and hangs without a probe — debug only.
4. **CoreDebug->DEMCR_TRCENA must be set** to enable ITM/DWT.
5. **Watch SWO baud** when changing system clocks.
6. **SystemView** turns RTT into an RTOS timeline visualizer.
7. **UART** is the only thing that works in the field without a probe — keep a path to it.

## See Also

- [Debugging](debugging.md)
- [JTAG/SWD](jtag_swd.md) — CoreSight, ITM block
- [HardFault Debugging](hardfault_debugging.md) — RTT survives faults
- [Interrupts](interrupts.md) — DWT, CYCCNT, vector table
