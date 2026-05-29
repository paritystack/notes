# HardFault Debugging on Cortex-M

## Overview

When a Cortex-M CPU executes an illegal instruction, dereferences a bad pointer, divides by zero with the trap enabled, or hits an unrecoverable bus error, it takes an **exception**. If the exception itself can't be handled — or if a fault occurs inside a fault handler — the CPU escalates to **HardFault**.

The classic symptom: your firmware is stuck in `HardFault_Handler` (which usually defaults to `while (1);`). The good news: Cortex-M preserves enough state for you to figure out **which instruction faulted**, **why**, and **with what registers**.

## Fault Hierarchy

Cortex-M3/M4/M7 split the fault sources across four handlers (M0/M0+ only has HardFault):

| Handler | Triggered By |
|---------|-------------|
| **MemManage_Handler** | MPU violations, exec from non-X region, stacking errors with MPU on |
| **BusFault_Handler** | Bad address (no slave at that bus location), wrong size, imprecise bus errors |
| **UsageFault_Handler** | Undefined instruction, divide by zero, unaligned access (if trap-on enabled), invalid EPSR state |
| **HardFault_Handler** | Any of the above when their handler is disabled, **OR** a fault inside a fault handler, **OR** vector fetch failed |

By default after reset, only `HardFault_Handler` is enabled — everything escalates. **Always enable the specific handlers during debug**:

```c
// Enable the fine-grained handlers so faults don't all show up as HardFault
SCB->SHCSR |= SCB_SHCSR_USGFAULTENA_Msk
            | SCB_SHCSR_BUSFAULTENA_Msk
            | SCB_SHCSR_MEMFAULTENA_Msk;

// Enable divide-by-zero and unaligned traps
SCB->CCR  |= SCB_CCR_DIV_0_TRP_Msk
           | SCB_CCR_UNALIGN_TRP_Msk;
```

Now a divide-by-zero takes you straight to UsageFault, not HardFault, and the CFSR bits below tell you exactly what happened.

## The Stack Frame at Fault Entry

On exception entry, Cortex-M auto-pushes 8 (or 26 with FPU) registers onto the stack that was in use when the fault hit. The handler can decode them.

```
                  Stack frame layout (basic, no FPU)
       
       SP (after entry) ──► ┌────────────┐
                             │   R0       │   ← arg0 / scratch
                             ├────────────┤
                             │   R1       │
                             ├────────────┤
                             │   R2       │
                             ├────────────┤
                             │   R3       │
                             ├────────────┤
                             │   R12      │
                             ├────────────┤
                             │   LR       │   ← return address of the caller
                             ├────────────┤
                             │   PC       │   ◄── INSTRUCTION THAT FAULTED
                             ├────────────┤
                             │   xPSR     │
                             └────────────┘
```

The **PC** in that frame is the gold: it's the address of (approximately) the instruction that triggered the fault. Pointing addr2line / `arm-none-eabi-addr2line` at it gives the source line.

```bash
arm-none-eabi-addr2line -e firmware.elf 0x080012ae
# → src/sensor.c:142
```

## EXC_RETURN: Which Stack Was Active?

When the fault fires, the CPU might have been on either MSP (handler / main) or PSP (thread / RTOS task). The LR register on entry is loaded with a special **EXC_RETURN** value that tells you.

| EXC_RETURN | Meaning |
|------------|---------|
| `0xFFFFFFF1` | Handler mode, MSP, basic frame |
| `0xFFFFFFF9` | Thread mode, MSP, basic frame |
| `0xFFFFFFFD` | Thread mode, **PSP**, basic frame |
| `0xFFFFFFE1/E9/ED` | Same as above but with FPU extended frame |

Bit 2 (`0x4`) of EXC_RETURN: 0 = MSP, 1 = PSP. This decides which SP to dereference to find the auto-pushed frame.

## A Diagnostic Fault Handler

This is the workhorse pattern: a HardFault handler that grabs the offending SP and either jumps to C code to print the state or just spins so the debugger can attach.

```c
__attribute__((naked)) void HardFault_Handler(void) {
    __asm volatile (
        "tst lr, #4              \n"   // bit 2 of EXC_RETURN: 1 = PSP
        "ite eq                  \n"
        "mrseq r0, msp           \n"
        "mrsne r0, psp           \n"
        "mov r1, lr              \n"   // pass EXC_RETURN too
        "b hardfault_report      \n"
    );
}

typedef struct {
    uint32_t r0, r1, r2, r3, r12;
    uint32_t lr;          // caller's LR
    uint32_t pc;          // faulting instruction
    uint32_t psr;
} cm_stack_frame_t;

void hardfault_report(cm_stack_frame_t* frame, uint32_t exc_return) {
    uint32_t cfsr  = SCB->CFSR;
    uint32_t hfsr  = SCB->HFSR;
    uint32_t mmfar = SCB->MMFAR;
    uint32_t bfar  = SCB->BFAR;

    printf("\n*** HARDFAULT ***\n");
    printf("PC    = 0x%08lx\n", frame->pc);
    printf("LR    = 0x%08lx\n", frame->lr);
    printf("PSR   = 0x%08lx\n", frame->psr);
    printf("R0-R3 = %08lx %08lx %08lx %08lx\n",
           frame->r0, frame->r1, frame->r2, frame->r3);
    printf("R12   = 0x%08lx\n", frame->r12);
    printf("CFSR  = 0x%08lx\n", cfsr);
    printf("HFSR  = 0x%08lx\n", hfsr);
    printf("MMFAR = 0x%08lx (valid=%d)\n", mmfar, !!(cfsr & (1<<7)));
    printf("BFAR  = 0x%08lx (valid=%d)\n", bfar,  !!(cfsr & (1<<15)));
    printf("EXC_RETURN = 0x%08lx (sp was %s)\n",
           exc_return, (exc_return & 4) ? "PSP" : "MSP");

    while (1) { __BKPT(0); }
}
```

In a release build, replace `printf` with writing the frame to a backup register or NVM region, then reset — so the next boot can report the previous crash.

## Decoding CFSR / HFSR / MMFAR / BFAR

The **Configurable Fault Status Register** (CFSR @ 0xE000ED28) is three sub-registers packed into 32 bits.

```
 31         16 15          8 7          0
┌─────────────┬─────────────┬─────────────┐
│   UFSR      │    BFSR     │    MMFSR    │
│ (UsageFault)│ (BusFault)  │ (MemManage) │
└─────────────┴─────────────┴─────────────┘
```

### MMFSR (bits 0-7) — MemManage Fault Status

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | IACCVIOL | Instruction access violation (exec from non-X region) |
| 1 | DACCVIOL | Data access violation (read/write to forbidden region) |
| 3 | MUNSTKERR | Fault during exception return unstacking |
| 4 | MSTKERR | Fault during exception entry stacking |
| 5 | MLSPERR | Lazy FPU state preservation failed |
| 7 | MMARVALID | **MMFAR is valid** — address is the offender |

### BFSR (bits 8-15) — Bus Fault Status

| Bit | Name | Meaning |
|-----|------|---------|
| 8 | IBUSERR | Bus error fetching instruction |
| 9 | PRECISERR | **Precise** data bus error — BFAR points at the bad address |
| 10 | IMPRECISERR | Imprecise data bus error (write buffer drained later) |
| 11 | UNSTKERR | Bus fault on exception return unstack |
| 12 | STKERR | Bus fault on exception entry stack |
| 13 | LSPERR | Lazy FPU bus error |
| 15 | BFARVALID | **BFAR is valid** |

### UFSR (bits 16-31) — Usage Fault Status

| Bit | Name | Meaning |
|-----|------|---------|
| 16 | UNDEFINSTR | Undefined instruction |
| 17 | INVSTATE | Invalid EPSR (e.g., tried to enter ARM state on Cortex-M) |
| 18 | INVPC | Invalid PC load (bad EXC_RETURN, return to non-thumb addr) |
| 19 | NOCP | Coprocessor access not supported (often: tried FP without enabling FPU) |
| 24 | UNALIGNED | Unaligned access with UNALIGN_TRP enabled |
| 25 | DIVBYZERO | Divide by zero with DIV_0_TRP enabled |

### HFSR (HardFault Status, 0xE000ED2C)

| Bit | Meaning |
|-----|---------|
| 1 | VECTTBL — fault reading vector table (bad VTOR? Flash unmapped?) |
| 30 | FORCED — escalated from another fault (look at CFSR) |
| 31 | DEBUGEVT — debug event |

**Workflow:** start at HFSR. If FORCED=1, read CFSR to find the original culprit. Then read MMFAR or BFAR if the matching VALID bit is set.

## Common Faults and What They Look Like

### Null Pointer Write

```c
uint32_t* p = NULL;
*p = 42;
```

- BFSR: `PRECISERR=1`, `BFARVALID=1`
- BFAR: `0x00000000`
- PC: address of the store instruction

### Stack Overflow Into MSP Underflow

Task stack overflows into another's memory, or MSP rolls into a region without RAM.

- Usually shows up as MMFSR `MSTKERR` during a later exception entry, or
- Random data corruption preceding a HardFault elsewhere.

Use an MPU region as a stack guard band, or enable the H7's **stack limit check** (`PSPLIM` register, ARMv8-M).

### Execution From RAM Without XN Off

Trying to run code from a region the MPU has marked execute-never.

- MMFSR `IACCVIOL=1`
- PC = address you tried to execute from

### FPU Used Without Enabling It

Cortex-M4F has a hardware FPU but it's disabled at reset.

- UFSR `NOCP=1`
- PC = the first FPU instruction the compiler emitted

```c
// Enable CP10 and CP11 (FPU)
SCB->CPACR |= ((3UL << 10*2) | (3UL << 11*2));
__DSB(); __ISB();
```

### Returning From an ISR With a Corrupted LR

Common when an ISR is written in raw assembly and stomps on LR, or an ISR is declared `__attribute__((interrupt))` on Cortex-M (it shouldn't be — Cortex-M handlers are normal functions).

- UFSR `INVPC=1`

### Unaligned 32-bit Load

```c
uint8_t buf[8];
uint32_t v = *(uint32_t*)(buf + 1);   // misaligned
```

- With `UNALIGN_TRP` enabled: UFSR `UNALIGNED=1`
- Without: silently succeeds on Cortex-M3/M4 for normal loads (LDR), faults for LDM/STM.

### Imprecise Bus Error

The most painful kind. The CPU's write buffer drained after the offending instruction had already retired, so PC points somewhere later. To turn imprecise into precise:

```c
// Disable write buffer (Cortex-M3/M4)
SCnSCB->ACTLR |= SCnSCB_ACTLR_DISDEFWBUF_Msk;
```

This slows things down — use only while hunting the bug.

## Catching Stack Overflows

Hardware help on ARMv8-M (Cortex-M23/M33/M55/M85, e.g., STM32U5): `MSPLIM` and `PSPLIM` registers cause an exception the instant SP drops below the limit. On older M3/M4/M7 cores there's no SP limit, so use:

- **MPU stack guard**: place an MPU region with no-access permissions just below each stack. Overflow → MemManage fault.
- **Canary words**: write a magic value at the bottom of each stack, periodically check it hasn't changed.
- **FreeRTOS `configCHECK_FOR_STACK_OVERFLOW = 2`**: kernel checks at every context switch via both methods.

```c
// Canary-based check
#define CANARY 0xDEADC0DE
extern uint32_t _sstack;

void check_stack(void) {
    if (*(uint32_t*)&_sstack != CANARY) {
        panic("MSP stack overflow");
    }
}
```

## GDB Recipe

When stopped in HardFault_Handler:

```
(gdb) bt                          # backtrace shows the fault handler
(gdb) p/x $sp                     # current SP
(gdb) p/x ((uint32_t*)$sp)[6]    # PC of faulting instruction
(gdb) p/x $lr                     # EXC_RETURN
(gdb) p/x *(SCB_Type*)0xE000ED00 # full SCB
(gdb) x/i 0x080012ae              # disassemble the faulting instruction
```

OpenOCD/JLink helper macros (saved in `.gdbinit`):

```gdb
define hardfault
  set $sp_real = ($lr & 4) ? $psp : $msp
  printf "PC  = 0x%08x\n", ((uint32_t*)$sp_real)[6]
  printf "LR  = 0x%08x\n", ((uint32_t*)$sp_real)[5]
  printf "PSR = 0x%08x\n", ((uint32_t*)$sp_real)[7]
  printf "CFSR= 0x%08x\n", *(uint32_t*)0xE000ED28
  printf "HFSR= 0x%08x\n", *(uint32_t*)0xE000ED2C
  printf "MMFAR=0x%08x\n", *(uint32_t*)0xE000ED34
  printf "BFAR =0x%08x\n", *(uint32_t*)0xE000ED38
end
```

## Persisting Fault Reports Across Reset

For field-deployed devices, dump the fault frame somewhere it survives the reset:

1. **Backup SRAM** (STM32 BKPSRAM, ~4 KB powered by VBAT) — write the frame, reset, app reads it on next boot.
2. **RTC backup registers** (small, ~80 bytes on F4, more on H7) — for a compact fault report.
3. **A flash sector** dedicated to crash dumps — slower to write but persistent without VBAT.

Then on next boot, ship the dump to your telemetry pipeline. This is how production firmware learns about field crashes.

## Common Pitfalls

### Pitfall 1: Default `while(1)` Handler Silently Hangs the Device

The CMSIS startup file has a weak `HardFault_Handler` that's an infinite loop. Production firmware needs at minimum a watchdog reset path, ideally a crash dump too.

### Pitfall 2: Faulting Inside the Fault Handler

If `hardfault_report` calls printf, and printf walks a corrupt heap, you fault again → HardFault locks up. Keep the fault path **dependency-minimal** (raw UART writes, no malloc, no FreeRTOS API).

### Pitfall 3: Reading Stale CFSR Bits

CFSR bits are sticky — clear them by writing 1 to clear (after you've recorded the value).

```c
SCB->CFSR = SCB->CFSR;  // W1C clears all set bits
```

### Pitfall 4: Compiler Reorders Stack Setup, Mangling the Frame

Use `__attribute__((naked))` on the entry stub so the compiler doesn't add prologue code that overwrites the frame layout.

### Pitfall 5: Calling Stack-Hungry Functions in the Fault Handler When MSP Is Almost Full

The fault may itself be a near-overflow. Switch to a dedicated emergency stack:

```c
__attribute__((naked)) void HardFault_Handler(void) {
    __asm volatile (
        "ldr sp, =_emergency_stack_top \n"
        "b hardfault_report             \n"
    );
}
```

## Summary

1. **Enable MemManage/BusFault/UsageFault handlers** so faults are categorized, not all-HardFault.
2. **CFSR + MMFAR/BFAR + stacked PC** is the diagnostic core.
3. **Decode EXC_RETURN bit 2** to pick MSP vs PSP for the frame.
4. **PRECISERR + BFARVALID = address you faulted on.**
5. **NOCP = used FPU without enabling it.**
6. **Imprecise errors** can be made precise by disabling the write buffer (debug only).
7. **Stack overflow protection**: MPU guard, canaries, RTOS stack-check, or ARMv8-M SP limit.
8. **Persist crash frames to backup SRAM / RTC regs** for field crash telemetry.
9. **Keep the fault handler dependency-light** — don't fault inside the fault handler.

## See Also

- [Interrupts](interrupts.md) — exception model, NVIC
- [Linker Scripts](linker_scripts.md) — stack placement, .ramfunc
- [JTAG/SWD](jtag_swd.md) — attaching a debugger to inspect SCB live
- [Watchdog](watchdog.md) — recovery when crash dump fails
