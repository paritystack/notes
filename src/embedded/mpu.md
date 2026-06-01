# Memory Protection Unit (MPU)

## Overview

The MPU is a hardware block in Cortex-M (and many other) cores that enforces **access rules on regions of the address map**: it can mark RAM as no-execute, flash as read-only, a task's data as private, or a guard page as no-access, and raise a fault the instant code violates a rule. It is *not* an [MMU](processor_design.md) — there is **no virtual memory, no address translation, no paging**; the MPU only adds permissions and memory attributes on top of the physical addresses that are already there. Think of it as a small, fast permission filter that turns silent memory-corruption bugs into immediate, debuggable [faults](hardfault_debugging.md), and as the mechanism that lets an [RTOS](../rtos/freertos.md) isolate tasks from each other. It is the lighter-weight sibling of [TrustZone-M](trustzone_m.md): the MPU separates *privileged vs unprivileged code's* access; TrustZone separates *secure vs non-secure worlds*.

```
   MMU (Cortex-A, Linux)            MPU (Cortex-M, RTOS/bare-metal)
   ┌──────────────┐                 ┌──────────────┐
   │ virtual addr │                 │ physical addr│ (unchanged)
   │   ↓ translate│                 │   ↓ check    │
   │ page tables  │                 │ ~8–16 regions│
   │   ↓          │                 │   ↓          │
   │ physical addr│                 │ allow / FAULT│
   │ + permissions│                 │ + attributes │
   └──────────────┘                 └──────────────┘
   Translation + protection         Protection only
```

## Regions

The MPU holds a small fixed number of programmable **regions** (typically 8, sometimes 16). Each region defines a base address, a size, and permissions/attributes. A memory access is checked against all regions; the **highest-numbered matching region wins**, which lets you carve a small exception out of a big permissive region (e.g. a read-only guard inside writable RAM).

### Armv7-M regions (Cortex-M3/M4/M7)

- Base address must be **aligned to the region size**.
- Size is a **power of two**, 32 B to 4 GB.
- Each region splits into 8 **subregions** you can individually disable — the only way to get non-power-of-two coverage.

### Armv8-M regions (Cortex-M23/M33/M55)

- Defined by **base and limit address** (32-byte granularity) — far more flexible, no power-of-two/alignment headache.
- Attributes come from an indexed **MAIR** table (like the Cortex-A scheme).

### Per-region permissions

| Setting | Controls |
|---------|----------|
| **AP** (Access Permission) | Read/Write/Read-only/No-access, separately for privileged vs unprivileged |
| **XN** (eXecute Never) | Whether instructions may be fetched from the region |
| **Attributes** | Cacheability, bufferability, shareability, device vs normal memory |

## What You Use It For

```
┌───────────────────────────────────────────────┐
│ Typical bare-metal / RTOS MPU layout           │
├───────────────────────────────────────────────┤
│ Flash (.text/.rodata) → RO, eXecutable         │
│ SRAM (.data/.bss)     → RW, eXecute-Never      │ ← blocks code-from-RAM exploits
│ Peripherals           → RW, Device, XN         │ ← strict ordering, no spec. fetch
│ Stack guard page      → No-access (size 32 B)  │ ← overflow → instant fault
│ DMA buffer            → Non-cacheable normal    │ ← coherency, see cache_tcm.md
└───────────────────────────────────────────────┘
```

The four highest-value uses:

1. **Catch stack overflow.** Place a tiny No-access region just past the end of a task stack; the overflowing push faults immediately instead of silently smashing the next variable. This is why [memory-management](memory_management.md) and MPU go together.
2. **Mark RAM no-execute (XN) and flash read-only.** Hardens against buffer-overflow-to-code attacks and catches wild function pointers / corrupted return addresses.
3. **Set DMA/[I2S](i2s.md) buffers non-cacheable.** On a [cached](cache_tcm.md) Cortex-M7 the MPU's attribute bits are how you avoid stale-data coherency bugs without manual clean/invalidate.
4. **Per-task isolation in an [RTOS](../rtos/freertos.md).** On a context switch the kernel reprograms a few regions so each unprivileged task can only touch its own stack and granted objects; a bug in one task faults instead of corrupting another.

## Faults

A violation raises a **MemManage fault** (or escalates to [HardFault](hardfault_debugging.md) if MemManage isn't enabled). The handler can decode exactly what happened:

```c
void MemManage_Handler(void) {
    uint32_t cfsr = SCB->CFSR;          // Configurable Fault Status Register
    if (cfsr & SCB_CFSR_MMARVALID_Msk)  // address is valid?
        uint32_t addr = SCB->MMFAR;     // the faulting address
    // MSTKERR/MUNSTKERR = stacking fault (likely stack overflow into guard)
    // DACCVIOL = data access violation, IACCVIOL = instruction fetch violation
    while (1);
}
```

`IACCVIOL` means something tried to execute from an XN region (corrupted PC); `DACCVIOL` + `MMFAR` points at the exact illegal data address. See [HardFault Debugging](hardfault_debugging.md) for the full CFSR decode.

## Minimal Setup

```c
// Armv7-M: region 0 = make all SRAM no-execute (XN)
MPU->RNR  = 0;
MPU->RBAR = 0x20000000;                     // SRAM base, 32B-aligned
MPU->RASR = (0x10 << 1)                      // SIZE: 2^(16+1)=128KB
          | (0x3 << 24)                      // AP: full RW
          | (1   << 28)                      // XN: execute never
          | (1   << 0);                      // ENABLE
MPU->CTRL = MPU_CTRL_PRIVDEFENA_Msk          // privileged code keeps default map
          | MPU_CTRL_ENABLE_Msk;
__DSB(); __ISB();                            // ensure it takes effect before next access
```

`PRIVDEFENA` is the common ergonomic choice: privileged code keeps the default memory map for any address *not* covered by a region, so you only have to describe the regions you actually want to restrict, rather than every byte of the address space.

## Where this connects

- [TrustZone-M](trustzone_m.md) — orthogonal axis: MPU = privileged/unprivileged, TrustZone = secure/non-secure. They compose (each world has its own MPU).
- [HardFault Debugging](hardfault_debugging.md) — MPU violations surface as MemManage/HardFault; CFSR + MMFAR tell you what and where.
- [Memory Management](memory_management.md) — MPU guard regions are the hardware behind stack-overflow detection and pool isolation.
- [Cache & TCM](cache_tcm.md) — MPU region attributes set cacheability; essential for DMA coherency on Cortex-M7.
- [FreeRTOS](../rtos/freertos.md) — the MPU port (`-MPU` kernels) reprograms regions per task for isolation.
- [Processor Design](processor_design.md) — contrast with the MMU/page tables of application-class cores.

## Pitfalls

1. **Confusing it with an MMU.** No translation, no paging, no virtual memory — only permission/attribute checks on physical addresses.
2. **Alignment/size rules (Armv7-M).** Base must be aligned to a power-of-two size. A region that "doesn't take" is usually a misaligned base or non-power-of-two size; use subregions for odd sizes.
3. **Forgetting `DSB/ISB` after reprogramming.** Memory accesses in the pipeline may use the old config; barrier before relying on the new rules.
4. **Region priority surprises.** Higher-numbered regions override lower ones on overlap — order matters when carving exceptions.
5. **Enabling MPU but not the MemManage fault.** Violations escalate straight to HardFault with less info. Enable MemManage (`SHCSR.MEMFAULTENA`) for clean diagnostics.
6. **Leaving peripherals cacheable/non-XN.** Peripheral regions should be Device memory + XN to prevent speculative fetches and reordering bugs.
7. **No default region for uncovered addresses.** Without `PRIVDEFENA`, any address not in a region faults — easy to lock yourself out. Either set it or cover everything.

## See Also

- [TrustZone-M](trustzone_m.md) — the security-world isolation axis
- [HardFault Debugging](hardfault_debugging.md) — decoding MPU faults
- [Memory Management](memory_management.md) — stack guards and pools
- [Cache & TCM](cache_tcm.md) — region attributes and DMA coherency
