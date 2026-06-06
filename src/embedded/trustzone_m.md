# TrustZone for Cortex-M

## Overview

TrustZone for Armv8-M (TZ-M) is a hardware-enforced separation between two execution states — **Secure** and **Non-Secure** — within a single Cortex-M CPU. The split lets you keep secrets (cryptographic keys, signed-firmware verification logic, secure storage) in a small, audited region while the bulk of the firmware (UI, networking, application logic) runs untrusted.

```
   Same CPU, same cycles. Two worlds.

   ┌───────────────────────────────────┐
   │       Cortex-M33 / M55 / M85      │
   │                                   │
   │  ┌─────────────┐  ┌─────────────┐ │
   │  │   Secure    │  │ Non-Secure  │ │
   │  │             │  │             │ │
   │  │ Crypto keys │  │ App code    │ │
   │  │ SB & PSA    │  │ FreeRTOS    │ │
   │  │ Trusted libs│  │ Drivers     │ │
   │  └─────────────┘  └─────────────┘ │
   └───────────────────────────────────┘
```

Non-secure code can call into pre-defined secure functions only via specific entry points. Anything else triggers a security fault. So even if the app is fully compromised, secrets stay in the secure partition.

This is the M-class analog of TrustZone for Cortex-A (used in mobile SoCs for TEE / OP-TEE). The mechanism is different from A-class but the goal is the same.

## When You Care

- Chips that support it: **Cortex-M23, M33, M35P, M55, M85** (Armv8-M and v8.1-M).
- Concrete examples: STM32L5, STM32U5, STM32H5, nRF5340, nRF54L, NXP LPC55Sxx, Renesas RA family, Microchip SAM L11.
- M3 / M4 / M7 (Armv7-M) do **not** have TrustZone-M. They have lesser mechanisms (Firewall, MPU) but no hardware-isolated secure state.

If you're targeting an older Cortex-M, this doc is mostly theory; ARM PSA (Platform Security Architecture) defines software-level analogs for non-TZ-M chips.

## Security States

Two orthogonal axes: **Secure / Non-Secure** state, and **Thread / Handler** mode (the usual Cortex-M execution-mode split).

```
            Thread            Handler
          ┌─────────┐       ┌─────────┐
Secure   │ S-thread │       │ S-handler│
          │ app     │       │ S-IRQs   │
          ├─────────┤       ├─────────┤
NS       │ NS-thread│       │ NS-handler│
          │ app     │       │ NS-IRQs  │
          └─────────┘       └─────────┘
```

The CPU has **two banked stack pointer pairs** (MSP_S/PSP_S and MSP_NS/PSP_NS), banked control registers, banked SysTick, banked MPU. As far as Non-Secure code is concerned, Secure simply doesn't exist — reads of secure addresses return bus error.

## Memory Partitioning

Three address-space attributes:

| Attribute | What |
|-----------|------|
| **Secure** | Only secure code can access. NS access faults. |
| **Non-Secure Callable (NSC)** | Secure code that NS can call as entry points. Lives at specific addresses. |
| **Non-Secure** | Either side can access. NS world's normal memory. |

Configured by:

- **SAU** (Security Attribution Unit) — programmable, internal to the CPU. 8 regions on most chips.
- **IDAU** (Implementation-Defined Attribution Unit) — vendor-fixed: STM32 builds an IDAU that respects bit 28 of the address: `0x0FFFFFFF` and below = secure mirror, `0x1xxxxxxx` and above = non-secure mirror, etc. Vendor-specific.

The effective attribute = SAU ∪ IDAU (most secure wins or per vendor rules). Configure SAU early in boot.

Typical memory map:

```
0x00000000  ┌──────────────────────┐
            │ Secure flash         │ S
            │  - bootloader        │
            │  - secure partition  │
            │  - NSC veneer table  │  NSC
            ├──────────────────────┤
0x08040000  │ Non-Secure flash     │ NS
            │  - app firmware      │
0x080FFFFF  └──────────────────────┘

0x20000000  ┌──────────────────────┐
            │ Secure SRAM          │ S
            │  - secure state      │
            │  - secrets in RAM    │
            ├──────────────────────┤
            │ Non-Secure SRAM      │ NS
0x2002FFFF  └──────────────────────┘
```

## Cross-State Calls (SG and BXNS)

Non-secure code calls into secure code via a **Non-Secure Callable (NSC)** region. NSC must contain `SG` (Secure Gateway) instructions at the entry point — any other instruction faults.

```
   NS code:                              S code (NSC entry):
   ───────                               ──────────────────
   bl secure_function    ──────────►   secure_function:
                                          SG                  ; transition to S
                                          B real_function     ; jump to secure
                                          ...
                                       real_function:
                                          ; runs in S
                                          ...
                                          BXNS lr             ; return to NS
```

Helper attributes (toolchain-supported) wrap this:

```c
// Compiled into the NSC region (.gnu.sgstubs) with SG at the start.
__attribute__((cmse_nonsecure_entry))
int secure_compute_signature(const uint8_t* msg, size_t len, uint8_t* sig) {
    // runs in S state
    return sign_with_secret_key(msg, len, sig);
}
```

GCC's `-mcmse` flag enables CMSE (Cortex-M Security Extension) intrinsics and generates the right veneer.

Going the other way (S calls into NS) is rarely needed but supported via `cmse_nonsecure_call` and `BLXNS`.

## Banking

Many resources are duplicated per state:

- **Stack pointers**: MSP_S, MSP_NS, PSP_S, PSP_NS.
- **CONTROL register**: SPSEL, FPCA banked per state.
- **SysTick**: separate timer for each — a secure tick and a non-secure tick.
- **MPU**: banked. Configure separately for S and NS.
- **Faults**: secure/non-secure fault status registers.

Interrupts are configured per-source: each interrupt can be either S or NS. NS code can't disable or even see S interrupts.

## Boot Flow

```
Reset
   │
   ▼
Secure boot (S state)
   - Configure SAU, set memory regions.
   - Verify secure partition signature.
   - Verify Non-Secure app signature.
   - Configure interrupt routing.
   - Set NS Vector Table base (VTOR_NS).
   - Transition to NS state via BLXNS or branch to NS reset handler.
   │
   ▼
Non-Secure app starts at its Reset_Handler
   - Sets up its stack, BSS, normal startup.
   - Runs application code.
```

Until the boot code transitions to NS, the entire CPU is in Secure. After transition, the CPU stays in NS unless calling NSC entry points.

## What Goes Where

| In Secure | In Non-Secure |
|-----------|---------------|
| Bootloader, signature verification | Application logic |
| Cryptographic keys, signing routines | Drivers (unless peripheral is secure-only) |
| Secure storage / counter | UI, networking stacks |
| TLS / mbedTLS for sensitive sessions | Filesystem |
| Anti-rollback eFuse access | RTOS scheduler (usually NS) |
| Watchdog (in safety-critical designs) | Sensor processing |
| RNG | OTA download (verify happens in S) |

**Rule of thumb**: smaller secure partition = smaller TCB = easier to audit. Don't drag the whole application into Secure.

## PSA: ARM's Software Architecture

ARM defined **Platform Security Architecture (PSA)** as the API/RoT spec for TrustZone-M. Components:

- **Secure Partition Manager (SPM)** — scheduler-like that runs secure partitions.
- **PSA APIs**: `psa_crypto`, `psa_storage`, `psa_attestation`, `psa_firmware_update`.
- **PSA Functional API** — what NS code calls into.

Two reference implementations:

- **Trusted Firmware-M (TF-M)**: open-source from Linaro, ports to STM32U5/L5/H5, nRF5340, NXP, etc.
- Vendor builds on top of TF-M (STM32CubeIDE wizards generate TF-M templates).

If you're starting fresh on a TZ-M part, build on TF-M rather than rolling your own secure-partition manager.

## A Concrete Use Case

Sensor product that needs:
- TLS-signed telemetry to cloud.
- OTA updates that must be authenticated.
- Per-device unique key that must not leak.

Without TZ-M:
- Private key in flash, app code can read it.
- Compromise of app = key leak.

With TZ-M:
- Private key lives in Secure flash + Secure SRAM.
- App calls `psa_sign_hash(key_id, hash, sig)` — key never crosses into NS.
- Even with NS code fully compromised (buffer overflow, malicious update), key remains protected.

## Build System

GCC/Clang flags:

```
-mcpu=cortex-m33 -mfloat-abi=hard -mfpu=fpv5-sp-d16
-mcmse                               # enable Cortex-M Security Extension
```

`-mcmse` activates `cmse_nonsecure_entry`, `cmse_nonsecure_call`, the SG generation, and the secure-gateway veneer placement.

Linker arranges:
- Secure image with its NSC region exporting an "import library" (a small .o with the entry-point addresses).
- Non-Secure image links against the import library so it knows where to call.

In practice, TF-M's build system handles this. Setting it up by hand is fiddly.

## Performance Cost

- **Cross-state call overhead**: ~10-20 cycles per call (SG + BXNS + back).
- **Banked register save/restore on transition**: a few additional cycles.
- **Memory access check overhead**: 0 — SAU/IDAU check is per-fetch, in parallel with memory access. No latency penalty.

For 99% of code, the cost is the call overhead at boundaries — negligible unless you cross states in a tight loop.

## Common Pitfalls

### Pitfall 1: SG Outside NSC Region

You declared an entry point but the linker put it in a non-NSC region. NS calls → SecureFault. Fix the linker script to place `.gnu.sgstubs` in NSC memory.

### Pitfall 2: Forgetting to Configure SAU

After reset, SAU is all-zero → entire memory map is whatever IDAU defaults to. Often "all secure" or "all NS". Boot code must configure SAU regions explicitly before exiting Secure.

### Pitfall 3: Stack Underflow on State Transition

Returning from S to NS uses NS stack. If you never set MSP_NS / PSP_NS, transition lands on garbage and faults instantly.

### Pitfall 4: Sharing Pointers Across States

NS passes a buffer pointer to an S API. S writes secret into that buffer. NS reads → leaks secret to the wrong world. Validate that NS-supplied pointers belong to NS memory:

```c
if (cmse_check_address_range(buf, len, CMSE_MPU_READ | CMSE_NONSECURE) == NULL) {
    return ERROR;
}
```

CMSE's `cmse_check_address_range` checks the SAU attribution of a pointer.

### Pitfall 5: Interrupt Routing Confusion

NVIC has per-IRQ targets — but if you route the wrong IRQ to NS state, NS handler runs in NS while the peripheral might need secure access. Audit IRQ routing at boot.

### Pitfall 6: Forgetting Secure Faults Are Different

`SecureFault_Handler` is its own vector. If you only handle `HardFault`, secure exceptions silently escalate. Implement secure faults and check `SCB->SFSR` (Secure Fault Status Register) to decode.

### Pitfall 7: Mixed-State Stack Frames in FPU Code

FPU registers are partially banked. Crossing states with the FPU active needs `__cmse_nonsecure_call` to handle saving FPU context. Toolchain helpers do this but only with the right flags.

### Pitfall 8: Debug Across States

A debugger has to opt-in to seeing Secure state — set by `DBGAUTH` signals or eFuses. Production silicon has Secure debug disabled; you can't single-step through Secure code on a locked chip.

### Pitfall 9: Confusing TZ-M With TZ-A

ARM TrustZone for Cortex-A (used in Android, Linux) is a completely different mechanism (banked privilege levels, monitor mode). Don't apply A-class ideas to M-class.

## Cheat Sheet

```
TZ-M chips:           Cortex-M23/M33/M35P/M55/M85
Two states:           Secure (S), Non-Secure (NS)
Memory attribution:   SAU (programmable) + IDAU (vendor-fixed)
Transition NS → S:    SG instruction at entry point in NSC region
Transition S → NS:    BLXNS / BXNS
Toolchain flag:       -mcmse
Build framework:      Trusted Firmware-M (TF-M) + vendor wrappers
APIs:                 PSA (Platform Security Architecture)
```

## Summary

1. **TZ-M splits one Cortex-M into Secure and Non-Secure states**, hardware-enforced.
2. **Memory is attributed S, NSC, or NS** via SAU + IDAU.
3. **NS calls into S only via SG entry points in NSC region.**
4. **Banked SP, MPU, SysTick, faults** per state.
5. **Keep the secure partition small** — easier to audit.
6. **Use Trusted Firmware-M** instead of building from scratch.
7. **PSA APIs (`psa_crypto`, `psa_storage`)** are the standard NS-callable surface.
8. **Validate NS-supplied pointers** with `cmse_check_address_range`.
9. **Audit interrupt routing** — IRQ → S or NS is per source.
10. **TZ-M ≠ TZ-A.** Different mechanism, same goal.

## See Also

- [Secure Boot](secure_boot.md) — TZ-M typically hosts the secure-boot verifier
- [Bootloaders](bootloaders.md) — boot transition to NS
- [HardFault Debugging](hardfault_debugging.md) — SecureFault is separate
- [Security](../security/index.html) — TLS, crypto algorithms used by the secure partition
