# CMSIS

## Overview

CMSIS (Cortex Microcontroller Software Interface Standard) is ARM's vendor-neutral C abstraction over Cortex-M cores. It standardizes register definitions, intrinsics, startup code, and a few common APIs so that code that uses `SCB->VTOR` or `NVIC_EnableIRQ()` looks the same on every Cortex-M chip from every silicon vendor.

```
┌─────────────────────────────────────────────────────┐
│        Application                                  │
├─────────────────────────────────────────────────────┤
│  Vendor HAL  │  Vendor LL  │  Bare-metal C / asm    │
├─────────────────────────────────────────────────────┤
│              CMSIS-Device  (stm32f4xx.h, ...)       │
│              CMSIS-Core    (core_cm4.h, NVIC, SCB)  │
├─────────────────────────────────────────────────────┤
│              Cortex-M Core (M0/M3/M4/M7/M33/...)    │
└─────────────────────────────────────────────────────┘
```

Without CMSIS, every silicon vendor would invent their own `enable_interrupt()` macro. With it, `__enable_irq()`, `NVIC_SetPriority()`, `SCB->AIRCR`, `DWT->CYCCNT` work everywhere.

## CMSIS Components

| Component | What it provides |
|-----------|------------------|
| **CMSIS-Core (Cortex-M)** | Headers for CPU registers (SCB, NVIC, SysTick, DWT, ITM, MPU, FPU), intrinsics (`__disable_irq`, `__DSB`, `__WFI`), reset/IRQ handlers |
| **CMSIS-Device** | Vendor-supplied: peripheral register layouts (`USART_TypeDef`), peripheral base addresses (`USART1`), startup file |
| **CMSIS-DSP** | Optimized fixed/floating-point DSP routines (FFT, FIR, IIR, matrix) |
| **CMSIS-NN** | Quantized neural-net kernels for inference on M4/M7/M33/M55 |
| **CMSIS-RTOS v2** | Vendor-neutral RTOS API (FreeRTOS, RTX, ThreadX adapters all exist) |
| **CMSIS-Driver** | Standard peripheral driver API (mostly used by Keil RTX, less common elsewhere) |
| **CMSIS-DAP** | Reference debug-probe firmware (DAPLink-style probes) |
| **CMSIS-SVD** | XML peripheral description used by IDEs / debuggers |
| **CMSIS-Pack** | ZIP format for distributing device support |

For most projects, you care about **CMSIS-Core + CMSIS-Device**. The rest are optional or specialized.

## What CMSIS-Core Gives You

### Register Maps

Every CPU peripheral exposed as a struct overlay at a fixed address:

```c
// From core_cm4.h
typedef struct {
    __IOM uint32_t CPUID;     // 0xE000ED00
    __IOM uint32_t ICSR;      // 0xE000ED04
    __IOM uint32_t VTOR;      // 0xE000ED08
    __IOM uint32_t AIRCR;     // 0xE000ED0C
    __IOM uint32_t SCR;
    __IOM uint32_t CCR;
    __IOM uint8_t  SHPR[12];
    __IOM uint32_t SHCSR;
    __IOM uint32_t CFSR;
    __IOM uint32_t HFSR;
    __IOM uint32_t DFSR;
    __IOM uint32_t MMFAR;
    __IOM uint32_t BFAR;
    // ...
} SCB_Type;

#define SCB ((SCB_Type *) 0xE000ED00)
```

So `SCB->VTOR = 0x08008000` writes to address 0xE000ED08. The `__IOM` macros encode read/write semantics (`volatile` plus const-correctness).

### NVIC API

```c
NVIC_EnableIRQ(USART1_IRQn);
NVIC_DisableIRQ(USART1_IRQn);
NVIC_SetPriority(USART1_IRQn, 5);
NVIC_GetPriority(USART1_IRQn);
NVIC_SetPendingIRQ(USART1_IRQn);
NVIC_ClearPendingIRQ(USART1_IRQn);
NVIC_GetPendingIRQ(USART1_IRQn);
NVIC_SystemReset();
```

These are inline functions in `core_cm4.h`. They compile to 2-3 instructions each.

### Core Intrinsics

```c
__disable_irq();    // CPSID i — block all maskable interrupts
__enable_irq();     // CPSIE i
__DSB();            // Data Synchronization Barrier
__DMB();            // Data Memory Barrier
__ISB();            // Instruction Synchronization Barrier
__WFI();            // Wait For Interrupt (sleep until IRQ)
__WFE();            // Wait For Event
__SEV();            // Send Event
__NOP();
__CLZ(x);           // Count leading zeros (single instruction)
__REV(x);           // Byte-reverse (endianness flip)
__RBIT(x);          // Bit-reverse (useful for CRC)
__LDREXW(addr);     // Load-exclusive (atomics)
__STREXW(val,addr); // Store-exclusive
```

`__CLZ` is the secret weapon for fast `log2`, priority queues, and finding-highest-set-bit code — it's a single cycle.

```c
static inline int log2_u32(uint32_t x) {
    return 31 - __CLZ(x);
}
```

### System Globals

```c
extern uint32_t SystemCoreClock;
void SystemInit(void);            // called from Reset_Handler
void SystemCoreClockUpdate(void);  // re-derive from current clock config
```

`SystemCoreClock` is the value HAL functions, `HAL_Delay`, FreeRTOS port code, and SysTick configuration all read. Keep it in sync after any clock change.

## CMSIS-Device

Vendor-specific. ST's `stm32f4xx.h` does:

```c
#if defined(STM32F407xx)
    #include "stm32f407xx.h"
#elif defined(STM32F429xx)
    #include "stm32f429xx.h"
// ...
#endif
```

And `stm32f407xx.h` brings in `core_cm4.h` plus per-peripheral structs:

```c
typedef struct {
    __IOM uint32_t CR1;
    __IOM uint32_t CR2;
    __IOM uint32_t SR;
    __IOM uint32_t DR;
    // ...
} USART_TypeDef;

#define USART1 ((USART_TypeDef*) USART1_BASE)
```

Plus IRQn enum, peripheral base addresses, bit-field macros (`USART_CR1_UE_Msk`, `USART_CR1_UE_Pos`).

You program at this level when writing LL code or pure bare-metal.

## HAL vs LL vs CMSIS Bare-Metal

ST gives you three abstraction levels on STM32 (Nordic, NXP, Renesas have analogous splits):

| Layer | Example | Size | Speed | Portability |
|-------|---------|------|-------|-------------|
| **HAL** | `HAL_UART_Transmit(&huart1, buf, n, 100);` | ~25 KB for a basic project | Slow (config validation, abstraction) | Easy to swap MCU families within ST |
| **LL** | `LL_USART_TransmitData8(USART1, byte);` | Small, ~few KB | Near bare-metal | STM32-only |
| **CMSIS bare-metal** | `USART1->DR = byte;` | Smallest | Fastest | Portable within ARM, but the peripherals themselves are vendor-specific |

### When to Use Each

**HAL**:
- Prototyping, when you want callbacks and don't care about every byte of flash.
- Mixing with CubeMX-generated init code.
- Teams without deep MCU experience — HAL's validation catches misconfigurations.

**LL**:
- Production firmware where you want compact, fast code but still want vendor-supplied register-name macros.
- Performance-critical paths.
- IRQ handlers and DMA setup (where HAL adds significant overhead per call).

**Bare-metal / CMSIS only**:
- Bootloaders (size-constrained, no HAL dependencies).
- Drivers being ported to another vendor's part.
- When the HAL/LL doesn't support a feature you need.

### Mixing Them

Perfectly valid: HAL for slow paths (init, infrequent operations), LL or bare-metal in IRQs and hot loops. ST's headers tolerate mixing because both ultimately poke the same registers.

```c
// Init with HAL (one-time, clear)
HAL_UART_Init(&huart1);

// Fast path: LL
LL_USART_TransmitData8(USART1, byte);

// Or even rawer:
while (!(USART1->SR & USART_SR_TXE)) {}
USART1->DR = byte;
```

## CMSIS-DSP

Library of optimized DSP primitives, written in CMSIS-Core intrinsics so it benefits from M4/M7 DSP extensions and the FPU. Single static library, picks the right code path at compile time based on `__ARM_FEATURE_DSP` etc.

Common functions:

```c
arm_fir_q15(&S, src, dst, blockSize);   // FIR filter
arm_iir_lattice_f32(&S, src, dst, N);    // IIR
arm_rfft_fast_f32(&S, src, dst, ifft);   // Real FFT
arm_mat_mult_f32(&MA, &MB, &MC);         // matrix multiply
arm_cmplx_mag_f32(src, dst, N);          // complex magnitude
arm_max_f32(src, N, &out, &idx);         // max + index
arm_sqrt_q15(in, &out);
```

If your code does any signal processing on a Cortex-M, **don't write it from scratch** — `arm_*` is faster and well-tested.

Linker: include `libarm_cortexM4lf_math.a` (or appropriate variant for your core/FPU).

## CMSIS-NN

Quantized 8/16-bit neural-net inference kernels. Used by TFLite-Micro on Cortex-M.

```c
arm_convolve_s8(...);
arm_fully_connected_s8(...);
arm_avgpool_s8(...);
arm_softmax_s8(...);
```

Real-world impact: a TinyML model that runs at 5 fps with naïve C can hit 30+ fps using CMSIS-NN on the same M4F.

## CMSIS-RTOS v2

A vendor-neutral RTOS API. Maps to:

- **Keil RTX5** (the reference implementation)
- **FreeRTOS** (via official adapter)
- **ThreadX** (Microsoft Azure RTOS, has CMSIS-RTOS wrapper)

```c
osThreadId_t blink_id;

void blink_task(void* arg) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
        osDelay(500);
    }
}

int main(void) {
    osKernelInitialize();
    blink_id = osThreadNew(blink_task, NULL, NULL);
    osKernelStart();
}
```

Useful when you want to swap RTOSes without rewriting application code. Otherwise the native FreeRTOS API is more common in the wild.

## CMSIS-SVD

Each chip ships with an SVD (System View Description) XML that describes every peripheral, register, and bit. IDEs, debuggers, and tools consume these:

- VSCode + Cortex-Debug extension: lets you view peripheral registers symbolically during a debug session.
- OpenOCD doesn't use SVD directly, but pyOCD does (`pyocd registers --show all`).
- `svd2rust`: generates Rust register types from SVD.

Always grab the SVD for your chip from the vendor's CMSIS-Pack. STM32F407's SVD is ~3 MB of XML — but it's how your debugger knows `RCC->APB1ENR` bits 17 is `USART2EN`.

## CMSIS-Pack

`.pack` file = ZIP containing:
- Device family description (PDSC XML)
- CMSIS-Device headers + startup
- Vendor HAL / drivers
- Example projects
- SVDs

Used by Keil µVision, IAR, MDK. Less common with Make/CMake workflows but worth knowing because that's where most vendor "official" support files live.

## What CMSIS Won't Solve

- **Peripheral programming differences** between vendors. STM32 UART ≠ Nordic UARTE ≠ NXP LPUART.
- **Clock tree config.** Each silicon vendor has different RCC organization.
- **Power-mode handling.** Different sleep states everywhere.
- **DMA channel mapping.** Vendor-specific.

CMSIS standardizes the **Cortex-M core**, not the peripherals around it. Anything beyond the CPU itself is still vendor turf.

## Practical Setup

Typical project layout:

```
project/
├── cmsis/
│   ├── core/                # CMSIS-Core headers (ARM)
│   │   ├── core_cm4.h
│   │   ├── cmsis_gcc.h
│   │   └── ...
│   └── device/              # CMSIS-Device (vendor)
│       ├── stm32f407xx.h
│       ├── stm32f4xx.h
│       ├── system_stm32f4xx.c
│       └── startup_stm32f407xx.s
├── src/
│   └── main.c
└── Makefile
```

Makefile fragment:

```make
CFLAGS += -DSTM32F407xx \
          -mcpu=cortex-m4 -mthumb \
          -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
          -Icmsis/core -Icmsis/device
```

The `-D` selects the variant header chain. The `-mfpu` / `-mfloat-abi` must match what CMSIS-DSP was built against if you link it.

## Common Pitfalls

### Pitfall 1: Wrong CMSIS Version for Your Toolchain

GCC 12+ expects CMSIS-Core 5.7+. Older CMSIS uses inline-asm syntax that newer GCC rejects. Symptom: hundreds of "expected `:` or `}`" errors in `cmsis_gcc.h`.

### Pitfall 2: Missing `-DSTM32F407xx` Define

Without the device define, `stm32f4xx.h` doesn't include the right variant. Compile fails with "no such file or directory: stm32fxxxxx.h" or peripherals undefined.

### Pitfall 3: SystemInit Not Called

CMSIS startup code calls `SystemInit()` before `main`. If you replaced the startup file with your own, you may have skipped it — and the clock tree isn't configured.

### Pitfall 4: FPU Mismatch

CMSIS-DSP linked as soft-float, project compiled hard-float (or vice-versa) → linker errors about "incompatible floating-point ABI". Match `-mfloat-abi` across the whole build.

### Pitfall 5: Mixing HAL Handles With LL

```c
UART_HandleTypeDef huart1;
HAL_UART_Init(&huart1);

LL_USART_Disable(USART1);     // disables without HAL knowing → HAL state stale
HAL_UART_Transmit(&huart1, buf, n, 100);  // blocks forever, UART is disabled
```

If you mix layers, the HAL state machine can desync from hardware.

### Pitfall 6: Volatile Mistakes

Forgetting `volatile` on a struct field in a custom CMSIS-Device header makes the compiler optimize repeated reads/writes. Always use `__IOM` / `__IM` / `__OM` macros for peripheral fields.

## Summary

1. **CMSIS-Core + CMSIS-Device** = the standard register-level API on Cortex-M.
2. **HAL** for fast development, **LL** for performance, **bare-metal CMSIS** for tightest code.
3. **`__CLZ`, `__DSB`, `__WFI`** are everyday intrinsics worth memorizing.
4. **`SystemCoreClock`** must stay in sync with actual clock config.
5. **CMSIS-DSP** beats hand-rolled DSP code by 5-10×; use it.
6. **CMSIS-RTOS v2** standardizes RTOS API, but most code uses native FreeRTOS.
7. **SVD files** drive symbolic peripheral views in debuggers.
8. **CMSIS standardizes the core, not the peripherals** — vendor-specific code is still required.

## See Also

- [Linker Scripts](linker_scripts.md) — startup file, vector table
- [Interrupts](interrupts.md) — NVIC API and intrinsics
- [Build Systems](build_systems.md) — incorporating CMSIS into Make/CMake
- [HardFault Debugging](hardfault_debugging.md) — SCB/CFSR via CMSIS
