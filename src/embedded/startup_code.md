# Startup Code & C Runtime

## Overview

Between power-on and the first line of your `main()` runs a small, often-overlooked stretch of code — the **reset handler and C runtime startup (crt0)** — whose job is to turn raw silicon into the environment a C program assumes: a valid stack, zero-initialized globals, copied initial values, and a configured clock. Nothing in C "just works" until this code has run. It is intimately tied to the [linker script](linker_scripts.md) (which defines *where* `.data`, `.bss`, and the stack live) and to [CMSIS](cmsis.md) (which provides `SystemInit` and the vector table). Understanding it is what lets you debug a chip that hangs *before* `main`, write a [bootloader](bootloaders.md) that jumps into an application, or shave boot time off a battery device.

```
   POWER-ON / RESET
        │
        ▼
   CPU loads from address 0x0:
     [0] → initial Stack Pointer (MSP)
     [1] → Reset_Handler address      ← entry point
        │
        ▼
   Reset_Handler:
     1. SystemInit()         configure clocks/FPU (optional, early)
     2. copy .data           flash → RAM (initialized globals)
     3. zero .bss            uninitialized globals = 0
     4. run C++ ctors        __libc_init_array / .init_array
     5. call main()
        │
        ▼
   main()  ←── your code finally starts here
```

## The Vector Table Is the Boot Record

On Cortex-M there is no separate "boot ROM jump" you write — the **first two 32-bit words of the vector table are the boot record**:

```
Address    Contents
0x0000     Initial Main Stack Pointer (MSP)   ← loaded into SP automatically
0x0004     Reset_Handler                      ← PC jumps here
0x0008     NMI_Handler
0x000C     HardFault_Handler
...        (one entry per exception/IRQ)
```

The hardware itself loads SP from word 0 and PC from word 1 — so by the time `Reset_Handler` executes, you *already have a working stack* (this is why the reset handler can be C). The table normally lives at the start of flash; a [bootloader](bootloaders.md) relocates it for the application by writing `SCB->VTOR`. See [Interrupts](interrupts.md) for how the rest of the table dispatches.

## What the Reset Handler Does

### 1. Copy `.data` (initialized globals)

A global like `int counter = 5;` needs its initial value `5` somewhere non-volatile (flash), but the variable lives in RAM. The startup code copies the initial-value image from flash to RAM, using symbols the [linker script](linker_scripts.md) exports:

```c
extern uint32_t _sidata;   // .data init image in FLASH (Start of Init DATA)
extern uint32_t _sdata, _edata;   // .data region in RAM
extern uint32_t _sbss,  _ebss;    // .bss region in RAM

void Reset_Handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;   // copy initialized globals
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0;   // zero uninitialized
    SystemInit();
    __libc_init_array();      // C++ static constructors, __attribute__((constructor))
    main();
    while (1);                // main should never return on an MCU
}
```

### 2. Zero `.bss` (uninitialized globals)

The C standard guarantees `static`/global variables with no initializer start at zero. Flash doesn't store a megabyte of zeros — the startup loop zeroes the whole `.bss` range. **Skip this and a `static int count;` you assume is 0 holds whatever garbage was in RAM at power-on** — a classic "works after full power cycle, fails after warm reset" bug.

### 3. `SystemInit()` and clocks

[CMSIS](cmsis.md) convention: a weakly-defined `SystemInit()` runs early to set up the PLL/[clock tree](clock_systems.md), enable the FPU, and configure flash wait-states *before* the main copy loop (so the copy runs at full speed). Do too much here and you slow every boot/wake; on a deep-sleep battery device this is worth trimming.

### 4. C++ constructors / init arrays

`__libc_init_array()` walks the `.init_array` section and calls every global C++ constructor and `__attribute__((constructor))` function. If you use C++ globals and skip this (or use a `-nostartfiles` build without restoring it), those objects are never constructed.

## Where It Comes From

| Source | What it provides |
|--------|------------------|
| **Vendor `startup_*.s`** | Hand-written assembly reset handler + vector table (ST, NXP) |
| **CMSIS `startup_device.c`** | Same, in C, with weak default handlers |
| **newlib / picolibc crt0** | The libc's `_start` doing the copy/zero/init-array |
| **Your own** | Bare-metal/`-nostdlib` projects often write a 30-line reset handler |

Weak symbols matter: vendor files declare every handler `__attribute__((weak, alias("Default_Handler")))` so unhandled interrupts spin in a default trap, and you override just the ones you need by defining a function with the same name.

## Where this connects

- [Linker Scripts](linker_scripts.md) — defines the `_sdata`/`_edata`/`_sbss`/`_ebss`/`_sidata` symbols and stack location the startup code copies between; the two are written as a pair.
- [CMSIS](cmsis.md) — supplies the vector table layout, `SystemInit`, and weak handler convention.
- [Bootloaders](bootloaders.md) — a bootloader *is* a second reset handler that sets `VTOR`/SP/PC and jumps to the app's vector table.
- [Clock Systems](clock_systems.md) — `SystemInit` configures the PLL and flash wait-states before `main`.
- [Interrupts](interrupts.md) — the vector table that boot uses is the same one that dispatches every IRQ.
- [Memory Management](memory_management.md) — `.data`/`.bss`/stack/heap regions established here are the runtime memory model.

## Pitfalls

1. **Skipping `.bss` zeroing.** Globals you assume are 0 hold RAM garbage. Symptom: passes after a cold power cycle (RAM happens to be 0), fails after warm reset.
2. **Wrong linker symbols.** A `.data` copy that uses mismatched `_sidata`/`_sdata` symbols corrupts globals before `main` even runs — a confusing pre-main crash.
3. **`main()` returning.** On an MCU there's no OS to return to; falling off `main` runs whatever follows in memory. End the reset handler with an infinite loop.
4. **Forgetting `__libc_init_array`.** C++ global objects are never constructed; their members are garbage. Common in `-nostartfiles` builds.
5. **Stack/heap collision.** Linker doesn't always check that the stack (top of RAM, growing down) won't crash into the heap/`.bss` (growing up). See [Memory Management](memory_management.md).
6. **Heavy `SystemInit`.** Doing slow peripheral setup before `main` bloats every wake from sleep on low-power devices.
7. **VTOR not set after a bootloader jump.** The app's interrupts vector to the bootloader's table; set `SCB->VTOR` to the app's vector table before enabling interrupts.

## See Also

- [Linker Scripts](linker_scripts.md) — the symbols and regions startup relies on
- [CMSIS](cmsis.md) — vector table and `SystemInit`
- [Bootloaders](bootloaders.md) — handing off to an application
- [Memory Management](memory_management.md) — the resulting runtime memory model
