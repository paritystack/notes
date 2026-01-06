# Linker Scripts (.ld)

> **Domain:** Embedded Systems, C/C++, Systems Programming
> **Key Concepts:** Memory Map, Sections, VMA vs LMA, Symbol Placement

In desktop programming, the OS decides where your program lives in RAM. In **Embedded Systems**, *you* decide. The **Linker Script** allows you to map your code and variables to specific physical addresses (Flash vs. RAM).

---

## 1. The Memory Map

A microcontroller has distinct memory regions.
*   **Flash (ROM):** Non-volatile. Stores code and constant data. (e.g., Starts at `0x08000000`).
*   **SRAM (RAM):** Volatile. Stores variables and stack. (e.g., Starts at `0x20000000`).

The Linker Script defines these:
```ld
MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
    RAM   (rwx): ORIGIN = 0x20000000, LENGTH = 128K
}
```

---

## 2. Standard Sections

The compiler produces object files with generic sections. The Linker Script groups them.

1.  **`.text` (Code):** Instructions. Read-only. Goes to **FLASH**.
2.  **`.rodata` (Read-Only Data):** Constants (`const int x = 5;`). Goes to **FLASH**.
3.  **`.data` (Initialized Data):** Global variables with values (`int x = 5;`).
    *   *Tricky:* Requires storage in FLASH (to survive power off) but must live in RAM (to be modifiable).
4.  **`.bss` (Block Started by Symbol):** Zero-initialized data (`int x;`).
    *   *Space:* Takes 0 space in FLASH. RAM is zeroed at startup.

---

## 3. VMA vs. LMA (The .data Paradox)

*   **LMA (Load Memory Address):** Where the data lives "at rest" (in the binary/Flash).
*   **VMA (Virtual Memory Address):** Where the CPU expects to find the data during execution (RAM).

For `.text`, VMA = LMA.
For `.data`, VMA != LMA.

**Startup Code Copy:**
The C Runtime (CRT) startup code (before `main()`) copies `.data` from Flash (LMA) to RAM (VMA) and zeros out `.bss`.

```ld
SECTIONS
{
    .text : {
        *(.text)
        *(.rodata)
    } > FLASH

    .data : {
        _sdata = .;        /* Symbol: Start of RAM data */
        *(.data)
        _edata = .;        /* Symbol: End of RAM data */
    } > RAM AT > FLASH     /* VMA in RAM, LMA in FLASH */
    
    _sidata = LOADADDR(.data); /* Symbol: Source in FLASH */

    .bss : {
        _sbss = .;
        *(.bss)
        _ebss = .;
    } > RAM
}
```

---

## 4. Advanced Placements

### 4.1. CCM RAM (Core Coupled Memory)
Fast RAM connected directly to the CPU. Good for DSP loops.
```c
// In C code
__attribute__((section(".ccmram"))) int critical_buffer[1024];

// In Linker Script
.ccmram : {
    *(.ccmram)
} > CCM_RAM
```

### 4.2. No-Init Section
Variables that survive a soft reboot (e.g., Crash Logs).
```ld
.noinit (NOLOAD) : {
    *(.noinit)
} > RAM
```
`(NOLOAD)` tells the programmer/flasher *not* to overwrite this section during upload, and the startup code *not* to zero it.

---

## 5. Symbols

The Linker Script can export symbols that C code can use.
*   `_stack_top`: Often set to the end of RAM.
*   `_heap_start`: Often set to `_ebss` (end of BSS).

**C Usage:**
```c
extern uint32_t _sdata; // These are addresses, not values!
extern uint32_t _edata;
extern uint32_t _sidata;

void startup() {
    uint32_t *src = &_sidata;
    uint32_t *dest = &_sdata;
    while (dest < &_edata) {
        *dest++ = *src++;
    }
}
```
