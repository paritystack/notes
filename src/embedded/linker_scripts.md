# Linker Scripts in Embedded Systems

## 1. Introduction to Linker Scripts

In the world of desktop software development (Windows, Linux, macOS), the operating system abstracts away the physical hardware. When you compile and run a C++ program on Linux, the OS loader places your executable into virtual memory, sets up the stack, and begins execution. The programmer rarely cares exactly *where* in physical RAM the program resides.

In **Embedded Systems** (especially "bare-metal" programming on microcontrollers), this abstraction does not exist. There is no OS loader to dynamically allocate memory or set up virtual address spaces. The programmer is directly responsible for mapping the compiled code and variables to the physical memory addresses provided by the microcontroller's hardware architecture.

This critical task is accomplished using a **Linker Script** (typically ending in `.ld`). The linker script must reflect the [ISA](isa.md)'s memory model and the [processor's](processor_design.md) physical memory map; the [startup code](startup_code.md) then copies `.data` and zeros `.bss` using the symbols the linker script exports.

A linker script is a text file that commands the linker (like GNU `ld` or LLVM `lld`) on how to combine various input object files (`.o`) into a single output executable (usually an `.elf` file, which is later converted to `.bin` or `.hex` for flashing). It dictates the exact memory layout of the final binary, specifying where the executable code (Flash), read-only constants (Flash), and readable/writable variables (SRAM) must be placed.

### Why are Linker Scripts Crucial?
1.  **Memory Heterogeneity:** Microcontrollers have multiple distinct memory banks (e.g., internal Flash, internal SRAM, Core-Coupled Memory (CCM), external SDRAM, EEPROM). The linker script manages this fragmentation.
2.  **Execution Requirements:** Code must execute from non-volatile memory (so it survives power cycles), but variables must live in RAM (so they can be modified).
3.  **Boot Process:** The microcontroller hardware expects certain structures (like the Vector Table) to exist at exact, hardcoded memory addresses (e.g., `0x00000000` or `0x08000000`).
4.  **Optimization:** Performance-critical code (like Interrupt Service Routines or DSP math) can be explicitly placed into tightly coupled, zero-wait-state RAM instead of slower Flash.

---

## 2. The Compilation and Linking Process

To understand linker scripts, one must understand the toolchain pipeline:

1.  **Preprocessing (`cpp`):** Expands macros (`#define`), includes headers (`#include`), and strips comments.
2.  **Compilation (`gcc` / `clang`):** Translates C/C++ source code into architecture-specific assembly language.
3.  **Assembly (`as`):** Translates assembly code into machine code, creating an **Object File (`.o`)**.
    *   *Crucial Detail:* At this stage, the object file is *relocatable*. If `main.c` calls a function `uart_init()` defined in `uart.c`, the compiler doesn't know the address of `uart_init`. It simply leaves a "blank space" and a note for the linker (a relocation entry) saying, "Please fill in the address of `uart_init` here later."
    *   The object file organizes its code and data into distinct **Sections** (e.g., `.text` for code, `.data` for variables).
4.  **Linking (`ld`):** The linker takes all object files, libraries (`.a`), and the **Linker Script**.
    *   **Section Merging:** It takes the `.text` section from `main.o`, the `.text` section from `uart.o`, and merges them into a single, massive `.text` output section.
    *   **Memory Assignment:** It uses the Linker Script to assign absolute physical addresses to these output sections.
    *   **Relocation (Symbol Resolution):** Now that everything has an absolute address, the linker goes back through the code and fills in all the "blank spaces" (e.g., replacing the placeholder for `uart_init` with `0x08000104`).

---

## 3. Basic Linker Script Syntax and Concepts

Linker scripts use a specialized, declarative language. The most common linker used in embedded systems is GNU `ld` (part of Binutils).

### The ENTRY Command
The `ENTRY` command specifies the exact instruction where program execution should begin. While the hardware vector table ultimately dictates the physical boot process, setting the entry point helps the linker understand what code is "alive" (preventing it from being stripped by garbage collection) and is used by debuggers (like GDB) to know where to halt initially.

```ld
ENTRY(Reset_Handler)
```

### The MEMORY Command
The `MEMORY` block defines the physical memory regions available on the target microcontroller. It tells the linker the names, attributes, start addresses, and lengths of the memory banks.

### The SECTIONS Command
The `SECTIONS` block is the heart of the linker script. It describes how the input sections (from the `.o` files) map to the output sections in the final ELF file, and which `MEMORY` regions those output sections should be placed in.

### The Location Counter (`.`)
The period (`.`) represents the **Location Counter**. It keeps track of the current memory address as the linker processes the `SECTIONS` block.
*   You can assign values to it to skip memory: `. = . + 0x1000;` (Advances the counter by 4KB).
*   You can read it to define symbols: `_start_of_data = .;` (Creates a symbol holding the current address).
*   It automatically increments as sections are laid down in memory.

---

## 4. The MEMORY Command in Detail

Before the linker can place code, it must know the physical bounds of the hardware. The `MEMORY` command provides this physical map.

### Syntax
```ld
MEMORY
{
  name [(attr)] : ORIGIN = origin, LENGTH = len
  ...
}
```

*   **`name`**: An arbitrary identifier for the region (e.g., `FLASH`, `RAM`, `CCMRAM`).
*   **`(attr)`**: Optional attributes that describe the properties of the memory. The linker uses these to automatically place sections if you don't explicitly specify a region in the `SECTIONS` block.
    *   `R` (Read-only)
    *   `W` (Read/Write)
    *   `X` (Executable)
    *   `A` (Allocatable)
    *   `I` / `L` (Initialized)
    *   `!` (Invert the attributes)
*   **`ORIGIN`**: The start address of the memory region. Can be abbreviated as `org` or `o`.
*   **`LENGTH`**: The size of the memory region in bytes. Can be abbreviated as `len` or `l`. Supports suffixes like `K` (Kilobytes) and `M` (Megabytes).

### Example: STM32F407 Microcontroller
An STM32F407 features 1MB of Flash, 112KB of main SRAM, and 64KB of Core Coupled Memory (CCM).

```ld
MEMORY
{
  /* Main execution memory */
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 1024K
  
  /* Main SRAM used for variables and stack */
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 112K
  
  /* Fast RAM coupled to the D-bus, excellent for DSP or critical data */
  CCMRAM (rw)     : ORIGIN = 0x10000000, LENGTH = 64K
  
  /* Backup SRAM that survives standby/reset if Vbat is present */
  BKPSRAM (rw)    : ORIGIN = 0x40024000, LENGTH = 4K
}
```

If the linker attempts to place a section into `FLASH` and the size of that section pushes the location counter past `0x08000000 + 1024K`, the linker will throw a fatal error:
`region 'FLASH' overflowed by XXX bytes`.

---

## 5. Standard Standard Sections Explained

Compilers organize compiled code and variables into standardized "sections". Understanding these sections is critical for writing a linker script.

### 5.1. `.text` (Executable Code)
Contains the actual machine instructions generated from your C/C++ functions.
*   **Characteristics:** Executable, Read-Only.
*   **Placement:** Placed in non-volatile `FLASH`.

### 5.2. `.rodata` (Read-Only Data)
Contains constants that cannot be modified at runtime.
*   **Examples:** String literals (`"Hello World"`), `const` variables (`const int pi = 3;`), lookup tables.
*   **Characteristics:** Read-Only.
*   **Placement:** Placed in non-volatile `FLASH` to save precious RAM.

### 5.3. `.data` (Initialized Data)
Contains global and `static` variables that are initialized to a non-zero value.
*   **Examples:** `int my_global = 42;`, `static float offset = 3.14;`.
*   **Characteristics:** Read/Write.
*   **The Paradox:** Because these variables have specific initial values, those values must be stored in `FLASH` so they exist when the device powers on. However, because they are variables (they can be changed by the program), they must reside in `RAM` during execution.
*   **Placement:** Requires a dual-address setup (VMA vs. LMA), which is resolved by the C-startup code.

### 5.4. `.bss` (Block Started by Symbol)
Contains global and `static` variables that are either uninitialized or explicitly initialized to zero.
*   **Examples:** `int counter;`, `static char buffer[1024];`, `int flags = 0;`.
*   **Characteristics:** Read/Write.
*   **Placement:** Placed in `RAM`.
*   **Optimization:** Because all these variables start as zero, the compiler does *not* store thousands of zeros in the Flash memory. Instead, it simply notes the total size of the `.bss` section. The C-startup code is responsible for writing zeros to this region of RAM before `main()` is called.

### 5.5. C++ Specific Sections
If compiling C++, the compiler generates additional sections for constructors and destructors of global/static objects.
*   **`.init_array` / `.ctors`:** Pointers to functions that must run *before* `main()` (constructors).
*   **`.fini_array` / `.dtors`:** Pointers to functions that must run *after* `main()` returns (destructors, though embedded systems rarely return from `main`).
*   **`.eh_frame` / `.ARM.exidx`:** Stack unwinding information used for C++ Exception Handling.

---

## 6. The SECTIONS Command in Detail

The `SECTIONS` command maps the input sections from object files into output sections in the ELF file and assigns them to `MEMORY` regions.

### Syntax
```ld
SECTIONS
{
  output_section_name [address] [(type)] :
  {
    input_section_command
    ...
  } > REGION [AT> LMA_REGION]
}
```

### Input Section Wildcards
You specify which input sections go into an output section using the syntax `filename(section_name)`. Usually, we use wildcards `*` to match all files.

```ld
SECTIONS
{
  .text :
  {
    /* Take the .text section from ALL input object files */
    *(.text)
    
    /* Take any section that starts with .text. from ALL input files.
       This is critical when compiling with -ffunction-sections, which
       puts every function in its own section (e.g., .text.uart_init)
       to allow the linker to garbage collect unused functions. */
    *(.text.*)
  } > FLASH
}
```

### VMA vs. LMA (Virtual vs. Load Memory Address)

This is the most confusing, yet most important concept in linker scripts.
Every section has two addresses:
1.  **LMA (Load Memory Address):** Where the data is stored "at rest" in the physical non-volatile memory (Flash/ROM). This is where the flashing tool writes the bytes.
2.  **VMA (Virtual Memory Address):** Where the CPU expects to find the data during execution.

For `.text` and `.rodata`, the VMA and LMA are identical (the CPU executes code directly out of Flash).

**For `.data`, the VMA and LMA are different.**
Variables in `.data` must be in RAM to be modified (VMA = RAM), but their initial values must be burned into Flash to survive a power cycle (LMA = Flash).

We use the `AT>` command to specify the LMA.

```ld
SECTIONS
{
  .data :
  {
    *(.data)           /* Input sections */
    *(.data.*)
  } > RAM AT> FLASH    /* VMA is RAM, LMA is FLASH */
}
```
If you omit `AT> FLASH`, the linker defaults the LMA to be the same as the VMA (RAM). If the flasher tool tries to burn the binary, it will attempt to write the initial values directly to RAM, which will be lost the moment the flasher disconnects power.

---

## 7. The Location Counter and Expressions

The location counter (`.`) tracks the current VMA address during linking. It can be manipulated to create gaps, align data, or calculate sizes.

### Alignment
CPUs (especially ARM Cortex-M) often require data to be aligned in memory. For example, a 32-bit integer should sit at an address divisible by 4. If you attempt an unaligned access (e.g., reading a 32-bit integer from address `0x20000001`), the CPU will trigger a hard fault.

We use the `ALIGN()` function in the linker script to ensure sections start at safe addresses.

```ld
SECTIONS
{
  .data :
  {
    . = ALIGN(4);      /* Force location counter to be a multiple of 4 */
    _sdata = .;        /* Record the start address */
    
    *(.data)
    
    . = ALIGN(4);      /* Ensure the end address is also aligned */
    _edata = .;        /* Record the end address */
  } > RAM AT> FLASH
}
```
*Note: Modifying the location counter (`. = ALIGN(4);`) injects padding bytes into the output section.*

---

## 8. Managing Symbols

Linker scripts define **symbols** that are exported to the C/C++ source code. These symbols do not hold data; they represent **memory addresses**.

### Defining Symbols
You create a symbol simply by assigning a value (usually the location counter) to a name.

```ld
_bss_start = .;
_heap_start = _bss_end;
_stack_top = ORIGIN(RAM) + LENGTH(RAM);
```

### The `PROVIDE` Keyword
If you define a symbol normally (`_my_sym = .;`), it will override any symbol with the same name in the C code, potentially causing silent bugs.
Using `PROVIDE` defines the symbol *only* if it is referenced but not defined in any of the compiled object files. It acts as a weak fallback.

```ld
PROVIDE(_stack_top = ORIGIN(RAM) + LENGTH(RAM));
```

### Using Linker Symbols in C Code
This is a frequent source of bugs for beginners. When you use a linker symbol in C, the C compiler treats the name as a variable, but the linker resolves it as an address.

**Incorrect Usage:**
```c
extern uint32_t _sdata;
// WRONG! This reads the 32-bit integer stored AT the address _sdata.
uint32_t start_address = _sdata; 
```

**Correct Usage:**
Because the symbol represents an address in the linker's mind, you must use the Address-Of operator (`&`) in C to extract that address. The actual type (`uint32_t`, `char`, etc.) doesn't matter, though `uint32_t` or an incomplete struct is standard convention.

```c
/* Declare the symbols defined in the linker script */
extern uint32_t _sdata;  /* Start of .data in RAM */
extern uint32_t _edata;  /* End of .data in RAM */
extern uint32_t _sidata; /* Start of .data initialization values in FLASH */

void my_function() {
    // CORRECT! Get the address the symbol represents.
    uint32_t *ram_start_ptr = &_sdata; 
}
```

---

## 9. The C Runtime (CRT) Startup Sequence

A C program does not start at `main()`. When a microcontroller resets, it jumps to a hardware-defined address (the Reset Vector), which points to a function usually written in assembly or naked C, called the **Reset Handler**.

The Reset Handler's primary job is to prepare the environment so `main()` can safely execute. This involves executing the dual-address paradox defined in the linker script.

### 9.1. The `.data` Copy Routine
The startup code must copy the initial values from Flash (LMA) to RAM (VMA).

**Linker Script Setup:**
```ld
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* VMA start */
    *(.data)
    *(.data.*)
    . = ALIGN(4);
    _edata = .;        /* VMA end */
  } > RAM AT> FLASH
  
  /* LOADADDR returns the absolute LMA of a section */
  _sidata = LOADADDR(.data);
```

**C Startup Code (`startup.c`):**
```c
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sidata;

void Reset_Handler(void) {
    uint32_t *src  = &_sidata; // Point to Flash
    uint32_t *dest = &_sdata;  // Point to RAM
    
    // Copy data from Flash to RAM
    while (dest < &_edata) {
        *dest++ = *src++;
    }
    
    // ... next step ...
}
```

### 9.2. The `.bss` Zero Routine
Uninitialized variables must be set to zero.

**Linker Script Setup:**
```ld
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)          /* Legacy uninitialized data */
    . = ALIGN(4);
    _ebss = .;
  } > RAM
```

**C Startup Code (`startup.c`):**
```c
extern uint32_t _sbss;
extern uint32_t _ebss;

    // ... continued from above ...

    uint32_t *bss_dest = &_sbss;
    
    // Initialize .bss to zero
    while (bss_dest < &_ebss) {
        *bss_dest++ = 0;
    }
```

### 9.3. Calling Constructors and `main()`
If C++ is used, the startup code must iterate through the `.init_array` section and execute all function pointers (to construct global objects) before finally branching to `main()`.

```c
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

    // Call static constructors
    int count = __init_array_end - __init_array_start;
    for (int i = 0; i < count; i++) {
        __init_array_start[i]();
    }

    // Initialize System/Clocks (vendor specific)
    SystemInit();

    // Call main application
    main();

    // main() should never return in an embedded system
    while(1);
}
```

---

## 10. Advanced Placements and Techniques

Beyond the standard sections, linker scripts provide immense power for optimizing and structuring embedded applications.

### 10.1. Tightly Coupled Memory (TCM) / CCM
Many high-performance microcontrollers (like STM32F4/F7/H7) have specialized RAM that runs at the exact clock speed of the CPU with zero wait states, bypassing the main bus matrix. This is ideal for intense DSP algorithms, PID loops, or critical Interrupt Service Routines (ISRs).

**C Code:**
We use compiler attributes to tag specific functions or variables to custom sections.
```c
__attribute__((section(".ccmram"))) 
void fast_dsp_filter(float* data) {
    // Critical math here...
}

__attribute__((section(".ccmram"))) 
float critical_buffer[1024];
```

**Linker Script:**
```ld
  .ccmram :
  {
    . = ALIGN(4);
    *(.ccmram)
    *(.ccmram.*)
    . = ALIGN(4);
  } > CCMRAM AT> FLASH
```
*(Note: If the `.ccmram` section contains initialized data or code, the startup routine must be modified to copy this section from Flash to CCMRAM, exactly like it does for `.data`.)*

### 10.2. The KEEP() Directive and Garbage Collection
Modern compilers use aggressive dead-code elimination. When compiling with `-ffunction-sections -fdata-sections` and linking with `--gc-sections`, the linker will discard any section that is not explicitly called by the program.

However, hardware structures like the **Vector Table** are never called by software; they are read directly by the CPU hardware during an interrupt. If the linker garbage collects the vector table, the device will immediately crash on boot.

The `KEEP()` directive overrides garbage collection, forcing the linker to retain the section.

```ld
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* NEVER strip this out! */
    . = ALIGN(4);
  } > FLASH
```

### 10.3. Uninitialized Data (No-Init)
Sometimes you want variables to survive a soft-reset (e.g., storing a crash dump, or maintaining a state machine across a watchdog reset). You don't want the C startup code to zero this data out, and you don't want the flasher to overwrite it.

**C Code:**
```c
__attribute__((section(".noinit"))) uint32_t crash_reason;
```

**Linker Script:**
The `(NOLOAD)` directive tells the linker/flasher that this section does not occupy space in the physical binary file and should not be downloaded to the device.
```ld
  .noinit (NOLOAD) :
  {
    . = ALIGN(4);
    *(.noinit)
    *(.noinit.*)
    . = ALIGN(4);
  } > RAM
```

### 10.4. Executing Code from RAM
Flash memory is inherently slow. While CPUs use caches to hide this latency, sometimes you need deterministic, absolute maximum performance. Furthermore, if your code is *reprogramming* the Flash memory (e.g., an OTA Bootloader), the CPU cannot execute code from Flash while the Flash controller is busy erasing a sector.

You must move the Flash-writing functions into RAM.

**C Code:**
```c
__attribute__((section(".ram_func"))) 
void flash_erase_sector(uint8_t sector) { ... }
```

**Linker Script:**
We treat this exactly like `.data`. It lives in RAM (VMA) but is stored in Flash (LMA).
```ld
  .ram_func :
  {
    . = ALIGN(4);
    _sram_func = .;
    *(.ram_func)
    *(.ram_func.*)
    . = ALIGN(4);
    _eram_func = .;
  } > RAM AT> FLASH
  
  _siram_func = LOADADDR(.ram_func);
```
*(Again, the C startup code must copy `_siram_func` to `_sram_func` before the code can be executed).*

### 10.5. DISCARDing Sections
If a library includes debugging information, legacy sections, or metadata that you absolutely do not want occupying space in your final binary, you can map them to the special `/DISCARD/` output section.

```ld
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
    *(.note.GNU-stack)
    *(.gnu_debuglink) 
    *(.gnu.lto_*)
  }
```

---

## 11. Anatomy of a Real-World Linker Script

Let's dissect a complete, production-grade linker script for an ARM Cortex-M microcontroller.

```ld
/* 1. Define the Entry Point */
ENTRY(Reset_Handler)

/* 2. Define the Highest Address of the Stack */
/* The stack grows downwards from the end of RAM */
_estack = ORIGIN(RAM) + LENGTH(RAM);

/* 3. Define Minimum Heap and Stack Sizes for validation */
_Min_Heap_Size  = 0x200;  /* 512 Bytes */
_Min_Stack_Size = 0x400;  /* 1024 Bytes */

/* 4. Define Memory Regions */
MEMORY
{
  FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 512K
  RAM (xrw)       : ORIGIN = 0x20000000, LENGTH = 128K
}

/* 5. Map Sections */
SECTIONS
{
  /* The startup code goes first into FLASH */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Vector table */
    . = ALIGN(4);
  } > FLASH

  /* The program code and other data goes into FLASH */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text.*)         /* .text.* sections (if -ffunction-sections) */
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata.*)
    . = ALIGN(4);
    _etext = .;        /* Define symbol for end of code */
  } > FLASH

  /* C++ Global Constructors */
  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } > FLASH
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } > FLASH

  /* Initialized data section. LMA in Flash, VMA in RAM */
  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* Create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data.*)
    . = ALIGN(4);
    _edata = .;        /* Define a global symbol at data end */
  } > RAM AT> FLASH

  /* Uninitialized data section. Placed in RAM */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;         /* Symbol at BSS start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;         /* Symbol at BSS end */
    __bss_end__ = _ebss;
  } > RAM

  /* User_heap_stack section, used to check that there is enough RAM left */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } > RAM

  /* Remove information from the standard libraries */
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }
}
```

### Explaining the Validation Block (`._user_heap_stack`)
The linker script does not natively know how large your stack or heap will grow at runtime. However, we defined `_Min_Stack_Size` and `_Min_Heap_Size` at the top.
In the `._user_heap_stack` section, we intentionally advance the location counter (`. = . + _Min_Heap_Size`) without outputting any data.
If the sum of `.data` + `.bss` + `Min_Heap` + `Min_Stack` exceeds the `LENGTH` of the `RAM` region defined in the `MEMORY` block, the linker will error out during compilation, warning you that your RAM is over-provisioned before you ever flash the board.

---

## 12. Directives, Includes, and Assertions

Linker scripts can be modular and perform logic checks.

### INCLUDE
You can break massive scripts into logical files. For example, keeping the memory layout separate from the section definitions makes porting to a new microcontroller easier.
```ld
/* memory.ld */
MEMORY { FLASH : ORIGIN = 0x08000000, LENGTH = 1M }

/* main.ld */
INCLUDE "memory.ld"
SECTIONS { ... }
```

### ASSERT
The `ASSERT` command evaluates a boolean expression. If false, it throws a fatal linking error with a custom message. Extremely useful for sanity checks.
```ld
/* Ensure the vector table doesn't exceed its hardware-allotted 1KB space */
ASSERT( (SIZEOF(.isr_vector) <= 0x400), "Error: Vector table is too large!" )

/* Ensure a bootloader and application do not overlap */
ASSERT( (_app_start > _bootloader_end), "Error: Application overlaps Bootloader!" )
```

### EXTERN
Forces the linker to include a specific symbol in the output, even if no other code references it. This prevents the garbage collector (`--gc-sections`) from stripping it out.
```ld
EXTERN(Reset_Handler)
```

---

## 13. Tooling and Debugging Linker Issues

Writing linker scripts is an iterative process, heavily reliant on command-line utilities (part of the GNU Binutils suite) to verify the results.

### 13.1. Generating a Map File
The Map File is the ultimate source of truth. It details exactly where the linker placed every function, variable, and section, and why certain libraries were pulled in.
**How to generate:** Pass `-Wl,-Map=output.map` to GCC during the linking phase.

**Example Map File Output:**
```text
Memory Configuration
Name             Origin             Length             Attributes
FLASH            0x08000000         0x00100000         xr
RAM              0x20000000         0x00020000         xrw

Linker script and memory map
.text           0x08000000      0x10f4
 *(.isr_vector)
 .isr_vector    0x08000000       0x188 startup_stm32.o
                0x08000000                g_pfnVectors
 *(.text)
 .text.main     0x08000188        0x42 main.o
                0x08000188                main
 .text.uart_init 0x080001ca       0x88 uart.o
                0x080001ca                uart_init
```
*Analysis:* We can clearly see the vector table is at the very beginning of Flash. The `main` function starts immediately after at `0x08000188`, occupying `0x42` (66) bytes.

### 13.2. Using `objdump`
`arm-none-eabi-objdump` allows you to inspect the compiled ELF file.
*   **View Sections:** `objdump -h firmware.elf` (Shows sizes, VMAs, and LMAs of all sections).
*   **Disassemble:** `objdump -d firmware.elf` (Translates machine code back to assembly, allowing you to see exactly what address a branch instruction targets).

### 13.3. Using `nm`
`arm-none-eabi-nm` lists the symbols from object files or executables.
*   `nm -S --size-sort firmware.elf` (Lists all variables and functions sorted by how much memory they consume. Invaluable for finding RAM hogs).

### 13.4. Using `readelf`
`readelf -S firmware.elf` provides a more detailed, raw view of the ELF section headers than `objdump`.

### 13.5. Common Linker Errors

*   **`undefined reference to 'uart_init'`**
    *   *Cause:* You called `uart_init()` in your C code, but the linker couldn't find its definition in any of the provided `.o` or `.a` files.
    *   *Fix:* Ensure `uart.c` is being compiled and linked, or check for typos (e.g., C++ name mangling issues missing an `extern "C"`).
*   **`region 'FLASH' overflowed by 1024 bytes`**
    *   *Cause:* The sum of `.text`, `.rodata`, and the LMA of `.data` exceeds the `LENGTH` defined in the `MEMORY` block for `FLASH`. Your program is too big for the chip.
    *   *Fix:* Optimize code size (`-Os`), remove unused libraries, or upgrade to a larger microcontroller.
*   **`multiple definition of 'global_buffer'`**
    *   *Cause:* You defined `int global_buffer[10];` in a header file included by multiple `.c` files, causing the compiler to emit the symbol multiple times.
    *   *Fix:* Use `extern int global_buffer[10];` in the header, and define the array in exactly *one* `.c` file.
*   **`section .data LMA overlaps previous sections`**
    *   *Cause:* You are trying to pack sections into Flash, but your location counter logic or hardcoded addresses caused a collision.

---

## 14. GNU ld vs. LLVM LLD

Historically, the embedded world exclusively used GNU Binutils (`ld` or `ld.bfd`). However, LLVM's linker (`lld`) is gaining immense popularity.

*   **Speed:** `lld` is designed to be significantly faster than GNU `ld`, often linking large projects 2x to 5x quicker.
*   **Compatibility:** `lld` is intended to be a drop-in replacement. It parses the exact same GNU Linker Script syntax.
*   **Usage:** You can instruct clang or gcc to use lld by passing the flag `-fuse-ld=lld`.
*   **Differences:** While syntactically compatible, `lld` handles Garbage Collection (`--gc-sections`) and symbol resolution slightly more aggressively than GNU `ld`. A sloppy linker script that implicitly relied on undefined GNU behaviors might break when transitioning to LLVM.

## Where this connects

- [ISA](isa.md) — the ISA defines alignment requirements, endianness, and section conventions the linker script must honour
- [Processor Design](processor_design.md) — placing ISR code in TCM (zero-wait-state RAM) requires explicit linker script placement
- [Startup Code](startup_code.md) — uses linker-exported `_sdata`, `_edata`, `_sbss`, `_ebss` symbols to initialise RAM
- [Memory Management](memory_management.md) — linker sections define the pool allocator's and stack's fixed addresses
- [OTA Updates](ota_updates.md) — dual-bank OTA requires the linker script to know each bank's base address and size

## Conclusion

The Linker Script is the bridge between software architecture and hardware architecture. Mastering it transitions a developer from simply writing algorithms to engineering a complete, robust embedded system capable of complex memory management, bootloading, and high-performance execution.