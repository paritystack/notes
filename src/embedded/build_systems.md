# Build Systems for Embedded

## Overview

Embedded builds differ from typical desktop builds in a few key ways:

- **Cross-compilation** — host x86_64, target ARM Thumb / RISC-V / Xtensa.
- **No standard library you can rely on** — newlib-nano, picolibc, or no libc at all.
- **Linker script controls memory layout** — sections, regions, alignment.
- **Output is not an executable** — it's a `.bin` / `.hex` / `.elf` you flash.
- **Per-MCU compile flags** matter a lot (`-mcpu=cortex-m4`, `-mfpu=fpv4-sp-d16`, `-mfloat-abi=hard`).

This doc covers the four mainstream options: Make, CMake, PlatformIO, and Zephyr's `west`. (Vendor IDEs like STM32CubeIDE, MCUXpresso, IAR are usually just Eclipse/proprietary GUIs over Make.)

## Toolchain Anatomy

Every embedded build needs at minimum:

```
arm-none-eabi-gcc           compiler
arm-none-eabi-as            assembler
arm-none-eabi-ld            linker
arm-none-eabi-objcopy       elf → bin / hex
arm-none-eabi-objdump       disassembler
arm-none-eabi-size          section sizes
arm-none-eabi-nm            symbol table
arm-none-eabi-gdb           debugger
arm-none-eabi-newlib        libc
```

For ARM Cortex-M, install from ARM's GNU Toolchain releases (latest is GCC 13). For ESP32, install `xtensa-esp32-elf-` or `riscv32-esp-elf-`. For RISC-V, use `riscv-none-elf-` or vendor-specific.

Critical compile flags:

```
-mcpu=cortex-m4                target CPU
-mthumb                        Thumb instruction set (mandatory on M)
-mfpu=fpv4-sp-d16              FPU variant
-mfloat-abi=hard               hardware float ABI (or 'soft' / 'softfp')
-mabi=aapcs                    ARM procedure call standard
-ffunction-sections            put each function in its own section
-fdata-sections                each data item in its own section
-Wl,--gc-sections              GC unused sections at link time → smaller binary
-specs=nano.specs              newlib-nano (smaller libc)
-specs=nosys.specs             stub syscalls so libc links
```

`-ffunction-sections + -Wl,--gc-sections` is the single biggest size win on most projects (often 30-50% smaller).

## Make

Lowest common denominator. Every embedded toolchain works with Make. Pattern from STM32CubeMX-generated Makefiles:

```make
TARGET     = firmware
PREFIX     = arm-none-eabi-
CC         = $(PREFIX)gcc
LD         = $(PREFIX)gcc
OBJCOPY    = $(PREFIX)objcopy
SIZE       = $(PREFIX)size

MCU_FLAGS  = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS     = $(MCU_FLAGS) -Wall -Os -g3 \
             -ffunction-sections -fdata-sections \
             -DSTM32F407xx -DUSE_HAL_DRIVER \
             -Iinc -Icmsis/core -Icmsis/device -Ihal/inc

LDSCRIPT   = STM32F407VGTx_FLASH.ld
LDFLAGS    = $(MCU_FLAGS) -specs=nano.specs -specs=nosys.specs \
             -T$(LDSCRIPT) -Wl,--gc-sections -Wl,-Map=$(TARGET).map

C_SOURCES  = $(wildcard src/*.c) $(wildcard hal/src/*.c)
ASM_SOURCES = startup_stm32f407xx.s
OBJECTS    = $(C_SOURCES:.c=.o) $(ASM_SOURCES:.s=.o)

all: $(TARGET).elf $(TARGET).bin $(TARGET).hex
	$(SIZE) $(TARGET).elf

$(TARGET).elf: $(OBJECTS)
	$(LD) $(LDFLAGS) $^ -o $@

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.hex: %.elf
	$(OBJCOPY) -O ihex $< $@

clean:
	rm -f $(OBJECTS) $(TARGET).elf $(TARGET).bin $(TARGET).hex $(TARGET).map

flash: $(TARGET).bin
	st-flash write $@ 0x08000000

debug: $(TARGET).elf
	$(PREFIX)gdb $< -ex "target extended-remote :3333"

.PHONY: all clean flash debug
```

**Pros**: simple, transparent, ubiquitous. Easy to read what's happening.
**Cons**: dependency tracking is manual (use `-MMD -MP` to autogenerate `.d` files). No multi-target / cross-platform abstraction. Adding a new vendor SDK = rewriting half the Makefile.

## CMake

The current default for serious embedded work. Modern STM32, Zephyr, NuttX, and most non-trivial projects use CMake.

### Toolchain File

The trick to cross-compiling with CMake is a **toolchain file** passed via `-DCMAKE_TOOLCHAIN_FILE=`:

```cmake
# arm-none-eabi.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_SIZE    arm-none-eabi-size)

# Without this, CMake tries to link a test executable during the
# compiler-check step and fails because there's no startup code yet.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
```

### Project CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(firmware LANGUAGES C ASM)

set(MCU_FLAGS -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard)

add_executable(firmware.elf
    src/main.c
    src/sensor.c
    startup_stm32f407xx.s
)

target_compile_options(firmware.elf PRIVATE
    ${MCU_FLAGS}
    -Wall -Os -g3
    -ffunction-sections -fdata-sections
)
target_compile_definitions(firmware.elf PRIVATE
    STM32F407xx USE_HAL_DRIVER
)
target_include_directories(firmware.elf PRIVATE
    inc cmsis/core cmsis/device hal/inc
)
target_link_options(firmware.elf PRIVATE
    ${MCU_FLAGS}
    -T${CMAKE_SOURCE_DIR}/STM32F407VGTx_FLASH.ld
    -specs=nano.specs -specs=nosys.specs
    -Wl,--gc-sections
    -Wl,-Map=firmware.map
)

# Post-build: generate .bin and .hex, print size
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:firmware.elf> firmware.bin
    COMMAND ${CMAKE_OBJCOPY} -O ihex   $<TARGET_FILE:firmware.elf> firmware.hex
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:firmware.elf>
)
```

Build:

```bash
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=arm-none-eabi.cmake
ninja -C build
```

**Pros**: scales to multi-component projects, integrates well with VSCode/CLion/IDEs, large ecosystem (FetchContent for pulling vendor SDKs).
**Cons**: steeper learning curve, error messages can be cryptic, sometimes feels like it's fighting you.

## PlatformIO

Python-based meta-build system. You declare your platform/board in `platformio.ini` and it handles toolchain installation, dependency fetching, build, flash, monitor.

```ini
# platformio.ini
[env:nucleo_f407vg]
platform = ststm32
board = nucleo_f407vg
framework = stm32cube
build_flags =
    -Os
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections
lib_deps =
    bblanchon/ArduinoJson@^7.0.0
    adafruit/Adafruit BME280 Library@^2.2.0
monitor_speed = 115200
upload_protocol = stlink
```

```bash
pio run                # build
pio run -t upload      # flash
pio device monitor     # serial console
```

**Pros**: zero-setup (installs toolchains, SDKs, libs automatically), supports 1000+ boards, integrates with VSCode beautifully, great for hobbyists and prototyping.
**Cons**: less control over the underlying build flags, slower than Ninja for large projects, opaque (you don't always know what it's actually compiling), heavy disk footprint (toolchains/SDKs balloon).

Use for: hobby projects, multi-board portability, getting started fast.
Avoid for: production firmware with custom toolchains or strict reproducibility.

## Zephyr west

Zephyr RTOS uses a custom meta-tool, `west`, around CMake + Kconfig + DTS (device tree).

```bash
west init -m https://github.com/zephyrproject-rtos/zephyr my_workspace
cd my_workspace
west update
west build -b nucleo_f407vg samples/basic/blinky
west flash
```

The board is described in **device tree** (`.dts`), kernel/config in **Kconfig**, build orchestrated by CMake. Zephyr is opinionated but powerful — its driver model means the same `gpio_pin_set()` call works on STM32, Nordic, NXP, etc.

**Pros**: huge driver ecosystem, vendor-neutral APIs, real RTOS, treats embedded like the kernel does (device tree, kconfig).
**Cons**: steep learning curve, large baseline size, "Zephyr way" can clash with simple bare-metal habits.

Use for: portable production firmware, projects targeting multiple MCU families, anything that benefits from a full RTOS + driver framework.

## Bazel and Bazel-likes

Bazel can build embedded firmware (toolchain rules exist for Cortex-M), but the ecosystem is thin. Worth it only if you're already a Bazel shop and want firmware in the same build graph as your services.

## SDKs and Their Build Systems

| Vendor | SDK | Build |
|--------|-----|-------|
| ST | STM32Cube | Make (CubeIDE generates), CMake (recent), HAL drivers |
| Nordic | nRF Connect SDK (NCS) | west / CMake (Zephyr-based) |
| Espressif | ESP-IDF | CMake + Kconfig + python wrapper (`idf.py`) |
| NXP | MCUXpresso | Eclipse + CMake export, Zephyr too |
| Raspberry Pi | Pico SDK | CMake |
| Renesas | RA / RX | e² studio (Eclipse), Make |
| Microchip | MPLAB X / Harmony | proprietary or Make export |

ESP-IDF deserves special mention — its `idf.py build` wraps CMake but adds component / Kconfig / partition-table magic that's hard to use without the wrapper.

```bash
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py menuconfig          # Kconfig TUI
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## Newlib vs Newlib-nano vs Picolibc

The C standard library matters for binary size:

| Library | Footprint | Notes |
|---------|-----------|-------|
| **newlib (full)** | ~50-80 KB minimum just for `printf` | Default in arm-none-eabi-gcc |
| **newlib-nano** | ~10-20 KB | `-specs=nano.specs`; reduced `printf`/`scanf` |
| **picolibc** | ~5-10 KB | Newer, more compact, used by Zephyr |
| **none** | 0 | Bare-metal; you implement what you need |

For most projects: `-specs=nano.specs -specs=nosys.specs`. Add `-u _printf_float` if you need `%f` support (nano strips it by default to save space).

## Output Files Explained

```
firmware.elf    binary with symbols, sections, debug info — used by GDB and probes
firmware.bin    raw flash image (no metadata) — for some flashing tools
firmware.hex    Intel HEX text format — addresses + data — universal flash format
firmware.map    text file: where every symbol/section landed in memory — invaluable
```

The `.map` file is your best debugging friend when you're trying to figure out "why is my flash 90% full when my source code is 10 KB?". Look for unexpected large symbols, missed `-fdata-sections`, or accidental linking of full libc.

## Reproducibility

- **Pin the toolchain version.** `arm-none-eabi-gcc 13.2.Rel1` is not the same as `13.2.0`.
- **Set `SOURCE_DATE_EPOCH`** if you care about deterministic builds (eliminates `__DATE__` / `__TIME__` variance).
- **Lock SDK versions.** ESP-IDF, Zephyr, etc. update often.
- **Check in or pin a `Dockerfile`** with the exact toolchain. Embedded firmware that you can't rebuild years from now is a liability.

## Common Pitfalls

### Pitfall 1: Float-ABI Mismatch

CMSIS-DSP built `-mfloat-abi=hard`, your project built `softfp` → linker error "uses VFP register arguments, ... does not". All objects in a link must agree on float ABI.

### Pitfall 2: Missing `--gc-sections`

Without `-ffunction-sections -fdata-sections -Wl,--gc-sections`, every function from libc + HAL gets linked, blowing up the binary. Often a 3× size difference.

### Pitfall 3: Linker Script Mismatch

Linker script says flash starts at 0x08000000, but `objcopy -O binary` produces a file that should be flashed at 0x08000000 — and you flash it at 0x00000000. Symptom: nothing happens, or hangs in vector fetch.

### Pitfall 4: Including All HAL Sources

Naïve `*.c` glob pulls in `stm32f4xx_hal_*.c` for every peripheral. Each adds ~5-50 KB even if unused (without GC sections). Be selective.

### Pitfall 5: Forgetting `-D` for Device Selection

`#error "Please select first the target STM32F4xx device used"` — you forgot `-DSTM32F407xx` (or your chip's macro).

### Pitfall 6: Mixing Compiler and Newlib Versions

System newlib doesn't match the compiler's expected ABI → mysterious `_close_r` / `_sbrk_r` linker errors. Use the newlib that shipped with your toolchain.

### Pitfall 7: Stale Build State After Toolchain Update

`build/` directory accumulates objects compiled with the old toolchain. Symptom: random unresolved symbols. `make clean` or wipe `build/`.

### Pitfall 8: `printf` Pulls in Float Support You Don't Need

`printf("%d\n", x)` somehow brings in 8 KB of float code. Cause: newlib detects any `%f` in your code (even in unused branches) and links the float printer. `-u _printf_float` (force include) or `-specs=nano.specs` (force exclude) control this.

## Quick Decision Guide

```
Hobby / prototyping?           → PlatformIO
Single MCU family, small team? → Make
Multi-MCU, scaling project?    → CMake
Multi-vendor or want RTOS?     → Zephyr west
ESP32?                         → ESP-IDF (idf.py)
Nordic?                        → NCS (west)
```

## Summary

1. **Toolchain = compiler + binutils + libc + debugger.** Pin versions.
2. **`-ffunction-sections -fdata-sections -Wl,--gc-sections`** is the size win.
3. **Make** = simple and ubiquitous; **CMake** = the modern default.
4. **PlatformIO** = best for prototyping and multi-board; **Zephyr west** = best for production multi-vendor.
5. **Linker scripts and float-ABI mismatches** are the top reasons builds break.
6. **`.map` file** tells you where flash went.
7. **newlib-nano** unless you need full libc.
8. **Make builds reproducible** — Docker, pinned toolchains.

## See Also

- [CMSIS](cmsis.md) — what you're typically pulling into the build
- [Linker Scripts](linker_scripts.md) — controls output layout
- [Bootloaders](bootloaders.md) — separate build for the bootloader image
- [GDB Embedded](gdb_embedded.md) — using the build output to debug
