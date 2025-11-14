# Embedded Systems Debugging

Comprehensive guide to debugging embedded systems, covering hardware and software debugging techniques, tools, and best practices.

## Overview

Embedded systems debugging is fundamentally different from desktop application debugging due to resource constraints, real-time requirements, and hardware interactions. Effective debugging requires understanding both software and hardware aspects of the system.

**Key Challenges:**
- Limited debugging resources (memory, CPU cycles)
- Real-time constraints
- Hardware dependencies and interactions
- No console or display output
- Difficult to reproduce timing-dependent bugs
- Production hardware may lack debug interfaces

**Debugging Approaches:**
- **Hardware Debugging**: JTAG, SWD, debuggers
- **Software Debugging**: Printf debugging, logging, assertions
- **Signal Analysis**: Logic analyzers, oscilloscopes
- **Protocol Analysis**: Bus analyzers (I2C, SPI, CAN)
- **Memory Analysis**: Memory dumps, stack traces

---

## Hardware Debug Interfaces

### JTAG (Joint Test Action Group)

JTAG is the industry-standard debugging interface for embedded systems.

**Features:**
- Full debugging capabilities (run, stop, step, breakpoints)
- Flash programming
- Boundary scan testing
- IEEE 1149.1 standard

**Pin Configuration:**
```
JTAG 20-Pin Header (ARM Standard)

┌─────────────────┐
│ 1  VTref   GND 2│
│ 3  nTRST   GND 4│
│ 5  TDI     GND 6│
│ 7  TMS    RTCK 8│
│ 9  TCK     GND 10│
│11  RTCK    GND 12│
│13  TDO     GND 14│
│15  RESET   GND 16│
│17  NC      GND 18│
│19  NC      GND 20│
└─────────────────┘

Essential Pins:
- TDI  (Test Data In)
- TDO  (Test Data Out)
- TMS  (Test Mode Select)
- TCK  (Test Clock)
- TRST (Test Reset) - Optional
```

**JTAG Chain:**
```
Debugger ──TCK──> Device1 ──> Device2 ──> Device3
         ──TDI──>   │          │          │
         <─TDO───   │          │          │
         ──TMS──────┴──────────┴──────────┘
```

### SWD (Serial Wire Debug)

SWD is ARM's 2-pin alternative to JTAG, offering similar capabilities with fewer pins.

**Features:**
- Only 2 pins required (SWDIO, SWCLK)
- Same debugging capabilities as JTAG
- Lower pin count
- Higher speed than JTAG
- ARM Cortex-M standard

**Pin Configuration:**
```
SWD Interface

┌──────────────┐
│ 1  VCC       │
│ 2  SWDIO     │  Data I/O
│ 3  GND       │
│ 4  SWCLK     │  Clock
│ 5  GND       │
│ 6  SWO       │  Serial Wire Output (optional)
│ 7  NC        │
│ 8  NC        │
│ 9  GND       │
│10  RESET     │  (optional but recommended)
└──────────────┘

Minimal SWD:
- SWDIO (Data)
- SWCLK (Clock)
- GND
- VCC (for level reference)
```

**SWD vs JTAG Comparison:**

| Feature | JTAG | SWD |
|---------|------|-----|
| Pins | 4-5 | 2 |
| Speed | Medium | Fast |
| Multi-device | Yes (chain) | No |
| ARM Specific | No | Yes |
| Recommended for | Multi-chip, production test | Single-chip, development |

### SWO (Serial Wire Output)

SWO provides one-way communication from target to debugger.

**Features:**
- Printf-style debugging
- ITM (Instrumentation Trace Macrocell)
- Minimal overhead
- Single pin (shares JTAG/SWD connector)

**Uses:**
```c
// ITM Printf implementation
void ITM_SendChar(char ch) {
    while (ITM->PORT[0].u32 == 0);  // Wait for port ready
    ITM->PORT[0].u8 = ch;
}

// Usage
ITM_SendChar('H');
ITM_SendChar('e');
ITM_SendChar('l');
ITM_SendChar('l');
ITM_SendChar('o');
ITM_SendChar('\n');
```

---

## Debug Tools and Hardware

### ST-LINK

Official debugger for STM32 microcontrollers.

**Versions:**
- **ST-LINK/V2**: Standalone debugger
- **ST-LINK/V2-1**: Integrated on Nucleo boards
- **ST-LINK/V3**: Latest version with faster speed

**Features:**
```
ST-LINK Capabilities:
- JTAG/SWD debugging
- Flash programming
- Virtual COM port (V2-1, V3)
- Voltage levels: 1.65V - 5.5V
- Speed: Up to 4 MHz (V3)
- Mass storage drag-and-drop (Nucleo boards)
```

**Connection Example:**
```
ST-LINK V2          STM32 Target
┌─────────┐         ┌────────────┐
│  SWDIO  ├────────>│  SWDIO     │
│  SWCLK  ├────────>│  SWCLK     │
│  GND    ├────────>│  GND       │
│  3.3V   ├────────>│  VDD       │ (optional)
│  RESET  ├────────>│  NRST      │ (optional)
│  SWO    ├<────────│  SWO (PB3) │ (optional)
└─────────┘         └────────────┘
```

**OpenOCD Configuration:**
```bash
# OpenOCD with ST-LINK
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

# Connect GDB
arm-none-eabi-gdb firmware.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) load
(gdb) continue
```

### J-Link

Professional debugger from SEGGER, supporting multiple architectures.

**Features:**
- Ultra-fast flash programming
- RTT (Real-Time Transfer) for printf debugging
- Unlimited flash breakpoints
- Flash download up to 3 MB/s
- Support for 5000+ devices

**J-Link Models:**

| Model | Speed | Features | Price Range |
|-------|-------|----------|-------------|
| **J-Link BASE** | Standard | Basic debugging | $400 |
| **J-Link PLUS** | Fast | Flash breakpoints | $500 |
| **J-Link ULTRA+** | Ultra-fast | High-speed trace | $1500 |
| **J-Link EDU** | Standard | Educational only | $60 |

**RTT (Real-Time Transfer):**
```c
#include "SEGGER_RTT.h"

// Initialization (in main)
SEGGER_RTT_Init();

// Printf-style output
SEGGER_RTT_printf(0, "Counter: %d\n", counter);

// Formatted output
SEGGER_RTT_WriteString(0, "Hello from RTT!\n");

// Read input
char input[32];
int bytes = SEGGER_RTT_Read(0, input, sizeof(input));
```

**RTT Viewer:**
```bash
# Terminal viewer for RTT output
JLinkRTTViewer

# Command line RTT client
JLinkRTTClient
```

### Black Magic Probe

Open-source, standalone ARM Cortex debugger.

**Features:**
- No OpenOCD required
- Native GDB support
- Built-in GDB server
- SWD and JTAG support
- USB to serial converter

**Usage:**
```bash
# Scan for devices
arm-none-eabi-gdb
(gdb) target extended-remote /dev/ttyACM0
(gdb) monitor swdp_scan
(gdb) attach 1

# Flash and debug
(gdb) load firmware.elf
(gdb) run
```

### CMSIS-DAP

ARM's open-source debug adapter protocol.

**Features:**
- Open standard
- USB HID interface
- No drivers required
- Cross-platform support

**Example Implementations:**
- DAPLink (ARM official)
- PyOCD (Python-based)
- OpenOCD support

---

## GDB for Embedded Systems

### ARM GDB Basics

```bash
# Start GDB with ELF file
arm-none-eabi-gdb firmware.elf

# Connect to OpenOCD
(gdb) target remote localhost:3333

# Connect to J-Link GDB server
(gdb) target remote localhost:2331

# Reset and halt target
(gdb) monitor reset halt

# Load program to flash
(gdb) load

# Verify flash
(gdb) compare-sections
```

### Essential GDB Commands

```gdb
# Execution control
continue (c)              # Continue execution
step (s)                  # Step into
next (n)                  # Step over
finish                    # Step out
until <line>              # Run until line

# Breakpoints
break main                # Break at function
break file.c:123          # Break at line
break *0x08000100         # Break at address
info breakpoints          # List breakpoints
delete 1                  # Delete breakpoint 1
disable 2                 # Disable breakpoint 2
enable 2                  # Enable breakpoint 2

# Hardware breakpoints (limited on embedded)
hbreak main               # Hardware breakpoint

# Watchpoints
watch variable            # Break on write
rwatch variable           # Break on read
awatch variable           # Break on read/write

# Memory examination
x/10x 0x20000000         # Examine 10 words (hex)
x/10i main               # Disassemble 10 instructions
x/s 0x08001000           # Examine string
info registers           # Show all registers
info reg r0 r1 r2        # Show specific registers

# Memory modification
set variable = 0x42
set {int}0x20000000 = 100

# Stack examination
backtrace (bt)           # Show call stack
frame 0                  # Select frame
info frame               # Current frame info
info locals              # Local variables
info args                # Function arguments

# Register access
info registers           # All registers
print $r0                # Read R0
set $r0 = 0x1234        # Write R0
print $pc                # Program counter
print $sp                # Stack pointer
```

### Advanced GDB Techniques

```gdb
# Define custom commands
define reset_run
  monitor reset halt
  load
  continue
end

# Pretty printing structures
print *myStruct
print myStruct.field

# Casting
print (uint32_t*)0x20000000
print *(uint32_t*)0x40021000  # Read peripheral register

# Call functions (dangerous in embedded!)
call printf("Debug: %d\n", value)

# Memory dump to file
dump binary memory dump.bin 0x20000000 0x20001000

# Conditional breakpoints
break main if counter > 100

# Command scripts
source debug_script.gdb

# Save breakpoints
save breakpoints bp.gdb

# Python scripting
python print("Custom debug output")
```

### GDB Init File (.gdbinit)

```gdb
# .gdbinit for STM32 debugging

# Connect to OpenOCD
target remote localhost:3333

# Enable TUI mode
#tui enable

# Load symbols
file firmware.elf

# Custom reset command
define reset_run
  monitor reset halt
  load
  monitor reset halt
  continue
end

# Custom flash command
define flash
  monitor reset halt
  load
  monitor reset halt
end

# Register display format
set print pretty on
set print array on

# Automatic peripheral register display
define show_gpio
  printf "GPIOA_IDR: 0x%08x\n", *(uint32_t*)0x40020010
  printf "GPIOA_ODR: 0x%08x\n", *(uint32_t*)0x40020014
end

# SVD (System View Description) support
# Requires GDB with SVD plugin
#svd load STM32F407.svd
```

---

## OpenOCD (Open On-Chip Debugger)

### Installation and Setup

```bash
# Install OpenOCD
# Ubuntu/Debian
sudo apt-get install openocd

# macOS
brew install openocd

# From source
git clone https://github.com/openocd-org/openocd.git
cd openocd
./bootstrap
./configure
make
sudo make install
```

### Configuration Files

**Interface Configuration:**
```tcl
# stlink.cfg
source [find interface/stlink.cfg]

# Set SWD mode
transport select hla_swd

# Adapter speed
adapter speed 4000
```

**Target Configuration:**
```tcl
# stm32f4x.cfg
source [find target/stm32f4x.cfg]

# Custom reset
reset_config srst_only

# Work area (RAM for flash programming)
$_TARGETNAME configure -work-area-phys 0x20000000
$_TARGETNAME configure -work-area-size 0x10000
```

### OpenOCD Commands

```bash
# Start OpenOCD
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

# Telnet to OpenOCD (default port 4444)
telnet localhost 4444

# OpenOCD commands via telnet
> reset halt             # Reset and halt
> flash write_image erase firmware.hex
> reset run             # Reset and run
> mdw 0x20000000 10     # Memory display word
> mww 0x20000000 0x1234 # Memory write word
> reg                   # Show registers
> step                  # Single step
> resume                # Continue execution
> shutdown              # Close OpenOCD
```

### OpenOCD Flash Programming

```bash
# Erase flash
> flash erase_sector 0 0 last

# Write to flash
> flash write_image erase firmware.bin 0x08000000
> flash write_image erase firmware.hex
> flash write_image erase firmware.elf

# Verify flash
> verify_image firmware.bin 0x08000000

# Flash info
> flash info 0
> flash banks
```

### OpenOCD with GDB

```bash
# Start OpenOCD (terminal 1)
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg

# GDB (terminal 2)
arm-none-eabi-gdb firmware.elf
(gdb) target remote localhost:3333
(gdb) monitor reset halt
(gdb) load
(gdb) monitor reset halt
(gdb) continue
```

### Custom OpenOCD Scripts

```tcl
# custom_flash.cfg
# Combined interface and target configuration

# Interface
source [find interface/stlink.cfg]
transport select hla_swd
adapter speed 4000

# Target
source [find target/stm32f4x.cfg]

# Custom procedures
proc flash_firmware {} {
    program firmware.elf verify reset
}

proc mass_erase {} {
    flash erase_sector 0 0 last
}

# Auto-run on startup
init
reset halt
```

**Usage:**
```bash
openocd -f custom_flash.cfg -c "flash_firmware" -c "shutdown"
```

---

## Software Debugging Techniques

### Printf Debugging

**Semihosting:**
```c
// Enable semihosting in GDB
// (gdb) monitor arm semihosting enable

#include <stdio.h>

int main(void) {
    printf("Starting application...\n");
    printf("Counter: %d\n", counter);
    return 0;
}
```

**UART Debugging:**
```c
void uart_puts(const char* str) {
    while (*str) {
        while (!(USART1->SR & USART_SR_TXE));
        USART1->DR = *str++;
    }
}

// Usage
uart_puts("Debug: Entered ISR\n");
```

**Custom Printf:**
```c
#include <stdarg.h>

void debug_printf(const char* format, ...) {
    char buffer[128];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    uart_puts(buffer);
}

// Usage
debug_printf("Value: %d, Status: 0x%02X\n", value, status);
```

### Assertions

```c
// Simple assertion
#define ASSERT(x) \
    if (!(x)) { \
        uart_puts("ASSERT FAILED: " #x "\n"); \
        while(1); \
    }

// Assertion with location
#define ASSERT_LOC(x) \
    if (!(x)) { \
        debug_printf("ASSERT: %s:%d\n", __FILE__, __LINE__); \
        while(1); \
    }

// Usage
ASSERT(buffer != NULL);
ASSERT(size <= MAX_BUFFER_SIZE);
```

### Logging Framework

```c
typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR
} log_level_t;

static log_level_t current_level = LOG_LEVEL_INFO;

void log_message(log_level_t level, const char* format, ...) {
    if (level < current_level) return;

    const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    char buffer[128];
    va_list args;

    // Timestamp
    uint32_t ticks = HAL_GetTick();
    snprintf(buffer, sizeof(buffer), "[%lu][%s] ", ticks, level_str[level]);
    uart_puts(buffer);

    // Message
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    uart_puts(buffer);
    uart_puts("\n");
}

// Convenience macros
#define LOG_DEBUG(...) log_message(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  log_message(LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WARN(...)  log_message(LOG_LEVEL_WARNING, __VA_ARGS__)
#define LOG_ERROR(...) log_message(LOG_LEVEL_ERROR, __VA_ARGS__)

// Usage
LOG_INFO("System initialized");
LOG_WARN("Buffer nearly full: %d/%d", count, MAX_SIZE);
LOG_ERROR("I2C timeout on address 0x%02X", addr);
```

### Stack Usage Monitoring

```c
// Fill stack with pattern at startup
void stack_fill(void) {
    extern uint32_t _sstack;  // Stack start (linker symbol)
    extern uint32_t _estack;  // Stack end
    uint32_t* p = &_sstack;

    while (p < &_estack) {
        *p++ = 0xDEADBEEF;
    }
}

// Check stack usage
uint32_t stack_usage(void) {
    extern uint32_t _sstack;
    uint32_t* p = &_sstack;
    uint32_t count = 0;

    while (*p == 0xDEADBEEF) {
        p++;
        count += 4;
    }

    return count;  // Unused stack in bytes
}

// Usage
stack_fill();  // Call before main
// ... application runs ...
uint32_t unused = stack_usage();
LOG_INFO("Stack unused: %lu bytes", unused);
```

### Watchdog Debugging

```c
// Watchdog with debug output
void feed_watchdog(const char* location) {
    HAL_IWDG_Refresh(&hiwdg);
    LOG_DEBUG("WDG fed from: %s", location);
}

#define FEED_WATCHDOG() feed_watchdog(__func__)

// In application
void task1(void) {
    FEED_WATCHDOG();
    // ... work ...
}

void task2(void) {
    FEED_WATCHDOG();
    // ... work ...
}
```

---

## Logic Analyzers

### Saleae Logic Analyzers

Popular USB logic analyzers for debugging digital signals.

**Models:**

| Model | Channels | Sample Rate | Price |
|-------|----------|-------------|-------|
| **Logic 8** | 8 digital | 100 MS/s | $399 |
| **Logic Pro 8** | 8 digital, 2 analog | 500 MS/s | $699 |
| **Logic Pro 16** | 16 digital, 2 analog | 500 MS/s | $999 |

**Key Features:**
- Protocol analyzers (I2C, SPI, UART, CAN, I2S, etc.)
- Custom protocol decoders
- Trigger conditions
- Export to CSV, VCD
- Cross-platform software

**Connection Example:**
```
Logic Analyzer          Target Device
┌────────────┐         ┌──────────────┐
│ CH0 (Black)├────────>│ I2C SCL      │
│ CH1 (Brown)├────────>│ I2C SDA      │
│ CH2 (Red)  ├────────>│ UART TX      │
│ CH3 (Orange├────────>│ UART RX      │
│ GND        ├────────>│ GND          │
└────────────┘         └──────────────┘

Voltage Levels: 3.3V or 5V (check device specs)
```

**Common Uses:**
```
I2C Protocol Analysis:
- Verify START/STOP conditions
- Check ACK/NACK responses
- Decode addresses and data
- Measure timing (setup, hold times)

SPI Protocol Analysis:
- Clock polarity and phase
- Data integrity
- Chip select timing
- Multi-device communication

UART Analysis:
- Baud rate verification
- Frame format (bits, parity, stop)
- Data corruption detection
- Timing issues
```

### Protocol Decoding

**I2C Decoder Settings:**
```
Sample Rate: 10 MS/s minimum
SCL: Channel 0
SDA: Channel 1
Bit Rate: Auto or manual (100k, 400k, 1M)
```

**SPI Decoder Settings:**
```
MOSI: Channel 0
MISO: Channel 1
SCK:  Channel 2
CS:   Channel 3
CPOL: 0 or 1
CPHA: 0 or 1
Bits per transfer: 8/16/32
```

**UART Decoder Settings:**
```
Signal: Channel 0
Baud Rate: 9600, 115200, etc.
Bits: 8
Parity: None, Even, Odd
Stop Bits: 1 or 2
Inverted: No
```

### Triggering

```
Simple Triggers:
- Rising edge on channel
- Falling edge on channel
- Both edges
- High/Low level

Advanced Triggers:
- Pulse width
- I2C address match
- SPI data pattern
- UART character sequence
```

### Open-Source Alternatives

**sigrok/PulseView:**
```bash
# Install
sudo apt-get install sigrok pulseview

# Supported devices
- Logic Pirate
- Bus Pirate
- Cypress FX2-based devices
- Many USB logic analyzers

# Protocol decoders (100+)
- I2C, SPI, UART, CAN, I2S
- 1-Wire, JTAG, SWD
- USB, Ethernet, HDMI
- Custom protocols (Python)
```

---

## Oscilloscopes

### Digital Oscilloscopes

Essential for analog signal analysis and timing measurements.

**Key Specifications:**
- **Bandwidth**: 50 MHz - 1 GHz (100 MHz typical for embedded)
- **Sample Rate**: At least 4x bandwidth
- **Channels**: 2 or 4
- **Memory Depth**: Deeper is better for capturing long events

**Common Embedded Use Cases:**

```
Power Supply Analysis:
- Voltage ripple
- Noise
- Startup transients
- Load regulation

Signal Integrity:
- Rise/fall times
- Overshoot/undershoot
- Ringing
- Cross-talk

Analog Signals:
- ADC input verification
- PWM duty cycle
- Sensor outputs
- Communication signal quality
```

### Oscilloscope Measurements

**Voltage Measurements:**
```
Vpp  - Peak-to-peak voltage
Vmax - Maximum voltage
Vmin - Minimum voltage
Vavg - Average voltage
Vrms - RMS voltage
```

**Timing Measurements:**
```
Period    - Signal period
Frequency - Signal frequency
Duty Cycle - PWM duty cycle
Rise Time  - 10% to 90% transition
Fall Time  - 90% to 10% transition
```

**Triggering:**
```
Edge Trigger:
- Rising/falling edge
- Threshold level

Pulse Width Trigger:
- Glitch capture
- Min/max pulse width

Protocol Triggers:
- I2C, SPI, UART, CAN
- Specific patterns
```

### Mixed-Signal Oscilloscopes (MSO)

Combines oscilloscope with logic analyzer.

**Advantages:**
- Correlate analog and digital signals
- See timing relationships
- Debug ADC/DAC systems
- Protocol analysis with signal quality

**Example Setup:**
```
MSO Channels:
- Analog CH1: Power supply 3.3V
- Analog CH2: PWM output signal
- Digital D0-D7: SPI bus
- Trigger: SPI CS falling edge

Use Case:
- Verify PWM signal quality
- Check power supply during SPI transfer
- Correlate digital activity with analog effects
```

---

## Common Debugging Patterns

### Hard Fault Handler

```c
// Enhanced hard fault handler with debugging info
void HardFault_Handler(void) {
    __asm volatile (
        "tst lr, #4                                     \n"
        "ite eq                                         \n"
        "mrseq r0, msp                                  \n"
        "mrsne r0, psp                                  \n"
        "b hard_fault_handler_c                         \n"
    );
}

void hard_fault_handler_c(uint32_t* hardfault_args) {
    volatile uint32_t stacked_r0;
    volatile uint32_t stacked_r1;
    volatile uint32_t stacked_r2;
    volatile uint32_t stacked_r3;
    volatile uint32_t stacked_r12;
    volatile uint32_t stacked_lr;
    volatile uint32_t stacked_pc;
    volatile uint32_t stacked_psr;
    volatile uint32_t cfsr;
    volatile uint32_t hfsr;
    volatile uint32_t dfsr;
    volatile uint32_t afsr;
    volatile uint32_t mmar;
    volatile uint32_t bfar;

    stacked_r0  = ((uint32_t)hardfault_args[0]);
    stacked_r1  = ((uint32_t)hardfault_args[1]);
    stacked_r2  = ((uint32_t)hardfault_args[2]);
    stacked_r3  = ((uint32_t)hardfault_args[3]);
    stacked_r12 = ((uint32_t)hardfault_args[4]);
    stacked_lr  = ((uint32_t)hardfault_args[5]);
    stacked_pc  = ((uint32_t)hardfault_args[6]);
    stacked_psr = ((uint32_t)hardfault_args[7]);

    // Fault status registers
    cfsr = (*((volatile uint32_t*)(0xE000ED28)));
    hfsr = (*((volatile uint32_t*)(0xE000ED2C)));
    dfsr = (*((volatile uint32_t*)(0xE000ED30)));
    afsr = (*((volatile uint32_t*)(0xE000ED3C)));
    mmar = (*((volatile uint32_t*)(0xE000ED34)));
    bfar = (*((volatile uint32_t*)(0xE000ED38)));

    // Print fault information (or save to flash/RAM)
    debug_printf("\n[Hard Fault]\n");
    debug_printf("R0  = 0x%08X\n", stacked_r0);
    debug_printf("R1  = 0x%08X\n", stacked_r1);
    debug_printf("R2  = 0x%08X\n", stacked_r2);
    debug_printf("R3  = 0x%08X\n", stacked_r3);
    debug_printf("R12 = 0x%08X\n", stacked_r12);
    debug_printf("LR  = 0x%08X\n", stacked_lr);
    debug_printf("PC  = 0x%08X\n", stacked_pc);
    debug_printf("PSR = 0x%08X\n", stacked_psr);
    debug_printf("CFSR= 0x%08X\n", cfsr);
    debug_printf("HFSR= 0x%08X\n", hfsr);
    debug_printf("DFSR= 0x%08X\n", dfsr);
    debug_printf("AFSR= 0x%08X\n", afsr);

    if (cfsr & 0x0080) debug_printf("MMAR= 0x%08X\n", mmar);
    if (cfsr & 0x8000) debug_printf("BFAR= 0x%08X\n", bfar);

    // Infinite loop or reset
    while(1);
}
```

### Memory Dump

```c
// Dump memory region
void memory_dump(uint32_t addr, uint32_t len) {
    uint8_t* p = (uint8_t*)addr;

    for (uint32_t i = 0; i < len; i += 16) {
        debug_printf("%08X: ", addr + i);

        // Hex values
        for (uint32_t j = 0; j < 16 && (i + j) < len; j++) {
            debug_printf("%02X ", p[i + j]);
        }

        // ASCII representation
        debug_printf(" |");
        for (uint32_t j = 0; j < 16 && (i + j) < len; j++) {
            char c = p[i + j];
            debug_printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        debug_printf("|\n");
    }
}

// Usage
memory_dump(0x20000000, 256);  // Dump 256 bytes from RAM start
```

### State Machine Debugging

```c
typedef enum {
    STATE_IDLE,
    STATE_INIT,
    STATE_RUNNING,
    STATE_ERROR
} system_state_t;

static system_state_t current_state = STATE_IDLE;

void set_state(system_state_t new_state) {
    const char* state_names[] = {"IDLE", "INIT", "RUNNING", "ERROR"};

    LOG_INFO("State: %s -> %s",
             state_names[current_state],
             state_names[new_state]);

    current_state = new_state;
}
```

### Peripheral Register Dump

```c
// Dump all GPIO registers
void dump_gpio_registers(GPIO_TypeDef* GPIOx) {
    debug_printf("GPIO Base: 0x%08X\n", (uint32_t)GPIOx);
    debug_printf("MODER:   0x%08X\n", GPIOx->MODER);
    debug_printf("OTYPER:  0x%08X\n", GPIOx->OTYPER);
    debug_printf("OSPEEDR: 0x%08X\n", GPIOx->OSPEEDR);
    debug_printf("PUPDR:   0x%08X\n", GPIOx->PUPDR);
    debug_printf("IDR:     0x%08X\n", GPIOx->IDR);
    debug_printf("ODR:     0x%08X\n", GPIOx->ODR);
}

// Usage
dump_gpio_registers(GPIOA);
```

### Circular Buffer Trace

```c
#define TRACE_BUFFER_SIZE 256

typedef struct {
    uint32_t timestamp;
    uint32_t pc;
    uint32_t data;
} trace_entry_t;

static trace_entry_t trace_buffer[TRACE_BUFFER_SIZE];
static volatile uint32_t trace_index = 0;

void trace_log(uint32_t data) {
    trace_entry_t* entry = &trace_buffer[trace_index % TRACE_BUFFER_SIZE];
    entry->timestamp = HAL_GetTick();
    entry->pc = (uint32_t)__builtin_return_address(0);
    entry->data = data;
    trace_index++;
}

void trace_dump(void) {
    uint32_t start = trace_index > TRACE_BUFFER_SIZE ?
                     trace_index - TRACE_BUFFER_SIZE : 0;

    for (uint32_t i = start; i < trace_index; i++) {
        trace_entry_t* entry = &trace_buffer[i % TRACE_BUFFER_SIZE];
        debug_printf("[%lu] PC:0x%08X Data:0x%08X\n",
                    entry->timestamp, entry->pc, entry->data);
    }
}
```

---

## Performance Profiling

### Cycle Counter

```c
// Enable DWT cycle counter (Cortex-M3/M4/M7)
void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // Enable trace
    DWT->CYCCNT = 0;                                  // Reset counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable counter
}

// Get current cycle count
uint32_t dwt_get_cycles(void) {
    return DWT->CYCCNT;
}

// Measure function execution time
uint32_t start = dwt_get_cycles();
my_function();
uint32_t cycles = dwt_get_cycles() - start;
debug_printf("Function took %lu cycles\n", cycles);
```

### Execution Time Measurement

```c
// Measure execution time in microseconds
uint32_t measure_us(void (*func)(void)) {
    uint32_t start = DWT->CYCCNT;
    func();
    uint32_t end = DWT->CYCCNT;

    // Assuming 168 MHz clock
    return (end - start) / 168;
}

// Usage
uint32_t time_us = measure_us(some_function);
debug_printf("Execution time: %lu us\n", time_us);
```

### GPIO Toggle Profiling

```c
// Use GPIO to visualize timing on oscilloscope
#define PROFILE_PIN_HIGH()  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_SET)
#define PROFILE_PIN_LOW()   HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, GPIO_PIN_RESET)
#define PROFILE_PIN_TOGGLE() HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_0)

// Measure function
void critical_function(void) {
    PROFILE_PIN_HIGH();
    // ... work ...
    PROFILE_PIN_LOW();
}

// Measure ISR
void TIM2_IRQHandler(void) {
    PROFILE_PIN_HIGH();
    // ... ISR work ...
    PROFILE_PIN_LOW();
}
```

---

## Troubleshooting Common Issues

### Device Not Detected

```bash
# Check connections
- Verify VCC, GND, SWDIO, SWCLK connections
- Check voltage levels (3.3V or 5V)
- Verify NRST if used
- Check for shorts

# OpenOCD diagnostics
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -d3

# ST-LINK utility (Windows)
- Use ST-LINK Utility to check connection
- Try firmware upgrade on ST-LINK

# Linux permissions
sudo usermod -aG dialout $USER
sudo cp 99-stlink.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Flash Programming Failures

```bash
# Erase flash first
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
  -c "init" -c "reset halt" -c "flash erase_sector 0 0 last" -c "shutdown"

# Mass erase
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
  -c "init" -c "reset halt" -c "stm32f4x mass_erase 0" -c "shutdown"

# Disable write protection
# (via option bytes - device specific)

# Check flash protection
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
  -c "init" -c "flash info 0" -c "shutdown"
```

### Breakpoint Issues

```gdb
# Hardware breakpoints (limited, usually 4-6)
info break
# If too many, use software breakpoints (requires flash)

# Cannot set software breakpoint in flash
# Solution: Enable flash patching
monitor flash breakpoints enable

# Breakpoint not hit
# Check optimization level
# Try volatile variables
# Verify code is actually executed
```

### SWD/JTAG Connection Lost

```bash
# Try different speeds
adapter speed 500  # Slower

# Connect under reset
reset_config srst_only srst_nogate
adapter assert srst

# Hot plug (connect while running)
# May not work with all targets

# Factory reset (if available)
# Use manufacturer tools
```

### Debug Output Not Working

```c
// Check UART configuration
- Baud rate match
- Pin configuration (TX/RX not swapped)
- Voltage levels
- GND connection

// Check semihosting
- Enabled in GDB
- Significant performance impact
- May not work in some configurations

// Check SWO
- Configured in debugger
- Correct clock speed
- Pin configured as SWO
```

---

## Best Practices

### 1. Use Version Control for Configurations

```bash
# Git repository structure
project/
├── .vscode/
│   ├── launch.json       # Debug configurations
│   └── tasks.json
├── .gdbinit             # GDB init
├── openocd.cfg          # OpenOCD config
└── debug_scripts/
    ├── flash.sh
    └── erase.sh
```

### 2. Defensive Programming

```c
// Always validate inputs
void process_data(uint8_t* data, size_t len) {
    ASSERT(data != NULL);
    ASSERT(len > 0 && len <= MAX_SIZE);

    // ... process ...
}

// Check return values
if (HAL_I2C_Master_Transmit(&hi2c1, addr, data, len, 100) != HAL_OK) {
    LOG_ERROR("I2C transmit failed");
    return ERROR;
}

// Initialize variables
int value = 0;  // Not garbage
uint8_t* ptr = NULL;  // Not random address
```

### 3. Reproducible Builds

```makefile
# Makefile with debug symbols
DEBUG = 1

ifeq ($(DEBUG), 1)
CFLAGS += -O0 -g3 -gdwarf-2
else
CFLAGS += -O2 -g0
endif

# Always include debug info in separate file
OBJCOPY_FLAGS = --only-keep-debug
objcopy $(OBJCOPY_FLAGS) firmware.elf firmware.debug
```

### 4. Test on Real Hardware Early

```
Development Flow:
1. Prototype on development board
2. Test on target hardware
3. Test in final enclosure
4. Test in target environment
5. Long-term reliability testing

Don't wait until production!
```

### 5. Document Hardware Debug Setup

```markdown
# Debug Setup for Project XYZ

## Hardware Connections
- ST-LINK V2 to SWD header (J1)
- UART console on PA9/PA10 (115200 8N1)
- Logic analyzer on I2C bus (PB6/PB7)

## OpenOCD Command
```bash
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg
```

## GDB Command
```bash
arm-none-eabi-gdb build/firmware.elf
(gdb) source .gdbinit
```

## Test Points
- TP1: 3.3V power
- TP2: Reset signal
- TP3: Status LED (PA5)
```

### 6. Use Static Analysis Tools

```bash
# Cppcheck
cppcheck --enable=all --inconclusive src/

# Compiler warnings
CFLAGS += -Wall -Wextra -Werror -Wpedantic

# PC-Lint (commercial)
# MISRA-C checker
```

### 7. Hardware Debug Headers

```
Design PCB with debug headers:

┌──────────────────────────────┐
│  [J1] SWD Header              │
│   1. VCC                      │
│   2. SWDIO                    │
│   3. GND                      │
│   4. SWCLK                    │
│   5. SWO                      │
│   6. RESET                    │
│                               │
│  [J2] UART Console            │
│   1. GND                      │
│   2. TX                       │
│   3. RX                       │
│                               │
│  [TP1-10] Test Points         │
│   - Power rails               │
│   - Critical signals          │
│   - Protocol buses            │
└──────────────────────────────┘
```

### 8. Keep Debug Code in Production

```c
#ifdef DEBUG
    #define DEBUG_PRINT(...) debug_printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...) ((void)0)
#endif

// Or use log levels
#if LOG_LEVEL >= LOG_LEVEL_DEBUG
    LOG_DEBUG("Detailed information");
#endif
```

### 9. Post-Mortem Debugging

```c
// Save crash info to flash or backup RAM
typedef struct {
    uint32_t magic;
    uint32_t pc;
    uint32_t lr;
    uint32_t sp;
    uint32_t fault_regs[8];
} crash_info_t;

void save_crash_info(crash_info_t* info) {
    // Write to backup RAM or flash
    // Check on next boot
}

// At startup
void check_crash_info(void) {
    if (crash_info.magic == CRASH_MAGIC) {
        debug_printf("Previous crash at PC: 0x%08X\n", crash_info.pc);
        // Analyze or send to server
        memset(&crash_info, 0, sizeof(crash_info));
    }
}
```

### 10. Use Continuous Integration

```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install ARM toolchain
        run: |
          sudo apt-get update
          sudo apt-get install gcc-arm-none-eabi
      - name: Build
        run: make
      - name: Run unit tests
        run: make test
      - name: Static analysis
        run: cppcheck --error-exitcode=1 src/
```

---

## Resources

### Documentation
- **ARM Cortex-M Debug**: ARM Debug Interface Architecture Specification
- **OpenOCD**: http://openocd.org/documentation/
- **GDB Manual**: https://sourceware.org/gdb/documentation/
- **SEGGER**: https://www.segger.com/products/debug-probes/j-link/

### Tools
- **OpenOCD**: Open-source debug adapter
- **PyOCD**: Python-based debugger
- **Black Magic Probe**: Standalone GDB server
- **Saleae Logic**: Logic analyzer software
- **sigrok/PulseView**: Open-source logic analyzer

### Books
- "The Definitive Guide to ARM Cortex-M3/M4" by Joseph Yiu
- "Embedded Systems Architecture" by Daniele Lacamera
- "Debugging Embedded Systems" by Chris Svec

---

Effective embedded debugging requires a combination of hardware tools, software techniques, and systematic approaches. Master these tools and patterns to efficiently diagnose and resolve issues in your embedded systems.
