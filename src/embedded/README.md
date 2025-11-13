# Embedded Systems

Comprehensive guide to embedded systems development, microcontrollers, and hardware interfacing.

## Table of Contents

1. [Introduction](#introduction)
2. [Development Platforms](#development-platforms)
3. [Core Concepts](#core-concepts)
4. [Communication Protocols](#communication-protocols)
5. [Peripheral Interfaces](#peripheral-interfaces)
6. [Getting Started](#getting-started)

## Introduction

Embedded systems are specialized computing systems designed to perform dedicated functions within larger mechanical or electrical systems. They combine hardware and software to control devices and interact with the physical world.

### Key Characteristics

- **Real-time Operation**: Deterministic response to events
- **Resource Constraints**: Limited memory, processing power, and energy
- **Reliability**: Must operate continuously for extended periods
- **Hardware Integration**: Direct interaction with sensors and actuators
- **Application-Specific**: Optimized for particular tasks

### Architecture Overview

```
┌─────────────────────────────────────────┐
│         Embedded System                  │
├─────────────────────────────────────────┤
│  Application Layer                       │
│  ├─ User Code                           │
│  └─ Libraries & Frameworks              │
├─────────────────────────────────────────┤
│  HAL/Drivers                            │
│  ├─ Peripheral Drivers                  │
│  └─ Hardware Abstraction Layer          │
├─────────────────────────────────────────┤
│  Microcontroller/Processor              │
│  ├─ CPU Core (ARM, AVR, RISC-V)        │
│  ├─ Memory (Flash, RAM, EEPROM)        │
│  ├─ Peripherals (GPIO, UART, SPI...)   │
│  └─ Clock & Power Management            │
├─────────────────────────────────────────┤
│  Hardware                               │
│  ├─ Sensors                             │
│  ├─ Actuators                           │
│  └─ External Interfaces                 │
└─────────────────────────────────────────┘
```

## Development Platforms

### Microcontroller Platforms

| Platform | Processor | Clock | Memory | Use Cases |
|----------|-----------|-------|--------|-----------|
| **[Arduino](arduino.md)** | AVR/ARM | 16-84 MHz | 2KB-256KB RAM | Prototyping, education, hobbyist projects |
| **[ESP32](esp32.md)** | Xtensa/RISC-V | 160-240 MHz | 520KB RAM | IoT, WiFi/BLE projects |
| **[STM32](stm32.md)** | ARM Cortex-M | 48-550 MHz | 32KB-2MB RAM | Professional, industrial applications |
| **[AVR](avr.md)** | AVR | 1-20 MHz | 512B-16KB RAM | Low-power, bare-metal programming |
| **[Raspberry Pi](raspberry_pi.md)** | ARM Cortex-A | 700MHz-2.4GHz | 512MB-8GB RAM | Linux-based, complex applications |

### Comparison Matrix

```
Complexity/Capability
    ^
    |
RPi |  ┌──────────┐
    |  │          │
STM32| │          │    ┌──────┐
    |  │          │    │      │
ESP32|  │          │    │      │  ┌─────┐
    |  │          │    │      │  │     │
ARD  |  │          │    │      │  │     │  ┌────┐
    |  │          │    │      │  │     │  │    │
AVR  |  │          │    │      │  │     │  │    │
    |  └──────────┴────┴──────┴──┴─────┴──┴────┘
    +──────────────────────────────────────────> Cost
       Low                                  High
```

## Core Concepts

### Memory Architecture

#### Flash Memory (Program Storage)
- Stores program code and constant data
- Non-volatile (persists without power)
- Typically 8KB to several MB
- Limited write cycles (10K-100K)

#### SRAM (Runtime Memory)
- Stores variables and stack during execution
- Volatile (lost when power removed)
- Fast access, limited size
- Critical resource in embedded systems

#### EEPROM (Persistent Data)
- Stores configuration and calibration data
- Non-volatile, byte-addressable
- Limited write cycles but higher than Flash
- Slower than SRAM

```
Memory Map Example (ATmega328P):
┌────────────────┐ 0x0000
│  Flash (32KB)  │
│  Program Code  │
├────────────────┤ 0x7FFF
│  SRAM (2KB)    │
│  Variables     │
│  Stack         │
├────────────────┤ 0x08FF
│  EEPROM (1KB)  │
│  Persistent    │
└────────────────┘ 0x03FF
```

### Power Management

#### Operating Modes
1. **Active Mode**: Full operation, highest power consumption
2. **Idle Mode**: CPU stopped, peripherals running
3. **Sleep Mode**: Most peripherals disabled
4. **Deep Sleep**: Minimal power, wake on interrupt only

#### Power Saving Techniques
```c
// Example: AVR Sleep Mode
#include <avr/sleep.h>
#include <avr/power.h>

void enterSleepMode() {
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    sleep_enable();
    
    // Disable unnecessary peripherals
    power_adc_disable();
    power_spi_disable();
    power_timer0_disable();
    
    sleep_mode();  // Enter sleep
    
    // Wake up here after interrupt
    sleep_disable();
    
    // Re-enable peripherals
    power_all_enable();
}
```

### Interrupt-Driven Programming

Interrupts allow the processor to respond to events immediately without polling.

```c
// Example: External Interrupt
volatile bool buttonPressed = false;

// Interrupt Service Routine (ISR)
void EXTI0_IRQHandler(void) {
    if (EXTI->PR & EXTI_PR_PR0) {
        buttonPressed = true;
        EXTI->PR |= EXTI_PR_PR0;  // Clear interrupt flag
    }
}

int main(void) {
    // Setup interrupt
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN;
    GPIOA->CRL &= ~GPIO_CRL_CNF0;
    GPIOA->CRL |= GPIO_CRL_CNF0_1;  // Input with pull-up
    
    AFIO->EXTICR[0] = AFIO_EXTICR1_EXTI0_PA;
    EXTI->IMR |= EXTI_IMR_MR0;
    EXTI->FTSR |= EXTI_FTSR_TR0;  // Falling edge
    
    NVIC_EnableIRQ(EXTI0_IRQn);
    
    while (1) {
        if (buttonPressed) {
            // Handle button press
            buttonPressed = false;
        }
        // Main loop continues
    }
}
```

## Communication Protocols

### Serial Protocols Overview

| Protocol | Type | Speed | Wires | Use Case |
|----------|------|-------|-------|----------|
| **[UART](uart.md)** | Asynchronous | Up to 1 Mbps | 2 (TX/RX) | Debug, GPS, Bluetooth modules |
| **[SPI](spi.md)** | Synchronous | Up to 50 Mbps | 4+ (MOSI/MISO/SCK/CS) | SD cards, displays, high-speed sensors |
| **[I2C](i2c.md)** | Synchronous | 100-400 kHz | 2 (SDA/SCL) | Sensors, RTCs, EEPROMs |
| **[CAN](can.md)** | Differential | Up to 1 Mbps | 2 (CAN_H/CAN_L) | Automotive, industrial |
| **[USB](usb.md)** | Differential | 1.5-480 Mbps | 2 (D+/D-) | PC interface, peripherals |

### Protocol Comparison

```
Speed (Mbps)
    ^
    |
100 |                    ┌─── USB 2.0
    |                    │
 50 |            ┌─── SPI│
    |            │       │
 10 |            │       │
    |            │       │
  1 |  ┌─ UART  │       │
    |  │    │   │       │
0.1 |  │ I2C│   │       │
    |  │    │   │       │
    └──┴────┴───┴───────┴────────> Complexity
       Low              High
```

## Peripheral Interfaces

### Digital I/O ([GPIO](gpio.md))
- General Purpose Input/Output pins
- Digital HIGH/LOW states
- Input modes: floating, pull-up, pull-down
- Output modes: push-pull, open-drain

### Analog Interfaces
- **[ADC](adc.md)**: Convert analog voltages to digital values
- **[DAC](dac.md)**: Convert digital values to analog voltages
- **[PWM](pwm.md)**: Pulse Width Modulation for analog-like output

### Timing and Control
- **[Timers](timers.md)**: Hardware timers for precise timing
- **[Interrupts](interrupts.md)**: Event-driven programming
- **[Watchdog](watchdog.md)**: System reliability and reset

### Specialized Interfaces
- **[RTC](rtc.md)**: Real-Time Clock for timekeeping
- **[SDIO](sdio.md)**: SD card interface
- **[Ethernet](ethernet.md)**: Network connectivity

## Getting Started

### Development Environment Setup

#### 1. Choose Your Platform
Start with Arduino for beginners, or jump to STM32/ESP32 for more advanced projects.

#### 2. Install Tools

**For Arduino:**
```bash
# Download Arduino IDE from arduino.cc
# Or use Arduino CLI
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
```

**For STM32:**
```bash
# Install STM32CubeIDE
# Download from st.com
# Or use PlatformIO
pip install platformio
```

**For ESP32:**
```bash
# Add ESP32 to Arduino IDE
# Or use ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh
```

#### 3. Hardware Setup

**Minimum Requirements:**
- Development board (Arduino Uno, ESP32, STM32 Nucleo, etc.)
- USB cable
- Computer with IDE installed
- Optional: Breadboard, jumper wires, components

**Development Kit:**
```
Essential Components:
├─ Microcontroller board
├─ USB cable
├─ Breadboard
├─ Jumper wires (male-male, male-female)
├─ LEDs and resistors (220Ω)
├─ Push buttons
├─ Potentiometer (10kΩ)
└─ Multimeter

Sensors (Optional):
├─ Temperature (DHT11/22, DS18B20)
├─ Distance (HC-SR04 ultrasonic)
├─ Light (LDR, BH1750)
└─ Motion (PIR, MPU6050)
```

### First Program: Blink LED

#### Arduino Version
```cpp
// Blink LED on pin 13
void setup() {
    pinMode(13, OUTPUT);
}

void loop() {
    digitalWrite(13, HIGH);
    delay(1000);
    digitalWrite(13, LOW);
    delay(1000);
}
```

#### STM32 HAL Version
```c
#include "stm32f4xx_hal.h"

int main(void) {
    HAL_Init();
    SystemClock_Config();
    
    __HAL_RCC_GPIOA_CLK_ENABLE();
    
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    
    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
        HAL_Delay(1000);
    }
}
```

#### Bare Metal AVR Version
```c
#include <avr/io.h>
#include <util/delay.h>

int main(void) {
    DDRB |= (1 << DDB5);  // Set PB5 as output
    
    while (1) {
        PORTB |= (1 << PORTB5);   // LED on
        _delay_ms(1000);
        PORTB &= ~(1 << PORTB5);  // LED off
        _delay_ms(1000);
    }
    
    return 0;
}
```

### Learning Path

```
┌─────────────────────────────────────────────┐
│ Level 1: Fundamentals                       │
├─────────────────────────────────────────────┤
│ • Digital I/O (LED, button)                 │
│ • Analog input (ADC, potentiometer)         │
│ • PWM (LED brightness, motor speed)         │
│ • Serial communication (UART debug)         │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Level 2: Intermediate                       │
├─────────────────────────────────────────────┤
│ • Timers and interrupts                     │
│ • I2C sensors (temperature, accelerometer)  │
│ • SPI devices (SD card, display)            │
│ • State machines                            │
└─────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│ Level 3: Advanced                           │
├─────────────────────────────────────────────┤
│ • DMA transfers                             │
│ • RTOS (FreeRTOS)                          │
│ • Low-power optimization                    │
│ • Bootloaders and OTA updates              │
└─────────────────────────────────────────────┘
```

### Common Project Ideas

1. **Beginner Projects**
   - LED blink and patterns
   - Button-controlled LED
   - Temperature monitor
   - Light-sensitive nightlight

2. **Intermediate Projects**
   - Digital thermometer with display
   - Motor speed controller
   - Distance measurement system
   - Data logger with SD card

3. **Advanced Projects**
   - Weather station with WiFi
   - Robot controller
   - Home automation system
   - Wireless sensor network

## Best Practices

### Code Organization
```c
// Good structure
├─ src/
│  ├─ main.c
│  ├─ drivers/
│  │  ├─ sensor.c
│  │  └─ display.c
│  └─ app/
│     ├─ control.c
│     └─ config.c
├─ inc/
│  ├─ sensor.h
│  ├─ display.h
│  └─ config.h
└─ Makefile
```

### Design Principles

1. **Keep ISRs Short**: Minimal processing in interrupt handlers
2. **Use Volatile**: For variables modified by ISRs
3. **Debounce Inputs**: Software or hardware debouncing for buttons
4. **Watchdog Timer**: Implement system recovery
5. **Power Efficiency**: Use sleep modes when idle
6. **Error Handling**: Check return values and handle failures
7. **Documentation**: Comment complex logic and register operations

### Debugging Techniques

```c
// UART debug output
void debug_print(const char* msg) {
    uart_send_string(msg);
}

// LED status indicators
#define LED_ERROR   GPIO_PIN_0
#define LED_OK      GPIO_PIN_1
#define LED_BUSY    GPIO_PIN_2

// Assert macro
#define ASSERT(expr) \
    if (!(expr)) { \
        debug_print("Assert failed: " #expr); \
        while(1);  // Halt \
    }
```

## Resources

### Documentation
- Platform-specific datasheets and reference manuals
- Peripheral application notes
- HAL/LL library documentation

### Tools
- **Oscilloscope**: Analyze signals and timing
- **Logic Analyzer**: Debug digital protocols
- **Multimeter**: Measure voltages and continuity
- **Debugger**: JTAG/SWD for step-through debugging

### Communities
- Arduino Forum
- STM32 Community
- ESP32 Forum
- Reddit: r/embedded, r/arduino
- Stack Overflow: Embedded tag

## See Also

- [GPIO Programming](gpio.md)
- [UART Communication](uart.md)
- [Interrupt Handling](interrupts.md)
- [Power Management](power_management.md)
- [Debugging Techniques](debugging.md)
