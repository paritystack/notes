# STM32 Microcontrollers

Comprehensive guide to STM32 development using HAL, CubeMX, and bare-metal programming.

## Table of Contents

1. [Introduction](#introduction)
2. [STM32 Families](#stm32-families)
3. [Development Setup](#development-setup)
4. [STM32CubeMX](#stm32cubemx)
5. [HAL Programming](#hal-programming)
6. [Bare Metal Programming](#bare-metal-programming)
7. [Common Peripherals](#common-peripherals)
8. [Advanced Topics](#advanced-topics)

## Introduction

STM32 is a family of 32-bit microcontrollers from STMicroelectronics based on ARM Cortex-M cores. They offer excellent performance, rich peripherals, and are widely used in professional and industrial applications.

### Key Features

- **ARM Cortex-M Cores**: M0, M0+, M3, M4, M7, M33
- **Clock Speed**: 48 MHz to 550 MHz
- **Memory**: 16 KB to 2 MB Flash, 4 KB to 1 MB RAM
- **Peripherals**: GPIO, UART, SPI, I2C, ADC, DAC, Timers, USB, CAN, Ethernet
- **Development Tools**: Free official IDE and HAL libraries
- **Price**: $1 to $20 depending on series

### Advantages

- Professional-grade reliability
- Extensive peripheral set
- Low power consumption
- Strong ecosystem and support
- Pin-compatible families
- Real-time performance

## STM32 Families

### Overview

| Family | Core | Speed | Flash | Use Case | Examples |
|--------|------|-------|-------|----------|----------|
| **F0** | M0 | 48 MHz | 16-256 KB | Entry-level, cost-sensitive | STM32F030 |
| **F1** | M3 | 72 MHz | 16-512 KB | General purpose, classic | STM32F103 (Blue Pill) |
| **F4** | M4 | 180 MHz | 256 KB-2 MB | High performance, DSP, FPU | STM32F407, F429 |
| **F7** | M7 | 216 MHz | 512 KB-2 MB | Very high performance | STM32F746 |
| **H7** | M7 | 480 MHz | 1-2 MB | Extreme performance | STM32H743 |
| **L0/L4** | M0+/M4 | 32-80 MHz | 16-512 KB | Ultra-low power | STM32L476 |
| **G0/G4** | M0+/M4 | 64-170 MHz | 32-512 KB | Mainstream, motor control | STM32G474 |

### Popular Development Boards

#### STM32 Nucleo Boards
```
┌────────────────────────────────┐
│  STM32 Nucleo-64               │
│                                 │
│  ┌─────────────────┐           │
│  │   STM32 MCU     │           │
│  │   (QFP64)       │           │
│  └─────────────────┘           │
│                                 │
│  [CN7] ═══════════════ [CN10]  │  Arduino Headers
│  [CN8] ═══════════════ [CN9]   │
│                                 │
│  [CN1] ST-LINK V2-1            │
│  [USB]                         │
└────────────────────────────────┘

Features:
- Integrated ST-LINK debugger/programmer
- Arduino Uno R3 compatible headers
- Morpho extension headers (full pin access)
- Virtual COM port
- Price: ~$15
```

#### Blue Pill (STM32F103C8T6)
```
┌──────────────────────────┐
│     STM32F103C8T6        │
│     "Blue Pill"          │
│                          │
│  [USB] ═══════════ [SWD] │
│                          │
│  ╔════════════════════╗  │
│  ║  Header Pins       ║  │
│  ║  (40 pins total)   ║  │
│  ╚════════════════════╝  │
│                          │
│  [3.3V] [5V] [GND]       │
└──────────────────────────┘

Specs:
- 72 MHz ARM Cortex-M3
- 64 KB Flash, 20 KB RAM
- 37 GPIO pins
- 2x SPI, 2x I2C, 3x USART
- 12-bit ADC, 2x DAC
- Price: ~$2
```

## Development Setup

### STM32CubeIDE (Recommended)

```bash
# Download from ST website:
# https://www.st.com/en/development-tools/stm32cubeide.html

# Linux installation:
sudo chmod +x st-stm32cubeide_*.sh
sudo ./st-stm32cubeide_*.sh

# Install udev rules for ST-LINK
sudo cp ~/STMicroelectronics/STM32Cube/STM32CubeIDE/Drivers/rules/*.* /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

### Alternative: Command Line Setup

```bash
# Install ARM toolchain
sudo apt install gcc-arm-none-eabi gdb-multiarch

# Install OpenOCD (programming/debugging)
sudo apt install openocd

# Install st-link utilities
sudo apt install stlink-tools

# Verify installation
arm-none-eabi-gcc --version
openocd --version
st-info --version
```

### PlatformIO Setup

```bash
pip install platformio

# Create project
pio init --board nucleo_f401re

# platformio.ini
[env:nucleo_f401re]
platform = ststm32
board = nucleo_f401re
framework = arduino
# or framework = stm32cube
```

## STM32CubeMX

STM32CubeMX is a graphical configuration tool that generates initialization code for STM32 microcontrollers.

### Creating a Project

1. **Start New Project**
   - File > New Project
   - Select your MCU or board
   - Click "Start Project"

2. **Configure Clock**
   - Clock Configuration tab
   - Set HSE/HSI source
   - Configure PLL multipliers
   - Set system clock (HCLK)

3. **Configure Peripherals**
   - Pinout & Configuration tab
   - Click on pins to assign functions
   - Configure peripheral parameters

4. **Generate Code**
   - Project Manager tab
   - Set project name and location
   - Select toolchain (STM32CubeIDE, Makefile, etc.)
   - Click "Generate Code"

### Example: Blink LED Configuration

```
1. Pinout Configuration:
   - Find LED pin (e.g., PC13 on Blue Pill)
   - Set as GPIO_Output
   - Label it "LED"

2. GPIO Configuration:
   - Mode: Output Push Pull
   - Pull-up/Pull-down: No pull-up and no pull-down
   - Maximum output speed: Low
   - User Label: LED

3. Clock Configuration:
   - HSE: 8 MHz (external crystal)
   - PLL: ×9 (72 MHz system clock)

4. Generate Code
```

### Project Structure

```
project/
├── Core/
│   ├── Inc/
│   │   ├── main.h
│   │   ├── stm32f1xx_it.h
│   │   └── stm32f1xx_hal_conf.h
│   └── Src/
│       ├── main.c
│       ├── stm32f1xx_it.c
│       └── system_stm32f1xx.c
├── Drivers/
│   ├── STM32F1xx_HAL_Driver/
│   └── CMSIS/
└── Makefile
```

## HAL Programming

### Basic HAL Blink

```c
/* main.c - Generated by CubeMX */
#include "main.h"

GPIO_InitTypeDef GPIO_InitStruct = {0};

void SystemClock_Config(void);
static void MX_GPIO_Init(void);

int main(void) {
    /* Initialize HAL Library */
    HAL_Init();
    
    /* Configure system clock */
    SystemClock_Config();
    
    /* Initialize GPIO */
    MX_GPIO_Init();
    
    /* Infinite loop */
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        HAL_Delay(1000);
    }
}

static void MX_GPIO_Init(void) {
    /* Enable GPIO Clock */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    
    /* Configure GPIO pin */
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

void SystemClock_Config(void) {
    /* Generated by CubeMX - configures clocks */
}
```

### GPIO Functions

```c
/* Write pin */
HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);    // High
HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);  // Low

/* Toggle pin */
HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);

/* Read pin */
GPIO_PinState state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0);

/* External interrupt */
HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);  // Call in ISR
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin);  // Override this
```

### Button with Interrupt

```c
/* Configure button with external interrupt in CubeMX:
   PA0 -> GPIO_EXTI0
   Mode: External Interrupt Mode with Rising edge trigger detection
   Pull-up: Pull-up
   
   In NVIC tab: Enable EXTI line0 interrupt
*/

/* main.c */
volatile uint8_t button_pressed = 0;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    
    while (1) {
        if (button_pressed) {
            button_pressed = 0;
            HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        }
    }
}

/* Interrupt callback - implement this */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    if (GPIO_Pin == GPIO_PIN_0) {
        button_pressed = 1;
    }
}

/* stm32f1xx_it.c - Generated by CubeMX */
void EXTI0_IRQHandler(void) {
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
}
```

### UART Communication

```c
/* Configure UART in CubeMX:
   USART1: PA9 (TX), PA10 (RX)
   Baud Rate: 115200
   Word Length: 8 Bits
   Stop Bits: 1
   Parity: None
*/

UART_HandleTypeDef huart1;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USART1_UART_Init();
    
    uint8_t msg[] = "Hello, STM32!\r\n";
    HAL_UART_Transmit(&huart1, msg, sizeof(msg)-1, HAL_MAX_DELAY);
    
    uint8_t rx_buffer[10];
    while (1) {
        /* Blocking receive */
        HAL_UART_Receive(&huart1, rx_buffer, 1, HAL_MAX_DELAY);
        
        /* Echo back */
        HAL_UART_Transmit(&huart1, rx_buffer, 1, HAL_MAX_DELAY);
    }
}

/* printf redirect */
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
```

### ADC Reading

```c
/* Configure ADC in CubeMX:
   ADC1, Channel 0 (PA0)
   Resolution: 12 bits
   Continuous Conversion: Disabled
*/

ADC_HandleTypeDef hadc1;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_ADC1_Init();
    
    while (1) {
        HAL_ADC_Start(&hadc1);
        HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
        uint32_t adc_value = HAL_ADC_GetValue(&hadc1);
        
        /* Convert to voltage (3.3V reference, 12-bit) */
        float voltage = (adc_value * 3.3f) / 4096.0f;
        
        printf("ADC: %lu, Voltage: %.2f V\r\n", adc_value, voltage);
        
        HAL_Delay(1000);
    }
}
```

### PWM Output

```c
/* Configure Timer in CubeMX:
   TIM2, Channel 1 (PA0)
   Mode: PWM Generation CH1
   Prescaler: 72-1 (1 MHz timer clock)
   Counter Period: 1000-1 (1 kHz PWM)
*/

TIM_HandleTypeDef htim2;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_TIM2_Init();
    
    /* Start PWM */
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
    
    while (1) {
        /* Fade in */
        for (uint16_t duty = 0; duty <= 1000; duty += 10) {
            __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, duty);
            HAL_Delay(10);
        }
        
        /* Fade out */
        for (uint16_t duty = 1000; duty > 0; duty -= 10) {
            __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, duty);
            HAL_Delay(10);
        }
    }
}
```

### I2C Communication

```c
/* Configure I2C in CubeMX:
   I2C1: PB6 (SCL), PB7 (SDA)
   Speed: 100 kHz (Standard Mode)
*/

I2C_HandleTypeDef hi2c1;

#define DEVICE_ADDR 0x68 << 1  // 7-bit address shifted

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_I2C1_Init();
    
    uint8_t tx_data = 0x00;
    uint8_t rx_data[2];
    
    while (1) {
        /* Write register address */
        HAL_I2C_Master_Transmit(&hi2c1, DEVICE_ADDR, &tx_data, 1, HAL_MAX_DELAY);
        
        /* Read data */
        HAL_I2C_Master_Receive(&hi2c1, DEVICE_ADDR, rx_data, 2, HAL_MAX_DELAY);
        
        HAL_Delay(1000);
    }
}
```

### SPI Communication

```c
/* Configure SPI in CubeMX:
   SPI1: PA5 (SCK), PA6 (MISO), PA7 (MOSI)
   Mode: Master
   Baud Rate Prescaler: 32
   Data Size: 8 Bits
*/

SPI_HandleTypeDef hspi1;

#define CS_PIN GPIO_PIN_4
#define CS_PORT GPIOA

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_SPI1_Init();
    MX_GPIO_Init();  // CS pin
    
    uint8_t tx_data[] = {0x01, 0x02, 0x03};
    uint8_t rx_data[3];
    
    while (1) {
        /* Select device */
        HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_RESET);
        
        /* Transfer data */
        HAL_SPI_TransmitReceive(&hspi1, tx_data, rx_data, 3, HAL_MAX_DELAY);
        
        /* Deselect device */
        HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_SET);
        
        HAL_Delay(1000);
    }
}
```

## Bare Metal Programming

### Direct Register Access

```c
/* Blink LED without HAL - STM32F103 */
#include "stm32f1xx.h"

int main(void) {
    /* Enable GPIOC clock */
    RCC->APB2ENR |= RCC_APB2ENR_IOPCEN;
    
    /* Configure PC13 as output push-pull, max speed 2 MHz */
    GPIOC->CRH &= ~(GPIO_CRH_MODE13 | GPIO_CRH_CNF13);
    GPIOC->CRH |= GPIO_CRH_MODE13_1;  // Output mode, 2 MHz
    
    while (1) {
        /* Toggle LED */
        GPIOC->ODR ^= GPIO_ODR_ODR13;
        
        /* Delay */
        for (volatile uint32_t i = 0; i < 1000000; i++);
    }
}
```

### GPIO Register Operations

```c
/* Set pin high */
GPIOC->BSRR = GPIO_BSRR_BS13;    // Bit Set

/* Set pin low */
GPIOC->BSRR = GPIO_BSRR_BR13;    // Bit Reset

/* Toggle pin */
GPIOC->ODR ^= GPIO_ODR_ODR13;

/* Read pin */
uint32_t state = GPIOA->IDR & GPIO_IDR_IDR0;
```

### UART Bare Metal

```c
/* Initialize UART1 - 115200 baud, 72 MHz clock */
void UART1_Init(void) {
    /* Enable clocks */
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN | RCC_APB2ENR_IOPAEN;
    
    /* Configure PA9 (TX) as alternate function push-pull */
    GPIOA->CRH &= ~(GPIO_CRH_MODE9 | GPIO_CRH_CNF9);
    GPIOA->CRH |= GPIO_CRH_MODE9_1 | GPIO_CRH_CNF9_1;
    
    /* Configure PA10 (RX) as input floating */
    GPIOA->CRH &= ~(GPIO_CRH_MODE10 | GPIO_CRH_CNF10);
    GPIOA->CRH |= GPIO_CRH_CNF10_0;
    
    /* Configure UART */
    USART1->BRR = 0x271;  // 115200 baud at 72 MHz
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

void UART1_SendChar(char c) {
    while (!(USART1->SR & USART_SR_TXE));
    USART1->DR = c;
}

char UART1_ReceiveChar(void) {
    while (!(USART1->SR & USART_SR_RXNE));
    return USART1->DR;
}
```

### Timer Interrupt

```c
/* Configure TIM2 for 1 second interrupt */
void TIM2_Init(void) {
    /* Enable TIM2 clock */
    RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;
    
    /* Configure timer:
       72 MHz / 7200 = 10 kHz
       10 kHz / 10000 = 1 Hz (1 second)
    */
    TIM2->PSC = 7200 - 1;      // Prescaler
    TIM2->ARR = 10000 - 1;     // Auto-reload
    TIM2->DIER |= TIM_DIER_UIE;  // Update interrupt enable
    TIM2->CR1 |= TIM_CR1_CEN;    // Enable timer
    
    /* Enable interrupt in NVIC */
    NVIC_EnableIRQ(TIM2_IRQn);
}

/* Interrupt handler */
void TIM2_IRQHandler(void) {
    if (TIM2->SR & TIM_SR_UIF) {
        TIM2->SR &= ~TIM_SR_UIF;  // Clear interrupt flag
        
        /* Toggle LED */
        GPIOC->ODR ^= GPIO_ODR_ODR13;
    }
}
```

## Common Peripherals

### DMA Transfer

```c
/* Configure DMA for UART TX in CubeMX:
   DMA1, Channel 4
   Direction: Memory to Peripheral
   Mode: Normal
*/

uint8_t tx_buffer[] = "Hello from DMA!\r\n";

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    
    while (1) {
        HAL_UART_Transmit_DMA(&huart1, tx_buffer, sizeof(tx_buffer)-1);
        HAL_Delay(1000);
    }
}

/* DMA transfer complete callback */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    /* Transfer complete - can start next */
}
```

### RTC (Real-Time Clock)

```c
/* Configure RTC in CubeMX:
   RTC Activated
   Clock Source: LSE (32.768 kHz)
*/

RTC_TimeTypeDef sTime;
RTC_DateTypeDef sDate;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_RTC_Init();
    
    /* Set time */
    sTime.Hours = 12;
    sTime.Minutes = 0;
    sTime.Seconds = 0;
    HAL_RTC_SetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
    
    /* Set date */
    sDate.Year = 24;
    sDate.Month = 1;
    sDate.Date = 15;
    HAL_RTC_SetDate(&hrtc, &sDate, RTC_FORMAT_BIN);
    
    while (1) {
        HAL_RTC_GetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
        HAL_RTC_GetDate(&hrtc, &sDate, RTC_FORMAT_BIN);
        
        printf("%02d:%02d:%02d\r\n", 
               sTime.Hours, sTime.Minutes, sTime.Seconds);
        
        HAL_Delay(1000);
    }
}
```

### Watchdog Timer

```c
/* Configure IWDG in CubeMX:
   Independent Watchdog
   Prescaler: 32
   Counter Reload Value: 4095 (max ~4 seconds)
*/

IWDG_HandleTypeDef hiwdg;

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_IWDG_Init();
    
    while (1) {
        /* Main program tasks */
        
        /* Refresh watchdog */
        HAL_IWDG_Refresh(&hiwdg);
        
        HAL_Delay(100);
    }
}
```

## Advanced Topics

### FreeRTOS Integration

```c
/* Enable FreeRTOS in CubeMX */
#include "FreeRTOS.h"
#include "task.h"

void Task1(void *argument);
void Task2(void *argument);

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    
    /* Create tasks */
    xTaskCreate(Task1, "Task1", 128, NULL, 1, NULL);
    xTaskCreate(Task2, "Task2", 128, NULL, 1, NULL);
    
    /* Start scheduler */
    vTaskStartScheduler();
    
    /* Never reached */
    while (1);
}

void Task1(void *argument) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void Task2(void *argument) {
    while (1) {
        /* Other task */
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
```

### Low Power Modes

```c
/* Enter Stop mode */
HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);

/* Enter Standby mode */
HAL_PWR_EnterSTANDBYMode();

/* Enter Sleep mode */
HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
```

### Bootloader

```c
/* Jump to bootloader (system memory) */
void JumpToBootloader(void) {
    void (*SysMemBootJump)(void);
    
    /* Set bootloader address (STM32F1: 0x1FFFF000) */
    volatile uint32_t addr = 0x1FFFF000;
    
    /* Disable interrupts */
    __disable_irq();
    
    /* Remap system memory to 0x00000000 */
    __HAL_RCC_SYSCFG_CLK_ENABLE();
    __HAL_SYSCFG_REMAPMEMORY_SYSTEMFLASH();
    
    /* Set jump address */
    SysMemBootJump = (void (*)(void)) (*((uint32_t *)(addr + 4)));
    
    /* Set main stack pointer */
    __set_MSP(*(uint32_t *)addr);
    
    /* Jump */
    SysMemBootJump();
    
    while (1);
}
```

## Best Practices

1. **Use CubeMX**: Generate initialization code automatically
2. **HAL vs LL**: HAL for ease, LL for performance
3. **Interrupts**: Keep ISRs short, use callbacks
4. **DMA**: Use for high-speed data transfers
5. **Power**: Disable unused peripherals
6. **Debugging**: Use SWD with ST-LINK
7. **Version Control**: Track CubeMX .ioc files

## Troubleshooting

### Common Issues

**Debugger Not Connecting:**
```bash
# Check ST-LINK connection
st-info --probe

# Reset ST-LINK
st-flash reset

# Update ST-LINK firmware
# Use STM32 ST-LINK Utility
```

**Clock Configuration:**
- Verify HSE frequency matches hardware
- Check PLL multipliers for target frequency
- Enable required peripheral clocks

**GPIO Not Working:**
- Enable GPIO clock first
- Check pin alternate functions
- Verify pin configuration (mode, speed, pull)

**Printf Not Working:**
```c
// Enable semi-hosting or retarget _write()
int _write(int file, char *ptr, int len) {
    HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
```

## Resources

- **STM32 Website**: https://www.st.com/stm32
- **CubeMX**: https://www.st.com/en/development-tools/stm32cubemx.html
- **HAL Documentation**: STM32 HAL user manual per family
- **Reference Manuals**: Detailed peripheral descriptions
- **Community**: https://community.st.com/

## See Also

- [ARM Cortex-M Architecture](arm.md)
- [GPIO Programming](gpio.md)
- [UART Communication](uart.md)
- [SPI Protocol](spi.md)
- [I2C Protocol](i2c.md)
- [FreeRTOS](../rtos/freertos.md)
