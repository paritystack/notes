# AVR Microcontrollers

Comprehensive guide to AVR microcontroller programming with register-level control and bare-metal development.

## Table of Contents

1. [Introduction](#introduction)
2. [AVR Architecture](#avr-architecture)
3. [Development Setup](#development-setup)
4. [Register Programming](#register-programming)
5. [GPIO Control](#gpio-control)
6. [Timers and Counters](#timers-and-counters)
7. [Interrupts](#interrupts)
8. [Communication Protocols](#communication-protocols)
9. [Advanced Topics](#advanced-topics)

## Introduction

AVR is a family of 8-bit RISC microcontrollers developed by Atmel (now Microchip). They are widely used in Arduino boards and embedded systems due to their simplicity, efficiency, and low cost.

### Key Features

- **8-bit RISC Architecture**: Harvard architecture with separate program and data memory
- **Clock Speed**: 1-20 MHz
- **Flash Memory**: 2-256 KB
- **SRAM**: 128 bytes - 16 KB
- **EEPROM**: 64 bytes - 4 KB
- **Peripherals**: GPIO, Timers, ADC, UART, SPI, I2C
- **Power Efficient**: Multiple sleep modes
- **Price**: $1-5

### Popular AVR Microcontrollers

| MCU | Flash | RAM | EEPROM | GPIO | ADC | Timers | Package | Use Case |
|-----|-------|-----|--------|------|-----|--------|---------|----------|
| **ATtiny13** | 1 KB | 64 B | 64 B | 6 | 4 | 1 | 8-pin | Ultra-small projects |
| **ATtiny85** | 8 KB | 512 B | 512 B | 6 | 4 | 2 | 8-pin | Small projects |
| **ATmega8** | 8 KB | 1 KB | 512 B | 23 | 6 | 3 | 28-pin | Entry level |
| **ATmega328P** | 32 KB | 2 KB | 1 KB | 23 | 6 | 3 | 28-pin | Arduino Uno |
| **ATmega2560** | 256 KB | 8 KB | 4 KB | 86 | 16 | 6 | 100-pin | Arduino Mega |

## AVR Architecture

### Memory Organization

```
┌──────────────────────────────────────┐
│        AVR Memory Map                │
├──────────────────────────────────────┤
│  Program Memory (Flash)              │
│  ┌────────────────────────────────┐  │
│  │ 0x0000: Interrupt Vectors      │  │
│  │ 0x0034: Program Code           │  │
│  │ ...                            │  │
│  │ End:    Bootloader (optional)  │  │
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│  Data Memory (SRAM)                  │
│  ┌────────────────────────────────┐  │
│  │ 0x0000-0x001F: Registers (R0-R31)│
│  │ 0x0020-0x005F: I/O Registers   │  │
│  │ 0x0060-0x00FF: Extended I/O    │  │
│  │ 0x0100-...   : SRAM            │  │
│  │ ...          : Stack (grows ↓) │  │
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│  EEPROM (Non-volatile)               │
│  ┌────────────────────────────────┐  │
│  │ 0x0000: User data storage      │  │
│  │ ...                            │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### Registers

#### General Purpose Registers
```
R0-R31:  32 general-purpose 8-bit registers
R26-R27: X pointer (XL, XH)
R28-R29: Y pointer (YL, YH)
R30-R31: Z pointer (ZL, ZH)
```

#### Status Register (SREG)
```
Bit 7: I - Global Interrupt Enable
Bit 6: T - Transfer bit
Bit 5: H - Half Carry Flag
Bit 4: S - Sign Flag
Bit 3: V - Overflow Flag
Bit 2: N - Negative Flag
Bit 1: Z - Zero Flag
Bit 0: C - Carry Flag
```

### ATmega328P Pinout

```
         ATmega328P (DIP-28)
              ┌───∪───┐
    RESET  1 ─┤       ├─ 28  PC5/ADC5/SCL
    RXD/D0 2 ─┤       ├─ 27  PC4/ADC4/SDA
    TXD/D1 3 ─┤       ├─ 26  PC3/ADC3
    INT0/D2 4 ─┤       ├─ 25  PC2/ADC2
    INT1/D3 5 ─┤       ├─ 24  PC1/ADC1
    D4     6 ─┤       ├─ 23  PC0/ADC0
    VCC    7 ─┤       ├─ 22  GND
    GND    8 ─┤       ├─ 21  AREF
    XTAL1  9 ─┤       ├─ 20  AVCC
    XTAL2 10 ─┤       ├─ 19  PB5/SCK
    D5    11 ─┤       ├─ 18  PB4/MISO
    D6    12 ─┤       ├─ 17  PB3/MOSI
    D7    13 ─┤       ├─ 16  PB2/SS
    D8    14 ─┤       ├─ 15  PB1/OC1A
              └───────┘

GPIO Ports:
  Port B (PB0-PB5): Digital I/O, SPI
  Port C (PC0-PC5): Analog input (ADC), I2C
  Port D (PD0-PD7): Digital I/O, UART, Interrupts
```

## Development Setup

### AVR-GCC Toolchain

```bash
# Install AVR tools (Ubuntu/Debian)
sudo apt install gcc-avr avr-libc avrdude

# Install on Arch Linux
sudo pacman -S avr-gcc avr-libc avrdude

# Install on macOS
brew install avr-gcc avr-libc avrdude

# Verify installation
avr-gcc --version
avrdude -v
```

### Project Structure

```
project/
├── main.c
├── Makefile
└── README.md
```

### Makefile Template

```makefile
# AVR Makefile
MCU = atmega328p
F_CPU = 16000000UL
BAUD = 9600

CC = avr-gcc
OBJCOPY = avr-objcopy
OBJDUMP = avr-objdump
SIZE = avr-size

TARGET = main
SRC = main.c

CFLAGS = -mmcu=$(MCU) -DF_CPU=$(F_CPU) -DBAUD=$(BAUD)
CFLAGS += -Os -Wall -Wextra -std=c99

# Programmer settings
PROGRAMMER = arduino
PORT = /dev/ttyUSB0

all: $(TARGET).hex

$(TARGET).elf: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^
	$(SIZE) $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

flash: $(TARGET).hex
	avrdude -c $(PROGRAMMER) -p $(MCU) -P $(PORT) -U flash:w:$<

clean:
	rm -f $(TARGET).elf $(TARGET).hex

.PHONY: all flash clean
```

### Compiling and Flashing

```bash
# Compile
make

# Flash to device
make flash

# Clean build files
make clean

# Manual commands
avr-gcc -mmcu=atmega328p -DF_CPU=16000000UL -Os -o main.elf main.c
avr-objcopy -O ihex -R .eeprom main.elf main.hex
avrdude -c arduino -p atmega328p -P /dev/ttyUSB0 -U flash:w:main.hex
```

## Register Programming

### Understanding Registers

AVR programming requires direct manipulation of hardware registers. Each peripheral has associated registers for control and data.

#### Register Operations

```c
#include <avr/io.h>

/* Set bit (set to 1) */
PORTB |= (1 << PB5);

/* Clear bit (set to 0) */
PORTB &= ~(1 << PB5);

/* Toggle bit */
PORTB ^= (1 << PB5);

/* Check bit */
if (PIND & (1 << PD2)) {
    // Bit is set
}

/* Set multiple bits */
PORTB |= (1 << PB0) | (1 << PB1) | (1 << PB2);

/* Clear multiple bits */
PORTB &= ~((1 << PB0) | (1 << PB1));

/* Write entire register */
PORTB = 0b10101010;
```

## GPIO Control

### Port Registers

Each GPIO port has three registers:
- **DDRx**: Data Direction Register (1 = Output, 0 = Input)
- **PORTx**: Port Output Register (Output value or pull-up enable)
- **PINx**: Port Input Register (Read input state)

### Basic GPIO Example

```c
#include <avr/io.h>
#include <util/delay.h>

int main(void) {
    /* Set PB5 (Arduino pin 13) as output */
    DDRB |= (1 << DDB5);
    
    /* Main loop */
    while (1) {
        /* Turn LED on */
        PORTB |= (1 << PORTB5);
        _delay_ms(1000);
        
        /* Turn LED off */
        PORTB &= ~(1 << PORTB5);
        _delay_ms(1000);
    }
    
    return 0;
}
```

### Button Input with Pull-up

```c
#include <avr/io.h>
#include <util/delay.h>

int main(void) {
    /* PB5 as output (LED) */
    DDRB |= (1 << DDB5);
    
    /* PD2 as input (button) */
    DDRD &= ~(1 << DDD2);
    
    /* Enable pull-up resistor on PD2 */
    PORTD |= (1 << PORTD2);
    
    while (1) {
        /* Check if button pressed (active low) */
        if (\!(PIND & (1 << PIND2))) {
            PORTB |= (1 << PORTB5);   // LED on
        } else {
            PORTB &= ~(1 << PORTB5);  // LED off
        }
        
        _delay_ms(10);  // Debounce delay
    }
    
    return 0;
}
```

### Multiple LED Control

```c
#include <avr/io.h>
#include <util/delay.h>

int main(void) {
    /* Set PB0-PB5 as outputs */
    DDRB = 0b00111111;
    
    while (1) {
        /* Running LED pattern */
        for (uint8_t i = 0; i < 6; i++) {
            PORTB = (1 << i);
            _delay_ms(200);
        }
        
        /* Reverse */
        for (uint8_t i = 6; i > 0; i--) {
            PORTB = (1 << (i-1));
            _delay_ms(200);
        }
    }
    
    return 0;
}
```

## Timers and Counters

AVR timers are versatile peripherals for timing, counting, PWM generation, and more.

### Timer0 (8-bit)

```c
#include <avr/io.h>
#include <avr/interrupt.h>

volatile uint32_t milliseconds = 0;

/* Timer0 overflow interrupt */
ISR(TIMER0_OVF_vect) {
    milliseconds++;
}

void timer0_init(void) {
    /* Set prescaler to 64 */
    TCCR0B |= (1 << CS01) | (1 << CS00);
    
    /* Enable overflow interrupt */
    TIMSK0 |= (1 << TOIE0);
    
    /* Enable global interrupts */
    sei();
}

int main(void) {
    DDRB |= (1 << DDB5);
    timer0_init();
    
    while (1) {
        if (milliseconds >= 1000) {
            milliseconds = 0;
            PORTB ^= (1 << PORTB5);
        }
    }
    
    return 0;
}
```

### PWM with Timer1 (16-bit)

```c
#include <avr/io.h>
#include <util/delay.h>

void pwm_init(void) {
    /* Set PB1 (OC1A) as output */
    DDRB |= (1 << DDB1);
    
    /* Fast PWM, 10-bit, non-inverted */
    TCCR1A |= (1 << WGM11) | (1 << WGM10);
    TCCR1A |= (1 << COM1A1);
    TCCR1B |= (1 << WGM12) | (1 << CS10);  // No prescaling
    
    /* Set initial duty cycle */
    OCR1A = 512;  // 50% duty cycle (0-1023)
}

int main(void) {
    pwm_init();
    
    while (1) {
        /* Fade in */
        for (uint16_t i = 0; i <= 1023; i += 10) {
            OCR1A = i;
            _delay_ms(20);
        }
        
        /* Fade out */
        for (uint16_t i = 1023; i > 0; i -= 10) {
            OCR1A = i;
            _delay_ms(20);
        }
    }
    
    return 0;
}
```

### Timer2 CTC Mode (Precise Timing)

```c
#include <avr/io.h>
#include <avr/interrupt.h>

volatile uint8_t flag = 0;

/* Timer2 compare match interrupt - fires every 1ms */
ISR(TIMER2_COMPA_vect) {
    static uint16_t count = 0;
    count++;
    
    if (count >= 1000) {  // 1 second
        count = 0;
        flag = 1;
    }
}

void timer2_init(void) {
    /* CTC mode */
    TCCR2A |= (1 << WGM21);
    
    /* Prescaler 64: 16MHz / 64 = 250kHz */
    TCCR2B |= (1 << CS22);
    
    /* Compare value for 1ms: 250kHz / 250 = 1kHz */
    OCR2A = 249;
    
    /* Enable compare match interrupt */
    TIMSK2 |= (1 << OCIE2A);
    
    sei();
}

int main(void) {
    DDRB |= (1 << DDB5);
    timer2_init();
    
    while (1) {
        if (flag) {
            flag = 0;
            PORTB ^= (1 << PORTB5);
        }
    }
    
    return 0;
}
```

## Interrupts

### External Interrupts

```c
#include <avr/io.h>
#include <avr/interrupt.h>

volatile uint8_t led_state = 0;

/* INT0 interrupt handler */
ISR(INT0_vect) {
    led_state = \!led_state;
    
    if (led_state) {
        PORTB |= (1 << PORTB5);
    } else {
        PORTB &= ~(1 << PORTB5);
    }
}

void int0_init(void) {
    /* PD2 as input with pull-up */
    DDRD &= ~(1 << DDD2);
    PORTD |= (1 << PORTD2);
    
    /* Trigger on falling edge */
    EICRA |= (1 << ISC01);
    
    /* Enable INT0 */
    EIMSK |= (1 << INT0);
    
    sei();
}

int main(void) {
    DDRB |= (1 << DDB5);
    int0_init();
    
    while (1) {
        /* Main loop can do other things */
    }
    
    return 0;
}
```

### Pin Change Interrupts

```c
#include <avr/io.h>
#include <avr/interrupt.h>

/* PCINT0 interrupt (PB0-PB7) */
ISR(PCINT0_vect) {
    /* Check which pin changed */
    if (\!(PINB & (1 << PINB0))) {
        // PB0 is low
        PORTB |= (1 << PORTB5);
    } else {
        PORTB &= ~(1 << PORTB5);
    }
}

void pcint_init(void) {
    /* Enable pull-up on PB0 */
    PORTB |= (1 << PORTB0);
    
    /* Enable PCINT0 (PB0) */
    PCMSK0 |= (1 << PCINT0);
    
    /* Enable pin change interrupt 0 */
    PCICR |= (1 << PCIE0);
    
    sei();
}

int main(void) {
    DDRB |= (1 << DDB5);
    DDRB &= ~(1 << DDB0);
    
    pcint_init();
    
    while (1) {
        /* Main loop */
    }
    
    return 0;
}
```

## Communication Protocols

### UART (Serial Communication)

```c
#include <avr/io.h>
#include <util/delay.h>

#define BAUD 9600
#define MYUBRR F_CPU/16/BAUD-1

void uart_init(void) {
    /* Set baud rate */
    UBRR0H = (MYUBRR >> 8);
    UBRR0L = MYUBRR;
    
    /* Enable transmitter and receiver */
    UCSR0B = (1 << TXEN0) | (1 << RXEN0);
    
    /* Set frame format: 8 data bits, 1 stop bit */
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void uart_transmit(uint8_t data) {
    /* Wait for empty transmit buffer */
    while (\!(UCSR0A & (1 << UDRE0)));
    
    /* Put data into buffer */
    UDR0 = data;
}

uint8_t uart_receive(void) {
    /* Wait for data */
    while (\!(UCSR0A & (1 << RXC0)));
    
    /* Get and return data */
    return UDR0;
}

void uart_print(const char* str) {
    while (*str) {
        uart_transmit(*str++);
    }
}

int main(void) {
    uart_init();
    
    uart_print("Hello, AVR\!\r\n");
    
    while (1) {
        uint8_t received = uart_receive();
        uart_transmit(received);  // Echo back
    }
    
    return 0;
}
```

### SPI Master

```c
#include <avr/io.h>

void spi_init(void) {
    /* Set MOSI, SCK, and SS as outputs */
    DDRB |= (1 << DDB3) | (1 << DDB5) | (1 << DDB2);
    
    /* Set MISO as input */
    DDRB &= ~(1 << DDB4);
    
    /* Enable SPI, Master mode, clock = F_CPU/16 */
    SPCR = (1 << SPE) | (1 << MSTR) | (1 << SPR0);
}

uint8_t spi_transfer(uint8_t data) {
    /* Start transmission */
    SPDR = data;
    
    /* Wait for transmission complete */
    while (\!(SPSR & (1 << SPIF)));
    
    /* Return received data */
    return SPDR;
}

int main(void) {
    spi_init();
    
    while (1) {
        /* Select device (SS low) */
        PORTB &= ~(1 << PORTB2);
        
        /* Send data */
        spi_transfer(0xAB);
        uint8_t received = spi_transfer(0x00);
        
        /* Deselect device (SS high) */
        PORTB |= (1 << PORTB2);
    }
    
    return 0;
}
```

### I2C (TWI) Master

```c
#include <avr/io.h>
#include <util/twi.h>

#define F_SCL 100000UL  // 100 kHz
#define TWI_BITRATE ((F_CPU / F_SCL) - 16) / 2

void i2c_init(void) {
    /* Set bit rate */
    TWBR = (uint8_t)TWI_BITRATE;
    
    /* Enable TWI */
    TWCR = (1 << TWEN);
}

void i2c_start(void) {
    /* Send start condition */
    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
    
    /* Wait for completion */
    while (\!(TWCR & (1 << TWINT)));
}

void i2c_stop(void) {
    /* Send stop condition */
    TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
}

void i2c_write(uint8_t data) {
    /* Load data */
    TWDR = data;
    
    /* Start transmission */
    TWCR = (1 << TWINT) | (1 << TWEN);
    
    /* Wait for completion */
    while (\!(TWCR & (1 << TWINT)));
}

uint8_t i2c_read_ack(void) {
    /* Enable ACK */
    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWEA);
    
    /* Wait for completion */
    while (\!(TWCR & (1 << TWINT)));
    
    return TWDR;
}

uint8_t i2c_read_nack(void) {
    /* Enable NACK */
    TWCR = (1 << TWINT) | (1 << TWEN);
    
    /* Wait for completion */
    while (\!(TWCR & (1 << TWINT)));
    
    return TWDR;
}

int main(void) {
    i2c_init();
    
    uint8_t device_addr = 0x68 << 1;  // 7-bit address
    uint8_t reg_addr = 0x00;
    
    while (1) {
        /* Write to device */
        i2c_start();
        i2c_write(device_addr | 0);  // Write mode
        i2c_write(reg_addr);
        i2c_write(0x42);  // Data
        i2c_stop();
        
        /* Read from device */
        i2c_start();
        i2c_write(device_addr | 0);  // Write mode
        i2c_write(reg_addr);
        i2c_start();  // Repeated start
        i2c_write(device_addr | 1);  // Read mode
        uint8_t data = i2c_read_nack();
        i2c_stop();
    }
    
    return 0;
}
```

## Advanced Topics

### ADC (Analog-to-Digital Converter)

```c
#include <avr/io.h>

void adc_init(void) {
    /* AVCC with external capacitor at AREF */
    ADMUX = (1 << REFS0);
    
    /* Enable ADC, prescaler 128 (125 kHz @ 16 MHz) */
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}

uint16_t adc_read(uint8_t channel) {
    /* Select channel (0-7) */
    ADMUX = (ADMUX & 0xF0) | (channel & 0x0F);
    
    /* Start conversion */
    ADCSRA |= (1 << ADSC);
    
    /* Wait for completion */
    while (ADCSRA & (1 << ADSC));
    
    return ADC;
}

int main(void) {
    uart_init();
    adc_init();
    
    while (1) {
        uint16_t value = adc_read(0);  // Read ADC0
        
        /* Convert to voltage (5V reference, 10-bit) */
        float voltage = (value * 5.0) / 1024.0;
        
        _delay_ms(100);
    }
    
    return 0;
}
```

### EEPROM Access

```c
#include <avr/io.h>
#include <avr/eeprom.h>

uint8_t EEMEM stored_value;  // EEPROM variable

void eeprom_write_byte_custom(uint16_t address, uint8_t data) {
    /* Wait for completion of previous write */
    while (EECR & (1 << EEPE));
    
    /* Set address and data registers */
    EEAR = address;
    EEDR = data;
    
    /* Write logical one to EEMPE */
    EECR |= (1 << EEMPE);
    
    /* Start eeprom write by setting EEPE */
    EECR |= (1 << EEPE);
}

uint8_t eeprom_read_byte_custom(uint16_t address) {
    /* Wait for completion of previous write */
    while (EECR & (1 << EEPE));
    
    /* Set address register */
    EEAR = address;
    
    /* Start eeprom read by writing EERE */
    EECR |= (1 << EERE);
    
    /* Return data from data register */
    return EEDR;
}

int main(void) {
    /* Using avr-libc functions (recommended) */
    eeprom_write_byte(&stored_value, 42);
    uint8_t value = eeprom_read_byte(&stored_value);
    
    /* Using custom functions */
    eeprom_write_byte_custom(0, 100);
    uint8_t val = eeprom_read_byte_custom(0);
    
    while (1);
    
    return 0;
}
```

### Sleep Modes

```c
#include <avr/io.h>
#include <avr/sleep.h>
#include <avr/interrupt.h>

ISR(INT0_vect) {
    /* Wake up from sleep */
}

int main(void) {
    /* Configure wake-up source */
    EIMSK |= (1 << INT0);
    sei();
    
    /* Set sleep mode */
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    
    while (1) {
        /* Enter sleep mode */
        sleep_mode();
        
        /* Wake up here and continue */
        PORTB ^= (1 << PORTB5);
    }
    
    return 0;
}
```

### Watchdog Timer

```c
#include <avr/io.h>
#include <avr/wdt.h>

int main(void) {
    /* Disable watchdog on reset */
    MCUSR &= ~(1 << WDRF);
    wdt_disable();
    
    /* Enable watchdog: 2 second timeout */
    wdt_enable(WDTO_2S);
    
    while (1) {
        /* Main program */
        
        /* Reset watchdog timer */
        wdt_reset();
    }
    
    return 0;
}
```

## Best Practices

1. **Use Register Macros**: `PORTB |= (1 << PB5)` instead of `PORTB |= 0x20`
2. **Volatile for ISR Variables**: `volatile uint8_t flag;`
3. **Minimize ISR Time**: Keep interrupt handlers short
4. **Proper Delays**: Use timers instead of `_delay_ms()` for long delays
5. **Power Management**: Disable unused peripherals, use sleep modes
6. **Debouncing**: Add delays or use interrupts with debounce logic
7. **Code Organization**: Separate initialization from main loop

## Troubleshooting

### Common Issues

**Program Not Running:**
- Check fuse bits (clock source, brown-out detection)
- Verify F_CPU matches actual clock speed
- Ensure power supply is stable

**Incorrect Baud Rate:**
- Verify F_CPU is correct
- Check UBRR calculation
- Use standard baud rates

**Fuse Bits:**
```bash
# Read fuses
avrdude -c arduino -p atmega328p -U lfuse:r:-:h -U hfuse:r:-:h -U efuse:r:-:h

# Set fuses (CAREFUL\!)
# Default for Arduino Uno: lfuse=0xFF, hfuse=0xDE, efuse=0xFD
avrdude -c arduino -p atmega328p -U lfuse:w:0xFF:m -U hfuse:w:0xDE:m -U efuse:w:0xFD:m
```

## Resources

- **AVR Libc Documentation**: https://www.nongnu.org/avr-libc/
- **Datasheets**: https://www.microchip.com/
- **AVR Tutorials**: https://www.avrfreaks.net/
- **Community**: AVRFreaks forum

## See Also

- [Arduino Programming](arduino.md) - Higher-level AVR programming
- [GPIO Concepts](gpio.md)
- [UART Communication](uart.md)
- [SPI Protocol](spi.md)
- [I2C Protocol](i2c.md)
- [Timers and PWM](timers.md)
