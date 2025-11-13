# Real-Time Clock (RTC) Modules

Comprehensive guide to RTC modules including DS1307, DS3231, and implementation examples.

## Table of Contents

1. [Introduction](#introduction)
2. [RTC Basics](#rtc-basics)
3. [DS1307](#ds1307)
4. [DS3231](#ds3231)
5. [PCF8523](#pcf8523)
6. [Arduino Examples](#arduino-examples)
7. [STM32 Examples](#stm32-examples)
8. [AVR Bare Metal](#avr-bare-metal)

## Introduction

Real-Time Clock (RTC) modules are specialized integrated circuits that keep accurate time even when the main system is powered off. They are essential for data logging, scheduling, timestamps, and time-based applications.

### Why Use an RTC Module?

- **Accurate Timekeeping**: Crystal oscillator provides precise time
- **Low Power**: Runs on backup battery for years
- **Independent Operation**: Maintains time when main power is off
- **Calendar Functions**: Handles dates, months, leap years automatically
- **Alarms**: Can trigger events at specific times

### Popular RTC Modules

| Module | Crystal | Accuracy | Battery | Temperature | I2C Addr | Price |
|--------|---------|----------|---------|-------------|----------|-------|
| **DS1307** | 32.768 kHz | ±2 min/month | CR2032 | No | 0x68 | $1 |
| **DS3231** | 32.768 kHz (TCXO) | ±2 min/year | CR2032 | Yes | 0x68 | $2-5 |
| **PCF8523** | 32.768 kHz | ±3 min/year | CR2032 | No | 0x68 | $2 |
| **MCP7940N** | 32.768 kHz | ±2 min/month | CR2032 | No | 0x6F | $1 |

## RTC Basics

### Time Representation

RTCs store time in BCD (Binary Coded Decimal) format:
```
Decimal 59 = 0101 1001 BCD
             5     9

Decimal to BCD: 59 = (5 << 4) | 9 = 0x59
BCD to Decimal: 0x59 = ((0x59 >> 4) * 10) + (0x59 & 0x0F) = 59
```

### BCD Conversion Functions

```c
// Decimal to BCD
uint8_t dec_to_bcd(uint8_t val) {
    return ((val / 10) << 4) | (val % 10);
}

// BCD to Decimal
uint8_t bcd_to_dec(uint8_t val) {
    return ((val >> 4) * 10) + (val & 0x0F);
}
```

### I2C Communication

All popular RTC modules use I2C interface:
```
Connections:
  RTC VCC  -> 3.3V or 5V
  RTC GND  -> GND
  RTC SDA  -> SDA (with pull-up resistor)
  RTC SCL  -> SCL (with pull-up resistor)
  
Pull-up resistors: 4.7kΩ typical
```

**Wiring Diagram:**
```
RTC Module          Microcontroller
    ┌────┐
VCC ┤    ├─ VCC (3.3V/5V)
GND ┤    ├─ GND
SDA ┤    ├─ SDA (with 4.7kΩ pull-up)
SCL ┤    ├─ SCL (with 4.7kΩ pull-up)
    └────┘
```

## DS1307

### Features

- **Accuracy**: ±2 minutes per month
- **Operating Voltage**: 4.5-5.5V (5V recommended)
- **Battery Backup**: CR2032 (typical)
- **Interface**: I2C (100 kHz)
- **Address**: 0x68 (fixed)
- **RAM**: 56 bytes of non-volatile SRAM
- **Output**: 1 Hz square wave

### Register Map

```
Register  Function
0x00      Seconds (00-59)
0x01      Minutes (00-59)
0x02      Hours (00-23 or 01-12)
0x03      Day of week (1-7)
0x04      Date (01-31)
0x05      Month (01-12)
0x06      Year (00-99)
0x07      Control (SQW output)
0x08-0x3F RAM (56 bytes)

Bit Layout:
Seconds:  0  | 10-sec | sec
          CH | 4 2 1  | 8 4 2 1

CH = Clock Halt bit (0 = running, 1 = stopped)
```

### Arduino DS1307 Library

```cpp
#include <Wire.h>
#include <RTClib.h>

RTC_DS1307 rtc;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    
    if (!rtc.begin()) {
        Serial.println("Couldn't find RTC");
        while (1);
    }
    
    if (!rtc.isrunning()) {
        Serial.println("RTC is NOT running, setting time...");
        // Set to compile time
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
        // Or set manually:
        // rtc.adjust(DateTime(2024, 1, 15, 12, 30, 0));
    }
}

void loop() {
    DateTime now = rtc.now();
    
    Serial.print(now.year(), DEC);
    Serial.print('/');
    Serial.print(now.month(), DEC);
    Serial.print('/');
    Serial.print(now.day(), DEC);
    Serial.print(" ");
    Serial.print(now.hour(), DEC);
    Serial.print(':');
    Serial.print(now.minute(), DEC);
    Serial.print(':');
    Serial.println(now.second(), DEC);
    
    delay(1000);
}
```

### DS1307 Bare Metal (Arduino)

```cpp
#include <Wire.h>

#define DS1307_ADDR 0x68

uint8_t dec_to_bcd(uint8_t val) {
    return ((val / 10) << 4) | (val % 10);
}

uint8_t bcd_to_dec(uint8_t val) {
    return ((val >> 4) * 10) + (val & 0x0F);
}

void ds1307_set_time(uint8_t hour, uint8_t min, uint8_t sec) {
    Wire.beginTransmission(DS1307_ADDR);
    Wire.write(0x00);  // Start at seconds register
    Wire.write(dec_to_bcd(sec) & 0x7F);  // Clear CH bit
    Wire.write(dec_to_bcd(min));
    Wire.write(dec_to_bcd(hour));
    Wire.endTransmission();
}

void ds1307_set_date(uint8_t day, uint8_t date, uint8_t month, uint8_t year) {
    Wire.beginTransmission(DS1307_ADDR);
    Wire.write(0x03);  // Start at day register
    Wire.write(dec_to_bcd(day));
    Wire.write(dec_to_bcd(date));
    Wire.write(dec_to_bcd(month));
    Wire.write(dec_to_bcd(year));
    Wire.endTransmission();
}

void ds1307_read_time(uint8_t *hour, uint8_t *min, uint8_t *sec) {
    Wire.beginTransmission(DS1307_ADDR);
    Wire.write(0x00);  // Start at seconds register
    Wire.endTransmission();
    
    Wire.requestFrom(DS1307_ADDR, 3);
    *sec = bcd_to_dec(Wire.read() & 0x7F);
    *min = bcd_to_dec(Wire.read());
    *hour = bcd_to_dec(Wire.read());
}

void setup() {
    Serial.begin(9600);
    Wire.begin();
    
    // Set time: 12:30:00
    ds1307_set_time(12, 30, 0);
    
    // Set date: Monday, 15/01/24
    ds1307_set_date(1, 15, 1, 24);
}

void loop() {
    uint8_t hour, min, sec;
    
    ds1307_read_time(&hour, &min, &sec);
    
    Serial.print(hour);
    Serial.print(":");
    Serial.print(min);
    Serial.print(":");
    Serial.println(sec);
    
    delay(1000);
}
```

## DS3231

### Features

- **Accuracy**: ±2 minutes per year (much better than DS1307)
- **Temperature Compensated**: TCXO provides better accuracy
- **Operating Voltage**: 2.3-5.5V
- **Battery Backup**: CR2032
- **Interface**: I2C (100-400 kHz)
- **Address**: 0x68 (fixed)
- **Temperature Sensor**: Built-in (±3°C accuracy)
- **Alarms**: Two programmable alarms
- **Square Wave Output**: 1Hz, 1.024kHz, 4.096kHz, 8.192kHz

### Register Map

```
Register  Function
0x00      Seconds (00-59)
0x01      Minutes (00-59)
0x02      Hours (00-23 or 01-12)
0x03      Day of week (1-7)
0x04      Date (01-31)
0x05      Month/Century (01-12)
0x06      Year (00-99)
0x07-0x0A Alarm 1
0x0B-0x0D Alarm 2
0x0E      Control
0x0F      Control/Status
0x10      Aging offset
0x11-0x12 Temperature
```

### Arduino DS3231 Library

```cpp
#include <Wire.h>
#include <RTClib.h>

RTC_DS3231 rtc;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    
    if (!rtc.begin()) {
        Serial.println("Couldn't find RTC");
        while (1);
    }
    
    if (rtc.lostPower()) {
        Serial.println("RTC lost power, setting time...");
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
}

void loop() {
    DateTime now = rtc.now();
    
    // Print time
    Serial.print(now.year(), DEC);
    Serial.print('/');
    Serial.print(now.month(), DEC);
    Serial.print('/');
    Serial.print(now.day(), DEC);
    Serial.print(" ");
    Serial.print(now.hour(), DEC);
    Serial.print(':');
    Serial.print(now.minute(), DEC);
    Serial.print(':');
    Serial.print(now.second(), DEC);
    
    // Print temperature
    float temp = rtc.getTemperature();
    Serial.print(" Temp: ");
    Serial.print(temp);
    Serial.println("°C");
    
    delay(1000);
}
```

### DS3231 Alarm Example

```cpp
#include <Wire.h>
#include <RTClib.h>

RTC_DS3231 rtc;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    rtc.begin();
    
    // Set alarm 1 for every day at 12:30:00
    rtc.setAlarm1(DateTime(0, 0, 0, 12, 30, 0), DS3231_A1_Hour);
    
    // Enable alarm interrupt
    rtc.disableAlarm(1);
    rtc.disableAlarm(2);
    rtc.clearAlarm(1);
    rtc.clearAlarm(2);
    rtc.writeSqwPinMode(DS3231_OFF);  // Disable square wave
}

void loop() {
    if (rtc.alarmFired(1)) {
        Serial.println("Alarm 1 triggered!");
        rtc.clearAlarm(1);
    }
    
    delay(1000);
}
```

### DS3231 Temperature Reading

```cpp
#include <Wire.h>

#define DS3231_ADDR 0x68
#define TEMP_MSB    0x11
#define TEMP_LSB    0x12

float ds3231_get_temperature() {
    Wire.beginTransmission(DS3231_ADDR);
    Wire.write(TEMP_MSB);
    Wire.endTransmission();
    
    Wire.requestFrom(DS3231_ADDR, 2);
    uint8_t msb = Wire.read();
    uint8_t lsb = Wire.read();
    
    // Combine MSB and LSB
    int16_t temp = (msb << 2) | (lsb >> 6);
    
    // Handle negative temperatures
    if (temp & 0x200) {
        temp |= 0xFC00;
    }
    
    return temp * 0.25;
}

void setup() {
    Serial.begin(9600);
    Wire.begin();
}

void loop() {
    float temperature = ds3231_get_temperature();
    
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println("°C");
    
    delay(1000);
}
```

## PCF8523

### Features

- **Accuracy**: ±3 minutes per year
- **Operating Voltage**: 1.8-5.5V
- **Battery Backup**: CR2032
- **Interface**: I2C (100-400 kHz)
- **Address**: 0x68 (fixed)
- **Alarm**: Single programmable alarm
- **Timer**: Countdown timer
- **Low Power**: Multiple power-saving modes

### Arduino PCF8523

```cpp
#include <Wire.h>
#include <RTClib.h>

RTC_PCF8523 rtc;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    
    if (!rtc.begin()) {
        Serial.println("Couldn't find RTC");
        while (1);
    }
    
    if (!rtc.initialized() || rtc.lostPower()) {
        Serial.println("RTC is NOT initialized, setting time...");
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
    
    // Start RTC
    rtc.start();
}

void loop() {
    DateTime now = rtc.now();
    
    Serial.print(now.year(), DEC);
    Serial.print('/');
    Serial.print(now.month(), DEC);
    Serial.print('/');
    Serial.print(now.day(), DEC);
    Serial.print(" ");
    Serial.print(now.hour(), DEC);
    Serial.print(':');
    Serial.print(now.minute(), DEC);
    Serial.print(':');
    Serial.println(now.second(), DEC);
    
    delay(1000);
}
```

## Arduino Examples

### Data Logger with RTC

```cpp
#include <Wire.h>
#include <RTClib.h>
#include <SD.h>

RTC_DS3231 rtc;
const int CS_PIN = 10;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    
    if (!rtc.begin()) {
        Serial.println("RTC error");
        while (1);
    }
    
    if (!SD.begin(CS_PIN)) {
        Serial.println("SD card error");
        while (1);
    }
}

void loop() {
    DateTime now = rtc.now();
    float temp = rtc.getTemperature();
    
    // Create filename
    char filename[13];
    sprintf(filename, "%04d%02d%02d.txt", 
            now.year(), now.month(), now.day());
    
    // Open file
    File dataFile = SD.open(filename, FILE_WRITE);
    
    if (dataFile) {
        // Write timestamp and data
        dataFile.print(now.hour());
        dataFile.print(":");
        dataFile.print(now.minute());
        dataFile.print(":");
        dataFile.print(now.second());
        dataFile.print(",");
        dataFile.println(temp);
        
        dataFile.close();
        
        Serial.println("Data logged");
    } else {
        Serial.println("Error opening file");
    }
    
    delay(60000);  // Log every minute
}
```

### Digital Clock Display

```cpp
#include <Wire.h>
#include <RTClib.h>
#include <LiquidCrystal.h>

RTC_DS3231 rtc;
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
    Wire.begin();
    rtc.begin();
    lcd.begin(16, 2);
    
    if (rtc.lostPower()) {
        rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    }
}

void loop() {
    DateTime now = rtc.now();
    
    // Display date on line 1
    lcd.setCursor(0, 0);
    lcd.print(now.day(), DEC);
    lcd.print('/');
    lcd.print(now.month(), DEC);
    lcd.print('/');
    lcd.print(now.year(), DEC);
    lcd.print("   ");
    
    // Display time on line 2
    lcd.setCursor(0, 1);
    if (now.hour() < 10) lcd.print('0');
    lcd.print(now.hour(), DEC);
    lcd.print(':');
    if (now.minute() < 10) lcd.print('0');
    lcd.print(now.minute(), DEC);
    lcd.print(':');
    if (now.second() < 10) lcd.print('0');
    lcd.print(now.second(), DEC);
    
    delay(1000);
}
```

### Alarm Clock

```cpp
#include <Wire.h>
#include <RTClib.h>

RTC_DS3231 rtc;

const int BUZZER_PIN = 9;
const int BUTTON_PIN = 2;

uint8_t alarm_hour = 7;
uint8_t alarm_minute = 30;
bool alarm_active = false;

void setup() {
    Serial.begin(9600);
    Wire.begin();
    rtc.begin();
    
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    
    // Set alarm
    rtc.setAlarm1(DateTime(0, 0, 0, alarm_hour, alarm_minute, 0), 
                  DS3231_A1_Hour);
}

void loop() {
    DateTime now = rtc.now();
    
    // Check alarm
    if (rtc.alarmFired(1)) {
        alarm_active = true;
        rtc.clearAlarm(1);
    }
    
    // Sound buzzer if alarm active
    if (alarm_active) {
        tone(BUZZER_PIN, 1000, 500);
        delay(1000);
        
        // Check for button press to stop
        if (digitalRead(BUTTON_PIN) == LOW) {
            alarm_active = false;
            noTone(BUZZER_PIN);
        }
    }
    
    // Display time
    Serial.print(now.hour());
    Serial.print(":");
    Serial.print(now.minute());
    Serial.print(":");
    Serial.println(now.second());
    
    delay(1000);
}
```

## STM32 Examples

### DS3231 with STM32 HAL

```c
#include "main.h"
#include <stdio.h>

I2C_HandleTypeDef hi2c1;

#define DS3231_ADDR (0x68 << 1)

uint8_t dec_to_bcd(uint8_t val) {
    return ((val / 10) << 4) | (val % 10);
}

uint8_t bcd_to_dec(uint8_t val) {
    return ((val >> 4) * 10) + (val & 0x0F);
}

void ds3231_set_time(uint8_t hour, uint8_t min, uint8_t sec) {
    uint8_t data[4];
    data[0] = 0x00;  // Start register
    data[1] = dec_to_bcd(sec);
    data[2] = dec_to_bcd(min);
    data[3] = dec_to_bcd(hour);
    
    HAL_I2C_Master_Transmit(&hi2c1, DS3231_ADDR, data, 4, HAL_MAX_DELAY);
}

void ds3231_read_time(uint8_t *hour, uint8_t *min, uint8_t *sec) {
    uint8_t reg = 0x00;
    uint8_t data[3];
    
    HAL_I2C_Master_Transmit(&hi2c1, DS3231_ADDR, &reg, 1, HAL_MAX_DELAY);
    HAL_I2C_Master_Receive(&hi2c1, DS3231_ADDR, data, 3, HAL_MAX_DELAY);
    
    *sec = bcd_to_dec(data[0]);
    *min = bcd_to_dec(data[1]);
    *hour = bcd_to_dec(data[2]);
}

float ds3231_get_temperature(void) {
    uint8_t reg = 0x11;
    uint8_t data[2];
    
    HAL_I2C_Master_Transmit(&hi2c1, DS3231_ADDR, &reg, 1, HAL_MAX_DELAY);
    HAL_I2C_Master_Receive(&hi2c1, DS3231_ADDR, data, 2, HAL_MAX_DELAY);
    
    int16_t temp = (data[0] << 2) | (data[1] >> 6);
    
    if (temp & 0x200) {
        temp |= 0xFC00;
    }
    
    return temp * 0.25;
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    
    // Set initial time
    ds3231_set_time(12, 30, 0);
    
    while (1) {
        uint8_t hour, min, sec;
        ds3231_read_time(&hour, &min, &sec);
        
        float temp = ds3231_get_temperature();
        
        printf("%02d:%02d:%02d Temp: %.2f°C\r\n", hour, min, sec, temp);
        
        HAL_Delay(1000);
    }
}
```

## AVR Bare Metal

### DS1307 with AVR (ATmega328P)

```c
#include <avr/io.h>
#include <util/delay.h>
#include <stdio.h>

#define DS1307_ADDR 0x68
#define F_SCL 100000UL
#define TWI_BITRATE ((F_CPU / F_SCL) - 16) / 2

uint8_t dec_to_bcd(uint8_t val) {
    return ((val / 10) << 4) | (val % 10);
}

uint8_t bcd_to_dec(uint8_t val) {
    return ((val >> 4) * 10) + (val & 0x0F);
}

void i2c_init(void) {
    TWBR = (uint8_t)TWI_BITRATE;
    TWCR = (1 << TWEN);
}

void i2c_start(void) {
    TWCR = (1 << TWINT) | (1 << TWSTA) | (1 << TWEN);
    while (!(TWCR & (1 << TWINT)));
}

void i2c_stop(void) {
    TWCR = (1 << TWINT) | (1 << TWSTO) | (1 << TWEN);
}

void i2c_write(uint8_t data) {
    TWDR = data;
    TWCR = (1 << TWINT) | (1 << TWEN);
    while (!(TWCR & (1 << TWINT)));
}

uint8_t i2c_read_ack(void) {
    TWCR = (1 << TWINT) | (1 << TWEN) | (1 << TWEA);
    while (!(TWCR & (1 << TWINT)));
    return TWDR;
}

uint8_t i2c_read_nack(void) {
    TWCR = (1 << TWINT) | (1 << TWEN);
    while (!(TWCR & (1 << TWINT)));
    return TWDR;
}

void ds1307_set_time(uint8_t hour, uint8_t min, uint8_t sec) {
    i2c_start();
    i2c_write((DS1307_ADDR << 1) | 0);
    i2c_write(0x00);  // Start register
    i2c_write(dec_to_bcd(sec));
    i2c_write(dec_to_bcd(min));
    i2c_write(dec_to_bcd(hour));
    i2c_stop();
}

void ds1307_read_time(uint8_t *hour, uint8_t *min, uint8_t *sec) {
    i2c_start();
    i2c_write((DS1307_ADDR << 1) | 0);
    i2c_write(0x00);  // Start register
    i2c_start();  // Repeated start
    i2c_write((DS1307_ADDR << 1) | 1);
    
    *sec = bcd_to_dec(i2c_read_ack());
    *min = bcd_to_dec(i2c_read_ack());
    *hour = bcd_to_dec(i2c_read_nack());
    
    i2c_stop();
}

int main(void) {
    i2c_init();
    uart_init();  // Assume UART is initialized
    
    // Set time to 12:30:00
    ds1307_set_time(12, 30, 0);
    
    while (1) {
        uint8_t hour, min, sec;
        ds1307_read_time(&hour, &min, &sec);
        
        printf("%02d:%02d:%02d\n", hour, min, sec);
        
        _delay_ms(1000);
    }
    
    return 0;
}
```

## Best Practices

1. **Battery Backup**: Always install backup battery for continuous operation
2. **Pull-up Resistors**: Ensure 4.7kΩ pull-ups on SDA and SCL
3. **Power Supply**: DS1307 requires 5V, DS3231 works with 3.3V-5V
4. **Initial Setup**: Set time after first power-on or battery change
5. **Lost Power Check**: Check and handle RTC power loss
6. **BCD Format**: Remember to convert between decimal and BCD
7. **I2C Speed**: Use 100 kHz for reliability, 400 kHz if needed

## Troubleshooting

### Common Issues

**RTC Not Responding:**
- Check I2C address (usually 0x68)
- Verify SDA/SCL connections
- Ensure pull-up resistors present
- Check power supply voltage

**Time Not Keeping:**
- Install backup battery (CR2032)
- Check battery voltage (should be ~3V)
- For DS1307: Clear CH (Clock Halt) bit
- Verify crystal oscillator is working

**Inaccurate Time:**
- DS1307: Normal (±2 min/month), consider DS3231
- DS3231: Check temperature effects
- Calibrate using aging offset register (DS3231)

**I2C Communication Errors:**
```cpp
// Check I2C scanner result
Wire.beginTransmission(0x68);
if (Wire.endTransmission() == 0) {
    Serial.println("RTC found at 0x68");
} else {
    Serial.println("RTC not found");
}
```

## Resources

- **DS1307 Datasheet**: Maxim Integrated
- **DS3231 Datasheet**: Maxim Integrated
- **RTClib Library**: https://github.com/adafruit/RTClib
- **I2C Protocol**: See [I2C documentation](i2c.md)

## See Also

- [I2C Communication](i2c.md)
- [Arduino Programming](arduino.md)
- [STM32 HAL](stm32.md)
- [AVR Programming](avr.md)
- [Data Logging](data_logging.md)
