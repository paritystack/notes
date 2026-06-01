# UART (Universal Asynchronous Receiver-Transmitter)

## Overview

UART is one of the most commonly used serial communication protocols in embedded systems. Unlike [SPI](spi.md) and [I2C](i2c.md), UART is asynchronous — meaning it doesn't require a shared clock signal between devices. This makes it simple, robust, and perfect for point-to-point communication between two devices. For multi-node industrial buses see [CAN](can.md); UART TX/RX lines are routed through [GPIO](gpio.md) alternate-function pins and often the first tool for [debugging](debugging.md) embedded firmware.

## Key Features

- **Asynchronous**: No shared clock required - devices use pre-agreed baud rates
- **Point-to-Point**: Communication between exactly two devices
- **Two-Wire Interface**: Only TX (transmit) and RX (receive) lines needed
- **Full-Duplex**: Can send and receive data simultaneously
- **Simple**: Easy to implement and debug
- **Universal**: Supported by virtually all microcontrollers

## Signal Lines

UART uses only two main signal lines (plus ground):

| Signal | Description |
|--------|-------------|
| **TX** | Transmit Data - Output from device |
| **RX** | Receive Data - Input to device |
| **GND** | Common ground reference |

**Important Wiring**: TX of device A connects to RX of device B, and vice versa!

```
Device A              Device B
  TX      -------->     RX
  RX      <--------     TX
  GND     ---------    GND
```

## How It Works

### Data Frame Structure

A typical UART data frame consists of:

```
  Start   Data Bits   Parity  Stop
   Bit    (5-9)       (Opt)  Bit(s)
        , , , , , , ,
        0 1 2 3 4 5 6 7
```

1. **Idle State**: Line is HIGH when no data is being sent
2. **Start Bit**: Single LOW bit signals beginning of frame
3. **Data Bits**: 5-9 bits of actual data (usually 8 bits)
4. **Parity Bit** (Optional): Error checking bit
5. **Stop Bit(s)**: 1, 1.5, or 2 HIGH bits signal end of frame

### Baud Rate

Baud rate is the speed of communication, measured in bits per second (bps).

**Common Baud Rates**:
- 9600 bps - Default for many applications
- 19200 bps
- 38400 bps
- 57600 bps
- 115200 bps - Common for debugging/logging
- 230400 bps
- 921600 bps - High-speed applications

**Formula**:
```
Bit Duration = 1 / Baud Rate
```
At 9600 baud: each bit takes ~104 microseconds

### Parity Bit

Parity is a simple error detection method:

- **Even Parity**: Parity bit set so total number of 1s is even
- **Odd Parity**: Parity bit set so total number of 1s is odd
- **None**: No parity bit (most common)

### Configuration Format

UART settings are often written as: **Baud-Data-Parity-Stop**

Examples:
- `9600-8-N-1`: 9600 baud, 8 data bits, No parity, 1 stop bit (most common)
- `115200-8-E-1`: 115200 baud, 8 data bits, Even parity, 1 stop bit

## Code Examples

### Arduino UART

```cpp
void setup() {
  // Initialize Serial (UART0) at 9600 baud
  Serial.begin(9600);

  // For other UART ports on boards like Arduino Mega:
  // Serial1.begin(115200);
  // Serial2.begin(9600);

  // Wait for serial port to connect
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB)
  }

  Serial.println("UART initialized!");
}

void loop() {
  // Sending data
  Serial.print("Temperature: ");
  Serial.println(25.5);

  // Sending formatted data
  char buffer[50];
  sprintf(buffer, "Value: %d, Time: %lu", 42, millis());
  Serial.println(buffer);

  // Reading data
  if (Serial.available() > 0) {
    // Read a single byte
    char incoming = Serial.read();

    // Read until newline
    String command = Serial.readStringUntil('\n');

    // Read with timeout (default 1000ms)
    Serial.setTimeout(500);
    int value = Serial.parseInt();

    Serial.print("Received: ");
    Serial.println(command);
  }

  delay(1000);
}
```

### ESP32 Multiple UARTs

```cpp
// ESP32 has 3 hardware UARTs
HardwareSerial SerialGPS(1);   // UART1
HardwareSerial SerialModem(2); // UART2

void setup() {
  // Serial0 (USB) - default pins
  Serial.begin(115200);

  // UART1 - custom pins (TX=17, RX=16)
  SerialGPS.begin(9600, SERIAL_8N1, 16, 17);

  // UART2 - custom pins (TX=25, RX=26)
  SerialModem.begin(115200, SERIAL_8N1, 26, 25);
}

void loop() {
  // Read from GPS on UART1
  if (SerialGPS.available()) {
    String gpsData = SerialGPS.readStringUntil('\n');
    Serial.println("GPS: " + gpsData);
  }

  // Read from modem on UART2
  if (SerialModem.available()) {
    String modemResponse = SerialModem.readStringUntil('\n');
    Serial.println("Modem: " + modemResponse);
  }
}
```

### STM32 HAL UART

```c
#include "stm32f4xx_hal.h"

UART_HandleTypeDef huart2;

void UART_Init(void) {
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart2);
}

// Blocking transmission
void UART_SendString(char *str) {
    HAL_UART_Transmit(&huart2, (uint8_t*)str, strlen(str), 100);
}

// Blocking reception
void UART_ReceiveData(uint8_t *buffer, uint16_t size) {
    HAL_UART_Receive(&huart2, buffer, size, 1000);
}

// Interrupt-based reception
void UART_ReceiveIT(uint8_t *buffer, uint16_t size) {
    HAL_UART_Receive_IT(&huart2, buffer, size);
}

// Callback when reception complete
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART2) {
        // Process received data
        // Re-enable reception
        HAL_UART_Receive_IT(&huart2, rxBuffer, RX_BUFFER_SIZE);
    }
}

// DMA-based high-speed transfer
void UART_Transmit_DMA(uint8_t *data, uint16_t size) {
    HAL_UART_Transmit_DMA(&huart2, data, size);
}
```

### Bare-Metal AVR (Arduino Uno)

```c
#include <avr/io.h>

#define BAUD 9600
#define UBRR_VALUE ((F_CPU / 16 / BAUD) - 1)

void UART_Init(void) {
    // Set baud rate
    UBRR0H = (UBRR_VALUE >> 8);
    UBRR0L = UBRR_VALUE;

    // Enable transmitter and receiver
    UCSR0B = (1 << TXEN0) | (1 << RXEN0);

    // Set frame format: 8 data bits, 1 stop bit, no parity
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

void UART_Transmit(uint8_t data) {
    // Wait for empty transmit buffer
    while (!(UCSR0A & (1 << UDRE0)));
    // Put data into buffer, sends the data
    UDR0 = data;
}

uint8_t UART_Receive(void) {
    // Wait for data to be received
    while (!(UCSR0A & (1 << RXC0)));
    // Get and return received data from buffer
    return UDR0;
}

void UART_Print(const char *str) {
    while (*str) {
        UART_Transmit(*str++);
    }
}
```

## Common Use Cases

### 1. Debugging and Logging
```cpp
// Real-time debugging output
Serial.print("Sensor value: ");
Serial.println(sensorValue);
Serial.print("Free RAM: ");
Serial.println(freeRam());
```

### 2. GPS Module Communication
```cpp
// Reading NMEA sentences from GPS
if (SerialGPS.available()) {
    String nmea = SerialGPS.readStringUntil('\n');
    if (nmea.startsWith("$GPGGA")) {
        parseGPS(nmea);
    }
}
```

### 3. Wireless Module (Bluetooth, WiFi)
```cpp
// AT command interface
SerialBT.println("AT+NAME=MyDevice");
delay(100);
String response = SerialBT.readString();
```

### 4. Sensor Communication
```cpp
// CO2 sensor command
Serial1.write(cmd, 9);
delay(100);
if (Serial1.available() >= 9) {
    Serial1.readBytes(response, 9);
}
```

### 5. PC Communication
```cpp
// Command protocol with PC
void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        switch(cmd) {
            case 'L': digitalWrite(LED, HIGH); break;
            case 'l': digitalWrite(LED, LOW); break;
            case 'T': Serial.println(readTemp()); break;
        }
    }
}
```

## UART vs Other Protocols

| Feature | UART | I2C | SPI |
|---------|------|-----|-----|
| **Wires** | 2 (+ GND) | 2 | 4+ |
| **Clock** | Asynchronous | Synchronous | Synchronous |
| **Devices** | 2 (point-to-point) | Many (multi-master) | 1 master, many slaves |
| **Speed** | Up to ~5 Mbps | Up to 3.4 Mbps | Up to 50+ MHz |
| **Distance** | Long (meters) | Short (< 1m) | Short (< 1m) |
| **Complexity** | Simple | Medium | Simple |
| **Error Detection** | Parity bit | ACK/NACK | None |

## Best Practices

### 1. Proper Baud Rate Calculation
```cpp
// Ensure both devices use exact same baud rate
// Check oscillator tolerance - should be < 2%

// For custom baud rates, verify with formula:
// UBRR = (F_CPU / (16 * BAUD)) - 1
```

### 2. Buffer Management
```cpp
// Check available space before reading
if (Serial.available() > 0) {
    int bytesToRead = Serial.available();
    for (int i = 0; i < bytesToRead; i++) {
        rxBuffer[i] = Serial.read();
    }
}

// Or use built-in methods
Serial.readBytes(rxBuffer, expectedSize);
```

### 3. Timeout Handling
```cpp
// Set appropriate timeout
Serial.setTimeout(500);  // 500ms

// Check for timeout
int value = Serial.parseInt();
if (value == 0 && Serial.peek() != '0') {
    // Timeout occurred
    Serial.println("Error: Timeout");
}
```

### 4. Flow Control (Hardware)
```
RTS (Request To Send) and CTS (Clear To Send)
Used for high-speed communications or when receiver
might not keep up with sender
```

### 5. Protocol Design
```cpp
// Add framing for reliable communication
// Example: <START>DATA<END>

void sendPacket(uint8_t *data, uint8_t len) {
    Serial.write(0x02);  // STX (Start of Text)
    for (int i = 0; i < len; i++) {
        Serial.write(data[i]);
    }
    uint8_t checksum = calculateChecksum(data, len);
    Serial.write(checksum);
    Serial.write(0x03);  // ETX (End of Text)
}
```

## Common Issues and Debugging

### Problem: Garbage Characters
**Causes**:
- Baud rate mismatch between devices
- Wrong oscillator frequency
- Noisy power supply

**Solutions**:
```cpp
// Try common baud rates systematically
Serial.begin(9600);   // Try this
Serial.begin(115200); // Then this

// Check your board's crystal frequency matches F_CPU
```

### Problem: Missing Characters
**Causes**:
- Buffer overflow (data arriving faster than processing)
- Insufficient interrupt priority

**Solutions**:
```cpp
// Increase serial buffer size (in HardwareSerial.cpp)
#define SERIAL_RX_BUFFER_SIZE 256

// Use hardware flow control
// Process data promptly in loop()
```

### Problem: First Character Lost
**Causes**:
- Receiver not initialized before transmitter sends
- Start bit detection issue

**Solutions**:
```cpp
// Add startup delay
void setup() {
    Serial.begin(9600);
    delay(100);  // Wait for UART to stabilize
}

// Send dummy byte first
Serial.write(0x00);
delay(10);
```

## Voltage Levels

### TTL UART (3.3V or 5V)
- **Logic HIGH**: 2.4V - 5V
- **Logic LOW**: 0V - 0.8V
- Most microcontrollers use this

### RS-232 UART (Legacy)
- **Logic HIGH (Space)**: -3V to -15V
- **Logic LOW (Mark)**: +3V to +15V
- Requires level shifter (MAX232, MAX3232)
- Longer cable runs possible

```cpp
// Using MAX232 level shifter
// MCU TX -> MAX232 T1IN -> MAX232 T1OUT -> PC RX
// MCU RX <- MAX232 R1OUT <- MAX232 R1IN <- PC TX
```

## ELI10 (Explain Like I'm 10)

Imagine you and your friend are in different rooms and want to talk using two cans connected by a string:

- **TX (Transmit)** is your mouth speaking into the can
- **RX (Receive)** is your ear listening from the can
- **Baud Rate** is how fast you talk - if one person talks super fast and the other listens slowly, you won't understand each other!
- **Start Bit** is like saying "Hey, listen!" before each word
- **Stop Bit** is like a pause after each word

The cool thing? Both of you can talk and listen at the same time because you have two strings (wires)!

The tricky part? You MUST both agree to talk at the same speed (baud rate) before starting, because there's no way to say "slow down!" once you've begun.

## Where this connects

- [SPI](spi.md) — synchronous, higher-speed multi-device alternative
- [I2C](i2c.md) — synchronous, multi-master two-wire alternative
- [CAN](can.md) — multi-node bus with built-in arbitration for automotive/industrial networks
- [GPIO](gpio.md) — UART TX/RX pins are configured via GPIO alternate-function registers
- [Interrupts](interrupts.md) — UART receive is almost always interrupt-driven or DMA-driven to avoid polling
- [Debugging](debugging.md) — UART is the primary debug output channel on most embedded targets

## Further Resources

- [UART Wikipedia](https://en.wikipedia.org/wiki/Universal_asynchronous_receiver-transmitter)
- [SparkFun Serial Communication Tutorial](https://learn.sparkfun.com/tutorials/serial-communication)
- [Arduino Serial Reference](https://www.arduino.cc/reference/en/language/functions/communication/serial/)
- [AN4666: STM32 UART Concepts](https://www.st.com/resource/en/application_note/an4666-stm32-uart-concepts-stmicroelectronics.pdf)
- [Baud Rate Calculator](http://wormfood.net/avrbaudcalc.php)
