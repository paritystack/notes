# Arduino Programming

Complete guide to Arduino development, from basics to advanced projects.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Arduino Language](#arduino-language)
4. [Digital I/O](#digital-io)
5. [Analog I/O](#analog-io)
6. [Serial Communication](#serial-communication)
7. [Libraries](#libraries)
8. [Common Projects](#common-projects)
9. [Advanced Topics](#advanced-topics)

## Introduction

Arduino is an open-source electronics platform based on easy-to-use hardware and software. It's designed for artists, designers, hobbyists, and anyone interested in creating interactive objects or environments.

### Arduino Boards Comparison

| Board | MCU | Clock | Flash | RAM | Digital I/O | Analog In | Price |
|-------|-----|-------|-------|-----|-------------|-----------|-------|
| **Uno** | ATmega328P | 16 MHz | 32 KB | 2 KB | 14 (6 PWM) | 6 | $ |
| **Mega 2560** | ATmega2560 | 16 MHz | 256 KB | 8 KB | 54 (15 PWM) | 16 | $$ |
| **Nano** | ATmega328P | 16 MHz | 32 KB | 2 KB | 14 (6 PWM) | 8 | $ |
| **Leonardo** | ATmega32u4 | 16 MHz | 32 KB | 2.5 KB | 20 (7 PWM) | 12 | $ |
| **Due** | AT91SAM3X8E | 84 MHz | 512 KB | 96 KB | 54 (12 PWM) | 12 | $$$ |
| **Nano 33 IoT** | SAMD21 | 48 MHz | 256 KB | 32 KB | 14 (11 PWM) | 8 | $$ |

### Arduino Uno Pinout

```
                   Arduino Uno
                ┌─────────────┐
                │   USB       │
                ├─────────────┤
    RESET  [ ]──┤ RESET    A0 ├──[ ] Analog Input
    3.3V   [ ]──┤ 3V3      A1 ├──[ ] Analog Input
    5V     [ ]──┤ 5V       A2 ├──[ ] Analog Input
    GND    [ ]──┤ GND      A3 ├──[ ] Analog Input
    GND    [ ]──┤ GND      A4 ├──[ ] Analog Input (I2C SDA)
    VIN    [ ]──┤ VIN      A5 ├──[ ] Analog Input (I2C SCL)
                │             │
    D0/RX  [ ]──┤ 0        13 ├──[ ] D13/SCK (LED_BUILTIN)
    D1/TX  [ ]──┤ 1        12 ├──[ ] D12/MISO
    D2     [ ]──┤ 2        11 ├──[ ] D11~/MOSI
    D3~    [ ]──┤ 3        10 ├──[ ] D10~
    D4     [ ]──┤ 4         9 ├──[ ] D9~
    D5~    [ ]──┤ 5         8 ├──[ ] D8
    D6~    [ ]──┤ 6         7 ├──[ ] D7
                └─────────────┘
    
    ~ = PWM capable
```

## Getting Started

### Installation

#### Arduino IDE
```bash
# Download from arduino.cc
# Or use package manager (Linux)
sudo apt install arduino

# Or use Arduino CLI
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
arduino-cli core update-index
arduino-cli core install arduino:avr
```

#### PlatformIO (Recommended for Advanced Users)
```bash
pip install platformio
platformio init --board uno
```

### Basic Program Structure

Every Arduino sketch has two required functions:

```cpp
void setup() {
    // Runs once when the board starts
    // Initialize pins, serial, libraries
}

void loop() {
    // Runs continuously after setup()
    // Main program logic goes here
}
```

### First Program: Blink LED

```cpp
// Blink the built-in LED
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);  // Set pin 13 as output
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);  // Turn LED on
    delay(1000);                       // Wait 1 second
    digitalWrite(LED_BUILTIN, LOW);   // Turn LED off
    delay(1000);                       // Wait 1 second
}
```

**Wiring:**
```
Arduino          Component
   13 ───────────┐
                 │
               ┌─┴─┐
               │LED│  Built-in LED
               └─┬─┘
                 │
   GND ──────────┘
```

## Arduino Language

### Data Types

```cpp
// Boolean
bool flag = true;

// Integers
byte value = 255;           // 0-255 (8-bit unsigned)
int temperature = -40;      // -32768 to 32767 (16-bit signed)
unsigned int count = 65535; // 0-65535 (16-bit unsigned)
long distance = 1000000L;   // 32-bit signed
unsigned long time = millis();  // 32-bit unsigned

// Floating Point
float voltage = 3.3;        // 32-bit, ~7 digits precision
double precise = 3.14159;   // Same as float on Arduino

// Characters and Strings
char letter = 'A';
char message[] = "Hello";   // C-style string
String text = "World";      // Arduino String class

// Arrays
int readings[10];           // Array of 10 integers
int values[] = {1, 2, 3};  // Initialized array
```

### Control Structures

```cpp
// If-else
if (temperature > 30) {
    digitalWrite(FAN_PIN, HIGH);
} else if (temperature > 20) {
    analogWrite(FAN_PIN, 128);
} else {
    digitalWrite(FAN_PIN, LOW);
}

// Switch-case
switch (state) {
    case 0:
        // Do something
        break;
    case 1:
        // Do something else
        break;
    default:
        // Default action
        break;
}

// For loop
for (int i = 0; i < 10; i++) {
    Serial.println(i);
}

// While loop
while (digitalRead(BUTTON_PIN) == HIGH) {
    // Wait for button release
}

// Do-while loop
do {
    value = analogRead(A0);
} while (value < 512);
```

### Functions

```cpp
// Function declaration
int addNumbers(int a, int b);

void setup() {
    Serial.begin(9600);
    int result = addNumbers(5, 3);
    Serial.println(result);  // Prints 8
}

// Function definition
int addNumbers(int a, int b) {
    return a + b;
}

// Function with default parameters
void blinkLED(int pin, int times = 1, int delayTime = 500) {
    for (int i = 0; i < times; i++) {
        digitalWrite(pin, HIGH);
        delay(delayTime);
        digitalWrite(pin, LOW);
        delay(delayTime);
    }
}

void loop() {
    blinkLED(13);           // Blink once
    blinkLED(13, 3);        // Blink 3 times
    blinkLED(13, 5, 200);   // Blink 5 times with 200ms delay
}
```

## Digital I/O

### Basic Digital Functions

```cpp
pinMode(pin, mode);        // Configure pin: INPUT, OUTPUT, INPUT_PULLUP
digitalWrite(pin, value);  // Write HIGH or LOW
int value = digitalRead(pin);  // Read HIGH or LOW
```

### LED Control

```cpp
// Simple LED control
const int LED_PIN = 9;

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
}
```

**Wiring:**
```
Arduino          Component
    9 ───────────┬─────────┐
                 │         │
               ┌─┴─┐     ┌─┴─┐
               │220│     │LED│
               │Ω  │     │ > │
               └─┬─┘     └─┬─┘
                 │         │
   GND ──────────┴─────────┘
```

### Button Input

```cpp
const int BUTTON_PIN = 2;
const int LED_PIN = 13;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // Internal pull-up resistor
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    int buttonState = digitalRead(BUTTON_PIN);
    
    if (buttonState == LOW) {  // Button pressed (active LOW)
        digitalWrite(LED_PIN, HIGH);
    } else {
        digitalWrite(LED_PIN, LOW);
    }
}
```

**Wiring:**
```
Arduino          Button
    2 ────────┬────┬──┬──── 5V (optional if using INPUT_PULLUP)
              │    │  │
            ┌─┴─┐ ┌┴──┴┐
            │10k│ │BTN │
            │Ω  │ └────┘
            └─┬─┘    │
              │      │
   GND ───────┴──────┘
```

### Debouncing

```cpp
const int BUTTON_PIN = 2;
const int LED_PIN = 13;
const int DEBOUNCE_DELAY = 50;  // milliseconds

int lastButtonState = HIGH;
int buttonState = HIGH;
unsigned long lastDebounceTime = 0;
bool ledState = false;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    int reading = digitalRead(BUTTON_PIN);
    
    // If the switch changed, due to noise or pressing
    if (reading != lastButtonState) {
        lastDebounceTime = millis();
    }
    
    if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
        // If the button state has changed
        if (reading != buttonState) {
            buttonState = reading;
            
            // Only toggle if the new button state is LOW (pressed)
            if (buttonState == LOW) {
                ledState = !ledState;
                digitalWrite(LED_PIN, ledState);
            }
        }
    }
    
    lastButtonState = reading;
}
```

## Analog I/O

### Analog Input (ADC)

```cpp
analogRead(pin);  // Read analog value (0-1023)
```

#### Reading a Potentiometer

```cpp
const int POT_PIN = A0;
const int LED_PIN = 9;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    int potValue = analogRead(POT_PIN);  // 0-1023
    
    // Convert to voltage (0-5V)
    float voltage = potValue * (5.0 / 1023.0);
    
    // Convert to LED brightness (0-255)
    int brightness = map(potValue, 0, 1023, 0, 255);
    
    Serial.print("Value: ");
    Serial.print(potValue);
    Serial.print(" Voltage: ");
    Serial.print(voltage);
    Serial.println("V");
    
    analogWrite(LED_PIN, brightness);
    delay(100);
}
```

**Wiring:**
```
Potentiometer       Arduino
     ┌────┐
  5V─┤1  3├─GND
     │    │
     │ 2  ├─A0 (wiper)
     └────┘
```

### Analog Output (PWM)

```cpp
analogWrite(pin, value);  // PWM output (0-255)
```

#### Fading LED

```cpp
const int LED_PIN = 9;  // Must be PWM pin (~)

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Fade in
    for (int brightness = 0; brightness <= 255; brightness++) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }
    
    // Fade out
    for (int brightness = 255; brightness >= 0; brightness--) {
        analogWrite(LED_PIN, brightness);
        delay(10);
    }
}
```

### Temperature Sensor (LM35)

```cpp
const int TEMP_PIN = A0;

void setup() {
    Serial.begin(9600);
}

void loop() {
    int reading = analogRead(TEMP_PIN);
    
    // Convert to voltage (0-5V)
    float voltage = reading * (5.0 / 1023.0);
    
    // Convert to temperature (LM35: 10mV per degree C)
    float temperatureC = voltage * 100.0;
    float temperatureF = (temperatureC * 9.0 / 5.0) + 32.0;
    
    Serial.print("Temperature: ");
    Serial.print(temperatureC);
    Serial.print("°C / ");
    Serial.print(temperatureF);
    Serial.println("°F");
    
    delay(1000);
}
```

**Wiring:**
```
LM35 Sensor         Arduino
    ┌────┐
 1 ─┤ VS ├─ 5V
    │    │
 2 ─┤Vout├─ A0
    │    │
 3 ─┤GND ├─ GND
    └────┘
```

## Serial Communication

### Basic Serial Functions

```cpp
Serial.begin(baudrate);        // Initialize serial (9600, 115200, etc.)
Serial.print(data);            // Print without newline
Serial.println(data);          // Print with newline
Serial.write(byte);            // Send raw byte
int available = Serial.available();  // Bytes available to read
char c = Serial.read();        // Read one byte
String line = Serial.readStringUntil('\n');  // Read until newline
```

### Serial Monitor Output

```cpp
void setup() {
    Serial.begin(9600);
    Serial.println("Arduino Ready!");
}

void loop() {
    int sensorValue = analogRead(A0);
    
    // Different formatting options
    Serial.print("Sensor: ");
    Serial.println(sensorValue);
    
    Serial.print("Hex: 0x");
    Serial.println(sensorValue, HEX);
    
    Serial.print("Binary: 0b");
    Serial.println(sensorValue, BIN);
    
    Serial.print("Float: ");
    float voltage = sensorValue * (5.0 / 1023.0);
    Serial.println(voltage, 2);  // 2 decimal places
    
    delay(1000);
}
```

### Serial Input

```cpp
String inputString = "";
bool stringComplete = false;

void setup() {
    Serial.begin(9600);
    inputString.reserve(200);  // Reserve space for efficiency
}

void loop() {
    // Check if data is available
    while (Serial.available()) {
        char inChar = (char)Serial.read();
        inputString += inChar;
        
        if (inChar == '\n') {
            stringComplete = true;
        }
    }
    
    // Process complete command
    if (stringComplete) {
        Serial.print("Received: ");
        Serial.println(inputString);
        
        // Process command
        if (inputString.startsWith("LED ON")) {
            digitalWrite(LED_BUILTIN, HIGH);
            Serial.println("LED turned ON");
        } else if (inputString.startsWith("LED OFF")) {
            digitalWrite(LED_BUILTIN, LOW);
            Serial.println("LED turned OFF");
        }
        
        // Clear the string
        inputString = "";
        stringComplete = false;
    }
}
```

## Libraries

### Built-in Libraries

#### Wire (I2C)

```cpp
#include <Wire.h>

void setup() {
    Wire.begin();  // Join I2C bus as master
}

void loop() {
    // Read from I2C device at address 0x68
    Wire.beginTransmission(0x68);
    Wire.write(0x00);  // Register address
    Wire.endTransmission();
    
    Wire.requestFrom(0x68, 1);  // Request 1 byte
    if (Wire.available()) {
        byte data = Wire.read();
    }
}
```

#### SPI

```cpp
#include <SPI.h>

const int CS_PIN = 10;

void setup() {
    SPI.begin();
    pinMode(CS_PIN, OUTPUT);
    digitalWrite(CS_PIN, HIGH);
}

void loop() {
    digitalWrite(CS_PIN, LOW);   // Select device
    SPI.transfer(0xAB);          // Send byte
    byte received = SPI.transfer(0x00);  // Receive byte
    digitalWrite(CS_PIN, HIGH);  // Deselect device
}
```

#### EEPROM

```cpp
#include <EEPROM.h>

void setup() {
    // Write byte to EEPROM
    EEPROM.write(0, 42);
    
    // Read byte from EEPROM
    byte value = EEPROM.read(0);
    
    // Update (only writes if different)
    EEPROM.update(0, 42);
    
    // Write/read other types
    int address = 0;
    float f = 3.14;
    EEPROM.put(address, f);
    EEPROM.get(address, f);
}
```

### Popular External Libraries

#### Servo Control

```cpp
#include <Servo.h>

Servo myServo;
const int SERVO_PIN = 9;

void setup() {
    myServo.attach(SERVO_PIN);
}

void loop() {
    // Sweep from 0 to 180 degrees
    for (int pos = 0; pos <= 180; pos++) {
        myServo.write(pos);
        delay(15);
    }
    
    // Sweep back
    for (int pos = 180; pos >= 0; pos--) {
        myServo.write(pos);
        delay(15);
    }
}
```

**Wiring:**
```
Servo Motor         Arduino
    ┌────┐
 R ─┤Red ├─ 5V (or external)
 B ─┤Brn ├─ GND
 O ─┤Org ├─ Pin 9 (PWM)
    └────┘
```

#### LiquidCrystal (LCD Display)

```cpp
#include <LiquidCrystal.h>

// RS, E, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
    lcd.begin(16, 2);  // 16x2 LCD
    lcd.print("Hello, World!");
}

void loop() {
    lcd.setCursor(0, 1);  // Column 0, Row 1
    lcd.print(millis() / 1000);
    lcd.print("s");
    delay(100);
}
```

**Wiring:**
```
LCD 16x2            Arduino
VSS  ──────────────  GND
VDD  ──────────────  5V
V0   ──────────────  Potentiometer (contrast)
RS   ──────────────  12
RW   ──────────────  GND
E    ──────────────  11
D4   ──────────────  5
D5   ──────────────  4
D6   ──────────────  3
D7   ──────────────  2
A    ──────────────  5V (backlight)
K    ──────────────  GND
```

#### DHT Temperature/Humidity Sensor

```cpp
#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11  // or DHT22

DHT dht(DHTPIN, DHTTYPE);

void setup() {
    Serial.begin(9600);
    dht.begin();
}

void loop() {
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();  // Celsius
    float temperatureF = dht.readTemperature(true);  // Fahrenheit
    
    if (isnan(humidity) || isnan(temperature)) {
        Serial.println("Failed to read from DHT sensor!");
        return;
    }
    
    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.print("% Temperature: ");
    Serial.print(temperature);
    Serial.println("°C");
    
    delay(2000);  // DHT11 minimum sampling period
}
```

## Common Projects

### Project 1: Traffic Light

```cpp
const int RED_LED = 10;
const int YELLOW_LED = 9;
const int GREEN_LED = 8;

void setup() {
    pinMode(RED_LED, OUTPUT);
    pinMode(YELLOW_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
}

void loop() {
    // Green light
    digitalWrite(GREEN_LED, HIGH);
    delay(5000);  // 5 seconds
    digitalWrite(GREEN_LED, LOW);
    
    // Yellow light
    digitalWrite(YELLOW_LED, HIGH);
    delay(2000);  // 2 seconds
    digitalWrite(YELLOW_LED, LOW);
    
    // Red light
    digitalWrite(RED_LED, HIGH);
    delay(5000);  // 5 seconds
    digitalWrite(RED_LED, LOW);
}
```

**Wiring:**
```
Arduino              LEDs
  10 ───┬───[220Ω]───[RED LED]───GND
        │
   9 ───┼───[220Ω]───[YEL LED]───GND
        │
   8 ───┴───[220Ω]───[GRN LED]───GND
```

### Project 2: Ultrasonic Distance Sensor

```cpp
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;

void setup() {
    Serial.begin(9600);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
}

void loop() {
    // Send ultrasonic pulse
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    
    // Measure echo duration
    long duration = pulseIn(ECHO_PIN, HIGH);
    
    // Calculate distance in cm
    // Speed of sound: 343 m/s = 0.0343 cm/µs
    // Distance = (duration / 2) * 0.0343
    float distance = duration * 0.0343 / 2;
    
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");
    
    delay(100);
}
```

**Wiring:**
```
HC-SR04             Arduino
 VCC ───────────────  5V
 Trig ──────────────  9
 Echo ──────────────  10
 GND ───────────────  GND
```

### Project 3: Light-Activated Switch

```cpp
const int LDR_PIN = A0;
const int LED_PIN = 13;
const int THRESHOLD = 500;  // Adjust based on lighting

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    int lightLevel = analogRead(LDR_PIN);
    
    Serial.print("Light Level: ");
    Serial.println(lightLevel);
    
    if (lightLevel < THRESHOLD) {
        digitalWrite(LED_PIN, HIGH);  // Turn on LED when dark
    } else {
        digitalWrite(LED_PIN, LOW);   // Turn off LED when bright
    }
    
    delay(100);
}
```

**Wiring:**
```
         5V
          │
        ┌─┴─┐
        │LDR│ (Light Dependent Resistor)
        └─┬─┘
          ├────── A0
        ┌─┴─┐
        │10k│ (Pull-down resistor)
        │Ω  │
        └─┬─┘
          │
         GND
```

### Project 4: Temperature-Controlled Fan

```cpp
#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define FAN_PIN 9  // PWM pin for fan control

DHT dht(DHTPIN, DHTTYPE);

const float TEMP_MIN = 25.0;  // Start fan at 25°C
const float TEMP_MAX = 35.0;  // Full speed at 35°C

void setup() {
    Serial.begin(9600);
    pinMode(FAN_PIN, OUTPUT);
    dht.begin();
}

void loop() {
    float temperature = dht.readTemperature();
    
    if (isnan(temperature)) {
        Serial.println("Failed to read temperature!");
        return;
    }
    
    // Calculate fan speed (0-255)
    int fanSpeed = 0;
    if (temperature < TEMP_MIN) {
        fanSpeed = 0;
    } else if (temperature > TEMP_MAX) {
        fanSpeed = 255;
    } else {
        fanSpeed = map(temperature * 10, TEMP_MIN * 10, TEMP_MAX * 10, 0, 255);
    }
    
    analogWrite(FAN_PIN, fanSpeed);
    
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.print("°C Fan Speed: ");
    Serial.print((fanSpeed * 100) / 255);
    Serial.println("%");
    
    delay(2000);
}
```

### Project 5: Simple Data Logger

```cpp
#include <SD.h>
#include <SPI.h>

const int CS_PIN = 10;
const int SENSOR_PIN = A0;

File dataFile;

void setup() {
    Serial.begin(9600);
    
    // Initialize SD card
    if (!SD.begin(CS_PIN)) {
        Serial.println("SD card initialization failed!");
        return;
    }
    Serial.println("SD card initialized.");
}

void loop() {
    int sensorValue = analogRead(SENSOR_PIN);
    float voltage = sensorValue * (5.0 / 1023.0);
    
    // Open file for writing
    dataFile = SD.open("datalog.txt", FILE_WRITE);
    
    if (dataFile) {
        // Write timestamp and value
        dataFile.print(millis());
        dataFile.print(",");
        dataFile.println(voltage);
        dataFile.close();
        
        Serial.print("Logged: ");
        Serial.println(voltage);
    } else {
        Serial.println("Error opening file!");
    }
    
    delay(1000);  // Log every second
}
```

## Advanced Topics

### Interrupts

```cpp
const int BUTTON_PIN = 2;  // Must be interrupt-capable pin
const int LED_PIN = 13;

volatile bool ledState = false;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    
    // Attach interrupt: pin, ISR function, trigger mode
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);
}

void loop() {
    // Main loop can do other things
    // LED toggle happens immediately when button pressed
}

// Interrupt Service Routine (keep it short!)
void buttonISR() {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
}
```

### Timers

```cpp
unsigned long previousMillis = 0;
const long interval = 1000;  // 1 second

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    unsigned long currentMillis = millis();
    
    // Non-blocking timing
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;
        
        // Toggle LED
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    }
    
    // Can do other things here
}
```

### Memory Optimization

```cpp
// Store strings in flash memory (PROGMEM)
const char message[] PROGMEM = "This string is stored in flash";

void setup() {
    Serial.begin(9600);
    
    // Read from flash memory
    char buffer[50];
    strcpy_P(buffer, message);
    Serial.println(buffer);
}

// Use F() macro for Serial.print
void loop() {
    Serial.println(F("This uses flash memory, not RAM"));
    delay(1000);
}
```

### Low Power Mode

```cpp
#include <avr/sleep.h>
#include <avr/power.h>

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(2, INPUT_PULLUP);
    
    // Enable interrupt for wake-up
    attachInterrupt(digitalPinToInterrupt(2), wakeUp, LOW);
}

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    
    // Enter sleep mode
    enterSleep();
}

void enterSleep() {
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    sleep_enable();
    
    // Disable peripherals
    power_adc_disable();
    power_spi_disable();
    power_timer0_disable();
    power_timer1_disable();
    power_timer2_disable();
    power_twi_disable();
    
    sleep_mode();  // Sleep here
    
    // Wake up here
    sleep_disable();
    power_all_enable();
}

void wakeUp() {
    // ISR to wake up
}
```

## Best Practices

### 1. Avoid delay() for Responsive Code

```cpp
// Bad: Blocking
void loop() {
    digitalWrite(LED1, HIGH);
    delay(1000);
    digitalWrite(LED2, HIGH);
    delay(500);
}

// Good: Non-blocking
unsigned long led1Time = 0;
unsigned long led2Time = 0;

void loop() {
    unsigned long now = millis();
    
    if (now - led1Time >= 1000) {
        digitalWrite(LED1, !digitalRead(LED1));
        led1Time = now;
    }
    
    if (now - led2Time >= 500) {
        digitalWrite(LED2, !digitalRead(LED2));
        led2Time = now;
    }
}
```

### 2. Use const for Pin Definitions

```cpp
// Good: Easy to change and read
const int LED_PIN = 13;
const int BUTTON_PIN = 2;
const int SENSOR_PIN = A0;
```

### 3. Check Return Values

```cpp
if (!SD.begin(CS_PIN)) {
    Serial.println("SD card failed!");
    while (1);  // Halt
}
```

### 4. Use Meaningful Variable Names

```cpp
// Bad
int x = analogRead(A0);

// Good
int lightLevel = analogRead(LIGHT_SENSOR_PIN);
```

### 5. Comment Complex Logic

```cpp
// Calculate distance from ultrasonic sensor
// Formula: distance (cm) = duration (µs) × 0.0343 / 2
// Division by 2 accounts for round-trip time
float distance = duration * 0.0343 / 2;
```

## Troubleshooting

### Common Issues

1. **Upload Failed**
   - Check correct board and port selected
   - Try pressing reset button before upload
   - Close Serial Monitor during upload

2. **Serial Monitor Shows Garbage**
   - Check baud rate matches code
   - Verify USB cable supports data (not just power)

3. **Sketch Too Large**
   - Remove unused libraries
   - Use PROGMEM for strings
   - Optimize code

4. **Unexpected Behavior**
   - Add Serial.println() for debugging
   - Check wiring and connections
   - Verify power supply adequate

## Resources

- **Official Documentation**: https://www.arduino.cc/reference/
- **Forum**: https://forum.arduino.cc/
- **Project Hub**: https://create.arduino.cc/projecthub
- **Libraries**: https://www.arduinolibraries.info/

## See Also

- [ESP32](esp32.md) - More powerful Arduino-compatible platform
- [AVR Programming](avr.md) - Low-level AVR microcontroller programming
- [GPIO](gpio.md) - Digital I/O concepts
- [UART](uart.md) - Serial communication details
- [SPI](spi.md) - SPI protocol
- [I2C](i2c.md) - I2C protocol
