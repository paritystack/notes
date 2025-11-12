# PWM (Pulse Width Modulation)

## Overview

Pulse Width Modulation (PWM) is a technique for controlling power delivery to electrical devices by rapidly switching between ON and OFF states. By varying the ratio of ON time to OFF time (duty cycle), you can control the average power delivered without actually changing the voltage level. This makes PWM highly efficient and versatile for applications ranging from LED dimming to motor control.

## Key Concepts

### Duty Cycle

The **duty cycle** is the percentage of time the signal is HIGH during one complete cycle.

```
Duty Cycle (%) = (Ton / (Ton + Toff)) × 100

Where:
- Ton = Time the signal is HIGH
- Toff = Time the signal is LOW
- Period = Ton + Toff
```

**Examples**:
- **0% duty cycle**: Always LOW (0V average)
- **25% duty cycle**: HIGH for 1/4 of the period
- **50% duty cycle**: HIGH for half the period
- **75% duty cycle**: HIGH for 3/4 of the period
- **100% duty cycle**: Always HIGH (full voltage)

### Frequency

The **frequency** determines how many ON/OFF cycles occur per second, measured in Hertz (Hz).

```
Frequency = 1 / Period
Period = 1 / Frequency
```

**Typical Frequencies**:
- **LED Dimming**: 500 Hz - 20 kHz (above flicker perception ~60 Hz)
- **Motor Control**: 1 kHz - 40 kHz
- **Audio**: 40 kHz+ (above human hearing)
- **Servo Motors**: 50 Hz (20ms period)

### Average Voltage

The average voltage delivered by PWM:

```
Average Voltage = Supply Voltage × (Duty Cycle / 100)

Example (5V supply):
- 0% duty → 0V average
- 25% duty → 1.25V average
- 50% duty → 2.5V average
- 100% duty → 5V average
```

### Visual Representation

```
100% Duty Cycle (Always ON):
█████████████████████████████

75% Duty Cycle:
██████████████████░░░░░░░

50% Duty Cycle:
█████████████░░░░░░░░░░░░

25% Duty Cycle:
██████░░░░░░░░░░░░░░░░░░░

0% Duty Cycle (Always OFF):
░░░░░░░░░░░░░░░░░░░░░░░░░
```

## How It Works

### Hardware PWM vs Software PWM

| Feature | Hardware PWM | Software PWM |
|---------|-------------|--------------|
| **Precision** | Very precise, timer-based | Can jitter with interrupts |
| **CPU Load** | Zero (handled by hardware) | High (CPU must toggle pin) |
| **Pins** | Limited (specific pins only) | Any digital pin |
| **Frequency** | High (up to MHz) | Low (few kHz max) |
| **Recommended** | Motors, audio, servos | Simple LED control |

### PWM Resolution

Resolution is the number of distinct duty cycle levels available:

| Resolution | Levels | Step Size (at 5V) |
|-----------|--------|-------------------|
| 8-bit | 256 | 19.5 mV |
| 10-bit | 1024 | 4.88 mV |
| 12-bit | 4096 | 1.22 mV |
| 16-bit | 65536 | 76 μV |

**Note**: Higher resolution requires lower maximum frequency:
```
Max Frequency = Clock Frequency / (2^Resolution)
```

## Code Examples

### Arduino PWM (Hardware)

```cpp
// Arduino Uno PWM pins: 3, 5, 6, 9, 10, 11
// Default frequency: ~490 Hz (pins 3,9,10,11) and ~980 Hz (pins 5,6)

const int ledPin = 9;
const int motorPin = 10;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(motorPin, OUTPUT);
}

void loop() {
  // analogWrite uses 8-bit resolution (0-255)

  // LED at 25% brightness
  analogWrite(ledPin, 64);  // 64/255 = 25%
  delay(1000);

  // LED at 50% brightness
  analogWrite(ledPin, 128);  // 128/255 = 50%
  delay(1000);

  // LED at 75% brightness
  analogWrite(ledPin, 192);  // 192/255 = 75%
  delay(1000);

  // LED at 100% brightness
  analogWrite(ledPin, 255);  // 255/255 = 100%
  delay(1000);
}

// Smooth fade effect
void fadeLED() {
  // Fade in
  for (int brightness = 0; brightness <= 255; brightness++) {
    analogWrite(ledPin, brightness);
    delay(5);
  }

  // Fade out
  for (int brightness = 255; brightness >= 0; brightness--) {
    analogWrite(ledPin, brightness);
    delay(5);
  }
}

// Change PWM frequency (Arduino Uno)
void setPWMFrequency(int pin, int divisor) {
  byte mode;

  if (pin == 5 || pin == 6 || pin == 9 || pin == 10) {
    switch(divisor) {
      case 1: mode = 0x01; break;    // 31.25 kHz
      case 8: mode = 0x02; break;    // 3.9 kHz
      case 64: mode = 0x03; break;   // 490 Hz (default for 9,10)
      case 256: mode = 0x04; break;  // 122 Hz
      case 1024: mode = 0x05; break; // 30 Hz
      default: return;
    }

    if (pin == 5 || pin == 6) {
      TCCR0B = (TCCR0B & 0b11111000) | mode;
    } else {
      TCCR1B = (TCCR1B & 0b11111000) | mode;
    }
  }
}

void setup() {
  pinMode(9, OUTPUT);
  setPWMFrequency(9, 1);  // Set pin 9 to 31.25 kHz
}
```

### ESP32 PWM (LEDC)

```cpp
// ESP32 uses LEDC (LED Control) for PWM
// 16 independent channels, configurable frequency and resolution

const int ledPin = 25;
const int pwmChannel = 0;      // Channel 0-15
const int pwmFrequency = 5000; // 5 kHz
const int pwmResolution = 8;   // 8-bit (0-255)

void setup() {
  // Configure PWM channel
  ledcSetup(pwmChannel, pwmFrequency, pwmResolution);

  // Attach pin to PWM channel
  ledcAttachPin(ledPin, pwmChannel);
}

void loop() {
  // Set duty cycle (0-255 for 8-bit)
  ledcWrite(pwmChannel, 128);  // 50% duty cycle
  delay(1000);

  ledcWrite(pwmChannel, 64);   // 25% duty cycle
  delay(1000);
}

// High resolution PWM (16-bit)
void setupHighResPWM() {
  const int pwmChannel = 0;
  const int pwmFreq = 1000;    // Lower freq for higher resolution
  const int pwmRes = 16;       // 16-bit (0-65535)

  ledcSetup(pwmChannel, pwmFreq, pwmRes);
  ledcAttachPin(ledPin, pwmChannel);

  // Set to 50% with 16-bit precision
  ledcWrite(pwmChannel, 32768);
}

// Multiple PWM channels for RGB LED
const int redPin = 25;
const int greenPin = 26;
const int bluePin = 27;

void setupRGB() {
  ledcSetup(0, 5000, 8);  // Red channel
  ledcSetup(1, 5000, 8);  // Green channel
  ledcSetup(2, 5000, 8);  // Blue channel

  ledcAttachPin(redPin, 0);
  ledcAttachPin(greenPin, 1);
  ledcAttachPin(bluePin, 2);
}

void setRGBColor(uint8_t r, uint8_t g, uint8_t b) {
  ledcWrite(0, r);
  ledcWrite(1, g);
  ledcWrite(2, b);
}

void loop() {
  setRGBColor(255, 0, 0);    // Red
  delay(1000);
  setRGBColor(0, 255, 0);    // Green
  delay(1000);
  setRGBColor(0, 0, 255);    // Blue
  delay(1000);
  setRGBColor(255, 255, 0);  // Yellow
  delay(1000);
}
```

### STM32 HAL PWM

```c
#include "stm32f4xx_hal.h"

TIM_HandleTypeDef htim3;

void PWM_Init(void) {
    TIM_OC_InitTypeDef sConfigOC = {0};

    // Timer 3 configuration for PWM
    htim3.Instance = TIM3;
    htim3.Init.Prescaler = 84 - 1;       // 84 MHz / 84 = 1 MHz
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = 1000 - 1;        // 1 MHz / 1000 = 1 kHz PWM
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    HAL_TIM_PWM_Init(&htim3);

    // Configure PWM channel 1
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 500;               // 50% duty cycle
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
    HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1);

    // Start PWM
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
}

void PWM_SetDutyCycle(uint16_t dutyCycle) {
    // dutyCycle: 0-1000 (0-100%)
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, dutyCycle);
}

void PWM_SetPercent(uint8_t percent) {
    // percent: 0-100
    uint16_t pulse = (percent * 1000) / 100;
    PWM_SetDutyCycle(pulse);
}
```

### Servo Control with PWM

```cpp
// Standard servo: 50 Hz (20ms period)
// Pulse width: 1ms (0°) to 2ms (180°)

const int servoPin = 9;

void setup() {
  pinMode(servoPin, OUTPUT);
}

void setServoAngle(int angle) {
  // Map angle (0-180) to pulse width (1000-2000 μs)
  int pulseWidth = map(angle, 0, 180, 1000, 2000);

  // Generate 50 Hz PWM signal
  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulseWidth);
  digitalWrite(servoPin, LOW);
  delayMicroseconds(20000 - pulseWidth);  // Complete 20ms period
}

void loop() {
  setServoAngle(0);      // 0 degrees
  delay(1000);
  setServoAngle(90);     // 90 degrees
  delay(1000);
  setServoAngle(180);    // 180 degrees
  delay(1000);
}

// Using Servo library (easier)
#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(9);     // Attach servo to pin 9
}

void loop() {
  myServo.write(0);      // 0 degrees
  delay(1000);
  myServo.write(90);     // 90 degrees
  delay(1000);
  myServo.write(180);    // 180 degrees
  delay(1000);
}
```

### Motor Control (H-Bridge)

```cpp
// Control DC motor speed and direction with L298N H-Bridge

const int motorPWM = 9;    // Speed control (PWM)
const int motorIN1 = 7;    // Direction control
const int motorIN2 = 8;    // Direction control

void setup() {
  pinMode(motorPWM, OUTPUT);
  pinMode(motorIN1, OUTPUT);
  pinMode(motorIN2, OUTPUT);
}

void setMotorSpeed(int speed) {
  // speed: -255 (full reverse) to +255 (full forward)

  if (speed > 0) {
    // Forward
    digitalWrite(motorIN1, HIGH);
    digitalWrite(motorIN2, LOW);
    analogWrite(motorPWM, speed);
  } else if (speed < 0) {
    // Reverse
    digitalWrite(motorIN1, LOW);
    digitalWrite(motorIN2, HIGH);
    analogWrite(motorPWM, -speed);
  } else {
    // Stop
    digitalWrite(motorIN1, LOW);
    digitalWrite(motorIN2, LOW);
    analogWrite(motorPWM, 0);
  }
}

void loop() {
  setMotorSpeed(128);    // 50% forward
  delay(2000);
  setMotorSpeed(255);    // 100% forward
  delay(2000);
  setMotorSpeed(0);      // Stop
  delay(1000);
  setMotorSpeed(-128);   // 50% reverse
  delay(2000);
}
```

## Common Applications

### 1. LED Dimming

```cpp
// Smooth breathing effect
void breathingLED(int pin) {
  const int maxBrightness = 255;
  const int minBrightness = 0;
  const int step = 5;
  const int delayTime = 30;

  // Breathe in
  for (int brightness = minBrightness; brightness <= maxBrightness; brightness += step) {
    analogWrite(pin, brightness);
    delay(delayTime);
  }

  // Breathe out
  for (int brightness = maxBrightness; brightness >= minBrightness; brightness -= step) {
    analogWrite(pin, brightness);
    delay(delayTime);
  }
}
```

### 2. RGB Color Mixing

```cpp
void setColorHSV(float h, float s, float v) {
  // Convert HSV to RGB
  float c = v * s;
  float x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
  float m = v - c;

  float r, g, b;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }

  analogWrite(redPin, (r + m) * 255);
  analogWrite(greenPin, (g + m) * 255);
  analogWrite(bluePin, (b + m) * 255);
}

// Rainbow effect
void rainbow() {
  for (int hue = 0; hue < 360; hue++) {
    setColorHSV(hue, 1.0, 1.0);
    delay(10);
  }
}
```

### 3. Speaker/Buzzer Tone Generation

```cpp
void playTone(int pin, int frequency, int duration) {
  int period = 1000000 / frequency;  // Period in microseconds
  int halfPeriod = period / 2;

  long cycles = ((long)frequency * duration) / 1000;

  for (long i = 0; i < cycles; i++) {
    digitalWrite(pin, HIGH);
    delayMicroseconds(halfPeriod);
    digitalWrite(pin, LOW);
    delayMicroseconds(halfPeriod);
  }
}

void playMelody() {
  playTone(buzzerPin, 262, 500);  // C4
  playTone(buzzerPin, 294, 500);  // D4
  playTone(buzzerPin, 330, 500);  // E4
  playTone(buzzerPin, 349, 500);  // F4
}

// Using tone() function (easier)
void playNote(int frequency) {
  tone(buzzerPin, frequency);
  delay(500);
  noTone(buzzerPin);
}
```

### 4. Fan Speed Control

```cpp
int targetTemp = 25;    // Target temperature
int currentTemp = 30;   // Read from sensor

void controlFan() {
  int tempDiff = currentTemp - targetTemp;

  int fanSpeed;
  if (tempDiff <= 0) {
    fanSpeed = 0;  // Too cold, fan off
  } else if (tempDiff >= 10) {
    fanSpeed = 255;  // Very hot, max speed
  } else {
    // Proportional control
    fanSpeed = map(tempDiff, 0, 10, 50, 255);
  }

  analogWrite(fanPin, fanSpeed);
}
```

### 5. Power Supply (Buck Converter)

PWM is used in switching power supplies to efficiently convert voltage:
- High frequency (20-100 kHz) minimizes inductor size
- Duty cycle controls output voltage
- Feedback loop maintains regulation

## Best Practices

### 1. Choose Appropriate Frequency

```cpp
// LED dimming: Use higher frequency to avoid flicker
// Human eye perceives flicker below ~60 Hz
setPWMFrequency(ledPin, 1);  // 31 kHz - no visible flicker

// Motor control: Balance between smoothness and efficiency
// Too high: Increased switching losses
// Too low: Audible noise, torque ripple
// Optimal: 10-25 kHz

// Audio: Must be above hearing range
// Humans hear up to ~20 kHz
// Use 40+ kHz for audio PWM
```

### 2. Filter PWM for Analog Output

```
PWM Pin ─── R ───┬─── Analog Output
                 │
                 C
                 │
                GND

Cutoff Frequency = 1 / (2π × R × C)

Example: R=1kΩ, C=10μF
fc = 1 / (2π × 1000 × 0.00001) ≈ 16 Hz
```

### 3. Protect Inductive Loads

```cpp
// Motors and solenoids are inductive
// Add flyback diode across load!

//        Motor
//    ┌────┴────┐
// PWM│         │GND
//    │   ▼─    │
//    └─────────┘
//    Flyback Diode
```

### 4. Avoid PWM on Critical Pins

```cpp
// Some Arduino pins share timers
// Changing frequency on one affects others!

// Pins 5 & 6 share Timer 0 (also used by millis/delay!)
// Pins 9 & 10 share Timer 1
// Pins 3 & 11 share Timer 2

// Changing Timer 0 frequency breaks millis() and delay()!
```

## Common Issues and Debugging

### Problem: LED Flickering
**Causes**: PWM frequency too low
**Solution**: Increase frequency above 60 Hz (ideally 500 Hz+)

### Problem: Motor Whining/Buzzing
**Causes**: PWM frequency in audible range
**Solution**: Increase frequency to 20+ kHz

### Problem: Servo Jittering
**Causes**: Incorrect pulse width or timing
**Solution**: Use dedicated Servo library, ensure 50 Hz signal

### Problem: PWM Not Working After Changing Frequency
**Causes**: Modified Timer 0 which breaks delay() and millis()
**Solution**: Use different timer, or use external library

## ELI10 (Explain Like I'm 10)

Imagine you have a light switch that you can flick on and off really, really fast - so fast that your eyes can't see it blinking!

**PWM is like that super-fast blinking:**
- If the light is ON for half the time and OFF for half the time, it looks 50% bright
- If it's ON for most of the time and OFF for a tiny bit, it looks almost fully bright
- If it's ON for only a tiny bit and OFF most of the time, it looks dim

**This works because**:
- Your eyes can't see things blinking faster than about 60 times per second
- So when we blink the light 500 or 1000 times per second, your brain sees a steady dimmed light!

**The cool part?**
- We're not actually reducing the voltage (which wastes energy as heat)
- We're just turning it on and off really fast (very efficient!)
- It's like running at full speed for short bursts vs. walking slowly all the time

**Duty cycle** is the percentage of time it's ON:
- 100% = always on (full brightness)
- 50% = on half the time (half brightness)
- 0% = always off (no light)

We use this same trick for controlling motor speeds, speakers, and lots of other things!

## Further Resources

- [Arduino PWM Guide](https://docs.arduino.cc/learn/microcontrollers/analog-output)
- [Secrets of Arduino PWM](https://www.arduino.cc/en/Tutorial/SecretsOfArduinoPWM)
- [ESP32 LEDC Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/ledc.html)
- [PWM Wikipedia](https://en.wikipedia.org/wiki/Pulse-width_modulation)
- [Motor Control with PWM](https://www.ti.com/lit/an/slva505/slva505.pdf)
