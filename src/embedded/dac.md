# DAC (Digital-to-Analog Converter)

## Overview

A Digital-to-Analog Converter (DAC) does the opposite of an [ADC](adc.md) — it converts discrete digital values from a microcontroller into continuous analog voltage signals. DACs are essential for generating analog outputs like audio signals, control voltages, and waveforms. When a true DAC is unavailable, [PWM](pwm.md) with an RC filter is a common software substitute. External DAC chips (e.g., MCP4725) connect via [I2C](i2c.md), and DMA-driven [Timers](timers.md) enable waveform generation without CPU involvement.

## Key Concepts

### What Does a DAC Do?

A DAC takes a digital number and outputs a corresponding analog voltage:

```
Digital Input (512) -> DAC -> Analog Output (2.5V)

For 10-bit DAC with 5V reference:
Voltage = (512 / 1023) * 5V = 2.5V
```

### Resolution

Just like ADCs, DAC resolution determines output precision:

| Bits | Levels | Voltage Step (5V) | Voltage Step (3.3V) |
|------|--------|-------------------|---------------------|
| 8-bit | 256 | 19.6 mV | 12.9 mV |
| 10-bit | 1024 | 4.88 mV | 3.22 mV |
| 12-bit | 4096 | 1.22 mV | 0.81 mV |
| 16-bit | 65536 | 76.3 uV | 50.4 uV |

### DAC vs PWM

Many microcontrollers don't have true DAC outputs, but can simulate analog using PWM:

| Feature | True DAC | PWM |
|---------|----------|-----|
| **Output** | True analog voltage | Digital pulses |
| **Smoothness** | Smooth DC voltage | Requires filtering |
| **Speed** | Fast settling | Limited by PWM frequency |
| **Filtering** | Not needed | Low-pass filter needed |
| **Complexity** | Hardware DAC required | Any digital pin |
| **Use Cases** | Audio, precise control | LED dimming, motor speed |

## How It Works

### Conversion Formula

```
Output Voltage = (Digital Value / (2^n - 1)) * Reference Voltage

Where:
- n = number of bits
- Digital Value = input code (0 to 2^n - 1)
- Reference Voltage = max output voltage
```

### Common DAC Architectures

1. **R-2R Ladder**: Uses resistor network (simple, cheap)
2. **Binary Weighted**: Uses weighted current sources
3. **Delta-Sigma**: High resolution, used in audio
4. **String**: Resistor divider network

## Code Examples

### Arduino Due (Built-in 12-bit DAC)

```cpp
// Arduino Due has two DAC pins: DAC0 and DAC1

void setup() {
  analogWriteResolution(12);  // Set DAC resolution to 12 bits (0-4095)
}

void loop() {
  // Output 1.65V on DAC0 (half of 3.3V reference)
  analogWrite(DAC0, 2048);  // 2048 / 4095 * 3.3V = 1.65V

  delay(1000);

  // Ramp voltage from 0V to 3.3V
  for (int value = 0; value < 4096; value++) {
    analogWrite(DAC0, value);
    delayMicroseconds(100);
  }
}

// Generate sine wave
void generateSineWave() {
  const int samples = 100;
  float frequency = 1000;  // 1 kHz

  for (int i = 0; i < samples; i++) {
    float angle = (2.0 * PI * i) / samples;
    int value = (sin(angle) + 1.0) * 2047.5;  // Scale to 0-4095
    analogWrite(DAC0, value);

    delayMicroseconds(1000000 / (frequency * samples));
  }
}
```

### ESP32 (Built-in 8-bit DAC)

```cpp
// ESP32 has two DAC channels: GPIO25 (DAC1) and GPIO26 (DAC2)

void setup() {
  // No special initialization needed for DAC
}

void loop() {
  // Output voltage (0-255 for 8-bit)
  // 0 = 0V, 255 = 3.3V
  dacWrite(25, 128);  // Output ~1.65V on GPIO25

  delay(1000);
}

// Generate sawtooth wave
void generateSawtoothWave() {
  for (int value = 0; value < 256; value++) {
    dacWrite(25, value);
    delayMicroseconds(10);
  }
}

// Generate triangle wave
void generateTriangleWave() {
  // Rising edge
  for (int value = 0; value < 256; value++) {
    dacWrite(25, value);
    delayMicroseconds(10);
  }

  // Falling edge
  for (int value = 255; value >= 0; value--) {
    dacWrite(25, value);
    delayMicroseconds(10);
  }
}

// Generate square wave
void generateSquareWave() {
  dacWrite(25, 255);  // HIGH
  delay(1);
  dacWrite(25, 0);    // LOW
  delay(1);
}

// Audio tone generation
void playTone(int frequency, int duration) {
  const int samples = 32;
  byte sineWave[samples];

  // Pre-calculate sine wave
  for (int i = 0; i < samples; i++) {
    sineWave[i] = (sin(2.0 * PI * i / samples) + 1.0) * 127.5;
  }

  unsigned long startTime = millis();
  int sampleDelay = 1000000 / (frequency * samples);

  while (millis() - startTime < duration) {
    for (int i = 0; i < samples; i++) {
      dacWrite(25, sineWave[i]);
      delayMicroseconds(sampleDelay);
    }
  }
}
```

### STM32 HAL DAC

```c
#include "stm32f4xx_hal.h"

DAC_HandleTypeDef hdac;

void DAC_Init(void) {
    DAC_ChannelConfTypeDef sConfig = {0};

    // Initialize DAC
    hdac.Instance = DAC;
    HAL_DAC_Init(&hdac);

    // Configure DAC channel 1
    sConfig.DAC_Trigger = DAC_TRIGGER_NONE;
    sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
    HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_1);

    // Start DAC
    HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
}

void DAC_SetVoltage(float voltage) {
    // Convert voltage to 12-bit value
    // Assuming 3.3V reference
    uint32_t value = (uint32_t)((voltage / 3.3f) * 4095.0f);

    if (value > 4095) value = 4095;

    HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, value);
}

void DAC_SetValue(uint16_t value) {
    HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, value);
}

// DMA-based waveform generation
uint16_t sineWave[100];

void DAC_GenerateSineWave_DMA(void) {
    // Pre-calculate sine wave
    for (int i = 0; i < 100; i++) {
        sineWave[i] = (uint16_t)((sin(2.0 * PI * i / 100.0) + 1.0) * 2047.5);
    }

    // Start DAC with DMA
    HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t*)sineWave, 100,
                      DAC_ALIGN_12B_R);

    // Configure timer to trigger DAC at specific rate
    // This enables continuous waveform output
}
```

### External DAC (MCP4725) via I2C

```cpp
#include <Wire.h>
#include <Adafruit_MCP4725.h>

Adafruit_MCP4725 dac;  // 12-bit DAC

void setup() {
  Serial.begin(115200);

  // Initialize MCP4725 (default address 0x62)
  if (!dac.begin(0x62)) {
    Serial.println("Failed to initialize MCP4725!");
    while (1);
  }

  Serial.println("MCP4725 initialized!");
}

void loop() {
  // Set voltage (0-4095 for 12-bit)
  // Vout = (value / 4095) * Vdd
  dac.setVoltage(2048, false);  // Output ~1.65V (Vdd/2)

  delay(1000);
}

// Ramp voltage smoothly
void rampVoltage(uint16_t start, uint16_t end, uint16_t steps) {
  int16_t increment = (end - start) / steps;

  for (uint16_t i = 0; i < steps; i++) {
    uint16_t value = start + (i * increment);
    dac.setVoltage(value, false);
    delay(10);
  }
}

// Generate precise voltage
void setVoltage(float voltage) {
  // Assuming 5V Vdd
  uint16_t value = (uint16_t)((voltage / 5.0) * 4095.0);
  dac.setVoltage(value, false);
}

// Store value in EEPROM (survives power cycle)
void saveVoltage(uint16_t value) {
  dac.setVoltage(value, true);  // true = write to EEPROM
}
```

### PWM as Pseudo-DAC (Arduino Uno)

```cpp
// Arduino Uno doesn't have true DAC, use PWM with filtering

const int pwmPin = 9;  // Any PWM pin

void setup() {
  pinMode(pwmPin, OUTPUT);

  // Increase PWM frequency for smoother output
  // Default: 980 Hz for pins 5,6 and 490 Hz for others
  // Setting for pin 9 and 10:
  TCCR1B = TCCR1B & 0b11111000 | 0x01;  // 31.25 kHz
}

void loop() {
  // Output 2.5V (50% duty cycle with 5V Vdd)
  analogWrite(pwmPin, 128);  // 0-255 range

  delay(1000);
}

// Hardware low-pass filter (required for PWM DAC):
// PWM Pin ----1kohm----, Output
//                      |
//                    10uF
//                      |
//                     GND
//
// Cutoff frequency = 1 / (2*pi * R * C) = ~16 Hz

// Convert voltage to PWM value
void setPWMVoltage(float voltage) {
  int pwmValue = (int)((voltage / 5.0) * 255.0);
  analogWrite(pwmPin, constrain(pwmValue, 0, 255));
}
```

## Common Applications

### 1. Audio Output

```cpp
// Simple audio playback
const byte audioSample[] = {128, 150, 172, 192, 209, ...};
const int sampleRate = 8000;  // 8 kHz

void playAudio() {
  for (int i = 0; i < sizeof(audioSample); i++) {
    dacWrite(25, audioSample[i]);
    delayMicroseconds(1000000 / sampleRate);
  }
}
```

### 2. Voltage Reference Generation

```cpp
// Generate precise reference voltage
void setReferenceVoltage(float voltage) {
  // Using 12-bit DAC with 3.3V reference
  uint16_t value = (uint16_t)((voltage / 3.3) * 4095);
  analogWrite(DAC0, value);
}

// Example: Generate 1.024V reference
void setup() {
  analogWriteResolution(12);
  setReferenceVoltage(1.024);  // Output constant 1.024V
}
```

### 3. Motor Speed Control

```cpp
// Control motor speed with voltage
void setMotorSpeed(int speedPercent) {
  // 0% = 0V, 100% = 3.3V
  int dacValue = map(speedPercent, 0, 100, 0, 255);
  dacWrite(25, dacValue);
}
```

### 4. LED Brightness (True Analog)

```cpp
// Unlike PWM, DAC gives true DC voltage
void setLEDBrightness(int percent) {
  int dacValue = map(percent, 0, 100, 0, 255);
  dacWrite(25, dacValue);
  // No flickering or PWM noise!
}
```

### 5. Signal Generation for Testing

```cpp
// Generate test signals
void generateDCOffset(float voltage) {
  uint16_t value = (uint16_t)((voltage / 3.3) * 4095);
  analogWrite(DAC0, value);
}

// Programmable voltage divider
void setProgrammableVoltage(float targetVoltage) {
  if (targetVoltage <= 3.3) {
    generateDCOffset(targetVoltage);
  }
}
```

## Waveform Generation

### Pre-calculated Waveform Tables

```cpp
// Sine wave lookup table (256 samples)
const uint8_t sineTable[256] PROGMEM = {
  127, 130, 133, 136, 139, 143, 146, 149,
  152, 155, 158, 161, 164, 167, 170, 173,
  // ... full 256 values
};

void generateSineFromTable(int frequency) {
  int delayTime = 1000000 / (frequency * 256);

  for (int i = 0; i < 256; i++) {
    uint8_t value = pgm_read_byte(&sineTable[i]);
    dacWrite(25, value);
    delayMicroseconds(delayTime);
  }
}
```

## Best Practices

### 1. Output Filtering

```
For cleaner output, add RC low-pass filter:

DAC Out ----100ohm----, Output
                       |
                     100nF
                       |
                      GND
```

### 2. Buffering

```
For driving loads, add op-amp buffer:

DAC Out ----, Op-Amp ---- Output
            |        |
            +--------+
              Feedback
```

### 3. Settling Time

```cpp
// Allow settling time after DAC update
void setDACWithSettling(uint16_t value) {
  analogWrite(DAC0, value);
  delayMicroseconds(10);  // Wait for output to settle
}
```

### 4. Reference Voltage Stability

```cpp
// Use external voltage reference for precision
// Internal reference can drift with temperature
```

## Common Issues and Debugging

### Problem: Output Voltage Incorrect
**Check**:
- Verify reference voltage
- Check calculation: (value / max) * Vref
- Ensure value doesn't exceed maximum
- Measure with high-impedance multimeter

### Problem: Noisy Output
**Solutions**:
- Add output filter capacitor (100nF)
- Use separate analog ground
- Add decoupling caps near DAC (0.1uF)
- Keep output wires short

### Problem: Can't Drive Load
**Solutions**:
- DAC outputs have limited current capability (~20mA typical)
- Add op-amp buffer for higher current
- Use darlington transistor for heavy loads

### Problem: Distorted Waveforms
**Check**:
- Update rate too slow for frequency
- Insufficient sample resolution
- Loading effect (add buffer)

## DAC Specifications to Consider

### 1. Resolution
- More bits = finer voltage control
- 8-bit usually sufficient for simple control
- 12-16 bit for audio and precision apps

### 2. Settling Time
- Time to reach final value
- Important for high-speed applications
- Typical: 1-10 us

### 3. Output Range
- Single-ended: 0V to Vref
- Bipolar: -Vref to +Vref (requires special circuit)

### 4. Update Rate
- How fast can DAC change values
- Audio: >40 kSPS
- Simple control: <1 kSPS

## ELI10 (Explain Like I'm 10)

Remember ADC is like a thermometer that converts smooth temperatures to numbers? DAC is the opposite!

Imagine you have a light dimmer switch:
- Instead of smoothly turning the knob, you can only pick from specific positions
- 8-bit DAC: You have 256 positions (0-255)
- 12-bit DAC: You have 4096 positions (way more precise!)

The DAC takes your number choice and creates a voltage:
- Digital number **0** -> 0 volts
- Digital number **128** (half) -> 1.65 volts
- Digital number **255** (max) -> 3.3 volts

It's like having a volume knob that you control with numbers instead of turning it by hand!

**PWM vs DAC**: PWM is like flashing a light super fast to make it look dimmer. DAC is like actually turning down the voltage - it's smoother and better for some jobs!

## Where this connects

- [ADC](adc.md) — the complementary input path; ADC reads analog, DAC outputs it
- [PWM](pwm.md) — substitute for a true DAC on MCUs without hardware DAC; requires RC low-pass filter
- [Timers](timers.md) — DMA-triggered timers drive periodic DAC updates for waveform generation
- [I2C](i2c.md) — external DAC chips (MCP4725) are controlled via I2C
- [GPIO](gpio.md) — DAC output pins are GPIO pins with analog alternate function

## Further Resources

- [DAC Tutorial - SparkFun](https://learn.sparkfun.com/tutorials/digital-to-analog-conversion)
- [Arduino Due DAC Reference](https://www.arduino.cc/reference/en/language/functions/analog-io/analogwrite/)
- [ESP32 DAC Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/dac.html)
- [MCP4725 Datasheet](https://www.microchip.com/en-us/product/MCP4725)
- [Audio with Arduino DAC](https://www.instructables.com/Arduino-Audio-Output/)
