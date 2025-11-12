# ADC (Analog-to-Digital Converter)

## Overview

An Analog-to-Digital Converter (ADC) is a hardware component that converts continuous analog signals (like voltage, temperature, light intensity) into discrete digital values that a microcontroller can process. ADCs are essential for interfacing with the real world, enabling microcontrollers to read sensors and analog inputs.

## Key Concepts

### What is an Analog Signal?

An analog signal is a continuous signal that can have any value within a range. Examples:
- Temperature: 0°C to 100°C
- Light intensity: 0 to maximum brightness
- Audio: continuous sound waves
- Voltage: 0V to 5V

### What is a Digital Value?

A digital value is a discrete number that represents the analog signal:
- 8-bit ADC: 0 to 255 (256 possible values)
- 10-bit ADC: 0 to 1023 (1024 possible values)
- 12-bit ADC: 0 to 4095 (4096 possible values)

### Resolution

Resolution determines how finely an ADC can distinguish between different analog values.

**Formula**:
```
Resolution = Reference Voltage / (2^n - 1)

Where n = number of bits
```

**Examples**:

| Bits | Levels | Resolution (5V ref) | Resolution (3.3V ref) |
|------|--------|---------------------|----------------------|
| 8-bit | 256 | 19.6 mV | 12.9 mV |
| 10-bit | 1024 | 4.88 mV | 3.22 mV |
| 12-bit | 4096 | 1.22 mV | 0.81 mV |
| 16-bit | 65536 | 76.3 ¼V | 50.4 ¼V |

**What this means**: A 10-bit ADC with 5V reference can distinguish voltage differences as small as ~4.88mV.

### Reference Voltage (VREF)

The reference voltage defines the maximum input voltage the ADC can measure.

- **Arduino Uno**: 5V (can use external ref)
- **ESP32**: 3.3V (default), 1.1V (attenuated)
- **STM32**: 3.3V (typically)

**Important**: Never exceed VREF on analog input pins!

### Sampling Rate

How many times per second the ADC can take a measurement, measured in:
- **SPS**: Samples Per Second
- **kSPS**: Thousand samples per second
- **MSPS**: Million samples per second

**Examples**:
- Arduino Uno: ~10 kSPS
- ESP32: ~100 kSPS
- STM32F4: Up to 2.4 MSPS
- External ADC (ADS1115): 860 SPS max

## How It Works

### Conversion Process

1. **Sample**: Capture the analog voltage at a specific moment
2. **Hold**: Maintain that voltage level during conversion
3. **Quantize**: Divide the voltage range into discrete levels
4. **Encode**: Convert to a binary number

```
Analog Input (2.5V) ’ ADC ’ Digital Output (512 for 10-bit at 5V ref)

Calculation: 2.5V / 5V × 1023 = 511.5 H 512
```

### Conversion Formula

```
Digital Value = (Analog Voltage / Reference Voltage) × (2^n - 1)

Analog Voltage = (Digital Value / (2^n - 1)) × Reference Voltage
```

## Code Examples

### Arduino (AVR) ADC

```cpp
// Simple analog read
const int sensorPin = A0;

void setup() {
  Serial.begin(9600);

  // Optional: Set analog reference
  // analogReference(DEFAULT);  // 5V on Uno
  // analogReference(INTERNAL); // 1.1V internal reference
  // analogReference(EXTERNAL); // External AREF pin
}

void loop() {
  // Read analog value (0-1023)
  int rawValue = analogRead(sensorPin);

  // Convert to voltage
  float voltage = rawValue * (5.0 / 1023.0);

  Serial.print("Raw: ");
  Serial.print(rawValue);
  Serial.print(" | Voltage: ");
  Serial.print(voltage);
  Serial.println(" V");

  delay(500);
}

// Reading multiple analog pins
void readMultipleSensors() {
  int sensors[] = {A0, A1, A2, A3};

  for (int i = 0; i < 4; i++) {
    int value = analogRead(sensors[i]);
    float voltage = value * (5.0 / 1023.0);
    Serial.print("Sensor ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(voltage);
  }
}
```

### ESP32 ADC

```cpp
// ESP32 has two ADC units with multiple channels
const int analogPin = 34;  // ADC1_CH6 (GPIO 34)

void setup() {
  Serial.begin(115200);

  // Set ADC resolution (9-12 bits)
  analogReadResolution(12);  // Default is 12 bits (0-4095)

  // Set ADC attenuation (changes measurement range)
  // ADC_0db: 0-1.1V
  // ADC_2_5db: 0-1.5V
  // ADC_6db: 0-2.2V (default)
  // ADC_11db: 0-3.3V
  analogSetAttenuation(ADC_11db);

  // Or set per pin
  analogSetPinAttenuation(analogPin, ADC_11db);
}

void loop() {
  int rawValue = analogRead(analogPin);

  // Convert to voltage (with 11db attenuation, 0-3.3V range)
  // Note: ESP32 ADC is non-linear, consider calibration
  float voltage = rawValue * (3.3 / 4095.0);

  Serial.print("Raw: ");
  Serial.print(rawValue);
  Serial.print(" | Voltage: ");
  Serial.println(voltage);

  delay(100);
}

// Better: Use calibrated read
#include "esp_adc_cal.h"

esp_adc_cal_characteristics_t adc_chars;

void setupCalibrated() {
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,
                           ADC_WIDTH_BIT_12, 1100, &adc_chars);
}

void loopCalibrated() {
  uint32_t voltage = analogRead(analogPin);
  voltage = esp_adc_cal_raw_to_voltage(voltage, &adc_chars);

  Serial.print("Calibrated voltage: ");
  Serial.print(voltage);
  Serial.println(" mV");
}
```

### STM32 HAL ADC

```c
#include "stm32f4xx_hal.h"

ADC_HandleTypeDef hadc1;

void ADC_Init(void) {
    ADC_ChannelConfTypeDef sConfig = {0};

    // Configure ADC
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode = DISABLE;
    hadc1.Init.ContinuousConvMode = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    HAL_ADC_Init(&hadc1);

    // Configure channel
    sConfig.Channel = ADC_CHANNEL_0;
    sConfig.Rank = 1;
    sConfig.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}

uint16_t ADC_Read(uint32_t channel) {
    ADC_ChannelConfTypeDef sConfig = {0};

    sConfig.Channel = channel;
    sConfig.Rank = 1;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);

    // Start conversion
    HAL_ADC_Start(&hadc1);

    // Wait for conversion to complete
    HAL_ADC_PollForConversion(&hadc1, 100);

    // Read value
    uint16_t value = HAL_ADC_GetValue(&hadc1);

    return value;
}

float ADC_ReadVoltage(uint32_t channel) {
    uint16_t raw = ADC_Read(channel);
    // Convert to voltage (assuming 3.3V reference)
    float voltage = (raw * 3.3f) / 4095.0f;
    return voltage;
}

// DMA-based continuous conversion
uint16_t adc_buffer[16];

void ADC_Start_DMA(void) {
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, 16);
}
```

### External ADC (ADS1115) via I2C

```cpp
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;  // 16-bit ADC

void setup() {
  Serial.begin(115200);

  // Initialize ADS1115
  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS1115!");
    while (1);
  }

  // Set gain
  // ads.setGain(GAIN_TWOTHIRDS);  // ±6.144V range
  // ads.setGain(GAIN_ONE);        // ±4.096V range
  ads.setGain(GAIN_TWO);           // ±2.048V range (default)
  // ads.setGain(GAIN_FOUR);       // ±1.024V range
  // ads.setGain(GAIN_EIGHT);      // ±0.512V range
  // ads.setGain(GAIN_SIXTEEN);    // ±0.256V range
}

void loop() {
  // Read single-ended from channel 0
  int16_t adc0 = ads.readADC_SingleEnded(0);
  float voltage0 = ads.computeVolts(adc0);

  // Read differential (channel 0 - channel 1)
  int16_t diff01 = ads.readADC_Differential_0_1();

  Serial.print("ADC0: ");
  Serial.print(adc0);
  Serial.print(" | Voltage: ");
  Serial.println(voltage0);

  delay(100);
}
```

## Common Applications

### 1. Temperature Sensors (Thermistor)

```cpp
const int thermistorPin = A0;
const float BETA = 3950;  // Beta coefficient
const float R0 = 10000;   // Resistance at 25°C
const float T0 = 298.15;  // 25°C in Kelvin

float readTemperature() {
  int raw = analogRead(thermistorPin);

  // Convert to resistance
  float R = 10000.0 * (1023.0 / raw - 1.0);

  // Steinhart-Hart equation
  float T = 1.0 / (1.0/T0 + (1.0/BETA) * log(R/R0));

  return T - 273.15;  // Convert to Celsius
}
```

### 2. Light Sensor (LDR/Photoresistor)

```cpp
const int ldrPin = A1;

int readLightLevel() {
  int rawValue = analogRead(ldrPin);

  // Convert to percentage
  int lightPercent = map(rawValue, 0, 1023, 0, 100);

  return lightPercent;
}
```

### 3. Potentiometer (Volume Control)

```cpp
const int potPin = A2;
const int ledPin = 9;  // PWM pin

void setup() {
  pinMode(ledPin, OUTPUT);
}

void loop() {
  int potValue = analogRead(potPin);

  // Map to PWM range (0-255)
  int brightness = map(potValue, 0, 1023, 0, 255);

  analogWrite(ledPin, brightness);
}
```

### 4. Battery Voltage Monitoring

```cpp
const int batteryPin = A3;
const float voltageDividerRatio = 2.0;  // R1=R2=10k

float readBatteryVoltage() {
  int raw = analogRead(batteryPin);

  // Convert to actual voltage
  float adcVoltage = raw * (5.0 / 1023.0);

  // Account for voltage divider
  float batteryVoltage = adcVoltage * voltageDividerRatio;

  return batteryVoltage;
}

void checkBattery() {
  float voltage = readBatteryVoltage();

  if (voltage < 3.3) {
    Serial.println("WARNING: Low battery!");
  }
}
```

### 5. Current Sensing (ACS712)

```cpp
const int currentSensorPin = A4;
const float sensitivity = 0.185;  // 185mV/A for ACS712-05B

float readCurrent() {
  int raw = analogRead(currentSensorPin);
  float voltage = raw * (5.0 / 1023.0);

  // Zero point is 2.5V (Vcc/2)
  float offsetVoltage = voltage - 2.5;

  // Calculate current
  float current = offsetVoltage / sensitivity;

  return current;
}
```

## Best Practices

### 1. Averaging for Stability

```cpp
float readAverageAnalog(int pin, int samples = 10) {
  long sum = 0;

  for (int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delay(10);  // Small delay between reads
  }

  return (float)sum / samples;
}
```

### 2. Handling Noise

```cpp
// Software low-pass filter (running average)
const int numReadings = 10;
int readings[numReadings];
int readIndex = 0;
int total = 0;

int smoothedRead(int pin) {
  total -= readings[readIndex];
  readings[readIndex] = analogRead(pin);
  total += readings[readIndex];
  readIndex = (readIndex + 1) % numReadings;

  return total / numReadings;
}
```

### 3. Proper Voltage Divider

```cpp
// To measure higher voltages, use voltage divider
// Vin     R1  ,    R2     GND
//             
//          ADC Pin

// Example: Measure 12V with 5V ADC
// R1 = 10k©, R2 = 7.5k©
// Vout = Vin × (R2 / (R1 + R2))
// Vout = 12V × (7.5 / 17.5) = 5.14V (slightly over, use 6.8k© for R2)
```

### 4. Calibration

```cpp
struct CalibrationData {
  float slope;
  float offset;
};

CalibrationData calibrate(int pin, float knownVoltage) {
  int rawValue = analogRead(pin);

  CalibrationData cal;
  cal.slope = knownVoltage / rawValue;
  cal.offset = 0;  // Adjust if needed

  return cal;
}

float calibratedRead(int pin, CalibrationData cal) {
  int raw = analogRead(pin);
  return (raw * cal.slope) + cal.offset;
}
```

## Common Issues and Debugging

### Problem: Noisy Readings
**Solutions**:
- Add 0.1¼F capacitor between analog pin and ground
- Use averaging/filtering in software
- Keep analog wires short and away from digital signals
- Use twisted pair cables for long runs
- Add ferrite beads on long cables

### Problem: Incorrect Voltage Readings
**Check**:
- Verify reference voltage is correct
- Check voltage divider calculations
- Ensure input doesn't exceed VREF
- Verify ground connection

### Problem: Slow Response
**Solutions**:
- Reduce averaging samples
- Check ADC clock/prescaler settings
- Use faster ADC if needed (external)
- Enable DMA for continuous sampling

## ELI10 (Explain Like I'm 10)

Imagine you have a thermometer that shows any temperature between 0°C and 100°C, but you can only report whole numbers:

- If the real temperature is 23.7°C, you might say "24°C"
- If it's 23.2°C, you might say "23°C"

An ADC does the same thing! It takes a smooth, continuous voltage (like the temperature) and converts it to a number your microcontroller can understand.

**Resolution** is like how many different numbers you can say:
- 8-bit ADC: Can say 256 different numbers (0-255)
- 10-bit ADC: Can say 1024 different numbers (0-1023)
- 12-bit ADC: Can say 4096 different numbers (0-4095)

More bits = more precise measurements = seeing smaller differences!

## Further Resources

- [ADC Tutorial - SparkFun](https://learn.sparkfun.com/tutorials/analog-to-digital-conversion)
- [Arduino analogRead() Reference](https://www.arduino.cc/reference/en/language/functions/analog-io/analogread/)
- [ESP32 ADC Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/adc.html)
- [ADC Noise Reduction Techniques - AVR](https://www.microchip.com/en-us/application-notes/an2538)
- [Understanding ADC Parameters](https://www.ti.com/lit/an/slaa013/slaa013.pdf)
