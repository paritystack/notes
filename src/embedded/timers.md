# Timers and Counters

## Overview

Timers and counters are essential hardware peripherals in microcontrollers that keep track of time, count events, generate precise delays, create PWM signals, and trigger interrupts at specific intervals. Unlike software delays (which block the CPU), hardware timers run independently, allowing your program to multitask efficiently.

## Key Concepts

### Timer vs Counter

| Feature | Timer | Counter |
|---------|-------|---------|
| **Clock Source** | Internal (system clock) | External (GPIO pin) |
| **Purpose** | Measure time intervals | Count external events |
| **Speed** | Fixed by clock | Variable (event-driven) |
| **Example Use** | Generate 1ms interrupts | Count encoder pulses |

### Timer Components

1. **Counter Register**: Stores current count value
2. **Prescaler**: Divides input clock to slow down counting
3. **Compare Register**: Value to trigger events when matched
4. **Auto-reload Register**: Value to reset counter to (for periodic timers)

### Timer Modes

1. **Basic Timer**: Simple counting up or down
2. **PWM Mode**: Generate pulse-width modulated signals
3. **Input Capture**: Measure external signal timing
4. **Output Compare**: Trigger events at specific times
5. **Encoder Mode**: Read quadrature encoders

## How It Works

### Clock and Prescaler

```
System Clock (16 MHz)
    “
Prescaler (÷256)
    “
Timer Clock (62.5 kHz)
    “
Counter increments at 62.5 kHz
```

**Formula**:
```
Timer Frequency = CPU Frequency / Prescaler

Timer Period = 1 / Timer Frequency

Overflow Time = (2^bits / Timer Frequency)
```

**Example** (Arduino Uno - 16 MHz):
```
Prescaler = 256
Timer Frequency = 16,000,000 / 256 = 62,500 Hz
Timer Period = 1 / 62,500 = 16 ¼s per tick

For 8-bit timer (0-255):
Overflow Time = 256 × 16 ¼s = 4.096 ms

For 16-bit timer (0-65535):
Overflow Time = 65,536 × 16 ¼s = 1.048 seconds
```

## Code Examples

### Arduino Timer Interrupt

```cpp
// Using Timer1 (16-bit) for 1ms interrupt

volatile unsigned long millisCounter = 0;

void setup() {
  Serial.begin(9600);

  // Stop interrupts during setup
  cli();

  // Reset Timer1
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // Set compare match register for 1ms
  // OCR1A = (16MHz / (prescaler × desired frequency)) - 1
  // OCR1A = (16,000,000 / (64 × 1000)) - 1 = 249
  OCR1A = 249;

  // Turn on CTC mode (Clear Timer on Compare Match)
  TCCR1B |= (1 << WGM12);

  // Set CS11 and CS10 bits for 64 prescaler
  TCCR1B |= (1 << CS11) | (1 << CS10);

  // Enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);

  // Enable global interrupts
  sei();
}

// Timer1 interrupt service routine (ISR)
ISR(TIMER1_COMPA_vect) {
  millisCounter++;

  // Your code here - keep it SHORT!
  // DO NOT use Serial.print() in ISR
}

void loop() {
  // Use millisCounter instead of millis()
  static unsigned long lastPrint = 0;

  if (millisCounter - lastPrint >= 1000) {
    lastPrint = millisCounter;
    Serial.println(millisCounter);
  }
}
```

### ESP32 Hardware Timer

```cpp
// ESP32 has 4 hardware timers (0-3)

hw_timer_t *timer = NULL;
volatile uint32_t timerCounter = 0;

void IRAM_ATTR onTimer() {
  timerCounter++;
  // Keep ISR short and fast!
}

void setup() {
  Serial.begin(115200);

  // Initialize timer (timer number, prescaler, count up)
  // ESP32 clock is 80 MHz
  // Prescaler of 80 gives 1 MHz (1 tick = 1 ¼s)
  timer = timerBegin(0, 80, true);

  // Attach interrupt function
  timerAttachInterrupt(timer, &onTimer, true);

  // Set alarm to trigger every 1ms (1000 ¼s)
  timerAlarmWrite(timer, 1000, true);  // true = auto-reload

  // Enable timer alarm
  timerAlarmEnable(timer);

  Serial.println("Timer initialized!");
}

void loop() {
  static uint32_t lastCount = 0;

  if (timerCounter - lastCount >= 1000) {
    lastCount = timerCounter;
    Serial.print("Timer count: ");
    Serial.println(timerCounter);
  }
}

// Ticker library (easier alternative)
#include <Ticker.h>

Ticker ticker;
volatile int count = 0;

void timerCallback() {
  count++;
}

void setup() {
  // Call timerCallback every 0.001 seconds (1ms)
  ticker.attach(0.001, timerCallback);
}
```

### STM32 HAL Timer

```c
#include "stm32f4xx_hal.h"

TIM_HandleTypeDef htim2;
volatile uint32_t timerTicks = 0;

void Timer_Init(void) {
    TIM_ClockConfigTypeDef sClockSourceConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};

    // TIM2 configuration
    // APB1 clock = 84 MHz (for STM32F4)
    // Prescaler = 8400 - 1 ’ 10 kHz timer clock
    // Period = 10 - 1 ’ 1 kHz interrupt (1ms)

    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 8400 - 1;      // 84 MHz / 8400 = 10 kHz
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 10 - 1;           // 10 kHz / 10 = 1 kHz (1ms)
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    HAL_TIM_Base_Init(&htim2);

    sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
    HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig);

    // Enable timer interrupt
    HAL_TIM_Base_Start_IT(&htim2);
}

// Timer interrupt callback
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2) {
        timerTicks++;
        // Your periodic code here
    }
}

// In main.c, enable interrupt in NVIC
void MX_TIM2_Init(void) {
    Timer_Init();
    HAL_NVIC_SetPriority(TIM2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(TIM2_IRQn);
}
```

### PWM Generation with Timers

```cpp
// Arduino PWM using Timer1

void setup() {
  // Set pins as output
  pinMode(9, OUTPUT);   // OC1A
  pinMode(10, OUTPUT);  // OC1B

  // Stop timer during configuration
  TCCR1A = 0;
  TCCR1B = 0;

  // Fast PWM mode, ICR1 as TOP
  // WGM13:0 = 14 (Fast PWM, TOP = ICR1)
  TCCR1A = (1 << WGM11);
  TCCR1B = (1 << WGM13) | (1 << WGM12);

  // Non-inverting mode for both channels
  TCCR1A |= (1 << COM1A1) | (1 << COM1B1);

  // Prescaler = 8
  TCCR1B |= (1 << CS11);

  // Set TOP value for desired frequency
  // PWM Frequency = F_CPU / (Prescaler × (1 + TOP))
  // For 50 Hz: TOP = 16,000,000 / (8 × 50) - 1 = 39999
  ICR1 = 39999;  // 50 Hz

  // Set duty cycle
  OCR1A = 3000;  // ~7.5% duty cycle on pin 9
  OCR1B = 6000;  // ~15% duty cycle on pin 10
}

// Servo control example
void setServoAngle(uint8_t angle) {
  // Servo expects 1ms-2ms pulse every 20ms (50 Hz)
  // 1ms = 0° = 2000 counts
  // 1.5ms = 90° = 3000 counts
  // 2ms = 180° = 4000 counts

  uint16_t pulse = map(angle, 0, 180, 2000, 4000);
  OCR1A = pulse;
}

void loop() {
  setServoAngle(0);
  delay(1000);
  setServoAngle(90);
  delay(1000);
  setServoAngle(180);
  delay(1000);
}
```

### Input Capture Mode

```cpp
// Measure frequency of external signal on pin 8 (ICP1)

volatile unsigned long captureTime1 = 0;
volatile unsigned long captureTime2 = 0;
volatile boolean newCapture = false;

void setup() {
  Serial.begin(9600);

  // Configure Timer1 for input capture
  TCCR1A = 0;
  TCCR1B = 0;

  // Prescaler = 64 (250 kHz timer, 4¼s resolution)
  TCCR1B |= (1 << CS11) | (1 << CS10);

  // Input Capture on rising edge
  TCCR1B |= (1 << ICES1);

  // Enable input capture interrupt
  TIMSK1 |= (1 << ICIE1);

  // Enable global interrupts
  sei();
}

// Input capture interrupt
ISR(TIMER1_CAPT_vect) {
  static boolean firstCapture = true;

  if (firstCapture) {
    captureTime1 = ICR1;
    firstCapture = false;
  } else {
    captureTime2 = ICR1;
    newCapture = true;
    firstCapture = true;
  }
}

void loop() {
  if (newCapture) {
    newCapture = false;

    // Calculate period
    unsigned long period = captureTime2 - captureTime1;

    // Calculate frequency
    // Timer runs at 250 kHz (4¼s per tick)
    float frequency = 250000.0 / period;

    Serial.print("Frequency: ");
    Serial.print(frequency);
    Serial.println(" Hz");
  }
}
```

## Common Applications

### 1. Precise Timing Without Delay

```cpp
unsigned long previousMillis = 0;
const long interval = 1000;

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // Execute every 1 second without blocking
    toggleLED();
  }

  // Other code runs continuously
  checkSensors();
  processData();
}
```

### 2. Multiple Periodic Tasks

```cpp
volatile uint32_t timerTicks = 0;

ISR(TIMER1_COMPA_vect) {
  timerTicks++;
}

void loop() {
  static uint32_t lastTask1 = 0;
  static uint32_t lastTask2 = 0;
  static uint32_t lastTask3 = 0;

  // Task 1: Every 10ms
  if (timerTicks - lastTask1 >= 10) {
    lastTask1 = timerTicks;
    readSensors();
  }

  // Task 2: Every 100ms
  if (timerTicks - lastTask2 >= 100) {
    lastTask2 = timerTicks;
    updateDisplay();
  }

  // Task 3: Every 1000ms
  if (timerTicks - lastTask3 >= 1000) {
    lastTask3 = timerTicks;
    sendData();
  }
}
```

### 3. Watchdog Timer

```cpp
#include <avr/wdt.h>

void setup() {
  // Enable watchdog timer (8 second timeout)
  wdt_enable(WDTO_8S);
}

void loop() {
  // Do work
  processData();

  // Reset watchdog (prevent system reset)
  wdt_reset();

  // If code hangs, watchdog resets system after 8 seconds
}
```

### 4. Real-Time Clock (RTC)

```cpp
// Using timer to maintain time

volatile uint32_t seconds = 0;
volatile uint16_t milliseconds = 0;

ISR(TIMER1_COMPA_vect) {
  milliseconds++;

  if (milliseconds >= 1000) {
    milliseconds = 0;
    seconds++;
  }
}

void getTime(uint8_t *hours, uint8_t *minutes, uint8_t *secs) {
  noInterrupts();
  uint32_t totalSeconds = seconds;
  interrupts();

  *hours = (totalSeconds / 3600) % 24;
  *minutes = (totalSeconds / 60) % 60;
  *secs = totalSeconds % 60;
}
```

### 5. Debouncing Buttons

```cpp
volatile uint32_t timerMs = 0;

ISR(TIMER1_COMPA_vect) {
  timerMs++;
}

const int buttonPin = 2;
const int debounceTime = 50;  // 50ms

bool readButtonDebounced() {
  static uint32_t lastDebounceTime = 0;
  static bool lastButtonState = HIGH;
  static bool buttonState = HIGH;

  bool reading = digitalRead(buttonPin);

  if (reading != lastButtonState) {
    lastDebounceTime = timerMs;
  }

  if ((timerMs - lastDebounceTime) > debounceTime) {
    if (reading != buttonState) {
      buttonState = reading;
      return (buttonState == LOW);  // Return true on button press
    }
  }

  lastButtonState = reading;
  return false;
}
```

## Timer Prescaler Values

### AVR (Arduino Uno/Nano/Mega)

| Prescaler | CS12 | CS11 | CS10 | Timer Frequency (16 MHz) |
|-----------|------|------|------|-------------------------|
| None | 0 | 0 | 0 | Stopped |
| 1 | 0 | 0 | 1 | 16 MHz |
| 8 | 0 | 1 | 0 | 2 MHz |
| 64 | 0 | 1 | 1 | 250 kHz |
| 256 | 1 | 0 | 0 | 62.5 kHz |
| 1024 | 1 | 0 | 1 | 15.625 kHz |

## Best Practices

### 1. Keep ISRs Short and Fast

```cpp
// BAD - Don't do this in ISR!
ISR(TIMER1_COMPA_vect) {
  Serial.println("Timer fired");  // Serial is slow!
  delay(100);                      // Blocks other interrupts!
  float result = complexCalculation();  // Takes too long!
}

// GOOD - Set flags, process in main loop
volatile bool timerFlag = false;

ISR(TIMER1_COMPA_vect) {
  timerFlag = true;  // Just set a flag
}

void loop() {
  if (timerFlag) {
    timerFlag = false;
    Serial.println("Timer fired");  // Do slow stuff here
    processData();
  }
}
```

### 2. Protect Shared Variables

```cpp
volatile uint32_t sharedCounter = 0;

ISR(TIMER1_COMPA_vect) {
  sharedCounter++;
}

void loop() {
  // BAD - Not atomic! Can be corrupted if interrupt occurs mid-read
  uint32_t localCopy = sharedCounter;

  // GOOD - Disable interrupts during multi-byte read
  noInterrupts();
  uint32_t localCopy = sharedCounter;
  interrupts();

  Serial.println(localCopy);
}
```

### 3. Calculate Timer Values Correctly

```cpp
// Formula for CTC mode:
// Compare Value = (F_CPU / (Prescaler × Desired_Frequency)) - 1

#define F_CPU 16000000UL
#define PRESCALER 64
#define DESIRED_HZ 1000  // 1 kHz

uint16_t compareValue = (F_CPU / (PRESCALER * DESIRED_HZ)) - 1;
// compareValue = (16000000 / (64 × 1000)) - 1 = 249

OCR1A = compareValue;
```

## Common Issues and Debugging

### Problem: Timer Interrupt Not Firing
**Check**:
- Global interrupts enabled (`sei()`)
- Specific timer interrupt enabled
- Prescaler and compare values calculated correctly
- Clock source selected
- ISR function name matches vector name

### Problem: Inaccurate Timing
**Causes**:
- Wrong prescaler calculation
- Integer overflow in calculations
- CPU frequency mismatch
- Crystal tolerance

### Problem: System Becomes Unresponsive
**Causes**:
- ISR takes too long (blocks other code)
- Interrupt firing too frequently
- Infinite loop in ISR
- Nested interrupts causing stack overflow

## ELI10 (Explain Like I'm 10)

Imagine you have a special alarm clock that can do cool tricks:

1. **Basic Timer**: Counts from 0 to 100, then starts over. Like counting seconds!

2. **Prescaler**: Instead of counting every second, you count every 10 seconds. It's like skipping numbers to count slower.

3. **Compare Match**: When the count reaches a special number (like 50), the alarm rings! Then it keeps counting.

4. **PWM**: The alarm flashes a light on and off really fast. By changing how long it stays on vs off, you can make the light look dimmer or brighter!

5. **Input Capture**: You press a button, and the timer remembers what number it was at. Press again, and you can figure out how long between presses!

The coolest part? The timer runs by itself in the background - you don't have to watch it! It's like having a helper that tells you when it's time to do something, while you focus on other tasks.

## Further Resources

- [Arduino Timer Interrupts](https://www.arduino.cc/reference/en/language/functions/interrupts/interrupts/)
- [AVR Timers Tutorial](http://www.protostack.com/blog/2011/01/comprehensive-guide-to-timers-on-avr/)
- [ESP32 Timer Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/timer.html)
- [STM32 Timer Cookbook](https://www.st.com/resource/en/application_note/cd00211314-stm32f101xx-and-stm32f103xx-rcc-configuration-examples-stmicroelectronics.pdf)
- [Secrets of Arduino PWM](https://docs.arduino.cc/learn/microcontrollers/analog-output)
