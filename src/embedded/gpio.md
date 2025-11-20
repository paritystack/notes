# GPIO

## General Purpose Input/Output (GPIO)

GPIO stands for General Purpose Input/Output. It is a generic pin on an integrated circuit or computer board whose behavior (including whether it is an input or output pin) can be controlled by the user at runtime. GPIO pins are a staple in embedded systems and microcontroller projects due to their versatility and ease of use.

### Key Features of GPIO

1. **Configurable Direction**: Each GPIO pin can be configured as either an input or an output. This allows the pin to either read signals from external devices (input) or send signals to external devices (output).

2. **Digital Signals**: GPIO pins typically handle digital signals, meaning they can be in one of two states: high (1) or low (0). The voltage levels corresponding to these states depend on the specific hardware but are commonly 3.3V or 5V for high and 0V for low.

3. **Interrupts**: Many GPIO pins support interrupts, which allow the pin to trigger an event in the software when a specific condition is met, such as a change in state. This is useful for responding to external events without constantly polling the pin.

4. **Pull-up/Pull-down Resistors**: GPIO pins often have configurable pull-up or pull-down resistors. These resistors ensure that the pin is in a known state (high or low) when it is not actively being driven by an external source.

5. **Debouncing**: When reading input from mechanical switches, GPIO pins can experience noise or "bouncing." Debouncing techniques, either in hardware or software, are used to ensure that the signal is stable and accurate.

### Common Uses of GPIO

- **LED Control**: Turning LEDs on and off or controlling their brightness using Pulse Width Modulation (PWM).
- **Button Inputs**: Reading the state of buttons or switches to trigger actions in the software.
- **Sensor Interfacing**: Reading data from various sensors like temperature, humidity, or motion sensors.
- **Communication**: Implementing simple communication protocols like I2C, SPI, or UART using GPIO pins.

### Example Code

Here is an example of how to configure and use a GPIO pin in a typical microcontroller environment (e.g., using the Arduino platform):
```cpp
// Define the pin number
const int ledPin = 13; // Pin number for the LED

void setup() {
  // Initialize the digital pin as an output.
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // Turn the LED on (HIGH is the voltage level)
  digitalWrite(ledPin, HIGH);
  // Wait for a second
  delay(1000);
  // Turn the LED off by making the voltage LOW
  digitalWrite(ledPin, LOW);
  // Wait for a second
  delay(1000);
}
```

## Hardware Registers and Low-Level Configuration

### GPIO Register Structure

GPIO pins are controlled through memory-mapped registers. Understanding these registers is crucial for low-level embedded programming and achieving optimal performance.

#### Common GPIO Registers

1. **Direction Register (DDR/DIR)**: Configures pin as input (0) or output (1)
2. **Data Output Register (PORT/OUT)**: Sets output value for pins configured as outputs
3. **Data Input Register (PIN/IN)**: Reads current state of pins configured as inputs
4. **Pull-up/Pull-down Register**: Enables internal resistors
5. **Alternate Function Register**: Selects special functions (UART, SPI, PWM, etc.)

### Memory-Mapped I/O

GPIO registers are accessed through specific memory addresses. Here's a conceptual example:

```c
// Example register addresses (hypothetical microcontroller)
#define GPIO_BASE_ADDR    0x40020000
#define GPIOA_MODER      (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x00))  // Mode register
#define GPIOA_OTYPER     (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x04))  // Output type
#define GPIOA_OSPEEDR    (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x08))  // Speed register
#define GPIOA_PUPDR      (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x0C))  // Pull-up/down
#define GPIOA_IDR        (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x10))  // Input data
#define GPIOA_ODR        (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x14))  // Output data
#define GPIOA_BSRR       (*(volatile uint32_t *)(GPIO_BASE_ADDR + 0x18))  // Bit set/reset
```

### Bit Manipulation Techniques

Efficient GPIO control often requires bit manipulation:

```c
// Set a specific bit (turn on pin 5)
GPIOA_ODR |= (1 << 5);

// Clear a specific bit (turn off pin 5)
GPIOA_ODR &= ~(1 << 5);

// Toggle a specific bit
GPIOA_ODR ^= (1 << 5);

// Read a specific bit
uint8_t pin_state = (GPIOA_IDR >> 5) & 0x01;

// Atomic bit set/reset using BSRR (faster, no read-modify-write)
GPIOA_BSRR = (1 << 5);      // Set pin 5
GPIOA_BSRR = (1 << (5+16)); // Reset pin 5
```

### Low-Level Configuration Example

```c
// Configure GPIO pin as output (bare-metal approach)
void gpio_init_output(volatile uint32_t *port_moder, uint8_t pin) {
    // Clear the mode bits for this pin
    *port_moder &= ~(0x3 << (pin * 2));
    // Set as output (01)
    *port_moder |= (0x1 << (pin * 2));
}

// Configure GPIO pin as input with pull-up
void gpio_init_input_pullup(volatile uint32_t *port_moder,
                             volatile uint32_t *port_pupdr,
                             uint8_t pin) {
    // Set as input (00)
    *port_moder &= ~(0x3 << (pin * 2));
    // Enable pull-up (01)
    *port_pupdr &= ~(0x3 << (pin * 2));
    *port_pupdr |= (0x1 << (pin * 2));
}
```

## Platform-Specific Examples

### STM32 (ARM Cortex-M)

#### Using STM32 HAL Library

```c
#include "stm32f4xx_hal.h"

void GPIO_Init_STM32(void) {
    // Enable GPIO clock
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO pin
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;  // Push-pull output
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Set pin high
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
}

// Using interrupts
void GPIO_EXTI_Init(void) {
    __HAL_RCC_GPIOC_CLK_ENABLE();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;  // Interrupt on falling edge
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    // Configure NVIC
    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}

// Interrupt handler
void EXTI15_10_IRQHandler(void) {
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13);
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    if (GPIO_Pin == GPIO_PIN_13) {
        // Button pressed - handle event
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
    }
}
```

#### Bare-Metal STM32

```c
#include "stm32f4xx.h"

void gpio_baremetal_init(void) {
    // Enable GPIOA clock
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

    // Configure PA5 as output
    GPIOA->MODER &= ~(0x3 << (5 * 2));  // Clear mode bits
    GPIOA->MODER |= (0x1 << (5 * 2));   // Set as output

    // Set output type as push-pull
    GPIOA->OTYPER &= ~(1 << 5);

    // Set speed to low
    GPIOA->OSPEEDR &= ~(0x3 << (5 * 2));

    // Toggle LED
    GPIOA->ODR ^= (1 << 5);
}
```

### ESP32 (Xtensa/RISC-V)

```c
#include "driver/gpio.h"

void gpio_esp32_init(void) {
    // Configure GPIO as output
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_NUM_2),  // GPIO 2
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);

    // Set GPIO level
    gpio_set_level(GPIO_NUM_2, 1);
}

// Interrupt example
void IRAM_ATTR gpio_isr_handler(void* arg) {
    uint32_t gpio_num = (uint32_t) arg;
    // Handle interrupt
}

void gpio_esp32_interrupt_init(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_NUM_0),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_NEGEDGE  // Interrupt on falling edge
    };
    gpio_config(&io_conf);

    // Install ISR service
    gpio_install_isr_service(0);
    gpio_isr_handler_add(GPIO_NUM_0, gpio_isr_handler, (void*) GPIO_NUM_0);
}
```

### Raspberry Pi (Linux-based)

#### Using gpiod Library (Modern Approach)

```c
#include <gpiod.h>

void gpio_rpi_init(void) {
    struct gpiod_chip *chip;
    struct gpiod_line *line;

    // Open GPIO chip
    chip = gpiod_chip_open("/dev/gpiochip0");

    // Get GPIO line (BCM pin 17)
    line = gpiod_chip_get_line(chip, 17);

    // Request line as output
    gpiod_line_request_output(line, "led", 0);

    // Set value
    gpiod_line_set_value(line, 1);

    // Cleanup
    gpiod_line_release(line);
    gpiod_chip_close(chip);
}
```

#### Using Python (RPi.GPIO)

```python
import RPi.GPIO as GPIO
import time

# Set BCM mode
GPIO.setmode(GPIO.BCM)

# Configure GPIO 17 as output
GPIO.setup(17, GPIO.OUT)

# Toggle LED
try:
    while True:
        GPIO.output(17, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(17, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
```

## Advanced Topics

### Pulse Width Modulation (PWM)

PWM generates analog-like signals using digital outputs by rapidly switching between high and low states.

#### Software PWM

```c
// Simple software PWM (blocking)
void software_pwm(uint8_t duty_cycle) {  // 0-100%
    for (int i = 0; i < 100; i++) {
        if (i < duty_cycle) {
            GPIO_SET(LED_PIN);
        } else {
            GPIO_CLEAR(LED_PIN);
        }
        delay_us(10);  // 10us per step = 1ms period = 1kHz
    }
}
```

#### Hardware PWM (STM32)

```c
void pwm_timer_init(void) {
    // Enable timer and GPIO clocks
    __HAL_RCC_TIM2_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO for alternate function (PWM)
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure timer for PWM
    TIM_HandleTypeDef htim2;
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 84 - 1;  // For 1MHz timer clock
    htim2.Init.Period = 1000 - 1;   // For 1kHz PWM
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    HAL_TIM_PWM_Init(&htim2);

    // Configure PWM channel
    TIM_OC_InitTypeDef sConfigOC = {0};
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 500;  // 50% duty cycle
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1);

    // Start PWM
    HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1);
}

// Change duty cycle
void set_pwm_duty(uint16_t duty) {
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, duty);
}
```

### Analog-to-Digital Conversion (ADC)

Many GPIO pins can be configured for analog input:

```c
// ESP32 ADC example
#include "driver/adc.h"

void adc_init(void) {
    // Configure ADC
    adc1_config_width(ADC_WIDTH_BIT_12);  // 12-bit resolution
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);  // GPIO36

    // Read ADC value
    int adc_value = adc1_get_raw(ADC1_CHANNEL_0);  // 0-4095
    float voltage = (adc_value / 4095.0) * 3.3;    // Convert to voltage
}
```

### External Interrupts

Interrupts allow immediate response to GPIO events without polling:

```c
// AVR (Arduino) interrupt example
volatile bool button_pressed = false;

void setup() {
    pinMode(2, INPUT_PULLUP);  // INT0 on pin 2
    attachInterrupt(digitalPinToInterrupt(2), button_isr, FALLING);
}

void button_isr() {
    button_pressed = true;  // Set flag (keep ISR short!)
}

void loop() {
    if (button_pressed) {
        button_pressed = false;
        // Handle button press
    }
}
```

### Alternate Function Modes

GPIO pins often support multiple functions:

```c
// STM32 - Configure GPIO for UART
void uart_gpio_init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // USART2 TX (PA2)
    GPIO_InitStruct.Pin = GPIO_PIN_2;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // USART2 RX (PA3)
    GPIO_InitStruct.Pin = GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

## Best Practices and Common Gotchas

### Electrical Considerations

#### Current Limits
- **Output Current**: Most GPIO pins can source/sink 4-25mA per pin
- **Total Current**: Check datasheet for maximum total current per port
- **Example**: STM32F4 can handle 25mA per pin, 120mA total per port
- **Use external drivers** for loads >10mA (relays, motors, high-power LEDs)

```c
// BAD: Directly driving high-current load
GPIO_SET(MOTOR_PIN);  // May damage GPIO!

// GOOD: Using transistor/MOSFET driver
GPIO_SET(MOTOR_DRIVER_PIN);  // Controls transistor which drives motor
```

#### Voltage Levels
- **5V Tolerant**: Check if GPIO is 5V tolerant before connecting 5V signals
- **Level Shifting**: Use level shifters when interfacing different voltage domains
- **Example**: Connecting 5V Arduino to 3.3V ESP32 requires level shifter

### Common Pitfalls

#### 1. Floating Inputs

```c
// BAD: Input without pull resistor
pinMode(BUTTON_PIN, INPUT);  // Pin floats, reads random values

// GOOD: Use internal pull-up
pinMode(BUTTON_PIN, INPUT_PULLUP);  // Stable high when button not pressed
```

#### 2. Forgotten Clock Enable

```c
// BAD: Configuring GPIO without enabling clock
GPIOA->MODER |= ...;  // Won't work!

// GOOD: Enable clock first
RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;  // Enable GPIOA clock
GPIOA->MODER |= ...;  // Now it works
```

#### 3. Race Conditions in ISRs

```c
// BAD: Complex processing in ISR
void gpio_isr() {
    process_sensor_data();  // Too slow!
    send_network_packet();  // Blocks other interrupts
}

// GOOD: Set flag and process in main loop
volatile bool data_ready = false;

void gpio_isr() {
    data_ready = true;  // Quick flag set
}

void loop() {
    if (data_ready) {
        data_ready = false;
        process_sensor_data();
    }
}
```

#### 4. Not Using Volatile for Shared Variables

```c
// BAD: Compiler may optimize away checks
bool flag = false;

void isr() {
    flag = true;
}

void loop() {
    while (!flag) { }  // May loop forever due to optimization!
}

// GOOD: Use volatile
volatile bool flag = false;
```

### Debouncing Strategies

#### Hardware Debouncing
- Add RC filter: 10kΩ resistor + 0.1µF capacitor
- Use Schmitt trigger buffer (e.g., 74HC14)

#### Software Debouncing

```c
#define DEBOUNCE_TIME_MS 50

bool read_button_debounced(uint8_t pin) {
    static uint32_t last_change_time = 0;
    static bool last_state = false;

    bool current_state = GPIO_READ(pin);
    uint32_t current_time = millis();

    if (current_state != last_state) {
        last_change_time = current_time;
        last_state = current_state;
    }

    if ((current_time - last_change_time) > DEBOUNCE_TIME_MS) {
        return current_state;
    }

    return last_state;
}
```

### Proper Initialization Sequence

```c
// Correct GPIO initialization order
void gpio_init_proper(void) {
    // 1. Enable clocks
    enable_peripheral_clock();

    // 2. Configure pin mode/direction
    configure_pin_mode();

    // 3. Configure pull-up/pull-down
    configure_pull_resistors();

    // 4. Configure output type (PP/OD)
    configure_output_type();

    // 5. Configure speed/slew rate
    configure_speed();

    // 6. Set initial output state (before enabling output!)
    set_initial_state();

    // 7. Enable interrupts if needed
    configure_interrupts();
}
```

### ESD Protection

- Use external protection diodes for exposed connectors
- Add series resistors (100Ω-1kΩ) to limit current
- Keep PCB traces short to minimize antenna effect
- Use proper grounding and shielding

### Drive Strength Considerations

```c
// Low speed for most applications (reduces EMI)
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

// High speed only when necessary (SPI, fast signals)
GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;  // Use sparingly
```

### Power Consumption

```c
// Disable unused GPIO to save power
void gpio_low_power_config(void) {
    // Configure unused pins as analog input (lowest power)
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;

    // Or enable pull-down to prevent floating
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLDOWN;
}
```
