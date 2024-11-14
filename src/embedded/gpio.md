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
