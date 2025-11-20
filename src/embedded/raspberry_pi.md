# Raspberry Pi

Complete guide to Raspberry Pi setup, GPIO programming, and projects.

## Table of Contents

1. [Introduction](#introduction)
2. [Hardware Overview](#hardware-overview)
3. [Setup and Installation](#setup-and-installation)
4. [GPIO Programming](#gpio-programming)
5. [Python Programming](#python-programming)
6. [C/C++ Programming](#cc-programming)
7. [Interfaces](#interfaces)
8. [Projects](#projects)

## Introduction

The Raspberry Pi is a series of small single-board computers developed by the Raspberry Pi Foundation. Unlike microcontrollers, it runs a full Linux operating system and can function as a complete desktop computer.

### Key Features

- **Full Linux Operating System** (Raspberry Pi OS based on Debian)
- **High Processing Power**: Multi-core ARM processors
- **Rich Connectivity**: USB, Ethernet, WiFi, Bluetooth, HDMI
- **GPIO Interface**: 40-pin header for hardware projects
- **Programming**: Python, C/C++, JavaScript, and more
- **Price**: $15-$75 depending on model

### Model Comparison

| Model | Processor | RAM | USB | Ethernet | WiFi/BT | GPIO | Price |
|-------|-----------|-----|-----|----------|---------|------|-------|
| **Pi Zero W** | Single 1GHz | 512MB | 1 micro | No | Yes | 40 | $15 |
| **Pi 3 B+** | Quad 1.4GHz | 1GB | 4 | Gigabit | Yes | 40 | $35 |
| **Pi 4 B** | Quad 1.5GHz | 2-8GB | 4 | Gigabit | Yes | 40 | $35-75 |
| **Pi 5** | Quad 2.4GHz | 4-8GB | 4 | Gigabit | Yes | 40 | $60-80 |
| **Pico** | Dual RP2040 | 264KB | 1 micro | No | No | 26 | $4 |

## Hardware Overview

### Raspberry Pi 4 Board Layout

```
┌────────────────────────────────────────────────────────┐
│  USB-C Power                        ┌──────────────┐  │
│     ┌─┐                             │   Ethernet   │  │
│     └─┘                             │   Port       │  │
│                                     └──────────────┘  │
│  ┌────────┐  ┌────────┐                              │
│  │  USB   │  │  USB   │  ┌──────────────┐           │
│  │  2.0   │  │  3.0   │  │   Dual HDMI  │           │
│  └────────┘  └────────┘  └──────────────┘           │
│                                                        │
│  ┌────────────────────┐         ┌──────┐            │
│  │   BCM2711 SoC      │         │Audio │            │
│  │   Quad Cortex-A72  │         │Jack  │            │
│  └────────────────────┘         └──────┘            │
│                                                        │
│  ┌──────────────┐  ┌────────────────────────────┐   │
│  │   Micro SD   │  │     40-pin GPIO Header     │   │
│  │   Card Slot  │  │                            │   │
│  └──────────────┘  └────────────────────────────┘   │
│                                                        │
│  [CSI Camera]           [DSI Display]                 │
└────────────────────────────────────────────────────────┘
```

### GPIO Pinout (40-pin Header)

```
           3V3  (1)  (2)  5V
     GPIO 2/SDA  (3)  (4)  5V
     GPIO 3/SCL  (5)  (6)  GND
         GPIO 4  (7)  (8)  GPIO 14/TXD
            GND  (9) (10)  GPIO 15/RXD
        GPIO 17 (11) (12)  GPIO 18/PWM
        GPIO 27 (13) (14)  GND
        GPIO 22 (15) (16)  GPIO 23
           3V3 (17) (18)  GPIO 24
  GPIO 10/MOSI (19) (20)  GND
   GPIO 9/MISO (21) (22)  GPIO 25
  GPIO 11/SCLK (23) (24)  GPIO 8/CE0
            GND (25) (26)  GPIO 7/CE1
        GPIO 0  (27) (28)  GPIO 1
        GPIO 5  (29) (30)  GND
        GPIO 6  (31) (32)  GPIO 12/PWM
   GPIO 13/PWM (33) (34)  GND
   GPIO 19/PWM (35) (36)  GPIO 16
        GPIO 26 (37) (38)  GPIO 20
            GND (39) (40)  GPIO 21

Power Pins: 3.3V (17mA max per pin), 5V (from USB)
PWM: GPIO 12, 13, 18, 19
SPI0: MOSI(10), MISO(9), SCLK(11), CE0(8), CE1(7)
I2C1: SDA(2), SCL(3)
UART: TXD(14), RXD(15)
```

## Setup and Installation

### Initial Setup

#### 1. Download Raspberry Pi OS

```bash
# Download Raspberry Pi Imager
# For Ubuntu/Debian:
sudo apt install rpi-imager

# For other systems, download from:
# https://www.raspberrypi.com/software/
```

#### 2. Flash SD Card

```bash
# Using Raspberry Pi Imager (GUI):
# 1. Choose OS: Raspberry Pi OS (32-bit/64-bit)
# 2. Choose SD card
# 3. Click "Write"

# Or using command line (Linux):
sudo dd if=2023-05-03-raspios-bullseye-armhf.img of=/dev/sdX bs=4M status=progress
sync
```

#### 3. Enable SSH (Headless Setup)

```bash
# Create empty 'ssh' file in boot partition
touch /media/username/boot/ssh

# Configure WiFi (optional)
cat > /media/username/boot/wpa_supplicant.conf << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YourNetworkName"
    psk="YourPassword"
    key_mgmt=WPA-PSK
}
EOF
```

#### 4. First Boot and Configuration

```bash
# Find your Pi on the network
ping raspberrypi.local

# Or use nmap to scan network
nmap -sn 192.168.1.0/24

# SSH into your Pi (default password: raspberry)
ssh pi@raspberrypi.local

# Run configuration tool
sudo raspi-config

# Essential configurations:
# 1. Change password (System Options -> Password)
# 2. Configure WiFi (System Options -> Wireless LAN)
# 3. Enable interfaces (Interface Options -> SSH, SPI, I2C, etc.)
# 4. Expand filesystem (Advanced Options -> Expand Filesystem)
# 5. Update system (Advanced Options -> Update)
```

#### 5. System Update

```bash
# Update package list
sudo apt update

# Upgrade all packages
sudo apt full-upgrade -y

# Install essential tools
sudo apt install -y vim git python3-pip python3-dev build-essential

# Update firmware (optional)
sudo rpi-update

# Reboot
sudo reboot
```

### Package Management

```bash
# Search for packages
apt search <package-name>

# Install package
sudo apt install <package-name>

# Remove package
sudo apt remove <package-name>

# Clean up
sudo apt autoremove
sudo apt clean
```

## GPIO Programming

### GPIO Basics

The Raspberry Pi has 40 GPIO pins that can be programmed for digital input/output, PWM, SPI, I2C, UART, and more.

**Important Safety Rules:**
- Maximum current per GPIO pin: 16mA
- Total current for all GPIO pins: 50mA
- GPIO pins are **3.3V** (not 5V tolerant!)
- Always use current-limiting resistors with LEDs
- Use level shifters for 5V devices

### GPIO Numbering

There are two numbering systems:

1. **BCM (Broadcom)**: Uses GPIO numbers (e.g., GPIO 17)
2. **BOARD**: Uses physical pin numbers (e.g., Pin 11)

```
BOARD Pin 11 = BCM GPIO 17
BOARD Pin 12 = BCM GPIO 18
```

### Pin Capabilities

```
Digital I/O:    All GPIO pins
PWM (Hardware): GPIO 12, 13, 18, 19 (4 channels)
PWM (Software): All GPIO pins
SPI:            GPIO 7-11 (SPI0), GPIO 16-21 (SPI1)
I2C:            GPIO 2-3 (I2C1), GPIO 0-1 (I2C0)
UART:           GPIO 14-15 (UART0)
```

## Python Programming

### Installing GPIO Libraries

```bash
# Install RPi.GPIO (traditional library)
sudo apt install python3-rpi.gpio

# Install gpiozero (modern, easier library)
sudo apt install python3-gpiozero

# Install pigpio (advanced features, better PWM)
sudo apt install python3-pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### RPi.GPIO Library

#### Basic LED Control

```python
import RPi.GPIO as GPIO
import time

# Set GPIO mode (BCM or BOARD)
GPIO.setmode(GPIO.BCM)

# Set GPIO warnings
GPIO.setwarnings(False)

# Define LED pin
LED_PIN = 17

# Setup pin as output
GPIO.setup(LED_PIN, GPIO.OUT)

# Turn LED on
GPIO.output(LED_PIN, GPIO.HIGH)
time.sleep(1)

# Turn LED off
GPIO.output(LED_PIN, GPIO.LOW)

# Clean up
GPIO.cleanup()
```

#### LED Blinking

```python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 17
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(0.5)
except KeyboardInterrupt:
    GPIO.cleanup()
```

#### Button Input

```python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

BUTTON_PIN = 27
LED_PIN = 17

# Setup button with pull-up resistor
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        # Button pressed when LOW (pull-up resistor)
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("Button pressed!")
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(0.1)
except KeyboardInterrupt:
    GPIO.cleanup()
```

#### Interrupt-Based Button

```python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 27
LED_PIN = 17

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT)

led_state = False

def button_callback(channel):
    global led_state
    led_state = not led_state
    GPIO.output(LED_PIN, led_state)
    print(f"LED {'ON' if led_state else 'OFF'}")

# Add event detection
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING,
                     callback=button_callback,
                     bouncetime=200)

try:
    print("Press button to toggle LED. Ctrl+C to exit.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
```

#### PWM (Pulse Width Modulation)

```python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 18  # Use hardware PWM pin

GPIO.setup(LED_PIN, GPIO.OUT)

# Create PWM instance (pin, frequency_hz)
pwm = GPIO.PWM(LED_PIN, 1000)

# Start PWM with 0% duty cycle
pwm.start(0)

try:
    while True:
        # Fade in
        for duty in range(0, 101, 5):
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.05)

        # Fade out
        for duty in range(100, -1, -5):
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.05)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
```

### gpiozero Library (Recommended)

gpiozero provides a simpler, more intuitive API.

#### LED Control

```python
from gpiozero import LED
from time import sleep

led = LED(17)

# Simple on/off
led.on()
sleep(1)
led.off()

# Blinking
led.blink(on_time=1, off_time=1)

# Keep program running
sleep(10)
```

#### Button Input

```python
from gpiozero import LED, Button
from signal import pause

led = LED(17)
button = Button(27)

# Simple callback
button.when_pressed = led.on
button.when_released = led.off

pause()  # Keep program running
```

#### PWM LED

```python
from gpiozero import PWMLED
from time import sleep

led = PWMLED(18)

# Pulse LED
led.pulse(fade_in_time=1, fade_out_time=1)

sleep(10)
```

#### Multiple Components

```python
from gpiozero import LED, Button, Buzzer
from signal import pause

red_led = LED(17)
green_led = LED(27)
button = Button(22)
buzzer = Buzzer(23)

def button_pressed():
    red_led.off()
    green_led.on()
    buzzer.beep(on_time=0.1, off_time=0.1, n=3)

def button_released():
    green_led.off()
    red_led.on()

button.when_pressed = button_pressed
button.when_released = button_released

red_led.on()  # Initial state
pause()
```

### pigpio Library (Advanced)

pigpio provides better hardware-timed PWM and servo control.

```python
import pigpio
import time

# Connect to pigpio daemon
pi = pigpio.pi()

LED_PIN = 18

# Set PWM (0-255)
pi.set_PWM_dutycycle(LED_PIN, 128)  # 50% brightness

# Fade LED
for i in range(256):
    pi.set_PWM_dutycycle(LED_PIN, i)
    time.sleep(0.01)

# Servo control (500-2500 microseconds)
SERVO_PIN = 12
pi.set_servo_pulsewidth(SERVO_PIN, 1500)  # Center position

# Clean up
pi.stop()
```

## C/C++ Programming

### Using WiringPi

```bash
# Install WiringPi
sudo apt install wiringpi

# Verify installation
gpio -v
gpio readall
```

#### Basic LED (C)

```c
#include <wiringPi.h>
#include <stdio.h>

#define LED_PIN 0  // WiringPi pin 0 = BCM GPIO 17

int main(void) {
    // Initialize wiringPi
    if (wiringPiSetup() == -1) {
        printf("Setup failed!\n");
        return 1;
    }

    // Set pin as output
    pinMode(LED_PIN, OUTPUT);

    // Blink LED
    for (int i = 0; i < 10; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(500);
        digitalWrite(LED_PIN, LOW);
        delay(500);
    }

    return 0;
}
```

Compile and run:
```bash
gcc -o led led.c -lwiringPi
sudo ./led
```

#### Button Input (C)

```c
#include <wiringPi.h>
#include <stdio.h>

#define BUTTON_PIN 2  // WiringPi pin 2 = BCM GPIO 27
#define LED_PIN 0

int main(void) {
    wiringPiSetup();

    pinMode(BUTTON_PIN, INPUT);
    pullUpDnControl(BUTTON_PIN, PUD_UP);
    pinMode(LED_PIN, OUTPUT);

    printf("Press button to control LED. Ctrl+C to exit.\n");

    while (1) {
        if (digitalRead(BUTTON_PIN) == LOW) {
            digitalWrite(LED_PIN, HIGH);
        } else {
            digitalWrite(LED_PIN, LOW);
        }
        delay(10);
    }

    return 0;
}
```

#### PWM (C)

```c
#include <wiringPi.h>
#include <stdio.h>

#define PWM_PIN 1  // WiringPi pin 1 = BCM GPIO 18

int main(void) {
    wiringPiSetup();

    pinMode(PWM_PIN, PWM_OUTPUT);

    // Fade in and out
    while (1) {
        // Fade in
        for (int i = 0; i <= 1024; i++) {
            pwmWrite(PWM_PIN, i);
            delay(2);
        }

        // Fade out
        for (int i = 1024; i >= 0; i--) {
            pwmWrite(PWM_PIN, i);
            delay(2);
        }
    }

    return 0;
}
```

### Using pigpio in C

```c
#include <pigpio.h>
#include <stdio.h>

#define LED_PIN 17

int main(void) {
    if (gpioInitialise() < 0) {
        printf("pigpio initialization failed\n");
        return 1;
    }

    gpioSetMode(LED_PIN, PI_OUTPUT);

    // Blink
    for (int i = 0; i < 10; i++) {
        gpioWrite(LED_PIN, 1);
        gpioDelay(500000);  // microseconds
        gpioWrite(LED_PIN, 0);
        gpioDelay(500000);
    }

    gpioTerminate();
    return 0;
}
```

Compile:
```bash
gcc -o led led.c -lpigpio -lrt -lpthread
sudo ./led
```

## Interfaces

### I2C (Inter-Integrated Circuit)

#### Enable I2C

```bash
# Enable via raspi-config
sudo raspi-config
# Interface Options -> I2C -> Yes

# Or edit config file
echo "dtparam=i2c_arm=on" | sudo tee -a /boot/config.txt
sudo reboot

# Install I2C tools
sudo apt install i2c-tools python3-smbus

# Detect I2C devices
i2cdetect -y 1
```

#### Python I2C Example (with smbus2)

```bash
pip3 install smbus2
```

```python
from smbus2 import SMBus
import time

# I2C address of device (e.g., 0x48 for ADS1115)
DEVICE_ADDRESS = 0x48

# I2C bus (1 for Pi 2/3/4, 0 for very old Pi)
bus = SMBus(1)

# Write byte
bus.write_byte_data(DEVICE_ADDRESS, 0x01, 0x00)

# Read byte
data = bus.read_byte_data(DEVICE_ADDRESS, 0x00)
print(f"Read: {data}")

# Read block of data
block = bus.read_i2c_block_data(DEVICE_ADDRESS, 0x00, 16)

bus.close()
```

#### I2C OLED Display Example

```bash
pip3 install adafruit-circuitpython-ssd1306 pillow
```

```python
import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306

# Create I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create display object (128x64 pixels)
display = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C)

# Clear display
display.fill(0)
display.show()

# Create image
image = Image.new("1", (display.width, display.height))
draw = ImageDraw.Draw(image)

# Draw text
draw.text((0, 0), "Hello, Pi!", fill=255)
draw.rectangle((0, 20, 128, 40), outline=255, fill=0)

# Display image
display.image(image)
display.show()
```

### SPI (Serial Peripheral Interface)

#### Enable SPI

```bash
# Enable via raspi-config
sudo raspi-config
# Interface Options -> SPI -> Yes

# Or edit config file
echo "dtparam=spi=on" | sudo tee -a /boot/config.txt
sudo reboot

# Install SPI tools
sudo apt install python3-spidev
```

#### Python SPI Example

```python
import spidev
import time

# Create SPI object
spi = spidev.SpiDev()

# Open SPI bus 0, device (CS) 0
spi.open(0, 0)

# Set SPI speed and mode
spi.max_speed_hz = 1000000
spi.mode = 0

# Send and receive data
data_out = [0x01, 0x02, 0x03]
data_in = spi.xfer2(data_out)

print(f"Sent: {data_out}")
print(f"Received: {data_in}")

spi.close()
```

#### MCP3008 ADC Example (SPI)

```python
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    """Read MCP3008 ADC channel (0-7)"""
    if channel < 0 or channel > 7:
        return -1

    # MCP3008 protocol
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

try:
    while True:
        value = read_adc(0)
        voltage = (value * 3.3) / 1024
        print(f"ADC: {value}, Voltage: {voltage:.2f}V")
        time.sleep(0.5)
except KeyboardInterrupt:
    spi.close()
```

### UART (Serial Communication)

#### Enable UART

```bash
# Disable serial console
sudo raspi-config
# Interface Options -> Serial Port
# Login shell: No
# Serial port hardware: Yes

# Edit config
sudo nano /boot/config.txt
# Add: enable_uart=1

sudo reboot

# Install pyserial
pip3 install pyserial
```

#### Python UART Example

```python
import serial
import time

# Open serial port
ser = serial.Serial(
    port='/dev/serial0',  # or /dev/ttyAMA0
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

# Write data
ser.write(b'Hello UART\n')

# Read data
while True:
    if ser.in_waiting > 0:
        data = ser.readline()
        print(f"Received: {data.decode().strip()}")
    time.sleep(0.1)

ser.close()
```

### 1-Wire (DS18B20 Temperature Sensor)

#### Enable 1-Wire

```bash
# Edit config
echo "dtoverlay=w1-gpio" | sudo tee -a /boot/config.txt
sudo reboot

# Load modules
sudo modprobe w1-gpio
sudo modprobe w1-therm

# Find sensor
ls /sys/bus/w1/devices/
# Look for 28-xxxxxxxxxxxx
```

#### Read Temperature

```python
import time

def read_temp():
    """Read DS18B20 temperature sensor"""
    # Replace with your sensor ID
    device_file = '/sys/bus/w1/devices/28-xxxxxxxxxxxx/w1_slave'

    with open(device_file, 'r') as f:
        lines = f.readlines()

    # Check if read was successful
    if lines[0].strip()[-3:] != 'YES':
        return None

    # Extract temperature
    temp_pos = lines[1].find('t=')
    if temp_pos != -1:
        temp_string = lines[1][temp_pos+2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c

    return None

while True:
    temp = read_temp()
    if temp is not None:
        print(f"Temperature: {temp:.2f}°C ({temp * 9/5 + 32:.2f}°F)")
    time.sleep(1)
```

## Projects

### 1. LED Traffic Light

```python
from gpiozero import LED
from time import sleep

red = LED(17)
yellow = LED(27)
green = LED(22)

while True:
    green.on()
    sleep(3)
    green.off()

    yellow.on()
    sleep(1)
    yellow.off()

    red.on()
    sleep(3)
    red.off()
```

**Wiring:**
- Red LED → GPIO 17 → 220Ω resistor → GND
- Yellow LED → GPIO 27 → 220Ω resistor → GND
- Green LED → GPIO 22 → 220Ω resistor → GND

### 2. DHT22 Temperature/Humidity Monitor

```bash
pip3 install adafruit-circuitpython-dht
sudo apt install libgpiod2
```

```python
import time
import board
import adafruit_dht

# Initialize DHT22 sensor on GPIO 4
dht = adafruit_dht.DHT22(board.D4)

while True:
    try:
        temperature = dht.temperature
        humidity = dht.humidity

        print(f"Temp: {temperature:.1f}°C")
        print(f"Humidity: {humidity:.1f}%")
        print()

    except RuntimeError as e:
        print(f"Error: {e}")

    time.sleep(2)
```

### 3. LCD Display (16x2 I2C)

```bash
pip3 install RPLCD smbus2
```

```python
from RPLCD.i2c import CharLCD
import time

# Create LCD object (I2C address usually 0x27 or 0x3F)
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

lcd.clear()
lcd.write_string('Hello, World!')

time.sleep(2)

lcd.clear()
lcd.cursor_pos = (0, 0)
lcd.write_string('Line 1')
lcd.cursor_pos = (1, 0)
lcd.write_string('Line 2')
```

### 4. Motion Sensor (PIR) Security System

```python
from gpiozero import MotionSensor, LED, Buzzer
from signal import pause
import datetime

pir = MotionSensor(4)
led = LED(17)
buzzer = Buzzer(27)

def motion_detected():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] MOTION DETECTED!")
    led.on()
    buzzer.beep(on_time=0.1, off_time=0.1, n=5)

def motion_stopped():
    print("Motion stopped")
    led.off()

pir.when_motion = motion_detected
pir.when_no_motion = motion_stopped

print("PIR security system started...")
pause()
```

### 5. Web Server GPIO Control

```bash
pip3 install flask
```

```python
from flask import Flask, render_template, request
from gpiozero import LED

app = Flask(__name__)
led = LED(17)

@app.route('/')
def index():
    status = "ON" if led.is_lit else "OFF"
    return f'''
        <h1>Raspberry Pi LED Control</h1>
        <p>LED Status: {status}</p>
        <form method="post" action="/toggle">
            <button type="submit">Toggle LED</button>
        </form>
    '''

@app.route('/toggle', methods=['POST'])
def toggle():
    led.toggle()
    return index()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Access at `http://raspberrypi.local:5000`

### 6. Ultrasonic Distance Sensor (HC-SR04)

```python
from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(echo=24, trigger=23, max_distance=4)

while True:
    distance = sensor.distance * 100  # Convert to cm
    print(f"Distance: {distance:.1f} cm")
    sleep(0.1)
```

### 7. Servo Motor Control

```python
from gpiozero import Servo
from time import sleep

servo = Servo(17)

# Move through positions
positions = [-1, -0.5, 0, 0.5, 1]  # -1 to 1

for pos in positions:
    servo.value = pos
    print(f"Position: {pos}")
    sleep(1)

# Sweep back and forth
while True:
    servo.min()
    sleep(1)
    servo.mid()
    sleep(1)
    servo.max()
    sleep(1)
```

### 8. RGB LED Control

```python
from gpiozero import RGBLED
from time import sleep

rgb = RGBLED(red=17, green=27, blue=22)

# Predefined colors
rgb.red = 1      # Red
sleep(1)
rgb.green = 1    # Yellow (red + green)
sleep(1)
rgb.red = 0      # Green
sleep(1)
rgb.blue = 1     # Cyan (green + blue)
sleep(1)
rgb.green = 0    # Blue
sleep(1)
rgb.red = 1      # Magenta (red + blue)
sleep(1)

# Custom colors (RGB values 0-1)
rgb.color = (1, 0.5, 0)    # Orange
sleep(1)
rgb.color = (0.5, 0, 0.5)  # Purple

# Cycle through colors
while True:
    rgb.pulse(fade_in_time=1, fade_out_time=1, on_color=(1, 0, 0))
    sleep(2)
```

### 9. Rotary Encoder

```python
from gpiozero import RotaryEncoder
from signal import pause

encoder = RotaryEncoder(a=17, b=18, max_steps=100)

def rotated():
    print(f"Position: {encoder.steps}")

encoder.when_rotated = rotated

print("Rotate the encoder...")
pause()
```

### 10. MQTT IoT Publisher

```bash
pip3 install paho-mqtt
```

```python
import paho.mqtt.client as mqtt
import time
import random

# MQTT broker settings
broker = "mqtt.eclipseprojects.io"
port = 1883
topic = "home/sensors/temperature"

client = mqtt.Client("RaspberryPi_Sensor")
client.connect(broker, port)

print(f"Publishing to {topic}...")

try:
    while True:
        # Simulate temperature reading
        temp = random.uniform(20.0, 30.0)

        message = f"{temp:.2f}"
        client.publish(topic, message)
        print(f"Published: {message}°C")

        time.sleep(5)
except KeyboardInterrupt:
    client.disconnect()
```

## Advanced Topics

### Camera Module

```bash
# Enable camera
sudo raspi-config
# Interface Options -> Camera -> Yes

# Install picamera2 (for Pi OS Bullseye+)
sudo apt install python3-picamera2

# Or legacy picamera
pip3 install picamera
```

#### Take Photo

```python
from picamera2 import Picamera2
import time

camera = Picamera2()

# Configure camera
config = camera.create_still_configuration()
camera.configure(config)

# Start camera
camera.start()
time.sleep(2)  # Allow camera to adjust

# Capture image
camera.capture_file("photo.jpg")

camera.stop()
print("Photo saved!")
```

#### Record Video

```python
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

camera = Picamera2()
encoder = H264Encoder()

# Start recording
camera.start_recording(encoder, "video.h264")
time.sleep(10)  # Record for 10 seconds
camera.stop_recording()
```

### systemd Service

Create a service to run your script on boot:

```bash
# Create service file
sudo nano /etc/systemd/system/myproject.service
```

```ini
[Unit]
Description=My Raspberry Pi Project
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/myproject
ExecStart=/usr/bin/python3 /home/pi/myproject/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable myproject.service
sudo systemctl start myproject.service

# Check status
sudo systemctl status myproject.service

# View logs
sudo journalctl -u myproject.service -f
```

### Performance Tips

```bash
# Check CPU temperature
vcgencmd measure_temp

# Check CPU frequency
vcgencmd measure_clock arm

# Monitor system resources
htop

# Overclock (edit /boot/config.txt)
# WARNING: May void warranty
over_voltage=6
arm_freq=2000

# Disable GUI for better performance
sudo systemctl set-default multi-target

# Re-enable GUI
sudo systemctl set-default graphical.target
```

### Backup and Restore

```bash
# Backup SD card (on Linux host)
sudo dd if=/dev/sdX of=pi_backup.img bs=4M status=progress

# Compress backup
gzip pi_backup.img

# Restore backup
gunzip pi_backup.img.gz
sudo dd if=pi_backup.img of=/dev/sdX bs=4M status=progress
```

## Resources

### Official Documentation
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [Raspberry Pi OS](https://www.raspberrypi.com/software/)
- [GPIO Pinout](https://pinout.xyz/)

### Libraries
- [gpiozero Documentation](https://gpiozero.readthedocs.io/)
- [RPi.GPIO Documentation](https://sourceforge.net/p/raspberry-gpio-python/wiki/Home/)
- [pigpio](http://abyz.me.uk/rpi/pigpio/)

### Projects and Tutorials
- [Raspberry Pi Projects](https://projects.raspberrypi.org/)
- [Adafruit Learn](https://learn.adafruit.com/category/raspberry-pi)
- [The MagPi Magazine](https://magpi.raspberrypi.com/)

### Community
- [Raspberry Pi Forums](https://forums.raspberrypi.com/)
- [r/raspberry_pi](https://www.reddit.com/r/raspberry_pi/)
- [Stack Exchange](https://raspberrypi.stackexchange.com/)

## Troubleshooting

### Common Issues

**GPIO permissions:**
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi
sudo reboot
```

**I2C/SPI not working:**
```bash
# Check if enabled
ls /dev/i2c* /dev/spi*

# Re-enable
sudo raspi-config
```

**WiFi connection issues:**
```bash
# Scan networks
sudo iwlist wlan0 scan

# Restart networking
sudo systemctl restart networking

# Check status
ifconfig wlan0
```

**SD card corruption:**
```bash
# Check filesystem
sudo fsck /dev/mmcblk0p2

# Use quality SD cards (Class 10, A1/A2 rated)
```

**Power issues:**
```bash
# Check for undervoltage
vcgencmd get_throttled
# 0x0 = OK
# 0x50000 = Throttled due to undervoltage

# Use official 5V 3A power supply
```