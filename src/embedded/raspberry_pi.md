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
