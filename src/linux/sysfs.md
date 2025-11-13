# sysfs

sysfs is a virtual filesystem that exports information about kernel subsystems, hardware devices, and associated device drivers to userspace.

## Overview

sysfs is mounted at `/sys` and provides:
- Device information
- Driver parameters
- Kernel configuration
- Power management settings

## Structure

```bash
/sys/
├── block/          # Block devices
├── bus/            # Bus types (pci, usb, etc.)
├── class/          # Device classes (network, input, etc.)
├── devices/        # Device tree
├── firmware/       # Firmware information
├── fs/             # Filesystem information
├── kernel/         # Kernel parameters
├── module/         # Loaded kernel modules
└── power/          # Power management
```

## Common Usage

```bash
# List block devices
ls /sys/block/

# Device information
cat /sys/class/net/eth0/address      # MAC address
cat /sys/class/net/eth0/speed        # Link speed
cat /sys/class/net/eth0/operstate    # Interface state

# CPU information
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
ls /sys/devices/system/cpu/cpu*/topology/

# GPU information
cat /sys/class/drm/card0/device/vendor
cat /sys/class/drm/card0/device/device

# USB devices
ls /sys/bus/usb/devices/

# Module parameters
ls /sys/module/*/parameters/
cat /sys/module/bluetooth/parameters/disable_esco
```

## Power Management

```bash
# CPU frequency scaling
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Device power state
cat /sys/class/net/eth0/device/power/runtime_status

# Display brightness
echo 50 | sudo tee /sys/class/backlight/*/brightness
```

## LED Control

```bash
# List LEDs
ls /sys/class/leds/

# Control LED
echo 1 > /sys/class/leds/led0/brightness
echo 0 > /sys/class/leds/led0/brightness

# LED trigger
echo "heartbeat" > /sys/class/leds/led0/trigger
```

sysfs provides a unified interface for interacting with kernel and hardware information.
