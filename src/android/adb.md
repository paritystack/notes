# Android Debug Bridge (ADB)

## Overview

Android Debug Bridge (ADB) is a versatile command-line tool that lets you communicate with an Android device. ADB facilitates a variety of device actions, such as installing and debugging apps, and it provides access to a Unix shell that you can use to run various commands on a device.

ADB is included in the Android SDK Platform Tools package and can be used with physical devices connected via USB or with emulators.

## Installation

### Linux/Mac
```bash
# Install via Android SDK Platform Tools
# Or use package manager
sudo apt install adb  # Ubuntu/Debian
brew install android-platform-tools  # macOS

# Verify installation
adb version
```

### Windows
```bash
# Download Android SDK Platform Tools from:
# https://developer.android.com/studio/releases/platform-tools

# Add to PATH and verify
adb version
```

## Setup and Connection

### Enable Developer Options
1. Go to Settings > About Phone
2. Tap "Build Number" 7 times
3. Go back to Settings > Developer Options
4. Enable "USB Debugging"

### Connect Device via USB
```bash
# List connected devices
adb devices

# Output example:
# List of devices attached
# 1234567890ABCDEF    device
# emulator-5554       device

# Connect to specific device
adb -s 1234567890ABCDEF shell
```

### Connect Device via WiFi
```bash
# Connect device via USB first, then:
# Get device IP address
adb shell ip addr show wlan0

# Enable TCP/IP mode on port 5555
adb tcpip 5555

# Disconnect USB and connect via WiFi
adb connect 192.168.1.100:5555

# Verify connection
adb devices

# Disconnect
adb disconnect 192.168.1.100:5555

# Return to USB mode
adb usb
```

## Basic Commands

### Device Management
```bash
# List all connected devices
adb devices -l

# Start ADB server
adb start-server

# Kill ADB server
adb kill-server

# Restart ADB server
adb kill-server && adb start-server

# Wait for device to be connected
adb wait-for-device

# Get device state
adb get-state

# Get device serial number
adb get-serialno
```

### Device Information
```bash
# Get device model
adb shell getprop ro.product.model

# Get Android version
adb shell getprop ro.build.version.release

# Get device manufacturer
adb shell getprop ro.product.manufacturer

# Get device serial number
adb shell getprop ro.serialno

# Get device resolution
adb shell wm size

# Get device density
adb shell wm density

# Display all properties
adb shell getprop

# Get battery status
adb shell dumpsys battery

# Get CPU information
adb shell cat /proc/cpuinfo

# Get memory information
adb shell cat /proc/meminfo
```

## App Management

### Installing and Uninstalling Apps
```bash
# Install APK
adb install app.apk

# Install APK to specific location
adb install -s /sdcard/app.apk

# Reinstall existing app (keep data)
adb install -r app.apk

# Install APK to SD card
adb install -s app.apk

# Uninstall app
adb uninstall com.example.app

# Uninstall app but keep data
adb uninstall -k com.example.app
```

### Package Information
```bash
# List all packages
adb shell pm list packages

# List third-party packages
adb shell pm list packages -3

# List system packages
adb shell pm list packages -s

# Search for specific package
adb shell pm list packages | grep keyword

# Get path of installed package
adb shell pm path com.example.app

# Get app information
adb shell dumpsys package com.example.app

# Clear app data
adb shell pm clear com.example.app

# Enable/Disable app
adb shell pm enable com.example.app
adb shell pm disable com.example.app
```

### Running Apps
```bash
# Start an activity
adb shell am start -n com.example.app/.MainActivity

# Start activity with data
adb shell am start -a android.intent.action.VIEW -d "https://example.com"

# Start service
adb shell am startservice com.example.app/.MyService

# Broadcast intent
adb shell am broadcast -a android.intent.action.BOOT_COMPLETED

# Force stop app
adb shell am force-stop com.example.app

# Kill app process
adb shell am kill com.example.app
```

## File Operations

### Copying Files
```bash
# Copy file from device to computer
adb pull /sdcard/file.txt ~/Desktop/

# Copy file from computer to device
adb push ~/Desktop/file.txt /sdcard/

# Copy directory recursively
adb pull /sdcard/DCIM/ ~/Pictures/

# Copy with progress display
adb pull /sdcard/large_file.mp4 .

# Push multiple files
adb push file1.txt file2.txt /sdcard/
```

### File System Navigation
```bash
# Access device shell
adb shell

# Navigate directories (once in shell)
cd /sdcard
ls -la
pwd

# Create directory
adb shell mkdir /sdcard/NewFolder

# Remove file
adb shell rm /sdcard/file.txt

# Remove directory
adb shell rm -r /sdcard/OldFolder

# Change file permissions
adb shell chmod 777 /sdcard/file.txt

# View file contents
adb shell cat /sdcard/file.txt

# Search for files
adb shell find /sdcard -name "*.txt"
```

## Logging and Debugging

### Logcat
```bash
# View all logs
adb logcat

# Clear log buffer
adb logcat -c

# View logs with specific priority
adb logcat *:E  # Error
adb logcat *:W  # Warning
adb logcat *:I  # Info
adb logcat *:D  # Debug
adb logcat *:V  # Verbose

# Filter by tag
adb logcat -s MyApp

# Filter by multiple tags
adb logcat -s MyApp:D ActivityManager:W

# Save logs to file
adb logcat > logfile.txt

# View logs with timestamp
adb logcat -v time

# View logs in different formats
adb logcat -v brief
adb logcat -v process
adb logcat -v tag
adb logcat -v thread
adb logcat -v raw
adb logcat -v long

# Filter using grep
adb logcat | grep "keyword"

# View specific buffer
adb logcat -b radio   # Radio/telephony logs
adb logcat -b events  # Event logs
adb logcat -b main    # Main application logs
adb logcat -b system  # System logs
adb logcat -b crash   # Crash logs

# Continuous monitoring with color
adb logcat -v color
```

### Bug Reports
```bash
# Generate bug report
adb bugreport

# Save bug report to file
adb bugreport > bugreport.txt

# Generate zipped bug report (Android 7.0+)
adb bugreport bugreport.zip
```

## Screen Control

### Screenshots and Screen Recording
```bash
# Take screenshot
adb shell screencap /sdcard/screenshot.png
adb pull /sdcard/screenshot.png

# Take screenshot (one command)
adb exec-out screencap -p > screenshot.png

# Record screen (Ctrl+C to stop)
adb shell screenrecord /sdcard/demo.mp4

# Record with time limit (max 180 seconds)
adb shell screenrecord --time-limit 30 /sdcard/demo.mp4

# Record with specific size
adb shell screenrecord --size 1280x720 /sdcard/demo.mp4

# Record with specific bitrate
adb shell screenrecord --bit-rate 6000000 /sdcard/demo.mp4

# Pull recorded video
adb pull /sdcard/demo.mp4
```

### Screen Input
```bash
# Tap at coordinates (x, y)
adb shell input tap 500 1000

# Swipe from (x1,y1) to (x2,y2) over duration ms
adb shell input swipe 500 1000 500 200 300

# Type text
adb shell input text "Hello%sWorld"  # %s represents space

# Press key
adb shell input keyevent KEYCODE_HOME
adb shell input keyevent KEYCODE_BACK
adb shell input keyevent KEYCODE_MENU
adb shell input keyevent 3  # Home key (key code)

# Common key codes
# KEYCODE_HOME = 3
# KEYCODE_BACK = 4
# KEYCODE_MENU = 82
# KEYCODE_POWER = 26
# KEYCODE_VOLUME_UP = 24
# KEYCODE_VOLUME_DOWN = 25
```

## System Control

### Power Management
```bash
# Reboot device
adb reboot

# Reboot to recovery mode
adb reboot recovery

# Reboot to bootloader
adb reboot bootloader

# Shutdown device (requires root)
adb shell reboot -p

# Wake up screen
adb shell input keyevent KEYCODE_WAKEUP

# Sleep screen
adb shell input keyevent KEYCODE_SLEEP
```

### Network
```bash
# Check WiFi status
adb shell dumpsys wifi

# Enable WiFi
adb shell svc wifi enable

# Disable WiFi
adb shell svc wifi disable

# Check network connectivity
adb shell ping -c 4 google.com

# Get IP address
adb shell ip addr show wlan0

# Enable/Disable data
adb shell svc data enable
adb shell svc data disable
```

### Settings
```bash
# Get setting value
adb shell settings get system screen_brightness

# Set setting value
adb shell settings put system screen_brightness 100

# Common settings namespaces:
# - system: User preferences
# - secure: Secure system settings
# - global: Device-wide settings

# Enable airplane mode
adb shell settings put global airplane_mode_on 1
adb shell am broadcast -a android.intent.action.AIRPLANE_MODE

# Disable animations (for testing)
adb shell settings put global window_animation_scale 0
adb shell settings put global transition_animation_scale 0
adb shell settings put global animator_duration_scale 0
```

## Advanced Commands

### Dumpsys
```bash
# Get system information
adb shell dumpsys

# Battery information
adb shell dumpsys battery

# Memory usage
adb shell dumpsys meminfo
adb shell dumpsys meminfo com.example.app

# CPU usage
adb shell dumpsys cpuinfo

# Display information
adb shell dumpsys display

# Activity information
adb shell dumpsys activity

# Current activity
adb shell dumpsys activity activities | grep mResumedActivity

# Package information
adb shell dumpsys package com.example.app

# Window information
adb shell dumpsys window
```

### Performance Monitoring
```bash
# Monitor CPU usage
adb shell top

# Monitor specific process
adb shell top | grep com.example.app

# Get process list
adb shell ps

# Get process info by name
adb shell ps | grep com.example.app

# Memory stats
adb shell procrank

# Disk usage
adb shell df

# Network statistics
adb shell netstat
```

### Database Operations
```bash
# Access app database (requires root or debuggable app)
adb shell run-as com.example.app

# Navigate to database directory
cd /data/data/com.example.app/databases/

# Pull database
adb exec-out run-as com.example.app cat databases/mydb.db > mydb.db

# Query database using sqlite3
adb shell "run-as com.example.app sqlite3 databases/mydb.db 'SELECT * FROM users;'"
```

## Testing and Automation

### Monkey Testing
```bash
# Generate random events
adb shell monkey -p com.example.app 1000

# Monkey with specific event types
adb shell monkey -p com.example.app --pct-touch 70 --pct-motion 30 1000

# Monkey with seed (reproducible)
adb shell monkey -p com.example.app -s 100 1000

# Throttle events (delay in ms)
adb shell monkey -p com.example.app --throttle 500 1000

# Ignore crashes and continue
adb shell monkey -p com.example.app --ignore-crashes 1000
```

### UI Automator
```bash
# Dump UI hierarchy
adb shell uiautomator dump

# Pull UI hierarchy XML
adb pull /sdcard/window_dump.xml

# Run UI Automator test
adb shell uiautomator runtest UiAutomatorTest.jar -c com.example.test.MyTest
```

## Scripting with ADB

### Batch Operations
```bash
#!/bin/bash

# Install app on all connected devices
for device in $(adb devices | grep -v "List" | awk '{print $1}'); do
    echo "Installing on device: $device"
    adb -s $device install app.apk
done

# Clear app data on all devices
for device in $(adb devices | grep -v "List" | awk '{print $1}'); do
    echo "Clearing data on device: $device"
    adb -s $device shell pm clear com.example.app
done
```

### Automated Screenshot Script
```bash
#!/bin/bash

# Take screenshot and save with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
filename="screenshot_${timestamp}.png"

adb exec-out screencap -p > "$filename"
echo "Screenshot saved: $filename"
```

### Log Filtering Script
```bash
#!/bin/bash

# Monitor logs for specific package
package="com.example.app"
adb logcat | grep --line-buffered "$package" | while read line; do
    echo "[$(date +"%H:%M:%S")] $line"
done
```

## Troubleshooting

### Common Issues

**Device not detected:**
```bash
# Check USB connection
lsusb  # Linux
system_profiler SPUSBDataType  # macOS

# Restart ADB
adb kill-server
adb start-server

# Check device authorization
# Accept the authorization prompt on device
```

**Permission denied:**
```bash
# Check USB debugging is enabled
# Revoke USB debugging authorizations and reconnect
# Settings > Developer Options > Revoke USB Debugging Authorizations

# Linux: Add udev rules
sudo vim /etc/udev/rules.d/51-android.rules
# Add: SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"
sudo udevadm control --reload-rules
```

**Multiple devices:**
```bash
# Specify device with -s flag
adb -s 1234567890ABCDEF shell

# Or use -d for physical device, -e for emulator
adb -d shell  # Physical device
adb -e shell  # Emulator
```

## Best Practices

1. Always specify device with `-s` when multiple devices are connected
2. Use `adb wait-for-device` in scripts before commands
3. Clear logcat before testing: `adb logcat -c`
4. Use appropriate log levels to reduce noise
5. Save important logs to files for later analysis
6. Be careful with `rm` commands - there's no undo
7. Test commands on emulator before using on physical device
8. Keep ADB updated with latest platform tools
9. Use `adb shell` for interactive sessions, direct commands for scripts
10. Always pull important data before performing system changes

## Security Considerations

- Disable USB debugging when not in development
- Be cautious when connecting to devices over WiFi
- Don't leave ADB over TCP/IP enabled on public networks
- Review USB debugging authorization requests carefully
- Use secure, trusted computers for ADB connections
- Never share bug reports publicly without reviewing contents first

## References

- [Official ADB Documentation](https://developer.android.com/studio/command-line/adb)
- [ADB Shell Commands](https://adbshell.com/)
- [Android Internals](internals.md)
- [Development Guide](development.md)

## Quick Reference Card

```bash
# Connection
adb devices                    # List devices
adb connect IP:5555           # Connect via WiFi

# Apps
adb install app.apk           # Install app
adb uninstall package.name    # Uninstall app
adb shell pm list packages    # List packages

# Files
adb push local remote         # Upload file
adb pull remote local         # Download file

# Shell
adb shell                     # Interactive shell
adb shell command             # Run single command

# Logs
adb logcat                    # View logs
adb logcat -c                 # Clear logs

# Screen
adb shell screencap /sdcard/s.png    # Screenshot
adb shell screenrecord /sdcard/v.mp4 # Record screen

# System
adb reboot                    # Reboot device
adb shell dumpsys battery     # Battery info
```
