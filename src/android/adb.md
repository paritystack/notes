# Android Debug Bridge (ADB)

## Overview

Android Debug Bridge (ADB) is the command-line tool that drives a device or emulator from a
host: installing/debugging apps, pushing/pulling files, and dropping into a Unix shell. It's
the entry point for almost everything in this section — `adb shell dumpsys` inspects the
services in [SystemServer & Core Services](system_server.md), `adb logcat` and `dumpsys
gfxinfo` chase the jank described in [Graphics Stack](graphics_stack.md) and
[Performance & Profiling](performance_profiling.md), `adb install`/`pm` exercise the packages
from [APK/AAB Packaging & Signing](app_signing.md), and `adb reboot bootloader` is the gateway
to [Verified Boot & OTA](verified_boot_ota.md). It ships in the **SDK Platform Tools** package.

### Architecture: client, server, daemon

ADB is a three-part system. The command you type is a **client**; it talks to a **server**
(a background process on your host, default TCP port **5037**) which multiplexes connections to
the **`adbd`** daemon running on each device/emulator.

```text
   host                                   device / emulator
 ┌───────────────────────────┐          ┌────────────────────────┐
 │ adb (client, your shell)  │          │  adbd  (daemon)         │
 │        │                  │          │    │                   │
 │        ▼                  │  USB or  │    ▼                   │
 │ adb server  ──────────────┼── TCP ───┼─▶ shell / pm / am /    │
 │ (localhost:5037)          │          │    dumpsys / file sync  │
 └───────────────────────────┘          └────────────────────────┘
```

The first `adb` command silently starts the server. `adbd` runs as the `shell` user on
production builds (limited privileges) and only as `root` on `userdebug`/`eng` builds via
`adb root` — which is why some commands below note "requires root".

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

`pm` (package manager) and `am` (activity manager) are shells over the system services of the
same name — see [SystemServer & Core Services](system_server.md). Signing/packaging details for
the APKs you install are in [APK/AAB Packaging & Signing](app_signing.md).

### Installing and Uninstalling Apps
```bash
# Install APK
adb install app.apk

# Reinstall existing app, keeping its data
adb install -r app.apk

# Allow downgrade to a lower versionCode (debuggable builds)
adb install -d app.apk

# Install a split-APK set (one app, multiple APKs — e.g. an app bundle's output)
adb install-multiple base.apk config.arm64.apk config.xxhdpi.apk

# Uninstall app
adb uninstall com.example.app

# Uninstall app but keep data and cache
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

# Start a service (use start-foreground-service on Android 8+ for background-start limits)
adb shell am start-service com.example.app/.MyService

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

Logcat is the first stop for crashes and ANRs; pair it with `dumpsys gfxinfo` and Perfetto for
frame/jank analysis (see [Performance & Profiling](performance_profiling.md) and
[Graphics Stack](graphics_stack.md)).

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

# Reboot to bootloader (fastboot — flashing, unlocking; see Verified Boot & OTA)
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

`dumpsys` calls the `dump()` method of a registered Binder service over IPC (see
[Binder](binder.md)); `adb shell dumpsys -l` lists every service you can dump. It's the single
most useful framework-diagnostic command.

```bash
# Get system information (all services — very large)
adb shell dumpsys

# List every dumpable service
adb shell dumpsys -l

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

`run-as <pkg>` runs a command as the app's UID, the only way to reach an app's private
`/data/data/<pkg>` sandbox without root — and it only works on **debuggable** builds (see the
sandbox model in [App Security](app_security.md)).

```bash
# Access app database (requires root or a debuggable app)
adb shell run-as com.example.app

# Navigate to database directory
cd /data/data/com.example.app/databases/

# Pull database
adb exec-out run-as com.example.app cat databases/mydb.db > mydb.db

# Query database using sqlite3
adb shell "run-as com.example.app sqlite3 databases/mydb.db 'SELECT * FROM users;'"
```

## Testing and Automation

These are the low-level drivers behind instrumented testing; for the Espresso/UI Automator/
Macrobenchmark layers that build on them, see [Android Testing](testing_android.md).

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

## Resources

- [ADB — Android Developers](https://developer.android.com/tools/adb)
- [`am` / `pm` / Activity Manager shell](https://developer.android.com/tools/adb#am)
- [dumpsys](https://developer.android.com/tools/dumpsys)
- [logcat command-line tool](https://developer.android.com/tools/logcat)
- [SDK Platform Tools release notes](https://developer.android.com/tools/releases/platform-tools)

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

### Related Files

- [SystemServer & Core Services](system_server.md) — the services behind `dumpsys`, `pm`, `am`
- [Binder](binder.md) — the IPC `dumpsys`/`service` calls travel over
- [Performance & Profiling](performance_profiling.md) — logcat, `gfxinfo`, Perfetto workflows
- [Graphics Stack](graphics_stack.md) — diagnosing jank with `dumpsys gfxinfo`/`SurfaceFlinger`
- [APK/AAB Packaging & Signing](app_signing.md) — what `adb install`/`pm` deploy
- [App Security](app_security.md) — the app sandbox `run-as` reaches into
- [Android Testing](testing_android.md) — Monkey/UI Automator and the test layers above them
- [Verified Boot & OTA](verified_boot_ota.md) — `adb reboot bootloader`/recovery and flashing
- [Platform Dev](platform_dev.md) — building AOSP and flashing devices
- [Android Internals](internals.md) — the architecture these commands inspect
