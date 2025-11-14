# udev - Linux Dynamic Device Management

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Rules System](#rules-system)
- [Basic Operations](#basic-operations)
- [Common Patterns](#common-patterns)
- [Advanced Topics](#advanced-topics)
- [Complete Use Cases](#complete-use-cases)
- [Programming with libudev](#programming-with-libudev)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Quick Reference](#quick-reference)
- [Integration Examples](#integration-examples)
- [References](#references)

## Introduction

udev is the Linux device manager responsible for dynamically managing device nodes in the `/dev` directory. It handles device events from the kernel, creates and removes device nodes, manages permissions and ownership, creates symbolic links, and can execute programs in response to device events.

### Key Features

- **Dynamic device node management**: Creates/removes `/dev` entries as hardware is added/removed
- **Persistent device naming**: Provides consistent names for devices across reboots
- **Event-driven architecture**: Responds to kernel events in real-time
- **Flexible rules system**: Powerful pattern matching and device configuration
- **Permission management**: Controls device node ownership, group, and permissions
- **Integration with systemd**: Tightly integrated with modern init systems
- **Extensible**: Supports custom helper programs and scripts
- **Hardware database**: Maintains metadata about devices

### Evolution

- **Pre-udev era**: Static `/dev` directory with pre-created device nodes
- **udev standalone** (2003-2012): Independent device manager
- **systemd integration** (2012+): Now part of systemd project
- **Modern udev**: Uses devtmpfs for initial `/dev` population

### Use Cases

- Automatically mount USB drives when inserted
- Assign persistent names to network interfaces
- Set custom permissions for development devices (Arduino, FPGA boards)
- Manage multiple identical USB serial devices
- Trigger backups when external drives connect
- Configure printers and scanners automatically
- Handle Android device connections for development
- Control LEDs or other hardware based on device state

## Architecture

udev sits between the Linux kernel and user space, managing the interface between hardware events and device nodes.

### Device Event Flow

```
Kernel Space                    User Space
┌─────────────┐              ┌──────────────┐
│   Kernel    │              │    udevd     │
│  (devices)  │              │   (daemon)   │
└──────┬──────┘              └──────┬───────┘
       │                            │
       │ uevent                     │
       ├───────────────────────────>│
       │                            │
       │                     ┌──────▼───────┐
┌──────▼──────┐              │ Rules Engine │
│    sysfs    │<─────────────┤  Processing  │
│  /sys/...   │   reads      └──────┬───────┘
└─────────────┘                     │
                              ┌─────▼──────┐
┌─────────────┐               │   Action   │
│  devtmpfs   │<──────────────┤  - NAME    │
│  /dev/...   │   creates     │  - SYMLINK │
└─────────────┘               │  - MODE    │
                              │  - RUN     │
                              └────────────┘
```

### Components Interaction

1. **Kernel** detects hardware change (device added/removed)
2. **Kernel** creates uevent and populates **sysfs** (`/sys`)
3. **devtmpfs** may create initial device node
4. **udevd** receives uevent from kernel via netlink socket
5. **udevd** reads device information from **sysfs**
6. **udevd** processes rules in order
7. **udevd** applies actions (create symlinks, set permissions, run programs)
8. **udevd** updates device database

### udevd Daemon Architecture

- Runs as PID 1's child (started by systemd)
- Listens on netlink socket for kernel events
- Processes events sequentially (by default) or in parallel (with restrictions)
- Maintains device database in `/run/udev/data/`
- Enforces timeouts on rule execution
- Handles both coldplug (boot-time) and hotplug (runtime) events

## Core Components

### udevd Daemon

The central daemon that processes device events.

**Location**: `/lib/systemd/systemd-udevd`

**Key responsibilities**:
- Receive events from kernel
- Process rule files
- Execute actions
- Manage device database
- Enforce security policies

**Configuration**: `/etc/udev/udev.conf`

```ini
# /etc/udev/udev.conf
udev_log=info
children_max=128
exec_delay=0
event_timeout=180
resolve_names=early
```

### udevadm Utility

The primary tool for interacting with udev.

**Subcommands**:

| Command | Purpose |
|---------|---------|
| `udevadm info` | Query device information |
| `udevadm monitor` | Monitor kernel events and udev processing |
| `udevadm test` | Simulate rule processing for a device |
| `udevadm trigger` | Request device events from kernel |
| `udevadm settle` | Wait for event queue to empty |
| `udevadm control` | Control udevd daemon behavior |

### Rules Files

Rules are stored in multiple directories, processed in lexical order:

**System rules** (don't modify):
- `/lib/udev/rules.d/` - Distribution-provided rules
- `/usr/lib/udev/rules.d/` - Package-installed rules

**Custom rules** (your rules go here):
- `/etc/udev/rules.d/` - Local administrator rules
- `/run/udev/rules.d/` - Runtime rules

**Priority**: Files in `/etc` override files in `/lib` with the same name. Numbering convention:
- `00-99`: System and architecture rules
- `60-69`: Storage and filesystem rules
- `70-79`: Network rules
- `80-89`: Local rules (your custom rules)
- `90-99`: Late rules

### Helper Programs

Located in `/lib/udev/`:

- `ata_id` - ATA device information
- `cdrom_id` - CD/DVD device identification
- `scsi_id` - SCSI device identification
- `usb_id` - USB device identification
- `mtd_probe` - Memory Technology Device identification

### libudev Library

C library for accessing udev functionality programmatically.

**Key features**:
- Device enumeration
- Event monitoring
- Property querying
- Asynchronous operation

### Hardware Database (hwdb)

Binary database for hardware-specific information.

**Locations**:
- `/lib/udev/hwdb.d/` - System database
- `/etc/udev/hwdb.d/` - Local overrides

**Update**:
```bash
systemd-hwdb update    # Compile text files to binary
```

## Rules System

### Rule File Syntax

Each rule consists of one or more key-value pairs separated by commas. Rules span a single logical line (use `\` for line continuation).

**Basic structure**:
```
MATCH_KEY==value, MATCH_KEY2==value, ASSIGNMENT_KEY=value, ASSIGNMENT_KEY2=value
```

**Example**:
```bash
# Match USB device and set permissions
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", \
    MODE="0660", GROUP="dialout"
```

### Match Keys

Match keys are used to identify devices. All match keys in a rule must match for the rule to apply.

| Key | Description | Example |
|-----|-------------|---------|
| `KERNEL` | Match device kernel name | `KERNEL=="sda"` |
| `SUBSYSTEM` | Match device subsystem | `SUBSYSTEM=="net"` |
| `DRIVER` | Match device driver | `DRIVER=="usb"` |
| `ATTR{filename}` | Match sysfs attribute | `ATTR{idVendor}=="046d"` |
| `ATTRS{filename}` | Match parent's sysfs attribute | `ATTRS{serial}=="ABC123"` |
| `ENV{key}` | Match environment variable | `ENV{ID_USB_DRIVER}=="usb-storage"` |
| `KERNELS` | Match device or parent kernel name | `KERNELS=="2-1.1"` |
| `SUBSYSTEMS` | Match device or parent subsystem | `SUBSYSTEMS=="usb"` |
| `DRIVERS` | Match device or parent driver | `DRIVERS=="usb-storage"` |
| `TAG` | Match device tag | `TAG=="systemd"` |
| `TEST{filename}` | Test file existence | `TEST=="/sys/module/kvm"` |
| `PROGRAM` | Execute program and match output | `PROGRAM=="/lib/udev/scsi_id -g $devnode"` |
| `RESULT` | Match result of last PROGRAM | `RESULT=="1234567890"` |

### Operators

| Operator | Meaning | Used With |
|----------|---------|-----------|
| `==` | Equality match | Match keys |
| `!=` | Inequality match | Match keys |
| `=` | Assign value | Assignment keys |
| `+=` | Append to value | Assignment keys |
| `-=` | Remove from value | Assignment keys |
| `:=` | Assign final value (prevent changes) | Assignment keys |

### Assignment Keys

Assignment keys define actions to take when a rule matches.

| Key | Description | Example |
|-----|-------------|---------|
| `NAME` | Device node name | `NAME="mydevice"` |
| `SYMLINK` | Symbolic link(s) to create | `SYMLINK+="disk/by-label/backup"` |
| `OWNER` | Device node owner | `OWNER="root"` |
| `GROUP` | Device node group | `GROUP="disk"` |
| `MODE` | Device node permissions | `MODE="0660"` |
| `TAG` | Add device tag | `TAG+="systemd"` |
| `ENV{key}` | Set environment variable | `ENV{ID_MODEL}="MyDisk"` |
| `RUN` | Execute program (deprecated) | `RUN+="/usr/local/bin/script.sh"` |
| `RUN{program}` | Execute program after event | `RUN{program}+="/bin/mount $devnode"` |
| `LABEL` | Named label for GOTO | `LABEL="my_label"` |
| `GOTO` | Jump to LABEL | `GOTO="my_label"` |
| `IMPORT` | Import variables from program/file | `IMPORT{program}="/lib/udev/usb_id"` |
| `OPTIONS` | Rule options | `OPTIONS+="last_rule"` |

### String Substitutions

Variables available in rules:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `$kernel` or `%k` | Kernel name | `sda` |
| `$number` or `%n` | Kernel number | `1` (from sda1) |
| `$devpath` or `%p` | Device path in `/sys` | `/devices/pci0000:00/...` |
| `$id` | Device ID | USB port number |
| `$driver` | Device driver | `usb-storage` |
| `$devnode` | Device node path | `/dev/sda1` |
| `$attr{file}` | Sysfs attribute value | `$attr{size}` |
| `$env{key}` | Environment variable | `$env{ID_SERIAL}` |
| `$major` or `%M` | Device major number | `8` |
| `$minor` or `%m` | Device minor number | `1` |
| `$result` or `%c` | Output of PROGRAM | varies |
| `$parent` | Parent device path | Parent device |
| `$name` | Device name (after NAME) | Custom name |
| `$links` | Space-separated symlinks | All symlinks |
| `$root` | udev runtime directory | `/run/udev` |
| `$sys` | sysfs mount point | `/sys` |
| `$tempnode` | Temporary device node | For testing |
| `%%` | Literal `%` | `%` |
| `$$` | Literal `$` | `$` |

### String Modifiers

Modify substitution values:

```bash
# Get last component of path
SYMLINK+="disk/by-path/$env{ID_PATH}/basename"

# Get all but last component
PROGRAM="/bin/echo $env{ID_PATH}/dirname"

# Replace characters
SYMLINK+="disk/by-label/$env{ID_FS_LABEL}/replace{' ', '_'}"
```

### Rule Processing Flow

1. **Event received**: udevd receives uevent from kernel
2. **Device matching**: Each rule file processed in lexical order
3. **Rule evaluation**: For each rule, all match keys must match
4. **Action execution**: Assignment keys are processed
5. **Early exit**: `OPTIONS="last_rule"` stops processing
6. **Database update**: Device properties stored in `/run/udev/data/`

### Rule File Ordering

Files are processed in lexical order. Use numeric prefixes to control order:

```bash
/etc/udev/rules.d/
├── 10-local-network.rules      # Processed first
├── 50-usb-devices.rules        # Processed second
└── 99-local-late.rules         # Processed last
```

Within a file, rules are processed top to bottom.

## Basic Operations

### Monitoring Device Events

Watch events in real-time:

```bash
# Monitor both kernel events and udev processing
udevadm monitor

# Monitor with more detail
udevadm monitor --environment --property

# Monitor specific subsystem
udevadm monitor --subsystem-match=block

# Monitor multiple subsystems
udevadm monitor --subsystem-match=block --subsystem-match=usb

# Monitor with kernel events only
udevadm monitor --kernel
```

Example output:
```
KERNEL[12345.678] add      /devices/pci0000:00/.../block/sdb (block)
UDEV  [12345.789] add      /devices/pci0000:00/.../block/sdb (block)
```

### Querying Device Information

Get detailed device information:

```bash
# Query by device node
udevadm info /dev/sda

# Query by device path
udevadm info --path=/sys/class/net/eth0

# Query with all properties
udevadm info --query=property /dev/sda

# Query specific property
udevadm info --query=property --property=ID_MODEL /dev/sda

# Query all properties including parent devices
udevadm info --attribute-walk /dev/sda

# Show device path
udevadm info --query=path /dev/sda

# Show symlinks
udevadm info --query=symlink /dev/sda
```

### Listing Device Attributes

Walk the device tree to see available attributes:

```bash
# Show all attributes for matching
udevadm info --attribute-walk --name=/dev/sda1

# Example output:
#   looking at device '/devices/pci0000:00/.../block/sda/sda1':
#     KERNEL=="sda1"
#     SUBSYSTEM=="block"
#     ATTR{size}=="1953525168"
#     ATTR{ro}=="0"
#   looking at parent device:
#     KERNELS=="sda"
#     SUBSYSTEMS=="block"
#     ATTRS{model}=="Samsung SSD 860"
```

### Testing Rules

Test rules without applying them:

```bash
# Test rules for a device
udevadm test /sys/class/net/eth0

# Test with debugging output
udevadm test --action=add /sys/class/block/sda

# Test and show only what would be executed
udevadm test /sys/class/block/sda 2>&1 | grep -E "RUN|SYMLINK|NAME"
```

### Triggering Events

Manually trigger device events:

```bash
# Trigger events for all devices
udevadm trigger

# Trigger for specific subsystem
udevadm trigger --subsystem-match=block

# Trigger for specific device
udevadm trigger --name-match=/dev/sda

# Trigger with specific action
udevadm trigger --action=change --subsystem-match=net

# Trigger for devices with specific attribute
udevadm trigger --attr-match=idVendor=046d

# Dry run (show what would be triggered)
udevadm trigger --dry-run --subsystem-match=usb
```

### Reloading Rules

Reload udev rules after making changes:

```bash
# Reload rules
udevadm control --reload-rules

# Reload and trigger events to apply new rules
udevadm control --reload-rules && udevadm trigger
```

### Waiting for Event Processing

Wait for udev queue to empty:

```bash
# Wait for all events to be processed
udevadm settle

# Wait with timeout (30 seconds)
udevadm settle --timeout=30

# Wait for specific event
udevadm trigger --name-match=/dev/sda && udevadm settle
```

### Controlling the Daemon

Control udevd behavior:

```bash
# Reload rules
udevadm control --reload

# Set log level
udevadm control --log-level=debug

# Stop executing rules (emergency)
udevadm control --stop-exec-queue

# Resume executing rules
udevadm control --start-exec-queue

# Show daemon status
systemctl status systemd-udevd
```

### Viewing Persistent Device Names

List persistent device naming schemes:

```bash
# View all symlinks for block devices
ls -la /dev/disk/by-*

# By UUID
ls -la /dev/disk/by-uuid/

# By label
ls -la /dev/disk/by-label/

# By path
ls -la /dev/disk/by-path/

# By ID
ls -la /dev/disk/by-id/

# By partition UUID
ls -la /dev/disk/by-partuuid/
```

### Examining the Device Database

View udev's internal database:

```bash
# Database location
ls -la /run/udev/data/

# Query database for device
udevadm info --query=all /dev/sda | grep "^[ES]:"
# E: = Environment variable
# S: = Symlink
```

## Common Patterns

### Network Device Naming

#### Persistent Interface Name by MAC Address

```bash
# /etc/udev/rules.d/70-persistent-net.rules
# Rename network interface to eth0 based on MAC address
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="00:11:22:33:44:55", NAME="eth0"

# Multiple interfaces
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="00:11:22:33:44:55", NAME="lan0"
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="00:11:22:33:44:56", NAME="wan0"
```

#### Custom Interface Names by Driver

```bash
# Name wireless interfaces
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="?*", ATTR{type}=="1", \
    KERNEL=="wlan*", NAME="wifi0"

# Name USB ethernet adapters
SUBSYSTEM=="net", ACTION=="add", DRIVERS=="r8152", NAME="usb-eth0"
```

#### Disable Predictable Network Names

```bash
# Force traditional eth0 style naming
SUBSYSTEM=="net", ACTION=="add", NAME="eth$env{IFINDEX}"
```

#### Name by PCI Slot

```bash
# Name interface by PCI bus position
SUBSYSTEM=="net", ACTION=="add", \
    KERNELS=="0000:02:00.0", NAME="lan-slot2"
```

### Storage Device Patterns

#### Persistent Disk Name by Serial Number

```bash
# /etc/udev/rules.d/60-persistent-storage.rules
# Create symlink for disk by serial number
SUBSYSTEM=="block", KERNEL=="sd?", \
    ATTRS{serial}=="S1234567890", \
    SYMLINK+="disk/by-serial-custom/$attrs{serial}"

# Specific partition
SUBSYSTEM=="block", KERNEL=="sd?1", \
    ATTRS{serial}=="S1234567890", \
    SYMLINK+="disk/my-backup-disk"
```

#### Persistent Name by Filesystem Label

```bash
# Create custom symlink for labeled filesystem
SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="BACKUP", \
    SYMLINK+="backup-disk"

# Multiple labels
SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="MEDIA", \
    SYMLINK+="media-disk"
SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="ARCHIVE", \
    SYMLINK+="archive-disk"
```

#### Persistent Name for USB Storage by Port

```bash
# Identify USB storage by physical port
SUBSYSTEM=="block", KERNEL=="sd?", \
    KERNELS=="2-1.4", \
    SYMLINK+="usb-port-front-left"

# Partition on specific USB port
SUBSYSTEM=="block", KERNEL=="sd?1", \
    KERNELS=="2-1.4", \
    SYMLINK+="usb-port-front-left-part1"
```

#### Auto-mount Detection with Tag

```bash
# Tag removable media for systemd automount
SUBSYSTEM=="block", ENV{ID_FS_USAGE}=="filesystem", \
    ENV{UDISKS_AUTO}="1", TAG+="systemd"
```

### USB Device Patterns

#### Set Permissions for USB Device

```bash
# Grant access to specific USB device
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", \
    MODE="0660", GROUP="dialout"

# Multiple products from same vendor
SUBSYSTEM=="usb", ATTR{idVendor}=="2341", \
    MODE="0660", GROUP="arduino", TAG+="uaccess"
```

#### Identify USB Device by Serial Number

```bash
# Match specific device instance
SUBSYSTEM=="usb", ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", \
    ATTRS{serial}=="ABC123", \
    SYMLINK+="usb-prolific-abc123"
```

#### USB Device by Manufacturer String

```bash
# Match by manufacturer and product strings
SUBSYSTEM=="usb", ATTRS{manufacturer}=="FTDI", \
    ATTRS{product}=="FT232R USB UART", \
    MODE="0660", GROUP="dialout"
```

#### Persistent Name for USB Serial Devices

```bash
# Create persistent name for USB-serial converter
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", \
    ATTRS{serial}=="FT123456", \
    SYMLINK+="ttyUSB-FTDI-FT123456"

# By position in device tree
SUBSYSTEM=="tty", KERNELS=="1-1.2", \
    SYMLINK+="ttyUSB-port-1"
```

### Multiple Identical Devices

#### Differentiate by USB Port

```bash
# /etc/udev/rules.d/80-usb-serial-ports.rules
# Top port
SUBSYSTEM=="tty", KERNELS=="2-1.1", \
    SYMLINK+="arduino-top"

# Bottom port
SUBSYSTEM=="tty", KERNELS=="2-1.2", \
    SYMLINK+="arduino-bottom"
```

#### Differentiate by Serial Number

```bash
# Create unique names for identical USB devices
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
    ATTRS{serial}=="001", SYMLINK+="cp210x-sensor1"

SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", \
    ATTRS{serial}=="002", SYMLINK+="cp210x-sensor2"
```

### Permission and Ownership Patterns

#### Developer Device Access

```bash
# Grant user access to FPGA development boards
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", \
    MODE="0666", GROUP="plugdev"

# STM32 programmers
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="3748", \
    MODE="0660", GROUP="developers", TAG+="uaccess"
```

#### Group-based Access Control

```bash
# Video capture devices accessible by video group
SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"

# Sound devices accessible by audio group
SUBSYSTEM=="sound", GROUP="audio", MODE="0660"

# Printer devices
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", \
    ENV{ID_USB_INTERFACES}=="*:0701??:*", \
    GROUP="lp", MODE="0660"
```

#### Security Devices

```bash
# YubiKey security key
SUBSYSTEM=="usb", ATTR{idVendor}=="1050", ATTR{idProduct}=="0407", \
    MODE="0660", GROUP="yubikey", TAG+="uaccess"

# Nitrokey
SUBSYSTEM=="usb", ATTR{idVendor}=="20a0", ATTR{idProduct}=="4108", \
    MODE="0660", GROUP="nitrokey"
```

### Symlink Creation Patterns

#### Multiple Symlinks for One Device

```bash
# Create multiple meaningful symlinks
SUBSYSTEM=="block", ENV{ID_SERIAL}=="WD_My_Passport_1234", \
    SYMLINK+="backup", \
    SYMLINK+="western-digital", \
    SYMLINK+="portable-hdd"
```

#### Directory-organized Links

```bash
# Organize devices in /dev subdirectories
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", \
    SYMLINK+="arduino/$attrs{serial}"

SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="*", \
    SYMLINK+="disk/by-custom-label/$env{ID_FS_LABEL}"
```

#### Application-specific Paths

```bash
# Create symlinks for specific applications
SUBSYSTEM=="video4linux", ATTRS{product}=="*Webcam*", \
    KERNEL=="video*", \
    SYMLINK+="video-webcam", \
    SYMLINK+="apps/skype/camera"
```

### Running Programs on Events

#### Execute Script on Device Add

```bash
# Run script when USB drive inserted
SUBSYSTEM=="block", KERNEL=="sd[a-z][0-9]", \
    ACTION=="add", \
    ENV{ID_FS_UUID}=="1234-5678", \
    RUN{program}+="/usr/local/bin/backup-script.sh"
```

#### Execute with Device Information

```bash
# Pass device info to script
SUBSYSTEM=="net", ACTION=="add", \
    RUN{program}+="/usr/local/bin/network-notify.sh $kernel $attr{address}"
```

#### Set Environment Variables for Programs

```bash
# Set environment for downstream processing
SUBSYSTEM=="block", KERNEL=="sd?1", \
    ENV{MY_MOUNT_POINT}="/mnt/external", \
    ENV{MY_DEVICE_TYPE}="external_hdd"
```

#### Use systemd Service Instead of RUN

Modern approach - trigger systemd service:

```bash
# Tag device to trigger systemd template service
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="1234-5678", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="backup@%k.service"
```

### Input Device Patterns

#### Keyboard and Mouse Permissions

```bash
# Grant seat access to input devices
SUBSYSTEM=="input", KERNEL=="event*", \
    TAG+="uaccess"

# Specific gaming devices
SUBSYSTEM=="input", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c52b", \
    MODE="0660", GROUP="gamers"
```

#### Touchscreen Configuration

```bash
# Tag touchscreen for X11
SUBSYSTEM=="input", KERNEL=="event*", \
    ENV{ID_INPUT_TOUCHSCREEN}=="1", \
    TAG+="touchscreen"
```

### Android Development

#### ADB Device Access

```bash
# /etc/udev/rules.d/51-android.rules
# Google devices
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0660", GROUP="adbusers", TAG+="uaccess"

# Samsung devices
SUBSYSTEM=="usb", ATTR{idVendor}=="04e8", MODE="0660", GROUP="adbusers", TAG+="uaccess"

# Generic Android devices
SUBSYSTEM=="usb", ENV{ID_USB_INTERFACES}=="*:ff420?:*", \
    MODE="0660", GROUP="adbusers", TAG+="uaccess"
```

### Serial Port Patterns

#### Set Line Discipline

```bash
# Configure serial port parameters
SUBSYSTEM=="tty", KERNEL=="ttyUSB*", \
    ATTRS{idVendor}=="0403", \
    RUN{program}+="/bin/stty -F /dev/%k 115200 cs8 -cstopb -parenb"
```

#### Bluetooth Serial Ports

```bash
# Bluetooth RFCOMM devices
SUBSYSTEM=="tty", KERNEL=="rfcomm*", \
    GROUP="dialout", MODE="0660"
```

## Advanced Topics

### Writing Portable Rules

#### Avoid Hardware-specific Paths

```bash
# Bad - hardware path changes
KERNEL=="ttyUSB0", SYMLINK+="mydevice"

# Good - use attributes
SUBSYSTEM=="tty", ATTRS{serial}=="ABC123", SYMLINK+="mydevice"
```

#### Use Parent Attributes for Stability

```bash
# Match against parent device (more stable)
SUBSYSTEM=="tty", SUBSYSTEMS=="usb", \
    ATTRS{idVendor}=="0403", \
    ATTRS{idProduct}=="6001", \
    SYMLINK+="usb-serial-ftdi"
```

#### Distribution-agnostic Rules

```bash
# Work across distributions
SUBSYSTEM=="block", ENV{ID_FS_UUID}!="", \
    SYMLINK+="disk/by-custom-uuid/$env{ID_FS_UUID}"

# Don't rely on specific package paths
# Import standard device identification
IMPORT{builtin}="usb_id"
IMPORT{builtin}="path_id"
```

### Performance Optimization

#### Minimize Rule Complexity

```bash
# Bad - multiple rules doing similar things
SUBSYSTEM=="block", KERNEL=="sd?", PROGRAM=="/usr/bin/script1.sh"
SUBSYSTEM=="block", KERNEL=="sd?", PROGRAM=="/usr/bin/script2.sh"

# Good - combine when possible
SUBSYSTEM=="block", KERNEL=="sd?", PROGRAM=="/usr/bin/combined-script.sh"
```

#### Use Early Exits

```bash
# Skip irrelevant subsystems early
SUBSYSTEM!="block", GOTO="end_block_rules"

# ... block-specific rules ...

LABEL="end_block_rules"
```

#### Avoid Slow PROGRAM Calls

```bash
# Bad - calling external program for each device
PROGRAM=="/usr/bin/slow-check.sh $kernel", RESULT=="1", ...

# Good - use built-in tests when possible
KERNEL=="sd?", TEST=="/sys/block/%k/queue/rotational", ...
```

#### Use Built-in String Matching

```bash
# Built-in matching is fast
KERNEL=="sd[a-z]", ...
ATTR{size}=="*[0-9]", ...

# Avoid external programs for simple checks
# Bad: PROGRAM=="/usr/bin/test -f /sys/..."
# Good: TEST=="/sys/..."
```

### Custom Helper Programs

#### Writing Helper Programs

Helper programs receive device information via environment variables:

```bash
#!/bin/bash
# /lib/udev/my-helper.sh
# Environment variables available:
# DEVPATH, SUBSYSTEM, ACTION, DEVNAME, MAJOR, MINOR, etc.

echo "Device: $DEVNAME"
echo "Subsystem: $SUBSYSTEM"
echo "Action: $ACTION"

# Return 0 for success
exit 0
```

#### Using Helper Output

```bash
# Capture program output
SUBSYSTEM=="block", IMPORT{program}="/lib/udev/scsi_id -g $devnode"

# Use the imported variables
SUBSYSTEM=="block", ENV{ID_SERIAL}=="?*", \
    SYMLINK+="disk/by-id/scsi-$env{ID_SERIAL}"
```

#### Timeout Handling

```bash
# Rules have execution timeout (default 180s)
# Long-running tasks should be spawned asynchronously
SUBSYSTEM=="block", ACTION=="add", \
    RUN{program}+="/usr/bin/systemd-run /usr/local/bin/long-task.sh"
```

### Hardware Database (hwdb)

#### Custom hwdb Entry

```bash
# /etc/udev/hwdb.d/90-custom-devices.hwdb
# USB device metadata
usb:v046Dp082D*
 ID_MODEL=Logitech_HD_Webcam_C615
 ID_VENDOR=Logitech

# PCI device
pci:v00008086d00001234*
 ID_MODEL=Intel_Custom_Device
```

Update and apply:
```bash
systemd-hwdb update
udevadm trigger --subsystem-match=usb
```

### Integration with systemd

#### Trigger systemd Mount

```bash
# Mount filesystem via systemd
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="1234-5678", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="mount-external.service"
```

#### Template Service Activation

```bash
# /etc/udev/rules.d/90-backup-device.rules
SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="BACKUP*", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="backup@$env{ID_FS_LABEL}.service"
```

Corresponding service file:
```ini
# /etc/systemd/system/backup@.service
[Unit]
Description=Backup service for %I
After=dev-disk-by\x2dlabel-%i.device

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh /dev/disk/by-label/%I
```

### Device Tagging

#### Using Tags for Classification

```bash
# Tag devices for different purposes
SUBSYSTEM=="block", ENV{ID_USB_DRIVER}=="usb-storage", \
    TAG+="backup-eligible"

# Process tagged devices differently
TAG=="backup-eligible", ENV{ID_FS_TYPE}=="ext4", \
    ENV{SYSTEMD_WANTS}="backup-check@%k.service"
```

### Handling Race Conditions

#### Wait for Device Initialization

```bash
# Some devices need time to initialize
SUBSYSTEM=="tty", KERNEL=="ttyACM*", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="wait-for-tty@%k.service"
```

#### Use udevadm settle in Scripts

```bash
#!/bin/bash
# Script that depends on device being ready

# Trigger event
udevadm trigger --name-match=/dev/sda

# Wait for processing
udevadm settle --timeout=10

# Now safe to proceed
mount /dev/sda1 /mnt
```

### Multi-path Device Management

#### Identify Multi-path Devices

```bash
# Tag multipath components
SUBSYSTEM=="block", ENV{DM_MULTIPATH_DEVICE_PATH}=="1", \
    TAG+="multipath"

# Use multipath-specific symlinks
SUBSYSTEM=="block", ENV{DM_UUID}=="mpath-*", \
    SYMLINK+="mapper/$env{DM_NAME}"
```

## Complete Use Cases

### Use Case 1: Auto-mount USB Drives to User Directories

**Objective**: Automatically mount USB drives to `/media/username/label` when inserted.

**Rule file**: `/etc/udev/rules.d/90-usb-automount.rules`

```bash
# Tag USB storage devices for automount
SUBSYSTEM=="block", ENV{ID_BUS}=="usb", ENV{ID_FS_USAGE}=="filesystem", \
    TAG+="systemd", ENV{SYSTEMD_WANTS}="usb-automount@%k.service"
```

**Systemd service**: `/etc/systemd/system/usb-automount@.service`

```ini
[Unit]
Description=Auto-mount USB drive %I
After=dev-%i.device

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/usb-mount.sh %I
ExecStop=/usr/local/bin/usb-unmount.sh %I
```

**Mount script**: `/usr/local/bin/usb-mount.sh`

```bash
#!/bin/bash
DEVICE=$1
USER=$(who | awk '{print $1}' | head -1)
LABEL=$(lsblk -no LABEL /dev/$DEVICE)
MOUNTPOINT="/media/$USER/${LABEL:-$DEVICE}"

mkdir -p "$MOUNTPOINT"
mount -o uid=$(id -u $USER),gid=$(id -g $USER) /dev/$DEVICE "$MOUNTPOINT"
chown $USER:$USER "$MOUNTPOINT"

# Notify user
sudo -u $USER DISPLAY=:0 notify-send "USB Drive Mounted" "$DEVICE mounted at $MOUNTPOINT"
```

**Unmount script**: `/usr/local/bin/usb-unmount.sh`

```bash
#!/bin/bash
DEVICE=$1
MOUNTPOINT=$(mount | grep "/dev/$DEVICE" | awk '{print $3}')

if [ -n "$MOUNTPOINT" ]; then
    umount "$MOUNTPOINT"
    rmdir "$MOUNTPOINT"
fi
```

### Use Case 2: Persistent Network Interface Naming for Servers

**Objective**: Ensure network interfaces have consistent names across reboots for a server with multiple NICs.

**Rule file**: `/etc/udev/rules.d/70-server-network.rules`

```bash
# Management interface (IPMI/BMC on motherboard)
SUBSYSTEM=="net", ACTION=="add", \
    ATTR{address}=="00:25:90:xx:xx:01", \
    NAME="mgmt0"

# Data plane interfaces (PCIe cards)
SUBSYSTEM=="net", ACTION=="add", \
    ATTR{address}=="00:1b:21:xx:xx:10", \
    NAME="data0"

SUBSYSTEM=="net", ACTION=="add", \
    ATTR{address}=="00:1b:21:xx:xx:11", \
    NAME="data1"

# Backup interface (onboard)
SUBSYSTEM=="net", ACTION=="add", \
    KERNELS=="0000:00:19.0", \
    NAME="backup0"

# Alternative: name by PCI slot
SUBSYSTEM=="net", ACTION=="add", \
    KERNELS=="0000:03:00.0", \
    NAME="slot3-net0"
```

**Verification**:
```bash
# Check interface names
ip link show

# Test rule without applying
udevadm test /sys/class/net/eth0 2>&1 | grep NAME

# Apply new rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=net --action=add
```

### Use Case 3: Managing Multiple Identical USB Serial Adapters

**Objective**: Differentiate between 3 identical USB-serial adapters for industrial sensors.

**Rule file**: `/etc/udev/rules.d/80-industrial-sensors.rules`

```bash
# Sensor 1 - Top USB port
SUBSYSTEM=="tty", KERNELS=="1-1.1", \
    ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", \
    SYMLINK+="sensors/temperature", \
    MODE="0660", GROUP="sensors"

# Sensor 2 - Middle USB port
SUBSYSTEM=="tty", KERNELS=="1-1.2", \
    ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", \
    SYMLINK+="sensors/pressure", \
    MODE="0660", GROUP="sensors"

# Sensor 3 - Bottom USB port
SUBSYSTEM=="tty", KERNELS=="1-1.3", \
    ATTRS{idVendor}=="067b", ATTRS{idProduct}=="2303", \
    SYMLINK+="sensors/humidity", \
    MODE="0660", GROUP="sensors"

# Notify monitoring system
SUBSYSTEM=="tty", KERNEL=="ttyUSB*", \
    SYMLINK=="sensors/*", \
    RUN{program}+="/usr/local/bin/sensor-notify.sh $env{DEVNAME} add"
```

**Notification script**: `/usr/local/bin/sensor-notify.sh`

```bash
#!/bin/bash
DEVICE=$1
ACTION=$2
LOGFILE=/var/log/sensors.log

echo "$(date): Sensor $ACTION on $DEVICE" >> $LOGFILE

# Restart monitoring service if all sensors present
if [ "$ACTION" = "add" ]; then
    if [ -e /dev/sensors/temperature ] && \
       [ -e /dev/sensors/pressure ] && \
       [ -e /dev/sensors/humidity ]; then
        systemctl restart sensor-monitoring.service
    fi
fi
```

**Finding USB port paths**:
```bash
# Plug in device and check kernel path
udevadm info --query=path --name=/dev/ttyUSB0
# Look for KERNELS value like "1-1.1"

# Or monitor as you plug in
udevadm monitor --kernel --subsystem-match=tty
```

### Use Case 4: Automated Backup on External Drive Insertion

**Objective**: Start backup automatically when specific external drive is connected.

**Rule file**: `/etc/udev/rules.d/90-backup-drive.rules`

```bash
# Identify backup drive by UUID
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="abcd1234-5678-90ef-ghij-klmnopqrstuv", \
    ENV{BACKUP_DRIVE}="true", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="external-backup.service"
```

**Service file**: `/etc/systemd/system/external-backup.service`

```ini
[Unit]
Description=External Drive Backup
After=dev-disk-by\x2duuid-abcd1234\x2d5678\x2d90ef\x2dghij\x2dklmnopqrstuv.device

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup-to-external.sh
```

**Backup script**: `/usr/local/bin/backup-to-external.sh`

```bash
#!/bin/bash

BACKUP_UUID="abcd1234-5678-90ef-ghij-klmnopqrstuv"
MOUNT_POINT="/mnt/backup"
SOURCE_DIR="/home"
BACKUP_DIR="$MOUNT_POINT/backups/$(hostname)"
LOG_FILE="/var/log/external-backup.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'): $1" | tee -a "$LOG_FILE"
}

# Mount if not already mounted
if ! mountpoint -q "$MOUNT_POINT"; then
    mkdir -p "$MOUNT_POINT"
    mount UUID="$BACKUP_UUID" "$MOUNT_POINT" || {
        log "ERROR: Failed to mount backup drive"
        exit 1
    }
fi

# Verify drive is correct
if [ ! -f "$MOUNT_POINT/.backup_drive_marker" ]; then
    log "ERROR: Drive verification failed"
    umount "$MOUNT_POINT"
    exit 1
fi

log "Starting backup to external drive"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform incremental backup using rsync
rsync -av --delete \
    --exclude='.cache' \
    --exclude='.local/share/Trash' \
    --log-file="$LOG_FILE" \
    "$SOURCE_DIR/" "$BACKUP_DIR/" || {
    log "ERROR: Backup failed"
    umount "$MOUNT_POINT"
    exit 1
}

# Create timestamp
date > "$BACKUP_DIR/.last_backup"

log "Backup completed successfully"

# LED notification (if supported)
echo 1 > /sys/class/leds/backup-led/brightness 2>/dev/null || true

# Don't unmount - let user safely remove
sync
```

### Use Case 5: Android Development Device Setup

**Objective**: Configure automatic access for Android devices connected via USB for ADB.

**Rule file**: `/etc/udev/rules.d/51-android-dev.rules`

```bash
# Google Nexus/Pixel devices
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0660", \
    GROUP="adbusers", TAG+="uaccess"

# Samsung devices
SUBSYSTEM=="usb", ATTR{idVendor}=="04e8", MODE="0660", \
    GROUP="adbusers", TAG+="uaccess"

# OnePlus devices
SUBSYSTEM=="usb", ATTR{idVendor}=="2a70", MODE="0660", \
    GROUP="adbusers", TAG+="uaccess"

# Xiaomi devices
SUBSYSTEM=="usb", ATTR{idVendor}=="2717", MODE="0660", \
    GROUP="adbusers", TAG+="uaccess"

# Generic Android devices (ADB interface)
SUBSYSTEM=="usb", ENV{ID_USB_INTERFACES}=="*:ff420?:*", \
    MODE="0660", GROUP="adbusers", TAG+="uaccess"

# Notify when Android device connected
SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", \
    RUN{program}+="/usr/local/bin/android-notify.sh add"

SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", ACTION=="remove", \
    RUN{program}+="/usr/local/bin/android-notify.sh remove"
```

**Setup**:
```bash
# Create adbusers group
sudo groupadd -r adbusers

# Add your user
sudo usermod -a -G adbusers $USER

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Notification script**: `/usr/local/bin/android-notify.sh`

```bash
#!/bin/bash
ACTION=$1
USER=$(who | grep -m1 "(:[0-9])" | awk '{print $1}')

if [ "$ACTION" = "add" ]; then
    # Start ADB server if not running
    sudo -u $USER adb start-server 2>/dev/null

    # Wait for device
    sleep 2

    # Check if device is authorized
    DEVICE_STATE=$(sudo -u $USER adb get-state 2>&1)

    if [[ "$DEVICE_STATE" == "device" ]]; then
        sudo -u $USER DISPLAY=:0 notify-send "Android Device" \
            "Device connected and authorized"
    else
        sudo -u $USER DISPLAY=:0 notify-send "Android Device" \
            "Device connected - check authorization on phone" \
            -u critical
    fi
else
    sudo -u $USER DISPLAY=:0 notify-send "Android Device" "Device disconnected"
fi
```

### Use Case 6: Industrial Equipment Device Management

**Objective**: Manage industrial PLC and HMI devices with automatic configuration.

**Rule file**: `/etc/udev/rules.d/80-industrial-plc.rules`

```bash
# Siemens PLC - Ethernet adapter
SUBSYSTEM=="net", KERNELS=="0000:03:00.0", \
    NAME="plc-eth", \
    ENV{PLC_INTERFACE}="true"

# Modbus RTU serial interface
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", \
    ATTRS{serial}=="PLC001", \
    SYMLINK+="plc/modbus-rtu", \
    MODE="0660", GROUP="industrial", \
    RUN{program}+="/bin/stty -F /dev/%k 19200 cs8 -cstopb -parenb"

# HMI touchscreen
SUBSYSTEM=="input", ATTRS{idVendor}=="0eef", ATTRS{idProduct}=="0001", \
    ENV{DEVNAME}=="*event*", \
    SYMLINK+="input/hmi-touch", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="hmi-calibration.service"

# Emergency stop button
SUBSYSTEM=="input", ATTRS{product}=="Emergency Stop", \
    SYMLINK+="input/emergency-stop", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="estop-monitor.service"

# Industrial sensors via USB
SUBSYSTEM=="tty", SUBSYSTEMS=="usb", \
    ATTRS{manufacturer}=="IndustrialSensors", \
    SYMLINK+="sensors/$attr{product}", \
    MODE="0660", GROUP="industrial"

# Start monitoring when all devices present
SUBSYSTEM=="tty", SYMLINK=="sensors/*", \
    RUN{program}+="/usr/local/bin/check-all-devices.sh"
```

**Device checker**: `/usr/local/bin/check-all-devices.sh`

```bash
#!/bin/bash

REQUIRED_DEVICES=(
    "/dev/plc/modbus-rtu"
    "/dev/input/hmi-touch"
    "/dev/sensors/temperature"
    "/dev/sensors/pressure"
)

all_present=true

for device in "${REQUIRED_DEVICES[@]}"; do
    if [ ! -e "$device" ]; then
        all_present=false
        logger "Industrial: Missing device $device"
    fi
done

if [ "$all_present" = true ]; then
    logger "Industrial: All devices present, starting production monitoring"
    systemctl start production-monitoring.service
fi
```

### Use Case 7: LED Control Based on Device State

**Objective**: Control chassis LEDs based on storage device activity.

**Rule file**: `/etc/udev/rules.d/90-storage-leds.rules`

```bash
# Disk activity LED control
SUBSYSTEM=="block", KERNEL=="sd[a-z]", \
    RUN{program}+="/usr/local/bin/set-disk-led.sh %k add"

SUBSYSTEM=="block", KERNEL=="sd[a-z]", ACTION=="remove", \
    RUN{program}+="/usr/local/bin/set-disk-led.sh %k remove"

# RAID array status
SUBSYSTEM=="block", KERNEL=="md*", \
    RUN{program}+="/usr/local/bin/raid-led-status.sh %k"
```

**LED control script**: `/usr/local/bin/set-disk-led.sh`

```bash
#!/bin/bash

DISK=$1
ACTION=$2
LED_BASE="/sys/class/leds"

# Map disk to LED (customize for your hardware)
case $DISK in
    sda)
        LED="disk0-led"
        ;;
    sdb)
        LED="disk1-led"
        ;;
    sdc)
        LED="disk2-led"
        ;;
    *)
        exit 0
        ;;
esac

LED_PATH="$LED_BASE/$LED/brightness"

if [ ! -f "$LED_PATH" ]; then
    exit 0
fi

if [ "$ACTION" = "add" ]; then
    echo 1 > "$LED_PATH"
else
    echo 0 > "$LED_PATH"
fi
```

### Use Case 8: Printer and Scanner Automatic Configuration

**Objective**: Auto-configure network printers and scanners when connected.

**Rule file**: `/etc/udev/rules.d/80-printers-scanners.rules`

```bash
# USB printers
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", \
    ENV{ID_USB_INTERFACES}=="*:0701??:*", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="configure-printer@$env{BUSNUM}-$env{DEVNUM}.service"

# USB scanners
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", \
    ENV{ID_USB_INTERFACES}=="*:070103:*", \
    MODE="0660", GROUP="scanner", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="configure-scanner@$env{BUSNUM}-$env{DEVNUM}.service"

# HP multi-function devices
SUBSYSTEM=="usb", ATTR{idVendor}=="03f0", \
    ATTRS{product}=="*LaserJet*", \
    MODE="0660", GROUP="lp", \
    RUN{program}+="/usr/local/bin/hp-device-setup.sh"
```

**Printer configuration service**: `/etc/systemd/system/configure-printer@.service`

```ini
[Unit]
Description=Auto-configure printer %I
After=cups.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/auto-add-printer.sh %I
```

**Auto-add script**: `/usr/local/bin/auto-add-printer.sh`

```bash
#!/bin/bash

DEVICE_ID=$1

# Wait for CUPS
sleep 2

# Get USB device info
VENDOR=$(udevadm info --query=property /dev/bus/usb/$DEVICE_ID | \
    grep ID_VENDOR= | cut -d= -f2)
MODEL=$(udevadm info --query=property /dev/bus/usb/$DEVICE_ID | \
    grep ID_MODEL= | cut -d= -f2)

PRINTER_NAME="${VENDOR}_${MODEL}"

# Check if already configured
if lpstat -p "$PRINTER_NAME" >/dev/null 2>&1; then
    logger "Printer $PRINTER_NAME already configured"
    exit 0
fi

# Add printer
lpadmin -p "$PRINTER_NAME" \
    -E \
    -v "usb://$DEVICE_ID" \
    -m everywhere

logger "Added printer: $PRINTER_NAME"

# Set as default if no default exists
if [ -z "$(lpstat -d 2>/dev/null)" ]; then
    lpadmin -d "$PRINTER_NAME"
    logger "Set $PRINTER_NAME as default printer"
fi
```

## Programming with libudev

### Basic Device Enumeration

```c
#include <libudev.h>
#include <stdio.h>

int main() {
    struct udev *udev;
    struct udev_enumerate *enumerate;
    struct udev_list_entry *devices, *dev_list_entry;

    // Create udev context
    udev = udev_new();
    if (!udev) {
        fprintf(stderr, "Cannot create udev context\n");
        return 1;
    }

    // Create enumeration
    enumerate = udev_enumerate_new(udev);
    udev_enumerate_add_match_subsystem(enumerate, "block");
    udev_enumerate_scan_devices(enumerate);
    devices = udev_enumerate_get_list_entry(enumerate);

    // Iterate through devices
    udev_list_entry_foreach(dev_list_entry, devices) {
        const char *path;
        struct udev_device *dev;

        path = udev_list_entry_get_name(dev_list_entry);
        dev = udev_device_new_from_syspath(udev, path);

        printf("Device: %s\n", udev_device_get_devnode(dev));
        printf("  Type: %s\n", udev_device_get_devtype(dev));
        printf("  Sysname: %s\n", udev_device_get_sysname(dev));

        udev_device_unref(dev);
    }

    udev_enumerate_unref(enumerate);
    udev_unref(udev);

    return 0;
}
```

**Compile**:
```bash
gcc -o list-devices list-devices.c $(pkg-config --cflags --libs libudev)
```

### Monitoring Device Events

```c
#include <libudev.h>
#include <stdio.h>
#include <poll.h>

int main() {
    struct udev *udev;
    struct udev_monitor *mon;
    struct pollfd fds[1];
    int ret;

    udev = udev_new();
    if (!udev) {
        fprintf(stderr, "Cannot create udev context\n");
        return 1;
    }

    // Create monitor
    mon = udev_monitor_new_from_netlink(udev, "udev");
    udev_monitor_filter_add_match_subsystem_devtype(mon, "usb", NULL);
    udev_monitor_enable_receiving(mon);

    // Setup polling
    fds[0].fd = udev_monitor_get_fd(mon);
    fds[0].events = POLLIN;

    printf("Monitoring USB devices...\n");

    while (1) {
        ret = poll(fds, 1, -1);
        if (ret > 0 && (fds[0].revents & POLLIN)) {
            struct udev_device *dev;

            dev = udev_monitor_receive_device(mon);
            if (dev) {
                printf("Action: %s\n", udev_device_get_action(dev));
                printf("Device: %s\n", udev_device_get_devnode(dev) ?: "N/A");
                printf("Vendor: %s\n",
                    udev_device_get_sysattr_value(dev, "idVendor") ?: "N/A");
                printf("Product: %s\n",
                    udev_device_get_sysattr_value(dev, "idProduct") ?: "N/A");
                printf("\n");

                udev_device_unref(dev);
            }
        }
    }

    udev_monitor_unref(mon);
    udev_unref(udev);

    return 0;
}
```

### Querying Device Properties

```c
#include <libudev.h>
#include <stdio.h>

void print_device_properties(struct udev_device *dev) {
    struct udev_list_entry *properties, *entry;

    properties = udev_device_get_properties_list_entry(dev);

    printf("Properties:\n");
    udev_list_entry_foreach(entry, properties) {
        printf("  %s=%s\n",
            udev_list_entry_get_name(entry),
            udev_list_entry_get_value(entry));
    }
}

int main(int argc, char *argv[]) {
    struct udev *udev;
    struct udev_device *dev;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <device_path>\n", argv[0]);
        return 1;
    }

    udev = udev_new();
    if (!udev) {
        fprintf(stderr, "Cannot create udev context\n");
        return 1;
    }

    // Create device from syspath or devnode
    dev = udev_device_new_from_syspath(udev, argv[1]);
    if (!dev) {
        dev = udev_device_new_from_devnum(udev, 'b', makedev(8, 0));
    }

    if (dev) {
        printf("Device node: %s\n", udev_device_get_devnode(dev));
        printf("Subsystem: %s\n", udev_device_get_subsystem(dev));
        printf("Device type: %s\n", udev_device_get_devtype(dev) ?: "N/A");

        print_device_properties(dev);

        udev_device_unref(dev);
    } else {
        fprintf(stderr, "Device not found\n");
    }

    udev_unref(udev);
    return 0;
}
```

### Complete Event Monitor with Filtering

```c
#include <libudev.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <signal.h>

static int running = 1;

void sighandler(int signum) {
    running = 0;
}

int main(int argc, char *argv[]) {
    struct udev *udev;
    struct udev_monitor *mon;
    struct pollfd fds[1];
    const char *subsystem = NULL;
    int ret;

    if (argc > 1) {
        subsystem = argv[1];
    }

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    udev = udev_new();
    if (!udev) {
        fprintf(stderr, "Cannot create udev context\n");
        return 1;
    }

    mon = udev_monitor_new_from_netlink(udev, "udev");

    if (subsystem) {
        printf("Filtering for subsystem: %s\n", subsystem);
        udev_monitor_filter_add_match_subsystem_devtype(mon, subsystem, NULL);
    }

    udev_monitor_enable_receiving(mon);

    fds[0].fd = udev_monitor_get_fd(mon);
    fds[0].events = POLLIN;

    printf("Monitoring device events (Ctrl+C to stop)...\n\n");

    while (running) {
        ret = poll(fds, 1, 1000);

        if (ret < 0) {
            break;
        }

        if (ret > 0 && (fds[0].revents & POLLIN)) {
            struct udev_device *dev;
            const char *action, *devnode, *subsys;

            dev = udev_monitor_receive_device(mon);
            if (dev) {
                action = udev_device_get_action(dev);
                devnode = udev_device_get_devnode(dev);
                subsys = udev_device_get_subsystem(dev);

                printf("EVENT: %s %s %s\n",
                    action ? action : "unknown",
                    subsys ? subsys : "unknown",
                    devnode ? devnode : "no_devnode");

                // Print relevant properties
                printf("  SYS_PATH: %s\n", udev_device_get_syspath(dev));

                const char *vendor = udev_device_get_sysattr_value(dev, "idVendor");
                const char *product = udev_device_get_sysattr_value(dev, "idProduct");

                if (vendor && product) {
                    printf("  USB_ID: %s:%s\n", vendor, product);
                }

                printf("\n");

                udev_device_unref(dev);
            }
        }
    }

    printf("Shutting down...\n");
    udev_monitor_unref(mon);
    udev_unref(udev);

    return 0;
}
```

**Compile and run**:
```bash
gcc -o udev-monitor udev-monitor.c $(pkg-config --cflags --libs libudev)
./udev-monitor block    # Monitor block devices only
```

## Troubleshooting

### Rules Not Being Applied

**Symptoms**: Device appears but rules don't take effect.

**Diagnosis**:
```bash
# Check rule syntax
udevadm test /sys/class/block/sda 2>&1 | grep -i error

# Verify rule is loaded
udevadm test /sys/class/block/sda 2>&1 | grep "Reading rules file"

# Check for conflicting rules
udevadm test /sys/class/block/sda 2>&1 | grep "NAME"
```

**Common causes**:

1. **Syntax errors in rules**:
```bash
# Wrong - missing comma
SUBSYSTEM=="block" KERNEL=="sda" SYMLINK+="mydisk"

# Correct
SUBSYSTEM=="block", KERNEL=="sda", SYMLINK+="mydisk"
```

2. **Incorrect match keys**:
```bash
# Check available attributes
udevadm info --attribute-walk --name=/dev/sda

# Verify your match keys exist in output
```

3. **Rule file name/location**:
```bash
# Must be in correct directory
ls -la /etc/udev/rules.d/
# Must end with .rules
# Must have numeric prefix (e.g., 80-custom.rules)
```

4. **Rules not reloaded**:
```bash
# Always reload after editing
udevadm control --reload-rules
udevadm trigger --name-match=/dev/sda
```

### Permission Denied Errors

**Symptoms**: Cannot access device even though it exists.

**Diagnosis**:
```bash
# Check device permissions
ls -l /dev/ttyUSB0

# Check user groups
groups

# Check udev database
udevadm info --query=property /dev/ttyUSB0 | grep -E "OWNER|GROUP|MODE"
```

**Solutions**:

1. **Add user to correct group**:
```bash
# For serial devices
sudo usermod -a -G dialout $USER

# For USB devices
sudo usermod -a -G plugdev $USER

# Log out and back in for changes to take effect
```

2. **Fix rule permissions**:
```bash
# /etc/udev/rules.d/80-mydevice.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", \
    MODE="0660", GROUP="dialout"
```

3. **Use TAG+="uaccess" for user sessions**:
```bash
# Allow currently logged-in user
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", \
    TAG+="uaccess"
```

### Device Not Recognized

**Symptoms**: Device connected but no `/dev` entry created.

**Diagnosis**:
```bash
# Check if kernel sees device
dmesg | tail -50

# Check if udev received event
udevadm monitor --kernel --property

# Check sysfs
ls -la /sys/bus/usb/devices/
```

**Solutions**:

1. **Driver issue**:
```bash
# Check if driver loaded
lsmod | grep <driver_name>

# Load driver manually
modprobe <driver_name>

# Check dmesg for errors
dmesg | grep -i error
```

2. **Wait for device initialization**:
```bash
# Some devices need time
udevadm settle --timeout=30
```

### Timing and Race Conditions

**Symptoms**: Rules work intermittently or device attributes unavailable.

**Diagnosis**:
```bash
# Monitor timing
udevadm monitor --property | ts '[%Y-%m-%d %H:%M:%.S]'

# Test multiple times
for i in {1..10}; do
    udevadm trigger --name-match=/dev/sda
    udevadm settle
    sleep 1
done
```

**Solutions**:

1. **Use WAIT_FOR or TEST**:
```bash
# Wait for file to exist
SUBSYSTEM=="block", KERNEL=="sd?", \
    WAIT_FOR="/sys/block/%k/queue/rotational"

# Test file exists
SUBSYSTEM=="block", KERNEL=="sd?", \
    TEST=="/sys/block/%k/queue/rotational", \
    ATTR{queue/rotational}=="0", \
    TAG+="ssd"
```

2. **Import parent attributes properly**:
```bash
# Use ATTRS (with S) to search up device tree
SUBSYSTEM=="tty", SUBSYSTEMS=="usb", \
    ATTRS{idVendor}=="0403", \
    SYMLINK+="mydevice"
```

3. **Use systemd for complex setups**:
```bash
# Tag for systemd instead of RUN
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="1234", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="mount-device.service"
```

### Rules Execution Timeout

**Symptoms**: Long-running scripts cause udev delays.

**Diagnosis**:
```bash
# Check for timeout errors
journalctl -u systemd-udevd | grep timeout

# Monitor event processing time
udevadm monitor --property | grep -E "SEQNUM|USEC_INITIALIZED"
```

**Solutions**:

1. **Use systemd-run for long tasks**:
```bash
# Bad - blocks udev
RUN{program}+="/usr/local/bin/long-script.sh"

# Good - runs asynchronously
RUN{program}+="/usr/bin/systemd-run /usr/local/bin/long-script.sh"
```

2. **Optimize scripts**:
```bash
# Move slow operations to background
#!/bin/bash
(
    # Long operation
    /usr/bin/process-device.sh "$DEVNAME"
) &
```

3. **Increase timeout (last resort)**:
```bash
# /etc/udev/udev.conf
event_timeout=300
```

### NAME Assignment Not Working

**Symptoms**: NAME assignment ignored, device keeps kernel name.

**Diagnosis**:
```bash
# Test rule
udevadm test /sys/class/net/eth0 2>&1 | grep NAME

# Check for conflicts
udevadm test /sys/class/net/eth0 2>&1 | grep -E "NAME|name"
```

**Solutions**:

1. **Use SYMLINK instead**:
```bash
# NAME only works for some subsystems
# Use SYMLINK for flexibility
SUBSYSTEM=="block", KERNEL=="sd?", \
    SYMLINK+="mydisk"
```

2. **Check NAME is allowed**:
```bash
# NAME works for network interfaces
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="...", NAME="eth0"

# NAME doesn't work well for block devices (use SYMLINK)
```

3. **Ensure no later rule overrides**:
```bash
# Use OPTIONS to prevent override
SUBSYSTEM=="net", ATTR{address}=="...", \
    NAME="eth0", \
    OPTIONS+="last_rule"
```

### Debugging Complex Rules

**Enable debug logging**:
```bash
# Increase log level
udevadm control --log-level=debug

# View logs
journalctl -u systemd-udevd -f

# Reset log level
udevadm control --log-level=info
```

**Test step by step**:
```bash
# Test single device
udevadm test --action=add /sys/class/block/sda 2>&1 | less

# Look for specific match
udevadm test /sys/class/block/sda 2>&1 | grep "ATTR{size}"

# Check what properties are available
udevadm info --query=property /dev/sda
```

**Verify rule matching**:
```bash
# See which rules matched
udevadm test /sys/class/block/sda 2>&1 | \
    grep -A2 "Reading rules file"
```

### Common Syntax Errors

```bash
# 1. Forgetting comma separator
# Wrong:
SUBSYSTEM=="block" KERNEL=="sda"
# Right:
SUBSYSTEM=="block", KERNEL=="sda"

# 2. Using = instead of ==
# Wrong:
SUBSYSTEM="block", KERNEL="sda"
# Right:
SUBSYSTEM=="block", KERNEL=="sda"

# 3. Wrong quote marks
# Wrong:
SUBSYSTEM=="block', KERNEL=="sda"
# Right:
SUBSYSTEM=="block", KERNEL=="sda"

# 4. Missing + for append
# Wrong (overwrites):
SYMLINK="disk1"
SYMLINK="disk2"  # Only disk2 exists
# Right (both exist):
SYMLINK+="disk1"
SYMLINK+="disk2"

# 5. Attribute syntax
# Wrong:
ATTRS="idVendor"=="0403"
# Right:
ATTRS{idVendor}=="0403"
```

### Performance Issues

**Symptoms**: Slow boot, delayed device recognition.

**Diagnosis**:
```bash
# Analyze boot performance
systemd-analyze blame | grep udev

# Monitor event processing
udevadm monitor --property | ts
```

**Solutions**:

1. **Reduce rule complexity**:
```bash
# Bad - runs program for every device
SUBSYSTEM=="block", PROGRAM="/usr/bin/check-device.sh"

# Good - limit scope
SUBSYSTEM=="block", KERNEL=="sd[a-z]", \
    ATTRS{vendor}=="SpecificVendor", \
    PROGRAM="/usr/bin/check-device.sh"
```

2. **Remove unnecessary rules**:
```bash
# Audit rules
ls -la /etc/udev/rules.d/
# Remove unused rules
```

3. **Use early exits**:
```bash
# Skip non-relevant devices early
SUBSYSTEM!="block", GOTO="end_block_rules"
KERNEL!="sd*", GOTO="end_block_rules"

# ... block-specific rules ...

LABEL="end_block_rules"
```

## Best Practices

### 1. Rule Organization

**Naming convention**:
- Use descriptive prefixes: `70-persistent-net.rules`
- Follow numbering: 60-69 storage, 70-79 network, 80-89 local
- One purpose per file

**Structure**:
```bash
/etc/udev/rules.d/
├── 70-network-naming.rules       # Network interface names
├── 80-usb-devices.rules          # USB device permissions
├── 85-serial-ports.rules         # Serial port mappings
└── 90-local-automation.rules     # Custom automation
```

### 2. Security Considerations

**Minimize permissions**:
```bash
# Bad - world writable
MODE="0666"

# Good - group writable only
MODE="0660", GROUP="dialout"
```

**Validate before execution**:
```bash
# Validate device attributes before running scripts
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="known-uuid", \
    TEST=="/usr/local/bin/safe-script.sh", \
    RUN{program}+="/usr/local/bin/safe-script.sh"
```

**Avoid running untrusted code**:
```bash
# Don't use device-controlled values in RUN
# Bad:
RUN{program}+="/bin/sh $attr{script}"  # DANGEROUS!

# Good:
ENV{SAFE_LABEL}="$env{ID_FS_LABEL}"
RUN{program}+="/usr/local/bin/process.sh"
```

**Use TAG+="uaccess" for user devices**:
```bash
# Give current user access
SUBSYSTEM=="usb", ATTR{idVendor}=="1234", \
    TAG+="uaccess"
```

### 3. Testing Strategies

**Test before deploying**:
```bash
# Always test rules
udevadm test /sys/class/block/sda

# Check syntax
udevadm test /sys/class/block/sda 2>&1 | grep -i error

# Dry run triggers
udevadm trigger --dry-run --subsystem-match=block
```

**Version control**:
```bash
# Track rule changes
cd /etc/udev/rules.d/
git init
git add *.rules
git commit -m "Initial udev rules"
```

**Backup before changes**:
```bash
# Backup existing rules
sudo cp -a /etc/udev/rules.d /etc/udev/rules.d.backup.$(date +%Y%m%d)
```

**Test in stages**:
```bash
# 1. Test rule syntax
udevadm test /sys/class/block/sda 2>&1 | grep -E "error|warning" -i

# 2. Check what would happen
udevadm test /sys/class/block/sda 2>&1 | grep -E "SYMLINK|NAME|RUN"

# 3. Apply to one device
udevadm trigger --name-match=/dev/sda

# 4. Apply to subsystem
udevadm trigger --subsystem-match=block

# 5. Apply to all
udevadm trigger
```

### 4. Documentation

**Comment your rules**:
```bash
# Purpose: Persistent naming for server NICs
# Created: 2024-01-15
# Author: admin

# Management interface (onboard NIC)
SUBSYSTEM=="net", ACTION=="add", ATTR{address}=="00:11:22:33:44:55", \
    NAME="mgmt0"
```

**Document device identification**:
```bash
# How to find these values:
# udevadm info --attribute-walk --name=/dev/ttyUSB0 | grep -E "idVendor|idProduct|serial"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", \
    ATTRS{serial}=="ABC123", \
    SYMLINK+="mydevice"
```

**Maintain a README**:
```bash
# /etc/udev/rules.d/README.md
cat > /etc/udev/rules.d/README.md << 'EOF'
# Custom udev Rules

## Overview
This directory contains custom udev rules for this system.

## Rules Files

- `70-network-naming.rules` - Persistent network interface names
- `80-usb-devices.rules` - USB device permissions for development
- `90-automation.rules` - Automated mounting and backups

## Testing Changes

After modifying rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Troubleshooting

See logs:
```bash
journalctl -u systemd-udevd
```
EOF
```

### 5. Avoid Common Pitfalls

**Don't hardcode device nodes**:
```bash
# Bad
PROGRAM=="/usr/bin/backup.sh /dev/sdb1"

# Good
PROGRAM=="/usr/bin/backup.sh $devnode"
```

**Use appropriate operators**:
```bash
# Match: use ==
KERNEL=="sda"

# Assign: use =
NAME="mydisk"

# Append: use +=
SYMLINK+="disk/by-label/backup"
```

**Limit PROGRAM usage**:
```bash
# Bad - slow
SUBSYSTEM=="block", PROGRAM="/usr/bin/get-info.sh", ...

# Good - use attributes
SUBSYSTEM=="block", ATTR{size}=="*", ...
```

**Watch for quoting issues**:
```bash
# Variables don't need quotes in assignments
NAME="disk-$env{ID_SERIAL}"  # Correct

# But be careful with spaces
SYMLINK+="My Disk"  # Creates two symlinks: "My" and "Disk"
SYMLINK+="My_Disk"  # Correct
```

## Quick Reference

### udevadm Commands

| Command | Description | Example |
|---------|-------------|---------|
| `info` | Query device information | `udevadm info /dev/sda` |
| `monitor` | Monitor events | `udevadm monitor --subsystem-match=block` |
| `test` | Simulate rule processing | `udevadm test /sys/class/block/sda` |
| `trigger` | Request device events | `udevadm trigger --name-match=/dev/sda` |
| `settle` | Wait for events | `udevadm settle --timeout=30` |
| `control --reload` | Reload rules | `udevadm control --reload-rules` |
| `control --log-level` | Set logging | `udevadm control --log-level=debug` |

### Match Keys Reference

| Key | Matches | Example |
|-----|---------|---------|
| `KERNEL` | Device kernel name | `KERNEL=="sda"` |
| `SUBSYSTEM` | Device subsystem | `SUBSYSTEM=="block"` |
| `DRIVER` | Device driver name | `DRIVER=="usb-storage"` |
| `ATTR{file}` | Sysfs attribute (current device) | `ATTR{idVendor}=="0403"` |
| `ATTRS{file}` | Sysfs attribute (any parent) | `ATTRS{serial}=="ABC123"` |
| `ENV{key}` | Environment variable | `ENV{ID_FS_TYPE}=="ext4"` |
| `TAG` | Device tag | `TAG=="systemd"` |
| `TEST{file}` | File existence | `TEST=="/sys/module/kvm"` |
| `PROGRAM` | Execute and match stdout | `PROGRAM=="/usr/bin/check.sh"` |
| `RESULT` | Match PROGRAM result | `RESULT=="match_this"` |

### Assignment Keys Reference

| Key | Action | Example |
|-----|--------|---------|
| `NAME` | Device node name | `NAME="mydevice"` |
| `SYMLINK` | Create symlink | `SYMLINK+="disk/backup"` |
| `OWNER` | Set owner | `OWNER="root"` |
| `GROUP` | Set group | `GROUP="disk"` |
| `MODE` | Set permissions | `MODE="0660"` |
| `TAG` | Add tag | `TAG+="systemd"` |
| `ENV{key}` | Set environment | `ENV{MY_VAR}="value"` |
| `RUN{program}` | Execute program | `RUN{program}+="/usr/bin/script.sh"` |
| `LABEL` | Named label | `LABEL="my_label"` |
| `GOTO` | Jump to label | `GOTO="my_label"` |
| `IMPORT` | Import variables | `IMPORT{program}="/lib/udev/usb_id"` |
| `OPTIONS` | Special options | `OPTIONS+="last_rule"` |

### Operators

| Operator | Meaning | Used With |
|----------|---------|-----------|
| `==` | Equal (match) | Match keys |
| `!=` | Not equal (match) | Match keys |
| `=` | Assign | Assignment keys |
| `+=` | Append | Assignment keys |
| `-=` | Remove | Assignment keys |
| `:=` | Assign final (no override) | Assignment keys |

### String Substitutions

| Pattern | Expands To | Example Value |
|---------|------------|---------------|
| `%k`, `$kernel` | Kernel device name | `sda` |
| `%n`, `$number` | Kernel number | `1` |
| `%p`, `$devpath` | Device path | `/devices/pci...` |
| `%M`, `$major` | Major number | `8` |
| `%m`, `$minor` | Minor number | `0` |
| `$attr{file}` | Sysfs attribute | varies |
| `$env{key}` | Environment variable | varies |
| `$devnode` | Device node path | `/dev/sda` |
| `$result` | PROGRAM output | varies |
| `%%` | Literal `%` | `%` |
| `$$` | Literal `$` | `$` |

### Common Subsystems

| Subsystem | Device Type | Example |
|-----------|-------------|---------|
| `block` | Block devices | `/dev/sda`, `/dev/nvme0n1` |
| `net` | Network interfaces | `eth0`, `wlan0` |
| `tty` | Serial/terminal | `/dev/ttyUSB0`, `/dev/ttyS0` |
| `usb` | USB devices | Various |
| `input` | Input devices | Keyboards, mice |
| `sound` | Audio devices | Sound cards |
| `video4linux` | Video devices | Webcams, capture cards |
| `scsi` | SCSI devices | Disks, optical drives |
| `pci` | PCI devices | Various |
| `hidraw` | HID devices | Raw HID access |

### Useful Attributes

**Block devices**:
- `size` - Device size in sectors
- `ro` - Read-only flag
- `removable` - Removable media flag
- `queue/rotational` - HDD (1) vs SSD (0)

**USB devices**:
- `idVendor` - USB vendor ID
- `idProduct` - USB product ID
- `serial` - Serial number
- `manufacturer` - Manufacturer string
- `product` - Product string

**Network devices**:
- `address` - MAC address
- `type` - Interface type
- `carrier` - Link status

## Integration Examples

### systemd Mount Units

Automatically mount devices using systemd:

**udev rule**: `/etc/udev/rules.d/90-automount.rules`
```bash
SUBSYSTEM=="block", ENV{ID_FS_UUID}=="1234-5678", \
    TAG+="systemd", \
    ENV{SYSTEMD_WANTS}="mnt-backup.mount"
```

**Mount unit**: `/etc/systemd/system/mnt-backup.mount`
```ini
[Unit]
Description=Backup Drive
After=dev-disk-by\x2duuid-1234\x2d5678.device

[Mount]
What=/dev/disk/by-uuid/1234-5678
Where=/mnt/backup
Type=ext4
Options=defaults,noatime

[Install]
WantedBy=multi-user.target
```

### Desktop Environment Integration

Integration with desktop notifications:

```bash
# /etc/udev/rules.d/90-desktop-notify.rules
SUBSYSTEM=="block", KERNEL=="sd[a-z][0-9]", \
    ACTION=="add", \
    ENV{ID_FS_LABEL}=="*", \
    RUN{program}+="/usr/local/bin/notify-user.sh add '%E{ID_FS_LABEL}'"

SUBSYSTEM=="block", KERNEL=="sd[a-z][0-9]", \
    ACTION=="remove", \
    RUN{program}+="/usr/local/bin/notify-user.sh remove"
```

**Notification script**:
```bash
#!/bin/bash
ACTION=$1
LABEL=$2
USER=$(who | grep -m1 '(:[0-9])' | awk '{print $1}')

if [ "$ACTION" = "add" ]; then
    sudo -u $USER DISPLAY=:0 notify-send \
        "USB Device Connected" \
        "Device: $LABEL" \
        -i drive-removable-media
else
    sudo -u $USER DISPLAY=:0 notify-send \
        "USB Device Removed" \
        -i drive-removable-media
fi
```

### Container Device Handling

Pass devices to containers:

```bash
# /etc/udev/rules.d/90-container-devices.rules
# Tag devices for container access
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", \
    TAG+="container-passthrough", \
    ENV{CONTAINER_NAME}="dev-environment"

# Notify container runtime
SUBSYSTEM=="usb", TAG=="container-passthrough", \
    RUN{program}+="/usr/local/bin/container-device-notify.sh $env{CONTAINER_NAME} $devnode"
```

### Virtual Machine Device Passthrough

Prepare devices for VM passthrough:

```bash
# /etc/udev/rules.d/80-vfio.rules
# Bind devices to VFIO driver for VM passthrough
SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{device}=="0x1234", \
    DRIVER=="nouveau", \
    RUN{program}+="/usr/bin/vfio-bind.sh %k"
```

## References

### Man Pages

- `man udev` - Overview of udev
- `man udevadm` - udevadm utility
- `man systemd-udevd` - udev daemon
- `man udev.conf` - udev configuration
- `man hwdb` - Hardware database

### Files and Directories

- `/lib/udev/rules.d/` - System rules
- `/etc/udev/rules.d/` - Custom rules
- `/run/udev/rules.d/` - Runtime rules
- `/etc/udev/udev.conf` - udev configuration
- `/run/udev/data/` - Device database
- `/sys/` - sysfs mount point

### Online Resources

- systemd.io - Official systemd/udev documentation
- kernel.org - Kernel device management documentation
- freedesktop.org - Historical udev documentation
- Arch Wiki - Comprehensive udev examples

### Related Tools

- `lsusb` - List USB devices
- `lspci` - List PCI devices
- `lsblk` - List block devices
- `hwinfo` - Hardware information
- `systemd-analyze` - Boot performance analysis
