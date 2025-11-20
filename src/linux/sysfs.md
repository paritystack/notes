# sysfs

sysfs is a virtual filesystem that exports information about kernel subsystems, hardware devices, and associated device drivers to userspace. Introduced in Linux 2.6, sysfs provides a structured, hierarchical view of the system's device model and allows both reading device information and configuring device/kernel parameters dynamically.

## Overview

sysfs is mounted at `/sys` and provides:
- Device information and topology
- Driver parameters and configuration
- Kernel configuration and runtime parameters
- Power management settings and controls
- Hardware monitoring and thermal management
- Resource allocation and cgroups
- Firmware and UEFI variables

## Fundamentals

### File Types and Attributes

sysfs exposes information through files that can be:
- **Read-only**: Information files (most device attributes)
- **Write-only**: Control files (some security/admin functions)
- **Read-write**: Configurable parameters

### Reading Attributes

```bash
# Reading values
cat /sys/class/net/eth0/address
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Some files contain multiple values
cat /sys/class/net/eth0/statistics/rx_bytes
```

### Writing Attributes

```bash
# Most write operations require root privileges
echo "performance" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Direct redirection (must be running as root)
sudo sh -c 'echo 1 > /sys/class/leds/led0/brightness'
```

### Important Characteristics

- **Temporary Changes**: All sysfs modifications are temporary and lost on reboot
- **Permissions**: Write operations typically require root/sudo
- **One Value Per File**: Each file generally contains a single value or closely related values
- **ASCII Format**: Values are human-readable ASCII strings
- **Symlinks**: Many entries are symlinks to actual device locations

## Structure

```bash
/sys/
├── block/          # Block devices (disks, partitions)
├── bus/            # Bus types (pci, usb, i2c, spi, etc.)
├── class/          # Device classes (network, input, graphics, etc.)
├── dev/            # Device number mappings (char/ and block/)
├── devices/        # Actual device hierarchy (the real device tree)
├── firmware/       # Firmware interfaces (acpi, dmi, efi, devicetree)
├── fs/             # Filesystem information (ext4, btrfs, cgroup, etc.)
├── hypervisor/     # Hypervisor information (if running in VM)
├── kernel/         # Kernel parameters and information
├── module/         # Loaded kernel modules and their parameters
└── power/          # System-wide power management
```

### Understanding the Hierarchy

- **`/sys/devices/`**: The canonical device tree, organized by how devices are connected
- **`/sys/class/`**: Devices grouped by functionality (all network interfaces, all input devices)
- **`/sys/bus/`**: Devices organized by bus type (all PCI devices, all USB devices)

Most entries in `/sys/class/` and `/sys/bus/` are symlinks pointing to `/sys/devices/`.

## Common Usage

### Block Devices

```bash
# List all block devices
ls /sys/block/
# Output: sda sdb nvme0n1 ...

# Device information
cat /sys/block/sda/size              # Size in 512-byte sectors
cat /sys/block/sda/queue/scheduler   # I/O scheduler
cat /sys/block/sda/device/model      # Drive model
cat /sys/block/sda/removable         # 1 if removable, 0 otherwise

# Partition information
ls /sys/block/sda/
cat /sys/block/sda/sda1/size

# Change I/O scheduler
echo "mq-deadline" | sudo tee /sys/block/sda/queue/scheduler

# Disk statistics
cat /sys/block/sda/stat
# Format: read IOs, read merges, sectors read, time reading (ms),
#         write IOs, write merges, sectors written, time writing (ms), ...
```

### Network Devices

Modern Linux uses predictable network interface names (not eth0):
- **ens0, eno1**: Onboard Ethernet
- **enp3s0**: PCI Ethernet
- **wlan0, wlp2s0**: Wireless
- **eth0**: Legacy naming (older systems or manual configuration)

```bash
# List network interfaces
ls /sys/class/net/
# Output: lo ens0 wlp2s0 ...

# Interface information
cat /sys/class/net/ens0/address           # MAC address
cat /sys/class/net/ens0/speed             # Link speed (Mbps)
cat /sys/class/net/ens0/operstate         # up/down/unknown
cat /sys/class/net/ens0/carrier           # 1 if link detected, 0 otherwise
cat /sys/class/net/ens0/mtu               # MTU size
cat /sys/class/net/ens0/type              # Hardware type (1=Ethernet)

# Statistics
cat /sys/class/net/ens0/statistics/rx_bytes
cat /sys/class/net/ens0/statistics/tx_bytes
cat /sys/class/net/ens0/statistics/rx_packets
cat /sys/class/net/ens0/statistics/rx_errors
cat /sys/class/net/ens0/statistics/rx_dropped

# Wireless information (if wireless device)
cat /sys/class/net/wlp2s0/wireless/status

# Change MTU
echo 9000 | sudo tee /sys/class/net/ens0/mtu
```

### CPU Information

```bash
# List CPUs
ls /sys/devices/system/cpu/
# Output: cpu0 cpu1 cpu2 cpu3 ... cpufreq/ cpuidle/ ...

# Current frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq    # In KHz
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq    # Hardware max
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq    # Hardware min

# Available frequencies
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

# Governor (power policy)
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# Common governors: performance, powersave, ondemand, conservative, schedutil

# CPU topology
cat /sys/devices/system/cpu/cpu0/topology/physical_package_id  # Socket
cat /sys/devices/system/cpu/cpu0/topology/core_id              # Core ID
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list # HT siblings

# Online/offline status
cat /sys/devices/system/cpu/cpu1/online
echo 0 | sudo tee /sys/devices/system/cpu/cpu1/online  # Disable CPU1

# Cache information
ls /sys/devices/system/cpu/cpu0/cache/
cat /sys/devices/system/cpu/cpu0/cache/index0/size     # L1 cache
cat /sys/devices/system/cpu/cpu0/cache/index2/size     # L3 cache
```

### GPU/Graphics Devices

```bash
# List graphics cards
ls /sys/class/drm/
# Output: card0 card1 ...

# GPU vendor and device ID
cat /sys/class/drm/card0/device/vendor    # 0x8086 (Intel), 0x10de (NVIDIA), 0x1002 (AMD)
cat /sys/class/drm/card0/device/device    # Device ID

# GPU power state (if supported)
cat /sys/class/drm/card0/device/power_state

# Current GPU frequency (Intel)
cat /sys/class/drm/card0/gt_cur_freq_mhz

# Display outputs
ls /sys/class/drm/card0-*
# Output: card0-HDMI-A-1 card0-DP-1 card0-eDP-1 ...

# Display status
cat /sys/class/drm/card0-HDMI-A-1/status    # connected/disconnected
cat /sys/class/drm/card0-HDMI-A-1/enabled   # enabled/disabled
cat /sys/class/drm/card0-eDP-1/dpms         # Display power management

# Framebuffer information
ls /sys/class/graphics/fb0/
cat /sys/class/graphics/fb0/name
cat /sys/class/graphics/fb0/virtual_size
```

### USB Devices

```bash
# List USB devices
ls /sys/bus/usb/devices/
# Output: 1-0:1.0 1-1 1-1:1.0 2-0:1.0 ...

# Device tree navigation
# Format: bus-port.subport:config.interface
# Example: 1-2.3 = bus 1, port 2, subport 3

# Device information
cat /sys/bus/usb/devices/1-2/manufacturer
cat /sys/bus/usb/devices/1-2/product
cat /sys/bus/usb/devices/1-2/serial
cat /sys/bus/usb/devices/1-2/version         # USB version
cat /sys/bus/usb/devices/1-2/speed           # Speed in Mbps

# USB IDs
cat /sys/bus/usb/devices/1-2/idVendor        # Vendor ID
cat /sys/bus/usb/devices/1-2/idProduct       # Product ID
cat /sys/bus/usb/devices/1-2/bDeviceClass    # Device class

# Power management
cat /sys/bus/usb/devices/1-2/power/autosuspend_delay_ms
cat /sys/bus/usb/devices/1-2/power/control   # auto/on
echo "on" | sudo tee /sys/bus/usb/devices/1-2/power/control  # Disable autosuspend

# Authorized (security)
cat /sys/bus/usb/devices/1-2/authorized      # 1=authorized, 0=not authorized
echo 0 | sudo tee /sys/bus/usb/devices/1-2/authorized  # Disable device
```

### PCI Devices

```bash
# List PCI devices
ls /sys/bus/pci/devices/
# Output: 0000:00:00.0 0000:00:02.0 ...

# Device information
cat /sys/bus/pci/devices/0000:00:02.0/vendor
cat /sys/bus/pci/devices/0000:00:02.0/device
cat /sys/bus/pci/devices/0000:00:02.0/class
cat /sys/bus/pci/devices/0000:00:02.0/resource  # Memory/IO resources

# Power management
cat /sys/bus/pci/devices/0000:00:02.0/power/control

# Remove/rescan devices (useful for hotplug)
echo 1 | sudo tee /sys/bus/pci/devices/0000:00:02.0/remove
echo 1 | sudo tee /sys/bus/pci/rescan
```

## Thermal Management

```bash
# List thermal zones
ls /sys/class/thermal/
# Output: cooling_device0 thermal_zone0 thermal_zone1 ...

# Temperature reading
cat /sys/class/thermal/thermal_zone0/type    # Type of sensor (acpitz, x86_pkg_temp)
cat /sys/class/thermal/thermal_zone0/temp    # Temperature in millidegrees Celsius
# Example output: 45000 = 45.0°C

# All thermal zones at once
for zone in /sys/class/thermal/thermal_zone*/; do
    echo -n "$(cat ${zone}type): "
    echo "scale=1; $(cat ${zone}temp)/1000" | bc
done

# Thermal trip points (thresholds)
cat /sys/class/thermal/thermal_zone0/trip_point_0_type  # passive/active/critical
cat /sys/class/thermal/thermal_zone0/trip_point_0_temp  # Threshold in millidegrees

# Cooling devices
ls /sys/class/thermal/cooling_device*/
cat /sys/class/thermal/cooling_device0/type             # Fan, processor
cat /sys/class/thermal/cooling_device0/max_state        # Maximum cooling state
cat /sys/class/thermal/cooling_device0/cur_state        # Current cooling state

# Manual fan control (if supported)
echo 5 | sudo tee /sys/class/thermal/cooling_device0/cur_state

# Hardware monitoring (hwmon)
ls /sys/class/hwmon/
cat /sys/class/hwmon/hwmon0/name
cat /sys/class/hwmon/hwmon0/temp1_input    # Temperature sensor 1
cat /sys/class/hwmon/hwmon0/temp1_label
cat /sys/class/hwmon/hwmon0/fan1_input     # Fan speed in RPM
```

## Power Management

### CPU Frequency Scaling

```bash
# Set governor for all CPUs
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set specific frequency (if using userspace governor)
echo "userspace" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "2400000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed

# Maximum/minimum frequencies
echo "3000000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "1200000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
```

### Device Power States

```bash
# Runtime power management status
cat /sys/class/net/ens0/device/power/runtime_status
# Values: active, suspended, suspending, resuming, unsupported

cat /sys/class/net/ens0/device/power/control
# Values: auto (enable runtime PM), on (disable runtime PM)

# Enable runtime power management
echo "auto" | sudo tee /sys/class/net/ens0/device/power/control

# Disable runtime power management
echo "on" | sudo tee /sys/class/net/ens0/device/power/control

# Wakeup capability
cat /sys/class/net/ens0/device/power/wakeup
echo "enabled" | sudo tee /sys/class/net/ens0/device/power/wakeup
```

### Display Brightness

```bash
# List backlight devices
ls /sys/class/backlight/
# Output: intel_backlight acpi_video0 ...

# Current and maximum brightness
cat /sys/class/backlight/intel_backlight/brightness
cat /sys/class/backlight/intel_backlight/max_brightness
cat /sys/class/backlight/intel_backlight/actual_brightness

# Set brightness (0 to max_brightness)
echo 500 | sudo tee /sys/class/backlight/intel_backlight/brightness

# Set percentage (calculate from max_brightness)
MAX=$(cat /sys/class/backlight/intel_backlight/max_brightness)
echo $((MAX * 50 / 100)) | sudo tee /sys/class/backlight/intel_backlight/brightness
```

### System Power State

```bash
# Available sleep states
cat /sys/power/state
# Output: freeze mem disk

# Enter suspend to RAM
echo "mem" | sudo tee /sys/power/state

# Enter hibernate
echo "disk" | sudo tee /sys/power/state

# Wakeup sources
cat /sys/power/wakeup_count
cat /sys/kernel/wakeup_reasons
```

## LED Control

```bash
# List LEDs
ls /sys/class/leds/
# Output: input3::capslock input3::numlock platform::mute ...

# LED information
cat /sys/class/leds/input3::capslock/brightness    # 0=off, 1=on
cat /sys/class/leds/input3::capslock/max_brightness

# Control LED
echo 1 | sudo tee /sys/class/leds/input3::capslock/brightness
echo 0 | sudo tee /sys/class/leds/input3::capslock/brightness

# Available triggers
cat /sys/class/leds/led0/trigger
# Output: [none] kbd-scrolllock kbd-numlock ... heartbeat cpu mmc0 ...

# Set trigger
echo "heartbeat" | sudo tee /sys/class/leds/led0/trigger
echo "cpu" | sudo tee /sys/class/leds/led0/trigger
echo "none" | sudo tee /sys/class/leds/led0/trigger  # Manual control

# Blinking (if trigger supports it)
echo "timer" | sudo tee /sys/class/leds/led0/trigger
echo 500 | sudo tee /sys/class/leds/led0/delay_on    # milliseconds on
echo 500 | sudo tee /sys/class/leds/led0/delay_off   # milliseconds off
```

## Kernel Parameters

```bash
# Kernel version and information
cat /sys/kernel/osrelease
cat /sys/kernel/ostype
cat /sys/kernel/version

# Boot command line
cat /sys/kernel/cmdline

# Kernel debugging
ls /sys/kernel/debug/     # Requires debugfs mount and root access

# Kernel configuration (if CONFIG_IKCONFIG enabled)
zcat /sys/kernel/config.gz

# Kexec (kernel crash dumps)
ls /sys/kernel/kexec_crash_loaded
cat /sys/kernel/kexec_crash_size

# Kernel messages
cat /sys/kernel/printk
# Format: console_loglevel default_message_loglevel minimum_console_loglevel default_console_loglevel

# Security modules
ls /sys/kernel/security/
cat /sys/kernel/security/lsm  # Loaded security modules

# Memory management
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag

# Profiling
cat /sys/kernel/profiling

# RCU (Read-Copy-Update)
ls /sys/kernel/rcu_expedited
ls /sys/kernel/rcu_normal
```

## Module Parameters

Kernel modules can expose configurable parameters through sysfs.

```bash
# List all loaded modules
ls /sys/module/

# List parameters for a specific module
ls /sys/module/bluetooth/parameters/
# Output: disable_esco disable_ertm ...

# Read parameter value
cat /sys/module/bluetooth/parameters/disable_esco

# Modify parameter (if writable)
echo "Y" | sudo tee /sys/module/bluetooth/parameters/disable_esco
echo "N" | sudo tee /sys/module/bluetooth/parameters/disable_esco

# Check if parameter is writable
ls -l /sys/module/bluetooth/parameters/disable_esco
# -rw-r--r-- = writable
# -r--r--r-- = read-only

# Common module parameters

# Sound (snd_hda_intel)
cat /sys/module/snd_hda_intel/parameters/power_save
echo 1 | sudo tee /sys/module/snd_hda_intel/parameters/power_save

# Networking (e1000e)
cat /sys/module/e1000e/parameters/InterruptThrottleRate

# KVM
cat /sys/module/kvm/parameters/halt_poll_ns
cat /sys/module/kvm_intel/parameters/nested

# Make parameters persistent (add to kernel command line or modprobe.conf)
# /etc/modprobe.d/audio.conf:
# options snd_hda_intel power_save=1
```

## Memory Management

```bash
# Memory information
ls /sys/devices/system/memory/

# Hugepages
ls /sys/kernel/mm/hugepages/
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages

# Allocate hugepages
echo 512 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Transparent Hugepage (THP)
cat /sys/kernel/mm/transparent_hugepage/enabled
# Output: [always] madvise never
echo "madvise" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

cat /sys/kernel/mm/transparent_hugepage/defrag
echo "defer" | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# NUMA information
ls /sys/devices/system/node/
cat /sys/devices/system/node/node0/meminfo
cat /sys/devices/system/node/node0/cpulist
cat /sys/devices/system/node/node0/distance  # NUMA distances

# Memory compaction
cat /sys/kernel/mm/compaction/order
cat /sys/kernel/mm/compaction/node-*-zone-*/

# Page allocation debugging
cat /sys/kernel/debug/extfrag/extfrag_index  # Requires debugfs

# OOM (Out of Memory) control
cat /sys/fs/cgroup/memory/memory.oom_control
```

## Cgroups (Control Groups)

Modern systems use cgroups v2 at `/sys/fs/cgroup/`.

```bash
# Cgroup controllers
cat /sys/fs/cgroup/cgroup.controllers
# Output: cpuset cpu io memory hugetlb pids rdma misc

# Current process cgroup
cat /proc/self/cgroup

# CPU controller
cat /sys/fs/cgroup/user.slice/cpu.max      # quota and period
cat /sys/fs/cgroup/user.slice/cpu.weight   # CPU shares (1-10000)
cat /sys/fs/cgroup/user.slice/cpu.stat

# Memory controller
cat /sys/fs/cgroup/user.slice/memory.current    # Current usage
cat /sys/fs/cgroup/user.slice/memory.max        # Memory limit
cat /sys/fs/cgroup/user.slice/memory.high       # Throttle threshold
cat /sys/fs/cgroup/user.slice/memory.stat       # Detailed statistics

# Set memory limit
echo "1G" | sudo tee /sys/fs/cgroup/mygroup/memory.max

# IO controller
cat /sys/fs/cgroup/user.slice/io.max
cat /sys/fs/cgroup/user.slice/io.stat

# Process IDs controller
cat /sys/fs/cgroup/user.slice/pids.current      # Current PIDs
cat /sys/fs/cgroup/user.slice/pids.max          # PID limit
```

## Firmware and UEFI

```bash
# DMI/SMBIOS information
ls /sys/firmware/dmi/tables/
cat /sys/firmware/dmi/tables/DMI              # Raw DMI table
ls /sys/devices/virtual/dmi/id/
cat /sys/devices/virtual/dmi/id/board_vendor
cat /sys/devices/virtual/dmi/id/board_name
cat /sys/devices/virtual/dmi/id/board_version
cat /sys/devices/virtual/dmi/id/bios_version
cat /sys/devices/virtual/dmi/id/bios_date
cat /sys/devices/virtual/dmi/id/product_name   # System model

# ACPI information
ls /sys/firmware/acpi/tables/
cat /sys/firmware/acpi/tables/DSDT > dsdt.dat

# EFI variables (requires efivarfs)
ls /sys/firmware/efi/efivars/
# Note: Modifying EFI variables can brick your system!

# EFI system information
cat /sys/firmware/efi/systab
ls /sys/firmware/efi/fw_platform_size    # 32 or 64 bit
cat /sys/firmware/efi/runtime-map/*/type

# Devicetree (ARM/embedded systems)
ls /sys/firmware/devicetree/base/
cat /sys/firmware/devicetree/base/model
cat /sys/firmware/devicetree/base/compatible
```

## Device Discovery and Debugging

### Finding Device Information

```bash
# Find device by name
find /sys/devices -name "*usb*" -type d

# Find network interface real path
readlink -f /sys/class/net/ens0

# Find all devices of a driver
ls /sys/bus/pci/drivers/e1000e/

# Find device driver
basename $(readlink /sys/class/net/ens0/device/driver)

# Device uevent information
cat /sys/class/net/ens0/uevent
# Output:
# INTERFACE=ens0
# IFINDEX=2
# DEVTYPE=...

# Modalias (for driver matching)
cat /sys/class/net/ens0/device/modalias
```

### Triggering udev Events

```bash
# Trigger udev event for a device
echo "add" | sudo tee /sys/class/net/ens0/uevent
echo "change" | sudo tee /sys/class/net/ens0/uevent

# Monitor udev events
udevadm monitor --environment --udev

# Force device rescan
echo "- - -" | sudo tee /sys/class/scsi_host/host*/scan  # SCSI rescan
echo 1 | sudo tee /sys/bus/pci/rescan                      # PCI rescan
```

### Interrupt Information

```bash
# Per-IRQ information
ls /sys/kernel/irq/
cat /sys/kernel/irq/0/chip_name
cat /sys/kernel/irq/0/hwirq
cat /sys/kernel/irq/0/type
cat /sys/kernel/irq/0/wakeup
cat /sys/kernel/irq/0/name

# Actions and affinity
cat /sys/kernel/irq/0/actions
cat /sys/kernel/irq/0/smp_affinity          # Bitmask
cat /sys/kernel/irq/0/smp_affinity_list     # CPU list

# Set IRQ affinity
echo "0,1" | sudo tee /sys/kernel/irq/0/smp_affinity_list
```

## Best Practices

### Making Changes Persistent

All sysfs changes are temporary and lost on reboot. To make them persistent:

**1. Using systemd-tmpfiles**

Create `/etc/tmpfiles.d/custom-sysfs.conf`:
```
# Set I/O scheduler for sda
w /sys/block/sda/queue/scheduler - - - - mq-deadline

# Set CPU governor
w /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor - - - - performance
```

**2. Using udev rules**

Create `/etc/udev/rules.d/99-custom.rules`:
```
# Set network MTU
SUBSYSTEM=="net", ACTION=="add", KERNEL=="ens0", ATTR{mtu}="9000"

# Disable USB autosuspend for specific device
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="c52b", ATTR{power/autosuspend}="-1"
```

**3. Using systemd service**

Create `/etc/systemd/system/custom-sysfs.service`:
```ini
[Unit]
Description=Custom sysfs settings
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'

[Install]
WantedBy=multi-user.target
```

**4. Using /etc/rc.local (legacy)**

Add commands to `/etc/rc.local` (if supported):
```bash
#!/bin/sh
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
exit 0
```

### Safety Warnings

- **Critical values**: Changing some values can crash your system or damage hardware
- **Temperature limits**: Don't disable thermal throttling without adequate cooling
- **Voltage/frequency**: Incorrect CPU/GPU settings can cause instability or hardware damage
- **EFI variables**: Modifying EFI variables incorrectly can brick your motherboard
- **Device authorization**: Disabling devices can leave your system unbootable
- **Always test**: Test changes before making them persistent
- **Document changes**: Keep track of what you modify for troubleshooting

### When to Use sysfs vs Other Tools

| Task | Prefer sysfs | Prefer other tool |
|------|--------------|-------------------|
| Reading device info | ✓ Direct, scriptable | `lspci`, `lsusb` for formatted output |
| Network config | Basic info only | `ip`, `nmcli` for full configuration |
| CPU frequency | ✓ Fine-grained control | `cpupower` for convenience |
| Power management | ✓ Device-specific | `systemctl suspend` for system-wide |
| Block devices | ✓ Live monitoring | `lsblk`, `blkid` for overview |
| Kernel parameters | ✓ Runtime changes | `/etc/sysctl.conf` for persistence |
| Module parameters | Reading values | `modprobe` for loading with params |

## Troubleshooting

### Permission Denied

```bash
# Problem: Cannot write to sysfs file
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# bash: /sys/.../scaling_governor: Permission denied

# Solutions:
# 1. Use sudo with tee
echo "performance" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# 2. Use sudo with sh -c
sudo sh -c 'echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'

# 3. Become root
sudo -i
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

### Invalid Argument Errors

```bash
# Problem: Write fails with invalid argument
echo "3000000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
# tee: /sys/.../scaling_max_freq: Invalid argument

# Causes:
# 1. Value out of range - check available values
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

# 2. Wrong format - check what format the file expects
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq  # See current format

# 3. Hardware doesn't support the value
cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq  # Check hardware limits
```

### Finding Missing Devices

```bash
# Device doesn't appear in expected location
ls /sys/class/net/eth0
# ls: cannot access '/sys/class/net/eth0': No such file or directory

# Solutions:
# 1. List all devices of that class
ls /sys/class/net/

# 2. Check if driver is loaded
lsmod | grep e1000e

# 3. Check kernel messages
dmesg | grep -i network

# 4. Check hardware presence
lspci | grep -i ethernet
lsusb  # For USB devices
```

### Read-Only Files

```bash
# Problem: File exists but can't be written
ls -l /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
# -r--r--r-- 1 root root 4096 ...

# This is intentional - cpuinfo_* files are read-only (hardware limits)
# Use scaling_* files instead:
echo "3000000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
```

### Tracing Device Paths

```bash
# Follow symlinks to find real device
ls -l /sys/class/net/ens0
# lrwxrwxrwx ... /sys/class/net/ens0 -> ../../devices/pci0000:00/0000:00:1f.6/net/ens0

# Get full real path
realpath /sys/class/net/ens0

# Find parent devices
readlink /sys/class/net/ens0/device
# ../../0000:00:1f.6

# Find which driver controls device
basename $(readlink /sys/class/net/ens0/device/driver)
# e1000e
```

### Files Disappearing

```bash
# Problem: sysfs path exists but files are missing
ls /sys/class/net/ens0/statistics/
# (some files missing)

# Causes:
# 1. Driver doesn't implement all statistics
# 2. Feature not supported by hardware
# 3. Device in wrong state (down vs up)

# Check device state
cat /sys/class/net/ens0/operstate

# Bring device up if needed
sudo ip link set ens0 up
```

## Summary

sysfs provides a unified, hierarchical interface for interacting with kernel subsystems and hardware. It enables both monitoring and configuration of system parameters, making it invaluable for system administration, debugging, performance tuning, and automation. Remember that changes are temporary unless made persistent through systemd, udev, or other initialization mechanisms.

For most use cases, prefer high-level tools (`ip`, `cpupower`, `ethtool`) for convenience, but use sysfs directly when you need fine-grained control, are writing scripts, or troubleshooting at a low level.
