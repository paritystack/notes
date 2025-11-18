# U-Boot

U-Boot (Universal Boot Loader) is an open-source, primary bootloader used in embedded systems. It provides a flexible and powerful environment for initializing hardware, loading operating systems, and facilitating firmware updates.

## Overview

### What is U-Boot?

U-Boot is a feature-rich bootloader that:
- Initializes hardware components (CPU, RAM, storage devices)
- Loads operating system kernels and device trees
- Provides interactive command-line interface
- Supports network booting and firmware updates
- Enables debugging and system diagnostics

### Common Use Cases

1. **Embedded Linux Systems**: ARM, MIPS, PowerPC boards
2. **Single Board Computers**: Raspberry Pi, BeagleBone
3. **Network Equipment**: Routers, switches
4. **IoT Devices**: Smart home devices, industrial controllers
5. **Development Boards**: Various SoC evaluation kits

## Architecture

### Boot Flow

```
Power On → ROM Boot Code → SPL/TPL → U-Boot Proper → Operating System
```

1. **ROM Boot Code**: First-stage bootloader (immutable, in SoC ROM)
2. **SPL (Secondary Program Loader)**: Minimal U-Boot for resource-constrained environments
3. **U-Boot Proper**: Full-featured bootloader
4. **OS Boot**: Loads Linux kernel, device tree, and initial ramdisk

### Components

- **Board Support Package (BSP)**: Board-specific initialization code
- **Device Drivers**: Support for various peripherals
- **File Systems**: FAT, ext2/3/4, UBIFS, JFFS2
- **Network Stack**: TFTP, NFS, HTTP support
- **Command Shell**: Interactive command-line interface

## Building U-Boot

### Prerequisites

```bash
# Install cross-compilation toolchain
sudo apt-get install gcc-arm-linux-gnueabi
sudo apt-get install device-tree-compiler
sudo apt-get install bison flex
```

### Basic Build Process

```bash
# Clone U-Boot repository
git clone https://github.com/u-boot/u-boot.git
cd u-boot

# Configure for specific board
make <board_name>_defconfig

# Example for Raspberry Pi 3
make rpi_3_defconfig

# Build
make CROSS_COMPILE=arm-linux-gnueabi-

# Build with specific number of threads
make -j8 CROSS_COMPILE=arm-linux-gnueabi-
```

### Common Build Targets

```bash
# ARM 32-bit
make CROSS_COMPILE=arm-linux-gnueabi-

# ARM 64-bit
make CROSS_COMPILE=aarch64-linux-gnu-

# MIPS
make CROSS_COMPILE=mips-linux-gnu-

# PowerPC
make CROSS_COMPILE=powerpc-linux-gnu-
```

### Output Files

- `u-boot.bin`: Raw binary image
- `u-boot.img`: Image with U-Boot header
- `u-boot.elf`: ELF executable (for debugging)
- `SPL/u-boot-spl.bin`: SPL binary (if configured)

## Configuration

### Device Tree

U-Boot uses device trees to describe hardware:

```dts
/ {
    model = "Custom Board";
    compatible = "vendor,board";

    memory {
        reg = <0x80000000 0x40000000>; // 1GB RAM at 0x80000000
    };

    chosen {
        bootargs = "console=ttyS0,115200";
        stdout-path = &uart0;
    };
};
```

### Environment Variables

U-Boot uses environment variables for configuration:

```bash
# View all environment variables
printenv

# Set variable
setenv bootargs 'console=ttyS0,115200 root=/dev/mmcblk0p2 rw'

# Save to persistent storage
saveenv

# Delete variable
setenv bootargs

# Boot with specific arguments
setenv bootcmd 'mmc dev 0; ext4load mmc 0:1 ${kernel_addr} /boot/zImage; bootz ${kernel_addr}'
```

### Default Environment

Common environment variables:

- `bootcmd`: Command(s) to execute automatically on boot
- `bootargs`: Kernel command-line arguments
- `bootdelay`: Delay (in seconds) before auto-boot
- `ipaddr`: Device IP address
- `serverip`: TFTP server IP address
- `ethaddr`: Ethernet MAC address
- `kernel_addr_r`: Kernel load address in RAM
- `fdt_addr_r`: Device tree load address in RAM

## Common Commands

### System Information

```bash
# Display version
version

# Board information
bdinfo

# CPU information
cpu info

# Memory test
mtest 0x80000000 0x80100000
```

### Memory Operations

```bash
# Display memory
md.b 0x80000000 100    # Display 100 bytes
md.w 0x80000000 50     # Display 50 words (16-bit)
md.l 0x80000000 25     # Display 25 long words (32-bit)

# Write to memory
mw.l 0x80000000 0xdeadbeef

# Copy memory
cp.b 0x80000000 0x81000000 1000  # Copy 1000 bytes

# Compare memory
cmp.b 0x80000000 0x81000000 1000

# Fill memory
mw.b 0x80000000 0xff 1000  # Fill 1000 bytes with 0xff
```

### Storage Operations

#### MMC/SD Card

```bash
# List MMC devices
mmc list

# Select MMC device
mmc dev 0

# Display partition info
mmc part

# Read from MMC to RAM
mmc read ${kernel_addr_r} 0x800 0x4000  # Read 0x4000 blocks from offset 0x800

# Write from RAM to MMC
mmc write ${kernel_addr_r} 0x800 0x4000
```

#### File System Operations

```bash
# List files (FAT)
fatls mmc 0:1 /

# Load file from FAT
fatload mmc 0:1 ${kernel_addr_r} /boot/zImage

# List files (ext4)
ext4ls mmc 0:2 /

# Load file from ext4
ext4load mmc 0:2 ${kernel_addr_r} /boot/zImage

# Get file size
ext4size mmc 0:2 /boot/zImage
```

#### USB Storage

```bash
# Initialize USB
usb start

# List USB devices
usb tree

# List storage devices
usb storage

# Access USB storage
fatls usb 0:1 /
fatload usb 0:1 ${kernel_addr_r} /kernel.img
```

### Network Operations

#### Network Configuration

```bash
# Set network parameters
setenv ipaddr 192.168.1.100
setenv netmask 255.255.255.0
setenv serverip 192.168.1.1
setenv gatewayip 192.168.1.1

# Display network settings
print ipaddr serverip

# Test connectivity
ping 192.168.1.1
```

#### TFTP Operations

```bash
# Load file via TFTP
tftp ${kernel_addr_r} zImage

# Load to specific address
tftp 0x82000000 devicetree.dtb

# Load and set filesize
tftp ${kernel_addr_r} zImage
echo ${filesize}
```

#### NFS Boot

```bash
# Set NFS parameters
setenv nfsroot /srv/nfs/rootfs
setenv nfsboot 'setenv bootargs root=/dev/nfs nfsroot=${serverip}:${nfsroot},v3,tcp ip=${ipaddr}; tftp ${kernel_addr_r} zImage; tftp ${fdt_addr_r} board.dtb; bootz ${kernel_addr_r} - ${fdt_addr_r}'

# Boot from NFS
run nfsboot
```

### Boot Commands

```bash
# Boot Linux kernel (ARM zImage)
bootz ${kernel_addr_r} - ${fdt_addr_r}

# Boot Linux kernel (legacy uImage)
bootm ${kernel_addr_r}

# Boot with initramfs
bootz ${kernel_addr_r} ${ramdisk_addr_r} ${fdt_addr_r}

# Boot from specific device
boot mmc 0:1

# Execute saved boot command
boot
```

## Boot Scripts

### Creating Boot Scripts

```bash
# Create text file (boot.cmd)
cat > boot.cmd << 'EOF'
echo "Loading kernel..."
ext4load mmc 0:2 ${kernel_addr_r} /boot/zImage
ext4load mmc 0:2 ${fdt_addr_r} /boot/board.dtb
setenv bootargs console=ttyS0,115200 root=/dev/mmcblk0p2 rw rootwait
echo "Booting kernel..."
bootz ${kernel_addr_r} - ${fdt_addr_r}
EOF

# Compile to boot.scr
mkimage -C none -A arm -T script -d boot.cmd boot.scr

# Place on boot partition
cp boot.scr /media/boot/
```

### Loading and Executing Scripts

```bash
# Load script from SD card
fatload mmc 0:1 ${scriptaddr} boot.scr

# Execute script
source ${scriptaddr}

# Or in one command
fatload mmc 0:1 ${scriptaddr} boot.scr; source ${scriptaddr}
```

### Automatic Boot Script

Set `bootcmd` to automatically load and execute script:

```bash
setenv bootcmd 'fatload mmc 0:1 ${scriptaddr} boot.scr; source ${scriptaddr}'
saveenv
```

## Firmware Updates

### Update Strategies

#### 1. TFTP Update

```bash
# Load new U-Boot via TFTP
tftp ${loadaddr} u-boot.bin

# Flash to storage
mmc dev 0
mmc write ${loadaddr} 0x100 0x800  # Write to offset 0x100, 0x800 blocks
```

#### 2. USB Update

```bash
# Initialize USB
usb start

# Load U-Boot from USB
fatload usb 0:1 ${loadaddr} u-boot.bin

# Flash to MMC
mmc dev 0
mmc write ${loadaddr} 0x100 0x800
```

#### 3. SD Card Update

```bash
# Load from SD card
fatload mmc 0:1 ${loadaddr} u-boot.bin

# Write to eMMC
mmc dev 1
mmc write ${loadaddr} 0x100 0x800
```

### Kernel Update Script

```bash
# boot_update.cmd
echo "Kernel Update Script"
if fatload mmc 0:1 ${kernel_addr_r} zImage.new; then
    echo "New kernel found, updating..."
    ext4write mmc 0:2 ${kernel_addr_r} /boot/zImage ${filesize}
    echo "Kernel updated successfully"
else
    echo "No update found, booting normally..."
fi
ext4load mmc 0:2 ${kernel_addr_r} /boot/zImage
ext4load mmc 0:2 ${fdt_addr_r} /boot/board.dtb
bootz ${kernel_addr_r} - ${fdt_addr_r}
```

## Advanced Features

### Secure Boot

#### Verified Boot

```bash
# Enable verified boot in configuration
CONFIG_FIT=y
CONFIG_FIT_SIGNATURE=y
CONFIG_RSA=y

# Create FIT image with signature
mkimage -f kernel.its kernel.itb
```

#### FIT Image Example (kernel.its)

```dts
/dts-v1/;

/ {
    description = "Signed Kernel Image";
    #address-cells = <1>;

    images {
        kernel {
            description = "Linux Kernel";
            data = /incbin/("zImage");
            type = "kernel";
            arch = "arm";
            os = "linux";
            compression = "none";
            load = <0x80008000>;
            entry = <0x80008000>;
            hash-1 {
                algo = "sha256";
            };
        };

        fdt {
            description = "Device Tree";
            data = /incbin/("board.dtb");
            type = "flat_dt";
            arch = "arm";
            compression = "none";
            hash-1 {
                algo = "sha256";
            };
        };
    };

    configurations {
        default = "config-1";
        config-1 {
            description = "Boot Configuration";
            kernel = "kernel";
            fdt = "fdt";
            signature-1 {
                algo = "sha256,rsa2048";
                key-name-hint = "dev";
                sign-images = "kernel", "fdt";
            };
        };
    };
};
```

### Falcon Mode (Fast Boot)

Falcon mode skips U-Boot shell and boots directly to OS:

```bash
# Prepare SPL to load kernel directly
setenv bootcmd 'spl export fdt ${kernel_addr_r} - ${fdt_addr_r}'

# Save configuration
saveenv

# SPL will load kernel directly on next boot
```

### DFU (Device Firmware Update)

```bash
# Configure DFU
setenv dfu_alt_info 'kernel ram ${kernel_addr_r} 0x1000000; dtb ram ${fdt_addr_r} 0x100000'

# Start DFU mode
dfu 0 ram 0

# From host PC
dfu-util -a kernel -D zImage
dfu-util -a dtb -D board.dtb
```

### UMS (USB Mass Storage)

Expose storage device as USB mass storage:

```bash
# Expose MMC as USB storage
ums 0 mmc 0

# From host, device appears as /dev/sdX
# Can be mounted and modified directly
```

## Debugging

### Serial Console

Default serial console configuration:
- Baud rate: 115200
- Data bits: 8
- Parity: None
- Stop bits: 1
- Flow control: None

### Debug Messages

```bash
# Enable verbose output
setenv bootargs ${bootargs} loglevel=7

# Debug specific subsystems
setenv debug 1
```

### Memory Dump

```bash
# Dump memory region to console
md.b ${kernel_addr_r} 0x100

# Search for pattern in memory
while true; do
    if itest.l *${search_addr} == 0xdeadbeef; then
        echo "Pattern found at ${search_addr}"
        exit
    fi
    setexpr search_addr ${search_addr} + 4
done
```

### GPIO Control

```bash
# Read GPIO
gpio input 42

# Set GPIO output
gpio set 42    # Set high
gpio clear 42  # Set low
gpio toggle 42 # Toggle state
```

### I2C Operations

```bash
# Scan I2C bus
i2c dev 0
i2c probe

# Read from I2C device
i2c read 0x50 0x00 1 ${loadaddr} 0x100

# Write to I2C device
i2c write ${loadaddr} 0x50 0x00 1 0x100
```

## Environment Storage

### Storage Locations

1. **MMC/SD Card**
```bash
CONFIG_ENV_IS_IN_MMC=y
CONFIG_ENV_OFFSET=0x400000
CONFIG_ENV_SIZE=0x2000
```

2. **SPI Flash**
```bash
CONFIG_ENV_IS_IN_SPI_FLASH=y
CONFIG_ENV_OFFSET=0x100000
CONFIG_ENV_SIZE=0x2000
```

3. **NAND Flash**
```bash
CONFIG_ENV_IS_IN_NAND=y
CONFIG_ENV_OFFSET=0x400000
CONFIG_ENV_SIZE=0x20000
```

4. **FAT Filesystem**
```bash
CONFIG_ENV_IS_IN_FAT=y
CONFIG_ENV_FAT_DEVICE_AND_PART="0:1"
CONFIG_ENV_FAT_FILE="uboot.env"
```

### Managing Environment

```bash
# Reset to default environment
env default -a

# Save current environment
saveenv

# Import environment from file
env import -t ${loadaddr} ${filesize}

# Export environment to file
env export -t ${loadaddr}
fatwrite mmc 0:1 ${loadaddr} uboot_env.txt ${filesize}
```

## Performance Optimization

### Boot Time Reduction

1. **Disable unnecessary features**
```bash
setenv bootdelay 0        # Skip delay
setenv silent 1           # Reduce console output
```

2. **Optimize boot command**
```bash
# Direct boot without scripts
setenv bootcmd 'mmc dev 0; ext4load mmc 0:2 0x82000000 /boot/zImage; ext4load mmc 0:2 0x88000000 /boot/board.dtb; bootz 0x82000000 - 0x88000000'
```

3. **Use Falcon mode**
```bash
# SPL loads kernel directly
CONFIG_SPL_OS_BOOT=y
```

### Memory Configuration

```bash
# Optimize cache settings
icache on
dcache on

# Set appropriate load addresses
setenv kernel_addr_r 0x82000000
setenv fdt_addr_r 0x88000000
setenv ramdisk_addr_r 0x88080000
```

## Troubleshooting

### Common Issues

#### 1. U-Boot Won't Start

**Symptoms**: No output on serial console

**Solutions**:
- Check serial connection (correct pins, baud rate)
- Verify power supply
- Check boot mode pins/switches
- Ensure U-Boot binary is correctly flashed

#### 2. Bad Magic Number

**Error**: `Bad Magic Number`

**Solutions**:
```bash
# Recreate boot image with correct header
mkimage -A arm -O linux -T kernel -C none -a 0x80008000 -e 0x80008000 -n "Linux" -d zImage uImage
```

#### 3. FDT Load Error

**Error**: `ERROR: Did not find a cmdline Flattened Device Tree`

**Solutions**:
- Verify device tree is loaded: `fdt addr ${fdt_addr_r}`
- Check device tree path and filename
- Ensure sufficient memory at fdt_addr_r

#### 4. Network Boot Fails

**Symptoms**: TFTP timeout or connection refused

**Solutions**:
```bash
# Verify network settings
print ipaddr serverip

# Test connectivity
ping ${serverip}

# Check TFTP server is running
# On host: sudo systemctl status tftpd-hpa

# Verify firewall allows TFTP (port 69)
```

#### 5. Environment Not Saving

**Symptoms**: `saveenv` fails or changes don't persist

**Solutions**:
- Check environment storage is configured correctly
- Verify storage device is accessible
- Ensure sufficient space and write permissions
- Try resetting environment: `env default -a; saveenv`

### Recovery Mode

#### Enter U-Boot Shell

1. Power on device
2. Press any key during bootdelay countdown
3. Interrupt auto-boot

#### Recover from Bad Environment

```bash
# Reset to default
env default -a
saveenv
reset
```

#### Recover from Bad Boot Command

```bash
# At U-Boot prompt
setenv bootcmd 'echo Safe mode; exit'
saveenv
reset
```

## Best Practices

### 1. Always Test Before Deployment

```bash
# Test boot without saving
bootm ${kernel_addr_r}

# Only save after verification
saveenv
```

### 2. Keep Backup Environment

```bash
# Export current environment
env export -t ${loadaddr}
fatwrite mmc 0:1 ${loadaddr} env_backup.txt ${filesize}
```

### 3. Use Meaningful Variable Names

```bash
# Good
setenv production_kernel '/boot/zImage-stable'

# Avoid
setenv k '/boot/zImage-stable'
```

### 4. Document Custom Scripts

```bash
# Add comments in boot scripts
echo "=== Custom Boot Script v1.2 ==="
echo "Loading kernel from eMMC..."
```

### 5. Implement Fallback Mechanisms

```bash
# Try primary, fallback to backup
if ext4load mmc 0:2 ${kernel_addr_r} /boot/zImage; then
    echo "Primary kernel loaded"
else
    echo "Primary failed, loading backup..."
    ext4load mmc 0:2 ${kernel_addr_r} /boot/zImage.backup
fi
```

### 6. Version Control Configuration

Keep track of U-Boot version and configuration:
```bash
# Add version to environment
setenv uboot_version 'U-Boot 2023.10 Custom Build 1.0'
```

## References

- [Official U-Boot Documentation](https://u-boot.readthedocs.io/)
- [U-Boot Source Repository](https://github.com/u-boot/u-boot)
- [Device Tree Specification](https://devicetree.org/)
- [Embedded Linux Wiki - U-Boot](https://elinux.org/U-Boot)
- [Bootlin Training Materials](https://bootlin.com/docs/)

## Related Topics

- Embedded Linux Boot Process
- Device Tree
- ARM Architecture
- Kernel Configuration
- Cross-compilation
- Embedded System Development
