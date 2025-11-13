# Cross Compilation

A comprehensive guide to cross compilation for Linux - building software on one platform (host) to run on a different platform (target).

## Table of Contents

- [Overview](#overview)
- [Why Cross Compilation?](#why-cross-compilation)
- [Terminology](#terminology)
- [Toolchain Setup](#toolchain-setup)
- [Cross Compiling the Linux Kernel](#cross-compiling-the-linux-kernel)
- [Cross Compiling User Space Applications](#cross-compiling-user-space-applications)
- [Build System Support](#build-system-support)
- [Root Filesystem Creation](#root-filesystem-creation)
- [Debugging Cross-Compiled Code](#debugging-cross-compiled-code)
- [Common Architectures](#common-architectures)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

**Cross Compilation** is the process of building executable code on one system (the **host**) that will run on a different system (the **target**). This is essential for embedded systems development where the target device may have limited resources or a different architecture.

### Typical Scenario

```
┌─────────────────────┐         ┌─────────────────────┐
│   Host System       │         │   Target System     │
│   x86_64 Linux      │         │   ARM Cortex-A8     │
│   Development PC    │  ────>  │   Embedded Board    │
│                     │         │                     │
│ - Fast CPU          │         │ - Slow CPU          │
│ - Lots of RAM       │         │ - Limited RAM       │
│ - Large Storage     │         │ - Small Storage     │
└─────────────────────┘         └─────────────────────┘
```

**Build on host → Deploy to target**

---

## Why Cross Compilation?

### Reasons for Cross Compilation

1. **Limited Target Resources**
   - Embedded devices lack CPU power, RAM, or storage for compilation
   - Building natively would take hours or fail due to memory constraints

2. **Architecture Differences**
   - Development machine (x86_64) differs from target (ARM, MIPS, etc.)
   - Cannot run x86 binaries on ARM without emulation

3. **Speed**
   - Powerful development machine compiles much faster than embedded target
   - Native compilation on Raspberry Pi: 2 hours → Cross compilation: 10 minutes

4. **Tooling**
   - Better development tools available on host
   - Easier debugging and profiling setup

5. **Consistency**
   - Reproducible builds across team
   - Controlled toolchain versions

### Example: Raspberry Pi

**Native compilation on Pi 3:**
```bash
# Building Linux kernel natively
$ time make -j4
real    120m0.000s   # 2 hours!
```

**Cross compilation on x86_64 PC:**
```bash
# Cross compiling same kernel
$ time make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- -j8
real    10m0.000s    # 10 minutes!
```

---

## Terminology

### Key Terms

- **Host**: System where compilation happens (your development PC)
- **Build**: System where build tools run (usually same as host)
- **Target**: System where compiled code will run (embedded device)

- **Toolchain**: Collection of tools for cross compilation
  - Compiler (gcc, clang)
  - Linker (ld)
  - Assembler (as)
  - Libraries (libc, libgcc)
  - Utilities (objcopy, objdump, strip)

- **Triple/Tuple**: Architecture specification format
  - Format: `arch-vendor-os-abi`
  - Example: `arm-linux-gnueabihf`
    - `arm`: Architecture (ARM)
    - `linux`: OS (Linux)
    - `gnueabihf`: ABI (GNU EABI Hard Float)

- **Sysroot**: Target system's root filesystem on host
  - Contains target's headers and libraries
  - Located on development machine
  - Mimics target's `/usr`, `/lib`, etc.

### Architecture Tuples

```bash
# Common architecture tuples
arm-linux-gnueabi         # ARM soft-float
arm-linux-gnueabihf       # ARM hard-float
aarch64-linux-gnu         # ARM 64-bit
mips-linux-gnu            # MIPS
mipsel-linux-gnu          # MIPS little-endian
powerpc-linux-gnu         # PowerPC
x86_64-w64-mingw32        # Windows 64-bit
```

---

## Toolchain Setup

### Option 1: Pre-built Toolchains

**Install from Distribution:**
```bash
# Ubuntu/Debian - ARM
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# ARM64
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# MIPS
sudo apt-get install gcc-mips-linux-gnu g++-mips-linux-gnu

# Verify installation
arm-linux-gnueabihf-gcc --version
```

**Linaro Toolchains:**
```bash
# Download from Linaro
wget https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabihf/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz

# Extract
tar xf gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz

# Add to PATH
export PATH=$PATH:$(pwd)/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf/bin

# Test
arm-linux-gnueabihf-gcc --version
```

### Option 2: Crosstool-NG

Build custom toolchains:

```bash
# Install crosstool-NG
git clone https://github.com/crosstool-ng/crosstool-ng
cd crosstool-ng
./bootstrap
./configure --prefix=/opt/crosstool-ng
make
sudo make install

# Add to PATH
export PATH=/opt/crosstool-ng/bin:$PATH

# Configure and build
ct-ng list-samples
ct-ng arm-unknown-linux-gnueabi
ct-ng menuconfig  # Configure as needed
ct-ng build

# Toolchain installed in ~/x-tools/arm-unknown-linux-gnueabi/
```

### Option 3: Buildroot

Creates complete embedded Linux system including toolchain:

```bash
# Download Buildroot
wget https://buildroot.org/downloads/buildroot-2023.02.tar.gz
tar xf buildroot-2023.02.tar.gz
cd buildroot-2023.02

# Configure
make menuconfig
# Target options -> Target Architecture -> ARM
# Toolchain -> Build toolchain

# Build
make

# Toolchain in output/host/usr/bin/
export PATH=$PATH:$(pwd)/output/host/usr/bin
```

### Setting Up Environment

**Permanent setup:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export CROSS_COMPILE=arm-linux-gnueabihf-
export ARCH=arm
export PATH=$PATH:/path/to/toolchain/bin

# Apply
source ~/.bashrc
```

**Project-specific:**
```bash
# Create toolchain.env
cat > toolchain.env << 'EOF'
export CROSS_COMPILE=arm-linux-gnueabihf-
export ARCH=arm
export PATH=/opt/arm-toolchain/bin:$PATH
export SYSROOT=/opt/arm-sysroot
EOF

# Source when needed
source toolchain.env
```

### Verifying Toolchain

```bash
# Check compiler
${CROSS_COMPILE}gcc --version
${CROSS_COMPILE}gcc -v

# Check target
${CROSS_COMPILE}gcc -dumpmachine
# Output: arm-linux-gnueabihf

# List all tools
ls -la $(dirname $(which ${CROSS_COMPILE}gcc))/${CROSS_COMPILE}*
```

---

## Cross Compiling the Linux Kernel

### Basic Kernel Cross Compilation

```bash
# Get kernel source
git clone https://github.com/torvalds/linux.git
cd linux

# Clean
make mrproper

# Configure for ARM (example: Versatile Express)
make ARCH=arm vexpress_defconfig

# Or use menuconfig
make ARCH=arm menuconfig

# Build
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- -j$(nproc)

# Build specific targets
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- modules
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- dtbs

# Install modules to staging directory
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
     INSTALL_MOD_PATH=/path/to/rootfs modules_install
```

### Raspberry Pi Kernel

```bash
# Clone Raspberry Pi kernel
git clone --depth=1 https://github.com/raspberrypi/linux
cd linux

# Pi 1, Zero, Zero W (32-bit)
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- bcmrpi_defconfig

# Pi 2, 3, 4 (32-bit)
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- bcm2709_defconfig

# Pi 3, 4 (64-bit)
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- bcm2711_defconfig

# Build
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage modules dtbs -j$(nproc)

# Install to SD card
export ROOTFS=/mnt/ext4
export BOOTFS=/mnt/fat32

# Install modules
sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
     INSTALL_MOD_PATH=$ROOTFS modules_install

# Install kernel
sudo cp arch/arm/boot/zImage $BOOTFS/kernel7.img
sudo cp arch/arm/boot/dts/*.dtb $BOOTFS/
sudo cp arch/arm/boot/dts/overlays/*.dtb* $BOOTFS/overlays/
```

### BeagleBone Black Kernel

```bash
# Clone kernel
git clone https://github.com/beagleboard/linux.git
cd linux

# Checkout stable branch
git checkout 5.10

# Configure
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- bb.org_defconfig

# Build
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage modules dtbs -j$(nproc)

# Create uImage (U-Boot format)
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- uImage \
     LOADADDR=0x80008000

# Install
sudo cp arch/arm/boot/uImage /media/$USER/BOOT/
sudo cp arch/arm/boot/dts/am335x-boneblack.dtb /media/$USER/BOOT/
sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
     INSTALL_MOD_PATH=/media/$USER/rootfs modules_install
```

### Kernel Configuration Tips

```bash
# Use existing config from target
scp user@target:/proc/config.gz .
zcat config.gz > .config
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- oldconfig

# Save custom config
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- savedefconfig
cp defconfig arch/arm/configs/myboard_defconfig

# Enable specific features
./scripts/config --enable CONFIG_FEATURE_NAME
./scripts/config --disable CONFIG_FEATURE_NAME
./scripts/config --module CONFIG_FEATURE_NAME
```

---

## Cross Compiling User Space Applications

### Simple C Program

```c
/* hello.c */
#include <stdio.h>

int main(void)
{
	printf("Hello from %s!\n",
#ifdef __arm__
		"ARM"
#elif __aarch64__
		"ARM64"
#elif __mips__
		"MIPS"
#else
		"unknown"
#endif
	);
	return 0;
}
```

**Compile:**
```bash
# Cross compile
arm-linux-gnueabihf-gcc hello.c -o hello

# Check architecture
file hello
# hello: ELF 32-bit LSB executable, ARM, version 1 (SYSV)

# Check dynamic libraries
arm-linux-gnueabihf-readelf -d hello | grep NEEDED
```

### Static vs Dynamic Linking

**Dynamic linking (default):**
```bash
# Requires target's libc at runtime
arm-linux-gnueabihf-gcc hello.c -o hello

# List dependencies
arm-linux-gnueabihf-ldd hello
# or
arm-linux-gnueabihf-readelf -d hello
```

**Static linking:**
```bash
# Includes all libraries in binary
arm-linux-gnueabihf-gcc hello.c -o hello -static

# Check - no dependencies
file hello
# hello: ELF 32-bit LSB executable, ARM, statically linked

# Size comparison
ls -lh hello
# Much larger with static linking
```

### Cross Compiling with Libraries

```c
/* http_client.c - requires libcurl */
#include <curl/curl.h>
#include <stdio.h>

int main(void)
{
	CURL *curl = curl_easy_init();
	if (curl) {
		curl_easy_cleanup(curl);
		printf("libcurl working!\n");
	}
	return 0;
}
```

**Without sysroot (will fail):**
```bash
arm-linux-gnueabihf-gcc http_client.c -o http_client -lcurl
# Error: curl/curl.h: No such file or directory
```

**With sysroot:**
```bash
# Install target libraries on host
sudo apt-get install libcurl4-openssl-dev:armhf

# Compile with sysroot
arm-linux-gnueabihf-gcc http_client.c -o http_client \
	--sysroot=/usr/arm-linux-gnueabihf \
	-lcurl

# Or set PKG_CONFIG
export PKG_CONFIG_PATH=/usr/arm-linux-gnueabihf/lib/pkgconfig
arm-linux-gnueabihf-gcc http_client.c -o http_client \
	$(pkg-config --cflags --libs libcurl)
```

### Makefile for Cross Compilation

```makefile
# Makefile
CC := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++
LD := $(CROSS_COMPILE)ld
AR := $(CROSS_COMPILE)ar
STRIP := $(CROSS_COMPILE)strip

CFLAGS := -Wall -O2
LDFLAGS :=

# Add sysroot if set
ifdef SYSROOT
CFLAGS += --sysroot=$(SYSROOT)
LDFLAGS += --sysroot=$(SYSROOT)
endif

TARGET := myapp
SRCS := main.c utils.c
OBJS := $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^
	$(STRIP) $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

**Usage:**
```bash
# Native compilation
make

# Cross compilation
make CROSS_COMPILE=arm-linux-gnueabihf-

# With sysroot
make CROSS_COMPILE=arm-linux-gnueabihf- SYSROOT=/opt/arm-sysroot
```

---

## Build System Support

### Autotools (./configure)

```bash
# Basic cross compilation
./configure --host=arm-linux-gnueabihf --prefix=/usr

# With sysroot
./configure \
	--host=arm-linux-gnueabihf \
	--prefix=/usr \
	--with-sysroot=/opt/arm-sysroot \
	CFLAGS="--sysroot=/opt/arm-sysroot" \
	LDFLAGS="--sysroot=/opt/arm-sysroot"

# Build and install
make
make DESTDIR=/path/to/rootfs install
```

**config.site for consistent configuration:**
```bash
# Create config.site
cat > arm-config.site << 'EOF'
# Cross compilation settings
ac_cv_func_malloc_0_nonnull=yes
ac_cv_func_realloc_0_nonnull=yes
EOF

# Use it
./configure --host=arm-linux-gnueabihf --prefix=/usr \
	CONFIG_SITE=arm-config.site
```

### CMake

**Toolchain file:**
```cmake
# arm-toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Specify the cross compiler
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# Where to look for libraries
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

**Build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-toolchain.cmake
make
```

**Or using environment variables:**
```bash
export CC=arm-linux-gnueabihf-gcc
export CXX=arm-linux-gnueabihf-g++
cmake ..
make
```

### Meson

**Cross file:**
```ini
# arm-cross.txt
[binaries]
c = 'arm-linux-gnueabihf-gcc'
cpp = 'arm-linux-gnueabihf-g++'
ar = 'arm-linux-gnueabihf-ar'
strip = 'arm-linux-gnueabihf-strip'
pkgconfig = 'arm-linux-gnueabihf-pkg-config'

[host_machine]
system = 'linux'
cpu_family = 'arm'
cpu = 'cortex-a8'
endian = 'little'

[properties]
sys_root = '/usr/arm-linux-gnueabihf'
```

**Build:**
```bash
meson setup build --cross-file arm-cross.txt
ninja -C build
```

---

## Root Filesystem Creation

### Using Buildroot

```bash
# Configure
make menuconfig

# Filesystem images -> ext2/3/4 root filesystem
# System configuration -> System hostname, root password
# Target packages -> Select packages

# Build
make

# Output
ls output/images/
# rootfs.ext4  zImage  *.dtb
```

### Using Yocto/OpenEmbedded

```bash
# Clone Poky
git clone -b kirkstone git://git.yoctoproject.org/poky
cd poky

# Initialize build
source oe-init-build-env

# Edit conf/local.conf
# MACHINE = "beaglebone-yocto"

# Build minimal image
bitbake core-image-minimal

# Output in tmp/deploy/images/beaglebone-yocto/
```

### Manual Root Filesystem

```bash
#!/bin/bash
# create-rootfs.sh

ROOTFS=/tmp/arm-rootfs
TOOLCHAIN=arm-linux-gnueabihf

# Create directory structure
mkdir -p $ROOTFS/{bin,sbin,etc,proc,sys,dev,lib,usr/{bin,sbin,lib},tmp,var,home,root}

# Copy libraries from toolchain
SYSROOT=$(${TOOLCHAIN}-gcc -print-sysroot)
cp -a $SYSROOT/lib/* $ROOTFS/lib/
cp -a $SYSROOT/usr/lib/* $ROOTFS/usr/lib/

# Install busybox (provides basic utilities)
git clone https://git.busybox.net/busybox
cd busybox
make ARCH=arm CROSS_COMPILE=$TOOLCHAIN- defconfig
make ARCH=arm CROSS_COMPILE=$TOOLCHAIN- -j$(nproc)
make ARCH=arm CROSS_COMPILE=$TOOLCHAIN- \
	CONFIG_PREFIX=$ROOTFS install
cd ..

# Create device nodes
sudo mknod -m 666 $ROOTFS/dev/null c 1 3
sudo mknod -m 666 $ROOTFS/dev/console c 5 1
sudo mknod -m 666 $ROOTFS/dev/tty c 5 0

# Create /etc/inittab
cat > $ROOTFS/etc/inittab << 'EOF'
::sysinit:/etc/init.d/rcS
::respawn:/sbin/getty 115200 console
::shutdown:/bin/umount -a -r
::restart:/sbin/init
EOF

# Create init script
mkdir -p $ROOTFS/etc/init.d
cat > $ROOTFS/etc/init.d/rcS << 'EOF'
#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t tmpfs none /tmp
echo "Boot complete"
EOF
chmod +x $ROOTFS/etc/init.d/rcS

# Create filesystem image
dd if=/dev/zero of=rootfs.ext4 bs=1M count=512
mkfs.ext4 rootfs.ext4
mkdir -p /mnt/rootfs
sudo mount rootfs.ext4 /mnt/rootfs
sudo cp -a $ROOTFS/* /mnt/rootfs/
sudo umount /mnt/rootfs

echo "Root filesystem created: rootfs.ext4"
```

---

## Debugging Cross-Compiled Code

### Remote GDB Debugging

**On target (ARM device):**
```bash
# Install gdbserver (if not already present)
# Run application under gdbserver
gdbserver :1234 ./myapp arg1 arg2
```

**On host (development PC):**
```bash
# Use cross-gdb
arm-linux-gnueabihf-gdb ./myapp

# In GDB
(gdb) target remote target-ip:1234
(gdb) break main
(gdb) continue
(gdb) step
(gdb) print variable
(gdb) backtrace
```

**GDB script for convenience:**
```bash
# .gdbinit
target remote 192.168.1.100:1234
break main
```

### QEMU User Mode

Run ARM binaries on x86 using QEMU:

```bash
# Install QEMU user mode
sudo apt-get install qemu-user-static

# Run ARM binary
qemu-arm-static -L /usr/arm-linux-gnueabihf ./hello

# With GDB
qemu-arm-static -L /usr/arm-linux-gnueabihf -g 1234 ./hello

# In another terminal
arm-linux-gnueabihf-gdb ./hello
(gdb) target remote :1234
```

### QEMU System Mode

Emulate entire ARM system:

```bash
# Install QEMU system
sudo apt-get install qemu-system-arm

# Run with kernel and rootfs
qemu-system-arm \
	-M vexpress-a9 \
	-kernel zImage \
	-dtb vexpress-v2p-ca9.dtb \
	-drive file=rootfs.ext4,if=sd,format=raw \
	-append "console=ttyAMA0 root=/dev/mmcblk0 rootwait" \
	-serial stdio \
	-net nic -net user
```

### Analyzing Binaries

```bash
# Check architecture
file myapp
arm-linux-gnueabihf-readelf -h myapp

# List symbols
arm-linux-gnueabihf-nm myapp

# Disassemble
arm-linux-gnueabihf-objdump -d myapp

# Check shared library dependencies
arm-linux-gnueabihf-readelf -d myapp | grep NEEDED

# Strings in binary
arm-linux-gnueabihf-strings myapp

# Size information
arm-linux-gnueabihf-size myapp
```

---

## Common Architectures

### ARM (32-bit)

```bash
# Soft-float (no FPU)
CROSS_COMPILE=arm-linux-gnueabi-
ARCH=arm

# Hard-float (with FPU)
CROSS_COMPILE=arm-linux-gnueabihf-
ARCH=arm

# Kernel config
make ARCH=arm multi_v7_defconfig
```

### ARM64 (AArch64)

```bash
CROSS_COMPILE=aarch64-linux-gnu-
ARCH=arm64

# Kernel config
make ARCH=arm64 defconfig
```

### MIPS

```bash
# Big-endian
CROSS_COMPILE=mips-linux-gnu-
ARCH=mips

# Little-endian
CROSS_COMPILE=mipsel-linux-gnu-
ARCH=mips

# Kernel config
make ARCH=mips malta_defconfig
```

### RISC-V

```bash
# 64-bit
CROSS_COMPILE=riscv64-linux-gnu-
ARCH=riscv

# 32-bit
CROSS_COMPILE=riscv32-linux-gnu-
ARCH=riscv

# Kernel config
make ARCH=riscv defconfig
```

### PowerPC

```bash
CROSS_COMPILE=powerpc-linux-gnu-
ARCH=powerpc

# Kernel config
make ARCH=powerpc pmac32_defconfig
```

---

## Troubleshooting

### Common Issues

**Issue: "No such file or directory" for header files**
```bash
# Problem: Headers not found
arm-linux-gnueabihf-gcc test.c
# test.c:1:10: fatal error: curl/curl.h: No such file or directory

# Solution: Install cross-compiled development package
sudo apt-get install libcurl4-openssl-dev:armhf

# Or specify include path
arm-linux-gnueabihf-gcc test.c \
	-I/usr/arm-linux-gnueabihf/include
```

**Issue: "cannot find -lxxx" linker errors**
```bash
# Problem: Library not found
# /usr/bin/ld: cannot find -lssl

# Solution: Install library for target architecture
sudo apt-get install libssl-dev:armhf

# Or specify library path
arm-linux-gnueabihf-gcc test.c -lssl \
	-L/usr/arm-linux-gnueabihf/lib
```

**Issue: Binary runs on host but not target**
```bash
# Check architecture
file myapp
# If says x86_64 instead of ARM, CROSS_COMPILE wasn't set

# Verify you're using cross compiler
which ${CROSS_COMPILE}gcc

# Check if it's stripped of debug info
${CROSS_COMPILE}readelf -S myapp | grep debug
```

**Issue: "Exec format error" on target**
```bash
# Problem: Wrong architecture or ABI mismatch

# Check target's actual architecture
ssh target 'uname -m'  # armv7l, aarch64, etc.

# Check binary architecture
file myapp

# For ARM: Check float ABI
${CROSS_COMPILE}readelf -A myapp | grep ABI
# Must match target's ABI (soft-float vs hard-float)
```

**Issue: Shared library not found on target**
```bash
# Error on target
./myapp: error while loading shared libraries: libfoo.so.1

# Solution 1: Copy library to target
scp /usr/arm-linux-gnueabihf/lib/libfoo.so.* target:/lib/

# Solution 2: Static linking
arm-linux-gnueabihf-gcc test.c -o myapp -static

# Solution 3: Use LD_LIBRARY_PATH on target
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
```

### Debugging Tips

```bash
# Verbose compiler output
arm-linux-gnueabihf-gcc -v test.c

# Show search paths
arm-linux-gnueabihf-gcc -print-search-dirs

# Show sysroot
arm-linux-gnueabihf-gcc -print-sysroot

# Preprocessor output only
arm-linux-gnueabihf-gcc -E test.c

# Show include paths
echo | arm-linux-gnueabihf-gcc -v -E -

# Test if toolchain works
arm-linux-gnueabihf-gcc -v
```

---

## Best Practices

### 1. Use Consistent Toolchain

```bash
# Bad: Mixing toolchains
gcc myapp.c  # Native compiler!
arm-linux-gnueabihf-gcc mylib.c

# Good: Use CROSS_COMPILE consistently
export CROSS_COMPILE=arm-linux-gnueabihf-
${CROSS_COMPILE}gcc myapp.c mylib.c
```

### 2. Separate Build Directories

```bash
# Keep source clean
mkdir -p build/arm build/x86

# ARM build
make O=build/arm ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-

# x86 build
make O=build/x86

# Clean specific build
rm -rf build/arm
```

### 3. Use Build Scripts

```bash
#!/bin/bash
# build-cross.sh

set -e  # Exit on error

# Configuration
export ARCH=arm
export CROSS_COMPILE=arm-linux-gnueabihf-
export INSTALL_PATH=/opt/target-root

# Build
echo "Building for $ARCH..."
make clean
make -j$(nproc)
make install DESTDIR=$INSTALL_PATH

echo "Build complete: $INSTALL_PATH"
```

### 4. Maintain Sysroot

```bash
# Organized sysroot
/opt/arm-sysroot/
├── usr/
│   ├── include/  # Headers
│   └── lib/      # Libraries
├── lib/          # System libraries
└── etc/          # Configuration files

# Set PKG_CONFIG for libraries
export PKG_CONFIG_PATH=/opt/arm-sysroot/usr/lib/pkgconfig
export PKG_CONFIG_SYSROOT_DIR=/opt/arm-sysroot
```

### 5. Version Control Binaries

```bash
# Tag releases
git tag -a v1.0-arm -m "ARM release v1.0"

# Separate binary artifacts
artifacts/
├── v1.0/
│   ├── arm/
│   │   ├── myapp
│   │   └── myapp.debug
│   ├── arm64/
│   └── x86_64/
```

### 6. Automate Testing

```bash
#!/bin/bash
# test-cross.sh

# Build
./build-cross.sh

# Copy to target
scp build/myapp target:/tmp/

# Run on target
ssh target "/tmp/myapp --test"

# Check exit code
if [ $? -eq 0 ]; then
	echo "Tests passed"
else
	echo "Tests failed"
	exit 1
fi
```

### 7. Document Dependencies

```bash
# dependencies.txt
Toolchain: gcc-arm-linux-gnueabihf-9.3
Libraries:
  - libssl-dev:armhf (>= 1.1.1)
  - libcurl4-openssl-dev:armhf (>= 7.68.0)
  - zlib1g-dev:armhf

Kernel: 5.10 or later
Bootloader: U-Boot 2021.01
```

### 8. Optimize for Target

```bash
# Compiler optimizations
CFLAGS="-O2 -march=armv7-a -mtune=cortex-a8 -mfpu=neon"

# Size optimization
CFLAGS="-Os -ffunction-sections -fdata-sections"
LDFLAGS="-Wl,--gc-sections"

# Strip debug info for production
${CROSS_COMPILE}strip --strip-all myapp
```

---

## Summary

Cross compilation is essential for embedded Linux development:

**Key Steps:**
1. Install or build a cross-compilation toolchain
2. Set `CROSS_COMPILE` and `ARCH` environment variables
3. Use `--sysroot` or install target libraries on host
4. Build with cross compiler instead of native compiler
5. Test on target device or QEMU emulator

**Essential Variables:**
- `CROSS_COMPILE`: Toolchain prefix (e.g., `arm-linux-gnueabihf-`)
- `ARCH`: Target architecture (e.g., `arm`, `arm64`, `mips`)
- `SYSROOT`: Target root filesystem path on host

**Common Workflows:**
- Kernel: `make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-`
- Autotools: `./configure --host=arm-linux-gnueabihf`
- CMake: `cmake -DCMAKE_TOOLCHAIN_FILE=arm-toolchain.cmake`
- Makefile: `make CROSS_COMPILE=arm-linux-gnueabihf-`

**Resources:**
- [Crosstool-NG](https://crosstool-ng.github.io/)
- [Buildroot](https://buildroot.org/)
- [Yocto Project](https://www.yoctoproject.org/)
- [Embedded Linux Wiki](https://elinux.org/)
