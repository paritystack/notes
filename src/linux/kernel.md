# Linux Kernel Architecture

A comprehensive guide to Linux kernel internals, architecture, system calls, modules, compilation, and debugging.

## Table of Contents

1. [Kernel Overview](#kernel-overview)
2. [Kernel Architecture](#kernel-architecture)
3. [Memory Management](#memory-management)
4. [Process Management](#process-management)
5. [System Calls](#system-calls)
6. [Kernel Modules](#kernel-modules)
7. [Device Drivers](#device-drivers)
8. [File Systems](#file-systems)
9. [Networking Stack](#networking-stack)
10. [Kernel Compilation](#kernel-compilation)
11. [Kernel Debugging](#kernel-debugging)
12. [Performance Tuning](#performance-tuning)

---

## Kernel Overview

The Linux kernel is a monolithic kernel that handles all system operations including process management, memory management, device drivers, and system calls.

### Kernel Architecture Types

**Monolithic Kernel (Linux)**
- All services run in kernel space
- Better performance (no context switching)
- Single address space
- Larger kernel size

**Microkernel**
- Minimal kernel (IPC, memory, scheduling)
- Services run in user space
- Better stability and security
- More context switches

**Hybrid Kernel**
- Combination of both approaches
- Examples: Windows NT, macOS

### Linux Kernel Features

```
- Preemptive multitasking
- Symmetric multiprocessing (SMP)
- Virtual memory management
- Loadable kernel modules
- Multiple filesystem support
- POSIX compliance
- Dynamic kernel memory allocation
- Networking stack (TCP/IP, IPv6)
- Advanced security features (SELinux, AppArmor)
- Real-time capabilities (PREEMPT_RT)
```

### Kernel Version Numbering

```bash
# Check kernel version
uname -r
# Output: 6.5.0-15-generic

# Format: MAJOR.MINOR.PATCH-BUILD-ARCH
# 6.5.0 - kernel version
# 15 - distribution build number
# generic - kernel flavor/variant
```

**Version Types:**
- **Mainline** - Latest features, active development
- **Stable** - Production-ready, bug fixes only
- **LTS (Long Term Support)** - Extended maintenance (2-6 years)
- **EOL (End of Life)** - No longer maintained

### Kernel Source Tree Structure

```
/usr/src/linux/
├── arch/           # Architecture-specific code (x86, ARM, etc.)
├── block/          # Block device drivers
├── crypto/         # Cryptographic API
├── Documentation/  # Kernel documentation
├── drivers/        # Device drivers
│   ├── char/       # Character devices
│   ├── block/      # Block devices
│   ├── net/        # Network devices
│   ├── gpu/        # Graphics drivers
│   └── usb/        # USB drivers
├── fs/             # File system implementations
│   ├── ext4/       # ext4 filesystem
│   ├── btrfs/      # Btrfs filesystem
│   └── nfs/        # Network file system
├── include/        # Header files
│   ├── linux/      # Linux-specific headers
│   └── uapi/       # User-space API headers
├── init/           # Kernel initialization
├── ipc/            # Inter-process communication
├── kernel/         # Core kernel code
│   ├── sched/      # Process scheduler
│   ├── time/       # Time management
│   └── irq/        # Interrupt handling
├── lib/            # Library routines
├── mm/             # Memory management
├── net/            # Networking stack
│   ├── ipv4/       # IPv4 implementation
│   ├── ipv6/       # IPv6 implementation
│   └── core/       # Core networking
├── samples/        # Sample code
├── scripts/        # Build scripts
├── security/       # Security modules (SELinux, AppArmor)
├── sound/          # Sound drivers
└── tools/          # Kernel tools and utilities
```

---

## Kernel Architecture

### Kernel Space vs User Space

```
+------------------------------------------+
|           User Space (Ring 3)            |
| +--------------------------------------+ |
| | User Applications                    | |
| | (web browsers, editors, games, etc.) | |
| +--------------------------------------+ |
|                   ↕                      |
| +--------------------------------------+ |
| | System Libraries (glibc, etc.)       | |
| +--------------------------------------+ |
+------------------------------------------+
                   ↕
         System Call Interface
                   ↕
+------------------------------------------+
|         Kernel Space (Ring 0)            |
| +--------------------------------------+ |
| | System Call Interface                | |
| +--------------------------------------+ |
| | Process    | Memory   | File System | |
| | Management | Manager  | Layer       | |
| +--------------------------------------+ |
| | Network    | IPC      | Security    | |
| | Stack      | Layer    | Modules     | |
| +--------------------------------------+ |
| | Device Drivers                       | |
| | (char, block, network)               | |
| +--------------------------------------+ |
| | Architecture-Specific Code           | |
| | (CPU, MMU, interrupts)               | |
| +--------------------------------------+ |
+------------------------------------------+
                   ↕
            Hardware Layer
```

### Key Kernel Components

#### 1. Process Scheduler

Manages CPU time allocation among processes.

```c
// Scheduling classes (from highest to lowest priority)
1. SCHED_DEADLINE  // Deadline scheduling (real-time)
2. SCHED_FIFO      // First-in-first-out (real-time)
3. SCHED_RR        // Round-robin (real-time)
4. SCHED_NORMAL    // Standard time-sharing (CFS)
5. SCHED_BATCH     // Batch processes
6. SCHED_IDLE      // Very low priority

// Completely Fair Scheduler (CFS) - default for SCHED_NORMAL
// - Uses red-black tree for O(log n) operations
// - Virtual runtime tracking
// - Fair CPU time distribution
```

**Check and modify scheduling:**
```bash
# View process scheduling info
ps -eo pid,pri,ni,comm,policy

# Change scheduling policy
chrt -f -p 99 PID        # Set to FIFO with priority 99
chrt -r -p 50 PID        # Set to Round-robin
chrt -o -p 0 PID         # Set to normal

# Change nice value (-20 to 19)
nice -n 10 command       # Run with nice value 10
renice -n 5 -p PID       # Change nice value of running process
```

#### 2. Memory Manager

Handles virtual memory, paging, and memory allocation.

```
Virtual Memory Layout (64-bit x86):

0x00007FFFFFFFFFFF  +------------------+
                    | User Stack       | (grows down)
                    +------------------+
                    | Memory Mapped    |
                    | Files & Libs     |
                    +------------------+
                    | Heap             | (grows up)
                    +------------------+
                    | BSS (uninit data)|
                    +------------------+
                    | Data (init data) |
                    +------------------+
0x0000000000400000  | Text (code)      |
                    +------------------+
                    | Reserved         |
0x0000000000000000  +------------------+

Kernel Space Layout:

0xFFFFFFFFFFFFFFFF  +------------------+
                    | Kernel Code/Data |
                    +------------------+
                    | Direct Mapping   |
                    | (Physical RAM)   |
                    +------------------+
                    | vmalloc Area     |
                    +------------------+
                    | Module Space     |
0xFFFF800000000000  +------------------+
```

**Memory zones:**
```
ZONE_DMA       - Memory for DMA (0-16MB on x86)
ZONE_DMA32     - Memory for 32-bit DMA (0-4GB)
ZONE_NORMAL    - Normal memory (above 4GB on 64-bit)
ZONE_HIGHMEM   - High memory (not directly mapped, 32-bit only)
ZONE_MOVABLE   - Memory that can be migrated
```

#### 3. Virtual File System (VFS)

Abstract layer for file system operations.

```c
// VFS Objects
struct super_block  // Mounted filesystem
struct inode        // File metadata
struct dentry       // Directory entry (name to inode mapping)
struct file         // Open file instance

// File operations structure
struct file_operations {
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    int (*open) (struct inode *, struct file *);
    int (*release) (struct inode *, struct file *);
    // ... more operations
};
```

#### 4. Network Stack

Implements network protocols and socket interface.

```
Layer Model:

Application Layer
       ↕
Socket Interface
       ↕
Transport Layer (TCP/UDP)
       ↕
Network Layer (IP)
       ↕
Link Layer (Ethernet, WiFi)
       ↕
Device Driver
       ↕
Hardware
```

---

## Memory Management

### Page Management

Linux uses paging for memory management:

```bash
# Check page size
getconf PAGE_SIZE
# Usually 4096 bytes (4KB)

# View memory info
cat /proc/meminfo
# MemTotal, MemFree, MemAvailable, Buffers, Cached, etc.

# Memory statistics
vmstat 1
# View paging, memory, CPU stats every second

# Detailed memory usage
cat /proc/PID/status | grep -i vm
cat /proc/PID/maps    # Memory mappings
```

### Memory Allocation

**Kernel Memory Allocation:**

```c
// Physically contiguous memory
kmalloc(size, GFP_KERNEL)     // Standard allocation
kfree(ptr)                     // Free memory

// Virtual contiguous memory
vmalloc(size)                  // Large allocations
vfree(ptr)

// Page-based allocation
alloc_pages(gfp_mask, order)   // 2^order pages
free_pages(addr, order)

// Flags (GFP = Get Free Pages)
GFP_KERNEL    // Standard, may sleep
GFP_ATOMIC    // Cannot sleep, for interrupts
GFP_USER      // User space allocation
GFP_DMA       // DMA-capable memory
```

### Memory Reclamation

**OOM Killer (Out-of-Memory):**

```bash
# View OOM score (higher = more likely to be killed)
cat /proc/PID/oom_score

# Adjust OOM score (-1000 to 1000)
echo -500 > /proc/PID/oom_score_adj  # Less likely to be killed
echo 500 > /proc/PID/oom_score_adj   # More likely to be killed

# Disable OOM killer for process
echo -1000 > /proc/PID/oom_score_adj

# View OOM killer logs
dmesg | grep -i "out of memory"
journalctl -k | grep -i "oom"
```

**Swapping:**

```bash
# View swap usage
swapon --show
free -h

# Create swap file
dd if=/dev/zero of=/swapfile bs=1M count=1024
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Control swappiness (0-100, default 60)
cat /proc/sys/vm/swappiness
echo 10 > /proc/sys/vm/swappiness  # Less aggressive swapping

# Make permanent in /etc/sysctl.conf
vm.swappiness=10
```

### Huge Pages

Improve performance for applications with large memory footprints:

```bash
# View huge page info
cat /proc/meminfo | grep -i huge

# Configure huge pages
echo 512 > /proc/sys/vm/nr_hugepages

# Transparent Huge Pages (THP)
cat /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled  # Recommended
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

---

## Process Management

### Process Representation

```c
// Task structure (Process Control Block)
struct task_struct {
    pid_t pid;                     // Process ID
    pid_t tgid;                    // Thread group ID
    struct task_struct *parent;    // Parent process
    struct list_head children;     // Child processes
    struct mm_struct *mm;          // Memory descriptor
    struct fs_struct *fs;          // Filesystem info
    struct files_struct *files;    // Open files
    int exit_state;                // Exit status
    unsigned int policy;           // Scheduling policy
    // ... many more fields
};
```

### Process States

```c
TASK_RUNNING           // Running or ready to run
TASK_INTERRUPTIBLE     // Sleeping, can be woken by signals
TASK_UNINTERRUPTIBLE   // Sleeping, cannot be interrupted
TASK_STOPPED           // Stopped (e.g., by SIGSTOP)
TASK_TRACED            // Being traced by debugger
EXIT_ZOMBIE            // Terminated, waiting for parent
EXIT_DEAD              // Final state before removal
```

**View process states:**
```bash
ps aux
# STAT column:
# R - Running
# S - Sleeping (interruptible)
# D - Sleeping (uninterruptible, usually I/O)
# T - Stopped
# Z - Zombie
# < - High priority
# N - Low priority
# + - Foreground process group

# Find stuck processes (uninterruptible sleep)
ps aux | awk '$8 ~ /D/'
```

### Process Creation

**fork() system call:**
```c
#include <unistd.h>
#include <stdio.h>

int main() {
    pid_t pid = fork();
    
    if (pid < 0) {
        // Fork failed
        perror("fork");
        return 1;
    } else if (pid == 0) {
        // Child process
        printf("Child: PID = %d\n", getpid());
    } else {
        // Parent process
        printf("Parent: PID = %d, Child PID = %d\n", getpid(), pid);
    }
    
    return 0;
}
```

**exec() system call:**
```c
#include <unistd.h>

int main() {
    char *args[] = {"/bin/ls", "-l", NULL};
    execv("/bin/ls", args);  // Replace current process
    
    // Only reached if exec fails
    perror("exec");
    return 1;
}
```

### Process Namespaces

Provide isolation for different resources:

```bash
# Namespace types
PID      # Process IDs
NET      # Network stack
MNT      # Mount points
IPC      # Inter-process communication
UTS      # Hostname and domain name
USER     # User and group IDs
CGROUP   # Control groups

# View process namespaces
ls -l /proc/self/ns/
lsns  # List namespaces

# Create new namespace
unshare --pid --fork bash  # New PID namespace
unshare --net bash         # New network namespace

# Enter namespace
nsenter --target PID --pid --uts --net bash
```

---

## System Calls

System calls provide the interface between user space and kernel space.

### System Call Mechanism

```
User Space:
  Application calls glibc function
         ↓
  glibc wrapper function
         ↓
  Software interrupt (int 0x80 or syscall instruction)
         ↓
Kernel Space:
  System call handler
         ↓
  Kernel function implementation
         ↓
  Return to user space
```

### Common System Calls

**Process Management:**
```c
fork()          // Create child process
exec()          // Execute program
exit()          // Terminate process
wait()          // Wait for child process
getpid()        // Get process ID
getppid()       // Get parent process ID
kill()          // Send signal to process
nice()          // Change priority
```

**File Operations:**
```c
open()          // Open file
close()         // Close file
read()          // Read from file
write()         // Write to file
lseek()         // Change file position
stat()          // Get file status
chmod()         // Change permissions
chown()         // Change ownership
link()          // Create hard link
unlink()        // Delete file
mkdir()         // Create directory
rmdir()         // Remove directory
```

**Memory Management:**
```c
brk()           // Change data segment size
mmap()          // Map file or device into memory
munmap()        // Unmap memory
mprotect()      // Change memory protection
mlock()         // Lock memory (prevent swapping)
```

**Networking:**
```c
socket()        // Create socket
bind()          // Bind socket to address
listen()        // Listen for connections
accept()        // Accept connection
connect()       // Connect to remote socket
send()          // Send data
recv()          // Receive data
shutdown()      // Shut down socket
```

### Tracing System Calls

**strace - Trace system calls:**

```bash
# Trace program execution
strace ls
strace -o output.txt ls    # Save to file

# Trace specific system calls
strace -e open,read ls     # Only open and read
strace -e trace=file ls    # All file operations
strace -e trace=network curl example.com

# Attach to running process
strace -p PID

# Count system call statistics
strace -c ls

# Follow child processes
strace -f ./program

# Timestamp system calls
strace -t ls               # Time of day
strace -T ls               # Time spent in each call

# Examples
strace -e trace=open,openat cat /etc/passwd
strace -c find / -name "*.log" 2>/dev/null
strace -p $(pgrep nginx | head -1)
```

### Writing a Simple System Call

**1. Add system call to kernel:**

```c
// kernel/sys.c
SYSCALL_DEFINE1(hello, char __user *, msg)
{
    char kernel_msg[256];
    
    if (copy_from_user(kernel_msg, msg, sizeof(kernel_msg)))
        return -EFAULT;
    
    printk(KERN_INFO "System call hello: %s\n", kernel_msg);
    return 0;
}
```

**2. Add to system call table:**

```c
// arch/x86/entry/syscalls/syscall_64.tbl
450    common    hello    sys_hello
```

**3. User space program:**

```c
#include <unistd.h>
#include <sys/syscall.h>

#define __NR_hello 450

int main() {
    syscall(__NR_hello, "Hello from user space!");
    return 0;
}
```

---

## Kernel Modules

Kernel modules allow dynamic loading of code into the running kernel.

### Module Basics

```bash
# List loaded modules
lsmod

# Module information
modinfo module_name

# Load module
modprobe module_name
insmod /path/to/module.ko

# Unload module
modprobe -r module_name
rmmod module_name

# Module dependencies
depmod -a

# Module parameters
modinfo -p module_name
modprobe module_name param=value
```

### Writing a Simple Module

**hello_module.c:**
```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A simple Hello World module");
MODULE_VERSION("1.0");

// Module initialization
static int __init hello_init(void)
{
    printk(KERN_INFO "Hello World module loaded\n");
    return 0;  // 0 = success
}

// Module cleanup
static void __exit hello_exit(void)
{
    printk(KERN_INFO "Hello World module unloaded\n");
}

// Register init and exit functions
module_init(hello_init);
module_exit(hello_exit);
```

**Makefile:**
```makefile
obj-m += hello_module.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

install:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules_install
	depmod -a
```

**Build and load:**
```bash
# Compile
make

# Load module
sudo insmod hello_module.ko

# Check kernel log
dmesg | tail

# Unload module
sudo rmmod hello_module

# Install system-wide
sudo make install
```

### Module with Parameters

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/moduleparam.h>

MODULE_LICENSE("GPL");

static int count = 1;
static char *name = "World";

module_param(count, int, S_IRUGO);
module_param(name, charp, S_IRUGO);

MODULE_PARM_DESC(count, "Number of times to greet");
MODULE_PARM_DESC(name, "Name to greet");

static int __init param_init(void)
{
    int i;
    for (i = 0; i < count; i++) {
        printk(KERN_INFO "Hello %s! (%d/%d)\n", name, i+1, count);
    }
    return 0;
}

static void __exit param_exit(void)
{
    printk(KERN_INFO "Goodbye %s!\n", name);
}

module_init(param_init);
module_exit(param_exit);
```

**Load with parameters:**
```bash
sudo insmod param_module.ko count=3 name="Linux"
```

### Character Device Driver

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "chardev"
#define BUFFER_SIZE 1024

MODULE_LICENSE("GPL");

static int major_number;
static char device_buffer[BUFFER_SIZE];
static int buffer_size = 0;

// File operations
static int dev_open(struct inode *inodep, struct file *filep)
{
    printk(KERN_INFO "chardev: Device opened\n");
    return 0;
}

static ssize_t dev_read(struct file *filep, char *buffer, 
                        size_t len, loff_t *offset)
{
    int bytes_read = 0;
    
    if (*offset >= buffer_size)
        return 0;
    
    bytes_read = buffer_size - *offset;
    if (bytes_read > len)
        bytes_read = len;
    
    if (copy_to_user(buffer, device_buffer + *offset, bytes_read))
        return -EFAULT;
    
    *offset += bytes_read;
    return bytes_read;
}

static ssize_t dev_write(struct file *filep, const char *buffer, 
                         size_t len, loff_t *offset)
{
    int bytes_written = len;
    
    if (bytes_written > BUFFER_SIZE)
        bytes_written = BUFFER_SIZE;
    
    if (copy_from_user(device_buffer, buffer, bytes_written))
        return -EFAULT;
    
    buffer_size = bytes_written;
    printk(KERN_INFO "chardev: Received %d bytes\n", bytes_written);
    return bytes_written;
}

static int dev_release(struct inode *inodep, struct file *filep)
{
    printk(KERN_INFO "chardev: Device closed\n");
    return 0;
}

static struct file_operations fops = {
    .open = dev_open,
    .read = dev_read,
    .write = dev_write,
    .release = dev_release,
};

static int __init chardev_init(void)
{
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    
    if (major_number < 0) {
        printk(KERN_ALERT "chardev: Failed to register\n");
        return major_number;
    }
    
    printk(KERN_INFO "chardev: Registered with major number %d\n", 
           major_number);
    printk(KERN_INFO "chardev: Create device with: mknod /dev/%s c %d 0\n",
           DEVICE_NAME, major_number);
    return 0;
}

static void __exit chardev_exit(void)
{
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "chardev: Unregistered\n");
}

module_init(chardev_init);
module_exit(chardev_exit);
```

**Using the device:**
```bash
# Load module
sudo insmod chardev.ko

# Create device node
sudo mknod /dev/chardev c <major_number> 0
sudo chmod 666 /dev/chardev

# Test device
echo "Hello" > /dev/chardev
cat /dev/chardev

# Cleanup
sudo rm /dev/chardev
sudo rmmod chardev
```

---

## Device Drivers

### Driver Types

**Character Devices:**
- Sequential access
- Examples: keyboards, serial ports, /dev/null
- Major/minor numbers for identification

**Block Devices:**
- Random access, buffered I/O
- Examples: hard drives, SSDs, USB drives
- Use page cache for performance

**Network Devices:**
- Packet transmission/reception
- Examples: Ethernet, WiFi, loopback
- Socket interface

### Device Model

```bash
# View device hierarchy
ls /sys/devices/
ls /sys/class/

# PCI devices
lspci -v
ls /sys/bus/pci/devices/

# USB devices
lsusb -v
ls /sys/bus/usb/devices/

# Block devices
lsblk
ls /sys/block/

# Network devices
ip link show
ls /sys/class/net/

# Device information
udevadm info --query=all --name=/dev/sda
```

### Device Management with udev

**udev rules** (/etc/udev/rules.d/):

```bash
# Example: Custom USB device rule
# /etc/udev/rules.d/99-usb-device.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="1234", ATTR{idProduct}=="5678", \
    MODE="0666", GROUP="users", SYMLINK+="mydevice"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Monitor udev events
udevadm monitor
```

---

## File Systems

### VFS Layer

The Virtual File System provides a common interface for all filesystems.

**Supported filesystems:**
```bash
cat /proc/filesystems
# ext4, btrfs, xfs, nfs, vfat, tmpfs, etc.

# Filesystem modules
ls /lib/modules/$(uname -r)/kernel/fs/
```

### ext4 Filesystem

```bash
# Create ext4 filesystem
mkfs.ext4 /dev/sdb1

# Filesystem check
fsck.ext4 /dev/sdb1
e2fsck -f /dev/sdb1

# Filesystem information
dumpe2fs /dev/sdb1
tune2fs -l /dev/sdb1

# Tune filesystem
tune2fs -m 1 /dev/sdb1           # Reserved blocks percentage
tune2fs -c 30 /dev/sdb1          # Max mount count
tune2fs -i 6m /dev/sdb1          # Check interval

# Enable/disable features
tune2fs -O has_journal /dev/sdb1   # Enable journaling
tune2fs -O ^has_journal /dev/sdb1  # Disable journaling
```

### Filesystem Debugging

```bash
# Debugfs - interactive ext2/ext3/ext4 debugger
debugfs /dev/sdb1
# Commands: ls, cd, stat, logdump, etc.

# View inode information
stat /path/to/file
ls -i /path/to/file              # Show inode number
debugfs -R "stat <inode_number>" /dev/sdb1

# Find deleted files
debugfs -R "lsdel" /dev/sdb1
```

---

## Networking Stack

### Network Layer Architecture

```
+-----------------+
| Application     |
+-----------------+
| Socket Layer    |
+-----------------+
| Protocol Layer  | (TCP, UDP, ICMP)
+-----------------+
| IP Layer        | (IPv4, IPv6, routing)
+-----------------+
| Link Layer      | (Ethernet, WiFi)
+-----------------+
| Device Driver   |
+-----------------+
| Hardware        |
+-----------------+
```

### Network Configuration

```bash
# View network configuration
ip addr show
ip route show
ip link show

# Network statistics
cat /proc/net/dev               # Interface statistics
cat /proc/net/tcp               # TCP connections
cat /proc/net/udp               # UDP connections
netstat -s                      # Protocol statistics

# Socket buffers
sysctl net.core.rmem_max        # Receive buffer
sysctl net.core.wmem_max        # Send buffer

# TCP parameters
sysctl net.ipv4.tcp_rmem        # TCP receive memory
sysctl net.ipv4.tcp_wmem        # TCP send memory
sysctl net.ipv4.tcp_congestion_control
```

### Network Debugging

See [networking.md](./networking.md) for detailed network debugging.

---

## Kernel Compilation

### Getting Kernel Source

```bash
# Download from kernel.org
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.5.tar.xz
tar -xf linux-6.5.tar.xz
cd linux-6.5

# Or use git
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux
git checkout v6.5

# Distribution specific
# Ubuntu/Debian
apt-get source linux-image-$(uname -r)

# Fedora/RHEL
dnf download --source kernel
```

### Kernel Configuration

```bash
cd /usr/src/linux

# Configuration methods
make config           # Text-based Q&A (tedious)
make menuconfig       # Text-based menu (ncurses)
make xconfig          # Qt-based GUI
make gconfig          # GTK-based GUI

# Use existing config
make oldconfig        # Update old config
make localmodconfig   # Only modules for current hardware
make defconfig        # Default configuration
cp /boot/config-$(uname -r) .config  # Copy running config

# Configuration file
.config               # Generated configuration
```

**Important config options:**
```bash
# General setup
CONFIG_LOCALVERSION="-custom"        # Custom kernel name
CONFIG_DEFAULT_HOSTNAME="myhost"

# Processor type
CONFIG_SMP=y                         # Symmetric multiprocessing
CONFIG_NR_CPUS=8                     # Number of CPUs

# Power management
CONFIG_CPU_FREQ=y                    # CPU frequency scaling
CONFIG_HIBERNATION=y

# Networking
CONFIG_NETFILTER=y                   # Firewall support
CONFIG_BRIDGE=y                      # Network bridging

# Filesystems
CONFIG_EXT4_FS=y                     # ext4 filesystem
CONFIG_BTRFS_FS=y                    # Btrfs filesystem

# Security
CONFIG_SECURITY_SELINUX=y            # SELinux support
CONFIG_SECURITY_APPARMOR=y           # AppArmor support

# Debugging
CONFIG_DEBUG_KERNEL=y                # Kernel debugging
CONFIG_KGDB=y                        # Kernel debugger
CONFIG_DEBUG_INFO=y                  # Debug symbols
```

### Building the Kernel

```bash
# Install build dependencies
# Ubuntu/Debian
sudo apt install build-essential libncurses-dev bison flex \
                 libssl-dev libelf-dev bc

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install ncurses-devel bison flex elfutils-libelf-devel \
                 openssl-devel bc

# Build kernel
make -j$(nproc)                      # Use all CPU cores

# Or build specific targets
make bzImage                         # Kernel image
make modules                         # Kernel modules
make dtbs                            # Device tree blobs (ARM)

# Install
sudo make modules_install            # Install modules to /lib/modules
sudo make install                    # Install kernel to /boot

# Manual installation
sudo cp arch/x86/boot/bzImage /boot/vmlinuz-6.5-custom
sudo cp System.map /boot/System.map-6.5-custom
sudo cp .config /boot/config-6.5-custom

# Update bootloader
sudo update-grub                     # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # Fedora/RHEL

# Reboot
sudo reboot
```

### Cross-Compilation

```bash
# Install cross-compiler
sudo apt install gcc-arm-linux-gnueabi

# Configure for target architecture
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi- defconfig

# Build
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabi- -j$(nproc)

# Example architectures
ARCH=arm                             # ARM 32-bit
ARCH=arm64                           # ARM 64-bit (aarch64)
ARCH=mips                            # MIPS
ARCH=powerpc                         # PowerPC
ARCH=riscv                           # RISC-V
```

### Kernel Patching

```bash
# Apply patch
patch -p1 < patch-file.patch

# Create patch
diff -Naur original/ modified/ > my-patch.patch

# Check if patch applies cleanly
patch -p1 --dry-run < patch-file.patch

# Reverse patch
patch -R -p1 < patch-file.patch
```

---

## Kernel Debugging

### printk - Kernel Logging

```c
#include <linux/printk.h>

// Log levels (from highest to lowest priority)
printk(KERN_EMERG "System is unusable\n");           // 0
printk(KERN_ALERT "Action must be taken\n");         // 1
printk(KERN_CRIT "Critical conditions\n");           // 2
printk(KERN_ERR "Error conditions\n");               // 3
printk(KERN_WARNING "Warning conditions\n");         // 4
printk(KERN_NOTICE "Normal but significant\n");      // 5
printk(KERN_INFO "Informational\n");                 // 6
printk(KERN_DEBUG "Debug-level messages\n");         // 7

// Default level (usually KERN_WARNING)
printk("Default level message\n");

// Dynamic debug (if CONFIG_DYNAMIC_DEBUG enabled)
pr_debug("Debug message\n");
```

**View kernel messages:**
```bash
dmesg                                # View kernel ring buffer
dmesg -w                             # Follow new messages
dmesg -l err                         # Only errors
dmesg --level=err,warn               # Errors and warnings
dmesg -T                             # Human-readable timestamps

journalctl -k                        # Kernel messages via systemd
journalctl -k -f                     # Follow kernel messages
journalctl -k --since "1 hour ago"

# Set console log level
dmesg -n 1                           # Only emergency messages to console
echo 7 > /proc/sys/kernel/printk     # All messages to console
```

### KGDB - Kernel Debugger

```bash
# Build kernel with debugging enabled
CONFIG_DEBUG_KERNEL=y
CONFIG_DEBUG_INFO=y
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y

# Boot with KGDB enabled
linux ... kgdboc=ttyS0,115200 kgdbwait

# Connect with GDB
gdb vmlinux
(gdb) target remote /dev/ttyS0
(gdb) break sys_open
(gdb) continue
```

### kdump - Kernel Crash Dumps

```bash
# Install kdump
# Ubuntu/Debian
sudo apt install kdump-tools

# Fedora/RHEL
sudo dnf install kexec-tools

# Configure kdump
# Edit /etc/default/kdump-tools (Debian) or /etc/sysconfig/kdump (RHEL)

# Reserve memory for crash kernel
# Add to kernel parameters: crashkernel=384M-:128M

# Enable kdump
sudo systemctl enable kdump
sudo systemctl start kdump

# Test crash
echo c > /proc/sysrq-trigger         # WARNING: Crashes system!

# Analyze crash dump
crash /usr/lib/debug/vmlinux-<version> /var/crash/vmcore
```

### Magic SysRq Key

Emergency kernel functions:

```bash
# Enable SysRq
echo 1 > /proc/sys/kernel/sysrq

# SysRq commands (Alt+SysRq+<key>)
# Or: echo <key> > /proc/sysrq-trigger

b - Reboot immediately
c - Crash (for kdump)
e - SIGTERM to all processes
f - OOM killer
h - Help
i - SIGKILL to all processes
k - Kill all on current console
m - Memory info
p - Current registers and flags
r - Keyboard raw mode
s - Sync all filesystems
t - Task list
u - Remount filesystems read-only
w - Tasks in uninterruptible sleep

# Safe reboot sequence (REISUB)
# R - Raw keyboard mode
# E - SIGTERM all
# I - SIGKILL all
# S - Sync disks
# U - Remount read-only
# B - Reboot
```

### ftrace - Function Tracer

```bash
# Mount debugfs
mount -t debugfs none /sys/kernel/debug

cd /sys/kernel/debug/tracing

# Available tracers
cat available_tracers
# function, function_graph, blk, wakeup, etc.

# Enable function tracer
echo function > current_tracer
echo 1 > tracing_on

# View trace
cat trace | head -20

# Stop tracing
echo 0 > tracing_on

# Trace specific function
echo sys_open > set_ftrace_filter
echo function > current_tracer
echo 1 > tracing_on

# Clear trace
echo > trace

# Example: Trace network stack
echo 1 > events/net/enable
echo 1 > tracing_on
# Generate network traffic
cat trace
```

### SystemTap

Dynamic tracing and instrumentation:

```bash
# Install SystemTap
sudo apt install systemtap systemtap-runtime

# Install kernel debug symbols
sudo apt install linux-image-$(uname -r)-dbgsym

# Simple script (hello.stp)
probe begin {
    printf("Hello, SystemTap!\n")
    exit()
}

# Run script
sudo stap hello.stp

# Trace system calls
sudo stap -e 'probe syscall.open { println(execname()) }'

# Count system calls
sudo stap -e '
    global count
    probe syscall.* {
        count[name]++
    }
    probe end {
        foreach (syscall in count-)
            printf("%20s: %d\n", syscall, count[syscall])
    }
' -c "ls -l /"
```

### perf - Performance Analysis

```bash
# Install perf
sudo apt install linux-tools-$(uname -r)

# Record CPU cycles
sudo perf record -a sleep 10

# View report
sudo perf report

# CPU profiling
sudo perf top

# Stat command
sudo perf stat ls -R /

# Trace system calls
sudo perf trace ls

# Record specific events
sudo perf record -e sched:sched_switch -a sleep 5
sudo perf script

# Hardware counters
perf list                          # List available events
sudo perf stat -e cache-misses,cache-references ls
```

---

## Performance Tuning

### sysctl Parameters

```bash
# View all parameters
sysctl -a

# View specific parameter
sysctl vm.swappiness

# Set temporarily
sudo sysctl vm.swappiness=10

# Set permanently (/etc/sysctl.conf or /etc/sysctl.d/)
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p                     # Reload configuration
```

**Important parameters:**

```bash
# Virtual Memory
vm.swappiness=10                   # Reduce swap usage
vm.dirty_ratio=10                  # Dirty page threshold for writeback
vm.dirty_background_ratio=5        # Background writeback threshold
vm.overcommit_memory=1             # Allow memory overcommit

# Network
net.core.rmem_max=134217728        # Max receive buffer
net.core.wmem_max=134217728        # Max send buffer
net.core.netdev_max_backlog=5000   # Input queue size
net.ipv4.tcp_rmem=4096 87380 67108864     # TCP read memory
net.ipv4.tcp_wmem=4096 65536 67108864     # TCP write memory
net.ipv4.tcp_congestion_control=bbr        # Congestion algorithm
net.ipv4.tcp_fastopen=3            # TCP Fast Open
net.ipv4.tcp_mtu_probing=1         # Path MTU discovery
net.ipv4.ip_forward=1              # IP forwarding

# File System
fs.file-max=2097152                # Max open files system-wide
fs.inotify.max_user_watches=524288 # Inotify watches

# Kernel
kernel.sysrq=1                     # Enable SysRq
kernel.panic=10                    # Reboot 10s after panic
kernel.pid_max=4194304             # Max PIDs
```

### I/O Schedulers

```bash
# View available schedulers
cat /sys/block/sda/queue/scheduler
# [mq-deadline] kyber bfq none

# Change scheduler
echo kyber > /sys/block/sda/queue/scheduler

# Schedulers:
# mq-deadline - Default, good for most workloads
# kyber - Low latency, good for SSDs
# bfq - Fair queueing, good for desktops
# none - No scheduling (for NVMe with low latency)

# Make permanent (udev rule)
# /etc/udev/rules.d/60-scheduler.rules
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="kyber"
```

### CPU Governor

```bash
# View current governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# performance powersave schedutil ondemand conservative

# Set governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Governors:
# performance - Max frequency
# powersave - Min frequency
# ondemand - Dynamic scaling (legacy)
# schedutil - Scheduler-driven (default, recommended)
# conservative - Gradual scaling

# Using cpupower
sudo cpupower frequency-set -g performance
sudo cpupower frequency-info
```

### Huge Pages

```bash
# Configure huge pages
echo 512 > /proc/sys/vm/nr_hugepages

# Transparent Huge Pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled  # Recommended

# View huge page usage
cat /proc/meminfo | grep -i huge

# Permanent configuration (/etc/sysctl.conf)
vm.nr_hugepages=512
```

### NUMA (Non-Uniform Memory Access)

```bash
# Check NUMA configuration
numactl --hardware

# View NUMA statistics
numastat

# Run program on specific NUMA node
numactl --cpunodebind=0 --membind=0 ./program

# Automatic NUMA balancing
echo 1 > /proc/sys/kernel/numa_balancing
```

---

## Practical Examples

### Monitoring System Performance

```bash
#!/bin/bash
# System performance monitoring script

echo "=== CPU Usage ==="
mpstat 1 5 | tail -1

echo -e "\n=== Memory Usage ==="
free -h

echo -e "\n=== Disk I/O ==="
iostat -xz 1 2 | tail -n +3

echo -e "\n=== Network ==="
sar -n DEV 1 1 | tail -3

echo -e "\n=== Top Processes by CPU ==="
ps aux --sort=-%cpu | head -6

echo -e "\n=== Top Processes by Memory ==="
ps aux --sort=-%mem | head -6

echo -e "\n=== Load Average ==="
uptime

echo -e "\n=== Kernel Parameters ==="
sysctl vm.swappiness net.ipv4.tcp_congestion_control
```

### Kernel Module Template

```c
/**
 * template_module.c - Template for kernel modules
 */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Template module");
MODULE_VERSION("1.0");

static int __init template_init(void)
{
    printk(KERN_INFO "template: Module loaded\n");
    
    // Initialize your code here
    
    return 0;
}

static void __exit template_exit(void)
{
    // Cleanup your code here
    
    printk(KERN_INFO "template: Module unloaded\n");
}

module_init(template_init);
module_exit(template_exit);
```

---

## Resources

### Official Documentation
- [Kernel Documentation](https://www.kernel.org/doc/html/latest/)
- [Linux Kernel Mailing List](https://lkml.org/)
- [Kernel Newbies](https://kernelnewbies.org/)

### Books
- "Linux Kernel Development" by Robert Love
- "Linux Device Drivers" by Jonathan Corbet
- "Understanding the Linux Kernel" by Daniel P. Bovet
- "Linux System Programming" by Robert Love

### Online Resources
- [The Linux Kernel Archives](https://www.kernel.org/)
- [LWN.net](https://lwn.net/) - Linux Weekly News
- [Bootlin Training Materials](https://bootlin.com/docs/)

### Development Tools
- Git - Version control
- cscope/ctags - Code navigation
- sparse - Static analyzer
- Coccinelle - Semantic patching
- QEMU - Virtualization for testing

---

This guide covers the fundamentals of Linux kernel architecture and development. The kernel is vast and constantly evolving, so continuous learning and experimentation are essential!
