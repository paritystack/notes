# Linux Kernel Debugging

Debugging the Linux kernel requires specialized tools and techniques due to its low-level nature.

## Kernel Log (dmesg)

```bash
# View kernel messages
dmesg

# Follow kernel log
dmesg -w
dmesg --follow

# Filter by level
dmesg -l err,warn

# Human-readable timestamps
dmesg -T

# Clear ring buffer
sudo dmesg -C
```

## Kernel Parameters

```bash
# View boot parameters
cat /proc/cmdline

# Add debug parameters (GRUB)
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX="... debug ignore_loglevel"

# Update GRUB
sudo update-grub
```

## printk Debugging

```c
// In kernel code
#include <linux/printk.h>

printk(KERN_INFO "Debug: value = %d\n", value);
printk(KERN_ERR "Error occurred\n");

// Log levels
KERN_EMERG, KERN_ALERT, KERN_CRIT, KERN_ERR,
KERN_WARNING, KERN_NOTICE, KERN_INFO, KERN_DEBUG
```

## KGDB (Kernel Debugger)

```bash
# Kernel configuration
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y

# Boot with kgdb
kgdboc=ttyS0,115200 kgdbwait

# Connect with GDB
gdb ./vmlinux
(gdb) target remote /dev/ttyS0
(gdb) continue
```

## Kernel Oops Analysis

```bash
# When kernel oops occurs, check dmesg
dmesg | tail -100

# Decode with scripts
./scripts/decode_stacktrace.sh vmlinux < oops.txt

# addr2line for addresses
addr2line -e vmlinux -f -i 0xffffffffc0123456
```

## SystemTap

```bash
# Install
sudo apt install systemtap

# Simple script (sys_open no longer exists on modern kernels; use a current symbol)
stap -e 'probe kernel.function("vfs_read") { println("read called") }'

# Trace system calls
stap -e 'probe syscall.* { printf("%s\n", name) }'
```

## ftrace

The built-in kernel tracer, driven through tracefs. Quick taste below; see
[ftrace & tracefs](ftrace.md) for the full treatment (function_graph, tracepoints, and the
`irqsoff`/`preemptoff` latency tracers).

```bash
# Enable function tracing
cd /sys/kernel/tracing      # or /sys/kernel/debug/tracing on older kernels
echo function > current_tracer
echo 1 > tracing_on
cat trace

# Trace specific function (sys_open no longer exists on modern kernels — use a current symbol)
echo vfs_read > set_ftrace_filter
echo function > current_tracer

# Disable
echo 0 > tracing_on
```

## Kernel Crash Dumps (kdump)

```bash
# Install kdump
sudo apt install kdump-tools

# Configure /etc/default/kdump-tools
USE_KDUMP=1

# Test
echo c | sudo tee /proc/sysrq-trigger

# Analyze with crash
crash /usr/lib/debug/boot/vmlinux-$(uname -r) /var/crash/*/dump.* 
```

Kernel debugging requires patience and specialized knowledge, but these tools make it manageable.

## Where this connects

- [GDB](gdb.md) — KGDB enables GDB-style debugging of the running kernel
- [Core dump](core_dump.md) — vmcore (kernel crash dump) is the kernel equivalent of a process core dump
- [ftrace & tracefs](ftrace.md) — the built-in tracer in depth: function_graph and latency tracers
- [Tools](tools.md) — tools like perf and ftrace are essential kernel debugging tools
- [../linux/kernel_patterns](../linux/kernel_patterns.md) — kernel code patterns help interpret debug output
