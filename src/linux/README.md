# Linux Documentation

A comprehensive guide to Linux system administration, commands, kernel architecture, and networking.

## Table of Contents

1. [Essential Commands](./commands.md) - Command reference and examples
2. [Kernel Architecture](./kernel.md) - Linux kernel internals and development
3. [Memory Management](./memory_management.md) - Virtual memory, paging, allocators, reclaim, and OOM
4. [Kernel Development Patterns](./kernel_patterns.md) - Common patterns and best practices for kernel development
5. [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless subsystem frameworks for WiFi drivers
6. [Driver Development](./driver_development.md) - Linux driver model and device driver development
7. [Device Tree](./device_tree.md) - Hardware description using Device Tree
8. [Cross Compilation](./cross_compilation.md) - Building for different architectures
9. [Networking](./networking.md) - Network configuration and troubleshooting
10. [Netfilter](./netfilter.md) - Packet filtering framework
11. [iptables](./iptables.md) - Firewall configuration
12. [Traffic Control (tc)](./tc.md) - Network traffic management
13. [WireGuard](./wireguard.md) - Modern VPN protocol and configuration
14. [systemd](./systemd.md) - Service management and init system
15. [sysctl](./sysctl.md) - Kernel parameter tuning at runtime
16. [sysfs](./sysfs.md) - Kernel/hardware information filesystem
17. [Netlink](./netlink.md) - Kernel-userspace communication interface
18. [eBPF](./ebpf.md) - Extended Berkeley Packet Filter for kernel programmability
19. [Control Groups (cgroups)](./cgroups.md) - Resource limiting, accounting, and isolation for process groups
20. [Process Internals (Kernel)](./process_internals.md) - How the kernel represents and schedules processes (task_struct, fork/exec, scheduler, context switch)
21. [Kernel Timers](./kernel_timers.md) - Timekeeping, delays, timer_list, hrtimers, and delayed work for kernel programming
22. [Synchronization](./synchronization.md) - Spinlocks, mutexes, semaphores, and rw locks: choosing and using kernel locks
23. [Interrupts & Deferred Work](./interrupts.md) - IRQ handling, top/bottom halves, softirqs, tasklets, threaded IRQs, workqueues, and NAPI
24. [CPU Scheduler](./scheduler.md) - Scheduling classes, CFS→EEVDF, real-time policies, preemption, SMP load balancing, and the cgroup cpu controller
25. [RCU (Read-Copy-Update)](./rcu.md) - Lock-free read-mostly synchronization: grace periods, publish/subscribe, and SRCU
26. [Block I/O Layer](./block_layer.md) - The storage stack: bios, blk-mq, I/O schedulers, and device-mapper (LVM, dm-crypt, RAID)
27. [Inter-Process Communication](./ipc.md) - Pipes, signals, shared memory, semaphores, futexes, eventfd, and Unix-socket fd passing
28. [ELF, Linking & Loading](./elf_linking.md) - ELF format, static/dynamic linking, ld.so, PLT/GOT, PIE/ASLR, and the vDSO
29. [Boot Process & initramfs](./boot_process.md) - Firmware → bootloader → kernel → initramfs → PID 1, and boot debugging
30. [Container Runtimes](./container_runtimes.md) - How namespaces + cgroups + seccomp + overlayfs compose into containers (OCI, runc, Docker, Podman)
31. [journald & Logging](./journald_logging.md) - Kernel ring buffer, systemd-journald, syslog/rsyslog, and logrotate

## Overview

This documentation covers essential Linux topics for system administrators, developers, and power users. Each section provides practical examples, use cases, and best practices.

## Getting Started

### For Beginners
Start with [Essential Commands](./commands.md) to learn the fundamental Linux commands that you'll use daily.

### For System Administrators
- [Essential Commands](./commands.md) - Master command-line tools
- [Networking](./networking.md) - Network configuration and diagnostics
- [Boot Process & initramfs](./boot_process.md) - Firmware, bootloader, initramfs, and PID 1 handoff
- [iptables](./iptables.md) - Firewall management
- [WireGuard](./wireguard.md) - VPN setup and management
- [journald & Logging](./journald_logging.md) - journald, syslog/rsyslog, and log rotation

### For Developers
- [Kernel Architecture](./kernel.md) - Understand Linux internals
- [Process Internals (Kernel)](./process_internals.md) - task_struct, scheduling, fork/exec, and context switching
- [CPU Scheduler](./scheduler.md) - Scheduling classes, CFS→EEVDF, RT policies, preemption, and the cgroup cpu controller
- [Inter-Process Communication](./ipc.md) - Pipes, signals, shared memory, futexes, and Unix-socket fd passing
- [ELF, Linking & Loading](./elf_linking.md) - ELF format, dynamic linking, ld.so, PLT/GOT, and the vDSO
- [Memory Management](./memory_management.md) - Virtual memory, paging, allocators, and reclaim
- [Kernel Development Patterns](./kernel_patterns.md) - Coding patterns and best practices
- [Kernel Timers](./kernel_timers.md) - Delays, timers, hrtimers, and timed deferred work
- [Synchronization](./synchronization.md) - Spinlocks, mutexes, semaphores, and rw locks
- [Interrupts & Deferred Work](./interrupts.md) - IRQ handling, bottom halves, workqueues, and NAPI
- [RCU (Read-Copy-Update)](./rcu.md) - Lock-free read-mostly synchronization and grace periods
- [Block I/O Layer](./block_layer.md) - Storage stack: bios, blk-mq, I/O schedulers, device-mapper
- [Driver Development](./driver_development.md) - Linux driver model and device drivers
- [Device Tree](./device_tree.md) - Hardware description and parsing
- [Control Groups (cgroups)](./cgroups.md) - Resource control for containers and services
- [Container Runtimes](./container_runtimes.md) - How namespaces, cgroups, seccomp, and overlayfs compose into containers
- [Cross Compilation](./cross_compilation.md) - Building for embedded systems
- [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless driver development
- [Essential Commands](./commands.md) - Development and debugging tools

### For Network Engineers
- [Networking](./networking.md) - Network stack and protocols
- [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless networking subsystem
- [Netfilter](./netfilter.md) - Packet filtering framework
- [Traffic Control](./tc.md) - QoS and traffic shaping
- [WireGuard](./wireguard.md) - Modern VPN implementation

## Key Topics

### System Administration
- User and permission management
- Process management and monitoring
- System resource monitoring
- Service management with systemd
- Log management and analysis

### Kernel Development
- Kernel architecture and components
- System calls and kernel modules
- Device drivers
- Kernel compilation and debugging

### Networking
- Network configuration (ip, ifconfig)
- Routing and bridging
- Packet filtering (iptables, nftables)
- Traffic shaping and QoS
- Network troubleshooting

## Quick Reference

### Most Used Commands
```bash
# File operations
ls -lah                    # List files with details
cd /path/to/directory      # Change directory
cp -r source dest          # Copy recursively
mv source dest             # Move/rename
rm -rf directory           # Remove recursively

# Text processing
grep pattern file          # Search for pattern
sed 's/old/new/g' file     # Replace text
awk '{print $1}' file      # Process columns

# System monitoring
top                        # Process viewer
htop                       # Enhanced process viewer
ps aux                     # List all processes
df -h                      # Disk usage
free -h                    # Memory usage

# Network
ip addr show               # Show IP addresses
ss -tulpn                  # Show listening ports
ping host                  # Test connectivity
curl url                   # HTTP client
```

### System Information
```bash
uname -a                   # Kernel version
lsb_release -a             # Distribution info
hostnamectl                # System hostname
uptime                     # System uptime
```

## Learning Path

1. **Basics** (1-2 weeks)
   - File system navigation
   - File manipulation
   - Text editors (vim, nano)
   - Basic shell scripting

2. **Intermediate** (2-4 weeks)
   - Process management
   - User management
   - Permissions and ownership
   - Package management
   - System services

3. **Advanced** (1-3 months)
   - Kernel modules
   - Network configuration
   - Firewall rules
   - Performance tuning
   - Security hardening

4. **Expert** (3-6 months)
   - Kernel development
   - Custom modules
   - Advanced networking
   - High availability systems
   - Container orchestration

## Best Practices

### Security
- Always use sudo instead of root login
- Keep system and packages updated
- Use SSH keys instead of passwords
- Enable and configure firewall
- Regular security audits
- Monitor system logs

### Performance
- Monitor system resources regularly
- Use appropriate file systems
- Optimize kernel parameters
- Implement proper backup strategies
- Use automation tools

### Documentation
- Document custom configurations
- Keep change logs
- Use version control for configs
- Create runbooks for common tasks

## Useful Resources

### Official Documentation
- [Linux Kernel Documentation](https://www.kernel.org/doc/html/latest/)
- [GNU Core Utilities](https://www.gnu.org/software/coreutils/manual/)
- [systemd Documentation](https://www.freedesktop.org/wiki/Software/systemd/)

### Community Resources
- [Linux Documentation Project](https://tldp.org/)
- [Arch Wiki](https://wiki.archlinux.org/)
- [Ubuntu Documentation](https://help.ubuntu.com/)

### Books
- "The Linux Command Line" by William Shotts
- "Linux Kernel Development" by Robert Love
- "UNIX and Linux System Administration Handbook"

## Contributing

When adding new documentation:
1. Follow the existing structure
2. Include practical examples
3. Add use cases and scenarios
4. Reference related sections
5. Keep examples tested and working

## Version Information

- Documentation maintained for Linux Kernel 5.x and 6.x
- Examples tested on Ubuntu 20.04/22.04 and Debian 11/12
- Command syntax may vary slightly between distributions
