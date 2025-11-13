# Linux Documentation

A comprehensive guide to Linux system administration, commands, kernel architecture, and networking.

## Table of Contents

1. [Essential Commands](./commands.md) - Command reference and examples
2. [Kernel Architecture](./kernel.md) - Linux kernel internals and development
3. [Kernel Development Patterns](./kernel_patterns.md) - Common patterns and best practices for kernel development
4. [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless subsystem frameworks for WiFi drivers
5. [Driver Development](./driver_development.md) - Linux driver model and device driver development
6. [Device Tree](./device_tree.md) - Hardware description using Device Tree
7. [Networking](./networking.md) - Network configuration and troubleshooting
8. [Netfilter](./netfilter.md) - Packet filtering framework
9. [iptables](./iptables.md) - Firewall configuration
10. [Traffic Control (tc)](./tc.md) - Network traffic management
11. [systemd](./systemd.md) - Service management and init system
12. [sysctl](./sysctl.md) - Kernel parameter tuning at runtime
13. [sysfs](./sysfs.md) - Kernel/hardware information filesystem

## Overview

This documentation covers essential Linux topics for system administrators, developers, and power users. Each section provides practical examples, use cases, and best practices.

## Getting Started

### For Beginners
Start with [Essential Commands](./commands.md) to learn the fundamental Linux commands that you'll use daily.

### For System Administrators
- [Essential Commands](./commands.md) - Master command-line tools
- [Networking](./networking.md) - Network configuration and diagnostics
- [iptables](./iptables.md) - Firewall management

### For Developers
- [Kernel Architecture](./kernel.md) - Understand Linux internals
- [Kernel Development Patterns](./kernel_patterns.md) - Coding patterns and best practices
- [Driver Development](./driver_development.md) - Linux driver model and device drivers
- [Device Tree](./device_tree.md) - Hardware description and parsing
- [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless driver development
- [Essential Commands](./commands.md) - Development and debugging tools

### For Network Engineers
- [Networking](./networking.md) - Network stack and protocols
- [cfg80211 & mac80211](./cfg80211_mac80211.md) - Wireless networking subsystem
- [Netfilter](./netfilter.md) - Packet filtering framework
- [Traffic Control](./tc.md) - QoS and traffic shaping

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
