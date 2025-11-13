# sysctl

sysctl is a tool for examining and changing kernel parameters at runtime. It's used to modify kernel behavior without rebooting.

## Basic Usage

```bash
# List all parameters
sysctl -a

# Get specific parameter
sysctl net.ipv4.ip_forward

# Set parameter (temporary)
sudo sysctl -w net.ipv4.ip_forward=1

# Load from configuration file
sudo sysctl -p /etc/sysctl.conf
```

## Common Parameters

```bash
# Network settings
net.ipv4.ip_forward = 1                    # Enable IP forwarding
net.ipv4.tcp_syncookies = 1                # SYN flood protection
net.core.somaxconn = 1024                  # Connection backlog
net.ipv4.tcp_max_syn_backlog = 2048        # SYN backlog

# Memory settings
vm.swappiness = 10                         # Swap preference (0-100)
vm.dirty_ratio = 15                        # Dirty page threshold
vm.overcommit_memory = 1                   # Memory overcommit

# File system
fs.file-max = 65536                        # Max open files
fs.inotify.max_user_watches = 524288       # inotify watches

# Kernel settings
kernel.sysrq = 1                           # Enable SysRq key
kernel.panic = 10                          # Reboot after panic (seconds)
```

## Persistent Configuration

```bash
# /etc/sysctl.conf or /etc/sysctl.d/99-custom.conf
net.ipv4.ip_forward = 1
vm.swappiness = 10
fs.file-max = 100000

# Apply configuration
sudo sysctl -p
```

## Performance Tuning

```bash
# High-performance networking
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.ipv4.tcp_congestion_control = bbr

# Database server optimization
vm.swappiness = 1
vm.dirty_background_ratio = 5
vm.dirty_ratio = 10
```

sysctl provides runtime kernel tuning for optimizing system performance and behavior.
