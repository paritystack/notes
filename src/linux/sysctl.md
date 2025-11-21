# sysctl

sysctl is a powerful tool for examining and changing kernel parameters at runtime. It allows system administrators to modify kernel behavior without rebooting, making it essential for performance tuning, security hardening, and troubleshooting.

## Basic Usage

```bash
# List all parameters
sysctl -a

# Search for specific parameters
sysctl -a | grep net.ipv4

# Get specific parameter
sysctl net.ipv4.ip_forward

# Set parameter (temporary - lost on reboot)
sudo sysctl -w net.ipv4.ip_forward=1

# Load from configuration file
sudo sysctl -p /etc/sysctl.conf

# Load from specific file
sudo sysctl -p /etc/sysctl.d/99-custom.conf

# Apply all files from sysctl.d
sudo sysctl --system
```

## Common Parameters

### Network Settings

```bash
# IP Forwarding - enables routing between network interfaces
net.ipv4.ip_forward = 1                    # IPv4 forwarding (required for routers/NAT)
net.ipv6.conf.all.forwarding = 1           # IPv6 forwarding

# TCP Security and Performance
net.ipv4.tcp_syncookies = 1                # SYN flood protection (prevents DoS)
net.ipv4.tcp_max_syn_backlog = 2048        # SYN backlog queue size
net.core.somaxconn = 1024                  # Max connection backlog (increase for busy servers)
net.core.netdev_max_backlog = 5000         # Packet queue size before kernel processing

# TCP Window Scaling - improves performance on high-latency networks
net.ipv4.tcp_window_scaling = 1            # Enable window scaling
net.ipv4.tcp_timestamps = 1                # TCP timestamps (needed for window scaling)

# Connection Tracking
net.ipv4.tcp_fin_timeout = 30              # Time to wait for final FIN (reduce for faster cleanup)
net.ipv4.tcp_keepalive_time = 300          # Time before sending keepalive probes (seconds)
net.ipv4.tcp_keepalive_probes = 5          # Number of keepalive probes
net.ipv4.tcp_keepalive_intvl = 15          # Interval between keepalive probes

# Network Buffer Sizes
net.core.rmem_default = 262144             # Default receive buffer size
net.core.wmem_default = 262144             # Default send buffer size
net.core.rmem_max = 16777216               # Maximum receive buffer size
net.core.wmem_max = 16777216               # Maximum send buffer size
```

### Memory and VM Settings

```bash
# Swappiness - controls swap usage preference (0-100)
vm.swappiness = 10                         # Low value: prefer RAM (good for databases)
                                           # High value (60): balanced (default)
                                           # Very high (100): aggressive swapping

# Dirty Pages - controls when data is written to disk
vm.dirty_ratio = 15                        # % of memory that can be dirty before sync writes
vm.dirty_background_ratio = 5              # % of memory before background writes begin
vm.dirty_expire_centisecs = 3000           # How old data must be to be written (1/100s)
vm.dirty_writeback_centisecs = 500         # How often to wake up write daemon

# Memory Overcommit - controls memory allocation behavior
vm.overcommit_memory = 0                   # 0: heuristic (default)
                                           # 1: always allow (can cause OOM)
                                           # 2: never overcommit (strict)
vm.overcommit_ratio = 50                   # % of RAM to allow when overcommit_memory=2

# Memory Management
vm.min_free_kbytes = 65536                 # Minimum free memory to maintain (KB)
vm.vfs_cache_pressure = 100                # Tendency to reclaim inode/dentry cache (default 100)
```

### File System Settings

```bash
# File Descriptors
fs.file-max = 65536                        # Max open files system-wide
fs.nr_open = 1048576                       # Max files a process can open

# Inotify - file system monitoring limits
fs.inotify.max_user_watches = 524288       # Max watched files per user (increase for IDEs)
fs.inotify.max_user_instances = 256        # Max inotify instances per user
fs.inotify.max_queued_events = 16384       # Max queued events

# AIO - asynchronous I/O
fs.aio-max-nr = 1048576                    # Max concurrent async I/O operations
```

### Kernel Settings

```bash
# SysRq Key - emergency kernel commands (Alt+SysRq+command)
kernel.sysrq = 1                           # 1: enable all, 0: disable
                                           # For specific: 244 (common safe subset)

# Panic Behavior
kernel.panic = 10                          # Reboot after kernel panic (seconds)
kernel.panic_on_oops = 1                   # Treat oops as panic (safer for production)

# Process Limits
kernel.pid_max = 65536                     # Maximum process ID value
kernel.threads-max = 65536                 # Maximum number of threads

# Shared Memory
kernel.shmmax = 68719476736                # Max shared memory segment size (bytes)
kernel.shmall = 4294967296                 # Total shared memory pages available
```

## Security Hardening

```bash
# Network Security - IP Spoofing Protection
net.ipv4.conf.all.rp_filter = 1            # Enable source address verification
net.ipv4.conf.default.rp_filter = 1        # Apply to new interfaces

# ICMP Settings - Protection against ICMP attacks
net.ipv4.icmp_echo_ignore_all = 0          # 0: respond to ping, 1: ignore ping
net.ipv4.icmp_echo_ignore_broadcasts = 1   # Ignore broadcast pings (smurf attack)
net.ipv4.icmp_ignore_bogus_error_responses = 1  # Ignore malformed ICMP errors

# IP Redirect Protection - prevents MITM attacks
net.ipv4.conf.all.accept_redirects = 0     # Don't accept ICMP redirects
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0     # Don't accept secure redirects
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.send_redirects = 0       # Don't send redirects
net.ipv4.conf.default.send_redirects = 0

# IPv6 Security
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_ra = 0            # Don't accept router advertisements
net.ipv6.conf.default.accept_ra = 0

# Source Routing - disable to prevent packet injection
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Suspicious Packets
net.ipv4.conf.all.log_martians = 1         # Log packets with impossible addresses
net.ipv4.conf.default.log_martians = 1

# Kernel Security - ASLR and Core Dumps
kernel.randomize_va_space = 2              # Full ASLR (address space layout randomization)
                                           # 0: disabled, 1: partial, 2: full
kernel.core_uses_pid = 1                   # Append PID to core dump filename
kernel.core_pattern = |/bin/false          # Disable core dumps for security
# Or specify location: /var/crash/core-%e-%p-%t

# Restrict dmesg Access - hide kernel logs from non-root
kernel.dmesg_restrict = 1                  # 1: only CAP_SYSLOG can read

# Restrict kernel pointers - prevent information leaks
kernel.kptr_restrict = 2                   # 0: visible, 1: root only, 2: always hidden

# Restrict perf events - prevent performance monitoring exploits
kernel.perf_event_paranoid = 3             # 3: restricted to root, 2: normal users limited

# Yama Security Module - ptrace restrictions
kernel.yama.ptrace_scope = 1               # 0: disabled, 1: restricted, 2: admin only, 3: disabled
```

## Persistent Configuration

Sysctl changes made with `sysctl -w` are temporary and lost after reboot. For persistent configuration:

```bash
# System-wide configuration
# /etc/sysctl.conf (legacy)
net.ipv4.ip_forward = 1
vm.swappiness = 10
fs.file-max = 100000

# Modular configuration (preferred)
# /etc/sysctl.d/99-custom.conf
# Files are processed in lexical order, higher numbers = higher priority
net.ipv4.ip_forward = 1
vm.swappiness = 10

# Apply configuration
sudo sysctl -p                             # Load /etc/sysctl.conf
sudo sysctl -p /etc/sysctl.d/99-custom.conf  # Load specific file
sudo sysctl --system                       # Load all config files
```

### Configuration File Locations (loaded in order)

```
/etc/sysctl.d/*.conf
/run/sysctl.d/*.conf
/usr/local/lib/sysctl.d/*.conf
/usr/lib/sysctl.d/*.conf
/lib/sysctl.d/*.conf
/etc/sysctl.conf
```

## Performance Tuning

### High-Performance Networking

```bash
# For servers handling many connections (web servers, load balancers)
net.core.rmem_max = 134217728              # 128 MB max receive buffer
net.core.wmem_max = 134217728              # 128 MB max send buffer
net.ipv4.tcp_rmem = 4096 87380 67108864    # Min, default, max read buffer
net.ipv4.tcp_wmem = 4096 65536 67108864    # Min, default, max write buffer
net.ipv4.tcp_congestion_control = bbr      # BBR congestion control (better than cubic)
net.core.somaxconn = 4096                  # Increase connection queue
net.core.netdev_max_backlog = 10000        # Increase packet processing queue
net.ipv4.tcp_max_syn_backlog = 8192        # Increase SYN backlog

# Fast connection recycling
net.ipv4.tcp_tw_reuse = 1                  # Reuse TIME-WAIT sockets
net.ipv4.tcp_fin_timeout = 15              # Reduce FIN-WAIT-2 timeout
```

### Database Server Optimization

```bash
# Minimize swapping (crucial for database performance)
vm.swappiness = 1                          # Almost never swap (keep data in RAM)

# Optimize dirty page handling for consistent performance
vm.dirty_background_ratio = 5              # Start background writes early
vm.dirty_ratio = 10                        # Force sync writes sooner
vm.dirty_expire_centisecs = 500            # Write data older than 5s
vm.dirty_writeback_centisecs = 100         # Check every 1s for dirty pages

# Increase shared memory (for PostgreSQL, Oracle, etc.)
kernel.shmmax = 68719476736                # 64 GB max shared memory segment
kernel.shmall = 4294967296                 # Total pages (64 GB with 4KB pages)

# File descriptors
fs.file-max = 2097152                      # Increase for many connections
```

### Container Hosts (Docker/Kubernetes)

```bash
# Networking for container environments
net.bridge.bridge-nf-call-iptables = 1     # Enable iptables for bridged traffic
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1                    # Required for container networking

# Inotify limits (many containers = many file watches)
fs.inotify.max_user_watches = 1048576      # Increase for container monitoring
fs.inotify.max_user_instances = 512

# Increase connection tracking
net.netfilter.nf_conntrack_max = 1000000   # More tracked connections
net.netfilter.nf_conntrack_buckets = 250000

# File descriptors
fs.file-max = 2097152                      # Many containers need many files
```

### High File I/O Systems

```bash
# For systems with heavy disk I/O (file servers, build systems)
vm.dirty_ratio = 10                        # Lower to avoid I/O stalls
vm.dirty_background_ratio = 5              # Start writing sooner
vm.vfs_cache_pressure = 50                 # Keep directory/inode cache longer

# AIO for databases and applications using async I/O
fs.aio-max-nr = 1048576                    # Increase async I/O capacity

# File descriptors
fs.file-max = 2097152                      # Increase for many open files
```

## Troubleshooting

### Verify Current Settings

```bash
# Check current runtime value
sysctl net.ipv4.ip_forward

# Check if setting is in config files
grep -r "net.ipv4.ip_forward" /etc/sysctl.conf /etc/sysctl.d/

# Compare current vs. default
sysctl -a | grep tcp_syncookies

# View all network settings
sysctl -a | grep ^net

# View all settings (warning: very long output)
sysctl -a
```

### Common Issues

**Changes don't persist after reboot:**
- Ensure settings are in `/etc/sysctl.conf` or `/etc/sysctl.d/*.conf`
- Run `sudo sysctl --system` to test loading all configs
- Check file syntax (no spaces around `=` is recommended)

**"Cannot stat /proc/sys/..." error:**
- Kernel module not loaded (e.g., `modprobe br_netfilter` for bridge settings)
- Parameter doesn't exist in your kernel version
- Check with `ls /proc/sys/...` to verify path

**"Permission denied" when setting parameter:**
- Must use `sudo` or run as root
- Some parameters cannot be changed (read-only)
- Check if parameter is namespaced (container-specific)

**Settings conflict or get overridden:**
- Files in `/etc/sysctl.d/` are loaded in lexical order
- Higher numbered files (e.g., `99-*.conf`) override lower ones
- `/etc/sysctl.conf` is loaded last and can override everything
- Check with `sysctl --system` to see load order

### Debugging Techniques

```bash
# Test before making persistent
sudo sysctl -w net.ipv4.ip_forward=1       # Test immediately
# If it works, add to config file

# Validate configuration file syntax
sudo sysctl -p /etc/sysctl.d/99-custom.conf --dry-run

# Monitor kernel messages for errors
sudo dmesg | tail -20

# Check which file set a parameter
grep -r "parameter_name" /etc/sysctl.conf /etc/sysctl.d/

# Reset parameter to default (reboot or check kernel docs for default)
# Example: most net.ipv4.ip_forward defaults to 0
sudo sysctl -w net.ipv4.ip_forward=0

# See all networking parameters and current values
sudo sysctl -a --pattern 'net.*'
```

### Testing Changes

```bash
# Before changing network settings
ping 8.8.8.8                               # Test connectivity

# After enabling IP forwarding
cat /proc/sys/net/ipv4/ip_forward          # Should show 1
# Test actual routing functionality

# After memory changes
free -h                                     # Check memory usage
cat /proc/meminfo                          # Detailed memory info

# Monitor impact of changes
vmstat 1                                    # System stats every second
sar -n DEV 1                               # Network stats (if sysstat installed)
```

## Best Practices

1. **Test before making persistent:** Always test with `sysctl -w` first
2. **Document changes:** Add comments in config files explaining why you changed values
3. **Use sysctl.d:** Create separate files in `/etc/sysctl.d/` for different purposes (e.g., `10-security.conf`, `90-performance.conf`)
4. **Monitor impact:** Watch system performance after changes to ensure improvements
5. **Know your workload:** Database servers, web servers, and desktop systems need different tuning
6. **Start conservative:** Make incremental changes and measure results
7. **Keep backups:** Save original values before making changes

sysctl provides powerful runtime kernel tuning for optimizing system performance, hardening security, and troubleshooting issues without requiring system reboots.
