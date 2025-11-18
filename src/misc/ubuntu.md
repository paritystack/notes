# Ubuntu

A comprehensive guide to Ubuntu Linux, covering fundamentals, system administration, package management, networking, security, and best practices.

## Table of Contents

1. [Ubuntu Fundamentals](#ubuntu-fundamentals)
2. [Installation and Setup](#installation-and-setup)
3. [Package Management](#package-management)
4. [File System and Storage](#file-system-and-storage)
5. [User and Permission Management](#user-and-permission-management)
6. [Process and Service Management](#process-and-service-management)
7. [Networking](#networking)
8. [Security](#security)
9. [System Monitoring and Performance](#system-monitoring-and-performance)
10. [Shell and Scripting](#shell-and-scripting)
11. [Cloud and Server Administration](#cloud-and-server-administration)
12. [Troubleshooting](#troubleshooting)

---

## Ubuntu Fundamentals

**Ubuntu** is a Debian-based Linux distribution that emphasizes ease of use, regular releases, and community-driven development. It's one of the most popular Linux distributions for desktops, servers, and cloud deployments.

### Ubuntu Philosophy

- **Free and Open Source**: Ubuntu is completely free to download, use, and share
- **Community-Driven**: Backed by Canonical Ltd. but driven by community
- **Regular Releases**: Predictable 6-month release cycle
- **Long-Term Support**: LTS releases supported for 5 years (10 years with Extended Security Maintenance)

### Ubuntu Versions

**Release Cycle:**
- **Standard Releases**: Supported for 9 months (e.g., 23.10, 24.04)
- **LTS Releases**: Long-Term Support, released every 2 years in April (e.g., 20.04, 22.04, 24.04)
- **Version Naming**: Year.Month format (24.04 = April 2024)
- **Codenames**: Alliterative animal names (Focal Fossa, Jammy Jellyfish, Noble Numbat)

**Current LTS Versions (as of 2025):**
- **Ubuntu 24.04 LTS (Noble Numbat)** - Latest LTS
- **Ubuntu 22.04 LTS (Jammy Jellyfish)** - Widely deployed
- **Ubuntu 20.04 LTS (Focal Fossa)** - Still supported

### Ubuntu Flavors

Official variants with different desktop environments:
- **Ubuntu Desktop**: GNOME desktop environment (default)
- **Kubuntu**: KDE Plasma desktop
- **Xubuntu**: Xfce desktop (lightweight)
- **Lubuntu**: LXQt desktop (very lightweight)
- **Ubuntu MATE**: MATE desktop
- **Ubuntu Budgie**: Budgie desktop
- **Ubuntu Server**: No GUI, optimized for servers

---

## Installation and Setup

### System Requirements

**Minimum Requirements:**
- **CPU**: 2 GHz dual-core processor
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 25 GB free space (minimum)
- **Display**: 1024×768 resolution

**Recommended for Desktop:**
- **CPU**: 3 GHz quad-core processor
- **RAM**: 8 GB or more
- **Storage**: 50+ GB SSD
- **GPU**: Modern graphics card for smooth desktop experience

### Installation Methods

#### 1. Clean Installation
Install Ubuntu as the only operating system on the computer.

```bash
# Download ISO from ubuntu.com
# Create bootable USB with tools like:
# - Rufus (Windows)
# - Etcher (Cross-platform)
# - dd (Linux)

# Example using dd (be careful with device names!)
sudo dd if=ubuntu-24.04-desktop-amd64.iso of=/dev/sdX bs=4M status=progress oflag=sync
```

#### 2. Dual Boot
Install Ubuntu alongside another operating system (e.g., Windows).

**Important Considerations:**
- Disable Fast Startup in Windows
- Disable Secure Boot (or configure for Ubuntu)
- Backup important data before partitioning
- Create separate partitions for / (root), /home, and swap

#### 3. Virtual Machine
Run Ubuntu inside VirtualBox, VMware, or KVM.

**Advantages:**
- No risk to existing system
- Easy to snapshot and restore
- Good for testing and learning

#### 4. WSL2 (Windows Subsystem for Linux)
Run Ubuntu within Windows 10/11.

```powershell
# Install WSL2 on Windows
wsl --install -d Ubuntu

# Or install specific version
wsl --install -d Ubuntu-22.04
```

### Post-Installation Setup

#### Update System
```bash
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Distribution upgrade (more comprehensive)
sudo apt full-upgrade -y

# Clean up
sudo apt autoremove -y
sudo apt autoclean
```

#### Install Essential Software
```bash
# Development tools
sudo apt install build-essential git curl wget vim -y

# Common utilities
sudo apt install htop tree net-tools openssh-server -y

# Additional codecs and fonts (for desktop)
sudo apt install ubuntu-restricted-extras -y
```

#### Configure System Settings
```bash
# Set timezone
sudo timedatectl set-timezone America/New_York

# Set hostname
sudo hostnamectl set-hostname my-ubuntu-server

# Configure automatic security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## Package Management

Ubuntu uses **APT** (Advanced Package Tool) and **dpkg** for package management. Packages are distributed in `.deb` format.

### APT Commands

#### Basic Package Operations

```bash
# Update package lists from repositories
sudo apt update

# Upgrade all installed packages
sudo apt upgrade

# Full upgrade (handles dependencies more aggressively)
sudo apt full-upgrade

# Install a package
sudo apt install package-name

# Install multiple packages
sudo apt install package1 package2 package3

# Install specific version
sudo apt install package-name=version

# Remove package (keep configuration files)
sudo apt remove package-name

# Remove package and configuration files
sudo apt purge package-name

# Remove unused dependencies
sudo apt autoremove

# Clean downloaded package files
sudo apt clean
sudo apt autoclean
```

#### Search and Information

```bash
# Search for packages
apt search keyword

# Show package information
apt show package-name

# List installed packages
apt list --installed

# List upgradable packages
apt list --upgradable

# Show package dependencies
apt depends package-name

# Show reverse dependencies (what depends on this package)
apt rdepends package-name
```

#### Advanced Package Management

```bash
# Hold a package (prevent upgrades)
sudo apt-mark hold package-name

# Unhold a package
sudo apt-mark unhold package-name

# Download package without installing
apt download package-name

# Simulate installation (dry run)
apt install -s package-name

# Fix broken dependencies
sudo apt --fix-broken install

# Reconfigure a package
sudo dpkg-reconfigure package-name
```

### Repository Management

#### Sources List

Ubuntu repositories are configured in `/etc/apt/sources.list` and `/etc/apt/sources.list.d/`.

**Repository Components:**
- **main**: Officially supported free and open-source software
- **restricted**: Proprietary drivers for devices
- **universe**: Community-maintained free and open-source software
- **multiverse**: Software restricted by copyright or legal issues

```bash
# View current repositories
cat /etc/apt/sources.list

# Example repository format
# deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
# deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
# deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
```

#### Add PPA (Personal Package Archive)

```bash
# Add PPA
sudo add-apt-repository ppa:user/ppa-name

# Remove PPA
sudo add-apt-repository --remove ppa:user/ppa-name

# Update after adding repository
sudo apt update
```

#### Third-Party Repositories

```bash
# Add repository with GPG key
curl -fsSL https://example.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/example.gpg
echo "deb [signed-by=/usr/share/keyrings/example.gpg] https://example.com/repo stable main" | sudo tee /etc/apt/sources.list.d/example.list

# Update package lists
sudo apt update
```

### Snap Packages

Snap is a universal package format for Linux applications.

```bash
# Install snap (usually pre-installed)
sudo apt install snapd

# Search for snaps
snap find keyword

# Install snap package
sudo snap install package-name

# List installed snaps
snap list

# Update all snaps
sudo snap refresh

# Update specific snap
sudo snap refresh package-name

# Remove snap
sudo snap remove package-name

# View snap information
snap info package-name
```

### Flatpak (Alternative Package System)

```bash
# Install Flatpak
sudo apt install flatpak

# Add Flathub repository
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo

# Install application
flatpak install flathub app.id

# Run application
flatpak run app.id

# Update applications
flatpak update
```

---

## File System and Storage

### Linux File System Hierarchy

**Standard Directory Structure:**

```
/                    # Root directory
├── bin/            # Essential user binaries (commands)
├── boot/           # Boot loader files (kernel, initrd)
├── dev/            # Device files
├── etc/            # System configuration files
├── home/           # User home directories
├── lib/            # Shared libraries
├── media/          # Removable media mount points
├── mnt/            # Temporary mount points
├── opt/            # Optional software packages
├── proc/           # Process information (virtual)
├── root/           # Root user's home directory
├── run/            # Runtime data
├── sbin/           # System binaries (admin commands)
├── srv/            # Service data
├── sys/            # System information (virtual)
├── tmp/            # Temporary files
├── usr/            # User programs and data
│   ├── bin/        # User commands
│   ├── lib/        # Libraries
│   ├── local/      # Local software
│   └── share/      # Shared data
└── var/            # Variable data (logs, caches)
    ├── log/        # Log files
    ├── cache/      # Application cache
    └── tmp/        # Temporary files preserved between reboots
```

### File System Commands

#### Navigation and File Operations

```bash
# Print working directory
pwd

# Change directory
cd /path/to/directory
cd ~                # Home directory
cd -                # Previous directory
cd ..               # Parent directory

# List files
ls                  # Basic listing
ls -l               # Long format (permissions, owner, size, date)
ls -la              # Include hidden files
ls -lh              # Human-readable sizes
ls -lt              # Sort by modification time
ls -lS              # Sort by size

# Create directory
mkdir directory-name
mkdir -p path/to/nested/directory  # Create parent directories

# Remove files/directories
rm file
rm -r directory     # Recursive removal
rm -rf directory    # Force recursive removal (dangerous!)

# Copy files/directories
cp source destination
cp -r source-dir destination-dir  # Recursive copy
cp -a source destination          # Archive mode (preserve attributes)

# Move/rename files
mv source destination

# Create empty file or update timestamp
touch filename

# View file contents
cat file            # Display entire file
less file           # Paginated view
head file           # First 10 lines
head -n 20 file     # First 20 lines
tail file           # Last 10 lines
tail -f file        # Follow file updates (useful for logs)

# Find files
find /path -name "filename"
find /path -type f -name "*.txt"
find /path -mtime -7              # Modified in last 7 days
find /path -size +100M            # Files larger than 100MB

# Search file contents
grep "pattern" file
grep -r "pattern" directory       # Recursive search
grep -i "pattern" file            # Case-insensitive
grep -n "pattern" file            # Show line numbers
```

#### Disk Usage and Management

```bash
# Disk space usage
df -h               # Show disk space (human-readable)
df -i               # Show inode usage

# Directory size
du -sh directory    # Summary size
du -h --max-depth=1 # Size of subdirectories

# List block devices
lsblk

# Partition information
sudo fdisk -l
sudo parted -l

# Mount filesystem
sudo mount /dev/sdX1 /mnt/mountpoint

# Unmount filesystem
sudo umount /mnt/mountpoint

# View mounted filesystems
mount | column -t

# Edit fstab for persistent mounts
sudo vim /etc/fstab
# Example entry:
# UUID=xxxx-xxxx /mnt/data ext4 defaults 0 2
```

#### File Permissions

**Permission Format:** `rwxrwxrwx` (User, Group, Others)
- **r** = read (4)
- **w** = write (2)
- **x** = execute (1)

```bash
# Change file permissions
chmod 755 file      # rwxr-xr-x
chmod 644 file      # rw-r--r--
chmod +x file       # Add execute permission
chmod -w file       # Remove write permission
chmod u+x file      # Add execute for user
chmod g-w file      # Remove write for group
chmod o=r file      # Set read-only for others

# Change ownership
chown user:group file
chown -R user:group directory  # Recursive

# Change group
chgrp group file

# View permissions
ls -l file
stat file           # Detailed file information
```

#### Links

```bash
# Hard link (same inode, same file)
ln source-file link-name

# Symbolic link (pointer to file)
ln -s source-file link-name
ln -s /path/to/directory link-name

# View link information
ls -l link-name
readlink link-name
```

---

## User and Permission Management

### User Management

```bash
# Add user
sudo adduser username

# Add user with specific UID and home directory
sudo useradd -u 1001 -m -s /bin/bash username

# Delete user (keep home directory)
sudo deluser username

# Delete user and home directory
sudo deluser --remove-home username

# Modify user
sudo usermod -l newname oldname    # Rename user
sudo usermod -d /new/home username # Change home directory
sudo usermod -s /bin/zsh username  # Change shell

# Lock/unlock user account
sudo passwd -l username    # Lock
sudo passwd -u username    # Unlock

# Set password
sudo passwd username

# View user information
id username
finger username
getent passwd username

# List all users
cat /etc/passwd
cut -d: -f1 /etc/passwd

# View currently logged-in users
who
w
users

# View last logins
last
lastlog
```

### Group Management

```bash
# Add group
sudo addgroup groupname

# Delete group
sudo delgroup groupname

# Add user to group
sudo usermod -aG groupname username
sudo adduser username groupname

# Remove user from group
sudo deluser username groupname

# View user's groups
groups username
id username

# View group information
getent group groupname

# List all groups
cat /etc/group
```

### Sudo and Privileges

```bash
# Add user to sudo group (grants sudo privileges)
sudo usermod -aG sudo username

# Edit sudoers file (use visudo for safety)
sudo visudo

# Example sudoers configurations:
# Allow user to run all commands
# username ALL=(ALL:ALL) ALL

# Allow user to run specific command without password
# username ALL=(ALL) NOPASSWD: /usr/bin/apt

# Allow group to run all commands
# %groupname ALL=(ALL:ALL) ALL

# Run command as another user
sudo -u username command

# Run command as root
sudo command

# Start interactive root shell
sudo -i
sudo su -

# View sudo permissions
sudo -l
```

---

## Process and Service Management

### Process Management

```bash
# List running processes
ps aux              # All processes, detailed
ps -ef              # All processes, different format
pstree              # Process tree

# Interactive process viewer
top                 # Basic process monitor
htop                # Enhanced process monitor (install: sudo apt install htop)
btop                # Modern process monitor (install: sudo apt install btop)

# Search for processes
ps aux | grep process-name
pgrep process-name
pidof process-name

# Process information
ps -p PID -o comm,pid,ppid,user,%cpu,%mem

# Kill processes
kill PID            # Graceful termination (SIGTERM)
kill -9 PID         # Force kill (SIGKILL)
killall process-name
pkill -f pattern

# Background and foreground jobs
command &           # Run in background
jobs                # List background jobs
fg %1               # Bring job 1 to foreground
bg %1               # Resume job 1 in background
Ctrl+Z              # Suspend current process
nohup command &     # Run process that survives terminal exit

# Process priority
nice -n 10 command  # Start with lower priority (+10)
renice -n -5 PID    # Change priority of running process

# Resource limits
ulimit -a           # Show all limits
ulimit -n 4096      # Set max open files
```

### Systemd Service Management

Systemd is the init system and service manager for Ubuntu.

```bash
# Service control
sudo systemctl start service-name
sudo systemctl stop service-name
sudo systemctl restart service-name
sudo systemctl reload service-name
sudo systemctl status service-name

# Enable/disable services at boot
sudo systemctl enable service-name
sudo systemctl disable service-name
sudo systemctl is-enabled service-name

# List services
systemctl list-units --type=service
systemctl list-units --type=service --state=running
systemctl list-unit-files --type=service

# View service logs
journalctl -u service-name
journalctl -u service-name -f           # Follow logs
journalctl -u service-name --since today
journalctl -u service-name --since "2024-01-01" --until "2024-01-31"

# System targets (runlevels)
systemctl get-default                   # Current target
sudo systemctl set-default multi-user.target  # Set default to non-GUI
sudo systemctl set-default graphical.target   # Set default to GUI
systemctl isolate rescue.target         # Switch to rescue mode
```

#### Create Custom Service

```bash
# Create service file
sudo vim /etc/systemd/system/myapp.service
```

```ini
[Unit]
Description=My Application Service
After=network.target

[Service]
Type=simple
User=myuser
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/start.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable myapp
sudo systemctl start myapp
sudo systemctl status myapp
```

---

## Networking

### Network Configuration

#### NetworkManager (Desktop)

```bash
# Command-line tool for NetworkManager
nmcli

# Show network devices
nmcli device status

# Show connections
nmcli connection show

# Connect to WiFi
nmcli device wifi list
nmcli device wifi connect SSID password PASSWORD

# Show IP configuration
nmcli device show eth0

# Restart NetworkManager
sudo systemctl restart NetworkManager
```

#### Netplan (Server)

Ubuntu uses Netplan for network configuration on servers.

```bash
# Configuration file location
/etc/netplan/*.yaml

# Example: Static IP configuration
sudo vim /etc/netplan/00-installer-config.yaml
```

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

```bash
# Apply configuration
sudo netplan apply

# Test configuration
sudo netplan try

# Generate debug information
sudo netplan --debug generate
```

#### Example: DHCP Configuration

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: true
```

### Network Commands

```bash
# Show IP addresses
ip addr show
ip a

# Show specific interface
ip addr show eth0

# Show routing table
ip route show
route -n

# Add/remove IP address
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip addr del 192.168.1.100/24 dev eth0

# Enable/disable interface
sudo ip link set eth0 up
sudo ip link set eth0 down

# Show network statistics
ip -s link

# DNS configuration
cat /etc/resolv.conf

# Set DNS servers (managed by systemd-resolved)
sudo vim /etc/systemd/resolved.conf

# Flush DNS cache
sudo systemd-resolve --flush-caches
sudo resolvectl flush-caches

# Test DNS resolution
nslookup example.com
dig example.com
host example.com

# Network connectivity tests
ping -c 4 8.8.8.8
ping6 -c 4 google.com

# Trace route
traceroute google.com
mtr google.com          # Better alternative (install: sudo apt install mtr)

# Show network connections
ss -tuln                # All listening TCP/UDP ports
netstat -tuln           # Old alternative
ss -tunap               # Show process names

# Show established connections
ss -tunap | grep ESTAB

# Port scanning
nmap localhost          # Install: sudo apt install nmap
```

### Firewall (UFW)

UFW (Uncomplicated Firewall) is Ubuntu's firewall frontend.

```bash
# Enable/disable firewall
sudo ufw enable
sudo ufw disable

# Check status
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow/deny ports
sudo ufw allow 22                # SSH
sudo ufw allow 80/tcp            # HTTP
sudo ufw allow 443/tcp           # HTTPS
sudo ufw deny 23                 # Deny telnet

# Allow from specific IP
sudo ufw allow from 192.168.1.100

# Allow from subnet
sudo ufw allow from 192.168.1.0/24

# Allow specific port from specific IP
sudo ufw allow from 192.168.1.100 to any port 22

# Delete rules
sudo ufw delete allow 80
sudo ufw delete 3               # Delete rule number 3

# Application profiles
sudo ufw app list
sudo ufw allow 'OpenSSH'
sudo ufw allow 'Nginx Full'

# Reset firewall
sudo ufw reset
```

### SSH Configuration

```bash
# Install SSH server
sudo apt install openssh-server

# SSH service management
sudo systemctl status ssh
sudo systemctl start ssh
sudo systemctl enable ssh

# SSH configuration file
sudo vim /etc/ssh/sshd_config

# Important SSH settings:
# Port 22
# PermitRootLogin no
# PubkeyAuthentication yes
# PasswordAuthentication no
# AllowUsers user1 user2

# Restart SSH after configuration changes
sudo systemctl restart ssh

# Generate SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy public key to remote server
ssh-copy-id user@remote-host

# Connect to remote server
ssh user@hostname
ssh -p 2222 user@hostname       # Custom port
ssh -i ~/.ssh/custom_key user@hostname

# SSH config file for shortcuts
vim ~/.ssh/config
```

```
Host myserver
    HostName 192.168.1.100
    User myuser
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

```bash
# Now connect with:
ssh myserver
```

---

## Security

### Security Best Practices

1. **Keep system updated**
2. **Use strong passwords or SSH keys**
3. **Enable and configure firewall**
4. **Disable root login**
5. **Use sudo instead of root**
6. **Install only necessary software**
7. **Regular backups**
8. **Monitor logs**
9. **Enable automatic security updates**

### Automatic Security Updates

```bash
# Install unattended-upgrades
sudo apt install unattended-upgrades

# Configure automatic updates
sudo dpkg-reconfigure -plow unattended-upgrades

# Configuration file
sudo vim /etc/apt/apt.conf.d/50unattended-upgrades

# Enable automatic reboot if required
# Uncomment and set:
# Unattended-Upgrade::Automatic-Reboot "true";
# Unattended-Upgrade::Automatic-Reboot-Time "02:00";
```

### AppArmor

AppArmor is a Mandatory Access Control (MAC) system for Linux.

```bash
# Check AppArmor status
sudo aa-status

# AppArmor modes:
# - enforce: Rules are enforced
# - complain: Rules violations are logged but not blocked
# - disabled: Profile not loaded

# Set profile to complain mode
sudo aa-complain /path/to/profile

# Set profile to enforce mode
sudo aa-enforce /path/to/profile

# Disable profile
sudo aa-disable /path/to/profile

# Reload all profiles
sudo systemctl reload apparmor
```

### Fail2Ban

Fail2Ban protects against brute-force attacks.

```bash
# Install Fail2Ban
sudo apt install fail2ban

# Start and enable
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# Configuration
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo vim /etc/fail2ban/jail.local

# Check status
sudo fail2ban-client status

# Check specific jail
sudo fail2ban-client status sshd

# Unban IP
sudo fail2ban-client set sshd unbanip 192.168.1.100
```

### File Integrity Monitoring

```bash
# Install AIDE (Advanced Intrusion Detection Environment)
sudo apt install aide

# Initialize database
sudo aideinit

# Move database to production location
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Check for changes
sudo aide --check

# Update database after legitimate changes
sudo aide --update
```

### Security Auditing

```bash
# View failed login attempts
sudo grep "Failed password" /var/log/auth.log

# View successful logins
sudo grep "Accepted" /var/log/auth.log

# View sudo usage
sudo grep "sudo" /var/log/auth.log

# Check for users with UID 0 (root privileges)
awk -F: '$3 == 0 {print $1}' /etc/passwd

# List all sudo users
getent group sudo

# Check for world-writable files
sudo find / -xdev -type f -perm -0002 -ls 2>/dev/null

# Check for files with no owner
sudo find / -xdev -nouser -ls 2>/dev/null

# Check for SUID/SGID files
sudo find / -xdev \( -perm -4000 -o -perm -2000 \) -type f -ls 2>/dev/null
```

---

## System Monitoring and Performance

### System Information

```bash
# System information
uname -a                # Kernel and system info
lsb_release -a          # Ubuntu version
hostnamectl             # System hostname and OS info

# CPU information
lscpu
cat /proc/cpuinfo
nproc                   # Number of CPU cores

# Memory information
free -h
cat /proc/meminfo

# Hardware information
sudo lshw               # Detailed hardware info
sudo lshw -short        # Short summary
sudo dmidecode          # DMI/SMBIOS information

# PCI devices
lspci
lspci -v                # Verbose

# USB devices
lsusb
lsusb -v                # Verbose

# Kernel modules
lsmod
modinfo module-name
```

### Performance Monitoring

```bash
# CPU usage
top                     # Real-time system monitor
htop                    # Enhanced version
mpstat 1                # CPU statistics (install: sudo apt install sysstat)

# Memory usage
free -h
vmstat 1                # Virtual memory statistics

# Disk I/O
iostat -x 1             # Disk I/O statistics (install: sudo apt install sysstat)
iotop                   # Real-time disk I/O monitor (install: sudo apt install iotop)

# Network I/O
iftop                   # Network bandwidth monitor (install: sudo apt install iftop)
nethogs                 # Network usage per process (install: sudo apt install nethogs)
nload                   # Network traffic monitor (install: sudo apt install nload)

# System load
uptime
w
cat /proc/loadavg

# Comprehensive system monitoring
dstat                   # Versatile system stats (install: sudo apt install dstat)
glances                 # All-in-one monitor (install: sudo apt install glances)
```

### Log Management

```bash
# System logs location
/var/log/

# Important log files:
/var/log/syslog         # General system log
/var/log/auth.log       # Authentication log
/var/log/kern.log       # Kernel log
/var/log/dmesg          # Boot messages
/var/log/apt/           # APT package manager logs

# View logs
sudo less /var/log/syslog
sudo tail -f /var/log/syslog        # Follow log in real-time

# Systemd journal (journalctl)
journalctl                          # All logs
journalctl -f                       # Follow logs
journalctl -u service-name          # Logs for specific service
journalctl -b                       # Logs from current boot
journalctl -b -1                    # Logs from previous boot
journalctl --since "2024-01-01"     # Logs since date
journalctl --since "1 hour ago"     # Recent logs
journalctl -p err                   # Only errors
journalctl -k                       # Kernel messages

# Journal disk usage
journalctl --disk-usage

# Clean old logs
sudo journalctl --vacuum-time=7d    # Keep only 7 days
sudo journalctl --vacuum-size=1G    # Keep only 1GB

# Configure log rotation
sudo vim /etc/logrotate.conf
```

### System Resource Limits

```bash
# View current limits
ulimit -a

# Common limits:
ulimit -n               # Max open files
ulimit -u               # Max user processes
ulimit -m               # Max memory size

# Set limits (temporary)
ulimit -n 4096          # Set max open files to 4096

# Permanent limits configuration
sudo vim /etc/security/limits.conf

# Example entries:
# username soft nofile 4096
# username hard nofile 8192
# * soft nproc 2048
# * hard nproc 4096
```

---

## Shell and Scripting

### Bash Shell Basics

```bash
# Shell configuration files
~/.bashrc               # Interactive non-login shell
~/.bash_profile         # Login shell (sources .bashrc)
~/.profile              # Login shell (fallback)
~/.bash_logout          # Executed on logout
~/.bash_history         # Command history

# Reload configuration
source ~/.bashrc

# Environment variables
echo $HOME
echo $PATH
echo $USER

# Set environment variable (temporary)
export VARIABLE=value

# Set permanent environment variable
echo 'export VARIABLE=value' >> ~/.bashrc

# View all environment variables
env
printenv

# Command history
history
history 10              # Last 10 commands
!123                    # Execute command number 123
!!                      # Execute last command
!$                      # Last argument of previous command
!*                      # All arguments of previous command

# Search history
Ctrl+R                  # Reverse search
history | grep keyword
```

### Bash Scripting

#### Basic Script Structure

```bash
#!/bin/bash
# Script description

# Variables
NAME="John"
AGE=30

# Output
echo "Hello, $NAME"
echo "Age: $AGE"

# Command substitution
CURRENT_DATE=$(date +%Y-%m-%d)
USER_COUNT=$(who | wc -l)

# Conditionals
if [ "$AGE" -gt 18 ]; then
    echo "Adult"
elif [ "$AGE" -eq 18 ]; then
    echo "Just became adult"
else
    echo "Minor"
fi

# Test operators:
# -eq  equal
# -ne  not equal
# -gt  greater than
# -lt  less than
# -ge  greater than or equal
# -le  less than or equal

# String comparisons
if [ "$NAME" = "John" ]; then
    echo "Name is John"
fi

# File tests
if [ -f "/path/to/file" ]; then
    echo "File exists"
fi

# -f  file exists and is regular file
# -d  directory exists
# -e  file exists (any type)
# -r  file is readable
# -w  file is writable
# -x  file is executable

# Loops
# For loop
for i in {1..5}; do
    echo "Number: $i"
done

for file in *.txt; do
    echo "Processing: $file"
done

# While loop
counter=0
while [ $counter -lt 5 ]; do
    echo "Counter: $counter"
    ((counter++))
done

# Functions
greet() {
    local name=$1
    echo "Hello, $name"
}

greet "Alice"

# Arrays
fruits=("apple" "banana" "orange")
echo "${fruits[0]}"     # First element
echo "${fruits[@]}"     # All elements
echo "${#fruits[@]}"    # Array length

# Command-line arguments
# $0  script name
# $1  first argument
# $2  second argument
# $@  all arguments
# $#  number of arguments

# Error handling
set -e                  # Exit on error
set -u                  # Error on undefined variable
set -o pipefail         # Pipeline fails if any command fails

# Exit codes
exit 0                  # Success
exit 1                  # General error
```

#### Practical Script Examples

**System Backup Script:**
```bash
#!/bin/bash

BACKUP_DIR="/backup"
SOURCE_DIR="/home/user/documents"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${DATE}.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create backup
echo "Creating backup..."
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" "$SOURCE_DIR"

if [ $? -eq 0 ]; then
    echo "Backup successful: ${BACKUP_FILE}"
else
    echo "Backup failed!" >&2
    exit 1
fi

# Remove backups older than 7 days
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete
echo "Old backups removed"
```

**System Monitoring Script:**
```bash
#!/bin/bash

# Check CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU Usage: ${CPU_USAGE}%"

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.2f"), $3/$2 * 100}')
echo "Memory Usage: ${MEM_USAGE}%"

# Check disk usage
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
echo "Disk Usage: ${DISK_USAGE}%"

# Alert if disk usage > 80%
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "WARNING: Disk usage is above 80%!" | mail -s "Disk Alert" admin@example.com
fi
```

---

## Cloud and Server Administration

### Ubuntu Server Basics

```bash
# Server installation
# - Download Ubuntu Server ISO
# - Install minimal server (no GUI)
# - Configure network during installation
# - Set up SSH access

# Common server packages
sudo apt install \
    openssh-server \
    ufw \
    fail2ban \
    htop \
    vim \
    git \
    build-essential \
    curl \
    wget
```

### Web Server Setup

#### Apache

```bash
# Install Apache
sudo apt install apache2

# Manage Apache service
sudo systemctl start apache2
sudo systemctl enable apache2
sudo systemctl status apache2

# Configuration files
/etc/apache2/apache2.conf       # Main config
/etc/apache2/sites-available/   # Virtual host configs
/etc/apache2/sites-enabled/     # Enabled sites (symlinks)

# Enable/disable sites
sudo a2ensite site-name
sudo a2dissite site-name

# Enable/disable modules
sudo a2enmod rewrite
sudo a2dismod autoindex

# Test configuration
sudo apache2ctl configtest

# Reload after changes
sudo systemctl reload apache2

# Document root
/var/www/html/

# Logs
/var/log/apache2/access.log
/var/log/apache2/error.log
```

#### Nginx

```bash
# Install Nginx
sudo apt install nginx

# Manage Nginx service
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl status nginx

# Configuration files
/etc/nginx/nginx.conf           # Main config
/etc/nginx/sites-available/     # Site configs
/etc/nginx/sites-enabled/       # Enabled sites (symlinks)

# Enable site
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload configuration
sudo systemctl reload nginx

# Document root
/var/www/html/

# Logs
/var/log/nginx/access.log
/var/log/nginx/error.log
```

### Database Servers

#### MySQL/MariaDB

```bash
# Install MySQL
sudo apt install mysql-server

# Secure installation
sudo mysql_secure_installation

# Login to MySQL
sudo mysql

# Or with password
mysql -u root -p

# Create database and user
CREATE DATABASE mydb;
CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'localhost';
FLUSH PRIVILEGES;

# Backup database
mysqldump -u root -p mydb > backup.sql

# Restore database
mysql -u root -p mydb < backup.sql
```

#### PostgreSQL

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Switch to postgres user
sudo -i -u postgres

# Create database
createdb mydb

# Create user
createuser --interactive

# Login to PostgreSQL
psql

# Backup database
pg_dump mydb > backup.sql

# Restore database
psql mydb < backup.sql
```

### Docker on Ubuntu

```bash
# Install Docker
sudo apt update
sudo apt install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker run hello-world

# Docker commands
docker ps               # List running containers
docker ps -a            # List all containers
docker images           # List images
docker pull image:tag   # Pull image
docker run image        # Run container
docker stop container   # Stop container
docker rm container     # Remove container
docker rmi image        # Remove image

# Docker Compose
docker compose up -d
docker compose down
docker compose logs -f
```

---

## Troubleshooting

### Boot Issues

```bash
# View boot messages
dmesg
dmesg | less
journalctl -b

# Boot into recovery mode
# 1. Reboot and hold Shift during boot
# 2. Select "Advanced options"
# 3. Select recovery mode
# 4. Choose "root" for root shell access

# Check disk errors
sudo fsck /dev/sdX1     # Unmount first!

# Reinstall GRUB
sudo grub-install /dev/sdX
sudo update-grub
```

### Network Troubleshooting

```bash
# Check interface status
ip link show

# Restart networking
sudo systemctl restart systemd-networkd
sudo systemctl restart NetworkManager

# Check DNS resolution
nslookup google.com
dig google.com
cat /etc/resolv.conf

# Test connectivity
ping -c 4 8.8.8.8       # Test internet
ping -c 4 192.168.1.1   # Test gateway

# Trace route issues
traceroute google.com
mtr google.com

# Check listening ports
sudo ss -tuln
sudo netstat -tuln

# Check firewall
sudo ufw status verbose
sudo iptables -L -n
```

### Disk Issues

```bash
# Check disk space
df -h
du -sh /*

# Check inodes
df -i

# Find large files
sudo find / -type f -size +100M -exec ls -lh {} \;
sudo du -h /var | sort -rh | head -20

# Check disk health
sudo smartctl -a /dev/sda       # Install: sudo apt install smartmontools

# Fix filesystem errors
# Boot from live USB and run:
sudo fsck -f /dev/sdX1
```

### Performance Issues

```bash
# Check system load
uptime
top
htop

# Check memory
free -h
sudo swapon --show

# Check disk I/O
iostat -x 1
iotop

# Find memory hogs
ps aux --sort=-%mem | head
ps aux --sort=-%cpu | head

# Check zombie processes
ps aux | grep 'Z'
```

### Package Manager Issues

```bash
# Fix broken packages
sudo apt --fix-broken install
sudo dpkg --configure -a

# Clean package cache
sudo apt clean
sudo apt autoclean
sudo apt autoremove

# Fix repository issues
sudo apt update --fix-missing

# Reconfigure packages
sudo dpkg-reconfigure package-name

# Force reinstall package
sudo apt install --reinstall package-name

# Lock file issues
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock*
sudo dpkg --configure -a
sudo apt update
```

### Service Issues

```bash
# Check service status
sudo systemctl status service-name

# View service logs
sudo journalctl -u service-name
sudo journalctl -u service-name -f

# Reload systemd
sudo systemctl daemon-reload

# Reset failed services
sudo systemctl reset-failed
```

### Common Error Messages

**"No space left on device"**
```bash
df -h                   # Check disk space
df -i                   # Check inodes
du -sh /*              # Find large directories
```

**"Permission denied"**
```bash
ls -l file             # Check permissions
sudo chown user:group file
sudo chmod 644 file
```

**"Command not found"**
```bash
which command          # Find command location
echo $PATH             # Check PATH variable
sudo apt install package-name
```

**"Unable to locate package"**
```bash
sudo apt update        # Update package lists
sudo add-apt-repository ppa:...  # Add repository if needed
```

---

## Best Practices and Tips

### System Maintenance

1. **Regular Updates**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Clean Old Kernels**
   ```bash
   # Remove old kernels (keep 2 most recent)
   sudo apt autoremove --purge
   ```

3. **Monitor Disk Space**
   ```bash
   df -h
   du -sh /* | sort -rh
   ```

4. **Review Logs Regularly**
   ```bash
   sudo journalctl -p err -b
   sudo tail -100 /var/log/syslog
   ```

5. **Backup Important Data**
   - Use rsync, tar, or dedicated backup tools
   - Test restoration periodically
   - Store backups off-site

### Security Hardening

1. **Disable root login**
   ```bash
   sudo passwd -l root
   ```

2. **Use SSH keys instead of passwords**

3. **Keep minimal software installed**
   ```bash
   sudo apt list --installed | wc -l
   sudo apt autoremove
   ```

4. **Enable automatic security updates**

5. **Monitor failed login attempts**
   ```bash
   sudo grep "Failed password" /var/log/auth.log
   ```

6. **Use strong passwords**
   ```bash
   # Generate strong password
   openssl rand -base64 32
   ```

### Performance Optimization

1. **Disable unnecessary services**
   ```bash
   systemctl list-unit-files --type=service --state=enabled
   sudo systemctl disable service-name
   ```

2. **Adjust swappiness**
   ```bash
   # View current value
   cat /proc/sys/vm/swappiness

   # Set temporarily
   sudo sysctl vm.swappiness=10

   # Set permanently
   echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
   ```

3. **Use SSD optimization (TRIM)**
   ```bash
   # Check TRIM support
   sudo fstrim -v /

   # Enable weekly TRIM
   sudo systemctl enable fstrim.timer
   ```

### Useful Aliases

Add to `~/.bashrc`:

```bash
# System updates
alias update='sudo apt update && sudo apt upgrade -y'
alias cleanup='sudo apt autoremove -y && sudo apt autoclean'

# Directory navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ll='ls -lah'

# System monitoring
alias ports='sudo ss -tuln'
alias mem='free -h'
alias disk='df -h'

# Safety nets
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
```

---

## Quick Reference

### Essential Commands Cheatsheet

```bash
# System
sudo apt update && sudo apt upgrade     # Update system
sudo reboot                             # Reboot
sudo shutdown -h now                    # Shutdown
hostnamectl                             # System info

# Files
ls -lah                                 # List files
cd /path                                # Change directory
cp source dest                          # Copy
mv source dest                          # Move/rename
rm file                                 # Remove
mkdir dir                               # Create directory
chmod 755 file                          # Change permissions
chown user:group file                   # Change owner

# Processes
ps aux                                  # List processes
top                                     # Monitor processes
kill PID                                # Kill process
systemctl status service                # Service status

# Network
ip a                                    # Show IP addresses
ping host                               # Test connectivity
ssh user@host                           # SSH connect
sudo ufw allow 22                       # Allow SSH through firewall

# Disk
df -h                                   # Disk space
du -sh dir                              # Directory size
mount /dev/sdX /mnt                     # Mount disk

# Logs
journalctl -f                           # Follow system log
tail -f /var/log/syslog                 # Follow syslog
```

---

**Last Updated**: January 2025
**Version**: Ubuntu 24.04 LTS (Noble Numbat)
