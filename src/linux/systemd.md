# systemd

systemd is a system and service manager for Linux operating systems. It provides aggressive parallelization capabilities, uses socket and D-Bus activation for starting services, offers on-demand starting of daemons, and maintains process tracking using Linux control groups.

## Overview

systemd replaces the traditional SysV init system and provides a more modern approach to system initialization and service management.

**Key Features:**
- Parallel service startup
- Socket and D-Bus activation
- On-demand service starting
- Process supervision
- Mount and automount point management
- Snapshot support
- System state snapshots
- Logging with journald

## Basic Concepts

**Units**: Resources that systemd manages
- **Service units** (.service): System services
- **Socket units** (.socket): IPC or network sockets
- **Target units** (.target): Group of units (like runlevels)
- **Mount units** (.mount): Mount points
- **Timer units** (.timer): Scheduled tasks
- **Device units** (.device): Device files
- **Path units** (.path): File/directory monitoring

## Service Management

### systemctl Commands

```bash
# Service control
sudo systemctl start service_name
sudo systemctl stop service_name
sudo systemctl restart service_name
sudo systemctl reload service_name         # Reload config without restart
sudo systemctl reload-or-restart service_name

# Enable/disable services (start at boot)
sudo systemctl enable service_name
sudo systemctl disable service_name
sudo systemctl enable --now service_name   # Enable and start

# Check service status
systemctl status service_name
systemctl is-active service_name
systemctl is-enabled service_name
systemctl is-failed service_name

# List services
systemctl list-units --type=service
systemctl list-units --type=service --state=running
systemctl list-units --type=service --state=failed
systemctl list-unit-files --type=service

# Show service configuration
systemctl cat service_name
systemctl show service_name

# Service dependencies
systemctl list-dependencies service_name
```

### Service Examples

```bash
# Common services
sudo systemctl status nginx
sudo systemctl restart sshd
sudo systemctl enable docker
sudo systemctl start postgresql

# Check all failed services
systemctl --failed

# Mask service (prevent from being started)
sudo systemctl mask service_name
sudo systemctl unmask service_name
```

## Creating Service Units

### Basic Service File

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=myapp
Group=myapp
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

### Service Types

```ini
# Type=simple (default)
[Service]
Type=simple
ExecStart=/usr/bin/myapp

# Type=forking (daemon that forks)
[Service]
Type=forking
PIDFile=/var/run/myapp.pid
ExecStart=/usr/bin/myapp --daemon

# Type=oneshot (runs once and exits)
[Service]
Type=oneshot
ExecStart=/usr/bin/backup-script.sh
RemainAfterExit=yes

# Type=notify (sends notification when ready)
[Service]
Type=notify
ExecStart=/usr/bin/myapp
NotifyAccess=main

# Type=dbus (acquires D-Bus name)
[Service]
Type=dbus
BusName=org.example.myapp
ExecStart=/usr/bin/myapp

# Type=idle (delays until all jobs finished)
[Service]
Type=idle
ExecStart=/usr/bin/myapp
```

### Advanced Service Configuration

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Web Application
Documentation=https://example.com/docs
After=network-online.target postgresql.service
Wants=network-online.target
Requires=postgresql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/myapp

# Environment
Environment="NODE_ENV=production"
Environment="PORT=3000"
EnvironmentFile=/etc/myapp/config

# Execution
ExecStartPre=/usr/bin/myapp-check-config
ExecStart=/usr/bin/node /opt/myapp/server.js
ExecReload=/bin/kill -HUP $MAINPID
ExecStop=/bin/kill -TERM $MAINPID

# Restart policy
Restart=on-failure
RestartSec=5s
StartLimitInterval=10min
StartLimitBurst=5

# Security
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/myapp
ReadWritePaths=/var/log/myapp

# Resource limits
LimitNOFILE=65536
MemoryLimit=1G
CPUQuota=200%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

[Install]
WantedBy=multi-user.target
```

### Service Management Workflow

```bash
# Create service file
sudo vim /etc/systemd/system/myapp.service

# Reload systemd configuration
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable --now myapp

# Check status
systemctl status myapp

# View logs
journalctl -u myapp -f

# Edit service (creates override)
sudo systemctl edit myapp

# Edit full service file
sudo systemctl edit --full myapp
```

## Timers (Cron Alternative)

### Timer Unit

```ini
# /etc/systemd/system/backup.timer
[Unit]
Description=Daily Backup Timer
Requires=backup.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 02:00:00
Persistent=true
Unit=backup.service

[Install]
WantedBy=timers.target
```

### Corresponding Service

```ini
# /etc/systemd/system/backup.service
[Unit]
Description=Backup Service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=backup
```

### Timer Management

```bash
# Enable and start timer
sudo systemctl enable --now backup.timer

# List timers
systemctl list-timers
systemctl list-timers --all

# Check timer status
systemctl status backup.timer

# View next run time
systemctl list-timers backup.timer

# Manual trigger
sudo systemctl start backup.service
```

### Timer Examples

```ini
# Every 5 minutes
OnCalendar=*:0/5

# Every hour
OnCalendar=hourly

# Every day at 3:00 AM
OnCalendar=*-*-* 03:00:00

# Every Monday at 9:00 AM
OnCalendar=Mon *-*-* 09:00:00

# First day of month
OnCalendar=*-*-01 00:00:00

# Relative to boot
OnBootSec=15min
OnUnitActiveSec=1h
```

## journalctl (Logging)

### Viewing Logs

```bash
# View all logs
journalctl

# Follow logs (like tail -f)
journalctl -f

# Recent logs
journalctl -n 50               # Last 50 lines
journalctl -n 100 --no-pager

# Service-specific logs
journalctl -u nginx
journalctl -u nginx -f
journalctl -u nginx --since today

# Multiple services
journalctl -u nginx -u postgresql

# Time-based filtering
journalctl --since "2024-01-01"
journalctl --since "2024-01-01 10:00" --until "2024-01-01 11:00"
journalctl --since "1 hour ago"
journalctl --since yesterday
journalctl --since "10 min ago"

# Priority filtering
journalctl -p err              # Errors only
journalctl -p warning          # Warnings and above
journalctl -p 0..3             # Emergency to error

# Kernel messages
journalctl -k
journalctl -k -b              # Current boot

# Boot-specific logs
journalctl -b                  # Current boot
journalctl -b -1               # Previous boot
journalctl --list-boots        # List all boots

# Specific process
journalctl _PID=1234

# Output formats
journalctl -o json             # JSON format
journalctl -o json-pretty
journalctl -o verbose
journalctl -o cat              # Just the message

# Disk usage
journalctl --disk-usage

# Verify integrity
journalctl --verify
```

### Journal Management

```bash
# Clean old logs
sudo journalctl --vacuum-time=7d    # Keep last 7 days
sudo journalctl --vacuum-size=500M  # Keep max 500MB
sudo journalctl --vacuum-files=5    # Keep max 5 files

# Rotate journals
sudo systemctl kill --signal=SIGUSR2 systemd-journald

# Configure retention
# /etc/systemd/journald.conf
[Journal]
SystemMaxUse=500M
SystemMaxFileSize=100M
SystemMaxFiles=5
RuntimeMaxUse=100M
MaxRetentionSec=7day
```

## Targets (Runlevels)

### Common Targets

```bash
# List targets
systemctl list-units --type=target

# Current target
systemctl get-default

# Change default target
sudo systemctl set-default multi-user.target
sudo systemctl set-default graphical.target

# Switch target
sudo systemctl isolate multi-user.target
sudo systemctl isolate rescue.target

# Common targets
# poweroff.target (runlevel 0)
# rescue.target (runlevel 1)
# multi-user.target (runlevel 3)
# graphical.target (runlevel 5)
# reboot.target (runlevel 6)
```

## System Management

### System Control

```bash
# Reboot/shutdown
sudo systemctl reboot
sudo systemctl poweroff
sudo systemctl halt
sudo systemctl suspend
sudo systemctl hibernate
sudo systemctl hybrid-sleep

# System state
systemctl is-system-running

# Reload systemd configuration
sudo systemctl daemon-reload

# Reexecute systemd
sudo systemctl daemon-reexec

# Show system boot time
systemd-analyze
systemd-analyze blame           # Show service startup times
systemd-analyze critical-chain  # Show critical startup chain
systemd-analyze plot > boot.svg # Generate SVG timeline

# List all units
systemctl list-units
systemctl list-units --all
systemctl list-unit-files

# Check configuration
sudo systemd-analyze verify /etc/systemd/system/myapp.service
```

### Socket Activation

```ini
# /etc/systemd/system/myapp.socket
[Unit]
Description=My App Socket

[Socket]
ListenStream=8080
Accept=no

[Install]
WantedBy=sockets.target

# /etc/systemd/system/myapp.service
[Unit]
Description=My App Service
Requires=myapp.socket

[Service]
ExecStart=/usr/bin/myapp
StandardInput=socket
```

## Path Units (File Monitoring)

```ini
# /etc/systemd/system/watch-config.path
[Unit]
Description=Watch Config Directory

[Path]
PathModified=/etc/myapp
Unit=process-config.service

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/process-config.service
[Unit]
Description=Process Config Changes

[Service]
Type=oneshot
ExecStart=/usr/local/bin/reload-config.sh
```

## User Services

```bash
# User service directory
~/.config/systemd/user/

# User commands (no sudo)
systemctl --user start myservice
systemctl --user enable myservice
systemctl --user status myservice

# User timers
systemctl --user list-timers

# Enable lingering (services run without login)
loginctl enable-linger username

# User journal
journalctl --user
journalctl --user -u myservice
```

### Example User Service

```ini
# ~/.config/systemd/user/myapp.service
[Unit]
Description=My User Application

[Service]
ExecStart=%h/bin/myapp
Restart=on-failure

[Install]
WantedBy=default.target
```

## Security Features

### Service Hardening

```ini
[Service]
# User/Group isolation
User=myapp
Group=myapp
DynamicUser=yes                # Create temporary user

# Filesystem restrictions
ProtectSystem=strict           # Read-only /usr, /boot, /efi
ProtectHome=true               # Inaccessible /home
PrivateTmp=true                # Private /tmp
ReadWritePaths=/var/lib/myapp  # Writable paths
ReadOnlyPaths=/etc/myapp
InaccessiblePaths=/root

# Namespace isolation
PrivateDevices=yes             # Private /dev
PrivateNetwork=yes             # Private network namespace
PrivateUsers=yes               # User namespace

# Capabilities
NoNewPrivileges=yes            # Prevent privilege escalation
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

# System calls
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources
SystemCallErrorNumber=EPERM

# Misc restrictions
RestrictAddressFamilies=AF_INET AF_INET6
RestrictNamespaces=yes
RestrictRealtime=yes
LockPersonality=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
MemoryDenyWriteExecute=yes
```

## Troubleshooting

### Common Issues

```bash
# Service won't start
systemctl status service_name
journalctl -u service_name -n 50
journalctl -xe

# Check service configuration
systemd-analyze verify /etc/systemd/system/myapp.service

# Dependency issues
systemctl list-dependencies service_name
systemctl list-dependencies --reverse service_name

# Stuck service
sudo systemctl kill service_name
sudo systemctl kill -s SIGKILL service_name

# Reset failed state
sudo systemctl reset-failed service_name
sudo systemctl reset-failed

# Show why service failed
systemctl status service_name --no-pager --full

# Debug mode
sudo SYSTEMD_LOG_LEVEL=debug systemctl start service_name

# Emergency shell
# Add to kernel command line: systemd.unit=emergency.target
```

### Debugging Services

```ini
# Add debug output to service
[Service]
Environment="DEBUG=true"
StandardOutput=journal+console
StandardError=journal+console

# Increase log level
LogLevel=debug

# Show environment
systemctl show-environment
systemctl show service_name
```

## Best Practices

```ini
# 1. Use After= and Wants= for dependencies
[Unit]
After=network-online.target
Wants=network-online.target

# 2. Set restart policy
[Service]
Restart=on-failure
RestartSec=5s
StartLimitInterval=10min
StartLimitBurst=5

# 3. Use specific user
[Service]
User=myapp
Group=myapp

# 4. Set working directory
[Service]
WorkingDirectory=/opt/myapp

# 5. Use environment files
[Service]
EnvironmentFile=/etc/myapp/config

# 6. Add security restrictions
[Service]
ProtectSystem=strict
PrivateTmp=true
NoNewPrivileges=true

# 7. Proper logging
[Service]
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myapp

# 8. Resource limits
[Service]
LimitNOFILE=65536
MemoryMax=1G

# 9. Use timers instead of cron
# Create .timer and .service files

# 10. Test configuration
sudo systemd-analyze verify myapp.service
```

## Quick Reference

### Service Management
| Command | Description |
|---------|-------------|
| `systemctl start` | Start service |
| `systemctl stop` | Stop service |
| `systemctl restart` | Restart service |
| `systemctl reload` | Reload configuration |
| `systemctl enable` | Enable at boot |
| `systemctl disable` | Disable at boot |
| `systemctl status` | Show service status |
| `systemctl is-active` | Check if active |
| `systemctl is-enabled` | Check if enabled |

### Journalctl
| Command | Description |
|---------|-------------|
| `journalctl -u SERVICE` | Service logs |
| `journalctl -f` | Follow logs |
| `journalctl -b` | Current boot logs |
| `journalctl --since` | Time-filtered logs |
| `journalctl -p err` | Error priority logs |
| `journalctl -k` | Kernel messages |

systemd provides a powerful, modern init system with extensive features for service management, logging, and system administration, making it the standard for most Linux distributions.
