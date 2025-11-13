# wpa_supplicant

A comprehensive guide to wpa_supplicant, the IEEE 802.11 authentication daemon for WiFi client connectivity on Linux.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration Files](#configuration-files)
- [Basic Usage](#basic-usage)
- [Network Configuration](#network-configuration)
- [Command-Line Interface](#command-line-interface)
- [wpa_cli Interactive Mode](#wpa_cli-interactive-mode)
- [Advanced Configuration](#advanced-configuration)
- [Security Modes](#security-modes)
- [Enterprise WiFi (802.1X)](#enterprise-wifi-8021x)
- [P2P WiFi Direct](#p2p-wifi-direct)
- [Troubleshooting](#troubleshooting)
- [Integration with systemd](#integration-with-systemd)
- [Best Practices](#best-practices)

---

## Overview

**wpa_supplicant** is a WPA/WPA2/WPA3 supplicant for Linux and other UNIX-like operating systems. It handles WiFi authentication and association for client stations.

### Key Features

- WPA/WPA2/WPA3-Personal (PSK)
- WPA/WPA2/WPA3-Enterprise (802.1X/EAP)
- WEP (deprecated, for legacy networks)
- Hotspot 2.0 (Passpoint)
- WiFi Protected Setup (WPS)
- WiFi Direct (P2P)
- Automatic network selection
- Dynamic reconfiguration via control interface

### Architecture

```
┌─────────────────────────────────────┐
│        User Space                   │
│                                     │
│  ┌──────────┐      ┌──────────┐   │
│  │  wpa_cli │      │ NetworkMgr│   │
│  └────┬─────┘      └────┬──────┘   │
│       │                 │           │
│       └────┬────────────┘           │
│            │ Control socket         │
│       ┌────▼──────────────┐         │
│       │  wpa_supplicant   │         │
│       └────┬──────────────┘         │
│            │ nl80211/WEXT           │
└────────────┼─────────────────────────┘
             │
┌────────────▼─────────────────────────┐
│        Kernel Space                  │
│  ┌──────────────────────────┐        │
│  │  cfg80211 / mac80211     │        │
│  └──────────┬───────────────┘        │
│             │                        │
│  ┌──────────▼───────────────┐        │
│  │    WiFi Driver           │        │
│  └──────────┬───────────────┘        │
└─────────────┼──────────────────────────┘
              │
       ┌──────▼──────┐
       │ WiFi Hardware│
       └─────────────┘
```

---

## Installation

### Debian/Ubuntu

```bash
sudo apt-get update
sudo apt-get install wpasupplicant

# Verify installation
wpa_supplicant -v
```

### Fedora/RHEL/CentOS

```bash
sudo dnf install wpa_supplicant

# Or for older systems
sudo yum install wpa_supplicant
```

### Arch Linux

```bash
sudo pacman -S wpa_supplicant
```

### Build from Source

```bash
# Download
git clone git://w1.fi/srv/git/hostap.git
cd hostap/wpa_supplicant

# Configure
cp defconfig .config
# Edit .config to enable features

# Build
make

# Install
sudo make install
```

---

## Configuration Files

### Main Configuration File

**Location:** `/etc/wpa_supplicant/wpa_supplicant.conf`

**Basic structure:**
```conf
# Global settings
ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=netdev
update_config=1
country=US

# Network configurations
network={
    ssid="MyNetwork"
    psk="password123"
}
```

### File Permissions

```bash
# Secure the configuration file
sudo chmod 600 /etc/wpa_supplicant/wpa_supplicant.conf
sudo chown root:root /etc/wpa_supplicant/wpa_supplicant.conf
```

### Global Parameters

```conf
# Control interface for wpa_cli
ctrl_interface=/var/run/wpa_supplicant

# Group that can access control interface
ctrl_interface_group=netdev

# Allow wpa_supplicant to update configuration
update_config=1

# Country code (affects regulatory domain)
country=US

# AP scanning mode
# 0 = driver takes care of scanning
# 1 = wpa_supplicant controls scanning (default)
# 2 = like 1, but use security policy
ap_scan=1

# Fast reauth for 802.1X
fast_reauth=1

# Enable P2P support
p2p_disabled=0
```

---

## Basic Usage

### Starting wpa_supplicant

```bash
# Basic usage
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf

# Options:
# -B: Run in background (daemon mode)
# -i: Network interface
# -c: Configuration file
# -D: Driver (nl80211, wext, etc.) - usually auto-detected
# -d: Enable debug output
# -dd: More verbose debug
```

### Starting with Debug Output

```bash
# Foreground with debug
sudo wpa_supplicant -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf -d

# Even more verbose
sudo wpa_supplicant -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf -dd
```

### Stopping wpa_supplicant

```bash
# Find process
ps aux | grep wpa_supplicant

# Kill process
sudo killall wpa_supplicant

# Or using systemd
sudo systemctl stop wpa_supplicant@wlan0
```

### Manual Connection Workflow

```bash
# 1. Bring interface up
sudo ip link set wlan0 up

# 2. Start wpa_supplicant
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf

# 3. Wait for connection (check with wpa_cli)
wpa_cli -i wlan0 status

# 4. Get IP address
sudo dhclient wlan0
# Or
sudo dhcpcd wlan0
```

---

## Network Configuration

### WPA/WPA2-Personal (PSK)

**ASCII passphrase:**
```conf
network={
    ssid="MyWiFi"
    psk="MyPassword123"
    key_mgmt=WPA-PSK
    priority=1
}
```

**Pre-computed PSK (more secure):**
```bash
# Generate PSK hash
wpa_passphrase "MyWiFi" "MyPassword123"

# Output:
network={
    ssid="MyWiFi"
    #psk="MyPassword123"
    psk=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
}
```

**In configuration file:**
```conf
network={
    ssid="MyWiFi"
    psk=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    key_mgmt=WPA-PSK
}
```

### WPA3-Personal (SAE)

```conf
network={
    ssid="MyWiFi-WPA3"
    psk="MyPassword123"
    key_mgmt=SAE
    ieee80211w=2  # Required for WPA3 (PMF)
}
```

### Open Network (No Security)

```conf
network={
    ssid="OpenWiFi"
    key_mgmt=NONE
}
```

### Hidden Network

```conf
network={
    ssid="HiddenSSID"
    scan_ssid=1  # Enable active scanning
    psk="password"
    key_mgmt=WPA-PSK
}
```

### WEP (Deprecated)

```conf
network={
    ssid="OldNetwork"
    key_mgmt=NONE
    wep_key0="1234567890"
    wep_tx_keyidx=0
}
```

### Multiple Networks with Priority

```conf
# Home network - highest priority
network={
    ssid="HomeWiFi"
    psk="homepassword"
    priority=10
}

# Work network
network={
    ssid="WorkWiFi"
    psk="workpassword"
    priority=5
}

# Coffee shop - lowest priority
network={
    ssid="CoffeeShop"
    key_mgmt=NONE
    priority=1
}
```

### BSSID-Specific Configuration

```conf
# Connect only to specific AP
network={
    ssid="MyWiFi"
    bssid=00:11:22:33:44:55
    psk="password"
}
```

---

## Command-Line Interface

### wpa_cli - Control Interface

**Basic commands:**
```bash
# Show status
wpa_cli -i wlan0 status

# Scan for networks
wpa_cli -i wlan0 scan
wpa_cli -i wlan0 scan_results

# List configured networks
wpa_cli -i wlan0 list_networks

# Add network
wpa_cli -i wlan0 add_network
# Returns: 0 (network ID)

# Set network parameters
wpa_cli -i wlan0 set_network 0 ssid '"MyWiFi"'
wpa_cli -i wlan0 set_network 0 psk '"password"'

# Enable network
wpa_cli -i wlan0 enable_network 0

# Select network
wpa_cli -i wlan0 select_network 0

# Save configuration
wpa_cli -i wlan0 save_config

# Remove network
wpa_cli -i wlan0 remove_network 0

# Disconnect
wpa_cli -i wlan0 disconnect

# Reconnect
wpa_cli -i wlan0 reconnect

# Reassociate
wpa_cli -i wlan0 reassociate
```

### Quick Connection

```bash
# One-liner to connect
wpa_cli -i wlan0 <<EOF
add_network
set_network 0 ssid "MyWiFi"
set_network 0 psk "password"
enable_network 0
save_config
quit
EOF
```

---

## wpa_cli Interactive Mode

### Starting Interactive Mode

```bash
wpa_cli -i wlan0
```

**Interactive session:**
```
wpa_cli v2.9
Copyright (c) 2004-2019, Jouni Malinen <j@w1.fi> and contributors

Interactive mode

> status
bssid=00:11:22:33:44:55
freq=2437
ssid=MyWiFi
id=0
mode=station
pairwise_cipher=CCMP
group_cipher=CCMP
key_mgmt=WPA2-PSK
wpa_state=COMPLETED
ip_address=192.168.1.100
address=aa:bb:cc:dd:ee:ff

> scan
OK
> scan_results
bssid / frequency / signal level / flags / ssid
00:11:22:33:44:55       2437    -45     [WPA2-PSK-CCMP][ESS]    MyWiFi
aa:bb:cc:dd:ee:ff       2462    -67     [WPA2-PSK-CCMP][ESS]    NeighborWiFi

> quit
```

### Common Interactive Commands

```
status                  - Show connection status
scan                    - Trigger network scan
scan_results            - Show scan results
list_networks           - List configured networks
select_network <id>     - Select network
enable_network <id>     - Enable network
disable_network <id>    - Disable network
remove_network <id>     - Remove network
add_network             - Add new network
set_network <id> <var> <value> - Set network parameter
save_config             - Save configuration
disconnect              - Disconnect from AP
reconnect               - Reconnect to AP
reassociate             - Force reassociation
terminate               - Terminate wpa_supplicant
quit                    - Exit wpa_cli
```

---

## Advanced Configuration

### Band Selection (2.4 GHz vs 5 GHz)

```conf
network={
    ssid="DualBandWiFi"
    psk="password"
    # Prefer 5 GHz
    freq_list=5180 5200 5220 5240 5260 5280 5300 5320
}
```

### Power Saving

```conf
# Global setting
# 0 = CAM (Constantly Awake Mode)
# 1 = PS mode (default)
# 2 = PS mode with max power saving
power_save=1
```

### Roaming

```conf
network={
    ssid="EnterpriseWiFi"
    psk="password"
    # Fast roaming (802.11r)
    key_mgmt=FT-PSK
    # Proactive key caching
    proactive_key_caching=1
    # BSS transition management
    bss_transition=1
}
```

### MAC Address Randomization

```conf
# Per-network MAC randomization
network={
    ssid="PublicWiFi"
    key_mgmt=NONE
    mac_addr=1  # Random MAC per network
}

# Global setting
mac_addr=1
# 0 = Use permanent MAC
# 1 = Random MAC per network
# 2 = Random MAC per SSID
```

### IPv6

```conf
# Disable IPv6 in wpa_supplicant
network={
    ssid="MyWiFi"
    psk="password"
    disable_ipv6=1
}
```

---

## Security Modes

### WPA2-Enterprise (EAP-PEAP/MSCHAPv2)

```conf
network={
    ssid="CorpWiFi"
    key_mgmt=WPA-EAP
    eap=PEAP
    identity="username@domain.com"
    password="userpassword"
    phase2="auth=MSCHAPV2"
    # Certificate verification
    ca_cert="/etc/ssl/certs/ca-bundle.crt"
    # Or skip verification (insecure!)
    # ca_cert="/etc/ssl/certs/ca-certificates.crt"
}
```

### WPA2-Enterprise (EAP-TLS with Certificates)

```conf
network={
    ssid="SecureCorpWiFi"
    key_mgmt=WPA-EAP
    eap=TLS
    identity="user@company.com"
    # Client certificate
    client_cert="/etc/wpa_supplicant/client.crt"
    # Private key
    private_key="/etc/wpa_supplicant/client.key"
    # Private key password
    private_key_passwd="keypassword"
    # CA certificate
    ca_cert="/etc/wpa_supplicant/ca.crt"
}
```

### WPA2-Enterprise (EAP-TTLS/PAP)

```conf
network={
    ssid="UniversityWiFi"
    key_mgmt=WPA-EAP
    eap=TTLS
    identity="student@university.edu"
    password="studentpass"
    phase2="auth=PAP"
    ca_cert="/etc/ssl/certs/ca-bundle.crt"
}
```

### Eduroam Configuration

```conf
network={
    ssid="eduroam"
    key_mgmt=WPA-EAP
    eap=PEAP
    identity="username@institution.edu"
    password="password"
    phase2="auth=MSCHAPV2"
    ca_cert="/etc/ssl/certs/ca-certificates.crt"
}
```

---

## Enterprise WiFi (802.1X)

### Certificate Management

```bash
# Download CA certificate
wget https://your-ca.com/ca.crt -O /etc/wpa_supplicant/ca.crt

# Set permissions
sudo chmod 600 /etc/wpa_supplicant/ca.crt

# Convert certificate format if needed
openssl x509 -inform DER -in ca.der -out ca.pem
```

### Anonymous Identity (Privacy)

```conf
network={
    ssid="CorpWiFi"
    key_mgmt=WPA-EAP
    eap=PEAP
    # Anonymous outer identity
    anonymous_identity="anonymous@company.com"
    # Real identity (inner)
    identity="realuser@company.com"
    password="password"
    phase2="auth=MSCHAPV2"
    ca_cert="/etc/wpa_supplicant/ca.crt"
}
```

### Domain Suffix Matching

```conf
network={
    ssid="SecureWiFi"
    key_mgmt=WPA-EAP
    eap=PEAP
    identity="user@company.com"
    password="password"
    phase2="auth=MSCHAPV2"
    # Verify server domain
    domain_suffix_match="radius.company.com"
    ca_cert="/etc/wpa_supplicant/ca.crt"
}
```

---

## P2P WiFi Direct

### Enable WiFi Direct

```conf
# Global setting
ctrl_interface=/var/run/wpa_supplicant
p2p_disabled=0
device_name=MyDevice
device_type=1-0050F204-1
```

### P2P Commands

```bash
# Start P2P mode
wpa_cli -i wlan0 p2p_find

# Stop search
wpa_cli -i wlan0 p2p_stop_find

# Connect to peer
wpa_cli -i wlan0 p2p_connect <peer_mac> pbc

# Group formation
wpa_cli -i wlan0 p2p_group_add

# Show peers
wpa_cli -i wlan0 p2p_peers
```

---

## Troubleshooting

### Check Status

```bash
# Interface status
ip link show wlan0

# wpa_supplicant status
wpa_cli -i wlan0 status

# Connection state
wpa_cli -i wlan0 status | grep wpa_state
# COMPLETED = connected
# SCANNING = scanning for networks
# ASSOCIATING = connecting
# DISCONNECTED = not connected
```

### Debug Logging

```bash
# Run in foreground with debug
sudo killall wpa_supplicant
sudo wpa_supplicant -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf -dd

# Check system logs
sudo journalctl -u wpa_supplicant@wlan0 -f

# dmesg for driver issues
dmesg | grep -i wifi
dmesg | grep -i wlan
```

### Common Issues

**Authentication failure:**
```bash
# Check password
wpa_passphrase "SSID" "password"

# Verify security mode
wpa_cli -i wlan0 scan_results
# Look for [WPA2-PSK-CCMP], [WPA3-SAE], etc.

# Check logs
sudo journalctl -u wpa_supplicant@wlan0 | grep -i "auth\|fail"
```

**Cannot scan networks:**
```bash
# Check if interface is up
sudo ip link set wlan0 up

# Check rfkill
rfkill list
sudo rfkill unblock wifi

# Manual scan
sudo iw dev wlan0 scan | grep SSID
```

**Frequent disconnections:**
```bash
# Check signal strength
watch -n 1 'iw dev wlan0 link'

# Disable power management
sudo iwconfig wlan0 power off

# Check logs for errors
sudo journalctl -u wpa_supplicant@wlan0 --since "10 minutes ago"
```

**Driver issues:**
```bash
# Check driver
lspci -k | grep -A 3 -i network
# Or for USB
lsusb
dmesg | grep -i firmware

# Reload driver
sudo modprobe -r <driver_name>
sudo modprobe <driver_name>
```

---

## Integration with systemd

### systemd Service

**Per-interface service:**
```bash
# Start service
sudo systemctl start wpa_supplicant@wlan0

# Enable on boot
sudo systemctl enable wpa_supplicant@wlan0

# Status
sudo systemctl status wpa_supplicant@wlan0

# Restart
sudo systemctl restart wpa_supplicant@wlan0
```

**Service file:** `/lib/systemd/system/wpa_supplicant@.service`
```ini
[Unit]
Description=WPA supplicant daemon (interface-specific version)
Requires=sys-subsystem-net-devices-%i.device
After=sys-subsystem-net-devices-%i.device
Before=network.target
Wants=network.target

[Service]
Type=simple
ExecStart=/sbin/wpa_supplicant -c/etc/wpa_supplicant/wpa_supplicant-%I.conf -i%I

[Install]
WantedBy=multi-user.target
```

### networkd Integration

**`/etc/systemd/network/25-wireless.network`:**
```ini
[Match]
Name=wlan0

[Network]
DHCP=yes
```

**Start services:**
```bash
sudo systemctl enable systemd-networkd
sudo systemctl enable wpa_supplicant@wlan0
sudo systemctl start systemd-networkd
sudo systemctl start wpa_supplicant@wlan0
```

---

## Best Practices

### Security

1. **Use encrypted PSK:**
```bash
# Generate PSK hash instead of plaintext
wpa_passphrase "SSID" "password" | sudo tee -a /etc/wpa_supplicant/wpa_supplicant.conf
```

2. **Secure configuration file:**
```bash
sudo chmod 600 /etc/wpa_supplicant/wpa_supplicant.conf
```

3. **Use WPA3 when available:**
```conf
network={
    ssid="MyWiFi"
    psk="password"
    key_mgmt=SAE WPA-PSK  # Try WPA3, fall back to WPA2
    ieee80211w=1  # Optional PMF
}
```

4. **Verify certificates for Enterprise:**
```conf
network={
    ssid="CorpWiFi"
    key_mgmt=WPA-EAP
    ca_cert="/path/to/ca.crt"
    domain_suffix_match="radius.company.com"
}
```

### Performance

1. **Disable unnecessary features:**
```conf
# Disable P2P if not needed
p2p_disabled=1

# Disable WPS
wps_disabled=1
```

2. **Optimize power saving:**
```conf
# For performance (disable power save)
power_save=0

# For battery (enable power save)
power_save=2
```

3. **Fast roaming:**
```conf
network={
    ssid="EnterpriseWiFi"
    key_mgmt=FT-PSK
    proactive_key_caching=1
}
```

### Reliability

1. **Network priority:**
```conf
# Higher priority = preferred
network={
    ssid="PrimaryWiFi"
    priority=10
}
network={
    ssid="BackupWiFi"
    priority=5
}
```

2. **Automatic reconnection:**
```bash
# systemd handles this automatically
sudo systemctl enable wpa_supplicant@wlan0
```

3. **Monitoring:**
```bash
# Watch connection status
watch -n 2 'wpa_cli -i wlan0 status | grep -E "wpa_state|ssid|ip_address"'
```

---

## Summary

**wpa_supplicant** is the standard WiFi client for Linux:

**Basic workflow:**
1. Configure networks in `/etc/wpa_supplicant/wpa_supplicant.conf`
2. Start: `sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf`
3. Manage: `wpa_cli -i wlan0 <command>`
4. Get IP: `sudo dhclient wlan0`

**Key commands:**
- `wpa_passphrase`: Generate PSK hash
- `wpa_supplicant`: Main daemon
- `wpa_cli`: Control interface
- `systemctl`: Manage service

**Common tasks:**
- Connect to WPA2: Set `ssid` and `psk`
- Enterprise WiFi: Configure EAP method
- Scan networks: `wpa_cli scan && wpa_cli scan_results`
- Debug: Run with `-dd` flag

**Resources:**
- [wpa_supplicant Documentation](https://w1.fi/wpa_supplicant/)
- [ArchWiki: wpa_supplicant](https://wiki.archlinux.org/title/Wpa_supplicant)
- [Ubuntu WiFi Guide](https://help.ubuntu.com/community/WifiDocs)
