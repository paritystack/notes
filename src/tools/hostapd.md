# hostapd

A comprehensive guide to hostapd, the IEEE 802.11 access point and authentication server for creating WiFi access points on Linux.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Basic Configuration](#basic-configuration)
- [Running hostapd](#running-hostapd)
- [Security Configurations](#security-configurations)
- [Advanced Features](#advanced-features)
- [Bridge Mode](#bridge-mode)
- [VLAN Support](#vlan-support)
- [RADIUS Authentication](#radius-authentication)
- [802.11n/ac/ax Configuration](#80211nacax-configuration)
- [Monitoring and Management](#monitoring-and-management)
- [Troubleshooting](#troubleshooting)
- [Integration with systemd](#integration-with-systemd)
- [Best Practices](#best-practices)

---

## Overview

**hostapd** (host access point daemon) is a user-space daemon for access point and authentication servers. It implements IEEE 802.11 access point management, IEEE 802.1X/WPA/WPA2/WPA3/EAP authenticators, and RADIUS authentication server.

### Key Features

- WiFi Access Point (AP) mode
- WPA/WPA2/WPA3-Personal and Enterprise
- Multiple SSIDs (up to 8 per radio)
- VLAN tagging
- 802.11n/ac/ax (WiFi 4/5/6)
- RADIUS authentication
- WPS (WiFi Protected Setup)
- Hotspot 2.0
- Dynamic VLAN assignment

### Use Cases

- Create WiFi hotspot on Linux
- Home router/AP
- Enterprise wireless access point
- Captive portal
- Guest WiFi network
- Testing and development

---

## Installation

### Debian/Ubuntu

```bash
sudo apt-get update
sudo apt-get install hostapd

# Verify installation
hostapd -v
```

### Fedora/RHEL/CentOS

```bash
sudo dnf install hostapd

# Or for older systems
sudo yum install hostapd
```

### Arch Linux

```bash
sudo pacman -S hostapd
```

### Build from Source

```bash
# Download
git clone git://w1.fi/srv/git/hostap.git
cd hostap/hostapd

# Configure
cp defconfig .config
# Edit .config to enable features

# Build
make

# Install
sudo make install
```

---

## Basic Configuration

### Minimal Configuration

**File:** `/etc/hostapd/hostapd.conf`

```conf
# Interface to use
interface=wlan0

# Driver (nl80211 is modern standard)
driver=nl80211

# WiFi network name
ssid=MyAccessPoint

# WiFi mode (a = 5GHz, g = 2.4GHz)
hw_mode=g

# WiFi channel
channel=6

# WPA2 settings
wpa=2
wpa_passphrase=MySecurePassword123
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
```

### Open Network (No Security)

```conf
interface=wlan0
driver=nl80211
ssid=OpenWiFi
hw_mode=g
channel=6
# No WPA settings = open network
```

### Basic WPA2 Access Point

```conf
# Interface configuration
interface=wlan0
driver=nl80211

# SSID configuration
ssid=MyWiFi
utf8_ssid=1

# Hardware mode
hw_mode=g
channel=6

# IEEE 802.11n
ieee80211n=1
wmm_enabled=1

# Security: WPA2-Personal
auth_algs=1
wpa=2
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
wpa_passphrase=SecurePassword123

# Logging
logger_syslog=-1
logger_syslog_level=2
logger_stdout=-1
logger_stdout_level=2

# Country code
country_code=US

# Max clients
max_num_sta=20
```

---

## Running hostapd

### Manual Start

```bash
# Check configuration syntax
sudo hostapd -t /etc/hostapd/hostapd.conf

# Run in foreground (for testing)
sudo hostapd /etc/hostapd/hostapd.conf

# Run in background
sudo hostapd -B /etc/hostapd/hostapd.conf

# With debug output
sudo hostapd -d /etc/hostapd/hostapd.conf
sudo hostapd -dd /etc/hostapd/hostapd.conf  # More verbose
```

### Complete Setup Script

```bash
#!/bin/bash
# setup-ap.sh

INTERFACE=wlan0
SSID="MyAccessPoint"
PASSWORD="MyPassword123"
CHANNEL=6

# Stop existing processes
sudo killall hostapd 2>/dev/null
sudo killall dnsmasq 2>/dev/null

# Configure interface
sudo ip link set $INTERFACE down
sudo ip addr flush dev $INTERFACE
sudo ip link set $INTERFACE up
sudo ip addr add 192.168.50.1/24 dev $INTERFACE

# Create hostapd config
cat > /tmp/hostapd.conf << EOF
interface=$INTERFACE
driver=nl80211
ssid=$SSID
hw_mode=g
channel=$CHANNEL
wmm_enabled=1
auth_algs=1
wpa=2
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
wpa_passphrase=$PASSWORD
EOF

# Start hostapd
sudo hostapd -B /tmp/hostapd.conf

# Configure DHCP (dnsmasq)
sudo dnsmasq -C /dev/null \
    --interface=$INTERFACE \
    --dhcp-range=192.168.50.10,192.168.50.100,12h \
    --no-daemon &

# Enable NAT
sudo sysctl net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i $INTERFACE -o eth0 -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o $INTERFACE -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "Access Point started: SSID=$SSID"
```

---

## Security Configurations

### WPA2-Personal (PSK)

```conf
interface=wlan0
ssid=SecureWiFi

# WPA2 with AES-CCMP
wpa=2
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
wpa_passphrase=VerySecurePassword123

# Optional: require PMF (Protected Management Frames)
ieee80211w=1
```

### WPA3-Personal (SAE)

```conf
interface=wlan0
ssid=WPA3WiFi

# WPA3-Personal (SAE)
wpa=2
wpa_key_mgmt=SAE
rsn_pairwise=CCMP
sae_password=SecureWPA3Password

# PMF is required for WPA3
ieee80211w=2

# SAE-specific settings
sae_pwe=2
sae_groups=19 20 21
```

### WPA2/WPA3 Transition Mode

```conf
interface=wlan0
ssid=TransitionWiFi

# Both WPA2 and WPA3
wpa=2
wpa_key_mgmt=WPA-PSK SAE
rsn_pairwise=CCMP

# For WPA2
wpa_passphrase=Password123

# For WPA3
sae_password=Password123

# PMF optional (required for WPA3 clients)
ieee80211w=1
```

### WPA2-Enterprise (802.1X)

```conf
interface=wlan0
ssid=EnterpriseWiFi

# WPA2-Enterprise
wpa=2
wpa_key_mgmt=WPA-EAP
rsn_pairwise=CCMP

# IEEE 802.1X
ieee8021x=1

# RADIUS server configuration
auth_server_addr=192.168.1.10
auth_server_port=1812
auth_server_shared_secret=radiussecret

# Optional: Accounting server
acct_server_addr=192.168.1.10
acct_server_port=1813
acct_server_shared_secret=radiussecret

# EAP configuration
eap_server=0
eapol_key_index_workaround=0
```

### Hidden SSID

```conf
interface=wlan0
ssid=HiddenNetwork
# Hide SSID in beacons
ignore_broadcast_ssid=1

wpa=2
wpa_passphrase=password
```

### MAC Address Filtering

```conf
interface=wlan0
ssid=FilteredWiFi

# MAC address ACL
macaddr_acl=1
# 0 = accept unless in deny list
# 1 = deny unless in accept list
# 2 = use external RADIUS

# Accept list
accept_mac_file=/etc/hostapd/accept.mac

# Deny list (if macaddr_acl=0)
deny_mac_file=/etc/hostapd/deny.mac

wpa=2
wpa_passphrase=password
```

**`/etc/hostapd/accept.mac`:**
```
00:11:22:33:44:55
aa:bb:cc:dd:ee:ff
```

---

## Advanced Features

### Multiple SSIDs (Multi-BSS)

**Main configuration** `/etc/hostapd/hostapd.conf`:
```conf
# Primary interface
interface=wlan0
driver=nl80211
ctrl_interface=/var/run/hostapd

# Channel configuration (shared by all BSS)
hw_mode=g
channel=6
ieee80211n=1

# Primary SSID
ssid=MainWiFi
wpa=2
wpa_passphrase=MainPassword

# Multiple BSSs
bss=wlan0_0
ssid=GuestWiFi
wpa=2
wpa_passphrase=GuestPassword
# Isolate guest clients
ap_isolate=1

bss=wlan0_1
ssid=IoTWiFi
wpa=2
wpa_passphrase=IoTPassword
```

### Client Isolation

```conf
interface=wlan0
ssid=IsolatedWiFi

# Prevent clients from communicating with each other
ap_isolate=1

wpa=2
wpa_passphrase=password
```

### 5 GHz Configuration

```conf
interface=wlan0
driver=nl80211

# 5 GHz band
hw_mode=a
channel=36

# Channel width
# HT40+ = 40 MHz (channels 36,40)
# VHT80 = 80 MHz
# VHT160 = 160 MHz
vht_oper_chwidth=1
vht_oper_centr_freq_seg0_idx=42

ssid=5GHz_WiFi
wpa=2
wpa_passphrase=password
```

### WPS (WiFi Protected Setup)

```conf
interface=wlan0
ssid=WPS_WiFi

wpa=2
wpa_passphrase=password

# Enable WPS
wps_state=2
eap_server=1
# Device information
device_name=Linux_AP
manufacturer=OpenSource
model_name=hostapd
model_number=1.0
config_methods=push_button keypad

# UUID (generate with uuidgen)
uuid=12345678-9abc-def0-1234-56789abcdef0
```

**Trigger WPS:**
```bash
# Push button
hostapd_cli wps_pbc

# PIN method
hostapd_cli wps_pin any 12345670
```

---

## Bridge Mode

### Bridge Configuration

```bash
# Create bridge
sudo ip link add name br0 type bridge
sudo ip link set br0 up

# Add Ethernet to bridge
sudo ip link set eth0 master br0

# Configure bridge IP
sudo ip addr add 192.168.1.1/24 dev br0
```

**hostapd.conf:**
```conf
interface=wlan0
bridge=br0
driver=nl80211

ssid=BridgedWiFi
hw_mode=g
channel=6

wpa=2
wpa_passphrase=password
```

### Complete Bridge Setup

```bash
#!/bin/bash
# bridge-ap.sh

WLAN=wlan0
ETH=eth0
BRIDGE=br0

# Create bridge
sudo ip link add name $BRIDGE type bridge
sudo ip link set $BRIDGE up

# Add Ethernet
sudo ip link set $ETH down
sudo ip addr flush dev $ETH
sudo ip link set $ETH master $BRIDGE
sudo ip link set $ETH up

# Configure bridge
sudo ip addr add 192.168.1.1/24 dev $BRIDGE

# hostapd config with bridge
cat > /tmp/hostapd-bridge.conf << EOF
interface=$WLAN
bridge=$BRIDGE
driver=nl80211
ssid=BridgedAP
hw_mode=g
channel=6
wpa=2
wpa_passphrase=password
EOF

# Start hostapd
sudo hostapd -B /tmp/hostapd-bridge.conf

# Start DHCP server on bridge
sudo dnsmasq --interface=$BRIDGE \
    --dhcp-range=192.168.1.100,192.168.1.200,12h
```

---

## VLAN Support

### Static VLAN Assignment

**hostapd.conf:**
```conf
interface=wlan0
ssid=MultiVLAN_WiFi

wpa=2
wpa_passphrase=password

# Enable dynamic VLAN
dynamic_vlan=1
vlan_file=/etc/hostapd/vlan.conf
```

**`/etc/hostapd/vlan.conf`:**
```
# VLAN_ID  VLAN_IFNAME
1          wlan0.1
10         wlan0.10
20         wlan0.20
```

### VLAN with RADIUS

```conf
interface=wlan0
ssid=Enterprise_VLAN

wpa=2
wpa_key_mgmt=WPA-EAP
ieee8021x=1

# RADIUS server
auth_server_addr=192.168.1.10
auth_server_port=1812
auth_server_shared_secret=secret

# Dynamic VLAN from RADIUS
dynamic_vlan=1
vlan_naming=1
```

---

## RADIUS Authentication

### Internal EAP Server

```conf
interface=wlan0
ssid=InternalEAP_WiFi

# Use hostapd's internal EAP server
ieee8021x=1
eap_server=1
eap_user_file=/etc/hostapd/hostapd.eap_user
ca_cert=/etc/hostapd/ca.pem
server_cert=/etc/hostapd/server.pem
private_key=/etc/hostapd/server-key.pem
private_key_passwd=keypassword

wpa=2
wpa_key_mgmt=WPA-EAP
rsn_pairwise=CCMP
```

**`/etc/hostapd/hostapd.eap_user`:**
```
# Phase 1 authentication
* PEAP
"user1" MSCHAPV2 "password1" [2]
"user2" MSCHAPV2 "password2" [2]

# TLS
"client1" TLS
```

### External RADIUS Server

```conf
interface=wlan0
ssid=RADIUS_WiFi

wpa=2
wpa_key_mgmt=WPA-EAP
ieee8021x=1

# Primary RADIUS server
auth_server_addr=192.168.1.10
auth_server_port=1812
auth_server_shared_secret=sharedsecret

# Backup RADIUS server
auth_server_addr=192.168.1.11
auth_server_port=1812
auth_server_shared_secret=sharedsecret

# Accounting
acct_server_addr=192.168.1.10
acct_server_port=1813
acct_server_shared_secret=sharedsecret

# Disable internal EAP
eap_server=0
```

---

## 802.11n/ac/ax Configuration

### 802.11n (WiFi 4) - 2.4 GHz

```conf
interface=wlan0
ssid=N_WiFi_2_4GHz

hw_mode=g
channel=6

# Enable 802.11n
ieee80211n=1
wmm_enabled=1

# HT capabilities
ht_capab=[HT40+][SHORT-GI-20][SHORT-GI-40][DSSS_CCK-40]

wpa=2
wpa_passphrase=password
```

### 802.11n (WiFi 4) - 5 GHz

```conf
interface=wlan0
ssid=N_WiFi_5GHz

hw_mode=a
channel=36

ieee80211n=1
wmm_enabled=1

# 40 MHz channel
ht_capab=[HT40+][SHORT-GI-20][SHORT-GI-40]

wpa=2
wpa_passphrase=password
```

### 802.11ac (WiFi 5)

```conf
interface=wlan0
ssid=AC_WiFi

hw_mode=a
channel=36

# 802.11n required
ieee80211n=1
ht_capab=[HT40+][SHORT-GI-20][SHORT-GI-40]

# 802.11ac
ieee80211ac=1
vht_capab=[MAX-MPDU-11454][SHORT-GI-80][TX-STBC-2BY1][RX-STBC-1]

# 80 MHz channel
vht_oper_chwidth=1
vht_oper_centr_freq_seg0_idx=42

wmm_enabled=1

wpa=2
wpa_passphrase=password
```

### 802.11ax (WiFi 6)

```conf
interface=wlan0
ssid=AX_WiFi

hw_mode=a
channel=36

# 802.11n
ieee80211n=1
ht_capab=[HT40+][SHORT-GI-20][SHORT-GI-40]

# 802.11ac
ieee80211ac=1
vht_oper_chwidth=1
vht_oper_centr_freq_seg0_idx=42

# 802.11ax
ieee80211ax=1
he_su_beamformer=1
he_su_beamformee=1
he_mu_beamformer=1

wmm_enabled=1

wpa=3  # WPA3
wpa_key_mgmt=SAE
sae_password=password
ieee80211w=2
```

---

## Monitoring and Management

### hostapd_cli

```bash
# Connect to running hostapd
hostapd_cli

# Or specify interface
hostapd_cli -i wlan0

# Get status
hostapd_cli status

# List connected stations
hostapd_cli all_sta

# Disconnect a station
hostapd_cli disassociate <MAC>

# Reload configuration
hostapd_cli reload

# Enable/disable
hostapd_cli disable
hostapd_cli enable
```

### Monitor Connected Clients

```bash
# List all stations
hostapd_cli all_sta

# Detailed station info
hostapd_cli sta <MAC_ADDRESS>

# Example output:
# dot11RSNAStatsSTAAddress=aa:bb:cc:dd:ee:ff
# dot11RSNAStatsVersion=1
# dot11RSNAStatsSelectedPairwiseCipher=00-0f-ac-4
# dot11RSNAStatsTKIPLocalMICFailures=0
# flags=[AUTH][ASSOC][AUTHORIZED]
```

### Signal Strength

```bash
# Show signal strength for connected clients
for mac in $(hostapd_cli all_sta | grep ^[0-9a-f] | cut -d' ' -f1); do
    echo "Station: $mac"
    hostapd_cli sta $mac | grep signal
done
```

---

## Troubleshooting

### Check Configuration

```bash
# Test configuration syntax
sudo hostapd -t /etc/hostapd/hostapd.conf

# Expected output: Configuration file: /etc/hostapd/hostapd.conf
```

### Debug Mode

```bash
# Run in foreground with debug
sudo systemctl stop hostapd
sudo hostapd -d /etc/hostapd/hostapd.conf

# More verbose
sudo hostapd -dd /etc/hostapd/hostapd.conf
```

### Common Issues

**Cannot start AP - device busy:**
```bash
# Check if NetworkManager is controlling interface
nmcli device status

# Unmanage interface
sudo nmcli device set wlan0 managed no

# Or disable NetworkManager for interface
# /etc/NetworkManager/NetworkManager.conf
[keyfile]
unmanaged-devices=mac:aa:bb:cc:dd:ee:ff

sudo systemctl restart NetworkManager
```

**Channel not available:**
```bash
# Check supported channels
iw list | grep -A 20 "Frequencies:"

# Check regulatory domain
iw reg get

# Set country code
sudo iw reg set US

# Or in hostapd.conf
country_code=US
ieee80211d=1
```

**Interface doesn't support AP mode:**
```bash
# Check supported modes
iw list | grep -A 10 "Supported interface modes:"

# Should show:
#   * AP
#   * AP/VLAN

# If not present, hardware doesn't support AP mode
```

**Authentication failures:**
```bash
# Check logs
sudo journalctl -u hostapd -f

# Common causes:
# 1. Wrong password
# 2. Incompatible security settings
# 3. Client doesn't support WPA3
# 4. PMF issues

# Try WPA2 for compatibility
wpa=2
wpa_key_mgmt=WPA-PSK
```

**No DHCP addresses:**
```bash
# Check if DHCP server is running
ps aux | grep dnsmasq

# Check interface has IP
ip addr show wlan0

# Test DHCP manually
sudo dnsmasq --no-daemon --interface=wlan0 \
    --dhcp-range=192.168.50.10,192.168.50.100,12h \
    --log-queries
```

---

## Integration with systemd

### systemd Service

```bash
# Enable and start
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl start hostapd

# Status
sudo systemctl status hostapd

# Logs
sudo journalctl -u hostapd -f
```

### Custom Service File

**`/etc/systemd/system/hostapd.service`:**
```ini
[Unit]
Description=Access point and authentication server
After=network.target

[Service]
Type=forking
PIDFile=/var/run/hostapd.pid
ExecStart=/usr/sbin/hostapd -B -P /var/run/hostapd.pid /etc/hostapd/hostapd.conf
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Configuration File Location

**`/etc/default/hostapd`:**
```bash
DAEMON_CONF="/etc/hostapd/hostapd.conf"
```

---

## Best Practices

### Security

1. **Use WPA3 when possible:**
```conf
wpa=2
wpa_key_mgmt=SAE
ieee80211w=2
```

2. **Strong passwords:**
```bash
# Minimum 12 characters
wpa_passphrase=MyVerySecurePassword123!
```

3. **Disable WPS in production:**
```conf
wps_state=0
```

4. **Enable PMF:**
```conf
ieee80211w=1  # Optional
# or
ieee80211w=2  # Required (WPA3)
```

5. **Guest network isolation:**
```conf
bss=wlan0_0
ssid=Guest
ap_isolate=1
```

### Performance

1. **Use 5 GHz for better performance:**
```conf
hw_mode=a
channel=36
```

2. **Enable 802.11n/ac:**
```conf
ieee80211n=1
ieee80211ac=1
wmm_enabled=1
```

3. **Choose non-overlapping channels:**
```
2.4 GHz: 1, 6, 11
5 GHz: Many options (36, 40, 44, 48...)
```

4. **Limit max clients:**
```conf
max_num_sta=50
```

### Reliability

1. **Set country code:**
```conf
country_code=US
ieee80211d=1
ieee80211h=1
```

2. **Enable logging:**
```conf
logger_syslog=-1
logger_syslog_level=2
```

3. **Automatic restart:**
```bash
sudo systemctl enable hostapd
```

---

## Summary

**hostapd** creates WiFi access points on Linux:

**Basic workflow:**
1. Configure `/etc/hostapd/hostapd.conf`
2. Start: `sudo hostapd /etc/hostapd/hostapd.conf`
3. Configure DHCP server (dnsmasq)
4. Enable IP forwarding and NAT (for internet sharing)

**Minimal config:**
```conf
interface=wlan0
ssid=MyWiFi
channel=6
wpa=2
wpa_passphrase=password
```

**Essential commands:**
- `hostapd -t`: Test configuration
- `hostapd_cli`: Control running AP
- `systemctl start hostapd`: Start service

**Common tasks:**
- WPA2 AP: Configure `wpa=2` and `wpa_passphrase`
- WPA3 AP: Use `key_mgmt=SAE` and `ieee80211w=2`
- Guest network: Use multi-BSS with `ap_isolate=1`
- Bridge mode: Set `bridge=br0`

**Resources:**
- [hostapd Documentation](https://w1.fi/hostapd/)
- [Linux Wireless](https://wireless.wiki.kernel.org/)
- [ArchWiki: Software access point](https://wiki.archlinux.org/title/Software_access_point)
