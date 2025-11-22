# OpenWRT

OpenWRT is a highly extensible, Linux-based operating system for embedded devices, primarily targeting wireless routers and network appliances. It replaces vendor firmware with a fully writable filesystem and package management, enabling customization and advanced networking features.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [First Boot](#first-boot)
   - [Web Interface (LuCI)](#web-interface-luci)
3. [UCI System](#uci-system)
   - [UCI Commands](#uci-commands)
   - [Configuration Files](#configuration-files)
   - [Batch Operations](#batch-operations)
4. [Network Configuration](#network-configuration)
   - [Interface Configuration](#interface-configuration)
   - [Bridge Configuration](#bridge-configuration)
   - [VLAN Setup](#vlan-setup)
   - [Wireless Configuration](#wireless-configuration)
   - [Network Protocols](#network-protocols)
5. [Package Management](#package-management)
   - [opkg Commands](#opkg-commands)
   - [Repository Management](#repository-management)
   - [Common Packages](#common-packages)
6. [Firewall Configuration](#firewall-configuration)
   - [Zones](#zones)
   - [Rules and Forwarding](#rules-and-forwarding)
   - [Port Forwarding](#port-forwarding)
   - [NAT Configuration](#nat-configuration)
7. [Services](#services)
   - [DHCP/DNS (dnsmasq)](#dhcpdns-dnsmasq)
   - [VPN Setup](#vpn-setup)
   - [QoS Configuration](#qos-configuration)
   - [Routing Protocols](#routing-protocols)
8. [Building OpenWRT](#building-openwrt)
   - [Build System Setup](#build-system-setup)
   - [Building Custom Images](#building-custom-images)
   - [Package Development](#package-development)
   - [Kernel Configuration](#kernel-configuration)
9. [Advanced Topics](#advanced-topics)
   - [Hotplug System](#hotplug-system)
   - [Custom Scripts](#custom-scripts)
   - [Network Namespaces](#network-namespaces)
   - [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
    - [Failsafe Mode](#failsafe-mode)
    - [Recovery Options](#recovery-options)
    - [Network Debugging](#network-debugging)
    - [Log Analysis](#log-analysis)
11. [Best Practices](#best-practices)
12. [Quick Reference](#quick-reference)

---

## Overview

**Key Features:**
- **Package Management**: Install/remove software with opkg
- **UCI**: Unified Configuration Interface for consistent config management
- **LuCI**: Web-based administration interface
- **Extensive Hardware Support**: 1000+ device profiles
- **Active Development**: Regular releases and security updates
- **Flexible Networking**: VLANs, bridges, advanced routing, QoS
- **Open Source**: GPL licensed, community-driven

**Use Cases:**
- Custom router/gateway firmware
- Wireless access points
- Network attached storage (NAS)
- IoT gateways
- VPN endpoints
- Network monitoring and traffic shaping
- Development platform for embedded networking

**Architecture:**
```
┌─────────────────────────────────────────┐
│         LuCI Web Interface              │
│              (HTTP/HTTPS)               │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         UCI (Configuration)             │
│    /etc/config/{network,wireless,...}   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Services & Daemons                │
│  netifd, dnsmasq, dropbear, firewall    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Linux Kernel                    │
│    (drivers, netfilter, wireless)       │
└─────────────────────────────────────────┘
```

---

## Getting Started

### Installation

**Download firmware for your device:**
```bash
# Example: TP-Link Archer C7 v2
wget https://downloads.openwrt.org/releases/23.05.3/targets/ath79/generic/\
openwrt-23.05.3-ath79-generic-tplink_archer-c7-v2-squashfs-factory.bin

# For upgrades from existing OpenWRT:
wget https://downloads.openwrt.org/releases/23.05.3/targets/ath79/generic/\
openwrt-23.05.3-ath79-generic-tplink_archer-c7-v2-squashfs-sysupgrade.bin
```

**Flash via stock firmware web interface:**
1. Access vendor's web interface (usually http://192.168.1.1)
2. Navigate to firmware upgrade section
3. Upload the `-factory.bin` file
4. Wait for device to reboot (3-5 minutes)

**Flash via TFTP (recovery method):**
```bash
# Set PC IP to 192.168.1.10
sudo ip addr add 192.168.1.10/24 dev eth0

# Start TFTP server
sudo dnsmasq -i eth0 --dhcp-range=192.168.1.100,192.168.1.200 \
  --enable-tftp --tftp-root=/tmp -d -u root --log-dhcp --bootp-dynamic

# Place firmware in /tmp/, power on router while holding reset button
```

### First Boot

**Initial access (no password set):**
```bash
# Via telnet (only available before password is set)
telnet 192.168.1.1

# Or via serial console (115200 8N1)
screen /dev/ttyUSB0 115200
```

**Set root password (enables SSH, disables telnet):**
```bash
passwd
# Enter new password twice

# SSH will now be available
ssh root@192.168.1.1
```

**Basic network setup:**
```bash
# Configure WAN interface (if using DHCP)
uci set network.wan.proto='dhcp'
uci commit network
/etc/init.d/network reload

# Configure LAN IP address
uci set network.lan.ipaddr='192.168.2.1'
uci commit network
/etc/init.d/network reload
```

### Web Interface (LuCI)

**Install LuCI (if not included):**
```bash
# Update package lists
opkg update

# Install minimal LuCI
opkg install luci

# Install full LuCI with SSL
opkg install luci luci-ssl

# Start web server
/etc/init.d/uhttpd start
/etc/init.d/uhttpd enable
```

**Access:**
- HTTP: http://192.168.1.1
- HTTPS: https://192.168.1.1 (if luci-ssl installed)
- Default login: root / (password you set)

**Install additional LuCI apps:**
```bash
# QoS management
opkg install luci-app-qos

# VPN (OpenVPN)
opkg install luci-app-openvpn

# Statistics
opkg install luci-app-statistics

# AdBlock
opkg install luci-app-adblock
```

---

## UCI System

UCI (Unified Configuration Interface) is OpenWRT's centralized configuration system. All system settings are stored in `/etc/config/` as simple text files.

### UCI Commands

**Show configuration:**
```bash
# Show all configuration
uci show

# Show specific config file
uci show network
uci show wireless
uci show firewall

# Show specific section
uci show network.lan
uci show wireless.@wifi-iface[0]

# Export in different formats
uci export network
```

**Get values:**
```bash
# Get specific option
uci get network.lan.ipaddr
# Output: 192.168.1.1

# Get section type
uci get network.lan
# Output: interface
```

**Set values:**
```bash
# Set option
uci set network.lan.ipaddr='192.168.2.1'
uci set network.lan.netmask='255.255.255.0'

# Create new section
uci set network.guest=interface
uci set network.guest.proto='static'
uci set network.guest.ipaddr='192.168.3.1'

# Add to list
uci add_list firewall.@zone[0].network='guest'
```

**Delete values:**
```bash
# Delete option
uci delete network.lan.ip6assign

# Delete section
uci delete network.guest

# Remove from list
uci del_list firewall.@zone[0].network='guest'
```

**Commit and apply changes:**
```bash
# Save changes to file
uci commit network

# Save all pending changes
uci commit

# Reload service
/etc/init.d/network reload

# Or restart service
/etc/init.d/network restart

# Revert uncommitted changes
uci revert network
```

### Configuration Files

**Common config files in /etc/config/:**

| File | Purpose |
|------|---------|
| `network` | Network interfaces, VLANs, routes |
| `wireless` | WiFi settings, SSIDs, encryption |
| `firewall` | Firewall zones, rules, forwarding |
| `dhcp` | DHCP/DNS settings (dnsmasq) |
| `system` | Hostname, timezone, LED configuration |
| `dropbear` | SSH server settings |
| `uhttpd` | Web server (LuCI) configuration |
| `qos` | Quality of Service settings |

**Example /etc/config/network:**
```
config interface 'loopback'
	option device 'lo'
	option proto 'static'
	option ipaddr '127.0.0.1'
	option netmask '255.0.0.0'

config interface 'lan'
	option device 'br-lan'
	option proto 'static'
	option ipaddr '192.168.1.1'
	option netmask '255.255.255.0'
	option ip6assign '60'

config interface 'wan'
	option device 'eth1'
	option proto 'dhcp'

config interface 'wan6'
	option device 'eth1'
	option proto 'dhcpv6'
```

### Batch Operations

**Using uci batch mode:**
```bash
uci batch << EOF
set network.lan.ipaddr='192.168.100.1'
set network.lan.netmask='255.255.255.0'
delete network.wan6
commit network
EOF
```

**Scripting with UCI:**
```bash
#!/bin/sh
# Configure guest network

uci set network.guest=interface
uci set network.guest.proto='static'
uci set network.guest.ipaddr='10.0.0.1'
uci set network.guest.netmask='255.255.255.0'
uci set network.guest.device='br-guest'

uci commit network
/etc/init.d/network reload
```

---

## Network Configuration

### Interface Configuration

**Static IP configuration:**
```bash
# Using UCI
uci set network.lan=interface
uci set network.lan.proto='static'
uci set network.lan.device='br-lan'
uci set network.lan.ipaddr='192.168.1.1'
uci set network.lan.netmask='255.255.255.0'
uci set network.lan.gateway='192.168.1.254'
uci set network.lan.dns='8.8.8.8 8.8.4.4'
uci commit network
/etc/init.d/network reload
```

**DHCP client:**
```bash
uci set network.wan=interface
uci set network.wan.proto='dhcp'
uci set network.wan.device='eth1'
uci commit network
/etc/init.d/network reload
```

**PPPoE (DSL/Fiber):**
```bash
uci set network.wan=interface
uci set network.wan.proto='pppoe'
uci set network.wan.device='eth1'
uci set network.wan.username='user@isp.com'
uci set network.wan.password='password'
uci set network.wan.ipv6='auto'
uci commit network
/etc/init.d/network reload
```

**Multiple IP addresses (alias):**
```bash
uci set network.lan2=interface
uci set network.lan2.proto='static'
uci set network.lan2.device='@lan'
uci set network.lan2.ipaddr='192.168.2.1'
uci set network.lan2.netmask='255.255.255.0'
uci commit network
/etc/init.d/network reload
```

**Check interface status:**
```bash
# Show all interfaces
ifstatus lan
ifstatus wan

# Using ip command
ip addr show
ip route show

# Using netifd
ubus list network.interface.*
ubus call network.interface.lan status
```

### Bridge Configuration

**Create bridge:**
```bash
# Define bridge device
uci set network.br_guest=device
uci set network.br_guest.type='bridge'
uci set network.br_guest.name='br-guest'
uci add_list network.br_guest.ports='eth0.3'

# Create interface on bridge
uci set network.guest=interface
uci set network.guest.proto='static'
uci set network.guest.device='br-guest'
uci set network.guest.ipaddr='10.0.0.1'
uci set network.guest.netmask='255.255.255.0'

uci commit network
/etc/init.d/network reload
```

**Bridge with wireless:**
```bash
# Bridge will be created automatically when wireless references the network
# See Wireless Configuration section
```

### VLAN Setup

**Switch configuration (newer devices with DSA):**
```bash
# Define bridge VLAN filtering
uci set network.@device[0]=device
uci set network.@device[0].name='br-lan'
uci set network.@device[0].type='bridge'
uci add_list network.@device[0].ports='lan1'
uci add_list network.@device[0].ports='lan2'
uci add_list network.@device[0].ports='lan3'
uci add_list network.@device[0].ports='lan4'

# Add VLAN
uci set network.@bridge-vlan[0]=bridge-vlan
uci set network.@bridge-vlan[0].device='br-lan'
uci set network.@bridge-vlan[0].vlan='10'
uci add_list network.@bridge-vlan[0].ports='lan1:t'
uci add_list network.@bridge-vlan[0].ports='lan2:u'

uci commit network
/etc/init.d/network reload
```

**Switch configuration (older swconfig-based):**
```bash
# Configure switch
uci set network.@switch_vlan[0]=switch_vlan
uci set network.@switch_vlan[0].device='switch0'
uci set network.@switch_vlan[0].vlan='1'
uci set network.@switch_vlan[0].ports='0 1 2 3 6t'

uci set network.@switch_vlan[1]=switch_vlan
uci set network.@switch_vlan[1].device='switch0'
uci set network.@switch_vlan[1].vlan='2'
uci set network.@switch_vlan[1].ports='4 6t'

uci commit network
/etc/init.d/network reload
```

**Tagged VLAN interface:**
```bash
# Create VLAN interface
uci set network.vlan10=device
uci set network.vlan10.type='8021q'
uci set network.vlan10.ifname='eth0'
uci set network.vlan10.vid='10'

# Create interface on VLAN
uci set network.iot=interface
uci set network.iot.proto='static'
uci set network.iot.device='eth0.10'
uci set network.iot.ipaddr='192.168.10.1'
uci set network.iot.netmask='255.255.255.0'

uci commit network
/etc/init.d/network reload
```

### Wireless Configuration

**Scan for wireless devices:**
```bash
# List wireless devices
uci show wireless

# Scan for networks (if enabled)
iw dev wlan0 scan

# Or using UCI
wifi status
```

**Basic WiFi AP setup:**
```bash
# Enable radio
uci set wireless.radio0.disabled='0'
uci set wireless.radio0.channel='6'
uci set wireless.radio0.txpower='20'
uci set wireless.radio0.country='US'

# Configure AP interface
uci set wireless.default_radio0=wifi-iface
uci set wireless.default_radio0.device='radio0'
uci set wireless.default_radio0.mode='ap'
uci set wireless.default_radio0.network='lan'
uci set wireless.default_radio0.ssid='MyOpenWRT'
uci set wireless.default_radio0.encryption='psk2'
uci set wireless.default_radio0.key='MyPassword123'

uci commit wireless
wifi reload
```

**5GHz configuration:**
```bash
uci set wireless.radio1.disabled='0'
uci set wireless.radio1.channel='36'
uci set wireless.radio1.htmode='VHT80'
uci set wireless.radio1.country='US'

uci set wireless.default_radio1=wifi-iface
uci set wireless.default_radio1.device='radio1'
uci set wireless.default_radio1.mode='ap'
uci set wireless.default_radio1.network='lan'
uci set wireless.default_radio1.ssid='MyOpenWRT-5G'
uci set wireless.default_radio1.encryption='psk2'
uci set wireless.default_radio1.key='MyPassword123'

uci commit wireless
wifi reload
```

**Guest network (isolated):**
```bash
# Configure guest WiFi
uci set wireless.guest=wifi-iface
uci set wireless.guest.device='radio0'
uci set wireless.guest.mode='ap'
uci set wireless.guest.network='guest'
uci set wireless.guest.ssid='Guest-WiFi'
uci set wireless.guest.encryption='psk2'
uci set wireless.guest.key='GuestPassword'
uci set wireless.guest.isolate='1'  # Client isolation

uci commit wireless
wifi reload
```

**WiFi client mode:**
```bash
uci set wireless.sta=wifi-iface
uci set wireless.sta.device='radio0'
uci set wireless.sta.mode='sta'
uci set wireless.sta.network='wwan'
uci set wireless.sta.ssid='UpstreamAP'
uci set wireless.sta.encryption='psk2'
uci set wireless.sta.key='UpstreamPassword'

# Create interface
uci set network.wwan=interface
uci set network.wwan.proto='dhcp'

uci commit wireless
uci commit network
wifi reload
```

**Advanced WiFi settings:**
```bash
# 802.11w (Management Frame Protection)
uci set wireless.default_radio0.ieee80211w='1'

# WPA3
uci set wireless.default_radio0.encryption='sae'
uci set wireless.default_radio0.key='StrongPassword'

# Hidden SSID
uci set wireless.default_radio0.hidden='1'

# MAC filtering
uci set wireless.default_radio0.macfilter='allow'
uci add_list wireless.default_radio0.maclist='00:11:22:33:44:55'

# Disable WPS
uci set wireless.default_radio0.wps_pushbutton='0'

uci commit wireless
wifi reload
```

### Network Protocols

**IPv6 configuration:**
```bash
# Enable IPv6 on LAN
uci set network.lan.ip6assign='64'

# DHCPv6 client on WAN
uci set network.wan6=interface
uci set network.wan6.device='@wan'
uci set network.wan6.proto='dhcpv6'
uci set network.wan6.reqaddress='try'
uci set network.wan6.reqprefix='auto'

uci commit network
/etc/init.d/network reload
```

**Static routes:**
```bash
# Add static route
uci add network route
uci set network.@route[-1].interface='lan'
uci set network.@route[-1].target='10.0.0.0'
uci set network.@route[-1].netmask='255.255.255.0'
uci set network.@route[-1].gateway='192.168.1.254'

uci commit network
/etc/init.d/network reload

# Verify routes
ip route show
```

**Policy routing:**
```bash
# Create routing table
echo "100 vpn" >> /etc/iproute2/rt_tables

# Add rule
uci add network rule
uci set network.@rule[-1].in='lan'
uci set network.@rule[-1].src='192.168.1.0/24'
uci set network.@rule[-1].lookup='100'

# Add route in table
uci add network route
uci set network.@route[-1].interface='vpn'
uci set network.@route[-1].target='0.0.0.0/0'
uci set network.@route[-1].table='100'

uci commit network
/etc/init.d/network reload
```

---

## Package Management

### opkg Commands

**Update package lists:**
```bash
opkg update
```

**Search for packages:**
```bash
# Search by name
opkg list | grep vpn

# Find package
opkg find 'openvpn*'

# Show package info
opkg info openvpn
```

**Install packages:**
```bash
# Install single package
opkg install openvpn

# Install multiple packages
opkg install openvpn luci-app-openvpn

# Force reinstall
opkg install --force-reinstall openvpn

# Install from URL
opkg install http://example.com/package.ipk
```

**Remove packages:**
```bash
# Remove package
opkg remove openvpn

# Remove with dependencies
opkg remove --autoremove openvpn

# Remove configuration too
opkg remove --force-removal-of-dependent-packages openvpn
```

**Upgrade packages:**
```bash
# List upgradable packages
opkg list-upgradable

# Upgrade single package
opkg upgrade openvpn

# Upgrade all packages (use with caution)
opkg list-upgradable | cut -f 1 -d ' ' | xargs opkg upgrade
```

**List packages:**
```bash
# List installed packages
opkg list-installed

# List all available packages
opkg list

# List changed config files
opkg list-changed-conffiles
```

### Repository Management

**Configuration file: /etc/opkg/distfeeds.conf**
```bash
# View repositories
cat /etc/opkg/distfeeds.conf

# Typical content:
# src/gz openwrt_core https://downloads.openwrt.org/releases/23.05.3/targets/ath79/generic/packages
# src/gz openwrt_base https://downloads.openwrt.org/releases/23.05.3/packages/mips_24kc/base
# src/gz openwrt_luci https://downloads.openwrt.org/releases/23.05.3/packages/mips_24kc/luci
# src/gz openwrt_packages https://downloads.openwrt.org/releases/23.05.3/packages/mips_24kc/packages
# src/gz openwrt_routing https://downloads.openwrt.org/releases/23.05.3/packages/mips_24kc/routing
```

**Add custom repository:**
```bash
echo "src/gz custom http://example.com/openwrt/packages" >> /etc/opkg/customfeeds.conf
opkg update
```

**Check package dependencies:**
```bash
# Show dependencies
opkg depends openvpn

# Show what depends on package
opkg whatdepends libc
```

### Common Packages

**Networking:**
```bash
# Network tools
opkg install tcpdump iperf3 mtr nmap ethtool

# Advanced routing
opkg install quagga bird2

# VPN
opkg install openvpn wireguard-tools

# Traffic control
opkg install tc sqm-scripts luci-app-sqm
```

**System utilities:**
```bash
# USB storage support
opkg install kmod-usb-storage block-mount kmod-fs-ext4

# File system tools
opkg install e2fsprogs fdisk

# File transfer
opkg install rsync wget-ssl curl

# Text editors
opkg install vim-full nano
```

**Monitoring:**
```bash
# System monitoring
opkg install htop iotop collectd luci-app-statistics

# Network monitoring
opkg install bwm-ng iftop

# SNMP
opkg install snmpd snmp-utils
```

**Services:**
```bash
# Samba file sharing
opkg install samba4-server luci-app-samba4

# NFS
opkg install nfs-kernel-server

# Print server
opkg install p910nd luci-app-p910nd

# DDNS
opkg install ddns-scripts luci-app-ddns
```

---

## Firewall Configuration

OpenWRT uses `fw4` (nftables-based) on recent versions, or `fw3` (iptables-based) on older versions. Configuration is done through UCI.

### Zones

**Default zones:**
- **lan**: Trusted local network
- **wan**: Untrusted internet connection
- **guest**: Isolated guest network (if configured)

**View zones:**
```bash
uci show firewall | grep zone
```

**Create new zone:**
```bash
# Create guest zone
uci add firewall zone
uci set firewall.@zone[-1].name='guest'
uci set firewall.@zone[-1].input='REJECT'
uci set firewall.@zone[-1].output='ACCEPT'
uci set firewall.@zone[-1].forward='REJECT'
uci set firewall.@zone[-1].network='guest'

uci commit firewall
/etc/init.d/firewall reload
```

**Zone defaults:**
```bash
# LAN zone (example)
config zone
	option name 'lan'
	option input 'ACCEPT'
	option output 'ACCEPT'
	option forward 'ACCEPT'
	list network 'lan'

# WAN zone (example)
config zone
	option name 'wan'
	option input 'REJECT'
	option output 'ACCEPT'
	option forward 'REJECT'
	option masq '1'        # Enable NAT
	option mtu_fix '1'     # MSS clamping
	list network 'wan'
	list network 'wan6'
```

### Rules and Forwarding

**Zone forwarding:**
```bash
# Allow LAN -> WAN
uci add firewall forwarding
uci set firewall.@forwarding[-1].src='lan'
uci set firewall.@forwarding[-1].dest='wan'

# Allow guest -> WAN (but not to LAN)
uci add firewall forwarding
uci set firewall.@forwarding[-1].src='guest'
uci set firewall.@forwarding[-1].dest='wan'

uci commit firewall
/etc/init.d/firewall reload
```

**Traffic rules:**
```bash
# Allow SSH from WAN
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-SSH-WAN'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].proto='tcp'
uci set firewall.@rule[-1].dest_port='22'
uci set firewall.@rule[-1].target='ACCEPT'

# Allow ping from WAN
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-Ping'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].proto='icmp'
uci set firewall.@rule[-1].icmp_type='echo-request'
uci set firewall.@rule[-1].target='ACCEPT'

# Block specific IP
uci add firewall rule
uci set firewall.@rule[-1].name='Block-BadIP'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].src_ip='1.2.3.4'
uci set firewall.@rule[-1].target='DROP'

uci commit firewall
/etc/init.d/firewall reload
```

**Rate limiting:**
```bash
# Limit SSH connection attempts
uci add firewall rule
uci set firewall.@rule[-1].name='SSH-Rate-Limit'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].proto='tcp'
uci set firewall.@rule[-1].dest_port='22'
uci set firewall.@rule[-1].limit='5/minute'
uci set firewall.@rule[-1].limit_burst='10'
uci set firewall.@rule[-1].target='ACCEPT'

uci commit firewall
/etc/init.d/firewall reload
```

### Port Forwarding

**Simple port forward:**
```bash
# Forward external port 8080 to internal server port 80
uci add firewall redirect
uci set firewall.@redirect[-1].name='HTTP-Forward'
uci set firewall.@redirect[-1].src='wan'
uci set firewall.@redirect[-1].src_dport='8080'
uci set firewall.@redirect[-1].dest='lan'
uci set firewall.@redirect[-1].dest_ip='192.168.1.100'
uci set firewall.@redirect[-1].dest_port='80'
uci set firewall.@redirect[-1].proto='tcp'
uci set firewall.@redirect[-1].target='DNAT'

uci commit firewall
/etc/init.d/firewall reload
```

**Port forward range:**
```bash
# Forward ports 5000-5010
uci add firewall redirect
uci set firewall.@redirect[-1].name='Media-Ports'
uci set firewall.@redirect[-1].src='wan'
uci set firewall.@redirect[-1].src_dport='5000-5010'
uci set firewall.@redirect[-1].dest='lan'
uci set firewall.@redirect[-1].dest_ip='192.168.1.50'
uci set firewall.@redirect[-1].proto='tcp udp'
uci set firewall.@redirect[-1].target='DNAT'

uci commit firewall
/etc/init.d/firewall reload
```

**DMZ (forward all ports):**
```bash
uci add firewall redirect
uci set firewall.@redirect[-1].name='DMZ'
uci set firewall.@redirect[-1].src='wan'
uci set firewall.@redirect[-1].proto='tcp udp'
uci set firewall.@redirect[-1].dest='lan'
uci set firewall.@redirect[-1].dest_ip='192.168.1.200'
uci set firewall.@redirect[-1].target='DNAT'

uci commit firewall
/etc/init.d/firewall reload
```

### NAT Configuration

**Source NAT (masquerading):**
```bash
# Enable masquerading on WAN zone
uci set firewall.@zone[1].masq='1'
uci commit firewall
/etc/init.d/firewall reload
```

**Static NAT:**
```bash
# SNAT: Change source IP
uci add firewall nat
uci set firewall.@nat[-1].name='Static-SNAT'
uci set firewall.@nat[-1].src='lan'
uci set firewall.@nat[-1].src_ip='192.168.1.100'
uci set firewall.@nat[-1].dest='wan'
uci set firewall.@nat[-1].target='SNAT'
uci set firewall.@nat[-1].snat_ip='203.0.113.10'

uci commit firewall
/etc/init.d/firewall reload
```

**Hairpin NAT (NAT reflection):**
```bash
# Allow accessing port forwards from LAN using WAN IP
uci set firewall.@defaults[0].syn_flood='1'
uci set firewall.@defaults[0].input='ACCEPT'
uci set firewall.@defaults[0].output='ACCEPT'
uci set firewall.@defaults[0].forward='REJECT'

# Add reflection
uci add firewall redirect
uci set firewall.@redirect[-1].name='HTTP-Reflection'
uci set firewall.@redirect[-1].src='lan'
uci set firewall.@redirect[-1].dest='lan'
uci set firewall.@redirect[-1].src_dip='192.168.1.1'  # Router LAN IP
uci set firewall.@redirect[-1].src_dport='80'
uci set firewall.@redirect[-1].dest_ip='192.168.1.100'
uci set firewall.@redirect[-1].dest_port='80'
uci set firewall.@redirect[-1].proto='tcp'
uci set firewall.@redirect[-1].target='DNAT'

uci commit firewall
/etc/init.d/firewall reload
```

---

## Services

### DHCP/DNS (dnsmasq)

**DHCP configuration:**
```bash
# Configure DHCP for LAN
uci set dhcp.lan=dhcp
uci set dhcp.lan.interface='lan'
uci set dhcp.lan.start='100'
uci set dhcp.lan.limit='150'
uci set dhcp.lan.leasetime='12h'

# Set DNS servers
uci set dhcp.lan.dhcp_option='6,8.8.8.8,8.8.4.4'

uci commit dhcp
/etc/init.d/dnsmasq reload
```

**Static DHCP leases:**
```bash
# Reserve IP for device
uci add dhcp host
uci set dhcp.@host[-1].name='myserver'
uci set dhcp.@host[-1].mac='00:11:22:33:44:55'
uci set dhcp.@host[-1].ip='192.168.1.50'

uci commit dhcp
/etc/init.d/dnsmasq reload
```

**DNS configuration:**
```bash
# Custom DNS servers
uci set dhcp.@dnsmasq[0].server='8.8.8.8'
uci add_list dhcp.@dnsmasq[0].server='1.1.1.1'

# Domain name
uci set dhcp.@dnsmasq[0].domain='home.local'
uci set dhcp.@dnsmasq[0].local='/home.local/'

# DNS rebind protection
uci set dhcp.@dnsmasq[0].rebind_protection='1'

# Increase cache size
uci set dhcp.@dnsmasq[0].cachesize='1000'

uci commit dhcp
/etc/init.d/dnsmasq reload
```

**Local DNS records:**
```bash
# Add A record
uci add dhcp domain
uci set dhcp.@domain[-1].name='router.home'
uci set dhcp.@domain[-1].ip='192.168.1.1'

# Add CNAME
uci add dhcp cname
uci set dhcp.@cname[-1].cname='www.home'
uci set dhcp.@cname[-1].target='router.home'

uci commit dhcp
/etc/init.d/dnsmasq reload
```

**DHCP relay:**
```bash
uci set dhcp.lan.ignore='1'
uci add dhcp relay
uci set dhcp.@relay[-1].interface='lan'
uci set dhcp.@relay[-1].server_addr='192.168.1.10'
uci set dhcp.@relay[-1].local_addr='192.168.1.1'

uci commit dhcp
/etc/init.d/dnsmasq reload
```

### VPN Setup

**OpenVPN server:**
```bash
# Install packages
opkg update
opkg install openvpn-openssl openvpn-easy-rsa luci-app-openvpn

# Generate keys (simplified)
cd /etc/openvpn
easyrsa init-pki
easyrsa build-ca nopass
easyrsa gen-dh
easyrsa build-server-full server nopass
easyrsa build-client-full client1 nopass

# Configure server
cat > /etc/openvpn/server.conf << EOF
port 1194
proto udp
dev tun
ca /etc/openvpn/pki/ca.crt
cert /etc/openvpn/pki/issued/server.crt
key /etc/openvpn/pki/private/server.key
dh /etc/openvpn/pki/dh.pem
server 10.8.0.0 255.255.255.0
push "redirect-gateway def1"
push "dhcp-option DNS 8.8.8.8"
keepalive 10 120
cipher AES-256-GCM
user nobody
group nogroup
persist-key
persist-tun
status /var/log/openvpn-status.log
verb 3
EOF

# Enable and start
/etc/init.d/openvpn enable
/etc/init.d/openvpn start

# Allow VPN through firewall
uci set firewall.@rule[-1].name='Allow-OpenVPN'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].dest_port='1194'
uci set firewall.@rule[-1].proto='udp'
uci set firewall.@rule[-1].target='ACCEPT'
uci commit firewall
/etc/init.d/firewall reload
```

**WireGuard:**
```bash
# Install WireGuard
opkg update
opkg install wireguard-tools luci-app-wireguard luci-proto-wireguard

# Generate keys
umask 077
wg genkey | tee /etc/wireguard/privatekey | wg pubkey > /etc/wireguard/publickey

# Configure interface
uci set network.wg0=interface
uci set network.wg0.proto='wireguard'
uci set network.wg0.private_key="$(cat /etc/wireguard/privatekey)"
uci set network.wg0.listen_port='51820'
uci add_list network.wg0.addresses='10.0.0.1/24'

# Add peer
uci add network wireguard_wg0
uci set network.@wireguard_wg0[-1].public_key='CLIENT_PUBLIC_KEY'
uci set network.@wireguard_wg0[-1].preshared_key='PRESHARED_KEY'
uci add_list network.@wireguard_wg0[-1].allowed_ips='10.0.0.2/32'
uci set network.@wireguard_wg0[-1].persistent_keepalive='25'

uci commit network
/etc/init.d/network reload

# Firewall rule
uci add firewall rule
uci set firewall.@rule[-1].name='Allow-WireGuard'
uci set firewall.@rule[-1].src='wan'
uci set firewall.@rule[-1].dest_port='51820'
uci set firewall.@rule[-1].proto='udp'
uci set firewall.@rule[-1].target='ACCEPT'
uci commit firewall
/etc/init.d/firewall reload
```

### QoS Configuration

**SQM (Smart Queue Management):**
```bash
# Install SQM
opkg update
opkg install sqm-scripts luci-app-sqm

# Configure SQM
uci set sqm.@queue[0]=queue
uci set sqm.@queue[0].interface='eth1'  # WAN interface
uci set sqm.@queue[0].enabled='1'
uci set sqm.@queue[0].download='50000'  # kbits
uci set sqm.@queue[0].upload='10000'    # kbits
uci set sqm.@queue[0].script='piece_of_cake.qos'
uci set sqm.@queue[0].qdisc='cake'
uci set sqm.@queue[0].debug_logging='0'

uci commit sqm
/etc/init.d/sqm enable
/etc/init.d/sqm start
```

**Traditional QoS:**
```bash
# Install QoS
opkg update
opkg install qos-scripts luci-app-qos

# Configure
uci set qos.wan=interface
uci set qos.wan.classgroup='Default'
uci set qos.wan.enabled='1'
uci set qos.wan.upload='1000'    # kbits
uci set qos.wan.download='5000'  # kbits

# Priority rules
uci set qos.@classify[0]=classify
uci set qos.@classify[0].target='Priority'
uci set qos.@classify[0].ports='22,53'
uci set qos.@classify[0].proto='tcp'

uci commit qos
/etc/init.d/qos enable
/etc/init.d/qos start
```

### Routing Protocols

**Static routing (see Network Configuration section)**

**Dynamic routing with BIRD:**
```bash
# Install BIRD
opkg update
opkg install bird2 bird2c

# Configure BIRD (/etc/bird.conf)
cat > /etc/bird.conf << EOF
router id 192.168.1.1;

protocol device {
}

protocol direct {
    ipv4;
}

protocol kernel {
    ipv4 {
        export all;
    };
}

protocol static {
    ipv4;
}

protocol ospf v2 {
    ipv4 {
        import all;
        export all;
    };

    area 0 {
        interface "br-lan" {
            type broadcast;
        };
    };
}
EOF

# Start BIRD
/etc/init.d/bird4 enable
/etc/init.d/bird4 start

# Check status
birdc show protocols
birdc show route
```

---

## Building OpenWRT

### Build System Setup

**Install build dependencies (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install build-essential clang flex bison g++ gawk \
  gcc-multilib g++-multilib gettext git libncurses-dev libssl-dev \
  python3-distutils rsync unzip zlib1g-dev file wget
```

**Clone OpenWRT source:**
```bash
git clone https://git.openwrt.org/openwrt/openwrt.git
cd openwrt

# Checkout stable branch
git checkout v23.05.3

# Update feeds
./scripts/feeds update -a
./scripts/feeds install -a
```

### Building Custom Images

**Configure build:**
```bash
# Start menuconfig
make menuconfig

# Key settings:
# - Target System: (select your device architecture)
# - Subtarget: (select specific target)
# - Target Profile: (select your device)
# - Target Images: (squashfs, ext4, etc.)
# - Select packages to include (with <Y> or <M>)
```

**Build firmware:**
```bash
# Download source packages
make download

# Build (use number of CPU cores + 1)
make -j$(nproc) V=s

# Output in bin/targets/<target>/<subtarget>/
```

**Incremental builds:**
```bash
# Clean kernel
make target/linux/clean

# Clean package
make package/<name>/clean

# Full clean
make clean

# Distribution clean (removes all)
make distclean
```

**Build single package:**
```bash
# Build package
make package/<name>/compile V=s

# Package will be in bin/packages/<arch>/
```

### Package Development

**Create package structure:**
```bash
# Create package directory
mkdir -p package/mypackage/src

# Makefile
cat > package/mypackage/Makefile << 'EOF'
include $(TOPDIR)/rules.mk

PKG_NAME:=mypackage
PKG_VERSION:=1.0
PKG_RELEASE:=1

include $(INCLUDE_DIR)/package.mk

define Package/mypackage
  SECTION:=utils
  CATEGORY:=Utilities
  TITLE:=My custom package
  DEPENDS:=+libc
endef

define Package/mypackage/description
  Description of my package
endef

define Build/Compile
	$(TARGET_CC) $(TARGET_CFLAGS) -o $(PKG_BUILD_DIR)/myapp \
		$(PKG_BUILD_DIR)/main.c
endef

define Package/mypackage/install
	$(INSTALL_DIR) $(1)/usr/bin
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/myapp $(1)/usr/bin/
endef

$(eval $(call BuildPackage,mypackage))
EOF

# Source code
cat > package/mypackage/src/main.c << 'EOF'
#include <stdio.h>

int main() {
    printf("Hello from OpenWRT!\n");
    return 0;
}
EOF

# Build package
make package/mypackage/compile V=s
```

**Install package on router:**
```bash
# Copy to router
scp bin/packages/*/base/mypackage_*.ipk root@192.168.1.1:/tmp/

# Install on router
ssh root@192.168.1.1
opkg install /tmp/mypackage_*.ipk
```

### Kernel Configuration

**Configure kernel:**
```bash
# Kernel menuconfig
make kernel_menuconfig

# Save configuration
# It will be saved in target/linux/<target>/config-*
```

**Add kernel module:**
```bash
# Create package/kernel/mymodule/Makefile
cat > package/kernel/mymodule/Makefile << 'EOF'
include $(TOPDIR)/rules.mk
include $(INCLUDE_DIR)/kernel.mk

PKG_NAME:=mymodule
PKG_RELEASE:=1

include $(INCLUDE_DIR)/package.mk

define KernelPackage/mymodule
  SUBMENU:=Other modules
  TITLE:=My kernel module
  FILES:=$(PKG_BUILD_DIR)/mymodule.ko
  AUTOLOAD:=$(call AutoLoad,50,mymodule,1)
endef

define KernelPackage/mymodule/description
  My custom kernel module
endef

define Build/Compile
	$(MAKE) -C "$(LINUX_DIR)" \
		M="$(PKG_BUILD_DIR)" \
		modules
endef

$(eval $(call KernelPackage,mymodule))
EOF
```

---

## Advanced Topics

### Hotplug System

**Hotplug scripts location: /etc/hotplug.d/**

**Network hotplug:**
```bash
# /etc/hotplug.d/iface/99-custom
#!/bin/sh

if [ "$ACTION" = "ifup" ] && [ "$INTERFACE" = "wan" ]; then
    logger "WAN interface came up"
    # Execute custom commands
fi

if [ "$ACTION" = "ifdown" ] && [ "$INTERFACE" = "wan" ]; then
    logger "WAN interface went down"
fi
```

**USB hotplug:**
```bash
# /etc/hotplug.d/usb/20-usb-storage
#!/bin/sh

if [ "$ACTION" = "add" ] && [ "$PRODUCT" = "specific/product/id" ]; then
    logger "USB device plugged in"
    # Mount, notify, etc.
fi
```

**Button hotplug:**
```bash
# /etc/hotplug.d/button/00-button
#!/bin/sh

if [ "$BUTTON" = "reset" ] && [ "$ACTION" = "pressed" ]; then
    if [ "$SEEN" -gt 5 ]; then
        logger "Reset button held for $SEEN seconds"
        # Perform action
    fi
fi
```

### Custom Scripts

**Init script template:**
```bash
# /etc/init.d/myservice
#!/bin/sh /etc/rc.common

START=95
STOP=10

USE_PROCD=1

start_service() {
    procd_open_instance
    procd_set_param command /usr/bin/myapp --option value
    procd_set_param respawn
    procd_set_param stdout 1
    procd_set_param stderr 1
    procd_close_instance
}

stop_service() {
    # Cleanup if needed
    :
}

reload_service() {
    # Reload config
    stop
    start
}
```

**Enable and control:**
```bash
chmod +x /etc/init.d/myservice
/etc/init.d/myservice enable
/etc/init.d/myservice start
/etc/init.d/myservice status
```

**Cron jobs:**
```bash
# Edit crontab
crontab -e

# Example: Reboot daily at 3 AM
0 3 * * * /sbin/reboot

# Example: Update DDNS every hour
0 * * * * /usr/lib/ddns/dynamic_dns_updater.sh
```

### Network Namespaces

**Create namespace for VPN:**
```bash
# Load kernel module
modprobe ip_nat_pptp

# Create namespace
ip netns add vpn

# Move interface to namespace
ip link set dev tun0 netns vpn

# Execute in namespace
ip netns exec vpn ip addr
ip netns exec vpn ip route add default dev tun0

# Run application in namespace
ip netns exec vpn transmission-daemon
```

### Performance Tuning

**Kernel network tuning:**
```bash
# /etc/sysctl.conf or /etc/sysctl.d/99-custom.conf
cat >> /etc/sysctl.d/99-custom.conf << EOF
# Increase network buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Enable TCP Fast Open
net.ipv4.tcp_fastopen = 3

# Increase connection tracking table size
net.netfilter.nf_conntrack_max = 65536

# Disable IPv6 (if not used)
net.ipv6.conf.all.disable_ipv6 = 1
EOF

# Apply changes
sysctl -p /etc/sysctl.d/99-custom.conf
```

**IRQ balancing:**
```bash
# Check IRQ assignments
cat /proc/interrupts

# Set IRQ affinity (example for eth0)
echo 2 > /proc/irq/$(awk '/eth0/ {print $1}' /proc/interrupts | tr -d ':')/smp_affinity
```

**Hardware offloading:**
```bash
# Enable software flow offloading
uci set firewall.@defaults[0].flow_offloading='1'

# Enable hardware flow offloading (if supported)
uci set firewall.@defaults[0].flow_offloading_hw='1'

uci commit firewall
/etc/init.d/firewall reload
```

---

## Troubleshooting

### Failsafe Mode

**Enter failsafe mode:**
1. Reboot router
2. When power LED starts flashing, rapidly press any button or key
3. Router will enter failsafe with IP 192.168.1.1

**Connect to failsafe:**
```bash
# Set PC IP to 192.168.1.2
sudo ip addr add 192.168.1.2/24 dev eth0

# Telnet to router
telnet 192.168.1.1

# Mount root filesystem
mount_root

# Make changes
vi /etc/config/network

# Reboot
reboot -f
```

**Reset to defaults in failsafe:**
```bash
# Factory reset
firstboot

# Reboot
reboot -f
```

### Recovery Options

**Reset via button:**
```bash
# Hold reset button during boot for 10+ seconds
# This will trigger firstboot on next boot
```

**TFTP recovery:**
```bash
# Most devices support TFTP recovery
# 1. Set PC IP to 192.168.1.10
# 2. Start TFTP server
# 3. Place firmware in TFTP root named appropriately
# 4. Power on router while holding reset
```

**Serial console recovery:**
```bash
# Connect serial adapter (115200 8N1)
screen /dev/ttyUSB0 115200

# Interrupt boot (press key when prompted)
# U-Boot commands available
# Can tftp boot, modify uboot env, etc.
```

**Recover from bad flash:**
```bash
# If still accessible via SSH
sysupgrade -n /tmp/openwrt-sysupgrade.bin

# Force sysupgrade
sysupgrade -F -n /tmp/firmware.bin

# From failsafe
mtd -r write /tmp/firmware.bin firmware
```

### Network Debugging

**Check interface status:**
```bash
# Netifd status
ubus call network.interface dump

# Interface details
ifstatus lan
ifstatus wan

# Physical interfaces
ip addr show
ip link show

# Wireless status
wifi status
iw dev wlan0 info
iw dev wlan0 station dump
```

**Connection tracking:**
```bash
# Show active connections
cat /proc/net/nf_conntrack

# Connection tracking stats
cat /proc/sys/net/netfilter/nf_conntrack_count
cat /proc/sys/net/netfilter/nf_conntrack_max

# Increase conntrack table
echo 65536 > /proc/sys/net/netfilter/nf_conntrack_max
```

**Packet capture:**
```bash
# Install tcpdump
opkg update
opkg install tcpdump

# Capture on interface
tcpdump -i br-lan -n

# Capture to file
tcpdump -i eth1 -w /tmp/capture.pcap

# Filter specific traffic
tcpdump -i any port 80 -n
tcpdump -i any host 192.168.1.100 -n
```

**Firewall debugging:**
```bash
# Check nftables rules (fw4)
nft list ruleset

# Check iptables rules (fw3)
iptables -L -v -n
iptables -t nat -L -v -n

# Enable firewall logging
uci set firewall.@defaults[0].input='REJECT'
uci set firewall.@defaults[0].output='ACCEPT'
uci set firewall.@defaults[0].forward='REJECT'
uci commit firewall
/etc/init.d/firewall reload

# Monitor logs
logread -f | grep firewall
```

**DNS debugging:**
```bash
# Test DNS resolution
nslookup google.com

# Check dnsmasq
ps | grep dnsmasq
netstat -ln | grep :53

# Dnsmasq logs
logread | grep dnsmasq

# Query local dnsmasq
nslookup google.com 127.0.0.1
```

### Log Analysis

**System logs:**
```bash
# View logs
logread

# Follow logs
logread -f

# Filter logs
logread | grep error
logread | grep kernel

# Clear logs
logread -c
```

**Service-specific logs:**
```bash
# Network changes
logread | grep netifd

# Wireless
logread | grep hostapd
logread | grep wpa_supplicant

# Firewall
logread | grep firewall

# DHCP
logread | grep dnsmasq
```

**Kernel messages:**
```bash
# Kernel ring buffer
dmesg

# Follow kernel messages
dmesg -w

# Clear kernel buffer
dmesg -c
```

**Enable debugging:**
```bash
# Verbose network debugging
uci set network.@globals[0].ula_prefix='auto'
uci commit network

# Increase log level
uci set system.@system[0].cronloglevel='8'
uci commit system

# Enable firewall logging for specific rule
uci set firewall.@rule[0].enabled='1'
uci set firewall.@rule[0].log='1'
uci set firewall.@rule[0].log_limit='10/minute'
uci commit firewall
```

---

## Best Practices

**1. Configuration backups:**
```bash
# Backup current config
sysupgrade -b /tmp/backup-$(date +%F).tar.gz

# Download backup
scp root@192.168.1.1:/tmp/backup-*.tar.gz ./

# Restore backup
sysupgrade -r /tmp/backup.tar.gz
```

**2. Keep system updated:**
```bash
# Regular updates
opkg update
opkg list-upgradable
opkg upgrade <package>

# Firmware updates (check release notes first!)
# Download appropriate sysupgrade image
sysupgrade -v /tmp/openwrt-sysupgrade.bin

# Keep settings
sysupgrade -v /tmp/openwrt-sysupgrade.bin

# Fresh install
sysupgrade -n /tmp/openwrt-sysupgrade.bin
```

**3. Security hardening:**
```bash
# Disable WAN SSH access
uci delete firewall.wan_ssh
uci commit firewall

# Change SSH port
uci set dropbear.@dropbear[0].Port='2222'
uci commit dropbear
/etc/init.d/dropbear restart

# Use SSH keys instead of passwords
cat ~/.ssh/id_rsa.pub | ssh root@192.168.1.1 'cat >> /etc/dropbear/authorized_keys'
uci set dropbear.@dropbear[0].PasswordAuth='0'
uci set dropbear.@dropbear[0].RootPasswordAuth='0'
uci commit dropbear
/etc/init.d/dropbear restart

# Disable WPS
uci set wireless.default_radio0.wps_pushbutton='0'
uci commit wireless
wifi reload

# Use strong WiFi encryption
uci set wireless.default_radio0.encryption='sae'  # WPA3
# Or at minimum:
uci set wireless.default_radio0.encryption='psk2+ccmp'  # WPA2
```

**4. Use UCI for configuration:**
- Provides validation and consistency
- Easy to script and automate
- Changes are atomic (commit/revert)
- Configuration survives upgrades

**5. Monitor system resources:**
```bash
# Check memory
free

# Check disk space
df -h

# Check processes
top
ps

# Install monitoring
opkg install collectd luci-app-statistics
```

**6. Document changes:**
```bash
# Keep notes in /etc/config/notes (custom file)
# Or use comments in init scripts
# Document firewall rules with descriptive names
```

**7. Test before committing:**
```bash
# Make changes
uci set network.lan.ipaddr='192.168.2.1'

# Test without committing
# If it breaks, revert:
uci revert network

# If it works:
uci commit network
```

**8. Use VLANs for network segmentation:**
- Separate guest, IoT, management networks
- Improved security and traffic management
- Better control over inter-VLAN routing

**9. Implement proper QoS:**
- Use SQM for bufferbloat control
- Prioritize important traffic
- Set appropriate bandwidth limits

**10. Regular maintenance:**
```bash
# Clean up old packages
opkg remove $(opkg list-installed | awk '{print $1}' | grep -v '^base-files$')

# Check for configuration issues
uci show | grep -i error

# Review logs periodically
logread | grep -i error
logread | grep -i fail
```

---

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `uci show` | Show all configuration |
| `uci commit` | Save pending changes |
| `opkg update` | Update package lists |
| `opkg install <pkg>` | Install package |
| `wifi reload` | Reload wireless config |
| `/etc/init.d/network reload` | Reload network config |
| `/etc/init.d/firewall reload` | Reload firewall |
| `logread -f` | Follow system logs |
| `ifstatus <iface>` | Show interface status |
| `sysupgrade -b <file>` | Backup configuration |

### Configuration Files

| File | Purpose |
|------|---------|
| `/etc/config/network` | Network interfaces, VLANs, routes |
| `/etc/config/wireless` | WiFi configuration |
| `/etc/config/firewall` | Firewall zones, rules |
| `/etc/config/dhcp` | DHCP and DNS settings |
| `/etc/config/system` | System settings (hostname, time, LEDs) |
| `/etc/config/dropbear` | SSH server configuration |
| `/etc/config/uhttpd` | Web server configuration |
| `/etc/rc.local` | Custom startup commands |

### Network Tools

| Command | Description |
|---------|-------------|
| `ip addr show` | Show IP addresses |
| `ip route show` | Show routing table |
| `ip link show` | Show network interfaces |
| `brctl show` | Show bridges |
| `iwinfo` | Show wireless info |
| `ping <host>` | Test connectivity |
| `traceroute <host>` | Trace route to host |
| `nslookup <domain>` | DNS lookup |
| `tcpdump -i <iface>` | Packet capture |

### Service Management

| Command | Description |
|---------|-------------|
| `/etc/init.d/<service> start` | Start service |
| `/etc/init.d/<service> stop` | Stop service |
| `/etc/init.d/<service> restart` | Restart service |
| `/etc/init.d/<service> reload` | Reload config |
| `/etc/init.d/<service> enable` | Enable at boot |
| `/etc/init.d/<service> disable` | Disable at boot |
| `ps` | List running processes |
| `top` | Process monitor |
| `killall <process>` | Kill process by name |

### File Locations

| Path | Contents |
|------|----------|
| `/etc/config/` | UCI configuration files |
| `/etc/init.d/` | Init scripts |
| `/etc/hotplug.d/` | Hotplug scripts |
| `/tmp/` | Temporary files (RAM) |
| `/overlay/` | Writable overlay filesystem |
| `/rom/` | Read-only base system |
| `/www/` | Web server files (LuCI) |
| `/lib/firmware/` | Device firmware files |

### Recovery Commands

| Action | Command/Method |
|--------|----------------|
| Enter failsafe | Press button during boot flash |
| Factory reset | `firstboot && reboot` |
| Reset WiFi password | `wifi down && uci delete wireless.@wifi-iface[0].key && wifi up` |
| Reset root password | In failsafe: `passwd` |
| Emergency flash | TFTP during boot |

### See Also

- [networking.md](networking.md) - Linux networking fundamentals
- [iptables.md](iptables.md) - iptables firewall configuration
- [nftables.md](nftables.md) - nftables firewall configuration
- [systemd.md](systemd.md) - systemd service management
- [kernel.md](kernel.md) - Linux kernel internals
- [cross_compilation.md](cross_compilation.md) - Cross-compilation techniques
- [yocto.md](yocto.md) - Yocto embedded Linux build system

---

**Official Resources:**
- OpenWRT Wiki: https://openwrt.org/docs/start
- Downloads: https://downloads.openwrt.org/
- Forum: https://forum.openwrt.org/
- Source Code: https://git.openwrt.org/
