# tcpdump

tcpdump is a powerful command-line packet analyzer tool for Unix-like operating systems. It allows users to capture, filter, and analyze network traffic in real-time or save it for later analysis. It's one of the most fundamental and widely-used tools for network troubleshooting, security analysis, and protocol debugging.

## Overview

tcpdump was originally written by Van Jacobson, Craig Leres, and Steven McCanne at the Lawrence Berkeley National Laboratory. It has been actively maintained and enhanced since 1988, making it one of the longest-standing network diagnostic tools.

**Key Features:**
- Real-time packet capture and analysis
- Powerful filtering using BPF (Berkeley Packet Filter) syntax
- Support for all major network protocols
- Capture to file for offline analysis
- Minimal resource overhead
- Available on virtually all Unix-like systems
- Precise timestamp information
- Flexible output formats
- Integration with other analysis tools (Wireshark, tshark)
- IPv4 and IPv6 support

**Common Use Cases:**
- Network troubleshooting and diagnostics
- Security analysis and incident response
- Protocol debugging and development
- Performance analysis and optimization
- Compliance monitoring and auditing
- Malware traffic analysis
- Network forensics
- Application behavior analysis
- Quality of Service (QoS) verification
- Bandwidth utilization monitoring

## Legal and Ethical Considerations

**IMPORTANT:** Capturing network traffic requires proper authorization and raises significant privacy and legal concerns. Unauthorized packet capture may be illegal in your jurisdiction and violate privacy laws.

**Best Practices:**
- Only capture traffic on networks you own or have explicit written permission to monitor
- Understand and comply with local privacy, wiretapping, and surveillance laws
- Inform users when monitoring may occur (where required by law or policy)
- Minimize captured data to only what's necessary for your purpose
- Secure captured files as they may contain sensitive information (passwords, personal data, confidential communications)
- Use encryption when transferring capture files
- Implement and follow data retention policies
- Redact or sanitize sensitive information before sharing captures
- Follow organizational security and privacy policies
- Be aware that packets may contain passwords, authentication tokens, personal data, and confidential business information
- Consider the ethical implications of monitoring, even when legally permitted
- Document the scope and purpose of all capture activities

## Basic Concepts

### How tcpdump Works

tcpdump operates at the data link layer, capturing packets directly from network interfaces using libpcap (Packet Capture library):

1. **Interface Selection** - Choose which network interface to monitor
2. **Filter Compilation** - BPF filter is compiled into bytecode
3. **Kernel-Level Filtering** - Kernel applies filter before copying packets to userspace
4. **Packet Capture** - Matching packets are captured via libpcap
5. **Packet Processing** - tcpdump decodes and displays or stores packets
6. **Output Generation** - Formatted output to screen or file

### Berkeley Packet Filter (BPF)

BPF is a powerful filtering language used by tcpdump:

- **Kernel-level filtering** - Filters applied in kernel space before packets reach userspace
- **Efficient** - Minimal CPU and memory overhead
- **Expressive** - Rich syntax for complex filtering conditions
- **Portable** - Works across different Unix-like operating systems
- **Compiled** - Filter expressions compiled to bytecode for performance

### Packet Capture Levels

tcpdump can capture at different levels:

1. **Link Layer** (Layer 2) - Ethernet frames, MAC addresses
2. **Network Layer** (Layer 3) - IP packets, routing information
3. **Transport Layer** (Layer 4) - TCP/UDP segments
4. **Application Layer** (Layer 7) - Protocol-specific data (HTTP, DNS, etc.)

### Capture File Formats

- **pcap** - Standard packet capture format (libpcap)
- **pcapng** - Next-generation pcap format (more features, metadata)
- Compatible with Wireshark, tshark, and other analysis tools

### Snapshot Length (snaplen)

The snapshot length determines how much of each packet to capture:

- Default: 262144 bytes (256 KB) on modern systems
- Full packets: Use `-s 0` or `-s 65535`
- Headers only: Use `-s 128` (saves space, faster)
- Optimal for most uses: `-s 0` (capture full packets)

## Installation

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install tcpdump

# RHEL/CentOS/Fedora
sudo yum install tcpdump
# or
sudo dnf install tcpdump

# macOS
# tcpdump is pre-installed on macOS
# To get the latest version:
brew install tcpdump

# Verify installation
tcpdump --version

# Check available interfaces
tcpdump -D

# Test capture (requires root/sudo)
sudo tcpdump -i eth0 -c 5
```

### Permission Setup

tcpdump requires root privileges or special capabilities to capture packets:

```bash
# Run with sudo (most common)
sudo tcpdump -i eth0

# Linux: Grant capture permissions to specific users (more secure)
# Add user to group with capture permissions
sudo groupadd pcap
sudo usermod -a -G pcap $USER
sudo chgrp pcap /usr/sbin/tcpdump
sudo chmod 750 /usr/sbin/tcpdump
sudo setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump

# Verify capabilities
getcap /usr/sbin/tcpdump

# Log out and back in for group changes to take effect
newgrp pcap

# macOS: Run as root or with sudo
sudo tcpdump -i en0

# Verify permissions work
tcpdump -D  # Should list interfaces without error
```

## Basic Operations

### Listing Network Interfaces

```bash
# List all available interfaces
tcpdump -D
sudo tcpdump -D

# Example output:
# 1.eth0 [Up, Running]
# 2.wlan0 [Up, Running]
# 3.lo [Up, Running, Loopback]
# 4.any (Pseudo-device that captures on all interfaces) [Up, Running]
# 5.docker0 [Up, Running]

# On macOS
tcpdump -D
# Example output:
# 1.en0 [Up, Running]
# 2.en1
# 3.bridge0
# 4.lo0 [Up, Running, Loopback]
```

### Basic Capture

```bash
# Capture on default interface
sudo tcpdump

# Capture on specific interface
sudo tcpdump -i eth0
sudo tcpdump -i wlan0

# Capture on all interfaces (Linux)
sudo tcpdump -i any

# Capture N packets and stop
sudo tcpdump -i eth0 -c 10        # Capture 10 packets
sudo tcpdump -i eth0 -c 100       # Capture 100 packets

# Capture without resolving hostnames (faster)
sudo tcpdump -i eth0 -n

# Capture without resolving hostnames or ports
sudo tcpdump -i eth0 -nn

# Capture with timestamps
sudo tcpdump -i eth0 -tttt        # Human-readable timestamps
sudo tcpdump -i eth0 -ttt         # Delta time since previous packet

# Stop capture with Ctrl+C
```

### Verbosity Levels

```bash
# Normal output (summary)
sudo tcpdump -i eth0

# Verbose (-v)
sudo tcpdump -i eth0 -v

# More verbose (-vv)
sudo tcpdump -i eth0 -vv

# Maximum verbosity (-vvv)
sudo tcpdump -i eth0 -vvv

# Verbosity shows:
# -v: TTL, identification, length, options
# -vv: Additional protocol details
# -vvv: Maximum protocol information
```

### Writing to Files

```bash
# Capture to file
sudo tcpdump -i eth0 -w capture.pcap

# Capture with packet count limit
sudo tcpdump -i eth0 -c 1000 -w capture.pcap

# Capture with file rotation (size-based)
sudo tcpdump -i eth0 -w capture.pcap -C 100  # New file every 100 MB

# Capture with file rotation (count-based)
sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 5  # Keep 5 files max

# Capture and display simultaneously
sudo tcpdump -i eth0 -w capture.pcap -v

# Append to existing file
sudo tcpdump -i eth0 -w capture.pcap -A

# Capture to stdout (pipe to other tools)
sudo tcpdump -i eth0 -w -
```

### Reading from Files

```bash
# Read from pcap file
tcpdump -r capture.pcap

# Read with filtering
tcpdump -r capture.pcap tcp port 80

# Read first N packets
tcpdump -r capture.pcap -c 10

# Read with verbosity
tcpdump -r capture.pcap -v
tcpdump -r capture.pcap -vv

# Read with specific timestamp format
tcpdump -r capture.pcap -tttt

# Read and write filtered packets to new file
tcpdump -r capture.pcap -w filtered.pcap 'tcp port 443'
```

## Capture Filters (BPF Syntax)

Capture filters use Berkeley Packet Filter (BPF) syntax. Filters are applied in the kernel, making them very efficient.

### Host Filters

```bash
# Capture traffic to/from specific host
sudo tcpdump host 192.168.1.1
sudo tcpdump host example.com

# Capture traffic FROM specific host (source)
sudo tcpdump src host 192.168.1.1

# Capture traffic TO specific host (destination)
sudo tcpdump dst host 192.168.1.1

# Multiple hosts
sudo tcpdump host 192.168.1.1 or host 192.168.1.2

# Exclude specific host
sudo tcpdump not host 192.168.1.1

# Traffic between two hosts
sudo tcpdump host 192.168.1.1 and host 192.168.1.2
```

### Network Filters

```bash
# Capture traffic from/to network
sudo tcpdump net 192.168.1.0/24
sudo tcpdump net 10.0.0.0/8

# Source network
sudo tcpdump src net 192.168.0.0/16

# Destination network
sudo tcpdump dst net 10.0.0.0/8

# Exclude network
sudo tcpdump not net 192.168.1.0/24

# Alternative mask notation
sudo tcpdump net 192.168.1.0 mask 255.255.255.0
```

### Port Filters

```bash
# Capture specific port (source or destination)
sudo tcpdump port 80
sudo tcpdump port 443

# Source port
sudo tcpdump src port 80

# Destination port
sudo tcpdump dst port 443

# Port range
sudo tcpdump portrange 8000-9000

# Multiple ports
sudo tcpdump port 80 or port 443
sudo tcpdump 'port 80 or port 443 or port 8080'

# Exclude port
sudo tcpdump not port 22

# Common service ports
sudo tcpdump port http        # Port 80
sudo tcpdump port https       # Port 443
sudo tcpdump port ssh         # Port 22
sudo tcpdump port domain      # Port 53 (DNS)
```

### Protocol Filters

```bash
# TCP traffic only
sudo tcpdump tcp

# UDP traffic only
sudo tcpdump udp

# ICMP traffic only (ping, etc.)
sudo tcpdump icmp

# ARP traffic
sudo tcpdump arp

# IP traffic (IPv4)
sudo tcpdump ip

# IPv6 traffic
sudo tcpdump ip6

# ICMP6 (IPv6 ICMP)
sudo tcpdump icmp6

# Specific protocol with port
sudo tcpdump tcp port 80
sudo tcpdump udp port 53

# Multiple protocols
sudo tcpdump 'tcp or udp'
sudo tcpdump 'icmp or arp'
```

### TCP Flags

```bash
# TCP SYN packets (connection initiation)
sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'

# TCP SYN-ACK packets
sudo tcpdump 'tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack)'

# TCP RST packets (connection reset)
sudo tcpdump 'tcp[tcpflags] & tcp-rst != 0'

# TCP FIN packets (connection close)
sudo tcpdump 'tcp[tcpflags] & tcp-fin != 0'

# TCP PSH packets (push data)
sudo tcpdump 'tcp[tcpflags] & tcp-push != 0'

# TCP ACK packets
sudo tcpdump 'tcp[tcpflags] & tcp-ack != 0'

# TCP URG packets
sudo tcpdump 'tcp[tcpflags] & tcp-urg != 0'

# TCP with no flags (NULL scan)
sudo tcpdump 'tcp[tcpflags] == 0'

# TCP SYN only (not SYN-ACK)
sudo tcpdump 'tcp[tcpflags] == tcp-syn'

# TCP FIN-ACK packets
sudo tcpdump 'tcp[tcpflags] & (tcp-fin|tcp-ack) == (tcp-fin|tcp-ack)'

# Xmas scan (FIN, PSH, URG)
sudo tcpdump 'tcp[tcpflags] & (tcp-fin|tcp-push|tcp-urg) != 0'
```

### Complex Filters

```bash
# Combine host and port
sudo tcpdump 'host 192.168.1.1 and port 80'

# Combine protocol and network
sudo tcpdump 'tcp and net 192.168.1.0/24'

# Multiple conditions with AND
sudo tcpdump 'host 192.168.1.1 and tcp and port 443'

# Multiple conditions with OR
sudo tcpdump 'host 192.168.1.1 or host 192.168.1.2'

# Complex boolean logic (use quotes!)
sudo tcpdump '(host 192.168.1.1 or host 192.168.1.2) and port 80'

# Exclude traffic (NOT)
sudo tcpdump 'not host 192.168.1.1 and not port 22'

# HTTP and HTTPS traffic
sudo tcpdump 'tcp port 80 or tcp port 443'

# DNS traffic (both TCP and UDP)
sudo tcpdump 'port 53'
sudo tcpdump 'tcp port 53 or udp port 53'

# Capture everything except SSH
sudo tcpdump 'not port 22'

# Specific host on specific ports
sudo tcpdump 'host 192.168.1.1 and (port 80 or port 443)'

# Non-local traffic only
sudo tcpdump 'not net 127.0.0.0/8'

# Traffic between two networks
sudo tcpdump 'net 192.168.1.0/24 and net 10.0.0.0/24'
```

### Ethernet and MAC Filters

```bash
# Capture by MAC address
sudo tcpdump ether host 00:11:22:33:44:55

# Source MAC
sudo tcpdump ether src 00:11:22:33:44:55

# Destination MAC
sudo tcpdump ether dst 00:11:22:33:44:55

# Broadcast traffic
sudo tcpdump ether broadcast

# Multicast traffic
sudo tcpdump ether multicast

# Specific EtherType
sudo tcpdump 'ether proto 0x0800'   # IPv4
sudo tcpdump 'ether proto 0x0806'   # ARP
sudo tcpdump 'ether proto 0x86dd'   # IPv6
```

### Packet Size Filters

```bash
# Packets less than size (bytes)
sudo tcpdump less 128

# Packets greater than size
sudo tcpdump greater 1000

# Packets of exact size
sudo tcpdump 'len == 64'

# Large packets (potential MTU issues or jumbograms)
sudo tcpdump greater 1500

# Small packets
sudo tcpdump less 60

# Size range using boolean logic
sudo tcpdump 'greater 100 and less 500'
```

### VLAN Filters

```bash
# Capture VLAN traffic
sudo tcpdump vlan

# Specific VLAN ID
sudo tcpdump 'vlan 100'

# VLAN and protocol
sudo tcpdump 'vlan and tcp'

# VLAN with specific traffic
sudo tcpdump 'vlan 100 and host 192.168.1.1'

# Multiple VLANs
sudo tcpdump 'vlan 100 or vlan 200'
```

### Advanced Protocol Filters

```bash
# HTTP GET requests (looking at payload)
sudo tcpdump -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'

# HTTP POST requests
sudo tcpdump -A 'tcp port 80 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x504f5354'

# DNS queries (QR bit = 0)
sudo tcpdump 'udp port 53 and udp[10] & 0x80 = 0'

# DNS responses (QR bit = 1)
sudo tcpdump 'udp port 53 and udp[10] & 0x80 = 0x80'

# ICMP echo request (ping)
sudo tcpdump 'icmp[icmptype] == icmp-echo'

# ICMP echo reply
sudo tcpdump 'icmp[icmptype] == icmp-echoreply'

# TCP packets with data (not just ACKs)
sudo tcpdump 'tcp[((tcp[12:1] & 0xf0) >> 2):4] != 0'

# IPv4 packets with DF (Don't Fragment) flag
sudo tcpdump 'ip[6] & 0x40 != 0'

# IPv4 fragmented packets
sudo tcpdump 'ip[6:2] & 0x1fff != 0 or ip[6] & 0x20 != 0'
```

## Display Options

### Output Format

```bash
# Default output (packet summary)
sudo tcpdump -i eth0

# ASCII output (-A) - shows packet content as ASCII
sudo tcpdump -A -i eth0

# Hex output (-x) - shows packet in hex
sudo tcpdump -x -i eth0

# Hex and ASCII (-X) - shows both hex and ASCII
sudo tcpdump -X -i eth0

# Hex with link-level header (-xx)
sudo tcpdump -xx -i eth0

# Hex and ASCII with link-level header (-XX)
sudo tcpdump -XX -i eth0

# Quiet output (less protocol information)
sudo tcpdump -q -i eth0
```

### Timestamp Formats

```bash
# No timestamp
sudo tcpdump -t -i eth0

# Absolute timestamp (default)
sudo tcpdump -i eth0

# Delta time (time since previous packet)
sudo tcpdump -ttt -i eth0

# Absolute time with date
sudo tcpdump -tttt -i eth0

# Time since first packet
sudo tcpdump -ttttt -i eth0

# Unix epoch timestamp
sudo tcpdump -tttttt -i eth0

# ISO 8601 format (with microseconds)
sudo tcpdump -ttttt -i eth0
```

### Verbosity and Detail

```bash
# Minimal output (quiet)
sudo tcpdump -q -i eth0

# Normal detail
sudo tcpdump -i eth0

# Verbose
sudo tcpdump -v -i eth0

# Very verbose
sudo tcpdump -vv -i eth0

# Maximum verbosity
sudo tcpdump -vvv -i eth0

# Suppress hostname resolution (-n)
sudo tcpdump -n -i eth0

# Suppress hostname and port resolution (-nn)
sudo tcpdump -nn -i eth0

# Don't suppress protocol names
sudo tcpdump -nnn -i eth0
```

### Line Buffering

```bash
# Line-buffered output (useful for piping)
sudo tcpdump -l -i eth0 | tee capture.log

# Unbuffered output
sudo tcpdump -U -i eth0

# Useful for real-time monitoring:
sudo tcpdump -l -i eth0 | grep "192.168.1.1"
```

### Packet Length Control

```bash
# Set snapshot length (bytes to capture per packet)
sudo tcpdump -s 0 -i eth0          # Capture full packets (recommended)
sudo tcpdump -s 65535 -i eth0      # Capture full packets (older systems)
sudo tcpdump -s 128 -i eth0        # Capture only headers
sudo tcpdump -s 512 -i eth0        # Headers + some payload

# Default on modern tcpdump: 262144 bytes
```

## Advanced Capture Techniques

### Packet Count and Duration

```bash
# Capture specific number of packets
sudo tcpdump -c 100 -i eth0

# Capture for specific duration (using timeout command)
sudo timeout 60 tcpdump -i eth0 -w capture.pcap    # 60 seconds
sudo timeout 5m tcpdump -i eth0 -w capture.pcap    # 5 minutes

# Capture until file size limit (approximate, with -C)
sudo tcpdump -i eth0 -w capture.pcap -C 100        # ~100 MB per file
```

### Ring Buffer Captures

```bash
# Rotate files by size (-C) and keep limited number (-W)
sudo tcpdump -i eth0 -w capture.pcap -C 50 -W 10
# Creates: capture.pcap0, capture.pcap1, ..., capture.pcap9
# Each file ~50 MB, keeps only 10 most recent files

# Time-based rotation (requires external tools)
# Use with timeout and a script:
for i in {1..10}; do
  timeout 60 sudo tcpdump -i eth0 -w capture-$i.pcap
done

# Rotate by size without limit on file count
sudo tcpdump -i eth0 -w capture.pcap -C 100

# Example: Long-term monitoring (24 hours, 1 hour per file)
for hour in {00..23}; do
  timeout 3600 sudo tcpdump -i eth0 -w capture-2024-01-15-${hour}.pcap
done
```

### Buffer Size

```bash
# Set buffer size (in KB) to prevent packet loss
# Larger buffer = less packet loss on busy networks
sudo tcpdump -B 4096 -i eth0 -w capture.pcap      # 4 MB buffer
sudo tcpdump -B 8192 -i eth0 -w capture.pcap      # 8 MB buffer

# Default buffer size varies by OS
# Linux default: typically 2 MB
# Increase for high-traffic capture
```

### Multiple Interface Capture

```bash
# Capture on all interfaces (Linux)
sudo tcpdump -i any -w capture.pcap

# Capture on specific interfaces sequentially
sudo tcpdump -i eth0 -w eth0-capture.pcap &
sudo tcpdump -i wlan0 -w wlan0-capture.pcap &

# Note: On Linux, -i any includes loopback
# To exclude loopback:
sudo tcpdump -i any not host 127.0.0.1 -w capture.pcap
```

### Immediate Mode

```bash
# Immediate mode: deliver packets immediately without buffering
sudo tcpdump -i eth0 --immediate-mode -w capture.pcap

# Useful for:
# - Real-time analysis
# - Low-latency requirements
# - Monitoring applications

# Trade-off: Higher CPU usage, potential performance impact
```

### Packet Direction

```bash
# Inbound packets only
sudo tcpdump -i eth0 -Q in

# Outbound packets only
sudo tcpdump -i eth0 -Q out

# Both directions (default)
sudo tcpdump -i eth0 -Q inout

# Note: Not supported on all systems
```

## Protocol-Specific Capture

### HTTP/HTTPS

```bash
# HTTP traffic (port 80)
sudo tcpdump -i eth0 -A 'tcp port 80'

# HTTP with specific host
sudo tcpdump -i eth0 -A 'tcp port 80 and host example.com'

# HTTP GET requests
sudo tcpdump -i eth0 -A 'tcp port 80 and (tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420)'

# HTTP POST requests
sudo tcpdump -i eth0 -A 'tcp port 80 and (tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x504f5354)'

# HTTPS traffic (port 443) - encrypted content
sudo tcpdump -i eth0 'tcp port 443'

# HTTP on non-standard ports
sudo tcpdump -i eth0 'tcp port 8080 or tcp port 8443'

# Capture HTTP with verbose output
sudo tcpdump -i eth0 -vvv -A 'tcp port 80'

# HTTP traffic to specific path (requires payload inspection)
sudo tcpdump -i eth0 -A 'tcp port 80' | grep -A 10 "GET /api/"
```

### DNS

```bash
# All DNS traffic (UDP and TCP port 53)
sudo tcpdump -i eth0 port 53

# DNS queries only
sudo tcpdump -i eth0 'udp port 53 and udp[10] & 0x80 = 0'

# DNS responses only
sudo tcpdump -i eth0 'udp port 53 and udp[10] & 0x80 = 0x80'

# DNS for specific domain
sudo tcpdump -i eth0 -vvv 'port 53' | grep -i example.com

# DNS with detailed output
sudo tcpdump -i eth0 -vvv -s 0 port 53

# DNS TCP traffic (large responses, zone transfers)
sudo tcpdump -i eth0 'tcp port 53'

# DNS queries to specific server
sudo tcpdump -i eth0 'dst host 8.8.8.8 and port 53'
```

### ICMP (Ping)

```bash
# All ICMP traffic
sudo tcpdump -i eth0 icmp

# ICMP echo requests (ping)
sudo tcpdump -i eth0 'icmp[icmptype] == icmp-echo'

# ICMP echo replies
sudo tcpdump -i eth0 'icmp[icmptype] == icmp-echoreply'

# ICMP unreachable messages
sudo tcpdump -i eth0 'icmp[icmptype] == icmp-unreach'

# ICMP time exceeded (traceroute)
sudo tcpdump -i eth0 'icmp[icmptype] == icmp-timxceed'

# ICMP to/from specific host
sudo tcpdump -i eth0 'icmp and host 192.168.1.1'

# ICMP6 (IPv6 ICMP)
sudo tcpdump -i eth0 icmp6

# Ping packets with details
sudo tcpdump -i eth0 -vv icmp
```

### ARP

```bash
# All ARP traffic
sudo tcpdump -i eth0 arp

# ARP requests
sudo tcpdump -i eth0 'arp[6:2] == 1'

# ARP replies
sudo tcpdump -i eth0 'arp[6:2] == 2'

# ARP for specific IP
sudo tcpdump -i eth0 'arp and host 192.168.1.1'

# ARP with MAC address details
sudo tcpdump -i eth0 -e arp

# Detect ARP spoofing (look for duplicate IPs with different MACs)
sudo tcpdump -i eth0 -e -n arp | grep "tell"

# Gratuitous ARP
sudo tcpdump -i eth0 'arp and arp[24:4] == arp[28:4]'
```

### DHCP

```bash
# All DHCP traffic
sudo tcpdump -i eth0 'port 67 or port 68'

# DHCP Discover
sudo tcpdump -i eth0 -v 'udp port 67 or udp port 68' | grep -i discover

# DHCP Offer
sudo tcpdump -i eth0 -v 'udp port 67 or udp port 68' | grep -i offer

# DHCP Request
sudo tcpdump -i eth0 -v 'udp port 67 or udp port 68' | grep -i request

# DHCP ACK
sudo tcpdump -i eth0 -v 'udp port 67 or udp port 68' | grep -i ack

# DHCP with full details
sudo tcpdump -i eth0 -vvv -s 0 'port 67 or port 68'
```

### TCP Connection Analysis

```bash
# TCP SYN packets (connection attempts)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0'

# TCP three-way handshake (SYN, SYN-ACK, ACK)
sudo tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn|tcp-ack) != 0'

# TCP connection termination (FIN)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-fin != 0'

# TCP resets
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-rst != 0'

# TCP retransmissions (requires analysis)
sudo tcpdump -i eth0 -vvv tcp

# Established connections (ACK flag set, no SYN)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-ack != 0 and tcp[tcpflags] & tcp-syn == 0'
```

### UDP Traffic

```bash
# All UDP traffic
sudo tcpdump -i eth0 udp

# UDP on specific port
sudo tcpdump -i eth0 'udp port 161'    # SNMP

# UDP excluding DNS
sudo tcpdump -i eth0 'udp and not port 53'

# UDP broadcast
sudo tcpdump -i eth0 'udp and dst host 255.255.255.255'

# UDP multicast
sudo tcpdump -i eth0 'udp and dst net 224.0.0.0/4'
```

### SMTP/Email

```bash
# SMTP traffic
sudo tcpdump -i eth0 'port 25 or port 587 or port 465'

# SMTP with payload
sudo tcpdump -i eth0 -A 'port 25'

# IMAP traffic
sudo tcpdump -i eth0 'port 143 or port 993'

# POP3 traffic
sudo tcpdump -i eth0 'port 110 or port 995'
```

### FTP

```bash
# FTP control channel
sudo tcpdump -i eth0 'port 21'

# FTP control with commands
sudo tcpdump -i eth0 -A 'port 21'

# FTP data channel (passive mode ports)
sudo tcpdump -i eth0 'port 20 or portrange 1024-65535'

# FTP active mode
sudo tcpdump -i eth0 'port 20'
```

### SSH

```bash
# SSH traffic
sudo tcpdump -i eth0 'port 22'

# SSH to specific host
sudo tcpdump -i eth0 'tcp port 22 and host 192.168.1.1'

# SSH connection establishment
sudo tcpdump -i eth0 'tcp port 22 and tcp[tcpflags] & tcp-syn != 0'
```

### Database Traffic

```bash
# MySQL/MariaDB
sudo tcpdump -i eth0 'port 3306'

# PostgreSQL
sudo tcpdump -i eth0 'port 5432'

# MongoDB
sudo tcpdump -i eth0 'port 27017'

# Redis
sudo tcpdump -i eth0 'port 6379'

# Microsoft SQL Server
sudo tcpdump -i eth0 'port 1433'
```

### VPN and Tunneling

```bash
# OpenVPN
sudo tcpdump -i eth0 'udp port 1194'

# IPSec (ESP)
sudo tcpdump -i eth0 'esp'

# IPSec (IKE)
sudo tcpdump -i eth0 'udp port 500 or udp port 4500'

# GRE tunnel
sudo tcpdump -i eth0 'proto gre'

# PPTP
sudo tcpdump -i eth0 'tcp port 1723'
```

## Analysis and Post-Processing

### Basic Analysis

```bash
# Count packets by type
tcpdump -r capture.pcap -n | awk '{print $3}' | sort | uniq -c | sort -rn

# Extract unique source IPs
tcpdump -r capture.pcap -n | awk '{print $3}' | cut -d'.' -f1-4 | sort -u

# Extract unique destination IPs
tcpdump -r capture.pcap -n | awk '{print $5}' | cut -d'.' -f1-4 | cut -d':' -f1 | sort -u

# Count packets per second
tcpdump -r capture.pcap -tttt | awk '{print $1, $2}' | cut -d'.' -f1 | uniq -c

# Find most active hosts
tcpdump -r capture.pcap -n | awk '{print $3}' | cut -d'.' -f1-4 | sort | uniq -c | sort -rn | head -10

# Extract all DNS queries
tcpdump -r capture.pcap -n 'port 53' | grep "A?"

# Find long connections (by packet count)
tcpdump -r capture.pcap -n | awk '{print $3, $5}' | sort | uniq -c | sort -rn | head -20
```

### Statistics

```bash
# Basic statistics (packet count)
tcpdump -r capture.pcap | wc -l

# Protocol distribution
tcpdump -r capture.pcap -n | awk '{print $NF}' | sort | uniq -c | sort -rn

# Ports accessed most
tcpdump -r capture.pcap -n | grep -oP '\d+\.\d+\.\d+\.\d+\.\K\d+' | sort | uniq -c | sort -rn | head -20

# Bandwidth by IP (approximation)
tcpdump -r capture.pcap -n | awk '{print $3, $NF}' | grep length | awk '{sum[$1]+=$2} END {for (ip in sum) print ip, sum[ip]}' | sort -k2 -rn
```

### Filtering and Extraction

```bash
# Extract packets to new file with filter
tcpdump -r capture.pcap -w filtered.pcap 'tcp port 443'

# Extract specific time range (requires timestamps)
tcpdump -r capture.pcap -w timerange.pcap \
  '((dst port 80) and (src net 192.168.1.0/24))'

# Extract packets for specific conversation
tcpdump -r capture.pcap -w conversation.pcap \
  '((host 192.168.1.1 and host 192.168.1.2) and port 80)'

# Split capture by protocol
tcpdump -r capture.pcap -w tcp.pcap tcp
tcpdump -r capture.pcap -w udp.pcap udp
tcpdump -r capture.pcap -w icmp.pcap icmp
```

### Combining with Other Tools

```bash
# Pipe to Wireshark/tshark
sudo tcpdump -i eth0 -w - | wireshark -k -i -
sudo tcpdump -i eth0 -w - | tshark -r -

# Pipe to grep for real-time filtering
sudo tcpdump -i eth0 -l | grep "192.168.1.1"

# Pipe to awk for custom processing
sudo tcpdump -i eth0 -n -l | awk '{print $3, $5, $6}'

# Save and analyze with tshark
sudo tcpdump -i eth0 -w capture.pcap
tshark -r capture.pcap -Y "http" -T fields -e ip.src -e http.request.uri

# Use with tcpflow for TCP stream reconstruction
sudo tcpdump -i eth0 -w - | tcpflow -r -

# Use with ssldump for SSL/TLS analysis
sudo tcpdump -i eth0 -w - | ssldump -r -

# Convert to text for analysis
tcpdump -r capture.pcap -n | tee capture.txt

# Parse with custom script
tcpdump -r capture.pcap -n | python3 analyze.py
```

### Time-Based Analysis

```bash
# Show packets with absolute timestamps
tcpdump -r capture.pcap -tttt

# Show packet timing deltas
tcpdump -r capture.pcap -ttt

# Extract packets from specific time
tcpdump -r capture.pcap -tttt | grep "2024-01-15 14:"

# Find gaps in traffic (look for large time deltas)
tcpdump -r capture.pcap -ttt | awk '{if ($1 > 1) print}'
```

## Performance and Optimization

### Reducing Packet Loss

```bash
# Increase buffer size
sudo tcpdump -B 8192 -i eth0 -w capture.pcap

# Use specific interface (not 'any')
sudo tcpdump -i eth0 -w capture.pcap    # Faster than -i any

# Disable name resolution
sudo tcpdump -nn -i eth0 -w capture.pcap

# Use efficient filter
sudo tcpdump -i eth0 'tcp port 80' -w capture.pcap    # Filter in kernel

# Write directly to fast storage
sudo tcpdump -i eth0 -w /dev/shm/capture.pcap    # RAM disk

# Reduce snapshot length for headers only
sudo tcpdump -s 128 -i eth0 -w capture.pcap

# Don't display while capturing
sudo tcpdump -i eth0 -w capture.pcap    # No output to screen

# Use immediate mode carefully (trade-off)
sudo tcpdump -i eth0 --immediate-mode -w capture.pcap
```

### Efficient Filtering

```bash
# Filter in kernel (BPF) rather than userspace
# Good: Kernel filters
sudo tcpdump -i eth0 'tcp port 80' -w capture.pcap

# Less efficient: Capture all, filter later
sudo tcpdump -i eth0 -w capture.pcap
tcpdump -r capture.pcap 'tcp port 80'

# Combine multiple conditions efficiently
sudo tcpdump -i eth0 '(tcp port 80 or tcp port 443) and net 192.168.1.0/24'

# Avoid complex filters if simple ones suffice
# Simple:
sudo tcpdump -i eth0 'port 80'
# Complex (unnecessary):
sudo tcpdump -i eth0 'tcp[2:2] == 80 or tcp[0:2] == 80'
```

### Storage Optimization

```bash
# Rotate files to manage disk space
sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 10

# Compress captures (with external tool)
sudo tcpdump -i eth0 -w - | gzip > capture.pcap.gz

# Read compressed captures
zcat capture.pcap.gz | tcpdump -r -

# Capture only headers (smallest size)
sudo tcpdump -s 96 -i eth0 -w headers.pcap

# Use pcapng format for better compression (with tshark)
sudo tcpdump -i eth0 -w - | tshark -r - -F pcapng -w capture.pcapng
```

### High-Speed Capture

```bash
# Dedicated capture for high-speed networks
# 1. Increase buffer size
# 2. Use specific interface
# 3. Disable name resolution
# 4. Use simple filter
# 5. Write to fast storage (SSD, RAM disk)

sudo tcpdump -B 16384 -i eth0 -nn -s 128 \
  'tcp port 80 or tcp port 443' \
  -w /mnt/fast-ssd/capture.pcap

# Multiple file writers (distribute load)
sudo tcpdump -i eth0 -w capture.pcap -C 500 -W 20 -B 16384

# Use packet capture accelerators (if available)
# Example with PF_RING (requires installation):
# sudo tcpdump -i eth0@1 -w capture.pcap    # PF_RING aware
```

## Common Use Cases and Patterns

### Network Troubleshooting

```bash
# Verify connectivity between hosts
sudo tcpdump -i eth0 'host 192.168.1.1 and host 192.168.1.2'

# Check if traffic reaches interface
sudo tcpdump -i eth0 -c 10 'host 8.8.8.8'

# Analyze TCP retransmissions (look for duplicate SEQ numbers)
sudo tcpdump -i eth0 -vvv -nn 'tcp and host 192.168.1.1'

# Check DNS resolution
sudo tcpdump -i eth0 -vvv 'port 53 and host 8.8.8.8'

# Verify routing (ICMP redirects)
sudo tcpdump -i eth0 'icmp[icmptype] == icmp-redirect'

# Monitor for packet loss (look for retransmissions, resets)
sudo tcpdump -i eth0 -vvv 'tcp' | grep -i "retrans\|reset"

# Check for MTU issues (fragmentation)
sudo tcpdump -i eth0 'ip[6:2] & 0x1fff != 0'

# Verify DHCP issues
sudo tcpdump -i eth0 -vvv 'port 67 or port 68'

# Check for duplicate IP addresses (ARP conflicts)
sudo tcpdump -i eth0 -e arp | grep "is-at"

# Trace path MTU discovery
sudo tcpdump -i eth0 'icmp and icmp[0] == 3 and icmp[1] == 4'
```

### Security Analysis

```bash
# Detect port scanning (many SYN packets)
sudo tcpdump -i eth0 -nn 'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0'

# Monitor for ARP spoofing
sudo tcpdump -i eth0 -e -n arp | awk '{print $12, $13}'

# Detect SYN flood attacks
sudo tcpdump -i eth0 -c 100 'tcp[tcpflags] & tcp-syn != 0' | wc -l

# Find failed connection attempts (RST packets)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-rst != 0'

# Detect suspicious DNS activity
sudo tcpdump -i eth0 -vvv -nn 'port 53' | grep -E "NXDomain|FormErr"

# Monitor for unusual traffic patterns
sudo tcpdump -i eth0 -nn 'tcp[tcpflags] == 0x00'    # NULL scan
sudo tcpdump -i eth0 -nn 'tcp[tcpflags] == 0x29'    # Xmas scan

# Capture unencrypted passwords (HTTP Basic Auth)
sudo tcpdump -i eth0 -A -s 0 'tcp port 80' | grep -i "authorization:"

# Monitor for malware beaconing (regular intervals)
sudo tcpdump -i eth0 -tttt 'dst host suspicious-ip'

# Detect MAC flooding
sudo tcpdump -i eth0 -e | awk '{print $2}' | sort | uniq -c | sort -rn

# Find plaintext protocols that should be encrypted
sudo tcpdump -i eth0 -A 'tcp port 23 or tcp port 21 or tcp port 110'
```

### Application Debugging

```bash
# Debug HTTP API calls
sudo tcpdump -i eth0 -A 'tcp port 80 and host api.example.com'

# Monitor database connections
sudo tcpdump -i eth0 'tcp port 3306 and host db-server'

# Debug web application (HTTP + HTTPS SYN)
sudo tcpdump -i eth0 '(tcp port 80) or (tcp port 443 and tcp[tcpflags] & tcp-syn != 0)'

# Monitor WebSocket connections
sudo tcpdump -i eth0 -A 'tcp port 80' | grep -i "upgrade: websocket"

# Debug REST API (with JSON payloads)
sudo tcpdump -i eth0 -A 'tcp port 80' | grep -A 20 "POST\|PUT\|PATCH"

# Check application performance (connection establishment time)
sudo tcpdump -i eth0 -ttt 'tcp port 80 and tcp[tcpflags] & (tcp-syn|tcp-ack) != 0'

# Monitor microservices communication
sudo tcpdump -i eth0 'tcp and portrange 8000-9000'

# Debug gRPC (HTTP/2)
sudo tcpdump -i eth0 -A 'tcp port 50051'

# Check session handling (cookies)
sudo tcpdump -i eth0 -A 'tcp port 80' | grep -i "cookie:"

# Monitor API rate limiting (count requests)
sudo tcpdump -i eth0 -l 'tcp port 80' | awk '{print $1}' | uniq -c
```

### Performance Analysis

```bash
# Monitor bandwidth usage (approximation)
sudo tcpdump -i eth0 -n | awk '{print $NF}' | grep length | cut -d: -f2 | \
  awk '{sum+=$1; count++} END {print "Total:", sum, "bytes", "Avg:", sum/count}'

# Identify heavy talkers (most traffic)
sudo tcpdump -i eth0 -nn -c 1000 | awk '{print $3}' | cut -d'.' -f1-4 | sort | uniq -c | sort -rn | head -10

# Check for network congestion (retransmissions)
sudo tcpdump -i eth0 -vvv 'tcp' | grep -i "retrans"

# Analyze connection setup time (SYN to SYN-ACK)
sudo tcpdump -i eth0 -ttt 'tcp[tcpflags] & (tcp-syn|tcp-ack) != 0'

# Monitor packet size distribution
sudo tcpdump -i eth0 -n | awk '{print $NF}' | grep length | cut -d: -f2 | sort -n | uniq -c

# Check for small packet flood (could indicate attack or inefficiency)
sudo tcpdump -i eth0 less 64 | wc -l

# Analyze TCP window sizes
sudo tcpdump -i eth0 -vvv 'tcp' | grep -i "win"

# Monitor packet rate
sudo tcpdump -i eth0 -c 100 -tttt | awk '{print $1, $2}' | cut -d'.' -f1 | uniq -c

# Identify applications by port
sudo tcpdump -i eth0 -nn | awk '{print $5}' | cut -d':' -f2 | sort | uniq -c | sort -rn
```

### VoIP and Real-Time Analysis

```bash
# Monitor SIP signaling
sudo tcpdump -i eth0 -A 'port 5060 or port 5061'

# Capture RTP streams
sudo tcpdump -i eth0 'udp portrange 10000-20000'

# Check for jitter and packet loss (analyze timestamps)
sudo tcpdump -i eth0 -tttt 'udp and portrange 10000-20000'

# Monitor quality (look for RTCP)
sudo tcpdump -i eth0 'udp and portrange 10000-20000' -vvv
```

### Container and Kubernetes Networking

```bash
# Monitor Docker bridge
sudo tcpdump -i docker0

# Monitor Kubernetes pod network
sudo tcpdump -i cni0

# Monitor overlay network traffic
sudo tcpdump -i flannel.1    # or weave, calico, etc.

# Capture traffic for specific container
sudo tcpdump -i docker0 'host container-ip'

# Monitor service mesh traffic (Istio/Envoy)
sudo tcpdump -i eth0 'tcp port 15001 or tcp port 15006'
```

### IPv6 Monitoring

```bash
# All IPv6 traffic
sudo tcpdump -i eth0 ip6

# IPv6 ICMP (ping6, neighbor discovery)
sudo tcpdump -i eth0 icmp6

# IPv6 neighbor discovery
sudo tcpdump -i eth0 'icmp6 and ip6[40] == 135'    # Neighbor Solicitation
sudo tcpdump -i eth0 'icmp6 and ip6[40] == 136'    # Neighbor Advertisement

# IPv6 router advertisements
sudo tcpdump -i eth0 'icmp6 and ip6[40] == 134'

# DHCPv6
sudo tcpdump -i eth0 'udp port 546 or udp port 547'
```

## Integration with Other Tools

### With Wireshark/tshark

```bash
# Capture with tcpdump, analyze with Wireshark
sudo tcpdump -i eth0 -w capture.pcap
wireshark capture.pcap

# Live capture to Wireshark
sudo tcpdump -i eth0 -w - | wireshark -k -i -

# Capture with tcpdump, analyze with tshark
sudo tcpdump -i eth0 -w capture.pcap
tshark -r capture.pcap -Y "http" -T fields -e ip.src -e http.request.uri

# Use tshark display filters on tcpdump captures
tcpdump -r capture.pcap -w - | tshark -r - -Y "tcp.analysis.retransmission"
```

### With tcpflow

```bash
# Capture with tcpdump, reconstruct streams with tcpflow
sudo tcpdump -i eth0 -w - tcp | tcpflow -r -

# Save and process
sudo tcpdump -i eth0 -w capture.pcap
tcpflow -r capture.pcap

# Extract HTTP content
sudo tcpdump -i eth0 -w - 'tcp port 80' | tcpflow -r - -e http
```

### With ngrep

```bash
# Capture with tcpdump for storage, use ngrep for pattern matching
sudo tcpdump -i eth0 -w capture.pcap
ngrep -I capture.pcap "password"

# Real-time pattern matching
sudo tcpdump -i eth0 -w - | ngrep -I - "HTTP"
```

### With Snort/Suricata

```bash
# Capture with tcpdump, analyze with Snort
sudo tcpdump -i eth0 -w capture.pcap
snort -r capture.pcap -c /etc/snort/snort.conf

# Live capture to Snort
sudo tcpdump -i eth0 -w - | snort -r - -c /etc/snort/snort.conf
```

### With Python/Scapy

```python
#!/usr/bin/env python3
from scapy.all import rdpcap, TCP, IP

# Read tcpdump capture file
packets = rdpcap('capture.pcap')

# Analyze packets
for pkt in packets:
    if pkt.haslayer(TCP) and pkt.haslayer(IP):
        print(f"{pkt[IP].src}:{pkt[TCP].sport} -> {pkt[IP].dst}:{pkt[TCP].dport}")
```

### With Shell Scripts

```bash
#!/bin/bash
# Monitor for high traffic and alert

INTERFACE="eth0"
THRESHOLD=1000  # packets per second
LOGFILE="/var/log/traffic-monitor.log"

while true; do
    COUNT=$(timeout 1 sudo tcpdump -i $INTERFACE -c 10000 2>/dev/null | wc -l)

    if [ $COUNT -gt $THRESHOLD ]; then
        echo "$(date): High traffic detected: $COUNT packets/sec" >> $LOGFILE
        # Send alert (email, SMS, etc.)
    fi

    sleep 1
done
```

### With Elasticsearch/Logstash

```bash
# Capture to JSON format (requires processing)
sudo tcpdump -i eth0 -w - | \
  tshark -r - -T json | \
  jq -c '.' | \
  curl -X POST "localhost:9200/packets/_doc" -H 'Content-Type: application/json' --data-binary @-

# Or use specialized tools like packetbeat
```

## Best Practices

### Capture Best Practices

1. **Always use appropriate filters**
   ```bash
   # Filter at capture time, not post-processing
   sudo tcpdump -i eth0 'tcp port 80' -w capture.pcap
   ```

2. **Use non-blocking DNS resolution**
   ```bash
   # Disable name resolution for performance
   sudo tcpdump -nn -i eth0
   ```

3. **Set appropriate snapshot length**
   ```bash
   # Full packets when needed
   sudo tcpdump -s 0 -i eth0 -w capture.pcap

   # Headers only for efficiency
   sudo tcpdump -s 128 -i eth0 -w headers.pcap
   ```

4. **Rotate files for long captures**
   ```bash
   sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 10
   ```

5. **Increase buffer size on busy networks**
   ```bash
   sudo tcpdump -B 8192 -i eth0 -w capture.pcap
   ```

6. **Use specific interface, not 'any'**
   ```bash
   # More efficient
   sudo tcpdump -i eth0 -w capture.pcap

   # Less efficient
   sudo tcpdump -i any -w capture.pcap
   ```

### Analysis Best Practices

1. **Start with high-level overview**
   ```bash
   # Get packet count
   tcpdump -r capture.pcap | wc -l

   # Protocol distribution
   tcpdump -r capture.pcap -n | awk '{print $NF}' | sort | uniq -c
   ```

2. **Use filters to narrow focus**
   ```bash
   # Focus on specific traffic
   tcpdump -r capture.pcap 'tcp port 443 and host 192.168.1.1'
   ```

3. **Export filtered traffic to separate files**
   ```bash
   tcpdump -r capture.pcap -w http.pcap 'tcp port 80'
   tcpdump -r capture.pcap -w dns.pcap 'port 53'
   ```

4. **Combine with specialized analysis tools**
   ```bash
   # Use Wireshark for GUI analysis
   wireshark capture.pcap

   # Use tshark for advanced filtering
   tshark -r capture.pcap -Y "http.request.method == POST"
   ```

### Security Best Practices

1. **Get proper authorization**
   - Written permission for all capture activities
   - Document scope and limitations
   - Follow organizational policies

2. **Protect captured data**
   ```bash
   # Restrict file permissions
   sudo tcpdump -i eth0 -w capture.pcap
   sudo chmod 600 capture.pcap

   # Encrypt sensitive captures
   gpg -c capture.pcap
   ```

3. **Minimize capture scope**
   ```bash
   # Only capture what's needed
   sudo tcpdump -i eth0 'host 192.168.1.1 and port 80'
   ```

4. **Sanitize before sharing**
   ```bash
   # Remove sensitive payload data
   tcpdump -r capture.pcap -w sanitized.pcap -s 128

   # Or use tcprewrite to anonymize IPs
   tcprewrite --infile=capture.pcap --outfile=anon.pcap --seed=12345 --skipl2broadcast
   ```

5. **Implement data retention**
   ```bash
   # Auto-delete old captures
   find /captures -name "*.pcap" -mtime +7 -delete
   ```

### Performance Best Practices

1. **Use kernel-level filtering (BPF)**
   ```bash
   # Efficient - filter in kernel
   sudo tcpdump -i eth0 'tcp port 80'

   # Inefficient - filter in userspace
   sudo tcpdump -i eth0 | grep "port 80"
   ```

2. **Write to fast storage**
   ```bash
   # Use SSD or RAM disk
   sudo tcpdump -i eth0 -w /dev/shm/capture.pcap
   ```

3. **Disable unnecessary output**
   ```bash
   # No screen output when writing to file
   sudo tcpdump -i eth0 -w capture.pcap >/dev/null 2>&1
   ```

4. **Monitor for packet drops**
   ```bash
   # Check statistics when stopping capture
   # Look for "packets dropped by kernel" message
   sudo tcpdump -i eth0 -c 1000
   # Output shows: "1000 packets captured, 0 packets dropped by kernel"
   ```

## Troubleshooting

### Permission Denied Errors

```bash
# Error: "tcpdump: eth0: You don't have permission to capture on that device"

# Solution 1: Use sudo
sudo tcpdump -i eth0

# Solution 2: Set capabilities (Linux)
sudo setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump

# Solution 3: Add user to group (distribution-specific)
sudo usermod -a -G wireshark $USER
# Log out and back in

# Verify permissions
getcap /usr/sbin/tcpdump
```

### Interface Not Found

```bash
# Error: "tcpdump: eth0: No such device exists"

# List available interfaces
tcpdump -D
ip link show

# Check interface name (might be different)
# Modern systems: enp0s3, wlp2s0, etc.
sudo tcpdump -i enp0s3

# Check if interface is up
sudo ip link set eth0 up
```

### No Packets Captured

```bash
# Issue: tcpdump runs but no packets shown

# Check 1: Verify traffic exists
ping 8.8.8.8    # In another terminal

# Check 2: Remove filter temporarily
sudo tcpdump -i eth0 -c 10

# Check 3: Use -i any to capture on all interfaces
sudo tcpdump -i any -c 10

# Check 4: Check for firewall blocking
sudo iptables -L

# Check 5: Verify interface is correct and up
ip addr show
```

### Packet Drops

```bash
# Issue: "packets dropped by kernel" message

# Solution 1: Increase buffer size
sudo tcpdump -B 16384 -i eth0 -w capture.pcap

# Solution 2: Use more specific filter
sudo tcpdump -i eth0 'tcp port 80' -w capture.pcap

# Solution 3: Reduce snapshot length
sudo tcpdump -s 128 -i eth0 -w capture.pcap

# Solution 4: Write to faster storage
sudo tcpdump -i eth0 -w /dev/shm/capture.pcap

# Solution 5: Use file rotation
sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 10

# Check system resources
top
df -h
```

### Name Resolution Slow

```bash
# Issue: tcpdump hangs or is very slow

# Solution: Disable name resolution
sudo tcpdump -nn -i eth0

# Disable only hostname resolution
sudo tcpdump -n -i eth0

# Check DNS configuration
cat /etc/resolv.conf
```

### Can't Read Capture File

```bash
# Error: "tcpdump: bad dump file format"

# Check file type
file capture.pcap

# Try with -r
tcpdump -r capture.pcap

# If compressed, decompress first
gunzip capture.pcap.gz
tcpdump -r capture.pcap

# Check file permissions
ls -l capture.pcap

# Verify file isn't corrupted
tcpdump -r capture.pcap -c 1
```

### Filter Syntax Errors

```bash
# Error: "tcpdump: syntax error in filter expression"

# Issue: Missing quotes around complex filters
# Wrong:
sudo tcpdump -i eth0 tcp port 80 and host 192.168.1.1

# Correct:
sudo tcpdump -i eth0 'tcp port 80 and host 192.168.1.1'

# Issue: Incorrect operator
# Wrong:
sudo tcpdump -i eth0 'port = 80'

# Correct:
sudo tcpdump -i eth0 'port 80'

# Test filter syntax
sudo tcpdump -d 'tcp port 80'    # Show compiled BPF code
```

### Timestamps Issues

```bash
# Issue: Timestamps not showing correctly

# Use explicit timestamp format
sudo tcpdump -tttt -i eth0    # Absolute with date

# Check system time
date

# Synchronize system time
sudo ntpdate pool.ntp.org
# or
sudo timedatectl set-ntp true
```

### High CPU Usage

```bash
# Issue: tcpdump consuming too much CPU

# Solution 1: Use more specific filter
sudo tcpdump -i eth0 'tcp port 80'

# Solution 2: Disable name resolution
sudo tcpdump -nn -i eth0

# Solution 3: Reduce verbosity
sudo tcpdump -q -i eth0

# Solution 4: Don't display output when writing to file
sudo tcpdump -i eth0 -w capture.pcap >/dev/null 2>&1

# Monitor CPU usage
top -p $(pgrep tcpdump)
```

### Storage Issues

```bash
# Issue: Running out of disk space

# Solution 1: Use file rotation
sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 10

# Solution 2: Reduce snapshot length
sudo tcpdump -s 128 -i eth0 -w capture.pcap

# Solution 3: Use specific filter
sudo tcpdump -i eth0 'tcp port 80' -w capture.pcap

# Solution 4: Compress captures
sudo tcpdump -i eth0 -w - | gzip > capture.pcap.gz

# Monitor disk usage
df -h
du -sh /path/to/captures/
```

## Quick Reference

### Essential Commands

```bash
# List interfaces
tcpdump -D

# Basic capture
sudo tcpdump -i eth0

# Capture to file
sudo tcpdump -i eth0 -w capture.pcap

# Read from file
tcpdump -r capture.pcap

# Capture with filter
sudo tcpdump -i eth0 'tcp port 80'

# Capture N packets
sudo tcpdump -i eth0 -c 100

# Verbose output
sudo tcpdump -i eth0 -v

# No name resolution
sudo tcpdump -nn -i eth0

# Hex and ASCII output
sudo tcpdump -X -i eth0

# Timestamps
sudo tcpdump -tttt -i eth0

# Full packets
sudo tcpdump -s 0 -i eth0

# Rotate files
sudo tcpdump -i eth0 -w capture.pcap -C 100 -W 10

# Increase buffer
sudo tcpdump -B 8192 -i eth0

# Read and write with filter
tcpdump -r input.pcap -w output.pcap 'tcp port 443'
```

### Common Filters

| Filter | Description |
|--------|-------------|
| `host 192.168.1.1` | Traffic to/from host |
| `src host 192.168.1.1` | Traffic from host |
| `dst host 192.168.1.1` | Traffic to host |
| `net 192.168.1.0/24` | Traffic to/from network |
| `port 80` | Traffic on port 80 |
| `src port 80` | Source port 80 |
| `dst port 443` | Destination port 443 |
| `portrange 8000-9000` | Port range |
| `tcp` | TCP traffic only |
| `udp` | UDP traffic only |
| `icmp` | ICMP traffic |
| `arp` | ARP traffic |
| `ip6` | IPv6 traffic |
| `tcp port 80` | TCP on port 80 |
| `not port 22` | Exclude port 22 |
| `tcp[tcpflags] & tcp-syn != 0` | TCP SYN packets |
| `ether host 00:11:22:33:44:55` | Specific MAC |
| `ether broadcast` | Broadcast traffic |
| `less 128` | Packets < 128 bytes |
| `greater 1000` | Packets > 1000 bytes |
| `vlan` | VLAN traffic |
| `vlan 100` | Specific VLAN |

### Output Options

| Option | Description |
|--------|-------------|
| `-w file` | Write to file |
| `-r file` | Read from file |
| `-C size` | Rotate files (MB) |
| `-W count` | Keep N files |
| `-c count` | Capture N packets |
| `-n` | No hostname resolution |
| `-nn` | No hostname/port resolution |
| `-v` | Verbose |
| `-vv` | More verbose |
| `-vvv` | Maximum verbosity |
| `-A` | ASCII output |
| `-X` | Hex and ASCII |
| `-x` | Hex output |
| `-XX` | Hex with link-level |
| `-e` | Print link-level header |
| `-q` | Quiet (less protocol info) |
| `-t` | No timestamp |
| `-tttt` | Full timestamp with date |
| `-ttt` | Delta time |
| `-s snaplen` | Snapshot length |
| `-S` | Absolute TCP sequence numbers |
| `-i interface` | Capture interface |
| `-i any` | All interfaces (Linux) |
| `-D` | List interfaces |
| `-B size` | Buffer size (KB) |
| `-l` | Line-buffered output |
| `-U` | Packet-buffered output |

### Timestamp Options

| Option | Format | Example |
|--------|--------|---------|
| (default) | Short | 12:34:56.789012 |
| `-t` | None | (no timestamp) |
| `-tt` | Unix epoch | 1642345678.123456 |
| `-ttt` | Delta | +0.001234 |
| `-tttt` | Full date/time | 2024-01-15 12:34:56.789012 |
| `-ttttt` | Since first packet | 0.001234 |

### Common Protocols and Ports

| Protocol | Port | Filter |
|----------|------|--------|
| HTTP | 80 | `tcp port 80` |
| HTTPS | 443 | `tcp port 443` |
| SSH | 22 | `tcp port 22` |
| FTP | 20, 21 | `tcp port 21` |
| Telnet | 23 | `tcp port 23` |
| SMTP | 25 | `tcp port 25` |
| DNS | 53 | `port 53` |
| DHCP | 67, 68 | `port 67 or port 68` |
| TFTP | 69 | `udp port 69` |
| POP3 | 110 | `tcp port 110` |
| NTP | 123 | `udp port 123` |
| SNMP | 161, 162 | `udp port 161` |
| IMAP | 143 | `tcp port 143` |
| LDAP | 389 | `tcp port 389` |
| SMTPS | 465 | `tcp port 465` |
| IMAPS | 993 | `tcp port 993` |
| POP3S | 995 | `tcp port 995` |
| MySQL | 3306 | `tcp port 3306` |
| RDP | 3389 | `tcp port 3389` |
| PostgreSQL | 5432 | `tcp port 5432` |
| Redis | 6379 | `tcp port 6379` |
| HTTP Alt | 8080 | `tcp port 8080` |
| MongoDB | 27017 | `tcp port 27017` |

### Useful Combinations

```bash
# Web traffic
sudo tcpdump -i eth0 'tcp port 80 or tcp port 443'

# DNS queries and responses
sudo tcpdump -i eth0 -vvv 'port 53'

# SSH connections
sudo tcpdump -i eth0 'tcp port 22'

# Email traffic
sudo tcpdump -i eth0 'port 25 or port 110 or port 143'

# Exclude SSH from capture
sudo tcpdump -i eth0 'not port 22'

# Capture from specific host on web ports
sudo tcpdump -i eth0 'host 192.168.1.1 and (port 80 or port 443)'

# All traffic except local
sudo tcpdump -i eth0 'not net 127.0.0.0/8'

# TCP SYN packets (connection attempts)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'

# ICMP packets (ping, etc.)
sudo tcpdump -i eth0 'icmp'

# ARP requests and replies
sudo tcpdump -i eth0 'arp'

# Broadcast and multicast
sudo tcpdump -i eth0 'broadcast or multicast'

# Large packets (potential issues)
sudo tcpdump -i eth0 'greater 1500'

# Small packets (potential floods)
sudo tcpdump -i eth0 'less 64'
```

## Advanced Filter Examples

### Application-Layer Filters

```bash
# HTTP GET requests
sudo tcpdump -i eth0 -A '(tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420)'

# HTTP POST requests
sudo tcpdump -i eth0 -A '(tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x504f5354)'

# HTTP responses
sudo tcpdump -i eth0 -A '(tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x48545450)'

# SSH version exchange
sudo tcpdump -i eth0 -A 'tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x5353482d'

# FTP commands
sudo tcpdump -i eth0 -A 'tcp port 21 and tcp[((tcp[12:1] & 0xf0) >> 2):4] != 0'

# DNS queries (QR bit = 0)
sudo tcpdump -i eth0 'udp port 53 and udp[10] & 0x80 = 0'

# DNS NXDOMAIN responses
sudo tcpdump -i eth0 'udp port 53 and udp[11] & 0x0f = 3'
```

### IP Header Filters

```bash
# IP packets with options
sudo tcpdump -i eth0 'ip[0] & 0x0f > 5'

# IP packets with DF (Don't Fragment) flag
sudo tcpdump -i eth0 'ip[6] & 0x40 != 0'

# IP fragmented packets
sudo tcpdump -i eth0 'ip[6:2] & 0x1fff != 0 or ip[6] & 0x20 != 0'

# IP packets with specific TTL
sudo tcpdump -i eth0 'ip[8] = 1'        # TTL = 1
sudo tcpdump -i eth0 'ip[8] < 10'       # TTL < 10

# IP packets with specific TOS
sudo tcpdump -i eth0 'ip[1] = 0x10'     # Low delay

# IP packets to multicast addresses
sudo tcpdump -i eth0 'dst net 224.0.0.0/4'

# IP broadcast packets
sudo tcpdump -i eth0 'dst host 255.255.255.255'
```

### TCP Header Filters

```bash
# TCP packets with urgent flag
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-urg != 0'

# TCP packets with ECN flags
sudo tcpdump -i eth0 'tcp[13] & 0x42 != 0'

# TCP packets with specific window size
sudo tcpdump -i eth0 'tcp[14:2] > 1000'

# TCP packets with data (not just ACKs)
sudo tcpdump -i eth0 'tcp[tcpflags] & tcp-push != 0'

# TCP keep-alive packets
sudo tcpdump -i eth0 'tcp[tcpflags] == tcp-ack and len == 54'

# TCP with options
sudo tcpdump -i eth0 'tcp[12] & 0xf0 > 0x50'

# TCP RST-ACK packets
sudo tcpdump -i eth0 'tcp[tcpflags] & (tcp-rst|tcp-ack) == (tcp-rst|tcp-ack)'
```

### ICMP Type Filters

```bash
# ICMP echo request (ping)
sudo tcpdump -i eth0 'icmp[0] = 8'

# ICMP echo reply
sudo tcpdump -i eth0 'icmp[0] = 0'

# ICMP destination unreachable
sudo tcpdump -i eth0 'icmp[0] = 3'

# ICMP time exceeded
sudo tcpdump -i eth0 'icmp[0] = 11'

# ICMP redirect
sudo tcpdump -i eth0 'icmp[0] = 5'

# ICMP port unreachable specifically
sudo tcpdump -i eth0 'icmp[0] = 3 and icmp[1] = 3'

# ICMP network unreachable
sudo tcpdump -i eth0 'icmp[0] = 3 and icmp[1] = 0'

# ICMP fragmentation needed (MTU discovery)
sudo tcpdump -i eth0 'icmp[0] = 3 and icmp[1] = 4'
```

## Scripting Examples

### Automated Monitoring Script

```bash
#!/bin/bash
# Monitor network traffic and alert on specific conditions

INTERFACE="eth0"
ALERT_EMAIL="admin@example.com"
ALERT_PORT="22"
ALERT_THRESHOLD=100

# Create log directory
mkdir -p /var/log/network-monitor

while true; do
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    LOGFILE="/var/log/network-monitor/monitor-$TIMESTAMP.log"

    # Capture 1 minute of traffic
    timeout 60 sudo tcpdump -i $INTERFACE -nn -w - 2>/dev/null | \
    tee >(cat > /tmp/capture-$TIMESTAMP.pcap) | \
    tcpdump -r - -nn "tcp port $ALERT_PORT" 2>/dev/null | \
    wc -l > /tmp/count.txt

    COUNT=$(cat /tmp/count.txt)

    if [ $COUNT -gt $ALERT_THRESHOLD ]; then
        echo "$(date): High traffic on port $ALERT_PORT: $COUNT packets" >> $LOGFILE
        # Send email alert
        echo "Alert: $COUNT packets on port $ALERT_PORT in last minute" | \
            mail -s "Network Alert" $ALERT_EMAIL
    fi

    # Cleanup old logs (keep 7 days)
    find /var/log/network-monitor -name "*.log" -mtime +7 -delete
    find /tmp -name "capture-*.pcap" -mmin +60 -delete

    sleep 60
done
```

### Connection Logger

```bash
#!/bin/bash
# Log all TCP connection attempts

LOG_FILE="/var/log/connections.log"

sudo tcpdump -i eth0 -nn -l \
  'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0' | \
while read line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $line" >> $LOG_FILE
done
```

### Bandwidth Monitor

```bash
#!/bin/bash
# Monitor bandwidth usage per host

INTERFACE="eth0"
DURATION=60  # seconds

echo "Monitoring bandwidth for $DURATION seconds..."

sudo tcpdump -i $INTERFACE -nn -tttt -l 2>/dev/null | \
awk -v duration=$DURATION '
BEGIN {
    start_time = systime()
}
{
    # Extract source IP and packet size
    if ($6 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/) {
        ip = $6
        gsub(/:.*/, "", ip)

        # Extract length
        for (i=1; i<=NF; i++) {
            if ($i == "length") {
                bytes[ip] += $(i+1)
            }
        }
    }

    # Check if duration elapsed
    if (systime() - start_time >= duration) {
        exit
    }
}
END {
    print "\nBandwidth usage by host:"
    print "========================"
    for (ip in bytes) {
        mb = bytes[ip] / 1024 / 1024
        printf "%-15s : %10.2f MB\n", ip, mb
    }
}
'
```

### Suspicious Activity Detector

```bash
#!/bin/bash
# Detect potential network attacks

INTERFACE="eth0"
LOG_FILE="/var/log/security-monitor.log"

echo "Starting security monitoring on $INTERFACE..."

{
    # Monitor for port scanning (many SYN packets)
    sudo tcpdump -i $INTERFACE -nn -l \
      'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0' | \
    awk '{print $3}' | cut -d'.' -f1-4 | \
    uniq -c | \
    awk '$1 > 20 {print "Port scan detected from", $2, "- SYN packets:", $1}' &

    # Monitor for SYN floods
    sudo tcpdump -i $INTERFACE -nn -l -c 1000 \
      'tcp[tcpflags] & tcp-syn != 0' | \
    wc -l | \
    awk '$1 > 100 {print "Potential SYN flood -", $1, "SYN packets in sample"}' &

    # Monitor for ARP spoofing
    sudo tcpdump -i $INTERFACE -e -nn -l arp | \
    awk '{print $8, $10}' | \
    sort | uniq -d | \
    awk '{print "Potential ARP spoofing detected - Duplicate IP/MAC:", $0}' &

    wait
} | while read alert; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $alert" | tee -a $LOG_FILE
done
```

## Conclusion

tcpdump is an essential and powerful tool for network analysis, troubleshooting, and security monitoring. Its efficiency, flexibility, and ubiquity make it indispensable for system administrators, network engineers, and security professionals.

**Key Takeaways:**

- Master BPF syntax for efficient kernel-level filtering
- Use appropriate capture options to minimize packet loss
- Understand protocol layers for effective analysis
- Combine with other tools (Wireshark, tshark) for comprehensive analysis
- Always consider legal and ethical implications
- Protect captured data as it contains sensitive information
- Start with broad captures and progressively narrow focus
- Use file rotation for long-term monitoring
- Leverage scripting for automated monitoring and alerting

**Learning Path:**

1. **Week 1**: Basic capture, simple filters (host, port, protocol)
2. **Week 2**: Reading/writing files, timestamp options, verbosity levels
3. **Week 3**: Complex filters, TCP flags, protocol-specific captures
4. **Week 4**: Advanced BPF syntax, performance optimization
5. **Month 2**: Integration with other tools, scripting, automation
6. **Month 3+**: Advanced analysis techniques, security monitoring, troubleshooting complex issues

**Essential Skills to Develop:**

- BPF filter construction
- Protocol analysis (TCP/IP, HTTP, DNS, etc.)
- Performance optimization techniques
- Security analysis and threat detection
- Troubleshooting methodology
- Scripting and automation
- Integration with analysis tools

**Resources:**

- tcpdump man page: `man tcpdump`
- tcpdump official site: https://www.tcpdump.org/
- BPF syntax reference: `man pcap-filter`
- Wireshark display filter reference (for tshark integration): https://www.wireshark.org/docs/dfref/
- libpcap documentation: https://www.tcpdump.org/pcap.html
- Practice captures: https://wiki.wireshark.org/SampleCaptures

**Remember:**

- Efficiency comes from proper filtering - use BPF filters to reduce captured traffic at the kernel level
- Security is paramount - always get authorization and protect captured data
- Context matters - understand what normal looks like before hunting for anomalies
- Continuous learning - network protocols and attack techniques constantly evolve
- Documentation - always document your capture methodology and findings

tcpdump's power lies in its simplicity and efficiency. While newer tools offer GUIs and advanced features, tcpdump remains the go-to tool for quick network analysis, remote troubleshooting, and scenarios where resources are limited. Master tcpdump, and you'll have a reliable tool for network analysis wherever you go.

Happy capturing!
