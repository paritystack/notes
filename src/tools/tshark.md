# TShark

TShark is the command-line version of Wireshark, the world's most popular network protocol analyzer. It provides powerful packet capture and analysis capabilities directly from the terminal, making it ideal for remote systems, scripting, automation, and situations where a GUI is unavailable or impractical.

## Overview

TShark was developed as part of the Wireshark project (formerly Ethereal) and shares the same robust protocol dissectors and analysis engine. It captures packets from network interfaces or reads saved capture files, providing detailed protocol information and statistics.

**Key Features:**
- Capture live network traffic from interfaces
- Read and analyze pcap/pcapng files
- Rich protocol dissection (supports 3000+ protocols)
- Flexible filtering (capture and display filters)
- Multiple output formats (text, JSON, XML, CSV, PDML, PS)
- Statistical analysis and summaries
- Expert information system
- Follow TCP/UDP/HTTP/TLS streams
- Conversation and endpoint analysis
- Protocol hierarchy statistics
- Scripting and automation friendly
- Remote capture capabilities
- Ring buffer and conditional capture
- Name resolution (MAC, network, transport)

**Common Use Cases:**
- Network troubleshooting and diagnostics
- Security analysis and incident response
- Application protocol debugging
- Performance analysis and optimization
- Compliance and audit logging
- Malware traffic analysis
- VoIP quality monitoring
- IoT device communication analysis
- API debugging and testing
- Network forensics

## Legal and Ethical Considerations

**IMPORTANT:** Capturing network traffic requires proper authorization and raises privacy concerns. Unauthorized packet capture may be illegal in your jurisdiction and violate privacy laws.

**Best Practices:**
- Only capture traffic on networks you own or have explicit written permission to monitor
- Understand and comply with local privacy and wiretapping laws
- Inform users when monitoring may occur (where required by law)
- Minimize captured data to what's necessary
- Secure captured files (they may contain sensitive data)
- Use encryption when transferring capture files
- Implement data retention policies
- Redact sensitive information before sharing captures
- Follow your organization's security and privacy policies
- Be aware that packets may contain passwords, personal data, and confidential information

## Basic Concepts

### How TShark Works

TShark operates in several modes:

1. **Live Capture Mode** - Captures packets from network interfaces in real-time
2. **File Read Mode** - Reads and analyzes previously saved capture files
3. **Pass-through Mode** - Reads from stdin or writes to stdout for piping
4. **Statistics Mode** - Generates statistics without detailed packet display

### Capture Process

The typical capture process:

1. **Interface Selection** - Choose network interface(s) to monitor
2. **Filter Application** - Apply capture filter (BPF) to reduce captured packets
3. **Packet Capture** - Capture packets via libpcap/WinPcap
4. **Protocol Dissection** - Analyze and decode protocol layers
5. **Display Filtering** - Apply display filter to captured packets
6. **Output Generation** - Format and display/save results

### Capture Filters vs Display Filters

Understanding the difference is crucial:

**Capture Filters (BPF - Berkeley Packet Filter):**
- Applied during packet capture
- Filter before packets are saved
- More efficient (reduces storage and memory)
- Limited syntax (traditional tcpdump syntax)
- Cannot filter on dissected protocol fields
- Examples: `tcp port 80`, `host 192.168.1.1`

**Display Filters:**
- Applied after packets are captured
- Filter for display/analysis only
- All packets still captured (unless capture filter used)
- Rich syntax (Wireshark filter language)
- Can filter on any dissected field
- Examples: `http.request.method == "POST"`, `tcp.analysis.retransmission`

### Protocol Dissectors

TShark uses protocol dissectors to decode packets:
- Automatically detects protocols
- Hierarchical dissection (Layer 2 → Layer 7)
- Over 3000 protocol dissectors
- Extensible via Lua plugins
- Heuristic dissectors for ambiguous protocols

### Network Interfaces

Interface types TShark can capture from:
- **Physical interfaces** - Ethernet, Wi-Fi, etc.
- **Virtual interfaces** - VPNs, bridges, tunnels
- **Loopback** - Local traffic (lo, lo0)
- **USB** - USB network devices
- **Bluetooth** - Bluetooth interfaces
- **Pipe interfaces** - Named pipes for remote capture
- **Stdin** - For piped input

### Packet Structure

Typical packet layers TShark dissects:
1. **Frame** - Physical layer information
2. **Link Layer** - Ethernet, Wi-Fi, PPP, etc.
3. **Network Layer** - IP, IPv6, ARP, ICMP
4. **Transport Layer** - TCP, UDP, SCTP
5. **Application Layer** - HTTP, DNS, TLS, SMB, etc.

## Installation

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install tshark

# During installation, allow non-root users to capture packets
# Add your user to wireshark group
sudo usermod -a -G wireshark $USER
# Log out and back in for group changes to take effect

# RHEL/CentOS/Fedora
sudo yum install wireshark
# or
sudo dnf install wireshark

# macOS
brew install wireshark
# This installs both Wireshark GUI and tshark

# Or download from official site
# https://www.wireshark.org/download.html

# Verify installation
tshark --version

# Check available interfaces
tshark -D

# Test capture (requires permissions)
sudo tshark -i eth0 -c 10
```

### Permission Setup

```bash
# Linux: Grant capture permissions to non-root users

# Method 1: Add user to wireshark group (Debian/Ubuntu)
sudo dpkg-reconfigure wireshark-common  # Select "Yes"
sudo usermod -a -G wireshark $USER
newgrp wireshark  # Activate group in current session

# Method 2: Set capabilities on dumpcap
sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/dumpcap

# Method 3: Use sudo (less secure)
sudo tshark -i eth0

# Verify permissions
tshark -D  # Should list interfaces without error

# macOS: Install ChmodBPF (happens during Wireshark installation)
# Check if ChmodBPF is loaded
sudo launchctl list | grep chmod

# Windows: Run as Administrator or install with packet capture privileges
```

## Basic Operations

### Listing Interfaces

```bash
# List all available interfaces
tshark -D

# Example output:
# 1. eth0
# 2. wlan0
# 3. any (Pseudo-device that captures on all interfaces)
# 4. lo (Loopback)

# List interfaces with details
tshark -D --list-interfaces

# List data link types for an interface
tshark -i eth0 -L

# List all interfaces (verbose)
tshark -D -v
```

### Basic Capture

```bash
# Capture on default interface
tshark

# Capture on specific interface
tshark -i eth0
tshark -i wlan0

# Capture on all interfaces
tshark -i any

# Capture N packets and stop
tshark -i eth0 -c 10      # Capture 10 packets
tshark -i eth0 -c 100     # Capture 100 packets

# Capture for specific duration
tshark -i eth0 -a duration:60     # Capture for 60 seconds
tshark -i eth0 -a duration:300    # Capture for 5 minutes

# Capture until file size reached
tshark -i eth0 -a filesize:10000  # Stop at ~10MB

# Capture to file
tshark -i eth0 -w capture.pcap
tshark -i eth0 -w capture.pcapng

# Capture to file with packet count limit
tshark -i eth0 -c 1000 -w capture.pcap

# Capture without displaying (quiet mode)
tshark -i eth0 -w capture.pcap -q

# Capture with snapshot length (truncate packets)
tshark -i eth0 -s 128         # Capture only first 128 bytes
tshark -i eth0 -s 0           # Capture full packets (default)
```

### Reading Capture Files

```bash
# Read from pcap file
tshark -r capture.pcap

# Read first N packets
tshark -r capture.pcap -c 10

# Read specific packet range
tshark -r capture.pcap -c 10   # First 10 packets

# Read and apply display filter
tshark -r capture.pcap -Y "http"
tshark -r capture.pcap -Y "tcp.port == 443"

# Read and get statistics
tshark -r capture.pcap -q -z io,phs

# Read from stdin
cat capture.pcap | tshark -r -

# Read from gzipped file
zcat capture.pcap.gz | tshark -r -
```

### Basic Display Options

```bash
# Verbose output
tshark -i eth0 -V

# Print packet summary (one line per packet)
tshark -i eth0

# Print full packet details
tshark -i eth0 -V

# Print specific fields only
tshark -i eth0 -T fields -e ip.src -e ip.dst -e tcp.port

# Print packet hex dump
tshark -i eth0 -x

# Print packet hex and ASCII
tshark -i eth0 -x -V

# Quiet mode (no output, useful with -w)
tshark -i eth0 -w capture.pcap -q
```

## Capture Filters (BPF Syntax)

Capture filters use Berkeley Packet Filter (BPF) syntax, the same as tcpdump.

### Host Filters

```bash
# Capture traffic to/from specific host
tshark -i eth0 -f "host 192.168.1.1"

# Capture traffic FROM specific host
tshark -i eth0 -f "src host 192.168.1.1"

# Capture traffic TO specific host
tshark -i eth0 -f "dst host 192.168.1.1"

# Capture traffic to/from hostname
tshark -i eth0 -f "host www.example.com"

# Multiple hosts
tshark -i eth0 -f "host 192.168.1.1 or host 192.168.1.2"

# Exclude host
tshark -i eth0 -f "not host 192.168.1.1"
```

### Network Filters

```bash
# Capture traffic from/to network
tshark -i eth0 -f "net 192.168.1.0/24"
tshark -i eth0 -f "net 10.0.0.0/8"

# Source network
tshark -i eth0 -f "src net 192.168.0.0/16"

# Destination network
tshark -i eth0 -f "dst net 10.0.0.0/8"

# Exclude network
tshark -i eth0 -f "not net 192.168.1.0/24"
```

### Port Filters

```bash
# Capture specific port
tshark -i eth0 -f "port 80"
tshark -i eth0 -f "port 443"

# Source port
tshark -i eth0 -f "src port 80"

# Destination port
tshark -i eth0 -f "dst port 443"

# Port range
tshark -i eth0 -f "portrange 8000-9000"

# Multiple ports
tshark -i eth0 -f "port 80 or port 443"
tshark -i eth0 -f "port 80 or port 443 or port 8080"

# Exclude port
tshark -i eth0 -f "not port 22"
```

### Protocol Filters

```bash
# TCP traffic only
tshark -i eth0 -f "tcp"

# UDP traffic only
tshark -i eth0 -f "udp"

# ICMP traffic only
tshark -i eth0 -f "icmp"

# ARP traffic
tshark -i eth0 -f "arp"

# IP traffic (IPv4)
tshark -i eth0 -f "ip"

# IPv6 traffic
tshark -i eth0 -f "ip6"

# Specific protocol with port
tshark -i eth0 -f "tcp port 80"
tshark -i eth0 -f "udp port 53"

# Multiple protocols
tshark -i eth0 -f "tcp or udp"
tshark -i eth0 -f "icmp or arp"
```

### TCP Flags

```bash
# TCP SYN packets
tshark -i eth0 -f "tcp[tcpflags] & tcp-syn != 0"

# TCP SYN-ACK packets
tshark -i eth0 -f "tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack)"

# TCP RST packets
tshark -i eth0 -f "tcp[tcpflags] & tcp-rst != 0"

# TCP FIN packets
tshark -i eth0 -f "tcp[tcpflags] & tcp-fin != 0"

# TCP PSH packets
tshark -i eth0 -f "tcp[tcpflags] & tcp-push != 0"

# TCP with no flags (NULL scan)
tshark -i eth0 -f "tcp[tcpflags] == 0"

# TCP with FIN, PSH, URG (Xmas scan)
tshark -i eth0 -f "tcp[tcpflags] & (tcp-fin|tcp-push|tcp-urg) != 0"
```

### Complex Filters

```bash
# Combine host and port
tshark -i eth0 -f "host 192.168.1.1 and port 80"

# Combine protocol and network
tshark -i eth0 -f "tcp and net 192.168.1.0/24"

# Multiple conditions with AND
tshark -i eth0 -f "host 192.168.1.1 and tcp and port 443"

# Multiple conditions with OR
tshark -i eth0 -f "host 192.168.1.1 or host 192.168.1.2"

# Complex boolean logic
tshark -i eth0 -f "(host 192.168.1.1 or host 192.168.1.2) and port 80"

# Exclude traffic
tshark -i eth0 -f "not host 192.168.1.1 and not port 22"

# HTTP and HTTPS traffic
tshark -i eth0 -f "tcp port 80 or tcp port 443"

# DNS traffic (TCP and UDP)
tshark -i eth0 -f "port 53"
tshark -i eth0 -f "tcp port 53 or udp port 53"

# Capture everything except SSH
tshark -i eth0 -f "not port 22"

# Specific host on specific ports
tshark -i eth0 -f "host 192.168.1.1 and (port 80 or port 443)"

# Non-local traffic only
tshark -i eth0 -f "not net 127.0.0.0/8"
```

### Ethernet and MAC Filters

```bash
# Capture by MAC address
tshark -i eth0 -f "ether host 00:11:22:33:44:55"

# Source MAC
tshark -i eth0 -f "ether src 00:11:22:33:44:55"

# Destination MAC
tshark -i eth0 -f "ether dst 00:11:22:33:44:55"

# Broadcast traffic
tshark -i eth0 -f "ether broadcast"

# Multicast traffic
tshark -i eth0 -f "ether multicast"

# Specific EtherType
tshark -i eth0 -f "ether proto 0x0800"  # IPv4
tshark -i eth0 -f "ether proto 0x0806"  # ARP
tshark -i eth0 -f "ether proto 0x86dd"  # IPv6
```

### Packet Size Filters

```bash
# Packets less than size
tshark -i eth0 -f "less 128"

# Packets greater than size
tshark -i eth0 -f "greater 1000"

# Packets of specific size
tshark -i eth0 -f "len == 64"

# Packets in size range (using boolean logic)
tshark -i eth0 -f "greater 100 and less 500"

# Large packets (potential performance issues)
tshark -i eth0 -f "greater 1500"
```

### VLAN Filters

```bash
# Capture VLAN traffic
tshark -i eth0 -f "vlan"

# Specific VLAN ID
tshark -i eth0 -f "vlan 100"

# VLAN and protocol
tshark -i eth0 -f "vlan and tcp"

# VLAN with specific traffic
tshark -i eth0 -f "vlan 100 and host 192.168.1.1"
```

## Display Filters (Wireshark Syntax)

Display filters use Wireshark's powerful filter language for detailed protocol analysis.

### Basic Syntax

```bash
# General syntax: protocol.field operator value

# Equals
tshark -r capture.pcap -Y "ip.src == 192.168.1.1"

# Not equals
tshark -r capture.pcap -Y "ip.src != 192.168.1.1"

# Logical AND
tshark -r capture.pcap -Y "ip.src == 192.168.1.1 and tcp.port == 80"

# Logical OR
tshark -r capture.pcap -Y "tcp.port == 80 or tcp.port == 443"

# Logical NOT
tshark -r capture.pcap -Y "not icmp"
tshark -r capture.pcap -Y "!(tcp.port == 22)"

# Parentheses for grouping
tshark -r capture.pcap -Y "(ip.src == 192.168.1.1 or ip.src == 192.168.1.2) and tcp.port == 80"
```

### IP Filters

```bash
# Source IP
tshark -Y "ip.src == 192.168.1.1"

# Destination IP
tshark -Y "ip.dst == 192.168.1.1"

# Source or destination IP (address)
tshark -Y "ip.addr == 192.168.1.1"

# IP subnet
tshark -Y "ip.src == 192.168.1.0/24"
tshark -Y "ip.addr == 10.0.0.0/8"

# Multiple IP addresses
tshark -Y "ip.src == 192.168.1.1 or ip.src == 192.168.1.2"

# IP address in set
tshark -Y "ip.addr in {192.168.1.1 192.168.1.2 192.168.1.3}"

# IPv4 only
tshark -Y "ip"

# IPv6 only
tshark -Y "ipv6"

# IP TTL
tshark -Y "ip.ttl < 10"
tshark -Y "ip.ttl == 64"

# IP fragmentation
tshark -Y "ip.flags.mf == 1"       # More fragments
tshark -Y "ip.frag_offset > 0"     # Fragmented packets
```

### TCP Filters

```bash
# TCP port (source or destination)
tshark -Y "tcp.port == 80"

# TCP source port
tshark -Y "tcp.srcport == 80"

# TCP destination port
tshark -Y "tcp.dstport == 443"

# TCP port range
tshark -Y "tcp.port >= 8000 and tcp.port <= 9000"

# TCP flags
tshark -Y "tcp.flags.syn == 1"              # SYN flag set
tshark -Y "tcp.flags.ack == 1"              # ACK flag set
tshark -Y "tcp.flags.fin == 1"              # FIN flag set
tshark -Y "tcp.flags.reset == 1"            # RST flag set
tshark -Y "tcp.flags.push == 1"             # PSH flag set
tshark -Y "tcp.flags.urg == 1"              # URG flag set

# SYN-ACK packets
tshark -Y "tcp.flags.syn == 1 and tcp.flags.ack == 1"

# TCP SYN only (connection initiation)
tshark -Y "tcp.flags.syn == 1 and tcp.flags.ack == 0"

# TCP RST packets
tshark -Y "tcp.flags.reset == 1"

# TCP window size
tshark -Y "tcp.window_size < 1000"

# TCP sequence number
tshark -Y "tcp.seq == 1"

# TCP acknowledgment number
tshark -Y "tcp.ack == 1"

# TCP analysis flags
tshark -Y "tcp.analysis.retransmission"     # Retransmissions
tshark -Y "tcp.analysis.duplicate_ack"      # Duplicate ACKs
tshark -Y "tcp.analysis.lost_segment"       # Lost segments
tshark -Y "tcp.analysis.fast_retransmission" # Fast retransmissions
tshark -Y "tcp.analysis.zero_window"        # Zero window
tshark -Y "tcp.analysis.window_full"        # Window full
tshark -Y "tcp.analysis.out_of_order"       # Out of order packets

# TCP stream
tshark -Y "tcp.stream == 0"                 # First TCP stream
tshark -Y "tcp.stream == 5"                 # Sixth TCP stream
```

### UDP Filters

```bash
# UDP port
tshark -Y "udp.port == 53"

# UDP source port
tshark -Y "udp.srcport == 5353"

# UDP destination port
tshark -Y "udp.dstport == 161"

# UDP length
tshark -Y "udp.length < 100"
tshark -Y "udp.length > 1000"

# UDP stream
tshark -Y "udp.stream == 0"
```

### HTTP Filters

```bash
# All HTTP traffic
tshark -Y "http"

# HTTP requests only
tshark -Y "http.request"

# HTTP responses only
tshark -Y "http.response"

# HTTP request methods
tshark -Y "http.request.method == GET"
tshark -Y "http.request.method == POST"
tshark -Y "http.request.method == PUT"
tshark -Y "http.request.method == DELETE"

# HTTP request URI
tshark -Y "http.request.uri contains \"/api/\""
tshark -Y "http.request.uri == \"/index.html\""

# HTTP host
tshark -Y "http.host == \"www.example.com\""
tshark -Y "http.host contains \"example\""

# HTTP user agent
tshark -Y "http.user_agent contains \"Mozilla\""
tshark -Y "http.user_agent contains \"curl\""

# HTTP response codes
tshark -Y "http.response.code == 200"
tshark -Y "http.response.code == 404"
tshark -Y "http.response.code == 500"
tshark -Y "http.response.code >= 400"        # Client/server errors

# HTTP response code categories
tshark -Y "http.response.code >= 200 and http.response.code < 300"  # Success
tshark -Y "http.response.code >= 300 and http.response.code < 400"  # Redirects
tshark -Y "http.response.code >= 400 and http.response.code < 500"  # Client errors
tshark -Y "http.response.code >= 500"                                # Server errors

# HTTP content type
tshark -Y "http.content_type contains \"application/json\""
tshark -Y "http.content_type contains \"text/html\""

# HTTP cookies
tshark -Y "http.cookie"
tshark -Y "http.set_cookie"

# HTTP authorization
tshark -Y "http.authorization"

# HTTP referer
tshark -Y "http.referer contains \"google\""

# HTTP with specific header
tshark -Y "http.header contains \"X-Custom-Header\""
```

### DNS Filters

```bash
# All DNS traffic
tshark -Y "dns"

# DNS queries only
tshark -Y "dns.flags.response == 0"

# DNS responses only
tshark -Y "dns.flags.response == 1"

# DNS query for specific name
tshark -Y "dns.qry.name == \"www.example.com\""
tshark -Y "dns.qry.name contains \"example\""

# DNS query type
tshark -Y "dns.qry.type == 1"              # A record
tshark -Y "dns.qry.type == 28"             # AAAA record
tshark -Y "dns.qry.type == 15"             # MX record
tshark -Y "dns.qry.type == 5"              # CNAME record
tshark -Y "dns.qry.type == 16"             # TXT record

# DNS response code
tshark -Y "dns.flags.rcode == 0"           # No error
tshark -Y "dns.flags.rcode == 3"           # NXDOMAIN (name error)

# DNS answer
tshark -Y "dns.a"                          # A record in answer
tshark -Y "dns.aaaa"                       # AAAA record in answer

# DNS with specific IP in answer
tshark -Y "dns.a == 192.168.1.1"

# DNS recursion desired
tshark -Y "dns.flags.recdesired == 1"
```

### TLS/SSL Filters

```bash
# All TLS traffic
tshark -Y "tls"
# (or "ssl" for older captures/versions)

# TLS handshake
tshark -Y "tls.handshake"

# TLS Client Hello
tshark -Y "tls.handshake.type == 1"

# TLS Server Hello
tshark -Y "tls.handshake.type == 2"

# TLS Certificate
tshark -Y "tls.handshake.type == 11"

# TLS handshake with specific SNI
tshark -Y "tls.handshake.extensions_server_name == \"www.example.com\""
tshark -Y "tls.handshake.extensions_server_name contains \"example\""

# TLS version
tshark -Y "tls.record.version == 0x0303"   # TLS 1.2
tshark -Y "tls.record.version == 0x0304"   # TLS 1.3

# TLS cipher suite
tshark -Y "tls.handshake.ciphersuite"

# TLS alert
tshark -Y "tls.alert_message"

# TLS application data
tshark -Y "tls.app_data"
```

### ICMP Filters

```bash
# All ICMP traffic
tshark -Y "icmp"

# ICMP echo request (ping)
tshark -Y "icmp.type == 8"

# ICMP echo reply
tshark -Y "icmp.type == 0"

# ICMP destination unreachable
tshark -Y "icmp.type == 3"

# ICMP time exceeded
tshark -Y "icmp.type == 11"

# ICMPv6
tshark -Y "icmpv6"
```

### ARP Filters

```bash
# All ARP traffic
tshark -Y "arp"

# ARP request
tshark -Y "arp.opcode == 1"

# ARP reply
tshark -Y "arp.opcode == 2"

# ARP for specific IP
tshark -Y "arp.dst.proto_ipv4 == 192.168.1.1"
tshark -Y "arp.src.proto_ipv4 == 192.168.1.1"

# Gratuitous ARP
tshark -Y "arp.opcode == 1 and arp.src.proto_ipv4 == arp.dst.proto_ipv4"
```

### DHCP Filters

```bash
# All DHCP traffic
tshark -Y "dhcp"

# DHCP Discover
tshark -Y "dhcp.option.dhcp == 1"

# DHCP Offer
tshark -Y "dhcp.option.dhcp == 2"

# DHCP Request
tshark -Y "dhcp.option.dhcp == 3"

# DHCP ACK
tshark -Y "dhcp.option.dhcp == 5"

# DHCP NAK
tshark -Y "dhcp.option.dhcp == 6"

# DHCP Release
tshark -Y "dhcp.option.dhcp == 7"
```

### SMB Filters

```bash
# All SMB traffic
tshark -Y "smb or smb2"

# SMB version 1
tshark -Y "smb"

# SMB version 2/3
tshark -Y "smb2"

# SMB commands
tshark -Y "smb2.cmd == 0"                  # Negotiate
tshark -Y "smb2.cmd == 1"                  # Session Setup
tshark -Y "smb2.cmd == 3"                  # Tree Connect
tshark -Y "smb2.cmd == 5"                  # Create
tshark -Y "smb2.cmd == 8"                  # Read
tshark -Y "smb2.cmd == 9"                  # Write

# SMB filename
tshark -Y "smb.file contains \"document\""
tshark -Y "smb2.filename contains \"document\""
```

### String Matching

```bash
# Contains string
tshark -Y "http.host contains \"example\""

# Matches regex (use matches operator)
tshark -Y "http.host matches \"^www\\..*\\.com$\""

# Case-insensitive contains
tshark -Y "http.host contains \"EXAMPLE\""  # Already case-insensitive

# String equals
tshark -Y "http.host == \"www.example.com\""

# String in set
tshark -Y "http.host in {\"example.com\" \"test.com\" \"demo.com\"}"
```

### Comparison Operators

```bash
# Equals
tshark -Y "tcp.port == 80"

# Not equals
tshark -Y "tcp.port != 22"

# Greater than
tshark -Y "frame.len > 1000"

# Less than
tshark -Y "ip.ttl < 10"

# Greater than or equal
tshark -Y "tcp.port >= 8000"

# Less than or equal
tshark -Y "tcp.port <= 9000"

# In range
tshark -Y "tcp.port >= 8000 and tcp.port <= 9000"
```

### Time-based Filters

```bash
# Frame time
tshark -Y "frame.time >= \"2024-01-01 00:00:00\""
tshark -Y "frame.time <= \"2024-12-31 23:59:59\""

# Time range
tshark -Y "frame.time >= \"2024-01-01 00:00:00\" and frame.time <= \"2024-01-01 23:59:59\""

# Frame time relative
tshark -Y "frame.time_relative > 10"        # More than 10 seconds into capture

# Frame time delta
tshark -Y "frame.time_delta > 1"            # More than 1 second since previous packet
```

### Packet Size Filters

```bash
# Frame length
tshark -Y "frame.len > 1000"
tshark -Y "frame.len < 100"
tshark -Y "frame.len == 54"

# Frame length range
tshark -Y "frame.len >= 100 and frame.len <= 500"

# IP length
tshark -Y "ip.len > 1400"
```

### Expert Information Filters

```bash
# Warnings
tshark -Y "expert.severity == warning"

# Errors
tshark -Y "expert.severity == error"

# Notes
tshark -Y "expert.severity == note"

# All expert info
tshark -Y "expert"

# TCP expert info
tshark -Y "tcp.analysis.flags"
```

### Complex Display Filters

```bash
# HTTP POST requests to specific host
tshark -Y "http.request.method == POST and http.host == \"api.example.com\""

# Failed HTTP requests
tshark -Y "http.response.code >= 400"

# Large HTTP responses
tshark -Y "http.response and frame.len > 10000"

# DNS queries without responses (potential issues)
tshark -Y "dns.flags.response == 0 and not dns.flags.response == 1"

# TCP retransmissions to specific IP
tshark -Y "tcp.analysis.retransmission and ip.dst == 192.168.1.1"

# TLS connections to specific domains
tshark -Y "tls.handshake.type == 1 and tls.handshake.extensions_server_name contains \"example\""

# Non-standard HTTP ports
tshark -Y "http and tcp.port != 80 and tcp.port != 443"

# Broadcast and multicast traffic
tshark -Y "eth.dst.ig == 1"  # Broadcast/multicast bit set

# IPv6 traffic on specific subnet
tshark -Y "ipv6.src == 2001:db8::/32"

# Suspicious DNS (queries to multiple IPs)
tshark -Y "dns.flags.response == 1 and dns.count.answers > 5"
```

## Output Formats and Field Extraction

### Output Format Options

```bash
# Default text output (one line per packet summary)
tshark -r capture.pcap

# Verbose/detailed output (full packet dissection)
tshark -r capture.pcap -V

# PDML (Packet Details Markup Language - XML)
tshark -r capture.pcap -T pdml

# PSML (Packet Summary Markup Language - XML)
tshark -r capture.pcap -T psml

# JSON output
tshark -r capture.pcap -T json

# JSON with raw hex
tshark -r capture.pcap -T jsonraw

# EK (Elasticsearch-friendly JSON)
tshark -r capture.pcap -T ek

# Fields (custom column output)
tshark -r capture.pcap -T fields -e frame.number -e ip.src -e ip.dst

# PS (PostScript - for printing)
tshark -r capture.pcap -T ps

# Text output with specific columns
tshark -r capture.pcap -T text
```

### Field Extraction

```bash
# Extract specific fields
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e tcp.port

# Multiple fields with delimiter
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -E separator=,

# Custom field separator (CSV format)
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e tcp.port -E separator=, -E quote=d

# Include header row
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -E header=y

# Aggregate fields (only unique values)
tshark -r capture.pcap -T fields -e ip.src -e ip.dst | sort | uniq

# Extract HTTP fields
tshark -r capture.pcap -Y "http.request" -T fields \
  -e frame.time -e ip.src -e http.request.method -e http.host -e http.request.uri

# Extract DNS queries
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields \
  -e frame.time -e ip.src -e dns.qry.name -e dns.qry.type

# Extract TLS SNI
tshark -r capture.pcap -Y "tls.handshake.type == 1" -T fields \
  -e frame.time -e ip.src -e ip.dst -e tls.handshake.extensions_server_name

# Extract specific TCP fields
tshark -r capture.pcap -Y "tcp" -T fields \
  -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport -e tcp.flags
```

### JSON Output Examples

```bash
# JSON output (pretty-printed)
tshark -r capture.pcap -T json | jq '.'

# JSON with specific packets
tshark -r capture.pcap -c 10 -T json

# JSON with display filter
tshark -r capture.pcap -Y "http" -T json

# Extract specific JSON fields
tshark -r capture.pcap -T json | jq '.[] | .layers.ip."ip.src"'

# JSON output to file
tshark -r capture.pcap -T json > output.json

# EK format for Elasticsearch
tshark -r capture.pcap -T ek

# EK with bulk format for Elasticsearch ingestion
tshark -r capture.pcap -T ek | while read line; do echo '{"index":{}}'; echo "$line"; done
```

### CSV Output

```bash
# Basic CSV output
tshark -r capture.pcap -T fields \
  -e frame.number -e frame.time -e ip.src -e ip.dst -e frame.len \
  -E header=y -E separator=, -E quote=d

# CSV with HTTP data
tshark -r capture.pcap -Y "http" -T fields \
  -e frame.time -e ip.src -e ip.dst \
  -e http.request.method -e http.host -e http.request.uri -e http.response.code \
  -E header=y -E separator=, -E quote=d -E occurrence=f

# Save to CSV file
tshark -r capture.pcap -T fields \
  -e frame.time -e ip.src -e ip.dst -e tcp.port \
  -E header=y -E separator=, -E quote=d > output.csv
```

### Custom Output Columns

```bash
# Print specific columns
tshark -r capture.pcap -T fields \
  -e frame.number \
  -e frame.time_relative \
  -e ip.src \
  -e ip.dst \
  -e _ws.col.Protocol \
  -e frame.len

# With better formatting using column
tshark -r capture.pcap -T fields \
  -e frame.number -e ip.src -e ip.dst \
  -E separator=/s | column -t

# Custom time format
tshark -r capture.pcap -t ad -T fields -e frame.time

# Time format options:
# -t r  : relative to first packet
# -t a  : absolute time
# -t ad : absolute with date
# -t d  : delta time (since previous packet)
# -t e  : epoch time
```

## Advanced Capture Techniques

### Ring Buffer Captures

```bash
# Ring buffer with file count limit
tshark -i eth0 -w capture.pcap -b files:10 -b filesize:10000
# Creates capture_00001.pcap, capture_00002.pcap, ... capture_00010.pcap
# Overwrites oldest file when limit reached

# Ring buffer with duration
tshark -i eth0 -w capture.pcap -b duration:60 -b files:24
# New file every 60 seconds, keep 24 files (24 hours of 1-minute captures)

# Ring buffer with file size
tshark -i eth0 -w capture.pcap -b filesize:100000 -b files:5
# New file when size reaches ~100MB, keep 5 files

# Combine multiple conditions
tshark -i eth0 -w capture.pcap -b filesize:50000 -b duration:300 -b files:10
# New file every 5 minutes OR 50MB, keep 10 files
```

### Conditional Captures

```bash
# Stop after N packets
tshark -i eth0 -c 1000 -w capture.pcap

# Stop after duration
tshark -i eth0 -a duration:3600 -w capture.pcap  # 1 hour

# Stop after file size
tshark -i eth0 -a filesize:100000 -w capture.pcap  # ~100MB

# Stop after N files
tshark -i eth0 -w capture.pcap -b files:5 -a files:5

# Multiple stop conditions (first met wins)
tshark -i eth0 -w capture.pcap -a duration:3600 -a filesize:100000
```

### Multiple Interface Capture

```bash
# Capture on multiple interfaces
tshark -i eth0 -i wlan0 -w capture.pcap

# Capture on all interfaces
tshark -i any -w capture.pcap

# Capture with interface in output
tshark -i eth0 -i wlan0 -T fields -e frame.interface_name -e ip.src -e ip.dst
```

### Snapshot Length (Packet Truncation)

```bash
# Capture only headers (first 128 bytes)
tshark -i eth0 -s 128 -w capture.pcap

# Capture full packets (default)
tshark -i eth0 -s 0 -w capture.pcap

# Minimal capture (Ethernet + IP + TCP headers)
tshark -i eth0 -s 54 -w capture.pcap

# Common snapshot lengths:
# 54-68: Headers only (Ethernet + IP + TCP/UDP)
# 128: Headers + some payload
# 256: Headers + moderate payload
# 1514: Full Ethernet frame
# 0: No limit (capture full packets)
```

### Buffer Size

```bash
# Set capture buffer size (in MB)
tshark -i eth0 -B 100 -w capture.pcap  # 100MB buffer

# Larger buffer for high-traffic capture
tshark -i eth0 -B 512 -w capture.pcap  # 512MB buffer

# Helps prevent packet loss on busy networks
```

### Name Resolution

```bash
# Disable all name resolution (faster)
tshark -n -r capture.pcap

# Enable MAC name resolution
tshark -N m -r capture.pcap

# Enable network name resolution (DNS)
tshark -N n -r capture.pcap

# Enable transport name resolution (port names)
tshark -N t -r capture.pcap

# Enable all name resolution
tshark -N mnt -r capture.pcap

# Disable name resolution during capture
tshark -i eth0 -n -w capture.pcap
```

### Monitor Mode (Wi-Fi)

```bash
# Enable monitor mode on Wi-Fi interface
sudo ip link set wlan0 down
sudo iw wlan0 set monitor control
sudo ip link set wlan0 up

# Capture in monitor mode
sudo tshark -i wlan0 -w wifi-capture.pcap

# Capture specific channel
sudo iw wlan0 set channel 6
sudo tshark -i wlan0 -w wifi-channel6.pcap

# Capture with radiotap headers
sudo tshark -i wlan0 -I -w wifi-monitor.pcap
```

### Remote Capture

```bash
# Capture on remote host via SSH and save locally
ssh user@remote-host "tshark -i eth0 -w -" > local-capture.pcap

# Capture on remote host and analyze locally in real-time
ssh user@remote-host "tshark -i eth0 -w -" | tshark -r - -Y "http"

# Remote capture with compression
ssh user@remote-host "tshark -i eth0 -w - | gzip -c" | gunzip -c > capture.pcap

# Using tcpdump on remote host
ssh user@remote-host "tcpdump -i eth0 -w -" | tshark -r -
```

## Statistics and Analysis

### Protocol Hierarchy Statistics

```bash
# Protocol hierarchy
tshark -r capture.pcap -q -z io,phs

# Shows percentage breakdown of protocols
# Example output:
# eth                                      100.00%
#   ip                                      95.00%
#     tcp                                   70.00%
#       http                                30.00%
#       tls                                 25.00%
#     udp                                   25.00%
#       dns                                 15.00%
```

### Conversation Statistics

```bash
# TCP conversations
tshark -r capture.pcap -q -z conv,tcp

# UDP conversations
tshark -r capture.pcap -q -z conv,udp

# IP conversations
tshark -r capture.pcap -q -z conv,ip

# Ethernet conversations
tshark -r capture.pcap -q -z conv,eth

# All conversations
tshark -r capture.pcap -q -z conv,tcp -z conv,udp
```

### Endpoint Statistics

```bash
# TCP endpoints
tshark -r capture.pcap -q -z endpoints,tcp

# UDP endpoints
tshark -r capture.pcap -q -z endpoints,udp

# IP endpoints
tshark -r capture.pcap -q -z endpoints,ip

# Ethernet endpoints
tshark -r capture.pcap -q -z endpoints,eth
```

### I/O Statistics

```bash
# I/O graph (packets per interval)
tshark -r capture.pcap -q -z io,stat,1  # 1 second intervals

# I/O stats with filters
tshark -r capture.pcap -q -z "io,stat,1,tcp,udp,icmp"

# I/O stats for specific filter
tshark -r capture.pcap -q -z "io,stat,1,http"

# Multiple interval types
tshark -r capture.pcap -q -z io,stat,1  # 1 second
tshark -r capture.pcap -q -z io,stat,60 # 1 minute
```

### HTTP Statistics

```bash
# HTTP requests by host
tshark -r capture.pcap -q -z http,tree

# HTTP request/response statistics
tshark -r capture.pcap -q -z http_req,tree

# HTTP response codes
tshark -r capture.pcap -q -z http_srv,tree

# HTTP request methods
tshark -r capture.pcap -Y "http.request" -T fields -e http.request.method | sort | uniq -c

# HTTP hosts
tshark -r capture.pcap -Y "http.request" -T fields -e http.host | sort | uniq -c

# HTTP user agents
tshark -r capture.pcap -Y "http.request" -T fields -e http.user_agent | sort | uniq
```

### DNS Statistics

```bash
# DNS statistics
tshark -r capture.pcap -q -z dns,tree

# DNS queries
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields -e dns.qry.name | sort | uniq -c

# DNS query types
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields -e dns.qry.type | sort | uniq -c

# DNS servers queried
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields -e ip.dst | sort | uniq -c

# DNS response times
tshark -r capture.pcap -Y "dns.flags.response == 1" -T fields -e dns.time
```

### TLS/SSL Statistics

```bash
# TLS handshake statistics
tshark -r capture.pcap -Y "tls.handshake" -q -z "io,stat,0,tls.handshake.type==1"

# TLS versions
tshark -r capture.pcap -Y "tls.handshake.version" -T fields -e tls.handshake.version | sort | uniq -c

# TLS SNI (Server Name Indication)
tshark -r capture.pcap -Y "tls.handshake.type == 1" -T fields \
  -e tls.handshake.extensions_server_name | sort | uniq -c

# TLS cipher suites
tshark -r capture.pcap -Y "tls.handshake.type == 2" -T fields \
  -e tls.handshake.ciphersuite | sort | uniq -c
```

### TCP Analysis Statistics

```bash
# TCP retransmissions
tshark -r capture.pcap -Y "tcp.analysis.retransmission" -q -z io,stat,0

# TCP duplicate ACKs
tshark -r capture.pcap -Y "tcp.analysis.duplicate_ack" -q -z io,stat,0

# TCP zero windows
tshark -r capture.pcap -Y "tcp.analysis.zero_window" -q -z io,stat,0

# TCP reset connections
tshark -r capture.pcap -Y "tcp.flags.reset == 1" -q -z io,stat,0

# TCP SYN/SYN-ACK/ACK statistics
tshark -r capture.pcap -q -z "io,stat,0,tcp.flags.syn==1 and tcp.flags.ack==0,tcp.flags.syn==1 and tcp.flags.ack==1"
```

### Service Response Time

```bash
# DNS response time
tshark -r capture.pcap -q -z "srt,dns"

# HTTP response time
tshark -r capture.pcap -q -z "srt,http"

# SMB response time
tshark -r capture.pcap -q -z "srt,smb"
```

### Expert Information

```bash
# All expert information
tshark -r capture.pcap -q -z expert

# Expert info summary
tshark -r capture.pcap -Y "expert" -T fields -e expert.message -e expert.severity

# Warnings only
tshark -r capture.pcap -Y "expert.severity == warning"

# Errors only
tshark -r capture.pcap -Y "expert.severity == error"
```

### Custom Statistics

```bash
# Count packets by source IP
tshark -r capture.pcap -T fields -e ip.src | sort | uniq -c | sort -rn

# Count packets by destination port
tshark -r capture.pcap -T fields -e tcp.dstport | sort | uniq -c | sort -rn

# Total bytes by IP address
tshark -r capture.pcap -T fields -e ip.src -e frame.len | \
  awk '{sum[$1]+=$2} END {for (ip in sum) print ip, sum[ip]}' | sort -k2 -rn

# Average packet size
tshark -r capture.pcap -T fields -e frame.len | \
  awk '{sum+=$1; count++} END {print sum/count}'

# Packets per second
tshark -r capture.pcap -T fields -e frame.time_epoch | \
  awk -F. '{print $1}' | uniq -c
```

## Following Streams

### TCP Stream Following

```bash
# Follow first TCP stream (stream 0)
tshark -r capture.pcap -q -z follow,tcp,ascii,0

# Follow specific TCP stream by number
tshark -r capture.pcap -q -z follow,tcp,ascii,5

# Follow TCP stream in hex
tshark -r capture.pcap -q -z follow,tcp,hex,0

# Follow TCP stream in raw format
tshark -r capture.pcap -q -z follow,tcp,raw,0

# Find stream number for specific connection
tshark -r capture.pcap -Y "ip.src == 192.168.1.1 and tcp.port == 80" -T fields -e tcp.stream | head -1

# Follow that stream
STREAM=$(tshark -r capture.pcap -Y "ip.src == 192.168.1.1 and tcp.port == 80" -T fields -e tcp.stream | head -1)
tshark -r capture.pcap -q -z follow,tcp,ascii,$STREAM
```

### UDP Stream Following

```bash
# Follow UDP stream
tshark -r capture.pcap -q -z follow,udp,ascii,0

# Follow specific UDP stream
tshark -r capture.pcap -q -z follow,udp,ascii,3
```

### HTTP Stream Following

```bash
# Follow HTTP stream
tshark -r capture.pcap -q -z follow,http,ascii,0

# Extract HTTP objects (files)
tshark -r capture.pcap --export-objects http,./exported-http-objects/

# List HTTP objects
tshark -r capture.pcap -q -z http,tree
```

### TLS Stream Following

```bash
# Follow TLS stream (shows encrypted data)
tshark -r capture.pcap -q -z follow,tls,ascii,0

# Decrypt TLS with key log file
tshark -r capture.pcap -o tls.keylog_file:sslkeys.log -q -z follow,tls,ascii,0

# Export TLS objects (if decrypted)
tshark -r capture.pcap -o tls.keylog_file:sslkeys.log --export-objects http,./exported/
```

## Protocol-Specific Analysis

### HTTP Analysis

```bash
# HTTP request summary
tshark -r capture.pcap -Y "http.request" -T fields \
  -e frame.number -e ip.src -e http.request.method -e http.host -e http.request.uri

# HTTP response summary
tshark -r capture.pcap -Y "http.response" -T fields \
  -e frame.number -e ip.src -e http.response.code -e http.content_length

# HTTP POST data
tshark -r capture.pcap -Y "http.request.method == POST" -T fields \
  -e http.host -e http.request.uri -e http.file_data

# HTTP cookies
tshark -r capture.pcap -Y "http.cookie" -T fields -e http.cookie

# HTTP with response time
tshark -r capture.pcap -Y "http.time" -T fields \
  -e http.request.full_uri -e http.response.code -e http.time

# Extract HTTP files
tshark -r capture.pcap --export-objects http,./http-exports/
```

### DNS Analysis

```bash
# DNS query-response pairs
tshark -r capture.pcap -Y "dns" -T fields \
  -e frame.time -e ip.src -e dns.qry.name -e dns.a -e dns.aaaa

# DNS query performance
tshark -r capture.pcap -Y "dns.flags.response == 1" -T fields \
  -e dns.qry.name -e dns.time | awk '{sum+=$2; count++} END {print sum/count}'

# DNS servers used
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields -e ip.dst | sort | uniq -c

# DNS NXDOMAINs
tshark -r capture.pcap -Y "dns.flags.rcode == 3" -T fields -e dns.qry.name

# DNS query types distribution
tshark -r capture.pcap -Y "dns.flags.response == 0" -T fields -e dns.qry.type | \
  awk '{types[$1]++} END {for (t in types) print t, types[t]}'

# Potential DNS tunneling (unusual query patterns)
tshark -r capture.pcap -Y "dns.qry.name.len > 50" -T fields -e dns.qry.name
```

### TLS/SSL Analysis

```bash
# TLS handshakes
tshark -r capture.pcap -Y "tls.handshake.type == 1" -T fields \
  -e frame.time -e ip.src -e ip.dst -e tls.handshake.extensions_server_name

# TLS versions in use
tshark -r capture.pcap -Y "tls.handshake.version" -T fields \
  -e tls.handshake.extensions_server_name -e tls.handshake.version

# TLS cipher suites offered
tshark -r capture.pcap -Y "tls.handshake.type == 1" -T fields \
  -e tls.handshake.ciphersuite

# TLS cipher suite selected
tshark -r capture.pcap -Y "tls.handshake.type == 2" -T fields \
  -e tls.handshake.extensions_server_name -e tls.handshake.ciphersuite

# TLS certificate details
tshark -r capture.pcap -Y "tls.handshake.type == 11" -T fields \
  -e x509sat.printableString -e x509ce.dNSName

# TLS alerts
tshark -r capture.pcap -Y "tls.alert_message" -T fields \
  -e frame.time -e ip.src -e ip.dst -e tls.alert_message.level -e tls.alert_message.desc

# Weak TLS versions
tshark -r capture.pcap -Y "tls.record.version < 0x0303"  # Older than TLS 1.2
```

### TCP Analysis

```bash
# TCP connection establishment (3-way handshake)
tshark -r capture.pcap -Y "tcp.flags.syn == 1"

# TCP connection completions
tshark -r capture.pcap -Y "tcp.flags.fin == 1 or tcp.flags.reset == 1"

# TCP retransmissions by host
tshark -r capture.pcap -Y "tcp.analysis.retransmission" -T fields -e ip.src | sort | uniq -c

# TCP window size issues
tshark -r capture.pcap -Y "tcp.analysis.zero_window or tcp.analysis.window_full"

# TCP reset connections
tshark -r capture.pcap -Y "tcp.flags.reset == 1" -T fields \
  -e frame.time -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport

# TCP duplicate ACKs
tshark -r capture.pcap -Y "tcp.analysis.duplicate_ack"

# TCP out-of-order packets
tshark -r capture.pcap -Y "tcp.analysis.out_of_order"

# TCP connections by port
tshark -r capture.pcap -Y "tcp.flags.syn == 1 and tcp.flags.ack == 0" -T fields \
  -e tcp.dstport | sort | uniq -c | sort -rn

# TCP handshake time (SYN to SYN-ACK)
tshark -r capture.pcap -Y "tcp.flags.syn == 1 and tcp.flags.ack == 1" -T fields \
  -e tcp.time_relative
```

### ICMP Analysis

```bash
# ICMP echo requests/replies (ping)
tshark -r capture.pcap -Y "icmp.type == 8 or icmp.type == 0" -T fields \
  -e frame.time -e ip.src -e ip.dst -e icmp.type -e icmp.seq

# ICMP unreachable messages
tshark -r capture.pcap -Y "icmp.type == 3" -T fields \
  -e frame.time -e ip.src -e ip.dst -e icmp.code

# ICMP time exceeded (traceroute)
tshark -r capture.pcap -Y "icmp.type == 11"

# Ping response times
tshark -r capture.pcap -Y "icmp.type == 0" -T fields -e icmp.resptime
```

### DHCP Analysis

```bash
# DHCP transactions
tshark -r capture.pcap -Y "dhcp" -T fields \
  -e frame.time -e dhcp.option.dhcp -e dhcp.ip.your -e dhcp.option.hostname

# DHCP discover/offer/request/ack flow
tshark -r capture.pcap -Y "dhcp" -T fields \
  -e frame.time -e dhcp.option.dhcp -e dhcp.hw.mac_addr -e dhcp.ip.your

# DHCP servers
tshark -r capture.pcap -Y "dhcp.option.dhcp == 2" -T fields -e ip.src | sort -u

# DHCP assigned IPs
tshark -r capture.pcap -Y "dhcp.option.dhcp == 5" -T fields \
  -e dhcp.hw.mac_addr -e dhcp.ip.your

# DHCP lease times
tshark -r capture.pcap -Y "dhcp" -T fields -e dhcp.option.dhcp_lease_time
```

### ARP Analysis

```bash
# ARP requests and replies
tshark -r capture.pcap -Y "arp" -T fields \
  -e frame.time -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 \
  -e arp.dst.hw_mac -e arp.dst.proto_ipv4

# ARP table building
tshark -r capture.pcap -Y "arp.opcode == 2" -T fields \
  -e arp.src.proto_ipv4 -e arp.src.hw_mac | sort -u

# Gratuitous ARP
tshark -r capture.pcap -Y "arp.opcode == 1 and arp.src.proto_ipv4 == arp.dst.proto_ipv4"

# ARP scans (many requests from one source)
tshark -r capture.pcap -Y "arp.opcode == 1" -T fields -e arp.src.proto_ipv4 | sort | uniq -c | sort -rn

# Potential ARP spoofing (duplicate IPs with different MACs)
tshark -r capture.pcap -Y "arp.opcode == 2" -T fields \
  -e arp.src.proto_ipv4 -e arp.src.hw_mac | sort | uniq
```

### SMB Analysis

```bash
# SMB file access
tshark -r capture.pcap -Y "smb2.cmd == 5" -T fields \
  -e frame.time -e ip.src -e smb2.filename

# SMB file reads/writes
tshark -r capture.pcap -Y "smb2.cmd == 8 or smb2.cmd == 9" -T fields \
  -e frame.time -e ip.src -e smb2.cmd -e smb2.filename

# SMB authentication
tshark -r capture.pcap -Y "smb2.cmd == 1" -T fields \
  -e frame.time -e ip.src -e ntlmssp.auth.username

# SMB shares accessed
tshark -r capture.pcap -Y "smb2.cmd == 3" -T fields -e smb2.tree | sort -u

# SMB errors
tshark -r capture.pcap -Y "smb2.nt_status != 0x00000000" -T fields \
  -e frame.time -e ip.src -e smb2.cmd -e smb2.nt_status
```

## Performance and Optimization

### Capture Performance

```bash
# Minimize packet loss with large buffer
tshark -i eth0 -B 512 -w capture.pcap

# Use capture filter to reduce load
tshark -i eth0 -f "tcp port 80" -w capture.pcap

# Disable name resolution for speed
tshark -i eth0 -n -w capture.pcap

# Truncate packets to reduce storage
tshark -i eth0 -s 128 -w capture.pcap

# Write directly to fast storage
tshark -i eth0 -w /dev/shm/capture.pcap

# Use multiple smaller files
tshark -i eth0 -w capture.pcap -b filesize:100000 -b files:10

# Disable display during capture
tshark -i eth0 -w capture.pcap -q
```

### Analysis Performance

```bash
# Use capture filter when possible (faster than display filter)
tshark -i eth0 -f "tcp port 80"    # Fast (capture filter)
# vs
tshark -i eth0 -Y "tcp.port == 80"  # Slower (display filter)

# Disable protocol dissection not needed
tshark -r capture.pcap -Y "ip.addr == 192.168.1.1" -d tcp.port==8080,http

# Read specific packet range
tshark -r capture.pcap -c 1000     # Read first 1000 packets

# Skip packets at beginning
tshark -r capture.pcap -Y "frame.number > 10000"

# Use two-pass filtering
# First pass: filter to smaller file
tshark -r large.pcap -Y "http" -w http-only.pcap
# Second pass: detailed analysis
tshark -r http-only.pcap -V

# Disable name resolution
tshark -r capture.pcap -n -Y "ip"

# Use fields instead of full dissection
tshark -r capture.pcap -T fields -e ip.src -e ip.dst  # Fast
# vs
tshark -r capture.pcap -V                             # Slow
```

### Memory Management

```bash
# Limit memory usage with ring buffer
tshark -i eth0 -w capture.pcap -b files:5 -b filesize:10000

# Process in chunks
for i in {1..10}; do
  tshark -r large.pcap -Y "frame.number >= $((($i-1)*10000)) and frame.number < $(($i*10000))"
done

# Stream processing (don't load all into memory)
tshark -r capture.pcap -T fields -e ip.src | sort | uniq -c
```

## Common Use Cases and Patterns

### Network Troubleshooting

```bash
# Verify connectivity between hosts
tshark -i eth0 -f "host 192.168.1.1 and host 192.168.1.2"

# Check if traffic reaches interface
tshark -i eth0 -f "host 8.8.8.8" -c 10

# Analyze TCP retransmissions (poor connection quality)
tshark -r capture.pcap -Y "tcp.analysis.retransmission" -q -z io,stat,1

# Check DNS resolution issues
tshark -i eth0 -Y "dns" -T fields -e frame.time -e dns.qry.name -e dns.flags.rcode

# Verify routing (ICMP redirects)
tshark -i eth0 -Y "icmp.type == 5"

# Identify packet loss
tshark -i eth0 -Y "tcp.analysis.lost_segment"

# Monitor bandwidth usage
tshark -i eth0 -q -z io,stat,1

# Check for duplicate IP addresses (ARP conflicts)
tshark -i eth0 -Y "arp.duplicate-address-detected"
```

### Security Analysis

```bash
# Detect port scanning
tshark -r capture.pcap -Y "tcp.flags.syn == 1 and tcp.flags.ack == 0" -T fields \
  -e ip.src -e tcp.dstport | awk '{print $1}' | sort | uniq -c | sort -rn

# Identify suspicious DNS queries (DGA domains, tunneling)
tshark -r capture.pcap -Y "dns.qry.name.len > 50 or dns.qry.name matches \"[a-z]{20,}\""

# Detect ARP spoofing
tshark -r capture.pcap -Y "arp" -T fields -e arp.src.proto_ipv4 -e arp.src.hw_mac | \
  sort | uniq -d

# Find unencrypted HTTP credentials
tshark -r capture.pcap -Y "http.authorization or http.cookie" -T fields \
  -e http.authorization -e http.cookie

# Detect SQL injection attempts
tshark -r capture.pcap -Y "http.request.uri contains \"union select\" or \
  http.request.uri contains \"1=1\""

# Identify malware C2 beaconing (regular intervals)
tshark -r capture.pcap -T fields -e ip.dst -e frame.time_epoch | \
  awk '{print $1, int($2)}' | uniq -c

# Find SMB null sessions
tshark -r capture.pcap -Y "smb2.cmd == 1 and ntlmssp.auth.username == \"\""

# Detect TLS downgrade attacks
tshark -r capture.pcap -Y "tls.record.version < 0x0303"

# Identify password spraying
tshark -r capture.pcap -Y "ntlmssp or kerberos" -T fields \
  -e ip.src -e ntlmssp.auth.username | sort | uniq -c
```

### Application Debugging

```bash
# Debug HTTP API calls
tshark -i eth0 -Y "http.host == \"api.example.com\"" -V

# Monitor database queries (MySQL example)
tshark -i eth0 -Y "mysql.query" -T fields -e mysql.query

# Debug web application errors
tshark -r capture.pcap -Y "http.response.code >= 500" -T fields \
  -e http.request.full_uri -e http.response.code

# Analyze SOAP/XML traffic
tshark -i eth0 -Y "http.content_type contains \"xml\"" -T fields -e http.file_data

# Debug REST API responses
tshark -i eth0 -Y "http and json" -T fields \
  -e http.request.full_uri -e http.response.code -e json.value.string

# Monitor application performance (HTTP response times)
tshark -r capture.pcap -Y "http.time" -T fields \
  -e http.request.full_uri -e http.time | \
  awk '{sum+=$2; count++} END {print "Average:", sum/count}'

# Debug WebSocket connections
tshark -i eth0 -Y "websocket" -T fields -e websocket.payload
```

### Performance Analysis

```bash
# Identify slow DNS responses
tshark -r capture.pcap -Y "dns.time > 0.1" -T fields \
  -e dns.qry.name -e dns.time -e ip.dst

# Find large HTTP responses
tshark -r capture.pcap -Y "http.content_length > 10000000" -T fields \
  -e http.request.uri -e http.content_length

# Analyze TCP window scaling issues
tshark -r capture.pcap -Y "tcp.window_size < 1000"

# Identify network congestion (TCP analysis)
tshark -r capture.pcap -Y "tcp.analysis.duplicate_ack or tcp.analysis.fast_retransmission"

# Monitor database query performance
tshark -r capture.pcap -Y "mysql" -T fields -e mysql.query -e mysql.response_time

# Analyze TLS handshake time
tshark -r capture.pcap -Y "tls.handshake.type == 2" -T fields \
  -e frame.time_relative -e tls.handshake.extensions_server_name

# Check for bandwidth-heavy hosts
tshark -r capture.pcap -T fields -e ip.src -e frame.len | \
  awk '{bytes[$1]+=$2} END {for(ip in bytes) print ip, bytes[ip]}' | sort -k2 -rn

# Identify chatty protocols
tshark -r capture.pcap -T fields -e _ws.col.Protocol | sort | uniq -c | sort -rn
```

### VoIP Analysis

```bash
# SIP call analysis
tshark -r capture.pcap -Y "sip" -T fields \
  -e sip.Method -e sip.from.user -e sip.to.user -e sip.Status-Line

# RTP stream statistics
tshark -r capture.pcap -q -z rtp,streams

# Analyze VoIP quality
tshark -r capture.pcap -q -z voip,stat

# Extract audio from RTP
tshark -r capture.pcap -Y "rtp" --export-objects rtp,./rtp-audio/

# Monitor SIP registration
tshark -i eth0 -Y "sip.Method == REGISTER"

# Track SIP calls
tshark -i eth0 -Y "sip.Method == INVITE or sip.Method == BYE"
```

### IoT Device Monitoring

```bash
# Monitor MQTT messages
tshark -i eth0 -Y "mqtt" -T fields -e mqtt.topic -e mqtt.msg

# CoAP requests
tshark -i eth0 -Y "coap" -T fields -e coap.opt.uri_path

# Zigbee analysis
tshark -i wpan0 -Y "zbee_aps"

# BLE (Bluetooth Low Energy) advertising
tshark -i bluetooth0 -Y "btle.advertising_address"

# UPnP device discovery
tshark -i eth0 -Y "ssdp" -T fields -e ssdp.server -e ssdp.location
```

## Automation and Scripting

### Bash Scripts

```bash
#!/bin/bash
# Capture HTTP traffic for 5 minutes and generate report

IFACE="eth0"
DURATION=300
OUTFILE="http-capture-$(date +%Y%m%d-%H%M%S).pcap"
REPORT="http-report-$(date +%Y%m%d-%H%M%S).txt"

# Capture
echo "Capturing HTTP traffic for $DURATION seconds..."
timeout $DURATION tshark -i $IFACE -f "tcp port 80 or tcp port 443" -w $OUTFILE -q

# Analyze
echo "Generating report..."
echo "HTTP Statistics" > $REPORT
echo "===============" >> $REPORT
echo "" >> $REPORT

echo "Top 10 Hosts:" >> $REPORT
tshark -r $OUTFILE -Y "http" -T fields -e http.host | sort | uniq -c | sort -rn | head -10 >> $REPORT

echo "" >> $REPORT
echo "HTTP Methods:" >> $REPORT
tshark -r $OUTFILE -Y "http.request" -T fields -e http.request.method | sort | uniq -c >> $REPORT

echo "" >> $REPORT
echo "Response Codes:" >> $REPORT
tshark -r $OUTFILE -Y "http.response" -T fields -e http.response.code | sort | uniq -c | sort -rn >> $REPORT

echo "Report saved to $REPORT"
```

### Continuous Monitoring

```bash
#!/bin/bash
# Monitor for suspicious DNS queries

IFACE="eth0"
ALERT_FILE="dns-alerts.log"

tshark -i $IFACE -Y "dns" -T fields -e frame.time -e ip.src -e dns.qry.name | \
while read timestamp src query; do
    # Alert on long domain names (potential DGA/tunneling)
    if [ ${#query} -gt 50 ]; then
        echo "$timestamp ALERT: Suspicious long DNS query from $src: $query" >> $ALERT_FILE
        echo "ALERT: Suspicious DNS query detected!"
    fi

    # Alert on queries to suspicious TLDs
    if [[ $query =~ \.(tk|ml|ga|cf)$ ]]; then
        echo "$timestamp ALERT: Query to suspicious TLD from $src: $query" >> $ALERT_FILE
    fi
done
```

### Python Integration

```python
#!/usr/bin/env python3
import subprocess
import json

def get_http_hosts(pcap_file):
    """Extract unique HTTP hosts from pcap file"""
    cmd = [
        'tshark',
        '-r', pcap_file,
        '-Y', 'http.request',
        '-T', 'fields',
        '-e', 'http.host'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    hosts = set(result.stdout.strip().split('\n'))
    return hosts

def get_dns_queries(pcap_file):
    """Get DNS queries as JSON"""
    cmd = [
        'tshark',
        '-r', pcap_file,
        '-Y', 'dns.flags.response == 0',
        '-T', 'json',
        '-e', 'frame.time',
        '-e', 'ip.src',
        '-e', 'dns.qry.name'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def monitor_interface(interface):
    """Monitor interface in real-time"""
    cmd = [
        'tshark',
        '-i', interface,
        '-Y', 'http',
        '-T', 'fields',
        '-e', 'ip.src',
        '-e', 'http.host',
        '-e', 'http.request.uri'
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    for line in process.stdout:
        fields = line.strip().split('\t')
        if len(fields) >= 3:
            src_ip, host, uri = fields[0], fields[1], fields[2]
            print(f"HTTP Request: {src_ip} -> {host}{uri}")

if __name__ == '__main__':
    # Example usage
    hosts = get_http_hosts('capture.pcap')
    print(f"Found {len(hosts)} unique HTTP hosts")

    # Monitor eth0
    # monitor_interface('eth0')
```

## TShark with Other Tools

### With tcpdump

```bash
# Capture with tcpdump, analyze with tshark
tcpdump -i eth0 -w - | tshark -r - -Y "http"

# Convert tcpdump filter to capture file
tcpdump -i eth0 -w capture.pcap "tcp port 80"
tshark -r capture.pcap -Y "http"
```

### With ngrep

```bash
# Combine ngrep for quick text search with tshark for detailed analysis
ngrep -q "password" tcp port 80
tshark -i eth0 -Y "http contains \"password\""
```

### With Scapy

```python
# Generate packets with Scapy, capture with tshark
from scapy.all import *

# In terminal: sudo tshark -i lo -f "icmp"
# Then in Python:
send(IP(dst="127.0.0.1")/ICMP())
```

### With Zeek (Bro)

```bash
# Use tshark for packet capture, Zeek for analysis
tshark -i eth0 -w capture.pcap
zeek -r capture.pcap

# Or pipe directly
tshark -i eth0 -w - | zeek -r -
```

### With Elasticsearch

```bash
# Send tshark output to Elasticsearch
tshark -r capture.pcap -T ek | \
while read line; do
    curl -X POST "localhost:9200/packets/_doc" \
         -H 'Content-Type: application/json' \
         -d "$line"
done

# Or use Filebeat/Logstash for better ingestion
```

### With Splunk

```bash
# Generate splunk-friendly format
tshark -r capture.pcap -T fields \
  -E header=y -E separator=, -E quote=d \
  -e frame.time -e ip.src -e ip.dst -e _ws.col.Protocol -e frame.len \
  > splunk-import.csv
```

## Troubleshooting

### Permission Issues

```bash
# Error: "Couldn't run /usr/bin/dumpcap in child process: Permission denied"

# Solution 1: Add user to wireshark group (Debian/Ubuntu)
sudo usermod -a -G wireshark $USER
newgrp wireshark

# Solution 2: Run as root (less secure)
sudo tshark -i eth0

# Solution 3: Set capabilities
sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/dumpcap

# Verify permissions
ls -l /usr/bin/dumpcap
getcap /usr/bin/dumpcap
```

### Interface Issues

```bash
# Error: "Capture interface not found"

# List interfaces
tshark -D
ip link show

# Check interface status
ip link show eth0

# Bring interface up
sudo ip link set eth0 up

# Check if interface supports capture
sudo tshark -i eth0 -c 1

# Try with "any" interface
sudo tshark -i any
```

### Capture Issues

```bash
# No packets captured

# Check capture filter syntax
tshark -i eth0 -f "tcp port 80" -c 10

# Remove capture filter to see all traffic
tshark -i eth0 -c 10

# Check if traffic exists
sudo tcpdump -i eth0 -c 10

# Increase buffer size
tshark -i eth0 -B 512

# Check for packet drops
tshark -i eth0 -q -z io,stat,1
```

### Display Filter Errors

```bash
# Error: "tshark: display filter syntax error"

# Test filter syntax
tshark -r capture.pcap -Y "tcp.port == 80" -c 1

# Use quotes around filter
tshark -r capture.pcap -Y "http.request.method == \"GET\""

# Check field names
tshark -G fields | grep -i "http.request"

# Validate filter
tshark -Y "tcp.port == 80" -c 0
```

### Performance Issues

```bash
# Slow capture or packet loss

# Use capture filter (not display filter)
tshark -i eth0 -f "tcp port 80"  # Fast
# instead of
tshark -i eth0 -Y "tcp.port == 80"  # Slow

# Increase buffer size
tshark -i eth0 -B 512

# Reduce packet size
tshark -i eth0 -s 128

# Disable name resolution
tshark -i eth0 -n

# Write to fast storage
tshark -i eth0 -w /dev/shm/capture.pcap

# Use ring buffer
tshark -i eth0 -w capture.pcap -b files:5 -b filesize:100000
```

### File Reading Issues

```bash
# Error: "tshark: The file is not a capture file"

# Check file type
file capture.pcap

# Try reading with -F option
tshark -r capture.pcap -F pcap

# Convert format if needed
tshark -r old-capture -F pcapng -w new-capture.pcapng

# Check file integrity
tshark -r capture.pcap -c 1 -V
```

### Memory Issues

```bash
# Out of memory errors

# Use ring buffer for large captures
tshark -i eth0 -w capture.pcap -b files:10 -b filesize:100000

# Process in chunks
tshark -r large.pcap -c 10000 > chunk1.txt

# Use streaming processing
tshark -r large.pcap -T fields -e ip.src | sort | uniq -c

# Limit output
tshark -r large.pcap -c 1000
```

### Name Resolution Issues

```bash
# Slow performance due to DNS lookups

# Disable all name resolution
tshark -n -r capture.pcap

# Disable specific resolution types
tshark -N n -r capture.pcap  # Disable network name resolution only

# Use custom hosts file
# Edit /etc/hosts then:
tshark -r capture.pcap
```

## Best Practices

### Capture Best Practices

1. **Use appropriate capture filters**
   ```bash
   # Filter at capture time, not display time
   tshark -i eth0 -f "tcp port 80"  # Good
   tshark -i eth0 -w all.pcap       # Then filter later - Bad for large captures
   ```

2. **Set proper ring buffer limits**
   ```bash
   # Prevent filling disk
   tshark -i eth0 -w capture.pcap -b files:10 -b filesize:100000
   ```

3. **Truncate packets when appropriate**
   ```bash
   # Save storage when full payload not needed
   tshark -i eth0 -s 128 -w headers.pcap
   ```

4. **Use quiet mode when writing files**
   ```bash
   # Reduce CPU usage
   tshark -i eth0 -w capture.pcap -q
   ```

5. **Monitor for packet loss**
   ```bash
   # Check capture statistics
   tshark -i eth0 -q -z io,stat,1
   ```

### Analysis Best Practices

1. **Start with statistics**
   ```bash
   # Get overview before detailed analysis
   tshark -r capture.pcap -q -z io,phs
   tshark -r capture.pcap -q -z conv,tcp
   ```

2. **Use appropriate filters**
   ```bash
   # Narrow down before detailed inspection
   tshark -r capture.pcap -Y "tcp.stream == 0" -V
   ```

3. **Extract relevant data only**
   ```bash
   # Don't dump everything
   tshark -r capture.pcap -T fields -e ip.src -e ip.dst
   ```

4. **Disable unnecessary dissection**
   ```bash
   # Faster analysis
   tshark -r capture.pcap -Y "frame.number < 100"
   ```

5. **Use two-pass analysis**
   ```bash
   # First pass: identify interesting streams
   tshark -r capture.pcap -Y "http.response.code >= 400" -T fields -e tcp.stream
   # Second pass: analyze specific streams
   tshark -r capture.pcap -Y "tcp.stream == 42" -V
   ```

### Privacy Best Practices

1. **Minimize capture scope**
   ```bash
   # Only capture what you need
   tshark -i eth0 -f "host 192.168.1.1 and port 80"
   ```

2. **Truncate packets to headers only**
   ```bash
   # Don't capture sensitive payload
   tshark -i eth0 -s 68 -w headers-only.pcap
   ```

3. **Secure capture files**
   ```bash
   # Set restrictive permissions
   tshark -i eth0 -w capture.pcap
   chmod 600 capture.pcap
   ```

4. **Anonymize IP addresses**
   ```bash
   # Use editcap for anonymization
   editcap -a 192.168.1.0/24:1.2.3.0/24 original.pcap anonymized.pcap
   ```

5. **Delete captures when done**
   ```bash
   # Don't keep captures longer than necessary
   find /captures -name "*.pcap" -mtime +7 -delete
   ```

### Security Best Practices

1. **Follow authorization requirements**
   - Get written permission before capturing
   - Document scope and limitations
   - Follow organizational policies

2. **Be aware of legal implications**
   - Understand local wiretapping laws
   - Know privacy regulations (GDPR, etc.)
   - Consider consent requirements

3. **Protect captured data**
   - Encrypt sensitive captures
   - Use secure transfer methods
   - Implement access controls

4. **Sanitize before sharing**
   - Remove sensitive information
   - Anonymize as appropriate
   - Redact confidential data

## Quick Reference

### Essential Commands

```bash
# List interfaces
tshark -D

# Capture live traffic
tshark -i eth0

# Capture to file
tshark -i eth0 -w capture.pcap

# Read from file
tshark -r capture.pcap

# Capture with filter
tshark -i eth0 -f "tcp port 80"

# Display with filter
tshark -r capture.pcap -Y "http"

# Verbose output
tshark -r capture.pcap -V

# Extract fields
tshark -r capture.pcap -T fields -e ip.src -e ip.dst

# JSON output
tshark -r capture.pcap -T json

# Statistics
tshark -r capture.pcap -q -z io,phs

# Follow TCP stream
tshark -r capture.pcap -q -z follow,tcp,ascii,0

# Quiet mode
tshark -i eth0 -w capture.pcap -q

# Capture N packets
tshark -i eth0 -c 100

# Ring buffer
tshark -i eth0 -w capture.pcap -b files:5 -b filesize:10000
```

### Common Capture Filters (BPF)

| Filter | Description |
|--------|-------------|
| `host 192.168.1.1` | Traffic to/from host |
| `net 192.168.1.0/24` | Traffic to/from network |
| `port 80` | Traffic on port 80 |
| `tcp` | TCP traffic only |
| `udp` | UDP traffic only |
| `icmp` | ICMP traffic only |
| `tcp port 80` | TCP traffic on port 80 |
| `src host 192.168.1.1` | Traffic from specific host |
| `dst port 443` | Traffic to port 443 |
| `not port 22` | Exclude port 22 |
| `tcp[tcpflags] & tcp-syn != 0` | TCP SYN packets |
| `portrange 8000-9000` | Port range |
| `ether host 00:11:22:33:44:55` | Specific MAC address |

### Common Display Filters

| Filter | Description |
|--------|-------------|
| `ip.addr == 192.168.1.1` | IP address (src or dst) |
| `tcp.port == 80` | TCP port (src or dst) |
| `http` | HTTP traffic |
| `http.request` | HTTP requests only |
| `http.response.code == 404` | HTTP 404 responses |
| `dns` | DNS traffic |
| `dns.qry.name contains "example"` | DNS queries containing text |
| `tls.handshake.type == 1` | TLS Client Hello |
| `tcp.analysis.retransmission` | TCP retransmissions |
| `tcp.flags.syn == 1` | TCP SYN flag set |
| `ip.src == 192.168.1.0/24` | Source IP in subnet |
| `frame.len > 1000` | Packets larger than 1000 bytes |
| `http.request.method == "POST"` | HTTP POST requests |
| `tcp.stream == 0` | First TCP stream |
| `expert` | Expert information |

### Output Format Options

| Option | Format |
|--------|--------|
| `-T text` | Default text output |
| `-T fields` | Custom field extraction |
| `-T json` | JSON format |
| `-T jsonraw` | JSON with raw hex |
| `-T ek` | Elasticsearch JSON |
| `-T pdml` | XML (PDML) |
| `-T ps` | PostScript |
| `-V` | Verbose packet details |
| `-x` | Hex and ASCII dump |

### Common Statistics

| Command | Statistics |
|---------|------------|
| `-q -z io,phs` | Protocol hierarchy |
| `-q -z conv,tcp` | TCP conversations |
| `-q -z endpoints,ip` | IP endpoints |
| `-q -z io,stat,1` | I/O statistics (1 sec intervals) |
| `-q -z http,tree` | HTTP statistics |
| `-q -z dns,tree` | DNS statistics |
| `-q -z expert` | Expert information |
| `-q -z follow,tcp,ascii,0` | Follow TCP stream 0 |

### Useful Field Names

| Field | Description |
|-------|-------------|
| `frame.number` | Packet number |
| `frame.time` | Timestamp |
| `frame.len` | Frame length |
| `eth.src` | Source MAC |
| `eth.dst` | Destination MAC |
| `ip.src` | Source IP |
| `ip.dst` | Destination IP |
| `ip.proto` | IP protocol |
| `tcp.srcport` | TCP source port |
| `tcp.dstport` | TCP destination port |
| `tcp.stream` | TCP stream index |
| `tcp.flags` | TCP flags |
| `udp.srcport` | UDP source port |
| `udp.dstport` | UDP destination port |
| `http.request.method` | HTTP method |
| `http.host` | HTTP host |
| `http.request.uri` | HTTP URI |
| `http.response.code` | HTTP response code |
| `dns.qry.name` | DNS query name |
| `dns.a` | DNS A record |
| `tls.handshake.extensions_server_name` | TLS SNI |

## Conclusion

TShark is an essential tool for network analysis, troubleshooting, and security investigations. Its command-line nature makes it ideal for remote systems, automation, and integration with other tools.

**Key Takeaways:**

- Understand the difference between capture and display filters
- Use capture filters for performance and efficiency
- Use display filters for detailed analysis
- Choose appropriate output formats for your use case
- Apply ring buffers for long-term monitoring
- Leverage statistics for quick insights
- Follow streams for application-level analysis
- Combine with other tools for comprehensive analysis
- Always consider legal and ethical implications
- Secure and protect captured data

**Learning Path:**

1. **Week 1**: Basic capture and reading, simple filters
2. **Week 2**: Display filters, field extraction, output formats
3. **Week 3**: Protocol analysis (HTTP, DNS, TCP, TLS)
4. **Week 4**: Statistics, expert information, stream following
5. **Month 2**: Advanced filters, performance optimization, automation
6. **Month 3+**: Integration, scripting, specialized analysis

**Resources:**
- Wireshark documentation: https://www.wireshark.org/docs/
- Display filter reference: https://www.wireshark.org/docs/dfref/
- TShark man page: `man tshark`
- Wireshark wiki: https://wiki.wireshark.org/
- Practice on sample captures: https://wiki.wireshark.org/SampleCaptures

TShark's power lies in its flexibility and depth. Master its basics first, then gradually explore advanced features as needed. Combined with proper authorization and ethical use, it becomes an invaluable tool for understanding network behavior and diagnosing issues.

Happy analyzing!
