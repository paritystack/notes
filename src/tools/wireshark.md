# Wireshark

Wireshark is the world's foremost and most widely-used network protocol analyzer. It allows you to see what's happening on your network at a microscopic level and is the de facto (and often de jure) standard across many commercial and non-profit enterprises, government agencies, and educational institutions.

## Overview

Wireshark was originally developed as Ethereal by Gerald Combs in 1998. The project was renamed to Wireshark in 2006 and continues to be actively developed by a large community of networking experts. It provides a graphical user interface (GUI) for deep inspection of hundreds of protocols, with more being added all the time.

**Key Features:**
- Live packet capture from network interfaces
- Deep inspection of 3000+ protocols with rich dissection
- Powerful display filter language for precise analysis
- VoIP analysis and playback capabilities
- Read/write support for many capture file formats
- Decryption support for many protocols (WEP, WPA/WPA2, SSL/TLS, IPsec)
- Export capabilities (XML, PostScript, CSV, plain text)
- Coloring rules for quick visual analysis
- Rich statistical analysis and graphing
- Multi-platform support (Windows, Linux, macOS, BSD)
- Command-line equivalents (TShark) for automation
- Lua scripting support for custom dissectors
- Expert information system for problem detection
- Follow TCP/UDP/HTTP/TLS streams
- Protocol hierarchy statistics
- Conversation and endpoint analysis
- IO graphs and flow visualization
- Time-sequence graphs (Stevens, tcptrace, throughput)
- Regular expression and binary pattern matching

**Common Use Cases:**
- Network troubleshooting and diagnostics
- Protocol development and analysis
- Network security analysis and forensics
- Educational purposes and learning
- Quality assurance and testing
- Malware analysis and reverse engineering
- VoIP call quality analysis
- Application performance monitoring
- Compliance validation and auditing
- Wireless network analysis (802.11, Bluetooth, Zigbee)

## Legal and Ethical Considerations

**CRITICAL:** Capturing and analyzing network traffic has serious legal and privacy implications. Unauthorized packet capture may violate wiretapping laws, privacy regulations, and organizational policies.

**Legal Requirements:**
- **Authorization**: Always obtain explicit written permission before capturing network traffic
- **Jurisdiction**: Understand local, state, and federal laws regarding network monitoring
- **Privacy Laws**: Comply with GDPR, HIPAA, CCPA, and other privacy regulations
- **Workplace Policies**: Follow organizational security and acceptable use policies
- **Consent**: Some jurisdictions require consent from parties being monitored
- **Data Protection**: Implement appropriate controls for captured data

**Best Practices:**
- Only capture on networks you own or have written authorization to monitor
- Define clear scope and boundaries for monitoring activities
- Minimize captured data to what is necessary
- Secure capture files with encryption and access controls
- Implement data retention and destruction policies
- Redact sensitive information before sharing captures
- Document authorization and justification for captures
- Use encrypted connections when transferring capture files
- Be aware captures may contain passwords, personal data, trade secrets
- Follow responsible disclosure for security vulnerabilities

**Ethical Considerations:**
- Respect privacy even when technically feasible to monitor
- Use capabilities only for legitimate purposes
- Minimize impact on network and systems
- Protect confidentiality of discovered information
- Consider consent and notification requirements

## Basic Concepts

### How Wireshark Works

Wireshark captures packets from network interfaces and provides detailed analysis through several layers:

1. **Packet Capture** - Uses libpcap (Unix/Linux) or WinPcap/Npcap (Windows) to capture raw packets
2. **Protocol Dissection** - Analyzes packet structure and decodes protocol layers
3. **Display Filtering** - Applies user-defined filters to show relevant packets
4. **Analysis** - Provides statistics, graphs, and expert information
5. **Export** - Saves data in various formats for further processing

### Capture vs Display Filters

Understanding the distinction is crucial:

**Capture Filters (BPF - Berkeley Packet Filter):**
- Applied during packet capture
- Determine which packets to capture
- Cannot be changed after capture starts
- More efficient (reduces storage and memory)
- Limited syntax (tcpdump-style)
- Cannot filter on dissected protocol fields
- Examples: `tcp port 80`, `host 192.168.1.1`

**Display Filters:**
- Applied after packets are captured
- Determine which packets to display
- Can be changed anytime
- Rich, powerful syntax
- Can filter on any dissected field
- All packets remain in capture file
- Examples: `http.request.method == "POST"`, `tcp.analysis.retransmission`

### Protocol Dissectors

Wireshark's strength lies in its protocol dissectors:

- **3000+ protocol dissectors** covering virtually all network protocols
- **Hierarchical dissection** from Layer 2 through Layer 7
- **Automatic protocol detection** based on ports and heuristics
- **Extensible** via Lua scripts and C plugins
- **Contextual decoding** based on conversation state
- **Reassembly** of fragmented packets and TCP streams

### Packet Structure

Wireshark displays packets in hierarchical layers:

1. **Frame** - Physical layer information (interface, time, length)
2. **Data Link Layer** - Ethernet, Wi-Fi, PPP, etc.
3. **Network Layer** - IP, IPv6, ARP, ICMP
4. **Transport Layer** - TCP, UDP, SCTP
5. **Application Layer** - HTTP, DNS, TLS, SMB, etc.

### Expert Information

Wireshark's expert system automatically detects:

- **Errors** - Malformed packets, checksum failures
- **Warnings** - Retransmissions, duplicate ACKs
- **Notes** - Unusual but valid occurrences
- **Chats** - Normal workflow information

## Installation

### Windows

```plaintext
1. Download from https://www.wireshark.org/download.html
2. Choose Windows Installer (.exe)
3. Run installer as Administrator
4. Select components:
   - Wireshark (GUI)
   - TShark (CLI)
   - Plugins/Extensions
   - Tools (editcap, mergecap, etc.)
5. Install Npcap when prompted (required for packet capture)
   ☑ Install Npcap in WinPcap API-compatible mode
   ☑ Support raw 802.11 traffic (for wireless)
6. Complete installation
7. Reboot if required

Note: Npcap is the modern replacement for WinPcap
```

### macOS

```bash
# Option 1: Download from website
# Visit https://www.wireshark.org/download.html
# Download macOS .dmg file
# Open DMG and drag to Applications
# Install ChmodBPF (included) for packet capture

# Option 2: Homebrew
brew install wireshark

# Grant capture permissions
# ChmodBPF is installed automatically
# Verify: sudo launchctl list | grep chmod

# Add your user to access_bpf group
sudo dseditgroup -o edit -a $USER -t user access_bpf

# Launch Wireshark
open -a Wireshark

# Or from terminal
wireshark
```

### Linux (Debian/Ubuntu)

```bash
# Install Wireshark
sudo apt update
sudo apt install wireshark

# During installation, select "Yes" to allow non-root users to capture packets

# Add your user to wireshark group
sudo usermod -a -G wireshark $USER

# Log out and back in, or activate group in current session
newgrp wireshark

# Verify permissions
groups | grep wireshark

# Alternative: reconfigure permissions
sudo dpkg-reconfigure wireshark-common
sudo usermod -a -G wireshark $USER

# Launch Wireshark
wireshark

# Or launch as root (not recommended)
sudo wireshark
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Fedora/Recent CentOS
sudo dnf install wireshark

# Older RHEL/CentOS
sudo yum install wireshark

# Permissions
sudo usermod -a -G wireshark $USER

# Set capabilities
sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/dumpcap

# Launch
wireshark
```

### Build from Source

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt install build-essential cmake libglib2.0-dev \
  libpcap-dev qtbase5-dev qttools5-dev-tools libqt5svg5-dev \
  qtmultimedia5-dev flex bison libssl-dev

# Download source
wget https://www.wireshark.org/download/src/wireshark-latest.tar.xz
tar xf wireshark-latest.tar.xz
cd wireshark-*

# Build
cmake -G Ninja ..
ninja

# Install
sudo ninja install

# Update library cache
sudo ldconfig
```

### Verify Installation

```bash
# Check version
wireshark --version

# List interfaces
tshark -D

# Test capture (requires permissions)
tshark -i eth0 -c 5
```

## User Interface

Wireshark's interface consists of several key components:

### Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│ Menu Bar: File, Edit, View, Go, Capture, Analyze, etc. │
├─────────────────────────────────────────────────────────┤
│ Main Toolbar: Start/Stop, Open, Save, Filter, etc.     │
├─────────────────────────────────────────────────────────┤
│ Filter Toolbar: [Display Filter Input] [Bookmarks]     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Packet List Pane (Top)                                 │
│ No. Time    Source      Destination   Protocol  Info   │
│ ─────────────────────────────────────────────────────  │
│  1  0.000   192.168.1.1 192.168.1.100 TCP      [SYN]   │
│  2  0.001   192.168.1.100 192.168.1.1 TCP    [SYN,ACK] │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Packet Details Pane (Middle)                           │
│ ▼ Frame 1: 74 bytes on wire                            │
│ ▼ Ethernet II                                          │
│   ▶ Internet Protocol Version 4                        │
│   ▶ Transmission Control Protocol                      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Packet Bytes Pane (Bottom)                             │
│ 0000  00 11 22 33 44 55 66 77 88 99 aa bb 08 00 45 00  │
│ 0010  00 3c 1c 46 40 00 40 06 b1 e6 c0 a8 01 01 c0 a8  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Status Bar: Packets: 1234 Displayed: 567 Marked: 3     │
└─────────────────────────────────────────────────────────┘
```

### Packet List Pane

The top pane shows a summary of each packet:

- **No.** - Packet number in capture
- **Time** - Timestamp (various formats available)
- **Source** - Source address (IP, MAC, etc.)
- **Destination** - Destination address
- **Protocol** - Highest-level protocol detected
- **Length** - Packet length
- **Info** - Protocol-specific information

**Features:**
- Click column header to sort
- Right-click for context menu
- Double-click to expand in details pane
- Color-coded based on coloring rules
- Customizable columns

### Packet Details Pane

The middle pane shows hierarchical packet structure:

- Expandable tree view of protocol layers
- Click ▶ to expand, ▼ to collapse
- Shows field names and values
- Highlights selected field in bytes pane
- Right-click for various actions
- Can apply filters from selected fields

### Packet Bytes Pane

The bottom pane shows raw packet data:

- Hexadecimal view on left
- ASCII representation on right
- Highlights correspond to selection in details pane
- Can be displayed as hex, bits, or decimal
- Useful for low-level analysis

### Menu Bar

**File Menu:**
- Open, Save, Export
- Merge capture files
- Import from hex dump
- Print packets
- Quit

**Edit Menu:**
- Find packets
- Mark/unmark packets
- Time reference
- Ignore packets
- Configuration profiles
- Preferences

**View Menu:**
- Zoom in/out
- Expand/collapse all
- Colorize packets
- Show/hide panes
- Time display format
- Name resolution
- Reload capture

**Go Menu:**
- Go to packet
- Next/previous packet
- First/last packet
- Go to conversation

**Capture Menu:**
- Start/stop capture
- Restart capture
- Capture options
- Capture interfaces
- Refresh interfaces

**Analyze Menu:**
- Display filters
- Display filter macros
- Apply as filter
- Prepare as filter
- Enabled protocols
- Decode as
- Follow stream
- Expert information
- Conversation filter

**Statistics Menu:**
- Capture file properties
- Protocol hierarchy
- Conversations
- Endpoints
- Packet lengths
- IO graphs
- Service response time
- Flow graphs
- HTTP, DNS statistics
- Much more...

**Telephony Menu:**
- VoIP calls
- RTP analysis
- RTP player
- SCTP analysis
- LTE MAC/RLC analysis
- GSM/UMTS analysis

**Wireless Menu:**
- Bluetooth
- WLAN traffic

**Tools Menu:**
- Firewall ACL rules
- Credentials
- Lua
- Dissector tables

**Help Menu:**
- Contents
- Manual pages
- Website
- FAQ
- About

### Toolbars

**Main Toolbar:**
- Start/Stop capture
- Restart capture
- Capture options
- Open file
- Save
- Close
- Reload
- Find packet
- Go to packet
- Go back/forward
- Auto-scroll
- Colorize
- Zoom in/out
- Resize columns

**Filter Toolbar:**
- Display filter input field
- Filter expression button
- Clear filter
- Apply filter
- Recent filters dropdown
- Filter bookmarks
- Save filter
- Add expression

### Status Bar

Shows real-time information:

**Left Side:**
- Capture status
- File name
- Profile name

**Middle:**
- Expert information summary (color-coded)
- Errors (red)
- Warnings (yellow)
- Notes (cyan)
- Chats (blue)

**Right Side:**
- Packet statistics:
  - Packets: total captured
  - Displayed: matching current filter
  - Marked: manually marked packets
  - Dropped: packets lost during capture
  - Load time: file loading duration

## Basic Operations

### Starting a Capture

**Method 1: Quick Start**
1. Launch Wireshark
2. Double-click interface on welcome screen
3. Capture starts immediately

**Method 2: Capture Options**
1. Click Capture → Options (or Ctrl+K)
2. Select interface(s)
3. Set capture filter (optional)
4. Configure options:
   - Promiscuous mode
   - Snapshot length
   - Buffer size
   - Capture file options
5. Click Start

**Capture Options Dialog:**
```
Input:
☑ Promiscuous mode
  Capture all packets (not just those destined for this interface)

☑ Monitor mode (for wireless)
  Capture all wireless traffic including management frames

Snapshot length: [automatic]
  Limit bytes captured per packet
  0 or blank = unlimited
  Common: 65535 (full packets), 96 (headers only)

Buffer size: [2] MB
  Kernel buffer for packet capture
  Increase for high-traffic networks

Capture filter: [tcp port 80]
  BPF syntax filter applied during capture

Output:
File: [browse...]
  Save directly to file

Create new file:
☐ Every [1000000] kilobytes
☐ Every [60] seconds
☐ After [1000] packets

Use ring buffer:
☑ Number of files: [10]
  Keep only N most recent files

Options:
☐ Stop capture after:
  ☐ [1000] packets
  ☐ [1000] kilobytes
  ☐ [60] seconds
  ☐ [10] files

☐ Update list of packets in real-time
☐ Automatically scroll during live capture
```

### Stopping a Capture

- Click red square Stop button
- Capture → Stop (Ctrl+E)
- Set automatic stop conditions in Capture Options

### Opening Capture Files

**Open File:**
1. File → Open (Ctrl+O)
2. Browse to file
3. Select file
4. Click Open

**Supported Formats:**
- pcap, pcapng (Wireshark native)
- snoop (Sun)
- LANalyzer
- Network Monitor (Microsoft)
- tcpdump
- Visual Networks
- Numerous others...

**Open Recent:**
- File → Open Recent
- Shows recently opened files

**Drag and Drop:**
- Drag .pcap/.pcapng file to Wireshark window

### Saving Captures

**Save As:**
1. File → Save As (Ctrl+Shift+S)
2. Choose location and filename
3. Select file format (pcap, pcapng)
4. Choose what to save:
   - All packets
   - Displayed packets (matching filter)
   - Marked packets
   - Range of packets
5. Click Save

**Quick Save:**
- File → Save (Ctrl+S)
- Saves in current format

**File Formats:**
- **pcapng** - Wireshark native, supports:
  - Multiple interfaces
  - Interface descriptions
  - Name resolution
  - Capture comments
  - Custom options
- **pcap** - Traditional format, maximum compatibility

### Merging Captures

**Merge Files:**
1. File → Merge
2. Select file to merge
3. Choose merge method:
   - Chronologically
   - Append to end
   - Prepend to beginning
4. Click OK

**Command Line (mergecap):**
```bash
# Merge multiple files chronologically
mergecap -w output.pcap file1.pcap file2.pcap file3.pcap

# Merge with specific snaplen
mergecap -w output.pcap -s 65535 file1.pcap file2.pcap
```

### Exporting Data

**Export Specified Packets:**
- File → Export Specified Packets
- Save subset based on current display filter

**Export Packet Dissections:**
- File → Export Packet Dissections
- Formats: plain text, CSV, JSON, C arrays

**Export Objects:**
- File → Export Objects → HTTP/DICOM/SMB/TFTP
- Extract files transferred over these protocols
- Shows list of files
- Select and save

**Export as C Arrays:**
- File → Export Packet Dissections → As "C" Arrays
- Useful for test data in development

## Capture Filters (BPF Syntax)

Capture filters use Berkeley Packet Filter syntax, identical to tcpdump.

### Basic Syntax

```
Qualifier:
  Type:      host, net, port, portrange
  Direction: src, dst, src or dst, src and dst
  Protocol:  ether, ip, ip6, arp, tcp, udp, icmp

Examples:
  host 192.168.1.1
  src host 192.168.1.1
  dst net 192.168.0.0/16
  tcp port 80
  udp port 53
```

### Host Filters

```
# Specific host (src or dst)
host 192.168.1.1
host www.example.com

# Source host
src host 192.168.1.1

# Destination host
dst host 192.168.1.1

# Multiple hosts
host 192.168.1.1 or host 192.168.1.2

# Exclude host
not host 192.168.1.1
```

### Network Filters

```
# Network range (CIDR)
net 192.168.1.0/24
net 10.0.0.0/8

# Source network
src net 192.168.0.0/16

# Destination network
dst net 10.0.0.0/8

# Exclude network
not net 192.168.1.0/24
```

### Port Filters

```
# Specific port (TCP or UDP)
port 80
port 443

# Source port
src port 80

# Destination port
dst port 443

# Port range
portrange 8000-9000

# Multiple ports
port 80 or port 443 or port 8080

# Exclude port
not port 22
```

### Protocol Filters

```
# TCP only
tcp

# UDP only
udp

# ICMP
icmp

# ARP
arp

# IPv4
ip

# IPv6
ip6

# Specific protocol and port
tcp port 80
udp port 53

# Multiple protocols
tcp or udp
icmp or arp
```

### TCP Flags

```
# TCP SYN packets
tcp[tcpflags] & tcp-syn != 0
tcp[13] & 2 != 0

# TCP SYN-ACK
tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack)

# TCP RST
tcp[tcpflags] & tcp-rst != 0

# TCP FIN
tcp[tcpflags] & tcp-fin != 0

# TCP PSH
tcp[tcpflags] & tcp-push != 0

# No flags (NULL scan)
tcp[tcpflags] == 0

# Xmas scan (FIN, PSH, URG)
tcp[tcpflags] & (tcp-fin|tcp-push|tcp-urg) != 0
```

### Ethernet/MAC Filters

```
# Specific MAC address
ether host 00:11:22:33:44:55

# Source MAC
ether src 00:11:22:33:44:55

# Destination MAC
ether dst 00:11:22:33:44:55

# Broadcast
ether broadcast

# Multicast
ether multicast

# EtherType
ether proto 0x0800  # IPv4
ether proto 0x0806  # ARP
ether proto 0x86dd  # IPv6
```

### VLAN Filters

```
# VLAN traffic
vlan

# Specific VLAN ID
vlan 100

# VLAN and protocol
vlan and tcp
vlan 100 and host 192.168.1.1
```

### Complex Filters

```
# Combine host and port
host 192.168.1.1 and port 80

# Protocol and network
tcp and net 192.168.1.0/24

# Multiple conditions (AND)
host 192.168.1.1 and tcp and port 443

# Multiple conditions (OR)
host 192.168.1.1 or host 192.168.1.2

# Complex boolean logic
(host 192.168.1.1 or host 192.168.1.2) and port 80

# Exclude traffic
not host 192.168.1.1 and not port 22

# HTTP and HTTPS
tcp port 80 or tcp port 443

# DNS (TCP and UDP)
port 53
tcp port 53 or udp port 53

# Everything except SSH
not port 22

# Specific host on multiple ports
host 192.168.1.1 and (port 80 or port 443)

# Non-local traffic
not net 127.0.0.0/8
```

### Packet Size Filters

```
# Less than size
less 128

# Greater than size
greater 1000

# Specific length
len == 64

# Range
greater 100 and less 500
```

## Display Filters

Display filters use Wireshark's powerful filtering language.

### Basic Syntax

```
General format:
  protocol.field operator value

Operators:
  ==  (equal)
  !=  (not equal)
  >   (greater than)
  <   (less than)
  >=  (greater than or equal)
  <=  (less than or equal)
  contains
  matches (regex)
  in (set membership)

Logical:
  and (or &&)
  or  (or ||)
  not (or !)
  xor

Parentheses for grouping:
  (expression1) and (expression2)
```

### Creating Filters

**Method 1: Type in Filter Toolbar**
- Click filter input field
- Type filter expression
- Press Enter or click Apply
- Background colors:
  - Green: valid syntax
  - Red: invalid syntax
  - Yellow: valid but unusual

**Method 2: Right-Click in Packet Details**
- Right-click on field
- "Apply as Filter" →
  - Selected
  - Not Selected
  - ... and Selected
  - ... or Selected
  - ... and not Selected
  - ... or not Selected

**Method 3: Expression Builder**
- Click "Expression..." button in filter toolbar
- Browse field hierarchy
- Select field
- Choose relation
- Enter value
- Click OK

**Method 4: Filter Bookmarks**
- Create frequently used filters
- Save with descriptive names
- Quick access from dropdown

### IP Filters

```
# Any IP (source or destination)
ip.addr == 192.168.1.1

# Source IP
ip.src == 192.168.1.1

# Destination IP
ip.dst == 192.168.1.1

# IP subnet
ip.addr == 192.168.1.0/24
ip.src == 10.0.0.0/8

# Multiple IPs
ip.addr == 192.168.1.1 or ip.addr == 192.168.1.2

# IP in set
ip.addr in {192.168.1.1 192.168.1.2 192.168.1.3}

# IPv4 only
ip

# IPv6 only
ipv6

# Specific IPv6
ipv6.addr == 2001:db8::1

# IP TTL
ip.ttl < 10
ip.ttl == 64

# IP fragmentation
ip.flags.mf == 1      # More fragments
ip.frag_offset > 0    # Fragmented

# IP protocol
ip.proto == 6         # TCP
ip.proto == 17        # UDP
ip.proto == 1         # ICMP
```

### TCP Filters

```
# TCP port (source or destination)
tcp.port == 80

# TCP source port
tcp.srcport == 80

# TCP destination port
tcp.dstport == 443

# Port range
tcp.port >= 8000 and tcp.port <= 9000

# TCP flags
tcp.flags.syn == 1              # SYN
tcp.flags.ack == 1              # ACK
tcp.flags.fin == 1              # FIN
tcp.flags.reset == 1            # RST
tcp.flags.push == 1             # PSH
tcp.flags.urg == 1              # URG

# SYN-ACK packets
tcp.flags.syn == 1 and tcp.flags.ack == 1

# SYN only (connection initiation)
tcp.flags.syn == 1 and tcp.flags.ack == 0

# TCP window size
tcp.window_size < 1000
tcp.window_size_scalefactor > 0

# TCP sequence and ack numbers
tcp.seq == 0
tcp.ack == 1

# TCP stream index
tcp.stream == 0              # First TCP stream
tcp.stream == 5              # Sixth TCP stream

# TCP analysis (expert info)
tcp.analysis.retransmission         # Retransmissions
tcp.analysis.duplicate_ack          # Duplicate ACKs
tcp.analysis.duplicate_ack_num > 2  # More than 2 dup ACKs
tcp.analysis.lost_segment           # Lost segments
tcp.analysis.fast_retransmission    # Fast retransmissions
tcp.analysis.zero_window            # Zero window
tcp.analysis.window_full            # Window full
tcp.analysis.out_of_order           # Out of order
tcp.analysis.reused_ports           # Reused ports

# TCP options
tcp.options.mss                     # MSS option present
tcp.options.wscale                  # Window scale
tcp.options.sack_perm               # SACK permitted
tcp.options.timestamp               # Timestamp

# TCP payload
tcp.payload                         # Has payload
tcp.len > 0                         # Has data
tcp.len > 1000                      # Large segments
```

### UDP Filters

```
# UDP port
udp.port == 53

# UDP source port
udp.srcport == 5353

# UDP destination port
udp.dstport == 161

# UDP length
udp.length < 100
udp.length > 1000

# UDP stream
udp.stream == 0

# UDP checksum
udp.checksum_bad == 1
```

### HTTP Filters

```
# Any HTTP
http

# HTTP requests
http.request

# HTTP responses
http.response

# HTTP methods
http.request.method == "GET"
http.request.method == "POST"
http.request.method == "PUT"
http.request.method == "DELETE"
http.request.method == "HEAD"

# HTTP URI
http.request.uri == "/index.html"
http.request.uri contains "/api/"
http.request.uri matches "\\.(jpg|png|gif)$"

# HTTP full URI
http.request.full_uri contains "example.com"

# HTTP host
http.host == "www.example.com"
http.host contains "example"

# HTTP user agent
http.user_agent contains "Mozilla"
http.user_agent contains "curl"
http.user_agent contains "bot"

# HTTP referer
http.referer contains "google"

# HTTP response codes
http.response.code == 200
http.response.code == 404
http.response.code == 500
http.response.code >= 400               # Errors
http.response.code >= 400 and http.response.code < 500  # Client errors
http.response.code >= 500               # Server errors

# HTTP content type
http.content_type contains "application/json"
http.content_type contains "text/html"
http.content_type contains "image/"

# HTTP content length
http.content_length > 1000000
http.content_length < 100

# HTTP cookies
http.cookie
http.set_cookie

# HTTP authorization
http.authorization

# HTTP request headers
http.request.line contains "Accept-Language"

# HTTP response headers
http.server contains "Apache"
http.server contains "nginx"

# HTTP version
http.request.version == "HTTP/1.1"
http.request.version == "HTTP/2"

# HTTP with specific header
http.header contains "X-Custom-Header"

# HTTP file data
http.file_data contains "password"

# HTTP response time
http.time > 1.0                # Responses slower than 1 second
```

### DNS Filters

```
# Any DNS
dns

# DNS queries
dns.flags.response == 0

# DNS responses
dns.flags.response == 1

# DNS query name
dns.qry.name == "www.example.com"
dns.qry.name contains "example"
dns.qry.name matches ".*\\.com$"

# DNS query type
dns.qry.type == 1              # A record
dns.qry.type == 28             # AAAA record
dns.qry.type == 5              # CNAME
dns.qry.type == 15             # MX
dns.qry.type == 16             # TXT
dns.qry.type == 2              # NS
dns.qry.type == 6              # SOA
dns.qry.type == 12             # PTR

# DNS response code
dns.flags.rcode == 0           # No error
dns.flags.rcode == 3           # NXDOMAIN (name error)
dns.flags.rcode == 2           # Server failure

# DNS answers
dns.a                          # A record in answer
dns.aaaa                       # AAAA record
dns.cname                      # CNAME

# Specific IP in DNS answer
dns.a == 192.168.1.1
dns.aaaa == 2001:db8::1

# DNS flags
dns.flags.authoritative == 1
dns.flags.truncated == 1
dns.flags.recdesired == 1
dns.flags.recavail == 1

# DNS answer count
dns.count.answers > 5
dns.count.answers == 0

# DNS response time
dns.time > 0.1

# DNS over TCP (unusual)
dns and tcp

# Long DNS queries (possible tunneling)
dns.qry.name.len > 50
```

### TLS/SSL Filters

```
# Any TLS
tls
ssl   # Older captures

# TLS handshake
tls.handshake
ssl.handshake

# TLS handshake types
tls.handshake.type == 1        # Client Hello
tls.handshake.type == 2        # Server Hello
tls.handshake.type == 11       # Certificate
tls.handshake.type == 12       # Server Key Exchange
tls.handshake.type == 14       # Server Hello Done
tls.handshake.type == 16       # Client Key Exchange
tls.handshake.type == 20       # Finished

# TLS SNI (Server Name Indication)
tls.handshake.extensions_server_name == "www.example.com"
tls.handshake.extensions_server_name contains "example"

# TLS version
tls.record.version == 0x0303   # TLS 1.2
tls.record.version == 0x0304   # TLS 1.3
tls.handshake.version == 0x0303

# TLS cipher suites
tls.handshake.ciphersuite
tls.handshake.ciphersuite == 0xc02f  # TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256

# TLS cipher suite in Client Hello
tls.handshake.ciphersuites

# TLS certificate
tls.handshake.certificate

# TLS application data
tls.app_data

# TLS alerts
tls.alert_message
tls.alert_message.level == 2   # Fatal
tls.alert_message.desc == 40   # Handshake failure

# TLS extensions
tls.handshake.extension.type == 0  # Server name
tls.handshake.extension.type == 10 # Supported groups
tls.handshake.extension.type == 13 # Signature algorithms

# Weak/old SSL
ssl.record.version < 0x0303    # Older than TLS 1.2

# Certificate details
x509sat.uTF8String
x509sat.printableString
x509ce.dNSName
```

### ICMP Filters

```
# All ICMP
icmp

# ICMP type
icmp.type == 8                 # Echo request (ping)
icmp.type == 0                 # Echo reply
icmp.type == 3                 # Destination unreachable
icmp.type == 5                 # Redirect
icmp.type == 11                # Time exceeded

# ICMP code
icmp.code == 0
icmp.code == 3                 # Port unreachable

# ICMPv6
icmpv6
icmpv6.type == 128             # Echo request
icmpv6.type == 129             # Echo reply

# ICMP echo request/reply pairs
icmp.type == 8 or icmp.type == 0

# ICMP response time
icmp.resptime
icmp.resptime > 0.1
```

### ARP Filters

```
# All ARP
arp

# ARP request
arp.opcode == 1

# ARP reply
arp.opcode == 2

# ARP for specific IP
arp.dst.proto_ipv4 == 192.168.1.1
arp.src.proto_ipv4 == 192.168.1.1

# ARP with specific MAC
arp.src.hw_mac == 00:11:22:33:44:55
arp.dst.hw_mac == 00:11:22:33:44:55

# Gratuitous ARP
arp.opcode == 1 and arp.src.proto_ipv4 == arp.dst.proto_ipv4

# ARP duplicate address detection
arp.duplicate-address-detected
arp.duplicate-address-frame
```

### DHCP Filters

```
# All DHCP
dhcp
bootp  # Alternative

# DHCP message types
dhcp.option.dhcp == 1          # Discover
dhcp.option.dhcp == 2          # Offer
dhcp.option.dhcp == 3          # Request
dhcp.option.dhcp == 5          # ACK
dhcp.option.dhcp == 6          # NAK
dhcp.option.dhcp == 7          # Release
dhcp.option.dhcp == 8          # Inform

# DHCP for specific MAC
dhcp.hw.mac_addr == 00:11:22:33:44:55

# DHCP assigned IP
dhcp.ip.your == 192.168.1.100

# DHCP server
dhcp.option.dhcp_server_id == 192.168.1.1

# DHCP hostname
dhcp.option.hostname contains "laptop"

# DHCP lease time
dhcp.option.dhcp_lease_time
dhcp.option.dhcp_lease_time > 86400

# DHCP domain name
dhcp.option.domain_name
```

### SMB Filters

```
# All SMB
smb or smb2

# SMB version 1
smb

# SMB version 2/3
smb2

# SMB commands (SMB2)
smb2.cmd == 0                  # Negotiate
smb2.cmd == 1                  # Session Setup
smb2.cmd == 3                  # Tree Connect
smb2.cmd == 5                  # Create
smb2.cmd == 8                  # Read
smb2.cmd == 9                  # Write
smb2.cmd == 6                  # Close

# SMB filename
smb2.filename contains "document"
smb.file contains "document"

# SMB tree (share)
smb2.tree

# SMB NT status
smb2.nt_status != 0x00000000   # Errors
smb2.nt_status == 0xc0000022   # Access denied

# NTLMSSP (authentication)
ntlmssp
ntlmssp.auth.username
ntlmssp.auth.domain
```

### FTP Filters

```
# FTP control
ftp

# FTP commands
ftp.request.command == "USER"
ftp.request.command == "PASS"
ftp.request.command == "RETR"
ftp.request.command == "STOR"
ftp.request.command == "LIST"

# FTP responses
ftp.response.code == 220       # Welcome
ftp.response.code == 230       # Login successful
ftp.response.code == 530       # Not logged in

# FTP data
ftp-data

# FTP arguments
ftp.request.arg
```

### SMTP Filters

```
# SMTP
smtp

# SMTP commands
smtp.req.command == "MAIL"
smtp.req.command == "RCPT"
smtp.req.command == "DATA"
smtp.req.command == "HELO"
smtp.req.command == "EHLO"

# SMTP responses
smtp.response.code == 250
smtp.response.code == 550

# Email addresses
smtp.req.parameter contains "@example.com"

# IMF (email message)
imf
imf.from contains "example.com"
imf.to contains "user@example.com"
imf.subject contains "urgent"
```

### Database Filters

```
# MySQL
mysql
mysql.query
mysql.command == 3             # Query
mysql.query contains "SELECT"

# PostgreSQL
pgsql
pgsql.query

# MongoDB
mongo
mongo.op == 2004               # Query

# Redis
redis
redis.command
```

### String Matching

```
# Contains
http.host contains "example"
dns.qry.name contains "google"

# Matches (regex)
http.host matches "^www\\..*\\.com$"
dns.qry.name matches ".*\\.(tk|ml|ga|cf)$"

# Case sensitivity (always case-insensitive)
http.host contains "EXAMPLE"  # Matches "example.com"

# Equals
http.host == "www.example.com"

# In set
http.host in {"example.com" "test.com" "demo.com"}
ip.addr in {192.168.1.1 192.168.1.2}
```

### Comparison Operators

```
# Equals
tcp.port == 80

# Not equals
tcp.port != 22

# Greater than
frame.len > 1000
tcp.window_size > 65535

# Less than
ip.ttl < 10
http.time < 0.01

# Greater than or equal
tcp.port >= 8000

# Less than or equal
tcp.port <= 9000

# Range
tcp.port >= 8000 and tcp.port <= 9000
frame.len > 100 and frame.len < 1500
```

### Frame/Packet Filters

```
# Frame number
frame.number == 100
frame.number > 1000
frame.number >= 100 and frame.number <= 200

# Frame time
frame.time >= "2024-01-01 00:00:00"
frame.time <= "2024-12-31 23:59:59"

# Time relative to first packet
frame.time_relative > 10

# Time delta (since previous packet)
frame.time_delta > 1.0
frame.time_delta_displayed > 0.5

# Frame length
frame.len > 1000
frame.len < 100
frame.len == 54

# Marked packets
frame.marked == 1

# Ignored packets
frame.ignored == 1

# Interface
frame.interface_name == "eth0"
```

### Expert Info Filters

```
# Any expert info
expert

# By severity
expert.severity == "error"
expert.severity == "warn"
expert.severity == "note"
expert.severity == "chat"

# By group
expert.group == "Checksum"
expert.group == "Sequence"
expert.group == "Malformed"
expert.group == "Protocol"

# Expert message
expert.message contains "Retransmission"
```

### Logical Operators

```
# AND (both conditions must be true)
ip.addr == 192.168.1.1 and tcp.port == 80
ip.addr == 192.168.1.1 && tcp.port == 80

# OR (either condition can be true)
tcp.port == 80 or tcp.port == 443
tcp.port == 80 || tcp.port == 443

# NOT (condition must be false)
not icmp
!icmp
!(tcp.port == 22)

# XOR (exclusive or)
tcp xor udp

# Parentheses for grouping
(ip.src == 192.168.1.1 or ip.src == 192.168.1.2) and tcp.port == 80
```

### Advanced Filter Examples

```
# HTTP POST requests to specific host
http.request.method == "POST" and http.host == "api.example.com"

# Failed HTTP requests
http.response.code >= 400

# Large HTTP responses
http.response and frame.len > 10000

# TCP retransmissions to specific IP
tcp.analysis.retransmission and ip.dst == 192.168.1.1

# TLS connections to specific domains
tls.handshake.type == 1 and tls.handshake.extensions_server_name contains "example"

# Non-standard HTTP ports
http and tcp.port != 80 and tcp.port != 443

# Broadcast and multicast
eth.dst.ig == 1

# IPv6 traffic on subnet
ipv6.src == 2001:db8::/32

# Suspicious DNS (many answers)
dns.flags.response == 1 and dns.count.answers > 10

# Slow DNS responses
dns.time > 0.5

# Zero window condition
tcp.analysis.zero_window

# Out of order packets
tcp.analysis.out_of_order

# SYN flood detection pattern
tcp.flags.syn == 1 and tcp.flags.ack == 0

# Fragmented IP packets
ip.frag_offset > 0 or ip.flags.mf == 1
```

### Filter Macros

Create reusable filter expressions:

1. Analyze → Display Filter Macros
2. Click "+"
3. Name: `mynetwork`
4. Text: `ip.addr == 192.168.1.0/24`
5. Use in filters: `${mynetwork} and http`

## Packet Analysis Features

### Following Streams

**Follow TCP Stream:**
1. Right-click on TCP packet
2. Follow → TCP Stream
3. New window shows conversation
4. Options:
   - ASCII
   - EBCDIC
   - Hex Dump
   - C Arrays
   - Raw
5. Filter automatically applied: `tcp.stream eq N`
6. Can save stream as file

**Follow UDP Stream:**
- Same as TCP stream
- Right-click UDP packet → Follow → UDP Stream

**Follow HTTP Stream:**
- Right-click HTTP packet → Follow → HTTP Stream
- Shows formatted HTTP request/response

**Follow TLS Stream:**
- Right-click TLS packet → Follow → TLS Stream
- Shows encrypted data (unless keys provided)
- With keys: shows decrypted data

**Stream Features:**
- Red text: client → server
- Blue text: server → client
- Find within stream
- Filter on current stream
- Save stream data

### Protocol Hierarchy

**View Protocol Breakdown:**
- Statistics → Protocol Hierarchy
- Shows:
  - Packet count per protocol
  - Percentage of total
  - Bytes per protocol
  - Hierarchical view

**Features:**
- Click to apply filter
- See protocol distribution
- Identify unusual protocols
- Export as text/CSV

### Conversations

**View Conversations:**
- Statistics → Conversations
- Tabs for different layers:
  - Ethernet
  - IPv4/IPv6
  - TCP
  - UDP
- Shows:
  - Address A ↔ Address B
  - Packets
  - Bytes
  - Bits/sec
  - Duration

**Features:**
- Sort by any column
- Follow stream from here
- Apply as filter
- Color packets
- Copy to clipboard
- Export data

### Endpoints

**View Endpoints:**
- Statistics → Endpoints
- Tabs for layers:
  - Ethernet
  - IPv4/IPv6
  - TCP
  - UDP
- Shows statistics per endpoint:
  - Packets
  - Bytes
  - Tx packets/bytes
  - Rx packets/bytes

**Features:**
- Map endpoint (GeoIP)
- Apply as filter
- Export data

### IO Graphs

**Create IO Graphs:**
- Statistics → IO Graph
- Default: packets per second over time
- Multiple graphs (up to 5)
- Per-graph options:
  - Display filter
  - Color
  - Style (Line, Impulse, Bar, etc.)
  - Y-axis metric:
    - Packets/Bytes/Bits per tick
    - Advanced (SUM, MIN, MAX, AVG)

**Use Cases:**
- Visualize traffic patterns
- Identify traffic spikes
- Compare protocols
- Analyze trends

**Example Graphs:**
```
Graph 1: All traffic (no filter)
Graph 2: tcp, color=blue
Graph 3: udp, color=green
Graph 4: http, color=red
Graph 5: dns, color=purple
```

### Expert Information

**View Expert Info:**
- Analyze → Expert Information
- Categorized by severity:
  - Errors (red)
  - Warnings (yellow)
  - Notes (cyan)
  - Chats (blue)

**Common Expert Info:**

**Errors:**
- Malformed packets
- Checksum errors
- Bad TCP

**Warnings:**
- TCP retransmissions
- TCP duplicate ACK
- TCP zero window
- TCP previous segment not captured
- HTTP response code ≥ 400

**Notes:**
- TCP fast retransmission
- TCP keep-alive
- HTTP compressed response

**Chats:**
- Connection establish/close
- Sequence number errors

**Features:**
- Click to navigate to packet
- Group by summary
- Apply as filter
- Export data

### Time Display Formats

**Change Time Display:**
- View → Time Display Format

**Options:**
- Date and Time of Day
- Time of Day
- Seconds Since Beginning of Capture
- Seconds Since Previous Captured Packet
- Seconds Since Previous Displayed Packet
- Seconds Since Epoch (1970-01-01)
- UTC Date and Time of Day
- UTC Time of Day

**Precision:**
- Automatic
- Seconds
- Deciseconds
- Centiseconds
- Milliseconds
- Microseconds
- Nanoseconds

### Name Resolution

**Enable/Disable Resolution:**
- View → Name Resolution

**Types:**
- Resolve MAC Addresses
- Resolve Transport Names (ports)
- Resolve Network Addresses (DNS)
- Use Captured DNS Packets
- Use External Resolvers

**Configure:**
- Edit → Preferences → Name Resolution
- Enable concurrent DNS lookups
- Maximum concurrent requests
- Custom hosts file
- Custom SMI MIB paths

### Time References

**Set Time Reference:**
- Right-click packet → Set/Unset Time Reference
- Packet marked with `*REF*`
- Relative times calculated from reference
- Multiple references allowed

**Use Cases:**
- Measure time between events
- Align multiple captures
- Focus on specific time periods

### Packet Marking

**Mark Packets:**
- Right-click → Mark/Unmark Packet (Ctrl+M)
- Marked packets shown with black background
- Mark all displayed: Edit → Mark All Displayed
- Unmark all: Edit → Unmark All Packets

**Use Cases:**
- Flag interesting packets
- Export only marked packets
- Navigate between important packets

### Ignoring Packets

**Ignore Packets:**
- Right-click → Ignore/Unignore Packet
- Ignored packets grayed out
- Hidden from statistics
- Can still be displayed

**Use Cases:**
- Exclude noise
- Focus on specific conversation
- Remove known-good traffic

### Packet Comments

**Add Comments:**
- Right-click → Packet Comment
- Add notes to specific packets
- Comments saved in pcapng format
- Comments shown in packet list

**Capture Comments:**
- Statistics → Capture File Properties
- Add overall capture comments
- Useful for documentation

## Coloring Rules

Wireshark uses coloring rules for visual packet identification.

### Default Coloring Rules

- **Light purple**: Bad TCP (errors, retransmissions)
- **Light green**: HTTP
- **Light blue**: UDP
- **Light yellow**: Routing protocols
- **Pink**: ICMP
- **Dark gray**: TCP
- **Light gray**: UDP
- **Black**: TCP packets with problems
- **Red text**: TCP errors
- **Yellow**: SMB
- **White**: Other traffic

### Viewing Coloring Rules

**Access:**
- View → Coloring Rules

**Rule Structure:**
- Name
- Filter expression
- Foreground color
- Background color
- Enabled checkbox

**Order Matters:**
- Rules evaluated top to bottom
- First matching rule wins
- Can reorder with buttons

### Creating Custom Coloring Rules

**Add New Rule:**
1. View → Coloring Rules
2. Click "+" button
3. Set name: "My Important Traffic"
4. Set filter: `ip.addr == 192.168.1.1`
5. Choose foreground color
6. Choose background color
7. Click OK

**Example Rules:**
```
Name: SSH Traffic
Filter: tcp.port == 22
Foreground: White
Background: Dark Blue

Name: DNS Errors
Filter: dns.flags.rcode != 0
Foreground: White
Background: Red

Name: Slow HTTP
Filter: http.time > 1.0
Foreground: Black
Background: Orange

Name: My Network
Filter: ip.src == 192.168.1.0/24 or ip.dst == 192.168.1.0/24
Foreground: Black
Background: Light Cyan
```

### Temporary Coloring

**Quick Colorize:**
- Right-click packet
- Colorize Conversation →
  - Ethernet
  - IPv4
  - TCP
  - UDP
- Automatically applies color to conversation
- Temporary (not saved)

**Reset Coloring:**
- View → Reset Coloring 1-10
- View → Colorize Packet List (toggle on/off)

### Exporting/Importing Rules

**Export Rules:**
- View → Coloring Rules
- Export button
- Save as text file

**Import Rules:**
- View → Coloring Rules
- Import button
- Select saved rules file

## Statistics Windows

### Capture File Properties

**View Properties:**
- Statistics → Capture File Properties

**Information Shown:**
- File name and path
- File format
- File size
- Packet size limits
- Time span
- Packet counts
- Data rate
- Interface information
- Comments

### Resolved Addresses

**View Resolved Names:**
- Statistics → Resolved Addresses

**Shows:**
- Ethernet (MAC) addresses
- IPv4/IPv6 addresses
- TCP/UDP ports
- From captured DNS responses
- From system resolution

**Export:**
- Can save as CSV or text

### Protocol Hierarchy Advanced

**Detailed Analysis:**
- Statistics → Protocol Hierarchy
- Shows nested protocols
- Percentage calculations
- Byte counts per protocol

**Apply as Filter:**
- Right-click protocol
- Apply as Filter
- Automatically filters display

### Packet Lengths

**Distribution:**
- Statistics → Packet Lengths

**Shows:**
- Packet size ranges
- Count per range
- Percentage
- Cumulative percentage

**Useful For:**
- Identifying traffic patterns
- Finding fragmentation
- Detecting anomalies

### HTTP Statistics

**HTTP Request/Response:**
- Statistics → HTTP → Requests
- Statistics → HTTP → Load Distribution
- Statistics → HTTP → Request Sequences

**Shows:**
- Request methods
- Status codes
- Hosts
- URIs
- Response times

### DNS Statistics

**DNS Analysis:**
- Statistics → DNS

**Shows:**
- Query types
- Response codes
- Top talkers
- Service response times

### Service Response Time

**Measure Performance:**
- Statistics → Service Response Time

**Protocols:**
- DNS
- HTTP
- SMB
- NFS
- iSCSI

**Metrics:**
- Min/Max/Average response time
- Request/Response pairs
- Distribution graphs

### Flow Graphs

**TCP Flow Graph:**
- Statistics → Flow Graph
- Shows TCP conversation flow
- Time sequence diagram
- Visualizes handshakes, data transfer, termination

**Options:**
- General flow graph (all flows)
- TCP flow graph (specific stream)
- Limit to displayed packets
- Show comment

### TCP Stream Graphs

**Advanced TCP Analysis:**
- Statistics → TCP Stream Graphs

**Graph Types:**

**1. Stevens Graph:**
- Sequence number vs. time
- Shows retransmissions
- Identifies packet loss
- Standard tcpdump-style

**2. tcptrace Graph:**
- Time sequence (tcptrace)
- Outstanding bytes
- Shows window scaling

**3. Throughput Graph:**
- Goodput over time
- Effective data transfer rate

**4. Round Trip Time Graph:**
- RTT vs. sequence number
- Latency analysis

**5. Window Scaling Graph:**
- Window size over time
- Congestion control visualization

**Features:**
- Zoom in/out
- Pan graph
- Switch between graphs
- Export as image

### Multicast Statistics

**Multicast Streams:**
- Statistics → Multicast Streams

**Shows:**
- Source address
- Multicast group
- Packets
- Bursts
- Max/Average burst rates

## Decryption

### TLS/SSL Decryption

**Using Pre-Master Secret Log:**

1. **Configure Browser/Application:**
   ```bash
   # Set environment variable (Firefox, Chrome, etc.)
   export SSLKEYLOGFILE=~/sslkeys.log

   # Windows
   set SSLKEYLOGFILE=%USERPROFILE%\sslkeys.log

   # macOS
   export SSLKEYLOGFILE=~/sslkeys.log
   ```

2. **Configure Wireshark:**
   - Edit → Preferences
   - Protocols → TLS (or SSL)
   - (Pre)-Master-Secret log filename: [browse to sslkeys.log]
   - Click OK

3. **Capture Traffic:**
   - Start browser/application
   - Generate HTTPS traffic
   - Keys automatically logged

4. **View Decrypted:**
   - HTTP traffic now visible
   - Follow HTTP stream shows decrypted data
   - Export HTTP objects works

**Using RSA Private Key:**

1. **Requirements:**
   - Server's private key file
   - RSA key exchange (not DHE/ECDHE)
   - Key not password-protected

2. **Configure:**
   - Edit → Preferences → Protocols → TLS
   - RSA keys list → Edit
   - Add entry:
     - IP address: 192.168.1.1
     - Port: 443
     - Protocol: http
     - Key File: [path to private key]
     - Password: [if protected]

3. **Limitations:**
   - Only works with RSA key exchange
   - Doesn't work with Forward Secrecy (DHE/ECDHE)
   - Need private key from server

### WPA/WPA2 Decryption

**Decrypt Wireless:**

1. **Requirements:**
   - Capture must include 4-way handshake
   - Know the PSK (pre-shared key / password)

2. **Configure:**
   - Edit → Preferences
   - Protocols → IEEE 802.11
   - Enable decryption
   - Decryption keys → Edit
   - Add key:
     - Type: wpa-pwd
     - Key: password:SSID

3. **Example:**
   ```
   Key type: wpa-pwd
   Key: MyPassword123:MyWiFiNetwork
   ```

4. **Capture 4-Way Handshake:**
   - Must capture client connecting
   - Or deauthenticate client to force reconnect
   - Wireshark shows "EAPOL" packets

5. **Verify Decryption:**
   - Should see decrypted data frames
   - IP, TCP, HTTP traffic visible

### IPsec Decryption

**Configure IPsec:**

1. **Edit → Preferences → Protocols → ESP**
2. **Add SA (Security Association):**
   - Protocol: ESP
   - SPI: [hex value]
   - Encryption algorithm
   - Encryption key
   - Authentication algorithm
   - Authentication key

3. **Obtain Keys:**
   - From IKE negotiation
   - From manual configuration
   - From IPsec logs

### Kerberos Decryption

**Decrypt Kerberos:**

1. **Requirements:**
   - Kerberos keytab file
   - Or specific keys

2. **Configure:**
   - Edit → Preferences → Protocols → KRB5
   - Kerberos keytab file: [path]

3. **Use:**
   - Decrypts Kerberos tickets
   - Shows encrypted payloads

## Exporting and Reporting

### Export Packet Dissections

**Export Formats:**
- File → Export Packet Dissections

**Formats Available:**

**1. Plain Text:**
- Human-readable
- Similar to screen output
- Options:
  - Packet summary line
  - Packet details
  - Packet bytes

**2. CSV:**
- Comma-separated values
- Specify fields to export
- Easy to import to Excel/database

**3. JSON:**
- Machine-readable
- Structured data
- All packet details

**4. C Arrays:**
- For test data in code
- Hex arrays

**5. PSML (XML):**
- Packet summary
- Structured XML

**6. PDML (XML):**
- Packet details
- Complete dissection

### Export Objects

**Extract Files:**
- File → Export Objects

**Protocols Supported:**
- HTTP/HTTPS
- SMB/SMB2
- TFTP
- DICOM
- IMF (Email)

**HTTP Export:**
1. File → Export Objects → HTTP
2. Window shows all HTTP objects:
   - Packet number
   - Hostname
   - Content type
   - Size
   - Filename
3. Select object(s)
4. Click "Save" or "Save All"

**Use Cases:**
- Extract downloaded files
- Analyze transferred data
- Forensic investigation
- Malware analysis

### Export Specified Packets

**Save Subset:**
1. File → Export Specified Packets
2. Choose:
   - All packets
   - Selected packet
   - Marked packets
   - First to last marked
   - Range
   - Displayed packets (current filter)
   - Captured packets (all)
3. Select format (pcap, pcapng)
4. Save

**Common Uses:**
- Share specific traffic
- Reduce file size
- Create test cases

### Export Bytes

**Export Packet Bytes:**
1. Select packet
2. Right-click in Packet Bytes pane
3. Export Packet Bytes
4. Save as binary file

**Use Cases:**
- Extract payloads
- Save file fragments
- Binary analysis

### Print Packets

**Print Options:**
1. File → Print
2. Choose:
   - Packet format:
     - Summary line only
     - Details (with summary line)
     - Bytes and summary
     - Details and bytes
   - Packet range
3. Print to:
   - Printer
   - File (PostScript, PDF)

### Statistics Export

Most statistics windows have export options:
- Copy to clipboard
- Save as CSV
- Save as plain text
- Save as XML

## Command-Line Tools

Wireshark includes several command-line tools.

### TShark

Command-line packet analyzer:

```bash
# Live capture
tshark -i eth0

# Read file
tshark -r capture.pcap

# With display filter
tshark -r capture.pcap -Y "http"

# Extract fields
tshark -r capture.pcap -T fields -e ip.src -e ip.dst

# See tshark.md for comprehensive documentation
```

### Editcap

Packet file editor:

```bash
# Split by packet count
editcap -c 1000 input.pcap output.pcap
# Creates output_00000.pcap, output_00001.pcap, etc.

# Split by time
editcap -i 60 input.pcap output.pcap
# New file every 60 seconds

# Extract packet range
editcap -r input.pcap output.pcap 100-200

# Remove duplicates
editcap -d input.pcap output.pcap
editcap -D 5 input.pcap output.pcap  # Duplicate window of 5

# Change timestamp
editcap -t +3600 input.pcap output.pcap  # Add 1 hour

# Adjust snaplen
editcap -s 96 input.pcap output.pcap  # Truncate to 96 bytes

# Extract only displayed packets (with filter)
tshark -r input.pcap -Y "http" -w output.pcap

# Change file format
editcap -F pcap input.pcapng output.pcap

# Anonymize IP addresses
editcap -a 192.168.1.0/24:10.0.0.0/24 input.pcap output.pcap
```

### Mergecap

Merge multiple capture files:

```bash
# Basic merge (chronological)
mergecap -w output.pcap file1.pcap file2.pcap file3.pcap

# Merge all files in directory
mergecap -w output.pcap *.pcap

# Append (don't sort)
mergecap -a -w output.pcap file1.pcap file2.pcap

# Set snapshot length
mergecap -s 65535 -w output.pcap file1.pcap file2.pcap

# Change output format
mergecap -F pcapng -w output.pcapng file1.pcap file2.pcap

# Verbose output
mergecap -v -w output.pcap file1.pcap file2.pcap
```

### Capinfos

Display capture file information:

```bash
# Basic info
capinfos capture.pcap

# Detailed info
capinfos -d capture.pcap

# Statistics
capinfos -s capture.pcap

# All info
capinfos -a capture.pcap

# Table format
capinfos -T capture.pcap

# Specific fields
capinfos -t -c -u capture.pcap
# -t: time
# -c: packet count
# -u: packet size

# Machine-readable
capinfos -M capture.pcap

# Multiple files
capinfos file1.pcap file2.pcap file3.pcap
```

**Output Example:**
```
File name:           capture.pcap
File type:           Wireshark/tcpdump/... - pcap
File encapsulation:  Ethernet
Packet size limit:   file hdr: 65535 bytes
Number of packets:   1234
File size:           567890 bytes
Data size:           554321 bytes
Capture duration:    123.456 seconds
Start time:          Mon Jan 1 12:00:00 2024
End time:            Mon Jan 1 12:02:03 2024
Data byte rate:      4491 bytes/s
Data bit rate:       35928 bits/s
Average packet size: 449.21 bytes
Average packet rate: 10 packets/s
```

### Text2pcap

Convert hex dump to pcap:

```bash
# Basic conversion
text2pcap hexdump.txt output.pcap

# Specify Ethernet encapsulation
text2pcap -e 0x0800 hexdump.txt output.pcap

# Add dummy Ethernet header
text2pcap -e 0x0800 -l 1 hexdump.txt output.pcap

# UDP encapsulation
text2pcap -u 1234,5678 hexdump.txt output.pcap
# Source port 1234, destination port 5678

# TCP encapsulation
text2pcap -T 1234,5678 hexdump.txt output.pcap

# With timestamp
text2pcap -t "%Y-%m-%d %H:%M:%S." hexdump.txt output.pcap
```

**Input Format:**
```
000000 00 11 22 33 44 55 66 77 88 99 aa bb 08 00 45 00
000010 00 3c 1c 46 40 00 40 06 b1 e6 c0 a8 01 01 c0 a8
```

### Dumpcap

Efficient packet capture (no GUI):

```bash
# List interfaces
dumpcap -D

# Capture on interface
dumpcap -i eth0 -w capture.pcap

# Capture with autostop
dumpcap -i eth0 -w capture.pcap -a duration:60

# Ring buffer
dumpcap -i eth0 -w capture.pcap -b filesize:10000 -b files:5

# With capture filter
dumpcap -i eth0 -f "tcp port 80" -w capture.pcap

# Multiple interfaces
dumpcap -i eth0 -i wlan0 -w capture.pcap

# Quiet mode
dumpcap -i eth0 -w capture.pcap -q
```

**Advantages:**
- Lower overhead than Wireshark
- No GUI processing
- Minimal packet loss
- Better for high-speed capture

### Rawshark

Read and analyze packets from stdin:

```bash
# Read from pipe
tcpdump -i eth0 -w - | rawshark -r -

# With fields
rawshark -r capture.pcap -d proto:http -F http.request.uri

# Process live capture
dumpcap -i eth0 -w - | rawshark -r -
```

## Configuration Profiles

Profiles allow different Wireshark configurations.

### Using Profiles

**Built-in Profiles:**
- Default
- Bluetooth
- Classic (old Wireshark look)

**Current Profile:**
- Shown in status bar (bottom right)
- Click to switch

### Creating Profiles

**Create New:**
1. Edit → Configuration Profiles
2. Click "+" button
3. Name: "Web Development"
4. Click OK

**Configure Profile:**
1. Switch to profile
2. Configure:
   - Columns
   - Coloring rules
   - Preferences
   - Display filters
   - Capture filters
3. Settings saved to profile

### Profile-Specific Settings

Each profile saves:
- Preferences
- Capture filter history
- Display filter history
- Coloring rules
- Disabled protocols
- Column settings
- Recent files
- Filter bookmarks

### Managing Profiles

**Import/Export:**
1. Configuration Profiles
2. Select profile
3. Copy or Delete buttons
4. Export/Import from directory:
   ```bash
   # Linux/macOS
   ~/.config/wireshark/profiles/

   # Windows
   %APPDATA%\Wireshark\profiles\
   ```

### Use Cases

**Example Profiles:**

**1. Web Development:**
- HTTP/HTTPS focused
- Custom columns: http.request.method, http.host, http.response.code
- Coloring: HTTP errors in red
- Common filters saved

**2. VoIP Analysis:**
- SIP/RTP focused
- Telephony windows accessible
- RTP stream analysis
- Custom columns for VoIP

**3. Security Analysis:**
- Expert info prominent
- Suspicious traffic colored
- Filters for common attacks
- Malware indicators

**4. Wireless:**
- 802.11 decryption configured
- WPA keys saved
- Wireless-specific filters
- Channel/signal columns

## Common Use Cases and Patterns

### Network Troubleshooting

**Connection Issues:**
```
1. Verify connectivity:
   - Filter: ip.addr == [target]
   - Check for responses

2. Check TCP handshake:
   - Filter: tcp.flags.syn == 1
   - Look for SYN, SYN-ACK, ACK

3. Identify failures:
   - Filter: tcp.flags.reset == 1 or icmp.type == 3
```

**Slow Network:**
```
1. Find retransmissions:
   - Filter: tcp.analysis.retransmission
   - Check percentage

2. Check TCP window:
   - Filter: tcp.analysis.zero_window
   - Indicates receiver overwhelmed

3. Analyze response times:
   - Statistics → Service Response Time
   - Identify slow services
```

**DNS Problems:**
```
1. Check DNS queries:
   - Filter: dns.flags.response == 0

2. Find errors:
   - Filter: dns.flags.rcode != 0
   - Look for NXDOMAIN, SERVFAIL

3. Slow resolution:
   - Filter: dns.time > 1.0
   - Check response times
```

### Security Analysis

**Port Scanning Detection:**
```
1. Many SYN packets:
   - Filter: tcp.flags.syn == 1 and tcp.flags.ack == 0
   - Statistics → Conversations
   - Look for one source, many destinations

2. Identify scanner:
   - Check source IP
   - Note targeted ports
   - Document scan pattern
```

**Malware Traffic:**
```
1. Suspicious connections:
   - Filter: http.request
   - Look for unusual user agents
   - Check for encoded data

2. DNS tunneling:
   - Filter: dns.qry.name.len > 50
   - Look for long random-looking domains

3. C2 beaconing:
   - Statistics → IO Graph
   - Look for regular intervals
   - Consistent packet sizes
```

**Credential Theft:**
```
1. Clear text passwords:
   - Filter: http.authorization
   - Follow → TCP Stream

2. FTP credentials:
   - Filter: ftp.request.command == "USER" or ftp.request.command == "PASS"

3. NTLM hashes:
   - Filter: ntlmssp.auth
```

### Application Debugging

**HTTP API Issues:**
```
1. Find failed requests:
   - Filter: http.response.code >= 400

2. Check specific endpoint:
   - Filter: http.request.uri contains "/api/users"

3. Analyze timing:
   - Filter: http.time > 1.0
   - Find slow requests
```

**Database Queries:**
```
1. MySQL slow queries:
   - Filter: mysql.query
   - Check query content

2. Failed connections:
   - Filter: mysql.error_message
```

**WebSocket Debugging:**
```
1. Find WebSocket traffic:
   - Filter: websocket

2. Check messages:
   - Filter: websocket.payload
   - Follow → TCP Stream
```

### VoIP Analysis

**SIP Call Analysis:**
```
1. View all calls:
   - Telephony → VoIP Calls
   - Shows all SIP sessions

2. Analyze specific call:
   - Select call
   - Click "Flow Sequence"
   - See call setup/teardown

3. Check RTP quality:
   - Telephony → RTP → Stream Analysis
   - Check packet loss, jitter, MOS
```

**Audio Playback:**
```
1. Telephony → RTP → RTP Player
2. Select streams
3. Click "Play"
4. Listen to audio quality
```

### Performance Analysis

**Identify Bandwidth Hogs:**
```
1. Statistics → Conversations → IPv4
2. Sort by "Bytes"
3. Identify top talkers
4. Apply as filter to investigate
```

**Protocol Distribution:**
```
1. Statistics → Protocol Hierarchy
2. See percentage breakdown
3. Identify unexpected protocols
```

**Traffic Over Time:**
```
1. Statistics → IO Graph
2. View packets/bytes per second
3. Identify spikes
4. Correlate with issues
```

### Wireless Analysis

**Find Networks:**
```
1. Filter: wlan.fc.type_subtype == 8
   (Beacon frames)
2. Statistics → WLAN Traffic
3. See all SSIDs
```

**Capture Handshake:**
```
1. Start wireless capture in monitor mode
2. Filter: eapol
3. Look for 4-way handshake (4 EAPOL packets)
4. Save for password cracking or decryption
```

**Analyze Performance:**
```
1. Check retries:
   - Filter: wlan.fc.retry == 1

2. Check signal:
   - Add column: radiotap.dbm_antsignal

3. Check channel utilization:
   - Statistics → WLAN Traffic
```

## Best Practices

### Capture Best Practices

**1. Use Capture Filters:**
- Filter at capture time to reduce data
- Save disk space and memory
- Improve performance
```
Examples:
host 192.168.1.1
tcp port 80 or tcp port 443
not port 22
```

**2. Set Appropriate Snapshot Length:**
- Full packets (default): 0 or 65535
- Headers only: 96 bytes
- Conservative: 256 bytes
```
Capture Options → Snapshot length: 96
```

**3. Use Ring Buffers:**
- Prevent disk fill
- Continuous monitoring
- Automatic rotation
```
Capture Options → Output:
☑ Create new file every 100000 kilobytes
☑ Use ring buffer: 10 files
```

**4. Name Files Descriptively:**
```
Good:
  2024-01-15_web-server-issue_eth0.pcapng
  prod-db-slowness-2024-01-15.pcap

Bad:
  capture1.pcap
  test.pcap
  new.pcap
```

**5. Document Captures:**
```
Statistics → Capture File Properties → Edit
Add comments:
  "Captured during reported slowness at 14:30
   Server: 192.168.1.50
   Client: 192.168.1.100
   Symptom: 5-second page load times"
```

### Analysis Best Practices

**1. Start with Statistics:**
```
Before diving into packets:
- Statistics → Protocol Hierarchy (what protocols?)
- Statistics → Conversations (who's talking?)
- Statistics → Endpoints (top talkers?)
- Analyze → Expert Information (problems?)
```

**2. Use Display Filters Progressively:**
```
Start broad, narrow down:
1. http
2. http and ip.addr == 192.168.1.1
3. http and ip.addr == 192.168.1.1 and http.response.code >= 400
```

**3. Follow Streams:**
```
For application-level analysis:
- Right-click → Follow → TCP/UDP/HTTP Stream
- See complete conversation
- Understand context
```

**4. Use Time References:**
```
Mark key events:
- Right-click → Set Time Reference
- Measure time between events
- Correlate with logs
```

**5. Apply Coloring Rules:**
```
Visual identification:
- Color errors in red
- Color important traffic
- Quick pattern recognition
```

**6. Save Work:**
```
Save display filters:
- Filter toolbar → Bookmark button
- Name filters descriptively
- Organize into categories
```

### Privacy and Security

**1. Minimize Capture Scope:**
- Only capture necessary traffic
- Use specific capture filters
- Limit to required interfaces

**2. Secure Capture Files:**
```bash
# Set restrictive permissions
chmod 600 capture.pcap

# Encrypt sensitive captures
gpg -c capture.pcap

# Secure transfer
scp capture.pcap user@secure-host:/encrypted/volume/
```

**3. Sanitize Before Sharing:**
- Remove sensitive data
- Anonymize IP addresses with editcap
- Redact passwords and credentials
- Filter to only relevant packets

**4. Data Retention:**
- Delete captures when no longer needed
- Follow organizational policies
- Don't keep captures indefinitely

**5. Access Control:**
- Limit who can capture traffic
- Audit packet capture activity
- Use separate profiles for different roles

### Performance Optimization

**1. Display Filter Performance:**
```
Fast filters:
- ip.addr == 192.168.1.1
- tcp.port == 80
- frame.number >= 100

Slow filters (avoid on large captures):
- matches (regex) operations
- complex string operations
- contains on large fields
```

**2. Large Capture Files:**
```
Strategies:
1. Split into smaller files (editcap)
2. Use command-line tshark for statistics
3. Filter and save subset
4. Increase system memory
5. Use faster storage (SSD)
```

**3. Disable Unnecessary Dissection:**
```
Analyze → Enabled Protocols
Disable protocols you don't need:
- Speeds up loading
- Reduces memory
- Faster filtering
```

**4. Limit Real-Time Updates:**
```
During capture:
☐ Update list of packets in real-time
☐ Automatically scroll during live capture

Enable only when needed for monitoring
```

### File Management

**1. Organize Captures:**
```
Directory structure:
~/captures/
  ├── 2024-01/
  │   ├── web-server/
  │   ├── database/
  │   └── network/
  ├── 2024-02/
  └── current/
```

**2. Use Consistent Naming:**
```
Format: YYYY-MM-DD_description_interface.ext
Examples:
  2024-01-15_slow-http_eth0.pcapng
  2024-01-15_dns-issues_any.pcap
```

**3. Document Investigations:**
```
Keep alongside capture:
  capture.pcapng
  capture_notes.txt
  capture_analysis.pdf
```

**4. Backup Important Captures:**
- Critical investigations
- Compliance evidence
- Security incidents
- Training examples

## Troubleshooting

### Permission Issues

**Problem:** Can't capture packets

**Linux:**
```bash
# Check groups
groups

# Add to wireshark group
sudo usermod -a -G wireshark $USER
newgrp wireshark

# Set capabilities
sudo setcap cap_net_raw,cap_net_admin+eip /usr/bin/dumpcap

# Verify
getcap /usr/bin/dumpcap
```

**macOS:**
```bash
# Check ChmodBPF
sudo launchctl list | grep chmod

# Reinstall if needed
# Run Wireshark installer's "Install ChmodBPF" package

# Check permissions
ls -la /dev/bpf*

# Add to access_bpf group
sudo dseditgroup -o edit -a $USER -t user access_bpf
```

**Windows:**
```plaintext
1. Run as Administrator
2. Reinstall Npcap
3. Check firewall settings
4. Verify Npcap service running:
   services.msc → Npcap Packet Driver (npf)
```

### No Interfaces Available

**Problem:** No interfaces shown

**Solutions:**
```bash
# Refresh interfaces
Capture → Refresh Interfaces (Ctrl+Shift+R)

# Check with dumpcap
dumpcap -D

# Check with tshark
tshark -D

# Check system interfaces
ip link show        # Linux
ifconfig           # macOS/BSD
ipconfig           # Windows

# Restart Wireshark
# Reboot system
```

### Slow Performance

**Problem:** Wireshark slow or freezing

**Solutions:**

**1. Large Capture File:**
```bash
# Split file
editcap -c 10000 large.pcap small.pcap
# Creates small_00000.pcap, small_00001.pcap, etc.

# Filter and save subset
tshark -r large.pcap -Y "http" -w http-only.pcap

# Use tshark for statistics
tshark -r large.pcap -q -z io,phs
```

**2. Disable Name Resolution:**
```
View → Name Resolution → Uncheck all
Or: Edit → Preferences → Name Resolution → Uncheck all
```

**3. Disable Protocols:**
```
Analyze → Enabled Protocols
Disable unused protocols
```

**4. Limit Real-Time Updates:**
```
Edit → Preferences → Capture
☐ Update list of packets in real-time
```

**5. Increase Memory:**
- Close other applications
- Add system RAM
- Use 64-bit Wireshark

### Display Issues

**Problem:** Packets not displayed correctly

**Solutions:**

**1. Wrong Dissector:**
```
Right-click packet → Decode As
Select correct protocol
```

**2. Missing Preferences:**
```
Edit → Preferences → Protocols → [Protocol]
Configure ports, options
```

**3. Corrupted Profile:**
```
Create new profile:
Edit → Configuration Profiles → New
```

**4. Reset Preferences:**
```
Close Wireshark

# Linux/macOS
rm -rf ~/.config/wireshark/preferences
rm -rf ~/.wireshark/preferences

# Windows
del %APPDATA%\Wireshark\preferences

Restart Wireshark
```

### Decryption Not Working

**Problem:** TLS/SSL not decrypting

**Check:**

**1. SSLKEYLOGFILE:**
```bash
# Verify environment variable set
echo $SSLKEYLOGFILE     # Linux/macOS
echo %SSLKEYLOGFILE%   # Windows

# Check file exists and has data
cat ~/sslkeys.log

# Verify configured in Wireshark
Edit → Preferences → Protocols → TLS
Check (Pre)-Master-Secret log filename
```

**2. Key Exchange:**
```
Check TLS handshake:
Filter: tls.handshake.type == 2

Look at Server Hello:
- Cipher suite used
- If DHE/ECDHE: forward secrecy (can't decrypt without keys)
- If RSA: can decrypt with private key
```

**3. Capture Complete Handshake:**
```
Must capture from beginning:
- Client Hello
- Server Hello
- Key exchange
- Finished messages

If missing: restart capture and regenerate traffic
```

**Problem:** WPA/WPA2 not decrypting

**Check:**

**1. 4-Way Handshake:**
```
Filter: eapol
Should see 4 EAPOL packets
If missing: deauth client to force reconnect
```

**2. Correct Password:**
```
Edit → Preferences → Protocols → IEEE 802.11
Decryption keys → Edit
Format: wpa-pwd
Key: password:SSID
```

**3. Key Format:**
```
Correct: MyPassword:MySSID
Wrong: MyPassword
Wrong: MySSID:MyPassword
```

### Capture Issues

**Problem:** Packets dropped during capture

**Solutions:**

**1. Increase Buffer:**
```
Capture Options → Input
Buffer size: 512 MB
```

**2. Use Dumpcap:**
```bash
# Lower overhead
dumpcap -i eth0 -w capture.pcap
```

**3. Faster Storage:**
```bash
# Write to SSD or RAM disk
dumpcap -i eth0 -w /dev/shm/capture.pcap
```

**4. Use Capture Filter:**
```
Reduce captured traffic:
tcp port 80 or tcp port 443
```

**5. Disable Display:**
```
Capture Options:
☐ Update list of packets in real-time
```

### Filter Syntax Errors

**Problem:** Display filter not working

**Common Errors:**

**1. Wrong Operator:**
```
Wrong: ip.addr = 192.168.1.1
Right: ip.addr == 192.168.1.1

Wrong: tcp.port = 80
Right: tcp.port == 80
```

**2. Missing Quotes:**
```
Wrong: http.host == www.example.com
Right: http.host == "www.example.com"

Wrong: http.request.method == POST
Right: http.request.method == "POST"
```

**3. Field Name:**
```
Wrong: http.response == 404
Right: http.response.code == 404

Wrong: ip.source == 192.168.1.1
Right: ip.src == 192.168.1.1
```

**Find Correct Field:**
```
1. Right-click field in packet details
2. Copy → Field Name
3. Paste in filter
```

**4. Boolean Logic:**
```
Wrong: ip.addr == 192.168.1.1 or 192.168.1.2
Right: ip.addr == 192.168.1.1 or ip.addr == 192.168.1.2

Wrong: tcp.port == 80 and 443
Right: tcp.port == 80 or tcp.port == 443
```

**Test Filter:**
- Type in filter toolbar
- Green background = valid
- Red background = invalid
- Yellow = unusual but valid

## Keyboard Shortcuts

### Navigation
- **Ctrl+Home** - First packet
- **Ctrl+End** - Last packet
- **Ctrl+Up** - Previous packet
- **Ctrl+Down** - Next packet
- **Ctrl+.** - Next packet in conversation
- **Ctrl+,** - Previous packet in conversation
- **Ctrl+G** - Go to packet
- **Ctrl+Left** - Go back in packet history
- **Ctrl+Right** - Go forward in packet history

### Capture
- **Ctrl+E** - Start/stop capture
- **Ctrl+R** - Restart capture
- **Ctrl+K** - Capture options

### Files
- **Ctrl+O** - Open file
- **Ctrl+S** - Save
- **Ctrl+Shift+S** - Save as
- **Ctrl+W** - Close file
- **Ctrl+Q** - Quit

### Display
- **Ctrl+M** - Mark/unmark packet
- **Ctrl+Shift+M** - Mark all displayed
- **Ctrl+Alt+M** - Unmark all
- **Ctrl+T** - Set/unset time reference
- **Ctrl+Alt+T** - Unset all time references
- **Ctrl+D** - Ignore/unignore packet

### Find
- **Ctrl+F** - Find packet
- **Ctrl+N** - Find next
- **Ctrl+B** - Find previous

### View
- **Ctrl++** - Zoom in
- **Ctrl+-** - Zoom out
- **Ctrl+0** - Normal size
- **Ctrl+Shift+Z** - Zoom to fit
- **F5** - Reload
- **F8** - Toggle packet bytes pane
- **F9** - Toggle packet list pane
- **F10** - Toggle packet details pane

### Filtering
- **Ctrl+Slash** - Apply display filter
- **Ctrl+Backslash** - Clear display filter
- **/** - Jump to filter toolbar

### Misc
- **Ctrl+I** - Capture interfaces
- **Ctrl+Shift+P** - Preferences
- **Space** - Toggle expand/collapse packet details
- **Tab** - Move between panes
- **Ctrl+C** - Copy (selected text or packet info)

## Quick Reference

### Common Display Filters

```
# IP addresses
ip.addr == 192.168.1.1
ip.src == 192.168.1.1
ip.dst == 192.168.1.1

# Ports
tcp.port == 80
udp.port == 53
tcp.srcport == 443
tcp.dstport == 8080

# Protocols
http
dns
tls or ssl
icmp
arp

# HTTP
http.request
http.response
http.request.method == "GET"
http.response.code == 404
http.host == "example.com"

# DNS
dns.qry.name contains "example"
dns.flags.rcode != 0

# TCP analysis
tcp.analysis.retransmission
tcp.analysis.duplicate_ack
tcp.analysis.zero_window
tcp.stream == 0

# Combinations
http and ip.addr == 192.168.1.1
tcp.port == 443 and tcp.flags.syn == 1
dns and not dns.flags.response
```

### Common Capture Filters

```
# Host
host 192.168.1.1
src host 192.168.1.1
dst host 192.168.1.1

# Network
net 192.168.1.0/24
src net 192.168.1.0/24

# Port
port 80
tcp port 443
udp port 53
portrange 8000-9000

# Protocol
tcp
udp
icmp
arp

# Combinations
host 192.168.1.1 and port 80
tcp and not port 22
net 192.168.1.0/24 and udp
```

### Color Meaning

- **Light Purple** - Bad TCP
- **Light Green** - HTTP
- **Light Blue** - UDP
- **Pink** - ICMP
- **Dark Gray** - TCP
- **Yellow** - Routing, SMB
- **Black Background** - Marked packets
- **Red Text** - Errors

### Statistics Locations

- **Protocol Hierarchy** - Statistics → Protocol Hierarchy
- **Conversations** - Statistics → Conversations
- **Endpoints** - Statistics → Endpoints
- **IO Graph** - Statistics → IO Graph
- **Flow Graph** - Statistics → Flow Graph
- **HTTP** - Statistics → HTTP
- **DNS** - Statistics → DNS

### Expert Info Severity

- **🔴 Errors** - Malformed, checksums, etc.
- **🟡 Warnings** - Retransmissions, dup ACKs
- **🔵 Notes** - Unusual but valid
- **⚪ Chats** - Normal workflow

## Conclusion

Wireshark is an incredibly powerful and comprehensive network analysis tool. Mastering it requires understanding both the technical aspects of network protocols and the features of the tool itself.

**Key Takeaways:**

**1. Authorization First:**
- Always get permission before capturing
- Understand legal implications
- Follow privacy requirements
- Document scope and purpose

**2. Capture Efficiently:**
- Use capture filters to reduce data
- Set appropriate snapshot lengths
- Use ring buffers for long-term monitoring
- Document captures with comments

**3. Analyze Methodically:**
- Start with statistics
- Use display filters progressively
- Follow streams for context
- Leverage expert information

**4. Master the Filters:**
- Understand capture vs display filters
- Learn the filter syntax
- Save common filters
- Use filter macros for complex expressions

**5. Use Visual Aids:**
- Apply coloring rules
- Use IO graphs
- View flow diagrams
- Check TCP stream graphs

**6. Protect Privacy:**
- Minimize capture scope
- Secure capture files
- Sanitize before sharing
- Follow data retention policies

**7. Stay Organized:**
- Use configuration profiles
- Name files descriptively
- Organize by project/date
- Document findings

**Learning Path:**

**Week 1-2: Basics**
- Install and configure
- Understand interface
- Capture and save packets
- Basic display filters
- Follow streams

**Week 3-4: Filtering**
- Master display filter syntax
- Create custom filters
- Use filter bookmarks
- Apply capture filters
- Boolean logic

**Month 2: Analysis**
- Protocol hierarchy
- Conversations and endpoints
- Expert information
- IO graphs
- TCP analysis

**Month 3: Advanced**
- Decryption (TLS, WPA)
- Custom coloring rules
- Statistics windows
- Stream graphs
- Configuration profiles

**Month 4+: Specialized**
- VoIP analysis
- Wireless troubleshooting
- Security analysis
- Performance tuning
- Automation with tshark

**Common Workflows:**

**Network Troubleshooting:**
1. Capture during problem
2. Check expert information
3. Find retransmissions
4. Analyze response times
5. Identify bottlenecks

**Security Analysis:**
1. Baseline normal traffic
2. Look for anomalies
3. Check for known patterns
4. Investigate suspicious IPs
5. Document findings

**Application Debugging:**
1. Filter to application
2. Follow relevant streams
3. Check error responses
4. Measure timing
5. Correlate with logs

**Resources:**

**Official:**
- Wireshark website: https://www.wireshark.org/
- User Guide: https://www.wireshark.org/docs/wsug_html_chunked/
- Wiki: https://wiki.wireshark.org/
- Display filter reference: https://www.wireshark.org/docs/dfref/
- Sample captures: https://wiki.wireshark.org/SampleCaptures

**Community:**
- Wireshark Q&A: https://ask.wireshark.org/
- Mailing lists: https://www.wireshark.org/lists/
- Bug tracker: https://bugs.wireshark.org/

**Training:**
- Wireshark University: https://www.wiresharktraining.com/
- Official training: https://www.wireshark.org/training/
- YouTube channel: Wireshark official channel
- Books: "Wireshark Network Analysis" by Laura Chappell

**Practice:**
- Use sample captures from wiki
- Capture your own traffic (with permission)
- Analyze different protocols
- Solve capture challenges
- CTF competitions

**Best Practices Summary:**

1. **Always authorize** your captures
2. **Use capture filters** to minimize data
3. **Start with statistics** before diving in
4. **Filter progressively** from broad to specific
5. **Follow streams** for application context
6. **Color packets** for visual identification
7. **Save work** with profiles and bookmarks
8. **Secure captures** with appropriate controls
9. **Document everything** for future reference
10. **Keep learning** - protocols and features

Wireshark is an essential tool for anyone working with networks. Whether you're troubleshooting connectivity issues, analyzing application behavior, investigating security incidents, or learning about network protocols, Wireshark provides the visibility and tools you need.

The more you use Wireshark, the more proficient you'll become at quickly identifying and resolving network issues. Practice on real traffic (with authorization), study different protocols, and gradually explore advanced features.

Remember: With great power comes great responsibility. Use Wireshark ethically, legally, and with proper authorization.

Happy analyzing!
