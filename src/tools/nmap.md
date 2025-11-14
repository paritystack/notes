# Nmap

Nmap (Network Mapper) is a free and open-source network discovery and security auditing tool. It is one of the most powerful and widely-used tools for network exploration, security scanning, and vulnerability assessment.

## Overview

Nmap was created by Gordon Lyon (Fyodor) and has been actively developed since 1997. It uses raw IP packets to determine what hosts are available on a network, what services those hosts are offering, what operating systems they are running, what type of packet filters/firewalls are in use, and dozens of other characteristics.

**Key Features:**
- Host discovery (identify devices on a network)
- Port scanning (enumerate open ports)
- Version detection (determine application name and version)
- OS detection (identify operating systems and hardware)
- Scriptable interaction with the target (NSE - Nmap Scripting Engine)
- Flexible target and port specification
- Support for IPv6
- Multiple output formats (normal, XML, grepable, script kiddie)
- Fast scanning (parallel scanning)
- Advanced techniques (idle scan, OS fingerprinting, firewall evasion)

**Common Use Cases:**
- Network inventory and asset management
- Security auditing and penetration testing
- Compliance validation
- Service upgrade monitoring
- Network troubleshooting
- Vulnerability assessment
- Identifying unauthorized devices or services

## Legal and Ethical Considerations

**IMPORTANT:** Only scan networks and systems you own or have explicit written permission to test. Unauthorized port scanning may be illegal in your jurisdiction and can be considered a precursor to hacking.

**Best Practices:**
- Always obtain written authorization before scanning
- Document the scope and limitations of your testing
- Be mindful of scan intensity on production systems
- Follow responsible disclosure practices for vulnerabilities
- Check local laws regarding network scanning
- Use appropriate timing to minimize network impact
- Inform network administrators of your activities

## Basic Concepts

### How Nmap Works

Nmap operates in several phases:

1. **Target enumeration** - Parse target specifications
2. **Host discovery** - Determine which hosts are up
3. **Reverse DNS resolution** - Look up hostnames
4. **Port scanning** - Determine port states
5. **Version detection** - Identify services and versions
6. **OS detection** - Fingerprint operating systems
7. **Traceroute** - Map network path to hosts
8. **Script scanning** - Run NSE scripts
9. **Output** - Format and display results

### Port States

Nmap classifies ports into six states:

- **open** - Application actively accepting connections
- **closed** - Port accessible (receives/responds to probes) but no application listening
- **filtered** - Cannot determine if open (packet filtering prevents probes from reaching port)
- **unfiltered** - Port accessible but cannot determine if open or closed
- **open|filtered** - Cannot determine if open or filtered
- **closed|filtered** - Cannot determine if closed or filtered

### Packet Types

Understanding packet types helps interpret scan results:

- **SYN (Synchronize)** - Initiate connection
- **ACK (Acknowledge)** - Confirm receipt
- **RST (Reset)** - Abort connection
- **FIN (Finish)** - Close connection
- **PSH (Push)** - Send data immediately
- **URG (Urgent)** - Prioritize data

## Installation

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install nmap

# RHEL/CentOS/Fedora
sudo yum install nmap
# or
sudo dnf install nmap

# macOS
brew install nmap

# Verify installation
nmap --version
```

## Target Specification

Nmap is flexible in how you specify targets.

### Single Host

```bash
# Scan single IP
nmap 192.168.1.1

# Scan hostname
nmap scanme.nmap.org

# Scan domain
nmap example.com
```

### Multiple Hosts

```bash
# Multiple IPs
nmap 192.168.1.1 192.168.1.2 192.168.1.3

# Space-separated list
nmap 192.168.1.1 192.168.1.5 192.168.1.10

# Multiple hostnames
nmap host1.example.com host2.example.com
```

### IP Ranges

```bash
# CIDR notation (most common)
nmap 192.168.1.0/24        # Entire /24 subnet (256 addresses)
nmap 192.168.1.0/25        # Half subnet (128 addresses)
nmap 10.0.0.0/8            # Entire class A network

# Hyphen range
nmap 192.168.1.1-20        # Scan .1 through .20
nmap 192.168.1-3.1         # Scan 192.168.1.1, 192.168.2.1, 192.168.3.1

# Wildcard (not CIDR, but convenient)
nmap 192.168.1.*           # Entire /24 subnet
nmap 192.168.*.1           # .1 address of all /16 subnets

# Octet ranges
nmap 192.168.1.1-254       # Skip .0 and .255
nmap 192.168.1,2,3.1       # Multiple specific octets
```

### Excluding Targets

```bash
# Exclude single host
nmap 192.168.1.0/24 --exclude 192.168.1.1

# Exclude multiple hosts
nmap 192.168.1.0/24 --exclude 192.168.1.1,192.168.1.5

# Exclude range
nmap 192.168.1.0/24 --exclude 192.168.1.1-10

# Exclude from file
nmap 192.168.1.0/24 --excludefile exclude.txt
```

### Input from File

```bash
# Read targets from file (one per line)
nmap -iL targets.txt

# Example targets.txt:
# 192.168.1.1
# 192.168.1.0/24
# example.com
# 10.0.0.1-50
```

### Random Targets

```bash
# Scan random IPs
nmap -iR 100               # Scan 100 random IPs
nmap -iR 0                 # Scan random IPs forever (Ctrl+C to stop)

# Exclude private ranges when using random
nmap -iR 100 --exclude 192.168.0.0/16,10.0.0.0/8,172.16.0.0/12
```

## Host Discovery (Ping Scanning)

Before port scanning, Nmap determines which hosts are online. This is called "host discovery" or "ping scanning."

### Default Discovery

```bash
# Default scan (ping scan + port scan)
nmap 192.168.1.0/24

# By default, Nmap sends:
# - ICMP echo request
# - TCP SYN to port 443
# - TCP ACK to port 80
# - ICMP timestamp request
```

### List Scan (No Discovery)

```bash
# Just list targets, don't scan
nmap -sL 192.168.1.0/24

# Useful for:
# - Verifying target list
# - Performing reverse DNS lookups
# - Understanding scan scope
```

### Ping Scan Only

```bash
# Only determine which hosts are up (no port scan)
nmap -sn 192.168.1.0/24
# Formerly known as -sP (deprecated)

# Fast way to:
# - Find live hosts
# - Create inventory
# - Map network
```

### Skip Host Discovery

```bash
# Treat all hosts as online (skip ping)
nmap -Pn 192.168.1.1

# Useful when:
# - Firewall blocks pings
# - You know host is up
# - Scanning single host
# - Behind aggressive firewall
```

### TCP SYN Ping

```bash
# TCP SYN ping to specific port
nmap -PS 192.168.1.1       # Default ports: 80,443
nmap -PS22 192.168.1.1     # Port 22
nmap -PS22,80,443 192.168.1.1  # Multiple ports
nmap -PS1-1000 192.168.1.1     # Port range
```

### TCP ACK Ping

```bash
# TCP ACK ping (useful for firewalls that block SYN)
nmap -PA 192.168.1.1       # Default ports: 80,443
nmap -PA22 192.168.1.1     # Port 22
nmap -PA80,443 192.168.1.1 # Multiple ports
```

### UDP Ping

```bash
# UDP ping to specific port
nmap -PU 192.168.1.1       # Default port: 40125
nmap -PU53 192.168.1.1     # Port 53 (DNS)
nmap -PU161 192.168.1.1    # Port 161 (SNMP)

# Useful for UDP-only devices
```

### ICMP Ping Types

```bash
# ICMP echo request (standard ping)
nmap -PE 192.168.1.1

# ICMP timestamp request
nmap -PP 192.168.1.1

# ICMP address mask request
nmap -PM 192.168.1.1

# Combine multiple ICMP types
nmap -PE -PP -PM 192.168.1.1
```

### ARP Ping

```bash
# ARP discovery (automatic on local network)
nmap -PR 192.168.1.0/24

# Most reliable on local Ethernet
# Bypasses IP-level filtering
# Automatic when scanning local subnet
```

### Disable DNS Resolution

```bash
# Skip DNS resolution (faster)
nmap -n 192.168.1.0/24

# Force DNS resolution (even for unresponsive hosts)
nmap -R 192.168.1.0/24

# Custom DNS servers
nmap --dns-servers 8.8.8.8,8.8.4.4 192.168.1.1
```

### Combined Discovery

```bash
# Multiple discovery methods for reliability
nmap -PE -PS22,80,443 -PA80,443 -PU53 192.168.1.0/24

# Aggressive discovery (uses multiple techniques)
nmap -A 192.168.1.1  # Includes host discovery, OS detection, version detection, traceroute
```

## Port Scanning Techniques

The core functionality of Nmap is port scanning. Different scan types have different strengths and weaknesses.

### TCP SYN Scan (Stealth Scan)

```bash
# Default scan type (requires root/admin)
nmap -sS 192.168.1.1
# Or simply:
sudo nmap 192.168.1.1

# How it works:
# 1. Send SYN packet
# 2. If SYN/ACK received -> port open
# 3. If RST received -> port closed
# 4. If no response -> port filtered
# 5. Send RST (don't complete handshake)

# Advantages:
# - Fast and efficient
# - Stealthy (doesn't complete TCP handshake)
# - Accurate results
# - Works against most targets

# Disadvantages:
# - Requires root/admin privileges
# - Still logged by many IDS/IPS systems
```

### TCP Connect Scan

```bash
# Full TCP connection (no root required)
nmap -sT 192.168.1.1

# How it works:
# 1. Complete full TCP 3-way handshake
# 2. If connection succeeds -> port open
# 3. If RST received -> port closed
# 4. If no response/timeout -> filtered

# Advantages:
# - No special privileges required
# - Works through certain firewalls
# - Reliable results

# Disadvantages:
# - Slower than SYN scan
# - More easily detected (logged)
# - Uses more network resources
```

### UDP Scan

```bash
# UDP port scan
nmap -sU 192.168.1.1

# Common UDP ports
nmap -sU -p 53,67,68,69,123,161,162,137,138,139 192.168.1.1

# Combined TCP and UDP scan
nmap -sS -sU -p U:53,161,T:21-25,80,443 192.168.1.1

# How it works:
# 1. Send UDP packet to port
# 2. If UDP response received -> port open
# 3. If ICMP port unreachable -> port closed
# 4. If no response -> port open|filtered

# Important notes:
# - UDP scans are SLOW (be patient)
# - Many UDP services don't respond to empty packets
# - Version detection (-sV) helps identify UDP services
# - Use --version-intensity for better UDP detection

# Speed up UDP scans:
nmap -sU --top-ports 20 192.168.1.1     # Scan only top 20 UDP ports
nmap -sU --host-timeout 30s 192.168.1.1 # Set timeout
nmap -sU -T4 192.168.1.1                # Aggressive timing
```

### TCP ACK Scan

```bash
# ACK scan (firewall rule mapping)
nmap -sA 192.168.1.1

# How it works:
# 1. Send ACK packet
# 2. If RST received -> port unfiltered
# 3. If no response -> port filtered

# Purpose:
# - Map firewall rulesets
# - Determine if firewall is stateful
# - Identify filtered ports
# - Does NOT determine open/closed
```

### TCP Window Scan

```bash
# Window scan (like ACK but checks TCP window)
nmap -sW 192.168.1.1

# How it works:
# - Similar to ACK scan
# - Examines TCP window field in RST packets
# - Some systems report positive window for open ports

# Less reliable than other scans
# System-dependent behavior
```

### TCP Maimon Scan

```bash
# Maimon scan (FIN/ACK probe)
nmap -sM 192.168.1.1

# How it works:
# - Sends FIN/ACK packet
# - Open and closed ports should respond with RST
# - Some systems drop packets for open ports

# Rarely useful in modern networks
# Named after Uriel Maimon
```

### TCP NULL, FIN, and Xmas Scans

```bash
# NULL scan (no flags set)
nmap -sN 192.168.1.1

# FIN scan (only FIN flag)
nmap -sF 192.168.1.1

# Xmas scan (FIN, PSH, URG flags - "lit up like Christmas tree")
nmap -sX 192.168.1.1

# How they work:
# - If RST received -> port closed
# - If no response -> port open|filtered
# - If ICMP unreachable -> port filtered

# Based on RFC 793 behavior:
# - Closed ports should respond with RST
# - Open ports should drop the packet

# Advantages:
# - Can bypass some non-stateful firewalls
# - May evade simple IDS

# Disadvantages:
# - Don't work against Windows (RFC non-compliant)
# - Unreliable results
# - Many modern systems don't follow RFC exactly
# - Not useful for most modern networks
```

### Custom TCP Scan

```bash
# Set custom TCP flags
nmap --scanflags URGACKPSHRSTSYNFIN 192.168.1.1

# Common flag combinations:
nmap --scanflags SYN 192.168.1.1        # Equivalent to -sS
nmap --scanflags ACK 192.168.1.1        # Equivalent to -sA
nmap --scanflags FIN 192.168.1.1        # Equivalent to -sF

# Combine with scan type:
nmap -sF --scanflags FIN,PSH,URG 192.168.1.1
```

### Idle/Zombie Scan

```bash
# Idle scan using zombie host
nmap -sI zombie.example.com target.example.com
nmap -sI 192.168.1.50 192.168.1.1

# How it works:
# 1. Find idle host with predictable IP ID sequence
# 2. Use zombie to scan target
# 3. Your IP never contacts target
# 4. Extremely stealthy

# Requirements:
# - Find suitable zombie host
# - Zombie must be truly idle
# - Zombie must have predictable IP ID

# Finding zombie candidates:
nmap --script ipidseq 192.168.1.0/24

# Advantages:
# - Ultimate stealth (your IP hidden)
# - Bypass IP-based filters

# Disadvantages:
# - Difficult to find good zombie
# - Slow
# - Requires specific conditions
# - Complex to set up
```

### IP Protocol Scan

```bash
# Scan for supported IP protocols
nmap -sO 192.168.1.1

# Determines which IP protocols are supported:
# - ICMP (1)
# - IGMP (2)
# - TCP (6)
# - UDP (17)
# - etc.

# Useful for:
# - Identifying protocol support
# - Firewall testing
# - Security auditing
```

### FTP Bounce Scan

```bash
# FTP bounce attack scan
nmap -b username:password@ftp.server.com target.example.com

# How it works:
# - Uses FTP server as proxy
# - Exploits FTP PORT command
# - Scans appear to come from FTP server

# Note:
# - Most FTP servers have patched this
# - Rarely works on modern systems
# - Mostly of historical interest
```

### SCTP INIT Scan

```bash
# SCTP INIT scan
nmap -sY 192.168.1.1

# SCTP equivalent of TCP SYN scan
# Used for SCTP protocol (Stream Control Transmission Protocol)
# Common in telecom/VoIP systems
```

### SCTP COOKIE ECHO Scan

```bash
# SCTP COOKIE ECHO scan
nmap -sZ 192.168.1.1

# More stealthy than INIT scan
# May bypass some firewalls
```

## Port Specification

Control which ports to scan.

### Default Ports

```bash
# Default: scan 1000 most common ports
nmap 192.168.1.1

# View which ports are scanned by default:
nmap --top-ports 10 -v 192.168.1.1
```

### Specific Ports

```bash
# Single port
nmap -p 22 192.168.1.1

# Multiple ports (comma-separated)
nmap -p 22,80,443 192.168.1.1

# Port range
nmap -p 1-100 192.168.1.1
nmap -p 20-25,80,443,8000-8100 192.168.1.1

# All ports (1-65535)
nmap -p- 192.168.1.1
nmap -p 1-65535 192.168.1.1

# Named ports
nmap -p http,https,ssh 192.168.1.1
```

### Port Range Shortcuts

```bash
# Ports from 1 to 1024
nmap -p -1024 192.168.1.1

# Ports from 1024 to 65535
nmap -p 1024- 192.168.1.1

# Port 80 and above
nmap -p 80- 192.168.1.1
```

### Protocol-Specific Ports

```bash
# TCP ports only
nmap -p T:80,443 192.168.1.1

# UDP ports only
nmap -p U:53,161 192.168.1.1

# Mixed TCP and UDP
nmap -p U:53,161,T:21-25,80 192.168.1.1

# All TCP ports
nmap -p T:- 192.168.1.1

# All UDP ports
nmap -sU -p U:- 192.168.1.1
```

### Top Ports

```bash
# Scan top N most common ports
nmap --top-ports 10 192.168.1.1    # Top 10
nmap --top-ports 100 192.168.1.1   # Top 100
nmap --top-ports 1000 192.168.1.1  # Top 1000

# Based on nmap-services frequency data
# Fast way to scan most likely ports
```

### Port Ratio

```bash
# Scan ports with ratio above threshold
nmap --port-ratio 0.1 192.168.1.1

# Ratio range: 0.0 to 1.0
# 0.1 = top 10% most common ports
# Higher ratio = fewer ports scanned
```

### Fast Scan

```bash
# Fast mode: scan fewer ports than default
nmap -F 192.168.1.1

# Scans only 100 most common ports
# Much faster than default 1000 ports
# Good for quick reconnaissance
```

### Exclude Ports

```bash
# Scan all ports except specified
nmap -p- --exclude-ports 22,80,443 192.168.1.1

# Exclude port range
nmap -p 1-1000 --exclude-ports 100-200 192.168.1.1
```

### Sequential Port Scanning

```bash
# Scan ports in order (not randomized)
nmap -r 192.168.1.1

# By default, nmap randomizes port order
# -r scans in numerical order
# Useful for troubleshooting
```

## Service and Version Detection

Identify services and their versions running on open ports.

### Version Detection

```bash
# Enable version detection
nmap -sV 192.168.1.1

# How it works:
# 1. Connect to open ports
# 2. Send probes
# 3. Analyze responses
# 4. Match against signature database
# 5. Report service name and version

# Example output:
# PORT    STATE SERVICE VERSION
# 22/tcp  open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.5
# 80/tcp  open  http    Apache httpd 2.4.41
# 443/tcp open  https   nginx 1.18.0
```

### Version Intensity

```bash
# Default intensity (7)
nmap -sV 192.168.1.1

# Light version detection (2) - faster, less comprehensive
nmap -sV --version-intensity 2 192.168.1.1
nmap -sV --version-light 192.168.1.1

# All probes (9) - slower, most comprehensive
nmap -sV --version-intensity 9 192.168.1.1
nmap -sV --version-all 192.168.1.1

# Custom intensity (0-9)
nmap -sV --version-intensity 5 192.168.1.1

# Intensity levels:
# 0 - Fastest, least accurate
# 2 - Light (--version-light)
# 7 - Default
# 9 - All probes (--version-all)
```

### Version Scan Trace

```bash
# Debug version detection
nmap -sV --version-trace 192.168.1.1

# Shows:
# - Probes sent
# - Responses received
# - Matching process
# - Useful for troubleshooting
```

### RPC Information

```bash
# Get RPC info
nmap -sR 192.168.1.1

# Determines RPC program and version
# Used with -sV for RPC services
```

## OS Detection

Identify operating system and hardware characteristics.

### Basic OS Detection

```bash
# Enable OS detection
nmap -O 192.168.1.1

# Requires at least one open and one closed port
# Uses TCP/IP stack fingerprinting
# Compares responses to signature database

# Example output:
# Running: Linux 4.X|5.X
# OS CPE: cpe:/o:linux:linux_kernel:4 cpe:/o:linux:linux_kernel:5
# OS details: Linux 4.15 - 5.6
```

### Aggressive OS Detection

```bash
# More aggressive OS detection
nmap -O --osscan-guess 192.168.1.1
nmap -O --fuzzy 192.168.1.1

# Makes best guess even with less confidence
# Useful when standard detection inconclusive
```

### OS Scan Limits

```bash
# Only scan hosts with at least one open and one closed port
nmap -O --osscan-limit 192.168.1.0/24

# Skip OS detection if requirements not met
# Speeds up large scans
```

### Maximum Retries

```bash
# Set max OS detection retries
nmap -O --max-os-tries 2 192.168.1.1

# Default: 5
# Lower = faster but less accurate
# Higher = more accurate but slower
```

## Aggressive Scanning

Combine multiple detection methods.

### Aggressive Scan

```bash
# Enable OS detection, version detection, script scanning, and traceroute
nmap -A 192.168.1.1

# Equivalent to:
nmap -O -sV -sC --traceroute 192.168.1.1

# Provides comprehensive information
# Slower and more intrusive
# Good for detailed single-host scans
```

### Traceroute

```bash
# Enable traceroute
nmap --traceroute 192.168.1.1

# Shows network path to host
# Useful for understanding routing
# Combined with topology mapping

# Example output:
# TRACEROUTE (using port 80/tcp)
# HOP RTT      ADDRESS
# 1   1.00 ms  192.168.1.254
# 2   5.00 ms  10.0.0.1
# 3   15.00 ms example.com (93.184.216.34)
```

## Timing and Performance

Control scan speed and resource usage.

### Timing Templates

```bash
# T0 - Paranoid (IDS evasion)
nmap -T0 192.168.1.1
# - One port at a time
# - 5 minutes between probes
# - Extremely slow
# - Maximum stealth

# T1 - Sneaky (IDS evasion)
nmap -T1 192.168.1.1
# - Serial scanning
# - 15 seconds between probes
# - Very slow
# - Reduced chance of detection

# T2 - Polite (slow scan, less bandwidth)
nmap -T2 192.168.1.1
# - Throttled to use less bandwidth
# - 0.4 seconds between probes
# - Slower than default
# - Reduced network load

# T3 - Normal (default)
nmap -T3 192.168.1.1
nmap 192.168.1.1
# - Default timing
# - Balanced speed and accuracy
# - Suitable for most networks

# T4 - Aggressive (fast networks)
nmap -T4 192.168.1.1
# - Assumes fast and reliable network
# - 10-minute timeout per host
# - Fast scanning
# - Recommended for modern networks

# T5 - Insane (very fast networks)
nmap -T5 192.168.1.1
# - Assumes extraordinarily fast network
# - 5-minute timeout per host
# - Extremely fast
# - May miss hosts/ports
# - Sacrifice accuracy for speed
```

### Fine-Grained Timing Control

```bash
# Minimum packets per second
nmap --min-rate 100 192.168.1.1

# Maximum packets per second
nmap --max-rate 1000 192.168.1.1

# Combine for precise control
nmap --min-rate 50 --max-rate 500 192.168.1.1

# Host timeout
nmap --host-timeout 30m 192.168.1.1    # 30 minutes
nmap --host-timeout 10s 192.168.1.1    # 10 seconds

# Scan delay (pause between probes)
nmap --scan-delay 1s 192.168.1.1       # 1 second delay
nmap --max-scan-delay 2s 192.168.1.1   # Maximum 2 seconds

# Initial RTT timeout
nmap --initial-rtt-timeout 100ms 192.168.1.1

# Minimum RTT timeout
nmap --min-rtt-timeout 50ms 192.168.1.1

# Maximum RTT timeout
nmap --max-rtt-timeout 500ms 192.168.1.1
```

### Parallelism

```bash
# Minimum parallel operations
nmap --min-parallelism 10 192.168.1.0/24

# Maximum parallel operations
nmap --max-parallelism 100 192.168.1.0/24

# Disable parallel operations (serial)
nmap --max-parallelism 1 192.168.1.0/24

# Host group sizes
nmap --min-hostgroup 50 192.168.1.0/24
nmap --max-hostgroup 100 192.168.1.0/24
```

### Maximum Retries

```bash
# Set maximum retries for port scanning
nmap --max-retries 2 192.168.1.1

# Default: 10
# Lower = faster but may miss ports
# Higher = more thorough but slower
```

### Timing Examples

```bash
# Fast scan of web servers
nmap -T4 -F -p 80,443,8080,8443 192.168.1.0/24

# Slow stealth scan
nmap -T1 -sS -p- 192.168.1.1

# Very fast scan sacrificing accuracy
nmap -T5 --max-retries 1 --max-scan-delay 10ms 192.168.1.0/24

# Rate-limited scan (100 packets/sec)
nmap --max-rate 100 192.168.1.0/24

# Patient comprehensive scan
nmap -T2 -p- -sV -O --version-all 192.168.1.1
```

## Firewall/IDS Evasion and Spoofing

Techniques to bypass firewalls and avoid detection.

**Warning:** These techniques may be detected by modern security systems. Use only on networks you're authorized to test.

### Packet Fragmentation

```bash
# Fragment packets
nmap -f 192.168.1.1

# Use 8-byte fragments (or smaller)
nmap -f -f 192.168.1.1

# Set custom MTU (must be multiple of 8)
nmap --mtu 16 192.168.1.1
nmap --mtu 24 192.168.1.1
nmap --mtu 32 192.168.1.1

# How it works:
# - Splits packets into fragments
# - May bypass simple packet filters
# - Some IDS can't handle fragments
# - Modern systems often reassemble correctly
```

### Decoy Scanning

```bash
# Use decoy IP addresses
nmap -D RND:10 192.168.1.1        # 10 random decoys
nmap -D decoy1,decoy2,decoy3 192.168.1.1

# Include your real IP in specific position
nmap -D decoy1,ME,decoy2 192.168.1.1

# Random decoys (specify count)
nmap -D RND:5 192.168.1.1

# How it works:
# - Nmap spoofs packets from decoy IPs
# - Target sees scans from multiple sources
# - Harder to identify real scanner
# - Your IP is still in the mix

# Best practices:
# - Use live IPs as decoys
# - Don't use too many (performance)
# - Combine with other evasion techniques
```

### Idle/Zombie Host

```bash
# Use zombie host for scanning
nmap -sI zombie.example.com 192.168.1.1

# Your IP never contacts target
# Target sees scans from zombie
# Ultimate stealth technique
```

### Source IP Spoofing

```bash
# Spoof source IP address
nmap -S 192.168.1.50 192.168.1.1

# Important notes:
# - Response goes to spoofed IP, not you
# - Only useful for specific scenarios
# - Requires raw packet privileges
# - May not work through most networks
# - Often blocked by ISPs

# Specify network interface
nmap -S 192.168.1.50 -e eth0 192.168.1.1
```

### Source Port Manipulation

```bash
# Use specific source port
nmap --source-port 53 192.168.1.1
nmap -g 53 192.168.1.1

# Common privileged ports:
nmap --source-port 20 192.168.1.1  # FTP data
nmap --source-port 53 192.168.1.1  # DNS
nmap --source-port 67 192.168.1.1  # DHCP

# How it works:
# - Some firewalls allow traffic from specific ports
# - DNS (53) and FTP (20) commonly allowed
# - May bypass simple firewall rules
# - Less effective on modern stateful firewalls
```

### Append Random Data

```bash
# Append random data to packets
nmap --data-length 25 192.168.1.1

# Pads packets with random data
# Changes packet size
# May evade signature-based detection
# Size in bytes (0-65535)
```

### IP Options

```bash
# Set IP options
nmap --ip-options "L 192.168.1.5 192.168.1.10" 192.168.1.1

# Loose source routing (L)
nmap --ip-options "L" 192.168.1.1

# Strict source routing (S)
nmap --ip-options "S" 192.168.1.1

# Record route (R)
nmap --ip-options "R" 192.168.1.1

# Timestamp (T)
nmap --ip-options "T" 192.168.1.1

# Rarely useful on modern networks
# Most routers ignore or strip IP options
```

### Invalid Checksums

```bash
# Send packets with bogus checksums
nmap --badsum 192.168.1.1

# How it works:
# - Real systems will drop invalid packets
# - Firewalls/IDS might not check checksums
# - If you get responses, firewall isn't checking

# Use case:
# - Firewall/IDS detection
# - Should not get responses from real hosts
```

### Randomize Targets

```bash
# Randomize target order
nmap --randomize-hosts 192.168.1.0/24

# Prevents detection patterns
# Target hosts scanned in random order
# Harder to correlate as single scan
```

### MAC Address Spoofing

```bash
# Spoof MAC address (requires raw packets)
nmap --spoof-mac 0 192.168.1.1          # Random MAC
nmap --spoof-mac Apple 192.168.1.1      # Apple vendor
nmap --spoof-mac Dell 192.168.1.1       # Dell vendor
nmap --spoof-mac 00:11:22:33:44:55 192.168.1.1  # Specific MAC

# Vendors: Cisco, Apple, Dell, HP, etc.
# Only works on same network segment
# Useful for MAC-based filtering
```

### Combined Evasion

```bash
# Multiple evasion techniques
nmap -f -T2 -D RND:10 --source-port 53 --data-length 25 192.168.1.1

# Fragment + slow timing + decoys + source port + random data
# Maximum evasion attempt
# Very slow but stealthy

# IDS evasion scan
nmap -T1 -f --mtu 16 -D RND:5 --randomize-hosts 192.168.1.0/24
```

## NSE (Nmap Scripting Engine)

The Nmap Scripting Engine (NSE) is one of Nmap's most powerful features, allowing for vulnerability detection, exploitation, advanced discovery, and more.

### Script Categories

NSE scripts are organized into categories:

- **auth** - Authentication and credentials
- **broadcast** - Broadcast discovery
- **brute** - Brute force attacks
- **default** - Default safe scripts (run with -sC)
- **discovery** - Network and service discovery
- **dos** - Denial of service (use carefully!)
- **exploit** - Active exploitation (dangerous!)
- **external** - External resources (whois, GeoIP, etc.)
- **fuzzer** - Fuzzing tests
- **intrusive** - Intrusive scripts (may crash services)
- **malware** - Malware detection
- **safe** - Safe scripts (unlikely to crash or alert)
- **version** - Enhanced version detection
- **vuln** - Vulnerability detection

### Running Scripts

```bash
# Run default scripts
nmap -sC 192.168.1.1
nmap --script=default 192.168.1.1

# Run specific script
nmap --script=http-title 192.168.1.1

# Run multiple scripts (comma-separated)
nmap --script=http-title,http-headers 192.168.1.1

# Run script category
nmap --script=vuln 192.168.1.1
nmap --script=safe 192.168.1.1
nmap --script=discovery 192.168.1.1

# Run multiple categories
nmap --script=vuln,exploit 192.168.1.1

# Boolean expressions
nmap --script "default or safe" 192.168.1.1
nmap --script "default and safe" 192.168.1.1
nmap --script "not intrusive" 192.168.1.1
nmap --script "(default or safe or intrusive) and not http-*" 192.168.1.1

# Wildcard patterns
nmap --script "http-*" 192.168.1.1
nmap --script "ssh-*" 192.168.1.1
nmap --script "smb-*" 192.168.1.1

# All scripts (not recommended - very slow)
nmap --script=all 192.168.1.1
```

### Script Arguments

```bash
# Pass arguments to scripts
nmap --script=http-title --script-args http.useragent="Mozilla/5.0" 192.168.1.1

# Multiple arguments
nmap --script=mysql-brute --script-args userdb=users.txt,passdb=passwords.txt 192.168.1.1

# Arguments from file
nmap --script=http-form-brute --script-args-file args.txt 192.168.1.1
```

### Script Help

```bash
# View script documentation
nmap --script-help http-title
nmap --script-help "http-*"
nmap --script-help all

# Update script database
nmap --script-updatedb
```

### Common Useful Scripts

#### HTTP Scripts

```bash
# HTTP title
nmap --script=http-title 192.168.1.1

# HTTP headers
nmap --script=http-headers 192.168.1.1

# HTTP methods
nmap --script=http-methods 192.168.1.1

# Find robots.txt
nmap --script=http-robots.txt 192.168.1.1

# Enumerate directories
nmap --script=http-enum 192.168.1.1

# Find backup files
nmap --script=http-backup-finder 192.168.1.1

# WordPress vulnerabilities
nmap --script=http-wordpress-enum 192.168.1.1

# SQL injection detection
nmap --script=http-sql-injection 192.168.1.1

# Cross-site scripting
nmap --script=http-xssed 192.168.1.1

# Web application firewall detection
nmap --script=http-waf-detect 192.168.1.1

# SSL/TLS information
nmap --script=ssl-cert,ssl-enum-ciphers -p 443 192.168.1.1

# Heartbleed vulnerability
nmap --script=ssl-heartbleed -p 443 192.168.1.1
```

#### SMB Scripts

```bash
# SMB OS discovery
nmap --script=smb-os-discovery 192.168.1.1

# SMB vulnerabilities
nmap --script=smb-vuln-* 192.168.1.1

# MS17-010 (EternalBlue)
nmap --script=smb-vuln-ms17-010 192.168.1.1

# Enumerate shares
nmap --script=smb-enum-shares 192.168.1.1

# Enumerate users
nmap --script=smb-enum-users 192.168.1.1

# SMB security mode
nmap --script=smb-security-mode 192.168.1.1

# SMB protocols
nmap --script=smb-protocols 192.168.1.1
```

#### SSH Scripts

```bash
# SSH host key
nmap --script=ssh-hostkey -p 22 192.168.1.1

# SSH authentication methods
nmap --script=ssh-auth-methods -p 22 192.168.1.1

# SSH2 protocol
nmap --script=ssh2-enum-algos -p 22 192.168.1.1

# SSH brute force (use carefully!)
nmap --script=ssh-brute -p 22 192.168.1.1
```

#### DNS Scripts

```bash
# DNS brute force subdomains
nmap --script=dns-brute example.com

# DNS zone transfer
nmap --script=dns-zone-transfer --script-args dns-zone-transfer.domain=example.com -p 53 192.168.1.1

# DNS recursion
nmap --script=dns-recursion -p 53 192.168.1.1

# DNS service discovery
nmap --script=dns-service-discovery -p 53 192.168.1.1
```

#### FTP Scripts

```bash
# FTP anonymous login
nmap --script=ftp-anon -p 21 192.168.1.1

# FTP bounce
nmap --script=ftp-bounce -p 21 192.168.1.1

# FTP vulnerabilities
nmap --script=ftp-vuln-* -p 21 192.168.1.1

# FTP brute force
nmap --script=ftp-brute -p 21 192.168.1.1
```

#### MySQL Scripts

```bash
# MySQL information
nmap --script=mysql-info -p 3306 192.168.1.1

# MySQL empty password
nmap --script=mysql-empty-password -p 3306 192.168.1.1

# MySQL users
nmap --script=mysql-users -p 3306 192.168.1.1

# MySQL databases
nmap --script=mysql-databases -p 3306 192.168.1.1

# MySQL brute force
nmap --script=mysql-brute -p 3306 192.168.1.1
```

#### MongoDB Scripts

```bash
# MongoDB info
nmap --script=mongodb-info -p 27017 192.168.1.1

# MongoDB databases
nmap --script=mongodb-databases -p 27017 192.168.1.1

# MongoDB brute force
nmap --script=mongodb-brute -p 27017 192.168.1.1
```

#### Vulnerability Detection

```bash
# All vulnerability scripts
nmap --script=vuln 192.168.1.1

# Specific vulnerability checks
nmap --script=vuln -p 80,443 192.168.1.1

# Common vulnerabilities
nmap --script=vulners 192.168.1.1  # Check against Vulners database

# Vulscan (requires installation)
nmap --script=vulscan 192.168.1.1
```

#### Malware Detection

```bash
# Check for backdoors
nmap --script=backdoor-check 192.168.1.1

# Check for malware
nmap --script=malware 192.168.1.1
```

#### Broadcast Scripts

```bash
# Discover DHCP servers
nmap --script=broadcast-dhcp-discover

# Discover DNS servers
nmap --script=broadcast-dns-service-discovery

# Discover NetBIOS
nmap --script=broadcast-netbios-master-browser

# Discover ping
nmap --script=broadcast-ping

# Multiple broadcast scripts
nmap --script=broadcast
```

### Script Output

```bash
# Verbose script output
nmap --script=http-title -v 192.168.1.1

# Debug script execution
nmap --script=http-title -d 192.168.1.1

# Script trace
nmap --script=http-title --script-trace 192.168.1.1

# Shows:
# - Script execution details
# - Network communication
# - Useful for debugging scripts
```

### Custom Scripts

NSE scripts are located in `/usr/share/nmap/scripts/` or similar.

```bash
# List all scripts
ls /usr/share/nmap/scripts/

# View script contents
cat /usr/share/nmap/scripts/http-title.nse

# Create custom script (basic example)
# Save as my-custom-script.nse

-- Script metadata
description = [[
Custom script description
]]

author = "Your Name"
license = "Same as Nmap"
categories = {"safe", "discovery"}

-- Dependencies
local shortport = require "shortport"
local http = require "http"

-- Port rule
portrule = shortport.http

-- Script action
action = function(host, port)
  local response = http.get(host, port, "/")
  return response.status
end
```

## Output Options

Control how Nmap displays and saves results.

### Normal Output

```bash
# Default output (to screen)
nmap 192.168.1.1

# Save normal output to file
nmap -oN scan.txt 192.168.1.1
nmap -oN results/scan.txt 192.168.1.1

# Human-readable format
# Similar to screen output
```

### XML Output

```bash
# Save as XML
nmap -oX scan.xml 192.168.1.1

# Machine-parseable format
# Best for processing with other tools
# Used by many Nmap GUIs

# Parse XML with xmllint
xmllint --format scan.xml

# Convert to HTML
xsltproc scan.xml -o scan.html
```

### Grepable Output

```bash
# Save grepable output
nmap -oG scan.gnmap 192.168.1.1

# Easy to parse with grep, awk, sed
# One line per host
# Useful for scripting

# Example parsing:
grep "open" scan.gnmap
grep "80/open" scan.gnmap | awk '{print $2}'
```

### Script Kiddie Output

```bash
# Leet speak output (for fun)
nmap -oS scan.txt 192.168.1.1

# Example: "Port" becomes "P0rt"
# Not useful for serious work
# Entertainment value only
```

### All Formats

```bash
# Save in all formats at once
nmap -oA scan 192.168.1.1

# Creates three files:
# - scan.nmap (normal)
# - scan.xml (XML)
# - scan.gnmap (grepable)

# Recommended for important scans
```

### Append to File

```bash
# Append to existing file (don't overwrite)
nmap -oN scan.txt --append-output 192.168.1.1

# Useful for:
# - Incremental scanning
# - Combining multiple scan results
# - Continuous monitoring
```

### Verbosity

```bash
# Verbose output (more details)
nmap -v 192.168.1.1

# Very verbose (even more details)
nmap -vv 192.168.1.1

# Shows progress and additional information:
# - Open ports discovered as found
# - Scan statistics
# - Timing information
# - Estimated completion time

# Recommended for:
# - Long-running scans
# - Troubleshooting
# - Understanding scan progress
```

### Debugging

```bash
# Debug output
nmap -d 192.168.1.1

# More debug output
nmap -dd 192.168.1.1

# Maximum debug (overwhelming detail)
nmap -ddd 192.168.1.1

# Shows:
# - Packet details
# - Timing calculations
# - Internal decisions
# - Useful for troubleshooting problems
```

### Packet Trace

```bash
# Show all packets sent and received
nmap --packet-trace 192.168.1.1

# Example output:
# SENT (0.0010s) TCP 192.168.1.100:54321 > 192.168.1.1:80 S
# RCVD (0.0015s) TCP 192.168.1.1:80 > 192.168.1.100:54321 SA

# Useful for:
# - Understanding scan behavior
# - Troubleshooting firewall issues
# - Learning network protocols
```

### Open Port Output

```bash
# Only show open ports
nmap --open 192.168.1.1

# Filters output to open ports only
# Cleaner results for large scans
# Recommended for most scans
```

### Reason Output

```bash
# Show reason for port state
nmap --reason 192.168.1.1

# Example:
# 22/tcp  open     ssh        syn-ack ttl 64
# 80/tcp  closed   http       reset ttl 64
# 443/tcp filtered https      no-response

# Shows why Nmap determined each state
# Useful for understanding results
```

### Statistics

```bash
# Show periodic timing statistics
nmap --stats-every 10s 192.168.1.0/24

# Displays progress every 10 seconds
# Shows:
# - Time elapsed
# - Percent complete
# - Estimated completion time

# Interactive statistics:
# Press 'v' during scan for verbose mode
# Press 'd' during scan for debug mode
# Press 'p' during scan to pause
# Press '?' for help
```

### Resume Scans

```bash
# Save scan state periodically
nmap -oA scan --stats-every 5m 192.168.1.0/16

# If scan interrupted, resume with:
nmap --resume scan.nmap

# Continues from where it left off
# Useful for:
# - Long-running scans
# - Unstable connections
# - Interrupted scans
```

### Iflist

```bash
# Show network interfaces and routes
nmap --iflist

# Displays:
# - Network interfaces
# - IP addresses
# - Routing table
# - Useful for understanding scan source
```

## Common Patterns and Use Cases

Practical examples for common scanning scenarios.

### Quick Network Discovery

```bash
# Fast ping sweep
nmap -sn 192.168.1.0/24

# Quick port scan with version detection
nmap -T4 -F 192.168.1.0/24

# Find all web servers
nmap -p 80,443,8080,8443 --open 192.168.1.0/24

# Quick scan with service detection
nmap -T4 -A -F 192.168.1.0/24
```

### Comprehensive Single Host Scan

```bash
# Full comprehensive scan
nmap -sS -sU -T4 -A -v -p 1-65535 192.168.1.1

# Break down:
# -sS: SYN scan
# -sU: UDP scan
# -T4: Aggressive timing
# -A: OS detection, version detection, script scanning, traceroute
# -v: Verbose output
# -p 1-65535: All ports

# Aggressive scan with all NSE scripts
nmap -T4 -A -v --script=all 192.168.1.1

# Thorough but patient scan
nmap -sS -sU -T2 -A -v -p- --version-all 192.168.1.1
```

### Service Enumeration

```bash
# Enumerate web servers
nmap -sV -p 80,443,8080,8443 --script=http-* 192.168.1.0/24

# Enumerate SMB/Windows hosts
nmap -sV -p 139,445 --script=smb-* 192.168.1.0/24

# Enumerate databases
nmap -sV -p 3306,5432,1433,27017 --script=*-info,*-databases 192.168.1.0/24

# Enumerate mail servers
nmap -sV -p 25,110,143,465,587,993,995 192.168.1.0/24

# Enumerate DNS servers
nmap -sV -p 53 --script=dns-* 192.168.1.0/24

# Enumerate SSH servers
nmap -sV -p 22 --script=ssh-* 192.168.1.0/24
```

### Vulnerability Assessment

```bash
# Basic vulnerability scan
nmap -sV --script=vuln 192.168.1.1

# Web application vulnerabilities
nmap -sV -p 80,443 --script=http-vuln-* 192.168.1.1

# SMB vulnerabilities (EternalBlue, etc.)
nmap -sV -p 445 --script=smb-vuln-* 192.168.1.0/24

# SSL/TLS vulnerabilities
nmap -sV -p 443 --script=ssl-* 192.168.1.1

# Comprehensive vulnerability scan
nmap -sV -p- --script=vuln,exploit 192.168.1.1
```

### Network Inventory

```bash
# Basic inventory
nmap -sn -oA inventory 192.168.1.0/24

# Detailed inventory
nmap -sS -sV -O -oA detailed-inventory 192.168.1.0/24

# Inventory with hostnames
nmap -sn -R -oA inventory-with-hostnames 192.168.1.0/24

# Extract IPs from inventory
grep "Up" inventory.gnmap | awk '{print $2}'

# Extract open ports
grep "open" inventory.gnmap | awk '{print $2, $4}'
```

### Large Network Scanning

```bash
# Fast sweep of large network
nmap -sn -T4 -oA sweep 10.0.0.0/8

# Top 100 ports on large network
nmap -T4 --top-ports 100 --open -oA top100 10.0.0.0/16

# Distributed scanning (split network)
nmap -T4 10.0.0.0/17 -oA scan1 &
nmap -T4 10.0.128.0/17 -oA scan2 &

# Rate-limited scan to avoid overload
nmap --max-rate 100 10.0.0.0/16

# Parallel scanning with GNU parallel
seq 1 254 | parallel -j 10 nmap -T4 -F 192.168.1.{}
```

### Stealth Reconnaissance

```bash
# Stealthy SYN scan with decoys
nmap -sS -T2 -f -D RND:10 192.168.1.1

# Extremely stealthy scan
nmap -sS -T0 -f --randomize-hosts --data-length 25 192.168.1.0/24

# Fragment packets with slow timing
nmap -sS -T1 -f --mtu 16 192.168.1.1

# Idle scan (most stealthy)
# First, find zombie host:
nmap --script=ipidseq 192.168.1.0/24
# Then use zombie:
nmap -sI zombie-host target-host
```

### Firewall Testing

```bash
# Test firewall rules
nmap -sA 192.168.1.1

# Check which ports are filtered
nmap -sS -p- --reason 192.168.1.1 | grep filtered

# Test with different source ports
nmap -sS --source-port 53 192.168.1.1    # DNS
nmap -sS --source-port 20 192.168.1.1    # FTP data

# Fragment scan
nmap -sS -f 192.168.1.1

# Test with bad checksums
nmap --badsum 192.168.1.1
```

### Web Server Analysis

```bash
# Basic web server scan
nmap -sV -p 80,443 --script=http-title,http-headers 192.168.1.1

# Comprehensive web scan
nmap -sV -p 80,443,8080,8443 \
  --script=http-enum,http-headers,http-methods,http-robots.txt,http-title,http-vuln-* \
  192.168.1.1

# SSL/TLS analysis
nmap -sV -p 443 \
  --script=ssl-cert,ssl-enum-ciphers,ssl-heartbleed,ssl-known-key \
  192.168.1.1

# Web application fingerprinting
nmap -sV -p 80,443 \
  --script=http-wordpress-enum,http-drupal-enum,http-joomla-brute \
  192.168.1.1
```

### Windows/SMB Scanning

```bash
# Basic SMB enumeration
nmap -sV -p 445 --script=smb-os-discovery 192.168.1.0/24

# Comprehensive SMB scan
nmap -sV -p 139,445 \
  --script=smb-os-discovery,smb-enum-shares,smb-enum-users,smb-security-mode \
  192.168.1.1

# Check for MS17-010 (EternalBlue)
nmap -sV -p 445 --script=smb-vuln-ms17-010 192.168.1.0/24

# All SMB vulnerabilities
nmap -sV -p 445 --script=smb-vuln-* 192.168.1.0/24

# Windows enumeration
nmap -sV -p 135,139,445,3389 \
  --script=smb-os-discovery,smb-security-mode,rdp-enum-encryption \
  192.168.1.0/24
```

### Database Scanning

```bash
# MySQL enumeration
nmap -sV -p 3306 \
  --script=mysql-info,mysql-databases,mysql-users,mysql-empty-password \
  192.168.1.0/24

# PostgreSQL enumeration
nmap -sV -p 5432 \
  --script=pgsql-brute 192.168.1.0/24

# MSSQL enumeration
nmap -sV -p 1433 \
  --script=ms-sql-info,ms-sql-ntlm-info,ms-sql-empty-password \
  192.168.1.0/24

# MongoDB enumeration
nmap -sV -p 27017 \
  --script=mongodb-info,mongodb-databases \
  192.168.1.0/24

# Redis enumeration
nmap -sV -p 6379 \
  --script=redis-info 192.168.1.0/24
```

### Email Server Scanning

```bash
# SMTP enumeration
nmap -sV -p 25,465,587 \
  --script=smtp-commands,smtp-enum-users,smtp-open-relay \
  192.168.1.1

# IMAP enumeration
nmap -sV -p 143,993 \
  --script=imap-capabilities 192.168.1.1

# POP3 enumeration
nmap -sV -p 110,995 \
  --script=pop3-capabilities 192.168.1.1

# Comprehensive mail server scan
nmap -sV -p 25,110,143,465,587,993,995 \
  --script=smtp-*,imap-*,pop3-* \
  192.168.1.1
```

### IoT and Embedded Device Scanning

```bash
# Common IoT ports
nmap -sS -p 23,80,443,1883,5683,8080,8883 192.168.1.0/24

# MQTT (IoT messaging)
nmap -sV -p 1883,8883 192.168.1.0/24

# CoAP (IoT protocol)
nmap -sU -p 5683 192.168.1.0/24

# UPnP discovery
nmap -sU -p 1900 --script=upnp-info 192.168.1.0/24

# Cameras and NVR
nmap -sV -p 554,8000,8080,8081 192.168.1.0/24
```

### VoIP Scanning

```bash
# SIP scanning
nmap -sU -p 5060 --script=sip-methods 192.168.1.0/24

# SIP enumeration
nmap -sU -p 5060,5061 \
  --script=sip-methods,sip-enum-users \
  192.168.1.0/24

# RTP ports
nmap -sU -p 10000-20000 192.168.1.1
```

### IPv6 Scanning

```bash
# IPv6 ping scan
nmap -6 -sn fe80::1-ff

# IPv6 port scan
nmap -6 2001:db8::1

# IPv6 with version detection
nmap -6 -sV 2001:db8::1

# Local IPv6 discovery
nmap -6 -sn --script=targets-ipv6-multicast-* fe80::/64
```

## Advanced Techniques

### Combining Scan Types

```bash
# TCP SYN and UDP scan together
nmap -sS -sU -p T:1-1000,U:53,161 192.168.1.1

# Multiple discovery methods
nmap -PE -PS22,80,443 -PA80,443 -PU53,161 192.168.1.0/24

# Comprehensive scan with all techniques
nmap -sS -sU -sV -O -A --script=default,vuln -p- 192.168.1.1
```

### Custom TCP Flags

```bash
# Custom flag combinations
nmap --scanflags SYNURG -p 80 192.168.1.1
nmap --scanflags SYNPSH -p 80 192.168.1.1

# Unusual flag combinations for firewall testing
nmap --scanflags URGPSHFIN -p 1-1000 192.168.1.1
```

### Performance Optimization

```bash
# Optimize for fast reliable network
nmap -T4 --min-rate 100 --max-retries 2 192.168.1.0/24

# Optimize for slow unreliable network
nmap -T2 --max-retries 5 --host-timeout 30m 192.168.1.0/24

# Balance speed and accuracy
nmap -T3 --max-retries 3 --max-scan-delay 500ms 192.168.1.0/24

# Maximum speed (sacrifice accuracy)
nmap -T5 --min-rate 1000 --max-retries 1 --host-timeout 5m 192.168.1.0/24
```

### Script Chaining

```bash
# Multiple script categories
nmap --script "default or safe or discovery" 192.168.1.1

# Exclude intrusive scripts
nmap --script "default and not intrusive" 192.168.1.1

# Specific script pattern
nmap --script "http-* and not http-brute" 192.168.1.1

# Complex boolean logic
nmap --script "(http-* or ssh-*) and not (brute or dos)" 192.168.1.1
```

### Output Processing

```bash
# Extract open ports from grepable output
grep "open" scan.gnmap | awk '{print $2, $3, $4}' > open-ports.txt

# Extract IPs with specific port open
grep "22/open" scan.gnmap | awk '{print $2}' > ssh-hosts.txt

# Count hosts by OS
grep "OS:" scan.nmap | sort | uniq -c

# Parse XML with grep
grep -oP '(?<=<port protocol="tcp" portid=")[^"]*' scan.xml

# Convert XML to CSV (with xsltproc)
xsltproc nmap-csv.xsl scan.xml > scan.csv
```

## Best Practices

### Legal and Ethical

1. **Always get authorization**
   - Written permission for all scans
   - Define scope clearly
   - Document authorization

2. **Follow responsible disclosure**
   - Report vulnerabilities properly
   - Give vendors time to fix
   - Coordinate publication

3. **Minimize impact**
   - Use appropriate timing
   - Avoid DOS conditions
   - Test during maintenance windows

4. **Document everything**
   - Keep scan logs
   - Document findings
   - Track remediation

### Technical Best Practices

1. **Start with discovery**
   ```bash
   # First, find live hosts
   nmap -sn 192.168.1.0/24 -oA discovery

   # Then scan live hosts
   nmap -iL live-hosts.txt -sV -oA detailed-scan
   ```

2. **Use appropriate timing**
   ```bash
   # Production networks: use T2 or T3
   nmap -T2 192.168.1.0/24

   # Lab networks: use T4
   nmap -T4 192.168.1.0/24
   ```

3. **Save all output formats**
   ```bash
   # Always use -oA for important scans
   nmap -sV -oA scan-$(date +%Y%m%d) 192.168.1.0/24
   ```

4. **Use version detection**
   ```bash
   # Version detection provides valuable context
   nmap -sV 192.168.1.1
   ```

5. **Scan in stages**
   ```bash
   # Stage 1: Discovery
   nmap -sn 192.168.1.0/24 -oA stage1-discovery

   # Stage 2: Port scan
   nmap -iL live-hosts.txt -F -oA stage2-ports

   # Stage 3: Detailed scan
   nmap -iL interesting-hosts.txt -sV -A -oA stage3-detailed
   ```

6. **Use NSE effectively**
   ```bash
   # Start with safe scripts
   nmap --script=safe 192.168.1.1

   # Progress to specific categories
   nmap --script=vuln 192.168.1.1
   ```

7. **Combine techniques**
   ```bash
   # TCP and UDP
   nmap -sS -sU -p T:80,443,U:53,161 192.168.1.1

   # Multiple discovery methods
   nmap -PE -PS -PA -PU 192.168.1.0/24
   ```

8. **Handle false positives**
   ```bash
   # Verify open ports
   nmap -sV -p 80 192.168.1.1

   # Increase version detection intensity
   nmap -sV --version-all -p 80 192.168.1.1
   ```

### Scan Strategy

1. **Network mapping**
   - Start with broad discovery
   - Identify subnets and segments
   - Map network topology

2. **Progressive scanning**
   - Quick scans first (ping sweep, top ports)
   - Detailed scans on interesting hosts
   - Comprehensive scans on critical targets

3. **Prioritization**
   - Scan critical assets first
   - Focus on internet-facing systems
   - Identify high-risk services

4. **Regular scanning**
   - Schedule periodic scans
   - Compare results over time
   - Track new services/hosts

## Troubleshooting

### Permission Issues

```bash
# Error: "You requested a scan type which requires root privileges"
# Solution: Use sudo
sudo nmap -sS 192.168.1.1

# Alternative: Use non-privileged scan
nmap -sT 192.168.1.1  # TCP connect scan

# Check Nmap capabilities (Linux)
getcap $(which nmap)

# Set capabilities (alternative to root)
sudo setcap cap_net_raw,cap_net_admin,cap_net_bind_service+eip $(which nmap)
```

### No Results or Timeouts

```bash
# Increase timeout
nmap --host-timeout 10m 192.168.1.1

# Skip host discovery (if firewall blocks pings)
nmap -Pn 192.168.1.1

# Increase retries
nmap --max-retries 5 192.168.1.1

# Use slower timing
nmap -T2 192.168.1.1

# Check network connectivity
ping 192.168.1.1
traceroute 192.168.1.1
```

### Slow Scans

```bash
# Use faster timing template
nmap -T4 192.168.1.0/24

# Scan fewer ports
nmap -F 192.168.1.0/24
nmap --top-ports 100 192.168.1.0/24

# Disable version detection
nmap -sS 192.168.1.0/24  # Without -sV

# Disable OS detection
nmap -sS 192.168.1.0/24  # Without -O

# Increase minimum rate
nmap --min-rate 100 192.168.1.0/24

# Reduce retries
nmap --max-retries 1 192.168.1.0/24
```

### Firewall Blocking

```bash
# Skip ping
nmap -Pn 192.168.1.1

# Try different scan types
nmap -sT 192.168.1.1  # Connect scan
nmap -sA 192.168.1.1  # ACK scan

# Use different source port
nmap --source-port 53 192.168.1.1
nmap --source-port 20 192.168.1.1

# Fragment packets
nmap -f 192.168.1.1

# Check what's being blocked
nmap --packet-trace -p 80 192.168.1.1
```

### Accuracy Issues

```bash
# Increase version detection intensity
nmap -sV --version-intensity 9 192.168.1.1

# Enable aggressive detection
nmap -A 192.168.1.1

# Disable ping (if causing issues)
nmap -Pn 192.168.1.1

# Use more probes
nmap --max-retries 5 192.168.1.1

# Verify with different scan types
nmap -sS -p 80 192.168.1.1
nmap -sT -p 80 192.168.1.1
nmap -sV -p 80 192.168.1.1
```

### Script Errors

```bash
# Update script database
nmap --script-updatedb

# Debug script execution
nmap --script=http-title --script-trace 192.168.1.1

# Check script help
nmap --script-help http-title

# Verify script exists
ls /usr/share/nmap/scripts/http-title.nse

# Run with debug
nmap --script=http-title -d 192.168.1.1
```

### DNS Issues

```bash
# Disable DNS resolution
nmap -n 192.168.1.1

# Use custom DNS servers
nmap --dns-servers 8.8.8.8,8.8.4.4 192.168.1.1

# Force DNS resolution
nmap -R 192.168.1.1
```

### Network Interface Issues

```bash
# List interfaces
nmap --iflist

# Specify interface
nmap -e eth0 192.168.1.1

# Specify source IP
nmap -S 192.168.1.100 -e eth0 192.168.1.1
```

## Quick Reference

### Essential Commands

```bash
# Basic scan
nmap 192.168.1.1

# Scan subnet
nmap 192.168.1.0/24

# Ping scan (no port scan)
nmap -sn 192.168.1.0/24

# Fast scan
nmap -F 192.168.1.1

# Scan specific ports
nmap -p 22,80,443 192.168.1.1

# Scan all ports
nmap -p- 192.168.1.1

# Version detection
nmap -sV 192.168.1.1

# OS detection
nmap -O 192.168.1.1

# Aggressive scan
nmap -A 192.168.1.1

# Script scan
nmap -sC 192.168.1.1
nmap --script=vuln 192.168.1.1

# Skip host discovery
nmap -Pn 192.168.1.1

# Save output
nmap -oA scan 192.168.1.1

# Verbose
nmap -v 192.168.1.1
```

### Common Options Table

| Option | Description |
|--------|-------------|
| `-sS` | TCP SYN scan (default) |
| `-sT` | TCP connect scan |
| `-sU` | UDP scan |
| `-sV` | Version detection |
| `-O` | OS detection |
| `-A` | Aggressive (OS, version, scripts, traceroute) |
| `-T0-T5` | Timing template (0=paranoid, 5=insane) |
| `-p` | Port specification |
| `-p-` | All ports |
| `-F` | Fast (100 ports) |
| `--top-ports N` | Scan top N ports |
| `-Pn` | Skip host discovery |
| `-sn` | Ping scan only (no port scan) |
| `-n` | No DNS resolution |
| `-R` | Always resolve DNS |
| `-v` | Verbose |
| `-d` | Debug |
| `-oN` | Normal output |
| `-oX` | XML output |
| `-oG` | Grepable output |
| `-oA` | All output formats |
| `--script` | Run NSE scripts |
| `--script-help` | Show script help |
| `--open` | Show only open ports |
| `--reason` | Show reason for port state |
| `--packet-trace` | Show packets sent/received |

### Port States

| State | Meaning |
|-------|---------|
| `open` | Application accepting connections |
| `closed` | Port accessible but no application |
| `filtered` | Packet filtering prevents determination |
| `unfiltered` | Accessible but state unknown |
| `open\|filtered` | Open or filtered (cannot determine) |
| `closed\|filtered` | Closed or filtered (cannot determine) |

### Timing Templates

| Template | Name | Use Case |
|----------|------|----------|
| `-T0` | Paranoid | IDS evasion |
| `-T1` | Sneaky | IDS evasion |
| `-T2` | Polite | Low bandwidth |
| `-T3` | Normal | Default |
| `-T4` | Aggressive | Fast networks |
| `-T5` | Insane | Very fast networks |

### Common NSE Script Categories

| Category | Description |
|----------|-------------|
| `auth` | Authentication testing |
| `broadcast` | Network discovery |
| `brute` | Brute force attacks |
| `default` | Default scripts |
| `discovery` | Service discovery |
| `dos` | Denial of service |
| `exploit` | Exploitation |
| `external` | External resources |
| `intrusive` | Intrusive tests |
| `malware` | Malware detection |
| `safe` | Safe scripts |
| `vuln` | Vulnerability detection |

## Conclusion

Nmap is an incredibly powerful and versatile network scanning tool. Mastering it requires understanding:

1. **Target specification** - Flexible ways to define what to scan
2. **Host discovery** - Determining which hosts are alive
3. **Port scanning** - Various techniques for different scenarios
4. **Service detection** - Identifying versions and configurations
5. **OS fingerprinting** - Determining operating systems
6. **NSE scripts** - Extending functionality for specific tasks
7. **Timing and performance** - Balancing speed and accuracy
8. **Output formats** - Processing and analyzing results

**Key Takeaways:**

- Always obtain proper authorization before scanning
- Start with discovery, then detailed scanning
- Use appropriate timing for the network
- Save output in multiple formats (-oA)
- Combine techniques for comprehensive results
- Understand what each scan type reveals
- Use NSE scripts for specific tasks
- Interpret results in context
- Verify findings with multiple methods
- Document everything for reporting

**Learning Path:**

1. **Week 1**: Basic scanning (-sn, -sS, -p, -sV)
2. **Week 2**: Output formats, timing templates
3. **Week 3**: OS detection, aggressive scanning
4. **Week 4**: NSE scripts, common use cases
5. **Month 2**: Advanced techniques, evasion, optimization
6. **Month 3+**: Custom scripts, integration, automation

Nmap is an essential tool for network administrators, security professionals, and penetration testers. The more you practice with it, the more valuable it becomes for understanding and securing networks.

**Resources:**
- Official Nmap documentation: https://nmap.org/docs.html
- Nmap book: https://nmap.org/book/
- NSE script documentation: https://nmap.org/nsedoc/
- Practice safely on authorized networks only

Happy scanning!
