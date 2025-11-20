# nftables

## Introduction

**nftables** is the modern Linux kernel packet filtering framework that replaces the legacy iptables, ip6tables, arptables, and ebtables frameworks. It was merged into the Linux kernel in version 3.13 (January 2014) and provides a unified interface for packet filtering, network address translation (NAT), and packet mangling.

### Why nftables?

The legacy iptables framework had several limitations:
- Separate tools for different protocols (iptables, ip6tables, arptables, ebtables)
- Code duplication across different tools
- Limited performance for large rulesets
- Rule updates were not atomic
- Complex syntax with many different match extensions

nftables addresses these issues with:
- **Unified syntax** across all protocols
- **Better performance** through improved data structures (sets, maps)
- **Atomic rule updates** - all-or-nothing rule changes
- **Simplified syntax** with more intuitive expressions
- **Built-in scripting** capabilities
- **Smaller kernel footprint** - generic classification engine
- **No protocol-specific kernel modules** required

### Key Features

- Single `nft` command for all operations
- Address family support: IPv4, IPv6, ARP, bridge, inet (IPv4+IPv6), netdev
- Advanced data structures: sets, maps, dictionaries
- Concatenations for multi-dimensional matching
- Native support for intervals and ranges
- Better integration with connection tracking
- JSON API for programmatic access
- Improved rule debugging and tracing

## Architecture

### Netfilter Hooks

nftables uses the same Netfilter hooks as iptables:

```
                                   ┌──────────────┐
                                   │   Network    │
                                   │   Interface  │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │  PREROUTING  │ ◄── DNAT, early filtering
                                   └──────┬───────┘
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                              ▼                       ▼
                       ┌─────────────┐         ┌──────────┐
                       │   FORWARD   │         │  INPUT   │ ◄── Local process
                       └──────┬──────┘         └──────────┘
                              │                       ▲
                              │                       │
                              ▼                 ┌──────────┐
                       ┌─────────────┐         │  OUTPUT  │ ◄── Locally generated
                       │ POSTROUTING │         └──────────┘
                       └──────┬──────┘               ▲
                              │                       │
                              └───────────┬───────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │   Network    │
                                   │   Interface  │
                                   └──────────────┘
```

### Address Families

nftables organizes rules by address family:

| Family | Description | Kernel support |
|--------|-------------|----------------|
| `ip` | IPv4 packets only | 3.13+ |
| `ip6` | IPv6 packets only | 3.13+ |
| `inet` | Both IPv4 and IPv6 (recommended) | 3.14+ |
| `arp` | ARP packets | 3.13+ |
| `bridge` | Bridge layer (ebtables replacement) | 3.18+ |
| `netdev` | Ingress hook (pre-routing) | 4.2+ |

### Components Hierarchy

```
Address Family (ip, ip6, inet, arp, bridge, netdev)
  │
  └── Table (filter, nat, mangle, etc. - arbitrary names)
       │
       └── Chain (input, forward, output, prerouting, postrouting - arbitrary names)
            │
            └── Rule (match expressions + verdict)
```

### Tables

Tables are containers for chains. Unlike iptables, table names are arbitrary and you can create custom tables:

```bash
# Create a custom table
nft add table inet my_firewall

# List all tables
nft list tables

# Delete a table
nft delete table inet my_firewall
```

### Chains

Chains contain rules. There are two types:

1. **Base chains** - attached to netfilter hooks
2. **Regular chains** - for jump targets (like custom chains in iptables)

Base chain properties:
- `type`: filter, nat, route
- `hook`: prerouting, input, forward, output, postrouting
- `priority`: integer value (lower = earlier processing)
- `policy`: accept, drop (default verdict)

### Rules

Rules consist of:
- **Match expressions** - conditions to match packets
- **Statements** - actions to take (verdict, counter, log, etc.)

### Verdicts

| Verdict | Description |
|---------|-------------|
| `accept` | Accept the packet |
| `drop` | Drop the packet silently |
| `reject` | Drop with ICMP/TCP RST error |
| `queue` | Pass to userspace |
| `continue` | Continue evaluation with next rule |
| `return` | Return to calling chain |
| `jump <chain>` | Jump to chain |
| `goto <chain>` | Goto chain (can't return) |

## Installation

### Debian/Ubuntu

```bash
# Install nftables
sudo apt update
sudo apt install nftables

# Check version
nft --version

# Enable service
sudo systemctl enable nftables
sudo systemctl start nftables
```

### Red Hat/CentOS/Fedora

```bash
# Install nftables
sudo dnf install nftables

# Enable service
sudo systemctl enable nftables
sudo systemctl start nftables
```

### Arch Linux

```bash
# Install nftables
sudo pacman -S nftables

# Enable service
sudo systemctl enable nftables
sudo systemctl start nftables
```

### Kernel Requirements

- Minimum kernel: 3.13
- Recommended: 4.14+ for full feature set
- Check kernel support:

```bash
# Check if nftables module is loaded
lsmod | grep nf_tables

# Load module manually
sudo modprobe nf_tables
```

## Basic Operations

### Creating Tables

```bash
# Create IPv4 table
nft add table ip my_filter

# Create IPv6 table
nft add table ip6 my_filter6

# Create dual-stack table (recommended)
nft add table inet my_filter

# Create table for bridge
nft add table bridge br_filter
```

### Creating Chains

```bash
# Create base chain for input filtering
nft add chain inet my_filter input { \
    type filter hook input priority 0 \; \
    policy drop \; \
}

# Create base chain for output
nft add chain inet my_filter output { \
    type filter hook output priority 0 \; \
    policy accept \; \
}

# Create regular chain (for jumping)
nft add chain inet my_filter tcp_chain
```

### Chain Priorities

Priority determines chain order (lower number = earlier processing):

| Priority | Name | Value | Description |
|----------|------|-------|-------------|
| NF_IP_PRI_CONNTRACK_DEFRAG | | -400 | Defragmentation |
| NF_IP_PRI_RAW | raw | -300 | Before connection tracking |
| NF_IP_PRI_SELINUX_FIRST | | -225 | SELinux operations |
| NF_IP_PRI_CONNTRACK | | -200 | Connection tracking |
| NF_IP_PRI_MANGLE | mangle | -150 | Packet mangling |
| NF_IP_PRI_NAT_DST | dstnat | -100 | DNAT |
| NF_IP_PRI_FILTER | filter | 0 | Standard filtering |
| NF_IP_PRI_SECURITY | security | 50 | Security tables |
| NF_IP_PRI_NAT_SRC | srcnat | 100 | SNAT |
| NF_IP_PRI_SELINUX_LAST | | 225 | SELinux |
| NF_IP_PRI_CONNTRACK_HELPER | | 300 | Connection tracking helpers |

### Adding Rules

```bash
# Add rule to accept established connections
nft add rule inet my_filter input ct state established,related accept

# Add rule to drop invalid packets
nft add rule inet my_filter input ct state invalid drop

# Add rule to accept SSH
nft add rule inet my_filter input tcp dport 22 accept

# Insert rule at beginning (position 0)
nft insert rule inet my_filter input position 0 ct state new accept

# Add rule with counter
nft add rule inet my_filter input tcp dport 80 counter accept

# Add rule with log
nft add rule inet my_filter input tcp dport 443 log prefix \"HTTPS: \" accept
```

### Listing Rules

```bash
# List all rules
nft list ruleset

# List specific table
nft list table inet my_filter

# List specific chain
nft list chain inet my_filter input

# List with handles (for deletion)
nft --handle list ruleset

# List in JSON format
nft -j list ruleset

# List with counters
nft list table inet my_filter
```

### Deleting Rules

```bash
# Delete rule by handle
nft delete rule inet my_filter input handle 5

# Delete entire chain
nft delete chain inet my_filter input

# Delete entire table
nft delete table inet my_filter

# Flush all rules in chain (keep chain)
nft flush chain inet my_filter input

# Flush all rules in table
nft flush table inet my_filter

# Flush everything
nft flush ruleset
```

### Saving and Restoring

```bash
# Save current ruleset to file
nft list ruleset > /etc/nftables.conf

# Restore from file
nft -f /etc/nftables.conf

# Atomic replacement (test mode)
nft -c -f /etc/nftables.conf  # Check syntax only
nft -f /etc/nftables.conf      # Apply if syntax OK

# Debian/Ubuntu service uses
/etc/nftables.conf
```

## Basic Filtering

### Protocol Filtering

```bash
# Accept TCP traffic
nft add rule inet my_filter input ip protocol tcp accept

# Accept UDP traffic
nft add rule inet my_filter input ip protocol udp accept

# Accept ICMP (IPv4)
nft add rule inet my_filter input ip protocol icmp accept

# Accept ICMPv6
nft add rule inet my_filter input meta l4proto ipv6-icmp accept

# Shorter syntax using meta
nft add rule inet my_filter input meta l4proto tcp accept
nft add rule inet my_filter input meta l4proto udp accept
```

### Port Filtering

```bash
# Single port (SSH)
nft add rule inet my_filter input tcp dport 22 accept

# Multiple ports
nft add rule inet my_filter input tcp dport { 80, 443, 8080 } accept

# Port range
nft add rule inet my_filter input tcp dport 1024-65535 accept

# Source port
nft add rule inet my_filter output tcp sport 22 accept

# Both source and destination
nft add rule inet my_filter forward tcp sport 1024-65535 tcp dport 80 accept
```

### IP Address Filtering

```bash
# Single IP address
nft add rule inet my_filter input ip saddr 192.168.1.100 accept

# CIDR notation
nft add rule inet my_filter input ip saddr 192.168.1.0/24 accept

# Multiple IPs
nft add rule inet my_filter input ip saddr { 192.168.1.100, 192.168.1.101 } accept

# IP range
nft add rule inet my_filter input ip saddr 192.168.1.100-192.168.1.200 accept

# Destination address
nft add rule inet my_filter output ip daddr 8.8.8.8 accept

# IPv6 address
nft add rule inet my_filter input ip6 saddr 2001:db8::/32 accept
```

### Interface Filtering

```bash
# Input interface
nft add rule inet my_filter input iifname "eth0" accept

# Output interface
nft add rule inet my_filter output oifname "eth0" accept

# Multiple interfaces
nft add rule inet my_filter input iifname { "eth0", "eth1" } accept

# Wildcard matching
nft add rule inet my_filter input iifname "eth*" accept

# Loopback
nft add rule inet my_filter input iifname "lo" accept
```

### MAC Address Filtering

```bash
# Match source MAC
nft add rule inet my_filter input ether saddr 00:11:22:33:44:55 accept

# Match destination MAC
nft add rule inet my_filter input ether daddr 00:11:22:33:44:55 accept

# Multiple MACs
nft add rule inet my_filter input ether saddr { \
    00:11:22:33:44:55, \
    00:11:22:33:44:56 \
} accept
```

### Connection State Filtering

```bash
# Accept established and related connections
nft add rule inet my_filter input ct state established,related accept

# Drop invalid packets
nft add rule inet my_filter input ct state invalid drop

# Accept new connections on specific port
nft add rule inet my_filter input tcp dport 80 ct state new accept

# Match specific states
nft add rule inet my_filter input ct state { established, related, new } accept

# Connection tracking states:
# - new: First packet of connection
# - established: Part of existing connection
# - related: Related to established connection (e.g., FTP data)
# - invalid: Packet doesn't match any connection
# - untracked: Packet marked to bypass tracking
```

### ICMP Filtering

```bash
# Accept all ICMP (IPv4)
nft add rule inet my_filter input ip protocol icmp accept

# Accept specific ICMP types (echo-request = ping)
nft add rule inet my_filter input icmp type echo-request accept

# Accept echo-reply
nft add rule inet my_filter input icmp type echo-reply accept

# Limit ping rate
nft add rule inet my_filter input icmp type echo-request limit rate 5/second accept

# ICMPv6 essential types (required for IPv6)
nft add rule inet my_filter input icmpv6 type { \
    destination-unreachable, \
    packet-too-big, \
    time-exceeded, \
    parameter-problem, \
    echo-request, \
    echo-reply \
} accept

# IPv6 neighbor discovery (required)
nft add rule inet my_filter input icmpv6 type { \
    nd-neighbor-solicit, \
    nd-neighbor-advert, \
    nd-router-advert, \
    nd-router-solicit \
} accept
```

### TCP Flags Filtering

```bash
# Match SYN packets (new connections)
nft add rule inet my_filter input tcp flags syn tcp flags != ack counter

# Drop NULL packets
nft add rule inet my_filter input tcp flags == 0 drop

# Drop FIN+SYN packets (XMAS)
nft add rule inet my_filter input tcp flags \& (fin|syn) == (fin|syn) drop

# Drop SYN+RST packets
nft add rule inet my_filter input tcp flags \& (syn|rst) == (syn|rst) drop

# Accept only SYN for new connections
nft add rule inet my_filter input ct state new tcp flags != syn drop
```

## NAT (Network Address Translation)

### NAT Table Setup

```bash
# Create NAT table
nft add table inet nat

# Create prerouting chain for DNAT
nft add chain inet nat prerouting { \
    type nat hook prerouting priority -100 \; \
}

# Create postrouting chain for SNAT
nft add chain inet nat postrouting { \
    type nat hook postrouting priority 100 \; \
}
```

### SNAT (Source NAT)

```bash
# SNAT to specific IP
nft add rule inet nat postrouting oifname "eth0" ip saddr 192.168.1.0/24 snat to 203.0.113.1

# SNAT with port range
nft add rule inet nat postrouting oifname "eth0" snat to 203.0.113.1:1024-65535

# SNAT to multiple IPs (load balancing)
nft add rule inet nat postrouting oifname "eth0" snat to 203.0.113.1-203.0.113.10

# Conditional SNAT
nft add rule inet nat postrouting oifname "eth0" ip daddr != 192.168.0.0/16 snat to 203.0.113.1
```

### Masquerading

Masquerading is SNAT with automatic IP detection (useful for dynamic IPs):

```bash
# Basic masquerading
nft add rule inet nat postrouting oifname "eth0" masquerade

# Masquerade with port range
nft add rule inet nat postrouting oifname "eth0" masquerade to :1024-65535

# Masquerade specific subnet
nft add rule inet nat postrouting oifname "ppp0" ip saddr 192.168.1.0/24 masquerade

# Masquerade for IPv6
nft add rule inet nat postrouting oifname "eth0" ip6 saddr fd00::/64 masquerade
```

### DNAT (Destination NAT)

```bash
# Port forwarding (simple)
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.100

# Port forwarding with port change
nft add rule inet nat prerouting iifname "eth0" tcp dport 8080 dnat to 192.168.1.100:80

# DNAT with IP range (load balancing)
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.100-192.168.1.110

# DNAT with port range
nft add rule inet nat prerouting tcp dport 5000-5999 dnat to 192.168.1.100:6000-6999

# Conditional DNAT
nft add rule inet nat prerouting iifname "eth0" ip saddr 203.0.113.0/24 tcp dport 80 dnat to 192.168.1.100
```

### Port Forwarding Examples

```bash
# Forward HTTP to internal server
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.10:80

# Forward HTTPS to internal server
nft add rule inet nat prerouting iifname "eth0" tcp dport 443 dnat to 192.168.1.10:443

# Forward SSH to different port
nft add rule inet nat prerouting iifname "eth0" tcp dport 2222 dnat to 192.168.1.10:22

# Forward range of ports
nft add rule inet nat prerouting iifname "eth0" tcp dport 3000-3100 dnat to 192.168.1.10

# Forward to multiple servers (round-robin)
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to numgen inc mod 3 map { \
    0 : 192.168.1.10, \
    1 : 192.168.1.11, \
    2 : 192.168.1.12 \
}
```

### 1:1 NAT

```bash
# Bidirectional 1:1 NAT
# Public IP 203.0.113.10 <-> Private IP 192.168.1.10

# DNAT (inbound)
nft add rule inet nat prerouting iifname "eth0" ip daddr 203.0.113.10 dnat to 192.168.1.10

# SNAT (outbound)
nft add rule inet nat postrouting oifname "eth0" ip saddr 192.168.1.10 snat to 203.0.113.10
```

### Hairpin NAT (NAT Loopback)

Allow internal hosts to access internal servers via public IP:

```bash
# Create filter table rule to allow forwarding
nft add rule inet filter forward iifname "br0" oifname "br0" accept

# DNAT for external access
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.10:80

# SNAT for hairpin (internal to internal via public IP)
nft add rule inet nat postrouting oifname "br0" ip saddr 192.168.1.0/24 ip daddr 192.168.1.10 masquerade

# DNAT for hairpin
nft add rule inet nat prerouting iifname "br0" tcp dport 80 dnat to 192.168.1.10:80
```

### Full NAT Example (Router)

```bash
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# Create tables
nft add table inet filter
nft add table inet nat

# Filter chains
nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }

# NAT chains
nft add chain inet nat prerouting { type nat hook prerouting priority -100 \; }
nft add chain inet nat postrouting { type nat hook postrouting priority 100 \; }

# Input rules
nft add rule inet filter input iifname "lo" accept
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input iifname "eth1" ct state new accept  # LAN
nft add rule inet filter input icmp type echo-request limit rate 5/second accept

# Forward rules
nft add rule inet filter forward ct state established,related accept
nft add rule inet filter forward iifname "eth1" oifname "eth0" accept  # LAN to WAN

# NAT rules
nft add rule inet nat postrouting oifname "eth0" masquerade

# Port forwarding (example: web server)
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.10
nft add rule inet filter forward iifname "eth0" oifname "eth1" tcp dport 80 ct state new accept
```

## Advanced Features - Sets

Sets are collections of elements that can be matched efficiently.

### Anonymous Sets

Anonymous sets are defined inline with the rule:

```bash
# Set of ports
nft add rule inet my_filter input tcp dport { 22, 80, 443 } accept

# Set of IP addresses
nft add rule inet my_filter input ip saddr { 192.168.1.1, 192.168.1.2 } accept

# Mixed set
nft add rule inet my_filter input tcp dport { 22, 80, 443, 8080-8090 } accept
```

### Named Sets

Named sets can be reused across multiple rules:

```bash
# Create set of IPv4 addresses
nft add set inet my_filter allowed_ips { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add elements to set
nft add element inet my_filter allowed_ips { 192.168.1.1, 192.168.1.2, 192.168.1.0/24 }

# Use set in rule
nft add rule inet my_filter input ip saddr @allowed_ips accept

# Create set of ports
nft add set inet my_filter web_ports { \
    type inet_service \; \
}

nft add element inet my_filter web_ports { 80, 443, 8080, 8443 }
nft add rule inet my_filter input tcp dport @web_ports accept
```

### Dynamic Sets (with Timeout)

```bash
# Create dynamic set with timeout
nft add set inet my_filter blocklist { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 1h \; \
}

# Add to set dynamically with update statement
nft add rule inet my_filter input tcp dport 22 ct state new \
    add @blocklist { ip saddr timeout 1h } \
    counter

# Drop packets from blocklist
nft add rule inet my_filter input ip saddr @blocklist drop
```

### Set Types

Available set types:

| Type | Description |
|------|-------------|
| `ipv4_addr` | IPv4 address |
| `ipv6_addr` | IPv6 address |
| `ether_addr` | Ethernet (MAC) address |
| `inet_proto` | Internet protocol (tcp, udp, etc.) |
| `inet_service` | Port number |
| `mark` | Packet mark |
| `ifname` | Interface name |

### Intervals in Sets

```bash
# Create set with interval support
nft add set inet my_filter ip_ranges { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add IP ranges
nft add element inet my_filter ip_ranges { \
    10.0.0.0-10.0.0.255, \
    192.168.0.0/16, \
    172.16.0.1-172.16.255.254 \
}

# Use in rule
nft add rule inet my_filter input ip saddr @ip_ranges drop
```

### Blacklist Example

```bash
# Create blacklist set
nft add set inet my_filter blacklist { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add malicious IPs
nft add element inet my_filter blacklist { \
    198.51.100.0/24, \
    203.0.113.0/24 \
}

# Drop traffic from blacklist (put early in chain)
nft insert rule inet my_filter input ip saddr @blacklist drop

# Add to blacklist on-the-fly
nft add element inet my_filter blacklist { 192.0.2.100 }
```

### Whitelist Example

```bash
# Create whitelist
nft add set inet my_filter whitelist { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add trusted IPs
nft add element inet my_filter whitelist { \
    192.168.1.0/24, \
    10.0.0.0/8 \
}

# Allow SSH only from whitelist
nft add rule inet my_filter input tcp dport 22 ip saddr @whitelist accept
nft add rule inet my_filter input tcp dport 22 drop
```

### Port Knock Implementation

```bash
# Create sets for port knock sequence
nft add set inet my_filter knock1 { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 5s \; \
}

nft add set inet my_filter knock2 { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 5s \; \
}

nft add set inet my_filter knock3 { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 5s \; \
}

nft add set inet my_filter authorized { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 30s \; \
}

# Knock sequence: 7000, 8000, 9000
nft add rule inet my_filter input tcp dport 7000 \
    add @knock1 { ip saddr }

nft add rule inet my_filter input ip saddr @knock1 tcp dport 8000 \
    add @knock2 { ip saddr }

nft add rule inet my_filter input ip saddr @knock2 tcp dport 9000 \
    add @authorized { ip saddr timeout 30s }

# Allow SSH from authorized IPs
nft add rule inet my_filter input ip saddr @authorized tcp dport 22 accept
```

## Advanced Features - Maps

Maps associate keys with values for dynamic packet handling.

### Basic Maps

```bash
# Create map for port-based NAT
nft add map inet nat port_nat { \
    type inet_service : ipv4_addr \; \
}

# Add mappings
nft add element inet nat port_nat { \
    80 : 192.168.1.10, \
    443 : 192.168.1.11, \
    25 : 192.168.1.12 \
}

# Use map for DNAT
nft add rule inet nat prerouting dnat to tcp dport map @port_nat
```

### Verdict Maps

Maps can return verdicts:

```bash
# Create verdict map
nft add map inet my_filter port_policy { \
    type inet_service : verdict \; \
}

# Add port policies
nft add element inet my_filter port_policy { \
    22 : accept, \
    80 : accept, \
    443 : accept, \
    23 : drop \
}

# Apply map
nft add rule inet my_filter input tcp dport vmap @port_policy
```

### IP-based Routing Map

```bash
# Create map for different SNAT based on destination
nft add map inet nat snat_map { \
    type ipv4_addr : ipv4_addr \; \
    flags interval \; \
}

nft add element inet nat snat_map { \
    10.0.0.0/8 : 192.168.1.1, \
    172.16.0.0/12 : 192.168.1.2, \
    0.0.0.0/0 : 203.0.113.1 \
}

nft add rule inet nat postrouting snat to ip daddr map @snat_map
```

### Counter Map

```bash
# Create map for per-port counters
nft add map inet my_filter port_counters { \
    type inet_service : counter \; \
}

# Initialize counters
nft add element inet my_filter port_counters { \
    80 : counter, \
    443 : counter, \
    22 : counter \
}

# Count packets per port
nft add rule inet my_filter input tcp dport vmap @port_counters { \
    80 : accept, \
    443 : accept, \
    22 : accept \
}
```

### Load Balancing with Maps

```bash
# Round-robin load balancing
nft add rule inet nat prerouting tcp dport 80 dnat to numgen inc mod 3 map { \
    0 : 192.168.1.10, \
    1 : 192.168.1.11, \
    2 : 192.168.1.12 \
}

# Random load balancing
nft add rule inet nat prerouting tcp dport 80 dnat to numgen random mod 3 map { \
    0 : 192.168.1.10, \
    1 : 192.168.1.11, \
    2 : 192.168.1.12 \
}

# Hash-based (consistent hashing)
nft add rule inet nat prerouting tcp dport 80 dnat to jhash ip saddr mod 3 map { \
    0 : 192.168.1.10, \
    1 : 192.168.1.11, \
    2 : 192.168.1.12 \
}
```

## Advanced Features - Concatenations

Concatenations allow matching multiple criteria simultaneously.

### Basic Concatenation

```bash
# Match IP and port combination
nft add rule inet my_filter input ip saddr . tcp dport { \
    192.168.1.10 . 22, \
    192.168.1.11 . 80, \
    192.168.1.12 . 443 \
} accept
```

### Concatenation Sets

```bash
# Create set with concatenation
nft add set inet my_filter allowed_connections { \
    type ipv4_addr . inet_service \; \
}

# Add elements
nft add element inet my_filter allowed_connections { \
    192.168.1.10 . 22, \
    192.168.1.10 . 80, \
    192.168.1.11 . 443 \
}

# Use in rule
nft add rule inet my_filter input ip saddr . tcp dport @allowed_connections accept
```

### Concatenation Maps

```bash
# Create map with concatenated key
nft add map inet nat dnat_map { \
    type ipv4_addr . inet_service : ipv4_addr . inet_service \; \
}

# Add mappings (public IP:port -> internal IP:port)
nft add element inet nat dnat_map { \
    203.0.113.1 . 80 : 192.168.1.10 . 80, \
    203.0.113.1 . 443 : 192.168.1.10 . 443, \
    203.0.113.1 . 25 : 192.168.1.11 . 25 \
}

# Use for DNAT
nft add rule inet nat prerouting dnat to ip daddr . tcp dport map @dnat_map
```

### Interface + IP Concatenation

```bash
# Create set for interface and IP
nft add set inet my_filter trusted_sources { \
    type ifname . ipv4_addr \; \
}

# Add trusted sources
nft add element inet my_filter trusted_sources { \
    "eth0" . 192.168.1.0/24, \
    "eth1" . 10.0.0.0/8 \
}

# Use in rule
nft add rule inet my_filter input iifname . ip saddr @trusted_sources accept
```

## Rate Limiting

### Basic Rate Limiting

```bash
# Limit ICMP to 5 per second
nft add rule inet my_filter input icmp type echo-request limit rate 5/second accept

# Limit SSH connections (per source)
nft add rule inet my_filter input tcp dport 22 ct state new limit rate 3/minute accept

# Limit with burst
nft add rule inet my_filter input tcp dport 80 limit rate 100/second burst 200 packets accept
```

### Rate Limiting Units

```bash
# Per second
nft add rule inet my_filter input limit rate 10/second accept

# Per minute
nft add rule inet my_filter input limit rate 60/minute accept

# Per hour
nft add rule inet my_filter input limit rate 1000/hour accept

# Per day
nft add rule inet my_filter input limit rate 10000/day accept

# Bytes per second
nft add rule inet my_filter input limit rate 1 mbytes/second accept
```

### Per-Source Rate Limiting

```bash
# Create dynamic set for rate limiting per IP
nft add set inet my_filter rate_limit { \
    type ipv4_addr \; \
    size 65535 \; \
    flags dynamic \; \
}

# Limit new connections per source
nft add rule inet my_filter input tcp dport 80 ct state new \
    meter rate_limit { ip saddr limit rate 10/second } accept

# Alternative syntax
nft add rule inet my_filter input tcp dport 22 ct state new \
    meter ssh_limit { ip saddr timeout 1m limit rate 3/minute } accept
```

### SYN Flood Protection

```bash
# Limit SYN packets
nft add rule inet my_filter input tcp flags syn tcp flags \& \(fin\|syn\|rst\|ack\) == syn \
    meter syn_flood { ip saddr timeout 10s limit rate 25/second burst 50 packets } accept

nft add rule inet my_filter input tcp flags syn drop
```

### HTTP Rate Limiting

```bash
# Limit HTTP requests per IP
nft add rule inet my_filter input tcp dport 80 ct state new \
    meter http_meter { ip saddr timeout 10s limit rate 20/second } accept

nft add rule inet my_filter input tcp dport 80 ct state new \
    log prefix "HTTP RATE LIMIT: " drop
```

### Quota Limiting

```bash
# Allow 1GB quota
nft add rule inet my_filter forward quota 1 gbytes accept
nft add rule inet my_filter forward counter drop

# Per-user quota (using marks)
nft add rule inet my_filter forward meta mark 1 quota 100 mbytes accept
nft add rule inet my_filter forward meta mark 1 drop
```

## Connection Tracking

### Connection Tracking States

```bash
# Accept established connections
nft add rule inet my_filter input ct state established accept

# Accept related connections (e.g., FTP data channel)
nft add rule inet my_filter input ct state related accept

# Drop invalid packets
nft add rule inet my_filter input ct state invalid drop

# Allow new connections on specific ports
nft add rule inet my_filter input tcp dport 80 ct state new accept
```

### Connection Tracking Helpers

```bash
# Load FTP helper
modprobe nf_conntrack_ftp

# Use FTP helper
nft add rule inet filter input ct helper "ftp" accept

# SIP helper
modprobe nf_conntrack_sip
nft add rule inet filter input tcp dport 5060 ct helper set "sip"
```

### Connection Mark (connmark)

```bash
# Mark connections
nft add rule inet my_filter prerouting tcp dport 80 ct mark set 1

# Match by connection mark
nft add rule inet my_filter forward ct mark 1 accept

# Restore mark from connection
nft add rule inet my_filter output ct mark != 0 meta mark set ct mark

# Save mark to connection
nft add rule inet my_filter input meta mark != 0 ct mark set meta mark
```

### Connection Zones

```bash
# Assign connection to zone
nft add rule inet my_filter prerouting ct zone 1

# Match by zone
nft add rule inet my_filter forward ct zone 1 accept
```

### Connection Limits

```bash
# Limit concurrent connections per source
nft add rule inet my_filter input tcp dport 22 ct state new \
    meter ssh_conn { ip saddr ct count over 3 } drop

# Limit connections globally
nft add rule inet my_filter input tcp dport 80 ct state new \
    add @connlimit { ip saddr ct count over 10 } drop
```

## Logging and Monitoring

### Basic Logging

```bash
# Log all dropped packets
nft add rule inet my_filter input counter log drop

# Log with prefix
nft add rule inet my_filter input log prefix "DROPPED: " drop

# Log to specific group (for ulogd)
nft add rule inet my_filter input log group 2 drop

# Log with level
nft add rule inet my_filter input log level warn prefix "SUSPICIOUS: " drop
```

### Log Levels

```bash
# Available levels: emerg, alert, crit, err, warn, notice, info, debug

nft add rule inet my_filter input log level emerg prefix "EMERGENCY: " drop
nft add rule inet my_filter input log level info prefix "INFO: " accept
```

### Selective Logging

```bash
# Log only SSH attempts
nft add rule inet my_filter input tcp dport 22 ct state new log prefix "SSH: "

# Log only rejected packets
nft add rule inet my_filter input tcp dport 23 log prefix "TELNET REJECT: " reject

# Log with rate limiting (avoid log flooding)
nft add rule inet my_filter input limit rate 5/minute log prefix "DROPPED: " drop
```

### Packet Counters

```bash
# Add counter to rule
nft add rule inet my_filter input tcp dport 80 counter accept

# Named counter
nft add counter inet my_filter http_counter

nft add rule inet my_filter input tcp dport 80 counter name http_counter accept

# View counters
nft list counter inet my_filter http_counter

# Reset counter
nft reset counter inet my_filter http_counter
```

### Traffic Statistics

```bash
# Add counters to chains
nft add chain inet my_filter input { \
    type filter hook input priority 0 \; \
    policy drop \; \
    counter \; \
}

# Per-rule statistics
nft add rule inet my_filter input tcp dport 22 counter name ssh_count accept
nft add rule inet my_filter input tcp dport 80 counter name http_count accept
nft add rule inet my_filter input tcp dport 443 counter name https_count accept

# View all counters
nft list counters inet my_filter
```

### Packet Tracing

```bash
# Enable tracing for specific packets
nft add rule inet my_filter prerouting tcp dport 80 meta nftrace set 1

# Monitor trace events (in another terminal)
nft monitor trace

# Alternative: use nftrace
modprobe nf_tables_trace
nft add rule inet my_filter input ip saddr 192.168.1.100 meta nftrace set 1

# View in kernel log
dmesg | grep -i trace
```

### Live Monitoring

```bash
# Monitor ruleset changes
nft monitor

# Monitor specific table
nft monitor tables

# Monitor trace events
nft monitor trace

# Monitor with JSON output
nft -j monitor
```

## Security Patterns

### Drop Invalid Packets

```bash
# Drop packets with invalid connection state
nft add rule inet my_filter input ct state invalid log prefix "INVALID: " drop

# Drop NULL packets (no flags set)
nft add rule inet my_filter input tcp flags == 0 log prefix "NULL: " drop

# Drop XMAS packets (all flags set)
nft add rule inet my_filter input tcp flags \& \(fin\|syn\|rst\|psh\|ack\|urg\) == \(fin\|syn\|rst\|psh\|ack\|urg\) drop

# Drop FIN without ACK
nft add rule inet my_filter input tcp flags \& \(fin\|ack\) == fin drop

# Drop SYN+FIN
nft add rule inet my_filter input tcp flags \& \(syn\|fin\) == \(syn\|fin\) drop
```

### SYN Flood Protection

```bash
# Method 1: Rate limiting
nft add rule inet my_filter input tcp flags syn tcp flags \& \(fin\|syn\|rst\|ack\) == syn \
    meter syn_meter { ip saddr timeout 10s limit rate 25/second burst 50 packets } accept
nft add rule inet my_filter input tcp flags syn log prefix "SYN FLOOD: " drop

# Method 2: SYN cookies (kernel parameter)
sysctl -w net.ipv4.tcp_syncookies=1

# Method 3: SYNPROXY (requires kernel 3.13+)
nft add rule inet my_filter input tcp dport 80 tcp flags syn notrack

nft add table inet synproxy_table
nft add chain inet synproxy_table prerouting { \
    type filter hook prerouting priority -300 \; \
}

nft add rule inet synproxy_table prerouting tcp dport 80 tcp flags syn \
    synproxy mss 1460 wscale 7 timestamp sack-perm
```

### Brute Force Protection (SSH)

```bash
# Create dynamic blocklist
nft add set inet my_filter ssh_blocklist { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 1h \; \
}

# Block IPs in blocklist
nft add rule inet my_filter input ip saddr @ssh_blocklist drop

# Add to blocklist after 5 attempts in 1 minute
nft add rule inet my_filter input tcp dport 22 ct state new \
    meter ssh_meter { ip saddr timeout 1m limit rate over 5/minute } \
    add @ssh_blocklist { ip saddr timeout 1h } drop

# Allow SSH with rate limit
nft add rule inet my_filter input tcp dport 22 ct state new \
    limit rate 3/minute accept
```

### Port Scan Detection

```bash
# Detect port scans (many SYN to different ports)
nft add set inet my_filter port_scanners { \
    type ipv4_addr \; \
    flags dynamic,timeout \; \
    timeout 1h \; \
}

# Block known scanners
nft add rule inet my_filter input ip saddr @port_scanners drop

# Detect scanning behavior
nft add rule inet my_filter input tcp flags syn ct state new \
    meter scan_meter { ip saddr timeout 10s limit rate over 20/second } \
    add @port_scanners { ip saddr timeout 1h } \
    log prefix "PORT SCAN: " drop
```

### Anti-Spoofing

```bash
# Drop packets from private IPs on WAN interface
nft add rule inet my_filter input iifname "eth0" ip saddr { \
    10.0.0.0/8, \
    172.16.0.0/12, \
    192.168.0.0/16, \
    127.0.0.0/8, \
    169.254.0.0/16, \
    224.0.0.0/4, \
    240.0.0.0/4 \
} log prefix "SPOOFED: " drop

# Drop packets with source address matching our network
nft add rule inet my_filter input iifname "eth0" ip saddr 203.0.113.0/24 drop

# Reverse path filtering (use kernel parameter)
sysctl -w net.ipv4.conf.all.rp_filter=1
```

### GeoIP Blocking

Requires ipset with GeoIP database:

```bash
# Create sets for countries (requires external GeoIP lists)
# Example: blocking traffic from specific countries

# Method 1: Using ipset (legacy)
# ipset create country-cn hash:net
# ipset add country-cn 1.0.1.0/24  # China IP blocks
# iptables -A INPUT -m set --match-set country-cn src -j DROP

# Method 2: Manual nftables sets
nft add set inet my_filter blocked_countries { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add IP ranges for specific countries (example)
nft add element inet my_filter blocked_countries { \
    1.0.1.0/24, \
    1.0.2.0/23 \
}

nft add rule inet my_filter input ip saddr @blocked_countries drop
```

### DDoS Protection

```bash
# Connection tracking table size (kernel parameter)
sysctl -w net.netfilter.nf_conntrack_max=1000000

# Drop flood of new connections
nft add rule inet my_filter input ct state new \
    meter ddos_meter { ip saddr timeout 10s limit rate over 100/second } \
    log prefix "DDoS: " drop

# Limit ICMP
nft add rule inet my_filter input icmp type echo-request \
    limit rate 5/second accept
nft add rule inet my_filter input icmp type echo-request drop

# UDP flood protection
nft add rule inet my_filter input meta l4proto udp \
    meter udp_meter { ip saddr timeout 10s limit rate over 50/second } drop

# Limit fragments
nft add rule inet my_filter input ip frag-off \& 0x1fff != 0 \
    limit rate 10/second accept
nft add rule inet my_filter input ip frag-off \& 0x1fff != 0 drop
```

### Bogon Filtering

```bash
# Create bogon set (IP addresses that should not appear on Internet)
nft add set inet my_filter bogons { \
    type ipv4_addr \; \
    flags interval \; \
}

# Add bogon ranges
nft add element inet my_filter bogons { \
    0.0.0.0/8, \
    10.0.0.0/8, \
    100.64.0.0/10, \
    127.0.0.0/8, \
    169.254.0.0/16, \
    172.16.0.0/12, \
    192.0.0.0/24, \
    192.0.2.0/24, \
    192.168.0.0/16, \
    198.18.0.0/15, \
    198.51.100.0/24, \
    203.0.113.0/24, \
    224.0.0.0/4, \
    240.0.0.0/4 \
}

# Drop bogons on WAN
nft add rule inet my_filter input iifname "eth0" ip saddr @bogons drop
nft add rule inet my_filter input iifname "eth0" ip daddr @bogons drop
```

## Complete Firewall Examples

### Example 1: Basic Server Firewall

```bash
#!/usr/sbin/nft -f

# Flush existing rules
flush ruleset

# Create filter table
table inet filter {
    # Input chain
    chain input {
        type filter hook input priority 0; policy drop;

        # Allow loopback
        iifname "lo" accept

        # Allow established/related
        ct state established,related accept

        # Drop invalid
        ct state invalid drop

        # Allow ICMP
        ip protocol icmp accept
        ip6 nexthdr ipv6-icmp accept

        # Allow SSH (with rate limit)
        tcp dport 22 ct state new limit rate 3/minute accept

        # Allow HTTP/HTTPS
        tcp dport { 80, 443 } ct state new accept

        # Log dropped packets
        limit rate 5/minute log prefix "DROPPED: "
    }

    # Forward chain
    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    # Output chain
    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

### Example 2: Web Server with Rate Limiting

```bash
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    # SSH brute force protection
    set ssh_blocklist {
        type ipv4_addr
        flags dynamic,timeout
        timeout 1h
    }

    # HTTP rate limiting
    set http_ratelimit {
        type ipv4_addr
        size 65535
        flags dynamic
    }

    chain input {
        type filter hook input priority 0; policy drop;

        # Loopback
        iifname "lo" accept

        # Established/related
        ct state established,related accept
        ct state invalid drop

        # ICMP
        ip protocol icmp limit rate 5/second accept
        ip6 nexthdr ipv6-icmp accept

        # SSH protection
        ip saddr @ssh_blocklist drop
        tcp dport 22 ct state new \
            meter ssh_meter { ip saddr timeout 1m limit rate over 5/minute } \
            add @ssh_blocklist { ip saddr timeout 1h } drop
        tcp dport 22 ct state new limit rate 3/minute accept

        # HTTP/HTTPS with rate limiting
        tcp dport { 80, 443 } ct state new \
            meter http_meter { ip saddr timeout 10s limit rate 50/second } accept
        tcp dport { 80, 443 } log prefix "HTTP RATE LIMIT: " drop

        # Log other drops
        limit rate 5/minute log prefix "DROPPED: "
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

### Example 3: Router/Gateway with NAT

```bash
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state established,related accept
        ct state invalid drop

        # Allow management from LAN
        iifname "eth1" tcp dport { 22, 80, 443 } ct state new accept

        # ICMP
        ip protocol icmp limit rate 5/second accept
        ip6 nexthdr ipv6-icmp accept
    }

    chain forward {
        type filter hook forward priority 0; policy drop;

        # Established/related
        ct state established,related accept
        ct state invalid drop

        # LAN to WAN
        iifname "eth1" oifname "eth0" accept

        # Port forwarding (will be handled by NAT)
        iifname "eth0" oifname "eth1" tcp dport { 80, 443 } ct state new accept

        # Log dropped forwards
        limit rate 5/minute log prefix "FWD DROP: "
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table inet nat {
    chain prerouting {
        type nat hook prerouting priority -100;

        # Port forwarding to internal web server
        iifname "eth0" tcp dport 80 dnat to 192.168.1.10
        iifname "eth0" tcp dport 443 dnat to 192.168.1.10
    }

    chain postrouting {
        type nat hook postrouting priority 100;

        # Masquerade LAN to WAN
        oifname "eth0" masquerade
    }
}
```

### Example 4: Multi-Zone Firewall (LAN/DMZ/WAN)

```bash
#!/usr/sbin/nft -f

flush ruleset

# Define variables
define LAN = eth1
define DMZ = eth2
define WAN = eth0
define DMZ_NET = 192.168.100.0/24
define LAN_NET = 192.168.1.0/24

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state established,related accept
        ct state invalid drop

        # Management from LAN only
        iifname $LAN tcp dport { 22, 443 } accept

        # ICMP
        ip protocol icmp limit rate 5/second accept
    }

    chain forward {
        type filter hook forward priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop

        # LAN to anywhere
        iifname $LAN oifname { $DMZ, $WAN } accept

        # DMZ to WAN only (not to LAN)
        iifname $DMZ oifname $WAN accept

        # WAN to DMZ (limited services)
        iifname $WAN oifname $DMZ ip daddr $DMZ_NET tcp dport { 80, 443 } accept

        # Explicit deny WAN to LAN
        iifname $WAN oifname $LAN log prefix "WAN->LAN BLOCKED: " drop

        limit rate 5/minute log prefix "FWD DROP: "
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table inet nat {
    chain prerouting {
        type nat hook prerouting priority -100;

        # DNAT to DMZ web server
        iifname $WAN tcp dport { 80, 443 } dnat to 192.168.100.10
    }

    chain postrouting {
        type nat hook postrouting priority 100;

        # SNAT for LAN
        iifname $LAN oifname $WAN snat to 203.0.113.1

        # SNAT for DMZ
        iifname $DMZ oifname $WAN snat to 203.0.113.2
    }
}
```

### Example 5: IPv4/IPv6 Dual-Stack Firewall

```bash
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Loopback
        iifname "lo" accept

        # Established/related
        ct state established,related accept
        ct state invalid drop

        # ICMP (IPv4)
        ip protocol icmp accept

        # ICMPv6 (essential for IPv6)
        icmpv6 type {
            destination-unreachable,
            packet-too-big,
            time-exceeded,
            parameter-problem,
            echo-request,
            echo-reply,
            nd-neighbor-solicit,
            nd-neighbor-advert,
            nd-router-advert,
            nd-router-solicit
        } accept

        # SSH (both IPv4 and IPv6)
        tcp dport 22 ct state new limit rate 3/minute accept

        # HTTP/HTTPS
        tcp dport { 80, 443 } ct state new accept

        # DHCPv6 client
        ip6 daddr fe80::/64 udp dport 546 accept

        # Log drops
        limit rate 5/minute log prefix "DROPPED: "
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

### Example 6: Container Host Firewall (Docker)

```bash
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state established,related accept
        ct state invalid drop

        # Allow from docker network
        iifname "docker0" accept

        # SSH
        tcp dport 22 ct state new limit rate 3/minute accept

        # Container exposed ports
        tcp dport { 80, 443, 8080 } ct state new accept

        ip protocol icmp limit rate 5/second accept
    }

    chain forward {
        type filter hook forward priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop

        # Docker containers to internet
        iifname "docker0" oifname "eth0" accept

        # Internet to docker containers (specific ports)
        iifname "eth0" oifname "docker0" tcp dport { 80, 443 } ct state new accept
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table inet nat {
    chain prerouting {
        type nat hook prerouting priority -100;

        # Port forwarding to containers
        tcp dport 8080 dnat to 172.17.0.2:80
    }

    chain postrouting {
        type nat hook postrouting priority 100;

        # Docker NAT
        oifname "eth0" ip saddr 172.17.0.0/16 masquerade
    }
}
```

## Migration from iptables

### Translation Tools

```bash
# Translate single iptables command
iptables-translate -A INPUT -p tcp --dport 80 -j ACCEPT
# Output: nft add rule ip filter INPUT tcp dport 80 counter accept

# Translate iptables-save output
iptables-save > iptables.rules
iptables-restore-translate -f iptables.rules > nftables.conf

# For IPv6
ip6tables-save > ip6tables.rules
ip6tables-restore-translate -f ip6tables.rules >> nftables.conf
```

### Common Pattern Conversions

#### Basic Rule Translation

```bash
# iptables
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# nftables
nft add rule inet filter input tcp dport 22 accept
```

#### Multiple Ports

```bash
# iptables
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT

# nftables
nft add rule inet filter input tcp dport { 80, 443 } accept
```

#### Port Range

```bash
# iptables
iptables -A INPUT -p tcp --dport 1024:65535 -j ACCEPT

# nftables
nft add rule inet filter input tcp dport 1024-65535 accept
```

#### Connection State

```bash
# iptables
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# nftables
nft add rule inet filter input ct state established,related accept
```

#### IP Address

```bash
# iptables
iptables -A INPUT -s 192.168.1.0/24 -j ACCEPT

# nftables
nft add rule inet filter input ip saddr 192.168.1.0/24 accept
```

#### Interface

```bash
# iptables
iptables -A INPUT -i eth0 -j ACCEPT

# nftables
nft add rule inet filter input iifname "eth0" accept
```

#### SNAT/Masquerade

```bash
# iptables
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# nftables
nft add rule inet nat postrouting oifname "eth0" masquerade
```

#### DNAT/Port Forwarding

```bash
# iptables
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:80

# nftables
nft add rule inet nat prerouting iifname "eth0" tcp dport 80 dnat to 192.168.1.10:80
```

#### Rate Limiting

```bash
# iptables
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 5/sec -j ACCEPT

# nftables
nft add rule inet filter input icmp type echo-request limit rate 5/second accept
```

#### LOG Target

```bash
# iptables
iptables -A INPUT -j LOG --log-prefix "DROPPED: " --log-level 4

# nftables
nft add rule inet filter input log prefix \"DROPPED: \" level warn
```

#### Recent Module (Connection Tracking)

```bash
# iptables (SSH brute force protection)
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP

# nftables
nft add rule inet filter input tcp dport 22 ct state new \
    meter ssh_meter { ip saddr timeout 1m limit rate over 3/minute } drop
```

#### ipset

```bash
# iptables with ipset
ipset create myset hash:ip
ipset add myset 192.168.1.1
iptables -A INPUT -m set --match-set myset src -j ACCEPT

# nftables
nft add set inet filter myset { type ipv4_addr \; }
nft add element inet filter myset { 192.168.1.1 }
nft add rule inet filter input ip saddr @myset accept
```

### Side-by-Side Examples

#### Example 1: Basic Firewall

```bash
# iptables
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# nftables
nft add table inet filter
nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }
nft add rule inet filter input iifname "lo" accept
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input tcp dport 22 accept
nft add rule inet filter input tcp dport 80 accept
```

#### Example 2: NAT Router

```bash
# iptables
echo 1 > /proc/sys/net/ipv4/ip_forward
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
iptables -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT

# nftables
echo 1 > /proc/sys/net/ipv4/ip_forward
nft add table inet nat
nft add chain inet nat postrouting { type nat hook postrouting priority 100 \; }
nft add rule inet nat postrouting oifname "eth0" masquerade
nft add table inet filter
nft add chain inet filter forward { type filter hook forward priority 0 \; }
nft add rule inet filter forward iifname "eth1" oifname "eth0" accept
nft add rule inet filter forward ct state established,related accept
```

### Migration Strategies

#### 1. Big Bang Migration

```bash
# Stop iptables
systemctl stop iptables
systemctl disable iptables

# Convert rules
iptables-save > /tmp/iptables.rules
iptables-restore-translate -f /tmp/iptables.rules > /etc/nftables.conf

# Review and edit /etc/nftables.conf
# Combine ip and ip6 tables into inet tables

# Load nftables
nft -f /etc/nftables.conf

# Enable service
systemctl enable nftables
systemctl start nftables
```

#### 2. Gradual Migration

```bash
# Keep iptables running for critical services
# Add nftables rules for new services

# Example: Keep iptables for SSH, add nftables for HTTP
# iptables handles SSH (port 22)
# nftables handles HTTP (port 80)

# Eventually migrate all rules to nftables
```

### Coexistence Considerations

- iptables and nftables can coexist but may cause confusion
- Both use the same Netfilter hooks
- Rules are evaluated in priority order (nftables priorities vs iptables chains)
- Recommended: use one framework only
- Check which framework is active:

```bash
# Check iptables
iptables -L -n

# Check nftables
nft list ruleset

# Remove iptables rules
iptables -F
ip6tables -F
iptables -X
ip6tables -X
```

## Scripting and Automation

### nft Scripting Language

```bash
#!/usr/sbin/nft -f

# Comments start with #

# Flush existing ruleset
flush ruleset

# Define variables
define WAN = eth0
define LAN = eth1
define SSH_PORT = 22
define WEB_PORTS = { 80, 443 }

# Tables and chains
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        # Use variables
        iifname $LAN accept
        tcp dport $SSH_PORT accept
        tcp dport $WEB_PORTS accept
    }
}
```

### Include Files

```bash
# Main file: /etc/nftables.conf
#!/usr/sbin/nft -f

flush ruleset

# Include definitions
include "/etc/nftables.d/defines.nft"

# Include tables
include "/etc/nftables.d/filter.nft"
include "/etc/nftables.d/nat.nft"
```

```bash
# File: /etc/nftables.d/defines.nft
define WAN = eth0
define LAN = eth1
define DMZ = eth2
```

```bash
# File: /etc/nftables.d/filter.nft
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;
        # rules...
    }
}
```

### Atomic Ruleset Replacement

```bash
#!/bin/bash

# Generate new ruleset
cat > /tmp/nftables.conf <<EOF
flush ruleset
table inet filter {
    chain input {
        type filter hook input priority 0; policy accept;
        # New rules...
    }
}
EOF

# Test syntax
if nft -c -f /tmp/nftables.conf; then
    echo "Syntax OK"
    # Apply atomically
    nft -f /tmp/nftables.conf
    echo "Rules applied"
else
    echo "Syntax error!"
    exit 1
fi
```

### Backup and Restore Scripts

```bash
#!/bin/bash
# backup-nftables.sh

BACKUP_DIR="/var/backups/nftables"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup current ruleset
nft list ruleset > "$BACKUP_DIR/nftables_$DATE.conf"

# Keep only last 10 backups
ls -t "$BACKUP_DIR"/nftables_*.conf | tail -n +11 | xargs -r rm

echo "Backup saved: $BACKUP_DIR/nftables_$DATE.conf"
```

```bash
#!/bin/bash
# restore-nftables.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "File not found: $BACKUP_FILE"
    exit 1
fi

# Test syntax
if nft -c -f "$BACKUP_FILE"; then
    echo "Restoring from: $BACKUP_FILE"
    nft -f "$BACKUP_FILE"
    echo "Rules restored"
else
    echo "Invalid ruleset file"
    exit 1
fi
```

### systemd Integration

```bash
# /etc/systemd/system/nftables-custom.service
[Unit]
Description=Custom nftables firewall
After=network-pre.target
Before=network.target
Wants=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/nft -f /etc/nftables/custom.conf
ExecReload=/usr/sbin/nft -f /etc/nftables/custom.conf
ExecStop=/usr/sbin/nft flush ruleset
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Dynamic Rule Management Script

```bash
#!/bin/bash
# manage-blocklist.sh

BLOCKLIST_SET="blocklist"
TABLE="inet filter"

case "$1" in
    add)
        nft add element $TABLE $BLOCKLIST_SET { $2 }
        echo "Added $2 to blocklist"
        ;;
    del)
        nft delete element $TABLE $BLOCKLIST_SET { $2 }
        echo "Removed $2 from blocklist"
        ;;
    list)
        nft list set $TABLE $BLOCKLIST_SET
        ;;
    flush)
        nft flush set $TABLE $BLOCKLIST_SET
        echo "Blocklist cleared"
        ;;
    *)
        echo "Usage: $0 {add|del|list|flush} [IP]"
        exit 1
        ;;
esac
```

## Debugging and Troubleshooting

### Common Errors

#### Error: "No such file or directory"

```bash
# Issue: nf_tables module not loaded
# Solution:
modprobe nf_tables
```

#### Error: "Operation not supported"

```bash
# Issue: Kernel too old or feature not compiled
# Check kernel version:
uname -r  # Need 3.13+

# Check if nftables is compiled:
grep NFT /boot/config-$(uname -r)
```

#### Error: "Could not process rule: No such file or directory"

```bash
# Issue: Set or chain doesn't exist
# List existing objects:
nft list tables
nft list sets
nft list chains

# Create missing objects first
```

#### Syntax Errors

```bash
# Test configuration without applying:
nft -c -f /etc/nftables.conf

# Common syntax issues:
# - Missing semicolons after chain properties
# - Wrong quotes (use double quotes for strings)
# - Incorrect operator (use '==' not '=')
```

### Rule Testing

```bash
# Method 1: Test in separate table
nft add table inet test
nft add chain inet test input { type filter hook input priority 0 \; }
nft add rule inet test input tcp dport 80 accept

# Test, then delete
nft delete table inet test

# Method 2: Check syntax only
nft -c add rule inet filter input tcp dport 80 accept

# Method 3: Use counters to verify matching
nft add rule inet filter input tcp dport 80 counter accept
# Generate traffic, then check:
nft list chain inet filter input
```

### Packet Tracing

```bash
# Enable tracing for specific traffic
nft add rule inet filter prerouting ip saddr 192.168.1.100 meta nftrace set 1

# Monitor traces
nft monitor trace

# Example output shows packet path through chains

# Disable tracing
nft delete rule inet filter prerouting handle <N>
```

### Listing Rules with Handles

```bash
# Show handles (needed for deletion)
nft --handle list ruleset

# List specific table with handles
nft --handle list table inet filter

# Delete rule by handle
nft delete rule inet filter input handle 5
```

### Counters and Statistics

```bash
# List rules with packet/byte counters
nft list table inet filter

# Reset counters
nft reset rules inet filter

# Reset specific counter
nft reset counter inet filter my_counter
```

### Connection Tracking

```bash
# View current connections
conntrack -L

# Count connections
conntrack -C

# Delete specific connection
conntrack -D -s 192.168.1.100

# Monitor new connections
conntrack -E
```

### Kernel Messages

```bash
# Check kernel log for nftables messages
dmesg | grep nf_tables

# Monitor live
journalctl -f -k | grep nft

# Check for errors
dmesg | grep -i error | grep nf
```

### Performance Profiling

```bash
# Check connection tracking table size
cat /proc/sys/net/netfilter/nf_conntrack_count
cat /proc/sys/net/netfilter/nf_conntrack_max

# Adjust if needed
sysctl -w net.netfilter.nf_conntrack_max=1000000

# Monitor CPU usage
top -p $(pgrep nft)

# Check memory usage
cat /proc/net/stat/nf_conntrack
```

### Debug Logging

```bash
# Enable nftables debug in kernel (requires debug build)
echo 'module nf_tables +p' > /sys/kernel/debug/dynamic_debug/control

# View debug output
dmesg -w | grep nf_tables
```

## Performance Optimization

### Rule Ordering

```bash
# GOOD: Most common rules first
nft add rule inet filter input ct state established,related accept  # Most traffic
nft add rule inet filter input tcp dport 80 accept                  # Common
nft add rule inet filter input tcp dport 22 accept                  # Less common

# BAD: Rare rules first
nft add rule inet filter input tcp dport 23 drop                    # Rarely matched
nft add rule inet filter input ct state established,related accept  # Most traffic
```

### Using Sets vs Multiple Rules

```bash
# INEFFICIENT: Multiple rules
nft add rule inet filter input tcp dport 80 accept
nft add rule inet filter input tcp dport 443 accept
nft add rule inet filter input tcp dport 8080 accept
nft add rule inet filter input tcp dport 8443 accept

# EFFICIENT: Single rule with set
nft add rule inet filter input tcp dport { 80, 443, 8080, 8443 } accept

# Even better: Named set (optimized data structure)
nft add set inet filter web_ports { type inet_service \; }
nft add element inet filter web_ports { 80, 443, 8080, 8443 }
nft add rule inet filter input tcp dport @web_ports accept
```

### Verdict Maps for Performance

```bash
# Instead of multiple rules with jumps:
# SLOW:
nft add rule inet filter input tcp dport 80 jump http_chain
nft add rule inet filter input tcp dport 443 jump https_chain
nft add rule inet filter input tcp dport 22 jump ssh_chain

# FAST: Use verdict map
nft add map inet filter service_map { type inet_service : verdict \; }
nft add element inet filter service_map { \
    80 : jump http_chain, \
    443 : jump https_chain, \
    22 : jump ssh_chain \
}
nft add rule inet filter input tcp dport vmap @service_map
```

### Early Dropping

```bash
# Drop invalid packets early (before expensive checks)
nft add chain inet filter input { type filter hook input priority -200 \; }
nft add rule inet filter input ct state invalid drop

# Main filter chain
nft add chain inet filter input_main { type filter hook input priority 0 \; }
# ... other rules
```

### Connection Tracking Optimization

```bash
# Increase connection tracking table size for high traffic
sysctl -w net.netfilter.nf_conntrack_max=1000000

# Reduce timeout for specific protocols
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_established=3600

# Disable tracking for specific traffic (stateless)
nft add rule inet raw prerouting tcp dport 80 notrack
nft add rule inet raw output tcp sport 80 notrack
```

### Hardware Offload

```bash
# For NICs with flow offload support (kernel 4.16+)
# Create flowtable for offloading
nft add flowtable inet filter f {
    hook ingress priority 0;
    devices = { eth0, eth1 };
}

# Offload established connections
nft add rule inet filter forward \
    ct state established flow add @f
```

### Memory Optimization

```bash
# Limit set size to prevent memory exhaustion
nft add set inet filter limited_set { \
    type ipv4_addr \; \
    size 10000 \; \
    flags dynamic,timeout \; \
}

# Monitor memory usage
cat /proc/slabinfo | grep nf_conntrack
```

### Benchmarking

```bash
# Benchmark rule lookup performance
# Generate traffic and measure:

# Before optimization
time nft list ruleset > /dev/null

# Add rules
# ... add thousands of rules

# After
time nft list ruleset > /dev/null

# Use iperf3 for throughput testing
# Server:
iperf3 -s

# Client:
iperf3 -c <server_ip> -t 30 -P 10
```

## Integration with Other Tools

### firewalld Backend

firewalld can use nftables as backend:

```bash
# /etc/firewalld/firewalld.conf
FirewallBackend=nftables

# Restart firewalld
systemctl restart firewalld

# Check backend
firewall-cmd --get-backend

# firewalld creates its own nftables tables
nft list tables
```

### Docker Integration

Docker uses iptables by default, but can coexist:

```bash
# Docker creates rules in iptables
# Your host firewall uses nftables

# Example: Allow Docker while using nftables for host
# nftables for host protection
nft add rule inet filter input iifname "docker0" accept
nft add rule inet filter forward iifname "docker0" accept

# Let Docker manage its own iptables rules
# They operate in different priority ranges
```

### fail2ban with nftables

Configure fail2ban to use nftables:

```bash
# /etc/fail2ban/jail.local
[DEFAULT]
banaction = nftables-multiport
banaction_allports = nftables-allports

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 5
bantime = 3600
```

Create nftables action:

```bash
# /etc/fail2ban/action.d/nftables-multiport.conf
[Definition]
actionstart = nft add table inet fail2ban
              nft add chain inet fail2ban input { type filter hook input priority -1 \; }

actionban = nft add rule inet fail2ban input ip saddr <ip> drop

actionunban = nft delete rule inet fail2ban input ip saddr <ip> drop
```

### libvirt Integration

libvirt can use nftables for VM networking:

```bash
# /etc/libvirt/network.conf
firewall_backend = "nftables"

# Restart libvirt
systemctl restart libvirtd

# libvirt creates its own tables
nft list table inet libvirt
```

### NetworkManager Dispatcher

Run nftables scripts on network events:

```bash
# /etc/NetworkManager/dispatcher.d/10-nftables
#!/bin/bash

INTERFACE=$1
ACTION=$2

case "$ACTION" in
    up)
        # Interface came up
        nft add rule inet filter input iifname "$INTERFACE" accept
        ;;
    down)
        # Interface went down
        nft delete rule inet filter input iifname "$INTERFACE" accept 2>/dev/null
        ;;
esac
```

### Kubernetes

Example nftables rules for Kubernetes nodes:

```bash
#!/usr/sbin/nft -f

flush ruleset

table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state established,related accept

        # Kubernetes API server
        tcp dport 6443 accept

        # Kubelet API
        tcp dport 10250 accept

        # NodePort services
        tcp dport 30000-32767 accept

        # CNI plugin (Calico, Flannel, etc.)
        # Adjust based on your CNI
        iifname "cni0" accept
        iifname "flannel.1" accept
    }

    chain forward {
        type filter hook forward priority 0; policy accept;

        # Allow pod-to-pod communication
        iifname "cni0" oifname "cni0" accept
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
```

## Best Practices

### Security Best Practices

1. **Default Deny Policy**
```bash
# Always use drop as default policy for input/forward
nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
```

2. **Drop Invalid Packets Early**
```bash
nft add rule inet filter input ct state invalid drop
```

3. **Rate Limit Everything**
```bash
# Especially management services
nft add rule inet filter input tcp dport 22 ct state new limit rate 3/minute accept
```

4. **Use Connection Tracking**
```bash
# Stateful filtering is more secure
nft add rule inet filter input ct state established,related accept
```

5. **Log Suspicious Activity**
```bash
# But rate limit logs to prevent flooding
nft add rule inet filter input limit rate 5/minute log prefix "SUSPICIOUS: " drop
```

### Rule Organization

1. **Use Includes for Modularity**
```bash
# /etc/nftables.conf
include "/etc/nftables.d/*.nft"
```

2. **Group Related Rules in Chains**
```bash
chain ssh_rules {
    # All SSH-related rules
}
chain web_rules {
    # All web-related rules
}
```

3. **Use Named Sets for Maintainability**
```bash
nft add set inet filter allowed_ips { type ipv4_addr \; }
# Easier to update than editing rules
```

### Documentation Practices

1. **Comment Your Rules**
```bash
# In nft script files:
# Allow SSH from management network
nft add rule inet filter input ip saddr 10.0.0.0/8 tcp dport 22 accept
```

2. **Keep Change Log**
```bash
# /etc/nftables.d/CHANGELOG
# 2025-01-15: Added rate limiting for SSH
# 2025-01-14: Opened port 8080 for new service
```

3. **Document Network Topology**
```bash
# Define and document interfaces
define WAN = eth0  # Internet connection
define LAN = eth1  # Internal network 192.168.1.0/24
define DMZ = eth2  # DMZ network 192.168.100.0/24
```

### Testing Practices

1. **Always Test Syntax Before Applying**
```bash
nft -c -f /etc/nftables.conf
```

2. **Test New Rules in Isolation**
```bash
# Create test table
nft add table inet test
# Test rules
# Delete when done
nft delete table inet test
```

3. **Keep a Rollback Plan**
```bash
# Backup before changes
nft list ruleset > /tmp/nftables.backup

# Auto-rollback script
nft -f /etc/nftables.conf && sleep 60 && nft -f /tmp/nftables.backup &
# If connection is lost, rules rollback after 60 seconds
```

4. **Use Counters for Verification**
```bash
# Add counters to verify rules are matching
nft add rule inet filter input tcp dport 80 counter accept
# Check counter values
nft list chain inet filter input
```

### Backup Strategies

1. **Automated Backups**
```bash
# Daily cron job
0 2 * * * nft list ruleset > /var/backups/nftables/$(date +\%Y\%m\%d).conf
```

2. **Version Control**
```bash
cd /etc/nftables.d
git init
git add *.nft
git commit -m "Initial firewall configuration"
```

3. **Configuration Management**
```bash
# Use Ansible, Puppet, or Chef to manage nftables
# Example Ansible task:
# - name: Deploy nftables configuration
#   template:
#     src: nftables.conf.j2
#     dest: /etc/nftables.conf
#   notify: reload nftables
```

### Monitoring Practices

1. **Monitor Rule Counters**
```bash
# Script to check counters
watch -n 5 'nft list ruleset | grep counter'
```

2. **Log Analysis**
```bash
# Analyze dropped packets
journalctl -k | grep "DROPPED:"

# Count by source IP
journalctl -k | grep "DROPPED:" | awk '{print $NF}' | sort | uniq -c | sort -rn
```

3. **Alert on Anomalies**
```bash
# Monitor for brute force
journalctl -f -k | grep "SSH BLOCK" | \
while read line; do
    echo "Alert: SSH brute force detected" | mail -s "Security Alert" admin@example.com
done
```

### Performance Monitoring

```bash
# Connection tracking usage
watch -n 1 'cat /proc/sys/net/netfilter/nf_conntrack_count'

# Memory usage
watch -n 5 'cat /proc/slabinfo | grep nf_conntrack'

# Network throughput
iftop -i eth0
```

## References and Resources

### Official Documentation

- **nftables Wiki**: https://wiki.nftables.org/
- **Netfilter Project**: https://www.netfilter.org/
- **Kernel Documentation**: https://www.kernel.org/doc/Documentation/networking/nftables.txt

### Man Pages

```bash
man nft
man libnftables
man libnftables-json
```

### RFCs

- RFC 3947: Negotiation of NAT-Traversal in the IKE
- RFC 4787: NAT Behavioral Requirements for UDP
- RFC 5382: NAT Behavioral Requirements for TCP
- RFC 7857: Updates to Network Address Translation (NAT) Behavioral Requirements

### Community Resources

- **Arch Linux Wiki**: https://wiki.archlinux.org/title/Nftables
- **Debian Wiki**: https://wiki.debian.org/nftables
- **Red Hat Documentation**: https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_and_managing_networking/getting-started-with-nftables_configuring-and-managing-networking

### Tools

- `nft` - nftables administration tool
- `nftables` - systemd service
- `iptables-translate` - Convert iptables rules to nftables
- `iptables-restore-translate` - Convert iptables-save output to nftables

### Comparison Resources

- **nftables vs iptables**: https://wiki.nftables.org/wiki-nftables/index.php/Moving_from_iptables_to_nftables
- **iptables-translate**: Built-in tool for migration

## ELI10 (Explain Like I'm 10)

### What is nftables?

Imagine your computer is a castle, and nftables is the guard at the gate. The guard decides who can come in and who has to stay out based on rules you give them.

### How Does It Work?

**Tables**: Think of these as different guardhouses. One for the main gate (filter), one for disguises (NAT), etc.

**Chains**: These are lists of questions the guard asks. "Are you a friend?" "Do you have the password?" "Are you from the village?"

**Rules**: These are specific instructions. "If someone is wearing a red hat, let them in." "If someone doesn't know the password, send them away."

### Why is nftables Better Than iptables?

**iptables** is like having different guards who don't talk to each other:
- One guard for people (IPv4)
- Another guard for elves (IPv6)
- Another guard for carriages (bridges)

**nftables** is like having one smart guard who handles everything and speaks multiple languages!

### Simple Example

```bash
# Create a guardhouse (table)
nft add table inet my_castle

# Create a question list for people coming in (chain)
nft add chain inet my_castle entrance { \
    type filter hook input priority 0 \; \
    policy drop \; \
}

# Let friends in (rule)
nft add rule inet my_castle entrance ip saddr 192.168.1.0/24 accept

# Let people with password in (rule)
nft add rule inet my_castle entrance tcp dport 22 accept
```

### Sets and Maps

**Sets** are like VIP lists:
```bash
# Create VIP list
nft add set inet my_castle vip_list { type ipv4_addr \; }

# Add people to list
nft add element inet my_castle vip_list { 192.168.1.1, 192.168.1.2 }

# VIPs can enter
nft add rule inet my_castle entrance ip saddr @vip_list accept
```

**Maps** are like giving different directions based on who you are:
```bash
# If you're going to room 80, go to building A
# If you're going to room 443, go to building B
nft add map inet my_castle directions { type inet_service : ipv4_addr \; }
nft add element inet my_castle directions { 80 : 192.168.1.10, 443 : 192.168.1.11 }
```

### NAT (Network Address Translation)

NAT is like having a disguise room. When people from inside the castle go outside, they all wear the same uniform so outsiders can't tell them apart. When outsiders want to visit someone inside, the guard knows who they really want to see and directs them correctly.

**Masquerading** (SNAT): Everyone leaving wears the same disguise
```bash
nft add rule inet nat postrouting oifname "eth0" masquerade
```

**Port Forwarding** (DNAT): Visitors asking for room 80 get sent to person 192.168.1.10
```bash
nft add rule inet nat prerouting tcp dport 80 dnat to 192.168.1.10
```

### Connection Tracking

This is like the guard remembering conversations:
- "Oh, you're the person I was talking to earlier, come in!" (established)
- "You're bringing the package for someone I let in earlier, okay!" (related)
- "I've never seen you before, state your business!" (new)

### Rate Limiting

This prevents one person from knocking on the door too many times:
```bash
# Only allow 5 knocks per second
nft add rule inet my_castle entrance limit rate 5/second accept
```

If someone knocks more than that, the guard gets suspicious and might block them!

### Remember

- **Tables** = Guardhouses
- **Chains** = Lists of questions
- **Rules** = Specific instructions
- **Sets** = Lists of allowed/blocked things
- **Maps** = Direction guides
- **NAT** = Disguise room
- **Connection Tracking** = Guard's memory

nftables is just a very organized way of telling your computer who to trust and who to keep out!