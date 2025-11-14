# Netfilter

Netfilter is a framework provided by the Linux kernel for packet filtering, network address translation (NAT), and other packet mangling. It allows system administrators to define rules for how packets should be handled by the kernel. Netfilter is the foundation upon which tools like iptables and nftables are built.

## Table of Contents

- [Architecture](#architecture)
- [Tables and Chains](#tables-and-chains)
- [Packet Flow](#packet-flow)
- [Basic Operations](#basic-operations)
- [Filtering Patterns](#filtering-patterns)
- [NAT Patterns](#nat-patterns)
- [Advanced Filtering](#advanced-filtering)
- [Connection Tracking](#connection-tracking)
- [Chain Management](#chain-management)
- [Common Use Cases](#common-use-cases)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Modern Alternatives](#modern-alternatives)

## Architecture

### Netfilter vs iptables/nftables

- **Netfilter**: The kernel-space framework providing hooks in the network stack
- **iptables**: User-space utility to configure IPv4 packet filtering rules (legacy)
- **ip6tables**: User-space utility for IPv6 packet filtering
- **nftables**: Modern replacement for iptables, offering better performance and syntax

### Netfilter Hooks

Netfilter provides five hook points in the kernel network stack where packets can be intercepted:

1. **NF_IP_PRE_ROUTING (PREROUTING)**: Triggered before routing decision
   - First hook after packet reception
   - Used for DNAT, traffic redirection

2. **NF_IP_LOCAL_IN (INPUT)**: For packets destined to local system
   - After routing decision, before local delivery
   - Used for incoming firewall rules

3. **NF_IP_FORWARD (FORWARD)**: For packets being routed through the system
   - For packets not destined for local system
   - Used in routers and gateways

4. **NF_IP_LOCAL_OUT (OUTPUT)**: For locally generated packets
   - Before routing decision for local packets
   - Used for outgoing firewall rules

5. **NF_IP_POST_ROUTING (POSTROUTING)**: After routing decision
   - Last hook before packet transmission
   - Used for SNAT and masquerading

## Tables and Chains

Netfilter organizes rules into tables, each serving a specific purpose. Each table contains chains that correspond to netfilter hooks.

### Filter Table

The default table for packet filtering.

**Chains**: INPUT, FORWARD, OUTPUT

**Purpose**: Decide whether to allow or deny packets

```bash
# View filter table
iptables -t filter -L -n -v
# or simply (filter is default)
iptables -L -n -v
```

### NAT Table

Used for Network Address Translation.

**Chains**: PREROUTING, OUTPUT, POSTROUTING

**Purpose**: Modify source or destination addresses

```bash
# View NAT table
iptables -t nat -L -n -v
```

**Note**: NAT table does not have INPUT or FORWARD chains because NAT occurs before routing (PREROUTING) or after routing (POSTROUTING).

### Mangle Table

Used for specialized packet alteration.

**Chains**: PREROUTING, INPUT, FORWARD, OUTPUT, POSTROUTING (all 5)

**Purpose**: Modify IP headers (TTL, TOS, MARK)

```bash
# View mangle table
iptables -t mangle -L -n -v
```

### Raw Table

Used for configuration exemptions from connection tracking.

**Chains**: PREROUTING, OUTPUT

**Purpose**: Mark packets to bypass connection tracking (performance optimization)

```bash
# View raw table
iptables -t raw -L -n -v
```

### Security Table

Used for Mandatory Access Control (MAC) networking rules.

**Chains**: INPUT, OUTPUT, FORWARD

**Purpose**: SELinux packet labeling

```bash
# View security table
iptables -t security -L -n -v
```

## Packet Flow

Understanding packet flow through netfilter is crucial for effective rule creation:

```
                               XXXXXXXXXXXXXXXXXX
                             XXX     Network    XXX
                               XXXXXXXXXXXXXXXXXX
                                       +
                                       |
                                       v
 +-------------+              +------------------+
 |table: filter| <---+        | table: raw       |
 |chain: INPUT |     |        | chain: PREROUTING|
 +-----+-------+     |        +--------+---------+
       |             |                 |
       v             |                 v
 [local process]     |        +------------------+
       |             |        | table: mangle    |
       |             |        | chain: PREROUTING|
       +-------------+        +--------+---------+
       |                               |
       v                               v
+--------------+              +------------------+
|table: filter |              | table: nat       |
|chain: OUTPUT |              | chain: PREROUTING|
+------+-------+              +--------+---------+
       |                               |
       v                               v
+---------------+            +--------------------+
|table: raw     |            | routing decision   |
|chain: OUTPUT  |            +--------+-----------+
+------+--------+                     |
       |                              |
       v                              +---------+---------+
+---------------+                     |                   |
|table: mangle  |                     v                   v
|chain: OUTPUT  |            +------------------+   +-------------+
+------+--------+            | table: filter    |   | table:filter|
       |                     | chain: FORWARD   |   | chain: INPUT|
       v                     +--------+---------+   +------+------+
+---------------+                     |                    |
|table: nat     |                     |                    v
|chain: OUTPUT  |                     |             [local process]
+------+--------+                     |
       |                              v
       v                     +-------------------+
+---------------+            | table: mangle     |
|table: mangle  |            | chain: FORWARD    |
|chain:POSTROUTE|            +--------+----------+
+------+--------+                     |
       |                              v
       v                     +-------------------+
+---------------+            | table: nat        |
|table: nat     |            | chain: POSTROUTING|
+------+--------+            +--------+----------+
       |                              |
       v                              v
       +------------------------------+
                     |
                     v
            +------------------+
            | table: mangle    |
            | chain:POSTROUTING|
            +--------+---------+
                     |
                     v
              XXXXXXXXXXXXXXXXXX
            XXX    Network     XXX
              XXXXXXXXXXXXXXXXXX
```

**Key Flow Points**:
- Incoming packets: Raw PREROUTING → Mangle PREROUTING → NAT PREROUTING → Routing Decision
- To local process: Filter INPUT → Local Process
- From local process: Raw OUTPUT → Mangle OUTPUT → NAT OUTPUT → Filter OUTPUT → Routing
- Forwarded: Filter FORWARD → Mangle FORWARD
- All outgoing: NAT POSTROUTING → Mangle POSTROUTING → Network

## Basic Operations

### Viewing Rules

```bash
# List all rules in filter table
iptables -L

# List with line numbers
iptables -L --line-numbers

# List with numeric output (no DNS resolution)
iptables -L -n

# List with verbose output (packet/byte counters)
iptables -L -v

# List specific chain
iptables -L INPUT

# List specific table
iptables -t nat -L

# Combine options for best output
iptables -t filter -L -n -v --line-numbers
```

### Adding Rules

```bash
# Append rule to end of chain (-A)
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Insert rule at specific position (-I)
iptables -I INPUT 1 -p tcp --dport 22 -j ACCEPT

# Insert at beginning (default position 1)
iptables -I INPUT -p tcp --dport 443 -j ACCEPT
```

### Deleting Rules

```bash
# Delete by specification
iptables -D INPUT -p tcp --dport 80 -j ACCEPT

# Delete by line number
iptables -D INPUT 3

# Flush all rules in chain
iptables -F INPUT

# Flush all rules in table
iptables -F

# Flush all rules in all tables
iptables -F
iptables -t nat -F
iptables -t mangle -F
iptables -t raw -F
```

### Modifying Rules

```bash
# Replace rule at specific line
iptables -R INPUT 2 -p tcp --dport 8080 -j ACCEPT

# Change default policy
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT
```

### Saving and Restoring Rules

```bash
# Save current rules (Debian/Ubuntu)
iptables-save > /etc/iptables/rules.v4
ip6tables-save > /etc/iptables/rules.v6

# Save current rules (RedHat/CentOS)
service iptables save

# Restore rules
iptables-restore < /etc/iptables/rules.v4

# Restore with testing (flush on error)
iptables-restore --test < /etc/iptables/rules.v4
```

### Resetting Firewall

```bash
# Reset everything to defaults
iptables -F                    # Flush all rules
iptables -X                    # Delete all custom chains
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
iptables -P INPUT ACCEPT       # Set default policies
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT
```

## Filtering Patterns

### Port-Based Filtering

```bash
# Allow specific TCP port
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow specific UDP port
iptables -A INPUT -p udp --dport 53 -j ACCEPT

# Allow port range
iptables -A INPUT -p tcp --dport 6000:6010 -j ACCEPT

# Allow multiple ports (requires multiport module)
iptables -A INPUT -p tcp -m multiport --dports 80,443,8080 -j ACCEPT

# Block specific port
iptables -A INPUT -p tcp --dport 23 -j DROP

# Allow source port
iptables -A INPUT -p tcp --sport 1024:65535 -j ACCEPT
```

### Protocol-Based Filtering

```bash
# Allow all TCP
iptables -A INPUT -p tcp -j ACCEPT

# Allow all UDP
iptables -A INPUT -p udp -j ACCEPT

# Allow ICMP (ping)
iptables -A INPUT -p icmp -j ACCEPT

# Allow specific ICMP types
iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-reply -j ACCEPT

# Block ICMP
iptables -A INPUT -p icmp -j DROP

# Allow GRE (VPN protocol)
iptables -A INPUT -p gre -j ACCEPT

# Allow ESP (IPsec)
iptables -A INPUT -p esp -j ACCEPT
```

### IP Address Filtering

```bash
# Allow from specific IP
iptables -A INPUT -s 192.168.1.100 -j ACCEPT

# Allow from subnet
iptables -A INPUT -s 192.168.1.0/24 -j ACCEPT

# Block specific IP
iptables -A INPUT -s 10.0.0.50 -j DROP

# Block subnet
iptables -A INPUT -s 172.16.0.0/16 -j DROP

# Allow to specific destination
iptables -A OUTPUT -d 8.8.8.8 -j ACCEPT

# Multiple source IPs (iprange module)
iptables -A INPUT -m iprange --src-range 192.168.1.100-192.168.1.200 -j ACCEPT

# Invert match (all except)
iptables -A INPUT ! -s 192.168.1.0/24 -j DROP
```

### Interface-Based Filtering

```bash
# Allow from specific interface
iptables -A INPUT -i eth0 -j ACCEPT

# Block from interface
iptables -A INPUT -i eth1 -j DROP

# Allow forwarding between interfaces
iptables -A FORWARD -i eth0 -o eth1 -j ACCEPT

# Allow from specific interface to specific destination
iptables -A INPUT -i eth0 -d 192.168.1.1 -j ACCEPT

# Wildcard interfaces
iptables -A INPUT -i eth+ -j ACCEPT  # eth0, eth1, eth2, etc.
iptables -A INPUT -i wlan+ -j ACCEPT  # wlan0, wlan1, etc.
```

### State-Based Filtering (Stateful Firewall)

Connection tracking allows stateful inspection:

```bash
# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Alternative using state module (deprecated, use conntrack)
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow new connections from specific IP
iptables -A INPUT -s 192.168.1.0/24 -m conntrack --ctstate NEW -j ACCEPT

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Allow only established connections
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT
```

**Connection States**:
- **NEW**: First packet of a new connection
- **ESTABLISHED**: Part of an established connection
- **RELATED**: Related to an established connection (e.g., FTP data channel, ICMP errors)
- **INVALID**: Packet doesn't belong to any known connection
- **UNTRACKED**: Packet marked in raw table to bypass tracking

### MAC Address Filtering

```bash
# Allow specific MAC address
iptables -A INPUT -m mac --mac-source 00:11:22:33:44:55 -j ACCEPT

# Block MAC address
iptables -A INPUT -m mac --mac-source AA:BB:CC:DD:EE:FF -j DROP

# Allow MAC and IP combination
iptables -A INPUT -s 192.168.1.100 -m mac --mac-source 00:11:22:33:44:55 -j ACCEPT
```

## NAT Patterns

### Source NAT (SNAT)

Replace source IP address of outgoing packets.

```bash
# SNAT to specific IP
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.1

# SNAT with port range
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.1:1024-65535

# SNAT to IP range (load balancing)
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.1-203.0.113.10

# SNAT for specific source network
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j SNAT --to-source 203.0.113.1
```

### Masquerading

Dynamic SNAT for connections with dynamic IP (like DHCP).

```bash
# Basic masquerading
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Masquerade specific subnet
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j MASQUERADE

# Masquerade with port range
iptables -t nat -A POSTROUTING -o ppp0 -j MASQUERADE --to-ports 1024-65535

# Enable IP forwarding (required for NAT/masquerading)
echo 1 > /proc/sys/net/ipv4/ip_forward
# Permanent: edit /etc/sysctl.conf
# net.ipv4.ip_forward = 1
```

### Destination NAT (DNAT)

Redirect traffic to different destination (port forwarding).

```bash
# Forward port 80 to internal server
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.100

# Forward port 8080 to port 80 on internal server
iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 192.168.1.100:80

# Forward from specific interface
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 443 -j DNAT --to-destination 192.168.1.100:443

# Load balance across multiple servers
iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode random --probability 0.5 -j DNAT --to-destination 192.168.1.100
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.101
```

### Port Forwarding (Complete Example)

```bash
# External port 2222 → Internal server port 22
# DNAT (redirect incoming)
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 2222 -j DNAT --to-destination 192.168.1.100:22

# FORWARD rule (allow through firewall)
iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 22 -m conntrack --ctstate NEW,ESTABLISHED,RELATED -j ACCEPT

# SNAT/MASQUERADE (if needed for return traffic)
iptables -t nat -A POSTROUTING -o eth1 -d 192.168.1.100 -j MASQUERADE
```

### Redirect (Local Port Redirection)

```bash
# Redirect local port 80 to 8080
iptables -t nat -A OUTPUT -p tcp --dport 80 -j REDIRECT --to-port 8080

# Redirect incoming to local port (transparent proxy)
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 3128

# Redirect from specific IP
iptables -t nat -A PREROUTING -s 192.168.1.0/24 -p tcp --dport 80 -j REDIRECT --to-port 8080
```

### 1:1 NAT (Bidirectional NAT)

```bash
# Map external IP to internal IP (both directions)
# Incoming
iptables -t nat -A PREROUTING -d 203.0.113.1 -j DNAT --to-destination 192.168.1.100
# Outgoing
iptables -t nat -A POSTROUTING -s 192.168.1.100 -j SNAT --to-source 203.0.113.1
```

## Advanced Filtering

### Rate Limiting

Protect against DoS attacks and limit connection rates.

```bash
# Limit SSH connections (3 per minute)
iptables -A INPUT -p tcp --dport 22 -m limit --limit 3/min --limit-burst 5 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP

# Limit ICMP (ping) requests
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s --limit-burst 3 -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# Limit HTTP connections
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Per-IP rate limiting (requires recent module)
iptables -A INPUT -p tcp --dport 80 -m recent --name HTTP --set
iptables -A INPUT -p tcp --dport 80 -m recent --name HTTP --update --seconds 60 --hitcount 20 -j DROP
```

### Connection Limiting

```bash
# Limit concurrent connections per IP (max 10)
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 10 -j REJECT

# Limit with specific message
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 10 -j REJECT --reject-with tcp-reset

# Limit per subnet mask
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 5 --connlimit-mask 24 -j REJECT

# Limit SSH connections per IP
iptables -A INPUT -p tcp --dport 22 -m connlimit --connlimit-above 3 -j DROP
```

### Recent Module (Dynamic Blacklisting)

Track and block IPs based on recent activity.

```bash
# SSH brute force protection
# Mark new SSH connections
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set --name SSH

# Block if more than 3 attempts in 60 seconds
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP

# Port scan protection
iptables -A INPUT -m recent --name portscan --rcheck --seconds 86400 -j DROP
iptables -A INPUT -m recent --name portscan --remove
iptables -A INPUT -p tcp --tcp-flags ALL FIN,URG,PSH -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL SYN,RST,ACK,FIN,URG -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -m recent --set --name portscan -j DROP

# Show recent list
cat /proc/net/xt_recent/SSH
```

### String Matching

Filter packets based on content.

```bash
# Block HTTP requests containing specific string
iptables -A INPUT -p tcp --dport 80 -m string --string "GET /admin" --algo bm -j DROP

# Block specific user agent
iptables -A INPUT -p tcp --dport 80 -m string --string "User-Agent: BadBot" --algo bm -j DROP

# Case insensitive match
iptables -A INPUT -p tcp --dport 80 -m string --string "wordpress" --algo bm --icase -j LOG --log-prefix "WordPress access: "

# Block outgoing traffic containing password
iptables -A OUTPUT -p tcp -m string --string "password=" --algo kmp -j REJECT
```

**Algorithms**:
- `bm`: Boyer-Moore (faster for longer strings)
- `kmp`: Knuth-Pratt-Morris (better for multiple pattern matching)

### Time-Based Rules

```bash
# Allow SSH only during business hours
iptables -A INPUT -p tcp --dport 22 -m time --timestart 09:00 --timestop 17:00 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j DROP

# Allow on specific days (Mon-Fri)
iptables -A INPUT -p tcp --dport 22 -m time --weekdays Mon,Tue,Wed,Thu,Fri -j ACCEPT

# Block during specific time range
iptables -A INPUT -p tcp --dport 80 -m time --timestart 02:00 --timestop 04:00 -j DROP

# Combine time and days
iptables -A FORWARD -m time --weekdays Mon,Tue,Wed,Thu,Fri --timestart 08:00 --timestop 18:00 -j ACCEPT
```

### GeoIP Filtering

Block or allow traffic from specific countries (requires xt_geoip module).

```bash
# Block traffic from specific country
iptables -A INPUT -m geoip --src-cc CN,RU -j DROP

# Allow only from specific countries
iptables -A INPUT -m geoip --src-cc US,CA,GB -j ACCEPT
iptables -A INPUT -j DROP

# Block to specific country
iptables -A OUTPUT -m geoip --dst-cc KP -j REJECT
```

### Owner Matching (OUTPUT Chain)

Filter based on process owner.

```bash
# Allow only root to make connections
iptables -A OUTPUT -m owner --uid-owner 0 -j ACCEPT
iptables -A OUTPUT -j DROP

# Block specific user from internet
iptables -A OUTPUT -m owner --uid-owner 1001 -j REJECT

# Allow specific group
iptables -A OUTPUT -m owner --gid-owner 1000 -j ACCEPT

# Block by process
iptables -A OUTPUT -m owner --uid-owner www-data -d 192.168.1.0/24 -j DROP
```

### TTL Manipulation

```bash
# Set TTL for outgoing packets
iptables -t mangle -A POSTROUTING -j TTL --ttl-set 64

# Increment TTL
iptables -t mangle -A POSTROUTING -j TTL --ttl-inc 1

# Decrement TTL
iptables -t mangle -A PREROUTING -j TTL --ttl-dec 1

# Match TTL value
iptables -A INPUT -m ttl --ttl-eq 64 -j ACCEPT
iptables -A INPUT -m ttl --ttl-gt 128 -j DROP
```

### Packet Marking

Mark packets for advanced routing or QoS.

```bash
# Mark packets in mangle table
iptables -t mangle -A PREROUTING -p tcp --dport 22 -j MARK --set-mark 1
iptables -t mangle -A PREROUTING -p tcp --dport 80 -j MARK --set-mark 2

# Match marked packets
iptables -A FORWARD -m mark --mark 1 -j ACCEPT

# Use with connmark (mark entire connection)
iptables -t mangle -A PREROUTING -j CONNMARK --restore-mark
iptables -t mangle -A PREROUTING -m mark --mark 0 -p tcp --dport 80 -j MARK --set-mark 2
iptables -t mangle -A PREROUTING -j CONNMARK --save-mark
```

## Connection Tracking

Connection tracking (conntrack) is a fundamental feature that enables stateful packet filtering.

### Understanding Conntrack

```bash
# View current connections
conntrack -L

# View specific protocol
conntrack -L -p tcp

# View connections by IP
conntrack -L -s 192.168.1.100
conntrack -L -d 192.168.1.100

# Count connections
conntrack -L -o extended | wc -l

# View connection statistics
cat /proc/net/nf_conntrack

# View conntrack limits
sysctl net.netfilter.nf_conntrack_max
sysctl net.netfilter.nf_conntrack_count
```

### Connection States

```
NEW → ESTABLISHED → (optional) RELATED → FIN_WAIT/CLOSE_WAIT → TIME_WAIT → CLOSED
```

### Conntrack Tuning

```bash
# Increase connection tracking table size
sysctl -w net.netfilter.nf_conntrack_max=262144

# Timeout settings (seconds)
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_established=7200
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_time_wait=120
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_close_wait=60
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_fin_wait=120

# UDP timeouts
sysctl -w net.netfilter.nf_conntrack_udp_timeout=30
sysctl -w net.netfilter.nf_conntrack_udp_timeout_stream=180

# Make permanent in /etc/sysctl.conf
cat >> /etc/sysctl.conf << EOF
net.netfilter.nf_conntrack_max = 262144
net.netfilter.nf_conntrack_tcp_timeout_established = 7200
net.netfilter.nf_conntrack_tcp_timeout_time_wait = 120
EOF
```

### Disable Connection Tracking (Performance)

For high-traffic servers, disable conntrack for specific traffic:

```bash
# Disable tracking for HTTP traffic
iptables -t raw -A PREROUTING -p tcp --dport 80 -j NOTRACK
iptables -t raw -A OUTPUT -p tcp --sport 80 -j NOTRACK

# Must also allow untracked traffic
iptables -A INPUT -p tcp --dport 80 -m conntrack --ctstate UNTRACKED -j ACCEPT
iptables -A OUTPUT -p tcp --sport 80 -m conntrack --ctstate UNTRACKED -j ACCEPT
```

### Conntrack Helpers

Handle protocols with dynamic ports (FTP, SIP, etc.):

```bash
# List available helpers
cat /proc/net/nf_conntrack_helper

# Load FTP helper
modprobe nf_conntrack_ftp
modprobe nf_nat_ftp

# Load SIP helper
modprobe nf_conntrack_sip
modprobe nf_nat_sip

# Configure in iptables
iptables -A INPUT -p tcp --dport 21 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
```

## Chain Management

### Creating Custom Chains

Custom chains improve organization and performance.

```bash
# Create custom chain
iptables -N CUSTOM_INPUT

# Add rules to custom chain
iptables -A CUSTOM_INPUT -p tcp --dport 80 -j ACCEPT
iptables -A CUSTOM_INPUT -p tcp --dport 443 -j ACCEPT
iptables -A CUSTOM_INPUT -j DROP

# Jump to custom chain from main chain
iptables -A INPUT -j CUSTOM_INPUT

# List custom chain
iptables -L CUSTOM_INPUT -n -v
```

### Common Custom Chain Patterns

```bash
# Create logging chain
iptables -N LOG_DROP
iptables -A LOG_DROP -j LOG --log-prefix "IPTables-Dropped: " --log-level 4
iptables -A LOG_DROP -j DROP

# Use logging chain
iptables -A INPUT -p tcp --dport 23 -j LOG_DROP

# Create SSH protection chain
iptables -N SSH_PROTECT
iptables -A SSH_PROTECT -m recent --name SSH --set
iptables -A SSH_PROTECT -m recent --name SSH --update --seconds 60 --hitcount 4 -j LOG_DROP
iptables -A SSH_PROTECT -j ACCEPT

# Use SSH protection
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j SSH_PROTECT

# Create web service chain
iptables -N WEB_SERVICES
iptables -A WEB_SERVICES -p tcp --dport 80 -j ACCEPT
iptables -A WEB_SERVICES -p tcp --dport 443 -j ACCEPT
iptables -A WEB_SERVICES -p tcp --dport 8080 -j ACCEPT
iptables -A WEB_SERVICES -j RETURN

# Jump to web services
iptables -A INPUT -j WEB_SERVICES
```

### Deleting Custom Chains

```bash
# Must flush chain first
iptables -F CUSTOM_INPUT

# Then delete it
iptables -X CUSTOM_INPUT

# Delete all custom chains (careful!)
iptables -X
```

### Default Policies

```bash
# Set restrictive default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# View current policies
iptables -L | grep policy

# Temporary accept-all (dangerous!)
iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT
```

## Common Use Cases

### Basic Firewall Setup

```bash
#!/bin/bash
# Basic server firewall

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow ping (limited)
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s --limit-burst 3 -j ACCEPT

# Log dropped packets
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-INPUT-dropped: " --log-level 4

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Web Server Protection

```bash
# Create web protection chain
iptables -N WEB_PROTECT

# SYN flood protection
iptables -A WEB_PROTECT -p tcp --syn -m limit --limit 2/s --limit-burst 30 -j ACCEPT
iptables -A WEB_PROTECT -p tcp --syn -j DROP

# Connection limiting per IP
iptables -A WEB_PROTECT -p tcp --dport 80 -m connlimit --connlimit-above 20 -j REJECT --reject-with tcp-reset
iptables -A WEB_PROTECT -p tcp --dport 443 -m connlimit --connlimit-above 20 -j REJECT --reject-with tcp-reset

# Block common attack patterns
iptables -A WEB_PROTECT -p tcp --dport 80 -m string --string "../../" --algo bm -j DROP
iptables -A WEB_PROTECT -p tcp --dport 80 -m string --string "SELECT * FROM" --algo bm -j DROP

# Rate limit new connections
iptables -A WEB_PROTECT -p tcp --dport 80 -m conntrack --ctstate NEW -m recent --set --name HTTP
iptables -A WEB_PROTECT -p tcp --dport 80 -m recent --update --seconds 60 --hitcount 50 --name HTTP -j DROP

# Accept legitimate traffic
iptables -A WEB_PROTECT -j ACCEPT

# Apply to INPUT
iptables -A INPUT -p tcp -m multiport --dports 80,443 -j WEB_PROTECT
```

### SSH Brute Force Protection

```bash
# Method 1: Using recent module
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j LOG --log-prefix "SSH-brute-force: "
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Method 2: Using limit module
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 3/min --limit-burst 3 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j DROP
```

### Home Router/Gateway NAT

```bash
#!/bin/bash
# Home router configuration

# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow from LAN
iptables -A INPUT -i eth1 -s 192.168.1.0/24 -j ACCEPT

# Allow forwarding from LAN to WAN
iptables -A FORWARD -i eth1 -o eth0 -s 192.168.1.0/24 -j ACCEPT

# NAT for LAN
iptables -t nat -A POSTROUTING -o eth0 -s 192.168.1.0/24 -j MASQUERADE

# Port forwarding example (web server on 192.168.1.100)
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.1.100:80
iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 80 -j ACCEPT

# Allow DNS for router itself
iptables -A INPUT -p udp --dport 53 -j ACCEPT

# Allow DHCP for router itself
iptables -A INPUT -p udp --dport 67:68 -j ACCEPT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Docker Network Integration

```bash
# Allow docker containers to internet
iptables -A FORWARD -i docker0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o docker0 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -t nat -A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE

# Expose container port
# Container 172.17.0.2:8080 → Host port 80
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 172.17.0.2:8080
iptables -A FORWARD -p tcp -d 172.17.0.2 --dport 8080 -j ACCEPT

# Isolate docker network from other networks
iptables -I FORWARD -i docker0 ! -o docker0 -j DROP
iptables -I FORWARD -i docker0 -o docker0 -j ACCEPT
```

### Load Balancing

```bash
# Simple round-robin load balancing
iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode nth --every 3 --packet 0 -j DNAT --to-destination 192.168.1.10:80
iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode nth --every 2 --packet 0 -j DNAT --to-destination 192.168.1.11:80
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.12:80

# Random load balancing
iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode random --probability 0.33 -j DNAT --to-destination 192.168.1.10:80
iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode random --probability 0.5 -j DNAT --to-destination 192.168.1.11:80
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.12:80
```

## Best Practices

### Security Considerations

1. **Default Deny Policy**
   ```bash
   # Start with restrictive defaults
   iptables -P INPUT DROP
   iptables -P FORWARD DROP
   iptables -P OUTPUT ACCEPT  # or DROP for maximum security
   ```

2. **Allow Loopback**
   ```bash
   # Always allow loopback interface
   iptables -A INPUT -i lo -j ACCEPT
   iptables -A OUTPUT -o lo -j ACCEPT
   ```

3. **Drop Invalid Packets**
   ```bash
   iptables -A INPUT -m conntrack --ctstate INVALID -j DROP
   ```

4. **Rate Limiting Critical Services**
   ```bash
   # Always rate limit SSH
   iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
   iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 --hitcount 4 -j DROP
   ```

5. **Log Suspicious Activity**
   ```bash
   # Log before dropping
   iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-denied: "
   ```

### Rule Ordering

**Critical**: Rules are processed top-to-bottom. First match wins!

```bash
# WRONG ORDER (SSH rule never reached)
iptables -A INPUT -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# CORRECT ORDER
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -j DROP

# Best practice: specific rules first, general rules last
iptables -A INPUT -i lo -j ACCEPT                              # 1. Loopback
iptables -A INPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT # 2. Established
iptables -A INPUT -p tcp --dport 22 -j ACCEPT                  # 3. Specific services
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -j LOG --log-prefix "dropped: "              # 4. Log
iptables -A INPUT -j DROP                                      # 5. Default deny
```

### Testing Safely

```bash
# Method 1: Auto-reset with at command
at now + 5 minutes <<< 'iptables -F; iptables -P INPUT ACCEPT; iptables -P OUTPUT ACCEPT; iptables -P FORWARD ACCEPT'
# Now test your rules; if you get locked out, rules reset in 5 minutes

# Method 2: Test script
#!/bin/bash
iptables-restore < /etc/iptables/test-rules.v4
echo "Rules applied. You have 60 seconds to test. Press Ctrl+C to keep, or wait to rollback."
sleep 60
iptables-restore < /etc/iptables/rules.v4
echo "Rolled back to previous rules"

# Method 3: iptables-apply (Debian/Ubuntu)
iptables-apply /etc/iptables/test-rules.v4
# Prompts for confirmation; auto-rollback if no response
```

### Backup and Restore

```bash
# Backup current rules
iptables-save > /root/iptables-backup-$(date +%Y%m%d-%H%M%S).rules

# Restore from backup
iptables-restore < /root/iptables-backup-20250114-120000.rules

# Automated backup before changes
#!/bin/bash
BACKUP_DIR="/root/iptables-backups"
mkdir -p $BACKUP_DIR
iptables-save > $BACKUP_DIR/rules-$(date +%Y%m%d-%H%M%S).v4
# Keep only last 10 backups
ls -t $BACKUP_DIR/rules-*.v4 | tail -n +11 | xargs -r rm
```

### Performance Optimization

1. **Put most common rules first**
   ```bash
   # High-traffic rules at top
   iptables -I INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
   ```

2. **Use custom chains for organization**
   ```bash
   # Jump to specific chain early
   iptables -A INPUT -p tcp --dport 80 -j WEB_CHAIN
   ```

3. **Minimize rule count**
   ```bash
   # Use multiport instead of multiple rules
   # INEFFICIENT
   iptables -A INPUT -p tcp --dport 80 -j ACCEPT
   iptables -A INPUT -p tcp --dport 443 -j ACCEPT
   iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

   # EFFICIENT
   iptables -A INPUT -p tcp -m multiport --dports 80,443,8080 -j ACCEPT
   ```

4. **Use NOTRACK for high-volume traffic**
   ```bash
   iptables -t raw -A PREROUTING -p tcp --dport 80 -j NOTRACK
   ```

### Persistence Across Reboots

**Debian/Ubuntu:**
```bash
# Install persistence package
apt-get install iptables-persistent

# Save rules
netfilter-persistent save

# Manual save
iptables-save > /etc/iptables/rules.v4
ip6tables-save > /etc/iptables/rules.v6
```

**RedHat/CentOS:**
```bash
# Save rules
service iptables save
# Saves to /etc/sysconfig/iptables

# Enable on boot
systemctl enable iptables
```

**Generic (systemd service):**
```bash
# Create restore script
cat > /etc/iptables/restore.sh << 'EOF'
#!/bin/bash
iptables-restore < /etc/iptables/rules.v4
ip6tables-restore < /etc/iptables/rules.v6
EOF
chmod +x /etc/iptables/restore.sh

# Create systemd service
cat > /etc/systemd/system/iptables-restore.service << 'EOF'
[Unit]
Description=Restore iptables rules
Before=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
ExecStart=/etc/iptables/restore.sh

[Install]
WantedBy=multi-user.target
EOF

systemctl enable iptables-restore.service
```

## Troubleshooting

### Debugging Rules

```bash
# Enable verbose logging for specific rule
iptables -A INPUT -p tcp --dport 80 -j LOG --log-prefix "HTTP-DEBUG: " --log-level 7

# Watch logs in real-time
tail -f /var/log/kern.log | grep iptables
# or
dmesg -w | grep iptables

# Trace packet path
# Add TRACE target in raw table
iptables -t raw -A PREROUTING -p tcp --dport 80 -j TRACE
iptables -t raw -A OUTPUT -p tcp --sport 80 -j TRACE

# View trace (requires iptables logging)
tail -f /var/log/kern.log | grep TRACE

# Don't forget to remove TRACE when done!
iptables -t raw -D PREROUTING -p tcp --dport 80 -j TRACE
```

### Common Issues

**Issue 1: Rules not persisting after reboot**
```bash
# Solution: Save and configure persistence
iptables-save > /etc/iptables/rules.v4
apt-get install iptables-persistent  # Debian/Ubuntu
```

**Issue 2: Locked out after applying rules**
```bash
# Prevention: Always allow established connections first
iptables -I INPUT 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Recovery: Access via console or KVM, then:
iptables -F
iptables -P INPUT ACCEPT
```

**Issue 3: NAT not working**
```bash
# Check IP forwarding
cat /proc/sys/net/ipv4/ip_forward  # Should be 1
echo 1 > /proc/sys/net/ipv4/ip_forward

# Verify NAT rules
iptables -t nat -L -n -v

# Check routing
ip route show
```

**Issue 4: Connection tracking table full**
```bash
# Check current usage
cat /proc/sys/net/netfilter/nf_conntrack_count
cat /proc/sys/net/netfilter/nf_conntrack_max

# Increase limit
sysctl -w net.netfilter.nf_conntrack_max=262144

# Or use NOTRACK for high-volume traffic
iptables -t raw -A PREROUTING -p tcp --dport 80 -j NOTRACK
```

**Issue 5: Performance degradation**
```bash
# Check rule count
iptables -L -n | wc -l

# Analyze rule hit counters
iptables -L -n -v --line-numbers

# Optimize: Move frequently matched rules to top
# Use ipset for large IP lists instead of many rules
```

### Diagnostic Commands

```bash
# Show all rules with packet counters
iptables -L -n -v --line-numbers

# Show all rules in all tables
for table in filter nat mangle raw security; do
    echo "=== Table: $table ==="
    iptables -t $table -L -n -v --line-numbers
done

# Check if module is loaded
lsmod | grep iptable
lsmod | grep nf_conntrack

# View connection tracking
conntrack -L
conntrack -L | wc -l  # Count connections

# View NAT translations
conntrack -L -p tcp --dport 80

# Performance stats
iptables -L -n -v -x  # Extended counters
```

### Testing Rules

```bash
# Test with specific packet
# Install hping3
apt-get install hping3

# Send SYN packet
hping3 -S -p 80 192.168.1.1

# Send UDP packet
hping3 --udp -p 53 8.8.8.8

# Test with netcat
nc -v -w 2 192.168.1.1 80

# Test with nmap
nmap -p 22,80,443 192.168.1.1
```

## Performance Tuning

### Connection Tracking Optimization

```bash
# Increase connection tracking table
sysctl -w net.netfilter.nf_conntrack_max=524288
sysctl -w net.netfilter.nf_conntrack_buckets=131072

# Reduce timeouts for high-traffic servers
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_established=600
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_time_wait=30
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_close_wait=30
sysctl -w net.netfilter.nf_conntrack_tcp_timeout_fin_wait=30

# Disable tracking for specific high-volume traffic
iptables -t raw -A PREROUTING -p tcp --dport 80 -j NOTRACK
iptables -t raw -A OUTPUT -p tcp --sport 80 -j NOTRACK
iptables -A INPUT -p tcp --dport 80 -m state --state UNTRACKED -j ACCEPT
iptables -A OUTPUT -p tcp --sport 80 -m state --state UNTRACKED -j ACCEPT
```

### Hash Limits

For rate limiting at scale:

```bash
# Use hashlimit instead of limit for per-IP limiting
iptables -A INPUT -p tcp --dport 80 \
    -m hashlimit --hashlimit-name http \
    --hashlimit-above 50/sec --hashlimit-burst 100 \
    --hashlimit-mode srcip -j DROP

# Per subnet limiting
iptables -A INPUT -p tcp --dport 22 \
    -m hashlimit --hashlimit-name ssh \
    --hashlimit-above 3/min \
    --hashlimit-mode srcip --hashlimit-srcmask 24 -j DROP
```

### ipset Integration

Use ipset for managing large IP lists efficiently:

```bash
# Install ipset
apt-get install ipset

# Create IP set
ipset create blacklist hash:ip hashsize 4096

# Add IPs to set
ipset add blacklist 192.168.1.100
ipset add blacklist 10.0.0.0/8

# Use in iptables (single rule for entire set!)
iptables -A INPUT -m set --match-set blacklist src -j DROP

# Manage set
ipset list blacklist
ipset del blacklist 192.168.1.100
ipset flush blacklist
ipset destroy blacklist

# Save/restore sets
ipset save > /etc/ipset.conf
ipset restore < /etc/ipset.conf
```

### Packet Processing Optimization

```bash
# Drop invalid packets early (raw table)
iptables -t raw -A PREROUTING -m conntrack --ctstate INVALID -j DROP

# Early accept for established connections
iptables -I INPUT 1 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Use conntrack instead of state module
# SLOWER
iptables -A INPUT -m state --state ESTABLISHED -j ACCEPT
# FASTER
iptables -A INPUT -m conntrack --ctstate ESTABLISHED -j ACCEPT
```

### Monitoring Performance

```bash
# Check packet processing rate
watch -n 1 'iptables -L -n -v -x'

# Identify slow rules
time iptables -L -n > /dev/null

# Profile rule matching
for i in $(seq 1 $(iptables -L INPUT --line-numbers | tail -n +3 | wc -l)); do
    echo -n "Rule $i: "
    iptables -L INPUT $i -n -v | grep -v '^Chain' | grep -v '^target'
done
```

## Modern Alternatives

### nftables

nftables is the modern replacement for iptables, offering:
- Better performance
- Simplified syntax
- Atomic rule updates
- No separate tools for IPv4/IPv6

**Basic nftables example:**
```bash
# Install
apt-get install nftables

# Basic firewall
nft add table inet filter
nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }

# Add rules
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input iif lo accept
nft add rule inet filter input tcp dport 22 accept
nft add rule inet filter input tcp dport { 80, 443 } accept

# List rules
nft list ruleset

# Save rules
nft list ruleset > /etc/nftables.conf
```

**Translation from iptables:**
```bash
# iptables to nftables
iptables-translate -A INPUT -p tcp --dport 80 -j ACCEPT
# Output: nft add rule ip filter INPUT tcp dport 80 counter accept

# Translate entire ruleset
iptables-save | iptables-restore-translate
ip6tables-save | ip6tables-restore-translate
```

### eBPF/XDP

Extended Berkeley Packet Filter (eBPF) and eXpress Data Path (XDP) provide ultra-high performance packet filtering:

- Runs in kernel context
- Processes packets before network stack
- Can achieve 10Gbps+ filtering rates

**Example use case:** DDoS mitigation at wire speed

```bash
# Requires modern kernel (4.8+) and tools
# Example with Cilium for Kubernetes
kubectl apply -f cilium.yaml

# Or standalone with bpfilter
# Coming in future kernels as iptables replacement
```

### Comparison

| Feature | iptables | nftables | eBPF/XDP |
|---------|----------|----------|----------|
| Performance | Good | Better | Excellent |
| Syntax | Complex | Simpler | Programmatic |
| IPv4/IPv6 | Separate | Unified | Unified |
| Atomic updates | No | Yes | Yes |
| Learning curve | Moderate | Moderate | Steep |
| Maturity | Very mature | Mature | Emerging |
| Use case | General firewall | General firewall | High-performance |

## Advanced Topics

### Transparent Proxy

Redirect traffic to proxy without client configuration:

```bash
# Redirect HTTP to Squid proxy (port 3128)
iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 80 -j REDIRECT --to-port 3128

# Prevent loop (don't redirect proxy's own traffic)
iptables -t nat -A OUTPUT -m owner --uid-owner proxy -j RETURN
iptables -t nat -A OUTPUT -p tcp --dport 80 -j REDIRECT --to-port 3128
```

### Policy Routing with fwmark

```bash
# Mark packets
iptables -t mangle -A PREROUTING -s 192.168.1.0/24 -j MARK --set-mark 1
iptables -t mangle -A PREROUTING -s 192.168.2.0/24 -j MARK --set-mark 2

# Add routing tables in /etc/iproute2/rt_tables
echo "1 ISP1" >> /etc/iproute2/rt_tables
echo "2 ISP2" >> /etc/iproute2/rt_tables

# Add routes
ip route add default via 10.0.1.1 table ISP1
ip route add default via 10.0.2.1 table ISP2

# Policy routing rules
ip rule add fwmark 1 table ISP1
ip rule add fwmark 2 table ISP2
```

### Bridge Filtering

Filter traffic between bridged interfaces:

```bash
# Enable bridge netfilter
modprobe br_netfilter
echo 1 > /proc/sys/net/bridge/bridge-nf-call-iptables

# Filter bridged traffic
iptables -A FORWARD -m physdev --physdev-in eth0 --physdev-out eth1 -j ACCEPT
```

### Conclusion

Netfilter is a powerful and flexible framework for packet filtering and manipulation in Linux. Understanding its architecture, tables, chains, and hooks is essential for effective network management and security. While iptables has been the traditional interface, modern alternatives like nftables and eBPF offer improved performance and capabilities for specific use cases.

**Key Takeaways**:
- Always use stateful filtering (`-m conntrack --ctstate ESTABLISHED,RELATED`)
- Follow default-deny principle for security
- Order rules from specific to general
- Test rules safely with auto-rollback mechanisms
- Monitor and optimize for performance
- Keep rules organized with custom chains
- Regular backups before changes
- Consider nftables for new deployments

## References

- [Netfilter Project](https://www.netfilter.org/)
- [iptables Tutorial](https://www.frozentux.net/iptables-tutorial/iptables-tutorial.html)
- [Linux Kernel Documentation - Netfilter](https://www.kernel.org/doc/Documentation/networking/netfilter/)
- [nftables Wiki](https://wiki.nftables.org/)
- [Netfilter Connection Tracking](https://conntrack-tools.netfilter.org/)
