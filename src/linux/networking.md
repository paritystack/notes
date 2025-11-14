# Linux Networking

Linux provides a comprehensive networking stack with powerful tools for configuration, monitoring, and troubleshooting. This guide covers network interfaces, routing, firewalling, debugging, and common networking patterns.

## Overview

Linux networking operates through multiple layers of the network stack, providing flexible and powerful network management capabilities.

**Key Concepts:**
- **Network Interface**: Hardware or virtual device for network communication
- **IP Address**: Unique identifier for devices on a network
- **Routing**: Directing network traffic between networks
- **Network Namespace**: Isolated network stack instance
- **Firewall**: Packet filtering and network security
- **Bridge**: Virtual switch connecting network interfaces

## Network Stack Layers

Linux implements the TCP/IP model:

1. **Application Layer**: HTTP, DNS, SSH, etc.
2. **Transport Layer**: TCP, UDP
3. **Network Layer**: IP, ICMP, routing
4. **Link Layer**: Ethernet, WiFi, ARP

## Network Interfaces

### Interface Types

Linux supports various interface types:

```bash
# List all interfaces
ip link show
ip addr show

# Show specific interface
ip link show eth0
ip addr show wlan0

# Show interface statistics
ip -s link show eth0
```

### Physical Interfaces

Physical network interfaces connect to hardware:

```bash
# Ethernet interfaces: eth0, eth1, ens33, enp3s0
# Wireless interfaces: wlan0, wlp2s0

# Bring interface up/down
sudo ip link set eth0 up
sudo ip link set eth0 down

# Check interface status
ip link show eth0
cat /sys/class/net/eth0/operstate

# View interface details
ethtool eth0
ethtool -i eth0  # Driver information
ethtool -S eth0  # Statistics
```

### Loopback Interface

The loopback interface (lo) enables local communication:

```bash
# View loopback
ip addr show lo

# Loopback is typically 127.0.0.1 (IPv4) and ::1 (IPv6)
ping 127.0.0.1
ping localhost
```

### Virtual Ethernet (VETH) Pairs

VETH pairs create virtual cable connections:

```bash
# Create veth pair
sudo ip link add veth0 type veth peer name veth1

# Bring them up
sudo ip link set veth0 up
sudo ip link set veth1 up

# Assign IP addresses
sudo ip addr add 10.0.0.1/24 dev veth0
sudo ip addr add 10.0.0.2/24 dev veth1

# Delete veth pair
sudo ip link delete veth0
```

### Dummy Interfaces

Dummy interfaces for testing and special purposes:

```bash
# Create dummy interface
sudo ip link add dummy0 type dummy

# Assign IP and bring up
sudo ip addr add 192.168.100.1/24 dev dummy0
sudo ip link set dummy0 up

# Delete dummy interface
sudo ip link delete dummy0
```

## TUN and TAP Interfaces

TUN and TAP are virtual network kernel interfaces that operate at different layers of the network stack.

### TUN Interface

A TUN (network TUNnel) interface is a virtual point-to-point network device that operates at the network layer (Layer 3). It is used to route IP packets.

**Key Features:**
- Operates at Layer 3 (Network Layer)
- Handles IP packets
- Used for routing and tunneling IP traffic
- Commonly used in VPNs

**Use Case Example:**
TUN interfaces create secure VPN connections between remote networks, allowing them to communicate as if on the same local network.

### TAP Interface

A TAP (network TAP) interface is a virtual network device that operates at the data link layer (Layer 2). It handles Ethernet frames.

**Key Features:**
- Operates at Layer 2 (Data Link Layer)
- Handles Ethernet frames
- Used for bridging and virtual machine networking
- Can create virtual switches

**Use Case Example:**
TAP interfaces connect virtual machines to virtual switches, allowing VMs to communicate with each other and the host as if connected to a physical Ethernet switch.

### Creating TUN and TAP Interfaces

```bash
# Install required package
sudo apt-get install uml-utilities

# Create TUN interface
sudo ip tuntap add dev tun0 mode tun
sudo ip addr add 10.0.1.1/24 dev tun0
sudo ip link set tun0 up

# Create TAP interface
sudo ip tuntap add dev tap0 mode tap
sudo ip addr add 10.0.2.1/24 dev tap0
sudo ip link set tap0 up

# Delete TUN/TAP interfaces
sudo ip link delete tun0
sudo ip link delete tap0

# Using tunctl (alternative)
sudo tunctl -t tap0 -u username
sudo ip link set tap0 up
```

### VPN with TUN Interface

```bash
# OpenVPN typically uses TUN
# /etc/openvpn/server.conf
dev tun
server 10.8.0.0 255.255.255.0
push "redirect-gateway def1"

# Start OpenVPN
sudo openvpn --config server.conf
```

## IP Address Management

### Assigning IP Addresses

```bash
# Add IPv4 address
sudo ip addr add 192.168.1.100/24 dev eth0

# Add IPv6 address
sudo ip addr add 2001:db8::1/64 dev eth0

# Add multiple addresses to same interface
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip addr add 192.168.1.102/24 dev eth0

# Remove IP address
sudo ip addr del 192.168.1.100/24 dev eth0

# Flush all addresses from interface
sudo ip addr flush dev eth0
```

### DHCP Configuration

```bash
# Request IP via DHCP (dhclient)
sudo dhclient eth0

# Release DHCP lease
sudo dhclient -r eth0

# Request IP via DHCP (dhcpcd)
sudo dhcpcd eth0

# Using NetworkManager
sudo nmcli device modify eth0 ipv4.method auto
```

### Static IP Configuration

```bash
# Temporary (lost on reboot)
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip link set eth0 up
sudo ip route add default via 192.168.1.1

# Permanent - Debian/Ubuntu (/etc/network/interfaces)
auto eth0
iface eth0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    gateway 192.168.1.1
    dns-nameservers 8.8.8.8 8.8.4.4

# Permanent - RHEL/CentOS (/etc/sysconfig/network-scripts/ifcfg-eth0)
DEVICE=eth0
BOOTPROTO=static
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8
ONBOOT=yes

# Permanent - Netplan (Ubuntu 18.04+) (/etc/netplan/01-netcfg.yaml)
network:
  version: 2
  ethernets:
    eth0:
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# Apply netplan config
sudo netplan apply
```

## Routing

### Viewing Routes

```bash
# Show routing table
ip route show
ip route list

# Show routing table for specific interface
ip route show dev eth0

# Show IPv6 routes
ip -6 route show

# Legacy command
route -n
netstat -rn
```

### Adding Static Routes

```bash
# Add route to network
sudo ip route add 10.0.0.0/24 via 192.168.1.1
sudo ip route add 10.0.0.0/24 via 192.168.1.1 dev eth0

# Add default gateway
sudo ip route add default via 192.168.1.1
sudo ip route add default via 192.168.1.1 dev eth0

# Add route with metric (priority)
sudo ip route add 10.0.0.0/24 via 192.168.1.1 metric 100

# IPv6 route
sudo ip -6 route add 2001:db8::/32 via 2001:db8::1
```

### Deleting Routes

```bash
# Delete specific route
sudo ip route del 10.0.0.0/24
sudo ip route del default via 192.168.1.1

# Delete all routes for interface
sudo ip route flush dev eth0
```

### Multiple Routing Tables

```bash
# View routing tables
ip rule list
cat /etc/iproute2/rt_tables

# Create custom routing table
# Add to /etc/iproute2/rt_tables:
# 100 custom

# Add routes to custom table
sudo ip route add 10.0.0.0/24 via 192.168.1.1 table custom
sudo ip route add default via 192.168.1.1 table custom

# Add rule to use custom table
sudo ip rule add from 192.168.1.100 table custom
sudo ip rule add iif eth1 table custom

# Delete rule
sudo ip rule del from 192.168.1.100 table custom
```

### Policy-Based Routing

```bash
# Route based on source IP
sudo ip rule add from 10.0.0.0/24 table 100
sudo ip route add default via 192.168.1.1 table 100

# Route based on destination
sudo ip rule add to 8.8.8.8 table 101
sudo ip route add default via 192.168.2.1 table 101

# Route based on interface
sudo ip rule add iif eth1 table 102

# Mark-based routing (with iptables)
sudo iptables -t mangle -A PREROUTING -s 10.0.0.0/24 -j MARK --set-mark 100
sudo ip rule add fwmark 100 table 100
```

## Network Namespaces

Network namespaces provide isolated network stacks:

```bash
# List namespaces
ip netns list

# Create namespace
sudo ip netns add netns1

# Execute command in namespace
sudo ip netns exec netns1 ip addr show
sudo ip netns exec netns1 bash  # Interactive shell

# Create veth pair and move to namespace
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth1 netns netns1

# Configure interfaces in namespace
sudo ip netns exec netns1 ip addr add 10.0.0.1/24 dev veth1
sudo ip netns exec netns1 ip link set veth1 up
sudo ip netns exec netns1 ip link set lo up

# Configure host side
sudo ip addr add 10.0.0.2/24 dev veth0
sudo ip link set veth0 up

# Delete namespace
sudo ip netns del netns1
```

### Container-like Isolation

```bash
# Create isolated network environment
sudo ip netns add isolated

# Create veth pair
sudo ip link add veth-host type veth peer name veth-isolated

# Move one end to namespace
sudo ip link set veth-isolated netns isolated

# Configure namespace
sudo ip netns exec isolated ip addr add 172.16.0.2/24 dev veth-isolated
sudo ip netns exec isolated ip link set veth-isolated up
sudo ip netns exec isolated ip link set lo up
sudo ip netns exec isolated ip route add default via 172.16.0.1

# Configure host
sudo ip addr add 172.16.0.1/24 dev veth-host
sudo ip link set veth-host up

# Enable NAT for namespace
sudo iptables -t nat -A POSTROUTING -s 172.16.0.0/24 -j MASQUERADE
sudo sysctl -w net.ipv4.ip_forward=1

# Test from namespace
sudo ip netns exec isolated ping 8.8.8.8
```

## Network Bridges

Bridges connect multiple network interfaces:

```bash
# Create bridge
sudo ip link add br0 type bridge

# Add interfaces to bridge
sudo ip link set eth0 master br0
sudo ip link set eth1 master br0

# Configure bridge
sudo ip addr add 192.168.1.1/24 dev br0
sudo ip link set br0 up

# Remove interface from bridge
sudo ip link set eth0 nomaster

# View bridge details
bridge link show
bridge fdb show  # Forwarding database

# Delete bridge
sudo ip link delete br0
```

### Bridge with TAP for VMs

```bash
# Create bridge for VMs
sudo ip link add br0 type bridge
sudo ip link set br0 up

# Add physical interface
sudo ip link set eth0 master br0

# Create TAP for VM
sudo ip tuntap add dev tap0 mode tap
sudo ip link set tap0 master br0
sudo ip link set tap0 up

# Start VM with tap0 interface
# qemu-system-x86_64 -netdev tap,id=net0,ifname=tap0,script=no -device virtio-net-pci,netdev=net0
```

### Bridge Configuration File

```bash
# /etc/network/interfaces (Debian/Ubuntu)
auto br0
iface br0 inet static
    address 192.168.1.1
    netmask 255.255.255.0
    bridge_ports eth0 eth1
    bridge_stp off
    bridge_fd 0
```

## VLAN Configuration

VLANs segment network traffic:

```bash
# Load 8021q module
sudo modprobe 8021q

# Create VLAN interface
sudo ip link add link eth0 name eth0.10 type vlan id 10

# Configure VLAN interface
sudo ip addr add 192.168.10.1/24 dev eth0.10
sudo ip link set eth0.10 up

# Create multiple VLANs
sudo ip link add link eth0 name eth0.20 type vlan id 20
sudo ip addr add 192.168.20.1/24 dev eth0.20
sudo ip link set eth0.20 up

# Remove VLAN interface
sudo ip link delete eth0.10

# View VLAN configuration
cat /proc/net/vlan/config
ip -d link show eth0.10
```

### VLAN Configuration File

```bash
# /etc/network/interfaces
auto eth0.10
iface eth0.10 inet static
    address 192.168.10.1
    netmask 255.255.255.0
    vlan-raw-device eth0
```

## Bonding and Teaming

### Network Bonding

Link aggregation for redundancy and bandwidth:

```bash
# Load bonding module
sudo modprobe bonding

# Create bond interface
sudo ip link add bond0 type bond mode active-backup
sudo ip link set bond0 up

# Add slaves to bond
sudo ip link set eth0 master bond0
sudo ip link set eth1 master bond0

# Configure bond
sudo ip addr add 192.168.1.100/24 dev bond0

# View bond status
cat /proc/net/bonding/bond0

# Remove bond
sudo ip link set eth0 nomaster
sudo ip link set eth1 nomaster
sudo ip link delete bond0
```

### Bonding Modes

```bash
# Mode 0: balance-rr (round-robin)
sudo ip link add bond0 type bond mode balance-rr

# Mode 1: active-backup (failover)
sudo ip link add bond0 type bond mode active-backup

# Mode 2: balance-xor
sudo ip link add bond0 type bond mode balance-xor

# Mode 3: broadcast
sudo ip link add bond0 type bond mode broadcast

# Mode 4: 802.3ad (LACP)
sudo ip link add bond0 type bond mode 802.3ad

# Mode 5: balance-tlb
sudo ip link add bond0 type bond mode balance-tlb

# Mode 6: balance-alb
sudo ip link add bond0 type bond mode balance-alb
```

### Bonding Configuration File

```bash
# /etc/network/interfaces
auto bond0
iface bond0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    bond-slaves eth0 eth1
    bond-mode active-backup
    bond-miimon 100
    bond-primary eth0
```

## Firewall and Packet Filtering

### iptables

Netfilter/iptables provides packet filtering and NAT:

```bash
# View current rules
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v
sudo iptables -t mangle -L -n -v

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
sudo ip6tables-save > /etc/iptables/rules.v6

# Restore rules
sudo iptables-restore < /etc/iptables/rules.v4

# Flush all rules
sudo iptables -F
sudo iptables -X
sudo iptables -t nat -F
sudo iptables -t mangle -F
```

### Basic iptables Rules

```bash
# Set default policies
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow ping
sudo iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT

# Drop invalid packets
sudo iptables -A INPUT -m state --state INVALID -j DROP

# Log dropped packets
sudo iptables -A INPUT -j LOG --log-prefix "IPTables-Dropped: "
sudo iptables -A INPUT -j DROP
```

### NAT and Masquerading

```bash
# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf

# Masquerade outgoing traffic
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# SNAT (Source NAT)
sudo iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -o eth0 -j SNAT --to-source 203.0.113.1

# DNAT (Destination NAT) - Port forwarding
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.100:8080

# Port forwarding example
sudo iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 8080 -j DNAT --to 192.168.1.100:80
sudo iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 80 -j ACCEPT
```

### Rate Limiting

```bash
# Limit SSH connections
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP

# Limit ping rate
sudo iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT
sudo iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# Connection limiting
sudo iptables -A INPUT -p tcp --syn --dport 80 -m connlimit --connlimit-above 20 -j REJECT
```

### Blocking by IP/Network

```bash
# Block specific IP
sudo iptables -A INPUT -s 203.0.113.50 -j DROP

# Block network
sudo iptables -A INPUT -s 203.0.113.0/24 -j DROP

# Block by country (using ipset)
sudo ipset create china hash:net
sudo ipset add china 1.2.3.0/24
sudo iptables -A INPUT -m set --match-set china src -j DROP
```

### nftables (Modern Replacement)

```bash
# List rules
sudo nft list ruleset

# Create table
sudo nft add table inet filter

# Create chains
sudo nft add chain inet filter input { type filter hook input priority 0 \; policy drop \; }
sudo nft add chain inet filter forward { type filter hook forward priority 0 \; policy drop \; }
sudo nft add chain inet filter output { type filter hook output priority 0 \; policy accept \; }

# Add rules
sudo nft add rule inet filter input iif lo accept
sudo nft add rule inet filter input ct state established,related accept
sudo nft add rule inet filter input tcp dport 22 accept
sudo nft add rule inet filter input tcp dport { 80, 443 } accept

# Save rules
sudo nft list ruleset > /etc/nftables.conf

# Load rules
sudo nft -f /etc/nftables.conf

# Flush rules
sudo nft flush ruleset
```

## Network Debugging and Monitoring

### Connectivity Testing

```bash
# Ping
ping 8.8.8.8
ping -c 4 google.com  # 4 packets
ping -i 0.2 8.8.8.8  # 0.2 second interval
ping -s 1500 8.8.8.8  # Large packet size

# Ping IPv6
ping6 2001:4860:4860::8888

# Traceroute
traceroute google.com
traceroute -n 8.8.8.8  # No DNS resolution
traceroute -T -p 80 google.com  # TCP SYN to port 80

# MTR (better than traceroute)
mtr google.com
mtr -n -c 100 8.8.8.8  # 100 cycles, no DNS
```

### Socket Statistics

```bash
# ss (modern netstat replacement)
ss -tuln  # TCP/UDP listening ports
ss -tupn  # TCP/UDP with process info
ss -tan  # All TCP connections, numeric
ss -s  # Summary statistics

# Show specific port
ss -tulpn | grep :80
ss -tulpn sport = :22

# Show established connections
ss -t state established

# Show listening sockets
ss -tl

# netstat (legacy)
netstat -tuln  # Listening ports
netstat -tupn  # With process info
netstat -s  # Statistics
netstat -i  # Interface statistics
```

### Packet Capture

```bash
# tcpdump
sudo tcpdump -i eth0
sudo tcpdump -i eth0 -n  # No DNS resolution
sudo tcpdump -i eth0 -c 100  # Capture 100 packets

# Filter by host
sudo tcpdump -i eth0 host 192.168.1.100
sudo tcpdump -i eth0 src 192.168.1.100
sudo tcpdump -i eth0 dst 192.168.1.100

# Filter by port
sudo tcpdump -i eth0 port 80
sudo tcpdump -i eth0 tcp port 22
sudo tcpdump -i eth0 udp port 53

# Save to file
sudo tcpdump -i eth0 -w capture.pcap
sudo tcpdump -i eth0 -w capture.pcap -C 100  # 100MB files
sudo tcpdump -i eth0 -G 3600 -w capture-%Y%m%d-%H%M%S.pcap  # Rotate hourly

# Read from file
tcpdump -r capture.pcap
tcpdump -r capture.pcap -n 'tcp port 80'

# Advanced filters
sudo tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn) != 0'  # SYN packets
sudo tcpdump -i eth0 'icmp[icmptype] = icmp-echo'  # Ping requests
sudo tcpdump -i eth0 -n -A 'port 80 and host 192.168.1.100'  # ASCII output
```

### Bandwidth Monitoring

```bash
# iftop (interactive)
sudo iftop -i eth0
sudo iftop -i eth0 -n  # No DNS resolution

# nethogs (per-process)
sudo nethogs eth0

# nload
nload eth0

# bmon
bmon -p eth0

# vnstat (statistics database)
vnstat -i eth0
vnstat -l -i eth0  # Live mode
vnstat -h -i eth0  # Hourly stats
vnstat -d -i eth0  # Daily stats
```

### Network Scanning

```bash
# nmap
nmap 192.168.1.1  # Basic scan
nmap -p 22,80,443 192.168.1.1  # Specific ports
nmap -p- 192.168.1.1  # All ports
nmap -sV 192.168.1.1  # Version detection
nmap -O 192.168.1.1  # OS detection
nmap -A 192.168.1.1  # Aggressive scan
nmap 192.168.1.0/24  # Network scan

# Network discovery
nmap -sn 192.168.1.0/24  # Ping scan
nmap -sL 192.168.1.0/24  # List scan

# arp-scan
sudo arp-scan -l  # Local network
sudo arp-scan --interface=eth0 192.168.1.0/24
```

### Interface Information

```bash
# ethtool
ethtool eth0  # Link status
ethtool -i eth0  # Driver info
ethtool -S eth0  # Statistics
ethtool -g eth0  # Ring buffer
ethtool -k eth0  # Offload features

# Set speed/duplex
sudo ethtool -s eth0 speed 1000 duplex full autoneg off

# ip command
ip -s link show eth0  # Statistics
ip -s -s link show eth0  # Detailed statistics
ip addr show eth0
ip route show dev eth0
ip neigh show dev eth0  # ARP cache
```

### ARP Operations

```bash
# View ARP cache
ip neigh show
arp -n

# Add static ARP entry
sudo ip neigh add 192.168.1.100 lladdr 00:11:22:33:44:55 dev eth0

# Delete ARP entry
sudo ip neigh del 192.168.1.100 dev eth0

# Flush ARP cache
sudo ip neigh flush dev eth0
sudo ip neigh flush all

# arping (ARP ping)
sudo arping -I eth0 192.168.1.1
```

## DNS Configuration

### DNS Resolution Files

```bash
# /etc/hosts - Local DNS
192.168.1.100  server1.local server1
192.168.1.101  server2.local server2
127.0.0.1      localhost

# /etc/resolv.conf - DNS servers
nameserver 8.8.8.8
nameserver 8.8.4.4
search example.com
options timeout:2 attempts:3

# /etc/nsswitch.conf - Name service switch
hosts: files dns myhostname
```

### systemd-resolved

```bash
# Status
systemd-resolve --status
resolvectl status

# Query DNS
resolvectl query google.com
systemd-resolve google.com

# Flush cache
sudo resolvectl flush-caches

# Configuration
# /etc/systemd/resolved.conf
[Resolve]
DNS=8.8.8.8 8.8.4.4
FallbackDNS=1.1.1.1
Domains=example.com
```

### DNS Testing Tools

```bash
# dig (detailed)
dig google.com
dig @8.8.8.8 google.com  # Specific DNS server
dig google.com A  # A record
dig google.com AAAA  # IPv6
dig google.com MX  # Mail servers
dig google.com NS  # Name servers
dig +short google.com  # Short output
dig -x 8.8.8.8  # Reverse lookup

# nslookup
nslookup google.com
nslookup google.com 8.8.8.8

# host
host google.com
host -t MX google.com
host 8.8.8.8
```

## Traffic Control (QoS)

### Basic Traffic Shaping

```bash
# View qdisc (queuing discipline)
tc qdisc show dev eth0

# Add bandwidth limit
sudo tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms

# Delete qdisc
sudo tc qdisc del dev eth0 root

# HTB (Hierarchical Token Bucket)
sudo tc qdisc add dev eth0 root handle 1: htb default 30
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 100mbit
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 30mbit ceil 50mbit
sudo tc class add dev eth0 parent 1:1 classid 1:30 htb rate 20mbit ceil 30mbit
```

### Priority Queuing

```bash
# PRIO qdisc
sudo tc qdisc add dev eth0 root handle 1: prio bands 3

# Add filters
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 match ip dport 22 0xffff flowid 1:1
sudo tc filter add dev eth0 parent 1: protocol ip prio 2 u32 match ip dport 80 0xffff flowid 1:2
```

### Rate Limiting

```bash
# Limit ingress bandwidth
sudo tc qdisc add dev eth0 handle ffff: ingress
sudo tc filter add dev eth0 parent ffff: protocol ip prio 50 u32 match ip src 0.0.0.0/0 police rate 10mbit burst 10k drop flowid :1

# Limit egress bandwidth
sudo tc qdisc add dev eth0 root tbf rate 10mbit latency 50ms burst 10k
```

## Common Networking Patterns

### NAT Gateway Setup

```bash
# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf

# Configure NAT
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i eth1 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o eth1 -j ACCEPT

# Save rules
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

### Port Forwarding

```bash
# Forward external port 8080 to internal 192.168.1.100:80
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination 192.168.1.100:80
sudo iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 80 -m state --state NEW,ESTABLISHED,RELATED -j ACCEPT

# Multiple ports
sudo iptables -t nat -A PREROUTING -p tcp --dport 8080:8090 -j DNAT --to-destination 192.168.1.100:80-90
```

### Transparent Proxy

```bash
# Redirect HTTP traffic to proxy
sudo iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 80 -j REDIRECT --to-port 3128
sudo iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 443 -j REDIRECT --to-port 3129
```

### Network Isolation

```bash
# Create isolated networks with namespaces
for i in {1..3}; do
    sudo ip netns add ns$i
    sudo ip link add veth-host$i type veth peer name veth-ns$i
    sudo ip link set veth-ns$i netns ns$i
    sudo ip addr add 10.0.$i.1/24 dev veth-host$i
    sudo ip link set veth-host$i up
    sudo ip netns exec ns$i ip addr add 10.0.$i.2/24 dev veth-ns$i
    sudo ip netns exec ns$i ip link set veth-ns$i up
    sudo ip netns exec ns$i ip link set lo up
    sudo ip netns exec ns$i ip route add default via 10.0.$i.1
done
```

### Load Balancer Setup

```bash
# Using iptables for simple load balancing
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode nth --every 2 --packet 0 -j DNAT --to-destination 192.168.1.10:80
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -m statistic --mode nth --every 2 --packet 1 -j DNAT --to-destination 192.168.1.11:80
```

### Multi-homed System

```bash
# System with multiple network interfaces
# eth0: 192.168.1.0/24 (internal)
# eth1: 203.0.113.0/24 (external)

# Internal traffic uses eth0 table
sudo ip route add default via 192.168.1.1 dev eth0 table 100
sudo ip rule add from 192.168.1.0/24 table 100

# External traffic uses eth1 table
sudo ip route add default via 203.0.113.1 dev eth1 table 101
sudo ip rule add from 203.0.113.0/24 table 101

# Main table default
sudo ip route add default via 192.168.1.1
```

### Container Networking Pattern

```bash
# Create bridge for containers
sudo ip link add docker0 type bridge
sudo ip addr add 172.17.0.1/16 dev docker0
sudo ip link set docker0 up

# Create container namespace
sudo ip netns add container1
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth1 netns container1

# Configure
sudo ip addr add 172.17.0.2/16 dev veth0
sudo ip link set veth0 master docker0
sudo ip link set veth0 up

sudo ip netns exec container1 ip addr add 172.17.0.3/16 dev veth1
sudo ip netns exec container1 ip link set veth1 up
sudo ip netns exec container1 ip link set lo up
sudo ip netns exec container1 ip route add default via 172.17.0.1

# NAT for containers
sudo iptables -t nat -A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE
```

## Performance Tuning

### Sysctl Network Parameters

```bash
# View all network parameters
sysctl -a | grep net

# TCP buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 67108864"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 67108864"

# Connection tracking
sudo sysctl -w net.netfilter.nf_conntrack_max=1000000
sudo sysctl -w net.nf_conntrack_max=1000000

# TCP optimization
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.ipv4.tcp_fastopen=3
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Socket backlog
sudo sysctl -w net.core.somaxconn=4096
sudo sysctl -w net.core.netdev_max_backlog=5000

# Make permanent
cat <<EOF | sudo tee -a /etc/sysctl.conf
# Network Performance Tuning
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 65536 67108864
net.ipv4.tcp_congestion_control=bbr
net.core.somaxconn=4096
EOF

# Apply
sudo sysctl -p
```

### Interface Optimization

```bash
# Increase ring buffer
sudo ethtool -G eth0 rx 4096 tx 4096

# Enable/disable offloading
sudo ethtool -K eth0 tso on
sudo ethtool -K eth0 gso on
sudo ethtool -K eth0 gro on
sudo ethtool -K eth0 lro on

# Set interrupt coalescence
sudo ethtool -C eth0 adaptive-rx on adaptive-tx on

# RSS (Receive Side Scaling)
sudo ethtool -L eth0 combined 4
```

## Security Best Practices

### Firewall Hardening

```bash
# Strict INPUT policy
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Anti-spoofing
sudo iptables -A INPUT -s 10.0.0.0/8 -i eth0 -j DROP
sudo iptables -A INPUT -s 172.16.0.0/12 -i eth0 -j DROP
sudo iptables -A INPUT -s 192.168.0.0/16 -i eth0 -j DROP

# Block invalid packets
sudo iptables -A INPUT -m state --state INVALID -j DROP
sudo iptables -A FORWARD -m state --state INVALID -j DROP

# SYN flood protection
sudo iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT
sudo iptables -A INPUT -p tcp --syn -j DROP

# Port scan protection
sudo iptables -N port-scanning
sudo iptables -A port-scanning -p tcp --tcp-flags SYN,ACK,FIN,RST RST -m limit --limit 1/s --limit-burst 2 -j RETURN
sudo iptables -A port-scanning -j DROP
```

### Kernel Security Parameters

```bash
# Disable IP forwarding (unless needed)
sudo sysctl -w net.ipv4.ip_forward=0

# SYN cookies protection
sudo sysctl -w net.ipv4.tcp_syncookies=1

# Ignore ICMP redirects
sudo sysctl -w net.ipv4.conf.all.accept_redirects=0
sudo sysctl -w net.ipv6.conf.all.accept_redirects=0

# Ignore source routed packets
sudo sysctl -w net.ipv4.conf.all.accept_source_route=0

# Reverse path filtering
sudo sysctl -w net.ipv4.conf.all.rp_filter=1

# Log martian packets
sudo sysctl -w net.ipv4.conf.all.log_martians=1

# Disable ICMP echo
sudo sysctl -w net.ipv4.icmp_echo_ignore_all=1
```

### Network Monitoring

```bash
# Monitor connections
watch -n 1 'ss -s'
watch -n 1 'netstat -i'

# Monitor iptables
watch -n 1 'iptables -L -n -v'

# Log suspicious activity
sudo iptables -A INPUT -m state --state INVALID -j LOG --log-prefix "Invalid packet: "
sudo iptables -A INPUT -p tcp --tcp-flags ALL NONE -j LOG --log-prefix "NULL scan: "
sudo iptables -A INPUT -p tcp --tcp-flags ALL ALL -j LOG --log-prefix "XMAS scan: "
```

## Troubleshooting

### Connection Issues

```bash
# 1. Check interface status
ip link show
ip addr show
ethtool eth0

# 2. Check IP configuration
ip addr show eth0
ip route show

# 3. Check gateway reachability
ping -c 4 $(ip route | grep default | awk '{print $3}')

# 4. Check DNS
cat /etc/resolv.conf
dig google.com
nslookup google.com

# 5. Check firewall
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v

# 6. Check listening services
ss -tulpn

# 7. Test specific port
telnet 192.168.1.1 80
nc -zv 192.168.1.1 80
curl -v telnet://192.168.1.1:80
```

### Routing Problems

```bash
# Check routing table
ip route show
ip route get 8.8.8.8

# Check ARP
ip neigh show

# Traceroute to destination
traceroute -n 8.8.8.8
mtr -n 8.8.8.8

# Check for asymmetric routing
sudo tcpdump -i any -n host 8.8.8.8
```

### DNS Failures

```bash
# Test DNS resolution
dig google.com
nslookup google.com
host google.com

# Check DNS servers
cat /etc/resolv.conf
systemd-resolve --status

# Test specific DNS server
dig @8.8.8.8 google.com
dig @1.1.1.1 google.com

# Flush DNS cache
sudo systemd-resolve --flush-caches
sudo resolvectl flush-caches

# Check /etc/hosts
cat /etc/hosts
```

### Performance Issues

```bash
# Check interface errors
ip -s link show eth0
ethtool -S eth0 | grep -i error
ethtool -S eth0 | grep -i drop

# Check bandwidth usage
iftop -i eth0
nethogs eth0
nload eth0

# Check latency
ping -c 100 8.8.8.8 | tail -1
mtr -r -c 100 8.8.8.8

# Check MTU issues
ping -M do -s 1472 8.8.8.8  # Test path MTU
tracepath 8.8.8.8

# Monitor connections
ss -s
ss -tan | awk '{print $1}' | sort | uniq -c
```

### Packet Loss

```bash
# Check interface statistics
ip -s -s link show eth0
ethtool -S eth0

# Monitor drops
watch -n 1 'ip -s link show eth0'

# Test with different packet sizes
ping -s 100 8.8.8.8
ping -s 1000 8.8.8.8
ping -s 1400 8.8.8.8

# Capture and analyze
sudo tcpdump -i eth0 -w capture.pcap
```

## NetworkManager vs systemd-networkd

### NetworkManager

```bash
# Status
nmcli general status
nmcli device status
nmcli connection show

# Create connection
nmcli connection add type ethernet ifname eth0 con-name eth0-static \
  ipv4.addresses 192.168.1.100/24 \
  ipv4.gateway 192.168.1.1 \
  ipv4.dns "8.8.8.8 8.8.4.4" \
  ipv4.method manual

# Modify connection
nmcli connection modify eth0-static ipv4.addresses 192.168.1.101/24

# Activate/deactivate
nmcli connection up eth0-static
nmcli connection down eth0-static

# Delete connection
nmcli connection delete eth0-static

# WiFi
nmcli device wifi list
nmcli device wifi connect SSID password PASSWORD
```

### systemd-networkd

```bash
# Enable service
sudo systemctl enable systemd-networkd
sudo systemctl start systemd-networkd

# Configuration files: /etc/systemd/network/

# Static IP (/etc/systemd/network/10-eth0.network)
[Match]
Name=eth0

[Network]
Address=192.168.1.100/24
Gateway=192.168.1.1
DNS=8.8.8.8
DNS=8.8.4.4

# DHCP (/etc/systemd/network/20-dhcp.network)
[Match]
Name=en*

[Network]
DHCP=yes

# Restart to apply
sudo systemctl restart systemd-networkd

# Status
networkctl status
networkctl list
```

## Configuration File Locations

```bash
# Network interfaces
/etc/network/interfaces         # Debian/Ubuntu
/etc/sysconfig/network-scripts/ # RHEL/CentOS
/etc/netplan/                   # Ubuntu 18.04+
/etc/systemd/network/           # systemd-networkd

# DNS
/etc/resolv.conf               # DNS servers
/etc/hosts                     # Local DNS
/etc/nsswitch.conf            # Name service switch
/etc/systemd/resolved.conf    # systemd-resolved

# Firewall
/etc/iptables/rules.v4        # iptables rules
/etc/nftables.conf            # nftables rules
/etc/firewalld/               # firewalld config

# Network services
/etc/services                 # Port/service mappings
/etc/protocols               # Protocol definitions

# Routing
/etc/iproute2/rt_tables      # Routing table names
```

## Useful Scripts and Aliases

### Network Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

# Network status
alias netstat-listening='ss -tulpn'
alias netstat-all='ss -tupan'
alias netstat-summary='ss -s'

# Quick interface info
alias myip='ip -4 addr show | grep -oP "(?<=inet\s)\d+(\.\d+){3}"'
alias myips='ip addr show | grep "inet "'
alias gateway='ip route | grep default'

# DNS
alias dns='cat /etc/resolv.conf'
alias flushd='sudo systemd-resolve --flush-caches'

# Firewall
alias fw-list='sudo iptables -L -n -v'
alias fw-nat='sudo iptables -t nat -L -n -v'

# Monitoring
alias bandwidth='sudo iftop -i eth0'
alias connections='watch -n 1 "ss -s"'

# Network test
alias testnet='ping -c 4 8.8.8.8 && ping -c 4 google.com'
```

### Network Check Script

```bash
#!/bin/bash
# network-check.sh - Quick network diagnostics

echo "=== Network Interfaces ==="
ip -br addr show

echo -e "\n=== Default Gateway ==="
ip route show default

echo -e "\n=== DNS Servers ==="
cat /etc/resolv.conf | grep nameserver

echo -e "\n=== Gateway Reachability ==="
GATEWAY=$(ip route | grep default | awk '{print $3}')
ping -c 3 $GATEWAY

echo -e "\n=== Internet Connectivity ==="
ping -c 3 8.8.8.8

echo -e "\n=== DNS Resolution ==="
nslookup google.com | grep -A1 "Name:"

echo -e "\n=== Listening Ports ==="
ss -tulpn | grep LISTEN

echo -e "\n=== Active Connections ==="
ss -s
```

### Port Scanner Script

```bash
#!/bin/bash
# port-scan.sh - Simple port scanner

HOST=$1
START_PORT=${2:-1}
END_PORT=${3:-1024}

if [ -z "$HOST" ]; then
    echo "Usage: $0 <host> [start_port] [end_port]"
    exit 1
fi

echo "Scanning $HOST ports $START_PORT-$END_PORT..."

for port in $(seq $START_PORT $END_PORT); do
    timeout 1 bash -c "echo >/dev/tcp/$HOST/$port" 2>/dev/null && \
        echo "Port $port: OPEN"
done
```

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `ip addr show` | Show IP addresses |
| `ip link show` | Show network interfaces |
| `ip route show` | Show routing table |
| `ip neigh show` | Show ARP cache |
| `ss -tulpn` | Show listening ports |
| `ping` | Test connectivity |
| `traceroute` | Trace packet route |
| `dig` | DNS lookup |
| `tcpdump` | Capture packets |
| `iptables -L` | List firewall rules |
| `nmcli` | NetworkManager CLI |
| `ethtool` | Interface configuration |

### Common Operations

| Task | Command |
|------|---------|
| Add IP address | `sudo ip addr add 192.168.1.100/24 dev eth0` |
| Bring interface up | `sudo ip link set eth0 up` |
| Add default route | `sudo ip route add default via 192.168.1.1` |
| Flush routes | `sudo ip route flush dev eth0` |
| Show connections | `ss -tan` |
| Enable NAT | `sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE` |
| Port forward | `sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to 192.168.1.100:8080` |
| Create bridge | `sudo ip link add br0 type bridge` |
| Create VLAN | `sudo ip link add link eth0 name eth0.10 type vlan id 10` |
| Create namespace | `sudo ip netns add ns1` |

Linux networking provides robust, flexible network management suitable for everything from simple connectivity to complex enterprise networking scenarios.
