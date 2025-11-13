# IP (Internet Protocol)

## Overview

IP (Internet Protocol) is the network layer protocol responsible for addressing and routing packets across networks. It provides the addressing scheme that allows devices to find each other on the internet.

## IP Versions

| Feature | IPv4 | IPv6 |
|---------|------|------|
| **Address Size** | 32 bits | 128 bits |
| **Address Format** | Decimal (192.168.1.1) | Hexadecimal (2001:db8::1) |
| **Total Addresses** | ~4.3 billion | 340 undecillion |
| **Header Size** | 20-60 bytes | 40 bytes (fixed) |
| **Checksum** | Yes | No (delegated to link layer) |
| **Fragmentation** | By routers | Source only |
| **Broadcast** | Yes | No (uses multicast) |
| **Configuration** | Manual or DHCP | SLAAC or DHCPv6 |
| **IPSec** | Optional | Mandatory |

## IPv4 Packet Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version|  IHL  |Type of Service|          Total Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Identification        |Flags|      Fragment Offset    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Time to Live |    Protocol   |         Header Checksum       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source Address                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination Address                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Options (if IHL > 5)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Data                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### IPv4 Header Fields

1. **Version** (4 bits): IP version (4 for IPv4)
2. **IHL** (4 bits): Internet Header Length (5-15, in 32-bit words)
3. **Type of Service** (8 bits): QoS, priority
4. **Total Length** (16 bits): Entire packet size (max 65,535 bytes)
5. **Identification** (16 bits): Fragment identification
6. **Flags** (3 bits):
   - Bit 0: Reserved (must be 0)
   - Bit 1: Don't Fragment (DF)
   - Bit 2: More Fragments (MF)
7. **Fragment Offset** (13 bits): Position of fragment
8. **Time to Live (TTL)** (8 bits): Max hops (decremented at each router)
9. **Protocol** (8 bits): Upper layer protocol (6=TCP, 17=UDP, 1=ICMP)
10. **Header Checksum** (16 bits): Error detection for header
11. **Source Address** (32 bits): Sender IP address
12. **Destination Address** (32 bits): Receiver IP address
13. **Options** (variable): Rarely used (security, timestamp, etc.)

## IPv4 Address Classes

### Traditional Class System (Obsolete, replaced by CIDR)

```
Class A: 0.0.0.0     to 127.255.255.255   /8   (16 million hosts)
         Network: 8 bits, Host: 24 bits

Class B: 128.0.0.0   to 191.255.255.255   /16  (65,536 hosts)
         Network: 16 bits, Host: 16 bits

Class C: 192.0.0.0   to 223.255.255.255   /24  (254 hosts)
         Network: 24 bits, Host: 8 bits

Class D: 224.0.0.0   to 239.255.255.255   (Multicast)

Class E: 240.0.0.0   to 255.255.255.255   (Reserved)
```

## Private IP Address Ranges

```
10.0.0.0        - 10.255.255.255     (10/8 prefix)
172.16.0.0      - 172.31.255.255     (172.16/12 prefix)
192.168.0.0     - 192.168.255.255    (192.168/16 prefix)

Used in LANs, not routed on internet (NAT required)
```

## Special IPv4 Addresses

```
0.0.0.0/8         - Current network (only valid as source)
127.0.0.0/8       - Loopback (127.0.0.1 = localhost)
169.254.0.0/16    - Link-local (APIPA, auto-config failed)
192.0.2.0/24      - Documentation/examples (TEST-NET-1)
198.18.0.0/15     - Benchmark testing
224.0.0.0/4       - Multicast
255.255.255.255   - Limited broadcast
```

## CIDR (Classless Inter-Domain Routing)

### CIDR Notation

```
192.168.1.0/24
            ^^
            Number of network bits

/24 = 255.255.255.0 netmask
24 bits for network, 8 bits for hosts
2^8 - 2 = 254 usable host addresses
```

### Common Subnet Masks

| CIDR | Netmask | Hosts | Use Case |
|------|---------|-------|----------|
| **/8** | 255.0.0.0 | 16,777,214 | Huge networks |
| **/16** | 255.255.0.0 | 65,534 | Large networks |
| **/24** | 255.255.255.0 | 254 | Small networks |
| **/25** | 255.255.255.128 | 126 | Subnet split |
| **/26** | 255.255.255.192 | 62 | Small subnet |
| **/27** | 255.255.255.224 | 30 | Very small |
| **/30** | 255.255.255.252 | 2 | Point-to-point |
| **/32** | 255.255.255.255 | 1 | Single host |

### Subnet Calculation Example

```
Network: 192.168.1.0/24

Network Address:    192.168.1.0
First Usable:       192.168.1.1
Last Usable:        192.168.1.254
Broadcast Address:  192.168.1.255
Total Hosts:        256
Usable Hosts:       254
```

### Subnetting Example

```
Original: 192.168.1.0/24 (254 hosts)

Split into 4 subnets (/26 each):
Subnet 1: 192.168.1.0/26    (192.168.1.1 - 192.168.1.62)
Subnet 2: 192.168.1.64/26   (192.168.1.65 - 192.168.1.126)
Subnet 3: 192.168.1.128/26  (192.168.1.129 - 192.168.1.190)
Subnet 4: 192.168.1.192/26  (192.168.1.193 - 192.168.1.254)

Each subnet: 62 usable hosts
```

## IPv6 Packet Format

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version| Traffic Class |           Flow Label                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Payload Length        |  Next Header  |   Hop Limit   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                         Source Address                        +
|                          (128 bits)                           |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                      Destination Address                      +
|                          (128 bits)                           |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### IPv6 Header Fields (40 bytes fixed)

1. **Version** (4 bits): IP version (6)
2. **Traffic Class** (8 bits): QoS, similar to ToS in IPv4
3. **Flow Label** (20 bits): QoS flow identification
4. **Payload Length** (16 bits): Data length (excluding header)
5. **Next Header** (8 bits): Protocol type (like IPv4 Protocol field)
6. **Hop Limit** (8 bits): Like IPv4 TTL
7. **Source Address** (128 bits)
8. **Destination Address** (128 bits)

## IPv6 Address Format

### Full Representation

```
2001:0db8:0000:0042:0000:8a2e:0370:7334
```

### Compressed Representation

```
# Remove leading zeros
2001:db8:0:42:0:8a2e:370:7334

# Replace consecutive zeros with ::
2001:db8:0:42::8a2e:370:7334

# Loopback
::1  (equivalent to 0:0:0:0:0:0:0:1)

# Unspecified
::  (equivalent to 0:0:0:0:0:0:0:0)
```

### IPv6 Address Types

| Type | Prefix | Example | Purpose |
|------|--------|---------|---------|
| **Global Unicast** | 2000::/3 | 2001:db8::1 | Internet routing |
| **Link-Local** | fe80::/10 | fe80::1 | Local network only |
| **Unique Local** | fc00::/7 | fd00::1 | Private (like RFC 1918) |
| **Multicast** | ff00::/8 | ff02::1 | One-to-many |
| **Loopback** | ::1/128 | ::1 | Localhost |
| **Unspecified** | ::/128 | :: | No address |

### Common IPv6 Multicast Addresses

```
ff02::1    All nodes on link
ff02::2    All routers on link
ff02::1:2  All DHCP servers
```

## IP Fragmentation

### Why Fragmentation?

```
MTU (Maximum Transmission Unit) varies by network:
- Ethernet: 1500 bytes
- WiFi: 2304 bytes
- PPPoE: 1492 bytes

Larger packets must be fragmented to fit MTU
```

### IPv4 Fragmentation Process

```
Original packet: 3000 bytes (1500 MTU)

Fragment 1:
  Identification: 12345
  Flags: More Fragments (MF) = 1
  Offset: 0
  Data: 1480 bytes

Fragment 2:
  Identification: 12345
  Flags: MF = 1
  Offset: 185 (1480/8)
  Data: 1480 bytes

Fragment 3:
  Identification: 12345
  Flags: MF = 0 (last fragment)
  Offset: 370 (2960/8)
  Data: 40 bytes

Receiver reassembles using Identification and Offset
```

### Don't Fragment (DF) Flag

```
DF = 1: Don't fragment, drop if too large
       Send ICMP "Fragmentation Needed" back

Used for Path MTU Discovery
```

## IP Routing

### Routing Decision Process

```
1. Check if destination is local (same subnet)
   ’ Send directly via ARP

2. If not local, find matching route:
   - Check routing table for most specific match
   - Use default gateway if no match

3. Send to next hop router

4. Repeat at each router until destination reached
```

### Example Routing Table

```
Destination     Gateway         Netmask         Interface
0.0.0.0         192.168.1.1     0.0.0.0         eth0      (Default)
192.168.1.0     0.0.0.0         255.255.255.0   eth0      (Local)
10.0.0.0        192.168.1.254   255.0.0.0       eth0      (Route)
```

### Longest Prefix Match

```
Routing table:
  10.0.0.0/8      ’ Gateway A
  10.1.0.0/16     ’ Gateway B
  10.1.2.0/24     ’ Gateway C

Packet to 10.1.2.5:
  Matches all three routes
  Most specific: /24
  ’ Use Gateway C
```

## TTL (Time to Live)

### Purpose

Prevents routing loops by limiting packet lifetime:

```
Source sets TTL = 64

Router 1: TTL = 63
Router 2: TTL = 62
Router 3: TTL = 61
...
Router N: TTL = 0 ’ Drop packet, send ICMP "Time Exceeded"
```

### Common TTL Values

```
Linux:    64
Windows:  128
Cisco:    255

Can identify OS based on initial TTL
```

### Traceroute Uses TTL

```
Send packet with TTL=1  ’ Router 1 responds
Send packet with TTL=2  ’ Router 2 responds
Send packet with TTL=3  ’ Router 3 responds
...
Maps the path to destination
```

## IP Commands and Tools

### ifconfig / ip (Linux)

```bash
# View IP configuration
ifconfig
ip addr show

# Assign IP address
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0
sudo ip addr add 192.168.1.100/24 dev eth0

# Enable/disable interface
sudo ifconfig eth0 up
sudo ip link set eth0 up
```

### ipconfig (Windows)

```cmd
# View IP configuration
ipconfig
ipconfig /all

# Renew DHCP lease
ipconfig /renew

# Release DHCP lease
ipconfig /release
```

### ping

```bash
# Test connectivity (ICMP Echo Request/Reply)
ping 192.168.1.1
ping -c 4 192.168.1.1  # Send 4 packets

# Test with specific packet size
ping -s 1400 192.168.1.1

# Set TTL
ping -t 10 192.168.1.1
```

### traceroute / tracert

```bash
# Linux
traceroute google.com

# Windows
tracert google.com

# UDP traceroute (Linux)
traceroute -U google.com

# ICMP traceroute
traceroute -I google.com
```

### netstat

```bash
# Show routing table
netstat -r
route -n

# Show all connections
netstat -an

# Show listening ports
netstat -ln
```

### ip route

```bash
# Show routing table
ip route show

# Add static route
sudo ip route add 10.0.0.0/8 via 192.168.1.254

# Delete route
sudo ip route del 10.0.0.0/8

# Add default gateway
sudo ip route add default via 192.168.1.1
```

## NAT (Network Address Translation)

### Why NAT?

```
Problem: IPv4 address exhaustion
Solution: Multiple private IPs share one public IP

Private Network (192.168.1.0/24)
  PC1: 192.168.1.10
  PC2: 192.168.1.11      ’  NAT Router  ’  Public IP: 203.0.113.5
  PC3: 192.168.1.12              “
                          Tracks connections
```

### NAT Types

#### 1. Source NAT (SNAT)

```
Outbound translation:
PC (192.168.1.10:5000) ’ NAT ’ Internet (203.0.113.5:6000)

Return traffic:
Internet (203.0.113.5:6000) ’ NAT ’ PC (192.168.1.10:5000)
```

#### 2. Destination NAT (DNAT) / Port Forwarding

```
Internet ’ Public IP:80 ’ NAT ’ Web Server (192.168.1.20:80)

External: 203.0.113.5:80
Internal: 192.168.1.20:80
```

#### 3. PAT (Port Address Translation) / NAT Overload

```
PC1: 192.168.1.10:5000  ’  203.0.113.5:6000
PC2: 192.168.1.11:5001  ’  203.0.113.5:6001
PC3: 192.168.1.12:5002  ’  203.0.113.5:6002

NAT tracks: Internal IP:Port ” Public Port
```

### NAT Table Example

```
Internal IP:Port     External IP:Port    Destination
192.168.1.10:5000    203.0.113.5:6000    8.8.8.8:53
192.168.1.11:5001    203.0.113.5:6001    1.1.1.1:443
192.168.1.10:5002    203.0.113.5:6002    93.184.216.34:80
```

## ICMP (Internet Control Message Protocol)

Part of IP suite, used for diagnostics and errors:

### Common ICMP Message Types

| Type | Code | Message | Use |
|------|------|---------|-----|
| **0** | 0 | Echo Reply | ping response |
| **3** | 0 | Dest Network Unreachable | Routing error |
| **3** | 1 | Dest Host Unreachable | Host down |
| **3** | 3 | Dest Port Unreachable | Port closed |
| **3** | 4 | Fragmentation Needed | MTU discovery |
| **8** | 0 | Echo Request | ping |
| **11** | 0 | Time Exceeded | TTL = 0 |
| **30** | 0 | Traceroute | Traceroute packet |

### Ping (ICMP Echo Request/Reply)

```
Client                          Server
  |                                |
  | ICMP Echo Request (Type 8)     |
  |------------------------------->|
  |                                |
  | ICMP Echo Reply (Type 0)       |
  |<-------------------------------|
  |                                |

Measures round-trip time (RTT)
```

## IP Best Practices

### 1. Subnet Properly

```
Don't use /24 for everything
- Small office: /26 (62 hosts)
- Department: /24 (254 hosts)
- Campus: /16 (65,534 hosts)
```

### 2. Reserve IP Ranges

```
192.168.1.1    - 192.168.1.10    Static (gateway, servers)
192.168.1.11   - 192.168.1.99    Static (printers, APs)
192.168.1.100  - 192.168.1.254   DHCP pool
```

### 3. Document Network

```
Maintain IP address management (IPAM)
- Which IPs are assigned
- What devices use them
- DHCP ranges
- Static assignments
```

### 4. Use Private IPs Internally

```
Never use public IPs internally
Use 10.x.x.x, 172.16-31.x.x, or 192.168.x.x
```

## ELI10

IP addresses are like street addresses for computers:

**IPv4 Address (192.168.1.100):**
- Like a home address with 4 numbers
- Each number is 0-255
- Almost ran out of addresses (like running out of street addresses in a city)

**IPv6 Address (2001:db8::1):**
- New address system with way more numbers
- Like adding ZIP+4 codes, apartment numbers, floor numbers
- So many addresses we'll never run out

**Routing:**
- Routers are like mail sorting facilities
- They look at the address and send the packet closer to its destination
- Each router knows which direction to send packets

**Private IPs:**
- Like apartment numbers (Apt 101, 102, 103)
- Work inside the building (local network)
- NAT is like the building's street address (everyone shares it for mail)

## Further Resources

- [RFC 791 - IPv4 Specification](https://tools.ietf.org/html/rfc791)
- [RFC 8200 - IPv6 Specification](https://tools.ietf.org/html/rfc8200)
- [RFC 1918 - Private Address Space](https://tools.ietf.org/html/rfc1918)
- [Subnet Calculator](https://www.subnet-calculator.com/)
- [CIDR to IPv4 Conversion](https://www.ipaddressguide.com/cidr)
