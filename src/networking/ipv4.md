# IPv4 (Internet Protocol version 4)

## Overview

IPv4 (Internet Protocol version 4) is the fourth version of the Internet Protocol and the first version to be widely deployed. It is the network layer protocol responsible for addressing and routing packets across networks, providing the addressing scheme that allows devices to find each other on the internet.

## Key Characteristics

| Feature | IPv4 |
|---------|------|
| **Address Size** | 32 bits |
| **Address Format** | Decimal dotted notation (192.168.1.1) |
| **Total Addresses** | ~4.3 billion (2³²) |
| **Header Size** | 20-60 bytes (variable) |
| **Checksum** | Yes (header checksum) |
| **Fragmentation** | By routers and source |
| **Broadcast** | Yes |
| **Configuration** | Manual or DHCP |
| **IPSec** | Optional |

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
   - Minimum: 5 (20 bytes)
   - Maximum: 15 (60 bytes)
3. **Type of Service** (8 bits): QoS, priority
   - Precedence (3 bits): Priority level
   - Delay, Throughput, Reliability (3 bits)
   - Used for traffic prioritization
4. **Total Length** (16 bits): Entire packet size including header (max 65,535 bytes)
5. **Identification** (16 bits): Fragment identification
   - All fragments of the same packet share this value
6. **Flags** (3 bits):
   - Bit 0: Reserved (must be 0)
   - Bit 1: Don't Fragment (DF) - prevents fragmentation
   - Bit 2: More Fragments (MF) - indicates more fragments follow
7. **Fragment Offset** (13 bits): Position of fragment in original packet (in 8-byte units)
8. **Time to Live (TTL)** (8 bits): Max hops (decremented at each router)
   - Prevents infinite routing loops
   - Typical initial values: 64 (Linux), 128 (Windows), 255 (Cisco)
9. **Protocol** (8 bits): Upper layer protocol
   - 1 = ICMP
   - 6 = TCP
   - 17 = UDP
10. **Header Checksum** (16 bits): Error detection for header only
    - Recalculated at each hop (because TTL changes)
11. **Source Address** (32 bits): Sender IPv4 address
12. **Destination Address** (32 bits): Receiver IPv4 address
13. **Options** (variable): Rarely used today
    - Security, timestamps, route recording, source routing

## IPv4 Address Classes

### Traditional Class System (Obsolete, replaced by CIDR)

The classful addressing system divided the IPv4 address space into five classes (A-E), but this system was wasteful and is now obsolete. It's still useful to understand for historical reasons.

```
Class A: 0.0.0.0     to 127.255.255.255   /8   (16,777,214 hosts)
         Network: 8 bits, Host: 24 bits
         First bit: 0
         Example: 10.0.0.0/8

Class B: 128.0.0.0   to 191.255.255.255   /16  (65,534 hosts)
         Network: 16 bits, Host: 16 bits
         First two bits: 10
         Example: 172.16.0.0/16

Class C: 192.0.0.0   to 223.255.255.255   /24  (254 hosts)
         Network: 24 bits, Host: 8 bits
         First three bits: 110
         Example: 192.168.1.0/24

Class D: 224.0.0.0   to 239.255.255.255   (Multicast)
         First four bits: 1110
         Used for multicast groups

Class E: 240.0.0.0   to 255.255.255.255   (Reserved)
         First four bits: 1111
         Reserved for experimental use
```

### Why Classes Were Abandoned

```
Problem: Wasteful allocation
- Small company needs 300 hosts
  - Class C (/24): Only 254 hosts (too small)
  - Class B (/16): 65,534 hosts (massive waste)

Solution: CIDR (Classless Inter-Domain Routing)
- Flexible subnet sizes
- Better address utilization
```

## Private IP Address Ranges

Private IP addresses are reserved for use in private networks and are not routed on the public internet. Network Address Translation (NAT) is required to access the internet.

```
10.0.0.0        - 10.255.255.255     (10.0.0.0/8)
                                      16,777,216 addresses
                                      Typically used in large enterprises

172.16.0.0      - 172.31.255.255     (172.16.0.0/12)
                                      1,048,576 addresses
                                      Medium-sized networks

192.168.0.0     - 192.168.255.255    (192.168.0.0/16)
                                      65,536 addresses
                                      Home and small office networks
```

### Advantages of Private Addresses

1. **Address Conservation**: Reuse addresses across different private networks
2. **Security**: Not directly accessible from the internet
3. **Flexibility**: Can use any addressing scheme internally
4. **Cost**: No need to purchase public IP addresses

## Special IPv4 Addresses

```
0.0.0.0/8         - Current network (only valid as source)
                    Used during boot before IP is configured

127.0.0.0/8       - Loopback addresses
                    127.0.0.1 = localhost (most common)
                    Packets sent to loopback never leave the host

169.254.0.0/16    - Link-local addresses (APIPA)
                    Auto-assigned when DHCP fails
                    169.254.0.0 and 169.254.255.255 reserved

192.0.2.0/24      - Documentation/examples (TEST-NET-1)
198.51.100.0/24   - Documentation (TEST-NET-2)
203.0.113.0/24    - Documentation (TEST-NET-3)
                    Safe to use in documentation, never routed

192.88.99.0/24    - 6to4 Relay Anycast (IPv6 transition)

198.18.0.0/15     - Benchmark testing
                    Network device testing

224.0.0.0/4       - Multicast (Class D)
                    224.0.0.0 - 239.255.255.255

255.255.255.255   - Limited broadcast
                    Sent to all hosts on local network segment
```

## CIDR (Classless Inter-Domain Routing)

CIDR replaced the classful addressing system, providing flexible subnetting and efficient address allocation.

### CIDR Notation

```
192.168.1.0/24
            ^^
            Number of network bits (subnet mask length)

/24 = 255.255.255.0 netmask
      24 bits for network, 8 bits for hosts
      2^8 = 256 total addresses
      2^8 - 2 = 254 usable host addresses
      (Network address and broadcast address reserved)
```

### CIDR Notation Breakdown

```
192.168.1.0/24

Binary:
11000000.10101000.00000001.00000000  (IP address)
11111111.11111111.11111111.00000000  (Subnet mask /24)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
24 network bits (1s)        ^^^^^^^^
                            8 host bits (0s)

Network portion: 192.168.1
Host portion: 0-255
```

### Common Subnet Masks

| CIDR | Netmask | Wildcard | Total | Usable | Use Case |
|------|---------|----------|-------|--------|----------|
| **/8** | 255.0.0.0 | 0.255.255.255 | 16,777,216 | 16,777,214 | Huge networks (Class A) |
| **/12** | 255.240.0.0 | 0.15.255.255 | 1,048,576 | 1,048,574 | Large ISPs |
| **/16** | 255.255.0.0 | 0.0.255.255 | 65,536 | 65,534 | Large networks (Class B) |
| **/20** | 255.255.240.0 | 0.0.15.255 | 4,096 | 4,094 | Medium businesses |
| **/24** | 255.255.255.0 | 0.0.0.255 | 256 | 254 | Small networks (Class C) |
| **/25** | 255.255.255.128 | 0.0.0.127 | 128 | 126 | Subnet split |
| **/26** | 255.255.255.192 | 0.0.0.63 | 64 | 62 | Small subnet |
| **/27** | 255.255.255.224 | 0.0.0.31 | 32 | 30 | Very small |
| **/28** | 255.255.255.240 | 0.0.0.15 | 16 | 14 | Tiny subnet |
| **/29** | 255.255.255.248 | 0.0.0.7 | 8 | 6 | Minimal |
| **/30** | 255.255.255.252 | 0.0.0.3 | 4 | 2 | Point-to-point links |
| **/31** | 255.255.255.254 | 0.0.0.1 | 2 | 2 | Point-to-point (RFC 3021) |
| **/32** | 255.255.255.255 | 0.0.0.0 | 1 | 1 | Single host route |

### Subnet Calculation Example

```
Network: 192.168.1.0/24

Binary calculation:
IP:      11000000.10101000.00000001.00000000
Mask:    11111111.11111111.11111111.00000000
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Network bits
                                    ^^^^^^^^ Host bits

Network Address:    192.168.1.0      (all host bits = 0)
First Usable:       192.168.1.1      (first host)
Last Usable:        192.168.1.254    (last host)
Broadcast Address:  192.168.1.255    (all host bits = 1)

Total Addresses:    2^8 = 256
Usable Hosts:       256 - 2 = 254
                    (exclude network and broadcast)
```

### Subnet Mask Calculation

To calculate subnet mask from CIDR:

```
/24 in binary:
11111111.11111111.11111111.00000000
^^^^^^^^ ^^^^^^^^ ^^^^^^^^ ^^^^^^^^
255      255      255      0

/26 in binary:
11111111.11111111.11111111.11000000
^^^^^^^^ ^^^^^^^^ ^^^^^^^^ ^^^^^^^^
255      255      255      192

Shortcut:
/24 = 256 - 2^(32-24) = 256 - 2^8 = 256 - 256 = 0 (last octet)
/26 = 256 - 2^(32-26) = 256 - 2^6 = 256 - 64 = 192 (last octet)
```

### Subnetting Example

```
Original Network: 192.168.1.0/24 (254 usable hosts)

Requirement: Split into 4 equal subnets

Calculation:
- Need 4 subnets = 2^2 subnets
- Borrow 2 bits from host portion
- New mask: /24 + 2 = /26
- Each subnet: 2^6 = 64 addresses, 62 usable

Result:
Subnet 1: 192.168.1.0/26
  Network:    192.168.1.0
  First Host: 192.168.1.1
  Last Host:  192.168.1.62
  Broadcast:  192.168.1.63

Subnet 2: 192.168.1.64/26
  Network:    192.168.1.64
  First Host: 192.168.1.65
  Last Host:  192.168.1.126
  Broadcast:  192.168.1.127

Subnet 3: 192.168.1.128/26
  Network:    192.168.1.128
  First Host: 192.168.1.129
  Last Host:  192.168.1.190
  Broadcast:  192.168.1.191

Subnet 4: 192.168.1.192/26
  Network:    192.168.1.192
  First Host: 192.168.1.193
  Last Host:  192.168.1.254
  Broadcast:  192.168.1.255
```

### Variable Length Subnet Masking (VLSM)

VLSM allows different subnet sizes within the same network:

```
Main Network: 10.0.0.0/8

Allocations:
Department A (needs 500 hosts): 10.0.0.0/23   (510 hosts)
Department B (needs 200 hosts): 10.0.2.0/24   (254 hosts)
Department C (needs 100 hosts): 10.0.3.0/25   (126 hosts)
Point-to-point link:             10.0.3.128/30 (2 hosts)
Server subnet:                   10.0.4.0/28   (14 hosts)

Benefits:
- Efficient address utilization
- Minimizes waste
- Flexible network design
```

## IP Fragmentation

### Why Fragmentation?

Every network has a Maximum Transmission Unit (MTU) that limits packet size:

```
Common MTU Values:
- Ethernet: 1500 bytes
- Wi-Fi: 2304 bytes
- PPPoE: 1492 bytes
- VPN: 1400 bytes (varies)
- Jumbo Frames: 9000 bytes

When packet > MTU:
- Must be fragmented to fit
- Or dropped if DF flag is set
```

### IPv4 Fragmentation Process

Fragmentation can occur at the source or any router along the path:

```
Original Packet: 3000 bytes data + 20 byte header = 3020 bytes
MTU: 1500 bytes
Data per fragment: 1500 - 20 (header) = 1480 bytes

Fragment 1:
  IP Header (20 bytes)
  Identification: 12345
  Flags: MF = 1 (More Fragments)
  Fragment Offset: 0
  Total Length: 1500
  Data: 1480 bytes

Fragment 2:
  IP Header (20 bytes)
  Identification: 12345
  Flags: MF = 1
  Fragment Offset: 185 (1480/8 = 185)
  Total Length: 1500
  Data: 1480 bytes

Fragment 3:
  IP Header (20 bytes)
  Identification: 12345
  Flags: MF = 0 (Last Fragment)
  Fragment Offset: 370 (2960/8 = 370)
  Total Length: 60
  Data: 40 bytes

Receiver:
1. Receives all three fragments
2. Checks Identification field (12345) to group them
3. Uses Fragment Offset to order them
4. Reassembles when MF = 0 (last fragment received)
```

### Fragment Offset Calculation

```
Fragment Offset is in 8-byte units:

Fragment 1: Offset 0     → Bytes 0-1479
Fragment 2: Offset 185   → Bytes 1480-2959 (185 × 8 = 1480)
Fragment 3: Offset 370   → Bytes 2960-2999 (370 × 8 = 2960)

Why 8-byte units?
- 13 bits for offset = max 8191
- 8191 × 8 = 65,528 bytes
- Covers max IP packet size (65,535 bytes)
```

### Don't Fragment (DF) Flag

```
DF = 0: Allow fragmentation
        Router can fragment if needed

DF = 1: Don't fragment
        Router drops packet if too large
        Sends ICMP "Fragmentation Needed" back to source

        ICMP Type 3, Code 4:
        - Includes MTU of the link
        - Source can adjust packet size

Used for Path MTU Discovery (PMTUD)
```

### Path MTU Discovery (PMTUD)

```
Process:
1. Source sends packet with DF=1 and large size
2. If too large, router drops and sends ICMP
3. Source reduces packet size and retries
4. Repeat until packet gets through
5. Source now knows the path MTU

Example:
Source → [MTU 1500] → Router A → [MTU 1400] → Router B → Dest

1. Send 1500-byte packet, DF=1
2. Router B drops it, sends ICMP: "Frag needed, MTU=1400"
3. Source retries with 1400-byte packets
4. Success! Path MTU = 1400
```

### Fragmentation Issues

```
Problems:
1. Performance overhead (reassembly)
2. Lost fragment = entire packet lost
3. Difficulty for firewalls/NAT
4. Security concerns (fragment attacks)

Best Practices:
- Avoid fragmentation when possible
- Use TCP MSS clamping
- Enable PMTUD
- Consider smaller packet sizes
```

## TTL (Time to Live)

### Purpose

TTL prevents routing loops by limiting packet lifetime:

```
Without TTL:
Router A → Router B → Router C → Router A → ...
Packet loops forever, congesting network

With TTL:
Source sets TTL = 64
Router 1: Decrements to 63
Router 2: Decrements to 62
...
Router 64: Decrements to 0
           → Drops packet
           → Sends ICMP "Time Exceeded" to source
```

### Common TTL Values

Different operating systems use different initial TTL values:

```
Operating System  Initial TTL
Linux             64
Windows           128
Cisco IOS         255
FreeBSD           64
Mac OS X          64
Solaris           255

Security Note:
Can fingerprint OS based on received TTL
Received TTL = Initial TTL - Hop Count
```

### TTL Example

```
Packet journey from Source to Destination:

Source (TTL=64)
  |
  v
Router 1 (TTL=63) → Decrements TTL
  |
  v
Router 2 (TTL=62) → Decrements TTL
  |
  v
Router 3 (TTL=61) → Decrements TTL
  |
  v
Destination (TTL=60) → Receives packet

Reverse calculation:
- Received packet with TTL=60
- If initial TTL was 64
- Hop count = 64 - 60 = 4 hops
```

### Traceroute Uses TTL

Traceroute maps network paths by manipulating TTL:

```
How traceroute works:

1. Send packet with TTL=1
   → First router decrements to 0
   → Router drops packet
   → Router sends ICMP "Time Exceeded"
   → We learn first router IP

2. Send packet with TTL=2
   → First router: TTL=1
   → Second router: TTL=0
   → Second router responds
   → We learn second router IP

3. Send packet with TTL=3
   → Continue until destination reached

Result: Map of all routers in path

Example output:
1  192.168.1.1     2ms
2  10.0.0.1        5ms
3  203.0.113.1     10ms
4  93.184.216.34   15ms (destination)
```

### Linux Traceroute Example

```bash
# Default (UDP probes)
traceroute google.com

# ICMP probes
traceroute -I google.com

# TCP SYN probes to port 80
traceroute -T -p 80 google.com

# Set max hops
traceroute -m 20 google.com

# Send 3 probes per hop (default)
traceroute -q 3 google.com
```

### Windows Tracert Example

```cmd
# ICMP probes (Windows default)
tracert google.com

# Set max hops
tracert -h 20 google.com

# Don't resolve addresses to hostnames
tracert -d google.com
```

## IP Routing

### Routing Decision Process

When a host needs to send an IP packet:

```
1. Determine if destination is local:
   - Perform bitwise AND of dest IP and subnet mask
   - Compare with local network address

   Example:
   Local IP: 192.168.1.100/24
   Dest IP:  192.168.1.50

   192.168.1.50 AND 255.255.255.0 = 192.168.1.0 (matches local network)
   → Send directly via ARP

2. If destination is remote:
   - Search routing table for matching route
   - Use longest prefix match algorithm
   - Forward to gateway for that route

3. If no specific route matches:
   - Use default gateway (0.0.0.0/0)

4. If no default gateway:
   - Destination unreachable error
```

### Example Routing Table

```
Destination     Gateway         Netmask         Interface   Metric
0.0.0.0         192.168.1.1     0.0.0.0         eth0        100    (Default route)
10.0.0.0        192.168.1.254   255.0.0.0       eth0        10     (Static route)
192.168.1.0     0.0.0.0         255.255.255.0   eth0        0      (Connected)
192.168.2.0     192.168.1.200   255.255.255.0   eth0        20     (Static route)
172.16.0.0      192.168.1.254   255.255.0.0     eth0        15     (Static route)
```

### Routing Table Lookup

```
Packet destination: 10.1.2.5

Routing table:
  0.0.0.0/0       → Gateway A    (Default route)
  10.0.0.0/8      → Gateway B    (Matches!)
  10.1.0.0/16     → Gateway C    (Matches! More specific)
  10.1.2.0/24     → Gateway D    (Matches! Most specific)
  192.168.1.0/24  → Local        (No match)

Longest Prefix Match Algorithm:
- All routes compared
- Most specific match wins (/24 > /16 > /8 > /0)
- Forward to Gateway D
```

### Viewing Routing Table

```bash
# Linux - traditional
route -n
netstat -rn

# Linux - modern
ip route show
ip route list

# Windows
route print
netstat -r

# Example output (Linux):
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         192.168.1.1     0.0.0.0         UG    100    0        0 eth0
192.168.1.0     0.0.0.0         255.255.255.0   U     0      0        0 eth0
```

## NAT (Network Address Translation)

### Why NAT?

```
Problem: IPv4 Address Exhaustion
- Only ~4.3 billion addresses
- Internet growth exceeded availability
- Need to conserve public IP addresses

Solution: NAT
- Private network uses private IPs (10.x, 172.16-31.x, 192.168.x)
- Single public IP shared by many devices
- Router translates between private and public
```

### How NAT Works

```
Private Network (192.168.1.0/24)
┌──────────────────────────────┐
│  PC1: 192.168.1.10           │
│  PC2: 192.168.1.11           │──→  NAT Router  ──→  Internet
│  PC3: 192.168.1.12           │     (Translates)      Public IP: 203.0.113.5
└──────────────────────────────┘

Outbound:
PC1 (192.168.1.10:5000) → NAT → Internet as (203.0.113.5:6000)

Inbound:
Internet → (203.0.113.5:6000) → NAT → PC1 (192.168.1.10:5000)

NAT maintains translation table to track connections
```

### NAT Types

#### 1. Static NAT (One-to-One)

```
One private IP ↔ One public IP

Configuration:
Private: 192.168.1.10  ↔  Public: 203.0.113.10
Private: 192.168.1.11  ↔  Public: 203.0.113.11

Use case:
- Web servers
- Mail servers
- Devices that need incoming connections
```

#### 2. Dynamic NAT (Many-to-Many)

```
Multiple private IPs ↔ Pool of public IPs

Configuration:
Private: 192.168.1.0/24
Public pool: 203.0.113.10 - 203.0.113.20

Connection:
PC1 (192.168.1.10) → Gets 203.0.113.10
PC2 (192.168.1.11) → Gets 203.0.113.11
PC3 (192.168.1.12) → Gets 203.0.113.12

When PC1 disconnects, 203.0.113.10 returns to pool
```

#### 3. PAT (Port Address Translation) / NAT Overload

Most common type, used in home routers:

```
Many private IPs ↔ Single public IP (different ports)

Translation table:
Internal IP:Port      External IP:Port    Remote IP:Port
192.168.1.10:5000  →  203.0.113.5:6000 →  8.8.8.8:53
192.168.1.11:5001  →  203.0.113.5:6001 →  1.1.1.1:443
192.168.1.12:5002  →  203.0.113.5:6002 →  93.184.216.34:80
192.168.1.10:5003  →  203.0.113.5:6003 →  142.250.185.46:443

Note: Same internal IP can have multiple external ports
```

#### 4. Port Forwarding (DNAT - Destination NAT)

Allow external connections to internal servers:

```
Configuration:
External: 203.0.113.5:80   → Internal: 192.168.1.20:80  (Web)
External: 203.0.113.5:443  → Internal: 192.168.1.20:443 (HTTPS)
External: 203.0.113.5:22   → Internal: 192.168.1.25:22  (SSH)
External: 203.0.113.5:3389 → Internal: 192.168.1.30:3389 (RDP)

Internet request to 203.0.113.5:80
→ Router forwards to 192.168.1.20:80
→ Web server responds
→ Router translates source back to 203.0.113.5:80
```

### NAT Translation Table Example

```
Protocol  Inside Local      Inside Global     Outside Local     Outside Global
TCP       192.168.1.10:5000 203.0.113.5:6000  8.8.8.8:53        8.8.8.8:53
TCP       192.168.1.11:5001 203.0.113.5:6001  1.1.1.1:443       1.1.1.1:443
TCP       192.168.1.10:5002 203.0.113.5:6002  93.184.216.34:80  93.184.216.34:80

Terminology:
- Inside Local: Private IP (before NAT)
- Inside Global: Public IP (after NAT)
- Outside Local: Remote IP (before NAT)
- Outside Global: Remote IP (after NAT)
```

### NAT Configuration Examples

#### Linux (iptables)

```bash
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# Basic NAT (masquerade)
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Or with specific IP
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.5

# Port forwarding
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 \
  -j DNAT --to-destination 192.168.1.20:80

# View NAT table
iptables -t nat -L -v
```

#### Cisco Router

```cisco
! Enable NAT
interface GigabitEthernet0/0
  ip nat outside

interface GigabitEthernet0/1
  ip nat inside

! NAT overload (PAT)
ip nat inside source list 1 interface GigabitEthernet0/0 overload
access-list 1 permit 192.168.1.0 0.0.0.255

! Port forwarding
ip nat inside source static tcp 192.168.1.20 80 203.0.113.5 80

! View NAT translations
show ip nat translations
show ip nat statistics
```

### NAT Disadvantages

```
1. Breaks end-to-end connectivity
   - Some protocols don't work (FTP active mode, SIP, H.323)
   - Requires ALG (Application Layer Gateway) for some apps

2. Performance overhead
   - Translation takes CPU time
   - Maintains state tables

3. Complicates peer-to-peer
   - NAT traversal techniques needed (STUN, TURN, ICE)

4. Hides internal topology
   - All traffic appears from one IP
   - Makes troubleshooting harder

5. Limited by port numbers
   - 65,535 ports per public IP
   - In practice, ~4000 concurrent connections
```

## ICMP (Internet Control Message Protocol)

ICMP is a network layer protocol used for diagnostics and error reporting. It's an integral part of IP.

### ICMP Message Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Type      |     Code      |          Checksum             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                         Message Body                          |
|                         (varies by type)                      |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Common ICMP Message Types

| Type | Code | Message | Description | Use |
|------|------|---------|-------------|-----|
| **0** | 0 | Echo Reply | Response to ping | ping response |
| **3** | 0 | Destination Network Unreachable | Cannot reach network | Routing error |
| **3** | 1 | Destination Host Unreachable | Cannot reach host | Host down/filtered |
| **3** | 2 | Destination Protocol Unreachable | Protocol not supported | Protocol error |
| **3** | 3 | Destination Port Unreachable | Port not listening | Port closed |
| **3** | 4 | Fragmentation Needed and DF Set | MTU exceeded with DF=1 | PMTUD |
| **3** | 13 | Communication Administratively Prohibited | Filtered by firewall | ACL/firewall |
| **5** | 0-3 | Redirect | Better route available | Route optimization |
| **8** | 0 | Echo Request | Ping request | ping |
| **11** | 0 | Time Exceeded in Transit | TTL reached 0 | traceroute |
| **11** | 1 | Fragment Reassembly Time Exceeded | Fragments timeout | Fragmentation issue |
| **12** | 0 | Parameter Problem | IP header error | Malformed packet |

### Ping (ICMP Echo Request/Reply)

Ping tests connectivity and measures round-trip time:

```
Client                          Server
  |                                |
  | ICMP Echo Request (Type 8)     |
  |   Identifier: 1234             |
  |   Sequence: 1                  |
  |   Data: 56 bytes               |
  |------------------------------->|
  |                                |
  | ICMP Echo Reply (Type 0)       |
  |   Identifier: 1234             |
  |   Sequence: 1                  |
  |   Data: 56 bytes (echoed)      |
  |<-------------------------------|
  |                                |

Round-Trip Time (RTT) measured
```

### Ping Examples

```bash
# Basic ping
ping 8.8.8.8

# Send specific number of packets
ping -c 4 8.8.8.8

# Set packet size
ping -s 1000 8.8.8.8

# Set interval (0.2 seconds)
ping -i 0.2 8.8.8.8

# Flood ping (requires root)
sudo ping -f 8.8.8.8

# Set TTL
ping -t 5 8.8.8.8

# Disable DNS resolution
ping -n 8.8.8.8

# Example output:
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=10.2 ms
64 bytes from 8.8.8.8: icmp_seq=2 ttl=117 time=9.8 ms
64 bytes from 8.8.8.8: icmp_seq=3 ttl=117 time=10.1 ms

--- 8.8.8.8 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 9.8/10.0/10.2/0.2 ms
```

### ICMP in Traceroute

```
Traceroute sends packets with increasing TTL:

Packet 1: TTL=1
  → Router 1 decrements to 0
  → Router 1 sends ICMP Type 11 (Time Exceeded)
  → Reveals Router 1 IP

Packet 2: TTL=2
  → Router 1: TTL=1
  → Router 2: TTL=0
  → Router 2 sends ICMP Type 11
  → Reveals Router 2 IP

Packet N: TTL=N
  → Destination reached
  → Sends ICMP Type 3 (Port Unreachable) or Echo Reply
  → Traceroute completes
```

## IPv4 Commands and Tools

### ifconfig / ip (Linux)

```bash
# View IP configuration (old style)
ifconfig

# View IP configuration (modern)
ip addr show
ip a

# Show specific interface
ip addr show eth0

# Assign IP address (temporary)
sudo ip addr add 192.168.1.100/24 dev eth0

# Remove IP address
sudo ip addr del 192.168.1.100/24 dev eth0

# Enable interface
sudo ip link set eth0 up

# Disable interface
sudo ip link set eth0 down

# Show interface statistics
ip -s link show eth0
```

### ipconfig (Windows)

```cmd
# View IP configuration
ipconfig

# View detailed configuration
ipconfig /all

# Renew DHCP lease
ipconfig /renew

# Release DHCP lease
ipconfig /release

# Flush DNS cache
ipconfig /flushdns

# Display DNS cache
ipconfig /displaydns
```

### ip route (Linux)

```bash
# Show routing table
ip route show
ip route list

# Add static route
sudo ip route add 10.0.0.0/8 via 192.168.1.254

# Add route via specific interface
sudo ip route add 10.0.0.0/8 dev eth0

# Delete route
sudo ip route del 10.0.0.0/8

# Add default gateway
sudo ip route add default via 192.168.1.1

# Delete default gateway
sudo ip route del default

# Change route
sudo ip route change 10.0.0.0/8 via 192.168.1.253

# Show route to specific destination
ip route get 8.8.8.8
```

### route (Linux/Windows)

```bash
# Linux - show routing table
route -n

# Linux - add route
sudo route add -net 10.0.0.0/8 gw 192.168.1.254

# Linux - delete route
sudo route del -net 10.0.0.0/8

# Windows - show routing table
route print

# Windows - add route
route add 10.0.0.0 mask 255.0.0.0 192.168.1.254

# Windows - delete route
route delete 10.0.0.0

# Windows - add persistent route
route -p add 10.0.0.0 mask 255.0.0.0 192.168.1.254
```

### arp (Address Resolution Protocol)

```bash
# View ARP cache
arp -a

# View ARP cache for specific interface (Linux)
arp -i eth0

# Add static ARP entry (Linux)
sudo arp -s 192.168.1.50 00:11:22:33:44:55

# Delete ARP entry (Linux)
sudo arp -d 192.168.1.50

# View ARP cache (modern Linux)
ip neigh show

# Delete ARP entry (modern Linux)
sudo ip neigh del 192.168.1.50 dev eth0
```

## IPv4 Best Practices

### 1. Subnet Design

```
Plan network hierarchy:

Organization: 10.0.0.0/8
├── Location A: 10.1.0.0/16
│   ├── Servers: 10.1.1.0/24
│   ├── Workstations: 10.1.2.0/24
│   └── Guests: 10.1.3.0/24
├── Location B: 10.2.0.0/16
│   ├── Servers: 10.2.1.0/24
│   └── Workstations: 10.2.2.0/24
└── Management: 10.255.0.0/16
    ├── Network Devices: 10.255.1.0/24
    └── Out-of-band: 10.255.2.0/24

Benefits:
- Logical organization
- Summarization for routing
- Security segmentation
- Growth flexibility
```

### 2. IP Address Allocation

```
Reserve ranges within each subnet:

Example subnet: 192.168.1.0/24

192.168.1.0         Network address (reserved)
192.168.1.1         Gateway (router)
192.168.1.2-10      Infrastructure (switches, APs)
192.168.1.11-50     Servers (static)
192.168.1.51-99     Printers/IoT (static)
192.168.1.100-254   DHCP pool (dynamic)
192.168.1.255       Broadcast address (reserved)

Document everything in IPAM (IP Address Management) system
```

### 3. Use Private IP Ranges

```
ALWAYS use private IPs internally:

Small networks:    192.168.x.0/24
Medium networks:   172.16.x.0/16 to 172.31.x.0/16
Large networks:    10.0.0.0/8

NEVER use:
- Public IPs internally (causes routing issues)
- TEST-NET ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24)
- Multicast ranges (224.0.0.0/4)
```

### 4. Network Documentation

```
Maintain detailed documentation:

Network Diagram:
- Physical topology
- Logical topology
- IP addressing scheme
- VLAN assignments

Spreadsheet/IPAM:
IP Address    | Hostname    | MAC Address       | Type   | Notes
192.168.1.1   | gateway     | 00:11:22:33:44:55 | Router | Primary gateway
192.168.1.10  | server1     | 00:11:22:33:44:66 | Server | Web server
192.168.1.11  | server2     | 00:11:22:33:44:77 | Server | Database
192.168.1.50  | printer1    | 00:11:22:33:44:88 | Printer| HP LaserJet
```

### 5. DHCP Configuration

```
DHCP best practices:

- Appropriate lease time:
  * Office: 8-24 hours
  * Guest: 1-4 hours
  * Mobile: 30-60 minutes

- Reserve space for static IPs

- Configure DHCP options:
  * Option 3: Default gateway
  * Option 6: DNS servers
  * Option 15: Domain name
  * Option 42: NTP servers

- Redundant DHCP servers (split scope or failover)

- Monitor DHCP scope utilization
```

### 6. Network Security

```
Security measures:

1. Subnetting for segmentation
   - Separate user, server, management networks
   - Use VLANs

2. Private IPs + NAT
   - Hide internal topology
   - Conserve public IPs

3. Disable unused services
   - No ICMP redirect
   - No source routing

4. Ingress/egress filtering
   - Block spoofed source IPs
   - RFC 3330 filtering

5. Monitor for IP conflicts
   - Detect ARP spoofing
   - DHCP snooping
```

### 7. Avoid IP Conflicts

```
Prevention:

1. Use DHCP for workstations
2. Static IPs for servers/infrastructure
3. Document all static assignments
4. Configure DHCP exclusions for static range
5. Use DHCP reservations for semi-static hosts
6. Enable IP conflict detection

Detection:
- arping before assigning static IP
- Monitor DHCP logs
- Use network scanning tools
- Enable DHCP conflict detection
```

## ELI10: IPv4 Explained Simply

Think of IPv4 addresses like street addresses for computers:

### IPv4 Address (192.168.1.100)
- Like a home address with 4 numbers
- Each number is between 0 and 255
- Separated by dots
- Uniquely identifies your computer on the network

### Why 4 Numbers?
```
Each number is 0-255 (256 possibilities)
256 × 256 × 256 × 256 = 4.3 billion addresses

Problem: We almost ran out!
- Too many computers, phones, tablets
- Solution: NAT (share one public address)
- Future: IPv6 (way more addresses)
```

### Private vs Public
```
Private IPs (like apartment numbers):
- 192.168.x.x (home networks)
- 10.x.x.x (big companies)
- Only work inside your building (network)

Public IPs (like street addresses):
- Work on the internet
- Must be unique worldwide
- Expensive and limited
```

### Subnets
```
Like organizing streets into neighborhoods:

City: 10.0.0.0/8 (whole city)
  └─ Neighborhood: 10.1.0.0/16 (one area)
      └─ Street: 10.1.1.0/24 (one street)
          └─ House: 10.1.1.100 (your house)

/24 means: First 3 numbers are the "street", last number is your "house number"
```

### NAT (Sharing One Address)
```
Your home:
- Router has public IP: 203.0.113.5 (street address)
- Devices have private IPs: 192.168.1.x (apartment numbers)
- Router is like mailroom: forwards mail to right apartment
```

### Routing
```
Routers are like mail sorting facilities:
- Look at destination address
- Decide which direction to send packet
- Pass to next router
- Repeat until destination reached
```

## Further Resources

- [RFC 791 - IPv4 Specification](https://tools.ietf.org/html/rfc791)
- [RFC 1918 - Private Address Space](https://tools.ietf.org/html/rfc1918)
- [RFC 950 - Internet Standard Subnetting Procedure](https://tools.ietf.org/html/rfc950)
- [RFC 1812 - Requirements for IPv4 Routers](https://tools.ietf.org/html/rfc1812)
- [RFC 3021 - Using 31-Bit Prefixes on IPv4 Point-to-Point Links](https://tools.ietf.org/html/rfc3021)
- [Subnet Calculator](https://www.subnet-calculator.com/)
- [CIDR to IPv4 Conversion](https://www.ipaddressguide.com/cidr)
- [IANA IPv4 Address Space Registry](https://www.iana.org/assignments/ipv4-address-space/)
