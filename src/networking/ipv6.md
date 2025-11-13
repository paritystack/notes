# IPv6 (Internet Protocol version 6)

## Overview

IPv6 (Internet Protocol version 6) is the most recent version of the Internet Protocol. It was developed to address the IPv4 address exhaustion problem and to provide improvements in routing, security, and network auto-configuration. IPv6 is designed to replace IPv4 and is the future of internet addressing.

## Key Characteristics

| Feature | IPv6 |
|---------|------|
| **Address Size** | 128 bits |
| **Address Format** | Hexadecimal colon notation (2001:db8::1) |
| **Total Addresses** | 340 undecillion (2¹²⁸ ≈ 3.4 × 10³⁸) |
| **Header Size** | 40 bytes (fixed, no options) |
| **Checksum** | No (delegated to link and transport layers) |
| **Fragmentation** | Source host only (not by routers) |
| **Broadcast** | No (replaced by multicast) |
| **Configuration** | SLAAC (Stateless Auto-Config) or DHCPv6 |
| **IPSec** | Mandatory (built-in security) |
| **Address Resolution** | NDP (Neighbor Discovery Protocol) instead of ARP |

## IPv6 Advantages Over IPv4

```
1. Vast Address Space
   - 340 undecillion addresses
   - Every grain of sand on Earth could have billions of IPs
   - No more address exhaustion

2. Simplified Header
   - Fixed 40-byte header (no options)
   - Faster processing by routers
   - Extension headers for optional features

3. Auto-Configuration
   - SLAAC: hosts configure themselves
   - No DHCP required (though DHCPv6 available)
   - Plug-and-play networking

4. Built-in Security
   - IPSec mandatory
   - Authentication and encryption
   - Better privacy features

5. Better Routing
   - Hierarchical addressing
   - Smaller routing tables
   - More efficient routing

6. No NAT Required
   - Every device gets public address
   - True end-to-end connectivity
   - Simplifies protocols (VoIP, gaming, P2P)

7. Multicast Improvements
   - No broadcast (more efficient)
   - Built-in multicast support
   - Scope-based addressing
```

## IPv6 Packet Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version| Traffic Class |           Flow Label                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Payload Length        |  Next Header  |   Hop Limit   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                                                               +
|                                                               |
+                         Source Address                        +
|                          (128 bits)                           |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                                                               +
|                                                               |
+                      Destination Address                      +
|                          (128 bits)                           |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                       Extension Headers                       +
|                         (if present)                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                           Payload                             +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### IPv6 Header Fields (40 bytes fixed)

1. **Version** (4 bits): IP version = 6
2. **Traffic Class** (8 bits): QoS and priority
   - Differentiated Services Code Point (DSCP): 6 bits
   - Explicit Congestion Notification (ECN): 2 bits
   - Similar to IPv4 Type of Service
3. **Flow Label** (20 bits): QoS flow identification
   - Identifies packets belonging to same flow
   - Used for QoS and ECMP (Equal-Cost Multi-Path)
   - Routers can treat flows differently
4. **Payload Length** (16 bits): Length of data after header
   - Does NOT include the 40-byte header itself
   - Maximum: 65,535 bytes
   - Jumbograms (>65,535) use Hop-by-Hop extension
5. **Next Header** (8 bits): Type of next header
   - Like IPv4 Protocol field
   - Values: 6=TCP, 17=UDP, 58=ICMPv6, 59=No next header
   - Or indicates extension header type
6. **Hop Limit** (8 bits): Maximum hops (like IPv4 TTL)
   - Decremented by each router
   - Packet dropped when reaches 0
   - Typical values: 64, 128, 255
7. **Source Address** (128 bits): Sender IPv6 address
8. **Destination Address** (128 bits): Receiver IPv6 address

### Comparison with IPv4 Header

```
Removed from IPv4:
- Header Length (IHL): Fixed at 40 bytes
- Identification, Flags, Fragment Offset: Moved to extension header
- Header Checksum: Redundant (link and transport layers handle it)
- Options: Replaced by extension headers

Added to IPv6:
- Flow Label: QoS identification

Renamed:
- TTL → Hop Limit
- Protocol → Next Header
- Type of Service → Traffic Class
```

## IPv6 Address Format

### Full Representation

```
2001:0db8:0000:0042:0000:8a2e:0370:7334

Structure:
- 8 groups of 4 hexadecimal digits
- Separated by colons
- Each group = 16 bits
- Total: 128 bits
```

### Address Compression Rules

#### Rule 1: Remove Leading Zeros

```
Original:
2001:0db8:0000:0042:0000:8a2e:0370:7334

After removing leading zeros:
2001:db8:0:42:0:8a2e:370:7334

Each group can have 1-4 hex digits
```

#### Rule 2: Compress Consecutive Zeros with `::`

```
Before: 2001:db8:0:42:0:8a2e:370:7334
After:  2001:db8:0:42::8a2e:370:7334

Before: 2001:db8:0:0:0:0:0:1
After:  2001:db8::1

Before: 0:0:0:0:0:0:0:1
After:  ::1  (loopback)

Before: 0:0:0:0:0:0:0:0
After:  ::  (unspecified)

IMPORTANT: Can only use :: once per address
(otherwise ambiguous which zeros are compressed)
```

### Special Addresses

```
::                        Unspecified address
                          (0.0.0.0 in IPv4)
                          Used before address is configured

::1                       Loopback address
                          (127.0.0.1 in IPv4)
                          Local host communication

::ffff:192.0.2.1          IPv4-mapped IPv6 address
                          Used for IPv4/IPv6 compatibility
                          Last 32 bits contain IPv4 address

2001:db8::/32             Documentation prefix
                          Reserved for examples (TEST-NET)

fe80::/10                 Link-local prefix
                          Auto-configured on every interface

ff00::/8                  Multicast prefix
```

## IPv6 Address Types

### 1. Unicast (One-to-One)

Address for a single interface.

#### Global Unicast Address (GUA)

```
Prefix: 2000::/3 (2000:0000 to 3fff:ffff)

Routable on the Internet (like public IPv4)

Format:
|    48 bits     | 16 bits |        64 bits        |
| Global Routing | Subnet  |    Interface ID       |
| Prefix         | ID      |                       |

Example:
2001:0db8:1234:0001:0000:0000:0000:0001
|-- Global --||Sub||--- Interface ID ---|

Typically:
- ISP assigns /48 or /56 to customer
- Customer has 65,536 (/48) or 256 (/56) subnets
- Each subnet is /64 with 2^64 addresses
```

#### Unique Local Address (ULA)

```
Prefix: fc00::/7 (fc00:: to fdff::)

Private addressing (like RFC 1918 in IPv4)
Not routed on public internet

Format:
fd00::/8 is used (fc00::/8 reserved for future)

|  8 bits | 40 bits  | 16 bits |    64 bits     |
|  fd     | Random   | Subnet  | Interface ID   |
|  prefix | Global   | ID      |                |
|         | ID       |         |                |

Example:
fd12:3456:789a:0001::1

Generation:
- fd prefix
- 40-bit random number (cryptographically generated)
- Ensures uniqueness even if networks merge
```

#### Link-Local Address

```
Prefix: fe80::/10

Automatically configured on every IPv6-enabled interface
Only valid on the local link (not routed)
Like IPv4 169.254.0.0/16 (APIPA)

Format:
fe80::interface-id/64

Examples:
fe80::1
fe80::20c:29ff:fe9d:8c6a

Uses:
- Neighbor Discovery Protocol (NDP)
- Router discovery
- Address autoconfiguration
- Local communication

Always present, even if GUA configured
```

### 2. Anycast (One-to-Nearest)

```
Address assigned to multiple interfaces
Packet delivered to nearest one (by routing metric)

Use cases:
- Load balancing
- Service discovery
- Root DNS servers (6 of 13 use anycast)

Same format as unicast (no special prefix)
Designated as anycast during configuration

Example:
Anycast: 2001:db8::1 assigned to 3 servers
Client sends to 2001:db8::1
Routers deliver to nearest server
```

### 3. Multicast (One-to-Many)

```
Prefix: ff00::/8

Replaces broadcast in IPv4
Packet delivered to all members of multicast group

Format:
|   8 bits   |  4 bits | 4 bits |       112 bits        |
|    ff      |  Flags  | Scope  |      Group ID         |

Flags (4 bits):
0000 = Permanent (well-known)
0001 = Temporary (transient)

Scope (4 bits):
1 = Interface-local
2 = Link-local
5 = Site-local
8 = Organization-local
e = Global
```

#### Common Multicast Addresses

```
Well-Known Multicast:

ff02::1                   All nodes on link
                          (Like 255.255.255.255 broadcast)

ff02::2                   All routers on link

ff02::1:2                 All DHCP servers/relays on link

ff02::1:ff00:0/104        Solicited-node multicast
                          Used in Neighbor Discovery

ff05::1:3                 All DHCP servers (site-local)

Solicited-Node Multicast:
Format: ff02::1:ff[last 24 bits of address]

Example:
Address: 2001:db8::1234:5678
Solicited-node: ff02::1:ff34:5678

Purpose: Efficient address resolution (NDP)
```

### 4. No Broadcast

```
IPv4 broadcast → IPv6 multicast

IPv4: 192.168.1.255 (broadcast to all)
IPv6: ff02::1 (all-nodes multicast)

Benefits:
- More efficient (only interested hosts listen)
- Reduces network noise
- Scalable
```

## IPv6 Address Structure

### EUI-64 (Extended Unique Identifier)

Method to generate interface ID from MAC address:

```
MAC Address: 00:1A:2B:3C:4D:5E

Step 1: Split in half
  00:1A:2B  :  3C:4D:5E

Step 2: Insert FF:FE in middle
  00:1A:2B:FF:FE:3C:4D:5E

Step 3: Flip 7th bit (Universal/Local bit)
  00 → 02 (in binary: 00000000 → 00000010)

Result: 02:1A:2B:FF:FE:3C:4D:5E

Step 4: Format as IPv6 interface ID
  021a:2bff:fe3c:4d5e

Full address:
  2001:db8:1234:5678:021a:2bff:fe3c:4d5e

Privacy concern: MAC address visible in IP
Solution: Privacy Extensions (RFC 4941)
```

### Privacy Extensions (RFC 4941)

```
Problem: EUI-64 exposes MAC address
         Allows tracking of devices

Solution: Random interface IDs
- Generated randomly
- Changed periodically (typically daily)
- Temporary addresses for outgoing connections

Example:
Stable:    2001:db8::21a:2bff:fe3c:4d5e  (EUI-64, for incoming)
Temporary: 2001:db8::a4b2:76d9:3e21:91f8 (random, for outgoing)

Benefits:
- Privacy protection
- Harder to track users
- Still allows stable addressing for servers
```

## IPv6 Subnetting

### Standard Subnet Size: /64

```
Why /64?

1. SLAAC requires /64
   - 64-bit prefix + 64-bit interface ID

2. Massive address space per subnet
   - 2^64 = 18,446,744,073,709,551,616 addresses
   - 18.4 quintillion addresses per subnet!
   - Will never run out

3. Standard recommendation
   - RFC 4291, RFC 5375

Even point-to-point links should use /64
(not /127 like IPv4 /30)
```

### Subnet Allocation Example

```
ISP allocates: 2001:db8::/32

Customer (Enterprise):
Receives: 2001:db8:abcd::/48

|       32 bits       | 16 bits | 16 bits |     64 bits      |
| ISP Prefix          | Customer| Subnet  | Interface ID     |
| 2001:db8            | abcd    | 0-ffff  |                  |

Customer has 2^16 = 65,536 subnets:
2001:db8:abcd:0000::/64
2001:db8:abcd:0001::/64
2001:db8:abcd:0002::/64
...
2001:db8:abcd:ffff::/64

Each subnet has 2^64 addresses
```

### Hierarchical Addressing

```
Organization: 2001:db8:abcd::/48

Building 1: 2001:db8:abcd:0100::/56
  Floor 1: 2001:db8:abcd:0101::/64
  Floor 2: 2001:db8:abcd:0102::/64
  Floor 3: 2001:db8:abcd:0103::/64

Building 2: 2001:db8:abcd:0200::/56
  Floor 1: 2001:db8:abcd:0201::/64
  Floor 2: 2001:db8:abcd:0202::/64

Servers: 2001:db8:abcd:1000::/56
  Web: 2001:db8:abcd:1001::/64
  Database: 2001:db8:abcd:1002::/64
  Email: 2001:db8:abcd:1003::/64

Benefits:
- Logical organization
- Easy summarization
- Simplified routing
- Room for growth
```

## IPv6 Auto-Configuration

### SLAAC (Stateless Address Auto-Configuration)

Automatic IPv6 configuration without DHCP:

```
Process:

1. Link-Local Address Generation
   Host creates link-local address (fe80::)
   Interface ID from EUI-64 or random

2. Duplicate Address Detection (DAD)
   Sends Neighbor Solicitation for its own address
   If no response → address is unique

3. Router Solicitation (RS)
   Host sends multicast RS to ff02::2 (all routers)
   "Are there any routers?"

4. Router Advertisement (RA)
   Router responds with:
   - Network prefix (e.g., 2001:db8:1234:5678::/64)
   - Default gateway address
   - DNS servers (if configured)
   - Other configuration flags

5. Global Address Formation
   Host combines:
   - Prefix from RA (2001:db8:1234:5678)
   - Interface ID (021a:2bff:fe3c:4d5e)
   - Result: 2001:db8:1234:5678:021a:2bff:fe3c:4d5e

6. DAD for Global Address
   Verify global address is unique

7. Ready!
   Host has link-local and global address
   No DHCP server needed!

Flags in RA:
- M (Managed): Use DHCPv6 for addresses
- O (Other): Use DHCPv6 for other info (DNS, NTP, etc.)
```

### DHCPv6 (Dynamic Host Configuration Protocol for IPv6)

Alternative/supplement to SLAAC:

```
Stateful DHCPv6:
- Like DHCPv4
- Server assigns addresses
- Tracks assignments
- Use when: Need centralized control

Stateless DHCPv6:
- SLAAC for address
- DHCPv6 for other info (DNS, domain, etc.)
- Use when: Need SLAAC + additional config

DHCPv6 Messages:
- SOLICIT: Client requests address
- ADVERTISE: Server offers address
- REQUEST: Client accepts offer
- REPLY: Server confirms

Multicast addresses:
- ff02::1:2 - All DHCP servers/relays on link
- ff05::1:3 - All DHCP servers (site-local)
```

### Router Advertisement Example

```
Router configuration (Linux):
# Enable IPv6 forwarding
net.ipv6.conf.all.forwarding = 1

# radvd configuration
interface eth0 {
    AdvSendAdvert on;
    prefix 2001:db8:1234:5678::/64 {
        AdvOnLink on;
        AdvAutonomous on;
    };
    RDNSS 2001:4860:4860::8888 {
    };
};

This advertises:
- Prefix: 2001:db8:1234:5678::/64
- DNS: 2001:4860:4860::8888 (Google DNS)
- Clients auto-configure themselves
```

## Neighbor Discovery Protocol (NDP)

NDP replaces ARP and adds functionality:

### NDP Functions

```
1. Router Discovery
   - Find routers on link
   - Get network prefix

2. Address Resolution
   - Map IPv6 address to MAC address
   - Replaces ARP

3. Duplicate Address Detection (DAD)
   - Verify address uniqueness

4. Neighbor Unreachability Detection
   - Monitor neighbor reachability

5. Redirect
   - Inform hosts of better next hop
```

### NDP Message Types (ICMPv6)

```
Type 133: Router Solicitation (RS)
  Sent by: Host
  To: ff02::2 (all routers)
  Purpose: "Are there routers here?"

Type 134: Router Advertisement (RA)
  Sent by: Router
  To: ff02::1 (all nodes) or unicast
  Purpose: "Here's my prefix and config"

Type 135: Neighbor Solicitation (NS)
  Sent by: Host
  To: Solicited-node multicast
  Purpose: "Who has this IPv6 address?" (like ARP request)
           "Is anyone using this address?" (DAD)

Type 136: Neighbor Advertisement (NA)
  Sent by: Host
  To: Unicast or ff02::1
  Purpose: "I have this IPv6 address" (like ARP reply)
           "I'm using this address" (DAD response)

Type 137: Redirect
  Sent by: Router
  To: Unicast (specific host)
  Purpose: "Use different router for this destination"
```

### Address Resolution Example

```
Host A wants to communicate with Host B:

Host A: 2001:db8::1
Host B: 2001:db8::2
Host B MAC: 00:11:22:33:44:55

1. Host A sends Neighbor Solicitation (NS):
   From: 2001:db8::1
   To: ff02::1:ff00:2 (solicited-node multicast for ::2)
   Question: "What's the MAC address of 2001:db8::2?"

2. Host B receives NS (listening on solicited-node multicast)

3. Host B sends Neighbor Advertisement (NA):
   From: 2001:db8::2
   To: 2001:db8::1 (unicast reply)
   Answer: "My MAC is 00:11:22:33:44:55"

4. Host A caches: 2001:db8::2 → 00:11:22:33:44:55

5. Host A sends packet directly to Host B

Neighbor cache entry:
2001:db8::2  dev eth0  lladdr 00:11:22:33:44:55  REACHABLE
```

### Duplicate Address Detection (DAD)

```
Before using any address:

1. Node creates address:
   - Link-local: fe80::1
   - Or global: 2001:db8::1

2. Node sends Neighbor Solicitation:
   From: :: (unspecified address)
   To: ff02::1:ff00:1 (solicited-node multicast)
   Target: fe80::1 (address being tested)
   Question: "Is anyone using fe80::1?"

3. Wait 1 second:
   - If NA received → Address in use (conflict!)
   - If no response → Address is unique ✓

4. If unique:
   - Mark address as valid
   - Start using it

If conflict detected:
- Link-local conflict: Generate new interface ID
- Global conflict: Manual intervention required
```

## IPv6 Extension Headers

Extension headers provide optional functionality without bloating main header:

### Extension Header Types

```
Next Header values:

0   = Hop-by-Hop Options (must be first if present)
43  = Routing Header
44  = Fragment Header
50  = Encapsulating Security Payload (ESP)
51  = Authentication Header (AH)
60  = Destination Options
59  = No Next Header (no more headers)
6   = TCP
17  = UDP
58  = ICMPv6
```

### Extension Header Chaining

```
Base IPv6 Header
  Next Header = 43 (Routing)
  ↓
Routing Header
  Next Header = 44 (Fragment)
  ↓
Fragment Header
  Next Header = 60 (Destination Options)
  ↓
Destination Options Header
  Next Header = 6 (TCP)
  ↓
TCP Header and Data

Recommended order (RFC 2460):
1. IPv6 base header
2. Hop-by-Hop Options
3. Destination Options (for intermediate destinations)
4. Routing
5. Fragment
6. Authentication (AH)
7. Encapsulating Security Payload (ESP)
8. Destination Options (for final destination)
9. Upper layer (TCP, UDP, ICMPv6, etc.)
```

### Fragment Header

```
Fragmentation only at source (not routers!)

Format:
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Next Header  |   Reserved    |      Fragment Offset    |Res|M|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Identification                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Next Header: Protocol after reassembly
Fragment Offset: Position in original packet (8-byte units)
M flag: More Fragments (1 = more, 0 = last)
Identification: Groups fragments together

Process:
1. Source tests path MTU
2. If packet > MTU, source fragments
3. Router that cannot forward sends ICMPv6 "Packet Too Big"
4. Source reduces packet size or fragments
5. Destination reassembles

Note: Routers never fragment!
```

## ICMPv6 (Internet Control Message Protocol for IPv6)

ICMPv6 is integral to IPv6 operation:

### ICMPv6 Message Types

```
Error Messages:

1   Destination Unreachable
    Code 0: No route to destination
    Code 1: Communication with destination administratively prohibited
    Code 3: Address unreachable
    Code 4: Port unreachable

2   Packet Too Big
    Used for Path MTU Discovery
    Includes MTU of next hop

3   Time Exceeded
    Code 0: Hop limit exceeded in transit
    Code 1: Fragment reassembly time exceeded

4   Parameter Problem
    Code 0: Erroneous header field
    Code 1: Unrecognized Next Header type

Informational Messages:

128 Echo Request (ping)
129 Echo Reply (ping response)

Neighbor Discovery (part of ICMPv6):

133 Router Solicitation
134 Router Advertisement
135 Neighbor Solicitation
136 Neighbor Advertisement
137 Redirect

Multicast Listener Discovery:

130 Multicast Listener Query
131 Multicast Listener Report
132 Multicast Listener Done
```

### Ping6 Example

```bash
# Basic ping
ping6 2001:4860:4860::8888

# Ping link-local (must specify interface)
ping6 fe80::1%eth0

# Set packet size
ping6 -s 1000 2001:4860:4860::8888

# Set hop limit
ping6 -t 5 2001:4860:4860::8888

# Example output:
PING 2001:4860:4860::8888(2001:4860:4860::8888) 56 data bytes
64 bytes from 2001:4860:4860::8888: icmp_seq=1 ttl=118 time=10.2 ms
64 bytes from 2001:4860:4860::8888: icmp_seq=2 ttl=118 time=9.9 ms

--- 2001:4860:4860::8888 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 9.900/10.050/10.200/0.150 ms
```

### Path MTU Discovery

```
IPv6 requires source to fragment:

1. Source sends large packet (1500 bytes)

2. Router with smaller MTU (1400 bytes):
   - Cannot fragment (not allowed in IPv6)
   - Drops packet
   - Sends ICMPv6 Type 2 "Packet Too Big"
   - Includes MTU value (1400)

3. Source receives ICMPv6:
   - Reduces packet size to 1400
   - Retransmits

4. Success!
   - Source caches PMTU for destination
   - Uses smaller packets for this destination

Benefits:
- No fragmentation overhead at routers
- Better performance
- Source controls fragmentation
```

## IPv6 Commands and Tools

### IPv6 Configuration (Linux)

```bash
# View IPv6 addresses
ip -6 addr show
ip -6 a

# Add IPv6 address
sudo ip -6 addr add 2001:db8::1/64 dev eth0

# Remove IPv6 address
sudo ip -6 addr del 2001:db8::1/64 dev eth0

# Enable IPv6 on interface
sudo sysctl -w net.ipv6.conf.eth0.disable_ipv6=0

# Disable IPv6 on interface
sudo sysctl -w net.ipv6.conf.eth0.disable_ipv6=1

# View IPv6 routing table
ip -6 route show

# Add IPv6 route
sudo ip -6 route add 2001:db8::/32 via 2001:db8::1

# Add default route
sudo ip -6 route add default via fe80::1 dev eth0

# View neighbor cache (NDP)
ip -6 neigh show
```

### IPv6 Configuration (Windows)

```cmd
# View IPv6 configuration
netsh interface ipv6 show config
ipconfig

# Add IPv6 address
netsh interface ipv6 add address "Ethernet" 2001:db8::1/64

# Remove IPv6 address
netsh interface ipv6 delete address "Ethernet" 2001:db8::1

# Add route
netsh interface ipv6 add route 2001:db8::/32 "Ethernet" 2001:db8::1

# View IPv6 routing table
netsh interface ipv6 show route
route print -6

# View neighbor cache
netsh interface ipv6 show neighbors
```

### Testing Connectivity

```bash
# Ping IPv6 address
ping6 2001:4860:4860::8888
ping -6 google.com

# Ping link-local (requires interface specification)
ping6 fe80::1%eth0
ping6 -I eth0 fe80::1

# Traceroute
traceroute6 google.com
traceroute -6 google.com

# TCP connection test
telnet 2001:4860:4860::8888 80
nc -6 google.com 80

# DNS lookup
host google.com
dig AAAA google.com
nslookup -type=AAAA google.com
```

### Network Diagnostics

```bash
# View IPv6 sockets
ss -6 -tuln
netstat -6 -tuln

# View IPv6 connections
ss -6 -tun
netstat -6 -tun

# tcpdump for IPv6
sudo tcpdump -i eth0 ip6
sudo tcpdump -i eth0 'icmp6'
sudo tcpdump -i eth0 'ip6 and tcp port 80'

# Neighbor Discovery monitoring
sudo tcpdump -i eth0 'icmp6 and (ip6[40] >= 133 and ip6[40] <= 137)'
```

## IPv6 Best Practices

### 1. Address Planning

```
Use /48 for sites:
- Gives 65,536 subnets
- Future-proof
- Standard recommendation

Use /64 for subnets:
- Required for SLAAC
- Standard LAN size
- Even for point-to-point

Use /56 for small sites:
- 256 subnets
- Acceptable for small deployments

Hierarchy example:
2001:db8:abcd::/48                Organization
  2001:db8:abcd:0100::/56         Building 1
    2001:db8:abcd:0101::/64       Floor 1
    2001:db8:abcd:0102::/64       Floor 2
  2001:db8:abcd:0200::/56         Building 2
  2001:db8:abcd:1000::/56         Data center
```

### 2. Dual Stack

```
Run IPv4 and IPv6 simultaneously:

Benefits:
- Smooth transition
- Backward compatibility
- No disruption

Implementation:
- Enable IPv6 on all interfaces
- Maintain IPv4 for legacy
- Configure both protocols on servers
- Use DNS with A and AAAA records

Eventually:
- IPv6-only for new deployments
- IPv4 only where necessary
```

### 3. Security

```
IPv6-specific security considerations:

1. ICMPv6 is essential
   - Don't block all ICMPv6
   - Allow NDP (types 133-137)
   - Allow PMTU Discovery (type 2)
   - Allow Echo Request/Reply (types 128-129)

2. Link-local security
   - fe80::/10 should stay local
   - Don't route link-local

3. Disable IPv6 if not using
   - But preferably, enable and secure it
   - Attacks can use IPv6 if enabled but unmonitored

4. RA Guard
   - Prevent rogue router advertisements
   - Protect against MITM attacks

5. Extension headers
   - Many firewalls can't inspect them
   - Consider filtering or limiting

6. Privacy Extensions
   - Enable for client devices
   - Prevents tracking via EUI-64
```

### 4. DNS Configuration

```
Always configure both records:

example.com.     IN  A      192.0.2.1        (IPv4)
example.com.     IN  AAAA   2001:db8::1      (IPv6)

Test both:
dig A example.com
dig AAAA example.com

Reverse DNS:
IPv4: 1.2.0.192.in-addr.arpa
IPv6: 1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa

Configure resolver:
/etc/resolv.conf:
nameserver 2001:4860:4860::8888
nameserver 2001:4860:4860::8844
nameserver 8.8.8.8
```

### 5. Firewalling

```bash
# ip6tables example

# Allow established connections
ip6tables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
ip6tables -A INPUT -i lo -j ACCEPT

# Allow ICMPv6
ip6tables -A INPUT -p ipv6-icmp -j ACCEPT

# Allow SSH
ip6tables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
ip6tables -A INPUT -p tcp --dport 80 -j ACCEPT
ip6tables -A INPUT -p tcp --dport 443 -j ACCEPT

# Drop invalid packets
ip6tables -A INPUT -m state --state INVALID -j DROP

# Default drop
ip6tables -P INPUT DROP
ip6tables -P FORWARD DROP
ip6tables -P OUTPUT ACCEPT
```

### 6. Monitoring

```bash
# Monitor NDP
ip -6 neigh show
watch -n 1 'ip -6 neigh show'

# Monitor IPv6 traffic
sudo iftop -f "ip6"
sudo nethogs -6

# View IPv6 statistics
netstat -s -6

# Monitor routing
ip -6 route show
watch -n 1 'ip -6 route show'

# Check for IPv6 connectivity
ping6 -c 1 2001:4860:4860::8888 && echo "IPv6 works" || echo "IPv6 fails"
```

## IPv6 Transition Mechanisms

### 1. Dual Stack

```
Run both IPv4 and IPv6:

Advantages:
+ Simple
+ No translation
+ Native performance

Disadvantages:
- Must manage both protocols
- Requires IPv4 addresses (scarce)

Best for: Long-term transition
```

### 2. Tunneling

#### 6in4 (Manual Tunnel)

```
IPv6 packets encapsulated in IPv4:

[IPv4 Header][IPv6 Header][Data]

Configuration:
# Linux
ip tunnel add ipv6tunnel mode sit remote 198.51.100.1 local 192.0.2.1
ip link set ipv6tunnel up
ip addr add 2001:db8::2/64 dev ipv6tunnel
ip route add ::/0 dev ipv6tunnel

Use case: Static IPv6 over IPv4
```

#### 6to4

```
Automatic tunneling using 2002::/16:

IPv4: 192.0.2.1
IPv6: 2002:c000:0201::/48
      (c000:0201 = 192.0.2.1 in hex)

Deprecated: Security issues
```

#### Teredo

```
Tunneling for NAT environments:

Prefix: 2001::/32

Use case: Windows clients behind NAT
Status: Deprecated, use native IPv6
```

### 3. NAT64/DNS64

```
Allow IPv6-only clients to access IPv4 services:

IPv6 client (2001:db8::1)
    ↓ Request "www.example.com"
DNS64 server
    ↓ Returns 64:ff9b::192.0.2.1 (synthesized AAAA)
IPv6 client
    ↓ Connects to 64:ff9b::192.0.2.1
NAT64 gateway
    ↓ Translates to IPv4: 192.0.2.1
IPv4 server (192.0.2.1)

Use case: IPv6-only networks accessing IPv4 internet
```

## IPv6 vs IPv4 Comparison

| Feature | IPv4 | IPv6 |
|---------|------|------|
| **Address length** | 32 bits | 128 bits |
| **Address format** | Decimal (192.0.2.1) | Hexadecimal (2001:db8::1) |
| **Address space** | 4.3 billion | 340 undecillion |
| **Header size** | 20-60 bytes (variable) | 40 bytes (fixed) |
| **Checksum** | Yes | No |
| **Fragmentation** | Routers and source | Source only |
| **Broadcast** | Yes | No (multicast) |
| **Multicast** | Optional | Built-in |
| **IPSec** | Optional | Mandatory |
| **Address resolution** | ARP | NDP |
| **Auto-configuration** | DHCP | SLAAC or DHCPv6 |
| **NAT** | Common | Not needed |
| **Options** | In header | Extension headers |
| **Jumbograms** | No | Yes (>65535 bytes) |
| **Mobile IP** | Extension | Built-in |

## ELI10: IPv6 Explained Simply

Think of IPv6 as a massive upgrade to the internet's addressing system:

### The Address Problem

```
IPv4 (old):
- Like phone numbers with 10 digits
- Only 4.3 billion addresses
- Running out (like phone numbers in 1990s)

IPv6 (new):
- Like phone numbers with 39 digits
- 340 undecillion addresses
- Enough for every atom on Earth to have trillions of IPs
- We'll NEVER run out
```

### Address Format

```
IPv4: 192.168.1.1
- Four numbers (0-255)
- Separated by dots

IPv6: 2001:db8::1
- Eight groups of hex digits (0-9, a-f)
- Separated by colons
- Can compress zeros with ::
```

### Auto-Configuration

```
IPv4:
- Need DHCP server
- Manual configuration for servers
- "Hey DHCP, give me an address!"

IPv6:
- Auto-configures itself (SLAAC)
- Listens for router
- Creates own address
- "I'll make my own address, thanks!"
```

### No More NAT

```
IPv4 with NAT:
Home: All devices share one public IP
      Like apartment building with one mailbox

IPv6:
Home: Every device gets its own public IP
      Like every apartment having its own mailbox
      Direct delivery, no sharing needed
```

### Better Security

```
IPv4:
- Security added later (IPSec optional)
- Like adding locks to old houses

IPv6:
- Security built-in (IPSec mandatory)
- Like new houses with locks included
```

### Link-Local Addresses

```
Every IPv6 device has:
1. Link-local (fe80::): For local network (like intercom)
2. Global (2001:...): For internet (like phone number)

Always have both, automatic!
```

## Further Resources

- [RFC 8200 - IPv6 Specification](https://tools.ietf.org/html/rfc8200)
- [RFC 4291 - IPv6 Addressing Architecture](https://tools.ietf.org/html/rfc4291)
- [RFC 4862 - IPv6 Stateless Address Autoconfiguration](https://tools.ietf.org/html/rfc4862)
- [RFC 4861 - Neighbor Discovery for IPv6](https://tools.ietf.org/html/rfc4861)
- [RFC 4941 - Privacy Extensions for SLAAC](https://tools.ietf.org/html/rfc4941)
- [RFC 3484 - Default Address Selection](https://tools.ietf.org/html/rfc3484)
- [IPv6 Test](https://test-ipv6.com/) - Test your IPv6 connectivity
- [Hurricane Electric IPv6 Certification](https://ipv6.he.net/certification/) - Free IPv6 training
