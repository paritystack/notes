# TCP/IP Model

## Overview

The TCP/IP Model (also called Internet Protocol Suite) is a practical, 4-layer networking model that describes how data is transmitted over the internet. Unlike the [OSI Model](osi_model.md), which is theoretical, TCP/IP is the actual model used in modern networks. Its layers map onto concrete protocols documented elsewhere in this section: [Ethernet/VLAN](ethernet_vlan.md) and [ARP](arp.md) at network access, [IP](ip.md) at the internet layer, [TCP](tcp.md) and [UDP](udp.md) at transport, and application protocols like [HTTP](http.md), [DNS](dns.md), and [DHCP](dhcp.md) on top.

## TCP/IP vs OSI Model

```
OSI Model (7 Layers)           TCP/IP Model (4 Layers)
---------------------------------------------------
7. Application            \
6. Presentation            >---> 4. Application
5. Session               /
4. Transport             ------> 3. Transport
3. Network               ------> 2. Internet
2. Data Link             \
1. Physical                >---> 1. Network Access
```

## The 4 Layers

### Layer 1: Network Access (Link Layer)

**Purpose:** Physical transmission of data on a network

**Combines:**
- OSI Physical Layer (Layer 1)
- OSI Data Link Layer (Layer 2)

**Responsibilities:**
- Physical addressing (MAC)
- Media access control
- Frame formatting
- Error detection
- Physical transmission

**Protocols/Technologies:**
- Ethernet (IEEE 802.3)
- WiFi (IEEE 802.11)
- PPP (Point-to-Point Protocol)
- ARP (Address Resolution Protocol)
- RARP (Reverse ARP)

**Example:**
```
Data from Internet Layer

      ↓

Add Ethernet Header:
  [Dest MAC: AA:BB:CC:DD:EE:FF]
  [Src MAC: 11:22:33:44:55:66]
  [Type: 0x0800 (IPv4)]
  [Data]
  [CRC Checksum]

      ↓

Convert to bits and transmit
```

### Layer 2: Internet Layer

**Purpose:** Routes packets across networks

**Equivalent to:** OSI Network Layer (Layer 3)

**Responsibilities:**
- Logical addressing (IP)
- Routing between networks
- Packet forwarding
- Fragmentation and reassembly
- Error reporting

**Key Protocols:**

| Protocol | Purpose | RFC |
|----------|---------|-----|
| **IP** | Internet Protocol (IPv4, IPv6) | RFC 791, 8200 |
| **ICMP** | Error reporting, diagnostics | RFC 792 |
| **IGMP** | Multicast group management | RFC 1112 |
| **IPsec** | Security (encryption, authentication) | RFC 4301 |

**Example: Packet Routing**
```
Source: 192.168.1.10 → Destination: 10.0.0.5

IP Layer adds header:
  [Version: 4]
  [TTL: 64]
  [Protocol: 6 (TCP)]
  [Source IP: 192.168.1.10]
  [Dest IP: 10.0.0.5]
  [Data]

Router at each hop:
1. Decrements TTL
2. Checks routing table
3. Forwards to next hop
4. Recalculates checksum
```

### Layer 3: Transport Layer

**Purpose:** End-to-end communication between applications

**Equivalent to:** OSI Transport Layer (Layer 4)

**Responsibilities:**
- Port-based multiplexing
- Connection management
- Reliability (for TCP)
- Flow control
- Error recovery

**Key Protocols:**

#### TCP (Transmission Control Protocol)

**Characteristics:**
- Connection-oriented
- Reliable delivery
- Ordered delivery
- Flow control
- Congestion control

**TCP Segment:**
```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Sequence Number                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Acknowledgment Number                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Offset|  Res  |     Flags     |            Window             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Checksum            |         Urgent Pointer        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Data                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**TCP Connection (3-Way Handshake):**
```
Client                          Server
  |                                |
  | SYN (seq=100)                  |
  |------------------------------->|
  |                                |
  | SYN-ACK (seq=200, ack=101)     |
  |<-------------------------------|
  |                                |
  | ACK (seq=101, ack=201)         |
  |------------------------------->|
  |                                |
  | Connection Established         |
```

#### UDP (User Datagram Protocol)

**Characteristics:**
- Connectionless
- Unreliable
- No ordering guarantee
- Low overhead
- Fast

**UDP Datagram:**
```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Length             |           Checksum            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Data                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**UDP Communication:**
```
Client                          Server
  |                                |
  | UDP Datagram                   |
  |------------------------------->|
  |                                |
  | No acknowledgment              |

Fire and forget!
```

### Layer 4: Application Layer

**Purpose:** Provides network services to applications

**Combines:**
- OSI Application Layer (Layer 7)
- OSI Presentation Layer (Layer 6)
- OSI Session Layer (Layer 5)

**Responsibilities:**
- Application-specific protocols
- Data formatting
- Session management
- User authentication

**Common Protocols:**

| Protocol | Port | Transport | Purpose |
|----------|------|-----------|---------|
| **HTTP** | 80 | TCP | Web pages |
| **HTTPS** | 443 | TCP | Secure web |
| **FTP** | 20/21 | TCP | File transfer |
| **SFTP** | 22 | TCP | Secure file transfer |
| **SSH** | 22 | TCP | Secure shell |
| **Telnet** | 23 | TCP | Remote terminal |
| **SMTP** | 25 | TCP | Send email |
| **DNS** | 53 | UDP/TCP | Name resolution |
| **DHCP** | 67/68 | UDP | IP configuration |
| **TFTP** | 69 | UDP | Simple file transfer |
| **HTTP/3** | 443 | UDP (QUIC) | Modern web |
| **NTP** | 123 | UDP | Time sync |
| **SNMP** | 161/162 | UDP | Network management |
| **POP3** | 110 | TCP | Email retrieval |
| **IMAP** | 143 | TCP | Email access |
| **RDP** | 3389 | TCP | Remote desktop |

**Example: HTTP Request**
```
Application Layer creates:

GET /index.html HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0
Accept: text/html

      ↓

Transport Layer (TCP):
  - Add TCP header
  - Source port: 54321
  - Dest port: 80
  - Establish connection

      ↓

Internet Layer (IP):
  - Add IP header
  - Resolve www.example.com to IP
  - Source: 192.168.1.10
  - Dest: 93.184.216.34

      ↓

Network Access Layer:
  - ARP for next hop MAC
  - Add Ethernet frame
  - Transmit bits
```

## Data Encapsulation in TCP/IP

### Sending Data

```
Step 1: Application creates data
  "GET /index.html HTTP/1.1\r\n..."

Step 2: Transport Layer adds header
  [TCP Header][HTTP Request] → TCP Segment

Step 3: Internet Layer adds header
  [IP Header][TCP Header][HTTP Request] → IP Packet

Step 4: Network Access adds header/trailer
  [Eth Header][IP][TCP][HTTP][Eth Trailer] → Ethernet Frame

Step 5: Convert to bits
  01001000110101... → Bits on wire
```

### Receiving Data

```
Step 1: Receive bits, extract frame
  [Eth Header][IP][TCP][HTTP][Eth Trailer]

Step 2: Check Ethernet checksum, remove header
  [IP Header][TCP Header][HTTP Request]

Step 3: Check IP checksum, route to TCP
  [TCP Header][HTTP Request]

Step 4: Process TCP segment, reassemble
  "GET /index.html HTTP/1.1\r\n..."

Step 5: Deliver to HTTP server application
```

## Complete Communication Example

### Browsing a Website

```
User types: http://www.example.com

=== Application Layer ===
1. Browser resolves domain name
   DNS Query: "What's the IP of www.example.com?"
   DNS Response: "93.184.216.34"

2. Browser creates HTTP request
   GET / HTTP/1.1
   Host: www.example.com

=== Transport Layer ===
3. TCP connection to port 80
   - 3-way handshake
   - Establish connection
   - Segment data if needed

=== Internet Layer ===
4. Create IP packet
   - Source: 192.168.1.10
   - Dest: 93.184.216.34
   - Protocol: TCP (6)
   - Add to routing queue

5. Routing
   - Check routing table
   - Forward to default gateway
   - Each router forwards packet

=== Network Access Layer ===
6. Resolve next hop MAC (ARP)
   - "Who has 192.168.1.1?"
   - "192.168.1.1 is at AA:BB:CC:DD:EE:FF"

7. Create Ethernet frame
   - Dest MAC: Gateway's MAC
   - Src MAC: PC's MAC
   - Add checksum

8. Transmit on physical medium
   - Convert to electrical signals
   - Send on Ethernet cable

=== Server Processes Request ===
9. Server receives, decapsulates
10. HTTP server processes request
11. Sends response back

=== Browser Receives Response ===
12. Decapsulate all layers
13. Browser renders HTML
```

## Protocol Interactions

### DNS Resolution
```
Application: DNS client
Transport: UDP port 53
Internet: IP packet to DNS server
Network Access: Ethernet to gateway

Query: www.example.com → 93.184.216.34
```

### Email Sending (SMTP)
```
Application: SMTP client (port 25)
Transport: TCP connection
Internet: Route to mail server IP
Network Access: Frame to gateway

MAIL FROM: alice@example.com
RCPT TO: bob@example.com
DATA
Subject: Hello
...
```

### File Transfer (FTP)
```
Application: FTP client
Transport:
  - Control: TCP port 21
  - Data: TCP port 20
Internet: IP to FTP server
Network Access: Ethernet frames

Commands on port 21:
  USER alice
  PASS secret123
  RETR file.txt

Data transfer on port 20
```

## Port Numbers

### Well-Known Ports (0-1023)

Require root/admin privileges:

```
20/21   FTP
22      SSH
23      Telnet
25      SMTP
53      DNS
67/68   DHCP
80      HTTP
110     POP3
143     IMAP
443     HTTPS
```

### Registered Ports (1024-49151)

For specific services:

```
3306    MySQL
5432    PostgreSQL
6379    Redis
8080    HTTP alternate
8443    HTTPS alternate
27017   MongoDB
```

### Dynamic/Private Ports (49152-65535)

Used by clients for outgoing connections:

```
Client opens connection:
  Source: 192.168.1.10:54321 (dynamic)
  Dest: 93.184.216.34:80 (well-known)
```

## TCP/IP Configuration

### Manual Configuration

```bash
# Set IP address
sudo ip addr add 192.168.1.100/24 dev eth0

# Set default gateway
sudo ip route add default via 192.168.1.1

# Set DNS server
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
```

### DHCP (Dynamic Host Configuration Protocol)

```
Client                          DHCP Server
  |                                |
  | DHCP Discover (broadcast)      |
  |------------------------------->|
  |                                |
  | DHCP Offer                     |
  |<-------------------------------|
  |   IP: 192.168.1.100            |
  |   Netmask: 255.255.255.0       |
  |   Gateway: 192.168.1.1         |
  |   DNS: 8.8.8.8                 |
  |                                |
  | DHCP Request                   |
  |------------------------------->|
  |                                |
  | DHCP ACK                       |
  |<-------------------------------|
  |                                |

Client now configured with:
  IP Address: 192.168.1.100
  Subnet Mask: 255.255.255.0
  Default Gateway: 192.168.1.1
  DNS Server: 8.8.8.8
  Lease Time: 24 hours
```

## TCP/IP Troubleshooting

### Layer 1: Network Access

```bash
# Check physical connection
ip link show
ethtool eth0

# Check link status
cat /sys/class/net/eth0/carrier

Symptoms: No link light, cable unplugged
```

### Layer 2: Network Access (Data Link)

```bash
# Check ARP table
arp -a
ip neigh show

# Check switch port
show mac address-table

Symptoms: Can't reach local devices
```

### Layer 3: Internet

```bash
# Check IP configuration
ip addr show
ifconfig

# Test gateway reachability
ping 192.168.1.1

# Check routing
ip route show
traceroute 8.8.8.8

Symptoms: No internet, can't reach remote hosts
```

### Layer 4: Transport

```bash
# Check listening ports
netstat -tuln
ss -tuln

# Test port connectivity
telnet example.com 80
nc -zv example.com 80

# Check firewall
iptables -L
ufw status

Symptoms: Connection refused, timeout
```

### Layer 5: Application

```bash
# Test HTTP
curl -v http://example.com

# Test DNS
dig example.com
nslookup example.com

# Test SMTP
telnet mail.example.com 25

Symptoms: Service not responding correctly
```

## TCP/IP Security

### Common Vulnerabilities

**1. IP Spoofing**
```
Attacker sends packets with fake source IP
Victim: 10.0.0.5
Attacker pretends to be: 10.0.0.5
```

**2. TCP SYN Flood**
```
Attacker sends many SYN packets
Server waits for ACK (never comes)
Server resources exhausted
```

**3. Man-in-the-Middle**
```
Attacker intercepts traffic between client and server
Can read or modify data
```

### Security Protocols

**IPsec (Internet Protocol Security)**
```
Provides:
  - Authentication Header (AH)
  - Encapsulating Security Payload (ESP)
  - Encryption and authentication

Used for VPNs
```

**TLS/SSL (Transport Layer Security)**
```
Encrypts application data
Provides:
  - Confidentiality (encryption)
  - Integrity (tampering detection)
  - Authentication (certificates)

Used for HTTPS, SMTPS, etc.
```

## TCP/IP Performance Tuning

### TCP Window Scaling

```
Default window: 65,535 bytes
With scaling: Up to 1 GB

Improves throughput on high-latency links
```

### TCP Congestion Control Algorithms

```
- Tahoe: Original algorithm
- Reno: Fast recovery
- CUBIC: Default in Linux (good for high-speed)
- BBR: Google's algorithm (optimal bandwidth)
```

### Monitoring TCP Performance

```bash
# TCP statistics
netstat -s
ss -s

# Per-connection statistics
ss -ti

# Packet captures
tcpdump -i any -w capture.pcap
```

## ELI10

TCP/IP is how computers talk to each other on the internet:

**Layer 1: Network Access (The Road)**
- Physical cables and WiFi
- Like the road system for mail delivery

**Layer 2: Internet (The Address)**
- IP addresses (like street addresses)
- Routers (like post offices) send packets the right way

**Layer 3: Transport (The Envelope)**
- TCP: Certified mail (guaranteed delivery, in order)
- UDP: Postcard (fast, but might get lost)

**Layer 4: Application (The Message)**
- The actual letter you're sending
- HTTP for websites, SMTP for email, etc.

**Example: Loading a website**
1. You type www.google.com
2. DNS finds Google's address (142.250.185.78)
3. TCP opens a connection (handshake)
4. HTTP sends "Give me the homepage"
5. Routers deliver packets to Google
6. Google sends back the HTML
7. Your browser shows the page

Each layer does its job without worrying about the others!

## Where this connects

- [OSI Model](osi_model.md) — the 7-layer theoretical model TCP/IP's 4 layers collapse
- [IP](ip.md) / [IPv4](ipv4.md) / [IPv6](ipv6.md) — the addressing and routing of the internet layer
- [TCP](tcp.md) / [UDP](udp.md) — the transport-layer protocols and their trade-offs
- [DNS](dns.md), [DHCP](dhcp.md) — application-layer plumbing that bootstraps connectivity
- [TLS/SSL](tls_ssl.md), [IPsec](ipsec.md) — security layered onto transport and internet layers
- [QUIC](quic.md) — HTTP/3's transport, blurring the transport/application boundary over UDP

## Further Resources

- [RFC 1122 - Requirements for Internet Hosts](https://tools.ietf.org/html/rfc1122)
- [RFC 1123 - Application and Support](https://tools.ietf.org/html/rfc1123)
- [TCP/IP Illustrated by W. Richard Stevens](https://en.wikipedia.org/wiki/TCP/IP_Illustrated)
- [TCP/IP Guide](http://www.tcpipguide.com/)
