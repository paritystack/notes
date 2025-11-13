# OSI Model (Open Systems Interconnection)

## Overview

The OSI Model is a conceptual framework that standardizes network communication into 7 layers. Each layer has specific responsibilities and communicates with the layers directly above and below it.

## The 7 Layers

```
Layer 7: Application    → User applications (HTTP, FTP, SMTP)
Layer 6: Presentation   → Data format, encryption (SSL/TLS)
Layer 5: Session        → Session management
Layer 4: Transport      → End-to-end delivery (TCP, UDP)
Layer 3: Network        → Routing, IP addressing
Layer 2: Data Link      → MAC addressing, switches
Layer 1: Physical       → Physical media, cables, signals
```

## Memory Aids

**Top to Bottom:** All People Seem To Need Data Processing

**Bottom to Top:** Please Do Not Throw Sausage Pizza Away

## Layer 1: Physical Layer

### Purpose
Transmits raw bits (0s and 1s) over physical media.

### Responsibilities
- Physical connection between devices
- Bit transmission and reception
- Voltage levels, timing, data rates
- Cable specifications
- Signal encoding

### Components
- **Cables**: Ethernet (Cat5e, Cat6), Fiber optic, Coaxial
- **Hubs**: Repeat signals to all ports
- **Repeaters**: Amplify signals
- **Network Interface Cards (NICs)**

### Encoding Examples

```
Manchester Encoding (Ethernet):
0: High-to-low transition
1: Low-to-high transition

   1     0     1     1     0
  _|‾|_ _‾|_  _|‾|_ _|‾|_ _‾|_
```

### Physical Media Types

| Medium | Speed | Distance | Use Case |
|--------|-------|----------|----------|
| **Cat5e** | 1 Gbps | 100m | Ethernet LAN |
| **Cat6** | 10 Gbps | 55m | High-speed LAN |
| **Fiber (MM)** | 10 Gbps | 550m | Building backbone |
| **Fiber (SM)** | 100 Gbps | 40km+ | Long distance |
| **WiFi** | 1-10 Gbps | 100m | Wireless LAN |

### Example: Bit Transmission

```
Computer A wants to send "Hello" (binary: 01001000...)

Physical Layer:
1. Convert bits to electrical signals
2. Transmit over cable at defined voltage levels
   High voltage (2.5V) = 1
   Low voltage (0V) = 0
3. Receiver samples signals and reconstructs bits
```

## Layer 2: Data Link Layer

### Purpose
Provides node-to-node data transfer with error detection.

### Responsibilities
- MAC (Media Access Control) addressing
- Frame formatting
- Error detection (CRC)
- Flow control
- Media access control

### Sub-layers
- **LLC** (Logical Link Control): Interface to Network Layer
- **MAC** (Media Access Control): Access to physical medium

### Components
- **Switches**: Forward frames based on MAC addresses
- **Bridges**: Connect network segments
- **Network Interface Cards**: Hardware MAC addresses

### Ethernet Frame Format

```
Preamble | SFD | Dest MAC | Src MAC | Type | Data | FCS
  7B     | 1B  |   6B     |   6B    | 2B   | 46-1500B | 4B

Preamble: 10101010... (synchronization)
SFD: Start Frame Delimiter (10101011)
Dest MAC: Destination hardware address
Src MAC: Source hardware address
Type: Protocol type (0x0800 = IPv4, 0x86DD = IPv6)
Data: Payload (46-1500 bytes)
FCS: Frame Check Sequence (CRC-32)
```

### MAC Address Format

```
AA:BB:CC:DD:EE:FF (48 bits / 6 bytes)

AA:BB:CC - OUI (Organizationally Unique Identifier)
           Vendor identification
DD:EE:FF - Device identifier

Example: 00:1A:2B:3C:4D:5E
```

### Example: Frame Forwarding

```
Switch MAC Address Table:
Port 1: AA:AA:AA:AA:AA:AA
Port 2: BB:BB:BB:BB:BB:BB
Port 3: CC:CC:CC:CC:CC:CC

Frame arrives on Port 1:
  Dest MAC: BB:BB:BB:BB:BB:BB

Switch looks up BB:BB:BB:BB:BB:BB → Port 2
Forwards frame only to Port 2
```

### ARP (Address Resolution Protocol)

Maps IP addresses to MAC addresses:

```
Host A wants to send to 192.168.1.5

1. Check ARP cache
2. If not found, broadcast ARP request:
   "Who has 192.168.1.5? Tell 192.168.1.10"

3. Host with 192.168.1.5 replies:
   "192.168.1.5 is at AA:BB:CC:DD:EE:FF"

4. Cache the mapping
5. Send frame to AA:BB:CC:DD:EE:FF
```

## Layer 3: Network Layer

### Purpose
Routes packets across networks from source to destination.

### Responsibilities
- Logical addressing (IP addresses)
- Routing
- Packet forwarding
- Fragmentation and reassembly
- Error handling (ICMP)

### Components
- **Routers**: Forward packets between networks
- **Layer 3 Switches**: Routing at hardware speed

### Protocols
- **IP** (IPv4, IPv6): Internet Protocol
- **ICMP**: Error reporting and diagnostics
- **OSPF, BGP, RIP**: Routing protocols

### Example: Routing Decision

```
Router receives packet for 10.1.2.5

Routing Table:
  10.1.0.0/16    via 192.168.1.1
  10.1.2.0/24    via 192.168.1.2
  0.0.0.0/0      via 192.168.1.254 (default)

Longest prefix match: 10.1.2.0/24
Forward to 192.168.1.2
```

### Packet Journey Example

```
PC1 (192.168.1.10) → Server (10.0.0.5)

Layer 3 decisions at each hop:
1. PC1: Not local subnet → Send to gateway (192.168.1.1)
2. Router1: Check route → Forward to Router2 (10.0.0.1)
3. Router2: Destination is local → Send to 10.0.0.5
```

## Layer 4: Transport Layer

### Purpose
Provides end-to-end communication and reliability.

### Responsibilities
- Segmentation and reassembly
- Port addressing
- Connection management
- Flow control
- Error recovery
- Multiplexing

### Protocols
- **TCP**: Reliable, connection-oriented
- **UDP**: Unreliable, connectionless

### Port Numbers

```
Source Port: Identifies sending application
Dest Port: Identifies receiving application

Well-known ports (0-1023):
  80  - HTTP
  443 - HTTPS
  22  - SSH
  53  - DNS

Registered ports (1024-49151):
  3306 - MySQL
  5432 - PostgreSQL

Dynamic ports (49152-65535):
  Ephemeral ports for client connections
```

### Example: TCP Connection

```
Client (192.168.1.10:5000) → Server (10.0.0.5:80)

Layer 4 provides:
1. Connection establishment (3-way handshake)
2. Reliable delivery (ACKs, retransmission)
3. Ordering (sequence numbers)
4. Flow control (window size)
5. Connection termination (4-way close)
```

### Multiplexing Example

```
Web browser opens multiple connections:

Tab 1: 192.168.1.10:5000 → google.com:443
Tab 2: 192.168.1.10:5001 → github.com:443
Tab 3: 192.168.1.10:5002 → stackoverflow.com:443

Transport layer demultiplexes based on port
```

## Layer 5: Session Layer

### Purpose
Manages sessions (connections) between applications.

### Responsibilities
- Session establishment, maintenance, termination
- Dialog control (half-duplex, full-duplex)
- Synchronization
- Token management

### Functions
- **Authentication**: Verify user credentials
- **Authorization**: Check permissions
- **Session restoration**: Resume interrupted sessions

### Examples

**RPC (Remote Procedure Call):**
```
Client                          Server
  |                                |
  | Session established            |
  |<------------------------------>|
  | Call remote procedure          |
  |------------------------------->|
  | Maintain session state         |
  |<------------------------------>|
  | Session terminated             |
```

**NetBIOS:**
- Session management for file/printer sharing
- Name registration and resolution

### Synchronization Points

```
File Transfer with checkpoints:

  0KB -------- 100KB -------- 200KB -------- 300KB
  ^            ^              ^              ^
  Sync 1       Sync 2         Sync 3         Complete

If failure at 250KB:
  Resume from Sync 2 (200KB)
```

## Layer 6: Presentation Layer

### Purpose
Translates data between application and network formats.

### Responsibilities
- Data format translation
- Encryption/decryption
- Compression/decompression
- Character encoding

### Functions

**1. Data Translation:**
```
ASCII ↔ EBCDIC
Big-endian ↔ Little-endian
JSON ↔ XML ↔ Binary
```

**2. Encryption:**
```
Plaintext: "Hello World"
    ↓
SSL/TLS Encryption
    ↓
Ciphertext: "3k#9$mL..."
```

**3. Compression:**
```
Original: 1000 bytes
    ↓
GZIP Compression
    ↓
Compressed: 300 bytes
```

### Examples

**SSL/TLS:**
```
Application sends: "GET / HTTP/1.1"
    ↓
Presentation Layer: Encrypts with TLS
    ↓
Transport Layer: Sends encrypted data
```

**Image Formats:**
- JPEG, PNG, GIF (compressed formats)
- Format conversion for display

**Character Encoding:**
```
String "Hello" in different encodings:
ASCII:  48 65 6C 6C 6F
UTF-8:  48 65 6C 6C 6F
UTF-16: 00 48 00 65 00 6C 00 6C 00 6F
```

## Layer 7: Application Layer

### Purpose
Provides network services directly to user applications.

### Responsibilities
- Application-level protocols
- User authentication
- Data representation
- Resource sharing

### Common Protocols

| Protocol | Port | Purpose |
|----------|------|---------|
| **HTTP/HTTPS** | 80/443 | Web browsing |
| **FTP** | 20/21 | File transfer |
| **SMTP** | 25 | Email sending |
| **POP3** | 110 | Email retrieval |
| **IMAP** | 143 | Email access |
| **DNS** | 53 | Name resolution |
| **DHCP** | 67/68 | IP configuration |
| **SSH** | 22 | Secure shell |
| **Telnet** | 23 | Remote terminal |
| **SNMP** | 161 | Network management |

### Example: HTTP Request

```
User clicks link in browser

Application Layer (HTTP):
  GET /index.html HTTP/1.1
  Host: example.com

Presentation Layer:
  Encrypt with TLS (HTTPS)

Session Layer:
  Maintain HTTPS session

Transport Layer:
  TCP connection to port 443

Network Layer:
  Route to example.com's IP

Data Link Layer:
  Frame with MAC address

Physical Layer:
  Transmit bits on wire
```

## Data Encapsulation

### Encapsulation Process (Sending)

```
Layer 7: User Data
            ↓
Layer 4: [TCP Header][Data] → Segment
            ↓
Layer 3: [IP Header][TCP Header][Data] → Packet
            ↓
Layer 2: [Eth Header][IP Header][TCP][Data][Eth Trailer] → Frame
            ↓
Layer 1: 010101110101... → Bits
```

### Decapsulation Process (Receiving)

```
Layer 1: Receive bits
            ↓
Layer 2: Remove Ethernet header/trailer → Frame
            ↓
Layer 3: Remove IP header → Packet
            ↓
Layer 4: Remove TCP header → Segment
            ↓
Layer 7: Deliver data to application
```

### PDU (Protocol Data Unit) Names

```
Layer 7-5: Data
Layer 4:   Segment (TCP) / Datagram (UDP)
Layer 3:   Packet
Layer 2:   Frame
Layer 1:   Bits
```

## Complete Communication Example

### Sending Email via SMTP

```
Layer 7 (Application):
  - SMTP client: "MAIL FROM: alice@example.com"
  - Creates email message

Layer 6 (Presentation):
  - Encode as ASCII
  - Compress if needed
  - Encrypt with TLS

Layer 5 (Session):
  - Establish SMTP session
  - Authenticate with mail server

Layer 4 (Transport):
  - TCP connection to port 25
  - Segment data
  - Add source/dest ports

Layer 3 (Network):
  - Add IP header
  - Source: 192.168.1.10
  - Dest: 10.0.0.5 (mail server)
  - Route to destination

Layer 2 (Data Link):
  - Add MAC addresses
  - Create Ethernet frame
  - Error checking (CRC)

Layer 1 (Physical):
  - Convert to electrical signals
  - Transmit on cable
```

## Troubleshooting by Layer

### Layer 1 (Physical) Issues
```
Symptoms: No connectivity, link down
Check:
  - Cable plugged in?
  - Cable damaged?
  - Port lights on?
  - Power on device?

Tools: Visual inspection, cable tester
```

### Layer 2 (Data Link) Issues
```
Symptoms: Can't reach other devices on LAN
Check:
  - MAC address conflicts?
  - Switch port errors?
  - VLAN configuration?
  - ARP table correct?

Tools: arp -a, show mac address-table
```

### Layer 3 (Network) Issues
```
Symptoms: Can't reach remote networks
Check:
  - IP address correct?
  - Subnet mask correct?
  - Gateway configured?
  - Routing table?

Tools: ping, traceroute, ip route
```

### Layer 4 (Transport) Issues
```
Symptoms: Can't connect to specific service
Check:
  - Port open?
  - Firewall blocking?
  - Service running?
  - TCP handshake succeeds?

Tools: telnet, nc (netcat), netstat
```

### Layer 7 (Application) Issues
```
Symptoms: Service accessible but not working
Check:
  - Application configuration?
  - Authentication failing?
  - Correct protocol version?
  - Application logs?

Tools: curl, application-specific tools
```

## OSI vs Real Protocols

### Where Real Protocols Fit

```
OSI Layer          Protocol Examples
---------------------------------------
7 - Application    HTTP, FTP, SMTP, DNS
6 - Presentation   SSL/TLS, JPEG, MPEG
5 - Session        NetBIOS, RPC
4 - Transport      TCP, UDP
3 - Network        IP, ICMP, OSPF, BGP
2 - Data Link      Ethernet, WiFi, PPP
1 - Physical       10BASE-T, 100BASE-TX
```

### TCP/IP Model Mapping

```
OSI Model              TCP/IP Model
-----------------------------------------
7 - Application  
6 - Presentation   → Application
5 - Session      
4 - Transport       → Transport
3 - Network         → Internet
2 - Data Link    
1 - Physical       → Network Access
```

## Benefits of Layered Approach

### 1. Modularity
```
Change one layer without affecting others
Example: Switch from WiFi to Ethernet
         (Only Layer 1/2 change, others unaffected)
```

### 2. Standardization
```
Multiple vendors can interoperate
Example: Any HTTP client can talk to any HTTP server
```

### 3. Troubleshooting
```
Systematic approach from bottom up:
1. Physical: Cable OK?
2. Data Link: Connected to switch?
3. Network: Can ping gateway?
4. Transport: Port open?
5. Application: Service running?
```

### 4. Development
```
Developers focus on their layer
Example: Web developer uses HTTP (Layer 7)
         Doesn't need to know about TCP internals
```

## ELI10

The OSI Model is like sending a letter through the mail:

**Layer 7 (Application):** You write a letter
- What you want to say

**Layer 6 (Presentation):** You format it nicely
- Maybe encrypt it (secret code)
- Compress it (make it smaller)

**Layer 5 (Session):** You start a conversation
- "Dear John" and "Sincerely, Alice"

**Layer 4 (Transport):** You put it in envelopes
- Split into pages if too long
- Number the pages so they can be reassembled

**Layer 3 (Network):** You write the address
- Where it's going
- Where it's from

**Layer 2 (Data Link):** Post office processes it
- Local post office routing
- Check if envelope is damaged

**Layer 1 (Physical):** The mail truck
- Physical delivery
- Roads, trucks, planes

Each layer does its job without worrying about the others!

## Further Resources

- [RFC 1122 - Internet Standards](https://tools.ietf.org/html/rfc1122)
- [OSI Model in Detail](https://www.cloudflare.com/learning/ddos/glossary/open-systems-interconnection-model-osi/)
- [Network+ Certification](https://www.comptia.org/certifications/network)
