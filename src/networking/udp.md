# UDP (User Datagram Protocol)

## Overview

UDP is a connectionless transport layer protocol that provides fast, unreliable data transmission. Unlike TCP, UDP doesn't guarantee delivery, ordering, or error checking, making it ideal for time-sensitive applications where speed matters more than reliability.

## UDP vs TCP

| Feature | UDP | TCP |
|---------|-----|-----|
| **Connection** | Connectionless | Connection-oriented |
| **Reliability** | Unreliable (no guarantee) | Reliable (guaranteed delivery) |
| **Ordering** | No ordering | Ordered delivery |
| **Speed** | Fast (low overhead) | Slower (more overhead) |
| **Header Size** | 8 bytes | 20-60 bytes |
| **Error Checking** | Optional checksum | Mandatory checksum + retransmission |
| **Flow Control** | None | Yes (window-based) |
| **Congestion Control** | None | Yes |
| **Use Cases** | Streaming, gaming, DNS, VoIP | File transfer, web, email |

## UDP Packet Format

```
 0      7 8     15 16    23 24    31
+--------+--------+--------+--------+
|     Source      |   Destination   |
|      Port       |      Port       |
+--------+--------+--------+--------+
|                 |                 |
|     Length      |    Checksum     |
+--------+--------+--------+--------+
|                                   |
|          Data octets ...          |
+-----------------------------------+
```

### Header Fields (8 bytes total)

1. **Source Port** (16 bits): Port number of sender (optional, can be 0)
2. **Destination Port** (16 bits): Port number of receiver
3. **Length** (16 bits): Length of UDP header + data (minimum 8 bytes)
4. **Checksum** (16 bits): Error checking (optional in IPv4, mandatory in IPv6)

### Example UDP Header

```
Source Port: 53210 (0xCFCA)
Destination Port: 53 (0x0035) - DNS
Length: 512 bytes
Checksum: 0x1A2B
```

**Hexadecimal representation:**
```
CF CA 00 35 02 00 1A 2B
[... 504 bytes of data ...]
```

## How UDP Works

### Sending Data

```
Application ’ Socket ’ UDP Layer ’ IP Layer ’ Network

1. Application writes data to UDP socket
2. UDP adds 8-byte header
3. UDP passes datagram to IP layer
4. IP sends packet to destination
5. No acknowledgment expected
```

### Receiving Data

```
Network ’ IP Layer ’ UDP Layer ’ Socket ’ Application

1. IP receives packet
2. IP passes to UDP based on protocol number (17)
3. UDP validates checksum (if present)
4. UDP delivers to application based on port
5. If port not listening, send ICMP "Port Unreachable"
```

## UDP Communication Flow

### One-Way Communication (Fire and Forget)

```
Client                          Server (port 9000)
  |                                |
  |  UDP Packet (Hello)            |
  |------------------------------->|
  |                                |
  |  UDP Packet (World)            |
  |------------------------------->|
  |                                |

No handshake, no acknowledgment
```

### Two-Way Communication (Request-Response)

```
Client                          Server
  |                                |
  |  DNS Query (Port 53)           |
  |------------------------------->|
  |                                |
  |  DNS Response                  |
  |<-------------------------------|
  |                                |

Application must handle timeouts and retries
```

## UDP Checksum Calculation

### Pseudo Header (for checksum calculation)

```
+--------+--------+--------+--------+
|          Source IP Address        |
+--------+--------+--------+--------+
|       Destination IP Address      |
+--------+--------+--------+--------+
|  zero  |Protocol| UDP Length      |
+--------+--------+--------+--------+
```

### Checksum Process

1. Create pseudo header from IP information
2. Concatenate: Pseudo header + UDP header + data
3. Divide into 16-bit words
4. Sum all 16-bit words
5. Add carry bits to result
6. Take one's complement

**Example:**
```python
def calculate_checksum(data):
    # Sum all 16-bit words
    total = sum(struct.unpack("!%dH" % (len(data)//2), data))

    # Add carry
    total = (total >> 16) + (total & 0xffff)
    total += (total >> 16)

    # One's complement
    return ~total & 0xffff
```

## Common UDP Ports

| Port | Service | Purpose |
|------|---------|---------|
| **53** | DNS | Domain name resolution |
| **67/68** | DHCP | Dynamic IP configuration |
| **69** | TFTP | Trivial File Transfer |
| **123** | NTP | Network Time Protocol |
| **161/162** | SNMP | Network management |
| **514** | Syslog | System logging |
| **520** | RIP | Routing protocol |
| **1900** | SSDP | Service discovery (UPnP) |
| **3478** | STUN | NAT traversal |
| **5353** | mDNS | Multicast DNS |

## UDP Use Cases

### 1. DNS (Domain Name System)

```
Client sends UDP query to port 53:
+----------------+
| DNS Query      |
| example.com?   |
+----------------+

Server responds:
+----------------+
| DNS Response   |
| 93.184.216.34  |
+----------------+

Fast lookup, retry if no response
```

### 2. Video Streaming

```
Server sends video frames continuously:
Frame 1 ’ Frame 2 ’ Frame 3 ’ Frame 4 ’ Frame 5

If Frame 3 is lost, continue with Frame 4
(Old frame is useless for live streaming)
```

### 3. Online Gaming

```
Game Client ’ Server: Player position updates (60 FPS)
Update 1: Player at (100, 200)
Update 2: Player at (101, 201)
Update 3: [LOST]
Update 4: Player at (103, 203)

Lost packet is okay - next update corrects position
```

### 4. VoIP (Voice over IP)

```
Continuous audio stream:
Packet 1: Audio 0-20ms
Packet 2: Audio 20-40ms
Packet 3: Audio 40-60ms [LOST]
Packet 4: Audio 60-80ms

Lost packet = brief audio glitch
Retransmission would cause worse delay
```

### 5. DHCP (IP Address Assignment)

```
Client                          Server
  |                                |
  | DHCP Discover (broadcast)      |
  |------------------------------->|
  |                                |
  | DHCP Offer                     |
  |<-------------------------------|
  |                                |
  | DHCP Request                   |
  |------------------------------->|
  |                                |
  | DHCP ACK                       |
  |<-------------------------------|
```

## UDP Socket Programming

### Python UDP Server

```python
import socket

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind to address and port
server_address = ('localhost', 9000)
sock.bind(server_address)

print(f"UDP server listening on {server_address}")

while True:
    # Receive data (up to 1024 bytes)
    data, client_address = sock.recvfrom(1024)
    print(f"Received {len(data)} bytes from {client_address}")
    print(f"Data: {data.decode()}")

    # Send response
    sock.sendto(b"Message received", client_address)
```

### Python UDP Client

```python
import socket

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_address = ('localhost', 9000)

try:
    # Send data
    message = b"Hello, UDP Server!"
    sock.sendto(message, server_address)

    # Receive response (with timeout)
    sock.settimeout(5.0)
    data, server = sock.recvfrom(1024)
    print(f"Received: {data.decode()}")

except socket.timeout:
    print("No response from server")
finally:
    sock.close()
```

### C UDP Server

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 9000
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    char buffer[BUFFER_SIZE];
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    // Setup server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind socket
    bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));

    printf("UDP server listening on port %d\n", PORT);

    while(1) {
        // Receive data
        int n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                        (struct sockaddr*)&client_addr, &addr_len);
        buffer[n] = '\0';

        printf("Received: %s\n", buffer);

        // Send response
        sendto(sockfd, "ACK", 3, 0,
               (struct sockaddr*)&client_addr, addr_len);
    }

    return 0;
}
```

## UDP Broadcast and Multicast

### Broadcast (One-to-All in subnet)

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Send to broadcast address
broadcast_address = ('255.255.255.255', 9000)
sock.sendto(b"Broadcast message", broadcast_address)
```

### Multicast (One-to-Many selected)

```python
import socket
import struct

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

# Sender
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"Multicast message", (MCAST_GRP, MCAST_PORT))

# Receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', MCAST_PORT))

# Join multicast group
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP),
                   socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

data, address = sock.recvfrom(1024)
```

## UDP Maximum Packet Size

### Theoretical Limits

```
IPv4:
- Max IP packet: 65,535 bytes
- IP header: 20 bytes (minimum)
- UDP header: 8 bytes
- Max UDP data: 65,507 bytes

IPv6:
- Max payload (jumbogram): 4,294,967,295 bytes
```

### Practical Limits (MTU)

```
Ethernet MTU: 1500 bytes
- IP header: 20 bytes
- UDP header: 8 bytes
- Safe UDP data: 1472 bytes

To avoid fragmentation:
- Stay under 1472 bytes for IPv4
- Stay under 1452 bytes for IPv6
```

## UDP Reliability Techniques

Since UDP doesn't provide reliability, applications must implement it:

### 1. Acknowledgments

```
Sender                          Receiver
  |                                |
  | Packet 1                       |
  |------------------------------->|
  |                                |
  | ACK 1                          |
  |<-------------------------------|
  |                                |
  | Packet 2                       |
  |------------------------------->|
  |                                |
  [timeout - no ACK received]
  |                                |
  | Packet 2 (resend)              |
  |------------------------------->|
  |                                |
  | ACK 2                          |
  |<-------------------------------|
```

### 2. Sequence Numbers

```
Application adds sequence numbers:
Packet 1: [Seq=1][Data]
Packet 2: [Seq=2][Data]
Packet 3: [Seq=3][Data]

Receiver detects missing packets
Requests retransmission if needed
```

### 3. Timeouts and Retries

```python
import socket
import time

def send_with_retry(sock, data, address, max_retries=3):
    for attempt in range(max_retries):
        sock.sendto(data, address)
        sock.settimeout(1.0)

        try:
            response, _ = sock.recvfrom(1024)
            return response
        except socket.timeout:
            print(f"Retry {attempt + 1}/{max_retries}")
            continue

    raise Exception("Max retries exceeded")
```

## UDP Advantages

1. **Low Latency**: No connection setup, immediate transmission
2. **Low Overhead**: 8-byte header vs TCP's 20+ bytes
3. **No Connection State**: Simpler, uses less memory
4. **Broadcast/Multicast**: Can send to multiple receivers
5. **Fast**: No waiting for acknowledgments
6. **Transaction-Oriented**: Good for request-response

## UDP Disadvantages

1. **Unreliable**: Packets may be lost, duplicated, or reordered
2. **No Flow Control**: Can overwhelm receiver
3. **No Congestion Control**: Can worsen network congestion
4. **No Security**: No encryption (use DTLS for secure UDP)
5. **Application Complexity**: Must implement reliability if needed

## UDP Security Considerations

### Vulnerabilities

1. **UDP Flood Attack**: Overwhelm server with UDP packets
2. **UDP Amplification**: Small request ’ large response (DNS, NTP)
3. **Spoofing**: Easy to fake source IP (no handshake)

### Mitigation

```
1. Rate limiting: Limit packets per second per source
2. Firewall rules: Block unnecessary UDP ports
3. Authentication: Verify sender identity
4. DTLS: Encrypted UDP (Datagram TLS)
```

### DTLS (Datagram TLS)

Secure UDP communication:

```
UDP + TLS-style encryption = DTLS

Used in:
- WebRTC
- VPN protocols
- IoT devices
```

## Monitoring UDP Traffic

### Using tcpdump

```bash
# Capture UDP traffic on port 53 (DNS)
tcpdump -i any udp port 53

# Capture all UDP traffic
tcpdump -i any udp

# Save to file
tcpdump -i any udp -w udp_capture.pcap

# View UDP packet details
tcpdump -i any udp -vv -X
```

### Using netstat

```bash
# Show UDP listening ports
netstat -un

# Show UDP statistics
netstat -su

# Show processes using UDP
netstat -unp
```

## ELI10

UDP is like sending postcards:

**TCP is like certified mail:**
- You get confirmation it arrived
- Items arrive in order
- Lost mail is resent
- But takes longer

**UDP is like postcards:**
- Just drop it in the mailbox and go
- Super fast - no waiting
- But might get lost
- Might arrive out of order
- No way to know if it arrived

**When to use UDP (postcards):**
- Quick questions (DNS: "What's this address?")
- Live streaming (watching a game - who cares about 1 missed frame?)
- Online games (your position updates 60 times per second)
- Video calls (slight glitch is better than delay)

**When to use TCP (certified mail):**
- Important files
- Web pages
- Emails
- Banking transactions

## Further Resources

- [RFC 768 - UDP Specification](https://tools.ietf.org/html/rfc768)
- [UDP vs TCP Explained](https://www.cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp/)
- [UDP Programming Guide](https://beej.us/guide/bgnet/html/)
