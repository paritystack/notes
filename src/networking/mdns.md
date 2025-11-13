# mDNS (Multicast DNS)

## Overview

mDNS (Multicast DNS) is a protocol that resolves hostnames to IP addresses within small networks without requiring a conventional DNS server. It's part of Zero Configuration Networking (Zeroconf) and enables devices to discover each other on local networks using the `.local` domain.

## Why mDNS?

### Traditional DNS Limitations

```
Problem: Home networks lack DNS servers

Traditional setup requires:
1. DNS server
2. Manual configuration
3. Static IP or DHCP integration
4. Administrative overhead

mDNS solution:
- No DNS server needed
- Automatic hostname resolution
- Zero configuration
- Works out of the box
```

### Use Cases

```
1. Printer discovery
   - printer.local → 192.168.1.100

2. File sharing
   - macbook.local → 192.168.1.50

3. IoT devices
   - raspberry-pi.local → 192.168.1.75

4. Local development
   - webserver.local → 127.0.0.1

5. Service discovery
   - Find all printers on network
   - Find all file servers
```

## How mDNS Works

### Query Process

```
Device wants to find "printer.local"

1. Send multicast query to 224.0.0.251:5353
   "Who has printer.local?"

2. All devices receive query

3. Device with hostname "printer" responds
   "I'm printer.local at 192.168.1.100"

4. Querying device caches response

5. Direct communication established
```

### Multicast Address

```
IPv4: 224.0.0.251
IPv6: ff02::fb
Port: 5353 (UDP)

All devices on local network listen to this address
```

## mDNS Message Format

### DNS-Compatible Format

mDNS uses standard DNS message format:

```
+---------------------------+
|        Header             |
+---------------------------+
|        Question           |
+---------------------------+
|        Answer             |
+---------------------------+
|        Authority          |
+---------------------------+
|        Additional         |
+---------------------------+
```

### Header Fields

```
ID: Usually 0 (multicast)
QR: Query (0) or Response (1)
OPCODE: 0 (standard query)
AA: Authoritative Answer (1 for responses)
TC: Truncated
RD: Recursion Desired (0 for mDNS)
RA: Recursion Available (0 for mDNS)
RCODE: Response code

Questions: Number of questions
Answers: Number of answer RRs
Authority: Number of authority RRs
Additional: Number of additional RRs
```

## mDNS Query Example

### Query Message

```
Multicast to 224.0.0.251:5353

Question:
  Name: printer.local
  Type: A (IPv4 address)
  Class: IN (Internet)
  QU bit: 0 (multicast query)

Header:
  ID: 0
  Flags: 0x0000 (standard query)
  Questions: 1
  Answers: 0
```

### Response Message

```
Multicast from 192.168.1.100:5353

Answer:
  Name: printer.local
  Type: A
  Class: IN | Cache-Flush bit
  TTL: 120 seconds
  Data: 192.168.1.100

Header:
  ID: 0
  Flags: 0x8400 (authoritative answer)
  Questions: 0
  Answers: 1
```

## mDNS Record Types

### Common Record Types

| Type | Purpose | Example |
|------|---------|---------|
| **A** | IPv4 address | `device.local → 192.168.1.10` |
| **AAAA** | IPv6 address | `device.local → fe80::1` |
| **PTR** | Pointer (service discovery) | `_http._tcp.local → webserver` |
| **SRV** | Service location | `webserver._http._tcp.local → device.local:80` |
| **TXT** | Text information | Service metadata |

### Service Discovery (DNS-SD)

```
PTR Record: Browse services
  _http._tcp.local → webserver._http._tcp.local

SRV Record: Service location
  webserver._http._tcp.local
    Target: myserver.local
    Port: 8080
    Priority: 0
    Weight: 0

TXT Record: Service metadata
  webserver._http._tcp.local
    "path=/admin"
    "version=1.0"

A Record: IP address
  myserver.local → 192.168.1.50
```

## mDNS Features

### 1. Multicast Queries

```
Traditional DNS (unicast):
  Client → DNS Server: "What's example.com?"
  DNS Server → Client: "93.184.216.34"

mDNS (multicast):
  Client → All devices: "Who has printer.local?"
  Printer → All devices: "I'm 192.168.1.100"

Benefits:
  - No dedicated server
  - All devices hear query
  - Multiple responses possible
```

### 2. Known-Answer Suppression

```
Query includes known answers to avoid redundant responses

Client has cached: printer.local → 192.168.1.100

Query:
  Question: printer.local?
  Known Answer: 192.168.1.100 (TTL > 50% remaining)

Printer sees cached answer is still valid
  → Doesn't respond (saves bandwidth)
```

### 3. Cache-Flush Bit

```
Purpose: Invalidate old cache entries

Response with cache-flush:
  printer.local → 192.168.1.100
  Class: IN | 0x8000 (cache-flush bit set)

Receivers:
  - Flush old records for printer.local
  - Cache new record
  - Prevents stale data
```

### 4. Continuous Verification

```
Querier sends query even if cached
  - Verify host still exists
  - Detect IP changes
  - Maintain fresh cache

If no response → remove from cache
```

### 5. Graceful Shutdown

```
Device going offline:

Send goodbye message:
  printer.local → 192.168.1.100
  TTL: 0 (indicates removal)

Other devices:
  - Remove from cache immediately
  - Don't wait for timeout
```

## Service Discovery with DNS-SD

### Browsing Services

```
Query:
  _services._dns-sd._udp.local PTR?

Response (all available service types):
  _http._tcp.local
  _printer._tcp.local
  _ssh._tcp.local
  _sftp-ssh._tcp.local
```

### Finding Specific Service

```
Query:
  _http._tcp.local PTR?

Response (all HTTP services):
  webserver._http._tcp.local
  api._http._tcp.local
  admin._http._tcp.local
```

### Getting Service Details

```
Query:
  webserver._http._tcp.local SRV?
  webserver._http._tcp.local TXT?

Response (SRV):
  Target: myserver.local
  Port: 8080
  Priority: 0
  Weight: 0

Response (TXT):
  path=/
  version=2.0
  https=true

Then resolve:
  myserver.local A? → 192.168.1.50
```

## mDNS Implementation

### Python Example (Query)

```python
import socket
import struct

MDNS_ADDR = '224.0.0.251'
MDNS_PORT = 5353

def query_mdns(hostname):
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
    sock.settimeout(2)

    # Build DNS query
    # Header: ID=0, Flags=0, Questions=1
    query = struct.pack('!HHHHHH', 0, 0, 1, 0, 0, 0)

    # Question: hostname, type A, class IN
    for part in hostname.split('.'):
        query += bytes([len(part)]) + part.encode()
    query += b'\x00'  # End of name
    query += struct.pack('!HH', 1, 1)  # Type A, Class IN

    # Send query
    sock.sendto(query, (MDNS_ADDR, MDNS_PORT))

    # Receive responses
    responses = []
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            responses.append((data, addr))
    except socket.timeout:
        pass

    sock.close()
    return responses

# Usage
responses = query_mdns('printer.local')
for data, addr in responses:
    print(f"Response from {addr}")
```

### Python Example (Responder using zeroconf)

```python
from zeroconf import ServiceInfo, Zeroconf
import socket

# Get local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

# Create service info
info = ServiceInfo(
    "_http._tcp.local.",
    "My Web Server._http._tcp.local.",
    addresses=[socket.inet_aton(local_ip)],
    port=8080,
    properties={
        'path': '/',
        'version': '1.0'
    },
    server=f"{hostname}.local."
)

# Register service
zeroconf = Zeroconf()
zeroconf.register_service(info)

print(f"Service registered: {hostname}.local:8080")

try:
    input("Press Enter to unregister...\n")
finally:
    zeroconf.unregister_service(info)
    zeroconf.close()
```

### Avahi (Linux)

```bash
# Install Avahi
sudo apt-get install avahi-daemon avahi-utils

# Check hostname
avahi-resolve -n hostname.local

# Browse services
avahi-browse -a

# Browse specific service
avahi-browse _http._tcp

# Publish service
avahi-publish -s "My Service" _http._tcp 8080 path=/
```

### Bonjour (macOS)

```bash
# Resolve hostname
dns-sd -G v4 hostname.local

# Browse services
dns-sd -B _http._tcp

# Resolve service
dns-sd -L "My Service" _http._tcp

# Register service
dns-sd -R "My Service" _http._tcp . 8080 path=/
```

### Windows

```powershell
# Windows 10+ includes mDNS support

# Resolve via PowerShell
Resolve-DnsName hostname.local

# Or use Bonjour SDK
# Download from Apple Developer
```

## mDNS Service Naming

### Format

```
<Instance>._<Service>._<Transport>.local

Examples:
  My Printer._printer._tcp.local
  Living Room._airplay._tcp.local
  Office Server._smb._tcp.local
  Kitchen Speaker._raop._tcp.local
```

### Common Service Types

```
_http._tcp        Web server
_https._tcp       Secure web server
_ssh._tcp         SSH server
_sftp-ssh._tcp    SFTP over SSH
_ftp._tcp         FTP server
_smb._tcp         Samba/Windows file sharing
_afpovertcp._tcp  Apple File Protocol
_printer._tcp     Printer
_ipp._tcp         Internet Printing Protocol
_airplay._tcp     AirPlay
_raop._tcp        Remote Audio Output Protocol
_spotify-connect._tcp  Spotify Connect
```

## mDNS Traffic Analysis

### Capturing mDNS

```bash
# tcpdump
sudo tcpdump -i any -n port 5353

# Wireshark
# Filter: udp.port == 5353
# Follow: Right-click → Follow → UDP Stream
```

### Example Capture

```
Query:
  192.168.1.10 → 224.0.0.251
  DNS Query: printer.local A?

Response:
  192.168.1.100 → 224.0.0.251
  DNS Answer: printer.local → 192.168.1.100 (TTL 120)
```

## mDNS Security Considerations

### Vulnerabilities

1. **No Authentication**
```
Anyone can claim to be "printer.local"
No verification of identity
Potential for spoofing
```

2. **Local Network Only**
```
mDNS doesn't cross routers
Limited to link-local multicast
Good for security (confined to LAN)
```

3. **Information Disclosure**
```
Services broadcast their presence
Attackers can enumerate:
  - Device names
  - Service types
  - IP addresses
  - Software versions
```

4. **Name Conflicts**
```
Two devices with same hostname
Both respond to queries
Can cause confusion
```

### Mitigation

```
1. Firewall rules
   - Block port 5353 on external interfaces
   - Allow only on trusted LANs

2. VLANs
   - Separate guest network
   - Prevent mDNS between VLANs

3. Unique hostnames
   - Avoid generic names
   - Include random identifier

4. Service filtering
   - Only advertise necessary services
   - Remove unused service announcements
```

## mDNS Performance

### Bandwidth Usage

```
Typical traffic:
  - Query: ~50 bytes
  - Response: ~100 bytes
  - Continuous verification: ~1-2 queries/minute

Low bandwidth impact
Efficient for local networks
```

### Cache Timing

```
TTL values:
  - Typical: 120 seconds (2 minutes)
  - High priority: 10 seconds
  - Low priority: 4500 seconds (75 minutes)

Refresh at 80% of TTL
Query again at 90% of TTL
Remove at 100% of TTL
```

## Troubleshooting mDNS

### Device not responding

```bash
# 1. Check mDNS daemon
sudo systemctl status avahi-daemon  # Linux
sudo launchctl list | grep mDNS     # macOS

# 2. Test multicast
ping -c 3 224.0.0.251

# 3. Check firewall
sudo iptables -L | grep 5353
sudo ufw status

# 4. Capture traffic
sudo tcpdump -i any port 5353

# 5. Resolve manually
avahi-resolve -n device.local
dns-sd -G v4 device.local
```

### Name conflicts

```
Error: "hostname.local already in use"

Solutions:
1. Rename device
   - hostname.local → hostname-2.local
   - Automatic on many systems

2. Check for duplicates
   - Ensure unique hostnames
   - Search network for conflicts
```

### Slow resolution

```
Causes:
  - Network congestion
  - Many mDNS devices
  - Packet loss

Solutions:
  - Reduce query frequency
  - Use unicast if possible
  - Cache aggressively
```

## mDNS vs DNS

| Feature | Traditional DNS | mDNS |
|---------|----------------|------|
| **Server** | Centralized server | Distributed (all devices) |
| **Configuration** | Manual setup | Zero configuration |
| **Scope** | Internet-wide | Local network only |
| **Domain** | Any TLD | `.local` only |
| **Protocol** | Unicast | Multicast |
| **Port** | 53 | 5353 |
| **Security** | DNSSEC available | No authentication |

## ELI10

mDNS is like asking a question to everyone in a classroom:

**Traditional DNS:**
- Raise your hand and ask the teacher
- Teacher has a list of everyone's desks
- Teacher tells you where Alice sits

**mDNS (Multicast DNS):**
- Stand up and ask: "Where's Alice?"
- Alice hears you and responds: "I'm here at desk 5!"
- Everyone hears both question and answer
- Next time someone asks, they already know

**Benefits:**
- No need for a teacher (DNS server)
- Works immediately
- Everyone learns everyone else's location

**Limitations:**
- Only works in one classroom (local network)
- Can't ask about people in other classrooms
- Everyone hears everything (less private)

**Real Examples:**
- "Where's the printer?" → "printer.local is at 192.168.1.100"
- "Where's my MacBook?" → "macbook.local is at 192.168.1.50"
- "Any web servers?" → "myserver.local has HTTP on port 8080"

It's perfect for homes and small offices where you just want things to work!

## Further Resources

- [RFC 6762 - mDNS Specification](https://tools.ietf.org/html/rfc6762)
- [RFC 6763 - DNS-SD](https://tools.ietf.org/html/rfc6763)
- [Avahi](https://www.avahi.org/)
- [Apple Bonjour](https://developer.apple.com/bonjour/)
- [zeroconf Python Library](https://github.com/jstasiak/python-zeroconf)
