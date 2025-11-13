# STUN (Session Traversal Utilities for NAT)

## Overview

STUN is a standardized network protocol that allows clients behind NAT (Network Address Translation) to discover their public IP address and the type of NAT they are behind. This information is crucial for establishing peer-to-peer connections in applications like VoIP, video conferencing, and WebRTC.

## The NAT Problem

### Why STUN is Needed

```
Private Network              NAT Router            Internet
                             (Public IP)
+------------------+         +---------+         +----------------+
| PC1: 192.168.1.10|  --->  | Router  |  --->  | Other peer     |
| PC2: 192.168.1.11|         | External IP:     | wants to       |
| PC3: 192.168.1.12|         | 203.0.113.5      | connect to you |
+------------------+         +---------+         +----------------+

Problem: How does external peer know your public IP and port?
Solution: STUN server tells you!
```

### Without STUN

```
Peer A (behind NAT) wants to connect to Peer B

Peer A knows only: 192.168.1.10 (private IP)
Peer B needs: 203.0.113.5:54321 (public IP:port)

Peer A can't tell Peer B how to reach it L
```

### With STUN

```
Peer A queries STUN server
STUN server responds: "I see you as 203.0.113.5:54321"
Peer A tells Peer B: "Connect to 203.0.113.5:54321"
Peer B connects successfully 
```

## STUN Architecture

```
Client                    STUN Server               Peer
(Behind NAT)              (Public IP)
  |                            |                      |
  | STUN Binding Request       |                      |
  |--------------------------->|                      |
  |                            |                      |
  | STUN Binding Response      |                      |
  |<---------------------------|                      |
  | (Your public IP:Port)      |                      |
  |                            |                      |
  | Send public IP:Port        |                      |
  |-------------------------------------------------->|
  |                            |                      |
  |        Direct connection established              |
  |<------------------------------------------------->|
```

## STUN Message Format

### Message Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|0 0|     STUN Message Type     |         Message Length        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Magic Cookie                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                     Transaction ID (96 bits)                  |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Attributes                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Header Fields

1. **Message Type** (16 bits):
   - Class: Request (0x00), Success Response (0x01), Error Response (0x11)
   - Method: Binding (0x001)

2. **Message Length** (16 bits):
   - Length of attributes (excluding 20-byte header)

3. **Magic Cookie** (32 bits):
   - Fixed value: 0x2112A442
   - Helps distinguish STUN from other protocols

4. **Transaction ID** (96 bits):
   - Unique identifier for matching requests/responses

### Message Types

| Type | Value | Description |
|------|-------|-------------|
| **Binding Request** | 0x0001 | Request public IP/port |
| **Binding Response** | 0x0101 | Success response with address |
| **Binding Error** | 0x0111 | Error response |

## STUN Attributes

### Common Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| **MAPPED-ADDRESS** | 0x0001 | Reflexive transport address (legacy) |
| **XOR-MAPPED-ADDRESS** | 0x0020 | XORed reflexive address (preferred) |
| **USERNAME** | 0x0006 | Username for authentication |
| **MESSAGE-INTEGRITY** | 0x0008 | HMAC-SHA1 hash |
| **ERROR-CODE** | 0x0009 | Error code and reason |
| **UNKNOWN-ATTRIBUTES** | 0x000A | Unknown required attributes |
| **REALM** | 0x0014 | Realm for authentication |
| **NONCE** | 0x0015 | Nonce for digest authentication |
| **SOFTWARE** | 0x8022 | Software version |
| **FINGERPRINT** | 0x8028 | CRC-32 of message |

### XOR-MAPPED-ADDRESS Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|0 0 0 0 0 0 0 0|    Family     |         X-Port                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                X-Address (Variable)                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Family: 0x01 (IPv4), 0x02 (IPv6)
X-Port: Port XORed with most significant 16 bits of magic cookie
X-Address: IP address XORed with magic cookie (and transaction ID for IPv6)
```

**Why XOR?**
- Prevents middle boxes from modifying the address
- Some NAT devices inspect and modify IP addresses in packets

## STUN Transaction Example

### Binding Request

```
Client → STUN Server (UDP port 3478)

Message Type: Binding Request (0x0001)
Message Length: 0
Magic Cookie: 0x2112A442
Transaction ID: 0xB7E7A701BC34D686FA87DFAE

No attributes in basic request
```

**Hexadecimal:**
```
00 01 00 00 21 12 A4 42
B7 E7 A7 01 BC 34 D6 86
FA 87 DF AE
```

### Binding Response

```
STUN Server → Client

Message Type: Binding Response (0x0101)
Message Length: 12 (length of attributes)
Magic Cookie: 0x2112A442
Transaction ID: 0xB7E7A701BC34D686FA87DFAE (same as request)

Attributes:
  XOR-MAPPED-ADDRESS:
    Family: IPv4 (0x01)
    Port: 54321 (XORed)
    IP: 203.0.113.5 (XORed)
```

**Information extracted:**
```
Your public IP address: 203.0.113.5
Your public port: 54321
NAT binding created: 192.168.1.10:5000 ↔ 203.0.113.5:54321
```

## NAT Types Discovered by STUN

### 1. Full Cone NAT

```
Internal: 192.168.1.10:5000

NAT creates mapping:
  192.168.1.10:5000 ↔ 203.0.113.5:6000

Any external host can send to 203.0.113.5:6000
  → Forwarded to 192.168.1.10:5000

Best for P2P (easy to traverse)
```

### 2. Restricted Cone NAT

```
Internal: 192.168.1.10:5000

NAT creates mapping:
  192.168.1.10:5000 ↔ 203.0.113.5:6000

External host 1.2.3.4 can send to 203.0.113.5:6000
  ONLY IF 192.168.1.10:5000 previously sent to 1.2.3.4

Moderate difficulty to traverse
```

### 3. Port Restricted Cone NAT

```
Internal: 192.168.1.10:5000

NAT creates mapping:
  192.168.1.10:5000 ↔ 203.0.113.5:6000

External host 1.2.3.4:7000 can send to 203.0.113.5:6000
  ONLY IF 192.168.1.10:5000 previously sent to 1.2.3.4:7000

More difficult to traverse
```

### 4. Symmetric NAT

```
Internal: 192.168.1.10:5000

NAT creates different mappings per destination:
  To host A: 192.168.1.10:5000 ↔ 203.0.113.5:6000
  To host B: 192.168.1.10:5000 ↔ 203.0.113.5:6001
  To host C: 192.168.1.10:5000 ↔ 203.0.113.5:6002

Difficult to traverse (may need TURN relay)
```

## STUN Usage in ICE

**ICE (Interactive Connectivity Establishment)** uses STUN:

### ICE Candidate Gathering

```
1. Host Candidate:
   Local IP: 192.168.1.10:5000

2. Server Reflexive Candidate (from STUN):
   Public IP: 203.0.113.5:6000

3. Relayed Candidate (from TURN):
   Relay IP: 198.51.100.1:7000

Try connections in order:
  1. Direct (host to host)
  2. Through NAT (server reflexive)
  3. Through relay (last resort)
```

### WebRTC Connection Flow

```
Peer A                    STUN Server              Peer B
  |                            |                      |
  | Get my public IP           |                      |
  |--------------------------->|                      |
  |                            |                      |
  | 203.0.113.5:6000           |                      |
  |<---------------------------|                      |
  |                            |                      |
  | Exchange candidates via signaling server          |
  |<------------------------------------------------->|
  |                            |                      |
  | Try connection             |                      |
  |<------------------------------------------------->|
  | Connectivity check (STUN)  |                      |
  |<------------------------------------------------->|
  |                            |                      |
  | Connection established     |                      |
  |<=================================================>|
```

## STUN Authentication

### Short-Term Credentials

```
Request:
  USERNAME: "alice:bob"
  MESSAGE-INTEGRITY: HMAC-SHA1(message, password)

Server validates:
  1. Check username exists
  2. Compute HMAC with stored password
  3. Compare with MESSAGE-INTEGRITY
  4. Accept or reject
```

### Long-Term Credentials

```
Request 1 (no credentials):
  Binding Request

Response 1:
  Error 401 Unauthorized
  REALM: "example.com"
  NONCE: "random-nonce-12345"

Request 2 (with credentials):
  USERNAME: "alice"
  REALM: "example.com"
  NONCE: "random-nonce-12345"
  MESSAGE-INTEGRITY: HMAC-SHA1(message, MD5(username:realm:password))

Response 2:
  Binding Success Response
  XOR-MAPPED-ADDRESS: ...
```

## STUN Client Implementation

### Python Example

```python
import socket
import struct
import hashlib
import hmac

STUN_SERVER = "stun.l.google.com"
STUN_PORT = 19302
MAGIC_COOKIE = 0x2112A442

def create_stun_binding_request():
    # Message type: Binding Request (0x0001)
    msg_type = 0x0001

    # Message length: 0 (no attributes)
    msg_length = 0

    # Transaction ID: 96 random bits
    transaction_id = os.urandom(12)

    # Pack header
    header = struct.pack(
        '!HHI',
        msg_type,
        msg_length,
        MAGIC_COOKIE
    ) + transaction_id

    return header, transaction_id

def parse_stun_response(data, transaction_id):
    # Parse header
    msg_type, msg_length, magic_cookie = struct.unpack('!HHI', data[:8])
    recv_transaction_id = data[8:20]

    # Verify transaction ID
    if recv_transaction_id != transaction_id:
        raise Exception("Transaction ID mismatch")

    # Parse attributes
    offset = 20
    while offset < len(data):
        attr_type, attr_length = struct.unpack('!HH', data[offset:offset+4])
        offset += 4

        if attr_type == 0x0020:  # XOR-MAPPED-ADDRESS
            # Parse XOR-MAPPED-ADDRESS
            family = data[offset + 1]
            x_port = struct.unpack('!H', data[offset+2:offset+4])[0]
            x_ip = struct.unpack('!I', data[offset+4:offset+8])[0]

            # Un-XOR
            port = x_port ^ (MAGIC_COOKIE >> 16)
            ip = x_ip ^ MAGIC_COOKIE
            ip_addr = socket.inet_ntoa(struct.pack('!I', ip))

            return ip_addr, port

        offset += attr_length

    return None, None

def get_public_ip_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3)

    try:
        # Create and send binding request
        request, transaction_id = create_stun_binding_request()
        sock.sendto(request, (STUN_SERVER, STUN_PORT))

        # Receive response
        data, addr = sock.recvfrom(1024)

        # Parse response
        public_ip, public_port = parse_stun_response(data, transaction_id)

        return public_ip, public_port
    finally:
        sock.close()

# Usage
public_ip, public_port = get_public_ip_port()
print(f"Public IP: {public_ip}:{public_port}")
```

### JavaScript (WebRTC) Example

```javascript
// Create RTCPeerConnection with STUN server
const configuration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

const pc = new RTCPeerConnection(configuration);

// Listen for ICE candidates
pc.onicecandidate = (event) => {
  if (event.candidate) {
    console.log('ICE Candidate:', event.candidate);
    // Send candidate to remote peer via signaling
  }
};

// Create offer to trigger ICE gathering
pc.createOffer()
  .then(offer => pc.setLocalDescription(offer))
  .then(() => {
    // ICE candidates will be gathered
    // and onicecandidate will be called
  });
```

## Public STUN Servers

### Free STUN Servers

```
Google:
  stun.l.google.com:19302
  stun1.l.google.com:19302
  stun2.l.google.com:19302
  stun3.l.google.com:19302
  stun4.l.google.com:19302

Twilio:
  global.stun.twilio.com:3478

OpenRelay:
  stun.relay.metered.ca:80
```

### Testing STUN Server

```bash
# Using stunclient (stuntman tools)
stunclient stun.l.google.com

# Output example:
# Binding test: success
# Local address: 192.168.1.10:45678
# Mapped address: 203.0.113.5:45678
```

## STUN Limitations

### 1. Doesn't Work with Symmetric NAT

```
STUN tells you: 203.0.113.5:6000
But when connecting to peer, NAT assigns: 203.0.113.5:6001

Peer can't connect to you
→ Need TURN relay
```

### 2. Requires UDP

```
Some networks block UDP
STUN won't work
→ Need TCP fallback or TURN over TCP
```

### 3. Firewall Issues

```
Restrictive firewalls may block P2P connections
Even with correct IP:port from STUN
→ Need TURN relay
```

### 4. No Data Relay

```
STUN only discovers address
Doesn't relay data
If direct connection fails, need TURN
```

## STUN vs TURN vs ICE

```
STUN:
  - Discovers public IP:port
  - Lightweight
  - No bandwidth cost
  - Doesn't always work

TURN:
  - Relays traffic
  - Always works
  - Bandwidth intensive
  - Costs money

ICE:
  - Uses both STUN and TURN
  - Tries STUN first
  - Falls back to TURN
  - Best of both worlds
```

## STUN Server Setup

### Using coturn

```bash
# Install
sudo apt-get install coturn

# Configure /etc/turnserver.conf
listening-port=3478
fingerprint
lt-cred-mech
use-auth-secret
static-auth-secret=YOUR_SECRET
realm=example.com
total-quota=100
stale-nonce=600
```

### Run STUN Server

```bash
# Start server
sudo turnserver -v

# Test locally
stunclient localhost
```

## ELI10

STUN is like asking a friend "What's my address?" when you can't see it yourself:

**The Problem:**
You live in an apartment building (NAT)
Someone outside wants to send you mail
They need your full address, not just "Apartment 5"

**STUN Solution:**
1. You call a friend outside (STUN server)
2. Friend says: "I see your address as 123 Main St, Apartment 5"
3. You tell pen pal: "Send letters to 123 Main St, Apt 5"
4. Pen pal can now reach you!

**NAT Types:**
- **Full Cone:** Anyone can mail you once you have the address
- **Restricted:** Only people you mailed first can mail back
- **Symmetric:** Building assigns different box for each sender (hard!)

**When STUN Doesn't Work:**
- Symmetric NAT: Address changes for each recipient
- Firewall: Building doesn't accept outside mail
- → Need TURN (a forwarding service)

**WebRTC Uses STUN:**
- Video calls discover how to reach each other
- Try direct connection first (with STUN)
- Use relay (TURN) if direct doesn't work

## Further Resources

- [RFC 5389 - STUN Specification](https://tools.ietf.org/html/rfc5389)
- [RFC 8489 - STUN Update](https://tools.ietf.org/html/rfc8489)
- [WebRTC and STUN](https://webrtc.org/)
- [Interactive STUN Test](https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/)
- [coturn Server](https://github.com/coturn/coturn)
