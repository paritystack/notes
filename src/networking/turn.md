# TURN (Traversal Using Relays around NAT)

## Overview

TURN is a protocol that helps establish connections between peers when direct peer-to-peer communication fails. Unlike STUN which only discovers addresses, TURN acts as a relay server that forwards traffic between peers when NAT or firewall restrictions prevent direct connections.

## Why TURN is Needed

### When STUN Fails

```
Scenario 1: Symmetric NAT
  Peer A behind Symmetric NAT
  Different public port for each destination
  STUN can't provide usable address
  → Need TURN relay

Scenario 2: Restrictive Firewall
  Corporate firewall blocks incoming P2P
  Even with correct address from STUN
  → Need TURN relay

Scenario 3: UDP Blocked
  Network blocks UDP traffic
  Can't use STUN or direct P2P
  → Need TURN over TCP
```

## TURN vs STUN

| Feature | STUN | TURN |
|---------|------|------|
| **Purpose** | Discover public address | Relay traffic |
| **Bandwidth** | Minimal (discovery only) | High (relays all data) |
| **Success Rate** | ~80% | ~100% |
| **Cost** | Free (public servers) | Expensive (bandwidth) |
| **Latency** | Low (direct connection) | Higher (via relay) |
| **When to Use** | First attempt | Fallback |

## TURN Architecture

### Basic Relay

```
Peer A                    TURN Server                Peer B
(Behind NAT)              (Public IP)                (Behind NAT)
192.168.1.10              198.51.100.1               10.0.0.5
  |                            |                         |
  | Allocate Request           |                         |
  |--------------------------->|                         |
  | Allocate Success           |                         |
  |<---------------------------|                         |
  | (Relayed address assigned) |                         |
  |                            |                         |
  | Send relayed address       |                         |
  | to Peer B via signaling    |                         |
  |                            |                         |
  |         Data               |          Data           |
  |--------------------------->|------------------------>|
  |                            | (TURN relays)           |
  |         Data               |          Data           |
  |<---------------------------|<------------------------|
```

### Allocation

```
Client requests allocation from TURN server:

1. Client: "I need a relay address"
2. TURN: "Here's 198.51.100.1:50000 for you"
3. Client: "Route traffic between me and Peer X"
4. TURN: "OK, I'll relay your traffic"

Allocation lifetime: 10 minutes (default, can be refreshed)
```

## TURN Message Types

### Key Operations

| Operation | Description |
|-----------|-------------|
| **Allocate** | Request relay address |
| **Refresh** | Extend allocation lifetime |
| **Send** | Send data through relay |
| **Data** | Receive data from relay |
| **CreatePermission** | Allow peer to send data |
| **ChannelBind** | Optimize data transfer |

### Allocate Request/Response

**Request:**
```
Client → TURN Server

Method: Allocate
Attributes:
  REQUESTED-TRANSPORT: UDP (17)
  LIFETIME: 600 seconds
  USERNAME: "alice"
  MESSAGE-INTEGRITY: HMAC
```

**Response:**
```
TURN Server → Client

Method: Allocate Success
Attributes:
  XOR-RELAYED-ADDRESS: 198.51.100.1:50000
  LIFETIME: 600 seconds
  XOR-MAPPED-ADDRESS: 203.0.113.5:54321 (client's public IP)
  MESSAGE-INTEGRITY: HMAC
```

## TURN Workflow

### 1. Allocation

```
Client                          TURN Server
  |                                |
  | Allocate Request               |
  | (credentials, transport)       |
  |------------------------------->|
  |                                |
  | Allocate Success               |
  | (relayed address)              |
  |<-------------------------------|
  |                                |

Allocation created:
  Client: 203.0.113.5:54321
  Relay: 198.51.100.1:50000
  Lifetime: 600 seconds
```

### 2. Permission

```
Client                          TURN Server
  |                                |
  | CreatePermission Request       |
  | (peer IP: 10.0.0.5)            |
  |------------------------------->|
  |                                |
  | CreatePermission Success       |
  |<-------------------------------|
  |                                |

TURN server now accepts traffic from 10.0.0.5
Permission lifetime: 300 seconds
```

### 3. Sending Data

#### Method A: Send Indication

```
Client                    TURN Server               Peer
  |                            |                      |
  | Send Indication            |                      |
  | To: 10.0.0.5:6000          |                      |
  | Data: "Hello"              |                      |
  |--------------------------->|                      |
  |                            | UDP: "Hello"         |
  |                            |--------------------->|
  |                            |                      |
```

#### Method B: Channel Binding (Optimized)

```
Client                    TURN Server               Peer
  |                            |                      |
  | ChannelBind Request        |                      |
  | Channel: 0x4000            |                      |
  | Peer: 10.0.0.5:6000        |                      |
  |--------------------------->|                      |
  |                            |                      |
  | ChannelBind Success        |                      |
  |<---------------------------|                      |
  |                            |                      |
  | ChannelData (0x4000)       |                      |
  | Data: "Hello"              |                      |
  |--------------------------->|                      |
  |                            | UDP: "Hello"         |
  |                            |--------------------->|

ChannelData has only 4-byte overhead (vs 36 bytes for Send)
More efficient for continuous data flow
```

### 4. Receiving Data

```
Peer                      TURN Server              Client
  |                            |                      |
  | UDP: "Reply"               |                      |
  |--------------------------->|                      |
  |                            | Data Indication      |
  |                            | From: 10.0.0.5:6000  |
  |                            | Data: "Reply"        |
  |                            |--------------------->|
  |                            |                      |
```

## TURN Message Format

### Send Indication

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|0 0|     STUN Message Type     |         Message Length        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Magic Cookie                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Transaction ID (96 bits)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   XOR-PEER-ADDRESS                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        DATA                                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### ChannelData Message

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Channel Number        |            Length             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                       Application Data                        |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Channel Number: 0x4000 - 0x7FFF
Length: Length of application data
```

## TURN Attributes

### Common Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| **XOR-RELAYED-ADDRESS** | 0x0016 | Relay transport address |
| **XOR-PEER-ADDRESS** | 0x0012 | Peer transport address |
| **DATA** | 0x0013 | Data to relay |
| **LIFETIME** | 0x000D | Allocation lifetime (seconds) |
| **REQUESTED-TRANSPORT** | 0x0019 | Desired transport (UDP=17) |
| **CHANNEL-NUMBER** | 0x000C | Channel number (0x4000-0x7FFF) |
| **EVEN-PORT** | 0x0018 | Request even port (RTP/RTCP) |
| **DONT-FRAGMENT** | 0x001A | Don't fragment |
| **RESERVATION-TOKEN** | 0x0022 | Token for port reservation |

## TURN Authentication

### Long-Term Credentials

```
Request 1 (no credentials):
  Allocate Request

Response 1:
  Error 401 Unauthorized
  REALM: "example.com"
  NONCE: "abcd1234"

Request 2 (with credentials):
  USERNAME: "alice"
  REALM: "example.com"
  NONCE: "abcd1234"
  MESSAGE-INTEGRITY: HMAC-SHA1(message, key)

Key = MD5(username:realm:password)

Response 2:
  Allocate Success Response
  XOR-RELAYED-ADDRESS: ...
  LIFETIME: 600
```

### Short-Term Credentials

```
Used within ICE (WebRTC):
  USERNAME: <random>:<random>
  PASSWORD: <shared secret>

Simpler, time-limited authentication
```

## TURN Allocation Lifecycle

```
1. Allocate (request relay)
   ↓
2. Success (relay assigned)
   ↓
3. CreatePermission (allow peers)
   ↓
4. ChannelBind (optimize transfer)
   ↓
5. Send/Receive Data
   ↓
6. Refresh (extend lifetime)
   ↓
7. Delete or Expire

Timeline:
  0s: Allocate
  300s: Refresh (extend to 900s)
  600s: Refresh (extend to 1200s)
  900s: Refresh (extend to 1500s)
  ...
  Stop refreshing: Allocation expires
```

## TURN Over Different Transports

### TURN over UDP

```
Default mode
Client → TURN Server: UDP
TURN Server → Peer: UDP

Fast, but UDP might be blocked
```

### TURN over TCP

```
Client → TURN Server: TCP
TURN Server → Peer: UDP

Works when UDP blocked
More overhead (TCP vs UDP)
```

### TURN over TLS

```
Client → TURN Server: TLS over TCP
TURN Server → Peer: UDP

Encrypted control channel
Works in restrictive environments
Port 443 (looks like HTTPS)
```

## ICE with TURN

### Candidate Priority

```
ICE tries candidates in order:

1. Host Candidate (local IP)
   Type: host
   Priority: Highest
   Example: 192.168.1.10:5000

2. Server Reflexive (STUN)
   Type: srflx
   Priority: High
   Example: 203.0.113.5:6000

3. Relayed (TURN)
   Type: relay
   Priority: Lowest (fallback)
   Example: 198.51.100.1:50000

Connection attempt:
  Try host → Try srflx → Try relay
  Use first successful connection
```

### WebRTC with TURN

```javascript
const configuration = {
  iceServers: [
    // STUN server (free)
    { urls: 'stun:stun.l.google.com:19302' },

    // TURN server (requires auth)
    {
      urls: 'turn:turn.example.com:3478',
      username: 'alice',
      credential: 'password123'
    },

    // TURN over TLS
    {
      urls: 'turns:turn.example.com:5349',
      username: 'alice',
      credential: 'password123'
    }
  ]
};

const pc = new RTCPeerConnection(configuration);
```

## TURN Server Setup

### Using coturn

**Install:**
```bash
sudo apt-get install coturn
```

**Configure `/etc/turnserver.conf`:**
```bash
# Listening ports
listening-port=3478
tls-listening-port=5349

# Relay ports
min-port=49152
max-port=65535

# Authentication
lt-cred-mech
user=alice:password123
realm=example.com

# Or use shared secret
use-auth-secret
static-auth-secret=my-secret-key

# Certificates (for TLS)
cert=/etc/ssl/turn.crt
pkey=/etc/ssl/turn.key

# Logging
log-file=/var/log/turnserver.log
verbose

# External IP (if behind NAT)
external-ip=203.0.113.5/192.168.1.10

# Limit resources
max-bps=1000000
total-quota=100
```

**Run:**
```bash
sudo turnserver -v
```

**Test:**
```bash
# Using turnutils
turnutils_uclient -v -u alice -w password123 turn.example.com
```

## TURN Bandwidth Considerations

### Bandwidth Usage

```
Video call: 2 Mbps per direction

Direct P2P (no TURN):
  Client A →→ Client B
  Total bandwidth: 4 Mbps (2 up + 2 down each)

Through TURN relay:
  Client A → TURN → Client B
  TURN bandwidth: 4 Mbps (2 in + 2 out)
  Each client: 4 Mbps (2 up + 2 down)

TURN server needs 2x the bandwidth!
```

### Cost Implications

```
Example: 1000 concurrent video calls through TURN
  Each call: 2 Mbps × 2 directions = 4 Mbps
  Total: 1000 × 4 Mbps = 4 Gbps

At $0.10/GB:
  4 Gbps = 0.5 GB/second
  Per hour: 1,800 GB = $180/hour
  Per day: 43,200 GB = $4,320/day

Why ICE tries direct connection first!
```

### Optimization Strategies

```
1. Prefer direct connections (STUN)
   - ~80% of connections succeed
   - Zero relay bandwidth

2. Short allocation lifetimes
   - Free up resources quickly
   - Prevent unused allocations

3. Connection quality monitoring
   - Switch from relay to direct if possible
   - ICE restart

4. Rate limiting
   - Prevent abuse
   - Fair resource sharing

5. Geographic distribution
   - Regional TURN servers
   - Reduce latency
```

## TURN Security

### Authentication Required

```
Public TURN servers = expensive bandwidth
Must authenticate:
  - Username/password
  - Time-limited credentials
  - Shared secrets
```

### Quota Management

```
Limit per user:
  - Bandwidth (bytes/sec)
  - Total data (GB)
  - Concurrent allocations
  - Allocation lifetime
```

### Access Control

```
Restrict by:
  - IP ranges (corporate network)
  - Time windows
  - User groups
```

## Monitoring TURN Server

### Key Metrics

```
1. Active allocations
   - Current number
   - Peak usage

2. Bandwidth
   - Total throughput
   - Per-client usage
   - Inbound/outbound ratio

3. Connections
   - Success rate
   - Allocation duration
   - Peak concurrent

4. Authentication
   - Failed attempts
   - Expired credentials

5. Resources
   - CPU usage
   - Memory
   - Network interfaces
   - Port exhaustion
```

### coturn Statistics

```bash
# Real-time stats
telnet localhost 5766

# Commands:
ps    # Print sessions
pid   # Show process info
pc    # Print configuration
```

## TURN Alternatives

### 1. Direct P2P (preferred)

```
Pros: Free, low latency
Cons: Doesn't always work
Success rate: ~80%
```

### 2. SIP/VoIP Gateways

```
Traditional VoIP infrastructure
Built-in media relays
More expensive
```

### 3. Media Servers

```
Janus, Jitsi, Kurento
Selective Forwarding Unit (SFU)
Different model than TURN
```

## Troubleshooting TURN

### Can't allocate

```bash
# Check TURN server is running
sudo systemctl status coturn

# Check listening ports
netstat -tuln | grep 3478

# Test with turnutils
turnutils_uclient -v turn.example.com
```

### Authentication fails

```bash
# Verify credentials
turnutils_uclient -u alice -w password123 turn.example.com

# Check realm configuration
grep realm /etc/turnserver.conf

# Check logs
tail -f /var/log/turnserver.log
```

### High latency

```
- Use geographically closer TURN server
- Check server load (CPU, bandwidth)
- Try TURN over TCP (sometimes faster)
- Monitor network path (traceroute)
```

## ELI10

TURN is like using a friend to pass notes in class:

**Without TURN (Direct):**
- You throw note directly to friend
- Fast and easy
- But teacher might catch it!

**With TURN (Through Relay):**
- You give note to trusted student
- They walk it over to your friend
- Slower, but always works
- Even if teacher is watching

**Why TURN?**

Imagine you're in Building A, friend in Building B:
- Can't throw note that far (NAT/firewall blocking)
- Need someone in the middle to help
- TURN server is that helpful person

**Costs:**
- Direct (free): Just toss the note
- TURN (expensive): Someone must carry every note back and forth
  - Video call = thousands of notes per second!
  - TURN server gets tired (bandwidth costs)

**Smart Strategy (ICE):**
1. Try throwing directly (host candidate)
2. Try from outside (STUN)
3. Last resort: Use TURN relay

Use TURN only when absolutely needed!

## Further Resources

- [RFC 5766 - TURN Specification](https://tools.ietf.org/html/rfc5766)
- [RFC 8656 - TURN Update](https://tools.ietf.org/html/rfc8656)
- [coturn Server](https://github.com/coturn/coturn)
- [WebRTC ICE](https://webrtc.org/)
- [TURN Server Providers](https://www.twilio.com/stun-turn)
