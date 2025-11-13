# PCP (Port Control Protocol)

## Overview

PCP (Port Control Protocol) is a protocol that allows hosts to control how incoming packets are forwarded by upstream devices such as NAT gateways and firewalls. It's the successor to NAT-PMP and provides more features and flexibility for port mapping and firewall control.

## Key Characteristics

```
Protocol: UDP
Port: 5351
RFC: 6887 (2013)
Predecessor: NAT-PMP (RFC 6886)

Features:
✓ Port mapping (like NAT-PMP)
✓ Firewall control
✓ IPv4 and IPv6 support
✓ Explicit lifetime management
✓ Multiple NATs/firewalls
✓ Third-party port mapping
✓ Failure detection
✓ Security improvements
```

## Why PCP?

### Problems with Manual Port Forwarding

```
Traditional Approach:
1. User logs into router web interface
2. Manually configures port forwarding
3. Must remember to remove when done
4. Doesn't work with multiple NATs
5. Requires user intervention

Problems:
- Not suitable for applications
- Doesn't scale
- Security risk (ports left open)
- Complex for users
```

### PCP Solution

```
Automated Approach:
1. Application requests port mapping via PCP
2. Router automatically configures forwarding
3. Mapping has lifetime (auto-expires)
4. Application can renew or delete
5. Works with cascaded NATs

Benefits:
✓ Fully automated
✓ Application-controlled
✓ Time-limited (secure)
✓ Works across multiple NATs
✓ Standardized protocol
```

## PCP vs UPnP vs NAT-PMP

```
Feature               PCP        UPnP-IGD    NAT-PMP
Protocol              UDP        HTTP/SOAP   UDP
Complexity            Medium     High        Low
IPv6 Support          Yes        Partial     No
Multiple NATs         Yes        No          No
Explicit Lifetime     Yes        No          Yes
Firewall Control      Yes        No          No
Third-party Mapping   Yes        No          No
Security              Good       Weak        Basic
Standardization       IETF RFC   UPnP Forum  IETF RFC

Use PCP when:
- Need IPv6 support
- Multiple NATs in path
- Firewall control needed
- Modern deployment

Use NAT-PMP when:
- Simple IPv4 NAT
- Apple ecosystem
- Lightweight solution

Use UPnP when:
- Legacy device support
- Already deployed
- Complex scenarios
```

## PCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Internet                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  ISP Router     │
         │  (PCP Server)   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  Home Router    │
         │  (PCP Server)   │ ← Responds to PCP requests
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  PCP Client     │ ← Sends PCP requests
         │  (Application)  │
         └─────────────────┘

Flow:
1. Client sends PCP request to server
2. Server creates/modifies mapping
3. Server responds with mapping details
4. Client maintains mapping with renewals
5. Mapping expires or client deletes it
```

## PCP Message Format

### Request Header

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 2  |R|   Opcode    |         Reserved              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Requested Lifetime (seconds)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|            PCP Client's IP Address (128 bits)                 |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
:                       Opcode-specific data                    :
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
:                       PCP Options (optional)                  :
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Fields:

Version (8 bits): Protocol version (2)
R (1 bit): 0 for request, 1 for response
Opcode (7 bits):
  - 0: ANNOUNCE
  - 1: MAP
  - 2: PEER
Reserved (16 bits): Must be 0
Requested Lifetime (32 bits): Seconds (0 = delete)
PCP Client IP (128 bits): Client's IP address
  - IPv4: ::ffff:a.b.c.d (IPv4-mapped)
  - IPv6: Full 128-bit address
```

### Response Header

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 2  |R|   Opcode    |   Reserved    | Result Code   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Lifetime (seconds)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Epoch Time (seconds)                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Reserved (96 bits)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
:                       Opcode-specific data                    :
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
:                       PCP Options (optional)                  :
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Result Code:
0: SUCCESS
1: UNSUPP_VERSION
2: NOT_AUTHORIZED
3: MALFORMED_REQUEST
4: UNSUPP_OPCODE
5: UNSUPP_OPTION
6: MALFORMED_OPTION
7: NETWORK_FAILURE
8: NO_RESOURCES
9: UNSUPP_PROTOCOL
10: USER_EX_QUOTA
11: CANNOT_PROVIDE_EXTERNAL
12: ADDRESS_MISMATCH
13: EXCESSIVE_REMOTE_PEERS

Epoch Time:
- Seconds since PCP server started
- Used to detect server reboots
- Client must refresh mappings if changed
```

## PCP Opcodes

### MAP Opcode

Create a mapping for inbound traffic:

```
MAP Request (after common header):

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Mapping Nonce                         |
|                         (96 bits)                             |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Protocol    |          Reserved (24 bits)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Internal Port          |  Suggested External Port      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|           Suggested External IP Address (128 bits)            |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Mapping Nonce:
- Random value to match request/response
- Prevents off-path attacks

Protocol:
- 6 = TCP
- 17 = UDP
- 0 = All protocols

Internal Port:
- Port on PCP client

Suggested External Port:
- Preferred external port
- 0 = server chooses

Suggested External IP:
- Preferred external IP
- 0 = server chooses

MAP Response (after common header):

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Mapping Nonce                         |
|                         (96 bits)                             |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Protocol    |          Reserved (24 bits)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Internal Port          |    Assigned External Port     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|           Assigned External IP Address (128 bits)             |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Server assigns:
- External port (may differ from suggested)
- External IP address
- Lifetime for mapping
```

### PEER Opcode

Create a mapping for bidirectional traffic with a specific peer:

```
PEER Request (after common header):

Similar to MAP, but includes remote peer address:

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Mapping Nonce                         |
|                         (96 bits)                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Protocol    |          Reserved (24 bits)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Internal Port          |  Suggested External Port      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|           Suggested External IP Address (128 bits)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Remote Peer Port        |     Reserved (16 bits)        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|              Remote Peer IP Address (128 bits)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Remote Peer Port:
- Port on remote peer

Remote Peer IP:
- IP address of remote peer

Use cases:
- P2P applications
- WebRTC
- VoIP
- Gaming
```

### ANNOUNCE Opcode

Solicit mappings from PCP-controlled devices:

```
Used by client to discover mappings after:
- Client restart
- Network change
- Epoch time mismatch

Server responds with all active mappings for client
```

## PCP Options

### THIRD_PARTY Option

Allow one host to request mappings for another:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Option Code=1|  Reserved     |   Option Length=16            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                Internal IP Address (128 bits)                 |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Use case:
- NAT gateway requests mapping for internal host
- Application server requests for clients
- Proxy services
```

### PREFER_FAILURE Option

Indicate client prefers error over server changing parameters:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Option Code=2|  Reserved     |   Option Length=0             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

With this option:
- Server must honor requested port/IP exactly
- Or return error
- No substitutions allowed

Without this option:
- Server can assign different port/IP
- Client should accept
```

### FILTER Option

Create a firewall filter:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Option Code=3|  Reserved     |   Option Length=20            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Reserved   | Prefix Length |       Remote Peer Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|              Remote Peer IP Address (128 bits)                |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Prefix Length:
- 0 = Allow all
- 1-128 = IP prefix match

Use case:
- Restrict mapping to specific source
- Security filtering
- Allow only known peers
```

## PCP Client Implementation

### Python Example

```python
import socket
import struct
import random
import time

class PCPClient:
    PCP_VERSION = 2
    PCP_SERVER_PORT = 5351
    OPCODE_MAP = 1
    OPCODE_PEER = 2

    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(3)

    def create_mapping(self, internal_port, external_port=0,
                      protocol=6, lifetime=3600):
        """
        Create a port mapping.

        Args:
            internal_port: Port on client
            external_port: Suggested external port (0 = any)
            protocol: 6=TCP, 17=UDP
            lifetime: Mapping lifetime in seconds

        Returns:
            (external_ip, external_port, lifetime)
        """
        # Generate random nonce
        nonce = random.randint(0, 2**96 - 1)

        # Build request
        request = self._build_map_request(
            nonce, protocol, internal_port,
            external_port, lifetime
        )

        # Send request
        self.sock.sendto(request, (self.server_ip, self.PCP_SERVER_PORT))

        try:
            # Receive response
            response, addr = self.sock.recvfrom(1024)
            return self._parse_map_response(response, nonce)
        except socket.timeout:
            raise Exception("PCP request timeout")

    def delete_mapping(self, internal_port, protocol=6):
        """Delete a mapping by setting lifetime to 0."""
        return self.create_mapping(
            internal_port,
            protocol=protocol,
            lifetime=0
        )

    def _build_map_request(self, nonce, protocol, internal_port,
                          external_port, lifetime):
        """Build MAP request packet."""
        # Common header
        version_r_opcode = (self.PCP_VERSION << 8) | self.OPCODE_MAP
        reserved = 0

        # Client IP (IPv4-mapped IPv6)
        client_ip = self._get_client_ip()
        client_ip_bytes = self._ipv4_to_ipv6_mapped(client_ip)

        # MAP opcode data
        nonce_bytes = nonce.to_bytes(12, 'big')
        protocol_byte = protocol
        reserved_24 = 0
        internal_port_field = internal_port
        external_port_field = external_port
        external_ip_bytes = bytes(16)  # All zeros = any

        # Pack request
        request = struct.pack(
            '!HHI',
            version_r_opcode,
            reserved,
            lifetime
        )
        request += client_ip_bytes
        request += nonce_bytes
        request += struct.pack(
            '!BxxxHH',
            protocol_byte,
            internal_port_field,
            external_port_field
        )
        request += external_ip_bytes

        return request

    def _parse_map_response(self, response, expected_nonce):
        """Parse MAP response packet."""
        # Parse common header
        version_r_opcode, reserved_result, lifetime, epoch = \
            struct.unpack('!HHII', response[:12])

        # Extract result code
        result = reserved_result & 0xFF

        if result != 0:
            raise Exception(f"PCP error: result code {result}")

        # Skip reserved bytes
        offset = 12 + 12  # Header + reserved

        # Parse MAP response data
        nonce_bytes = response[offset:offset+12]
        nonce = int.from_bytes(nonce_bytes, 'big')

        if nonce != expected_nonce:
            raise Exception("Nonce mismatch")

        offset += 12

        protocol, internal_port, external_port = \
            struct.unpack('!BxxxHH', response[offset:offset+8])

        offset += 8

        external_ip_bytes = response[offset:offset+16]
        external_ip = self._ipv6_mapped_to_ipv4(external_ip_bytes)

        return (external_ip, external_port, lifetime)

    def _get_client_ip(self):
        """Get client's local IP address."""
        # Connect to PCP server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self.server_ip, self.PCP_SERVER_PORT))
            return s.getsockname()[0]
        finally:
            s.close()

    def _ipv4_to_ipv6_mapped(self, ipv4):
        """Convert IPv4 address to IPv4-mapped IPv6."""
        parts = [int(p) for p in ipv4.split('.')]
        # ::ffff:a.b.c.d
        return bytes([0]*10 + [0xff, 0xff] + parts)

    def _ipv6_mapped_to_ipv4(self, ipv6_bytes):
        """Convert IPv4-mapped IPv6 to IPv4."""
        if ipv6_bytes[:12] == bytes([0]*10 + [0xff, 0xff]):
            # IPv4-mapped
            return '.'.join(str(b) for b in ipv6_bytes[12:])
        else:
            # Full IPv6 - return as string
            parts = struct.unpack('!8H', ipv6_bytes)
            return ':'.join(f'{p:x}' for p in parts)

    def close(self):
        self.sock.close()


# Usage example
if __name__ == '__main__':
    # Find PCP server (usually gateway)
    gateway = '192.168.1.1'

    client = PCPClient(gateway)

    try:
        # Create mapping for local port 8080
        external_ip, external_port, lifetime = \
            client.create_mapping(
                internal_port=8080,
                external_port=8080,  # Suggest same port
                protocol=6,          # TCP
                lifetime=3600        # 1 hour
            )

        print(f"Mapping created:")
        print(f"  External: {external_ip}:{external_port}")
        print(f"  Internal: localhost:8080")
        print(f"  Lifetime: {lifetime} seconds")

        # Keep mapping alive
        print("\nMapping active. Press Ctrl+C to delete...")
        try:
            while True:
                # Renew every 30 minutes
                time.sleep(1800)
                external_ip, external_port, lifetime = \
                    client.create_mapping(8080, protocol=6, lifetime=3600)
                print(f"Mapping renewed: {lifetime}s remaining")
        except KeyboardInterrupt:
            pass

        # Delete mapping
        print("\nDeleting mapping...")
        client.delete_mapping(8080)
        print("Mapping deleted")

    finally:
        client.close()
```

### Node.js Example

```javascript
const dgram = require('dgram');
const crypto = require('crypto');

class PCPClient {
  constructor(serverIP) {
    this.serverIP = serverIP;
    this.serverPort = 5351;
    this.socket = dgram.createSocket('udp4');
    this.PCP_VERSION = 2;
    this.OPCODE_MAP = 1;
  }

  async createMapping(internalPort, externalPort = 0, protocol = 6, lifetime = 3600) {
    const nonce = crypto.randomBytes(12);

    const request = this.buildMapRequest(
      nonce,
      protocol,
      internalPort,
      externalPort,
      lifetime
    );

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('PCP request timeout'));
      }, 3000);

      this.socket.once('message', (response) => {
        clearTimeout(timeout);
        try {
          const result = this.parseMapResponse(response, nonce);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });

      this.socket.send(request, this.serverPort, this.serverIP);
    });
  }

  buildMapRequest(nonce, protocol, internalPort, externalPort, lifetime) {
    const buffer = Buffer.alloc(60);
    let offset = 0;

    // Version and opcode
    buffer.writeUInt8(this.PCP_VERSION, offset++);
    buffer.writeUInt8(this.OPCODE_MAP, offset++);

    // Reserved
    buffer.writeUInt16BE(0, offset);
    offset += 2;

    // Lifetime
    buffer.writeUInt32BE(lifetime, offset);
    offset += 4;

    // Client IP (IPv4-mapped)
    buffer.fill(0, offset, offset + 10);
    offset += 10;
    buffer.writeUInt16BE(0xffff, offset);
    offset += 2;
    // Would write actual IP here
    offset += 4;

    // Nonce
    nonce.copy(buffer, offset);
    offset += 12;

    // Protocol
    buffer.writeUInt8(protocol, offset);
    offset += 4; // 1 byte + 3 reserved

    // Ports
    buffer.writeUInt16BE(internalPort, offset);
    offset += 2;
    buffer.writeUInt16BE(externalPort, offset);
    offset += 2;

    // External IP (all zeros = any)
    buffer.fill(0, offset, offset + 16);

    return buffer;
  }

  parseMapResponse(response, expectedNonce) {
    let offset = 0;

    // Parse header
    const version = response.readUInt8(offset++);
    const opcode = response.readUInt8(offset++) & 0x7f;
    const reserved = response.readUInt8(offset++);
    const result = response.readUInt8(offset++);
    const lifetime = response.readUInt32BE(offset);
    offset += 4;

    if (result !== 0) {
      throw new Error(`PCP error: result code ${result}`);
    }

    // Skip epoch and reserved
    offset += 16;

    // Check nonce
    const nonce = response.slice(offset, offset + 12);
    if (!nonce.equals(expectedNonce)) {
      throw new Error('Nonce mismatch');
    }
    offset += 12;

    // Parse MAP data
    const protocol = response.readUInt8(offset);
    offset += 4; // 1 byte + 3 reserved

    const internalPort = response.readUInt16BE(offset);
    offset += 2;
    const externalPort = response.readUInt16BE(offset);
    offset += 2;

    // External IP
    const externalIP = this.parseIP(response.slice(offset, offset + 16));

    return { externalIP, externalPort, lifetime };
  }

  parseIP(buffer) {
    // Check if IPv4-mapped
    if (buffer.slice(0, 12).equals(Buffer.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff]))) {
      return `${buffer[12]}.${buffer[13]}.${buffer[14]}.${buffer[15]}`;
    }
    // IPv6
    const parts = [];
    for (let i = 0; i < 16; i += 2) {
      parts.push(buffer.readUInt16BE(i).toString(16));
    }
    return parts.join(':');
  }

  close() {
    this.socket.close();
  }
}

// Usage
const client = new PCPClient('192.168.1.1');

client.createMapping(8080, 8080, 6, 3600)
  .then(result => {
    console.log('Mapping created:');
    console.log(`  External: ${result.externalIP}:${result.externalPort}`);
    console.log(`  Lifetime: ${result.lifetime}s`);
  })
  .catch(error => {
    console.error('Error:', error.message);
  })
  .finally(() => {
    client.close();
  });
```

## PCP Server Discovery

```
Methods to find PCP server:

1. DHCP Option
   - Option 128 (DHCPv4)
   - Option 86 (DHCPv6)
   - Contains PCP server IP address

2. Default Gateway
   - Try gateway address first
   - Most common case

3. Well-Known Anycast Address
   - IPv4: (none defined)
   - IPv6: (none defined yet)

4. Manual Configuration
   - User configures PCP server
   - For complex networks

Discovery process:
1. Check DHCP options
2. Try default gateway
3. Try manual config
4. Give up (no PCP available)
```

## Security Considerations

```
Authentication:
- PCP has no built-in authentication
- Relies on network trust
- Server trusts requests from local network

Threats:
1. Unauthorized mappings
   - Malware opens ports
   - Mitigation: Firewall rules on server

2. Mapping hijacking
   - Another host modifies mapping
   - Mitigation: Nonce verification

3. Denial of service
   - Exhaust mapping resources
   - Mitigation: Per-client quotas

4. Information disclosure
   - Reveal internal topology
   - Mitigation: Restrict query responses

Best practices:
- Deploy PCP-aware firewall
- Monitor mapping activity
- Set reasonable quotas
- Log suspicious requests
- Use short lifetimes
```

## Common Use Cases

### 1. Gaming

```python
# Game server
pcp = PCPClient('192.168.1.1')

# Create mapping for game server
external_ip, external_port, _ = pcp.create_mapping(
    internal_port=27015,  # Game server port
    external_port=27015,
    protocol=17,  # UDP
    lifetime=7200  # 2 hours
)

print(f"Server address: {external_ip}:{external_port}")
print("Share this with friends to join!")

# Register with matchmaking
register_with_matchmaking(external_ip, external_port)

# Keep mapping alive
while game_running:
    time.sleep(3600)
    pcp.create_mapping(27015, protocol=17, lifetime=7200)
```

### 2. P2P Applications

```python
# P2P file sharing
pcp = PCPClient(gateway)

# Create PEER mapping for specific peer
peer_ip = '203.0.113.50'
peer_port = 6881

mapping = pcp.create_peer_mapping(
    internal_port=6881,
    peer_ip=peer_ip,
    peer_port=peer_port,
    protocol=6,  # TCP
    lifetime=3600
)

print(f"Connected to peer: {peer_ip}:{peer_port}")
print(f"Via external: {mapping['external_ip']}:{mapping['external_port']}")
```

### 3. IoT Devices

```python
# Smart home device
pcp = PCPClient(gateway)

# Create long-lived mapping
external_ip, external_port, lifetime = pcp.create_mapping(
    internal_port=8883,  # MQTT over TLS
    protocol=6,
    lifetime=86400  # 24 hours
)

# Register with cloud service
register_device(device_id, external_ip, external_port)

# Renew daily
schedule_renewal(pcp, 8883, 86400)
```

## Troubleshooting

```bash
# Check if PCP server is responding
nc -u 192.168.1.1 5351

# Send test request (hex)
echo -n "020100000000..." | nc -u 192.168.1.1 5351

# tcpdump PCP traffic
sudo tcpdump -i any -n udp port 5351

# Example output:
# Request
# 02 01 00 00 00 0e 10 00  # Version, opcode, reserved, lifetime
# 00 00 00 00 00 00 00 00  # Client IP (first 8 bytes)
# 00 00 ff ff c0 a8 01 64  # Client IP (last 8 bytes)
# ...

# Check router logs
# Look for "PCP" or "port mapping"

# Test with pcpdump (if available)
pcpdump -i eth0

# Common issues:
# - Router doesn't support PCP
# - PCP disabled in router config
# - Firewall blocks UDP 5351
# - Multiple NATs in path
# - Quota exceeded
```

## ELI10: PCP Explained Simply

PCP is like asking the gatekeeper to let your friends visit:

### Without PCP (Manual)
```
You: "Mom, can you open the door at 3pm for my friend?"
Mom: Manually opens door at 3pm
Friend: Can enter
Problem: Mom must remember, manual work
```

### With PCP (Automatic)
```
You: "Open door for 2 hours when friend arrives"
Smart Lock: Automatically opens
Friend: Arrives, enters
Smart Lock: Closes after 2 hours

Benefits:
- Automatic
- Time-limited
- You control it
- No manual work
```

### Real Network
```
Your App: "Need port 8080 open for 1 hour"
Router: Creates port mapping
Internet: Can now reach your app
Router: Closes port after 1 hour

Secure because:
- Time-limited
- Application controlled
- Automatic cleanup
```

## Further Resources

### Specifications
- [RFC 6887 - PCP](https://tools.ietf.org/html/rfc6887)
- [RFC 6886 - NAT-PMP](https://tools.ietf.org/html/rfc6886)
- [RFC 7488 - PCP Server Discovery](https://tools.ietf.org/html/rfc7488)

### Implementations
- [libpcp](https://github.com/libpcp/pcp) - C library
- [go-pcp](https://github.com/koron/go-pcp) - Go implementation
- [miniupnpc](https://github.com/miniupnp/miniupnp) - Includes PCP support

### Tools
- pcpdump - PCP packet analyzer
- pcptest - PCP testing tool

### Comparison
- [PCP vs UPnP-IGD](https://tools.ietf.org/html/rfc6887#appendix-B)
- [PCP vs NAT-PMP](https://tools.ietf.org/html/rfc6887#appendix-A)
