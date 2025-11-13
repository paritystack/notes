# NAT-PMP (NAT Port Mapping Protocol)

## Overview

NAT-PMP (NAT Port Mapping Protocol) is a network protocol for establishing port forwarding rules in a NAT gateway automatically. It provides a simple, lightweight mechanism for applications to request port mappings without manual configuration. NAT-PMP was developed by Apple and later standardized as RFC 6886.

## Key Characteristics

```
Protocol: UDP
Port: 5351
RFC: 6886 (2013)
Developed by: Apple Inc.
Successor: PCP (Port Control Protocol)

Features:
✓ Automatic port mapping
✓ Simple protocol (easy to implement)
✓ UDP-based (low overhead)
✓ Time-limited mappings
✓ Gateway discovery
✓ External address discovery
✓ Lightweight

Limitations:
✗ IPv4 only
✗ Single NAT only
✗ No authentication
✗ Limited features vs PCP
```

## Why NAT-PMP?

### The Problem

```
Traditional Port Forwarding:
1. User manually logs into router
2. Navigates to port forwarding settings
3. Adds rule: External Port → Internal IP:Port
4. Application must document this for users
5. Users often configure incorrectly
6. Ports left open indefinitely

Issues:
- Not user-friendly
- Security risk (forgotten mappings)
- Doesn't work for non-technical users
- Can't be automated by applications
```

### NAT-PMP Solution

```
Automatic Approach:
1. Application requests mapping via NAT-PMP
2. Router creates mapping automatically
3. Mapping has expiration time
4. Application renews as needed
5. Mapping removed when no longer needed

Benefits:
✓ Zero user configuration
✓ Automatic cleanup
✓ Application-controlled
✓ Simple to implement
✓ Secure (time-limited)
```

## NAT-PMP vs Alternatives

```
Feature             NAT-PMP    UPnP-IGD    PCP
Protocol            UDP        HTTP/SOAP   UDP
Complexity          Low        High        Medium
IPv6 Support        No         Partial     Yes
Port              5351       Variable    5351
Packet Size         12 bytes   KB+         24+ bytes
Overhead            Minimal    High        Low
Deployment          Apple      Wide        Growing
Year Introduced     2005       2000        2013

Use NAT-PMP when:
- IPv4 only network
- Simple requirements
- Apple ecosystem
- Lightweight solution
- Easy implementation

Use PCP when:
- Need IPv6
- Modern deployment
- Advanced features
- Multiple NATs

Use UPnP when:
- Legacy compatibility
- Already deployed
- Complex scenarios
```

## Protocol Design

### Message Types

```
Request Types (Client → NAT Gateway):
- Opcode 0: Determine external IP address
- Opcode 1: Map UDP port
- Opcode 2: Map TCP port

Response Types (NAT Gateway → Client):
- Opcode 128: External IP address response
- Opcode 129: UDP port mapping response
- Opcode 130: TCP port mapping response

All opcodes in responses have bit 7 set (add 128)
```

### Packet Format

```
All NAT-PMP packets start with:

 0                   1
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 0  |    Opcode     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Version: Always 0
Opcode: Request or response type
```

## External IP Address Request

### Request Format

```
 0                   1
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 0  |  Opcode = 0   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total: 2 bytes

Purpose:
- Discover NAT gateway's external IP
- Check if NAT-PMP is supported
- Verify connectivity to gateway
```

### Response Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 0  | Opcode = 128  |        Result Code            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Seconds Since Start of Epoch                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     External IP Address                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total: 12 bytes

Result Code:
0: Success
1: Unsupported Version
2: Not Authorized/Refused
3: Network Failure
4: Out of Resources
5: Unsupported Opcode

Seconds Since Start of Epoch:
- Time since gateway booted/restarted
- Used to detect gateway reboots
- Incremented every second

External IP Address:
- Gateway's public IP address
- 32-bit IPv4 address
```

## Port Mapping Request

### Request Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 0  | Opcode (1/2)  |          Reserved (0)         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Internal Port          |     Suggested External Port   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Requested Port Mapping Lifetime                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total: 12 bytes

Opcode:
- 1 = UDP port mapping
- 2 = TCP port mapping

Reserved: Must be 0

Internal Port:
- Port on the client machine
- Port application is listening on

Suggested External Port:
- Preferred external port
- 0 = gateway chooses
- Non-zero = client preference

Requested Lifetime:
- Duration in seconds
- 0 = delete mapping
- Recommended: 3600 (1 hour)
- Maximum: 2^32 - 1 seconds
```

### Response Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Version = 0  | Opcode (129/130) |        Result Code         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Seconds Since Start of Epoch                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Internal Port          |      Mapped External Port     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                Port Mapping Lifetime (seconds)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total: 16 bytes

Opcode:
- 129 = UDP port mapping response
- 130 = TCP port mapping response

Mapped External Port:
- Actual external port assigned
- May differ from suggested port
- 0 = mapping failed or deleted

Port Mapping Lifetime:
- Actual lifetime granted
- May be less than requested
- Gateway may reduce based on policy
```

## Gateway Discovery

### How to Find NAT Gateway

```
Method 1: Default Gateway (Recommended)
- Use system's default gateway
- Most common case
- Works in 99% of deployments

import socket
import struct

def get_default_gateway():
    """Get default gateway IP (Linux)."""
    with open('/proc/net/route') as f:
        for line in f:
            fields = line.strip().split()
            if fields[1] == '00000000':  # Default route
                gateway_hex = fields[2]
                # Convert hex to IP
                gateway_int = int(gateway_hex, 16)
                return socket.inet_ntoa(struct.pack('<I', gateway_int))
    return None

# Or use netifaces library
import netifaces
gws = netifaces.gateways()
gateway = gws['default'][netifaces.AF_INET][0]

Method 2: DHCP Option
- DHCP Option 120 (NAT-PMP Gateway)
- Rarely used in practice

Method 3: Multicast (Legacy)
- Send to 224.0.0.1 (all hosts)
- Gateway responds
- Not recommended

Best Practice:
Always try default gateway first
```

## Client Implementation

### Python Example

```python
import socket
import struct
import time

class NATPMPClient:
    def __init__(self, gateway_ip):
        self.gateway_ip = gateway_ip
        self.gateway_port = 5351
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(3.0)

    def get_external_ip(self):
        """
        Get NAT gateway's external IP address.

        Returns:
            (external_ip, epoch_seconds)
        """
        # Build request
        request = struct.pack('!BB', 0, 0)  # Version 0, Opcode 0

        # Send request
        self.sock.sendto(request, (self.gateway_ip, self.gateway_port))

        try:
            # Receive response
            response, addr = self.sock.recvfrom(1024)

            # Parse response
            if len(response) < 12:
                raise Exception("Invalid response length")

            version, opcode, result_code, epoch, ext_ip = \
                struct.unpack('!BBHII', response)

            if result_code != 0:
                raise Exception(f"Error: result code {result_code}")

            # Convert IP to string
            external_ip = socket.inet_ntoa(struct.pack('!I', ext_ip))

            return (external_ip, epoch)

        except socket.timeout:
            raise Exception("Request timeout - NAT-PMP not supported?")

    def add_port_mapping(self, internal_port, external_port=0,
                        protocol='tcp', lifetime=3600):
        """
        Add a port mapping.

        Args:
            internal_port: Port on local machine
            external_port: Desired external port (0 = any)
            protocol: 'tcp' or 'udp'
            lifetime: Mapping duration in seconds (0 = delete)

        Returns:
            (mapped_external_port, actual_lifetime, epoch)
        """
        # Build request
        opcode = 1 if protocol == 'udp' else 2
        request = struct.pack(
            '!BBHHHI',
            0,              # Version
            opcode,         # 1=UDP, 2=TCP
            0,              # Reserved
            internal_port,
            external_port,
            lifetime
        )

        # Send request
        self.sock.sendto(request, (self.gateway_ip, self.gateway_port))

        try:
            # Receive response
            response, addr = self.sock.recvfrom(1024)

            # Parse response
            if len(response) < 16:
                raise Exception("Invalid response length")

            version, resp_opcode, result_code, epoch, \
                int_port, ext_port, actual_lifetime = \
                struct.unpack('!BBHIHHI', response)

            if result_code != 0:
                raise Exception(f"Error: result code {result_code}")

            return (ext_port, actual_lifetime, epoch)

        except socket.timeout:
            raise Exception("Request timeout")

    def delete_port_mapping(self, internal_port, protocol='tcp'):
        """Delete a port mapping by setting lifetime to 0."""
        return self.add_port_mapping(
            internal_port,
            external_port=0,
            protocol=protocol,
            lifetime=0
        )

    def close(self):
        self.sock.close()


# Usage example
if __name__ == '__main__':
    # Get gateway from system
    import netifaces
    gws = netifaces.gateways()
    gateway = gws['default'][netifaces.AF_INET][0]

    print(f"Using gateway: {gateway}")

    client = NATPMPClient(gateway)

    try:
        # Get external IP
        external_ip, epoch = client.get_external_ip()
        print(f"External IP: {external_ip}")
        print(f"Gateway uptime: {epoch} seconds")

        # Add port mapping
        print("\nCreating port mapping...")
        external_port, lifetime, epoch = client.add_port_mapping(
            internal_port=8080,
            external_port=8080,  # Prefer 8080
            protocol='tcp',
            lifetime=3600  # 1 hour
        )

        print(f"Mapping created:")
        print(f"  Internal: localhost:8080")
        print(f"  External: {external_ip}:{external_port}")
        print(f"  Lifetime: {lifetime} seconds")

        # Keep mapping alive
        print("\nMapping active. Press Ctrl+C to delete...")
        try:
            last_epoch = epoch
            while True:
                time.sleep(1800)  # Renew every 30 minutes

                # Renew mapping
                external_port, lifetime, epoch = client.add_port_mapping(
                    internal_port=8080,
                    protocol='tcp',
                    lifetime=3600
                )

                # Check for gateway reboot
                if epoch < last_epoch:
                    print("Warning: Gateway rebooted! Mapping recreated.")

                last_epoch = epoch
                print(f"Mapping renewed: {lifetime}s remaining")

        except KeyboardInterrupt:
            pass

        # Delete mapping
        print("\nDeleting mapping...")
        client.delete_port_mapping(8080, 'tcp')
        print("Mapping deleted")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        client.close()
```

### JavaScript/Node.js Example

```javascript
const dgram = require('dgram');

class NATPMPClient {
  constructor(gatewayIP) {
    this.gatewayIP = gatewayIP;
    this.gatewayPort = 5351;
    this.socket = dgram.createSocket('udp4');
  }

  getExternalIP() {
    return new Promise((resolve, reject) => {
      // Build request
      const request = Buffer.alloc(2);
      request.writeUInt8(0, 0);  // Version
      request.writeUInt8(0, 1);  // Opcode

      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 3000);

      this.socket.once('message', (response) => {
        clearTimeout(timeout);

        try {
          const version = response.readUInt8(0);
          const opcode = response.readUInt8(1);
          const resultCode = response.readUInt16BE(2);

          if (resultCode !== 0) {
            throw new Error(`Error: result code ${resultCode}`);
          }

          const epoch = response.readUInt32BE(4);
          const ipBytes = [
            response.readUInt8(8),
            response.readUInt8(9),
            response.readUInt8(10),
            response.readUInt8(11)
          ];
          const externalIP = ipBytes.join('.');

          resolve({ externalIP, epoch });
        } catch (error) {
          reject(error);
        }
      });

      this.socket.send(request, this.gatewayPort, this.gatewayIP);
    });
  }

  addPortMapping(internalPort, externalPort = 0, protocol = 'tcp', lifetime = 3600) {
    return new Promise((resolve, reject) => {
      // Build request
      const request = Buffer.alloc(12);
      request.writeUInt8(0, 0);  // Version
      request.writeUInt8(protocol === 'udp' ? 1 : 2, 1);  // Opcode
      request.writeUInt16BE(0, 2);  // Reserved
      request.writeUInt16BE(internalPort, 4);
      request.writeUInt16BE(externalPort, 6);
      request.writeUInt32BE(lifetime, 8);

      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 3000);

      this.socket.once('message', (response) => {
        clearTimeout(timeout);

        try {
          const resultCode = response.readUInt16BE(2);

          if (resultCode !== 0) {
            throw new Error(`Error: result code ${resultCode}`);
          }

          const epoch = response.readUInt32BE(4);
          const mappedPort = response.readUInt16BE(10);
          const actualLifetime = response.readUInt32BE(12);

          resolve({
            externalPort: mappedPort,
            lifetime: actualLifetime,
            epoch
          });
        } catch (error) {
          reject(error);
        }
      });

      this.socket.send(request, this.gatewayPort, this.gatewayIP);
    });
  }

  deletePortMapping(internalPort, protocol = 'tcp') {
    return this.addPortMapping(internalPort, 0, protocol, 0);
  }

  close() {
    this.socket.close();
  }
}

// Usage
const os = require('os');

function getDefaultGateway() {
  // Simple gateway detection (platform-specific)
  const interfaces = os.networkInterfaces();
  // This is simplified - use proper gateway detection in production
  return '192.168.1.1';
}

const gateway = getDefaultGateway();
const client = new NATPMPClient(gateway);

async function main() {
  try {
    // Get external IP
    const { externalIP, epoch } = await client.getExternalIP();
    console.log(`External IP: ${externalIP}`);
    console.log(`Gateway uptime: ${epoch}s`);

    // Add port mapping
    const mapping = await client.addPortMapping(8080, 8080, 'tcp', 3600);
    console.log('Mapping created:');
    console.log(`  External: ${externalIP}:${mapping.externalPort}`);
    console.log(`  Lifetime: ${mapping.lifetime}s`);

    // Renew periodically
    setInterval(async () => {
      const renewed = await client.addPortMapping(8080, 8080, 'tcp', 3600);
      console.log(`Mapping renewed: ${renewed.lifetime}s`);
    }, 30 * 60 * 1000);  // Every 30 minutes

  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
```

## Mapping Lifetime Management

### Recommended Practices

```
1. Initial Lifetime
   - Request 3600 seconds (1 hour)
   - Gateway may grant less
   - Never request > 1 day

2. Renewal Strategy
   - Renew at 50% of lifetime
   - If lifetime is 3600s, renew at 1800s
   - Provides safety margin

3. Exponential Backoff
   - If renewal fails, retry with backoff
   - 1s, 2s, 4s, 8s, 16s, 32s
   - Eventually recreate mapping

4. Epoch Monitoring
   - Check epoch in each response
   - If epoch < last_epoch: gateway rebooted
   - Recreate all mappings

5. Cleanup
   - Always delete mappings when done
   - Set lifetime=0 to delete
   - Graceful shutdown
```

### Example: Lifetime Management

```python
class MappingManager:
    def __init__(self, client, internal_port, protocol='tcp'):
        self.client = client
        self.internal_port = internal_port
        self.protocol = protocol
        self.external_port = None
        self.lifetime = None
        self.last_epoch = None
        self.running = False

    def start(self):
        """Create and maintain mapping."""
        self.running = True

        # Create initial mapping
        self._create_mapping()

        # Renewal loop
        while self.running:
            # Sleep for half of lifetime
            sleep_time = self.lifetime / 2
            time.sleep(sleep_time)

            if not self.running:
                break

            try:
                # Renew mapping
                ext_port, lifetime, epoch = self.client.add_port_mapping(
                    self.internal_port,
                    self.external_port,  # Request same port
                    self.protocol,
                    3600
                )

                # Check for gateway reboot
                if epoch < self.last_epoch:
                    print("Gateway rebooted - mapping recreated")

                self.external_port = ext_port
                self.lifetime = lifetime
                self.last_epoch = epoch

                print(f"Mapping renewed: {lifetime}s")

            except Exception as e:
                print(f"Renewal failed: {e}")
                # Retry with backoff
                self._retry_with_backoff()

    def _create_mapping(self):
        """Create initial mapping."""
        ext_port, lifetime, epoch = self.client.add_port_mapping(
            self.internal_port,
            0,  # Any port
            self.protocol,
            3600
        )

        self.external_port = ext_port
        self.lifetime = lifetime
        self.last_epoch = epoch

        print(f"Mapping created: :{ext_port} -> localhost:{self.internal_port}")

    def _retry_with_backoff(self):
        """Retry with exponential backoff."""
        delays = [1, 2, 4, 8, 16, 32]

        for delay in delays:
            time.sleep(delay)
            try:
                self._create_mapping()
                return
            except Exception as e:
                print(f"Retry failed: {e}")

        print("All retries failed")
        self.running = False

    def stop(self):
        """Stop and delete mapping."""
        self.running = False

        try:
            self.client.delete_port_mapping(self.internal_port, self.protocol)
            print("Mapping deleted")
        except Exception as e:
            print(f"Failed to delete mapping: {e}")


# Usage
client = NATPMPClient(gateway)
manager = MappingManager(client, 8080, 'tcp')

# Start in background thread
import threading
thread = threading.Thread(target=manager.start)
thread.start()

# Application runs...

# Cleanup on exit
manager.stop()
thread.join()
client.close()
```

## Security Considerations

```
Threats:

1. Unauthorized Mappings
   - Malware can open ports
   - No authentication in protocol
   - Mitigation: Monitor gateway logs

2. Resource Exhaustion
   - Many mappings consume gateway resources
   - DoS via mapping requests
   - Mitigation: Gateway enforces limits

3. Information Disclosure
   - External IP revealed
   - Internal topology visible
   - Mitigation: Minimal, inherent to NAT

4. Spoofing
   - Off-path attacker sends fake responses
   - Mitigation: Check source IP/port

Best Practices:

1. Only request needed mappings
2. Use shortest lifetime necessary
3. Delete mappings when done
4. Monitor for unexpected mappings
5. Validate response source
6. Handle errors gracefully
```

## Troubleshooting

```bash
# Test if gateway supports NAT-PMP
nc -u 192.168.1.1 5351

# Send external IP request (hex)
echo -n "\x00\x00" | nc -u 192.168.1.1 5351

# Expected response (hex):
# 00 80 00 00 SSSS SSSS EE EE EE EE
# 00: Version
# 80: Opcode (128 = external IP response)
# 00 00: Result (success)
# SSSS SSSS: Epoch seconds
# EE EE EE EE: External IP

# tcpdump NAT-PMP traffic
sudo tcpdump -i any -n udp port 5351 -X

# Check if gateway has NAT-PMP enabled
# Router admin interface → Port Forwarding → NAT-PMP

# Common issues:
# - Gateway doesn't support NAT-PMP
# - NAT-PMP disabled in gateway
# - Firewall blocks UDP 5351
# - Wrong gateway address
# - Gateway behind another NAT

# Test with real client
pip install nat-pmp
natpmpc -g 192.168.1.1 -a 8080 8080 tcp 3600
```

## Comparison with Other Protocols

### NAT-PMP vs PCP

```
NAT-PMP:
+ Simple, easy to implement
+ Low overhead (12-16 bytes)
+ Widely supported (Apple devices)
+ Battle-tested (since 2005)
- IPv4 only
- Single NAT only
- Limited features

PCP:
+ IPv4 and IPv6
+ Multiple NATs
+ More features (PEER, filters)
+ Modern design
- More complex
- Less deployed
- Larger packets

Migration Path:
- PCP designed as NAT-PMP successor
- PCP port (5351) intentionally same
- Clients can try both
```

### Feature Comparison

```
Feature                  NAT-PMP    PCP       UPnP-IGD
Packet Size              12-16B     24+B      KB+
Round Trips              1          1         Multiple
IPv6                     No         Yes       Partial
Lifetime Management      Yes        Yes       No
Third-party Mapping      No         Yes       No
Firewall Control         No         Yes       No
Authentication           No         No        No
Complexity               Low        Medium    High
Apple Support            Native     Native    Emulated
Linux Support            Good       Good      Good
```

## Common Use Cases

### 1. BitTorrent Client

```python
# BitTorrent client
client = NATPMPClient(gateway)

# Map port for incoming connections
port = 6881
ext_port, lifetime, _ = client.add_port_mapping(
    internal_port=port,
    external_port=port,
    protocol='tcp',
    lifetime=7200  # 2 hours
)

print(f"Listening on port {ext_port}")

# Announce to tracker with external port
announce_to_tracker(ext_port)

# Maintain mapping while downloading
while downloading:
    time.sleep(3600)
    client.add_port_mapping(port, protocol='tcp', lifetime=7200)

# Cleanup
client.delete_port_mapping(port, 'tcp')
```

### 2. VoIP Application

```python
# VoIP client
client = NATPMPClient(gateway)

# Map SIP and RTP ports
sip_port = 5060
rtp_port = 16384

# SIP (TCP)
sip_ext, _, _ = client.add_port_mapping(
    sip_port, sip_port, 'tcp', 3600
)

# RTP (UDP)
rtp_ext, _, _ = client.add_port_mapping(
    rtp_port, rtp_port, 'udp', 3600
)

# Register with external address
external_ip, _ = client.get_external_ip()
register_with_server(external_ip, sip_ext, rtp_ext)
```

### 3. Game Server

```python
# Game server
client = NATPMPClient(gateway)

# Map game port
game_port = 27015
ext_port, lifetime, _ = client.add_port_mapping(
    game_port, game_port, 'udp', 7200
)

external_ip, _ = client.get_external_ip()

# Advertise server
advertise_server(f"{external_ip}:{ext_port}")

print(f"Server accessible at {external_ip}:{ext_port}")
```

## ELI10: NAT-PMP Explained Simply

NAT-PMP is like asking your house to automatically open a window:

### Without NAT-PMP
```
You: Want friend to visit
Problem: Door is locked
Solution: Ask parent to unlock door manually
Issue: Parent must remember, manual work
```

### With NAT-PMP
```
You: "Please open door for 1 hour"
Smart House: Opens door automatically
Friend: Can enter for 1 hour
Smart House: Locks door after 1 hour

Automatic + Safe!
```

### In Computer Terms
```
Your App: "Need port 8080 open for 1 hour"
Router: Opens port 8080 automatically
Internet: Can now reach your app on port 8080
Router: Closes port after 1 hour

Benefits:
- No manual configuration
- Automatic cleanup
- Time-limited (secure)
- Application controls it
```

## Further Resources

### Specifications
- [RFC 6886 - NAT-PMP](https://tools.ietf.org/html/rfc6886)
- [RFC 6887 - PCP](https://tools.ietf.org/html/rfc6887) (Successor)

### Implementations
- [libnatpmp](https://github.com/miniupnp/libnatpmp) - C library
- [nat-pmp](https://pypi.org/project/nat-pmp/) - Python
- [nat-upnp](https://www.npmjs.com/package/nat-upnp) - Node.js

### Tools
- natpmpc - Command-line client
- [NAT Port Mapping Protocol](https://en.wikipedia.org/wiki/NAT_Port_Mapping_Protocol)

### Apple Documentation
- [NAT-PMP on macOS](https://developer.apple.com/library/archive/qa/qa1458/_index.html)
- Bonjour implementation includes NAT-PMP
