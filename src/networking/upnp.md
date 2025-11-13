# UPnP (Universal Plug and Play)

## Overview

UPnP is a set of networking protocols that enables devices on a network to seamlessly discover each other and establish functional network services for data sharing, communications, and entertainment. It allows devices to automatically configure themselves and announce their presence to other devices.

## UPnP Components

```
1. Discovery (SSDP)
   - Find devices on network
   - Announce presence

2. Description
   - Device capabilities
   - Services offered

3. Control
   - Invoke actions
   - Query state

4. Eventing
   - Subscribe to state changes
   - Receive notifications

5. Presentation
   - Web-based UI
   - Human interaction
```

## UPnP Architecture

```
Control Point (Client)         Device (Server)
       |                            |
       | 1. Discovery (SSDP)        |
       |<-------------------------->|
       |                            |
       | 2. Description (XML)       |
       |--------------------------->|
       |<---------------------------|
       |                            |
       | 3. Control (SOAP)          |
       |--------------------------->|
       |<---------------------------|
       |                            |
       | 4. Eventing (GENA)         |
       |--------------------------->|
       | (Subscribe)                |
       |<---------------------------|
       | (Events)                   |
```

## SSDP (Simple Service Discovery Protocol)

### Discovery Process

**Device Announcement:**
```
Device joins network:

NOTIFY * HTTP/1.1
Host: 239.255.255.250:1900
Cache-Control: max-age=1800
Location: http://192.168.1.100:8080/description.xml
NT: upnp:rootdevice
NTS: ssdp:alive
Server: Linux/5.4 UPnP/1.0 MyDevice/1.0
USN: uuid:12345678-1234-1234-1234-123456789abc::upnp:rootdevice

Sent to multicast address 239.255.255.250:1900
Announces device presence
```

**Device Search (M-SEARCH):**
```
Control point searches for devices:

M-SEARCH * HTTP/1.1
Host: 239.255.255.250:1900
Man: "ssdp:discover"
ST: ssdp:all
MX: 3
(Search for all devices, wait up to 3 seconds)

Multicast to 239.255.255.250:1900
```

**Device Response:**
```
HTTP/1.1 200 OK
Cache-Control: max-age=1800
Location: http://192.168.1.100:8080/description.xml
Server: Linux/5.4 UPnP/1.0 MyDevice/1.0
ST: upnp:rootdevice
USN: uuid:12345678-1234-1234-1234-123456789abc::upnp:rootdevice

Unicast response back to control point
```

### SSDP Multicast

```
IPv4 Address: 239.255.255.250
Port: 1900 (UDP)

All UPnP devices listen on this address
Used for discovery announcements
```

### Search Targets (ST)

```
ssdp:all                    - All devices and services
upnp:rootdevice             - Root devices only
uuid:<device-uuid>          - Specific device
urn:schemas-upnp-org:device:<deviceType>:<version>
urn:schemas-upnp-org:service:<serviceType>:<version>

Examples:
  ST: urn:schemas-upnp-org:device:MediaRenderer:1
  ST: urn:schemas-upnp-org:service:ContentDirectory:1
```

## Device Description

### Description XML

```xml
<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <specVersion>
    <major>1</major>
    <minor>0</minor>
  </specVersion>
  <device>
    <deviceType>urn:schemas-upnp-org:device:MediaRenderer:1</deviceType>
    <friendlyName>Living Room TV</friendlyName>
    <manufacturer>Samsung</manufacturer>
    <manufacturerURL>http://www.samsung.com</manufacturerURL>
    <modelDescription>Smart TV</modelDescription>
    <modelName>UN55TU8000</modelName>
    <modelNumber>8000</modelNumber>
    <serialNumber>123456789</serialNumber>
    <UDN>uuid:12345678-1234-1234-1234-123456789abc</UDN>
    <presentationURL>http://192.168.1.100:8080/</presentationURL>

    <serviceList>
      <service>
        <serviceType>urn:schemas-upnp-org:service:AVTransport:1</serviceType>
        <serviceId>urn:upnp-org:serviceId:AVTransport</serviceId>
        <SCPDURL>/service/AVTransport/scpd.xml</SCPDURL>
        <controlURL>/service/AVTransport/control</controlURL>
        <eventSubURL>/service/AVTransport/event</eventSubURL>
      </service>
    </serviceList>
  </device>
</root>
```

### Service Description (SCPD)

```xml
<?xml version="1.0"?>
<scpd xmlns="urn:schemas-upnp-org:service-1-0">
  <specVersion>
    <major>1</major>
    <minor>0</minor>
  </specVersion>

  <actionList>
    <action>
      <name>Play</name>
      <argumentList>
        <argument>
          <name>Speed</name>
          <direction>in</direction>
          <relatedStateVariable>TransportPlaySpeed</relatedStateVariable>
        </argument>
      </argumentList>
    </action>

    <action>
      <name>Stop</name>
    </action>
  </actionList>

  <serviceStateTable>
    <stateVariable sendEvents="yes">
      <name>TransportState</name>
      <dataType>string</dataType>
      <allowedValueList>
        <allowedValue>PLAYING</allowedValue>
        <allowedValue>STOPPED</allowedValue>
        <allowedValue>PAUSED_PLAYBACK</allowedValue>
      </allowedValueList>
    </stateVariable>

    <stateVariable sendEvents="no">
      <name>TransportPlaySpeed</name>
      <dataType>string</dataType>
      <defaultValue>1</defaultValue>
    </stateVariable>
  </serviceStateTable>
</scpd>
```

## UPnP Control (SOAP)

### Action Invocation

**Request:**
```xml
POST /service/AVTransport/control HTTP/1.1
Host: 192.168.1.100:8080
Content-Type: text/xml; charset="utf-8"
SOAPAction: "urn:schemas-upnp-org:service:AVTransport:1#Play"
Content-Length: 299

<?xml version="1.0"?>
<s:Envelope
  xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
  s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
  <s:Body>
    <u:Play xmlns:u="urn:schemas-upnp-org:service:AVTransport:1">
      <InstanceID>0</InstanceID>
      <Speed>1</Speed>
    </u:Play>
  </s:Body>
</s:Envelope>
```

**Response:**
```xml
HTTP/1.1 200 OK
Content-Type: text/xml; charset="utf-8"
Content-Length: 250

<?xml version="1.0"?>
<s:Envelope
  xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
  s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
  <s:Body>
    <u:PlayResponse xmlns:u="urn:schemas-upnp-org:service:AVTransport:1">
    </u:PlayResponse>
  </s:Body>
</s:Envelope>
```

### Error Response

```xml
HTTP/1.1 500 Internal Server Error
Content-Type: text/xml; charset="utf-8"

<?xml version="1.0"?>
<s:Envelope
  xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
  s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
  <s:Body>
    <s:Fault>
      <faultcode>s:Client</faultcode>
      <faultstring>UPnPError</faultstring>
      <detail>
        <UPnPError xmlns="urn:schemas-upnp-org:control-1-0">
          <errorCode>701</errorCode>
          <errorDescription>Transition not available</errorDescription>
        </UPnPError>
      </detail>
    </s:Fault>
  </s:Body>
</s:Envelope>
```

## UPnP Eventing (GENA)

### Subscribe to Events

**Request:**
```
SUBSCRIBE /service/AVTransport/event HTTP/1.1
Host: 192.168.1.100:8080
Callback: <http://192.168.1.50:8888/notify>
NT: upnp:event
Timeout: Second-1800
```

**Response:**
```
HTTP/1.1 200 OK
SID: uuid:subscription-12345
Timeout: Second-1800
```

### Initial Event (State Snapshot)

```xml
NOTIFY /notify HTTP/1.1
Host: 192.168.1.50:8888
Content-Type: text/xml
NT: upnp:event
NTS: upnp:propchange
SID: uuid:subscription-12345
SEQ: 0

<?xml version="1.0"?>
<e:propertyset xmlns:e="urn:schemas-upnp-org:event-1-0">
  <e:property>
    <TransportState>STOPPED</TransportState>
  </e:property>
  <e:property>
    <CurrentTrack>1</CurrentTrack>
  </e:property>
</e:propertyset>
```

### Subsequent Events

```xml
NOTIFY /notify HTTP/1.1
Host: 192.168.1.50:8888
Content-Type: text/xml
NT: upnp:event
NTS: upnp:propchange
SID: uuid:subscription-12345
SEQ: 1

<?xml version="1.0"?>
<e:propertyset xmlns:e="urn:schemas-upnp-org:event-1-0">
  <e:property>
    <TransportState>PLAYING</TransportState>
  </e:property>
</e:propertyset>
```

### Unsubscribe

```
UNSUBSCRIBE /service/AVTransport/event HTTP/1.1
Host: 192.168.1.100:8080
SID: uuid:subscription-12345
```

## UPnP IGD (Internet Gateway Device)

### Port Mapping

One of the most common UPnP uses:

**Add Port Mapping Request:**
```xml
POST /control/WANIPConnection HTTP/1.1
Host: 192.168.1.1:5000
Content-Type: text/xml; charset="utf-8"
SOAPAction: "urn:schemas-upnp-org:service:WANIPConnection:1#AddPortMapping"

<?xml version="1.0"?>
<s:Envelope ...>
  <s:Body>
    <u:AddPortMapping xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
      <NewRemoteHost></NewRemoteHost>
      <NewExternalPort>8080</NewExternalPort>
      <NewProtocol>TCP</NewProtocol>
      <NewInternalPort>8080</NewInternalPort>
      <NewInternalClient>192.168.1.50</NewInternalClient>
      <NewEnabled>1</NewEnabled>
      <NewPortMappingDescription>My Web Server</NewPortMappingDescription>
      <NewLeaseDuration>0</NewLeaseDuration>
    </u:AddPortMapping>
  </s:Body>
</s:Envelope>
```

**Result:**
```
External: <public-ip>:8080
    ↓
Internal: 192.168.1.50:8080

Automatic NAT traversal!
```

### Get External IP

```xml
POST /control/WANIPConnection HTTP/1.1
SOAPAction: "urn:schemas-upnp-org:service:WANIPConnection:1#GetExternalIPAddress"

<u:GetExternalIPAddress xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">
</u:GetExternalIPAddress>
```

**Response:**
```xml
<u:GetExternalIPAddressResponse>
  <NewExternalIPAddress>203.0.113.5</NewExternalIPAddress>
</u:GetExternalIPAddressResponse>
```

## Common UPnP Device Types

```
MediaServer              - Content provider (NAS, PC)
MediaRenderer            - Content consumer (TV, speaker)
InternetGatewayDevice    - Router/NAT
WANConnectionDevice      - WAN connection management
PrinterBasic             - Network printer
Scanner                  - Network scanner
HVAC                     - Heating/cooling control
Lighting                 - Smart lights
SecurityDevice           - Cameras, sensors
```

## UPnP Client Implementation

### Python Example (Discovery)

```python
import socket

SSDP_ADDR = '239.255.255.250'
SSDP_PORT = 1900

def discover_devices():
    # M-SEARCH message
    message = '\r\n'.join([
        'M-SEARCH * HTTP/1.1',
        f'Host: {SSDP_ADDR}:{SSDP_PORT}',
        'Man: "ssdp:discover"',
        'ST: ssdp:all',
        'MX: 3',
        '',
        ''
    ])

    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    # Send M-SEARCH
    sock.sendto(message.encode(), (SSDP_ADDR, SSDP_PORT))

    # Receive responses
    devices = []
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            response = data.decode()

            # Parse location
            for line in response.split('\r\n'):
                if line.startswith('Location:'):
                    location = line.split(':', 1)[1].strip()
                    devices.append(location)
                    break
    except socket.timeout:
        pass

    sock.close()
    return devices

# Usage
devices = discover_devices()
for device in devices:
    print(f"Found device: {device}")
```

### Python Example (Control)

```python
import requests
import xml.etree.ElementTree as ET

def control_device(control_url, service_type, action, args):
    # Build SOAP envelope
    soap_body = f'''<?xml version="1.0"?>
    <s:Envelope
      xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
      s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
      <s:Body>
        <u:{action} xmlns:u="{service_type}">
          {''.join(f'<{k}>{v}</{k}>' for k, v in args.items())}
        </u:{action}>
      </s:Body>
    </s:Envelope>'''

    headers = {
        'Content-Type': 'text/xml; charset="utf-8"',
        'SOAPAction': f'"{service_type}#{action}"'
    }

    response = requests.post(control_url, data=soap_body, headers=headers)
    return response.text

# Usage
control_url = 'http://192.168.1.100:8080/service/AVTransport/control'
service_type = 'urn:schemas-upnp-org:service:AVTransport:1'
action = 'Play'
args = {'InstanceID': '0', 'Speed': '1'}

result = control_device(control_url, service_type, action, args)
print(result)
```

## UPnP Tools

### Command Line Tools

```bash
# upnpc (miniupnpc)
# Install: apt-get install miniupnpc

# Discover IGD devices
upnpc -l

# Get external IP
upnpc -s

# Add port mapping
upnpc -a 192.168.1.50 8080 8080 TCP

# List port mappings
upnpc -L

# Delete port mapping
upnpc -d 8080 TCP
```

### GUI Tools

```
- UPnP Inspector (Linux)
- UPnP Test Tool (Windows)
- Device Spy (UPnP Forum)
```

## UPnP Security Issues

### Major Vulnerabilities

#### 1. No Authentication

```
Any device can control any other device
No password required
No encryption

Attack: Malicious app opens ports in router
```

#### 2. Port Forwarding Abuse

```
Malware can:
  - Open ports in router
  - Expose internal services
  - Create backdoors

Example:
  Malware opens port 3389 (RDP)
  Attacker can remotely access PC
```

#### 3. SSDP Amplification DDoS

```
Attacker spoofs source IP as victim
Sends M-SEARCH to many UPnP devices
Devices respond to victim
Victim overwhelmed with traffic

Amplification factor: 30x-50x
```

#### 4. XML External Entity (XXE)

```
Malicious device description:
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>

Can read local files
Server-side request forgery
```

### Security Best Practices

```
1. Disable UPnP on router
   - If not needed, turn it off
   - Most secure option

2. Use UPnP-UP (UPnP with User Profile)
   - Authentication layer
   - Access control

3. Firewall rules
   - Block SSDP multicast from WAN
   - Limit UPnP to trusted VLANs

4. Whitelist devices
   - Only allow known devices
   - MAC address filtering

5. Monitor port mappings
   - Regular audits
   - Alert on unexpected changes

6. Update firmware
   - Patch vulnerabilities
   - Keep devices current
```

## UPnP vs Alternatives

### vs Manual Port Forwarding

```
UPnP:
  Pros: Automatic, easy
  Cons: Security risk, no control

Manual:
  Pros: Secure, controlled
  Cons: Technical knowledge required
```

### vs NAT-PMP / PCP

```
NAT-PMP (Apple):
  - Similar to UPnP
  - Simpler protocol
  - Better security

PCP (Port Control Protocol):
  - Successor to NAT-PMP
  - IETF standard
  - IPv6 support
```

### vs STUN/TURN

```
UPnP: Local network discovery and control
STUN/TURN: NAT traversal for P2P connections

Different use cases, can complement each other
```

## ELI10

UPnP is like devices introducing themselves and asking for help:

**Discovery (Meeting New Friends):**
```
New TV joins network:
  TV: "Hi everyone! I'm a TV and can play videos!"
  All devices hear the announcement
  Your phone: "Cool, I found a TV!"
```

**Control (Asking for Favors):**
```
Phone to TV: "Can you play this video?"
TV: "Sure! Playing now."

Gaming console to router: "Can you open port 3478?"
Router: "Done! Port is open."
```

**Problems (Security Issues):**
```
Bad actor: "Hey router, open all ports!"
Router: "OK!" (No questions asked)
  → This is dangerous!

Better approach:
  Router: "Who are you? Do you have permission?"
  Bad actor: "Uh... never mind."
```

**When to Use:**
- Home media streaming
- Gaming (automatic port opening)
- Smart home devices
- Printing

**When to Disable:**
- Public networks
- When security is critical
- Enterprise environments
- If you don't need it

**Rule of Thumb:**
- Home network: Convenient (but understand risks)
- Business network: Usually disable
- Gaming: Helpful for matchmaking
- Important: Monitor what ports get opened!

## Further Resources

- [UPnP Forum](https://openconnectivity.org/developer/specifications/upnp-resources)
- [RFC 6970 - UPnP IGD-PCP Interworking](https://tools.ietf.org/html/rfc6970)
- [UPnP Device Architecture](http://upnp.org/specs/arch/UPnP-arch-DeviceArchitecture-v2.0.pdf)
- [miniupnpc Library](https://miniupnp.tuxfamily.org/)
- [Security Concerns](https://www.upnp-hacks.org/)
