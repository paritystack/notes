# QoS Management

Quality of Service (QoS) management in WiFi ensures that different types of traffic receive appropriate prioritization and treatment based on their requirements. Modern WiFi standards (802.11ax/WiFi 6 and later) introduce advanced QoS mechanisms that enable fine-grained control over how applications and streams are handled.

## Overview

WiFi QoS has evolved from basic WMM (WiFi Multimedia) access categories to sophisticated stream-based classification systems. The key QoS mechanisms include:

- **WMM (WiFi Multimedia)**: Foundation of WiFi QoS with 4 access categories
- **QoS Map**: Mapping between IP layer DSCP values and WiFi access categories
- **MSCS (Mirrored Stream Classification Service)**: Stream classification for uplink traffic
- **SCS (Stream Classification Service)**: Bidirectional stream classification and QoS
- **DSCP Policy**: Network-driven QoS policy for applications

### Access Categories (WMM)

WiFi Multimedia (WMM) defines four access categories for traffic prioritization:

| Access Category | Acronym | Priority | Typical Use Cases |
|----------------|---------|----------|-------------------|
| Voice | AC_VO | Highest | VoIP, real-time voice |
| Video | AC_VI | High | Video streaming, conferencing |
| Best Effort | AC_BE | Normal | General internet, web browsing |
| Background | AC_BK | Lowest | File downloads, backups |

Each access category has different EDCA (Enhanced Distributed Channel Access) parameters that control channel access timing and contention behavior.

## QoS Map

QoS Map provides the mechanism to translate IP layer QoS markings (DSCP values) to WiFi access categories. This enables end-to-end QoS from the application layer through the WiFi network.

### Purpose

- Bridges Layer 3 (IP) QoS to Layer 2 (WiFi) QoS
- Allows applications to signal their QoS requirements using DSCP
- Enables consistent QoS treatment across wired and wireless networks
- Configurable by the Access Point (AP) and communicated to clients

### DSCP to Access Category Mapping

The default mapping follows RFC 8325 recommendations, but can be customized:

```
DSCP Range          → Access Category
------------------------------------
EF (46), CS6 (48)   → AC_VO (Voice)
AF41-AF43 (34-38)   → AC_VI (Video)
CS4 (32), AF31-AF33 → AC_VI (Video)
AF21-AF23 (18-22)   → AC_BE (Best Effort)
CS0 (0), DF (0)     → AC_BE (Best Effort)
CS1 (8), CS2 (16)   → AC_BK (Background)
```

### Configuration

QoS Map is negotiated during association and can be updated dynamically:

1. **AP Advertisement**: AP includes QoS Map Set element in association response
2. **Client Processing**: Client applies the mapping to outbound traffic
3. **Dynamic Updates**: AP can send QoS Map Configure frames to update mapping

### QoS Map Set Format

The QoS Map Set element consists of:
- **DSCP Exception fields**: Individual DSCP values mapped to specific ACs
- **DSCP Range fields**: Continuous ranges of DSCP values mapped to ACs

Example QoS Map configuration:
```
Exceptions:
  DSCP 46 → AC_VO
  DSCP 34 → AC_VI

Ranges:
  DSCP 0-7   → AC_BK
  DSCP 8-15  → AC_BE
  DSCP 16-31 → AC_BE
  DSCP 32-47 → AC_VI
  DSCP 48-63 → AC_VO
```

### Use Cases

- **Enterprise Networks**: Ensure voice/video traffic gets priority
- **Carrier WiFi**: Apply operator QoS policies to subscriber traffic
- **Home Networks**: Prioritize gaming or video streaming over downloads
- **Public Hotspots**: Differentiate service tiers based on QoS

### Implementation Notes

- Clients should honor the QoS Map provided by the AP
- Upstream traffic classification uses DSCP-to-AC mapping
- Downstream traffic is classified by the AP before transmission
- QoS Map support is mandatory in WiFi 6 certified devices

## MSCS (Mirrored Stream Classification Service)

MSCS, introduced in 802.11aa and enhanced in 802.11ax, allows a client (STA) to request that the AP mirror the QoS classification applied to a specific traffic stream. This is particularly useful for uplink traffic where the client knows the application requirements.

### Purpose

- Client-initiated QoS classification for uplink streams
- Ensures consistent QoS treatment in both directions
- Reduces latency for time-sensitive applications
- Optimizes airtime usage for classified streams

### How MSCS Works

1. **Stream Detection**: Client identifies a traffic stream (by 5-tuple: src IP, dst IP, src port, dst port, protocol)
2. **MSCS Request**: Client sends MSCS Request frame to AP with stream classifiers
3. **AP Processing**: AP classifies the stream and applies appropriate QoS
4. **Mirroring**: AP applies the same classification to corresponding downstream traffic
5. **MSCS Response**: AP confirms acceptance with MSCS Response frame

### Stream Classification

MSCS uses TCLAS (Traffic Classification) elements to identify streams:

```
TCLAS Elements:
- Classifier Type 4 (IP and higher layer parameters):
  • Source IP address
  • Destination IP address
  • Source port
  • Destination port
  • Protocol (TCP/UDP)
  • DSCP value
```

### MSCS Frame Exchange

```
Client (STA)                          Access Point (AP)
     |                                       |
     |  MSCS Request (TCLAS, QoS params)    |
     |-------------------------------------->|
     |                                       | [Process & Classify]
     |       MSCS Response (Accept/Reject)   |
     |<--------------------------------------|
     |                                       |
     |  [Uplink stream with QoS]            |
     |-------------------------------------->|
     |                                       |
     |  [Downlink stream with mirrored QoS] |
     |<--------------------------------------|
```

### MSCS Parameters

- **Stream Timeout**: Duration for which classification remains active
- **TCLAS Processing**: How multiple TCLAS elements are combined (AND/OR)
- **User Priority**: Requested user priority (0-7)
- **Stream Status**: Active, inactive, or being modified

### Use Cases

- **Video Conferencing**: Ensure low latency for bidirectional video/audio
- **Online Gaming**: Prioritize game traffic for minimal lag
- **VoIP Applications**: QoS for voice calls over WiFi
- **Industrial IoT**: Time-sensitive sensor data and control traffic

### Benefits

- **Application-Aware QoS**: Applications directly signal their requirements
- **Reduced Overhead**: AP doesn't need deep packet inspection
- **Bidirectional Consistency**: Same QoS in both directions
- **Dynamic Classification**: Can be updated as application needs change

### Limitations

- Requires WiFi 6 or later
- Client and AP must both support MSCS
- Limited number of concurrent streams (implementation-dependent)
- Stream identification requires stable 5-tuple (challenging with NAT)

## SCS (Stream Classification Service)

SCS, introduced in 802.11be (WiFi 7), is an evolution of MSCS that provides more comprehensive stream classification capabilities. SCS supports both uplink and downlink stream classification with enhanced flexibility.

### Purpose

- Advanced stream-based QoS for WiFi 7 networks
- Bidirectional stream classification with independent parameters
- Support for complex traffic patterns and multiple streams
- Enable application-specific QoS policies

### SCS vs MSCS

| Feature | MSCS | SCS |
|---------|------|-----|
| Standard | 802.11ax (WiFi 6) | 802.11be (WiFi 7) |
| Direction | Primarily uplink | Bidirectional |
| Complexity | Basic stream identification | Advanced classification |
| Stream Control | Mirrored QoS | Independent QoS per direction |
| Scalability | Limited streams | More concurrent streams |

### SCS Architecture

SCS provides a framework for:
1. **Stream Identification**: Flexible classifiers beyond 5-tuple
2. **QoS Assignment**: Per-stream QoS parameters
3. **Stream Grouping**: Multiple related streams with coordinated QoS
4. **Dynamic Adaptation**: Runtime adjustment based on network conditions

### SCS Request/Response

```
Client initiates SCS:
┌─────────────────────────────────────────┐
│ SCS Request Frame                       │
├─────────────────────────────────────────┤
│ • Stream ID(s)                          │
│ • TCLAS elements (stream classifiers)   │
│ • TCLAS Processing rule                 │
│ • QoS Characteristics:                  │
│   - Service class                       │
│   - Minimum data rate                   │
│   - Maximum latency                     │
│   - Mean data rate                      │
│   - Burst size                          │
│ • Stream timeout                        │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ SCS Response Frame                      │
├─────────────────────────────────────────┤
│ • Stream ID                             │
│ • Status (Accept/Reject/Modify)         │
│ • Accepted QoS parameters               │
│ • Alternative suggestions (if rejected) │
└─────────────────────────────────────────┘
```

### Enhanced Classification

SCS supports advanced classifiers:

- **Layer 2**: MAC addresses, Ethernet type
- **Layer 3**: IPv4/IPv6 addresses, DSCP, flow label
- **Layer 4**: TCP/UDP ports, protocol type
- **Application Layer**: URL patterns, application signatures
- **Temporal**: Time-of-day based classification

### QoS Characteristics

SCS allows specifying detailed QoS requirements:

```
Service Class Types:
• BE (Best Effort): Default internet traffic
• BK (Background): Bulk data transfer
• EE (Excellent Effort): Better than BE, not time-critical
• CL (Controlled Load): Moderate latency requirements
• VI (Video): Low latency, moderate jitter tolerance
• VO (Voice): Ultra-low latency, minimal jitter
• NC (Network Control): Critical network management
```

Performance Parameters:
- **Minimum Data Rate**: Guaranteed throughput
- **Maximum Latency**: Latency bound for the stream
- **Peak Data Rate**: Burst handling capacity
- **Mean Data Rate**: Average throughput requirement
- **Burst Size**: Maximum burst the stream will generate
- **Delay Bound**: Maximum acceptable delay

### Multi-Link Operation (MLO) Support

In WiFi 7, SCS integrates with Multi-Link Operation:

- **Per-Link Classification**: Different QoS on different links
- **Link Aggregation**: Combine links for high-priority streams
- **Load Balancing**: Distribute streams across links based on QoS
- **Failover**: Automatic stream migration on link failure

### SCS Stream States

```
Stream Lifecycle:
┌─────────┐  Request  ┌─────────┐  Traffic   ┌────────┐
│ Pending │─────────→│ Active  │──────────→│ Active │
└─────────┘  Accepted └─────────┘  Flowing   └────────┘
                                                  │
                                                  │ Timeout
                                                  ↓
                                              ┌────────┐
                                              │ Ended  │
                                              └────────┘
```

### Use Cases

- **8K Video Streaming**: High bandwidth, low latency requirements
- **Cloud Gaming**: Ultra-low latency with guaranteed throughput
- **AR/VR Applications**: Strict latency and jitter requirements
- **Multi-Stream Apps**: Different QoS for video, audio, control channels
- **Enterprise Collaboration**: Simultaneous video, voice, screen sharing

### Implementation Considerations

- **Stream Limits**: APs have finite resources for concurrent SCS streams
- **Admission Control**: APs may reject requests if resources unavailable
- **Fallback Mechanisms**: Applications should handle SCS rejection gracefully
- **Battery Impact**: SCS requires active stream management (power trade-off)

## DSCP Policy

DSCP (Differentiated Services Code Point) Policy enables network operators and enterprises to enforce QoS policies at the network edge. The AP communicates DSCP policy to clients, instructing them how to mark their traffic.

### Purpose

- Network-driven QoS policy enforcement
- Standardize DSCP marking across all clients
- Enable operator/enterprise control over application QoS
- Simplify QoS configuration for end users

### DSCP Policy Framework

DSCP Policy consists of:
1. **Policy Advertisement**: AP advertises supported policies
2. **Policy Query**: Client can query for specific policies
3. **Policy Application**: Client marks traffic according to policy
4. **Policy Update**: Dynamic policy changes propagated to clients

### DSCP Policy Element

The DSCP Policy element includes:

```
Policy Attributes:
┌──────────────────────────────────────┐
│ • Policy ID                          │
│ • Request Type Control               │
│ • Domain Name (e.g., *.company.com)  │
│ • DSCP Value(s) to apply             │
│ • Port Range                         │
│ • Protocol (TCP/UDP/both)            │
│ • Direction (uplink/downlink/both)   │
│ • Policy Lifetime                    │
└──────────────────────────────────────┘
```

### Policy Types

1. **Domain-Based Policy**
   - Apply DSCP based on domain name (URL)
   - Example: "*.zoom.us" → DSCP EF (46)
   - Useful for SaaS applications

2. **Application-Based Policy**
   - Identify applications by signature
   - Apply appropriate DSCP marking
   - Example: "Microsoft Teams" → DSCP 34

3. **Port-Based Policy**
   - Traditional port-based classification
   - Example: Port 5060 (SIP) → DSCP EF

4. **Protocol-Based Policy**
   - Classify by protocol type
   - Example: UDP → DSCP 34 (for RTP)

### Policy Query and Response

```
Client                                    AP
  |                                        |
  |  DSCP Policy Query                     |
  |  (Request policies for domain/app)     |
  |--------------------------------------->|
  |                                        |
  |  DSCP Policy Response                  |
  |  (Policy elements for requested items) |
  |<---------------------------------------|
  |                                        |
  |  Apply DSCP marking to traffic         |
  |--------------------------------------->|
```

### Example Policy Configurations

**Enterprise Video Conferencing**
```
Policy: Zoom
  Domain: *.zoom.us
  DSCP: 34 (AF41) for video
  DSCP: 46 (EF) for audio
  Direction: Both
```

**VoIP Services**
```
Policy: VoIP
  Ports: 5060-5061 (SIP), 10000-20000 (RTP)
  Protocol: UDP
  DSCP: 46 (EF)
  Direction: Both
```

**Cloud Storage (Background)**
```
Policy: Cloud Backup
  Domain: *.dropbox.com, *.onedrive.com
  DSCP: 8 (CS1)
  Direction: Uplink
```

### Policy Enforcement

- **Client-Side Marking**: Client marks packets according to policy
- **AP Verification**: AP can verify and override if needed
- **Policy Hierarchy**: More specific policies override general ones
- **Default Behavior**: Unmarked traffic uses default QoS Map

### Integration with QoS Map

DSCP Policy and QoS Map work together:

```
Application → DSCP Policy → DSCP Marking → QoS Map → Access Category

Example:
Zoom Call → "Zoom Policy" → DSCP 46 → QoS Map → AC_VO
```

### Benefits

- **Centralized Management**: Network admins control QoS policy
- **Consistency**: All clients use same DSCP markings
- **Application Awareness**: Policies based on actual applications
- **Flexibility**: Policies can be updated without client changes
- **Multi-Vendor**: Works across different client devices

### Use Cases

1. **Enterprise Networks**
   - Prioritize business-critical applications (Teams, Zoom)
   - Deprioritize personal streaming services
   - Enforce bandwidth policies per application

2. **Carrier WiFi**
   - Differentiate service tiers
   - Prioritize operator services
   - Enforce fair usage policies

3. **Public Hotspots**
   - Premium QoS for paid tiers
   - Basic QoS for free access
   - Protect against bandwidth abuse

4. **Educational Institutions**
   - Prioritize learning platforms
   - Limit gaming and streaming
   - Ensure fair access for all users

### Implementation Requirements

- **DNS-Based Identification**: Many policies rely on domain names
- **TLS/HTTPS Support**: Policy must work with encrypted traffic
- **Client Support**: Requires WiFi 6 (802.11ax) or later
- **Policy Storage**: Clients cache policies for performance
- **Privacy Considerations**: Domain-based policies may reveal user activity

### Security Considerations

- **Policy Authenticity**: Ensure policies come from legitimate AP
- **Privacy**: Domain monitoring for policy application
- **Tampering**: Prevent malicious policy injection
- **Override Protection**: AP can override client markings if needed

## Comparison of QoS Mechanisms

| Feature | QoS Map | MSCS | SCS | DSCP Policy |
|---------|---------|------|-----|-------------|
| **Standard** | 802.11 | 802.11ax | 802.11be | 802.11ax |
| **Direction** | Both | Uplink (mirrored) | Both | Both |
| **Granularity** | DSCP ranges | Per-stream | Per-stream | Per-app/domain |
| **Initiated By** | AP | Client | Client | AP |
| **Complexity** | Low | Medium | High | Medium |
| **Application Awareness** | No | Partial | Yes | Yes |
| **Dynamic** | Semi | Yes | Yes | Yes |

## Best Practices

### For Network Administrators

1. **Start with QoS Map**: Establish baseline DSCP-to-AC mapping
2. **Enable DSCP Policy**: Define policies for known applications
3. **Monitor SCS/MSCS Usage**: Understand application QoS needs
4. **Admission Control**: Limit concurrent high-priority streams
5. **Test and Validate**: Verify QoS behavior with real applications

### For Application Developers

1. **Use Standard DSCP Values**: Follow RFC 8325 recommendations
2. **Request MSCS/SCS**: For latency-sensitive applications
3. **Handle Rejection**: Gracefully degrade if QoS not available
4. **Minimize Streams**: Don't over-request high-priority QoS
5. **Test Without QoS**: Ensure app works on basic WiFi

### For End Users

1. **Update Firmware**: Ensure AP supports modern QoS features
2. **WiFi 6/7 Devices**: Newer devices have better QoS support
3. **Prioritize Applications**: Configure router to prioritize important traffic
4. **Monitor Performance**: Use QoS-aware monitoring tools

## Troubleshooting QoS Issues

### Common Problems

1. **QoS Not Working**
   - Check if AP and client both support the QoS mechanism
   - Verify QoS is enabled on the AP
   - Ensure DSCP markings are preserved through the network

2. **Inconsistent Performance**
   - Check for QoS Map mismatches
   - Verify MSCS/SCS requests are accepted
   - Monitor airtime usage per access category

3. **High Priority Traffic Not Prioritized**
   - Verify DSCP markings are correct
   - Check QoS Map configuration
   - Ensure WMM is enabled

### Diagnostic Commands (Linux)

```bash
# Check QoS capabilities
iw dev wlan0 info | grep -i qos

# View current QoS Map
iw dev wlan0 station dump | grep -i "qos\|wmm"

# Monitor WiFi QoS statistics
tc -s qdisc show dev wlan0

# Capture QoS frames
tcpdump -i wlan0 -v 'type mgt subtype action'
```

### Wireshark Analysis

Filter for QoS-related frames:
```
# QoS Map frames
wlan.fixed.action_code == 4

# MSCS frames
wlan.fixed.action_code == 5

# SCS frames
wlan.fixed.action_code == 6

# DSCP Policy frames
wlan.ext_tag.number == 108
```

## Future Developments

### WiFi 7 Enhancements

- **Enhanced SCS**: More sophisticated stream classification
- **Multi-Link QoS**: Coordinated QoS across multiple links
- **AI-Driven QoS**: Machine learning for dynamic QoS optimization
- **Latency Guarantees**: Stricter bounds for time-sensitive traffic

### Emerging Use Cases

- **Extended Reality (XR)**: Ultra-low latency for AR/VR
- **Cloud Gaming**: Guaranteed performance for game streaming
- **Autonomous Vehicles**: V2X communication with QoS
- **Industrial Automation**: Deterministic WiFi for Industry 4.0

## References

- **IEEE 802.11-2020**: WiFi standard with QoS Map and MSCS
- **IEEE 802.11be**: WiFi 7 with SCS enhancements
- **RFC 8325**: Mapping DSCP to WiFi Access Categories
- **Wi-Fi Alliance**: WMM and QoS certification programs
- **IETF Diffserv**: DSCP definitions and usage

---

Modern WiFi QoS mechanisms provide sophisticated tools for ensuring application performance in wireless networks. By understanding and properly implementing QoS Map, MSCS, SCS, and DSCP Policy, networks can deliver excellent user experiences for latency-sensitive and bandwidth-intensive applications.
