# Networking

Comprehensive networking reference covering protocols, models, and networking fundamentals.

## Networking Models

### [OSI Model](osi_model.md)
The 7-layer conceptual framework for network communication:
- Layer 7: Application
- Layer 6: Presentation
- Layer 5: Session
- Layer 4: Transport
- Layer 3: Network
- Layer 2: Data Link
- Layer 1: Physical

### [TCP/IP Model](tcp_ip_model.md)
The practical 4-layer model used in modern networks:
- Application Layer
- Transport Layer
- Internet Layer
- Network Access Layer

## Layer 2 / Layer 3 Fundamentals

### [ARP (Address Resolution Protocol)](arp.md)
- IP-to-MAC mapping on local networks
- Request/reply exchange and cache lifecycle
- Gratuitous ARP, proxy ARP, ARP probes
- ARP spoofing attacks and Dynamic ARP Inspection
- NDP comparison for IPv6

### [Ethernet & VLAN](ethernet_vlan.md)
- Ethernet frame format and MAC addresses
- Switch MAC learning and forwarding
- 802.1Q VLAN tagging, access vs trunk ports
- Native VLAN and VLAN hopping
- Spanning Tree Protocol (STP, RSTP)
- Link aggregation (LACP), PoE, jumbo frames

### [DHCP (Dynamic Host Configuration Protocol)](dhcp.md)
- DORA exchange (Discover/Offer/Request/Ack)
- Lease lifecycle and renewal (T1/T2)
- DHCP options and PXE boot
- DHCP relay agents and Option 82
- DHCPv6 and SLAAC interaction
- Rogue DHCP, DHCP snooping, starvation attacks

### [MTU, PMTUD & Fragmentation](mtu_pmtud.md)
- MTU on Ethernet, tunnels, and jumbo frames
- IPv4 fragmentation vs IPv6 (source-only)
- Path MTU Discovery and ICMP black holes
- MSS clamping for VPNs and tunnels
- Container / overlay network MTU pitfalls

### [Multicast (IGMP & PIM)](multicast_igmp_pim.md)
- Multicast group addressing (224.0.0.0/4, IPv6 ff00::/8)
- IGMP v2/v3 join/leave and L2 IGMP snooping
- PIM-SM / PIM-DM / SSM and rendezvous points
- Reverse Path Forwarding (RPF) loop prevention
- Use cases: IPTV, market data, mDNS, VXLAN BUM traffic

## Core Protocols

### [IP (Internet Protocol)](ip.md)
- Network-layer addressing and routing overview
- IPv4 vs IPv6 at a glance
- Packet structure and the role of IP in the stack
- Bridges into the version-specific notes below

### [IPv4 (Internet Protocol version 4)](ipv4.md)
- 32-bit addressing and packet format
- Address classes and private IP ranges
- Subnetting and CIDR notation
- Routing and fragmentation
- NAT (Network Address Translation)
- ICMP diagnostics and tools

### [IPv6 (Internet Protocol version 6)](ipv6.md)
- 128-bit addressing and packet format
- Address types (unicast, multicast, anycast)
- SLAAC and auto-configuration
- Neighbor Discovery Protocol (NDP)
- Extension headers
- ICMPv6 and transition mechanisms

### [TCP (Transmission Control Protocol)](tcp.md)
- Reliable, connection-oriented communication
- 3-way handshake
- Flow control and congestion control
- Sequence numbers and acknowledgments
- Connection termination

### [UDP (User Datagram Protocol)](udp.md)
- Fast, connectionless communication
- Low overhead (8-byte header)
- No reliability guarantees
- Use cases: DNS, streaming, gaming, VoIP
- Socket programming examples

### [HTTP/HTTPS](http.md)
- Web communication protocol
- Request methods (GET, POST, PUT, DELETE)
- Status codes
- Headers and caching
- Authentication and CORS
- REST API design

### [QUIC / HTTP/3](quic.md)
- UDP-based transport with built-in TLS 1.3
- Stream multiplexing without head-of-line blocking
- 0-RTT / 1-RTT connection establishment
- Connection migration across network changes
- HTTP/3 mapping and comparison to [HTTP/2](http2.md)

## Routing & Traffic Engineering

### [BGP & Anycast](bgp_anycast.md)
- Path-vector routing between autonomous systems
- eBGP vs iBGP, route attributes and policy
- Anycast for CDN/DNS load distribution and resilience
- Route propagation, hijacks, and RPKI

### [OSPF & IS-IS](ospf_isis.md)
- Link-state interior gateway protocols
- Areas, LSAs, and SPF (Dijkstra) computation
- Adjacency formation and flooding
- OSPF vs IS-IS design trade-offs

### [QoS & Traffic Shaping](qos_traffic_shaping.md)
- DiffServ/DSCP marking, ECN, Per-Hop Behaviors
- Policing vs shaping, token buckets
- Linux tc qdiscs (HTB, fq_codel, CAKE), bufferbloat
- Prioritizing voice/[RTP](rtp.md), wireless WMM

## Application Protocols

### [SSH (Secure Shell)](ssh.md)
- KEX, host keys, user authentication
- Public key auth and SSH agent
- Port forwarding (local, remote, dynamic SOCKS)
- ProxyJump, multiplexing, certificates
- sshd hardening

### [HTTP/2](http2.md)
- Binary framing and stream multiplexing
- HPACK header compression
- Flow control and prioritization
- ALPN negotiation, h2 vs h2c
- Server push (deprecated) and Early Hints
- vs HTTP/1.1 and HTTP/3 (see [QUIC](quic.md))

### [gRPC](grpc.md)
- Protobuf schemas and code generation
- Four call types (unary, server-streaming, client-streaming, bidirectional)
- Status codes, deadlines, metadata
- Interceptors and middleware
- mTLS and per-call auth
- Reflection, grpcurl, gRPC-Web

### [IoT Protocols (MQTT + CoAP)](iot_protocols.md)
- MQTT pub/sub, QoS levels, retained messages
- Last Will and Testament, keep-alive
- MQTT 5 features (shared subs, properties)
- CoAP REST-like over UDP
- Confirmable messages, Observe, block-wise transfer
- DTLS, CoAP-HTTP proxying

### [Email Protocols (SMTP / IMAP / POP3)](email_protocols.md)
- SMTP relay vs submission, envelope vs headers
- IMAP vs POP3 mailbox access models
- STARTTLS vs implicit TLS ports
- SPF, DKIM, DMARC, MTA-STS, DANE
- Anti-spoofing and spam defenses

### [Time Synchronization (NTP & PTP)](ntp_ptp.md)
- NTP stratum hierarchy, offset/delay calculation
- chrony configuration and operation
- PTP (IEEE 1588) for sub-microsecond accuracy
- Hardware timestamping, boundary/transparent clocks
- NTS security and amplification abuse

## Name Resolution

### [DNS (Domain Name System)](dns.md)
- Translates domain names to IP addresses
- DNS hierarchy and record types
- Query and response messages
- DNS caching and TTL
- DNSSEC security
- DNS over HTTPS (DoH) and DNS over TLS (DoT)
- Public DNS servers

### [mDNS (Multicast DNS)](mdns.md)
- Zero-configuration networking
- Local network name resolution (.local domain)
- Service discovery (DNS-SD)
- Avahi and Bonjour implementations
- Use cases: printers, file sharing, IoT devices

## NAT Traversal

### [STUN (Session Traversal Utilities for NAT)](stun.md)
- Discovers public IP address and port
- Detects NAT type
- Enables peer-to-peer connections
- Used in WebRTC and VoIP
- Message format and examples
- Public STUN servers

### [TURN (Traversal Using Relays around NAT)](turn.md)
- Relays traffic when direct connection fails
- Fallback for restrictive NATs and firewalls
- Bandwidth-intensive
- Used with ICE in WebRTC
- Server setup with coturn
- Cost considerations

### [ICE (Interactive Connectivity Establishment)](ice.md)
- Framework for establishing peer-to-peer connections
- Combines STUN and TURN for NAT traversal
- Candidate gathering and connectivity checks
- Priority-based path selection
- Handles symmetric NAT and firewalls
- Used by WebRTC and VoIP

### [PCP (Port Control Protocol)](pcp.md)
- Automatic port mapping and firewall control
- Successor to NAT-PMP with IPv6 support
- MAP and PEER opcodes for different use cases
- Works with multiple NATs in path
- Third-party mappings and explicit lifetimes
- Used by modern applications and IoT

### [NAT-PMP (NAT Port Mapping Protocol)](nat_pmp.md)
- Simple automatic port forwarding protocol
- Lightweight UDP-based (12-16 byte packets)
- IPv4 support with time-limited mappings
- Developed by Apple, widely deployed
- Gateway discovery and external IP detection
- Used by BitTorrent, VoIP, and gaming

## Real-Time Communication

### [WebSocket](websocket.md)
- Full-duplex bidirectional communication
- Low-latency persistent connections
- WebSocket handshake and frame format
- Client and server implementations
- Use cases: chat, live updates, gaming
- Authentication and security
- Heartbeat and reconnection strategies

### [WebRTC (Web Real-Time Communication)](webrtc.md)
- Browser-based peer-to-peer communication
- Video, audio, and data channels
- getUserMedia API and RTCPeerConnection
- Signaling and SDP offer/answer
- Media codecs and quality adaptation
- Security with mandatory encryption
- Simulcast and bandwidth management

### [RTP / RTCP / SRTP](rtp.md)
- Real-time media transport over UDP
- Sequence numbers, timestamps, jitter buffers
- RTCP sender/receiver reports and quality stats
- Payload types and codec carriage
- SRTP encryption and DTLS-SRTP keying

### [SIP & VoIP](sip_voip.md)
- SIP signaling (INVITE/ACK/BYE), registrars, proxies
- SDP offer/answer codec negotiation
- Relationship to [RTP](rtp.md) media and [ICE](ice.md) NAT traversal
- SIPS, SRTP, and DTLS-SRTP security
- Tooling: sipp, Wireshark VoIP analysis

## VPN & Overlay Networks

### [WireGuard](wireguard.md)
- Noise protocol handshake, cryptokey routing
- Static keypairs and AllowedIPs
- PersistentKeepalive for NAT traversal
- Roaming across networks
- Tailscale / Headscale / Netbird control planes
- Comparison to OpenVPN/IPsec

### [IPsec](ipsec.md)
- ESP vs AH, transport vs tunnel mode
- IKEv2 negotiation and Security Associations
- NAT-T (UDP 4500 encapsulation)
- strongSwan / Libreswan configuration
- Linux XFRM framework
- Site-to-site and road-warrior patterns

### [Container Networking](container_networking.md)
- Linux network namespaces, veth pairs, bridges
- Docker bridge mode and port publishing
- Kubernetes pod networking model
- CNI plugins (Flannel, Calico, Cilium)
- kube-proxy modes (iptables, IPVS, eBPF)
- NetworkPolicy and service routing

### [Overlay Networks (VXLAN, GRE, Geneve)](overlay_networks.md)
- L2-in-L3 encapsulation primitives
- VXLAN with multicast / EVPN / unicast control planes
- GRE for simple point-to-point tunnels
- Geneve TLV extensibility (modern replacement)
- MTU and encryption considerations

## Network Discovery

### [UPnP (Universal Plug and Play)](upnp.md)
- Automatic device discovery
- Zero-configuration setup
- SSDP (Simple Service Discovery Protocol)
- Port forwarding (IGD)
- Security considerations
- Common device types

## Security

### [Firewalls](firewalls.md)
- Packet filtering
- Stateful inspection
- Application layer firewalls
- Next-generation firewalls (NGFW)
- iptables, ufw, firewalld configurations
- NAT and port forwarding
- Firewall architectures (DMZ, screened subnet)
- Security best practices

### [TLS/SSL & PKI](tls_ssl.md)
- TLS 1.3 handshake (1-RTT/0-RTT) vs TLS 1.2
- X.509 certificates, chains, CA trust, OCSP stapling
- SNI, ALPN, ECH extensions
- Mutual TLS (mTLS) and forward secrecy
- ACME / Let's Encrypt, openssl inspection

### [802.1X / RADIUS / NAC](nac_8021x.md)
- Port-based network access control
- Supplicant / authenticator / RADIUS server model
- EAP methods (EAP-TLS, PEAP, EAP-TTLS), EAPOL
- RADIUS AAA and dynamic VLAN assignment
- MAC Authentication Bypass and posture/NAC

## Quick Reference

### Protocol Port Numbers

| Protocol | Port | Transport | Purpose |
|----------|------|-----------|---------|
| **HTTP** | 80 | TCP | Web pages |
| **HTTPS** | 443 | TCP | Secure web |
| **HTTP/3 (QUIC)** | 443 | UDP | Secure web over QUIC |
| **SSH** | 22 | TCP | Secure shell |
| **FTP** | 20/21 | TCP | File transfer |
| **DNS** | 53 | UDP/TCP | Name resolution |
| **DHCP** | 67/68 | UDP | IP configuration |
| **SMTP** | 25/587 | TCP | Email relay / submission |
| **SMTPS** | 465 | TCP | SMTP over TLS |
| **POP3 / POP3S** | 110 / 995 | TCP | Email retrieval |
| **IMAP / IMAPS** | 143 / 993 | TCP | Email access |
| **NTP** | 123 | UDP | Time synchronization |
| **PTP** | 319/320 | UDP | Precision time |
| **SIP / SIPS** | 5060 / 5061 | UDP/TCP | VoIP signaling |
| **RADIUS** | 1812/1813 | UDP | AAA auth / accounting |
| **STUN** | 3478 | UDP | NAT discovery |
| **SSDP** | 1900 | UDP | UPnP discovery |
| **mDNS** | 5353 | UDP | Local DNS |

### Common Network Tools

```bash
# Connectivity Testing
ping <host>                    # Test reachability
traceroute <host>              # Trace route to host

# DNS Lookup
dig <domain>                   # DNS query
nslookup <domain>              # DNS lookup
host <domain>                  # Simple DNS lookup

# Network Configuration
ifconfig                       # Network interface config (legacy)
ip addr show                   # Show IP addresses
ip route show                  # Show routing table

# Port Scanning
netstat -tuln                  # Show listening ports
ss -tuln                       # Socket statistics
nc -zv <host> <port>           # Check if port is open

# Packet Capture
tcpdump -i any                 # Capture all traffic
tcpdump port 80                # Capture HTTP traffic
wireshark                      # GUI packet analyzer

# Service Discovery
avahi-browse -a                # Browse mDNS services
upnpc -l                       # List UPnP devices
```

### Private IP Address Ranges

```
10.0.0.0        - 10.255.255.255     (10/8 prefix)
172.16.0.0      - 172.31.255.255     (172.16/12 prefix)
192.168.0.0     - 192.168.255.255    (192.168/16 prefix)
```

### Common Subnet Masks

| CIDR | Netmask | Hosts | Typical Use |
|------|---------|-------|-------------|
| /8 | 255.0.0.0 | 16,777,214 | Very large networks |
| /16 | 255.255.0.0 | 65,534 | Large networks |
| /24 | 255.255.255.0 | 254 | Small networks |
| /30 | 255.255.255.252 | 2 | Point-to-point links |

## Protocol Relationships

```
Application Layer:
  HTTP, FTP, SMTP, DNS, DHCP, SSH
       |
       v
Transport Layer:
  TCP (reliable) or UDP (fast)
       |
       v
Network Layer:
  IP (routing and addressing)
       |
       v
Data Link Layer:
  Ethernet, WiFi (MAC addresses)
       |
       v
Physical Layer:
  Cables, signals, physical media
```

## Troubleshooting Flow

```
1. Physical Layer
   - Cable connected?
   - Link lights on?
   -> Use: Visual inspection, ethtool

2. Data Link Layer
   - MAC address correct?
   - Switch working?
   -> Use: arp -a, show mac address-table

3. Network Layer
   - IP address assigned?
   - Can ping gateway?
   - Routing correct?
   -> Use: ip addr, ping, traceroute

4. Transport Layer
   - Port open?
   - Firewall blocking?
   - Service running?
   -> Use: netstat, telnet, nc

5. Application Layer
   - Service configured correctly?
   - Authentication working?
   - Application logs?
   -> Use: curl, application-specific tools
```

## Security Best Practices

### Network Segmentation
- Separate networks by function (guest, IoT, corporate)
- Use VLANs for logical separation
- Firewall rules between segments

### Access Control
- Implement firewall rules (default deny)
- Use strong authentication
- Enable logging and monitoring
- Regular security audits

### Encryption
- Use HTTPS instead of HTTP
- Enable DNS over HTTPS/TLS
- Use VPN for remote access
- Encrypt sensitive traffic

### Updates and Patches
- Keep firmware updated
- Patch vulnerabilities promptly
- Disable unused services
- Remove default credentials

## Common Scenarios

### Home Network Setup
1. Router assigns private IPs (192.168.1.x)
2. DHCP provides automatic configuration
3. NAT translates private to public IP
4. DNS resolves domain names (8.8.8.8)
5. Devices use mDNS for local discovery

### WebRTC Video Call
1. STUN discovers public IP addresses
2. ICE gathers connection candidates
3. Signaling server exchanges candidates
4. Direct P2P connection attempted
5. TURN relay used if P2P fails

### Smart Home Devices
1. Devices announce via mDNS (device.local)
2. UPnP enables automatic port forwarding
3. Devices discover each other (SSDP)
4. Control via local network
5. Cloud connection for remote access

## Further Learning

### Online Resources
- [Wireshark Tutorial](https://www.wireshark.org/docs/)
- [TCP/IP Guide](http://www.tcpipguide.com/)
- [Network+ Certification](https://www.comptia.org/certifications/network)
- [RFC Editor](https://www.rfc-editor.org/)

### Books
- *TCP/IP Illustrated* by W. Richard Stevens
- *Computer Networks* by Andrew Tanenbaum
- *Network Warrior* by Gary Donahue

### Practice
- Set up home lab with VirtualBox/VMware
- Use Packet Tracer for simulations
- Capture and analyze traffic with Wireshark
- Configure firewall rules
- Set up services (DNS, DHCP, web server)
