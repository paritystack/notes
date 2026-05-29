# Overlay Networks (VXLAN, GRE, Geneve)

## Overview

An **overlay network** is a virtual network built on top of an existing (underlay) network. Packets from the overlay are wrapped in extra headers (encapsulated) and sent across the underlay, where intermediate routers just see "normal" packets. At the destination, the wrapper is stripped and the inner packet is delivered as if the two endpoints were on the same LAN.

Overlays let you build virtual L2 or L3 networks that span physical boundaries — across racks, data centers, clouds — without renumbering or coordinating with the underlying network.

```
Tenant view:      VM1 ─────────────── VM2     (same "L2")
                    \                 /
                     \   overlay     /
Underlay reality:    Host A ────── Host B     (multi-hop IP routed)
                  encapsulates    decapsulates
```

This note covers the three encapsulations you'll meet most:

- **VXLAN** — UDP-based, L2-over-L3, the data center default
- **GRE** — generic, lightweight, predates VXLAN
- **Geneve** — modern successor to VXLAN with extensible TLV headers

## Why Overlays?

```
Problem:
  - Cloud / DC needs millions of tenant networks
  - Underlying physical network is one big L3 fabric
  - Can't give every tenant their own VLAN (4094 VLAN limit)
  - Can't expect underlay to know per-tenant routing
  - Moving VMs across racks needs L2 mobility

Solution: overlay
  - Tunnels between hosts carry tenant traffic
  - Underlay just routes IP between hosts (boring, scalable)
  - Tenants get isolated L2/L3 networks
  - Identifier (VNI, key) selects which tenant the packet belongs to
```

## GRE (Generic Routing Encapsulation)

The oldest of the bunch. RFC 2784. IP protocol number **47**. Encapsulates any L3 protocol inside IP.

### Frame Format

```
[ Outer IP | GRE | Inner packet ]

GRE header (4-byte minimum):
+--------+--------+--------+--------+
|C|R|K|S|0  Resv  | Version |  Proto |
+--------+--------+--------+--------+
[ Checksum     (if C=1) ]
[ Key          (if K=1) ]
[ Sequence     (if S=1) ]

Proto field: 0x6558 for transparent Ethernet bridging (TEB), 0x0800 for IPv4 inner, etc.
```

### What GRE Provides (and Doesn't)

```
✓ Tunnels any L3 protocol (IP, IPX, etc.)
✓ Minimal overhead (24 bytes total with outer IPv4)
✓ Can carry L2 frames with GRETAP variant
✓ Optional sequencing, checksums, key (32-bit "session id")
✗ No encryption (combine with IPsec for that)
✗ No multi-tenancy (one tunnel per peer pair, unless you use Key)
✗ Stateless, no failover signaling
```

### Linux GRE Tunnel

```bash
# IP-in-IP-over-GRE tunnel between two hosts
sudo ip tunnel add gre1 mode gre \
  remote 198.51.100.2 local 198.51.100.1 ttl 255
sudo ip addr add 10.99.0.1/30 dev gre1
sudo ip link set gre1 up

# Now 10.99.0.1 can reach 10.99.0.2 on the other side
ping 10.99.0.2

# L2 (Ethernet-in-GRE) variant
sudo ip link add gretap1 type gretap remote 198.51.100.2 local 198.51.100.1
```

### Typical Uses

```
- Linking two IPv4 sites with simple point-to-point tunnel
- Multicast transport (IP multicast over IP unicast)
- DMVPN hub-and-spoke (mGRE)
- Pod-to-pod connectivity in Calico (IP-in-IP mode)
- Carrying IPv6 over IPv4 (6in4 is the related but distinct mechanism)
```

GRE is largely supplanted by VXLAN/Geneve for new designs, but it's still everywhere — especially in legacy enterprise WAN.

## VXLAN (Virtual eXtensible LAN)

RFC 7348. Encapsulates **Ethernet frames inside UDP**. The dominant overlay in modern data centers and Kubernetes CNIs (Flannel, Cilium).

### Frame Format

```
[ Outer Eth | Outer IP | UDP | VXLAN | Inner Eth | Inner IP | Inner Payload ]

VXLAN header (8 bytes):
+--------+--------+--------+--------+
|R|R|R|R|I|R|R|R|       Reserved   |
+--------+--------+--------+--------+
|             VNI (24 bits)         |
+--------+--------+--------+--------+
|       Reserved (8 bits)            |
+--------+--------+--------+--------+

VNI (VXLAN Network Identifier): 24 bits → 16M unique tenant networks
UDP destination port: 4789 (IANA standard)
```

Total overhead: **50 bytes** (outer IPv4 + UDP + VXLAN) → tunnel MTU 1450 on a 1500 underlay.

### How VXLAN Works

```
1. Host A wants to send a frame to Host B in the same VNI
2. Source VTEP (VXLAN Tunnel Endpoint) at hypervisor/node:
   - Wraps inner Ethernet frame in VXLAN + UDP + outer IP
   - Outer source IP = local VTEP, outer dst IP = remote VTEP
3. Underlay routes the UDP packet normally (boring IP)
4. Destination VTEP:
   - Strips outer headers
   - Looks up VNI to find tenant context
   - Delivers inner frame to local VMs/containers
```

### MAC Learning Modes

The control plane question: how does VTEP A know that MAC X is reachable via VTEP B?

```
1. Multicast / flood-and-learn (RFC 7348 original)
   - All VTEPs in a VNI subscribe to a multicast group
   - Unknown unicast / broadcast / ARP → flood via multicast
   - VTEPs learn from inner src MAC + outer src IP
   - Simple, but needs multicast in underlay (rare in public cloud)

2. Static unicast peer list
   - Each VTEP configured with peer list
   - Floods to all peers via unicast UDP
   - Easy but doesn't scale beyond ~hundreds

3. EVPN (Ethernet VPN, RFC 7432, with BGP signaling)
   - BGP distributes MAC/IP-to-VTEP mappings
   - No flooding required
   - Industry-standard data center fabric (NVIDIA Cumulus, Arista, Cisco ACI)

4. Cloud-native (e.g., Cilium, Kubernetes CNI)
   - Control plane reads pod IP assignments
   - Builds FDB entries via netlink
```

### Linux VXLAN

```bash
# Multicast-based VXLAN
sudo ip link add vxlan10 type vxlan \
  id 10 group 239.1.1.1 dev eth0 dstport 4789

# Unicast (point-to-point)
sudo ip link add vxlan10 type vxlan \
  id 10 remote 198.51.100.5 local 198.51.100.1 dstport 4789

# Add to bridge to connect local VMs/containers
sudo ip link add br10 type bridge
sudo ip link set vxlan10 master br10
sudo ip addr add 10.10.0.1/24 dev br10
sudo ip link set br10 up
sudo ip link set vxlan10 up

# Inspect
ip -d link show vxlan10
bridge fdb show dev vxlan10

# Manually add remote MAC/VTEP mapping
sudo bridge fdb append 00:00:00:00:00:00 dev vxlan10 dst 198.51.100.5
sudo bridge fdb append 11:22:33:44:55:66 dev vxlan10 dst 198.51.100.5
```

### Capturing VXLAN

```bash
# Outer (UDP 4789) - normal
sudo tcpdump -i eth0 udp port 4789

# Decapsulated - feed Wireshark which natively decodes VXLAN

# Force tcpdump to decode
sudo tcpdump -i eth0 -nn 'udp port 4789' -vv
```

## Geneve (Generic Network Virtualization Encapsulation)

RFC 8926. The "VXLAN done right" — same wire shape but **extensible**. Used by OVN, Cilium (recent), VMware NSX-T, Open vSwitch overlays.

### Frame Format

```
[ Outer Eth | Outer IP | UDP | Geneve | Inner Eth | Inner Payload ]

Geneve header (8 bytes fixed + variable options):
+--------+--------+--------+--------+
|Ver|Opt Len|O|C|    Rsvd  | Proto  |
+--------+--------+--------+--------+
|         VNI (24 bits)              |  Reserved (8) |
+--------+--------+--------+--------+
|         Variable TLV options       |
+------------------------------------+

UDP destination port: 6081
```

### Why Geneve

```
✓ VNI (same 24-bit space as VXLAN)
✓ Extensible TLV (Type-Length-Value) options:
    - encryption metadata
    - security tags
    - tenant context beyond just VNI
    - load-balancing hints
✓ Vendors can add their own option types without breaking interop
✓ AWS Gateway Load Balancer uses Geneve to pass traffic through firewalls
```

VXLAN headers can't grow; Geneve is built to. Modern overlays default to Geneve.

### Linux Geneve

```bash
sudo ip link add geneve1 type geneve id 100 remote 198.51.100.5
sudo ip addr add 10.20.0.1/24 dev geneve1
sudo ip link set geneve1 up
```

## VXLAN/Geneve vs GRE: When to Use What

| Feature | GRE | VXLAN | Geneve |
|---------|-----|-------|--------|
| **Wraps** | L3 (or L2 with GRETAP) | L2 (Ethernet) | L2 (Ethernet) |
| **Outer** | IP proto 47 | UDP 4789 | UDP 6081 |
| **Overhead** | 24B | 50B | 50B+ (with options) |
| **Tenant ID** | 32-bit Key (optional) | 24-bit VNI | 24-bit VNI + options |
| **Multipath (ECMP)** | poor (no UDP src port) | good (varies UDP src port per flow) | good |
| **Extensible** | rigid | rigid | TLV options |
| **Multicast underlay** | requires for BUM | supported | supported |
| **Modern DC overlay** | rarely | yes (legacy) | yes (new) |

The "ECMP" point matters: routers hash on 5-tuple (src/dst IP, src/dst port, protocol) to spread traffic across equal-cost paths. GRE has no ports → all GRE between two hosts hashes to one link. VXLAN/Geneve use UDP and vary source port per inner flow → traffic spreads.

## Control Plane Options

The encapsulation is only half the story — something has to tell each tunnel endpoint "MAC X / IP Y lives at VTEP Z."

### Multicast (flood-and-learn)

```
Pros: simple, no extra protocol
Cons: requires multicast in underlay (uncommon outside enterprise switches)
```

### Static / unicast head-end replication

```
Pros: works on any underlay
Cons: scales to hundreds, not thousands
```

### BGP EVPN

```
The data center standard (NVIDIA Cumulus, Arista EOS, Cisco NX-OS)

EVPN address families distribute:
  - Route Type 2: MAC/IP advertisements
  - Route Type 3: multicast tunnel endpoint
  - Route Type 5: IP prefix routes (for L3 tenant routing)

Used with VXLAN or Geneve. Eliminates flooding.
```

### Software-Defined (cloud / Kubernetes)

```
Cilium / Calico / Flannel / OVN:
  - Watch pod/VM creation events
  - Push FDB entries to each node via API/netlink
  - No multicast, no BGP, no manual config
```

## Combining With Encryption

VXLAN/Geneve/GRE are **plaintext**. Adversary on the underlay can read everything. For untrusted underlays, combine with:

```
IPsec transport mode encrypting outer UDP
  IPsec ESP wraps the entire VXLAN packet
  Standard in many cloud overlays (AWS Transit Gateway encryption, etc.)

WireGuard tunnel as underlay
  Run VXLAN inside a WireGuard mesh
  Common in self-hosted multi-cloud setups (Tailscale subnet relays, Netmaker)

Cilium WireGuard / IPsec mode
  Encrypts node-to-node traffic transparently
  Setting in CNI config
```

## MTU and Overhead

Crucial: every encapsulation steals bytes from the inner MTU.

```
Underlay MTU: 1500
  − Outer Eth (14)
  − Outer IPv4 (20)
  − UDP (8)
  − VXLAN (8)
  = 1450 inner MTU

For IPv6 outer: 1430
For Geneve with options: 1450 minus option bytes
For GRE: 1476
For VXLAN over IPsec: ~1380 (varies)
```

If you don't lower the pod/VM MTU to match, you get the classic "small packets work, large transfers hang" symptom. See [mtu_pmtud.md](mtu_pmtud.md).

```bash
# Set MTU on VXLAN interface
sudo ip link set vxlan10 mtu 1450
```

### Jumbo Frames Cure-All

Set underlay MTU to 9000 (jumbo) → inner MTU 8950 → never have to worry about overhead. Most data center fabrics do this. Public cloud generally caps at 9001 (AWS), 8500 (others).

## Performance

```
VXLAN/Geneve in software (Linux kernel):
  ~5-10 Gbps single-flow, line rate with multiple flows

VXLAN with NIC offload (hardware VTEP):
  Line rate at 25/100G
  Mellanox/NVIDIA, Intel E810, Broadcom support

GRE: similar story
GRE with checksum: slower (per-packet calculation in software)
```

Checks to enable:
```bash
sudo ethtool -k eth0 | grep tx-udp_tnl
sudo ethtool -K eth0 tx-udp_tnl-segmentation on
```

## Wireshark / tcpdump Decoding

```
Filter outer encapsulation:
  udp.port == 4789   # VXLAN
  udp.port == 6081   # Geneve
  ip.proto == 47     # GRE

Wireshark automatically decodes inner packet if the dst port matches a known overlay.
Custom ports: Edit → Preferences → Protocols → VXLAN → UDP port

Inspect VNI:
  vxlan.vni == 100
  geneve.vni == 100
  gre.key == 100
```

## Use Case Examples

### Data Center Fabric (VXLAN + EVPN)

```
Spine-and-leaf fabric:
  - Leaves are VTEPs
  - Spines route IP only (boring underlay)
  - BGP-EVPN distributes MAC/IP info
  - L2 stretch and L3 routing across the whole DC
  - VLAN-like semantics, scales to millions
```

### Kubernetes CNI

```
Flannel VXLAN:
  - VTEP on each node (flannel.1)
  - VNI 1 = entire cluster (single tenant)
  - Pod IPs from PodCIDR per node
  - Direct kernel netlink, no flooding

Cilium VXLAN/Geneve:
  - Same idea, eBPF-accelerated
  - Multi-tenant via NetworkPolicy
```

### Multi-Site Connectivity

```
Two on-prem sites + AWS VPC:
  - WireGuard mesh between gateways (encryption + traversal)
  - VXLAN tunnels inside WireGuard (per-tenant L2 stretch)
  - Or Geneve with NSX-T / Aviatrix / Megaport overlays
```

### AWS Gateway Load Balancer (Geneve)

```
GWLB injects Geneve-encapsulated traffic into a fleet of inspection
appliances (firewalls, IDS), which decap, inspect, re-encap, send back.
Transparent insertion of L3 firewalls into VPC traffic.
```

## Common Issues

### Tunnel up, no traffic

```
✓ Check FDB / control plane has MAC/IP entries
✓ Check firewall on underlay isn't dropping UDP 4789/6081 or IP proto 47
✓ Check VTEP IPs match (no NAT in between unless explicitly handled)
✓ Tag mismatch: VNI must match on both ends
```

### "Works for small packets, breaks for big"

MTU. Set inner MTU = underlay MTU − overhead, or enable PMTUD on the underlay.

### Flooding excessive

```
Multicast mode: too many BUM frames → multicast group too noisy
  → switch to head-end replication or EVPN

Symptom: high traffic between idle VMs
Diagnosis: tcpdump on underlay, see flood frames
```

### Asymmetric routing

In EVPN setups, ingress traffic comes one path, return goes another → can break stateful firewalls. Use **distributed gateway** (gateway IP active on every leaf) or fix flow symmetry with policy.

## Quick Reference

| Need | Use |
|------|-----|
| Simple point-to-point L3 tunnel | GRE |
| Multi-tenant L2 stretch in DC | VXLAN |
| Modern DC with extensible options | Geneve |
| Container/pod networking | VXLAN or Geneve via CNI |
| Encryption | IPsec / WireGuard, optionally wrapping overlay |
| Cross-cloud overlay | WireGuard + VXLAN, or commercial SD-WAN |
| Multicast over unicast | GRE or VXLAN multicast |

## ELI10

Imagine you want to mail a letter to your friend in another city, but the local post office only delivers within town. So you stuff your letter into a bigger envelope addressed to your friend's town post office, and write **"please re-deliver to:"** with the original address on a sticky note inside. The town post office opens the outer envelope, reads the sticky note, and delivers your letter locally on that side.

That outer envelope is **encapsulation**.

- **GRE** = a plain unmarked manila folder.
- **VXLAN** = a manila folder with a numbered tag (VNI) so the receiving post office knows which company department the letter belongs to.
- **Geneve** = a manila folder with a tag *and* slots for extra sticky notes (TLV options) that future post offices might want to read.

The neat part: the postal trucks in between don't care what's inside — they just route the outer envelope. So you can build private "departments" that span any number of cities (overlay networks across data centers), and the postal system (underlay) doesn't have to know about them.

## Further Resources

- [RFC 7348 - VXLAN](https://tools.ietf.org/html/rfc7348)
- [RFC 8926 - Geneve](https://tools.ietf.org/html/rfc8926)
- [RFC 2784 - GRE](https://tools.ietf.org/html/rfc2784)
- [RFC 7432 - BGP MPLS-Based Ethernet VPN (EVPN)](https://tools.ietf.org/html/rfc7432)
- [NVIDIA Cumulus Linux EVPN docs](https://docs.nvidia.com/networking-ethernet-software/cumulus-linux/Network-Virtualization/Ethernet-Virtual-Private-Network-EVPN/)
- [Cilium overlay routing docs](https://docs.cilium.io/en/stable/network/concepts/routing/)
- [OVN architecture (Geneve user)](https://www.ovn.org/support/dist-docs/ovn-architecture.7.html)
