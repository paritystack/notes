# MTU, PMTUD & Fragmentation

## Overview

**MTU** (Maximum Transmission Unit) is the largest packet size a link can carry in a single frame. **PMTUD** (Path MTU Discovery) is how endpoints figure out the smallest MTU along a path. **Fragmentation** is what happens when a packet is too big — [IPv4](ipv4.md) routers can split it; [IPv6](ipv6.md) routers can't. Get any of this wrong and [TCP](tcp.md) connections silently hang. [Ethernet](ethernet_vlan.md) sets the standard MTU of 1500 bytes; [QUIC](quic.md) and [WebRTC](webrtc.md) must also respect PMTUD.

## What is MTU?

```
Frame on Ethernet:
+----+----+----+----+----+----+----+----+
|  L2 header (14B)  |  Payload  | FCS  |
+----+----+----+----+----+----+----+----+
                    └────┬──────┘
                         │
                  MTU = max payload
                  Ethernet default: 1500 bytes
```

The MTU is the **L3 payload** size — the largest IP packet that fits in one L2 frame. It excludes the L2 header and frame check sequence.

### Common MTUs

| Medium | MTU | Notes |
|--------|-----|-------|
| Ethernet (standard) | 1500 | The universal default |
| Ethernet (jumbo) | 9000 | Datacenter / storage |
| IEEE 802.11 Wi-Fi | 2304 | Usually capped to 1500 in practice |
| PPPoE (DSL) | 1492 | Eats 8B for PPPoE header |
| GRE tunnel | 1476 | 1500 − 24B GRE/IP header |
| IPsec ESP tunnel | ~1438 | Varies with cipher/auth |
| WireGuard | 1420 | 80B overhead vs 1500 outer |
| VXLAN | 1450 | 50B overhead vs 1500 outer |
| Loopback (Linux) | 65536 | No real link |
| Minimum IPv4 | 68 | Required by RFC 791 |
| Minimum IPv6 | 1280 | Required by RFC 8200 |

### Where Bytes Go on Ethernet

```
Wire-level Ethernet:
  7  preamble
  1  SFD
  6  dst MAC
  6  src MAC
  2  EtherType
[ 4  optional 802.1Q VLAN tag ]
1500 payload (= MTU)
  4  FCS
 12  interframe gap
─────
~38 + 1500 = 1538 wire bytes per full-MTU frame
```

## Fragmentation

When a packet exceeds the next-hop link's MTU, something has to give.

### IPv4 Fragmentation

```
Original packet (3000 bytes) hits link with MTU 1500

Router splits into 3 fragments:
  Fragment 1: IP header + 1480 bytes (offset 0,    MF=1)
  Fragment 2: IP header + 1480 bytes (offset 185,  MF=1)
  Fragment 3: IP header + 40 bytes   (offset 370,  MF=0)

  (Offsets are in 8-byte units: 1480/8 = 185)

Destination reassembles using:
  - IP identification field (groups fragments)
  - Fragment offset
  - MF (More Fragments) flag

Header fields used:
  Identification (16 bits): per-packet ID
  Flags (3 bits):
    bit 0: reserved
    bit 1: DF (Don't Fragment)
    bit 2: MF (More Fragments)
  Fragment Offset (13 bits): position in 8-byte units
```

### IPv6: No Router Fragmentation

In IPv6, only the **source** can fragment — routers must drop oversized packets and return ICMPv6 "Packet Too Big". This forces PMTUD to actually work, but breaks if ICMPv6 is filtered.

### Why Fragmentation is Bad

```
1. Performance hit
   - Extra packets, extra headers
   - Reassembly buffers consume memory at receiver

2. Reliability
   - Lose ONE fragment → entire datagram dropped
   - Effective loss rate goes up

3. Security
   - Fragmentation attacks: overlapping/tiny fragments
     bypass IDS/firewall reassembly
   - Stateless firewalls can't inspect L4 of fragments 2+

4. NAT and load balancers struggle
   - Only first fragment has L4 ports
   - Need flow tracking to handle the rest
```

The modern answer is **don't fragment** — use PMTUD instead.

## PMTUD (Path MTU Discovery, RFC 1191)

The sender sets the **DF (Don't Fragment)** bit on every IPv4 packet. Any router whose outgoing link is too small drops the packet and returns ICMP "Fragmentation Needed" (type 3, code 4) including the smaller MTU.

```
Sender (MTU 1500)
   │
   ▼  packet size 1500, DF=1
Router A (link to B has MTU 1400)
   │
   ▼  drops packet
   │  sends ICMP "Fragmentation needed, next-hop MTU = 1400"
   ▼
Sender
   │  records 1400 for this destination
   ▼  retransmits with size ≤ 1400
   │
   ▼
... continues until packets fit end to end
```

### TCP and PMTUD

TCP integrates with PMTUD via its **MSS** (Maximum Segment Size):

```
MSS = MTU − 20 (IP header) − 20 (TCP header) = 1460 for Ethernet

Announced in SYN:
  TCP Options: MSS = 1460

Each side uses min(my MSS, their MSS) for sending.

When ICMP Frag-Needed arrives, sender lowers MSS for that flow.
```

### IPv6 PMTUD

IPv6 doesn't have DF (it's implied — routers never fragment). PMTUD uses ICMPv6 type 2 ("Packet Too Big"). Same principle: sender shrinks until packets fit.

## ICMP Black Holes

PMTUD breaks when intermediate firewalls drop ICMP messages. The sender retransmits the same too-big packet, gets no response, and the connection hangs forever.

### Symptoms

```
- TCP handshake (small packets) completes fine
- HTTP request (small) sent fine
- HTTP response (large) never arrives
- Connection times out
- `curl` hangs after `> GET / HTTP/1.1`
- `ping` works
- `ping -s 1472` works
- `ping -s 1473 -M do` fails silently (no reply)
```

### Why So Common

Lazy firewall configs block all ICMP "for security." This kills PMTUD. Symptoms appear only with specific destinations behind smaller-MTU tunnels (VPN, GRE, PPPoE).

### Mitigation: PMTUD Black Hole Detection

Modern TCP stacks detect repeated retransmits and progressively shrink MSS, even without ICMP feedback. Linux: `net.ipv4.tcp_mtu_probing`.

```bash
# Enable PMTUD black-hole probing
sudo sysctl -w net.ipv4.tcp_mtu_probing=1
# 0 = disabled
# 1 = enabled when ICMP black hole suspected (default on most distros now)
# 2 = always enabled
```

### Mitigation: MSS Clamping

The router at the MTU boundary rewrites TCP MSS in SYN packets to fit its own MTU. Eliminates the need for PMTUD on TCP.

```bash
# Linux iptables: clamp MSS on a tunnel interface
sudo iptables -t mangle -A FORWARD -o wg0 -p tcp --tcp-flags SYN,RST SYN \
  -j TCPMSS --clamp-mss-to-pmtu

# Or set explicitly
sudo iptables -t mangle -A FORWARD -o wg0 -p tcp --tcp-flags SYN,RST SYN \
  -j TCPMSS --set-mss 1340
```

Standard practice on VPN gateways, GRE tunnels, PPPoE routers.

## Jumbo Frames

Frames up to 9000 bytes. Less header overhead per byte, fewer interrupts, higher throughput on fast networks.

```
1500 MTU: ~94% efficient (1500 / 1538 wire bytes)
9000 MTU: ~99% efficient (9000 / 9038 wire bytes)

Real wins:
  - iSCSI / NFS / database replication
  - 10G+ Ethernet in datacenters
  - GPU clusters

Cautions:
  - All hops must support it (one 1500 hop → fragmentation/drops)
  - Not routable across the internet (always 1500)
  - Switches need configured jumbo support
```

```bash
# Set MTU on Linux
sudo ip link set eth0 mtu 9000
ip link show eth0

# Persistent: /etc/netplan/, /etc/network/interfaces, etc.
```

## Tunneling Overhead

Every encapsulation layer eats MTU. This is the #1 source of mystery MTU bugs.

```
Stack: TCP over IPv4 over IPsec over GRE over Ethernet

Outer Ethernet: 1500 MTU
GRE header:       -24 → 1476
IPsec ESP+IV+auth: -38 → 1438 (varies)
Inner IPv4 hdr:    -20 → 1418
TCP header:        -20 → 1398
TCP payload max:        1398 bytes

If anything in the path assumes 1460 MSS → fragmentation or black-holed PMTUD
```

### VPN MTU Guidance

```
WireGuard:    set inner MTU 1420 (it auto-adjusts in recent versions)
OpenVPN UDP:  ~1450
OpenVPN TCP:  ~1432
IPsec tunnel: 1400 is safe default
GRE only:     1476
PPPoE:        1492 (DSL gotcha)
```

When in doubt: set MTU to 1400 on the tunnel interface and use MSS clamping.

## Measuring MTU

### ping with DF

```bash
# Linux: -M do sets DF, -s sets payload (28B overhead: 20 IP + 8 ICMP)
ping -M do -s 1472 8.8.8.8       # 1472 + 28 = 1500
ping -M do -s 1473 8.8.8.8       # should fail if path MTU is 1500

# Binary search downwards until success → MTU = payload + 28

# macOS / BSD
ping -D -s 1472 8.8.8.8

# Windows
ping -f -l 1472 8.8.8.8
```

### tracepath

Built-in PMTUD discovery, no root needed:

```bash
tracepath 8.8.8.8

# Output:
# 1?: [LOCALHOST]   pmtu 1500
# 1:  192.168.1.1   2.345ms
# 2:  10.0.0.1      8.123ms
# 2:  10.0.0.1      pmtu 1400
# 3:  8.8.8.8       12.456ms reached
```

The `pmtu 1400` line is where MTU dropped along the path.

### Linux PMTUD Cache

```bash
# Per-destination cache
ip route get 8.8.8.8

# Output includes MTU:
# 8.8.8.8 via 192.168.1.1 dev eth0 src 192.168.1.10 mtu 1400

# Flush PMTU cache
sudo ip route flush cache
```

## Reading Wireshark for MTU Issues

```
1. Filter: icmp.type == 3 and icmp.code == 4
   → All "Fragmentation Needed" messages

2. Filter: ip.flags.df == 1 and ip.len > 1500
   → Anyone trying to send big DF packets

3. Filter: ip.flags.mf == 1 or ip.frag_offset > 0
   → Actual fragments

4. Filter: tcp.options.mss_val
   → Inspect MSS in SYN
```

## Linux MTU sysctls

```bash
# Per-route MTU cache lifetime
net.ipv4.route.mtu_expires = 600

# Minimum MTU after PMTUD shrinkage
net.ipv4.route.min_pmtu = 552

# Disable PMTUD entirely (don't do this in production)
net.ipv4.ip_no_pmtu_disc = 1

# Black hole probing (above)
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_base_mss = 1024
```

## Best Practices

### When designing networks

```
✓ Keep MTU consistent end-to-end where you can
✓ Document MTU on every tunnel, including overhead breakdown
✓ Allow ICMP type 3 code 4 (and ICMPv6 type 2) through firewalls
✓ Use MSS clamping on tunnel ingress
✓ Test with `ping -M do` at full MTU after any tunnel change
```

### When debugging weird hangs

```
1. Does small ping work? → connectivity OK
2. Does `ping -M do -s 1472` work? → path MTU is 1500
3. Does `curl https://small-page` work? → small TCP OK
4. Does `curl https://large-page` hang? → MTU/PMTUD issue
5. tracepath to find the bottleneck hop
6. Check tunnels in path, fix MTU or add MSS clamping
```

## Edge Cases & Gotchas

### Hyper-V / EC2 / Cloud NIC

Some virtualization adds invisible headers; the guest sees 1500 but the underlay is smaller. Use `ip route get` and `tracepath` from inside the VM.

### Docker / Kubernetes

Default container MTU often differs from host. Overlay networks (VXLAN, IPIP, WireGuard CNIs) need MTU configuration on `cni0`, `flannel.1`, `cilium_vxlan`, etc.

```
Host MTU: 1500
VXLAN overhead: 50
Pod MTU should be: 1450
```

Misconfigured pod MTU = mysterious slow/hung connections, especially for large responses.

### IPv6 Minimum 1280

Any IPv6-capable link must support at least 1280-byte MTU. Tunneled IPv6 over IPv4 (e.g., 6in4) often needs explicit MTU 1280 on the tunnel.

### TCP Segmentation Offload (TSO/GSO/LRO)

The NIC handles TCP segmentation in hardware. tcpdump may show packets larger than MTU — they're not really on the wire that way, they get split by the NIC. Disable for debugging:

```bash
sudo ethtool -K eth0 tso off gso off gro off lro off
```

### IP Fragments and NAT

Many NATs drop non-first fragments because they don't have L4 ports to translate. Sources of weird "works at home, fails on hotspot" bugs.

## Quick Reference

| Scenario | Fix |
|----------|-----|
| TCP hangs after SYN+ACK on a tunnel | MSS clamping on tunnel |
| `ping` works, `curl` hangs | ICMP black hole — enable tcp_mtu_probing |
| Datacenter throughput poor | Jumbo frames if all hops support |
| App says "broken pipe" only on large responses | PMTUD broken, clamp MSS |
| `ip route get` shows tiny MTU | Stale PMTUD cache → `ip route flush cache` |
| Docker pods can't reach external services | Pod MTU > underlay MTU − overhead |

## ELI10

Think of MTU as the size of the biggest box that fits through the smallest door on a delivery route.

You pack a 2-foot-wide package (the MTU), but one door on the way is only 1.5 feet wide. Two things can happen:

1. **Fragmentation**: the delivery person cuts the package into smaller pieces and re-tapes them at the destination. Annoying, slow, lose one piece and the whole thing is ruined.

2. **PMTUD**: the delivery person sends a note back: "use 1.5-foot boxes next time." You repack everything smaller. Faster on subsequent shipments — but if mailmen ignore the note (firewalls block ICMP), you have a **black hole**: packages keep getting refused with no explanation.

**MSS clamping** is a smart sorting center near the small door that automatically opens incoming boxes and labels them "1.5ft max" so senders know without having to be told.

## Further Resources

- [RFC 791 - IPv4 (fragmentation)](https://tools.ietf.org/html/rfc791)
- [RFC 1191 - Path MTU Discovery](https://tools.ietf.org/html/rfc1191)
- [RFC 8201 - IPv6 PMTUD](https://tools.ietf.org/html/rfc8201)
## Where this connects

- [IPv4](ipv4.md) — IPv4 allows in-network fragmentation; PMTUD avoids it
- [IPv6](ipv6.md) — IPv6 never fragments in-network; PMTUD is mandatory
- [TCP](tcp.md) — TCP sets MSS based on PMTUD to avoid fragmentation
- [Ethernet/VLAN](ethernet_vlan.md) — the 1500-byte Ethernet MTU is the most common PMTUD constraint
- [QUIC](quic.md) — QUIC performs its own PMTUD (DPLPMTUD) at the application layer

- [RFC 4821 - Packetization Layer PMTUD](https://tools.ietf.org/html/rfc4821)
- [Linux PMTUD documentation](https://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt)
