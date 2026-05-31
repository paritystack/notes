# Multicast (IGMP & PIM)

## Overview

**Multicast** delivers a single stream of packets to *many* receivers efficiently: the
sender transmits once, and the network duplicates packets only where paths diverge. It sits
between unicast (one-to-one) and broadcast (one-to-all-on-the-segment).

```
Unicast:    sender sends N copies, one per receiver        (wasteful for N big)
Broadcast:  one copy, every host on the L2 segment gets it (can't cross routers)
Multicast:  one copy; network forks it only at branch points to interested receivers

  Source ── R1 ──┬── R2 ── Host A  (joined)
                 │         Host B  (joined)
                 └── R3 ── Host C  (not joined → R3 gets no traffic)
```

Used for IPTV, financial market data feeds, [RTP](rtp.md)-based streaming, service
discovery ([mDNS](mdns.md) uses 224.0.0.251), routing protocols
([OSPF](ospf_isis.md) uses 224.0.0.5/6), and [VXLAN](overlay_networks.md) underlay BUM
traffic. Multicast is mostly an *enterprise/datacenter/provider* technology — it generally
does **not** traverse the public Internet.

## Addressing

```
IPv4 multicast: 224.0.0.0/4  (224.0.0.0 – 239.255.255.255), class "D"
  224.0.0.0/24      link-local, never routed (224.0.0.1 all hosts, .2 all routers,
                                              .5/.6 OSPF, .251 mDNS, .252 LLMNR)
  232.0.0.0/8       SSM (Source-Specific Multicast)
  239.0.0.0/8       administratively scoped / "private" multicast (org-local)

IPv6 multicast: ff00::/8
  ff02::1 all nodes,  ff02::2 all routers,  ff02::fb mDNS,  scope encoded in the address

L2 mapping (so NICs/switches can filter in hardware):
  IPv4 group → MAC 01:00:5e:xx:xx:xx  (low 23 bits of the IP copied in)
  IPv6 group → MAC 33:33:xx:xx:xx:xx  (low 32 bits)
  Note: the 23-bit IPv4 map is lossy → 32 IPs collide onto one MAC.
```

## How receivers join: IGMP (IPv4) / MLD (IPv6)

Hosts tell their **local router** which groups they want. IGMP (Internet Group Management
Protocol) runs between hosts and the first-hop router; MLD is the IPv6 equivalent (carried
in ICMPv6).

```
Host                                   Router (IGMP querier)
  |                                         |  periodically:
  |   <------ Membership Query ------------ |  "anyone want any group?"
  |                                         |
  | --- Membership Report (join G) -------> |  host wants group G
  |                                         |  → router starts forwarding G to this segment
  |                                         |
  | --- Leave Group (IGMPv2) -------------> |  host done; router sends group-specific
  |                                         |    query, prunes if no one answers

IGMP versions:
  v1  join only, timeout-based leave
  v2  adds explicit Leave + group-specific queries (faster pruning)
  v3  adds SOURCE filtering (INCLUDE/EXCLUDE lists) → required for SSM
```

### IGMP snooping (the switch optimization)

A plain L2 switch floods multicast like broadcast — defeating the point. **IGMP snooping**
lets the switch peek at IGMP reports and forward each group only out the ports that
actually joined.

```
Without snooping:   group G floods ALL switch ports (every host's NIC must discard it)
With snooping:      switch builds a per-group port list → G goes only to joined ports
  Needs an IGMP querier on the VLAN to keep state fresh (often the router, or the switch).
```

## How it's routed: PIM

Between routers, **PIM (Protocol Independent Multicast)** builds the distribution tree. It's
"protocol independent" because it reuses the existing unicast routing table
([OSPF](ospf_isis.md)/[BGP](bgp_anycast.md)) for its **Reverse Path Forwarding (RPF)**
check rather than computing its own.

```
RPF check: a multicast packet is accepted ONLY if it arrived on the interface the router
           would USE to send unicast back toward the source. Otherwise drop.
           → this is what prevents multicast loops.
```

### PIM modes

```
PIM-DM (Dense Mode):  flood-and-prune. Assume everyone wants it, push everywhere, then
                      prune branches with no receivers. Fine only when receivers are dense.

PIM-SM (Sparse Mode): explicit join. Nobody gets traffic until they ask. The dominant mode.
   - A Rendezvous Point (RP) is a meeting router known to all.
   - Receivers' routers join a shared tree (RPT) rooted at the RP: (*,G).
   - Source's router registers with the RP; traffic flows RP → receivers.
   - Once flowing, routers can switch to the Shortest-Path Tree (SPT): (S,G), bypassing RP.

PIM-SSM (Source-Specific): receiver names BOTH source and group (S,G) up front (via IGMPv3),
                      so NO RP is needed. Simplest and most secure; ideal for one-to-many
                      (IPTV). Uses 232.0.0.0/8.

BIDIR-PIM: bidirectional shared trees for many-to-many (e.g. conferencing) without (S,G) state.
```

```
PIM-SM shared tree → SPT switchover:

  Source ─ DR ──register──► RP ──shared tree──► Receivers' DRs ──► Hosts
                                  (initial path via RP)
  then:
  Source ─ DR ───────────── SPT (S,G) direct ──► Receivers   (optimal path, RP offloaded)
```

## Verifying multicast

```bash
# Show multicast group memberships on a host (Linux)
ip maddr show
netstat -gn

# Join a group for testing and watch traffic
# (socat can join + dump; or use iperf in multicast mode)
iperf -s -u -B 239.1.1.1 -i 1            # receiver joins 239.1.1.1
iperf -c 239.1.1.1 -u -T 5 -t 30          # source sends, TTL 5

# Cisco-style operational checks (for reference)
#   show ip igmp groups
#   show ip mroute              (multicast routing table: (*,G) and (S,G) entries)
#   show ip pim neighbor
#   show ip pim rp mapping
```

## Pitfalls & operational notes

```
- TTL/scope: link-local groups (224.0.0.0/24) are never forwarded; set TTL>1 to leave a segment.
- No IGMP snooping → multicast floods the VLAN like broadcast and crushes Wi-Fi (sent at low
  basic rates). Enable snooping + a querier on every multicast VLAN.
- No querier = snooping state expires = flooding returns. Configure one explicitly.
- RPF failures silently drop traffic when unicast and multicast topologies differ
  (asymmetric routing) — a classic "multicast just stops" cause.
- Multicast doesn't cross the Internet; for wide delivery use overlays/relays or unicast
  replication (CDNs do unicast).
- It's UDP-based: no built-in reliability/congestion control — apps add FEC or retransmit
  (e.g. RTP + NACK, or PGM).
```

## Related

- [mDNS](mdns.md) — uses link-local multicast 224.0.0.251 / ff02::fb for discovery
- [RTP](rtp.md) — multicast media distribution
- [OSPF / IS-IS](ospf_isis.md) — OSPF uses multicast; PIM borrows its unicast table for RPF
- [Overlay Networks (VXLAN)](overlay_networks.md) — multicast as a BUM-traffic transport
- [Ethernet & VLAN](ethernet_vlan.md) — IGMP snooping operates per-VLAN at L2
- [IPv6](ipv6.md) — MLD and ff00::/8 multicast
