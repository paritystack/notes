# OSPF & IS-IS

## Overview

**OSPF** (Open Shortest Path First) and **IS-IS** (Intermediate System to Intermediate System) are the two dominant **link-state interior gateway protocols (IGPs)** — used *within* a single Autonomous System to compute optimal paths between routers. They serve the same purpose; they differ mainly in heritage (IETF vs ISO), packaging, and how they're deployed in practice.

For interview prep: knowing **link-state vs distance-vector**, **LSDB**, **SPF/Dijkstra**, **areas**, and OSPF vs IS-IS tradeoffs is usually enough. This note covers both because they share 80% of their concepts and the differences are easy.

```
IGP vs EGP:
  IGP — routes inside one AS (OSPF, IS-IS, EIGRP, RIP)
  EGP — routes between ASes (BGP)

Link-state IGPs:
  Every router knows the full topology
  Local SPF computation per router
  Fast convergence
  
Distance-vector (RIP, EIGRP):
  Each router knows only neighbor distances
  "Routing by rumor"
  Slower convergence
```

## Link-State Basics

```
1. Each router discovers its directly-connected neighbors (Hello protocol)
2. Each router builds an LSA (Link-State Advertisement)
   = list of its links and costs
3. LSAs are FLOODED to every router in the area
4. Every router assembles all LSAs into an identical LSDB
   (Link-State Database) — same map of the network
5. Each router runs Dijkstra's SPF on its LSDB
   → shortest-path tree rooted at itself
6. Routes installed into RIB / FIB
```

The brilliant property: **everyone computes their own routes from the same map**, so there are no loops as long as the LSDBs match.

## OSPF (RFC 2328 for v2, RFC 5340 for v3)

### Versions

```
OSPFv2 — IPv4 only
OSPFv3 — IPv6 (and can carry IPv4 with extensions)
```

### Packet Types

| # | Type | Purpose |
|---|------|---------|
| 1 | Hello | Discover neighbors, maintain adjacency |
| 2 | Database Description (DBD) | Summarize LSDB during sync |
| 3 | Link State Request (LSR) | Request specific LSAs |
| 4 | Link State Update (LSU) | Carry LSAs |
| 5 | Link State Acknowledgment (LSAck) | Reliability for LSUs |

OSPF runs directly over IP (**protocol number 89**) — no TCP/UDP. Uses multicast `224.0.0.5` (AllSPFRouters) and `224.0.0.6` (AllDRouters).

### LSA Types (commonly seen)

```
1  Router LSA       links of one router (flooded within area)
2  Network LSA      router IDs attached to a multi-access network
3  Summary LSA      inter-area route advertisement
4  ASBR Summary     reach the AS boundary router
5  External LSA     routes redistributed from other protocols (e.g., BGP)
7  NSSA External    external routes in a not-so-stubby area
```

### Adjacency State Machine

```
Down → Init → 2-Way → ExStart → Exchange → Loading → Full
                       │
                       (DBD/LSR/LSU exchange)
```

`Full` means LSDBs synchronized, routing in operation.

### Areas

OSPF networks are split into **areas** to limit LSA flooding scope.

```
Backbone (Area 0)
   │
   ├── Area 1 (regular)
   ├── Area 2 (stub — no external LSAs)
   ├── Area 3 (totally stubby — no externals + no inter-area details)
   └── Area 4 (NSSA — stub but allows local externals)

All non-backbone areas must connect to Area 0
(directly or via virtual links — avoid virtual links in practice).

ABR (Area Border Router): on the boundary, summarizes between areas.
ASBR (AS Boundary Router): redistributes from another protocol.
```

Why areas: massive LSDBs → slow SPF + lots of flooding → bad. Areas reduce blast radius.

### Network Types

```
Broadcast (Ethernet)
  Elects DR/BDR (Designated Router / Backup) to reduce O(n²) flooding
  Hello 10s / Dead 40s

Point-to-Point
  Two routers only, no DR
  Hello 10s / Dead 40s

NBMA (Frame Relay etc.)
  Manual neighbors, DR election
  Hello 30s / Dead 120s

Point-to-Multipoint
  Treats hub-spoke as multiple P2P, no DR
```

### DR / BDR

On broadcast networks, all routers form full adjacency with the DR (and BDR). LSAs flow via DR, avoiding n² floods.

```
DR election:
  Highest router priority wins (configurable; 0 means "never DR")
  Tiebreaker: highest router ID

DR sends to AllDRouters (224.0.0.6)
Routers send to AllSPFRouters (224.0.0.5)
```

### Metric

```
Cost = ref_bandwidth / interface_bandwidth

Cisco default ref_bandwidth = 100 Mbps:
  10 Mbps  → cost 10
  100 Mbps → cost 1
  1 Gbps   → cost 1   (!)  ← needs adjustment
  10 Gbps  → cost 1   (!)

Modern networks: set auto-cost reference-bandwidth 100000 (or higher)
```

### OSPFv2 Config (Cisco-style)

```
router ospf 1
 router-id 192.0.2.1
 auto-cost reference-bandwidth 100000
 area 0 authentication message-digest
 network 10.1.1.0 0.0.0.255 area 0
 network 10.1.2.0 0.0.0.255 area 1
 passive-interface default
 no passive-interface GigabitEthernet0/0
 default-information originate
!
interface GigabitEthernet0/0
 ip ospf message-digest-key 1 md5 myKey
```

### Linux (FRR)

```
router ospf
 ospf router-id 192.0.2.1
 network 10.1.1.0/24 area 0
 passive-interface default
 no passive-interface eth0
!
interface eth0
 ip ospf cost 10
 ip ospf hello-interval 5
 ip ospf dead-interval 20
```

```bash
vtysh -c "show ip ospf neighbor"
vtysh -c "show ip ospf database"
vtysh -c "show ip ospf interface"
vtysh -c "show ip route ospf"
```

## IS-IS (ISO 10589 / RFC 1195)

The ISO-developed alternative. Often preferred in ISP backbones (Level3, telcos, big cloud providers) for stability and extensibility. Runs **directly on L2** — no IP required, which makes it ideal for bootstrap and very robust.

### Naming Quirks (legacy ISO terminology)

```
IS  = "Intermediate System" = router
ES  = "End System" = host
PDU = "Protocol Data Unit" = packet

Levels (instead of areas):
  Level 1 (L1): within an area
  Level 2 (L2): between areas (backbone)
  L1/L2 router: ABR equivalent
```

### Addresses (NETs)

```
A NET (Network Entity Title) identifies an IS-IS router.

Format:
  49.0001.0000.0c12.3456.00
  └─┘ └──┘ └────────────┘ └┘
   │   │         │         └─ NSEL (always 00 for routers)
   │   │         └─ System ID (6 bytes, unique per router)
   │   └─ Area ID
   └─ AFI (49 = private)

Example: 49.0001.1921.6800.1001.00 (looks like a router-id encoded)
```

### Levels

```
L1 routers: like OSPF intra-area, only know their area
L2 routers: like OSPF Area 0, run the backbone
L1/L2: bridge between

In practice, ISPs often run everything as L2 only — flat backbone.
```

### Packet Types

```
Hello (IIH)        — neighbor discovery
LSP (Link-State Packet) — equivalent to LSAs (carries router's links)
CSNP (Complete Sequence Numbers PDU)  — full LSDB summary
PSNP (Partial SNP) — request/ack LSPs
```

### Why ISPs Prefer IS-IS

```
✓ Runs on L2 — no IP dependency, easier to bootstrap
✓ Single protocol can carry routes for any address family
  (IPv4, IPv6, MPLS labels via SR-MPLS) without redesign
✓ TLV-based LSPs → easily extended with new info
✓ Generally less verbose, fewer LSDB types
✓ Hierarchical Levels work cleanly for large backbones
✓ Believed to be more stable under churn

Cons:
✗ Less common in enterprise — fewer engineers know it
✗ Tooling sometimes weaker outside ISP gear
```

### FRR IS-IS Config

```
router isis CORE
 net 49.0001.0000.0c12.3456.00
 is-type level-2-only
 metric-style wide
!
interface eth0
 ip router isis CORE
 ipv6 router isis CORE
 isis circuit-type level-2-only
 isis metric 10
```

```bash
vtysh -c "show isis neighbor"
vtysh -c "show isis database"
vtysh -c "show ip route isis"
```

## Convergence

```
Failure detected → BFD or hello-timeout (sub-second with BFD)
Router floods updated LSP/LSA
LSDB updates across area
Each router re-runs SPF (microseconds for small areas, hundreds of ms for big)
FIB updates → forwarding switches

End-to-end: 50-200ms with BFD + tuned timers
Without BFD: hello-dead time (10-40s default OSPF)
```

### Tuning for Fast Convergence

```
1. BFD (Bidirectional Forwarding Detection)
   Sub-second link failure detection
   OSPF/IS-IS triggers reconvergence on BFD down

2. SPF throttling
   Avoid running SPF on every tiny event
   Wait + back-off when flapping

3. iSPF / incremental SPF
   Only recompute affected nodes

4. LFA / Remote-LFA / TI-LFA
   Precomputed backup paths → 50ms protection
   Segment Routing helps a lot here
```

## OSPF vs IS-IS

| Feature | OSPF | IS-IS |
|---------|------|-------|
| Standard body | IETF | ISO (also IETF for IP extensions) |
| Runs over | IP (proto 89) | L2 directly |
| Areas | Backbone + others | Level 1 / Level 2 |
| Address families | OSPFv2 IPv4, OSPFv3 IPv6 | Both in one protocol |
| Extensibility | Limited (LSA types) | TLV (flexible) |
| Adoption | Enterprise, K8s underlays | ISP/cloud backbones |
| Auth | MD5, HMAC-SHA | Cleartext, MD5, HMAC-SHA |
| LSDB scaling | Multiple LSA types complicate | Cleaner LSP structure |
| Modern features | OSPFv3 SR, OSPFv3 BGP-LS | IS-IS SR-MPLS / SRv6 |

For most interviews, "OSPF or IS-IS" can be answered as "same fundamentals, OSPF is the IETF flavor, IS-IS is what ISPs use."

## OSPF/IS-IS vs BGP

| | OSPF / IS-IS | BGP |
|---|--------------|-----|
| Scope | inside one AS | between ASes |
| Algorithm | Dijkstra SPF | path-vector + policy |
| Convergence | fast (ms-s) | slow (s-min) |
| Knows topology | yes (full LSDB) | no (just paths) |
| Selection | shortest metric | policy + AS_PATH + tiebreakers |
| Number of prefixes | thousands | millions (full table) |
| Use | move data inside an AS | reach the rest of the internet |

Most networks run **both**: IS-IS or OSPF as IGP for the underlay (loopback reachability), BGP overlaid for prefix reach and policy.

## Common Issues

### Adjacency stuck in ExStart

MTU mismatch on the link. OSPF's DBD packets get rejected. Set matching MTU or use `ip ospf mtu-ignore`.

### Routes missing in one area

```
✓ Are you advertising the prefix into OSPF?
✓ Are you summarizing wrong on the ABR?
✓ Did the area type filter the LSA (stub area drops type 5)?
```

### Flapping / unstable convergence

```
✓ Hello timer too short for a flaky link
✓ BFD misconfigured
✓ Hardware/optical issue
✓ Damping config too aggressive
```

### Cost / metric issues

```
Default cost on 10G interfaces = 1 (bug)
Fix: auto-cost reference-bandwidth 100000

Two equal-cost paths → ECMP. Want one preferred? Adjust cost.
```

### Router ID collision

Two routers with same router-id → adjacency goes haywire. Always set explicit router IDs from your management /32 loopbacks.

## When You'd Pick OSPF vs IS-IS vs BGP for Internal Routing

```
Small enterprise (< 50 routers):
  OSPF — well-known, GUI tooling, fine

ISP / large WAN:
  IS-IS — flat L2 backbone, easy MPLS extensions

Data center fabric (modern):
  BGP — even inside; uses unnumbered eBGP for spine-leaf
  Or OSPF if simpler / smaller

Hyperscale (Google/Meta-style):
  Often homegrown SDN with BGP-LS or proprietary protocols
```

## Tools

```bash
# FRR / Quagga
vtysh                            # interactive shell
vtysh -c "show ip ospf neighbor"
vtysh -c "show ip ospf database"
vtysh -c "show isis database"

# BIRD
birdc show protocols
birdc show ospf neighbors

# Wireshark filters
ospf
ospf.msg == 1                    # Hello packets
isis

# Capture
sudo tcpdump -i eth0 -nn proto ospf
sudo tcpdump -i eth0 -nn 'ether host 09:00:2b:00:00:14 or 09:00:2b:00:00:15'  # IS-IS multicast
```

## ELI10

Imagine you're delivering pizzas in a city, and every intersection (router) has a magic map app.

In **distance-vector** routing (RIP, ancient), every intersection just asks its neighbors "how far is each restaurant?" and trusts them. You don't actually have a map — you go by hearsay. Slow to react when streets close.

In **link-state** routing (OSPF, IS-IS), every intersection broadcasts: **"I have streets to these intersections, here's the speed limit on each."** Every intersection collects all these announcements and builds the same complete map. Then each intersection runs the same algorithm (Dijkstra) to figure out **its own** shortest path to every destination.

If a street closes, the affected intersection shouts "this street is dead!" and within milliseconds everyone's map updates and reroutes.

**OSPF** and **IS-IS** do basically the same thing, in different historical packaging:
- OSPF was designed for IP networks specifically; the IETF created it.
- IS-IS came from the ISO OSI world, runs without needing IP, and is more flexible — that's why phone companies and big ISPs love it.

**Areas / Levels** are how you avoid every intersection in the country needing to know about every street in every city — group nearby intersections, summarize their info at the border, keep the big maps small.

**BGP** is the *separate* problem of how to deliver pizzas to other cities (other ASes) — a different protocol with very different concerns (money, contracts, trust).

## Further Resources

- [RFC 2328 - OSPF v2](https://tools.ietf.org/html/rfc2328)
- [RFC 5340 - OSPF v3 for IPv6](https://tools.ietf.org/html/rfc5340)
- [RFC 1195 - IS-IS for IP](https://tools.ietf.org/html/rfc1195)
- [ISO/IEC 10589 - IS-IS](https://www.iso.org/standard/30932.html)
- [Cisco OSPF design guide](https://www.cisco.com/c/en/us/support/docs/ip/open-shortest-path-first-ospf/7039-1.html)
- [FRR documentation](http://docs.frrouting.org/)
- [Network Warrior by Gary A. Donahue](https://www.oreilly.com/library/view/network-warrior-2nd/9781449307974/)
