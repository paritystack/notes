# BGP & Anycast

## Overview

**BGP** (Border Gateway Protocol) is the routing protocol that makes the internet work. It's how the ~80,000 independent networks ("Autonomous Systems") that compose the internet tell each other which IP prefixes they can deliver. **Anycast** is a routing trick — announcing the same IP from many places — built on top of BGP, used by every major CDN, DNS provider, and DDoS-mitigation service.

This note covers BGP fundamentals (enough to understand "the AWS outage was a BGP leak"), then how anycast leverages it for low-latency global services.

## BGP at a Glance

```
The internet:
  ~80,000 ASes, each owning some IP prefixes
  Connected by physical fiber, peering agreements, and contracts
  Need a way to say "I can reach prefix X via path Y"

BGP-4 (RFC 4271, 2006):
  - Path-vector protocol
  - Runs over TCP port 179
  - Exchanged between routers configured as "BGP peers"
  - Each AS advertises its prefixes and learns routes to others
  - Policy-driven: routing decisions follow business rules,
    not just shortest path
```

## Autonomous Systems

An **AS** is a network under one administrative control with a unique ASN (Autonomous System Number).

```
ASN format:
  16-bit:  1 - 65535       (1995-era)
  32-bit:  65536 - 4294967295 (RFC 4893)

Examples:
  AS15169  Google
  AS32934  Facebook/Meta
  AS13335  Cloudflare
  AS16509  Amazon (AWS)
  AS8075   Microsoft

Private ASNs (won't appear on internet):
  64512 - 65535 (16-bit)
  4200000000 - 4294967294 (32-bit)
```

You get an ASN by becoming a member of a Regional Internet Registry (ARIN, RIPE, APNIC, LACNIC, AFRINIC).

## BGP Sessions

### eBGP vs iBGP

```
eBGP (External BGP):
  Between routers in DIFFERENT ASes
  TTL=1 by default (must be directly connected)
  Updates AS_PATH (prepends own ASN)

iBGP (Internal BGP):
  Between routers in the SAME AS
  Doesn't modify AS_PATH
  Requires full mesh OR route reflectors
  Used to propagate eBGP-learned routes internally
```

### Establishing a Session

```
1. TCP connection on port 179
2. OPEN message exchange (ASN, hold time, capabilities)
3. KEEPALIVE (every ~30s)
4. UPDATE messages (announce/withdraw prefixes)
5. NOTIFICATION on errors → tears down session

States:
  Idle → Connect → Active → OpenSent → OpenConfirm → Established
```

```bash
# Cisco-style status
show ip bgp summary
show ip bgp neighbors

# Linux (FRR / BIRD)
vtysh -c "show bgp summary"
birdc show protocols
```

## BGP UPDATE Message

The core of BGP — announces a prefix with attributes describing the path to it.

```
Withdrawn Routes (list of prefixes no longer reachable)
Path Attributes:
  ORIGIN          IGP / EGP / Incomplete
  AS_PATH         sequence of ASNs traversed
  NEXT_HOP        IP to forward to
  MULTI_EXIT_DISC suggested entry point (between two ASes)
  LOCAL_PREF      iBGP-only preference
  COMMUNITIES     metadata tags
  ATOMIC_AGGREGATE / AGGREGATOR
NLRI (Network Layer Reachability Info):
  list of prefixes being announced
```

### Sample BGP Route

```
*> 8.8.8.0/24    198.51.100.1   0   0  15169 i
  │ │            │              │   │  │     └ origin (i=IGP)
  │ │            │              │   │  └────── AS_PATH (Google's AS)
  │ │            │              │   └───────── MED
  │ │            │              └───────────── LOCAL_PREF
  │ │            └──────────────────────────── NEXT_HOP
  │ └─────────────────────────────────────── prefix
  └─────────────────────────────────────── best route marker
```

## Path Selection (Cisco's classic order)

When multiple routes exist for the same prefix:

```
1. Highest WEIGHT (Cisco-only, local)
2. Highest LOCAL_PREF (preferred for outbound traffic)
3. Locally-originated prefer
4. Shortest AS_PATH
5. Lowest ORIGIN (IGP > EGP > Incomplete)
6. Lowest MED (preferred for inbound)
7. eBGP > iBGP
8. Lowest IGP cost to NEXT_HOP
9. Oldest path
10. Lowest router ID (tiebreaker)
```

In practice the top three (or just `LOCAL_PREF` for outbound and `AS_PATH` length for everyone else) decide most routes. The rest are tiebreakers and operational knobs.

## Business Relationships

BGP routing is shaped by **money**, not topology. Three main relationship types:

```
1. Customer ↔ Provider (paid transit)
   Customer pays provider for internet access
   Provider announces customer's routes to the world
   Customer learns full internet routes (or default)

2. Peer ↔ Peer (settlement-free)
   Two networks exchange traffic for free
   Each only announces their own + their customers' routes
   (Never propagate one peer's routes to another peer)

3. Sibling
   Same admin, different ASes — generally exchange everything
```

### Valley-Free Routing

```
A path through the internet should look like a valley:
  customer → ... → provider → peer → provider → ... → customer

No "valleys-within-valleys" — that would mean a provider
is carrying transit traffic between two peers, for free.
```

Violations create route leaks (next section).

## Communities

Tags attached to routes for policy signaling. Format `ASN:value`.

```
65000:100   "prefer this route over others"
65000:666   "blackhole — drop traffic to this prefix"
3856:911    Provider-specific "do not export to peers"
NO_EXPORT   well-known: don't advertise outside AS
```

Providers publish their community lists so customers can influence routing (e.g., "tag with 7018:33 to prepend 3 times").

## Outages and Leaks

### Route Leaks

```
Customer's router accidentally announces "I can reach the world"
to its provider (instead of just its own prefixes).

If the provider accepts and re-announces, traffic for huge
swaths of the internet routes through the small customer.
Massive congestion, latency, sometimes outages.

Famous: 2008 Pakistan / YouTube; 2017 Level3; 2019 Verizon/Cloudflare leak.
```

### Route Hijacks (intentional or accidental)

```
AS announces a prefix it doesn't own (more specific = preferred).
Traffic intended for the real owner now flows to the hijacker.

Mitigations:
  - RPKI (Resource Public Key Infrastructure)
    Cryptographic statements of "AS X is authorized to announce prefix Y"
    Routers reject ROA-invalid routes
  - IRR-based prefix filtering (older, by-hand)
  - BGPsec (signs the AS_PATH — slow adoption)
```

### How Most Outages Actually Happen

```
- Misconfigured filter, accidentally suppress correct routes
- Maintenance withdrew a major peer link
- BGP session flap due to TCP MD5 mismatch
- "Big provider had a config push" — Facebook 2021, AWS multiple times
```

## Convergence and Stability

```
A prefix becomes unreachable somewhere:
  - Router withdraws route via UPDATE
  - Withdrawal propagates AS by AS
  - Routers run path selection
  - New best path installed
  - 30s to several minutes globally

Damping (less common now): if a prefix flaps too much,
suppress it for a while. Causes problems with anycast flaps.

Add-paths, BGP-LS, BMP, etc. are extensions for faster convergence
and better visibility.
```

## BGP Toolchain

### Show what the internet sees

```bash
# Looking glass tools (web)
https://lg.he.net/         # Hurricane Electric
https://lg.ripe.net/       # RIPE NCC
https://stat.ripe.net/     # rich AS / prefix data

# Command line via Team Cymru
whois -h whois.cymru.com " -v 8.8.8.8"
whois -h whois.cymru.com " AS15169"

# Trace AS path
mtr --aslookup 8.8.8.8
```

### Open-source BGP daemons

```
FRR (FRRouting)    fork of Quagga, used in containers/Cumulus
BIRD              lightweight, popular in IXPs and route servers
GoBGP             Go implementation, programmable
ExaBGP            Python, often used for traffic injection
OpenBGPD          OpenBSD, very clean codebase
```

### FRR minimal config

```
router bgp 65001
  bgp router-id 192.0.2.1
  neighbor 192.0.2.254 remote-as 65000
  neighbor 192.0.2.254 password verylongsecret

  address-family ipv4 unicast
    network 198.51.100.0/24
    neighbor 192.0.2.254 prefix-list MY-PREFIXES out
    neighbor 192.0.2.254 route-map RM-IN in
  exit-address-family
!
ip prefix-list MY-PREFIXES seq 5 permit 198.51.100.0/24
!
route-map RM-IN permit 10
  set local-preference 200
```

### Inspect

```bash
vtysh -c "show ip bgp"
vtysh -c "show ip bgp summary"
vtysh -c "show ip bgp 8.8.8.8/24"
vtysh -c "show ip route bgp"
```

## BGP in the Data Center

Modern data centers use **BGP as IGP** (instead of OSPF/IS-IS):

```
Spine-and-leaf fabric:
  Each leaf is its own AS (or shared private ASN)
  Each spine is its own AS
  ECMP across all spines
  Pod IPs / VTEP IPs distributed via BGP

Pros:
  - Simple, single protocol for both DC and external
  - Mature implementations
  - Policy via communities
  - Scales to thousands of nodes
```

Frameworks: Cumulus's "Auto-MLAG / EVPN", FRR with EVPN extensions, Cisco ACI, Arista EOS.

### BGP unnumbered

You don't need IPs on every BGP link — peer over link-local IPv6 (`fe80::/10`), discovered via NDP. Simplifies config dramatically.

## Anycast

### The Idea

```
Multiple servers, in different physical locations, announce the SAME IP via BGP.

Client traffic gets routed (by BGP path selection) to the nearest
or best-performing instance.

  client → router → "I have 4 paths to 1.1.1.1"
                   "shortest AS path: this one"
                   → traffic to nearest Cloudflare PoP
```

### Why Anycast

```
✓ Low latency: clients hit the closest instance
✓ Built-in load distribution: each region serves its own users
✓ Resilience: an instance failing withdraws its announcement,
  traffic flows to the next-nearest
✓ DDoS dispersion: attack from one botnet hits one region;
  legit users elsewhere are unaffected
✓ No DNS magic: works at the IP layer, transparent to apps
```

### Who Uses Anycast

```
DNS: 1.1.1.1 (Cloudflare), 8.8.8.8 (Google), 9.9.9.9 (Quad9), root servers
CDN: Cloudflare, Fastly, Akamai
NTP: time.cloudflare.com, time.google.com
Public APIs: Cloudflare Workers, Fastly Compute
DDoS mitigation: Akamai Prolexic, Cloudflare Magic Transit
TLS-friendly: NTP/DNS over QUIC anycast services
```

### Anycast Quirks

```
1. Routes can change mid-session
   Client connected to anycast IP via TCP. Routing change
   sends new packets to a different instance → connection
   resets. Real risk for long-lived TCP but minimal for short DNS lookups.

   Mitigations:
   - Use UDP (DNS, QUIC handle this)
   - Sticky routing via cookies and session migration (QUIC connection IDs!)
   - Keep BGP convergence tight, avoid flap damping

2. "Closest" by BGP ≠ closest by geography
   AS_PATH length is the proxy. A neighbor AS may be physically far.
   Real-world: 60-90% of users hit a "good" PoP; some are
   misrouted (Comcast user routed to Europe via a transit oddity).

   Mitigations:
   - More PoPs everywhere
   - Tuning prepends / community tags per region
   - Per-PoP advertise different more-specific prefixes
```

### Anycast vs DNS-based Load Balancing

```
DNS GeoIP:
  Client → resolver → DNS server picks IP based on resolver location
  Returns A record for nearest server
  
  Pros: more control, custom logic
  Cons: depends on resolver location (not user); TTL granularity;
        traffic shifting takes minutes

Anycast:
  Client → resolves to one IP → BGP routes to nearest instance
  No DNS logic, sub-second failover possible
  
  Pros: real network proximity, instant failover, simple
  Cons: less granular control, depends on BGP behavior
```

Often **combined**: DNS picks an anycast cluster IP per region, then anycast distributes inside the cluster.

## Anycast Implementation

### Simple Anycast Service

```
At each PoP:
  - Server has loopback interface with anycast IP (e.g., 192.0.2.1/32)
  - Local router peers BGP with server (or with another local router)
  - Server announces 192.0.2.1/32 when healthy
  - Withdraws on health failure (uses BFD or BGP keepalive timeout)

ExaBGP, GoBGP, BIRD, FRR all work. Or use cloud-managed:
  AWS: Global Accelerator (uses anycast under the hood)
  GCP: Anycast IP via Premium Tier
  Cloudflare: Magic Transit / Spectrum
```

### "Anycast" inside an AS (no BGP)

Same trick using IGP routes (OSPF):
```
Multiple servers each have a /32 loopback IP = same address
Local IGP picks the shortest-path instance
Common for internal anycast services
```

## Combining BGP with Other Tools

```
BGP + RPKI       cryptographic prefix authorization
BGP + EVPN       L2/L3 VPNs over MPLS or VXLAN
BGP + Segment Routing  IGP-aware path engineering via labels
BGP + FlowSpec   distribute fine-grained filtering rules (DDoS scrubbing)
BGP + BMP        export routing info to monitoring systems
```

## Common BGP Interview Questions

```
Q: How does BGP find the best route?
A: Walk the path-selection ladder (LOCAL_PREF → AS_PATH → ...).

Q: What's the difference between iBGP and eBGP?
A: Same AS vs different AS. iBGP doesn't modify AS_PATH and
   requires full mesh or route reflectors.

Q: Why is BGP "policy-driven"?
A: Business decides routing, not just topology. ASes choose what
   to announce/accept based on contracts.

Q: What's a route leak?
A: An AS accidentally re-announces routes it shouldn't (typically
   transit-customer learned routes to another transit-customer
   or peer). Violates valley-free routing.

Q: How does anycast work?
A: Same IP announced from multiple locations; BGP routes clients
   to the "best" path's instance.

Q: What if BGP is down?
A: The AS becomes a black hole for any prefix it owns. Hence
   importance of redundant sessions, transit providers,
   BGP secure routing.

Q: How does Cloudflare put 1.1.1.1 everywhere?
A: They announce 1.0.0.0/24 via BGP from every PoP. The internet
   routes each user to the topologically nearest instance.
```

## ELI10

The internet is a city of fortresses (autonomous systems). Each fortress has its own gates (BGP routers) and its own ZIP codes (IP prefixes). When you send a letter, your local post office doesn't know how to get to every ZIP code in the world — instead, it has a giant gossip notebook.

**BGP is the gossip protocol.** Every gate shouts to its neighbors: "I can reach ZIP code 8.8.8.0/24 — and to get there, you go through these fortresses (AS_PATH)." Neighbors gossip what they heard to their neighbors. Eventually every gate has a notebook of "to get to ZIP code X, here are the routes I know; the shortest one is via these fortresses."

But fortresses don't just pick the shortest path — they pick the one that **costs less money** (LOCAL_PREF). The "shortest path" tiebreaker is just used when business policy hasn't decided.

**Anycast** is a magic trick: the same ZIP code exists in many cities at once, and the gossip notebook says different cities for different senders. So a letter from Tokyo goes to Tokyo's copy, and a letter from London goes to London's copy. If Tokyo's office burns down, the next letter just routes to Singapore automatically — no DNS, no fancy software.

The dangerous moments: when someone in the gossip game lies ("I have the shortest path to all of Pakistan!") and everyone believes them. That's how Pakistan accidentally took down YouTube, and how AWS or Facebook occasionally vanish from the internet for an hour.

## Further Resources

- [RFC 4271 - BGP-4](https://tools.ietf.org/html/rfc4271)
- [RFC 4760 - Multiprotocol Extensions (MP-BGP)](https://tools.ietf.org/html/rfc4760)
- [RFC 6480 - RPKI Architecture](https://tools.ietf.org/html/rfc6480)
- [Cloudflare: How BGP works](https://blog.cloudflare.com/tag/bgp/)
- [BGP wedgies and route leaks (Cumulus)](https://cumulusnetworks.com/learn/bgp/)
- [bgp.tools](https://bgp.tools/) — live BGP routing data
- [routing.party](https://routing.party/) — Andree Toonk's BGP visualization
- [Pete Lumbis - BGP for the data center](https://www.nvidia.com/en-us/networking/border-gateway-protocol/)
