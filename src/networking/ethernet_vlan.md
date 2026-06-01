# Ethernet & VLAN

## Overview

Ethernet (IEEE 802.3) is the dominant Layer 2 technology for wired LANs. VLANs (Virtual LANs, IEEE 802.1Q) let a single physical Ethernet network be partitioned into multiple logical broadcast domains. Together they form the backbone of every modern data center and office network. [ARP](arp.md) resolves [IP](ip.md) addresses to Ethernet MAC addresses; [MTU/PMTUD](mtu_pmtud.md) is set by the Ethernet link MTU (typically 1500 bytes); see also the [embedded Ethernet](../embedded/ethernet.md) page for MCU-level Ethernet integration.

## Ethernet Basics

### Frame Format

```
 0                   1                   2                   3
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Preamble (7 bytes) + SFD (1 byte) [hardware]           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                Destination MAC Address (6 bytes)              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Source MAC Address (6 bytes)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       EtherType / Length (2 bytes)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Payload (46-1500 bytes)                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Frame Check Sequence (FCS, 4 bytes, CRC32)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Field | Size | Purpose |
|-------|------|---------|
| Preamble + SFD | 8 B | Bit-level sync, stripped by NIC |
| Destination MAC | 6 B | Target NIC (or broadcast/multicast) |
| Source MAC | 6 B | Sender NIC |
| EtherType | 2 B | Protocol of payload (0x0800 = IPv4, 0x86DD = IPv6, 0x0806 = ARP, 0x8100 = VLAN-tagged) |
| Payload | 46–1500 B | The encapsulated packet |
| FCS | 4 B | CRC32, detects corruption |

### MAC Addresses

```
48 bits, written hex-colon notation:

  aa:bb:cc:dd:ee:ff
  └──┬──┘ └──┬──┘
     │       │
     │       └─ NIC-specific (24 bits, assigned by vendor)
     └────────  OUI (24 bits, Organizationally Unique Identifier)

Special bits in first byte:
  bit 0 (LSB): I/G — Individual (0) or Group/Multicast (1)
                FF:FF:FF:FF:FF:FF = broadcast
                01:00:5E:xx:xx:xx = IPv4 multicast
                33:33:xx:xx:xx:xx = IPv6 multicast
  bit 1:       U/L — Universal (0, vendor-burned) or Local (1)
```

### Frame Sizes

```
Minimum frame: 64 bytes (including header + FCS)
  → Payload must be ≥ 46 bytes (padded with zeros if shorter)
  → Required for CSMA/CD collision detection on classic Ethernet

Maximum frame: 1518 bytes (standard)
  → 1500-byte payload (the famous "1500 MTU")
  → +18 bytes (14 header + 4 FCS)

Jumbo frames: up to 9000 bytes (non-standard, datacenter use)
  → Less per-packet overhead, better throughput
  → All devices in path must agree
```

## How Switching Works

Switches build a **MAC address table** (also called a CAM table) by watching source MACs on incoming frames.

```
Initial state: empty MAC table

1. Host A (MAC AA) sends frame on port 1 to Host B (MAC BB)
   Switch: learns "AA is on port 1"
   Switch: doesn't know where BB is → floods to all ports (except 1)

2. Host B replies from port 3
   Switch: learns "BB is on port 3"
   Switch: sees dst AA → looks up table → forwards out port 1 only

3. Subsequent frames A↔B → forwarded directly between ports 1 and 3
```

### MAC Table Entries

```
Switch> show mac address-table

VLAN  MAC Address       Type     Port
1     00:1a:2b:3c:4d:5e DYNAMIC  Gi0/1
1     00:1a:2b:3c:4d:5f DYNAMIC  Gi0/2
10    00:11:22:33:44:55 DYNAMIC  Gi0/3
10    aa:bb:cc:dd:ee:ff STATIC   Gi0/24
```

Entries age out (default 300s) if no traffic seen.

### Flooding

Frames are flooded out all ports (except the source) when:
- Destination MAC unknown (unicast flooding)
- Destination is broadcast (`FF:FF:FF:FF:FF:FF`)
- Destination is multicast (unless IGMP snooping is enabled)

## Broadcast Domains

A broadcast domain is the set of devices that receive each other's broadcasts.

```
All ports of one VLAN on connected switches = one broadcast domain

Broadcasts hit every device in the domain:
  - ARP requests
  - DHCP discovers
  - NetBIOS / mDNS
  - Misconfigured chatty apps

Large broadcast domains = bad:
  - Wasted bandwidth (all NICs interrupt)
  - Security risk (anyone sees ARP, DHCP)
  - Fault scope (broadcast storms take down everything)

Rule of thumb: keep broadcast domains ≤ ~250 hosts
```

VLANs (next section) are how you split a broadcast domain without buying more switches.

## VLANs

A **VLAN** is a logical Layer 2 segment, identified by a 12-bit VLAN ID (1–4094).

### Why VLANs

```
Without VLANs:
  - Need separate physical switch for each network
  - Engineering, Finance, Guest all on separate hardware

With VLANs:
  - Single switch hosts many isolated networks
  - VLAN 10 = Engineering
  - VLAN 20 = Finance
  - VLAN 30 = Guest
  - Frames in VLAN 10 cannot reach VLAN 20 without a router
```

### VLAN ID Numbering

```
0      Reserved
1      Default VLAN (untagged on most switches)
2-1001 Standard VLANs
1002-1005 Reserved (legacy Token Ring/FDDI)
1006-4094 Extended VLANs
4095   Reserved
```

## 802.1Q Tagging

When traffic from multiple VLANs needs to traverse a single link (e.g., switch-to-switch), each frame gets a **VLAN tag** inserted between source MAC and EtherType.

### Tagged Frame Format

```
+----------------+--------------+--------+--------+---------+-----+
| Dst MAC (6)    | Src MAC (6)  | TPID   | TCI    | EthType | ... |
|                |              | 0x8100 | 16 bits|         |     |
+----------------+--------------+--------+--------+---------+-----+

TCI (Tag Control Information):
  PCP (3 bits): Priority Code Point (QoS, 0-7)
  DEI (1 bit):  Drop Eligible Indicator
  VID (12 bits): VLAN ID (0-4094)
```

The 4-byte tag pushes max frame size from 1518 to **1522 bytes** (or 1504 payload MTU on tagged links — confusing!).

### Wireshark View

```
Ethernet II, Src: 00:11:22:33:44:55, Dst: aa:bb:cc:dd:ee:ff
802.1Q Virtual LAN, PRI: 0, DEI: 0, ID: 10
Internet Protocol Version 4, Src: 192.168.10.5, Dst: 192.168.10.1
```

## Switch Port Modes

### Access Port

Belongs to exactly one VLAN. Strips tags on egress, adds VLAN context on ingress.

```
End-user device → access port → VLAN 10
  - Frames from device: untagged
  - Switch: associates them with VLAN 10
  - Frames to device: tag stripped before sending
```

### Trunk Port

Carries multiple VLANs, tags frames with VLAN ID.

```
Switch ↔ Switch trunk port:
  - Each frame tagged with its VLAN
  - Receiver knows which VLAN each frame belongs to
  - One physical link = many logical VLANs
```

### Native VLAN

On a trunk, one VLAN can be **native** — its frames are sent **untagged**. The other end must agree on the native VLAN.

```
Cisco default: VLAN 1 native
  - Untagged frames received → assumed VLAN 1
  - VLAN 1 frames sent → without tag
  - Other VLANs always tagged

Why: backwards compatibility with non-VLAN-aware devices
Security risk: VLAN hopping (see below)
Best practice: change native VLAN to an unused ID, never use VLAN 1
```

## Cisco-style Config (illustrative)

```
! Create VLANs
vlan 10
 name ENGINEERING
vlan 20
 name FINANCE
vlan 99
 name MGMT

! Access port for an engineer
interface GigabitEthernet0/1
 switchport mode access
 switchport access vlan 10
 spanning-tree portfast

! Trunk port to another switch
interface GigabitEthernet0/24
 switchport mode trunk
 switchport trunk allowed vlan 10,20,99
 switchport trunk native vlan 99
```

## Linux VLAN

```bash
# Create a VLAN sub-interface
sudo ip link add link eth0 name eth0.10 type vlan id 10
sudo ip addr add 192.168.10.5/24 dev eth0.10
sudo ip link set dev eth0.10 up

# View
ip -d link show eth0.10

# Persistent with netplan / NetworkManager / systemd-networkd
# (config varies by distro)

# Tear down
sudo ip link del eth0.10
```

## Inter-VLAN Routing

VLANs are isolated at Layer 2. To talk between them, you need a router or L3 switch.

```
Method 1: Router-on-a-stick
  - Single router interface, trunk port
  - One sub-interface per VLAN
  - Slower (router CPU bottleneck)

Method 2: L3 switch with SVIs (Switched Virtual Interfaces)
  - Switch acts as router between VLANs
  - Each VLAN has an SVI (e.g., interface vlan 10)
  - Hardware ASIC speed
  - Modern default
```

```
interface vlan 10
 ip address 192.168.10.1 255.255.255.0
interface vlan 20
 ip address 192.168.20.1 255.255.255.0
ip routing
```

Hosts in VLAN 10 use 192.168.10.1 as their default gateway. The switch routes packets to VLAN 20.

## Spanning Tree Protocol (STP)

If you have redundant links between switches, you get loops. Frames go round forever, broadcasts melt the network. STP prevents this by computing a loop-free tree.

### How STP Works

```
1. Elect Root Bridge (switch with lowest Bridge ID)
2. Each non-root switch picks its Root Port (lowest cost path to root)
3. Each segment elects a Designated Port (forwards toward root)
4. All other ports → BLOCKING state (no forwarding)

If a link fails, blocked ports re-converge to active forwarding.
```

### Port States (classic STP)

```
DISABLED → BLOCKING → LISTENING (15s) → LEARNING (15s) → FORWARDING
```

30 seconds to converge — painful, end users see DHCP timeouts.

### RSTP (802.1w) & MSTP (802.1s)

- **RSTP** — Rapid STP, converges in <1s. Default on most modern switches.
- **MSTP** — Multiple STP, one tree per group of VLANs, allows load balancing.

### PortFast / EdgePort

For access ports (only hosts attached, no switches), skip STP states and go straight to FORWARDING. Avoids 30s DHCP timeouts on PC boot. Cisco: `spanning-tree portfast`.

### BPDU Guard

If a PortFast port unexpectedly receives a BPDU (someone plugged in a switch), shut the port down — prevents accidental loops and rogue switches.

## VLAN Security

### VLAN Hopping (Double Tagging)

```
Attack: send a frame with two 802.1Q tags
  Outer tag: native VLAN of trunk
  Inner tag: target victim VLAN

Switch behavior (vulnerable):
  1. Receives frame, sees native VLAN tag → strips it
  2. Forwards frame on trunk
  3. Next switch sees the inner tag → forwards to victim VLAN

Mitigation: never use the native VLAN for any real traffic.
  Trunk: native vlan 999  (an unused ID)
  All real VLANs always tagged
```

### Switch Spoofing

Attacker sends DTP (Dynamic Trunking Protocol) frames to trick a switch into making their port a trunk → access to all VLANs.

```
Mitigation:
  switchport mode access      ! force access mode
  switchport nonegotiate      ! disable DTP
```

### Private VLANs (PVLANs)

Sub-divide a VLAN: hosts in the same PVLAN can't talk to each other, only to a promiscuous port (gateway).

```
Use case: hotel/dorm networks
  - All rooms in VLAN 100
  - PVLAN isolation prevents guest-to-guest snooping
  - Each guest can still reach the internet via gateway
```

## Link Aggregation (LACP / 802.3ad)

Combine multiple physical links into one logical link for bandwidth + redundancy.

```
LACP negotiation:
  - Both ends send LACPDUs
  - Form a bundle ("port-channel" / "bond")
  - Hash on src/dst MAC/IP/port to distribute flows

Note: single flow ≤ one link's speed. LACP doesn't speed up single TCP connection.
```

Linux:
```bash
sudo ip link add bond0 type bond mode 802.3ad
sudo ip link set eth0 master bond0
sudo ip link set eth1 master bond0
sudo ip link set bond0 up
```

## Power over Ethernet (PoE)

Ethernet cables can also carry DC power to devices (phones, APs, cameras).

| Standard | Power | Use |
|----------|-------|-----|
| 802.3af (PoE) | 15.4 W | IP phones, basic APs |
| 802.3at (PoE+) | 25.5 W | Pan-tilt cameras, mid APs |
| 802.3bt (PoE++/4PPoE) | 60–90 W | Wi-Fi 6 APs, displays, lighting |

## Modern Cabling Speeds

| Standard | Speed | Cable | Distance |
|----------|-------|-------|----------|
| 100BASE-TX | 100 Mb/s | Cat5 | 100m |
| 1000BASE-T | 1 Gb/s | Cat5e | 100m |
| 2.5GBASE-T | 2.5 Gb/s | Cat5e | 100m |
| 5GBASE-T | 5 Gb/s | Cat6 | 100m |
| 10GBASE-T | 10 Gb/s | Cat6a | 100m |
| 10GBASE-SR | 10 Gb/s | Multimode fiber | 300m |
| 40/100/400G | very fast | Single-mode fiber | km |

## Useful Commands

```bash
# View interfaces and MACs
ip link show
ip -d link show eth0

# Watch link state
ethtool eth0                     # speed, duplex, link
ethtool -S eth0                  # statistics
sudo ethtool -p eth0 5           # blink port LED for 5s

# Capture VLAN-tagged traffic
sudo tcpdump -i eth0 -e vlan
sudo tcpdump -i eth0 vlan 10

# Show ARP / MAC table on switch (Cisco)
show mac address-table
show vlan brief
show interfaces trunk
show spanning-tree

# Linux bridge (software switch)
sudo ip link add br0 type bridge
sudo ip link set eth0 master br0
sudo ip link set eth1 master br0
bridge fdb show                  # MAC forwarding table
```

## Comparison Table

| Concept | Layer | Purpose |
|---------|-------|---------|
| **Hub** | L1 | Repeats bits to all ports (obsolete) |
| **Switch** | L2 | Forwards by MAC, builds CAM table |
| **VLAN** | L2 | Logical partitioning of switch ports |
| **L3 Switch** | L2+L3 | Switch + hardware IP routing |
| **Router** | L3 | Forwards between IP subnets |
| **STP/RSTP** | L2 | Loop prevention |
| **LACP** | L2 | Link bundling |

## Troubleshooting

### Wrong VLAN

```
Symptom: device can't get DHCP / wrong subnet
Check:
  - switchport access vlan X matches DHCP scope
  - cable in correct port
  - port not err-disabled (show interfaces status)
```

### Trunk Mismatch

```
Symptom: some VLANs work between switches, others don't
Check:
  - switchport trunk allowed vlan on both ends
  - native vlan matches both ends (else CDP warns)
```

### Spanning-Tree Loop

```
Symptom: random high CPU, broadcast storm, network "slow"
Check:
  show spanning-tree summary
  show spanning-tree blocked
  Look for ports flapping or in unexpected forwarding state
```

### MAC Flapping

```
Symptom: log messages "MAC X moving between ports A and B"
Cause: usually a loop (cable between two switch ports without STP)
       or a misbehaving device with two NICs sharing a MAC
```

## ELI10

**Ethernet** is the postal system inside a building. Every mailbox (NIC) has a unique address (MAC). Letters (frames) have a sender and receiver address.

A **switch** is the mail clerk. At first, it doesn't know which mailbox is on which floor — when a letter arrives, it copies it to every floor (flooding). It watches return mail to learn ("oh, Alice's mailbox is on floor 3") and starts delivering directly.

A **VLAN** is like dividing the building into separate companies that share the same hallways and mailroom but can't read each other's mail. The mail clerk stamps each letter with a company tag (802.1Q) when it travels between mailrooms.

**Spanning Tree** is the rule that says "even if there are two hallways between mailrooms, only one is open at a time, so a letter never loops forever."

## Further Resources

- [IEEE 802.3 - Ethernet](https://standards.ieee.org/standard/802_3-2018.html)
- [IEEE 802.1Q - VLAN tagging](https://standards.ieee.org/standard/802_1Q-2018.html)
## Where this connects

- [ARP](arp.md) — maps IP addresses to MAC addresses within an Ethernet broadcast domain
- [IP](ip.md) — IP packets are encapsulated inside Ethernet frames
- [MTU/PMTUD](mtu_pmtud.md) — standard Ethernet MTU (1500 bytes) sets the baseline for PMTUD
- [OSPF/IS-IS](ospf_isis.md) — runs over Ethernet links to establish routing adjacencies
- [Embedded Ethernet](../embedded/ethernet.md) — MCU-level Ethernet MAC/PHY integration

- [Cisco - Configuring VLANs](https://www.cisco.com/c/en/us/td/docs/switches/lan/catalyst3850/software/release/16-1/configuration_guide/vlan/b_161_vlan_3850_cg.html)
- [Linux bridging documentation](https://wiki.linuxfoundation.org/networking/bridge)
