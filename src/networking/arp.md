# ARP (Address Resolution Protocol)

## Overview

ARP maps [IP](ip.md) addresses to MAC addresses on a local network. When your machine wants to send a packet to `192.168.1.1`, it needs to know which physical [Ethernet](ethernet_vlan.md) card to send the frame to — ARP answers that question. ARP operates at Layer 2.5 (between data link and network), runs only on the local segment, and is the glue that makes [IPv4](ipv4.md)-over-Ethernet work. [DHCP](dhcp.md) uses ARP for duplicate-address detection; IPv6 replaces ARP with ICMPv6 Neighbor Discovery.

## Why ARP?

```
Application: "Send data to 192.168.1.100"
       │
       ▼
IP Layer: "I have an IP, I need a MAC to put in the Ethernet frame"
       │
       ▼
ARP: "Who has 192.168.1.100? Tell 192.168.1.10"
       │
       ▼
Reply: "192.168.1.100 is at aa:bb:cc:dd:ee:ff"
       │
       ▼
Frame leaves with dst MAC = aa:bb:cc:dd:ee:ff
```

Without ARP, you'd need to manually maintain a MAC table for every device on every machine.

## How ARP Works

### The Basic Exchange

```
Host A (192.168.1.10, MAC: 11:11:11:11:11:11)
wants to talk to 192.168.1.100

Step 1: Check ARP cache
  arp -n
  → Not found

Step 2: Broadcast ARP Request
  Ethernet:
    dst MAC: FF:FF:FF:FF:FF:FF (broadcast)
    src MAC: 11:11:11:11:11:11
    type:    0x0806 (ARP)
  ARP:
    operation: 1 (request)
    sender MAC: 11:11:11:11:11:11
    sender IP:  192.168.1.10
    target MAC: 00:00:00:00:00:00 (unknown)
    target IP:  192.168.1.100

Step 3: All devices on segment receive it
  Devices with different IPs: ignore
  Device with 192.168.1.100: respond

Step 4: ARP Reply (unicast)
  Ethernet:
    dst MAC: 11:11:11:11:11:11
    src MAC: aa:bb:cc:dd:ee:ff
  ARP:
    operation: 2 (reply)
    sender MAC: aa:bb:cc:dd:ee:ff
    sender IP:  192.168.1.100
    target MAC: 11:11:11:11:11:11
    target IP:  192.168.1.10

Step 5: Both sides cache
  A learns: 192.168.1.100 → aa:bb:cc:dd:ee:ff
  B learns: 192.168.1.10  → 11:11:11:11:11:11
```

### Cross-Subnet Communication

ARP only works within a broadcast domain. To reach a remote IP:

```
Host A (192.168.1.10) wants to send to 10.0.0.5

1. A checks: is 10.0.0.5 on my subnet (192.168.1.0/24)? No.
2. A consults routing table → next hop is gateway 192.168.1.1
3. A does ARP for 192.168.1.1 (not 10.0.0.5)
4. A sends frame to gateway's MAC, but IP dst stays 10.0.0.5
5. Gateway routes the packet onward
```

This is why every machine ARPs for its default gateway constantly.

## ARP Packet Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Hardware Type (1)         |     Protocol Type (0x0800)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| HW Addr Len(6)| Proto Len (4) |       Operation (1 or 2)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Sender Hardware Address (bytes 0-3)              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Sender HW Addr (bytes 4-5)    | Sender Protocol Addr (0-1)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Sender Protocol Addr (2-3)    | Target HW Address (bytes 0-1) |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Target Hardware Address (bytes 2-5)              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               Target Protocol Address (4 bytes)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Field | Value | Notes |
|-------|-------|-------|
| Hardware Type | 1 | Ethernet |
| Protocol Type | 0x0800 | IPv4 |
| HW Addr Len | 6 | MAC = 48 bits |
| Proto Len | 4 | IPv4 = 32 bits |
| Operation | 1 / 2 | Request / Reply |

Total: 28 bytes ARP payload + 14 bytes Ethernet header = 42 bytes (padded to 60 on wire).

## ARP Cache

Every device caches recent mappings to avoid ARP'ing on every packet.

```bash
# Linux
ip neigh show
arp -n

# Output:
192.168.1.1    dev eth0  lladdr aa:bb:cc:dd:ee:ff  REACHABLE
192.168.1.50   dev eth0  lladdr 11:22:33:44:55:66  STALE

# macOS / BSD
arp -a
arp -d 192.168.1.50          # delete entry

# Windows
arp -a
arp -d *
netsh interface ip delete arpcache
```

### Entry States (Linux NUD - Neighbour Unreachability Detection)

| State | Meaning |
|-------|---------|
| **REACHABLE** | Confirmed working, recently used |
| **STALE** | Cached but unconfirmed; will probe before use |
| **DELAY** | Pending probe |
| **PROBE** | Actively sending unicast probe |
| **FAILED** | No response, will re-ARP next time |
| **PERMANENT** | Static entry, never expires |

### Cache Lifetime

```
Linux defaults (sysctl net.ipv4.neigh.default.*):
  gc_stale_time:      60s   # mark STALE after idle
  base_reachable_time: 30s   # how long REACHABLE lasts
  gc_thresh1: 128            # min entries before GC starts
  gc_thresh3: 1024           # hard max

Tune for large L2 domains (e.g., container hosts):
  sysctl -w net.ipv4.neigh.default.gc_thresh3=4096
```

## ARP Variants

### Gratuitous ARP

A device announces its own IP→MAC without being asked. Two uses:

```
1. Announce presence on join / IP change
   "Hey everyone, I'm 192.168.1.10 at MAC X"
   → updates everyone's cache
   → detects duplicate IPs (someone else replies)

2. Failover (VRRP, keepalived)
   Backup gateway takes over IP
   → blasts gratuitous ARP
   → switches/hosts update MAC tables
   → traffic redirects in seconds
```

Format: ARP **request** where sender IP == target IP. Receivers update their cache; no reply expected.

### ARP Probe

Used in DHCP — after getting an IP, the client sends an ARP request for that IP with sender IP `0.0.0.0`. If anyone replies, the IP is taken (conflict) and the client sends DHCPDECLINE.

### Proxy ARP

A router answers ARP requests on behalf of hosts on another subnet, transparently relaying traffic. Used to merge two subnets or to handle hosts that don't know about routing.

```
Subnet 192.168.1.0/24 with router doing proxy ARP for 192.168.2.0/24

Host A (192.168.1.10) ARPs for 192.168.2.20 (wrong subnet from its perspective)
  → Router replies with its own MAC
  → A sends frame to router
  → Router forwards to actual 192.168.2.20

Enable on Linux:
  sysctl -w net.ipv4.conf.eth0.proxy_arp=1

Mostly legacy. Causes mysterious bugs. Avoid in new designs.
```

### RARP (Reverse ARP)

"I know my MAC, what's my IP?" — used by diskless workstations in the 80s. Obsoleted by BOOTP and DHCP.

## ARP Spoofing (Poisoning)

The original IP→MAC binding has no authentication. Anyone can claim any IP.

### The Attack

```
Real layout:
  Victim:  192.168.1.10
  Gateway: 192.168.1.1 (MAC: GG:GG:GG:GG:GG:GG)
  Attacker: 192.168.1.50 (MAC: AA:AA:AA:AA:AA:AA)

Attack:
  1. Attacker sends gratuitous ARP:
     "192.168.1.1 is at AA:AA:AA:AA:AA:AA"
     → Victim caches gateway = attacker's MAC

  2. Attacker also tells gateway:
     "192.168.1.10 is at AA:AA:AA:AA:AA:AA"
     → Gateway caches victim = attacker's MAC

  3. All traffic now flows through attacker:
     Victim ──> Attacker ──> Gateway ──> Internet
                    │
                    └──> sees / modifies everything
```

### Tools (for testing your own network)

```bash
# arpspoof (dsniff package)
sudo arpspoof -i eth0 -t 192.168.1.10 192.168.1.1

# ettercap
sudo ettercap -T -M arp:remote /192.168.1.10// /192.168.1.1//

# bettercap
sudo bettercap -iface eth0
> set arp.spoof.targets 192.168.1.10
> arp.spoof on
```

### Mitigations

```
1. Static ARP entries (for critical hosts)
   arp -s 192.168.1.1 aa:bb:cc:dd:ee:ff

2. Dynamic ARP Inspection (DAI) on switches
   - Uses DHCP snooping binding table
   - Drops ARP frames that don't match
   - Cisco: ip arp inspection vlan 10

3. Encryption (HTTPS, SSH, VPN)
   - ARP spoofing still works, but MITM sees encrypted traffic only
   - TLS pinning prevents downgrade attacks

4. arpwatch
   - Daemon that logs MAC changes for IPs
   - Alerts on suspicious flips

5. 802.1X / port security
   - Authenticate devices before allowing L2 access
   - Limit MACs per switch port
```

## ARP for IPv6: NDP

IPv6 doesn't use ARP at all. Instead, **Neighbor Discovery Protocol (NDP)** runs over ICMPv6 and uses multicast (not broadcast).

| Feature | ARP (IPv4) | NDP (IPv6) |
|---------|------------|------------|
| Protocol | Own L2 ethertype | ICMPv6 |
| Discovery method | Broadcast | Multicast (solicited-node) |
| Cache verification | Timeouts | NUD (built-in) |
| Router discovery | Separate (DHCP/static) | Built-in (RA messages) |
| Authentication | None | SEND (rarely deployed) |
| Address autoconfig | DHCP only | SLAAC built-in |

NDP messages:
- **NS** (Neighbor Solicitation) — like ARP Request
- **NA** (Neighbor Advertisement) — like ARP Reply
- **RS** (Router Solicitation)
- **RA** (Router Advertisement)
- **Redirect**

## Practical ARP Commands

```bash
# View cache
ip neigh
ip -s neigh                      # with stats
arp -a

# Add static entry
sudo ip neigh add 192.168.1.50 lladdr aa:bb:cc:dd:ee:ff dev eth0

# Delete entry
sudo ip neigh del 192.168.1.50 dev eth0

# Flush all
sudo ip neigh flush all
sudo ip neigh flush dev eth0

# Send gratuitous ARP (announce presence)
sudo arping -U -I eth0 192.168.1.10
sudo ip neigh change 192.168.1.10 nud reachable lladdr ...

# Ping using ARP only (no IP)
sudo arping -c 3 -I eth0 192.168.1.1

# Trigger ARP for a host
ping -c 1 192.168.1.100          # then check cache

# Watch ARP traffic
sudo tcpdump -i eth0 -n arp
sudo tcpdump -i eth0 -e arp      # show MAC addresses

# Wireshark filter:
arp
arp.opcode == 1                  # requests only
arp.duplicate-address-detected   # built-in conflict detection
```

## Reading ARP Traffic

```
17:23:45 ARP, Request who-has 192.168.1.100 tell 192.168.1.10, length 28
17:23:45 ARP, Reply 192.168.1.100 is-at aa:bb:cc:dd:ee:ff, length 46
17:23:46 ARP, Request who-has 192.168.1.1 (ff:ff:ff:ff:ff:ff) tell 192.168.1.10, length 28
17:23:46 ARP, Reply 192.168.1.1 is-at 11:22:33:44:55:66, length 46
```

## Gotchas

### ARP Cache Poisoning by Mistake

VRRP/HSRP failover, network restoration after split-brain, or a misconfigured second DHCP server can all cause incorrect ARP entries. Symptom: connectivity from some hosts but not others.

### Same MAC on Two IPs

Aliased interfaces (multiple IPs on one NIC) reply with the same MAC for multiple ARP queries. Normal.

### ARP Storms

Misconfigured network with a loop (no STP) causes broadcast frames — including ARP — to flood endlessly. Switch CPU goes to 100%, network dies.

### Silent ARP Failures

Some firewalls and overlay networks block ARP. Symptom: TCP connections hang, ping fails, even though both hosts are on the same subnet. Check `ip neigh` for FAILED entries.

### Container/Cloud ARP Limits

Linux's default `gc_thresh3=1024` is fine for desktops but small for container hosts with thousands of veth interfaces. Bump it.

### Wi-Fi and ARP

Wi-Fi APs sometimes drop broadcast frames for power saving. ARP becomes unreliable. Modern APs proxy ARP on behalf of associated clients (ARP suppression).

## ARP vs Related Concepts

| Concept | Layer | Purpose |
|---------|-------|---------|
| **ARP** | L2.5 | IP → MAC on local segment |
| **NDP** | L3 (ICMPv6) | IPv6 equivalent of ARP + more |
| **DNS** | L7 | Hostname → IP |
| **MAC learning** | L2 | Switch builds port → MAC table |
| **Routing** | L3 | Choose next-hop IP |
| **DHCP** | L7 | Assign IP, gateway, DNS |

## ELI10

ARP is what happens when you want to send a letter inside your apartment building, but you only know the person's name and not their apartment number.

You yell down the hallway: **"Hey, who's Alice? I have a letter for her!"** (broadcast ARP request)

Alice hears and yells back: **"That's me, apartment 304!"** (ARP reply)

Now you know to slip the letter under door 304. Next time you write to Alice, you remember (ARP cache). After a while, you forget (cache timeout) and need to yell again.

**ARP spoofing** is like a stranger yelling **"I'm Alice!"** when you ask. You hand your mail to them instead.

## Further Resources

- [RFC 826 - ARP](https://tools.ietf.org/html/rfc826)
- [RFC 5227 - IPv4 Address Conflict Detection](https://tools.ietf.org/html/rfc5227)
## Where this connects

- [Ethernet/VLAN](ethernet_vlan.md) — ARP runs inside Ethernet frames on the local segment
- [IPv4](ipv4.md) — ARP resolves IPv4 addresses; IPv6 uses ICMPv6 Neighbor Discovery instead
- [DHCP](dhcp.md) — DHCP clients ARP for the offered address before accepting it (DAD)
- [Firewalls](firewalls.md) — ARP spoofing attacks can be mitigated with dynamic ARP inspection (DAI)

- [RFC 4861 - Neighbor Discovery for IPv6](https://tools.ietf.org/html/rfc4861)
- [Linux kernel ARP documentation](https://www.kernel.org/doc/Documentation/networking/arp.txt)
