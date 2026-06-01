# DHCP (Dynamic Host Configuration Protocol)

## Overview

DHCP automatically assigns [IP](ip.md) addresses and other network configuration to devices on a network. Without it, every device would need manual IP configuration. DHCP runs over [UDP](udp.md) (server port 67, client port 68) and is the reason your laptop "just works" when you join a Wi-Fi network. It also provides the [DNS](dns.md) server address and default gateway; [ARP](arp.md) is used during the DHCP address-conflict-detection phase.

## Why DHCP?

### Without DHCP

```
Manual configuration per device:
  - IP address
  - Subnet mask
  - Default gateway
  - DNS servers
  - NTP servers
  - Domain name

Problems:
  - Doesn't scale (hundreds of devices)
  - IP conflicts
  - Hard to change network parameters
  - Mobile devices break when moving networks
```

### With DHCP

```
Device joins network → gets full configuration in ~1 second
  - IP address from a pool
  - All other parameters from server
  - Lease-based (auto-renewal)
  - Centrally managed
```

## The DORA Exchange

DHCP uses a four-step handshake: **D**iscover, **O**ffer, **R**equest, **A**cknowledge.

```
Client                                Server
  |                                     |
  |--- DHCPDISCOVER (broadcast) ------->|   "Anyone there?"
  |     src: 0.0.0.0:68                 |
  |     dst: 255.255.255.255:67         |
  |                                     |
  |<-- DHCPOFFER (broadcast/unicast) ---|   "Try 192.168.1.50"
  |     Offered IP, lease, options      |
  |                                     |
  |--- DHCPREQUEST (broadcast) -------->|   "I'll take it"
  |     Echoes offered IP               |
  |                                     |
  |<-- DHCPACK (broadcast/unicast) -----|   "Confirmed, 24h lease"
  |     Final configuration             |
  |                                     |
  | --- Network configured, lease active ---
```

### Why Broadcast?

The client has no IP yet, so it can't unicast. The server broadcasts the offer because the client can't receive unicast without an IP either. The destination MAC is the client's hardware address (which the server learned from the DISCOVER frame).

### DHCPDISCOVER

```
BOOTP message type: 1 (request)
Hardware type: 1 (Ethernet)
Hardware addr length: 6
Hops: 0
Transaction ID: 0x3903F326 (random)
Seconds: 0
Flags: 0x8000 (broadcast)
Client IP: 0.0.0.0
Your IP: 0.0.0.0
Server IP: 0.0.0.0
Gateway IP: 0.0.0.0
Client MAC: 00:0b:82:01:fc:42

Options:
  Magic cookie: 0x63825363
  Message Type: 1 (DHCPDISCOVER)
  Parameter Request List: [1, 3, 6, 15, 31, 33, 43, 44, 46, 47, 119, 121]
    (subnet mask, router, DNS, domain, etc.)
  Host Name: "laptop"
  Vendor Class: "MSFT 5.0"
```

### DHCPOFFER

```
BOOTP message type: 2 (reply)
Your IP: 192.168.1.50      ← offered
Server IP: 192.168.1.1     ← from this server
Gateway IP: 0.0.0.0
Client MAC: 00:0b:82:01:fc:42

Options:
  Message Type: 2 (DHCPOFFER)
  Subnet Mask: 255.255.255.0
  Router: 192.168.1.1
  DNS Servers: 8.8.8.8, 8.8.4.4
  Lease Time: 86400 (24 hours)
  Server Identifier: 192.168.1.1
```

### DHCPREQUEST

```
Options:
  Message Type: 3 (DHCPREQUEST)
  Requested IP: 192.168.1.50      ← echoing offer
  Server Identifier: 192.168.1.1  ← which server we picked
```

### DHCPACK

```
Final confirmation with all parameters
Client may now use the IP
ARP probe to verify no conflict
```

## DHCP States

```
INIT ──DISCOVER──> SELECTING ──REQUEST──> REQUESTING ──ACK──> BOUND
                                                                 │
                                                                 │ T1 (50% lease)
                                                                 ▼
                                                              RENEWING
                                                                 │
                                                                 │ T2 (87.5% lease)
                                                                 ▼
                                                              REBINDING
                                                                 │
                                                                 │ lease expires
                                                                 ▼
                                                                INIT
```

### Lease Renewal Timers

| Timer | Default | Action |
|-------|---------|--------|
| **T1** | 50% of lease | Unicast RENEW to original server |
| **T2** | 87.5% of lease | Broadcast REBIND to any server |
| **Expire** | 100% of lease | Release IP, restart from INIT |

Renewal is just a DHCPREQUEST/DHCPACK exchange — no DISCOVER needed if the same server is reachable.

## DHCP Options

Options carry configuration beyond just the IP. There are hundreds defined; common ones:

| Option | Name | Purpose |
|--------|------|---------|
| **1** | Subnet Mask | `255.255.255.0` |
| **3** | Router | Default gateway |
| **6** | DNS Servers | Resolver list |
| **15** | Domain Name | `example.com` |
| **42** | NTP Servers | Time sync |
| **51** | Lease Time | Seconds |
| **53** | Message Type | DISCOVER/OFFER/REQUEST/ACK |
| **54** | Server Identifier | DHCP server IP |
| **55** | Parameter Request List | What client wants |
| **66** | TFTP Server | For PXE boot |
| **67** | Bootfile Name | PXE boot image |
| **121** | Classless Static Routes | Override default route |

### PXE Boot

Options 66 and 67 enable diskless boot — the device gets an IP, fetches a kernel from a TFTP server, and boots over the network. Heavily used in data centers for provisioning.

## DHCP Relay Agents

DHCP uses broadcast, which doesn't cross routers. In larger networks, you'd need a DHCP server on every subnet — instead, routers run **DHCP relay agents** (also called `ip helper-address` in Cisco).

```
Subnet A (192.168.1.0/24)        Subnet B (10.0.0.0/24)
        Client                        DHCP Server
          │                                ▲
          │ DISCOVER (broadcast)           │
          ▼                                │
        Router ─── relays unicast ────────┘
        (sets giaddr = 192.168.1.1)
```

### giaddr Field

The **gateway IP address** field tells the server which subnet the request came from, so it picks the right pool. The server's reply goes back to the relay, which broadcasts it on the originating subnet.

### Option 82 (Relay Agent Information)

Carries metadata like the switch port and VLAN the client connected on. Used for tracking, security, and pool selection.

## DHCP Server Configuration

### ISC dhcpd Example

```
# /etc/dhcp/dhcpd.conf

default-lease-time 600;
max-lease-time 7200;
authoritative;

subnet 192.168.1.0 netmask 255.255.255.0 {
  range 192.168.1.100 192.168.1.200;
  option routers 192.168.1.1;
  option subnet-mask 255.255.255.0;
  option domain-name-servers 8.8.8.8, 8.8.4.4;
  option domain-name "example.local";
  option ntp-servers 192.168.1.1;
}

# Static reservation by MAC
host printer {
  hardware ethernet 00:11:22:33:44:55;
  fixed-address 192.168.1.50;
}
```

### dnsmasq (simpler, common on routers)

```
# /etc/dnsmasq.conf
dhcp-range=192.168.1.100,192.168.1.200,24h
dhcp-option=3,192.168.1.1            # router
dhcp-option=6,8.8.8.8,8.8.4.4        # DNS
dhcp-host=00:11:22:33:44:55,192.168.1.50,printer
```

## DHCP Client Tools

```bash
# Linux (dhclient)
sudo dhclient -v eth0           # Request lease
sudo dhclient -r eth0           # Release lease
cat /var/lib/dhcp/dhclient.leases

# systemd-networkd
networkctl status eth0

# NetworkManager
nmcli connection show
nmcli device show eth0

# macOS
sudo ipconfig set en0 DHCP      # Trigger DHCP
ipconfig getpacket en0          # Show full DHCP info

# Windows
ipconfig /release
ipconfig /renew
ipconfig /all
```

## DHCPv6

IPv6 has two configuration modes that often coexist with DHCPv6:

```
1. SLAAC (Stateless Address Autoconfiguration)
   - Router advertises prefix
   - Device derives address from prefix + MAC (EUI-64) or random
   - No DHCP server needed

2. DHCPv6 Stateful
   - Server assigns full IPv6 address
   - Similar to DHCPv4 but uses multicast (ff02::1:2)
   - Server port 547, client port 546

3. DHCPv6 Stateless
   - SLAAC for address
   - DHCPv6 for other options (DNS, NTP)
   - Common hybrid mode
```

### Key Differences from DHCPv4

```
- Multicast instead of broadcast (IPv6 has no broadcast)
- DUID instead of MAC for client ID
- Prefix delegation (IA_PD) for routers
- No DORA — uses Solicit/Advertise/Request/Reply
```

## DHCP Security

### Rogue DHCP Servers

A malicious or misconfigured device runs a DHCP server, hands out wrong gateways, and intercepts traffic (MITM).

```
Attack:
  1. Attacker connects to LAN
  2. Runs DHCP server with own IP as gateway
  3. Wins race with legitimate server
  4. All victim traffic routes through attacker
```

### DHCP Snooping

A switch feature that prevents rogue DHCP servers:

```
Switch ports classified as:
  - Trusted: connects to legit DHCP server / uplink
  - Untrusted: connects to clients

DHCP server messages (OFFER, ACK) only allowed from trusted ports.
Untrusted client messages (DISCOVER, REQUEST) are inspected and
a binding table is built (MAC → IP → port → VLAN).
```

The binding table also feeds **Dynamic ARP Inspection** and **IP Source Guard** to prevent ARP spoofing and IP spoofing.

### DHCP Starvation

Attacker floods DHCPDISCOVER with spoofed MACs, exhausting the pool. Legitimate clients can't get leases. Mitigation: rate limiting on switch ports, `ip dhcp snooping limit rate`.

### Option 82 Spoofing

Relays normally add option 82; switches should drop option 82 on untrusted ports to prevent spoofing.

## DHCP Lease Process Edge Cases

### Client Returning to Same Network

Client remembers its previous IP and starts at REBINDING state:

```
DHCPREQUEST (with previous IP in options)
  → if available: DHCPACK
  → if taken: DHCPNAK, restart from INIT
```

### Address Conflict Detection

After receiving DHCPACK, client sends an ARP probe for the assigned IP. If someone replies, client sends DHCPDECLINE and restarts.

### Lease Expiry While Disconnected

```
Lease 24h, device offline for 48h
  → Lease expired on server, IP returned to pool
  → On reconnect: DISCOVER from scratch
  → May get a different IP
```

## DHCP Packet Capture

```bash
# Capture DHCP traffic
sudo tcpdump -i any -n 'port 67 or port 68' -vv

# Watch live in Wireshark
# Filter: bootp or dhcp
```

### Example Capture

```
17:23:45.123 BOOTP/DHCP, Request from 00:0b:82:01:fc:42, length 300
  Client-Ethernet-Address 00:0b:82:01:fc:42
  Hostname "laptop"
  DHCP-Message Option 53, length 1: Discover
  Parameter-Request Option 55, length 12:
    Subnet-Mask, Default-Gateway, Domain-Name-Server, ...

17:23:45.234 BOOTP/DHCP, Reply, length 300, xid 0x3903f326
  Your-IP 192.168.1.50
  Server-IP 192.168.1.1
  Lease-Time Option 51: 86400
  Subnet-Mask Option 1: 255.255.255.0
  Default-Gateway Option 3: 192.168.1.1
  Domain-Name-Server Option 6: 8.8.8.8, 8.8.4.4
  DHCP-Message Option 53, length 1: Offer
```

## Troubleshooting

### Client gets no IP

```bash
# 1. Check link
ip link show eth0                # Is interface up?

# 2. Try manual DHCP request
sudo dhclient -v eth0            # See DISCOVER/OFFER/REQUEST/ACK

# 3. Capture traffic
sudo tcpdump -i eth0 port 67 or port 68

# 4. Check server logs
journalctl -u isc-dhcp-server
tail -f /var/log/syslog | grep dhcpd

# 5. Verify pool not exhausted
dhcp-lease-list
```

### 169.254.x.x address

This is **APIPA** (link-local). It means DHCP failed entirely — server unreachable or no lease available. Device self-assigned a link-local address.

### Wrong subnet IP

Likely a relay agent misconfiguration, or two DHCP servers handing out conflicting pools.

## DHCP vs Static IP vs SLAAC

| Method | Config | Use case |
|--------|--------|----------|
| **Static** | Manual | Servers, routers, infrastructure |
| **DHCP reservation** | MAC-based fixed IP | Servers that need stable IP but central management |
| **DHCP dynamic** | Random from pool | Workstations, laptops, phones |
| **SLAAC** | IPv6 prefix + auto suffix | IPv6 client devices |
| **APIPA** | Self-assigned 169.254.x | Fallback when DHCP fails |

## Performance & Scale

```
Typical lease times:
  - Public Wi-Fi: 1-4 hours (high churn)
  - Office: 8-24 hours
  - Home: 24h-7d
  - Servers: ∞ (infinite/reservation)

DHCP server can handle:
  - Thousands of clients per minute
  - Pool size = subnet size minus reserved
  - Bottleneck: usually database I/O for lease storage
```

## ELI10

DHCP is like checking into a hotel:

1. **DISCOVER** — You walk up to the front desk: "I need a room."
2. **OFFER** — Receptionist: "Room 304 is available, it's $100/night."
3. **REQUEST** — You: "I'll take 304."
4. **ACK** — Receptionist: "Here's your key, checkout's at noon tomorrow."

The key (IP address) is yours for the lease period. Before checkout (T1), you can ask to extend. If you don't show up to extend, the room goes back into the pool and someone else can use it.

A "rogue DHCP server" is like a scammer in the lobby pretending to be the receptionist and giving you a key to the wrong building.

## Further Resources

- [RFC 2131 - DHCP](https://tools.ietf.org/html/rfc2131)
- [RFC 2132 - DHCP Options](https://tools.ietf.org/html/rfc2132)
- [RFC 8415 - DHCPv6](https://tools.ietf.org/html/rfc8415)
## Where this connects

- [UDP](udp.md) — DHCP uses broadcast UDP (ports 67/68) before the client has an IP address
- [IPv4](ipv4.md) — DHCP is the primary dynamic address assignment mechanism for IPv4
- [DNS](dns.md) — DHCP response includes the DNS server address clients should use
- [ARP](arp.md) — DHCP clients use ARP to detect address conflicts before accepting a lease

- [ISC Kea (modern DHCP server)](https://www.isc.org/kea/)
- [dnsmasq](https://thekelleys.org.uk/dnsmasq/doc.html)
