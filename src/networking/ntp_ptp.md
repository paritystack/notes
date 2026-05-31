# Time Synchronization (NTP & PTP)

## Overview

Accurate clocks matter more than they seem: [TLS](tls_ssl.md) certificate validity windows,
[Kerberos](nac_8021x.md)/auth tickets, log correlation across hosts, distributed databases
(ordering, leases), and financial/regulatory timestamping all break when clocks drift.
Two protocols dominate:

```
NTP (Network Time Protocol)   → millisecond accuracy over WANs; ubiquitous; software-only
PTP (Precision Time Protocol) → sub-microsecond on a LAN; needs hardware timestamping
                                 (IEEE 1588) for telecom, finance, industrial, 5G fronthaul
```

Both run over [UDP](udp.md) (NTP on 123; PTP on 319/320).

## NTP

### Stratum hierarchy

NTP organizes time sources into **strata** by distance from a reference clock.

```
  Stratum 0  reference clocks: GPS, atomic, radio (DCF77/WWVB)  — not on the network
      |
  Stratum 1  servers directly attached to a stratum-0 source ("primary")
      |
  Stratum 2  sync from stratum-1 servers
      |
  Stratum 3  sync from stratum-2  ... (up to stratum 15; 16 = unsynchronized)

Lower stratum = closer to truth. Clients typically use several servers and pick the best.
```

### How NTP computes offset & delay

NTP exchanges four timestamps and solves for the clock **offset** while cancelling out the
network round-trip — assuming the path is roughly symmetric.

```
   Client                                Server
     t1  ── request (carries t1) ──────►
                                          t2  (server receive)
                                          t3  (server transmit)
     t4  ◄── response (carries t2,t3) ──

   round-trip delay  δ = (t4 - t1) - (t3 - t2)
   clock offset      θ = ((t2 - t1) + (t3 - t4)) / 2

   The client steers its clock toward +θ. Asymmetric paths are NTP's main error source.
```

NTP doesn't jump the clock around: it **disciplines** it — slewing (gradually speeding/
slowing) for small errors, stepping only for large ones, and statistically filtering/
selecting among multiple servers to reject bad ones ("falsetickers").

### chrony (the modern default)

`chrony` largely replaces the old `ntpd`; it converges faster, handles intermittent
connectivity and laptops/VMs better.

```bash
# /etc/chrony/chrony.conf
pool 2.pool.ntp.org iburst        # multiple servers, fast initial sync
server time.cloudflare.com iburst
makestep 1.0 3                    # step (not slew) if off by >1s, for first 3 updates
rtcsync                           # keep the hardware RTC disciplined

# Operations
chronyc tracking                  # current offset, stratum, root delay/dispersion
chronyc sources -v                # candidate servers and their state (* = selected)
chronyc sourcestats
timedatectl                       # systemd view of clock/sync status
```

### NTP security

```
- Authentication: legacy symmetric keys; modern NTS (Network Time Security, RFC 8915)
  uses TLS to key-exchange, then authenticates NTP packets — defeats MITM time-shifting.
- Amplification DDoS: the old "monlist" command returned huge replies to a spoofed source.
  → Disable monlist / mode 6-7 queries; rate-limit; restrict who can query.
- A spoofed clock can expire/forge TLS certs and auth tickets → time is a security dependency.
```

## PTP (IEEE 1588)

When milliseconds aren't enough (5G radio sync, power grids, trading, broadcast video,
industrial control), **PTP** reaches nanoseconds — but only on a controlled LAN with
hardware help.

### Why PTP is more accurate

```
NTP timestamps in software → OS scheduling/queuing jitter limits it to ~ms.
PTP timestamps in the NIC/switch hardware, as close to the wire as possible,
   and switches CORRECT for their own queuing delay → ns accuracy.
```

### Roles and clock types

```
Grandmaster (GM)   the best clock (usually GPS-locked); root of the timing tree
Ordinary Clock     an endpoint (master or slave)
Boundary Clock     a switch that is a slave upstream and a master downstream
                   (regenerates time, hides its hop)
Transparent Clock  a switch that measures how long a PTP packet dwelled inside it and
                   writes that into a correction field → downstream subtracts it

Best Master Clock Algorithm (BMCA) elects the grandmaster automatically.
```

### Sync exchange

```
  Master                                 Slave
    |── Sync (t1) ─────────────────────►  | (records t2)
    |── Follow_Up (precise t1) ────────►  |   (two-step: exact t1 sent after)
    |                                     |
    |  ◄──────────── Delay_Req (t3) ───── | (slave asks)
    |── Delay_Resp (t4) ───────────────►  |
    offset and path delay derived like NTP, but with hardware-grade timestamps.
```

```
Hardware needs:
  - NIC with PTP hardware clock (PHC):   check `ethtool -T eth0`
  - Switches as boundary/transparent clocks for ns accuracy (otherwise their jitter dominates)
Linux tooling: linuxptp — `ptp4l` (the PTP daemon) + `phc2sys` (sync the system clock to the PHC)
Profiles: default (1588), telecom G.8275.1/.2, power C37.238, AES67 (audio/video).
```

## NTP vs PTP — choosing

```
                 NTP                         PTP
  Accuracy       1–50 ms (WAN), ~ms LAN      sub-µs to ns (LAN, with HW)
  Scope          Internet / WAN              single LAN / admin domain
  Hardware       none (software clock)       NIC PHC + PTP-aware switches
  Transport      UDP/123, mostly unicast     UDP/319-320 or L2, often multicast
  Setup          trivial (point at a pool)   significant (profiles, HW, boundary clocks)
  Use            general servers, laptops    5G, trading, broadcast, industrial, power
```

For most servers: run **chrony against a good NTP pool**. Reach for PTP only when you have a
hard sub-millisecond requirement and the hardware to back it.

## Related

- [UDP](udp.md) — transport for both protocols
- [TLS/SSL](tls_ssl.md) — certificate validity depends on correct time; NTS reuses TLS
- [802.1X / RADIUS](nac_8021x.md) — Kerberos/auth tickets are time-sensitive
- [Multicast](multicast_igmp_pim.md) — PTP commonly uses multicast distribution
- [QoS](qos_traffic_shaping.md) — path asymmetry/jitter is the enemy of time accuracy
