# QoS & Traffic Shaping

## Overview

By default a network treats every packet equally and forwards "as fast as it can" — which
falls apart when a link is congested. **Quality of Service (QoS)** is the set of mechanisms
that decide, when there isn't enough bandwidth for everyone, *whose* packets go first,
*whose* wait, and *whose* get dropped. The goal is to protect latency-sensitive traffic
(voice, video, gaming) from bulk traffic (backups, downloads).

```
The three QoS knobs:
  Classification & marking → label packets by importance (DSCP)
  Queuing / scheduling     → serve important queues first
  Shaping / policing       → limit rate (delay vs drop excess)
```

QoS matters most for real-time flows: [RTP](rtp.md) media in a [SIP/VoIP](sip_voip.md) or
[WebRTC](webrtc.md) call needs low jitter and low loss; a backup over [TCP](tcp.md) just
wants throughput and tolerates delay.

## Why it's needed: bufferbloat

```
A single fat TCP download fills a big router buffer. Now your VoIP packets sit
BEHIND a queue full of bulk data → hundreds of ms of latency on an idle-looking link.

  [ bulk bulk bulk bulk bulk | voice ]  → voice waits for all the bulk to drain

That's "bufferbloat": oversized dumb FIFO buffers + greedy TCP = terrible latency.
Fix: smart queue management (fq_codel / CAKE) that keeps queues short and isolates flows.
```

## DiffServ: marking packets (DSCP)

Modern QoS uses **DiffServ** (Differentiated Services): each packet carries a marking in
the IP header's DS field, and routers apply **Per-Hop Behaviors (PHB)** based on it.

```
IPv4 ToS byte / IPv6 Traffic Class byte:

   0   1   2   3   4   5   6   7
 +---+---+---+---+---+---+---+---+
 |        DSCP (6 bits)  | ECN(2)|
 +---+---+---+---+---+---+---+---+
   DSCP = which class/PHB           ECN = congestion signaling (below)

Common DSCP values (PHBs):
  EF   (46)  Expedited Forwarding   → voice/RTP: low loss, low latency, low jitter
  AF4x (34..) Assured Forwarding    → video; 4 classes × 3 drop precedences
  AFxx        AF11..AF43            → tiered bulk/business traffic
  CS0  (0)   Default / best effort
  CS6/CS7    network control (routing protocols)
```

```
DiffServ domain:
  At the EDGE: classify and mark (trust boundary — re-mark untrusted DSCP to 0)
  In the CORE: just honor the mark, queue accordingly (stateless, scalable)
```

(Contrast: **IntServ/RSVP** reserves bandwidth per-flow end to end — accurate but doesn't
scale, so DiffServ won.)

## ECN: congestion without dropping

**Explicit Congestion Notification** lets a router *mark* a packet "I'm congested" instead
of dropping it. The receiver echoes the mark back; the [TCP](tcp.md) sender slows down —
avoiding the loss + retransmit penalty entirely.

```
ECN bits:  00 not-ECN-capable   10/01 ECN-capable (ECT)   11 Congestion Experienced (CE)
  Router near-full → set CE instead of dropping → sender backs off, no retransmit.
```

## Policing vs Shaping

Both limit a flow to a target rate, but differently:

```
POLICING                              SHAPING
  drops (or re-marks) excess            buffers excess and sends later
  no buffering → no added delay         adds delay, smooths bursts
  causes TCP retransmits                avoids drops, queue can overflow
  used by ISPs to enforce caps          used by sender to fit a downstream rate

  rate ──┐   drop  drop                 rate ──┐  (queue) smooths the curve
         └────────────                          └──────────────
```

Token-bucket is the usual model: tokens accrue at the committed rate (CIR), a burst (Bc)
is allowed up to the bucket size; packets without tokens are dropped (policing) or queued
(shaping).

## Queuing disciplines (Linux `tc`)

The Linux traffic-control subsystem (`tc`) attaches **qdiscs** to interfaces.

```
Classful (you build a hierarchy of bandwidth classes):
  HTB  (Hierarchy Token Bucket)  → most common; guaranteed + ceiling rates per class
  HFSC                            → precise latency + bandwidth decoupling

Classless (drop-in queue managers):
  pfifo_fast  → old default dumb FIFO (bufferbloat-prone)
  fq_codel    → modern default: flow queuing + CoDel AQM, kills bufferbloat
  cake        → fq_codel + shaping + DSCP-aware tins; ideal for a home gateway egress
  fq          → pacing for TCP, used with BBR
```

### Example: prioritize VoIP, cap a backup (HTB)

```bash
# Shape egress on eth0 to 90 Mbit, split into priority classes
tc qdisc add dev eth0 root handle 1: htb default 30

tc class add dev eth0 parent 1:  classid 1:1  htb rate 90mbit
tc class add dev eth0 parent 1:1 classid 1:10 htb rate 10mbit ceil 90mbit prio 0  # VoIP
tc class add dev eth0 parent 1:1 classid 1:20 htb rate 40mbit ceil 90mbit prio 1  # interactive
tc class add dev eth0 parent 1:1 classid 1:30 htb rate 40mbit ceil 50mbit prio 2  # bulk/backup

# Keep latency low within each class
tc qdisc add dev eth0 parent 1:10 fq_codel
tc qdisc add dev eth0 parent 1:20 fq_codel
tc qdisc add dev eth0 parent 1:30 fq_codel

# Classify by DSCP EF (voice) into the priority class
tc filter add dev eth0 parent 1: protocol ip prio 1 \
   u32 match ip dsfield 0xb8 0xfc flowid 1:10        # 0xb8 = EF(46)<<2
```

### Simplest modern fix: CAKE on the gateway

```bash
# One line that beats most hand-tuned setups for a home/office uplink
tc qdisc add dev eth0 root cake bandwidth 90mbit diffserv4
# Set bandwidth ~5-10% under your real uplink so the queue lives in Linux, not the modem.
```

## Wireless QoS

[WiFi](nac_8021x.md) and other shared media add their own QoS: **802.11e / WMM**
(Wi-Fi Multimedia) maps traffic into four Access Categories — Voice, Video, Best Effort,
Background — giving voice frames shorter contention windows for the airtime.

## Where to apply QoS (and where you can't)

```
- QoS only helps at the BOTTLENECK and only on traffic YOU schedule (usually egress).
- Your downstream is shaped by your ISP; you can only indirectly manage it (e.g. CAKE
  ingress shaping, or relying on TCP/ECN backoff).
- DSCP marks are routinely bleached (reset to 0) crossing the public Internet — they're
  trustworthy only within your own administrative domain or an SLA'd WAN/MPLS link.
```

## Security & pitfalls

```
- Untrusted DSCP is a DoS vector: a host marking everything EF would starve others.
  → Re-mark/clamp at the trust boundary.
- Policing TCP causes retransmit storms; prefer shaping where you control the sender.
- Shape a few % below the real link rate, or the dumb buffer downstream reintroduces bloat.
- QoS cannot create bandwidth — under sustained overload, something must be delayed/dropped.
```

## Related

- [RTP / RTCP](rtp.md) and [SIP/VoIP](sip_voip.md) — the classic EF-marked, jitter-sensitive traffic
- [TCP](tcp.md) — congestion control, ECN interaction, pacing (BBR + fq)
- [IPv4](ipv4.md) / [IPv6](ipv6.md) — the ToS / Traffic Class byte carrying DSCP
- [MTU & PMTUD](mtu_pmtud.md) — packet sizing interacts with queue latency
- [Firewalls](firewalls.md) — classification/marking often done alongside filtering
