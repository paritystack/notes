# WireGuard

## Overview

WireGuard is a modern VPN protocol designed for simplicity, performance, and security. Created by Jason Donenfeld and merged into the Linux kernel (v5.6, 2020), it's a deliberate rewrite that replaces the complexity of [IPsec](ipsec.md)/OpenVPN with a minimal, opinionated design: ~4,000 lines of kernel code vs hundreds of thousands for alternatives. It runs over [UDP](udp.md), uses the same modern AEAD crypto family as [TLS 1.3](tls_ssl.md), and leans on the kernel's [IP](ip.md) routing to decide which peer handles which destination.

```
WireGuard's core idea:
  - No connection state machine, no rekeying dance
  - Just peers with public keys and allowed IPs
  - Stateless UDP transport
  - Modern crypto, no negotiation
  - "Cryptokey routing": public key = routing identity
```

## Why WireGuard

```
Versus OpenVPN:
  ✓ Much faster (in-kernel, no userspace context switch)
  ✓ Lower latency (no TLS handshake)
  ✓ Roams seamlessly across networks (4G ↔ Wi-Fi)
  ✓ Tiny attack surface

Versus IPsec:
  ✓ One protocol instead of three (ESP + IKE + auth)
  ✓ Config in 10 lines instead of 100
  ✓ Better NAT traversal
  ✓ No negotiation phase

Trade-offs:
  ✗ No certificate-based auth (just static keys)
  ✗ No dynamic peer discovery (you list every peer)
  ✗ No L2 (it's L3 only)
  ✗ Same UDP packet → adversary sees who talks to whom
```

## How It Works

### Cryptokey Routing

Each peer has a **static keypair**. Public keys identify peers, and each peer is associated with a list of IP ranges it's allowed to send/receive (`AllowedIPs`).

```
Peer A (key PKa)                          Peer B (key PKb)
  AllowedIPs:                               AllowedIPs:
    PKb → 10.0.0.0/24                         PKa → 10.1.0.0/24

When packet arrives:
  - Decrypt with private key
  - Check source IP is in sender's AllowedIPs
  - If not → drop (anti-spoofing)

When packet sent:
  - Look up destination IP in routing table → which peer?
  - Encrypt with that peer's public key
  - Send via UDP to peer's Endpoint
```

This is the simplification: **no separate routing config**. Public keys ARE the routing identity.

### Noise Protocol Framework

WireGuard uses **Noise_IK** for the handshake — Curve25519 ECDH, ChaCha20-Poly1305 AEAD, BLAKE2s, HKDF. No ciphersuite negotiation = no downgrade attacks.

```
4 fixed cryptographic primitives:
  Key exchange:  Curve25519 ECDH
  AEAD:          ChaCha20-Poly1305
  Hash:          BLAKE2s
  KDF:           HKDF
```

Want a different cipher? **Tough** — that's a feature. If a primitive breaks, the whole protocol is versioned and replaced.

### Handshake Messages

```
1. Initiator → Responder: Handshake Initiation
   (ephemeral key, encrypted static pubkey, timestamp)

2. Responder → Initiator: Handshake Response
   (ephemeral key, derived shared secret)

3. Data exchange begins (no further handshake needed
   until rekey timer, ~2 minutes)
```

Handshakes happen at most every 2 minutes or after key rotation. Rest of the time it's pure data packets with no overhead.

### Packet Format

```
+------+-----+----+-----+----------+--------+
| Type | Resv| Sender|Counter| Encrypted   |
|  1B  | 3B  |  4B   |  8B   | Payload     |
+------+-----+----+-----+----------+--------+

Type 1: Handshake Initiation
Type 2: Handshake Response
Type 3: Cookie Reply (DoS mitigation)
Type 4: Transport Data

Overhead per data packet: ~32 bytes
  16B header + 16B Poly1305 auth tag
```

## Configuration

WireGuard configs are dead simple. There's a server/client, but the protocol treats them symmetrically — both sides are peers.

### Generate Keys

```bash
wg genkey | tee privatekey | wg pubkey > publickey
# privatekey: keep secret
# publickey:  share with peers

# Optional: pre-shared key for post-quantum hardening
wg genpsk > preshared
```

### Server Config (`/etc/wireguard/wg0.conf`)

```
[Interface]
PrivateKey = SERVER_PRIVATE_KEY
ListenPort = 51820
Address    = 10.0.0.1/24
# Forward traffic and NAT (if this is the gateway)
PostUp     = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown   = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
# Alice's laptop
PublicKey  = ALICE_PUBLIC_KEY
AllowedIPs = 10.0.0.2/32

[Peer]
# Bob's phone
PublicKey  = BOB_PUBLIC_KEY
AllowedIPs = 10.0.0.3/32
```

### Client Config

```
[Interface]
PrivateKey = ALICE_PRIVATE_KEY
Address    = 10.0.0.2/24
DNS        = 10.0.0.1

[Peer]
# Server
PublicKey   = SERVER_PUBLIC_KEY
Endpoint    = vpn.example.com:51820
AllowedIPs  = 0.0.0.0/0, ::/0      # full tunnel — route ALL traffic via VPN
# AllowedIPs = 10.0.0.0/24         # split tunnel — only VPN subnet
PersistentKeepalive = 25            # NAT keepalive
```

### Start/Stop

```bash
sudo wg-quick up wg0
sudo wg-quick down wg0

# Or with systemd
sudo systemctl enable --now wg-quick@wg0

# Inspect runtime state
sudo wg show
sudo wg show wg0
```

### Sample `wg show` output

```
interface: wg0
  public key: SERVER_PUBLIC_KEY
  private key: (hidden)
  listening port: 51820

peer: ALICE_PUBLIC_KEY
  endpoint: 198.51.100.5:48211
  allowed ips: 10.0.0.2/32
  latest handshake: 32 seconds ago
  transfer: 12.4 MiB received, 31.2 MiB sent

peer: BOB_PUBLIC_KEY
  endpoint: 203.0.113.7:51820
  allowed ips: 10.0.0.3/32
  latest handshake: 1 minute, 5 seconds ago
  transfer: 850 KiB received, 2.1 MiB sent
```

## AllowedIPs: The Key Concept

`AllowedIPs` does double duty:

```
On EGRESS (sending):
  - Acts as a routing table
  - "Send packets destined for these IPs through this peer"

On INGRESS (receiving):
  - Acts as an ACL
  - "Only accept packets from this peer if source IP matches"
  - Anti-spoofing built into the protocol
```

### Split vs Full Tunnel

```
AllowedIPs = 10.0.0.0/24         → split tunnel (only VPN subnet)
AllowedIPs = 192.168.1.0/24, 10.0.0.0/24  → split with multiple LANs
AllowedIPs = 0.0.0.0/0, ::/0     → full tunnel (route all internet too)
```

When you set `AllowedIPs = 0.0.0.0/0`, `wg-quick` automatically installs a default route through `wg0`. To carve out exceptions, use policy routing or `Table = off`.

## NAT and Connectivity

### PersistentKeepalive

Most clients sit behind NAT. The NAT mapping times out when idle. `PersistentKeepalive = 25` makes the client send a tiny keepalive every 25 seconds (well under typical NAT timeouts).

```
[Peer]
PublicKey = SERVER_PUBLIC_KEY
Endpoint  = vpn.example.com:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
```

Server-side peers (with public IPs) don't need it; clients almost always do.

### Roaming

When a peer's source IP/port changes (Wi-Fi → cellular), WireGuard automatically updates its `Endpoint` — the next valid encrypted packet from the new address is enough. No reconnection visible to the user.

### No Endpoint Specified?

A peer with no `Endpoint` is a "passive" peer — it can only respond to incoming connections, not initiate. Typical for server-side config entries (server doesn't know where roaming clients will be).

## Linux Kernel Interface

```bash
# Create interface manually (alternative to wg-quick)
sudo ip link add wg0 type wireguard
sudo ip address add 10.0.0.1/24 dev wg0
sudo wg set wg0 listen-port 51820 private-key /etc/wireguard/private
sudo wg set wg0 peer ALICE_PUBLIC_KEY allowed-ips 10.0.0.2/32
sudo ip link set wg0 up

# Add/remove peers at runtime
sudo wg set wg0 peer NEW_PUBLIC_KEY allowed-ips 10.0.0.50/32
sudo wg set wg0 peer REVOKED_PUBLIC_KEY remove

# Reload config
sudo wg syncconf wg0 <(wg-quick strip wg0)
```

## Platforms

```
Linux:       in-kernel since 5.6 (preferred)
             userspace fallback: wireguard-go
FreeBSD:     kernel module
OpenBSD:     in-tree
macOS:       WireGuard app (App Store) — userspace + system extension
Windows:     WireGuard for Windows — wintun driver
iOS/Android: native apps, kernel/userspace
OPNsense, pfSense, MikroTik, etc. all support it
```

## Hub-and-Spoke vs Mesh

### Hub-and-Spoke (common)

```
        Server
       /  |  \
      /   |   \
  Client Client Client

Each client peers only with the server.
Client-to-client traffic routes via server.
Easy to manage, server is a bottleneck.
```

### Mesh

```
  A ──── B
  │ \  / │
  │  \/  │
  │  /\  │
  │ /  \ │
  C ──── D

Every peer talks to every other peer directly.
Configure each side's AllowedIPs with all subnets.
Better latency, more config.
Use a control plane (Tailscale, Netbird, Headscale) to automate.
```

### Tailscale / Headscale / Netbird

Wrappers around WireGuard that solve its weakness (manual peer setup):

```
- Control plane assigns IPs, distributes peer pubkeys
- Coordinates NAT traversal (STUN-style)
- SSO/MFA integration
- ACLs (which peers can reach which ports)
- WireGuard does the data plane; control plane handles config

Tailscale = SaaS
Headscale = self-hosted Tailscale-compatible server
Netbird   = self-hosted alternative
```

## Performance

```
- ~1 Gbps single-threaded on commodity laptop
- ~5+ Gbps multi-core on modern server
- AES-NI not needed (ChaCha20 is fast everywhere)
- Lower latency than OpenVPN (~0.5ms overhead vs ~3-5ms)
- MTU 1420 default (1500 - 80 bytes overhead)
```

### MTU Gotcha

```
Outer IP/UDP/WireGuard header: 60 bytes (IPv4) or 80 (IPv6)
WireGuard sets MTU 1420 inside the tunnel by default.

If the underlying link has lower MTU (PPPoE, mobile), reduce wg MTU.
Otherwise: TCP works (PMTUD), UDP apps may fragment or fail silently.

Manual override:
  MTU = 1380
```

See [mtu_pmtud.md](mtu_pmtud.md) for the full story.

## Security Properties

### What WireGuard Provides

```
✓ Confidentiality (ChaCha20)
✓ Integrity (Poly1305)
✓ Authentication (each peer's static key)
✓ Forward secrecy (ephemeral DH per session)
✓ Identity hiding (static keys not sent in plaintext)
✓ Replay protection (counter + window)
✓ DoS protection (cookie mechanism under load)
✓ Stealth — no response to unauthenticated packets
  (port scanners see nothing; can't even tell WG is running)
```

### What It Doesn't Provide

```
✗ No certificate-based PKI (keys are static & trusted on first config)
✗ No "tunnel up/down" events (it's stateless)
✗ No L2 (can't bridge VLANs)
✗ No multicast / broadcast (it's IP-only)
✗ No per-user logging (peer = key, not username)
✗ Traffic analysis is possible (someone watching UDP can see packet sizes/timing)
```

### Pre-shared Keys (post-quantum)

Add a symmetric `PresharedKey` between peers for hybrid security. If Curve25519 ever falls to quantum computers, the PSK still protects the channel.

```
[Peer]
PublicKey   = ...
PresharedKey = $(wg genpsk)
```

## Operational Patterns

### Adding a new client

```
On client:
  wg genkey | tee privatekey | wg pubkey > publickey
  (write client config with server's public key + endpoint)

On server:
  Edit /etc/wireguard/wg0.conf, add [Peer] block with client's pubkey + AllowedIPs
  wg syncconf wg0 <(wg-quick strip wg0)   # apply without disconnecting others
```

### Revoking a client

```
Edit server config, remove [Peer] block.
wg syncconf wg0 <(wg-quick strip wg0)
```

If you suspect the private key is compromised, rotate the **server's** key — invalidates all clients.

### QR codes for mobile

```bash
# Generate client config, then:
qrencode -t ansiutf8 < client.conf

# Phone scans QR → instant WireGuard config in mobile app
```

### Per-device peers (recommended)

Don't share keys across devices. Each phone/laptop/server gets its own keypair + AllowedIP. Lets you revoke individual devices cleanly.

## Troubleshooting

```bash
# Is the interface up?
ip link show wg0
sudo wg show wg0

# Is the kernel module loaded?
lsmod | grep wireguard

# Are packets actually flowing?
sudo tcpdump -i wg0 -n
sudo tcpdump -i eth0 -n udp port 51820   # underlay

# Handshake never completes
- Check firewall on server: UDP 51820 open?
- Check NAT — client behind multiple layers?
- Public keys match? (typos very common)
- AllowedIPs include the source on each side?
- Time skew? (handshake uses TAI64N timestamps)

# Connects but no traffic
- IP forwarding enabled? sysctl -w net.ipv4.ip_forward=1
- NAT/masquerade rule present? iptables -t nat -L POSTROUTING
- DNS set on client? (full tunnel breaks default resolver)
- MTU issue? Try lowering to 1380.

# kernel log
journalctl -k | grep -i wireguard
dmesg | grep -i wireguard
```

## WireGuard vs Alternatives

| Feature | WireGuard | OpenVPN | IPsec | SSH Tunnel |
|---------|-----------|---------|-------|------------|
| **Layer** | L3 | L3 (TUN) / L2 (TAP) | L3 | L4 (TCP) |
| **Transport** | UDP only | UDP or TCP | IP proto 50 (ESP), UDP for NAT-T | TCP |
| **Codebase** | ~4K lines | ~100K lines | massive | medium |
| **Performance** | excellent | moderate | excellent | poor (TCP-in-TCP) |
| **Mobile roaming** | seamless | reconnect on switch | reconnect | reconnect |
| **Cert PKI** | no | yes | yes | yes |
| **NAT traversal** | built-in | works | needs NAT-T | works |
| **Setup difficulty** | very low | medium | high | trivial for one connection |
| **In-kernel** | yes (Linux) | no | yes (Linux) | no |
| **Browser/casual user** | manual | GUI clients | GUI clients | nope |

## ELI10

WireGuard is the no-nonsense VPN. Think of each computer as a person with a unique nameplate (their public key) glued to their forehead. To send a secret note, you just write it, seal it with the recipient's nameplate stamp, and toss it into the mailbox. Anyone can pick it up, but only the person whose nameplate matches can open it.

There's no "let's negotiate which cipher to use" — everyone uses the same envelope style. There's no "let me prove I am who I say I am via a 14-step ritual" — your nameplate IS your identity. If you switch from Wi-Fi to cell, your nameplate is still you, so the conversation just continues.

The only address book each person carries is **"this person's nameplate handles letters for these street addresses"** (AllowedIPs). That's literally all the configuration.

## Where this connects

- [IPsec](ipsec.md) — the older, heavier VPN stack WireGuard set out to replace
- [TLS/SSL](tls_ssl.md) — shares the ChaCha20-Poly1305 / Curve25519 crypto, different handshake
- [UDP](udp.md) — WireGuard's stateless transport; `PersistentKeepalive` keeps NAT mappings alive
- [IP](ip.md) / [IPv6](ipv6.md) — cryptokey routing maps `AllowedIPs` onto L3 forwarding
- [MTU/PMTUD](mtu_pmtud.md) — tunnel overhead forces a lower inner MTU
- [STUN](stun.md) — the NAT-traversal trick that mesh control planes (Tailscale) use to connect peers

## Further Resources

- [wireguard.com](https://www.wireguard.com/)
- [WireGuard whitepaper](https://www.wireguard.com/papers/wireguard.pdf)
- [Tailscale: How NAT Traversal works](https://tailscale.com/blog/how-nat-traversal-works/)
- [Headscale (self-hosted Tailscale)](https://github.com/juanfont/headscale)
- [Noise Protocol Framework](https://noiseprotocol.org/)
