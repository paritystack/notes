# IPsec

## Overview

IPsec (Internet Protocol Security) is a suite of protocols for authenticating and encrypting [IP](ip.md) packets. Unlike [TLS](tls_ssl.md) (which sits above [TCP](tcp.md)), IPsec works at Layer 3, so it can secure **any IP traffic** transparently — TCP, [UDP](udp.md), ICMP, even other tunnels. It's the dominant site-to-site VPN technology in enterprises and is heavily used in 4G/5G mobile networks. [Firewalls](firewalls.md) must be aware of IPsec to pass ESP/AH packets correctly.

## What IPsec Is (and Isn't)

```
IPsec is a FRAMEWORK, not a single protocol. It includes:

  ESP    Encapsulating Security Payload   (encryption + auth)
  AH     Authentication Header            (auth only, no encryption)
  IKE    Internet Key Exchange            (key negotiation + setup)
  IPComp IP Compression                   (rarely used now)

Plus dozens of related RFCs for ciphers, NAT traversal, IPv6, mobile, etc.
```

When people say "IPsec VPN," they usually mean **IKEv2 + ESP in tunnel mode**, the modern standard.

## Why IPsec (and When to Skip It)

```
Strengths:
  ✓ Standardized across vendors (Cisco, Juniper, MikroTik, Linux strongSwan)
  ✓ Site-to-site is its sweet spot
  ✓ Hardware acceleration widespread (NICs, routers)
  ✓ Works under FIPS / Common Criteria compliance regimes
  ✓ X.509 certificates for auth at scale

Pain points:
  ✗ Configuration is verbose and error-prone
  ✗ Multiple overlapping standards (IKEv1 vs IKEv2; legacy and modern modes)
  ✗ Many cipher choices → easy to misconfigure
  ✗ NAT traversal requires explicit NAT-T
  ✗ Debugging often requires reading hex dumps

For new deployments: consider WireGuard for simplicity unless
compliance, vendor mandates, or hardware-offload requirements force IPsec.
```

## Protocol Stack

```
┌──────────────────────────────────────┐
│  Application data                    │
├──────────────────────────────────────┤
│  TCP / UDP / ICMP                    │
├──────────────────────────────────────┤
│  ESP or AH                           │   ← IPsec data plane
├──────────────────────────────────────┤
│  IP                                  │
└──────────────────────────────────────┘

Parallel control plane:
┌──────────────────────────────────────┐
│  IKE (UDP 500, 4500 for NAT-T)       │   ← Negotiates SAs
└──────────────────────────────────────┘
```

## ESP (Encapsulating Security Payload)

The main IPsec data protocol — provides confidentiality, integrity, and authentication. IP protocol number **50**.

### ESP Packet Format

```
+----------------+                    ← original IP header (transport mode)
| Outer IP hdr   |                    ← new IP header (tunnel mode)
+----------------+
| ESP Header     |  SPI (4B) | Seq Number (4B)
+----------------+
| Payload Data   |  encrypted (cipher-dependent)
+----------------+
| Padding        |  cipher block alignment
+----------------+
| Pad Length     |  1 byte
+----------------+
| Next Header    |  1 byte (e.g., 6 = TCP, 4 = IP for tunnel mode)
+----------------+
| ESP Auth Data  |  ICV (cipher-dependent, often 12 or 16 bytes)
+----------------+
```

- **SPI** (Security Parameters Index): receiver uses this to find the matching SA (cipher, key)
- **Sequence Number**: replay protection
- **ICV** (Integrity Check Value): authenticates header + payload

### Modern Cipher Suites

| Cipher | Notes |
|--------|-------|
| `AES-GCM-128`, `AES-GCM-256` | AEAD, hardware accelerated, default modern |
| `AES-CTR + HMAC-SHA2` | Older, separate enc+auth |
| `ChaCha20-Poly1305` | Software-friendly, no AES-NI needed |
| `3DES`, `DES` | **Avoid**, weak/broken |
| `MD5`, `SHA-1` | **Avoid** for HMAC |

## AH (Authentication Header)

Provides integrity + authentication, but **no encryption**. IP protocol number **51**. Rarely used today — ESP can do the same job and adds encryption. AH also breaks with NAT (it authenticates the whole IP header, including addresses NAT rewrites).

```
Use cases:
  - Compliance requires integrity but cannot have encryption (rare)
  - Legacy interop
```

## Modes: Transport vs Tunnel

### Transport Mode

Only the **payload** of the IP packet is encrypted. Original IP header is preserved.

```
Before:
  [IP hdr | TCP | data]

After (ESP transport):
  [IP hdr | ESP | TCP | data (encrypted) | ESP trailer]
```

Use case: end-to-end between two hosts that both speak IPsec (e.g., two servers in a datacenter, or mobile networks).

### Tunnel Mode

The entire original packet is encrypted and wrapped in a **new** IP header. This is what "IPsec VPN" usually means.

```
Before:
  [IP hdr | TCP | data]

After (ESP tunnel):
  [Outer IP hdr | ESP | Inner IP hdr | TCP | data (encrypted) | ESP trailer]
```

Use case: site-to-site VPN, remote access. Outer IPs are the VPN gateways; inner IPs are the LAN endpoints.

## Security Associations (SAs)

An SA is a **one-way contract** between two peers:
- SPI
- Cipher + key
- Auth algorithm + key
- Mode (transport/tunnel)
- Lifetime (seconds + bytes)

For bidirectional communication, you need **two SAs** (one per direction). They're bundled into an **SA pair**. The collection of all SAs is the **SAD** (Security Association Database). The **SPD** (Security Policy Database) decides which traffic gets which SA.

```bash
# Linux SAD/SPD inspection
sudo ip xfrm state
sudo ip xfrm policy
```

## IKE (Internet Key Exchange)

IKE negotiates SAs over UDP 500. Two major versions exist; only IKEv2 should be used now.

### IKEv1 (legacy, RFC 2409)

- Phase 1: establish a secure channel (IKE_SA)
  - **Main Mode** (6 messages, identity-protecting)
  - **Aggressive Mode** (3 messages, faster, leaks identity, weakens to brute force — **avoid**)
- Phase 2: establish IPsec SAs (Quick Mode)
- Lots of cipher negotiation flexibility, lots of footguns

### IKEv2 (modern, RFC 7296)

- Always 4 messages for initial setup
- Built-in NAT-T, MOBIKE (mobility), EAP auth
- Simpler state machine
- Better error reporting
- Supports modern crypto suites

```
IKE_SA_INIT (request)
  ↳ Negotiate cipher proposals
  ↳ Diffie-Hellman key exchange
  ↳ Nonces

IKE_SA_INIT (response)
  ↳ Selected proposal
  ↳ DH response
  ↳ Nonces

IKE_AUTH (request)
  ↳ Encrypted identity
  ↳ Authentication (PSK or cert)
  ↳ TSi/TSr (traffic selectors — what subnets to tunnel)
  ↳ Child SA proposal (ESP cipher etc.)

IKE_AUTH (response)
  ↳ Responder identity
  ↳ Auth payload
  ↳ Selected child SA proposal
  ↳ Tunnel is up!
```

### Authentication

```
1. PSK (pre-shared key)
   - Simple, but doesn't scale, must distribute secret
2. RSA / ECDSA certificates (X.509)
   - PKI-based, recommended at scale
   - Cert pinning or CA-based trust
3. EAP (RADIUS, etc.)
   - User-based auth for remote access ("road warrior")
   - Often EAP-MSCHAPv2 or EAP-TLS
```

## NAT Traversal (NAT-T)

ESP doesn't have port numbers, so NAT can't track it. IPsec wraps ESP in UDP:

```
Detection during IKE:
  Both sides exchange NAT_DETECTION_SOURCE_IP / DESTINATION_IP hashes
  If hashes don't match expected → NAT detected
  Switch IKE port from 500 → 4500
  Encapsulate ESP in UDP port 4500: [UDP | ESP | inner]
```

Standard on every modern IPsec stack. Without NAT-T, IPsec breaks behind any home router.

```
Without NAT-T:                With NAT-T:
  ESP (IP proto 50)             UDP/4500 [ESP]
  IKE over UDP/500              IKE over UDP/4500
```

## strongSwan Configuration Example

The most widely used IPsec implementation on Linux. Modern config uses `swanctl.conf`.

```
# /etc/swanctl/conf.d/site-to-site.conf

connections {
    site-to-site {
        version = 2                        # IKEv2
        local_addrs  = 198.51.100.1
        remote_addrs = 203.0.113.5

        local {
            auth = pubkey
            certs = vpn-gateway.crt
            id = vpn.us.example.com
        }
        remote {
            auth = pubkey
            id = vpn.eu.example.com
        }

        children {
            site-to-site {
                local_ts  = 10.1.0.0/16
                remote_ts = 10.2.0.0/16
                esp_proposals = aes256gcm16-x25519
                start_action = trap        # bring up on first packet
            }
        }

        version = 2
        proposals = aes256-sha384-x25519
        unique = replace
        send_certreq = no
    }
}

# Apply
swanctl --load-all
swanctl --initiate --child site-to-site

# Status
swanctl --list-sas
swanctl --list-conns
```

### Cipher Proposal Syntax

```
aes256gcm16-x25519
└──┬──┘ └─┬┘ └─┬─┘
   │     │    └── DH group (for PFS during rekey)
   │     └──────  ICV length (16 bytes = 128 bits)
   └────────────  Cipher + mode (AES-256-GCM)
```

### `ipsec.conf` (older syntax, still common)

```
conn site-to-site
    keyexchange=ikev2
    auto=start
    left=198.51.100.1
    leftsubnet=10.1.0.0/16
    leftid=@vpn.us.example.com
    leftcert=vpn-gateway.crt
    right=203.0.113.5
    rightsubnet=10.2.0.0/16
    rightid=@vpn.eu.example.com
    ike=aes256-sha384-x25519!
    esp=aes256gcm16!
```

## Remote Access (Road Warrior)

```
Server config:
  Authentication: EAP-MSCHAPv2 + cert
  Pool: hand out 10.99.0.0/24 to clients
  DNS: 10.99.0.1

Clients use built-in OS support:
  Windows: Settings → VPN → IKEv2
  macOS:   System Settings → VPN → IKEv2
  iOS/Android: Strongswan app or native
  Linux: NetworkManager-strongswan or swanctl
```

## Linux Kernel XFRM Framework

The Linux IPsec data plane is `xfrm` (transformer). Userspace (strongSwan, Libreswan) speaks XFRM netlink to install policies and SAs.

```bash
# View SAs
sudo ip xfrm state

# View SPD policies
sudo ip xfrm policy

# Monitor in real time
sudo ip xfrm monitor

# Manually add a transport-mode SA (rarely needed; use IKE)
sudo ip xfrm state add src 198.51.100.1 dst 203.0.113.5 \
  proto esp spi 0x100 reqid 1 mode tunnel \
  aead 'rfc4106(gcm(aes))' 0x... 128
```

## Performance

```
With AES-NI hardware:
  ~5-10 Gbps per core (AES-GCM)

Without hardware:
  Use ChaCha20-Poly1305 instead — much faster in software

CPU bottlenecks:
  - Single SA pinned to one queue → one core
  - Use RSS / multiple SAs for parallelism

MTU overhead:
  Tunnel mode + ESP + NAT-T can eat ~50-80 bytes
  Default MTU on tunnel interfaces: 1438-1450
  Always enable PMTUD or clamp MSS (see mtu_pmtud.md)
```

## Common Configurations

### Site-to-Site VPN

```
HQ (gateway A)                          Branch (gateway B)
 LAN 10.1.0.0/16                         LAN 10.2.0.0/16
       │                                        │
       └──────── IPsec tunnel ─────────────────┘
              (esp, tunnel mode, IKEv2)

Hosts in 10.1.0.0/16 reach 10.2.0.0/16 transparently.
```

### Spoke-to-Hub VPN

Each branch tunnels to HQ; branches reach each other via HQ (or via DMVPN-style dynamic peering).

### Remote Access

```
Mobile user → IKEv2 to corporate gateway
  Assigned 10.99.0.5 from pool
  Routes 10.0.0.0/8 via tunnel
  Internet either tunneled (full) or local (split)
```

### Per-host IPsec (transport mode)

Two specific servers encrypt their direct communication without modifying network topology.

## Debugging

```bash
# strongSwan logs
journalctl -u strongswan-starter -f
tail -f /var/log/charon.log

# Loglevels in swanctl.conf:
filelog {
    charon.log {
        path = /var/log/charon.log
        default = 2
        ike = 3
        cfg = 3
        net = 2
    }
}

# Show SAs and policies
swanctl --list-sas
ip xfrm state
ip xfrm policy

# Tcpdump (IKE + NAT-T + ESP)
sudo tcpdump -i any 'udp port 500 or udp port 4500 or esp'

# Wireshark filters
isakmp
esp
udp.port == 4500
```

### Common Issues

```
"NO_PROPOSAL_CHOSEN"
  → cipher mismatch, check ike/esp proposals on both sides

"INVALID_ID_INFORMATION"
  → identities don't match (leftid/rightid vs cert subject)

Authentication failed (cert)
  → CA chain not loaded, expired cert, clock skew

Tunnel up but no traffic
  → traffic selectors don't match (leftsubnet/rightsubnet),
    or routing isn't pointing traffic at tunnel,
    or NAT mangling source IPs before SPD lookup

NAT-T not detected behind NAT
  → some old gear blocks UDP 500/4500, or doesn't translate ESP at all
```

## Security Best Practices

```
✓ Use IKEv2, not IKEv1
✓ Use AES-GCM (AEAD) over CBC+HMAC
✓ Use ECDSA P-256/384 or RSA-3072+ for certs
✓ Use X25519 or P-256 for DH (avoid MODP < 2048)
✓ Set PFS group on every Child SA (rekey gets new DH)
✓ Reasonable SA lifetimes (e.g., 8 hours / 1 GB)
✓ Strict traffic selectors (avoid 0.0.0.0/0 unless intended)
✓ Reject weak proposals — use `!` in old syntax / strict in new
✓ Cert validation: CA + revocation (CRL/OCSP)
✓ Enable IKE fragmentation if behind small-MTU paths
✗ Don't use PSK with road-warrior (only for fixed gateway pairs)
✗ Don't use aggressive mode (IKEv1)
✗ Don't expose IKE without rate limiting (DoS surface)
```

## IPsec vs WireGuard vs OpenVPN

| Aspect | IPsec | WireGuard | OpenVPN |
|--------|-------|-----------|---------|
| Layer | L3 | L3 | L2/L3 |
| Transport | ESP/AH or UDP-encaps | UDP | UDP or TCP |
| Crypto agility | Many options | Fixed | Many options |
| Cert PKI | yes | no | yes |
| In-kernel | yes (Linux xfrm) | yes (Linux) | no |
| Performance | excellent (HW accel) | excellent | moderate |
| Site-to-site | excellent | works | works |
| Remote access | strong (built-in OS support) | needs apps | yes (clients) |
| NAT traversal | NAT-T | built-in | built-in |
| Complexity | high | very low | medium |
| Compliance (FIPS/CC) | yes | not yet widely | partial |

Rough rule of thumb:
- **WireGuard**: greenfield, simplicity wins
- **IPsec**: vendor interop, compliance, existing infrastructure
- **OpenVPN**: when you need TCP-443 to evade firewalls

## ELI10

IPsec is the Cold War spy radio: it has so many knobs and dials (cipher suites, key exchange groups, modes, authentication methods) that even the spies need a thick manual. Once you set up two compatible radios (gateways), they encrypt all the messages going between two embassies (LANs) — and nobody outside can tell what they're discussing.

There are two main "envelope styles":
- **Transport mode**: encrypt only the letter inside (between two people who both have radios).
- **Tunnel mode**: encrypt the whole sealed envelope including the recipient address (between two embassies, used to forward mail for everyone inside).

The complicated handshake (**IKE**) is two ambassadors agreeing in advance on which radio frequency to use, what code book, and rotating keys regularly. **WireGuard** is the modern alternative — same outcome, but using a single fixed-frequency radio with a one-page instruction sheet.

## Further Resources

- [RFC 4301 - Security Architecture for IP](https://tools.ietf.org/html/rfc4301)
- [RFC 4303 - ESP](https://tools.ietf.org/html/rfc4303)
- [RFC 7296 - IKEv2](https://tools.ietf.org/html/rfc7296)
- [strongSwan documentation](https://docs.strongswan.org/)
- [Libreswan project](https://libreswan.org/)
## Where this connects

- [IP](ip.md) — IPsec operates at Layer 3, adding authentication/encryption directly to IP packets
- [TLS/SSL](tls_ssl.md) — TLS secures individual TCP connections; IPsec secures all traffic between two endpoints
- [Firewalls](firewalls.md) — firewalls must permit UDP 500/4500 (IKE) and ESP (protocol 50) for IPsec
- [UDP](udp.md) — IKEv2 negotiation and NAT-T encapsulation use UDP

- [Linux XFRM Framework](https://wiki.strongswan.org/projects/strongswan/wiki/IPsecLinuxKernel)
- [NIST SP 800-77: Guide to IPsec VPNs](https://csrc.nist.gov/publications/detail/sp/800-77/rev-1/final)
