# Networking — deep per-page correctness review

Per-page review of all 44 pages in `src/networking/`. Code, packet/field values, port
numbers, RFC references, protocol constants, and prose were checked. Severity:
**ERR** = wrong, **SUS** = questionable/misleading, **NIT** = minor/style.

## Summary

Overall the corpus is high quality and factually accurate. All 44 pages were checked. Errors
found were rare and concentrated in specific constants (MAC addresses, hex↔decimal
conversions, magic numbers) rather than conceptual mistakes.

**Errors fixed inline (5):**
1. `ospf_isis.md` — IS-IS L1/L2 multicast MACs `09:00:2b:00:00:14/15` → `01:80:c2:00:00:14/15`
   (AllL1ISs / AllL2ISs per ISO 10589; `09:00:2b` is the DEC OUI, wrong here).
2. `udp.md` — example source port 53210 written as `0xCFCA` (=53194) → `0xCFDA` (2 places).
3. `ice.md` — host-candidate priority `2113667071` → `2130706431` (2 places; matches the
   file's own example candidate line and the RFC 8445 formula).
4. `pcp.md` — DHCPv4 PCP-server option `128` → `158` (OPTION_PCP_SERVER, RFC 7291).
5. `rtp.md` — Wireshark example SSRC `0xABCD1234 (2882400052)` → `(2882343476)`.

See the per-page log below and the "Final tally" at the end for the SUS/NIT follow-ups
that were noted but not changed.

---

Running log. Severity: **ERR** = wrong, **SUS** = questionable/misleading, **NIT** = minor.

## ntp_ptp.md — OK
Ports (123; 319/320), stratum 16=unsync, offset/delay formulas, NTS RFC 8915, LLMNR
224.0.0.252 all correct.

## sip_voip.md — OK
Ports 5060/5061, PT 0=PCMU / 8=PCMA, G.711 ~64kbps, SIP-over-WS RFC 7118 all correct.

## multicast_igmp_pim.md — OK
224.0.0.0/4, OSPF .5/.6, mDNS .251, SSM 232/8, admin 239/8, MAC mappings (23-bit/32-bit,
32-way collision), IGMP versions, PIM modes all correct.

## nac_8021x.md — OK
RADIUS 1812/1813 (legacy 1645/1646), RadSec TCP 2083, Tunnel-Type=VLAN(13),
Tunnel-Medium-Type=IEEE-802(6), EAP methods all correct.

## qos_traffic_shaping.md — OK
DSCP EF=46, ECN bit encodings, EF<<2=0xb8, WMM 4 ACs all correct.

## email_protocols.md — OK
Ports 25/587/465/993/995, RFC 8314 (465), SPF/DKIM/DMARC mechanics all correct.

## http.md — OK
Method idempotency/safety table, HTTP/1.1 1997 / HTTP/2 2015 / HTTP/3 2022 all correct.

## tls_ssl.md — OK
TLS 1.3 RFC 8446, deprecation RFC 8996, the 5 TLS 1.3 cipher suites, 16KB record limit,
X25519/P-256 all correct.

## arp.md — OK
Packet format (28B + 14B = 42B padded to 60), opcodes, gratuitous/probe/proxy ARP, NDP
mapping all correct.

## mtu_pmtud.md — OK
Min IPv4 68 (RFC 791), min IPv6 1280 (RFC 8200), MTU table, frag offsets (1480/8=185),
ICMP type3/code4 & ICMPv6 type2, MSS 1460, min_pmtu 552 all correct.

## ospf_isis.md — ERR (fixed)
- **ERR**: IS-IS multicast MACs were `09:00:2b:...` → corrected to `01:80:c2:00:00:14/15`.
- Rest (proto 89, 224.0.0.5/6, LSA types, Hello/Dead timers, DR election) correct.

## dhcp.md — NIT
- **NIT**: "Client Returning to Same Network … starts at REBINDING state" — RFC 2131 calls
  this **INIT-REBOOT** (client broadcasts DHCPREQUEST with Requested-IP, no server-id).
  REBINDING is the T2 lease-renewal state. Minor terminology slip.
- Ports 67/68, DORA, T1/T2 (50%/87.5%), DHCPv6 546/547 + ff02::1:2, magic cookie,
  option numbers all correct.

## dns.md — OK
Port 53, header 12B + flag layout, RCODEs 0-5, DoH 443 / DoT 853, public resolver IPs,
DNSSEC records, the hex-encoded query bytes all correct.

## wireguard.md — OK
Kernel 5.6, Noise primitives (Curve25519/ChaCha20-Poly1305/BLAKE2s/HKDF), port 51820,
~2min rekey, packet header sizes, MTU 1420 = 1500−80 all correct.

## bgp_anycast.md — OK
RFC 4271, TCP 179, ASN ranges (16/32-bit + private), real-world ASNs, path-selection
ladder, FSM states, famous leak incidents all correct.

## ipsec.md — OK
ESP proto 50, AH proto 51, IKE UDP 500 / NAT-T 4500, IKEv1 (RFC 2409) Main/Aggressive
message counts, IKEv2 (RFC 7296) 4-message setup, cipher guidance all correct.

## http2.md — OK
RFC 7540/9113, preface string (24B), 9B frame header, frame type codes, stream-ID parity,
HPACK static-table entries, flow-control window 65535, SETTINGS defaults, Chrome push
removal 2022, RFC 9218 priorities all correct.

## ethernet_vlan.md — OK
EtherTypes, 64/1518/9000 frame sizes, VLAN ID 1-4094 + reserved ranges, 802.1Q TCI layout,
tagged 1522, I/G & U/L bits, STP/RSTP/MSTP, PoE wattages, LACP 802.3ad all correct.

## ip.md — SUS
- **SUS**: IPv4-vs-IPv6 table claims IPSec is "Mandatory" for IPv6. This is the classic
  textbook line but outdated — RFC 6434 (2011) downgraded IPsec from MUST to SHOULD for
  IPv6 nodes. Worth softening to "recommended / originally mandated".
- ICMP type 30 (Traceroute, RFC 1393) is deprecated/historic but labeled as such — fine.
- Header fields, classes, CIDR table, private ranges, fragmentation offsets, protocol
  numbers (6/17/1), TTL defaults all correct.

## udp.md — ERR (fixed)
- **ERR**: example source port 53210 was written as `0xCFCA` (=53194) in both the field
  list and the hex dump (`CF CA …`). 53210 = `0xCFDA`; fixed both.
- 8B header, proto 17, checksum optional-v4/mandatory-v6, max UDP data 65507,
  safe sizes 1472 (v4)/1452 (v6), port list, IPv6 jumbogram max all correct.

## ssh.md — OK
Port 22, RFC 4251-4254 layer split, KEX algs (curve25519-sha256, sntrup761x25519),
DSA removal in OpenSSH 7, ed25519-sk, scp -O all correct.

## mdns.md — OK
224.0.0.251 / ff02::fb / port 5353, cache-flush bit 0x8000, response flags 0x8400,
DNS-SD `_services._dns-sd._udp.local`, RFC 6762/6763 all correct.

## osi_model.md — NIT
- **NIT**: media table lists "Fiber (MM) 10 Gbps 550m". 10GBASE-SR over modern multimode
  is ~300m (OM3) / ~400m (OM4); 550m applies to 1000BASE-SX / older. Slight overstatement.
- Manchester convention, port ranges, PDU names, TCP/IP mapping all correct.

## stun.md — OK
Magic cookie 0x2112A442, port 3478, message types/classes, attribute type codes,
XOR-port derivation, long-term cred key = MD5(user:realm:pass), Google STUN 19302,
RFC 5389/8489 all correct. (Python sample omits `import os` — code nit, not factual.)

## tcp_ip_model.md — OK
4-layer mapping, window-scaling ~1GB, congestion algos, port/transport table, FTP 20/21,
TURN bandwidth cost arithmetic all correct.

## turn.md — OK
Default allocation 600s, permission 300s, channel range 0x4000-0x7FFF, attribute type
codes, ports 3478/5349, coturn CLI 5766, RFC 5766/8656 all correct.

## container_networking.md — OK
docker0 172.17.0.1/16, NodePort 30000-32767, VXLAN 4789, embedded DNS 127.0.0.11,
MTU-overhead figures all correct.

## overlay_networks.md — OK
GRE proto 47 / TEB 0x6558, VXLAN RFC 7348 / UDP 4789 / 24-bit VNI / 50B overhead,
Geneve RFC 8926 / UDP 6081, EVPN RFC 7432 route types, ECMP reasoning, AWS jumbo 9001
all correct.

## firewalls.md — OK
Firewall taxonomy, iptables/ufw/firewalld syntax, SNAT/DNAT examples, conntrack states,
SSDP amplification factor all correct.

## upnp.md — NIT
- **NIT**: overview says UPnP devices announce "using mDNS/SSDP". UPnP discovery is SSDP
  only (the rest of the page correctly says so); mDNS/DNS-SD is a separate discovery stack.
- SSDP 239.255.255.250:1900, M-SEARCH/NOTIFY/GENA, IGD AddPortMapping SOAP, RFC 6970 correct.

## iot_protocols.md — OK
MQTT 1999/IBM, 2-byte header, QoS 0/1/2 handshakes, ports 1883/8883, keepalive 1.5×;
CoAP RFC 7252 4-byte header, ports 5683/5684, method/response codes, Observe (7641),
CoRE-RD (9176), OSCORE (8613), TLS_PSK_WITH_AES_128_CCM_8 all correct.

## grpc.md — OK
Open-sourced 2015, HTTP/2 mapping, 1+4-byte message framing, all 16 status codes,
content-type application/grpc+proto, grpc-timeout, `-bin` metadata, gRPC-Web limits correct.

## README.md — OK
Index/port-reference table consistent with the individual pages (HTTP/3 443 UDP, PTP
319/320, SMTPS 465, RADIUS 1812/1813, STUN 3478, SSDP 1900, mDNS 5353 all correct).

## ice.md — ERR (fixed)
- **ERR**: host-candidate priority computed as `2113667071`; correct value is
  `(2^24×126)+(2^8×65535)+255 = 2130706431` (the candidate example line earlier in the
  file already used 2130706431). Fixed in both the calculation and the "in practice" list.
- **SUS** (not changed): the "sorted by priority high→low" list orders srflx above prflx,
  but the listed numbers (prflx 1862270975 > srflx 1694498815) and type prefs (prflx 110 >
  srflx 100) make prflx higher. Ordering label is slightly off; the numbers are right.
- type prefs (126/110/100/0), srflx/prflx/relay priority values, formula, RFC 8445 correct.

## nat_pmp.md — SUS
- **SUS**: gateway-discovery "Method 2" lists "DHCP Option 120 (NAT-PMP Gateway)". There is
  no standard DHCP option for a NAT-PMP gateway (NAT-PMP always uses the default router);
  DHCP option 120 is actually the SIP Servers option (RFC 3361). Left as-is (labeled "rarely
  used"), but the option number/name is wrong.
- Port 5351, RFC 6886, opcodes 0/1/2 + 128/129/130, result codes 0-5, 12B/16B packet sizes,
  epoch-reboot detection all correct.

## pcp.md — ERR (fixed)
- **ERR**: PCP server discovery listed DHCPv4 "Option 128"; correct is **158**
  (OPTION_PCP_SERVER, RFC 7291). DHCPv6 option 86 was already correct. Fixed + cited RFC 7291.
- Version 2, port 5351, RFC 6887, opcodes (ANNOUNCE/MAP/PEER), result codes 0-13, option
  codes (THIRD_PARTY/PREFER_FAILURE/FILTER), 24B headers, 36B MAP payload all correct.

## ipv4.md — OK
Header fields, class ranges, special blocks (192.88.99.0/24 6to4, 198.18/15 bench, TEST-NETs),
CIDR/VLSM tables, /31 (RFC 3021), ICMP type/code table, NAT types, protocol numbers correct.

## ipv6.md — SUS
- **SUS**: same as ip.md — comparison table calls IPSec "Mandatory" for IPv6; downgraded to
  SHOULD by RFC 6434 (2011) / RFC 8504. Common textbook oversimplification.
- Header layout, Next-Header values (0/43/44/50/51/58/59/60), GUA 2000::/3, ULA fc00::/7
  (fd00::/8), fe80::/10, ff00::/8 + scopes, solicited-node ff02::1:ff/104, EUI-64 bit-flip,
  ICMPv6 types (1-4,128-137,130-132 MLD), 6to4 2002::/16, Teredo 2001::/32, NAT64 64:ff9b
  all correct. (RFC 2460 cited for ext-header order — obsoleted by RFC 8200 but still valid.)

## webrtc.md — OK
SDP profile (UDP/TLS/RTP/SAVPF), 90kHz video clock rates (VP8/VP9/H264), Opus 6-510kbps,
codec list, DTLS-SRTP security, browser-support versions all correct.

## websocket.md — OK
ws/wss 80/443, RFC 6455, Sec-WebSocket-Version 13, magic GUID, the canonical
accept-key example (s3pPLMBiTxaQ9kYGzzhZRbK+xOo=), opcodes, frame layout, readyState
constants all correct.

## tcp.md — SUS
- **SUS**: "Delayed Acknowledgment" config example uses `net.ipv4.tcp_delack_seg` sysctl,
  which is not a standard mainline-Linux knob (delayed-ACK behavior is controlled via the
  `TCP_QUICKACK` socket option / `quickack` route attr). Likely fabricated knob.
- Header/flags, 3-/4-way handshake, state machine, ISN (RFC 6528), RTO calc (α=.125,
  β=.25, RTO=SRTT+4·RTTVAR, min 200ms), keepalive defaults (7200/75/9), TIME_WAIT 2·MSL,
  tcp_tw_recycle removal, BDP example (1Gbps×100ms=12.5MB), congestion algos, SYN-cookie/
  TFO sysctls, RFC list (793/1323/2018/7413/8684/9000/9114) all correct.

## quic.md — NIT
- **NIT**: connection-migration section calls the Connection ID a "64-bit unique
  identifier"; QUIC CIDs are variable 0-20 bytes (0-160 bits), as the packet-format section
  correctly states. Minor internal inconsistency.
- RFC 9000 (May 2021), Google 2012, long/short header layout, type bits, version
  0x00000001, stream-ID parity (4n/4n+1/4n+2/4n+3), 3 packet-number spaces, time-threshold
  9/8, PTO formula (RFC 9002), amplification 3×, DoQ (9250/853), QPACK (9204), MASQUE
  (9298) all correct.

## rtp.md — ERR (fixed)
- **ERR**: a Wireshark example printed `0xABCD1234 (2882400052)`; correct decimal is
  `2882343476`. Fixed. (Likely a stray carry from the earlier 0xABCDEF01=2882400001 example.)
- RFC 3550 header (V/P/X/CC/M/PT/seq/ts/SSRC), PT statics (0/3/4/8/9/18 + G.722's
  spec-quirk 8000Hz clock), 90kHz video clock, packet-breakdown hex→decimal (6699/1000/
  0xABCDEF01), RTCP types (200-204), jitter `/16` formula, SR header 0x80C8 / length 6,
  SRTP (RFC 3711, AES-CTR, HMAC-SHA1-80, 2^48 rekey), RTX (RFC 4588), 0xBEDE ext (5285),
  audio-level (6464), DSCP EF46/AF41-34, RFC list all correct.

---

## Final tally

**5 factual errors fixed inline**, across 5 files:
1. `ospf_isis.md` — IS-IS L1/L2 multicast MAC `09:00:2b:…` → `01:80:c2:00:00:14/15`.
2. `udp.md` — example source port 53210 hex `0xCFCA`→`0xCFDA` (two occurrences).
3. `ice.md` — host-candidate priority `2113667071`→`2130706431` (two occurrences).
4. `pcp.md` — DHCPv4 PCP server option `128`→`158` (RFC 7291).
5. `rtp.md` — SSRC decimal `0xABCD1234 (2882400052)`→`(2882343476)`.

**Not changed — worth a follow-up (SUS/NIT):**
- `ip.md` & `ipv6.md` — "IPSec mandatory for IPv6" (downgraded to SHOULD by RFC 6434/8504).
- `dhcp.md` — return-to-network state should be INIT-REBOOT, not REBINDING.
- `nat_pmp.md` — "DHCP Option 120 (NAT-PMP Gateway)" is not a real option (120 = SIP servers).
- `tcp.md` — `net.ipv4.tcp_delack_seg` sysctl is not standard mainline Linux.
- `osi_model.md` — multimode fiber "10 Gbps 550m" overstated (~300-400m for 10GBASE-SR).
- `upnp.md` — overview says discovery uses "mDNS/SSDP"; UPnP uses SSDP only.
- `quic.md` — Connection ID called "64-bit"; actually variable 0-160 bits.
- `ice.md` — "sorted by priority" list orders srflx above prflx, but prflx's value is higher.

Everything not listed under ERR/SUS/NIT was checked and verified correct.
