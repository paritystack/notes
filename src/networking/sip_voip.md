# SIP & VoIP

## Overview

Voice over IP (VoIP) splits a phone call into two planes: **signaling** (find the other
party, ring them, negotiate codecs, hang up) and **media** (the actual audio/video). SIP is
the dominant signaling protocol; the media itself rides [RTP](rtp.md).

```
Signaling:  SIP  (Session Initiation Protocol) — set up / tear down / modify sessions
Description: SDP (Session Description Protocol) — codecs, ports, IPs (carried inside SIP)
Media:      RTP / RTCP (see rtp.md) — the audio/video packets and their stats
Security:   SIPS (SIP over TLS) for signaling; SRTP / DTLS-SRTP for media

SIP only sets up the call; it does NOT carry the audio. RTP does, on separate ports.
```

SIP looks and feels like [HTTP](http.md): text-based, request/response, header-rich,
URI-addressed (`sip:alice@example.com`). It's the same family of design as the web.
[WebRTC](webrtc.md) is the browser cousin — different signaling, but the same RTP/SRTP media
and the same [ICE](ice.md)/[STUN](stun.md)/[TURN](turn.md) NAT traversal underneath.

## SIP components

```
  UA (User Agent)   the phone / softphone / app (acts as UAC when calling, UAS when called)
  Registrar         records where a user currently is (REGISTER → location service)
  Proxy server      routes SIP requests toward the callee (like a router for signaling)
  Redirect server   tells the caller a better address to try
  B2BUA             Back-to-Back UA: sits in the middle of BOTH call legs (PBXes, SBCs)
  SBC               Session Border Controller: security/NAT/transcoding gateway at the edge
```

## How a call works

### Registration (where are you?)

```
  Phone ── REGISTER sip:example.com  Contact: <sip:alice@192.0.2.5:5060> ──► Registrar
        ◄── 200 OK ──────────────────────────────────────────────────────
  Now the registrar knows alice is reachable at 192.0.2.5:5060 (for some expiry).
```

### Call setup — the INVITE / 200 / ACK three-way

```
  Alice (UAC)                  Proxy                    Bob (UAS)
     | INVITE (SDP offer) ──►   | INVITE ──────────────► |
     | ◄── 100 Trying           | ◄── 100 Trying         |
     | ◄── 180 Ringing ─────────| ◄── 180 Ringing ────── |  (Bob's phone rings)
     | ◄── 200 OK (SDP answer) ─| ◄── 200 OK ─────────── |  (Bob answers)
     | ACK ─────────────────────────────────────────────► |  (often goes direct)
     |                                                     |
     |══════════ RTP media flows DIRECTLY, peer-to-peer ══════════|  (NOT through proxy)
     |                                                     |
     | BYE ─────────────────────────────────────────────► |  (either side hangs up)
     | ◄── 200 OK                                          |
```

Note the pattern: **signaling** may traverse proxies, but **media (RTP)** flows directly
between endpoints whenever NAT allows — that's why VoIP needs NAT traversal.

### Response codes (HTTP-like)

```
1xx  provisional   100 Trying, 180 Ringing, 183 Session Progress
2xx  success       200 OK
3xx  redirect      302 Moved Temporarily
4xx  client error  401/407 auth required, 404 Not Found, 486 Busy Here, 408 Timeout
5xx  server error  500, 503 Service Unavailable
6xx  global        603 Decline
```

## SDP — negotiating the media

The INVITE carries an SDP **offer**; the 200 OK carries the **answer**. They agree on
codecs, RTP ports, and direction.

```
v=0
o=alice 271828 271828 IN IP4 192.0.2.5
s=-
c=IN IP4 192.0.2.5                 ← where to send media
t=0 0
m=audio 49170 RTP/AVP 0 8 96       ← audio, RTP port 49170, payload types offered
a=rtpmap:0 PCMU/8000               ← PT 0 = G.711 µ-law
a=rtpmap:8 PCMA/8000               ← PT 8 = G.711 a-law
a=rtpmap:96 opus/48000/2           ← PT 96 = Opus (dynamic)
a=sendrecv                          ← media direction
```

Common codecs: **G.711** (PCMU/PCMA, uncompressed-ish, ~64 kbps, universal), **Opus**
(modern, adaptive, low-latency — also WebRTC's default), **G.722** (HD voice), G.729
(low-bandwidth legacy).

## NAT traversal — the hard part

SDP advertises private IPs that are useless across NAT, and RTP uses dynamic UDP ports that
firewalls block. Solutions, mostly shared with [WebRTC](webrtc.md):

```
SBC / media relay   enterprise edge device rewrites SDP and relays RTP (B2BUA)
STUN (stun.md)      endpoint discovers its public IP:port, rewrites SDP
TURN (turn.md)      relay media when direct paths fail (symmetric NAT)
ICE  (ice.md)       tries all candidate paths, picks one that works
SIP ALG             router "helper" that rewrites SIP — notoriously buggy, usually disable it
Keep-alives         (PersistentKeepalive-style) hold NAT bindings open
```

## Security

```
Signaling:
  SIP digest auth (401/407, MD5 challenge) — weak; protects REGISTER/INVITE
  SIPS = SIP over TLS (see tls_ssl.md), port 5061 — encrypts signaling, prevents tampering

Media:
  SRTP        encrypts/authenticates RTP (see rtp.md) with AES + HMAC
  Keying:
    SDES      keys in the SDP — only safe if signaling is over TLS (else keys leak!)
    DTLS-SRTP key exchange via DTLS handshake on the media path (WebRTC's choice; best)
    ZRTP      in-media DH key agreement, no PKI

Threats: toll fraud (compromised PBX dialing premium numbers), SPIT (spam calls),
         eavesdropping on unencrypted RTP, registration hijacking, INVITE floods (DoS).
```

## Transport & ports

```
SIP:  UDP or TCP 5060 (cleartext), TLS 5061 (SIPS). UDP is traditional; TCP/TLS for large
      messages or security. SIP can also run over WebSocket (RFC 7118) for browser clients.
RTP:  dynamic even UDP ports; RTCP on the next odd port (see rtp.md).
```

## Tooling

```bash
# Capture and follow SIP signaling
tcpdump -i any -n port 5060 -A
# Wireshark: filter `sip`, then Telephony → VoIP Calls to replay/inspect a call;
#   Telephony → RTP → RTP Streams to analyze jitter/loss of the media.

# Command-line softphones / testers
#   pjsua (pjproject), baresip, linphone-cli
#   sipp  — SIP load testing / scripted call flows
sipp -sn uac 203.0.113.10        # run a basic UAC scenario against a server
```

## Related

- [RTP / RTCP / SRTP](rtp.md) — the media plane SIP sets up; jitter buffers, codecs, stats
- [WebRTC](webrtc.md) — browser real-time comms: same media/NAT stack, different signaling
- [ICE](ice.md) / [STUN](stun.md) / [TURN](turn.md) — NAT traversal for the RTP media
- [TLS/SSL](tls_ssl.md) — SIPS signaling and DTLS-SRTP / SDES keying
- [QoS](qos_traffic_shaping.md) — voice is the canonical EF-marked, jitter-sensitive traffic
- [HTTP](http.md) — SIP borrows its text/request-response/status-code design
