# HTTP/2

## Overview

HTTP/2 (RFC 7540, 2015; updated by RFC 9113) is a major revision of [HTTP](http.md) that retains the same semantics (methods, headers, status codes) but completely changes the **wire format**. It replaces text framing with a binary protocol that multiplexes many streams over a single [TCP](tcp.md) connection, eliminating head-of-line blocking at the HTTP layer. HTTP/3 builds on these ideas but runs over [QUIC](quic.md) instead of TCP. [gRPC](grpc.md) uses HTTP/2 as its transport; [TLS](tls_ssl.md) is required for browser-facing HTTP/2.

## Why HTTP/2?

### HTTP/1.1 Pain Points

```
1. One request per connection (HTTP/1.0)
   Fixed by keep-alive in 1.1, still serialized

2. Head-of-line blocking
   Request 2 waits for response 1 to finish

3. Multiple connections required (~6/origin)
   Wasteful, doesn't help small assets

4. Verbose headers, no compression
   Same Cookie/User-Agent sent in every request

5. Text-based parsing
   Slower, ambiguous (request smuggling attacks)

6. No prioritization
   Can't tell server "this CSS first, then images"
```

### HTTP/2 Solutions

```
✓ Binary framing
✓ Multiplexing (many streams per connection)
✓ Header compression (HPACK)
✓ Server push (deprecated, but it tried)
✓ Stream priorities
✓ Flow control per stream
```

## Connection Setup

### Negotiation

HTTP/2 cannot be used blindly — both sides must agree first.

#### Over TLS (h2)

Almost all real HTTP/2 traffic uses TLS with **ALPN** (Application-Layer Protocol Negotiation):

```
Client TLS ClientHello:
  ALPN extension: ["h2", "http/1.1"]

Server TLS ServerHello:
  ALPN extension: "h2"

→ TLS handshake completes, both speak HTTP/2 immediately
```

#### Over Plain TCP (h2c) — rare

Browsers don't support h2c. Used internally:

```
Client → Server:
  GET / HTTP/1.1
  Connection: Upgrade, HTTP2-Settings
  Upgrade: h2c
  HTTP2-Settings: <base64>

Server → Client:
  HTTP/1.1 101 Switching Protocols
  Connection: Upgrade
  Upgrade: h2c
```

Or with prior knowledge — client just sends the connection preface directly.

### Connection Preface

The first thing the client sends after negotiation:

```
PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n
(24 bytes, ASCII)
```

This magic string ensures the peer is actually doing HTTP/2 (an HTTP/1.1 server would error out).

After the preface, both sides exchange `SETTINGS` frames.

## Frame Format

Everything in HTTP/2 is a **frame**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Length (24 bits)              |  Type  | Flags|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|R|                 Stream Identifier (31)                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Frame Payload (variable)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- 9-byte fixed header
- Length: payload size (default max 16 KiB, settable to 16 MiB)
- Type: which frame (DATA, HEADERS, SETTINGS, etc.)
- Stream Identifier: which stream this frame belongs to (0 = connection-level)

### Frame Types

| Type | Code | Purpose |
|------|------|---------|
| **DATA** | 0x0 | Request/response body |
| **HEADERS** | 0x1 | Request/response headers (HPACK-compressed) |
| **PRIORITY** | 0x2 | Stream priority hints (deprecated) |
| **RST_STREAM** | 0x3 | Cancel a stream |
| **SETTINGS** | 0x4 | Connection params (window size, max streams) |
| **PUSH_PROMISE** | 0x5 | Server push (deprecated in browsers) |
| **PING** | 0x6 | Keepalive / latency measurement |
| **GOAWAY** | 0x7 | Graceful shutdown |
| **WINDOW_UPDATE** | 0x8 | Flow control credit |
| **CONTINUATION** | 0x9 | More HEADERS (if exceeded frame size) |

## Streams and Multiplexing

A **stream** is a logical bidirectional sequence of frames sharing a Stream ID. Multiple streams ride one TCP connection in parallel.

```
TCP connection
   │
   ├── Stream 1: GET /index.html         → 200 OK + HTML body
   ├── Stream 3: GET /style.css          → 200 OK + CSS
   ├── Stream 5: GET /app.js             → 200 OK + JS
   ├── Stream 7: GET /logo.png           → 200 OK + image
   └── Stream 9: GET /api/data           → 200 OK + JSON

All happening concurrently — frames interleaved on the wire.
```

### Stream ID Rules

```
- Client-initiated: ODD IDs (1, 3, 5, 7, …)
- Server-initiated (push): EVEN IDs (2, 4, …)
- Stream 0: reserved for connection-level frames (SETTINGS, PING, GOAWAY)
- IDs monotonically increase; once closed, never reused
- Max: 2^31 - 1 streams per connection (then must reconnect)
```

### Stream Lifecycle

```
                            ┌──────┐
              send PP ─────►│ idle │◄───── recv PP
                            └──┬───┘
                               │  send H / recv H
                               ▼
                      ┌──────────────┐
                      │     open     │
                      └──┬────────┬──┘
                  send ES│        │recv ES
                         ▼        ▼
                ┌─────────┐    ┌──────────┐
                │ half-   │    │ half-    │
                │ closed  │    │ closed   │
                │ (local) │    │ (remote) │
                └────┬────┘    └────┬─────┘
                     │ recv ES      │ send ES
                     ▼              ▼
                              ┌──────────┐
                              │  closed  │
                              └──────────┘

H  = HEADERS frame
ES = END_STREAM flag
PP = PUSH_PROMISE
```

A stream is **closed** when both sides have sent END_STREAM. A `RST_STREAM` immediately tears it down.

## HPACK: Header Compression

HTTP headers repeat across requests (Cookie, User-Agent, Accept-*). HPACK (RFC 7541) compresses them using:

### 1. Static Table

Predefined table of common header name+value pairs:

```
Index | Name              | Value
  1   | :authority        |
  2   | :method           | GET
  3   | :method           | POST
  4   | :path             | /
  5   | :path             | /index.html
  6   | :scheme           | http
  7   | :scheme           | https
  8   | :status           | 200
  ...
 17   | accept-encoding   | gzip, deflate
 ...
```

Reference by index → 1 byte instead of full header.

### 2. Dynamic Table

Both endpoints maintain a per-connection table of recently seen headers. Index them on subsequent requests:

```
Request 1: user-agent: Mozilla/5.0 (X11; Linux x86_64) ...
  → New entry added to dynamic table, indexed (e.g., 62)

Request 2: simply reference index 62
```

### 3. Huffman Coding

Static Huffman table compresses string literals (used when not indexed).

### HPACK Tradeoff: CRIME-Style Attacks

Compression of secrets + attacker-controlled data leaks via size oracles. HPACK has mitigations (never-indexed headers like `cookie`, `authorization`).

## Flow Control

To prevent fast senders from overwhelming slow receivers, HTTP/2 has **per-stream** and **per-connection** flow control.

```
Each side advertises a receive window (initially 65535 bytes).
DATA frames consume window.
Receiver sends WINDOW_UPDATE to grant more credit.

Two levels:
  - Per-stream window
  - Per-connection window

Both must have credit for data to flow.
```

```
Initial state:
  Connection window: 65535
  Stream 3 window:   65535

Server sends 65535 bytes on stream 3 → all windows exhausted
Server must STOP and wait for WINDOW_UPDATE

Client sends:
  WINDOW_UPDATE(stream=3, increment=65535)
  WINDOW_UPDATE(stream=0, increment=65535)   # connection-level

Server can resume.
```

Bad flow control tuning is a common cause of HTTP/2 throughput problems — defaults are very small for high-latency links.

## SETTINGS Frame

Connection-level parameters exchanged at startup and updateable mid-connection:

```
SETTINGS_HEADER_TABLE_SIZE        (default 4096)
SETTINGS_ENABLE_PUSH              (1)
SETTINGS_MAX_CONCURRENT_STREAMS   (no default, often 100)
SETTINGS_INITIAL_WINDOW_SIZE      (65535)
SETTINGS_MAX_FRAME_SIZE           (16384)
SETTINGS_MAX_HEADER_LIST_SIZE     (no limit)
```

Settings take effect after the peer ACKs with an empty SETTINGS frame.

## Server Push (Deprecated)

Server preemptively sends responses for resources it knows the client will need:

```
Client → GET /index.html
Server → PUSH_PROMISE: I'm going to send you /style.css and /app.js
Server → HEADERS + DATA for /index.html
Server → HEADERS + DATA for /style.css
Server → HEADERS + DATA for /app.js
```

In theory: eliminates round trips for known dependencies.
In practice: cache mismatches, complexity, mixed evidence on benefits.

**Chrome removed push support in 2022.** Other browsers followed. Use HTTP early hints (103) instead:

```
Server → 103 Early Hints
         Link: </style.css>; rel=preload; as=style
Server → 200 OK
         <html>...</html>
```

## Stream Prioritization

Original HTTP/2 had a complex dependency tree (parent stream, weight 1-256, exclusive bit). It was buggy and inconsistently implemented.

**Replaced by RFC 9218 (Extensible Prioritization Scheme)** — a simple header:

```
Priority: u=1, i

u = urgency (0=highest, 7=lowest, default 3)
i = incremental delivery (true if streaming, e.g., video)
```

Server uses this to schedule frames intelligently.

## Tools & Inspection

```bash
# curl
curl --http2 -v https://example.com
curl --http2 -I https://example.com

# nghttp client
nghttp -v https://example.com
nghttp -nv https://example.com   # show frames

# Check ALPN
openssl s_client -connect example.com:443 -alpn h2 -tls1_2
# → look for "ALPN protocol: h2"

# Server check
nghttpd -v 8080 server.key server.crt    # h2 server
nghttpd --no-tls 8080                    # h2c server

# Wireshark filter
http2
http2.streamid == 3
http2.type == 1                  # HEADERS frames
http2.type == 0                  # DATA frames
```

### Example curl --http2 -v Output

```
* ALPN, server accepted to use h2
* Using HTTP2, server supports multiplexing
* Connection state changed (HTTP/2 confirmed)
* Copying HTTP/2 data in stream buffer to connection buffer
* Using Stream ID: 1
> GET / HTTP/2
> Host: example.com
> user-agent: curl/8.5.0
> accept: */*
>
< HTTP/2 200
< content-type: text/html
< content-length: 1256
```

## Server Configuration Examples

### Nginx

```
server {
    listen 443 ssl http2;
    ssl_certificate     /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    http2_max_concurrent_streams 128;
    # http2_push_preload on;  # deprecated, browsers ignore
}
```

### Caddy

HTTP/2 is on by default with HTTPS:

```
example.com {
    reverse_proxy localhost:8080
}
```

### Go (net/http)

`net/http` server speaks HTTP/2 automatically over TLS:

```go
http.ListenAndServeTLS(":443", "cert.pem", "key.pem", handler)
```

For h2c (cleartext, e.g., behind a proxy):

```go
h2s := &http2.Server{}
h := h2c.NewHandler(handler, h2s)
http.ListenAndServe(":8080", h)
```

## HTTP/2 vs HTTP/1.1 vs HTTP/3

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---------|----------|--------|--------|
| Transport | TCP | TCP | **QUIC (UDP)** |
| Framing | Text | Binary frames | Binary frames |
| Multiplexing | None (1 req/conn) | Per-connection | Per-stream (no HOL block) |
| Header compression | None | HPACK | QPACK |
| Server push | No | Yes (deprecated) | Optional |
| Connection setup | 1+ RTT | 1+ RTT TLS | 0-1 RTT |
| HOL blocking | App-level | TCP-level (still!) | None |
| ALPN identifier | `http/1.1` | `h2` / `h2c` | `h3` |
| Encryption | Optional | Practically required | Mandatory (QUIC requires TLS 1.3) |

### TCP Head-of-Line: HTTP/2's Achilles Heel

Multiplexing in HTTP/2 eliminates **HTTP**-level HOL blocking, but TCP's strict in-order delivery means one lost packet stalls all streams. HTTP/3 fixes this by moving to QUIC (which has independent streams at the transport layer).

```
HTTP/1.1: Streams blocked at HTTP layer
HTTP/2:   Streams parallel at HTTP layer, blocked at TCP layer
HTTP/3:   Streams parallel at both layers (QUIC)
```

## Pseudo-Headers

HTTP/2 replaces the HTTP/1.1 request line and status line with pseudo-headers (prefix `:`):

### Request

```
:method     GET
:scheme     https
:authority  example.com:443
:path       /api/users?id=1
```

(No more `Host:`; `:authority` replaces it.)

### Response

```
:status     200
```

Pseudo-headers must appear before regular headers in HEADERS frames.

## Common Gotchas

### Connection Coalescing

Browsers reuse one HTTP/2 connection for multiple hostnames if their certs cover both and they resolve to the same IP. Saves connections but can break sharding strategies from HTTP/1.

### Domain Sharding Counterproductive

HTTP/1 sites split assets across `a.cdn.com`, `b.cdn.com` to bypass 6-connection limits. With HTTP/2, this **hurts** — you defeat multiplexing. Consolidate origins.

### Long-running Streams

If a stream blocks waiting on the app, it can pin connection-level flow control. Watch out for gRPC servers that never close streams.

### Misconfigured Reverse Proxies

```
Browser ──h2──► Nginx ──http/1.1──► Backend
```

Backend can't use HTTP/2 features. For full benefit, enable HTTP/2 to backend (Nginx `grpc_pass` or `proxy_http_version 1.1` won't do it; use `grpc_pass` for gRPC, or upstream h2c).

### HTTP/2 with Plain TCP (h2c)

Browsers refuse h2c — they require TLS+ALPN for HTTP/2. Only useful between trusted services (e.g., behind a load balancer).

## When HTTP/2 Hurts

```
- Tiny mobile networks with high loss → TCP HOL kills throughput
  → use HTTP/3
- CPU-constrained edge devices → HPACK/TLS overhead
- Long-lived bulk transfers (single large file) → no benefit
- WebSocket replacement attempts → use real WebSocket or HTTP/3
```

## ELI10

HTTP/1.1 is like a one-lane road to a restaurant: cars (requests) take turns, one at a time. If the first car stalls, everyone waits.

HTTP/2 is the same restaurant with a multi-lane road through the same gate (TCP connection). Many cars can drive in parallel, and they all share one entrance — saves the cost of opening multiple gates.

But: it's still **one road**. If a tree falls anywhere on it (a TCP packet is lost), all the lanes have to stop until it's cleared. HTTP/3 builds a separate magical road for each car (QUIC streams).

**HPACK** is the maître d' who remembers your usual order so you don't have to recite it every time — just say "the usual."

**Server push** was the waiter bringing you water before you asked. Nice idea, but they kept bringing water you already had, so the restaurant stopped doing it.

## Further Resources

- [RFC 9113 - HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html)
- [RFC 7541 - HPACK](https://tools.ietf.org/html/rfc7541)
- [RFC 9218 - Extensible Priorities](https://www.rfc-editor.org/rfc/rfc9218.html)
- [HTTP/2 Explained (free book) by Daniel Stenberg](https://daniel.haxx.se/http2/)
## Where this connects

- [HTTP](http.md) — the HTTP/1.1 semantics HTTP/2 preserves while changing the wire format
- [QUIC](quic.md) — HTTP/3 takes HTTP/2 semantics and replaces TCP with QUIC
- [TLS/SSL](tls_ssl.md) — required for HTTP/2 in all browser implementations
- [gRPC](grpc.md) — the primary user of HTTP/2 multiplexing for RPC streaming
- [WebSocket](websocket.md) — HTTP/2 has a native `CONNECT` tunnel alternative to WebSocket upgrades

- [nghttp2 - HTTP/2 reference implementation](https://nghttp2.org/)
- [High Performance Browser Networking, ch. 12 (free online)](https://hpbn.co/http2/)
