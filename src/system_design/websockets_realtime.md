# WebSockets & Realtime Transports

Push-from-server is fundamentally different from request-response. Picking the right transport is half the battle; scaling stateful connections is the other half.

## Transport Options

| Transport | Direction | Connection | Overhead | Use when |
|---|---|---|---|---|
| **Short polling** | C→S only | Per request | High | Low-rate updates, dead simple |
| **Long polling** | C→S, S→C delayed | Held open until event | Medium | Compat with old proxies/firewalls |
| **Server-Sent Events (SSE)** | S→C only | Long-lived HTTP | Low | One-way push (notifications, feeds) |
| **WebSocket** | Bidirectional | Long-lived TCP, frames | Low | Chat, games, collaborative editing |
| **WebTransport** | Bidirectional, multiplexed | QUIC (UDP) | Lowest | New apps, head-of-line-free |
| **gRPC streaming** | Bi/uni | HTTP/2 | Low | Service-to-service realtime |
| **MQTT** | Pub/sub | Persistent TCP | Lowest | IoT, mobile (battery-friendly) |

## Decision Tree

```
Need server → client push?
├── No → Use polling or regular REST
├── Yes, one-way only?
│   ├── Browser → SSE (HTTP, simpler, auto-reconnect)
│   └── Service-to-service → gRPC server streaming
└── Yes, bidirectional?
    ├── Browser → WebSocket
    ├── Mobile/IoT → MQTT (or WS over secure mux)
    └── New web app needing multiplexed streams → WebTransport
```

## WebSocket Anatomy

### Handshake
```
Client: GET /ws HTTP/1.1
        Host: example.com
        Upgrade: websocket
        Connection: Upgrade
        Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
        Sec-WebSocket-Version: 13

Server: HTTP/1.1 101 Switching Protocols
        Upgrade: websocket
        Connection: Upgrade
        Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

After 101, the TCP connection is a WebSocket — framed bidirectional binary/text.

### Frames
| Opcode | Meaning |
|---|---|
| 0x0 | Continuation |
| 0x1 | Text (UTF-8) |
| 0x2 | Binary |
| 0x8 | Close |
| 0x9 | Ping |
| 0xA | Pong |

Always send Ping/Pong every ~30s to detect dead connections and keep NAT/LB sessions warm.

## SSE Anatomy

```
GET /events HTTP/1.1
Accept: text/event-stream

HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

event: order_update
id: 42
data: {"order_id":123,"status":"shipped"}

retry: 5000

event: notification
data: {"msg":"hello"}
```

- Auto-reconnect built into browsers.
- `id:` enables resume via `Last-Event-ID` header.
- One-way only — for chat you'd pair with REST POST for sends.

## Scaling Stateful Connections

The fundamental constraint: **a WebSocket is stateful**. The user is bound to a specific edge process.

### Connection Density per Box

| Resource | Limit | Mitigation |
|---|---|---|
| File descriptors | 1M+ if `ulimit -n` raised | `fs.file-max`, `fs.nr_open` |
| TCP ephemeral ports | 64K per source IP | Multiple IPs, source port reuse |
| Memory | ~10 KB per idle conn | Smaller buffers, message bus offload |
| epoll capacity | Tens of millions | Use efficient I/O (Go, Rust, Node, libuv) |

**Realistic envelope:** 100K–1M concurrent WS per modern box with tuning.

### Edge Tier Architecture

```
                    Anycast LB / GeoDNS
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
           ┌────────┐    ┌────────┐    ┌────────┐
           │ WS Edge│    │ WS Edge│    │ WS Edge│ (sticky to user)
           │  Pod 1 │    │  Pod 2 │    │  Pod 3 │
           └───┬────┘    └───┬────┘    └───┬────┘
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Pub/Sub bus        │
                  │  (Redis Pub/Sub,    │
                  │   NATS, Kafka)      │
                  └─────────────────────┘
                             │
                             ▼
                       App services
```

**Pattern:** edge handles connections, business logic in stateless app services, messaging glues them. To send a message to "user 42", look up their edge pod (presence map) and route via pub/sub.

### Presence Map
Tracks which edge pod holds each user's connection.

```
key: user_id
val: { edge_pod_id, conn_id, connected_at, last_heartbeat }
TTL: 60s (refreshed by heartbeat)
```

Stored in Redis or DynamoDB. On disconnect, delete; on reconnect, update.

### Sticky vs Re-route
- **Sticky load balancing** (cookie-based or hash-based on user_id) so the same user lands on the same pod after reconnect. Helps caches.
- **Re-route on disconnect**: don't try to preserve pod affinity if pod is unhealthy — let LB pick anew.

## Reliability Patterns

### Heartbeats
- Client → server ping every 30s; server replies pong.
- Server → client ping if no traffic for 60s.
- Either side drops connection if no response in 2× period.

**Why?** NAT/LB drop idle TCP after 60–300s silently. Ping keeps it open AND surfaces dead conns fast.

### Reconnect with backoff
```
attempt 1: 1s
attempt 2: 2s + jitter
attempt 3: 4s + jitter
... cap at 30-60s
```

Jitter prevents thundering herd after backend recovers.

### Resume / replay
For ordered streams (chat, feeds), give each message a server-side sequence number. On reconnect, client sends last seen seq; server replays missed events.

```
{ "type":"sub", "topic":"chat:room123", "since_seq":4521 }
```

Bound replay storage (e.g., last 10K msgs per topic). Beyond that, force full reload.

### Backpressure
Slow client → server-side write buffer grows → memory pressure → OOM. Mitigations:
- Bounded send queue per conn; drop or disconnect when full.
- Apply per-conn flow control (advertise window).
- Reject "bursty" clients — coalesce updates server-side.

## Common Real-time Workloads

### Chat / Messaging
- WebSocket between client and edge.
- Edge publishes message to pub/sub topic per conversation.
- Other edges subscribe to topics for their connected users.
- Persistent store (Cassandra) for history.
- See `design_chat_system.md`.

### Live Feeds (Twitter live tweet stream, stock prices)
- SSE often sufficient (one-way).
- Fan-out from publisher → topic → many subscribers.

### Collaborative Editing (Google Docs, Figma)
- WebSocket carrying CRDT/OT operations.
- Server reconciles concurrent edits, broadcasts.
- Snapshot + ops log for crash recovery.

### Live Location (Uber driver tracking)
- Client sends location every few seconds (small WS frame).
- Server aggregates, pushes nearby driver positions to rider.
- See `design_ride_sharing.md`.

### Multiplayer Games
- WebTransport or raw UDP preferred over WS.
- Authority server, client prediction, rollback.

## Protocol-Layer Pitfalls

| Pitfall | Fix |
|---|---|
| Proxy strips `Upgrade` header | Use proper L4 LB or WS-aware L7 LB |
| TLS termination loses persistence | Terminate at LB *and* keep TCP to backend persistent |
| HTTP/2 multiplexing limits SSE | Use HTTP/2 for SSE — gives multiplexed streams, no 6-conn limit |
| Mobile NAT timeouts | Heartbeats every 25-30s |
| Browser tab background throttling | Server-side keepalive, no client-driven polling |

## Authentication for Long-lived Connections

You can't easily attach auth headers after the WS handshake. Options:

1. **Auth on handshake**: `Sec-WebSocket-Protocol: bearer.{jwt}` or `?token=` in URL. Validate on upgrade.
2. **First-message auth**: connect anonymously, send `{type:"auth", token:...}` as first frame. Server holds in pending state until valid.
3. **Cookie + same-origin**: same auth as REST.

**Token expiry mid-connection?** Either disconnect on expiry (forcing reconnect with fresh token) or implement in-band refresh (server sends `{type:"refresh"}`, client responds with new token).

## Cross-platform Notes

- **iOS**: backgrounded apps lose sockets. Use APNs for delivery while backgrounded, reconnect on foreground.
- **Android**: same with FCM. Doze mode kills sockets aggressively.
- **Mobile networks**: bandwidth & battery — keep messages small (binary, compressed), heartbeat infrequently.

## Capacity Sizing Example

```
10M concurrent users, 100 KB RAM/conn = 1 TB RAM total
@ 64 GB boxes → ~16 boxes minimum, 2× for redundancy = 32

Heartbeat: 10M × 100 B / 30s = 33 MB/s
Outbound msg rate: 1 msg/user/min × 1 KB = 167 MB/s ≈ 1.3 Gbps
```

## Interview Cheat: Real-time in a Design

When you hear "live", "real-time", "instant" — reach for:
1. **Pick a transport** with justification (WS for bi-dir, SSE for one-way).
2. **Edge tier separation** — connections at edge, state stateless behind.
3. **Pub/sub fan-out** between edges.
4. **Presence map** for routing.
5. **Heartbeat + reconnect + resume**.
6. **Backpressure** consideration.

## Common Interview Gotchas

- "Why not WS for everything?" — Auth, caching, backwards-compat, mobile battery, NAT, infrastructure friendliness all suffer.
- "What if connection drops mid-message?" — Sequence numbers + replay.
- "How do you scale to 100M?" — Horizontal edge tier + pub/sub. Presence map.
- "What about ordering across edges?" — Per-topic ordering via single-writer pub/sub partition. Global ordering is expensive — usually unnecessary.

## Related

- `design_chat_system.md` — full case study
- `message_queues.md` — pub/sub bus underneath
- [Load balancing](load_balancing.md) — WS-aware load balancers need sticky sessions
- [Design patterns](design_patterns.md) — backpressure, circuit-breaker patterns for realtime systems

## Where this connects

- [Load balancing](load_balancing.md) — WebSocket connections are stateful; load balancers must use sticky routing
- [Message queues](message_queues.md) — fan-out via pub/sub or queues delivers messages to multiple WebSocket connections
- [Databases](databases.md) — message persistence often uses Cassandra or Redis for realtime chat
