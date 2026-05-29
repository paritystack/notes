# Distributed ID Generation

Generating unique IDs across many nodes without coordination is a load-bearing primitive in nearly every system design — tweets, orders, messages, traces. The choice of ID scheme constrains your database, sorting, and bandwidth.

## Requirements Matrix

| Requirement | Why |
|---|---|
| **Unique** | Non-negotiable. |
| **Distributed** | No single bottleneck. |
| **K-sortable / monotonic** | Index locality, time-ordered queries, cursor pagination |
| **Compact** | Smaller index, smaller payloads, fewer bytes on the wire |
| **Predictable layout** | Maybe — enumerability is a security concern |
| **No coordination** | Lowest latency |

These conflict. Pick the right trade.

## The Lineup

| Scheme | Size | Sortable | Coordinator? | Use when |
|---|---|---|---|---|
| **DB auto-increment** | 8 B | Strict ↑ | DB (bottleneck) | Single-master OLTP |
| **UUID v4 (random)** | 16 B | ❌ | None | Don't care about sort, want guaranteed uniqueness |
| **UUID v7** | 16 B | ✅ (time-prefixed) | None | Modern default — sortable + unique |
| **ULID** | 16 B (26 char) | ✅ | None | Like UUIDv7 but Crockford base32 |
| **Snowflake** | 8 B | ✅ | Worker ID assignment | High write throughput, compact 64-bit |
| **KSUID** | 20 B | ✅ | None | URL-safe, sortable |
| **MongoDB ObjectId** | 12 B | ✅ | None | Mongo-native |
| **Stripe-style prefixed** | Variable | ❌ | One of above + prefix | Human-debuggable APIs (e.g. `ord_8a3f…`) |

## Snowflake (Twitter, the Classic)

64-bit ID, partitioned bitwise:
```
| 1 bit | 41 bits        | 10 bits   | 12 bits  |
| sign  | timestamp (ms) | worker ID | sequence |
   0     since epoch       0-1023     0-4095/ms
```

- **41 bits ms** → ~69 years from custom epoch.
- **10 bits worker** → 1024 unique workers.
- **12 bits sequence** → 4096 IDs per worker per ms = 4M IDs/sec/worker.

```
Per worker peak: 4096 × 1000 = 4.1M IDs/sec
1024 workers:    4.2B IDs/sec global capacity
```

### Worker ID assignment
- **Static config**: pod/host gets fixed ID. Brittle.
- **ZooKeeper / etcd lease**: dynamic, expires on shutdown. Standard.
- **MAC-address hash**: convenient but collision risk → fence with coordination service.

### Clock skew problem
Snowflake breaks if clocks go backward (NTP correction, leap second).

Mitigations:
- Refuse to issue IDs if `now < last_ts`; wait until clock catches up.
- Use a monotonic clock as the floor.
- Reserve a sentinel sequence when ms hasn't ticked but sequence exhausted (sleep).

### Variants
- **Sonyflake**: 39-bit time at 10 ms resolution, 8-bit machine, 16-bit seq.
- **Discord Snowflake**: Twitter format, custom epoch (Jan 2015).
- **Mastodon-flake**: 41-bit time, 22-bit random suffix.
- **Instagram**: shard ID instead of worker ID — embed the shard you're writing to in the ID itself.

## UUID v7

128-bit:
```
| 48 bits        | 4 bits | 12 bits | 2 bits | 62 bits |
| unix_ts_ms     | ver=7  | rand_a  | var=10 | rand_b  |
```

K-sortable by time prefix, otherwise random. Standard since RFC 9562 (May 2024).

### Why UUIDv7 over Snowflake
- **No coordination**: no worker-ID assignment, no clock-skew handshake.
- **128-bit**: random suffix collision probability negligible at any rate.
- **DB-friendly**: time-prefixed means B-tree inserts at the tail, no random IO.

### Why Snowflake over UUIDv7
- **8 bytes vs 16** → half the index size, half the network bytes per ID.
- **Pure numeric** → faster comparisons, no string handling.
- At very large scale (Twitter-class), the size halving is material.

## ULID

128-bit, encoded as 26 Crockford base32:
```
01ARZ3NDEKTSV4RRFFQ69G5FAV
└────┬────┘└─────┬─────────┘
  10 char     16 char
  timestamp   randomness
```

Functionally identical to UUIDv7 with a friendlier encoding. Case-insensitive, no `I/L/O/U` to avoid ambiguity.

## KSUID (Segment)

160-bit, 27 chars base62:
```
| 32-bit time (s) | 128-bit random |
```

Second-precision timestamp; bigger than ULID but URL-safe and well-supported.

## Stripe-Style Prefixed IDs

```
cus_8a3f1c2e9d4b...    customer
ord_..                 order
pi_..                  payment intent
```

- Prefix is the type → URL inspection reveals object class.
- Body is UUID/Snowflake/random base.
- Two-way mapping: helps debugging, helps observability.
- Versionable: `cus_test_…` vs `cus_…`.

## DB Auto-Increment

Single-master sequence. Pros: dense, small (8 B), strict ordering. Cons:

- **Bottleneck**: every write contends for the sequence. ~10-50K/s ceiling.
- **Hot-spotting**: monotonic IDs → last B-tree leaf is hot.
- **No multi-master**: can't safely shard.
- **Enumeration risk**: `/orders/1001` invites scraping.

Mitigations:
- **Sequence cache** per worker (allocate ranges of 1000 at a time).
- **Shard sequences** (each shard has its own; embed shard ID).
- **HiLo pattern**: get block from DB, generate locally.

## Multi-Master Sequences

When you must use auto-increment across many writers:
- **Different starting offsets**: master A uses odd, master B even (`SET auto_increment_increment=2`).
- **Reserved ranges per shard**: shard 1: 1-1B, shard 2: 1B-2B, etc.
- **Cluster-aware sequences**: CockroachDB, Spanner have built-in.

## Sortable IDs and Hot-Spotting

Time-sortable IDs (Snowflake, UUIDv7) write to the **tail** of B-tree indexes. Pros: sequential IO, cache-friendly. Cons: that page is hot.

Mitigations:
- **Reverse the high-order bits**: spreads writes across the tree.
- **Hash-then-prefix-with-time**: hybrid for read locality + write spread.
- **Bucketed indexes**: write to a hash-bucketed table, query via union.

But: in practice, the hot-tail problem is solved by the LSM-tree (Cassandra, RocksDB) approach where writes go to memtable first. If your DB is B-tree (Postgres, MySQL), watch for this.

## Security Considerations

| Concern | Bad ID | Fix |
|---|---|---|
| Enumeration | Auto-increment | Use UUIDv7/ULID for public IDs |
| Volume disclosure | Sequential | Random or coarse-time IDs |
| Timing leaks | UUIDv7 reveals exact creation time | Coarsen timestamp; use UUIDv4 for sensitive ops |

**Common pattern**: dual IDs — `id` (numeric, internal, indexed) + `public_id` (UUID, opaque, in URLs).

## Selection Cheatsheet

| Want | Pick |
|---|---|
| Smallest, sortable, big scale | Snowflake |
| Modern default, no coordinator | UUID v7 |
| Human/URL-friendly + sortable | ULID |
| Mongo | ObjectId |
| OLTP, single-region, low write rate | DB auto-increment |
| Public-facing API | Prefixed (Stripe-style) over UUIDv7 |
| Tracing / spans | 128-bit random (W3C trace-id) |

## Capacity Examples

```
Snowflake @ 1 worker: 4.1M IDs/sec
Snowflake @ 1024 workers, fully utilized: 4.2B IDs/sec
UUIDv7: no coordination → trivially generated at any rate per process
Auto-increment Postgres sequence: ~25K/sec realistic ceiling
```

## Interview Cheat

If the question involves "many writers", "many shards", "trace ID", or "object ID at scale":
- Default answer: **UUID v7 or Snowflake**.
- Justify: no coordination (vs auto-increment), time-sortable (for index locality + cursor pagination).
- For public APIs, mention prefixed IDs for debuggability.
- For high write rate, prefer Snowflake's 8 bytes.
- Mention clock-skew handling if you pick Snowflake.

## Pitfalls

- **Storing UUIDs as strings**: 36 bytes vs 16. Use binary/uuid columns.
- **Sorting on UUIDv4**: it's random — sort by created_at, not the ID.
- **Snowflake without clock guard**: NTP-induced backward jumps can produce duplicates.
- **Reusing worker IDs**: pod restart → same ID → potential dupes if old IDs still in flight. Lease via etcd/ZK with TTL.
- **Embedding shard ID in ID, then resharding**: now the ID lies about location. Use a routing table, not embedded shard.

## Related

- `databases.md` — sequence vs uuid pk tradeoffs
- `distributed_systems.md` — clock synchronization, vector clocks
- `consistent_hashing.md` — sharding by ID prefix
- `design_url_shortener.md` — case study using ID gen heavily
