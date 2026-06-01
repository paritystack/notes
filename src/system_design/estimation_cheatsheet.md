# Estimation Cheatsheet

Numbers and templates for back-of-envelope calculations. Memorize the latency table and the powers-of-ten table; everything else derives from them.

## Latency Numbers (Memorize)

```
L1 cache              0.5 ns
Branch mispredict     5   ns
L2 cache              7   ns        (14× L1)
Mutex lock/unlock     25  ns
Main memory ref       100 ns        (200× L1, 20× L2)
Compress 1KB (zippy)  3   µs
Send 1KB over 1Gbps   10  µs
Read 4KB from SSD     150 µs        (~1,000× memory)
Read 1MB seq from mem 250 µs
Round trip same DC    500 µs
Read 1MB seq SSD      1   ms        (4× memory)
HDD seek              10  ms        (20× SSD)
Read 1MB seq HDD      20  ms        (80× memory)
Packet CA→Netherlands 150 ms
```

### Quick mnemonics
- **L1: 1ns, RAM: 100ns, SSD: 100µs, DC roundtrip: 500µs, disk seek: 10ms, cross-Atlantic: 150ms.**
- Memory is **100×** faster than SSD. SSD is **100×** faster than HDD seek.
- Sequential memory read of 1MB ≈ 250µs. Sequential SSD ≈ 1ms. Sequential HDD ≈ 20ms.

## Powers of Ten

| Magnitude | Bytes | Example |
|---|---|---|
| KB | 10³ | A short article |
| MB | 10⁶ | A photo |
| GB | 10⁹ | A movie |
| TB | 10¹² | A small DB |
| PB | 10¹⁵ | YouTube uploads / day |
| EB | 10¹⁸ | Global mobile data / month |

## Time-to-QPS Conversions

```
Seconds in a day  = 86,400 ≈ 10⁵
Seconds in a year = 31.5M ≈ 3 × 10⁷

X events/day → X / 10⁵ events/sec (avg)
1B events/day → 10K QPS avg
100M events/day → 1K QPS avg
1M events/day → ~12 QPS avg
```

### Peak multipliers

- **Web traffic:** peak ≈ 2–3× average.
- **Social/news feed:** peak ≈ 5× average (event-driven spikes).
- **Banking/finance:** peak ≈ 10× average (end-of-day, paydays).
- **Streaming:** peak ≈ 4× average (evening hours).

## Storage Sizing

### Per-row sizes (rough)
| Entity | Size |
|---|---|
| UUID | 16 B |
| Timestamp | 8 B |
| Username | ~30 B |
| Email | ~30 B |
| Tweet text | ~280 B |
| Tweet w/ metadata | ~500 B |
| URL (avg) | ~80 B |
| Chat message | ~100 B |
| Image (compressed) | ~200 KB |
| Audio (1 min, MP3) | ~1 MB |
| Video (1 min, 1080p) | ~50 MB |
| Video (1 min, 4K) | ~300 MB |

### Worked example: tweet storage
```
DAU         = 200M
tweets/user = 2/day
tweets/day  = 400M
size        = 500B
per day     = 400M × 500B = 200 GB
per year    = 73 TB
5 years     = 365 TB (text only)
```

## Bandwidth Sizing

```
Bandwidth = QPS × payload size

Example: 1M RPS, 1KB response → 1 GB/s = 8 Gbps
Example: 100K video viewers × 5 Mbps each → 500 Gbps
```

### Common payload sizes
| Response | Size |
|---|---|
| JSON status | 100 B |
| Auth token | 1 KB |
| Web page (no img) | 100 KB |
| Web page (full) | 2 MB |
| Mobile API resp | 5 KB |

## Connection Sizing

For long-lived connections (WebSocket, gRPC streaming, MQTT):

- **One Linux box** with tuning: ~1M concurrent TCP connections (ephemeral-port and file-descriptor limits raised).
- **Memory per conn:** ~10 KB baseline, more with buffers. 1M conns → ~10 GB RAM.
- **Heartbeat cost:** 1M conns × ping every 30s × 100B = ~3 MB/s.

## Cache Sizing

```
Working set × hit rate target = cache size

Example: 80% hit rate on 1B tweets, hot 10% accounts for 90% of reads
  → cache 100M tweets × 500B = 50 GB
```

### Rule-of-thumb hit rates
- Read-mostly KV (Redis as cache): aim for **95%+** hit rate.
- Web page cache (CDN): aim for **90%+** hit rate.
- DB query cache: **70–85%** typical.

## Replication & Sharding Math

```
Replication factor 3 + erasure coding overhead ≈ multiplier of 3-4×
Cassandra RF=3, real disk usage = 3× data

Sharding: target ~1-10 TB per shard, ~10-100K QPS per shard
Tweets 1 PB total → 100-1000 shards
```

## Server Throughput Rules of Thumb

| Workload | Single modern box |
|---|---|
| Static HTTP (Nginx) | 100K–500K RPS |
| Dynamic app (Node/Python) | 5K–50K RPS |
| Java/Go service | 50K–200K RPS |
| MySQL OLTP | 5K–20K writes/s |
| PostgreSQL OLTP | 5K–25K writes/s |
| Redis | 100K–1M ops/s |
| Cassandra | 20K–50K writes/s/node |
| Kafka broker | 100K–1M msgs/s |
| Elasticsearch indexing | 10K–50K docs/s/node |

## Cluster Sizing

```
Nodes needed = peak QPS / per-node throughput × headroom (1.5-2×)

Example: 1M peak RPS, Cassandra at 30K writes/s/node
  → 1M / 30K × 2 = ~67 nodes
```

## Network Fundamentals

```
1 Gbps   = 125 MB/s     ≈ 1 movie/sec
10 Gbps  = 1.25 GB/s    ≈ 10 movies/sec
100 Gbps = 12.5 GB/s
```

### Cross-region latency (typical)
- Same DC: 0.5 ms
- Same region (DCs): 1–5 ms
- US East ↔ US West: 70 ms
- US ↔ Europe: 90–110 ms
- US ↔ Asia: 130–180 ms
- Anywhere ↔ Anywhere worst: 300 ms

## Worked Example: News Feed

```
Given: 200M DAU, each loads feed 10×/day, 100 followees average

Reads/day  = 200M × 10 = 2B feed loads → 23K RPS avg, ~100K peak
Writes/day = 200M × 2 tweets = 400M → 4.6K WPS avg, ~14K peak

Fan-out cost (push):
  400M tweets × 100 followers avg = 40B writes/day to timelines
  → 460K WPS into timeline store. Needs sharding.

Cache (timelines):
  200M users × 800 tweets × 500B = 80 TB
  Hot 10% = 8 TB → fits in a 20-node Redis cluster
```

## When to Stop Estimating

Stop once you've:
1. Confirmed scale (QPS, storage, bandwidth) is in the right ballpark.
2. Identified the bottleneck this implies (write-heavy? cache-heavy? bandwidth-heavy?).
3. Justified at least one architectural choice with a number.

Don't:
- Do precise arithmetic on the board.
- Calculate to 3 significant figures.
- Estimate things that don't drive a decision.

## Related

- `interview_framework.md` — when in the interview to do this
- [Scalability](scalability.md) — what to do once you've established the load
- [Caching](caching.md) — how to spend the cache budget

## Where this connects

- [Scalability](scalability.md) — load estimates from this cheatsheet drive the scalability decisions
- [Caching](caching.md) — cache budget calculation uses storage/throughput numbers from this sheet
- [Interview framework](interview_framework.md) — back-of-envelope estimation is phase 2 of the interview framework
