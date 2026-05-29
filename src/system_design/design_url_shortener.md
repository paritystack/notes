# Design a URL Shortener (TinyURL / Bitly)

Classic warm-up. Tests ID generation, KV-store sizing, read-heavy cache design, and custom-alias collisions.

## 1. Requirements

### Functional
- POST a long URL → return short URL (`https://sho.rt/abc123`).
- GET short URL → 301/302 redirect to original.
- Optional: custom alias (`/my-link`).
- Optional: expiration timestamp.
- Optional: per-link analytics (click count).
- Per-user link management (auth).

### Non-functional
- 100M new URLs/month.
- 100:1 read:write ratio.
- P99 redirect latency < 50 ms.
- 99.99% availability for redirects (the read path).
- 5 years retention.

### Out of scope
- Rich analytics, A/B testing, link previews, malware scanning.

## 2. Capacity Estimation

```
New URLs:    100M/month / (30 × 86400) ≈ 40 WPS avg, 200 WPS peak
Reads:       40 × 100 = 4,000 RPS avg, 20K RPS peak

Storage (5y):
  100M/mo × 60 mo = 6B URLs
  per row: shortcode 7B + long URL ~100B + user_id 8B + ts 8B + meta 20B ≈ 150B
  6B × 150B = 900 GB total

Bandwidth:
  Reads: 20K × 200B (redirect payload) = 4 MB/s
  Writes: 200 × 500B = 0.1 MB/s
```

Modest scale. Single Postgres + Redis would work; we'll design for headroom.

### Short code length
- Base62 (`[A-Za-z0-9]`, 62 chars).
- 6 chars → 62⁶ = 56B → enough for ~6B URLs comfortably.
- 7 chars → 62⁷ = 3.5T → enough for >100× our 6B target.

Pick **7 chars** for collision headroom and future growth.

## 3. API Design

```
POST /v1/links
Body: { long_url, custom_alias?, expires_at? }
Auth: Bearer token
Resp: 201 { short_url, code, expires_at }

GET /:code
Resp: 301 Location: <long_url>   (or 404)

GET /v1/links?cursor=&limit=50
Resp: 200 { items: [...], next_cursor }

DELETE /v1/links/:code
Resp: 204
```

**301 vs 302**: 301 (permanent) is cacheable by browsers — slashes your read load but kills per-click analytics. 302 (temporary) hits your server every time. **Default 302**, allow 301 opt-in.

## 4. Data Model

```
links (Postgres or KV store):
  code         CHAR(7)   PK
  long_url     TEXT
  user_id      BIGINT    INDEXED
  created_at   TIMESTAMPTZ
  expires_at   TIMESTAMPTZ NULL
  click_count  BIGINT    (lazily updated)

users:
  user_id      BIGINT    PK
  email, ...

clicks (optional, write-heavy):
  code, ts, ip_hash, referrer  → batched into time-series store
```

**Why one row per code:** look up by code is the hot read. Single point query.

**Index on user_id**: needed for the "my links" page, but not on critical redirect path.

## 5. High-Level Architecture

```
                   ┌──── CDN (edge cache) ──┐
   client ────────►│                        │
                   │  301 served at edge    │
                   └────────┬───────────────┘
                            │ miss
                            ▼
                  ┌──────────────────┐
                  │   API Gateway    │
                  │ (LB, rate-limit) │
                  └────────┬─────────┘
                ┌──────────┼──────────┐
                ▼          ▼          ▼
            ┌───────┐  ┌───────┐  ┌──────────┐
            │ Write │  │ Read  │  │ Analytics│
            │ svc   │  │ svc   │  │ pipeline │
            └───┬───┘  └───┬───┘  └────┬─────┘
                │          │           │
                │   ┌──────▼──────┐    │
                │   │   Redis     │    │  cache: code→long_url
                │   └──────┬──────┘    │
                ▼          ▼ miss      ▼
            ┌──────────────────┐    ┌─────────┐
            │   Postgres /     │    │ Kafka   │
            │   KV (codes)     │    │ → ClickDB
            └──────────────────┘    └─────────┘
```

## 6. Deep Dives

### A. Short Code Generation

Three viable schemes:

| Scheme | Pros | Cons |
|---|---|---|
| **Hash long URL** (MD5/SHA, take first 7 base62 chars) | Stateless, dedup same URL automatically | Collisions → retry with salt; expensive to check |
| **Counter + base62** | Dense, sequential | Predictable (enumerable); needs distributed counter |
| **Random + collision check** | Simple | DB roundtrip per insert |
| **Pre-allocated key range** (preferred) | No DB hit on insert | Need allocator service |

**Recommended: pre-allocated ranges.**

Run a Key Generation Service (KGS) that pre-mints unused 7-char codes into a pool. Writers pull a code from the pool and use it directly.

```
   ┌─────────┐                ┌───────────┐
   │ KGS     │  Reserve N     │ Pool (DB) │
   │ workers │ ──────────────►│ used/free │
   └────┬────┘                └───────────┘
        │
        ▼ (in-memory buffer of unused codes)
   Each writer instance keeps ~1000 codes
   Pulls more when buffer drains
```

Benefits:
- **No collisions** (DB enforces uniqueness on mint).
- **No DB roundtrip** on insert critical path.
- **Predictability avoided** if pool is shuffled.

Crash recovery: pulled-but-unused codes leak. At 7 chars / 62⁷ = 3.5T codes, leak is irrelevant.

**Custom aliases**: separate path. Check `INSERT … ON CONFLICT` on the aliases table; reject on conflict.

### B. Read Path Optimization

Read path is the hot path. Tiered cache:

```
1. CDN edge (Cache-Control: 301 long max-age=86400)
       ↓ miss
2. Redis (code → long_url), hot working set
       ↓ miss
3. Postgres (authoritative)
```

**CDN**: only useful with 301s. If 302s for analytics, skip CDN, go straight to Redis. With CDN: most 301 redirects never hit your origin.

**Redis sizing**: 6B URLs × 200B = 1.2 TB total. Hot 5% per Zipf ≈ 60 GB. Fits in a small Redis cluster comfortably. LRU eviction.

**Bloom filter** in front of Redis: avoids cache lookup for definitely-missing codes (malformed or never-issued). 6B entries at 1% FPR → 7 GB Bloom; useful but optional.

### C. Click Analytics Without Slowing Redirects

Don't increment `click_count` synchronously — it'd bottleneck Postgres.

```
Read svc returns 302
      └──► async event → Kafka → ClickHouse / Druid
                                         │
                                         └──► aggregations
                                              (per-link daily counts)
```

For the "my links" dashboard, aggregate counts joined back via batch job, written to `links.click_count` hourly. Real-time count optional.

## 7. Bottlenecks & Scaling

| At 10× | Hits the wall | Fix |
|---|---|---|
| 200K RPS reads | Postgres reads | Add Redis replicas; rely on CDN |
| 2K WPS writes | Postgres writes | Shard by code hash; bigger KGS pool |
| 60 GB hot cache | Redis memory | Cluster; consistent hashing |
| 9 TB total | Postgres single node | Shard (code prefix or hash); migrate to Cassandra |
| 200 Mbps egress | Origin bandwidth | CDN absorbs |

### Sharding strategy
Hash on `code` → assign to one of N shards. Even distribution by design (codes are uniform-random). Routing layer in API.

## 8. Tradeoffs & Alternatives

| Choice | Picked | Why |
|---|---|---|
| Postgres vs Cassandra | Postgres | Sub-TB, simple. Cassandra justified at 10×. |
| KGS pre-mint vs hash | Pre-mint | Avoids collisions, no DB on write path. |
| 301 vs 302 | 302 default | Want click counts; 301 opt-in for noncritical links. |
| Redis vs Memcached | Redis | Persistence, eviction policies, cluster mode. |
| Kafka vs direct ClickHouse | Kafka | Decouples; bursts absorbed; replayable. |

## 9. Follow-up Questions (Expect These)

- **"How do you prevent abuse / spam / malware?"** — Rate limit by user; phishing/safebrowsing API check on write; quarantine new links until verified.
- **"How do you delete a link safely?"** — Soft delete + tombstone in cache; expire CDN entry via tag-based purge.
- **"What if a link expires?"** — Background sweeper removes from cache + DB. Redirect path checks `expires_at`. Bloom filter rebuilt periodically.
- **"How would you handle billions of clicks/day?"** — Edge logs aggregated into ClickHouse; per-link counters via Redis HyperLogLog if cardinality (unique visitors) matters.
- **"What if two users want the same custom alias simultaneously?"** — Unique constraint + INSERT … ON CONFLICT. First-write wins.
- **"How do you migrate to a new code scheme?"** — Dual-write old + new; gradual rewrites; keep old codes forever (URLs in the wild).

## Related

- `id_generation.md` — alternative code-generation schemes
- `caching.md` — read-side cache design
- `cdn.md` — edge caching of redirects
- `databases.md` — sharding strategies
