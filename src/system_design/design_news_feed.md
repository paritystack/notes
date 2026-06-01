# Design a News Feed (Twitter / Facebook)

The most-asked system design problem. Tests fan-out strategy, celebrity-skew handling, sharding, caching, and ranking.

## 1. Requirements

### Functional
- Post a tweet (text up to 280 chars + optional media).
- Follow / unfollow a user.
- View a **home timeline**: posts from followees, newest first.
- View a **user timeline**: own posts.
- Like / reply / retweet.

### Non-functional
- 300M DAU, ~500M MAU.
- Read:write ratio ≈ 100:1.
- Home timeline P99 < 200 ms.
- Eventual consistency OK (a few seconds delay tolerated).
- 99.99% read availability.

### Out of scope
- Ranking algorithm details (assume reverse-chronological core + simple ranking).
- DMs, search, trends.

## 2. Capacity Estimation

```
DAU                 = 300M
Tweets/user/day     = 2
Writes (tweets)/day = 600M  → 7K WPS avg, 20K WPS peak
Reads (feed loads)  = 300M × 10 = 3B/day → 35K RPS avg, 175K peak
Follows/user        = 200 average, 5K p95, 100M p99 (celebrities)

Storage:
  tweet            ≈ 500 B (text+meta)
  600M/day × 500B  = 300 GB/day → 100 TB/year text
  Media (separate, in object store) → multi-PB
  Timeline cache   (per user, last 800 tweet refs × 16B = 12 KB)
                   300M × 12 KB ≈ 3.6 TB hot
```

## 3. API Design

```
POST /v1/tweets
  Body: { text, media_ids? }
  Resp: { tweet_id, created_at }

GET /v1/timeline/home?cursor=&limit=20
  Resp: { tweets:[...], next_cursor }

GET /v1/timeline/user/:user_id?cursor=&limit=20

POST /v1/follow      { followee_id }
DELETE /v1/follow    { followee_id }

POST /v1/like        { tweet_id }
```

Cursor pagination, not offset. Cursors encode (timestamp, tweet_id) for stable iteration.

## 4. Data Model

```
users (PostgreSQL)
  user_id (Snowflake) PK
  handle UNIQUE, name, created_at

tweets (Cassandra; partition by user_id, cluster by tweet_id DESC)
  user_id, tweet_id PK
  text, media_ids[], created_at

follows (Cassandra)
  follower_id, followee_id PK
  created_at
  + inverse table partitioned by followee_id (for fan-out)

likes (Cassandra)
  user_id, tweet_id PK, ts

home_timeline_cache (Redis sorted set, key = user_id)
  member = tweet_id, score = created_at_micros
  capped to ~800 entries
```

**Why Cassandra for tweets:** write-heavy, time-series shape, range queries by user, no transactions needed across tweets.

**Why per-user sorted set in Redis:** instant `ZREVRANGE` for "give me last 20 tweets ≤ cursor".

## 5. High-Level Architecture

```
   Clients ───► API GW ───┐
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   ┌─────────┐      ┌───────────┐      ┌──────────┐
   │ Tweet   │      │ Timeline  │      │ Follow   │
   │ svc     │      │ svc       │      │ svc      │
   └────┬────┘      └─────┬─────┘      └────┬─────┘
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────┐     ┌───────────┐     ┌──────────┐
   │Cassandra │     │  Redis    │     │Cassandra │
   │ tweets   │     │ timelines │     │ follows  │
   └────┬─────┘     └─────▲─────┘     └──────────┘
        │                 │
        ▼                 │ push fan-out
   ┌───────────────────┐  │
   │ Kafka: new_tweets │──┘
   └───────────────────┘
        │
        ▼
   ┌───────────────┐
   │ Fan-out svc   │ (looks up followers, pushes tweet_id into each timeline)
   └───────────────┘
```

## 6. Deep Dives

### A. Fan-out Strategy: Push vs Pull vs Hybrid

| Approach | Write path | Read path | Best for |
|---|---|---|---|
| **Pull (fan-out on read)** | O(1): just insert tweet | O(N) per read: gather + merge tweets from N followees | Heavy posters with few followers |
| **Push (fan-out on write)** | O(F) per tweet: write tweet_id into F follower timelines | O(1) read: ZREVRANGE on user's cache | Normal users, read-heavy |
| **Hybrid** | Push for ≤K follower users; pull for celebrities | Read merges pushed timeline + pull from celebrities followed | Real-world skew |

**The celebrity problem (push-only is broken):**
- Celebrity has 100M followers.
- One tweet → 100M writes into 100M timeline caches.
- Even at 50K writes/sec/Redis-node → 33 minutes to propagate, plus huge tail load.

**Hybrid (recommended):**

```
On write:
  if author.followers < 10_000:
      push tweet_id into each follower's timeline cache
  else:
      mark author as 'celebrity', don't push

On read:
  base_timeline = ZREVRANGE redis:user_id
  celebrities = follows.where(follower=user, followee.is_celebrity)
  recent_celeb_tweets = fetch latest few from each celebrity (cached per-celeb)
  return merge(base_timeline, recent_celeb_tweets, sorted by ts)
```

Celebrities have their own hot per-author timeline cache (effectively pull, but cached). Reading a celebrity author timeline once costs the same regardless of follower count.

### B. Sharding & Hot Keys

**Tweet store**: partition by user_id (author). Hot author = hot partition (Elon, BTS).
- Replicate hot author partitions to more nodes.
- Pre-warm cache on celebrity tweets.

**Timeline cache**: partition by user_id (consumer). Even distribution, no hot keys (no one user reads 1000× more).

**Fan-out workers**: shard by author_id. Sequential dispatch keeps order per author.

### C. Ranking (Lightly)

Reverse-chronological is the default. Real Twitter overlays a ranker:

```
score = w1*recency + w2*engagement_pred + w3*author_affinity + w4*topic_match + ...
```

Train a model on labeled engagement signals. Run inference at read time or pre-compute candidate scores.

Tradeoff: heavier reads, but better engagement. Hybrid: fetch top 200 candidates by recency, score with ranker, return top 20.

### D. Failure Modes

| Failure | Effect | Mitigation |
|---|---|---|
| Redis timeline cache down | Reads fall back to pull-from-Cassandra | Slower (multi-partition fetch); accept temporary degraded UX |
| Fan-out worker behind | Delayed timeline updates | Backpressure to publisher; alert on Kafka lag |
| Cassandra hot partition | Author can't tweet | Pre-shard celebrity author tables; rate-limit per-author |
| Cross-region delay | Stale timelines after a follow | "Follow" UI eventually-consistent badge |

## 7. Bottlenecks & Scaling

| At 10× (3B DAU, 70K WPS) | Bottleneck | Fix |
|---|---|---|
| Fan-out writes | 70K × 200 avg followers = 14M writes/sec into timelines | More Redis shards; batch writes; tighter celebrity threshold |
| Read RPS 1.75M | Redis | Replica reads; CDN for public profile timelines |
| Storage 1 PB/year | Cassandra | Already designed for it; add nodes |
| Celebrity tail | Fan-out time grows | Lower celebrity threshold; pull-only for top-1% authors |

## 8. Tradeoffs & Alternatives

| Decision | Picked | Alt | Why |
|---|---|---|---|
| Hybrid fan-out | ✅ | Push-only | Push-only fails at celebrity scale |
| Redis sorted set | ✅ | Postgres timelines | RAM speed; cap evicts naturally |
| Cassandra for tweets | ✅ | Postgres | Write throughput + time-series shape |
| Kafka for fan-out | ✅ | Direct RPC | Backpressure, retry, replay |
| Eventual consistency | ✅ | Strong | Read SLAs require it; users tolerate seconds-stale |

## 9. Follow-up Questions

- **"How do you handle a user with 100M followers?"** — Hybrid fan-out (above). Tweet is not pushed; readers pull from a per-celebrity timeline cache when assembling their feed.
- **"What if a user follows 5K accounts?"** — Their home timeline is still O(1) to read (Redis sorted set). Fan-out cost is paid at write time (5K extra writes per tweet from anyone they follow), absorbed by Kafka.
- **"How fast is a new tweet visible?"** — From own user timeline: immediate (write-through). From followers' home timelines: seconds (Kafka + fan-out lag). Acceptable.
- **"How do you back-fill a timeline after follow?"** — On follow event, run a one-off job to merge followee's recent tweets into follower's timeline. Or accept that newly-followed content shows up only via celeb path.
- **"How do you handle deletes?"** — Tombstone in `tweets`. Background sweep removes from timelines, or check tombstone on read (smarter). Cache eviction by tweet-id tag.
- **"What about a global trending feed?"** — Separate pipeline: stream of all tweets → hashtag/keyword extraction → time-windowed count-min sketch → top-K → cache.
- **"Personalized ranking?"** — Two-stage: candidate gen (recency + follow graph), then ranker. Ranker is its own service.
- **"Multi-region?"** — Region-local Redis caches; cross-region Kafka for fan-out; tolerate few-second cross-region lag.

## Related

- `caching.md` — Redis sorted-set sizing
- `message_queues.md` — Kafka fan-out pattern
- `consistent_hashing.md` — sharding the timeline cache
- [Databases](databases.md) — Cassandra wide-row patterns for feed storage
- [ID generation](id_generation.md) — Snowflake tweet IDs for time-ordered feeds

## Where this connects

- [Caching](caching.md) — pre-computed fan-out feeds cached in Redis for read performance
- [Message queues](message_queues.md) — async fan-out on post creation via Kafka or similar
- [Databases](databases.md) — Cassandra for the feed store, MySQL for user/post metadata
- [Consistent hashing](consistent_hashing.md) — shard user feeds across cache/DB nodes
