# Interview Framework

A repeatable playbook for system-design interviews. Spend the first ~5 min on requirements, the next ~5 on estimation, ~10 on HLD, ~15–20 on deep dives, ~5 on bottlenecks & wrap-up.

## The 7 Phases

```
┌──────────────┐   ┌─────────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌──────┐
│  Clarify &   │ → │  Capacity   │ → │   API   │ → │  Data   │ → │   HLD    │ → │ Deep Dive  │ → │ Wrap │
│  Requirements│   │  Estimation │   │  Design │   │  Model  │   │ Diagram  │   │  + Scale   │   │  Up  │
└──────────────┘   └─────────────┘   └─────────┘   └─────────┘   └──────────┘   └────────────┘   └──────┘
   ~5 min            ~5 min            ~5 min       ~5 min         ~10 min         ~15-20 min       ~5 min
```

## Phase 1: Clarify & Requirements (~5 min)

**Goal:** convert a vague prompt ("design Twitter") into a concrete contract.

### Always ask
- **Scope:** which features? Cut ruthlessly. ("Should we include DMs? Trending? Ads?")
- **Scale:** DAU, peak QPS, read/write ratio.
- **SLAs:** P99 latency target, availability (99.9 vs 99.99).
- **Geography:** single region vs global. Data residency?
- **Consistency:** strong vs eventual? Per-feature.

### Output (write this on the board)

```
Functional:
  - Post a tweet (text only, 280 chars)
  - Follow a user
  - View home timeline (latest first)

Non-functional:
  - 200M DAU, 100:1 read:write
  - Timeline P99 < 200ms
  - Eventual consistency OK for timeline
  - 99.99% availability
```

**Pitfall:** Don't dive into design before pinning scope. If you don't, you'll be redesigning halfway through.

## Phase 2: Capacity Estimation (~5 min)

See `estimation_cheatsheet.md` for the numbers. Aim for **QPS, storage, bandwidth**.

### Template
```
DAU            = 200M
Active hours   = 200M * 8h / 24h ≈ 67M concurrent
Writes/day     = 200M * 2 tweets ≈ 400M ≈ 4,600 WPS (avg)
Peak WPS       = 4,600 * 3 ≈ 14K WPS
Reads/day      = writes * 100 = 40B reads ≈ 460K RPS avg, 1.4M peak

Storage:
  tweet size   = 280B text + 200B metadata ≈ 500B
  per day      = 400M * 500B = 200 GB/day
  per year     = 73 TB/year (text only)

Bandwidth:
  write in     = 14K * 500B = 7 MB/s
  read out     = 1.4M * 1KB = 1.4 GB/s
```

**Pitfall:** don't get arithmetic-perfect. Order of magnitude is the goal.

## Phase 3: API Design (~5 min)

Define 3–5 endpoints that exercise the core feature set. REST or gRPC — both fine. Show request/response shapes.

```
POST /v1/tweets
  Body: { text: string, media_ids: [string] }
  Response: { tweet_id, created_at }

GET /v1/timeline?user_id=&cursor=&limit=20
  Response: { tweets: [...], next_cursor }

POST /v1/follow
  Body: { follower_id, followee_id }
```

**Tradeoff:** REST = browser-friendly, cacheable. gRPC = compact, streaming, internal-service-friendly. GraphQL = client-driven payloads but harder to cache.

## Phase 4: Data Model (~5 min)

Sketch the entities. Don't write full DDL — just key fields and which store.

```
users (PostgreSQL):
  user_id PK, handle, name, created_at

tweets (Cassandra, partition by user_id):
  tweet_id PK (Snowflake), user_id, text, created_at

follows (Cassandra):
  (follower_id, followee_id), created_at
  Secondary by followee_id for fan-out lookups

timeline_cache (Redis, key = user_id):
  sorted set: tweet_id → created_at (cap 800)
```

**Pitfall:** picking SQL for everything. Mix stores — relational for users, wide-column for tweets, KV for timelines.

## Phase 5: High-Level Diagram (~10 min)

Draw boxes + arrows. Label the protocol on each edge. Show client → LB → service tier → data tier.

```
 [Mobile] [Web]
     │       │
     ▼       ▼
  ┌──────────────┐
  │ API Gateway  │ (auth, rate-limit, routing)
  └──────┬───────┘
         │
   ┌─────┴──────┬──────────────┐
   ▼            ▼              ▼
 [Tweet svc] [Timeline svc] [Follow svc]
   │            │              │
   ├──> Kafka (fan-out events)
   ▼            ▼              ▼
 [Cassandra]  [Redis]       [Cassandra]
```

Narrate as you draw: "Write goes to Tweet svc, persists to Cassandra, emits a fan-out event to Kafka. Timeline svc consumes, pushes into Redis sorted sets for each follower."

## Phase 6: Deep Dives (~15–20 min)

The interviewer will steer here. Have 2–3 "I can dive on this" candidates ready per problem. Common deep-dive targets:

| Topic | Trigger |
|---|---|
| Fan-out strategy | Anything with feeds/timelines |
| Sharding key | Anything write-heavy |
| Cache invalidation | Anything read-heavy |
| Geo-replication | Anything global |
| Hot-key/celebrity | Anything with skewed access |
| Idempotency | Anything mutating |
| Backpressure | Anything streaming |

**Structure each deep dive:**
1. Restate the problem in one line.
2. Two or three options.
3. Tradeoff matrix.
4. Your pick + why.

## Phase 7: Bottlenecks & Wrap-up (~5 min)

Walk through the diagram and call out **what breaks first at 10×**.

- DB writes? → shard, batch, async.
- Hot key (celebrity)? → hybrid fan-out (push for normal, pull for celebrity).
- Cache evictions? → bigger cluster, smarter eviction.
- Cross-region writes? → conflict resolution (LWW, CRDTs).

End with **what you'd do next** if you had more time. Shows maturity.

## Cheat Sheet: What Interviewers Score

| Signal | Good | Bad |
|---|---|---|
| Drives the conversation | Proposes, then asks "thoughts?" | Waits for prompts |
| Trade-offs | "X is faster but Y is consistent — picking X because…" | Picks one option, never justifies |
| Estimation | Sanity-checks decisions with numbers | Hand-waves scale |
| Depth | Names specific tech + why (Cassandra LSM for write-heavy) | "Some NoSQL DB" |
| Edge cases | Brings up failure, hot keys, eventual consistency | Only happy path |
| Time mgmt | Hits all phases | Burns 25 min on requirements |

## Common Anti-Patterns

- **Solution before problem:** "We'll use Kafka and Redis" before scope is set.
- **Buzzword bingo:** Naming tech with no justification.
- **Monolithic deep-dive:** Spending all time on one component, leaving others undefined.
- **Ignoring scale:** Designing for 10K users when the prompt said 100M.
- **No tradeoffs:** Every decision is "the right way." There isn't one.
- **Skipping data model:** Going straight from API to architecture without data shape.

## Templates Reused Across Case Studies

Each case study in this directory follows the same skeleton. If you've cracked one, you can crack the next:

1. **Requirements** — functional + non-functional
2. **Capacity Estimation** — QPS, storage, bandwidth
3. **API Design** — 3–5 endpoints
4. **Data Model** — entities + store choice
5. **High-Level Architecture** — boxes + arrows
6. **Deep Dives** — 2–3 hard parts
7. **Bottlenecks & Scaling** — what breaks at 10×
8. **Tradeoffs & Alternatives** — what we picked, what we didn't
9. **Follow-up Questions** — anticipated interviewer probes

## Related

- `estimation_cheatsheet.md` — numbers to memorize
- `scalability.md` — horizontal/vertical scaling primitives
- `design_patterns.md` — patterns interviewers expect you to name-drop
- `distributed_systems.md` — CAP, consensus, replication background
