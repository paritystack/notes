# Design Typeahead / Autocomplete (Google Search Suggest)

Tests trie at scale, top-K aggregation, freshness vs latency tradeoffs, and personalization.

## 1. Requirements

### Functional
- As user types a prefix, return top suggestions ordered by popularity.
- Suggestions update from real search traffic.
- Optional: personalization (user history, location).
- Spell-tolerant ("acomodation" → "accommodation").
- Multi-language.

### Non-functional
- P99 latency < 100 ms (the whole point — instant suggest).
- 1B queries/day = ~12K QPS avg, 50–100K QPS peak (typing emits many partial queries).
- 100s of millions of unique prefixes.
- Suggestions refresh within hours of trend shifts.

### Out of scope
- The actual search backend.
- Voice input, image suggestion.

## 2. Capacity Estimation

```
Users typing/day      = 500M
Avg keystrokes/query  = 6
Lookups/query         = 6 (one per keystroke; debounce ~150 ms)
Lookups/day           = 500M × 4 effective = 2B
QPS avg               = 23K, peak 100K

Vocabulary:
  Unique queries      = ~500M
  Avg query length    = 25 chars
  Top-K per prefix    = 10
  Number of distinct prefixes ≈ 1B (all up to 30 chars)
  Storage of (prefix → top10):
       1B × (30B prefix + 10 × 30B suggestions) ≈ 330 GB

Update rate:
  ~12K WPS of completed queries to log
  → batch aggregate hourly/daily
```

## 3. API Design

```
GET /v1/suggest?q=acco&loc=US&lang=en&user_id=
  Resp: {
    suggestions: [
      { text: "accommodation", score: 0.92 },
      { text: "accountant",    score: 0.81 },
      ...
    ]
  }
```

Hot path. Cache liberally. POST only if you need to send rich context (rarely).

## 4. Data Model

### Online (read) store
```
prefix_index (Redis hash or in-memory trie)
  key   = normalized prefix ("acco")
  value = sorted list [(suggestion, score)] capped at 10
```

OR

```
trie nodes (compact in memory)
  per node: char, top-10 cached, children pointers
```

### Offline (aggregation) store
```
query_log (Kafka → ClickHouse / BigQuery)
  ts, normalized_query, user_id, locale, click_result_id

query_counts (rebuild target)
  query, count, score, last_seen
```

## 5. High-Level Architecture

```
   Client ──► API GW ──► Suggest svc (stateless)
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Prefix store (Redis /  │
                  │ shared in-mem replicas)│
                  └────────────┬───────────┘
                               │
                               ▼ (offline build)
                  ┌────────────────────────┐
                  │ Aggregation pipeline    │
                  │ Kafka → Flink/Spark     │
                  └────────────┬───────────┘
                               │
                               ▲
   Search box → query log → Kafka
```

Two loops: **fast read** (suggest svc + in-memory trie) and **slow write** (aggregation + rebuild + warm).

## 6. Deep Dives

### A. Data Structure: Trie vs Sorted Map

| Structure | Lookup | Memory | Update |
|---|---|---|---|
| **Trie** | O(prefix_len) | Lots (one node per char) | In-place |
| **Trie + per-node top-K** | O(prefix_len) | Larger but eliminates per-query aggregation | In-place |
| **Hash map (prefix → top-K)** | O(1) | Coarsely huge (every prefix duplicated) | Rebuild |
| **Sorted suffix array** | O(log N) | Compact | Rebuild |

**Recommended for an interview**: trie with top-K cached at each node.

Lookup:
```
traverse trie char-by-char following prefix
arrive at node for "acco"
return node.top_k    # pre-baked
```

No per-query aggregation. Cost is one trie traversal + a copy.

Per-node top-K is updated offline by the aggregation pipeline.

### B. Building Top-K Per Prefix

Naively: for every prefix, scan all queries starting with it, sort by score, take top K. O(prefixes × queries) — dead.

Efficient: **build bottom-up**.

```
1. Aggregate query counts (last 24h or weighted EWMA) over Kafka log.
2. Insert each query into trie with its score.
3. DFS upward: each node's top_k = merge children's top_k + queries ending at node, take top K.
```

Merge at each node is K-way merge, O(K × children). Total build O(total_chars × K) ≈ minutes for ~500M queries.

Build runs hourly or daily in a job, output published to read tier.

### C. Distribution / Sharding

In-memory trie at scale (~30 GB working set) — too big for single box if including all locales.

**Partition by prefix-first-N-chars**:
- Shard 1: prefixes starting with "a-c".
- Shard 2: "d-f"...
- Routing layer hashes by first character (or first 2) → shard.

Or partition by **locale + first char** (en-US "a-c", en-US "d-f", ...).

Each shard fits in memory. Replicate 3× for HA.

### D. Personalization

Two layers:
1. **Global suggestions** (the trie above).
2. **Personalized re-rank** at query time:
   - Fetch user's recent queries (Redis per-user top-N).
   - Boost suggestions matching user history or category.
   - Boost based on locale + recent location.

```
suggestions = global_trie.lookup(prefix)
user_boost  = user_profile_cache.get(user_id)
re-rank suggestions by (global_score + alpha * personalization_score)
```

Adds ~5 ms; worth it.

### E. Freshness vs Stability

Trending searches need fast incorporation; stable queries need cache effectiveness.

```
Two-tier:
  base_trie:   updated daily, large historical window (weight by recency)
  delta_trie:  updated every 5–15 min, last hour's traffic
  merge at lookup time (small additional cost)
```

Delta layer captures news/trending; base layer captures stable head queries.

### F. Spell Tolerance

- **Edit distance index**: precompute candidates within edit distance 1–2 of common queries.
- **Phonetic hash** (Metaphone, Soundex): match by sound.
- **Symspell**: delete-only index for O(1) candidate generation up to edit distance 2.

When base lookup returns < K, fall back to fuzzy lookup at edit distance 1, then 2.

### G. Caching

- **Edge cache** (CDN): cache by (prefix, locale) for non-personalized prefixes ~60s TTL. Big hit-rate win on common prefixes ("ho", "fac"...).
- **In-process**: LRU of last N prefixes served — saves trie traversal.

### H. Privacy

Query logs are sensitive. Mitigations:
- Strip PII (UA, IP) from aggregation pipeline.
- Anonymize/sample at logging tier.
- Drop low-count queries (k-anonymity: only suggest if ≥ k users queried in window).
- Honor "incognito" / DNT — don't log.

## 7. Bottlenecks & Scaling

| At 10× (1M QPS) | Bottleneck | Fix |
|---|---|---|
| Read RPS | Per-shard CPU on trie traversal | More replicas; CDN for non-personalized |
| Build time | Single Flink job | Parallelize build per-shard |
| Memory growth | Trie size | Compact trie (radix/Patricia); UTF-8 compression |
| Personalization fetch | Redis | Cache user profile next to suggest svc |

## 8. Tradeoffs & Alternatives

| Decision | Picked | Alt | Why |
|---|---|---|---|
| Trie | ✅ | Hash map per prefix | Memory efficient; natural prefix lookup |
| Pre-baked top-K | ✅ | On-demand sort | Latency win |
| Sharded by first char | ✅ | Hash sharding | Range queries on prefix are natural |
| Hourly delta | ✅ | Real-time updates | Cost vs benefit; minutes-fresh is sufficient |
| Base + delta tiers | ✅ | Single | Trend incorporation without rebuilding all |

## 9. Follow-up Questions

- **"How do you handle 'a' (extremely hot prefix)?"** — Cache result aggressively; precompute it; high hit rate at edge. Even with millions of QPS for "a", payload is tiny.
- **"What about non-Latin scripts (CJK, Arabic, RTL)?"** — Trie keys are Unicode codepoints, not bytes. Normalize via NFKC. Separate tries per locale.
- **"How do you debounce?"** — Client-side: 100–150 ms after last keystroke before issuing request. Server-side: dedup in-flight requests per session.
- **"How do you avoid serving offensive suggestions?"** — Block-list filter post-aggregation; classifier for novel offensive completions; manual review for high-traffic items.
- **"What if Redis goes down?"** — Read-replica failover; fallback to a smaller compiled trie loaded at process startup. UX: degraded suggest, never error.
- **"How would you do A/B test suggestion ranking?"** — Treatment assignment per user; serve algorithm A or B based on user-hash modulo. Log impressions; analyze CTR offline.
- **"How does this differ from search?"** — Search has different latency budget (~300ms vs 100ms), more aggressive rewriting, deep ranking. Typeahead is shallow + fast + presentation-only.
- **"Multi-region?"** — Per-region trie replicas; users routed to nearest region; locale-specific shards in each.

## Related

- `caching.md` — edge caching of suggestions
- `data_structures/inverted_index.md` — adjacent search-side structure
- `consistent_hashing.md` — sharding the read tier
- `message_queues.md` — Kafka log → aggregation
- `design_news_feed.md` — similar two-loop (read/write) pattern
