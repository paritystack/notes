# Design a Video Streaming Service (YouTube / Netflix)

Tests CDN strategy, adaptive bitrate streaming, transcoding pipelines, metadata at scale, and recommendation/discovery.

## 1. Requirements

### Functional
- Upload videos (creator path).
- Watch videos on web/mobile/TV.
- Browse, search, recommend.
- Comments, likes (lightweight, not the focus).
- Live streaming (briefly, in deep dives).

### Non-functional
- 2B MAU, 500M DAU.
- 1B hours watched/day.
- 500 hours of video uploaded/minute.
- Global; sub-2s start time (P99).
- 99.99% playback availability.
- Adaptive bitrate; smooth quality under varying network.

### Out of scope
- DRM details, ads, monetization.

## 2. Capacity Estimation

```
Watched hours/day  = 1B → 720M concurrent at peak (1B / 24 * 8 peak hr factor 3)
Concurrent viewers = ~50M peak
Bitrate            = 1080p ≈ 5 Mbps; mix avg ≈ 3 Mbps
Peak egress        = 50M × 3 Mbps = 150 Tbps     ← CDN territory
Origin egress      = with 99% CDN hit rate, ~1.5 Tbps to origin

Upload:
  500 hr/min = 8.3 hr/s of new video
  raw bitrate 50 Mbps × 8.3 hr/s × 3600 = ~1.5 TB/s
  After transcoding to multi-bitrate stack: ~5-10× storage

Storage:
  After transcode, per minute uploaded ≈ 100-500 MB across all bitrates
  500 hr/min × 60 × 250 MB ≈ 7.5 TB/min ingested
  ~11 PB/day, ~4 EB/year (raw + transcoded, retained indefinitely)

Metadata DB:
  Video count ~5B
  Per-video metadata ~5 KB → 25 TB
```

## 3. API Design

```
# Upload
POST /v1/uploads
  Body: { title, description, tags, visibility }
  Resp: { video_id, upload_url (presigned), chunk_size }

PUT  {upload_url}?chunk=N    # multipart upload to object store
POST /v1/uploads/:id/complete

# Playback
GET /v1/videos/:id              # metadata
GET /v1/videos/:id/manifest     # HLS/DASH manifest URL
GET /watch/{video_id}/manifest.m3u8   # HLS playlist (CDN)
GET /watch/{video_id}/720p_seg{n}.ts  # video segments (CDN)

# Discovery
GET /v1/home               # personalized feed
GET /v1/search?q=...
GET /v1/videos/:id/related

# Engagement
POST /v1/videos/:id/like
POST /v1/videos/:id/comments
GET  /v1/videos/:id/comments
```

## 4. Data Model

```
videos (Spanner / Vitess / Postgres+Vitess)
  video_id PK, creator_id, title, description, tags[],
  status (uploading|processing|live|removed),
  duration_s, thumbnail_url,
  visibility (public|unlisted|private), created_at

video_assets (per video, multiple)
  video_id, asset_id, type (video|audio|caption|thumbnail),
  storage_url, mime, bitrate, resolution, codec, language

watch_history (Bigtable)
  user_id, ts PK
  video_id, watched_seconds, completed

likes (KV)
  user_id + video_id PK

comments (Cassandra, partition by video_id)
  video_id, comment_id (Snowflake) PK
  user_id, body, replied_to, created_at

view_counts (Redis HyperLogLog per video for unique; counter for total)
  → reconciled to Bigtable hourly
```

## 5. High-Level Architecture

```
              ┌─── Viewers ──────────────────────────────┐
              │                                          │
              ▼                                          │
   ┌─────────────────────┐   manifest (URLs)             │
   │ Video metadata API  │ ◄──────────────────────────── │
   └──────────┬──────────┘                               │
              │                                          │
              ▼                                          │ segments
       Postgres/Spanner                               (HLS/DASH)
                                                          ▼
                                              ┌──────────────────┐
                                              │ CDN edge PoPs    │
                                              └────────┬─────────┘
                                                       │ miss
                                                       ▼
                                              ┌──────────────────┐
                                              │ Origin: object   │
                                              │  store (S3/GCS)  │
                                              └──────────────────┘
                                                       ▲
                                                       │
              ┌─── Uploaders ──────────────────────────┘
              │  multipart upload → object store
              ▼
       Upload service → Object store (raw)
                              │
                              ▼
                  ┌─────────────────────────┐
                  │  Transcoding pipeline    │
                  │  Kafka → workers → DAG   │
                  └─────────────┬────────────┘
                                ▼
                       Object store (renditions)
                                │
                                ▼
                       Metadata DB updated (status=ready)
                                │
                                ▼
                       Search/recommend indexing pipelines
```

## 6. Deep Dives

### A. Upload & Transcoding Pipeline

Upload:
1. Client requests upload URL → presigned multipart to object store.
2. Client uploads chunks in parallel; resumes on failure.
3. `complete` triggers metadata update + enqueues transcoding job.

Transcoding (the heavy lift):

```
Raw upload (e.g., 4K HEVC, 50 Mbps)
       │
       ▼
   Chunker: split into 4-8 s segments, parallelize
       │
       ▼
   Job queue (Kafka)
       │
       ▼
   Worker fleet:
     for each (segment, target_bitrate, codec, container):
       transcode → upload → record asset
       
   Targets:
     240p / 144p (mobile, low-bw)  → H.264
     360p / 480p                    → H.264
     720p / 1080p                   → H.264 / VP9
     1440p / 2160p (4K)            → AV1 (better compression)
     audio                          → AAC / Opus
     captions                       → WebVTT
     thumbnails / preview gif       → JPEG / WebP
```

Per-segment parallelism → huge throughput. A 1-hour video transcodes in minutes across hundreds of workers.

**Storage by codec**:
- H.264 universal compat but big files.
- VP9 ~30% smaller, less compat.
- AV1 ~50% smaller, slow to encode (expensive CPU), modern devices only.
- Strategy: H.264 ladder always; VP9/AV1 for popular content (warm renditions on demand).

### B. Adaptive Bitrate Streaming (ABR)

**HLS** (Apple) and **DASH** (everyone else):
- Master manifest lists variants (different bitrates/codecs).
- Each variant has its own playlist of 4-10 s segments.
- Player downloads, monitors throughput, switches variant per segment.

```
master.m3u8:
  #EXT-X-STREAM-INF:BANDWIDTH=400000,RESOLUTION=360x240
  240p.m3u8
  #EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=1280x720
  720p.m3u8
  ...
```

```
720p.m3u8:
  #EXTINF:6.0, seg0.ts
  #EXTINF:6.0, seg1.ts
  ...
```

Player heuristics (rate-based, buffer-based, hybrid like BOLA). Switch up when buffer high + bandwidth headroom; switch down before buffer empty.

**Why 4–10 s segments?**
- Shorter → faster switches, more requests, more overhead.
- Longer → fewer requests, slower switches.
- 4 s is the sweet spot for VOD; 1–2 s for low-latency live.

### C. CDN Strategy

```
50M viewers × 3 Mbps = 150 Tbps peak. 
Origin would die.
Mitigation: CDN with 99%+ hit rate on segments.
```

Tiers (see `cdn.md`):
1. **Edge** (1000s of PoPs): hot segments per region.
2. **Regional shield**: collapses misses, prewarms popular videos.
3. **Origin**: object store; rarely hit for popular content.

**Tactics**:
- Multi-CDN: Akamai + CloudFront + custom (Open Connect / YouTube cache). Steer traffic by per-PoP latency/cost.
- **Prewarm on publish**: popular creator → push variants to top regions before any viewer requests.
- **ISP boxes**: Netflix's Open Connect / YouTube caches placed *inside* ISP networks. Cheaper for ISP, faster for users.

### D. Personalization & Recommendations

Two-stage funnel:
1. **Candidate generation**: collaborative filter (user × video matrix factorization), recent watches, channel subscriptions, trending. Produces ~thousand candidates.
2. **Ranker**: deep model predicting watch-completion probability, scored over candidates. Pick top ~20 for the row.

Pipeline:
- Offline: train daily on Spark/TFX.
- Online: feature store (Redis/Bigtable) per-user features, per-video features.
- Inference at request time via TensorFlow Serving / TorchServe.

**Cold start**: trending + topic-based fallback.

### E. Live Streaming (Quick Look)

Differences from VOD:
- Real-time encoder ingest (RTMP, SRT, WebRTC).
- Segments produced on-the-fly; manifest grows.
- Lower latency target (5–15s VOD-style; 1–3s LL-HLS; <1s WebRTC).
- DVR window (rewind buffer of last N minutes).

Architecture:
```
Encoder → Ingest endpoint → Transcoder (live ladder) →
  Origin packager (HLS/DASH) → CDN → Viewers
```

### F. View Counts (Harder Than It Looks)

Naive: increment counter on play → trivial fraud, brittle.

Real approach:
1. **Player heartbeats** every 10–30 s with watched_time.
2. **Stream pipeline** (Kafka → Flink): aggregate per-video per-window.
3. **Bot/abuse filtering**: IP rate, UA patterns, ML scoring.
4. **Approximate counter** (HyperLogLog) for unique views per video per day.
5. **Reconciled hourly** into the metadata DB; UI shows cached "1.2M views" approximated; precise count updated periodically.

Public count lags reality but is robust.

## 7. Bottlenecks & Scaling

| At 10× | Bottleneck | Fix |
|---|---|---|
| 1.5 Pbps egress | CDN capacity | More PoPs; better hit ratio; ISP caches |
| Transcoding fleet | CPU | GPU/ASIC accelerators (NVENC, custom silicon) |
| Recommendation latency | Ranker infra | Pre-compute candidate sets; cache top-K per user |
| Search indexing | Elasticsearch | Sharded ES cluster; offline rebuilds |
| Hot video flash | Origin meltdown | Pre-warm + regional shield + request coalescing |

## 8. Tradeoffs & Alternatives

| Decision | Picked | Alt | Why |
|---|---|---|---|
| HLS+DASH | ✅ | RTMP | Browser/device support, CDN-friendly |
| Pre-bake ladder | ✅ | Just-in-time per-bitrate | Pre-bake: predictable cost. JIT: cheaper for cold content. Real systems do both. |
| Object store + CDN | ✅ | File-system origin | Scale, durability, cost |
| Multi-CDN | ✅ | Single | Failure isolation; pricing leverage |
| Approx view counts | ✅ | Strict | Acceptable UX; massive cost win |

## 9. Follow-up Questions

- **"How do you store 4 EB?"** — Object store with erasure coding (e.g., 10+4 Reed-Solomon → 1.4× overhead vs 3× replication). Tier by access frequency: hot to NVMe, cold to HDD or tape.
- **"How fast is upload-to-watchable?"** — Few minutes typical: chunked transcode runs in parallel. Some platforms expose a "preview" rendition (240p only) within seconds and add higher bitrates over time.
- **"How do you handle copyright (Content ID)?"** — Fingerprint each upload, compare to rights-holder DB; flag or remove on match. Massive fingerprint index, async job.
- **"How does Netflix differ from YouTube?"** — YouTube optimizes for UGC and long-tail discovery (huge cold corpus). Netflix has small (~10K titles) curated catalog, optimizes recommendation depth and pre-positioning (Open Connect literally caches the entire catalog at ISP edge).
- **"How do you serve 4K to a slow connection?"** — You don't. ABR drops to 720p/480p. UI may hide 4K option if estimated bandwidth too low.
- **"How do you handle live for the Super Bowl (concurrent ~10M)?"** — Multi-CDN, pre-warmed PoPs, shield tier, single ingest with multiple egress paths. Capacity reserved with CDN providers in advance.
- **"What if a creator goes viral and CDN doesn't have their videos cached anywhere?"** — Request coalescing at shield + edge collapses N misses into 1 origin fetch; first viewer waits ~hundreds of ms, rest hit cache.
- **"DRM?"** — Widevine / FairPlay / PlayReady. Encrypt segments; player fetches license per session; license server validates entitlement (subscription, geography).

## Related

- `cdn.md` — heavy detail on the read path
- `caching.md` — multi-tier caching
- `message_queues.md` — transcoding pipeline (Kafka)
- `databases.md` — Spanner vs Bigtable choice
- `design_news_feed.md` — recommendation/feed parallels
