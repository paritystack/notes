# Content Delivery Networks (CDN)

A geographically distributed network of edge servers that cache content close to users. The single biggest lever for global read latency.

## When to Use a CDN

| Use case | Why |
|---|---|
| Static assets (JS, CSS, images, video) | Cache hit at edge → no origin trip |
| Software downloads | Bandwidth offload to edge |
| API responses (GET, cacheable) | Trim 100+ ms off global P99 |
| Live streaming | HLS/DASH segments cached and prefetched |
| Dynamic content with edge compute | Personalize at the edge instead of origin |

## When NOT to Use a CDN
- Personalized POST traffic with no caching potential.
- Strong-consistency reads (CDN is eventually consistent).
- Tiny user base in one region — added cost, no latency win.

## Architecture

```
                          ┌──────────────────────────┐
       User (Tokyo)       │     CDN Edge (Tokyo)     │
            │             │   ┌──────────────────┐   │
            │ 5 ms        │   │  Object cache    │   │
            └────────────►│   │  (NVMe + RAM)    │   │
                          │   └────────┬─────────┘   │
                          │            │ miss        │
                          │            ▼             │
                          │   ┌──────────────────┐   │
                          │   │  Regional shield │   │  (Singapore)
                          │   │  (parent tier)   │   │
                          │   └────────┬─────────┘   │
                          │            │ miss        │
                          └────────────┼─────────────┘
                                       │ 150 ms
                                       ▼
                              ┌────────────────┐
                              │  Origin (US)   │
                              │   S3 / app     │
                              └────────────────┘
```

### Tiers
1. **Edge PoP** (Point of Presence): closest to user. Hundreds globally.
2. **Regional shield / parent**: aggregates misses from multiple PoPs in a region to reduce origin load.
3. **Origin**: source of truth. Usually app server or object store.

**Why shield?** Without it, every cold PoP independently misses to origin. A flash crowd in a new region can DDOS origin. Shield collapses N PoP misses into 1 origin hit.

## Caching Mechanics

### Cache key
Default = `URL + Host + selected headers`. Tune carefully:
- **Vary on `Accept-Encoding`**: separate gzip vs brotli vs uncompressed.
- **Vary on `Accept-Language`**: only if you serve localized content (else cache fragments).
- **Vary on cookies**: kills cache hit rate. Strip non-essential cookies at edge.

### Cache control
```
Cache-Control: public, max-age=86400, s-maxage=604800, stale-while-revalidate=3600
```

| Directive | Effect |
|---|---|
| `max-age` | Browser cache TTL |
| `s-maxage` | CDN-only TTL (overrides max-age for CDN) |
| `stale-while-revalidate` | Serve stale while refreshing in background |
| `stale-if-error` | Serve stale on origin failure |
| `no-store` | Never cache |
| `private` | Browser only, not CDN |

### TTL strategy
- **Immutable assets** (`/static/app.abc123.js`): `max-age=31536000, immutable`.
- **HTML**: `s-maxage=60, stale-while-revalidate=300` — short freshness, soft fallback.
- **API JSON** (cacheable): `s-maxage=10` — even 10s deflects huge load.

## Invalidation (the Hard Part)

> "There are only two hard things in CS: naming and cache invalidation."

### Strategies
| Strategy | Latency | Cost | Use when |
|---|---|---|---|
| **TTL expiry (passive)** | Slow (TTL) | Free | Default; can tolerate staleness |
| **Versioned URLs** | Instant | None | Immutable assets (hash in filename) |
| **Purge by URL** | seconds | $$ | Specific invalidations |
| **Purge by tag** | seconds | $$ | Bulk invalidation (e.g., all SKU 123) |
| **Soft purge** | seconds | $ | Mark stale, revalidate on next hit |
| **Hard purge** | minutes | $$$ | Compliance: remove now |

**Tag-based purge** is the goldilocks: tag a response with `cache-tag: product-123, category-foo`, purge by tag when product or category changes.

### Pitfall: thundering herd on invalidation
Purging a hot URL across 500 PoPs → all of them miss simultaneously → origin gets hammered. Mitigations:
- Use shield tier to collapse misses.
- Use **request coalescing** at edge: 1000 concurrent misses for the same URL → 1 origin fetch.
- Soft purge so PoPs serve stale-while-revalidating.

## Push vs Pull CDN

| Mode | How |
|---|---|
| **Pull** | Edge fetches from origin on first request. Default. |
| **Push** | You upload content to CDN. Used for predictable hot content (releases, video VOD). |

Most CDNs are pull. Push makes sense when you know what will be hot and want to prewarm.

## Edge Compute

Modern CDNs (Cloudflare Workers, Fastly Compute@Edge, CloudFront Functions, Lambda@Edge) run code at PoPs.

Use cases:
- **A/B test routing**: pick variant at edge, no origin trip.
- **Header rewriting / auth checks**: reject bad tokens before origin.
- **Personalization**: small payload mutations on cached HTML.
- **Geographic routing**: send EU users to EU backend.

Tradeoff: cold start latency, limited runtime (V8 isolates, WASM, restricted libraries).

## Video Streaming via CDN

```
Encoder → Manifest (HLS .m3u8 / DASH .mpd) + Segments (4-10s .ts files)
              │
              ▼
         Object storage (origin)
              │
              ▼
        Shield (parent tier, prewarmed)
              │
              ▼
        Edge PoPs (serve segments)
              │
              ▼
            Players
```

**Why segment-based?** Each 4–10 s segment is a separately cacheable URL. Players request the next segment as they play. Live latency is dominated by segment length × playlist depth (typically 30 s). Low-latency HLS (LL-HLS) chunks segments further for ~3 s glass-to-glass.

**Hot content amplification:** 1 popular live event → millions of viewers, but only one segment per chunk. Hit rate at edge approaches 100%.

## Security & DDoS

CDNs absorb traffic by virtue of capacity. Common features:
- **L3/L4 DDoS scrubbing**: drop SYN floods, amplification attacks at edge.
- **L7 WAF**: regex/ML rules, rate limit by IP/path.
- **Bot management**: JS challenge, hCaptcha, fingerprinting.
- **mTLS to origin**: prevent origin-bypass attacks.
- **Token-signed URLs**: `?expires=…&sig=…` for premium content.

## Performance Optimizations

| Trick | Effect |
|---|---|
| HTTP/3 (QUIC) at edge | Sub-RTT handshake, no head-of-line blocking |
| Brotli compression | ~20% smaller than gzip for text |
| Image format negotiation | Serve WebP/AVIF based on Accept |
| Resize at edge | Single origin asset → many viewport-specific renditions |
| Connection coalescing | One TCP conn from edge serves many users |

## Provider Landscape

| Provider | Strengths |
|---|---|
| **Cloudflare** | Largest network, generous free tier, Workers |
| **Akamai** | Enterprise, oldest, video |
| **Fastly** | VCL programmability, instant purge |
| **AWS CloudFront** | Tight S3/Lambda integration |
| **Google Cloud CDN** | Tied to GCP load balancer |
| **Bunny / KeyCDN** | Cheap, simple |

## Interview Cheat: When the Word "Global" Appears

If the interviewer says "users worldwide" → mention CDN within the first minute. It's the cheapest latency win and signals you think about geography.

Expected breakdown:
1. **What's cacheable?** Static + cacheable GET. Note what isn't (POST, personalized).
2. **Cache key + TTL strategy.** Versioned filenames, short TTL on HTML.
3. **Invalidation.** Tag-based purge.
4. **Origin protection.** Shield tier, request coalescing.
5. **Edge compute** for personalization without origin trips.

## Common Interview Gotchas

- **"Why not just put nginx in every region?"** — You'd be reinventing a CDN poorly. Real CDNs have 200+ PoPs, anycast routing, peered networks, DDoS scrubbing.
- **"What if the user is in a region with no PoP?"** — Anycast routes them to the closest available PoP, still wins.
- **"How does CDN handle uploads?"** — They don't, really. Direct-to-origin or use edge upload + async to origin (CloudFront does this).
- **"Cache stampede protection?"** — Request coalescing at edge. Stale-while-revalidate. Don't bypass cache during refresh.

## Related

- `caching.md` — caching principles that apply to CDN tiers
- `load_balancing.md` — anycast and global load balancing
- `design_video_streaming.md` — heavy CDN case study
- `scalability.md` — CDN as a horizontal-scaling primitive for reads
