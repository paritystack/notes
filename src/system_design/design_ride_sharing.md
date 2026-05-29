# Design a Ride-Sharing Service (Uber / Lyft)

Tests geospatial indexing, real-time matching, surge pricing, and high-frequency location updates.

## 1. Requirements

### Functional
- Rider requests a ride with pickup + drop-off.
- System finds the nearest available driver.
- Both parties see live location during the trip.
- Pricing computed (base + distance + surge).
- Trip state machine: requested → matched → picked-up → completed.
- Driver app sends location every few seconds.

### Non-functional
- 10M DAU, 5M trips/day, 500K concurrent drivers online.
- Driver location updates every 4 s.
- Match latency P99 < 3 s.
- Geographically distributed; ~50 cities.
- 99.99% availability for matching path.

### Out of scope
- Payments (separate service), fraud, ratings.

## 2. Capacity Estimation

```
Active drivers (peak)  = 500K
Location updates       = 500K / 4s = 125K updates/s
                         (mostly absorbed by edge tier)

Trips/day              = 5M → ~60/s avg, peak ~500/s
Match operations       = 500/s × geo lookups
Storage per trip       = 2 KB metadata + GPS trace (~10 KB) → 12 KB
                         5M × 12 KB = 60 GB/day → 22 TB/year

Bandwidth from drivers = 125K × 200 B = 25 MB/s (manageable)
```

## 3. API Design

```
# Driver
WS: { type:"location", lat, lng, heading, speed, ts }
WS: { type:"available" | "unavailable" }
WS receives: { type:"match", trip_id, rider, pickup }

# Rider
POST /v1/trips
  Body: { pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, vehicle_type }
  Resp: { trip_id, status:"matching" }
WS receives: { type:"matched", driver, eta }
WS receives: { type:"driver_location", lat, lng }

POST /v1/trips/:id/cancel
GET  /v1/trips/:id
GET  /v1/eta?lat=&lng=     # for "nearest car" preview
```

## 4. Data Model

```
drivers (Postgres)
  driver_id PK, name, vehicle_type, rating, status (offline|available|on_trip)

driver_locations (Redis / in-memory grid)
  key = geohash_cell or h3_cell
  value = sorted set of driver_ids with last_update_ts
  TTL on driver entry ~30s (drops stale)

trips (Cassandra; partition by city_id, cluster by trip_id)
  trip_id (Snowflake), city_id, driver_id, rider_id
  pickup_lat/lng, dropoff_lat/lng, status, requested_at, completed_at
  fare, surge_multiplier, distance_km

trip_locations (Cassandra; partition by trip_id)
  trip_id, ts, lat, lng

surge_zones (Redis)
  key = zone_id (geohash prefix)
  value = current_multiplier (1.0-3.0)
```

## 5. High-Level Architecture

```
   ┌── Rider app ──┐                        ┌── Driver app ──┐
   │ WS persistent │                        │ WS, GPS every 4s│
   └───────┬───────┘                        └────────┬────────┘
           │                                         │
       Edge LB (geo-aware, sticky by user_id)
           │                                         │
   ┌───────▼────────┐                       ┌────────▼──────┐
   │ Rider edge tier│                       │ Driver edge   │
   └───────┬────────┘                       └────────┬──────┘
           │                                         │
           ▼                                         ▼
   ┌──────────────┐                       ┌────────────────────┐
   │ Match svc    │◄──── query nearby ───►│ Location svc        │
   │ + Surge svc  │                       │ (geo index in Redis)│
   └──────┬───────┘                       └─────────┬───────────┘
          │                                         │
          ▼                                         ▼
   ┌──────────────┐                       ┌────────────────────┐
   │ Trip svc     │                       │ Trip-locations svc │
   │ (state mach.)│                       │ (writes trace)     │
   └──────┬───────┘                       └─────────┬──────────┘
          ▼                                         ▼
   Cassandra (trips)                         Cassandra (locations)
          │
          ▼
   Kafka (trip events) → Billing, Analytics, ML, Notifications
```

## 6. Deep Dives

### A. Geospatial Indexing

Naive lookup: linear scan over all drivers. O(N) — dead at scale.

**Geohash** (Google's encoding):
- Recursive grid: each char halves the cell.
- 6-char geohash ≈ 1.2 km × 0.6 km cell.
- Lookup nearby: query the cell + 8 neighbors.

**S2 cells** (Google's library):
- Hierarchical, ~30 levels.
- Cells map to a 64-bit integer along a Hilbert curve → range queries possible.

**H3** (Uber's library):
- Hexagonal cells.
- All neighbors equidistant (unlike square geohash).
- Multi-resolution (16 levels).
- Easy "k-ring" expansion: get cell + N rings of neighbors.

**Pick H3** for ride-sharing: hex symmetry matches the matching pattern.

```
Location update from driver:
   cell = h3(lat, lng, resolution=9)   # ~150m cells
   ZADD driver_cells:{cell} score=ts member=driver_id
   EXPIRE per-driver TTL 30s

Query nearby for rider request:
   cell = h3(rider_lat, rider_lng, 9)
   neighbors = k_ring(cell, k=2)   # ~750m radius
   candidates = union(ZRANGE driver_cells:{c}) for c in neighbors
   filter candidates: available, vehicle_type, rating
   compute road distance + ETA to each
   pick best (lowest ETA)
```

Resolution 9 gives ~150m cells; a k=2 ring covers ~750m. For sparse areas (rural), expand to k=4 or more.

### B. Matching Algorithm

Simple "nearest" is wrong. Real matching:

1. **Spatial filter**: candidates within k-ring of pickup.
2. **Eligibility**: vehicle type, driver online, no current trip, not just rejected this trip.
3. **ETA computation**: road-network-aware (not Euclidean). Use cached road graph + traffic.
4. **Scoring**: ETA + driver acceptance rate + rating + fairness (queue position).
5. **Offer**: send to top driver; wait N seconds for accept; on reject/timeout, offer next.

**Batching for efficiency in dense areas**:
- Accumulate trip requests in 1–2 s windows per cell.
- Solve as a bipartite matching (Hungarian algorithm) over batched requests + nearby drivers.
- Optimizes globally: better than greedy 1-at-a-time.

### C. Surge Pricing

Per-zone (geohash prefix or H3 cell at resolution 7 ≈ 5km).

```
For each zone every 30s:
   demand = trips_requested_in_zone_last_5m
   supply = available_drivers_in_zone
   ratio = demand / max(supply, 1)
   if ratio > threshold:
       multiplier = min(3.0, 1.0 + (ratio - threshold) * k)
   else:
       multiplier = 1.0
   set surge_zones:{zone_id} = multiplier
```

- Stream into Kafka → real-time aggregator → Redis update.
- Riders see surge before confirming.
- Smoothing: cap rate of change (don't jump 1.0 → 3.0 instantaneously).

### D. Location Update Firehose

500K drivers × every 4s = 125K updates/s. Don't write every update to durable storage.

```
Driver → WS → Edge → Location svc:
   1. Update Redis geo index (overwrite, TTL refresh)
   2. Push to in-memory ring buffer (last N updates per driver)
   3. Sample 1-in-5 updates → Kafka → Cassandra (trip_locations)
   4. If driver on trip: forward to rider's edge (WS push) for live tracker
```

Cassandra writes 25K/s sampled; full-fidelity in Kafka for replay.

### E. Trip State Machine

```
requested → matching → matched → driver_arriving → in_progress → completed
                  ↓
              cancelled
                  ↓
              no_drivers_found (timeout 30s)
```

Each state transition:
- Idempotent (state action + transition check)
- Emit Kafka event (`trip.matched`, `trip.completed`)
- Update rider + driver via WS
- Downstream consumers: billing, ratings, analytics

Use saga pattern for cross-service compensations (e.g., rollback hold-on-card if trip cancelled).

## 7. Bottlenecks & Scaling

| At 10× (50M drivers, 50M trips/day) | Bottleneck | Fix |
|---|---|---|
| Location updates 1.25M/s | Redis CPU | Shard by city; multiple Redis clusters |
| Match latency in dense cities | Single match worker queue | Shard matchers by H3 cell; parallel solvers |
| Cassandra writes for trip_locations | 250K writes/s | Wider sampling (1-in-10); aggressive TTL (7d) |
| Cross-city traffic | None — naturally partitioned | City-locality routing at edge |

**Per-city deployment**: each city has its own stack instance. No cross-city queries on hot path.

## 8. Tradeoffs & Alternatives

| Decision | Picked | Alt | Why |
|---|---|---|---|
| H3 | ✅ | Geohash, S2 | Hex symmetry; equidistant neighbors |
| Redis for location index | ✅ | Spatial DB (PostGIS) | Throughput; ephemeral data |
| WebSocket for drivers | ✅ | Periodic POST | Server push for matching offer |
| Sampled trace storage | ✅ | Full-fidelity | Cost; can recover from Kafka if needed |
| Per-city deployments | ✅ | Single global | Latency; failure isolation |

## 9. Follow-up Questions

- **"How do you avoid the same driver being offered two trips at once?"** — Lock the driver during offer (Redis SETNX with 15s TTL); release on accept/reject/timeout.
- **"What if the rider moves while waiting?"** — Re-trigger geo lookup on significant rider movement; reassign if better match exists and current driver hasn't been engaged yet.
- **"How do you handle 'find me a car' previews without commitment?"** — Same geo lookup, returns count + ETA estimate; no driver locking.
- **"What if a driver's network is flaky?"** — TTL on Redis entry (30 s) drops them from index. Driver reconnects, sends fresh location. Trip-in-progress drivers get longer TTL + last-known-location showed to rider with "connection lost" UI.
- **"How does ETA work?"** — Road graph + edge weights from traffic data, A* or contraction hierarchies for routing. Pre-computed tile lookups for common origin-destination patterns.
- **"How do you scale Match svc?"** — Shard by city; within a city, by H3 cell range. Match workers run as stateless consumers of per-cell request streams.
- **"What about airports / events with surge spikes?"** — Pre-deploy capacity (incentivize drivers to position there); higher surge cap; reservation queue for ride-share waits.
- **"How does Uber handle 100% driver-side failure (entire pod dies)?"** — Edge → reconnect to new pod; pull last state from Redis + DB. Trip-in-progress: 60-90 s window where driver location is stale but trip is "in flight"; rider sees last-known position with a stale indicator.
- **"Disaster recovery for an entire city?"** — Active-active across two regions per city; DNS failover; Cassandra cross-region replication for trip data; in-flight matches lost (recreated by rider retry).

## Related

- `websockets_realtime.md` — driver connection tier
- `caching.md` — Redis geo index
- `message_queues.md` — Kafka event stream
- `databases.md` — Cassandra time-series writes
- `id_generation.md` — Snowflake trip IDs
