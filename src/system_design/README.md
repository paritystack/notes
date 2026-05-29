# System Design

Designing large-scale distributed systems for performance, scalability, and reliability.

## Foundations

- **[Scalability](scalability.md)**: Horizontal vs vertical scaling, load balancing strategies
- **[Caching](caching.md)**: Cache strategies, invalidation, distributed caching
- **[Load Balancing](load_balancing.md)**: Algorithms, OSI layers, health checks
- **[Consistent Hashing](consistent_hashing.md)**: Even partitioning with minimal remapping
- **[Rate Limiting](rate_limiting.md)**: Throttling algorithms (token bucket, leaky bucket)
- **[RPC](rpc.md)**: Remote Procedure Call frameworks and patterns

## Distributed Systems

- **[Distributed Systems](distributed_systems.md)**: CAP, consistency models, replication, partitioning
- **[Distributed Consensus](distributed_consensus.md)**: Paxos, Raft, consistency models
- **[Databases](databases.md)**: SQL vs NoSQL, sharding, replication
- **[Message Queues](message_queues.md)**: Asynchronous processing, event-driven architecture

## Architecture

- **[Microservices](microservices.md)**: Service decomposition, communication, deployment
- **[Design Patterns](design_patterns.md)**: API gateway, BFF, circuit breaker, saga, CQRS, event sourcing

## Concept Primers

Focused interview-style references for topics that show up in nearly every design.

- **[CDN](cdn.md)**: Edge caching, origin shield, invalidation, edge compute
- **[Observability](observability.md)**: Logs/metrics/traces, OpenTelemetry, RED/USE, SLO alerting
- **[WebSockets & Realtime](websockets_realtime.md)**: WS/SSE/long-poll, scaling stateful connections
- **[Idempotency](idempotency.md)**: Idempotency keys, dedup, exactly-once illusions
- **[ID Generation](id_generation.md)**: Snowflake, UUIDv7, ULID, KSUID

## Interview Toolkit

- **[Interview Framework](interview_framework.md)**: The 7-phase playbook (clarify → estimate → API → data → HLD → deep dive → wrap)
- **[Estimation Cheatsheet](estimation_cheatsheet.md)**: Latency numbers, QPS/storage math, sizing templates

## Case Studies

Canonical design problems. Each follows the same template: Requirements → Estimation → API → Data model → HLD → Deep dives → Bottlenecks → Tradeoffs → Follow-ups.

- **[URL Shortener](design_url_shortener.md)** (TinyURL/Bitly): ID gen, KV store, read-heavy cache
- **[News Feed](design_news_feed.md)** (Twitter/Facebook): Fan-out, celebrity problem
- **[Chat System](design_chat_system.md)** (WhatsApp/Slack): WS, ordering, group fan-out
- **[Ride Sharing](design_ride_sharing.md)** (Uber/Lyft): Geospatial, matching, surge
- **[Video Streaming](design_video_streaming.md)** (YouTube/Netflix): CDN, ABR, transcoding
- **[Typeahead](design_typeahead.md)** (Google Suggest): Trie at scale, top-K, freshness

## Key Concepts

- **Throughput**: Requests per second
- **Latency**: Response time
- **Availability**: Uptime percentage
- **Consistency**: Data correctness
- **Partition Tolerance**: Handling failures

## Design Goals

1. **Reliability**: Surviving failures
2. **Scalability**: Growing with demand
3. **Performance**: Fast responses
4. **Maintainability**: Easy to update

## How to Approach a Design

See `interview_framework.md` for the full playbook. In short:

1. Clarify requirements & scope (functional + non-functional)
2. Capacity estimation (QPS, storage, bandwidth)
3. API design
4. Data model
5. High-level architecture
6. Deep dives on hard parts
7. Bottlenecks & tradeoffs

## Navigation

Foundations and primers first, then case studies. The case studies cross-link back to primers for depth.
