# Change Data Capture & Streaming

## Overview

Change Data Capture (CDC) is the practice of capturing every row-level change in a database
— inserts, updates, deletes — as a stream of events, so other systems can react in real
time. It's the glue that keeps a [relational system-of-record](postgres.md) in sync with
search indexes ([Elasticsearch](elasticsearch.md)), caches ([Redis](redis.md)), data
warehouses, and microservices — without brittle dual-writes. CDC reads the database's own
[write-ahead log](database_internals.md) and publishes to a log like [Kafka](kafka.md),
making it a cornerstone of event-driven [system design](../system_design/message_queues.md).

```
The problem CDC solves — DUAL WRITES are a trap:
  app → DB  ✓
  app → search index  ✗ (crashes here → DB and index now disagree, no rollback)
CDC fixes it: write ONLY to the DB; derive everything else from its change stream.
  DB ──(WAL/binlog)──► CDC ──► Kafka ──► [search] [cache] [warehouse] [services]
```

## How CDC works — log-based vs the alternatives

```
Log-based CDC (the good way)   read the DB's transaction log (Postgres WAL / logical
                               decoding, MySQL binlog, Mongo oplog).
   ✓ captures EVERY change in commit order, low overhead, no schema changes, includes deletes
   → Debezium is the de-facto open-source tool.

Query-based (polling)          SELECT ... WHERE updated_at > last_seen on a timer.
   ✗ misses deletes, misses intermediate updates, adds DB load, needs an updated_at column.

Trigger-based                  DB triggers write changes to an audit table.
   ✗ adds write overhead and trigger maintenance; intrusive.
```

Log-based wins because the log is already the source of truth the database uses for
[durability and replication](replication_sharding.md) — CDC is just another consumer of it.

## The outbox pattern

The standard way for a microservice to publish events **atomically** with its own data,
avoiding dual writes:

```
In ONE local transaction:
  INSERT INTO orders ...           (the business change)
  INSERT INTO outbox (event) ...   (the event to publish)
COMMIT   ← both succeed or both fail (atomic, same DB)

Then CDC tails the `outbox` table → publishes to Kafka → marks/deletes the row.
⇒ the event is published if and only if the data was committed. No lost/phantom events.
```

This is how you get reliable event publishing without distributed transactions. See
[microservices](../system_design/microservices.md).

## Delivery semantics & ordering

```
Ordering        log-based CDC preserves per-key (per-row) order via the commit log;
                Kafka partitions keyed by primary key keep a row's events in order.
At-least-once   the norm — failures cause replays → consumers must be IDEMPOTENT.
                Use the change's LSN/offset or a version to dedupe. See
                ../system_design/idempotency.md.
Exactly-once    only with transactional sinks (Kafka transactions, idempotent upserts).
Snapshots       on first run, CDC takes an initial SNAPSHOT of existing rows, then
                switches to streaming incremental changes.
```

## Stream processing

Once changes are a stream, you can transform, join, and aggregate them in flight:

```
Kafka Streams / ksqlDB   joins, windowed aggregations on Kafka topics (JVM).
Apache Flink             powerful stateful stream processing, event-time, exactly-once.
Apache Spark Structured Streaming   micro-batch; unifies batch + stream.
Materialized views       e.g. Materialize, RisingWave — keep a SQL view always up to date
                         from a CDC stream.
```

```
Stream vs batch (and the convergence):
  Batch   process bounded data periodically (nightly ETL).        high latency, simple.
  Stream  process unbounded data continuously as it arrives.       low latency.
  Modern "streaming ETL" (ELT): CDC → Kafka → transform → warehouse, near real-time.
```

## Common CDC pipelines

```
DB → search:     Postgres ──Debezium──► Kafka ──► Elasticsearch sink   (keep search fresh)
DB → cache:      changes invalidate/update Redis entries               (cache coherence)
DB → warehouse:  CDC ──► Kafka ──► Snowflake/BigQuery/ClickHouse        (real-time analytics)
microservices:   outbox ──► Kafka ──► other services react             (event-driven)
DB → DB:         cross-region or heterogeneous replication
```

## Watch out for

```
Schema evolution   a column add/rename ripples downstream → use a schema registry
                   (Avro/Protobuf) and compatibility rules.
Backpressure       a slow consumer must not break the source DB; Kafka decouples them
                   (the log buffers). See ../system_design/message_queues.md.
Replication slots  (Postgres) an un-consumed logical slot pins WAL → disk fills up.
                   Monitor slot lag; a dead CDC consumer can take down the source DB.
Deletes & tombstones  ensure deletes propagate (log-based does; polling doesn't).
```

## Where this connects

- **[Kafka](kafka.md)** — the durable log CDC publishes to.
- **[Database internals](database_internals.md)** — the WAL/binlog CDC reads.
- **[Elasticsearch](elasticsearch.md)** / **[Redis](redis.md)** — common CDC sinks.
- **[Message queues](../system_design/message_queues.md)** / **[Idempotency](../system_design/idempotency.md)**
  — delivery semantics and consumer design.
- **[Microservices](../system_design/microservices.md)** — the outbox pattern for events.

## Pitfalls

- **Dual writes** — writing to DB and another system separately; they will diverge. Use CDC/outbox.
- **Polling instead of log-based** — misses deletes and intermediate updates, loads the DB.
- **Non-idempotent consumers** — at-least-once delivery replays events; design for dedupe.
- **Ignoring Postgres replication-slot lag** — a stalled consumer fills the disk and stalls the DB.
- **No schema-evolution plan** — an upstream column change silently breaks every consumer.
