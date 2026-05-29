# ClickHouse

ClickHouse is an open-source, column-oriented OLAP (Online Analytical Processing) database management system designed for real-time analytical queries over very large datasets. It's commonly used for observability, log/metric storage, product analytics, and ad-tech workloads where billions of rows must be scanned in sub-second time.

## Overview

ClickHouse stores data column-by-column on disk, compresses each column independently, and executes queries with a vectorized engine that processes blocks of values at a time. It scales horizontally via sharding and replication, and supports a SQL dialect close to standard SQL.

**Key Features:**
- Column-oriented storage with per-column codecs
- Vectorized query execution (SIMD-friendly)
- Massive parallel processing (MPP) across cores and shards
- Real-time `INSERT`s — millions of rows/sec per node
- SQL with rich analytical extensions (arrays, higher-order functions, approximate aggregates)
- High compression ratios (ZSTD, LZ4, Delta, Gorilla)
- Built-in integrations with Kafka, S3, MySQL, PostgreSQL, HDFS
- Materialized views that update on insert
- Distributed and replicated tables out of the box

## Architecture

```
┌─────────────────────────────────────────────────┐
│              ClickHouse Cluster                 │
│                                                 │
│   Shard 1            Shard 2          Shard 3   │
│  ┌────────┐         ┌────────┐       ┌────────┐ │
│  │Replica1│◄────────┤Replica1│◄──────┤Replica1│ │
│  │Replica2│         │Replica2│       │Replica2│ │
│  └────────┘         └────────┘       └────────┘ │
│        ▲                ▲                 ▲     │
│        └────────────────┼─────────────────┘     │
│                         │                       │
│                ┌────────┴────────┐              │
│                │ ClickHouse Keeper│              │
│                │  (or ZooKeeper)  │              │
│                └─────────────────┘              │
└─────────────────────────────────────────────────┘
```

- **Shards** split data horizontally; each shard owns a subset of rows
- **Replicas** within a shard hold identical copies for HA
- **ClickHouse Keeper** (or ZooKeeper) coordinates replication metadata
- **Parts** are immutable on-disk pieces; background **merges** combine them
- **Granules** (8192 rows by default) are the unit of the sparse primary index

## Installation

```bash
# Ubuntu/Debian (official repo)
sudo apt install -y apt-transport-https ca-certificates dirmngr
GNUPGHOME=$(mktemp -d) sudo GNUPGHOME="$GNUPGHOME" gpg --no-default-keyring \
    --keyring /usr/share/keyrings/clickhouse-keyring.gpg \
    --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 8919F6BD2B48D754
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main" \
    | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt update
sudo apt install -y clickhouse-server clickhouse-client

# Start the server
sudo systemctl start clickhouse-server

# Docker
docker run -d --name ch -p 8123:8123 -p 9000:9000 \
    --ulimit nofile=262144:262144 clickhouse/clickhouse-server

# macOS
brew install clickhouse

# Single-binary (any Linux/macOS)
curl https://clickhouse.com/ | sh
./clickhouse server   # in one terminal
./clickhouse client   # in another

# Verify
clickhouse-client --version
```

## Basic Usage

```bash
# Interactive client
clickhouse-client
clickhouse-client --host 127.0.0.1 --port 9000 --user default --password ''

# Execute a single query
clickhouse-client --query "SELECT version()"

# Pipe data in
cat data.csv | clickhouse-client --query "INSERT INTO events FORMAT CSV"

# Pick an output format
clickhouse-client --query "SELECT * FROM events LIMIT 5" --format PrettyCompact
clickhouse-client --query "SELECT * FROM events LIMIT 5 FORMAT JSONEachRow"

# Run a SQL file
clickhouse-client --multiquery < script.sql

# HTTP interface (port 8123)
curl 'http://localhost:8123/?query=SELECT+version()'
echo 'SELECT 1' | curl 'http://localhost:8123/' --data-binary @-
```

## Data Types

```sql
-- Integers (signed/unsigned, 8–256 bit)
Int8, Int16, Int32, Int64, Int128, Int256
UInt8, UInt16, UInt32, UInt64, UInt128, UInt256

-- Floating point and fixed-point
Float32, Float64
Decimal(P, S)     -- e.g., Decimal(18, 4)

-- Strings
String            -- arbitrary length, UTF-8
FixedString(N)    -- exactly N bytes

-- Dates and times
Date              -- 2 bytes, days since epoch
Date32            -- 4 bytes, wider range
DateTime          -- second precision
DateTime64(3)     -- millisecond; (6) micro, (9) nano

-- Identifiers and enums
UUID
Enum8('a' = 1, 'b' = 2)
Enum16(...)

-- Nullable wrapper (carries a per-value null mask)
Nullable(String)

-- LowCardinality dictionary-encodes repeated values; great for status, country, etc.
LowCardinality(String)

-- Containers
Array(T)              -- e.g., Array(UInt32)
Tuple(T1, T2, ...)    -- positional struct
Map(K, V)             -- key/value map
Nested(col1 T1, col2 T2)  -- parallel arrays

-- IPs and geo
IPv4, IPv6, Point, Polygon, MultiPolygon
```

## Table Engines: MergeTree Family

The MergeTree family is the workhorse for production analytics.

```sql
-- Plain MergeTree: append-friendly columnar storage
CREATE TABLE events (
    event_time  DateTime,
    user_id     UInt64,
    event_type  LowCardinality(String),
    url         String,
    revenue     Decimal(18, 4)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_type, user_id, event_time)
SETTINGS index_granularity = 8192;

-- ReplacingMergeTree: dedupes by ORDER BY key during merges (keeps latest by version col)
CREATE TABLE users (
    user_id    UInt64,
    name       String,
    email      String,
    updated_at DateTime
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY user_id;

-- SummingMergeTree: sums numeric columns sharing the same ORDER BY key
CREATE TABLE daily_revenue (
    day      Date,
    country  LowCardinality(String),
    revenue  Decimal(18, 4),
    orders   UInt64
)
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (day, country);

-- AggregatingMergeTree: stores AggregateFunction states; finalize with -Merge
CREATE TABLE uniq_users_daily (
    day        Date,
    country    LowCardinality(String),
    users      AggregateFunction(uniq, UInt64)
)
ENGINE = AggregatingMergeTree
ORDER BY (day, country);

-- CollapsingMergeTree: pair +1/-1 sign rows to "cancel" a previous row
CREATE TABLE state_changes (
    id     UInt64,
    state  String,
    sign   Int8
)
ENGINE = CollapsingMergeTree(sign)
ORDER BY id;

-- VersionedCollapsingMergeTree: like Collapsing but tolerates out-of-order inserts
CREATE TABLE state_changes_v (
    id      UInt64,
    state   String,
    sign    Int8,
    version UInt64
)
ENGINE = VersionedCollapsingMergeTree(sign, version)
ORDER BY id;
```

**Key clauses:**
- `PARTITION BY` — physical partitioning; usually a month/day expression
- `ORDER BY` — also the sparse primary key; pick prefix for cardinality climbing low→high
- `PRIMARY KEY` (optional, defaults to `ORDER BY`) — can be a shorter prefix
- `SAMPLE BY` — enables `SELECT ... SAMPLE 0.1` for probabilistic queries

## Other Engines

```sql
-- Log family (append-only, no indexes; small data)
CREATE TABLE small (id UInt64, val String) ENGINE = Log;
CREATE TABLE tiny  (id UInt64, val String) ENGINE = TinyLog;

-- In-memory
CREATE TABLE scratch (id UInt64) ENGINE = Memory;

-- Federated reads
CREATE TABLE mysql_users
ENGINE = MySQL('mysql:3306', 'db', 'users', 'user', 'pass');

CREATE TABLE pg_users
ENGINE = PostgreSQL('pg:5432', 'db', 'users', 'user', 'pass');

-- File-backed
CREATE TABLE from_file (id UInt64, val String)
ENGINE = File(CSV, '/var/lib/clickhouse/user_files/data.csv');

-- URL-backed
CREATE TABLE remote_csv (id UInt64, val String)
ENGINE = URL('https://example.com/data.csv', CSV);

-- View (logical, no storage)
CREATE VIEW active AS SELECT * FROM users WHERE active = 1;

-- Materialized view (stored, populated on every INSERT to source)
CREATE MATERIALIZED VIEW mv_events_hourly
ENGINE = SummingMergeTree
ORDER BY (hour, event_type) AS
SELECT toStartOfHour(event_time) AS hour, event_type, count() AS cnt
FROM events GROUP BY hour, event_type;
```

## Schema Design

```sql
-- LowCardinality dictionary-encodes repeated strings (status, country, device)
CREATE TABLE pageviews (
    ts      DateTime,
    country LowCardinality(String),
    device  LowCardinality(String),
    path    String,
    user_id UInt64
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (country, device, ts);

-- Per-column compression codecs
CREATE TABLE metrics (
    ts       DateTime  CODEC(DoubleDelta, ZSTD(3)),
    metric   LowCardinality(String) CODEC(ZSTD(3)),
    value    Float64   CODEC(Gorilla, ZSTD(3)),
    tags     Array(String) CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (metric, ts);

-- TTL: expire old rows, or move to a slower disk
CREATE TABLE logs (
    ts      DateTime,
    level   LowCardinality(String),
    msg     String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (level, ts)
TTL ts + INTERVAL 30 DAY DELETE,
    ts + INTERVAL 7  DAY TO VOLUME 'cold';

-- Skip indexes (data-skipping, not lookup indexes)
ALTER TABLE logs ADD INDEX idx_msg msg TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4;
ALTER TABLE logs ADD INDEX idx_level level TYPE set(100) GRANULARITY 4;
```

**Common codecs:**
- `LZ4` (default), `LZ4HC`, `ZSTD(level)`
- `Delta(N)` — diff against previous value (good for timestamps, counters)
- `DoubleDelta` — delta of deltas (regular timestamps)
- `Gorilla` — Facebook's float compression for slowly changing series
- `T64` — fixed-width integer packing
- `NONE` — no compression

## CRUD Operations

```sql
-- INSERT (always batch; ClickHouse loathes single-row inserts)
INSERT INTO events VALUES
    ('2026-05-28 10:00:00', 1, 'click', '/home', 0),
    ('2026-05-28 10:00:01', 2, 'view',  '/home', 0);

-- INSERT from SELECT
INSERT INTO events_archive
SELECT * FROM events WHERE event_time < '2025-01-01';

-- INSERT with FORMAT (HTTP/stdin)
INSERT INTO events FORMAT JSONEachRow
{"event_time":"2026-05-28 10:00:00","user_id":1,"event_type":"click","url":"/","revenue":0}

-- SELECT
SELECT event_type, count() FROM events
WHERE event_time >= today() - 7
GROUP BY event_type
ORDER BY count() DESC;

-- Mutations (heavy; rewrite parts in background)
ALTER TABLE events UPDATE revenue = 0 WHERE revenue < 0;
ALTER TABLE events DELETE WHERE event_time < '2024-01-01';

-- Lightweight DELETE (faster, async cleanup)
DELETE FROM events WHERE user_id = 42;

-- Force a merge (deduplicate ReplacingMergeTree, etc.) — use sparingly
OPTIMIZE TABLE users FINAL;

-- Read latest deduplicated state without materializing the merge
SELECT * FROM users FINAL WHERE user_id = 1;
```

## Reading External Data

```sql
-- Local file
SELECT * FROM file('data.csv', 'CSV', 'id UInt64, name String');

-- S3
SELECT * FROM s3(
    'https://bucket.s3.amazonaws.com/path/*.parquet',
    'AKIA...', 'SECRET...',
    'Parquet'
);

-- URL
SELECT * FROM url('https://example.com/data.csv', 'CSV', 'id UInt64, val String');

-- MySQL / PostgreSQL ad-hoc reads
SELECT * FROM mysql('host:3306', 'db', 'tbl', 'user', 'pass');
SELECT * FROM postgresql('host:5432', 'db', 'tbl', 'user', 'pass');

-- Insert from a file path on the client
INSERT INTO events FROM INFILE 'events.csv.gz' FORMAT CSV;

-- Common formats
-- CSV, CSVWithNames, TSV, TSVWithNames, JSONEachRow, Parquet, ORC, Arrow,
-- ProtobufSingle, Avro, Native, Values, Pretty, PrettyCompact, Vertical
```

## Writing and Exporting

```sql
-- Write a query result to a file on the client
SELECT * FROM events
INTO OUTFILE 'events.parquet'
FORMAT Parquet;

-- Write to S3
INSERT INTO FUNCTION s3(
    'https://bucket.s3.amazonaws.com/out/data.parquet',
    'AKIA...', 'SECRET...',
    'Parquet'
)
SELECT * FROM events WHERE event_time >= today() - 1;
```

## Aggregations

```sql
-- Standard aggregates
SELECT
    count(),
    countIf(revenue > 0)        AS paying_events,
    sum(revenue),
    avg(revenue),
    min(event_time),
    max(event_time),
    uniq(user_id)               AS approx_users,        -- HyperLogLog
    uniqExact(user_id)          AS exact_users,
    uniqCombined(user_id)       AS hll_combined,
    quantile(0.95)(revenue),
    quantileTDigest(0.99)(revenue),
    groupArray(event_type)      AS sample_types
FROM events;

-- Combinators: -If, -Array, -State, -Merge, -OrNull, -Resample
SELECT
    sumIf(revenue, event_type = 'purchase')       AS purchase_rev,
    avgArray([1.0, 2.0, 3.0])                     AS avg_of_array,
    quantilesState(0.5, 0.9, 0.99)(revenue)       AS q_state   -- stored in AggregatingMergeTree
FROM events;

-- Later, finalize a stored state with -Merge
SELECT quantilesMerge(0.5, 0.9, 0.99)(q_state) FROM agg_table;
```

## Window Functions

```sql
SELECT
    user_id,
    event_time,
    revenue,
    row_number() OVER (PARTITION BY user_id ORDER BY event_time) AS n,
    sum(revenue) OVER (
        PARTITION BY user_id
        ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cum_revenue,
    lag(revenue) OVER (PARTITION BY user_id ORDER BY event_time)  AS prev_rev,
    lead(revenue) OVER (PARTITION BY user_id ORDER BY event_time) AS next_rev
FROM events;
```

## Joins

```sql
-- Standard joins
SELECT u.user_id, u.name, sum(e.revenue)
FROM users u
LEFT JOIN events e USING (user_id)
GROUP BY u.user_id, u.name;

-- ASOF join: nearest-match on a sort key (great for trades/quotes)
SELECT t.symbol, t.ts, q.bid, q.ask
FROM trades AS t
ASOF LEFT JOIN quotes AS q
  ON t.symbol = q.symbol AND t.ts >= q.ts;

-- ANY join: take any matching row (cheaper than full join)
SELECT u.user_id, ANY o.order_id
FROM users u
ANY LEFT JOIN orders o USING (user_id);

-- GLOBAL join: send the right side to every shard (use on Distributed tables)
SELECT *
FROM events_dist e
GLOBAL JOIN users_small u ON e.user_id = u.user_id;

-- Join algorithms (hint via settings)
SET join_algorithm = 'partial_merge';   -- or 'hash', 'parallel_hash', 'grace_hash', 'auto'
```

## Arrays and Higher-Order Functions

```sql
-- Flatten an array into rows
SELECT user_id, tag
FROM events
ARRAY JOIN tags AS tag;

-- Higher-order functions
SELECT
    arrayMap(x -> x * 2, [1, 2, 3])              AS doubled,
    arrayFilter(x -> x > 1, [1, 2, 3])           AS gt1,
    arrayReduce('sum', [1, 2, 3])                AS total,
    arraySum(x -> x * x, [1, 2, 3])              AS sum_sq,
    arrayCount(x -> x > 0, [-1, 0, 1, 2])        AS positives,
    arrayExists(x -> x = 'click', ['view','click']) AS has_click,
    arrayDistinct([1, 1, 2, 3, 3])               AS uniq_vals,
    groupArray(event_type)                       AS all_types,
    groupUniqArray(event_type)                   AS uniq_types
FROM events GROUP BY user_id;
```

## Materialized Views

```sql
-- Hourly rollup that updates on every INSERT into events
CREATE MATERIALIZED VIEW mv_hourly
ENGINE = SummingMergeTree
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, event_type) AS
SELECT
    toStartOfHour(event_time) AS hour,
    event_type,
    count()                   AS events,
    sum(revenue)              AS revenue
FROM events
GROUP BY hour, event_type;

-- MV with AggregatingMergeTree (preserves the aggregate states)
CREATE MATERIALIZED VIEW mv_uniq
ENGINE = AggregatingMergeTree
ORDER BY (day, event_type) AS
SELECT
    toDate(event_time)               AS day,
    event_type,
    uniqState(user_id)               AS users_state
FROM events
GROUP BY day, event_type;

-- Read final values back with -Merge
SELECT day, event_type, uniqMerge(users_state) AS users
FROM mv_uniq GROUP BY day, event_type;

-- Refreshable MV (recomputes on a schedule; v23.12+)
CREATE MATERIALIZED VIEW mv_daily REFRESH EVERY 1 HOUR
ENGINE = MergeTree ORDER BY day AS
SELECT toDate(event_time) AS day, count() AS n FROM events GROUP BY day;
```

## Projections

```sql
-- Alternate physical layout that the optimizer can choose
ALTER TABLE events ADD PROJECTION proj_by_user (
    SELECT user_id, event_time, revenue
    ORDER BY user_id
);

ALTER TABLE events MATERIALIZE PROJECTION proj_by_user;
```

## Partitioning and TTL

```sql
-- Drop or detach a partition (instant, no rewrite)
ALTER TABLE events DROP PARTITION '202401';
ALTER TABLE events DETACH PARTITION '202401';
ALTER TABLE events ATTACH PARTITION '202401';

-- Move a partition to a different disk volume
ALTER TABLE events MOVE PARTITION '202401' TO VOLUME 'cold';

-- TTL: per-row expiration and per-column expiration
ALTER TABLE events MODIFY TTL event_time + INTERVAL 90 DAY;
ALTER TABLE events MODIFY COLUMN raw_body String TTL event_time + INTERVAL 7 DAY;
```

## Distributed Tables and Replication

```sql
-- ReplicatedMergeTree (replicated within a shard)
CREATE TABLE events_local ON CLUSTER '{cluster}' (
    event_time DateTime,
    user_id    UInt64,
    event_type LowCardinality(String),
    revenue    Decimal(18, 4)
)
ENGINE = ReplicatedMergeTree(
    '/clickhouse/tables/{shard}/events',  -- ZooKeeper/Keeper path
    '{replica}'
)
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_type, user_id, event_time);

-- Distributed engine fan-outs queries across shards
CREATE TABLE events ON CLUSTER '{cluster}' AS events_local
ENGINE = Distributed('{cluster}', currentDatabase(), events_local, rand());

-- Ad-hoc cross-shard read without a Distributed table
SELECT count() FROM cluster('{cluster}', currentDatabase(), events_local);
SELECT count() FROM remote('host1,host2', currentDatabase(), events_local);
```

Macros (`{cluster}`, `{shard}`, `{replica}`) are defined per-host in `config.xml`.

## Integrations

```sql
-- Kafka source: stream rows in
CREATE TABLE events_kafka (
    event_time DateTime,
    user_id    UInt64,
    event_type String
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:9092',
    kafka_topic_list = 'events',
    kafka_group_name = 'ch_events',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 2;

-- Materialized view bridges Kafka -> MergeTree
CREATE MATERIALIZED VIEW mv_kafka_to_events TO events AS
SELECT * FROM events_kafka;

-- S3: read Parquet directly (no copy needed)
SELECT count() FROM s3(
    'https://bucket.s3.amazonaws.com/events/*.parquet',
    'AKIA...', 'SECRET...',
    'Parquet'
);

-- PostgreSQL federated read
SELECT * FROM postgresql('pg:5432', 'app', 'orders', 'user', 'pass')
WHERE created_at >= today() - 1;
```

## Performance Optimization

```sql
-- Show the query plan
EXPLAIN SELECT * FROM events WHERE event_type = 'click';

-- Show the execution pipeline (operators, threads)
EXPLAIN PIPELINE SELECT count() FROM events;

-- Show physical access estimates
EXPLAIN ESTIMATE SELECT * FROM events WHERE event_time >= today() - 1;

-- Useful system tables
SELECT * FROM system.query_log
WHERE event_time >= now() - INTERVAL 1 HOUR AND type = 'QueryFinish'
ORDER BY query_duration_ms DESC LIMIT 10;

SELECT database, table, sum(rows), formatReadableSize(sum(bytes_on_disk)) AS size
FROM system.parts WHERE active GROUP BY database, table;

SELECT * FROM system.merges;       -- active background merges
SELECT * FROM system.mutations;    -- active ALTER UPDATE/DELETE
SELECT * FROM system.replicas;     -- replication lag, queue size
SELECT * FROM system.settings WHERE changed;

-- Skip indexes (data-skipping)
ALTER TABLE events ADD INDEX idx_user user_id TYPE minmax GRANULARITY 4;
ALTER TABLE events ADD INDEX idx_type event_type TYPE set(0) GRANULARITY 4;
ALTER TABLE events ADD INDEX idx_url  url        TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE events ADD INDEX idx_msg  msg        TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 4;
ALTER TABLE events ADD INDEX idx_ng   msg        TYPE ngrambf_v1(3, 32768, 3, 0) GRANULARITY 4;

-- Read-in-order tricks
SET optimize_read_in_order = 1;
SET optimize_aggregation_in_order = 1;
```

**Skip index types:** `minmax`, `set(max_rows)`, `bloom_filter(false_positive)`, `tokenbf_v1`, `ngrambf_v1`.

## Settings and Configuration

```sql
-- Session-level settings
SET max_threads = 8;
SET max_memory_usage = 10000000000;        -- 10 GB
SET max_execution_time = 60;
SET join_algorithm = 'parallel_hash';
SET allow_experimental_lightweight_delete = 1;

-- Inspect everything
SELECT * FROM system.settings WHERE name LIKE '%memory%';

-- Per-query
SELECT count() FROM events SETTINGS max_threads = 16;
```

Server-level config lives in `/etc/clickhouse-server/config.xml` (ports, storage, clusters, ZooKeeper/Keeper) and users/profiles/quotas in `/etc/clickhouse-server/users.xml`. Prefer drop-ins under `config.d/` and `users.d/` for overrides.

## Client Libraries

```python
# clickhouse-connect (recommended; pure Python, HTTP)
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost', username='default')
rows = client.query("SELECT count() FROM events").result_rows
df   = client.query_df("SELECT * FROM events LIMIT 100")
client.insert('events', [(datetime.now(), 1, 'click', '/', 0)])

# clickhouse-driver (native TCP protocol)
from clickhouse_driver import Client
c = Client(host='localhost')
c.execute("INSERT INTO events VALUES", rows)
print(c.execute("SELECT count() FROM events"))

# Raw HTTP
# curl 'http://localhost:8123/?query=SELECT+1'
```

Other officially supported clients: Go (`clickhouse-go`), JDBC, ODBC, C++, Node.js (`@clickhouse/client`), Rust (`clickhouse.rs`).

## Best Practices

```sql
-- 1. Batch inserts. Aim for 10k+ rows per INSERT, not row-at-a-time
INSERT INTO events VALUES (...), (...), (...);

-- 2. Pick ORDER BY by ascending cardinality of frequently filtered columns
ENGINE = MergeTree ORDER BY (country, event_type, event_time);

-- 3. Use LowCardinality for repeated strings (status, country, device)
event_type LowCardinality(String)

-- 4. Partition by month or day, not by minute/hour (avoid tiny partitions)
PARTITION BY toYYYYMM(event_time)

-- 5. Avoid frequent ALTER UPDATE/DELETE; they rewrite whole parts
-- Prefer ReplacingMergeTree + insert new versions, or lightweight DELETE

-- 6. Use FINAL only for ad-hoc reads, never in hot paths
SELECT * FROM users FINAL WHERE user_id = 1;

-- 7. Push heavy rollups into materialized views (AggregatingMergeTree + -State)

-- 8. Apply ZSTD on cold-ish data, LZ4 (default) on hot data
value Float64 CODEC(Gorilla, ZSTD(3))

-- 9. Don't use Nullable when you can use a sentinel default (Nullable adds a mask)

-- 10. Avoid SELECT * on wide tables; column pruning is most of the win
SELECT event_time, revenue FROM events WHERE ...;
```

## Quick Reference

| Item | What it is |
|------|------------|
| `MergeTree` | Default columnar engine — partitioned, sparse-indexed |
| `ReplacingMergeTree(ver)` | Dedupes rows sharing the ORDER BY key |
| `SummingMergeTree` | Auto-sums numeric columns on merge |
| `AggregatingMergeTree` | Stores `AggregateFunction` states |
| `ReplicatedMergeTree` | Above engines but replicated via Keeper |
| `Distributed` | Query fan-out across shards |
| `Kafka` / `S3` / `MySQL` / `PostgreSQL` | Source/sink engines |
| `system.query_log` | Historical query stats |
| `system.parts` | On-disk parts per table |
| `system.merges` / `system.mutations` | Background work in flight |
| `system.replicas` | Replication health and lag |
| `LowCardinality(T)` | Dictionary-encoded column |
| `CODEC(ZSTD, Delta, Gorilla, T64)` | Per-column compression |
| `OPTIMIZE TABLE t FINAL` | Force a merge (use sparingly) |
| `EXPLAIN [PIPELINE|ESTIMATE]` | Inspect plans and IO estimates |
| `uniq` / `quantileTDigest` | Approximate aggregates |
| `arrayMap` / `ARRAY JOIN` | Higher-order array ops |
| `ASOF JOIN` | Nearest-match temporal join |
| `clickhouse-client --query "..."` | One-shot CLI query |
| `:8123` HTTP / `:9000` native | Default ports |

ClickHouse excels at scanning billions of rows for analytical aggregations in milliseconds, making it the go-to engine for real-time analytics, observability backends, and any workload where columnar compression and vectorized execution pay off.
