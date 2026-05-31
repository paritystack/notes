# Apache Cassandra

## Overview

Cassandra is a distributed, **wide-column** NoSQL database built for massive write
throughput, linear scalability, and no single point of failure. It's the canonical
**leaderless** (Dynamo-style) store: every node is equal, data is replicated across a ring,
and consistency is *tunable* per query. Where a [relational database](postgres.md) optimizes
for flexible queries on one big node, Cassandra optimizes for writes and availability across
many nodes — a concrete instance of the [replication & sharding](replication_sharding.md)
and [LSM-tree](database_internals.md) ideas, and of the AP corner of [CAP](acid_vs_base.md).

```
Pick Cassandra when you have:
  ✓ huge write volume (IoT, time-series, event logs, messaging)
  ✓ need for always-on availability across regions
  ✓ known, fixed query patterns
Avoid when you need:
  ✗ ad-hoc queries, joins, aggregations  ✗ strong transactional consistency
```

## Architecture: the ring

No leader. Nodes form a ring; data is placed by hashing the partition key onto the ring
(consistent hashing). Any node can serve any request (acting as **coordinator**).

```
           Node A
         /        \
   Node E          Node B      client → ANY node (coordinator) → routes to replicas
        |          |           gossip protocol shares cluster state peer-to-peer
   Node D          Node C      add a node → it takes a slice of the ring; data rebalances
         \        /
           (token ring; partition key → token → owning nodes)
```

Replication factor (RF) = how many nodes hold each row (e.g. RF=3). There's no special
primary copy — all replicas are equal. See [consistent hashing](../system_design/consistent_hashing.md).

## Tunable consistency

Cassandra's signature feature: choose the consistency/latency trade-off **per query** via
the quorum formula.

```
W + R > RF   ⇒   reads are guaranteed to see the latest write (strong consistency)

Consistency levels (per read & per write):
  ONE          ack from 1 replica        fastest, weakest
  QUORUM       majority (RF/2 + 1)        balanced; QUORUM read + QUORUM write = strong
  LOCAL_QUORUM majority in local DC       multi-region without cross-DC latency
  ALL          every replica             strongest, least available

Example RF=3:  write QUORUM(2) + read QUORUM(2) → 2+2 > 3 → strong & still tolerates 1 node down.
```

This is **eventual consistency** by default, made as-strong-as-needed per operation —
contrast with the fixed [ACID](acid_vs_base.md) guarantees of an RDBMS.

## Data model: query-first

The hardest mental shift coming from SQL: **you model around your queries, not your data**.
There are no joins and no ad-hoc filtering, so you design tables to answer specific queries
— and **denormalize/duplicate** data across tables freely.

```
Primary key = (partition key, clustering columns)
  Partition key  → which node(s) store the row; determines distribution. ALWAYS in the query.
  Clustering cols → sort order WITHIN a partition; enable range scans on that partition.

  PRIMARY KEY ((user_id), message_ts)
    partition by user_id  → all of a user's messages co-located on the same replicas
    cluster by message_ts → stored sorted by time → efficient "latest N messages"

Rule: one table per query pattern. Need two access paths? Make two tables (dual writes).
```

```
Anti-patterns that kill Cassandra:
  ✗ huge "hot" partitions (one partition key gets all the traffic/data)
  ✗ unbounded partitions (a partition that grows forever)
  ✗ queries without the partition key → full-cluster scan (ALLOW FILTERING = red flag)
  ✗ low-cardinality partition keys → uneven distribution
```

## Storage engine: LSM under the hood

Writes are fast because Cassandra is LSM-tree based — append to a commit log + memtable,
flush to immutable SSTables, compact in the background.

```
write → commit log (durability) + memtable (RAM)
      → flush memtable → SSTable (immutable, sorted on disk)
      → compaction merges SSTables, drops tombstones
read  → check memtable + SSTables; Bloom filters skip SSTables that can't have the key
delete → writes a TOMBSTONE (not in-place); space reclaimed at compaction
```

See [database internals — LSM trees](database_internals.md) and [Bloom filters](../data_structures/bloom_filter.md).

```
Tombstone trap: heavy deletes/TTLs create tombstones that must be scanned past on reads.
  Too many tombstones in a partition → slow reads and query failures. Model to avoid
  delete-heavy patterns; use TTLs thoughtfully.
```

## Writes, reads, repair

```
Hinted handoff   if a replica is down, the coordinator stores a "hint" and delivers it later.
Read repair      reads detect stale replicas and fix them in the background.
Anti-entropy repair  scheduled `nodetool repair` reconciles divergent replicas (essential
                 ops hygiene — skipping it lets data drift).
Lightweight transactions (LWT)  Paxos-based compare-and-set for the rare case you need
                 linearizable conditional writes (IF NOT EXISTS) — much slower; use sparingly.
```

## CQL — familiar but not SQL

```
CQL looks like SQL (tables, SELECT/INSERT) but: NO joins, NO subqueries, NO arbitrary
WHERE, NO aggregations across partitions. INSERT and UPDATE are both upserts.
  CREATE TABLE messages (user_id uuid, ts timeuuid, body text,
                         PRIMARY KEY ((user_id), ts)) WITH CLUSTERING ORDER BY (ts DESC);
```

## Ecosystem & alternatives

```
ScyllaDB        C++ rewrite of Cassandra, same model, much higher per-node throughput.
DynamoDB        AWS-managed wide-column store with the same Dynamo lineage.
DataStax        commercial Cassandra (Astra = managed).
```

## Where this connects

- **[Replication & sharding](replication_sharding.md)** — leaderless replication, quorums,
  the ring.
- **[Database internals](database_internals.md)** — LSM trees, tombstones, compaction.
- **[ACID vs BASE](acid_vs_base.md)** — AP/eventual-consistency trade-offs.
- **[Consistent hashing](../system_design/consistent_hashing.md)** — token-ring data placement.
- **[NoSQL](nosql.md)** — where wide-column fits among NoSQL families.

## Pitfalls

- **Modeling like SQL** — normalizing and expecting joins; model per query, denormalize.
- **Hot/unbounded partitions** — the #1 Cassandra performance killer.
- **`ALLOW FILTERING`** — convenient in dev, catastrophic in prod (full scan).
- **Delete-heavy workloads** — tombstones accumulate and slow reads.
- **Skipping repair** — replicas silently diverge without scheduled anti-entropy repair.
- **Using LWT everywhere** — Paxos per write throws away Cassandra's speed advantage.
