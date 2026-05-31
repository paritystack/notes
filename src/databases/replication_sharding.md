# Replication & Sharding

## Overview

A single database server eventually hits a wall — too much data, too many reads, or it
becomes a single point of failure. **Replication** (copies of the same data) buys
availability and read scale; **sharding/partitioning** (splitting the data) buys write scale
and capacity. They're orthogonal and usually combined. This is where [database internals](database_internals.md)
(the WAL that ships between nodes) meets [system design](../system_design/databases.md), the
[CAP theorem](acid_vs_base.md), and [consistent hashing](../system_design/consistent_hashing.md).

```
Replication  → SAME data on N nodes     → availability + read scaling + durability
Sharding     → DIFFERENT data on N nodes → write scaling + storage capacity
Most large systems do BOTH: shard the data, replicate each shard.
```

## Replication topologies

### Leader–follower (primary–replica)

The most common model. One node takes writes; the WAL stream is shipped to read-only
replicas.

```
            writes
   client ────────► [ LEADER ] ──WAL stream──► [ follower ] ──► reads
                         │                  └─► [ follower ] ──► reads
   reads can be served by followers → read scaling.
   Leader fails → promote a follower (failover).
```

### Multi-leader

Multiple nodes accept writes (e.g. one per region). Scales writes and tolerates regional
outages, but introduces **write conflicts** that need resolution (last-write-wins, CRDTs —
see [CRDTs](../data_structures/crdt.md)).

### Leaderless (Dynamo-style) — Cassandra, DynamoDB

Any node accepts reads and writes; clients write to several replicas and read from several,
using quorums to stay consistent. See [Cassandra](cassandra.md).

```
Quorum:  W + R > N  ⇒  read set overlaps write set ⇒ reads see the latest write.
  N=3, W=2, R=2 is the classic tunable-consistency setting.
```

## Synchronous vs asynchronous

The fundamental durability-vs-latency trade-off in replication:

```
Synchronous     leader waits for replica(s) to ACK before confirming the commit.
                ✓ no data loss on failover   ✗ higher write latency; stalls if replica slow
Asynchronous    leader confirms immediately; replicas catch up.
                ✓ fast writes                ✗ replication LAG → recent writes can be lost
                                               if the leader dies before they propagate
Semi-sync       wait for ONE replica, not all → a common middle ground.
```

## Replication lag and its anomalies

Async replicas are slightly behind, which breaks naive read-your-writes:

```
User posts a comment (→ leader), then reloads (← lagging follower) and it's GONE.

Fixes:
  Read-your-writes      route a user's reads to the leader briefly after they write.
  Monotonic reads       pin a user to one replica so they never see time go backwards.
  Consistent prefix     ensure causally-related writes are read in order.
```

These are **eventual consistency** symptoms; see [ACID vs BASE](acid_vs_base.md).

## Failover

```
Leader dies → detect (heartbeat timeout) → elect/promote a follower → redirect writes.
Hazards:
  Split-brain   two nodes both think they're leader → divergent writes. Prevent with a
                consensus/quorum (Raft/Paxos) or fencing. See ../algorithms/raft.md.
  Lost writes   async writes not yet replicated are gone after promotion.
  Failback      old leader returns and must catch up as a follower, not fight for the role.
```

## Sharding (horizontal partitioning)

Split one logical table across many nodes by a **shard key**. The choice of key is the whole
game — it determines balance and which queries stay fast.

```
Range sharding      shard by key ranges (A–F, G–M, …).
                    ✓ efficient range scans   ✗ HOTSPOTS (sequential keys → one shard)
Hash sharding       shard by hash(key).
                    ✓ even distribution       ✗ range queries hit every shard
Consistent hashing  add/remove nodes while moving minimal data (see consistent_hashing.md).
Directory/lookup    a lookup service maps keys → shards (flexible, extra hop).
```

```
A GOOD shard key is:
  high-cardinality   (many distinct values → spreads load)
  evenly accessed    (no single hot value)
  aligned with queries (most queries hit ONE shard, not all)
Bad keys: monotonic IDs/timestamps (hotspot the newest shard), low-cardinality
  fields like country/status (skew).
```

## The cost of sharding

Sharding breaks things that were free on a single node — defer it until you must:

```
✗ Cross-shard JOINs        now a distributed, slow operation (or app-side stitching)
✗ Cross-shard transactions need 2-phase commit / sagas → see ../system_design/microservices.md
✗ Global secondary indexes hard; queries not on the shard key must scatter-gather
✗ Re-sharding              moving data when a shard grows hot is operationally painful
✓ Mitigation              keep related data in the same shard (co-location by tenant/user)
```

## Partitioning vs sharding (terminology)

```
Partitioning   splitting a table — can be within ONE node (Postgres declarative
               partitioning) or across nodes.
Sharding       partitioning specifically ACROSS multiple nodes/machines.
Vertical part. splitting COLUMNS (hot vs cold) into separate tables/stores.
Horizontal     splitting ROWS — what "sharding" usually means.
```

## Where this connects

- **[Database internals](database_internals.md)** — the WAL stream that replication ships.
- **[Consistent hashing](../system_design/consistent_hashing.md)** — minimal-movement sharding.
- **[ACID vs BASE](acid_vs_base.md)** / **[CAP](../system_design/databases.md)** — the
  consistency trade-offs replication forces.
- **[Raft](../algorithms/raft.md)** — leader election and avoiding split-brain.
- **[Cassandra](cassandra.md)** — leaderless replication + tunable consistency in practice.

## Pitfalls

- **Reading from async replicas and expecting your own writes** — handle replication lag.
- **A monotonic shard key** — sends all new writes to one hot shard.
- **Sharding too early** — you lose joins/transactions; scale vertically and with read
  replicas first.
- **No fencing on failover** — split-brain silently corrupts data.
- **Forgetting cross-shard queries** — a query not on the shard key fans out to every node.
