# Database Internals

## Overview

This is the "how does it actually work" layer beneath [SQL](sql.md), [PostgreSQL](postgres.md),
and the NoSQL stores. Every database has to answer the same physical questions: how are rows
laid out on disk, how do you find one fast, how do you survive a crash mid-write, and how do
concurrent transactions avoid stepping on each other? The answers — pages, B-trees vs LSM
trees, write-ahead logging, and MVCC — explain the performance and consistency behaviour you
see at the SQL layer, and inform [query optimization](query_optimization.md),
[indexing](indexing.md), and [replication](replication_sharding.md).

```
The storage stack:
  SQL / query  ─►  query planner  ─►  execution  ─►  access methods (indexes)
                                                         │
                                              storage engine (pages, buffer pool)
                                                         │
                                              durability (WAL)  ─►  disk
```

## Pages — the unit of storage

Databases don't read individual rows from disk; they read fixed-size **pages** (blocks),
typically 4–16 KB, because disk and OS I/O work in blocks.

```
Page (e.g. 8 KB in Postgres):
  ┌─────────────────────────────────────┐
  │ header │ item pointers →             │
  │        ······· free space ········   │
  │              ← tuples (rows)         │
  └─────────────────────────────────────┘
  Pointers grow down, tuples grow up; they meet when the page is full.
```

The **buffer pool** is an in-memory cache of hot pages. Reads check the pool first (cache
hit) before touching disk; writes modify pages in memory ("dirty" pages) and are flushed
later. Hit ratio is the single biggest lever on read performance — RAM access is ~100,000×
faster than disk.

## Storage engines: B-tree vs LSM tree

The central design choice. It determines whether a database is optimized for reads or writes.

### B-tree (read-optimized) — Postgres, MySQL/InnoDB, most RDBMS

A balanced tree kept sorted by key; updates happen **in place**.

```
                 [ 50 | 90 ]              ← internal nodes route the search
                /    |     \
          [..30] [50..80] [90..]          ← leaf nodes hold keys → row pointers
   O(log n) lookups; leaves often linked for range scans.
   Reads: excellent. Writes: random in-place writes (more I/O, fragmentation).
```

### LSM tree (write-optimized) — Cassandra, RocksDB, LevelDB, ClickHouse-ish

Buffer writes in memory, flush sorted batches sequentially; merge in the background.

```
  writes → [ memtable (in RAM, sorted) ]
                  │ flush when full (sequential write — fast)
           [ SSTable L0 ] [ SSTable L0 ] ...        immutable, sorted files
                  │ compaction merges & drops tombstones
           [ SSTable L1 ........... ] [ L2 ... ]

  Writes: sequential & fast (great for high ingest). Reads: may check multiple
  SSTables → use Bloom filters (see ../data_structures/bloom_filter.md) to skip them.
  Deletes are "tombstones"; space reclaimed at compaction.
```

```
                 B-tree            LSM tree
  Reads          fast              slower (multi-file, Bloom-filtered)
  Writes         slower (random)   fast (sequential, batched)
  Space          fragmentation     write amplification (compaction)
  Best for       OLTP, mixed       write-heavy ingest, time-series, logs
```

## Write-ahead logging (WAL) — durability

How a database keeps the **D** in [ACID](acid_vs_base.md) and survives a crash mid-write.

```
The rule (write-ahead): append the change to a sequential LOG and fsync it BEFORE
modifying the actual data pages.

  commit → append redo record to WAL → fsync → ACK client
                                              (data pages flushed lazily later)

Crash recovery:
  REDO   replay committed WAL records not yet flushed to data pages
  UNDO   roll back changes from transactions that never committed
```

Why it works: the log is a *sequential* append (fast) and is the source of truth. Even if
the machine dies before dirty pages reach disk, the committed log lets recovery rebuild the
correct state. The same WAL stream is what powers **replication** (ship the log to replicas)
and **point-in-time recovery**. See [replication & sharding](replication_sharding.md).

## Concurrency control — isolation

How concurrent transactions get the **I** in ACID without serializing everything.

### Locking (pessimistic)

```
Shared (read) locks, Exclusive (write) locks; two-phase locking (2PL) guarantees
serializability. Cost: blocking, and DEADLOCKS (T1 waits on T2 waits on T1) →
the DB detects a cycle and aborts a victim.
```

### MVCC (Multi-Version Concurrency Control) — Postgres, MySQL/InnoDB, Oracle

Keep **multiple versions** of each row so readers never block writers and vice versa.

```
Each row version is tagged with the transaction that created/expired it.
A transaction sees a consistent SNAPSHOT as of its start time.

  Reader  → sees the version valid at its snapshot (old data, no waiting)
  Writer  → creates a NEW version; old version stays for current readers

  ⇒ "readers don't block writers, writers don't block readers" — the big MVCC win.
  Cost: old versions accumulate → need garbage collection (Postgres VACUUM).
```

## Isolation levels

The SQL standard trades isolation against concurrency. Higher levels prevent more anomalies
but cost more:

```
Level             Dirty read  Non-repeatable  Phantom    Notes
─────────────────────────────────────────────────────────────────────────────
Read Uncommitted   possible     possible       possible   rarely used
Read Committed     prevented    possible       possible   Postgres default
Repeatable Read    prevented    prevented      possible*  MySQL default (*InnoDB blocks phantoms)
Serializable       prevented    prevented      prevented   as if transactions ran one-at-a-time
```

See [ACID vs BASE](acid_vs_base.md) for the consistency-model context and how distributed
stores relax these guarantees.

## Putting it together — a write's journey

```
INSERT → parse & plan → acquire row version (MVCC) → modify page in buffer pool
       → append redo record to WAL → fsync WAL → ACK
       → (later) flush dirty page to disk; (later) VACUUM old versions
```

## Where this connects

- **[Indexing](indexing.md)** — B-trees and friends as access methods.
- **[Query optimization](query_optimization.md)** — the planner reasons about pages,
  selectivity, and access methods.
- **[Replication & sharding](replication_sharding.md)** — WAL shipping and partitioning.
- **[System design: databases](../system_design/databases.md)** — these internals at scale.
- **[Advanced trees](../data_structures/advanced_trees.md)** / **[Bloom filter](../data_structures/bloom_filter.md)**
  — the data structures inside the engine.

## Pitfalls

- **Assuming all DBs are B-trees** — write-heavy workloads belong on LSM engines.
- **Ignoring the buffer pool** — a workload that doesn't fit in RAM falls off a cliff.
- **Long-running transactions under MVCC** — they pin old row versions and bloat the table
  (Postgres VACUUM can't reclaim them).
- **Misunderstanding isolation defaults** — Read Committed (Postgres) still allows
  non-repeatable reads; know your level.
