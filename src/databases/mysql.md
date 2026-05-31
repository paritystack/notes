# MySQL

## Overview

MySQL is the world's most widely deployed open-source relational database — the "M" in the
LAMP stack and the backbone of countless web applications. Alongside [PostgreSQL](postgres.md)
it's one of the two RDBMS engines you'll meet most; this note focuses on what's distinctive
about MySQL and how it differs from Postgres, building on [SQL](sql.md) fundamentals and
[database internals](database_internals.md) (InnoDB is a B-tree/WAL/MVCC engine).

```
MySQL ecosystem:
  MySQL (Oracle)   the original; widely used, Oracle-stewarded.
  MariaDB          community fork after Oracle's acquisition; mostly drop-in compatible.
  Percona Server   performance-focused MySQL distribution.
Storage engine that matters: InnoDB (default) — transactional, ACID, row-locking.
```

## Storage engines (a MySQL-specific concept)

Unlike most databases, MySQL has a **pluggable storage engine** layer — the SQL front end
is separate from the engine that stores rows.

```
InnoDB   DEFAULT. ACID transactions, row-level locking, MVCC, crash recovery, FKs,
         clustered primary-key index. → use this for almost everything.
MyISAM   legacy. Table-level locking, NO transactions, NO crash safety. Avoid for new
         work (you'll still see it in old schemas and some read-only/full-text cases).
MEMORY   in-RAM tables; volatile. Niche.
```

## InnoDB internals worth knowing

```
Clustered PK index: rows are stored INSIDE the primary-key B-tree (the PK *is* the table).
  ⇒ PK lookups & range scans are fast.
  ⇒ secondary indexes store the PK as their row pointer → a large/random PK (UUID) bloats
    every secondary index and hurts insert locality. Prefer a compact AUTO_INCREMENT PK,
    or a time-ordered UUID (UUIDv7). See indexing.md.

MVCC + redo/undo logs: readers see a consistent snapshot; writers don't block readers.
Buffer pool: innodb_buffer_pool_size is the #1 tuning knob — size it to hold the hot set
  (often 50–75% of RAM on a dedicated DB host). See database_internals.md.
```

## Transactions & isolation

```
Default isolation: REPEATABLE READ (Postgres defaults to READ COMMITTED — a real
  behavioural difference between the two).
InnoDB uses next-key locking (gap locks) to prevent phantom rows at REPEATABLE READ.
Autocommit is ON by default — wrap multi-statement work in BEGIN ... COMMIT.
```

See [database internals — isolation levels](database_internals.md) and [ACID vs BASE](acid_vs_base.md).

## Replication

MySQL's replication is mature and ubiquitous; it ships the **binary log** (binlog) to replicas.

```
Async (default)    leader writes binlog; replicas pull & replay → read scaling, some lag.
Semi-sync          leader waits for ≥1 replica to ACK receipt → less data loss on failover.
GTID-based         Global Transaction IDs make failover/repositioning robust.
Binlog formats:    ROW (default, safe), STATEMENT (compact, risky), MIXED.
Group Replication / InnoDB Cluster: built-in HA with automatic failover (Paxos-based).
```

The binlog also drives **change data capture** (Debezium reads it) — see [CDC & streaming](cdc_streaming.md).
For the general theory, see [replication & sharding](replication_sharding.md).

## MySQL vs PostgreSQL — practical differences

```
                     MySQL (InnoDB)              PostgreSQL
  Default isolation  REPEATABLE READ             READ COMMITTED
  Table storage      clustered by PRIMARY KEY    heap (unordered) + secondary indexes
  Extensibility      storage engines             rich types, extensions (PostGIS, etc.)
  JSON               JSON + functional indexes   JSONB (binary, indexable, richer ops)
  Standards/features Postgres tends to lead       window funcs, CTEs, types, partial idx
  Ecosystem reflex   web apps, read-heavy        complex queries, analytics, GIS
Both are excellent; choose on features & team familiarity, not folklore.
```

## SQL & syntax notes

```
- Backtick identifiers `col`; LIMIT n OFFSET m for pagination.
- AUTO_INCREMENT for surrogate keys; ON DUPLICATE KEY UPDATE for upserts.
- utf8mb4 charset (NOT "utf8", which is a broken 3-byte subset that can't store emoji).
- Engine/charset matter at table creation: ENGINE=InnoDB DEFAULT CHARSET=utf8mb4.
- EXPLAIN / EXPLAIN ANALYZE for plans — see query_optimization.md.
- Strict SQL mode ON to reject silent truncation/invalid data (older MySQL was lax).
```

## Operations

```
Backups:   mysqldump (logical, simple), Percona XtraBackup (hot physical), binlog for PITR.
Tuning:    innodb_buffer_pool_size, innodb_flush_log_at_trx_commit (durability vs speed),
           slow query log + pt-query-digest to find offenders.
Scaling:   read replicas first; sharding (Vitess) when writes outgrow one node.
HA:        InnoDB Cluster / Group Replication, or orchestrators (Orchestrator, ProxySQL).
```

## Where this connects

- **[PostgreSQL](postgres.md)** — the other major RDBMS; key differences above.
- **[SQL](sql.md)** — shared query language.
- **[Database internals](database_internals.md)** — InnoDB B-tree/MVCC/WAL mechanics.
- **[Indexing](indexing.md)** — clustered-PK implications for secondary indexes.
- **[Replication & sharding](replication_sharding.md)** / **[CDC](cdc_streaming.md)** —
  binlog-based replication and capture.

## Pitfalls

- **Using `utf8`** instead of `utf8mb4` — silently can't store 4-byte chars (emoji, some CJK).
- **MyISAM for new tables** — no transactions, no crash safety; use InnoDB.
- **Random UUID primary keys** — bloat secondary indexes and wreck insert locality.
- **Assuming Postgres defaults** — MySQL defaults to REPEATABLE READ, clustered PK storage.
- **Forgetting strict mode** — older configs silently truncate/accept bad data.
