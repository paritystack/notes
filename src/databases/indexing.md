# Indexing

## Overview

An index is a secondary data structure that lets the database find rows without scanning the
whole table — the difference between `O(log n)` and `O(n)` on every lookup. Indexing is the
most direct performance lever in a database, but every index is a trade: faster reads for
slower writes and more storage. This note covers the index types, when each applies, and how
to design them — extending [database internals](database_internals.md) (B-trees) and feeding
[query optimization](query_optimization.md) (the optimizer chooses which index to use), with
practical detail beyond what [database design](database_design.md) introduces.

```
The fundamental trade-off:
  index speeds up   READS  (lookups, joins, sorts, range scans)
  index slows down  WRITES (every INSERT/UPDATE/DELETE must update every index)
  and costs         STORAGE
⇒ index for your real query patterns; don't index everything "just in case".
```

## How an index helps

```
Without index: WHERE email = 'x'  → Seq Scan, read all N rows.
With B-tree on email: walk the tree O(log N) → jump straight to the row pointer.

An index stores (key → row location), kept sorted/organized so search is fast.
```

## Index types

```
B-tree        DEFAULT. Sorted → supports =, <, >, BETWEEN, ORDER BY, prefix LIKE 'abc%'.
              99% of indexes. See database_internals.md.
Hash          O(1) equality only (no ranges, no sorting). Niche.
Bitmap        one bitmap per distinct value; great for LOW-cardinality columns
              (gender, status) and combining many predicates (common in OLAP/warehouses).
GIN           inverted index for "value contains" — full-text search, JSONB, arrays.
              See ../data_structures/inverted_index.md and elasticsearch.md.
GiST / SP-GiST generalized search tree — geospatial (PostGIS), ranges, nearest-neighbour.
BRIN          tiny "min/max per block" index for huge, naturally-ordered tables
              (time-series): cheap, coarse. See ../databases/duckdb.md for column stores.
```

## Clustered vs non-clustered

A crucial physical distinction that governs how the table itself is stored:

```
Clustered index    the table rows ARE stored in index order (the index IS the table).
                   One per table. Range scans & PK lookups are very fast.
                   → MySQL/InnoDB: the PRIMARY KEY is the clustered index.
Non-clustered      a separate structure pointing back to the row (heap or clustered key).
                   Many per table. → Postgres tables are HEAP-organized; all indexes are
                   secondary (point to a tuple id / ctid).
```

Implication (InnoDB): secondary indexes store the *primary key* as the row pointer, so a
huge/random PK (e.g. a UUID) bloats every secondary index and hurts insert locality — prefer
a compact, monotonic surrogate PK.

## Composite (multi-column) indexes

Order matters, and it follows the **leftmost-prefix rule**:

```
INDEX (a, b, c) can serve queries filtering:
  ✓ a            ✓ a, b           ✓ a, b, c
  ✗ b alone      ✗ c alone        ✗ b, c   (no leading 'a' → can't use it)

Design rule of thumb (ESR): Equality columns first, then Sort/range, then the Range column.
  WHERE tenant_id = ? AND status = ? ORDER BY created_at
    → INDEX (tenant_id, status, created_at)
```

## Covering indexes & index-only scans

If an index contains *every* column a query needs, the database answers from the index alone
— never touching the table (an **index-only scan**), the fastest possible read.

```
SELECT email, name FROM users WHERE tenant_id = ?
  INDEX (tenant_id) INCLUDE (email, name)   ← Postgres INCLUDE / "covering" index
  → no heap fetch; everything is in the index.
```

## Specialized & space-saving indexes

```
Partial index    index only the rows you query: WHERE active = true.
                 Smaller, faster, cheaper to maintain. (Postgres)
Expression index index a computed value: INDEX (LOWER(email)) → makes
                 WHERE LOWER(email)=? sargable. See query_optimization.md.
Unique index     enforces uniqueness AND speeds lookups (backs PK/UNIQUE constraints).
Full-text (GIN)  tokenized search; for serious search use elasticsearch.md.
```

## When NOT to index / when indexes don't help

```
✗ Low-cardinality column queried for the common value (status='active' = 90% of rows):
  the optimizer will Seq Scan anyway — see query_optimization.md selectivity.
✗ Small tables — a Seq Scan is already cheap.
✗ Write-heavy tables with rarely-used indexes — pure overhead.
✗ Non-sargable predicates — a function on the column disables the index.
✗ Too many indexes — each one taxes every write and consumes RAM/cache.
```

## Maintenance

```
- Indexes fragment/bloat over time → REINDEX (Postgres) / OPTIMIZE (MySQL) periodically.
- Find unused indexes (pg_stat_user_indexes idx_scan = 0) and DROP them — they only cost.
- Find MISSING indexes from slow-query logs and pg_stat_statements.
- After bulk loads: build indexes AFTER loading data (faster), then ANALYZE.
```

## Where this connects

- **[Database internals](database_internals.md)** — B-tree/LSM mechanics underneath.
- **[Query optimization](query_optimization.md)** — how the optimizer picks an index, SARGability.
- **[Database design](database_design.md)** — schema and key choices that drive indexing.
- **[Inverted index](../data_structures/inverted_index.md)** / **[Elasticsearch](elasticsearch.md)**
  — the structure behind full-text/GIN indexes.

## Pitfalls

- **Indexing every column** — kills write throughput and wastes cache; index for queries.
- **Wrong composite order** — violating the leftmost-prefix rule makes the index unusable.
- **Random UUID primary keys** in clustered engines (InnoDB) — bloat and poor insert locality.
- **Forgetting the index can be skipped** — low selectivity makes a full scan cheaper.
- **Never dropping unused indexes** — they silently tax every write forever.
