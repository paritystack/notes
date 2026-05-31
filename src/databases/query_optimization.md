# Query Optimization

## Overview

A SQL query says *what* you want, not *how* to get it. The **query optimizer** turns that
declarative request into an efficient execution plan — choosing access methods, join
algorithms, and join order. Understanding how it decides (and how to read its plan via
`EXPLAIN`) is the highest-leverage performance skill in databases: a missing index or a bad
join order can make the same query 1000× slower. This builds on [database internals](database_internals.md)
(pages, the buffer pool) and [indexing](indexing.md), and applies directly to
[SQL](sql.md) and [PostgreSQL](postgres.md).

```
SQL text → parse → rewrite → PLAN (optimizer) → execute
                                  │
                  cost-based: estimate the cost of alternative plans,
                  pick the cheapest, using table STATISTICS.
```

## How the optimizer thinks: cost & statistics

A cost-based optimizer estimates how expensive each candidate plan is (in I/O + CPU) and
picks the cheapest. The estimates come from **statistics** the database keeps about each
table.

```
Statistics (Postgres: ANALYZE; auto-collected): row counts, value distributions
  (histograms), number of distinct values, most-common values, null fraction.

Selectivity = fraction of rows a predicate keeps.
  WHERE status='active'  →  if 90% are active, selectivity 0.9 (a scan is fine)
  WHERE id = 42          →  selectivity ~1/N (an index is a huge win)

Stale statistics = bad estimates = bad plans. Keep ANALYZE current after big data changes.
```

## Reading EXPLAIN

`EXPLAIN` shows the chosen plan; `EXPLAIN ANALYZE` actually runs it and shows real timings
and row counts. The gap between *estimated* and *actual* rows reveals stale stats.

```
EXPLAIN ANALYZE SELECT ... :
  Nested Loop  (cost=.. rows=10 ..) (actual time=.. rows=9500 ..)   ← est 10, actual 9500!
    -> Seq Scan on orders   ...                                     ← red flag: full scan
    -> Index Scan using idx_customer ...

Read it INSIDE-OUT / BOTTOM-UP: leaf nodes run first, feed parents.
Watch for: Seq Scan on big tables, estimate≫actual row mismatches, expensive Sort/Hash spills.
```

## Access methods — how a table is read

```
Seq Scan      read the whole table. Best when returning a LARGE fraction of rows.
Index Scan    walk the index, fetch matching rows. Best for SELECTIVE predicates.
Index-Only Scan  answer entirely from the index (covering index) — no table fetch. Fastest.
Bitmap Scan   build a bitmap of matching pages, then read them in order — good for
              medium selectivity / combining multiple indexes.
```

The optimizer's key judgement: an index is NOT always faster. Fetching 80% of rows via an
index causes random I/O for almost every page — a sequential scan is cheaper. This is why a
"correct" index sometimes goes unused.

## Join algorithms

```
Nested Loop     for each row in A, look up matches in B.
                Great when A is tiny and B is indexed on the join key. O(A·B) otherwise.
Hash Join       build a hash table on the smaller side, probe with the larger.
                Best for large, unsorted, equality joins. Needs memory (can spill to disk).
Merge Join      sort both sides on the join key, walk in lockstep.
                Best when inputs are already sorted (e.g. from indexes) or for range joins.
```

```
Choosing intuition:
  small × indexed-large   → Nested Loop
  large × large, equality → Hash Join
  both pre-sorted         → Merge Join
```

**Join order** matters enormously: joining the most selective tables first keeps
intermediate result sets small. The optimizer searches join orders (and falls back to
heuristics/genetic search when there are too many tables).

## Common fixes (what actually speeds things up)

```
- Add the right index (on filter/join/sort columns); use COVERING indexes for index-only
  scans. See indexing.md.
- Update statistics (ANALYZE) — fixes bad row estimates and bad plans.
- SELECT only needed columns — enables index-only scans, less I/O.
- Make predicates SARGABLE: WHERE created_at >= '2024-01-01'  NOT  WHERE date(created_at)=...
  (a function on the column disables the index).
- Avoid N+1 queries — one join beats 1000 round-trips (an app/ORM problem, not the DB's).
- Fix the join: ensure both sides of the join key are indexed and same type (avoid casts).
- Paginate with keyset/seek (WHERE id > last) not large OFFSET (which scans & discards).
- Watch implicit type casts and `OR` chains that defeat indexes.
```

## SARGability — let the index work

```
"Search ARGument ABLE" = a predicate the index can use directly.
  ✓ WHERE email = $1
  ✓ WHERE created_at >= $1 AND created_at < $2
  ✗ WHERE LOWER(email) = $1          → index on email unused (index LOWER(email) instead)
  ✗ WHERE created_at::date = $1      → function on column kills the index
  ✗ WHERE col + 0 = $1               → arithmetic on column
```

## Materialization & caching

```
Materialized views   precompute & store an expensive query result; refresh on schedule.
Query result caching  app-level (Redis) for hot, rarely-changing results. See redis.md.
Denormalization       trade write cost/consistency for read speed on hot paths.
Partitioning          prune irrelevant partitions before scanning. See database_internals.md.
```

## Where this connects

- **[Indexing](indexing.md)** — the access methods the optimizer chooses among.
- **[Database internals](database_internals.md)** — pages, buffer pool, B-trees underneath.
- **[SQL](sql.md)** / **[PostgreSQL](postgres.md)** — `EXPLAIN`, `ANALYZE`, planner knobs.
- **[Redis](redis.md)** — caching to avoid the query entirely.

## Pitfalls

- **Premature indexing** — every index slows writes and costs space; index for real query
  patterns, not guesses.
- **Stale statistics** — the most common cause of a suddenly-slow query after a data load.
- **Non-sargable predicates** — wrapping the column in a function silently disables its index.
- **Large OFFSET pagination** — `OFFSET 1000000` scans and throws away a million rows.
- **Trusting EXPLAIN over EXPLAIN ANALYZE** — estimates lie; measure with real row counts.
