# DuckDB

DuckDB is an in-process SQL OLAP (Online Analytical Processing) database management system designed for analytical query workloads. It's often described as "SQLite for analytics."

## Overview

DuckDB is optimized for analytical queries with columnar storage, vectorized execution, and minimal dependencies. It is often compared to [SQLite](sqlite.md) (OLTP/transactional) — both are serverless and in-process, but DuckDB targets OLAP workloads. For server-grade analytics at massive scale, [ClickHouse](clickhouse.md) is the distributed alternative. [PostgreSQL](postgres.md) is a common source when DuckDB is used for local analysis of exported data.

**Key Features:**
- In-process, embedded database (no server, links into your process)
- Columnar storage with per-column compression (FSST, dictionary, RLE, ALP)
- ACID compliant with MVCC snapshot isolation
- Vectorized query execution on chunks of 2048 values
- Morsel-driven parallelism across all CPU cores
- No external dependencies — a single library
- Standard SQL plus a rich "friendly SQL" dialect
- Direct querying of CSV, Parquet, JSON, Arrow without import
- Larger-than-memory execution via automatic spilling
- Extensions for httpfs/S3, Iceberg, Delta, spatial, full-text, vector search

**When to reach for it:**
- Local analytics on files (Parquet, CSV) that are too big for Pandas/Polars but don't justify a cluster
- Embedded analytics inside an application (similar role to SQLite, OLAP role instead of OLTP)
- Notebook / ad-hoc data exploration with SQL
- ETL pipelines that need a fast SQL engine without infrastructure
- Lakehouse-style reads against Iceberg/Delta tables on object storage

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       DuckDB (in-process)                      │
│                                                                │
│  SQL ─► Parser ─► Binder ─► Optimizer ─► Physical Plan         │
│                                              │                 │
│                                              ▼                 │
│                              ┌───────────────────────────┐     │
│                              │  Vectorized Executor      │     │
│                              │  (DataChunk = 2048 rows)  │     │
│                              │  Morsel-driven scheduler  │     │
│                              └───────────┬───────────────┘     │
│                                          │                     │
│              ┌───────────────────────────┼─────────────────┐   │
│              ▼                           ▼                 ▼   │
│       ┌────────────┐             ┌─────────────┐    ┌────────┐ │
│       │  Buffer    │             │  External   │    │ Catalog│ │
│       │  Manager   │             │  Readers    │    │ + WAL  │ │
│       │ (block I/O)│             │ (httpfs,    │    │        │ │
│       └─────┬──────┘             │  parquet,   │    └────────┘ │
│             │                    │  csv, json) │                │
│             ▼                    └─────────────┘                │
│   ┌────────────────────┐                                        │
│   │ Single-file *.duckdb│                                       │
│   │ (block-based,       │                                       │
│   │  columnar storage)  │                                       │
│   └────────────────────┘                                        │
└────────────────────────────────────────────────────────────────┘
```

- **Single-file storage**: a database is one `.duckdb` file plus a small `.wal`. Copy, move, or attach over the network like a SQLite file.
- **Block-based columnar layout**: rows are grouped into row groups (~120K rows); each column is stored separately with its own compression scheme.
- **Vectorized execution**: operators process `DataChunk`s of up to 2048 values, keeping data in CPU cache and letting the compiler emit SIMD.
- **Morsel-driven parallelism**: scans are split into morsels (~100K rows) and stolen across threads — no manual partition keys needed.
- **External readers**: `read_parquet`, `read_csv`, `read_json`, `httpfs`, `iceberg_scan`, `delta_scan` plug into the same vectorized pipeline as native tables.

## Storage Format

- **File layout**: header block → catalog → row-group metadata → column data blocks → free list. Block size is 256 KB by default.
- **Row groups**: ~122,880 rows (60 × 2048) by default. Each holds per-column compressed data plus min/max zone maps and null bitmaps.
- **Per-column compression** is chosen per row group based on data shape:
  - `Constant` for single-value columns
  - `RLE` for runs (e.g. sorted timestamps)
  - `Dictionary` for low-cardinality strings
  - `FSST` (Fast Static Symbol Table) for compressible strings
  - `Bit-packing` and `FOR` (frame-of-reference) for tight integers
  - `ALP` / `Chimp` / `Patas` for floating-point columns
- **Zone maps**: every row group stores min/max per column → predicate filters skip entire row groups before touching data.
- **Checkpoint vs WAL**: writes go to the WAL first, then a checkpoint rewrites the affected blocks. `PRAGMA force_checkpoint;` flushes manually.
- **Compatibility**: storage version is stable across minor releases since v0.10. `pragma storage_info('table_name')` exposes per-block layout.

## Vectorized Execution

- **DataChunk** is the unit of data flow — a column-major mini-batch of up to 2048 values per column, kept hot in L1/L2 cache.
- **Push-based pipeline**: operators in a query plan hand chunks to the next operator without materializing intermediate rows.
- **Vector types**: `FLAT`, `CONSTANT`, `DICTIONARY`, `SEQUENCE` — operators short-circuit on `CONSTANT`/`SEQUENCE` instead of expanding to FLAT.
- **Hash aggregates and joins** use vectorized probes; hash tables are partitioned across threads.
- **SIMD**: tight inner loops (integer comparisons, arithmetic, decompression) auto-vectorize. Use `pragma enable_progress_bar` to watch long queries.

## MVCC & Transactions

- **Snapshot isolation**: each transaction sees a stable snapshot; readers never block writers.
- **Single-writer**: only one writer at a time *per database file*. Multiple processes can attach the same file in read-only mode.
- **Optimistic concurrency control**: write/write conflicts raise `TransactionContext Error: ... Conflict on tuple` at commit time.
- **No row-level locks** — different from Postgres; you cannot `SELECT ... FOR UPDATE`.
- **Implicit transactions**: every statement runs in its own transaction unless wrapped in `BEGIN ... COMMIT`.
- **WAL durability**: `pragma wal_autocheckpoint = '...'` controls checkpoint cadence; `pragma disable_checkpoint_on_shutdown` for tests.

```sql
-- Inspect the current transaction state
SELECT * FROM duckdb_settings() WHERE name LIKE '%transaction%';

-- Force-flush WAL into the main file
PRAGMA force_checkpoint;
```

## Larger-than-Memory Execution

DuckDB streams and spills automatically when working sets exceed `memory_limit`.

- **Out-of-core operators**: hash join, hash aggregate, ORDER BY, window functions all spill to `temp_directory`.
- **Spill files** are compressed; you'll see them as `duckdb_temp_*` under the configured temp dir.
- **Tune**: set `memory_limit` to ~75% of RAM and `temp_directory` to a fast local disk (NVMe).
- **Avoid OOMs from string blow-up**: very wide `VARCHAR` columns expand in hash tables; cast to `BLOB` or `UUID` when applicable.

```sql
SET memory_limit = '8GB';
SET temp_directory = '/var/tmp/duckdb';
SET max_temp_directory_size = '200GB';

-- See current spill usage
SELECT * FROM duckdb_temporary_files();
```

## Parallelism

- `SET threads TO N` controls the worker pool; default = physical cores.
- Morsel-driven scheduling means parallelism is automatic for scans, joins, aggregates, and CSV/Parquet ingest.
- Parallelism *hurts* on very small queries (overhead dominates) — set `threads = 1` in tight benchmarks.
- `EXPLAIN ANALYZE` reports per-operator timing including idle thread time.

## Installation

```bash
# Ubuntu/Debian
sudo apt install duckdb

# macOS
brew install duckdb

# Python
pip install duckdb

# From binary
wget https://github.com/duckdb/duckdb/releases/download/v0.9.2/duckdb_cli-linux-amd64.zip
unzip duckdb_cli-linux-amd64.zip
sudo mv duckdb /usr/local/bin/

# Verify
duckdb --version
```

## Basic Usage

```bash
# Start DuckDB CLI
duckdb

# Create/open database file
duckdb mydb.duckdb

# In-memory database
duckdb :memory:

# Execute command from shell
duckdb mydb.duckdb "SELECT * FROM users;"

# Execute SQL file
duckdb mydb.duckdb < script.sql

# Exit
.quit
```

## Python API

```python
import duckdb

# Connect to database
con = duckdb.connect('mydb.duckdb')

# In-memory database
con = duckdb.connect(':memory:')

# Execute query
result = con.execute("SELECT * FROM users").fetchall()

# Fetch as DataFrame
df = con.execute("SELECT * FROM users").df()

# Direct DataFrame query
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = duckdb.query("SELECT * FROM df WHERE a > 1").df()

# Close connection
con.close()
```

## Table Operations

```sql
-- Create table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table from query
CREATE TABLE new_users AS
SELECT * FROM users WHERE created_at > '2024-01-01';

-- Show tables
SHOW TABLES;
.tables

-- Describe table
DESCRIBE users;
.schema users

-- Drop table
DROP TABLE users;
```

## Reading External Files

```sql
-- Read CSV
SELECT * FROM read_csv_auto('data.csv');

-- Read CSV with options
SELECT * FROM read_csv('data.csv',
    header=true,
    delim=',',
    quote='"',
    types={'id': 'INTEGER', 'name': 'VARCHAR'}
);

-- Create table from CSV
CREATE TABLE users AS
SELECT * FROM read_csv_auto('users.csv');

-- Read Parquet
SELECT * FROM read_parquet('data.parquet');
SELECT * FROM 'data.parquet';  -- Shorthand

-- Read multiple Parquet files
SELECT * FROM read_parquet(['file1.parquet', 'file2.parquet']);
SELECT * FROM read_parquet('data/*.parquet');

-- Read JSON
SELECT * FROM read_json_auto('data.json');
SELECT * FROM read_json('data.json', format='array');

-- Read JSON lines
SELECT * FROM read_json_auto('data.jsonl', format='newline_delimited');
```

## Writing to Files

```sql
-- Export to CSV
COPY users TO 'users.csv' (HEADER, DELIMITER ',');

-- Export to Parquet
COPY users TO 'users.parquet' (FORMAT PARQUET);

-- Export query result
COPY (SELECT * FROM users WHERE active = true)
TO 'active_users.parquet' (FORMAT PARQUET);

-- Export to JSON
COPY users TO 'users.json';
```

## CRUD Operations

```sql
-- Insert
INSERT INTO users (username, email)
VALUES ('john', 'john@example.com');

-- Insert multiple
INSERT INTO users (username, email) VALUES
    ('alice', 'alice@example.com'),
    ('bob', 'bob@example.com');

-- Insert from SELECT
INSERT INTO users (username, email)
SELECT username, email FROM temp_users;

-- Select
SELECT * FROM users;
SELECT * FROM users WHERE username LIKE 'jo%';
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;

-- Update
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;
```

## Analytical Queries

```sql
-- Window functions
SELECT
    username,
    created_at,
    ROW_NUMBER() OVER (ORDER BY created_at) AS row_num,
    RANK() OVER (ORDER BY created_at) AS rank,
    DENSE_RANK() OVER (ORDER BY created_at) AS dense_rank,
    NTILE(4) OVER (ORDER BY created_at) AS quartile
FROM users;

-- Moving average
SELECT
    date,
    revenue,
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_7d
FROM sales;

-- Cumulative sum
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) AS cumulative_total
FROM transactions;

-- Percent rank
SELECT
    username,
    score,
    PERCENT_RANK() OVER (ORDER BY score) AS percentile
FROM scores;
```

## Friendly SQL

DuckDB ships a set of SQL dialect extensions that cut boilerplate. None of these exist in standard SQL but they are widely used in DuckDB code.

### Star expressions

```sql
-- Exclude columns from a star
SELECT * EXCLUDE (password_hash, internal_id) FROM users;

-- Replace columns in-place (rename/transform without listing all columns)
SELECT * REPLACE (LOWER(email) AS email, created_at::DATE AS created_at)
FROM users;

-- Rename columns in the projection
SELECT * RENAME (username AS handle, created_at AS signup_date)
FROM users;

-- Combine: exclude + replace
SELECT * EXCLUDE (password_hash) REPLACE (LOWER(email) AS email)
FROM users;

-- COLUMNS() picks columns dynamically
SELECT COLUMNS(c -> c LIKE 'metric_%') FROM measurements;
SELECT COLUMNS('^sales_\d{4}$') FROM revenue;  -- regex
SELECT MIN(COLUMNS(* EXCLUDE id)) FROM measurements;  -- apply MIN to each column
```

### GROUP BY ALL / ORDER BY ALL

```sql
-- Group by every non-aggregated SELECT expression automatically
SELECT region, product, SUM(amount) AS total
FROM sales
GROUP BY ALL;  -- equivalent to GROUP BY region, product

-- Order by every SELECT expression
SELECT region, product, SUM(amount) FROM sales GROUP BY ALL ORDER BY ALL;
```

### QUALIFY

Filters the output of window functions without wrapping in a subquery.

```sql
-- Top 3 orders per customer
SELECT customer_id, order_id, amount,
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rn
FROM orders
QUALIFY rn <= 3;

-- Without QUALIFY, this would need a CTE or subquery
```

### FROM-first syntax

```sql
-- Lets you start typing the table name first (good for autocomplete)
FROM users SELECT username, email WHERE active;
FROM users;  -- shorthand for SELECT * FROM users
```

### Chained comparisons and string slicing

```sql
SELECT * FROM measurements WHERE 0 < value < 100;
SELECT 'hello world'[1:5];     -- 'hello' (1-indexed, inclusive)
SELECT 'hello world'[-5:];     -- 'world'
SELECT [1, 2, 3, 4, 5][2:4];   -- [2, 3, 4]
```

### List comprehensions and lambdas

```sql
-- List comprehension
SELECT [x * 2 FOR x IN [1, 2, 3, 4, 5] IF x > 2];  -- [6, 8, 10]

-- Lambdas with higher-order list functions
SELECT list_transform([1, 2, 3, 4], x -> x * x);   -- [1, 4, 9, 16]
SELECT list_filter([1, 2, 3, 4], x -> x % 2 = 0);  -- [2, 4]
SELECT list_reduce([1, 2, 3, 4], (acc, x) -> acc + x);  -- 10

-- Combined with COLUMNS
SELECT list_transform([col_a, col_b, col_c], x -> x * 100) FROM data;
```

### UNION BY NAME

```sql
-- Standard UNION matches by column position. UNION BY NAME aligns by name
-- and fills missing columns with NULL.
SELECT 1 AS a, 2 AS b
UNION BY NAME
SELECT 3 AS b, 4 AS c;
-- Result: (a=1, b=2, c=NULL), (a=NULL, b=3, c=4)
```

### Sampling

```sql
-- Reservoir sample (exact N rows)
SELECT * FROM huge_table USING SAMPLE 1000;

-- Bernoulli sample (each row independently with probability)
SELECT * FROM huge_table USING SAMPLE 5 PERCENT;

-- System sample (block-level, much faster, less uniform)
SELECT * FROM huge_table USING SAMPLE 5 PERCENT (system, 42);  -- seed=42

-- Sample inside a subquery
SELECT AVG(price) FROM (FROM products USING SAMPLE 10000);
```

### Positional and LATERAL joins

```sql
-- POSITIONAL JOIN matches rows by row order (zip-like)
SELECT * FROM ts POSITIONAL JOIN values;

-- LATERAL lets the right side reference the left side row-by-row
SELECT u.id, top_orders.*
FROM users u,
     LATERAL (
         SELECT * FROM orders o
         WHERE o.user_id = u.id
         ORDER BY o.amount DESC
         LIMIT 3
     ) top_orders;
```

### Trailing commas, multi-line strings, and friendly errors

```sql
SELECT
    id,
    name,
    email,        -- trailing comma allowed
FROM users
GROUP BY
    id,
    name,
;                  -- trailing comma allowed
```

### IS DISTINCT FROM and NULL-safe equality

```sql
SELECT * FROM users WHERE last_login IS DISTINCT FROM previous_login;
-- True even when one side is NULL — unlike `=`
```

## Aggregations

```sql
-- Basic aggregations
SELECT
    COUNT(*) AS total,
    COUNT(DISTINCT user_id) AS unique_users,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    STDDEV(amount) AS std_dev,
    MEDIAN(amount) AS median_amount
FROM orders;

-- Group by with ROLLUP
SELECT
    category,
    subcategory,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, subcategory);

-- Group by with CUBE
SELECT
    region,
    product,
    SUM(revenue) AS total
FROM sales
GROUP BY CUBE (region, product);

-- GROUPING SETS
SELECT
    region,
    product,
    SUM(revenue) AS total
FROM sales
GROUP BY GROUPING SETS ((region), (product), ());
```

## Time Series

```sql
-- Generate date series
SELECT * FROM generate_series(
    TIMESTAMP '2024-01-01',
    TIMESTAMP '2024-12-31',
    INTERVAL '1 day'
) AS t(date);

-- Time bucket
SELECT
    time_bucket(INTERVAL '1 hour', timestamp) AS hour,
    COUNT(*) AS events,
    AVG(value) AS avg_value
FROM events
GROUP BY hour
ORDER BY hour;

-- Date truncation
SELECT
    date_trunc('month', created_at) AS month,
    COUNT(*) AS user_count
FROM users
GROUP BY month;

-- Extract date parts
SELECT
    EXTRACT(year FROM created_at) AS year,
    EXTRACT(month FROM created_at) AS month,
    EXTRACT(day FROM created_at) AS day,
    EXTRACT(hour FROM created_at) AS hour
FROM events;
```

## Joins

```sql
-- Inner join
SELECT u.username, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- Left join
SELECT u.username, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Right join
SELECT u.username, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- Full outer join
SELECT u.username, o.amount
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;

-- Cross join
SELECT u.username, p.name
FROM users u
CROSS JOIN products p;

-- Join with USING
SELECT * FROM users u
JOIN orders o USING (user_id);

-- ASOF join (temporal join)
SELECT * FROM trades
ASOF JOIN quotes
ON trades.symbol = quotes.symbol
AND trades.timestamp >= quotes.timestamp;
```

## Common Table Expressions (CTEs)

```sql
-- Basic CTE
WITH active_users AS (
    SELECT * FROM users WHERE active = true
)
SELECT * FROM active_users WHERE created_at > '2024-01-01';

-- Multiple CTEs
WITH
    active_users AS (
        SELECT * FROM users WHERE active = true
    ),
    recent_orders AS (
        SELECT * FROM orders WHERE created_at > '2024-01-01'
    )
SELECT u.username, COUNT(o.id) AS order_count
FROM active_users u
LEFT JOIN recent_orders o ON u.id = o.user_id
GROUP BY u.username;

-- Recursive CTE
WITH RECURSIVE countdown(n) AS (
    SELECT 10 AS n
    UNION ALL
    SELECT n - 1 FROM countdown WHERE n > 1
)
SELECT * FROM countdown;
```

## Pivot and Unpivot

```sql
-- Pivot
PIVOT sales
ON product_category
USING SUM(amount)
GROUP BY region;

-- Manual pivot
SELECT
    region,
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = 'Clothing' THEN amount ELSE 0 END) AS clothing,
    SUM(CASE WHEN category = 'Food' THEN amount ELSE 0 END) AS food
FROM sales
GROUP BY region;

-- Unpivot
UNPIVOT sales
ON electronics, clothing, food
INTO NAME category VALUE amount;
```

## String Functions

```sql
-- String operations
SELECT
    UPPER(username) AS upper_name,
    LOWER(username) AS lower_name,
    CONCAT(first_name, ' ', last_name) AS full_name,
    SUBSTRING(email, 1, 5) AS email_prefix,
    LENGTH(username) AS name_length,
    REPLACE(email, '@gmail.com', '@example.com') AS new_email,
    SPLIT_PART(email, '@', 1) AS email_user,
    TRIM(username) AS trimmed,
    REGEXP_MATCHES(text, '[0-9]+') AS numbers,
    REGEXP_REPLACE(text, '[0-9]', 'X') AS masked
FROM users;

-- String aggregation
SELECT
    category,
    STRING_AGG(product_name, ', ') AS products
FROM products
GROUP BY category;

-- List functions
SELECT
    LIST(['a', 'b', 'c']) AS my_list,
    LIST_VALUE('a', 'b', 'c') AS another_list,
    [1, 2, 3] AS numeric_list;

SELECT list[1] FROM (SELECT [1, 2, 3] AS list);
```

## Array and Struct Operations

```sql
-- Arrays
SELECT [1, 2, 3, 4, 5] AS numbers;
SELECT LIST_VALUE(1, 2, 3, 4, 5) AS numbers;
SELECT UNNEST([1, 2, 3]) AS num;

-- Array aggregation
SELECT LIST(username) AS all_users FROM users;

-- Struct
SELECT {'name': 'John', 'age': 30} AS person;
SELECT person.name FROM (SELECT {'name': 'John', 'age': 30} AS person);

-- Nested structures
SELECT {
    'user': {'name': 'John', 'email': 'john@example.com'},
    'orders': [1, 2, 3]
} AS complex_data;
```

## Constraints and Indexes

```sql
-- Primary key
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL
);

-- Check constraint
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    price DECIMAL CHECK (price > 0),
    quantity INTEGER CHECK (quantity >= 0)
);

-- Create index
CREATE INDEX idx_users_email ON users(email);

-- Drop index
DROP INDEX idx_users_email;

-- Show indexes
PRAGMA show_index('users');
```

## Transactions

```sql
-- Begin transaction
BEGIN TRANSACTION;

INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- Commit
COMMIT;

-- Rollback
ROLLBACK;
```

## Views

```sql
-- Create view
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE active = true;

-- Use view
SELECT * FROM active_users;

-- Drop view
DROP VIEW active_users;
```

## Macros (SQL UDFs)

Macros are inlined SQL functions — no per-row interpreter overhead.

```sql
-- Scalar macro
CREATE MACRO add_one(x) AS x + 1;
SELECT add_one(41);  -- 42

-- Multi-argument with defaults
CREATE MACRO bucket(value, width := 10) AS (value // width) * width;
SELECT bucket(57);             -- 50
SELECT bucket(57, width := 5); -- 55

-- Table macro (returns a relation)
CREATE MACRO recent_orders(days) AS TABLE
    SELECT * FROM orders
    WHERE created_at > CURRENT_DATE - INTERVAL (days) DAY;

SELECT * FROM recent_orders(7);

-- Inspect macros
SELECT * FROM duckdb_functions() WHERE function_type = 'macro';
```

## Python UDFs

Python functions registered with type hints are callable from SQL.

```python
import duckdb
from duckdb.typing import VARCHAR, INTEGER

con = duckdb.connect()

# Scalar UDF
def cap(s: str) -> str:
    return s.upper()

con.create_function('cap', cap, [VARCHAR], VARCHAR)
con.sql("SELECT cap('hello')").show()  # HELLO

# Vectorized UDF (PyArrow) — orders of magnitude faster for batch work
import pyarrow as pa
def add_one_arrow(arr: pa.Array) -> pa.Array:
    import pyarrow.compute as pc
    return pc.add(arr, 1)

con.create_function('add_one', add_one_arrow,
                    [INTEGER], INTEGER, type='arrow')
con.sql("SELECT add_one(range) FROM range(5)").show()
```

## Prepared Statements

```sql
PREPARE top_n_orders(INTEGER) AS
    SELECT * FROM orders ORDER BY amount DESC LIMIT $1;

EXECUTE top_n_orders(10);

DEALLOCATE top_n_orders;
```

```python
# Positional placeholders
con.execute("SELECT * FROM users WHERE id = ?", [42])

# Named placeholders
con.execute("SELECT * FROM users WHERE id = $id AND active = $a",
            {'id': 42, 'a': True})

# Bulk insert with executemany
con.executemany("INSERT INTO users(id, email) VALUES (?, ?)",
                [(1, 'a@x'), (2, 'b@x'), (3, 'c@x')])
```

## Recursive CTEs

```sql
-- Graph traversal: find all descendants of node 1
WITH RECURSIVE descendants(id, parent_id, depth) AS (
    SELECT id, parent_id, 0 FROM nodes WHERE id = 1
    UNION ALL
    SELECT n.id, n.parent_id, d.depth + 1
    FROM nodes n
    JOIN descendants d ON n.parent_id = d.id
)
SELECT * FROM descendants ORDER BY depth, id;

-- Fibonacci
WITH RECURSIVE fib(i, a, b) AS (
    SELECT 1, 0, 1
    UNION ALL
    SELECT i + 1, b, a + b FROM fib WHERE i < 20
)
SELECT i, a FROM fib;
```

## Advanced Window Functions

```sql
-- Range frame in time units
SELECT
    sensor_id, ts, value,
    AVG(value) OVER (
        PARTITION BY sensor_id
        ORDER BY ts
        RANGE BETWEEN INTERVAL '5' MINUTE PRECEDING AND CURRENT ROW
    ) AS avg_5m
FROM readings;

-- EXCLUDE clauses (CURRENT ROW, GROUP, TIES, NO OTHERS)
SELECT id, score,
       SUM(score) OVER (ORDER BY id
                        ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING
                        EXCLUDE CURRENT ROW) AS context_sum
FROM scores;

-- Named window definitions
SELECT
    region,
    ts,
    SUM(revenue) OVER w7  AS rev_7d,
    SUM(revenue) OVER w30 AS rev_30d
FROM daily_revenue
WINDOW
    w7  AS (PARTITION BY region ORDER BY ts ROWS 6 PRECEDING),
    w30 AS (PARTITION BY region ORDER BY ts ROWS 29 PRECEDING);
```

## Approximate Aggregates

For huge datasets where exact answers are too expensive.

```sql
-- HyperLogLog cardinality estimate
SELECT approx_count_distinct(user_id) FROM events;

-- T-Digest quantiles
SELECT
    approx_quantile(latency_ms, 0.50) AS p50,
    approx_quantile(latency_ms, 0.99) AS p99,
    approx_quantile(latency_ms, [0.5, 0.9, 0.99]) AS quantiles
FROM requests;

-- Reservoir sample of values (exact, bounded memory)
SELECT reservoir_quantile(value, 0.95, 8192) FROM measurements;

-- Top-K most common values (space-saving)
SELECT approx_top_k(country, 10) FROM users;
```

## Sequences

```sql
CREATE SEQUENCE order_id_seq START 1000 INCREMENT 1;

CREATE TABLE orders (
    id INTEGER DEFAULT nextval('order_id_seq'),
    user_id INTEGER,
    amount DECIMAL
);

INSERT INTO orders (user_id, amount) VALUES (1, 99.99);
SELECT currval('order_id_seq');  -- 1000

-- Reset
ALTER SEQUENCE order_id_seq RESTART WITH 5000;
DROP SEQUENCE order_id_seq;
```

## Performance Optimization

### Plans and profiling

```sql
-- Logical plan, physical plan, both
EXPLAIN SELECT * FROM users WHERE username = 'john';
EXPLAIN (FORMAT TEXT) SELECT ...;
EXPLAIN (FORMAT JSON) SELECT ...;

-- Run + measure
EXPLAIN ANALYZE SELECT * FROM users JOIN orders ON users.id = orders.user_id;

-- Detailed JSON profile to a file
SET enable_profiling = 'json';
SET profiling_output = '/tmp/profile.json';
SET profiling_mode = 'detailed';

-- Reset profiling
SET enable_profiling = '';

-- Progress bar for long queries
SET enable_progress_bar = true;

-- Threading and memory
SET threads TO 4;
SET memory_limit = '4GB';
SET temp_directory = '/path/to/temp';
```

### Parquet writes

```sql
-- Single file
COPY events TO 'events.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

-- Tune row group size to your read pattern (larger = better scan throughput,
-- smaller = better filter pushdown selectivity)
COPY events TO 'events.parquet'
(FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION ZSTD);

-- Hive-partitioned write (per-partition directories with key=value names)
COPY events TO 'events/'
(FORMAT PARQUET, PARTITION_BY (year, month), OVERWRITE_OR_IGNORE);

-- File-per-thread for parallel ingest downstream
COPY (FROM events) TO 'shards/' (FORMAT PARQUET, PER_THREAD_OUTPUT true);
```

### Parquet reads and pushdown

```sql
-- Filter pushdown: predicate pushes into the Parquet reader,
-- skipping row groups via min/max statistics.
SELECT * FROM 's3://lake/events/*.parquet'
WHERE event_date BETWEEN '2025-05-01' AND '2025-05-31';

-- Projection pushdown: only requested columns are decoded.
SELECT user_id, amount FROM 's3://lake/orders/*.parquet';

-- Hive partition pruning: directory keys turn into predicates.
SELECT * FROM read_parquet('s3://lake/events/*/*/*.parquet',
                            hive_partitioning = true)
WHERE year = 2025 AND month = 5;

-- What pushes down? Equality, ranges, IN-lists, AND/OR — yes.
-- Function calls on the column (LOWER(col) = 'x') usually don't.
-- Cast the literal, not the column.
```

### Indexes

DuckDB rarely needs explicit indexes — zone maps + columnar layout + vectorized scans cover most analytical workloads. Add ART indexes only for selective point lookups in a primary/foreign key column.

```sql
-- ART (adaptive radix tree) index
CREATE INDEX idx_orders_user ON orders(user_id);

-- Auto-created for PRIMARY KEY and UNIQUE constraints

-- Inspect
SELECT * FROM duckdb_indexes();
```

### Common pitfalls

- **Row-at-a-time inserts** kill throughput. Batch with `INSERT INTO ... SELECT FROM read_parquet(...)` or `executemany`.
- **CSV inference on huge files** is slow — pass explicit `types=` to `read_csv`.
- **Cross joins from missing predicates** — DuckDB will happily do `N×M`. Watch the cardinality column in `EXPLAIN`.
- **String columns in joins** — cast to integer keys when possible; dictionary-encoded VARCHARs are fast but still wider than `INTEGER`.
- **Updating large tables**: prefer `CREATE TABLE new AS SELECT ...; DROP old; ALTER new RENAME` for full rewrites.
- **`SELECT *` over remote Parquet**: pay for every column. List the ones you need.

### Benchmarking checklist

- Run on a `:memory:` database for pure compute; on a file DB for full pipeline.
- `PRAGMA disable_object_cache;` between runs to compare cold-cache performance.
- Use `EXPLAIN ANALYZE` (not wallclock) for per-operator times.
- Pin threads (`SET threads TO 1`) when comparing algorithmic changes.
- Use `time_bucket` + `tpch(SF)` / `tpcds(SF)` extensions for reproducible workloads.

```sql
INSTALL tpch; LOAD tpch;
CALL dbgen(sf = 1);
PRAGMA tpch(7);
```

## Settings and Configuration

```sql
-- Show settings
SELECT * FROM duckdb_settings();

-- Set configuration
SET memory_limit = '8GB';
SET threads TO 8;
SET max_memory = '16GB';
SET temp_directory = '/tmp';

-- Progress bar
SET enable_progress_bar = true;

-- Profiling
SET enable_profiling = true;
SET profiling_mode = 'detailed';
```

## Importing from Other Databases

```sql
-- Attach SQLite database
ATTACH 'mydb.sqlite' AS sqlite_db (TYPE SQLITE);
SELECT * FROM sqlite_db.users;

-- Attach PostgreSQL
ATTACH 'dbname=mydb user=postgres host=localhost' AS pg_db (TYPE POSTGRES);
SELECT * FROM pg_db.users;

-- Copy data
CREATE TABLE local_users AS
SELECT * FROM pg_db.users;

-- Detach
DETACH sqlite_db;
```

## Extensions

DuckDB ships a minimal core; capabilities like S3 reads, Iceberg, full-text search, and vector search are loaded on demand.

### Managing extensions

```sql
-- List available extensions and their state
SELECT extension_name, installed, loaded, description
FROM duckdb_extensions();

-- Install + load (the common pair)
INSTALL httpfs;
LOAD httpfs;

-- Force a specific repo (community extensions live in 'community')
INSTALL h3 FROM community;
LOAD h3;

-- Pin to a specific version
INSTALL httpfs VERSION 'v1.1.3';

-- Autoload trusted extensions on first reference
SET autoinstall_known_extensions = true;
SET autoload_known_extensions = true;

-- Where extensions live
SELECT current_setting('extension_directory');
```

### httpfs and S3

`httpfs` reads `http(s)://`, `s3://`, `gcs://`, `r2://`, `azure://`, and `hf://` URLs as if they were local files.

```sql
INSTALL httpfs; LOAD httpfs;

-- One-off public file
SELECT COUNT(*) FROM 'https://example.com/data.parquet';

-- S3 credentials via SECRET (preferred over environment vars)
CREATE SECRET aws_prod (
    TYPE S3,
    KEY_ID 'AKIA...',
    SECRET 'wJal...',
    REGION 'us-east-1'
);

-- Or use the AWS provider chain (env, profile, instance role)
CREATE SECRET aws_chain (
    TYPE S3,
    PROVIDER credential_chain,
    CHAIN 'config;sts;env;sso',
    REGION 'us-east-1'
);

-- Read from S3
SELECT * FROM 's3://my-bucket/events/2025/*/*.parquet'
WHERE event_type = 'purchase'
LIMIT 100;

-- Globs with Hive-style partition discovery
SELECT *
FROM read_parquet('s3://my-bucket/events/year=*/month=*/*.parquet',
                  hive_partitioning = true);

-- Write back to S3
COPY (SELECT * FROM purchases WHERE year = 2025)
TO 's3://my-bucket/exports/purchases_2025/'
(FORMAT PARQUET, PARTITION_BY (month), OVERWRITE_OR_IGNORE);

-- Hugging Face datasets (hf:// is built on httpfs)
SELECT * FROM 'hf://datasets/squad/squad/plain_text/train-*.parquet' LIMIT 5;
```

### Iceberg

```sql
INSTALL iceberg; LOAD iceberg;

-- Scan an Iceberg table by path
SELECT * FROM iceberg_scan('s3://lake/warehouse/db/orders');

-- Read a specific snapshot
SELECT * FROM iceberg_scan(
    's3://lake/warehouse/db/orders',
    snapshot_id = 1234567890
);

-- Inspect metadata
SELECT * FROM iceberg_snapshots('s3://lake/warehouse/db/orders');
SELECT * FROM iceberg_metadata('s3://lake/warehouse/db/orders');

-- Attach a REST catalog (Polaris, Nessie, etc.)
ATTACH 'warehouse' AS lake (
    TYPE ICEBERG,
    ENDPOINT 'https://catalog.example.com',
    TOKEN 'abc...'
);
SELECT * FROM lake.db.orders;
```

### Delta Lake

```sql
INSTALL delta; LOAD delta;

SELECT * FROM delta_scan('s3://lake/delta/events');

-- Time travel
SELECT * FROM delta_scan('s3://lake/delta/events', version = 42);
```

### JSON

```sql
-- The json extension is built in. Reads detect format automatically.
SELECT * FROM read_json_auto('logs.json');
SELECT * FROM read_json('logs.jsonl', format = 'newline_delimited');

-- JSON path operators
SELECT
    j->'user'->>'name'      AS user_name,    -- ->> returns VARCHAR
    j->'tags'->0            AS first_tag,    -- -> returns JSON
    json_extract(j, '$.user.email') AS email,
    json_extract_string(j, '$.user.email') AS email_str
FROM events;

-- Materialize JSON into typed STRUCT
SELECT json_transform(
    '{"id": 1, "ts": "2025-05-28T10:00:00"}',
    '{"id": "INTEGER", "ts": "TIMESTAMP"}'
);

-- Aggregate to JSON
SELECT json_group_array(json_object('id', id, 'name', name))
FROM users;
```

### Spatial

```sql
INSTALL spatial; LOAD spatial;

-- Create geometries
SELECT ST_Point(longitude, latitude) AS pt FROM cities;

-- Distance (in CRS units; use ST_Distance_Sphere for meters on WGS84)
SELECT a.name, b.name, ST_Distance(a.pt, b.pt) AS dist
FROM cities a, cities b
WHERE a.name = 'Berlin' AND b.name <> a.name
ORDER BY dist
LIMIT 5;

-- Read GeoParquet / Shapefile / GeoJSON / GDAL formats
SELECT * FROM ST_Read('countries.shp');
SELECT * FROM ST_Read('admin.geojson');

-- Spatial joins
SELECT c.name, COUNT(*) AS station_count
FROM cities c, stations s
WHERE ST_Within(s.geom, c.boundary)
GROUP BY c.name;
```

### Full-text search

```sql
INSTALL fts; LOAD fts;

-- Build an index over text columns
PRAGMA create_fts_index(
    'documents',     -- table
    'id',            -- key column
    'title', 'body', -- text columns
    stemmer = 'porter',
    stopwords = 'english',
    ignore = '(\\.|[^a-z])+'
);

-- BM25 score
SELECT id, title, fts_main_documents.match_bm25(id, 'machine learning') AS score
FROM documents
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT 10;

-- Drop the index
PRAGMA drop_fts_index('documents');
```

### Vector similarity search (vss)

For embeddings / ANN search; pairs well with [[hnsw]] and [[product_quantization]] notes.

```sql
INSTALL vss; LOAD vss;

-- Fixed-length float array column
CREATE TABLE docs (
    id INTEGER,
    title VARCHAR,
    embedding FLOAT[384]
);

INSERT INTO docs VALUES (1, 'duckdb', [0.1, 0.2, /* ... */]::FLOAT[384]);

-- Distance functions
SELECT id,
       array_distance(embedding, $1::FLOAT[384])      AS l2,
       array_cosine_distance(embedding, $1::FLOAT[384]) AS cos,
       array_inner_product(embedding, $1::FLOAT[384])  AS dot
FROM docs
ORDER BY cos
LIMIT 10;

-- HNSW index for fast top-k (only when persisted DB; experimental flag)
SET hnsw_enable_experimental_persistence = true;
CREATE INDEX docs_hnsw ON docs USING HNSW (embedding)
WITH (metric = 'cosine', M = 16, ef_construction = 200);

-- Query uses the index transparently
SELECT id, array_cosine_distance(embedding, $1::FLOAT[384]) AS d
FROM docs ORDER BY d LIMIT 10;
```

### Postgres / MySQL / SQLite scanners

```sql
INSTALL postgres; LOAD postgres;
INSTALL mysql;    LOAD mysql;
-- sqlite is bundled

-- Attach a live Postgres instance and query directly
CREATE SECRET pg_dev (
    TYPE POSTGRES,
    HOST 'localhost', PORT 5432, DATABASE 'app',
    USER 'analyst', PASSWORD 's3cret'
);
ATTACH '' AS pg (TYPE POSTGRES, SECRET pg_dev);

-- Cross-engine join: DuckDB-local Parquet ⋈ Postgres rows
SELECT u.id, u.email, p.total_spent
FROM pg.users u
JOIN read_parquet('s3://lake/spend/*.parquet') p
  ON u.id = p.user_id;

-- Write back to Postgres
COPY (SELECT * FROM local_results)
TO pg.public.results (USE_TMP_FILE false);
```

### Other useful extensions

```sql
INSTALL excel;  LOAD excel;
SELECT * FROM read_xlsx('quarterly.xlsx', sheet = 'Q2');

INSTALL avro; LOAD avro;
SELECT * FROM read_avro('events.avro');

INSTALL icu; LOAD icu;
-- Adds proper unicode collation and timezone support
SELECT '2025-05-28 10:00 America/New_York'::TIMESTAMPTZ;

-- Community extensions worth knowing
INSTALL h3 FROM community;          -- Uber H3 hex grid
INSTALL prql FROM community;        -- PRQL → SQL
INSTALL ulid FROM community;        -- ULID generation
INSTALL substrait FROM community;   -- Substrait plan I/O
```

## Python Integration

```python
import duckdb
import pandas as pd

# Create connection
con = duckdb.connect('mydb.duckdb')

# Query to DataFrame
df = con.execute("SELECT * FROM users").df()

# Register DataFrame as table
con.register('df_users', df)
result = con.execute("SELECT * FROM df_users WHERE age > 30").df()

# Direct query on DataFrame
result = duckdb.query("SELECT * FROM df WHERE column_a > 10").df()

# Arrow integration
import pyarrow as pa
arrow_table = con.execute("SELECT * FROM users").arrow()

# Register Arrow table
con.register('arrow_users', arrow_table)

# Relation API
rel = con.table('users')
result = rel.filter('age > 30').project('username, email').df()

# Close
con.close()
```

## Arrow Integration

DuckDB and Apache Arrow share an in-memory format; data moves between them zero-copy.

```python
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

con = duckdb.connect()

# Arrow → DuckDB (registered as a virtual table, no copy)
table = pq.read_table('events.parquet')
con.register('events_arrow', table)
con.sql("SELECT user_id, SUM(amount) FROM events_arrow GROUP BY user_id").show()

# DuckDB → Arrow (zero-copy)
arrow_table = con.sql("SELECT * FROM events").arrow()

# RecordBatchReader for streaming
reader = con.sql("FROM huge_table").fetch_record_batch(rows_per_batch=10_000)
for batch in reader:
    process(batch)

# Arrow Database Connectivity (ADBC) — DBAPI 2.0 over Arrow
import adbc_driver_duckdb.dbapi as adbc
conn = adbc.connect('mydb.duckdb')
cur = conn.cursor()
cur.execute("SELECT * FROM events LIMIT 100")
arrow_table = cur.fetch_arrow_table()
```

## Polars Integration

```python
import duckdb
import polars as pl

# Query a Polars DataFrame directly
df = pl.read_parquet('events.parquet')
result = duckdb.sql("SELECT user_id, SUM(amount) FROM df GROUP BY 1").pl()

# Lazy frames work too (DuckDB collects on execute)
lf = pl.scan_parquet('events.parquet').filter(pl.col('amount') > 0)
duckdb.sql("FROM lf SELECT * USING SAMPLE 1000").pl()

# Round-trip
duckdb.sql("FROM read_parquet('events.parquet')").pl()
```

## dbt-duckdb

```yaml
# profiles.yml
analytics:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: dev.duckdb
      threads: 8
      extensions:
        - httpfs
        - iceberg
      settings:
        memory_limit: '8GB'
      external_root: 's3://lake/analytics'
```

```sql
-- models/orders_daily.sql
{{ config(materialized='external', format='parquet',
          partition_by=['order_date']) }}

SELECT
    order_date,
    SUM(amount) AS revenue,
    COUNT(*)    AS orders
FROM {{ source('raw', 'orders') }}
GROUP BY order_date
```

When to reach for `dbt-duckdb`: local development of a warehouse model graph, CI tests against canned fixtures, lakehouse-on-Parquet without a warehouse.

## MotherDuck

MotherDuck is a hosted DuckDB; it executes parts of a query in the cloud and others locally, transparently.

```sql
-- Attach a cloud DB by URL (DUCKDB_TOKEN env var holds the auth token)
ATTACH 'md:' AS cloud;
ATTACH 'md:my_database' AS prod;

-- Hybrid query: local CSV ⋈ cloud table
SELECT u.id, u.name, SUM(l.amount)
FROM cloud.users u
JOIN 'local_events.csv' l ON l.user_id = u.id
GROUP BY u.id, u.name;

-- Share read-only access
GRANT SELECT ON cloud.users TO 'colleague@example.com';
```

## Language Clients

```javascript
// Node.js — @duckdb/node-api (Neo client)
import { DuckDBInstance } from '@duckdb/node-api';
const instance = await DuckDBInstance.create('mydb.duckdb');
const conn = await instance.connect();
const reader = await conn.runAndReadAll('SELECT 42 AS answer');
console.log(reader.getRows());
```

```r
# R
library(duckdb)
con <- dbConnect(duckdb(), 'mydb.duckdb')
dbGetQuery(con, "SELECT COUNT(*) FROM users")
dbDisconnect(con, shutdown = TRUE)
```

```java
// Java — JDBC
Class.forName("org.duckdb.DuckDBDriver");
try (Connection conn = DriverManager.getConnection("jdbc:duckdb:mydb.duckdb");
     Statement stmt = conn.createStatement();
     ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM users")) {
    while (rs.next()) System.out.println(rs.getLong(1));
}
```

```go
// Go — marcboeker/go-duckdb
import (
    "database/sql"
    _ "github.com/marcboeker/go-duckdb"
)
db, _ := sql.Open("duckdb", "mydb.duckdb")
defer db.Close()
var n int
db.QueryRow("SELECT COUNT(*) FROM users").Scan(&n)
```

```rust
// Rust — duckdb crate
use duckdb::{Connection, params};
let conn = Connection::open("mydb.duckdb")?;
let mut stmt = conn.prepare("SELECT COUNT(*) FROM users")?;
let n: i64 = stmt.query_row([], |r| r.get(0))?;
```

## HTTP Server

```sql
-- Community extension: serves SQL over HTTP for tooling integration
INSTALL httpserver FROM community; LOAD httpserver;
SELECT httpserve_start('0.0.0.0', 9999, 'optional_basic_auth');

-- Test
-- curl -G 'http://localhost:9999/' --data-urlencode 'q=SELECT 42'
```

## CLI Commands

```bash
# Meta-commands
.help              # Show help
.tables            # List tables
.schema            # Show all schemas
.schema users      # Show table schema
.mode              # Show output mode
.mode csv          # Set CSV output
.mode json         # Set JSON output
.mode markdown     # Set Markdown output
.output file.csv   # Output to file
.timer on          # Show query timing
.maxrows 100       # Limit output rows
.quit              # Exit
```

## Best Practices

```sql
-- 1. Use columnar storage (Parquet) for large datasets
COPY large_table TO 'data.parquet' (FORMAT PARQUET);

-- 2. Leverage parallel execution
SET threads TO 8;

-- 3. Use appropriate data types
CREATE TABLE optimized (
    id INTEGER,
    name VARCHAR,
    value DOUBLE,
    date DATE
);

-- 4. Create indexes for frequently filtered columns
CREATE INDEX idx_users_email ON users(email);

-- 5. Use window functions instead of self-joins
SELECT username, LAG(score) OVER (ORDER BY date) AS prev_score
FROM scores;

-- 6. Partition large queries
SELECT * FROM large_table
WHERE date >= '2024-01-01' AND date < '2024-02-01';

-- 7. Use CTEs for readability
WITH filtered AS (SELECT * FROM users WHERE active = true)
SELECT * FROM filtered;

-- 8. Analyze queries for optimization
EXPLAIN ANALYZE SELECT * FROM complex_query;

-- 9. Read directly from files when possible
SELECT * FROM 'data.parquet' WHERE column > 100;

-- 10. Use appropriate compression
COPY data TO 'compressed.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
```

## DuckDB vs SQLite

| Aspect | DuckDB | SQLite |
|---|---|---|
| Workload | OLAP (analytics) | OLTP (transactions) |
| Storage | Columnar, compressed | Row-based pages |
| Execution | Vectorized, multi-threaded | Tuple-at-a-time, single-thread per query |
| Concurrency | Snapshot isolation, single writer | Multi-reader, single writer (file-locked) |
| Typical query | Scan/aggregate 100M+ rows | Lookup by primary key, small joins |
| File format | `.duckdb`, columnar | `.sqlite`, row-paged |
| Best for | Notebooks, ETL, embedded analytics | Apps with concurrent small reads/writes |

Both are embedded, ACID, single-file. The right answer is usually "use both" — SQLite for app state, DuckDB for analysis.

## DuckDB vs ClickHouse

| Aspect | DuckDB | ClickHouse |
|---|---|---|
| Deployment | Embedded library | Distributed server |
| Scale | Single node (TB-ish) | Cluster (PB-scale) |
| Ingest model | Batch (`COPY`, `INSERT FROM`) | Streaming (Kafka, real-time inserts) |
| Replication | No (single-file) | Built-in (replicas, shards) |
| Operations | None | ZooKeeper/Keeper + monitoring |
| SQL dialect | Postgres-flavored | Custom (close to SQL) |
| Best for | Ad-hoc analysis, lakehouse reads, ETL | Logs, metrics, ad-tech at huge scale |

Use DuckDB when there's no cluster to run; use ClickHouse when ingest rate or dataset size requires one.

## DuckDB vs Polars

| Aspect | DuckDB | Polars |
|---|---|---|
| API | SQL | DataFrame (Python/Rust) |
| Execution | Vectorized, push-based, parallel | Vectorized (Arrow2), parallel, lazy |
| Persistence | Native DB file + Parquet | Reads/writes files only |
| Joins | Hash, merge, asof, lateral, positional | Hash, merge, asof, cross |
| Larger-than-memory | Yes (spilling) | Streaming engine (partial) |
| Catalog | Yes (tables, views, attached DBs) | No |
| Best for | SQL users, multi-table workloads, persistence | Imperative pipelines, tight Python loops |

They interoperate zero-copy via Arrow — pick the API that fits the task, not the engine.

## DuckDB vs Pandas

| Aspect | DuckDB | Pandas |
|---|---|---|
| Memory model | Columnar, compressed, out-of-core | In-memory NumPy arrays |
| Parallelism | All cores by default | Single-threaded (without ext libs) |
| SQL | First-class | None natively |
| Joins on large data | Spills to disk | OOMs |
| Type system | SQL types, including STRUCT/LIST | NumPy dtypes (object-y for strings) |
| API ergonomics | SQL strings | Pythonic chaining |

Common pattern: keep Pandas for last-mile work, push joins/aggregations down to DuckDB.

```python
# DuckDB query reading a Pandas DataFrame directly
import duckdb, pandas as pd
df = pd.read_csv('big.csv')
result = duckdb.sql("FROM df SELECT region, SUM(amount) GROUP BY region").df()
```

## Decision Cheatsheet

- **Need a SQL engine inside an app, embedded, no server?** → DuckDB (or SQLite for OLTP).
- **Querying Parquet/CSV files locally?** → DuckDB, faster than reading into Pandas.
- **Building a warehouse on object storage?** → DuckDB + dbt-duckdb or MotherDuck.
- **Hundreds of concurrent writers?** → not DuckDB. Use Postgres/ClickHouse.
- **Stream ingest of millions of events/sec?** → not DuckDB. Use ClickHouse/Kafka.
- **Notebook exploration, mixed Python and SQL?** → DuckDB Python API.

## Quick Reference

| Command | Description |
|---------|-------------|
| `read_csv_auto('file.csv')` | Read CSV file (infer types) |
| `read_parquet('s3://b/**/*.parquet')` | Read Parquet (glob, remote OK) |
| `read_json_auto('events.jsonl')` | Read JSON / JSONL |
| `iceberg_scan('s3://lake/db/t')` | Read Iceberg table |
| `delta_scan('s3://lake/delta/t')` | Read Delta table |
| `COPY tbl TO 'file.parquet' (FORMAT PARQUET)` | Export to Parquet |
| `COPY tbl TO 'dir/' (PARTITION_BY (y, m))` | Hive-partitioned write |
| `INSTALL ext; LOAD ext` | Install + load extension |
| `CREATE SECRET s (TYPE S3, ...)` | Store credentials |
| `ATTACH 'md:'` | Attach MotherDuck |
| `ATTACH 'pg.dbname=...' (TYPE POSTGRES)` | Attach Postgres |
| `SELECT * EXCLUDE (col)` | Star with exclusions |
| `SELECT * REPLACE (expr AS col)` | Star with replacements |
| `GROUP BY ALL` | Group by every non-agg select |
| `QUALIFY rn = 1` | Filter on window output |
| `USING SAMPLE 1000` | Reservoir sample |
| `UNION BY NAME` | Column-name-aligned union |
| `EXPLAIN ANALYZE` | Plan + per-op timing |
| `PRAGMA enable_profiling = 'json'` | Detailed JSON profile |
| `PRAGMA show_index('tbl')` | Inspect indexes |
| `SET threads TO 8` | Thread count |
| `SET memory_limit = '8GB'` | RAM budget |
| `SET temp_directory = '/var/tmp'` | Spill location |
| `array_cosine_distance(v, q)` | Vector similarity |
| `approx_count_distinct(c)` | HLL cardinality |
| `time_bucket(INTERVAL '1 hour', ts)` | Time bucketing |
| `DESCRIBE tbl` / `.schema tbl` | Show schema |
| `.tables` / `SHOW TABLES` | List tables |
| `.mode csv` | CLI output format |
| `VACUUM` / `ANALYZE` | Optimize / refresh stats |
| `PRAGMA force_checkpoint` | Flush WAL to file |

DuckDB excels at analytical queries on local data files, making it perfect for data analysis, ETL pipelines, embedded analytics, and lakehouse-style reads against Iceberg/Delta on object storage.

## Where this connects

- [SQLite](sqlite.md) — OLTP counterpart; both are in-process and serverless, but SQLite targets transactional workloads
- [ClickHouse](clickhouse.md) — distributed OLAP alternative for scale; DuckDB is single-node, ClickHouse clusters to petabytes
- [PostgreSQL](postgres.md) — common source database; DuckDB's Postgres scanner reads Postgres tables directly
- [NoSQL](nosql.md) — DuckDB can query Parquet/JSON files produced by NoSQL systems (Kafka, MongoDB exports)
