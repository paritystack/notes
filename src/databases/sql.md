# SQL (Structured Query Language)

## Overview

SQL is the standard language for querying and managing relational databases. It is governed by the ISO/IEC 9075 standard (SQL-92, SQL:1999, SQL:2003, SQL:2016, SQL:2023), though every engine — PostgreSQL, MySQL, SQL Server, Oracle, SQLite, DuckDB — implements its own dialect with extensions and gaps. This page sticks to standard SQL and flags dialect-specific quirks inline.

SQL is *declarative*: you describe the result you want, and the engine's query planner decides how to compute it (which indexes to use, which join algorithm, in what order). Understanding how the planner thinks is what separates writing SQL from writing *good* SQL.

## Data Types

Standard column types fall into a few families. Exact spellings vary by engine.

```sql
-- Numeric
SMALLINT          -- 2-byte int
INTEGER / INT     -- 4-byte int
BIGINT            -- 8-byte int
NUMERIC(p, s)     -- exact decimal, p digits total, s after the point
DECIMAL(p, s)     -- synonym for NUMERIC
REAL              -- 4-byte float (approximate)
DOUBLE PRECISION  -- 8-byte float (approximate)

-- Character
CHAR(n)           -- fixed-length, padded with spaces
VARCHAR(n)        -- variable-length, max n characters
TEXT              -- variable-length, no fixed limit (most engines)

-- Boolean
BOOLEAN           -- TRUE / FALSE / NULL

-- Date and time
DATE              -- 2026-05-28
TIME              -- 14:30:00
TIMESTAMP         -- date + time, no timezone
TIMESTAMP WITH TIME ZONE   -- timestamptz; stores UTC, displays per session
INTERVAL          -- a duration ('3 days', '2 hours')

-- Binary / large objects
BLOB / BYTEA      -- raw bytes
JSON / JSONB      -- structured documents (Postgres uses JSONB for indexed access)

-- Identifiers
UUID              -- 128-bit identifier
SERIAL            -- Postgres: auto-incrementing INTEGER (legacy; prefer GENERATED)
GENERATED ALWAYS AS IDENTITY  -- standard auto-increment (SQL:2003)

-- Collections (Postgres, others)
INTEGER[]         -- one-dimensional array
```

Use `NUMERIC` for money, never `REAL`/`DOUBLE` — floating-point rounding will eat you alive.

## DDL (Schema Definition)

Data Definition Language statements create, alter, and drop schema objects.

```sql
-- Create a schema (namespace)
CREATE SCHEMA reporting;
DROP SCHEMA reporting CASCADE;

-- Create a table
CREATE TABLE users (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email       VARCHAR(255) NOT NULL UNIQUE,
    name        VARCHAR(100),
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- IF NOT EXISTS prevents errors on re-run
CREATE TABLE IF NOT EXISTS users (...);

-- Alter table
ALTER TABLE users ADD COLUMN age INTEGER;
ALTER TABLE users DROP COLUMN age;
ALTER TABLE users RENAME COLUMN name TO full_name;
ALTER TABLE users ALTER COLUMN email TYPE TEXT;
ALTER TABLE users RENAME TO accounts;

-- Drop
DROP TABLE users;
DROP TABLE IF EXISTS users CASCADE;  -- CASCADE drops dependent FKs/views

-- Truncate (fast bulk delete, can't ROLLBACK in all engines)
TRUNCATE TABLE users;
```

## Constraints

Constraints declare rules the engine enforces on every write. Push validation into the schema where you can — application-layer checks lie.

```sql
CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,                            -- unique + not null
    user_id     INTEGER NOT NULL,                               -- mandatory
    total       NUMERIC(10, 2) NOT NULL CHECK (total >= 0),     -- value rule
    status      VARCHAR(20) NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','paid','shipped','cancelled')),
    coupon_code VARCHAR(20),
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (user_id, coupon_code),                              -- composite uniqueness
    FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE                                       -- delete orders when user deleted
        ON UPDATE RESTRICT                                      -- block user.id changes
);

-- Add constraints later
ALTER TABLE orders ADD CONSTRAINT chk_total_positive CHECK (total > 0);
ALTER TABLE orders DROP CONSTRAINT chk_total_positive;
```

`ON DELETE` options: `CASCADE` (delete children), `SET NULL`, `SET DEFAULT`, `RESTRICT` (block), `NO ACTION` (block, but checked at end of statement).

## Basic Queries

### SELECT

```sql
SELECT column1, column2 FROM table WHERE condition;
SELECT * FROM users WHERE age > 18;
SELECT DISTINCT city FROM customers;

-- Sorting and limiting
SELECT * FROM users ORDER BY created_at DESC, id ASC;
SELECT * FROM users LIMIT 10 OFFSET 20;     -- pages 3 (10 per page)
SELECT * FROM users ORDER BY id FETCH FIRST 10 ROWS ONLY;  -- SQL standard
-- SQL Server: SELECT TOP 10 * FROM users;
```

### INSERT

```sql
-- Single row
INSERT INTO users (name, email) VALUES ('John', 'john@example.com');

-- Multiple rows in one statement (always cheaper than many INSERTs)
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob',   'bob@example.com');

-- From a query
INSERT INTO users_archive (id, name, email)
SELECT id, name, email FROM users WHERE is_active = FALSE;

-- Return generated values (Postgres, SQL Server uses OUTPUT)
INSERT INTO users (name) VALUES ('Jane') RETURNING id, created_at;
```

### UPDATE

```sql
UPDATE users SET age = 30 WHERE id = 1;
UPDATE products SET price = price * 1.1;

-- Update from another table (Postgres syntax)
UPDATE orders o
SET status = 'flagged'
FROM users u
WHERE o.user_id = u.id AND u.is_active = FALSE;
```

### DELETE

```sql
DELETE FROM users WHERE id = 1;
DELETE FROM logs WHERE created_at < '2023-01-01';

-- Delete using a join (Postgres)
DELETE FROM orders o
USING users u
WHERE o.user_id = u.id AND u.is_active = FALSE;
```

## Filtering & Operators

```sql
-- Comparison
WHERE age = 18
WHERE age <> 18           -- != also works in most engines
WHERE age > 18 AND age < 65
WHERE name LIKE 'Jo%'     -- % = any chars, _ = single char
WHERE name ILIKE 'jo%'    -- case-insensitive (Postgres)

-- Sets
WHERE id IN (1, 2, 3)
WHERE id NOT IN (1, 2, 3)
WHERE id IN (SELECT user_id FROM orders)

-- Ranges
WHERE created_at BETWEEN '2026-01-01' AND '2026-12-31'
WHERE price NOT BETWEEN 10 AND 20

-- Null tests
WHERE deleted_at IS NULL
WHERE deleted_at IS NOT NULL

-- Pattern matching (engine-specific)
WHERE email ~ '^[a-z]+@'      -- Postgres POSIX regex
WHERE email REGEXP '^[a-z]+@' -- MySQL
WHERE email LIKE 'a%' ESCAPE '\'  -- literal % needs escape
```

## NULL Handling

NULL means *unknown*. Three-valued logic (TRUE / FALSE / UNKNOWN) makes NULL surprisingly easy to mishandle.

```sql
-- These all evaluate to UNKNOWN, NOT TRUE
SELECT NULL = NULL;           -- UNKNOWN
SELECT NULL <> NULL;          -- UNKNOWN
SELECT NULL = 5;              -- UNKNOWN
SELECT NOT NULL;              -- UNKNOWN

-- WHERE filters keep only TRUE rows — UNKNOWN rows drop out silently
SELECT * FROM users WHERE manager_id <> 7;  -- excludes rows where manager_id IS NULL

-- The fix: explicit null handling
SELECT * FROM users WHERE manager_id <> 7 OR manager_id IS NULL;

-- Or use IS DISTINCT FROM (treats NULLs as comparable)
SELECT * FROM users WHERE manager_id IS DISTINCT FROM 7;

-- Replace nulls with a default
SELECT COALESCE(nickname, name, 'Unknown') FROM users;

-- Treat a sentinel value as NULL
SELECT NULLIF(score, -1) FROM games;   -- -1 becomes NULL

-- Aggregates skip NULLs silently
SELECT AVG(score) FROM games;          -- NULL scores excluded
SELECT COUNT(*) FROM games;            -- counts all rows
SELECT COUNT(score) FROM games;        -- counts non-NULL scores only
```

## CASE Expressions

CASE is SQL's `if/else`. Use it inline in SELECT, WHERE, ORDER BY, GROUP BY — anywhere an expression is allowed.

```sql
-- Searched CASE (most flexible)
SELECT name,
    CASE
        WHEN age < 13 THEN 'child'
        WHEN age < 20 THEN 'teen'
        WHEN age < 65 THEN 'adult'
        ELSE 'senior'
    END AS age_group
FROM users;

-- Simple CASE (compares one value)
SELECT
    CASE status
        WHEN 'paid'    THEN 'Done'
        WHEN 'pending' THEN 'Waiting'
        ELSE 'Other'
    END
FROM orders;

-- Conditional aggregation (a CASE inside an aggregate)
SELECT
    COUNT(*) FILTER (WHERE status = 'paid')      AS paid_count,    -- SQL:2003 FILTER
    COUNT(CASE WHEN status = 'paid' THEN 1 END)  AS paid_count_v2  -- portable equivalent
FROM orders;
```

## Joins

```sql
-- INNER JOIN: only matching rows
SELECT u.name, o.id
FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: all rows from left, NULLs where no match
SELECT u.name, o.id
FROM users u LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN: all rows from right (mirror of LEFT — usually rewritable as LEFT)
SELECT u.name, o.id
FROM users u RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN: all rows from both sides
SELECT u.name, o.id
FROM users u FULL OUTER JOIN orders o ON u.id = o.user_id;

-- CROSS JOIN: Cartesian product — every row of A paired with every row of B
SELECT s.size, c.color FROM sizes s CROSS JOIN colors c;

-- USING shorthand (when join columns share a name)
SELECT name, total FROM users u JOIN orders o USING (user_id);

-- Self join (a table joined to itself)
SELECT e.name AS employee, m.name AS manager
FROM employees e LEFT JOIN employees m ON e.manager_id = m.id;

-- Multiple join conditions
SELECT *
FROM orders o JOIN shipments s
    ON s.order_id = o.id AND s.shipped_at >= o.created_at;

-- Anti-join: rows in A with no match in B
SELECT u.* FROM users u
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
```

Prefer `NOT EXISTS` over `NOT IN` when the right side can contain NULLs — `NOT IN (NULL, ...)` returns no rows.

## Subqueries

A query nested inside another query.

```sql
-- Scalar subquery: returns one value
SELECT name,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count
FROM users u;

-- Subquery in WHERE
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 1000);

-- Correlated subquery (references outer table — re-evaluated per outer row)
SELECT u.name FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.total > 1000
);

-- Subquery in FROM (derived table)
SELECT category, avg_price
FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products GROUP BY category
) AS t
WHERE avg_price > 100;

-- ANY / ALL
SELECT * FROM products WHERE price > ALL (SELECT price FROM products WHERE category = 'books');
SELECT * FROM products WHERE price = ANY (SELECT price FROM featured);  -- equivalent to IN
```

Most planners will rewrite `IN (subquery)` and `EXISTS` to the same plan, but `EXISTS` is more explicit about intent and handles NULLs cleanly.

## Common Table Expressions (CTEs)

A CTE is a named subquery defined at the top of a statement. Use them to break complex queries into readable steps.

```sql
-- Single CTE
WITH active_users AS (
    SELECT id, name FROM users WHERE is_active = TRUE
)
SELECT * FROM active_users WHERE name LIKE 'A%';

-- Multiple CTEs (comma-separated, each can reference earlier ones)
WITH
recent_orders AS (
    SELECT user_id, SUM(total) AS spent
    FROM orders
    WHERE created_at > CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
),
big_spenders AS (
    SELECT user_id FROM recent_orders WHERE spent > 1000
)
SELECT u.name, r.spent
FROM users u
JOIN recent_orders r ON r.user_id = u.id
WHERE u.id IN (SELECT user_id FROM big_spenders);

-- Recursive CTE: walk a tree (here, employee → manager chain)
WITH RECURSIVE org_chart AS (
    -- Anchor: top-level managers
    SELECT id, name, manager_id, 1 AS depth
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive step: join children to the partial result
    SELECT e.id, e.name, e.manager_id, oc.depth + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY depth, name;

-- Recursive CTE for a sequence (1..10)
WITH RECURSIVE nums(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 10
)
SELECT * FROM nums;
```

Always include a termination condition in the recursive step or the query won't stop.

## Set Operations

Combine result sets that have the same column structure.

```sql
-- UNION: rows from either query, duplicates removed
SELECT email FROM customers
UNION
SELECT email FROM suppliers;

-- UNION ALL: same but keeps duplicates (much faster — no dedup sort)
SELECT email FROM customers
UNION ALL
SELECT email FROM suppliers;

-- INTERSECT: rows in both
SELECT email FROM customers
INTERSECT
SELECT email FROM newsletter_subscribers;

-- EXCEPT (MINUS in Oracle): rows in first that aren't in second
SELECT email FROM customers
EXCEPT
SELECT email FROM unsubscribed;
```

## Aggregation

```sql
SELECT COUNT(*) FROM users;
SELECT COUNT(DISTINCT email) FROM users;
SELECT AVG(price), SUM(amount), MIN(price), MAX(price) FROM products;
SELECT STRING_AGG(name, ', ') FROM users;  -- LISTAGG in Oracle, GROUP_CONCAT in MySQL

-- GROUP BY: collapse rows sharing a key
SELECT department, COUNT(*), AVG(salary)
FROM employees
GROUP BY department;

-- HAVING: filter groups (WHERE filters rows before grouping)
SELECT department, AVG(salary) AS avg_sal
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;

-- FILTER clause (SQL:2003): per-aggregate predicate
SELECT
    COUNT(*) FILTER (WHERE status = 'active')   AS active_count,
    COUNT(*) FILTER (WHERE status = 'inactive') AS inactive_count,
    AVG(salary) FILTER (WHERE department = 'Eng') AS eng_avg
FROM employees;

-- Grouping sets: multiple GROUP BYs in one query
SELECT department, role, COUNT(*)
FROM employees
GROUP BY GROUPING SETS ((department), (role), (department, role), ());

-- ROLLUP: hierarchical subtotals (department, then grand total)
SELECT department, role, COUNT(*)
FROM employees
GROUP BY ROLLUP (department, role);

-- CUBE: every combination of groupings
SELECT department, role, COUNT(*)
FROM employees
GROUP BY CUBE (department, role);
```

## Window Functions

Window functions compute per-row results over a related set of rows (the *window*) without collapsing the result set the way GROUP BY does.

```sql
-- Anatomy: FUNCTION() OVER (PARTITION BY ... ORDER BY ... frame)

-- Ranking
SELECT name, department, salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn,
    RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS rnk,    -- 1,2,2,4
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS drnk,   -- 1,2,2,3
    NTILE(4)     OVER (ORDER BY salary)                              AS quartile
FROM employees;

-- Running total / cumulative sum
SELECT date, amount,
    SUM(amount) OVER (ORDER BY date) AS running_total
FROM transactions;

-- Moving average (frame clause defines the window of rows)
SELECT date, price,
    AVG(price) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7d
FROM prices;

-- Compare to previous/next row
SELECT date, price,
    LAG(price)  OVER (ORDER BY date) AS prev_price,
    LEAD(price) OVER (ORDER BY date) AS next_price,
    price - LAG(price) OVER (ORDER BY date) AS daily_change
FROM prices;

-- First/last value in a partition
SELECT name, department, salary,
    FIRST_VALUE(name) OVER (PARTITION BY department ORDER BY salary DESC) AS top_paid,
    LAST_VALUE(name)  OVER (
        PARTITION BY department ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS bottom_paid
FROM employees;
```

Frame defaults are surprising: `ORDER BY` without an explicit frame defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which is why `LAST_VALUE` so often returns the current row unless you set the frame.

`RANGE` works on logical value ranges (peers tied on ORDER BY share a frame); `ROWS` works on physical row positions.

## String Functions

```sql
-- Concatenation: || is standard, CONCAT is widely supported
SELECT first_name || ' ' || last_name FROM users;
SELECT CONCAT(first_name, ' ', last_name) FROM users;

-- Length, case
SELECT CHAR_LENGTH(name), UPPER(name), LOWER(name) FROM users;

-- Substring (positions are 1-indexed)
SELECT SUBSTRING(email FROM 1 FOR 5) FROM users;
SELECT SUBSTRING(email, 1, 5) FROM users;       -- shorthand
SELECT POSITION('@' IN email) FROM users;

-- Trim
SELECT TRIM('  hello  ');                       -- 'hello'
SELECT TRIM(LEADING '0' FROM '00042');          -- '42'
SELECT BTRIM(name, ' ');                        -- Postgres

-- Replace
SELECT REPLACE(phone, '-', '');

-- Split (engine-specific)
SELECT SPLIT_PART(email, '@', 2) FROM users;    -- Postgres: domain part
SELECT SUBSTRING_INDEX(email, '@', -1) FROM users;  -- MySQL

-- Pad
SELECT LPAD(order_id::text, 8, '0');
SELECT RPAD(name, 20, '.');

-- Regex (Postgres)
SELECT regexp_replace(phone, '[^0-9]', '', 'g');
SELECT regexp_matches(text, '([A-Z]+)([0-9]+)');
```

## Date/Time Functions

```sql
-- Current values
SELECT CURRENT_DATE;                         -- 2026-05-28
SELECT CURRENT_TIME;
SELECT CURRENT_TIMESTAMP;                    -- with timezone
SELECT LOCALTIMESTAMP;                       -- without timezone
SELECT NOW();                                -- Postgres / MySQL alias

-- Extract parts
SELECT EXTRACT(YEAR FROM created_at);
SELECT EXTRACT(DOW  FROM created_at);        -- day of week
SELECT EXTRACT(EPOCH FROM created_at);       -- unix timestamp

-- Truncate to a unit (group by month, etc.)
SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*)
FROM orders
GROUP BY month;

-- Arithmetic with intervals
SELECT created_at + INTERVAL '7 days'  FROM orders;
SELECT created_at - INTERVAL '1 month' FROM orders;
SELECT CURRENT_DATE - DATE '2026-01-01';     -- difference in days

-- Format / parse
SELECT TO_CHAR(created_at, 'YYYY-MM-DD');    -- Postgres / Oracle
SELECT DATE_FORMAT(created_at, '%Y-%m-%d');  -- MySQL
SELECT TO_DATE('2026-05-28', 'YYYY-MM-DD');

-- Time zones
SELECT created_at AT TIME ZONE 'UTC';
SELECT created_at AT TIME ZONE 'America/New_York';
```

Store timestamps in UTC (`TIMESTAMPTZ` in Postgres). Convert at the edges of the system.

## JSON Operations

Standard SQL/JSON (SQL:2016) defines `JSON_VALUE`, `JSON_QUERY`, `JSON_TABLE`, but Postgres uses its older `->`/`->>` operators. Most modern engines now have first-class JSON support.

```sql
-- Table with a JSON column
CREATE TABLE events (
    id   INTEGER PRIMARY KEY,
    data JSON                       -- Postgres prefers JSONB for indexing
);

INSERT INTO events (id, data) VALUES
    (1, '{"type": "click", "user": {"id": 42}, "tags": ["a","b"]}');

-- Postgres operators
SELECT data -> 'user'                       FROM events;  -- JSON object
SELECT data ->> 'type'                      FROM events;  -- text
SELECT data #> '{user,id}'                  FROM events;  -- nested as JSON
SELECT data #>> '{user,id}'                 FROM events;  -- nested as text
SELECT data -> 'tags' -> 0                  FROM events;  -- array index
SELECT * FROM events WHERE data @> '{"type":"click"}';    -- containment

-- Standard SQL/JSON path expressions (SQL:2016)
SELECT JSON_VALUE(data, '$.user.id')        FROM events;  -- scalar
SELECT JSON_QUERY(data, '$.tags')           FROM events;  -- subtree
SELECT * FROM events WHERE JSON_EXISTS(data, '$.user.id');

-- Building JSON
SELECT JSON_OBJECT('id' VALUE id, 'name' VALUE name) FROM users;
SELECT JSON_ARRAYAGG(name) FROM users;
```

Index JSON keys you query often: in Postgres, `CREATE INDEX ON events USING GIN (data jsonb_path_ops);`.

## UPSERT

Insert if the row is new, update if it conflicts. The syntax is one of the worst-standardized parts of SQL.

```sql
-- Standard SQL:2003 MERGE (Postgres 15+, SQL Server, Oracle, DB2)
MERGE INTO users AS u
USING (VALUES (1, 'jane@example.com', 'Jane')) AS s(id, email, name)
ON u.id = s.id
WHEN MATCHED THEN UPDATE SET email = s.email, name = s.name
WHEN NOT MATCHED THEN INSERT (id, email, name) VALUES (s.id, s.email, s.name);

-- Postgres / SQLite: INSERT ... ON CONFLICT
INSERT INTO users (id, email, name)
VALUES (1, 'jane@example.com', 'Jane')
ON CONFLICT (id) DO UPDATE
    SET email = EXCLUDED.email,
        name  = EXCLUDED.name;

-- Postgres: skip duplicates
INSERT INTO users (email) VALUES ('a@example.com')
ON CONFLICT (email) DO NOTHING;

-- MySQL: INSERT ... ON DUPLICATE KEY UPDATE
INSERT INTO users (id, email, name) VALUES (1, 'jane@example.com', 'Jane')
ON DUPLICATE KEY UPDATE email = VALUES(email), name = VALUES(name);
```

## Indexes

Indexes trade write speed and disk space for read speed. Add them based on actual query patterns, not guesses.

```sql
-- Single-column B-tree (default for most engines)
CREATE INDEX idx_users_email ON users(email);

-- Composite index — order matters. Use the leftmost prefix rule.
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
-- This index helps: WHERE user_id = ?    and    WHERE user_id = ? AND created_at > ?
-- This index does NOT help: WHERE created_at > ?

-- Unique index (also enforces uniqueness)
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Partial index: only index rows matching a predicate (smaller, faster)
CREATE INDEX idx_orders_pending ON orders(created_at) WHERE status = 'pending';

-- Expression / functional index: index a computed value
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
-- Now WHERE LOWER(email) = ? can use the index

-- Covering index (INCLUDE non-key columns to satisfy queries from the index alone)
CREATE INDEX idx_orders_user ON orders(user_id) INCLUDE (total, status);

-- Inspect / drop
DROP INDEX idx_users_email;
```

### Index types (Postgres-flavored, common elsewhere)

| Type     | Good for                                                          |
|----------|-------------------------------------------------------------------|
| B-tree   | Equality and range on ordered values (default)                    |
| Hash     | Equality only, no range                                           |
| GIN      | Containment queries: arrays, full-text, JSONB                     |
| GiST     | Geometric / nearest-neighbor / range types                        |
| BRIN     | Very large tables where values correlate with physical row order  |

When indexes hurt: write-heavy tables (every INSERT/UPDATE/DELETE must update every index), tables that fit in memory and scan faster than they seek, columns with very low cardinality (boolean flags).

## Views

A view is a saved query that looks like a table.

```sql
-- Standard view: runs the query every time
CREATE VIEW active_users AS
SELECT id, name, email FROM users WHERE is_active = TRUE;

SELECT * FROM active_users;
DROP VIEW active_users;

-- Materialized view: stores the result on disk; must be refreshed
CREATE MATERIALIZED VIEW user_order_stats AS
SELECT user_id, COUNT(*) AS order_count, SUM(total) AS lifetime_value
FROM orders GROUP BY user_id;

REFRESH MATERIALIZED VIEW user_order_stats;
-- REFRESH MATERIALIZED VIEW CONCURRENTLY user_order_stats;  -- Postgres, needs unique idx

-- Updatable view: a simple SELECT can sometimes be INSERTed/UPDATEd through
CREATE VIEW vip_users AS SELECT * FROM users WHERE tier = 'vip';
UPDATE vip_users SET email = 'new@example.com' WHERE id = 1;
```

Materialized views are a cache. Standard views are an abstraction.

## Transactions

A transaction groups statements so they commit or roll back together.

```sql
BEGIN;  -- or START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;       -- save all changes
-- ROLLBACK;  -- undo all changes since BEGIN

-- Savepoints: partial rollback
BEGIN;
    INSERT INTO orders (...) VALUES (...);
    SAVEPOINT before_risky;
    UPDATE inventory SET qty = qty - 1 WHERE id = 99;
    -- oh no
    ROLLBACK TO SAVEPOINT before_risky;
    -- order INSERT survives
COMMIT;
```

### Isolation levels (SQL:1992)

| Level             | Dirty read | Non-repeatable read | Phantom read |
|-------------------|------------|---------------------|--------------|
| READ UNCOMMITTED  | possible   | possible            | possible     |
| READ COMMITTED    | prevented  | possible            | possible     |
| REPEATABLE READ   | prevented  | prevented           | possible     |
| SERIALIZABLE      | prevented  | prevented           | prevented    |

```sql
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
-- ...
COMMIT;
```

- **Dirty read**: see uncommitted data from another transaction.
- **Non-repeatable read**: same row read twice returns different values.
- **Phantom read**: same query returns different *rows* twice (new rows appeared).

Defaults vary: Postgres = READ COMMITTED, MySQL InnoDB = REPEATABLE READ, SQL Server = READ COMMITTED. SERIALIZABLE in Postgres uses SSI (serializable snapshot isolation) and may abort transactions with a serialization error — your app must retry.

### Locking

```sql
-- Lock rows you're about to update (pessimistic)
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;

-- Lock for read but allow other reads (block writes)
SELECT * FROM accounts WHERE id = 1 FOR SHARE;

-- Skip rows another transaction has locked (queue / worker patterns)
SELECT * FROM jobs WHERE status = 'queued'
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT 1;
```

## Permissions

```sql
-- Users vs roles: a role is a permission grouping; users are roles you can log in as
CREATE ROLE readonly;
CREATE USER analyst WITH PASSWORD 'secret';
GRANT readonly TO analyst;

-- Grants
GRANT SELECT ON users TO readonly;
GRANT SELECT, INSERT, UPDATE ON orders TO app_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO admin;
GRANT USAGE ON SCHEMA reporting TO analyst;

-- Revoke
REVOKE INSERT ON orders FROM app_user;

-- Row-level security (Postgres)
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_orders_own ON orders
    FOR SELECT USING (user_id = current_setting('app.user_id')::int);
```

## Query Optimization & EXPLAIN

The planner picks a plan; `EXPLAIN` shows it. `EXPLAIN ANALYZE` actually runs the query and reports real timings.

```sql
EXPLAIN SELECT * FROM users WHERE email = 'a@b.com';
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'a@b.com';
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;   -- Postgres: also shows cache hits
```

What to look for:
- **Seq Scan vs Index Scan vs Index Only Scan** — sequential scans on big tables are usually the problem.
- **Estimated vs actual row counts** — large divergence means stats are stale (`ANALYZE table;`) or the planner can't estimate well.
- **Join algorithm** — Nested Loop (good for small inputs), Hash Join (good for big unsorted inputs), Merge Join (good for sorted inputs).
- **Sort / Hash spilling to disk** — increase `work_mem` or rewrite to need less sorting.

### Common performance pitfalls

```sql
-- Function on indexed column kills index use
WHERE LOWER(email) = 'a@b.com'    -- bad, unless you have an expression index
WHERE email ILIKE 'a@b.com'       -- bad, no index help

-- Leading wildcard kills LIKE index use
WHERE email LIKE '%example.com'   -- bad
WHERE email LIKE 'alice%'         -- ok

-- Implicit type cast disables index
WHERE user_id = '42'              -- bad if user_id is INTEGER

-- OR across columns often defeats indexes — rewrite as UNION
SELECT * FROM users WHERE email = 'x' OR phone = 'y';
SELECT * FROM users WHERE email = 'x'
UNION
SELECT * FROM users WHERE phone = 'y';

-- N+1 query: don't loop selects in the app; use JOIN or IN

-- SELECT *: pulls every column over the network; specify what you need
```

General rules:
1. Index columns used in WHERE, JOIN, ORDER BY.
2. Keep statistics fresh: `ANALYZE` after large data changes.
3. Batch writes — many small inserts in one transaction beats many transactions.
4. `LIMIT` early when paging.
5. For paging, prefer keyset (`WHERE id > last_seen_id ORDER BY id LIMIT 20`) over `OFFSET` once the offset gets large.

## Common Patterns

### Find duplicates

```sql
SELECT email, COUNT(*)
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

### Top N per group

```sql
-- Window function approach (portable)
SELECT * FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
    FROM employees
) t WHERE rn <= 3;

-- Postgres shortcut: DISTINCT ON (one row per group)
SELECT DISTINCT ON (department) department, name, salary
FROM employees
ORDER BY department, salary DESC;
```

### Pivot (rows to columns)

```sql
SELECT
    user_id,
    SUM(CASE WHEN status = 'paid'    THEN total ELSE 0 END) AS paid_total,
    SUM(CASE WHEN status = 'pending' THEN total ELSE 0 END) AS pending_total
FROM orders
GROUP BY user_id;
```

### Running total

```sql
SELECT date, amount,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY date) AS running_total
FROM transactions;
```

### Gaps and islands (group consecutive sequences)

```sql
-- "Islands" of consecutive logins per user, using the row-number difference trick
SELECT user_id, MIN(login_date) AS start, MAX(login_date) AS end
FROM (
    SELECT user_id, login_date,
        login_date - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date)) * INTERVAL '1 day'
            AS grp
    FROM logins
) t
GROUP BY user_id, grp;
```

### Keyset pagination

```sql
-- Instead of: ... ORDER BY id LIMIT 20 OFFSET 100000  -- slow on big tables
SELECT * FROM users
WHERE id > :last_seen_id
ORDER BY id
LIMIT 20;
```

### Data validation

```sql
SELECT * FROM users WHERE email NOT LIKE '%@%.%';
```

## SQL Injection

Never concatenate user input into SQL. Always use parameterized queries.

```python
# Bad — string concatenation: user can submit "'; DROP TABLE users; --"
cur.execute("SELECT * FROM users WHERE name = '" + name + "'")

# Good — parameterized: driver escapes and binds safely
cur.execute("SELECT * FROM users WHERE name = %s", (name,))
```

```sql
-- Server-side prepared statement (Postgres)
PREPARE find_user (text) AS
    SELECT * FROM users WHERE email = $1;

EXECUTE find_user('a@b.com');
DEALLOCATE find_user;
```

Parameterized queries also let the planner cache the plan across calls.

## ACID Properties

- **Atomicity**: a transaction is all-or-nothing.
- **Consistency**: a transaction takes the database from one valid state to another (constraints, triggers, application invariants).
- **Isolation**: concurrent transactions don't interfere (controlled by isolation level).
- **Durability**: once committed, data survives crashes.

NoSQL systems often relax these in exchange for availability and scale; see [`acid_vs_base.md`](acid_vs_base.md).

## SQL Dialects

Common differences between major engines:

| Feature              | PostgreSQL                  | MySQL                            | SQL Server                  | SQLite                     | Oracle                          |
|----------------------|-----------------------------|----------------------------------|-----------------------------|----------------------------|---------------------------------|
| Auto-increment       | `GENERATED ... IDENTITY` / `SERIAL` | `AUTO_INCREMENT`         | `IDENTITY(1,1)`             | `INTEGER PRIMARY KEY`      | `GENERATED ... IDENTITY`        |
| Limit                | `LIMIT n OFFSET m`          | `LIMIT m, n`                     | `OFFSET m ROWS FETCH NEXT n` | `LIMIT n OFFSET m`        | `FETCH FIRST n ROWS ONLY`       |
| String concat        | `\|\|` or `CONCAT`          | `CONCAT` (`\|\|` = OR by default) | `+` or `CONCAT`             | `\|\|`                     | `\|\|` or `CONCAT`              |
| Current timestamp    | `NOW()` / `CURRENT_TIMESTAMP` | `NOW()`                        | `GETDATE()`                 | `CURRENT_TIMESTAMP`        | `SYSDATE` / `CURRENT_TIMESTAMP` |
| Boolean              | `BOOLEAN` (true type)       | `TINYINT(1)`                     | `BIT`                       | none (use INTEGER)         | none (use NUMBER(1) or CHAR(1)) |
| UPSERT               | `ON CONFLICT`               | `ON DUPLICATE KEY UPDATE`        | `MERGE`                     | `ON CONFLICT`              | `MERGE`                         |
| Case-insensitive LIKE| `ILIKE`                     | `LIKE` (default collation)       | `LIKE` (default collation)  | `LIKE` (default ASCII-only)| `LIKE` with `UPPER()`           |
| Identifier quoting   | `"name"`                    | `` `name` ``                     | `[name]` or `"name"`        | `"name"`                   | `"name"`                        |

When code targets multiple engines, lean on standard SQL and isolate dialect quirks behind a thin layer.

## ELI10

SQL is like a filing system for data:

- **SELECT**: "Show me these files"
- **INSERT**: "Add a new file"
- **UPDATE**: "Change this file"
- **DELETE**: "Throw this file away"

Joins are how you combine drawers from different filing cabinets ("for each customer, find their orders"). Indexes are the tabs that let you skip flipping through every page. Transactions are envelopes — everything inside happens together or not at all. Window functions let you write notes in the margin of each row that compare it to its neighbors.

## Further Resources

- [SQL Tutorial (W3Schools)](https://www.w3schools.com/sql/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [Use The Index, Luke!](https://use-the-index-luke.com/) — practical index tuning
- [Modern SQL](https://modern-sql.com/) — what's in recent SQL standards
- [Explain Plans (Postgres)](https://www.postgresql.org/docs/current/sql-explain.html)
- [SQL:2016 standard overview](https://en.wikipedia.org/wiki/SQL:2016)
