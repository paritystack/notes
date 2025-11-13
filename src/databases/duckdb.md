# DuckDB

DuckDB is an in-process SQL OLAP (Online Analytical Processing) database management system designed for analytical query workloads. It's often described as "SQLite for analytics."

## Overview

DuckDB is optimized for analytical queries with columnar storage, vectorized execution, and minimal dependencies.

**Key Features:**
- In-process, embedded database
- Columnar storage for analytics
- ACID compliant
- Vectorized query execution
- No external dependencies
- SQL compatible
- Direct querying of CSV, Parquet, JSON

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

## Performance Optimization

```sql
-- Analyze query plan
EXPLAIN SELECT * FROM users WHERE username = 'john';
EXPLAIN ANALYZE SELECT * FROM users JOIN orders ON users.id = orders.user_id;

-- Vacuum and analyze
VACUUM;
ANALYZE users;

-- Parallel query execution (automatic)
SET threads TO 4;

-- Memory limit
SET memory_limit = '4GB';

-- Temp directory
SET temp_directory = '/path/to/temp';
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

## Quick Reference

| Command | Description |
|---------|-------------|
| `read_csv_auto('file.csv')` | Read CSV file |
| `read_parquet('file.parquet')` | Read Parquet file |
| `COPY table TO 'file.csv'` | Export to CSV |
| `EXPLAIN ANALYZE` | Show query plan |
| `SET threads TO 8` | Set thread count |
| `DESCRIBE table` | Show table schema |
| `.tables` | List tables |
| `.mode csv` | Set output format |
| `VACUUM` | Optimize database |
| `ANALYZE` | Update statistics |

DuckDB excels at analytical queries on local data files, making it perfect for data analysis, ETL pipelines, and embedded analytics applications.
