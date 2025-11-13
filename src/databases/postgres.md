# PostgreSQL

PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development. It's known for its reliability, feature robustness, and performance.

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql@15
brew services start postgresql@15

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib
sudo postgresql-setup initdb
sudo systemctl start postgresql

# Check version
psql --version
```

## Basic Usage

```bash
# Connect as postgres user
sudo -u postgres psql

# Connect to specific database
psql -U username -d database_name

# Connect to remote database
psql -h hostname -U username -d database_name

# Execute SQL file
psql -U username -d database_name -f script.sql

# Execute command from shell
psql -U username -d database_name -c "SELECT * FROM users;"
```

## Database Operations

```sql
-- Create database
CREATE DATABASE mydb;

-- List databases
\l
\list

-- Connect to database
\c mydb
\connect mydb

-- Drop database
DROP DATABASE mydb;

-- Create database with options
CREATE DATABASE mydb
    WITH OWNER = myuser
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;
```

## Table Operations

```sql
-- Create table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- List tables
\dt
\dt+  -- with sizes

-- Describe table
\d users
\d+ users  -- detailed

-- Drop table
DROP TABLE users;
DROP TABLE IF EXISTS users;

-- Alter table
ALTER TABLE users ADD COLUMN age INTEGER;
ALTER TABLE users DROP COLUMN age;
ALTER TABLE users RENAME COLUMN username TO user_name;
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
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

-- Insert with RETURNING
INSERT INTO users (username, email)
VALUES ('jane', 'jane@example.com')
RETURNING id, username;

-- Select
SELECT * FROM users;
SELECT username, email FROM users WHERE id = 1;
SELECT * FROM users WHERE username LIKE 'jo%';
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;

-- Update
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;
UPDATE users SET email = 'newemail@example.com' WHERE id = 1 RETURNING *;

-- Delete
DELETE FROM users WHERE id = 1;
DELETE FROM users WHERE created_at < '2023-01-01';
```

## Indexes

```sql
-- Create index
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Unique index
CREATE UNIQUE INDEX idx_users_username_unique ON users(username);

-- Composite index
CREATE INDEX idx_users_name_email ON users(username, email);

-- Partial index
CREATE INDEX idx_active_users ON users(username) WHERE active = true;

-- Full-text search index
CREATE INDEX idx_users_fulltext ON users USING GIN(to_tsvector('english', username || ' ' || email));

-- List indexes
\di
SELECT * FROM pg_indexes WHERE tablename = 'users';

-- Drop index
DROP INDEX idx_users_username;
```

## Constraints

```sql
-- Primary key
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- Foreign key
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id)
);

-- Unique constraint
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);

-- Check constraint
ALTER TABLE products ADD CONSTRAINT products_price_positive
    CHECK (price > 0);

-- Not null
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- Default
ALTER TABLE users ALTER COLUMN active SET DEFAULT true;
```

## Joins

```sql
-- Inner join
SELECT u.username, o.id AS order_id
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- Left join
SELECT u.username, o.id AS order_id
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Right join
SELECT u.username, o.id AS order_id
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- Full outer join
SELECT u.username, o.id AS order_id
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;

-- Self join
SELECT e1.name AS employee, e2.name AS manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;
```

## Aggregations

```sql
-- Count
SELECT COUNT(*) FROM users;
SELECT COUNT(DISTINCT email) FROM users;

-- Sum, Avg, Min, Max
SELECT
    COUNT(*) AS total_orders,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount
FROM orders;

-- Group by
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id;

-- Having
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id
HAVING COUNT(*) > 5;

-- Window functions
SELECT
    username,
    created_at,
    ROW_NUMBER() OVER (ORDER BY created_at) AS row_num,
    RANK() OVER (ORDER BY created_at) AS rank,
    LAG(created_at) OVER (ORDER BY created_at) AS prev_created
FROM users;
```

## Transactions

```sql
-- Begin transaction
BEGIN;

INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Commit
COMMIT;

-- Rollback
ROLLBACK;

-- Savepoint
BEGIN;
INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
SAVEPOINT my_savepoint;
UPDATE users SET email = 'new@example.com' WHERE username = 'test';
ROLLBACK TO my_savepoint;
COMMIT;
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

-- Materialized view
CREATE MATERIALIZED VIEW user_stats AS
SELECT
    user_id,
    COUNT(*) AS order_count,
    SUM(amount) AS total_spent
FROM orders
GROUP BY user_id;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW user_stats;

-- Drop view
DROP VIEW active_users;
DROP MATERIALIZED VIEW user_stats;
```

## Functions and Procedures

```sql
-- Create function
CREATE OR REPLACE FUNCTION get_user_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM users);
END;
$$ LANGUAGE plpgsql;

-- Call function
SELECT get_user_count();

-- Function with parameters
CREATE OR REPLACE FUNCTION get_user_by_id(user_id INTEGER)
RETURNS TABLE(username VARCHAR, email VARCHAR) AS $$
BEGIN
    RETURN QUERY
    SELECT u.username, u.email
    FROM users u
    WHERE u.id = user_id;
END;
$$ LANGUAGE plpgsql;

-- Call
SELECT * FROM get_user_by_id(1);

-- Procedure (PostgreSQL 11+)
CREATE OR REPLACE PROCEDURE add_user(
    p_username VARCHAR,
    p_email VARCHAR
)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO users (username, email)
    VALUES (p_username, p_email);
END;
$$;

-- Call procedure
CALL add_user('newuser', 'new@example.com');
```

## Triggers

```sql
-- Create trigger function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER update_users_modtime
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- List triggers
\dft
SELECT * FROM pg_trigger WHERE tgrelid = 'users'::regclass;

-- Drop trigger
DROP TRIGGER update_users_modtime ON users;
```

## JSON Operations

```sql
-- JSON column
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    data JSONB
);

-- Insert JSON
INSERT INTO events (data) VALUES ('{"type": "click", "count": 1}');

-- Query JSON
SELECT data->>'type' AS event_type FROM events;
SELECT * FROM events WHERE data->>'type' = 'click';
SELECT * FROM events WHERE data->'count' > '5';

-- Update JSON
UPDATE events SET data = jsonb_set(data, '{count}', '10') WHERE id = 1;

-- JSON aggregation
SELECT jsonb_agg(username) FROM users;
SELECT jsonb_object_agg(id, username) FROM users;
```

## Full-Text Search

```sql
-- Create tsvector column
ALTER TABLE articles ADD COLUMN textsearch tsvector;

-- Update tsvector
UPDATE articles SET textsearch =
    to_tsvector('english', title || ' ' || body);

-- Create index
CREATE INDEX idx_articles_textsearch ON articles USING GIN(textsearch);

-- Search
SELECT title FROM articles
WHERE textsearch @@ to_tsquery('english', 'postgresql & performance');

-- Ranking
SELECT title, ts_rank(textsearch, query) AS rank
FROM articles, to_tsquery('english', 'postgresql') query
WHERE textsearch @@ query
ORDER BY rank DESC;
```

## User Management

```sql
-- Create user
CREATE USER myuser WITH PASSWORD 'mypassword';

-- Create role
CREATE ROLE readonly;

-- Grant privileges
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
GRANT SELECT, INSERT, UPDATE ON users TO myuser;

-- Revoke privileges
REVOKE INSERT ON users FROM myuser;

-- Alter user
ALTER USER myuser WITH PASSWORD 'newpassword';
ALTER USER myuser WITH SUPERUSER;

-- Drop user
DROP USER myuser;

-- List users
\du
SELECT * FROM pg_user;
```

## Backup and Restore

```bash
# Dump database
pg_dump -U username -d mydb > mydb_backup.sql
pg_dump -U username -d mydb -F c > mydb_backup.dump

# Dump specific table
pg_dump -U username -d mydb -t users > users_backup.sql

# Dump all databases
pg_dumpall -U postgres > all_dbs.sql

# Restore from SQL file
psql -U username -d mydb < mydb_backup.sql

# Restore from custom format
pg_restore -U username -d mydb mydb_backup.dump

# Restore specific table
pg_restore -U username -d mydb -t users mydb_backup.dump
```

## Performance Tuning

```sql
-- Analyze table
ANALYZE users;

-- Vacuum
VACUUM users;
VACUUM FULL users;
VACUUM ANALYZE users;

-- Explain query
EXPLAIN SELECT * FROM users WHERE username = 'john';
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'john';

-- Query statistics
SELECT * FROM pg_stat_user_tables WHERE relname = 'users';
SELECT * FROM pg_stat_user_indexes WHERE relname = 'users';

-- Active connections
SELECT * FROM pg_stat_activity;

-- Kill query
SELECT pg_cancel_backend(pid);
SELECT pg_terminate_backend(pid);

-- Table size
SELECT pg_size_pretty(pg_total_relation_size('users'));
```

## Configuration

```bash
# postgresql.conf key settings

# Memory
shared_buffers = 256MB          # 25% of RAM
effective_cache_size = 1GB      # 50-75% of RAM
work_mem = 4MB
maintenance_work_mem = 64MB

# WAL
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 1GB

# Query planner
random_page_cost = 1.1         # For SSD
effective_io_concurrency = 200  # For SSD

# Connections
max_connections = 100

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_duration = on
log_min_duration_statement = 1000  # Log queries > 1s
```

## psql Commands

```bash
# Meta-commands
\?              # Help on psql commands
\h ALTER TABLE  # Help on SQL command
\l              # List databases
\c dbname       # Connect to database
\dt             # List tables
\dt+            # List tables with sizes
\d tablename    # Describe table
\d+ tablename   # Detailed table info
\di             # List indexes
\dv             # List views
\df             # List functions
\du             # List users
\dn             # List schemas
\timing         # Toggle timing
\x              # Toggle expanded output
\q              # Quit
\! command      # Execute shell command
\i file.sql     # Execute SQL file
\o file.txt     # Output to file
\o              # Output to stdout
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `\l` | List databases |
| `\c database` | Connect to database |
| `\dt` | List tables |
| `\d table` | Describe table |
| `\di` | List indexes |
| `\du` | List users |
| `EXPLAIN` | Show query plan |
| `VACUUM` | Cleanup database |
| `pg_dump` | Backup database |
| `psql -f file.sql` | Execute SQL file |

PostgreSQL is a robust, feature-rich database system suitable for applications ranging from small projects to large-scale enterprise systems.
