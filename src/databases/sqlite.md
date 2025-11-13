# SQLite

SQLite is a C-language library that implements a small, fast, self-contained, high-reliability, full-featured SQL database engine. It's the most widely deployed database in the world.

## Overview

SQLite is embedded into the application, requiring no separate server process. The entire database is stored in a single cross-platform file.

**Key Features:**
- Serverless, zero-configuration
- Self-contained (single file database)
- Cross-platform
- ACID compliant
- Supports most SQL standards
- Public domain (no license required)

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install sqlite3

# macOS (pre-installed, or use Homebrew)
brew install sqlite

# CentOS/RHEL
sudo yum install sqlite

# Verify
sqlite3 --version
```

## Basic Usage

```bash
# Create/open database
sqlite3 mydb.db

# Open existing database
sqlite3 existing.db

# Execute command from shell
sqlite3 mydb.db "SELECT * FROM users;"

# Execute SQL file
sqlite3 mydb.db < script.sql

# Dump database
sqlite3 mydb.db .dump > backup.sql

# Exit
.quit
.exit
```

## Database Operations

```sql
-- Attach database
ATTACH DATABASE 'other.db' AS other;

-- List databases
.databases

-- Detach
DETACH DATABASE other;

-- Backup database
.backup backup.db

-- Restore from backup
.restore backup.db
```

## Table Operations

```sql
-- Create table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- List tables
.tables
.schema

-- Show table schema
.schema users
PRAGMA table_info(users);

-- Drop table
DROP TABLE users;
DROP TABLE IF EXISTS users;

-- Rename table
ALTER TABLE users RENAME TO customers;

-- Add column
ALTER TABLE users ADD COLUMN age INTEGER;

-- Rename column (SQLite 3.25.0+)
ALTER TABLE users RENAME COLUMN username TO user_name;

-- Drop column (SQLite 3.35.0+)
ALTER TABLE users DROP COLUMN age;
```

## Data Types

```sql
-- SQLite has 5 storage classes
-- INTEGER, REAL, TEXT, BLOB, NULL

CREATE TABLE examples (
    int_col INTEGER,
    real_col REAL,
    text_col TEXT,
    blob_col BLOB,

    -- Type affinity examples
    bool_col BOOLEAN,         -- Stored as INTEGER (0 or 1)
    date_col DATE,            -- Stored as TEXT, INTEGER, or REAL
    datetime_col DATETIME,
    varchar_col VARCHAR(100), -- Stored as TEXT
    decimal_col DECIMAL(10,2) -- Stored as REAL or TEXT
);
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

-- Insert or replace
INSERT OR REPLACE INTO users (id, username, email)
VALUES (1, 'john', 'newemail@example.com');

-- Insert or ignore
INSERT OR IGNORE INTO users (username, email)
VALUES ('john', 'john@example.com');

-- Select
SELECT * FROM users;
SELECT username, email FROM users WHERE id = 1;
SELECT * FROM users WHERE username LIKE 'jo%';
SELECT * FROM users ORDER BY created_at DESC LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;

-- Update
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;

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
CREATE INDEX idx_active_users ON users(username) WHERE active = 1;

-- Expression index
CREATE INDEX idx_users_lower_username ON users(LOWER(username));

-- List indexes
.indexes
.indexes users
PRAGMA index_list(users);

-- Show index info
PRAGMA index_info(idx_users_username);

-- Drop index
DROP INDEX idx_users_username;
```

## Constraints

```sql
-- Primary key
CREATE TABLE products (
    id INTEGER PRIMARY KEY,  -- Alias for rowid
    name TEXT NOT NULL
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);

-- Foreign key (must enable)
PRAGMA foreign_keys = ON;

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Unique constraint
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE
);

-- Check constraint
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    price REAL CHECK(price > 0),
    quantity INTEGER CHECK(quantity >= 0)
);

-- Not null
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL
);

-- Default value
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
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

-- Cross join
SELECT u.username, p.name
FROM users u
CROSS JOIN products p;

-- Natural join (not recommended)
SELECT * FROM users NATURAL JOIN orders;
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

-- Group concat
SELECT user_id, GROUP_CONCAT(product_name, ', ') AS products
FROM order_items
GROUP BY user_id;
```

## Transactions

```sql
-- Begin transaction
BEGIN TRANSACTION;

INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Commit
COMMIT;

-- Rollback
ROLLBACK;

-- Transaction modes
BEGIN DEFERRED TRANSACTION;   -- Default
BEGIN IMMEDIATE TRANSACTION;  -- Acquire write lock
BEGIN EXCLUSIVE TRANSACTION;  -- Exclusive access

-- Savepoint
BEGIN;
INSERT INTO users (username, email) VALUES ('test', 'test@example.com');
SAVEPOINT sp1;
UPDATE users SET email = 'new@example.com' WHERE username = 'test';
ROLLBACK TO sp1;
COMMIT;
```

## Views

```sql
-- Create view
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE active = 1;

-- Use view
SELECT * FROM active_users;

-- Temporary view
CREATE TEMP VIEW temp_users AS
SELECT * FROM users WHERE created_at > date('now', '-7 days');

-- Drop view
DROP VIEW active_users;
```

## Triggers

```sql
-- Before insert trigger
CREATE TRIGGER validate_email
BEFORE INSERT ON users
BEGIN
    SELECT CASE
        WHEN NEW.email NOT LIKE '%@%' THEN
            RAISE(ABORT, 'Invalid email format')
    END;
END;

-- After insert trigger
CREATE TRIGGER log_user_creation
AFTER INSERT ON users
BEGIN
    INSERT INTO audit_log (table_name, action, timestamp)
    VALUES ('users', 'INSERT', datetime('now'));
END;

-- Update trigger
CREATE TRIGGER update_modified_time
AFTER UPDATE ON users
BEGIN
    UPDATE users SET updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- Instead of trigger (for views)
CREATE TRIGGER update_active_users
INSTEAD OF UPDATE ON active_users
BEGIN
    UPDATE users SET email = NEW.email WHERE id = NEW.id;
END;

-- List triggers
.schema users
SELECT * FROM sqlite_master WHERE type = 'trigger';

-- Drop trigger
DROP TRIGGER validate_email;
```

## Date and Time

```sql
-- Current date/time
SELECT date('now');               -- 2024-01-15
SELECT time('now');               -- 14:30:45
SELECT datetime('now');           -- 2024-01-15 14:30:45
SELECT strftime('%Y-%m-%d %H:%M', 'now');

-- Date arithmetic
SELECT date('now', '+7 days');
SELECT date('now', '-1 month');
SELECT datetime('now', '+5 hours');
SELECT date('now', 'start of month');
SELECT date('now', 'start of year');

-- Extract parts
SELECT strftime('%Y', 'now') AS year;
SELECT strftime('%m', 'now') AS month;
SELECT strftime('%d', 'now') AS day;
SELECT strftime('%H', 'now') AS hour;

-- Julian day
SELECT julianday('now');
SELECT julianday('now') - julianday('2024-01-01');

-- Unix timestamp
SELECT strftime('%s', 'now');  -- Unix timestamp
SELECT datetime(1234567890, 'unixepoch');  -- From timestamp
```

## JSON Operations (SQLite 3.38.0+)

```sql
-- JSON functions
SELECT json('{"name":"John","age":30}');

-- Extract value
SELECT json_extract('{"name":"John","age":30}', '$.name');
SELECT '{"name":"John","age":30}' -> 'name';  -- Shorthand

-- Array operations
SELECT json_each.value
FROM json_each('[1,2,3,4,5]');

-- Store JSON
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    data TEXT
);

INSERT INTO events (data) VALUES ('{"type":"click","count":1}');

-- Query JSON
SELECT * FROM events
WHERE json_extract(data, '$.type') = 'click';

-- Update JSON
UPDATE events
SET data = json_set(data, '$.count', json_extract(data, '$.count') + 1)
WHERE id = 1;
```

## Full-Text Search

```sql
-- Create FTS5 table
CREATE VIRTUAL TABLE articles_fts USING fts5(
    title,
    body,
    content=articles,
    content_rowid=id
);

-- Populate FTS table
INSERT INTO articles_fts(rowid, title, body)
SELECT id, title, body FROM articles;

-- Search
SELECT * FROM articles_fts WHERE articles_fts MATCH 'sqlite performance';

-- Ranking
SELECT *, rank FROM articles_fts
WHERE articles_fts MATCH 'sqlite'
ORDER BY rank;

-- Phrase search
SELECT * FROM articles_fts WHERE articles_fts MATCH '"sqlite database"';

-- Column-specific search
SELECT * FROM articles_fts WHERE title MATCH 'tutorial';
```

## Pragma Statements

```sql
-- Database info
PRAGMA database_list;
PRAGMA table_info(users);
PRAGMA index_list(users);
PRAGMA foreign_key_list(orders);

-- Performance
PRAGMA cache_size = 10000;        -- Pages in cache
PRAGMA page_size = 4096;          -- Page size in bytes
PRAGMA journal_mode = WAL;        -- Write-Ahead Logging
PRAGMA synchronous = NORMAL;      -- Sync mode
PRAGMA temp_store = MEMORY;       -- Temp tables in memory

-- Foreign keys
PRAGMA foreign_keys = ON;
PRAGMA foreign_keys;              -- Check status

-- Integrity check
PRAGMA integrity_check;
PRAGMA quick_check;

-- Database size
PRAGMA page_count;
PRAGMA page_size;
-- Total size = page_count * page_size

-- Optimization
PRAGMA optimize;
VACUUM;
```

## Performance Optimization

```sql
-- Enable WAL mode (Write-Ahead Logging)
PRAGMA journal_mode = WAL;

-- Increase cache size
PRAGMA cache_size = -64000;  -- 64MB

-- Disable synchronous (faster but less safe)
PRAGMA synchronous = OFF;
PRAGMA synchronous = NORMAL;  -- Balanced

-- Analyze tables
ANALYZE;
ANALYZE users;

-- Vacuum database
VACUUM;

-- Batch inserts
BEGIN TRANSACTION;
-- Multiple INSERT statements
COMMIT;

-- Use prepared statements (in code)
-- Better performance and security

-- Indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);
```

## Backup and Recovery

```bash
# Backup database
sqlite3 mydb.db ".backup backup.db"
sqlite3 mydb.db .dump > backup.sql
cp mydb.db mydb_backup.db  # Simple copy

# Restore from backup
sqlite3 newdb.db ".restore backup.db"
sqlite3 newdb.db < backup.sql

# Export to CSV
.mode csv
.output users.csv
SELECT * FROM users;
.output stdout

# Import from CSV
.mode csv
.import users.csv users
```

## SQLite CLI Commands

```bash
# Meta-commands
.help               # Show help
.databases          # List databases
.tables             # List tables
.schema             # Show all schemas
.schema users       # Show table schema
.indexes users      # Show indexes
.mode column        # Column output mode
.mode csv           # CSV output mode
.mode json          # JSON output mode
.headers on         # Show column headers
.width 10 20 30     # Set column widths
.output file.txt    # Output to file
.output stdout      # Output to screen
.read file.sql      # Execute SQL file
.timer on           # Show execution time
.quit               # Exit
```

## Common Patterns

```sql
-- Upsert (Insert or Update)
INSERT INTO users (id, username, email)
VALUES (1, 'john', 'john@example.com')
ON CONFLICT(id) DO UPDATE SET
    username = excluded.username,
    email = excluded.email;

-- Conditional insert
INSERT INTO users (username, email)
SELECT 'john', 'john@example.com'
WHERE NOT EXISTS (
    SELECT 1 FROM users WHERE username = 'john'
);

-- Auto-increment
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT
);

-- Get last insert rowid
SELECT last_insert_rowid();

-- Pagination
SELECT * FROM users
ORDER BY id
LIMIT 10 OFFSET 20;

-- Random row
SELECT * FROM users ORDER BY RANDOM() LIMIT 1;
```

## Best Practices

```sql
-- 1. Enable foreign keys
PRAGMA foreign_keys = ON;

-- 2. Use WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- 3. Use transactions for bulk operations
BEGIN TRANSACTION;
-- Multiple operations
COMMIT;

-- 4. Create indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);

-- 5. Use INTEGER PRIMARY KEY for auto-increment
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT
);

-- 6. Analyze database periodically
ANALYZE;

-- 7. Use prepared statements in code
-- Prevents SQL injection and improves performance

-- 8. Vacuum database periodically
VACUUM;

-- 9. Use appropriate data types
-- SQLite is flexible but using correct types helps

-- 10. Regular backups
-- Use .backup command or copy the database file
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `.tables` | List tables |
| `.schema TABLE` | Show table structure |
| `.mode column` | Set output format |
| `.headers on` | Show column headers |
| `.backup FILE` | Backup database |
| `.import FILE TABLE` | Import CSV |
| `PRAGMA foreign_keys=ON` | Enable foreign keys |
| `PRAGMA journal_mode=WAL` | Enable WAL mode |
| `VACUUM` | Optimize database |
| `ANALYZE` | Update statistics |

SQLite is ideal for embedded systems, mobile apps, desktop applications, and scenarios where a simple, reliable, serverless database is needed.
