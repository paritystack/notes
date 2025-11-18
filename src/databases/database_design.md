# Database Design

A comprehensive guide to designing robust, scalable, and maintainable database schemas.

## Overview

Database design is the process of organizing data according to a database model. Good database design ensures data integrity, minimizes redundancy, and optimizes performance.

## Database Design Process

### 1. Requirements Analysis

Understand what data needs to be stored and how it will be used:

```
Business Requirements:
├─ What data needs to be stored?
├─ Who will access the data?
├─ What operations will be performed?
├─ What are the performance requirements?
└─ What are the scalability needs?
```

### 2. Conceptual Design (ER Diagram)

Create an Entity-Relationship diagram:

```
┌─────────────┐         ┌─────────────┐
│   User      │         │    Order    │
├─────────────┤         ├─────────────┤
│ id (PK)     │────────<│ id (PK)     │
│ name        │   1:N   │ user_id(FK) │
│ email       │         │ total       │
│ created_at  │         │ status      │
└─────────────┘         │ created_at  │
                        └─────────────┘
```

### 3. Logical Design (Schema)

Convert ER diagram to relational schema:

```sql
-- Entities become tables
-- Attributes become columns
-- Relationships become foreign keys

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Physical Design

Optimize for performance:

```sql
-- Add indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Add partitioning for large tables
CREATE TABLE orders (
    id SERIAL,
    user_id INTEGER,
    total DECIMAL(10,2),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Normalization

The process of organizing data to reduce redundancy and improve data integrity.

### First Normal Form (1NF)

**Rule**: Each column contains atomic (indivisible) values, no repeating groups.

**Bad** (Not 1NF):
```sql
CREATE TABLE users (
    id INT,
    name VARCHAR(100),
    phone_numbers VARCHAR(255)  -- "555-1234, 555-5678"
);
```

**Good** (1NF):
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE user_phones (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    phone_number VARCHAR(20)
);
```

### Second Normal Form (2NF)

**Rule**: 1NF + no partial dependencies (all non-key columns depend on the entire primary key).

**Bad** (Not 2NF):
```sql
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    product_name VARCHAR(100),  -- Depends only on product_id
    product_price DECIMAL(10,2), -- Depends only on product_id
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

**Good** (2NF):
```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2)
);

CREATE TABLE order_items (
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);
```

### Third Normal Form (3NF)

**Rule**: 2NF + no transitive dependencies (non-key columns depend only on the primary key).

**Bad** (Not 3NF):
```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    customer_name VARCHAR(100),  -- Depends on customer_id, not order id
    customer_email VARCHAR(255), -- Transitive dependency
    total DECIMAL(10,2)
);
```

**Good** (3NF):
```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(255)
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total DECIMAL(10,2)
);
```

### Boyce-Codd Normal Form (BCNF)

**Rule**: 3NF + for every dependency X → Y, X must be a superkey.

**Example**:
```sql
-- Not BCNF: professor determines course, but professor is not a superkey
CREATE TABLE teaching (
    student_id INT,
    course_id INT,
    professor_id INT,
    PRIMARY KEY (student_id, course_id)
);

-- BCNF: Split into two tables
CREATE TABLE course_professors (
    course_id INT PRIMARY KEY,
    professor_id INT
);

CREATE TABLE student_courses (
    student_id INT,
    course_id INT REFERENCES course_professors(course_id),
    PRIMARY KEY (student_id, course_id)
);
```

## Denormalization

Intentionally introducing redundancy for performance.

### When to Denormalize

1. **Read-heavy workloads** where JOINs are expensive
2. **Reporting and analytics** queries
3. **Caching frequently accessed data**
4. **Reducing JOIN complexity**

### Example: Denormalization for Performance

**Normalized** (requires JOINs):
```sql
SELECT
    o.id,
    o.total,
    c.name as customer_name,
    c.email as customer_email
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

**Denormalized** (faster reads):
```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    customer_name VARCHAR(100),  -- Denormalized
    customer_email VARCHAR(255), -- Denormalized
    total DECIMAL(10,2)
);

-- Much faster query
SELECT id, total, customer_name, customer_email
FROM orders;
```

**Trade-offs**:
- Faster reads
- More storage space
- Complex updates (must update multiple places)
- Risk of data inconsistency

## Relationship Types

### One-to-One (1:1)

```sql
-- User has one profile
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE user_profiles (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    bio TEXT,
    avatar_url VARCHAR(255)
);
```

### One-to-Many (1:N)

```sql
-- User has many posts
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50)
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(200),
    content TEXT
);
```

### Many-to-Many (M:N)

Requires a junction/join table:

```sql
-- Students enroll in many courses
-- Courses have many students
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- Junction table
CREATE TABLE enrollments (
    student_id INTEGER REFERENCES students(id),
    course_id INTEGER REFERENCES courses(id),
    enrolled_at TIMESTAMP DEFAULT NOW(),
    grade VARCHAR(2),
    PRIMARY KEY (student_id, course_id)
);
```

### Self-Referencing Relationship

```sql
-- Employee manager hierarchy
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

-- Query: Find all employees under a manager
WITH RECURSIVE employee_tree AS (
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    SELECT e.id, e.name, e.manager_id, et.level + 1
    FROM employees e
    JOIN employee_tree et ON e.manager_id = et.id
)
SELECT * FROM employee_tree;
```

## Primary Keys

### Surrogate Keys (Recommended)

Auto-incrementing integers or UUIDs:

```sql
-- Auto-increment (PostgreSQL)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE
);

-- UUID (better for distributed systems)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE
);
```

### Natural Keys

Use existing data as key:

```sql
-- Email as natural key
CREATE TABLE users (
    email VARCHAR(255) PRIMARY KEY,
    name VARCHAR(100)
);

-- Composite natural key
CREATE TABLE flight_bookings (
    flight_number VARCHAR(10),
    seat_number VARCHAR(5),
    passenger_name VARCHAR(100),
    PRIMARY KEY (flight_number, seat_number)
);
```

**When to use**:
- Natural keys: When the value is truly unique and stable
- Surrogate keys: Most other cases (recommended default)

## Foreign Keys

### Basic Foreign Key

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id)
);
```

### Cascade Options

```sql
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id)
        ON DELETE CASCADE        -- Delete posts when user deleted
        ON UPDATE CASCADE        -- Update posts when user id changes
);

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id)
        ON DELETE SET NULL       -- Set to NULL when post deleted
);

CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id)
        ON DELETE RESTRICT       -- Prevent deletion if referenced
);
```

### Composite Foreign Keys

```sql
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER,
    item_number INTEGER,
    quantity INTEGER,
    FOREIGN KEY (order_id, item_number)
        REFERENCES inventory(warehouse_id, product_id)
);
```

## Indexes

### B-Tree Index (Default)

Best for equality and range queries:

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Range query benefits from index
SELECT * FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';
```

### Hash Index

Best for exact equality:

```sql
CREATE INDEX idx_users_email_hash ON users USING HASH (email);

-- Fast exact match
SELECT * FROM users WHERE email = 'user@example.com';
```

### Composite (Multi-column) Index

```sql
CREATE INDEX idx_users_name_email ON users(last_name, first_name);

-- Fast (uses index)
SELECT * FROM users WHERE last_name = 'Smith';
SELECT * FROM users WHERE last_name = 'Smith' AND first_name = 'John';

-- Slow (doesn't use index - missing leftmost column)
SELECT * FROM users WHERE first_name = 'John';
```

### Partial Index

Index only subset of rows:

```sql
CREATE INDEX idx_active_users ON users(email)
WHERE status = 'active';

-- Fast query on active users
SELECT * FROM users WHERE email = 'user@example.com' AND status = 'active';
```

### Covering Index

Include extra columns for index-only scans:

```sql
CREATE INDEX idx_users_email_covering ON users(email)
INCLUDE (name, created_at);

-- Can be answered entirely from index
SELECT name, created_at FROM users WHERE email = 'user@example.com';
```

### Full-Text Index

```sql
-- PostgreSQL
CREATE INDEX idx_posts_content_fts ON posts USING GIN(to_tsvector('english', content));

SELECT * FROM posts
WHERE to_tsvector('english', content) @@ to_tsquery('database & design');
```

### Spatial Index

For geographic and geometric data:

```sql
-- PostgreSQL with PostGIS
CREATE EXTENSION postgis;

CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    coordinates GEOGRAPHY(POINT, 4326)
);

-- Create spatial index (GiST - Generalized Search Tree)
CREATE INDEX idx_locations_coordinates ON locations USING GIST(coordinates);

-- Find locations within 1000 meters
SELECT name FROM locations
WHERE ST_DWithin(
    coordinates,
    ST_MakePoint(-122.4194, 37.7749)::geography,
    1000
);

-- MySQL spatial index
CREATE SPATIAL INDEX idx_location ON locations(coordinates);
```

**Use cases**: Maps, location-based services, geographic queries

## Data Types

### Choosing the Right Type

```sql
CREATE TABLE users (
    -- Integer types
    id BIGSERIAL,                    -- Auto-increment 64-bit integer
    age SMALLINT,                    -- 16-bit (-32768 to 32767)
    views INTEGER,                   -- 32-bit

    -- Strings
    username VARCHAR(50),            -- Variable, max 50 chars
    bio TEXT,                        -- Unlimited text
    country_code CHAR(2),           -- Fixed 2 chars (e.g., 'US')

    -- Decimal
    price DECIMAL(10,2),            -- Exact decimal (10 digits, 2 decimal places)
    rating NUMERIC(3,1),            -- Same as DECIMAL

    -- Floating point (avoid for money!)
    latitude FLOAT,
    longitude DOUBLE PRECISION,

    -- Date/Time
    created_at TIMESTAMP,           -- Date and time
    birth_date DATE,                -- Date only
    login_time TIME,                -- Time only
    updated_at TIMESTAMPTZ,         -- Timestamp with timezone

    -- Boolean
    is_active BOOLEAN,

    -- JSON
    preferences JSONB,              -- Binary JSON (faster, indexable)
    metadata JSON,                  -- Text JSON

    -- UUID
    session_id UUID,

    -- Array (PostgreSQL)
    tags TEXT[],

    -- Enum
    status user_status              -- Custom enum type
);

-- Create enum type
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'banned');
```

### Type Best Practices

```sql
-- DON'T: Use VARCHAR without limit
description VARCHAR                 -- Avoid

-- DO: Set reasonable limits
description VARCHAR(500)

-- DON'T: Use FLOAT/DOUBLE for money
price FLOAT                        -- WRONG! Precision issues

-- DO: Use DECIMAL/NUMERIC
price DECIMAL(10,2)               -- Correct

-- DON'T: Store dates as strings
date_field VARCHAR(10)            -- '2024-01-15'

-- DO: Use proper date types
date_field DATE                   -- Proper type, can use date functions
```

## Constraints

### NOT NULL

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL
);
```

### UNIQUE

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL
);

-- Composite unique constraint
CREATE TABLE user_roles (
    user_id INTEGER,
    role_id INTEGER,
    UNIQUE (user_id, role_id)
);
```

### CHECK

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) CHECK (price >= 0),
    stock INTEGER CHECK (stock >= 0),
    discount_percent INTEGER CHECK (discount_percent BETWEEN 0 AND 100)
);

-- Complex check constraint
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    email VARCHAR(255),
    CONSTRAINT valid_adult_email CHECK (
        (age >= 18 AND email IS NOT NULL) OR age < 18
    )
);
```

### DEFAULT

```sql
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    status VARCHAR(20) DEFAULT 'draft',
    views INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    is_published BOOLEAN DEFAULT false
);
```

## Query Optimization

### EXPLAIN Plans

Analyze query performance:

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- Look for:
-- - Seq Scan (bad for large tables) → add index
-- - Index Scan (good)
-- - Bitmap Heap Scan (okay)
-- - High cost numbers
-- - Many rows processed
```

### N+1 Query Problem

**Bad** (1 + N queries):
```python
# 1 query for posts
posts = db.query("SELECT * FROM posts LIMIT 10")

# N queries for authors (10 more queries!)
for post in posts:
    author = db.query("SELECT * FROM users WHERE id = ?", post.user_id)
    print(f"{post.title} by {author.name}")
```

**Good** (1 query with JOIN):
```python
posts = db.query("""
    SELECT p.*, u.name as author_name
    FROM posts p
    JOIN users u ON p.user_id = u.id
    LIMIT 10
""")

for post in posts:
    print(f"{post.title} by {post.author_name}")
```

**Good** (2 queries with eager loading):
```python
# Fetch all posts
posts = db.query("SELECT * FROM posts LIMIT 10")
user_ids = [p.user_id for p in posts]

# Fetch all users in one query
users = db.query("SELECT * FROM users WHERE id IN (?)", user_ids)
user_map = {u.id: u for u in users}

for post in posts:
    print(f"{post.title} by {user_map[post.user_id].name}")
```

### Query Caching

#### Application-Level Caching

```python
import redis

cache = redis.Redis(host='localhost', port=6379)

def get_user(user_id):
    # Try cache first
    cache_key = f"user:{user_id}"
    cached = cache.get(cache_key)

    if cached:
        return json.loads(cached)

    # Cache miss - query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)

    # Store in cache (expire after 1 hour)
    cache.setex(cache_key, 3600, json.dumps(user))

    return user
```

#### Database Query Cache

```sql
-- MySQL query cache (deprecated in MySQL 8.0)
SET query_cache_type = ON;
SET query_cache_size = 1048576;  -- 1MB

-- PostgreSQL shared_buffers (acts as cache)
-- In postgresql.conf:
-- shared_buffers = 256MB

-- Materialized views (pre-computed results)
CREATE MATERIALIZED VIEW user_order_stats AS
SELECT
    u.id,
    u.name,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- Refresh periodically
REFRESH MATERIALIZED VIEW user_order_stats;

-- Query is now instant
SELECT * FROM user_order_stats WHERE id = 123;
```

### Prepared Statements

Prevent SQL injection and improve performance:

```python
# Bad - SQL injection risk, no caching
user_input = "admin' OR '1'='1"
query = f"SELECT * FROM users WHERE username = '{user_input}'"
db.execute(query)  # VULNERABLE!

# Good - parameterized query
import psycopg2

conn = psycopg2.connect(...)
cursor = conn.cursor()

# Prepared statement (query plan cached)
cursor.execute(
    "SELECT * FROM users WHERE username = %s",
    (user_input,)  # Safe - treated as data, not SQL
)

# Explicitly prepare (useful for repeated queries)
cursor.execute("PREPARE user_query AS SELECT * FROM users WHERE id = $1")
cursor.execute("EXECUTE user_query(123)")
cursor.execute("EXECUTE user_query(456)")
cursor.execute("EXECUTE user_query(789)")
```

**Benefits**:
- **Security**: Prevents SQL injection
- **Performance**: Query plan cached and reused
- **Type safety**: Database validates parameters

### Connection Pooling

```python
from psycopg2 import pool

# Create connection pool
db_pool = pool.ThreadedConnectionPool(
    minconn=5,     # Keep 5 connections open
    maxconn=20,    # Max 20 concurrent connections
    host='localhost',
    database='myapp'
)

# Get connection from pool
conn = db_pool.getconn()
try:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
finally:
    # Return connection to pool (don't close!)
    db_pool.putconn(conn)
```

**Benefits**:
- Avoid connection setup overhead
- Limit database load
- Better resource utilization
- Handle connection spikes

## Common Design Patterns

### Soft Delete

Keep deleted records for audit/recovery:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    deleted_at TIMESTAMP NULL
);

-- "Delete" user
UPDATE users SET deleted_at = NOW() WHERE id = 123;

-- Query active users
SELECT * FROM users WHERE deleted_at IS NULL;

-- Index for performance
CREATE INDEX idx_users_active ON users(id) WHERE deleted_at IS NULL;
```

### Audit Trail / History Tracking

Track all changes:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE users_audit (
    audit_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(10),           -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_by INTEGER,
    changed_at TIMESTAMP DEFAULT NOW()
);

-- Trigger to log changes
CREATE OR REPLACE FUNCTION audit_users()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        INSERT INTO users_audit(user_id, action, old_data)
        VALUES (OLD.id, 'DELETE', row_to_json(OLD));
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO users_audit(user_id, action, old_data, new_data)
        VALUES (NEW.id, 'UPDATE', row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO users_audit(user_id, action, new_data)
        VALUES (NEW.id, 'INSERT', row_to_json(NEW));
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION audit_users();
```

### Optimistic Locking

Prevent lost updates:

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2),
    version INTEGER DEFAULT 1    -- Version number
);

-- Update with version check
UPDATE products
SET price = 29.99, version = version + 1
WHERE id = 123 AND version = 5;

-- If 0 rows affected, concurrent update occurred
```

### Polymorphic Associations

One table references multiple tables:

```sql
-- Option 1: Separate foreign keys (recommended)
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    content TEXT,
    post_id INTEGER REFERENCES posts(id),
    photo_id INTEGER REFERENCES photos(id),
    CHECK (
        (post_id IS NOT NULL AND photo_id IS NULL) OR
        (post_id IS NULL AND photo_id IS NOT NULL)
    )
);

-- Option 2: Type field (less type-safe)
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    content TEXT,
    commentable_type VARCHAR(50),  -- 'Post' or 'Photo'
    commentable_id INTEGER
);
```

### Tag System

```sql
-- Simple tagging
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT
);

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE
);

CREATE TABLE post_tags (
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (post_id, tag_id)
);

-- Find posts with specific tag
SELECT p.* FROM posts p
JOIN post_tags pt ON p.id = pt.post_id
JOIN tags t ON pt.tag_id = t.id
WHERE t.name = 'database';

-- Find posts with multiple tags
SELECT p.* FROM posts p
JOIN post_tags pt ON p.id = pt.post_id
JOIN tags t ON pt.tag_id = t.id
WHERE t.name IN ('database', 'design')
GROUP BY p.id
HAVING COUNT(DISTINCT t.id) = 2;  -- Must have both tags
```

### Tree Structures

#### Adjacency List (Simple)

```sql
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    parent_id INTEGER REFERENCES categories(id)
);

-- Query with recursive CTE
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 1 as depth
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    SELECT c.id, c.name, c.parent_id, ct.depth + 1
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;
```

#### Nested Set Model (Fast reads)

```sql
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    lft INTEGER NOT NULL,
    rgt INTEGER NOT NULL
);

-- Example data:
-- Electronics (1, 10)
--   ├─ Computers (2, 5)
--   │   └─ Laptops (3, 4)
--   └─ Phones (6, 9)
--       └─ Smartphones (7, 8)

-- Get all descendants (very fast)
SELECT * FROM categories
WHERE lft > 2 AND rgt < 5;  -- All under Computers
```

### Temporal Data (Time-Based Versioning)

Track how data changes over time:

#### Valid Time (Business Time)

Track when data is valid in the real world:

```sql
CREATE TABLE product_prices (
    product_id INTEGER,
    price DECIMAL(10,2),
    valid_from DATE,
    valid_to DATE,
    PRIMARY KEY (product_id, valid_from)
);

-- Insert price changes
INSERT INTO product_prices VALUES
    (1, 19.99, '2024-01-01', '2024-06-30'),
    (1, 24.99, '2024-07-01', '2024-12-31'),
    (1, 29.99, '2025-01-01', '9999-12-31');  -- Current price

-- Get price on specific date
SELECT price FROM product_prices
WHERE product_id = 1
  AND '2024-08-15' BETWEEN valid_from AND valid_to;

-- Get current price
SELECT price FROM product_prices
WHERE product_id = 1
  AND CURRENT_DATE BETWEEN valid_from AND valid_to;
```

#### Transaction Time (System Time)

Track when data was stored in the database:

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    salary DECIMAL(10,2),
    transaction_start TIMESTAMP DEFAULT NOW(),
    transaction_end TIMESTAMP DEFAULT '9999-12-31 23:59:59'
);

-- Update creates new version, marks old version as ended
UPDATE employees
SET transaction_end = NOW()
WHERE id = 1 AND transaction_end = '9999-12-31 23:59:59';

INSERT INTO employees (id, name, salary, transaction_start)
VALUES (1, 'Alice', 75000, NOW());

-- Get current version
SELECT * FROM employees
WHERE id = 1 AND transaction_end = '9999-12-31 23:59:59';

-- Get history (all versions)
SELECT * FROM employees WHERE id = 1 ORDER BY transaction_start;

-- Point-in-time query (what was the salary on Jan 1?)
SELECT salary FROM employees
WHERE id = 1
  AND transaction_start <= '2024-01-01'
  AND transaction_end > '2024-01-01';
```

#### Bi-Temporal (Both Valid and Transaction Time)

```sql
CREATE TABLE insurance_policies (
    policy_id INTEGER,
    customer_id INTEGER,
    coverage DECIMAL(10,2),
    -- Business time (when policy is valid)
    valid_from DATE,
    valid_to DATE,
    -- System time (when we knew about it)
    transaction_start TIMESTAMP DEFAULT NOW(),
    transaction_end TIMESTAMP DEFAULT '9999-12-31 23:59:59',
    PRIMARY KEY (policy_id, valid_from, transaction_start)
);

-- Query: What coverage did we think customer had on 2024-06-01,
--        as of what we knew on 2024-03-15?
SELECT coverage FROM insurance_policies
WHERE customer_id = 123
  AND '2024-06-01' BETWEEN valid_from AND valid_to
  AND '2024-03-15' BETWEEN transaction_start AND transaction_end;
```

#### PostgreSQL Temporal Tables

```sql
-- PostgreSQL 12+ supports period ranges
CREATE EXTENSION btree_gist;

CREATE TABLE room_bookings (
    room_id INTEGER,
    guest_name VARCHAR(100),
    during TSRANGE,  -- Time range type
    EXCLUDE USING GIST (room_id WITH =, during WITH &&)  -- Prevent overlaps
);

-- Insert bookings
INSERT INTO room_bookings VALUES
    (101, 'Alice', '[2024-01-01 14:00, 2024-01-03 10:00)');

-- This will fail - overlapping booking!
INSERT INTO room_bookings VALUES
    (101, 'Bob', '[2024-01-02 14:00, 2024-01-04 10:00)');

-- This succeeds - no overlap
INSERT INTO room_bookings VALUES
    (101, 'Bob', '[2024-01-03 14:00, 2024-01-05 10:00)');
```

## ACID vs BASE

### ACID (Traditional SQL Databases)

**Atomicity**: All operations in a transaction succeed or all fail:
```sql
BEGIN TRANSACTION;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;  -- Both succeed or both rollback
```

**Consistency**: Database moves from one valid state to another:
```sql
-- Constraint ensures consistency
ALTER TABLE accounts ADD CONSTRAINT positive_balance
    CHECK (balance >= 0);

-- This transaction will fail - maintains consistency
UPDATE accounts SET balance = balance - 1000 WHERE balance = 100;
```

**Isolation**: Concurrent transactions don't interfere:
```sql
-- Transaction 1
BEGIN TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;  -- Reads 100
-- Transaction 2 updates balance to 200 here
SELECT balance FROM accounts WHERE id = 1;  -- Still reads 100 (isolation)
COMMIT;
```

**Durability**: Committed transactions persist:
```sql
COMMIT;  -- Once committed, data survives crashes
```

**Isolation Levels**:
```sql
-- Read Uncommitted: Can see uncommitted changes (dirty reads)
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- Read Committed: Only see committed changes (default in PostgreSQL)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Repeatable Read: Same query returns same result in transaction
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Serializable: Full isolation, transactions appear serial
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### BASE (NoSQL/Distributed Systems)

**Basically Available**: System appears to work most of the time
**Soft state**: State may change without input (replication lag)
**Eventual consistency**: System becomes consistent over time

```python
# Example: Eventually consistent cache
def update_user(user_id, name):
    # Write to primary database
    primary_db.update(user_id, name)

    # Asynchronously update cache (eventual consistency)
    async_update_cache(user_id, name)

    # Immediately after, cache might be stale
    # But will eventually be consistent

def get_user(user_id):
    # Might return old value briefly
    return cache.get(user_id)
```

**When to use**:
- **ACID**: Financial transactions, inventory, critical data
- **BASE**: Social media feeds, recommendations, analytics

## Partitioning and Sharding

### Partitioning (Within Single Database)

Split large table into smaller pieces:

#### Range Partitioning

```sql
-- Partition by date range
CREATE TABLE orders (
    id SERIAL,
    user_id INTEGER,
    total DECIMAL(10,2),
    created_at DATE
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2023 PARTITION OF orders
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE orders_2024 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE orders_2025 PARTITION OF orders
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Queries automatically use correct partition
SELECT * FROM orders WHERE created_at = '2024-06-15';  -- Only scans orders_2024
```

#### List Partitioning

```sql
CREATE TABLE sales (
    id SERIAL,
    region VARCHAR(50),
    amount DECIMAL(10,2)
) PARTITION BY LIST (region);

CREATE TABLE sales_north PARTITION OF sales
    FOR VALUES IN ('US-NORTH', 'CA-NORTH');

CREATE TABLE sales_south PARTITION OF sales
    FOR VALUES IN ('US-SOUTH', 'MX');

CREATE TABLE sales_europe PARTITION OF sales
    FOR VALUES IN ('UK', 'DE', 'FR');
```

#### Hash Partitioning

```sql
CREATE TABLE users (
    id SERIAL,
    email VARCHAR(255)
) PARTITION BY HASH (id);

CREATE TABLE users_0 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE users_1 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE users_2 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE users_3 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### Sharding (Across Multiple Databases)

Distribute data across separate database instances:

```python
# Simple hash-based sharding
class ShardedDatabase:
    def __init__(self, shard_count=4):
        self.shards = [
            connect_to_db(f'shard_{i}')
            for i in range(shard_count)
        ]

    def get_shard(self, user_id):
        shard_num = hash(user_id) % len(self.shards)
        return self.shards[shard_num]

    def get_user(self, user_id):
        shard = self.get_shard(user_id)
        return shard.query("SELECT * FROM users WHERE id = ?", user_id)

    def create_user(self, user_id, data):
        shard = self.get_shard(user_id)
        return shard.execute("INSERT INTO users ...", data)

# Usage
db = ShardedDatabase(shard_count=4)
user = db.get_user(12345)  # Routes to correct shard
```

**Sharding Strategies**:

1. **Range-based**: Users 0-999 in Shard 1, 1000-1999 in Shard 2
2. **Hash-based**: Hash user ID, mod by shard count
3. **Geographic**: US users in US shard, EU users in EU shard
4. **Directory-based**: Lookup table maps keys to shards

**Challenges**:
- Cross-shard queries expensive
- Rebalancing when adding shards
- Hotspots if data not distributed evenly
- Transaction complexity across shards

## Schema Versioning

### Migration Pattern

```sql
-- migrations/001_create_users.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- migrations/002_add_users_name.sql
ALTER TABLE users ADD COLUMN name VARCHAR(100);

-- migrations/003_add_users_status.sql
ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';
CREATE INDEX idx_users_status ON users(status);

-- Track migrations
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW()
);
```

### Zero-Downtime Migrations

Techniques for migrating without service interruption:

#### 1. Additive Changes (Safe)

```sql
-- ✓ Add new column with default (no downtime)
ALTER TABLE users ADD COLUMN phone VARCHAR(20) DEFAULT NULL;

-- ✓ Add new index (concurrent, no locks in PostgreSQL)
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);

-- ✓ Add new table
CREATE TABLE user_preferences (
    user_id INTEGER REFERENCES users(id),
    theme VARCHAR(20)
);
```

#### 2. Multi-Step Column Rename

**Don't do this** (breaks running code):
```sql
ALTER TABLE users RENAME COLUMN name TO full_name;  -- ❌ Instant breakage!
```

**Do this** (multi-step, zero downtime):

```sql
-- Step 1: Add new column
ALTER TABLE users ADD COLUMN full_name VARCHAR(100);

-- Step 2: Backfill data (in batches for large tables)
UPDATE users SET full_name = name WHERE full_name IS NULL;

-- Step 3: Deploy code that writes to both columns
-- Application now writes to both 'name' and 'full_name'

-- Step 4: Deploy code that reads from new column
-- Application now reads from 'full_name', still writes to both

-- Step 5: Drop old column (only after all code updated)
ALTER TABLE users DROP COLUMN name;
```

#### 3. Multi-Step Column Type Change

```sql
-- Don't: ALTER TABLE users ALTER COLUMN id TYPE BIGINT;  -- ❌ Locks table!

-- Step 1: Add new column
ALTER TABLE users ADD COLUMN id_new BIGINT;

-- Step 2: Backfill (in batches)
UPDATE users SET id_new = id::BIGINT WHERE id_new IS NULL;

-- Step 3: Add unique constraint and index
ALTER TABLE users ADD CONSTRAINT users_id_new_unique UNIQUE (id_new);
CREATE INDEX CONCURRENTLY idx_users_id_new ON users(id_new);

-- Step 4: Update application to use new column

-- Step 5: Swap columns (if needed) or drop old column
ALTER TABLE users DROP COLUMN id;
ALTER TABLE users RENAME COLUMN id_new TO id;
```

#### 4. Removing NOT NULL Constraint

```sql
-- Step 1: Remove constraint
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- Step 2: Deploy code that handles NULL values

-- Step 3: Optionally re-add constraint if needed
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
```

#### 5. Batch Processing for Large Tables

```python
# Don't: UPDATE users SET status = 'active';  -- Locks entire table!

# Do: Update in batches
def backfill_status_in_batches():
    batch_size = 1000
    last_id = 0

    while True:
        # Update batch
        result = db.execute("""
            UPDATE users
            SET status = 'active'
            WHERE id > %s
              AND id <= %s + %s
              AND status IS NULL
        """, (last_id, last_id, batch_size))

        if result.rowcount == 0:
            break  # No more rows to update

        last_id += batch_size
        time.sleep(0.1)  # Brief pause to avoid overwhelming DB
```

### Rollback Strategies

Plan for migration failures:

#### 1. Reversible Migrations

Every migration should have a rollback:

```sql
-- migrations/005_add_user_status.sql (UP)
ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT 'active';
CREATE INDEX idx_users_status ON users(status);

-- migrations/005_add_user_status_rollback.sql (DOWN)
DROP INDEX IF EXISTS idx_users_status;
ALTER TABLE users DROP COLUMN status;
```

#### 2. Migration with Rollback Tracking

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR(255),
    applied_at TIMESTAMP DEFAULT NOW(),
    rolled_back_at TIMESTAMP NULL
);

-- Apply migration
INSERT INTO schema_migrations (version, name)
VALUES (5, 'add_user_status');

-- Rollback
UPDATE schema_migrations
SET rolled_back_at = NOW()
WHERE version = 5;
```

#### 3. Testing Migrations

```bash
# Test on staging/development first
$ psql staging_db < migrations/005_add_user_status.sql

# Verify migration
$ psql staging_db -c "SELECT * FROM users LIMIT 1;"

# Test rollback
$ psql staging_db < migrations/005_add_user_status_rollback.sql

# Verify rollback worked
$ psql staging_db -c "\d users"
```

#### 4. Gradual Rollout

```python
# Use feature flags to gradually enable new schema
def get_user_status(user_id):
    if feature_flag_enabled('use_new_status_column'):
        return db.query("SELECT status FROM users WHERE id = ?", user_id)
    else:
        # Fallback to old logic
        return calculate_status_from_other_fields(user_id)

# Enable for 5% of users first
# If stable, increase to 50%, then 100%
```

#### 5. Backup Before Migration

```bash
# Always backup before major migrations
$ pg_dump -h localhost -U postgres mydb > backup_before_migration.sql

# Run migration
$ psql mydb < migrations/006_major_change.sql

# If something goes wrong, restore
$ psql mydb < backup_before_migration.sql
```

## Design Anti-Patterns to Avoid

### 1. God Table

**Bad**: One massive table with everything:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    -- 100+ columns here...
    last_login TIMESTAMP,
    preferences TEXT,
    -- Don't do this!
);
```

**Good**: Split into related tables:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255)
);

CREATE TABLE user_profiles (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    bio TEXT,
    avatar_url VARCHAR(255)
);

CREATE TABLE user_preferences (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    theme VARCHAR(20),
    notifications BOOLEAN
);
```

### 2. EAV (Entity-Attribute-Value) Anti-pattern

**Bad**: Flexible but slow and complex:
```sql
CREATE TABLE eav_data (
    entity_id INTEGER,
    attribute_name VARCHAR(100),
    attribute_value TEXT
);

-- Query becomes nightmare
SELECT * FROM eav_data WHERE entity_id = 1;
```

**Good**: Use proper columns or JSONB:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    properties JSONB  -- Use JSONB for truly dynamic data
);

-- Much better query
SELECT properties->>'theme' as theme FROM users WHERE id = 1;
```

### 3. Premature Optimization

**Bad**: Over-engineering before knowing requirements:
```sql
-- Don't create 50 indexes upfront
CREATE INDEX idx_1 ON users(email);
CREATE INDEX idx_2 ON users(name);
CREATE INDEX idx_3 ON users(created_at);
-- ... 47 more indexes
```

**Good**: Start simple, measure, then optimize:
```sql
-- Start with essential indexes
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- Add more indexes based on actual query patterns
```

## Best Practices

1. **Use meaningful names**: `user_id` not `uid`, `created_at` not `crtd`
2. **Be consistent**: Stick to naming conventions (snake_case vs camelCase)
3. **Add timestamps**: Every table should have `created_at`, often `updated_at`
4. **Use UUIDs for distributed systems**: Better than auto-increment IDs
5. **Index foreign keys**: Almost always need indexes on FK columns
6. **Document your schema**: Add comments to tables and columns
7. **Plan for growth**: Consider partitioning for large tables
8. **Use transactions**: Maintain data integrity for multi-step operations
9. **Regular backups**: Automate database backups
10. **Monitor performance**: Track slow queries and optimize

## Tools

- **Schema Design**: dbdiagram.io, draw.io, Lucidchart
- **Migrations**: Flyway, Liquibase, Alembic (Python), Migrate (Go)
- **ORMs**: SQLAlchemy, Django ORM, Hibernate, Entity Framework
- **Database Clients**: pgAdmin, DBeaver, TablePlus, DataGrip

## Further Reading

- [Database Normalization](https://en.wikipedia.org/wiki/Database_normalization)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQL Style Guide](https://www.sqlstyle.guide/)
- [Database Design Patterns](https://www.martinfowler.com/eaaCatalog/)
