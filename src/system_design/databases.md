# Databases

## Overview

Databases are the foundation of most distributed systems, providing persistent storage and data management at scale.

## SQL vs NoSQL

### SQL (Relational Databases)

Structured data with predefined schemas:

```
┌─────────────────────────┐
│ Users Table             │
├─────┬──────────┬────────┤
│ ID  │ Name     │ Email  │
├─────┼──────────┼────────┤
│ 1   │ Alice    │ a@...  │
│ 2   │ Bob      │ b@...  │
└─────┴──────────┴────────┘
```

**Popular**: PostgreSQL, MySQL, Oracle

**Pros**:
- ACID guarantees
- Strong consistency
- Complex queries (JOINs)
- Mature tooling

**Cons**:
- Rigid schema
- Vertical scaling
- Complex sharding

### NoSQL (Non-Relational)

Flexible schemas for different use cases:

**Document Stores** (MongoDB, CouchDB):
```json
{
  "id": "123",
  "name": "Alice",
  "email": "alice@example.com",
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```

**Key-Value Stores** (Redis, DynamoDB):
```
user:123 → {"name": "Alice", "email": "..."}
session:abc → {"userId": 123, "expires": ...}
```

**Column-Family** (Cassandra, HBase):
```
Row Key: user123
├─ profile:name = "Alice"
├─ profile:email = "alice@..."
├─ activity:last_login = "2025-01-15"
└─ activity:login_count = 42
```

**Graph Databases** (Neo4j, ArangoDB):
```
(Alice)-[:FOLLOWS]->(Bob)
(Alice)-[:LIKES]->(Post1)
(Bob)-[:CREATED]->(Post1)
```

**Pros**:
- Flexible schema
- Horizontal scaling
- High performance
- Specific use case optimization

**Cons**:
- Eventual consistency
- Limited transactions
- Complex queries harder

## Database Comparison

| Feature | SQL | NoSQL |
|---------|-----|-------|
| **Schema** | Fixed | Flexible |
| **Scaling** | Vertical | Horizontal |
| **Consistency** | Strong | Eventual |
| **Transactions** | ACID | BASE |
| **Queries** | Complex JOINs | Simple lookups |
| **Use Case** | Financial, ERP | Social, IoT, Logs |

## ACID vs BASE

### ACID (SQL)

**Atomicity**: All or nothing
```sql
BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- Both succeed or both fail
```

**Consistency**: Valid state always
**Isolation**: Concurrent transactions don't interfere
**Durability**: Committed data persists

### BASE (NoSQL)

**Basically Available**: System works most of the time
**Soft state**: State may change without input
**Eventual consistency**: Data becomes consistent eventually

```
Time 0: Write to Node A → value = 100
Time 1: Node B still has → value = 50 (stale)
Time 2: Replication complete → value = 100 (consistent)
```

## Database Replication

### Master-Slave (Primary-Replica)

```
         Master (Write)
         /      |      \
        /       |       \
    Slave1   Slave2   Slave3
    (Read)   (Read)   (Read)
```

**Pattern**:
```python
# Write to master
master_db.execute("INSERT INTO users VALUES (...)")

# Read from replica
replica_db.execute("SELECT * FROM users WHERE id = 1")
```

**Pros**: Read scalability
**Cons**: Write bottleneck, replication lag

### Master-Master (Multi-Master)

```
    Master1 ←→ Master2
      ↕            ↕
   Writes       Writes
```

Both accept writes and sync:

```python
# Can write to either
db1.write("user:123", data)
db2.write("user:456", data)
# Sync between masters
```

**Pros**: Write scalability, high availability
**Cons**: Conflict resolution, complexity

### Replication Strategies

**Synchronous**: Wait for all replicas
```
Write → Master → Wait for Slaves → Ack
(Slow but consistent)
```

**Asynchronous**: Don't wait for replicas
```
Write → Master → Ack (fast)
             ↓
        Replicate later
```

**Semi-Synchronous**: Wait for at least one
```
Write → Master → Wait for 1 Slave → Ack
             ↓
        Others async
```

## Database Sharding

Partition data across multiple databases:

### Horizontal Sharding (Row-based)

```
Shard 1: Users 0-999
  ├─ user:0
  ├─ user:500
  └─ user:999

Shard 2: Users 1000-1999
  ├─ user:1000
  └─ user:1999

Shard 3: Users 2000-2999
  ├─ user:2000
  └─ user:2999
```

**Sharding Key**: User ID

```python
def get_shard(user_id):
    shard_num = user_id // 1000
    return shards[shard_num]

# Route to correct shard
shard = get_shard(user_id=1500)  # → Shard 2
user = shard.query("SELECT * FROM users WHERE id = 1500")
```

### Hash-Based Sharding

```python
def get_shard(key):
    hash_value = hash(key)
    shard_num = hash_value % num_shards
    return shards[shard_num]

# Example
get_shard("user_alice")  # → hash → 42 → shard 2
get_shard("user_bob")    # → hash → 17 → shard 1
```

**Pros**: Even distribution
**Cons**: Hard to add shards (rehashing)

### Range-Based Sharding

```
Shard 1: A-H (Alice, Bob, Charlie...)
Shard 2: I-P (Ian, John, Kate...)
Shard 3: Q-Z (Quinn, Rachel, Steve...)
```

**Pros**: Easy range queries
**Cons**: Uneven distribution (hotspots)

### Geographic Sharding

```
Shard US-East: Users in US East
Shard US-West: Users in US West
Shard EU: Users in Europe
Shard ASIA: Users in Asia
```

**Pros**: Low latency, data compliance
**Cons**: Cross-region queries expensive

### Vertical Sharding

Split by tables/columns:

```
Shard 1: User profiles
  ├─ users table
  └─ profiles table

Shard 2: User activity
  ├─ posts table
  ├─ comments table
  └─ likes table
```

## Database Partitioning

### List Partitioning

```sql
CREATE TABLE orders (
    id INT,
    region VARCHAR(50)
) PARTITION BY LIST (region) (
    PARTITION p_north VALUES IN ('NY', 'MA', 'CT'),
    PARTITION p_south VALUES IN ('TX', 'FL', 'GA'),
    PARTITION p_west VALUES IN ('CA', 'OR', 'WA')
);
```

### Range Partitioning

```sql
CREATE TABLE sales (
    id INT,
    sale_date DATE
) PARTITION BY RANGE (YEAR(sale_date)) (
    PARTITION p_2023 VALUES LESS THAN (2024),
    PARTITION p_2024 VALUES LESS THAN (2025),
    PARTITION p_2025 VALUES LESS THAN (2026)
);
```

### Hash Partitioning

```sql
CREATE TABLE users (
    id INT,
    name VARCHAR(100)
) PARTITION BY HASH(id)
PARTITIONS 4;
```

## Indexing Strategies

### B-Tree Index (Default)

```
        [50]
       /    \
   [25]      [75]
   /  \      /  \
[10][40] [60][90]
```

**Use**: Range queries, sorting
```sql
CREATE INDEX idx_name ON users(name);
SELECT * FROM users WHERE name BETWEEN 'A' AND 'M';
```

### Hash Index

```
hash(key) → bucket
user:123 → bucket 5
user:456 → bucket 2
```

**Use**: Exact matches
```sql
CREATE INDEX idx_email USING HASH ON users(email);
SELECT * FROM users WHERE email = 'alice@example.com';
```

### Composite Index

```sql
CREATE INDEX idx_name_age ON users(name, age);

-- Fast
SELECT * FROM users WHERE name = 'Alice' AND age = 30;

-- Fast (leftmost prefix)
SELECT * FROM users WHERE name = 'Alice';

-- Slow (missing leftmost)
SELECT * FROM users WHERE age = 30;
```

### Full-Text Index

```sql
CREATE FULLTEXT INDEX idx_content ON posts(content);

SELECT * FROM posts
WHERE MATCH(content) AGAINST('database sharding');
```

## Query Optimization

### Use EXPLAIN

```sql
EXPLAIN SELECT * FROM users
WHERE email = 'alice@example.com';

-- Output shows:
-- type: index (good)
-- rows: 1 (good)
-- type: ALL (bad - full scan)
```

### Avoid N+1 Queries

**Bad**:
```python
# 1 query for posts
posts = db.query("SELECT * FROM posts")
# N queries for users
for post in posts:
    user = db.query("SELECT * FROM users WHERE id = ?", post.user_id)
```

**Good**:
```python
# 1 query with JOIN
posts = db.query("""
    SELECT posts.*, users.name
    FROM posts
    JOIN users ON posts.user_id = users.id
""")
```

### Pagination

**Bad** (large offset):
```sql
SELECT * FROM posts
ORDER BY created_at DESC
LIMIT 1000 OFFSET 100000;  -- Slow!
```

**Good** (cursor-based):
```sql
SELECT * FROM posts
WHERE created_at < '2025-01-01 12:00:00'
ORDER BY created_at DESC
LIMIT 1000;
```

## Connection Pooling

Reuse database connections:

```python
from psycopg2 import pool

# Create pool
db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=20,
    host='localhost',
    database='myapp'
)

# Get connection from pool
conn = db_pool.getconn()
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()

# Return to pool
db_pool.putconn(conn)
```

**Benefits**:
- Avoid connection overhead
- Limit concurrent connections
- Better resource utilization

## Database Design Patterns

### Write-Ahead Log (WAL)

Log changes before applying:
```
1. Write change to log (durable)
2. Apply change to database
3. Mark log entry as complete
```

**Recovery**: Replay log after crash

### Materialized Views

Pre-computed query results:

```sql
CREATE MATERIALIZED VIEW user_stats AS
SELECT
    user_id,
    COUNT(*) as post_count,
    MAX(created_at) as last_post
FROM posts
GROUP BY user_id;

-- Refresh periodically
REFRESH MATERIALIZED VIEW user_stats;
```

### Database per Service

Microservices pattern:

```
Service A → Database A
Service B → Database B
Service C → Database C
```

**Pros**: Service independence
**Cons**: Complex transactions, data duplication

## Common Operations

### Bulk Insert

```sql
-- Slow
INSERT INTO users VALUES (1, 'Alice');
INSERT INTO users VALUES (2, 'Bob');
INSERT INTO users VALUES (3, 'Charlie');

-- Fast
INSERT INTO users VALUES
    (1, 'Alice'),
    (2, 'Bob'),
    (3, 'Charlie');
```

### Soft Delete

Keep deleted records:

```sql
ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP;

-- "Delete"
UPDATE users SET deleted_at = NOW() WHERE id = 123;

-- Query active users
SELECT * FROM users WHERE deleted_at IS NULL;
```

### Audit Trail

Track all changes:

```sql
CREATE TABLE audit_log (
    id INT PRIMARY KEY,
    table_name VARCHAR(50),
    record_id INT,
    action VARCHAR(10),
    old_value JSON,
    new_value JSON,
    changed_by INT,
    changed_at TIMESTAMP
);

-- Trigger on update
CREATE TRIGGER audit_users
AFTER UPDATE ON users
FOR EACH ROW
    INSERT INTO audit_log VALUES (...);
```

## Choosing a Database

| Use Case | Database Type | Examples |
|----------|---------------|----------|
| **Transactions** | SQL | PostgreSQL, MySQL |
| **High writes** | NoSQL (Key-Value) | Redis, DynamoDB |
| **Documents** | NoSQL (Document) | MongoDB, CouchDB |
| **Time series** | NoSQL (Column) | Cassandra, InfluxDB |
| **Relationships** | Graph | Neo4j, ArangoDB |
| **Full-text search** | Search engine | Elasticsearch, Solr |
| **Caching** | In-memory | Redis, Memcached |

## ELI10

Databases are like different types of filing systems:

- **SQL**: Like a library with strict card catalog - everything has a place, find books easily
- **Document DB**: Like folders with papers - flexible, can add any notes
- **Key-Value**: Like a locker - give key number, get contents fast
- **Graph**: Like a friendship map - see how people connect

**Replication**: Making copies of books in multiple libraries (backup, faster access)
**Sharding**: Splitting books across libraries (A-M in one, N-Z in another)

Choose the right tool for the job!

## Further Resources

- [Database Internals](https://www.databass.dev/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB University](https://university.mongodb.com/)
- [Database Design Patterns](https://en.wikipedia.org/wiki/Database_design)
