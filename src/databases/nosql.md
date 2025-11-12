# NoSQL Databases

## Overview

NoSQL databases store data in non-relational formats (documents, key-value, graph, etc.). Designed for scalability, flexibility, and high-performance.

## Types

### Document Databases (MongoDB)
```javascript
// Insert
db.users.insertOne({ name: "John", age: 30, email: "john@example.com" });

// Find
db.users.findOne({ name: "John" });
db.users.find({ age: { $gt: 25 } });

// Update
db.users.updateOne({ _id: ObjectId("...") }, { $set: { age: 31 } });

// Delete
db.users.deleteOne({ name: "John" });

// Aggregation
db.users.aggregate([
  { $match: { age: { $gt: 25 } } },
  { $group: { _id: "$city", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]);
```

### Key-Value Stores (Redis)
```bash
# Strings
SET key value
GET key
INCR counter

# Lists
LPUSH mylist "a" "b" "c"
LPOP mylist
LRANGE mylist 0 -1

# Sets
SADD myset "a" "b" "c"
SMEMBERS myset

# Hashes
HSET user:1 name "John" age 30
HGET user:1 name
HGETALL user:1

# Expiration
EXPIRE key 3600  # 1 hour TTL
```

### Column-Family (Cassandra)
```sql
-- Wide, denormalized columns
CREATE TABLE users (
  user_id UUID PRIMARY KEY,
  name TEXT,
  email TEXT,
  created_at TIMESTAMP,
  metadata MAP<TEXT, TEXT>
);
```

### Graph Databases (Neo4j)
```cypher
// Create
CREATE (n:Person {name: "John", age: 30})
CREATE (m:Company {name: "Acme"})
CREATE (n)-[:WORKS_AT]->(m)

// Query
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WHERE p.age > 25
RETURN p.name, c.name

// Find friends
MATCH (p:Person {name: "John"})-[:FRIEND*1..2]-(friend)
RETURN friend.name
```

## CAP Theorem

Every distributed database trades off:
- **Consistency**: All nodes see same data
- **Availability**: System always responsive
- **Partition Tolerance**: Survive network splits

**You can have 2 of 3:**
- **CP**: Strong consistency, unavailable during partitions (Spanner)
- **AP**: Always available, eventual consistency (Dynamo, Cassandra)
- **CA**: Consistent and available, can't handle partitions (traditional DB)

## Use Cases

| Database | Best For |
|----------|----------|
| **MongoDB** | Flexible schema, documents |
| **Redis** | Caching, sessions, real-time |
| **Cassandra** | Time-series, massive scale |
| **Neo4j** | Graph queries, relationships |
| **Elasticsearch** | Full-text search, logs |

## Data Modeling

```python
# Denormalization (NoSQL style)
# One document with all info
{
  "_id": "user_1",
  "name": "John",
  "orders": [
    { "id": "order_1", "amount": 100 },
    { "id": "order_2", "amount": 200 }
  ]
}

# vs SQL (normalization)
# users table + orders table + JOIN
```

## ELI10

NoSQL is like a flexible filing system:
- **Document DB**: Store complete documents (like PDF files)
- **Key-Value**: Simple lookup (like phone book)
- **Graph**: Show relationships (like social network)
- **Column**: Organize by columns not rows (like spreadsheet)

Trade flexibility and speed for less strict structure!

## Further Resources

- [MongoDB Docs](https://docs.mongodb.com/)
- [Redis Commands](https://redis.io/commands/)
- [Neo4j Guide](https://neo4j.com/docs/)
