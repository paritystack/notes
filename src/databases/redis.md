# Redis

Redis (Remote Dictionary Server) is an open-source, in-memory data structure store used as a database, cache, message broker, and streaming engine. Known for its high performance and versatility, Redis supports various data structures and is widely used for real-time applications.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Data Structures](#data-structures)
- [Common Operations](#common-operations)
- [Caching Strategies](#caching-strategies)
- [Pub/Sub Messaging](#pubsub-messaging)
- [Redis with Node.js](#redis-with-nodejs)
- [Best Practices](#best-practices)
- [Performance and Persistence](#performance-and-persistence)

---

## Introduction

**Key Features:**
- In-memory data storage
- Sub-millisecond latency
- Multiple data structures (strings, hashes, lists, sets, sorted sets)
- Pub/Sub messaging
- Transactions
- Lua scripting
- Persistence options (RDB, AOF)
- Replication and high availability
- Clustering for horizontal scaling

**Use Cases:**
- Caching
- Session storage
- Real-time analytics
- Leaderboards and counting
- Rate limiting
- Message queues
- Real-time chat applications
- Geospatial data

---

## Installation and Setup

### Install Redis

**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

**Docker:**
```bash
docker run -d -p 6379:6379 --name redis redis:latest
```

### Redis CLI

```bash
# Connect to Redis
redis-cli

# Test connection
127.0.0.1:6379> PING
PONG

# Select database (0-15)
127.0.0.1:6379> SELECT 1

# Get all keys
127.0.0.1:6379> KEYS *

# Clear database
127.0.0.1:6379> FLUSHDB

# Clear all databases
127.0.0.1:6379> FLUSHALL
```

---

## Data Structures

### Strings

```bash
# Set and get
SET name "John Doe"
GET name

# Set with expiration (seconds)
SETEX session:123 3600 "user_data"

# Set if not exists
SETNX key "value"

# Multiple set/get
MSET key1 "value1" key2 "value2"
MGET key1 key2

# Increment/decrement
SET counter 10
INCR counter        # 11
INCRBY counter 5    # 16
DECR counter        # 15
DECRBY counter 3    # 12

# Append
APPEND key "more_data"

# Get length
STRLEN key
```

### Hashes (Objects)

```bash
# Set hash field
HSET user:1 name "John" age 30 email "john@example.com"

# Get hash field
HGET user:1 name

# Get all fields
HGETALL user:1

# Get multiple fields
HMGET user:1 name email

# Check if field exists
HEXISTS user:1 name

# Delete field
HDEL user:1 age

# Get all keys/values
HKEYS user:1
HVALS user:1

# Increment hash field
HINCRBY user:1 loginCount 1
```

### Lists

```bash
# Push to list
LPUSH mylist "first"    # Push to left
RPUSH mylist "last"     # Push to right

# Pop from list
LPOP mylist             # Pop from left
RPOP mylist             # Pop from right

# Get range
LRANGE mylist 0 -1      # Get all
LRANGE mylist 0 9       # Get first 10

# Get by index
LINDEX mylist 0

# List length
LLEN mylist

# Trim list
LTRIM mylist 0 99       # Keep first 100 items

# Blocking pop (for queues)
BLPOP mylist 0          # Block until item available
```

### Sets

```bash
# Add members
SADD myset "member1" "member2" "member3"

# Get all members
SMEMBERS myset

# Check membership
SISMEMBER myset "member1"

# Remove member
SREM myset "member1"

# Set operations
SUNION set1 set2        # Union
SINTER set1 set2        # Intersection
SDIFF set1 set2         # Difference

# Random member
SRANDMEMBER myset
SPOP myset              # Pop random member

# Set size
SCARD myset
```

### Sorted Sets (Leaderboards)

```bash
# Add members with scores
ZADD leaderboard 100 "player1" 200 "player2" 150 "player3"

# Get range by rank
ZRANGE leaderboard 0 9           # Top 10 (ascending)
ZREVRANGE leaderboard 0 9        # Top 10 (descending)

# Get range with scores
ZRANGE leaderboard 0 9 WITHSCORES

# Get rank
ZRANK leaderboard "player1"      # Ascending rank
ZREVRANK leaderboard "player1"   # Descending rank

# Get score
ZSCORE leaderboard "player1"

# Increment score
ZINCRBY leaderboard 50 "player1"

# Range by score
ZRANGEBYSCORE leaderboard 100 200

# Count in range
ZCOUNT leaderboard 100 200

# Remove member
ZREM leaderboard "player1"
```

---

## Common Operations

### Key Management

```bash
# Set expiration
EXPIRE key 60           # Expire in 60 seconds
EXPIREAT key 1609459200 # Expire at timestamp
TTL key                 # Get time to live
PERSIST key             # Remove expiration

# Delete keys
DEL key1 key2 key3

# Check if key exists
EXISTS key

# Get key type
TYPE key

# Rename key
RENAME oldkey newkey
RENAMENX oldkey newkey  # Rename if new key doesn't exist

# Get all keys matching pattern
KEYS user:*
SCAN 0 MATCH user:* COUNT 10  # Better for production
```

### Transactions

```bash
MULTI
SET key1 "value1"
SET key2 "value2"
INCR counter
EXEC

# With watch (optimistic locking)
WATCH key
MULTI
SET key "new_value"
EXEC
```

---

## Caching Strategies

### Cache-Aside (Lazy Loading)

```javascript
async function getUser(id) {
  const cacheKey = `user:${id}`;

  // Try cache first
  let user = await redis.get(cacheKey);

  if (user) {
    return JSON.parse(user);
  }

  // Cache miss - load from database
  user = await db.users.findById(id);

  // Store in cache
  await redis.setex(cacheKey, 3600, JSON.stringify(user));

  return user;
}
```

### Write-Through Cache

```javascript
async function updateUser(id, data) {
  const cacheKey = `user:${id}`;

  // Update database
  const user = await db.users.updateById(id, data);

  // Update cache
  await redis.setex(cacheKey, 3600, JSON.stringify(user));

  return user;
}
```

### Write-Behind (Write-Back) Cache

```javascript
async function updateUser(id, data) {
  const cacheKey = `user:${id}`;

  // Update cache immediately
  await redis.setex(cacheKey, 3600, JSON.stringify(data));

  // Queue database write
  await redis.lpush('user:updates', JSON.stringify({ id, data }));

  return data;
}

// Background worker
async function processUpdates() {
  while (true) {
    const update = await redis.brpop('user:updates', 0);
    if (update) {
      const { id, data } = JSON.parse(update[1]);
      await db.users.updateById(id, data);
    }
  }
}
```

---

## Pub/Sub Messaging

### Basic Pub/Sub

```javascript
const redis = require('redis');

// Publisher
const publisher = redis.createClient();

publisher.publish('news', 'Breaking news!');

// Subscriber
const subscriber = redis.createClient();

subscriber.subscribe('news');

subscriber.on('message', (channel, message) => {
  console.log(`Received from ${channel}: ${message}`);
});

// Pattern subscribe
subscriber.psubscribe('user:*');

subscriber.on('pmessage', (pattern, channel, message) => {
  console.log(`Pattern ${pattern}, Channel ${channel}: ${message}`);
});
```

### Real-Time Chat Example

```javascript
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const redis = require('redis');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const publisher = redis.createClient();
const subscriber = redis.createClient();

subscriber.subscribe('chat:messages');

// Handle Redis messages
subscriber.on('message', (channel, message) => {
  if (channel === 'chat:messages') {
    io.emit('message', JSON.parse(message));
  }
});

// Handle WebSocket connections
io.on('connection', (socket) => {
  console.log('User connected');

  socket.on('message', (msg) => {
    const message = {
      user: socket.id,
      text: msg,
      timestamp: Date.now()
    };

    // Publish to Redis
    publisher.publish('chat:messages', JSON.stringify(message));

    // Store in list
    redis.lpush('chat:history', JSON.stringify(message));
    redis.ltrim('chat:history', 0, 99); // Keep last 100 messages
  });

  socket.on('disconnect', () => {
    console.log('User disconnected');
  });
});

server.listen(3000);
```

---

## Redis with Node.js

### Using node-redis

```bash
npm install redis
```

**Basic Usage:**
```javascript
const redis = require('redis');

const client = redis.createClient({
  url: 'redis://localhost:6379'
});

client.on('error', (err) => console.error('Redis error:', err));
client.on('connect', () => console.log('Connected to Redis'));

await client.connect();

// String operations
await client.set('key', 'value');
const value = await client.get('key');

// Hash operations
await client.hSet('user:1', 'name', 'John');
await client.hSet('user:1', 'age', '30');
const user = await client.hGetAll('user:1');

// List operations
await client.lPush('mylist', 'item1');
await client.rPush('mylist', 'item2');
const items = await client.lRange('mylist', 0, -1);

// Set operations
await client.sAdd('myset', 'member1');
await client.sAdd('myset', 'member2');
const members = await client.sMembers('myset');

// Sorted set operations
await client.zAdd('leaderboard', { score: 100, value: 'player1' });
const top = await client.zRange('leaderboard', 0, 9, { REV: true });

await client.disconnect();
```

### Caching Middleware (Express)

```javascript
const redis = require('redis');
const client = redis.createClient();

await client.connect();

function cache(duration) {
  return async (req, res, next) => {
    const key = `cache:${req.originalUrl}`;

    try {
      const cachedResponse = await client.get(key);

      if (cachedResponse) {
        return res.json(JSON.parse(cachedResponse));
      }

      // Modify res.json to cache response
      const originalJson = res.json.bind(res);

      res.json = (body) => {
        client.setex(key, duration, JSON.stringify(body));
        return originalJson(body);
      };

      next();
    } catch (error) {
      next();
    }
  };
}

// Usage
app.get('/api/users', cache(300), async (req, res) => {
  const users = await db.users.findAll();
  res.json(users);
});
```

### Session Storage

```javascript
const session = require('express-session');
const RedisStore = require('connect-redis').default;
const redis = require('redis');

const redisClient = redis.createClient();
await redisClient.connect();

app.use(
  session({
    store: new RedisStore({ client: redisClient }),
    secret: 'your-secret',
    resave: false,
    saveUninitialized: false,
    cookie: {
      secure: false, // Set true for HTTPS
      httpOnly: true,
      maxAge: 1000 * 60 * 60 * 24 // 1 day
    }
  })
);

app.get('/', (req, res) => {
  if (req.session.views) {
    req.session.views++;
  } else {
    req.session.views = 1;
  }
  res.send(`Views: ${req.session.views}`);
});
```

### Rate Limiting

```javascript
async function rateLimiter(userId, maxRequests = 10, windowSeconds = 60) {
  const key = `rate_limit:${userId}`;
  const current = await client.incr(key);

  if (current === 1) {
    await client.expire(key, windowSeconds);
  }

  if (current > maxRequests) {
    const ttl = await client.ttl(key);
    throw new Error(`Rate limit exceeded. Try again in ${ttl} seconds`);
  }

  return {
    remaining: maxRequests - current,
    reset: windowSeconds
  };
}

// Middleware
async function rateLimitMiddleware(req, res, next) {
  const userId = req.user?.id || req.ip;

  try {
    const result = await rateLimiter(userId);
    res.set('X-RateLimit-Remaining', result.remaining);
    res.set('X-RateLimit-Reset', result.reset);
    next();
  } catch (error) {
    res.status(429).json({ error: error.message });
  }
}

app.use(rateLimitMiddleware);
```

### Distributed Locking

```javascript
async function acquireLock(lockKey, timeout = 10000) {
  const lockValue = Math.random().toString(36);
  const result = await client.set(lockKey, lockValue, {
    NX: true,
    PX: timeout
  });

  if (result === 'OK') {
    return lockValue;
  }

  return null;
}

async function releaseLock(lockKey, lockValue) {
  const script = `
    if redis.call("get", KEYS[1]) == ARGV[1] then
      return redis.call("del", KEYS[1])
    else
      return 0
    end
  `;

  return await client.eval(script, {
    keys: [lockKey],
    arguments: [lockValue]
  });
}

// Usage
async function criticalSection() {
  const lock = await acquireLock('resource:lock');

  if (!lock) {
    throw new Error('Could not acquire lock');
  }

  try {
    // Perform critical operation
    await performOperation();
  } finally {
    await releaseLock('resource:lock', lock);
  }
}
```

---

## Best Practices

### 1. Key Naming Conventions

```bash
# Use descriptive, hierarchical names
user:1:profile
user:1:sessions
order:12345:items

# Use consistent separators
user:1:profile  # Colon-separated
user_1_profile  # Underscore-separated

# Include type in key name
string:user:1:name
hash:user:1
list:user:1:notifications
```

### 2. Set Expiration Times

```javascript
// Always set TTL for cache keys
await client.setex('cache:key', 3600, 'value');

// Use appropriate expiration times
const MINUTE = 60;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

await client.setex('session:123', 30 * MINUTE, data);
await client.setex('cache:user:1', 1 * HOUR, data);
await client.setex('temp:verification', 10 * MINUTE, code);
```

### 3. Use Pipelines for Multiple Commands

```javascript
const pipeline = client.multi();

pipeline.set('key1', 'value1');
pipeline.set('key2', 'value2');
pipeline.incr('counter');
pipeline.hSet('user:1', 'name', 'John');

const results = await pipeline.exec();
```

### 4. Handle Connection Errors

```javascript
const client = redis.createClient({
  url: 'redis://localhost:6379',
  socket: {
    reconnectStrategy: (retries) => {
      if (retries > 10) {
        return new Error('Max retries reached');
      }
      return retries * 100;
    }
  }
});

client.on('error', (err) => {
  console.error('Redis error:', err);
});

client.on('reconnecting', () => {
  console.log('Reconnecting to Redis...');
});

client.on('ready', () => {
  console.log('Redis is ready');
});
```

### 5. Memory Management

```javascript
// Set maxmemory and eviction policy in redis.conf
// maxmemory 256mb
// maxmemory-policy allkeys-lru

// Monitor memory usage
const info = await client.info('memory');
console.log(info);

// Use SCAN instead of KEYS
let cursor = 0;
do {
  const result = await client.scan(cursor, {
    MATCH: 'user:*',
    COUNT: 100
  });
  cursor = result.cursor;
  const keys = result.keys;
  // Process keys
} while (cursor !== 0);
```

---

## Performance and Persistence

### Persistence Options

**RDB (Redis Database Backup):**
```conf
# redis.conf
save 900 1      # Save if 1 key changed in 15 minutes
save 300 10     # Save if 10 keys changed in 5 minutes
save 60 10000   # Save if 10000 keys changed in 1 minute

dbfilename dump.rdb
dir /var/lib/redis
```

**AOF (Append Only File):**
```conf
# redis.conf
appendonly yes
appendfilename "appendonly.aof"

# Sync strategy
appendfsync always      # Slowest, safest
appendfsync everysec   # Good balance (recommended)
appendfsync no         # Fastest, least safe
```

### Replication

```bash
# On replica
redis-cli
> REPLICAOF master-host 6379

# Check replication status
> INFO replication
```

### Monitoring

```javascript
// Monitor commands
client.monitor((err, res) => {
  console.log(res);
});

// Get stats
const info = await client.info();
console.log(info);

// Slow log
const slowlog = await client.slowlog('GET', 10);
console.log(slowlog);
```

---

## Resources

**Official Documentation:**
- [Redis Documentation](https://redis.io/documentation)
- [Redis Commands](https://redis.io/commands)
- [node-redis](https://github.com/redis/node-redis)

**Tools:**
- [RedisInsight](https://redis.com/redis-enterprise/redis-insight/) - GUI
- [redis-cli](https://redis.io/topics/rediscli) - Command line
- [redis-benchmark](https://redis.io/topics/benchmarks) - Performance testing

**Learning:**
- [Redis University](https://university.redis.com/)
- [Try Redis](https://try.redis.io/)
- [Redis Patterns](https://redis.io/topics/patterns)
