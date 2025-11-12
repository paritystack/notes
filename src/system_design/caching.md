# Caching Strategies

## Overview

Caching stores frequently accessed data in fast memory to reduce latency and database load.

## Cache Levels

```
L1: Browser cache (browser memory)
  ↓
L2: CDN cache (edge servers)
  ↓
L3: Application cache (Redis)
  ↓
L4: Database cache (MySQL buffer pool)
  ↓
Database (disk, slowest)
```

## Caching Policies

### Cache-Aside (Lazy Loading)

```
1. Check cache
2. If miss: Load from database
3. Store in cache
4. Return to client
```

```python
def get_user(user_id):
    # Check cache
    cached = redis.get(f"user:{user_id}")
    if cached:
        return cached

    # Load from DB
    user = db.get_user(user_id)

    # Store in cache
    redis.set(f"user:{user_id}", user, ex=3600)

    return user
```

### Write-Through

Write to cache AND database simultaneously:

```
Update Request
  ↓
Cache ← updated
  ↓
Database ← updated
```

Ensures consistency but slower writes.

### Write-Behind (Write-Back)

Write to cache, asynchronously to database:

```
Update Request
  ↓
Cache ← updated (fast)
  ↓
Queue for DB
  ↓
Database ← updated (later)
```

Fast but risk of data loss.

## Invalidation Strategies

### TTL (Time-To-Live)

```python
redis.set("key", value, ex=3600)  # Expires in 1 hour
```

**Pros**: Simple
**Cons**: Stale data until expiry

### Event-Based

Invalidate when data changes:

```python
def update_user(user_id, data):
    db.update_user(user_id, data)
    redis.delete(f"user:{user_id}")  # Invalidate
```

**Pros**: Fresh data
**Cons**: Complex logic

### LRU (Least Recently Used)

Remove least used items when full:

```
[recent] A B C D E [old]
Remove E if memory full
```

## Cache Eviction Policies

| Policy | Behavior |
|--------|----------|
| **LRU** | Remove least recently used |
| **LFU** | Remove least frequently used |
| **FIFO** | Remove oldest |
| **Random** | Remove random |

## Distributed Caching

Using Redis for distributed cache:

```python
import redis

cache = redis.Redis(host='localhost', port=6379)

# Set
cache.set('key', 'value')
cache.setex('key', 3600, 'value')  # With TTL

# Get
value = cache.get('key')

# Delete
cache.delete('key')

# Multi-key
cache.mget(['key1', 'key2', 'key3'])
```

## Cache Stampede

Problem: Multiple requests load same expired key

```
3 requests arrive
Cache expired for key X
All 3 hit database (thundering herd)
```

**Solution**: Lock pattern

```python
def get_cached(key):
    value = cache.get(key)
    if value:
        return value

    if cache.get(f"{key}:lock"):
        # Wait, someone loading
        return wait_for_cache(key)

    # Set lock, load data
    cache.set(f"{key}:lock", "1", ex=5)
    value = load_from_db(key)
    cache.set(key, value)
    cache.delete(f"{key}:lock")
    return value
```

## Common Caching Patterns

### Cache Coherence
Multiple caches have same data

### Cache Penetration
Request for non-existent key hits DB repeatedly

**Solution**: Cache negative results
```python
cache.set(f"user:{id}", None, ex=60)
```

### Cache Avalanche
Many keys expire simultaneously

**Solution**: Randomize TTLs
```python
ttl = 3600 + random(0, 600)
cache.set(key, value, ex=ttl)
```

## When NOT to Cache

- Constantly changing data
- Very frequently read, rarely write
- Small datasets
- Rare access patterns

## ELI10

Cache is like keeping your favorite book on your desk:
- Fast access (don't go to library)
- Runs out of space (limited shelf)
- Need to replace old books (eviction)
- Book gets outdated (invalidation)

Trade memory for speed!

## Further Resources

- [Redis Documentation](https://redis.io/docs/)
- [Caching Patterns](https://codeahoy.com/2017/08/11/caching-strategies-and-patterns/)
