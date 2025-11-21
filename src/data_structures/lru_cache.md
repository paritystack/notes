# LRU Cache (Least Recently Used Cache)

## Table of Contents
- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Design Principles](#design-principles)
- [Data Structure Design](#data-structure-design)
- [Operations](#operations)
  - [Get](#get-operation)
  - [Put](#put-operation)
- [Time and Space Complexity](#time-and-space-complexity)
- [Implementation](#implementation)
  - [Python Implementation](#python-implementation)
  - [JavaScript Implementation](#javascript-implementation)
- [Variations](#variations)
  - [LFU Cache](#lfu-cache)
  - [TTL Cache](#ttl-cache)
  - [LRU-K](#lru-k)
- [Real-World Applications](#real-world-applications)
- [Common Problems](#common-problems)
- [Interview Patterns](#interview-patterns)
- [Advanced Topics](#advanced-topics)

## Overview

An **LRU (Least Recently Used) Cache** is a data structure that stores a limited number of items and automatically evicts the least recently used item when the cache reaches capacity. It's one of the most common cache eviction policies.

### Why LRU Cache?

LRU caches are essential for:
- **Memory management** - Limited memory requires eviction strategies
- **Performance optimization** - Fast access to frequently used data
- **Resource management** - CPU caches, web browsers, databases
- **System design** - Distributed caching (Redis, Memcached)

### Real-World Applications

1. **Operating Systems**: Page replacement in virtual memory
2. **Web Browsers**: Browser cache management
3. **Databases**: Query result caching (MySQL, PostgreSQL)
4. **CDNs**: Content delivery and caching
5. **APIs**: Rate limiting and response caching
6. **Applications**: Recently viewed items, autocomplete suggestions

## Key Concepts

### Least Recently Used (LRU) Policy

**Rule**: When cache is full, remove the item that was accessed longest ago

```
Cache capacity: 3

Access pattern: 1, 2, 3, 1, 4
                        ↑
                   Cache full, need to evict

Cache state:
[3, 2, 1] → Access 1 → [1, 3, 2]
[1, 3, 2] → Access 4 → [4, 1, 3] (evict 2, least recently used)
```

### Access Updates Recency

Both **get** and **put** operations update an item's position:

```
Initial: [3, 2, 1]
Get(2):  [2, 3, 1]  ← 2 moved to front (most recent)
Put(4):  [4, 2, 3]  ← 1 evicted, 4 added at front
```

### Cache Hit vs Miss

- **Cache Hit**: Key exists in cache, return value (fast)
- **Cache Miss**: Key not in cache, fetch from source (slow)

```python
value = cache.get(key)
if value is not None:
    # Cache HIT - O(1)
    return value
else:
    # Cache MISS - fetch from database/API
    value = fetch_from_source(key)
    cache.put(key, value)
    return value
```

## Design Principles

### Requirements

1. **O(1) get operation**: Fast lookup by key
2. **O(1) put operation**: Fast insertion and eviction
3. **O(1) eviction**: Quickly identify and remove LRU item
4. **Fixed capacity**: Maintain size limit
5. **Recency tracking**: Track access order efficiently

### Why Hash Table + Doubly Linked List?

| Requirement | Data Structure | Reason |
|------------|----------------|--------|
| O(1) lookup | **Hash Table** | Direct access by key |
| O(1) insertion/deletion | **Doubly Linked List** | Can remove from middle efficiently |
| Track access order | **Doubly Linked List** | Maintain order from MRU to LRU |
| Move to front | **Doubly Linked List** | Easy reordering with prev/next pointers |

### Architecture Diagram

```
Hash Table (for O(1) lookup)
┌─────────┬──────────┐
│  Key    │  Node*   │  ← Points to node in linked list
├─────────┼──────────┤
│  "a"    │  ●───────┼──┐
│  "b"    │  ●───────┼─┐│
│  "c"    │  ●───────┼┐││
└─────────┴──────────┘│││
                      │││
Doubly Linked List (for maintaining order)
                      │││
    head ←──────────────┘││
    ↓                  ││
┌────────┐  ┌────────┐ ↓│ ┌────────┐
│ key: c │←→│ key: b │←→│→│ key: a │  ← tail (LRU)
│ val: 3 │  │ val: 2 │   │ val: 1 │
└────────┘  └────────┘   └────────┘
    ↑
Most Recently Used (MRU)
```

## Data Structure Design

### Node Structure

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key      # Store key for eviction
        self.value = value
        self.prev = None    # Previous node (more recent)
        self.next = None    # Next node (less recent)
```

### Why Store Key in Node?

When evicting, we need to delete from hash table too:

```python
# Remove LRU node
lru_node = self.tail.prev
self.remove_node(lru_node)
del self.cache[lru_node.key]  # ← Need key to delete from hash table
```

### Sentinel Nodes (Dummy Head/Tail)

Using dummy head/tail nodes simplifies edge cases:

```python
# Without sentinels - need to check if head/tail is None
if self.head is None:
    self.head = node
else:
    # Handle insertion...

# With sentinels - no null checks needed!
self.head.next = node
node.prev = self.head
```

## Operations

### Get Operation

**Goal**: Return value for key, mark as most recently used

**Steps**:
1. Check if key exists in hash table
2. If not found, return -1 (or None)
3. If found, move node to head (most recent)
4. Return value

**Visual Example**:
```
Before get("b"):
head → [c] ←→ [b] ←→ [a] ← tail
        MRU          LRU

After get("b"):
head → [b] ←→ [c] ←→ [a] ← tail
        MRU          LRU
```

**Time Complexity**: O(1)

### Put Operation

**Goal**: Insert/update key-value pair, evict LRU if needed

**Steps**:
1. If key exists:
   - Update value
   - Move to head (most recent)
2. If key doesn't exist:
   - Check if cache is full
   - If full, remove LRU node (tail.prev) and delete from hash table
   - Create new node
   - Add to head
   - Add to hash table

**Visual Example**:
```
Cache capacity: 3

Before put(4, "d") - Cache is full:
head → [3:"c"] ←→ [2:"b"] ←→ [1:"a"] ← tail
        MRU                    LRU

After put(4, "d") - Evicted key=1:
head → [4:"d"] ←→ [3:"c"] ←→ [2:"b"] ← tail
        MRU                    LRU
```

**Time Complexity**: O(1)

## Time and Space Complexity

| Operation | Time Complexity | Explanation |
|-----------|----------------|-------------|
| get(key) | O(1) | Hash table lookup + move to head |
| put(key, value) | O(1) | Hash table insert + add to head + possible eviction |
| evict() | O(1) | Remove from tail + delete from hash table |

**Space Complexity**: O(capacity)
- Hash table: O(capacity)
- Doubly linked list: O(capacity)
- Total: O(capacity)

## Implementation

### Python Implementation

**Complete Implementation with Sentinel Nodes**:

```python
class Node:
    """Doubly linked list node"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    LRU Cache implementation using hash table + doubly linked list.

    Maintains O(1) time complexity for both get and put operations.
    """

    def __init__(self, capacity: int):
        """
        Initialize LRU cache with given capacity.

        Args:
            capacity: Maximum number of items in cache
        """
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Sentinel nodes to simplify edge cases
        self.head = Node()  # Dummy head (most recent)
        self.tail = Node()  # Dummy tail (least recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove_node(self, node: Node) -> None:
        """
        Remove node from its current position in linked list.

        Args:
            node: Node to remove

        Time Complexity: O(1)
        """
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: Node) -> None:
        """
        Add node right after head (most recent position).

        Args:
            node: Node to add

        Time Complexity: O(1)
        """
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        """
        Move node to head (mark as most recently used).

        Args:
            node: Node to move

        Time Complexity: O(1)
        """
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> Node:
        """
        Remove and return LRU node (node before tail).

        Returns:
            The removed node

        Time Complexity: O(1)
        """
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node

    def get(self, key: int) -> int:
        """
        Get value for key, mark as recently used.

        Args:
            key: Key to look up

        Returns:
            Value if found, -1 otherwise

        Time Complexity: O(1)
        """
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._move_to_head(node)  # Mark as recently used
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Insert or update key-value pair.
        Evicts LRU item if cache is full.

        Args:
            key: Key to insert/update
            value: Value to store

        Time Complexity: O(1)
        """
        if key in self.cache:
            # Update existing key
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # Insert new key
            new_node = Node(key, value)

            self.cache[key] = new_node
            self._add_to_head(new_node)

            if len(self.cache) > self.capacity:
                # Remove LRU item
                lru_node = self._remove_tail()
                del self.cache[lru_node.key]


# Example usage
cache = LRUCache(2)  # Capacity = 2

cache.put(1, 1)       # Cache: {1=1}
cache.put(2, 2)       # Cache: {1=1, 2=2}
print(cache.get(1))   # Returns 1, Cache: {2=2, 1=1}
cache.put(3, 3)       # Evicts key 2, Cache: {1=1, 3=3}
print(cache.get(2))   # Returns -1 (not found)
cache.put(4, 4)       # Evicts key 1, Cache: {3=3, 4=4}
print(cache.get(1))   # Returns -1 (not found)
print(cache.get(3))   # Returns 3
print(cache.get(4))   # Returns 4
```

**Using Python's OrderedDict**:

```python
from collections import OrderedDict

class LRUCache:
    """
    LRU Cache using OrderedDict (simpler but same complexity).

    OrderedDict maintains insertion order and supports move_to_end().
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)

        self.cache[key] = value

        if len(self.cache) > self.capacity:
            # Remove first item (least recent)
            self.cache.popitem(last=False)


# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # Evicts key 2
print(cache.get(2))  # -1
```

### JavaScript Implementation

**Complete Implementation**:

```javascript
class Node {
    constructor(key = 0, value = 0) {
        this.key = key;
        this.value = value;
        this.prev = null;
        this.next = null;
    }
}

class LRUCache {
    /**
     * Initialize LRU cache with given capacity
     * @param {number} capacity - Maximum cache size
     */
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map(); // key -> Node

        // Sentinel nodes
        this.head = new Node();
        this.tail = new Node();
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    /**
     * Remove node from linked list
     * @param {Node} node - Node to remove
     */
    _removeNode(node) {
        const prevNode = node.prev;
        const nextNode = node.next;
        prevNode.next = nextNode;
        nextNode.prev = prevNode;
    }

    /**
     * Add node right after head
     * @param {Node} node - Node to add
     */
    _addToHead(node) {
        node.prev = this.head;
        node.next = this.head.next;
        this.head.next.prev = node;
        this.head.next = node;
    }

    /**
     * Move node to head (mark as most recently used)
     * @param {Node} node - Node to move
     */
    _moveToHead(node) {
        this._removeNode(node);
        this._addToHead(node);
    }

    /**
     * Remove and return LRU node
     * @returns {Node} The removed node
     */
    _removeTail() {
        const lruNode = this.tail.prev;
        this._removeNode(lruNode);
        return lruNode;
    }

    /**
     * Get value for key
     * @param {number} key
     * @returns {number} Value if found, -1 otherwise
     */
    get(key) {
        if (!this.cache.has(key)) {
            return -1;
        }

        const node = this.cache.get(key);
        this._moveToHead(node);
        return node.value;
    }

    /**
     * Insert or update key-value pair
     * @param {number} key
     * @param {number} value
     */
    put(key, value) {
        if (this.cache.has(key)) {
            // Update existing
            const node = this.cache.get(key);
            node.value = value;
            this._moveToHead(node);
        } else {
            // Insert new
            const newNode = new Node(key, value);
            this.cache.set(key, newNode);
            this._addToHead(newNode);

            if (this.cache.size > this.capacity) {
                // Evict LRU
                const lruNode = this._removeTail();
                this.cache.delete(lruNode.key);
            }
        }
    }
}

// Example usage
const cache = new LRUCache(2);
cache.put(1, 1);
cache.put(2, 2);
console.log(cache.get(1));  // 1
cache.put(3, 3);             // Evicts key 2
console.log(cache.get(2));  // -1
```

**Using JavaScript Map (Simpler)**:

```javascript
class LRUCache {
    /**
     * LRU Cache using Map (maintains insertion order in ES6+)
     */
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key) {
        if (!this.cache.has(key)) {
            return -1;
        }

        // Move to end (most recent)
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    put(key, value) {
        if (this.cache.has(key)) {
            // Delete to re-insert (moves to end)
            this.cache.delete(key);
        }

        this.cache.set(key, value);

        if (this.cache.size > this.capacity) {
            // Remove first item (LRU)
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }
}
```

**TypeScript Version with Generic Types**:

```typescript
class LRUCache<K, V> {
    private capacity: number;
    private cache: Map<K, V>;

    constructor(capacity: number) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key: K): V | undefined {
        if (!this.cache.has(key)) {
            return undefined;
        }

        const value = this.cache.get(key)!;
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    put(key: K, value: V): void {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        }

        this.cache.set(key, value);

        if (this.cache.size > this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }

    has(key: K): boolean {
        return this.cache.has(key);
    }

    size(): number {
        return this.cache.size;
    }
}

// Usage with type safety
const cache = new LRUCache<string, number>(100);
cache.put("user:123", 42);
const value = cache.get("user:123"); // Type: number | undefined
```

## Variations

### LFU Cache (Least Frequently Used)

Evicts item with lowest access frequency instead of least recent:

```python
class LFUCache:
    """
    LFU Cache - evicts least frequently used item.
    Breaks ties using LRU among items with same frequency.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> (value, freq)
        self.freq_map = {}  # freq -> OrderedDict of keys
        self.min_freq = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        value, freq = self.cache[key]

        # Remove from current frequency list
        del self.freq_map[freq][key]
        if not self.freq_map[freq] and freq == self.min_freq:
            self.min_freq += 1

        # Add to next frequency list
        new_freq = freq + 1
        if new_freq not in self.freq_map:
            self.freq_map[new_freq] = OrderedDict()
        self.freq_map[new_freq][key] = None

        self.cache[key] = (value, new_freq)
        return value

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return

        if key in self.cache:
            _, freq = self.cache[key]
            self.cache[key] = (value, freq)
            self.get(key)  # Update frequency
            return

        if len(self.cache) >= self.capacity:
            # Evict LFU (and LRU among LFU)
            evict_key, _ = self.freq_map[self.min_freq].popitem(last=False)
            del self.cache[evict_key]

        # Add new key with frequency 1
        self.cache[key] = (value, 1)
        self.min_freq = 1
        if 1 not in self.freq_map:
            self.freq_map[1] = OrderedDict()
        self.freq_map[1][key] = None
```

### TTL Cache (Time-To-Live)

Items expire after a certain time:

```python
import time

class TTLCache:
    """
    Cache with time-to-live (TTL) for entries.
    Entries expire after ttl seconds.
    """

    def __init__(self, capacity: int, ttl: float):
        self.capacity = capacity
        self.ttl = ttl
        self.cache = {}  # key -> (value, expiry_time)
        self.access_order = OrderedDict()  # key -> None (for LRU)

    def _is_expired(self, key: int) -> bool:
        """Check if entry has expired"""
        if key not in self.cache:
            return True
        _, expiry = self.cache[key]
        return time.time() > expiry

    def get(self, key: int) -> int:
        if self._is_expired(key):
            if key in self.cache:
                del self.cache[key]
                del self.access_order[key]
            return -1

        value, expiry = self.cache[key]
        self.access_order.move_to_end(key)
        return value

    def put(self, key: int, value: int) -> None:
        expiry_time = time.time() + self.ttl

        if key in self.cache:
            self.cache[key] = (value, expiry_time)
            self.access_order.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Evict LRU
                lru_key, _ = self.access_order.popitem(last=False)
                del self.cache[lru_key]

            self.cache[key] = (value, expiry_time)
            self.access_order[key] = None
```

### LRU-K

Considers K most recent accesses instead of just the last one:

```python
class LRU2Cache:
    """
    LRU-2: Evicts item with oldest second-to-last access.
    More resistant to one-time access patterns.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.history = {}  # key -> list of access times

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        # Record access time
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(time.time())

        # Keep only last 2 accesses
        if len(self.history[key]) > 2:
            self.history[key].pop(0)

        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.get(key)  # Update access time
            return

        if len(self.cache) >= self.capacity:
            # Find key with oldest second-to-last access
            evict_key = min(
                self.history.keys(),
                key=lambda k: self.history[k][-2] if len(self.history[k]) > 1 else float('inf')
            )
            del self.cache[evict_key]
            del self.history[evict_key]

        self.cache[key] = value
        self.history[key] = [time.time()]
```

## Real-World Applications

### 1. Database Query Cache

```python
class DatabaseCache:
    """Cache database query results"""

    def __init__(self, capacity: int = 1000):
        self.cache = LRUCache(capacity)
        self.stats = {'hits': 0, 'misses': 0}

    def query(self, sql: str):
        # Try cache first
        result = self.cache.get(sql)

        if result != -1:
            self.stats['hits'] += 1
            return result

        # Cache miss - execute query
        self.stats['misses'] += 1
        result = self._execute_query(sql)
        self.cache.put(sql, result)
        return result

    def _execute_query(self, sql: str):
        # Execute actual database query
        pass
```

### 2. Web API Response Cache

```javascript
class APICache {
    constructor(capacity = 100, ttl = 300) { // 5 min TTL
        this.cache = new LRUCache(capacity);
        this.ttl = ttl * 1000; // Convert to ms
        this.timestamps = new Map();
    }

    async fetch(url) {
        const cached = this.cache.get(url);
        const timestamp = this.timestamps.get(url);

        // Check if cached and not expired
        if (cached && Date.now() - timestamp < this.ttl) {
            console.log('Cache HIT:', url);
            return cached;
        }

        // Cache miss or expired
        console.log('Cache MISS:', url);
        const response = await fetch(url);
        const data = await response.json();

        this.cache.put(url, data);
        this.timestamps.set(url, Date.now());

        return data;
    }
}
```

### 3. In-Memory Session Store

```python
class SessionStore:
    """Store user sessions with LRU eviction"""

    def __init__(self, max_sessions: int = 10000):
        self.sessions = LRUCache(max_sessions)

    def create_session(self, session_id: str, user_data: dict):
        self.sessions.put(session_id, user_data)

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, user_data: dict):
        self.sessions.put(session_id, user_data)

    def delete_session(self, session_id: str):
        # Implementation depends on cache structure
        pass
```

## Common Problems

### LeetCode Problems

| Problem | Difficulty | Key Concept |
|---------|-----------|-------------|
| [146. LRU Cache](https://leetcode.com/problems/lru-cache/) | Medium | Standard LRU implementation |
| [460. LFU Cache](https://leetcode.com/problems/lfu-cache/) | Hard | Frequency-based eviction |
| [1756. Design Most Recently Used Queue](https://leetcode.com/problems/design-most-recently-used-queue/) | Medium | Modified LRU |

### Related Interview Problems

1. **Design browser history** - LRU for back/forward navigation
2. **Implement autocomplete** - LRU for recent searches
3. **Design file system cache** - LRU for file contents
4. **Rate limiter** - Token bucket with LRU

## Interview Patterns

### Pattern 1: Basic LRU Cache

**Template**:
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

**When to use**: Standard LRU cache problem

### Pattern 2: LRU with Expiration

**Template**:
```python
def get(self, key):
    if key in self.cache:
        value, expiry = self.cache[key]
        if time.time() < expiry:
            self._move_to_head(key)
            return value
        else:
            del self.cache[key]  # Expired
    return -1
```

**When to use**: Caching with TTL requirements

### Pattern 3: Multi-level Cache

**Template**:
```python
class MultiLevelCache:
    def __init__(self, l1_size, l2_size):
        self.l1 = LRUCache(l1_size)  # Fast, small
        self.l2 = LRUCache(l2_size)  # Slower, larger

    def get(self, key):
        # Check L1 first
        value = self.l1.get(key)
        if value != -1:
            return value

        # Check L2
        value = self.l2.get(key)
        if value != -1:
            self.l1.put(key, value)  # Promote to L1
            return value

        return -1
```

**When to use**: Hierarchical caching systems

## Advanced Topics

### Thread-Safe LRU Cache

```python
import threading

class ThreadSafeLRUCache:
    """Thread-safe LRU cache using locks"""

    def __init__(self, capacity: int):
        self.cache = LRUCache(capacity)
        self.lock = threading.RLock()

    def get(self, key: int) -> int:
        with self.lock:
            return self.cache.get(key)

    def put(self, key: int, value: int) -> None:
        with self.lock:
            self.cache.put(key, value)
```

### Distributed LRU Cache

```python
class DistributedLRUCache:
    """
    Distributed cache using consistent hashing.
    Each node maintains its own LRU cache.
    """

    def __init__(self, nodes: list, capacity_per_node: int):
        self.nodes = nodes
        self.caches = {node: LRUCache(capacity_per_node) for node in nodes}

    def _get_node(self, key: int) -> str:
        """Consistent hashing to determine node"""
        return self.nodes[hash(key) % len(self.nodes)]

    def get(self, key: int) -> int:
        node = self._get_node(key)
        return self.caches[node].get(key)

    def put(self, key: int, value: int) -> None:
        node = self._get_node(key)
        self.caches[node].put(key, value)
```

### Write-Through vs Write-Back Cache

```python
class WriteThroughCache:
    """
    Write-through: Write to cache AND database immediately.
    Guarantees consistency but slower writes.
    """

    def put(self, key, value):
        self.cache.put(key, value)
        self.database.write(key, value)  # Synchronous


class WriteBackCache:
    """
    Write-back: Write to cache first, database later (async).
    Faster writes but risk of data loss.
    """

    def put(self, key, value):
        self.cache.put(key, value)
        self.dirty_keys.add(key)
        # Async flush to database later
```

## Key Takeaways

1. **Hybrid data structure**: Hash table for O(1) lookup + doubly linked list for O(1) reordering
2. **Sentinel nodes**: Simplify edge cases (no null checks)
3. **Store key in node**: Needed for eviction (to delete from hash table)
4. **Both get and put update recency**: Any access marks item as recently used
5. **OrderedDict shortcut**: Python's OrderedDict can simplify implementation
6. **Map ordering**: JavaScript Map maintains insertion order (ES6+)
7. **Trade-offs**: LRU is simple and effective but not optimal for all access patterns

## When to Use LRU Cache

✅ **Use when**:
- Limited memory and need automatic eviction
- Recent items likely to be accessed again (temporal locality)
- Want simple, predictable eviction policy
- Need O(1) get and put operations

❌ **Don't use when**:
- Access frequency matters more than recency (use LFU)
- Need persistent storage (use database)
- Working set larger than cache (high miss rate)
- Need complex eviction logic (custom policy)

---

**Time to Implement**: 15-20 minutes (with doubly linked list)

**Space Complexity**: O(capacity)

**Most Common Interview Question**: [LeetCode 146. LRU Cache](https://leetcode.com/problems/lru-cache/)

**Pro Tip**: Master the doubly linked list + hash table approach, but know OrderedDict/Map shortcuts for quick implementation.
