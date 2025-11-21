# Hash Tables

## Overview

A hash table (hash map) stores key-value pairs with $O(1)$ average-case lookup, insertion, and deletion. It uses a hash function to map keys to array indices.

## How It Works

### Hash Function
Converts key to index:
```
hash("name") = 5
hash(123) = 2
hash("email") = 5  (collision!)
```

### Collision Resolution

When two keys hash to the same index, we need a strategy to handle the collision.

#### 1. Chaining (Separate Chaining)
Store multiple values at same index using linked lists or arrays.

```
Index 0: []
Index 1: []
Index 2: [123 -> "John"]
Index 3: [456 -> "Jane", 789 -> "Jack"]
Index 4: []
```

**Pros**: Simple, handles high load factors well
**Cons**: Extra memory for pointers, cache performance

#### 2. Linear Probing
Find next empty slot by checking sequentially: `(hash(key) + i) % size`

```
Insert "cat" at index 5 (occupied)
Try: index 5 → 6 → 7 (empty, insert here)
```

**Example sequence**:
```
hash("cat") = 5, hash("dog") = 5
Insert "cat": table[5] = "cat"
Insert "dog": table[5] occupied → try table[6] → insert
```

**Pros**: Good cache locality, no extra memory
**Cons**: Primary clustering (consecutive filled slots)

#### 3. Quadratic Probing
Probe using quadratic increments: `(hash(key) + i²) % size`

```
Insert "cat" at index 5 (occupied)
Try: 5 → 5+1² → 5+2² → 5+3² ...
     5 → 6 → 9 → 14 ...
```

**Pros**: Reduces primary clustering
**Cons**: Secondary clustering, may not probe all slots

#### 4. Double Hashing
Use second hash function for step size: `(hash1(key) + i × hash2(key)) % size`

```
hash1("cat") = 5, hash2("cat") = 3
Try: 5 → 5+3 → 5+6 → 5+9 ...
     5 → 8 → 11 → 14 ...
```

**Pros**: Minimizes clustering
**Cons**: Requires two hash functions

#### 5. Robin Hood Hashing
An optimization of linear probing that "steals from the rich to give to the poor". Elements that have probed farther from their ideal position can displace elements closer to their ideal position.

```
Insert key with hash=5:
- Probe distance (PD) = how far from ideal position
- If current slot occupied, compare probe distances
- If new key's PD > existing key's PD, swap and continue with displaced key

Example:
Position: 0  1  2  3  4  5  6  7
Item:     A     B        C
PD:       0     1        0

Insert D (hash=5, occupied by C with PD=0):
- D's PD would be 0
- Move to position 6, D's PD = 1
- Position 6 has C with PD=0
- Since D's PD(1) > C's PD(0), swap them
Position: 0  1  2  3  4  5  6  7
Item:     A     B        D  C
PD:       0     1        1  1
```

**Pros**: Reduces variance in probe lengths, better average case
**Cons**: More complex insertion logic

#### 6. Cuckoo Hashing
Uses two hash functions and two tables. Each key has two possible positions (one in each table). On collision, displace existing key and rehash it to alternate position.

```
Table 1: h1(key) = key % size1
Table 2: h2(key) = (key // size1) % size2

Insert process:
1. Try h1(key) in Table 1
2. If occupied, kick out existing key
3. Try h2(kicked_key) in Table 2
4. If occupied, kick that out and try h1 in Table 1
5. Continue until empty spot (or cycle detected → rehash all)

Example:
Insert 20: Table1[h1(20)] = 20
Insert 30: Table1[h1(30)] = 30
Insert 40: Collision at Table1[h1(40)]
  → Move 30 to Table2[h2(30)]
  → Insert 40 in Table1
```

**Pros**: Guaranteed $O(1)$ worst-case lookup (check 2 positions only)
**Cons**: Complex insertion, may need full rehash on cycle

#### 7. Hopscotch Hashing
Combines advantages of chaining and open addressing. Each bucket has a "neighborhood" (e.g., next H positions). Keys stay within H positions of their ideal location.

```
Neighborhood size H = 4
Position: 0  1  2  3  4  5  6  7
Bitmap:   1011         1100

Insert key with hash=0:
- If position 0-3 have space, insert there
- If full, use linear probing but keep within neighborhood
- May need to "hop" existing keys to make room

Example (H=4):
hash(K1)=0, stored at position 2 (within 0-3)
hash(K2)=0, stored at position 3 (within 0-3)
hash(K3)=5, stored at position 6 (within 5-8)
```

**Pros**: Good cache performance, bounded search time
**Cons**: Complex implementation, requires bitmap

### Hash Function Design

A good hash function should have these properties:

1. **Deterministic**: Same key always produces same hash
2. **Uniform Distribution**: Spreads keys evenly across table
3. **Fast Computation**: $O(1)$ time complexity
4. **Minimize Collisions**: Different keys rarely produce same hash

#### Common Hash Techniques

**Division Method**: `hash(key) = key % table_size`
```python
# Best with prime number table sizes
hash(123) = 123 % 11 = 2
```

**Multiplication Method**: `hash(key) = floor(m × (key × A mod 1))`
```python
# A ≈ 0.6180339887 (golden ratio), m = table size
A = 0.6180339887
hash(123) = floor(10 * (123 * A % 1)) = 6
```

**Universal Hashing**: Random selection from hash function family
```python
# Reduces worst-case collision probability
# h(k) = ((a*k + b) mod p) mod m
# where p is prime, a,b are random
```

**String Hashing**: Polynomial rolling hash
```python
def hash_string(s, prime=31, mod=10**9+7):
    h = 0
    for char in s:
        h = (h * prime + ord(char)) % mod
    return h
```

**FNV-1a Hash**: Fast, non-cryptographic hash with good distribution
```python
def fnv1a_hash(data, bits=32):
    """FNV-1a hash - popular for hash tables
    32-bit version shown (also available in 64-bit, 128-bit, etc.)
    """
    if bits == 32:
        FNV_prime = 16777619
        offset_basis = 2166136261
        mod = 2**32
    else:  # 64-bit
        FNV_prime = 1099511628211
        offset_basis = 14695981039346656037
        mod = 2**64

    hash_value = offset_basis
    for byte in str(data).encode('utf-8'):
        hash_value = hash_value ^ byte  # XOR with byte
        hash_value = (hash_value * FNV_prime) % mod
    return hash_value

# Example:
# fnv1a_hash("hello") → consistent 32-bit integer
```

**MurmurHash**: Industry-standard non-cryptographic hash
```python
# MurmurHash3 is commonly used in production systems
# Available in libraries like mmh3
# Known for excellent distribution and speed
# Used by: Redis, Cassandra, Hadoop, etc.

import mmh3  # pip install mmh3
hash_value = mmh3.hash("hello", seed=0)  # 32-bit
hash_value = mmh3.hash128("hello", seed=0)  # 128-bit
```

#### Cryptographic vs Non-Cryptographic Hashing

| Property | Non-Cryptographic (e.g., MurmurHash, FNV) | Cryptographic (e.g., SHA-256, MD5) |
|----------|-------------------------------------------|-----------------------------------|
| **Speed** | Very fast (ns per key) | Slower (microseconds) |
| **Purpose** | Hash tables, checksums, partitioning | Security, integrity verification |
| **Collision Resistance** | Good enough for hash tables | Extremely high (intentionally hard) |
| **Reversibility** | Not applicable | Computationally infeasible |
| **When to Use** | Hash tables, Bloom filters, load balancing | Passwords, digital signatures, certificates |

**Rule of Thumb**: Use non-cryptographic hashes for data structures (faster, sufficient). Use cryptographic hashes only when security matters.

**Custom Object Hashing**:
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        # Combine hashes of immutable attributes
        # Use XOR (^) or tuple hashing
        return hash((self.x, self.y))

    def __eq__(self, other):
        # Must implement __eq__ when implementing __hash__
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# Now Point objects can be used as dict keys or in sets
point_dict = {Point(1, 2): "origin"}
point_set = {Point(1, 2), Point(3, 4)}
```

## Operations

| Operation | Average | Worst |
|-----------|---------|-------|
| **Get** | $O(1)$ | $O(n)$ |
| **Set** | $O(1)$ | $O(n)$ |
| **Delete** | $O(1)$ | $O(n)$ |
| **Space** | $O(n)$ | $O(n)$ |

**Note**: Worst case occurs when all keys hash to same index (all collisions). With good hash function and proper load factor, average case is maintained.

### Load Factor & Resizing

**Load Factor** (α) measures how full the hash table is:
$$\alpha = \frac{n}{m}$$
where $n$ = number of elements, $m$ = table size

#### Performance Impact

```
α = 0.5  →  Fast lookups, more memory
α = 0.75 →  Balanced (typical threshold)
α = 1.0  →  Table full (chaining only)
α > 1.0  →  More collisions, slower ops
```

#### Resizing Strategy

When load factor exceeds threshold (typically 0.75):
1. Create new table (usually 2× size)
2. Rehash all existing entries
3. Insert into new table

```python
def resize(self):
    old_table = self.table
    self.size *= 2
    self.table = [[] for _ in range(self.size)]
    self.count = 0

    # Rehash all entries
    for bucket in old_table:
        for key, value in bucket:
            self.set(key, value)
```

**Cost**: $O(n)$ for resize, but amortized $O(1)$ per operation

## Python Implementation

```python
# Built-in dict
d = {"key": "value"}
d.get("key")  # $O(1)$
d["key"] = "new_value"
del d["key"]

# Custom with chaining, resizing, and load factor
class HashTable:
    def __init__(self, size=10, load_factor_threshold=0.75):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.count = 0
        self.load_factor_threshold = load_factor_threshold

    def _hash(self, key):
        """Hash function: O(1)"""
        return hash(key) % self.size

    def _load_factor(self):
        """Calculate current load factor"""
        return self.count / self.size

    def _resize(self):
        """Resize and rehash when load factor exceeds threshold: O(n)"""
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0

        for bucket in old_table:
            for key, value in bucket:
                self.set(key, value)

    def set(self, key, value):
        """Insert or update: O(1) average"""
        index = self._hash(key)

        # Update if key exists
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return

        # Insert new key-value
        self.table[index].append((key, value))
        self.count += 1

        # Resize if needed
        if self._load_factor() > self.load_factor_threshold:
            self._resize()

    def get(self, key):
        """Retrieve value: O(1) average"""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def delete(self, key):
        """Remove key-value pair: O(1) average"""
        index = self._hash(key)
        bucket = self.table[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.count -= 1
                return True
        return False

    def __len__(self):
        return self.count

    def __contains__(self, key):
        return self.get(key) is not None
```

### Open Addressing Implementation (Linear Probing)

```python
class OpenAddressHashTable:
    """Hash table using open addressing with linear probing
    All elements stored directly in table array (no chaining)
    """
    class _DeletedEntry:
        """Sentinel to mark deleted slots"""
        pass

    DELETED = _DeletedEntry()

    def __init__(self, size=10, load_factor_threshold=0.7):
        self.size = size
        self.table = [None] * size
        self.count = 0
        self.load_factor_threshold = load_factor_threshold

    def _hash(self, key):
        """Hash function: O(1)"""
        return hash(key) % self.size

    def _probe(self, key):
        """Linear probing to find slot: O(1) average, O(n) worst"""
        index = self._hash(key)
        original = index

        while self.table[index] is not None:
            if self.table[index] is not self.DELETED:
                stored_key, _ = self.table[index]
                if stored_key == key:
                    return index, True  # Found existing key

            # Linear probe to next slot
            index = (index + 1) % self.size

            # Prevent infinite loop (shouldn't happen with proper load factor)
            if index == original:
                raise Exception("Hash table is full")

        return index, False  # Found empty slot

    def _load_factor(self):
        """Calculate current load factor"""
        return self.count / self.size

    def _resize(self):
        """Resize and rehash: O(n)"""
        old_table = self.table
        self.size *= 2
        self.table = [None] * self.size
        self.count = 0

        for item in old_table:
            if item is not None and item is not self.DELETED:
                key, value = item
                self.set(key, value)

    def set(self, key, value):
        """Insert or update: O(1) average"""
        index, found = self._probe(key)

        if found:
            # Update existing
            self.table[index] = (key, value)
        else:
            # Insert new
            self.table[index] = (key, value)
            self.count += 1

            # Resize if load factor too high
            if self._load_factor() > self.load_factor_threshold:
                self._resize()

    def get(self, key):
        """Retrieve value: O(1) average"""
        index = self._hash(key)
        original = index

        while self.table[index] is not None:
            if self.table[index] is not self.DELETED:
                stored_key, value = self.table[index]
                if stored_key == key:
                    return value

            index = (index + 1) % self.size
            if index == original:
                break

        return None

    def delete(self, key):
        """Remove key-value pair: O(1) average"""
        index = self._hash(key)
        original = index

        while self.table[index] is not None:
            if self.table[index] is not self.DELETED:
                stored_key, _ = self.table[index]
                if stored_key == key:
                    # Mark as deleted (don't set to None to maintain probe chain)
                    self.table[index] = self.DELETED
                    self.count -= 1
                    return True

            index = (index + 1) % self.size
            if index == original:
                break

        return False

    def __len__(self):
        return self.count

    def __contains__(self, key):
        return self.get(key) is not None
```

### Python Dictionary Variants - Practical Usage

```python
# 1. dict - Standard hash table
d = {}
d['key'] = 'value'
d.get('key', 'default')  # Safe access with default

# 2. defaultdict - Auto-initialize missing keys
from collections import defaultdict

# Counting
word_count = defaultdict(int)
for word in words:
    word_count[word] += 1  # No KeyError, starts at 0

# Grouping
groups = defaultdict(list)
for item in items:
    groups[item.category].append(item)  # No KeyError

# 3. Counter - Specialized for counting
from collections import Counter

counts = Counter([1, 2, 2, 3, 3, 3])
# Counter({3: 3, 2: 2, 1: 1})

counts.most_common(2)  # [(3, 3), (2, 2)]
counts.update([1, 1, 4])  # Add more counts

# 4. OrderedDict - Maintains insertion order (before Python 3.7)
from collections import OrderedDict

# Note: Regular dict maintains order in Python 3.7+
# OrderedDict still useful for:
# - Explicit ordering semantics
# - move_to_end() method
# - Equality checks that consider order

ordered = OrderedDict()
ordered['first'] = 1
ordered['second'] = 2
ordered.move_to_end('first')  # Move to end

# 5. ChainMap - Combine multiple dicts
from collections import ChainMap

defaults = {'color': 'blue', 'size': 'medium'}
user_settings = {'color': 'red'}
config = ChainMap(user_settings, defaults)
# config['color'] → 'red' (from user_settings)
# config['size'] → 'medium' (from defaults)
```

### Performance Considerations

**Python's dict implementation**:
- Uses open addressing with randomized probing (not simple linear)
- Maintains 2/3 load factor (resizes at ~67% full)
- Keys must be hashable (immutable: str, int, tuple, frozenset)
- Optimized for memory and speed in CPython

**When to use each variant**:
```python
# Use dict when:
# - Standard key-value mapping
# - Keys exist before access

# Use defaultdict when:
# - Building collections (lists, sets, counts)
# - Avoiding KeyError checks

# Use Counter when:
# - Counting frequencies
# - Need most_common() or arithmetic operations

# Use set when:
# - Only need keys (no values)
# - Membership testing
# - Set operations (union, intersection, difference)

# Example: Find common elements
set1 = {1, 2, 3}
set2 = {2, 3, 4}
common = set1 & set2  # {2, 3}
```

## Common Problems

### Two Sum
```python
def two_sum(arr, target):
    seen = {}
    for num in arr:
        if target - num in seen:
            return [seen[target - num], arr.index(num)]
        seen[num] = arr.index(num)
    return []
```

### Duplicate Detection
```python
def has_duplicates(arr):
    return len(arr) != len(set(arr))
```

### Group Anagrams
```python
def group_anagrams(strs):
    """Group strings that are anagrams: O(n*k) where k=avg string length"""
    anagrams = {}
    for s in strs:
        # Sort string as key (or use char count tuple)
        key = ''.join(sorted(s))
        if key not in anagrams:
            anagrams[key] = []
        anagrams[key].append(s)
    return list(anagrams.values())

# Example: ["eat", "tea", "tan", "ate", "nat", "bat"]
# Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

### Longest Consecutive Sequence
```python
def longest_consecutive(nums):
    """Find longest consecutive sequence: O(n)"""
    if not nums:
        return 0

    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Only start counting if num is start of sequence
        if num - 1 not in num_set:
            current = num
            streak = 1

            while current + 1 in num_set:
                current += 1
                streak += 1

            longest = max(longest, streak)

    return longest

# Example: [100, 4, 200, 1, 3, 2] → 4 (sequence: 1,2,3,4)
```

### Subarray Sum Equals K
```python
def subarray_sum(nums, k):
    """Count subarrays with sum = k: O(n)"""
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Handle subarrays starting at index 0

    for num in nums:
        prefix_sum += num

        # If (prefix_sum - k) exists, we found subarray(s)
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]

        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

    return count

# Example: [1, 2, 3], k=3 → 2 (subarrays: [3], [1,2])
```

### First Non-Repeating Character
```python
def first_unique_char(s):
    """Find first non-repeating character index: O(n)"""
    char_count = {}

    # Count frequencies
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    # Find first with count 1
    for i, char in enumerate(s):
        if char_count[char] == 1:
            return i

    return -1

# Example: "leetcode" → 0 (char 'l')
# Example: "loveleetcode" → 2 (char 'v')
```

### Valid Sudoku
```python
def is_valid_sudoku(board):
    """Check if Sudoku board is valid (no duplicates in rows/cols/boxes): O(1)
    Board is 9x9, so technically O(81) = O(1)
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            num = board[r][c]
            if num == '.':
                continue

            # Calculate which 3x3 box this cell belongs to
            box_idx = (r // 3) * 3 + (c // 3)

            # Check if number already exists in row, col, or box
            if num in rows[r] or num in cols[c] or num in boxes[box_idx]:
                return False

            rows[r].add(num)
            cols[c].add(num)
            boxes[box_idx].add(num)

    return True

# Example:
# board = [
#   ["5","3",".",".","7",".",".",".","."],
#   ["6",".",".","1","9","5",".",".","."],
#   ...
# ] → True/False
```

### Top K Frequent Elements
```python
def top_k_frequent(nums, k):
    """Find k most frequent elements: O(n log k) using heap, O(n) using bucket sort"""
    from collections import Counter
    import heapq

    # Method 1: Using heap - O(n log k)
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

    # Method 2: Bucket sort - O(n) but more memory
    count = Counter(nums)
    # Create buckets where index = frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    # Collect top k from highest frequency buckets
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[freq])
        if len(result) >= k:
            return result[:k]

# Example: [1,1,1,2,2,3], k=2 → [1,2]
```

### LRU Cache
```python
class LRUCache:
    """Least Recently Used Cache with O(1) get and put
    Combines hash table + doubly linked list
    """
    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        # Dummy head and tail for easier manipulation
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Remove node from linked list"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        """Add node right after head (most recently used)"""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        """Get value and mark as recently used: O(1)"""
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node.val

    def put(self, key, value):
        """Put key-value, evict LRU if over capacity: O(1)"""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_front(node)
        else:
            # Add new
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)

            if len(self.cache) > self.capacity:
                # Remove LRU (node before tail)
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]

# Usage:
# cache = LRUCache(2)
# cache.put(1, 1)
# cache.put(2, 2)
# cache.get(1)      # returns 1
# cache.put(3, 3)   # evicts key 2
# cache.get(2)      # returns -1 (not found)
```

### Isomorphic Strings
```python
def is_isomorphic(s, t):
    """Check if two strings are isomorphic (bijective mapping): O(n)"""
    if len(s) != len(t):
        return False

    s_to_t = {}
    t_to_s = {}

    for c1, c2 in zip(s, t):
        # Check s -> t mapping
        if c1 in s_to_t:
            if s_to_t[c1] != c2:
                return False
        else:
            s_to_t[c1] = c2

        # Check t -> s mapping (must be bijective)
        if c2 in t_to_s:
            if t_to_s[c2] != c1:
                return False
        else:
            t_to_s[c2] = c1

    return True

# Example: "egg", "add" → True (e->a, g->d)
# Example: "foo", "bar" → False (o maps to both o and a)
```

### Intersection of Two Arrays
```python
def intersection(nums1, nums2):
    """Find unique common elements: O(n + m)"""
    # Method 1: Using sets - simple and clean
    return list(set(nums1) & set(nums2))

    # Method 2: Using hash table - more control
    seen = set(nums1)
    result = set()
    for num in nums2:
        if num in seen:
            result.add(num)
    return list(result)

def intersect_with_counts(nums1, nums2):
    """Find intersection with duplicates (each element appears min(count1, count2) times)"""
    from collections import Counter

    count1 = Counter(nums1)
    result = []

    for num in nums2:
        if count1[num] > 0:
            result.append(num)
            count1[num] -= 1

    return result

# Example: [1,2,2,1], [2,2] → [2,2]
# Example: [4,9,5], [9,4,9,8,4] → [4,9] or [9,4]
```

### Word Pattern
```python
def word_pattern(pattern, s):
    """Check if string follows a pattern (bijective mapping): O(n)"""
    words = s.split()

    if len(pattern) != len(words):
        return False

    char_to_word = {}
    word_to_char = {}

    for char, word in zip(pattern, words):
        # Check pattern -> word mapping
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word

        # Check word -> pattern mapping (must be bijective)
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char

    return True

# Example: pattern="abba", s="dog cat cat dog" → True
# Example: pattern="abba", s="dog cat cat fish" → False
# Example: pattern="aaaa", s="dog dog dog dog" → True
```

## ELI10

Think of hash tables like library catalogs:
- **Hash function** = catalog system (tells you which shelf)
- **Index** = shelf number
- **Value** = book

Instead of searching every shelf, the system instantly tells you which one!

## Further Resources

- [LeetCode Hash Table](https://leetcode.com/tag/hash-table/)
- [Hash Table Wikipedia](https://en.wikipedia.org/wiki/Hash_table)
