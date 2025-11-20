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

## ELI10

Think of hash tables like library catalogs:
- **Hash function** = catalog system (tells you which shelf)
- **Index** = shelf number
- **Value** = book

Instead of searching every shelf, the system instantly tells you which one!

## Further Resources

- [LeetCode Hash Table](https://leetcode.com/tag/hash-table/)
- [Hash Table Wikipedia](https://en.wikipedia.org/wiki/Hash_table)
