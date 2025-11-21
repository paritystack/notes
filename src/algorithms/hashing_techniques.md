# Hashing Techniques

## Overview
Hashing is a fundamental technique for achieving O(1) average-case time complexity for search, insert, and delete operations. This guide covers advanced hashing techniques used in competitive programming and system design.

## 1. Hash Function Design

### Properties of Good Hash Functions

1. **Deterministic**: Same input always produces same hash
2. **Uniform Distribution**: Evenly distributes keys across hash table
3. **Efficient**: Fast to compute
4. **Avalanche Effect**: Small input changes cause large hash changes

### Basic Hash Functions

```python
def simple_hash(key, table_size):
    """Basic modulo hash function"""
    return key % table_size

def string_hash(s, table_size):
    """Simple string hash using polynomial rolling"""
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % table_size
    return hash_value

def djb2_hash(s):
    """DJB2 hash algorithm - popular for strings"""
    hash_value = 5381
    for char in s:
        hash_value = ((hash_value << 5) + hash_value) + ord(char)
        hash_value &= 0xFFFFFFFF  # Keep 32-bit
    return hash_value

def fnv_hash(s):
    """FNV-1a hash - good distribution"""
    FNV_prime = 0x01000193
    FNV_offset = 0x811c9dc5

    hash_value = FNV_offset
    for char in s:
        hash_value ^= ord(char)
        hash_value = (hash_value * FNV_prime) & 0xFFFFFFFF

    return hash_value
```

### Universal Hashing

```python
import random

class UniversalHash:
    """
    Universal hashing family
    h(k) = ((a*k + b) mod p) mod m
    where p is prime > universe size
    """
    def __init__(self, table_size, prime=2**31 - 1):
        self.m = table_size
        self.p = prime
        self.a = random.randint(1, prime - 1)
        self.b = random.randint(0, prime - 1)

    def hash(self, key):
        return ((self.a * key + self.b) % self.p) % self.m

# Usage
hasher = UniversalHash(1000)
hash_value = hasher.hash(12345)
```

## 2. Rolling Hash

Rolling hash allows efficient computation of hash values for sliding windows.

### Rabin-Karp Algorithm

Used for pattern matching.

```python
class RollingHash:
    """
    Polynomial rolling hash for string matching
    hash(s) = s[0]*p^(n-1) + s[1]*p^(n-2) + ... + s[n-1]*p^0 (mod m)
    """
    def __init__(self, base=31, mod=10**9 + 7):
        self.base = base
        self.mod = mod

    def compute_hash(self, s):
        """Compute hash of string s"""
        hash_value = 0
        for char in s:
            hash_value = (hash_value * self.base + ord(char)) % self.mod
        return hash_value

    def compute_hash_with_powers(self, s):
        """Compute hash and power values for rolling"""
        n = len(s)
        hash_value = 0
        power = 1

        for i in range(n):
            hash_value = (hash_value * self.base + ord(s[i])) % self.mod
            if i < n - 1:
                power = (power * self.base) % self.mod

        return hash_value, power

    def roll_hash(self, old_hash, old_char, new_char, power):
        """
        Remove old_char from left, add new_char to right
        old_hash: current hash value
        old_char: character being removed
        new_char: character being added
        power: base^(window_size-1) mod m
        """
        # Remove leftmost character
        old_hash = (old_hash - ord(old_char) * power) % self.mod
        # Shift and add new character
        old_hash = (old_hash * self.base + ord(new_char)) % self.mod

        return old_hash

# Example: Pattern matching
def rabin_karp(text, pattern):
    """Find all occurrences of pattern in text"""
    if not text or not pattern or len(pattern) > len(text):
        return []

    rh = RollingHash()
    m, n = len(pattern), len(text)

    # Compute pattern hash and power
    pattern_hash, power = rh.compute_hash_with_powers(pattern)

    # Compute initial window hash
    window_hash = rh.compute_hash(text[:m])

    result = []
    if window_hash == pattern_hash and text[:m] == pattern:
        result.append(0)

    # Roll through text
    for i in range(1, n - m + 1):
        window_hash = rh.roll_hash(
            window_hash,
            text[i - 1],
            text[i + m - 1],
            power
        )

        if window_hash == pattern_hash and text[i:i + m] == pattern:
            result.append(i)

    return result

# Example
text = "ababcabcab"
pattern = "abc"
print(rabin_karp(text, pattern))
# Output: [2, 5]
```

### Double Hashing for Collision Reduction

```python
class DoubleRollingHash:
    """Use two hash functions to reduce false positives"""
    def __init__(self):
        self.hash1 = RollingHash(base=31, mod=10**9 + 7)
        self.hash2 = RollingHash(base=37, mod=10**9 + 9)

    def compute_hash(self, s):
        """Return tuple of two hash values"""
        return (self.hash1.compute_hash(s), self.hash2.compute_hash(s))

    def matches(self, hash1, hash2):
        """Check if two hashes match"""
        return hash1[0] == hash2[0] and hash1[1] == hash2[1]
```

### Longest Duplicate Substring

```python
def longest_duplicate_substring(s):
    """
    LeetCode 1044: Longest Duplicate Substring
    Binary search + rolling hash
    """
    def search(length):
        """Check if there's duplicate substring of given length"""
        seen = set()
        rh = RollingHash()

        # Compute hash for first window
        hash_val, power = rh.compute_hash_with_powers(s[:length])
        seen.add(hash_val)

        # Roll through string
        for i in range(1, len(s) - length + 1):
            hash_val = rh.roll_hash(hash_val, s[i-1], s[i+length-1], power)

            if hash_val in seen:
                return i  # Return start index of duplicate

            seen.add(hash_val)

        return -1

    # Binary search on length
    left, right = 1, len(s)
    result_idx = 0
    result_len = 0

    while left <= right:
        mid = (left + right) // 2
        idx = search(mid)

        if idx != -1:
            result_idx = idx
            result_len = mid
            left = mid + 1
        else:
            right = mid - 1

    return s[result_idx:result_idx + result_len] if result_len > 0 else ""
```

## 3. Collision Resolution Strategies

### Separate Chaining

Each bucket contains a linked list of entries.

```python
class HashTableChaining:
    """Hash table with separate chaining"""
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        """Insert or update key-value pair"""
        bucket = self._hash(key)

        # Check if key exists, update value
        for i, (k, v) in enumerate(self.table[bucket]):
            if k == key:
                self.table[bucket][i] = (key, value)
                return

        # Add new entry
        self.table[bucket].append((key, value))

    def get(self, key):
        """Retrieve value for key"""
        bucket = self._hash(key)

        for k, v in self.table[bucket]:
            if k == key:
                return v

        raise KeyError(key)

    def delete(self, key):
        """Remove key-value pair"""
        bucket = self._hash(key)

        for i, (k, v) in enumerate(self.table[bucket]):
            if k == key:
                del self.table[bucket][i]
                return

        raise KeyError(key)
```

### Open Addressing - Linear Probing

```python
class HashTableLinearProbing:
    """Hash table with linear probing"""
    def __init__(self, size=100):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def _probe(self, index):
        """Linear probing: (index + 1) mod size"""
        return (index + 1) % self.size

    def insert(self, key, value):
        """Insert key-value pair"""
        if self.count >= self.size * 0.7:  # Load factor check
            self._resize()

        index = self._hash(key)

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return
            index = self._probe(index)

        self.keys[index] = key
        self.values[index] = value
        self.count += 1

    def get(self, key):
        """Retrieve value for key"""
        index = self._hash(key)

        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = self._probe(index)

        raise KeyError(key)

    def _resize(self):
        """Resize hash table when load factor is high"""
        old_keys = self.keys
        old_values = self.values

        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0

        for key, value in zip(old_keys, old_values):
            if key is not None:
                self.insert(key, value)
```

### Open Addressing - Quadratic Probing

```python
class HashTableQuadraticProbing:
    """Hash table with quadratic probing"""
    def __init__(self, size=100):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def _hash(self, key):
        return hash(key) % self.size

    def _probe(self, index, i):
        """Quadratic probing: (index + i^2) mod size"""
        return (index + i * i) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        i = 0

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return

            i += 1
            index = self._probe(self._hash(key), i)

            if i >= self.size:
                raise Exception("Hash table is full")

        self.keys[index] = key
        self.values[index] = value
```

### Double Hashing

```python
class HashTableDoubleHashing:
    """Hash table with double hashing"""
    def __init__(self, size=100):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def _hash1(self, key):
        return hash(key) % self.size

    def _hash2(self, key):
        """Second hash function (must not return 0)"""
        return 1 + (hash(key) % (self.size - 1))

    def _probe(self, index, i, key):
        """Double hashing: (h1(key) + i*h2(key)) mod size"""
        return (index + i * self._hash2(key)) % self.size

    def insert(self, key, value):
        index = self._hash1(key)
        i = 0

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return

            i += 1
            index = self._probe(self._hash1(key), i, key)

        self.keys[index] = key
        self.values[index] = value
```

## 4. Hash-Based Problem Patterns

### Frequency Counting

```python
from collections import Counter, defaultdict

def find_frequency_patterns(arr):
    """Common frequency-based operations"""
    # Method 1: Using Counter
    freq = Counter(arr)

    # Most common elements
    most_common = freq.most_common(3)

    # Elements with frequency > k
    k = 2
    frequent = [x for x, count in freq.items() if count > k]

    # Method 2: Using defaultdict
    freq_dict = defaultdict(int)
    for x in arr:
        freq_dict[x] += 1

    return freq, most_common, frequent
```

### Two Sum Pattern

```python
def two_sum(nums, target):
    """LeetCode 1: Two Sum"""
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return []

def two_sum_all_pairs(nums, target):
    """Find all pairs that sum to target"""
    seen = set()
    pairs = set()

    for num in nums:
        complement = target - num

        if complement in seen:
            pairs.add(tuple(sorted([num, complement])))

        seen.add(num)

    return list(pairs)
```

### Group Anagrams

```python
def group_anagrams(strs):
    """LeetCode 49: Group Anagrams"""
    groups = defaultdict(list)

    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())

# Alternative: Use character count as key
def group_anagrams_optimized(strs):
    groups = defaultdict(list)

    for s in strs:
        # Count characters
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1

        # Use tuple as key (lists aren't hashable)
        key = tuple(count)
        groups[key].append(s)

    return list(groups.values())
```

### Longest Consecutive Sequence

```python
def longest_consecutive(nums):
    """
    LeetCode 128: Longest Consecutive Sequence
    O(n) time using hash set
    """
    if not nums:
        return 0

    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # Only start counting from sequence start
        if num - 1 not in num_set:
            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            max_length = max(max_length, length)

    return max_length
```

### Subarray Sum Equals K

```python
def subarray_sum(nums, k):
    """
    LeetCode 560: Subarray Sum Equals K
    Use prefix sum + hash map
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # prefix_sum -> frequency

    for num in nums:
        prefix_sum += num

        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]

        # Update frequency
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

    return count
```

### Find Duplicate Subtrees

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_duplicate_subtrees(root):
    """
    LeetCode 652: Find Duplicate Subtrees
    Serialize subtrees and use hash map
    """
    def serialize(node):
        if not node:
            return "#"

        # Serialize current subtree
        serial = f"{node.val},{serialize(node.left)},{serialize(node.right)}"

        # Track seen subtrees
        subtrees[serial].append(node)

        return serial

    from collections import defaultdict
    subtrees = defaultdict(list)
    serialize(root)

    # Return roots of duplicate subtrees
    return [nodes[0] for nodes in subtrees.values() if len(nodes) > 1]
```

## 5. Bloom Filters

Probabilistic data structure for membership testing with space efficiency.

**Properties**:
- No false negatives (if says "not present", definitely not present)
- Possible false positives (if says "present", might not be present)
- Space-efficient

```python
import hashlib

class BloomFilter:
    """
    Simple Bloom Filter implementation
    """
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size

    def _hashes(self, item):
        """Generate multiple hash values"""
        hashes = []
        for i in range(self.num_hashes):
            # Use different seeds for each hash function
            h = hashlib.md5(f"{item}{i}".encode())
            hash_val = int(h.hexdigest(), 16) % self.size
            hashes.append(hash_val)
        return hashes

    def add(self, item):
        """Add item to bloom filter"""
        for hash_val in self._hashes(item):
            self.bit_array[hash_val] = True

    def might_contain(self, item):
        """Check if item might be in the set"""
        return all(self.bit_array[h] for h in self._hashes(item))

    def definitely_not_contains(self, item):
        """Check if item is definitely not in the set"""
        return not self.might_contain(item)

# Usage
bf = BloomFilter(size=1000, num_hashes=3)

# Add items
bf.add("apple")
bf.add("banana")
bf.add("orange")

# Check membership
print(bf.might_contain("apple"))    # True
print(bf.might_contain("grape"))    # False (probably)
print(bf.definitely_not_contains("grape"))  # True (definitely)
```

### Counting Bloom Filter

```python
class CountingBloomFilter:
    """
    Bloom filter that supports deletions
    Uses counters instead of bits
    """
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.counters = [0] * size

    def _hashes(self, item):
        hashes = []
        for i in range(self.num_hashes):
            h = hashlib.md5(f"{item}{i}".encode())
            hash_val = int(h.hexdigest(), 16) % self.size
            hashes.append(hash_val)
        return hashes

    def add(self, item):
        """Increment counters for item"""
        for hash_val in self._hashes(item):
            self.counters[hash_val] += 1

    def remove(self, item):
        """Decrement counters for item"""
        for hash_val in self._hashes(item):
            if self.counters[hash_val] > 0:
                self.counters[hash_val] -= 1

    def might_contain(self, item):
        """Check if item might be in the set"""
        return all(self.counters[h] > 0 for h in self._hashes(item))
```

### Optimal Bloom Filter Parameters

```python
import math

def optimal_bloom_parameters(n, p):
    """
    Calculate optimal bloom filter parameters
    n: expected number of elements
    p: desired false positive rate
    """
    # Optimal size
    m = -((n * math.log(p)) / (math.log(2) ** 2))

    # Optimal number of hash functions
    k = (m / n) * math.log(2)

    return int(math.ceil(m)), int(math.ceil(k))

# Example: 1000 elements, 1% false positive rate
size, num_hashes = optimal_bloom_parameters(1000, 0.01)
print(f"Size: {size}, Hash functions: {num_hashes}")
# Size: 9586, Hash functions: 7
```

## 6. Advanced Techniques

### Consistent Hashing

Used in distributed systems.

```python
import hashlib
import bisect

class ConsistentHash:
    """
    Consistent hashing for distributed systems
    """
    def __init__(self, nodes=None, replicas=3):
        self.replicas = replicas
        self.ring = {}  # hash -> node
        self.sorted_keys = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        """Hash function for ring"""
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

    def add_node(self, node):
        """Add node to hash ring"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
            bisect.insort(self.sorted_keys, hash_val)

    def remove_node(self, node):
        """Remove node from hash ring"""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            del self.ring[hash_val]
            self.sorted_keys.remove(hash_val)

    def get_node(self, key):
        """Get node responsible for key"""
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # Find first node clockwise from hash
        idx = bisect.bisect(self.sorted_keys, hash_val)
        idx = idx % len(self.sorted_keys)

        return self.ring[self.sorted_keys[idx]]

# Usage
ch = ConsistentHash(nodes=['node1', 'node2', 'node3'])
print(ch.get_node('key1'))  # node2
print(ch.get_node('key2'))  # node1
```

### Cuckoo Hashing

```python
class CuckooHash:
    """
    Cuckoo hashing with two hash functions
    """
    def __init__(self, size=100):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.max_iterations = 100

    def _hash1(self, key):
        return hash(key) % self.size

    def _hash2(self, key):
        return (hash(key) // self.size) % self.size

    def insert(self, key, value):
        """Insert key-value pair"""
        for _ in range(self.max_iterations):
            # Try table1
            idx1 = self._hash1(key)
            if self.table1[idx1] is None:
                self.table1[idx1] = (key, value)
                return True

            # Evict from table1
            old_key, old_value = self.table1[idx1]
            self.table1[idx1] = (key, value)

            # Try to insert evicted item into table2
            idx2 = self._hash2(old_key)
            if self.table2[idx2] is None:
                self.table2[idx2] = (old_key, old_value)
                return True

            # Evict from table2
            key, value = old_key, old_value
            old_key, old_value = self.table2[idx2]
            self.table2[idx2] = (key, value)
            key, value = old_key, old_value

        # Rehashing needed
        return False

    def search(self, key):
        """Search for key"""
        idx1 = self._hash1(key)
        if self.table1[idx1] and self.table1[idx1][0] == key:
            return self.table1[idx1][1]

        idx2 = self._hash2(key)
        if self.table2[idx2] and self.table2[idx2][0] == key:
            return self.table2[idx2][1]

        raise KeyError(key)
```

## Performance Comparison

| Technique | Average Case | Worst Case | Space | Use Case |
|-----------|-------------|------------|-------|----------|
| Chaining | O(1) | O(n) | O(n+m) | General purpose |
| Linear Probing | O(1) | O(n) | O(n) | Cache-friendly |
| Quadratic Probing | O(1) | O(n) | O(n) | Reduce clustering |
| Double Hashing | O(1) | O(n) | O(n) | Better distribution |
| Cuckoo Hashing | O(1) | O(1) | O(n) | Guaranteed O(1) lookup |
| Bloom Filter | O(k) | O(k) | O(m) | Space-efficient membership |

## Common Pitfalls

1. **Hash collisions** - Always handle collisions properly
2. **Load factor** - Resize when table is too full
3. **Hash function quality** - Poor hash functions cause clustering
4. **Rolling hash overflow** - Use proper modulo arithmetic
5. **Bloom filter tuning** - Choose size and hash count carefully

## Practice Problems

### Easy
- LeetCode 1: Two Sum
- LeetCode 217: Contains Duplicate
- LeetCode 242: Valid Anagram

### Medium
- LeetCode 49: Group Anagrams
- LeetCode 128: Longest Consecutive Sequence
- LeetCode 560: Subarray Sum Equals K
- LeetCode 1044: Longest Duplicate Substring

### Hard
- LeetCode 149: Max Points on a Line
- LeetCode 652: Find Duplicate Subtrees
- LeetCode 1044: Longest Duplicate Substring

## Resources

- [Hash Table - Wikipedia](https://en.wikipedia.org/wiki/Hash_table)
- [Bloom Filter Calculator](https://hur.st/bloomfilter/)
- [Consistent Hashing](https://www.toptal.com/big-data/consistent-hashing)
- [CP-Algorithms: Hashing](https://cp-algorithms.com/string/string-hashing.html)
