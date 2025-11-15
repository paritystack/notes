# Bloom Filter

## Overview

A **Bloom filter** is a space-efficient probabilistic data structure designed to test whether an element is a member of a set. Invented by Burton Howard Bloom in 1970, it trades perfect accuracy for significant space savings, making it invaluable in scenarios where memory is constrained and occasional false positives are acceptable.

**Key Characteristics:**
- **Space-efficient**: Uses significantly less memory than traditional hash tables or sets
- **Probabilistic**: May return false positives but never false negatives
- **Fast operations**: Constant time O(k) for insertions and queries
- **No deletions**: Standard Bloom filters don't support element removal
- **No element retrieval**: Can only test membership, not retrieve stored values

## The Problem It Solves

Consider scenarios where you need to check membership in a massive set:
- Does this email address exist in our 10 million user database?
- Has this URL been visited before (out of billions)?
- Is this word misspelled (checking against a 500k word dictionary)?

Traditional approaches (hash tables, binary search trees) require O(n) space where n is the number of elements. Bloom filters can represent the same set using a fraction of the space, with a controllable error rate.

## How It Works

### Data Structure

A Bloom filter consists of:
1. **Bit array** of size `m` (all bits initially set to 0)
2. **k independent hash functions** (h_1, h_2, ..., h_k), each mapping elements to positions in the bit array [0, m-1]

### Operations

#### Insert(x)
To add an element x to the set:
1. Compute k hash values: h_1(x), h_2(x), ..., h_k(x)
2. Set bits at all k positions to 1

```
Insert "apple":
h_1("apple") = 3  -> set bit[3] = 1
h_2("apple") = 7  -> set bit[7] = 1
h_3("apple") = 12 -> set bit[12] = 1
```

#### Query(x)
To test if element x is in the set:
1. Compute k hash values: h_1(x), h_2(x), ..., h_k(x)
2. Check if ALL k bit positions are set to 1
   - If all are 1: **possibly in set** (might be false positive)
   - If any is 0: **definitely not in set** (guaranteed correct)

```
Query "apple":
Check bit[3], bit[7], bit[12]
All are 1 -> "possibly in set"

Query "banana":
h_1("banana") = 3  -> bit[3] = 1 YES
h_2("banana") = 5  -> bit[5] = 0 NO
Result: "definitely not in set"
```

### Visual Example

```
Initial state (m=16 bits):
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

After Insert("cat") with h_1=2, h_2=7, h_3=13:
[0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0]

After Insert("dog") with h_1=2, h_2=9, h_3=14:
[0,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0]
     ^           ^     ^  ^
  overlap       new   old new

Query("cat"): Check positions 2,7,13 -> all 1 -> "possibly in set" YES
Query("bird"): h_1=5, h_2=9, h_3=11 -> position 5 is 0 -> "not in set" YES
Query("fox"): h_1=2, h_2=7, h_3=9 -> all 1 -> "possibly in set" NO (FALSE POSITIVE!)
```

## Mathematical Analysis

### False Positive Probability

After inserting n elements into a Bloom filter of size m bits using k hash functions:

**Probability a specific bit is still 0:**
```
p_0 = (1 - 1/m)^(kn)
```

**Probability a specific bit is 1:**
```
p_1 = 1 - (1 - 1/m)^(kn) ~= 1 - e^(-kn/m)
```

**False positive probability (all k bits are 1 by chance):**
```
P(false positive) = p_1^k = (1 - e^(-kn/m))^k
```

### Optimal Number of Hash Functions

To minimize false positive rate for given m and n:

```
k_optimal = (m/n) * ln(2) ~= 0.693 * (m/n)
```

With optimal k:
```
P(false positive) ~= (1/2)^k = 0.6185^(m/n)
```

### Optimal Bit Array Size

For desired false positive probability p and n elements:

```
m = -n * ln(p) / (ln(2))^2
m ~= -1.44 * n * log_2(p)
```

### Example Calculation

**Scenario**: Store 1 million elements with 1% false positive rate

```
n = 1,000,000
p = 0.01

m = -1,000,000 * ln(0.01) / (ln(2))^2
m ~= 9,585,058 bits ~= 1.14 MB

k = 0.693 * (9,585,058 / 1,000,000)
k ~= 6.64 ~= 7 hash functions
```

Compare to hash table: ~20 MB (assuming 20 bytes per entry)
**Space savings: ~95%**

## Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Insert    | O(k)           | O(m) bits total  |
| Query     | O(k)           | O(m) bits total  |
| Delete    | N/A*           | -                |

*Standard Bloom filters don't support deletion

Where:
- k = number of hash functions (typically small constant, 3-10)
- m = bit array size
- n = number of inserted elements

**Space efficiency**: m = O(n log_2(1/p)) bits, where p is desired false positive rate

## Properties

### Guarantees

1. **No false negatives**: If Query(x) returns "not in set", x is definitely not in the set
2. **Possible false positives**: If Query(x) returns "in set", x might not actually be in the set
3. **Monotonicity**: False positive rate only increases as more elements are added
4. **Union-friendly**: Two Bloom filters with same m and k can be combined with bitwise OR

### Limitations

1. **Cannot remove elements**: Setting bits to 0 could affect other elements (collision)
2. **Cannot enumerate elements**: Can't list what's in the filter
3. **Cannot count elements**: Can estimate, but not get exact count
4. **Fixed capacity**: Performance degrades if you exceed the designed capacity

## Variations and Extensions

### 1. Counting Bloom Filter

**Problem**: Standard Bloom filters can't delete elements

**Solution**: Replace each bit with a counter (typically 3-4 bits)
- Insert: increment counters
- Delete: decrement counters
- Query: check if all counters > 0

**Trade-off**: Uses 3-4x more space but supports deletions

```
Standard:  [1, 0, 1, 1, 0]
Counting:  [3, 0, 2, 1, 0]  (can decrement safely)
```

### 2. Scalable Bloom Filter

**Problem**: Fixed capacity - performance degrades with more elements

**Solution**: Chain of multiple Bloom filters with increasing sizes
- When one filter reaches capacity, create a new larger one
- Query checks all filters in sequence

**Trade-off**: Maintains target false positive rate, slightly slower queries

### 3. Cuckoo Filter

**Improvements over Bloom**:
- Supports deletions
- Better space efficiency for low false positive rates (< 3%)
- Better lookup performance

**How**: Uses cuckoo hashing with buckets storing fingerprints

### 4. Quotient Filter

**Advantages**:
- Supports deletions
- Better cache locality
- Supports merging and resizing

**How**: Uses quotienting technique with clustering

### 5. Blocked Bloom Filter

**Optimization**: Partition bit array into cache-line-sized blocks
- Better CPU cache utilization
- Each element hashes to single block

### 6. Compressed Bloom Filter

**Use case**: Network transmission
- Compress the bit array
- Trade computation for bandwidth

## Implementation

### Python Implementation

```python
import math
import mmh3  # MurmurHash3 library
from bitarray import bitarray

class BloomFilter:
    def __init__(self, expected_elements, false_positive_rate):
        """
        Initialize Bloom filter with optimal parameters

        Args:
            expected_elements (int): Expected number of elements
            false_positive_rate (float): Desired false positive probability
        """
        # Calculate optimal bit array size
        self.size = self._optimal_size(expected_elements, false_positive_rate)

        # Calculate optimal number of hash functions
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)

        # Initialize bit array
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

        self.elements_added = 0

    def _optimal_size(self, n, p):
        """Calculate optimal bit array size"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _optimal_hash_count(self, m, n):
        """Calculate optimal number of hash functions"""
        k = (m / n) * math.log(2)
        return int(k)

    def _hash(self, item, seed):
        """Generate hash for item with given seed"""
        return mmh3.hash(item, seed) % self.size

    def add(self, item):
        """Add item to the Bloom filter"""
        for i in range(self.hash_count):
            position = self._hash(item, i)
            self.bit_array[position] = 1
        self.elements_added += 1

    def contains(self, item):
        """Check if item might be in the set"""
        for i in range(self.hash_count):
            position = self._hash(item, i)
            if self.bit_array[position] == 0:
                return False
        return True

    def current_false_positive_rate(self):
        """Calculate current false positive probability"""
        k = self.hash_count
        m = self.size
        n = self.elements_added
        return (1 - math.exp(-k * n / m)) ** k

# Usage example
bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)

# Add elements
words = ["apple", "banana", "cherry", "date"]
for word in words:
    bf.add(word)

# Query
print(bf.contains("apple"))   # True (definitely added)
print(bf.contains("banana"))  # True (definitely added)
print(bf.contains("grape"))   # False or True (if false positive)

print(f"Current FP rate: {bf.current_false_positive_rate():.4f}")
```

### JavaScript Implementation

```javascript
class BloomFilter {
    constructor(expectedElements, falsePositiveRate) {
        this.size = this.optimalSize(expectedElements, falsePositiveRate);
        this.hashCount = this.optimalHashCount(this.size, expectedElements);
        this.bitArray = new Uint8Array(Math.ceil(this.size / 8));
        this.elementsAdded = 0;
    }

    optimalSize(n, p) {
        const m = -(n * Math.log(p)) / (Math.log(2) ** 2);
        return Math.ceil(m);
    }

    optimalHashCount(m, n) {
        const k = (m / n) * Math.log(2);
        return Math.ceil(k);
    }

    hash(item, seed) {
        // Simple hash function (use better hash in production)
        let hash = seed;
        for (let i = 0; i < item.length; i++) {
            hash = ((hash << 5) - hash) + item.charCodeAt(i);
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash) % this.size;
    }

    setBit(position) {
        const byteIndex = Math.floor(position / 8);
        const bitIndex = position % 8;
        this.bitArray[byteIndex] |= (1 << bitIndex);
    }

    getBit(position) {
        const byteIndex = Math.floor(position / 8);
        const bitIndex = position % 8;
        return (this.bitArray[byteIndex] & (1 << bitIndex)) !== 0;
    }

    add(item) {
        for (let i = 0; i < this.hashCount; i++) {
            const position = this.hash(item, i);
            this.setBit(position);
        }
        this.elementsAdded++;
    }

    contains(item) {
        for (let i = 0; i < this.hashCount; i++) {
            const position = this.hash(item, i);
            if (!this.getBit(position)) {
                return false;
            }
        }
        return true;
    }

    currentFalsePositiveRate() {
        const k = this.hashCount;
        const m = this.size;
        const n = this.elementsAdded;
        return Math.pow(1 - Math.exp(-k * n / m), k);
    }
}

// Usage
const bf = new BloomFilter(10000, 0.01);
bf.add("user@example.com");
console.log(bf.contains("user@example.com"));  // true
console.log(bf.contains("other@example.com")); // likely false
```

## Real-World Applications

### 1. Database Systems

**Problem**: Avoid expensive disk reads for non-existent keys

**Solution**: Google Bigtable, Apache Cassandra, LevelDB
- Keep Bloom filter in memory for each SSTable
- Query filter before disk read
- If filter says "not present", skip disk I/O
- Typical savings: 80-90% reduction in disk reads

```
Query for key "user:12345":
1. Check Bloom filter (in memory) -> "not present"
2. Skip disk read entirely
3. Return "key not found"

Savings: ~10ms disk seek
```

### 2. Web Caching (Squid, Varnish)

**Use**: Track cached URLs without storing full URLs in memory

```
Before fetching remote page:
- Check Bloom filter for URL
- If "not in cache", fetch from origin
- If "possibly in cache", check cache (might be false positive)
```

### 3. Chrome Browser - Malicious URL Detection

**Implementation**:
- Local Bloom filter contains millions of known malicious URLs
- Before visiting site, check local filter
- If match, query Google Safe Browsing API
- Reduces API calls by >99%

### 4. Bitcoin - SPV Clients

**Use**: Lightweight clients filter transactions

```
Client creates Bloom filter with its addresses
Sends filter to full node
Full node returns only matching transactions
Reduces bandwidth by 1000x
```

### 5. Spell Checkers

**Classic use case**:
- Dictionary of 500k words -> ~1 MB Bloom filter
- Quick check if word exists
- If "not present", definitely misspelled
- If "present", verify with full dictionary

### 6. Network Routers

**Application**: Packet filtering, DDoS protection
- Track IP addresses sending traffic
- Detect distributed attacks
- High-speed filtering (millions of packets/second)

### 7. Distributed Systems - Eventual Consistency

**Example**: Apache Cassandra anti-entropy
- Each node maintains Bloom filter of its keys
- During repair, compare filters
- Only sync differences
- Reduces network traffic significantly

### 8. Akamai CDN

**Use**: Efficiently track cached content across global network
- Each edge server maintains Bloom filter
- Coordinate cache invalidation
- Minimize inter-server communication

## Advantages

1. **Extreme space efficiency**: 10-20x smaller than hash tables
2. **Constant-time operations**: O(k) regardless of set size
3. **Simple implementation**: Easy to code and understand
4. **Cache-friendly**: Small enough to fit in CPU cache
5. **Parallelizable**: Multiple hash functions can compute in parallel
6. **Set operations**: Easy union (bitwise OR) of filters
7. **Privacy-preserving**: Can't extract original elements

## Disadvantages

1. **False positives**: Cannot be eliminated, only controlled
2. **No deletions**: Standard version doesn't support removal
3. **Fixed capacity**: Optimal for predetermined size
4. **No element retrieval**: Can't list stored elements
5. **Cannot count**: Can't get exact element count
6. **Hash function dependency**: Quality affects performance
7. **Degrading performance**: FP rate increases with more elements

## Comparison with Other Data Structures

### vs. Hash Table

| Aspect | Bloom Filter | Hash Table |
|--------|--------------|------------|
| Space | O(n log(1/p)) bits | O(n) words |
| False positives | Yes (controlled) | No |
| False negatives | No | No |
| Lookup time | O(k) | O(1) average |
| Supports deletion | No* | Yes |
| Exact membership | No | Yes |
| Element retrieval | No | Yes |

*Except counting Bloom filters

### vs. Cuckoo Filter

| Aspect | Bloom Filter | Cuckoo Filter |
|--------|--------------|---------------|
| Deletion support | No | Yes |
| Space efficiency (p<3%) | Worse | Better |
| Lookup time | O(k) | O(1) typical |
| Implementation complexity | Simple | Moderate |
| Worst-case lookup | O(k) | O(1) |

### When to Use Each

**Use Bloom Filter when:**
- You never need to delete elements
- Extreme space efficiency is critical
- Simple implementation preferred
- False positive rate > 1%

**Use Hash Table when:**
- You need exact membership
- Element retrieval is required
- Deletions are frequent
- Memory is not constrained

**Use Cuckoo Filter when:**
- You need deletions
- False positive rate < 3%
- Better lookup performance needed

## Parameter Selection Guide

### Step 1: Determine Requirements

```
n = expected number of elements
p = acceptable false positive rate
```

### Step 2: Calculate Optimal Parameters

```python
import math

def calculate_bloom_parameters(n, p):
    """
    Calculate optimal Bloom filter parameters

    Args:
        n: expected number of elements
        p: desired false positive rate

    Returns:
        m: bit array size
        k: number of hash functions
        memory_mb: memory usage in MB
    """
    # Bit array size
    m = -(n * math.log(p)) / (math.log(2) ** 2)
    m = int(math.ceil(m))

    # Number of hash functions
    k = (m / n) * math.log(2)
    k = int(math.ceil(k))

    # Memory usage
    memory_mb = m / (8 * 1024 * 1024)

    return {
        'bit_array_size': m,
        'hash_functions': k,
        'memory_mb': round(memory_mb, 2),
        'bits_per_element': round(m/n, 2)
    }

# Examples
print(calculate_bloom_parameters(1_000_000, 0.01))
# {'bit_array_size': 9585059, 'hash_functions': 7,
#  'memory_mb': 1.14, 'bits_per_element': 9.59}

print(calculate_bloom_parameters(1_000_000, 0.001))
# {'bit_array_size': 14377589, 'hash_functions': 10,
#  'memory_mb': 1.71, 'bits_per_element': 14.38}
```

### Common Configurations

| Elements | FP Rate | Bits/Element | Hash Funcs | Memory (1M elements) |
|----------|---------|--------------|------------|---------------------|
| n        | 0.1     | 4.79         | 3          | 0.57 MB            |
| n        | 0.01    | 9.59         | 7          | 1.14 MB            |
| n        | 0.001   | 14.38        | 10         | 1.71 MB            |
| n        | 0.0001  | 19.17        | 13         | 2.28 MB            |

### Trade-off Analysis

```
Doubling the bit array size (m):
- Reduces false positive rate by ~50%
- Increases memory by 2x
- Requires more hash functions

Doubling hash functions (k):
- More computation per operation
- Better distribution
- Diminishing returns after optimal k
```

## Hash Function Selection

### Requirements

1. **Uniform distribution**: Hash values evenly distributed
2. **Independence**: Hash functions should be independent
3. **Fast computation**: Critical for performance
4. **Low collision rate**: Minimize hash collisions

### Recommended Hash Functions

**Production Quality:**
- MurmurHash3 (fast, good distribution)
- xxHash (extremely fast)
- CityHash (Google, optimized for strings)
- FNV-1a (simple, decent)

**Cryptographic (overkill for Bloom):**
- SHA-256 (slow but perfect distribution)
- Blake2 (fast cryptographic)

### Double Hashing Technique

Generate k hash functions from just 2:

```python
def get_hash(item, i, hash1, hash2, m):
    """
    Generate i-th hash using double hashing
    h_i(x) = (h1(x) + i * h2(x)) mod m
    """
    return (hash1 + i * hash2) % m

# Example
import mmh3

item = "example"
hash1 = mmh3.hash(item, seed=0) % m
hash2 = mmh3.hash(item, seed=1) % m

# Generate k hashes
hashes = [get_hash(item, i, hash1, hash2, m) for i in range(k)]
```

**Advantage**: Only compute 2 hashes instead of k

## Advanced Topics

### Estimating Number of Elements

Given a Bloom filter with X bits set to 1:

```
n_estimated = -(m/k) * ln(1 - X/m)

Where:
m = bit array size
k = number of hash functions
X = number of bits set to 1
```

### Union and Intersection

**Union** (elements in A OR B):
```python
union_filter = bloom_a | bloom_b  # bitwise OR
```

**Intersection** (approximate):
```python
intersection_filter = bloom_a & bloom_b  # bitwise AND
# Note: higher false positive rate
```

### Monitoring Filter Saturation

```python
def saturation_level(bloom_filter):
    """Calculate percentage of bits set to 1"""
    bits_set = sum(bloom_filter.bit_array)
    total_bits = bloom_filter.size
    return (bits_set / total_bits) * 100

# If saturation > 50%, consider resizing
# Optimal saturation ~= 50% when k is optimal
```

### Adaptive Bloom Filters

Dynamically adjust parameters based on actual usage:

```python
class AdaptiveBloomFilter:
    def add(self, item):
        super().add(item)

        # Check if saturation exceeds threshold
        if self.saturation_level() > 0.7:
            self.expand()

    def expand(self):
        # Create larger filter
        # Rehash all elements (requires storing or tracking them)
        pass
```

## Performance Tuning

### Memory Access Patterns

**Problem**: Random access to large bit array = cache misses

**Solutions:**
1. **Blocked Bloom Filter**: Hash to cache-line-sized blocks
2. **Partitioned Bloom Filter**: Multiple smaller filters
3. **Prefetching**: Issue prefetch instructions

### Parallelization

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_query(bloom_filter, items):
    with ThreadPoolExecutor() as executor:
        results = executor.map(bloom_filter.contains, items)
    return list(results)
```

### Hardware Acceleration

- **SIMD instructions**: Parallel bit operations
- **GPU acceleration**: Massive parallel hash computation
- **FPGAs**: Custom Bloom filter circuits for networking hardware

## Common Pitfalls

### 1. Wrong Parameter Calculation

```python
# WRONG: Using wrong formula
m = n * 10  # arbitrary multiplier

# RIGHT: Use proper formula
m = -(n * math.log(p)) / (math.log(2) ** 2)
```

### 2. Poor Hash Function

```python
# WRONG: Using Python's hash() (not uniform)
def bad_hash(item):
    return hash(item) % m

# RIGHT: Use proper hash function
import mmh3
def good_hash(item):
    return mmh3.hash(item) % m
```

### 3. Exceeding Capacity

```python
# Monitor and warn
if bloom.elements_added > bloom.expected_elements:
    logging.warning("Bloom filter exceeding capacity!")
    logging.warning(f"Current FP rate: {bloom.current_false_positive_rate()}")
```

### 4. Assuming Exact Membership

```python
# WRONG: Treating as exact set
if bloom.contains(email):
    send_email(email)  # might send to non-existent email!

# RIGHT: Verify on positive match
if bloom.contains(email):
    if email_exists_in_database(email):  # verify
        send_email(email)
```

## Testing Bloom Filters

```python
import random
import string

def test_bloom_filter():
    # Create filter
    bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)

    # Test 1: No false negatives
    added = set()
    for i in range(10000):
        word = ''.join(random.choices(string.ascii_letters, k=10))
        bf.add(word)
        added.add(word)

    # All added elements must be found
    for word in added:
        assert bf.contains(word), f"False negative for {word}!"

    # Test 2: Measure false positive rate
    false_positives = 0
    test_count = 100000

    for i in range(test_count):
        word = ''.join(random.choices(string.ascii_letters, k=10))
        if word not in added and bf.contains(word):
            false_positives += 1

    actual_fp_rate = false_positives / test_count
    print(f"Expected FP rate: 0.01")
    print(f"Actual FP rate: {actual_fp_rate:.4f}")

    # Should be close to expected (within tolerance)
    assert abs(actual_fp_rate - 0.01) < 0.005

test_bloom_filter()
```

## References and Further Reading

### Original Paper
- Bloom, Burton H. (1970). "Space/Time Trade-offs in Hash Coding with Allowable Errors"

### Modern Variations
- Fan et al. (2014). "Cuckoo Filter: Practically Better Than Bloom"
- Almeida et al. (2007). "Scalable Bloom Filters"

### Applications
- Google Bigtable paper (2006)
- Bitcoin BIP 37 (Bloom filtering)
- Cassandra documentation on Bloom filters

### Online Resources
- [Wikipedia: Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter)
- [Interactive Bloom Filter Visualization](https://llimllib.github.io/bloomfilter-tutorial/)
- [Bloom Filter Calculator](https://hur.st/bloomfilter/)

### Books
- "Probabilistic Data Structures and Algorithms" by Andrii Gakhov
- "Algorithms and Data Structures for Massive Datasets" by Dzejla Medjedovic

---

## Quick Reference Card

```
================================================
BLOOM FILTER CHEAT SHEET
================================================

Optimal bit array size:
  m = -n * ln(p) / (ln 2)^2

Optimal hash functions:
  k = (m/n) * ln(2)

False positive rate:
  p ~= (1 - e^(-kn/m))^k

Bits per element (optimal):
  m/n = -log_2(p) / ln(2) ~= 1.44 log_2(1/p)

Time Complexity:
  Insert: O(k)
  Query:  O(k)

Space: O(n log(1/p)) bits
================================================
```
