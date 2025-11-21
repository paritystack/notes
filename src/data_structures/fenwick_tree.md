# Fenwick Tree (Binary Indexed Tree)

## Table of Contents
- [Overview](#overview)
- [The Problem It Solves](#the-problem-it-solves)
- [How It Works](#how-it-works)
- [Key Concepts](#key-concepts)
- [Structure and Properties](#structure-and-properties)
- [Core Operations](#core-operations)
  - [Update](#update-operation)
  - [Query](#query-operation)
  - [Range Update](#range-update)
- [Time and Space Complexity](#time-and-space-complexity)
- [Implementation](#implementation)
  - [Python Implementation](#python-implementation)
  - [JavaScript Implementation](#javascript-implementation)
- [Variations](#variations)
- [Applications](#applications)
- [Common Problems](#common-problems)
- [Comparison with Other Structures](#comparison-with-other-structures)
- [Interview Patterns](#interview-patterns)
- [Advanced Topics](#advanced-topics)

## Overview

A **Fenwick Tree** (also called **Binary Indexed Tree** or **BIT**) is a data structure that efficiently supports prefix sum queries and point updates on an array. Invented by Peter Fenwick in 1994, it provides an elegant and space-efficient alternative to segment trees for certain range query problems.

### Key Characteristics

- **Compact**: Uses only n+1 elements (same as input array)
- **Efficient**: O(log n) for both query and update
- **Simple**: Easier to implement than segment trees
- **Bit manipulation based**: Uses clever binary representation tricks
- **Invertible operations only**: Works for operations with inverses (sum, XOR)

### Why Fenwick Trees?

**When to choose Fenwick Tree over Segment Tree:**
- Only need prefix sums or range sums (invertible operations)
- Want simpler, more compact code
- Space efficiency matters (n vs 4n)
- Don't need complex aggregations (min/max/GCD)

**Fenwick Tree advantages:**
- Half the memory of segment tree
- Simpler implementation (20-30 lines vs 60+ lines)
- Better constant factors (faster in practice)
- Elegant bit manipulation approach

## The Problem It Solves

### Classic Problem: Range Sum with Updates

**Given:**
- Array of n elements
- Multiple queries:
  - `prefix_sum(i)`: Sum of elements from index 0 to i
  - `range_sum(l, r)`: Sum of elements from index l to r
  - `update(i, delta)`: Add delta to element at index i

**Naive Approaches:**

| Approach | Update | Query | Trade-off |
|----------|--------|-------|-----------|
| Array | O(1) | O(n) | Slow queries |
| Prefix sum array | O(n) | O(1) | Slow updates |
| Fenwick Tree | O(log n) | O(log n) | Balanced! |

### Real-World Examples

1. **Cumulative Frequency**: Count occurrences up to a value
2. **Inversion Count**: Count inversions in array with updates
3. **2D Range Sums**: Sum of rectangle in matrix
4. **Order Statistics**: Find kth smallest element dynamically

## How It Works

### The Brilliant Insight

Fenwick Tree leverages binary representation to create a hierarchy of partial sums. Each index stores the sum of a specific range determined by its binary representation.

### Binary Representation Magic

Key observation: Every positive integer can be uniquely represented as a sum of powers of 2.

```
13 = 1101₂ = 8 + 4 + 1 = 2³ + 2² + 2⁰
```

**Fenwick Tree idea**: Store partial sums corresponding to these powers of 2.

### LSB (Least Significant Bit) Operation

The core operation is extracting the rightmost set bit:

```
LSB(x) = x & (-x)

Examples:
12 = 1100₂ → LSB(12) = 0100₂ = 4
10 = 1010₂ → LSB(10) = 0010₂ = 2
7  = 0111₂ → LSB(7)  = 0001₂ = 1
```

**Why it works:**
- `-x` in two's complement flips bits and adds 1
- `x & (-x)` isolates the rightmost 1-bit

### Tree Structure

Each index i in the Fenwick Tree stores the sum of a range:
- Range length = LSB(i)
- Range ends at i
- Range starts at i - LSB(i) + 1

```
Visual representation for array of size 8:

Index:  1   2   3   4   5   6   7   8
Range: [1] [1,2] [3] [1,4] [5] [5,6] [7] [1,8]
       1   2     1   4     1   2     1   8     (range lengths)

Tree structure:
                    8 [1-8]
                   / \
                  /   \
                 /     \
                4 [1-4] \
               / \       \
              /   \       \
             2[1-2] 6[5-6] \
            / \    / \      \
           1   3  5   7      (leaf-like)

BIT[i] represents sum of range [i - LSB(i) + 1, i]
```

### Example Walkthrough

Array: `[3, 2, -1, 6, 5, 4, -3, 3]` (0-indexed in problem, 1-indexed in BIT)

**Build Fenwick Tree:**

```
Index i | LSB(i) | Range | Sum
--------|--------|-------|----
1       | 1      | [1,1] | 3
2       | 2      | [1,2] | 5
3       | 1      | [3,3] | -1
4       | 4      | [1,4] | 10
5       | 1      | [5,5] | 5
6       | 2      | [5,6] | 9
7       | 1      | [7,7] | -3
8       | 8      | [1,8] | 19

BIT = [0, 3, 5, -1, 10, 5, 9, -3, 19]
      ^
      (index 0 unused)
```

**Query prefix_sum(6)** (sum of first 6 elements):

```
Start at index 6:
sum = 0
sum += BIT[6] = 9     (covers [5,6])
6 -= LSB(6) = 6 - 2 = 4

sum += BIT[4] = 10    (covers [1,4])
4 -= LSB(4) = 4 - 4 = 0

Result: 19
Path: 6 → 4 → 0 (stop)
```

**Update index 3 by +5:**

```
Start at index 3:
BIT[3] += 5           (BIT[3] = -1 + 5 = 4)
3 += LSB(3) = 3 + 1 = 4

BIT[4] += 5           (BIT[4] = 10 + 5 = 15)
4 += LSB(4) = 4 + 4 = 8

BIT[8] += 5           (BIT[8] = 19 + 5 = 24)
8 += LSB(8) = 8 + 8 = 16 (out of bounds, stop)

Path: 3 → 4 → 8 → stop
```

## Key Concepts

### 1. Binary Index Decomposition

Every index can be viewed as a hierarchical structure:

```
Index 6 (binary 110):
- Covers [5, 6] (length 2)
- Parent is index 4 (remove rightmost 1)
- Child connections based on bit patterns
```

### 2. Parent-Child Relationship

**To find parent** (used in update):
```
parent(i) = i + LSB(i)
```

**To find "previous" in query chain** (used in query):
```
previous(i) = i - LSB(i)
```

### 3. Responsibility Ranges

Each index is "responsible" for a specific range:

```
Responsibility pattern:
- Indices ending in ...0001: cover 1 element
- Indices ending in ...0010: cover 2 elements
- Indices ending in ...0100: cover 4 elements
- Indices ending in ...1000: cover 8 elements
```

### 4. Query Path

To compute prefix sum up to index i:
- Start at i
- Add BIT[i] to sum
- Jump to i - LSB(i)
- Repeat until i = 0

The path always goes through O(log n) indices.

### 5. Update Path

To update index i:
- Start at i
- Update BIT[i]
- Jump to i + LSB(i) (parent)
- Repeat until i > n

The path always touches O(log n) indices.

## Structure and Properties

### Array Representation

```python
# 1-indexed array (index 0 unused for convenience)
BIT = [0] * (n + 1)

# Index 0 is typically unused
# Indices 1 to n store partial sums
```

### Key Properties

1. **Space**: O(n) - exactly n+1 elements
2. **1-indexed**: Convention uses 1-indexed for cleaner bit operations
3. **Range coverage**: BIT[i] stores sum of range [i - LSB(i) + 1, i]
4. **Height**: O(log n) - maximum path length
5. **Invertibility**: Only works for operations with inverses

### Mathematical Foundation

**Prefix sum decomposition:**

Any prefix sum can be computed as a sum of O(log n) disjoint ranges stored in the BIT.

```
prefix_sum(13) = prefix_sum(1101₂)
                = sum[1-8] + sum[9-12] + sum[13-13]
                = BIT[8] + BIT[12] + BIT[13]
```

The decomposition follows the binary representation of the index!

## Core Operations

### Update Operation

**Purpose**: Add `delta` to element at index `i`

**Algorithm:**
```python
def update(i, delta):
    """Add delta to element at index i (1-indexed)"""
    while i <= n:
        BIT[i] += delta
        i += i & (-i)  # Move to parent
```

**Process:**
1. Update BIT[i] (affects range ending at i)
2. Move to parent: i += LSB(i)
3. Repeat until out of bounds

**Visualization:**
```
Update index 3:

    8
   /|
  4 |
 /| |
2 | |
|X| |    X = index 3
1 3 5 7

Path: 3 → 4 → 8
All ancestors of index 3 get updated
```

**Time Complexity**: O(log n)
- At most ⌈log₂ n⌉ iterations
- Each iteration: O(1) bit operation

### Query Operation

**Purpose**: Compute prefix sum from index 1 to i

**Algorithm:**
```python
def prefix_sum(i):
    """Get sum of elements from 1 to i (1-indexed)"""
    total = 0
    while i > 0:
        total += BIT[i]
        i -= i & (-i)  # Move to previous
    return total
```

**Process:**
1. Add BIT[i] to sum (get range ending at i)
2. Remove last covered range: i -= LSB(i)
3. Repeat until i = 0

**Visualization:**
```
Query prefix_sum(7):

    8
   /|
  4 |
 /| |
2 | 6
| | |\
1 3 5 7*

Path: 7 → 6 → 4 → 0
Collect: BIT[7] + BIT[6] + BIT[4]
```

**Range Sum Query:**
```python
def range_sum(left, right):
    """Get sum of elements from left to right (1-indexed)"""
    return prefix_sum(right) - prefix_sum(left - 1)
```

**Time Complexity**: O(log n)
- At most ⌈log₂ n⌉ iterations
- Range sum: 2 × O(log n) = O(log n)

### Build Operation

**Efficient construction from array:**

**Method 1: Repeated Updates** - O(n log n)
```python
def build_slow(arr):
    """Build BIT by updating each element"""
    BIT = [0] * (len(arr) + 1)
    for i, val in enumerate(arr, 1):
        update(i, val)
```

**Method 2: Direct Construction** - O(n)
```python
def build_fast(arr):
    """Build BIT in linear time"""
    n = len(arr)
    BIT = [0] * (n + 1)

    # Copy array values
    for i in range(1, n + 1):
        BIT[i] = arr[i - 1]

    # Update parent nodes
    for i in range(1, n + 1):
        parent = i + (i & -i)
        if parent <= n:
            BIT[parent] += BIT[i]

    return BIT
```

**How O(n) build works:**
- Start with each BIT[i] = arr[i-1]
- Each node adds itself to its parent
- Each element is touched exactly once as a child
- Total: O(n) operations

### Range Update

**Problem**: Add delta to all elements in range [l, r]

**Solution**: Use difference array technique

```python
class RangeUpdateBIT:
    def __init__(self, n):
        self.n = n
        self.diff_BIT = [0] * (n + 1)

    def range_update(self, left, right, delta):
        """Add delta to range [left, right]"""
        self._update(left, delta)
        self._update(right + 1, -delta)

    def _update(self, i, delta):
        while i <= self.n:
            self.diff_BIT[i] += delta
            i += i & (-i)

    def point_query(self, i):
        """Get value at index i after all updates"""
        total = 0
        while i > 0:
            total += self.diff_BIT[i]
            i -= i & (-i)
        return total
```

**Complexity**: O(log n) per range update

## Time and Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Build (slow) | O(n log n) | O(n) | Using n updates |
| Build (fast) | O(n) | O(n) | Direct construction |
| Point update | O(log n) | - | Add to single element |
| Point query | O(log n) | - | Get single element value |
| Prefix sum | O(log n) | - | Sum [1, i] |
| Range sum | O(log n) | - | Two prefix sums |
| Range update | O(log n) | O(n) | Using difference array |

**Space Complexity**: O(n)
- BIT array: n+1 elements
- No auxiliary structures needed
- Much better than segment tree's 4n

## Implementation

### Python Implementation

**Basic Fenwick Tree (Range Sum Queries):**

```python
class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for prefix sum queries.

    Supports:
    - Point update: add delta to element at index
    - Prefix sum: sum of elements from index 1 to i
    - Range sum: sum of elements from index l to r

    Note: Uses 1-indexed convention internally
    """

    def __init__(self, n):
        """
        Initialize Fenwick Tree for n elements.

        Args:
            n (int): Number of elements

        Time: O(n)
        """
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    @staticmethod
    def from_array(arr):
        """
        Build Fenwick Tree from existing array.

        Args:
            arr (list): Input array (0-indexed)

        Returns:
            FenwickTree: Built tree

        Time: O(n)
        """
        n = len(arr)
        ft = FenwickTree(n)

        # Copy values
        for i in range(1, n + 1):
            ft.tree[i] = arr[i - 1]

        # Build tree by propagating to parents
        for i in range(1, n + 1):
            parent = i + (i & -i)
            if parent <= n:
                ft.tree[parent] += ft.tree[i]

        return ft

    def update(self, i, delta):
        """
        Add delta to element at index i.

        Args:
            i (int): Index (1-indexed)
            delta: Value to add

        Time: O(log n)
        """
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # Move to parent

    def prefix_sum(self, i):
        """
        Get sum of elements from index 1 to i.

        Args:
            i (int): Index (1-indexed)

        Returns:
            Sum of arr[1..i]

        Time: O(log n)
        """
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)  # Remove last bit
        return total

    def range_sum(self, left, right):
        """
        Get sum of elements from index left to right.

        Args:
            left (int): Left boundary (1-indexed)
            right (int): Right boundary (1-indexed)

        Returns:
            Sum of arr[left..right]

        Time: O(log n)
        """
        if left > 1:
            return self.prefix_sum(right) - self.prefix_sum(left - 1)
        return self.prefix_sum(right)


# Example usage
arr = [3, 2, -1, 6, 5, 4, -3, 3]
ft = FenwickTree.from_array(arr)

print("Prefix sum [1, 4]:", ft.prefix_sum(4))     # 10
print("Range sum [2, 5]:", ft.range_sum(2, 5))     # 12

ft.update(3, 5)  # Add 5 to index 3
print("Range sum [2, 5]:", ft.range_sum(2, 5))     # 17
```

**2D Fenwick Tree:**

```python
class FenwickTree2D:
    """2D Fenwick Tree for rectangle sum queries"""

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x, y, delta):
        """Add delta to element at (x, y). Time: O(log n * log m)"""
        i = x
        while i <= self.rows:
            j = y
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def prefix_sum(self, x, y):
        """Sum of rectangle from (1,1) to (x,y). Time: O(log n * log m)"""
        total = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                total += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return total

    def range_sum(self, x1, y1, x2, y2):
        """Sum of rectangle from (x1,y1) to (x2,y2)"""
        return (self.prefix_sum(x2, y2)
                - self.prefix_sum(x1 - 1, y2)
                - self.prefix_sum(x2, y1 - 1)
                + self.prefix_sum(x1 - 1, y1 - 1))
```

### JavaScript Implementation

```javascript
class FenwickTree {
    constructor(n) {
        this.n = n;
        this.tree = new Array(n + 1).fill(0);
    }

    static fromArray(arr) {
        const n = arr.length;
        const ft = new FenwickTree(n);

        for (let i = 1; i <= n; i++) {
            ft.tree[i] = arr[i - 1];
        }

        for (let i = 1; i <= n; i++) {
            const parent = i + (i & -i);
            if (parent <= n) {
                ft.tree[parent] += ft.tree[i];
            }
        }

        return ft;
    }

    update(i, delta) {
        while (i <= this.n) {
            this.tree[i] += delta;
            i += i & (-i);
        }
    }

    prefixSum(i) {
        let total = 0;
        while (i > 0) {
            total += this.tree[i];
            i -= i & (-i);
        }
        return total;
    }

    rangeSum(left, right) {
        if (left > 1) {
            return this.prefixSum(right) - this.prefixSum(left - 1);
        }
        return this.prefixSum(right);
    }
}

// Example usage
const arr = [3, 2, -1, 6, 5, 4, -3, 3];
const ft = FenwickTree.fromArray(arr);

console.log('Prefix sum [1,4]:', ft.prefixSum(4));
console.log('Range sum [2,5]:', ft.rangeSum(2, 5));
```

## Variations

### 1. Range Update Range Query (RURQ)

Use two Fenwick Trees for both range updates and range queries.

### 2. Order Statistics Tree

Find kth smallest element using coordinate compression and BIT.

### 3. 2D/3D Fenwick Trees

Extend to multiple dimensions for rectangle/cuboid sum queries.

## Applications

1. **Cumulative Frequency Queries**: Count occurrences in range
2. **Inversion Count**: Count inversions with dynamic updates
3. **Range Sum 2D**: Matrix rectangle sum queries
4. **Dynamic Ranking**: Maintain rankings with score updates
5. **Coordinate Compression**: Handle large value ranges efficiently

## Common Problems

| Problem | Difficulty | Key Technique |
|---------|-----------|---------------|
| [307. Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | Medium | Basic BIT |
| [308. Range Sum Query 2D - Mutable](https://leetcode.com/problems/range-sum-query-2d-mutable/) | Hard | 2D BIT |
| [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | Hard | BIT + coordinate compression |
| [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/) | Hard | BIT + prefix sum |
| [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) | Hard | BIT + merge sort |
| [1649. Create Sorted Array through Instructions](https://leetcode.com/problems/create-sorted-array-through-instructions/) | Hard | BIT for order statistics |

## Comparison with Other Structures

### Fenwick Tree vs Segment Tree

| Feature | Fenwick Tree | Segment Tree |
|---------|--------------|--------------|
| **Space** | O(n) | O(4n) |
| **Code complexity** | Simpler | More complex |
| **Operations** | Sum, XOR (invertible) | Any associative |
| **Range update** | With tricks | Native support |
| **Speed** | Faster constants | Slower constants |

**Use Fenwick Tree when:** Only need sum/XOR, want simpler code, space matters

**Use Segment Tree when:** Need min/max/GCD, frequent range updates, more flexibility

## Interview Patterns

### Pattern 1: Coordinate Compression

For large value ranges (10^9), compress to smaller range before using BIT.

### Pattern 2: Count Smaller/Larger Elements

Process array right-to-left, use BIT to count elements seen so far.

### Pattern 3: 2D Range Queries

Use 2D BIT for matrix rectangle sum problems.

## Key Takeaways

1. **Fenwick Tree = BIT manipulation magic** for prefix sums
2. **LSB operation `x & (-x)`** is the core technique
3. **1-indexed convention** makes implementation cleaner
4. **O(log n) operations** with excellent constant factors
5. **Works only for invertible operations** (sum, XOR)
6. **Simpler and faster than segment tree** for applicable problems
7. **Coordinate compression** essential for large ranges

## When to Use Fenwick Trees

✅ **Use when:**
- Need range sum queries with point updates
- Operations are invertible (sum, XOR)
- Want simple, efficient implementation
- Space efficiency matters

❌ **Don't use when:**
- Need min/max/GCD (use segment tree)
- Need frequent range updates (segment tree better)
- Array is static (use prefix sum)
- Operations aren't invertible

---

**Time to Implement**: 15-20 minutes

**Most Common Uses**: Range sum queries, inversion counting, order statistics, 2D range sums
