# Segment Trees

## Table of Contents
- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Structure and Properties](#structure-and-properties)
- [Core Operations](#core-operations)
  - [Build](#build-operation)
  - [Query](#query-operation)
  - [Update](#update-operation)
- [Lazy Propagation](#lazy-propagation)
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

A **Segment Tree** is a tree data structure used for storing information about array intervals (segments). It allows efficient querying and updating of array ranges.

### Why Segment Trees?

Segment trees excel at problems requiring:
- **Range queries** - Find min/max/sum over array range [L, R]
- **Range updates** - Update all elements in range [L, R]
- **Dynamic data** - Array elements change frequently
- **Multiple queries** - Many queries on the same array

### The Range Query Problem

**Problem**: Given array `arr[]` and many queries:
- Find sum/min/max of elements from index L to R
- Update element at index i

**Naive approach**:
- Query: O(n) - iterate through range
- Update: O(1) - change single element

**With segment tree**:
- Query: O(log n)
- Update: O(log n)
- Build: O(n)

### Real-World Applications

1. **Database Systems**: Range aggregation queries
2. **Graphics/Gaming**: Collision detection, ray tracing
3. **Financial Systems**: Time-series analysis, stock price ranges
4. **GIS Systems**: Spatial range queries
5. **Network Monitoring**: Traffic analysis over time windows

## Key Concepts

### Segments and Intervals

A segment tree divides an array into segments (intervals):

```
Array: [1, 3, 5, 7, 9, 11]
Indices: 0  1  2  3  4  5

Segments:
[0,5] - entire array
[0,2], [3,5] - two halves
[0,1], [2,2], [3,4], [5,5] - smaller segments
[0,0], [1,1], [2,2], [3,3], [4,4], [5,5] - individual elements
```

### Tree Structure

Each node represents a segment and stores aggregate information:

```
                    [0-5]: sum=36
                    /            \
           [0-2]: sum=9        [3-5]: sum=27
           /        \          /          \
    [0-1]: sum=4  [2]: 5  [3-4]: sum=16  [5]: 11
    /      \                /       \
[0]: 1   [1]: 3        [3]: 7    [4]: 9
```

### Node Representation

- **Leaf nodes**: Individual array elements
- **Internal nodes**: Aggregate of children (sum, min, max, etc.)
- **Root**: Aggregate of entire array

### Binary Tree Property

- Left child represents left half of segment
- Right child represents right half of segment
- Height = O(log n)

## Structure and Properties

### Array Representation

Segment trees are typically stored in an array (like heaps):

```python
tree = [0] * (4 * n)  # Array to store segment tree

# Node indexing (1-based):
# Node i:
#   Left child: 2*i
#   Right child: 2*i + 1
#   Parent: i//2
```

### Size of Segment Tree

For array of size n:
- If n is power of 2: tree size = 2n - 1
- General case: tree size ≤ 4n (to handle all cases safely)

**Why 4n?**
- Height h = ⌈log₂ n⌉
- Maximum nodes = 2^(h+1) - 1
- For safety, allocate 4n

### Tree Height

- Height = ⌈log₂ n⌉
- Example: array of 8 elements → height = 3

## Core Operations

### Build Operation

**Purpose**: Construct segment tree from array

**Recursive approach**:
1. Base case: If segment has one element, store it
2. Recursive case:
   - Build left child for left half
   - Build right child for right half
   - Current node = aggregate of children

**Visual Example**:
```
Array: [1, 3, 5, 7]

Build process (sum tree):
1. Build [0-3]
   - Build [0-1]: Build [0-0]=1, [1-1]=3 → sum=4
   - Build [2-3]: Build [2-2]=5, [3-3]=7 → sum=12
   - [0-3] = 4 + 12 = 16
```

**Time Complexity**: O(n)
- Visit each array element once
- Each element appears in log n nodes

### Query Operation

**Purpose**: Find aggregate (sum/min/max) over range [L, R]

**Three cases for segment [node_L, node_R]**:

1. **Complete overlap**: [L, R] completely covers [node_L, node_R]
   - Return node value

2. **No overlap**: [L, R] doesn't intersect [node_L, node_R]
   - Return identity (0 for sum, ∞ for min, -∞ for max)

3. **Partial overlap**: [L, R] partially overlaps [node_L, node_R]
   - Query both children, combine results

**Visual Example**:
```
Query sum [1, 3]:

                    [0-3]: 16
                    /        \
           [0-1]: 4          [2-3]: 12
           /    \            /      \
       [0]: 1  [1]: 3    [2]: 5   [3]: 7

Query [1,3]:
- [0-3]: Partial overlap → go deeper
- [0-1]: Partial overlap → go deeper
  - [0]: No overlap → skip
  - [1]: Complete overlap → return 3
- [2-3]: Complete overlap → return 12
Result: 3 + 12 = 15
```

**Time Complexity**: O(log n)
- Visit at most 4 nodes per level
- Tree height = log n

### Update Operation

**Purpose**: Change value at index i and update affected segments

**Approach**:
1. Find leaf node for index i
2. Update leaf with new value
3. Propagate changes up to root

**Visual Example**:
```
Update index 1: 3 → 10

Before:                      After:
        [0-3]: 16                    [0-3]: 23
        /        \                   /        \
   [0-1]: 4    [2-3]: 12       [0-1]: 11   [2-3]: 12
   /    \       /      \        /     \      /      \
[0]: 1 [1]: 3 [2]: 5 [3]: 7  [0]: 1 [1]: 10 [2]: 5 [3]: 7

Changed nodes: [1], [0-1], [0-3]
```

**Time Complexity**: O(log n)
- Update log n nodes (path from leaf to root)

## Lazy Propagation

### The Problem

**Range update** with standard approach:
- Update all elements in [L, R]
- Time: O(n log n) - update each element individually

This is inefficient for frequent range updates!

### The Solution: Lazy Propagation

**Idea**: Postpone updates to children until necessary

**Lazy array**: Stores pending updates for each node

```python
lazy = [0] * (4 * n)  # Lazy propagation array

# lazy[i] != 0 means:
# "Node i and its subtree need to be updated by lazy[i]"
```

### How Lazy Propagation Works

1. **Range update [L, R]**:
   - Mark nodes with lazy values
   - Don't update children yet

2. **Query**:
   - Before querying node, apply pending lazy updates
   - Push lazy values to children

3. **Update children only when needed**:
   - Reduces redundant work
   - O(log n) for range update!

**Visual Example**:
```
Range update: Add 10 to [1, 3]

Before:
        [0-3]: 16
        /        \
   [0-1]: 4    [2-3]: 12

After (with lazy):
        [0-3]: 16, lazy=0
        /              \
   [0-1]: 4          [2-3]: 12
   lazy=10           lazy=10

When querying [1, 1]:
- Visit [0-1], see lazy=10
- Apply: [0-1] = 4 + 10*2 = 24
- Push to children: [0].lazy=10, [1].lazy=10
- Clear [0-1].lazy = 0
```

### Time Complexity with Lazy Propagation

- **Range update**: O(log n) - instead of O(n log n)
- **Range query**: O(log n) - same as before
- **Point update**: O(log n) - same as before

## Time and Space Complexity

| Operation | Time Complexity | Explanation |
|-----------|----------------|-------------|
| Build | O(n) | Process each element once |
| Point query | O(log n) | Traverse from root to leaf |
| Range query | O(log n) | Visit O(log n) nodes |
| Point update | O(log n) | Update path from leaf to root |
| Range update (naive) | O(n log n) | Update each element |
| Range update (lazy) | O(log n) | Mark nodes lazily |

**Space Complexity**: O(n)
- Segment tree array: O(4n) = O(n)
- Lazy array (if used): O(4n) = O(n)
- Total: O(n)

## Implementation

### Python Implementation

**Basic Segment Tree (Sum)**:

```python
class SegmentTree:
    """
    Segment tree for range sum queries and point updates.

    Supports:
    - Range sum query: sum of elements in [L, R]
    - Point update: update single element at index i
    """

    def __init__(self, arr):
        """
        Initialize segment tree from array.

        Args:
            arr: Input array

        Time Complexity: O(n)
        """
        self.n = len(arr)
        self.arr = arr[:]
        self.tree = [0] * (4 * self.n)
        self._build(0, 0, self.n - 1)

    def _build(self, node, start, end):
        """
        Build segment tree recursively.

        Args:
            node: Current node index in tree
            start: Start index of segment
            end: End index of segment
        """
        if start == end:
            # Leaf node
            self.tree[node] = self.arr[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        # Build left and right subtrees
        self._build(left_child, start, mid)
        self._build(right_child, mid + 1, end)

        # Internal node = sum of children
        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, L, R):
        """
        Query sum of elements in range [L, R].

        Args:
            L: Left boundary (inclusive)
            R: Right boundary (inclusive)

        Returns:
            Sum of elements in [L, R]

        Time Complexity: O(log n)
        """
        return self._query(0, 0, self.n - 1, L, R)

    def _query(self, node, start, end, L, R):
        """
        Recursive range query helper.

        Args:
            node: Current node index
            start, end: Current segment [start, end]
            L, R: Query range [L, R]

        Returns:
            Sum of overlapping portion
        """
        # No overlap
        if R < start or L > end:
            return 0

        # Complete overlap
        if L <= start and end <= R:
            return self.tree[node]

        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_sum = self._query(left_child, start, mid, L, R)
        right_sum = self._query(right_child, mid + 1, end, L, R)

        return left_sum + right_sum

    def update(self, index, value):
        """
        Update element at index to new value.

        Args:
            index: Index to update
            value: New value

        Time Complexity: O(log n)
        """
        self.arr[index] = value
        self._update(0, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        """
        Recursive update helper.

        Args:
            node: Current node index
            start, end: Current segment [start, end]
            index: Index to update
            value: New value
        """
        if start == end:
            # Leaf node
            self.tree[node] = value
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if index <= mid:
            self._update(left_child, start, mid, index, value)
        else:
            self._update(right_child, mid + 1, end, index, value)

        # Update current node
        self.tree[node] = self.tree[left_child] + self.tree[right_child]


# Example usage
arr = [1, 3, 5, 7, 9, 11]
seg_tree = SegmentTree(arr)

print(seg_tree.query(1, 3))  # Sum of arr[1:4] = 3+5+7 = 15
seg_tree.update(1, 10)       # arr[1] = 10
print(seg_tree.query(1, 3))  # Sum = 10+5+7 = 22
```

**Segment Tree with Lazy Propagation**:

```python
class LazySegmentTree:
    """
    Segment tree with lazy propagation for efficient range updates.

    Supports:
    - Range sum query: sum of elements in [L, R]
    - Range update: add value to all elements in [L, R]
    """

    def __init__(self, arr):
        self.n = len(arr)
        self.arr = arr[:]
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # Lazy propagation array
        self._build(0, 0, self.n - 1)

    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._build(left_child, start, mid)
        self._build(right_child, mid + 1, end)
        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def _push_lazy(self, node, start, end):
        """
        Push pending lazy updates to children.

        Args:
            node: Current node
            start, end: Current segment
        """
        if self.lazy[node] == 0:
            return  # No pending update

        # Apply pending update to current node
        self.tree[node] += (end - start + 1) * self.lazy[node]

        # Push to children if not leaf
        if start != end:
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self.lazy[left_child] += self.lazy[node]
            self.lazy[right_child] += self.lazy[node]

        # Clear lazy value
        self.lazy[node] = 0

    def range_update(self, L, R, value):
        """
        Add value to all elements in range [L, R].

        Args:
            L: Left boundary
            R: Right boundary
            value: Value to add

        Time Complexity: O(log n)
        """
        self._range_update(0, 0, self.n - 1, L, R, value)

    def _range_update(self, node, start, end, L, R, value):
        # Apply pending updates first
        self._push_lazy(node, start, end)

        # No overlap
        if R < start or L > end:
            return

        # Complete overlap
        if L <= start and end <= R:
            self.lazy[node] += value
            self._push_lazy(node, start, end)
            return

        # Partial overlap - recurse on children
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._range_update(left_child, start, mid, L, R, value)
        self._range_update(right_child, mid + 1, end, L, R, value)

        # Update current node after children updated
        self._push_lazy(left_child, start, mid)
        self._push_lazy(right_child, mid + 1, end)
        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, L, R):
        """
        Query sum of elements in range [L, R].

        Time Complexity: O(log n)
        """
        return self._query(0, 0, self.n - 1, L, R)

    def _query(self, node, start, end, L, R):
        # Apply pending updates
        self._push_lazy(node, start, end)

        # No overlap
        if R < start or L > end:
            return 0

        # Complete overlap
        if L <= start and end <= R:
            return self.tree[node]

        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_sum = self._query(left_child, start, mid, L, R)
        right_sum = self._query(right_child, mid + 1, end, L, R)

        return left_sum + right_sum


# Example usage
arr = [1, 3, 5, 7, 9, 11]
seg_tree = LazySegmentTree(arr)

print(seg_tree.query(1, 3))        # Sum = 15
seg_tree.range_update(1, 3, 10)    # Add 10 to arr[1], arr[2], arr[3]
print(seg_tree.query(1, 3))        # Sum = 15 + 30 = 45
```

**Segment Tree for Minimum Queries**:

```python
class MinSegmentTree:
    """Segment tree for range minimum queries"""

    def __init__(self, arr):
        self.n = len(arr)
        self.arr = arr[:]
        self.tree = [float('inf')] * (4 * self.n)
        self._build(0, 0, self.n - 1)

    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
            return

        mid = (start + end) // 2
        left = 2 * node + 1
        right = 2 * node + 2

        self._build(left, start, mid)
        self._build(right, mid + 1, end)
        self.tree[node] = min(self.tree[left], self.tree[right])

    def query(self, L, R):
        """Find minimum in range [L, R]"""
        return self._query(0, 0, self.n - 1, L, R)

    def _query(self, node, start, end, L, R):
        if R < start or L > end:
            return float('inf')

        if L <= start and end <= R:
            return self.tree[node]

        mid = (start + end) // 2
        left = 2 * node + 1
        right = 2 * node + 2

        left_min = self._query(left, start, mid, L, R)
        right_min = self._query(right, mid + 1, end, L, R)

        return min(left_min, right_min)

    def update(self, index, value):
        """Update element at index"""
        self.arr[index] = value
        self._update(0, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        left = 2 * node + 1
        right = 2 * node + 2

        if index <= mid:
            self._update(left, start, mid, index, value)
        else:
            self._update(right, mid + 1, end, index, value)

        self.tree[node] = min(self.tree[left], self.tree[right])
```

### JavaScript Implementation

**Basic Segment Tree**:

```javascript
class SegmentTree {
    /**
     * Initialize segment tree for range sum queries
     * @param {number[]} arr - Input array
     */
    constructor(arr) {
        this.n = arr.length;
        this.arr = [...arr];
        this.tree = new Array(4 * this.n).fill(0);
        this._build(0, 0, this.n - 1);
    }

    _build(node, start, end) {
        if (start === end) {
            this.tree[node] = this.arr[start];
            return;
        }

        const mid = Math.floor((start + end) / 2);
        const leftChild = 2 * node + 1;
        const rightChild = 2 * node + 2;

        this._build(leftChild, start, mid);
        this._build(rightChild, mid + 1, end);
        this.tree[node] = this.tree[leftChild] + this.tree[rightChild];
    }

    /**
     * Query sum of range [L, R]
     * @param {number} L - Left boundary
     * @param {number} R - Right boundary
     * @returns {number} Sum of elements
     */
    query(L, R) {
        return this._query(0, 0, this.n - 1, L, R);
    }

    _query(node, start, end, L, R) {
        // No overlap
        if (R < start || L > end) {
            return 0;
        }

        // Complete overlap
        if (L <= start && end <= R) {
            return this.tree[node];
        }

        // Partial overlap
        const mid = Math.floor((start + end) / 2);
        const leftChild = 2 * node + 1;
        const rightChild = 2 * node + 2;

        const leftSum = this._query(leftChild, start, mid, L, R);
        const rightSum = this._query(rightChild, mid + 1, end, L, R);

        return leftSum + rightSum;
    }

    /**
     * Update element at index to new value
     * @param {number} index - Index to update
     * @param {number} value - New value
     */
    update(index, value) {
        this.arr[index] = value;
        this._update(0, 0, this.n - 1, index, value);
    }

    _update(node, start, end, index, value) {
        if (start === end) {
            this.tree[node] = value;
            return;
        }

        const mid = Math.floor((start + end) / 2);
        const leftChild = 2 * node + 1;
        const rightChild = 2 * node + 2;

        if (index <= mid) {
            this._update(leftChild, start, mid, index, value);
        } else {
            this._update(rightChild, mid + 1, end, index, value);
        }

        this.tree[node] = this.tree[leftChild] + this.tree[rightChild];
    }
}

// Example usage
const arr = [1, 3, 5, 7, 9, 11];
const segTree = new SegmentTree(arr);

console.log(segTree.query(1, 3));  // 15
segTree.update(1, 10);
console.log(segTree.query(1, 3));  // 22
```

## Variations

### 1. Maximum/Minimum Segment Tree

Change aggregate function from sum to min/max:

```python
# In build and update:
self.tree[node] = max(self.tree[left_child], self.tree[right_child])

# In query (no overlap case):
return float('-inf')  # For max
return float('inf')   # For min
```

### 2. GCD Segment Tree

Store GCD of range:

```python
import math

def _build(self, node, start, end):
    if start == end:
        self.tree[node] = self.arr[start]
        return

    # ... build children ...

    self.tree[node] = math.gcd(self.tree[left], self.tree[right])
```

### 3. Count Segment Tree

Count elements satisfying property:

```python
def _query(self, node, start, end, L, R, threshold):
    """Count elements > threshold in [L, R]"""
    # Store counts in segment tree
    # Useful for range frequency queries
```

### 4. 2D Segment Tree

For 2D range queries (rectangle sums):

```python
class SegmentTree2D:
    """Segment tree for 2D range queries"""

    def __init__(self, matrix):
        self.n = len(matrix)
        self.m = len(matrix[0])
        # Build tree of trees
        # Each node contains a 1D segment tree
```

## Applications

### 1. Range Sum Queries

**Problem**: Given array, answer many range sum queries

```python
arr = [1, 3, 5, 7, 9, 11]
seg_tree = SegmentTree(arr)

# Multiple queries
print(seg_tree.query(0, 2))  # sum(arr[0:3])
print(seg_tree.query(3, 5))  # sum(arr[3:6])
print(seg_tree.query(1, 4))  # sum(arr[1:5])
```

### 2. Range Minimum/Maximum Query (RMQ)

**Problem**: Find minimum in any range [L, R]

```python
min_tree = MinSegmentTree([3, 2, 4, 5, 1, 1, 5, 3])

print(min_tree.query(0, 4))  # min of [3,2,4,5,1] = 1
print(min_tree.query(3, 7))  # min of [5,1,1,5,3] = 1
```

**Application**: Stock prices, sensor readings, temperature data

### 3. Counting Inversions with Updates

**Problem**: Count pairs (i, j) where i < j but arr[i] > arr[j], with updates

```python
def count_inversions_with_updates(arr, queries):
    """
    Count inversions dynamically as array changes.
    Uses coordinate compression + segment tree.
    """
    # Compress coordinates
    # Build segment tree
    # Process updates and queries
```

### 4. Range GCD

**Problem**: Find GCD of all elements in range [L, R]

```python
class GCDSegmentTree:
    def query(self, L, R):
        """Find GCD of arr[L:R+1]"""
        return self._query(0, 0, self.n - 1, L, R)

# Application: Number theory problems, competitive programming
```

### 5. Lazy Propagation: Range Assignment

**Problem**: Set all elements in [L, R] to value v

```python
class AssignmentSegmentTree:
    """Range assignment with lazy propagation"""

    def range_assign(self, L, R, value):
        """Set arr[L:R+1] = value"""
        # Use lazy array to mark assignment
        # Apply only when needed
```

## Common Problems

### LeetCode Problems

| Problem | Difficulty | Key Concept |
|---------|-----------|-------------|
| [307. Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/) | Medium | Basic segment tree |
| [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) | Hard | Segment tree + sorting |
| [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/) | Hard | Segment tree + prefix sum |
| [406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/) | Medium | Segment tree for kth element |
| [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) | Hard | Merge sort or segment tree |
| [699. Falling Squares](https://leetcode.com/problems/falling-squares/) | Hard | Coordinate compression + segment tree |
| [715. Range Module](https://leetcode.com/problems/range-module/) | Hard | Segment tree with lazy propagation |
| [732. My Calendar III](https://leetcode.com/problems/my-calendar-iii/) | Hard | Sweep line or segment tree |
| [850. Rectangle Area II](https://leetcode.com/problems/rectangle-area-ii/) | Hard | Coordinate compression + segment tree |

### Codeforces/Competitive Programming

- **Range Update Range Query (RURQ)**: Lazy propagation essential
- **Dynamic RMQ**: Find min/max with frequent updates
- **Persistent Segment Trees**: Time-travel queries
- **2D Range Queries**: Matrix sum/min/max queries

## Comparison with Other Structures

### Segment Tree vs Fenwick Tree (BIT)

| Feature | Segment Tree | Fenwick Tree |
|---------|-------------|--------------|
| **Implementation** | More complex | Simpler |
| **Space** | 4n | n+1 |
| **Range query** | O(log n) | O(log n) |
| **Point update** | O(log n) | O(log n) |
| **Range update** | O(log n) with lazy | Not directly supported |
| **Flexibility** | Any associative operation | Only invertible operations |
| **Use case** | Min/max/GCD/custom | Sum/XOR (invertible ops) |

**When to use Segment Tree**:
- Need range updates (with lazy propagation)
- Operation is not invertible (min, max, GCD)
- Need custom/complex aggregation

**When to use Fenwick Tree**:
- Only need range sum or XOR
- Want simpler implementation
- Space is concern (though 4n vs n is usually not critical)

### Segment Tree vs Prefix Sum

| Feature | Segment Tree | Prefix Sum |
|---------|-------------|------------|
| **Build** | O(n) | O(n) |
| **Query** | O(log n) | O(1) |
| **Update** | O(log n) | O(n) |
| **Use when** | Frequent updates | Static array or rare updates |

### Segment Tree vs Sparse Table

| Feature | Segment Tree | Sparse Table |
|---------|-------------|--------------|
| **Build** | O(n) | O(n log n) |
| **Query** | O(log n) | O(1) |
| **Update** | O(log n) | O(n log n) rebuild |
| **Space** | O(n) | O(n log n) |
| **Use when** | Need updates | Static RMQ, no updates |

## Interview Patterns

### Pattern 1: Basic Range Query

**Template**:
```python
class Solution:
    def rangeQuery(self, arr, queries):
        seg_tree = SegmentTree(arr)
        results = []
        for L, R in queries:
            results.append(seg_tree.query(L, R))
        return results
```

**When to use**: Multiple queries on static or infrequently changing array

### Pattern 2: Range Update Point Query

**Template**:
```python
# Use difference array + segment tree
# Or lazy segment tree
def range_update(L, R, value):
    seg_tree.range_update(L, R, value)

def point_query(index):
    return seg_tree.query(index, index)
```

**When to use**: Many range updates, query individual elements

### Pattern 3: Coordinate Compression

**Template**:
```python
def solve_with_compression(arr):
    # Compress large range to small range
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}

    compressed_arr = [compress[x] for x in arr]
    seg_tree = SegmentTree(compressed_arr)
    # ... use compressed indices
```

**When to use**: Values span large range but few unique values

### Pattern 4: Dynamic RMQ

**Template**:
```python
min_tree = MinSegmentTree(arr)

for query in queries:
    if query[0] == 'update':
        min_tree.update(query[1], query[2])
    else:  # query
        print(min_tree.query(query[1], query[2]))
```

**When to use**: Range min/max with updates

## Advanced Topics

### Persistent Segment Tree

**Idea**: Keep all previous versions of tree

```python
class PersistentSegmentTree:
    """
    Allows querying any previous version of the tree.
    Useful for time-travel queries.

    Each update creates new nodes only on update path (O(log n) space per version).
    Old nodes remain accessible.
    """

    class Node:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def update(self, root, start, end, index, value):
        """
        Create new version with update at index.
        Returns new root (old root unchanged).
        """
        if start == end:
            return self.Node(value)

        mid = (start + end) // 2
        new_node = self.Node()

        if index <= mid:
            new_node.left = self.update(root.left, start, mid, index, value)
            new_node.right = root.right  # Reuse old right subtree
        else:
            new_node.left = root.left  # Reuse old left subtree
            new_node.right = self.update(root.right, mid + 1, end, index, value)

        new_node.val = new_node.left.val + new_node.right.val
        return new_node
```

**Applications**:
- Version control systems
- Time-travel queries
- Kth smallest in range

### Dynamic Segment Tree

**Idea**: Create nodes only when needed (sparse segment tree)

```python
class DynamicSegmentTree:
    """
    For very large ranges (e.g., 10^9) but few active elements.
    Create nodes on-demand instead of pre-allocating.
    """

    class Node:
        def __init__(self):
            self.val = 0
            self.left = None
            self.right = None

    def __init__(self, L, R):
        self.root = self.Node()
        self.L = L  # Minimum index
        self.R = R  # Maximum index
```

**Use case**: When coordinate range is huge (10^9) but only few updates

### Merge Sort Tree

**Idea**: Each node stores sorted list of its range

```python
class MergeSortTree:
    """
    Each node stores sorted array of elements in its range.
    Allows counting elements in range [L, R] less than k.
    """

    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = [self.arr[start]]
            return

        # Build children
        # ...

        # Merge sorted arrays from children
        self.tree[node] = sorted(
            self.tree[left_child] + self.tree[right_child]
        )

    def count_less_than(self, L, R, k):
        """Count elements in [L, R] less than k"""
        # Use binary search in each node's sorted array
```

## Key Takeaways

1. **Use segment trees for range queries** with updates on dynamic arrays
2. **Lazy propagation** is essential for efficient range updates
3. **Space is O(4n)** for safety (or 2n for perfect binary tree)
4. **All operations O(log n)** after O(n) build
5. **Flexible aggregation** - works for any associative operation (sum, min, max, GCD, etc.)
6. **0-indexed or 1-indexed** - be consistent throughout implementation
7. **Iterative vs recursive** - recursive is more intuitive, iterative can be faster
8. **Node indexing**: left child = 2*i, right child = 2*i+1 (1-indexed) or 2*i+1, 2*i+2 (0-indexed)

## When to Use Segment Trees

✅ **Use when**:
- Need range queries (sum/min/max/GCD) on dynamic array
- Array elements change frequently (updates)
- Need range updates (with lazy propagation)
- Operation is associative but not invertible

❌ **Don't use when**:
- Array is static → use prefix sum or sparse table
- Only need point queries → use array
- Only need range sum with updates → Fenwick tree is simpler
- Need fast kth element → consider other structures (order statistics tree)

---

**Time to Implement**: 20-30 minutes (basic), 40+ minutes (with lazy propagation)

**Space Complexity**: O(n)

**Most Common Interview Uses**:
- Range sum/min/max with updates
- Count of elements satisfying property in range
- Dynamic RMQ problems

**Pro Tip**: Master the recursive implementation first, understand the three cases (complete/no/partial overlap), then learn lazy propagation for range updates.
