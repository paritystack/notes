# Sparse Tables

## Table of Contents
- [Overview](#overview)
- [The Problem It Solves](#the-problem-it-solves)
- [How It Works](#how-it-works)
- [Key Concepts](#key-concepts)
- [Structure and Properties](#structure-and-properties)
- [Core Operations](#core-operations)
  - [Build](#build-operation)
  - [Query](#query-operation)
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

A **Sparse Table** is a data structure that answers **static range queries** in O(1) time after O(n log n) preprocessing. It's particularly efficient for **idempotent operations** like minimum, maximum, GCD, and LCM.

### Key Characteristics

- **Static**: Array doesn't change after construction
- **Fast queries**: O(1) query time
- **Preprocessing**: O(n log n) build time
- **Space**: O(n log n) memory
- **Idempotent operations**: Works best with min, max, GCD, etc.
- **No updates**: Cannot modify array after building

### Why Sparse Tables?

**Best for:**
- Range Minimum Query (RMQ) / Range Maximum Query
- Static arrays with many queries
- When O(1) query time is critical
- Operations where overlapping ranges are okay

**Advantages:**
- Faster queries than segment tree (O(1) vs O(log n))
- Simpler than segment tree for RMQ
- No updates means no balancing needed

**Real-world usage:**
- Lowest Common Ancestor (LCA) in trees
- Range statistics on static data
- Computational geometry
- Database query optimization

## The Problem It Solves

### Range Query Problem

**Given:**
- Static array A of n elements
- Many range queries: find min/max/GCD of A[L..R]

**Approaches:**

| Approach | Build | Query | Update | Space |
|----------|-------|-------|--------|-------|
| Naive | O(1) | O(n) | O(1) | O(n) |
| Segment Tree | O(n) | O(log n) | O(log n) | O(n) |
| **Sparse Table** | **O(n log n)** | **O(1)** | **N/A** | **O(n log n)** |

**Key insight**: If array is static, we can precompute all useful range information!

### Example Problem

**Range Minimum Query (RMQ):**

```
Array: [3, 1, 4, 1, 5, 9, 2, 6]

Query RMQ(2, 5): min of [4, 1, 5, 9] = 1
Query RMQ(0, 7): min of [3, 1, 4, 1, 5, 9, 2, 6] = 1
Query RMQ(4, 6): min of [5, 9, 2] = 2
```

**With sparse table**: All queries answered in O(1)!

## How It Works

### The Core Idea

**Key observation**: Any range [L, R] can be covered by at most 2 overlapping ranges of length 2^k.

```
For range [L, R]:
- Find k where 2^k ≤ (R - L + 1) < 2^(k+1)
- Split into: [L, L+2^k-1] and [R-2^k+1, R]
- These ranges overlap, but that's okay for min/max!

Example: RMQ(1, 6) with length 6
k = 2 (since 2^2 = 4 ≤ 6 < 8 = 2^3)
Range 1: [1, 4] (length 4)
Range 2: [3, 6] (length 4)
min([1,4]) and min([3,6]) = overall min
```

### Preprocessing Table

Build 2D table where:
```
table[i][j] = result of operation on range [i, i + 2^j - 1]

Example for min operation:
table[0][0] = A[0]           (range [0, 0], length 1)
table[0][1] = min(A[0], A[1]) (range [0, 1], length 2)
table[0][2] = min(A[0..3])    (range [0, 3], length 4)
table[0][3] = min(A[0..7])    (range [0, 7], length 8)
```

### Visual Example

```
Array: [3, 1, 4, 1, 5, 9, 2, 6]

Sparse Table (minimum):

j=0 (length 1):  [3, 1, 4, 1, 5, 9, 2, 6]
j=1 (length 2):  [1, 1, 1, 1, 5, 2, 2]
j=2 (length 4):  [1, 1, 1, 1, 2]
j=3 (length 8):  [1]

table[i][j] = minimum of range [i, i+2^j-1]

Example:
table[0][0] = 3           (min of [3])
table[0][1] = min(3,1)=1  (min of [3,1])
table[0][2] = min(3,1,4,1)=1 (min of [3,1,4,1])
table[2][1] = min(4,1)=1  (min of [4,1])
```

## Key Concepts

### 1. Power of 2 Ranges

**Why powers of 2?**
- Any number can be represented as sum of powers of 2 (binary)
- Any range can be covered by at most 2 ranges of length 2^k
- Enables O(1) queries through precomputation

### 2. Idempotent Operations

**Definition**: Operation f is idempotent if f(a, a) = a

**Examples:**
- min(a, a) = a ✓
- max(a, a) = a ✓
- gcd(a, a) = a ✓
- sum(a, a) = 2a ✗ (not idempotent)

**Why important?**
- Overlapping ranges okay for idempotent operations
- Enables O(1) query by using 2 precomputed ranges

### 3. Range Decomposition

**Key formula for query [L, R]:**

```
k = floor(log₂(R - L + 1))
result = operation(table[L][k], table[R - 2^k + 1][k])
```

**Example:**
```
Query RMQ(2, 7):
Length = 7 - 2 + 1 = 6
k = floor(log₂(6)) = 2  (2^2 = 4)

min([2, 7]) = min(table[2][2], table[4][2])
            = min(min([2,5]), min([4,7]))
            = min(1, 2) = 1
```

### 4. Logarithm Precomputation

For fast queries, precompute log₂ values:

```python
log_table = [0] * (n + 1)
for i in range(2, n + 1):
    log_table[i] = log_table[i // 2] + 1
```

Avoids expensive log computation during queries.

## Structure and Properties

### Table Structure

**2D Array:**
```
table[i][j] = operation result for range [i, i + 2^j - 1]

Dimensions:
- i: 0 to n-1 (starting position)
- j: 0 to log₂(n) (power of 2)

Size: O(n log n)
```

### Mathematical Properties

**Recurrence relation:**
```
table[i][0] = A[i]  (base case: single element)

table[i][j] = operation(table[i][j-1], table[i + 2^(j-1)][j-1])

Explanation:
Range [i, i+2^j-1] splits into:
- Left half: [i, i+2^(j-1)-1]
- Right half: [i+2^(j-1), i+2^j-1]
```

### Example Construction

```
Array: [3, 1, 4, 1, 5]

Step 1: j=0 (base case)
table[0][0] = 3
table[1][0] = 1
table[2][0] = 4
table[3][0] = 1
table[4][0] = 5

Step 2: j=1 (length 2)
table[0][1] = min(table[0][0], table[1][0]) = min(3, 1) = 1
table[1][1] = min(table[1][0], table[2][0]) = min(1, 4) = 1
table[2][1] = min(table[2][0], table[3][0]) = min(4, 1) = 1
table[3][1] = min(table[3][0], table[4][0]) = min(1, 5) = 1

Step 3: j=2 (length 4)
table[0][2] = min(table[0][1], table[2][1]) = min(1, 1) = 1
table[1][2] = min(table[1][1], table[3][1]) = min(1, 1) = 1
```

## Core Operations

### Build Operation

**Goal**: Construct sparse table from array

**Algorithm:**
```python
def build_sparse_table(arr):
    n = len(arr)
    max_log = int(math.log2(n)) + 1

    # Initialize table
    table = [[0] * max_log for _ in range(n)]

    # Base case: j = 0 (single elements)
    for i in range(n):
        table[i][0] = arr[i]

    # Fill table using DP
    j = 1
    while (1 << j) <= n:  # 2^j <= n
        i = 0
        while (i + (1 << j)) <= n:  # i + 2^j <= n
            table[i][j] = min(
                table[i][j - 1],
                table[i + (1 << (j - 1))][j - 1]
            )
            i += 1
        j += 1

    return table
```

**Step-by-step:**
1. Determine maximum power of 2 needed: ⌊log₂(n)⌋
2. Initialize 2D table: table[n][log n]
3. Fill base case: copy array elements
4. Fill using DP: combine two halves for each range

**Time Complexity**: O(n log n)
- Outer loop: O(log n) levels
- Inner loop: O(n) per level
- Total: O(n log n)

**Space Complexity**: O(n log n)

### Query Operation

**Goal**: Answer range query [L, R] in O(1)

**Algorithm:**
```python
def query(table, log_table, L, R):
    # Find k where 2^k ≤ (R - L + 1)
    k = log_table[R - L + 1]

    # Query two overlapping ranges
    return min(
        table[L][k],
        table[R - (1 << k) + 1][k]
    )
```

**Visualization:**
```
Query RMQ(2, 7):
        0   1   2   3   4   5   6   7
Array: [3,  1,  4,  1,  5,  9,  2,  6]
                    L           R

Length = 7 - 2 + 1 = 6
k = floor(log₂(6)) = 2

Range 1: [2, 5] (start at L, length 4)
Range 2: [4, 7] (end at R, length 4)

           [2, 3, 4, 5]
                   [4, 5, 6, 7]
                ^^^^^^^ overlap

min(table[2][2], table[4][2])
= min(min([2,5]), min([4,7]))
= min(1, 2) = 1
```

**Time Complexity**: O(1)
- Logarithm lookup: O(1) (precomputed)
- Two array accesses: O(1)
- Min operation: O(1)

## Time and Space Complexity

### Summary Table

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Build | O(n log n) | O(n log n) | One-time preprocessing |
| Query | O(1) | - | Idempotent operations only |
| Update | N/A | - | Static structure |
| Space | - | O(n log n) | ~n × log₂(n) cells |

### Space Breakdown

For array of size n:
- Rows: n
- Columns: ⌊log₂(n)⌋ + 1
- Total cells: n × (⌊log₂(n)⌋ + 1)

**Examples:**
- n = 100: 100 × 7 = 700 cells
- n = 1,000: 1,000 × 10 = 10,000 cells
- n = 1,000,000: 1,000,000 × 20 = 20,000,000 cells

**Memory:**
- With int (4 bytes): ~4n log n bytes
- n = 100,000: ~8 MB

## Implementation

### Python Implementation

**Complete Sparse Table for RMQ:**

```python
import math

class SparseTable:
    """
    Sparse Table for Range Minimum Query (RMQ).

    Supports:
    - O(n log n) build time
    - O(1) query time
    - Static array (no updates)

    Can be adapted for max, GCD, LCM by changing operation.
    """

    def __init__(self, arr):
        """
        Build sparse table from array.

        Args:
            arr (list): Input array

        Time: O(n log n)
        Space: O(n log n)
        """
        self.n = len(arr)
        self.max_log = int(math.log2(self.n)) + 1

        # Build table
        self.table = [[0] * self.max_log for _ in range(self.n)]
        self._build(arr)

        # Precompute logarithms for O(1) query
        self.log_table = self._build_log_table()

    def _build(self, arr):
        """Build sparse table using dynamic programming"""
        # Base case: single elements
        for i in range(self.n):
            self.table[i][0] = arr[i]

        # Build table level by level
        j = 1
        while (1 << j) <= self.n:  # 2^j <= n
            i = 0
            while (i + (1 << j)) <= self.n:
                # Combine two halves
                left = self.table[i][j - 1]
                right = self.table[i + (1 << (j - 1))][j - 1]
                self.table[i][j] = min(left, right)  # Change to max for RMaxQ
                i += 1
            j += 1

    def _build_log_table(self):
        """Precompute floor(log2(i)) for all i"""
        log_table = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            log_table[i] = log_table[i // 2] + 1
        return log_table

    def query(self, left, right):
        """
        Get minimum value in range [left, right].

        Args:
            left (int): Left boundary (inclusive)
            right (int): Right boundary (inclusive)

        Returns:
            Minimum value in range

        Time: O(1)
        """
        # Find k where 2^k <= length
        length = right - left + 1
        k = self.log_table[length]

        # Query two overlapping ranges
        left_min = self.table[left][k]
        right_min = self.table[right - (1 << k) + 1][k]

        return min(left_min, right_min)

    def __str__(self):
        """String representation for debugging"""
        result = "Sparse Table:\n"
        for j in range(self.max_log):
            if (1 << j) > self.n:
                break
            result += f"j={j} (length {1 << j}): "
            for i in range(self.n):
                if i + (1 << j) <= self.n:
                    result += f"{self.table[i][j]} "
            result += "\n"
        return result


# Example usage
if __name__ == "__main__":
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

    st = SparseTable(arr)

    print("Original array:", arr)
    print("\n" + str(st))

    # Test queries
    queries = [(0, 4), (2, 7), (5, 9), (1, 8)]

    for left, right in queries:
        result = st.query(left, right)
        actual = min(arr[left:right+1])
        print(f"RMQ({left}, {right}): {result} (expected: {actual})")
```

**Sparse Table for Range Maximum Query:**

```python
class SparseTableMax:
    """Sparse Table for Range Maximum Query"""

    def __init__(self, arr):
        self.n = len(arr)
        self.max_log = int(math.log2(self.n)) + 1
        self.table = [[0] * self.max_log for _ in range(self.n)]
        self.log_table = [0] * (self.n + 1)

        # Build log table
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1

        # Build sparse table
        for i in range(self.n):
            self.table[i][0] = arr[i]

        j = 1
        while (1 << j) <= self.n:
            for i in range(self.n - (1 << j) + 1):
                left = self.table[i][j - 1]
                right = self.table[i + (1 << (j - 1))][j - 1]
                self.table[i][j] = max(left, right)  # Max instead of min
            j += 1

    def query(self, left, right):
        """Get maximum value in range [left, right]"""
        k = self.log_table[right - left + 1]
        return max(
            self.table[left][k],
            self.table[right - (1 << k) + 1][k]
        )
```

**Sparse Table for GCD:**

```python
import math

class SparseTableGCD:
    """Sparse Table for Range GCD Query"""

    def __init__(self, arr):
        self.n = len(arr)
        self.max_log = int(math.log2(self.n)) + 1
        self.table = [[0] * self.max_log for _ in range(self.n)]
        self.log_table = [0] * (self.n + 1)

        # Build log table
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1

        # Build sparse table
        for i in range(self.n):
            self.table[i][0] = arr[i]

        j = 1
        while (1 << j) <= self.n:
            for i in range(self.n - (1 << j) + 1):
                left = self.table[i][j - 1]
                right = self.table[i + (1 << (j - 1))][j - 1]
                self.table[i][j] = math.gcd(left, right)
            j += 1

    def query(self, left, right):
        """Get GCD of range [left, right]"""
        k = self.log_table[right - left + 1]
        return math.gcd(
            self.table[left][k],
            self.table[right - (1 << k) + 1][k]
        )


# Example
arr = [12, 18, 24, 36, 6]
st_gcd = SparseTableGCD(arr)
print(st_gcd.query(0, 4))  # GCD of [12, 18, 24, 36, 6] = 6
print(st_gcd.query(0, 2))  # GCD of [12, 18, 24] = 6
```

### JavaScript Implementation

```javascript
class SparseTable {
    constructor(arr) {
        this.n = arr.length;
        this.maxLog = Math.floor(Math.log2(this.n)) + 1;

        // Build table
        this.table = Array.from({length: this.n}, () =>
            new Array(this.maxLog).fill(0)
        );
        this.build(arr);

        // Build log table
        this.logTable = new Array(this.n + 1).fill(0);
        for (let i = 2; i <= this.n; i++) {
            this.logTable[i] = this.logTable[Math.floor(i / 2)] + 1;
        }
    }

    build(arr) {
        // Base case
        for (let i = 0; i < this.n; i++) {
            this.table[i][0] = arr[i];
        }

        // Fill table
        for (let j = 1; (1 << j) <= this.n; j++) {
            for (let i = 0; i + (1 << j) <= this.n; i++) {
                const left = this.table[i][j - 1];
                const right = this.table[i + (1 << (j - 1))][j - 1];
                this.table[i][j] = Math.min(left, right);
            }
        }
    }

    query(left, right) {
        const k = this.logTable[right - left + 1];
        const leftMin = this.table[left][k];
        const rightMin = this.table[right - (1 << k) + 1][k];
        return Math.min(leftMin, rightMin);
    }
}

// Example usage
const arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
const st = new SparseTable(arr);

console.log(`RMQ(2, 7): ${st.query(2, 7)}`);  // 1
console.log(`RMQ(0, 4): ${st.query(0, 4)}`);  // 1
console.log(`RMQ(5, 9): ${st.query(5, 9)}`);  // 2
```

## Variations

### 1. 2D Sparse Table

**For 2D range queries** (rectangle minimum):

```python
class SparseTable2D:
    """2D Sparse Table for rectangle minimum queries"""

    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.log_r = int(math.log2(self.rows)) + 1
        self.log_c = int(math.log2(self.cols)) + 1

        # 4D table: [row][col][log_row][log_col]
        self.table = [[[[0 for _ in range(self.log_c)]
                        for _ in range(self.log_r)]
                       for _ in range(self.cols)]
                      for _ in range(self.rows)]

        self._build(matrix)

    def _build(self, matrix):
        """Build 2D sparse table"""
        # Base case: single cells
        for i in range(self.rows):
            for j in range(self.cols):
                self.table[i][j][0][0] = matrix[i][j]

        # Fill for different rectangle sizes
        for log_r in range(self.log_r):
            for log_c in range(self.log_c):
                if log_r == 0 and log_c == 0:
                    continue

                h = 1 << log_r
                w = 1 << log_c

                for i in range(self.rows - h + 1):
                    for j in range(self.cols - w + 1):
                        # Combine 4 sub-rectangles
                        if log_c > 0:
                            left = self.table[i][j][log_r][log_c - 1]
                            right = self.table[i][j + (w // 2)][log_r][log_c - 1]
                            self.table[i][j][log_r][log_c] = min(left, right)
                        else:
                            top = self.table[i][j][log_r - 1][log_c]
                            bottom = self.table[i + (h // 2)][j][log_r - 1][log_c]
                            self.table[i][j][log_r][log_c] = min(top, bottom)

    def query(self, r1, c1, r2, c2):
        """Query minimum in rectangle (r1,c1) to (r2,c2)"""
        log_r = int(math.log2(r2 - r1 + 1))
        log_c = int(math.log2(c2 - c1 + 1))

        # Query 4 overlapping rectangles
        tl = self.table[r1][c1][log_r][log_c]
        tr = self.table[r1][c2 - (1 << log_c) + 1][log_r][log_c]
        bl = self.table[r2 - (1 << log_r) + 1][c1][log_r][log_c]
        br = self.table[r2 - (1 << log_r) + 1][c2 - (1 << log_c) + 1][log_r][log_c]

        return min(tl, tr, bl, br)
```

### 2. Sparse Table with Index

**Return both value and index:**

```python
class SparseTableWithIndex:
    """Sparse table that returns minimum value and its index"""

    def __init__(self, arr):
        self.n = len(arr)
        self.arr = arr
        self.max_log = int(math.log2(self.n)) + 1
        # Store indices instead of values
        self.table = [[0] * self.max_log for _ in range(self.n)]
        self.log_table = [0] * (self.n + 1)

        # Build
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1

        for i in range(self.n):
            self.table[i][0] = i  # Store index

        j = 1
        while (1 << j) <= self.n:
            for i in range(self.n - (1 << j) + 1):
                left_idx = self.table[i][j - 1]
                right_idx = self.table[i + (1 << (j - 1))][j - 1]

                # Store index of minimum
                if arr[left_idx] <= arr[right_idx]:
                    self.table[i][j] = left_idx
                else:
                    self.table[i][j] = right_idx
            j += 1

    def query(self, left, right):
        """Return (min_value, min_index)"""
        k = self.log_table[right - left + 1]

        left_idx = self.table[left][k]
        right_idx = self.table[right - (1 << k) + 1][k]

        if self.arr[left_idx] <= self.arr[right_idx]:
            min_idx = left_idx
        else:
            min_idx = right_idx

        return self.arr[min_idx], min_idx
```

### 3. Disjoint Sparse Table

**Better cache locality, same complexity:**

Divide array into disjoint blocks and build sparse table within each.

## Applications

### 1. Range Minimum/Maximum Query

**Classic application:**

```python
def solve_rmq_queries(arr, queries):
    """
    Answer multiple RMQ queries efficiently.

    Args:
        arr: Static array
        queries: List of (left, right) range queries

    Returns:
        List of minimum values for each query

    Time: O(n log n + q) where q = number of queries
    """
    st = SparseTable(arr)

    results = []
    for left, right in queries:
        results.append(st.query(left, right))

    return results
```

### 2. Lowest Common Ancestor (LCA)

**Using Euler tour + RMQ:**

```python
class LCA:
    """
    Lowest Common Ancestor using sparse table.
    Convert tree LCA to RMQ problem.
    """

    def __init__(self, tree, root):
        """
        Build LCA structure.

        Args:
            tree: Adjacency list representation
            root: Root node
        """
        self.tree = tree
        self.n = len(tree)

        # Euler tour: nodes visited in DFS
        self.euler = []
        # First occurrence of each node in euler tour
        self.first = [-1] * self.n
        # Depth of each node in tour
        self.depth = []

        # Run DFS to build euler tour
        self._dfs(root, -1, 0)

        # Build sparse table on depths
        self.st = SparseTableWithIndex(self.depth)

    def _dfs(self, node, parent, d):
        """DFS to build Euler tour"""
        self.first[node] = len(self.euler)
        self.euler.append(node)
        self.depth.append(d)

        for child in self.tree[node]:
            if child != parent:
                self._dfs(child, node, d + 1)
                self.euler.append(node)
                self.depth.append(d)

    def query(self, u, v):
        """
        Find LCA of nodes u and v.

        Time: O(1) after O(n log n) preprocessing
        """
        left = min(self.first[u], self.first[v])
        right = max(self.first[u], self.first[v])

        _, min_idx = self.st.query(left, right)
        return self.euler[min_idx]


# Example usage
tree = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1],
    4: [1],
    5: [2]
}

lca = LCA(tree, root=0)
print(f"LCA(3, 4): {lca.query(3, 4)}")  # 1
print(f"LCA(3, 5): {lca.query(3, 5)}")  # 0
```

### 3. Sliding Window Minimum/Maximum

**Fixed-size window queries:**

```python
def sliding_window_minimum(arr, k):
    """
    Find minimum in each window of size k.

    Args:
        arr: Input array
        k: Window size

    Returns:
        List of minimums for each window

    Time: O(n log n + n) = O(n log n)
    """
    st = SparseTable(arr)

    result = []
    for i in range(len(arr) - k + 1):
        result.append(st.query(i, i + k - 1))

    return result


# Example
arr = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(sliding_window_minimum(arr, k))  # [-1, -3, -3, -3, 3, 3]
```

### 4. Range GCD Queries

```python
def range_gcd_queries(arr, queries):
    """Answer multiple range GCD queries"""
    st = SparseTableGCD(arr)

    results = []
    for left, right in queries:
        results.append(st.query(left, right))

    return results
```

### 5. Longest Increasing Subsequence (LIS) Queries

**Query LIS in subarray [L, R]:**

Combine sparse table with additional preprocessing.

## Common Problems

### LeetCode Problems

| Problem | Difficulty | Technique |
|---------|-----------|-----------|
| [1606. Find Servers That Handled Most Number of Requests](https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/) | Hard | Range queries |
| [1851. Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/) | Hard | Sparse table + sorting |
| [1938. Maximum Genetic Difference Query](https://leetcode.com/problems/maximum-genetic-difference-query/) | Hard | Tree queries |

### Competitive Programming

Common patterns:
1. **Static RMQ with many queries**
2. **LCA in trees**
3. **2D matrix minimum in rectangles**
4. **Range GCD/LCM queries**

## Comparison with Other Structures

### Sparse Table vs Segment Tree

| Feature | Sparse Table | Segment Tree |
|---------|--------------|--------------|
| **Query time** | O(1) | O(log n) |
| **Build time** | O(n log n) | O(n) |
| **Update** | N/A (static) | O(log n) |
| **Space** | O(n log n) | O(n) |
| **Operations** | Idempotent only | Any associative |
| **Implementation** | Simpler | More complex |
| **Use when** | Static data | Dynamic data |

**Choose Sparse Table when:**
- Array is static (no updates)
- Need O(1) queries
- Operation is idempotent
- Space O(n log n) acceptable

**Choose Segment Tree when:**
- Need updates
- Non-idempotent operations (sum)
- Space is limited
- O(log n) query acceptable

### Sparse Table vs Square Root Decomposition

| Feature | Sparse Table | Sqrt Decomposition |
|---------|--------------|-------------------|
| **Query** | O(1) | O(√n) |
| **Build** | O(n log n) | O(n) |
| **Space** | O(n log n) | O(n) |
| **Update** | N/A | O(1) or O(√n) |
| **Simplicity** | Moderate | Very simple |

### Sparse Table vs Prefix Arrays

| Feature | Sparse Table | Prefix Sum |
|---------|--------------|------------|
| **Query** | O(1) | O(1) |
| **Build** | O(n log n) | O(n) |
| **Operations** | Min, max, GCD | Sum (invertible) |
| **Space** | O(n log n) | O(n) |

**Use prefix arrays** for sum queries (simpler).
**Use sparse table** for min/max/GCD queries.

## Interview Patterns

### Pattern 1: Static RMQ

```python
def solve_static_rmq(arr, queries):
    """
    Multiple queries on static array.
    Build once, query many times.
    """
    st = SparseTable(arr)
    return [st.query(l, r) for l, r in queries]
```

### Pattern 2: LCA Problems

Convert tree problems to RMQ using Euler tour technique.

### Pattern 3: Sliding Window Optimization

Use sparse table when window queries are complex.

## Advanced Topics

### Binary Lifting

**Related technique** for tree problems:
- Similar idea of power-of-2 jumps
- Used in LCA, kth ancestor queries

### Range Mode Query

**Harder problem**: Most frequent element in range
- Cannot use sparse table directly (not idempotent)
- Need modified approach

### Fractional Cascading

**Optimization** for multiple sparse tables:
- Share common structure
- Speed up queries further

## Key Takeaways

1. **Sparse tables = O(1) RMQ** on static arrays
2. **Precompute power-of-2 ranges** in O(n log n)
3. **Overlap okay for idempotent** operations (min, max, GCD)
4. **Not for sum** (not idempotent - overlaps count twice)
5. **Space O(n log n)** but queries O(1)
6. **Classic application: LCA** using Euler tour
7. **Better than segment tree** when no updates needed
8. **Simple to implement** (~50-60 lines)

## When to Use Sparse Tables

✅ **Use when:**
- Array is static (no updates)
- Many range queries needed
- Operation is idempotent (min, max, GCD, LCM)
- O(1) query time required
- LCA queries in trees

❌ **Don't use when:**
- Need updates (use segment tree)
- Operation not idempotent like sum (use prefix sums or segment tree)
- Space O(n log n) too large
- Few queries (not worth preprocessing)

---

**Time to Implement**: 20-30 minutes

**Space Complexity**: O(n log n)

**Most Common Interview Uses**:
- Range minimum/maximum queries
- Lowest common ancestor (LCA)
- Static range statistics
- Competitive programming RMQ

**Pro Tip**: Sparse tables are perfect when you need blazing-fast queries on static data. Remember: overlapping ranges are okay only for idempotent operations! For sum queries, use prefix sums instead. Master the O(1) query technique - it's a game-changer for many problems!
