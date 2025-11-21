# Union-Find (Disjoint Set Union)

## Table of Contents
- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Basic Structure](#basic-structure)
- [Core Operations](#core-operations)
  - [MakeSet](#makeset)
  - [Find](#find)
  - [Union](#union)
- [Optimizations](#optimizations)
  - [Path Compression](#path-compression)
  - [Union by Rank](#union-by-rank)
  - [Union by Size](#union-by-size)
- [Time Complexity](#time-complexity)
- [Implementation](#implementation)
  - [Python Implementation](#python-implementation)
  - [JavaScript Implementation](#javascript-implementation)
- [Applications](#applications)
- [Common Problems](#common-problems)
- [Interview Patterns](#interview-patterns)
- [Advanced Topics](#advanced-topics)

## Overview

Union-Find (also called Disjoint Set Union or DSU) is a data structure that efficiently tracks a partition of a set into disjoint (non-overlapping) subsets. It provides near-constant-time operations to:
- Add new sets
- Merge sets together
- Find the representative (or "parent") of a set

### Why Union-Find?

Union-Find is crucial for problems involving:
- **Connected components** in graphs
- **Dynamic connectivity** queries
- **Cycle detection** in undirected graphs
- **Minimum spanning trees** (Kruskal's algorithm)
- **Network connectivity** problems
- **Image processing** (connected regions)

### Real-World Applications

1. **Social Networks**: Finding friend circles or groups
2. **Network Design**: Determining if nodes are connected
3. **Image Processing**: Segmenting connected regions
4. **Compiler Design**: Register allocation
5. **Game Development**: Fog of war, territory control

## Key Concepts

### Disjoint Sets

A collection of sets where no element appears in more than one set:
```
Sets: {1, 2, 3}, {4, 5}, {6, 7, 8, 9}
Element 1 belongs only to the first set
```

### Representative/Parent

Each set has a representative element (often the root of a tree):
```
Set {1, 2, 3} might have 1 as representative
Set {4, 5} might have 4 as representative
```

### Forest of Trees

Internally, Union-Find represents each set as a tree:
```
    1         4       6
   / \        |      /|\
  2   3       5     7 8 9

Three separate trees = three disjoint sets
```

## Basic Structure

The most basic implementation uses an array where each element points to its parent:

```
Index:  0  1  2  3  4  5  6  7  8  9
Parent: 0  1  1  1  4  4  6  6  6  6

Element 2's parent is 1
Element 1's parent is itself (root)
```

## Core Operations

### MakeSet

**Purpose**: Initialize a new element as its own set

**Operation**: Set parent[x] = x

**Time Complexity**: O(1)

```python
def make_set(x):
    parent[x] = x
    rank[x] = 0  # If using union by rank
```

### Find

**Purpose**: Find the representative (root) of the set containing element x

**Basic Implementation** (without optimization):
```python
def find(x):
    if parent[x] == x:
        return x
    return find(parent[x])
```

**Time Complexity** (without optimization): O(n) worst case

**Visual Example**:
```
Finding representative of 3:
    1
   / \
  2   3

3 -> parent is 1
1 -> parent is itself
Return 1 (representative)
```

### Union

**Purpose**: Merge two sets containing elements x and y

**Basic Implementation**:
```python
def union(x, y):
    root_x = find(x)
    root_y = find(y)

    if root_x != root_y:
        parent[root_x] = root_y  # Attach one tree to another
```

**Time Complexity** (without optimization): O(n) due to Find operation

**Visual Example**:
```
Before union(2, 4):
    1       3
    |       |
    2       4

After union(2, 4):
    1
    |
    2
    |
    3
    |
    4

Or (depending on which root becomes parent):
    3
   / \
  1   4
  |
  2
```

## Optimizations

### Path Compression

**Idea**: During Find operation, make every node on the path point directly to the root

**Implementation**:
```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Recursive path compression
    return parent[x]
```

**Visual Effect**:
```
Before find(4):
    1
    |
    2
    |
    3
    |
    4

After find(4):
    1
   /|\
  2 3 4
```

**Iterative Version**:
```python
def find(x):
    root = x
    while parent[root] != root:
        root = parent[root]

    # Second pass: update all nodes to point to root
    while parent[x] != root:
        next_node = parent[x]
        parent[x] = root
        x = next_node

    return root
```

**Two-Pass Path Compression** (most common in interviews):
```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Compress path
    return parent[x]
```

### Union by Rank

**Idea**: Always attach the shorter tree under the taller tree to keep trees balanced

**Rank**: Upper bound on the height of the tree

**Implementation**:
```python
def union(x, y):
    root_x = find(x)
    root_y = find(y)

    if root_x == root_y:
        return  # Already in same set

    # Attach smaller rank tree under larger rank tree
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1  # Increase rank only when equal
```

**Visual Example**:
```
Union by rank prevents this:        Creates balanced tree:
    1                                   1     3
    |                                  / \    |
    2                                 2   4   5
    |           becomes this:
    3      (bad)                     Instead of tall chain
    |
    4
    |
    5
```

### Union by Size

**Alternative**: Track size (number of nodes) instead of rank

**Implementation**:
```python
def union(x, y):
    root_x = find(x)
    root_y = find(y)

    if root_x == root_y:
        return

    # Attach smaller tree to larger tree
    if size[root_x] < size[root_y]:
        parent[root_x] = root_y
        size[root_y] += size[root_x]
    else:
        parent[root_y] = root_x
        size[root_x] += size[root_y]
```

**When to use**: Union by size is useful when you need to know set sizes

## Time Complexity

| Operation | Without Optimization | With Path Compression | With Union by Rank | With Both |
|-----------|---------------------|----------------------|-------------------|-----------|
| MakeSet   | O(1)                | O(1)                 | O(1)              | O(1)      |
| Find      | O(n)                | O(log n) amortized   | O(log n)          | O(α(n))   |
| Union     | O(n)                | O(log n) amortized   | O(log n)          | O(α(n))   |
| Connected | O(n)                | O(log n) amortized   | O(log n)          | O(α(n))   |

**α(n)**: Inverse Ackermann function - grows extremely slowly (≤ 4 for all practical n)

### Space Complexity
- O(n) for storing parent array
- O(n) for storing rank/size array (if used)
- Total: O(n)

## Implementation

### Python Implementation

**Complete Implementation with Both Optimizations**:

```python
class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.
    Efficiently maintains disjoint sets and supports union and find operations.
    """

    def __init__(self, n):
        """
        Initialize n disjoint sets {0}, {1}, ..., {n-1}

        Args:
            n: Number of elements
        """
        self.parent = list(range(n))  # parent[i] = i initially
        self.rank = [0] * n           # All trees have rank 0 initially
        self.count = n                # Number of disjoint sets

    def find(self, x):
        """
        Find the representative (root) of the set containing x.
        Uses path compression for optimization.

        Args:
            x: Element to find

        Returns:
            Representative of the set containing x

        Time Complexity: O(α(n)) amortized
        """
        if self.parent[x] != x:
            # Path compression: make x point directly to root
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Merge the sets containing x and y.
        Uses union by rank for optimization.

        Args:
            x: Element in first set
            y: Element in second set

        Returns:
            True if sets were merged, False if already in same set

        Time Complexity: O(α(n)) amortized
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank: attach smaller rank tree under larger rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1  # One fewer disjoint set
        return True

    def connected(self, x, y):
        """
        Check if x and y are in the same set.

        Args:
            x: First element
            y: Second element

        Returns:
            True if x and y are in the same set

        Time Complexity: O(α(n)) amortized
        """
        return self.find(x) == self.find(y)

    def get_count(self):
        """
        Get the number of disjoint sets.

        Returns:
            Number of disjoint sets

        Time Complexity: O(1)
        """
        return self.count


# Example usage
uf = UnionFind(10)  # Create 10 disjoint sets {0}, {1}, ..., {9}

# Perform unions
uf.union(0, 1)  # Merge {0} and {1} -> {0, 1}
uf.union(2, 3)  # Merge {2} and {3} -> {2, 3}
uf.union(0, 2)  # Merge {0, 1} and {2, 3} -> {0, 1, 2, 3}

# Check connectivity
print(uf.connected(0, 3))  # True
print(uf.connected(0, 4))  # False

# Get number of components
print(uf.get_count())  # 7 (one component with 4 elements + 6 single elements)
```

**Implementation with Union by Size**:

```python
class UnionFindBySize:
    """
    Union-Find with path compression and union by size.
    Useful when you need to track the size of each component.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n  # Each set initially has size 1
        self.count = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by size: attach smaller tree to larger tree
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]

        self.count -= 1
        return True

    def get_size(self, x):
        """Get the size of the component containing x"""
        return self.size[self.find(x)]
```

### JavaScript Implementation

**Complete Implementation**:

```javascript
class UnionFind {
    /**
     * Initialize Union-Find with n elements
     * @param {number} n - Number of elements
     */
    constructor(n) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = Array(n).fill(0);
        this.count = n;
    }

    /**
     * Find the representative of the set containing x
     * @param {number} x - Element to find
     * @returns {number} Representative of the set
     */
    find(x) {
        if (this.parent[x] !== x) {
            // Path compression
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    /**
     * Merge the sets containing x and y
     * @param {number} x - First element
     * @param {number} y - Second element
     * @returns {boolean} True if sets were merged
     */
    union(x, y) {
        const rootX = this.find(x);
        const rootY = this.find(y);

        if (rootX === rootY) {
            return false; // Already in same set
        }

        // Union by rank
        if (this.rank[rootX] < this.rank[rootY]) {
            this.parent[rootX] = rootY;
        } else if (this.rank[rootX] > this.rank[rootY]) {
            this.parent[rootY] = rootX;
        } else {
            this.parent[rootY] = rootX;
            this.rank[rootX]++;
        }

        this.count--;
        return true;
    }

    /**
     * Check if x and y are in the same set
     * @param {number} x - First element
     * @param {number} y - Second element
     * @returns {boolean} True if in same set
     */
    connected(x, y) {
        return this.find(x) === this.find(y);
    }

    /**
     * Get the number of disjoint sets
     * @returns {number} Number of components
     */
    getCount() {
        return this.count;
    }
}

// Example usage
const uf = new UnionFind(10);

uf.union(0, 1);
uf.union(2, 3);
uf.union(0, 2);

console.log(uf.connected(0, 3)); // true
console.log(uf.connected(0, 4)); // false
console.log(uf.getCount());      // 7
```

**TypeScript Version with Types**:

```typescript
class UnionFind {
    private parent: number[];
    private rank: number[];
    private count: number;

    constructor(n: number) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = Array(n).fill(0);
        this.count = n;
    }

    find(x: number): number {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    union(x: number, y: number): boolean {
        const rootX = this.find(x);
        const rootY = this.find(y);

        if (rootX === rootY) return false;

        if (this.rank[rootX] < this.rank[rootY]) {
            this.parent[rootX] = rootY;
        } else if (this.rank[rootX] > this.rank[rootY]) {
            this.parent[rootY] = rootX;
        } else {
            this.parent[rootY] = rootX;
            this.rank[rootX]++;
        }

        this.count--;
        return true;
    }

    connected(x: number, y: number): boolean {
        return this.find(x) === this.find(y);
    }

    getCount(): number {
        return this.count;
    }
}
```

## Applications

### 1. Connected Components in Graphs

**Problem**: Find the number of connected components in an undirected graph

```python
def count_components(n, edges):
    """
    Count connected components in undirected graph.

    Args:
        n: Number of nodes (0 to n-1)
        edges: List of edges [(u, v), ...]

    Returns:
        Number of connected components
    """
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    return uf.get_count()

# Example
edges = [(0, 1), (1, 2), (3, 4)]
print(count_components(5, edges))  # 2 components: {0,1,2} and {3,4}
```

### 2. Cycle Detection in Undirected Graph

**Problem**: Detect if an undirected graph has a cycle

```python
def has_cycle(n, edges):
    """
    Detect cycle in undirected graph.

    If we try to union two nodes that are already connected,
    adding that edge would create a cycle.
    """
    uf = UnionFind(n)

    for u, v in edges:
        if uf.connected(u, v):
            return True  # Found cycle
        uf.union(u, v)

    return False

# Example
edges = [(0, 1), (1, 2), (2, 0)]  # Forms a triangle (cycle)
print(has_cycle(3, edges))  # True
```

### 3. Kruskal's Minimum Spanning Tree

**Problem**: Find minimum spanning tree using Kruskal's algorithm

```python
def kruskal_mst(n, edges):
    """
    Find minimum spanning tree using Kruskal's algorithm.

    Args:
        n: Number of vertices
        edges: List of (weight, u, v) tuples

    Returns:
        Total weight of MST and list of edges in MST
    """
    uf = UnionFind(n)
    edges.sort()  # Sort by weight

    mst_weight = 0
    mst_edges = []

    for weight, u, v in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst_weight += weight
            mst_edges.append((u, v, weight))

    return mst_weight, mst_edges

# Example
edges = [
    (1, 0, 1),  # (weight, u, v)
    (2, 0, 2),
    (3, 1, 2),
    (4, 1, 3),
    (5, 2, 3)
]
weight, mst = kruskal_mst(4, edges)
print(f"MST weight: {weight}")  # 7
print(f"MST edges: {mst}")      # [(0,1), (0,2), (1,3)]
```

### 4. Friend Circles / Social Networks

**Problem**: Find number of friend circles

```python
def find_circle_num(is_friend):
    """
    Find number of friend circles.

    Args:
        is_friend: n x n matrix where is_friend[i][j] = 1 if i and j are friends

    Returns:
        Number of friend circles
    """
    n = len(is_friend)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if is_friend[i][j] == 1:
                uf.union(i, j)

    return uf.get_count()

# Example: LeetCode 547
friends = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
]
print(find_circle_num(friends))  # 2 circles
```

### 5. Account Merge

**Problem**: Merge accounts belonging to the same person

```python
def accounts_merge(accounts):
    """
    Merge accounts that share at least one email.

    LeetCode 721: Accounts Merge
    """
    uf = UnionFind(len(accounts))
    email_to_id = {}

    # Build email to account mapping
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            else:
                email_to_id[email] = i

    # Group emails by root account
    root_to_emails = {}
    for email, acc_id in email_to_id.items():
        root = uf.find(acc_id)
        if root not in root_to_emails:
            root_to_emails[root] = []
        root_to_emails[root].append(email)

    # Build result
    result = []
    for root, emails in root_to_emails.items():
        name = accounts[root][0]
        result.append([name] + sorted(emails))

    return result
```

### 6. Redundant Connection

**Problem**: Find the edge that can be removed to make tree

```python
def find_redundant_connection(edges):
    """
    Find edge that creates a cycle (can be removed to form tree).

    LeetCode 684: Redundant Connection
    """
    uf = UnionFind(len(edges) + 1)

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates a cycle

    return []
```

## Common Problems

### LeetCode Problems Using Union-Find

| Problem | Difficulty | Key Concept |
|---------|-----------|-------------|
| [547. Friend Circles](https://leetcode.com/problems/number-of-provinces/) | Medium | Connected components |
| [684. Redundant Connection](https://leetcode.com/problems/redundant-connection/) | Medium | Cycle detection |
| [685. Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/) | Hard | Directed graph cycles |
| [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/) | Medium | Component grouping |
| [737. Sentence Similarity II](https://leetcode.com/problems/sentence-similarity-ii/) | Medium | Transitive relationships |
| [765. Couples Holding Hands](https://leetcode.com/problems/couples-holding-hands/) | Hard | Permutation cycles |
| [803. Bricks Falling When Hit](https://leetcode.com/problems/bricks-falling-when-hit/) | Hard | Reverse time union-find |
| [952. Largest Component Size by Common Factor](https://leetcode.com/problems/largest-component-size-by-common-factor/) | Hard | Mathematical grouping |
| [1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) | Medium | MST (Kruskal's) |
| [1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/) | Medium | Connected components |

## Interview Patterns

### Pattern 1: Count Components

**Template**:
```python
def count_components(n, connections):
    uf = UnionFind(n)
    for u, v in connections:
        uf.union(u, v)
    return uf.get_count()
```

**When to use**: Any problem asking for number of groups, clusters, or components

### Pattern 2: Cycle Detection

**Template**:
```python
def has_cycle(edges):
    uf = UnionFind(max_node + 1)
    for u, v in edges:
        if uf.connected(u, v):
            return True  # Cycle found
        uf.union(u, v)
    return False
```

**When to use**: Detecting cycles in undirected graphs

### Pattern 3: Minimum Spanning Tree

**Template**:
```python
def mst(n, edges):
    edges.sort(key=lambda x: x[0])  # Sort by weight
    uf = UnionFind(n)
    total_weight = 0

    for weight, u, v in edges:
        if uf.union(u, v):
            total_weight += weight

    return total_weight
```

**When to use**: Finding minimum cost to connect all nodes

### Pattern 4: Grouping by Property

**Template**:
```python
def group_by_property(items):
    uf = UnionFind(len(items))
    property_to_id = {}

    for i, item in enumerate(items):
        for prop in get_properties(item):
            if prop in property_to_id:
                uf.union(i, property_to_id[prop])
            else:
                property_to_id[prop] = i

    # Collect groups...
```

**When to use**: Merging items that share common properties (accounts, emails, etc.)

### Pattern 5: Largest Component

**Template**:
```python
def largest_component(n, edges):
    uf = UnionFindBySize(n)
    for u, v in edges:
        uf.union(u, v)

    return max(uf.size)
```

**When to use**: Finding the size of the largest connected component

## Advanced Topics

### Offline Queries with Union-Find

Process queries in reverse order to handle deletions:

```python
def process_with_deletions(n, edges, deletions):
    """
    Handle edge deletions by processing in reverse.
    Add edges in reverse order of deletion.
    """
    edge_set = set(map(tuple, edges))
    delete_set = set(map(tuple, deletions))

    # Start with edges not deleted
    remaining = edge_set - delete_set
    uf = UnionFind(n)

    for u, v in remaining:
        uf.union(u, v)

    results = []

    # Process deletions in reverse (= additions)
    for u, v in reversed(deletions):
        results.append(uf.get_count())
        uf.union(u, v)

    return list(reversed(results))
```

### Weighted Union-Find

Track additional information like distances:

```python
class WeightedUnionFind:
    """
    Union-Find that tracks relative weights/distances between elements.
    Useful for problems involving relationships with values.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.weight = [0] * n  # Weight relative to parent

    def find(self, x):
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.weight[x] += self.weight[original_parent]
        return self.parent[x]

    def union(self, x, y, w):
        """
        Union with relationship: weight[x] + w = weight[y]
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Calculate weight from root_x to root_y
        self.parent[root_x] = root_y
        self.weight[root_x] = self.weight[y] - self.weight[x] - w
```

### Persistent Union-Find

Maintain version history for undo operations:

```python
class PersistentUnionFind:
    """
    Supports rollback to previous states.
    Useful for backtracking algorithms.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # Stack of (x, old_parent, old_rank, y, old_rank_y)

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        # Save state for rollback
        self.history.append((
            root_y,
            self.parent[root_y],
            root_x,
            self.rank[root_x]
        ))

        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

    def rollback(self):
        """Undo last union operation"""
        if not self.history:
            return

        child, old_parent, parent, old_rank = self.history.pop()
        self.parent[child] = old_parent
        self.rank[parent] = old_rank
```

## Key Takeaways

1. **Use both optimizations**: Path compression + union by rank for near-constant time
2. **Remember amortized complexity**: O(α(n)) is effectively O(1) in practice
3. **Track component count**: Decrement count on each successful union
4. **Union returns boolean**: Useful for cycle detection (returns False if already connected)
5. **Choice of union by rank vs size**: Use rank for general case, size when you need component sizes
6. **Common interview pattern**: Initialize UF, process edges/relationships, query result
7. **Not for directed graphs**: Union-Find works for undirected relationships only

## Related Data Structures

- **Segment Trees**: For range queries with updates
- **Disjoint Set Forests**: Alternative name for Union-Find
- **Link-Cut Trees**: Dynamic tree connectivity with more operations
- **Tarjan's Offline LCA**: Uses Union-Find for lowest common ancestor queries

---

**Time to Implement**: 5-10 minutes (in interview setting)

**Space Complexity**: O(n) for parent and rank arrays

**When to Use**: Problems involving connectivity, grouping, or equivalence relations

**When NOT to Use**: Directed graphs, shortest paths, need to enumerate members of a set
