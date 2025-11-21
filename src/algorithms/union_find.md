# Union-Find (Disjoint Set Union)

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use Union-Find](#when-to-use-union-find)
  - [Key Operations](#key-operations)
- [Basic Implementation](#basic-implementation)
- [Optimization 1: Union by Rank](#optimization-1-union-by-rank)
- [Optimization 2: Path Compression](#optimization-2-path-compression)
- [Optimized Union-Find](#optimized-union-find)
- [Common Patterns](#common-patterns)
- [Interview Problems](#interview-problems)
- [Advanced Applications](#advanced-applications)
- [Complexity Analysis](#complexity-analysis)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Union-Find** (also called **Disjoint Set Union** or **DSU**) is a data structure that efficiently tracks and merges disjoint sets. It's particularly powerful for:

- Detecting cycles in undirected graphs
- Finding connected components
- Kruskal's Minimum Spanning Tree algorithm
- Dynamic connectivity problems

**Key characteristics:**
- Tracks partitioning of elements into disjoint sets
- Two main operations: **Union** (merge sets) and **Find** (find set representative)
- With optimizations: nearly O(1) amortized time per operation
- Simple to implement but very powerful

## ELI10 Explanation

Imagine you have a classroom with students, and you want to organize them into friend groups.

**Initial state:** Everyone is in their own group (alone).

**Union (make friends):**
When Alice and Bob become friends, we merge their groups. If Alice was already friends with Charlie, now Bob is also in the same group as Charlie!

**Find (which group?):**
To check if two students are in the same friend group, we find the "leader" of each group. If they have the same leader, they're friends!

**The smart part:**
Instead of remembering everyone in each group, we just remember:
1. Each person's "parent" (who they point to)
2. Eventually, everyone points to a "leader" who points to themselves

It's like a tree where the root is the group leader!

## Core Concepts

### When to Use Union-Find

Union-Find is perfect for:

1. **Connected components** in graphs
2. **Cycle detection** in undirected graphs
3. **Dynamic connectivity** - connections change over time
4. **Grouping/clustering** elements
5. **Minimum spanning tree** (Kruskal's algorithm)
6. **Grid connectivity** problems

**Keywords to look for:**
- "connected components"
- "group elements"
- "detect cycles"
- "dynamic connectivity"
- "friends of friends"

### Key Operations

```python
"""
1. FIND(x): Find the representative (root) of x's set
   - Returns the "leader" of the set containing x
   - Used to check if two elements are in same set

2. UNION(x, y): Merge the sets containing x and y
   - Combines two sets into one
   - One root becomes child of the other

3. CONNECTED(x, y): Check if x and y are in same set
   - Simply: find(x) == find(y)
"""
```

## Basic Implementation

### Naive Union-Find

```python
class UnionFind:
    """
    Basic Union-Find without optimizations.

    Time complexity:
    - find(): O(n) worst case (chain)
    - union(): O(n) worst case
    """

    def __init__(self, n: int):
        """Initialize n disjoint sets."""
        # parent[i] = parent of element i
        # Initially, each element is its own parent (root)
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """
        Find root of x's set.

        Follow parent pointers until reaching root (parent[x] == x).
        """
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Make root_y the parent of root_x
            self.parent[root_x] = root_y

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

# Example usage
uf = UnionFind(10)

# Union some elements
uf.union(1, 2)
uf.union(2, 3)
uf.union(4, 5)

print(uf.connected(1, 3))  # True (1-2-3 connected)
print(uf.connected(1, 4))  # False (different sets)
print(uf.connected(4, 5))  # True (4-5 connected)

uf.union(3, 5)
print(uf.connected(1, 4))  # True (now all connected!)
```

### Visualizing Operations

```python
def visualize_union_find():
    """Demonstrate Union-Find operations visually."""
    uf = UnionFind(6)

    print("Initial state: Each element is its own set")
    print(f"Parent array: {uf.parent}")
    print("Sets: {0} {1} {2} {3} {4} {5}\n")

    print("Union(0, 1):")
    uf.union(0, 1)
    print(f"Parent array: {uf.parent}")
    print("Sets: {0,1} {2} {3} {4} {5}\n")

    print("Union(2, 3):")
    uf.union(2, 3)
    print(f"Parent array: {uf.parent}")
    print("Sets: {0,1} {2,3} {4} {5}\n")

    print("Union(0, 2):")
    uf.union(0, 2)
    print(f"Parent array: {uf.parent}")
    print("Sets: {0,1,2,3} {4} {5}\n")

    print(f"Connected(1, 3)? {uf.connected(1, 3)}")  # True
    print(f"Connected(1, 5)? {uf.connected(1, 5)}")  # False

visualize_union_find()
```

## Optimization 1: Union by Rank

**Problem with naive union:** Can create long chains (O(n) find time).

**Solution:** Always attach smaller tree under larger tree.

```python
class UnionFindByRank:
    """
    Union-Find with union by rank optimization.

    Rank = approximate depth of tree.
    Attach smaller rank tree under larger rank tree.

    Time complexity:
    - find(): O(log n)
    - union(): O(log n)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n  # Rank of each tree

    def find(self, x: int) -> int:
        """Find root of x's set."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """
        Merge sets containing x and y.
        Returns True if sets were different (union happened).
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
            # Same rank: choose one as parent, increment rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

# Example
uf = UnionFindByRank(10)
uf.union(1, 2)
uf.union(3, 4)
uf.union(1, 3)  # Merges {1,2} with {3,4}
print(uf.connected(2, 4))  # True
```

## Optimization 2: Path Compression

**Further optimization:** During find(), make all nodes point directly to root.

```python
class UnionFindPathCompression:
    """
    Union-Find with path compression.

    During find(), flatten the tree by making all nodes
    point directly to the root.

    Time complexity:
    - find(): Amortized nearly O(1)
    - union(): Amortized nearly O(1)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """
        Find root with path compression.

        Recursively find root and compress path.
        """
        if self.parent[x] != x:
            # Path compression: point x directly to root
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def find_iterative(self, x: int) -> int:
        """Iterative version with two passes."""
        # First pass: find root
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Second pass: compress path
        while x != root:
            next_node = self.parent[x]
            self.parent[x] = root
            x = next_node

        return root

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        self.parent[root_x] = root_y
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
```

## Optimized Union-Find

**Best implementation:** Combine both optimizations!

```python
class UnionFind:
    """
    Optimized Union-Find with both:
    1. Union by rank (or size)
    2. Path compression

    Time complexity: O(α(n)) amortized per operation
    where α(n) is inverse Ackermann function (< 5 for practical n)

    This is the standard implementation you should use!
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of disjoint sets

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union by rank with path compression.
        Returns True if union happened, False if already connected.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already connected

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1  # Decreased number of sets
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in same set."""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """Return number of disjoint sets."""
        return self.count

# Example with detailed output
def demonstrate_optimized_uf():
    uf = UnionFind(8)

    print(f"Initial sets: {uf.get_count()}")

    operations = [
        (0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (2, 5)
    ]

    for x, y in operations:
        if uf.union(x, y):
            print(f"Union({x}, {y}) - Sets now: {uf.get_count()}")
        else:
            print(f"Union({x}, {y}) - Already connected!")

    print(f"\nFinal sets: {uf.get_count()}")
    print(f"Connected(0, 7)? {uf.connected(0, 7)}")  # True
    print(f"Connected(0, 3)? {uf.connected(0, 3)}")  # True

demonstrate_optimized_uf()
```

### Union by Size (Alternative)

```python
class UnionFindBySize:
    """
    Union by size instead of rank.
    Often simpler and equally effective.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n  # Size of each set
        self.count = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by size: attach smaller to larger
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]

        self.count -= 1
        return True

    def get_size(self, x: int) -> int:
        """Get size of set containing x."""
        return self.size[self.find(x)]
```

## Common Patterns

### Pattern 1: Count Connected Components

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    """
    LeetCode 323: Count connected components in undirected graph.

    Strategy: Use Union-Find to merge connected nodes.
    Remaining count = number of components.

    Time: O(E * α(n)), Space: O(n)
    """
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    return uf.get_count()

# Example
edges = [[0, 1], [1, 2], [3, 4]]
print(count_components(5, edges))  # 2 components: {0,1,2} and {3,4}
```

### Pattern 2: Detect Cycle in Undirected Graph

```python
def has_cycle(n: int, edges: list[list[int]]) -> bool:
    """
    Detect if undirected graph has a cycle.

    Strategy: If union(u, v) returns False, u and v
    were already connected → cycle exists!

    Time: O(E * α(n)), Space: O(n)
    """
    uf = UnionFind(n)

    for u, v in edges:
        if not uf.union(u, v):
            return True  # Already connected, cycle found!

    return False

# Example
edges1 = [[0, 1], [1, 2], [2, 0]]  # Triangle - has cycle
edges2 = [[0, 1], [1, 2]]           # Line - no cycle

print(has_cycle(3, edges1))  # True
print(has_cycle(3, edges2))  # False
```

### Pattern 3: Find Redundant Connection

```python
def find_redundant_connection(edges: list[list[int]]) -> list[int]:
    """
    LeetCode 684: Find edge that creates cycle in tree.

    Tree with n nodes has n-1 edges. One extra edge creates cycle.
    Return the last edge that creates the cycle.

    Time: O(n * α(n)), Space: O(n)
    """
    uf = UnionFind(len(edges) + 1)  # 1-indexed

    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates cycle

    return []

# Example
edges = [[1, 2], [1, 3], [2, 3]]
print(find_redundant_connection(edges))  # [2, 3]
```

### Pattern 4: Number of Provinces

```python
def find_circle_num(is_connected: list[list[int]]) -> int:
    """
    LeetCode 547: Find number of provinces (friend circles).

    is_connected[i][j] = 1 if person i and j are friends.

    Time: O(n² * α(n)), Space: O(n)
    """
    n = len(is_connected)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                uf.union(i, j)

    return uf.get_count()

# Example
is_connected = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
]
print(find_circle_num(is_connected))  # 2 provinces
```

## Interview Problems

### Problem 1: Accounts Merge

```python
def accounts_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    LeetCode 721: Merge accounts with common emails.

    Strategy:
    1. Map each email to an ID
    2. Union accounts with same emails
    3. Group emails by root ID

    Time: O(N * K * α(N)) where N = accounts, K = max emails
    Space: O(N * K)
    """
    email_to_id = {}
    email_to_name = {}
    uf = UnionFind(len(accounts))

    # Assign IDs and union accounts with same email
    for i, account in enumerate(accounts):
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name

            if email in email_to_id:
                uf.union(i, email_to_id[email])
            else:
                email_to_id[email] = i

    # Group emails by root account
    root_to_emails = {}
    for email, account_id in email_to_id.items():
        root = uf.find(account_id)
        if root not in root_to_emails:
            root_to_emails[root] = []
        root_to_emails[root].append(email)

    # Build result
    result = []
    for root, emails in root_to_emails.items():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))

    return result

# Example
accounts = [
    ["John", "john@mail.com", "john_newyork@mail.com"],
    ["John", "john00@mail.com"],
    ["Mary", "mary@mail.com"],
    ["John", "john_newyork@mail.com", "john00@mail.com"]
]
print(accounts_merge(accounts))
```

### Problem 2: Number of Islands II

```python
def num_islands2(m: int, n: int, positions: list[list[int]]) -> list[int]:
    """
    LeetCode 305: Number of islands after each addLand operation.

    Strategy: Use Union-Find with 2D coordinates.

    Time: O(k * α(mn)) where k = operations
    Space: O(mn)
    """
    def get_id(x: int, y: int) -> int:
        """Convert 2D coordinate to 1D ID."""
        return x * n + y

    uf = UnionFind(m * n)
    grid = [[0] * n for _ in range(m)]
    result = []
    count = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for x, y in positions:
        if grid[x][y] == 1:
            result.append(count)
            continue

        grid[x][y] = 1
        count += 1
        current_id = get_id(x, y)

        # Try to union with adjacent islands
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                neighbor_id = get_id(nx, ny)
                if uf.union(current_id, neighbor_id):
                    count -= 1  # Merged two islands

        result.append(count)

    return result

# Example
positions = [[0,0], [0,1], [1,2], [2,1]]
print(num_islands2(3, 3, positions))  # [1, 1, 2, 3]
```

### Problem 3: Smallest String With Swaps

```python
def smallest_string_with_swaps(s: str, pairs: list[list[int]]) -> str:
    """
    LeetCode 1202: Find lexicographically smallest string after swaps.

    Strategy:
    1. Union indices that can be swapped (transitively)
    2. For each component, sort characters
    3. Distribute sorted characters back

    Time: O(n log n + E * α(n)), Space: O(n)
    """
    n = len(s)
    uf = UnionFind(n)

    # Union all swappable pairs
    for i, j in pairs:
        uf.union(i, j)

    # Group indices by root
    root_to_indices = {}
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_indices:
            root_to_indices[root] = []
        root_to_indices[root].append(i)

    # Sort characters in each group
    result = list(s)
    for indices in root_to_indices.values():
        # Get characters at these indices
        chars = sorted([s[i] for i in indices])
        # Put sorted chars back
        for i, char in zip(sorted(indices), chars):
            result[i] = char

    return ''.join(result)

# Example
s = "dcab"
pairs = [[0,3], [1,2]]
print(smallest_string_with_swaps(s, pairs))  # "bacd"
# Can swap: 0↔3, 1↔2
# Groups: {0,3} and {1,2}
# Sort: "dc" → "cd" and "ab" → "ab"
# Result: "bacd"
```

### Problem 4: Satisfiability of Equality Equations

```python
def equations_possible(equations: list[str]) -> bool:
    """
    LeetCode 990: Check if all equality equations can be satisfied.

    equations[i] is "a==b" or "a!=b"

    Strategy:
    1. Process all "==" equations, union variables
    2. Check all "!=" equations, return False if same set

    Time: O(n * α(26)), Space: O(26) = O(1)
    """
    uf = UnionFind(26)  # 26 letters

    # Process equality equations
    for eq in equations:
        if eq[1] == '=':
            a = ord(eq[0]) - ord('a')
            b = ord(eq[3]) - ord('a')
            uf.union(a, b)

    # Check inequality equations
    for eq in equations:
        if eq[1] == '!':
            a = ord(eq[0]) - ord('a')
            b = ord(eq[3]) - ord('a')
            if uf.connected(a, b):
                return False  # Contradiction!

    return True

# Examples
print(equations_possible(["a==b", "b!=a"]))  # False (contradiction)
print(equations_possible(["a==b", "b==c", "a!=c"]))  # False
print(equations_possible(["a==b", "b!=c", "c==a"]))  # False
print(equations_possible(["a!=b", "b!=c", "c!=a"]))  # True
```

## Advanced Applications

### Kruskal's Minimum Spanning Tree

```python
def kruskal_mst(n: int, edges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """
    Kruskal's algorithm for Minimum Spanning Tree.

    edges: list of (u, v, weight)

    Strategy:
    1. Sort edges by weight
    2. Add edge if it doesn't create cycle (use Union-Find)
    3. Stop when n-1 edges added

    Time: O(E log E), Space: O(n)
    """
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])  # Sort by weight
    mst = []

    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            if len(mst) == n - 1:
                break  # MST complete

    return mst

# Example
edges = [
    (0, 1, 10),
    (0, 2, 6),
    (0, 3, 5),
    (1, 3, 15),
    (2, 3, 4)
]
mst = kruskal_mst(4, edges)
print(f"MST edges: {mst}")
print(f"Total weight: {sum(w for _, _, w in mst)}")
```

### Online Connectivity Queries

```python
class DynamicConnectivity:
    """
    Handle dynamic connectivity with online queries.

    Supports:
    - Add edge
    - Query if two nodes connected
    """

    def __init__(self, n: int):
        self.uf = UnionFind(n)

    def add_edge(self, u: int, v: int) -> None:
        """Add edge between u and v."""
        self.uf.union(u, v)

    def is_connected(self, u: int, v: int) -> bool:
        """Check if u and v are connected."""
        return self.uf.connected(u, v)

    def count_components(self) -> int:
        """Get current number of components."""
        return self.uf.get_count()

# Example
dc = DynamicConnectivity(6)
print(f"Components: {dc.count_components()}")  # 6

dc.add_edge(0, 1)
dc.add_edge(2, 3)
print(f"Components: {dc.count_components()}")  # 4

print(f"0-1 connected? {dc.is_connected(0, 1)}")  # True
print(f"0-2 connected? {dc.is_connected(0, 2)}")  # False

dc.add_edge(1, 2)
print(f"0-3 connected? {dc.is_connected(0, 3)}")  # True
```

## Complexity Analysis

### Time Complexity

| Operation | Naive | With Rank | With Path Compression | Both Optimizations |
|-----------|-------|-----------|----------------------|-------------------|
| Find | O(n) | O(log n) | Amortized O(log n) | Amortized O(α(n))* |
| Union | O(n) | O(log n) | Amortized O(log n) | Amortized O(α(n))* |
| Connected | O(n) | O(log n) | Amortized O(log n) | Amortized O(α(n))* |

*α(n) is the inverse Ackermann function, which grows extremely slowly:
- α(10^9) < 5
- For all practical purposes, α(n) ≤ 4
- Can be considered nearly constant!

### Space Complexity

- **O(n)** for parent array
- **O(n)** for rank/size array (if used)
- Total: **O(n)**

## When to Use

**Use Union-Find when:**

1. **Connected components** in undirected graph
   - "How many groups?"
   - "Are these elements connected?"

2. **Cycle detection** in undirected graph
   - Simpler than DFS for this specific case
   - Used in Kruskal's MST

3. **Dynamic connectivity**
   - Connections added over time
   - Need to query connectivity online

4. **Grouping/clustering** elements
   - Social networks (friends of friends)
   - Image segmentation
   - Equivalence classes

5. **Minimum Spanning Tree** (Kruskal's algorithm)

**Don't use when:**
- Need to track actual paths (use BFS/DFS)
- Directed graphs (Union-Find is for undirected)
- Need to delete edges efficiently (harder with Union-Find)
- Need to iterate through elements in a set (not supported)

## Common Pitfalls

### 1. Forgetting Path Compression

```python
# Inefficient:
def find(self, x: int) -> int:
    while self.parent[x] != x:
        x = self.parent[x]
    return x

# Optimized with path compression:
def find(self, x: int) -> int:
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]
```

### 2. Not Using Union by Rank/Size

```python
# Bad: Can create long chains
def union(self, x: int, y: int):
    root_x = self.find(x)
    root_y = self.find(y)
    self.parent[root_x] = root_y  # Arbitrary choice

# Good: Balance trees
def union(self, x: int, y: int):
    root_x = self.find(x)
    root_y = self.find(y)

    if self.rank[root_x] < self.rank[root_y]:
        self.parent[root_x] = root_y
    elif self.rank[root_x] > self.rank[root_y]:
        self.parent[root_y] = root_x
    else:
        self.parent[root_y] = root_x
        self.rank[root_x] += 1
```

### 3. Incorrect Initialization

```python
# Wrong: All elements point to 0
self.parent = [0] * n

# Correct: Each element is its own parent
self.parent = list(range(n))
```

### 4. Off-by-One with Indexing

```python
# If nodes are 1-indexed (common in graph problems)
uf = UnionFind(n + 1)  # Create n+1 elements

# If nodes are 0-indexed
uf = UnionFind(n)  # Create n elements
```

### 5. Not Checking if Union Succeeded

```python
# Missing information:
uf.union(u, v)

# Better: track if union happened
if uf.union(u, v):
    # Successfully merged two different sets
    count -= 1
else:
    # Already in same set (cycle detected!)
    return True
```

## Practice Problems

### Easy
1. **Find if Path Exists in Graph** (LeetCode 1971)
2. **The Earliest Moment When Everyone Become Friends** (LeetCode 1101)

### Medium
3. **Number of Connected Components** (LeetCode 323)
4. **Graph Valid Tree** (LeetCode 261)
5. **Number of Provinces** (LeetCode 547)
6. **Redundant Connection** (LeetCode 684)
7. **Accounts Merge** (LeetCode 721)
8. **Most Stones Removed** (LeetCode 947)
9. **Satisfiability of Equality Equations** (LeetCode 990)
10. **Smallest String With Swaps** (LeetCode 1202)
11. **Minimize Malware Spread** (LeetCode 924)
12. **Regions Cut By Slashes** (LeetCode 959)
13. **Sentence Similarity II** (LeetCode 737)

### Hard
14. **Number of Islands II** (LeetCode 305)
15. **Redundant Connection II** (LeetCode 685) - directed graph
16. **Minimize Malware Spread II** (LeetCode 928)
17. **Bricks Falling When Hit** (LeetCode 803)

### Classic Algorithms
18. **Kruskal's MST** - implement using Union-Find
19. **Image Segmentation** - connected components
20. **Social Network Analysis** - friend circles

## Additional Resources

### Visualizations
- **VisuAlgo**: Union-Find visualization
- **Algorithm Visualizer**: DSU animations

### Tutorials
- **Princeton Algorithms**: Union-Find lecture (Sedgewick)
- **CP-Algorithms**: Disjoint Set Union
- **GeeksforGeeks**: Union-Find detailed guide

### Papers
- Tarjan & van Leeuwen: "Worst-case Analysis of Set Union Algorithms"
- Inverse Ackermann function analysis

### Videos
- **MIT OCW**: Union-Find lecture
- **Princeton Coursera**: Algorithms Part I (Week 1)

### Practice Platforms
- LeetCode Tag: Union-Find (40+ problems)
- Codeforces: DSU problems
- AtCoder: Union-Find practice

---

**Key Takeaways:**
1. Union-Find tracks disjoint sets with near-constant time operations
2. Two key optimizations: path compression + union by rank
3. Perfect for connectivity, grouping, and cycle detection
4. Kruskal's MST uses Union-Find
5. Time complexity: O(α(n)) ≈ O(1) amortized with optimizations
6. Simple to implement but extremely powerful

Master Union-Find and you'll efficiently solve graph connectivity problems that would be much harder with other approaches!
