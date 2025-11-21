# Skip Lists

## Table of Contents
- [Overview](#overview)
- [The Problem It Solves](#the-problem-it-solves)
- [How It Works](#how-it-works)
- [Key Concepts](#key-concepts)
- [Structure and Properties](#structure-and-properties)
- [Core Operations](#core-operations)
  - [Search](#search-operation)
  - [Insert](#insert-operation)
  - [Delete](#delete-operation)
- [Time and Space Complexity](#time-and-space-complexity)
- [Implementation](#implementation)
  - [Python Implementation](#python-implementation)
  - [JavaScript Implementation](#javascript-implementation)
- [Variations](#variations)
- [Applications](#applications)
- [Comparison with Other Structures](#comparison-with-other-structures)
- [Interview Patterns](#interview-patterns)
- [Advanced Topics](#advanced-topics)

## Overview

A **Skip List** is a probabilistic data structure that allows O(log n) search, insertion, and deletion operations on an ordered sequence of elements. Invented by William Pugh in 1990, it provides an elegant alternative to balanced binary search trees using randomization instead of complex balancing rules.

### Key Characteristics

- **Probabilistic**: Uses random coin flips instead of deterministic balancing
- **Simple**: Much easier to implement than AVL or Red-Black trees
- **Efficient**: O(log n) expected time for search, insert, delete
- **Space efficient**: O(n) expected space (with small constants)
- **Concurrent-friendly**: Easier to make lock-free than balanced trees
- **Ordered**: Maintains sorted order of elements

### Why Skip Lists?

**Advantages over Balanced BSTs:**
- Simpler implementation (no rotations or color management)
- Easier to understand and reason about
- Better for concurrent access (lock-free implementations)
- Predictable performance in practice
- Natural support for range queries

**Real-world usage:**
- Redis Sorted Sets
- LevelDB (replaced by LSM trees but inspired design)
- Apache Lucene (MemoryIndex)
- Concurrent data structures

## The Problem It Solves

### The Search Problem in Sorted Lists

**Given:** A sorted linked list with n elements

**Need:** Fast search, insert, delete operations

**Approaches:**

| Structure | Search | Insert | Delete | Complexity |
|-----------|--------|--------|--------|------------|
| Sorted Array | O(log n) | O(n) | O(n) | Binary search but expensive updates |
| Linked List | O(n) | O(n) | O(n) | Sequential scan |
| BST (balanced) | O(log n) | O(log n) | O(log n) | Complex rotations |
| **Skip List** | **O(log n)** | **O(log n)** | **O(log n)** | **Simple + Fast** |

### The Brilliant Insight

**Question:** How can we speed up search in a sorted linked list?

**Answer:** Add "express lanes" at multiple levels!

```
Level 3: 1 -----------------> 25
Level 2: 1 --------> 13 ----> 25 --------> 45
Level 1: 1 -> 7 ---> 13 ----> 25 -> 33 --> 45 -> 56
Level 0: 1 -> 7 -> 9 -> 13 -> 17 -> 25 -> 33 -> 40 -> 45 -> 56
```

Search for 33:
- Start at top level (Level 3): 1 → 25 (too small) → next level
- Level 2: 25 → 45 (too big) → next level
- Level 1: 25 → 33 (found!)

Only 3 jumps instead of 7!

## How It Works

### Layered Linked Lists

A skip list consists of multiple levels of linked lists:
- **Level 0**: Contains all elements (base list)
- **Level 1+**: Progressively sparser "express lanes"
- **Higher levels**: Skip more elements for faster traversal

### Randomized Heights

**Key question:** Which elements should appear in higher levels?

**Answer:** Use a coin flip!

When inserting element x:
1. Insert at level 0 (always)
2. Flip coin: if heads, insert at level 1
3. Flip coin: if heads, insert at level 2
4. Continue until tails (or max level reached)

**Probability**: Element appears at level k with probability 1/2^k

```
Level 3: ~12.5% of elements
Level 2: ~25% of elements
Level 1: ~50% of elements
Level 0: 100% of elements
```

### Search Path

To search for value x:
1. Start at top-left (highest level, head node)
2. Move right while next.value < x
3. Move down one level
4. Repeat until found or reach bottom

```
Search for 40:

Level 2: HEAD -> 13 -> 25 -> 45 (25 < 40 < 45, go down)
                      ↓
Level 1:             25 -> 33 -> 45 (33 < 40 < 45, go down)
                           ↓
Level 0:                  33 -> 40 (found!)
```

## Key Concepts

### 1. Node Structure

Each node contains:
- **Key/value**: The data stored
- **Forward pointers**: Array of next pointers at each level
- **Level**: Height of the node (random)

```
Node at level 3:
┌─────┐
│  25 │
├─────┤
│ [0] │ → next at level 0
│ [1] │ → next at level 1
│ [2] │ → next at level 2
│ [3] │ → next at level 3
└─────┘
```

### 2. Level Distribution

With probability p = 0.5:
- 50% of nodes have height 1 (level 0 only)
- 25% of nodes have height 2 (levels 0-1)
- 12.5% of nodes have height 3 (levels 0-2)
- ...

**Expected max height**: O(log n)

### 3. Express Lane Analogy

Think of skip list as highway system:
- Level 0: Local roads (all intersections)
- Level 1: Highway (half the exits)
- Level 2: Interstate (quarter of the exits)
- Level 3: Major interstate (few exits)

To reach destination:
1. Start on highest highway available
2. Take highway until past destination
3. Drop to lower level
4. Repeat

### 4. Probabilistic Balancing

Unlike AVL/Red-Black trees:
- **No rotations needed**
- **No deterministic balance** constraints
- **Balance via randomization**

**With high probability** (1 - 1/n^c):
- Height is O(log n)
- Operations are O(log n)

### 5. Invariants

1. **Sorted order**: All levels maintain sorted order
2. **Subset property**: Level k+1 ⊆ Level k
3. **Reachability**: All nodes reachable from head

## Structure and Properties

### Mathematical Properties

**For skip list with n elements and probability p = 0.5:**

1. **Expected number of levels**:
   ```
   L = log₂(n)
   ```

2. **Expected number of pointers**:
   ```
   Total pointers = n * (1 + p + p² + ...) = n * 1/(1-p) = 2n
   ```

3. **Expected search time**:
   ```
   T(n) = O(log n)
   ```

4. **Space**:
   ```
   S(n) = O(n)  (expected)
   ```

### Probability Analysis

**Theorem**: With probability ≥ 1 - 1/n, search time is O(log n)

**Proof sketch:**
- Probability a node reaches level k: p^k = (1/2)^k
- Expected number of nodes at level k: n/2^k
- Height exceeds c·log n with probability ≤ 1/n^c
- Search path length bounded by height

### Space Overhead

**Expected space per node**:
```
E[pointers per node] = 1 + p + p² + p³ + ...
                     = 1/(1-p)
                     = 2  (when p = 0.5)
```

So on average, each node has 2 forward pointers.

**Total space**: O(n) expected, O(n log n) with high probability

## Core Operations

### Search Operation

**Goal**: Find if element x exists in skip list

**Algorithm:**
```
1. current = head
2. For level from max_level down to 0:
     While current.forward[level].key < x:
         current = current.forward[level]
3. current = current.forward[0]
4. Return current.key == x
```

**Visualization:**
```
Search for 25:

Level 2: H ---> 13 ---> 25*
              ↓
Level 1: H -> 7 -> 13 -> 25*
                        ↓
Level 0: H -> 7 -> 9 -> 13 -> 17 -> 25* (found!)

Path: H₂ → 13₂ → 25₂ → 25₁ → 25₀
```

**Time Complexity**: O(log n) expected
- Each level: O(1/p) expected steps right
- Number of levels: O(log n) expected
- Total: O(log n)

### Insert Operation

**Goal**: Insert new element x

**Algorithm:**
```
1. Find insert position (like search)
   - Keep track of update[] array (last node before insert at each level)
2. Generate random level for new node
3. Create new node with random level
4. Update forward pointers at each level
```

**Detailed Steps:**
```python
def insert(x):
    # Step 1: Find insert position
    update = [None] * MAX_LEVEL
    current = head

    for i in range(level, -1, -1):
        while current.forward[i] and current.forward[i].key < x:
            current = current.forward[i]
        update[i] = current  # Last node before insert position

    # Step 2: Generate random level
    new_level = random_level()

    # Step 3: Create new node
    new_node = Node(x, new_level)

    # Step 4: Update pointers
    for i in range(new_level):
        new_node.forward[i] = update[i].forward[i]
        update[i].forward[i] = new_node
```

**Visualization:**
```
Insert 23 (randomly assigned level 2):

Before:
Level 2: H ---> 13 ---> 25
Level 1: H -> 7 -> 13 -> 25
Level 0: H -> 7 -> 9 -> 13 -> 17 -> 25

After:
Level 2: H ---> 13 ---> 23 ---> 25
Level 1: H -> 7 -> 13 -> 23 -> 25
Level 0: H -> 7 -> 9 -> 13 -> 17 -> 23 -> 25

Update array points to last nodes before 23:
update[2] = 13
update[1] = 13
update[0] = 17
```

**Time Complexity**: O(log n) expected

### Delete Operation

**Goal**: Remove element x from skip list

**Algorithm:**
```
1. Find node to delete (like search)
   - Keep track of update[] array
2. If node exists:
     For each level:
         Update forward pointers to bypass deleted node
     Free node memory
```

**Detailed Steps:**
```python
def delete(x):
    # Step 1: Find node
    update = [None] * MAX_LEVEL
    current = head

    for i in range(level, -1, -1):
        while current.forward[i] and current.forward[i].key < x:
            current = current.forward[i]
        update[i] = current

    current = current.forward[0]

    # Step 2: Delete if found
    if current and current.key == x:
        for i in range(level + 1):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]

        # Decrease level if top levels empty
        while level > 0 and head.forward[level] is None:
            level -= 1
```

**Visualization:**
```
Delete 13:

Before:
Level 2: H ---> 13 ---> 25
Level 1: H -> 7 -> 13 -> 25
Level 0: H -> 7 -> 9 -> 13 -> 17 -> 25

After:
Level 2: H ---------------> 25
Level 1: H -> 7 ----------> 25
Level 0: H -> 7 -> 9 -----> 17 -> 25
```

**Time Complexity**: O(log n) expected

### Random Level Generation

**Core of skip list's probabilistic nature:**

```python
def random_level(p=0.5, max_level=16):
    """
    Generate random level for new node.
    With probability p, increase level by 1.
    """
    level = 0
    while random.random() < p and level < max_level - 1:
        level += 1
    return level
```

**Distribution:**
- Level 0: 50% probability
- Level 1: 25% probability
- Level 2: 12.5% probability
- Level k: (1/2)^(k+1) probability

## Time and Space Complexity

### Time Complexity

| Operation | Average | Worst Case | Notes |
|-----------|---------|------------|-------|
| Search | O(log n) | O(n) | Worst case if very unlucky with random levels |
| Insert | O(log n) | O(n) | Includes search time |
| Delete | O(log n) | O(n) | Includes search time |
| Range query | O(log n + k) | O(n) | k = number of elements in range |

**Probabilistic guarantee**: With probability ≥ 1 - 1/n:
- All operations are O(log n)
- Height is O(log n)

### Space Complexity

| Metric | Expected | Worst Case |
|--------|----------|------------|
| Total space | O(n) | O(n log n) |
| Pointers per node | 2 (p=0.5) | O(log n) |
| Maximum level | O(log n) | O(log n) |

**Practical**: With p = 0.5, space overhead is ~2x compared to simple linked list

## Implementation

### Python Implementation

**Complete Skip List Implementation:**

```python
import random

class Node:
    """Node in skip list with multiple forward pointers"""

    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)  # Forward pointers at each level


class SkipList:
    """
    Skip List implementation with O(log n) expected time operations.

    Features:
    - Search: O(log n) expected
    - Insert: O(log n) expected
    - Delete: O(log n) expected
    - Ordered traversal: O(n)
    """

    def __init__(self, max_level=16, p=0.5):
        """
        Initialize skip list.

        Args:
            max_level (int): Maximum number of levels
            p (float): Probability for level increase (typically 0.5 or 0.25)
        """
        self.max_level = max_level
        self.p = p
        self.level = 0  # Current maximum level in use
        self.header = Node(None, None, max_level)  # Sentinel head node

    def random_level(self):
        """
        Generate random level for new node.

        Returns:
            int: Random level between 0 and max_level-1

        Time: O(log n) expected
        """
        level = 0
        while random.random() < self.p and level < self.max_level - 1:
            level += 1
        return level

    def search(self, key):
        """
        Search for key in skip list.

        Args:
            key: Key to search for

        Returns:
            Value if found, None otherwise

        Time: O(log n) expected
        """
        current = self.header

        # Start from highest level and move down
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        # Move to level 0
        current = current.forward[0]

        # Check if we found the key
        if current and current.key == key:
            return current.value
        return None

    def insert(self, key, value):
        """
        Insert key-value pair into skip list.

        Args:
            key: Key to insert
            value: Value to insert

        Time: O(log n) expected
        """
        # Array to store rightmost node at each level before insert position
        update = [None] * self.max_level
        current = self.header

        # Find insert position
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        # Move to level 0
        current = current.forward[0]

        # If key exists, update value
        if current and current.key == key:
            current.value = value
            return

        # Generate random level for new node
        new_level = self.random_level()

        # If new level is higher than current level, update header pointers
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # Create new node
        new_node = Node(key, value, new_level)

        # Update forward pointers
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def delete(self, key):
        """
        Delete key from skip list.

        Args:
            key: Key to delete

        Returns:
            bool: True if deleted, False if not found

        Time: O(log n) expected
        """
        update = [None] * self.max_level
        current = self.header

        # Find node to delete
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # If found, delete it
        if current and current.key == key:
            # Update forward pointers
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            # Decrease level if top levels are empty
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            return True

        return False

    def display(self):
        """Display skip list structure (for debugging)"""
        print("\nSkip List Structure:")
        for i in range(self.level, -1, -1):
            print(f"Level {i}: ", end="")
            node = self.header.forward[i]
            while node:
                print(f"{node.key} ", end="")
                node = node.forward[i]
            print()

    def range_query(self, start_key, end_key):
        """
        Get all key-value pairs in range [start_key, end_key].

        Args:
            start_key: Start of range (inclusive)
            end_key: End of range (inclusive)

        Returns:
            list: List of (key, value) tuples in range

        Time: O(log n + k) where k is number of results
        """
        result = []
        current = self.header

        # Find starting position
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < start_key:
                current = current.forward[i]

        # Move to level 0 and collect results
        current = current.forward[0]
        while current and current.key <= end_key:
            result.append((current.key, current.value))
            current = current.forward[0]

        return result

    def __iter__(self):
        """Iterator for in-order traversal"""
        current = self.header.forward[0]
        while current:
            yield (current.key, current.value)
            current = current.forward[0]

    def __len__(self):
        """Count number of elements. Time: O(n)"""
        count = 0
        current = self.header.forward[0]
        while current:
            count += 1
            current = current.forward[0]
        return count


# Example usage
if __name__ == "__main__":
    sl = SkipList()

    # Insert elements
    elements = [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]
    for elem in elements:
        sl.insert(elem, f"value_{elem}")

    print("After insertions:")
    sl.display()

    # Search
    print(f"\nSearch for 19: {sl.search(19)}")
    print(f"Search for 20: {sl.search(20)}")

    # Range query
    print(f"\nRange [9, 21]: {sl.range_query(9, 21)}")

    # Delete
    print("\nDeleting 12...")
    sl.delete(12)
    sl.display()

    # Iteration
    print("\nAll elements:")
    for key, value in sl:
        print(f"{key}: {value}")
```

**Simplified Version:**

```python
class SimpleSkipList:
    """Minimal skip list implementation"""

    class Node:
        def __init__(self, key, level):
            self.key = key
            self.forward = [None] * (level + 1)

    def __init__(self):
        self.max_level = 16
        self.p = 0.5
        self.level = 0
        self.head = self.Node(None, self.max_level)

    def random_level(self):
        level = 0
        while random.random() < self.p and level < self.max_level - 1:
            level += 1
        return level

    def search(self, key):
        curr = self.head
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]
        curr = curr.forward[0]
        return curr and curr.key == key

    def insert(self, key):
        update = [self.head] * self.max_level
        curr = self.head

        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]
            update[i] = curr

        level = self.random_level()
        if level > self.level:
            self.level = level

        node = self.Node(key, level)
        for i in range(level + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
```

### JavaScript Implementation

```javascript
class Node {
    constructor(key, value, level) {
        this.key = key;
        this.value = value;
        this.forward = new Array(level + 1).fill(null);
    }
}

class SkipList {
    constructor(maxLevel = 16, p = 0.5) {
        this.maxLevel = maxLevel;
        this.p = p;
        this.level = 0;
        this.header = new Node(null, null, maxLevel);
    }

    randomLevel() {
        let level = 0;
        while (Math.random() < this.p && level < this.maxLevel - 1) {
            level++;
        }
        return level;
    }

    search(key) {
        let current = this.header;

        for (let i = this.level; i >= 0; i--) {
            while (current.forward[i] && current.forward[i].key < key) {
                current = current.forward[i];
            }
        }

        current = current.forward[0];
        return current && current.key === key ? current.value : null;
    }

    insert(key, value) {
        const update = new Array(this.maxLevel);
        let current = this.header;

        for (let i = this.level; i >= 0; i--) {
            while (current.forward[i] && current.forward[i].key < key) {
                current = current.forward[i];
            }
            update[i] = current;
        }

        current = current.forward[0];

        if (current && current.key === key) {
            current.value = value;
            return;
        }

        const newLevel = this.randomLevel();

        if (newLevel > this.level) {
            for (let i = this.level + 1; i <= newLevel; i++) {
                update[i] = this.header;
            }
            this.level = newLevel;
        }

        const newNode = new Node(key, value, newLevel);

        for (let i = 0; i <= newLevel; i++) {
            newNode.forward[i] = update[i].forward[i];
            update[i].forward[i] = newNode;
        }
    }

    delete(key) {
        const update = new Array(this.maxLevel);
        let current = this.header;

        for (let i = this.level; i >= 0; i--) {
            while (current.forward[i] && current.forward[i].key < key) {
                current = current.forward[i];
            }
            update[i] = current;
        }

        current = current.forward[0];

        if (current && current.key === key) {
            for (let i = 0; i <= this.level; i++) {
                if (update[i].forward[i] !== current) break;
                update[i].forward[i] = current.forward[i];
            }

            while (this.level > 0 && this.header.forward[this.level] === null) {
                this.level--;
            }

            return true;
        }

        return false;
    }

    *[Symbol.iterator]() {
        let current = this.header.forward[0];
        while (current) {
            yield [current.key, current.value];
            current = current.forward[0];
        }
    }
}

// Example usage
const sl = new SkipList();
sl.insert(3, 'three');
sl.insert(6, 'six');
sl.insert(9, 'nine');

console.log(sl.search(6));  // 'six'
console.log(sl.search(5));  // null

for (const [key, value] of sl) {
    console.log(`${key}: ${value}`);
}
```

## Variations

### 1. Deterministic Skip List

Instead of random levels, use deterministic pattern (e.g., 1-2-4 pattern).

**Advantage**: Guaranteed O(log n) worst case
**Disadvantage**: More complex, loses simplicity benefit

### 2. Indexable Skip List

Add size information at each node to support kth element queries.

```python
class IndexableNode:
    def __init__(self, key, level):
        self.key = key
        self.forward = [None] * (level + 1)
        self.width = [1] * (level + 1)  # Distance to next node

    def find_kth(self, k):
        """Find kth element in O(log n)"""
        current = self.head
        pos = 0

        for i in range(self.level, -1, -1):
            while current.forward[i] and pos + current.width[i] <= k:
                pos += current.width[i]
                current = current.forward[i]

        return current.key
```

### 3. Concurrent Skip List

Lock-free skip list for concurrent access.

**Key idea**: Use atomic compare-and-swap (CAS) operations

**Advantage**: High concurrency, no global locks

**Used in**: Java's `ConcurrentSkipListMap`, `ConcurrentSkipListSet`

### 4. Skip List with Finger Search

Maintain "fingers" pointing to recently accessed nodes.

**Advantage**: O(log k) search when searching near previous position
**Use case**: Temporal locality in access patterns

## Applications

### 1. Redis Sorted Sets

Redis uses skip lists for sorted sets (ZADD, ZRANGE commands).

**Why skip lists over balanced trees:**
- Simpler implementation
- Better for range queries (just walk forward pointers)
- Memory efficient
- Good cache locality

```python
# Redis-style sorted set operations
class RedisSortedSet:
    def __init__(self):
        self.skip_list = SkipList()
        self.dict = {}  # member -> score mapping

    def zadd(self, member, score):
        """Add member with score"""
        if member in self.dict:
            self.skip_list.delete((self.dict[member], member))
        self.skip_list.insert((score, member), member)
        self.dict[member] = score

    def zrange(self, start, stop):
        """Get members in score range"""
        result = []
        for (score, member), _ in self.skip_list:
            if score >= start and score <= stop:
                result.append(member)
        return result
```

### 2. In-Memory Databases

LevelDB's MemTable uses skip lists.

**Advantages:**
- Fast ordered insertion
- Efficient range scans
- Good for LSM-tree first level

### 3. Priority Queues

Skip lists can implement priority queues with O(log n) operations.

```python
class SkipListPriorityQueue:
    def __init__(self):
        self.sl = SkipList()

    def insert(self, priority, item):
        self.sl.insert(priority, item)

    def extract_min(self):
        """Get and remove minimum element"""
        min_node = self.sl.header.forward[0]
        if min_node:
            self.sl.delete(min_node.key)
            return min_node.value
        return None

    def peek_min(self):
        """Get minimum without removing"""
        min_node = self.sl.header.forward[0]
        return min_node.value if min_node else None
```

### 4. Range Query Systems

Efficiently answer range queries on ordered data.

```python
def range_aggregation(skip_list, start, end, aggregate_fn):
    """
    Aggregate values in range using custom function.
    Examples: sum, max, count, etc.
    """
    result = aggregate_fn.identity()
    current = skip_list.header

    # Find start
    for i in range(skip_list.level, -1, -1):
        while current.forward[i] and current.forward[i].key < start:
            current = current.forward[i]

    # Aggregate range
    current = current.forward[0]
    while current and current.key <= end:
        result = aggregate_fn.combine(result, current.value)
        current = current.forward[0]

    return result
```

### 5. Computational Geometry

Skip lists for interval trees and segment storage.

**Use case**: Store intervals and query overlaps

## Comparison with Other Structures

### Skip List vs Balanced BST

| Feature | Skip List | Balanced BST (AVL/RB) |
|---------|-----------|----------------------|
| **Implementation** | Simple | Complex (rotations) |
| **Search** | O(log n) expected | O(log n) worst case |
| **Insert** | O(log n) expected | O(log n) worst case |
| **Delete** | O(log n) expected | O(log n) worst case |
| **Space** | O(n) expected | O(n) worst case |
| **Balancing** | Probabilistic | Deterministic |
| **Code lines** | ~100-150 | ~300-500 |
| **Concurrency** | Easier (lock-free possible) | Harder |
| **Range queries** | Natural | Need in-order traversal |
| **Cache locality** | Better (sequential) | Worse (pointer chasing) |

**When to use Skip List:**
- Want simpler implementation
- Need concurrent access
- Frequent range queries
- Probabilistic guarantees acceptable

**When to use BST:**
- Need guaranteed O(log n) worst case
- Memory is extremely tight
- Want deterministic behavior

### Skip List vs Hash Table

| Feature | Skip List | Hash Table |
|---------|-----------|------------|
| **Ordered** | Yes | No |
| **Search** | O(log n) | O(1) average |
| **Range query** | O(log n + k) | O(n) |
| **Space** | O(n) | O(n) |
| **Dynamic resizing** | Not needed | Needed (expensive) |

**Use Skip List when:** Need ordering, range queries, no resizing
**Use Hash Table when:** Only need point queries, no ordering required

### Skip List vs B-Tree

| Feature | Skip List | B-Tree |
|---------|-----------|--------|
| **Disk-friendly** | No | Yes |
| **Node size** | Small | Large (cache line) |
| **Fanout** | 2 (binary) | High (configurable) |
| **Use case** | In-memory | Disk-based databases |

## Interview Patterns

### Pattern 1: Basic Operations

```python
def solve_with_skip_list(operations):
    """
    Implement data structure supporting:
    - insert(x)
    - search(x)
    - delete(x)
    All in O(log n)
    """
    sl = SkipList()

    for op, value in operations:
        if op == 'insert':
            sl.insert(value, value)
        elif op == 'search':
            print(sl.search(value))
        else:  # delete
            sl.delete(value)
```

### Pattern 2: Range Queries

```python
def range_sum_query(nums, queries):
    """
    Process range sum queries efficiently.
    Build skip list, use for range aggregation.
    """
    sl = SkipList()

    # Build skip list with prefix sums
    prefix = 0
    for i, num in enumerate(nums):
        prefix += num
        sl.insert(i, prefix)

    results = []
    for l, r in queries:
        right_sum = sl.search(r)
        left_sum = sl.search(l - 1) if l > 0 else 0
        results.append(right_sum - left_sum)

    return results
```

### Pattern 3: Design Problems

**LeetCode 1396: Design Underground System**

Skip lists can efficiently handle timestamp-based queries.

### Pattern 4: Kth Element

With indexable skip list:

```python
def find_kth_smallest(stream, k):
    """
    Maintain kth smallest in stream.
    Use indexable skip list for O(log n) kth queries.
    """
    sl = IndexableSkipList()

    for value in stream:
        sl.insert(value, value)
        if len(sl) >= k:
            print(f"Kth smallest: {sl.kth_element(k)}")
```

## Advanced Topics

### Probabilistic Analysis

**Theorem (Pugh 1990)**: In a skip list with n elements and p = 1/2:

1. **Expected number of levels**:
   ```
   E[L(n)] = log₂(n) + O(1)
   ```

2. **Search path length**:
   ```
   E[C(n)] = 2 log₂(n)  (expected comparisons)
   ```

3. **Space**:
   ```
   E[S(n)] = 2n  (expected number of pointers)
   ```

**Proof sketch** uses geometric distribution and linearity of expectation.

### Optimal p Value

**Question**: What's the best probability p?

**Answer**: Depends on priorities:
- **p = 1/2**: Good balance (most common choice)
- **p = 1/4**: Less space (1.33n pointers), slower search
- **p = 1/e**: Optimal for minimizing search time (theoretical)

**Trade-off**:
```
Lower p → Less space, more levels, slower search
Higher p → More space, fewer levels, faster search
```

### Lock-Free Implementation

Key challenges for concurrent skip list:
1. **Insertion**: Mark forward pointers atomically
2. **Deletion**: Logical delete then physical removal
3. **Memory reclamation**: Use hazard pointers or epoch-based

**Pseudo-code pattern:**
```
insert_concurrent(key):
    while True:
        find insert position
        attempt CAS to link new node
        if CAS succeeds:
            break
        else:
            retry with updated position
```

### Finger Search Optimization

**Idea**: Start search from recently accessed position instead of head.

**Benefit**: O(log k) time when target is distance k from finger

**Implementation**: Maintain finger pointer, update on each access

## Key Takeaways

1. **Skip lists = Probabilistic balanced trees** using randomization
2. **Simple implementation** compared to AVL/Red-Black trees
3. **O(log n) expected time** for search/insert/delete
4. **O(n) expected space** with low constants
5. **Excellent for concurrent** access (lock-free possible)
6. **Natural range query** support via forward pointers
7. **Used in production** (Redis, LevelDB)
8. **Random level = coin flips** until tails

## When to Use Skip Lists

✅ **Use when:**
- Want simple balanced structure
- Need ordered data with O(log n) operations
- Require range queries
- Building concurrent data structure
- Prefer probabilistic over deterministic

❌ **Don't use when:**
- Need guaranteed worst-case O(log n)
- Memory is extremely constrained
- Only need point queries (use hash table)
- Implementing for disk-based system (use B-tree)

---

**Time to Implement**: 30-40 minutes (basic), 60+ minutes (with all features)

**Space Complexity**: O(n) expected

**Most Common Interview Uses**:
- Design ordered data structure
- Range query problems
- Alternative to BST in system design

**Pro Tip**: Skip lists shine in their simplicity. When asked to implement a balanced tree, consider suggesting a skip list as an alternative - it's much easier to code correctly in an interview setting!
