# Persistent Data Structures

## Table of Contents
- [Overview](#overview)
- [Fundamental Concepts](#fundamental-concepts)
- [Types of Persistence](#types-of-persistence)
- [Implementation Techniques](#implementation-techniques)
- [Persistent Data Structures](#persistent-data-structures)
  - [Persistent Array](#persistent-array)
  - [Persistent Linked List](#persistent-linked-list)
  - [Persistent Stack](#persistent-stack)
  - [Persistent Tree](#persistent-tree)
  - [Persistent Segment Tree](#persistent-segment-tree)
- [Implementation](#implementation)
- [Applications](#applications)
- [Comparison](#comparison)
- [Common Problems](#common-problems)
- [Advanced Topics](#advanced-topics)

## Overview

A **persistent data structure** preserves all previous versions of itself when modified. Unlike ephemeral (regular) data structures that destroy old versions on updates, persistent structures allow access to any historical version.

### Key Characteristics

- **Immutability**: Original versions remain unchanged
- **Version history**: All past states accessible
- **Structural sharing**: Reuses unchanged parts between versions
- **Space efficient**: O(1) or O(log n) space per update (not O(n))
- **Time efficient**: Usually small overhead over ephemeral structures

### Why Persistent Data Structures?

**Benefits:**
- Naturally thread-safe (immutable)
- Easy undo/redo functionality
- Time-travel debugging
- Functional programming paradigm
- Concurrent access without locks
- Version control systems

**Trade-offs:**
- Slightly more space than ephemeral
- Small constant factor overhead
- More complex implementation

**Real-world usage:**
- Git version control (tree structures)
- Clojure/Haskell standard libraries
- React state management (immutable updates)
- Database snapshots
- Blockchain (immutable ledger)

## Fundamental Concepts

### Ephemeral vs Persistent

**Ephemeral (Regular) Data Structure:**
```
Version 0: [1, 2, 3]
Update: set(1, 5)
Version 1: [1, 5, 3]  ← Version 0 destroyed

Access version 0? IMPOSSIBLE
```

**Persistent Data Structure:**
```
Version 0: [1, 2, 3]
Update: set(1, 5)
Version 1: [1, 5, 3]  ← Version 0 still exists

Access version 0? ✓ Returns [1, 2, 3]
Access version 1? ✓ Returns [1, 5, 3]
```

### Structural Sharing

**Key idea**: Share unchanged parts between versions instead of copying everything.

```
Version 0:  A → B → C → D
               ↙
Version 1:  A → E → F → D
            ↑   ↑   ↑   ↑
         shared new new shared

Only E and F are new nodes. A and D are shared!
Space: O(changes), not O(n)
```

### Copy-on-Write Semantics

**Principle**: When modifying, create new nodes only for changed parts.

```python
# Ephemeral update
def update_ephemeral(arr, index, value):
    arr[index] = value  # Mutates in-place
    return arr

# Persistent update
def update_persistent(arr, index, value):
    new_arr = arr.copy()  # Copy everything
    new_arr[index] = value
    return new_arr  # Return new version

# Efficient persistent update
def update_persistent_efficient(node, index, value):
    if index == 0:
        return Node(value, node.next)  # New node, share rest
    return Node(node.value,
                update_persistent_efficient(node.next, index-1, value))
    # Only copy path to changed node!
```

## Types of Persistence

### 1. Partial Persistence

**Definition**: Can access all versions but only modify the newest version.

```
Version 0 ────→ Version 1 ────→ Version 2
   ↓               ↓               ↓
 read            read          read/write

Can only update version 2 (latest)
Can read any version
```

**Use cases:**
- Version history
- Undo functionality
- Debugging (time-travel)

**Example**: Most functional data structures

### 2. Full Persistence

**Definition**: Can access and modify any version.

```
Version 0 ────→ Version 1 ────→ Version 2
   ↓      ↘        ↓               ↓
 r/w      ↘      r/w             r/w
           ↘
          Version 1.1

Branching history! Any version can be modified.
```

**Use cases:**
- Git branching
- Parallel timelines
- Concurrent updates

**More complex to implement** but more powerful.

### 3. Confluent Persistence

**Definition**: Can merge versions (beyond branching).

```
Version 0 ────→ Version 1
   ↓               ↓
                Version 2
                   ↓
            Version 3 (merge 1 and 2)
```

**Rarely implemented** due to complexity.

### 4. Functional Persistence

**Definition**: Pure functional style - no mutation ever.

- Every operation returns new version
- Original completely unchanged
- Most common in functional programming

## Implementation Techniques

### Technique 1: Path Copying

**Idea**: Copy entire path from root to modified node.

```
Original tree:
        A
       / \
      B   C
     / \
    D   E

Update D to D':
        A'          (new root)
       / \
      B'  C         (new B, shared C)
     / \
    D'  E           (new D', shared E)

Only nodes on path copied: A, B, D
Nodes C and E are shared
```

**Complexity:**
- Time: O(log n) for trees
- Space: O(log n) per update

**Implementation:**
```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def update(node, path, new_value):
    """
    Update node at path (list of L/R directions).
    Returns new root with update applied.
    """
    if not path:
        return TreeNode(new_value, node.left, node.right)

    if path[0] == 'L':
        new_left = update(node.left, path[1:], new_value)
        return TreeNode(node.value, new_left, node.right)
    else:
        new_right = update(node.right, path[1:], new_value)
        return TreeNode(node.value, node.left, new_right)
```

### Technique 2: Fat Nodes

**Idea**: Store all versions of a field in each node.

```
Node with history:
┌─────────────────────┐
│ Value versions:     │
│   v0: 5             │
│   v3: 7             │
│   v7: 9             │
│ Left child:         │
│   v0-v2: NodeB      │
│   v3+:   NodeC      │
└─────────────────────┘

Each field tracks which versions it applies to
```

**Complexity:**
- Time: O(log m) where m = number of versions
- Space: O(1) amortized per update

**Advantage**: Efficient for many versions

**Disadvantage**: More complex bookkeeping

### Technique 3: Node Copying

**Idea**: Copy modified nodes entirely.

```python
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

def update_at(node, index, new_value):
    """Create new list with update at index"""
    if index == 0:
        return Node(new_value, node.next)  # Share tail

    return Node(node.value, update_at(node.next, index - 1, new_value))
```

**Simple but effective** for linked structures.

### Technique 4: Modification Boxes

**Idea**: Store modifications separately, apply lazily.

```
Version 0: Original array
Mods for v1: {index: 3, value: 7}
Mods for v2: {index: 5, value: 9}

Query version 2 at index 3:
  Check v2 mods: not there
  Check v1 mods: found! return 7
```

**Useful for persistent arrays** with few modifications.

## Persistent Data Structures

### Persistent Array

**Challenge**: Arrays require contiguous memory - hard to share structure.

**Solution 1: Path Copying with Tree Representation**

Represent array as balanced tree with O(log n) depth:

```
Array: [a, b, c, d, e, f, g, h]

Binary tree (breadth-first):
            root
         /        \
    [a,b,c,d]    [e,f,g,h]
     /    \        /    \
  [a,b] [c,d]  [e,f] [g,h]
  /  \   / \    / \    / \
 a   b  c  d   e  f   g  h
```

Update creates O(log n) new nodes.

**Implementation:**
```python
class PersistentArray:
    """
    Persistent array using binary tree representation.

    Operations:
    - get(i): O(log n)
    - set(i, val): O(log n), returns new version
    - Space: O(log n) per update
    """

    class Node:
        def __init__(self, value=None, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, arr):
        """Build from array. Time: O(n)"""
        self.size = len(arr)
        self.root = self._build(arr, 0, len(arr) - 1)

    def _build(self, arr, left, right):
        """Build tree from array range"""
        if left == right:
            return self.Node(value=arr[left])

        mid = (left + right) // 2
        return self.Node(
            left=self._build(arr, left, mid),
            right=self._build(arr, mid + 1, right)
        )

    def get(self, index):
        """Get value at index. Time: O(log n)"""
        return self._get(self.root, 0, self.size - 1, index)

    def _get(self, node, left, right, index):
        if left == right:
            return node.value

        mid = (left + right) // 2
        if index <= mid:
            return self._get(node.left, left, mid, index)
        else:
            return self._get(node.right, mid + 1, right, index)

    def set(self, index, value):
        """
        Create new version with update at index.
        Returns new PersistentArray.
        Time: O(log n)
        """
        new_pa = PersistentArray([])
        new_pa.size = self.size
        new_pa.root = self._set(self.root, 0, self.size - 1, index, value)
        return new_pa

    def _set(self, node, left, right, index, value):
        """Create new node path with update"""
        if left == right:
            return self.Node(value=value)

        mid = (left + right) // 2
        if index <= mid:
            new_left = self._set(node.left, left, mid, index, value)
            return self.Node(left=new_left, right=node.right)
        else:
            new_right = self._set(node.right, mid + 1, right, index, value)
            return self.Node(left=node.left, right=new_right)


# Example usage
arr = PersistentArray([1, 2, 3, 4, 5])
print(arr.get(2))  # 3

arr2 = arr.set(2, 10)  # Create new version
print(arr.get(2))   # 3 (original unchanged)
print(arr2.get(2))  # 10 (new version)
```

**Solution 2: Modification Log**

For sparse updates:
```python
class PersistentArrayLog:
    """Persistent array using modification log"""

    def __init__(self, base_array):
        self.base = base_array
        self.mods = {}  # {index: value}
        self.parent = None

    def get(self, index):
        """O(number of versions in chain)"""
        if index in self.mods:
            return self.mods[index]
        if self.parent:
            return self.parent.get(index)
        return self.base[index]

    def set(self, index, value):
        """O(1) - creates new version"""
        new_version = PersistentArrayLog(self.base)
        new_version.parent = self
        new_version.mods = {index: value}
        return new_version
```

### Persistent Linked List

**Easy to implement** due to natural sharing:

```python
class PersistentList:
    """
    Persistent singly linked list.

    Operations:
    - cons(x): O(1) - add to front
    - head(): O(1) - get first element
    - tail(): O(1) - get rest of list
    - get(i): O(i) - get ith element
    """

    class Node:
        def __init__(self, value, next=None):
            self.value = value
            self.next = next

    def __init__(self, node=None):
        self.head_node = node

    def cons(self, value):
        """
        Add element to front (create new version).
        Time: O(1)
        """
        new_node = self.Node(value, self.head_node)
        return PersistentList(new_node)

    def head(self):
        """Get first element. Time: O(1)"""
        if not self.head_node:
            raise IndexError("Empty list")
        return self.head_node.value

    def tail(self):
        """Get rest of list. Time: O(1)"""
        if not self.head_node:
            raise IndexError("Empty list")
        return PersistentList(self.head_node.next)

    def get(self, index):
        """Get element at index. Time: O(i)"""
        current = self.head_node
        for _ in range(index):
            if not current:
                raise IndexError("Index out of bounds")
            current = current.next

        if not current:
            raise IndexError("Index out of bounds")
        return current.value

    def to_list(self):
        """Convert to Python list for display"""
        result = []
        current = self.head_node
        while current:
            result.append(current.value)
            current = current.next
        return result


# Example usage
lst1 = PersistentList()
lst2 = lst1.cons(3).cons(2).cons(1)  # [1, 2, 3]
lst3 = lst2.cons(0)  # [0, 1, 2, 3]

print(lst2.to_list())  # [1, 2, 3] - unchanged
print(lst3.to_list())  # [0, 1, 2, 3]

# Structural sharing:
lst3.head_node.next is lst2.head_node  # True!
```

### Persistent Stack

**Trivial with persistent list:**

```python
class PersistentStack:
    """Stack using persistent list"""

    def __init__(self):
        self.list = PersistentList()

    def push(self, value):
        """O(1)"""
        new_stack = PersistentStack()
        new_stack.list = self.list.cons(value)
        return new_stack

    def pop(self):
        """O(1) - returns (value, new_stack)"""
        value = self.list.head()
        new_stack = PersistentStack()
        new_stack.list = self.list.tail()
        return value, new_stack

    def peek(self):
        """O(1)"""
        return self.list.head()


# Example
s1 = PersistentStack()
s2 = s1.push(1).push(2).push(3)

val, s3 = s2.pop()  # val=3, s3=[1,2]
print(val)  # 3
print(s2.peek())  # 3 (s2 unchanged)
print(s3.peek())  # 2
```

### Persistent Tree

**Binary Search Tree with path copying:**

```python
class PersistentBST:
    """
    Persistent Binary Search Tree.

    Operations:
    - insert(x): O(log n) - returns new tree
    - search(x): O(log n)
    - delete(x): O(log n) - returns new tree
    """

    class Node:
        def __init__(self, key, left=None, right=None):
            self.key = key
            self.left = left
            self.right = right

    def __init__(self, root=None):
        self.root = root

    def insert(self, key):
        """
        Insert key and return new tree version.
        Time: O(log n)
        Space: O(log n) - nodes on path
        """
        new_bst = PersistentBST()
        new_bst.root = self._insert(self.root, key)
        return new_bst

    def _insert(self, node, key):
        """Insert and return new root"""
        if node is None:
            return self.Node(key)

        if key < node.key:
            new_left = self._insert(node.left, key)
            return self.Node(node.key, new_left, node.right)
        elif key > node.key:
            new_right = self._insert(node.right, key)
            return self.Node(node.key, node.left, new_right)
        else:
            return node  # Duplicate, no change

    def search(self, key):
        """Search for key. Time: O(log n)"""
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return False

        if key == node.key:
            return True
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def inorder(self):
        """Get sorted list of keys"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)


# Example usage
bst1 = PersistentBST()
bst2 = bst1.insert(5).insert(3).insert(7)
bst3 = bst2.insert(4).insert(6)

print(bst2.inorder())  # [3, 5, 7]
print(bst3.inorder())  # [3, 4, 5, 6, 7]
print(bst2.search(4))  # False (bst2 unchanged)
print(bst3.search(4))  # True
```

### Persistent Segment Tree

**Powerful for competitive programming:**

```python
class PersistentSegmentTree:
    """
    Persistent segment tree for range queries.

    Common in competitive programming for:
    - Kth smallest in range across versions
    - Range queries on different time points
    """

    class Node:
        def __init__(self, sum_val=0, left=None, right=None):
            self.sum = sum_val
            self.left = left
            self.right = right

    def __init__(self, arr):
        self.n = len(arr)
        self.roots = []  # Store root for each version
        self.roots.append(self._build(arr, 0, self.n - 1))

    def _build(self, arr, l, r):
        """Build initial tree"""
        if l == r:
            return self.Node(sum_val=arr[l])

        mid = (l + r) // 2
        left_child = self._build(arr, l, mid)
        right_child = self._build(arr, mid + 1, r)

        return self.Node(
            sum_val=left_child.sum + right_child.sum,
            left=left_child,
            right=right_child
        )

    def update(self, version, pos, value):
        """
        Update position in given version.
        Creates new version.
        Returns new version number.

        Time: O(log n)
        Space: O(log n)
        """
        old_root = self.roots[version]
        new_root = self._update(old_root, 0, self.n - 1, pos, value)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node, l, r, pos, value):
        """Create new path with update"""
        if l == r:
            return self.Node(sum_val=value)

        mid = (l + r) // 2

        if pos <= mid:
            new_left = self._update(node.left, l, mid, pos, value)
            return self.Node(
                sum_val=new_left.sum + node.right.sum,
                left=new_left,
                right=node.right
            )
        else:
            new_right = self._update(node.right, mid + 1, r, pos, value)
            return self.Node(
                sum_val=node.left.sum + new_right.sum,
                left=node.left,
                right=new_right
            )

    def query(self, version, ql, qr):
        """
        Query sum in range [ql, qr] for given version.
        Time: O(log n)
        """
        root = self.roots[version]
        return self._query(root, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if ql > r or qr < l:
            return 0

        if ql <= l and r <= qr:
            return node.sum

        mid = (l + r) // 2
        left_sum = self._query(node.left, l, mid, ql, qr)
        right_sum = self._query(node.right, mid + 1, r, ql, qr)

        return left_sum + right_sum


# Example usage
arr = [1, 2, 3, 4, 5]
pst = PersistentSegmentTree(arr)

print(pst.query(0, 0, 4))  # Version 0: sum([1,2,3,4,5]) = 15

v1 = pst.update(0, 2, 10)  # Version 1: arr[2] = 10
print(pst.query(0, 0, 4))  # Version 0: still 15
print(pst.query(v1, 0, 4))  # Version 1: 1+2+10+4+5 = 22

v2 = pst.update(0, 0, 5)  # Branch from version 0
print(pst.query(v2, 0, 4))  # 5+2+3+4+5 = 19
```

## Applications

### 1. Undo/Redo Functionality

**Text editor with unlimited undo:**

```python
class TextEditor:
    """Text editor with undo/redo using persistent list"""

    def __init__(self):
        self.versions = [PersistentList()]  # Version history
        self.current = 0  # Current version index

    def insert(self, char):
        """Insert character at front (simplified)"""
        new_text = self.versions[self.current].cons(char)

        # Discard future versions (like real editors)
        self.versions = self.versions[:self.current + 1]
        self.versions.append(new_text)
        self.current += 1

    def undo(self):
        """Go to previous version"""
        if self.current > 0:
            self.current -= 1

    def redo(self):
        """Go to next version"""
        if self.current < len(self.versions) - 1:
            self.current += 1

    def get_text(self):
        """Get current text"""
        return self.versions[self.current].to_list()


# Example
editor = TextEditor()
editor.insert('o')
editor.insert('l')
editor.insert('l')
editor.insert('e')
editor.insert('h')
print(''.join(reversed(editor.get_text())))  # "hello"

editor.undo()
editor.undo()
print(''.join(reversed(editor.get_text())))  # "hel"

editor.redo()
print(''.join(reversed(editor.get_text())))  # "hell"
```

### 2. Version Control (Git-like)

**Simplified version control:**

```python
class VersionControl:
    """Simple VCS using persistent trees"""

    def __init__(self):
        self.commits = {}  # commit_id -> tree root
        self.branches = {'main': 0}  # branch -> commit_id
        self.next_id = 0

        # Initial commit (empty tree)
        self.commits[0] = None
        self.next_id = 1

    def commit(self, branch, tree_root):
        """Create new commit on branch"""
        commit_id = self.next_id
        self.next_id += 1

        self.commits[commit_id] = tree_root
        self.branches[branch] = commit_id

        return commit_id

    def branch(self, new_branch, from_branch):
        """Create new branch from existing"""
        commit_id = self.branches[from_branch]
        self.branches[new_branch] = commit_id

    def checkout(self, branch):
        """Get tree at branch head"""
        commit_id = self.branches[branch]
        return self.commits[commit_id]
```

### 3. Concurrent Data Structures

**Lock-free concurrent updates:**

```python
import threading

class ConcurrentPersistentList:
    """
    Thread-safe concurrent list using persistence.
    No locks needed - each thread works with its version.
    """

    def __init__(self):
        self.versions = {0: PersistentList()}
        self.next_version = 1
        self.version_lock = threading.Lock()

    def add(self, base_version, value):
        """
        Add value based on base_version.
        Returns new version number.
        """
        base_list = self.versions[base_version]
        new_list = base_list.cons(value)

        with self.version_lock:
            version = self.next_version
            self.next_version += 1
            self.versions[version] = new_list

        return version

    def get(self, version):
        """Get list at version (thread-safe read)"""
        return self.versions[version]
```

### 4. Functional Programming

**Immutable data structures in functional style:**

```python
def functional_map(f, lst):
    """
    Map function over persistent list.
    Returns new list, original unchanged.
    """
    if lst.head_node is None:
        return lst

    return functional_map(f, lst.tail()).cons(f(lst.head()))


def functional_filter(pred, lst):
    """Filter persistent list"""
    if lst.head_node is None:
        return lst

    rest = functional_filter(pred, lst.tail())

    if pred(lst.head()):
        return rest.cons(lst.head())
    return rest


# Example
lst = PersistentList().cons(3).cons(2).cons(1)  # [1,2,3]
doubled = functional_map(lambda x: x * 2, lst)  # [2,4,6]
evens = functional_filter(lambda x: x % 2 == 0, lst)  # [2]

print(lst.to_list())      # [1, 2, 3] - unchanged
print(doubled.to_list())  # [2, 4, 6]
print(evens.to_list())    # [2]
```

### 5. Kth Smallest in Range Query

**Using persistent segment tree:**

```python
def kth_smallest_in_range(arr, queries):
    """
    Answer queries: kth smallest in range [l, r].

    Uses persistent segment tree with coordinate compression.
    Each version represents prefix of array.

    Time: O(n log n + q log² n)
    """
    # Build persistent segment tree for each prefix
    n = len(arr)

    # Coordinate compression
    sorted_vals = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_vals)}

    # Build PST for each prefix
    # pst[i] = frequency count for arr[0..i]
    # ... (implementation details)

    # For each query (l, r, k):
    #   Binary search on value
    #   Check count in range [l, r] using PST
    pass
```

### 6. Time-Travel Debugging

```python
class DebugState:
    """Capture program state at each step"""

    def __init__(self):
        self.states = []  # List of persistent structures
        self.current_step = 0

    def capture_state(self, variables):
        """Save current state"""
        # Deep copy using persistent structures
        state = {k: v for k, v in variables.items()}
        self.states.append(state)
        self.current_step = len(self.states) - 1

    def goto_step(self, step):
        """Time travel to specific step"""
        if 0 <= step < len(self.states):
            self.current_step = step
            return self.states[step]

    def get_current(self):
        """Get current state"""
        return self.states[self.current_step]
```

## Comparison

### Persistent vs Ephemeral Structures

| Feature | Ephemeral | Persistent |
|---------|-----------|------------|
| **Versions** | Current only | All versions |
| **Space** | O(n) | O(n + m log n)* |
| **Time overhead** | None | Small (log factor) |
| **Thread safety** | Needs locks | Naturally safe |
| **Undo/redo** | Complex | Trivial |
| **Implementation** | Simple | More complex |
| **Use when** | Single version | Multiple versions needed |

*m = number of updates

### Different Implementation Techniques

| Technique | Time | Space | Complexity |
|-----------|------|-------|------------|
| **Full copying** | O(n) | O(mn) | Simple |
| **Path copying** | O(log n) | O(m log n) | Moderate |
| **Fat nodes** | O(log m) | O(n + m) | Complex |
| **Modification log** | O(m) | O(m) | Simple |

### Persistent Structures Comparison

| Structure | Get | Update | Space/Update |
|-----------|-----|--------|--------------|
| **Array** | O(log n) | O(log n) | O(log n) |
| **List** | O(i) | O(i) | O(i) |
| **Stack/Queue** | O(1) | O(1) | O(1) |
| **BST** | O(log n) | O(log n) | O(log n) |
| **Segment Tree** | O(log n) | O(log n) | O(log n) |

## Common Problems

### LeetCode/Interview Problems

| Problem | Difficulty | Technique |
|---------|-----------|-----------|
| [1146. Snapshot Array](https://leetcode.com/problems/snapshot-array/) | Medium | Persistent array |
| [1483. Kth Ancestor of a Tree Node](https://leetcode.com/problems/kth-ancestor-of-a-tree-node/) | Hard | Binary lifting (persistent) |
| Version control system design | Hard | Persistent trees |
| Undo/redo implementation | Medium | Version history |

### Competitive Programming

**Classic problems:**
1. **Kth smallest in range across updates**
2. **Range queries at different time points**
3. **Tree path queries with modifications**
4. **Dynamic connectivity with rollback**

### Interview Patterns

**Pattern 1: Version History**
```python
class VersionedDataStructure:
    def __init__(self):
        self.versions = []
        self.current = None

    def update(self, data):
        # Create new version
        new_version = create_new(self.current, data)
        self.versions.append(new_version)
        self.current = new_version

    def get_version(self, version_id):
        return self.versions[version_id]
```

**Pattern 2: Branching**
```python
def branch_from_version(base_version, modification):
    # Create new branch from base
    new_version = copy_structure(base_version)
    apply_modification(new_version, modification)
    return new_version
```

## Advanced Topics

### Purely Functional Data Structures

**Okasaki's work** on functional data structures:

**Key insights:**
- Lazy evaluation enables amortized O(1) operations
- Memoization improves repeated access
- Scheduled computations for better worst-case

**Example: Banker's Queue**
```python
class BankersQueue:
    """
    Persistent queue with O(1) amortized operations.
    Uses lazy evaluation and invariant maintenance.
    """

    def __init__(self, front, rear):
        # Invariant: len(front) >= len(rear)
        self.front = front
        self.rear = rear
        self._maintain_invariant()

    def _maintain_invariant(self):
        if len(self.rear) > len(self.front):
            # Rotate: move rear to front (reversed)
            self.front = self.front + list(reversed(self.rear))
            self.rear = []

    def enqueue(self, x):
        """O(1) amortized"""
        return BankersQueue(self.front, self.rear + [x])

    def dequeue(self):
        """O(1) amortized"""
        if not self.front:
            raise IndexError("Empty queue")
        return self.front[0], BankersQueue(self.front[1:], self.rear)
```

### Confluent Persistence

**Merging versions:**

```python
class ConfluentPersistent:
    """
    Allows merging two versions.
    Much more complex than regular persistence.
    """

    def merge(self, version1, version2, merge_function):
        """
        Merge two versions using merge_function.
        Like git merge!
        """
        # Identify common ancestor
        ancestor = find_common_ancestor(version1, version2)

        # Three-way merge
        result = three_way_merge(
            ancestor,
            version1,
            version2,
            merge_function
        )

        return result
```

### Lock-Free Concurrent Algorithms

**Using Compare-and-Swap (CAS):**

```python
class LockFreePersistentStack:
    """
    Lock-free concurrent stack using persistent structures.
    Multiple threads can push/pop without locks.
    """

    def __init__(self):
        self.head = None

    def push(self, value):
        """Thread-safe push using CAS"""
        while True:
            old_head = self.head
            new_node = Node(value, old_head)

            # Atomic compare-and-swap
            if compare_and_swap(self.head, old_head, new_node):
                break
            # Retry if another thread modified head

    def pop(self):
        """Thread-safe pop using CAS"""
        while True:
            old_head = self.head
            if old_head is None:
                raise IndexError("Empty")

            new_head = old_head.next

            if compare_and_swap(self.head, old_head, new_head):
                return old_head.value
            # Retry if another thread modified head
```

### Space Optimization Techniques

**1. Garbage Collection of Old Versions:**
```python
def gc_old_versions(structure, keep_recent=10):
    """Keep only recent N versions"""
    if len(structure.versions) > keep_recent:
        # Remove old versions
        structure.versions = structure.versions[-keep_recent:]
```

**2. Compression:**
```python
def compress_version_chain(versions):
    """
    Compress long chains by periodic full snapshots.
    Trade-off: space vs access time.
    """
    SNAPSHOT_INTERVAL = 100

    compressed = []
    for i, version in enumerate(versions):
        if i % SNAPSHOT_INTERVAL == 0:
            # Full snapshot
            compressed.append(materialize(version))
        else:
            # Delta from previous
            compressed.append(delta(versions[i-1], version))

    return compressed
```

### Functional Reactive Programming (FRP)

**Persistent structures in reactive systems:**

```python
class Observable:
    """
    Reactive observable using persistent structures.
    Each update creates new version, notifies subscribers.
    """

    def __init__(self, initial_value):
        self.versions = [initial_value]
        self.subscribers = []

    def update(self, transform):
        """Update and notify"""
        new_value = transform(self.versions[-1])
        self.versions.append(new_value)

        for subscriber in self.subscribers:
            subscriber(new_value)

    def subscribe(self, callback):
        """Subscribe to updates"""
        self.subscribers.append(callback)
```

## Key Takeaways

1. **Persistent structures preserve history** - all versions accessible
2. **Structural sharing** - O(log n) space per update, not O(n)
3. **Path copying** - most common technique for trees
4. **Naturally thread-safe** - immutability eliminates race conditions
5. **Perfect for undo/redo** - version history built-in
6. **Functional programming** - foundation of purely functional languages
7. **Trade-off**: Small time/space overhead for immutability benefits
8. **Use in production**: Git, Clojure, React state management

## When to Use Persistent Data Structures

✅ **Use when:**
- Need version history or undo/redo
- Multiple threads accessing data (lock-free)
- Functional programming paradigm
- Snapshots required frequently
- Immutability is desired
- Debugging with time-travel

❌ **Don't use when:**
- Only need current version
- Memory is extremely constrained
- Performance critical (tight loops)
- Simple ephemeral structure sufficient
- Mutation is more natural for problem

---

**Time to Implement**:
- Simple (list/stack): 20-30 minutes
- Medium (BST): 40-60 minutes
- Complex (segment tree): 90+ minutes

**Space Complexity**: O(n + m log n) for m updates on size n

**Most Common Uses**:
- Undo/redo systems
- Version control
- Functional programming
- Concurrent systems
- Time-travel debugging

**Pro Tip**: Start with persistent linked lists to understand the concepts, then progress to trees. Remember: immutability + structural sharing = efficient persistence. The key insight is that you don't copy everything - just the path from root to modified node!
