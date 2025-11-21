# Advanced Tree Structures

## Overview

This guide covers advanced tree variants that extend beyond basic binary search trees. These structures provide self-balancing properties, optimized access patterns, or specialized use cases like database indexing.

## Table of Contents

1. [Splay Trees](#splay-trees)
2. [Treaps](#treaps)
3. [AVL vs Red-Black Trees](#avl-vs-red-black-trees)
4. [B+ Trees](#b-trees)
5. [Comparison Summary](#comparison-summary)

## Splay Trees

### Concept

Splay trees are self-adjusting binary search trees where recently accessed elements are moved to the root through **splaying** operations. This provides excellent amortized performance for sequences with temporal locality.

**Key idea**: Frequently accessed nodes stay near the root, making subsequent accesses fast.

### Properties

1. **No balance information stored** - simpler than AVL/Red-Black
2. **Self-optimizing** - adapts to access patterns
3. **Amortized O(log n)** for all operations
4. **Worst case O(n)** for single operation (rare)
5. **Excellent cache performance** - recently used items are near root

### Splaying Operations

Splaying moves a node to the root using three types of rotations:

#### 1. Zig (single rotation)

When node is child of root.

```
     Root                x
      /        =>         \
     x                    Root
```

#### 2. Zig-Zig (same-side double rotation)

When node and parent are both left or both right children.

```
       z                     x
      /                       \
     y         =>              y
    /                           \
   x                             z
```

#### 3. Zig-Zag (opposite-side double rotation)

When node is left-right or right-left.

```
     z                   x
    /                   / \
   y         =>        y   z
    \
     x
```

### Implementation

```python
class SplayNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    """
    Self-adjusting binary search tree
    Amortized O(log n) for all operations
    """
    def __init__(self):
        self.root = None

    def _rotate_right(self, node):
        """
        Right rotation around node
            node              left
            /  \              /  \
          left  C    =>      A   node
          / \                     / \
         A   B                   B   C
        """
        left = node.left
        node.left = left.right
        if left.right:
            left.right.parent = node

        left.parent = node.parent
        if not node.parent:
            self.root = left
        elif node == node.parent.left:
            node.parent.left = left
        else:
            node.parent.right = left

        left.right = node
        node.parent = left

    def _rotate_left(self, node):
        """
        Left rotation around node
          node                right
          /  \                /  \
         A  right    =>     node  C
             / \            / \
            B   C          A   B
        """
        right = node.right
        node.right = right.left
        if right.left:
            right.left.parent = node

        right.parent = node.parent
        if not node.parent:
            self.root = right
        elif node == node.parent.left:
            node.parent.left = right
        else:
            node.parent.right = right

        right.left = node
        node.parent = right

    def _splay(self, node):
        """
        Move node to root using rotations
        Three cases: zig, zig-zig, zig-zag
        """
        while node.parent:
            parent = node.parent
            grandparent = parent.parent

            if not grandparent:
                # Zig: node is child of root
                if node == parent.left:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)

            elif node == parent.left and parent == grandparent.left:
                # Zig-zig: both left children
                self._rotate_right(grandparent)
                self._rotate_right(parent)

            elif node == parent.right and parent == grandparent.right:
                # Zig-zig: both right children
                self._rotate_left(grandparent)
                self._rotate_left(parent)

            elif node == parent.right and parent == grandparent.left:
                # Zig-zag: left-right
                self._rotate_left(parent)
                self._rotate_right(grandparent)

            else:
                # Zig-zag: right-left
                self._rotate_right(parent)
                self._rotate_left(grandparent)

    def search(self, key):
        """
        Search and splay found node to root
        Time: O(log n) amortized
        """
        node = self._search_helper(self.root, key)
        if node:
            self._splay(node)
        return node.value if node else None

    def _search_helper(self, node, key):
        """Helper to find node without splaying"""
        if not node or node.key == key:
            return node

        if key < node.key:
            return self._search_helper(node.left, key)
        return self._search_helper(node.right, key)

    def insert(self, key, value=None):
        """
        Insert and splay new node to root
        Time: O(log n) amortized
        """
        # Standard BST insert
        if not self.root:
            self.root = SplayNode(key, value)
            return

        node = self.root
        while True:
            if key < node.key:
                if not node.left:
                    node.left = SplayNode(key, value)
                    node.left.parent = node
                    self._splay(node.left)
                    return
                node = node.left
            elif key > node.key:
                if not node.right:
                    node.right = SplayNode(key, value)
                    node.right.parent = node
                    self._splay(node.right)
                    return
                node = node.right
            else:
                # Key exists, update value and splay
                node.value = value
                self._splay(node)
                return

    def delete(self, key):
        """
        Delete key and splay parent to root
        Time: O(log n) amortized
        """
        node = self._search_helper(self.root, key)
        if not node:
            return False

        # Splay node to root
        self._splay(node)

        # Split into two subtrees
        if not node.left:
            self.root = node.right
            if self.root:
                self.root.parent = None
        else:
            left_subtree = node.left
            self.root = node.right
            if self.root:
                self.root.parent = None

            # Find max in left subtree and splay it
            max_node = left_subtree
            while max_node.right:
                max_node = max_node.right
            left_subtree.parent = None
            self.root = left_subtree
            self._splay(max_node)

            # Attach right subtree
            self.root.right = node.right
            if node.right:
                node.right.parent = self.root

        return True

    def find_min(self):
        """Find minimum and splay to root"""
        if not self.root:
            return None

        node = self.root
        while node.left:
            node = node.left

        self._splay(node)
        return node.key

    def find_max(self):
        """Find maximum and splay to root"""
        if not self.root:
            return None

        node = self.root
        while node.right:
            node = node.right

        self._splay(node)
        return node.key
```

### Complexity Analysis

| Operation | Worst Case | Amortized | Notes |
|-----------|-----------|-----------|-------|
| Search | O(n) | O(log n) | Splays accessed node to root |
| Insert | O(n) | O(log n) | New node becomes root |
| Delete | O(n) | O(log n) | Parent splayed to root |
| Min/Max | O(n) | O(log n) | Result splayed to root |

**Space**: O(n) - no balance information needed

### When to Use Splay Trees

**Advantages:**
- Excellent for temporal locality (accessing same items repeatedly)
- No balance information to store
- Simpler implementation than AVL/Red-Black
- Self-optimizing for access patterns
- Good cache performance

**Disadvantages:**
- Worst-case O(n) for single operation
- Not suitable for real-time systems (unpredictable)
- Poor for uniform random access
- Modifications during traversal (not thread-safe)

**Use cases:**
- Caching mechanisms (LRU-like behavior)
- Text editors (recent operations)
- Network routing tables
- Memory allocators

### Splay Tree Optimizations

```python
class OptimizedSplayTree(SplayTree):
    """Splay tree with additional optimizations"""

    def __init__(self):
        super().__init__()
        self.size = 0

    def split(self, key):
        """
        Split tree into two trees: keys < key and keys >= key
        Time: O(log n) amortized
        """
        # Search for key (splays it or closest to root)
        node = self._search_helper(self.root, key)
        if node:
            self._splay(node)

        if not self.root or self.root.key < key:
            # All keys < key
            return self, SplayTree()

        # Split at root
        left_tree = SplayTree()
        left_tree.root = self.root.left
        if left_tree.root:
            left_tree.root.parent = None

        self.root.left = None
        return left_tree, self

    def join(self, other):
        """
        Join two trees where all keys in self < all keys in other
        Time: O(log n) amortized
        """
        if not self.root:
            return other
        if not other.root:
            return self

        # Find max in self
        max_node = self.root
        while max_node.right:
            max_node = max_node.right
        self._splay(max_node)

        # Attach other as right subtree
        self.root.right = other.root
        other.root.parent = self.root
        return self

    def range_query(self, low, high):
        """
        Find all keys in range [low, high]
        Time: O(k + log n) where k is output size
        """
        result = []

        def inorder(node):
            if not node:
                return
            if low < node.key:
                inorder(node.left)
            if low <= node.key <= high:
                result.append((node.key, node.value))
            if node.key < high:
                inorder(node.right)

        inorder(self.root)
        return result
```

## Treaps

### Concept

**Treap** = **Tree** + **Heap**

A treap is a randomized binary search tree where each node has:
1. **Key** - maintains BST property
2. **Priority** - maintains heap property (randomly assigned)

**BST property**: Left subtree keys < node key < right subtree keys
**Heap property**: Node priority > children priorities (max-heap)

### Key Insight

Random priorities ensure expected O(log n) height without complex balancing.

### Structure

```python
import random

class TreapNode:
    def __init__(self, key, value=None, priority=None):
        self.key = key
        self.value = value
        self.priority = priority if priority else random.random()
        self.left = None
        self.right = None

class Treap:
    """
    Randomized binary search tree
    Expected O(log n) height with high probability
    """
    def __init__(self):
        self.root = None

    def _rotate_right(self, node):
        """
        Right rotation
            y              x
           / \            / \
          x   C    =>    A   y
         / \                / \
        A   B              B   C
        """
        left = node.left
        node.left = left.right
        left.right = node
        return left

    def _rotate_left(self, node):
        """
        Left rotation
          x                y
         / \              / \
        A   y      =>    x   C
           / \          / \
          B   C        A   B
        """
        right = node.right
        node.right = right.left
        right.left = node
        return right

    def insert(self, key, value=None, priority=None):
        """
        Insert with random priority
        Time: O(log n) expected
        """
        self.root = self._insert(self.root, key, value, priority)

    def _insert(self, node, key, value, priority):
        """Recursive insert maintaining BST and heap properties"""
        if not node:
            return TreapNode(key, value, priority)

        if key < node.key:
            node.left = self._insert(node.left, key, value, priority)
            # Restore heap property
            if node.left.priority > node.priority:
                node = self._rotate_right(node)

        elif key > node.key:
            node.right = self._insert(node.right, key, value, priority)
            # Restore heap property
            if node.right.priority > node.priority:
                node = self._rotate_left(node)

        else:
            # Key exists, update value
            node.value = value

        return node

    def delete(self, key):
        """
        Delete key by rotating it down to leaf
        Time: O(log n) expected
        """
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        """Recursive delete"""
        if not node:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Found node to delete
            if not node.left and not node.right:
                return None
            elif not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Both children exist - rotate node down
                if node.left.priority > node.right.priority:
                    node = self._rotate_right(node)
                    node.right = self._delete(node.right, key)
                else:
                    node = self._rotate_left(node)
                    node.left = self._delete(node.left, key)

        return node

    def search(self, key):
        """
        Standard BST search
        Time: O(log n) expected
        """
        node = self.root
        while node:
            if key == node.key:
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def split(self, key):
        """
        Split into two treaps: keys < key and keys >= key
        Time: O(log n) expected
        """
        # Insert dummy node with infinite priority
        dummy = TreapNode(key, None, float('inf'))
        self.root = self._insert_node(self.root, dummy)

        # Dummy is now root, split at children
        left_treap = Treap()
        right_treap = Treap()
        left_treap.root = dummy.left
        right_treap.root = dummy.right

        return left_treap, right_treap

    def _insert_node(self, node, new_node):
        """Insert existing node (for split operation)"""
        if not node:
            return new_node

        if new_node.key < node.key:
            node.left = self._insert_node(node.left, new_node)
            if node.left.priority > node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert_node(node.right, new_node)
            if node.right.priority > node.priority:
                node = self._rotate_left(node)

        return node

    def merge(self, other):
        """
        Merge two treaps
        Assumes all keys in self < all keys in other
        Time: O(log n) expected
        """
        self.root = self._merge(self.root, other.root)

    def _merge(self, left, right):
        """Recursively merge two treaps"""
        if not left:
            return right
        if not right:
            return left

        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def range_query(self, low, high):
        """
        Find all keys in range [low, high]
        Time: O(k + log n) where k is output size
        """
        result = []

        def inorder(node):
            if not node:
                return
            if low < node.key:
                inorder(node.left)
            if low <= node.key <= high:
                result.append((node.key, node.value))
            if node.key < high:
                inorder(node.right)

        inorder(self.root)
        return result
```

### Implicit Treap (for arrays)

Treaps can represent arrays with efficient operations:

```python
class ImplicitTreap:
    """
    Treap for array operations
    Supports insert, delete, reverse in O(log n)
    """
    def __init__(self):
        self.root = None

    class Node:
        def __init__(self, value):
            self.value = value
            self.priority = random.random()
            self.size = 1  # Subtree size
            self.left = None
            self.right = None

    def _get_size(self, node):
        """Get subtree size"""
        return node.size if node else 0

    def _update_size(self, node):
        """Update subtree size"""
        if node:
            node.size = 1 + self._get_size(node.left) + self._get_size(node.right)

    def _split(self, node, pos):
        """
        Split at position pos
        Returns (left, right) where left has pos elements
        """
        if not node:
            return None, None

        left_size = self._get_size(node.left)

        if pos <= left_size:
            left, node.left = self._split(node.left, pos)
            self._update_size(node)
            return left, node
        else:
            node.right, right = self._split(node.right, pos - left_size - 1)
            self._update_size(node)
            return node, right

    def _merge(self, left, right):
        """Merge two treaps"""
        if not left:
            return right
        if not right:
            return left

        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            self._update_size(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            self._update_size(right)
            return right

    def insert(self, pos, value):
        """Insert value at position pos"""
        left, right = self._split(self.root, pos)
        new_node = self.Node(value)
        self.root = self._merge(self._merge(left, new_node), right)

    def delete(self, pos):
        """Delete element at position pos"""
        left, right = self._split(self.root, pos)
        _, right = self._split(right, 1)
        self.root = self._merge(left, right)

    def get(self, pos):
        """Get element at position pos"""
        node = self.root
        while node:
            left_size = self._get_size(node.left)
            if pos == left_size:
                return node.value
            elif pos < left_size:
                node = node.left
            else:
                pos -= left_size + 1
                node = node.right
        return None
```

### Complexity Analysis

| Operation | Expected Time | Worst Case | Notes |
|-----------|--------------|------------|-------|
| Search | O(log n) | O(n) | Randomization makes O(n) unlikely |
| Insert | O(log n) | O(n) | Expected height is O(log n) |
| Delete | O(log n) | O(n) | Rotates node to leaf |
| Split | O(log n) | O(n) | Splits at key |
| Merge | O(log n) | O(n) | Assumes sorted order |

**Space**: O(n) - stores priority per node

### When to Use Treaps

**Advantages:**
- Simple implementation
- Expected O(log n) without complex balancing
- Efficient split/merge operations
- Good for randomized algorithms
- Deterministic behavior with fixed seeds

**Disadvantages:**
- Not guaranteed O(log n) (probabilistic)
- Extra space for priorities
- Slower constants than AVL/Red-Black
- Not cache-friendly (random priorities)

**Use cases:**
- When simple implementation is priority
- Persistent data structures (functional programming)
- Implicit sequences (array operations)
- Randomized algorithms

## AVL vs Red-Black Trees

Both are self-balancing BSTs with O(log n) operations, but with different tradeoffs.

### AVL Trees

**Balance condition**: For every node, heights of left and right subtrees differ by at most 1.

#### Structure

```python
class AVLNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # Height of subtree

class AVLTree:
    """
    Strictly balanced BST
    Height difference <= 1 at every node
    """
    def __init__(self):
        self.root = None

    def _height(self, node):
        """Get height of node"""
        return node.height if node else 0

    def _balance_factor(self, node):
        """Balance factor = left height - right height"""
        if not node:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node):
        """Update height based on children"""
        if node:
            node.height = 1 + max(self._height(node.left),
                                   self._height(node.right))

    def _rotate_right(self, y):
        """
        Right rotation
            y              x
           / \            / \
          x   C    =>    A   y
         / \                / \
        A   B              B   C
        """
        x = y.left
        B = x.right

        x.right = y
        y.left = B

        self._update_height(y)
        self._update_height(x)

        return x

    def _rotate_left(self, x):
        """Left rotation"""
        y = x.right
        B = y.left

        y.left = x
        x.right = B

        self._update_height(x)
        self._update_height(y)

        return y

    def _rebalance(self, node):
        """Rebalance node if needed"""
        self._update_height(node)
        balance = self._balance_factor(node)

        # Left heavy
        if balance > 1:
            if self._balance_factor(node.left) < 0:
                # Left-Right case
                node.left = self._rotate_left(node.left)
            # Left-Left case
            return self._rotate_right(node)

        # Right heavy
        if balance < -1:
            if self._balance_factor(node.right) > 0:
                # Right-Left case
                node.right = self._rotate_right(node.right)
            # Right-Right case
            return self._rotate_left(node)

        return node

    def insert(self, key, value=None):
        """Insert and rebalance - O(log n)"""
        self.root = self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        # Standard BST insert
        if not node:
            return AVLNode(key, value)

        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value
            return node

        # Rebalance
        return self._rebalance(node)

    def delete(self, key):
        """Delete and rebalance - O(log n)"""
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Found node to delete
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            # Two children: replace with inorder successor
            min_node = self._find_min(node.right)
            node.key = min_node.key
            node.value = min_node.value
            node.right = self._delete(node.right, min_node.key)

        # Rebalance
        return self._rebalance(node)

    def _find_min(self, node):
        """Find minimum in subtree"""
        while node.left:
            node = node.left
        return node

    def search(self, key):
        """Standard BST search - O(log n)"""
        node = self.root
        while node:
            if key == node.key:
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None
```

### Red-Black Trees

**Balance conditions**:
1. Every node is red or black
2. Root is black
3. All leaves (NIL) are black
4. Red nodes have black children (no two red nodes in a row)
5. All paths from node to leaves have same number of black nodes

#### Structure

```python
class RBNode:
    def __init__(self, key, value=None, color='RED'):
        self.key = key
        self.value = value
        self.color = color  # 'RED' or 'BLACK'
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    """
    Self-balancing BST with color properties
    Max height: 2 * log(n+1)
    """
    def __init__(self):
        self.NIL = RBNode(None, color='BLACK')  # Sentinel
        self.root = self.NIL

    def _rotate_left(self, x):
        """Left rotation"""
        y = x.right
        x.right = y.left

        if y.left != self.NIL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def _rotate_right(self, y):
        """Right rotation"""
        x = y.left
        y.left = x.right

        if x.right != self.NIL:
            x.right.parent = y

        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x

        x.right = y
        y.parent = x

    def insert(self, key, value=None):
        """Insert and fix violations - O(log n)"""
        # Standard BST insert
        node = RBNode(key, value, 'RED')
        node.left = self.NIL
        node.right = self.NIL

        parent = None
        current = self.root

        while current != self.NIL:
            parent = current
            if node.key < current.key:
                current = current.left
            elif node.key > current.key:
                current = current.right
            else:
                # Key exists, update
                current.value = value
                return

        node.parent = parent
        if parent is None:
            self.root = node
        elif node.key < parent.key:
            parent.left = node
        else:
            parent.right = node

        # Fix violations
        self._fix_insert(node)

    def _fix_insert(self, node):
        """Fix red-black properties after insert"""
        while node.parent and node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                if uncle.color == 'RED':
                    # Case 1: Uncle is red - recolor
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child - rotate left
                        node = node.parent
                        self._rotate_left(node)
                    # Case 3: Node is left child - rotate right
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._rotate_right(node.parent.parent)
            else:
                # Mirror cases
                uncle = node.parent.parent.left

                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._rotate_left(node.parent.parent)

        self.root.color = 'BLACK'

    def search(self, key):
        """Standard BST search - O(log n)"""
        node = self.root
        while node != self.NIL:
            if key == node.key:
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def delete(self, key):
        """Delete and fix violations - O(log n)"""
        node = self._search_node(key)
        if node == self.NIL:
            return False

        self._delete_node(node)
        return True

    def _search_node(self, key):
        """Find node with key"""
        node = self.root
        while node != self.NIL and node.key != key:
            if key < node.key:
                node = node.left
            else:
                node = node.right
        return node

    def _delete_node(self, node):
        """Delete node and fix violations"""
        original_color = node.color

        if node.left == self.NIL:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            x = node.left
            self._transplant(node, node.left)
        else:
            # Two children: replace with successor
            successor = self._find_min(node.right)
            original_color = successor.color
            x = successor.right

            if successor.parent == node:
                x.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor

            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            successor.color = node.color

        if original_color == 'BLACK':
            self._fix_delete(x)

    def _transplant(self, u, v):
        """Replace subtree u with subtree v"""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _fix_delete(self, node):
        """Fix red-black properties after delete"""
        while node != self.root and node.color == 'BLACK':
            if node == node.parent.left:
                sibling = node.parent.right

                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self._rotate_left(node.parent)
                    sibling = node.parent.right

                if sibling.left.color == 'BLACK' and sibling.right.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    if sibling.right.color == 'BLACK':
                        sibling.left.color = 'BLACK'
                        sibling.color = 'RED'
                        self._rotate_right(sibling)
                        sibling = node.parent.right

                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.right.color = 'BLACK'
                    self._rotate_left(node.parent)
                    node = self.root
            else:
                # Mirror cases
                sibling = node.parent.left

                if sibling.color == 'RED':
                    sibling.color = 'BLACK'
                    node.parent.color = 'RED'
                    self._rotate_right(node.parent)
                    sibling = node.parent.left

                if sibling.right.color == 'BLACK' and sibling.left.color == 'BLACK':
                    sibling.color = 'RED'
                    node = node.parent
                else:
                    if sibling.left.color == 'BLACK':
                        sibling.right.color = 'BLACK'
                        sibling.color = 'RED'
                        self._rotate_left(sibling)
                        sibling = node.parent.left

                    sibling.color = node.parent.color
                    node.parent.color = 'BLACK'
                    sibling.left.color = 'BLACK'
                    self._rotate_right(node.parent)
                    node = self.root

        node.color = 'BLACK'

    def _find_min(self, node):
        """Find minimum in subtree"""
        while node.left != self.NIL:
            node = node.left
        return node
```

### Detailed Comparison

#### 1. Balance Guarantees

| Property | AVL Tree | Red-Black Tree |
|----------|----------|----------------|
| Max height | 1.44 * log(n+2) | 2 * log(n+1) |
| Balance strictness | Height diff ≤ 1 | Relaxed (color rules) |
| Tree shape | More balanced | Less balanced |

```python
# AVL: Stricter balance
#        5           Height = 3
#       / \
#      3   7         All balanced
#     / \ / \
#    2  4 6  8

# Red-Black: Allows more imbalance
#        5 (B)       Height = 4 (allowed)
#       / \
#    3(R)  7(B)
#   /     / \
#  2(B)  6(R) 8(R)   More nodes on one side OK
```

#### 2. Operations Performance

| Operation | AVL Tree | Red-Black Tree | Winner |
|-----------|----------|----------------|--------|
| Search | **Faster** (more balanced) | Slightly slower | AVL |
| Insert | Slower (more rotations) | **Faster** (fewer rotations) | RB |
| Delete | Slower (more rotations) | **Faster** (fewer rotations) | RB |
| Lookup-heavy | **Better** | Good | AVL |
| Insert-heavy | Good | **Better** | RB |

```python
# Rotation counts per operation (average)

# AVL Tree:
# Insert: ~1.5 rotations (needs rebalancing)
# Delete: ~2 rotations (stricter balance)

# Red-Black Tree:
# Insert: ~0.5 rotations (recoloring often enough)
# Delete: ~0.5 rotations (relaxed balance)
```

#### 3. Memory Usage

```python
class AVLNode:
    # Memory per node:
    # - key: 8 bytes
    # - value: 8 bytes
    # - left: 8 bytes
    # - right: 8 bytes
    # - height: 4 bytes (int)
    # Total: ~36 bytes + object overhead
    pass

class RBNode:
    # Memory per node:
    # - key: 8 bytes
    # - value: 8 bytes
    # - left: 8 bytes
    # - right: 8 bytes
    # - parent: 8 bytes
    # - color: 1 byte (can use 1 bit)
    # Total: ~41 bytes + object overhead
    pass

# Winner: AVL (slightly less memory if no parent pointer)
# In practice: similar memory usage
```

#### 4. Implementation Complexity

| Aspect | AVL Tree | Red-Black Tree |
|--------|----------|----------------|
| Code complexity | Simpler | More complex |
| Insert cases | 4 cases | 3 cases + recoloring |
| Delete cases | 4 cases | 6 cases + recoloring |
| Debugging | Easier | Harder |

#### 5. Real-World Usage

**AVL Trees used in:**
- Databases with more reads than writes
- In-memory indices
- File systems (lookup tables)
- Window management in GUIs

**Red-Black Trees used in:**
- C++ STL map/set
- Java TreeMap/TreeSet
- Linux kernel (process scheduling)
- Databases with balanced read/write

```python
# Python doesn't have built-in AVL or RB trees
# But you can use sortedcontainers (uses B-trees)
from sortedcontainers import SortedDict

# Or implement your own based on needs:

def choose_tree(read_write_ratio):
    """
    Guide for choosing between AVL and Red-Black
    """
    if read_write_ratio > 2:
        return "AVL Tree - More lookups than modifications"
    elif read_write_ratio < 0.5:
        return "Red-Black Tree - More modifications than lookups"
    else:
        return "Red-Black Tree - Balanced read/write"

# Examples:
print(choose_tree(5))    # AVL Tree
print(choose_tree(0.3))  # Red-Black Tree
print(choose_tree(1))    # Red-Black Tree
```

#### 6. When to Choose Each

**Choose AVL when:**
- Lookup-intensive applications
- Simpler implementation needed
- Strictly balanced tree required
- Memory is not critical concern
- Database indices with mostly reads

**Choose Red-Black when:**
- Insert/delete-heavy workload
- Need consistent performance
- Used in standard libraries (compatibility)
- Slightly less memory per node
- General-purpose balanced tree

## B+ Trees

### Concept

B+ Trees are self-balancing tree structures optimized for systems that read/write large blocks of data, like databases and file systems.

**Key characteristics:**
1. **High fanout** - many children per node (100s)
2. **All data in leaves** - internal nodes only store keys
3. **Leaves linked** - sequential access is fast
4. **Disk-friendly** - minimizes disk I/O

### Structure

```
                    [50 | 100]                  <- Internal node (keys only)
                   /     |     \
                  /      |      \
           [20|35]   [60|75|90]  [110|125]     <- Internal nodes
           /  |  \      / | | \      /  |  \
          /   |   \    /  | |  \    /   |   \
     [1..19][20..34][35..49][50..59][60..74]  <- Leaf nodes (data)
         ↓      ↓       ↓       ↓       ↓
       next   next    next    next    next     <- Linked leaves
```

### Properties

1. **Order m**: Maximum m children per node
2. **Internal nodes**: Store m-1 keys, m children
3. **Leaf nodes**: Store records/pointers to records
4. **Balance**: All leaves at same level
5. **Occupancy**: Nodes at least ⌈m/2⌉ children (except root)
6. **Sequential access**: Leaves form linked list

### Implementation

```python
class BPlusTreeNode:
    """Node in B+ tree"""
    def __init__(self, is_leaf=False, order=4):
        self.keys = []          # Keys in node
        self.values = []        # Values (only in leaves)
        self.children = []      # Child pointers (only in internal)
        self.is_leaf = is_leaf
        self.next = None        # Next leaf (only in leaves)
        self.order = order

class BPlusTree:
    """
    B+ Tree optimized for disk access
    Order m means max m children per node
    """
    def __init__(self, order=4):
        """
        order: Maximum number of children per node
        Common values: 100-200 for disk-based systems
        """
        self.root = BPlusTreeNode(is_leaf=True, order=order)
        self.order = order

    def search(self, key):
        """
        Search for key
        Time: O(log_m n) where m is order
        Disk I/O: O(log_m n) = height of tree
        """
        node = self.root

        # Navigate to leaf
        while not node.is_leaf:
            # Binary search in keys (small array)
            i = self._find_key_position(node.keys, key)
            node = node.children[i]

        # Search in leaf
        i = self._find_key_position(node.keys, key)
        if i < len(node.keys) and node.keys[i] == key:
            return node.values[i]
        return None

    def _find_key_position(self, keys, key):
        """Binary search for position of key"""
        left, right = 0, len(keys)
        while left < right:
            mid = (left + right) // 2
            if keys[mid] < key:
                left = mid + 1
            else:
                right = mid
        return left

    def insert(self, key, value):
        """
        Insert key-value pair
        Time: O(log_m n)
        Disk I/O: O(log_m n) reads + O(log_m n) writes
        """
        # Find leaf to insert
        leaf = self._find_leaf(key)

        # Insert into leaf
        i = self._find_key_position(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            # Key exists, update
            leaf.values[i] = value
            return

        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)

        # Check if leaf needs splitting
        if len(leaf.keys) > self.order - 1:
            self._split_leaf(leaf)

    def _find_leaf(self, key):
        """Navigate to leaf that should contain key"""
        node = self.root

        while not node.is_leaf:
            i = self._find_key_position(node.keys, key)
            node = node.children[i]

        return node

    def _split_leaf(self, leaf):
        """
        Split leaf node when it overflows
        Creates new leaf and propagates split up
        """
        mid = len(leaf.keys) // 2

        # Create new leaf
        new_leaf = BPlusTreeNode(is_leaf=True, order=self.order)
        new_leaf.keys = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        new_leaf.next = leaf.next

        # Update original leaf
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.next = new_leaf

        # Propagate split up
        split_key = new_leaf.keys[0]

        if leaf == self.root:
            # Create new root
            new_root = BPlusTreeNode(is_leaf=False, order=self.order)
            new_root.keys = [split_key]
            new_root.children = [leaf, new_leaf]
            self.root = new_root
        else:
            self._insert_in_parent(leaf, split_key, new_leaf)

    def _insert_in_parent(self, left, key, right):
        """Insert key and right child into parent of left"""
        parent = self._find_parent(self.root, left)

        # Find position to insert
        i = self._find_key_position(parent.keys, key)
        parent.keys.insert(i, key)
        parent.children.insert(i + 1, right)

        # Check if parent needs splitting
        if len(parent.keys) > self.order - 1:
            self._split_internal(parent)

    def _split_internal(self, node):
        """Split internal node when it overflows"""
        mid = len(node.keys) // 2

        # Create new internal node
        new_node = BPlusTreeNode(is_leaf=False, order=self.order)
        split_key = node.keys[mid]
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]

        # Update original node
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        # Propagate split up
        if node == self.root:
            new_root = BPlusTreeNode(is_leaf=False, order=self.order)
            new_root.keys = [split_key]
            new_root.children = [node, new_node]
            self.root = new_root
        else:
            self._insert_in_parent(node, split_key, new_node)

    def _find_parent(self, current, child):
        """Find parent of child node (helper for split)"""
        if current.is_leaf or child in current.children:
            return current

        for c in current.children:
            if not c.is_leaf:
                result = self._find_parent(c, child)
                if result:
                    return result

        return None

    def range_query(self, low, high):
        """
        Find all keys in range [low, high]
        Time: O(log_m n + k) where k is output size
        Very efficient due to leaf linking!
        """
        result = []

        # Find starting leaf
        node = self._find_leaf(low)

        # Traverse leaves using next pointer
        while node:
            for i, key in enumerate(node.keys):
                if low <= key <= high:
                    result.append((key, node.values[i]))
                elif key > high:
                    return result
            node = node.next

        return result

    def delete(self, key):
        """
        Delete key
        Time: O(log_m n)
        May trigger merging/redistribution
        """
        leaf = self._find_leaf(key)

        # Find and remove key
        i = self._find_key_position(leaf.keys, key)
        if i >= len(leaf.keys) or leaf.keys[i] != key:
            return False  # Key not found

        leaf.keys.pop(i)
        leaf.values.pop(i)

        # Check if leaf needs merging
        min_keys = (self.order - 1) // 2
        if len(leaf.keys) < min_keys and leaf != self.root:
            self._handle_underflow(leaf)

        return True

    def _handle_underflow(self, node):
        """Handle underflow by borrowing or merging"""
        # Simplified version - full implementation is complex
        # Should check siblings and parent
        pass

    def print_tree(self, node=None, level=0):
        """Print tree structure"""
        if node is None:
            node = self.root

        print("  " * level + f"{'[Leaf]' if node.is_leaf else '[Internal]'} Keys: {node.keys}")

        if not node.is_leaf:
            for child in node.children:
                self.print_tree(child, level + 1)
```

### Database Optimizations

```python
class DatabaseBPlusTree(BPlusTree):
    """
    B+ Tree optimized for database use
    Includes caching and disk I/O tracking
    """
    def __init__(self, order=100, page_size=4096):
        """
        order: Typically 100-200 for databases
        page_size: Disk block size (4KB typical)
        """
        super().__init__(order)
        self.page_size = page_size
        self.cache = {}  # Simple LRU cache
        self.disk_reads = 0
        self.disk_writes = 0

    def _read_node(self, node_id):
        """Read node from disk (simulated)"""
        if node_id in self.cache:
            return self.cache[node_id]

        # Simulate disk read
        self.disk_reads += 1
        # In real system: read from disk
        node = self._load_from_disk(node_id)
        self.cache[node_id] = node
        return node

    def _write_node(self, node_id, node):
        """Write node to disk (simulated)"""
        self.disk_writes += 1
        self.cache[node_id] = node
        # In real system: write to disk

    def _load_from_disk(self, node_id):
        """Simulate loading node from disk"""
        # In real implementation:
        # 1. Seek to position on disk
        # 2. Read page_size bytes
        # 3. Deserialize into BPlusTreeNode
        pass

    def bulk_load(self, sorted_data):
        """
        Bulk load sorted data efficiently
        Much faster than sequential inserts
        Time: O(n) vs O(n log n)
        """
        if not sorted_data:
            return

        # Build leaves
        leaves = []
        max_keys_per_leaf = self.order - 1

        for i in range(0, len(sorted_data), max_keys_per_leaf):
            chunk = sorted_data[i:i + max_keys_per_leaf]
            leaf = BPlusTreeNode(is_leaf=True, order=self.order)
            leaf.keys = [k for k, v in chunk]
            leaf.values = [v for k, v in chunk]
            leaves.append(leaf)

        # Link leaves
        for i in range(len(leaves) - 1):
            leaves[i].next = leaves[i + 1]

        # Build internal levels bottom-up
        level = leaves
        while len(level) > 1:
            new_level = []
            max_children = self.order

            for i in range(0, len(level), max_children):
                children = level[i:i + max_children]
                internal = BPlusTreeNode(is_leaf=False, order=self.order)
                internal.children = children
                internal.keys = [child.keys[0] for child in children[1:]]
                new_level.append(internal)

            level = new_level

        self.root = level[0]

    def get_statistics(self):
        """Get tree statistics"""
        stats = {
            'height': self._get_height(),
            'num_nodes': self._count_nodes(),
            'num_keys': self._count_keys(),
            'disk_reads': self.disk_reads,
            'disk_writes': self.disk_writes,
            'avg_keys_per_node': 0
        }

        if stats['num_nodes'] > 0:
            stats['avg_keys_per_node'] = stats['num_keys'] / stats['num_nodes']

        return stats

    def _get_height(self, node=None):
        """Calculate tree height"""
        if node is None:
            node = self.root

        if node.is_leaf:
            return 1

        return 1 + self._get_height(node.children[0])

    def _count_nodes(self, node=None):
        """Count total nodes"""
        if node is None:
            node = self.root

        count = 1
        if not node.is_leaf:
            for child in node.children:
                count += self._count_nodes(child)

        return count

    def _count_keys(self, node=None):
        """Count total keys"""
        if node is None:
            node = self.root

        count = len(node.keys)
        if not node.is_leaf:
            for child in node.children:
                count += self._count_keys(child)

        return count
```

### Complexity Analysis

| Operation | Time | Disk I/O | Notes |
|-----------|------|----------|-------|
| Search | O(log_m n) | O(log_m n) | m = order (fanout) |
| Insert | O(log_m n) | O(log_m n) | May cause splits |
| Delete | O(log_m n) | O(log_m n) | May cause merges |
| Range query | O(log_m n + k) | O(log_m n + k/b) | k results, b per block |
| Bulk load | O(n) | O(n) | Much faster than n inserts |

**Key insight**: With order m = 100, height is very small:
- 1 million keys: height ≈ 3
- 1 billion keys: height ≈ 4

**Space**: O(n)

### Why B+ Trees for Databases

```python
def compare_trees_for_database():
    """
    Why B+ trees dominate database indexing
    """

    # Example: 1 million records

    # Binary Search Tree (AVL/Red-Black):
    bst_height = 20  # log₂(1,000,000) ≈ 20
    bst_disk_reads = 20  # One per level

    # B+ Tree (order 100):
    bplus_height = 3  # log₁₀₀(1,000,000) ≈ 3
    bplus_disk_reads = 3  # One per level

    print(f"BST: {bst_disk_reads} disk reads")
    print(f"B+ Tree: {bplus_disk_reads} disk reads")
    print(f"Improvement: {bst_disk_reads / bplus_disk_reads:.1f}x faster")

    # Range query example
    print("\nRange query for 1000 consecutive keys:")
    print(f"BST: Need to traverse tree 1000 times = ~20,000 disk reads")
    print(f"B+ Tree: Navigate to start (3) + scan leaves (10) = 13 disk reads")
    print(f"Improvement: ~1500x faster for range queries!")

# Output:
# BST: 20 disk reads
# B+ Tree: 3 disk reads
# Improvement: 6.7x faster
#
# Range query for 1000 consecutive keys:
# BST: Need to traverse tree 1000 times = ~20,000 disk reads
# B+ Tree: Navigate to start (3) + scan leaves (10) = 13 disk reads
# Improvement: ~1500x faster for range queries!
```

### Real-World Usage

**PostgreSQL:**
```sql
-- B+ tree index (default)
CREATE INDEX idx_users_email ON users(email);

-- Range query benefits from B+ tree
SELECT * FROM users WHERE email >= 'a' AND email < 'b';
```

**MySQL/InnoDB:**
- Primary key: Clustered B+ tree index
- Secondary keys: Non-clustered B+ tree indices
- Data stored in leaf nodes of primary key B+ tree

**SQLite:**
- Uses B+ trees for indices
- Stores entire rows in leaf nodes

### B+ Tree Variants

```python
class BStarTree(BPlusTree):
    """
    B* tree variant
    Nodes kept 2/3 full (instead of 1/2)
    Better space utilization
    """
    def __init__(self, order=4):
        super().__init__(order)
        self.min_occupancy = (2 * order) // 3  # 2/3 full

class BTreeWithDuplicates(BPlusTree):
    """
    B+ tree supporting duplicate keys
    Useful for non-unique indices
    """
    def insert(self, key, value):
        """Allow duplicate keys"""
        leaf = self._find_leaf(key)
        # Always insert, even if key exists
        i = self._find_key_position(leaf.keys, key)
        leaf.keys.insert(i, key)
        leaf.values.insert(i, value)

        if len(leaf.keys) > self.order - 1:
            self._split_leaf(leaf)

class ConcurrentBPlusTree(BPlusTree):
    """
    B+ tree with concurrent access support
    Uses latch coupling (lock coupling)
    """
    import threading

    def __init__(self, order=4):
        super().__init__(order)
        self.locks = {}  # Node locks

    def search_concurrent(self, key):
        """
        Search with concurrent access
        Uses latch coupling to allow multiple readers
        """
        # Simplified - real implementation uses read/write locks
        pass
```

## Comparison Summary

### Feature Comparison

| Feature | Splay Tree | Treap | AVL | Red-Black | B+ Tree |
|---------|-----------|-------|-----|-----------|---------|
| Balance | Amortized | Expected | Strict | Relaxed | High fanout |
| Height | O(log n) amortized | O(log n) expected | 1.44 log n | 2 log n | log_m n |
| Search | O(log n) amortized | O(log n) expected | **O(log n)** | O(log n) | O(log_m n) |
| Insert | O(log n) amortized | O(log n) expected | O(log n) | **O(log n)** | O(log_m n) |
| Delete | O(log n) amortized | O(log n) expected | O(log n) | **O(log n)** | O(log_m n) |
| Split | **O(log n)** | **O(log n)** | O(n) | O(n) | O(log n) |
| Merge | **O(log n)** | **O(log n)** | O(n) | O(n) | O(log n) |
| Range query | O(k + log n) | O(k + log n) | O(k + log n) | O(k + log n) | **O(k + log_m n)** |
| Memory per node | Keys only | Key + priority | Key + height | Key + color | Multiple keys |
| Complexity | Simple | Simple | Moderate | Complex | Complex |
| Self-optimizing | **Yes** | No | No | No | No |
| Guaranteed | No | No | **Yes** | **Yes** | **Yes** |

### Use Case Guide

```python
def choose_tree(requirements):
    """
    Decision guide for choosing tree structure
    """

    if requirements['medium'] == 'disk':
        return "B+ Tree - optimized for disk I/O"

    if requirements['pattern'] == 'temporal_locality':
        return "Splay Tree - recent items fast"

    if requirements['split_merge_heavy']:
        return "Treap or Splay Tree - efficient split/merge"

    if requirements['lookups'] > requirements['updates'] * 2:
        return "AVL Tree - optimized for searches"

    if requirements['updates'] > requirements['lookups']:
        return "Red-Black Tree - balanced update performance"

    if requirements['simplicity'] == 'high':
        return "Treap - simple probabilistic structure"

    if requirements['standard_library']:
        return "Red-Black Tree - used in STL, Java"

    return "Red-Black Tree - good general-purpose choice"

# Examples
print(choose_tree({
    'medium': 'disk',
    'pattern': 'random',
    'split_merge_heavy': False,
    'lookups': 1000,
    'updates': 100,
    'simplicity': 'medium',
    'standard_library': False
}))
# Output: B+ Tree - optimized for disk I/O

print(choose_tree({
    'medium': 'memory',
    'pattern': 'temporal_locality',
    'split_merge_heavy': False,
    'lookups': 1000,
    'updates': 100,
    'simplicity': 'medium',
    'standard_library': False
}))
# Output: Splay Tree - recent items fast

print(choose_tree({
    'medium': 'memory',
    'pattern': 'random',
    'split_merge_heavy': False,
    'lookups': 10000,
    'updates': 1000,
    'simplicity': 'medium',
    'standard_library': False
}))
# Output: AVL Tree - optimized for searches
```

### Performance Characteristics

```python
import random
import time

def benchmark_trees(n=10000):
    """Benchmark different tree structures"""

    # Generate test data
    keys = list(range(n))
    random.shuffle(keys)

    results = {}

    # Test each tree
    for tree_name, tree_class in [
        ('Splay', SplayTree),
        ('AVL', AVLTree),
        ('Red-Black', RedBlackTree),
    ]:
        tree = tree_class()

        # Insertion
        start = time.time()
        for key in keys:
            tree.insert(key)
        insert_time = time.time() - start

        # Lookups
        random.shuffle(keys)
        start = time.time()
        for key in keys[:1000]:
            tree.search(key)
        search_time = time.time() - start

        results[tree_name] = {
            'insert': insert_time,
            'search': search_time
        }

    return results

# Typical results (relative):
# Splay Tree: Fast inserts, adapts to access pattern
# AVL Tree: Slower inserts, fastest searches
# Red-Black: Balanced performance
```

## Best Practices

### 1. Tree Selection

```python
# Disk-based storage (database)
tree = DatabaseBPlusTree(order=100)

# In-memory, lookup-heavy
tree = AVLTree()

# In-memory, balanced workload
tree = RedBlackTree()

# Temporal locality access
tree = SplayTree()

# Simple, randomized
tree = Treap()
```

### 2. Order Selection for B+ Trees

```python
def calculate_optimal_order(page_size=4096, key_size=8, pointer_size=8):
    """
    Calculate optimal B+ tree order based on disk page size
    Goal: Fit one node in one disk page
    """
    # Internal node: (order-1) keys + order pointers
    # order * pointer_size + (order-1) * key_size ≤ page_size

    order = (page_size + key_size) // (key_size + pointer_size)
    return order

# Example for typical database:
order = calculate_optimal_order(page_size=4096, key_size=8, pointer_size=8)
print(f"Optimal order: {order}")  # ~256
```

### 3. Concurrency Considerations

```python
# Single-threaded: Use simplest appropriate structure
tree = AVLTree()

# Multi-threaded reads: Any structure + read-write lock
import threading
lock = threading.RLock()

def thread_safe_search(tree, key):
    with lock:
        return tree.search(key)

# Multi-threaded writes: Consider lock-free structures
# or use external synchronization
```

## Related Topics

- [Trees](trees.md) - Basic tree structures and concepts
- [Heaps](heaps.md) - Heap data structure
- [Complexity Guide](complexity_guide.md) - Algorithm analysis
- [Implementation Patterns](implementation_patterns.md) - Common coding patterns

## Summary

### Key Takeaways

1. **Splay Trees**: Self-optimizing for temporal locality, amortized O(log n)
2. **Treaps**: Simple randomized trees with expected O(log n)
3. **AVL**: Strictest balance, best for lookup-heavy workloads
4. **Red-Black**: Relaxed balance, best for balanced read/write
5. **B+ Trees**: High fanout, optimal for disk-based storage

### Quick Reference

```python
# Choose based on:
# 1. Storage medium (disk vs memory)
# 2. Access pattern (random vs temporal)
# 3. Workload (read-heavy vs write-heavy)
# 4. Complexity tolerance
# 5. Guaranteed vs amortized performance

# Memory + lookup-heavy → AVL
# Memory + balanced → Red-Black
# Memory + temporal → Splay
# Memory + simple → Treap
# Disk + database → B+ Tree
```

Remember: Choose the right tree for your specific use case. No single tree structure is best for everything!
