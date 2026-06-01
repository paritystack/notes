# Segment Tree

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use a Segment Tree](#when-to-use-a-segment-tree)
  - [Tree Structure and Indexing](#tree-structure-and-indexing)
  - [Memory Layout](#memory-layout)
- [Basic Segment Tree (Recursive)](#basic-segment-tree-recursive)
  - [Range Sum](#range-sum)
  - [Range Minimum](#range-minimum)
  - [Range Maximum](#range-maximum)
  - [Range GCD](#range-gcd)
- [Range Update + Point Query (Difference Trick)](#range-update--point-query-difference-trick)
- [Range Update + Range Query (Lazy Propagation)](#range-update--range-query-lazy-propagation)
  - [Range Add + Range Sum](#range-add--range-sum)
  - [Range Assign + Range Min](#range-assign--range-min)
  - [Combined Add and Assign](#combined-add-and-assign)
- [Iterative Segment Tree](#iterative-segment-tree)
  - [Bottom-Up Build](#bottom-up-build)
  - [Iterative Point Update](#iterative-point-update)
  - [Iterative Range Query](#iterative-range-query)
- [Generic / Templated Segment Tree](#generic--templated-segment-tree)
- [Segment Tree with Multiple Operations](#segment-tree-with-multiple-operations)
- [Searching on a Segment Tree](#searching-on-a-segment-tree)
  - [Find First Element ≥ X](#find-first-element--x)
  - [kth Element / Walk on Tree](#kth-element--walk-on-tree)
- [Segment Tree on Coordinates (Coordinate Compression)](#segment-tree-on-coordinates-coordinate-compression)
- [Merge Sort Tree](#merge-sort-tree)
- [2D Segment Tree](#2d-segment-tree)
- [Persistent Segment Tree](#persistent-segment-tree)
- [Segment Tree Beats (Overview)](#segment-tree-beats-overview)
- [Common Patterns](#common-patterns)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

A **segment tree** is a balanced binary tree that stores aggregate information about contiguous ranges of an array, enabling two operations to run in **O(log n)** time:

1. **Point or range update** of one or more array values.
2. **Range query** (sum, min, max, GCD, count of zeros, etc.) over any contiguous subarray.

It is one of the most versatile data structures in competitive programming. Any binary operation that is **associative** (and has a known identity element) can be supported. Many problems that look like "perform Q updates and Q queries on an array of size N" are solved with a segment tree in O((N + Q) log N).

Related data structures and notes:
- [[fenwick_tree]] — simpler, less general, but faster constant factor for prefix-sum style problems.
- [[sparse_table]] — answers static, idempotent range queries in O(1) but doesn't support updates.
- [[lca]] — uses segment trees or sparse tables internally for Euler-tour LCA.
- [[heavy_light_decomposition]] — decomposes a tree into chains so that segment trees handle path queries.
- [[mo_algorithm]] — an alternative when problems are offline and not naturally aggregable.

## ELI10 Explanation

Imagine you have a row of 8 number cards and you keep being asked: *"What is the sum of cards 3 through 6?"* and *"Change card 5 to a new value."*

A naive approach recomputes the sum every time, looking at every card — that's slow if the row is long. A segment tree builds a **pyramid of sums** over the row:

- The **bottom layer** is the row of 8 cards.
- The next layer up has 4 cells, each holding the sum of 2 cards beneath it.
- The next has 2 cells, each summing 4 cards.
- The top has 1 cell holding the sum of all 8.

To find the sum of cards 3 through 6, you don't need to look at every card. You walk down the pyramid, grabbing the **largest pre-computed blocks** that fit entirely inside your range. Worst case you grab about 2 × (height of pyramid) blocks — and the pyramid is only `log n` tall.

To change card 5, you update card 5 at the bottom and then walk **upward**, recomputing only the cells directly above it. That's also `log n` cells.

The key insight: with a balanced binary tree of summaries, both queries and updates touch at most `O(log n)` cells.

## Core Concepts

### When to Use a Segment Tree

Reach for a segment tree when **all** of these are true:

1. You have an array (or something that maps to an array — e.g., compressed coordinates, an Euler tour of a tree).
2. You need to mix **updates and queries** in arbitrary interleaved order.
3. The query is an **aggregate over a contiguous range** — sum, min, max, GCD, XOR, product mod p, count of distinct, etc.
4. The aggregate operation is **associative**: `f(a, f(b, c)) == f(f(a, b), c)`. This is required to combine partial results from sub-ranges.

**Identity element** (e.g., 0 for sum, +∞ for min) helps you handle empty ranges cleanly.

If the array is **static** (no updates), a [[sparse_table]] is often simpler and faster for idempotent ops (min, max, GCD). If you only need **prefix sums with point updates**, a [[fenwick_tree]] is shorter to code with a better constant factor.

### Tree Structure and Indexing

A segment tree over an array of size n uses a binary tree with n leaves. Internal nodes represent the union of their children's ranges.

There are two common indexing conventions:

**1-indexed, root at 1 (recursive):**
- Node `v` represents some range `[l, r]`.
- Left child = `2*v`, right child = `2*v + 1`.
- Allocate `4 * n` slots to be safe (tight bound is `2 * 2^⌈log2(n)⌉`, but `4n` is the standard safe choice).

**0-indexed, iterative (Atcoder/Codeforces style):**
- Round n up to a power of two (or use the size-n flat layout from "Efficient and easy segment trees").
- Leaves live at indices `[n, 2n)`, internal nodes at `[1, n)`. Index 0 is unused.
- Parent of `v` = `v // 2`, left child = `2*v`, right child = `2*v + 1`.

The recursive form is more flexible (lazy propagation, persistent variants). The iterative form is shorter, faster in practice, and great for problems without lazy updates.

### Memory Layout

For a segment tree of n leaves:
- **Recursive (1-indexed)**: allocate an array of size `4n` to safely cover all possible internal nodes.
- **Iterative (rounded power of two)**: allocate an array of size `2 * next_pow2(n)`.
- **Iterative (flat, no rounding)**: allocate exactly `2n` slots.

For lazy propagation, you also need a `lazy[]` array of the same size as the tree.

## Basic Segment Tree (Recursive)

The recursive form is the easiest to read and the most flexible. We'll start with sum, then swap the aggregate function for other associative operations.

### Range Sum

```python
class SegTreeSum:
    """
    Recursive segment tree for range sum + point update.
    Indices in the underlying array are 0..n-1.
    Internal tree is 1-indexed; node 1 is the root.
    """

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, value):
        """Set data[idx] = value."""
        self._update(1, 0, self.n - 1, idx, value)

    def _update(self, node, l, r, idx, value):
        if l == r:
            self.tree[node] = value
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, value)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, ql, qr):
        """Return sum of data[ql..qr] inclusive."""
        if ql > qr:
            return 0
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0  # disjoint
        if ql <= l and r <= qr:
            return self.tree[node]  # fully covered
        mid = (l + r) // 2
        return (
            self._query(2 * node, l, mid, ql, qr)
            + self._query(2 * node + 1, mid + 1, r, ql, qr)
        )


# Example
st = SegTreeSum([1, 3, 5, 7, 9, 11])
print(st.query(1, 3))  # 3 + 5 + 7 = 15
st.update(1, 10)       # data becomes [1, 10, 5, 7, 9, 11]
print(st.query(1, 3))  # 10 + 5 + 7 = 22
print(st.query(0, 5))  # 1 + 10 + 5 + 7 + 9 + 11 = 43
```

**How it works:**
- Each node covers an interval `[l, r]`. Leaves cover singletons.
- A query splits the requested range into O(log n) maximal sub-intervals, each fully matching some node.
- A point update walks from a leaf up to the root, refreshing O(log n) ancestors.

### Range Minimum

Just swap the aggregate (`+` → `min`) and the identity (`0` → `+∞`).

```python
import math

class SegTreeMin:
    IDENTITY = math.inf

    def __init__(self, data):
        self.n = len(data)
        self.tree = [self.IDENTITY] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    @staticmethod
    def _combine(a, b):
        return a if a < b else b

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, value):
        self._update(1, 0, self.n - 1, idx, value)

    def _update(self, node, l, r, idx, value):
        if l == r:
            self.tree[node] = value
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, value)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, value)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        if ql > qr:
            return self.IDENTITY
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return self.IDENTITY
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        return self._combine(
            self._query(2 * node, l, mid, ql, qr),
            self._query(2 * node + 1, mid + 1, r, ql, qr),
        )
```

### Range Maximum

```python
class SegTreeMax(SegTreeMin):
    IDENTITY = -math.inf

    @staticmethod
    def _combine(a, b):
        return a if a > b else b
```

### Range GCD

Range GCD is associative (and idempotent), so it fits the same skeleton. The identity is 0 because `gcd(x, 0) == x` for any x.

```python
from math import gcd

class SegTreeGCD(SegTreeMin):
    IDENTITY = 0

    @staticmethod
    def _combine(a, b):
        return gcd(a, b)
```

The same skeleton trivially supports XOR (identity 0), product mod p (identity 1), boolean AND/OR (identities True/False), bitwise AND (identity all-ones), and so on.

## Range Update + Point Query (Difference Trick)

If you only need **range updates** and **point queries** (no range queries), you can avoid lazy propagation entirely. Maintain a difference array `d[]` where `d[i] = a[i] - a[i-1]`. Then:

- Range add of `+v` on `[l, r]` becomes two point updates: `d[l] += v`, `d[r+1] -= v`.
- The value at index `i` is the prefix sum `d[0] + d[1] + ... + d[i]`, which is a prefix sum query.

Build a sum segment tree over `d[]` and you get range-add + point-query in O(log n) per op. This is the simplest non-trivial use of a segment tree and a great mental warm-up before lazy propagation.

```python
class RangeAddPointQuery:
    """
    Range add + point query using a segment tree on the difference array.
    """

    def __init__(self, n):
        self.n = n
        self.st = SegTreeSum([0] * n)

    def range_add(self, l, r, value):
        """Add `value` to a[l..r] inclusive."""
        # update d[l] += value
        cur = self.st.query(l, l)
        self.st.update(l, cur + value)
        if r + 1 < self.n:
            cur = self.st.query(r + 1, r + 1)
            self.st.update(r + 1, cur - value)

    def point_query(self, idx):
        """Return a[idx] = prefix sum of d up to idx."""
        return self.st.query(0, idx)
```

In practice a [[fenwick_tree]] is the natural choice for this pattern — it's shorter and ~2× faster.

## Range Update + Range Query (Lazy Propagation)

When you need **both** range updates and range queries, you need **lazy propagation**: a `lazy[]` array that stashes pending updates at internal nodes and pushes them down only when necessary.

The pattern is:
1. **Apply** an update to a node by modifying its aggregate value AND saving the update in `lazy[node]`.
2. **Push** the lazy value down to children before descending past a node.
3. **Compose** lazy values when stacking multiple pending updates.

### Range Add + Range Sum

The canonical example. Add `+v` to all elements in `[l, r]`, and query the sum of any range.

When we add `v` to every element in a node covering range of length `len`, the node's stored sum increases by `v * len`. We record `v` in `lazy[node]` so descendants can apply it later.

```python
class SegTreeRangeAddSum:
    """
    Range add + range sum with lazy propagation.
    """

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _apply(self, node, l, r, value):
        """Apply 'add value' to the whole range [l, r] under `node`."""
        self.tree[node] += value * (r - l + 1)
        self.lazy[node] += value

    def _push(self, node, l, r):
        """Push pending updates from `node` down to its children."""
        if self.lazy[node] != 0:
            mid = (l + r) // 2
            self._apply(2 * node, l, mid, self.lazy[node])
            self._apply(2 * node + 1, mid + 1, r, self.lazy[node])
            self.lazy[node] = 0

    def update(self, ql, qr, value):
        """Add `value` to a[ql..qr] inclusive."""
        if ql > qr:
            return
        self._update(1, 0, self.n - 1, ql, qr, value)

    def _update(self, node, l, r, ql, qr, value):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply(node, l, r, value)
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self._update(2 * node, l, mid, ql, qr, value)
        self._update(2 * node + 1, mid + 1, r, ql, qr, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, ql, qr):
        if ql > qr:
            return 0
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        return (
            self._query(2 * node, l, mid, ql, qr)
            + self._query(2 * node + 1, mid + 1, r, ql, qr)
        )


# Example
st = SegTreeRangeAddSum([0, 0, 0, 0, 0])
st.update(1, 3, 5)         # add 5 to indices 1..3
print(st.query(0, 4))      # 5+5+5 = 15
st.update(0, 4, 1)         # add 1 to everyone
print(st.query(0, 4))      # 15 + 5 = 20
print(st.query(2, 2))      # 5 + 1 = 6
```

### Range Assign + Range Min

For range **assignment** (set everything in `[l, r]` to `v`), the lazy value overwrites rather than accumulates. We need a sentinel "no pending assignment" — typically `None` or some impossible value.

```python
import math

class SegTreeRangeAssignMin:
    IDENTITY = math.inf
    NO_ASSIGN = None

    def __init__(self, data):
        self.n = len(data)
        self.tree = [self.IDENTITY] * (4 * self.n)
        self.lazy = [self.NO_ASSIGN] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _apply(self, node, value):
        self.tree[node] = value
        self.lazy[node] = value

    def _push(self, node):
        if self.lazy[node] is not self.NO_ASSIGN:
            self._apply(2 * node, self.lazy[node])
            self._apply(2 * node + 1, self.lazy[node])
            self.lazy[node] = self.NO_ASSIGN

    def update(self, ql, qr, value):
        self._update(1, 0, self.n - 1, ql, qr, value)

    def _update(self, node, l, r, ql, qr, value):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply(node, value)
            return
        self._push(node)
        mid = (l + r) // 2
        self._update(2 * node, l, mid, ql, qr, value)
        self._update(2 * node + 1, mid + 1, r, ql, qr, value)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return self.IDENTITY
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node)
        mid = (l + r) // 2
        return min(
            self._query(2 * node, l, mid, ql, qr),
            self._query(2 * node + 1, mid + 1, r, ql, qr),
        )
```

Notice that range-assign + range-min doesn't need to scale by range length — assigning to a range of any size still gives a minimum equal to the assigned value.

### Combined Add and Assign

What if you want to support **both** range add and range assign on the same tree? You need a richer lazy state with **two** components and well-defined composition rules.

Conventions:
- Assignment is "stronger" than add. If a pending assign exists at a node, any new add must collapse into it.
- A new assign wipes out any pending add.

```python
class SegTreeAddAssignSum:
    """
    Range add, range assign, and range sum.
    Lazy stores (add_val, assign_val) where assign_val=None means "no pending assign".
    Application order: assign first, then add.
    """

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy_add = [0] * (4 * self.n)
        self.lazy_assign = [None] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _apply_assign(self, node, l, r, value):
        self.tree[node] = value * (r - l + 1)
        self.lazy_assign[node] = value
        self.lazy_add[node] = 0  # assign wipes any pending add

    def _apply_add(self, node, l, r, value):
        self.tree[node] += value * (r - l + 1)
        if self.lazy_assign[node] is not None:
            self.lazy_assign[node] += value
        else:
            self.lazy_add[node] += value

    def _push(self, node, l, r):
        mid = (l + r) // 2
        if self.lazy_assign[node] is not None:
            v = self.lazy_assign[node]
            self._apply_assign(2 * node, l, mid, v)
            self._apply_assign(2 * node + 1, mid + 1, r, v)
            self.lazy_assign[node] = None
        if self.lazy_add[node] != 0:
            v = self.lazy_add[node]
            self._apply_add(2 * node, l, mid, v)
            self._apply_add(2 * node + 1, mid + 1, r, v)
            self.lazy_add[node] = 0

    def range_assign(self, ql, qr, value):
        self._assign(1, 0, self.n - 1, ql, qr, value)

    def _assign(self, node, l, r, ql, qr, value):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply_assign(node, l, r, value)
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self._assign(2 * node, l, mid, ql, qr, value)
        self._assign(2 * node + 1, mid + 1, r, ql, qr, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_add(self, ql, qr, value):
        self._add(1, 0, self.n - 1, ql, qr, value)

    def _add(self, node, l, r, ql, qr, value):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self._apply_add(node, l, r, value)
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self._add(2 * node, l, mid, ql, qr, value)
        self._add(2 * node + 1, mid + 1, r, ql, qr, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        return (
            self._query(2 * node, l, mid, ql, qr)
            + self._query(2 * node + 1, mid + 1, r, ql, qr)
        )
```

**Composition rule**: the order matters. If we always apply assignment before add, then composing operations becomes "if an assign is queued, fold the new add into the assigned value; otherwise stack adds normally."

## Iterative Segment Tree

For problems with **only point updates** and **only range queries** (no lazy propagation), an iterative segment tree is shorter, faster, and easier to memorize. This is the famous "Efficient and Easy Segment Tree" pattern from the Codeforces blog.

### Bottom-Up Build

```python
class IterativeSegTree:
    """
    Iterative segment tree for point update + range query.
    Uses 1-indexed internal nodes with leaves at [n, 2n).
    Indices in the user array are 0..n-1.
    """

    def __init__(self, data, combine, identity):
        self.n = len(data)
        self.combine = combine
        self.identity = identity
        self.tree = [identity] * (2 * self.n)

        # place leaves
        for i, v in enumerate(data):
            self.tree[self.n + i] = v
        # build internal nodes bottom-up
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = combine(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, idx, value):
        """Point update: data[idx] = value."""
        i = idx + self.n
        self.tree[i] = value
        i //= 2
        while i:
            self.tree[i] = self.combine(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def query(self, l, r):
        """Inclusive range query on data[l..r]."""
        res_l = self.identity
        res_r = self.identity
        l += self.n
        r += self.n + 1  # convert to half-open [l, r)
        while l < r:
            if l & 1:
                res_l = self.combine(res_l, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res_r = self.combine(self.tree[r], res_r)
            l //= 2
            r //= 2
        return self.combine(res_l, res_r)


# Example: range sum
st = IterativeSegTree([1, 3, 5, 7, 9, 11], combine=lambda a, b: a + b, identity=0)
print(st.query(1, 3))  # 15
st.update(1, 10)
print(st.query(1, 3))  # 22

# Example: range min
import math
st_min = IterativeSegTree([4, 1, 7, 3, 9], combine=min, identity=math.inf)
print(st_min.query(1, 3))  # 1
```

**Why the asymmetric accumulator?** Because the combine operation may not be commutative (e.g., matrix multiplication, string concatenation). We accumulate left fragments in `res_l` left-to-right and right fragments in `res_r` right-to-left, then combine at the end.

For commutative ops (sum, min, max, GCD, XOR), you can simplify to a single accumulator.

### Iterative Point Update

The point update is straightforward — overwrite the leaf and walk up to the root, refreshing each ancestor.

### Iterative Range Query

The trick: at each level, if `l` is a right child, it contributes its own value and you move right; if `r` is a right child (in the half-open form), the left sibling at `r-1` contributes. Then both indices move up a level. This processes O(log n) cells without any recursion.

## Generic / Templated Segment Tree

For maximum reuse, define a segment tree parameterized by:
- The element type and identity.
- The combine function.
- (Optionally) a lazy type, identity, apply function, and compose function.

```python
class GenericSegTree:
    """
    Generic point-update segment tree.
      combine: associative function (a, b) -> a*b
      identity: identity element for combine
    """

    def __init__(self, size_or_data, combine, identity):
        if isinstance(size_or_data, int):
            data = [identity] * size_or_data
        else:
            data = list(size_or_data)
        self.n = len(data)
        self.combine = combine
        self.identity = identity
        self.tree = [identity] * (4 * max(1, self.n))
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self.combine(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, value):
        self._update(1, 0, self.n - 1, idx, value)

    def _update(self, node, l, r, idx, value):
        if l == r:
            self.tree[node] = value
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, value)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, value)
        self.tree[node] = self.combine(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        if ql > qr or self.n == 0:
            return self.identity
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return self.identity
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        return self.combine(
            self._query(2 * node, l, mid, ql, qr),
            self._query(2 * node + 1, mid + 1, r, ql, qr),
        )
```

## Segment Tree with Multiple Operations

Sometimes you need a node that stores a small **struct** rather than a single value. Classic examples:

**Max subarray sum (Kadane on a tree).** Each node stores `(total, prefix_max, suffix_max, best)`:

```python
class MaxSubarraySegTree:
    """
    Range max subarray sum with point updates.
    Each node stores four values:
        total:      sum of the range
        prefix:     best sum starting at the left endpoint
        suffix:     best sum ending at the right endpoint
        best:       best subarray sum entirely inside the range
    """

    IDENTITY = (0, -math.inf, -math.inf, -math.inf)

    def __init__(self, data):
        self.n = len(data)
        self.tree = [self.IDENTITY] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1, data)

    @staticmethod
    def _leaf(x):
        return (x, x, x, x)

    @staticmethod
    def _combine(a, b):
        if a == MaxSubarraySegTree.IDENTITY:
            return b
        if b == MaxSubarraySegTree.IDENTITY:
            return a
        total = a[0] + b[0]
        prefix = max(a[1], a[0] + b[1])
        suffix = max(b[2], b[0] + a[2])
        best = max(a[3], b[3], a[2] + b[1])
        return (total, prefix, suffix, best)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = self._leaf(data[l])
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, value):
        self._update(1, 0, self.n - 1, idx, value)

    def _update(self, node, l, r, idx, value):
        if l == r:
            self.tree[node] = self._leaf(value)
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, value)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, value)
        self.tree[node] = self._combine(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)[3]

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return self.IDENTITY
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        return self._combine(
            self._query(2 * node, l, mid, ql, qr),
            self._query(2 * node + 1, mid + 1, r, ql, qr),
        )
```

The same pattern generalizes to "longest run of equal characters," "number of inversions in a range" (combine cross-counts), and so on.

## Searching on a Segment Tree

A segment tree lets you do more than aggregate — you can **descend the tree** to find the first/last index satisfying some predicate in O(log n), provided the predicate has a useful monotonic structure with respect to the aggregate.

### Find First Element ≥ X

Given a max segment tree, find the smallest index `i ≥ l` such that `a[i] ≥ X`.

```python
def first_at_least(seg, x, l=0):
    """
    Return smallest index i >= l with a[i] >= x, or -1 if none.
    `seg` is a SegTreeMax over a[0..n-1].
    """

    def descend(node, lo, hi):
        if seg.tree[node] < x:
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if seg.tree[2 * node] >= x:
                node = 2 * node
                hi = mid
            else:
                node = 2 * node + 1
                lo = mid + 1
        return lo

    # walk the cover decomposition of [l, n-1], pick first segment whose
    # max >= x, then descend within it.
    def go(node, lo, hi):
        if hi < l or seg.tree[node] < x:
            return -1
        if lo >= l:
            return descend(node, lo, hi)
        mid = (lo + hi) // 2
        left = go(2 * node, lo, mid)
        if left != -1:
            return left
        return go(2 * node + 1, mid + 1, hi)

    return go(1, 0, seg.n - 1)
```

### kth Element / Walk on Tree

If your segment tree counts elements (e.g., frequency over compressed values), you can find the **kth smallest** in O(log n) by walking left or right based on the count stored at each subtree.

```python
def kth_smallest(seg, k):
    """
    Walk the tree to find the kth-smallest element (1-indexed).
    `seg.tree[node]` holds the count of items in node's range.
    Assumes leaves correspond to (compressed) values 0..n-1.
    Returns the value index, or -1 if k is out of range.
    """
    if seg.tree[1] < k:
        return -1
    node = 1
    lo, hi = 0, seg.n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        left_count = seg.tree[2 * node]
        if left_count >= k:
            node = 2 * node
            hi = mid
        else:
            k -= left_count
            node = 2 * node + 1
            lo = mid + 1
    return lo
```

This is the foundation of an **order-statistics tree** built on top of a segment tree of frequencies.

## Segment Tree on Coordinates (Coordinate Compression)

When the array's logical domain is huge (e.g., 10^9 possible x-coordinates) but only N distinct values appear, **compress** them to 0..N-1 first.

```python
def coordinate_compress(values):
    """Return (compressed: list[int], decompress: list[T])."""
    sorted_unique = sorted(set(values))
    rank = {v: i for i, v in enumerate(sorted_unique)}
    return [rank[v] for v in values], sorted_unique


# Example: count of x in [lo, hi] across an event stream
events = [(7, +1), (3, +1), (3, -1), (10, +1)]
xs = [x for x, _ in events]
compressed, decomp = coordinate_compress(xs)
n = len(decomp)
st = SegTreeSum([0] * n)
for (x, delta), ci in zip(events, compressed):
    st.update(ci, st.query(ci, ci) + delta)
# Now query "count in [lo, hi]" via binary search on decomp + range sum.
```

This trick is what lets you do range queries on points in 2D (sweep + segment tree on y-coordinate), interval scheduling, and similar problems where the universe is sparse.

## Merge Sort Tree

A **merge sort tree** is a segment tree where each node stores the **sorted list** of values in its range. Build time: O(n log n) total, O(n log n) memory. It answers offline "how many elements in `[l, r]` are ≤ X" in O(log² n) per query via binary search inside the O(log n) covering nodes.

```python
import bisect

class MergeSortTree:
    """
    Segment tree where each node holds the sorted multiset of its range.
    Supports query: count of values in a[l..r] that are <= x.
    No efficient updates — this is a static structure.
    """

    def __init__(self, data):
        self.n = len(data)
        self.tree = [[] for _ in range(4 * self.n)]
        self._build(1, 0, self.n - 1, data)

    def _build(self, node, l, r, data):
        if l == r:
            self.tree[node] = [data[l]]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid, data)
        self._build(2 * node + 1, mid + 1, r, data)
        # merge two sorted lists
        a, b = self.tree[2 * node], self.tree[2 * node + 1]
        self.tree[node] = self._merge(a, b)

    @staticmethod
    def _merge(a, b):
        out, i, j = [], 0, 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                out.append(a[i]); i += 1
            else:
                out.append(b[j]); j += 1
        out.extend(a[i:]); out.extend(b[j:])
        return out

    def count_leq(self, ql, qr, x):
        return self._query(1, 0, self.n - 1, ql, qr, x)

    def _query(self, node, l, r, ql, qr, x):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return bisect.bisect_right(self.tree[node], x)
        mid = (l + r) // 2
        return (
            self._query(2 * node, l, mid, ql, qr, x)
            + self._query(2 * node + 1, mid + 1, r, ql, qr, x)
        )
```

When you need updates as well, a **persistent segment tree** or a **wavelet tree** is usually preferable.

## 2D Segment Tree

A 2D segment tree maintains a tree of trees — each node in the outer tree is itself a segment tree over the second dimension. Build and query are both O(log² n) per operation; memory is O(n² log² n) for the dense version (so it's only practical for small grids or as a sparse / "segment tree of sorted vectors").

For most 2D problems, prefer:
- **2D Fenwick** if you only need point updates + 2D rectangle prefix sums.
- **Offline sweep line + 1D segment tree** if queries can be sorted.
- **k-d tree** for non-axis-aligned queries.

A practical 2D BIT example lives in [[fenwick_tree]].

## Persistent Segment Tree

A **persistent** segment tree keeps every historical version of the tree by creating O(log n) new nodes per update rather than mutating in place. Each version is identified by the root it returns.

Use cases:
- **kth element in a range** (Wavelet-tree alternative, "Persistence + range" trick).
- **Time-travel queries**: "what was the state after the i-th update?"
- **Online problems** that need read access to past versions.

Sketch:

```python
class PersistentSegTree:
    """
    Sketch of a persistent segment tree over [0, n).
    Nodes are dicts: {'l': lc, 'r': rc, 'val': v}.
    Each update returns a new root.
    """

    def __init__(self, n):
        self.n = n
        self.roots = [self._build(0, n - 1)]

    def _build(self, l, r):
        if l == r:
            return {'l': None, 'r': None, 'val': 0}
        mid = (l + r) // 2
        return {
            'l': self._build(l, mid),
            'r': self._build(mid + 1, r),
            'val': 0,
        }

    def update(self, prev_version, idx, delta):
        new_root = self._update(self.roots[prev_version], 0, self.n - 1, idx, delta)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node, l, r, idx, delta):
        if l == r:
            return {'l': None, 'r': None, 'val': node['val'] + delta}
        mid = (l + r) // 2
        if idx <= mid:
            new_l = self._update(node['l'], l, mid, idx, delta)
            new_r = node['r']
        else:
            new_l = node['l']
            new_r = self._update(node['r'], mid + 1, r, idx, delta)
        return {'l': new_l, 'r': new_r, 'val': new_l['val'] + new_r['val']}

    def query(self, version, ql, qr):
        return self._query(self.roots[version], 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return 0
        if ql <= l and r <= qr:
            return node['val']
        mid = (l + r) // 2
        return self._query(node['l'], l, mid, ql, qr) + self._query(node['r'], mid + 1, r, ql, qr)
```

For competitive Python, persistent trees are usually implemented with parallel arrays (`left[]`, `right[]`, `val[]`) and an integer "next node" counter rather than dict objects — the constant factor is far lower.

## Segment Tree Beats (Overview)

**Segment Tree Beats** is an advanced lazy-propagation technique invented by Jiry that supports operations like:
- "For each i in [l, r], set a[i] = min(a[i], x)" (chmin).
- "Sum query on [l, r]" alongside the chmin.

Each node tracks the max, the second max, and the count of maxes. The update only descends into a subtree if `x` lies strictly between the second max and the max; otherwise it can be applied in O(1) to the whole subtree. Amortized complexity is O((n + q) log² n).

The implementation is substantially more complex than vanilla lazy propagation and is rarely needed outside specific olympiad problems. If you encounter "chmin/chmax + sum" or "Histogram on Segments" / "Picture" style problems, look this up specifically.

## Common Patterns

| Problem pattern | Tree variant |
|---|---|
| "Range sum + point update" | Basic sum segment tree (or [[fenwick_tree]]) |
| "Range min/max + point update" | Basic min/max segment tree (or [[sparse_table]] if static) |
| "Range add + range sum" | Lazy: add, sum |
| "Range assign + range sum" | Lazy: assign sentinel + sum scaled by range length |
| "Range add + range min/max" | Lazy: add (no length scaling for min/max) |
| "Max subarray sum + point update" | Struct nodes (prefix/suffix/best/total) |
| "First index ≥ X" | Descend the tree |
| "kth smallest in range" | Persistent segment tree or merge sort tree |
| "Count distinct in range (offline)" | Sweep + segment tree of "last occurrence" indices |
| "Range queries on points in 2D" | Sweep + 1D segment tree on the other axis |
| "Path query on a tree" | Segment tree + [[heavy_light_decomposition]] |
| "Subtree query on a tree" | Segment tree on Euler-tour positions |

## Interview & Contest Problems

**LeetCode**
- 307. Range Sum Query — Mutable (basic sum)
- 308. Range Sum Query 2D — Mutable (2D BIT or 2D segment tree)
- 327. Count of Range Sum (merge sort tree or BIT on compressed coordinates)
- 715. Range Module (range assign + range query on intervals)
- 732. My Calendar III (sweep + segment tree)
- 850. Rectangle Area II (sweep line + segment tree of lengths)
- 1157. Online Majority Element in Subarray (segment tree with frequency)

**Classic competitive problems**
- "Hotel" / "Hotel queries" (find first segment of K free rooms)
- "Stars" / Inversions count (BIT or segment tree on compressed values)
- "Picture" / Klee's algorithm (segment tree of segments)
- "K-th order statistic in a range" (persistent segment tree)
- "Subarray with given XOR" (segment tree + bit tricks)
- "Maximum sum of non-overlapping subarrays" (struct nodes)

## Complexity Analysis

| Operation | Recursive | Iterative |
|---|---|---|
| Build | O(n) | O(n) |
| Point update | O(log n) | O(log n) |
| Range query | O(log n) | O(log n) |
| Range update (lazy) | O(log n) amortized | — (rarely used iteratively) |
| Space | O(4n) | O(2n) |

For **segment tree beats**: O((n + q) log² n) amortized for the chmin/chmax + sum bundle.

For **persistent segment tree**: O(log n) extra nodes per update; memory grows with the update count.

For **merge sort tree**: O(n log n) build and memory, O(log² n) per range count query.

## Common Pitfalls

1. **Forgetting to allocate 4 * n.** A tighter bound exists (`2 * 2^⌈log2 n⌉`) but `4n` is the safe rule of thumb. Off-by-one allocations are a very common WA source.
2. **Off-by-one in inclusive vs. half-open ranges.** Pick one convention (most templates here use **inclusive** `[l, r]`) and stick with it. The iterative implementation uses **half-open** `[l, r)` internally; convert at the boundary.
3. **Forgetting to push lazy before descending.** Skipping `_push` while recursing into children is the #1 lazy-propagation bug.
4. **Wrong scaling for range updates.** Range "add v on [l, r]" with a sum aggregate must add `v * (r - l + 1)` at any node fully covered by the update.
5. **Incorrect lazy composition.** For range-assign + range-add together, you must decide an order and ensure that stacking the two follows it consistently.
6. **Mutating Python lists at deep recursion.** CPython's default recursion limit is ~1000. For n up to 10^5, set `sys.setrecursionlimit(...)` or use the iterative version.
7. **Identity collisions.** Using `math.inf` for an int problem is OK, but be careful with `-1` or `0` as sentinels — pick something the data can't naturally produce.
8. **Non-associative combine functions.** Segment trees require associativity. "Average" is not directly associative — store `(sum, count)` per node instead.
9. **Updating before building.** If you forget to call `_build` for `n > 0`, you'll get all-zeros until an update arrives.
10. **Confusing 0-indexed tree vs. 1-indexed tree.** Mixing the two conventions in the same file is the most common source of silent index errors.

## Practice Problems

| Source | Problem |
|---|---|
| Codeforces EDU | ITMO Segment Tree (Steps 1–5) |
| LeetCode | 307, 308, 315, 327, 715, 850, 1157 |
| SPOJ | HORRIBLE, LITE, KQUERY, GSS1–GSS5 |
| AtCoder | ABC 153 F, ABC 157 E, ABC 185 F, ABC 287 G |
| CSES | Dynamic Range Sum Queries, Range Update Queries, Hotel Queries, List Removals, Polynomial Queries |
| Codeforces | EDU Segment Tree, 339D, 380C, 522D, 803G |

## Additional Resources

- "Efficient and easy segment trees" — Codeforces blog by Al.Cash (the iterative template here).
- ITMO Codeforces EDU course on segment trees (free, with judge).
- CP-Algorithms: "Segment Tree" and "Segment Tree Beats" articles.
- "Competitive Programmer's Handbook" by Antti Laaksonen — chapter on segment trees.
- "Competitive Programming 4" by Halim & Halim — segment tree case studies.

## Where this connects

- [Fenwick tree](fenwick_tree.md) — simpler alternative for prefix sums only; less code but less general
- [Sparse table](sparse_table.md) — O(1) RMQ for static data; segment tree handles dynamic updates
- [Heavy-light decomposition](heavy_light_decomposition.md) — uses segment trees for path queries on trees
- [Data structures/segment_trees](../data_structures/segment_trees.md) — the data structure reference page
