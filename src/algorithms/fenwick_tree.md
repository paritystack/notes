# Fenwick Tree (Binary Indexed Tree)

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use a Fenwick Tree](#when-to-use-a-fenwick-tree)
  - [The Lowest-Set-Bit Trick](#the-lowest-set-bit-trick)
  - [Why It Works](#why-it-works)
- [Basic Fenwick Tree](#basic-fenwick-tree)
  - [Point Update + Prefix Sum](#point-update--prefix-sum)
  - [Range Sum Query](#range-sum-query)
  - [Construction in O(n)](#construction-in-on)
- [Range Update + Point Query](#range-update--point-query)
- [Range Update + Range Query (Two-BIT Trick)](#range-update--range-query-two-bit-trick)
- [Fenwick Tree on Other Operations](#fenwick-tree-on-other-operations)
  - [Prefix XOR](#prefix-xor)
  - [Prefix Max with Restrictions](#prefix-max-with-restrictions)
  - [Multiplicative Prefix](#multiplicative-prefix)
- [2D Fenwick Tree](#2d-fenwick-tree)
  - [2D Point Update + Rectangle Sum](#2d-point-update--rectangle-sum)
  - [2D Range Update + Point Query](#2d-range-update--point-query)
- [Finding the kth Smallest in O(log n)](#finding-the-kth-smallest-in-olog-n)
- [Counting Inversions](#counting-inversions)
- [Offline Coordinate Compression Pattern](#offline-coordinate-compression-pattern)
- [BIT vs. Segment Tree](#bit-vs-segment-tree)
- [Common Patterns](#common-patterns)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

A **Fenwick tree** (also called a **Binary Indexed Tree** or **BIT**) is a compact data structure that supports two operations in **O(log n)** time:

1. **Point update**: add a value to a single index.
2. **Prefix aggregate**: compute the sum (or XOR, or any other group operation) of a prefix.

From those two primitives you get **range sum queries** via the standard subtraction trick: `sum(l..r) = prefix(r) - prefix(l-1)`.

It is the workhorse of competitive programming for any problem that smells like "running total + updates":
- Number of elements ≤ X seen so far (compressed BIT).
- Inversion counting in O(n log n).
- 2D rectangle sums on a static grid with point updates.
- Sliding-window order statistics via Fenwick + binary lift.

Related notes:
- [[segment_tree]] — more general (any associative op, range updates with lazy), but ~2× larger constant factor.
- [[sparse_table]] — for static idempotent queries, no updates.
- [[mo_algorithm]] — for offline range queries where the aggregate isn't easily decomposable.

## ELI10 Explanation

Imagine you keep a daily diary of "how many cookies I ate today" and frequently ask "how many cookies total from day 1 through day d?"

If you store one number per day and add them up on every query, that's slow. If you store the running total instead, queries are O(1), but a correction ("oops, I forgot 3 cookies on day 5") forces you to update **every** running total from day 5 onward.

The Fenwick tree gives you the best of both worlds. It stores a clever set of **partial sums** — not one per day and not one giant running total, but `log n` carefully-chosen overlapping sums. The size of each partial sum is determined by the binary representation of the day number.

When you query "total through day d", you walk down the binary expansion of d, adding ~log d partial sums. When you update day i, you walk up through ~log n partial sums that contain day i. Both walks use the same trick: **strip the lowest set bit** of the index at each step.

The result: every update and every prefix query touches only `log n` cells.

## Core Concepts

### When to Use a Fenwick Tree

Reach for a Fenwick tree when **all** of these hold:

1. You need **prefix-style aggregates** with **point updates** (or one of the canonical extensions).
2. The aggregate is a **group operation**: associative, with an identity, **and an inverse**. Sum, XOR, and "multiplication in a field" qualify. Min, max, and GCD do **not** (no inverse).
3. The query domain is a single dimension (or two, for 2D BIT) of size known up front.

If the aggregate has no inverse, fall back to [[segment_tree]] or — if the array is static — [[sparse_table]].

### The Lowest-Set-Bit Trick

The Fenwick tree's magic identifier is `i & -i`, the **lowest set bit** of `i`.

In two's complement, `-i` flips every bit of `i` and adds 1, so `i & -i` isolates just the lowest 1-bit. Examples:

```
 i = 12 = 0b1100   →  i & -i = 0b0100 = 4
 i =  6 = 0b0110   →  i & -i = 0b0010 = 2
 i =  7 = 0b0111   →  i & -i = 0b0001 = 1
 i = 16 = 0b10000  →  i & -i = 0b10000 = 16
```

The Fenwick tree is **1-indexed**. Cell `tree[i]` stores the sum of `a[i - lowbit(i) + 1 .. i]`. So cell 12 stores the sum of indices 9..12; cell 8 stores 1..8; cell 6 stores 5..6; cell 7 stores just index 7.

To compute `prefix(i)`, you start at `i`, take `tree[i]`, then jump to `i -= lowbit(i)` and repeat until `i == 0`.

To **update** index `i` by `+v`, you start at `i`, add `v` to `tree[i]`, then jump to `i += lowbit(i)` until `i > n`.

### Why It Works

Any positive integer `i` has a unique binary expansion. The set of partial sums `[i - lowbit(i) + 1, i]` for all i forms a **proper covering** such that any prefix `[1..k]` decomposes into a disjoint union of at most `popcount(k) ≤ log₂ k + 1` such intervals. That gives you the O(log n) prefix query.

Similarly, the cells that include index `i` are exactly those at positions `i, i + lowbit(i), (i + lowbit(i)) + lowbit(i + lowbit(i)), ...` — also O(log n) of them. That gives you the O(log n) update.

## Basic Fenwick Tree

### Point Update + Prefix Sum

```python
class Fenwick:
    """
    Classic 1-indexed Fenwick tree supporting:
      update(i, v):  add v to a[i]
      prefix(i):     sum of a[1..i]
      query(l, r):   sum of a[l..r] inclusive
    Construct over an array of length n (1-indexed internally).
    """

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        """Add `delta` to a[i]. i is 1-indexed."""
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        """Return sum of a[1..i]."""
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def query(self, l, r):
        """Inclusive range sum a[l..r], 1-indexed."""
        if r < l:
            return 0
        return self.prefix(r) - self.prefix(l - 1)


# Example
bit = Fenwick(8)
for i, v in enumerate([3, 2, -1, 6, 5, 4, -3, 3], start=1):
    bit.update(i, v)
print(bit.prefix(5))     # 3 + 2 + -1 + 6 + 5 = 15
print(bit.query(3, 6))   # -1 + 6 + 5 + 4 = 14
bit.update(4, -1)        # a[4]: 6 → 5
print(bit.query(3, 6))   # -1 + 5 + 5 + 4 = 13
```

If you prefer **0-indexed** access at the API surface, just shift in the wrapper:

```python
class Fenwick0:
    def __init__(self, n):
        self._f = Fenwick(n)

    def update(self, i, delta):  self._f.update(i + 1, delta)
    def prefix(self, i):         return self._f.prefix(i + 1)  # sum a[0..i]
    def query(self, l, r):       return self._f.query(l + 1, r + 1)
```

### Range Sum Query

Falls out for free from `prefix(r) - prefix(l - 1)` as above. The subtraction trick requires that the underlying operation has an inverse (sum, XOR, addition mod p, etc.).

### Construction in O(n)

The naive way to construct a Fenwick over a starting array is `n` calls to `update(...)`, costing O(n log n). There's a slicker O(n) method that uses the parent relation `parent(i) = i + lowbit(i)`:

```python
def build_fenwick(arr):
    """
    Build a Fenwick over arr (0-indexed) in O(n).
    Returns a Fenwick instance.
    """
    n = len(arr)
    f = Fenwick(n)
    # copy initial values into tree[1..n]
    for i in range(n):
        f.tree[i + 1] = arr[i]
    # propagate each cell to its parent in one sweep
    for i in range(1, n + 1):
        parent = i + (i & -i)
        if parent <= n:
            f.tree[parent] += f.tree[i]
    return f
```

This is rarely a bottleneck, but it's nice to know — it also explains the structural relationship "every cell flows up to its parent."

## Range Update + Point Query

If you only need **range updates** and **point queries**, you can use a single Fenwick tree as a **difference array** in disguise.

Let `d[i] = a[i] - a[i-1]`. Then:
- A range add of `+v` on `[l, r]` becomes two point updates on `d`: `d[l] += v`, `d[r + 1] -= v`.
- The value at position `i` is `a[i] = d[1] + d[2] + ... + d[i] = prefix_d(i)`.

```python
class FenwickRangeAddPoint:
    """
    Range-add + point-query using BIT on a difference array.
    All indices 1-based.
    """

    def __init__(self, n):
        self.bit = Fenwick(n)
        self.n = n

    def range_add(self, l, r, v):
        self.bit.update(l, v)
        if r + 1 <= self.n:
            self.bit.update(r + 1, -v)

    def point_get(self, i):
        return self.bit.prefix(i)


# Example
f = FenwickRangeAddPoint(6)
f.range_add(2, 4, 3)     # a = [0, 3, 3, 3, 0, 0]
f.range_add(3, 6, 1)     # a = [0, 3, 4, 4, 1, 1]
print(f.point_get(2))    # 3
print(f.point_get(4))    # 4
print(f.point_get(5))    # 1
```

## Range Update + Range Query (Two-BIT Trick)

The most powerful pure-BIT pattern: **range add + range sum** in O(log n) per op using **two** Fenwick trees.

Let `d[i] = a[i] - a[i-1]` so `a[i] = Σ d[j] for j ≤ i`. Then for an inclusive prefix sum:

```
prefix_a(p) = Σ_{i=1..p} a[i]
            = Σ_{i=1..p} Σ_{j=1..i} d[j]
            = Σ_{j=1..p} d[j] * (p - j + 1)
            = (p + 1) * Σ_{j=1..p} d[j]  -  Σ_{j=1..p} j * d[j]
```

So we maintain two BITs:
- `B1[i]` over `d[i]`.
- `B2[i]` over `i * d[i]`.

A range add `+v` on `[l, r]` updates **both** BITs at two positions each: at `l` we add `+v` to `B1` and `+v * l` (with the right sign) to `B2`; at `r+1` we subtract. The exact updates are:

```
B1: +v at l,         -v at r+1
B2: +v*(l-1) at l,   -v*r at r+1
```

And `prefix(p) = (p) * B1.prefix(p) - B2.prefix(p)`.

```python
class FenwickRangeAddRangeSum:
    """
    Range-add + range-sum BIT.
    Two underlying Fenwick trees encode prefix sums of a[1..p].
    All indices 1-based.
    """

    def __init__(self, n):
        self.n = n
        self.B1 = Fenwick(n)
        self.B2 = Fenwick(n)

    def _internal_update(self, i, v):
        self.B1.update(i, v)
        self.B2.update(i, v * (i - 1))

    def range_add(self, l, r, v):
        self._internal_update(l, v)
        self._internal_update(r + 1, -v)

    def _prefix(self, p):
        return p * self.B1.prefix(p) - self.B2.prefix(p)

    def range_sum(self, l, r):
        return self._prefix(r) - self._prefix(l - 1)


# Example
f = FenwickRangeAddRangeSum(6)
f.range_add(2, 4, 3)         # a = [0, 3, 3, 3, 0, 0]
print(f.range_sum(1, 6))     # 0+3+3+3+0+0 = 9
f.range_add(3, 6, 1)         # a = [0, 3, 4, 4, 1, 1]
print(f.range_sum(1, 6))     # 13
print(f.range_sum(3, 5))     # 4 + 4 + 1 = 9
```

A range-add + range-sum BIT is **shorter and faster** than the equivalent lazy segment tree and is one of the highest-leverage data structures to memorize for contests.

## Fenwick Tree on Other Operations

### Prefix XOR

XOR is its own inverse, so the same Fenwick template works:

```python
class FenwickXor:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, v):
        while i <= self.n:
            self.tree[i] ^= v
            i += i & -i

    def prefix_xor(self, i):
        x = 0
        while i > 0:
            x ^= self.tree[i]
            i -= i & -i
        return x

    def range_xor(self, l, r):
        return self.prefix_xor(r) ^ self.prefix_xor(l - 1)
```

### Prefix Max with Restrictions

A standard Fenwick **cannot** answer arbitrary range max — max has no inverse. But it **can** answer **prefix max**: "what is the max of a[1..i]?" with point updates that **only increase**.

```python
class FenwickPrefixMax:
    """
    Prefix max queries with point updates that only increase the value.
    Cannot do range max in general; cannot decrease values.
    """

    def __init__(self, n):
        self.n = n
        self.tree = [-(1 << 60)] * (n + 1)

    def update(self, i, v):
        while i <= self.n:
            if v > self.tree[i]:
                self.tree[i] = v
            i += i & -i

    def prefix_max(self, i):
        m = -(1 << 60)
        while i > 0:
            if self.tree[i] > m:
                m = self.tree[i]
            i -= i & -i
        return m
```

This shows up in the standard O(n log n) **Longest Increasing Subsequence** with coordinate compression.

### Multiplicative Prefix

For prefix **products** modulo a prime, multiplication is invertible (modular inverse), so a Fenwick works. Skip this if any value can be 0 — division by zero kills the trick. For arbitrary moduli or non-invertible values, use [[segment_tree]].

## 2D Fenwick Tree

### 2D Point Update + Rectangle Sum

```python
class Fenwick2D:
    """
    2D BIT: point update + rectangle prefix sum.
    All indices 1-based; tree dimensions are (rows+1) x (cols+1).
    """

    def __init__(self, rows, cols):
        self.r = rows
        self.c = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, x, y, delta):
        i = x
        while i <= self.r:
            j = y
            while j <= self.c:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i

    def prefix(self, x, y):
        """Sum over rectangle (1, 1) to (x, y)."""
        s = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                s += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return s

    def rect_sum(self, x1, y1, x2, y2):
        """Sum over inclusive rectangle (x1, y1) to (x2, y2)."""
        return (
            self.prefix(x2, y2)
            - self.prefix(x1 - 1, y2)
            - self.prefix(x2, y1 - 1)
            + self.prefix(x1 - 1, y1 - 1)
        )


# Example
g = Fenwick2D(4, 5)
g.update(2, 3, 7)
g.update(3, 4, 2)
print(g.rect_sum(1, 1, 4, 5))  # 9
print(g.rect_sum(3, 3, 4, 5))  # 2
```

### 2D Range Update + Point Query

The 2D analogue of the difference-array trick: a range add over rectangle `(x1, y1, x2, y2)` becomes four point updates on a 2D BIT, and a point query is a rectangle prefix sum.

```python
class Fenwick2DRangeAddPoint:
    def __init__(self, rows, cols):
        self.bit = Fenwick2D(rows, cols)

    def range_add(self, x1, y1, x2, y2, v):
        self.bit.update(x1, y1, v)
        self.bit.update(x2 + 1, y1, -v)
        self.bit.update(x1, y2 + 1, -v)
        self.bit.update(x2 + 1, y2 + 1, v)

    def point_get(self, x, y):
        return self.bit.prefix(x, y)
```

For 2D **range update + range query** you can extend the two-BIT trick to **four** BITs — implementation is fiddly but follows the same algebra.

## Finding the kth Smallest in O(log n)

If a Fenwick tree maintains the frequency of compressed values 1..n, you can binary-lift down the tree to find the kth-smallest in O(log n) — twice as fast as `O(log² n)` with binary search over `prefix()`.

```python
def kth(bit, k):
    """
    Given a Fenwick over a frequency array of size n,
    return the smallest index i such that prefix(i) >= k. O(log n).
    """
    pos = 0
    LOG = bit.n.bit_length()
    pw = 1 << LOG
    while pw > 0:
        nxt = pos + pw
        if nxt <= bit.n and bit.tree[nxt] < k:
            pos = nxt
            k -= bit.tree[nxt]
        pw >>= 1
    return pos + 1
```

Quick proof of correctness: at each step the "is `tree[nxt] < k`?" check decides whether the kth element lies in the implicit half-tree rooted at `nxt`. If it doesn't (i.e., enough mass is in `[pos+1, nxt]`), we descend; otherwise we skip past and decrement `k`.

This pattern is the heart of "Order Statistics Tree" implementations in C++ and gives you `select(k)` in `O(log n)` without storing duplicates explicitly.

## Counting Inversions

The canonical Fenwick application. An **inversion** is a pair `(i, j)` with `i < j` and `a[i] > a[j]`. Counting inversions in O(n log n):

```python
def count_inversions(a):
    """
    Count inversions in `a` using a Fenwick over compressed ranks.
    Inversions = pairs (i, j) with i < j and a[i] > a[j].
    """
    sorted_unique = sorted(set(a))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}  # 1-indexed ranks
    m = len(sorted_unique)
    bit = Fenwick(m)
    inv = 0
    for x in reversed(a):
        r = rank[x]
        # how many elements seen so far (to the right) are strictly smaller than x?
        inv += bit.prefix(r - 1)
        bit.update(r, 1)
    return inv


print(count_inversions([2, 4, 1, 3, 5]))  # 3 (pairs: (2,1), (4,1), (4,3))
```

The same `reverse + BIT` pattern counts "for each i, how many j > i with a[j] < a[i]" — useful in many sub-problems (e.g., LeetCode 315).

## Offline Coordinate Compression Pattern

When values are huge (up to 10^9) but only `n` distinct ones appear, you don't want a Fenwick of size 10^9. The pattern:

```python
def offline_with_compression(events):
    """
    events: list of (op, *args) where op references a value.
    All values are first mapped to ranks 1..k for some k <= n.
    """
    values = sorted({v for op, *args in events for v in args if isinstance(v, int)})
    rank = {v: i + 1 for i, v in enumerate(values)}
    bit = Fenwick(len(values))
    out = []
    for op, *args in events:
        if op == 'add':
            (v,) = args
            bit.update(rank[v], 1)
        elif op == 'count_leq':
            (v,) = args
            out.append(bit.prefix(rank[v]))
    return out
```

You can also compress on the fly using `bisect`:

```python
import bisect

values = sorted({...})
def rk(x):
    return bisect.bisect_right(values, x)  # 1-indexed rank of largest value <= x
```

This is the bread-and-butter setup for inversion variants, "count of smaller elements after self," sweep-line algorithms, and segment intersection problems.

## BIT vs. Segment Tree

| Aspect | Fenwick | Segment tree |
|---|---|---|
| Memory | `n + 1` cells | `4n` (recursive) or `2n` (iterative) |
| Constant factor | Very small (one cache-friendly array, no recursion) | 2–5× larger |
| Code length | ~20 lines | ~80 lines (lazy ~150) |
| Supports any associative op | Group ops only (sum/XOR/mult mod p) | Any associative op |
| Range updates | Two-BIT trick (no lazy needed) | Lazy propagation |
| Range max/min | No (no inverse) — only prefix max if monotone | Yes |
| kth element | O(log n) via binary lift | O(log n) via descend |
| Persistent variant | Awkward | Natural ([[segment_tree]] § Persistent) |

**Heuristic**: if the problem fits a BIT, use it — shorter and faster. If you need any of {range min/max, range chmin/chmax, complex node structs}, reach for [[segment_tree]].

## Common Patterns

| Problem pattern | Recipe |
|---|---|
| "Count of elements ≤ X to the left of i" | Sweep i left-to-right, query `prefix(rank(X))`, update `rank(a[i])` |
| "Count inversions" | Reverse + BIT on compressed ranks |
| "LIS in O(n log n)" | Sweep, `prefix_max(rank(a[i]) - 1) + 1`, update |
| "Range sum + point update" | Plain BIT |
| "Range add + point query" | Difference-array BIT |
| "Range add + range sum" | Two-BIT trick |
| "Find smallest i with prefix(i) ≥ k" | Binary lift on BIT |
| "Rectangle sum + point update on a grid" | 2D BIT |
| "Rectangle add + point query on a grid" | 2D difference-array BIT (4 updates per add) |
| "Median in a sliding window" | BIT + binary lift (or two heaps) |

## Interview & Contest Problems

**LeetCode**
- 307. Range Sum Query — Mutable (basic BIT)
- 315. Count of Smaller Numbers After Self (compressed BIT + reverse sweep)
- 327. Count of Range Sum (BIT or merge sort tree)
- 406. Queue Reconstruction by Height (BIT to find kth empty slot)
- 493. Reverse Pairs (BIT on compressed values)
- 673. Number of Longest Increasing Subsequence (BIT of (length, count) pairs)
- 1395. Count Number of Teams (two BITs: smaller/greater)
- 1409. Queries on a Permutation With Key (BIT of "still present" markers)
- 2179. Count Good Triplets in an Array (BIT pair sweep)

**Competitive**
- CSES "Static Range Sum Queries" (warm-up)
- CSES "Dynamic Range Sum Queries"
- CSES "Range Update Queries"
- CSES "Forest Queries" (2D BIT)
- CSES "List Removals" (BIT + binary lift)
- Codeforces 459D "Pashmak and Parmida's problem"
- SPOJ INVCNT

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Build (naive) | O(n log n) | O(n) |
| Build (O(n) sweep) | O(n) | O(n) |
| Point update | O(log n) | — |
| Prefix query | O(log n) | — |
| Range query (sum/XOR via subtraction) | O(log n) | — |
| 2D point update | O(log² n) | O(n²) |
| 2D rectangle sum | O(log² n) | — |
| kth via binary lift | O(log n) | — |

Constant factor: extremely low. In CPython, an array-backed BIT is typically the fastest non-trivial data structure you can use; for tight contests use `array.array('q', ...)` or NumPy if applicable.

## Common Pitfalls

1. **1-indexed vs. 0-indexed.** The implementation here uses **1-indexed** access internally. Forgetting to shift indices is the #1 BIT bug. Pick a side and stick with it.
2. **`prefix(0)` must be 0.** The loop `while i > 0` handles this automatically — don't rewrite it to `while i >= 0`.
3. **Updating with `lowbit` from 0.** If `i == 0`, `i & -i == 0` and the loop never advances. Always start updates from `i >= 1`.
4. **Using a BIT for min/max in general.** Doesn't work — no inverse. Only the "prefix max with monotone updates" pattern works.
5. **Negative indices.** Range `[l, r]` with `r < l` should return 0 (the identity). Always guard with `if r < l: return 0`.
6. **Off-by-one in `range_add(r + 1, ...)`.** If `r == n`, you must skip the second update (or size the tree to `n + 1` cells).
7. **2D BIT memory.** A 2000 × 2000 BIT is fine; a 10^5 × 10^5 dense BIT is 10 GB and not. Compress one axis or sweep.
8. **Forgetting compression.** A naive BIT of size 10^9 will TLE/MLE. Always compress to `n` ranks first when values are sparse.
9. **Mixing the two-BIT trick with point updates.** If your problem has both range-add and single-point updates, treat single-point as a range-add over `[i, i]` rather than mixing two APIs that share the same underlying BIT.
10. **Overflow in C++ ports.** Python's arbitrary precision hides this, but if you port the two-BIT trick to C++, the `B2` value can overflow 64-bit for the maximum input — use `__int128` or be careful about modulus.

## Practice Problems

| Source | Problem |
|---|---|
| CSES | Range Queries section (Static Range Sum, Dynamic Range Sum, Range Update, Forest Queries 2D, List Removals) |
| LeetCode | 307, 315, 327, 406, 493, 673, 1395, 1409, 2179 |
| Codeforces | 459D, 540E, 597C, 652D, 793F, 869E |
| SPOJ | INVCNT, FENTREE, MATSUM, KQUERY |
| AtCoder | Library Checker: "Point Add Range Sum", "Range Affine Range Sum" (latter needs lazy segtree) |

## Additional Resources

- CP-Algorithms: "Fenwick Tree" article — the canonical reference.
- Peter Fenwick's 1994 paper "A new data structure for cumulative frequency tables."
- "Competitive Programmer's Handbook" by Antti Laaksonen, BIT chapter.
- TopCoder tutorial: "Binary Indexed Trees" by boba5551.
- Codeforces blog: "Fenwick trees" series — many variations and tricks.

## Where this connects

- [Segment tree](segment_tree.md) — more powerful alternative; supports arbitrary range queries with lazy propagation
- [Sparse table](sparse_table.md) — O(1) RMQ for static (immutable) data; Fenwick handles dynamic updates
- [Data structures/fenwick_tree](../data_structures/fenwick_tree.md) — the data structure reference page
