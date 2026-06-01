# Sparse Table

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use a Sparse Table](#when-to-use-a-sparse-table)
  - [Idempotent vs. Non-Idempotent Operations](#idempotent-vs-non-idempotent-operations)
  - [The 2^k Decomposition](#the-2k-decomposition)
- [Basic Sparse Table (Range Min)](#basic-sparse-table-range-min)
- [Generic Sparse Table](#generic-sparse-table)
- [Range Max, GCD, Bitwise AND/OR](#range-max-gcd-bitwise-andor)
- [Non-Idempotent Operations: Disjoint Sparse Table](#non-idempotent-operations-disjoint-sparse-table)
- [Sparse Table for LCA (Euler Tour + RMQ)](#sparse-table-for-lca-euler-tour--rmq)
- [2D Sparse Table](#2d-sparse-table)
- [Comparison with Other Structures](#comparison-with-other-structures)
- [Common Patterns](#common-patterns)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

A **sparse table** is a static data structure that pre-processes an array in **O(n log n)** time and answers **range queries in O(1)** for any **idempotent** binary operation: min, max, GCD, bitwise AND, bitwise OR.

The price is that it's **static** — no updates after construction. If you need updates, use [[segment_tree]] or [[fenwick_tree]] instead.

The O(1) query speed makes the sparse table the gold standard for:
- **Range Minimum Query (RMQ)** on a static array.
- The **Euler-tour + RMQ** approach to **[[lca]]** that delivers O(1) LCA queries after O(n log n) preprocessing.
- Inner loops where even O(log n) per query is too slow.

Related:
- [[fenwick_tree]] — updates but only group operations; O(log n) per op.
- [[segment_tree]] — updates and any associative op; O(log n) per op.
- [[lca]] — uses sparse table internally for O(1) LCA.
- [[mo_algorithm]] — offline alternative when nothing else fits.

## ELI10 Explanation

Imagine you have a long bookshelf and frequently get asked "what's the **shortest** book between positions 14 and 37?"

You could check every book in the range — slow on a long shelf. Or you could pre-compute the shortest book in **every power-of-two-length window** that fits on the shelf. So you'd know:

- the shortest book in every window of length 1 (trivial — each book itself),
- the shortest book in every window of length 2,
- length 4, length 8, length 16, ..., up to the longest window that fits.

Now for any query "shortest from i to j" (length `L = j - i + 1`), pick the largest power of two `k = 2^⌊log₂ L⌋ ≤ L`. The range `[i, j]` is **fully covered** by two overlapping windows of length `k`: one starting at `i` and one ending at `j`. The shortest book in the range is the minimum of the two pre-computed values.

Because **min is idempotent** (min(x, x) = x), overlap doesn't double-count. Two table lookups → answer in O(1).

For sum, this overlap trick **doesn't** work directly (the overlapped part gets counted twice). For sum, use a [[fenwick_tree]] or prefix array.

## Core Concepts

### When to Use a Sparse Table

Use a sparse table when:

1. The array is **static** — no updates after build.
2. The query is a **range aggregate**.
3. The aggregate is **idempotent**: `f(x, x) = x`. Min, max, GCD, bitwise AND, bitwise OR all qualify.
4. You can afford `O(n log n)` build time and `O(n log n)` memory.
5. You need **O(1)** query (or very low constant factor).

If the aggregate is **not idempotent** (sum, product, XOR), you can use a **disjoint sparse table** (O(1) query, O(n log n) build) or fall back to prefix sums (`O(1)` query, `O(n)` build, but only sum-style decompositions).

### Idempotent vs. Non-Idempotent Operations

| Operation | Idempotent? | Sparse table OK? |
|---|---|---|
| min | yes (`min(x, x) = x`) | yes |
| max | yes | yes |
| GCD | yes (`gcd(x, x) = x`) | yes |
| bitwise AND | yes | yes |
| bitwise OR | yes | yes |
| sum | no (`x + x ≠ x`) | no — but **disjoint sparse table** works |
| XOR | no | no — but disjoint sparse table works |
| product | no | no — but disjoint sparse table works |
| matrix product | no (also non-commutative) | only disjoint variant |

### The 2^k Decomposition

For any range length `L ≥ 1`, let `k = ⌊log₂ L⌋`. Then `2^k ≤ L < 2^(k+1)`, and **two windows of length 2^k** suffice to cover `[i, j]`:

- one starting at `i` (covering `[i, i + 2^k - 1]`),
- one ending at `j` (covering `[j - 2^k + 1, j]`).

For idempotent operations, the overlap doesn't matter. For non-idempotent, we need a different decomposition (see [Non-Idempotent Operations](#non-idempotent-operations-disjoint-sparse-table)).

## Basic Sparse Table (Range Min)

```python
import math

class SparseTableMin:
    """
    Sparse table for range minimum queries on a static array.
    Build: O(n log n) time and memory.
    Query: O(1).
    """

    def __init__(self, data):
        n = len(data)
        self.n = n
        K = max(1, n.bit_length())   # need entries for k = 0..⌊log2 n⌋
        self.K = K
        # table[k][i] = min of data[i..i + 2^k - 1]
        self.table = [data[:]]
        for k in range(1, K):
            length = 1 << k
            if length > n:
                break
            prev = self.table[k - 1]
            row = [0] * (n - length + 1)
            half = 1 << (k - 1)
            for i in range(n - length + 1):
                row[i] = min(prev[i], prev[i + half])
            self.table.append(row)

        # Precompute floor(log2) for each length 1..n (avoids math.log in hot loop)
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1

    def query(self, l, r):
        """Inclusive min over data[l..r]."""
        length = r - l + 1
        k = self.log[length]
        return min(self.table[k][l], self.table[k][r - (1 << k) + 1])


# Example
data = [7, 2, 3, 0, 5, 10, 3, 12, 18]
st = SparseTableMin(data)
print(st.query(0, 4))  # min(7,2,3,0,5) = 0
print(st.query(3, 7))  # min(0,5,10,3,12) = 0
print(st.query(6, 8))  # min(3,12,18) = 3
```

Memory: `O(n log n)` because `log₂ n` rows each up to size `n`.

The **precomputed log table** matters: calling `math.log2` repeatedly in tight loops is ~10× slower than an integer lookup, and `n.bit_length() - 1` works but is also slower per call than the precomputed array.

## Generic Sparse Table

Parameterize the combine function so you can reuse the same template for min, max, GCD, AND, OR:

```python
class SparseTable:
    """
    Sparse table over an idempotent associative op.
    """

    def __init__(self, data, combine):
        n = len(data)
        self.n = n
        self.combine = combine
        K = max(1, n.bit_length())
        self.table = [data[:]]
        for k in range(1, K):
            length = 1 << k
            if length > n:
                break
            prev = self.table[k - 1]
            half = 1 << (k - 1)
            row = [combine(prev[i], prev[i + half]) for i in range(n - length + 1)]
            self.table.append(row)
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1

    def query(self, l, r):
        k = self.log[r - l + 1]
        return self.combine(self.table[k][l], self.table[k][r - (1 << k) + 1])


# Usage
from math import gcd
st_max = SparseTable([3, 1, 4, 1, 5, 9, 2, 6], combine=max)
st_gcd = SparseTable([12, 18, 24, 30, 6, 60], combine=gcd)
print(st_max.query(2, 5))  # max(4,1,5,9) = 9
print(st_gcd.query(0, 4))  # gcd(12,18,24,30,6) = 6
```

## Range Max, GCD, Bitwise AND/OR

All four operations are idempotent and supported directly via the generic template.

```python
# Range max
st = SparseTable([3, 1, 4, 1, 5, 9, 2, 6], combine=max)

# Range GCD
st = SparseTable([12, 18, 24, 30, 6, 60], combine=gcd)

# Range bitwise AND
st = SparseTable([0b1110, 0b1101, 0b1011, 0b0111], combine=lambda a, b: a & b)

# Range bitwise OR
st = SparseTable([0b0001, 0b0010, 0b0100], combine=lambda a, b: a | b)
```

A common subproblem: "find the **longest** subarray whose GCD equals `g`." Sparse table + two-pointer or sparse table + binary search both give O(n log² n) or O(n log n) solutions.

## Non-Idempotent Operations: Disjoint Sparse Table

For non-idempotent operations like sum, XOR, product, or matrix multiplication, the overlap trick fails. The **disjoint sparse table** (also called **sparse table on disjoint intervals**) works around this:

For each level k, partition the array into blocks of size `2^k`. For each query position `m` in a block, pre-compute:
- The prefix aggregate from `m` to the right end of its block.
- The suffix aggregate from `m` to the left end of its block.

A query `[l, r]` finds the highest bit at which `l` and `r` differ (i.e., they sit in different blocks at level k). The answer is `suffix[l] · prefix[r]`.

```python
class DisjointSparseTable:
    """
    Disjoint sparse table for any associative operation (need NOT be idempotent).
    Build: O(n log n). Query: O(1).
    """

    def __init__(self, data, combine, identity):
        n = len(data)
        self.n = n
        self.combine = combine
        self.identity = identity
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1
        K = max(1, n.bit_length())
        # table[k] is a list of length n
        # for each block of size 2^k, table[k][i] holds:
        #   for i in left half of block:  combine of data[i..block_mid]
        #   for i in right half:           combine of data[block_mid+1..i]
        self.table = [data[:]]
        for k in range(1, K):
            block = 1 << k
            row = [identity] * n
            for mid in range(block - 1, n, 2 * block):
                # left half: positions [mid - block + 1 .. mid]
                acc = identity
                for i in range(mid, mid - block, -1):
                    if i < 0:
                        break
                    acc = combine(data[i], acc)
                    row[i] = acc
                # right half: positions [mid + 1 .. mid + block]
                acc = identity
                for i in range(mid + 1, min(n, mid + 1 + block)):
                    acc = combine(acc, data[i])
                    row[i] = acc
            self.table.append(row)

    def query(self, l, r):
        if l == r:
            return self.table[0][l]
        k = self.log[l ^ r]  # highest bit where l and r differ
        return self.combine(self.table[k][l], self.table[k][r])


# Example: range sum (non-idempotent)
dst = DisjointSparseTable([1, 3, 5, 7, 9, 11, 13, 15], combine=lambda a, b: a + b, identity=0)
print(dst.query(1, 4))  # 3 + 5 + 7 + 9 = 24
print(dst.query(0, 7))  # 64
```

For pure prefix sums, a plain prefix-sum array is simpler, faster, and uses O(n) memory. The disjoint sparse table earns its keep when the operation is associative but **not** decomposable by subtraction — e.g., **matrix product** (non-commutative, non-invertible in general) or **range maximum subarray sum** with struct nodes.

## Sparse Table for LCA (Euler Tour + RMQ)

A landmark application: combining sparse tables with an **Euler tour** of a tree gives an O(n log n) preprocessing and **O(1) per query** LCA — the fastest practical LCA in most contest contexts. Full treatment in [[lca]], but the sketch:

1. Run a DFS, recording the **Euler tour** (sequence of node visits) and the **depth** of each tour entry.
2. Record the **first occurrence** of each node in the tour.
3. Build a sparse table over the **depth array**, with the combine function that returns the **index of the smaller depth** (a tie-broken argmin):

```python
def lca_combine(a, b):
    # a, b are (depth, tour_index); return the entry with smaller depth
    return a if a[0] <= b[0] else b
```

4. For `lca(u, v)`, look up `first[u]` and `first[v]`, query the sparse table over that range, and return the node at that tour index.

```python
def lca(u, v):
    lo = min(first[u], first[v])
    hi = max(first[u], first[v])
    _, tour_idx = sparse_table.query(lo, hi)
    return euler_tour[tour_idx]
```

## 2D Sparse Table

For range-min queries on a static 2D grid, build a sparse table of sparse tables:

```python
class SparseTable2DMin:
    """
    O(1) range-min queries on a 2D static grid.
    Memory: O(R C log R log C).
    """

    def __init__(self, grid):
        self.R = len(grid)
        self.C = len(grid[0]) if self.R else 0
        KR = max(1, self.R.bit_length())
        KC = max(1, self.C.bit_length())
        self.KR = KR
        self.KC = KC
        # table[kr][kc][i][j] = min of subgrid (i..i+2^kr-1, j..j+2^kc-1)
        self.table = [[None] * KC for _ in range(KR)]
        # base level
        self.table[0][0] = [row[:] for row in grid]
        for kc in range(1, KC):
            width = 1 << kc
            if width > self.C:
                break
            prev = self.table[0][kc - 1]
            half = 1 << (kc - 1)
            new = [[min(prev[i][j], prev[i][j + half])
                    for j in range(self.C - width + 1)]
                   for i in range(self.R)]
            self.table[0][kc] = new
        for kr in range(1, KR):
            height = 1 << kr
            if height > self.R:
                break
            for kc in range(KC):
                width = 1 << kc
                if width > self.C:
                    break
                prev = self.table[kr - 1][kc]
                half = 1 << (kr - 1)
                new = [[min(prev[i][j], prev[i + half][j])
                        for j in range(len(prev[0]))]
                       for i in range(self.R - height + 1)]
                self.table[kr][kc] = new
        self.log = [0] * (max(self.R, self.C) + 1)
        for i in range(2, len(self.log)):
            self.log[i] = self.log[i // 2] + 1

    def query(self, r1, c1, r2, c2):
        kr = self.log[r2 - r1 + 1]
        kc = self.log[c2 - c1 + 1]
        a = self.table[kr][kc][r1][c1]
        b = self.table[kr][kc][r1][c2 - (1 << kc) + 1]
        c = self.table[kr][kc][r2 - (1 << kr) + 1][c1]
        d = self.table[kr][kc][r2 - (1 << kr) + 1][c2 - (1 << kc) + 1]
        return min(a, b, c, d)
```

The four-corner combine gives O(1) per query; memory grows fast, so 2D sparse tables are practical for grids up to a few thousand on a side.

## Comparison with Other Structures

| Aspect | Prefix sum | Sparse table | Fenwick | Segment tree |
|---|---|---|---|---|
| Build | O(n) | O(n log n) | O(n) | O(n) |
| Query | O(1) | O(1) | O(log n) | O(log n) |
| Update | O(n) (rebuild) | O(n log n) (rebuild) | O(log n) | O(log n) |
| Supports min/max/GCD | sum only | yes | sum only | yes |
| Supports range updates | no | no | with tricks | yes (lazy) |
| Memory | O(n) | O(n log n) | O(n) | O(n) |

If the array is static and the op is idempotent, **sparse table is unbeatable** for query throughput. As soon as updates appear, switch to BIT or segment tree.

## Common Patterns

| Problem pattern | Tool |
|---|---|
| Static range min/max | Sparse table |
| Static range GCD | Sparse table |
| Static range AND/OR | Sparse table |
| Static range sum (cheap) | Prefix sum |
| Static range sum + arbitrary op (matrix etc.) | Disjoint sparse table |
| Static LCA preprocessing | Sparse table over Euler tour |
| "Longest subarray with GCD ≥ g" | Sparse table + two-pointer / binary search |
| "Max in sliding window" (offline) | Sparse table (or monotonic queue for online) |
| 2D range min on small grids | 2D sparse table |

## Interview & Contest Problems

**LeetCode**
- 1043. Partition Array for Maximum Sum (range max via sparse table simplifies DP)
- 1521. Find a Value of a Mysterious Function Closest to Target (range bitwise AND on static array)
- 1851. Minimum Interval to Include Each Query (offline, sparse table on sorted intervals)
- 2334. Subarray With Elements Greater Than Varying Threshold (range min)

**Competitive**
- SPOJ RMQSQ — vanilla RMQ
- Codeforces 514D "R2D2 and Droid Army" (range bitwise AND)
- Codeforces 6E "Exposition" (sparse table + two pointers)
- AtCoder ABC 282 E (offline + sparse min on values)
- CSES "Static Range Minimum Queries"
- CSES "Range Update Queries" (after preprocessing, queries on static values)

**For LCA-via-sparse-table**: any LCA-flavored problem on trees with up to 10^6 nodes — see [[lca]].

## Complexity Analysis

| Operation | Sparse | Disjoint sparse | 2D sparse |
|---|---|---|---|
| Build | O(n log n) | O(n log n) | O(R C log R log C) |
| Query | O(1) | O(1) | O(1) |
| Memory | O(n log n) | O(n log n) | O(R C log R log C) |
| Updates | not supported | not supported | not supported |

For `n = 10^6`, the build is ~20M cells — fine in C++, tight in Python. For `n = 10^7+`, use a Fenwick or segment tree to save the log factor of memory.

## Common Pitfalls

1. **Forgetting the log table.** Calling `math.log2` or `bit_length` inside the query is slow enough to fail tight time limits. Always precompute `log[1..n]` once.
2. **Off-by-one on query bounds.** Query is inclusive `[l, r]`; the second lookup is at `r - (1 << k) + 1`, not `r - (1 << k)`. Drawing this out on paper once cures the bug forever.
3. **Using sparse table for sum.** It will produce wrong answers because `x + x ≠ x`. Use prefix sums or a disjoint sparse table.
4. **Updates.** Sparse table is **static**. Rebuilding on every update is O(n log n) and almost always the wrong choice — switch structures.
5. **Memory blowup on 2D.** A naive 2D sparse table can be massive. Estimate `R * C * log R * log C` before allocating.
6. **Identity element confusion** in the generic template — pass the right identity (∞ for min, 0 for GCD/XOR, 0 for sum, etc.).
7. **Forgetting empty ranges.** Decide upfront whether you support `l > r`; if not, assert it.
8. **Treating GCD as decomposable by subtraction.** GCD is idempotent (so the overlap trick works) but **not** invertible (so you can't use the `prefix(r) - prefix(l-1)` trick that BIT uses).

## Practice Problems

| Source | Problem |
|---|---|
| CSES | Static Range Minimum Queries, Range Update Queries (variant), Static Range Sum Queries (use prefix sum, not sparse) |
| SPOJ | RMQSQ, MKTHNUM (combo) |
| Codeforces | 514D, 5C (with two pointers), 6E, 1359D |
| Library Checker | "Static Range Min Query" (use sparse table) |
| AtCoder | ABC 282 E (offline sparse min), ABC 290 E (sparse + sweep) |
| LeetCode | 1851, 2334, 1521 |

## Additional Resources

- CP-Algorithms: "Sparse Table" article (also covers disjoint sparse table).
- Bender & Farach-Colton's original paper on Euler-tour LCA via sparse RMQ.
- "Competitive Programmer's Handbook" by Antti Laaksonen — sparse table chapter.
- Codeforces blog: "Sparse Table" by Errichto.
- For the O(1) build + O(1) query Cartesian-tree-based RMQ, see "The LCA Problem Revisited" (Bender & Farach-Colton, 2000) — interesting reading but rarely needed in practice; the O(n log n) sparse table is faster in real life.

## Where this connects

- [Segment tree](segment_tree.md) — mutable alternative; O(log n) queries/updates
- [Fenwick tree](fenwick_tree.md) — dynamic prefix sums when data changes
- [LCA](lca.md) — sparse table enables O(1) LCA queries via the equivalence between RMQ and LCA
- [Data structures/sparse_table](../data_structures/sparse_table.md) — the data structure reference page
