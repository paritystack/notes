# Mo's Algorithm (Offline Range Queries via Sqrt Decomposition)

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use Mo's Algorithm](#when-to-use-mos-algorithm)
  - [The Sorting Trick](#the-sorting-trick)
  - [Two-Pointer Walk Cost](#two-pointer-walk-cost)
- [Basic Mo's Algorithm](#basic-mos-algorithm)
  - [Skeleton](#skeleton)
  - [Worked Example: Distinct Elements in a Range](#worked-example-distinct-elements-in-a-range)
- [Choosing the Block Size](#choosing-the-block-size)
- [Even-Odd Block Sorting Optimization](#even-odd-block-sorting-optimization)
- [Hilbert-Order Mo's](#hilbert-order-mos)
- [Mo's with Updates (Mo + Time)](#mos-with-updates-mo--time)
- [Mo on Trees](#mo-on-trees)
- [Add-Only Mo's (Rollback Mo)](#add-only-mos-rollback-mos)
- [Common Patterns](#common-patterns)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Mo's Algorithm** is an offline technique for batch-processing range queries on a static array in `O((N + Q) · √N · F)` total time, where `F` is the cost of incrementally adding or removing one element from the current window's answer.

Unlike a [[segment_tree]] or [[fenwick_tree]] — which need the aggregate to be associative and decomposable — Mo's works for **any** aggregate you can maintain with O(1) (or `O(F)`) add/remove operations on a sliding window. That makes it the tool of choice for "weird" aggregates:

- "How many **distinct** elements are in `a[l..r]`?"
- "What's the **mode** in `a[l..r]`?"
- "How many pairs `(i, j)` in `[l, r]` have `a[i] == a[j]`?"
- "Sum of `f(c)` for c = frequency of each value in `a[l..r]`?"

Where Mo's shines: when the aggregate doesn't decompose nicely (no inverse for BIT, no associativity for segtree) but **does** support O(1) updates as you grow/shrink the window by one element.

Related notes:
- [[segment_tree]] — preferred when the aggregate is associative.
- [[fenwick_tree]] — preferred when the aggregate is also invertible.
- [[sparse_table]] — preferred for static, idempotent aggregates.
- [[union_find]] — sometimes paired with Mo's for connectivity queries.

## ELI10 Explanation

Imagine a librarian who must answer many "how many distinct books are between shelves L and R?" questions. Each question takes time proportional to how many books they have to inspect.

If the librarian re-walks every range from scratch, the work is huge. But what if they walk **incrementally**? Starting from a small range, when the next question's range is "almost the same" as the current one, they only need to add the few new books and remove the few that left.

The catch: if successive questions jump around, the librarian wastes time walking back and forth. The fix is to **reorder** the questions cleverly:

- Divide the shelves into roughly `√N`-sized **blocks**.
- Sort questions by `(block of L, then R)`.

After this sort, walking through the questions in order makes both `L` and `R` move at a controlled rate. Total walking distance across all questions is `O((N + Q) √N)` — much better than rescanning each range.

The price: you can't answer questions online (you need them all in advance), and the algorithm only works when adding/removing one element from the window has a known, fast effect on the answer.

## Core Concepts

### When to Use Mo's Algorithm

Reach for Mo's when **all** of the following are true:

1. The array is **static** (no updates between queries — but see [Mo's with Updates](#mos-with-updates-mo--time) for an extension).
2. Queries can be batched and processed **offline**.
3. You can maintain the aggregate with **O(1)** or **O(log n)** `add(x)` and `remove(x)` operations applied as the window grows or shrinks by one element.
4. No simpler data structure fits — e.g., the aggregate isn't decomposable by a segment tree, or coding the segtree is prohibitively complex.

If you can write the aggregate as a `combine(a, b)` for some associative op, prefer a [[segment_tree]] — it answers each query in `O(log n)` instead of `O(√n)`.

### The Sorting Trick

Given queries `(l_i, r_i)`, sort them by `(l_i / BLOCK, r_i)` where `BLOCK = ⌈√N⌉`. After this sort:

- For queries in the same block of `l`, the `r` pointer moves **monotonically rightward**. Total cost of `r` moves across the block: `O(N)`.
- The `l` pointer moves within a block of size `BLOCK = √N`. Total cost: `O(√N)` per query.
- Total: `O(N · √N + Q · √N) = O((N + Q) √N)`.

That's why the block size `√N` is optimal — it balances the two costs.

### Two-Pointer Walk Cost

Inside each block, `r` increases monotonically across queries (so its total movement is at most `N`). Across `√N` blocks, total `r` movement is `O(N √N) = O(N · √N)`.

`l` can swing across an `√N`-wide window per query, so its total movement is `O(Q · √N)`.

Combined: `O((N + Q) √N)` *operations*. If each add/remove costs `O(F)`, the algorithm is `O((N + Q) √N · F)`.

## Basic Mo's Algorithm

### Skeleton

```python
import math

def mos_algorithm(n, queries, add, remove, current_answer):
    """
    Generic Mo's algorithm.

    n:                size of the underlying array (positions 0..n-1).
    queries:          list of (l, r) inclusive, 0-indexed.
    add(i):           extend window to include index i; updates internal state.
    remove(i):        shrink window to exclude index i; updates internal state.
    current_answer(): return the aggregate for the current window.

    Returns: list of answers in the order of the input queries.
    """
    Q = len(queries)
    block = max(1, int(math.isqrt(n)))
    # attach original index to each query, then sort
    order = sorted(
        range(Q),
        key=lambda i: (queries[i][0] // block, queries[i][1]),
    )

    answers = [None] * Q
    cur_l, cur_r = 0, -1   # window covers [cur_l..cur_r], empty when cur_r < cur_l
    for qi in order:
        l, r = queries[qi]
        # grow window outward first (avoid l>r transient if shrinking)
        while cur_r < r:
            cur_r += 1; add(cur_r)
        while cur_l > l:
            cur_l -= 1; add(cur_l)
        # then shrink
        while cur_r > r:
            remove(cur_r); cur_r -= 1
        while cur_l < l:
            remove(cur_l); cur_l += 1
        answers[qi] = current_answer()
    return answers
```

**Order of moves matters.** Always **grow outward first** then **shrink inward**, to avoid a `cur_l > cur_r` transient that would invalidate `remove(...)` if the aggregate doesn't handle empty windows gracefully.

### Worked Example: Distinct Elements in a Range

```python
def distinct_in_ranges(a, queries):
    """
    For each query (l, r), return the number of distinct values in a[l..r].
    """
    n = len(a)
    freq = [0] * (max(a) + 1)
    state = {'distinct': 0}

    def add(i):
        x = a[i]
        if freq[x] == 0:
            state['distinct'] += 1
        freq[x] += 1

    def remove(i):
        x = a[i]
        freq[x] -= 1
        if freq[x] == 0:
            state['distinct'] -= 1

    def current_answer():
        return state['distinct']

    return mos_algorithm(n, queries, add, remove, current_answer)


# Example
a = [1, 2, 1, 3, 1, 2, 4, 5]
queries = [(0, 3), (2, 5), (1, 7), (4, 7)]
print(distinct_in_ranges(a, queries))   # [3, 3, 5, 4]
```

**Why this fits Mo's perfectly**: a segment tree can't answer "count of distinct" directly — distinctness depends on the whole range, not local pieces. But the sliding-window form is trivial: one frequency counter, O(1) per add/remove.

## Choosing the Block Size

The classic block size `BLOCK = ⌈√N⌉` is optimal in the asymptotic sense, but the constant factor benefits from tuning:

- **`BLOCK = √(N · Q / 2)` style**: when `Q > N`, lowering the block size below `√N` helps. When `Q < N`, raising it helps.
- **`BLOCK = N / √(Q · 2/3)`** is a common heuristic from competitive blogs.
- For Python, with its high constant factor on Python-level loops, slightly **larger** blocks (`BLOCK = int(n / math.sqrt(q * 2/3))` or `BLOCK = 2 * isqrt(n)`) often runs faster because it reduces the number of `r`-pointer resets.

When benchmarking, try a few block sizes against the time limit and pick the one that fits.

## Even-Odd Block Sorting Optimization

A common micro-optimization: inside each block, sort by `r` **ascending** for even blocks and **descending** for odd blocks. This means `r` doesn't have to "reset" at every block boundary — it sweeps back and forth.

```python
def mos_alt_block_sort(n, queries, add, remove, current_answer):
    Q = len(queries)
    block = max(1, int(math.isqrt(n)))

    def key(i):
        l, r = queries[i]
        b = l // block
        return (b, r if b & 1 == 0 else -r)

    order = sorted(range(Q), key=key)
    # ... rest identical to mos_algorithm
```

Empirically, this saves ~20–40% on tight Mo's problems.

## Hilbert-Order Mo's

For optimal constant factor, sort queries along a **Hilbert curve** on the 2D plane of `(l, r)` coordinates. The Hilbert ordering minimizes total Manhattan movement across all queries, beating block-based sorting in practice by ~30%.

```python
def hilbert_order(x, y, pow2, rotate=0):
    """
    Hilbert curve index for point (x, y) on a 2^pow2 by 2^pow2 grid.
    """
    if pow2 == 0:
        return 0
    half = 1 << (pow2 - 1)
    seg = (1 if x >= half else 0, 1 if y >= half else 0)
    quad_index = [(0, 0), (0, 1), (1, 1), (1, 0)].index(seg)
    sub_x = x & (half - 1)
    sub_y = y & (half - 1)
    # rotate sub-quadrant
    if quad_index == 0:
        sub_x, sub_y = sub_y, sub_x
    elif quad_index == 3:
        sub_x, sub_y = half - 1 - sub_y, half - 1 - sub_x
    return (quad_index << (2 * (pow2 - 1))) + hilbert_order(sub_x, sub_y, pow2 - 1)
```

In practice, the simpler even-odd block-sort optimization is "good enough" for most contest problems; Hilbert is reserved for extreme cases.

## Mo's with Updates (Mo + Time)

If there are **point updates** mixed in with queries, classical Mo's doesn't apply. The extension **Mo + Time** treats `(l, r, t)` as 3D queries, where `t` is the "timestamp" (number of updates that have happened so far). Sort by `(l // B, r // B, t)` and walk three pointers.

Total cost: `O((N + Q) · N^{2/3})` with block size `N^{2/3}`.

```python
def mos_with_updates(n, ops):
    """
    ops: list of operations in chronological order, each either
         ('Q', l, r) or ('U', idx, new_value).
    Returns answers in query-input order.
    """
    queries = []
    updates = []
    for op in ops:
        if op[0] == 'Q':
            queries.append((op[1], op[2], len(updates)))   # (l, r, time)
        else:
            updates.append((op[1], op[2]))                # (idx, new_value)

    block = max(1, round(n ** (2 / 3)))
    Q = len(queries)
    order = sorted(
        range(Q),
        key=lambda i: (queries[i][0] // block,
                       queries[i][1] // block,
                       queries[i][2]),
    )

    # ... maintain cur_l, cur_r, cur_t and walk three pointers,
    # applying/un-applying updates as cur_t changes.
    # When an update (idx, new_val) is applied:
    #   if idx in current window: remove old value, add new value.
    #   swap (a[idx], new_val) so unapplying restores.
    # Skeleton omitted for brevity; mirrors the basic Mo's with one extra
    # pointer.
```

The total work is `O((N + Q) N^{2/3})`. For `N = 10^5`, that's ~2×10^8 — borderline in Python, comfortable in C++.

## Mo on Trees

For path queries on a **tree** (e.g., "distinct colors on the path from `u` to `v`"), you can apply Mo's to a special linearization of the tree:

1. Compute the **Euler tour**, but record **two** positions per node: `tin[v]` (entry) and `tout[v]` (exit).
2. For a query `(u, v)`:
   - If `u` is an ancestor of `v` (or vice versa), use the range `[tin[ancestor], tin[descendant]]`. The path's nodes appear an odd number of times in this range; nodes appearing an **even** number of times are off-path.
   - Otherwise, use `[tout[u], tin[v]]` (with `tin[u] ≤ tin[v]`) and **extra-include the LCA** at the end.
3. Walk the resulting 1D array with Mo's, tracking which nodes are "active" (toggled in/out on each visit). The aggregate counts only active nodes.

This is one of the more delicate Mo's variants — make sure to handle the LCA case explicitly. See SPOJ COT2 and Codeforces 375D for canonical problems.

## Add-Only Mo's (Rollback Mo's)

Some aggregates only have a fast `add(...)` and **no** efficient `remove(...)` — e.g., maintaining "max subarray sum in current window" or DSU-with-rollback connectivity queries.

**Rollback Mo's** restricts movement: within each block of `l`, only `r` increases (no shrinks). Each query is processed from scratch on its `l` boundary, then `r` extends to the query's right endpoint, then the partial state is **rolled back** to the boundary using a stack of "undo" operations. Total cost: still `O((N + Q) √N)`.

Pairs well with [[union_find]] **with rollback** (no path compression, union by rank/size with a stack of undo records).

```
For each block B:
    reset state to "empty window"
    let r = (B * BLOCK) - 1
    for each query (l, r_q) with l // BLOCK == B in increasing r_q:
        # extend r outward
        while r < r_q: r += 1; add(r)
        # temporary left extension with rollback
        snapshot = state_snapshot()
        for i in range(l, B * BLOCK): add(i)
        answers[qi] = current_answer()
        rollback(snapshot)
```

## Common Patterns

| Problem pattern | Tool |
|---|---|
| Count distinct in range | Mo's + frequency array |
| Mode (most frequent value) in range | Mo's + frequency + bucket of frequencies |
| Range mex (smallest missing) | Mo's + frequency + small-to-large |
| Sum of (frequency)² in range | Mo's + maintain sum on add/remove |
| Number of pairs (i, j) with a[i] == a[j] | Mo's + frequency: delta on add = `cur_freq[x]` |
| Distinct colors on a tree path | Mo on trees |
| Range queries with point updates (aggregate non-decomposable) | Mo + Time |
| Max subarray sum / connectivity queries | Rollback Mo + DSU/with rollback |
| "How many subarrays of [l,r] have property P?" | Often Mo's-friendly |

## Interview & Contest Problems

**LeetCode**: rare in interviews because problem sizes/patterns favor segment trees or BITs. The closest are offline + sliding-window flavored problems.

**Competitive**
- SPOJ DQUERY (distinct in range) — the canonical introductory problem.
- SPOJ FREQ2 (frequency-based queries)
- SPOJ COT2 (distinct colors on path — Mo on trees)
- SPOJ POWERFUL (sum of `f(c)²` in range)
- Codeforces 86D "Powerful array" (Mo's + maintenance of `sum(c * c * x)`)
- Codeforces 220B "Little Elephant and Array" (Mo's + frequency)
- Codeforces 940F "Machine Learning" (Mo + Time + mex over frequencies)
- Codeforces 617E "XOR and Favorite Number" (Mo's + prefix-XOR + count pairs)
- Codeforces 375D "Tree and Queries" (Mo on trees)
- Codeforces 444C "Distinct paths" (variant of Mo's on graphs)
- AtCoder ABC 174 F (distinct in range — Mo's or offline BIT both work)
- AtCoder ABC 242 G (count adjacent equal pairs — Mo's)
- AtCoder ABC 293 G (count `a[i] == a[j] == a[k]` triples in range — Mo's with combinatorial maintenance)
- Library Checker: "Static Range Mode Query"

## Complexity Analysis

| Variant | Time |
|---|---|
| Classical Mo's | `O((N + Q) √N · F)` |
| Mo + Time (with updates) | `O((N + Q) N^{2/3} · F)` |
| Hilbert-order Mo's | same asymptotic, ~30% smaller constant |
| Rollback Mo's | `O((N + Q) √N · (F + R))` where R = rollback cost per op |

`F` is the cost of one `add` or `remove`. Common values: O(1) for frequency counters, O(log n) if you maintain an order statistic, O(α(n)) for DSU with rollback.

For `N = Q = 10^5` and O(1) updates: ~3×10^7 ops — fine in C++ (under 1s), borderline in Python (likely 5–10 s with simple structures). Often you trade simplicity for a segment tree once Python is involved.

## Common Pitfalls

1. **Move order matters.** Always grow window outward (extend `r`, decrement `l`) **before** shrinking (increment `l`, decrement `r`). Reversing the order can create a transient `l > r` state and corrupt counters.
2. **`add` and `remove` must mirror exactly.** Any asymmetry — even a stray `+= 0` — accumulates errors across thousands of moves.
3. **Block-size selection.** For Python, default to `int(math.isqrt(n))`; tune if needed. For C++, classical `√N` is fine.
4. **Empty windows.** Always reset `cur_l = 0, cur_r = -1` initially. After any query, `r >= l` should hold (the algorithm guarantees this if you grow first).
5. **Mutating queries during sort.** Always sort the **indices** of queries, not the queries themselves — otherwise you can't put answers back in the original order.
6. **Frequency overflow.** For large value ranges, allocate `freq` based on the **maximum** value or compress coordinates first.
7. **Mo on trees: ancestor check.** Forgetting to handle the LCA case separately produces silently wrong answers for `u` ≠ `v` and `u` not an ancestor of `v`.
8. **Mo + Time: pointer-walk direction.** When `cur_t > query_t`, you need to **un-apply** updates (swap back), not apply new ones. Symmetric bug to the move-order issue.
9. **Rollback Mo with non-rollback DSU.** Path compression breaks rollback. Use **union by rank/size only** plus an undo stack.
10. **Python performance.** Pure-Python Mo's at `N = 10^5` can TLE. Use `array.array('i', ...)` for `freq`, avoid attribute lookups inside the hot loop, and consider PyPy if available.

## Practice Problems

| Source | Problem |
|---|---|
| SPOJ | DQUERY, FREQ2, COT2, POWERFUL, GIVEAWAY |
| Codeforces | 86D, 220B, 372D, 375D, 617E, 813F, 86D, 940F, 1093F |
| AtCoder | ABC 174 F, ABC 242 G, ABC 293 G |
| Library Checker | Static Range Mode Query |
| CSES | Range Queries section (some fit Mo's, some don't) |

## Additional Resources

- CP-Algorithms: "Mo's algorithm" article (covers classical and Mo on trees).
- TopCoder tutorial: "An alternative sorting order for Mo's algorithm" (the Hilbert order).
- Anudeep Nekkanti's blog: "MO'S ALGORITHM" (the introductory tutorial that popularized it for competitive programming).
- "Algorithms on Trees and Graphs" — Mo on trees chapter.
- Codeforces blog: "Mo's algorithm on trees" by adamant.
