# Heavy-Light Decomposition (HLD)

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [Heavy and Light Edges](#heavy-and-light-edges)
  - [Chains](#chains)
  - [Why O(log n)?](#why-olog-n)
- [Building the Decomposition](#building-the-decomposition)
  - [Pass 1: Subtree Sizes](#pass-1-subtree-sizes)
  - [Pass 2: Chain Assignment and Linearization](#pass-2-chain-assignment-and-linearization)
- [Path Queries with a Segment Tree](#path-queries-with-a-segment-tree)
- [Subtree Queries via Euler Tour](#subtree-queries-via-euler-tour)
- [Edge Weights vs. Vertex Weights](#edge-weights-vs-vertex-weights)
- [Path Updates](#path-updates)
- [Computing LCA from HLD](#computing-lca-from-hld)
- [Common Patterns](#common-patterns)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Heavy-Light Decomposition (HLD)** is a tree-decomposition technique that lets you answer **path queries** and **subtree queries** on a tree in `O(log² n)` per operation by reducing them to range queries on a linear array. The linear array is then handled by a [[segment_tree]] (or a [[fenwick_tree]] when the aggregate supports it).

Typical queries enabled by HLD:
- **Path query**: aggregate (sum, max, min, GCD, …) of node or edge values on the path from `u` to `v`.
- **Subtree query**: aggregate over an entire subtree.
- **Path update**: add a value to every node/edge on the path from `u` to `v`.
- **Subtree update**: add a value to every node in a subtree.
- **LCA**: free byproduct of the decomposition.

HLD is the canonical tool when:
- The tree is **static** (no edge insertions/removals).
- Queries are **arbitrary path or subtree queries** (not just root-to-node).
- The aggregate operation is **associative**, and updates make sense.

For dynamic trees, look at link-cut trees instead.

Related:
- [[segment_tree]] — the data structure HLD piggybacks on.
- [[lca]] — HLD computes LCA naturally; conversely, LCA structures don't subsume HLD.
- [[fenwick_tree]] — usable instead of segment tree when the aggregate is a group operation (sum/XOR).
- [[sparse_table]] — for read-only path queries you can sometimes avoid HLD entirely.

## ELI10 Explanation

Imagine a family tree where you keep being asked: "What is the sum of allowances of every person on the chain from grandkid X to grandkid Y?"

Naively, you walk from X up to their common ancestor with Y, then back down to Y, adding allowances one by one. If the tree is tall, this is slow.

HLD's trick: at every parent, choose the child with the **biggest** family below them as the **heavy** child. Color the edge to that child **black**, and all other edges **gray**.

Two facts about this coloring:
- Black edges line up into **chains** running down the tree.
- From any leaf to the root, you only cross at most `log n` **gray** edges. Why? Crossing a gray edge means leaving your current heavy chain — but each time you do, you enter a subtree that's at least **twice as small**, so it can happen only `log n` times.

To answer "sum on path from X to Y," walk up from each toward the LCA. Each step either:
- Slides up a heavy chain by a single segment-tree range query, or
- Jumps a gray edge.

Both kinds of steps happen `O(log n)` times. With a segment-tree query at `O(log n)` per step, total time is `O(log² n)`.

## Core Concepts

### Heavy and Light Edges

For each non-leaf vertex `v`, choose the child `c` with the **largest subtree size**. The edge `(v, c)` is the **heavy edge**; all other edges from `v` are **light**.

(Ties are broken arbitrarily — the algorithm works for any consistent tie-breaker.)

### Chains

Following heavy edges from any vertex gives you the **heavy chain** containing it. Each vertex belongs to exactly one chain. The "top" of a chain is its highest vertex (closest to the root).

Define:
- `head[v]` = top of `v`'s chain.
- `parent[v]` = parent in the tree.
- `pos[v]` = position of `v` in the linearization (DFS order that visits heavy children first).

Two vertices share a chain iff `head[u] == head[v]`.

### Why O(log n)?

**Claim**: any root-to-leaf path crosses at most `⌊log₂ n⌋` light edges.

**Proof sketch**: each light edge `(v, c)` has `size(c) ≤ size(v) / 2`, because `c` was **not** chosen as the heavy child, so some sibling has `size ≥ size(c)`, and `c + sibling ≤ size(v) - 1`. So `size` at least halves with every light edge taken — at most `log₂ n` such jumps before you hit a vertex of size 1.

Each chain contributes one segment-tree query (O(log n)), so the total per-query cost is `O(chains_visited × log n) = O(log² n)`.

## Building the Decomposition

The build has two DFS passes.

### Pass 1: Subtree Sizes

```python
import sys
sys.setrecursionlimit(300_000)


def compute_sizes(n, adj, root=0):
    """
    Compute size[v] = number of nodes in v's subtree,
    parent[v], depth[v], and heavy[v] = chosen heavy child (or -1).
    Iterative DFS to avoid recursion limits.
    """
    parent = [-1] * n
    depth = [0] * n
    size = [1] * n
    heavy = [-1] * n
    order = []
    stack = [(root, -1, iter(adj[root]))]
    parent[root] = -1
    # We need a post-order traversal: simulate it with two passes
    # Phase A: discover order
    visited = [False] * n
    visited[root] = True
    while stack:
        u, par, it = stack[-1]
        try:
            v = next(it)
            if v == par:
                continue
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                stack.append((v, u, iter(adj[v])))
        except StopIteration:
            order.append(u)
            stack.pop()
    # Phase B: compute sizes in post-order and pick heavy child
    for u in order:
        max_sz = 0
        for v in adj[u]:
            if v == parent[u]:
                continue
            size[u] += size[v]
            if size[v] > max_sz:
                max_sz = size[v]
                heavy[u] = v
    return parent, depth, size, heavy
```

### Pass 2: Chain Assignment and Linearization

```python
def assign_chains(n, adj, parent, heavy, root=0):
    """
    Walk down chains, assigning each vertex to a head and a position
    in the linear array.
      head[v] = top of v's chain
      pos[v]  = position in the linearization (0..n-1)
      tour[i] = vertex at position i
    """
    head = [0] * n
    pos = [0] * n
    tour = [0] * n
    timer = 0

    # iterative DFS: visit heavy child first to keep chain contiguous
    stack = [(root, root)]
    while stack:
        v, h = stack.pop()
        head[v] = h
        pos[v] = timer
        tour[timer] = v
        timer += 1
        # push light children FIRST so heavy child is processed next
        for u in adj[v]:
            if u == parent[v] or u == heavy[v]:
                continue
            stack.append((u, u))   # u starts its own chain
        if heavy[v] != -1:
            stack.append((heavy[v], h))
    return head, pos, tour
```

The crucial line is "push heavy child **last** so it's popped **first**." That keeps each chain contiguous in `pos[]` — so a chain range corresponds to an interval `[pos[head[v]], pos[v]]`, which is exactly what a segment tree wants.

## Path Queries with a Segment Tree

We need a segment tree over the linearized array. We initialize each cell `pos[v]` to the value at vertex `v`.

```python
class HLD:
    """
    Heavy-light decomposition supporting point/range/path/subtree queries
    backed by a generic segment tree.
    """

    def __init__(self, n, adj, values, root=0):
        self.n = n
        self.adj = adj
        self.root = root
        self.parent, self.depth, self.size, self.heavy = compute_sizes(n, adj, root)
        self.head, self.pos, self.tour = assign_chains(n, adj, self.parent, self.heavy, root)
        # initialize the segment tree in the linearized order
        linearized = [values[self.tour[i]] for i in range(n)]
        self.st = SegTreeSum(linearized)

    def update_point(self, v, value):
        self.st.update(self.pos[v], value)

    def query_subtree(self, v):
        """Aggregate over v's whole subtree."""
        return self.st.query(self.pos[v], self.pos[v] + self.size[v] - 1)

    def query_path(self, u, v):
        """Aggregate over the path u..v (inclusive of both endpoints)."""
        res = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            res += self.st.query(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]
        # now u and v are on the same chain
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        res += self.st.query(self.pos[u], self.pos[v])
        return res
```

(Use `SegTreeSum` from [[segment_tree]], or any segment tree with `update(idx, val)` and `query(l, r)`.)

**Walkthrough of `query_path(u, v)`**:
1. While `u` and `v` are on different chains, jump the higher-headed one up to its chain's top, aggregating along the way.
2. When both are on the same chain, do one final segment-tree query for the in-chain interval and return.

The `parent[head[u]]` jump is the **light edge** — it's the one edge of the chain hierarchy that the inner loop relies on.

## Subtree Queries via Euler Tour

The chain assignment **also** gives a valid Euler tour (preorder where heavy child comes first). A subtree at `v` occupies exactly the range `[pos[v], pos[v] + size[v] - 1]` — see `query_subtree` above. This makes HLD a dual-purpose structure: it handles both path and subtree queries on the **same** segment tree.

## Edge Weights vs. Vertex Weights

If weights live on **edges** rather than vertices, conventionally store each edge's weight at the **lower** endpoint (the deeper of the two endpoints). Then:

- A path query `u..v` should **exclude** the LCA's stored value, because that value corresponds to the edge **above** the LCA, which is not on the path.
- A subtree query at `v` is fine as written (the edge from `parent(v)` to `v` is included by being stored at `v`, which is the root of the subtree).

Concretely, when both endpoints land on the same chain, query `[pos[lca] + 1, pos[v]]` instead of `[pos[lca], pos[v]]`. Implement this by a small adjustment in `query_path`:

```python
def query_path_edges(self, u, v):
    res = 0
    while self.head[u] != self.head[v]:
        if self.depth[self.head[u]] < self.depth[self.head[v]]:
            u, v = v, u
        res += self.st.query(self.pos[self.head[u]], self.pos[u])
        u = self.parent[self.head[u]]
    if self.depth[u] > self.depth[v]:
        u, v = v, u
    # exclude the LCA itself for edge-weighted queries
    if self.pos[u] + 1 <= self.pos[v]:
        res += self.st.query(self.pos[u] + 1, self.pos[v])
    return res
```

## Path Updates

For path **updates** (add `+d` to every node on the path `u..v`), use a lazy-propagating segment tree (see [[segment_tree]] § Lazy Propagation) and replace each `st.query(...)` call in `query_path` with `st.update(...)`. The chain decomposition is unchanged.

```python
def update_path(self, u, v, delta):
    """Add `delta` to every node on the path from u to v inclusive."""
    while self.head[u] != self.head[v]:
        if self.depth[self.head[u]] < self.depth[self.head[v]]:
            u, v = v, u
        self.st.update(self.pos[self.head[u]], self.pos[u], delta)
        u = self.parent[self.head[u]]
    if self.depth[u] > self.depth[v]:
        u, v = v, u
    self.st.update(self.pos[u], self.pos[v], delta)
```

(Here `self.st.update(l, r, delta)` is a **range** update on the segment tree — replace with the right call for whichever lazy segtree variant you're using.)

## Computing LCA from HLD

The path traversal **already** computes the LCA — it's the deeper of the two endpoints once both are on the same chain.

```python
def lca(self, u, v):
    while self.head[u] != self.head[v]:
        if self.depth[self.head[u]] < self.depth[self.head[v]]:
            u, v = v, u
        u = self.parent[self.head[u]]
    return u if self.depth[u] < self.depth[v] else v
```

This is `O(log n)` per query — competitive with binary lifting (see [[lca]]).

## Common Patterns

| Problem pattern | Recipe |
|---|---|
| Path sum / max / GCD with point updates | HLD + segment tree |
| Path sum / max with range path updates | HLD + lazy segment tree |
| Subtree sum / max with subtree updates | HLD's Euler interval + lazy segment tree |
| LCA in O(log n) | HLD chain-jumping (the loop in `lca` above) |
| Mark all nodes on a path | HLD + lazy range assign |
| Color subtrees, query path | HLD with two operations on the same segment tree |
| Path queries on edges | HLD with "store edge at lower endpoint" trick |
| Combined path + subtree queries | HLD gives both for the price of one segment tree |
| Online k-th heaviest edge on path | HLD + merge-sort tree (rare) |
| Dynamic edge weights | HLD + segment tree (edges become point updates) |

## Interview & Contest Problems

**LeetCode**: rare in interviews because problem sizes are typically small. Look at:
- 1483. Kth Ancestor of a Tree Node (HLD overkill, but works)
- 2322. Minimum Score After Removals on a Tree (subtree queries on Euler tour, related)

**Competitive**
- SPOJ QTREE, QTREE2, QTREE3, QTREE4, QTREE5 (the canonical HLD series)
- SPOJ GSS7 (max subarray + path queries on trees)
- Codeforces 343D "Water Tree" (range assign + point query on tree)
- Codeforces 487E "Tourists" (HLD with BCC for "biconnected" path min — advanced)
- Codeforces 916E "Jamie and Tree" (HLD + LCA + subtree shifts)
- AtCoder ABC 294 G (HLD weighted)
- Library Checker: "Vertex Add Path Sum," "Vertex Add Subtree Sum," "Vertex Set Path Composite"

## Complexity Analysis

| Operation | Time |
|---|---|
| Build (two DFS + segment tree init) | O(n) |
| Path query / path update | O(log² n) |
| Subtree query / subtree update | O(log n) |
| LCA via HLD | O(log n) |
| Point update | O(log n) |
| Memory | O(n) for HLD state + segment tree size |

The `log² n` per path query is rarely a bottleneck; for `n = 10^5`, that's ~280 operations per query.

If queries are read-only and you use a [[sparse_table]] for the underlying RMQ instead of a segment tree, path queries become `O(log n)` (no segment-tree log factor) — but you lose updates.

## Common Pitfalls

1. **Pushing the heavy child first.** When emitting chains in the second pass, the heavy child must come **last** on the stack so it's popped **first**. If you forget this, chains break across non-contiguous positions and the segment tree no longer corresponds to chain ranges.
2. **Edge weights at the LCA.** For edge-weighted queries, **exclude** the LCA's stored value. Off-by-one here is the most common bug in HLD path code.
3. **Heavy child of leaves.** Leaves have no children; `heavy[leaf] = -1`. Be careful not to dereference `heavy[v]` when it equals `-1`.
4. **Depth of `head[u]` vs. depth of `u`.** When deciding which endpoint to jump in the chain loop, compare `depth[head[u]]` vs. `depth[head[v]]` (not the depths of the endpoints themselves).
5. **Segment tree size.** Allocate based on `n`, not the original tree max — the linearized array has exactly `n` elements.
6. **Updating values in the original index space.** After HLD, you must operate on `pos[v]`, not `v`, when calling the segment tree.
7. **Recursion depth.** As elsewhere in tree code, prefer iterative DFS for `n ≥ 10^5`. Both passes above are iterative.
8. **Mutating tree shape.** HLD is static. Any edge insertion or deletion invalidates the entire decomposition — there's no shortcut to update HLD incrementally; rebuild from scratch.
9. **Forgetting the `head` field needs initialization.** Some implementations conflate `head[root]` with `root` itself — always set explicitly, never rely on default `-1`.
10. **Comparing chains via vertex equality.** Use `head[u] == head[v]` to test "same chain," not `u == v`.

## Practice Problems

| Source | Problem |
|---|---|
| SPOJ | QTREE, QTREE2, QTREE3, QTREE4, QTREE5, GSS7 |
| CSES | Path Queries, Path Queries II, Subtree Queries, Distinct Colors |
| Codeforces | 343D, 487E, 825G, 916E, 1023F, 1156F |
| Library Checker | Vertex Add Path Sum, Vertex Add Subtree Sum, Vertex Set Path Composite |
| AtCoder | ABC 294 G, ABC 287 G (combine with segment tree on path) |
| HackerEarth / HackerRank | "Tree Queries" series |

## Additional Resources

- CP-Algorithms: "Heavy-light decomposition" article.
- "Competitive Programming 4" by Halim & Halim — tree decomposition chapter.
- AntiForest's HLD tutorial on Codeforces.
- SPOJ QTREE editorials — a tour through how HLD is applied to varied path-query problems.
- Codeforces blog by adamant: "Heavy-light decomposition" with explicit recurrences for non-trivial aggregates.

## Where this connects

- [LCA](lca.md) — HLD builds on LCA for path decomposition; LCA is computed during HLD preprocessing
- [Segment tree](segment_tree.md) — HLD decomposes paths into chains; segment trees answer range queries on each chain
- [Trees](../data_structures/trees.md) — HLD is a tree algorithm that decomposes a tree into heavy and light chains
