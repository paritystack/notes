# Lowest Common Ancestor (LCA)

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [Tree Conventions](#tree-conventions)
  - [Approach Comparison](#approach-comparison)
- [Binary Lifting](#binary-lifting)
  - [Build](#build)
  - [Query](#query)
  - [kth Ancestor](#kth-ancestor)
- [Euler Tour + RMQ (Sparse Table)](#euler-tour--rmq-sparse-table)
- [Tarjan's Offline LCA (Union-Find)](#tarjans-offline-lca-union-find)
- [Heavy-Path Decomposition for LCA](#heavy-path-decomposition-for-lca)
- [Distance on a Tree](#distance-on-a-tree)
- [Auxiliary Tree / Virtual Tree](#auxiliary-tree--virtual-tree)
- [LCA in Dynamic Trees](#lca-in-dynamic-trees)
- [Common Patterns and Applications](#common-patterns-and-applications)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Comparison](#complexity-comparison)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

The **Lowest Common Ancestor (LCA)** of two nodes `u` and `v` in a rooted tree is the deepest node that is an ancestor of both. LCA is the foundation of countless tree problems:

- **Distance on a tree**: `dist(u, v) = depth(u) + depth(v) - 2 * depth(lca(u, v))`.
- **Path queries**: combine prefix-from-root information at `u`, `v`, and `lca`.
- **k-th ancestor** (closely related): for jump games, [[heavy_light_decomposition]] queries.
- **Auxiliary tree / virtual tree**: rebuild a smaller tree on a subset of "interesting" nodes.

This note covers the three most useful LCA algorithms:

| Method | Preprocess | Query | Notes |
|---|---|---|---|
| **Binary lifting** | O(n log n) | O(log n) | Most popular; also gives kth ancestor for free |
| **Euler tour + sparse table** | O(n log n) | O(1) | Fastest queries; uses [[sparse_table]] internally |
| **Tarjan's offline** | O((n + q) α(n)) | amortized α(n) per query | Smallest constant; needs all queries up front |

Related:
- [[sparse_table]] — backbone of the Euler-tour LCA.
- [[heavy_light_decomposition]] — HLD also gives LCA in O(log n) as a side effect.
- [[union_find]] — Tarjan's offline LCA uses DSU.

## ELI10 Explanation

Think of a family tree where you mark one person as the root (the oldest known ancestor). For any two cousins, the LCA is the **most recent shared ancestor** — the lowest in the tree that they both descend from.

If you stood at one cousin and walked **up** toward the root, recording every ancestor, the LCA is the **first** ancestor you'd find that's also an ancestor of the other cousin.

To make this fast, we pre-compute "jumps of length 1, 2, 4, 8, ..." from each node toward the root. Then for any two cousins:

1. **Lift the deeper one** up to the same depth as the shallower one (using binary jumps).
2. If they're now the same node, that's the LCA.
3. Otherwise, **lift both** simultaneously by the largest jump that keeps them different. Repeat with smaller jumps.
4. After all jumps, both are at depth-1 below the LCA. Their parent is the LCA.

That's binary lifting in a nutshell — `O(log n)` time per query, using only `O(n log n)` preprocessing.

## Core Concepts

### Tree Conventions

Throughout this note:

- `n` is the number of nodes.
- The tree is **rooted** at node 0 (or 1 if you prefer; sample code uses 0).
- Edges are given as an adjacency list.
- `parent[v]` is the immediate parent; `parent[root] = -1` (or `root` itself in some conventions).
- `depth[v]` is the distance from `root` to `v` in edges.

```python
def build_tree(n, edges, root=0):
    """
    edges: list of (u, v) undirected.
    Returns adjacency list `adj`, parent[], depth[], BFS/DFS order.
    """
    from collections import deque
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [-1] * n
    depth = [0] * n
    order = []
    visited = [False] * n
    q = deque([root])
    visited[root] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)
    return adj, parent, depth, order
```

### Approach Comparison

| Need | Pick |
|---|---|
| Online queries, simple implementation | Binary lifting |
| Online queries, query time matters more than build | Euler tour + sparse table |
| All queries known up front, very tight memory | Tarjan's offline |
| Already doing HLD anyway | HLD gives LCA for free |

Binary lifting is the default in most competitive code because it doubles as a `kth_ancestor` primitive.

## Binary Lifting

The idea: for each node `v` and each power of two `k`, pre-compute `up[v][k] = 2^k`-th ancestor of `v`. Then any ancestor jump of depth `d` decomposes into the binary representation of `d` and takes `popcount(d) ≤ log n` table lookups.

### Build

```python
import sys
sys.setrecursionlimit(300_000)

class BinaryLiftingLCA:
    """
    Binary-lifting LCA.
    Build: O(n log n) time and memory.
    Query: O(log n).
    Supports kth_ancestor(v, k) for free.
    """

    def __init__(self, n, adj, root=0):
        self.n = n
        self.LOG = max(1, (n - 1).bit_length())
        self.up = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n
        self._build(adj, root)
        for k in range(1, self.LOG):
            for v in range(n):
                mid = self.up[k - 1][v]
                self.up[k][v] = self.up[k - 1][mid] if mid != -1 else -1

    def _build(self, adj, root):
        # iterative BFS to populate up[0] and depth
        from collections import deque
        self.up[0][root] = -1
        seen = [False] * self.n
        seen[root] = True
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    self.up[0][v] = u
                    self.depth[v] = self.depth[u] + 1
                    q.append(v)

    def kth_ancestor(self, v, k):
        """Return the k-th ancestor of v, or -1 if v doesn't have one."""
        if k > self.depth[v]:
            return -1
        i = 0
        while k > 0 and v != -1:
            if k & 1:
                v = self.up[i][v]
            k >>= 1
            i += 1
        return v

    def lca(self, u, v):
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        # lift u to the same depth as v
        diff = self.depth[u] - self.depth[v]
        u = self.kth_ancestor(u, diff)
        if u == v:
            return u
        for k in range(self.LOG - 1, -1, -1):
            if self.up[k][u] != self.up[k][v]:
                u = self.up[k][u]
                v = self.up[k][v]
        return self.up[0][u]


# Example
#         0
#       / | \
#      1  2  3
#     / \    |
#    4   5   6
#   /
#  7
edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7)]
adj = [[] for _ in range(8)]
for a, b in edges:
    adj[a].append(b); adj[b].append(a)
lca = BinaryLiftingLCA(8, adj, root=0)
print(lca.lca(7, 5))          # 1
print(lca.lca(7, 6))          # 0
print(lca.lca(4, 5))          # 1
print(lca.kth_ancestor(7, 2)) # 1
print(lca.kth_ancestor(7, 3)) # 0
print(lca.kth_ancestor(7, 4)) # -1
```

### Query

The query has two phases:

1. **Equalize depth**: lift the deeper node up by `depth(u) - depth(v)` using binary jumps. After this both are at the same depth.
2. **Joint lift**: walk through powers of two from largest to smallest. At each step, if `up[k][u] != up[k][v]`, jump both. The invariant is "after all jumps both still differ" so after the loop both are children of the LCA, and the answer is `up[0][u]`.

The greedy works because every legal jump distance `≤ depth - 1` decomposes uniquely in binary.

### kth Ancestor

Same `up` table — handy for "stone game with k jumps," LeetCode 1483 ("Kth Ancestor of a Tree Node"), and as a primitive for [[heavy_light_decomposition]] when traversing chains.

## Euler Tour + RMQ (Sparse Table)

This approach trades some preprocessing for **O(1) per query** LCA. It's the asymptotically fastest practical LCA.

Idea:
1. Perform a DFS that records the **Euler tour** of the tree: visit `u`, then for each child `c`, recurse into `c` and append `u` again on the way back. The tour has length `2n - 1`.
2. Record `depth_of_tour[i]` = depth of the node at tour position `i`.
3. Record `first[v]` = first index of node `v` in the tour.
4. Build a [[sparse_table]] over the depth array, with combine that returns the index of the smaller depth.

For `lca(u, v)`:
- Let `lo = min(first[u], first[v])`, `hi = max(...)`.
- Find the index `i` in `[lo, hi]` with minimum depth.
- Return `tour[i]`.

The LCA shows up as the minimum-depth node visited along the Euler tour between two occurrences of `u` and `v` — that's the deepest node that's an ancestor of both.

```python
class EulerTourLCA:
    """
    O(n log n) build, O(1) query LCA via Euler tour + sparse-table RMQ.
    Uses an iterative DFS to avoid Python recursion limits.
    """

    def __init__(self, n, adj, root=0):
        self.n = n
        self.tour = []         # nodes in Euler order
        self.depth_of = []     # depth at each tour position
        self.first = [-1] * n
        self._dfs(adj, root)
        m = len(self.depth_of)
        # sparse table holds index into depth_of (i.e., the tour position
        # of the argmin depth in a given range).
        self.log = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log[i] = self.log[i // 2] + 1
        K = max(1, m.bit_length())
        self.st = [list(range(m))]
        for k in range(1, K):
            length = 1 << k
            if length > m:
                break
            prev = self.st[k - 1]
            half = 1 << (k - 1)
            row = [0] * (m - length + 1)
            for i in range(m - length + 1):
                a = prev[i]
                b = prev[i + half]
                row[i] = a if self.depth_of[a] <= self.depth_of[b] else b
            self.st.append(row)

    def _dfs(self, adj, root):
        # iterative DFS that constructs the Euler tour
        stack = [(root, -1, iter(adj[root]))]
        depth = 0
        self.tour.append(root)
        self.depth_of.append(depth)
        self.first[root] = 0
        while stack:
            u, parent, it = stack[-1]
            try:
                v = next(it)
                if v == parent:
                    continue
                depth += 1
                self.tour.append(v)
                self.depth_of.append(depth)
                self.first[v] = len(self.tour) - 1
                stack.append((v, u, iter(adj[v])))
            except StopIteration:
                stack.pop()
                if stack:
                    depth -= 1
                    self.tour.append(stack[-1][0])
                    self.depth_of.append(depth)

    def _argmin_range(self, l, r):
        k = self.log[r - l + 1]
        a = self.st[k][l]
        b = self.st[k][r - (1 << k) + 1]
        return a if self.depth_of[a] <= self.depth_of[b] else b

    def lca(self, u, v):
        lo, hi = self.first[u], self.first[v]
        if lo > hi:
            lo, hi = hi, lo
        return self.tour[self._argmin_range(lo, hi)]
```

The Euler tour has length `2n - 1`, so the sparse table holds about `2n log n` entries. For very large `n` (≥ 5×10^5) this can blow Python memory budgets; binary lifting is more compact.

## Tarjan's Offline LCA (Union-Find)

If **all queries are known up front**, Tarjan's algorithm gives near-linear total time: `O((n + q) · α(n))` using a [[union_find]]. The trick is to bucket queries at each node and resolve them during a single post-order DFS.

```python
class TarjanLCA:
    """
    Offline LCA via DSU.
    Resolve all queries in a single DFS in O((n + q) α(n)).
    """

    def __init__(self, n, adj, root=0):
        self.n = n
        self.adj = adj
        self.root = root
        self.parent_dsu = list(range(n))
        self.ancestor = list(range(n))
        self.visited = [False] * n

    def _find(self, x):
        while self.parent_dsu[x] != x:
            self.parent_dsu[x] = self.parent_dsu[self.parent_dsu[x]]
            x = self.parent_dsu[x]
        return x

    def _union(self, a, b):
        # Union with `a` becoming the new representative (call after recursion)
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent_dsu[rb] = ra

    def answer_queries(self, queries):
        """
        queries: list of (u, v) pairs.
        Returns a list of LCAs in the same order.
        """
        out = [None] * len(queries)
        # bucket queries by both endpoints
        buckets = [[] for _ in range(self.n)]
        for qi, (u, v) in enumerate(queries):
            buckets[u].append((v, qi))
            buckets[v].append((u, qi))

        # iterative post-order DFS
        # stack frames: (node, parent, child_iter, in_or_out)
        stack = [[self.root, -1, iter(self.adj[self.root])]]
        while stack:
            top = stack[-1]
            u, par, it = top
            self.visited[u] = True
            try:
                w = next(it)
                if w == par:
                    continue
                stack.append([w, u, iter(self.adj[w])])
                continue
            except StopIteration:
                pass
            # post-order: resolve queries involving u
            for (v, qi) in buckets[u]:
                if self.visited[v] and out[qi] is None:
                    out[qi] = self.ancestor[self._find(v)]
            # union u with its parent's ancestor pointer
            if par != -1:
                self._union(par, u)
                # ancestor of u's component is parent's ancestor
                self.ancestor[self._find(par)] = par
            stack.pop()
        return out
```

Tarjan is rarely used outside of contests where every other approach is too slow on a specific input. The constant factor is excellent, but the offline restriction kills it in interactive settings.

## Heavy-Path Decomposition for LCA

If you already need [[heavy_light_decomposition]] for path queries, you can answer LCA as a byproduct:

```python
def lca_hld(u, v, head, parent, depth):
    """
    Given head[v] (top of v's heavy chain), parent[], depth[], compute LCA in O(log n).
    """
    while head[u] != head[v]:
        if depth[head[u]] < depth[head[v]]:
            u, v = v, u
        u = parent[head[u]]
    return u if depth[u] < depth[v] else v
```

This walks up the chain hierarchy at most `O(log n)` jumps. Pair with HLD's segment tree for path queries.

## Distance on a Tree

```python
def tree_distance(lca_struct, depth, u, v):
    return depth[u] + depth[v] - 2 * depth[lca_struct.lca(u, v)]
```

**Why it works**: the path `u → v` goes up from `u` to `lca`, then down from `lca` to `v`. The depths of `u` and `v` count each step from the root, but the portion above `lca` is counted twice — subtract it.

For **weighted** trees, replace `depth[]` with `sum_to_root[]` (sum of edge weights from root to node). The formula is identical.

## Auxiliary Tree / Virtual Tree

Given a tree on `n` nodes and a subset `S` of `k` "important" nodes, you can build an **auxiliary tree** of size O(k) that preserves all pairwise LCAs among `S`. This is the canonical trick for "process a query set of tree nodes in O(k log k) instead of O(n)."

Algorithm:
1. Sort `S` by DFS in-time (i.e., the order they appear in a preorder DFS).
2. Add the LCA of every consecutive pair to the set.
3. Sort the augmented set again by in-time.
4. Build a tree by iterating with a stack: each new node connects to the deepest stack node that is its ancestor.

```python
def build_virtual_tree(important, lca_struct, tin):
    """
    important: list of "important" node ids.
    lca_struct: anything with .lca(u, v).
    tin: DFS in-time per node.
    Returns (root_of_virtual_tree, virtual_adj).
    """
    nodes = sorted(set(important), key=lambda v: tin[v])
    extra = []
    for i in range(len(nodes) - 1):
        extra.append(lca_struct.lca(nodes[i], nodes[i + 1]))
    augmented = sorted(set(nodes + extra), key=lambda v: tin[v])

    adj = {v: [] for v in augmented}
    stack = [augmented[0]]
    for v in augmented[1:]:
        while len(stack) >= 2 and not _is_ancestor(stack[-1], v, lca_struct):
            stack.pop()
        adj[stack[-1]].append(v)
        adj[v].append(stack[-1])
        stack.append(v)
    return augmented[0], adj

def _is_ancestor(a, b, lca_struct):
    return lca_struct.lca(a, b) == a
```

Virtual trees power many tree-DP-with-queries problems where the constraints look like "Q queries, each on up to K nodes, with total K bounded."

## LCA in Dynamic Trees

If the tree itself is **changing** (edges added/removed), classical LCA structures don't suffice. Options:
- **Link-Cut Trees**: O(log n) amortized per operation, including LCA.
- **Euler Tour Trees**: maintain a balanced BST over the Euler tour.
- **Top trees**: theoretical optimum but rarely implemented.

These are out of scope for this note; mention them when interviewers ask "what if the tree changes?"

## Common Patterns and Applications

| Pattern | Use |
|---|---|
| Distance(u, v) | `depth[u] + depth[v] - 2 * depth[lca(u,v)]` |
| Path from u to v (set of edges/vertices) | Walk up from both to lca |
| Aggregate on path (sum, min, gcd) | Prefix-from-root + LCA, or [[heavy_light_decomposition]] |
| k-th node on path u → v | Find lca; choose side based on depths; use kth_ancestor |
| Subtree-vs-path queries | Combine LCA with Euler-tour subtree intervals |
| "Is u an ancestor of v?" | `lca(u, v) == u` |
| Tree diameter (offline) | Two BFS/DFS suffice — but LCA helps for "diameter of subset" |
| Auxiliary tree | When queries restrict to small subset of nodes |
| Online tree DP | Combine binary lifting with stored DP values up the chain |

## Interview & Contest Problems

**LeetCode**
- 236. Lowest Common Ancestor of a Binary Tree (small inputs — recursive)
- 1483. Kth Ancestor of a Tree Node (binary lifting)
- 1644. LCA of Binary Tree II (handle missing nodes)
- 1650. LCA of Binary Tree III (uses parent pointers — different approach)
- 1740. Find Distance in a Binary Tree (LCA + distance formula)

**Competitive**
- CSES "Company Queries I" (kth ancestor) and "Company Queries II" (LCA)
- CSES "Distance Queries" (LCA + distance formula)
- CSES "Path Queries" / "Path Queries II" (LCA + BIT/segment tree on Euler tour)
- Codeforces 191C "Fools and Roads" (LCA + difference on tree)
- Codeforces 519E "A and B and Lecture Rooms" (counting midpoints, needs LCA + kth ancestor)
- AtCoder ABC 294 G (HLD using LCA)
- SPOJ LCA, QTREE, QTREE2

## Complexity Comparison

| Algorithm | Build | Query | Memory | Online? |
|---|---|---|---|---|
| Naive (parent walk) | O(n) | O(n) | O(n) | yes |
| Binary lifting | O(n log n) | O(log n) | O(n log n) | yes |
| Euler tour + sparse table | O(n log n) | O(1) | O(n log n) | yes |
| Tarjan's offline (DSU) | O((n + q) α(n)) | amortized α(n) | O(n + q) | no |
| HLD | O(n) | O(log n) | O(n) | yes |
| Link-Cut Tree | O(n log n) build | O(log n) amortized | O(n) | yes (and edits) |

Practical defaults:
- **Contest**: binary lifting unless query count dominates (then Euler+sparse).
- **Library code**: binary lifting for `kth_ancestor` reuse.
- **Memory tight on huge n**: Tarjan offline.

## Common Pitfalls

1. **Recursion depth.** Python defaults to ~1000 recursion. Trees with `n = 10^5+` need iterative DFS or a high `sys.setrecursionlimit`.
2. **Off-by-one in `LOG`.** Need `LOG ≥ ⌈log₂ n⌉`. Setting `LOG = n.bit_length()` is safe for any `n ≥ 1`.
3. **Wrong root.** Ensure `parent[root] = -1` or a self-pointer consistently; mishandling root causes infinite jumps.
4. **Disconnected forests.** All algorithms here assume one connected tree. For a forest, treat each tree independently and report -1 when queried across roots.
5. **Treating undirected edges twice.** When building adjacency, don't re-enter the parent during DFS.
6. **Mixing 0- vs. 1-indexed nodes.** Always check the problem statement; print warnings during local testing.
7. **Forgetting to set `first[v]` on the second visit.** In Euler tour, only the **first** visit to `v` should set `first[v]` — overwriting it on the return visit breaks queries.
8. **Stale ancestor pointer in Tarjan.** After unioning a child, you must immediately set `ancestor[find(parent)] = parent`. Forgetting this returns wrong LCAs.
9. **Edges vs. vertex weights.** For path-sum problems, distinguishing whether weights live on edges or vertices changes how you index the segment tree under HLD.
10. **Using the wrong `combine` in sparse-table RMQ.** The combine returns the **index** of the smaller depth, not the depth value itself.

## Practice Problems

| Source | Problem |
|---|---|
| CSES | Company Queries I & II, Distance Queries, Path Queries (I & II), Subtree Queries |
| LeetCode | 236, 1123, 1483, 1644, 1650, 1740 |
| Codeforces | 191C, 519E, 609E, 832D, 1023F, 1062F |
| SPOJ | LCA, QTREE (and follow-ups), POLICEMEN |
| Library Checker | LCA, Jump on Tree |
| AtCoder | ABC 187 E, ABC 209 D, ABC 294 G, ABC 337 G |

## Additional Resources

- CP-Algorithms: "Lowest Common Ancestor — Binary Lifting" and "LCA with Sparse Table" articles.
- Bender & Farach-Colton 2000: "The LCA Problem Revisited" (Euler-tour reduction to RMQ).
- TopCoder tutorial: "Range Minimum Query and Lowest Common Ancestor" by danielp.
- "Competitive Programming 4" by Halim & Halim — LCA chapter.
- CSES Problem Set guide for tree algorithms.

## Where this connects

- [Trees](../data_structures/trees.md) — LCA is a fundamental tree query; finding common ancestors uses tree structure
- [Sparse table](sparse_table.md) — binary lifting for O(log n) LCA; sparse tables enable O(1) RMQ-based LCA
- [Heavy-light decomposition](heavy_light_decomposition.md) — HLD uses LCA to identify paths and split them into chains
