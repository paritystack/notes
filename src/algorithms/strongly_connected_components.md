# Strongly Connected Components, Articulation Points, Bridges, and 2-SAT

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [Directed vs. Undirected Connectivity](#directed-vs-undirected-connectivity)
  - [DFS Tree Anatomy](#dfs-tree-anatomy)
- [Strongly Connected Components](#strongly-connected-components)
  - [Kosaraju's Algorithm](#kosarajus-algorithm)
  - [Tarjan's SCC Algorithm](#tarjans-scc-algorithm)
  - [Condensation DAG](#condensation-dag)
- [Articulation Points (Cut Vertices)](#articulation-points-cut-vertices)
- [Bridges (Cut Edges)](#bridges-cut-edges)
- [Biconnected Components](#biconnected-components)
- [2-SAT via SCC](#2-sat-via-scc)
- [Applications](#applications)
- [Interview & Contest Problems](#interview--contest-problems)
- [Complexity Analysis](#complexity-analysis)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

This note bundles **connectivity in directed graphs** with the closely-related concepts in undirected graphs:

- **Strongly Connected Components (SCCs)**: maximal sets of vertices in a directed graph where every pair of vertices is mutually reachable.
- **Condensation DAG**: the directed acyclic graph formed by contracting each SCC to a single node.
- **Articulation points** (cut vertices) and **bridges** (cut edges) in an undirected graph: vertices/edges whose removal increases the number of connected components.
- **Biconnected components**: maximal subgraphs with no articulation points internal to them.
- **2-SAT**: a satisfiability problem on 2-clauses, solvable in linear time via the implication-graph SCC condensation.

All of these are solved by single DFS-tree analyses in O(n + m). They appear across competitive programming, compiler dataflow analysis, network design, and constraint solving.

Related notes:
- [[graph_algorithms]] — DFS, BFS, basics of graph traversal.
- [[union_find]] — alternative for some connectivity problems in undirected graphs.
- [[network_flow]] — many connectivity-related problems reduce to flow.

## ELI10 Explanation

Imagine a city of one-way streets. You'd like to know which intersections are "tightly connected" — meaning if you start at intersection A, you can drive to B **and** drive back from B to A. Such groups of intersections form **strongly connected components (SCCs)**.

If you shrink each SCC to a single super-node and keep only the streets between super-nodes, you get a **DAG** (no cycles) called the **condensation**. The original problem on a complicated directed graph often reduces to a much simpler DAG problem on the condensation.

For two-way streets (undirected), the analogous questions are about **bottlenecks**:
- An **articulation point** is an intersection that, if shut down, splits the city in two.
- A **bridge** is a road whose closure splits the city in two.

Both are found by carefully tracking which DFS-tree edges are "essential" and which are "redundant" because back-edges provide alternative routes.

**2-SAT** is a logic puzzle: each clue says "if you choose A=true then B must be true." Building an implication graph and finding SCCs tells you whether a consistent assignment exists.

## Core Concepts

### Directed vs. Undirected Connectivity

- **Undirected**: "connected" means there's a path. Components partition the vertices uniquely.
- **Directed**:
  - **Weakly connected**: the underlying undirected graph is connected.
  - **Strongly connected**: every pair of vertices is mutually reachable. SCCs partition the vertices.

### DFS Tree Anatomy

During DFS, edges fall into four categories:

| Category | Definition |
|---|---|
| **Tree edges** | `(u, v)` where `v` is first visited from `u`. |
| **Back edges** | `(u, v)` where `v` is an ancestor of `u` (already on the DFS stack). |
| **Forward edges** | `(u, v)` where `v` is a descendant of `u` but not via tree edge alone. (Only in directed graphs.) |
| **Cross edges** | `(u, v)` where `v` is in a different subtree. (Only in directed graphs.) |

Almost every algorithm in this note tracks two arrays:
- `disc[v]` = DFS discovery time of `v`.
- `low[v]` = smallest `disc[w]` reachable from `v`'s subtree via tree edges plus **one** back edge.

The relationship between `disc[v]` and `low[v]` reveals articulation points, bridges, and SCC boundaries.

## Strongly Connected Components

Both classical algorithms run in O(n + m); pick based on style preference and the secondary structure you want.

### Kosaraju's Algorithm

Two passes; conceptually simple:

1. Run DFS on the graph, recording vertices in the order they **finish** (push to a stack).
2. Transpose the graph (reverse every edge).
3. Run DFS on the transpose, popping start vertices off the stack. Each tree in this DFS is one SCC.

```python
import sys
sys.setrecursionlimit(300_000)


def kosaraju(n, adj):
    """
    n: number of nodes (0-indexed).
    adj: adjacency list (directed).
    Returns (num_components, comp[]) where comp[v] is the SCC id of v.
    SCC ids are assigned in topological order on the condensation DAG.
    """
    # 1) order by finish time on original graph (iterative DFS)
    order = []
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, iter(adj[start]))]
        visited[start] = True
        while stack:
            u, it = stack[-1]
            found = False
            for v in it:
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, iter(adj[v])))
                    found = True
                    break
            if not found:
                order.append(u)
                stack.pop()

    # 2) build transposed graph
    radj = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            radj[v].append(u)

    # 3) DFS on transpose in reverse-finish order
    comp = [-1] * n
    cid = 0
    for start in reversed(order):
        if comp[start] != -1:
            continue
        comp[start] = cid
        stack = [start]
        while stack:
            u = stack.pop()
            for v in radj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return cid, comp


# Example
#  0 -> 1, 1 -> 2, 2 -> 0  (SCC {0,1,2})
#  1 -> 3, 3 -> 4, 4 -> 3  (SCC {3,4})
adj = [[1], [2, 3], [0], [4], [3]]
cnt, comp = kosaraju(5, adj)
print(cnt)            # 2
print(comp)           # e.g. [1, 1, 1, 0, 0]  (ids depend on order)
```

**Why it works**: the finish-time ordering ensures that we visit SCCs in **reverse topological order** on the condensation. After transposing, descending from a vertex `v` in this order can only reach `v`'s own SCC — vertices in earlier SCCs are unreachable in the transpose because their forward edges (to `v`'s SCC) became backward edges (from `v`'s SCC).

### Tarjan's SCC Algorithm

A single DFS using `disc[]`, `low[]`, and an auxiliary stack:

```python
def tarjan_scc(n, adj):
    """
    Tarjan's SCC algorithm, iterative to avoid Python recursion limits.
    Returns (num_components, comp[]) with comp ids in REVERSE topological order
    on the condensation (i.e., earlier ids are "later" in topo order).
    """
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []          # current SCC candidate stack
    comp = [-1] * n
    cid = 0
    timer = 0

    for start in range(n):
        if disc[start] != -1:
            continue
        # iterative DFS frames: (u, iterator over adj[u])
        frames = [(start, iter(adj[start]))]
        disc[start] = low[start] = timer; timer += 1
        stack.append(start); on_stack[start] = True

        while frames:
            u, it = frames[-1]
            recurse = False
            for v in it:
                if disc[v] == -1:
                    disc[v] = low[v] = timer; timer += 1
                    stack.append(v); on_stack[v] = True
                    frames.append((v, iter(adj[v])))
                    recurse = True
                    break
                elif on_stack[v]:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if recurse:
                continue
            # post-order for u
            frames.pop()
            if frames:
                p, _ = frames[-1]
                if low[u] < low[p]:
                    low[p] = low[u]
            # is u the root of an SCC?
            if low[u] == disc[u]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = cid
                    if w == u:
                        break
                cid += 1
    return cid, comp
```

**Key invariant**: `low[u] == disc[u]` exactly when `u` is the **root** of an SCC. At that moment, every vertex pushed onto `stack` after `u` is in the same SCC as `u`, and we pop them out.

Tarjan emits SCCs in **reverse topological order** on the condensation (the first SCC found is a sink in the condensation). Kosaraju, by contrast, emits in **forward topological order** (the first SCC found is a source).

### Condensation DAG

Once you have `comp[]`, building the condensation is trivial:

```python
def condense(adj, comp, num_sccs):
    """Return adjacency list of the condensation DAG (no duplicate edges, no self-loops)."""
    out = [set() for _ in range(num_sccs)]
    for u, nbrs in enumerate(adj):
        cu = comp[u]
        for v in nbrs:
            cv = comp[v]
            if cu != cv:
                out[cu].add(cv)
    return [list(s) for s in out]
```

Once condensed:
- **Number of components**: `num_sccs`.
- **Longest path** = longest path in DAG (DP in O(n + m)).
- **Min edges to add to make the whole graph strongly connected**: `max(sources, sinks)` if there's more than one SCC, else 0. (Sources = SCCs with in-degree 0; sinks = SCCs with out-degree 0.)
- **Topological order**: trivial because Tarjan/Kosaraju already produce it.

## Articulation Points (Cut Vertices)

In an **undirected** graph, a vertex `v` is an articulation point if removing it (and its incident edges) increases the number of connected components.

DFS rule:
- **Root**: articulation iff it has ≥ 2 tree-children.
- **Non-root**: articulation iff it has some tree-child `c` with `low[c] >= disc[v]` (no back-edge from `c`'s subtree skips above `v`).

```python
def articulation_points(n, adj):
    """
    Returns the set of articulation points in an undirected graph.
    """
    disc = [-1] * n
    low = [0] * n
    is_ap = [False] * n
    timer = 0

    for root in range(n):
        if disc[root] != -1:
            continue
        # iterative DFS
        parent = [-1] * n
        children = [0] * n
        stack = [(root, iter(adj[root]))]
        disc[root] = low[root] = timer; timer += 1
        while stack:
            u, it = stack[-1]
            recurse = False
            for v in it:
                if disc[v] == -1:
                    parent[v] = u
                    children[u] += 1
                    disc[v] = low[v] = timer; timer += 1
                    stack.append((v, iter(adj[v])))
                    recurse = True
                    break
                elif v != parent[u]:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if recurse:
                continue
            stack.pop()
            if stack:
                p, _ = stack[-1]
                if low[u] < low[p]:
                    low[p] = low[u]
                if parent[p] != -1 and low[u] >= disc[p]:
                    is_ap[p] = True
        if children[root] >= 2:
            is_ap[root] = True
    return [v for v in range(n) if is_ap[v]]
```

**Edge case**: in an undirected graph, if a parent and child are connected by **multiple edges**, the back-edge logic must distinguish "different edge to parent" from "same edge as parent." A clean way is to track **edge ids** rather than vertex ids when comparing against the parent.

## Bridges (Cut Edges)

A bridge is an edge `(u, v)` whose removal disconnects the graph.

DFS rule: edge `(u, v)` (where `v` is a tree-child of `u`) is a bridge iff `low[v] > disc[u]`.

```python
def bridges(n, edges):
    """
    Returns list of bridge edges as (u, v) pairs.
    edges: list of (u, v) undirected edges (may include duplicates).
    """
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    disc = [-1] * n
    low = [0] * n
    timer = 0
    bridge_ids = []

    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer; timer += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, parent_edge, it = stack[-1]
            recurse = False
            for v, eid in it:
                if eid == parent_edge:
                    continue  # skip the edge we came in on
                if disc[v] == -1:
                    disc[v] = low[v] = timer; timer += 1
                    stack.append((v, eid, iter(adj[v])))
                    recurse = True
                    break
                else:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if recurse:
                continue
            stack.pop()
            if stack:
                p, _, _ = stack[-1]
                if low[u] > disc[p]:
                    bridge_ids.append((p, u))
                if low[u] < low[p]:
                    low[p] = low[u]
    return bridge_ids
```

**Why `disc[v]` (not `low[v]`) on a back edge?** Using `low[v]` for back edges would incorrectly propagate over re-entry into the subtree and miss bridges. The convention "tree edges: `low[u] = min(low[u], low[v])`; back edges: `low[u] = min(low[u], disc[v])`" is universal.

## Biconnected Components

A **biconnected component (BCC)** of an undirected graph is a maximal subgraph that remains connected after the removal of any single vertex. Equivalently: a maximal set of edges with no internal articulation points. BCCs partition the **edges** (not vertices — articulation points belong to multiple BCCs).

Algorithm: same DFS as for articulation points, plus an auxiliary edge stack. When you detect an articulation, pop edges off the stack until you pop the tree-edge `(p, u)`. That popped set is one BCC.

```python
def biconnected_components(n, edges):
    """
    Return list of BCCs; each BCC is a list of edge ids.
    """
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    disc = [-1] * n
    low = [0] * n
    timer = 0
    edge_stack = []
    bccs = []

    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer; timer += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, parent_edge, it = stack[-1]
            recurse = False
            for v, eid in it:
                if eid == parent_edge:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer; timer += 1
                    edge_stack.append(eid)
                    stack.append((v, eid, iter(adj[v])))
                    recurse = True
                    break
                elif disc[v] < disc[u]:
                    # back edge going up
                    edge_stack.append(eid)
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if recurse:
                continue
            stack.pop()
            if stack:
                p, _, _ = stack[-1]
                if low[u] >= disc[p]:
                    # pop edges up to and including tree edge (p, u)
                    comp_edges = []
                    while edge_stack:
                        e = edge_stack.pop()
                        comp_edges.append(e)
                        # find the tree edge into u; it's the last one pushed
                        # before recursing into u
                        if e == parent_edge_of(u):
                            break
                    # NOTE: in the iterative form, we track parent_edge per
                    # frame so we can compare here. The cleanest implementation
                    # stashes that tree edge id in the frame itself.
                    bccs.append(comp_edges)
                if low[u] < low[p]:
                    low[p] = low[u]
    return bccs
```

The pseudocode above keeps the algorithm visible; in production you'd inline the parent-edge tracking and avoid the helper. For undirected BCC code that's well-tested, see CP-Algorithms.

The **block-cut tree** built from BCCs and articulation points is a powerful auxiliary structure for tree-DP-like problems on general undirected graphs.

## 2-SAT via SCC

**2-SAT**: given `n` boolean variables and `m` clauses each of the form `(a ∨ b)` (literals), decide satisfiability.

**Implication graph**: each clause `(a ∨ b)` is equivalent to two implications `¬a → b` and `¬b → a`. Build a directed graph on `2n` nodes (one per literal: `x_i` at index `2i`, `¬x_i` at index `2i + 1`) and add both implications.

**Theorem**: the formula is satisfiable iff no SCC contains both `x` and `¬x`.

**Witness**: after computing SCCs in **reverse topological order** (which Tarjan gives natively), assign each variable to the literal whose SCC appears **later** in the order:
- If `comp[x_i] > comp[¬x_i]`, assign `x_i = true`; otherwise `false`.

```python
class TwoSAT:
    """
    2-SAT solver via implication graph + Tarjan's SCC.
    Variables are 0..n-1.
    Literal encoding: 2*i = x_i,  2*i + 1 = ¬x_i.
    """

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(2 * n)]

    def _lit(self, var, value):
        return 2 * var + (0 if value else 1)

    def add_clause(self, var_a, val_a, var_b, val_b):
        """Add (lit_a OR lit_b)."""
        a = self._lit(var_a, val_a)
        b = self._lit(var_b, val_b)
        # ¬a → b
        self.adj[a ^ 1].append(b)
        # ¬b → a
        self.adj[b ^ 1].append(a)

    def add_implication(self, var_a, val_a, var_b, val_b):
        """Add (lit_a → lit_b)."""
        a = self._lit(var_a, val_a)
        b = self._lit(var_b, val_b)
        self.adj[a].append(b)
        # contrapositive
        self.adj[b ^ 1].append(a ^ 1)

    def force(self, var, val):
        """Force variable to a value: add clause (lit OR lit)."""
        self.add_clause(var, val, var, val)

    def solve(self):
        num_scc, comp = tarjan_scc(2 * self.n, self.adj)
        assignment = [False] * self.n
        for i in range(self.n):
            if comp[2 * i] == comp[2 * i + 1]:
                return None  # unsatisfiable
            # Tarjan gives SCC ids in REVERSE topological order:
            # smaller id = "later" in topo. The literal whose SCC has a SMALLER
            # id is the one to set true.
            assignment[i] = comp[2 * i] < comp[2 * i + 1]
        return assignment


# Example: variables x0, x1, x2
# Clauses: (x0 OR x1), (¬x0 OR x2), (¬x1 OR ¬x2)
s = TwoSAT(3)
s.add_clause(0, True,  1, True)
s.add_clause(0, False, 2, True)
s.add_clause(1, False, 2, False)
print(s.solve())  # one valid assignment, e.g. [True, False, True]
```

**Note on the comparison**: the exact direction (`<` vs `>`) depends on which SCC numbering convention you use. In the Tarjan implementation above, SCCs are emitted in reverse topological order, so smaller comp id = later in topo, so we pick the literal with smaller comp id. Test on a known-satisfiable instance to confirm direction.

## Applications

| Problem | Approach |
|---|---|
| Number of SCCs | Kosaraju or Tarjan |
| Topological sort of condensation | SCC algorithm output is already in order |
| Min edges to make graph strongly connected | `max(num_sources, num_sinks)` of condensation (or 0 if it's a single SCC) |
| "Mother vertex" / "vertex reaching all others" | Last finished vertex in DFS; check it actually reaches all |
| Functional graph cycle detection | Each node has one out-edge; SCC = cycle + in-trees |
| Constraint propagation | 2-SAT on Horn-like clauses |
| Find all bridges | DFS with low link |
| Find all articulation points | DFS with low link |
| Block-cut tree (BCC) | DFS with edge stack |
| Reachability between fixed pairs (small set) | Compute SCCs, then DAG reachability via bit DP |

## Interview & Contest Problems

**LeetCode**
- 1192. Critical Connections in a Network (bridges)
- 1568. Minimum Number of Days to Disconnect Island (articulation-flavored)
- 2360. Longest Cycle in a Graph (functional graph SCC)
- 1192 variants for cut-vertex problems
- 802. Find Eventual Safe States (reverse-graph + topo on condensation)

**Competitive**
- Codeforces 999E "Reachability from the Capital" (sources in condensation)
- Codeforces 22E "Scheme" (min edges for strongly connected)
- Codeforces 427C "Checkposts" (min cost + count over SCCs)
- Codeforces 1239D "Catowice City" (2-SAT)
- Codeforces 776D "The Door Problem" (2-SAT)
- SPOJ TOUR (2-SAT)
- CSES "Planets and Kingdoms" (SCCs), "Coin Collector" (DAG longest path on condensation)
- Library Checker: "Strongly Connected Components," "Two SAT," "Biconnected Components"

## Complexity Analysis

| Algorithm | Time | Space |
|---|---|---|
| Kosaraju | O(n + m) | O(n + m) (transpose graph) |
| Tarjan SCC | O(n + m) | O(n) |
| Articulation points / bridges | O(n + m) | O(n) |
| Biconnected components | O(n + m) | O(n + m) (edge stack) |
| 2-SAT (Tarjan-based) | O(n + m) (n = variables, m = clauses) | O(n + m) |

For huge n in Python (10^6+), the iterative form is essential. The Tarjan and bridge implementations above keep an explicit DFS stack with per-frame iterators, which avoids the ~1000-deep CPython recursion limit.

## Common Pitfalls

1. **Recursion depth in Python.** Always use iterative DFS for n > ~5000.
2. **Back edge vs. parent edge in undirected DFS.** If you only track parent **vertex**, multi-edges between `u` and `parent(u)` get misclassified. Track parent **edge id** instead.
3. **Wrong update for back edge.** Use `low[u] = min(low[u], disc[v])` for back edges, NOT `low[v]`. Using `low[v]` is a subtle bug that hides bridges.
4. **`>=` vs. `>` for articulation/bridge condition.** Articulation: `low[child] >= disc[u]`. Bridge: `low[child] > disc[u]`. Don't swap these.
5. **Root articulation check.** The root is an articulation iff it has at least **two** tree children. A normal `low/disc` check doesn't detect the root.
6. **SCC order convention.** Tarjan emits SCCs in **reverse** topological order; Kosaraju in **forward** topological order. Always check before using SCC ids as topo positions.
7. **2-SAT direction.** Match the assignment rule to your SCC-ordering convention. If you mix Tarjan's "comp ids reverse topo" with the wrong inequality, your solver silently emits unsatisfying assignments.
8. **Forgetting self-loops and multi-edges.** Tarjan handles them but the bridge/articulation logic on undirected graphs is sensitive. A self-loop is never a bridge.
9. **Disconnected graphs.** Wrap your DFS in a loop over all start vertices.
10. **Forgetting to handle isolated vertices in BCC.** A vertex with no incident edges forms its own (degenerate) BCC.

## Practice Problems

| Source | Problem |
|---|---|
| CSES | Planets and Kingdoms (SCC), Strongly Connected Edges, Coin Collector (condensation DP), Giant Pizza (2-SAT) |
| LeetCode | 1192, 1568, 802, 2360 |
| Codeforces | 22E, 427C, 999E, 1239D, 776D, 1213F |
| SPOJ | TOUR, ARRPRM (2-SAT), SUBMERGE (BCC), EC_P (bridges) |
| Library Checker | Strongly Connected Components, Two SAT, Biconnected Components, Articulation Points |
| AtCoder | ABC 245 F (SCC + DP), ABC 357 E (functional graph SCC) |
| UVa | 11838 "Come and Go" (SCC), 315 "Network" (articulation) |

## Additional Resources

- CP-Algorithms: "Strongly Connected Components" (Tarjan & Kosaraju), "Cut Vertices and Bridges," "2-SAT" articles.
- Tarjan's original 1972 paper "Depth-first search and linear graph algorithms."
- "Competitive Programming 4" by Halim & Halim — graph chapter on connectivity.
- Aspvall, Plass, Tarjan 1979: "A linear-time algorithm for testing the truth of certain quantified boolean formulas" (the original 2-SAT-via-SCC paper).
- Errichto's YouTube series on SCC and 2-SAT.
