# Network Flow

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [The Max-Flow Problem](#the-max-flow-problem)
  - [The Residual Graph](#the-residual-graph)
  - [Max-Flow Min-Cut Theorem](#max-flow-min-cut-theorem)
- [Algorithms at a Glance](#algorithms-at-a-glance)
  - [Ford-Fulkerson Method](#ford-fulkerson-method)
  - [Edmonds-Karp](#edmonds-karp)
  - [Dinic's Algorithm](#dinics-algorithm)
- [Applications](#applications)
- [Complexity Comparison](#complexity-comparison)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Network flow** models transport through a directed graph whose edges have **capacities**.
It answers questions like *"what is the most water these pipes can carry from the reservoir
to the city?"* or *"where is the bottleneck in this traffic network?"* A surprising number
of combinatorial problems — bipartite matching, image segmentation, scheduling, project
selection — turn out to be flow problems in disguise.

This page is the **concept companion** to the flow material in
[Graph Algorithms](graph_algorithms.md), which holds the full, runnable implementations of
the algorithms summarised here. Read this for the intuition and the theorems; go there to
copy the code.

| Algorithm | Path search | Time | When to use |
|---|---|---|---|
| **Ford-Fulkerson** (method) | any augmenting path | `O(E · maxflow)` | Pseudo-polynomial; only with small integer capacities |
| **Edmonds-Karp** | BFS (shortest path) | `O(V·E²)` | Simple, polynomial, fine for moderate graphs |
| **Dinic's** | BFS levels + blocking flow | `O(V²·E)` | Standard high-performance choice; `O(E·√V)` on unit/bipartite graphs |

Related notes:
- [[graph_algorithms]] — full Edmonds-Karp, min-cut, and Dinic's implementations.
- [[strongly_connected_components]] — many connectivity problems reduce to flow.
- [[union_find]] — alternative for some undirected connectivity questions.

## ELI10 Explanation

Picture a network of water pipes. There's a **source** `s` (a reservoir) and a **sink** `t`
(a city). Every pipe can carry only so much water per second — its **capacity**. You want to
push as much water as possible from `s` to `t` at once.

Two rules apply:

1. **You can't overfill a pipe.** Flow on a pipe never exceeds its capacity.
2. **Water doesn't pile up.** At every junction except `s` and `t`, the water flowing in
   equals the water flowing out.

The clever trick is that you're allowed to **"undo"** water you previously sent. If you
shoved water down a pipe and later realise a different route is better, you can cancel part
of the earlier flow and reroute it. Keep finding new routes (including these undo routes)
until no route from `s` to `t` has any spare room. The total you've pushed is the **maximum
flow** — and it's exactly the size of the cheapest "wall" you could build to cut the city
off from the reservoir. That's the magic of max-flow min-cut.

## Core Concepts

### The Max-Flow Problem

- **Input:**
  - A directed graph `G = (V, E)`.
  - A **source** `s` (where flow originates) and a **sink** `t` (where flow drains).
  - A **capacity** `c(u, v) > 0` on each edge `(u, v)`.
- **Goal:** find a flow `f` from `s` to `t` of maximum value subject to:
  1. **Capacity constraint:** `f(u, v) ≤ c(u, v)` on every edge.
  2. **Flow conservation:** for every node except `s` and `t`, flow in = flow out.

### The Residual Graph

The engine behind every max-flow algorithm is the **residual graph**: for each edge it
tracks remaining capacity, *plus* a backward edge that represents the ability to cancel
existing flow. Given an edge with capacity 10 carrying 3 units of flow:

```
            capacity 10, flow 3
        u ───────────────────────▶ v

   residual graph:
        u ───── 7 (room left) ────▶ v     forward: push up to 7 more
        u ◀──── 3 (cancel flow) ──── v     backward: undo up to 3 already sent
```

An **augmenting path** is any path from `s` to `t` along edges with positive residual
capacity. Pushing flow equal to the path's smallest residual capacity (its **bottleneck**)
strictly increases total flow. Repeat until no augmenting path remains.

### Max-Flow Min-Cut Theorem

The fundamental theorem of network flow ties the maximum flow to the cheapest way to
*disconnect* `s` from `t`.

- An **s-t cut** partitions the vertices into two sets `S ∋ s` and `T ∋ t`.
- The **capacity** of the cut is the sum of capacities of edges crossing from `S` to `T`.

```
        ┌──────── S ────────┐ ┊ ┌──────── T ────────┐
        s ──▶ a ──▶ b ──────────▶ c ──▶ t
                       └── 4 ──┊── crossing edges ──▶
                          cut capacity = sum of S→T edge capacities
```

> **Theorem:** the value of the maximum flow equals the capacity of the **minimum** s-t cut.

**Intuition:** flow is throttled by the tightest bottleneck, and the min cut *is* that
bottleneck. After running a max-flow algorithm, the set of vertices still reachable from `s`
in the residual graph is exactly the `S` side of a minimum cut — that's how you recover the
cut itself (see [Graph Algorithms](graph_algorithms.md) for the `find_min_cut` routine).

## Algorithms at a Glance

All three share the Ford-Fulkerson skeleton — *find an augmenting path, push its bottleneck,
update the residual graph, repeat* — and differ only in **how** they choose the path. Full
implementations live in [Graph Algorithms](graph_algorithms.md#ford-fulkerson-method).

### Ford-Fulkerson Method

The generic template:

1. Initialise all flow to 0.
2. While an augmenting path `s → t` exists in the residual graph: find its bottleneck and
   push that much flow, updating residual capacities.
3. Return the total flow.

Ford-Fulkerson is a **method**, not a concrete algorithm, because it never says *how* to
find the path. With a naïve search and irrational capacities it can even fail to terminate;
with integer capacities it runs in `O(E · maxflow)` — fine when `maxflow` is tiny, dangerous
otherwise.

### Edmonds-Karp

Ford-Fulkerson with **BFS** to always pick the augmenting path with the fewest edges. This
single choice guarantees termination in `O(V·E²)` regardless of capacity values — the
standard, easy-to-trust polynomial flow algorithm.

### Dinic's Algorithm

The high-performance default for contests and large graphs:

- **Level graph:** a BFS from `s` labels each node with its distance, keeping only edges that
  advance to the next level.

  ```
  level:  0      1       2       3
          s ──▶ a ──▶  c ──▶   t
           └──▶ b ──▶  c
          (only forward, level+1 edges survive)
  ```

- **Blocking flow:** a DFS then saturates the level graph, pushing flow along many paths
  before the next BFS rebuilds levels.

Dinic's runs in `O(V²·E)` in general, and a remarkable `O(E·√V)` on **unit-capacity**
networks — which is exactly what bipartite matching reduces to.

## Applications

1. **Bipartite matching** — connect a source to all applicants and all jobs to a sink, each
   edge capacity 1; **max flow = maximum matching**. (See the reduction in
   [Graph Algorithms](graph_algorithms.md).)
2. **Image segmentation** — separate foreground from background by computing a min cut over a
   pixel graph.
3. **Airline / fleet scheduling** — can a fleet cover a set of flights? Model coverage as
   flow feasibility.
4. **Project selection & closure** — pick a maximum-profit subset under prerequisite
   constraints via min cut.
5. **Circulation with demands** — flows with lower bounds, where edges *must* carry a minimum
   amount.

## Complexity Comparison

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Ford-Fulkerson (DFS) | `O(E · maxflow)` | `O(V)` | Pseudo-polynomial; integer capacities only |
| Edmonds-Karp (BFS) | `O(V·E²)` | `O(V)` | Polynomial, independent of capacities |
| Dinic's | `O(V²·E)` | `O(V + E)` | `O(E·√V)` on unit/bipartite graphs |

Use an **adjacency list** for sparse graphs. The adjacency-matrix formulation often shown in
textbooks costs `O(V²)` per BFS and blows up on large sparse networks.

## Common Pitfalls

- **Forgetting the reverse edge.** Without backward residual edges you can't cancel
  sub-optimal flow, and the algorithm returns a value below the true maximum. Always add a
  `0`-capacity reverse edge when you add a forward edge.
- **Confusing the method with an algorithm.** "Ford-Fulkerson" alone doesn't fix the path
  search; its termination and complexity depend entirely on that choice (BFS → Edmonds-Karp).
- **Real-valued capacities.** With irrational capacities a naïve augmenting-path search may
  loop forever; BFS-based Edmonds-Karp avoids this.
- **Adjacency-matrix blowup.** `O(V²)` per search is wasteful on sparse graphs — prefer an
  adjacency-list residual structure.
- **Reading the min cut wrong.** The cut is the set reachable from `s` in the *residual*
  graph after max flow, not in the original graph.

## Practice Problems

- **Maximum Flow** — classic `s`–`t` max flow (e.g. SPOJ FASTFLOW, judge "Network Flow").
- **Bipartite matching** — LeetCode 1066 "Campus Bikes II", maximum matching on a grid.
- **Minimum Cut** — partition a graph at the cheapest bottleneck.
- **Project selection / maximum closure** — profit maximisation under prerequisites.
- **Edge-disjoint / vertex-disjoint paths** — count via unit-capacity flow (Menger's theorem).

## Additional Resources

- [[graph_algorithms]] — runnable Edmonds-Karp, min-cut, and Dinic's code.
- CLRS, *Introduction to Algorithms*, Ch. 26 "Maximum Flow".
- Competitive Programming literature on Dinic's and push-relabel for the fastest variants.
