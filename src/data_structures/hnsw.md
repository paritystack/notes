# HNSW: Hierarchical Navigable Small World Graphs

> **Key Concepts:** Approximate Nearest Neighbor (ANN), Multi-layer Graph, Greedy Search, Neighbor Selection Heuristic, Logarithmic Search

## Table of Contents
- [Overview](#overview)
- [Background: The Nearest Neighbor Problem](#background-the-nearest-neighbor-problem)
- [Foundations: NSW and Skip Lists](#foundations-nsw-and-skip-lists)
- [HNSW Structure](#hnsw-structure)
- [Search Algorithm](#search-algorithm)
- [Insert Algorithm](#insert-algorithm)
- [Neighbor Selection Heuristic](#neighbor-selection-heuristic)
- [Key Parameters](#key-parameters)
- [Implementation](#implementation)
- [Complexity Analysis](#complexity-analysis)
- [Distance Metrics](#distance-metrics)
- [Deletes](#deletes)
- [Concurrency](#concurrency)
- [Filtered Search](#filtered-search)
- [Disk-Resident Variants](#disk-resident-variants)
- [Real-World Systems](#real-world-systems)
- [When to Use HNSW](#when-to-use-hnsw)
- [Tuning Guide](#tuning-guide)
- [Common Pitfalls](#common-pitfalls)
- [Related Topics](#related-topics)

## Overview

**HNSW (Hierarchical Navigable Small World)** is a graph-based index for *approximate* nearest neighbor (ANN) search in high-dimensional vector spaces. It is the dominant in-memory ANN index in production systems (pgvector, Qdrant, Weaviate, Milvus, Lucene 9+, FAISS, hnswlib, Elasticsearch).

Given a query vector `q` and a dataset of `N` vectors, HNSW returns the `k` closest vectors to `q` (under a chosen distance metric) in **expected O(log N)** time with high recall, using O(N · M) memory for graph edges (typically `M = 8..48`).

The structure was introduced by Malkov & Yashunin (2016) and combines two ideas:
1. **Navigable Small-World (NSW) graphs** — graphs where greedy routing finds near-optimal paths in logarithmic steps (Kleinberg, 2000).
2. **Hierarchical layering** — a skip-list-style multi-level graph that lets the search start "from far away" and zoom in.

### Why HNSW Dominates

| Property | HNSW | IVF-PQ | Brute Force |
|---|---|---|---|
| Build cost | High | Medium | None |
| Query latency | Low (log N) | Medium | O(N) |
| Memory | High (~M floats/vec extra) | Low (compressed) | Just vectors |
| Recall@10 (typical) | 0.95–0.99 | 0.85–0.95 | 1.0 |
| Update friendliness | Good (incremental insert) | Poor (needs retraining) | Trivial |
| Tunability | `M`, `efConstruction`, `efSearch` | `nlist`, `nprobe`, `M_pq` | None |

HNSW wins on **latency × recall** for in-memory workloads up to ~100M vectors. For larger or memory-constrained workloads, IVF-PQ or DiskANN become preferable.

## Background: The Nearest Neighbor Problem

### Exact k-NN

Given:
- Dataset `D = {x_1, ..., x_N}` where each `x_i ∈ R^d`
- Query `q ∈ R^d`
- Distance `dist(·, ·)` (L2, inner product, cosine)

Find the `k` vectors in `D` closest to `q`.

**Brute force:** Compute `dist(q, x_i)` for all `i`, return top-k. O(N · d) per query.

For modern embeddings (`d = 384..4096`) and large `N`, brute force is too slow. Space-partitioning trees (k-d trees, ball trees) work well in low dimensions but degrade to O(N) when `d > ~20` — the **curse of dimensionality**.

### Approximate k-NN (ANN)

Trade exactness for speed: return *most* of the true top-k. **Recall@k** measures quality:

```
recall@k = |returned_top_k ∩ true_top_k| / k
```

ANN methods aim for **recall ≥ 0.95** at 10–100× the QPS of brute force.

Major ANN families:
- **Tree-based:** Annoy (random projection trees), Spotify
- **Hashing-based:** LSH (locality-sensitive hashing) — see [[minhash-lsh]]
- **Quantization-based:** PQ, OPQ, ScaNN — see [[product-quantization]]
- **Graph-based:** HNSW, NSG, DiskANN — *this document*

Graph-based methods consistently lead recall-vs-latency benchmarks on dense embeddings (ann-benchmarks.com).

## Foundations: NSW and Skip Lists

### Navigable Small-World Graphs

A graph is **navigable** if greedy routing — at each step, move to the neighbor closest to the target — finds the destination in `O(polylog N)` steps.

Kleinberg (2000) showed this requires a mix of:
- **Short-range edges** (to local neighbors) — enable convergence to exact answer
- **Long-range edges** (to distant nodes) — enable fast traversal across the graph

A pure k-NN graph (each node connects to its `k` nearest neighbors) has only short edges; greedy search gets stuck in local minima. Random edges alone produce a small-world graph but poor recall.

NSW builds long-range edges *incrementally as a byproduct of insertion order*: early-inserted points become "hubs" because later insertions probabilistically connect to them across the whole space.

### Skip List Analogy

A [skip list](skip_lists.md) is a multi-layer linked list where each node appears in layer `ℓ` with probability `p^ℓ`. Search starts at the top layer, scans horizontally, drops down when overshoot occurs. Result: O(log N) search.

HNSW applies the same idea to graphs:

```
Layer 2:         A ───────────────── E
                 │                   │
Layer 1:    A ── C ── D ──────────── E ── G
            │    │    │              │    │
Layer 0:    A ── B ── C ── D ── E ── F ── G ── H
            (all nodes, fully connected NSW)
```

- **Top layers** are sparse — only the "highest-rolling" nodes appear there. They act as long-range entry points.
- **Bottom layer (layer 0)** contains all nodes and supports the final precise search.

The hierarchy lets search start from a far-away entry point and descend, halving the work at each level — the same logic as a skip list.

## HNSW Structure

### Per-Node State

For each node `v`:
- `vector`: the d-dimensional embedding
- `level(v)`: integer ≥ 0, sampled at insertion time
- `neighbors[ℓ]`: list of edges at each layer `ℓ` where `0 ≤ ℓ ≤ level(v)`

Edges are **undirected** in the abstract graph but stored as adjacency lists on both endpoints.

### Global State

- `entry_point`: the highest-level node (search starts here)
- `max_level`: current top layer
- `M`: max neighbors per node at layers `ℓ ≥ 1`
- `M_max0` = `2 * M`: max neighbors at layer 0 (more, because layer 0 carries everything)
- `mL = 1 / ln(M)`: level-generation normalization factor
- `efConstruction`: dynamic candidate list size during build
- `efSearch`: dynamic candidate list size during query (set per-query)

### Level Assignment

Each new node draws a level from a geometric distribution:

```
level = floor(-ln(uniform(0, 1)) * mL)
```

With `mL = 1/ln(M)`:
- P(level = 0) ≈ 1 - 1/M
- P(level = 1) ≈ (1/M)(1 - 1/M)
- P(level = ℓ) decays geometrically

So with `M = 16`, ~94% of nodes only exist at layer 0, ~6% reach layer 1, ~0.4% reach layer 2, etc. The number of layers grows as `O(log N)`.

### Graph Shape per Layer

| Layer | Nodes | Edges/node | Purpose |
|---|---|---|---|
| 0 | All `N` | `M_max0` (2M) | Final precise search |
| 1 | ~N/M | `M` | Mid-range routing |
| 2 | ~N/M² | `M` | Mid-range routing |
| ... | ... | ... | ... |
| top | ~1 | `M` | Long-range entry |

## Search Algorithm

HNSW search has two phases:
1. **Coarse phase** (layers `max_level` → `1`): greedy 1-NN search, find an entry point near `q` at layer 0.
2. **Fine phase** (layer 0): beam search with width `ef`, return top-k.

### Greedy Search at a Single Layer

```
search-layer(q, entry, ef, layer):
    visited      = {entry}
    candidates   = min-heap by dist(node, q),  init {entry}
    best         = max-heap by dist(node, q),  init {entry}
    while candidates non-empty:
        c = candidates.pop-min()           # closest unexplored
        f = best.top()                      # furthest of best
        if dist(c, q) > dist(f, q): break  # no improvement possible
        for e in c.neighbors[layer]:
            if e in visited: continue
            visited.add(e)
            if dist(e, q) < dist(f, q) or len(best) < ef:
                candidates.push(e)
                best.push(e)
                if len(best) > ef: best.pop()
    return best                              # ef closest seen
```

Key invariants:
- `candidates` is a *frontier* — closest unexpanded nodes.
- `best` is the *result set* — at most `ef` closest nodes seen so far.
- Stop when the frontier's closest is farther than the result set's furthest — no more improvements possible.

For the coarse phase, `ef = 1` (pure greedy 1-NN). For the fine phase, `ef = efSearch ≥ k`.

### Full Search

```
search(q, k, efSearch):
    ep = entry_point
    for ℓ in max_level downto 1:
        ep = search-layer(q, ep, ef=1, layer=ℓ).closest()
    result_set = search-layer(q, ep, ef=efSearch, layer=0)
    return top-k from result_set
```

**Why this works:** The coarse phase navigates from far away to a node *near* `q` in O(log N) hops (each layer has ~N/M^ℓ nodes; one greedy step per layer). The fine phase explores `O(efSearch · M)` neighborhoods around that entry, which dominates the total cost.

## Insert Algorithm

```
insert(v, M, M_max0, efConstruction):
    L = current max_level
    ℓ = floor(-ln(rand()) * mL)
    v.level = ℓ

    # Phase 1: descend from top to ℓ+1 with ef=1, just routing
    ep = entry_point
    for layer in L downto ℓ+1:
        ep = search-layer(v.vector, ep, ef=1, layer).closest()

    # Phase 2: from layer min(L, ℓ) down to 0, do real insertion
    for layer in min(L, ℓ) downto 0:
        candidates = search-layer(v.vector, ep, efConstruction, layer)
        M_layer = M_max0 if layer == 0 else M
        neighbors = select-neighbors-heuristic(v, candidates, M_layer)
        for n in neighbors:
            add_edge(v, n, layer)            # bidirectional
            if len(n.neighbors[layer]) > M_layer:
                n.neighbors[layer] = select-neighbors-heuristic(
                    n, n.neighbors[layer], M_layer)
        ep = neighbors                       # entry for next layer

    if ℓ > L:
        entry_point = v
        max_level   = ℓ
```

Insert is essentially "search to find good neighbors, then wire bidirectionally, then re-prune any node that exceeds its degree budget."

The asymmetry between layers (`M_max0 = 2M` vs `M`) reflects layer 0 carrying all traffic and benefiting from extra connectivity, while upper layers stay sparse for fast routing.

## Neighbor Selection Heuristic

The choice of which `M` candidates to keep as neighbors is *the* recall-determining decision in HNSW. Naïve "keep the M closest" produces clustering: a new node connects to its tight cluster and the graph fragments.

### Algorithm 4 (Heuristic Selection) — Malkov & Yashunin

```
select-neighbors-heuristic(q, candidates, M, extendCandidates=False, keepPruned=True):
    if extendCandidates:
        # also consider neighbors-of-candidates
        for c in list(candidates):
            for c2 in c.neighbors[layer]:
                candidates.add(c2)

    result = []
    discarded = []
    W = min-heap by dist(_, q)
    for c in candidates: W.push(c)

    while W non-empty and len(result) < M:
        c = W.pop-min()
        # keep c only if it adds diversity:
        # c is closer to q than to any already-selected r
        is_diverse = all(dist(c, q) < dist(c, r) for r in result)
        if is_diverse:
            result.append(c)
        else:
            discarded.append(c)

    if keepPruned:
        # fill remaining slots from discarded (closest first)
        while discarded and len(result) < M:
            result.append(discarded.pop(0))

    return result
```

### Why the Heuristic Matters

The "closest M" approach picks neighbors that are all in the same direction from `q` — typically the dense cluster `q` belongs to. The graph becomes a bunch of dense cliques weakly connected by random hub nodes; greedy search frequently gets stuck.

The heuristic enforces **angular diversity**: a candidate `c` is kept only if it is closer to `q` than to any already-selected neighbor `r`. This means selected neighbors "spread out" around `q`, giving greedy search escape routes in many directions.

ASCII intuition:

```
Without heuristic (closest M=4):       With heuristic (M=4):
                                       
        ●  ●  ● ●                              ●  ● 
      q                                    q
                                         ●        ●

   All neighbors in one cluster.        Neighbors in four directions.
   Search stuck if target is left.      Search can escape any direction.
```

### `extendCandidates` and `keepPruned`

- **extendCandidates=True**: search wider by including neighbors-of-candidates before pruning. Higher recall, slower build. Typically off.
- **keepPruned=True**: if heuristic prunes too aggressively and result < M, fill remaining slots with closest discards. Default on.

## Key Parameters

| Parameter | Typical | Effect |
|---|---|---|
| `M` | 8–48 | Max neighbors per node (upper layers). Higher → better recall, more memory, slower build. |
| `M_max0` | 2M | Max neighbors at layer 0. Larger gives extra connectivity where most search happens. |
| `efConstruction` | 100–500 | Beam width at build. Higher → better graph quality (better recall at query time), slower build. |
| `efSearch` | 50–500 | Beam width at query. Higher → better recall, slower query. Tunable per query. |
| `mL` | `1/ln(M)` | Level normalization. Default value is near-optimal. |

### Rules of Thumb
- `efSearch ≥ k`. To get recall, you must search at least `k` candidates.
- `efConstruction` ~ 2× `efSearch` gives a good build/query tradeoff.
- `M = 16` is a strong default for most embedding dimensions (~128–768).
- Increase `M` for higher-dimensional or hard datasets (`M = 32..48`).
- Memory per vector ≈ `d * 4 bytes (vector) + M * 4 bytes (neighbor IDs) * avg_levels_per_node`. With `M=16` and `d=768`: ~3 KB vector + ~150 B graph.

## Implementation

A working in-memory HNSW in Python. Production implementations (hnswlib, FAISS) use SIMD, contiguous memory layouts, and lock-free reads, but the algorithm is identical.

```python
import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Node:
    id: int
    vector: list[float]
    level: int
    neighbors: list[list[int]] = field(default_factory=list)  # neighbors[layer] = [node_id, ...]


class HNSW:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        distance: Callable[[list[float], list[float]], float] = None,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.distance = distance or self._l2
        self.nodes: dict[int, Node] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self._rng = random.Random(seed)

    @staticmethod
    def _l2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.mL)

    # ------------------------------------------------------------------
    # Core search at a single layer
    # ------------------------------------------------------------------
    def _search_layer(self, q: list[float], entry_ids: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        visited = set(entry_ids)
        # candidates: min-heap of (dist, id) — closest unexplored
        candidates = []
        # best: max-heap of (-dist, id) — top-ef closest seen
        best = []
        for nid in entry_ids:
            d = self.distance(q, self.nodes[nid].vector)
            heapq.heappush(candidates, (d, nid))
            heapq.heappush(best, (-d, nid))

        while candidates:
            d_c, c = heapq.heappop(candidates)
            d_f = -best[0][0]
            if d_c > d_f:
                break
            for e in self.nodes[c].neighbors[layer]:
                if e in visited:
                    continue
                visited.add(e)
                d_e = self.distance(q, self.nodes[e].vector)
                if len(best) < ef or d_e < -best[0][0]:
                    heapq.heappush(candidates, (d_e, e))
                    heapq.heappush(best, (-d_e, e))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [(-d, nid) for d, nid in best]  # (dist, id)

    # ------------------------------------------------------------------
    # Heuristic neighbor selection (Algorithm 4)
    # ------------------------------------------------------------------
    def _select_neighbors_heuristic(
        self,
        q: list[float],
        candidates: list[tuple[float, int]],
        M: int,
        keep_pruned: bool = True,
    ) -> list[int]:
        # candidates: list of (dist_to_q, id)
        candidates = sorted(candidates)  # closest first
        result: list[tuple[float, int]] = []
        discarded: list[tuple[float, int]] = []
        for d_qc, c in candidates:
            if len(result) >= M:
                break
            is_diverse = True
            for _, r in result:
                d_cr = self.distance(self.nodes[c].vector, self.nodes[r].vector)
                if d_cr < d_qc:
                    is_diverse = False
                    break
            if is_diverse:
                result.append((d_qc, c))
            else:
                discarded.append((d_qc, c))
        if keep_pruned:
            while discarded and len(result) < M:
                result.append(discarded.pop(0))
        return [nid for _, nid in result]

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(self, node_id: int, vector: list[float]) -> None:
        ℓ = self._random_level()
        node = Node(id=node_id, vector=vector, level=ℓ, neighbors=[[] for _ in range(ℓ + 1)])
        self.nodes[node_id] = node

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = ℓ
            return

        ep = [self.entry_point]
        # Phase 1: greedy descend from top down to ℓ+1
        for layer in range(self.max_level, ℓ, -1):
            results = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = [results[0][1]]

        # Phase 2: real insert from min(L, ℓ) down to 0
        for layer in range(min(self.max_level, ℓ), -1, -1):
            results = self._search_layer(vector, ep, ef=self.ef_construction, layer=layer)
            M_layer = self.M_max0 if layer == 0 else self.M
            neighbors = self._select_neighbors_heuristic(vector, results, M_layer)
            node.neighbors[layer] = neighbors
            # Add reverse edges and prune if needed
            for nbr_id in neighbors:
                nbr = self.nodes[nbr_id]
                nbr.neighbors[layer].append(node_id)
                if len(nbr.neighbors[layer]) > M_layer:
                    # re-prune nbr's neighbor list using heuristic
                    cands = [
                        (self.distance(nbr.vector, self.nodes[x].vector), x)
                        for x in nbr.neighbors[layer]
                    ]
                    nbr.neighbors[layer] = self._select_neighbors_heuristic(
                        nbr.vector, cands, M_layer
                    )
            ep = [nid for _, nid in results]

        if ℓ > self.max_level:
            self.entry_point = node_id
            self.max_level = ℓ

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(self, q: list[float], k: int, ef_search: int = 50) -> list[tuple[float, int]]:
        if self.entry_point is None:
            return []
        ep = [self.entry_point]
        for layer in range(self.max_level, 0, -1):
            results = self._search_layer(q, ep, ef=1, layer=layer)
            ep = [results[0][1]]
        results = self._search_layer(q, ep, ef=max(ef_search, k), layer=0)
        results.sort()
        return results[:k]
```

### Usage

```python
import random

random.seed(0)
hnsw = HNSW(dim=64, M=16, ef_construction=100, seed=0)

# Insert 10k random vectors
for i in range(10_000):
    v = [random.random() for _ in range(64)]
    hnsw.insert(i, v)

# Query
q = [random.random() for _ in range(64)]
results = hnsw.search(q, k=10, ef_search=64)
for dist, idx in results:
    print(idx, dist)
```

For production, prefer:
- `hnswlib` (C++ with Python bindings, used by Qdrant, Weaviate)
- `faiss.IndexHNSWFlat` (Facebook AI Research)
- `pgvector` HNSW index for PostgreSQL

## Complexity Analysis

| Operation | Time | Notes |
|---|---|---|
| Search | O(log N) expected | Each layer: O(M · ef) distance comps; O(log N) layers |
| Insert | O(log N · M · efConstruction) | Dominated by efConstruction beam at each layer |
| Memory | O(N · M · d_id + N · d · 4) | d_id = 4-8 bytes per neighbor id, plus vectors |

The O(log N) bound assumes the level distribution is balanced and the graph is well-connected. In adversarial constructions (e.g., all points on a low-dim manifold), search can degrade. In practice on real embeddings, observed scaling matches theory.

### Concrete Example (M=16, d=768, N=10M)

- Levels expected: `log_M(N) ≈ log_16(10^7) ≈ 5.8`
- Avg edges per node: ~M · sum_ℓ p^ℓ ≈ M · M/(M-1) ≈ 17 at layer 0, less at upper layers
- Memory: `10M * (768 * 4 + 32 * 4)` ≈ 30 GB for vectors + 1.3 GB for graph
- Query: traverses ~`efSearch * avg_edges` nodes — `50 * 17 ≈ 850` distance computations vs `10M` for brute force

## Distance Metrics

HNSW is metric-agnostic — any distance/dissimilarity that satisfies *informally* "triangle-inequality-ish behavior" works. Common choices:

| Metric | Formula | Notes |
|---|---|---|
| L2 (Euclidean²) | `Σ(a_i - b_i)²` | Skip the sqrt — monotonic with L2 |
| Inner Product | `-Σ a_i · b_i` (negate to minimize) | Not a true metric; HNSW still works in practice |
| Cosine | `1 - (a·b) / (‖a‖‖b‖)` | Normalize vectors → equivalent to inner product |
| Manhattan (L1) | `Σ|a_i - b_i|` | Supported in some libs |
| Hamming | popcount(a XOR b) | For binary vectors |

For cosine similarity, **normalize vectors once at insert time** and use inner product at query — much faster than recomputing norms.

### Inner Product Caveat

IP is not a true metric (a · a is not the minimum distance from a to itself). The neighbor selection heuristic — which assumes "diverse" means "distinct directions" — can produce odd results with IP if magnitudes vary wildly. Solution: normalize.

## Deletes

HNSW does **not** support true deletes natively. Removing a node disconnects its neighbors and degrades graph quality, especially if the node was a hub.

### Tombstone (Soft Delete)

```python
def delete(self, node_id: int) -> None:
    self.deleted.add(node_id)

def search(self, q, k, ef_search):
    results = self._search_internal(q, k * 2, ef_search)  # over-fetch
    return [(d, nid) for d, nid in results if nid not in self.deleted][:k]
```

- ✅ Simple, O(1) delete
- ❌ Graph still traverses deleted nodes (slows search)
- ❌ Memory not reclaimed
- ❌ Recall degrades as fraction of deleted grows

### Heuristic Repair (FAISS, Qdrant)

When a node is deleted:
1. For each neighbor `n` of the deleted node at each layer:
   - Replace the deleted entry in `n.neighbors[layer]` with a candidate found by re-searching from `n`.
2. If the deleted node was the entry point, pick the highest-level remaining node.

Repair preserves graph quality but is expensive (`O(efConstruction · M · levels)` per delete).

### Periodic Rebuild

For workloads with bulk deletes (e.g., document expiry), tombstone until deleted fraction crosses a threshold (e.g., 20%), then rebuild the index from scratch.

```
deleted_fraction = |deleted| / |nodes|
if deleted_fraction > 0.2:
    rebuild_index(alive_only=True)
```

This is what Lucene's segmented HNSW does — old segments are eventually merged into clean ones.

## Concurrency

### Read-Heavy Workloads

HNSW reads are lock-free if the graph is built and frozen:
- `_search_layer` only reads neighbor lists and vectors.
- Multiple threads can search the same index simultaneously with no synchronization.

### Concurrent Inserts

Concurrent inserts are tricky. Two approaches:

**1. Coarse-grained lock (hnswlib default for builds)**
- Serialize all inserts. Simple, correct.
- Throughput limited; fine for batch builds.

**2. Per-node locks (FAISS, Qdrant)**
- Lock individual neighbor lists when modifying.
- Lock ordering: lock by node ID to avoid deadlock.
- Reads remain mostly lock-free; snapshot neighbor list under brief read lock or use atomic pointer swap.

**3. Versioned graph (Weaviate, Vespa)**
- Inserts go into a delta. Periodic merge applies delta to main index under brief stop-the-world.
- Search reads the current snapshot pointer atomically.

### Common Pitfall: Concurrent Insert + Search

A search may visit a partially-inserted node (its neighbor list isn't fully populated). Mitigations:
- Mark nodes "visible" only after insert completes (don't add to entry-point pool until done).
- Atomic publish: build the full neighbor list off-graph, then atomically swap in the node.

## Filtered Search

Real systems often need to combine ANN search with metadata filters: "find similar images, but only those tagged 'cat' and uploaded after 2024-01-01".

### Pre-filter (Brute Force on Filtered Subset)

If the filter is highly selective (`<1%` of corpus matches), just compute brute-force ANN over matching vectors. Skip HNSW entirely.

### Post-filter

1. Run HNSW search with `efSearch * overfetch_factor` (e.g., 10×).
2. Filter results to those matching predicate.
3. Return top-k of survivors.

- ✅ Simple
- ❌ Fails when filter is very selective — may return < k results even after over-fetching massively
- ❌ Bad worst-case recall

### Filter-Aware Traversal (Milvus, Qdrant, Weaviate)

Modify `_search_layer` to track candidates separately from results:
- Candidates: all visited nodes (used for graph traversal).
- Results: only nodes matching the predicate (returned to user).

```python
def _search_layer_filtered(self, q, entry_ids, ef, layer, predicate):
    visited = set(entry_ids)
    candidates = [(dist(q, n), n) for n in entry_ids]
    heapify(candidates)
    results = [(dist(q, n), n) for n in entry_ids if predicate(n)]
    # ... traverse via candidates, append to results only if predicate(e)
```

Stop condition is trickier: must continue until *results* are saturated, not just candidates. Slower than vanilla search but maintains recall.

### ACORN, FilteredHNSW

Recent research (ACORN, 2024) builds HNSW with extra edges that maintain connectivity *within* common filter classes. The graph has more edges but preserves recall under arbitrary filters.

## Disk-Resident Variants

When the dataset exceeds RAM, in-memory HNSW fails. Options:

### DiskANN (Microsoft Research, 2019)
- Single-layer graph (no HNSW hierarchy — uses graph diameter optimization instead).
- PQ-compressed vectors in RAM for approximate distance.
- Full vectors on SSD, fetched on demand for re-ranking.
- 1B-vector indexes on a single machine with ~64 GB RAM.

### HNSW-on-SSD (qdrant, weaviate, marqo)
- Memory-map graph and vectors.
- Page cache + access pattern works OK if graph fits in RAM and vectors mostly don't.
- Slower than DiskANN due to random IO on every greedy step.

### Tiered HNSW (Vespa, Lucene)
- Recent hot data in RAM HNSW.
- Cold data in compressed on-disk format (often IVF-PQ).
- Query both; merge results.

## Real-World Systems

| System | Notes |
|---|---|
| **hnswlib** | Reference C++ implementation by Malkov. Python bindings. Used by Qdrant and Weaviate. |
| **FAISS** | Facebook AI Research. `IndexHNSWFlat`, `IndexHNSWPQ`, `IndexHNSWSQ`. SIMD-optimized. |
| **pgvector** | Postgres extension. `CREATE INDEX ... USING hnsw (embedding vector_l2_ops)`. M, ef_construction GUCs. |
| **Lucene** | Java HNSW since v9.0. Used by Elasticsearch, OpenSearch, Solr. Segmented + Lucene's merge policy handles deletes. |
| **Qdrant** | Rust. Filtered HNSW with payload indexing. Per-node locks for concurrent writes. |
| **Weaviate** | Go. HNSW + multi-tenancy + hybrid search. |
| **Milvus** | Multi-index backend, HNSW + DiskANN + IVF. |
| **Vespa** | Yahoo/Verizon. HNSW + tiered storage. |
| **MongoDB Atlas Vector Search** | HNSW under the hood (Lucene-based). |
| **Redis (RediSearch)** | HNSW + FLAT indexes for VSS. |

## When to Use HNSW

### Use HNSW when:
- Vector count: 10K – 100M
- Vectors fit in RAM (or can fit with quantization)
- Latency target: <100ms p99
- Recall target: ≥ 0.9
- Updates: incremental, low-volume

### Consider alternatives when:
- **Brute force**: N < 10K, or recall must be exactly 1.0
- **IVF-PQ** ([[product-quantization]]): N > 100M, RAM-limited, lower recall tolerable
- **DiskANN**: billion-scale, single-machine, tight memory budget
- **LSH** ([[minhash-lsh]]): set similarity (Jaccard), not vector L2
- **k-d tree**: dimensions < 20 — see [[spatial-structures]]
- **ScaNN**: very-high-recall regime with anisotropic loss

## Tuning Guide

### Recall Too Low
1. Increase `efSearch` (cheapest fix; per-query)
2. Increase `efConstruction` (requires rebuild)
3. Increase `M` (more memory, requires rebuild)
4. Check normalization / distance metric (cosine vs IP)
5. Inspect data: is it actually low-dim manifold? (try PCA preview)

### Latency Too High
1. Decrease `efSearch`
2. Reduce `k` (overfetch less if you can)
3. Use scalar quantization (SQ8) — 4× memory, ~2% recall hit
4. Use PQ on vectors (`HNSWPQ`)
5. Pin index to NUMA-local memory

### Memory Too High
1. Decrease `M`
2. Use SQ or PQ for vectors
3. Switch to DiskANN for cold tiers
4. Shard across machines

### Build Too Slow
1. Decrease `efConstruction`
2. Parallel inserts (if your impl supports)
3. Use FAISS's `IndexHNSWFlat::train` shortcut
4. Pre-cluster + insert in cluster order (locality-friendly)

## Common Pitfalls

1. **Forgetting to normalize vectors for cosine similarity**
   - Cosine ≡ inner product on unit vectors only.
   - Compute `v / ‖v‖` once at insert time; index in IP space.

2. **Using `efSearch < k`**
   - Returns garbage. Must satisfy `efSearch ≥ k`.

3. **Setting `M` too high**
   - Memory blows up linearly. `M = 64` is rarely better than `M = 32` and uses 2× memory.

4. **Ignoring concurrent insert/search races**
   - Searches can return stale or partial results during inserts.
   - Use library that handles this (hnswlib, faiss) — don't reinvent.

5. **Treating deletes as cheap**
   - Without repair, recall silently degrades as deletes accumulate.
   - Monitor `deleted / total` and rebuild on threshold.

6. **No warmup before benchmarking**
   - First queries page in graph from disk — orders of magnitude slower.
   - Always warm up with 1000+ queries before measuring.

7. **Comparing recall across different metrics**
   - L2 recall and IP recall on the same data are different things.
   - Always benchmark with your *actual* metric.

8. **Dimension mismatch silently works**
   - Inserting a 768-dim vector into a 512-dim index may not error but produces garbage.
   - Assert dimension on insert.

## Related Topics

- [[product-quantization]] — vector compression and IVF-based ANN; often combined with HNSW
- [[minhash-lsh]] — hash-based ANN for set similarity (Jaccard)
- [[spatial-structures]] — k-d trees, R-trees for low-dim spatial search
- [[skip-lists]] — direct conceptual ancestor of HNSW's layered structure
- [[inverted-index]] — keyword search; often hybridized with HNSW for dense+sparse retrieval

External:
- `ai/vector_databases.md` — application-level view of vector DBs that use HNSW
- `ai/rag.md` — RAG pipelines that depend on vector ANN

## Summary

HNSW is the standard for in-memory ANN search because it hits the **best recall-vs-latency tradeoff** on real embeddings while supporting incremental updates. The two ideas that make it work — **multi-layer hierarchy** (for fast routing across the dataset) and the **heuristic neighbor selection** (for graph diversity that prevents greedy search from getting stuck) — are independently elegant and together produce a structure that scales to ~100M vectors per node with sub-100ms queries.

For most teams building vector search in 2026, the question is not "should I use HNSW" but "which HNSW implementation matches my stack" — pgvector for Postgres shops, Lucene-HNSW for Elasticsearch shops, hnswlib/FAISS for custom serving paths, and managed services (Pinecone, Weaviate Cloud, MongoDB Atlas) for everyone else. The data structure is mature; the engineering is in the integration.
