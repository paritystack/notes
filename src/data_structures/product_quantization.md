# Product Quantization & IVF

> **Key Concepts:** Vector Quantization, Subspace Decomposition, Codebooks, Symmetric & Asymmetric Distance, Inverted File Index, OPQ, ScaNN

## Overview

Product quantization (PQ) compresses high-dimensional vectors into compact codes for approximate nearest-neighbor search at scale. Part of [probabilistic structures](probabilistic.md) (approximation family). Compare [MinHash/LSH](minhash_lsh.md) for set-similarity ANN and [spatial structures](spatial_structures.md) (KD-tree) for exact low-dimensional ANN.

## Table of Contents
- [Overview](#overview)
- [Vector Quantization Basics](#vector-quantization-basics)
- [Product Quantization](#product-quantization)
- [Symmetric vs Asymmetric Distance](#symmetric-vs-asymmetric-distance)
- [IVF: Inverted File Index](#ivf-inverted-file-index)
- [IVF-PQ: The Combination](#ivf-pq-the-combination)
- [OPQ: Optimized Product Quantization](#opq-optimized-product-quantization)
- [ScaNN: Anisotropic Vector Quantization](#scann-anisotropic-vector-quantization)
- [Residual Quantization](#residual-quantization)
- [Implementation](#implementation)
- [Complexity Analysis](#complexity-analysis)
- [Memory and Recall Tradeoffs](#memory-and-recall-tradeoffs)
- [Real-World Systems](#real-world-systems)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Related Topics](#related-topics)

## Introduction

**Product Quantization (PQ)** is a vector compression technique that allows you to store and search billion-scale high-dimensional vector collections in a fraction of the memory of raw vectors, with controllable accuracy loss. Combined with the **Inverted File Index (IVF)** for partitioning, it forms the basis of FAISS's most popular indexes (`IVF*,PQ*`) and powers vector search at Facebook, Spotify, Pinterest, and most large-scale ANN deployments where raw vector storage isn't viable.

Where [HNSW](hnsw.md) gives you near-perfect recall in exchange for high memory (vectors + graph), PQ+IVF gives you 10-100× memory reduction at the cost of some recall — the appropriate tradeoff when N exceeds ~10M and you can't afford to store full vectors in RAM.

### What PQ Buys You

| Metric | Raw vectors | PQ (m=8, k*=256) |
|---|---|---|
| Memory per vector (d=128, fp32) | 512 bytes | **8 bytes** |
| Memory per vector (d=768, fp32) | 3072 bytes | **8-32 bytes** |
| Distance comp cost | O(d) flops | O(m) table lookups |
| Recall@10 vs raw | 1.00 | 0.85-0.95 |

A 1B-vector index (d=768) drops from 3 TB (impractical) to ~32 GB (single machine). That difference is why PQ exists.

## Vector Quantization Basics

**Quantization** = mapping a continuous vector to one of a finite set of representatives.

### Scalar Quantization (SQ)

Quantize each dimension independently. Common variants:
- **SQ8**: 8-bit per dim. 4× compression vs fp32.
- **SQ4**: 4-bit per dim. 8× compression, more recall loss.
- **Binary (SQ1)**: 1-bit per dim. 32× compression. Useful as a coarse filter.

```
v = [0.13, -0.45, 0.78, 0.02]    # fp32: 16 bytes
SQ8: per-dim, map [min..max] to [0..255]
   → [33, 0, 255, 18]              # 4 bytes
```

Simple and fast, but treats each dimension independently. Misses correlations between dimensions.

### Vector Quantization (VQ)

Pick `k` representative vectors (a **codebook**) `C = {c_1, ..., c_k}`. For any input `v`, replace it with the index of its closest codebook vector:

```
encode(v) = argmin_i dist(v, c_i)        # an integer in [0, k)
decode(i) = c_i                          # approximate reconstruction
```

Codebook learned via **k-means** on a sample of the dataset.

- ✅ Captures correlations between dimensions
- ✅ Better recall per bit than SQ
- ❌ Codebook size grows exponentially with desired precision: 16-bit codes need 65K centroids, and learning + storing them is expensive.

For 64-bit codes (the regime PQ targets), you'd need 2^64 centroids — clearly impossible.

## Product Quantization

**Key insight (Jégou, Douze & Schmid, 2011):** instead of one giant codebook over all `d` dimensions, split the vector into `m` sub-vectors and quantize each sub-vector independently with its own small codebook.

### Construction

Given vectors of dim `d`, choose:
- `m` = number of sub-quantizers (typical: `d/4` or `8, 16, 32`)
- `k*` = centroids per sub-quantizer (universally `256` — fits in a byte)

Then:
1. Split each vector into `m` sub-vectors of dim `d/m`:
   ```
   v = [v_0, v_1, ..., v_{m-1}]  where v_j ∈ R^{d/m}
   ```
2. Train `m` independent k-means with `k* = 256` clusters, each on a sample of the corresponding sub-vector positions:
   ```
   C_0, C_1, ..., C_{m-1}   each ∈ R^{k* × d/m}
   ```
3. Encode a vector by replacing each sub-vector with its closest centroid's index:
   ```
   code(v) = [argmin_i ‖v_0 - C_0[i]‖, ..., argmin_i ‖v_{m-1} - C_{m-1}[i]‖]
          ∈ {0..255}^m       # m bytes total
   ```

### Why It Works

The total number of representable vectors is `(k*)^m = 256^m`. For `m = 8`: 2^64 distinct codes. That's the resolution of a 64-bit codebook *without* training one — instead, you train m=8 little 256-centroid codebooks, each over d/m dimensions. The total training cost is `m × O(k* × d/m × iters × samples) = O(k* × d × iters × samples)` — completely tractable.

The decomposition assumes sub-vector spaces are roughly independent — which they often *aren't* for raw embeddings. OPQ (below) fixes that.

### Memory

`m` bytes per vector. For typical `m = 8..32`:
- 8 bytes/vector → 1B vectors in 8 GB
- 16 bytes/vector → 1B vectors in 16 GB
- 32 bytes/vector → 1B vectors in 32 GB

vs ~3 TB for fp32 1B × 768-dim vectors.

### Reconstruction

```
decode(code) = concat(C_0[code[0]], C_1[code[1]], ..., C_{m-1}[code[m-1]])
```

The reconstruction is approximate. Quantization error is the L2 distance between `v` and its reconstruction; PQ training minimizes this in aggregate.

## Symmetric vs Asymmetric Distance

How do we compute `dist(query, db_vector)` when the database vector is PQ-encoded? Two strategies.

### Symmetric Distance Computation (SDC)

Encode the query too, then look up the precomputed distance between code centroids:

```
encode_q = pq_encode(q)
dist_sym(q, v) = Σ_j precomputed_centroid_dist(encode_q[j], encode_v[j])
```

The full `m × k* × k*` distance table can be precomputed once and stored in memory (`m × 256 × 256 × 4 bytes` = `m × 256 KB`).

- ✅ Very fast: m table lookups per distance
- ❌ Higher error: query is also quantized → loses information

### Asymmetric Distance Computation (ADC)

Keep the query in original form; quantize only the database. At query time, compute:

```
# Precompute once per query (cheap):
LUT[j][i] = ‖q_j - C_j[i]‖²    for j in [0..m), i in [0..256)

# Per-database-vector cost:
dist_asym(q, v) = Σ_j LUT[j][code_v[j]]
```

- ✅ Better recall (query unquantized)
- ✅ Fast: m table lookups per distance, after one `m × 256` LUT computation per query
- ❌ Slightly more setup per query

**ADC is the default.** SDC is only used for query-side acceleration in some hybrid systems.

ASCII visualization for m=4, k*=256, d=128:

```
Query q (d=128) split into 4 sub-vectors of dim 32.
For each sub-vector j, compute distances to all 256 centroids in C_j:

j=0: [d(q_0, c_0^0), d(q_0, c_0^1), ..., d(q_0, c_0^255)]    # 256 floats
j=1: [d(q_1, c_1^0), d(q_1, c_1^1), ..., d(q_1, c_1^255)]
j=2: [...]
j=3: [...]

Per DB vector v with code [a, b, c, d]:
  approx_dist²(q, v) = LUT[0][a] + LUT[1][b] + LUT[2][c] + LUT[3][d]
                     = 4 table lookups + 3 additions
```

For a 1B-vector index with m=8: 8 lookups + 7 adds per distance. Highly SIMD-friendly.

## IVF: Inverted File Index

PQ accelerates distance computation but doesn't reduce the number of comparisons — you still scan all N vectors. **IVF** adds coarse partitioning: a first-stage clustering that limits search to a fraction of the data.

### Construction

1. Cluster the dataset into `nlist` partitions using k-means:
   ```
   coarse_quantizer = KMeans(n_clusters=nlist).fit(sample)
   coarse_centroids = coarse_quantizer.cluster_centers_   # shape (nlist, d)
   ```
2. Assign each DB vector to its closest coarse centroid:
   ```
   inverted_lists[i] = [v ∈ DB : argmin_c dist(v, c) == i]
   ```
   This is "inverted" in the same sense as an [inverted index](inverted_index.md) — for each cluster (≈ "term"), store the list of vectors (≈ "docs") that belong to it.

### Query

At search time:
1. Find the `nprobe` coarse centroids closest to the query:
   ```
   top_clusters = top_nprobe(dist(q, coarse_centroids))
   ```
2. Search only the inverted lists of those clusters.

```
nlist  = 1024 (e.g.)
nprobe = 8

Total vectors searched ≈ N × nprobe/nlist
                       = N × 8/1024
                       = N / 128   →  128× speedup
```

### Tradeoffs

| nprobe | Speed | Recall |
|---|---|---|
| 1 | Fastest | Lowest (only the right cluster, no spillover) |
| 8-32 | Balanced | Common operating point |
| nlist | Brute force | Recall = 1.0 (defeats the purpose) |

Rule of thumb: `nprobe ≈ √nlist` is a reasonable starting point.

### Choosing nlist

- Too small (e.g., 16): each cluster huge → little speedup.
- Too large (e.g., 1M): coarse quantization at query time becomes expensive (the "find closest centroid" step is O(nlist · d)).
- Sweet spot: `nlist ≈ √N` for balance. For N=1M, nlist≈1024-4096.

## IVF-PQ: The Combination

The combination most heavily used in practice:

```
1. Cluster dataset into nlist coarse clusters (coarse_quantizer).
2. For each DB vector v:
     c = argmin_i dist(v, coarse_centroids[i])
     residual = v - coarse_centroids[c]
     pq_code  = pq_encode(residual)   # encode residual, not v itself
     inverted_lists[c].append((v_id, pq_code))
3. Search:
     For each of nprobe nearest coarse centroids c:
       For each (v_id, pq_code) in inverted_lists[c]:
         residual_q = q - coarse_centroids[c]
         dist²(q, v) ≈ adc(residual_q, pq_code)
       Track top-K
```

**Why encode residuals, not vectors?** Within a single cluster, residuals are small and centered around zero. PQ codebooks trained on residuals capture the *within-cluster* variation, which is what discriminates vectors that all live in the same neighborhood. Much better recall than PQ on raw vectors.

### Two-Stage Codebook Training (FAISS standard)

```
1. Train coarse_quantizer (k-means) on sample → coarse_centroids
2. Compute residuals = sample - coarse_centroids[assignment]
3. Train PQ codebooks (m × 256 sub-quantizers) on residuals
```

## OPQ: Optimized Product Quantization

PQ's assumption is that sub-vector spaces are independent. Real embeddings often have strong cross-dimension correlations and energy concentrated in a few directions — violating this assumption.

**OPQ** (Ge, He, Ke & Sun, 2013) preconditions vectors with a learned rotation matrix `R` so that after rotation, energy is evenly distributed across sub-vectors and they are maximally independent:

```
v' = R · v        # rotate
pq_code = pq_encode(v')   # quantize rotated vector
```

The rotation `R` is learned jointly with the PQ codebooks via alternating optimization (rotate to minimize quantization error → retrain codebooks → repeat). Adds 5–15% recall over vanilla PQ for typical embeddings, at trivial extra cost (one matrix multiply per query).

In FAISS, factory strings like `OPQ32_128,IVF4096,PQ32` chain a 32-byte OPQ pretransform with IVF and 32-byte PQ.

## ScaNN: Anisotropic Vector Quantization

**ScaNN** (Guo et al., Google Research, 2020) extends PQ with an asymmetric, distance-aware loss: training penalizes quantization errors *parallel* to high-similarity directions more than errors orthogonal to them. The intuition: for inner-product / cosine retrieval, errors that change the projection onto the query direction matter more than errors that move you sideways.

ScaNN achieves the highest published recall@10 / latency on ann-benchmarks for many datasets, particularly for inner-product similarity. Available as Google's open-source `scann` library.

## Residual Quantization

A generalization of PQ: instead of one quantization stage, apply multiple stages, each quantizing the residual from the previous stage.

```
v_0 = v
for stage in [0, 1, 2]:
    code[stage] = encode(v_stage)
    v_{stage+1} = v_stage - decode(code[stage])   # residual
```

Final reconstruction = sum of all stage decodings.

- ✅ Higher recall than single-stage PQ at same code length
- ❌ More expensive to encode (multiple k-means)
- Used in: FAISS `IndexResidual`, AQ (Additive Quantization, Babenko & Lempitsky)

**LSQ (Local Search Quantization)** and **AQ** are advanced residual variants that jointly optimize multiple stages.

## Implementation

A complete IVF-PQ index in NumPy, exercising every concept above.

```python
import numpy as np
from typing import Optional


class PQ:
    """Product Quantizer.

    Splits d-dim vectors into m sub-vectors of dim d/m, trains k* = 256
    centroids per sub-vector via k-means.
    """

    def __init__(self, m: int = 8, k_star: int = 256, n_iter: int = 25, seed: int = 0):
        self.m = m
        self.k_star = k_star
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.codebooks: Optional[np.ndarray] = None  # shape: (m, k_star, d_sub)

    def fit(self, X: np.ndarray) -> "PQ":
        n, d = X.shape
        assert d % self.m == 0, "dim must be divisible by m"
        d_sub = d // self.m
        self.d_sub = d_sub
        self.codebooks = np.zeros((self.m, self.k_star, d_sub), dtype=np.float32)
        for j in range(self.m):
            sub = X[:, j * d_sub:(j + 1) * d_sub]
            self.codebooks[j] = self._kmeans(sub, self.k_star)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        codes = np.empty((n, self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = X[:, j * self.d_sub:(j + 1) * self.d_sub]
            # closest centroid per sub-vector
            d2 = ((sub[:, None, :] - self.codebooks[j][None, :, :]) ** 2).sum(axis=2)
            codes[:, j] = d2.argmin(axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        n, _ = codes.shape
        parts = [self.codebooks[j][codes[:, j]] for j in range(self.m)]
        return np.concatenate(parts, axis=1)

    def asymmetric_dist_table(self, q: np.ndarray) -> np.ndarray:
        """Return LUT shape (m, k_star) of ‖q_j - C_j[i]‖² for one query q."""
        lut = np.empty((self.m, self.k_star), dtype=np.float32)
        for j in range(self.m):
            q_sub = q[j * self.d_sub:(j + 1) * self.d_sub]
            lut[j] = ((self.codebooks[j] - q_sub[None, :]) ** 2).sum(axis=1)
        return lut

    def adc(self, q: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric distance computation for one query vs many encoded vecs."""
        lut = self.asymmetric_dist_table(q)
        # sum over m sub-quantizers
        return lut[np.arange(self.m), codes].sum(axis=1)

    # ------------------------------------------------------------------
    def _kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        n = X.shape[0]
        init_idx = self.rng.choice(n, size=k, replace=False)
        centers = X[init_idx].astype(np.float32).copy()
        for _ in range(self.n_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assign = d2.argmin(axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = assign == c
                if mask.any():
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[self.rng.integers(0, n)]  # reinit empty
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return centers


class IVFPQ:
    """Inverted-File index with PQ-encoded residuals."""

    def __init__(self, dim: int, nlist: int = 256, m_pq: int = 8, n_iter: int = 25, seed: int = 0):
        self.dim = dim
        self.nlist = nlist
        self.coarse_centroids: Optional[np.ndarray] = None  # (nlist, dim)
        self.pq = PQ(m=m_pq, k_star=256, n_iter=n_iter, seed=seed)
        self.inverted_lists: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(nlist)]
        self._kmeans_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def train(self, X: np.ndarray) -> None:
        # 1. Train coarse quantizer
        self.coarse_centroids = self.pq._kmeans(X, self.nlist).astype(np.float32)
        # 2. Compute residuals for training PQ
        assignments = self._assign_coarse(X)
        residuals = X - self.coarse_centroids[assignments]
        # 3. Train PQ on residuals
        self.pq.fit(residuals)

    def add(self, ids: np.ndarray, X: np.ndarray) -> None:
        assignments = self._assign_coarse(X)
        residuals = X - self.coarse_centroids[assignments]
        codes = self.pq.encode(residuals)
        for i, c, code in zip(ids, assignments, codes):
            self.inverted_lists[c].append((int(i), code))

    def search(self, q: np.ndarray, k: int = 10, nprobe: int = 8) -> list[tuple[float, int]]:
        # 1. Find nprobe closest coarse centroids
        d2_coarse = ((self.coarse_centroids - q[None, :]) ** 2).sum(axis=1)
        top_clusters = np.argpartition(d2_coarse, nprobe)[:nprobe]

        # 2. Scan the chosen inverted lists with ADC on residual
        best: list[tuple[float, int]] = []
        for c in top_clusters:
            entries = self.inverted_lists[c]
            if not entries:
                continue
            residual_q = q - self.coarse_centroids[c]
            codes = np.array([e[1] for e in entries], dtype=np.uint8)
            # adc(residual_q, code) already estimates the full ‖q - v‖²
            # since residual_q = q - c and the code encodes v - c.
            # Do NOT add d2_coarse[c] — that would double-count the coarse term.
            dists = self.pq.adc(residual_q, codes)
            for (vid, _), d in zip(entries, dists):
                best.append((float(d), vid))

        best.sort()
        return best[:k]

    def _assign_coarse(self, X: np.ndarray) -> np.ndarray:
        # Distance from each row to each coarse centroid
        d2 = ((X[:, None, :] - self.coarse_centroids[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1).astype(np.int32)
```

### Usage

```python
rng = np.random.default_rng(0)
N, d = 100_000, 128
X = rng.normal(size=(N, d)).astype(np.float32)
ids = np.arange(N)

idx = IVFPQ(dim=d, nlist=256, m_pq=8)
idx.train(X[:20_000])    # train on a sample
idx.add(ids, X)           # encode + insert all

q = rng.normal(size=d).astype(np.float32)
hits = idx.search(q, k=10, nprobe=8)
for dist, vid in hits:
    print(vid, dist)
```

For production, use FAISS. The above is for understanding; FAISS adds SIMD, GPU support, multi-threading, and many index variants.

## Complexity Analysis

| Operation | Cost |
|---|---|
| Train (PQ codebooks) | O(m · k* · d/m · iters · sample_size) = O(k* · d · iters · sample_size) |
| Train (coarse k-means) | O(nlist · d · iters · sample_size) |
| Encode one vector | O(d/m · k* · m) = O(d · k*) — dominated by sub-centroid distances |
| Build LUT for query | O(d · k*) |
| Distance comp (per DB vec) | O(m) lookups + adds |
| Query (search) | O(d · nlist + nprobe · (avg_list_len) · m) |
| Storage per vec | m bytes |

For `N = 10^9, d = 768, m = 16, nlist = 4096, nprobe = 32`:
- Memory: ~16 GB for codes + ~12 MB for coarse centroids + ~0.8 MB for PQ codebooks ≈ **16 GB**
- Query: `768 × 4096 + 32 × ~244K × 16 = 3M + 125M = ~128M ops`, vs `768 × 10^9 = 770 G` brute force. **6000× speedup**.

## Memory and Recall Tradeoffs

| Index | Bytes/vec | Recall@10 (typical) | QPS (1B vecs, single machine) |
|---|---|---|---|
| Flat (no quantization) | 4d | 1.00 | <1 (impossible at scale) |
| IVF Flat (no PQ) | 4d + 4 | 0.95-0.99 | 100s |
| IVF-SQ8 | d + 4 | 0.90-0.97 | 1000s |
| IVF-PQ (m=64) | 64 + 4 | 0.85-0.93 | 5000s |
| IVF-PQ (m=16) | 16 + 4 | 0.70-0.85 | 10000s |
| OPQ-IVF-PQ | same as PQ | +5-15% recall | similar |

Real numbers vary heavily with embedding type. SIFT-1B is "easy"; OpenAI-ada-002 embeddings are "hard" (need more bytes for the same recall).

### Memory vs Recall Knobs

- **Increase m**: more bytes/vec, more recall.
- **Increase nlist**: more partitions; faster scan per cluster, more cluster boundaries to cross.
- **Increase nprobe**: probe more clusters; better recall, slower query.
- **Add OPQ**: free 5-15% recall.
- **Switch to ScaNN AQ**: even more recall at same bits, more training cost.

## Real-World Systems

| System | Variant Used | Notes |
|---|---|---|
| **FAISS** | IVF-PQ, IVF-SQ8, OPQ, HNSW-PQ, ResidualQ | Reference implementation; SIMD + GPU. |
| **ScaNN** | AVQ + tree-based partitioning | Google Research; highest published recall/latency. |
| **DiskANN** | PQ in RAM + full vec on SSD | Microsoft; billion-scale, single machine. |
| **Milvus** | IVF-PQ, HNSW, DiskANN | Multi-backend; one of the most-deployed vector DBs. |
| **Pinecone** | Proprietary; PQ + graph hybrid | Managed service. |
| **Vald** | NGT (graph) + PQ | Yahoo Japan; PQ for compression of NGT vectors. |
| **Annoy** | Random projection trees, not PQ | Spotify; predates PQ-IVF popularity. |
| **Vespa** | HNSW + PQ-compressed cold storage | Tiered store. |
| **pgvector 0.7+** | IVFFlat, HNSW; PQ is requested feature | PQ not built-in yet, third-party `pg_idkit`. |

## When to Use

### Choose IVF-PQ when:
- Vector count: 10M – 100B (the regime HNSW can't fit in RAM)
- Memory-constrained budget (cloud cost-sensitive, or single-machine deployment)
- Lower recall (~0.85-0.95) is acceptable
- Batch updates rather than streaming (PQ training requires retraining for distribution shifts)

### Choose HNSW when ([HNSW](hnsw.md)):
- 10K – 100M vectors fit in RAM
- Recall ≥ 0.95 required
- Streaming inserts common

### Choose DiskANN when:
- 100M – 10B vectors, SSD available, RAM constrained
- Latency tolerance up to ~10ms (vs ~1ms for HNSW)

### Combine PQ + HNSW (`IndexHNSWPQ`)
- HNSW graph over PQ-compressed vectors
- Memory of PQ, query speed of HNSW
- Slight extra recall loss vs HNSW on raw vectors

### Choose Flat (no quantization) when:
- < 100K vectors
- Recall must be exact (= 1.0)
- Re-ranking phase after coarse PQ retrieval

## Common Pitfalls

1. **Training PQ codebooks on too few samples**
   - Need ~256k–1M samples for stable k-means at k*=256.
   - Too few samples → codebooks overfit, bad recall on real data.

2. **Training and indexing distributions diverge**
   - PQ assumes the indexed data is similar to training data.
   - If you train on month-1 embeddings then index month-6 embeddings from a different model checkpoint → recall craters.
   - **Retrain when distribution shifts.**

3. **`d` not divisible by `m`**
   - PQ splits into m equal-size sub-vectors. If `d % m != 0`, either pad or use a different m.
   - Common choices: m ∈ {4, 8, 16, 32, 48, 64}.

4. **Using IVF-PQ with too-small nprobe**
   - `nprobe = 1` gives terrible recall — many true neighbors are in adjacent clusters.
   - Start `nprobe = √nlist`; tune for recall target.

5. **Computing the residual query against the wrong centroid (or double-counting it)**
   - Codes encode `v - c` (the residual from the cluster's coarse centroid `c`). At query time you must use `residual_q = q - c` for *that* cluster, so `adc(residual_q, code)` already estimates the full `‖q - v‖²`.
   - Do **not** also add `‖q - c‖²` — that double-counts the coarse term and corrupts cross-cluster ranking. In our implementation: `best.append((float(d), vid))`.

6. **Comparing SDC and ADC numbers as if they're the same**
   - SDC has more error than ADC. A benchmark using SDC underestimates achievable recall.

7. **Not normalizing for inner-product / cosine**
   - PQ works in L2 by default. For cosine, normalize vectors before insert and query — IP on unit vectors equals cosine.

8. **Using float64 codebooks**
   - PQ centroids should be float32. Double precision doubles memory with no recall benefit.

9. **Choosing `m` too large**
   - Diminishing returns past ~64 bytes/vec for most embeddings.
   - At m = d (one byte per dim), you've reinvented SQ8 and lost PQ's main advantage.

10. **Skipping OPQ when embeddings are correlated**
    - Most LLM embeddings have strong cross-dim correlations (rotational symmetry of the embedding model).
    - OPQ is essentially free recall — always test it.

## Related Topics

- [HNSW](hnsw.md) — the in-memory alternative; often combined with PQ for memory-constrained graph indexes
- [MinHash/LSH](minhash_lsh.md) — hash-based ANN for set similarity (not vector L2)
- [inverted index](inverted_index.md) — the keyword counterpart; hybrid retrieval combines both
- [probabilistic structures](probabilistic.md) — sketches (HLL, CMS) share the "lossy compression for queries" philosophy
- [spatial structures](spatial_structures.md) — low-dim alternatives (k-d, ball trees)

External:
- `ai/vector_databases.md` — application-level discussion of vector databases that use PQ/IVF
- `ai/rag.md` — RAG pipelines often use IVF-PQ for the document corpus
- `machine_learning/quantization.md` — model weight quantization (different domain, similar ideas)

## Summary

Product Quantization solved the central problem of large-scale ANN: how to store and compare billion-scale vector collections within commodity RAM. By decomposing high-dimensional vectors into sub-vectors and quantizing each independently with a tiny codebook, PQ compresses 768-dim float32 vectors from 3 KB to 8-64 bytes while keeping distance computations to a handful of table lookups.

IVF adds the second key piece — coarse partitioning so search only touches a small fraction of the data — and the **IVF-PQ** combination is the workhorse of FAISS, Milvus, DiskANN, and virtually every production vector search system above ~10M vectors. OPQ and ScaNN refine the codebook learning to recover even more recall. The combined memory footprint and query latency are what made web-scale and billion-document semantic search economically viable.

For 2026 production: use HNSW when vectors fit in RAM with margin (best latency-recall); use IVF-PQ or HNSW-PQ when they don't; use DiskANN when even compressed vectors exceed RAM. The decision is driven almost entirely by `bytes_per_vector × N` vs available memory.

## Where this connects

- [MinHash/LSH](minhash_lsh.md) — set-similarity ANN counterpart; for sparse set data rather than dense vectors
- [Spatial structures](spatial_structures.md) — exact ANN alternative for low-dimensional (≤20) data
- [Probabilistic structures](probabilistic.md) — PQ is another member of the approximation/lossy family
