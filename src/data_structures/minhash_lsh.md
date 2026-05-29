# MinHash & Locality-Sensitive Hashing

> **Key Concepts:** Jaccard Similarity, Permutation Sampling, Signatures, Banding, Locality-Sensitive Hashing, b-bit MinHash, SuperMinHash, Weighted MinHash

## Table of Contents
- [Overview](#overview)
- [Jaccard Similarity](#jaccard-similarity)
- [Brute Force vs Sketching](#brute-force-vs-sketching)
- [MinHash: The Core Idea](#minhash-the-core-idea)
- [Constructing Signatures](#constructing-signatures)
- [Locality-Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
- [Tuning Bands and Rows](#tuning-bands-and-rows)
- [b-bit MinHash](#b-bit-minhash)
- [SuperMinHash](#superminhash)
- [Weighted MinHash](#weighted-minhash)
- [Implementation](#implementation)
- [Complexity Analysis](#complexity-analysis)
- [SimHash (Companion Technique)](#simhash-companion-technique)
- [LSH for Other Distances](#lsh-for-other-distances)
- [Real-World Systems](#real-world-systems)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Related Topics](#related-topics)

## Overview

**MinHash** is a probabilistic data structure that estimates the **Jaccard similarity** between two sets in O(1) time, using fixed-size signatures (a few KB) regardless of original set size. **Locality-Sensitive Hashing (LSH)** uses MinHash signatures to build a sub-linear-time index for "find all pairs of sets with Jaccard similarity above a threshold."

Together, they solve at web scale: near-duplicate detection (Google's web-page dedup), document similarity for search/recommendation, training-data deduplication for LLMs (huge in 2023-2026 — required for GPT-4, Llama, Claude training pipelines), plagiarism detection, entity resolution, and genomics k-mer comparison.

For dense vectors (embeddings), see [[hnsw]] / [[product-quantization]]. MinHash is the right tool when your data is naturally **set-shaped**: tokens, n-grams, shingles, k-mers, item interactions, feature membership.

### Why It Matters

| Problem | Naive | MinHash + LSH |
|---|---|---|
| Compute Jaccard(A, B) | O(\|A\| + \|B\|) | O(k) signature comparison |
| Find all pairs with sim > 0.8 in N sets | O(N² · avg_set_size) | O(N · 1/p(false_positive)) |
| Dedupe 1B documents | weeks | hours |
| Cluster 10M near-duplicates | impossible | tractable |

## Jaccard Similarity

For sets A and B:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Range: [0, 1]. 1 = identical, 0 = disjoint.

Properties:
- A distance metric (`1 - Jaccard`) — symmetric, identity-of-indiscernibles, triangle inequality.
- Insensitive to set size when scaled properly (unlike raw intersection size).
- Natural for sets of "shingles" (k-grams of text) → text similarity.

### Shingling Text into Sets

A document becomes a set via **shingling**: take every contiguous k-gram of words or characters.

```
doc  = "the quick brown fox"
3-word shingles = {"the quick brown", "quick brown fox"}
3-char shingles = {"the", "he ", "e q", " qu", "qui", "uic", ...}
```

Two near-duplicate documents (one word swapped, one paragraph reordered) share most shingles → high Jaccard. Two unrelated documents share few → low Jaccard. Shingling makes textual similarity a set-similarity problem.

Typical choices:
- Word shingles, k=5-9: classic web dedup (Broder, 1997).
- Character n-grams, n=5-10: robust to small edits.
- Token shingles for LLM training data dedup: k=13 for 13-gram dedup (Pile, RedPajama).

## Brute Force vs Sketching

For N sets:
- **Pairwise Jaccard**: N²/2 comparisons, each O(set size). Infeasible past N=1M.
- **Inverted index**: build a token→sets map; for each pair sharing tokens, compute Jaccard. Still O(N²) worst case for dense shingles.
- **MinHash signatures**: fixed-size summary per set. O(N · k) total to compute; O(k) per pair-comparison.
- **LSH**: index signatures so only candidate-similar pairs are even compared. Sublinear in N.

## MinHash: The Core Idea

### The Magic Identity

Let `h` be a uniformly random hash function from elements to {0, 1, ..., M-1} for large M (essentially "a random permutation of the universe").

```
Pr[ argmin_{x ∈ A} h(x) == argmin_{x ∈ B} h(x) ] = Jaccard(A, B)
```

In words: **the probability that two sets have the same minimum-hash element equals their Jaccard similarity.**

### Why This Works

Consider the universe of elements that appear in `A ∪ B`. The element with the smallest hash value is equally likely to be any one of them (by uniformity of h). For both sets to share that minimum, the smallest-hashing element must be in the intersection `A ∩ B`. So:

```
Pr[min(h(A)) == min(h(B))] = |A ∩ B| / |A ∪ B| = Jaccard(A, B)
```

One hash function gives a noisy 0-or-1 estimate. Use `k` independent hash functions, count how many give matches, divide by k → unbiased estimate of Jaccard.

### Signatures

The MinHash **signature** of a set is the vector:

```
sig(A) = [min_x h_1(x), min_x h_2(x), ..., min_x h_k(x)]   ∈ N^k
```

`k` is typically 64–512. Signatures are fixed-size regardless of |A|.

Estimating Jaccard:

```
estimated_jaccard(A, B) = |{i : sig(A)[i] == sig(B)[i]}| / k
```

This is unbiased; standard error is `≈ √(j(1-j)/k)`. For `k=200, j=0.7`: stderr ≈ 0.032.

## Constructing Signatures

### Naive: k Hash Functions

```python
def minhash(set_, k, hashes):
    sig = [float('inf')] * k
    for x in set_:
        for i in range(k):
            h = hashes[i](x)
            if h < sig[i]:
                sig[i] = h
    return sig
```

Cost: O(|set| · k). For large k and large sets, that's expensive.

### Universal Hashing

Generate `k` hash functions cheaply by:

```
h_i(x) = (a_i * x + b_i) mod p
```

where `p` is a large prime and `(a_i, b_i)` are k random `(a, b)` pairs. Approximates k independent uniform hashes. Standard in `datasketch` and most production libraries.

### Optimization: One-Permutation Hashing (Li & König, 2010)

Instead of k full hashes, hash each element once into k "bins" by partitioning the hash range. Each bin's MinHash is the minimum within it.

Cost: O(|set|) total (one hash per element, not k hashes per element). ~k× speedup. Slight bias if some bins are empty — filled by neighbor-bin densification (Shrivastava 2014).

### Optimization: BottomK MinHash

Take the bottom-k smallest hash values from a *single* hash function:

```
sig(A) = sorted(h(x) for x in A)[:k]
```

Better statistical properties (smaller variance per signature element) than k independent hashes for the same k. Slightly different similarity estimator. Used by Mash (genomics) and some dedup pipelines.

## Locality-Sensitive Hashing (LSH)

MinHash signatures let you compute pairwise Jaccard quickly, but you still need O(N²) signature comparisons to find all similar pairs. LSH indexes signatures to retrieve only candidate-similar pairs in sub-linear time.

### Banding Technique (the canonical LSH for MinHash)

Split the k-element signature into `b` **bands** of `r` rows each (`b · r = k`):

```
sig = [v_0, v_1, ..., v_{k-1}]
band_0 = (v_0, v_1, ..., v_{r-1})
band_1 = (v_r, v_{r+1}, ..., v_{2r-1})
...
band_{b-1} = (v_{k-r}, ..., v_{k-1})
```

For each band, hash the entire band-tuple to a bucket. Two sets become **candidate pairs** if any band hashes them to the same bucket.

### Probability of Becoming a Candidate Pair

Let `s = Jaccard(A, B)`. The probability they match in a single band (all r positions agree) is `s^r`. The probability they match in *at least one* of b bands is:

```
P(candidate | sim = s) = 1 - (1 - s^r)^b
```

This **s-curve** sharply rises around a tunable threshold.

```
s     P(candidate)  for (b=20, r=5)
0.2   0.006
0.4   0.187
0.5   0.470
0.6   0.802         ← roughly the threshold
0.7   0.964
0.8   0.998
0.9   ~1.0
```

The threshold (50% probability of candidacy) is approximately:

```
t ≈ (1/b)^(1/r)
```

### LSH Index

```
For each set A:
  compute sig(A)
  for each band i in [0..b):
    bucket_key = hash(band_i(sig(A)))
    buckets[i][bucket_key].add(A)

Query(B): collect all candidates from any band's bucket containing B.
For each candidate, verify exact Jaccard (signature or original).
```

Memory: O(N · b) hash table entries. Query cost: O(b + #candidates).

## Tuning Bands and Rows

You control:
- `k = b · r` (signature size)
- `b` (number of bands; affects recall)
- `r` (rows per band; affects precision)

### Choosing for a Threshold

Pick the similarity threshold `t` (e.g., 0.8 for near-dup detection). Then:
- Larger `r`: steeper s-curve at the threshold (sharper cutoff)
- More `b`: more chances to match (higher recall, more false candidates)

For `t = 0.8`:
- `k = 128, b = 32, r = 4` → threshold ≈ 0.56 (too low — many false candidates)
- `k = 128, b = 16, r = 8` → threshold ≈ 0.84 (close to target)
- `k = 128, b = 8, r = 16` → threshold ≈ 0.92 (above target — false negatives)

Practical recipe: solve for `(b, r)` given desired threshold and acceptable error rates. `datasketch.MinHashLSH` has a built-in optimizer.

### S-Curve Visualization

```
P(candidate)
1.0 |                  ___________
    |                 /
    |                /
0.5 |- - - - - - - -+- - - - - - - - -
    |              /
    |             /
0.0 |___________/
    0          t                  1.0
                similarity →
```

Steeper around `t` is better (fewer false positives just below, fewer false negatives just above).

## b-bit MinHash

Storing full 32-bit or 64-bit hash values is wasteful — only the equality comparisons matter, not the actual magnitudes.

**b-bit MinHash** (Li & König, 2010) stores only the `b` least significant bits of each MinHash. Comparison reduces to bit-equality:

```
sig_b(A) = [h_1(A) & mask_b, ..., h_k(A) & mask_b]    # k × b bits total
```

Estimator changes slightly (collisions due to truncated hashes have a small baseline rate):

```
estimated_jaccard ≈ (observed_match_rate - 2^{-b}) / (1 - 2^{-b})
```

Typical: b = 1 or 2 bits. Yields 16-32× memory reduction with minor accuracy loss for high-similarity regimes.

For very high similarity (near-duplicate detection, threshold > 0.8), 1-bit MinHash works well. For mid-range similarity (0.3-0.7), use more bits.

## SuperMinHash

**SuperMinHash** (Ertl, 2017) uses a single hash function and *Fisher-Yates*-style sampling to produce k MinHash-equivalent values with much lower per-element cost.

- O(|set|) total time, not O(|set| · k)
- Lower variance than basic one-permutation MinHash
- Drop-in replacement for MinHash signatures

Implementation is subtle (must track sampled positions and handle empty bins), but the result is the same: k-element signature usable for Jaccard estimation and LSH.

## Weighted MinHash

When elements have **weights** (token frequencies, TF-IDF scores), generalize Jaccard:

```
WeightedJaccard(A, B) = Σ min(w_A(x), w_B(x)) / Σ max(w_A(x), w_B(x))
```

**Consistent Weighted Sampling (CWS)** (Manasse et al., 2010) extends MinHash to weighted sets. Per element, generate not one hash but a hash drawn from a distribution scaled by the weight; the minimum is the weighted MinHash.

ICWS (Improved CWS, Ioffe 2010) and 0-bit CWS (Li 2015) are the practical implementations.

Used for: TF-IDF-weighted document similarity, scientific paper similarity (citation weights), recommendation systems with item ratings.

## Implementation

A complete MinHash + LSH implementation with shingling, signature construction, and an LSH index for similarity search.

```python
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass


# ------------------------------------------------------------------
# Shingling
# ------------------------------------------------------------------
def shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles."""
    tokens = text.lower().split()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def char_shingles(text: str, k: int = 5) -> set[str]:
    """Character-level k-shingles."""
    return {text[i:i + k] for i in range(len(text) - k + 1)}


# ------------------------------------------------------------------
# MinHash
# ------------------------------------------------------------------
MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1


def _hash32(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


@dataclass
class MinHash:
    num_perm: int
    seed: int = 1
    _coeffs: list[tuple[int, int]] = None
    sig: list[int] = None

    def __post_init__(self):
        rng = random.Random(self.seed)
        self._coeffs = [
            (rng.randint(1, MERSENNE_PRIME - 1), rng.randint(0, MERSENNE_PRIME - 1))
            for _ in range(self.num_perm)
        ]
        self.sig = [MAX_HASH] * self.num_perm

    def update(self, x: str) -> None:
        h = _hash32(x)
        for i, (a, b) in enumerate(self._coeffs):
            v = ((a * h + b) % MERSENNE_PRIME) & MAX_HASH
            if v < self.sig[i]:
                self.sig[i] = v

    def update_batch(self, xs) -> None:
        for x in xs:
            self.update(x)

    def jaccard(self, other: "MinHash") -> float:
        assert self.num_perm == other.num_perm
        return sum(1 for a, b in zip(self.sig, other.sig) if a == b) / self.num_perm


# ------------------------------------------------------------------
# LSH (banding)
# ------------------------------------------------------------------
class MinHashLSH:
    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0, "num_perm must divide evenly by bands"
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: list[dict[bytes, list[str]]] = [defaultdict(list) for _ in range(bands)]
        self.signatures: dict[str, list[int]] = {}

    def insert(self, key: str, mh: MinHash) -> None:
        assert mh.num_perm == self.num_perm
        self.signatures[key] = mh.sig
        for i in range(self.bands):
            band = tuple(mh.sig[i * self.rows:(i + 1) * self.rows])
            bucket_key = hashlib.md5(repr(band).encode()).digest()
            self.buckets[i][bucket_key].append(key)

    def query(self, mh: MinHash, min_jaccard: float = 0.0) -> list[tuple[str, float]]:
        candidates: set[str] = set()
        for i in range(self.bands):
            band = tuple(mh.sig[i * self.rows:(i + 1) * self.rows])
            bucket_key = hashlib.md5(repr(band).encode()).digest()
            candidates.update(self.buckets[i].get(bucket_key, []))

        results = []
        for c in candidates:
            sim = self._sig_jaccard(mh.sig, self.signatures[c])
            if sim >= min_jaccard:
                results.append((c, sim))
        results.sort(key=lambda x: -x[1])
        return results

    @staticmethod
    def _sig_jaccard(a: list[int], b: list[int]) -> float:
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    # Compute approximate threshold for current (b, r)
    @property
    def threshold(self) -> float:
        return (1.0 / self.bands) ** (1.0 / self.rows)


# ------------------------------------------------------------------
# Auto-tuner: pick (bands, rows) for a target threshold
# ------------------------------------------------------------------
def best_bands_rows(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (b, r) such that (1/b)^(1/r) ≈ threshold, with b*r = num_perm."""
    best = (1, num_perm, abs((1.0)**(1.0 / num_perm) - threshold))
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        est = (1.0 / b) ** (1.0 / r)
        err = abs(est - threshold)
        if err < best[2]:
            best = (b, r, err)
    return (best[0], best[1])
```

### Usage

```python
docs = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "the quick brown fox jumped over a lazy dog",
    "d3": "completely unrelated text about astronomy and stars",
    "d4": "a quick brown fox jumps over the lazy dog yesterday",
}

mhs = {}
for doc_id, text in docs.items():
    mh = MinHash(num_perm=128)
    for sh in shingles(text, k=3):
        mh.update(sh)
    mhs[doc_id] = mh

# Direct similarity:
print("d1 vs d2:", mhs["d1"].jaccard(mhs["d2"]))  # ~0.4-0.6
print("d1 vs d3:", mhs["d1"].jaccard(mhs["d3"]))  # ~0.0

# LSH for "near-duplicate" search
b, r = best_bands_rows(num_perm=128, threshold=0.5)
lsh = MinHashLSH(num_perm=128, bands=b)
for doc_id, mh in mhs.items():
    lsh.insert(doc_id, mh)

print("Threshold:", lsh.threshold)
query_mh = MinHash(num_perm=128)
for sh in shingles("quick brown fox over the lazy dog", k=3):
    query_mh.update(sh)
print("Matches:", lsh.query(query_mh, min_jaccard=0.3))
```

For production: use `datasketch` (well-tested Python lib), Spark MLlib's `MinHashLSH`, or `simhash`/`Mash` for specialized variants.

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Build signature for one set of size n | O(n · k) classic / O(n) one-perm | O(k) |
| Compare two signatures | O(k) | — |
| Estimate Jaccard standard error | ≈ √(j(1-j)/k) | — |
| LSH insert | O(k) | O(b) buckets per item |
| LSH query (output #C candidates) | O(b + #C) | — |
| Total LSH index for N sets | O(N · b) | O(N · b) |

### Concrete: Dedup 100M Documents

- 100M docs, ~1000 shingles each, k=128.
- Signatures: 100M × 128 × 4 bytes = 51 GB. Sharded across machines.
- LSH: 100M × 16 bands = 1.6B hash entries. Distributed hash table or Spark shuffle.
- Query throughput: ~10K QPS per machine for the candidate phase.
- Total dedup of 100M docs: ~hours on a small cluster (vs centuries for naive pairwise).

## SimHash (Companion Technique)

**SimHash** (Charikar 2002, used by Google for web crawl dedup since the 2000s) is a different LSH technique for **cosine** similarity in high-dim binary space, often confused with MinHash. Different use case.

### How SimHash Works

For each feature (token), compute a random hash → a vector of ±1s. Sum (weighted by feature frequency) to get a fingerprint vector. Take the sign bit-by-bit:

```
fingerprint(doc) ∈ {0, 1}^b      (e.g., b = 64)
```

Two documents have **Hamming distance** in fingerprints proportional to their cosine distance. Near-duplicates → small Hamming distance.

### SimHash vs MinHash

| | MinHash | SimHash |
|---|---|---|
| Metric | Jaccard (set similarity) | Cosine (vector similarity) |
| Signature | k integers (e.g., 128 × 4 B = 512 B) | b bits (e.g., 64 bits = 8 B) |
| LSH | Banding | Hamming-LSH (bit-prefix tables) |
| Used by | Web dedup, training-data dedup | Google web-dedup (post-2007) |

SimHash is more compact; MinHash is more accurate on small set-similarity differences.

## LSH for Other Distances

The LSH framework generalizes — anywhere you have a hash family that maps similar items to the same bucket with high probability, you can build a sub-linear similarity index.

| Distance | LSH Family | Notes |
|---|---|---|
| Jaccard | MinHash | This document |
| Cosine | SimHash, hyperplane LSH | Above |
| L2 / Euclidean | p-stable LSH (Datar et al.) | Project onto random vector, quantize |
| Hamming | Bit-sampling LSH | Random bit positions |
| Inner product | Asymmetric LSH (Shrivastava & Li) | Augment vectors with norms |
| Edit distance | Embedding into Hamming via shingling | Approximate |

For dense vectors and inner-product / cosine, modern graph-based ANN ([[hnsw]]) typically outperforms LSH on recall-vs-latency. LSH remains popular for:
- Theoretical analyzability (PAC-style guarantees)
- Streaming / online settings (LSH naturally supports inserts)
- Hardware-friendly bit operations (SimHash, Hamming-LSH on GPU/FPGA)
- Set similarity specifically (where MinHash wins)

## Real-World Systems

| System | Use Case |
|---|---|
| **datasketch** (Python) | Reference MinHash + LSH library; weighted MinHash, HyperLogLog. |
| **Apache Spark MLlib** | `MinHashLSH` for large-scale similarity over distributed datasets. |
| **Google** | SimHash for web page dedup at crawl time. |
| **Mash / sourmash** | Bottom-k MinHash for genome / metagenome comparison. |
| **Bigslice / Flockdb** | LSH at Twitter for similar-account / similar-tweet detection. |
| **The Pile, RedPajama, FineWeb** | MinHash dedup for LLM training corpora. **Dedup at this scale is what makes LLMs trainable** — duplicates inflate loss, harm generalization. |
| **GPT-4 / Claude / Llama training pipelines** | MinHash + LSH for dedup of pretraining text; one of the most-cited stages of "data work" in model papers. |
| **PostgreSQL `pg_trgm`** | Trigram similarity (related idea, different mechanics). |
| **Lucene `MinHashFilter`** | Tokenstream filter to produce MinHash terms; index those for LSH-like recall. |
| **ScaNN, FAISS LSH** | LSH for dense vectors (less competitive than HNSW today). |

## When to Use

### Use MinHash + LSH when:
- Data is naturally **set-shaped** (shingles, n-grams, k-mers, feature sets)
- Similarity metric is Jaccard or its weighted variant
- Need near-dup detection at corpus scale (10M – 100B documents)
- LLM training-data dedup
- Want streaming / online inserts (no global retraining)
- Need PAC-style guarantees on recall

### Use SimHash when:
- Cosine / dot-product similarity on weighted feature vectors
- Need ultra-compact fingerprints (64-128 bits per item)
- Web-page near-dup specifically

### Use HNSW / IVF-PQ instead when:
- Data is **dense vectors** from neural embeddings
- Cosine/L2 on continuous-valued vectors
- Need higher recall than LSH typically achieves

### Use exact methods when:
- N < ~1M and pairwise is feasible
- Recall must be exactly 1.0
- Set sizes are small enough that O(\|A ∩ B\|) is fast

## Common Pitfalls

1. **Confusing MinHash and SimHash**
   - MinHash for Jaccard (sets); SimHash for cosine (vectors). They're not interchangeable.

2. **Using too few permutations (k)**
   - Standard error scales as 1/√k. For tight similarity discrimination (distinguish 0.7 from 0.75), need k ≥ 256.

3. **Picking bad shingle size k**
   - Too small (k=1): too many shingles match by chance, everything is "similar."
   - Too large (k=20): even paraphrases lose nearly all shingles.
   - Typical sweet spots: k=5-9 words, k=3-5 characters for general text. k=13 for LLM training dedup. k=21 for genomic data (Mash default).

4. **Forgetting that LSH gives candidates, not answers**
   - Always verify with full similarity computation. LSH false positives are intrinsic.

5. **Bands/rows imbalanced for threshold**
   - If your threshold is 0.85 and (b,r) gives threshold 0.5, you'll get millions of false candidates.
   - Use the auto-tuner or solve `(1/b)^(1/r) = t` for your target.

6. **Single hash function for k permutations**
   - All permutations identical → signature is just the single MinHash, k=1 effectively.
   - Ensure independent `(a_i, b_i)` for each i.

7. **Not handling empty sets**
   - Empty set's MinHash is all `MAX_HASH`. Two empty sets "match" with Jaccard 1, which is technically correct but often undesired.
   - Filter empties upstream.

8. **Ignoring set size when interpreting Jaccard**
   - Jaccard(A,B) = 0.5 means very different things for |A|=10 vs |A|=10M.
   - For huge sets where one slightly contains the other, prefer **containment** (|A ∩ B| / min(|A|, |B|)).

9. **Forgetting weighted MinHash for weighted features**
   - Plain MinHash on TF-IDF features ignores weights — incorrect.

10. **Storing 64-bit MinHashes when 16 or 32 would do**
    - For threshold > 0.8 and ~128 permutations, 16-bit MinHash hashes work fine.
    - Memory savings compound at scale.

## Related Topics

- [[hnsw]] — dense-vector ANN; the right tool for embeddings, MinHash for sets
- [[product-quantization]] — vector compression; complements HNSW for memory-bound dense ANN
- [[bloom-filter]] — probabilistic set membership (different goal: "does x belong to S?")
- [[probabilistic]] — sketches family overview (HLL, CMS, T-Digest, MinHash)
- [[inverted-index]] — keyword search; complementary for hybrid retrieval / dedup pipelines
- [[hash-tables]] — universal hashing background that MinHash relies on

External:
- `algorithms/hashing_techniques.md` — universal hashing, perfect hashing
- `algorithms/string_algorithms.md` — shingling, edit distance, related text-similarity primitives
- `ai/rag.md` — training-data dedup in RAG / LLM pipelines

## Summary

MinHash answers "how similar are these two sets?" with a fixed-size signature instead of touching the original sets. The trick — that the probability of matching min-hashes equals Jaccard similarity — is elegant and the standard error is straightforward to control by tuning the signature length. LSH wraps signatures in a hash-bucket index that finds candidate-similar pairs in sublinear time, making "find all pairs above similarity threshold T in a corpus of N" feasible at the scale of billions.

The technique is foundational for web-scale dedup (Broder's original use case at AltaVista), and in 2023-2026 it became central to LLM training: every published frontier model (GPT-4, Claude, Llama, Gemini, DeepSeek) describes a MinHash-LSH-based dedup pass over pretraining data, because un-dedup'd corpora produce drastically worse models. SuperMinHash, b-bit MinHash, and weighted MinHash refine the technique for specific accuracy/memory regimes, but the core MinHash + banding algorithm from the late 1990s remains the workhorse.

For 2026 practice: reach for `datasketch` for Python, Spark `MinHashLSH` for distributed pipelines, and don't confuse it with SimHash (cosine, not Jaccard) or with HNSW/PQ (dense embeddings, not sets). When your data is set-shaped, MinHash is unbeatable.
