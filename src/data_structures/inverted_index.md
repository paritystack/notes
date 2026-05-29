# Inverted Index

> **Key Concepts:** Term Dictionary, Posting Lists, Skip Pointers, Integer Compression, Segmented Index, Top-K Retrieval

## Table of Contents
- [Overview](#overview)
- [The Search Problem](#the-search-problem)
- [Structure](#structure)
- [Term Dictionary](#term-dictionary)
- [Posting Lists](#posting-lists)
- [Integer Compression](#integer-compression)
- [Skip Pointers](#skip-pointers)
- [Building an Index](#building-an-index)
- [Query Processing](#query-processing)
- [Top-K Retrieval](#top-k-retrieval)
- [Segmented Indexes and Updates](#segmented-indexes-and-updates)
- [Implementation](#implementation)
- [Complexity Analysis](#complexity-analysis)
- [Distributed Inverted Index](#distributed-inverted-index)
- [Real-World Systems](#real-world-systems)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Related Topics](#related-topics)

## Overview

An **inverted index** is the data structure that powers full-text search: keyword search engines, log search, code search, document retrieval, e-commerce search, and the keyword side of hybrid (sparse+dense) retrieval. It inverts the natural document→terms mapping into a terms→documents mapping, enabling O(matching documents) query cost instead of O(total corpus size).

Every search engine you have ever used — Google, Elasticsearch/OpenSearch, Solr, Tantivy, Bleve, Sphinx, Whoosh, Vespa, MeiliSearch, Algolia, Lucene — is built around an inverted index. The data structure is ~50 years old; the engineering around it is what differentiates implementations.

### What an Inverted Index Gives You

| Operation | Cost |
|---|---|
| `term → docs containing term` | O(\|posting list\|) |
| `term1 AND term2 → docs containing both` | O(min(\|L1\|, \|L2\|)) with skip pointers |
| `term1 OR term2` | O(\|L1\| + \|L2\|) |
| `phrase "term1 term2"` | O(\|L1\| + \|L2\|) using positional lists |
| `top-K by score` | O(\|results\| · log K), faster with WAND/MaxScore pruning |

vs. brute force scan (O(N · avg_doc_size)), the speedup is many orders of magnitude on real corpora.

## The Search Problem

Given:
- A corpus of `N` documents `D = {d_1, ..., d_N}`
- A query `q` (a set of terms, phrase, or boolean expression)
- A relevance scoring function `score(d, q)` (BM25, TF-IDF, custom)

Return the top-K documents ranked by score.

Brute force: scan every document, compute its score, keep top-K. Cost O(N · |query| · avg_doc_size). For `N = 10^9` documents and microsecond budgets, this is impossible.

Inverted index: precompute, for each term `t`, the list of documents containing `t` (and optionally positions, frequencies, payloads). At query time, intersect/union these lists. Cost proportional to the *matching* document count, not the corpus size.

## Structure

```
TERM DICTIONARY                     POSTING LISTS
+----------+-----------------+      +-----------------------------------+
| "apple"  | ptr → [posts]   | ───▶ | docID:3 tf:2 pos:[5, 17]          |
| "banana" | ptr → [posts]   | ───▶ | docID:7 tf:1 pos:[12]             |
| "cat"    | ptr → [posts]   | ───▶ | docID:12 tf:4 pos:[2, 8, 30, 50]  |
| ...      |                 |      | ...                               |
+----------+-----------------+      +-----------------------------------+
```

Two parts:
1. **Term dictionary**: maps each term to its posting list location.
2. **Posting lists**: per-term lists of `(docID, frequency, positions, ...)`.

Posting lists are sorted by `docID` (always), which enables fast intersection via merge-style scans and skip pointers.

### Posting Entry Variants

| What's stored | Use case |
|---|---|
| `docID` only | Filter / boolean retrieval |
| `docID, tf` | TF-IDF / BM25 scoring |
| `docID, tf, positions` | Phrase queries, proximity ranking |
| `docID, tf, positions, payloads` | Custom per-position scoring (e.g., field weight, attention) |

Modern systems (Lucene) let you choose per-field which to store, since positions ~10× the index size of `docID, tf`.

## Term Dictionary

Maps `term → posting_list_pointer`. Critical that this is small (fits in memory) and fast (O(1) or O(log V) per lookup, where V = vocabulary size).

### Data Structure Choices

| Structure | Lookup | Memory | Prefix Search | Used By |
|---|---|---|---|---|
| Hash table | O(1) | Highest | No | MeiliSearch (parts) |
| Sorted array + binary search | O(log V) | Low | No | Old Lucene |
| B-tree | O(log V) | Medium | Yes | Some research systems |
| Trie | O(\|term\|) | Medium | Yes | Whoosh, Lucene (in-mem) |
| **FST (Finite State Transducer)** | O(\|term\|) | **Very low** | Yes | **Lucene, Tantivy** |
| Burst trie / HAT-trie | O(\|term\|) | Low | Yes | Research |

### FSTs: The Modern Default

A **Finite State Transducer** stores a sorted set of strings → values in a compressed, lookup-friendly DAG. Lucene popularized FSTs as the term dictionary because:
- ~10 bytes per term (vs ~30+ for a trie) for English text — shared prefixes *and* shared suffixes.
- O(|term|) lookups, no hashing required.
- Naturally supports prefix queries (auto-complete), range queries, fuzzy queries (Levenshtein automaton intersection).
- Sequential building from sorted input.

ASCII sketch of an FST for `{cat, cats, dog}`:

```
        c                  a                  t                  ε(→ptr_cat)
   ●────────▶ ●────────▶ ●────────▶ ●─────────────▶ (cat posting list)
                                     │
                                     │ s
                                     ▼
                                     ● ───── ε (→ptr_cats)
   d                  o                  g
   ●────────▶ ●────────▶ ●────────▶ ●─────────────▶ (dog posting list)
```

Lookups walk the FST one character at a time; the value (posting list pointer) is accumulated along the path.

### Trie Alternative

A [trie](tries.md) achieves the same prefix-sharing without the suffix-sharing of an FST. Simpler to build and update but uses 2–5× the memory. For small in-memory indexes (test code, embedded), tries are fine; for production-scale indexes, FSTs win.

### Two-Level Term Dictionary (Lucene)

For very large vocabularies, Lucene splits the dictionary into:
- `tip` file: sparse "skip" entries — every Nth term — in an FST. Fits in heap.
- `tim` file: full term list, on disk. Lookup uses the FST to find the right disk block, then scans within the block.

This pattern (sparse in-memory index, full on-disk lookup table) is the same as B+ trees and LSM SST tables.

## Posting Lists

A posting list is a sequence of postings sorted by `docID`. The naïve representation:

```
list_for("apple") = [(3, 2, [5, 17]), (7, 1, [12]), (12, 4, [2, 8, 30, 50]), ...]
```

For a 1B-document corpus with millions of postings per common term, naïve storage is prohibitive. Posting lists are **compressed**, both for space (10–20× reduction typical) and speed (compressed lists fit in L3 / RAM, decoded on the fly faster than reading uncompressed from disk).

### Delta Encoding (the universal trick)

Since `docID`s are sorted, store **gaps** instead of values:

```
docIDs:  [3, 7, 12, 45, 100, 102, 103]
gaps:    [3, 4,  5, 33,  55,   2,   1]    ← smaller numbers → better compression
```

Positions within a single document are also stored as gaps from the previous position.

After delta encoding, the integers are small most of the time. The next question is how to pack small integers efficiently.

## Integer Compression

Compression methods for sorted integer sequences. Most are designed for posting list gaps specifically.

### 1. VarByte (Variable-Length Byte)

Use 7 bits per byte for data, 1 bit as a continuation flag.

```
encode(130):
  130 = 0b10000010
  → byte0: 0b10000010 (low 7 bits + continuation flag)
  → byte1: 0b00000001 (high bits, no continuation)
  
Result: [0b10000010, 0b00000001]  → 2 bytes
```

- ✅ Simple, branchy, OK speed
- ✅ Each integer independently encoded (no block alignment)
- ❌ Branchy decode (slow on modern CPUs)
- ❌ Worst case wastes 1 bit per byte

**Used by:** Protocol Buffers, classic Lucene (`vInt`).

### 2. Group Varint

Pack 4 integers together: 1 byte of length descriptors (2 bits × 4) + 4–16 data bytes.

```
descriptor byte: aa bb cc dd     # aa = bytes of int1 minus 1, etc.
```

- ✅ Branch-free decode (lookup table)
- ✅ ~50% faster than VarByte
- ❌ Slightly worse compression (length byte overhead)

**Used by:** Google's internal indexes, snappy-style codecs.

### 3. FOR (Frame of Reference)

For a block of `k` integers (typically `k = 128`):
- Find `min` and `max`.
- Store `min` once.
- Store `(x - min)` using `ceil(log2(max - min + 1))` bits each.

```
block = [100, 105, 110, 102, 108]
min = 100, max - min = 10, bits = 4
encoded: min=100, deltas=[0, 5, 10, 2, 8] (4 bits each)
```

- ✅ Compact, fixed bit-width per block (fast SIMD decode)
- ❌ Single outlier inflates the block

### 4. PFOR-Delta (Patched Frame of Reference)

FOR plus "patching" for outliers:
- Pick a bit-width `b` such that 90% of values fit.
- Store outliers in a side list with their positions.

```
block = [100, 105, 110, 102, 999, 108]
b = 4 (fits 90%)
delta list: [0, 5, 10, 2, _exception_, 8] (4 bits each)
exception: position=4, value=999
```

- ✅ Best compression + speed combo for typical posting lists
- ✅ SIMD-friendly main decode loop
- ❌ Slightly more complex

**Used by:** Lucene 4.0+ default (`Lucene41` codec), Tantivy.

### 5. Simple9 / Simple16

Pack as many small integers as possible into a single 32-bit word using a 4-bit selector that encodes the layout.

| Selector | Layout |
|---|---|
| 0 | 28 × 1-bit |
| 1 | 14 × 2-bit |
| 2 | 9 × 3-bit (+1 bit waste) |
| ... | ... |
| 8 | 1 × 28-bit |

- ✅ Word-aligned (fast)
- ✅ Decent compression
- ❌ Wastes some bits when values don't fit selector slots evenly

**Used by:** Older Lucene, research systems.

### 6. Roaring Bitmaps

For dense posting lists (e.g., very common terms, "stopword"-like), store as a bitmap. For sparse lists, store as a sorted array. Roaring partitions the docID space into chunks of 2^16 and chooses the best representation per chunk:
- Array (sparse, <4096 set bits)
- Bitmap (dense, ≥4096 set bits)
- Run-length (long contiguous runs)

- ✅ Very fast set operations (AND, OR) — directly on bitmaps
- ✅ Good compression across all densities
- ❌ Worse than PFOR-Delta for the medium-sparsity regime

**Used by:** Druid (segment bitmap indexes), Elasticsearch (filter cache), Lucene (`BitDocIdSet`).

### Comparison Table

| Codec | Bits/int (typical) | Decode speed | Best for |
|---|---|---|---|
| VarByte | 8–16 | Medium | Simple, single ints |
| Group Varint | 8–14 | Fast | Small blocks |
| FOR | 4–12 | Very fast | Uniform blocks |
| PFOR-Delta | 4–10 | Very fast | Mixed (real posting lists) |
| Simple9/16 | 4–8 | Fast | Small values |
| Roaring | depends | Very fast (set ops) | Sparse-to-dense bitmaps |

For posting list compression, **PFOR-Delta is the de-facto standard** in modern search systems.

## Skip Pointers

When intersecting two posting lists `L1` (short) and `L2` (long), a naïve merge scans both linearly: O(|L1| + |L2|).

With **skip pointers** in `L2`, you can leap forward instead of stepping one entry at a time:

```
L2 = [3, 8, 15, 22, 30, 45, 50, 67, 80, 92, 100, ...]
                  ^skip→ 50         ^skip→ 100
```

When matching `L1[i] = 47`, instead of scanning `L2` from `22`, jump to `50` (overshoots), then back-scan from `45`.

### Skip Pointer Structure

Two-level skips (Lucene-style):
- Every `√|L|` entries, store a skip pointer with `(docID_target, file_offset)`.
- For very long lists, multi-level skips (skip-of-skips) like a skip list.

Inside a list, the skip data is interleaved with postings or stored in a parallel stream.

### Why Skip Pointers Matter

For a 2-term AND query where `L1` has 100 entries and `L2` has 10M entries:
- Without skips: 10M operations
- With √-skips on `L2`: ~100 × √10M ≈ 316K operations
- 30× speedup

Most short-AND-long queries are this shape (one rare term + one common term).

## Building an Index

### In-Memory Inverted Index

Simplest. Buffer postings in memory, dump to disk at end.

```python
from collections import defaultdict

class InMemoryIndex:
    def __init__(self):
        self.postings: dict[str, list[tuple[int, int, list[int]]]] = defaultdict(list)
        self.doc_count = 0

    def add_doc(self, doc_id: int, text: str) -> None:
        tokens = text.lower().split()
        term_positions: dict[str, list[int]] = defaultdict(list)
        for i, tok in enumerate(tokens):
            term_positions[tok].append(i)
        for term, positions in term_positions.items():
            self.postings[term].append((doc_id, len(positions), positions))
        self.doc_count += 1
```

Works fine up to a few hundred MB of text; then RAM runs out.

### Sort-Based Disk Build (SPIMI, BSBI)

For large corpora, the classic single-pass-in-memory-indexing (**SPIMI**) algorithm:

```
1. For each document:
     emit (term, docID, tf, positions) tuples
     buffer in memory until heap fills
2. When buffer is full:
     sort by (term, docID)
     flush to a "run" file on disk
3. After all docs processed:
     merge all run files using k-way merge
     result is the final sorted posting stream
4. Build term dictionary index over the sorted stream
```

This produces the index in O(corpus_size / RAM) passes, where each pass is sequential I/O — the model that Lucene uses internally for segment writes.

### Lucene Model (Segmented)

Lucene never has a single "the index" file — it has many **segments**, each a complete mini-inverted-index. New documents go into a small in-memory segment, periodically flushed to disk. A background merge process combines small segments into larger ones.

```
RAM:    [active segment, ~100 docs]
DISK:   [seg_001 (1K docs)]
        [seg_002 (1K docs)]
        [seg_003 (10K docs)]
        [seg_004 (100K docs)]
        ...
```

This is essentially an [LSM-tree](advanced_trees.md) structure for inverted indexes. The merge policy (when to compact, how much to combine) is a major performance lever.

## Query Processing

### Boolean AND (Intersection)

```python
def intersect(L1: list[int], L2: list[int]) -> list[int]:
    """Both lists sorted ascending."""
    i = j = 0
    result = []
    while i < len(L1) and j < len(L2):
        if L1[i] == L2[j]:
            result.append(L1[i])
            i += 1; j += 1
        elif L1[i] < L2[j]:
            i += 1
        else:
            j += 1
    return result
```

With skip pointers, `L2[j]` advancement becomes a skip-then-step.

For multi-term AND, intersect shortest-first to minimize intermediate result size:

```python
def intersect_many(lists: list[list[int]]) -> list[int]:
    lists.sort(key=len)
    result = lists[0]
    for L in lists[1:]:
        result = intersect(result, L)
        if not result:
            return []
    return result
```

### Boolean OR (Union)

Merge-style scan, output each docID at most once:

```python
def union(L1: list[int], L2: list[int]) -> list[int]:
    i = j = 0
    result = []
    while i < len(L1) and j < len(L2):
        if L1[i] == L2[j]:
            result.append(L1[i]); i += 1; j += 1
        elif L1[i] < L2[j]:
            result.append(L1[i]); i += 1
        else:
            result.append(L2[j]); j += 1
    result.extend(L1[i:]); result.extend(L2[j:])
    return result
```

For N-way OR, use a min-heap over the list heads.

### Boolean NOT

`L1 AND NOT L2`: scan `L1`, skip any entry that appears in `L2`.

Pure NOT (`NOT L2`) cannot be evaluated against the inverted index directly — you'd have to enumerate every docID not in `L2`. Implementations require at least one positive term in the query.

### Phrase Queries

For `"new york"`, find docs where positions of `new` and `york` are consecutive.

```
1. Intersect docID lists for "new" and "york" → candidate docs
2. For each candidate doc:
     get positions of "new"  → P1
     get positions of "york" → P2
     for each p in P1:
         if (p + 1) in P2: phrase match!
```

This is why positional postings are necessary for phrase / proximity queries. Without positions, you must scan the document text after retrieval.

### Wildcard / Prefix Queries

`car*` matches `car, cars, carbon, ...`. Use the term dictionary's prefix-search capability (FST or trie) to enumerate matching terms, then OR their posting lists.

For arbitrary wildcards (`*car*`), you need an additional structure like a **permuterm index** or n-gram index.

## Top-K Retrieval

For ranked search (BM25, TF-IDF), you want top-K docs by score, not all matches. With millions of matches but K = 10, exhaustive scoring is wasteful.

### Naïve

```python
def topk_naive(query_terms, K):
    scores = defaultdict(float)
    for t in query_terms:
        idf = log(N / df(t))
        for doc, tf in posting_list(t):
            scores[doc] += bm25(tf, idf, doc_len[doc])
    return heapq.nlargest(K, scores.items(), key=lambda x: x[1])
```

Computes scores for every matching document. For common-term OR queries, this is expensive.

### WAND (Weak AND, Broder et al., 2003)

For each term, precompute `max_score(t)` — the maximum BM25 contribution any document could get from `t`.

At query time, maintain `threshold = K-th highest score so far`. For a candidate doc to potentially make the top-K, the sum of `max_score(t)` over terms it contains must exceed `threshold`.

```
Sort terms by their next docID.
Compute pivot doc such that sum of max_scores of leading terms ≥ threshold.
If pivot doc has the leading term in it, score it; otherwise skip leading lists forward.
```

WAND skips large chunks of doc IDs that *cannot* make the top-K. In practice, 10-100× speedup on typical web search queries.

### BlockMaxWAND (Ding & Suel, 2011)

Refine WAND by tracking `max_score` per *block* (not per term). When a block's max score sum can't beat the threshold, skip the whole block.

**Used by:** Lucene 8+ (default ranking query implementation), Tantivy.

### MaxScore

Alternative top-K algorithm with slightly different early-termination strategy. Sometimes faster than WAND on disjunctive queries with many terms.

## Segmented Indexes and Updates

The fundamental tradeoff: inverted indexes are heavily optimized for read, lightly for write. Updates require either:
- **In-place updates** (rare): hard because deletions break sorted-list invariants and skip pointers.
- **Tombstones + immutable segments** (common): mark deleted docs, write new docs to a new segment. Merge eventually compacts.

### Lucene Segment Lifecycle

```
1. New docs go into a memory segment.
2. When the in-mem segment is full, flush to disk as a new segment file.
3. A delete marks the deleted docs in a "liveDocs" bitmap (per segment).
4. A periodic merger picks segments to combine (by size tier or time).
   - Reads all postings from the chosen segments.
   - Skips entries whose docs are tombstoned.
   - Writes a new merged segment.
   - Atomically swaps in the new segment, deletes the old ones.
```

Query time: iterate all segments, intersect/union their results, filter tombstones, merge top-K.

### Merge Policy Choices

- **TieredMergePolicy** (Lucene default): combine segments of roughly equal size, limit segment count.
- **Log-structured merge**: aggressive merging of small segments, fewer total segments → faster queries but more write amplification.
- **No-merge / append-only**: writes go to immutable segments, never merged. Query latency degrades with segment count.

This is essentially the same problem space as **[LSM trees](advanced_trees.md)** — the strategies translate directly.

## Implementation

A simple end-to-end inverted index in Python with VarByte compression, skip pointers, and BM25 scoring.

```python
import math
import heapq
from collections import defaultdict


# ------------------------------------------------------------------
# VarByte encoding
# ------------------------------------------------------------------
def vbyte_encode(n: int) -> bytes:
    out = bytearray()
    while True:
        if n < 128:
            out.append(n | 0x80)  # high bit = "last byte"
            return bytes(out)
        out.append(n & 0x7F)
        n >>= 7


def vbyte_decode(buf: bytes, pos: int) -> tuple[int, int]:
    n = 0
    shift = 0
    while True:
        b = buf[pos]; pos += 1
        if b & 0x80:
            n |= (b & 0x7F) << shift
            return n, pos
        n |= b << shift
        shift += 7


def encode_postings(doc_ids: list[int]) -> bytes:
    """Delta + VarByte."""
    out = bytearray()
    prev = 0
    for d in doc_ids:
        out += vbyte_encode(d - prev)
        prev = d
    return bytes(out)


def decode_postings(buf: bytes) -> list[int]:
    out = []
    pos = 0
    prev = 0
    while pos < len(buf):
        gap, pos = vbyte_decode(buf, pos)
        prev += gap
        out.append(prev)
    return out


# ------------------------------------------------------------------
# Index builder
# ------------------------------------------------------------------
class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, list[tuple[int, int, list[int]]]] = defaultdict(list)
        self.doc_len: dict[int, int] = {}
        self.doc_count = 0
        self.term_doc_freq: dict[str, int] = defaultdict(int)
        self._doc_terms_seen: dict[int, set[str]] = defaultdict(set)

    def add_doc(self, doc_id: int, text: str) -> None:
        tokens = text.lower().split()
        self.doc_len[doc_id] = len(tokens)
        self.doc_count += 1
        term_positions: dict[str, list[int]] = defaultdict(list)
        for i, tok in enumerate(tokens):
            term_positions[tok].append(i)
        for term, positions in term_positions.items():
            self.postings[term].append((doc_id, len(positions), positions))
            self.term_doc_freq[term] += 1

    # ------------------------------------------------------------------
    # Boolean queries
    # ------------------------------------------------------------------
    def intersect(self, terms: list[str]) -> list[int]:
        if not terms or any(t not in self.postings for t in terms):
            return []
        lists = sorted(
            ([p[0] for p in self.postings[t]] for t in terms),
            key=len,
        )
        result = lists[0]
        for L in lists[1:]:
            result = self._merge_and(result, L)
            if not result:
                return []
        return result

    @staticmethod
    def _merge_and(a: list[int], b: list[int]) -> list[int]:
        i = j = 0
        out = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i]); i += 1; j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    def phrase(self, terms: list[str]) -> list[int]:
        if len(terms) < 2:
            return self.intersect(terms)
        candidates = self.intersect(terms)
        per_term_pos = {
            t: {d: p for d, _, p in self.postings[t] if d in set(candidates)}
            for t in terms
        }
        out = []
        for d in candidates:
            positions_first = per_term_pos[terms[0]].get(d, [])
            ok = False
            for p in positions_first:
                if all(
                    (p + i) in per_term_pos[terms[i]].get(d, set())
                    for i in range(1, len(terms))
                ):
                    ok = True
                    break
            if ok:
                out.append(d)
        return out

    # ------------------------------------------------------------------
    # BM25 top-K
    # ------------------------------------------------------------------
    def bm25_topk(self, query: list[str], k: int = 10, k1: float = 1.2, b: float = 0.75) -> list[tuple[int, float]]:
        avgdl = sum(self.doc_len.values()) / max(self.doc_count, 1)
        N = self.doc_count
        scores: dict[int, float] = defaultdict(float)
        for term in query:
            if term not in self.postings:
                continue
            df = self.term_doc_freq[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf, _ in self.postings[term]:
                dl = self.doc_len[doc_id]
                norm = (1 - b) + b * dl / avgdl
                scores[doc_id] += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])
```

### Usage

```python
idx = InvertedIndex()
idx.add_doc(1, "the quick brown fox jumps over the lazy dog")
idx.add_doc(2, "the brown fox is quick")
idx.add_doc(3, "lazy dogs sleep all day")

print(idx.intersect(["brown", "fox"]))     # [1, 2]
print(idx.phrase(["brown", "fox"]))         # [1, 2]
print(idx.bm25_topk(["fox", "lazy"], k=3))  # ranked
```

For production use Tantivy (Rust, fast, easy Python bindings via `tantivy-py`) or Lucene through Elasticsearch / OpenSearch.

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Indexing one doc | O(doc length) | O(doc length) |
| Bulk indexing N docs | O(corpus size) sequential I/O | O(index size) |
| Term lookup | O(\|term\|) FST / O(log V) sorted | — |
| Single-term query | O(\|posting list\|) | — |
| AND of k terms (skip pointers) | O(min list size · √max list size · k) | — |
| BM25 top-K (naïve) | O(Σ\|posting lists\|) | O(matching docs) |
| BM25 top-K (BlockMaxWAND) | O(K · log(matching) + skipped blocks) | O(K) |

**Index size:** typically 10-30% of original text for `docID + tf` only, 50-100% with positions. Compression and merging close the gap; raw text rarely re-stored (stored separately, or omitted entirely if only search is needed).

## Distributed Inverted Index

Single-machine indexes work to ~1B docs. Beyond that, shard.

### Sharding Strategies

**Document partitioning** (most common):
- Each shard holds a subset of documents.
- Each shard has a complete inverted index over its docs.
- Query is **scatter-gather**: send query to all shards, merge top-K from each.

Pros: balanced load, simple.
Cons: every query hits every shard.

**Term partitioning**:
- Each shard owns a subset of terms (e.g., by hash).
- Query for term t goes only to its shard.

Pros: fewer shards touched per query (single-term: 1 shard).
Cons: multi-term queries become network-bound; load imbalance from common terms.

Document partitioning is the universal choice (Elasticsearch shards, Solr cores, Google's index).

### Replication

Each shard replicated for HA and read throughput. Reads scatter-gather across one replica per shard.

### Coordination

A query coordinator:
1. Sends the query to all shards (1 replica each).
2. Collects top-K-per-shard.
3. Merges into global top-K.
4. Fetches stored fields for the final K docs from the appropriate shards.

This two-phase pattern (top-K-per-shard, then fetch) is universal.

## Real-World Systems

| System | Language | Notes |
|---|---|---|
| **Lucene** | Java | The reference inverted index. Powers Elasticsearch, OpenSearch, Solr, MongoDB Atlas Search. FST term dictionary, PFOR-Delta postings, BlockMaxWAND ranking. |
| **Tantivy** | Rust | Lucene-inspired, very fast. Used by Meilisearch, Quickwit. Python bindings via `tantivy-py`. |
| **Vespa** | Java/C++ | Yahoo. Integrated inverted + dense vector + structured filtering. Big-data scale. |
| **Bleve** | Go | Simpler Lucene-alike for Go services. |
| **Whoosh** | Python | Pure Python, simple, slow. Good for learning / small indexes. |
| **Sphinx / Manticore** | C++ | Older but still used for fast text search over DB content. |
| **Xapian** | C++ | Probabilistic ranking, used in some library catalog systems. |
| **PostgreSQL GIN** | C | Generalized inverted index. Stores `(token, [tids])`. Used for `tsvector` full-text and JSONB. |
| **MongoDB text index** | C++ | Per-collection inverted index over string fields. |
| **MeiliSearch** | Rust | Typo-tolerant, instant search; bitmap-based posting representation. |
| **Quickwit** | Rust | Tantivy-based, object-storage native (S3 segments). |
| **Algolia** | proprietary | Custom inverted index optimized for sub-50ms search-as-you-type. |

## When to Use

### Use an inverted index when:
- Keyword / text search is the primary access pattern
- Documents have natural tokenization (words, code identifiers, log fields)
- You need boolean / phrase / proximity queries
- Top-K relevance ranking matters (BM25, custom scoring)
- Hybrid retrieval: combine with dense ([[hnsw]]) for "best of both"

### Don't use an inverted index for:
- Pure semantic similarity (use [[hnsw]] / [[product-quantization]])
- Substring search on non-tokenizable data (use [[suffix-arrays]] or n-gram index)
- Range / numeric / point lookups (use B-trees, range structures, columnar stores)
- Fuzzy matching where edit distance is the primary signal (use BK-trees or Levenshtein automata over the term dictionary)

### Hybrid Retrieval (Modern Best Practice)

For RAG and modern search:
- Run inverted-index BM25 (keyword recall, exact match guarantees)
- Run HNSW dense ANN (semantic recall)
- Fuse with RRF (Reciprocal Rank Fusion) or learned ranker

The two signals are complementary: BM25 catches exact-term matches (proper nouns, code identifiers, rare terms) that embeddings often miss; dense catches paraphrases and semantic equivalents that BM25 misses.

## Common Pitfalls

1. **Stopwords removed too aggressively**
   - Phrase queries like `"to be or not to be"` break if all terms are stopwords.
   - Modern indexes keep them (they compress well anyway).

2. **Tokenizer / analyzer mismatch between index and query**
   - Indexer lowercases, query doesn't (or vice versa) → no matches.
   - Always normalize identically on both sides.

3. **Positions disabled, then a phrase query is issued**
   - Phrase queries silently return wrong results (just intersect doc lists).
   - Check the field's `index_options` in Lucene.

4. **Too many small segments**
   - Each query touches every segment. Latency = ~constant × segment count.
   - Tune merge policy; force-merge after bulk loads.

5. **Skip pointers misimplemented for compressed lists**
   - Skip targets must be at compression-block boundaries, not arbitrary offsets.

6. **BM25 IDF undefined when df = 0**
   - Standard BM25 formula has `log((N - df + 0.5)/(df + 0.5) + 1)` — the `+1` ensures non-negativity.
   - Naïve `log(N/df)` blows up at `df = 0`.

7. **Treating term frequency as a length proxy**
   - BM25's `b` parameter normalizes by doc length. Without it, long docs always rank higher.

8. **Forgetting tombstones on read**
   - Direct posting-list scans bypass the liveDocs bitmap → return deleted docs.
   - Always filter through liveDocs.

9. **Shard count fixed at index creation**
   - Resharding requires reindexing in most systems (Lucene, Elasticsearch).
   - Plan shard count for expected growth.

10. **No analysis testing**
    - Tokenization bugs (stemming, asciifolding, edge n-grams) silently produce wrong relevance.
    - Always use the index's `_analyze` API on representative text.

## Related Topics

- [[hnsw]] — dense vector ANN; hybridized with inverted index for modern search
- [[suffix-arrays]] — substring search when terms aren't word-bounded
- [[tries]] — alternative to FST for in-memory term dictionaries
- [[advanced-trees]] — LSM-tree pattern underlies Lucene segment management
- [[probabilistic]] — Bloom filters used in inverted indexes for negative lookups
- [[bloom-filter]] — block-level skip optimization
- [[minhash-lsh]] — set/near-dup search at scale (complementary to inverted index for dedup)

External:
- `algorithms/string_algorithms.md` — tokenization, stemming, and text-processing algorithms
- `ai/rag.md` — retrieval-augmented generation uses inverted indexes for sparse retrieval

## Summary

The inverted index is the oldest and most-deployed information-retrieval data structure. Its modern incarnation in Lucene combines a compact term dictionary (FST), heavily compressed posting lists (PFOR-Delta), skip pointers for fast intersection, immutable segments for write-friendly updates, and BlockMaxWAND for sub-linear top-K retrieval. These pieces evolved over 30 years and together form the foundation of nearly every search system in production.

In a world increasingly excited about dense retrieval, the inverted index remains essential: it offers exactness, recall on rare terms, sub-millisecond filtering, and tiny memory footprints per query. Hybrid retrieval — inverted-index BM25 fused with HNSW dense ANN — is the consensus best practice for 2026 search systems, and that combination only works because both data structures play to their distinct strengths.
