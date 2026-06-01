# Embeddings & Reranking

## Overview

The numerical backbone of semantic search and [RAG](rag.md): turning text (or images, code)
into vectors whose geometry encodes meaning, then ranking results by similarity. This page is
the "how the vectors are made and ordered" layer beneath [vector databases](vector_databases.md)
(which store and search them) and [chunking strategies](chunking_strategies.md) (which decide
what gets embedded). The two-stage **retrieve-then-rerank** pattern here is what lifts RAG and
[GraphRAG](graphrag.md) quality. Embedding models are themselves [transformers](transformers_architecture.md),
usually small encoder models.

```
  text ──► embedding model ──► vector ──► similarity search ──► candidates
                                                                    │
                                              reranker (cross-encoder)
                                                                    │
                                                              top results
```

## Bi-encoders vs cross-encoders

The key architectural split, and the reason retrieval has two stages:

```
  BI-ENCODER (fast, for retrieval)        CROSS-ENCODER (accurate, for reranking)
  query ─► [encoder] ─► vec_q             [query + doc] ─► [encoder] ─► score
  doc   ─► [encoder] ─► vec_d             one joint pass per pair
  score = cos(vec_q, vec_d)              no precompute, can't index
  precompute all doc vecs once →          O(candidates) passes per query →
  scales to billions                      only feasible on a shortlist
```

A **bi-encoder** embeds query and document *separately*, so document vectors are precomputed
and indexed — fast over millions of docs but slightly blurry (no query-document interaction). A
**cross-encoder** feeds the pair *together* and reads the attention between them, far more
accurate but can't be precomputed. So: bi-encoder retrieves top-100, cross-encoder **reranks**
to top-5.

## Similarity metrics

```
  cosine        — angle between vectors; ignores magnitude (most common for text)
  dot product   — cosine × magnitudes; used when vectors are NOT normalized
  euclidean (L2)— straight-line distance; equivalent to cosine for unit vectors
```

Most embedding models are trained for cosine and emit normalized vectors, where cosine and dot
product coincide. Match your [vector DB's](vector_databases.md) metric to the model's training.

## Choosing an embedding model

**MTEB** (Massive Text Embedding Benchmark) is the standard leaderboard — covers retrieval,
clustering, classification, reranking. Selection axes:

| Axis | Trade-off |
|------|-----------|
| Dimensions (384 → 4096) | bigger = more nuance but more storage/compute |
| Context length | long-doc models (8k+) vs sentence models (512) |
| Domain | general vs code/legal/biomedical specialists |
| Multilingual | one space across languages vs English-only |
| Open vs API | self-host (e5, BGE, GTE, Nomic) vs hosted (OpenAI, Cohere, Voyage) |

**Matryoshka** embeddings are trained so a *prefix* of the vector is still useful — store 1024
dims, truncate to 256 for cheap first-pass search, rerank with the full vector.

## Reranking

After cheap vector retrieval, a reranker reorders the shortlist. Options:

- **Cross-encoder rerankers** — Cohere Rerank, BGE-reranker, Jina; highest quality.
- **Hybrid fusion** — combine dense (vector) and sparse (keyword/BM25) results with
  **Reciprocal Rank Fusion (RRF)** before/after reranking. Catches exact terms that dense
  search misses (names, IDs, codes).

```
  RRF: score(d) = Σ 1 / (k + rank_i(d))   across each ranker i
  blends lexical + semantic without tuning weights
```

## Where this connects

- [Vector databases](vector_databases.md) — store embeddings and do the ANN retrieval step.
- [RAG](rag.md) — retrieve-then-rerank is the retrieval half of RAG quality.
- [Chunking strategies](chunking_strategies.md) — what you embed determines what you can
  retrieve.
- [GraphRAG](graphrag.md) — embeddings still power entity/community matching there.
- [Transformers architecture](transformers_architecture.md) — encoder models produce the
  vectors.

## Pitfalls

- **Skipping the reranker.** Pure bi-encoder retrieval leaves a lot of precision on the table;
  a cross-encoder rerank is usually the cheapest big quality win.
- **Metric mismatch.** Using dot product on un-normalized vectors (or cosine on vectors trained
  for dot) silently degrades ranking.
- **Embedding the wrong granularity.** Whole documents blur meaning; tiny fragments lose
  context — see [chunking strategies](chunking_strategies.md).
- **No keyword fallback.** Dense search misses exact identifiers, SKUs, and rare names; add
  sparse/hybrid retrieval.
- **Changing models without re-embedding.** Vectors from different models aren't comparable;
  swapping the embedder means rebuilding the whole index.
