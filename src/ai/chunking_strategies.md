# Chunking & Ingestion

## Overview

Before [RAG](rag.md) can retrieve anything, raw documents must be parsed, split, and turned into
[embeddings](embeddings.md) stored in a [vector database](vector_databases.md). **Chunking** —
deciding how to cut documents into retrievable units — is the most underrated lever on RAG
quality: retrieval can only return chunks you created, so a bad split caps the ceiling no
[reranker](embeddings.md) can raise. This page covers the ingestion pipeline and the chunking
choices within it; it also feeds [GraphRAG](graphrag.md) extraction.

```
  source files ─► parse/extract ─► clean ─► chunk ─► embed ─► index
  (PDF, HTML,      (text + layout)  (dedupe)  (split)  (vectors) (vector DB)
   docx, code)
```

## The tension

```
  Chunks too BIG                      Chunks too SMALL
  ──────────────                      ────────────────
  retrieval pulls irrelevant text     each chunk lacks context
  embeddings blur many topics         meaning is fragmented
  wastes context window               need many chunks to answer
```

The goal is one **coherent idea per chunk** — big enough to stand alone, small enough to be
about one thing.

## Chunking strategies

| Strategy | How | Best for |
|----------|-----|----------|
| **Fixed-size** | N tokens with M overlap | quick baseline, uniform text |
| **Recursive** | split on ¶ → sentence → word until under size | general prose (good default) |
| **Document/structure-aware** | split on Markdown headings, code functions, HTML sections | structured docs, code |
| **Semantic** | embed sentences, cut where similarity drops | topic-shifting documents |
| **Late chunking** | embed the whole doc, then pool per-chunk | preserves long-range context |

**Overlap** (e.g. 10–20%) carries context across boundaries so a sentence split mid-idea still
has its neighbors:

```
  [.... chunk 1 ....]
              [.... chunk 2 ....]   ← overlap region repeated in both
```

## Parsing matters more than the splitter

Garbage in, garbage out. PDFs are the classic trap — multi-column layouts, tables, and headers
get mangled by naïve text extraction. Layout-aware parsers (**unstructured**, **LlamaParse**,
**Docling**, vision models for scanned docs) preserve reading order and tables. For code, parse
by **AST** so functions/classes stay intact.

## Enrichment patterns

Beyond raw splitting, these consistently improve retrieval:

- **Metadata** — attach source, section, date, author to each chunk; enables filtered search
  ("only 2024 docs") in the [vector DB](vector_databases.md).
- **Contextual retrieval** — prepend an LLM-generated summary of *where this chunk sits* in the
  document before embedding (Anthropic's technique; big recall gains).
- **Parent-document / small-to-big** — embed small chunks for precise matching, but return the
  larger **parent** chunk to the LLM for context.
- **Hypothetical questions / summaries** — index a generated question or summary that points
  back to the chunk, matching how users actually phrase queries.

## Where this connects

- [RAG](rag.md) — chunking is the ingestion half; it bounds retrieval quality.
- [Embeddings](embeddings.md) — each chunk becomes a vector; granularity must suit the model.
- [Vector databases](vector_databases.md) — store chunks + metadata for filtered search.
- [GraphRAG](graphrag.md) — entity extraction runs per chunk; chunk quality drives graph
  quality.

## Pitfalls

- **Tuning the splitter before fixing the parser.** Mangled PDF/table extraction dwarfs any
  chunk-size tuning.
- **No overlap.** Hard cuts strand ideas across boundaries; a little overlap is cheap insurance.
- **One strategy for all content.** Code, prose, and tables need different splits; route by
  type.
- **Forgetting metadata.** Without it you can't filter or cite sources, and you can't expire
  stale content.
- **Re-chunking without re-embedding.** Changing chunk boundaries invalidates existing vectors;
  reindex.
