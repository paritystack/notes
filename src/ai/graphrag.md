# GraphRAG

## Overview

Standard [RAG](rag.md) retrieves isolated chunks by [embedding](embeddings.md) similarity — great
for "find the passage that answers X," poor for questions that require *connecting* facts spread
across a corpus ("what are the main themes?", "how is A related to C through B?"). **GraphRAG**
builds a **knowledge graph** of entities and relationships from your documents, then retrieves
over that structure. It complements rather than replaces vector RAG, and still relies on
[chunking](chunking_strategies.md), [embeddings](embeddings.md), and a
[vector database](vector_databases.md) for parts of the pipeline. Popularized by Microsoft's
GraphRAG project.

```
  Vector RAG:  query ──► similar chunks ──► answer        (local, fact lookup)
  GraphRAG:    query ──► entities/relations/communities ──► answer
                              (global, connect-the-dots reasoning)
```

## Building the graph (indexing)

An LLM does **entity and relationship extraction** over each chunk, producing a graph:

```
  chunk: "Ada Lovelace worked with Charles Babbage on the Analytical Engine."
        │  LLM extraction
        ▼
  (Ada Lovelace) ──[collaborated with]──► (Charles Babbage)
  (Ada Lovelace) ──[wrote notes on]─────► (Analytical Engine)
  (Charles Babbage) ──[designed]────────► (Analytical Engine)
```

Steps:
1. **Chunk** documents (see [chunking strategies](chunking_strategies.md)).
2. **Extract** entities (nodes) and relationships (edges) per chunk via an LLM prompt.
3. **Resolve** duplicates ("ML" = "machine learning") into canonical nodes.
4. **Detect communities** — graph-clustering algorithms (e.g. **Leiden**) group densely
   connected entities into hierarchical clusters.
5. **Summarize** each community with an LLM into a "community report."

This indexing is **LLM-heavy and expensive** — every chunk gets at least one extraction call.
That cost is the main trade-off versus plain vector RAG.

## Querying: local vs global

```
  LOCAL search  — "What did Ada Lovelace work on?"
    find the entity → walk its neighborhood → gather connected chunks → answer
    (like vector RAG but graph-expanded)

  GLOBAL search — "What are the major themes across all documents?"
    map over community summaries → partial answers → reduce into final answer
    (impossible for vector RAG — no single chunk holds the theme)
```

Global search is the headline capability: it answers corpus-wide, aggregative questions by
reasoning over the **community summaries** instead of any individual passage.

## When to use it

| Use GraphRAG when… | Stick with vector RAG when… |
|--------------------|------------------------------|
| multi-hop / connect-the-dots questions | direct fact lookup |
| "themes", "summaries", "relationships" | high query volume, low latency |
| corpus has rich entity structure | indexing budget is tight |
| corpus is fairly static | docs change constantly |

A common production pattern is **hybrid**: route fact-lookup queries to vector RAG and
aggregative/relational queries to GraphRAG.

## Where this connects

- [RAG](rag.md) — GraphRAG is a structural extension of it; both inject retrieved context.
- [Embeddings](embeddings.md) — still used for entity matching and node similarity.
- [Vector databases](vector_databases.md) — store node/community embeddings alongside the graph.
- [Chunking strategies](chunking_strategies.md) — extraction quality depends on good chunks.
- [Tool use](tool_use.md) — agents can query the graph as a tool for multi-hop reasoning.

## Pitfalls

- **Indexing cost shock.** Per-chunk LLM extraction over a large corpus can cost far more than
  embedding it; estimate before committing.
- **Bad entity resolution.** Un-merged duplicate entities fragment the graph and break
  multi-hop retrieval.
- **Using it for everything.** For simple fact lookup it's slower and pricier than vector RAG
  with no benefit.
- **Stale graphs.** Frequently changing corpora need re-indexing; GraphRAG suits stable
  knowledge bases.
- **Extraction is lossy.** The LLM may miss or invent relationships; the graph is an
  approximation, not ground truth — validate critical edges.
