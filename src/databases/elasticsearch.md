# Elasticsearch

## Overview

Elasticsearch is a distributed search and analytics engine built on Apache Lucene. Where a
[relational database](postgres.md) is optimized for storing and transacting on exact rows,
Elasticsearch is optimized for **full-text search** ("find documents relevant to these
words, ranked by relevance") and **aggregations over huge volumes** (logs, metrics, events).
Its core is the [inverted index](../data_structures/inverted_index.md); it's the search/log
backbone of the ELK stack (see [logging](../devops/logging.md)). It's a [NoSQL](nosql.md)
document store with search as the headline feature.

```
Use Elasticsearch for:
  ✓ full-text search (relevance-ranked), fuzzy/typo-tolerant, autocomplete
  ✓ log & event analytics at scale (ELK / Elastic Stack)
  ✓ faceted search, aggregations, dashboards (Kibana)
Don't use it as:
  ✗ your primary system of record (no real transactions; near-real-time, not ACID)
  ✗ a relational store needing joins & strong consistency
```

## The inverted index — why search is fast

A normal index maps row → values. An **inverted index** maps each *term* → the list of
documents containing it, so a search is a fast lookup, not a scan.

```
Documents:  1:"the quick fox"   2:"quick brown dog"   3:"lazy fox"

Inverted index (term → postings):
  quick → [1, 2]      fox → [1, 3]      brown → [2]      lazy → [3]

Query "quick fox" → intersect postings(quick)∩postings(fox) = [1] → instant.
```

Text is **analyzed** at index time into terms, which is what makes search smart:

```
Analysis pipeline:  raw text → tokenizer → filters → terms
  "The Quick Foxes!" → [the, quick, foxes] → lowercase → stopword-removal(the)
                     → stemming(foxes→fox) → [quick, fox]
Query text is analyzed the SAME way → "FOX" matches "foxes". Mismatched analyzers at
index vs query time is the #1 "why doesn't my search match?" bug.
```

See [inverted index](../data_structures/inverted_index.md) for the data structure in depth.

## Relevance scoring (BM25)

Elasticsearch doesn't just match — it **ranks** by relevance using BM25 (an improved TF-IDF):

```
A term boosts a document's score when it is:
  - frequent IN the document        (term frequency, TF) — with diminishing returns
  - rare ACROSS the corpus          (inverse document frequency, IDF)
  - in a shorter field              (length normalization)
⇒ rare, on-topic words rank documents higher than common ones.
You can tune with boosts, function scoring, and (increasingly) vector/semantic search.
```

## Data model & distribution

```
Index     ~ a "table"/collection of JSON documents (schemaless-ish; has a mapping)
Document  a JSON object (the unit of indexing & retrieval)
Mapping   field types & analyzers (text vs keyword matters — see below)
Shard     an index is split into shards (each a Lucene index) → horizontal scale
Replica   copies of shards → availability + read throughput
Node/Cluster  shards spread across nodes; cluster rebalances automatically
```

```
text  vs  keyword  (the classic mapping gotcha):
  text     → analyzed → full-text search, NOT good for exact match/sort/aggregate
  keyword  → stored verbatim → exact match, sorting, aggregations, term filters
  Often index a field BOTH ways (a multi-field): name (text) + name.keyword (keyword).
```

## Querying

```
Query DSL (JSON):
  match        full-text, analyzed, relevance-scored  ("search box")
  term         exact, non-analyzed (use on keyword fields)
  bool         combine must / should / must_not / filter
  filter ctx   yes/no, no scoring → CACHEABLE & faster (use for ranges, exact filters)
  query ctx    "how well does it match?" → scored
Aggregations:  terms, histogram, date_histogram, metrics (avg/sum/percentiles),
               nested aggs → the engine behind Kibana dashboards & faceted search.
```

## Consistency & durability model

Elasticsearch trades strict consistency for search scale — know the model:

```
Near-real-time: indexed docs become searchable after a REFRESH (default ~1s), not instantly.
Durability: a translog (write-ahead log) per shard protects un-flushed writes.
Not a transactional store: no multi-document ACID transactions, no joins (denormalize, or
  use nested/parent-child with caveats). Often used ALONGSIDE a system-of-record DB,
  fed via CDC. See cdc_streaming.md.
```

## The Elastic Stack (ELK)

```
Elasticsearch  store + search + aggregate
Logstash / Beats / Fluentd   ingest & transform logs/metrics/events
Kibana         visualize, dashboard, explore
Common use: centralized logging & observability. See ../devops/logging.md and
            ../devops/observability.md.
```

## Operations

```
Sharding: too many small shards = overhead; too few = can't scale/rebalance. Size shards
  ~10–50 GB; you can't change a shard count without reindexing → plan or use rollover.
Index lifecycle management (ILM): hot/warm/cold/delete tiers for time-series log data.
Reindex API: to change mappings/analyzers (mappings are mostly immutable once set).
Watch heap/JVM, mapping explosions (too many fields), and unbounded aggregations.
```

## Alternatives

```
OpenSearch     AWS's open-source fork of Elasticsearch (post-license-change).
Apache Solr    the other Lucene-based search server.
pg full-text / Meilisearch / Typesense  lighter options for smaller search needs.
Vector DBs     for semantic/embedding search — see ../ai/vector_databases.md (ES also does
               kNN vector search now).
```

## Where this connects

- **[Inverted index](../data_structures/inverted_index.md)** — the core data structure.
- **[NoSQL](nosql.md)** — document-store family context.
- **[Logging](../devops/logging.md)** / **[Observability](../devops/observability.md)** —
  the ELK use case.
- **[CDC & streaming](cdc_streaming.md)** — keeping ES in sync with a source-of-truth DB.
- **[Vector databases](../ai/vector_databases.md)** — semantic search complement.

## Pitfalls

- **Using it as the system of record** — near-real-time, no transactions; pair it with a DB.
- **text vs keyword confusion** — exact-match/sort/aggregate on a `text` field fails or misbehaves.
- **Analyzer mismatch** — different analysis at index vs query time = silent no-matches.
- **Shard sizing** — too many tiny shards is a top cause of cluster instability.
- **Unbounded queries/aggregations** — deep pagination and huge aggs blow up heap.
