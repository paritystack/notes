# Agent Memory & Context Engineering

## Overview

An [agent](agent_frameworks.md)'s only working state is what fits in its context window. **Memory**
is how an agent carries information across turns and sessions despite that finite window, and
**context engineering** is the discipline of deciding what occupies the window at each step.
This deepens [Agentic Context Engineering (ACE)](ace.md) and underpins
[multi-agent systems](multi_agent_systems.md) and [coding agents](coding_agents.md). Long-term
memory is typically backed by [vector databases](vector_databases.md) and
[RAG](rag.md)-style retrieval; managing the window ties into
[inference optimization](inference_optimization.md) and [prompt caching](prompt_caching.md).

```
  ┌─────────────────── context window (finite) ───────────────────┐
  │ system │ tools │ retrieved memory │ recent turns │ scratchpad │
  └────────────────────────────────────────────────────────────────┘
        ▲                    ▲
   long-term store    working memory (this session)
   (vector DB, files)
```

## Types of memory

| Type | Lifetime | Stored where | Example |
|------|----------|--------------|---------|
| **Working / short-term** | this session | the context window | recent messages, tool results |
| **Episodic** | across sessions | vector DB / log | "last week the user preferred X" |
| **Semantic** | long-lived | vector DB / KG | facts, user profile, docs |
| **Procedural** | long-lived | prompt / skills | how to do a task ([skills](skills.md)) |

The analogy: working memory is RAM, long-term memory is disk, and retrieval is the page-in.

## Context engineering — the core problem

The window is a **budget**, not a free buffer. More tokens cost money and latency, and quality
degrades when the window is bloated:

- **"Lost in the middle"** — models attend best to the start and end; facts buried in a long
  middle get ignored.
- **Context rot / distraction** — irrelevant or contradictory context pulls the model
  off-task.
- **Cost & latency** — every token is paid for on every step of an agent loop.

So the job is to put the *right* tokens in, not the *most*.

```
  Bad:  dump everything  ──► huge window, distracted, expensive
  Good: retrieve + compress ──► small, relevant window, sharp + cheap
```

## Techniques

- **Summarization / compaction** — when history grows, replace old turns with an LLM summary.
  ([Claude Code](cli.md) does this when context fills.)
- **Retrieval-augmented memory** — embed past interactions; fetch only the relevant ones per
  turn (semantic/episodic recall via [RAG](rag.md)).
- **Scratchpads / external memory** — let the agent write notes to a file or state object and
  read them back, offloading from the window.
- **Structured state** — keep a compact JSON of goals/progress instead of replaying raw
  transcripts.
- **Memory writing policy** — decide *what's worth remembering* (reflection); naïvely storing
  everything pollutes recall.
- **Eviction** — recency + relevance scoring to drop stale memories, like a cache.

## Frameworks

**LangGraph** (checkpointers, store), **Letta/MemGPT** (OS-style paging of memory in and out of
the window), **Mem0**, and **Zep** provide memory layers. The MemGPT insight: treat the context
window like virtual memory and *page* information between a small fast context and a large slow
store.

## Where this connects

- [ACE](ace.md) — the broader framing of engineering an agent's context.
- [Agent frameworks](agent_frameworks.md) — memory is a core agent component.
- [Vector databases](vector_databases.md) / [RAG](rag.md) — the long-term store and recall path.
- [Prompt caching](prompt_caching.md) — caching the stable prefix makes long context cheaper.
- [Multi-agent systems](multi_agent_systems.md) — shared vs per-agent memory.

## Pitfalls

- **Remembering everything.** Unfiltered memory recall surfaces noise; store and retrieve
  selectively.
- **Summaries that drop the key fact.** Compaction is lossy — pin critical state (IDs,
  decisions) so it survives summarization.
- **Ignoring "lost in the middle."** Put the most important context at the start or end of the
  window, not buried.
- **Unbounded growth.** Without eviction, memory stores and windows balloon in cost and
  latency.
- **Stale memory treated as truth.** Old preferences/facts may be outdated; timestamp and
  revalidate.
