# Prompt Caching

## Overview

A long, stable prefix — a big system prompt, tool definitions, a document, few-shot examples —
gets reprocessed on every API call, wasting compute and money. **Prompt caching** stores the
model's intermediate state (the [KV cache](inference_optimization.md)) for a prefix so repeated
calls skip recomputing it, cutting cost and latency dramatically. It's the API-level surface of
the prefix-sharing made possible by [PagedAttention](inference_optimization.md), a key cost lever
tracked in [LLM observability](llm_observability.md), and especially impactful for
[agents](agent_frameworks.md) and [coding agents](cli.md) that resend a large context every turn.

```
  Without caching: [─── 20k-token system prompt ───][query] ← reprocessed every call
  With caching:    [════ cached prefix (cheap) ════][query] ← only the new part is full price
```

## How it works

Caching exploits the fact that the transformer's work on a prefix depends only on that prefix.
Compute the [KV cache](inference_optimization.md) for the stable part once, store it keyed by the
exact token sequence, and on a later request with the same prefix, **load** it instead of
recomputing.

```
  request 1: compute KV for prefix P  ──► store under hash(P)
  request 2: prefix P matches  ──► load KV (a cache HIT) ──► only process the suffix
```

Two consequences define how you use it:
1. The match is on an **exact token prefix** from the start — one different token early
   invalidates everything after it.
2. Caches are **ephemeral** (a short TTL, e.g. ~5 min, refreshed on hit) — built for bursts of
   related calls, not long-term storage.

## Provider models

- **Anthropic** — explicit: you mark up to 4 `cache_control` breakpoints in the prompt. Cache
  *writes* cost a bit more than normal tokens; cache *reads* are ~10% of the normal price.
- **OpenAI** — automatic: prefixes over ~1024 tokens are cached transparently with a discount on
  cached input tokens; no markup, no control.
- **Gemini** — both implicit caching and explicit *context caching* with a chosen TTL.
- **Local / [vLLM](vllm.md)** — **automatic prefix caching (APC)** reuses KV blocks across
  requests sharing a prefix, via [PagedAttention](inference_optimization.md).

## Structuring prompts to cache

The rule: **stable content first, variable content last.**

```
  ┌──────────── cacheable prefix (put first) ────────────┐ ┌── variable (last) ──┐
  │ system prompt │ tool defs │ few-shot │ big document   │ │ user query / turn   │
  └───────────────────────────────────────────────────────┘ └─────────────────────┘
                    ▲ cache breakpoint here
```

- Never put a timestamp, request ID, or user name *before* the cacheable bulk — it busts the
  cache every call.
- In an agent loop, keep the system prompt + tools + history prefix identical so each turn hits
  the cache and only the latest message is full price.

## Where this connects

- [Inference optimization](inference_optimization.md) — caching reuses the KV cache; APC and
  PagedAttention are the mechanism.
- [LLM observability](llm_observability.md) — track **cache hit rate** and cached-token cost.
- [Agent memory](agent_memory.md) — stable, well-ordered context maximizes cache hits.
- [Claude Code CLI](cli.md) — resends large repo/context each turn; caching is what makes that
  affordable.
- [vLLM](vllm.md) — automatic prefix caching for self-hosted serving.

## Pitfalls

- **Variable data in the prefix.** A timestamp or per-request token near the top invalidates the
  whole cache — keep volatile content at the end.
- **Expecting persistence.** Caches expire in minutes; they help bursts, not a query you run once
  an hour.
- **Ignoring write cost (Anthropic).** Caching a prefix used only once can cost *more* than not
  caching; it pays off across repeated reads.
- **Prefix drift.** Reordering tools or editing the system prompt between calls silently drops
  the hit rate — watch it in [observability](llm_observability.md).
- **Assuming cross-provider parity.** Explicit vs automatic, TTLs, and pricing differ; don't port
  assumptions between APIs.
