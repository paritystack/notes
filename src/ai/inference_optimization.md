# LLM Inference Optimization

## Overview

Why is generating tokens slow and memory-hungry, and what do serving systems do about it?
This page covers the mechanics — **KV cache**, **continuous batching**, **PagedAttention**,
**speculative decoding**, **FlashAttention**, and **parallelism** — that sit under
[vLLM](vllm.md), TGI, and the [local runners](local_inference.md). It complements
[quantization](../machine_learning/quantization.md) (shrinking weights) and
[prompt caching](prompt_caching.md) (reusing computed prefixes), and explains the latency and
cost numbers you track in [LLM observability](llm_observability.md). The architecture under
all of it is the [transformer](transformers_architecture.md).

```
  Two phases of generation:
  ┌──────────────┐        ┌────────────────────────────┐
  │  PREFILL     │        │  DECODE                     │
  │  process the │  ───►  │  generate tokens one at a   │
  │  whole prompt│        │  time, reusing the KV cache │
  │  (parallel)  │        │  (sequential, memory-bound) │
  └──────────────┘        └────────────────────────────┘
   compute-bound            memory-bandwidth-bound
```

## The KV cache — the central object

Self-attention computes Key and Value vectors for every token. During generation each new
token attends to **all previous** tokens, so re-deriving their K/V each step would be O(n²)
wasted work. Instead they're cached: compute K/V once per token, store it, reuse it.

```
  Without cache: token t recomputes K,V for tokens 0..t   (quadratic)
  With cache:    token t computes its own K,V, reads the rest from cache (linear)
```

The catch is memory. The cache grows linearly with sequence length × layers × heads:

```
  KV cache bytes ≈ 2 × layers × heads × head_dim × seq_len × batch × dtype_bytes
  (the 2 = K and V)
```

For a 7B model at 4k context this is ~1–2 GB; at 128k context it dwarfs the weights. The KV
cache, not the weights, is usually what limits how many concurrent requests you can serve.
Mitigations baked into modern models: **Grouped-Query Attention (GQA)** and **Multi-Query
Attention (MQA)** share K/V across query heads, cutting cache size several-fold.

## Continuous (in-flight) batching

The GPU is wasted on one request at a time, but requests finish at different lengths. **Static
batching** waits for the whole batch to finish (slowest request gates everyone). **Continuous
batching** evicts finished sequences and admits new ones every step, keeping the GPU full.

```
  Static:      [req A ████████████████]  ← short reqs idle, waiting
               [req B ████]·············
  Continuous:  [req A ████████████████]
               [req B ████][req C ██████]  ← slot reused immediately
```

This is the single biggest throughput win for a serving system and the core feature of
[vLLM](vllm.md) and TGI.

## PagedAttention

Naively the KV cache for each request is one contiguous block sized to the *max* length —
huge internal fragmentation. **PagedAttention** (vLLM) borrows OS virtual memory: split the
cache into fixed **blocks**, allocate on demand, map via a block table. Near-zero waste, and
blocks can be **shared** across requests with a common prefix (the basis of
[prompt caching](prompt_caching.md)).

```
  Logical KV (per request)        Physical blocks (shared pool)
  [b0][b1][b2]  ──block table──►  [#7][#3][#9]   ← non-contiguous, on demand
  shared prefix → same physical block, copy-on-write
```

## Speculative decoding

Decode is sequential and memory-bound, so the GPU is underused per step. **Speculative
decoding** uses a small **draft model** to propose several tokens cheaply, then the large
**target model** verifies them all in *one* parallel forward pass. Accepted tokens are free
speedup; rejects fall back to normal decoding. Output is identical to the target model.

```
  draft (small):   proposes  "the cat sat on"   (4 cheap tokens)
  target (large):  verifies all 4 in 1 pass  ──► accept 3, reject 1, continue
  net: ~2–3× faster decode, same distribution
```

Variants: **Medusa** (extra heads predict ahead), **EAGLE**, and **n-gram/prompt lookup**
(draft from the prompt itself, no second model).

## FlashAttention

A GPU-kernel rewrite of attention. Standard attention materializes the full N×N score matrix
in slow HBM. **FlashAttention** tiles the computation and keeps intermediates in fast on-chip
SRAM, never writing the full matrix — same result, less memory I/O, big speedup for long
sequences. It's an exact computation, not an approximation, and is on by default in most
modern stacks.

## Parallelism for big models

When a model exceeds one GPU:

- **Tensor parallelism** — split each layer's matrices across GPUs; they communicate every
  layer (needs fast interconnect like NVLink). Lowers latency.
- **Pipeline parallelism** — put different layers on different GPUs; activations flow through.
  Less communication, but pipeline bubbles.
- **Expert parallelism** — for [MoE](../machine_learning/moe.md) models, place experts on
  different GPUs.

## Quick reference

| Technique | Optimizes | Trade-off |
|-----------|-----------|-----------|
| KV cache | avoid recompute | memory grows with context |
| GQA/MQA | KV cache size | slight quality loss |
| Continuous batching | throughput | scheduler complexity |
| PagedAttention | memory waste / prefix sharing | block-table overhead |
| Speculative decoding | decode latency | needs draft model / extra heads |
| FlashAttention | memory I/O | kernel/hardware specific |
| [Quantization](../machine_learning/quantization.md) | weight + KV size | quality at low bits |

## Where this connects

- [vLLM](vllm.md) — production engine that implements continuous batching + PagedAttention.
- [Local inference](local_inference.md) — same KV-cache and offload constraints on consumer
  hardware.
- [Prompt caching](prompt_caching.md) — prefix sharing built on cached KV blocks.
- [Quantization](../machine_learning/quantization.md) — orthogonal lever that shrinks weights
  and KV cache.
- [Transformers architecture](transformers_architecture.md) — attention is what's being
  optimized here.

## Pitfalls

- **Tuning throughput while latency suffers.** Big batches raise tokens/sec but each user
  waits longer; pick the metric that matters and watch it in [observability](llm_observability.md).
- **Ignoring the KV cache in capacity planning.** Concurrency is usually KV-bound, not
  weight-bound; long contexts slash how many requests fit.
- **Speculative decoding with a poorly matched draft model.** Low acceptance rate means you
  pay for the draft and get little speedup.
- **Assuming FlashAttention/PagedAttention are approximations.** They're exact — no quality
  cost, so there's rarely a reason to disable them.
- **Cranking context length "because the model supports it."** Memory and prefill cost scale
  with it; long contexts are expensive even when supported.
