# Attention & Flash Attention

How the attention operation actually scales — the head-sharing variants (MHA/MQA/GQA), the KV cache that makes autoregressive decoding affordable, and the IO-aware kernels (FlashAttention) that turned a memory-bound bottleneck into a hardware-friendly one.

## Table of Contents

1. [Overview](#overview)
2. [Recap: Scaled Dot-Product Attention](#recap-scaled-dot-product-attention)
3. [Multi-Head Attention, and Why It's Expensive at Inference](#multi-head-attention-and-why-its-expensive-at-inference)
4. [MHA → MQA → GQA](#mha--mqa--gqa)
5. [The KV Cache](#the-kv-cache)
6. [FlashAttention: IO-Aware Exact Attention](#flashattention-io-aware-exact-attention)
7. [Other Efficiency Directions](#other-efficiency-directions)
8. [Best Practices](#best-practices)
9. [Common Pitfalls](#common-pitfalls)
10. [Where this connects](#where-this-connects)

## Overview

Attention is the sequence-mixing primitive at the heart of every
[transformer](transformers.md): each token builds its new representation as a weighted sum
of the others, where the weights are learned from content rather than position. The
[transformers](transformers.md) page derives the core operation and the full multi-head
implementation. This page picks up where that leaves off and asks the question that
dominates real systems: **how does attention scale?**

The answer threads through the rest of the repo's efficiency cluster. Attention's cost is
quadratic in sequence length and, at inference, bound by **memory bandwidth** rather than
arithmetic — which is why it connects so tightly to the [CUDA](cuda.md) memory hierarchy,
to [quantization](quantization.md) (the KV cache is often the first thing you quantize),
and to the same scaling pressures that motivated [Mixture-of-Experts](moe.md). Three ideas
do most of the heavy lifting: **head-sharing variants** (MQA/GQA) shrink the cache,
the **KV cache** removes redundant recomputation, and **FlashAttention** removes redundant
memory traffic.

```
  this page's arc
  ┌────────────────┐   too much   ┌──────────────┐   too much    ┌────────────────┐
  │ O(n²) attention │ ───memory──▶ │  KV cache +  │ ───traffic──▶ │ FlashAttention │
  │   (the recap)   │              │   MQA/GQA    │               │ (IO-aware)     │
  └────────────────┘              └──────────────┘               └────────────────┘
```

## Recap: Scaled Dot-Product Attention

Given queries `Q`, keys `K`, and values `V`, attention scores every query against every key,
normalizes with softmax, and mixes the values:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

The `√d_k` keeps the dot products from growing with dimension and pushing softmax into
saturated, low-gradient regions. Shapes, for a single head:

```
Q : (seq_q, d_k)        Q·Kᵀ : (seq_q, seq_k)   ← the n×n scores matrix
K : (seq_k, d_k)        softmax over last dim    ← this is the costly object
V : (seq_k, d_v)        out  : (seq_q, d_v)
```

```python
import torch.nn.functional as F

# logits in, never pre-softmaxed; is_causal applies the lower-triangular mask
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The full derivation, the multi-head wrapper, masking, and a worked numeric example live in
[transformers](transformers.md). The one fact to carry forward: building `Q·Kᵀ` is
**O(n²)** in both time and memory. For a 8K-token sequence that scores matrix is 64M
entries *per head, per layer* — the bottleneck everything below attacks.

## Multi-Head Attention, and Why It's Expensive at Inference

Multi-head attention (MHA) runs `H` attention operations in parallel on `d_model/H`-sized
slices, then concatenates — letting different heads specialize (syntax, coreference,
position). That much is covered in [transformers](transformers.md).

The cost that page doesn't dwell on shows up at **inference**. During autoregressive
decoding you generate one token at a time, and each new token must attend to the keys and
values of *every previous token*. Recomputing all past `K`/`V` at every step is quadratic
work; caching them (next section) trades it for memory. With standard MHA that cache holds
`H` separate key and value heads per layer — and for long contexts the **KV cache, not the
weights, becomes the memory bottleneck**. MQA and GQA exist to shrink exactly this.

## MHA → MQA → GQA

The variants form a spectrum on a single knob: **how many key/value heads** the `H` query
heads share.

```
        MHA                    GQA (groups=2)              MQA
   Q0 Q1 Q2 Q3             Q0 Q1   Q2 Q3              Q0 Q1 Q2 Q3
   │  │  │  │               \ /     \ /                \  \ /  /
   K0 K1 K2 K3              KV0     KV1                  KV0
   (H query, H KV)         (H query, G KV)          (H query, 1 KV)
   full quality            tunable middle           smallest cache
```

- **MHA** — `H` query heads, `H` KV heads. Best quality, biggest cache. The original design.
- **MQA** (Multi-Query) — `H` query heads share a **single** KV head. Cuts KV memory by a
  factor of `H`, dramatically speeding decoding, at some quality cost. Used by PaLM and
  Falcon.
- **GQA** (Grouped-Query) — the practical compromise: `G` KV heads, each shared by `H/G`
  query heads. Recovers most of MHA's quality at a fraction of the cache. The default for
  modern open models (Llama-2 70B, Llama-3, Mistral, Qwen).

| Variant | KV heads | KV-cache size | Quality | Used by |
|---|---|---|---|---|
| MHA | H | 1× (baseline) | best | GPT-2/3, early models |
| GQA | G (e.g. 8) | H/G smaller | ~MHA | Llama-2/3, Mistral, Qwen |
| MQA | 1 | H× smaller | slightly lower | PaLM, Falcon |

Implementation-wise the only change is that the cached `K`/`V` are **expanded** back to `H`
heads before the score step (or, better, the kernel broadcasts them without copying):

```python
def repeat_kv(x, n_rep):
    # x: (batch, n_kv_heads, seq, d_head) -> (batch, n_kv_heads*n_rep, seq, d_head)
    b, n_kv, s, d = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, s, d)
            .reshape(b, n_kv * n_rep, s, d))

k = repeat_kv(k_cache, n_heads // n_kv_heads)   # GQA: n_kv_heads < n_heads
v = repeat_kv(v_cache, n_heads // n_kv_heads)
```

## The KV Cache

Without a cache, generating token `t` recomputes the keys and values for tokens `0..t-1`
every step — pure waste, since those projections never change. The **KV cache** stores them
once and appends one column per generated token:

```
step 1   K: [k0]                 prompt encoded once, then…
step 2   K: [k0 k1]              each new token appends its own k,v
step 3   K: [k0 k1 k2]           and attends over the whole cache
   …          ▲
              └ only the newest column is computed each step → O(n) work
```

This is what makes decoding linear instead of quadratic — but it moves the pressure to
memory. The cache size is:

```
kv_bytes = 2 · n_layers · n_kv_heads · d_head · seq_len · batch · dtype_bytes
           ▲                ▲                                        ▲
        K and V         shrunk by GQA/MQA                    fp16=2, int8=1, int4=0.5
```

For a 70B-class model at long context this runs to **tens of GB per request** — frequently
larger than the activations and rivaling the weights. Three levers shrink it, and they
compose:

- **GQA/MQA** — fewer `n_kv_heads` (above).
- **[KV-cache quantization](quantization.md)** — store `K`/`V` in int8/int4. The cache
  tolerates low precision better than weights, so this is usually the *first* thing to
  quantize when you're memory-bound at long context.
- **PagedAttention** — instead of one contiguous per-request buffer (which forces
  worst-case allocation and fragments badly), allocate the cache in fixed-size blocks like
  OS virtual memory, so sequences grow on demand and share prefix blocks. This is the core
  trick behind [vLLM](../ai/vllm.md); see [inference optimization](../ai/inference_optimization.md)
  for how it interacts with batching and scheduling.

## FlashAttention: IO-Aware Exact Attention

The standard implementation's hidden cost is **memory traffic**, not math. It materializes
the full `n×n` scores matrix in slow off-chip **HBM** (GPU global memory), reads it back to
apply softmax, writes it again, then reads it once more for the value multiply. The
arithmetic is cheap; the round-trips to HBM dominate — attention is **memory-bandwidth
bound**.

FlashAttention restructures the computation to **never materialize the full matrix**. It
tiles `Q`, `K`, `V` into blocks small enough to fit in fast on-chip **SRAM** (the
[CUDA](cuda.md) shared memory / register space — see the tiled matmul pattern there),
streams the K/V blocks past each Q block, and accumulates the output incrementally using
**online softmax**: a running max and a running denominator that let it correct earlier
partial sums as new blocks arrive, so the result is **mathematically exact**.

```
   HBM (slow, GB)                         SRAM (fast, KB)
  ┌───────────────┐    load tiles        ┌──────────────┐
  │ Q  K  V       │ ───────────────────▶ │ Qi  Kj  Vj   │
  │               │                      │  ↓ compute    │
  │ O (output)    │ ◀─── write Oi ────── │ online softmax│
  └───────────────┘   (never write n×n)  └──────────────┘
       no S = QKᵀ matrix ever touches HBM
```

The online-softmax recurrence, per new K/V block, keeps the result stable without seeing
all scores at once:

```
m_new = max(m, rowmax(Sj))                    # running max
l     = exp(m - m_new)·l + rowsum(exp(Sj - m_new))   # running denominator
O     = exp(m - m_new)·O + exp(Sj - m_new)·Vj        # rescale + accumulate
m     = m_new
```

The payoff: memory drops from **O(n²) to O(n)**, HBM traffic falls sharply, and you get a
large wall-clock speedup with **identical numerics** (it is exact attention, not an
approximation). FlashAttention-2 improved work partitioning across warps and FA-3 targets
Hopper-class hardware. In practice you rarely write the kernel: PyTorch's
`F.scaled_dot_product_attention` auto-selects a flash backend when shapes and dtypes allow,
and serving stacks like [vLLM](../ai/vllm.md) and [local inference](../ai/local_inference.md)
runtimes use it by default.

## Other Efficiency Directions

Approaches that change the *math* rather than the kernel — useful, but with caveats:

- **Sliding-window / local attention** — each token attends only to the last `w` tokens,
  making cost linear; stacked layers still propagate information globally. Used in Mistral.
- **Sparse attention** — attend to a learned or fixed subset of positions (strided, block,
  global tokens). Reduces work but adds complexity and can lose long-range links.
- **Linear / low-rank attention** — approximate the softmax kernel (Performer, Linformer) to
  get O(n) attention. Attractive asymptotically, but quality and hardware-efficiency have
  kept them niche.

The honest summary: for the dense models that dominate production, **exact attention with
FlashAttention + GQA + a paged KV cache** won the race — it's faster *and* exact, so most
approximate schemes only pay off at extreme context lengths.

## Best Practices

- **Call `F.scaled_dot_product_attention`** rather than hand-rolling `softmax(QKᵀ)·V` — it
  auto-selects FlashAttention/memory-efficient backends and the masking is correct.
- **Choose GQA for new models** — near-MHA quality at a fraction of the KV-cache cost; it's
  the de-facto default.
- **Quantize the KV cache before the weights** when you're memory-bound at long context —
  the cache tolerates int8 well and is often the larger consumer.
- **Profile to confirm you're bandwidth-bound**, not compute-bound, before reaching for
  FlashAttention-style fixes — they target memory traffic specifically.
- **Use a paged KV cache** (vLLM-style) for serving many concurrent, variable-length
  requests; it slashes fragmentation and enables prefix sharing.

## Common Pitfalls

- **Pre-softmaxing the inputs** — `scaled_dot_product_attention` takes raw logits; applying
  softmax yourself double-normalizes and silently corrupts outputs.
- **Wrong causal mask with a KV cache** — during incremental decoding the single new query
  must see *all* cached keys; reusing a square `is_causal` mask sized for the prompt masks
  the cache incorrectly.
- **Forgetting `repeat_kv`** — with GQA/MQA the cached KV heads must be broadcast to match
  the query-head count, or shapes mismatch (or worse, broadcast wrong).
- **Assuming FlashAttention changes the result** — it's exact; if outputs differ
  meaningfully from a reference, it's a masking/dtype bug, not the kernel.
- **KV-cache OOM at long context** — memory grows linearly with sequence × batch; size it
  with the formula above and apply GQA + quantization + paging before raising context length.
- **Materializing scores for "debuggability"** — printing the `n×n` attention map at long
  context defeats the whole point and can OOM; sample a small slice instead.

## Where this connects

- [Transformers](transformers.md) — the core attention derivation and full multi-head
  implementation this page builds on
- [Positional encodings](positional_encoding.md) — RoPE is applied to Q/K before the cache
  write; where the position-indexing pitfall here originates
- [Neural networks](neural_networks.md) — attention as a differentiable layer in the stack
- [CUDA](cuda.md) — the SRAM/HBM memory hierarchy and tiled-matmul pattern FlashAttention
  exploits
- [Quantization](quantization.md) — int8/int4 KV-cache compression
- [Mixture-of-Experts](moe.md) — the other major lever for scaling transformer compute
- [State space models](state_space_models.md) — Mamba's sub-quadratic alternative that avoids
  the O(n²) cost and the growing KV cache
- [PyTorch](pytorch.md) — `F.scaled_dot_product_attention` and its flash backends
- [vLLM](../ai/vllm.md) — PagedAttention and high-throughput serving
- [Inference optimization](../ai/inference_optimization.md),
  [Local inference](../ai/local_inference.md) — where KV-cache and flash attention decisions
  play out in practice
- [Transformer architecture](../ai/transformers_architecture.md) — the surrounding model
  structure
