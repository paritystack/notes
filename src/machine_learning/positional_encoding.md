# Positional Encodings

How transformers recover the notion of *order* that attention throws away — from the original sinusoidal scheme through relative biases to RoPE, the rotary default behind every modern open LLM, plus the tricks that stretch a model past its trained context length.

## Table of Contents

1. [Overview](#overview)
2. [Why Attention Needs Position](#why-attention-needs-position)
3. [Sinusoidal (Absolute)](#sinusoidal-absolute)
4. [Learned Absolute Embeddings](#learned-absolute-embeddings)
5. [Relative Position Encodings](#relative-position-encodings)
6. [RoPE: Rotary Position Embedding](#rope-rotary-position-embedding)
7. [ALiBi](#alibi)
8. [Context-Length Extrapolation](#context-length-extrapolation)
9. [Comparison](#comparison)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

The [attention](attention.md) operation at the core of every [transformer](transformers.md)
is **permutation-invariant**: `softmax(Q·Kᵀ / √d)·V` treats its inputs as a *set*, not a
sequence. Shuffle the tokens and the outputs shuffle identically — the math contains no
notion of which token came first. Since "the dog bit the man" and "the man bit the dog"
must mean different things, position has to be injected explicitly. That injection is the
**positional encoding**, and the design space splits into two families:

```
  absolute  → give each token a signal that encodes its index i
              (added to the embedding, or baked into Q/K)

  relative  → bias the attention between tokens by their distance (i − j),
              never by the absolute indices themselves
```

Relative schemes generalize better — what matters for language is usually *how far apart*
two tokens are, not their absolute slot — and the field has converged on relative behavior.
The historical arc runs **sinusoidal → learned absolute → relative biases / ALiBi → RoPE**,
with RoPE (rotary) now the default in LLaMA, Mistral, Qwen, and Gemma. Models that process
sequences recurrently — RNNs and [state space models](state_space_models.md) — get order
*for free* from their step-by-step state update and need no positional encoding at all;
this page is about the transformer's problem specifically.

## Why Attention Needs Position

Take the attention output for query position `i`. It is a weighted sum over *all* key/value
positions, and the weights depend only on the **content** of `Q_i` and each `K_j` — never
on `i` or `j`:

```
permute the input tokens with any permutation π:

   x'  = π(x)                  → Q', K', V' are π-permutations of Q, K, V
   out' = Attention(Q',K',V')  = π( Attention(Q,K,V) )

the output just permutes the same way — order carries no information
```

Contrast this with the architectures that don't need a positional encoding:

```
  RNN / LSTM   order from the sequential hidden-state recurrence h_t = f(h_{t-1}, x_t)
  CNN          order from the fixed spatial layout of the kernel window
  SSM (Mamba)  order from the recurrent state scan — see state_space_models.md
  Transformer  NO inherent order → must add a positional encoding
```

So the encoder/decoder stack adds position information *before or inside* attention. The
rest of this page is the menu of how.

## Sinusoidal (Absolute)

The original "Attention Is All You Need" Transformer added a fixed, non-learned signal to
each token embedding: a vector of sines and cosines at **geometrically spaced
frequencies**, so each dimension oscillates at a different wavelength.

```
PE(pos, 2k)   = sin( pos / 10000^(2k/d) )
PE(pos, 2k+1) = cos( pos / 10000^(2k/d) )

  dim 0,1   ── fast oscillation  (short wavelength) → fine position
  dim 2,3   ── slower …
  dim d-2,d-1 ─ very slow        (long wavelength)  → coarse position

think of a binary clock in continuous form: each frequency is one "bit"
of the position, together they pin down pos uniquely
```

The trick is that a fixed linear map can shift any encoding from `pos` to `pos + k`
(rotating each sin/cos pair), so the model can learn to attend by *relative* offset even
though the encoding itself is absolute. It also needs **no parameters** and can — in
principle — produce encodings for positions longer than any seen in training, giving it a
weak form of extrapolation. The full implementation (the `PositionalEncoding` module) is
already worked out on the [transformers](transformers.md) page; this page won't re-paste it.

## Learned Absolute Embeddings

The simplest alternative: skip the math and just **learn** one embedding vector per
position, exactly like a token embedding, then add it to the token embedding at the input.

```python
import torch.nn as nn

tok_emb = nn.Embedding(vocab_size, d_model)
pos_emb = nn.Embedding(max_len,    d_model)   # one learned vector per position

x = tok_emb(tokens) + pos_emb(torch.arange(seq_len))   # add, then feed the stack
```

This is what BERT and GPT-2 use. It's trivial and works well *within* the trained range,
but it has a hard ceiling: there is no embedding for `pos ≥ max_len`, so the model
**cannot run on sequences longer than it was trained on** without adding and retraining
new rows. No extrapolation, and the position table costs `max_len × d_model` parameters.

## Relative Position Encodings

The conceptual shift: encode the **distance** `i − j` between a query and key rather than
their absolute indices. This matches the intuition that grammar and meaning depend on
relative offsets, and it naturally generalizes to unseen lengths.

- **Shaw et al. (2018)** — add a learned vector keyed by the clipped relative distance into
  the key (and value) of the attention score.
- **Transformer-XL** — reparameterize the score into content and position terms so a
  relative signal can span across cached segments, enabling longer effective context.
- **T5 relative bias** — the cleanest variant: a **learned scalar** per *bucketed* distance,
  added directly to the attention logits before softmax. Buckets are fine for nearby
  positions and coarse (logarithmic) for far ones, and the bias is shared across layers.

```
score(i, j) = Q_i · K_jᵀ / √d  +  b[ bucket(i − j) ]
                                   ▲
                          learned scalar, per head, indexed by relative distance
```

Because the bias is a function of distance only, a T5 model handles longer sequences
gracefully — distant buckets it never saw simply saturate rather than break.

## RoPE: Rotary Position Embedding

RoPE is the modern default. Instead of *adding* a position signal, it **rotates** the query
and key vectors by an angle proportional to their position, in 2-D subspaces. The key
property: after rotation, the dot product `Q_i · K_j` depends only on the **relative offset
`i − j`** — you get relative behavior with an absolute, parameter-free operation applied
right inside attention.

```
split each head's d-dim vector into d/2 pairs; rotate pair k by angle  pos · θ_k

   [x0]      [cos mθ   −sin mθ] [x0]          θ_k = base^(−2k/d),  base = 10000
   [x1]  →   [sin mθ    cos mθ] [x1]          m = position index

   pair 0 (k=0): rotates fast      ── fine-grained position
   pair d/2-1:   rotates slowly    ── coarse position
        (same geometric frequency ladder as the sinusoidal scheme)

dot product of two rotated vectors at positions i and j
   = function of (i − j) only   ← this is why RoPE is "relative"
```

```python
import torch

def apply_rope(x, pos, base=10000.0):
    # x: (..., seq, d_head) with d_head even; pos: (seq,) position indices
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float) / d)   # (d/2,)
    ang = pos[:, None] * theta[None, :]                                # (seq, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                                # even / odd dims
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

Why it won for decoder LLMs (LLaMA, Mistral, Qwen, Gemma):

- **Relative** behavior without a learned bias table — pure geometry.
- **No extra parameters** and negligible compute (a per-element rotate).
- Applied to **Q and K only** (not V), *after* the projection and *before* the score, so it
  drops cleanly into multi-head attention and the [KV cache](attention.md) — you cache the
  already-rotated keys.
- Extends to long context via frequency scaling (next section), which is much of why it
  displaced learned absolute embeddings.

## ALiBi

ALiBi (Attention with Linear Biases) goes further and uses **no positional embedding at
all**. It adds a static, head-specific penalty to the attention logits that grows linearly
with distance — nearer tokens are favored, farther ones discounted:

```
score(i, j) = Q_i · K_jᵀ / √d  −  m_h · (i − j)        (for j ≤ i, causal)
                                   ▲
                          fixed per-head slope (a geometric sequence over heads),
                          NOT learned, NOT added to the embedding
```

Different heads get different slopes `m_h`, so some attend locally and others globally. With
nothing to "run out of," ALiBi extrapolates to far longer sequences than it trained on — its
headline feature — at essentially zero cost and complexity. Used by BLOOM and MPT.

## Context-Length Extrapolation

The practical headache: a model trained at 4K tokens must often serve 32K+. Absolute and
learned schemes simply break past `max_len`; RoPE degrades because positions imply rotation
angles far outside the trained range. Two RoPE-specific fixes dominate:

```
Position Interpolation (PI)   squeeze positions into the trained range:
   pos' = pos · (L_train / L_target)     → angles stay in-distribution
   needs a short fine-tune; simple, effective

NTK-aware / YaRN              scale the RoPE base θ instead of the positions,
   stretching low frequencies (long-range) more than high (local),
   preserving fine resolution; YaRN adds a temperature/attention-scale tweak
   → best quality, the common choice for long-context releases
```

These are typically applied with a brief long-context fine-tune (often a parameter-efficient
[LoRA](lora.md) adapter) so the model adapts to the rescaled positions. ALiBi, by contrast,
needs none of this — its linear bias extrapolates by construction.

## Comparison

| Scheme | Type | Where applied | Learned params | Extrapolation | Used by |
|---|---|---|---|---|---|
| **Sinusoidal** | absolute | added to input embedding | none | weak | original Transformer |
| **Learned absolute** | absolute | added to input embedding | `max_len·d` | **none** (hard cap) | BERT, GPT-2 |
| **T5 relative bias** | relative | added to logits | small (per bucket/head) | good | T5 |
| **RoPE** | relative | rotates Q, K | none | good (with PI/YaRN) | LLaMA, Mistral, Qwen, Gemma |
| **ALiBi** | relative | linear penalty on logits | none | **strong** | BLOOM, MPT |

## Best Practices

- **Default to RoPE for a new decoder LLM** — relative behavior, no parameters, cache-
  friendly, and a well-trodden path to long context via YaRN/PI.
- **Reach for ALiBi when length extrapolation is the priority** and you want zero added
  machinery — it stretches well past the trained range with no fine-tune.
- **Keep the RoPE base θ identical between training and inference.** It's part of the model
  definition, not a free knob.
- **Apply RoPE after the Q/K projection and before writing the KV cache** — store the
  *rotated* keys so cached and fresh tokens share one position convention.
- **For long-context adaptation, scale RoPE (PI/YaRN) plus a short fine-tune** rather than
  training from scratch; a [LoRA](lora.md) adapter is often enough.
- **Don't add a positional encoding to a [state space model](state_space_models.md)** — the
  recurrence already carries order.

## Common Pitfalls

- **Mismatched RoPE base θ between training and serving** — a silent, severe quality
  collapse; the keys and queries rotate by inconsistent angles. Pin θ to the checkpoint.
- **Wrong position index with a KV cache** — during incremental decoding the new token's
  position must continue from the **cache length**, not reset to 0. This mirrors the causal-
  mask pitfall on the [attention](attention.md) page: get the offset wrong and the rotation
  (or absolute index) is applied to the wrong slot.
- **Exceeding a learned-embedding `max_len`** — there is no row for it; the lookup errors or
  (worse) wraps. Learned absolute encodings cannot extrapolate, full stop.
- **Rotating V with RoPE** — RoPE applies to **Q and K only**; the value vectors are not
  rotated. Rotating V corrupts the mixed output.
- **Assuming sinusoidal/RoPE "just works" at 4× context** — angles drift out of the trained
  distribution and quality degrades; apply PI/YaRN and a short fine-tune first.
- **Double-counting position** — adding learned absolute embeddings *and* RoPE, or T5 bias
  *and* sinusoidal, usually hurts; pick one scheme.

## Where this connects

- [Attention](attention.md) — where RoPE is applied (Q/K, pre-cache) and the KV-cache
  position-indexing pitfall this page mirrors
- [Transformers](transformers.md) — the "why position matters" intro and the full
  sinusoidal `PositionalEncoding` implementation
- [State space models](state_space_models.md) — order comes from the recurrent scan, so no
  positional encoding is needed
- [LoRA](lora.md) — long-context fine-tuning to adapt a model to PI/YaRN-rescaled positions
- [Neural networks](neural_networks.md) — embeddings as the input layer the position signal
  is added to
- [PyTorch](pytorch.md) — `nn.Embedding` for learned/absolute encodings and the rotate ops
  behind RoPE
