# State Space Models & Mamba

A family of sequence models — S4, then Mamba — that scale **linearly** with sequence length instead of quadratically, offering a sub-quadratic alternative to attention for very long contexts.

## Table of Contents

1. [Overview](#overview)
2. [The Continuous State Space Model](#the-continuous-state-space-model)
3. [Discretization](#discretization)
4. [Two Views: Recurrence vs Convolution](#two-views-recurrence-vs-convolution)
5. [S4 and HiPPO](#s4-and-hippo)
6. [Mamba: the Selective SSM](#mamba-the-selective-ssm)
7. [Why It Matters: Complexity and the KV-Cache Angle](#why-it-matters-complexity-and-the-kv-cache-angle)
8. [Hybrids](#hybrids)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)
11. [Where this connects](#where-this-connects)

## Overview

[Attention](attention.md) is the engine behind [transformers](transformers.md), but it has a
hard scaling problem: comparing every token to every other token costs **O(n²)** compute and
memory in the sequence length `n`. Double the context and you quadruple the cost. State space
models (SSMs) take a different route — they process a sequence like a recurrence, carrying a
fixed-size hidden state forward token by token, which makes them **O(n)** in time and
**O(1)** in state size at inference. That is a fundamentally different scaling axis from
[mixture-of-experts](moe.md) (which scales parameters, not sequence cost).

The lineage runs: classical linear **state space models** from control theory →
**S4** (structured SSMs that finally trained well on long sequences) →
**Mamba** (a *selective* SSM that closed much of the quality gap with transformers). The key
tension throughout is that the same math has **two equivalent forms** — a parallelizable
convolution for training and an efficient recurrence for inference — and Mamba's central trick
is to make the model input-dependent without losing that efficiency.

## The Continuous State Space Model

An SSM borrows the classic linear-system formulation from control theory. A 1-D input signal
`u(t)` is mapped to an output `y(t)` through a latent state vector `x(t)`:

```
x'(t) = A x(t) + B u(t)        # state evolves: dynamics A, input projection B
y(t)  = C x(t) + D u(t)        # output read-out C, skip connection D

  x ∈ ℝᴺ  (hidden state of size N)
  A ∈ ℝᴺˣᴺ  B ∈ ℝᴺˣ¹  C ∈ ℝ¹ˣᴺ  D ∈ ℝ
```

Intuitively this is a **linear RNN** with structured dynamics: `A` controls how the memory
state decays and mixes over time, `B` writes the current input into that memory, and `C` reads
it out. `D` is just a residual/skip term. The promise is long-range memory — if `A` is chosen
well, information can persist across thousands of steps without the vanishing-gradient problems
that plague vanilla RNNs.

## Discretization

Real sequences are discrete tokens, not a continuous signal, so the continuous parameters
`(A, B)` must be converted to discrete ones `(Ā, B̄)` using a **step size** `Δ` (often via a
zero-order hold):

```
Ā = exp(Δ A)
B̄ = (Δ A)⁻¹ (exp(Δ A) − I) · Δ B        # zero-order-hold discretization

discrete recurrence:
  xₖ = Ā xₖ₋₁ + B̄ uₖ
  yₖ = C xₖ
```

`Δ` acts like a learnable resolution / timescale: a large `Δ` makes the model focus on the
current input, a small `Δ` makes it retain more of the past. As we'll see, making `Δ`
**input-dependent** is one of the things that turns a plain SSM into Mamba.

## Two Views: Recurrence vs Convolution

Because the recurrence is **linear** (no nonlinearity inside the loop), you can unroll it into
a single **convolution** with a fixed kernel — and this duality is what makes SSMs practical:

```
Recurrent view (O(n), inference)        Convolutional view (parallel, training)

  x₀→x₁→x₂→ ... →xₙ                       y = u  *  K̄
   ↑   ↑   ↑                              K̄ = (C B̄, C Ā B̄, C Ā² B̄, ...)
  u₀  u₁  u₂                              a length-n global kernel built from
  step-by-step, constant memory          the same A, B, C — one big FFT/conv
```

- **Training** uses the convolutional form: the whole output is computed in parallel across
  the sequence (like a [convolution](convolution.md) with a very long kernel), so GPUs stay
  busy.
- **Inference** uses the recurrent form: generate one token at a time, updating a fixed-size
  state `x` — no growing cache, constant memory per step.

You train fast *and* decode cheaply, getting the best of RNNs and convolutions — as long as
the parameters stay constant across time, which is what makes the convolution kernel
well-defined.

## S4 and HiPPO

A naive SSM with a random `A` does not learn long-range dependencies well, and the matrix
powers `Āᵏ` are numerically unstable. **S4** (Structured State Spaces, Gu et al., 2021) fixed
this with two ideas:

- **HiPPO initialization** — a specific structured `A` matrix derived to optimally compress a
  signal's history into the state (it makes the state approximate the input's running history
  via orthogonal polynomials). This gave SSMs genuine long-range memory.
- **A structured (diagonal-plus-low-rank, later just diagonal) parameterization** of `A` that
  makes the convolution kernel efficient and stable to compute.

S4 was the breakthrough that let SSMs dominate the **Long Range Arena** benchmark, handling
sequences of tens of thousands of steps where transformers ran out of memory. But it was still
a **linear time-invariant** system: the same `A, B, C` applied to every token, which limits
its ability to selectively remember or ignore content based on what the token actually is.

## Mamba: the Selective SSM

Mamba (Gu & Dao, 2023), also called **S6**, makes the SSM **selective**: the parameters `B`,
`C`, and `Δ` become **functions of the input** rather than fixed matrices. Now the model can
decide, per token, what to write into memory, what to read out, and how fast to forget —
something a constant SSM (and the original LTI assumption) cannot do.

```
selective SSM:
  Δ, B, C  =  linear(uₖ)        # input-dependent, vary at every step
  xₖ = Ā(Δ) xₖ₋₁ + B̄(Δ) uₖ
  yₖ = C xₖ
```

The cost of selectivity: the parameters now change every step, so the model is no longer
time-invariant and the **convolution view breaks** — you can't precompute a single fixed
kernel. Mamba recovers parallel training with a **hardware-aware parallel scan** (an
associative scan kept in fast SRAM, fused to avoid materializing the large state in HBM —
the same IO-aware philosophy as [FlashAttention](attention.md)).

The full **Mamba block** wraps the selective SSM with a gated, conv-augmented structure:

```
            ┌─────────────────────────────────┐
  x ──►RMSNorm──►Linear─┬─►Conv1d─►SiLU─►SSM──►(⊗)──►Linear──►(+)──► out
            │           │                       ▲          ▲
            │           └─►Linear─────►SiLU─────┘          │
            └────────────────── residual ─────────────────┘
```

A short causal [convolution](convolution.md) provides local context, the selective SSM
provides long-range mixing, a SiLU-gated branch modulates the output (a gated-MLP flavor), and
the whole thing sits in a [pre-norm RMSNorm](normalization.md) residual block just like a
transformer layer.

## Why It Matters: Complexity and the KV-Cache Angle

The headline is scaling. Attention pays O(n²) to relate all token pairs and, at inference,
must keep a **KV cache** that grows linearly with every generated token (see
[attention](attention.md)). An SSM compresses all history into a **fixed-size state**, so the
memory at step `n` is constant:

| | Attention (Transformer) | SSM / Mamba |
|---|---|---|
| Training compute | O(n²) | O(n) (or O(n log n)) |
| Inference per token | O(n) — attends to all past KV | **O(1)** — update fixed state |
| Inference memory | KV cache grows with `n` | **constant** recurrent state |
| Exact recall / copying | strong (can look up any token) | weaker (lossy fixed state) |

That constant-memory decoding is a big deal for long-context generation, streaming, and edge
deployment. The trade-off is the last row: because all history is squeezed into a
fixed-size state, pure SSMs are weaker at **exact associative recall** — tasks like copying a
specific earlier token or precise in-context lookup, where attention's ability to address any
past position directly is hard to beat.

## Hybrids

In practice the strongest long-context models **interleave** a few attention layers among many
Mamba layers, getting linear scaling for most of the depth while keeping attention's exact
recall where it counts. Examples include **Jamba** (Mamba + attention +
[mixture-of-experts](moe.md)) and other Mamba/Transformer hybrids. The recipe — mostly SSM for
cheap sequence mixing, sprinkled attention for precision, optional MoE for parameter scaling —
is currently the most practical way to ship very-long-context models.

```python
# Illustrative selective-scan recurrence (NOT the fused kernel) — one step at a time.
# A: (d, N) diagonal log-dynamics; x,B,C,Δ are input-dependent per step.
def ssm_step(u_k, state, A, dt_k, B_k, C_k):
    A_bar = torch.exp(dt_k.unsqueeze(-1) * A)          # discretize A with Δ
    B_bar = dt_k.unsqueeze(-1) * B_k                    # simplified B̄
    state = A_bar * state + B_bar * u_k.unsqueeze(-1)   # xₖ = Ā xₖ₋₁ + B̄ uₖ
    y_k   = (state * C_k).sum(-1)                       # yₖ = C xₖ
    return y_k, state                                   # constant-size state carried forward
```

## Best Practices

- **Reach for SSMs/Mamba when context is very long** (tens of thousands of tokens), when you
  need **streaming/constant-memory decoding**, or for **edge** inference where a growing KV
  cache is prohibitive.
- **Prefer a hybrid** (mostly Mamba with a few attention layers) over a pure SSM when the task
  needs precise recall, retrieval, or copying — it's the practical sweet spot.
- **Use RMSNorm in pre-norm residual blocks**, exactly as in [transformers](transformers.md);
  Mamba reuses the same [normalization](normalization.md) machinery.
- **Trust the official fused selective-scan kernel** for training — a naive Python scan is
  fine for understanding but far too slow for real models.

## Common Pitfalls

- **Expecting transformer-level associative recall from a pure SSM** — the fixed-size state is
  lossy; add attention layers if exact lookup/copying matters.
- **Numerical instability in discretization** — naive matrix powers `Āᵏ` blow up; rely on the
  structured/diagonal parameterization and stable discretization that S4/Mamba use.
- **Conflating SSMs with vanilla RNNs** — SSMs are *linear* recurrences (no nonlinearity in
  the loop), which is precisely what enables the parallel convolution/scan; classic RNNs
  cannot be unrolled that way.
- **Assuming Mamba is "just" a convolution** — selectivity makes parameters input-dependent
  and time-varying, breaking the fixed-kernel convolution view and requiring the scan.

## Where this connects

- [Attention](attention.md) — the O(n²) mechanism and growing KV cache that SSMs aim to avoid;
  FlashAttention's IO-aware idea reappears in Mamba's fused scan
- [Transformers](transformers.md) — the dominant architecture SSMs are positioned against and
  hybridized with
- [Mixture of Experts](moe.md) — a complementary scaling axis (parameters); combined with
  Mamba in models like Jamba
- [Convolution](convolution.md) — the convolutional view of the SSM kernel; Mamba's local conv
- [Normalization](normalization.md) — RMSNorm pre-norm blocks used inside Mamba
- [Neural networks](neural_networks.md), [Deep learning](deep_learning.md) — SSMs as a
  sequence-modeling architecture and the RNN lineage they refine
