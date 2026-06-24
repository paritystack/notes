# Normalization Layers

The layers that keep activations well-scaled so deep networks train fast and stay stable — BatchNorm, LayerNorm, RMSNorm, and GroupNorm — and the question of *where* to put them relative to the residual connection.

## Table of Contents

1. [Overview](#overview)
2. [Internal Covariate Shift — and the Modern View](#internal-covariate-shift--and-the-modern-view)
3. [The Common Recipe](#the-common-recipe)
4. [BatchNorm](#batchnorm)
5. [LayerNorm](#layernorm)
6. [RMSNorm](#rmsnorm)
7. [GroupNorm (and InstanceNorm)](#groupnorm-and-instancenorm)
8. [Where the Norm Goes: Pre-LN vs Post-LN](#where-the-norm-goes-pre-ln-vs-post-ln)
9. [Comparison](#comparison)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

A normalization layer rescales the activations flowing through a network so each unit sees a
well-conditioned distribution — roughly zero mean and unit variance — regardless of what the
layers below it are doing. This matters because deep [neural networks](neural_networks.md)
are hard to train: as weights shift during [optimization](optimizers.md), the distribution of
inputs to each layer drifts, gradients explode or vanish, and the [learning rate](optimizers.md)
that worked at step 0 destabilizes at step 1000. Normalization tames that drift, letting you
use higher learning rates, weaker initialization, and far deeper stacks.

Every normalization layer shares the same core operation — standardize, then re-scale and
re-shift with learnable parameters — and they differ only in **which axes** the statistics
are computed over:

```
y = γ · (x − μ) / √(σ² + ε) + β

  μ, σ²  = mean / variance over some chosen set of axes
  γ, β   = learnable per-channel scale and shift (the "affine" params)
  ε      = small constant for numerical stability (e.g. 1e-5)
```

The choice of axes is the whole story. BatchNorm averages over the batch; LayerNorm averages
over the features of a single example; GroupNorm splits channels into groups. That one design
decision determines whether the layer depends on batch size, behaves differently at training
vs. inference, and suits CNNs or [transformers](transformers.md). Modern LLMs have largely
standardized on **RMSNorm in a pre-norm residual block**, but understanding why takes the
progression below.

## Internal Covariate Shift — and the Modern View

BatchNorm (Ioffe & Szegedy, 2015) was originally motivated by **internal covariate shift**:
the idea that as earlier layers update, the distribution of inputs to later layers keeps
changing, forcing them to continually re-adapt. Normalizing each layer's inputs to a stable
distribution was the proposed fix, and it worked spectacularly — networks trained faster and
tolerated much higher learning rates.

The *explanation*, however, turned out to be incomplete. Later work (Santurkar et al., 2018,
*"How Does Batch Normalization Help Optimization?"*) showed that BatchNorm's benefit comes
less from reducing covariate shift and more from **smoothing the loss landscape** — it makes
the gradients more predictable and Lipschitz, so larger, more confident steps stay safe. The
practical takeaway is unchanged (normalize, and training gets easier), but the modern framing
is "better-conditioned optimization surface," not "stable input statistics."

## The Common Recipe

Every layer here does the same three things:

```
1. Standardize:  x̂ = (x − μ) / √(σ² + ε)     # zero mean, unit variance over chosen axes
2. Scale:        γ · x̂                        # learnable, lets the layer undo normalization
3. Shift:        + β                          # learnable bias
```

The learnable `γ` and `β` are important: pure standardization would rob the layer of
representational power (e.g. it could not exploit the saturating region of a sigmoid), so the
affine transform lets the network *recover* any scale/shift it actually needs — even the
identity, by learning `γ = √σ²` and `β = μ`. The variants below change only **what μ and σ²
are computed over**.

```
Activation tensor for a sequence/vision batch:

  BatchNorm  → normalize over (N, [L])  per channel   ← depends on batch
  LayerNorm  → normalize over (C)        per token     ← batch-independent
  GroupNorm  → normalize over (C/groups) per token     ← batch-independent
  RMSNorm    → like LayerNorm but no mean-centering
```

## BatchNorm

BatchNorm normalizes each feature (channel) using statistics computed **across the batch**.
For a CNN activation of shape `(N, C, H, W)`, it pools the mean and variance over `N·H·W` for
each of the `C` channels:

```
per channel c:
  μ_c   = mean over all (n, h, w)
  σ²_c  = var  over all (n, h, w)
  y     = γ_c · (x − μ_c) / √(σ²_c + ε) + β_c
```

The catch: at inference you often have a single example, so there is no batch to average over.
BatchNorm solves this by keeping **running (EMA) estimates** of the mean and variance during
training and using those frozen statistics at eval time. This makes BatchNorm one of the few
layers whose behavior **differs between train and eval mode**:

```python
import torch.nn as nn

bn = nn.BatchNorm2d(64)        # 64 channels; tracks running_mean / running_var

bn.train()                     # uses batch stats, updates running estimates
y = bn(x)

bn.eval()                      # uses frozen running stats — deterministic
y = bn(x)
```

The `momentum` argument controls how fast the running stats update
(`running = (1−momentum)·running + momentum·batch`). BatchNorm shines in CNNs with large
batches but struggles when batches are small (noisy statistics) or when examples are not
i.i.d. across the batch dimension — which is exactly the case for variable-length sequences,
making it a poor fit for transformers and RNNs.

## LayerNorm

LayerNorm sidesteps the batch dependence entirely: it normalizes across the **feature
dimension of each example independently**. For a transformer activation of shape
`(N, L, D)`, each token's `D`-dimensional vector is standardized on its own:

```
per token (n, l):
  μ   = mean over the D features
  σ²  = var  over the D features
  y   = γ · (x − μ) / √(σ² + ε) + β        # γ, β are length-D vectors
```

Because the statistics never touch the batch or sequence axes, LayerNorm behaves
**identically in train and eval mode** and works with any batch size — including batch size 1
during autoregressive generation. That is why it became the default for
[transformers](transformers.md) and [attention](attention.md)-based models.

```python
ln = nn.LayerNorm(768)         # normalize over the last dim (model width)
y = ln(x)                      # x: (batch, seq, 768) → same shape, per-token normalized
```

## RMSNorm

RMSNorm (Zhang & Sennrich, 2019) observed that LayerNorm's **mean-centering** contributes
little, and dropped it. It rescales by the root-mean-square of the activations only — no
subtraction of the mean, and (usually) no `β` bias:

```
LayerNorm:  y = γ · (x − μ) / √(σ² + ε) + β
RMSNorm:    y = γ · x / √(mean(x²) + ε)        # no mean subtraction, no bias

  RMS(x) = √( (1/D) · Σ xᵢ² )
```

The payoff is **fewer operations** (no mean, no centering) and fewer parameters, with no
measurable quality loss. Combined with its slight edge in throughput at scale, RMSNorm is now
the normalization of choice in most modern LLMs — LLaMA, Mistral, Gemma, and friends — almost
always in a pre-norm configuration.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # γ only, no bias
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms
```

## GroupNorm (and InstanceNorm)

GroupNorm (Wu & He, 2018) is a middle ground for vision. It splits the `C` channels into `G`
groups and normalizes over each group's channels (plus the spatial dims) **per example**, so
like LayerNorm it is independent of batch size — which is what makes it the go-to when
batches are tiny (detection, segmentation, high-resolution images that only fit a few per
GPU, where BatchNorm's statistics get too noisy).

```
GroupNorm with G groups on (N, C, H, W):
  for each example n, each group g:
      normalize over (C/G channels) × H × W

Special cases:
  G = 1  → LayerNorm-like (all channels in one group)
  G = C  → InstanceNorm  (each channel normalized on its own; used in style transfer)
```

```python
gn = nn.GroupNorm(num_groups=32, num_channels=256)   # common default: 32 groups
y  = gn(x)                                            # x: (N, 256, H, W)
```

It's the bridge between LayerNorm (one group) and InstanceNorm (one channel per group), and
because it never looks at the batch, it has no train/eval discrepancy and no running stats to
maintain.

## Where the Norm Goes: Pre-LN vs Post-LN

In a residual block you can place the norm either **inside** the residual branch before the
sublayer (pre-norm) or **after** the residual add (post-norm). The original Transformer used
post-norm; essentially all modern LLMs use pre-norm.

```
Post-LN (original Transformer)        Pre-LN (modern default)

  x ──►(+)──► LayerNorm ──► out         x ─┬───────────────►(+)──► out
       ▲                                   │                  ▲
       │                                   └─ LayerNorm ─ sublayer
   sublayer                                    (norm first)
       ▲
       x                              the residual path stays "clean" —
                                      no norm on the identity stream
```

- **Pre-LN** keeps an unnormalized identity path from input to output, so gradients flow
  cleanly through the residual stream. This makes deep stacks far more stable, tolerates
  larger learning rates, and reduces (or removes) the need for a long learning-rate
  [warmup](optimizers.md) — the main reason it won for large models.
- **Post-LN** can reach slightly *better* final quality when it trains successfully, because
  normalization acts on the full residual output, but it is finicky: deep post-LN
  transformers often diverge without careful warmup and initialization.

A common modern refinement is to add a final norm right before the output head (and some
architectures even normalize both the input and output of each sublayer). The practical
default for a new model is **pre-norm with RMSNorm**.

## Comparison

| Layer | Normalizes over | Batch-dependent? | Mean-centered? | Train≠Eval? | Typical use |
|---|---|---|---|---|---|
| **BatchNorm** | batch (+ spatial) per channel | **yes** | yes | **yes** (running stats) | CNNs, large batches |
| **LayerNorm** | features per token | no | yes | no | Transformers, RNNs |
| **RMSNorm** | features per token | no | **no** | no | Modern LLMs (LLaMA etc.) |
| **GroupNorm** | channel groups per example | no | yes | no | Vision, small batches |

## Best Practices

- **Transformers/LLMs → LayerNorm or RMSNorm, pre-norm.** RMSNorm if you want the cheaper,
  modern default; both are batch-independent and safe for batch-size-1 generation.
- **CNNs with big batches → BatchNorm.** It still gives the best results when statistics are
  reliable.
- **Small-batch vision → GroupNorm** (e.g. detection/segmentation), where BatchNorm's batch
  statistics are too noisy.
- **Exclude norm parameters (`γ`, `β`) and biases from weight decay** — decaying them hurts
  with no benefit. See the parameter-group split in [optimizers](optimizers.md).
- **Call `model.eval()` before inference** so BatchNorm uses its frozen running statistics
  (and dropout is disabled).
- **Pre-norm reduces the need for warmup**; if you must use post-norm, warm up carefully.

## Common Pitfalls

- **Forgetting `model.eval()`** with BatchNorm → it keeps using noisy per-batch statistics at
  inference, giving unstable, batch-dependent predictions.
- **Tiny batches with BatchNorm** → the mean/variance estimates are noise; switch to
  GroupNorm or LayerNorm.
- **Fine-tuning without freezing BatchNorm stats** → a few large-LR steps corrupt the running
  statistics learned on the pretraining data; freeze them (eval mode for the BN layers) when
  fine-tuning on a small dataset.
- **Placing the norm on the wrong side of the residual** → silently turns a stable pre-norm
  block into a fragile post-norm one (or vice versa); double-check the residual wiring.
- **Decaying `γ`/`β`** → quietly degraded results; put norm params in a no-decay group.
- **Using BatchNorm in a sequence model** → batch statistics mix unrelated tokens across
  variable-length examples; use LayerNorm.

## Where this connects

- [Optimizers](optimizers.md) — warmup interacts with pre/post-norm; norm params excluded
  from weight decay
- [Neural networks](neural_networks.md) — normalization is a standard layer in the stack
- [Transformers](transformers.md), [Attention](attention.md) — LayerNorm/RMSNorm in the
  residual block; the pre-norm default
- [Convolution](convolution.md) — BatchNorm and GroupNorm in CNN architectures
- [Deep learning](deep_learning.md) — why deep nets need normalization to train at all
- [State space models](state_space_models.md) — Mamba blocks also use RMSNorm in pre-norm
  residuals
- [PyTorch](pytorch.md) — `nn.BatchNorm*d`, `nn.LayerNorm`, `nn.GroupNorm` implementations
