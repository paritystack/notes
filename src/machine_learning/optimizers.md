# Optimizers & Learning-Rate Schedules

The algorithms that turn gradients into weight updates — from plain SGD to AdamW — and the schedules that steer the learning rate over the course of training.

## Table of Contents

1. [Overview](#overview)
2. [The Gradient Descent Family](#the-gradient-descent-family)
3. [Momentum and Nesterov](#momentum-and-nesterov)
4. [Adaptive Methods: AdaGrad → RMSProp → Adam → AdamW](#adaptive-methods)
5. [Weight Decay vs. L2 Regularization](#weight-decay-vs-l2-regularization)
6. [Learning-Rate Schedules](#learning-rate-schedules)
7. [Large-Scale & Memory-Efficient Optimizers](#large-scale--memory-efficient-optimizers)
8. [Gradient Clipping and Accumulation](#gradient-clipping-and-accumulation)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)
11. [Where this connects](#where-this-connects)

## Overview

An optimizer is the rule that consumes a loss gradient and produces the next set of
weights. It sits at the heart of the training loop for every [neural network](neural_networks.md)
and every [deep learning](deep_learning.md) model: backpropagation computes `∂L/∂w`, and
the optimizer decides *how far* and *in what direction* to step. The choice of optimizer
and its learning-rate schedule is often the difference between a model that converges
smoothly and one that diverges or stalls. In [PyTorch](pytorch.md) this is the
`torch.optim` package; in [JAX](jax.md) it is typically [Optax](jax.md). Modern LLM
training has largely standardized on **AdamW + warmup + cosine decay**, but understanding
why takes the full progression below.

```
  training step
  ┌──────────────┐   gradients    ┌───────────────┐   updated weights
  │ forward pass │ ─────────────▶ │   optimizer   │ ──────────────────▶
  │  + loss      │  ∂L/∂w (autograd)│ step()        │   w ← w − lr · g~
  └──────────────┘                └───────────────┘
        ▲                                 │
        └─────────── next batch ──────────┘
```

## The Gradient Descent Family

All first-order optimizers descend the loss surface by stepping opposite the gradient:

```
w ← w − η · ∇L(w)        η = learning rate
```

The variants differ in **how much data** each gradient estimate uses:

| Variant | Gradient computed over | Trade-off |
|---|---|---|
| Batch GD | the entire dataset | accurate gradient, slow, memory-heavy |
| Stochastic GD (SGD) | one sample | noisy, fast, can escape shallow minima |
| Mini-batch GD | a batch (e.g. 32–4096) | the practical default; vectorizes on GPU |

In practice "SGD" almost always means mini-batch SGD. The noise from small batches acts
as a mild regularizer, but too-noisy gradients slow convergence — hence momentum and
adaptive methods below.

```
loss surface (1-D slice)
   L
   │   .                     plain GD takes uniform steps downhill;
   │    \.                   it crawls along flat valleys and can
   │     \ `.                oscillate across steep ravines.
   │      \   `._
   │       \_____`-.________
   └───────────────────────▶ w
```

```python
import torch

model = torch.nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for x, y in loader:
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()      # autograd fills .grad
    opt.step()           # w ← w − lr · grad
```

## Momentum and Nesterov

Plain SGD treats each step independently. **Momentum** accumulates an exponentially
decaying moving average of past gradients, so the optimizer keeps rolling in a consistent
direction and damps oscillations across ravines:

```
v ← β·v + ∇L(w)          (β ≈ 0.9, the momentum coefficient)
w ← w − η·v
```

Think of `v` as the velocity of a ball rolling downhill: consistent gradients build speed,
while gradients that flip sign cancel out. **Nesterov accelerated gradient (NAG)** is a
refinement that evaluates the gradient at the *look-ahead* position `w − η·β·v`, giving a
more responsive correction near the minimum.

```python
# SGD with Nesterov momentum — still strong for vision/CNN training
opt = torch.optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, nesterov=True)
```

SGD+momentum often *generalizes* better than adaptive methods on
[convolutional](convolution.md) vision models, which is why ResNet-style recipes still use
it. Transformers, by contrast, are usually trained with Adam-family optimizers.

## Adaptive Methods

The idea: give each parameter its **own** effective learning rate, scaled by the history
of its gradients. This helps when features have very different scales or frequencies.

**AdaGrad** accumulates squared gradients and divides by their root. It adapts well to
sparse features but the denominator grows monotonically, so the LR eventually decays to
zero — fatal for long training.

```
G ← G + g²
w ← w − η · g / (√G + ε)
```

**RMSProp** fixes AdaGrad's vanishing LR by using an *exponential* moving average of
squared gradients instead of a running sum:

```
G ← β·G + (1−β)·g²        (β ≈ 0.99)
w ← w − η · g / (√G + ε)
```

**Adam** combines momentum (first moment `m`) with RMSProp's adaptive scaling (second
moment `v`), plus a bias correction so the early steps aren't biased toward zero:

```
m ← β₁·m + (1−β₁)·g                 (β₁ = 0.9   → momentum)
v ← β₂·v + (1−β₂)·g²                (β₂ = 0.999 → variance)
m̂ = m / (1 − β₁ᵗ)                   bias correction at step t
v̂ = v / (1 − β₂ᵗ)
w ← w − η · m̂ / (√v̂ + ε)
```

```python
# Adam from scratch — one parameter tensor, to show the moments
m = torch.zeros_like(w); v = torch.zeros_like(w)
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3

for t in range(1, steps + 1):
    g = w.grad
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    w.data -= lr * m_hat / (v_hat.sqrt() + eps)
```

**AdamW** is the version used for essentially all modern LLM and [transformer](transformers.md)
training. Its single change — *decoupled weight decay* — is explained next, and it matters
enough that you should reach for `AdamW`, not `Adam`, by default.

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                        betas=(0.9, 0.999), weight_decay=0.1)
```

## Weight Decay vs. L2 Regularization

These are identical for plain SGD but **diverge** for adaptive optimizers — the subtlety
AdamW exists to fix.

- **L2 regularization** adds `λ·‖w‖²` to the *loss*, so the penalty term `2λw` flows
  through the adaptive denominator `√v̂`. Parameters with large historical gradients get
  *less* decay — usually not what you want.
- **Decoupled weight decay (AdamW)** applies the shrink directly to the weights, *outside*
  the adaptive scaling:

```
w ← w − η · ( m̂ / (√v̂ + ε)  +  λ·w )
                                  └── applied directly, not via the gradient
```

This is the same family of regularization discussed in the
[ML README](README.md) (Ridge/L2), but correctly decoupled for Adam. A common refinement:
**exclude bias and normalization (LayerNorm/BatchNorm) parameters from weight decay** —
decaying them hurts with no benefit.

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim == 1 or name.endswith(".bias"):   # norms + biases
        no_decay.append(p)
    else:
        decay.append(p)
opt = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

## Learning-Rate Schedules

A fixed learning rate is rarely optimal: you want large steps early to make fast progress
and small steps late to settle into a minimum. Schedules vary `η` over the course of
training.

```
η
│       ___________
│      /           \                  warmup ↗ then cosine decay ↘
│     /              \___              — the LLM-standard schedule
│    /                   \___
│   /                        \____
│  /                              \___
└─/──────────────────────────────────▶ step
  └warmup┘
```

- **Warmup** — ramp `η` linearly from ~0 over the first few hundred/thousand steps. Adam's
  second-moment estimate is unreliable early on; warmup prevents the huge, destabilizing
  first updates that otherwise blow up transformer training.
- **Cosine decay** — smoothly anneal from the peak LR to ~0 following a half-cosine. The
  de facto default for pretraining.
- **Linear decay** — simpler straight-line anneal, common for fine-tuning.
- **Step decay** — multiply LR by a factor (e.g. 0.1) at fixed milestones; classic for
  vision.
- **One-cycle** — ramp up then down within a single cycle; enables higher peak LRs.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(opt, start_factor=0.01, total_iters=500)
cosine = CosineAnnealingLR(opt, T_max=total_steps - 500)
sched  = SequentialLR(opt, [warmup, cosine], milestones=[500])

for step, (x, y) in enumerate(loader):
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()
    sched.step()          # advance the schedule every step
```

The [Hugging Face](hugging_face.md) `Trainer` exposes these via
`lr_scheduler_type="cosine"` and `warmup_steps`, and [Unsloth](unsloth.md) /
[LoRA](lora.md) fine-tuning recipes inherit the same knobs.

## Large-Scale & Memory-Efficient Optimizers

Adam keeps **two extra tensors** (`m` and `v`) per parameter, so optimizer state can
*triple* the memory of the weights themselves — a major cost when training large models.
Several optimizers address this:

- **8-bit Adam** (bitsandbytes) — stores `m`/`v` in 8-bit with block-wise
  [quantization](quantization.md), cutting optimizer-state memory ~75% with negligible
  quality loss. Heavily used in [Unsloth](unsloth.md) and QLoRA setups.
- **Adafactor** — factorizes the second-moment matrix into row/column statistics, so
  memory is sub-linear in parameter count. Popularized by T5; good when even 8-bit Adam is
  too heavy.
- **Lion** (EvoLved Sign Momentum) — tracks only momentum and updates via the *sign* of an
  interpolated gradient. Half the state of Adam, competitive results, sensitive to LR
  tuning.
- **Shampoo / Muon** — second-order-ish preconditioning over weight matrices; can improve
  convergence per step at higher compute cost, and have shown strong results in recent
  large-scale training runs.

```python
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, weight_decay=0.1)
```

## Gradient Clipping and Accumulation

Two stability/scaling tricks that live alongside the optimizer:

**Gradient clipping** caps the global gradient norm to prevent exploding gradients
(common in RNNs and early transformer training):

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```

**Gradient accumulation** sums gradients over several micro-batches before stepping,
simulating a larger effective batch size without the memory — essential when a true large
batch won't fit on the GPU:

```python
accum = 4
for i, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y) / accum
    loss.backward()
    if (i + 1) % accum == 0:
        opt.step()
        opt.zero_grad()
```

## Best Practices

- **Default to AdamW** for transformers/LLMs; **SGD+Nesterov momentum** remains a strong,
  better-generalizing choice for CNN vision models.
- **Always warm up** Adam-family optimizers, then **cosine-decay** for pretraining (linear
  decay is fine for short fine-tunes).
- **Exclude biases and norm parameters** from weight decay.
- **Tune the learning rate first** — it is the single most impactful hyperparameter; a
  rough LR-range test beats blind grid search.
- **Use 8-bit Adam or Adafactor** when optimizer state dominates memory.
- **Clip gradients** (`max_norm=1.0`) for training stability.

## Common Pitfalls

- **Learning rate too high** → loss diverges to NaN/Inf. Lower the peak LR or add warmup.
- **No warmup with Adam** → unstable first steps on transformers, often a loss spike then
  divergence.
- **Decaying norm/bias weights** → quietly degraded results; split parameter groups.
- **Using `Adam` instead of `AdamW`** → weight decay gets entangled with the adaptive
  denominator and regularizes weakly.
- **Forgetting `opt.zero_grad()`** → PyTorch *accumulates* gradients across steps, silently
  inflating updates.
- **Calling `sched.step()` at the wrong cadence** → per-epoch vs. per-step mismatch makes
  the LR anneal far too fast or too slow.
- **Resuming training without optimizer state** → reloading only model weights drops `m`/`v`
  and the schedule position, causing a visible loss bump.

## Where this connects

- [Neural networks](neural_networks.md), [Deep learning](deep_learning.md) — the training
  loop optimizers drive
- [Loss functions](loss_functions.md) — the objective whose gradient the optimizer consumes
- [Normalization](normalization.md) — pre/post-norm interacts with warmup; norm params excluded from weight decay
- [PyTorch](pytorch.md) `torch.optim`, [JAX](jax.md) / Optax — the implementations
- [Convolution](convolution.md) — where SGD+momentum still shines
- [Transformers](transformers.md) — where AdamW + warmup + cosine became standard
- [Quantization](quantization.md), [Unsloth](unsloth.md), [LoRA](lora.md) — 8-bit and
  memory-efficient optimizer states for large-model fine-tuning
- [Hugging Face](hugging_face.md) — `Trainer` scheduler/optimizer configuration
- [Fine-tuning](../ai/fine_tuning.md) — choosing LR, schedule, and weight decay in practice
