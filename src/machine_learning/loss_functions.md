# Loss Functions

The objective a model actually minimizes — the single scalar that turns "how wrong is this prediction" into a gradient the optimizer can follow.

## Table of Contents

1. [Overview](#overview)
2. [Loss vs. Metric](#loss-vs-metric)
3. [Regression Losses](#regression-losses)
4. [Classification Losses](#classification-losses)
5. [Cross-Entropy Deep-Dive](#cross-entropy-deep-dive)
6. [Imbalance and Hard-Example Losses](#imbalance-and-hard-example-losses)
7. [Metric-Learning and Embedding Losses](#metric-learning-and-embedding-losses)
8. [Generative and RL Objectives](#generative-and-rl-objectives)
9. [Regularization Terms in the Loss](#regularization-terms-in-the-loss)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

A loss function `L(ŷ, y)` maps a model's prediction `ŷ` and the target `y` to a single
non-negative scalar that the training loop drives toward zero. It is the *objective* —
everything else in the loop serves it. The forward pass produces predictions, the loss
scores them, [backpropagation](neural_networks.md) differentiates that score with respect
to every weight, and the [optimizer](optimizers.md) takes a step opposite the gradient:

```
  ┌──────────────┐  ŷ      ┌──────────────┐  L (scalar)   ┌──────────────┐
  │ forward pass │ ──────▶ │ LOSS L(ŷ, y) │ ────────────▶ │   backprop   │
  └──────────────┘         └──────────────┘   ∂L/∂ŷ       │  ∂L/∂w        │
        ▲                                                  └──────┬───────┘
        │                        ┌──────────────┐                │ gradients
        └──────── next batch ────│  optimizer   │◀───────────────┘
                                 └──────────────┘  w ← w − η·∇L
```

Because the gradient flows *through* the loss, the loss must be **differentiable** (almost
everywhere) with respect to the model output, and its shape determines what the model
prioritizes — penalize large errors quadratically and the model chases outliers; penalize
them linearly and it ignores them. Choosing the loss is therefore a modeling decision, not
a detail: it encodes what "good" means for the task. The objective also depends on the
learning paradigm — labeled targets in [supervised learning](supervised_learning.md),
reconstruction or likelihood in [generative models](generative_models.md), and reward in
[reinforcement learning](reinforcement_learning.md). In [PyTorch](pytorch.md) these live in
`torch.nn` (e.g. `nn.MSELoss`, `nn.CrossEntropyLoss`) and the functional forms in
`torch.nn.functional`.

## Loss vs. Metric

Loss and [metric](metrics.md) are easy to conflate but play different roles:

| | Loss | Metric |
|---|---|---|
| Purpose | what the optimizer minimizes during training | how *you* judge the model afterward |
| Constraint | must be differentiable (drives gradients) | can be anything, even non-differentiable |
| Examples | cross-entropy, MSE, focal | accuracy, F1, BLEU, AUC, mAP |
| Audience | the optimizer | humans / stakeholders |

You optimize a *surrogate* loss because the metric you truly care about is often a step
function with zero gradient almost everywhere. Accuracy, for instance, doesn't change as a
logit nudges from 0.7 to 0.8, so its gradient is useless — cross-entropy is the smooth
proxy you minimize *instead*, and accuracy is what you report. When the two diverge (loss
dropping while validation F1 stalls) it usually signals the surrogate is misaligned with
the goal — a class-imbalance or threshold problem. See [metrics](metrics.md) for the
evaluation side; this page is about the training objective.

## Regression Losses

Regression predicts a continuous value, so the loss measures distance between `ŷ` and `y`.
The choice hinges almost entirely on **how you want to treat outliers**.

```
  loss
   │        MSE (L2) ─ quadratic, outlier-sensitive
   │       ╱
   │      ╱  ╭── Huber ── quadratic near 0, linear in the tails
   │     ╱ ╭╯
   │    ╱╭╯╭──────  MAE (L1) ── linear, outlier-robust, kink at 0
   │   ╱╭╯╭╯
   └──┴┴┴────────────────▶  error (ŷ − y)
```

- **MSE / L2** — `mean((ŷ − y)²)`. The default. Smooth everywhere; its gradient `2(ŷ − y)`
  grows with the error, so a few large residuals dominate training. Optimal under Gaussian
  noise (it's the maximum-likelihood loss for it) and predicts the conditional **mean**.
- **MAE / L1** — `mean(|ŷ − y|)`. Robust to outliers — every residual contributes a
  constant-magnitude gradient `±1`, so one bad label can't dominate. Predicts the
  conditional **median**. Downside: the non-smooth kink at zero slows final convergence.
- **Huber / Smooth-L1** — quadratic for `|error| < δ`, linear beyond. Best of both: smooth
  near the optimum like MSE, robust in the tails like MAE. `δ` sets the crossover; widely
  used in object-detection box regression.
- **Log-cosh** — `log(cosh(ŷ − y))`. A smooth, twice-differentiable Huber-like loss with no
  `δ` to tune.

```python
import torch.nn as nn

mse   = nn.MSELoss()          # L2 — outlier-sensitive, predicts the mean
mae   = nn.L1Loss()           # L1 — robust, predicts the median
huber = nn.HuberLoss(delta=1.0)  # quadratic inside δ, linear outside
```

## Classification Losses

Classification predicts a probability distribution over discrete classes, and the canonical
objective is **cross-entropy** — the negative log-likelihood of the correct class.

**Binary cross-entropy (BCE).** For a single yes/no target `y ∈ {0,1}` and predicted
probability `p`:

```
BCE = −[ y·log(p) + (1−y)·log(1−p) ]
```

In practice, never feed a sigmoid'd probability into BCE — pass **raw logits** to
`BCEWithLogitsLoss`, which fuses the sigmoid and the log via the log-sum-exp trick for
numerical stability (avoids `log(0) = −∞` and overflow). This also gives you `pos_weight`
for imbalanced positives.

**Categorical cross-entropy.** For `K` classes with one-hot target and predicted
distribution `p = softmax(z)`:

```
CE = −Σ_k y_k · log(p_k)   =   −log(p_correct)   (since y is one-hot)
```

PyTorch's `nn.CrossEntropyLoss` takes **logits and an integer class index** — it applies
`log_softmax` + `NLLLoss` internally. Do **not** softmax beforehand:

```python
import torch.nn as nn

# Binary: raw logits in, never sigmoid first
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))  # 3:1 positive weighting

# Multi-class: logits of shape (N, K), integer targets of shape (N,)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = ce(logits, target_indices)   # NOT softmax(logits)
```

## Cross-Entropy Deep-Dive

Cross-entropy is the workhorse of classification and language modeling, and it's worth
seeing *why*. Minimizing cross-entropy is equivalent to **maximum-likelihood estimation**:
`−log p_correct` is exactly the negative log-likelihood of the data under the model. It is
also **KL divergence** up to a constant:

```
CE(y, p) = H(y) + KL(y ‖ p)
```

Since the entropy `H(y)` of a fixed target is constant, minimizing cross-entropy minimizes
`KL(y ‖ p)` — the model is being pulled to match the target distribution. This is the same
machinery [knowledge distillation](knowledge_distillation.md) uses, except there the target
is the teacher's *soft* distribution (with a temperature) rather than a one-hot label.

**Next-token prediction.** [Transformers](transformers.md) and other language models train
on token-level cross-entropy: at each position, predict the distribution over the
vocabulary and penalize `−log p(correct_next_token)`. Averaged and exponentiated, this loss
*is* **perplexity** — the headline LM metric — which is why CE is also the natural objective
for [fine-tuning](../ai/fine_tuning.md).

**Label smoothing.** Replacing the hard one-hot target with `(1−ε)` on the true class and
`ε/K` spread over the rest discourages the model from driving the correct logit to `+∞`. It
improves calibration and generalization at a small cost in raw accuracy — standard in modern
image and language training.

## Imbalance and Hard-Example Losses

When classes are skewed (1% positives) or most examples are trivially easy, plain
cross-entropy lets the majority/easy cases dominate the gradient:

- **Class-weighted CE** — multiply each class's loss by a weight (`weight=` in
  `CrossEntropyLoss`, `pos_weight=` in `BCEWithLogitsLoss`), typically inversely
  proportional to class frequency.
- **Focal loss** — `−(1 − p_t)^γ · log(p_t)`. The `(1 − p_t)^γ` factor down-weights
  already-confident (easy) examples so training focuses on hard ones; the workhorse of
  dense object detection. `γ = 2` is the common default.
- **Dice / Tversky loss** — derived from set-overlap (F1-like) scores, popular in
  segmentation where foreground pixels are rare; optimizes overlap directly rather than
  per-pixel likelihood.

```
focal modulation:   (1 − p_t)^γ
  p_t = 0.99 (easy, confident)  →  factor ≈ 0.0001   ← almost ignored
  p_t = 0.50 (hard, uncertain)  →  factor ≈ 0.25     ← keeps full attention
```

## Metric-Learning and Embedding Losses

Instead of predicting a label, these losses shape an **embedding space** so that distance
encodes similarity — the basis for retrieval, [embeddings](../ai/embeddings.md), and
face/speaker verification.

- **Contrastive loss** — pulls matching pairs together, pushes non-matching pairs apart
  beyond a margin `m`.
- **Triplet loss** — over `(anchor, positive, negative)`: enforces
  `d(a, p) + margin < d(a, n)`. Effectiveness hinges on **hard-negative mining**.
- **InfoNCE / NT-Xent** — the contrastive-learning workhorse (SimCLR, CLIP): treat the
  matching pair as the positive among a batch of negatives and apply cross-entropy over
  cosine similarities scaled by a temperature. Self-supervised pretraining at scale.

```
triplet:   d(anchor, positive)  +  margin   <   d(anchor, negative)
           ╰──── pull closer ────╯               ╰─ push away ─╯
```

## Generative and RL Objectives

Beyond supervised targets, whole model families are defined by their loss. Brief pointers:

- **Adversarial (GAN)** — a min-max game: the generator minimizes what the discriminator
  maximizes. See [generative models](generative_models.md) and
  [deep generative models](deep_generative_models.md).
- **VAE / ELBO** — maximize the evidence lower bound = reconstruction term + a
  `KL(q ‖ prior)` regularizer that keeps the latent space well-behaved.
- **Diffusion** — train a network to predict the noise added at a random timestep; the loss
  is a simple MSE between true and predicted noise (see `../ai/stable_diffusion.md`).
- **Policy gradient / PPO** — maximize expected reward; PPO clips the probability ratio to
  bound each update. See [reinforcement learning](reinforcement_learning.md) and
  [deep reinforcement learning](deep_reinforcement_learning.md). RLHF reuses this machinery
  with a learned reward model.

## Regularization Terms in the Loss

The training objective is often `data loss + λ · regularizer`. The regularizer doesn't look
at the labels — it constrains the weights themselves:

```
L_total = L_data(ŷ, y)  +  λ · Ω(w)
                            ╰── L1: λ·Σ|w|   (sparsity)
                                L2: λ·Σw²    (small weights)
```

- **L1 penalty** drives weights to exactly zero → sparse models / feature selection.
- **L2 penalty** shrinks weights smoothly → the classic "weight decay" effect.

A subtlety: adding an L2 term to the loss is **not** identical to weight decay once you use
an adaptive optimizer like Adam — `AdamW` decouples the two. That distinction lives in the
[optimizers](optimizers.md) page (Weight Decay vs. L2 Regularization); it's an optimizer
concern, not a loss-shape one.

## Best Practices

- **Match the loss to the noise/error model**: Gaussian noise → MSE; heavy-tailed or
  outlier-prone → MAE/Huber; probabilistic classification → cross-entropy.
- **Pass logits, not probabilities** to `CrossEntropyLoss` / `BCEWithLogitsLoss` — the
  fused versions are numerically stable.
- **Reach for the metric you care about**: if you'll report F1 on imbalanced data, train
  with weighted/focal CE, not vanilla CE.
- **Add label smoothing** (`ε ≈ 0.1`) for large-scale classification and LM training.
- **Watch the loss *and* a real metric** during training — a falling loss with a flat
  metric means the surrogate is misaligned.
- **Scale multi-term losses**: when summing several objectives, normalize or tune their
  relative weights so one doesn't swamp the others.

## Common Pitfalls

- **Double softmax** → passing `softmax(logits)` into `nn.CrossEntropyLoss`, which softmaxes
  again. Symptom: training barely moves. Pass raw logits.
- **Wrong target shape** → `CrossEntropyLoss` wants integer class **indices** `(N,)`, not
  one-hot `(N, K)`; mixing these throws or silently mistrains.
- **Wrong `reduction`** → `'mean'` vs `'sum'` vs `'none'` changes the effective learning
  rate; `'sum'` makes the loss scale with batch size.
- **`log(0) = −∞` / NaNs** → hand-rolling `log(p)` on a sigmoid output near 0 or 1.
  Use the `*WithLogits` variants.
- **Mismatched `pos_weight` / class weights** → weighting the wrong direction worsens the
  imbalance instead of fixing it; sanity-check on a tiny batch.
- **Optimizing a non-differentiable metric** → you can't backprop through accuracy or BLEU;
  minimize a differentiable surrogate and *report* the metric.
- **Forgetting to ignore padding** → in sequence models, include `ignore_index` so pad
  tokens don't contribute to the loss.

## Where this connects

- [Optimizers](optimizers.md) — the loss produces the gradient the optimizer consumes; also
  hosts the weight-decay-vs-L2 distinction
- [Neural networks](neural_networks.md), [Deep learning](deep_learning.md) — backprop
  differentiates the loss through the network
- [Supervised learning](supervised_learning.md) — where most regression/classification
  losses live
- [Metrics](metrics.md) — the evaluation counterpart you report but usually can't optimize
  directly
- [Knowledge distillation](knowledge_distillation.md) — KL-divergence soft-target loss with
  temperature
- [Transformers](transformers.md) — next-token cross-entropy and perplexity
- [Generative models](generative_models.md), [Deep generative models](deep_generative_models.md)
  — adversarial and ELBO objectives
- [Reinforcement learning](reinforcement_learning.md), [Deep reinforcement learning](deep_reinforcement_learning.md)
  — reward-based / policy-gradient objectives
- [PyTorch](pytorch.md) — `torch.nn` loss modules and `torch.nn.functional`
- [Embeddings](../ai/embeddings.md), [Fine-tuning](../ai/fine_tuning.md) — contrastive losses
  and the CE objective used in practice
