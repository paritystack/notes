# Regularization & Generalization

What stops a model from memorizing its training set and lets it work on data it has never
seen. **Generalization** is the goal — low error on unseen data — and **regularization** is the
toolbox of techniques that buy it, usually by limiting effective capacity or injecting noise.
This page is the hub: the specific mechanics live on neighboring pages
([optimizers](optimizers.md), [loss functions](loss_functions.md), [normalization](normalization.md)),
and this page ties them together and adds the modern generalization story.

## Table of Contents

1. [Overview](#overview)
2. [The Generalization Gap](#the-generalization-gap)
3. [Penalty-Based: L1 / L2 / Weight Decay](#penalty-based-l1--l2--weight-decay)
4. [Noise-Based: Dropout & Friends](#noise-based-dropout--friends)
5. [Data-Based: Augmentation & Label Smoothing](#data-based-augmentation--label-smoothing)
6. [Training-Based: Early Stopping & Schedules](#training-based-early-stopping--schedules)
7. [Architectural & Implicit Regularization](#architectural--implicit-regularization)
8. [The Modern Picture: Double Descent & Grokking](#the-modern-picture-double-descent--grokking)
9. [Comparison](#comparison)
10. [Where this connects](#where-this-connects)
11. [Pitfalls](#pitfalls)

## Overview

A model that drives training loss to zero hasn't necessarily *learned* anything — it may have
memorized. The [bias–variance tradeoff](README.md) frames the failure modes: too little
capacity **underfits** (high bias), too much **overfits** (high variance), and regularization
shifts a high-variance model back toward the sweet spot without throwing away capacity.

```
   error
     │   ╲                              ╱  validation (test)
     │    ╲                          ╱
     │     ╲__________          ___╱
     │                ╲______╱            ← sweet spot
     │                 ╲
     │                  ╲________________ training
     └──────────────────────────────────► model capacity / training time
          underfit        overfit →
```

Regularization techniques fall into a few families, by *where* they intervene:

```
  penalty   constrain the weights        L1 / L2 / weight decay
  noise     perturb activations          dropout, stochastic depth
  data      enlarge / soften the data    augmentation, label smoothing, mixup
  training  stop or schedule learning    early stopping, LR decay
  implicit  the optimizer/architecture   SGD noise, BatchNorm, residuals
```

## The Generalization Gap

The quantity that matters is the **generalization gap** = (test error − training error). A
small training error with a large gap means overfitting; the whole toolbox below exists to
shrink the gap, ideally without raising training error much.

```
  generalization gap = E_test − E_train

  overfit:  E_train ≈ 0,  E_test high   → big gap   → regularize harder / get more data
  underfit: E_train high, E_test high    → small gap → more capacity / train longer
```

The single most reliable "regularizer" is **more (and more diverse) data** — every technique
below is a stand-in for data you don't have.

## Penalty-Based: L1 / L2 / Weight Decay

Add a term that penalizes large weights, so the optimizer trades a little data-fit for smaller
parameters:

```
  L_total = L_data + λ · Ω(w)
                       ├─ L1: λ·Σ|w|   → sparse weights (feature selection)
                       └─ L2: λ·Σw²    → small, smooth weights ("weight decay")
```

L1 pushes weights to *exactly* zero (sparsity); L2 shrinks them smoothly. The catch — **L2 in
the loss is not the same as weight decay for adaptive optimizers** — is an optimizer concern,
covered in depth at [optimizers · Weight Decay vs. L2](optimizers.md#weight-decay-vs-l2-regularization),
with the loss-side framing on the [loss functions](loss_functions.md#regularization-terms-in-the-loss)
page. Use **AdamW** (decoupled decay) by default.

## Noise-Based: Dropout & Friends

**Dropout** (Srivastava et al., 2014) randomly zeroes a fraction `p` of activations each forward
pass during training, forcing the network not to rely on any single unit — an implicit ensemble
over exponentially many sub-networks.

```
  train:  h_i · Bernoulli(1−p) / (1−p)    drop units, rescale survivors (inverted dropout)
  eval:   h_i                             no dropout, no rescale (full network)
```

The train/eval asymmetry is why you **must** call `model.eval()` at inference. Variants:
**DropConnect** (drop weights, not activations), **Dropout2d** (drop whole channels in convnets),
**stochastic depth** (drop whole residual blocks), and **DropEdge** for
[graph networks](graph_neural_networks.md). Note dropout and [BatchNorm](normalization.md)
interact poorly when stacked naively — a known ordering pitfall.

## Data-Based: Augmentation & Label Smoothing

If you can't get more data, manufacture variety from what you have:

- **Data augmentation** — label-preserving transforms (crop, flip, color jitter for images;
  back-translation for text). The network sees a "new" example each epoch. `mixup`/`CutMix`
  blend two examples *and their labels*, which both augments and smooths decision boundaries.
- **Label smoothing** — replace hard one-hot targets with `(1−ε)` on the true class and `ε`
  spread over the rest, so the model stops chasing infinite-confidence logits:

```
  hard:    [0, 1, 0, 0]
  smooth:  [ε/3, 1−ε, ε/3, ε/3]    typically ε = 0.1
```

Label smoothing improves [calibration](metrics.md) and is standard in transformer training; the
objective-level detail is on the [loss functions](loss_functions.md) page.

## Training-Based: Early Stopping & Schedules

**Early stopping** monitors validation loss and halts when it stops improving — arguably the
cheapest regularizer, since it directly targets the generalization gap by not training into the
overfit regime.

```
  watch val loss; keep the best checkpoint; stop after `patience` epochs of no improvement
```

[Learning-rate schedules](optimizers.md) (warmup + cosine decay) act as soft regularizers too:
a decaying LR settles into flatter, better-generalizing minima. **Gradient clipping** prevents
destabilizing updates that would otherwise demand a tiny, under-regularized LR.

## Architectural & Implicit Regularization

Much regularization isn't an explicit penalty at all — it's baked into the model and optimizer:

- **Parameter sharing** — convolutions (weight sharing across space) and RNNs (across time)
  drastically cut effective parameters versus a dense layer.
- **[Normalization](normalization.md)** — BatchNorm's mini-batch statistics inject noise and
  smooth the loss landscape, which is partly why it regularizes.
- **Residual connections** — ease optimization and, with stochastic depth, regularize directly.
- **Implicit regularization of SGD** — mini-batch gradient noise biases training toward
  **flat minima**, which tend to generalize better than sharp ones. This is why plain SGD often
  generalizes better than full-batch or aggressive adaptive methods, and it's a big part of why
  over-parameterized nets generalize at all.

## The Modern Picture: Double Descent & Grokking

Classical bias–variance predicts test error rises once you overfit. Deep learning violated this,
producing phenomena the toolbox above doesn't fully explain:

```
  DOUBLE DESCENT
  test
  error │  ╱╲                         ___
        │ ╱  ╲          interpolation╱   ╲___  ← 2nd descent: bigger models
        │╱    ╲________ threshold __╱           generalize BETTER
        └──────────────●──────────────────► model size / params
              classical  (train error hits 0 here)
```

- **Double descent** — test error first falls, rises to a peak at the **interpolation
  threshold** (just enough capacity to fit the data exactly), then falls *again* as models grow
  past it. Over-parameterization is not automatically bad.
- **Grokking** — on some tasks a model memorizes training data (train acc 100%, val acc near
  chance) and then, after *much* further training, suddenly generalizes. Weight decay is often
  what eventually tips it into the generalizing solution.

Practical upshot: bigger models + appropriate regularization (especially weight decay) often
beat the "right-sized" model classical theory would pick — the regime modern LLM and vision
training operates in.

## Comparison

| Technique | Family | What it constrains | When to reach for it |
|---|---|---|---|
| **L2 / weight decay** | penalty | weight magnitude | almost always (use AdamW) |
| **L1** | penalty | weight sparsity | want feature selection / sparse model |
| **Dropout** | noise | co-adaptation of units | dense layers, smaller datasets |
| **Stochastic depth** | noise | depth | very deep residual nets |
| **Augmentation / mixup** | data | invariance, boundary | vision/audio, limited data |
| **Label smoothing** | data | logit over-confidence | classification, transformers |
| **Early stopping** | training | training time | always cheap to add |
| **BatchNorm / residuals** | implicit | landscape / optimization | architectural default |

## Where this connects

- [Optimizers](optimizers.md#weight-decay-vs-l2-regularization) — weight decay vs. L2, AdamW,
  LR schedules and gradient clipping as regularizers
- [Loss functions](loss_functions.md#regularization-terms-in-the-loss) — the `data loss + λ·Ω(w)`
  objective and label smoothing
- [Normalization](normalization.md) — BatchNorm's regularizing noise and the dropout-ordering pitfall
- [Deep learning](deep_learning.md) — dropout and weight-decay implementations
- [Metrics](metrics.md) — measuring the generalization gap, calibration, and validation curves
- [Self-supervised learning](self_supervised_learning.md) and [transfer learning](transfer_learning.md)
  — pretraining is itself a powerful regularizer (a strong data prior)
- [Convolution](convolution.md) and [graph neural networks](graph_neural_networks.md) — parameter
  sharing and structural regularization
- [Maths · Optimization](../maths/optimization.md) — flat-minima / implicit-regularization theory

## Pitfalls

- **Tuning regularization on the test set** — that's leakage; use a validation split, keep test
  for the final number.
- **Forgetting `model.eval()`** — leaves dropout and BatchNorm in training mode at inference,
  silently degrading and randomizing outputs.
- **Stacking dropout after BatchNorm** — the variance shift between train and eval hurts; prefer
  one or the other, or order them carefully.
- **Over-regularizing** — too-high dropout/weight decay underfits; the symptom is high training
  *and* validation error (small gap, both bad).
- **L2 with Adam expecting weight decay** — use AdamW; coupled L2 decays large-gradient
  parameters too little.
- **Augmentations that break the label** — over-aggressive crops/rotations can remove the object
  or flip semantics (a "6" rotated into a "9").
- **Assuming bigger = overfit** — double descent says past the interpolation threshold, larger
  models often generalize better; don't shrink reflexively.
- **Early stopping too eagerly** — noisy validation curves; use a patience window and keep the
  best checkpoint, not the last.
