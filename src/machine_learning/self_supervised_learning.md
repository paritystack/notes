# Self-Supervised Learning

How a model learns useful representations from **unlabeled** data by inventing its own
supervision — masking part of the input and predicting it, or pulling two augmented views of
the same example together in embedding space. SSL is the pretraining engine behind modern
encoders (BERT, the LLMs, CLIP, DINO) that [transfer learning](transfer_learning.md) then
adapts to downstream tasks.

## Table of Contents

1. [Overview](#overview)
2. [Pretext Tasks](#pretext-tasks)
3. [Contrastive Methods](#contrastive-methods)
4. [Non-Contrastive & Self-Distillation](#non-contrastive--self-distillation)
5. [Masked & Generative SSL](#masked--generative-ssl)
6. [Multimodal SSL](#multimodal-ssl)
7. [Heads & Evaluation](#heads--evaluation)
8. [Comparison](#comparison)
9. [Where this connects](#where-this-connects)
10. [Pitfalls](#pitfalls)

## Overview

[Supervised learning](supervised_learning.md) needs a human label for every example;
[unsupervised learning](unsupervised_learning.md) looks for structure with no targets at all.
**Self-supervised learning sits between them**: there are no human labels, but the model still
trains on a *supervised* objective — the labels are derived automatically from the data
itself. Hide a word and predict it; rotate an image and predict the angle; encode two crops of
one photo and make their vectors agree. The "free" signal is everywhere, so SSL scales to the
entire internet without annotation cost.

```
  supervised      (x, y_human)         → expensive labels, limited data
  unsupervised    (x)                  → no targets, e.g. clustering / density
  self-supervised (x → corrupt → x')   → label invented FROM x: predict the missing part
                                          → cheap, web-scale "labels"
```

The product of SSL is not a classifier but an **encoder** that maps inputs to dense
representations. A two-phase recipe dominates the field:

```
  1. pretrain   huge unlabeled corpus  +  pretext task   →  general-purpose encoder
  2. adapt      small labeled set      +  linear probe / fine-tune  →  downstream task
```

This is exactly the [transfer learning](transfer_learning.md) workflow: SSL *produces* the
pretrained weights that fine-tuning then specializes. Pretraining an LLM with next-token
prediction is the most consequential instance of SSL in practice.

## Pretext Tasks

A **pretext task** is a fake supervised problem whose solution forces the network to learn the
structure you actually care about. The label comes from withholding or corrupting part of the
input and asking the model to recover it.

```
   input x ──► corrupt / hide a part ──► x'      (model sees x')
                                          │
                                          ▼  predict the withheld part
                                       target = the part you removed   (free label)
```

Classic pretext tasks:

```
  masking          hide tokens / image patches, predict them      (BERT, MAE)
  next-token       predict the next token given the prefix        (GPT / LLM pretraining)
  jigsaw           shuffle patches, predict their arrangement
  rotation         rotate {0,90,180,270}°, predict the angle
  colorization     greyscale in, predict the colors
```

The task is a *means*, not the goal — once pretraining is done the pretext head is usually
thrown away and only the encoder is kept. Note that LLM pretraining is itself a pretext task
(predict the next [token](tokenization.md)), which is why the line between "self-supervised
learning" and "language-model pretraining" is mostly historical.

## Contrastive Methods

Contrastive SSL learns by comparison: two augmentations of the *same* image form a **positive
pair** that should land close together in embedding space, while everything else in the batch
is a **negative** that should be pushed apart.

```
        image ─┬─ aug₁ ─► encoder ─► z₁ ┐
               │                        ├─ pull together (positive pair)
               └─ aug₂ ─► encoder ─► z₂ ┘
                                         z_other ─ push apart (negatives)
```

**SimCLR** keeps it minimal: strong augmentation, a shared encoder, a small projection head,
and a large batch to supply many in-batch negatives. **MoCo** removes the dependence on huge
batches by maintaining a **momentum encoder** (an exponential moving average of the main
encoder) that fills a **queue/memory bank** of negatives, decoupling the number of negatives
from the batch size.

The objective is **InfoNCE / NT-Xent** — softmax cross-entropy that treats the positive as the
correct "class" among all negatives, over temperature-scaled cosine similarities. The math and
a complete SimCLR training loop already live elsewhere, so this page links rather than repeats:
see the InfoNCE/NT-Xent derivation in [loss functions](loss_functions.md#metric-learning-and-embedding-losses)
and the runnable `SimCLR` module in [transfer learning](transfer_learning.md#self-supervised-pre-training).

```python
import torch, torch.nn.functional as F

def info_nce(z1, z2, temperature=0.5):
    """One positive pair per row; all other rows in the batch are negatives."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2N, d)
    sim = z @ z.T / temperature                           # cosine sim matrix
    sim.fill_diagonal_(float("-inf"))                     # exclude self-similarity
    n = z1.size(0)
    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

Contrastive methods are powerful but **hungry for negatives** and **sensitive to augmentation**
— properties that motivated the next family.

## Non-Contrastive & Self-Distillation

Can you learn good representations *without* negatives? The trivial failure mode is
**representation collapse** — the encoder maps every input to the same constant vector, which
makes positives trivially "agree." Non-contrastive methods avoid collapse with architectural
asymmetry instead of negatives.

```
  BYOL / SimSiam:                          DINO:
  online  ─ aug₁ ─► enc ─► predictor ─► p   student ─► softmax(/τ_s)
                                       │            │ cross-entropy
  target  ─ aug₂ ─► enc' ─► z  (stop-grad)  teacher ─► softmax(/τ_t) ─ centered
            (EMA of online, stop-grad)      (EMA of student, stop-grad)
```

- **BYOL** — two networks, *online* and *target*. The online net has an extra **predictor**
  head and is trained to predict the target net's representation; the target is an EMA
  (momentum) copy and receives a **stop-gradient**. The asymmetry (predictor + EMA) is what
  keeps it off the collapsed solution despite having no negatives.
- **SimSiam** — shows you don't even need the momentum target: a plain Siamese network with a
  predictor on one branch and a **stop-gradient** on the other is enough.
- **DINO** — self-**distillation** with no labels: a student matches a momentum **teacher**'s
  output distribution over a "prototype" space, with **centering + sharpening** of the teacher
  to prevent collapse. Its attention maps segment objects without ever being told to.

The recurring trick is some combination of **stop-gradient, an EMA target, and a predictor or
centering** — each breaks the symmetry that would otherwise let the network cheat by collapsing.
This connects to [knowledge distillation](knowledge_distillation.md), except the teacher here is
a slowly-updated copy of the student rather than a separate, larger model.

## Masked & Generative SSL

The other major family skips comparison entirely and just **reconstructs corrupted input** —
the same idea as denoising autoencoders, scaled up on [transformers](transformers.md).

```
  MAE:   mask 75% of patches ─► tiny encoder sees only the 25% visible
                              ─► lightweight decoder reconstructs the missing pixels
         (asymmetric: heavy encoder on few tokens = cheap pretraining)

  BERT:  mask ~15% of tokens  ─► bidirectional encoder predicts the masked tokens (MLM)
```

- **MAE (Masked Autoencoders)** — mask a *high* fraction of image patches (~75%), encode only
  the visible ones, and let a small decoder rebuild the rest. The high mask ratio makes the
  task hard enough to force semantic learning and makes the encoder cheap (it processes a
  quarter of the patches).
- **BERT-style MLM** — mask ~15% of [tokens](tokenization.md) and predict them with full
  bidirectional context. Contrast with **autoregressive** LM pretraining (GPT), which predicts
  the *next* token using only left context — bidirectional masking is better for understanding
  tasks, causal prediction is what enables generation.

Masked modeling needs no negatives and no augmentation tuning, which is part of why it scales
so cleanly — but its objective is reconstruction, so the representations can be more
low-level than contrastive ones unless the task is made hard (the MAE mask-ratio insight).

## Multimodal SSL

Contrastive learning generalizes beyond a single modality. **CLIP** trains an image encoder and
a text encoder jointly so that an image and its caption land near each other, using
**in-batch** image-text pairs as positives and all other pairings as negatives — InfoNCE across
two towers.

```
  N (image, caption) pairs ─► image enc → I  ▒ matched pairs on the diagonal
                              text  enc → T  ▒ maximize diagonal, minimize off-diagonal
  similarity = I · Tᵀ  →  cross-entropy over rows AND columns
```

The result is a shared embedding space that enables **zero-shot classification** (compare an
image to text prompts like "a photo of a {class}") and powers text-conditioned generation.
This page covers *how the encoder is trained*; for the applied retrieval/similarity side see
[embeddings & reranking](../ai/embeddings.md) and [RAG](../ai/rag.md), and for CLIP-conditioned
image generation see [Stable Diffusion](../ai/stable_diffusion.md).

## Heads & Evaluation

A detail that trips people up: the **projection head** used during pretraining is usually
**discarded** afterward. The representation that transfers best is the encoder output *before*
the projection, not the projected vector the loss was computed on.

```
  encoder ─► h  (representation kept for downstream)
              └─► projection g(h) = z  (used only for the contrastive loss, then thrown away)
```

SSL encoders are evaluated in two standard ways:

- **Linear probe** — freeze the encoder, train only a linear classifier on top. Measures how
  *linearly separable* the learned features are; cheap and a good proxy for representation
  quality.
- **Fine-tune** — unfreeze the encoder and train end-to-end on the downstream task. Higher
  ceiling, but mixes "good features" with "good adaptation," so it's a weaker measure of the
  representation alone. See [transfer learning](transfer_learning.md) for the full freeze-vs-
  fine-tune spectrum.

## Comparison

| Family | Examples | Needs negatives? | Anti-collapse mechanism | Pretext |
|---|---|---|---|---|
| **Contrastive** | SimCLR, MoCo | yes (batch / memory bank) | negatives | augment two views, InfoNCE |
| **Non-contrastive** | BYOL, SimSiam, DINO | no | stop-grad + EMA / predictor / centering | predict other view's embedding |
| **Masked / generative** | MAE, BERT-MLM | no | n/a (reconstruction) | reconstruct masked input |
| **Multimodal contrastive** | CLIP | yes (in-batch pairs) | negatives | align image ↔ text |
| **Autoregressive** | GPT pretraining | no | n/a | predict next token |

## Where this connects

- [Transfer learning](transfer_learning.md) — SSL produces the pretrained encoder that
  fine-tuning then adapts; the freeze-vs-fine-tune choices live there
- [Loss functions](loss_functions.md#metric-learning-and-embedding-losses) — the InfoNCE/NT-Xent
  and triplet/contrastive objective math, not repeated here
- [Unsupervised learning](unsupervised_learning.md) — the no-human-labels neighbor; SSL adds a
  self-generated supervised target
- [Transformers](transformers.md) and [tokenization & embeddings](tokenization.md) — masked and
  autoregressive LM pretraining are SSL; the encoder this page trains is the embedding layer
- [Knowledge distillation](knowledge_distillation.md) — DINO/BYOL self-distill from a momentum
  teacher rather than a separate large model
- [Convolution](convolution.md) and [deep learning](deep_learning.md) — the CNN/ViT backbones
  that SSL pretrains
- [Embeddings & reranking](../ai/embeddings.md) and [RAG](../ai/rag.md) — applied use of CLIP /
  sentence encoders downstream
- [Stable Diffusion](../ai/stable_diffusion.md) — CLIP text encoders condition image generation
- [Interesting papers](interesting_papers.md#self-supervised-learning) — SimCLR / BYOL / MAE
  summaries

## Pitfalls

- **Representation collapse** — every input maps to the same vector. The central risk for
  non-contrastive methods; guard with stop-gradient, an EMA target, a predictor, or centering.
- **Augmentation sensitivity** — contrastive results swing hard on the augmentation recipe; too
  weak and the task is trivial, too strong and positives stop being semantically equivalent.
- **Negatives ≈ batch size** — naive contrastive needs many negatives, so it craves large
  batches; MoCo's memory bank / momentum encoder exists precisely to break that coupling.
- **False negatives** — two different images of the same class are treated as negatives and
  pushed apart, fighting the very structure you want.
- **Keeping the projection head** — transferring the projected `z` instead of the encoder output
  `h` hurts; discard the head for downstream use.
- **Linear-probe vs. fine-tune mismatch** — a method can win on linear probe but lose after
  fine-tuning (or vice-versa); report the protocol that matches your deployment.
- **Mask ratio too low (masked SSL)** — an easy reconstruction task learns low-level texture,
  not semantics; MAE's 75% mask ratio is high on purpose.
