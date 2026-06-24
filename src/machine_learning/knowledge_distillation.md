# Knowledge Distillation

Training a small, cheap **student** model to mimic a large, accurate **teacher** — transferring not just the right answers but the teacher's full probability distribution, the "dark knowledge" hidden in its confidences.

## Table of Contents

1. [Overview](#overview)
2. [Soft Targets and Temperature](#soft-targets-and-temperature)
3. [The Distillation Loss in Code](#the-distillation-loss-in-code)
4. [Types of Knowledge](#types-of-knowledge)
5. [Distillation Schemes](#distillation-schemes)
6. [Notable Examples](#notable-examples)
7. [KD vs. Other Compression](#kd-vs-other-compression)
8. [Common Pitfalls](#common-pitfalls)
9. [Where this connects](#where-this-connects)

## Overview

Knowledge distillation (KD) is a model-compression technique: a small **student** network
is trained to reproduce the behavior of a large **teacher** (or an ensemble of teachers),
so you can deploy the cheap student at inference time while keeping most of the teacher's
accuracy. It is the third pillar of compression alongside
[quantization](quantization.md) (fewer bits per weight) and pruning (fewer weights) — and
the three **compose**: a common production recipe is *distill, then quantize*.

The key insight (Hinton et al., 2015) is that a trained teacher's **full output
distribution** carries far more information than the one-hot ground-truth label. When a
classifier says "7% cat, 90% dog, 3% wolf", the relative weights on the wrong classes
encode how the teacher *generalizes* — cats look more dog-like than wolf-like. These soft
probabilities are the "dark knowledge" the student learns from, on top of (or instead of)
the hard labels used in ordinary [supervised learning](supervised_learning.md).

```
            ┌─────────────────────────┐
   input ──▶│   TEACHER (frozen, big)  │──▶ soft logits ─┐
     │      └─────────────────────────┘                 │  KL divergence
     │                                                   ▼  (soften both
     │      ┌─────────────────────────┐            ┌──────────┐ with T)
     └─────▶│   STUDENT (trainable)    │──▶ logits ─┤   loss   │
            └─────────────────────────┘      │     └──────────┘
                                             └──▶ cross-entropy with hard label
```

This is closely related to [transfer learning](transfer_learning.md): both reuse knowledge
from a pretrained model, but transfer learning copies *weights* and fine-tunes, whereas
distillation copies *behavior* into a (usually smaller, often differently-shaped) network.

## Soft Targets and Temperature

A standard softmax produces a peaky distribution — once a model is confident, the wrong
classes get probabilities near zero, so they carry almost no gradient signal. KD raises a
**temperature** `T` inside the softmax to soften the distribution and expose those small
inter-class relationships:

```
              exp(z_i / T)
  p_i(T) = ─────────────────        T = 1 → ordinary softmax
            Σ_j exp(z_j / T)        T > 1 → softer, more informative
```

```
logits z = [4.0, 2.0, 0.5]

 T = 1 (peaky)          T = 4 (soft)
 █                       █
 █                       █  █
 █                       █  █  █
 █  ▂  .                 █  █  █
 ────────                ────────
 the T=1 bars hide the   T=4 reveals the runner-up
 runner-up structure     structure the student learns
```

The student is trained on **two** terms — match the softened teacher, *and* still get the
true label right:

```
L = α · T² · KL( softmax(z_student/T) ‖ softmax(z_teacher/T) )   ← distillation term
  + (1 − α) · CE( softmax(z_student),  y_true )                  ← standard hard-label term
```

The `T²` factor is easy to forget and important: softening by `T` shrinks the gradients of
the soft-target term by roughly `1/T²`, so multiplying by `T²` keeps the two loss terms on
comparable scales as you tune `T`. Typical values: `T` between 2 and 10, `α` around
0.5–0.9. The optimizer and schedule are the same ones used for any
[neural network](neural_networks.md) training — see [optimizers](optimizers.md).

## The Distillation Loss in Code

A complete [PyTorch](pytorch.md) training step. Note `KLDivLoss` expects **log-probabilities**
for the input and **probabilities** for the target, and you want `reduction="batchmean"` to
get a proper mean over the batch:

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # Soft targets: KL between temperature-softened distributions
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher,
                         reduction="batchmean") * (T * T)   # ← the T² factor

    # Hard targets: ordinary cross-entropy with the ground truth
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


teacher.eval()                          # teacher is frozen
for x, labels in loader:
    with torch.no_grad():
        t_logits = teacher(x)           # no gradients through the teacher
    s_logits = student(x)

    loss = distillation_loss(s_logits, t_logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

The teacher runs under `no_grad()` and `eval()` — it is never updated, just queried. Its
forward passes can also be **precomputed once** and cached if the teacher is expensive and
the dataset fits, turning every epoch after the first into pure student training.

## Types of Knowledge

What exactly does the student imitate? Three broad families, increasingly invasive:

- **Response-based** — match the teacher's *output* logits/probabilities. This is classic
  Hinton KD and the snippet above: simple, architecture-agnostic, the default.
- **Feature-based** — match *intermediate* activations ("hints"). FitNets train the student
  to reproduce a teacher hidden layer (with a small projection to reconcile widths). This
  transfers richer representational structure but needs a layer-to-layer mapping.
- **Relation-based** — match *relationships* between examples or layers, e.g. the pairwise
  similarity structure of a batch in feature space, rather than any single activation. The
  student learns the geometry of the teacher's representation.

```
response-based      feature-based            relation-based
   logits           hidden activations       similarity(x_i, x_j)
     ▲                  ▲     ▲                  ▲   ▲   ▲
 [teacher out]     [teacher mid layers]     [pairwise structure]
```

## Distillation Schemes

How teacher and student are trained relative to each other:

- **Offline** — a fully pretrained, frozen teacher; the student trains against its fixed
  outputs. By far the most common and the easiest to reason about.
- **Online** — teacher and student (or several peer students) train *simultaneously*,
  learning from each other's current predictions. Useful when no strong pretrained teacher
  exists (e.g. *deep mutual learning*).
- **Self-distillation** — the teacher and student share the same architecture. A model
  distills from an earlier copy of itself, from its own deeper layers into shallower ones,
  or "born-again networks" where a student of identical capacity surprisingly *outperforms*
  its teacher.

## Notable Examples

KD shows up across vision, NLP, and modern LLMs:

- **DistilBERT** — a 6-layer student of 12-layer BERT, distilled during pretraining. ~40%
  smaller and ~60% faster while retaining ~97% of BERT's GLUE performance. The canonical
  [transformer](transformers.md) distillation; shipped via [Hugging Face](hugging_face.md).
- **TinyBERT** — adds feature-based distillation of attention matrices and hidden states,
  pushing compression further with a two-stage (general + task) distillation.
- **Sequence-level KD** — for translation/seq2seq, the student is trained on the teacher's
  *generated* output sequences (beam-search hypotheses) rather than per-token distributions
  — distillation through the data itself.
- **LLM distillation** — two flavors dominate today. *Hard / data distillation*: a large
  teacher generates instruction-response data used to fine-tune a smaller model (the basis
  of many open "distilled" chat models). *On-policy / logit distillation*: the student
  generates, and is trained to match the teacher's token distribution on its own samples
  (reduces train/inference mismatch). Both pair naturally with
  [LoRA](lora.md)/QLoRA so the student fine-tune stays cheap — see also
  [fine-tuning](../ai/fine_tuning.md).

## KD vs. Other Compression

| Technique | What it reduces | Needs retraining? | Composable with KD |
|---|---|---|---|
| **Knowledge distillation** | model *capacity* (smaller architecture) | yes (train the student) | — |
| [Quantization](quantization.md) | *bits per weight* (FP16→INT8/INT4) | optional (PTQ) or yes (QAT) | yes — distill then quantize |
| Pruning | *number of weights* (sparsity) | usually yes (fine-tune after) | yes — prune then distill back |
| [LoRA](lora.md) | *trainable params* during fine-tune | adds small adapters | yes — LoRA-fine-tune the student |
| [MoE](moe.md) | *active* params per token | architectural | distill a dense student from an MoE teacher |

They attack different axes and stack: e.g. distill a 70B teacher into a 7B student, then
INT4-quantize the student for edge deployment. KD is the only one that changes the model's
*shape*; the others shrink a fixed architecture.

## Common Pitfalls

- **Capacity gap too large** — a tiny student cannot absorb a giant teacher; accuracy
  *drops*. Use an intermediate-size "teacher assistant", or shrink in stages.
- **Forgetting the `T²` factor** — without it, raising `T` quietly shrinks the distillation
  gradient and the hard-label term dominates, undoing the point of soft targets.
- **Wrong `KLDivLoss` arguments** — `F.kl_div` wants `log_softmax` for the input and
  `softmax` for the target; swapping them or using `reduction="mean"` (averages over every
  element, not the batch) silently mis-scales the loss.
- **Distilling a miscalibrated teacher** — the student inherits the teacher's
  overconfidence or biases. A teacher with poor calibration makes for poor soft targets.
- **Mismatched temperature at train vs. inference** — `T` is a *training-time* device; the
  deployed student uses `T = 1`. Leaving `T` in at inference distorts probabilities.
- **Data mismatch** — distilling on data far from the deployment distribution transfers the
  wrong behavior; the student only learns the teacher *where you query it*.
- **Tuning teacher updates** — in offline KD, forgetting `teacher.eval()` /
  `torch.no_grad()` wastes memory and can let BatchNorm/dropout drift the teacher's outputs.

## Where this connects

- [Quantization](quantization.md), [LoRA](lora.md), [MoE](moe.md) — sibling efficiency
  techniques that compose with distillation
- [Transfer learning](transfer_learning.md) — reuse pretrained knowledge by copying weights
  vs. copying behavior
- [Transformers](transformers.md), [Hugging Face](hugging_face.md) — DistilBERT/TinyBERT and
  the libraries that ship distilled models
- [Neural networks](neural_networks.md), [Deep learning](deep_learning.md) — the student
  training loop
- [Supervised learning](supervised_learning.md) — hard labels vs. soft targets
- [Loss functions](loss_functions.md) — the KL-divergence / cross-entropy objective the
  distillation loss is built from
- [Optimizers](optimizers.md) — the optimizer/schedule that drives student training
- [PyTorch](pytorch.md) — `F.kl_div`, `F.cross_entropy` used above
- [Fine-tuning](../ai/fine_tuning.md) — LLM distillation as data generation + small-model
  fine-tune
