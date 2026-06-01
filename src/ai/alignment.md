# Alignment (Applied)

## Overview

A base [LLM](llms.md) trained only to predict the next token is a capable but unhelpful
text-completer — it doesn't follow instructions, refuse harmful requests, or match a preferred
style. **Alignment** is the post-training that turns it into an assistant: making the model
*helpful, honest, and harmless*. This page is the applied tour of the techniques — **SFT**,
**RLHF**, **DPO**, **Constitutional AI** — and how they relate to [fine-tuning](fine_tuning.md)
and runtime [guardrails](guardrails.md). The reinforcement-learning theory lives in
[RL](../machine_learning/reinforcement_learning.md); parameter-efficient methods in
[LoRA](../machine_learning/lora.md).

```
  pretraining ─► SFT ─► preference tuning (RLHF / DPO) ─► aligned model
  next-token    follow   learn human preferences,
  prediction    examples  refuse, match style
```

## The post-training pipeline

**1. Supervised fine-tuning (SFT).** Train on curated (prompt → ideal response) pairs so the
model learns the *format* of being an assistant — following instructions, using a chat template.
This is ordinary [fine-tuning](fine_tuning.md), often with [LoRA](../machine_learning/lora.md).

**2. Preference tuning.** SFT teaches one good answer; preference tuning teaches *which of two
answers is better*, capturing nuance SFT can't (tone, helpfulness, safety). Two main approaches:

### RLHF — Reinforcement Learning from Human Feedback

```
  1. Humans rank model outputs (A > B)         → preference dataset
  2. Train a REWARD MODEL to predict that ranking
  3. RL (PPO) optimizes the LLM to maximize reward,
     with a KL penalty so it stays near the SFT model
```

The KL penalty prevents **reward hacking** — drifting into gibberish that games the reward
model. Powerful but complex: a separate reward model, an unstable RL loop, four models in memory.
(RL details: [reinforcement learning](../machine_learning/reinforcement_learning.md).)

### DPO — Direct Preference Optimization

```
  preference pairs (chosen, rejected) ──► single loss directly on the LLM
  no reward model, no RL loop
```

DPO derives a loss that pushes up the probability of *chosen* responses and down the *rejected*
ones, achieving RLHF-like results with a fraction of the complexity. It's now the common default;
variants include **IPO**, **KTO**, and **ORPO** (which folds SFT and preference tuning into one
step).

## Constitutional AI & RLAIF

Human labeling is the bottleneck. **RLAIF** (RL from *AI* Feedback) replaces human rankers with
an LLM judge. **Constitutional AI** (Anthropic) goes further: the model critiques and revises its
own outputs against a written set of principles (a "constitution"), generating its own preference
data.

```
  response ─► model critiques it vs the constitution ─► model revises ─► preference pair
  scales alignment without human labels for every example
```

## Alignment vs guardrails

```
  Alignment  — bakes behavior INTO the weights (training time)  → robust, costly to change
  Guardrails — checks AROUND the model (runtime)                → flexible, bypassable
```

Both are needed: an aligned model refuses most harm on its own; [guardrails](guardrails.md)
catch what slips through and enforce app-specific policy. Neither is complete alone (see
[LLM security](llm_security.md)).

## Where this connects

- [Fine-tuning](fine_tuning.md) — SFT is the first alignment stage; same machinery.
- [LoRA](../machine_learning/lora.md) — parameter-efficient SFT/DPO.
- [Reinforcement learning](../machine_learning/reinforcement_learning.md) — the theory behind
  RLHF/PPO.
- [Guardrails](guardrails.md) / [LLM security](llm_security.md) — runtime safety complements
  trained alignment.
- [Reasoning models](reasoning_models.md) — RL is also used to train reasoning, not just
  preferences.

## Pitfalls

- **Reward hacking.** Without the KL constraint, RLHF optimizes the reward model into nonsense;
  preference signals are proxies, not the true goal.
- **Alignment tax.** Heavy alignment can dent raw capability; balance helpfulness against
  safety.
- **Over-refusal.** Aggressive harmlessness training makes the model decline benign requests —
  a real UX cost.
- **Preference data quality.** Garbage rankings → garbage alignment; the dataset is the product.
- **Treating alignment as sufficient for security.** It reduces but doesn't eliminate jailbreaks;
  pair with [guardrails](guardrails.md) and least privilege.
