# RLHF & Preference Optimization

How a pretrained model that merely *predicts text* becomes one that's *helpful, honest, and harmless* — the alignment pipeline of supervised fine-tuning, reward modeling, and RLHF, and the direct-optimization methods (DPO, GRPO) that increasingly replace the RL loop.

## Table of Contents

1. [Overview](#overview)
2. [The Three-Stage Pipeline](#the-three-stage-pipeline)
3. [Stage 1: Supervised Fine-Tuning (SFT)](#stage-1-supervised-fine-tuning-sft)
4. [Stage 2: The Reward Model](#stage-2-the-reward-model)
5. [Stage 3: PPO — RL from Human Feedback](#stage-3-ppo--rl-from-human-feedback)
6. [DPO: Skipping the Reward Model](#dpo-skipping-the-reward-model)
7. [GRPO and RL for Reasoning](#grpo-and-rl-for-reasoning)
8. [Other Preference Methods](#other-preference-methods)
9. [Comparison](#comparison)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

A [transformer](transformers.md) pretrained on next-token prediction learns to *continue*
text, not to *follow instructions* or *prefer good answers over bad ones*. **Alignment** is
the post-training process that closes that gap, and its dominant recipe is **RLHF**
(Reinforcement Learning from Human Feedback): collect human judgments about which model
outputs are better, distill them into a learned reward, and optimize the model against it.

```
  pretrained LM  ──SFT──►  instruction-following  ──reward model──►  RLHF / DPO  ──►  aligned
  (predicts text)          (does what's asked)     (scores quality)   (prefers good)  assistant
```

The core difficulty is that "good" — helpful, truthful, safe, well-styled — has no
differentiable loss. RLHF's insight (InstructGPT, 2022) is to learn that objective from
**pairwise comparisons** (humans find "A is better than B" far easier than scoring in
isolation), then use [reinforcement learning](deep_reinforcement_learning.md) to push the
policy toward higher reward. The modern wrinkle is that you often don't need the RL machinery
at all: **DPO** optimizes the same preference objective with a simple
[classification-style loss](loss_functions.md), and **GRPO** powers the reasoning-model
("RLVR") wave. This page covers the whole arc.

## The Three-Stage Pipeline

```
  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────────┐
  │ 1. SFT       │──►│ 2. Reward Model   │──►│ 3. RLHF (PPO)           │
  │ demos →      │   │ A>B pairs → scalar│   │ maximize reward,        │
  │ imitate      │   │ reward r(x,y)     │   │ stay near SFT (KL)      │
  └─────────────┘   └──────────────────┘   └────────────────────────┘
        │                                            │
        └──────────── DPO collapses 2+3 into one supervised step ──────────────┘
```

## Stage 1: Supervised Fine-Tuning (SFT)

SFT is ordinary [supervised](supervised_learning.md) next-token training on
high-quality `(prompt, ideal response)` demonstrations. It teaches the base model the
*format* of being an assistant — answering questions, following instructions, using the chat
template — and produces the starting policy for everything downstream.

```
  loss = cross-entropy on the response tokens only (prompt tokens masked out)
```

SFT alone gets you a usable instruction model (this is what "instruct" checkpoints often are),
but it can only imitate the demonstrations it's shown; it has no signal about what *not* to do
or how to rank two plausible answers. That's what preference optimization adds. SFT is
frequently done with parameter-efficient [LoRA](lora.md) to keep it cheap.

## Stage 2: The Reward Model

The reward model (RM) turns human preferences into a differentiable score. Collect pairs where
a human labeled response `y_w` (winner) better than `y_l` (loser) for the same prompt `x`,
then train a model — usually the SFT model with the LM head replaced by a **scalar output** —
so the winner scores higher. The loss is the **Bradley-Terry** pairwise objective:

```
  L_RM = − log σ( r(x, y_w) − r(x, y_l) )

   r(x, y)  = scalar reward (a transformer + linear head)
   σ        = sigmoid; pushes the winner's reward above the loser's
```

```
  human ranks:   y_w  ≻  y_l        (for the same prompt x)
                  │        │
                  ▼        ▼
            r(x,y_w)   r(x,y_l)      train so the gap is large and positive
```

The RM is a learned, automatable proxy for human judgment — it can score *any* new output, far
faster than asking a human. Its quality caps the whole pipeline: a weak or biased RM is the
thing RLHF will ruthlessly exploit.

## Stage 3: PPO — RL from Human Feedback

Now treat generation as an RL problem: the **policy** is the LM, an **action** is emitting a
token, and the **reward** comes from the RM at the end of the response. The standard optimizer
is **PPO** ([Proximal Policy Optimization](deep_reinforcement_learning.md)), with one
critical addition — a **KL penalty** that keeps the policy from drifting too far from the SFT
reference:

```
  objective = E[ r(x, y) ]  −  β · KL( π_θ(y|x) ‖ π_ref(y|x) )
                  ▲                     ▲
            reward-model score    leash to the SFT model (prevents reward hacking)
```

```
  prompt ─► policy π_θ generates y ─► reward model scores r(x,y)
                    ▲                          │
                    │   PPO update (clipped)   │  + KL penalty vs π_ref
                    └──────────────────────────┘   keeps language fluent & on-distribution
```

The KL term is the heart of it: without a leash, the policy quickly finds **degenerate
outputs that the RM scores highly but humans hate** (reward hacking) — repetitive flattery,
keyword stuffing, exploiting RM blind spots. PPO works but is **operationally heavy**: it runs
*four* models at once (policy, reference, reward, and a value/critic), is sensitive to
hyperparameters, and is unstable to tune. That cost is exactly what DPO set out to remove.

## DPO: Skipping the Reward Model

Direct Preference Optimization (Rafailov et al., 2023) is the pivotal simplification. Its
derivation shows the RLHF objective has a **closed-form optimal policy**, which can be
rearranged so the *policy itself implicitly defines the reward* — letting you optimize directly
on the preference pairs with a **simple supervised loss, no RM and no RL loop**:

```
  L_DPO = − log σ( β · [ log π_θ(y_w|x)/π_ref(y_w|x)  −  log π_θ(y_l|x)/π_ref(y_l|x) ] )
                         └──── raise prob of winner ────┘  └──── lower prob of loser ────┘

   π_ref = frozen SFT model;  β controls how far π_θ may move from it (the KL leash, baked in)
```

The same Bradley-Terry preference signal, optimized as a single stable classification-style
objective over `(x, y_w, y_l)` triples. DPO needs only **two** models (policy + frozen
reference), trains like ordinary fine-tuning, and matches or beats PPO on many benchmarks — it
is now the default for open-model preference tuning, often as a [LoRA](lora.md) run.

```
  PPO:  SFT → train RM → RL loop (4 models, unstable)
  DPO:  SFT → one supervised pass on preference pairs (2 models, stable)   ◄ much simpler
```

The trade-off: DPO learns *offline* from a fixed preference set, so it can't explore new
responses the way online PPO can, and it's somewhat sensitive to `β` and to overfitting the
preference data. **Online DPO** and iterative variants reintroduce fresh samples to recover
some of PPO's exploration.

## GRPO and RL for Reasoning

GRPO (Group Relative Policy Optimization, from DeepSeekMath/R1) is the method behind the
reasoning-model wave. It targets **verifiable** domains — math, code, anything with an
automatic correctness check — so the reward is a *program*, not a learned RM: this is **RLVR**,
RL from Verifiable Rewards. Its key efficiency trick is dropping PPO's value/critic network and
estimating the advantage from a **group of samples** for the same prompt:

```
  for each prompt: sample G responses, score each with a rule/verifier
  advantage_i = (reward_i − mean(rewards)) / std(rewards)     ← group-relative baseline
  PPO-style clipped update using these advantages + a KL leash to the reference

  → no separate value model (one fewer network than PPO); reward = correctness check
```

Because the baseline is the group mean, GRPO needs no critic, cutting memory and complexity.
With verifiable rewards there's no RM to hack, which is why it scales to long chain-of-thought
[reasoning](thinking.md). It still requires generating many samples per prompt, so it's
compute-hungry at rollout time.

## Other Preference Methods

- **RLAIF** — replace human labelers with an LLM judge ("Constitutional AI"); scales
  preference collection, at the cost of inheriting the judge's biases.
- **KTO** (Kahneman-Tversky Optimization) — needs only a binary good/bad label per response,
  not pairwise comparisons, so it uses cheaper unpaired data.
- **IPO** — a DPO variant with a different loss that's more robust to overfitting deterministic
  preferences.
- **ORPO** — folds preference optimization *into* SFT with an odds-ratio penalty, removing the
  separate reference model and the two-stage split entirely.
- **Rejection sampling / Best-of-n** — generate many candidates, keep the highest-RM-scoring
  one as new SFT data (the simplest "RLHF-lite", used in LLaMA-2 as an SFT booster).

## Comparison

| Method | Needs reward model? | RL loop? | Models in memory | Notes |
|---|---|---|---|---|
| **SFT** | no | no | 1 | imitation only; the starting point |
| **PPO (RLHF)** | yes | yes | 4 (policy, ref, RM, critic) | powerful, unstable, costly |
| **DPO** | **no** | **no** | 2 (policy, ref) | supervised loss; the open-model default |
| **GRPO (RLVR)** | no (uses a verifier) | yes | 2–3 (no critic) | reasoning models; group baseline |
| **KTO** | no | no | 2 | binary labels, unpaired data |
| **Rejection sampling** | yes (to rank) | no | 1 + RM | best-of-n → new SFT data |

## Best Practices

- **Always SFT first.** Preference methods refine an instruction-following model; they can't
  bootstrap one from a base model.
- **Keep the KL leash (β) honest.** Too small → reward hacking / mode collapse; too large → the
  model barely changes. It's the single most important RLHF/DPO knob.
- **Prefer DPO as the default** for open-model alignment — far simpler and more stable than PPO,
  with comparable quality; reach for PPO only when you need online exploration.
- **Use GRPO/RLVR when you have a verifier** (math, code, unit tests) — a rule-based reward
  can't be hacked the way a learned RM can.
- **Invest in preference-data quality and RM calibration** — the RM (or preference set) is the
  ceiling on everything downstream.
- **Run alignment with [LoRA](lora.md)** to keep SFT/DPO cheap; full fine-tuning is rarely
  needed for preference tuning.
- **Evaluate on held-out preferences and real tasks**, not just reward — rising reward with
  falling human ratings is the classic reward-hacking signature.

## Common Pitfalls

- **Reward hacking** — the policy maximizes the RM while degrading for humans (verbosity,
  sycophancy, keyword stuffing). Tighten KL, improve the RM, or switch to a verifiable reward.
- **KL leash mis-set** — too loose collapses fluency/diversity; too tight learns nothing.
- **Mismatched reference model** — DPO/PPO assume `π_ref` is the SFT model the data was drawn
  near; using a different reference breaks the implicit-reward derivation.
- **Skipping SFT** — running DPO/PPO straight on a base model gives weak, unstable results.
- **Length/verbosity bias** — RMs often reward longer answers; preferences drift toward
  rambling unless you length-normalize or de-bias.
- **Overfitting DPO** — with deterministic or noisy preferences DPO can sharpen too hard; watch
  for collapse and consider IPO or a smaller learning rate.
- **Confusing the PPO here with game-playing PPO** — same algorithm, but the "environment" is a
  reward *model* over text, and the KL-to-reference term is essential, not optional.

## Where this connects

- [Deep reinforcement learning](deep_reinforcement_learning.md) — PPO, policy gradients, and
  advantage estimation, the RL backbone of RLHF/GRPO
- [Reinforcement learning](reinforcement_learning.md) — the policy/reward/value foundations
- [Transformers](transformers.md) — the model being aligned (policy, reward, reference all
  share this architecture)
- [Loss functions](loss_functions.md) — the Bradley-Terry / DPO objectives are sigmoid-based
  pairwise losses
- [LoRA](lora.md) — parameter-efficient SFT and DPO, how alignment is usually run in practice
- [Supervised learning](supervised_learning.md) — SFT is standard supervised next-token training
- [Thinking](thinking.md) — RLVR/GRPO is how chain-of-thought reasoning models are trained
- [LLM decoding & sampling](decoding_sampling.md) — alignment reshapes the distribution that
  decoding then samples from
- [Hugging Face](hugging_face.md) — TRL implements SFT, reward modeling, PPO, DPO, and GRPO
