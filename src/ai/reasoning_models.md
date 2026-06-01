# Reasoning Models & Test-Time Compute

## Overview

A class of [LLMs](llms.md) — OpenAI's o-series, [DeepSeek R1](deepseek_r1.md), Gemini Thinking,
Claude's extended thinking, Qwen QwQ — trained to **think before answering**, spending extra
compute at inference to reason through hard problems. This page is the *applied model landscape*
and the **test-time compute** idea behind it; the broader reasoning theory (chain-of-thought,
reasoning architectures) lives in [machine_learning/thinking](../machine_learning/thinking.md),
and the training relies on [RL](../machine_learning/reinforcement_learning.md) and
[alignment](alignment.md). It reframes the [prompt-engineering](prompt_engineering.md) habit of
"let's think step by step" as something baked into the model.

```
  Standard LLM:   prompt ──────────────► answer        (fixed compute)
  Reasoning model: prompt ─► [long internal reasoning] ─► answer
                                  (variable, scalable compute)
```

## Test-time compute scaling

The headline insight: you can buy better answers with **more inference compute**, not just more
training. Letting a model generate a long chain of reasoning (or explore many) before answering
improves accuracy on math, code, and logic — a new scaling axis alongside model size and data.

```
  Train-time scaling:  bigger model + more data  → smarter base
  Test-time scaling:   more reasoning tokens / samples per query → better answers
```

How the extra compute is spent:
- **Long chain-of-thought** — generate a long internal reasoning trace, then the final answer.
- **Self-consistency** — sample many reasoning paths, take the majority answer.
- **Search / best-of-N** — generate candidates, score with a verifier/reward model, pick best.
- **Reflection** — critique and revise an answer before committing.

## How they're trained

The breakthrough ([DeepSeek R1](deepseek_r1.md) showed it openly) is **RL with verifiable
rewards**: for math/code, correctness is checkable, so the model is rewarded for reaching the
right answer regardless of *how* — and it *learns* to produce long, effective reasoning on its
own.

```
  generate reasoning + answer ─► check answer (tests / math) ─► reward
        ▲                                                          │
        └──────────── RL updates toward what works ◄────────────────┘
  emergent: longer reasoning, self-correction, backtracking
```

This differs from [alignment](alignment.md)'s preference tuning: the reward is *objective
correctness*, not human preference. See [RL](../machine_learning/reinforcement_learning.md).

## When to use one

| Reach for a reasoning model | Use a standard model |
|-----------------------------|----------------------|
| math, logic, hard coding/debugging | chat, summarization, extraction |
| multi-step planning | simple, latency-sensitive tasks |
| tasks where correctness > speed/cost | high-volume, cheap calls |

Trade-offs: reasoning tokens cost money and **latency** (seconds of "thinking"), and on easy
tasks they **overthink** with no benefit. Many providers expose an **effort/budget** knob to
cap thinking. Prompting differs too — reasoning models often want *less* hand-holding (don't
force your own step-by-step; let them reason).

## Where this connects

- [DeepSeek R1](deepseek_r1.md) — the open-weight reasoning model; concrete instance of this
  page.
- [machine_learning/thinking](../machine_learning/thinking.md) — chain-of-thought and reasoning
  theory.
- [Reinforcement learning](../machine_learning/reinforcement_learning.md) / [alignment](alignment.md)
  — RL with verifiable rewards trains the reasoning.
- [Coding agents](coding_agents.md) — reasoning + verification loops power hard SWE tasks.
- [LLM evaluation](llm_evaluation.md) — GPQA/AIME/SWE-bench measure reasoning gains.

## Pitfalls

- **Using them for everything.** Overkill on simple tasks — slower, pricier, no quality gain.
- **Over-prompting.** Forcing your own chain-of-thought can fight a model trained to reason its
  own way; keep instructions lighter.
- **Ignoring latency.** Seconds of thinking break real-time UX; budget the effort knob.
- **Trusting the visible reasoning.** Shown "thinking" may be summarized or post-hoc — it's not
  a faithful audit trail of how the answer was derived.
- **Assuming reasoning fixes hallucination.** It reduces logic errors, not necessarily factual
  ones; grounding ([RAG](rag.md)) is still needed.
