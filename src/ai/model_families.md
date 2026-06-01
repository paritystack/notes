# Model Families

## Overview

A practical map of the major [LLM](llms.md) families — who makes them, whether they're
open-weight or API-only, and how to choose. The space moves fast, so this favors *durable
distinctions* (provider, openness, size tiers, specialization) over leaderboard snapshots, which
go stale in weeks (see [LLM evaluation](llm_evaluation.md) for how they're compared). It ties
together the model-specific pages here — [Llama](llama.md), [Phi](phi.md),
[DeepSeek R1](deepseek_r1.md) — and informs [local inference](local_inference.md) (what you can
run) and cost/routing in [LLM observability](llm_observability.md).

```
  Two worlds:
  API-only (closed weights)        Open-weight (downloadable)
  GPT · Claude · Gemini            Llama · Mistral · Qwen · Gemma · DeepSeek · Phi
  frontier, hosted, per-token      self-host, fine-tune, run local
```

## Closed / API-only

| Family | Maker | Notes |
|--------|-------|-------|
| **GPT / o-series** | OpenAI | broad; o-series are [reasoning models](reasoning_models.md) |
| **Claude** | Anthropic | strong coding/agentic, long context, [tool use](tool_use.md); powers [Claude Code](cli.md) |
| **Gemini** | Google | natively multimodal, very long context, tight GCP integration |
| **Grok** | xAI | frontier contender |

Frontier capability, no hardware needed, but you pay per token and data leaves your boundary
(mind [LLM security](llm_security.md)).

## Open-weight

| Family | Maker | Notes |
|--------|-------|-------|
| **Llama** | Meta | the open default; huge ecosystem — see [Llama](llama.md) |
| **Mistral / Mixtral** | Mistral AI | efficient; Mixtral popularized [MoE](../machine_learning/moe.md) |
| **Qwen** | Alibaba | strong multilingual + coding; wide size range incl. QwQ reasoning |
| **Gemma** | Google | open siblings of Gemini; good small models |
| **DeepSeek** | DeepSeek | V3 (large MoE) + [R1](deepseek_r1.md) reasoning, open + cheap |
| **Phi** | Microsoft | small models, big quality-per-param — see [Phi](phi.md) |

Open weights let you [self-host](local_inference.md), [fine-tune](fine_tuning.md), and inspect —
at the cost of running the infrastructure. Check the **license**: some ("open weight") restrict
commercial use or scale; not all are truly open source.

## Size tiers & selection

```
  small (1–8B)    on-device, cheap, fast       Phi, Gemma, Llama-8B, Qwen-small
  mid (8–70B)     workhorse, self-hostable     Llama-70B, Mixtral, Qwen-32B
  frontier (100B+ / API) hardest tasks         GPT, Claude, Gemini, DeepSeek-V3
```

Choosing is a multi-objective trade-off, not "best model":

```
  capability ◄──► cost ◄──► latency ◄──► privacy/control
```

- **Match the model to the task** — a small model for classification/extraction, a frontier or
  [reasoning model](reasoning_models.md) for hard coding/math.
- **Route**, don't standardize — cheap model first, escalate on failure ([observability](llm_observability.md)
  shows where).
- **Open vs API** — pick open when privacy, cost-at-scale, fine-tuning, or offline use dominate;
  API when you want the frontier with zero ops.
- **Watch architecture trends** — [MoE](../machine_learning/moe.md) (sparse, cheap inference for
  big capacity) and long-context are now common across families.

## Where this connects

- [LLMs](llms.md) — the general foundation these families instantiate.
- [Llama](llama.md) / [Phi](phi.md) / [DeepSeek R1](deepseek_r1.md) — deep dives on specific
  families.
- [Local inference](local_inference.md) — which open models fit your hardware.
- [Reasoning models](reasoning_models.md) — the reasoning tier within several families.
- [LLM evaluation](llm_evaluation.md) — how families are benchmarked (with caveats).

## Pitfalls

- **Chasing the leaderboard.** Rankings churn weekly and are gameable; validate on *your*
  [evals](llm_evaluation.md), not a benchmark.
- **"Open" ≠ open source.** Read the license — usage, commercial, and scale restrictions vary.
- **One model for everything.** Over-paying a frontier model for trivial calls, or under-serving
  hard tasks with a small one; route by difficulty.
- **Ignoring total cost.** Self-hosting trades per-token fees for GPU ops and engineering time —
  not automatically cheaper.
- **Pinning to a version forever.** Families iterate fast; budget for periodic re-evaluation and
  migration.
