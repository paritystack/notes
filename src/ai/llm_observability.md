# LLM Observability & LLMOps

## Overview

Once an LLM app is in production, you need to see what it's actually doing: which prompts ran,
what the model returned, how many tokens it cost, where latency went, and why a given answer was
wrong. **LLM observability** is tracing and monitoring adapted to non-deterministic,
multi-step LLM systems; **LLMOps** is the broader operational practice around it. It's the
production feedback loop that feeds [LLM evaluation](llm_evaluation.md) (traces become eval
cases), exposes the cost/latency that [inference optimization](inference_optimization.md) and
[prompt caching](prompt_caching.md) address, and is essential for debugging
[agents](agent_frameworks.md) and [multi-agent systems](multi_agent_systems.md).

```
  request ─► [retrieve] ─► [LLM call] ─► [tool] ─► [LLM call] ─► response
              │             │            │          │
              └──── one trace, nested spans, each timed + costed ────┘
```

## Why LLM observability is different

Traditional APM assumes deterministic, fast, cheap function calls. LLM systems break all three:

- **Non-deterministic** — same input, different output; you must inspect actual generations, not
  just status codes.
- **Multi-step** — a request fans out into retrieval, several LLM calls, and tool uses; you need
  the whole **trace**, not one log line.
- **Token-priced** — cost is per-token and variable, so it must be tracked as a first-class
  metric.
- **Quality is fuzzy** — "200 OK" says nothing about whether the answer was good (that's
  [evaluation](llm_evaluation.md)).

## What to capture

```
  Trace  ── the full request
   └─ Span ── one step (LLM call, retrieval, tool)
        ├─ inputs (prompt, messages, params)
        ├─ outputs (completion, tool result)
        ├─ tokens (prompt / completion / cached)
        ├─ latency (TTFT, total)
        ├─ cost (derived from tokens × model price)
        └─ metadata (model, user, session, version)
```

Key metrics: **token usage & cost**, **latency** (especially **TTFT** — time to first token —
for streaming UX), **error/timeout rate**, **cache hit rate** (see
[prompt caching](prompt_caching.md)), and **quality signals** (eval scores, user feedback).

## Tooling

**Langfuse** (open-source), **LangSmith** (LangChain), **Arize Phoenix**, **Helicone**,
**Braintrust**, **W&B Weave**. **OpenTelemetry** has emerging GenAI semantic conventions, so
LLM traces increasingly flow into standard observability backends. Most SDKs auto-instrument
common frameworks; otherwise you wrap calls with a decorator/callback.

## The LLMOps loop

Observability closes the dev↔prod loop:

```
  prod traces ─► find failures ─► add to eval set ─► fix prompt/model
       ▲                                                    │
       └──────────── deploy + monitor ◄──────────────────────┘
```

Operational concerns beyond tracing: **prompt versioning** (treat prompts as deployable
artifacts), **cost controls** (budgets, rate limits, model routing — cheap model first,
escalate on failure), **caching** ([prompt caching](prompt_caching.md) + response cache), and
**online evals** (sample prod traffic, score with [LLM-as-judge](llm_evaluation.md)).

## Where this connects

- [LLM evaluation](llm_evaluation.md) — observability supplies the data; evals score it.
- [Inference optimization](inference_optimization.md) / [prompt caching](prompt_caching.md) —
  the levers you pull when traces show cost/latency problems.
- [Agent frameworks](agent_frameworks.md) / [multi-agent systems](multi_agent_systems.md) —
  tracing is how you debug opaque multi-step loops.
- [Model families](model_families.md) — cost/quality data informs model routing.

## Pitfalls

- **Logging only status codes.** You must capture actual inputs/outputs to debug LLM behavior;
  metrics alone don't explain a bad answer.
- **Ignoring cost until the bill.** Token spend compounds fast in agent loops; track per-request
  cost from day one.
- **Logging secrets/PII.** Prompts and completions often contain sensitive data — redact before
  storing, and mind retention (ties to [LLM security](llm_security.md)).
- **Monitoring without evals.** Traces tell you *what* happened, not whether it was *good*; pair
  with [evaluation](llm_evaluation.md).
- **No prompt versioning.** Without it you can't correlate a quality regression with the change
  that caused it.
