# Guardrails & Moderation

## Overview

If [LLM security](llm_security.md) is the threat model, **guardrails** are the controls: the
input/output checks that keep an LLM application within policy — blocking unsafe content,
catching [prompt injection](llm_security.md), validating structure, and stopping the model from
going off-topic or leaking data. They wrap the model rather than change it, and lean on
[structured outputs](structured_outputs.md) for validation and [LLM evaluation](llm_evaluation.md)
to measure false-positive/negative rates. Think of them as a firewall around the
[LLM](llms.md) / [agent](agent_frameworks.md).

```
  user input ─► [input guardrails] ─► LLM ─► [output guardrails] ─► response
                 PII, injection,            moderation, schema,
                 topic, jailbreak           grounding, leakage
```

## Input vs output guardrails

| Stage | Checks | Action on fail |
|-------|--------|----------------|
| **Input** | injection/jailbreak, off-topic, PII, banned terms | block, strip, reroute |
| **Output** | toxicity, PII leakage, schema validity, hallucination/grounding, brand/policy | block, regenerate, redact |

Input guardrails run *before* the expensive LLM call (cheap to reject early); output guardrails
run *after* and may trigger a regeneration loop.

## What guardrails check

- **Content moderation** — toxicity, hate, self-harm, sexual content, violence. Hosted
  classifiers (OpenAI Moderation, Azure Content Safety, AWS) or model-based (**Llama Guard**,
  **ShieldGemma**).
- **Prompt-injection / jailbreak detection** — classifiers that flag override attempts (see
  [LLM security](llm_security.md)).
- **PII detection & redaction** — tools like **Presidio** to mask emails, SSNs, cards before
  the prompt is stored or sent.
- **Topical / scope** — keep a support bot from answering legal or competitor questions.
- **Format validation** — schema/grammar checks via [structured outputs](structured_outputs.md);
  reject malformed JSON.
- **Grounding / hallucination** — verify the answer is supported by retrieved context (ties to
  [RAG](rag.md) faithfulness in [evaluation](llm_evaluation.md)).

## How they're implemented

```
  Rules/regex     — fast, deterministic (block lists, PII patterns)
  Classifier model— ML check (toxicity, injection) — small + cheap
  LLM-as-judge    — flexible policy check (on-topic? grounded?) — slow + pricey
  Schema/grammar  — structural validity (constrained decoding)
```

Most production systems **layer** these: cheap deterministic checks first, model-based checks
only when needed. Frameworks: **NeMo Guardrails** (NVIDIA, with a rail DSL), **Guardrails AI**
(validators + re-ask), **Llama Guard** (open safety classifier), plus provider moderation APIs.

```
  NeMo-style rails:
    define flow: if input matches "off topic" → refuse politely
    define flow: check output for PII → redact
```

## Where this connects

- [LLM security](llm_security.md) — guardrails implement the defenses; one layer of defense in
  depth, not the whole answer.
- [Structured outputs](structured_outputs.md) — schema validation is an output guardrail.
- [LLM evaluation](llm_evaluation.md) — measure guardrail precision/recall; tune the trade-off.
- [RAG](rag.md) — grounding checks reduce hallucination in retrieval apps.
- [Alignment](alignment.md) — model-level safety (training) vs guardrails (runtime) are
  complementary.

## Pitfalls

- **Guardrails as the only defense.** They're bypassable classifiers; pair with least privilege
  and sandboxing from [LLM security](llm_security.md).
- **Over-blocking.** Aggressive filters frustrate legitimate users (false positives); measure
  and tune against real traffic.
- **Latency/cost creep.** LLM-as-judge on every request doubles cost and latency; reserve it for
  cases cheap checks can't cover.
- **Validating shape, not safety.** A schema-valid output can still be harmful or wrong;
  structure ≠ policy.
- **Static block lists.** Adversaries adapt; rules need maintenance and a model-based backstop.
