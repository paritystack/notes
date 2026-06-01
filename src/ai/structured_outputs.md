# Structured Outputs

## Overview

Getting an LLM to return data your program can parse — valid JSON matching a schema, a specific
enum, a function-call payload — *reliably*, not just usually. This is distinct from
[tool use](tool_use.md): tool use is the model *deciding to call* a function; structured output
is *constraining the text it generates* to a shape. The two overlap because function-calling
APIs use schema-constrained generation under the hood. The enforcement happens during decoding,
so it ties into [inference optimization](inference_optimization.md), and well-shaped outputs
are the foundation of [guardrails](guardrails.md) and reliable [agents](agent_frameworks.md).

```
  Prompt-and-pray            Constrained decoding
  ───────────────            ────────────────────
  "return JSON"              grammar masks invalid tokens at each step
  parse → maybe fail         parse → always valid by construction
  retry on error             no retry needed
```

## The spectrum of reliability

From weakest to strongest guarantee:

| Approach | Guarantee | How |
|----------|-----------|-----|
| Prompting ("respond in JSON") | none | hope + retries |
| JSON mode | valid JSON syntax | provider restricts to JSON grammar |
| Schema / structured outputs | matches *your* schema | provider enforces a JSON Schema |
| Constrained decoding (local) | any formal grammar | you supply a grammar/regex |

## Constrained decoding — how it actually works

At each step the model produces a probability over the whole vocabulary. A **grammar** (derived
from your schema) computes which tokens are *legal* next given what's been generated, and the
sampler **masks** everything else to zero before picking. The output literally cannot violate
the structure.

```
  state: {"name": "Ada", "age":
  legal next tokens: digits only  ──► mask out  ", letters, { etc.
  ⇒ guaranteed to continue with a number
```

Libraries: **Outlines**, **llama.cpp GBNF grammars**, **XGrammar**, **lm-format-enforcer**.
Local runners ([local inference](local_inference.md)) expose this directly; vLLM supports
guided decoding via `guided_json` / `guided_regex` / `guided_grammar`.

## Provider APIs

**OpenAI** — `response_format` with a JSON Schema and `strict: true` guarantees a conforming
object:

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "person", "strict": True,
                                     "schema": {...}}})
```

**Anthropic** — there's no separate JSON mode; the idiomatic path is [tool use](tool_use.md):
define a tool whose input schema *is* your output schema and force that tool. The tool input
comes back as a validated object.

**Instructor** — a popular library that wires a **Pydantic** model to any provider: you get a
typed object back, with automatic validation and re-ask on failure.

```python
import instructor
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-4o", response_model=Person,
    messages=[{"role": "user", "content": "Ada Lovelace, 36"}])
# person is a typed Person, validated
```

## Structured output vs. tool use

```
  Tool use:           model chooses WHEN to call + WITH WHAT args  (agency)
  Structured output:  model is FORCED to emit a given shape         (formatting)
  Forced tool call = structured output (single tool, tool_choice="required")
```

Use structured outputs for extraction, classification, and data pipelines. Use [tool
use](tool_use.md) when the model should decide whether and which action to take. See
[agent frameworks](agent_frameworks.md) for combining both in a loop.

## Where this connects

- [Tool use](tool_use.md) — function calling is schema-constrained generation; the two share
  machinery.
- [Inference optimization](inference_optimization.md) — grammar masking happens in the sampler
  during decode.
- [Guardrails](guardrails.md) — schema validation is the first line of output checking.
- [Local inference](local_inference.md) — GBNF grammars and Outlines give full local control.
- [LLM evaluation](llm_evaluation.md) — structured outputs make programmatic grading easy.

## Pitfalls

- **Over-constraining.** Forcing a rigid schema can hurt reasoning quality; let the model
  think in a free-text field first, then emit the structured part.
- **Schema ≠ semantics.** Constrained decoding guarantees *shape*, not *correctness* — the JSON
  validates but the values can still be wrong or hallucinated.
- **Deep/recursive schemas.** Some providers cap nesting or reject certain JSON Schema features
  (`strict` mode forbids many); check support before designing.
- **Retries that mask the real problem.** If you need many re-asks, the prompt or schema is the
  issue, not luck.
- **Confusing JSON mode with schema mode.** Plain JSON mode guarantees syntax only, not that
  your fields are present.
