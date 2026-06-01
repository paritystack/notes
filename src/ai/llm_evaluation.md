# LLM Evaluation

## Overview

How do you know an LLM application got *better* after a prompt tweak, model swap, or
[fine-tune](fine_tuning.md)? Unlike traditional software, outputs are open-ended and
non-deterministic, so "does it pass?" needs a deliberate evaluation strategy. This page covers
**benchmarks** (how models are compared) and **evals** (how *your* application is measured),
including **LLM-as-judge**. It's the measurement counterpart to
[LLM observability](llm_observability.md) (which captures production traces to evaluate) and
draws on classic [metrics](../machine_learning/metrics.md). Good evals depend on
[structured outputs](structured_outputs.md) for programmatic grading and drive
[prompt engineering](prompt_engineering.md) and [coding-agent](coding_agents.md) progress.

```
  Benchmarks: compare MODELS on shared tasks   (is GPT-5 > Claude on coding?)
  Evals:      measure YOUR APP on YOUR tasks    (did v2 of my prompt regress?)
```

## Benchmarks (model-level)

Standard leaderboards for capability comparison:

| Benchmark | Measures |
|-----------|----------|
| **MMLU / MMLU-Pro** | broad knowledge across 57 subjects |
| **GPQA** | graduate-level science (hard, google-proof) |
| **HumanEval / MBPP** | code generation (pass@k on unit tests) |
| **SWE-bench (Verified)** | real GitHub issue resolution ([coding agents](coding_agents.md)) |
| **MATH / AIME** | competition mathematics ([reasoning](reasoning_models.md)) |
| **MT-Bench / Chatbot Arena** | conversation quality (Arena uses human pairwise votes) |

Caveats: **contamination** (benchmarks leak into training data, inflating scores) and
**overfitting** to the leaderboard. Treat benchmarks as a rough prior, not proof your use case
will work.

## Evals (application-level)

The part that actually matters for shipping. Build a **dataset** of representative inputs with
known-good expectations, then score outputs:

```
  inputs ─► your app ─► outputs ─► scorer ─► aggregate metric
            (prompt/             (exact match,
             chain/agent)         rubric, judge)
```

Scoring methods, weakest assumption to strongest:

- **Exact / structural match** — deterministic; works for classification, extraction
  ([structured outputs](structured_outputs.md)), code-that-runs-tests.
- **Reference-based metrics** — BLEU/ROUGE/semantic similarity vs a gold answer; cheap but
  crude for open text.
- **LLM-as-judge** — an LLM grades the output against a rubric (see below).
- **Human review** — gold standard, slowest; reserve for calibration and high-stakes cases.

## LLM-as-judge

Use a strong model to score outputs — the only scalable way to grade open-ended generation.

```
  judge prompt: "Score 1–5 for faithfulness to the context. Rubric: …
                 Output: <answer>  Context: <docs>"  ──► score + rationale
```

Make it reliable:
- **Pairwise > pointwise** — "is A or B better?" is more stable than absolute scores.
- **Rubrics + chain-of-thought** — force reasoning before the score.
- **Known biases** — position bias (favoring the first option), verbosity bias (longer = better),
  self-preference (favoring its own family). Randomize order; calibrate against human labels.
- For [RAG](rag.md): measure **faithfulness** (grounded in context?), **answer relevance**, and
  **context relevance** — the **RAGAS** framework formalizes these.

## Tooling & process

Frameworks: **OpenAI Evals**, **LangSmith**, **Braintrust**, **promptfoo**, **DeepEval**,
**RAGAS**, **lm-evaluation-harness**. The durable practice matters more than the tool:

- Start with **~20 real failure cases**, not a giant synthetic set.
- Run evals in **CI** so prompt/model changes are gated like code.
- Add every production bug to the eval set (regression suite).
- Track **online** signals too: thumbs, edits, task completion (see
  [observability](llm_observability.md)).

## Where this connects

- [LLM observability](llm_observability.md) — production traces become eval data; eval-in-prod.
- [Metrics](../machine_learning/metrics.md) — precision/recall/F1 and reference metrics.
- [Structured outputs](structured_outputs.md) — make outputs programmatically gradable.
- [Prompt engineering](prompt_engineering.md) / [fine-tuning](fine_tuning.md) — evals are how
  you tell whether changes help.
- [RAG](rag.md) — faithfulness/relevance evals (RAGAS) for retrieval pipelines.

## Pitfalls

- **No eval set at all.** "Looks good in the playground" doesn't catch regressions; build a
  dataset early.
- **Trusting benchmark scores for your task.** Contamination and domain mismatch make
  leaderboards weak predictors of real performance.
- **Naïve LLM-as-judge.** Position/verbosity/self-preference biases skew scores unless you
  control for them.
- **Optimizing the metric, not the outcome.** A high ROUGE or judge score can diverge from user
  value (Goodhart's law).
- **Tiny or unrepresentative datasets.** Evals only generalize if the data mirrors real usage.
