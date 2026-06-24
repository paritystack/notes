# LLM Decoding & Sampling

How an autoregressive model turns a vector of next-token logits into actual text — the deterministic searches (greedy, beam), the stochastic samplers (temperature, top-k, top-p, min-p), the penalties that curb repetition, and speculative decoding, the trick that makes generation faster without changing the output distribution.

## Table of Contents

1. [Overview](#overview)
2. [From Logits to a Distribution](#from-logits-to-a-distribution)
3. [Greedy & Beam Search](#greedy--beam-search)
4. [Temperature](#temperature)
5. [Top-k, Top-p, and Min-p](#top-k-top-p-and-min-p)
6. [Repetition Penalties](#repetition-penalties)
7. [Speculative Decoding](#speculative-decoding)
8. [Constrained & Structured Decoding](#constrained--structured-decoding)
9. [Comparison](#comparison)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

A [transformer](transformers.md) language model is a **next-token predictor**: given a
prefix, it emits a vector of logits — one score per [vocabulary token](tokenization.md) — and
the *decoding strategy* decides which token to actually pick. That token is appended, fed back
in, and the loop repeats. Decoding is therefore a separate, post-training choice that has
enormous influence on output quality, diversity, and cost — the *same* model can sound robotic
or creative, terse or rambling, purely by how you sample.

```
  prefix ──► transformer ──► logits (V,) ──► decoder ──► next token ──┐
     ▲                                       (greedy / sample / beam)  │
     └─────────────────── append, repeat ────────────────────────────┘
                          (each step reuses the KV cache — see attention.md)
```

The strategies split into **deterministic search** (greedy, beam — best for tasks with a
single right answer) and **stochastic sampling** (temperature + top-k/p — best for open-ended
generation). On top of those sit *penalties* (anti-repetition) and *acceleration* (speculative
decoding, which changes speed, not the distribution). Each step reuses the [KV
cache](attention.md), so the marginal cost is one forward pass per generated token.

## From Logits to a Distribution

The raw logits become probabilities through a softmax. Decoding strategies all operate on this
distribution — by taking its argmax, reshaping it, or sampling from it:

```
  p_i = softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

  z   : logits over the whole vocabulary  (one per token)
  p   : probability the model assigns to each token being next
```

Everything below is a recipe for collapsing `p` into a single chosen token.

## Greedy & Beam Search

**Greedy** decoding takes the single highest-probability token at every step. It's fast and
deterministic, but myopic — a locally optimal token can doom the rest of the sequence, and it
tends to produce bland, repetitive text on open-ended prompts.

**Beam search** keeps the `B` most probable *partial sequences* (beams) at each step, expanding
all of them and pruning back to the top `B` by cumulative log-probability:

```
  beam width B = 3, scoring by Σ log p:

  step t        step t+1 (expand each beam, keep best 3 overall)
  ┌ The cat        The cat sat   (-2.1)  ◄ keep
  ├ The dog   ──►  The cat ran   (-2.4)  ◄ keep
  └ A bird         The dog ran   (-2.6)  ◄ keep
                   …others pruned…
```

Beam search finds higher-probability sequences than greedy and shines on **closed-ended**
tasks — translation, summarization, constrained generation — where there's a single best
answer. On **open-ended** generation it backfires: high-probability text is often dull and
degenerate (the "neural text degeneration" finding), and beams collapse to near-duplicates.
Use a **length penalty** to stop it favoring short sequences, since `Σ log p` shrinks with
length.

## Temperature

Temperature `T` rescales the logits *before* softmax, sharpening or flattening the
distribution:

```
  p_i = softmax(z / T)_i

  T → 0    distribution spikes on the argmax        → approaches greedy (deterministic)
  T = 1    unchanged (the model's native distribution)
  T > 1    distribution flattens                    → more random, more diverse
  T → ∞    approaches uniform                        → gibberish
```

Temperature is the master diversity knob. Low `T` (0.0–0.3) for factual/code tasks where you
want the most likely answer; moderate `T` (0.7–1.0) for creative writing. It is almost always
combined with a truncation sampler (top-k/p) so that flattening doesn't hand probability mass
to the long tail of nonsense tokens.

## Top-k, Top-p, and Min-p

Pure temperature sampling can still pick from thousands of low-probability tokens. Truncation
samplers cut the tail *before* sampling:

```
  top-k      keep the k highest-prob tokens, renormalize, sample
             ── fixed count; can be too many when one token dominates,
                too few when the distribution is flat

  top-p      keep the smallest set whose cumulative prob ≥ p (nucleus sampling),
  (nucleus)     renormalize, sample
             ── adaptive size: tight when the model is confident, wide when unsure

  min-p      keep tokens with prob ≥ (min_p × p_max)
             ── threshold relative to the top token; robust at high temperature
```

```
  distribution:  ▁▂▆█▃▁▁▁▁▁     "█" = most likely token

  top-k=2   → keep {█, ▆}                       (always 2)
  top-p=0.9 → keep {█, ▆, ▃} until cumsum≥0.9   (adaptive)
  min-p=0.1 → keep tokens ≥ 0.1·p(█)            (relative floor)
```

**Top-p (nucleus)** is the most widely used default for chat — its adaptive cutoff matches the
model's confidence. A typical creative-writing recipe is `temperature=0.8, top_p=0.95`.
**Min-p** is newer and pairs well with higher temperatures because its threshold scales with
the peak. These compose: temperature reshapes, then top-k/p/min-p truncates, then you sample.

## Repetition Penalties

Autoregressive models fall into loops ("the the the", repeated phrases). Several penalties
discourage this by down-weighting tokens already seen:

```
  repetition_penalty  divide logits of already-generated tokens by r (>1) before softmax
  presence_penalty    subtract a flat amount from any token seen at least once
  frequency_penalty   subtract proportional to how often the token has appeared
  no_repeat_ngram     hard-ban any n-gram that already occurred
```

`presence`/`frequency` penalties (OpenAI-style) are additive on logits; `repetition_penalty`
(HF-style) is multiplicative. Use them sparingly — too strong and the model avoids necessary
words (articles, a subject's name) and degrades fluency.

## Speculative Decoding

Generation is **memory-bandwidth bound**: each token needs a full forward pass that mostly
reloads weights from memory (see [attention](attention.md) and the KV cache). Speculative
decoding (Leviathan et al., 2023) exploits the fact that a big model can *verify* several
tokens in **one** forward pass, even though it can only *generate* one at a time.

```
  1. a small, fast DRAFT model proposes k tokens cheaply:   t1 t2 t3 t4
  2. the big TARGET model scores all k in ONE forward pass (parallel)
  3. accept the longest prefix that matches the target's own distribution;
     reject at the first disagreement and resample that one token from the target
  4. repeat from the accepted point

  → if the draft is usually right, you get ~2–3 tokens per big-model pass
    instead of 1 — a real speedup with NO change to the output distribution
```

The crucial guarantee: the accept/reject rule (a form of rejection sampling) makes the output
**distributionally identical** to sampling from the target model alone — it's a pure latency
optimization, not an approximation. Variants avoid a separate draft model: **Medusa** adds
extra prediction heads, **EAGLE** drafts in feature space, and **n-gram / prompt lookup**
drafts by copying from the context. Serving stacks like [vLLM](../ai/vllm.md) implement these;
see [inference optimization](../ai/inference_optimization.md).

## Constrained & Structured Decoding

When the output must obey a schema (JSON, a regex, a grammar, a fixed set of choices), you can
**mask the logits** at each step to allow only tokens that keep the output valid:

```
  at each step:  logits[token not allowed by grammar/state] = −inf   →  softmax → sample
                 (a finite-state machine / grammar tracks what's legal next)
```

This guarantees well-formed output (valid JSON every time) with zero retries — the model
physically cannot emit an illegal token. Libraries like Outlines, `guidance`, and XGrammar
compile a JSON schema or regex into the per-step mask. The trade-off: over-constraining can
hurt quality by forcing the model off its natural distribution, so constrain structure, not
content.

## Comparison

| Strategy | Deterministic? | Diversity | Best for | Typical setting |
|---|---|---|---|---|
| **Greedy** | yes | none | short factual answers, code | — |
| **Beam search** | yes | low | translation, summarization | `B=4`, length penalty |
| **Temperature** | no | tunable | the master knob | `0.0–0.3` factual, `0.7–1.0` creative |
| **Top-k** | no | medium | simple truncation | `k=40` |
| **Top-p (nucleus)** | no | adaptive | **default for chat** | `p=0.9–0.95` |
| **Min-p** | no | adaptive | high-temperature sampling | `0.05–0.1` |
| **Speculative** | matches target | n/a (speed) | latency, unchanged output | draft model / Medusa |

## Best Practices

- **Match the strategy to the task.** Greedy/low-temperature for code, math, extraction,
  classification; temperature + top-p for chat and creative text; beam search for
  translation/summarization.
- **A sensible chat default is `temperature≈0.7, top_p≈0.9`** — adjust temperature first.
- **Set `temperature=0` (or greedy) for reproducibility** and evaluation; sampling makes
  outputs non-deterministic.
- **Combine truncation with temperature**, not temperature alone — top-p/min-p cut the tail
  that flattening would otherwise expose.
- **Reach for speculative decoding for latency**, knowing it doesn't change the output
  distribution; pick a draft model from the same family/tokenizer.
- **Use constrained decoding for machine-readable output** (JSON/regex) instead of parsing-and-
  retrying.
- **Always set a stop condition** — `<eos>`, a max-tokens cap, and any stop strings.

## Common Pitfalls

- **Beam search on open-ended generation** — produces bland, repetitive, degenerate text;
  sample instead.
- **High temperature with no truncation** — flattening hands mass to the nonsense tail; add
  top-p/min-p.
- **Forgetting `temperature=0` isn't fully deterministic across hardware** — kernel/reduction
  order can still differ; pin the backend for exact reproducibility.
- **Over-strong repetition penalties** — the model starts dodging necessary words and grammar
  breaks down.
- **Mismatched draft/target tokenizers in speculative decoding** — the verification step is
  meaningless if the two models tokenize differently; they must share a vocabulary.
- **Ignoring the chat template / stop tokens** — generation runs past `<eos>` or never emits it,
  producing run-on or truncated output (see [tokenization](tokenization.md)).
- **Assuming speculative decoding changes quality** — it doesn't; if outputs differ from plain
  sampling beyond RNG, the accept/reject implementation is buggy.

## Where this connects

- [Transformers](transformers.md) — the model that produces the next-token logits
- [Tokenization & embeddings](tokenization.md) — logits are over the vocabulary; `<eos>` and
  chat templates govern when generation stops
- [Attention](attention.md) — the KV cache makes each decoding step O(1) in past tokens;
  bandwidth-bound decoding is what speculative decoding attacks
- [vLLM](../ai/vllm.md), [Inference optimization](../ai/inference_optimization.md) — where
  speculative decoding, batching, and sampling are implemented in serving
- [Local inference](../ai/local_inference.md) — sampling parameters exposed by local runtimes
- [RLHF & preference optimization](rlhf.md) — preference tuning shifts the very distribution
  decoding samples from
- [Hugging Face](hugging_face.md) — `model.generate(...)` and its `GenerationConfig`
