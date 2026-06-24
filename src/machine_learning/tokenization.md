# Tokenization & Embeddings

How raw text becomes the integer IDs and dense vectors a transformer actually consumes — the subword algorithms (BPE, WordPiece, Unigram/SentencePiece), the vocabulary that sits between them and the model, and the embedding table that turns IDs into geometry.

## Table of Contents

1. [Overview](#overview)
2. [Why Subwords](#why-subwords)
3. [Byte-Pair Encoding (BPE)](#byte-pair-encoding-bpe)
4. [Byte-Level BPE](#byte-level-bpe)
5. [WordPiece](#wordpiece)
6. [Unigram & SentencePiece](#unigram--sentencepiece)
7. [Special Tokens & the Chat Template](#special-tokens--the-chat-template)
8. [The Embedding Table](#the-embedding-table)
9. [Tied Embeddings](#tied-embeddings)
10. [Comparison](#comparison)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)
13. [Where this connects](#where-this-connects)

## Overview

A [transformer](transformers.md) never sees text. It sees a sequence of **integer token
IDs**, looks each one up in an **embedding table** to get a dense vector, adds a
[positional encoding](positional_encoding.md), and runs [attention](attention.md) over the
result. Tokenization is the front door — the deterministic, *non-learned* map from a string
to that ID sequence — and the embedding table is the first learned layer that turns IDs into
geometry the network can manipulate.

```
  "tokenizing"
       │  tokenizer  (BPE / WordPiece / Unigram — fixed, not trained with the model)
       ▼
  [token, izing]  →  IDs [1923, 4382]
       │  embedding lookup  (learned: nn.Embedding(vocab, d_model))
       ▼
  vectors (2, d_model)  →  + positional encoding  →  transformer stack
```

The vocabulary size `V` is one of the most consequential hyperparameters in the whole model:
it sets the width of the embedding table and the output softmax (`V × d_model` parameters
*each*), the average tokens-per-word (hence effective context length and inference cost), and
how gracefully the model handles rare words, code, and other languages. Modern tokenizers are
almost all **subword** schemes — a middle ground between character-level (tiny vocab, very
long sequences) and word-level (huge vocab, brittle on anything unseen).

## Why Subwords

The two naive extremes both fail:

```
  word-level       vocab = every word         → V in the millions, every typo / new
                                                 word / rare inflection is <UNK>
  character-level  vocab = ~256 bytes          → no <UNK> ever, but sequences are
                                                 5–10× longer → quadratic attention pain
  subword          frequent words stay whole,  → small fixed V (~30k–256k), no <UNK>,
                   rare ones split into pieces    sequences only modestly longer
```

Subword tokenization gets the best of both: common words (`the`, `running`) are single
tokens, while rare or novel strings (`antidisestablishmentarianism`, `f1ngetune`) decompose
into known pieces, so the model can still represent them. The vocabulary is **learned once**
from a training corpus by a frequency-driven algorithm, then frozen and shipped with the
model.

## Byte-Pair Encoding (BPE)

BPE (Sennrich et al., 2016, adapted from a compression algorithm) builds the vocabulary by
**greedily merging the most frequent adjacent pair** of symbols, repeatedly, until it hits
the target vocab size.

```
start: characters       l o w </w>  l o w e r </w>  n e w e s t </w>
                        (</w> marks a word boundary)

count pairs, merge the most frequent:
  step 1   most frequent pair = (e, s)   →  merge to "es"
  step 2   most frequent pair = (es, t)  →  merge to "est"
  step 3   most frequent pair = (l, o)   →  merge to "lo"
  …
vocab = base chars + every merge, in order
```

Encoding a new word replays the **learned merge list in priority order**: split into
characters, then apply merges greedily. The merge rules are deterministic, so encoding is
reproducible. GPT-2/3/4 and LLaMA all use BPE variants. The downside in its original form:
it operates on Unicode characters, so a character it never saw at training (an unusual emoji,
a rare script) has no representation — which byte-level BPE fixes.

## Byte-Level BPE

Byte-level BPE (GPT-2's contribution) runs BPE over the **256 raw UTF-8 bytes** instead of
Unicode characters. Because every possible string is *already* a sequence of bytes, the base
alphabet is complete:

```
  base vocab = 256 bytes  →  ANY string is representable, <UNK> is impossible
  merges then build up common byte sequences into single tokens, exactly like BPE
```

This guarantees **zero out-of-vocabulary** for any input — text, code, emoji, binary-ish
junk — at the cost of a few extra tokens for non-Latin scripts (whose multi-byte UTF-8
characters start as 2–4 byte tokens). It is the de-facto standard for modern LLMs.

## WordPiece

WordPiece (used by BERT) is BPE's close cousin. The structural difference: instead of merging
the *most frequent* pair, it merges the pair that most increases the **likelihood of the
training corpus** under a unigram language model — roughly, it picks the merge with the best
`count(AB) / (count(A)·count(B))` score rather than raw `count(AB)`.

```
  BPE:       merge argmax  count(A, B)
  WordPiece: merge argmax  count(A,B) / (count(A) · count(B))   ← favors informative pairs

  continuation pieces are marked, e.g.  "playing" → ["play", "##ing"]
```

The `##` prefix marks a piece that continues the previous token (no leading space), which is
how the tokenizer distinguishes `un` the word from `un` the prefix.

## Unigram & SentencePiece

The **Unigram** model (Kudo, 2018) inverts the bottom-up approach. It starts from a *large*
candidate vocabulary and **prunes** tokens that contribute least to corpus likelihood, keeping
a probabilistic model where each token has a probability and a word can be segmented multiple
ways. At encoding time it picks the **most probable segmentation** (Viterbi), and at training
time it can *sample* segmentations (subword regularization) for robustness.

**SentencePiece** is the *tooling*, not an algorithm — a library (Google) that runs BPE or
Unigram directly on **raw text with no pre-tokenization**, treating the input as a stream of
Unicode and encoding the space itself as a visible meta-symbol `▁`:

```
  "Hello world"  →  ["▁Hello", "▁world"]      (▁ = U+2581, marks a leading space)

  language-agnostic: no whitespace splitting needed → works for Chinese/Japanese/Thai
  reversible: detokenize = concatenate and replace ▁ with space → exact original text
```

This whitespace-as-symbol trick makes SentencePiece **fully reversible** (lossless
round-trip) and language-agnostic, which is why T5, LLaMA, and most multilingual models use
it.

## Special Tokens & the Chat Template

Beyond content tokens, the vocabulary reserves **special tokens** the model is trained to
treat structurally:

```
  <bos> / <s>        beginning of sequence
  <eos> / </s>       end of sequence — the model emits this to STOP generating
  <pad>              padding to batch unequal-length sequences (masked out)
  <unk>              unknown (effectively unused with byte-level BPE)
  chat roles         <|system|> <|user|> <|assistant|> <|im_start|> …
```

Instruction-tuned models wrap conversations in a **chat template** — a model-specific string
format that interleaves these role markers. Using the wrong template (or none) at inference is
a leading cause of degraded chat quality; always apply the tokenizer's own
`apply_chat_template`. The [Hugging Face](hugging_face.md) tokenizer ships the correct one.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

ids = tok("tokenizing is fun")["input_ids"]      # encode → list of ints
text = tok.decode(ids)                            # decode → string

msgs = [{"role": "user", "content": "Hi"}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
```

## The Embedding Table

The first learned layer maps each ID to a `d_model`-dimensional vector via a lookup table —
literally a `V × d_model` matrix indexed by the integer ID:

```python
import torch.nn as nn
emb = nn.Embedding(vocab_size, d_model)   # the table: row i is token i's vector
x = emb(input_ids)                        # (batch, seq) ints → (batch, seq, d_model)
```

Lookup is just indexing (no matmul), and the rows are trained by backprop like any other
parameter — semantically related tokens drift toward nearby vectors. For a 128k-vocab,
4096-wide model this single table is `128k × 4096 ≈ 525M` parameters — often the largest
single layer, which is why large-vocab models lean on **tied embeddings**.

> Note: this **input** embedding (one vector per *vocabulary token*) is different from a
> sentence/document **embedding model** (one vector per *text*, for retrieval/similarity) used
> in [RAG](../ai/rag.md). Same word, different object.

## Tied Embeddings

The output layer of a language model is also a `V × d_model` matrix — it projects the final
hidden state back to a logit per vocabulary token. **Weight tying** shares one matrix for
both the input embedding and this output projection:

```
  input:   ids ──[ E ]──► hidden          E : (V, d_model)
  output:  hidden ──[ Eᵀ ]──► logits       same E, transposed

  saves V·d_model params (~half a billion for a big vocab) and usually improves quality
```

Tying (Press & Wolf, 2017) saves a full embedding table's worth of parameters and tends to
*help* perplexity, since the same notion of "token i" is used to read and to write. Most
models tie; some very large ones untie to give the output head independent capacity.

## Comparison

| Algorithm | Merge/selection rule | OOV handling | Reversible | Used by |
|---|---|---|---|---|
| **BPE** | most frequent adjacent pair | rare chars → `<unk>` | mostly | early NMT |
| **Byte-level BPE** | frequent pair over **bytes** | **none possible** | yes | GPT-2/3/4, LLaMA tokenizers |
| **WordPiece** | max corpus likelihood (`##` pieces) | `<unk>` | mostly | BERT, DistilBERT |
| **Unigram** | prune low-likelihood tokens; prob. segmentation | none (sampling) | yes | T5, ALBERT, XLNet |
| **SentencePiece** | *tooling* for BPE/Unigram on raw text (`▁`) | depends on model | **yes** | LLaMA, T5, multilingual |

## Best Practices

- **Always use the model's own tokenizer.** Vocab, merges, and special tokens are part of the
  checkpoint; a mismatched tokenizer silently produces garbage IDs.
- **Apply the model's chat template** (`apply_chat_template`) for instruction models — don't
  hand-concatenate role strings.
- **Match `<bos>`/`<eos>` handling at inference to training.** Many tokenizers add `<bos>`
  automatically; double-adding it shifts the distribution.
- **Pick vocab size deliberately.** Bigger vocab → shorter sequences (cheaper attention) but a
  fatter embedding/softmax; ~32k is common for English, 128k–256k for multilingual/code.
- **Tie embeddings** unless you have a specific reason not to — free parameter savings and
  usually better perplexity.
- **Budget tokens, not characters/words.** Cost, context limits, and rate limits are all
  per-token; measure with the actual tokenizer.

## Common Pitfalls

- **Wrong tokenizer for the model** — the single most common cause of "the model got dumber":
  IDs don't line up with what the embedding table learned.
- **Forgetting the chat template** — feeding raw user text to an instruction model skips the
  role markers it was trained on, degrading quality.
- **Double `<bos>`** — manually prepending `<bos>` when the tokenizer already adds it, or
  vice-versa; check `add_special_tokens`.
- **Counting words instead of tokens** — non-Latin scripts, code, and rare words explode into
  many tokens; a "500-word" prompt can be 1500+ tokens.
- **Resizing the vocab without `resize_token_embeddings`** — adding special tokens leaves the
  embedding table the wrong size and indexes out of bounds.
- **Assuming token boundaries equal word boundaries** — subwords split mid-word; logit/probability
  analysis ("which *word* did the model pick?") must reassemble pieces.
- **Untied output head left randomly initialized** when fine-tuning from a tied checkpoint —
  produces noise until trained.

## Where this connects

- [Transformers](transformers.md) — the embedding lookup is the input layer feeding attention
- [Positional encodings](positional_encoding.md) — added to token embeddings before the stack
- [Attention](attention.md) — sequence length (in tokens) drives the O(n²) cost
- [LLM decoding & sampling](decoding_sampling.md) — the output softmax over the vocabulary is
  where decoding picks the next token ID
- [Hugging Face](hugging_face.md) — `AutoTokenizer`, fast tokenizers, and chat templates
- [Neural networks](neural_networks.md) — `nn.Embedding` as a learned lookup layer
- [RAG](../ai/rag.md) — contrast with sentence/document embedding models for retrieval
- [PyTorch](pytorch.md) — `nn.Embedding` and weight-tying implementation
