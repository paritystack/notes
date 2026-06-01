# Local LLM Inference

## Overview

Running open-weight models on your own hardware — laptop, workstation, or a single GPU box —
instead of calling a hosted API. This is the practical counterpart to [vLLM](vllm.md) (which
targets servers and throughput): the tools here optimize for *getting a model running locally*
with limited VRAM. The ecosystem centers on **llama.cpp** and the **GGUF** file format, with
**Ollama** and **LM Studio** as friendly front-ends. It leans heavily on
[quantization](../machine_learning/quantization.md) to shrink models like
[Llama](llama.md), [Phi](phi.md), Mistral, Qwen, and [Gemma](model_families.md) down to sizes
that fit consumer hardware. For *why* a model is slow or memory-hungry, see
[inference optimization](inference_optimization.md).

```
  Hosted API (OpenAI/Anthropic)   vs.   Local inference (this page)
  ───────────────────────────           ───────────────────────────
  frontier models, no hardware          open weights, your hardware
  per-token cost, data leaves           free after download, private
  vLLM/TGI behind the scenes            llama.cpp / Ollama / MLX
```

## The stack: llama.cpp and GGUF

**llama.cpp** is a C/C++ inference engine that runs transformer models on CPU and GPU with
aggressive quantization. It defines **GGUF** (GGML Universal Format), the de-facto file format
for local models: a single file bundling weights, tokenizer, and metadata. Almost every local
tool is a wrapper around llama.cpp or reads GGUF.

```
  HF model (safetensors, fp16)
        │  convert + quantize
        ▼
   model.gguf  ──►  llama.cpp  ──►  Ollama / LM Studio / text-generation-webui
   (one file)       (engine)        (front-ends)
```

GGUF files are named by quantization level. The common ones:

| Quant | Bits/weight | Quality | Use when |
|-------|-------------|---------|----------|
| `Q8_0` | 8 | ~lossless | you have the VRAM |
| `Q6_K` | 6.6 | excellent | sweet spot for quality |
| `Q4_K_M` | ~4.8 | very good | **default** — best size/quality trade-off |
| `Q3_K_M` | ~3.9 | usable | tight on memory |
| `Q2_K` | ~2.6 | degraded | last resort |

`K` quants (k-means) are smarter than the old linear quants; `_M`/`_S`/`_L` denote medium/
small/large variants. See [quantization](../machine_learning/quantization.md) for the math.

## Sizing: will it fit?

The dominant cost is **weights** plus the **KV cache** (which grows with context length —
see [inference optimization](inference_optimization.md)). A rough rule:

```
  VRAM needed ≈ (params × bits_per_weight / 8)  +  KV cache  +  overhead
  e.g. 7B at Q4_K_M ≈ 7e9 × 4.8/8 ≈ 4.2 GB weights + ~1 GB KV ≈ ~5–6 GB
```

| Model size | Q4_K_M weights | Fits on |
|------------|----------------|---------|
| 7–8B | ~4–5 GB | 8 GB GPU / 16 GB Mac |
| 13–14B | ~8 GB | 12 GB GPU |
| 32–34B | ~20 GB | 24 GB GPU (3090/4090) |
| 70B | ~40 GB | 2× 24 GB or 48 GB |

If a model doesn't fit, llama.cpp **offloads** some layers to GPU and runs the rest on CPU
(`-ngl N` = number of GPU layers). Partial offload is much slower because layers stream across
the PCIe bus. On Apple Silicon, **unified memory** means the GPU can use most of system RAM —
a 64 GB Mac runs 70B models that need a multi-GPU rig on PC.

## Ollama

The most popular local runner — a daemon + CLI that manages model downloads and serves an
OpenAI-compatible API. Models are pulled from its registry by name and tag.

```bash
ollama pull llama3.1:8b          # downloads a GGUF
ollama run llama3.1:8b "explain quantization"

# OpenAI-compatible endpoint on localhost:11434
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"hi"}]}'
```

A **Modelfile** customizes a model (system prompt, parameters, adapters) like a Dockerfile:

```
FROM llama3.1:8b
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
SYSTEM "You are a terse senior engineer."
```

```bash
ollama create my-assistant -f Modelfile
```

## Other runners

- **LM Studio** — a desktop GUI for discovering, downloading, and chatting with GGUF models;
  also exposes an OpenAI-compatible local server. Best for non-CLI users.
- **llama.cpp server** — `llama-server` gives you the raw engine with full control over
  sampling, slots, and continuous batching; what Ollama wraps.
- **MLX / mlx-lm** — Apple's framework, fastest path on Apple Silicon (uses the GGUF-adjacent
  MLX format rather than GGUF).
- **text-generation-webui ("oobabooga")** — a feature-rich web UI supporting GGUF, GPTQ, EXL2.
- **vLLM / TGI** — server-grade, GPU-only, for throughput; covered in [vLLM](vllm.md).

## Where this connects

- [vLLM](vllm.md) — the high-throughput server counterpart; local runners trade throughput
  for fit and simplicity.
- [Quantization](../machine_learning/quantization.md) — the mechanism that makes local
  inference possible; GGUF quant levels are its applied form.
- [Inference optimization](inference_optimization.md) — KV cache, batching, and speculative
  decoding explain local latency/memory.
- [Model families](model_families.md) — which open-weight models to run locally.
- [Fine-tuning](fine_tuning.md) — [LoRA](../machine_learning/lora.md) adapters can be merged
  into a GGUF or loaded at runtime.

## Pitfalls

- **Quant too aggressive.** `Q2_K`/`Q3` noticeably degrade reasoning and code. Prefer `Q4_K_M`
  or higher unless memory forces otherwise.
- **Forgetting the KV cache.** A long context window can need more memory than the weights;
  budget for it or cap `num_ctx`.
- **Partial GPU offload feels broken.** It isn't — CPU layers are just slow. Either fit fully
  in VRAM or accept the slowdown.
- **Context window ≠ training context.** Setting `num_ctx` beyond what the model was trained
  for produces garbage; check the model card.
- **Assuming local = private but logging to a cloud UI.** Some front-ends phone home or sync;
  verify if privacy is the reason you went local.
