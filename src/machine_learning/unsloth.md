# Unsloth

## Overview

Unsloth is an open-source library that makes fine-tuning and inference of large
language models dramatically faster and lighter — roughly **2× faster** training
with **up to ~70–80% less VRAM** and no loss in accuracy. It is a drop-in layer
over the usual stack: it wraps [Hugging Face](hugging_face.md) Transformers,
PEFT, and TRL behind a `FastLanguageModel` API, swaps the hot paths for
hand-written Triton kernels with manual autograd, and loads weights in 4-bit so
that [LoRA](lora.md) / QLoRA fine-tuning (see [Quantization](quantization.md))
fits on a single consumer GPU. The models are still ordinary
[Transformers](transformers.md) modules running on [PyTorch](pytorch.md), so
everything you know about [fine-tuning](../ai/fine_tuning.md) carries over —
Unsloth just removes the speed and memory bottlenecks. It supports Llama,
Mistral, Qwen, Gemma, Phi and similar architectures.

## How it speeds things up

Unsloth patches the model after loading: the standard Transformers attention,
RoPE, RMSNorm and cross-entropy paths are replaced with fused Triton kernels,
and the base weights are held in 4-bit (bitsandbytes NF4) while only the LoRA
adapters train.

```
            ┌──────────────────────────────────────────┐
            │            FastLanguageModel               │
            │  (drop-in over HF Transformers + PEFT/TRL) │
            ├──────────────────────────────────────────┤
            │  Triton kernels + manual autograd          │
            │   • fused attention / flash-style          │
            │   • RoPE, RMSNorm, SwiGLU                   │
            │   • fused cross-entropy (no big logits)     │
            ├──────────────────────────────────────────┤
            │  Base weights: 4-bit NF4 (bitsandbytes)     │
            │  Trainable:    LoRA adapters (fp16/bf16)    │
            ├──────────────────────────────────────────┤
            │            PyTorch + CUDA / Triton          │
            └──────────────────────────────────────────┘
```

Key tricks:

- **Hand-written Triton kernels** for the attention, normalization and MLP
  blocks, avoiding generic-autograd overhead.
- **Manual backward passes** that recompute cheaply instead of caching large
  activations — this is where most of the VRAM savings come from.
- **Fused cross-entropy** that never materializes the full `(seq × vocab)`
  logits tensor, the usual memory spike at the LM head.
- **4-bit loading** of the frozen base model, same idea as QLoRA's NF4.
- Together these allow **longer context lengths** on the same card.

## Quickstart

The flow mirrors the PEFT + TRL pattern in [LoRA](lora.md), with
`FastLanguageModel` replacing the raw `from_pretrained` / `get_peft_model` calls.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load base model in 4-bit (Unsloth ships pre-quantized checkpoints)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,            # auto: bf16 on Ampere+, else fp16
    load_in_4bit=True,
)

# Attach LoRA adapters via Unsloth's patched PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,        # 0 is optimized (fast path)
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # extra-efficient variant
)

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="./unsloth-llama3",
    ),
)

trainer.train()

# Fast inference on the same object
FastLanguageModel.for_inference(model)
inputs = tokenizer("### Instruction:\nExplain LoRA in one line.\n\n### Response:\n",
                   return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=64)[0]))
```

## Memory & speed

Approximate, hardware-dependent figures comparing standard 4-bit QLoRA against
Unsloth for the same LoRA config on one GPU:

| Aspect            | Standard QLoRA      | Unsloth                  |
|-------------------|---------------------|--------------------------|
| Training speed    | 1×                  | ~2× faster               |
| VRAM use          | baseline            | ~40–80% less             |
| Max context       | baseline            | noticeably longer        |
| Accuracy          | reference           | matches (no degradation) |

Treat the numbers as directional — exact gains depend on model, sequence
length, batch size and GPU. The free open-source tier targets **single-GPU**;
multi-GPU/distributed support is more limited.

## Exporting models

After training, the adapters can be merged and exported for downstream serving:

```python
# Merge LoRA into 16-bit weights and push to the Hub
model.save_pretrained_merged("merged_16bit", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("user/llama3-ft", tokenizer, save_method="merged_16bit")

# Export to GGUF for llama.cpp / Ollama (optionally quantized)
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")

# merged_16bit weights also load directly into vLLM for fast serving
```

GGUF output plugs straight into Ollama/llama.cpp for
[local inference](../ai/local_inference.md); merged 16-bit weights are ready for
[vLLM](../ai/vllm.md) or a normal Transformers load.

## Pitfalls

- **Single-GPU focus.** The free library is tuned for one GPU; multi-GPU and
  fully distributed training are limited or gated.
- **Version coupling.** Unsloth monkey-patches Transformers/PEFT internals, so a
  Transformers/PEFT upgrade can break it — pin compatible versions.
- **Architecture support is finite.** Only families with hand-written kernels
  (Llama, Mistral, Qwen, Gemma, Phi, …) get the full speedup; others fall back
  or are unsupported.
- **4-bit quality caveats.** As with QLoRA, NF4 base weights can slightly affect
  quality on sensitive tasks; evaluate before merging — see
  [Quantization](quantization.md).
- **Benchmarks vary.** Headline "2× / 80%" numbers are best-case; measure on
  your own model and hardware.

## Where this connects

- [LoRA](lora.md) — Unsloth accelerates LoRA/QLoRA training; the config and
  adapter workflow are the same.
- [Quantization](quantization.md) — the 4-bit NF4 base-weight loading is the
  QLoRA quantization scheme.
- [Hugging Face](hugging_face.md) — Unsloth wraps Transformers/PEFT/TRL and
  pushes results to the Hub.
- [Transformers](transformers.md) — the patched modules are standard
  transformer blocks with fused kernels.
- [PyTorch](pytorch.md) — the custom kernels run as Triton/PyTorch autograd ops.
- [../ai/fine_tuning](../ai/fine_tuning.md) — Unsloth is a fast backend for the
  general LLM fine-tuning workflow.
