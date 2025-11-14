# DeepSeek R1 - Open Source Reasoning Model

Complete guide to DeepSeek R1, the open-source reasoning model that rivals OpenAI's o1, from setup to fine-tuning and deployment.

## Table of Contents
- [Introduction](#introduction)
- [Model Versions](#model-versions)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Prompt Engineering](#prompt-engineering)
- [Fine-tuning](#fine-tuning)
- [Deployment](#deployment)
- [Common Patterns & Operations](#common-patterns--operations)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Introduction

DeepSeek R1, released on January 20, 2025, represents a breakthrough in open-source reasoning models. Built on reinforcement learning (RL), it achieves performance comparable to OpenAI's o1 on complex reasoning tasks while being fully open-source under an MIT license.

### Key Features

- **Advanced Reasoning**: Native chain-of-thought reasoning capabilities
- **Open Source**: MIT licensed for commercial and academic use
- **Competitive Performance**: Matches or exceeds o1 on mathematical and coding benchmarks
- **Multiple Sizes**: From 1.5B to 671B parameters (distilled and full models)
- **Long Context**: 128K token context window
- **Self-Verification**: Built-in reasoning verification and error correction
- **Efficient Architecture**: Mixture of Experts (MoE) design

### Benchmark Performance

| Task | DeepSeek R1 | OpenAI o1 |
|------|-------------|-----------|
| AIME 2024 (Math) | 79.8% | 79.2% |
| MATH-500 | 97.3% | 97.3% |
| Codeforces | 96.3 (2,029 Elo) | Similar |
| GPQA Diamond | 71.5% | Comparable |
| MMLU | Superior to V3 | - |

### Architecture Highlights

- **671B Total Parameters** (37B activated per forward pass)
- **Mixture of Experts (MoE)**: Efficient routing to specialized expert networks
- **Multi-head Latent Attention (MLA)**: Reduces KV-cache to 5-13% of traditional methods
- **Rotary Position Embeddings (RoPE)**: Enhanced position encoding
- **61 Hidden Layers**: Deep architecture for complex reasoning

## Model Versions

### DeepSeek R1 (Full Model)

**Released**: January 2025

```python
# Note: Direct Transformers support not yet available
# Use vLLM or refer to DeepSeek-V3 repo
```

**Specifications:**
- Total Parameters: 671B
- Activated Parameters: 37B per forward pass
- Context Length: 128K tokens
- Architecture: MoE
- License: MIT

**Use Cases:**
- Complex mathematical reasoning
- Advanced coding challenges
- Multi-step logical problems
- Research applications

### DeepSeek R1 Zero

**Training Approach**: Pure RL without supervised fine-tuning

```python
# Same infrastructure as DeepSeek R1
# Demonstrates RL-only training effectiveness
```

**Key Differences:**
- No SFT phase (pure RL)
- Emerged reasoning behaviors autonomously
- Research-focused variant

### Distilled Models (Qwen-based)

#### DeepSeek-R1-Distill-Qwen-1.5B

```bash
# Ollama
ollama pull deepseek-r1:1.5b
ollama run deepseek-r1:1.5b

# vLLM
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Features:**
- Smallest variant for edge deployment
- Fast inference on consumer hardware
- Suitable for resource-constrained environments

#### DeepSeek-R1-Distill-Qwen-7B

```bash
ollama pull deepseek-r1:7b
ollama run deepseek-r1:7b
```

**Performance:**
- AIME 2024: 55.5%
- Good balance of size and capability
- Runs on 6GB+ VRAM GPUs

#### DeepSeek-R1-Distill-Qwen-14B

```bash
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
```

**Features:**
- Enhanced reasoning over 7B
- Mid-range deployment option
- Excellent for local development

#### DeepSeek-R1-Distill-Qwen-32B

```bash
# vLLM with tensor parallelism
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager

# SGLang
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --trust-remote-code \
    --tp 2
```

**Performance:**
- AIME 2024: 72.6%
- MATH-500: 94.3%
- LiveCodeBench: 57.2%
- Outperforms OpenAI o1-mini on multiple benchmarks

### Distilled Models (Llama-based)

#### DeepSeek-R1-Distill-Llama-8B

```bash
ollama pull deepseek-r1:8b
ollama run deepseek-r1:8b

# Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
```

**Features:**
- Based on Llama architecture
- Compatible with Llama ecosystem
- Good for fine-tuning on custom tasks

#### DeepSeek-R1-Distill-Llama-70B

```bash
ollama pull deepseek-r1:70b
ollama run deepseek-r1:70b
```

**Features:**
- Highest-capacity distilled model
- Excellent reasoning capabilities
- Production-ready performance

### Model Comparison

| Model | Parameters | AIME 2024 | MATH-500 | VRAM (FP16) | Use Case |
|-------|------------|-----------|----------|-------------|----------|
| R1-Distill-Qwen-1.5B | 1.5B | ~20% | ~60% | 3GB | Edge/Mobile |
| R1-Distill-Qwen-7B | 7B | 55.5% | ~85% | 14GB | Desktop |
| R1-Distill-Llama-8B | 8B | ~57% | ~86% | 16GB | Standard |
| R1-Distill-Qwen-14B | 14B | ~65% | ~90% | 28GB | Mid-range |
| R1-Distill-Qwen-32B | 32B | 72.6% | 94.3% | 64GB | High-end |
| R1-Distill-Llama-70B | 70B | ~76% | ~96% | 140GB | Production |
| DeepSeek R1 | 671B (37B active) | 79.8% | 97.3% | 140GB+ (MoE) | Research/Max quality |

## Architecture

### Mixture of Experts (MoE)

```
Total Parameters: 671B
Activated per Forward Pass: 37B (~5.5%)

┌─────────────────────────────────────┐
│         Input Embedding             │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │   Router    │
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│Expert1│  │Expert2│  │Expert3│ ... (multiple experts)
└───┬───┘  └───┬───┘  └───┬───┘
    └──────────┼──────────┘
               ▼
        ┌─────────────┐
        │   Output    │
        └─────────────┘
```

**Benefits:**
- Efficient compute: Only 37B params active per token
- Specialized expertise: Routing to relevant expert clusters
- Scalability: Add experts without linear compute increase

### Multi-head Latent Attention (MLA)

```python
# Traditional Attention KV-cache
traditional_kv_cache = num_heads * head_dim * sequence_length * 2 * bytes_per_param
# Example: 32 * 128 * 4096 * 2 * 2 = 64 MB per layer

# MLA Latent KV-cache (5-13% reduction)
mla_latent_cache = latent_dim * sequence_length * 2 * bytes_per_param
# Example: 512 * 4096 * 2 * 2 = 8 MB per layer (~87% reduction)
```

**Key Innovation:**
1. Compress K and V into low-dimensional latent vectors during training
2. Store only latent representations in KV-cache
3. Decompress on-the-fly during inference
4. Dramatically reduces memory overhead

### Layer Structure (61 Hidden Layers)

```
Layer Pattern:
┌─────────────────────────┐
│  Input from prev layer  │
└────────────┬────────────┘
             │
        ┌────▼────┐
        │ RoPE    │  (Rotary Position Embeddings)
        └────┬────┘
             │
        ┌────▼────┐
        │   MLA   │  (Multi-head Latent Attention)
        └────┬────┘
             │
        ┌────▼────┐
        │ RMSNorm │
        └────┬────┘
             │
        ┌────▼────┐
        │ MoE FFN │  (Mixture of Experts Feed Forward)
        └────┬────┘
             │
        ┌────▼────┐
        │ RMSNorm │
        └────┬────┘
             │
    ┌────────▼────────┐
    │ Output to next  │
    │     layer       │
    └─────────────────┘
```

### Training Methodology

```
Phase 1: Supervised Fine-Tuning (SFT)
├── Curated long chain-of-thought examples
├── 800K high-quality reasoning samples
└── Initial reasoning pattern formation

Phase 2: Reinforcement Learning (RL)
├── Policy gradient optimization
├── Self-verification rewards
├── Error correction incentivization
└── Emergent behaviors:
    ├── Chain-of-thought reasoning
    ├── Self-reflection
    ├── Verification steps
    └── Logical decomposition
```

**R1-Zero Variant**: Skipped Phase 1 entirely, demonstrating RL can develop reasoning from scratch.

## Installation & Setup

### Via Ollama (Easiest for Local Use)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run distilled models
ollama pull deepseek-r1:1.5b   # Smallest
ollama pull deepseek-r1:7b     # Balanced
ollama pull deepseek-r1:8b     # Llama-based
ollama pull deepseek-r1:14b    # Mid-range
ollama pull deepseek-r1:32b    # High-quality
ollama pull deepseek-r1:70b    # Best distilled

# Interactive chat
ollama run deepseek-r1:7b
```

**Python usage:**

```python
import ollama

# Simple generation
response = ollama.generate(
    model='deepseek-r1:7b',
    prompt='Solve: If x^2 + 5x + 6 = 0, find x.',
)
print(response['response'])

# Chat interface
messages = [
    {
        'role': 'user',
        'content': 'Explain the time complexity of quicksort'
    }
]

response = ollama.chat(
    model='deepseek-r1:7b',
    messages=messages
)
print(response['message']['content'])
```

### Via vLLM (Production Inference)

```bash
# Install vLLM
pip install vllm

# Serve distilled models
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# For larger models with tensor parallelism
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager
```

**Python client:**

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    tensor_parallel_size=1
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.6,  # Recommended: 0.5-0.7
    top_p=0.95,
    max_tokens=2048
)

# Generate
prompts = [
    "Write a Python function to find prime numbers up to n",
    "Explain the concept of gradient descent"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}\n")
```

### Via SGLang (Fast Inference Engine)

```bash
# Install
pip install "sglang[all]"

# Serve model
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --trust-remote-code \
    --tp 1
```

**Python usage:**

```python
import sglang as sgl

@sgl.function
def reasoning_task(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=1024, temperature=0.6))

# Run
state = reasoning_task.run(
    question="What is the derivative of x^3 + 2x^2 + 5?",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

print(state["answer"])
```

### Via Hugging Face Transformers

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load distilled model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Generate
prompt = "Calculate the factorial of 10 step by step"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Via API Providers

#### Together.ai

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-together-api-key",
    base_url="https://api.together.xyz/v1"
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "Solve: 2x + 5 = 15"}
    ],
    temperature=0.6,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

#### Fireworks.ai

```python
import fireworks.client

fireworks.client.api_key = "your-fireworks-api-key"

response = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/deepseek-r1",
    messages=[{
        "role": "user",
        "content": "Explain binary search algorithm"
    }],
    temperature=0.6
)

print(response.choices[0].message.content)
```

## Basic Usage

### Simple Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Direct generation
prompt = "What is the square root of 144?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,  # CRITICAL: 0.5-0.7 range
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Chat Format

**IMPORTANT**: DeepSeek R1 works best WITHOUT system prompts. Put all instructions in user messages.

```python
# ❌ AVOID: System prompts reduce effectiveness
messages_bad = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Solve this problem..."}
]

# ✅ RECOMMENDED: All instructions in user message
messages_good = [
    {"role": "user", "content": "Solve this problem step by step: ..."}
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages_good,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate
outputs = model.generate(
    input_ids,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][input_ids.shape[-1]:],
    skip_special_tokens=True
)
print(response)
```

### Enforcing Reasoning with `<think>` Tags

```python
# Force model to show reasoning
prompt = """Solve the following problem. Begin your response with <think> to show your reasoning process.

Problem: A train travels 120 km in 2 hours. If it maintains the same speed, how far will it travel in 5 hours?"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# Output will include:
# <think>
# The train travels 120 km in 2 hours
# Speed = distance / time = 120 / 2 = 60 km/h
# For 5 hours: distance = speed × time = 60 × 5 = 300 km
# </think>
# The train will travel 300 km in 5 hours.
```

### Multi-turn Conversation

```python
conversation = []

def chat(user_message):
    # Add user message
    conversation.append({"role": "user", "content": user_message})

    # Generate response
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    # Add to conversation
    conversation.append({"role": "assistant", "content": response})

    return response

# Use
print(chat("What is 15 factorial?"))
print(chat("Now divide that by 120"))
print(chat("Express the result in scientific notation"))
```

### Batch Processing

```python
prompts = [
    "What is the time complexity of merge sort?",
    "Explain the difference between TCP and UDP",
    "Calculate: (3x + 5)(2x - 1)"
]

# Tokenize with padding
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)

# Decode
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

## Prompt Engineering

DeepSeek R1 requires a **fundamentally different** prompting approach than traditional LLMs.

### Critical Guidelines

#### ❌ DON'T:

1. **Don't use few-shot examples** - They degrade performance
```python
# ❌ AVOID
prompt = """
Q: What is 2+2?
A: 4

Q: What is 3+3?
A: 6

Q: What is 5+5?
A: """
```

2. **Don't add explicit chain-of-thought instructions** - R1 does this natively
```python
# ❌ AVOID
prompt = "Let's think step by step. First, ... Second, ... Third, ..."
```

3. **Don't use system prompts** - Put everything in user message
```python
# ❌ AVOID
messages = [
    {"role": "system", "content": "You are an expert mathematician..."},
    {"role": "user", "content": "Solve..."}
]
```

4. **Don't overload with context** - Be concise and clear
```python
# ❌ AVOID
prompt = "Given the following extensive background information... [5 paragraphs]... now solve..."
```

#### ✅ DO:

1. **Use minimal, clear prompts**
```python
# ✅ GOOD
prompt = "Solve: If f(x) = 3x^2 + 2x - 5, find f(4)"
```

2. **State the problem directly**
```python
# ✅ GOOD
prompt = "Compare the advantages and disadvantages of SQL vs NoSQL databases"
```

3. **Use structured input when needed**
```python
# ✅ GOOD
prompt = """Analyze these three options:
A. Cloud deployment
B. On-premise servers
C. Hybrid approach

Evaluate cost, scalability, and security for each."""
```

4. **Request specific output formats**
```python
# ✅ GOOD
prompt = "List the prime numbers between 1 and 50. Format as a Python list."
```

### Optimal Parameters

```python
# Recommended configuration
generation_config = {
    "temperature": 0.6,      # Range: 0.5-0.7 (prevents loops)
    "top_p": 0.95,          # Recommended value
    "max_new_tokens": 2048,  # Adjust based on task
    "do_sample": True,
    "repetition_penalty": 1.0  # Usually not needed
}

outputs = model.generate(**inputs, **generation_config)
```

### Chain-of-Draft (CoD) Technique

Reduce token usage by 80% while maintaining quality:

```python
# Standard reasoning (verbose)
prompt = "Solve this complex calculus problem: ..."
# Output: 2000+ tokens with full reasoning

# Chain-of-Draft (efficient)
prompt = """Solve this complex calculus problem: ...

Think step by step, but only keep a minimum draft for each thinking step."""

# Output: ~400 tokens with condensed reasoning, same accuracy
```

### Template Patterns

#### Mathematical Problems

```python
template = """Solve the following problem:

Problem: {problem}

Show your work and provide the final answer."""

prompt = template.format(
    problem="Find the derivative of f(x) = x^3 * sin(x)"
)
```

#### Code Generation

```python
template = """Write a {language} function that {description}.

Requirements:
- {requirement1}
- {requirement2}
- Include error handling"""

prompt = template.format(
    language="Python",
    description="implements a binary search tree",
    requirement1="Support insert, search, and delete operations",
    requirement2="Maintain BST properties"
)
```

#### Analysis Tasks

```python
template = """Analyze the following scenario:

{scenario}

Provide:
1. Key insights
2. Potential risks
3. Recommended actions"""

prompt = template.format(
    scenario="A startup wants to migrate from monolith to microservices"
)
```

#### Comparison Tasks

```python
template = """Compare {option_a} vs {option_b}:

Evaluate:
- Performance
- Scalability
- Cost
- Ease of use

Provide a recommendation."""

prompt = template.format(
    option_a="PostgreSQL",
    option_b="MongoDB"
)
```

### Advanced Prompting Techniques

#### Self-Verification Prompting

```python
prompt = """Solve: x^2 - 7x + 12 = 0

After solving, verify your answer by substituting back into the original equation."""
```

#### Multi-Part Problems

```python
prompt = """Problem: A rectangle has a perimeter of 30 cm and an area of 50 cm².

Find:
1. The length
2. The width
3. The diagonal length"""
```

#### Constraint-Based Prompting

```python
prompt = """Generate a regex pattern that matches:
- Valid email addresses
- Must include @ symbol
- Domain must end in .com, .org, or .net
- No special characters except . and _

Provide the pattern and explain each component."""
```

## Fine-tuning

### LoRA Fine-tuning (Recommended)

```bash
pip install transformers peft accelerate datasets torch
```

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Load base model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                    # Rank (8, 16, 32)
    lora_alpha=32,          # Scaling factor (2*r typical)
    target_modules=[        # Target attention layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~8M / 7B (~0.1%)

# Prepare dataset
dataset = load_dataset("your-dataset")

def format_prompt(example):
    # Format for reasoning tasks
    return {
        "text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}"
    }

dataset = dataset.map(format_prompt)

# Tokenize
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./deepseek-r1-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=100,
    optim="adamw_torch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"] if "test" in tokenized_dataset else None
)

# Train
trainer.train()

# Save
model.save_pretrained("./deepseek-r1-lora-final")
tokenizer.save_pretrained("./deepseek-r1-lora-final")
```

### QLoRA Fine-tuning (4-bit Quantization)

```bash
pip install bitsandbytes
```

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config (same as above)
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)

# Training args with 8-bit optimizer
training_args = TrainingArguments(
    output_dir="./deepseek-r1-qlora",
    optim="paged_adamw_8bit",  # 8-bit optimizer
    fp16=True,  # or bf16
    # ... rest of args
)

# Train
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### Using Fine-tuned Model

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "./deepseek-r1-lora-final"
)

# Merge for faster inference (optional)
model = model.merge_and_unload()

# Use normally
tokenizer = AutoTokenizer.from_pretrained("./deepseek-r1-lora-final")
inputs = tokenizer("Your prompt", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Dataset Preparation for Reasoning

```python
# Format data for reasoning tasks
dataset_dict = {
    "train": [
        {
            "problem": "Find the area of a circle with radius 5",
            "solution": "Area = πr² = π(5)² = 25π ≈ 78.54 square units"
        },
        {
            "problem": "What is 15! / 13!?",
            "solution": "15! / 13! = 15 × 14 × 13! / 13! = 15 × 14 = 210"
        },
        # ... more examples
    ]
}

from datasets import Dataset
dataset = Dataset.from_dict(dataset_dict)

# Or load from files
# JSON Lines format:
# {"problem": "...", "solution": "..."}
# {"problem": "...", "solution": "..."}

dataset = load_dataset("json", data_files="train.jsonl")
```

### Hyperparameter Recommendations

```python
# Small models (1.5B-7B)
small_model_config = {
    "lora_r": 8,
    "lora_alpha": 16,
    "learning_rate": 3e-4,
    "batch_size": 8,
    "gradient_accumulation": 2
}

# Medium models (8B-14B)
medium_model_config = {
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation": 4
}

# Large models (32B-70B)
large_model_config = {
    "lora_r": 32,
    "lora_alpha": 64,
    "learning_rate": 1e-4,
    "batch_size": 2,
    "gradient_accumulation": 8
}
```

### Using Axolotl for Simplified Training

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

Create `deepseek_r1_config.yml`:

```yaml
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

datasets:
  - path: your-dataset.jsonl
    type: alpaca

num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0002

warmup_steps: 100
optimizer: paged_adamw_8bit
lr_scheduler: cosine

output_dir: ./deepseek-r1-tuned

bf16: true
tf32: true
gradient_checkpointing: true
```

Train:

```bash
accelerate launch -m axolotl.cli.train deepseek_r1_config.yml
```

## Deployment

### FastAPI Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()

# Load model at startup
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    response: str
    tokens_used: int

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_used = len(outputs[0])

        return GenerateResponse(
            response=result,
            tokens_used=tokens_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Usage:

```bash
# Run server
python server.py

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is recursion?",
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

### vLLM Production Server

```bash
# Start server
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# With GPU specification
CUDA_VISIBLE_DEVICES=0,1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2
```

**Client usage:**

```python
from openai import OpenAI

# vLLM provides OpenAI-compatible API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth by default
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    messages=[
        {"role": "user", "content": "Explain binary trees"}
    ],
    temperature=0.6,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
RUN pip3 install vllm transformers torch

# Download model (or mount as volume)
RUN python3 -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')"

# Expose port
EXPOSE 8000

# Run server
CMD ["vllm", "serve", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", \
     "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t deepseek-r1-server .
docker run --gpus all -p 8000:8000 deepseek-r1-server
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseek-r1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepseek-r1
  template:
    metadata:
      labels:
        app: deepseek-r1
    spec:
      containers:
      - name: deepseek-r1
        image: deepseek-r1-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: deepseek-r1-service
spec:
  selector:
    app: deepseek-r1
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### AWS SageMaker Deployment

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# HuggingFace model configuration
huggingface_model = HuggingFaceModel(
    model_data="s3://your-bucket/model.tar.gz",  # Or use hub
    transformers_version='4.37',
    pytorch_version='2.1',
    py_version='py310',
    role=sagemaker.get_execution_role(),
    env={
        'HF_MODEL_ID': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'HF_TASK': 'text-generation'
    }
)

# Deploy
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge'
)

# Use
response = predictor.predict({
    'inputs': 'What is machine learning?',
    'parameters': {
        'max_new_tokens': 512,
        'temperature': 0.6
    }
})

print(response[0]['generated_text'])
```

### LangChain Integration

```bash
pip install langchain langchain-community
```

```python
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95
)

# LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create chain
template = """Problem: {problem}

Solve this step by step."""

prompt = PromptTemplate(template=template, input_variables=["problem"])
chain = LLMChain(llm=llm, prompt=prompt)

# Use
result = chain.run("Find the roots of x^2 - 5x + 6 = 0")
print(result)
```

## Common Patterns & Operations

### Mathematical Problem Solving

```python
def solve_math_problem(problem: str) -> str:
    """Solve mathematical problems with reasoning"""
    prompt = f"""Solve the following mathematical problem. Show your reasoning.

Problem: {problem}

Begin with <think> to show your work."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.6,
        top_p=0.95
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Examples
print(solve_math_problem("What is the derivative of x^3 + 2x^2 - 5x + 3?"))
print(solve_math_problem("Solve the system: 2x + y = 7, x - y = 2"))
print(solve_math_problem("Find the area under y=x^2 from x=0 to x=3"))
```

### Code Generation

```python
def generate_code(description: str, language: str = "Python") -> str:
    """Generate code with explanation"""
    prompt = f"""Write a {language} function that {description}.

Requirements:
- Include docstring
- Add error handling
- Use type hints (if applicable)
- Provide usage example"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
code = generate_code(
    "implements a binary search algorithm on a sorted list",
    "Python"
)
print(code)
```

### Code Review & Debugging

```python
def review_code(code: str) -> str:
    """Review code for issues and improvements"""
    prompt = f"""Review the following code. Identify:
1. Potential bugs
2. Performance issues
3. Security concerns
4. Suggested improvements

Code:
```
{code}
```

Provide detailed analysis."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.6)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
buggy_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
"""

print(review_code(buggy_code))
```

### Logical Reasoning

```python
def logical_reasoning(premise: str, question: str) -> str:
    """Perform logical reasoning on given premises"""
    prompt = f"""Given the following information:

{premise}

Question: {question}

Think through this logically and provide a reasoned answer."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1536, temperature=0.6)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
premise = """
- All programmers know at least one language
- Alice is a programmer
- Bob knows Python
- Python is a programming language
"""

question = "Does Alice necessarily know Python?"
print(logical_reasoning(premise, question))
```

### Data Analysis

```python
def analyze_data(data_description: str, question: str) -> str:
    """Analyze data and answer questions"""
    prompt = f"""Dataset: {data_description}

Question: {question}

Analyze the data and provide insights with reasoning."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1536, temperature=0.6)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
data = """
Sales data for Q1 2025:
- January: $50,000 (100 customers)
- February: $65,000 (120 customers)
- March: $72,000 (130 customers)
"""

analysis = analyze_data(
    data,
    "What is the trend in average revenue per customer?"
)
print(analysis)
```

### Comparative Analysis

```python
def compare_options(options: list, criteria: list) -> str:
    """Compare multiple options across criteria"""
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    criteria_text = "\n".join([f"- {c}" for c in criteria])

    prompt = f"""Compare the following options:

{options_text}

Evaluation criteria:
{criteria_text}

Provide a detailed comparison and recommendation."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.6)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
result = compare_options(
    options=[
        "PostgreSQL",
        "MongoDB",
        "MySQL"
    ],
    criteria=[
        "Performance",
        "Scalability",
        "Ease of use",
        "ACID compliance"
    ]
)
print(result)
```

### Question Answering with Context

```python
def qa_with_context(context: str, question: str) -> str:
    """Answer questions based on provided context"""
    prompt = f"""Context:
{context}

Question: {question}

Answer based on the context provided."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.6)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
context = """
The Python programming language was created by Guido van Rossum and first
released in 1991. Python emphasizes code readability with significant whitespace.
It supports multiple programming paradigms including procedural, object-oriented,
and functional programming.
"""

answer = qa_with_context(context, "When was Python first released?")
print(answer)
```

## Advanced Techniques

### Retrieval-Augmented Generation (RAG)

```bash
pip install langchain chromadb sentence-transformers
```

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

# Load documents
documents = [
    "DeepSeek R1 is an open-source reasoning model released in January 2025.",
    "It uses a Mixture of Experts architecture with 671B parameters.",
    "The model achieves 79.8% on AIME 2024 mathematics benchmark.",
    # ... more documents
]

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.create_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # HuggingFacePipeline from earlier
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query
question = "What is DeepSeek R1's performance on mathematics?"
result = qa_chain.run(question)
print(result)
```

### Function Calling / Tool Use

```python
import json
import re

def execute_function_call(response: str, available_functions: dict):
    """Execute function calls from model responses"""
    # Extract function call from response
    pattern = r'\{"function":\s*"(\w+)",\s*"parameters":\s*(\{[^}]+\})\}'
    match = re.search(pattern, response)

    if match:
        func_name = match.group(1)
        params = json.loads(match.group(2))

        if func_name in available_functions:
            return available_functions[func_name](**params)

    return None

# Define tools
def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions"""
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except:
        return "Error in calculation"

def get_weather(location: str) -> dict:
    """Get weather for location (simulated)"""
    return {
        "location": location,
        "temperature": 22,
        "condition": "sunny"
    }

available_functions = {
    "calculate": calculate,
    "get_weather": get_weather
}

# System prompt with tools
tools_description = """
Available functions:
1. calculate(expression) - Evaluate math expressions
2. get_weather(location) - Get weather for a location

To use a function, respond with:
{"function": "function_name", "parameters": {"param": "value"}}
"""

prompt = f"""{tools_description}

User: What's 15 * 23 + 45?

Respond with a function call."""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.6)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Execute function
result = execute_function_call(response, available_functions)
print(f"Result: {result}")
```

### Constrained Generation

```bash
pip install outlines
```

```python
import outlines

# Load model for outlines
model = outlines.models.transformers(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# JSON schema constraint
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "experience_years": {"type": "integer"}
    },
    "required": ["name", "age", "skills"]
}

generator = outlines.generate.json(model, schema)
result = generator("Generate a software engineer profile:")
print(json.dumps(result, indent=2))

# Regex constraint
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
generator = outlines.generate.regex(model, email_pattern)
email = generator("Generate a professional email address:")
print(email)

# Multiple choice
choices = ["Python", "JavaScript", "Java", "C++", "Go"]
generator = outlines.generate.choice(model, choices)
language = generator("What is the best language for web backends?")
print(language)
```

### Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_response(prompt: str):
    """Generate response with streaming"""
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
        "streamer": streamer
    }

    # Generate in thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream output
    print("Response: ", end="", flush=True)
    for text in streamer:
        print(text, end="", flush=True)
    print()

    thread.join()

# Use
stream_response("Explain how recursion works in programming")
```

### Multi-Step Reasoning

```python
def multi_step_solver(problem: str, max_steps: int = 5) -> str:
    """Solve problems through iterative reasoning"""
    conversation = []

    # Initial problem
    conversation.append({
        "role": "user",
        "content": f"""Solve this problem step by step:

{problem}

Provide one reasoning step at a time."""
    })

    for step in range(max_steps):
        input_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        conversation.append({"role": "assistant", "content": response})

        # Check if solution is complete
        if "final answer" in response.lower() or "conclusion" in response.lower():
            break

        # Ask for next step
        conversation.append({
            "role": "user",
            "content": "Continue with the next step."
        })

    return "\n\n".join([msg["content"] for msg in conversation if msg["role"] == "assistant"])

# Example
solution = multi_step_solver("""
A company's revenue grows by 20% each year. If the revenue in 2023 was $100,000,
what will be the total revenue over 5 years (2023-2027)?
""")
print(solution)
```

## Best Practices

### 1. Temperature Settings

```python
# Mathematical/coding tasks - Lower temperature
generation_config_precise = {
    "temperature": 0.5,
    "top_p": 0.95,
    "do_sample": True
}

# Creative/open-ended tasks - Medium temperature
generation_config_balanced = {
    "temperature": 0.6,  # Recommended
    "top_p": 0.95,
    "do_sample": True
}

# Brainstorming/diverse outputs - Higher temperature
generation_config_creative = {
    "temperature": 0.7,  # Max recommended
    "top_p": 0.95,
    "do_sample": True
}

# ❌ AVOID: Temperature > 0.7 causes repetition loops
generation_config_bad = {
    "temperature": 0.9,  # Too high!
    "top_p": 0.95
}
```

### 2. Memory Management

```python
import torch
import gc

def clear_gpu_memory():
    """Clear GPU cache"""
    gc.collect()
    torch.cuda.empty_cache()

# After inference
outputs = model.generate(...)
result = tokenizer.decode(outputs[0])
del outputs
clear_gpu_memory()

# Use context manager for automatic cleanup
class ModelInference:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory()

    def generate(self, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use
with ModelInference() as inference:
    result = inference.generate("Your prompt")
```

### 3. Batch Processing

```python
def batch_inference(prompts: list, batch_size: int = 4) -> list:
    """Process prompts in batches for efficiency"""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id
        )

        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

        # Clear memory after each batch
        del inputs, outputs
        torch.cuda.empty_cache()

    return results
```

### 4. Error Handling

```python
def safe_generate(prompt: str, max_retries: int = 3) -> str:
    """Generate with error handling and retries"""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.95
            )

            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Cleanup
            del inputs, outputs
            torch.cuda.empty_cache()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e):
                if attempt < max_retries - 1:
                    print(f"OOM error, retrying... (attempt {attempt + 1})")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise Exception("Persistent OOM error after retries")
            else:
                raise

        except Exception as e:
            print(f"Error during generation: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                raise

    return "Error: Could not generate response"
```

### 5. Prompt Validation

```python
def validate_and_format_prompt(prompt: str, max_length: int = 4096) -> str:
    """Validate and format prompts before generation"""
    # Remove excessive whitespace
    prompt = " ".join(prompt.split())

    # Check length
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        print(f"Warning: Prompt too long ({len(tokens)} tokens), truncating...")
        tokens = tokens[:max_length]
        prompt = tokenizer.decode(tokens)

    # Ensure no system prompt patterns
    if prompt.strip().startswith("System:"):
        print("Warning: Removing system prompt prefix")
        prompt = prompt.replace("System:", "").strip()

    return prompt

# Use
prompt = validate_and_format_prompt("Your very long prompt here...")
```

### 6. Model Selection Guide

```python
def select_model(task_type: str, hardware: dict) -> str:
    """Recommend model based on task and hardware"""
    vram_gb = hardware.get("vram_gb", 0)

    task_recommendations = {
        "math": {
            "min_quality": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "high_quality": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        },
        "coding": {
            "min_quality": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "high_quality": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        },
        "reasoning": {
            "min_quality": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "high_quality": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        }
    }

    # Select based on VRAM
    if vram_gb < 8:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    elif vram_gb < 16:
        return task_recommendations.get(task_type, {}).get("min_quality", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    elif vram_gb < 80:
        return task_recommendations.get(task_type, {}).get("high_quality", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    else:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Example
recommended = select_model("math", {"vram_gb": 24})
print(f"Recommended model: {recommended}")
```

### 7. Monitoring & Logging

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_with_metrics(prompt: str) -> dict:
    """Generate with performance metrics"""
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = len(inputs["input_ids"][0])

    logger.info(f"Input tokens: {input_tokens}")

    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6
    )
    gen_time = time.time() - gen_start

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens = len(outputs[0])

    total_time = time.time() - start_time
    tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0

    metrics = {
        "response": result,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_time": gen_time,
        "total_time": total_time,
        "tokens_per_second": tokens_per_second
    }

    logger.info(f"Generation metrics: {metrics}")

    return metrics
```

## Resources

### Official

- [DeepSeek GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [Hugging Face Models](https://huggingface.co/deepseek-ai)
- [DeepSeek Website](https://www.deepseek.com/)
- [Research Paper](https://arxiv.org/abs/2501.12948)

### Model Hubs

- [Ollama Library](https://ollama.com/library/deepseek-r1)
- [Together.ai](https://www.together.ai/)
- [Fireworks.ai](https://fireworks.ai/)

### Tools & Libraries

- [vLLM](https://github.com/vllm-project/vllm) - Fast inference
- [SGLang](https://github.com/sgl-project/sglang) - Efficient serving
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Fine-tuning
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient training
- [Outlines](https://github.com/outlines-dev/outlines) - Structured generation

### Tutorials & Guides

- [Together.ai Prompting Guide](https://docs.together.ai/docs/prompting-deepseek-r1)
- [DataCamp R1 Guide](https://www.datacamp.com/tutorial/deepseek-r1-ollama)
- [Microsoft Azure Fine-tuning](https://techcommunity.microsoft.com/blog/machinelearningblog/fine-tuning-deepseek-r1-distill-llama-8b-with-pytorch-fsdp-qlora-on-azure-machin/4377965)

### Community

- [Hugging Face Forums](https://huggingface.co/deepseek-ai)
- r/LocalLLaMA
- [GitHub Discussions](https://github.com/deepseek-ai/DeepSeek-R1/discussions)

## Conclusion

DeepSeek R1 represents a milestone in open-source AI, bringing advanced reasoning capabilities to the community. Its MIT license, competitive performance, and range of model sizes make it suitable for everything from edge deployment to production-scale applications.

**Key Takeaways:**

- **Start Small**: Test with 1.5B-7B distilled models first
- **Use Ollama**: Easiest way to get started locally
- **Simple Prompts**: Avoid few-shot examples and explicit CoT
- **Temperature 0.6**: Critical for preventing repetition loops
- **No System Prompts**: Put all instructions in user messages
- **LoRA for Fine-tuning**: Parameter-efficient customization
- **vLLM for Production**: Fast, scalable inference serving
- **Monitor Performance**: Track tokens/sec and memory usage

The model's native reasoning capabilities, combined with its open-source nature, make it an excellent choice for applications requiring complex problem-solving, mathematical reasoning, code generation, and logical analysis.
