# LoRA (Low-Rank Adaptation)

## Overview

LoRA (Low-Rank Adaptation of Large Language Models) is a parameter-efficient fine-tuning (PEFT) technique that dramatically reduces the computational and memory requirements for adapting large pre-trained models to downstream tasks. Instead of fine-tuning all model parameters, LoRA freezes the pre-trained weights and injects trainable low-rank decomposition matrices into each layer of the transformer architecture.

**Key Advantages:**
- **Memory Efficiency**: Reduces trainable parameters by 10,000x for large models
- **Storage Efficiency**: Adapter weights are tiny (often <100MB vs multi-GB full models)
- **No Inference Latency**: Adapters can be merged into base weights
- **Task Switching**: Multiple adapters can be stored and swapped dynamically
- **Same Performance**: Matches or exceeds full fine-tuning quality on most tasks

**When to Use LoRA:**
- Fine-tuning large language models (7B+ parameters)
- Limited computational resources (consumer GPUs)
- Need to maintain multiple task-specific versions
- Production deployment with multiple use cases
- Rapid experimentation and iteration

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Architecture and Implementation](#architecture-and-implementation)
4. [QLoRA: Quantized LoRA](#qlora-quantized-lora)
5. [Common Patterns](#common-patterns)
6. [Operations](#operations)
7. [Configuration and Hyperparameters](#configuration-and-hyperparameters)
8. [Implementation Examples](#implementation-examples)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

## Fundamentals

### How LoRA Works

Traditional fine-tuning updates all parameters of a pre-trained model:
```
W_finetuned = W_pretrained + ΔW
```

LoRA constrains the update ΔW to have a low-rank structure:
```
W_finetuned = W_pretrained + B·A
```

Where:
- `W ∈ ℝ^(d×k)`: Original pre-trained weight matrix (frozen)
- `B ∈ ℝ^(d×r)`: Trainable low-rank matrix
- `A ∈ ℝ^(r×k)`: Trainable low-rank matrix
- `r << min(d,k)`: Rank of adaptation (typically 4-64)

**Key Insight**: The update to pre-trained weights lies in a low-dimensional subspace. Most adaptation can be captured by low-rank matrices.

### Parameter Reduction

For a weight matrix of shape (d × k):
- **Full fine-tuning**: d × k trainable parameters
- **LoRA**: r × (d + k) trainable parameters

**Example** (GPT-3 175B):
- Full fine-tuning: 175B parameters
- LoRA (r=4): ~18M parameters (~0.01% of original)

## Mathematical Foundation

### Low-Rank Decomposition

LoRA leverages the hypothesis that the update matrix ΔW has a low "intrinsic rank":

```
ΔW = BA
```

Where:
- Rank r is chosen such that r << min(d,k)
- B is initialized with random Gaussian
- A is initialized to zero (ensuring ΔW = 0 at start)

### Forward Pass

Original transformation:
```
h = Wx
```

With LoRA:
```
h = Wx + BAx = Wx + (BA)x
```

The scaling factor α/r is applied:
```
h = Wx + (α/r)·BAx
```

Where α is a constant that controls the magnitude of adaptation.

### Gradient Flow

During backpropagation:
- W remains frozen (no gradients)
- Only A and B receive gradients
- Effective learning occurs in low-dimensional subspace

**Computational Advantage**:
```
Memory for gradients: O(r·(d+k)) vs O(d·k)
Update computation: O(r·(d+k)) vs O(d·k)
```

### Theoretical Justification

**Intrinsic Dimensionality**: Research shows that learned adaptations often lie in low-dimensional subspaces. LoRA exploits this by explicitly constraining updates to low-rank matrices.

**Connection to SVD**: If we perform SVD on a full fine-tuned ΔW:
```
ΔW = UΣV^T
```
Most singular values are close to zero, suggesting low intrinsic rank.

## Architecture and Implementation

### Target Modules

LoRA can be applied to various transformer components:

**Attention Matrices** (most common):
- `q_proj`: Query projection
- `k_proj`: Key projection
- `v_proj`: Value projection
- `o_proj`: Output projection

**Feed-Forward Networks**:
- `gate_proj`: Gate projection (for architectures like LLaMA)
- `up_proj`: Up projection
- `down_proj`: Down projection

**Embedding Layers**:
- Input embeddings
- Output embeddings (LM head)

### Layer Structure

```
┌─────────────────────────────────┐
│    Original Transformer Layer    │
├─────────────────────────────────┤
│                                 │
│  ┌──────────────┐               │
│  │ Pre-trained  │ (Frozen)      │
│  │   Weights W  │────┐          │
│  └──────────────┘    │          │
│                      ▼          │
│  ┌──────┐  ┌──────┐  ┌────┐    │
│  │  B   │  │  A   │  │ +  │───▶│ Output
│  │ (d×r)│─▶│ (r×k)│─▶│    │    │
│  └──────┘  └──────┘  └────┘    │
│  Trainable  Trainable    ^      │
│                          │      │
│                        Input    │
│                                 │
└─────────────────────────────────┘
```

### Initialization Strategy

```python
# LoRA initialization (standard approach)
def init_lora_weights(A, B, r):
    # Initialize A with random Gaussian
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))

    # Initialize B to zero
    nn.init.zeros_(B)

    # This ensures ΔW = BA = 0 at initialization
    # Model starts with pre-trained behavior
```

## QLoRA: Quantized LoRA

QLoRA combines quantization with LoRA for even more efficient fine-tuning. It enables training 65B+ parameter models on consumer GPUs.

### Core Innovations

**4-bit NormalFloat (NF4)**:
- Custom data type optimized for normally distributed weights
- Information-theoretically optimal for Gaussian distributions
- Better preservation of model quality than standard INT4

**Double Quantization**:
- Quantizes the quantization constants themselves
- Saves additional 0.37 bits per parameter on average

**Paged Optimizers**:
- Uses unified memory to handle optimizer state spikes
- Prevents out-of-memory errors during training

### QLoRA Architecture

```
┌─────────────────────────────────────┐
│         QLoRA Architecture          │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────┐               │
│  │ Pre-trained      │               │
│  │ Weights W        │               │
│  │ (Quantized 4-bit)│───┐           │
│  └──────────────────┘   │           │
│        (Frozen)          ▼           │
│                     Dequantize       │
│                          │           │
│                          ▼           │
│  ┌──────┐  ┌──────┐  ┌────┐         │
│  │  B   │  │  A   │  │ +  │────▶    │
│  │(FP16)│─▶│(FP16)│─▶│    │    Output│
│  └──────┘  └──────┘  └────┘         │
│  Trainable  Trainable                │
│                                      │
└──────────────────────────────────────┘
```

### NF4 Quantization

```python
import torch
import bitsandbytes as bnb

# NF4 quantization process
def quantize_nf4(weights):
    """
    Quantize weights to 4-bit NormalFloat
    """
    # Compute normalization constants
    absmax = torch.max(torch.abs(weights))

    # NF4 quantization bins (optimized for normal distribution)
    nf4_bins = [
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495,
        0.0, 0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ]

    # Map weights to nearest bin
    normalized = weights / absmax
    quantized = torch.zeros_like(weights, dtype=torch.uint8)

    for i, val in enumerate(normalized.flatten()):
        # Find closest bin
        idx = min(range(len(nf4_bins)),
                 key=lambda i: abs(nf4_bins[i] - val))
        quantized.view(-1)[i] = idx

    return quantized, absmax

# Dequantization
def dequantize_nf4(quantized, absmax, nf4_bins):
    """
    Dequantize NF4 back to float16
    """
    dequantized = torch.zeros_like(quantized, dtype=torch.float16)

    for i, idx in enumerate(quantized.flatten()):
        dequantized.view(-1)[i] = nf4_bins[idx] * absmax

    return dequantized
```

### Memory Comparison

For a 65B parameter model:

| Method | Memory Required | Trainable Params |
|--------|----------------|------------------|
| Full FP32 Fine-tuning | ~260 GB | 65B |
| Full FP16 Fine-tuning | ~130 GB | 65B |
| LoRA (FP16) | ~80 GB | ~84M (r=8) |
| QLoRA (NF4) | ~48 GB | ~84M (r=8) |

QLoRA makes 65B model fine-tuning possible on a single 48GB GPU!

### QLoRA Training Process

1. **Load base model in 4-bit**:
   - Weights quantized to NF4
   - Stored in GPU memory

2. **Forward pass**:
   - Dequantize weights to FP16 on-the-fly
   - Compute activations in FP16/BF16
   - Apply LoRA adapters (FP16)

3. **Backward pass**:
   - Compute gradients for LoRA adapters only
   - Base model weights remain frozen and quantized

4. **Optimizer step**:
   - Update only LoRA parameters
   - Use paged optimizers for state management

## Common Patterns

### Pattern 1: Single-Task Fine-Tuning

**Use Case**: Adapt a general model to a specific task (e.g., medical Q&A, code generation, sentiment analysis).

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# Train normally
# ... training loop ...

# Save adapter weights only
model.save_pretrained("./lora_medical_qa")
```

### Pattern 2: Multi-Task with Adapter Switching

**Use Case**: One base model, multiple task-specific adapters that can be swapped at runtime.

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Load different adapters for different tasks
medical_model = PeftModel.from_pretrained(base_model, "./lora_medical")
legal_model = PeftModel.from_pretrained(base_model, "./lora_legal")
code_model = PeftModel.from_pretrained(base_model, "./lora_code")

# Use different adapters
def generate_medical(prompt):
    return medical_model.generate(prompt)

def generate_legal(prompt):
    return legal_model.generate(prompt)

# Or dynamically swap adapters
model = PeftModel.from_pretrained(base_model, "./lora_medical")
output1 = model.generate(medical_prompt)

model.set_adapter("legal")  # Switch adapter
output2 = model.generate(legal_prompt)
```

### Pattern 3: Progressive Rank Adaptation

**Use Case**: Start with low rank for fast experimentation, increase for final training.

```python
# Phase 1: Quick exploration with low rank
config_phase1 = LoraConfig(r=4, lora_alpha=8, ...)
model = get_peft_model(base_model, config_phase1)
# Train for few epochs to validate approach

# Phase 2: Higher rank for better performance
config_phase2 = LoraConfig(r=32, lora_alpha=64, ...)
model = get_peft_model(base_model, config_phase2)
# Train to convergence
```

### Pattern 4: Selective Layer Targeting

**Use Case**: Apply LoRA only to specific layers where adaptation is most beneficial.

```python
# Target only attention in middle layers
lora_config = LoraConfig(
    r=16,
    target_modules=[
        "model.layers.12.self_attn.q_proj",
        "model.layers.12.self_attn.v_proj",
        "model.layers.13.self_attn.q_proj",
        "model.layers.13.self_attn.v_proj",
        # ... layers 12-20 only
    ],
    # Or use regex patterns
    # target_modules=r".*layers\.(1[2-9]|20)\.self_attn\.(q|v)_proj",
)
```

### Pattern 5: Merge and Deploy

**Use Case**: Production deployment where you want a single model file without adapter overhead.

```python
from peft import PeftModel

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Merge adapter into base weights
merged_model = peft_model.merge_and_unload()

# Save as standard model (no LoRA dependency)
merged_model.save_pretrained("./merged_model")

# Now can be used without PEFT library
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

### Pattern 6: Multi-Adapter Composition

**Use Case**: Combine multiple adapters for composite capabilities.

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Load and combine multiple adapters
model = PeftModel.from_pretrained(base_model, "./lora_instruction", adapter_name="instruction")
model.load_adapter("./lora_style", adapter_name="style")
model.load_adapter("./lora_domain", adapter_name="domain")

# Use weighted combination
model.set_adapter(["instruction", "style", "domain"])
model.set_adapter_weights([0.5, 0.3, 0.2])  # Weighted mix

# Generate with combined capabilities
output = model.generate(prompt)
```

## Operations

### Training with LoRA

**Basic Training Loop**:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Load and prepare dataset
dataset = load_dataset("your_dataset")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train
trainer.train()

# Save LoRA adapter
model.save_pretrained("./final_lora_adapter")
```

### Training with QLoRA

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Double quantization
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=64,                      # Higher rank for large models
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Rest of training is identical to standard LoRA
# ... training loop ...
```

### Inference with LoRA

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base-model",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("base-model")

# Generate
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Merging Adapters

```python
from peft import PeftModel

# Load model with adapter
base_model = AutoModelForCausalLM.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Method 1: Merge and unload (creates new model)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# Method 2: Merge in place (modifies base model)
peft_model.merge_adapter()
# Now peft_model has merged weights

# Method 3: Unmerge (reverse the merge)
peft_model.unmerge_adapter()
# Back to base weights + adapter separation
```

### Saving and Loading

```python
# Save adapter only (efficient)
model.save_pretrained("./my_lora_adapter")
# Creates: adapter_config.json, adapter_model.bin (~10-100 MB)

# Save with optimizer state for resuming training
trainer.save_model("./checkpoint_dir")
# Creates: adapter files + optimizer state + training state

# Load for inference
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(model, "./my_lora_adapter")

# Load for continued training
model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(model, "./checkpoint_dir", is_trainable=True)
```

### Memory-Efficient Loading

```python
# For large models, load in 8-bit
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    load_in_8bit=True,           # 8-bit quantization
    device_map="auto",           # Automatic device placement
    torch_dtype=torch.float16
)

# Or use model sharding across GPUs
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map="balanced",       # Balance across GPUs
    torch_dtype=torch.float16
)
```

## Configuration and Hyperparameters

### LoraConfig Parameters

```python
from peft import LoraConfig

config = LoraConfig(
    # Core LoRA parameters
    r=8,                           # Rank of adaptation matrices
    lora_alpha=16,                 # Scaling factor (often 2*r)
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    lora_dropout=0.1,              # Dropout for LoRA layers

    # Bias handling
    bias="none",                   # "none", "all", "lora_only"

    # Task type
    task_type="CAUSAL_LM",         # "CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", etc.

    # Advanced options
    fan_in_fan_out=False,          # For Conv1D layers (GPT-2)
    modules_to_save=None,          # Additional modules to train fully

    # Initialization
    init_lora_weights=True,        # Initialize LoRA weights
    layers_to_transform=None,      # Specific layers (None = all)
    layers_pattern=None,           # Regex pattern for layers
)
```

### Parameter Selection Guide

**Rank (r)**:
- **Low (4-8)**: Fast experimentation, simple tasks, limited data
- **Medium (16-32)**: Most production use cases, good balance
- **High (64-128)**: Complex tasks, large datasets, maximum quality
- **Rule of thumb**: Start with 8, increase if underfitting

**Alpha (lora_alpha)**:
- Controls effective learning rate: `effective_lr = (alpha/r) * lr`
- **Common values**: 16, 32, 64
- **Rule of thumb**: Set to 2×r for stability
- Higher alpha = larger updates from adapters

**Target Modules**:

```python
# Minimal (fastest training, least parameters)
target_modules=["q_proj", "v_proj"]

# Balanced (recommended for most cases)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# Maximum (best performance, more parameters)
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # FFN
]

# Using patterns (for specific architectures)
target_modules=r".*\.(q_proj|v_proj|k_proj|o_proj)"
```

**Dropout (lora_dropout)**:
- **0.0**: No regularization, risk of overfitting
- **0.05-0.1**: Standard choice for most tasks
- **0.1-0.2**: High regularization for small datasets
- Applies dropout to LoRA layers during training

**Bias**:
- `"none"`: Don't train bias terms (most common)
- `"all"`: Train all bias parameters
- `"lora_only"`: Only train LoRA bias terms

### Training Hyperparameters

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output
    output_dir="./lora_training",

    # Batch size and accumulation
    per_device_train_batch_size=4,     # Adjust based on GPU memory
    gradient_accumulation_steps=4,      # Effective batch = 4 * 4 = 16

    # Learning rate
    learning_rate=2e-4,                 # LoRA: 1e-4 to 3e-4 typical
    lr_scheduler_type="cosine",         # "linear", "cosine", "constant"
    warmup_ratio=0.03,                  # 3% warmup

    # Training duration
    num_train_epochs=3,
    max_steps=-1,                       # Use epochs instead

    # Optimization
    optim="adamw_torch",                # "adamw_torch", "adamw_8bit", "paged_adamw_8bit"
    weight_decay=0.01,
    max_grad_norm=1.0,                  # Gradient clipping

    # Precision
    fp16=True,                          # Use FP16 (V100, RTX)
    bf16=False,                         # Use BF16 (A100, H100)

    # Logging and saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,                 # Keep only 2 checkpoints
    evaluation_strategy="steps",
    eval_steps=100,

    # Performance
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True,               # Group similar lengths
)
```

### Recommended Configurations by Use Case

**Quick Experimentation**:
```python
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
training_args = TrainingArguments(num_train_epochs=1, learning_rate=3e-4)
```

**Production Fine-tuning (7B model)**:
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05
)
training_args = TrainingArguments(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4
)
```

**Large Model (70B+) with QLoRA**:
```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

## Implementation Examples

### Example 1: Fine-tune LLaMA for Instruction Following

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load instruction dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")

# Format prompts
def format_instruction(example):
    if example.get("input"):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./llama-instruction-lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
    ),
)

trainer.train()
model.save_pretrained("./llama-instruction-lora-final")
```

### Example 2: Multi-Task Adapter Management

```python
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

class MultiTaskLoRAModel:
    def __init__(self, base_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.adapters = {}
        self.current_adapter = None

    def add_adapter(self, name, adapter_path=None, config=None):
        """Add a new adapter"""
        if adapter_path:
            # Load existing adapter
            self.adapters[name] = adapter_path
        elif config:
            # Create new adapter for training
            model = get_peft_model(self.base_model, config)
            self.adapters[name] = model

    def switch_adapter(self, name):
        """Switch to a different adapter"""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found")

        adapter_path = self.adapters[name]
        self.current_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path
        )
        self.current_adapter = name

    def generate(self, prompt, **kwargs):
        """Generate with current adapter"""
        if self.current_adapter is None:
            raise ValueError("No adapter selected")

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.current_model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
manager = MultiTaskLoRAModel("meta-llama/Llama-2-7b-hf")

# Add adapters
manager.add_adapter("medical", "./lora_medical")
manager.add_adapter("legal", "./lora_legal")
manager.add_adapter("code", "./lora_code")

# Use different adapters
manager.switch_adapter("medical")
medical_response = manager.generate("What are symptoms of diabetes?")

manager.switch_adapter("code")
code_response = manager.generate("Write a Python function to sort a list")
```

### Example 3: LoRA with Custom Training Loop

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-medium"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

# Prepare data (example)
train_dataset = ...  # Your dataset
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

# Training loop
model.train()
for epoch in range(3):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        # Prepare inputs
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    print(f"Epoch {epoch+1} - Average Loss: {epoch_loss / len(train_dataloader):.4f}")

# Save adapter
model.save_pretrained("./custom_trained_lora")
```

### Example 4: Evaluation and Comparison

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def evaluate_perplexity(model, tokenizer, dataset, max_samples=100):
    """Evaluate model perplexity on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            if i >= max_samples:
                break

            inputs = tokenizer(example["text"], return_tensors="pt",
                             truncation=True, max_length=512).to("cuda")

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA model
lora_model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")

# Load test dataset
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Evaluate
print("Evaluating base model...")
base_perplexity = evaluate_perplexity(base_model, tokenizer, test_dataset)
print(f"Base model perplexity: {base_perplexity:.2f}")

print("Evaluating LoRA model...")
lora_perplexity = evaluate_perplexity(lora_model, tokenizer, test_dataset)
print(f"LoRA model perplexity: {lora_perplexity:.2f}")

print(f"Improvement: {((base_perplexity - lora_perplexity) / base_perplexity * 100):.2f}%")
```

## Advanced Topics

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes pre-trained weights into magnitude and direction components, applying LoRA to the directional component.

```
W' = m · (W_dir + ΔW_dir)
```

Where:
- `m`: Magnitude component (trained)
- `W_dir`: Directional component (frozen)
- `ΔW_dir`: Low-rank adaptation of direction

**Advantages**:
- Better learning capacity than vanilla LoRA
- More stable training
- Improved performance on complex tasks

```python
from peft import LoraConfig

# Enable DoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,  # Enable DoRA
    target_modules=["q_proj", "v_proj"]
)
```

### AdaLoRA (Adaptive LoRA)

Adaptively allocates rank budget across different weight matrices based on importance.

**Key Ideas**:
- Start with budget of total rank across all layers
- Prune less important singular values during training
- Redistribute rank to more important matrices

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    init_r=12,              # Initial rank
    target_r=8,             # Target rank after pruning
    beta1=0.85,             # Regularization
    beta2=0.85,
    tinit=200,              # Start pruning after tinit steps
    tfinal=1000,            # Final pruning step
    deltaT=10,              # Pruning frequency
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)
```

### LoRA+ (Improved Optimizer)

Uses different learning rates for A and B matrices.

**Insight**: Matrix B (initialized to zero) should learn faster than matrix A.

```python
# Manual implementation
def get_lora_plus_optimizer(model, lr_B=1e-3, lr_A=1e-4, weight_decay=0.01):
    """
    Create optimizer with different learning rates for A and B matrices
    """
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "lora_B" in n],
            "lr": lr_B,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "lora_A" in n],
            "lr": lr_A,
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "lora" not in n and p.requires_grad],
            "lr": lr_A,
            "weight_decay": weight_decay
        }
    ]

    return torch.optim.AdamW(param_groups)

# Usage
optimizer = get_lora_plus_optimizer(model, lr_B=2e-4, lr_A=2e-5)
```

### VeRA (Vector-based Random Matrix Adaptation)

Shares the same low-rank matrices across all layers, using layer-specific scaling vectors.

**Benefits**:
- Even fewer trainable parameters than LoRA
- Maintains competitive performance
- Faster training

```python
# Conceptual structure (not in standard PEFT yet)
# Shared matrices: A_shared, B_shared
# Per-layer: scaling vectors d_i, b_i

# Forward pass for layer i:
# h = Wx + (d_i ⊙ (B_shared · A_shared · x)) ⊙ b_i
```

### LoRA with Mixture of Experts (MoE)

Combine LoRA with MoE for specialized adapters.

```python
import torch.nn as nn

class MoELoRA(nn.Module):
    def __init__(self, base_model, num_experts=4, r=8):
        super().__init__()
        self.base_model = base_model
        self.num_experts = num_experts

        # Multiple LoRA experts
        self.experts = nn.ModuleList([
            get_peft_model(base_model, LoraConfig(r=r))
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # Compute gating scores
        gate_scores = torch.softmax(self.gate(x.mean(dim=1)), dim=-1)

        # Weighted combination of expert outputs
        output = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            output += gate_scores[:, i:i+1] * expert_out

        return output
```

### Spectral Regularization for LoRA

Regularize the singular values of LoRA updates.

```python
def spectral_regularization_loss(lora_A, lora_B, lambda_reg=0.01):
    """
    Regularize singular values of BA to prevent overfitting
    """
    # Compute product
    W_delta = torch.mm(lora_B, lora_A)

    # SVD
    U, S, V = torch.svd(W_delta)

    # Regularization: encourage low-rank structure
    reg_loss = lambda_reg * torch.sum(S)

    return reg_loss

# Add to training loop
loss = model(**inputs).loss + spectral_regularization_loss(model.lora_A, model.lora_B)
```

### LoRA Dropout Variants

**Standard Dropout**:
```python
config = LoraConfig(lora_dropout=0.1)  # Dropout in LoRA layers
```

**Stochastic Depth for LoRA**:
```python
class StochasticLoRA(nn.Module):
    def __init__(self, lora_layer, drop_prob=0.1):
        super().__init__()
        self.lora = lora_layer
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_prob:
            return 0  # Skip LoRA entirely
        else:
            return self.lora(x)
```

## Best Practices

### 1. Choosing Rank

**Start Low, Scale Up**:
```python
# Experimentation phase
r = 4  # Quick iterations

# Validation phase
r = 8  # Verify approach works

# Production phase
r = 16-32  # Maximize performance
```

**Rank vs. Model Size**:
- Small models (1B-7B): r = 8-16
- Medium models (7B-13B): r = 16-32
- Large models (30B-70B): r = 32-64
- Very large models (70B+): r = 64-128

### 2. Target Module Selection

**For Attention-Only Tasks** (Q&A, classification):
```python
target_modules=["q_proj", "v_proj"]  # Minimum
```

**For Generation Tasks** (chat, summarization):
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Recommended
```

**For Maximum Performance** (complex reasoning):
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]  # All
```

### 3. Learning Rate Guidelines

```python
# LoRA typically needs higher LR than full fine-tuning
learning_rate = 1e-4  # Full fine-tuning typical
learning_rate = 2e-4  # LoRA typical
learning_rate = 3e-4  # LoRA with low rank or complex task
```

### 4. Batch Size and Accumulation

```python
# Goal: Effective batch size of 64-128
per_device_batch_size = 4      # Fit in GPU memory
gradient_accumulation_steps = 16  # Effective batch = 4 * 16 = 64
```

### 5. Data Quality over Quantity

**LoRA is data-efficient**:
- 1,000 high-quality examples > 10,000 noisy examples
- Focus on diverse, representative samples
- Remove duplicates and low-quality data

### 6. Monitoring Training

**Key Metrics**:
```python
# Watch for:
# 1. Training loss decreasing steadily
# 2. Validation loss not increasing (no overfitting)
# 3. Gradient norms stable (no exploding/vanishing)
# 4. Learning rate warmup completed

from transformers import TrainerCallback

class MonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}:")
            print(f"  Loss: {logs.get('loss', 'N/A'):.4f}")
            print(f"  LR: {logs.get('learning_rate', 'N/A'):.2e}")
            print(f"  Grad Norm: {logs.get('grad_norm', 'N/A'):.4f}")

trainer = Trainer(..., callbacks=[MonitorCallback()])
```

### 7. Preventing Overfitting

```python
# Multiple strategies:
lora_config = LoraConfig(
    r=8,                    # Lower rank = less capacity
    lora_dropout=0.1,       # Dropout regularization
)

training_args = TrainingArguments(
    weight_decay=0.01,      # L2 regularization
    max_grad_norm=1.0,      # Gradient clipping
    eval_steps=100,         # Frequent evaluation
    save_total_limit=2,     # Don't save too many checkpoints
)

# Early stopping
from transformers import EarlyStoppingCallback
trainer = Trainer(..., callbacks=[EarlyStoppingCallback(patience=3)])
```

### 8. Merging for Production

```python
# When deploying, merge for efficiency
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./lora_adapter")
merged_model = model.merge_and_unload()

# Quantize merged model for inference
from transformers import AutoModelForCausalLM

merged_model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    load_in_8bit=True,  # or load_in_4bit=True
    device_map="auto"
)
```

### 9. Version Control for Adapters

```
project/
├── base_models/
│   └── llama-2-7b/
├── adapters/
│   ├── v1_medical/
│   │   ├── adapter_config.json
│   │   └── adapter_model.bin
│   ├── v2_medical/
│   └── v1_legal/
├── configs/
│   ├── medical_lora.yaml
│   └── legal_lora.yaml
└── training_logs/
```

### 10. Common Pitfalls to Avoid

**1. Using rank that's too high**:
```python
# Bad: r=256 (too high, may overfit)
# Good: r=16 (appropriate for most tasks)
```

**2. Forgetting to set pad_token**:
```python
# Bad: tokenizer without pad_token
# Good:
tokenizer.pad_token = tokenizer.eos_token
```

**3. Not using gradient checkpointing for large models**:
```python
# Good: Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

**4. Training on too little data**:
```python
# Minimum: 100-500 examples
# Recommended: 1,000-10,000 examples
# Ideal: 10,000+ high-quality examples
```

**5. Not testing before merging**:
```python
# Always evaluate adapter before merging
eval_results = trainer.evaluate()
if eval_results['eval_loss'] < threshold:
    merged_model = model.merge_and_unload()
```

### 11. Testing and Validation

```python
def comprehensive_test(model, tokenizer, test_cases):
    """
    Test model on diverse examples
    """
    results = []

    for category, examples in test_cases.items():
        category_results = []

        for example in examples:
            prompt = example['prompt']
            expected = example.get('expected')

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=100)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            category_results.append({
                'prompt': prompt,
                'response': response,
                'expected': expected,
                'match': expected in response if expected else None
            })

        results.append({
            'category': category,
            'results': category_results,
            'success_rate': sum(1 for r in category_results if r['match']) / len(category_results) if expected else None
        })

    return results

# Usage
test_cases = {
    'medical': [
        {'prompt': 'What is diabetes?', 'expected': 'blood sugar'},
        {'prompt': 'Symptoms of COVID-19?', 'expected': 'fever'}
    ],
    'general': [
        {'prompt': 'Capital of France?', 'expected': 'Paris'}
    ]
}

results = comprehensive_test(model, tokenizer, test_cases)
```

### 12. Resource Estimation

**Memory Requirements** (7B model):
```python
# Base model (FP16): ~14 GB
# LoRA adapters (r=16): ~50 MB
# Optimizer states: ~100 MB
# Gradients: ~50 MB
# Activations (batch=4): ~2-4 GB
# Total: ~18-20 GB (fits on RTX 4090)

# With 4-bit quantization:
# Base model (NF4): ~3.5 GB
# Total: ~7-9 GB (fits on RTX 3090)
```

**Training Time** (estimates for 1000 examples):
- 7B model, r=16, 1×A100: ~30 minutes
- 7B model, r=16, 1×RTX 4090: ~1 hour
- 70B model, r=64, 1×A100 (QLoRA): ~4-6 hours

---

## Summary

LoRA revolutionizes fine-tuning by:
1. **Efficiency**: Train massive models on consumer hardware
2. **Flexibility**: Multiple adapters for multiple tasks
3. **Performance**: Match or exceed full fine-tuning
4. **Practicality**: Deployable in production

**Key Takeaways**:
- Start with r=8-16 for most tasks
- Use QLoRA for models >30B parameters
- Target attention layers first, add FFN if needed
- Monitor for overfitting with small datasets
- Merge adapters for production deployment
- Version control your adapters
- Test thoroughly before deployment

LoRA has become the de facto standard for fine-tuning large language models, enabling individuals and small teams to customize state-of-the-art models for their specific needs without massive computational resources.
