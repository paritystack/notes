# Microsoft Phi Models

## Overview

Microsoft's Phi model family represents a series of Small Language Models (SLMs) that deliver strong performance relative to their size, particularly excelling in reasoning-focused tasks. The Phi models are distinguished by their focus on data quality, strategic use of synthetic data, and efficient architecture that enables deployment on edge devices and local environments.

**Key Characteristics:**
- Small model sizes (3.8B to 14B parameters)
- Strong reasoning capabilities despite compact size
- Open source under MIT license
- Optimized for on-device deployment
- No cloud connectivity required for inference

## Model Family

### Phi-4 (Latest - December 2024)

**Phi-4 (14B parameters)**
- Architecture: Decoder-only transformer
- Parameters: 14 billion
- Default context length: 4096 tokens
- Extended context: 16K tokens (during midtraining)
- Focus: Complex reasoning and mathematical tasks
- Training: Centrally focused on data quality with strategic synthetic data incorporation
- Performance: Strong performance on reasoning benchmarks relative to size

**Phi-4-mini (3.8B parameters)**
- Dense, decoder-only transformer
- Grouped-query attention mechanism
- Vocabulary size: 200,000 tokens
- Shared input-output embeddings
- Optimized for: Speed and efficiency
- Ideal for: Resource-constrained environments

**Phi-4-multimodal (5.6B parameters)**
- Unified architecture integrating: Speech, Vision, Text
- Top performer on Huggingface OpenASR leaderboard (WER: 6.14% as of Feb 2025)
- Previous best: 6.5%
- Use cases: Multi-modal applications requiring speech and vision understanding

### Phi-3 Family

**Phi-3-mini (3.8B parameters)**
- Baseline small model
- Optimized for mobile and edge deployment
- Capable of running on phones

**Phi-3-small**
- Hybrid attention mechanism:
  - Alternating dense attention layers
  - Blocksparse attention layers
- Optimizes KV cache savings
- Maintains long context retrieval performance

**Phi-3-medium (14B parameters)**
- Same tokenizer and architecture as Phi-3-mini
- Architecture specs:
  - 40 attention heads
  - 40 layers
  - Embedding dimension: 5120
- Enhanced capacity for complex tasks

**Phi-3-MoE (Mixture of Experts)**
- Activated parameters: 6.6B
- Total parameters: 42B
- Routing: Top-2 among 16 expert networks
- Expert architecture: Separate GLU networks
- Efficiency: Sparse activation enables large capacity with moderate compute

## Architecture Details

### Core Architecture

```
Model Type: Decoder-only Transformer
Training Recipe:
├── High-quality curated data
├── Strategic synthetic data generation
├── Multi-stage training curriculum
└── Advanced post-training techniques

Key Features:
├── Grouped Query Attention (GQA)
├── Efficient KV cache management
├── Optimized tokenizer (200K vocabulary for Phi-4-mini)
└── Shared input-output embeddings
```

### Attention Mechanisms

**Standard Dense Attention** (Phi-4, Phi-3-mini, Phi-3-medium)
- Full attention across all positions
- Standard transformer architecture
- Grouped-query attention for efficiency

**Hybrid Attention** (Phi-3-small)
- Alternates between dense and blocksparse layers
- Reduces memory footprint
- Maintains performance on long sequences

**MoE Architecture** (Phi-3-MoE)
- 16 expert networks with top-2 routing
- Each token processed by 2 of 16 experts
- Sparse activation reduces compute requirements

## Fine-Tuning

### When to Fine-Tune

Fine-tune Phi models when:
1. Domain-specific language or terminology is required
2. Task-specific behavior needs optimization
3. Custom instruction following is needed
4. Adapting to proprietary data or workflows
5. Improving performance on specific benchmark tasks

### Fine-Tuning Approaches

#### 1. Full Fine-Tuning
- Updates all model parameters
- Highest accuracy potential
- Requires significant compute resources
- Memory intensive

#### 2. LoRA (Low-Rank Adaptation)
- Adds trainable low-rank matrices to attention layers
- Freezes base model weights
- Memory efficient
- Recommended approach for most use cases

#### 3. QLoRA (Quantized LoRA)
- Combines 4-bit quantization with LoRA
- Quantizes base model to 4-bit
- Trains only LoRA adapters in higher precision
- Minimal memory footprint
- Ideal for consumer GPUs

### LoRA Configuration Best Practices

#### Rank and Alpha Settings

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # LoRA rank (8-16 is sufficient baseline)
    lora_alpha=16,           # Alpha = rank for small datasets
    target_modules=[
        "q_proj",            # Query projection
        "k_proj",            # Key projection
        "v_proj",            # Value projection
        "o_proj",            # Output projection
        "gate_proj",         # MLP gate
        "down_proj",         # MLP down
        "up_proj"            # MLP up
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Key Guidelines:**
- **Rank**: 8-16 is sufficient for most tasks (higher ranks not necessarily better)
- **Alpha**: Set `alpha = rank` for small datasets
- **Avoid**: Using `2*rank` or `4*rank` on small datasets (often unstable)
- **Target Modules**: Include all attention and MLP projection layers

#### Phi-2 Specific Configuration

```python
# Phi-2 uses Wqkv instead of separate q/k/v projections
lora_config_phi2 = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Hyperparameters

#### Learning Rate

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    learning_rate=2e-5,              # Start conservative
    lr_scheduler_type="constant",     # Constant schedule works well
    warmup_steps=100,
    max_steps=1000,
    # Alternative learning rates to try:
    # 5e-5: More aggressive
    # 8e-4: Maximum recommended for LoRA
)
```

**Guidelines:**
- **DO NOT** use high learning rates (1e-3, 2e-4) with LoRA
- **Recommended range**: 2e-5 to 8e-4
- **Start with**: 2e-5 or 5e-5 for safety
- **Schedule**: Constant learning rate (per QLoRA author Tim Dettmers)
- **Warmup**: 100-500 steps helps stabilization

#### Precision and Memory Management

```python
from transformers import TrainingArguments, BitsAndBytesConfig
import torch

# Use bfloat16 for training (NOT fp16)
training_args = TrainingArguments(
    bf16=True,                        # Use bfloat16
    fp16=False,                       # Avoid fp16 (causes NaN errors)
    gradient_checkpointing=True,      # Reduce memory usage
    gradient_accumulation_steps=4,    # Effective batch size = batch * accum
    per_device_train_batch_size=1,    # Adjust based on memory
    optim="paged_adamw_8bit",        # Memory-efficient optimizer
)

# QLoRA quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

**Key Points:**
- **Use bfloat16**: Better dynamic range, fewer NaN issues than fp16
- **Avoid fp16**: Known to cause NaN errors with Phi-2
- **Gradient Checkpointing**: Trades compute for memory
- **Gradient Accumulation**: Simulates larger batch sizes
- **Optimizer Choices**:
  - `paged_adamw_8bit`: Best balance (recommended)
  - `adamw_torch`: Standard but memory intensive
  - `sgd`: Memory efficient but slower convergence

#### Batch Size Strategy

```python
# Strategy 1: Small batch with gradient accumulation
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
# Effective batch size = 1 * 8 = 8

# Strategy 2: Larger batch if memory allows
per_device_train_batch_size = 4
gradient_accumulation_steps = 2
# Effective batch size = 4 * 2 = 8
```

**Considerations:**
- Check GPU memory with long context lengths (4K, 8K tokens)
- OOM errors common with large context + large batch
- Use gradient checkpointing if memory constrained
- Monitor actual GPU utilization

### Complete Fine-Tuning Example

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# 1. Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load model and tokenizer
model_id = "microsoft/phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Prepare dataset
dataset = load_dataset("your-dataset")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./phi-4-finetuned",
    learning_rate=2e-5,
    lr_scheduler_type="constant",
    warmup_steps=100,
    max_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    optim="paged_adamw_8bit",
    report_to="tensorboard"
)

# 7. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

# 8. Save
model.save_pretrained("./phi-4-lora-adapters")
tokenizer.save_pretrained("./phi-4-lora-adapters")
```

### Data Preparation Best Practices

#### Dataset Format

```python
# Instruction-following format
data = [
    {
        "instruction": "Explain quantum computing",
        "input": "",
        "output": "Quantum computing is..."
    },
    {
        "instruction": "Translate to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]

# Convert to prompt template
def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# Apply to dataset
formatted_data = [format_instruction(item) for item in data]
```

#### Dataset Size Guidelines

- **Minimum**: 100-500 high-quality examples
- **Optimal**: 1,000-10,000 examples for specialized tasks
- **Large-scale**: 10,000+ for broad domain adaptation

**Quality over Quantity:**
- Clean, well-formatted data is critical
- Remove duplicates and low-quality samples
- Balance class distributions
- Include diverse examples

## Common Patterns

### 1. Prompt Engineering

#### Basic Completion

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

prompt = "Explain the concept of recursion in programming:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

#### Instruction-Following

```python
instruction_prompt = """### Instruction:
Write a Python function to calculate the Fibonacci sequence.

### Response:"""

inputs = tokenizer(instruction_prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

#### Few-Shot Learning

```python
few_shot_prompt = """Classify the sentiment of movie reviews.

Review: "This movie was amazing! Best film of the year."
Sentiment: Positive

Review: "Terrible acting and boring plot."
Sentiment: Negative

Review: "It was okay, nothing special."
Sentiment: Neutral

Review: "Absolutely loved every minute of it!"
Sentiment:"""

# Model continues with prediction
```

### 2. Chain-of-Thought Reasoning

```python
cot_prompt = """Solve this math problem step by step:

Problem: If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?

Let's solve this step by step:
"""

# Phi-4 excels at mathematical reasoning with CoT prompts
```

### 3. Context Window Management

```python
# For long documents, use sliding window approach
def process_long_document(text, chunk_size=3000, overlap=500):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    results = []
    for chunk in chunks:
        prompt = f"Summarize the following text:\n\n{chunk}"
        # Process each chunk
        results.append(generate(prompt))

    return results
```

### 4. Multi-Modal Applications (Phi-4-multimodal)

```python
# Phi-4-multimodal supports vision, speech, and text
from transformers import AutoProcessor, AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained("microsoft/phi-4-multimodal")
processor = AutoProcessor.from_pretrained("microsoft/phi-4-multimodal")

# Image + Text
image = load_image("path/to/image.jpg")
prompt = "Describe what you see in this image"
inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs)

# Speech + Text
audio = load_audio("path/to/audio.wav")
prompt = "Transcribe this audio"
inputs = processor(text=prompt, audio=audio, return_tensors="pt")
outputs = model.generate(**inputs)
```

### 5. Retrieval-Augmented Generation (RAG)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# Setup vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Setup Phi model
phi_pipeline = HuggingFacePipeline.from_model_id(
    model_id="microsoft/phi-4",
    task="text-generation",
    device=0
)

# RAG pipeline
def rag_query(question):
    # Retrieve relevant context
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate answer with context
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    return phi_pipeline(prompt)
```

### 6. Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Generate in separate thread
generation_kwargs = dict(
    inputs=input_ids,
    streamer=streamer,
    max_length=500,
    temperature=0.7
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream output
for text in streamer:
    print(text, end="", flush=True)

thread.join()
```

## Operations

### Deployment Options

#### 1. Local Deployment (PyTorch)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Standard loading
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

# Inference
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 2. Quantized Deployment

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    quantization_config=bnb_config,
    device_map="auto"
)

# Memory usage:
# - Phi-2 unquantized: ~6.5GB VRAM
# - Phi-2 4-bit NF4: ~2.1GB loading, ~5GB during inference
# - Phi-4 unquantized: ~14.96GB
# - Phi-4 4-bit: ~5.42GB (~64% reduction)
```

#### 3. GGUF Format (llama.cpp)

```bash
# Download GGUF quantized models
# Available quantizations: Q2_K, Q4_K, Q6_K, Q8_0

# Using llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run inference
./main -m phi-2-Q4_K.gguf -p "Your prompt here" -n 200
```

#### 4. ONNX Runtime

```python
from optimum.onnxruntime import ORTModelForCausalLM

# Export to ONNX
model = ORTModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    export=True
)

# Optimized inference
model.save_pretrained("phi-4-onnx")
```

#### 5. Mobile/Edge Deployment

```python
# Platform-specific optimizations:

# Intel OpenVINO (x86 processors)
# - Supports INT4, INT8, FP16, FP32
# - Optimized for Intel CPUs and GPUs

# Qualcomm QNN (Snapdragon)
# - Optimized for mobile ARM processors
# - Hardware acceleration support

# Apple MLX (Apple Silicon)
# - Native M1/M2/M3 optimization
# - Metal acceleration

# NVIDIA CUDA (NVIDIA GPUs)
# - Full GPU acceleration
# - TensorRT optimization
```

### Inference Optimization

#### 1. Batch Processing

```python
# Process multiple prompts efficiently
prompts = [
    "Translate to Spanish: Hello world",
    "Summarize: The quick brown fox...",
    "Calculate: 15 * 24"
]

# Batch tokenization
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Batch generation
outputs = model.generate(
    **inputs,
    max_length=100,
    pad_token_id=tokenizer.pad_token_id
)

# Decode all
results = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
```

#### 2. KV Cache Optimization

```python
# Enable past_key_values caching for faster generation
def generate_with_cache(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")

    # First token
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:].argmax(dim=-1)

    generated = [next_token.item()]

    # Subsequent tokens use cache
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token.item())

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)
```

#### 3. Flash Attention

```python
# Use Flash Attention 2 for faster inference
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Requires flash-attn package
    device_map="auto"
)

# Significant speedup for long sequences
```

#### 4. Speculative Decoding

```python
# Use smaller model for draft, larger for verification
draft_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini")
target_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# Can achieve 2-3x speedup with quality preservation
```

### Quantization Deep Dive

#### Quantization Methods

**1. Post-Training Quantization (PTQ)**
- No retraining required
- Quick conversion process
- Slight accuracy degradation
- Methods: Dynamic, Static, Weight-only

**2. Quantization-Aware Training (QAT)**
- Retrains with quantization in mind
- Better accuracy preservation
- Longer process
- More compute intensive

#### Quantization Formats Comparison

| Format | Bits | Size (Phi-4) | Speed | Quality | Use Case |
|--------|------|--------------|-------|---------|----------|
| FP32 | 32 | ~56GB | Baseline | Best | Training |
| FP16 | 16 | ~28GB | 1.5-2x | Excellent | GPU inference |
| BF16 | 16 | ~28GB | 1.5-2x | Excellent | Training & inference |
| INT8 | 8 | ~14GB | 2-3x | Very Good | Production |
| NF4 | 4 | ~5.4GB | 1.3x* | Good | Memory-constrained |
| Q4_K | 4 | ~5.5GB | 2-4x | Good | Edge devices |
| Q2_K | 2 | ~3GB | 3-5x | Fair | Extreme edge |

*4-bit inference slower than FP16 but enables larger models in limited memory

#### Implementation Examples

**BitsAndBytes Quantization**

```python
from transformers import BitsAndBytesConfig
import torch

# 8-bit quantization
config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# 4-bit quantization (NF4)
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # Double quantization
    bnb_4bit_quant_type="nf4",            # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    quantization_config=config_4bit,
    device_map="auto"
)
```

**GPTQ Quantization (Auto-Round)**

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    "microsoft/phi-4",
    quantize_config=quantize_config
)

# Quantize with calibration data
model.quantize(calibration_data)

# Save
model.save_quantized("phi-4-gptq")
```

**AWQ Quantization**

```python
from awq import AutoAWQForCausalLM

# Load model
model = AutoAWQForCausalLM.from_pretrained("microsoft/phi-4")

# Quantize
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})

# Save
model.save_quantized("phi-4-awq")
```

### Performance Benchmarks

#### Decoding Speed (Phi-2)

| Configuration | Tokens/Second | Memory (VRAM) |
|---------------|---------------|---------------|
| FP16 | 21 | 6.5GB |
| 4-bit NF4 | 15.7 | 2.1GB (load), 5GB (inference) |

#### Memory Footprint (Phi-4)

| Configuration | Memory | Reduction |
|---------------|---------|-----------|
| Unquantized | 14.96GB | - |
| 4-bit | 5.42GB | 64% |
| 2-bit | ~3GB | 80% |

### Serving Architecture

#### Single Model Serving

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerationRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        temperature=request.temperature
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### vLLM High-Performance Serving

```python
from vllm import LLM, SamplingParams

# Initialize with PagedAttention
llm = LLM(
    model="microsoft/phi-4",
    tensor_parallel_size=1,
    dtype="half",
    max_model_len=4096
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

# Batch inference
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

#### Text Generation Inference (TGI)

```bash
# Run with Docker
docker run --gpus all \
  -p 8080:80 \
  -v $(pwd)/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id microsoft/phi-4 \
  --max-total-tokens 4096 \
  --max-input-length 3584

# Query the endpoint
curl http://localhost:8080/generate \
  -X POST \
  -d '{"inputs":"What is machine learning?","parameters":{"max_new_tokens":200}}' \
  -H 'Content-Type: application/json'
```

### Monitoring and Evaluation

#### Performance Metrics

```python
import time
import torch

def benchmark_model(model, tokenizer, prompt, runs=10):
    # Warmup
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_length=100)

    # Benchmark
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        outputs = model.generate(**inputs, max_length=100)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_sec = tokens_generated / avg_time

    return {
        "avg_time": avg_time,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_sec
    }

# Run benchmark
results = benchmark_model(model, tokenizer, "Explain quantum computing:")
print(f"Average generation time: {results['avg_time']:.2f}s")
print(f"Tokens per second: {results['tokens_per_second']:.2f}")
```

#### Quality Evaluation

```python
from evaluate import load

# Perplexity
perplexity = load("perplexity")
results = perplexity.compute(predictions=predictions, model_id="microsoft/phi-4")

# BLEU score (for translation tasks)
bleu = load("bleu")
results = bleu.compute(predictions=predictions, references=references)

# ROUGE score (for summarization)
rouge = load("rouge")
results = rouge.compute(predictions=predictions, references=references)
```

## Advanced Techniques

### 1. Model Merging

```python
from transformers import AutoModelForCausalLM
import torch

# Load base and fine-tuned models
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")
ft_model = AutoModelForCausalLM.from_pretrained("./phi-4-finetuned")

# Merge with weighted average
alpha = 0.7  # Weight for fine-tuned model

for name, param in base_model.named_parameters():
    if name in ft_model.state_dict():
        param.data = alpha * ft_model.state_dict()[name] + (1 - alpha) * param.data

base_model.save_pretrained("./phi-4-merged")
```

### 2. Multi-Adapter Loading

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# Load multiple adapters
model_with_adapter1 = PeftModel.from_pretrained(base_model, "./adapter1")
model_with_adapter2 = PeftModel.from_pretrained(base_model, "./adapter2")

# Switch between adapters dynamically
def generate_with_adapter(prompt, adapter_name):
    if adapter_name == "adapter1":
        model = model_with_adapter1
    else:
        model = model_with_adapter2

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0])
```

### 3. Constrained Generation

```python
from transformers import LogitsProcessor

class ForceWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, force_word_ids):
        self.force_word_ids = force_word_ids

    def __call__(self, input_ids, scores):
        if len(input_ids[0]) in self.force_word_ids:
            word_ids = self.force_word_ids[len(input_ids[0])]
            mask = torch.full_like(scores, float('-inf'))
            mask[:, word_ids] = 0
            scores = scores + mask
        return scores

# Use in generation
logits_processor = LogitsProcessorList([
    ForceWordsLogitsProcessor({5: [tokenizer.encode("yes")[0]]})
])

outputs = model.generate(
    **inputs,
    logits_processor=logits_processor
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM)

**Solutions:**
- Enable gradient checkpointing
- Reduce batch size
- Increase gradient accumulation steps
- Use quantization (4-bit or 8-bit)
- Reduce sequence length
- Use DeepSpeed ZeRO optimization

```python
# Example fix
training_args = TrainingArguments(
    per_device_train_batch_size=1,        # Reduce from 4 to 1
    gradient_accumulation_steps=16,        # Increase from 4 to 16
    gradient_checkpointing=True,           # Enable checkpointing
    deepspeed="ds_config.json"            # Use DeepSpeed
)
```

#### 2. NaN Loss During Training

**Causes:**
- Using fp16 instead of bfloat16
- Learning rate too high
- Gradient explosion

**Solutions:**
```python
# Use bfloat16
training_args = TrainingArguments(
    bf16=True,
    fp16=False,
    learning_rate=2e-5,              # Lower learning rate
    max_grad_norm=1.0,               # Gradient clipping
)
```

#### 3. Slow Inference

**Solutions:**
- Use quantization
- Enable Flash Attention 2
- Batch requests
- Use KV cache
- Consider vLLM or TGI for serving

#### 4. Poor Fine-Tuning Results

**Diagnostics:**
- Check data quality and format
- Verify learning rate and schedule
- Monitor training loss curve
- Evaluate on validation set
- Check for overfitting

**Solutions:**
- Increase dataset size
- Adjust learning rate
- Add regularization (dropout)
- Use early stopping
- Try different LoRA ranks

## Best Practices Summary

### Training
1. ✅ Use bfloat16 for training (not fp16)
2. ✅ Start with learning rate 2e-5 to 5e-5
3. ✅ Use constant learning rate schedule
4. ✅ Set LoRA rank 8-16 (higher not always better)
5. ✅ Enable gradient checkpointing for memory
6. ✅ Use QLoRA for consumer GPUs
7. ✅ Monitor validation metrics to prevent overfitting

### Inference
1. ✅ Quantize to 4-bit for memory-constrained devices
2. ✅ Use Flash Attention 2 for long sequences
3. ✅ Enable KV cache for faster generation
4. ✅ Batch requests when possible
5. ✅ Use vLLM or TGI for production serving
6. ✅ Profile and monitor performance metrics

### Deployment
1. ✅ Choose quantization based on hardware/accuracy tradeoff
2. ✅ Use platform-specific optimizations (OpenVINO, QNN, MLX)
3. ✅ Implement proper error handling and retries
4. ✅ Monitor memory usage and latency
5. ✅ Cache frequent requests when appropriate
6. ✅ Set appropriate timeout values

## Resources

### Official Links
- **Hugging Face Hub**: https://huggingface.co/microsoft/phi-4
- **Azure AI**: https://azure.microsoft.com/en-us/products/phi
- **Phi-4 Technical Report**: https://www.microsoft.com/en-us/research/publication/phi-4-technical-report/
- **Phi-3 Technical Report**: https://arxiv.org/abs/2404.14219

### Tools and Libraries
- **Transformers**: https://github.com/huggingface/transformers
- **PEFT**: https://github.com/huggingface/peft
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes
- **vLLM**: https://github.com/vllm-project/vllm
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Text Generation Inference**: https://github.com/huggingface/text-generation-inference

### Community
- **Phi-3 Cookbook**: https://github.com/microsoft/Phi-3CookBook
- **Discussions**: https://huggingface.co/microsoft/phi-4/discussions
- **Issues**: https://github.com/microsoft/phi-4/issues

## License

Microsoft Phi models are released under the **MIT License**, allowing commercial use, modification, and distribution with minimal restrictions.
