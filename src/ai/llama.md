# Llama Models - Meta AI

Complete guide to Meta's Llama family of open-source language models, from setup to fine-tuning and deployment.

## Table of Contents
- [Introduction](#introduction)
- [Model Versions](#model-versions)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Fine-tuning](#fine-tuning)
- [Quantization](#quantization)
- [Inference Optimization](#inference-optimization)
- [Deployment](#deployment)
- [Advanced Techniques](#advanced-techniques)

## Introduction

Llama (Large Language Model Meta AI) is Meta's family of open-source foundation language models. Released as open-weights models, they've become the foundation for countless applications and fine-tuned variants.

### Key Features

- **Open Source**: Freely available weights
- **Strong Performance**: Competitive with closed models
- **Multiple Sizes**: From 1B to 70B+ parameters
- **Commercial Friendly**: Permissive license
- **Active Ecosystem**: Huge community support
- **Efficient**: Optimized for deployment

### Architecture

- **Transformer-based**: Decoder-only architecture
- **RMSNorm**: Root Mean Square Layer Normalization
- **SwiGLU**: Activation function
- **Rotary Embeddings**: Position encoding
- **Grouped-Query Attention**: Efficient attention mechanism

## Model Versions

### Llama 3.2 (Latest)

**Released**: September 2024

#### Llama 3.2 1B/3B (Edge Models)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Features:**
- 1B and 3B parameter versions
- Optimized for mobile and edge devices
- Multilingual support
- 128K context length
- Excellent for on-device inference

#### Llama 3.2 11B/90B (Vision Models)

```python
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image

model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

# Load image
image = Image.open("photo.jpg")

# Create prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What's in this image?"}
        ]
    }
]

# Process and generate
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0], skip_special_tokens=True))
```

**Features:**
- Multimodal (text + vision)
- 11B and 90B variants
- Image understanding
- Visual question answering

### Llama 3.1

**Released**: July 2024

```python
# 8B - Fast, efficient
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# 70B - High capability
model_name = "meta-llama/Llama-3.1-70B-Instruct"

# 405B - Most capable (requires multiple GPUs)
model_name = "meta-llama/Llama-3.1-405B-Instruct"
```

**Features:**
- 128K context window
- Multilingual (8 languages)
- Tool use capabilities
- Improved reasoning
- 8B, 70B, and 405B sizes

### Llama 3

**Released**: April 2024

```python
# 8B
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# 70B
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
```

**Features:**
- 8K context window
- Strong performance
- Better instruction following
- 8B and 70B sizes

### Llama 2

**Released**: July 2023

```python
# 7B
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 13B
model_name = "meta-llama/Llama-2-13b-chat-hf"

# 70B
model_name = "meta-llama/Llama-2-70b-chat-hf"
```

**Features:**
- 4K context window
- 7B, 13B, and 70B sizes
- Still widely used

### Model Comparison

| Model | Parameters | Context | VRAM (FP16) | Use Case |
|-------|------------|---------|-------------|----------|
| Llama 3.2 1B | 1B | 128K | 2GB | Edge/Mobile |
| Llama 3.2 3B | 3B | 128K | 6GB | Edge/Desktop |
| Llama 3.1 8B | 8B | 128K | 16GB | Standard |
| Llama 3.2 11B Vision | 11B | 128K | 22GB | Multimodal |
| Llama 3.1 70B | 70B | 128K | 140GB | High-end |
| Llama 3.2 90B Vision | 90B | 128K | 180GB | Vision tasks |
| Llama 3.1 405B | 405B | 128K | 810GB | Best quality |

## Installation & Setup

### Via Hugging Face Transformers

```bash
# Install dependencies
pip install transformers torch accelerate

# For quantization
pip install bitsandbytes

# For training
pip install peft datasets
```

### Via Ollama (Easy Local Setup)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2

# Run
ollama run llama3.2
```

Python usage:
```python
import requests

def query_ollama(prompt):
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()['response']

result = query_ollama("What is machine learning?")
print(result)
```

### Via llama.cpp (Efficient C++ Implementation)

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download model (GGUF format)
# From Hugging Face or converted locally

# Run inference
./main -m models/llama-3.2-1B-Instruct-Q4_K_M.gguf -p "Hello, how are you?"
```

Python bindings:
```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=35  # Adjust for GPU
)

output = llm(
    "Explain quantum computing",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
)

print(output['choices'][0]['text'])
```

### Via vLLM (Production Inference)

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    tensor_parallel_size=1
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256
)

# Generate
prompts = ["What is AI?", "Explain Python"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Basic Usage

### Simple Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate
prompt = "Write a Python function to calculate factorial:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Chat Format

```python
# Proper chat formatting
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

### Multi-turn Conversation

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."}
]

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
        max_new_tokens=256,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    
    # Add assistant response
    conversation.append({"role": "assistant", "content": response})
    
    return response

# Use
print(chat("What is Python?"))
print(chat("How do I install it?"))
print(chat("Give me a simple example."))
```

### Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Prepare input
messages = [{"role": "user", "content": "Write a short story about AI"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

# Generate in thread
generation_kwargs = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "temperature": 0.8,
    "streamer": streamer
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream output
for text in streamer:
    print(text, end="", flush=True)

thread.join()
```

## Fine-tuning

### QLoRA Fine-tuning (Most Popular)

Efficient fine-tuning with quantization:

```bash
pip install transformers peft accelerate bitsandbytes datasets
```

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Load model with quantization
model_name = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~16M / total: 3B (~0.5%)

# Prepare dataset
dataset = load_dataset("your-dataset")

def format_instruction(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return {"text": text}

dataset = dataset.map(format_instruction)

# Tokenize
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['input_ids'] for f in data])
    }
)

trainer.train()

# Save
model.save_pretrained("./llama-lora")
tokenizer.save_pretrained("./llama-lora")
```

### Using Fine-tuned Model

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA
model = PeftModel.from_pretrained(base_model, "./llama-lora")

# Generate
tokenizer = AutoTokenizer.from_pretrained("./llama-lora")
inputs = tokenizer("Your prompt here", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Full Fine-tuning (Requires More Resources)

```python
from transformers import Trainer, TrainingArguments

# Load model normally (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-fullft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    deepspeed="ds_config.json"  # For multi-GPU
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
```

### Using Axolotl (Simplified Training)

```bash
# Install
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

Create config `llama_qlora.yml`:
```yaml
base_model: meta-llama/Llama-3.2-3B-Instruct
model_type: LlamaForCausalLM

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
  - path: your-dataset
    type: alpaca

num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0002

output_dir: ./llama-qlora-out
```

Train:
```bash
accelerate launch -m axolotl.cli.train llama_qlora.yml
```

## Quantization

### BitsAndBytes Quantization

```python
from transformers import BitsAndBytesConfig

# 8-bit
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# 4-bit (QLoRA)
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config_4bit,
    device_map="auto"
)
```

### GGUF Quantization (llama.cpp)

```bash
# Convert to GGUF
python convert_hf_to_gguf.py \
    --model-dir models/Llama-3.2-3B-Instruct \
    --outfile llama-3.2-3b-instruct.gguf

# Quantize
./quantize \
    llama-3.2-3b-instruct.gguf \
    llama-3.2-3b-instruct-Q4_K_M.gguf \
    Q4_K_M
```

Quantization formats:
- `Q4_0`: 4-bit, fastest, lowest quality
- `Q4_K_M`: 4-bit, good quality (recommended)
- `Q5_K_M`: 5-bit, better quality
- `Q8_0`: 8-bit, high quality

### GPTQ Quantization

```bash
pip install auto-gptq
```

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantize
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantize_config=quantize_config
)

# Save
model.save_quantized("llama-3.2-3b-gptq")

# Load
model = AutoGPTQForCausalLM.from_quantized(
    "llama-3.2-3b-gptq",
    device_map="auto"
)
```

### AWQ Quantization

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM

# Quantize
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
model.save_quantized("llama-3.2-3b-awq")

# Load
model = AutoAWQForCausalLM.from_quantized("llama-3.2-3b-awq")
```

## Inference Optimization

### Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
```

### Batch Inference

```python
# Process multiple prompts efficiently
prompts = [
    "What is Python?",
    "Explain machine learning",
    "How do computers work?"
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
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

# Decode
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### KV Cache Optimization

```python
# Enable static KV cache for faster inference
model.generation_config.cache_implementation = "static"
model.generation_config.max_length = 512

# Or use with generate
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    use_cache=True,
    cache_implementation="static"
)
```

### TensorRT-LLM

```bash
# Build TensorRT engine
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Convert and build
python examples/llama/convert_checkpoint.py \
    --model_dir models/Llama-3.2-3B-Instruct \
    --output_dir ./trt_ckpt \
    --dtype float16

trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16
```

## Deployment

### FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model once at startup
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=True
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

### vLLM Server

```bash
# Start server
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Client
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": "What is AI?",
        "max_tokens": 256
    }'
```

Python client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    prompt="Explain quantum computing",
    max_tokens=256
)

print(response.choices[0].text)
```

### Text Generation Inference (TGI)

```bash
# Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.2-3B-Instruct

# Client
curl http://localhost:8080/generate \
    -X POST \
    -d '{"inputs":"What is Python?","parameters":{"max_new_tokens":256}}' \
    -H 'Content-Type: application/json'
```

### LangChain Integration

```bash
pip install langchain langchain-community
```

```python
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7
)

# LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create chain
template = "Question: {question}\n\nAnswer:"
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

# Use
result = chain.run("What is machine learning?")
print(result)
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
documents = ["Your document text here..."]

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.create_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
query = "What does the document say about AI?"
result = qa_chain.run(query)
print(result)
```

### Function Calling

```python
import json

def get_current_weather(location: str, unit: str = "celsius"):
    """Get current weather for a location"""
    # Simulated function
    return {"location": location, "temperature": 22, "unit": unit}

# Define tools
tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

# System prompt
system_prompt = f"""You are a helpful assistant with access to tools.
Available tools: {json.dumps(tools, indent=2)}

When you need to use a tool, output JSON: {{"tool": "tool_name", "parameters": {{...}}}}
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather in Paris?"}
]

# Generate
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids, max_new_tokens=256)
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

# Parse and execute tool call
if "tool" in response:
    tool_call = json.loads(response)
    if tool_call["tool"] == "get_current_weather":
        result = get_current_weather(**tool_call["parameters"])
        print(f"Weather: {result}")
```

### Constrained Generation

```bash
pip install outlines
```

```python
import outlines

# Load model
model = outlines.models.transformers("meta-llama/Llama-3.2-1B-Instruct")

# JSON schema constraint
schema = """{
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}}
    }
}"""

generator = outlines.generate.json(model, schema)
result = generator("Generate a person profile:")
print(result)

# Regex constraint
phone_pattern = r"\d{3}-\d{3}-\d{4}"
generator = outlines.generate.regex(model, phone_pattern)
phone = generator("Generate a US phone number:")
print(phone)
```

## Best Practices

### 1. Model Selection

```python
# Choose based on requirements
model_selection = {
    "mobile/edge": "meta-llama/Llama-3.2-1B-Instruct",
    "desktop/low_vram": "meta-llama/Llama-3.2-3B-Instruct",
    "standard": "meta-llama/Llama-3.1-8B-Instruct",
    "high_quality": "meta-llama/Llama-3.1-70B-Instruct",
    "vision": "meta-llama/Llama-3.2-11B-Vision-Instruct"
}
```

### 2. Prompt Templates

```python
# Use consistent templates
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

def format_chat(user_message, system=SYSTEM_PROMPT):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message}
    ]
```

### 3. Memory Management

```python
import torch
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# After large operations
outputs = model.generate(...)
result = tokenizer.decode(outputs[0])
del outputs
clear_memory()
```

### 4. Error Handling

```python
def safe_generate(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < max_retries - 1:
                torch.cuda.empty_cache()
                continue
            raise
```

## Resources

### Official
- [Meta Llama](https://llama.meta.com/)
- [Hugging Face Models](https://huggingface.co/meta-llama)
- [Llama Recipes](https://github.com/facebookresearch/llama-recipes)

### Tools
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

### Fine-tuning
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [PEFT](https://github.com/huggingface/peft)

### Community
- r/LocalLLaMA
- Hugging Face Forums
- Discord communities

## Conclusion

Llama models provide a powerful, open-source foundation for AI applications. Whether you're running a 1B model on a mobile device or deploying a 70B model in production, the ecosystem offers tools and techniques for every use case.

Key takeaways:
- **Start small**: Test with 1B/3B models first
- **Quantize**: Use 4-bit for efficient inference
- **Fine-tune**: QLoRA for custom domains
- **Optimize**: vLLM/TGI for production
- **Monitor**: Watch memory and performance

The open-source nature and active community make Llama models an excellent choice for both research and production applications.
