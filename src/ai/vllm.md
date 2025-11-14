# vLLM: High-Performance LLM Inference and Serving

## Overview

vLLM is a fast and easy-to-use library for large language model (LLM) inference and serving. It's designed to achieve high throughput and efficient memory management through innovative techniques like PagedAttention and continuous batching.

### Key Features

- **High Performance**: 10-20x higher throughput than HuggingFace Transformers
- **PagedAttention**: Efficient memory management inspired by virtual memory and paging in OS
- **Continuous Batching**: Dynamic request batching for optimal GPU utilization
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Multi-GPU Support**: Tensor parallelism and pipeline parallelism
- **Quantization**: Support for AWQ, GPTQ, SqueezeLLM, and more
- **Streaming Output**: Real-time token generation
- **LoRA Support**: Efficient fine-tuned model serving

### Why vLLM?

- **Memory Efficiency**: Up to 2x improvement in memory usage through PagedAttention
- **Throughput**: Handles concurrent requests efficiently with continuous batching
- **Ease of Use**: Simple Python API and OpenAI-compatible server
- **Production Ready**: Battle-tested in real-world deployments

## Core Concepts

### PagedAttention

PagedAttention is the key innovation that makes vLLM efficient:

- **Problem**: Traditional LLM inference wastes memory storing KV caches contiguously
- **Solution**: Store KV caches in non-contiguous memory blocks (pages)
- **Benefits**:
  - Eliminates memory fragmentation
  - Enables sharing KV caches across requests (for parallel sampling)
  - Allows preemption and swapping of requests

**Key Parameters**:
- `block_size`: Size of each memory block (typically 16 tokens)
- `max_num_seqs`: Maximum number of sequences processed simultaneously
- `max_num_batched_tokens`: Maximum tokens in a batch

### Continuous Batching

Unlike traditional static batching, vLLM uses continuous (dynamic) batching:

- **Static Batching**: Wait for all sequences to complete before processing new batch
- **Continuous Batching**: Add new requests as soon as existing ones complete
- **Result**: Higher GPU utilization and lower latency

### Memory Management

vLLM's memory hierarchy:
1. **GPU Memory**: Primary KV cache storage
2. **CPU Memory**: Swap space for preempted requests
3. **Disk**: Optional persistent cache storage

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+
- GPU with compute capability 7.0+ (V100, T4, A100, H100, etc.)

### Installation Methods

#### Via pip (Recommended)

```bash
# Install vLLM with CUDA 12.1
pip install vllm

# Or with specific CUDA version
pip install vllm-cuda118  # For CUDA 11.8
pip install vllm-cuda121  # For CUDA 12.1
```

#### Via Docker

```bash
# Pull official image
docker pull vllm/vllm-openai:latest

# Run server
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

#### From Source

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Verification

```bash
# Test installation
python -c "import vllm; print(vllm.__version__)"
```

## Common Operations

### 1. Starting a vLLM Server

#### Basic Server Start

```bash
# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

#### Production Server Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-model-len 4096 \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name llama2-70b
```

**Key Parameters**:
- `--model`: HuggingFace model name or local path
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--pipeline-parallel-size`: Number of pipeline stages
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--max-num-seqs`: Max concurrent sequences
- `--max-model-len`: Maximum sequence length
- `--dtype`: Data type (auto, half, float16, bfloat16, float)
- `--quantization`: Quantization method (awq, gptq, squeezellm)

### 2. Python API Usage

#### Basic Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# Single prompt
prompt = "Explain quantum computing in simple terms:"
outputs = llm.generate(prompt, sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
```

#### Batch Processing

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9
)

# Multiple prompts
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?"
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# Batch generation
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

#### Advanced Sampling Parameters

```python
sampling_params = SamplingParams(
    # Temperature sampling
    temperature=0.8,          # Randomness (0=deterministic, 1+=creative)
    top_p=0.95,              # Nucleus sampling
    top_k=50,                # Top-k sampling

    # Length control
    max_tokens=1024,         # Maximum tokens to generate
    min_tokens=10,           # Minimum tokens to generate

    # Stopping conditions
    stop=["</s>", "\n\n"],   # Stop sequences

    # Penalties
    presence_penalty=0.1,    # Penalize repeated topics
    frequency_penalty=0.1,   # Penalize repeated tokens
    repetition_penalty=1.1,  # Alternative repetition control

    # Beam search
    n=1,                     # Number of completions
    best_of=1,               # Generate best_of and return n best
    use_beam_search=False,   # Use beam search instead of sampling

    # Other
    logprobs=None,           # Return log probabilities
    skip_special_tokens=True # Skip special tokens in output
)
```

### 3. OpenAI-Compatible API

#### Using with OpenAI Python Client

```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM doesn't require API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain vLLM in one sentence."}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

#### Streaming Responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Streaming chat completion
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Write a short story."}],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Using with curl

```bash
# Completion request
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.7
    }'

# Chat completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100
    }'
```

### 4. Streaming in Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=512,
    stream=True  # Enable streaming
)

prompt = "Write a detailed explanation of vLLM:"

# Stream tokens as they're generated
for output in llm.generate(prompt, sampling_params):
    for token_output in output.outputs:
        print(token_output.text, end="", flush=True)
```

## Advanced Features

### Multi-GPU Configuration

#### Tensor Parallelism

Split model layers across multiple GPUs:

```python
from vllm import LLM

# Use 4 GPUs with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    dtype="bfloat16"
)
```

```bash
# Server with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4
```

**Best for**: Large models that don't fit on single GPU

#### Pipeline Parallelism

Split model vertically across pipeline stages:

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=2,
    tensor_parallel_size=2  # Can combine both
)
```

**Best for**: Very large models with high throughput requirements

### Quantization

#### AWQ (Activation-aware Weight Quantization)

```python
from vllm import LLM

# Load AWQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half"
)
```

```bash
# Server with AWQ
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-AWQ \
    --quantization awq \
    --dtype half
```

#### GPTQ

```python
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq"
)
```

#### SqueezeLLM

```python
llm = LLM(
    model="squeeze-ai-lab/sq-llama-2-7b-w4",
    quantization="squeezellm"
)
```

**Quantization Benefits**:
- **AWQ**: 4-bit quantization, minimal accuracy loss, fast inference
- **GPTQ**: 4-bit quantization, good for memory-constrained deployments
- **SqueezeLLM**: Ultra-low bit quantization with sparse matrix multiplication

### LoRA Adapters

Serve multiple LoRA adapters with a single base model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules \
        sql-lora=/path/to/sql-adapter \
        code-lora=/path/to/code-adapter \
    --max-lora-rank 64
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Use specific LoRA adapter
response = client.chat.completions.create(
    model="sql-lora",  # Specify LoRA adapter name
    messages=[{"role": "user", "content": "Generate SQL query"}]
)
```

### Speculative Decoding

Use a smaller draft model to speed up generation:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",  # Target model
    speculative_model="meta-llama/Llama-2-7b-hf",  # Draft model
    num_speculative_tokens=5
)
```

**Benefits**: 1.5-2x speedup for large models with minimal quality impact

## Configuration & Optimization

### Memory Optimization

#### GPU Memory Utilization

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.95,  # Use 95% of GPU memory
    swap_space=4  # 4GB CPU swap space
)
```

**Guidelines**:
- Start with `0.9` and increase if no OOM errors
- Leave headroom for CUDA kernels and buffers
- Use `swap_space` for handling request spikes

#### Block Size and Batching

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    block_size=16,  # Tokens per block (default: 16)
    max_num_seqs=256,  # Max concurrent sequences
    max_num_batched_tokens=8192  # Max tokens per batch
)
```

**Tuning Tips**:
- Larger `block_size`: Better memory efficiency, less flexibility
- Larger `max_num_seqs`: Higher throughput, more memory usage
- `max_num_batched_tokens`: Balance throughput vs. latency

### Performance Tuning

#### For High Throughput

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=512,  # High concurrency
    gpu_memory_utilization=0.95,
    dtype="bfloat16",
    enforce_eager=False,  # Use CUDA graph
    max_model_len=2048  # Limit sequence length
)
```

#### For Low Latency

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    max_num_seqs=32,  # Lower concurrency
    gpu_memory_utilization=0.8,
    dtype="float16"
)
```

#### Data Types

```python
# Options: auto, half, float16, bfloat16, float32
llm = LLM(model="...", dtype="bfloat16")
```

**Recommendations**:
- `bfloat16`: Best for A100/H100, good numerical stability
- `float16`: Good for V100/T4, faster than float32
- `auto`: Let vLLM choose based on model and hardware

### Environment Variables

```bash
# CUDA optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO  # For debugging multi-GPU

# vLLM configuration
export VLLM_USE_MODELSCOPE=True  # Use ModelScope hub
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Use Flash Attention
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # Worker process method

# Logging
export VLLM_LOGGING_LEVEL=INFO
```

## Common Patterns

### Pattern 1: Production API Server

```python
# production_server.py
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize engine
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    max_num_seqs=256
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        request_id = f"req-{asyncio.current_task().get_name()}"
        results_generator = engine.generate(
            request.prompt,
            sampling_params,
            request_id
        )

        final_output = None
        async for output in results_generator:
            final_output = output

        return {"text": final_output.outputs[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Pattern 2: Batch Processing Pipeline

```python
# batch_processor.py
from vllm import LLM, SamplingParams
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            max_num_seqs=batch_size,
            gpu_memory_utilization=0.95
        )
        self.batch_size = batch_size

    def process_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> List[str]:
        """Process a batch of prompts"""
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def process_large_dataset(
        self,
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> List[str]:
        """Process dataset in batches"""
        results = []

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_results = self.process_batch(batch, sampling_params)
            results.extend(batch_results)

            print(f"Processed {min(i + self.batch_size, len(prompts))}/{len(prompts)}")

        return results

# Usage
processor = BatchProcessor("meta-llama/Llama-2-7b-hf", batch_size=64)
prompts = ["Prompt 1", "Prompt 2", ...]  # Large dataset
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
results = processor.process_large_dataset(prompts, sampling_params)
```

### Pattern 3: Dynamic Request Routing

```python
# router.py
from vllm import LLM, SamplingParams
from enum import Enum
from typing import Dict

class ModelSize(Enum):
    SMALL = "7b"
    MEDIUM = "13b"
    LARGE = "70b"

class ModelRouter:
    def __init__(self):
        self.models: Dict[ModelSize, LLM] = {
            ModelSize.SMALL: LLM("meta-llama/Llama-2-7b-hf"),
            ModelSize.MEDIUM: LLM(
                "meta-llama/Llama-2-13b-hf",
                tensor_parallel_size=2
            ),
            ModelSize.LARGE: LLM(
                "meta-llama/Llama-2-70b-hf",
                tensor_parallel_size=4
            )
        }

    def route_request(self, prompt: str, complexity: str = "auto") -> str:
        """Route request to appropriate model based on complexity"""
        if complexity == "auto":
            # Simple heuristic: route by prompt length
            model_size = (
                ModelSize.LARGE if len(prompt) > 1000
                else ModelSize.MEDIUM if len(prompt) > 500
                else ModelSize.SMALL
            )
        else:
            model_size = ModelSize[complexity.upper()]

        llm = self.models[model_size]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        output = llm.generate(prompt, sampling_params)

        return output[0].outputs[0].text

# Usage
router = ModelRouter()
result = router.route_request("Short question", complexity="auto")
```

### Pattern 4: Error Handling and Retries

```python
# robust_client.py
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustVLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        messages: list,
        model: str = "default",
        **kwargs
    ) -> str:
        """Generate with automatic retries"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_with_fallback(
        self,
        messages: list,
        primary_model: str,
        fallback_model: str,
        **kwargs
    ) -> tuple[str, str]:
        """Try primary model, fallback to secondary on failure"""
        try:
            result = self.generate(messages, model=primary_model, **kwargs)
            return result, primary_model
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, using fallback")
            result = self.generate(messages, model=fallback_model, **kwargs)
            return result, fallback_model

# Usage
client = RobustVLLMClient()
messages = [{"role": "user", "content": "Hello!"}]
response, model_used = client.generate_with_fallback(
    messages,
    primary_model="llama-70b",
    fallback_model="llama-7b"
)
```

### Pattern 5: Monitoring and Metrics

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from vllm import LLM, SamplingParams
import time
from typing import List

# Prometheus metrics
REQUEST_COUNT = Counter('vllm_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('vllm_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('vllm_active_requests', 'Active requests')
TOKENS_GENERATED = Counter('vllm_tokens_generated_total', 'Total tokens generated')
REQUEST_ERRORS = Counter('vllm_request_errors_total', 'Total errors')

class MonitoredLLM:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name)
        # Start Prometheus metrics server
        start_http_server(9090)

    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        REQUEST_COUNT.inc(len(prompts))
        ACTIVE_REQUESTS.inc(len(prompts))

        start_time = time.time()

        try:
            outputs = self.llm.generate(prompts, sampling_params)

            # Track tokens generated
            for output in outputs:
                TOKENS_GENERATED.inc(len(output.outputs[0].token_ids))

            return outputs
        except Exception as e:
            REQUEST_ERRORS.inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            ACTIVE_REQUESTS.dec(len(prompts))

# Usage
llm = MonitoredLLM("meta-llama/Llama-2-7b-hf")
# Metrics available at http://localhost:9090/metrics
```

### Pattern 6: Caching Layer

```python
# caching.py
from vllm import LLM, SamplingParams
from functools import lru_cache
import hashlib
import json
from typing import Optional
import redis

class CachedLLM:
    def __init__(self, model_name: str, redis_url: Optional[str] = None):
        self.llm = LLM(model=model_name)
        self.redis_client = redis.from_url(redis_url) if redis_url else None

    def _cache_key(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Generate cache key from prompt and params"""
        params_str = json.dumps({
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
        }, sort_keys=True)

        key_str = f"{prompt}:{params_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def generate(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Generate with caching"""
        # Check cache
        if self.redis_client:
            cache_key = self._cache_key(prompt, sampling_params)
            cached = self.redis_client.get(cache_key)

            if cached:
                return cached.decode('utf-8')

        # Generate
        output = self.llm.generate(prompt, sampling_params)
        result = output[0].outputs[0].text

        # Store in cache
        if self.redis_client:
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                result
            )

        return result

# Usage
llm = CachedLLM("meta-llama/Llama-2-7b-hf", redis_url="redis://localhost:6379")
```

## Model Management

### Loading Models

#### From HuggingFace Hub

```python
llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

#### From Local Path

```python
llm = LLM(model="/path/to/local/model")
```

#### With Custom Tokenizer

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tokenizer="meta-llama/Llama-2-7b-hf",
    tokenizer_mode="auto"  # or "slow"
)
```

#### With Authentication

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or in code
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    download_dir="/custom/cache/dir"
)
```

### Supported Model Architectures

vLLM supports many popular architectures:

- **LLaMA & LLaMA 2**: Meta's LLaMA family
- **Mistral & Mixtral**: Mistral AI models
- **GPT-2, GPT-J, GPT-NeoX**: GPT variants
- **OPT**: Meta's OPT models
- **BLOOM**: BigScience BLOOM
- **Falcon**: TII Falcon models
- **MPT**: MosaicML MPT
- **Qwen**: Alibaba Qwen
- **Baichuan**: Baichuan models
- **Yi**: 01.AI Yi models
- **DeepSeek**: DeepSeek models
- **Phi**: Microsoft Phi models
- **Gemma**: Google Gemma

### Model Warmup

```python
# Warm up model with sample request
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Warm up
_ = llm.generate("Hello", SamplingParams(max_tokens=1))

# Now ready for production requests
```

## Monitoring & Debugging

### Logging Configuration

```python
import logging

# Configure vLLM logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# vLLM-specific loggers
logging.getLogger('vllm').setLevel(logging.DEBUG)
logging.getLogger('vllm.engine').setLevel(logging.INFO)
```

### Server Metrics Endpoint

vLLM server exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Key Metrics**:
- `vllm:num_requests_running`: Currently running requests
- `vllm:num_requests_waiting`: Queued requests
- `vllm:gpu_cache_usage_perc`: GPU cache utilization
- `vllm:cpu_cache_usage_perc`: CPU cache utilization
- `vllm:time_to_first_token_seconds`: TTFT latency
- `vllm:time_per_output_token_seconds`: Token generation speed

### Debug Mode

```bash
# Enable debug logging
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --log-level debug
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:8000/health

# Returns:
# {"status": "ok"}

# Model info
curl http://localhost:8000/v1/models
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: CUDA OOM, crash during model loading

**Solutions**:
```python
# Reduce GPU memory utilization
llm = LLM(model="...", gpu_memory_utilization=0.8)

# Reduce max sequence length
llm = LLM(model="...", max_model_len=2048)

# Enable CPU swap
llm = LLM(model="...", swap_space=8)

# Use quantization
llm = LLM(model="...", quantization="awq")

# Use tensor parallelism
llm = LLM(model="...", tensor_parallel_size=2)
```

#### 2. Slow Generation

**Symptoms**: Low throughput, high latency

**Solutions**:
```python
# Increase batch size
llm = LLM(model="...", max_num_seqs=256)

# Use CUDA graph
llm = LLM(model="...", enforce_eager=False)

# Optimize data type
llm = LLM(model="...", dtype="bfloat16")

# Check GPU utilization
nvidia-smi dmon -s u
```

#### 3. Model Loading Failures

**Symptoms**: Cannot load model, missing files

**Solutions**:
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface
huggingface-cli download meta-llama/Llama-2-7b-hf

# Verify model path
ls -la /path/to/model/

# Check authentication
export HF_TOKEN=your_token
```

#### 4. Networking Issues in Multi-GPU

**Symptoms**: NCCL errors, timeout in distributed setup

**Solutions**:
```bash
# Debug NCCL
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable P2P if issues

# Check GPU visibility
nvidia-smi topo -m

# Verify CUDA version
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Debugging

```python
# Enable profiling
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
) as prof:
    llm.generate(prompt, sampling_params)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices

### 1. Resource Allocation

- **Memory**: Start with `gpu_memory_utilization=0.9`, adjust based on OOM
- **Batch Size**: Larger `max_num_seqs` for throughput, smaller for latency
- **Parallelism**: Use tensor parallelism for large models (>70B params)

### 2. Model Selection

- **7B models**: Single GPU, low latency applications
- **13B-30B models**: 1-2 GPUs, balanced performance
- **70B+ models**: 4-8 GPUs, maximum quality

### 3. Optimization Strategy

1. **Start simple**: Single GPU, default settings
2. **Profile**: Measure throughput and latency
3. **Scale horizontally**: Add tensor parallelism if needed
4. **Optimize memory**: Tune `gpu_memory_utilization`, consider quantization
5. **Fine-tune batching**: Adjust `max_num_seqs` and `max_num_batched_tokens`

### 4. Production Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - VLLM_LOGGING_LEVEL=INFO
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    command: >
      --model meta-llama/Llama-2-7b-hf
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.95
      --max-num-seqs 256
      --host 0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 5. Security Considerations

```python
# Add authentication
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    # ... generate logic
```

```bash
# Rate limiting with nginx
# nginx.conf
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location / {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://vllm_backend:8000;
    }
}
```

### 6. Cost Optimization

- **Use quantization**: 4-bit AWQ reduces memory by ~4x
- **Right-size models**: Don't use 70B when 7B suffices
- **Batch aggressively**: Higher throughput = lower cost per request
- **Monitor utilization**: Scale down during low traffic

## Integration Examples

### With LangChain

```python
from langchain.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize vLLM
llm = VLLM(
    model="meta-llama/Llama-2-7b-hf",
    trust_remote_code=True,
    max_new_tokens=512,
    temperature=0.7
)

# Create chain
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

# Run
result = chain.run("What is vLLM?")
print(result)
```

### With Ray Serve

```python
from ray import serve
from vllm import LLM, SamplingParams
import ray

ray.init()
serve.start()

@serve.deployment(
    ray_actor_options={"num_gpus": 2},
    max_concurrent_queries=100
)
class VLLMDeployment:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=2
        )

    def __call__(self, request):
        prompt = request.query_params["prompt"]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        output = self.llm.generate(prompt, sampling_params)
        return output[0].outputs[0].text

VLLMDeployment.deploy()
```

### With Kubernetes

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-2-7b-hf
          - --tensor-parallel-size
          - "2"
          - --gpu-memory-utilization
          - "0.95"
        resources:
          limits:
            nvidia.com/gpu: 2
          requests:
            nvidia.com/gpu: 2
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

## References

- **Official Documentation**: https://docs.vllm.ai/
- **GitHub Repository**: https://github.com/vllm-project/vllm
- **Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **Blog**: https://blog.vllm.ai/
- **Discord Community**: https://discord.gg/vllm

## Quick Reference

### Common Commands

```bash
# Start server
python -m vllm.entrypoints.openai.api_server --model <model>

# Check version
python -c "import vllm; print(vllm.__version__)"

# Test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model>", "prompt": "Hello", "max_tokens": 50}'

# Monitor GPU
nvidia-smi dmon -s u -d 1

# Check metrics
curl http://localhost:8000/metrics
```

### Key Parameters Cheat Sheet

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `tensor_parallel_size` | Multi-GPU distribution | 1, 2, 4, 8 |
| `gpu_memory_utilization` | GPU memory fraction | 0.8-0.95 |
| `max_num_seqs` | Concurrent sequences | 32-512 |
| `max_model_len` | Max sequence length | 2048, 4096, 8192 |
| `dtype` | Precision | bfloat16, float16 |
| `quantization` | Quantization method | awq, gptq |
| `temperature` | Randomness | 0.0-2.0 |
| `top_p` | Nucleus sampling | 0.9-1.0 |
| `max_tokens` | Generation limit | 128-2048 |
