# Artificial Intelligence (AI) Documentation

A comprehensive guide to modern AI technologies, tools, and best practices.

> **Scope:** Applied LLM/GenAI systems and tooling — RAG, vector DBs, prompting, agents,
> MCP/tools, serving (vLLM, local inference), LLMOps/eval/security, model families, and
> image/audio tools. For model *internals*, architectures, and training, see the
> [Machine Learning](../machine_learning/README.md) section.

## Overview

This directory contains documentation on various AI topics, focusing on practical applications, implementation guides, and best practices for working with modern AI systems.

## Contents

### Foundations

- [Large Language Models (LLMs)](./llms.md) — fundamentals, training, inference, API usage
- [Transformers Architecture](./transformers_architecture.md) — attention, embeddings, the core building block
- [Generative AI](./generative_ai.md) — text, image, audio, and multimodal generation overview
- [Model Families](./model_families.md) — GPT/Claude/Gemini vs Llama/Mistral/Qwen/Gemma/DeepSeek/Phi, selection
- [Reasoning Models & Test-Time Compute](./reasoning_models.md) — o-series/R1, long CoT, RL with verifiable rewards

### Prompting

- [Prompt Engineering](./prompt_engineering.md) — patterns, Chain-of-Thought, few-/zero-shot
- [Software Development Prompts](./software_dev_prompts.md) — proven prompt patterns for coding workflows
- [Structured Outputs](./structured_outputs.md) — JSON mode, constrained/grammar-guided decoding, Instructor/Outlines

### RAG & Retrieval

- [Retrieval Augmented Generation (RAG)](./rag.md) — grounding LLMs in external knowledge
- [Vector Databases](./vector_databases.md) — ANN search, HNSW/IVF, similarity metrics
- [Embeddings & Reranking](./embeddings.md) — bi- vs cross-encoders, MTEB, rerankers, hybrid search
- [Chunking & Ingestion](./chunking_strategies.md) — parsing, chunk sizing, contextual retrieval, metadata
- [GraphRAG](./graphrag.md) — knowledge-graph retrieval, community summaries, local vs global search

### Agents & Tools

- [Agent Frameworks](./agent_frameworks.md) — plan/act/observe loops, LangChain/LangGraph/CrewAI
- [Multi-Agent Systems](./multi_agent_systems.md) — supervisor/swarm/pipeline patterns, handoffs, shared state
- [Agent Memory & Context Engineering](./agent_memory.md) — working/episodic/semantic memory, compaction, retrieval
- [Agentic Context Engineering (ACE)](./ace.md) — evolving context playbooks for agents
- [Tool Use](./tool_use.md) — function calling, schemas, tool orchestration
- [Model Context Protocol (MCP)](./mcp.md) — open standard connecting assistants to external systems
- [Skills](./skills.md) — packaged, model-invocable capabilities (SKILL.md)
- [Coding & Computer-Use Agents](./coding_agents.md) — autonomous coding loops, SWE-bench, browser/computer use
- [Claude Code CLI](./cli.md) — plan mode, subagents, hooks, MCP, multi-model workflows

### Inference & Serving

- [Local LLM Inference](./local_inference.md) — Ollama, llama.cpp, GGUF, LM Studio, VRAM sizing
- [Inference Optimization](./inference_optimization.md) — KV cache, continuous batching, PagedAttention, speculative decoding, FlashAttention
- [vLLM](./vllm.md) — high-throughput serving engine
- [Prompt Caching](./prompt_caching.md) — KV-prefix reuse, cache breakpoints, cost/latency savings

### Models & Training

- [Llama](./llama.md) — Meta's open-weight family
- [Phi](./phi.md) — Microsoft's small high-quality models
- [DeepSeek R1](./deepseek_r1.md) — open-source reasoning model
- [Fine-Tuning](./fine_tuning.md) — adaptation, LoRA/QLoRA, dataset prep
- [Alignment](./alignment.md) — SFT, RLHF, DPO, Constitutional AI (applied post-training)

### Image & Audio

- [Stable Diffusion](./stable_diffusion.md) — text-to-image generation
- [Flux.1](./fluxdev.md) — Black Forest Labs' image model
- [ComfyUI](./comfyui.md) — node-based diffusion workflows
- [Whisper](./whisper.md) — speech-to-text transcription

### Evaluation & LLMOps

- [LLM Evaluation](./llm_evaluation.md) — benchmarks, evals, LLM-as-judge, RAGAS
- [LLM Observability & LLMOps](./llm_observability.md) — tracing, token/cost/latency, the prod feedback loop

### Security & Safety

- [LLM Security](./llm_security.md) — prompt injection, jailbreaks, OWASP LLM Top 10, defenses
- [Guardrails & Moderation](./guardrails.md) — input/output checks, content moderation, PII, validation

## Key AI Concepts

### Large Language Models (LLMs)

LLMs are neural networks trained on vast amounts of text data to understand and generate human-like text. Key characteristics:

- **Scale**: Billions to trillions of parameters
- **Training**: Self-supervised learning on diverse text corpora
- **Capabilities**: Text generation, reasoning, code writing, translation, etc.
- **Examples**: GPT-4, Claude, Llama, PaLM, Mistral

### Transformer Architecture

The foundation of modern LLMs:

```
Input → Tokenization → Embedding → 
  Positional Encoding → 
  Multi-Head Attention → 
  Feed Forward → 
  Layer Norm → 
  Output
```

Key components:
- **Self-Attention**: Allows model to weigh importance of different tokens
- **Positional Encoding**: Provides sequence order information
- **Feed-Forward Networks**: Process attention outputs
- **Residual Connections**: Enable training of deep networks

### Diffusion Models

State-of-the-art image generation approach:

1. **Forward Process**: Gradually add noise to images
2. **Reverse Process**: Learn to denoise, generating new images
3. **Conditioning**: Guide generation with text, images, or other inputs

## Popular AI Tools & Frameworks

### For LLMs

```bash
# OpenAI API
pip install openai

# Anthropic Claude
pip install anthropic

# Hugging Face Transformers
pip install transformers torch

# LangChain for LLM applications
pip install langchain langchain-community

# LlamaIndex for RAG
pip install llama-index
```

### For Image Generation

```bash
# Stable Diffusion WebUI (AUTOMATIC1111)
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
./webui.sh

# ComfyUI (node-based interface)
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt

# Diffusers library
pip install diffusers transformers accelerate
```

### For Model Training & Fine-tuning

```bash
# Hugging Face ecosystem
pip install transformers datasets accelerate peft bitsandbytes

# PyTorch
pip install torch torchvision torchaudio

# DeepSpeed for distributed training
pip install deepspeed

# Axolotl for fine-tuning
git clone https://github.com/OpenAccess-AI-Collective/axolotl
```

## Quick Start Examples

### Using OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

### Using Anthropic Claude

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
    ]
)

print(message.content[0].text)
```

### Using Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
prompt = "What is the theory of relativity?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate image
prompt = "a serene mountain landscape at sunset, oil painting style"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("output.png")
```

## Best Practices

### 1. Prompt Engineering
- Be specific and clear in your instructions
- Provide context and examples
- Use system prompts to set behavior
- Iterate and refine based on outputs

### 2. Model Selection
- Choose the right model for your task
- Balance capability vs. cost vs. speed
- Consider fine-tuning for specialized tasks
- Use quantization for resource constraints

### 3. Safety & Ethics
- Implement content filtering
- Monitor for bias and fairness
- Respect copyright and attribution
- Ensure data privacy and security

### 4. Performance Optimization
- Use batch processing when possible
- Implement caching for repeated queries
- Optimize prompts for token efficiency
- Use streaming for real-time responses

## Resources

### Official Documentation
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Documentation](https://docs.anthropic.com)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Stability AI](https://stability.ai/documentation)

### Learning Resources
- [DeepLearning.AI Courses](https://www.deeplearning.ai/)
- [Fast.ai](https://www.fast.ai/)
- [Hugging Face Course](https://huggingface.co/learn)
- [Papers with Code](https://paperswithcode.com/)

### Community
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Discord Communities](https://discord.gg/stablediffusion)

## Contributing

This documentation is continuously updated with new techniques, models, and best practices. Each section contains practical examples and code snippets that you can use immediately.

## License

This documentation is provided for educational purposes. Please refer to individual model and tool licenses for usage terms.
