# Large Language Models (LLMs)

## Overview

Large Language Models are transformer-based neural networks trained on massive text corpora to predict and generate human language. They've revolutionized AI with capabilities in translation, summarization, question-answering, and reasoning.

## Architecture Basics

### Transformers
Built on **self-attention** mechanism:
- **Query-Key-Value**: "What am I looking for?" ’ "Where's the relevant info?" ’ "Get the info"
- **Multi-head Attention**: Multiple attention patterns in parallel
- **Feed-forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training

### Scaling Laws
Performance improves predictably with:
- **Model size** (parameters): 7B ’ 70B ’ 700B
- **Dataset size**: More tokens = better performance
- **Compute**: More training = better convergence

## Popular Models

| Model | Size | Training Data | Strengths |
|-------|------|---------------|-----------|
| GPT-4 | ~1.7T params | ~13T tokens | Reasoning, coding, creative |
| Claude | ~100B params | High quality data | Instruction following, safety |
| Llama 2 | 7B-70B | 2T tokens | Open-source, efficient |
| Mistral | 7B-8x7B | 7T tokens | Fast, efficient |
| Palm 2 | ~340B | High quality | Reasoning, math |

## Training Process

### 1. Pre-training
```
Objective: Predict next token
Input: "The cat sat on the"
Target: "mat"

Loss = -log P(mat | previous tokens)
```

Train on unlabeled internet text (unsupervised)

### 2. Supervised Fine-tuning
```
Input: "What is 2+2?"
Target: "2+2=4"
```

Train on labeled examples (supervised)

### 3. RLHF (Reinforcement Learning from Human Feedback)
```
1. Generate multiple responses
2. Humans rank by quality
3. Train reward model
4. Use reward to optimize policy
```

## Key Concepts

### Tokenization
Convert text to numbers:
```
"Hello world" ’ [15339, 1159]
```

### Embeddings
Represent tokens as vectors in semantic space:
```
king - man + woman H queen
```

### Context Window
Maximum tokens model can consider:
- GPT-3: 2K tokens
- GPT-4: 32K - 128K tokens
- Claude: 100K+ tokens
- Llama 2: 4K tokens

### Temperature
Controls randomness of output:
- **0**: Deterministic (always same answer)
- **0.7**: Balanced (varied but coherent)
- **1+**: Creative (more random)

## Prompting Techniques

### 1. Zero-Shot
```
Question: What is 2+2?
Answer: 4
```

### 2. Few-Shot
```
Question: What is 3+3?
Answer: 6

Question: What is 2+2?
Answer:
```

### 3. Chain-of-Thought
```
Q: If there are 3 apples and you add 2 more, how many are there?

A: Let me think step by step:
1. Start with 3 apples
2. Add 2 more
3. Total: 3 + 2 = 5
```

### 4. Role-Based
```
You are a helpful Python expert.
Q: How do I reverse a list?
A: [explanations as Python expert]
```

## Limitations

### Hallucinations
Making up false information confidently:
```
Q: What's the capital of Atlantis?
A: The capital is Poseidiopolis. (Made up!)
```

### Knowledge Cutoff
No information beyond training data:
```
Q: Who won the 2025 World Cup?
A: I don't have info beyond April 2024.
```

### Context Length
Can't process extremely long documents

### Reasoning
Struggles with:
- Multi-step complex logic
- Mathematics (prone to errors)
- Counting tokens accurately

## Fine-tuning Approaches

### Full Fine-tuning
Update all parameters (expensive):
```
Memory: O(parameters)
Time: O(tokens)
```

### LoRA (Low-Rank Adaptation)
Add small trainable matrices (efficient):
```python
# Instead of: W' = W + ”W
# Use: W' = W + A×B (where A, B << W)
```

### QLoRA
Quantized LoRA (even more efficient):
- 4-bit quantization
- Reduces memory to ~6GB for 7B model

## Applications

| Use Case | Technique | Example |
|----------|-----------|---------|
| **Chat** | Conversation history | ChatGPT |
| **Code** | In-context learning | GitHub Copilot |
| **Search** | Semantic ranking | Perplexity AI |
| **Translation** | Multilingual models | Google Translate |
| **Summarization** | Extractive/abstractive | Claude summarization |

## Costs & Efficiency

### API Usage
```
Pricing: $ per 1M tokens
GPT-4: $30 input, $60 output
Claude: $8 input, $24 output
```

### Running Locally
```
Model Size | VRAM Needed | Speed
7B params  | 16GB        | Fast
13B params | 24GB        | Medium
70B params | 80GB (GPU)  | Slow
```

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Perplexity** | How well model predicts text |
| **BLEU** | Translation quality |
| **ROUGE** | Summarization quality |
| **Human Eval** | Actual user satisfaction |

## Best Practices

### 1. Prompt Engineering
```
L Bad: "Write code"
 Good: "Write Python function that takes list and returns sorted list in ascending order"
```

### 2. Breaking Complex Tasks
```
Instead of: "Analyze this company and give investment advice"
Try:
1. "Summarize this company's financials"
2. "What are the main risks?"
3. "What are growth opportunities?"
4. "Should we invest?"
```

### 3. Verification
Always verify facts from authoritative sources

## ELI10

Imagine teaching a child language by:
1. Reading millions of books
2. Learning to predict next word
3. Getting feedback on quality
4. Adjusting understanding

That's basically how LLMs learn! They become really good at continuing conversations in natural human language.

The trick: They learn statistics of language, not true understanding. So they might confidently say wrong things (hallucinations).

## Future Directions

- **Multimodal**: Understanding images + text + audio
- **Long Context**: Processing entire books
- **Reasoning**: Better at logic puzzles
- **Efficiency**: Running on phones/devices
- **Robotics**: Language guiding physical actions

## Further Resources

- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT Blog](https://openai.com/research/)
- [Hugging Face](https://huggingface.co/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
