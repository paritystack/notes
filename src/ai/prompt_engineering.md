# Prompt Engineering

A comprehensive guide to crafting effective prompts for Large Language Models (LLMs).

## Table of Contents
- [Introduction](#introduction)
- [Core Principles](#core-principles)
- [Fundamental Techniques](#fundamental-techniques)
- [Advanced Techniques](#advanced-techniques)
- [Prompt Patterns](#prompt-patterns)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Examples by Task](#examples-by-task)

## Introduction

Prompt engineering is the practice of designing inputs to get desired outputs from LLMs. It's both an art and a science, requiring understanding of:
- How models process and interpret text
- What patterns yield consistent results
- How to balance specificity with flexibility

## Core Principles

### 1. Clarity and Specificity

Be explicit about what you want:

```
❌ Bad: "Write about dogs"
✅ Good: "Write a 300-word informative article about the benefits of adopting rescue dogs, including health, cost, and emotional aspects."
```

### 2. Context Provision

Give the model necessary background:

```
❌ Bad: "What should I do?"
✅ Good: "I'm a Python developer with 3 years of experience. I want to transition into machine learning. What skills should I prioritize learning first?"
```

### 3. Format Specification

Define the desired output structure:

```
❌ Bad: "Tell me about the solar system"
✅ Good: "List the planets in our solar system in a markdown table with columns: Name, Distance from Sun (AU), and One Interesting Fact."
```

### 4. Role Assignment

Set the model's perspective:

```
"You are an experienced DevOps engineer. Explain Kubernetes deployments to a junior developer who has only worked with traditional hosting."
```

## Fundamental Techniques

### Zero-Shot Prompting

Direct instruction without examples:

```
Prompt: "Classify the sentiment of this review: 'The product arrived damaged but customer service was helpful.' Choose: positive, negative, or mixed."

Output: "mixed"
```

### Few-Shot Prompting

Provide examples to guide the model:

```
Classify movie reviews as positive or negative:

Review: "A masterpiece of cinema!"
Sentiment: positive

Review: "Boring and predictable plot."
Sentiment: negative

Review: "Waste of time and money."
Sentiment: negative

Review: "Incredible performances by the cast."
Sentiment: positive

Review: "The special effects were amazing but the story was weak."
Sentiment: 
```

### Chain-of-Thought (CoT)

Encourage step-by-step reasoning:

```
Prompt: "A cafeteria has 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step by step."

Output:
"Let's solve this step by step:
1. Starting apples: 23
2. Used for lunch: 23 - 20 = 3 apples remaining
3. Bought more: 3 + 6 = 9 apples
Answer: The cafeteria has 9 apples."
```

### Zero-Shot Chain-of-Thought

Add "Let's think step by step" to enable reasoning:

```
Prompt: "If a train travels 120 miles in 2 hours, then speeds up and travels 180 miles in the next 2 hours, what's the average speed for the entire journey? Let's think step by step."
```

### Self-Consistency

Generate multiple reasoning paths and choose the most consistent:

```python
# Ask the same question with slight variations
prompts = [
    "Calculate 15% tip on $47.50. Show your work.",
    "What's a 15% tip on a $47.50 bill? Explain your calculation.",
    "If my bill is $47.50 and I want to leave 15%, how much is the tip?"
]

# The most common answer is likely correct
```

## Advanced Techniques

### Tree of Thoughts (ToT)

Explore multiple reasoning branches:

```
Problem: Design a marketing campaign for a new eco-friendly water bottle.

Let's explore three different approaches:

Approach 1: Sustainability Focus
- Highlight environmental impact
- Partner with conservation organizations
- Target eco-conscious millennials
[Evaluate pros/cons]

Approach 2: Innovation Focus
- Emphasize unique design features
- Tech-forward marketing
- Target early adopters
[Evaluate pros/cons]

Approach 3: Health & Wellness Focus
- Connect to healthy lifestyle
- Partner with fitness influencers
- Target health-conscious consumers
[Evaluate pros/cons]

Now, let's combine the best elements...
```

### ReAct (Reasoning + Acting)

Interleave reasoning with actions:

```
Task: Find information about the latest Python version

Thought: I need to find current Python version information
Action: Search for "latest Python version 2025"
Observation: Python 3.13 was released in October 2024
Thought: I should verify this is the most recent stable version
Action: Check Python.org official releases
Observation: Confirmed, Python 3.13 is the latest stable version
Answer: The latest Python version is 3.13, released in October 2024
```

### Prompt Chaining

Break complex tasks into steps:

```python
# Step 1: Research
prompt1 = "List 5 key features of electric vehicles vs gasoline cars"

# Step 2: Analyze (using output from step 1)
prompt2 = f"Given these EV features: {output1}, which three are most important for urban commuters?"

# Step 3: Synthesize
prompt3 = f"Based on these priority features: {output2}, write a 100-word recommendation"
```

### Automatic Prompt Engineering (APE)

Let the model optimize its own prompts:

```
Meta-prompt: "I want to classify customer support tickets into categories: billing, technical, general inquiry. Generate 5 different prompts that would work well for this classification task."
```

## Prompt Patterns

### The Persona Pattern

```
"Act as [role] with [characteristics]. Your task is to [objective]."

Example:
"Act as a senior software architect with 15 years of experience in microservices. Review this code design and suggest improvements for scalability."
```

### The Template Pattern

```
"[Action] about [topic] in [format] with [constraints]."

Example:
"Write about artificial intelligence in a blog post format with a friendly tone, 500 words max, aimed at non-technical readers."
```

### The Constraint Pattern

```
"[Task]. You must [requirement 1]. You must [requirement 2]. You cannot [restriction]."

Example:
"Write a product description. You must include benefits, not just features. You must use active voice. You cannot use technical jargon."
```

### The Refinement Pattern

```
Initial prompt → Generate → Critique → Revise

Example:
"Write a haiku about coding."
[output]
"Now critique this haiku for syllable count and imagery."
[critique]
"Revise the haiku based on the critique."
```

### The Comparative Pattern

```
"Compare [A] and [B] in terms of [criteria 1], [criteria 2], and [criteria 3]. Present as [format]."

Example:
"Compare REST API and GraphQL in terms of performance, flexibility, and ease of use. Present as a comparison table."
```

### The Instruction-Context-Format (ICF) Pattern

```
# Instruction
[What to do]

# Context
[Background information]

# Format
[How to structure the output]

Example:
# Instruction
Explain how photosynthesis works

# Context
The audience is 5th-grade students learning about plant biology for the first time

# Format
Use an analogy with a familiar concept, then provide 3-5 simple bullet points
```

## Best Practices

### 1. Use Delimiters

Clearly separate different parts of your prompt:

```
Summarize the text delimited by triple quotes.

Text: """
[long text here]
"""

Requirements:
- 3 sentences maximum
- Highlight main argument
- Use neutral tone
```

### 2. Specify Output Format

```
"Provide your answer as a JSON object with the following structure:
{
  "summary": "brief overview",
  "key_points": ["point1", "point2", "point3"],
  "recommendation": "actionable advice"
}"
```

### 3. Request Step-by-Step Thinking

```
"Before answering, explain your reasoning process. Then provide the final answer clearly labeled."
```

### 4. Use Examples Strategically

```
# For few-shot learning, provide diverse examples:
Input: "The cat sat on the mat" → Simple sentence
Input: "Although tired, she completed the marathon" → Complex sentence
Input: "Run!" → Imperative sentence
Input: "Is it raining?" → Interrogative sentence
Input: "What a beautiful day!" → 
```

### 5. Iterate and Refine

```python
# Version 1: Too vague
"Write code for a web scraper"

# Version 2: More specific
"Write Python code for a web scraper using BeautifulSoup"

# Version 3: Complete specification
"Write Python code using BeautifulSoup to scrape product names and prices from an e-commerce site. Include error handling for missing elements and rate limiting to respect the server."
```

### 6. Control Length

```
"Explain quantum entanglement in [50/100/200] words"
"Provide a [brief/moderate/detailed] explanation"
"Summarize in [2-3 sentences/one paragraph/300 words]"
```

### 7. Set the Temperature

Understand model parameters:

```python
# Creative tasks (high temperature: 0.7-1.0)
{"temperature": 0.9}
# "Write a creative story about a time-traveling cat"

# Factual tasks (low temperature: 0.0-0.3)
{"temperature": 0.1}
# "What is the capital of France?"

# Balanced tasks (medium temperature: 0.4-0.6)
{"temperature": 0.5}
# "Explain the pros and cons of remote work"
```

## Common Pitfalls

### 1. Ambiguity

```
❌ "Tell me about Python"
✅ "Explain Python's list comprehension syntax with 3 examples"
```

### 2. Conflicting Instructions

```
❌ "Write a detailed brief summary"
✅ "Write a summary in 2-3 sentences covering the main points"
```

### 3. Assuming Knowledge

```
❌ "Debug this code" [without context]
✅ "This Python function should sort a list but returns an error. Debug it: [code]. The error message is: [error]"
```

### 4. Overcomplicating

```
❌ [500-word prompt with 20 constraints]
✅ [Clear, focused prompt with 3-5 key requirements]
```

### 5. Not Testing Variations

Always try multiple phrasings:
- "List the benefits"
- "What are the advantages"
- "Explain why this is useful"

## Examples by Task

### Code Generation

```
Task: Create a Python function

Prompt:
"Write a Python function named 'calculate_statistics' that:
- Takes a list of numbers as input
- Returns a dictionary with: mean, median, mode, and standard deviation
- Handles edge cases (empty list, single value)
- Includes docstring with examples
- Uses only standard library modules"
```

### Data Analysis

```
Task: Analyze sales data

Prompt:
"Given this sales data in CSV format:
[data]

Perform the following analysis:
1. Calculate total revenue by product category
2. Identify the top 3 performing products
3. Calculate month-over-month growth rate
4. Provide 3 actionable insights

Present findings in a structured format with clear headers."
```

### Content Writing

```
Task: Write a blog post

Prompt:
"Write a 600-word blog post about 'The Future of Remote Work'

Structure:
- Engaging headline
- Hook in first paragraph
- 3 main sections with subheadings
- Include statistics or examples
- Conclude with actionable takeaway

Tone: Professional yet conversational
Audience: Mid-level professionals and managers
SEO keywords: remote work, hybrid model, workplace flexibility"
```

### Summarization

```
Task: Summarize a technical document

Prompt:
"Summarize the following technical documentation:

[document]

Create two versions:
1. Executive Summary (100 words): High-level overview for non-technical stakeholders
2. Technical Summary (300 words): Key technical details for engineering team

Highlight any critical warnings or breaking changes."
```

### Translation with Context

```
Task: Contextual translation

Prompt:
"Translate the following English text to Spanish:

'The system is down'

Context: This is an IT status message displayed to users during an outage.
Requirements:
- Use appropriate technical terminology
- Maintain professional tone
- Ensure clarity for non-technical users"
```

### Code Review

```
Task: Review code quality

Prompt:
"Review this Python code for:

[code]

Evaluate:
1. Code quality and readability
2. Performance considerations
3. Potential bugs or edge cases
4. Security issues
5. Best practices adherence

Provide specific suggestions with code examples where applicable.
Rate each category from 1-5 and explain your ratings."
```

### Question Answering

```
Task: Answer with citations

Prompt:
"Answer the following question using only information from the provided text. Quote relevant passages to support your answer.

Text: [document]

Question: [question]

Format:
- Direct answer (1-2 sentences)
- Supporting evidence (2-3 quoted passages)
- Confidence level (high/medium/low)"
```

### Creative Writing

```
Task: Story generation

Prompt:
"Write a short story (500 words) with these elements:

Setting: Cyberpunk city in 2150
Protagonist: AI rights activist
Conflict: Choice between following the law or doing what's right
Theme: Question of consciousness and personhood
Tone: Noir detective style

Include:
- Vivid sensory details
- Internal monologue
- Unexpected twist ending"
```

## Advanced Prompt Engineering

### Meta-Prompting

```
"I need to create prompts for classifying customer emails. First, analyze what makes a good classification prompt, then generate 3 examples of effective prompts for this task."
```

### Prompt Optimization Loop

```python
initial_prompt = "Explain machine learning"

optimization_prompt = f"""
Original prompt: "{initial_prompt}"

This prompt is too vague. Improve it by:
1. Adding specific focus area
2. Defining target audience
3. Specifying depth of explanation
4. Setting output format

Provide an optimized version.
"""
```

### System Prompts (API Usage)

```python
# For chat-based models
system_prompt = """You are a Python expert specializing in data science.
Your responses should:
- Include working code examples
- Explain complex concepts simply
- Suggest best practices
- Warn about common pitfalls
- Use type hints and documentation"""

user_prompt = "How do I handle missing data in pandas?"

# API call structure
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
```

### Constitutional AI Prompting

Build in safety and ethical guidelines:

```
"[Task description]

Guidelines:
- Provide factual, unbiased information
- Acknowledge uncertainty when appropriate
- Avoid harmful or discriminatory content
- Cite sources when making factual claims
- Respect privacy and confidentiality"
```

## Prompt Engineering Tools

### LangChain Prompt Templates

```python
from langchain import PromptTemplate

template = """
You are a {role} with expertise in {domain}.

Task: {task}

Context: {context}

Provide your response in {format} format.
"""

prompt = PromptTemplate(
    input_variables=["role", "domain", "task", "context", "format"],
    template=template
)

final_prompt = prompt.format(
    role="data scientist",
    domain="machine learning",
    task="explain overfitting",
    context="teaching beginners",
    format="simple terms with examples"
)
```

### Prompt Versioning

```python
# Track prompt iterations
prompts = {
    "v1.0": "Summarize this text",
    "v1.1": "Summarize this text in 100 words",
    "v1.2": "Summarize this text in 100 words, focusing on key insights",
    "v2.0": "Provide a 100-word summary highlighting: 1) main argument, 2) supporting evidence, 3) conclusions"
}
```

## Measuring Prompt Quality

### Evaluation Criteria

1. **Consistency**: Same prompt → similar outputs
2. **Accuracy**: Outputs match expected results
3. **Efficiency**: Minimal tokens for desired result
4. **Robustness**: Works with variations in input
5. **Clarity**: Unambiguous instructions

### Testing Framework

```python
def test_prompt(prompt, test_cases, model):
    results = []
    
    for test_input, expected_output in test_cases:
        full_prompt = prompt.format(input=test_input)
        actual_output = model.generate(full_prompt)
        
        results.append({
            'input': test_input,
            'expected': expected_output,
            'actual': actual_output,
            'match': evaluate_match(expected_output, actual_output)
        })
    
    return results
```

## Resources

### Practice Platforms
- [PromptPerfect](https://promptperfect.jina.ai/)
- [Learn Prompting](https://learnprompting.org/)
- [OpenAI Playground](https://platform.openai.com/playground)
- [Anthropic Console](https://console.anthropic.com/)

### Reading
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- Research papers on prompting techniques

### Communities
- r/PromptEngineering
- Discord servers for AI tools
- Twitter/X AI communities

## Conclusion

Prompt engineering is an iterative process. Start simple, test thoroughly, and refine based on results. The key is understanding both your task requirements and how the model interprets instructions.

Remember: The best prompt is the one that consistently produces the results you need with minimal tokens and maximum clarity.
