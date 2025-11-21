# Thinking and Reasoning in Machine Learning

Reasoning is the ability to perform multi-step logical inference, break down complex problems into intermediate steps, and arrive at conclusions through structured thinking. This capability distinguishes advanced AI systems from simple pattern matchers, enabling models to tackle problems requiring explicit reasoning chains, mathematical derivations, and step-by-step problem solving.

## Table of Contents

1. [Introduction to Reasoning in ML](#introduction-to-reasoning-in-ml)
2. [Chain-of-Thought Prompting](#chain-of-thought-prompting)
3. [Reasoning Architectures](#reasoning-architectures)
4. [Training Models to Reason](#training-models-to-reason)
5. [Advanced Reasoning Techniques](#advanced-reasoning-techniques)
6. [Practical Implementation](#practical-implementation)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Best Practices](#best-practices)

## Introduction to Reasoning in ML

### What is Reasoning?

Reasoning in machine learning refers to the model's ability to:
- Decompose complex problems into simpler sub-problems
- Generate intermediate reasoning steps toward a solution
- Apply logical rules and common-sense knowledge
- Self-verify and correct its thinking process
- Generalize problem-solving strategies to new domains

### Why Models Struggle with Reasoning

Traditional language models trained on next-token prediction excel at pattern matching but struggle with:
- **Implicit reasoning**: Standard training doesn't explicitly teach step-by-step thinking
- **Working memory**: Transformers process in parallel, making sequential reasoning challenging
- **Depth vs breadth**: Models learn surface patterns rather than deep logical structures
- **Compositionality**: Difficulty combining learned concepts in novel ways

### The Intuition: Making Thinking Explicit

The key insight is that **reasoning should be externalized**. Instead of expecting models to reason implicitly in their hidden states, we can:
1. **Prompt models** to show their work (Chain-of-Thought)
2. **Train models** to generate reasoning steps (supervised reasoning)
3. **Reward models** for correct reasoning processes (reinforcement learning)

This is analogous to how humans solve problems: we don't just jump to answers, we work through steps on paper or in our minds explicitly.

## Chain-of-Thought Prompting

Chain-of-Thought (CoT) prompting is a technique that encourages models to generate intermediate reasoning steps before producing a final answer.

### Zero-Shot Chain-of-Thought

The simplest form: just add "Let's think step by step" to your prompt.

```python
# Zero-shot CoT example
def zero_shot_cot_prompt(question):
    """
    Constructs a zero-shot CoT prompt by appending the magic phrase.

    Intuition: The phrase "Let's think step by step" triggers reasoning
    patterns the model learned during pre-training from educational content.
    """
    return f"{question}\n\nLet's think step by step:"

# Example usage
question = "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed?"
prompt = zero_shot_cot_prompt(question)

# Model output would be:
# "Let's think step by step:
# 1. First, I need to find the speed of the train
# 2. Speed = Distance / Time = 120 km / 2 hours = 60 km/h
# 3. Now I can find the distance for 5 hours
# 4. Distance = Speed × Time = 60 km/h × 5 hours = 300 km
# Therefore, the train will travel 300 km in 5 hours."
```

**Why it works**: Pre-training data contains many examples of step-by-step explanations (textbooks, tutorials, Q&A sites). The phrase activates these learned patterns.

### Few-Shot Chain-of-Thought

Provide examples with explicit reasoning steps to guide the model.

```python
def few_shot_cot_prompt(question, examples):
    """
    Constructs a few-shot CoT prompt with demonstration examples.

    Args:
        question: The question to answer
        examples: List of (question, reasoning, answer) tuples

    Intuition: Examples show the model the exact format and depth
    of reasoning expected, acting as in-context learning templates.
    """
    prompt_parts = []

    # Add examples
    for q, reasoning, answer in examples:
        prompt_parts.append(f"Q: {q}")
        prompt_parts.append(f"A: {reasoning}")
        prompt_parts.append(f"Answer: {answer}\n")

    # Add the actual question
    prompt_parts.append(f"Q: {question}")
    prompt_parts.append("A: ")

    return "\n".join(prompt_parts)

# Example usage
examples = [
    (
        "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?",
        "Roger started with 5 balls. 2 cans of 3 balls each is 2 × 3 = 6 balls. 5 + 6 = 11.",
        "11"
    ),
    (
        "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many do they have?",
        "They started with 23 apples. They used 20, so 23 - 20 = 3 apples left. They bought 6 more, so 3 + 6 = 9.",
        "9"
    )
]

question = "A parking lot had 12 cars. 7 more cars arrived and then 5 cars left. How many cars are there now?"
prompt = few_shot_cot_prompt(question, examples)
```

**Key insight**: The examples demonstrate the reasoning *structure* (break down, calculate intermediate steps, combine) that the model should follow.

### Self-Consistency

Generate multiple reasoning paths and take the majority vote for robustness.

```python
import torch
from collections import Counter

def self_consistency_decode(model, tokenizer, prompt, n_samples=5, temperature=0.7):
    """
    Implements self-consistency by sampling multiple reasoning paths.

    Intuition: Different reasoning paths may make different mistakes,
    but the correct answer should appear most frequently across samples.
    This is like solving a problem multiple ways to verify your answer.

    Args:
        model: Language model
        tokenizer: Corresponding tokenizer
        prompt: CoT prompt
        n_samples: Number of reasoning paths to generate
        temperature: Sampling temperature (> 0 for diversity)

    Returns:
        Most common final answer across all reasoning paths
    """
    answers = []

    # Generate multiple reasoning paths
    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt")

        # Sample with temperature for diversity
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract final answer (assume it's after "Answer:" or similar)
        answer = extract_final_answer(response)
        answers.append(answer)

    # Return majority vote
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]

    print(f"Answer distribution: {dict(answer_counts)}")
    print(f"Majority answer '{most_common_answer}' appeared {count}/{n_samples} times")

    return most_common_answer

def extract_final_answer(text):
    """
    Extracts the final answer from a reasoning chain.
    Simple heuristic: looks for "Answer:" or takes last number.
    """
    if "Answer:" in text:
        answer_part = text.split("Answer:")[-1].strip()
        # Extract first number or word
        import re
        match = re.search(r'\d+\.?\d*|\w+', answer_part)
        return match.group(0) if match else answer_part
    return text.strip().split()[-1]

# Example usage:
# prompt = few_shot_cot_prompt(question, examples)
# final_answer = self_consistency_decode(model, tokenizer, prompt, n_samples=10)
```

**Intuition**: This mirrors human problem-solving: if you solve a math problem three different ways and get the same answer, you're more confident it's correct.

### Least-to-Most Prompting

Break complex problems into simpler sub-problems sequentially.

```python
def least_to_most_prompt(problem, decomposition_examples, solution_examples):
    """
    Implements least-to-most prompting with two stages:
    1. Decomposition: Break the problem into sub-problems
    2. Sequential solving: Solve each sub-problem using previous solutions

    Intuition: Complex problems are hard to solve directly, but easy
    when broken into simpler pieces. Each piece builds on the previous one.
    """

    # Stage 1: Problem decomposition
    decomp_prompt = "Break this problem into simpler sub-problems:\n\n"
    for problem_ex, subproblems_ex in decomposition_examples:
        decomp_prompt += f"Problem: {problem_ex}\n"
        decomp_prompt += f"Sub-problems:\n{subproblems_ex}\n\n"

    decomp_prompt += f"Problem: {problem}\nSub-problems:\n"

    # Get sub-problems from model (simplified - would use actual model call)
    # subproblems = model.generate(decomp_prompt)

    # Stage 2: Sequential solving
    # Solve each sub-problem using solutions from previous sub-problems
    solution_prompt = ""
    previous_solutions = []

    # for subproblem in subproblems:
    #     # Include context from previous solutions
    #     context = "\n".join(previous_solutions)
    #     current_prompt = f"{context}\n\nSolve: {subproblem}"
    #     solution = model.generate(current_prompt)
    #     previous_solutions.append(f"{subproblem} -> {solution}")

    return decomp_prompt  # Simplified return

# Example
decomposition_examples = [
    (
        "What is the sum of all even numbers from 1 to 100?",
        "1. Identify all even numbers from 1 to 100\n2. Add them together"
    ),
    (
        "If Alice has twice as many apples as Bob, and Bob has 3 more apples than Charlie, and Charlie has 5 apples, how many apples does Alice have?",
        "1. Find how many apples Bob has\n2. Find how many apples Alice has"
    )
]
```

**Key insight**: This mimics how we solve complex problems - we don't tackle everything at once, we build up from simpler foundations.

## Reasoning Architectures

### How Transformers Enable Reasoning

Transformers can perform reasoning through:
1. **Self-attention**: Relates different parts of the reasoning chain
2. **Deep layers**: Each layer refines the reasoning representation
3. **Positional encoding**: Maintains the sequential structure of reasoning steps

```python
import torch
import torch.nn as nn

class ReasoningTransformer(nn.Module):
    """
    A transformer architecture optimized for multi-step reasoning.

    Key modifications for reasoning:
    1. Deeper network for more reasoning steps
    2. Intermediate supervision at each layer
    3. Explicit reasoning token embeddings
    """

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=12,
                 reasoning_tokens=["<think>", "<step>", "</think>"]):
        super().__init__()

        self.d_model = d_model

        # Token embeddings with special reasoning tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer layers - deeper for more reasoning capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Reasoning step classifier (auxiliary task)
        # Helps the model learn to identify reasoning stages
        self.step_classifier = nn.Linear(d_model, 10)  # e.g., 10 reasoning step types

    def forward(self, input_ids, return_hidden_states=False):
        """
        Forward pass with optional intermediate hidden states.

        Args:
            input_ids: (batch_size, seq_len) token ids
            return_hidden_states: Whether to return all layer outputs

        Returns:
            logits: (batch_size, seq_len, vocab_size) next token predictions
            hidden_states: Optional list of hidden states from each layer
        """
        # Embedding and positional encoding
        x = self.embedding(input_ids) * (self.d_model ** 0.5)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Transformer encoding
        # Each layer refines the reasoning representation
        hidden_states = []
        for layer in self.transformer.layers:
            x = layer(x)
            if return_hidden_states:
                hidden_states.append(x)

        # Output projection to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        if return_hidden_states:
            return logits, hidden_states
        return logits

    def get_reasoning_representations(self, input_ids, reasoning_token_positions):
        """
        Extracts hidden representations at reasoning token positions.
        Useful for analyzing what the model learns at each reasoning step.

        Intuition: The hidden states at <think> and <step> tokens
        should encode the reasoning progress up to that point.
        """
        _, hidden_states = self.forward(input_ids, return_hidden_states=True)

        # Extract representations at reasoning tokens
        # reasoning_token_positions: (batch_size, num_reasoning_tokens)
        reasoning_reps = []
        for layer_hidden in hidden_states:
            # Gather hidden states at specific positions
            batch_size, seq_len, d_model = layer_hidden.shape
            # Simplified - would use advanced indexing in practice
            reasoning_reps.append(layer_hidden)

        return reasoning_reps

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)
```

**Intuition**: Each transformer layer can be seen as a "reasoning step". Layer 1 might identify key information, layer 5 might combine facts, layer 10 might draw conclusions. Deeper networks = more reasoning capacity.

### Scratchpad and Working Memory

Allow models to use intermediate "scratchpad" space for calculations.

```python
class ScratchpadReasoning(nn.Module):
    """
    Implements explicit scratchpad mechanism for multi-step reasoning.

    Intuition: Just like humans use paper for calculations, models benefit
    from explicit intermediate representation space. The scratchpad stores
    partial results that can be referenced in later reasoning steps.
    """

    def __init__(self, d_model=512, scratchpad_size=128, num_slots=10):
        super().__init__()

        self.d_model = d_model
        self.num_slots = num_slots

        # Scratchpad: learnable memory slots for intermediate computations
        # Think of this as "scratch paper" with fixed number of lines
        self.scratchpad = nn.Parameter(torch.randn(num_slots, scratchpad_size))

        # Write controller: decides what to write to scratchpad
        self.write_query = nn.Linear(d_model, scratchpad_size)
        self.write_key = nn.Linear(d_model, scratchpad_size)
        self.write_value = nn.Linear(d_model, scratchpad_size)

        # Read controller: reads from scratchpad
        self.read_query = nn.Linear(d_model, scratchpad_size)

        # Integration: combines input with scratchpad content
        self.integrate = nn.Linear(d_model + scratchpad_size, d_model)

    def forward(self, hidden_state):
        """
        Process hidden state with scratchpad read/write operations.

        Args:
            hidden_state: (batch_size, seq_len, d_model) current reasoning state

        Returns:
            updated_state: (batch_size, seq_len, d_model) reasoning state enhanced with scratchpad
        """
        batch_size, seq_len, _ = hidden_state.shape

        # Write operation: store intermediate results
        # Query: what am I trying to store?
        # Key: where should I store it (which slot)?
        # Value: what should I store?
        write_q = self.write_query(hidden_state)  # (batch, seq_len, scratchpad_size)
        write_k = self.write_key(hidden_state)
        write_v = self.write_value(hidden_state)

        # Attention over scratchpad slots for writing
        # Determines which slots to update
        write_scores = torch.matmul(write_k, self.scratchpad.t())  # (batch, seq_len, num_slots)
        write_weights = torch.softmax(write_scores, dim=-1)

        # Update scratchpad (simplified - in practice would use gating)
        # This is where intermediate calculations get stored
        updated_scratchpad = self.scratchpad + torch.matmul(
            write_weights.transpose(1, 2).mean(0),  # Average over batch and sequence
            write_v.mean(1)  # Average pooling
        ).unsqueeze(1)

        # Read operation: retrieve relevant intermediate results
        read_q = self.read_query(hidden_state)  # (batch, seq_len, scratchpad_size)

        # Attention over scratchpad to read relevant information
        read_scores = torch.matmul(read_q, updated_scratchpad.t())  # (batch, seq_len, num_slots)
        read_weights = torch.softmax(read_scores, dim=-1)

        # Retrieve scratchpad content
        scratchpad_content = torch.matmul(read_weights, updated_scratchpad)  # (batch, seq_len, scratchpad_size)

        # Integrate scratchpad content with hidden state
        combined = torch.cat([hidden_state, scratchpad_content], dim=-1)
        updated_state = self.integrate(combined)

        return updated_state

# Usage example
def reasoning_with_scratchpad_example():
    """
    Example of how scratchpad helps with multi-step arithmetic.

    Problem: "What is (12 + 8) × (15 - 5)?"

    Without scratchpad: Model struggles to hold intermediate results
    With scratchpad:
        Step 1: Calculate 12 + 8 = 20, store in scratchpad
        Step 2: Calculate 15 - 5 = 10, store in scratchpad
        Step 3: Read both results, multiply: 20 × 10 = 200
    """
    d_model = 512
    scratchpad = ScratchpadReasoning(d_model=d_model)

    # Simulate reasoning steps
    # Each hidden state represents the model's understanding at one step
    step1_state = torch.randn(1, 10, d_model)  # Understanding "12 + 8"
    step1_output = scratchpad(step1_state)  # Stores result "20" in scratchpad

    step2_state = torch.randn(1, 10, d_model)  # Understanding "15 - 5"
    step2_output = scratchpad(step2_state)  # Stores result "10" in scratchpad

    step3_state = torch.randn(1, 10, d_model)  # Understanding "multiply the results"
    step3_output = scratchpad(step3_state)  # Reads both results and computes final answer

    print(f"Final reasoning state shape: {step3_output.shape}")
    return step3_output
```

**Key insight**: The scratchpad acts like working memory. Without it, the model must encode all intermediate results in the hidden state implicitly, which is difficult for complex multi-step problems.

## Training Models to Reason

### Process Supervision vs Outcome Supervision

Two paradigms for training reasoning models:

**Outcome Supervision**: Reward only the final answer
- Pro: Easy to collect (just need correct answers)
- Con: Doesn't teach *how* to reason, just pattern matching

**Process Supervision**: Reward each intermediate reasoning step
- Pro: Teaches correct reasoning process, better generalization
- Con: Requires expensive step-by-step annotations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProcessRewardModel(nn.Module):
    """
    Reward model that scores each reasoning step.

    Used in RL training to provide fine-grained feedback on reasoning quality.
    Similar to how a teacher corrects each step of a math problem, not just
    the final answer.
    """

    def __init__(self, d_model=512, num_layers=6):
        super().__init__()

        # Encoder for reasoning step
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=num_layers
        )

        # Step quality scorer
        # Predicts: Is this reasoning step correct? Is it helpful?
        self.step_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Single score: step quality
        )

    def forward(self, step_embeddings):
        """
        Score the quality of reasoning steps.

        Args:
            step_embeddings: (batch_size, num_steps, d_model)
                Each position is one reasoning step

        Returns:
            step_scores: (batch_size, num_steps) quality score for each step
        """
        # Encode reasoning chain with context
        encoded = self.encoder(step_embeddings)  # (batch, num_steps, d_model)

        # Score each step
        step_scores = self.step_scorer(encoded).squeeze(-1)  # (batch, num_steps)

        return step_scores

    def compute_process_reward(self, step_embeddings, step_labels):
        """
        Computes process-based reward for RL training.

        Args:
            step_embeddings: Embeddings of generated reasoning steps
            step_labels: Ground truth labels for each step (1 = correct, 0 = incorrect, -1 = neutral)

        Returns:
            reward: Reward signal for each step
        """
        step_scores = self.forward(step_embeddings)

        # Reward = score if step is correct, penalty if incorrect
        # Intuition: Reinforce each correct reasoning step individually
        reward = step_scores * step_labels

        return reward

def train_with_process_supervision(model, reward_model, dataloader, optimizer, device='cuda'):
    """
    Training loop with process supervision.

    Intuition: Instead of just "your final answer is wrong", we give feedback like:
    - "Step 1 is correct"
    - "Step 2 has an error - you should have added, not multiplied"
    - "Step 3 is correct given step 2, but built on wrong foundation"

    This teaches the model the correct reasoning *process*, not just answer patterns.
    """
    model.train()
    reward_model.eval()

    for batch in dataloader:
        # batch contains: question, reasoning_steps, step_labels, final_answer
        questions = batch['questions'].to(device)
        target_steps = batch['reasoning_steps'].to(device)  # Gold reasoning
        step_labels = batch['step_labels'].to(device)  # 1 if step correct, 0 otherwise

        # Generate reasoning steps
        generated_steps = model.generate_reasoning(questions)

        # Get embeddings for each step
        step_embeddings = model.encode_steps(generated_steps)

        # Compute process-level rewards
        with torch.no_grad():
            process_rewards = reward_model.compute_process_reward(
                step_embeddings,
                step_labels
            )

        # Language modeling loss (standard next-token prediction)
        lm_loss = F.cross_entropy(
            generated_steps.view(-1, model.vocab_size),
            target_steps.view(-1)
        )

        # Reinforcement learning loss weighted by process rewards
        # Steps with higher rewards are reinforced more
        rl_loss = -torch.mean(process_rewards * torch.log_softmax(generated_steps, dim=-1))

        # Combined loss: learn correct tokens AND correct reasoning process
        loss = lm_loss + 0.1 * rl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"LM Loss: {lm_loss.item():.4f}, RL Loss: {rl_loss.item():.4f}")
```

**Key insight**: Process supervision is like having a tutor who corrects your work step-by-step vs. one who just says "wrong answer, try again". The former teaches reasoning, the latter encourages guessing.

### Reinforcement Learning for Reasoning (o1-style Training)

Training reasoning models using RL with verifiable rewards.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ReasoningPolicyModel(nn.Module):
    """
    Policy model for RL-based reasoning training.

    This is conceptually similar to how OpenAI's o1 model is trained:
    1. Model generates reasoning chains (policy)
    2. Chains are verified (reward)
    3. Good reasoning patterns are reinforced
    """

    def __init__(self, base_model, value_head=True):
        super().__init__()

        self.base_model = base_model  # Pre-trained language model

        # Value head for advantage estimation (used in PPO)
        if value_head:
            self.value_head = nn.Linear(base_model.d_model, 1)

    def forward(self, input_ids, return_values=False):
        """
        Forward pass returning policy logits and optionally values.
        """
        logits = self.base_model(input_ids)  # (batch, seq_len, vocab_size)

        if return_values and hasattr(self, 'value_head'):
            # Get hidden states for value prediction
            hidden = self.base_model.get_hidden_states(input_ids)
            values = self.value_head(hidden)  # (batch, seq_len, 1)
            return logits, values

        return logits

    def sample_reasoning_chain(self, question, max_steps=20, temperature=1.0):
        """
        Sample a reasoning chain from the policy.

        This is the exploration phase where model tries different reasoning approaches.
        """
        current_tokens = question
        reasoning_chain = []
        log_probs = []

        for step in range(max_steps):
            # Get policy distribution over next token
            logits = self.forward(current_tokens)

            # Temperature scaling for exploration
            scaled_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Sample next token
            dist = Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)

            reasoning_chain.append(next_token)
            log_probs.append(log_prob)

            # Append to context
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(-1)], dim=-1)

            # Stop if we generate end-of-reasoning token
            if next_token == self.base_model.eos_token_id:
                break

        return reasoning_chain, log_probs

def verify_reasoning_chain(question, reasoning_chain, answer, verifier):
    """
    Verify if a reasoning chain is correct.

    For math: execute the chain and check if final answer is correct
    For logic: use formal verifier
    For code: run the code and check output

    Intuition: We can verify many tasks even if we can't easily generate solutions.
    This asymmetry (verification easier than generation) is key to RL training.
    """
    # Extract final answer from reasoning chain
    predicted_answer = extract_final_answer(reasoning_chain)

    # Binary reward: correct or not
    if predicted_answer == answer:
        return 1.0

    # Can also use partial credit for:
    # - Correct intermediate steps
    # - Nearly correct answers
    # - Valid reasoning even if wrong conclusion
    partial_score = verifier.score_reasoning(reasoning_chain)

    return partial_score

def ppo_reasoning_training_step(policy_model, ref_model, batch, optimizer,
                                 ppo_clip=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    One PPO training step for reasoning.

    PPO (Proximal Policy Optimization) is used because:
    1. Stable training - doesn't deviate too far from current policy
    2. Sample efficient - reuses experience for multiple updates
    3. Effective for language - used in RLHF and reasoning model training

    Training flow:
    1. Sample reasoning chains from current policy
    2. Verify chains and compute rewards
    3. Update policy to increase probability of high-reward chains
    4. Use KL penalty to prevent distribution shift
    """

    questions = batch['questions']
    answers = batch['answers']

    # Phase 1: Sampling
    # Generate multiple reasoning chains per question
    all_log_probs = []
    all_rewards = []
    all_values = []

    with torch.no_grad():
        for question, answer in zip(questions, answers):
            # Sample reasoning chain
            chain, log_probs = policy_model.sample_reasoning_chain(question)

            # Verify and get reward
            reward = verify_reasoning_chain(question, chain, answer, verifier=None)

            all_log_probs.append(torch.stack(log_probs))
            all_rewards.append(reward)

    # Phase 2: Advantage computation
    # Advantage = how much better is this action than expected?
    log_probs = torch.stack(all_log_probs)
    rewards = torch.tensor(all_rewards)

    # Compute advantages (simplified - would use GAE in practice)
    # High advantage = this reasoning chain was better than usual
    advantages = rewards - rewards.mean()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Phase 3: Policy update
    # Re-evaluate log probs under current policy
    current_log_probs = []
    current_values = []

    for question in questions:
        logits, values = policy_model(question, return_values=True)
        # ... compute log probs for the sampled actions
        # current_log_probs.append(...)
        # current_values.append(...)

    # Ratio for importance sampling
    # ratio = π_new / π_old
    ratio = torch.exp(torch.stack(current_log_probs) - log_probs)

    # PPO clipped objective
    # Prevents too large policy updates
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss - how well do we predict returns?
    value_loss = F.mse_loss(torch.stack(current_values).squeeze(), rewards)

    # Entropy bonus - encourages exploration
    # Without this, model might collapse to one reasoning pattern
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    # Combined loss
    loss = policy_loss + value_coef * value_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_reward': rewards.mean().item()
    }
```

**Intuition**: This is like learning to solve problems by trial and error with feedback:
1. **Sample**: Try different reasoning approaches
2. **Verify**: Check which ones lead to correct answers
3. **Reinforce**: Make successful reasoning patterns more likely
4. **Explore**: Keep trying new approaches (entropy bonus)

The key advantage over supervised learning: we only need correct answers (easy to get), not step-by-step solutions (expensive to annotate).

### Self-Taught Reasoner (STaR)

Iteratively generate reasoning, keep correct ones, fine-tune, and repeat.

```python
def star_training(model, unlabeled_questions, labeled_few_shot, iterations=5):
    """
    Self-Taught Reasoner (STaR) training algorithm.

    Key idea: Use the model to generate its own training data
    1. Generate reasoning chains for questions
    2. Keep only chains that lead to correct answers
    3. Fine-tune on these correct chains
    4. Repeat - model gets progressively better at reasoning

    Intuition: Like a student who:
    - Solves practice problems
    - Checks answers
    - Studies only the problems they got right (to learn correct patterns)
    - Gets better over time

    Args:
        model: Base language model
        unlabeled_questions: Questions with answers but no reasoning chains
        labeled_few_shot: Small set of examples with reasoning for prompting
        iterations: Number of STaR iterations
    """

    for iteration in range(iterations):
        print(f"\n=== STaR Iteration {iteration + 1}/{iterations} ===")

        # Step 1: Generate reasoning chains
        generated_data = []

        for question, answer in unlabeled_questions:
            # Use few-shot prompting to generate reasoning
            prompt = few_shot_cot_prompt(question, labeled_few_shot)

            # Generate reasoning chain
            reasoning_chain = model.generate(prompt, max_length=512)

            # Extract predicted answer
            predicted_answer = extract_final_answer(reasoning_chain)

            # Step 2: Filter - keep only correct reasoning chains
            if predicted_answer == answer:
                # This reasoning chain led to correct answer
                # Add it to training data
                generated_data.append({
                    'question': question,
                    'reasoning': reasoning_chain,
                    'answer': answer
                })

        print(f"Generated {len(generated_data)} correct reasoning chains")

        # Step 3: Rationalization - for questions we got wrong,
        # generate reasoning that leads to the correct answer
        # This is done by giving the model the answer and asking it to backtrack

        for question, answer in unlabeled_questions:
            # Check if we haven't solved this yet
            if not any(d['question'] == question for d in generated_data):
                # Rationalization: "Given the answer is X, explain the reasoning"
                rationalization_prompt = f"{question}\n\nThe answer is {answer}. Let's work out the reasoning:"

                reasoning = model.generate(rationalization_prompt, max_length=512)

                generated_data.append({
                    'question': question,
                    'reasoning': reasoning,
                    'answer': answer
                })

        print(f"After rationalization: {len(generated_data)} training examples")

        # Step 4: Fine-tune on generated correct reasoning chains
        fine_tune_dataloader = create_dataloader(generated_data, batch_size=8)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for epoch in range(2):  # Few epochs per iteration
            for batch in fine_tune_dataloader:
                # Standard language modeling loss
                loss = model.compute_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Step 5: Evaluate improvement
        accuracy = evaluate_reasoning(model, test_set)
        print(f"Iteration {iteration + 1} accuracy: {accuracy:.2%}")

        # Model should get progressively better at reasoning
        # Early iterations: few correct chains, limited improvement
        # Later iterations: more correct chains, better generalization

    return model

def create_dataloader(data, batch_size):
    """Creates a simple dataloader from generated reasoning data."""
    # Simplified implementation
    return data  # Would actually create proper PyTorch DataLoader

def evaluate_reasoning(model, test_set):
    """Evaluates model reasoning accuracy on test set."""
    correct = 0
    for question, answer in test_set:
        prediction = extract_final_answer(model.generate(question))
        if prediction == answer:
            correct += 1
    return correct / len(test_set)
```

**Key insight**: STaR bootstraps reasoning ability:
- Iteration 1: Model might only solve 20% correctly, learns from those
- Iteration 2: Now solves 35% correctly, more training data
- Iteration 3: 50% correct, even more training data
- ...
- Final: Strong reasoning ability learned from model's own correct solutions

## Advanced Reasoning Techniques

### Tree-of-Thoughts (ToT)

Explore multiple reasoning paths in a tree structure, evaluate and prune.

```python
import torch
from dataclasses import dataclass
from typing import List, Optional
from queue import PriorityQueue

@dataclass
class ThoughtNode:
    """
    A node in the tree of thoughts.

    Each node represents a partial reasoning state.
    """
    state: str  # Current reasoning state
    parent: Optional['ThoughtNode']
    children: List['ThoughtNode']
    value: float  # Estimated value/quality of this thought
    depth: int

    def __lt__(self, other):
        # For priority queue - higher value = better thought
        return self.value > other.value

class TreeOfThoughts:
    """
    Tree-of-Thoughts reasoning framework.

    Instead of following a single reasoning chain, explores multiple paths:
    1. Generate several possible next thoughts
    2. Evaluate each thought's promise
    3. Expand most promising thoughts
    4. Backtrack if path seems unproductive

    Intuition: Like chess - consider multiple moves ahead, evaluate positions,
    prune bad branches, explore promising ones deeply.
    """

    def __init__(self, model, value_model, max_depth=5, beam_width=3):
        self.model = model  # Generates possible thoughts
        self.value_model = value_model  # Evaluates thought quality
        self.max_depth = max_depth
        self.beam_width = beam_width

    def generate_thoughts(self, current_state, num_thoughts=5):
        """
        Generate possible next reasoning steps from current state.

        Example current_state: "To solve 8x + 5 = 29, I need to"
        Possible thoughts:
        - "isolate x by subtracting 5 from both sides"
        - "divide both sides by 8"  (wrong - would skip a step)
        - "first subtract 5 to get 8x = 24"
        """
        prompt = f"{current_state}\n\nPossible next steps:\n1."

        thoughts = []
        for _ in range(num_thoughts):
            # Generate diverse thoughts with sampling
            thought = self.model.generate(
                prompt,
                max_length=100,
                temperature=0.8,
                do_sample=True
            )
            thoughts.append(thought)

        return thoughts

    def evaluate_thought(self, thought, goal):
        """
        Evaluate how promising a thought is.

        Metrics:
        - Correctness: Is this step logically valid?
        - Progress: Does it move toward the goal?
        - Efficiency: Is it a productive step?
        """
        # Use value model to score
        value = self.value_model.score(thought, goal)
        return value

    def solve(self, problem, goal):
        """
        Solve a problem using tree-of-thoughts search.

        Uses best-first search with beam pruning.
        """
        # Initialize root node
        root = ThoughtNode(
            state=problem,
            parent=None,
            children=[],
            value=0.0,
            depth=0
        )

        # Priority queue for best-first search
        frontier = PriorityQueue()
        frontier.put(root)

        best_solution = None
        best_value = float('-inf')

        nodes_explored = 0

        while not frontier.empty() and nodes_explored < 100:
            # Get most promising node
            current_node = frontier.get()
            nodes_explored += 1

            # Check if we've reached max depth or found solution
            if current_node.depth >= self.max_depth:
                if current_node.value > best_value:
                    best_value = current_node.value
                    best_solution = current_node
                continue

            # Generate possible next thoughts
            next_thoughts = self.generate_thoughts(
                current_node.state,
                num_thoughts=self.beam_width * 2
            )

            # Evaluate each thought
            evaluated_thoughts = []
            for thought in next_thoughts:
                value = self.evaluate_thought(thought, goal)
                evaluated_thoughts.append((thought, value))

            # Keep only top beam_width thoughts (pruning)
            evaluated_thoughts.sort(key=lambda x: x[1], reverse=True)
            top_thoughts = evaluated_thoughts[:self.beam_width]

            # Create child nodes for top thoughts
            for thought, value in top_thoughts:
                child_state = current_node.state + "\n" + thought
                child_node = ThoughtNode(
                    state=child_state,
                    parent=current_node,
                    children=[],
                    value=value,
                    depth=current_node.depth + 1
                )
                current_node.children.append(child_node)
                frontier.put(child_node)

        # Reconstruct best reasoning path
        if best_solution:
            path = []
            node = best_solution
            while node is not None:
                path.append(node.state)
                node = node.parent
            path.reverse()
            return path

        return None

# Usage example
def tree_of_thoughts_example():
    """
    Example: Solving a complex math problem with ToT.

    Problem: "Find x if 3x^2 + 5x - 2 = 0"

    Tree exploration:
    Root: "Find x if 3x^2 + 5x - 2 = 0"
    ├─ Thought 1: "Use quadratic formula" (value: 0.9)
    │  ├─ "Identify a=3, b=5, c=-2" (value: 0.95)
    │  └─ "Calculate discriminant b^2-4ac" (value: 0.93)
    ├─ Thought 2: "Try factoring" (value: 0.7)
    │  ├─ "Look for factors of 3 and -2" (value: 0.6)
    │  └─ "Try (3x-1)(x+2)=0" (value: 0.5)  # Wrong, pruned
    └─ Thought 3: "Complete the square" (value: 0.6)

    Best path selected: Quadratic formula route (highest values)
    """
    # Initialize (simplified - would use actual models)
    tot = TreeOfThoughts(
        model=None,  # Would be actual language model
        value_model=None,  # Would be actual value model
        max_depth=5,
        beam_width=3
    )

    problem = "Find x if 3x^2 + 5x - 2 = 0"
    goal = "Determine the value(s) of x"

    # solution_path = tot.solve(problem, goal)

    print("Tree-of-Thoughts exploration enables:")
    print("1. Exploring multiple solution strategies")
    print("2. Recovering from wrong turns (backtracking)")
    print("3. Finding non-obvious solutions")
    print("4. Evaluating reasoning quality before committing")
```

**Intuition**: Linear CoT is like walking down one path. ToT is like looking at a map, considering multiple routes, and choosing the best one. You might explore several paths partially before committing.

### Self-Refinement and Critique

Model critiques its own reasoning and iteratively improves.

```python
class SelfRefiningReasoner:
    """
    Implements self-refinement: model critiques and improves its reasoning.

    Process:
    1. Generate initial reasoning chain
    2. Critique the reasoning (find flaws)
    3. Refine based on critique
    4. Repeat until confident or max iterations

    Intuition: Like proofreading your own work - you catch mistakes
    and fix them iteratively.
    """

    def __init__(self, model, max_refinements=3):
        self.model = model
        self.max_refinements = max_refinements

    def generate_initial_reasoning(self, question):
        """Generate first attempt at reasoning."""
        prompt = f"{question}\n\nLet's solve this step by step:"
        reasoning = self.model.generate(prompt, max_length=512)
        return reasoning

    def critique_reasoning(self, question, reasoning):
        """
        Generate critique of current reasoning.

        Looks for:
        - Logical errors
        - Calculation mistakes
        - Unjustified assumptions
        - Missing steps
        """
        critique_prompt = f"""Question: {question}

Reasoning:
{reasoning}

Please critique this reasoning. Identify any errors, gaps, or improvements:"""

        critique = self.model.generate(critique_prompt, max_length=300)
        return critique

    def refine_reasoning(self, question, reasoning, critique):
        """
        Improve reasoning based on critique.
        """
        refine_prompt = f"""Question: {question}

Previous reasoning:
{reasoning}

Critique:
{critique}

Please provide improved reasoning that addresses the critique:"""

        refined_reasoning = self.model.generate(refine_prompt, max_length=512)
        return refined_reasoning

    def solve_with_refinement(self, question):
        """
        Solve question with iterative self-refinement.
        """
        # Initial attempt
        reasoning = self.generate_initial_reasoning(question)

        history = [{
            'iteration': 0,
            'reasoning': reasoning,
            'critique': None
        }]

        # Refinement loop
        for iteration in range(1, self.max_refinements + 1):
            # Critique current reasoning
            critique = self.critique_reasoning(question, reasoning)

            # Check if critique indicates reasoning is good
            if self.is_critique_satisfied(critique):
                print(f"Reasoning satisfactory after {iteration} iterations")
                break

            # Refine based on critique
            reasoning = self.refine_reasoning(question, reasoning, critique)

            history.append({
                'iteration': iteration,
                'reasoning': reasoning,
                'critique': critique
            })

        return reasoning, history

    def is_critique_satisfied(self, critique):
        """
        Check if critique indicates reasoning is acceptable.

        Simple heuristic: look for positive indicators
        """
        positive_phrases = [
            "looks good",
            "appears correct",
            "no errors found",
            "reasoning is sound"
        ]
        return any(phrase in critique.lower() for phrase in positive_phrases)

# Example usage
def self_refinement_example():
    """
    Example of self-refinement improving reasoning.

    Question: "If a car travels 60 km in 45 minutes, what's its speed in km/h?"

    Iteration 0 (Initial):
    "Speed = 60 km / 45 min = 1.33 km/min"

    Critique 0:
    "The answer is in km/min but the question asks for km/h. Need to convert."

    Iteration 1 (Refined):
    "Speed = 60 km / 45 min
     First convert 45 min to hours: 45/60 = 0.75 hours
     Speed = 60 km / 0.75 hours = 80 km/h"

    Critique 1:
    "Reasoning looks good. Correctly converted units and calculated."

    Final answer: 80 km/h (correct)
    """
    refiner = SelfRefiningReasoner(model=None, max_refinements=3)

    question = "If a car travels 60 km in 45 minutes, what's its speed in km/h?"
    # final_reasoning, history = refiner.solve_with_refinement(question)

    print("Self-refinement enables:")
    print("1. Error detection and correction")
    print("2. Iterative improvement of reasoning")
    print("3. Catching edge cases and unit errors")
    print("4. More robust final answers")
```

**Key insight**: Humans don't get everything right on the first try - we check our work, find mistakes, and correct them. Models can do the same.

## Practical Implementation

### Complete End-to-End Reasoning System

Putting it all together: a production-ready reasoning system.

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class ProductionReasoningSystem:
    """
    Complete reasoning system combining multiple techniques.

    Features:
    - Chain-of-thought prompting
    - Self-consistency for robustness
    - Self-refinement for accuracy
    - Confidence estimation
    - Structured output parsing
    """

    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.eval()

    def solve(self, question, strategy='self_consistency', num_samples=5):
        """
        Solve a question using specified reasoning strategy.

        Args:
            question: The problem to solve
            strategy: 'simple', 'self_consistency', or 'refined'
            num_samples: Number of samples for self-consistency

        Returns:
            answer: Final answer
            reasoning: Reasoning chain(s)
            confidence: Confidence score
        """
        if strategy == 'simple':
            return self._simple_cot(question)
        elif strategy == 'self_consistency':
            return self._self_consistency_solve(question, num_samples)
        elif strategy == 'refined':
            return self._refined_solve(question)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _simple_cot(self, question):
        """Simple chain-of-thought reasoning."""
        prompt = self._build_cot_prompt(question)

        reasoning = self._generate(prompt, temperature=0.7)
        answer = self._extract_answer(reasoning)
        confidence = 0.5  # No confidence estimation for simple CoT

        return answer, reasoning, confidence

    def _self_consistency_solve(self, question, num_samples):
        """Self-consistency with multiple reasoning paths."""
        prompt = self._build_cot_prompt(question)

        reasoning_paths = []
        answers = []

        # Generate multiple reasoning paths
        for _ in range(num_samples):
            reasoning = self._generate(prompt, temperature=0.8)
            answer = self._extract_answer(reasoning)

            reasoning_paths.append(reasoning)
            answers.append(answer)

        # Aggregate answers
        final_answer, confidence = self._aggregate_answers(answers)

        # Return most common answer with all reasoning paths
        return final_answer, reasoning_paths, confidence

    def _refined_solve(self, question):
        """Self-refinement for high accuracy."""
        # Initial reasoning
        reasoning = self._simple_cot(question)[1]

        # Refine up to 2 times
        for _ in range(2):
            critique = self._critique(question, reasoning)

            if "correct" in critique.lower() or "good" in critique.lower():
                break

            reasoning = self._refine(question, reasoning, critique)

        answer = self._extract_answer(reasoning)
        confidence = 0.8  # Higher confidence after refinement

        return answer, reasoning, confidence

    def _build_cot_prompt(self, question, few_shot=True):
        """Build chain-of-thought prompt."""
        if few_shot:
            # Include examples
            examples = self._get_few_shot_examples()
            prompt = examples + f"\n\nQ: {question}\nA: Let's think step by step."
        else:
            prompt = f"{question}\n\nLet's think step by step:"

        return prompt

    def _generate(self, prompt, max_length=512, temperature=0.7):
        """Generate text from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after prompt)
        response = generated_text[len(prompt):].strip()

        return response

    def _extract_answer(self, reasoning):
        """Extract final answer from reasoning chain."""
        # Look for common answer patterns
        import re

        # Pattern 1: "Therefore, X" or "So, X"
        patterns = [
            r"Therefore,?\s+(.+?)(?:\.|$)",
            r"So,?\s+(.+?)(?:\.|$)",
            r"The answer is\s+(.+?)(?:\.|$)",
            r"Final answer:\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: last sentence
        sentences = reasoning.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else reasoning.strip()

    def _aggregate_answers(self, answers):
        """Aggregate multiple answers and compute confidence."""
        from collections import Counter

        # Normalize answers (lowercase, strip)
        normalized = [ans.lower().strip() for ans in answers]

        # Count occurrences
        counts = Counter(normalized)
        most_common_answer, count = counts.most_common(1)[0]

        # Confidence = frequency of most common answer
        confidence = count / len(answers)

        # Return original case version of most common answer
        for ans in answers:
            if ans.lower().strip() == most_common_answer:
                return ans, confidence

        return most_common_answer, confidence

    def _critique(self, question, reasoning):
        """Generate critique of reasoning."""
        prompt = f"""Question: {question}

Reasoning: {reasoning}

Critique this reasoning. Is it correct? Any errors or improvements?"""

        return self._generate(prompt, temperature=0.5)

    def _refine(self, question, reasoning, critique):
        """Refine reasoning based on critique."""
        prompt = f"""Question: {question}

Previous reasoning: {reasoning}

Critique: {critique}

Provide improved reasoning:"""

        return self._generate(prompt, temperature=0.5)

    def _get_few_shot_examples(self):
        """Get few-shot examples for prompting."""
        return """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls each is 2 × 3 = 6 balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many do they have?
A: They started with 23 apples. They used 20, so 23 - 20 = 3 apples left. They bought 6 more, so 3 + 6 = 9. The answer is 9."""

# Usage example
def complete_example():
    """
    Complete example of using the reasoning system.
    """
    # Initialize system
    system = ProductionReasoningSystem()

    # Example questions
    questions = [
        "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed?",
        "A store has 45 apples. They sell 17 in the morning and 12 in the afternoon. How many are left?",
        "What is the sum of all prime numbers less than 20?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        # Solve with different strategies
        for strategy in ['simple', 'self_consistency', 'refined']:
            print(f"\n--- Strategy: {strategy} ---")

            answer, reasoning, confidence = system.solve(
                question,
                strategy=strategy,
                num_samples=5 if strategy == 'self_consistency' else 1
            )

            print(f"Answer: {answer}")
            print(f"Confidence: {confidence:.2%}")

            if isinstance(reasoning, list):
                print(f"Reasoning paths: {len(reasoning)}")
                print(f"Example path: {reasoning[0][:200]}...")
            else:
                print(f"Reasoning: {reasoning[:200]}...")
```

## Evaluation and Metrics

### Measuring Reasoning Quality

```python
class ReasoningEvaluator:
    """
    Comprehensive evaluation of reasoning capabilities.
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_dataset(self, model, dataset):
        """
        Evaluate model on reasoning dataset.

        Returns multiple metrics:
        - Final answer accuracy
        - Step-level accuracy
        - Reasoning coherence
        - Faithfulness (reasoning matches answer)
        """
        results = {
            'final_accuracy': [],
            'step_accuracy': [],
            'coherence_scores': [],
            'faithfulness_scores': []
        }

        for example in dataset:
            question = example['question']
            gold_answer = example['answer']
            gold_steps = example.get('reasoning_steps', [])

            # Generate reasoning
            reasoning = model.generate_reasoning(question)
            predicted_answer = extract_final_answer(reasoning)
            predicted_steps = parse_reasoning_steps(reasoning)

            # Final answer accuracy
            final_correct = (predicted_answer == gold_answer)
            results['final_accuracy'].append(float(final_correct))

            # Step-level accuracy (if gold steps available)
            if gold_steps:
                step_acc = self.evaluate_steps(predicted_steps, gold_steps)
                results['step_accuracy'].append(step_acc)

            # Coherence: do steps logically follow?
            coherence = self.evaluate_coherence(predicted_steps)
            results['coherence_scores'].append(coherence)

            # Faithfulness: does reasoning support the answer?
            faithfulness = self.evaluate_faithfulness(reasoning, predicted_answer)
            results['faithfulness_scores'].append(faithfulness)

        # Aggregate results
        return {
            'final_accuracy': sum(results['final_accuracy']) / len(results['final_accuracy']),
            'step_accuracy': sum(results['step_accuracy']) / len(results['step_accuracy']) if results['step_accuracy'] else 0,
            'avg_coherence': sum(results['coherence_scores']) / len(results['coherence_scores']),
            'avg_faithfulness': sum(results['faithfulness_scores']) / len(results['faithfulness_scores'])
        }

    def evaluate_steps(self, predicted_steps, gold_steps):
        """
        Evaluate step-level accuracy.

        Compares each predicted reasoning step to gold standard.
        """
        if not gold_steps:
            return 0.0

        correct_steps = 0
        for pred_step, gold_step in zip(predicted_steps, gold_steps):
            if self.steps_match(pred_step, gold_step):
                correct_steps += 1

        return correct_steps / max(len(predicted_steps), len(gold_steps))

    def steps_match(self, step1, step2):
        """Check if two reasoning steps are equivalent."""
        # Simplified - would use semantic similarity in practice
        return step1.strip().lower() == step2.strip().lower()

    def evaluate_coherence(self, steps):
        """
        Evaluate logical coherence of reasoning chain.

        Checks:
        - Each step follows from previous
        - No contradictions
        - Logical flow
        """
        if not steps:
            return 0.0

        coherence_score = 0.0

        for i in range(1, len(steps)):
            # Check if step i follows from step i-1
            # In practice, would use entailment model
            if self.step_follows(steps[i-1], steps[i]):
                coherence_score += 1

        return coherence_score / max(len(steps) - 1, 1)

    def step_follows(self, prev_step, curr_step):
        """Check if current step logically follows from previous."""
        # Simplified heuristic
        # In practice, use NLI model or logical inference
        return True  # Placeholder

    def evaluate_faithfulness(self, reasoning, answer):
        """
        Evaluate if reasoning actually supports the final answer.

        Faithfulness measures if the answer follows from the reasoning.
        Low faithfulness = answer doesn't match reasoning (hallucination)
        """
        # Check if answer appears in reasoning
        if str(answer).lower() in reasoning.lower():
            # Check if it's derived, not just assumed
            answer_position = reasoning.lower().index(str(answer).lower())
            reasoning_before_answer = reasoning[:answer_position]

            # Heuristic: answer should appear late in reasoning
            relative_position = answer_position / len(reasoning)

            if relative_position > 0.5:  # Answer in second half
                return 1.0
            else:
                return 0.5  # Answer too early, might be assumed

        return 0.0  # Answer not in reasoning

def parse_reasoning_steps(reasoning_text):
    """Parse reasoning text into individual steps."""
    # Split by common step indicators
    import re

    # Look for numbered steps or sentence boundaries
    steps = re.split(r'\d+\.|;|\n', reasoning_text)
    steps = [s.strip() for s in steps if s.strip()]

    return steps
```

### Benchmark Datasets

Common reasoning benchmarks:
- **GSM8K**: Grade school math word problems (8,500 questions)
- **MATH**: Competition mathematics (12,500 problems)
- **StrategyQA**: Questions requiring implicit multi-hop reasoning
- **ARC**: Science questions requiring reasoning
- **BIG-Bench**: Diverse reasoning tasks

```python
def load_gsm8k_dataset():
    """
    Load GSM8K dataset for math reasoning evaluation.

    GSM8K tests multi-step arithmetic reasoning.
    Each problem requires 2-8 reasoning steps.
    """
    # Example GSM8K problem
    example = {
        'question': "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        'answer': "72",
        'reasoning': "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May."
    }

    return [example]  # Would load full dataset in practice

def evaluate_on_gsm8k(model):
    """
    Evaluate reasoning model on GSM8K.

    Reports:
    - Overall accuracy
    - Accuracy by number of reasoning steps required
    - Common error patterns
    """
    dataset = load_gsm8k_dataset()
    evaluator = ReasoningEvaluator()

    results = evaluator.evaluate_dataset(model, dataset)

    print("GSM8K Evaluation Results:")
    print(f"Final Answer Accuracy: {results['final_accuracy']:.2%}")
    print(f"Step-Level Accuracy: {results['step_accuracy']:.2%}")
    print(f"Reasoning Coherence: {results['avg_coherence']:.2%}")
    print(f"Answer Faithfulness: {results['avg_faithfulness']:.2%}")
```

## Best Practices

### When to Use Different Reasoning Techniques

1. **Simple CoT**: Fast, lightweight tasks; good baseline
2. **Self-consistency**: When accuracy is critical; have compute budget
3. **Self-refinement**: Complex problems; avoiding silly mistakes
4. **Tree-of-Thoughts**: Multi-step planning; exploring solution space
5. **Process supervision**: Training; have step-wise annotations
6. **RL training**: Training; have verifiable outcomes but not steps

### Practical Tips

```python
# Tip 1: Temperature tuning for reasoning
def optimal_temperature_for_task(task_type):
    """
    Different tasks benefit from different temperatures.

    Low temperature (0.1-0.3):
    - Math problems with one right answer
    - Logical deduction

    Medium temperature (0.5-0.7):
    - General reasoning
    - Balanced exploration/exploitation

    High temperature (0.8-1.0):
    - Creative problem solving
    - Brainstorming multiple approaches
    """
    temperatures = {
        'math': 0.2,
        'logic': 0.3,
        'general': 0.7,
        'creative': 0.9
    }
    return temperatures.get(task_type, 0.7)

# Tip 2: Prompt engineering for better reasoning
def effective_reasoning_prompts():
    """
    Collection of effective reasoning prompts.
    """
    prompts = {
        'step_by_step': "Let's solve this step by step:",
        'first_principles': "Let's think about this from first principles:",
        'pros_cons': "Let's consider the pros and cons:",
        'structured': "Let's break this down:\n1. What do we know?\n2. What do we need to find?\n3. How can we solve it?",
        'verification': "Let's solve this and then verify our answer:"
    }
    return prompts

# Tip 3: Combining techniques
def hybrid_reasoning_approach(model, question, importance='high'):
    """
    Combine multiple techniques based on problem importance.

    Low importance: Simple CoT
    Medium importance: Self-consistency (3 samples)
    High importance: Self-consistency + refinement
    Critical importance: Tree-of-thoughts with refinement
    """
    if importance == 'low':
        return model.simple_cot(question)
    elif importance == 'medium':
        return model.self_consistency(question, n=3)
    elif importance == 'high':
        answer, reasoning, _ = model.self_consistency(question, n=5)
        refined = model.refine(question, reasoning)
        return refined
    else:  # critical
        paths = model.tree_of_thoughts(question)
        best_path = max(paths, key=lambda p: p.value)
        refined = model.refine(question, best_path.reasoning)
        return refined

# Tip 4: Detecting when model is uncertain
def estimate_uncertainty(model, question, n_samples=5):
    """
    Estimate model's uncertainty about reasoning.

    High entropy in answers = high uncertainty
    High agreement = low uncertainty (confident)
    """
    answers = []
    for _ in range(n_samples):
        reasoning = model.generate_reasoning(question)
        answer = extract_final_answer(reasoning)
        answers.append(answer)

    # Compute answer entropy
    from collections import Counter
    counts = Counter(answers)
    total = len(answers)

    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * (p.log() if p > 0 else 0)

    # Normalize entropy
    max_entropy = -(1/n_samples) * n_samples * (1/n_samples).log()
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        'uncertainty': normalized_entropy,
        'agreement': counts.most_common(1)[0][1] / total,
        'answer_distribution': dict(counts)
    }
```

### Common Pitfalls and Solutions

```python
"""
Common Pitfalls in Reasoning Systems:

1. Reasoning shortcuts
   Problem: Model generates answer first, then rationalizes
   Solution: Use process supervision; verify steps independently

2. Hallucinated calculations
   Problem: Model "shows work" but arithmetic is wrong
   Solution: Use external calculator; verify each calculation

3. Inconsistent reasoning
   Problem: Steps contradict each other
   Solution: Coherence checking; self-consistency

4. Length bias
   Problem: Longer reasoning chains favored regardless of quality
   Solution: Normalize rewards by length; quality > quantity

5. Training-test mismatch
   Problem: Train on short problems, test on long ones
   Solution: Curriculum learning; gradually increase difficulty

6. Reasoning collapse
   Problem: Model learns to skip reasoning, jump to answer
   Solution: Require explicit steps; reward intermediate correctness
"""

def mitigate_reasoning_shortcuts(model, question):
    """
    Prevent model from generating answer first then rationalizing.

    Solution: Force step-by-step generation with constrained decoding.
    """
    # Require specific format: Step 1, Step 2, ..., Answer
    prompt = f"""{question}

Please solve this following the format:
Step 1: [first step]
Step 2: [second step]
...
Final Answer: [answer]

Begin:"""

    # Could add constrained decoding to enforce format
    reasoning = model.generate(prompt)
    return reasoning

def verify_calculations(reasoning):
    """
    Verify arithmetic in reasoning chain using external calculator.

    Extracts calculations and checks them programmatically.
    """
    import re

    # Find arithmetic expressions
    # e.g., "3 + 5 = 8", "12 / 4 = 3"
    calc_pattern = r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'

    calculations = re.findall(calc_pattern, reasoning)

    errors = []
    for num1, op, num2, claimed_result in calculations:
        num1, num2, claimed_result = float(num1), float(num2), float(claimed_result)

        # Compute correct result
        if op == '+':
            correct_result = num1 + num2
        elif op == '-':
            correct_result = num1 - num2
        elif op == '*':
            correct_result = num1 * num2
        elif op == '/':
            correct_result = num1 / num2 if num2 != 0 else float('inf')

        # Check if claimed result is correct
        if abs(correct_result - claimed_result) > 0.01:
            errors.append({
                'expression': f"{num1} {op} {num2}",
                'claimed': claimed_result,
                'correct': correct_result
            })

    if errors:
        print(f"Found {len(errors)} arithmetic errors:")
        for error in errors:
            print(f"  {error['expression']} = {error['claimed']} (should be {error['correct']})")
        return False

    return True
```

## Conclusion

Reasoning in machine learning transforms models from pattern matchers into problem solvers. Key takeaways:

1. **Explicit reasoning is crucial**: Make thinking visible through chain-of-thought
2. **Multiple approaches exist**: From simple prompting to complex RL training
3. **Process matters**: Training on reasoning steps beats training on answers alone
4. **Verification enables learning**: If you can verify correctness, you can train with RL
5. **Iteration improves quality**: Self-refinement and tree search find better solutions
6. **Combine techniques**: Hybrid approaches leverage strengths of different methods

The field continues to evolve rapidly with models like OpenAI's o1 demonstrating that explicit reasoning training produces qualitatively better problem-solving abilities. As models grow and techniques improve, we're moving toward AI systems that truly understand problems rather than just pattern-match solutions.

### Further Reading

- **Papers**:
  - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
  - "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)
  - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
  - "STaR: Self-Taught Reasoner" (Zelikman et al., 2022)
  - "Let's Verify Step by Step" (Lightman et al., 2023) - Process supervision

- **Resources**:
  - OpenAI o1 system card
  - GSM8K and MATH datasets
  - BIG-Bench reasoning tasks

- **Related Topics**:
  - See reinforcement_learning.md for RL fundamentals
  - See transformers.md for architecture details
  - See deep_learning.md for neural network basics
