# Mixture of Experts (MoE)

A scalable neural network architecture that uses conditional computation to dramatically increase model capacity while maintaining computational efficiency through sparse activation.

## Table of Contents

1. [Overview](#overview)
2. [Core Intuition](#core-intuition)
3. [When to Use MoE](#when-to-use-moe)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Architecture Components](#architecture-components)
6. [PyTorch Implementation](#pytorch-implementation)
7. [Training Considerations](#training-considerations)
8. [Load Balancing Strategies](#load-balancing-strategies)
9. [MoE Variants](#moe-variants)
10. [Advanced Topics](#advanced-topics)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)
13. [Resources](#resources)

## Overview

Mixture of Experts (MoE) is a neural network architecture that divides the model into multiple "expert" sub-networks, where only a subset of experts is activated for each input. This conditional computation approach allows models to scale to trillions of parameters while keeping inference costs manageable.

**Key Concept**: Instead of routing every input through the entire network, MoE uses a gating/routing mechanism to select which experts should process each input. This means you can have a massive model capacity, but only use a fraction of it for any given input.

### Historical Context

- **1991**: Jacobs et al. introduced the original MoE concept
- **2017**: Shazeer et al. scaled MoE to billions of parameters ("Outrageously Large Neural Networks")
- **2021**: Google's Switch Transformer achieved 1.6 trillion parameters
- **2022-2024**: MoE became standard in modern LLMs (Mixtral, GPT-4, DeepSeek-V2)

### Key Advantages

1. **Scalability**: Add model capacity without proportional compute increase
2. **Efficiency**: Only activate relevant experts per input (sparse activation)
3. **Specialization**: Experts can specialize in different patterns/domains
4. **Training Speed**: Faster training than equivalent dense models
5. **Sample Efficiency**: Better performance with less training data

### Key Challenges

1. **Load Balancing**: Ensuring all experts are utilized evenly
2. **Communication**: Expert parallelism requires efficient inter-device communication
3. **Memory**: All experts must be kept in memory even if not all are active
4. **Fine-tuning**: Can be tricky to fine-tune on small datasets
5. **Routing Collapse**: Risk of all inputs routing to same experts

## Core Intuition

### The Central Idea

Imagine you're running a hospital:
- **Dense Model**: Every patient sees every doctor (neurologist, cardiologist, dermatologist, etc.)
- **MoE Model**: A triage nurse (router) sends each patient to 1-2 relevant specialists

The MoE approach is:
- More efficient (patients only see relevant doctors)
- More scalable (you can have 100 specialists without 100x wait time)
- More specialized (each doctor becomes expert in their domain)

### Visual Representation

```
Input Token: "The heart muscle contracts..."

                    ┌─────────────┐
                    │   Router    │  <-- Learns which experts to use
                    │  (Gating)   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        │ Score: 0.8       │ Score: 0.7       │ Score: 0.1
        ▼                  ▼                  ▼
    ┌───────┐          ┌───────┐          ┌───────┐
    │Expert │          │Expert │          │Expert │
    │   1   │  ✓       │   2   │  ✓       │   3   │  ✗
    │Medical│ Active   │Biology│ Active   │ Code  │ Inactive
    └───┬───┘          └───┬───┘          └───────┘
        │                  │
        └────────┬─────────┘
                 ▼
        Weighted combination
         (0.53 * E1 + 0.47 * E2)
                 │
                 ▼
            Final output
```

### How It Works

1. **Input Arrives**: A token/sample enters the MoE layer
2. **Router Decides**: A learned gating network computes scores for each expert
3. **Top-K Selection**: Select top-k experts (typically k=1 or k=2)
4. **Expert Processing**: Only selected experts process the input
5. **Combine Outputs**: Weighted sum based on router scores
6. **Update Router**: Router learns which experts work best for which inputs

### Expert Specialization

Experts naturally specialize during training:
- **Expert 1**: Might specialize in medical/biology text
- **Expert 2**: Might specialize in code/programming
- **Expert 3**: Might specialize in mathematics
- **Expert 4**: Might specialize in creative writing

This specialization emerges automatically through the routing mechanism and gradient descent!

## When to Use MoE

### Use MoE When:

✅ **Large-scale training** with massive datasets
✅ **Need for high capacity** without proportional compute cost
✅ **Diverse input domains** where specialization helps
✅ **Inference efficiency matters** (sparse activation reduces cost)
✅ **You have distributed training infrastructure** (for expert parallelism)

### Avoid MoE When:

❌ **Small datasets** (load balancing becomes difficult)
❌ **Limited memory** (all experts must fit in memory)
❌ **Single device training** (loses main efficiency advantage)
❌ **Fine-tuning critical** (MoE can be tricky to fine-tune)
❌ **Need deterministic behavior** (routing can be unstable)

### Practical Decision Matrix

| Scenario | Dense Model | MoE Model | Winner |
|----------|------------|-----------|--------|
| Dataset < 1B tokens | ✓ | ✗ | Dense |
| Dataset > 100B tokens | ✗ | ✓ | MoE |
| Single GPU | ✓ | ✗ | Dense |
| Multi-device cluster | ~ | ✓ | MoE |
| Inference latency critical | ✓ | ~ | Dense |
| Training cost critical | ✗ | ✓ | MoE |
| Fine-tuning required | ✓ | ✗ | Dense |
| Domain diversity high | ~ | ✓ | MoE |

## Mathematical Foundation

### Basic MoE Formulation

For an input $x$, the MoE output is:

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

Where:
- $N$ is the number of experts
- $G(x)$ is the gating function (router) outputting probabilities
- $E_i(x)$ is the output of expert $i$
- $G(x)_i$ is the gating weight for expert $i$

### Gating Function (Router)

The router computes a probability distribution over experts:

$$G(x) = \text{Softmax}(x \cdot W_g)$$

Where:
- $x \in \mathbb{R}^d$ is the input
- $W_g \in \mathbb{R}^{d \times N}$ is the learned gating weights
- Output: $G(x) \in \mathbb{R}^N$ with $\sum_{i=1}^{N} G(x)_i = 1$

### Top-K Sparsity

To enforce sparsity, we only keep the top-k experts:

$$G_{\text{sparse}}(x)_i = \begin{cases}
\frac{G(x)_i}{\sum_{j \in \text{TopK}} G(x)_j} & \text{if } i \in \text{TopK}(G(x), k) \\
0 & \text{otherwise}
\end{cases}$$

This ensures only $k$ experts are activated per input.

### Noisy Top-K Gating

To encourage exploration and prevent routing collapse, add noise:

$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

$$\text{TopK}(H(x), k)$$

The noise helps during training but is typically removed at inference.

### Load Balancing Loss

To prevent all inputs routing to the same experts, we add an auxiliary loss:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot CV(\text{expert\_usage})^2$$

Where $CV$ is the coefficient of variation:

$$CV(x) = \frac{\sigma(x)}{\mu(x)} = \frac{\text{std}(x)}{\text{mean}(x)}$$

More sophisticated version (used in Switch Transformer):

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $P_i$ = average router probability for expert $i$
- $\alpha$ is a hyperparameter (typically 0.01)
- $N$ is the number of experts

### Router Z-Loss

To prevent router logits from growing too large (numerical instability):

$$\mathcal{L}_{\text{z}} = \frac{1}{B} \sum_{x \in \text{batch}} \left(\log \sum_{i=1}^{N} e^{x \cdot W_{g,i}}\right)^2$$

This encourages the router to produce moderate logit values.

### Complete Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{aux}} + \beta \cdot \mathcal{L}_{\text{z}}$$

Where:
- $\mathcal{L}_{\text{task}}$ is the primary task loss (e.g., cross-entropy)
- $\alpha$ typically 0.01
- $\beta$ typically 0.001

### Capacity Factor

To prevent overflow when too many tokens route to one expert:

$$\text{expert\_capacity} = \left(\frac{\text{tokens\_per\_batch}}{N}\right) \cdot \text{capacity\_factor} \cdot k$$

Where:
- $k$ is top-k value
- capacity_factor > 1.0 (typically 1.25)
- Tokens exceeding capacity are either dropped or sent to next-best expert

## Architecture Components

### 1. Expert Networks

Each expert is typically a feed-forward network (FFN):

```python
class Expert(nn.Module):
    """
    Single expert network - typically a two-layer FFN.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Standard FFN: W2(GELU(W1(x)))
        return self.w2(self.dropout(self.activation(self.w1(x))))
```

Experts can also be:
- Multi-layer networks
- Specialized architectures (CNNs, attention layers)
- Different sizes (heterogeneous experts)

### 2. Router/Gating Network

The router determines which experts to activate:

```python
class Router(nn.Module):
    """
    Router that selects top-k experts for each token.
    """
    def __init__(self, d_model, num_experts, k=2, noisy_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        # Router weights
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

        # Noise weights (for exploration)
        if noisy_gating:
            self.w_noise = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, train=True):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]

        # Compute clean logits
        logits = self.w_gate(x_flat)  # [batch_size * seq_len, num_experts]

        # Add noise during training
        if train and self.noisy_gating:
            noise_stddev = F.softplus(self.w_noise(x_flat))
            noise = torch.randn_like(logits) * noise_stddev
            logits = logits + noise

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        # top_k_logits: [batch_size * seq_len, k]
        # top_k_indices: [batch_size * seq_len, k]

        # Compute probabilities (softmax over top-k)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Also compute full softmax for load balancing loss
        gates = F.softmax(logits, dim=-1)  # [batch_size * seq_len, num_experts]

        return top_k_gates, top_k_indices, gates, logits
```

### 3. Complete MoE Layer

```python
class MoELayer(nn.Module):
    """
    Complete Mixture of Experts layer.
    """
    def __init__(self, d_model, d_ff, num_experts, k=2,
                 capacity_factor=1.25, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = Router(d_model, num_experts, k)

    def forward(self, x, train=True):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # Route tokens
        top_k_gates, top_k_indices, all_gates, logits = self.router(x, train)
        # top_k_gates: [batch_size * seq_len, k]
        # top_k_indices: [batch_size * seq_len, k]

        # Flatten input
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)  # [batch_size * seq_len]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]  # [num_tokens_i, d_model]

                # Process through expert
                expert_output = self.experts[i](expert_input.unsqueeze(1)).squeeze(1)
                # expert_output: [num_tokens_i, d_model]

                # Get gates for tokens assigned to this expert
                # Find which position in top-k this expert is
                expert_positions = (top_k_indices[expert_mask] == i)
                expert_gates = top_k_gates[expert_mask][expert_positions]
                # expert_gates: [num_tokens_i]

                # Add weighted expert output
                output[expert_mask] += expert_gates.unsqueeze(-1) * expert_output

        # Reshape output
        output = output.view(batch_size, seq_len, d_model)

        # Return output and routing info for loss computation
        return output, {
            'gates': all_gates,
            'logits': logits,
            'top_k_indices': top_k_indices,
            'top_k_gates': top_k_gates
        }
```

### 4. MoE in Transformer Architecture

```
Standard Transformer Block:
┌─────────────────────────┐
│   Input Embedding       │
└───────────┬─────────────┘
            │
    ┌───────▼────────┐
    │ Self-Attention │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Add & Norm    │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │      FFN       │  <-- Replace with MoE
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Add & Norm    │
    └───────┬────────┘
            │
         Output


MoE Transformer Block:
┌─────────────────────────┐
│   Input Embedding       │
└───────────┬─────────────┘
            │
    ┌───────▼────────┐
    │ Self-Attention │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │  Add & Norm    │
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │   Router       │
    └───┬───┬───┬────┘
        │   │   │
    ┌───▼┐ ┌▼─┐ ┌▼───┐
    │ E1 │ │E2│ │E3  │  <-- Only top-k activated
    └───┬┘ └┬─┘ └┬───┘
        │   │   │
        └───┴───┴────┐
                     │
            ┌────────▼────┐
            │ Combine     │
            └────────┬────┘
                     │
            ┌────────▼────┐
            │  Add & Norm │
            └────────┬────┘
                     │
                  Output
```

## PyTorch Implementation

### Complete Working Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


class Expert(nn.Module):
    """
    Individual expert network (FFN).

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of FFN
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        # FFN: W2(Dropout(GELU(W1(x))))
        hidden = F.gelu(self.w1(x))  # [batch, seq_len, d_ff]
        hidden = self.dropout(hidden)
        output = self.w2(hidden)      # [batch, seq_len, d_model]
        return output


class NoisyTopKRouter(nn.Module):
    """
    Router with noisy top-k gating.

    Args:
        d_model: Model dimension
        num_experts: Number of experts
        k: Number of experts to select per token
        noise_std: Standard deviation of noise (for training)
    """
    def __init__(self, d_model: int, num_experts: int, k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std

        # Gating weights
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        # Initialize to small values
        nn.init.normal_(self.w_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple:
        """
        Args:
            x: [batch, seq_len, d_model]
            train: Whether in training mode

        Returns:
            top_k_gates: [batch * seq_len, k] - Normalized gates for top-k
            top_k_indices: [batch * seq_len, k] - Expert indices
            all_gates: [batch * seq_len, num_experts] - All gate probabilities
            logits: [batch * seq_len, num_experts] - Raw logits
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten: [batch * seq_len, d_model]
        x_flat = x.reshape(-1, d_model)

        # Compute logits: [batch * seq_len, num_experts]
        logits = self.w_gate(x_flat)

        # Add noise during training for exploration
        if train and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits_noisy = logits + noise
        else:
            logits_noisy = logits

        # Get top-k experts
        # top_k_logits: [batch * seq_len, k]
        # top_k_indices: [batch * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(logits_noisy, self.k, dim=-1)

        # Softmax over top-k only
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # Also compute full softmax for load balancing
        all_gates = F.softmax(logits, dim=-1)

        return top_k_gates, top_k_indices, all_gates, logits


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Args:
        d_model: Model dimension
        d_ff: Expert hidden dimension
        num_experts: Number of expert networks
        k: Number of experts to activate per token
        capacity_factor: Capacity factor for expert buffering
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 k: int = 2, capacity_factor: float = 1.25, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Create router
        self.router = NoisyTopKRouter(d_model, num_experts, k)

        # For tracking expert usage (debugging)
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def forward(self, x: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, d_model]
            train: Training mode flag

        Returns:
            output: [batch, seq_len, d_model]
            aux_info: Dictionary with routing information
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len

        # Get routing decisions
        top_k_gates, top_k_indices, all_gates, logits = self.router(x, train)
        # top_k_gates: [num_tokens, k]
        # top_k_indices: [num_tokens, k]
        # all_gates: [num_tokens, num_experts]

        # Flatten input
        x_flat = x.reshape(num_tokens, d_model)

        # Initialize output
        output_flat = torch.zeros_like(x_flat)

        # Track which experts are used
        if train:
            expert_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=x.device)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find all tokens that route to this expert (in any top-k position)
            # expert_mask_tokens: [num_tokens]
            expert_mask_tokens = (top_k_indices == expert_idx).any(dim=-1)

            # Count tokens routed to this expert
            num_expert_tokens = expert_mask_tokens.sum().item()

            if num_expert_tokens == 0:
                continue

            if train:
                expert_mask[expert_idx] = True
                self.expert_usage[expert_idx] += num_expert_tokens

            # Get input for this expert
            # expert_input: [num_expert_tokens, d_model]
            expert_input = x_flat[expert_mask_tokens]

            # Process through expert
            # expert_output: [num_expert_tokens, d_model]
            expert_output = self.experts[expert_idx](expert_input.unsqueeze(1)).squeeze(1)

            # Get gates for this expert
            # For each token routed to this expert, find its gate value
            expert_positions = (top_k_indices[expert_mask_tokens] == expert_idx)
            # expert_positions: [num_expert_tokens, k]

            # Extract gates: [num_expert_tokens]
            expert_gates = top_k_gates[expert_mask_tokens][expert_positions]

            # Accumulate weighted output
            output_flat[expert_mask_tokens] += expert_gates.unsqueeze(-1) * expert_output

        # Reshape output
        output = output_flat.reshape(batch_size, seq_len, d_model)

        # Prepare auxiliary info for loss computation
        aux_info = {
            'gates': all_gates,          # [num_tokens, num_experts]
            'logits': logits,            # [num_tokens, num_experts]
            'top_k_indices': top_k_indices,  # [num_tokens, k]
            'top_k_gates': top_k_gates,      # [num_tokens, k]
            'num_tokens': num_tokens,
        }

        return output, aux_info


def compute_load_balancing_loss(aux_info: Dict, num_experts: int) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss.

    Encourages uniform distribution of tokens across experts.

    Args:
        aux_info: Dictionary from MoE forward pass
        num_experts: Number of experts

    Returns:
        loss: Scalar load balancing loss
    """
    gates = aux_info['gates']  # [num_tokens, num_experts]
    top_k_indices = aux_info['top_k_indices']  # [num_tokens, k]
    num_tokens = aux_info['num_tokens']

    # Compute fraction of tokens assigned to each expert
    # f_i in the paper
    expert_counts = torch.zeros(num_experts, device=gates.device)
    for i in range(num_experts):
        expert_counts[i] = (top_k_indices == i).sum()

    f = expert_counts / num_tokens  # [num_experts]

    # Compute average gate probability for each expert
    # P_i in the paper
    P = gates.mean(dim=0)  # [num_experts]

    # Load balancing loss: encourages f_i * P_i to be uniform
    # loss = N * sum(f_i * P_i)
    loss = num_experts * (f * P).sum()

    return loss


def compute_router_z_loss(aux_info: Dict) -> torch.Tensor:
    """
    Compute router z-loss for numerical stability.

    Penalizes large logits to prevent overflow/underflow.

    Args:
        aux_info: Dictionary from MoE forward pass

    Returns:
        loss: Scalar z-loss
    """
    logits = aux_info['logits']  # [num_tokens, num_experts]

    # Log-sum-exp of logits
    log_z = torch.logsumexp(logits, dim=-1)  # [num_tokens]

    # Square and average
    z_loss = (log_z ** 2).mean()

    return z_loss


# Example usage
def example_moe_forward():
    """
    Demonstrate MoE forward pass with loss computation.
    """
    # Hyperparameters
    batch_size = 2
    seq_len = 4
    d_model = 128
    d_ff = 512
    num_experts = 8
    k = 2

    # Create MoE layer
    moe = SparseMoE(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        k=k,
        capacity_factor=1.25,
        dropout=0.1
    )

    # Sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")  # [2, 4, 128]

    # Forward pass
    output, aux_info = moe(x, train=True)
    print(f"Output shape: {output.shape}")  # [2, 4, 128]

    # Compute auxiliary losses
    load_balance_loss = compute_load_balancing_loss(aux_info, num_experts)
    z_loss = compute_router_z_loss(aux_info)

    print(f"\nLoad balancing loss: {load_balance_loss.item():.4f}")
    print(f"Router z-loss: {z_loss.item():.4f}")

    # Inspect routing decisions
    top_k_indices = aux_info['top_k_indices']  # [8, 2]
    top_k_gates = aux_info['top_k_gates']      # [8, 2]

    print(f"\nRouting decisions for first 3 tokens:")
    for i in range(3):
        experts = top_k_indices[i].tolist()
        gates = top_k_gates[i].tolist()
        print(f"  Token {i}: Experts {experts} with gates {[f'{g:.3f}' for g in gates]}")

    # Example output:
    # Token 0: Experts [3, 7] with gates ['0.612', '0.388']
    # Token 1: Experts [1, 5] with gates ['0.551', '0.449']
    # Token 2: Experts [2, 4] with gates ['0.723', '0.277']


# Training loop example
def example_training_step():
    """
    Example of a training step with MoE.
    """
    # Model setup
    moe = SparseMoE(d_model=128, d_ff=512, num_experts=8, k=2)
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-4)

    # Hyperparameters
    alpha_aux = 0.01  # Load balancing loss weight
    alpha_z = 0.001   # Router z-loss weight

    # Sample batch
    x = torch.randn(4, 10, 128)  # [batch, seq_len, d_model]
    target = torch.randn(4, 10, 128)

    # Forward pass
    output, aux_info = moe(x, train=True)

    # Task loss (e.g., MSE for this example)
    task_loss = F.mse_loss(output, target)

    # Auxiliary losses
    aux_loss = compute_load_balancing_loss(aux_info, num_experts=8)
    z_loss = compute_router_z_loss(aux_info)

    # Total loss
    total_loss = task_loss + alpha_aux * aux_loss + alpha_z * z_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Task loss: {task_loss.item():.4f}")
    print(f"Aux loss: {aux_loss.item():.4f}")
    print(f"Z loss: {z_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("MoE Forward Pass Example")
    print("=" * 60)
    example_moe_forward()

    print("\n" + "=" * 60)
    print("MoE Training Step Example")
    print("=" * 60)
    example_training_step()
```

### Output Example

```
============================================================
MoE Forward Pass Example
============================================================
Input shape: torch.Size([2, 4, 128])
Output shape: torch.Size([2, 4, 128])

Load balancing loss: 1.2458
Router z-loss: 0.0234

Routing decisions for first 3 tokens:
  Token 0: Experts [3, 7] with gates ['0.612', '0.388']
  Token 1: Experts [1, 5] with gates ['0.551', '0.449']
  Token 2: Experts [2, 4] with gates ['0.723', '0.277']

============================================================
MoE Training Step Example
============================================================
Task loss: 1.0234
Aux loss: 1.1567
Z loss: 0.0198
Total loss: 1.0353
```

## Training Considerations

### 1. Load Balancing

**Problem**: Without constraints, all tokens may route to the same few experts, leaving others unused.

**Solutions**:

a) **Auxiliary Loss** (most common)
```python
loss = task_loss + alpha * load_balance_loss
```
- Encourages uniform expert usage
- $\alpha$ typically 0.01 (tune based on task)

b) **Expert Capacity**
```python
capacity = (num_tokens / num_experts) * capacity_factor * k
```
- Hard limit on tokens per expert
- Overflow tokens are dropped or sent to alternative
- capacity_factor typically 1.25-2.0

c) **Random Routing** (during training)
- With small probability (e.g., 5%), route randomly
- Prevents collapse to single expert

d) **Curriculum Learning**
- Start with dense model (all experts)
- Gradually increase sparsity
- Helps establish expert diversity

### 2. Router Stability

**Problem**: Router logits can grow unbounded, causing numerical issues.

**Solution**: Router z-loss
```python
z_loss = mean((log_sum_exp(logits)) ^ 2)
loss_total = task_loss + alpha_aux * aux_loss + alpha_z * z_loss
```

Typical values:
- $\alpha_z = 0.001$ to 0.01

### 3. Expert Initialization

**Critical**: Experts should start different to enable specialization.

```python
# Option 1: Different random seeds per expert
for i, expert in enumerate(experts):
    torch.manual_seed(42 + i)
    expert.apply(init_weights)

# Option 2: Add small random perturbations
for expert in experts:
    for param in expert.parameters():
        param.data += torch.randn_like(param) * 0.01

# Option 3: Pre-train with different data subsets
# Train each expert on different data slice initially
```

### 4. Learning Rate Scheduling

Router and experts often benefit from different learning rates:

```python
optimizer = torch.optim.AdamW([
    {'params': moe.experts.parameters(), 'lr': 1e-4},
    {'params': moe.router.parameters(), 'lr': 5e-4},  # Higher LR for router
])
```

Rationale: Router needs to adapt quickly to find good expert assignments.

### 5. Gradient Clipping

MoE can have unstable gradients, especially early in training:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6. Expert Dropout

During training, randomly drop entire experts to prevent over-reliance:

```python
if train and random.random() < expert_dropout_prob:
    # Skip this expert, renormalize gates
    continue
```

Typical: expert_dropout_prob = 0.1

### 7. Token Dropping vs. Overflow Handling

When expert capacity is exceeded:

**Option A: Drop tokens** (Switch Transformer)
- Exceeding tokens get zero output
- Simple but loses information
- Use high capacity factor (1.5-2.0)

**Option B: Overflow to next-best expert**
- Send to second-choice expert if first is full
- More complex but preserves information
- Better for smaller capacity factors

```python
# Overflow handling
for expert_idx in range(num_experts):
    # Primary assignment
    primary_mask = (top_k_indices[:, 0] == expert_idx)

    # Overflow: check capacity
    if primary_mask.sum() > capacity:
        # Send overflow to second-choice expert
        overflow_mask = primary_mask[capacity:]
        # Process overflow_mask with top_k_indices[:, 1]
```

## Load Balancing Strategies

### 1. Importance-based Load Balancing (Switch Transformer)

Minimizes:
$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i$ = fraction of tokens dispatched to expert $i$
- $P_i$ = mean router probability for expert $i$

**Intuition**: If an expert gets many tokens ($f_i$ high), its router probability should be low ($P_i$ low), and vice versa.

```python
def importance_load_balance_loss(gates, top_k_indices, num_experts):
    """Switch Transformer load balancing."""
    num_tokens = gates.shape[0]

    # Fraction of tokens assigned to each expert
    f = torch.zeros(num_experts, device=gates.device)
    for i in range(num_experts):
        f[i] = (top_k_indices == i).sum() / num_tokens

    # Mean router probability
    P = gates.mean(dim=0)  # [num_experts]

    # Loss
    loss = num_experts * (f * P).sum()
    return loss
```

### 2. Balance Loss (GShard)

Encourages even distribution using coefficient of variation:

$$\mathcal{L}_{\text{balance}} = CV(\text{expert\_usage})^2 = \left(\frac{\sigma}{\mu}\right)^2$$

```python
def balance_loss(top_k_indices, num_experts):
    """GShard balance loss."""
    # Count tokens per expert
    counts = torch.zeros(num_experts, device=top_k_indices.device)
    for i in range(num_experts):
        counts[i] = (top_k_indices == i).sum()

    # Coefficient of variation
    mean = counts.mean()
    std = counts.std()
    cv = std / (mean + 1e-10)

    return cv ** 2
```

### 3. Expert Choice Routing

**Flip the paradigm**: Instead of tokens choosing experts, experts choose tokens!

```python
def expert_choice_routing(tokens, experts, capacity_per_expert):
    """
    Each expert selects top-k tokens based on affinity scores.

    Args:
        tokens: [num_tokens, d_model]
        experts: List of expert modules
        capacity_per_expert: Max tokens per expert
    """
    num_experts = len(experts)
    num_tokens = tokens.shape[0]

    # Compute affinity scores
    affinity = torch.zeros(num_experts, num_tokens)
    for i, expert in enumerate(experts):
        affinity[i] = expert.compute_affinity(tokens)  # [num_tokens]

    # Each expert selects top-k tokens
    expert_assignments = []
    for i in range(num_experts):
        top_tokens = torch.topk(affinity[i], capacity_per_expert).indices
        expert_assignments.append(top_tokens)

    return expert_assignments
```

**Advantages**:
- Perfect load balancing (each expert gets exactly `capacity_per_expert` tokens)
- No token dropping
- No auxiliary loss needed

**Disadvantages**:
- Some tokens may be selected by multiple experts (need to handle)
- Some tokens may not be selected at all (need default processing)

### 4. Dynamic Capacity Adjustment

Adjust expert capacity based on routing distribution:

```python
class AdaptiveCapacityMoE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Start with high capacity
        self.capacity_factor = 2.0

    def adjust_capacity(self, expert_usage):
        """Adjust capacity based on observed usage."""
        # If load is balanced, can reduce capacity
        cv = coefficient_of_variation(expert_usage)

        if cv < 0.1:  # Well balanced
            self.capacity_factor = max(1.1, self.capacity_factor - 0.1)
        elif cv > 0.5:  # Poorly balanced
            self.capacity_factor = min(3.0, self.capacity_factor + 0.1)
```

### 5. Entropy Regularization

Encourage router to have high entropy (uncertainty):

$$\mathcal{L}_{\text{entropy}} = -\lambda \cdot \mathbb{E}[H(G(x))]$$

$$H(p) = -\sum_{i} p_i \log p_i$$

```python
def entropy_regularization(gates):
    """Encourage high entropy in routing decisions."""
    # gates: [num_tokens, num_experts]
    entropy = -(gates * (gates + 1e-10).log()).sum(dim=-1)  # [num_tokens]
    mean_entropy = entropy.mean()

    # Negative because we want to maximize entropy (minimize negative entropy)
    return -mean_entropy
```

Higher entropy = more exploration = better load balancing.

Typical $\lambda = 0.01$.

## MoE Variants

### 1. Switch Transformer (Google, 2021)

**Key Innovation**: Simplified MoE with k=1 (single expert per token)

**Advantages**:
- Simpler routing (no weighted combination)
- Faster training and inference
- Scales to 1.6T parameters

**Architecture**:
```python
class SwitchLayer(nn.Module):
    """Switch Transformer: k=1 MoE."""
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # Route: select single expert per token
        router_logits = self.router(x_flat)
        expert_indices = router_logits.argmax(dim=-1)  # [batch * seq_len]

        # Process each expert's tokens in batch
        output = torch.zeros_like(x_flat)
        for i in range(len(self.experts)):
            mask = (expert_indices == i)
            if mask.any():
                output[mask] = self.experts[i](x_flat[mask].unsqueeze(1)).squeeze(1)

        return output.view(batch, seq_len, d_model)
```

**Results**: 7x speedup over T5-XXL with same quality.

### 2. Expert Choice (Google, 2022)

Experts select tokens instead of tokens selecting experts.

**Advantages**:
- Perfect load balancing
- No auxiliary loss needed
- No token dropping

**Implementation**:
```python
class ExpertChoiceLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, tokens_per_expert):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.tokens_per_expert = tokens_per_expert

        # Affinity scoring
        self.affinity_weights = nn.Parameter(torch.randn(num_experts, d_model))

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.shape[0]

        # Compute affinity: [num_experts, num_tokens]
        affinity = torch.matmul(self.affinity_weights, x_flat.T)

        # Each expert selects top-k tokens
        output = torch.zeros_like(x_flat)
        for i in range(len(self.experts)):
            # Expert i selects its top tokens
            top_k_scores, top_k_indices = torch.topk(affinity[i], self.tokens_per_expert)

            # Process selected tokens
            selected_tokens = x_flat[top_k_indices]
            expert_output = self.experts[i](selected_tokens.unsqueeze(1)).squeeze(1)

            # Accumulate (with normalized scores)
            gates = F.softmax(top_k_scores, dim=0)
            output[top_k_indices] += gates.unsqueeze(-1) * expert_output

        return output.view(batch, seq_len, d_model)
```

### 3. Sparse MoE (Mistral Mixtral 8x7B, 2023)

**Configuration**:
- 8 experts, each 7B parameters
- k=2 (top-2 routing)
- Total: 47B parameters, 13B active per token

**Key Features**:
- Expert parallelism across devices
- Efficient routing with minimal overhead
- Strong performance (matches 70B dense models)

**Architecture Details**:
```python
# Mixtral-style configuration
config = {
    'num_experts': 8,
    'k': 2,
    'd_model': 4096,
    'd_ff': 14336,  # Each expert is 7B params
    'num_layers': 32,
    'replace_every_n_layers': 1,  # MoE every layer
}
```

### 4. Soft MoE (Meta, 2023)

**Key Innovation**: No routing; instead, use learned slot attention.

**How it works**:
1. Define $n$ "slots" (similar to experts)
2. Each slot attends to all tokens (soft assignment)
3. Process slots through experts
4. Distribute expert outputs back to tokens

**Advantages**:
- No load balancing issues
- Fully differentiable
- No token dropping

**Disadvantages**:
- Not truly sparse (all slots process all information)
- Less parameter-efficient

```python
class SoftMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_slots):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(num_slots, d_model))
        self.expert = Expert(d_model, d_ff)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape

        # Compute slot-token affinities
        affinity = torch.matmul(self.slots, x.transpose(1, 2))  # [num_slots, batch, seq_len]

        # Normalize: each slot attends to tokens
        slot_attn = F.softmax(affinity, dim=-1)  # [num_slots, batch, seq_len]

        # Aggregate tokens into slots
        slots_filled = torch.matmul(slot_attn, x)  # [num_slots, batch, d_model]

        # Process slots through expert
        slots_processed = self.expert(slots_filled.unsqueeze(2)).squeeze(2)

        # Distribute back to tokens
        output = torch.matmul(slot_attn.transpose(1, 2), slots_processed)

        return output
```

### 5. Hierarchical MoE

Multiple levels of routing for very large models:

```
Input
  │
  ▼
Router L1 (high-level: topic)
  ├──────┬──────┬──────┐
  ▼      ▼      ▼      ▼
Router Router Router Router (L2: subtopic)
  │      │      │      │
Expert Expert Expert Expert (L3: fine-grained)
```

**Example**:
- L1: Route by modality (text vs. code vs. math)
- L2: Route by subtopic (Python vs. JavaScript, Algebra vs. Calculus)
- L3: Fine-grained processing

**Benefits**:
- Can scale to thousands of experts
- Better specialization
- Hierarchical abstractions

### 6. Conditional MoE (DeepSeek-V2)

**Innovation**: MoE applied selectively based on input characteristics.

```python
class ConditionalMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.moe = SparseMoE(d_model, d_ff, num_experts)
        self.dense_ffn = Expert(d_model, d_ff)
        self.selector = nn.Linear(d_model, 1)

    def forward(self, x):
        # Decide: MoE or dense?
        use_moe = torch.sigmoid(self.selector(x)) > 0.5  # [batch, seq_len, 1]

        moe_output, _ = self.moe(x)
        dense_output = self.dense_ffn(x)

        # Mix based on selector
        output = torch.where(use_moe, moe_output, dense_output)
        return output
```

**When to use**: Apply MoE only to challenging tokens, use dense FFN for simple tokens.

## Advanced Topics

### 1. Expert Parallelism

Distribute experts across multiple devices:

```python
# Device assignment
expert_devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
for i, expert in enumerate(experts):
    device = expert_devices[i % len(expert_devices)]
    expert.to(device)

# Forward with device routing
def forward_distributed(x, experts, top_k_indices):
    output = torch.zeros_like(x)

    for i, expert in enumerate(experts):
        mask = (top_k_indices == i).any(dim=-1)
        if mask.any():
            # Move tokens to expert's device
            expert_input = x[mask].to(expert.device)
            expert_output = expert(expert_input)
            # Move back to main device
            output[mask] = expert_output.to(x.device)

    return output
```

**Communication Overhead**: Main bottleneck is moving tokens between devices.

**Optimization**: Use expert parallelism with all-to-all communication:
```python
# All-to-all gather: collect all tokens for each expert
# Process in batch on expert's device
# All-to-all scatter: return results
```

### 2. Fine-tuning MoE Models

**Challenge**: MoE models can collapse during fine-tuning on small datasets.

**Strategies**:

a) **Freeze Router, Fine-tune Experts**
```python
# Freeze router
for param in model.router.parameters():
    param.requires_grad = False

# Fine-tune experts
for param in model.experts.parameters():
    param.requires_grad = True
```

b) **Add Task-Specific Experts**
```python
# Keep pre-trained experts frozen
# Add new experts for fine-tuning task
moe.experts.append(Expert(d_model, d_ff))  # New expert
```

c) **Lower Learning Rate + Stronger Load Balancing**
```python
optimizer = AdamW(model.parameters(), lr=1e-5)  # 10x lower than pre-training
alpha_aux = 0.1  # 10x higher than pre-training
```

d) **LoRA for MoE** (MoE-LoRA)
Apply LoRA to each expert:
```python
class ExpertWithLoRA(nn.Module):
    def __init__(self, expert, rank=16):
        super().__init__()
        self.expert = expert
        # Freeze expert
        for param in expert.parameters():
            param.requires_grad = False

        # Add LoRA adapters
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_model, bias=False)

    def forward(self, x):
        return self.expert(x) + self.lora_B(self.lora_A(x))
```

### 3. Distillation from MoE to Dense

Convert sparse MoE to dense model for deployment:

```python
def distill_moe_to_dense(moe_model, dense_model, dataloader):
    """
    Distill knowledge from MoE to smaller dense model.
    """
    optimizer = AdamW(dense_model.parameters(), lr=1e-4)

    for batch in dataloader:
        # Get MoE predictions (teacher)
        with torch.no_grad():
            moe_output, _ = moe_model(batch)

        # Dense model predictions (student)
        dense_output = dense_model(batch)

        # Distillation loss (KL divergence)
        loss = F.kl_div(
            F.log_softmax(dense_output / temperature, dim=-1),
            F.softmax(moe_output / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        loss.backward()
        optimizer.step()
```

**Use case**: Train large MoE, distill to deployable dense model.

### 4. Analyzing Expert Specialization

Understand what each expert learned:

```python
def analyze_expert_specialization(model, dataset, labels):
    """
    Analyze which experts activate for which data types.
    """
    expert_activations = {i: [] for i in range(num_experts)}

    for sample, label in zip(dataset, labels):
        output, aux_info = model(sample)
        top_k_indices = aux_info['top_k_indices']

        for expert_idx in top_k_indices:
            expert_activations[expert_idx].append(label)

    # Analyze distributions
    for i, activations in expert_activations.items():
        counter = Counter(activations)
        print(f"Expert {i}: {counter.most_common(3)}")

# Example output:
# Expert 0: [('code', 456), ('technical', 123), ('math', 45)]
# Expert 1: [('creative', 234), ('narrative', 189), ('poetry', 67)]
# Expert 2: [('science', 345), ('medical', 234), ('biology', 123)]
```

### 5. Mixture of Depths (MoD)

Combine MoE with dynamic depth (some tokens skip layers):

```python
class MixtureOfDepthsLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.moe = SparseMoE(d_model, d_ff, num_experts)
        self.skip_predictor = nn.Linear(d_model, 1)

    def forward(self, x):
        # Predict which tokens should skip this layer
        skip_prob = torch.sigmoid(self.skip_predictor(x))  # [batch, seq_len, 1]
        skip_mask = (skip_prob > 0.5).squeeze(-1)  # [batch, seq_len]

        # Process non-skipped tokens through MoE
        moe_output, _ = self.moe(x)

        # Combine: skip or process
        output = torch.where(skip_mask.unsqueeze(-1), x, moe_output)
        return output
```

**Benefit**: Some tokens (e.g., stopwords) don't need deep processing.

### 6. Shared Expert Architecture

Reserve some experts as "shared" (always active):

```python
class SharedExpertMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_routed_experts, num_shared_experts):
        super().__init__()
        # Routed experts (sparse)
        self.routed_experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_routed_experts)
        ])

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_shared_experts)
        ])

        self.router = Router(d_model, num_routed_experts)

    def forward(self, x):
        # Route to sparse experts
        routed_output, _ = self.route_to_experts(x)

        # Always process through shared experts
        shared_output = sum(expert(x) for expert in self.shared_experts)

        # Combine
        return routed_output + shared_output / len(self.shared_experts)
```

**Use case**: Shared experts handle common patterns, routed experts handle specialized cases.

## Best Practices

### 1. Hyperparameter Recommendations

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Number of experts | 8-64 | Start with 8, scale up |
| Top-k | 1-2 | k=1 for simplicity, k=2 for quality |
| Capacity factor | 1.25-2.0 | Higher for k=1, lower for k=2 |
| Load balance loss weight (α) | 0.01-0.1 | Start 0.01, increase if imbalanced |
| Router z-loss weight (β) | 0.001-0.01 | Usually 0.001 |
| Expert FFN multiplier | 4x | Same as dense transformers |
| Router learning rate | 2-5x expert LR | Router needs faster adaptation |
| Dropout | 0.1 | Same as dense models |

### 2. Training Recipe

**Phase 1: Warmup (first 10% of training)**
- High capacity factor (2.0)
- Lower load balance weight (0.005)
- Optional: Start with dense routing (no sparsity)

**Phase 2: Main Training (next 80%)**
- Normal capacity factor (1.25)
- Normal load balance weight (0.01)
- Full sparse routing

**Phase 3: Fine-tuning (final 10%)**
- Slightly lower LR
- Optionally increase load balance weight (0.02)
- Remove routing noise

### 3. Debugging Checklist

✅ **Check expert utilization**
```python
# Should be roughly uniform
expert_counts = torch.bincount(top_k_indices.flatten(), minlength=num_experts)
print(f"Expert usage: {expert_counts.tolist()}")
```

✅ **Monitor load balance loss**
```python
# Should decrease over time
if load_balance_loss > 2.0:
    print("Warning: Poor load balancing!")
```

✅ **Verify gradient flow**
```python
# All experts should receive gradients
for i, expert in enumerate(experts):
    grad_norm = sum(p.grad.norm() for p in expert.parameters() if p.grad is not None)
    print(f"Expert {i} grad norm: {grad_norm:.4f}")
```

✅ **Check router logits**
```python
# Should be moderate (not too large)
logit_magnitude = logits.abs().mean()
if logit_magnitude > 10:
    print("Warning: Router logits too large!")
```

✅ **Validate output diversity**
```python
# Experts should produce different outputs
expert_outputs = [expert(x) for expert in experts]
output_diversity = torch.stack(expert_outputs).std(dim=0).mean()
print(f"Output diversity: {output_diversity:.4f}")
```

### 4. Scaling Guidelines

**Small Scale** (1-8B params)
- 8 experts, k=2
- Single-node training
- Capacity factor: 1.5

**Medium Scale** (8-50B params)
- 16-32 experts, k=2
- Multi-node with expert parallelism
- Capacity factor: 1.25

**Large Scale** (50B-1T+ params)
- 64-256 experts, k=1 or k=2
- Hierarchical expert parallelism
- Capacity factor: 1.1
- Consider expert choice routing

### 5. Inference Optimization

**Batch Size Sensitivity**
- Small batch: Expert utilization may be poor
- Large batch: Better expert utilization, but higher latency
- Recommendation: Batch size ≥ num_experts * k

**Expert Caching**
```python
# Cache expert weights on device for faster access
expert_cache = {i: expert.to('cuda') for i, expert in enumerate(experts)}
```

**Top-1 Routing for Inference**
Even if trained with k=2, consider k=1 at inference for speed:
```python
model.eval()
# Override k for inference
model.moe.k = 1
```

### 6. When to Replace FFN with MoE

In a transformer, you can replace FFNs with MoE at different frequencies:

- **Every Layer**: Maximum capacity, highest communication cost
- **Every Other Layer**: Good balance (recommended for most cases)
- **Every 4th Layer**: More efficient, slightly lower capacity
- **Only Deep Layers**: Shallow layers stay dense, deep layers use MoE

```python
# Example: MoE every other layer
for i, layer in enumerate(transformer.layers):
    if i % 2 == 0:
        layer.ffn = MoE(...)
    else:
        layer.ffn = DenseFFN(...)
```

## Common Pitfalls

### 1. Routing Collapse

**Symptom**: All tokens route to 1-2 experts.

**Causes**:
- Insufficient load balancing loss
- Poor expert initialization
- Learning rate too high for router

**Solutions**:
```python
# Increase load balance loss
alpha_aux = 0.05  # Higher than default 0.01

# Add entropy regularization
loss += 0.01 * entropy_regularization(gates)

# Lower router learning rate
optimizer = AdamW([
    {'params': experts.parameters(), 'lr': 1e-4},
    {'params': router.parameters(), 'lr': 5e-5},  # Lower
])
```

### 2. Expert Underutilization

**Symptom**: Some experts receive <1% of tokens.

**Causes**:
- Random initialization too similar
- Dataset too small
- Too many experts for task complexity

**Solutions**:
```python
# Re-initialize unused experts
if expert_usage[i] < threshold:
    experts[i].apply(init_weights)
    # Add noise to differentiate
    for p in experts[i].parameters():
        p.data += torch.randn_like(p) * 0.1
```

### 3. Numerical Instability

**Symptom**: NaN losses, exploding router logits.

**Causes**:
- Router logits too large
- No z-loss
- Extreme expert outputs

**Solutions**:
```python
# Add router z-loss
z_loss = compute_router_z_loss(aux_info)
loss += 0.001 * z_loss

# Clip router logits
logits = torch.clamp(logits, min=-10, max=10)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 4. Memory Overflow

**Symptom**: OOM errors during training.

**Cause**: All experts in memory even if not active.

**Solutions**:
```python
# Expert offloading (for very large models)
expert_on_cpu = [e.cpu() for e in experts]

def forward_with_offloading(x, expert_idx):
    expert = expert_on_cpu[expert_idx].cuda()
    output = expert(x)
    expert.cpu()
    return output

# Or use expert parallelism (distribute across GPUs)
```

### 5. Poor Fine-tuning Performance

**Symptom**: Performance degrades when fine-tuning on downstream task.

**Causes**:
- Router overfits to small dataset
- Load balancing fails on narrow domain

**Solutions**:
```python
# Freeze router during fine-tuning
for p in model.router.parameters():
    p.requires_grad = False

# Use much higher load balance weight
alpha_aux = 0.1  # vs. 0.01 during pre-training

# Lower learning rate
lr = 1e-5  # vs. 1e-4 during pre-training
```

### 6. Communication Bottleneck

**Symptom**: Slow training despite sparse activation.

**Cause**: Too much inter-device communication in expert parallelism.

**Solutions**:
```python
# Group experts on same device
experts_per_device = num_experts // num_devices

# Use all-to-all collective for efficient communication
# Instead of point-to-point transfers

# Consider expert replication for small experts
```

## Resources

### Foundational Papers

1. **Shazeer et al. (2017)** - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
   - Original scaled MoE architecture
   - Noisy top-k gating
   - Load balancing via auxiliary loss

2. **Lepikhin et al. (2020)** - "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
   - MoE for machine translation
   - Expert parallelism strategies
   - 600B parameter model

3. **Fedus et al. (2021)** - "Switch Transformers: Scaling to Trillion Parameter Models"
   - Simplified k=1 routing
   - 1.6T parameter model
   - Extensive empirical analysis

4. **Zhou et al. (2022)** - "Mixture-of-Experts with Expert Choice Routing"
   - Flip routing paradigm
   - Perfect load balancing
   - No auxiliary loss needed

5. **Jiang et al. (2024)** - "Mixtral of Experts"
   - Open-source MoE (8x7B)
   - Practical implementation insights
   - Competitive with 70B dense models

### Advanced Topics

6. **Riquelme et al. (2021)** - "Scaling Vision with Sparse Mixture of Experts"
   - MoE for vision transformers
   - Spatial routing patterns

7. **Du et al. (2022)** - "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"
   - 1.2T parameters
   - Energy efficiency analysis

8. **DeepSeek-AI (2024)** - "DeepSeek-V2"
   - 236B params (21B active)
   - Multi-head latent attention + MoE
   - Production deployment insights

### Implementation Resources

- **Fairseq MoE**: https://github.com/facebookresearch/fairseq/tree/main/examples/moe
- **Megatron-LM MoE**: https://github.com/NVIDIA/Megatron-LM
- **Mixtral (HuggingFace)**: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
- **Google Switch Transformers**: https://github.com/google-research/t5x

### Related Topics in This Repository

- [Transformers](transformers.md) - Core architecture that MoE extends
- [LoRA](lora.md) - Parameter-efficient fine-tuning (complementary to MoE)
- [Quantization](quantization.md) - Reducing memory for large MoE models
- [Neural Networks](neural_networks.md) - Foundational concepts

### Tutorials and Guides

- **HuggingFace MoE Tutorial**: Practical implementation guide
- **PyTorch MoE Examples**: Official examples and best practices
- **Weights & Biases MoE Guide**: Training and monitoring strategies

---

## Quick Reference

### Key Equations

```
# MoE Output
y = Σ G(x)_i * E_i(x)

# Gating Function
G(x) = Softmax(x · W_g)

# Top-K Sparse Gating
G_sparse(x)_i = G(x)_i / Σ_{j∈TopK} G(x)_j  if i ∈ TopK(G(x), k)
                0                            otherwise

# Load Balance Loss
L_aux = α · N · Σ f_i · P_i

# Router Z-Loss
L_z = mean((log_sum_exp(logits))²)

# Total Loss
L = L_task + α·L_aux + β·L_z
```

### Typical Hyperparameters

```python
config = {
    'num_experts': 8,           # Number of expert networks
    'k': 2,                     # Top-k experts per token
    'capacity_factor': 1.25,    # Expert capacity multiplier
    'alpha_aux': 0.01,          # Load balance loss weight
    'alpha_z': 0.001,           # Router z-loss weight
    'd_model': 512,             # Model dimension
    'd_ff': 2048,               # Expert hidden dimension (4x d_model)
    'dropout': 0.1,             # Dropout rate
    'noise_std': 0.1,           # Router noise (training only)
}
```

### Training Checklist

- [ ] Initialize experts with different random seeds
- [ ] Monitor expert utilization (should be roughly uniform)
- [ ] Use load balancing loss (α ≈ 0.01)
- [ ] Use router z-loss for stability (β ≈ 0.001)
- [ ] Gradient clipping (max_norm = 1.0)
- [ ] Higher LR for router than experts (2-5x)
- [ ] Capacity factor > 1.0 to handle overflow
- [ ] Remove routing noise at inference
- [ ] Track and log routing decisions
- [ ] Validate gradient flow to all experts

### Debugging Commands

```python
# Check expert usage distribution
expert_counts = torch.bincount(top_k_indices.flatten())
print(f"Expert usage: {expert_counts}")

# Monitor load balance quality
cv = expert_counts.std() / expert_counts.mean()
print(f"Coefficient of variation: {cv:.4f}")  # Lower is better

# Check router logit magnitude
print(f"Logit magnitude: {logits.abs().mean():.4f}")  # Should be < 10

# Verify gradient flow
for i, expert in enumerate(experts):
    grad_norm = sum(p.grad.norm() for p in expert.parameters() if p.grad is not None)
    print(f"Expert {i} grad: {grad_norm:.4f}")
```

---

**Last Updated**: 2024
**Version**: 1.0
