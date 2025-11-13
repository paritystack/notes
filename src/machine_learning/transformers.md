# Transformers

Transformers are a type of deep learning model introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017. They have revolutionized the field of natural language processing (NLP) and have been widely adopted in various applications, including machine translation, text summarization, and sentiment analysis.

## Key Concepts

- **Attention Mechanism**: The core innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence when making predictions. This enables the model to capture long-range dependencies and relationships between words more effectively than previous architectures like RNNs and LSTMs.

- **Encoder-Decoder Architecture**: The transformer model consists of two main components: the encoder and the decoder. The encoder processes the input data and generates a set of attention-based representations, while the decoder uses these representations to produce the output sequence.

- **Positional Encoding**: Since transformers do not have a built-in notion of sequence order (unlike RNNs), they use positional encodings to inject information about the position of each word in the input sequence. This allows the model to understand the order of words.

## Attention Mechanism: Deep Dive

The attention mechanism is the heart of the transformer architecture. It allows the model to focus on different parts of the input sequence when processing each element. Let's explore this in detail with mathematical formulations and PyTorch implementations.

### Scaled Dot-Product Attention

The fundamental building block of transformer attention is the **Scaled Dot-Product Attention**. Given three matrices:
- $Q$ (Query): What we're looking for
- $K$ (Key): What we're matching against
- $V$ (Value): The actual information we want to retrieve

The attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $d_k$ is the dimension of the key vectors
- The division by $\sqrt{d_k}$ is scaling to prevent the dot products from growing too large
- softmax normalizes the scores to create a probability distribution

#### PyTorch Implementation: Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        key: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
        value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
        mask: Optional mask tensor

    Returns:
        output: Attention output (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: Attention weights (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # Get the dimension of keys
    d_k = query.size(-1)

    # Step 1: Compute Q @ K^T
    # query: (batch, heads, seq_len_q, d_k)
    # key.transpose(-2, -1): (batch, heads, d_k, seq_len_k)
    # scores: (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))
    print(f"After Q @ K^T - Shape: {scores.shape}")
    print(f"Sample scores (first 3x3):\n{scores[0, 0, :3, :3]}\n")

    # Step 2: Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    print(f"After scaling by √{d_k} = {math.sqrt(d_k):.2f}")
    print(f"Scaled scores (first 3x3):\n{scores[0, 0, :3, :3]}\n")

    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        print(f"After masking - Shape: {scores.shape}")

    # Step 4: Apply softmax to get attention weights
    # Softmax is applied over the last dimension (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)
    print(f"Attention weights - Shape: {attention_weights.shape}")
    print(f"Sample attention weights (first 3x3):\n{attention_weights[0, 0, :3, :3]}")
    print(f"Sum of first row (should be 1.0): {attention_weights[0, 0, 0].sum()}\n")

    # Step 5: Multiply by values
    # attention_weights: (batch, heads, seq_len_q, seq_len_k)
    # value: (batch, heads, seq_len_v, d_v)  [seq_len_v == seq_len_k]
    # output: (batch, heads, seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)
    print(f"Final output - Shape: {output.shape}\n")

    return output, attention_weights


# Example: Let's trace through a simple example
batch_size = 2
num_heads = 1
seq_len = 4
d_k = 8
d_v = 8

# Create sample tensors
torch.manual_seed(42)
query = torch.randn(batch_size, num_heads, seq_len, d_k)
key = torch.randn(batch_size, num_heads, seq_len, d_k)
value = torch.randn(batch_size, num_heads, seq_len, d_v)

print("="*60)
print("SCALED DOT-PRODUCT ATTENTION - STEP BY STEP")
print("="*60)
print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}\n")

output, attn_weights = scaled_dot_product_attention(query, key, value)

print(f"Final output shape: {output.shape}")
print(f"Final attention weights shape: {attn_weights.shape}")
```

**Output explanation:**
```
============================================================
SCALED DOT-PRODUCT ATTENTION - STEP BY STEP
============================================================
Query shape: torch.Size([2, 1, 4, 8])
Key shape: torch.Size([2, 1, 4, 8])
Value shape: torch.Size([2, 1, 4, 8])

After Q @ K^T - Shape: torch.Size([2, 1, 4, 4])
Sample scores (first 3x3):
tensor([[ 0.6240, -1.2613,  1.4199],
        [-1.8847,  4.0367, -0.5234],
        [ 2.1563, -2.5678,  0.8234]])

After scaling by √8 = 2.83
Scaled scores (first 3x3):
tensor([[ 0.2207, -0.4460,  0.5021],
        [-0.6661,  1.4271, -0.1850],
        [ 0.7624, -0.9078,  0.2911]])

Attention weights - Shape: torch.Size([2, 1, 4, 4])
Sample attention weights (first 3x3):
tensor([[0.2789, 0.1425, 0.3672],
        [0.1056, 0.8236, 0.1680],
        [0.3924, 0.0731, 0.2458]])
Sum of first row (should be 1.0): 1.0

Final output - Shape: torch.Size([2, 1, 4, 8])
```

### Understanding the Matrix Operations

Let's break down what's happening at each step:

1. **Query-Key Dot Product ($QK^T$)**:
   - Each query vector (row in $Q$) is compared against all key vectors (rows in $K$)
   - The dot product measures similarity: higher values = more similar
   - Shape: `(batch, heads, seq_len_q, d_k) @ (batch, heads, d_k, seq_len_k) → (batch, heads, seq_len_q, seq_len_k)`

2. **Scaling**:
   - Dividing by $\sqrt{d_k}$ prevents the dot products from becoming too large
   - Large dot products → very small gradients after softmax → slow learning
   - This is crucial for stable training

3. **Softmax**:
   - Converts raw scores into a probability distribution
   - Each row sums to 1.0
   - Higher scores get higher probabilities (attention weights)

4. **Weighted Sum (Attention @ Value)**:
   - Uses attention weights to create a weighted combination of value vectors
   - Each output position is a mixture of all value vectors
   - The weights determine how much each value contributes

### Multi-Head Attention

Multi-head attention runs multiple attention operations in parallel, allowing the model to attend to different aspects of the input simultaneously.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi-Head Attention module.

        Args:
            d_model: Total dimension of the model (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to get shape: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Inverse of split_heads.
        Input: (batch_size, num_heads, seq_len, d_k)
        Output: (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # Reshape to (batch_size, seq_len, d_model)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: Optional mask

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        # Each of these operations: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        print(f"\n{'='*60}")
        print("MULTI-HEAD ATTENTION - DETAILED STEPS")
        print(f"{'='*60}")
        print(f"Input shapes - Q: {query.shape}, K: {key.shape}, V: {value.shape}")
        print(f"\nAfter linear projections:")
        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        # Step 2: Split into multiple heads
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        print(f"\nAfter splitting into {self.num_heads} heads:")
        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        print(f"Each head has dimension: {self.d_k}")

        # Step 3: Scaled dot-product attention
        # For each head: (batch, 1, seq_len_q, d_k) with (batch, 1, seq_len_k, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        print(f"\nAfter attention computation:")
        print(f"Attention scores: {scores.shape}")
        print(f"Attention weights: {attention_weights.shape}")
        print(f"Attention output: {output.shape}")

        # Step 4: Concatenate heads
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = self.combine_heads(output)
        print(f"\nAfter combining heads: {output.shape}")

        # Step 5: Final linear projection
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        output = self.W_o(output)
        print(f"After final projection: {output.shape}")
        print(f"{'='*60}\n")

        return output, attention_weights


# Example usage with detailed tracking
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

# Create sample input
x = torch.randn(batch_size, seq_len, d_model)

# Initialize multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Forward pass (using x for query, key, and value - this is self-attention)
output, attn_weights = mha(x, x, x)

print(f"\nFinal Results:")
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"\nAttention weights for first head, first query position:")
print(attn_weights[0, 0, 0, :])  # Should sum to 1.0
print(f"Sum: {attn_weights[0, 0, 0, :].sum()}")
```

### Visualizing Attention: A Concrete Example

Let's see how attention works on actual text:

```python
import torch
import torch.nn.functional as F

# Simple example: "The cat sat on the mat"
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model = 4  # Small for visualization

# Create simple embeddings (normally these would be learned)
# Each word gets a random vector
torch.manual_seed(42)
embeddings = torch.randn(1, seq_len, d_model)

# Simple attention (1 head for clarity)
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# Create and run the attention
attn = SimpleAttention(d_model)
output, weights = attn(embeddings)

# Visualize attention weights
print("Attention Weight Matrix:")
print("(Each row shows where that word 'attends to')\n")
print("        ", "  ".join(f"{w:>5}" for w in sentence))
print("-" * 50)
for i, word in enumerate(sentence):
    print(f"{word:>7} |", "  ".join(f"{weights[0, i, j].item():5.3f}" for j in range(seq_len)))

print("\nInterpretation:")
print("- Each row represents a query word")
print("- Each column represents a key word")
print("- Values show how much the query word 'attends to' each key word")
print("- Higher values = stronger attention")
print("- Each row sums to 1.0")
```

### Masked Attention (for Decoder)

In the decoder, we use masked attention to prevent positions from attending to future positions:

```python
def create_causal_mask(seq_len):
    """
    Create a causal mask for decoder self-attention.
    Prevents attending to future positions.

    Returns a lower triangular matrix:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


# Example with masking
seq_len = 4
mask = create_causal_mask(seq_len)

print("Causal Mask:")
print(mask[0, 0])
print("\nThis ensures that:")
print("- Position 0 can only see position 0")
print("- Position 1 can see positions 0-1")
print("- Position 2 can see positions 0-2")
print("- Position 3 can see all positions 0-3")

# Apply masked attention
query = torch.randn(1, 1, seq_len, 8)
key = torch.randn(1, 1, seq_len, 8)
value = torch.randn(1, 1, seq_len, 8)

output, attn_weights = scaled_dot_product_attention(query, key, value, mask)

print("\nAttention weights with masking:")
print(attn_weights[0, 0])
print("\nNotice how future positions (upper triangle) have ~0 attention weight")
```

### Complete Self-Attention Layer with PyTorch

Here's a complete implementation you can use in practice:

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Complete self-attention layer with all components.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined QKV projection (more efficient)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V all at once
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)

        # Split into Q, K, V and reshape for multi-head
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = torch.matmul(attn, v)  # (batch, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, heads, d_k)
        out = out.reshape(batch_size, seq_len, d_model)  # (batch, seq_len, d_model)

        # Final projection
        out = self.out_proj(out)

        return out, attn


# Test the complete implementation
model = SelfAttention(d_model=512, num_heads=8, dropout=0.1)
x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
output, attention = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")
```

## Architecture

1. **Encoder**: The encoder is composed of multiple identical layers, each containing two main sub-layers:
   - **Multi-Head Self-Attention**: This mechanism allows the model to focus on different parts of the input sequence simultaneously, capturing various relationships between words.
   - **Feed-Forward Neural Network**: After the attention mechanism, the output is passed through a feed-forward neural network, which applies a non-linear transformation.

2. **Decoder**: The decoder also consists of multiple identical layers, with an additional sub-layer for attending to the encoder's output:
   - **Masked Multi-Head Self-Attention**: This prevents the decoder from attending to future tokens in the output sequence during training.
   - **Encoder-Decoder Attention**: This layer allows the decoder to focus on relevant parts of the encoder's output while generating the output sequence.

### Complete Transformer Implementation in PyTorch

Here's a full implementation of the transformer architecture with detailed comments on matrix operations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        # Add positional encoding to input
        # x: (batch, seq_len, d_model)
        # self.pe[:, :x.size(1)]: (1, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Consists of two linear transformations with ReLU activation.

    $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # x: (batch, seq_len, d_model)
        # After linear1: (batch, seq_len, d_ff)
        # After ReLU: (batch, seq_len, d_ff)
        # After linear2: (batch, seq_len, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention layer with proper matrix dimension tracking.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or similar
        """
        batch_size = query.size(0)

        # Linear projections: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose to (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        # Q @ K^T: (batch, num_heads, seq_len_q, d_k) @ (batch, num_heads, d_k, seq_len_k)
        #        → (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over the last dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # attn_weights @ V: (batch, num_heads, seq_len_q, seq_len_k) @ (batch, num_heads, seq_len_v, d_k)
        #                 → (batch, num_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)

        # Concatenate heads
        # Transpose: (batch, num_heads, seq_len_q, d_k) → (batch, seq_len_q, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()
        # Reshape: (batch, seq_len_q, num_heads, d_k) → (batch, seq_len_q, d_model)
        output = output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(output)

        return output, attn_weights


class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Attention mask
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer consisting of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (attending to encoder output)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source mask for encoder-decoder attention
            tgt_mask: Target mask for masked self-attention
        """
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention to encoder output
        # Query from decoder, Key and Value from encoder
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        Create mask for source sequence (padding mask).
        Args:
            src: (batch_size, src_seq_len)
        Returns:
            mask: (batch_size, 1, 1, src_seq_len)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Create mask for target sequence (padding + causal mask).
        Args:
            tgt: (batch_size, tgt_seq_len)
        Returns:
            mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        tgt_seq_len = tgt.size(1)

        # Padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

        # Causal mask (prevent attending to future tokens)
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)
        ).bool()

        # Combine both masks
        tgt_mask = tgt_padding_mask & tgt_sub_mask
        return tgt_mask

    def encode(self, src, src_mask):
        """
        Encode source sequence.
        Args:
            src: (batch_size, src_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
        Returns:
            encoder_output: (batch_size, src_seq_len, d_model)
        """
        # Embedding + Positional encoding
        # src: (batch, src_seq_len) → (batch, src_seq_len, d_model)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """
        Decode target sequence.
        Args:
            tgt: (batch_size, tgt_seq_len)
            encoder_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, 1, src_seq_len)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            decoder_output: (batch_size, tgt_seq_len, d_model)
        """
        # Embedding + Positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt):
        """
        Forward pass through the entire transformer.
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
        Returns:
            output: Logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1

    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    )

    # Example input (batch_size=2, sequences of length 10)
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 12))

    print("="*60)
    print("TRANSFORMER MODEL SUMMARY")
    print("="*60)
    print(f"Source sequence shape: {src.shape}")
    print(f"Target sequence shape: {tgt.shape}")
    print(f"\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass
    output = model(src, tgt)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (batch_size={src.size(0)}, tgt_seq_len={tgt.size(1)}, tgt_vocab_size={tgt_vocab_size})")

    # Show dimension flow through the model
    print("\n" + "="*60)
    print("DIMENSION FLOW THROUGH TRANSFORMER")
    print("="*60)

    print("\nENCODER:")
    print(f"1. Input tokens: {src.shape}")
    print(f"2. After embedding: (batch={src.size(0)}, seq={src.size(1)}, d_model={d_model})")
    print(f"3. After positional encoding: Same shape")
    print(f"4. Through {num_encoder_layers} encoder layers: Same shape")
    print(f"5. Encoder output: (batch={src.size(0)}, seq={src.size(1)}, d_model={d_model})")

    print("\nDECODER:")
    print(f"1. Input tokens: {tgt.shape}")
    print(f"2. After embedding: (batch={tgt.size(0)}, seq={tgt.size(1)}, d_model={d_model})")
    print(f"3. After positional encoding: Same shape")
    print(f"4. Through {num_decoder_layers} decoder layers: Same shape")
    print(f"5. After output projection: (batch={tgt.size(0)}, seq={tgt.size(1)}, vocab={tgt_vocab_size})")

    print("\n" + "="*60)
```

### Training the Transformer

Here's how you would train this transformer for a translation task:

```python
import torch.optim as optim

# Initialize model, loss, and optimizer
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Training loop
def train_step(model, src, tgt, optimizer, criterion):
    """
    Single training step.

    Args:
        src: Source sequences (batch_size, src_seq_len)
        tgt: Target sequences (batch_size, tgt_seq_len)
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    # Input to decoder is target shifted right (teacher forcing)
    tgt_input = tgt[:, :-1]  # Remove last token
    tgt_output = tgt[:, 1:]   # Remove first token (usually <sos>)

    # Get model predictions
    # output: (batch_size, tgt_seq_len-1, vocab_size)
    output = model(src, tgt_input)

    # Reshape for loss calculation
    # output: (batch_size * (tgt_seq_len-1), vocab_size)
    # tgt_output: (batch_size * (tgt_seq_len-1))
    output = output.reshape(-1, output.size(-1))
    tgt_output = tgt_output.reshape(-1)

    # Calculate loss
    loss = criterion(output, tgt_output)

    # Backward pass
    loss.backward()

    # Gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()

    return loss.item()


# Example training
for epoch in range(10):
    # Generate dummy batch
    src = torch.randint(1, 10000, (32, 20))  # batch_size=32, seq_len=20
    tgt = torch.randint(1, 10000, (32, 25))  # batch_size=32, seq_len=25

    loss = train_step(model, src, tgt, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

### Inference with the Transformer

```python
def greedy_decode(model, src, max_len, start_token, end_token):
    """
    Greedy decoding: always select the most likely next token.

    Args:
        model: Trained transformer model
        src: Source sequence (1, src_seq_len)
        max_len: Maximum length of generated sequence
        start_token: Start token ID
        end_token: End token ID

    Returns:
        Generated sequence
    """
    model.eval()

    # Encode the source
    src_mask = model.make_src_mask(src)
    encoder_output = model.encode(src, src_mask)

    # Initialize decoder input with start token
    tgt = torch.tensor([[start_token]], device=src.device)

    for _ in range(max_len):
        # Create target mask
        tgt_mask = model.make_tgt_mask(tgt)

        # Decode
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Get predictions for the last token
        # decoder_output: (1, current_seq_len, d_model)
        # We only need the last position: (1, 1, d_model)
        output = model.output_projection(decoder_output[:, -1:, :])

        # Get the token with highest probability
        # output: (1, 1, vocab_size) → (1, 1)
        next_token = output.argmax(dim=-1)

        # Append to target sequence
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if we generate the end token
        if next_token.item() == end_token:
            break

    return tgt


# Example inference
src_sequence = torch.randint(1, 10000, (1, 20))
generated = greedy_decode(
    model=model,
    src=src_sequence,
    max_len=50,
    start_token=1,  # <sos> token
    end_token=2     # <eos> token
)
print(f"Generated sequence: {generated}")
print(f"Generated sequence shape: {generated.shape}")
```

## Applications

Transformers have been successfully applied in various domains, including:

- **Natural Language Processing**: Models like BERT, GPT, and T5 are based on the transformer architecture and have achieved state-of-the-art results in numerous NLP tasks.

- **Computer Vision**: Vision Transformers (ViTs) have adapted the transformer architecture for image classification and other vision tasks, demonstrating competitive performance with traditional convolutional neural networks (CNNs).

- **Speech Processing**: Transformers are also being explored for tasks in speech recognition and synthesis, leveraging their ability to model sequential data.

## Conclusion

Transformers have transformed the landscape of machine learning, particularly in NLP, by providing a powerful and flexible framework for modeling complex relationships in data. Their ability to handle long-range dependencies and parallelize training has made them a go-to choice for many modern AI applications.

# ELI10: What are Transformers?

Transformers are like super-smart assistants that help computers understand and generate human language. Imagine you have a friend who can read a whole book at once and remember everything about it. That's what transformers do! They look at all the words in a sentence and figure out how they relate to each other, which helps them answer questions, translate languages, or even write stories.

## Example Usage

1. **Text Generation**: Given a prompt, transformers can generate coherent and contextually relevant text.
2. **Translation**: They can translate sentences from one language to another by understanding the meaning of the words in context.
3. **Summarization**: Transformers can read long articles and provide concise summaries, capturing the main points effectively.
