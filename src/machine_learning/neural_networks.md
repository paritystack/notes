# Neural Networks

## Overview

A neural network is a machine learning model inspired by biological brains. It consists of interconnected nodes (neurons) organized in layers that learn patterns from data.

## Table of Contents

### Fundamentals
- [Basic Architecture](#basic-architecture)
- [Key Components](#key-components)
- [Training Process](#training-process)
- [Code Example](#code-example-pytorch)
- [Network Types](#network-types)
- [Hyperparameters](#hyperparameters)

### Advanced Topics
- [Modern Architectures](#modern-architectures)
- [Advanced Training Techniques](#advanced-training-techniques)
- [Practical Considerations](#practical-considerations)

### Resources
- [Training Tips](#training-tips)
- [Common Issues](#common-issues)
- [Quick Reference](#quick-reference)
- [Further Resources](#further-resources)
- [ELI10](#eli10)

---

# Fundamentals

## Basic Architecture

```
Input Layer    Hidden Layers          Output Layer
     o              o                    o
     o              o                    o
     o              o                    o
     o              o
     o              o                    o
     o              o
   [n inputs]   [hidden units]   [output units]
```

## Key Components

### Neurons
Each neuron applies transformation: $\text{output} = \text{activation}(\text{weights} \cdot \text{inputs} + \text{bias})$

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| ReLU | $\max(0, x)$ | $[0, \infty)$ | Hidden layers |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary classification |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$ probabilities | Multi-class output |
| Linear | $x$ | $(-\infty, \infty)$ | Regression output |

### Layers

1. **Input Layer**: Raw data (28x28 pixels, word embeddings, etc.)
2. **Hidden Layers**: Learn complex patterns through non-linear transformations
3. **Output Layer**: Final predictions

## Training Process

### Forward Pass
Input flows through network:
```
x → w1 + b1 → activation → ... → output
```

### Loss Function
Measures prediction error:
- **MSE** (regression): Mean squared error
- **Cross-Entropy** (classification): Measures probability difference

### Backpropagation
Calculates gradients and updates weights:
```
1. Compute loss
2. Calculate gradients: ∂(loss)/∂(weights)
3. Update weights: w = w - learning_rate × gradient
4. Repeat
```

### Optimizers

| Optimizer | Learning | Best For |
|-----------|----------|----------|
| SGD | Fixed or decaying | Simple tasks |
| Momentum | Accelerated | Faster convergence |
| Adam | Adaptive | Most modern tasks |
| RMSprop | Adaptive | Deep networks |

## Code Examples

### Basic Neural Network

#### PyTorch Implementation

```python
import torch
import torch.nn as nn
from torch.optim import Adam

# Define network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create network
model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras

# Define network (Sequential API)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# Functional API (for complex architectures)
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Model with dropout and batch norm
class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(loader), 100. * correct / total

# Main training loop
def train_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model

# Usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AdvancedNN(input_size=784, hidden_sizes=[512, 256, 128], output_size=10).to(device)

train_dataset = CustomDataset(train_data, train_labels)
val_dataset = CustomDataset(val_data, val_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

model = train_model(model, train_loader, val_loader, epochs=50, device=device)
```

### Custom Layer Example

```python
class CustomAttentionLayer(nn.Module):
    """Simple self-attention layer"""
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
```

### Inference Example

```python
def inference(model, input_data, device='cuda'):
    """Run inference with proper preprocessing"""
    model.eval()

    # Preprocess
    input_tensor = torch.from_numpy(input_data).float()
    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=-1)
        predicted_class = output.argmax(dim=-1)

    return predicted_class.cpu().numpy(), probabilities.cpu().numpy()

# Batch inference
predictions, probs = inference(model, test_data)

# Single sample
single_pred, single_probs = inference(model, test_data[0:1])
print(f"Predicted: {single_pred[0]}, Confidence: {single_probs[0][single_pred[0]]:.2%}")
```

## Network Types

### Feedforward Neural Networks (FNN)
- Data flows one direction only
- Simplest type, works for structured data

### Convolutional Neural Networks (CNN)
- Specialized for image processing
- Uses filters to extract spatial features
- Reduces parameters through weight sharing

### Recurrent Neural Networks (RNN)
- Processes sequences (text, time series)
- Maintains hidden state between inputs
- Variants: LSTM, GRU (better long-term memory)

### Transformers
- Attention-based architecture
- Parallel processing of sequences
- Powers modern LLMs (GPT, BERT)

## Hyperparameters

| Parameter | Impact | Typical Values |
|-----------|--------|-----------------|
| Learning Rate | Convergence speed, stability | 0.001 - 0.1 |
| Batch Size | Memory, stability | 32 - 256 |
| Hidden Units | Capacity | 64 - 2048 |
| Epochs | Training duration | 10 - 100 |
| Dropout | Regularization | 0.3 - 0.5 |

## Training Tips

### 1. Data Preprocessing
```python
# Normalize inputs
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
```

### 2. Early Stopping
```python
# Stop if validation loss doesn't improve
if val_loss > best_loss:
    patience -= 1
    if patience == 0:
        break
best_loss = min(best_loss, val_loss)
```

### 3. Learning Rate Scheduling
```python
# Decrease learning rate over time
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(100):
    # train...
    scheduler.step()
```

### 4. Regularization
- **L1/L2**: Penalize large weights
- **Dropout**: Randomly disable neurons
- **Batch Normalization**: Normalize activations

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Underfitting | Model too simple | Increase hidden units, epochs |
| Overfitting | Model too complex | Add dropout, L2 regularization |
| Vanishing Gradients | Gradients $\to$ 0 | Use ReLU, batch norm |
| Exploding Gradients | Gradients $\to \infty$ | Gradient clipping |

---

# Advanced Topics

## Modern Architectures

### ResNet (Residual Networks)
**Key Innovation**: Skip connections (residual connections)

```
x → Conv → ReLU → Conv → (+) → ReLU
                          ↑
                          x (skip connection)
```

**Benefits**:
- Enables training of very deep networks (100+ layers)
- Prevents vanishing gradients
- Easier optimization landscape

**Use Cases**: Image classification, feature extraction backbone

### Transformers
**Key Innovation**: Self-attention mechanism replaces recurrence

**Architecture**:
```
Input Embeddings + Positional Encoding
         ↓
Multi-Head Self-Attention
         ↓
Feed-Forward Network
         ↓
Output Predictions
```

**Attention Formula**: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

**Variants**:
- **BERT**: Bidirectional, masked language modeling
- **GPT**: Autoregressive, next-token prediction
- **T5**: Encoder-decoder, text-to-text framework
- **Vision Transformers (ViT)**: Apply to image patches

**Use Cases**: NLP, computer vision, multimodal AI

### Generative Adversarial Networks (GANs)
**Key Innovation**: Two networks compete in a game

**Architecture**:
- **Generator**: Creates fake samples from noise
- **Discriminator**: Distinguishes real from fake

```python
# Training loop
for epoch in epochs:
    # Train Discriminator
    real_loss = criterion(D(real_data), ones)
    fake_loss = criterion(D(G(noise)), zeros)
    d_loss = real_loss + fake_loss

    # Train Generator
    g_loss = criterion(D(G(noise)), ones)  # Fool discriminator
```

**Variants**:
- **DCGAN**: Convolutional architecture
- **StyleGAN**: High-quality image generation with style control
- **CycleGAN**: Unpaired image-to-image translation
- **Pix2Pix**: Paired image translation

**Use Cases**: Image generation, data augmentation, style transfer

### Diffusion Models
**Key Innovation**: Learn to denoise images through iterative refinement

**Forward Process**: Add noise gradually over T steps
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**Reverse Process**: Neural network learns to denoise
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Key Models**:
- **DDPM** (Denoising Diffusion Probabilistic Models): Foundation
- **DDIM**: Faster sampling with fewer steps
- **Stable Diffusion**: Text-to-image generation with latent diffusion
- **Imagen/DALL-E**: High-quality text-to-image synthesis

**Advantages**:
- More stable training than GANs
- High-quality generation
- Easy to condition on text, class labels, etc.

**Use Cases**: Image generation, inpainting, super-resolution, text-to-image

### State-Space Models
**Key Innovation**: Efficient sequence modeling with linear complexity

**Classical State-Space**:
$$h_t = Ah_{t-1} + Bx_t$$
$$y_t = Ch_t + Dx_t$$

**S4 (Structured State Spaces)**:
- Handles long sequences efficiently (10k+ tokens)
- Structured matrices for efficient computation
- Better than Transformers for very long contexts

**Mamba**:
- Selective state-space model
- Input-dependent state transitions
- 5x faster than Transformers for long sequences
- O(n) complexity vs O(n²) for attention

**Use Cases**: Long document processing, time series, genomics, audio

### Neural Radiance Fields (NeRF)
**Key Innovation**: Represent 3D scenes as continuous functions

**Architecture**:
- Input: 5D coordinates (x, y, z, θ, φ)
- Output: Color (RGB) and volume density (σ)
- Network: MLP with positional encoding

**Training**:
- Render images by ray marching through scene
- Minimize difference with actual photographs
- Learns implicit 3D representation

**Variants**:
- **Instant-NGP**: 1000x faster training with hash encoding
- **Mip-NeRF**: Anti-aliasing and multi-scale representation
- **NeRF-W**: Wild scenes with varying conditions

**Use Cases**: 3D reconstruction, novel view synthesis, virtual reality

### Graph Neural Networks (GNNs)
**Key Innovation**: Process graph-structured data

**Message Passing**:
$$h_v^{(k+1)} = \text{UPDATE}\left(h_v^{(k)}, \text{AGGREGATE}(\{h_u^{(k)} : u \in N(v)\})\right)$$

**Variants**:
- **GCN** (Graph Convolutional Networks)
- **GraphSAGE**: Scalable inductive learning
- **GAT** (Graph Attention Networks): Attention-based aggregation
- **GIN** (Graph Isomorphism Networks): Maximum discriminative power

**Use Cases**: Social networks, molecules, recommendation systems, knowledge graphs

## Advanced Training Techniques

### Modern Optimizers

#### AdamW (Adam with Weight Decay)
**Improvement over Adam**: Decouples weight decay from gradient updates

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01  # L2 regularization
)
```

**Why it's better**: Fixes weight decay implementation in Adam, improves generalization

#### Lion (Evolved Sign Momentum)
**Key Feature**: Memory-efficient, uses only sign of gradients

```python
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,  # Use ~3-10x smaller LR than Adam
    weight_decay=0.1
)
```

**Benefits**: 2x memory reduction, often matches or beats AdamW

#### Adafactor
**Key Feature**: Adaptive learning rates with reduced memory

**Benefits**:
- Reduces optimizer memory from O(parameters) to O(√parameters)
- Good for large language models
- No need to tune learning rate as carefully

#### Optimizer Comparison

| Optimizer | Memory | Speed | Stability | Best For |
|-----------|--------|-------|-----------|----------|
| SGD | Low | Fast | High | Simple tasks, fine-tuning |
| Adam | High | Fast | Medium | General purpose |
| AdamW | High | Fast | High | Most modern tasks |
| Lion | Medium | Fast | High | Large models, limited memory |
| Adafactor | Low | Medium | Medium | Very large models (LLMs) |

### Learning Rate Schedules

#### Warmup
**Purpose**: Prevent instability in early training

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
```

#### Cosine Annealing
**Purpose**: Smoothly decrease learning rate to fine-tune at end

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6  # Minimum LR
)

# With warmup
from torch.optim.lr_scheduler import SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[5]
)
```

#### One Cycle Policy
**Purpose**: Fast training with super-convergence

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)
```

### Mixed Precision Training

**Purpose**: Faster training, reduced memory (2x speedup typical)

#### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass in mixed precision
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**How it works**:
- FP16 for most operations (faster, less memory)
- FP32 for stability-critical operations
- Gradient scaling prevents underflow

**Typical speedup**: 1.5-3x on modern GPUs (A100, V100, RTX 30xx+)

#### BFloat16
**Alternative to FP16**: Same range as FP32, less precision

```python
model = model.to(torch.bfloat16)  # Convert model
# or
with autocast(dtype=torch.bfloat16):
    outputs = model(batch_x)
```

**When to use**: TPUs, newer GPUs (A100, H100), more stable than FP16

### Gradient Accumulation

**Purpose**: Simulate larger batch sizes without memory

```python
accumulation_steps = 4  # Effective batch = batch_size × 4

optimizer.zero_grad()
for i, (batch_x, batch_y) in enumerate(train_loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss = loss / accumulation_steps  # Normalize loss

    loss.backward()

    # Update only every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Trade-off**: More memory efficient, but slower training

### Gradient Clipping

**Purpose**: Prevent exploding gradients

```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**When to use**: RNNs, very deep networks, unstable training

### Transfer Learning & Fine-Tuning

#### Feature Extraction
**Strategy**: Freeze pretrained layers, train only new layers

```python
# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

#### Fine-Tuning
**Strategy**: Train entire model with small learning rate

```python
# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different learning rates for different layers
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.layer1.parameters(), 'lr': 1e-5}
])
```

#### Discriminative Learning Rates
**Strategy**: Lower layers learn slower (more general features)

```python
def get_layer_lr(layer_depth, base_lr=1e-3, decay=0.9):
    return base_lr * (decay ** layer_depth)

param_groups = []
for i, layer in enumerate(model.layers):
    param_groups.append({
        'params': layer.parameters(),
        'lr': get_layer_lr(i)
    })
optimizer = torch.optim.Adam(param_groups)
```

### Progressive Resizing

**Strategy**: Train on small images first, then larger

```python
# Start with 128x128
train_dataset = ImageDataset(transform=resize_to(128))
train_model(model, train_dataset, epochs=5)

# Increase to 224x224
train_dataset = ImageDataset(transform=resize_to(224))
train_model(model, train_dataset, epochs=5)

# Final size 384x384
train_dataset = ImageDataset(transform=resize_to(384))
train_model(model, train_dataset, epochs=5)
```

**Benefits**: Faster training, acts as regularization, better accuracy

## Practical Considerations

### Hardware & Performance

#### GPU Memory Management

**Check available memory**:
```python
import torch

print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

**Memory optimization strategies**:

| Technique | Memory Savings | Speed Impact | When to Use |
|-----------|---------------|--------------|-------------|
| Reduce batch size | High | Slower | Out of memory errors |
| Mixed precision (FP16) | 2x | Faster | Always (modern GPUs) |
| Gradient accumulation | High | Slower | Need large effective batch |
| Gradient checkpointing | High | 20-30% slower | Very deep networks |
| Model parallelism | Scales | Depends | Model > single GPU |

**Gradient Checkpointing**:
```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Recompute activations during backward pass
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

**When to use**: Very deep networks (ResNet-200+, GPT-3), trades compute for memory

#### Hardware Selection

| Use Case | Recommended GPU | VRAM Needed | Notes |
|----------|----------------|-------------|-------|
| Small models (<100M params) | RTX 3060, T4 | 6-8 GB | Good for learning |
| Medium models (100M-1B) | RTX 4090, A10 | 12-24 GB | Most research |
| Large models (1B-10B) | A100 (40GB) | 40-80 GB | Professional use |
| Very large (10B+) | A100 (80GB), H100 | 80+ GB | Multi-GPU required |

**Cloud options**:
- **AWS**: p3.2xlarge (V100), p4d.24xlarge (A100)
- **GCP**: a2-highgpu-1g (A100)
- **Azure**: NC-series (V100, A100)
- **Lambda Labs**, **RunPod**: Cost-effective alternatives

### Model Deployment

#### Export to ONNX
**Purpose**: Framework-agnostic format for inference

```python
import torch.onnx

# Export PyTorch model
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Load with ONNX Runtime (faster inference)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_array})
```

**Benefits**: 2-5x faster inference, works across frameworks

#### Quantization
**Purpose**: Reduce model size and inference time

**Post-Training Quantization** (easiest):
```python
import torch.quantization

# PyTorch dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

# Size reduction: 4x smaller, 2-3x faster
```

**Quantization-Aware Training** (best accuracy):
```python
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train as normal
for epoch in range(num_epochs):
    train_one_epoch(model_prepared)

# Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)
```

**Results**:
- **INT8**: 4x smaller, 2-4x faster, ~1% accuracy loss
- **INT4**: 8x smaller, 3-6x faster, ~2-3% accuracy loss

#### TensorRT Optimization
**Purpose**: Maximum inference speed on NVIDIA GPUs

```python
import torch_tensorrt

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float16}
)

# Save compiled model
torch.jit.save(trt_model, "model_trt.ts")
```

**Typical speedup**: 2-10x faster than native PyTorch

#### Model Serving

**TorchServe**:
```bash
# Create model archive
torch-model-archiver --model-name resnet \
    --version 1.0 \
    --serialized-file model.pt \
    --handler image_classifier

# Start server
torchserve --start --model-store model_store --models resnet=resnet.mar
```

**FastAPI** (simple custom server):
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("model.pt")

@app.post("/predict")
def predict(data: dict):
    input_tensor = preprocess(data)
    with torch.no_grad():
        output = model(input_tensor)
    return {"prediction": output.tolist()}
```

### Distributed Training

#### Data Parallelism (DDP)
**Strategy**: Replicate model on multiple GPUs, split data

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Wrap model
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler)

# Training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
    for batch in train_loader:
        # Training code...
```

**Launch**:
```bash
torchrun --nproc_per_node=4 train.py
```

**Scaling**: Near-linear speedup (4 GPUs → ~3.8x faster)

#### Fully Sharded Data Parallel (FSDP)
**Strategy**: Shard model parameters across GPUs

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=default_auto_wrap_policy,
    mixed_precision=MixedPrecision(param_dtype=torch.float16)
)
```

**When to use**: Model doesn't fit on single GPU (LLMs, very deep networks)

#### DeepSpeed
**Framework**: Advanced distributed training optimizations

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

for batch in train_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**ZeRO stages**:
- **ZeRO-1**: Shard optimizer states (4x memory reduction)
- **ZeRO-2**: + Shard gradients (8x reduction)
- **ZeRO-3**: + Shard parameters (linear scaling to 1000s of GPUs)

### Production Best Practices

#### Model Versioning
```python
# Save with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'hyperparameters': config,
    'timestamp': datetime.now()
}, f'model_v{version}.pt')
```

#### Monitoring & Logging
```python
import wandb  # or tensorboard

wandb.init(project="my-project")
wandb.config.update(hyperparameters)

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    })
```

#### Input Validation
```python
def validate_input(tensor):
    # Check shape
    assert tensor.shape[1:] == (3, 224, 224), f"Expected (3,224,224), got {tensor.shape[1:]}"

    # Check value range
    assert tensor.min() >= 0 and tensor.max() <= 1, "Input must be normalized to [0,1]"

    # Check for NaN/Inf
    assert not torch.isnan(tensor).any(), "Input contains NaN"
    assert not torch.isinf(tensor).any(), "Input contains Inf"
```

#### A/B Testing
```python
def predict_with_ab_test(input_data):
    model_version = random.choice(['v1', 'v2'], p=[0.9, 0.1])  # 90/10 split

    if model_version == 'v1':
        return model_v1(input_data), 'v1'
    else:
        return model_v2(input_data), 'v2'
```

## ELI10

Think of a neural network like learning to draw:

1. **Input Layer**: You see a cat
2. **Hidden Layers**: Brain recognizes ears -> whiskers -> tail (learns patterns)
3. **Output Layer**: Brain says "This is a cat!"

The network learns by:
- Making predictions (forward pass)
- Checking if wrong (loss)
- Adjusting "how to recognize cats" (backprop)
- Repeating until accurate

More hidden layers = learns more complex patterns!

---

# Quick Reference

## Common Patterns Cheat Sheet

### Model Initialization

```python
# PyTorch
model = MyModel().to(device)
model.load_state_dict(torch.load('model.pt'))  # Load weights
torch.save(model.state_dict(), 'model.pt')     # Save weights

# TensorFlow
model = MyModel()
model.load_weights('model.h5')     # Load weights
model.save_weights('model.h5')     # Save weights
model.save('full_model.keras')     # Save full model
```

### Training Loop Pattern

```python
# PyTorch standard training loop
model.train()  # Set to training mode
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()           # Clear gradients
    outputs = model(batch_x)        # Forward pass
    loss = criterion(outputs, batch_y)
    loss.backward()                 # Compute gradients
    optimizer.step()                # Update weights

# TensorFlow/Keras
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### Validation/Inference Pattern

```python
# PyTorch
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        # Compute metrics...

# TensorFlow
model.evaluate(test_dataset)
predictions = model.predict(test_data)
```

### Layer Types Quick Reference

| Task | Layer Type | PyTorch | TensorFlow |
|------|-----------|---------|------------|
| Fully Connected | Dense/Linear | `nn.Linear(in, out)` | `Dense(units)` |
| Convolution 2D | Conv | `nn.Conv2d(in, out, kernel)` | `Conv2D(filters, kernel)` |
| Max Pooling | Pooling | `nn.MaxPool2d(kernel)` | `MaxPool2D(pool_size)` |
| Dropout | Regularization | `nn.Dropout(p)` | `Dropout(rate)` |
| Batch Norm | Normalization | `nn.BatchNorm1d(features)` | `BatchNormalization()` |
| ReLU | Activation | `nn.ReLU()` | `Activation('relu')` |
| LSTM | Recurrent | `nn.LSTM(in, hidden)` | `LSTM(units)` |

### Loss Functions

| Task | Loss Function | PyTorch | TensorFlow |
|------|--------------|---------|------------|
| Binary Classification | BCE | `nn.BCELoss()` | `'binary_crossentropy'` |
| Multi-class | CrossEntropy | `nn.CrossEntropyLoss()` | `'categorical_crossentropy'` |
| Multi-class (sparse) | CrossEntropy | `nn.CrossEntropyLoss()` | `'sparse_categorical_crossentropy'` |
| Regression | MSE | `nn.MSELoss()` | `'mse'` |
| Regression | MAE | `nn.L1Loss()` | `'mae'` |

### Optimizer Selection Guide

```python
# Starting point for most tasks
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Fine-tuning pretrained models
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Memory constrained
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)

# Simple tasks, maximum control
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Learning Rate Schedule Pattern

```python
# Warmup + Cosine Annealing (recommended)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

# In training loop
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()  # Update learning rate
```

### Debugging Checklist

**Loss not decreasing?**
1. Check learning rate (try 1e-3, 1e-4, 1e-5)
2. Verify data preprocessing (normalization, scaling)
3. Check loss function matches task
4. Ensure labels are correct format
5. Try simpler model first

**Loss is NaN/Inf?**
1. Reduce learning rate
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check for division by zero
4. Use mixed precision carefully
5. Check input data for NaN/Inf values

**Overfitting (train acc >> val acc)?**
1. Add dropout: `nn.Dropout(0.3)`
2. Add weight decay: `weight_decay=0.01`
3. Reduce model size
4. Get more training data
5. Add data augmentation
6. Early stopping

**Underfitting (both accuracies low)?**
1. Increase model capacity (more layers/units)
2. Train longer
3. Reduce regularization
4. Check data quality
5. Try different architecture

### Common Hyperparameter Ranges

```python
hyperparameters = {
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],      # Most critical
    'batch_size': [16, 32, 64, 128, 256],            # Memory dependent
    'hidden_units': [64, 128, 256, 512, 1024],       # Task dependent
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],           # Regularization
    'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2],    # L2 regularization
    'epochs': [10, 20, 50, 100],                     # Use early stopping
}
```

### Data Preprocessing Template

```python
from torchvision import transforms

# Image preprocessing (standard)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])

# Numerical data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)  # Use same scaler!
```

### Model Architecture Rules of Thumb

| Guideline | Recommendation |
|-----------|---------------|
| **Depth** | Start with 2-3 hidden layers, add more if underfitting |
| **Width** | Hidden size typically 50-500% of input size |
| **Output layer** | No activation for regression, softmax for classification |
| **Hidden activation** | ReLU for most cases, GELU for transformers |
| **Batch norm** | After linear layer, before activation |
| **Dropout** | After activation, 0.3-0.5 typical |
| **Residual connections** | For networks >10 layers |

### GPU Memory Estimation

```
Memory = Model Params × (4 bytes if FP32, 2 bytes if FP16)
        + Activations × Batch Size × 4 bytes
        + Gradients ≈ 2 × Model Memory
        + Optimizer State ≈ 2 × Model Memory (Adam)

Total ≈ 4-6× Model Size for training (Adam, FP32)
Total ≈ 2-3× Model Size for training (Adam, FP16)
```

**Example**: 100M parameter model
- FP32: 400 MB model + overhead = ~2 GB training
- FP16: 200 MB model + overhead = ~1 GB training

### Performance Optimization Priority

1. **Mixed precision training** (FP16) - 2x speedup, 2x memory reduction
2. **Increase batch size** - Better GPU utilization
3. **Use DataLoader with num_workers** - Parallel data loading
4. **Pin memory** - `DataLoader(..., pin_memory=True)`
5. **Compile model** (PyTorch 2.0+) - `model = torch.compile(model)`
6. **Profile code** - Find bottlenecks before optimizing

## Further Resources

- [Neural Networks Visualization](https://playground.tensorflow.org/)
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/watch?v=aircAruvnKk)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
