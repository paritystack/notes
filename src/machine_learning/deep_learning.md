# Deep Learning

Deep learning uses artificial neural networks with multiple layers to learn hierarchical representations of data.

## Table of Contents

1. [Neural Networks Fundamentals](#neural-networks-fundamentals)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Optimization](#optimization)
5. [Regularization](#regularization)
6. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks)
7. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks)
8. [Attention Mechanisms](#attention-mechanisms)
9. [Batch Normalization](#batch-normalization)
10. [Advanced Architectures](#advanced-architectures)

## Neural Networks Fundamentals

### Perceptron

The basic building block of neural networks.

**Mathematical Formulation:**
```
y = f(Σ(w_i * x_i) + b)
```

Where:
- x_i: inputs
- w_i: weights
- b: bias
- f: activation function

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple perceptron
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Example usage
input_dim = 4
model = Perceptron(input_dim)
x = torch.randn(32, input_dim)
output = model(x)
print(f"Output shape: {output.shape}")
```

### Multi-Layer Perceptron (MLP)

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example: 3-layer MLP
model = MLP(input_dim=10, hidden_dims=[64, 32], output_dim=2)
x = torch.randn(16, 10)
output = model(x)
print(f"Output shape: {output.shape}")
```

### Backpropagation

**Forward Pass:**
```
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])
```

**Backward Pass (Chain Rule):**
```
dL/dW^[l] = dL/da^[l] · da^[l]/dz^[l] · dz^[l]/dW^[l]
```

```python
# Manual backpropagation example
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
```

## Activation Functions

### Common Activation Functions

```python
import torch.nn.functional as F

# ReLU (Rectified Linear Unit)
def relu(x):
    return torch.max(torch.zeros_like(x), x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Tanh
def tanh(x):
    return torch.tanh(x)

# Softmax (for multi-class classification)
def softmax(x, dim=-1):
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

# GELU (Gaussian Error Linear Unit)
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Swish/SiLU
def swish(x):
    return x * torch.sigmoid(x)

# Visualization
x = torch.linspace(-5, 5, 100)
activations = {
    'ReLU': F.relu(x),
    'Leaky ReLU': F.leaky_relu(x, 0.1),
    'Sigmoid': torch.sigmoid(x),
    'Tanh': torch.tanh(x),
    'GELU': F.gelu(x),
    'Swish': x * torch.sigmoid(x)
}

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, y) in zip(axes.flatten(), activations.items()):
    ax.plot(x.numpy(), y.numpy())
    ax.set_title(name)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

## Loss Functions

### Classification Losses

```python
# Binary Cross-Entropy
def binary_cross_entropy(predictions, targets):
    return -torch.mean(targets * torch.log(predictions + 1e-8) + 
                      (1 - targets) * torch.log(1 - predictions + 1e-8))

# Categorical Cross-Entropy
def categorical_cross_entropy(predictions, targets):
    return -torch.mean(torch.sum(targets * torch.log(predictions + 1e-8), dim=1))

# Focal Loss (for imbalanced datasets)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Using PyTorch built-in losses
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()
criterion_nll = nn.NLLLoss()

# Example
predictions = torch.rand(32, 10)
targets = torch.randint(0, 10, (32,))
loss = criterion_ce(predictions, targets)
```

### Regression Losses

```python
# Mean Squared Error (MSE)
criterion_mse = nn.MSELoss()

# Mean Absolute Error (MAE)
criterion_mae = nn.L1Loss()

# Smooth L1 Loss (Huber Loss)
criterion_smooth = nn.SmoothL1Loss()

# Custom loss example
class CustomRegressionLoss(nn.Module):
    def __init__(self):
        super(CustomRegressionLoss, self).__init__()
    
    def forward(self, predictions, targets):
        mse = torch.mean((predictions - targets) ** 2)
        mae = torch.mean(torch.abs(predictions - targets))
        return mse + 0.1 * mae
```

## Optimization

### Optimizers

```python
# Stochastic Gradient Descent (SGD)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (Adaptive Moment Estimation)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Adagrad
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.01)

# Training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, OneCycleLR
)

# Step decay
scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential decay
scheduler_exp = ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Reduce on plateau
scheduler_plateau = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10, verbose=True
)

# One Cycle Policy
scheduler_onecycle = OneCycleLR(
    optimizer, max_lr=0.01, epochs=100, steps_per_epoch=len(train_loader)
)

# Usage in training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    # Step the scheduler
    scheduler_plateau.step(val_loss)  # For ReduceLROnPlateau
    # OR
    scheduler_step.step()  # For other schedulers
    
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
```

## Regularization

### Dropout

```python
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(MLPWithDropout, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Dropout variants
dropout = nn.Dropout(p=0.5)  # Standard dropout
dropout_2d = nn.Dropout2d(p=0.5)  # For Conv2d
dropout_3d = nn.Dropout3d(p=0.5)  # For Conv3d
alpha_dropout = nn.AlphaDropout(p=0.5)  # For SELU activation
```

### Weight Decay (L2 Regularization)

```python
# Weight decay in optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Manual L2 regularization
def l2_regularization(model, lambda_l2=0.01):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2)
    return lambda_l2 * l2_loss

# In training loop
loss = criterion(output, target) + l2_regularization(model)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Convolutional Neural Networks

CNNs are specialized for processing grid-like data (images, videos).

### Basic CNN Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Example usage
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)  # Batch of 4 RGB 32x32 images
output = model(x)
print(f"Output shape: {output.shape}")  # [4, 10]
```

### Modern CNN Architectures

#### ResNet (Residual Networks)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ResNet-18
model = ResNet([2, 2, 2, 2])
```

#### Inception Module

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU()
        )
        
        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
```

### Advanced CNN Techniques

```python
# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=kernel_size//2, groups=in_channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## Recurrent Neural Networks

RNNs process sequential data by maintaining hidden state.

### Basic RNN

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN output
        out, hn = self.rnn(x, h0)
        # out shape: (batch, seq_len, hidden_size)
        
        # Use last time step
        out = self.fc(out[:, -1, :])
        return out

# Example
model = SimpleRNN(input_size=10, hidden_size=64, output_size=2)
x = torch.randn(32, 20, 10)  # (batch, seq_len, features)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 2]
```

### LSTM (Long Short-Term Memory)

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last time step
        out = self.fc(out[:, -1, :])
        return out

# Bidirectional LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True
        )
        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

### GRU (Gated Recurrent Unit)

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Comparison
models = {
    'RNN': SimpleRNN(10, 64, 2),
    'LSTM': LSTMModel(10, 64, 2),
    'GRU': GRUModel(10, 64, 2)
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name} parameters: {params}")
```

## Attention Mechanisms

Attention allows models to focus on relevant parts of the input.

### Self-Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.embed_dim)
        
        # Final linear layer
        output = self.out(attention_output)
        return output, attention_weights

# Example
attention = SelfAttention(embed_dim=512, num_heads=8)
x = torch.randn(32, 10, 512)  # (batch, seq_len, embed_dim)
output, weights = attention(x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Batch Normalization

Normalizes layer inputs to improve training.

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Other normalization techniques
# Layer Normalization (better for RNNs/Transformers)
layer_norm = nn.LayerNorm(normalized_shape=[128])

# Group Normalization
group_norm = nn.GroupNorm(num_groups=8, num_channels=64)

# Instance Normalization (used in style transfer)
instance_norm = nn.InstanceNorm2d(num_features=64)
```

## Advanced Architectures

### Vision Transformer (ViT)

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # Use cls token
        x = self.head(x)
        
        return x
```

## Practical Tips

1. **Initialize Weights Properly**: Use Xavier/He initialization
2. **Monitor Gradients**: Check for vanishing/exploding gradients
3. **Use Mixed Precision Training**: Faster training with similar accuracy
4. **Data Augmentation**: Improves generalization
5. **Gradient Accumulation**: Train with larger effective batch sizes
6. **Model Checkpointing**: Save best models during training

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Resources

- "Deep Learning" by Goodfellow, Bengio, and Courville
- PyTorch Documentation: https://pytorch.org/docs/
- TensorFlow Documentation: https://www.tensorflow.org/
- Papers with Code: https://paperswithcode.com/

