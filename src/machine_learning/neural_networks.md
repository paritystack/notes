# Neural Networks

## Overview

A neural network is a machine learning model inspired by biological brains. It consists of interconnected nodes (neurons) organized in layers that learn patterns from data.

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

## Code Example (PyTorch)

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

## Modern Architectures

### ResNet (Residual Networks)
Skip connections prevent vanishing gradients in deep networks

### Attention Mechanisms
Query-Key-Value mechanism enables transformers

### Vision Transformers (ViT)
Apply transformer architecture to image patches

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

## Further Resources

- [Neural Networks Visualization](https://playground.tensorflow.org/)
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/watch?v=aircAruvnKk)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
