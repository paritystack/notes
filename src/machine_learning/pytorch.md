# PyTorch

## Overview

PyTorch is a deep learning framework developed by Meta (Facebook) that provides:
- **Dynamic computation graphs**: Build networks on-the-fly (unlike static graphs in TensorFlow)
- **Pythonic API**: Natural, intuitive syntax for building neural networks
- **GPU acceleration**: Seamless CUDA support for fast training
- **Rich ecosystem**: Tools for NLP, computer vision, reinforcement learning
- **Production ready**: Deploy with TorchScript, ONNX, or mobile

## Installation

```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Core Concepts

### Tensors

Tensors are the fundamental building blocks - N-dimensional arrays:

```python
import torch

# Creating tensors
t1 = torch.tensor([1, 2, 3])           # From list
t2 = torch.zeros(3, 4)                 # Zeros tensor
t3 = torch.ones(2, 3)                  # Ones tensor
t4 = torch.randn(3, 4)                 # Random normal distribution
t5 = torch.arange(0, 10, 2)            # Range: [0, 2, 4, 6, 8]

# Tensor properties
print(t1.shape)                         # torch.Size([3])
print(t1.dtype)                         # torch.int64
print(t1.device)                        # cpu

# Move to GPU
if torch.cuda.is_available():
    t1 = t1.cuda()                      # or t1.to('cuda')
    print(t1.device)                    # cuda:0

# Tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

c = a + b                               # Element-wise addition
d = a * b                               # Element-wise multiplication
e = torch.dot(a, b)                     # Dot product: 32.0
f = torch.matmul(a.view(3, 1), b.view(1, 3))  # Matrix multiplication

# Reshaping
x = torch.randn(2, 3, 4)
y = x.view(6, 4)                        # Reshape to (6, 4)
z = x.reshape(-1)                       # Flatten (auto-infer dimension)
```

### Autograd (Automatic Differentiation)

PyTorch computes gradients automatically:

```python
import torch

# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 2.0], requires_grad=True)

# Forward pass
z = x.pow(2).sum() + (y * x).sum()      # z = x^2 + y*x

# Backward pass (compute gradients)
z.backward()

print(x.grad)                           # dz/dx
print(y.grad)                           # dz/dy

# Example: dz/dx = 2*x + y = [5, 8] for x=[2,3], y=[1,2]
```

### Neural Network Building

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input: 28*28=784, Output: 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)    # 10 output classes

    def forward(self, x):
        x = x.view(x.size(0), -1)       # Flatten: (batch, 784)
        x = F.relu(self.fc1(x))         # ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                 # No activation (raw logits)
        return x

# Create model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)

# Check model architecture
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

## Datasets and DataLoaders

### Custom Dataset

Create custom datasets by inheriting from `torch.utils.data.Dataset`:

```python
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: List or array of inputs
            labels: List or array of labels
            transform: Optional transformations to apply
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """Return sample at index idx"""
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Usage
X = torch.randn(1000, 28, 28)  # 1000 images of 28x28
y = torch.randint(0, 10, (1000,))  # 1000 labels (10 classes)

dataset = CustomDataset(X, y)
print(f"Dataset size: {len(dataset)}")
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}")
```

### Image Dataset with Transforms

```python
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create datasets
train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
test_dataset = ImageDataset(test_paths, test_labels, transform=test_transform)
```

### Built-in Datasets

PyTorch provides common datasets in `torchvision.datasets`:

```python
from torchvision import datasets, transforms

# MNIST
mnist_train = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

mnist_test = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# CIFAR-10
cifar10 = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# ImageNet (large, requires manual download)
imagenet = datasets.ImageNet(
    root='./data',
    split='train',
    transform=transforms.ToTensor()
)

# Print dataset info
print(f"Dataset size: {len(mnist_train)}")
sample, label = mnist_train[0]
print(f"Sample shape: {sample.shape}, Label: {label}")
```

### DataLoader

DataLoader handles batching, shuffling, and parallel loading:

```python
from torch.utils.data import DataLoader

# Create DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,              # Samples per batch
    shuffle=True,               # Shuffle order every epoch
    num_workers=4,              # Parallel workers for data loading
    pin_memory=True,            # Pin memory for faster GPU transfer
    drop_last=True              # Drop last incomplete batch
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,              # Don't shuffle test data
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Iterate through batches
for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print(f"  Input shape: {batch_x.shape}")  # (32, 1, 28, 28)
    print(f"  Labels shape: {batch_y.shape}")  # (32,)

    if batch_idx == 0:
        break
```

### Data Splits

```python
from torch.utils.data import random_split

# Original dataset
dataset = CustomDataset(X, y)

# Split into train (70%), val (15%), test (15%)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset,
    [train_size, val_size, test_size]
)

# Create loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
```

### Data Augmentation Strategies

```python
from torchvision import transforms

# For images
augmentation = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# For text (custom)
class TextAugmentation:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size

    def __call__(self, tokens):
        # Random dropout of tokens
        if torch.rand(1) > 0.5:
            mask = torch.rand(len(tokens)) > 0.1
            tokens = tokens[mask]
        return tokens

# Custom augmentation
class MixupAugmentation:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        """Mixup data augmentation"""
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        batch_size = batch_x.size(0)

        index = torch.randperm(batch_size)
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        mixed_y = lam * batch_y.float() + (1 - lam) * batch_y[index].float()

        return mixed_x, mixed_y
```

### DataLoader Performance Tips

```python
# Good configuration
loader = DataLoader(
    dataset,
    batch_size=64,              # Larger batches for efficiency
    shuffle=True,
    num_workers=4,              # Use multiple workers (2-4 per GPU)
    pin_memory=True,            # Pin to CPU memory for GPU transfer
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2           # Prefetch batches (2-4 recommended)
)

# Monitor data loading performance
import time

start = time.time()
for batch in loader:
    pass
elapsed = time.time() - start
print(f"Time to load {len(loader)} batches: {elapsed:.2f}s")

# If loading is slow:
# - Increase num_workers
# - Check disk speed (SSD vs HDD)
# - Use pin_memory=True
# - Reduce image resolution if possible
# - Use data compression
```

### Combining Datasets

```python
from torch.utils.data import ConcatDataset, Subset

# Concatenate multiple datasets
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

# Subset of dataset
indices = list(range(0, 100))  # First 100 samples
subset = Subset(dataset, indices)

# Weighted sampling (e.g., for imbalanced data)
from torch.utils.data import WeightedRandomSampler

weights = [1.0 if label == 0 else 10.0 for label in dataset.labels]
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler  # Use sampler instead of shuffle
)
```

## Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

# Create dataloader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Backward pass
        optimizer.zero_grad()           # Clear old gradients
        loss.backward()                 # Compute new gradients
        optimizer.step()                # Update parameters

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

## Convolutional Neural Networks

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input: (batch, 3, 32, 32) - 3 channels, 32x32 images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)               # (batch, 32, 32, 32)
        x = F.relu(x)
        x = self.pool(x)                # (batch, 32, 16, 16)

        # Conv block 2
        x = self.conv2(x)               # (batch, 64, 16, 16)
        x = F.relu(x)
        x = self.pool(x)                # (batch, 64, 8, 8)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)       # (batch, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)
```

## Recurrent Neural Networks

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Input shape: (batch, seq_len, input_size)
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) - final hidden state

        # Use last hidden state for classification
        last_hidden = h_n[-1]           # (batch, hidden_size)
        out = self.fc(last_hidden)      # (batch, output_size)
        return out

model = RNN(input_size=100, hidden_size=256, num_layers=2, output_size=10).to(device)
```

## Model Evaluation

```python
# Evaluation mode (disables dropout, batch norm uses running stats)
model.eval()

correct = 0
total = 0

with torch.no_grad():                   # Disable gradient computation
    for batch_x, batch_y in test_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        logits = model(batch_x)
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# Switch back to training mode
model.train()
```

## Saving and Loading Models

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleNet().to(device)
model.load_state_dict(torch.load('model.pth'))

# Save entire checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## Common Optimizers

```python
import torch.optim as optim

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (adaptive learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# In training loop:
for epoch in range(num_epochs):
    # ... training code ...
    scheduler.step()                    # Decay learning rate
```

## Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()       # Combines LogSoftmax + NLLLoss
criterion = nn.BCEWithLogitsLoss()      # Binary classification

# Regression
criterion = nn.MSELoss()                # Mean Squared Error
criterion = nn.L1Loss()                 # Mean Absolute Error
criterion = nn.SmoothL1Loss()           # Huber loss

# Custom loss
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        return (pred - target).pow(2).mean()
```

## Advanced Techniques

### Batch Normalization

```python
class BNNetwork(nn.Module):
    def __init__(self):
        super(BNNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Normalize features
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)                 # Normalize after linear layer
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()

    with autocast():                    # Automatically cast to float16 where safe
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Time Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| **Forward pass** | O(n * hidden_size) for dense layers |
| **Backward pass** | O(n * hidden_size) (2-3x forward) |
| **Conv2D** | O(H * W * C_in * K^2) per sample |
| **LSTM** | O(seq_len * hidden_size^2) per sample |

## Best Practices

1. **Use DataLoader** for batching and shuffling
2. **Track metrics** with tensorboard or wandb
3. **Use gradient clipping** for unstable training
4. **Normalize inputs** (mean=0, std=1)
5. **Monitor learning** - plot loss and metrics
6. **Save checkpoints** periodically during training
7. **Use model.eval()** during validation/testing
8. **Pin memory** for faster data loading: `DataLoader(..., pin_memory=True)`

## Common Issues

### Out of Memory
```python
# Solution 1: Reduce batch size
batch_size = 16  # Instead of 32

# Solution 2: Gradient accumulation
accumulation_steps = 4
for i, (batch_x, batch_y) in enumerate(dataloader):
    logits = model(batch_x)
    loss = criterion(logits, batch_y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### NaN Loss
- Learning rate too high
- Batch normalization issues
- Unstable loss function
- Check for gradient clipping

### Slow Training
- Use GPU (move model and data to CUDA)
- Increase batch size
- Use mixed precision training
- Profile with `torch.profiler`

## ELI10

PyTorch is like a smart building assistant:

1. **You design the blueprint** (define the network architecture)
2. **PyTorch remembers every step** (autograd tracks all operations)
3. **You show examples** (training data)
4. **PyTorch automatically learns** (backpropagation adjusts weights)
5. **It gets better each time** (more epochs = better performance)

It's like learning to cook - you follow the recipe, taste the result, adjust ingredients, and get better over time!

## Further Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Specialization with PyTorch](https://www.coursera.org/learn/neural-networks-deep-learning)
- [PyTorch Lightning](https://lightning.ai/) - High-level wrapper
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP with PyTorch
- [Fast.ai](https://course.fast.ai/) - Practical deep learning course
