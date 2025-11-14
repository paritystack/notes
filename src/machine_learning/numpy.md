# NumPy for Machine Learning

NumPy is the foundational numerical computing library for Python and forms the backbone of the ML/AI ecosystem. Understanding NumPy deeply is essential for efficient machine learning implementations.

## Table of Contents
- [Why NumPy for ML](#why-numpy-for-ml)
- [Array Creation Patterns](#array-creation-patterns)
- [Indexing and Slicing](#indexing-and-slicing)
- [Broadcasting](#broadcasting)
- [Vectorization](#vectorization)
- [Reshaping and Transformations](#reshaping-and-transformations)
- [Matrix Operations](#matrix-operations)
- [Statistical Operations](#statistical-operations)
- [Linear Algebra](#linear-algebra)
- [Random Number Generation](#random-number-generation)
- [Advanced Patterns](#advanced-patterns)
- [Performance Optimization](#performance-optimization)
- [Common ML Patterns](#common-ml-patterns)

---

## Why NumPy for ML

**Speed**: NumPy operations are implemented in C and are vectorized, making them 10-100x faster than pure Python loops.

**Memory Efficiency**: Contiguous memory layout and fixed data types reduce overhead.

**Foundation**: PyTorch, TensorFlow, and scikit-learn all build on NumPy conventions.

**Broadcasting**: Implicit expansion of arrays enables concise, efficient code.

```python
import numpy as np

# Pure Python (slow)
result = []
for i in range(1000000):
    result.append(i ** 2)

# NumPy (fast)
result = np.arange(1000000) ** 2
```

---

## Array Creation Patterns

### Basic Creation

```python
# From lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Specify dtype for memory efficiency
arr_int8 = np.array([1, 2, 3], dtype=np.int8)      # 1 byte per element
arr_float32 = np.array([1, 2, 3], dtype=np.float32) # 4 bytes per element
arr_float64 = np.array([1, 2, 3], dtype=np.float64) # 8 bytes per element (default)
```

### Initialization Patterns for ML

```python
# Zeros - common for initializing gradients or counts
zeros = np.zeros((3, 4))
zeros_like = np.zeros_like(existing_array)

# Ones - useful for bias initialization
ones = np.ones((3, 4))
ones_like = np.ones_like(existing_array)

# Empty - fastest, doesn't initialize (use when you'll overwrite)
empty = np.empty((3, 4))

# Full - initialize with specific value
full = np.full((3, 4), 0.01)  # Initialize all to 0.01

# Identity matrix - common in linear algebra
identity = np.eye(5)
identity_offset = np.eye(5, k=1)  # Offset diagonal

# Ranges
arange = np.arange(0, 10, 2)        # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]
logspace = np.logspace(0, 2, 5)     # [1, 10, 100] logarithmically spaced

# Meshgrid - useful for coordinate generation
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)  # Create 2D coordinate grids
```

### Random Initialization (Modern API)

```python
# Modern way (NumPy 1.17+)
rng = np.random.default_rng(seed=42)

# Uniform distribution [0, 1)
uniform = rng.random((3, 4))

# Normal/Gaussian distribution
normal = rng.normal(loc=0, scale=1, size=(3, 4))

# Xavier/Glorot initialization for neural networks
n_in, n_out = 784, 256
xavier = rng.normal(0, np.sqrt(2 / (n_in + n_out)), (n_in, n_out))

# He initialization (for ReLU networks)
he = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))

# Integer random values
randint = rng.integers(0, 10, size=(3, 4))

# Choice (sampling)
choices = rng.choice([1, 2, 3, 4, 5], size=10, replace=True)
```

---

## Indexing and Slicing

### Basic Indexing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Single element
arr[0]        # 0
arr[-1]       # 9

# Slicing: [start:stop:step]
arr[2:5]      # [2, 3, 4]
arr[::2]      # [0, 2, 4, 6, 8] - every second element
arr[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - reverse
arr[5:]       # [5, 6, 7, 8, 9]
arr[:5]       # [0, 1, 2, 3, 4]
```

### Multidimensional Indexing

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Element access
matrix[0, 0]           # 1
matrix[1, 2]           # 6

# Row and column slicing
matrix[0, :]           # [1, 2, 3] - first row
matrix[:, 0]           # [1, 4, 7] - first column
matrix[:2, :2]         # [[1, 2], [4, 5]] - top-left 2x2

# Stride tricks
matrix[::2, ::2]       # Every other row and column
```

### Boolean Indexing (Critical for ML)

```python
arr = np.array([1, -2, 3, -4, 5, -6])

# Boolean mask
mask = arr > 0
positive = arr[mask]   # [1, 3, 5]

# Inline
positive = arr[arr > 0]
even = arr[arr % 2 == 0]

# Compound conditions
arr[(arr > 0) & (arr < 4)]  # [1, 3]
arr[(arr < 0) | (arr > 4)]  # [-2, -4, 5, -6]

# Filtering outliers
data = np.random.randn(1000)
mean, std = data.mean(), data.std()
filtered = data[np.abs(data - mean) < 2 * std]  # Remove outliers beyond 2 sigma

# Setting values with boolean indexing
arr[arr < 0] = 0  # Clip negative values to 0 (ReLU activation!)
```

### Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Index with array of integers
indices = np.array([0, 2, 4])
arr[indices]  # [10, 30, 50]

# Multidimensional fancy indexing
matrix = np.arange(12).reshape(3, 4)
rows = np.array([0, 2, 2])
cols = np.array([1, 3, 0])
matrix[rows, cols]  # Elements at (0,1), (2,3), (2,0)

# Batch indexing (common in ML)
batch = np.random.randn(32, 10)  # 32 samples, 10 classes
labels = np.array([3, 1, 5, ...])  # True class for each sample
selected_logits = batch[np.arange(32), labels]  # Logits for true classes
```

### Advanced Slicing Tricks

```python
# Ellipsis (...) - all remaining dimensions
tensor = np.random.randn(2, 3, 4, 5)
tensor[0, ...]        # Same as tensor[0, :, :, :]
tensor[..., 0]        # Same as tensor[:, :, :, 0]

# np.newaxis or None - add dimension
arr = np.array([1, 2, 3])
arr[:, np.newaxis]    # Shape (3, 1) - column vector
arr[np.newaxis, :]    # Shape (1, 3) - row vector
```

---

## Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes efficiently without copying data.

### Broadcasting Rules

1. If arrays have different dimensions, pad the smaller shape with ones on the left
2. Arrays are compatible if, for each dimension, the sizes are equal or one of them is 1
3. After broadcasting, each dimension becomes the maximum of the two

```python
# Rule visualization
A:     (3, 4, 5)
B:        (1, 5)
Result:(3, 4, 5)

A:     (3, 1, 5)
B:     (3, 4, 1)
Result:(3, 4, 5)
```

### Common Broadcasting Patterns

```python
# Scalar with array
arr = np.array([1, 2, 3, 4])
arr * 2  # [2, 4, 6, 8]

# 1D with 2D (very common in ML)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row_vector = np.array([10, 20, 30])
matrix + row_vector
# [[11, 22, 33],
#  [14, 25, 36]]

# Broadcasting for normalization
data = np.random.randn(100, 5)  # 100 samples, 5 features
mean = data.mean(axis=0)        # Shape (5,)
std = data.std(axis=0)          # Shape (5,)
normalized = (data - mean) / std  # Broadcasting happens automatically

# Column vector broadcasting
col_vector = np.array([[1], [2], [3]])  # Shape (3, 1)
row_vector = np.array([10, 20, 30])     # Shape (3,)
result = col_vector + row_vector
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]
```

### Practical ML Examples

```python
# Batch normalization
batch = np.random.randn(32, 64, 64, 3)  # 32 images, 64x64, 3 channels
mean = batch.mean(axis=(0, 1, 2), keepdims=True)  # Shape (1, 1, 1, 3)
std = batch.std(axis=(0, 1, 2), keepdims=True)
normalized_batch = (batch - mean) / (std + 1e-8)

# Distance matrix computation
X = np.random.randn(100, 50)  # 100 samples, 50 features
# Pairwise squared distances using broadcasting
X_expanded = X[:, np.newaxis, :]   # Shape (100, 1, 50)
X2_expanded = X[np.newaxis, :, :]  # Shape (1, 100, 50)
distances = np.sum((X_expanded - X2_expanded) ** 2, axis=2)  # (100, 100)

# Attention mechanism (simplified)
Q = np.random.randn(10, 64)  # 10 queries, 64 dims
K = np.random.randn(20, 64)  # 20 keys, 64 dims
# Compute attention scores
scores = Q @ K.T  # (10, 20)
```

### Broadcasting Pitfalls

```python
# Unintended broadcasting
a = np.random.randn(3, 1)
b = np.random.randn(4, 1)
# a + b raises error - shapes (3,1) and (4,1) incompatible

# Accidental dimension loss
a = np.random.randn(5, 1)
b = a.flatten()  # Shape (5,) not (5, 1)
# Now b broadcasts differently!

# Always check shapes
assert a.shape == expected_shape, f"Shape mismatch: {a.shape}"
```

---

## Vectorization

Vectorization is the process of replacing explicit loops with array operations. It's fundamental to writing efficient NumPy code.

### Why Vectorization Matters

```python
import time

# Non-vectorized
data = list(range(1000000))
start = time.time()
result = [x ** 2 for x in data]
print(f"Loop: {time.time() - start:.4f}s")

# Vectorized
data = np.arange(1000000)
start = time.time()
result = data ** 2
print(f"Vectorized: {time.time() - start:.4f}s")
# Typically 50-100x faster!
```

### Basic Vectorization Patterns

```python
# Element-wise operations (automatically vectorized)
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b         # [11, 22, 33, 44]
a * b         # [10, 40, 90, 160]
a ** b        # [1, 1048576, ...]
np.sin(a)     # [0.841, 0.909, 0.141, -0.757]
np.exp(a)     # [2.718, 7.389, 20.085, 54.598]

# Comparison operators
a > 2         # [False, False, True, True]
np.maximum(a, 2)  # [2, 2, 3, 4] - element-wise max
```

### Replacing Loops with Vectorization

```python
# Example 1: Sigmoid activation
def sigmoid_loop(x):
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = 1 / (1 + np.exp(-x[i]))
    return result

def sigmoid_vectorized(x):
    return 1 / (1 + np.exp(-x))

# Example 2: Pairwise distances
def distances_loop(X, Y):
    n, m = len(X), len(Y)
    distances = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distances[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return distances

def distances_vectorized(X, Y):
    # Using broadcasting
    return np.sqrt(np.sum((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2, axis=2))

# Example 3: Moving average
def moving_average_loop(arr, window):
    result = np.zeros(len(arr) - window + 1)
    for i in range(len(result)):
        result[i] = arr[i:i+window].mean()
    return result

def moving_average_vectorized(arr, window):
    # Using convolution
    return np.convolve(arr, np.ones(window)/window, mode='valid')
```

### Advanced Vectorization

```python
# Conditional operations - use np.where instead of if/else
x = np.random.randn(100)
# Bad
result = np.zeros_like(x)
for i in range(len(x)):
    result[i] = x[i] if x[i] > 0 else 0

# Good - vectorized ReLU
result = np.where(x > 0, x, 0)
# Even better
result = np.maximum(x, 0)

# Multiple conditions - use np.select
x = np.arange(-5, 6)
conditions = [x < -2, (x >= -2) & (x <= 2), x > 2]
choices = [-1, 0, 1]
result = np.select(conditions, choices)

# Vectorized gradient clipping
gradients = np.random.randn(1000, 100)
clip_value = 1.0
norm = np.linalg.norm(gradients, axis=1, keepdims=True)
gradients = np.where(norm > clip_value,
                     gradients * clip_value / norm,
                     gradients)
```

---

## Reshaping and Transformations

### Basic Reshaping

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape - returns view if possible
arr.reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

arr.reshape(2, 6)
arr.reshape(2, 2, 3)  # 3D array

# Infer dimension with -1
arr.reshape(3, -1)    # NumPy calculates: (3, 4)
arr.reshape(-1, 2)    # (6, 2)
arr.reshape(-1)       # Flatten to 1D

# Reshape and transpose in one go
arr.reshape(3, 4, order='F')  # Fortran-style (column-major)
```

### Flatten vs Ravel vs Reshape

```python
matrix = np.array([[1, 2], [3, 4]])

# flatten() - always returns a copy
flat1 = matrix.flatten()
flat1[0] = 999
# matrix unchanged

# ravel() - returns view if possible (more efficient)
flat2 = matrix.ravel()
flat2[0] = 999
# matrix[0, 0] is now 999!

# reshape(-1) - same as ravel
flat3 = matrix.reshape(-1)
```

### Transposition and Axis Manipulation

```python
# 2D transpose
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# Multi-dimensional transpose
tensor = np.random.randn(2, 3, 4)
tensor.transpose(2, 0, 1)  # Move axes: (4, 2, 3)
tensor.transpose()         # Reverse all axes: (4, 3, 2)

# Swapaxes - swap two specific axes
tensor.swapaxes(0, 2)  # Swap first and last: (4, 3, 2)

# moveaxis - more intuitive for single axis moves
tensor_moved = np.moveaxis(tensor, 0, -1)  # Move first axis to last: (3, 4, 2)
```

### Dimension Manipulation

```python
arr = np.array([1, 2, 3])

# Add dimensions
arr_col = arr[:, np.newaxis]      # Shape (3, 1)
arr_row = arr[np.newaxis, :]      # Shape (1, 3)
arr_3d = arr[:, np.newaxis, np.newaxis]  # Shape (3, 1, 1)

# Using expand_dims
arr_col = np.expand_dims(arr, axis=1)  # Shape (3, 1)
arr_3d = np.expand_dims(arr, axis=(1, 2))  # Shape (3, 1, 1)

# Remove dimensions - squeeze
arr_squeezed = np.squeeze(arr_col)  # Back to (3,)

# Broadcast to specific shape
arr_broadcast = np.broadcast_to(arr[:, np.newaxis], (3, 5))
# [[1, 1, 1, 1, 1],
#  [2, 2, 2, 2, 2],
#  [3, 3, 3, 3, 3]]
```

### Stacking and Splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack (along axis 0)
np.vstack([a, b])
# [[1, 2, 3],
#  [4, 5, 6]]

# Horizontal stack (along axis 1)
np.hstack([a, b])
# [1, 2, 3, 4, 5, 6]

# Stack along new axis
np.stack([a, b], axis=0)  # Shape (2, 3)
np.stack([a, b], axis=1)  # Shape (3, 2)

# Concatenate - general stacking
np.concatenate([a, b], axis=0)

# Splitting
arr = np.arange(9)
np.split(arr, 3)  # [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
np.array_split(arr, 4)  # Unequal splits allowed

# 2D splitting
matrix = np.random.randn(6, 4)
np.vsplit(matrix, 3)  # Split into 3 horizontal slices
np.hsplit(matrix, 2)  # Split into 2 vertical slices
```

### Practical ML Reshaping Examples

```python
# Batch flattening for fully connected layer
batch_images = np.random.randn(32, 28, 28, 1)  # 32 MNIST images
flattened = batch_images.reshape(32, -1)        # (32, 784)

# Channel manipulation (NHWC to NCHW)
nhwc = np.random.randn(10, 224, 224, 3)
nchw = nhwc.transpose(0, 3, 1, 2)  # (10, 3, 224, 224)

# Reshape for sequence processing
time_series = np.random.randn(1000, 10)  # 1000 timesteps, 10 features
batched = time_series.reshape(-1, 50, 10)  # 20 sequences of 50 timesteps

# Tile for data augmentation
pattern = np.array([1, 2, 3])
tiled = np.tile(pattern, 5)  # [1, 2, 3, 1, 2, 3, ...]
tiled_2d = np.tile(pattern, (3, 1))  # Repeat as rows
```

---

## Matrix Operations

### Element-wise vs Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication (Hadamard product)
A * B
# [[ 5, 12],
#  [21, 32]]

# Matrix multiplication
A @ B  # Python 3.5+ operator
np.dot(A, B)
np.matmul(A, B)
# [[19, 22],
#  [43, 50]]
```

### Different Multiplication Operations

```python
# 1D arrays - dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# 2D matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # Shape (3, 5)

# Batch matrix multiplication
batch_A = np.random.randn(10, 3, 4)
batch_B = np.random.randn(10, 4, 5)
batch_C = batch_A @ batch_B  # Shape (10, 3, 5)

# Outer product
a = np.array([1, 2, 3])
b = np.array([4, 5])
np.outer(a, b)
# [[ 4,  5],
#  [ 8, 10],
#  [12, 15]]

# Inner product (same as dot for 1D)
np.inner(a, b)  # Only if same length

# Kronecker product
np.kron(A, B)  # Tensor product
```

### Matrix Properties

```python
A = np.array([[1, 2], [3, 4]])

# Trace (sum of diagonal)
np.trace(A)  # 1 + 4 = 5

# Determinant
np.linalg.det(A)  # -2.0

# Rank
np.linalg.matrix_rank(A)  # 2

# Norm
np.linalg.norm(A)           # Frobenius norm (default)
np.linalg.norm(A, 'fro')    # Frobenius norm
np.linalg.norm(A, 2)        # Spectral norm
np.linalg.norm(A, 'nuc')    # Nuclear norm

# Condition number
np.linalg.cond(A)  # Ratio of largest to smallest singular value
```

### Advanced Matrix Operations

```python
# Matrix power
A = np.array([[1, 2], [3, 4]])
np.linalg.matrix_power(A, 3)  # A @ A @ A

# Matrix exponential (important in physics, ODEs)
from scipy.linalg import expm
expm(A)

# Batch operations
batch = np.random.randn(100, 10, 10)
# Batch determinant
dets = np.linalg.det(batch)  # Shape (100,)

# Einsum for complex operations (see Advanced Patterns)
# Batch matrix trace
traces = np.einsum('bii->b', batch)
```

---

## Statistical Operations

### Basic Statistics

```python
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Central tendency
np.mean(data)          # Overall mean
np.median(data)        # Median
np.percentile(data, 50)  # Same as median
np.quantile(data, 0.5)   # Same as median

# Spread
np.std(data)           # Standard deviation
np.var(data)           # Variance
np.ptp(data)           # Peak to peak (max - min)

# Extremes
np.min(data)
np.max(data)
np.argmin(data)  # Index of minimum
np.argmax(data)  # Index of maximum
```

### Axis-wise Operations

```python
data = np.random.randn(100, 5)

# Along columns (across samples)
feature_means = np.mean(data, axis=0)  # Shape (5,)
feature_stds = np.std(data, axis=0)

# Along rows (across features)
sample_means = np.mean(data, axis=1)   # Shape (100,)

# Keep dimensions for broadcasting
feature_means = np.mean(data, axis=0, keepdims=True)  # Shape (1, 5)
normalized = (data - feature_means) / np.std(data, axis=0, keepdims=True)

# Multiple axes
tensor = np.random.randn(10, 20, 30, 40)
mean_spatial = np.mean(tensor, axis=(1, 2))  # Average over dimensions 1 and 2
```

### Normalization Techniques

```python
# Z-score normalization (standardization)
def standardize(X, axis=0):
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + 1e-8)

# Min-max normalization
def min_max_normalize(X, axis=0):
    min_val = np.min(X, axis=axis, keepdims=True)
    max_val = np.max(X, axis=axis, keepdims=True)
    return (X - min_val) / (max_val - min_val + 1e-8)

# L2 normalization (unit vectors)
def l2_normalize(X, axis=1):
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / (norm + 1e-8)

# Batch normalization (simplified)
def batch_norm(X, gamma=1, beta=0, epsilon=1e-8):
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return gamma * X_norm + beta

# Whitening (decorrelation)
def whiten(X):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    U, S, Vt = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-8)) @ U.T
    return X_centered @ W
```

### Statistical Functions

```python
# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
np.cumsum(arr)     # [ 1,  3,  6, 10, 15]
np.cumprod(arr)    # [ 1,  2,  6, 24, 120]

# Correlation and covariance
data = np.random.randn(100, 5)
np.corrcoef(data, rowvar=False)  # Correlation matrix (5, 5)
np.cov(data, rowvar=False)       # Covariance matrix (5, 5)

# Histogram
values, bins = np.histogram(data, bins=10)
values, bins = np.histogram(data, bins='auto')  # Automatic binning

# Percentiles and quantiles
np.percentile(data, [25, 50, 75])  # Quartiles
np.quantile(data, [0.25, 0.5, 0.75])

# Binning
digitized = np.digitize(data, bins=[-1, 0, 1])  # Classify into bins

# Weighted statistics
weights = np.random.rand(100)
np.average(data, weights=weights, axis=0)
```

---

## Linear Algebra

### Matrix Decompositions

```python
# Eigenvalue decomposition
A = np.random.randn(5, 5)
A = A + A.T  # Make symmetric
eigenvalues, eigenvectors = np.linalg.eig(A)

# For symmetric matrices (faster and more stable)
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Singular Value Decomposition (SVD)
M = np.random.randn(10, 5)
U, S, Vt = np.linalg.svd(M, full_matrices=False)
# M ≈ U @ np.diag(S) @ Vt
# U: (10, 5), S: (5,), Vt: (5, 5)

# QR decomposition
Q, R = np.linalg.qr(M)
# M = Q @ R, Q is orthogonal, R is upper triangular

# Cholesky decomposition (for positive definite matrices)
A = np.random.randn(5, 5)
A = A.T @ A  # Make positive definite
L = np.linalg.cholesky(A)
# A = L @ L.T, L is lower triangular

# LU decomposition (requires scipy)
from scipy.linalg import lu
P, L, U = lu(A)
```

### Solving Linear Systems

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # x = [2, 3]

# Least squares solution (when system is overdetermined)
# Solve ||Ax - b||^2
A = np.random.randn(100, 5)
b = np.random.randn(100)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Matrix inverse
A_inv = np.linalg.inv(A)
# But prefer solving instead: x = np.linalg.solve(A, b)
# rather than: x = np.linalg.inv(A) @ b

# Pseudo-inverse (Moore-Penrose)
A = np.random.randn(10, 5)
A_pinv = np.linalg.pinv(A)
```

### Matrix Factorizations for ML

```python
# PCA using SVD
def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project onto top components
    components = Vt[:n_components]
    X_pca = X_centered @ components.T

    # Explained variance
    explained_variance = (S ** 2) / (len(X) - 1)
    explained_variance_ratio = explained_variance[:n_components] / explained_variance.sum()

    return X_pca, components, explained_variance_ratio

# Low-rank approximation
M = np.random.randn(100, 50)
U, S, Vt = np.linalg.svd(M, full_matrices=False)
k = 10  # Keep top 10 components
M_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Power iteration for top eigenvector
def power_iteration(A, num_iterations=100):
    v = np.random.randn(A.shape[1])
    for _ in range(num_iterations):
        v = A @ v
        v = v / np.linalg.norm(v)
    eigenvalue = v @ A @ v
    return eigenvalue, v
```

---

## Random Number Generation

### Modern Random API

```python
# Create generator with seed
rng = np.random.default_rng(42)

# Uniform distributions
rng.random((3, 4))               # [0, 1)
rng.uniform(0, 10, size=(3, 4))  # [0, 10)
rng.integers(0, 100, size=10)    # [0, 100)

# Normal/Gaussian
rng.normal(loc=0, scale=1, size=(3, 4))
rng.standard_normal((3, 4))  # mean=0, std=1

# Other distributions
rng.exponential(scale=1.0, size=100)
rng.poisson(lam=5, size=100)
rng.binomial(n=10, p=0.5, size=100)
rng.beta(a=2, b=5, size=100)
rng.gamma(shape=2, scale=1, size=100)
rng.multinomial(n=10, pvals=[0.2, 0.3, 0.5], size=20)
```

### Sampling and Shuffling

```python
rng = np.random.default_rng(42)

# Random choice
data = np.arange(100)
sample = rng.choice(data, size=10, replace=False)  # Without replacement

# Weighted sampling
weights = np.array([0.1, 0.2, 0.3, 0.4])
samples = rng.choice(4, size=1000, p=weights)

# Shuffle
arr = np.arange(10)
rng.shuffle(arr)  # In-place shuffle

# Permutation (returns shuffled copy)
perm = rng.permutation(arr)
perm_indices = rng.permutation(len(arr))

# Random partitioning for train/test split
indices = rng.permutation(len(data))
train_size = int(0.8 * len(data))
train_indices = indices[:train_size]
test_indices = indices[train_size:]
```

### Reproducibility

```python
# Global seed (legacy, not recommended)
np.random.seed(42)

# Better: use Generator instances
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
# rng1 and rng2 produce identical sequences

# Independent streams
from numpy.random import SeedSequence, Generator, PCG64

ss = SeedSequence(12345)
child_seeds = ss.spawn(10)  # Create 10 independent streams
streams = [Generator(PCG64(s)) for s in child_seeds]

# Each stream is independent
samples = [stream.random(100) for stream in streams]

# Save and restore state
state = rng.bit_generator.state
# ... later ...
rng.bit_generator.state = state  # Restore exact state
```

### Initialization Strategies for Neural Networks

```python
rng = np.random.default_rng(42)

def init_weights(shape, method='xavier', rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_in, n_out = shape

    if method == 'xavier' or method == 'glorot':
        # Xavier/Glorot initialization (for tanh, sigmoid)
        limit = np.sqrt(6 / (n_in + n_out))
        return rng.uniform(-limit, limit, shape)

    elif method == 'he':
        # He initialization (for ReLU)
        std = np.sqrt(2 / n_in)
        return rng.normal(0, std, shape)

    elif method == 'lecun':
        # LeCun initialization
        std = np.sqrt(1 / n_in)
        return rng.normal(0, std, shape)

    elif method == 'orthogonal':
        # Orthogonal initialization
        flat_shape = (n_in, n_out)
        a = rng.normal(0, 1, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q

    else:
        return rng.normal(0, 0.01, shape)

# Dropout mask
def dropout_mask(shape, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(shape) > p
    return mask / (1 - p)  # Inverted dropout

# Data augmentation noise
def add_gaussian_noise(X, std=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, std, X.shape)
    return X + noise
```

---

## Advanced Patterns

### Einstein Summation (einsum)

Einstein summation is a compact notation for array operations. It's extremely powerful once you understand it.

```python
# Basics
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)

# Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
c = np.einsum('ij,jk->ik', a, b)
# Same as: a @ b

# Trace: sum_i A[i,i]
A = np.random.randn(5, 5)
trace = np.einsum('ii->', A)
# Same as: np.trace(A)

# Diagonal: D[i] = A[i,i]
diag = np.einsum('ii->i', A)
# Same as: np.diag(A)

# Transpose: B[j,i] = A[i,j]
b = np.einsum('ij->ji', a)
# Same as: a.T

# Batch matrix multiplication
batch_a = np.random.randn(10, 3, 4)
batch_b = np.random.randn(10, 4, 5)
batch_c = np.einsum('bij,bjk->bik', batch_a, batch_b)
# Same as: batch_a @ batch_b

# Dot product: sum_i a[i] * b[i]
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.einsum('i,i->', a, b)
# Same as: np.dot(a, b)

# Outer product: C[i,j] = a[i] * b[j]
outer = np.einsum('i,j->ij', a, b)
# Same as: np.outer(a, b)

# Element-wise multiplication and sum: sum_ij A[i,j] * B[i,j]
A = np.random.randn(3, 4)
B = np.random.randn(3, 4)
result = np.einsum('ij,ij->', A, B)
# Same as: np.sum(A * B)
```

### Complex einsum Examples for ML

```python
# Attention mechanism
Q = np.random.randn(10, 8, 64)  # batch, query_len, dim
K = np.random.randn(10, 12, 64)  # batch, key_len, dim
V = np.random.randn(10, 12, 64)  # batch, value_len, dim

# Compute attention scores: scores[b,i,j] = sum_d Q[b,i,d] * K[b,j,d]
scores = np.einsum('bid,bjd->bij', Q, K) / np.sqrt(64)

# Apply attention to values: output[b,i,d] = sum_j scores[b,i,j] * V[b,j,d]
attention_weights = softmax(scores, axis=-1)
output = np.einsum('bij,bjd->bid', attention_weights, V)

# Bilinear operation: y[b] = sum_ij x1[b,i] * W[i,j] * x2[b,j]
x1 = np.random.randn(32, 10)
x2 = np.random.randn(32, 20)
W = np.random.randn(10, 20)
y = np.einsum('bi,ij,bj->b', x1, W, x2)

# Batch trace
batch = np.random.randn(100, 5, 5)
traces = np.einsum('bii->b', batch)

# Frobenius norm squared
frob_sq = np.einsum('ij,ij->', A, A)
```

### Universal Functions (ufuncs)

```python
# Create custom ufunc
def relu_scalar(x):
    return max(0, x)

relu = np.frompyfunc(relu_scalar, 1, 1)  # 1 input, 1 output
# Note: This is for educational purposes; use np.maximum(x, 0) in practice

# Accumulate methods
arr = np.array([1, 2, 3, 4, 5])
np.add.accumulate(arr)     # [1, 3, 6, 10, 15] - cumsum
np.multiply.accumulate(arr) # [1, 2, 6, 24, 120] - cumprod

# Reduce methods
np.add.reduce(arr)         # 15 - sum
np.multiply.reduce(arr)    # 120 - product
np.maximum.reduce(arr)     # 5 - max

# Outer methods
np.add.outer(arr[:3], arr[:3])
# [[2, 3, 4],
#  [3, 4, 5],
#  [4, 5, 6]]

# At method (in-place operations at indices)
arr = np.array([1, 2, 3, 4, 5])
np.add.at(arr, [0, 2, 4], 10)  # arr: [11, 2, 13, 4, 15]
```

### Memory Views and Copies

```python
# View vs copy
arr = np.arange(10)
view = arr[::2]        # View - no data copied
copy = arr[::2].copy() # Explicit copy

view[0] = 999
# arr is modified!

copy[0] = 999
# arr is unchanged

# Check if it's a view
view.base is arr  # True
copy.base is None  # True

# Some operations return views
arr.reshape(2, 5)  # View (if possible)
arr.T              # View
arr[::2]           # View

# Some operations return copies
arr.flatten()      # Copy
arr + 1            # Copy
arr[[0, 2, 4]]     # Copy (fancy indexing)

# Avoid copies with out parameter
arr = np.random.randn(1000)
result = np.empty_like(arr)
np.sin(arr, out=result)  # Compute in-place, no extra memory

# Compound operations
arr = np.random.randn(1000)
arr += 1  # In-place, no copy
arr *= 2  # In-place, no copy
# vs
arr = arr + 1  # Creates new array
```

### Advanced Indexing Patterns

```python
# Multi-dimensional boolean indexing
data = np.random.randn(10, 5)
mask = data > 0
positive_values = data[mask]  # 1D array of positive values

# Keep structure with np.where
data_clipped = np.where(data > 0, data, 0)  # ReLU

# np.where with conditions
condition = data > 0.5
result = np.where(condition, data * 2, data / 2)

# np.select for multiple conditions
conditions = [
    data < -1,
    (data >= -1) & (data < 0),
    (data >= 0) & (data < 1),
    data >= 1
]
choices = [-1, 0, 0, 1]
result = np.select(conditions, choices, default=0)

# np.choose (limited to small number of choices)
indices = np.array([0, 1, 2, 1, 0])
choices = np.array([[1, 2, 3, 4, 5],
                    [10, 20, 30, 40, 50],
                    [100, 200, 300, 400, 500]])
result = np.choose(indices, choices)  # [1, 20, 300, 40, 5]

# Advanced batch indexing
batch = np.random.randn(32, 10)
indices = np.array([3, 1, 5, ...])  # 32 indices
selected = batch[np.arange(32), indices]  # 32 values

# Meshgrid for pairwise operations
x = np.array([1, 2, 3])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y, indexing='ij')
# X: [[1, 1],    Y: [[10, 20],
#      [2, 2],        [10, 20],
#      [3, 3]]        [10, 20]]
```

---

## Performance Optimization

### Memory Layout

```python
# C-order (row-major) vs Fortran-order (column-major)
arr_c = np.array([[1, 2], [3, 4]], order='C')  # Default
arr_f = np.array([[1, 2], [3, 4]], order='F')

# Check memory order
arr_c.flags['C_CONTIGUOUS']  # True
arr_f.flags['F_CONTIGUOUS']  # True

# Performance implication
# Iterating over rows is faster for C-order
# Iterating over columns is faster for F-order

# Use appropriate order for your access pattern
matrix_c = np.random.randn(1000, 1000, order='C')
matrix_f = np.random.randn(1000, 1000, order='F')

# Row-wise operations faster on C-order
row_sums_c = matrix_c.sum(axis=1)  # Fast

# Column-wise operations faster on F-order
col_sums_f = matrix_f.sum(axis=0)  # Fast
```

### In-place Operations

```python
# Avoid creating intermediate arrays
arr = np.random.randn(1000000)

# Bad - creates temporary arrays
result = (arr + 1) * 2 - 3

# Better - use in-place operations
arr += 1
arr *= 2
arr -= 3

# Use out parameter
arr = np.random.randn(1000)
result = np.empty_like(arr)
np.add(arr, 1, out=result)
np.multiply(result, 2, out=result)
np.subtract(result, 3, out=result)

# Compound operations
np.add(arr, 1, out=arr)  # Reuse input array
```

### Avoiding Copies

```python
# Slicing creates views (usually)
arr = np.arange(100)
view = arr[10:20]  # No copy

# Advanced indexing creates copies
copy = arr[[1, 5, 10]]  # Copy created

# Reshaping returns view if possible
view = arr.reshape(10, 10)  # View
copy = arr.reshape(10, 10, order='F')  # Copy (change order)

# Check if operation creates copy
original = np.arange(12)
reshaped = original.reshape(3, 4)
reshaped.base is original  # True - it's a view

# Explicit copy when needed
independent = arr.copy()
```

### Vectorization for Speed

```python
# Profile your code
import time

# Slow - Python loop
arr = np.random.randn(1000000)
start = time.time()
result = np.zeros_like(arr)
for i in range(len(arr)):
    result[i] = arr[i] ** 2 if arr[i] > 0 else 0
print(f"Loop: {time.time() - start:.4f}s")

# Fast - vectorized
start = time.time()
result = np.where(arr > 0, arr ** 2, 0)
print(f"Vectorized: {time.time() - start:.4f}s")
# Typically 50-100x faster

# Use specialized functions
# Bad
result = np.sqrt(np.sum(arr ** 2))
# Good
result = np.linalg.norm(arr)  # Optimized implementation
```

### Memory-efficient Operations

```python
# Generator expressions for large data
def process_batches(data, batch_size):
    n_batches = len(data) // batch_size
    for i in range(n_batches):
        yield data[i*batch_size:(i+1)*batch_size]

# Memory-mapped arrays for huge datasets
mmap = np.memmap('large_file.dat', dtype='float32', mode='r', shape=(1000000, 1000))
# Only loads data into memory when accessed

# Delete intermediate results
large_array = np.random.randn(10000, 10000)
result = np.sum(large_array, axis=0)
del large_array  # Free memory

# Use smaller dtypes when possible
arr_64 = np.random.randn(1000000).astype(np.float64)  # 8 MB
arr_32 = np.random.randn(1000000).astype(np.float32)  # 4 MB
arr_16 = np.random.randn(1000000).astype(np.float16)  # 2 MB
```

### Numba for Ultimate Speed

```python
from numba import jit, prange

# Accelerate with JIT compilation
@jit(nopython=True)
def compute_pairwise_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = 0.0
            for k in range(X.shape[1]):
                d += (X[i, k] - X[j, k]) ** 2
            distances[i, j] = np.sqrt(d)
            distances[j, i] = distances[i, j]
    return distances

# Parallel execution
@jit(nopython=True, parallel=True)
def parallel_sum_squares(arr):
    result = 0.0
    for i in prange(len(arr)):
        result += arr[i] ** 2
    return result
```

---

## Common ML Patterns

### One-Hot Encoding

```python
# Method 1: Using np.eye
labels = np.array([0, 2, 1, 0, 3])
n_classes = 4
one_hot = np.eye(n_classes)[labels]
# [[1, 0, 0, 0],
#  [0, 0, 1, 0],
#  [0, 1, 0, 0],
#  [1, 0, 0, 0],
#  [0, 0, 0, 1]]

# Method 2: Manual
def one_hot_encode(labels, n_classes):
    one_hot = np.zeros((len(labels), n_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# Reverse: one-hot to labels
labels_recovered = np.argmax(one_hot, axis=1)
```

### Train-Test Split

```python
def train_test_split(X, y, test_size=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(X)
    indices = rng.permutation(n)

    split_idx = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# K-fold cross-validation indices
def k_fold_indices(n, k=5, shuffle=True, random_state=None):
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_size = n // k
    for i in range(k):
        test_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([indices[:i*fold_size],
                                    indices[(i+1)*fold_size:]])
        yield train_idx, test_idx
```

### Mini-batch Generation

```python
def generate_batches(X, y, batch_size, shuffle=True, random_state=None):
    """Generator for mini-batches"""
    n = len(X)
    rng = np.random.default_rng(random_state)

    if shuffle:
        indices = rng.permutation(n)
        X, y = X[indices], y[indices]

    n_batches = n // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        yield X[start:end], y[start:end]

    # Last batch (if incomplete)
    if n % batch_size != 0:
        yield X[n_batches*batch_size:], y[n_batches*batch_size:]

# Usage
for X_batch, y_batch in generate_batches(X_train, y_train, batch_size=32):
    # Train on batch
    pass
```

### Distance Computations

```python
# Euclidean distance matrix (vectorized)
def euclidean_distances(X, Y=None):
    """
    Compute pairwise Euclidean distances
    If Y is None, compute distances within X
    """
    if Y is None:
        Y = X

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)
    distances = X_norm + Y_norm - 2 * X @ Y.T

    # Handle numerical errors
    distances = np.maximum(distances, 0)
    return np.sqrt(distances)

# Cosine similarity
def cosine_similarity(X, Y=None):
    if Y is None:
        Y = X

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X_norm @ Y_norm.T

# Manhattan distance
def manhattan_distances(X, Y=None):
    if Y is None:
        Y = X
    return np.sum(np.abs(X[:, np.newaxis] - Y[np.newaxis, :]), axis=2)
```

### Activation Functions

```python
# ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Sigmoid
def sigmoid(x):
    # Numerically stable
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Softmax (numerically stable)
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### Loss Functions

```python
# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Cross-entropy (numerically stable)
def cross_entropy(y_true, y_pred, epsilon=1e-15):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

# Binary cross-entropy
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical cross-entropy
def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Hinge loss (SVM)
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
```

### Convolution Operations

```python
# 1D convolution (simple implementation)
def conv1d(x, kernel, stride=1, padding=0):
    if padding > 0:
        x = np.pad(x, padding, mode='constant')

    n = len(x)
    k = len(kernel)
    output_size = (n - k) // stride + 1
    output = np.zeros(output_size)

    for i in range(output_size):
        start = i * stride
        output[i] = np.sum(x[start:start+k] * kernel)

    return output

# 2D convolution (simple, unoptimized)
def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    h, w = image.shape
    kh, kw = kernel.shape

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            output[i, j] = np.sum(image[r:r+kh, c:c+kw] * kernel)

    return output

# Pooling operations
def max_pool2d(x, pool_size=2, stride=2):
    h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            output[i, j] = np.max(x[r:r+pool_size, c:c+pool_size])

    return output

def avg_pool2d(x, pool_size=2, stride=2):
    h, w = x.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            output[i, j] = np.mean(x[r:r+pool_size, c:c+pool_size])

    return output
```

### Gradient Checking

```python
def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences
    Useful for debugging backpropagation
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + epsilon
        fx_plus = f(x)

        x[idx] = old_value - epsilon
        fx_minus = f(x)

        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        x[idx] = old_value
        it.iternext()

    return grad

def gradient_check(f, x, analytic_grad, epsilon=1e-5):
    """Check if analytic gradient is correct"""
    numerical_grad = numerical_gradient(f, x, epsilon)

    # Relative error
    numerator = np.linalg.norm(numerical_grad - analytic_grad)
    denominator = np.linalg.norm(numerical_grad) + np.linalg.norm(analytic_grad)
    rel_error = numerator / (denominator + 1e-8)

    print(f"Relative error: {rel_error}")
    return rel_error < 1e-5  # Threshold for "correct"
```

### Data Augmentation Helpers

```python
# Image augmentation primitives
def random_flip(image, horizontal=True, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        axis = 1 if horizontal else 0
        return np.flip(image, axis=axis)
    return image

def random_rotation_90(image, p=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p:
        k = rng.integers(1, 4)  # 90, 180, or 270 degrees
        return np.rot90(image, k=k)
    return image

def random_crop(image, crop_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    h, w = image.shape[:2]
    ch, cw = crop_size

    top = rng.integers(0, h - ch + 1)
    left = rng.integers(0, w - cw + 1)

    return image[top:top+ch, left:left+cw]

def normalize_image(image, mean, std):
    """Normalize image with mean and std per channel"""
    return (image - mean) / std
```

---

## Summary

NumPy mastery is essential for ML engineering. Key takeaways:

1. **Vectorization is king**: Avoid Python loops, use array operations
2. **Broadcasting enables elegance**: Learn the rules, use them everywhere
3. **Memory matters**: Understand views vs copies, use appropriate dtypes
4. **Use the right tool**: einsum for complex operations, specialized functions when available
5. **Profile your code**: Measure before optimizing
6. **Build on NumPy conventions**: Your code will integrate better with the ecosystem

**Next Steps:**
- Practice implementing ML algorithms from scratch in NumPy
- Study PyTorch/TensorFlow source code to see NumPy patterns at scale
- Profile your code to identify bottlenecks
- Learn Numba for the last 10x speedup when vectorization isn't enough

**Resources:**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy Paper (Nature)](https://www.nature.com/articles/s41586-020-2649-2)
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
- [Advanced NumPy](https://scipy-lectures.org/advanced/advanced_numpy/)
