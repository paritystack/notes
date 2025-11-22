# JAX

JAX is a Python library for high-performance numerical computing and machine learning research. It combines the familiar NumPy API with automatic differentiation, JIT compilation via XLA, and easy parallelization across GPUs and TPUs. JAX embraces functional programming principles and composable transformations.

**Key Features:**
- NumPy-compatible API with GPU/TPU acceleration
- Automatic differentiation (forward and reverse mode)
- JIT compilation for performance optimization
- Vectorization (vmap) and parallelization (pmap)
- Functional approach with pure functions and immutability

## Installation

```bash
# CPU version
pip install jax jaxlib

# GPU version (CUDA 12)
pip install -U "jax[cuda12]"

# TPU version
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Common ML libraries
pip install flax optax
```

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Automatic Differentiation](#automatic-differentiation)
3. [JIT Compilation](#jit-compilation)
4. [Vectorization with vmap](#vectorization-with-vmap)
5. [Parallelization with pmap](#parallelization-with-pmap)
6. [Random Numbers](#random-numbers)
7. [PyTrees](#pytrees)
8. [Neural Networks with Flax](#neural-networks-with-flax)
9. [Optimization with Optax](#optimization-with-optax)
10. [Advanced Techniques](#advanced-techniques)
11. [Best Practices](#best-practices)
12. [Common Issues](#common-issues)
13. [Further Resources](#further-resources)

## Core Concepts

### JAX Arrays

JAX arrays are similar to NumPy arrays but immutable and designed for acceleration.

```python
import jax.numpy as jnp
import numpy as np

# Create JAX arrays (similar to NumPy)
x = jnp.array([1, 2, 3, 4, 5])
y = jnp.linspace(0, 10, 100)
z = jnp.zeros((3, 4))

# NumPy compatibility
np_array = np.array([1, 2, 3])
jax_array = jnp.array(np_array)  # Convert NumPy to JAX
back_to_numpy = np.array(jax_array)  # Convert JAX to NumPy

# Most NumPy operations work identically
result = jnp.dot(x, x)  # 55
mean = jnp.mean(y)
reshaped = z.reshape(4, 3)
```

### Immutability

JAX arrays are immutable - operations return new arrays rather than modifying in place.

```python
# This doesn't work in JAX (will raise an error)
# x[0] = 10  # TypeError: JAX arrays are immutable

# Instead, use .at[] syntax for updates
x = jnp.array([1, 2, 3, 4, 5])
x_updated = x.at[0].set(10)  # Returns new array: [10, 2, 3, 4, 5]
x_incremented = x.at[1].add(5)  # Returns: [1, 7, 3, 4, 5]

# Multiple updates
x_multi = x.at[0].set(10).at[2].multiply(3)  # [10, 2, 9, 4, 5]

# Slice updates
y = jnp.zeros(10)
y_updated = y.at[2:5].set(jnp.array([7, 8, 9]))  # Updates indices 2, 3, 4
```

### Pure Functions

JAX transformations require pure functions (no side effects, deterministic output).

```python
# Pure function (good for JAX)
def pure_function(x):
    return x ** 2 + 2 * x + 1

# Impure function (avoid with JAX transformations)
counter = 0
def impure_function(x):
    global counter
    counter += 1  # Side effect!
    return x ** 2

# Another pure function example
def compute_loss(params, x, y):
    """Pure function - output depends only on inputs"""
    predictions = params['w'] * x + params['b']
    return jnp.mean((predictions - y) ** 2)
```

## Automatic Differentiation

JAX provides powerful automatic differentiation through `grad`, `value_and_grad`, and more.

### Basic Gradients

```python
import jax
from jax import grad, value_and_grad

# Simple function
def f(x):
    return x ** 3 + 2 * x ** 2 - 5 * x + 3

# Compute gradient (derivative)
df_dx = grad(f)
print(df_dx(2.0))  # 3*(2^2) + 4*2 - 5 = 15.0

# Get both value and gradient
value, gradient = value_and_grad(f)(2.0)
print(f"f(2.0) = {value}, f'(2.0) = {gradient}")  # f(2.0) = 5.0, f'(2.0) = 15.0
```

### Gradients with Multiple Arguments

```python
# Function with multiple arguments
def loss(params, x, y):
    w, b = params
    pred = w * x + b
    return jnp.mean((pred - y) ** 2)

# Gradient with respect to first argument (params)
grad_loss = grad(loss)

params = (2.0, 0.5)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 4.0, 6.0])

grads = grad_loss(params, x, y)  # Gradient w.r.t. params
print(grads)  # Tuple of gradients for w and b

# Specify which argument to differentiate
grad_wrt_x = grad(loss, argnums=1)  # Gradient w.r.t. x (second argument)
grad_wrt_y = grad(loss, argnums=2)  # Gradient w.r.t. y (third argument)

# Multiple arguments at once
grad_multi = grad(loss, argnums=(0, 1))  # Gradients w.r.t. params and x
```

### Jacobians and Hessians

```python
from jax import jacfwd, jacrev, hessian

# Jacobian (for vector-valued functions)
def vector_function(x):
    return jnp.array([x[0]**2, x[1]**3, x[0]*x[1]])

x = jnp.array([2.0, 3.0])

# Forward-mode Jacobian (efficient for few inputs, many outputs)
jacobian_fwd = jacfwd(vector_function)(x)
print(jacobian_fwd)
# [[4.  0. ]
#  [0.  27.]
#  [3.  2. ]]

# Reverse-mode Jacobian (efficient for many inputs, few outputs)
jacobian_rev = jacrev(vector_function)(x)

# Hessian (second derivatives)
def scalar_function(x):
    return x[0]**3 + x[1]**2 + x[0]*x[1]

hess = hessian(scalar_function)(x)
print(hess)
# [[12.  1.]
#  [ 1.  2.]]
```

### Gradients in Machine Learning

```python
# Typical ML loss function
def mse_loss(params, batch):
    x, y = batch
    predictions = params['w'] @ x + params['b']
    return jnp.mean((predictions - y) ** 2)

# Initialize parameters
params = {
    'w': jnp.array([[0.5, 0.3], [0.2, 0.8]]),
    'b': jnp.array([0.1, 0.2])
}

batch = (jnp.ones((2, 10)), jnp.ones((2, 10)))

# Compute loss and gradients
loss_val, grads = value_and_grad(mse_loss)(params, batch)
print(f"Loss: {loss_val}")
print(f"Gradients: {grads}")

# Update parameters (simple gradient descent)
learning_rate = 0.01
params_updated = jax.tree_map(
    lambda p, g: p - learning_rate * g,
    params, grads
)
```

## JIT Compilation

JIT (Just-In-Time) compilation via XLA dramatically improves performance by compiling functions to optimized machine code.

### Basic JIT

```python
from jax import jit
import time

# Regular function
def slow_function(x):
    return jnp.sum(x ** 2) + jnp.mean(x) * 2

# JIT-compiled function
fast_function = jit(slow_function)

# Or use decorator
@jit
def another_fast_function(x):
    return jnp.sum(x ** 2) + jnp.mean(x) * 2

# Benchmark
x = jnp.ones((1000, 1000))

# First call compiles (slower)
start = time.time()
result1 = fast_function(x)
first_call_time = time.time() - start

# Subsequent calls use compiled version (much faster)
start = time.time()
result2 = fast_function(x)
subsequent_call_time = time.time() - start

print(f"First call: {first_call_time:.4f}s (includes compilation)")
print(f"Second call: {subsequent_call_time:.4f}s (cached)")
```

### JIT with Static Arguments

```python
from functools import partial

# Functions with static arguments
@partial(jit, static_argnums=(1,))
def power_function(x, n):
    """n is static - will recompile if n changes"""
    return x ** n

result = power_function(jnp.array([1, 2, 3]), 2)  # Compiles for n=2
result = power_function(jnp.array([4, 5, 6]), 2)  # Reuses compilation
result = power_function(jnp.array([7, 8, 9]), 3)  # Recompiles for n=3

# Static argument names
@partial(jit, static_argnames=['activation'])
def apply_activation(x, activation='relu'):
    if activation == 'relu':
        return jnp.maximum(0, x)
    elif activation == 'tanh':
        return jnp.tanh(x)
    else:
        return x
```

### JIT Constraints

```python
# Good: Array shapes known at compile time
@jit
def good_function(x):
    return x.reshape(10, -1)  # Shape is concrete

# Bad: Control flow based on array values
@jit
def bad_function(x):
    if x[0] > 0:  # Error! Can't JIT boolean from array
        return x * 2
    else:
        return x * 3

# Good: Use jnp.where for conditional logic
@jit
def good_conditional(x):
    return jnp.where(x > 0, x * 2, x * 3)

# Good: Use lax.cond for branching
from jax import lax

@jit
def good_branching(x, flag):
    return lax.cond(
        flag,
        lambda x: x * 2,  # True branch
        lambda x: x * 3,  # False branch
        x
    )
```

## Vectorization with vmap

`vmap` automatically vectorizes functions, eliminating manual loops and improving performance.

### Basic vmap

```python
from jax import vmap

# Function that works on single example
def predict(params, x):
    return params['w'] @ x + params['b']

params = {'w': jnp.array([0.5, 0.3]), 'b': 1.0}

# Single example
x_single = jnp.array([1.0, 2.0])
result_single = predict(params, x_single)  # Shape: ()

# Batch of examples (manual loop - slow)
x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
results_manual = jnp.array([predict(params, x) for x in x_batch])

# Vectorized (automatic - fast!)
predict_batch = vmap(predict, in_axes=(None, 0))
# in_axes=(None, 0) means: don't batch params, batch x along axis 0
results_vmap = predict_batch(params, x_batch)  # Shape: (3,)
```

### Advanced vmap

```python
# Batch over specific axes
def matrix_multiply(A, B):
    return A @ B

A_batch = jnp.ones((5, 3, 4))  # 5 matrices of shape (3, 4)
B = jnp.ones((4, 2))

# Apply to each matrix in the batch
result = vmap(matrix_multiply, in_axes=(0, None))(A_batch, B)
print(result.shape)  # (5, 3, 2)

# Batch over multiple axes
def compute_distances(x, y):
    return jnp.linalg.norm(x - y)

X = jnp.ones((10, 3))  # 10 points in 3D
Y = jnp.ones((20, 3))  # 20 points in 3D

# Compute all pairwise distances
distances = vmap(
    lambda x: vmap(lambda y: compute_distances(x, y))(Y)
)(X)
print(distances.shape)  # (10, 20)

# Or use vmap with out_axes
@vmap
def normalize_rows(matrix):
    return matrix / jnp.linalg.norm(matrix, axis=1, keepdims=True)
```

### vmap for Gradients

```python
# Compute per-example gradients
def loss(params, x, y):
    pred = params['w'] @ x + params['b']
    return (pred - y) ** 2

params = {'w': jnp.array([0.5, 0.3]), 'b': 1.0}
x_batch = jnp.ones((100, 2))
y_batch = jnp.ones(100)

# Per-example gradients
per_example_grads = vmap(
    grad(loss),
    in_axes=(None, 0, 0)
)(params, x_batch, y_batch)

print(per_example_grads['w'].shape)  # (100, 2) - gradient for each example

# Average gradient
avg_grad = jax.tree_map(lambda g: jnp.mean(g, axis=0), per_example_grads)
```

## Parallelization with pmap

`pmap` parallelizes computation across multiple devices (GPUs/TPUs).

### Basic pmap

```python
from jax import pmap, local_device_count

# Check available devices
n_devices = local_device_count()
print(f"Available devices: {n_devices}")

# Function to parallelize
def f(x):
    return x ** 2 + x

# Create data for each device
x = jnp.arange(n_devices * 10).reshape(n_devices, 10)
print(x.shape)  # (n_devices, 10)

# Parallelize across devices
f_parallel = pmap(f)
result = f_parallel(x)
print(result.shape)  # (n_devices, 10)
# Each device processes one slice of the batch
```

### pmap for Training

```python
# Parallel training step
@pmap
def train_step(params, batch):
    x, y = batch
    loss_val, grads = value_and_grad(loss)(params, x, y)
    # Update params
    new_params = jax.tree_map(
        lambda p, g: p - 0.01 * g,
        params, grads
    )
    return new_params, loss_val

# Replicate parameters across devices
params = {'w': jnp.ones(10), 'b': 0.0}
params_replicated = jax.tree_map(
    lambda x: jnp.array([x] * n_devices),
    params
)

# Shard batch across devices
batch_size = 32
x_batch = jnp.ones((batch_size, 10))
y_batch = jnp.ones(batch_size)

# Reshape to (n_devices, batch_per_device, ...)
x_sharded = x_batch.reshape(n_devices, -1, 10)
y_sharded = y_batch.reshape(n_devices, -1)

batch_sharded = (x_sharded, y_sharded)

# Train in parallel
new_params, losses = train_step(params_replicated, batch_sharded)
print(f"Loss on each device: {losses}")
```

### Collective Operations

```python
# Communication between devices
@pmap
def allreduce_mean(x):
    return lax.pmean(x, axis_name='devices')

# Use axis_name to specify communication
@pmap(axis_name='batch')
def normalize_across_devices(x):
    # Compute mean across all devices
    global_mean = lax.pmean(x, 'batch')
    return x - global_mean
```

## Random Numbers

JAX uses explicit PRNG keys for reproducibility and parallelization.

### PRNG Keys

```python
from jax import random

# Create a random key
key = random.PRNGKey(0)

# Generate random numbers
random_uniform = random.uniform(key, shape=(5,))
print(random_uniform)

# WRONG: Reusing the same key gives same numbers
random_1 = random.uniform(key, shape=(3,))
random_2 = random.uniform(key, shape=(3,))  # Same as random_1!

# CORRECT: Split keys for independent random numbers
key, subkey1, subkey2 = random.split(key, 3)
random_1 = random.uniform(subkey1, shape=(3,))
random_2 = random.uniform(subkey2, shape=(3,))  # Different from random_1

# Common pattern: split before each use
key, subkey = random.split(key)
x = random.normal(subkey, shape=(10,))

key, subkey = random.split(key)
y = random.normal(subkey, shape=(10,))
```

### Random Distributions

```python
key = random.PRNGKey(42)

# Uniform distribution
key, subkey = random.split(key)
uniform = random.uniform(subkey, shape=(5,), minval=0, maxval=10)

# Normal distribution
key, subkey = random.split(key)
normal = random.normal(subkey, shape=(5,))

# Categorical
key, subkey = random.split(key)
logits = jnp.array([1.0, 2.0, 3.0])
samples = random.categorical(subkey, logits, shape=(10,))

# Permutation
key, subkey = random.split(key)
x = jnp.arange(10)
shuffled = random.permutation(subkey, x)

# Random choice
key, subkey = random.split(key)
indices = random.choice(subkey, 100, shape=(10,), replace=False)
```

### Random in Training Loops

```python
def train_step(key, params, batch):
    # Split key for dropout
    key, dropout_key = random.split(key)

    def loss_fn(params):
        x, y = batch
        # Use dropout_key for randomness
        pred = forward_with_dropout(params, x, dropout_key)
        return jnp.mean((pred - y) ** 2)

    loss, grads = value_and_grad(loss_fn)(params)
    new_params = update_params(params, grads)

    return key, new_params, loss

# Training loop
key = random.PRNGKey(0)
for epoch in range(10):
    for batch in data_loader:
        key, params, loss = train_step(key, params, batch)
```

## PyTrees

PyTrees are nested structures (dicts, lists, tuples) that JAX can traverse and transform.

### Working with PyTrees

```python
import jax.tree_util as tree

# PyTree examples
params_dict = {
    'layer1': {'w': jnp.ones((5, 3)), 'b': jnp.zeros(5)},
    'layer2': {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)}
}

params_list = [jnp.ones(5), jnp.zeros(3), jnp.ones((2, 2))]

# tree_map: Apply function to all leaves
scaled = tree.tree_map(lambda x: x * 2, params_dict)

# Combine two pytrees
grads = tree.tree_map(lambda x: jnp.ones_like(x), params_dict)
updated = tree.tree_map(
    lambda p, g: p - 0.01 * g,
    params_dict, grads
)

# tree_leaves: Get all leaf values
leaves = tree.tree_leaves(params_dict)
print(f"Number of parameters: {sum(x.size for x in leaves)}")

# tree_structure: Get structure without values
structure = tree.tree_structure(params_dict)

# Flatten and unflatten
flat, treedef = tree.tree_flatten(params_dict)
reconstructed = tree.tree_unflatten(treedef, flat)
```

### Custom PyTree Classes

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MLPParams:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def tree_flatten(self):
        # Return (children, aux_data)
        children = (self.weights, self.biases)
        aux_data = None
        return children, aux_data

    def tree_unflatten(aux_data, children):
        return MLPParams(*children)

# Now can use with tree_map
params = MLPParams([jnp.ones((5, 3))], [jnp.zeros(5)])
scaled = tree.tree_map(lambda x: x * 2, params)
```

## Neural Networks with Flax

Flax is a high-level neural network library built on JAX.

### Basic Flax Module

```python
from flax import linen as nn

class SimpleMLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # Layers are created on first call
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# Initialize model
model = SimpleMLP(hidden_dim=128, output_dim=10)

# Initialize parameters
key = random.PRNGKey(0)
dummy_input = jnp.ones((1, 784))  # Batch of 1, 784 features
params = model.init(key, dummy_input)

# Forward pass
output = model.apply(params, dummy_input)
print(output.shape)  # (1, 10)
```

### Training with Flax

```python
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=10)(x)
        return x

# Create training step
@jit
def train_step(params, batch, rng):
    x, y = batch

    def loss_fn(params):
        logits = CNN().apply(params, x, training=True, rngs={'dropout': rng})
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(params)
    return grads, loss

# Initialize
model = CNN()
key = random.PRNGKey(0)
key, init_key = random.split(key)
dummy_x = jnp.ones((1, 28, 28, 1))
params = model.init(init_key, dummy_x, training=False)
```

### State Management (BatchNorm)

```python
class ModelWithBatchNorm(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(features=128)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

# Initialize with batch_stats
model = ModelWithBatchNorm()
variables = model.init(key, dummy_input, training=False)
params = variables['params']
batch_stats = variables['batch_stats']

# Training step with mutable state
def train_step_with_bn(params, batch_stats, batch):
    x, y = batch

    def loss_fn(params):
        logits, new_batch_stats = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            x, training=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean((logits - y) ** 2)
        return loss, new_batch_stats

    (loss, new_batch_stats), grads = value_and_grad(loss_fn, has_aux=True)(params)
    return grads, loss, new_batch_stats['batch_stats']
```

## Optimization with Optax

Optax provides composable gradient transformations and optimizers.

### Basic Optimizers

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=0.001)

# Initialize optimizer state
params = {'w': jnp.ones((5, 3)), 'b': jnp.zeros(5)}
opt_state = optimizer.init(params)

# Training step
def train_step(params, opt_state, batch):
    loss, grads = value_and_grad(compute_loss)(params, batch)

    # Update using optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

# Training loop
for epoch in range(100):
    for batch in data_loader:
        params, opt_state, loss = train_step(params, opt_state, batch)
```

### Common Optimizers

```python
# SGD with momentum
optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# Adam
optimizer = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999)

# AdamW (Adam with weight decay)
optimizer = optax.adamw(learning_rate=0.001, weight_decay=0.01)

# RMSprop
optimizer = optax.rmsprop(learning_rate=0.001)

# Learning rate schedules
schedule = optax.exponential_decay(
    init_value=0.1,
    transition_steps=1000,
    decay_rate=0.99
)
optimizer = optax.adam(learning_rate=schedule)

# Cosine decay
schedule = optax.cosine_decay_schedule(
    init_value=0.1,
    decay_steps=10000
)
optimizer = optax.adam(learning_rate=schedule)
```

### Gradient Transformations

```python
# Combine transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.scale_by_adam(),  # Adam updates
    optax.scale(-0.001)  # Learning rate
)

# Gradient accumulation
optimizer = optax.MultiSteps(
    optax.adam(0.001),
    every_k_schedule=4  # Accumulate over 4 steps
)

# Different learning rates for different parameters
def label_fn(path, _):
    if 'bias' in path:
        return 'bias'
    return 'weight'

optimizer = optax.multi_transform(
    {
        'weight': optax.adam(0.001),
        'bias': optax.adam(0.01)  # Higher LR for biases
    },
    label_fn
)
```

## Advanced Techniques

### Custom Gradients

```python
from jax import custom_vjp

# Define custom gradient for a function
@custom_vjp
def clip_gradient(x):
    return x

def clip_gradient_fwd(x):
    # Forward pass
    return x, None

def clip_gradient_bwd(res, g):
    # Custom backward pass - clip gradients
    return (jnp.clip(g, -1.0, 1.0),)

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

# Use in computation
def loss_with_clipped_grad(x):
    x = clip_gradient(x)  # Gradients will be clipped
    return x ** 2

grad_fn = grad(loss_with_clipped_grad)
print(grad_fn(5.0))  # Gradient is clipped to 1.0 instead of 10.0
```

### Scan for Loops

```python
from jax.lax import scan

# Efficient loop implementation
def rnn_cell(carry, x):
    h = carry
    h_new = jnp.tanh(jnp.dot(W_h, h) + jnp.dot(W_x, x))
    return h_new, h_new

# Initialize
W_h = jnp.ones((10, 10))
W_x = jnp.ones((10, 5))
h_0 = jnp.zeros(10)
xs = jnp.ones((20, 5))  # Sequence of 20 inputs

# Run RNN with scan (much faster than Python loop)
final_h, all_h = scan(rnn_cell, h_0, xs)
print(all_h.shape)  # (20, 10) - hidden states for each timestep

# Reverse scan
final_h, all_h = scan(rnn_cell, h_0, xs, reverse=True)
```

### Checkpointing (Gradient Checkpointing)

```python
from jax.checkpoint import checkpoint

# Regular function (stores all intermediates)
def expensive_layer(x):
    for _ in range(100):
        x = jnp.tanh(x @ W + b)
    return x

# Checkpointed version (recomputes on backward pass)
expensive_layer_checkpointed = checkpoint(expensive_layer)

# Use in larger model to save memory
def large_model(x):
    x = expensive_layer_checkpointed(x)
    x = another_layer(x)
    return x
```

### Custom Training Loop

```python
def create_train_state(rng, learning_rate, input_shape):
    """Create initial training state"""
    model = SimpleMLP(hidden_dim=128, output_dim=10)
    params = model.init(rng, jnp.ones(input_shape))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return model, params, optimizer, opt_state

@jit
def train_step(model, params, opt_state, optimizer, batch):
    """Single training step"""
    x, y = batch

    def loss_fn(params):
        logits = model.apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Compute accuracy
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)

    return params, opt_state, loss, accuracy

@jit
def eval_step(model, params, batch):
    """Evaluation step"""
    x, y = batch
    logits = model.apply(params, x)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, accuracy

# Full training loop
def train(num_epochs, train_data, val_data):
    rng = random.PRNGKey(0)
    model, params, optimizer, opt_state = create_train_state(
        rng, learning_rate=0.001, input_shape=(1, 784)
    )

    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = 0.0, 0.0
        for batch in train_data:
            params, opt_state, loss, acc = train_step(
                model, params, opt_state, optimizer, batch
            )
            train_loss += loss
            train_acc += acc

        # Validation
        val_loss, val_acc = 0.0, 0.0
        for batch in val_data:
            loss, acc = eval_step(model, params, batch)
            val_loss += loss
            val_acc += acc

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

    return params
```

## Best Practices

### Performance Optimization

- **Use JIT compilation** for all performance-critical functions
- **Vectorize with vmap** instead of Python loops
- **Batch operations** to maximize hardware utilization
- **Avoid unnecessary array copies** - JAX arrays are immutable but efficient
- **Profile your code** using `jax.profiler`

```python
# Good: Vectorized and JIT-compiled
@jit
def efficient_computation(x):
    return vmap(lambda a: jnp.sum(a ** 2))(x)

# Bad: Python loop
def inefficient_computation(x):
    return jnp.array([jnp.sum(a ** 2) for a in x])
```

### Memory Management

- **Use gradient checkpointing** for very deep networks
- **Clear unused arrays** to free memory
- **Use `jax.device_put`** to control array placement
- **Monitor memory** with `jax.local_devices()`

```python
from jax import device_put

# Explicitly place array on device
x_gpu = device_put(x, device=jax.devices()[0])

# Clear cache if needed (after debugging)
from jax import clear_caches
clear_caches()
```

### Debugging

- **Disable JIT** during debugging: set `JAX_DISABLE_JIT=1` environment variable
- **Use `jax.debug.print()`** inside JIT-compiled functions
- **Check for NaNs** with `jax.debug.check_nan()`

```python
import jax.debug as debug

@jit
def debug_function(x):
    debug.print("x = {}", x)  # Prints during execution
    y = x ** 2
    debug.print("y = {}", y)
    return y

# Check for NaNs
x = jnp.array([1.0, float('nan'), 3.0])
# debug.check_nan(x)  # Raises error if NaNs present
```

### Code Organization

- **Separate model definition from training logic**
- **Use configuration files** (e.g., with Hydra) for hyperparameters
- **Modularize transformations** (grad, jit, vmap) for reusability
- **Type hints** improve code clarity

```python
from typing import Dict, Tuple
import jax.numpy as jnp

def forward(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """Type-annotated forward pass"""
    return params['w'] @ x + params['b']

def loss_fn(params: Dict, batch: Tuple) -> float:
    """Type-annotated loss function"""
    x, y = batch
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)
```

### Random Number Management

- **Always split keys** before use
- **Pass keys explicitly** through function calls
- **Don't reuse keys** - leads to correlated randomness

```python
# Good: Proper key management
def training_epoch(key, params, data):
    losses = []
    for batch in data:
        key, subkey = random.split(key)
        params, loss = train_step(params, batch, subkey)
        losses.append(loss)
    return key, params, losses

# Bad: Reusing key
def bad_training_epoch(key, params, data):
    for batch in data:
        params, loss = train_step(params, batch, key)  # Same key every time!
```

## Common Issues

### ConcretizationTypeError

**Problem:** Trying to use array values in control flow inside JIT.

```python
# Error
@jit
def bad(x):
    if x.sum() > 0:  # Can't convert array to bool in JIT
        return x
    return -x

# Solution: Use jnp.where or lax.cond
@jit
def good(x):
    return jnp.where(x.sum() > 0, x, -x)
```

### TracerArrayConversionError

**Problem:** Trying to convert JAX array to NumPy array inside transformation.

```python
# Error
@jit
def bad(x):
    return np.array(x)  # Can't convert to NumPy in JIT

# Solution: Use JAX operations
@jit
def good(x):
    return jnp.array(x)
```

### UnexpectedTracerError

**Problem:** Leaking tracers outside of transformations.

```python
# Error
cached_value = None

@jit
def bad(x):
    global cached_value
    cached_value = x  # Leaks tracer!
    return x * 2

# Solution: Keep everything inside the function
@jit
def good(x):
    temp = x  # Local variable
    return temp * 2
```

### Out of Memory

**Problem:** Running out of GPU memory.

**Solutions:**
- Use gradient checkpointing for large models
- Reduce batch size
- Use mixed precision training
- Clear caches: `jax.clear_caches()`

```python
# Use smaller dtype
x = jnp.array(data, dtype=jnp.float16)  # Instead of float32

# Gradient checkpointing
from jax.checkpoint import checkpoint
layer = checkpoint(expensive_layer)
```

### Slow First Iteration

**Problem:** First call to JIT function is slow.

**Explanation:** This is normal - JAX compiles on first call. Subsequent calls are fast.

```python
# Warm up by calling once
@jit
def f(x):
    return x ** 2

_ = f(jnp.ones(10))  # Compilation happens here
# Now subsequent calls are fast
result = f(jnp.ones(10))  # Fast!
```

## Further Resources

### Official Documentation
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)

### Tutorials
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Thinking in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

### Advanced Topics
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Advanced Autodiff](https://jax.readthedocs.io/en/latest/notebooks/Advanced_Autodiff.html)
- [Parallelism](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)

### Community
- [JAX Discussions](https://github.com/google/jax/discussions)
- [JAX Examples](https://github.com/google/jax/tree/main/examples)
