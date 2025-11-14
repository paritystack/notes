# Quantization

## Overview

Quantization is the process of reducing the precision of numerical representations in neural networks, typically converting high-precision floating-point weights and activations to lower-precision formats like integers. This technique is fundamental for deploying machine learning models efficiently on resource-constrained devices and achieving faster inference with minimal accuracy loss.

In modern deep learning, quantization has become essential for:
- Deploying large language models (LLMs) on consumer hardware
- Running neural networks on edge devices (smartphones, IoT)
- Reducing inference costs in production systems
- Enabling real-time applications with strict latency requirements

## Fundamentals

### Numerical Representations

Neural networks traditionally use floating-point arithmetic:

| Format | Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|------|----------|----------|-------|-----------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits |
| FP16 | 16 | 1 | 5 | 10 | ±65,504 | ~3 decimal digits |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2 decimal digits |
| INT8 | 8 | 1 | - | 7 | -128 to 127 | Discrete |
| INT4 | 4 | 1 | - | 3 | -8 to 7 | Discrete |

**Brain Float 16 (BF16)**: Maintains FP32's range with reduced precision, ideal for training.

**Integer Formats**: Fixed-point arithmetic, faster on specialized hardware.

### Quantization Mathematics

The core quantization operation maps continuous values to discrete levels:

```
Quantization: q = round(x / scale) + zero_point
Dequantization: x_approx = (q - zero_point) * scale
```

**Parameters**:
- `scale`: Scaling factor determining step size
- `zero_point`: Offset for asymmetric quantization
- `q`: Quantized integer value
- `x`: Original floating-point value

### Symmetric Quantization

Zero-point is 0, simplifying computation:

```
scale = max(|x_max|, |x_min|) / (2^(b-1) - 1)
q = round(x / scale)
```

For INT8: `scale = max(|x_max|, |x_min|) / 127`

**Example**:
```python
import numpy as np

def symmetric_quantize(x, num_bits=8):
    """Symmetric quantization"""
    qmax = 2**(num_bits - 1) - 1  # 127 for INT8
    scale = np.max(np.abs(x)) / qmax
    q = np.round(x / scale).astype(np.int8)
    return q, scale

# Example
x = np.array([1.5, -2.3, 0.5, 3.1])
q, scale = symmetric_quantize(x)
print(f"Original: {x}")
print(f"Quantized: {q}")
print(f"Scale: {scale}")

# Dequantize
x_dequant = q * scale
print(f"Dequantized: {x_dequant}")
print(f"Error: {np.abs(x - x_dequant)}")
```

### Asymmetric Quantization

Uses both scale and zero-point for full range utilization:

```
scale = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
q = round(x / scale) + zero_point
```

For UINT8: Full range [0, 255] is utilized.

**Example**:
```python
def asymmetric_quantize(x, num_bits=8):
    """Asymmetric quantization"""
    qmin = 0
    qmax = 2**num_bits - 1  # 255 for UINT8

    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - round(x_min / scale)

    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax).astype(np.uint8)

    return q, scale, zero_point

# Example with positive-only activations (ReLU output)
x = np.array([0.2, 1.5, 0.8, 3.1])
q, scale, zp = asymmetric_quantize(x)
print(f"Original: {x}")
print(f"Quantized: {q}")
print(f"Scale: {scale}, Zero-point: {zp}")

# Dequantize
x_dequant = (q - zp) * scale
print(f"Dequantized: {x_dequant}")
```

## Why Quantization?

### Model Size Reduction

Quantization directly reduces model size by using fewer bits per parameter:

| Precision | Memory per Parameter | 7B Model Size | Reduction |
|-----------|---------------------|---------------|-----------|
| FP32 | 4 bytes | 28 GB | Baseline |
| FP16 | 2 bytes | 14 GB | 2× |
| INT8 | 1 byte | 7 GB | 4× |
| INT4 | 0.5 bytes | 3.5 GB | 8× |

**Example**: LLaMA-7B model:
- FP32: ~28 GB (unusable on consumer GPUs)
- INT8: ~7 GB (fits on RTX 3090)
- INT4: ~3.5 GB (runs on MacBook Pro)

### Inference Speed Improvement

Integer operations are significantly faster than floating-point:

| Operation | NVIDIA A100 Throughput | Speedup |
|-----------|----------------------|---------|
| FP32 | 19.5 TFLOPS | 1× |
| FP16 (Tensor Core) | 312 TFLOPS | 16× |
| INT8 (Tensor Core) | 624 TOPS | 32× |

**Memory Bandwidth**: Moving data is often the bottleneck
- INT8 requires 4× less memory bandwidth than FP32
- Critical for large models where compute is memory-bound

### Energy Efficiency

Lower precision = lower energy consumption:

| Operation | Energy (pJ) | Relative |
|-----------|-------------|----------|
| INT8 ADD | 0.03 | 1× |
| FP16 ADD | 0.4 | 13× |
| FP32 ADD | 0.9 | 30× |
| FP32 MULT | 3.7 | 123× |

Essential for:
- Mobile devices (battery life)
- Edge computing (power constraints)
- Data centers (operational costs)

### Edge Deployment

Many edge devices only support integer operations:
- ARM Cortex-M processors
- Google Edge TPU
- Qualcomm Hexagon DSP
- Apple Neural Engine

Quantization enables running sophisticated models on these devices.

## Types of Quantization

### Post-Training Quantization (PTQ)

Quantize a pre-trained model without retraining. Fast but may lose accuracy.

#### Dynamic Quantization

Quantizes weights statically, activations dynamically at runtime.

**Characteristics**:
- Weights: Quantized and stored as INT8
- Activations: Quantized on-the-fly during inference
- No calibration data needed
- Best for memory-bound models (LSTMs, Transformers)

**PyTorch Example**:
```python
import torch
import torch.quantization

# Original model
model = MyTransformer()
model.eval()

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
    dtype=torch.qint8
)

# Inference
with torch.no_grad():
    output = quantized_model(input_tensor)

# Check size reduction
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"Size reduction: {original_size / quantized_size:.2f}×")
```

**When to use**:
- Quick deployment without accuracy loss
- LSTM/Transformer models
- When activation distribution changes per input

#### Static Quantization

Quantizes both weights and activations using calibration data.

**Characteristics**:
- Weights: Pre-quantized to INT8
- Activations: Pre-computed scale/zero-point from calibration
- Requires representative calibration dataset
- Best for convolutional networks
- Maximum performance gain

**PyTorch Example**:
```python
import torch
import torch.quantization

# Prepare model for quantization
model = MyConvNet()
model.eval()

# Specify quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86 CPUs

# Fuse operations (Conv + BatchNorm + ReLU)
torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)

# Prepare for static quantization
torch.quantization.prepare(model, inplace=True)

# Calibration: Run representative data through model
with torch.no_grad():
    for batch in calibration_data_loader:
        model(batch)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)

# Save quantized model
torch.save(model.state_dict(), 'quantized_model.pth')

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

**Calibration Best Practices**:
```python
def calibrate_model(model, data_loader, num_batches=100):
    """
    Calibrate quantization parameters
    """
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(images)
    return model

# Use diverse calibration data
# 100-1000 samples usually sufficient
calibrated_model = calibrate_model(prepared_model, val_loader, num_batches=200)
```

### Quantization-Aware Training (QAT)

Simulates quantization during training to maintain accuracy.

**Characteristics**:
- Fake quantization in forward pass
- Full precision gradients in backward pass
- Highest accuracy for aggressive quantization
- Requires training time and data

**How it works**:
1. Forward pass: Apply quantization (fake quant nodes)
2. Compute loss with quantized values
3. Backward pass: Use straight-through estimators
4. Update weights in full precision

**PyTorch Example**:
```python
import torch
import torch.quantization

# Start with pre-trained model
model = MyModel()
model.load_state_dict(torch.load('pretrained.pth'))

# Set to training mode
model.train()

# Configure QAT
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Prepare for QAT
torch.quantization.prepare_qat(model, inplace=True)

# Fine-tune with quantization simulation
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5  # Fine-tuning epochs
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Convert to fully quantized model
model.eval()
torch.quantization.convert(model, inplace=True)

# Evaluate
accuracy = evaluate(model, test_loader)
print(f"Quantized model accuracy: {accuracy:.2f}%")
```

**Fake Quantization**:
```python
class FakeQuantize(torch.nn.Module):
    """Simulates quantization effects during training"""

    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.zero_point = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Quantize
        q = torch.clamp(
            torch.round(x / self.scale + self.zero_point),
            self.qmin, self.qmax
        )
        # Dequantize
        x_fake_quant = (q - self.zero_point) * self.scale
        return x_fake_quant
```

## Quantization Granularity

### Per-Tensor Quantization

Single scale/zero-point for entire tensor.

**Advantages**:
- Simpler implementation
- Faster computation
- Lower memory overhead

**Disadvantages**:
- Less accurate for tensors with wide value ranges
- Outliers affect entire tensor

```python
def per_tensor_quantize(tensor, num_bits=8):
    """Quantize entire tensor with single scale"""
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)

    q = torch.clamp(
        torch.round(tensor / scale + zero_point),
        qmin, qmax
    )
    return q, scale, zero_point
```

### Per-Channel Quantization

Different scale/zero-point per output channel.

**Advantages**:
- Higher accuracy, especially for convolutional layers
- Handles per-channel variance better

**Disadvantages**:
- More complex
- Requires hardware support

**Applied to**: Weights (not activations, due to hardware constraints)

```python
def per_channel_quantize(weight, num_bits=8):
    """
    Quantize per output channel (conv filters)
    weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    """
    out_channels = weight.shape[0]
    qmin, qmax = -(2**(num_bits-1)), 2**(num_bits-1) - 1

    scales = []
    zero_points = []
    q_weight = torch.zeros_like(weight, dtype=torch.int8)

    for ch in range(out_channels):
        ch_weight = weight[ch]
        ch_min, ch_max = ch_weight.min(), ch_weight.max()

        # Symmetric quantization per channel
        scale = max(abs(ch_min), abs(ch_max)) / qmax
        scales.append(scale)
        zero_points.append(0)

        q_weight[ch] = torch.clamp(
            torch.round(ch_weight / scale),
            qmin, qmax
        ).to(torch.int8)

    return q_weight, torch.tensor(scales), torch.tensor(zero_points)

# Example
conv_weight = torch.randn(64, 3, 3, 3)  # 64 filters
q_weight, scales, zps = per_channel_quantize(conv_weight)
print(f"Original shape: {conv_weight.shape}")
print(f"Quantized shape: {q_weight.shape}")
print(f"Scales per channel: {scales.shape}")
```

### Group Quantization

Quantize groups of channels together (compromise between per-tensor and per-channel).

```python
def group_quantize(weight, group_size=4, num_bits=4):
    """Group quantization for weights"""
    out_channels = weight.shape[0]
    num_groups = (out_channels + group_size - 1) // group_size

    scales = []
    q_weight = torch.zeros_like(weight, dtype=torch.int8)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, out_channels)
        group_weight = weight[start:end]

        scale = group_weight.abs().max() / (2**(num_bits-1) - 1)
        scales.append(scale)

        q_weight[start:end] = torch.round(group_weight / scale)

    return q_weight, torch.tensor(scales)
```

## Advanced Quantization Techniques

### Mixed Precision Quantization

Use different precision for different layers based on sensitivity.

**Strategy**:
1. Profile layer sensitivity to quantization
2. Keep sensitive layers in higher precision
3. Aggressively quantize insensitive layers

```python
def quantize_mixed_precision(model, sensitivity_dict):
    """
    Apply different quantization based on layer sensitivity
    sensitivity_dict: {layer_name: num_bits}
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in sensitivity_dict:
                bits = sensitivity_dict[name]
                if bits == 8:
                    # Standard INT8 quantization
                    quantize_layer(module, num_bits=8)
                elif bits == 4:
                    # Aggressive INT4 quantization
                    quantize_layer(module, num_bits=4)
                else:
                    # Keep in FP16
                    module.half()

# Example sensitivity analysis
def analyze_sensitivity(model, data_loader):
    """Measure accuracy drop per layer"""
    baseline_acc = evaluate(model, data_loader)
    sensitivity = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Temporarily quantize this layer
            original_weight = module.weight.data.clone()
            module.weight.data = quantize_dequantize(original_weight, num_bits=8)

            acc = evaluate(model, data_loader)
            sensitivity[name] = baseline_acc - acc

            # Restore
            module.weight.data = original_weight

    return sensitivity
```

### GPTQ (GPT Quantization)

Advanced post-training quantization for large language models using layer-wise quantization with Hessian information.

**Key Idea**: Minimize reconstruction error layer-by-layer using second-order information.

```python
# Using auto-gptq library
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure GPTQ
quantize_config = BaseQuantizeConfig(
    bits=4,  # INT4 quantization
    group_size=128,  # Group size for quantization
    desc_act=False,  # Activation order
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)

# Prepare calibration data
from datasets import load_dataset
calibration_data = load_dataset("c4", split="train[:1000]")

def prepare_calibration(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

calibration_dataset = calibration_data.map(prepare_calibration)

# Quantize
model.quantize(calibration_dataset)

# Save quantized model
model.save_quantized("./llama-7b-gptq-4bit")

# Load and use
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
quantized_model = AutoGPTQForCausalLM.from_quantized("./llama-7b-gptq-4bit")

# Generate
input_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids
output = quantized_model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0]))
```

**GPTQ Algorithm**:
1. Process model layer-by-layer
2. For each layer, use Hessian matrix to determine optimal quantization
3. Update weights to minimize reconstruction error
4. Use Cholesky decomposition for efficient computation

### AWQ (Activation-aware Weight Quantization)

Protects weights corresponding to important activations.

**Key Insight**: Not all weights are equally important. Weights that multiply with large activations are more critical.

```python
# Using AutoAWQ library
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval"  # Calibration dataset
)

# Save
model.save_quantized("./llama-7b-awq-4bit")

# Load and inference
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("./llama-7b-awq-4bit", fuse_layers=True)
```

**AWQ Method**:
1. Observe activation distributions
2. Scale weights based on activation magnitudes
3. Quantize scaled weights
4. Adjust scales to maintain equivalence

### SmoothQuant

Migrates quantization difficulty from activations to weights.

**Problem**: Activations often have larger outliers than weights, making them harder to quantize.

**Solution**: Apply mathematically equivalent transformations to smooth activations.

```python
def smooth_quant(weight, activation, alpha=0.5):
    """
    SmoothQuant transformation
    Y = (Xdiag(s)^(-1)) · (diag(s)W) = X · W
    where s = max(|X|)^α / max(|W|)^(1-α)
    """
    # Calculate smoothing scales
    activation_absmax = activation.abs().max(dim=0).values
    weight_absmax = weight.abs().max(dim=0).values

    scales = (activation_absmax ** alpha) / (weight_absmax ** (1 - alpha))

    # Apply smoothing
    smoothed_weight = weight * scales.unsqueeze(0)
    smoothed_activation = activation / scales.unsqueeze(0)

    return smoothed_weight, smoothed_activation, scales

# Integration with quantization
class SmoothQuantLinear(torch.nn.Module):
    def __init__(self, linear_layer, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.scales = None
        self.quantized_weight = None

    def calibrate(self, activations):
        """Calibrate smoothing scales"""
        self.scales = calculate_smooth_scales(
            self.weight, activations, self.alpha
        )
        smoothed_weight = self.weight * self.scales
        self.quantized_weight = quantize(smoothed_weight)

    def forward(self, x):
        smoothed_x = x / self.scales
        return F.linear(smoothed_x, self.quantized_weight)
```

### LLM.int8()

Decomposes matrix multiplication into INT8 and FP16 components.

**Key Idea**: Most values can be quantized to INT8, but rare outliers are kept in FP16.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure LLM.int8()
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Outlier threshold
    llm_int8_has_fp16_weight=False
)

# Load model with INT8 quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# Model automatically uses INT8 for most operations
# Outliers are processed in FP16
output = model.generate(input_ids, max_length=100)
```

**How it works**:
1. Identify outlier features (magnitude > threshold)
2. Separate into two matrix multiplications:
   - Regular features: INT8 × INT8
   - Outlier features: FP16 × FP16
3. Combine results

### 4-bit Quantization with NormalFloat (NF4)

Introduced in QLoRA, optimized for normally distributed weights.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
    bnb_4bit_use_double_quant=True,  # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=nf4_config,
    device_map="auto"
)

# Can even fine-tune in 4-bit with LoRA
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train with 4-bit base model + 16-bit LoRA adapters
trainer.train()
```

**NF4 Quantization Bins**: Optimized for Gaussian distributions
```python
# NF4 quantization levels (non-uniform)
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495,
    0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
]
```

## Quantization for Different Architectures

### Convolutional Neural Networks (CNNs)

CNNs are relatively robust to quantization due to:
- Spatial redundancy in image data
- Batch normalization stabilization
- ReLU activations (non-negative, easier to quantize)

**Best Practices**:
```python
def quantize_cnn(model):
    """Quantize CNN model"""
    # 1. Fuse operations
    torch.quantization.fuse_modules(
        model,
        [['conv1', 'bn1', 'relu']],
        inplace=True
    )

    # 2. Use per-channel quantization for conv layers
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.default_per_channel_weight_observer
    )

    # 3. First and last layers: keep higher precision or use symmetric
    # model.conv1.qconfig = custom_qconfig_fp16
    # model.fc.qconfig = custom_qconfig_fp16

    return model

# Layer fusion example
model = models.resnet18(pretrained=True)
model.eval()

# Fuse Conv-BN-ReLU
fused_model = torch.quantization.fuse_modules(
    model,
    [
        ['conv1', 'bn1', 'relu'],
        ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'],
        # ... more layers
    ]
)
```

**Quantization-friendly Architecture**:
```python
class QuantizableMobileNetV2(nn.Module):
    """MobileNetV2 designed for quantization"""

    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Use quantization-friendly operations
        self.features = nn.Sequential(
            # Depthwise separable convolutions
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # ... more layers
        )
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)  # Dequantize output
        return x
```

### Transformers and Large Language Models

Transformers are more sensitive to quantization due to:
- Attention mechanisms with softmax (outliers)
- Layer normalization
- Large embedding tables
- Accumulated errors over many layers

**Challenges**:
1. **Outlier features**: Some dimensions have extreme values
2. **Embedding tables**: Large memory footprint
3. **Attention scores**: Sensitive to precision

**Solutions**:
```python
# 1. Layer-wise quantization sensitivity
def quantize_transformer_selective(model):
    """Selectively quantize transformer components"""
    for name, module in model.named_modules():
        if 'attention' in name:
            # Keep attention in higher precision
            module.qconfig = get_qconfig_fp16()
        elif 'mlp' in name or 'feed_forward' in name:
            # Aggressively quantize feed-forward
            module.qconfig = get_qconfig_int8()
        elif 'layernorm' in name:
            # Keep normalization in FP16
            module.qconfig = None

# 2. Quantize with outlier handling
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,  # Uses LLM.int8()
    device_map="auto",
    max_memory={0: "20GB", "cpu": "30GB"}
)

# 3. K-V cache quantization for faster inference
class QuantizedAttention(nn.Module):
    """Attention with quantized K-V cache"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kv_bits = 8  # Quantize cached keys/values

    def forward(self, hidden_states, past_key_value=None):
        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Quantize K, V for caching
        if self.training:
            # During training, use FP
            past_key_value = (key, value)
        else:
            # During inference, quantize K-V cache
            key_q, key_scale = quantize_tensor(key, self.kv_bits)
            value_q, value_scale = quantize_tensor(value, self.kv_bits)
            past_key_value = (key_q, key_scale, value_q, value_scale)

        # Attention computation...
        return output, past_key_value
```

**GPTQ for LLMs**:
```python
# Comprehensive GPTQ quantization
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=True,  # Better accuracy
    sym=False,  # Asymmetric quantization
    true_sequential=True,  # Sequential quantization
    model_name_or_path=None,
    model_file_base_name="model"
)

# Quantize
model.quantize(
    examples=calibration_data,
    batch_size=1,
    use_triton=True,  # Faster with Triton kernels
    autotune_warmup_after_quantized=True
)
```

### Vision Transformers (ViT)

Combine challenges of both CNNs and Transformers:

```python
def quantize_vit(model, quantize_attention=False):
    """Quantize Vision Transformer"""
    for name, module in model.named_modules():
        if 'patch_embed' in name:
            # Patch embedding: keep higher precision
            module.qconfig = get_qconfig_fp16()
        elif 'attn' in name and not quantize_attention:
            # Attention: conditional quantization
            module.qconfig = None
        elif 'mlp' in name:
            # MLP blocks: aggressive INT8
            module.qconfig = get_qconfig_int8()

    return model

# PTQ for ViT
def ptq_vision_transformer(model, calibration_loader):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Selectively quantize
    quantize_vit(model, quantize_attention=False)

    # Prepare
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with image data
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images)

    # Convert
    torch.quantization.convert(model, inplace=True)
    return model
```

### Recurrent Neural Networks (RNNs/LSTMs)

RNNs benefit significantly from dynamic quantization:

```python
# Dynamic quantization for LSTM
model = nn.LSTM(input_size=256, hidden_size=512, num_layers=2)

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.LSTM, nn.Linear},
    dtype=torch.qint8
)

# For static quantization of RNNs (more complex)
class QuantizableLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x, hidden=None):
        x = self.quant(x)
        output, hidden = self.lstm(x, hidden)
        output = self.dequant(output)
        return output, hidden
```

## Practical Implementation Examples

### Example 1: Quantizing ResNet for Image Classification

```python
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 1. Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# 2. Prepare data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

calibration_dataset = datasets.ImageFolder('imagenet/val', transform=transform)
calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 3. Fuse modules
model.fuse_model()  # Fuse Conv-BN-ReLU

# 4. Set quantization config
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 5. Prepare for calibration
torch.quantization.prepare(model, inplace=True)

# 6. Calibrate
print("Calibrating...")
num_calibration_batches = 100
with torch.no_grad():
    for i, (images, _) in enumerate(calibration_loader):
        if i >= num_calibration_batches:
            break
        model(images)
        if (i + 1) % 10 == 0:
            print(f"Calibrated {i + 1} batches")

# 7. Convert to quantized model
torch.quantization.convert(model, inplace=True)

# 8. Evaluate
def evaluate(model, data_loader, num_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if num_batches and i >= num_batches:
                break
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

print("Evaluating quantized model...")
accuracy = evaluate(model, calibration_loader, num_batches=200)
print(f"Quantized model accuracy: {accuracy:.2f}%")

# 9. Save quantized model
torch.save(model.state_dict(), 'resnet50_quantized.pth')

# 10. Compare model sizes
def print_model_size(model, label):
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / 1e6
    print(f"{label}: {size_mb:.2f} MB")
    os.remove("temp.pth")

original_model = models.resnet50(pretrained=True)
print_model_size(original_model, "Original FP32")
print_model_size(model, "Quantized INT8")
```

### Example 2: QAT for Custom Model

```python
import torch
import torch.nn as nn
import torch.quantization

class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'],
             ['conv2', 'bn2', 'relu2']],
            inplace=True
        )

# 1. Train FP32 model first
model = CustomModel(num_classes=10)
# ... training code ...
torch.save(model.state_dict(), 'model_fp32.pth')

# 2. Prepare for QAT
model.load_state_dict(torch.load('model_fp32.pth'))
model.train()

# Fuse layers
model.fuse_model()

# Set QAT config
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Prepare QAT
torch.quantization.prepare_qat(model, inplace=True)

# 3. Fine-tune with QAT
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    # Validation
    model.eval()
    val_acc = evaluate(model, val_loader)
    print(f'Epoch {epoch}, Validation Accuracy: {val_acc:.2f}%')

# 4. Convert to fully quantized model
model.eval()
torch.quantization.convert(model, inplace=True)

# 5. Final evaluation
test_acc = evaluate(model, test_loader)
print(f'Quantized model test accuracy: {test_acc:.2f}%')

# 6. Save
torch.save(model.state_dict(), 'model_qat_int8.pth')
```

### Example 3: Quantizing BERT for NLP

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# 1. Load model
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 2. Dynamic quantization (easiest for transformers)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# 3. Test inference
text = "This movie was fantastic! I loved every minute of it."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    # Original model
    output_fp32 = model(**inputs)
    # Quantized model
    output_int8 = quantized_model(**inputs)

print("FP32 logits:", output_fp32.logits)
print("INT8 logits:", output_int8.logits)

# 4. Compare sizes
def get_model_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / 1e6
    os.remove("temp.pth")
    return size

fp32_size = get_model_size(model)
int8_size = get_model_size(quantized_model)

print(f"FP32 model: {fp32_size:.2f} MB")
print(f"INT8 model: {int8_size:.2f} MB")
print(f"Compression ratio: {fp32_size / int8_size:.2f}×")

# 5. Benchmark inference speed
import time

def benchmark(model, inputs, num_runs=100):
    # Warmup
    for _ in range(10):
        model(**inputs)

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(**inputs)
    end = time.time()

    return (end - start) / num_runs

fp32_time = benchmark(model, inputs)
int8_time = benchmark(quantized_model, inputs)

print(f"FP32 inference: {fp32_time*1000:.2f} ms")
print(f"INT8 inference: {int8_time*1000:.2f} ms")
print(f"Speedup: {fp32_time / int8_time:.2f}×")
```

### Example 4: 4-bit LLM Quantization with bitsandbytes

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype
)

# 2. Load model in 4-bit
model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across GPUs
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Generate text
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 4. Memory usage
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

# 5. Can even fine-tune with QLoRA
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Prepare for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Now you can fine-tune with standard training loop
# Only LoRA adapters are trained (in FP32/BF16)
# Base model stays in 4-bit
```

## Performance Analysis and Benchmarking

### Measuring Quantization Impact

```python
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score

class QuantizationBenchmark:
    """Comprehensive quantization benchmarking"""

    def __init__(self, model_fp32, model_quantized, test_loader):
        self.model_fp32 = model_fp32
        self.model_quantized = model_quantized
        self.test_loader = test_loader

    def measure_accuracy(self, model, num_batches=None):
        """Measure model accuracy"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                if num_batches and i >= num_batches:
                    break
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds) * 100

    def measure_latency(self, model, num_runs=100):
        """Measure inference latency"""
        model.eval()

        # Get a sample batch
        sample_input, _ = next(iter(self.test_loader))

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }

    def measure_throughput(self, model, duration=10):
        """Measure throughput (samples/sec)"""
        model.eval()
        sample_input, _ = next(iter(self.test_loader))
        batch_size = sample_input.size(0)

        num_batches = 0
        start = time.time()

        with torch.no_grad():
            while time.time() - start < duration:
                _ = model(sample_input)
                num_batches += 1

        elapsed = time.time() - start
        throughput = (num_batches * batch_size) / elapsed
        return throughput

    def measure_model_size(self, model):
        """Measure model size in MB"""
        torch.save(model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / 1e6
        os.remove("temp_model.pth")
        return size_mb

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("=" * 60)
        print("Quantization Benchmark Results")
        print("=" * 60)

        # Accuracy
        print("\n[1] Accuracy")
        fp32_acc = self.measure_accuracy(self.model_fp32)
        quant_acc = self.measure_accuracy(self.model_quantized)
        print(f"  FP32:      {fp32_acc:.2f}%")
        print(f"  Quantized: {quant_acc:.2f}%")
        print(f"  Drop:      {fp32_acc - quant_acc:.2f}%")

        # Model Size
        print("\n[2] Model Size")
        fp32_size = self.measure_model_size(self.model_fp32)
        quant_size = self.measure_model_size(self.model_quantized)
        print(f"  FP32:      {fp32_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Reduction: {fp32_size / quant_size:.2f}×")

        # Latency
        print("\n[3] Latency (ms)")
        fp32_latency = self.measure_latency(self.model_fp32)
        quant_latency = self.measure_latency(self.model_quantized)
        print(f"  FP32:      {fp32_latency['mean']:.2f} ± {fp32_latency['std']:.2f}")
        print(f"  Quantized: {quant_latency['mean']:.2f} ± {quant_latency['std']:.2f}")
        print(f"  Speedup:   {fp32_latency['mean'] / quant_latency['mean']:.2f}×")

        # Throughput
        print("\n[4] Throughput (samples/sec)")
        fp32_throughput = self.measure_throughput(self.model_fp32)
        quant_throughput = self.measure_throughput(self.model_quantized)
        print(f"  FP32:      {fp32_throughput:.2f}")
        print(f"  Quantized: {quant_throughput:.2f}")
        print(f"  Improvement: {quant_throughput / fp32_throughput:.2f}×")

        print("\n" + "=" * 60)

        return {
            'accuracy': {'fp32': fp32_acc, 'quantized': quant_acc},
            'size': {'fp32': fp32_size, 'quantized': quant_size},
            'latency': {'fp32': fp32_latency, 'quantized': quant_latency},
            'throughput': {'fp32': fp32_throughput, 'quantized': quant_throughput}
        }

# Usage
benchmark = QuantizationBenchmark(model_fp32, model_int8, test_loader)
results = benchmark.run_full_benchmark()
```

### Profiling Quantization Errors

```python
def analyze_quantization_error(model_fp32, model_quantized, data_loader):
    """Analyze per-layer quantization errors"""

    # Hook to capture activations
    activations_fp32 = {}
    activations_quant = {}

    def get_activation(name, storage):
        def hook(model, input, output):
            storage[name] = output.detach()
        return hook

    # Register hooks
    for name, module in model_fp32.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(get_activation(name, activations_fp32))

    for name, module in model_quantized.named_modules():
        if isinstance(module, (nn.quantized.Conv2d, nn.quantized.Linear)):
            module.register_forward_hook(get_activation(name, activations_quant))

    # Run inference
    sample_input, _ = next(iter(data_loader))
    with torch.no_grad():
        _ = model_fp32(sample_input)
        _ = model_quantized(sample_input)

    # Compute errors
    errors = {}
    for name in activations_fp32:
        if name in activations_quant:
            fp32_act = activations_fp32[name]
            quant_act = activations_quant[name].dequantize() if hasattr(
                activations_quant[name], 'dequantize'
            ) else activations_quant[name]

            mse = torch.mean((fp32_act - quant_act) ** 2).item()
            mae = torch.mean(torch.abs(fp32_act - quant_act)).item()
            relative_error = mae / (torch.mean(torch.abs(fp32_act)).item() + 1e-8)

            errors[name] = {
                'mse': mse,
                'mae': mae,
                'relative_error': relative_error
            }

    # Print results
    print("\nPer-Layer Quantization Error Analysis:")
    print(f"{'Layer':<40} {'MSE':<15} {'MAE':<15} {'Relative Error'}")
    print("-" * 80)
    for name, err in sorted(errors.items(), key=lambda x: x[1]['relative_error'], reverse=True):
        print(f"{name:<40} {err['mse']:<15.6f} {err['mae']:<15.6f} {err['relative_error']:.4f}")

    return errors
```

## Common Challenges and Solutions

### Challenge 1: Accuracy Degradation

**Problem**: Quantized model has significantly lower accuracy.

**Solutions**:

1. **Use QAT instead of PTQ**:
```python
# If PTQ gives poor accuracy, switch to QAT
model.train()
torch.quantization.prepare_qat(model, inplace=True)
# Fine-tune for 3-5 epochs
```

2. **Increase calibration data**:
```python
# Use more diverse calibration samples
num_calibration_batches = 1000  # Instead of 100
```

3. **Mixed precision**:
```python
# Keep sensitive layers in higher precision
for name, module in model.named_modules():
    if 'attention' in name or name == 'classifier':
        module.qconfig = fp16_qconfig
```

4. **Per-channel quantization**:
```python
# Use per-channel for weights
model.qconfig = torch.quantization.QConfig(
    activation=default_observer,
    weight=per_channel_weight_observer  # More accurate
)
```

### Challenge 2: Outliers in Activations

**Problem**: Few extreme values dominate quantization range.

**Solutions**:

1. **Clip outliers**:
```python
class ClippedObserver(torch.quantization.MinMaxObserver):
    def __init__(self, percentile=99.9, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    def forward(self, x_orig):
        x = x_orig.detach()
        min_val = torch.quantile(x, (100 - self.percentile) / 100)
        max_val = torch.quantile(x, self.percentile / 100)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig
```

2. **SmoothQuant approach**:
```python
# Migrate difficulty from activations to weights
smoothed_weight, smoothed_activation = smooth_quant(
    weight, activation, alpha=0.5
)
```

3. **Mixed INT8/FP16** (LLM.int8()):
```python
# Process outliers separately in FP16
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0  # Outlier threshold
)
```

### Challenge 3: Batch Normalization Issues

**Problem**: Batch norm statistics change after quantization.

**Solutions**:

1. **Fuse BN with Conv**:
```python
# Always fuse before quantization
torch.quantization.fuse_modules(
    model,
    [['conv', 'bn', 'relu']],
    inplace=True
)
```

2. **Recalibrate BN**:
```python
def recalibrate_bn(model, data_loader, num_batches=100):
    """Recalculate BN statistics after quantization"""
    model.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            model(inputs)
    model.eval()
    return model
```

### Challenge 4: First/Last Layer Sensitivity

**Problem**: First and last layers are often more sensitive to quantization.

**Solution**: Keep them in higher precision
```python
def selective_quantization(model):
    """Quantize all layers except first and last"""
    # Set default config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Override first layer
    model.conv1.qconfig = None  # Keep FP32

    # Override last layer
    model.fc.qconfig = None  # Keep FP32

    return model
```

### Challenge 5: Hardware-Specific Issues

**Problem**: Quantized model doesn't run efficiently on target hardware.

**Solutions**:

1. **Use appropriate backend**:
```python
# For x86 CPUs
qconfig = torch.quantization.get_default_qconfig('fbgemm')

# For ARM CPUs
qconfig = torch.quantization.get_default_qconfig('qnnpack')
```

2. **Ensure operator support**:
```python
# Check if operator is supported
from torch.quantization import get_default_qconfig_propagation_list
supported_ops = get_default_qconfig_propagation_list()
```

3. **Use framework-specific quantization**:
```python
# For mobile deployment
from torch.utils.mobile_optimizer import optimize_for_mobile

quantized_model = quantize_dynamic(model)
scripted_model = torch.jit.script(quantized_model)
optimized_model = optimize_for_mobile(scripted_model)
```

## Hardware Considerations

### CPU Quantization

**x86 CPUs** (Intel/AMD):
- Use `fbgemm` backend
- INT8 via VNNI (Vector Neural Network Instructions) on modern CPUs
- Best for server deployments

```python
# Configure for x86
import torch.backends.quantized as quantized_backends
quantized_backends.engine = 'fbgemm'

qconfig = torch.quantization.get_default_qconfig('fbgemm')
```

**ARM CPUs**:
- Use `qnnpack` backend
- Optimized for mobile devices
- Supports NEON instructions

```python
# Configure for ARM
torch.backends.quantized.engine = 'qnnpack'

qconfig = torch.quantization.get_default_qconfig('qnnpack')
```

### GPU Quantization

**NVIDIA GPUs**:
- Tensor Cores support INT8/INT4
- TensorRT for deployment
- Significant speedup for INT8

```python
# Using TensorRT via torch2trt
from torch2trt import torch2trt

# Create quantized model
x = torch.ones((1, 3, 224, 224)).cuda()
model_trt = torch2trt(
    model,
    [x],
    fp16_mode=False,
    int8_mode=True,
    int8_calib_dataset=calibration_dataset
)
```

### Mobile/Edge Devices

**TensorFlow Lite** for mobile:
```python
import tensorflow as tf

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Full integer quantization
def representative_dataset():
    for data in calibration_data:
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

**ONNX Runtime**:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

model_input = 'model.onnx'
model_output = 'model_quantized.onnx'

quantize_dynamic(
    model_input,
    model_output,
    weight_type=QuantType.QInt8
)
```

**CoreML** for iOS:
```python
import coremltools as ct

# Convert PyTorch to CoreML with quantization
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    convert_to="neuralnetwork",
    minimum_deployment_target=ct.target.iOS14
)

# Quantize to INT8
model_int8 = ct.quantize_weights(coreml_model, nbits=8)
model_int8.save("model_quantized.mlmodel")
```

## Tools and Libraries

### PyTorch Quantization
```python
import torch.quantization
# Built-in, well-integrated with PyTorch ecosystem
# Supports dynamic, static, and QAT
```

### TensorFlow/TFLite
```python
import tensorflow as tf
# Excellent mobile support via TFLite
# Supports post-training and QAT
```

### ONNX Runtime
```python
from onnxruntime.quantization import quantize_dynamic
# Framework-agnostic
# Good for cross-platform deployment
```

### bitsandbytes
```python
import bitsandbytes as bnb
# Specialized for LLMs
# Supports 4-bit, 8-bit quantization
# LLM.int8() and NF4
```

### Auto-GPTQ
```python
from auto_gptq import AutoGPTQForCausalLM
# State-of-the-art LLM quantization
# GPTQ algorithm implementation
```

### AutoAWQ
```python
from awq import AutoAWQForCausalLM
# Activation-aware quantization
# Often better than GPTQ for inference
```

### Intel Neural Compressor
```python
from neural_compressor import Quantization
# Comprehensive quantization toolkit
# Supports multiple frameworks
```

### NVIDIA TensorRT
```python
import tensorrt as trt
# High-performance inference
# INT8/FP16 optimization
```

## Best Practices

1. **Start with Dynamic Quantization**
   - Easiest to implement
   - No calibration needed
   - Good baseline

2. **Calibration Data Quality**
   - Use representative data
   - 100-1000 samples usually sufficient
   - Diverse coverage of input distribution

3. **Layer-wise Sensitivity Analysis**
   - Identify sensitive layers
   - Keep them in higher precision
   - Aggressively quantize insensitive layers

4. **Fuse Operations**
   - Always fuse Conv-BN-ReLU
   - Reduces quantization error
   - Improves performance

5. **Measure Everything**
   - Accuracy
   - Latency
   - Throughput
   - Model size
   - Memory usage

6. **Target Hardware Matters**
   - Use appropriate backend (fbgemm/qnnpack)
   - Test on actual deployment hardware
   - Profile performance

7. **Quantization-Aware Architecture**
   - Avoid operations that don't quantize well
   - Use ReLU6 instead of other activations
   - Consider architecture during design

8. **Version Control Quantized Models**
   - Track quantization configs
   - Document calibration process
   - Maintain reproducibility

## Resources and Papers

### Foundational Papers

1. **"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"**
   - Jacob et al., 2018
   - Introduced per-channel quantization and fake quantization

2. **"A Survey of Quantization Methods for Efficient Neural Network Inference"**
   - Gholami et al., 2021
   - Comprehensive overview of quantization techniques

3. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"**
   - Dettmers et al., 2022
   - Outlier-aware quantization for LLMs

4. **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"**
   - Frantar et al., 2023
   - State-of-the-art PTQ for LLMs

5. **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**
   - Lin et al., 2023
   - Protects salient weights

6. **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**
   - Xiao et al., 2023
   - Smooths activation outliers

7. **"QLoRA: Efficient Finetuning of Quantized LLMs"**
   - Dettmers et al., 2023
   - 4-bit quantization with LoRA fine-tuning

### Tutorials and Guides

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [TensorFlow Lite Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

### Libraries and Tools

- PyTorch: `torch.quantization`
- TensorFlow: `tf.quantization`, TFLite
- ONNX Runtime: `onnxruntime.quantization`
- bitsandbytes: `bitsandbytes`
- Auto-GPTQ: `auto-gptq`
- AutoAWQ: `autoawq`
- Intel Neural Compressor: `neural-compressor`

### Datasets for Calibration

- ImageNet (computer vision)
- C4, WikiText (language models)
- COCO (object detection)
- Custom domain-specific data (recommended)

## Summary

Quantization is an essential technique for deploying neural networks efficiently:

- **Reduces model size** by 4-8× (INT8, INT4)
- **Increases inference speed** by 2-4× on appropriate hardware
- **Enables edge deployment** on resource-constrained devices
- **Maintains accuracy** with proper techniques (QAT, calibration)

**Key Takeaways**:
1. Choose quantization method based on constraints (time, accuracy, hardware)
2. Dynamic quantization: quickest start, good for RNNs/Transformers
3. Static quantization: best performance for CNNs
4. QAT: highest accuracy for aggressive quantization
5. Modern LLMs: GPTQ, AWQ, or bitsandbytes for 4-bit quantization
6. Always measure: accuracy, latency, model size, throughput
7. Hardware matters: use appropriate backend and test on target device

Quantization transforms impractical models into deployable solutions, making AI accessible on everything from smartphones to data centers.
