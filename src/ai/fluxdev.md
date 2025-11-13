# Flux.1 - Black Forest Labs

Complete guide to Flux.1, the next-generation image generation model from the creators of Stable Diffusion.

## Table of Contents
- [Introduction](#introduction)
- [Model Variants](#model-variants)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Prompt Engineering](#prompt-engineering)
- [Parameters](#parameters)
- [Comparison with Other Models](#comparison-with-other-models)
- [Advanced Techniques](#advanced-techniques)
- [Optimization](#optimization)

## Introduction

Flux.1 is a state-of-the-art image generation model developed by Black Forest Labs, the team behind the original Stable Diffusion. Released in 2024, it represents a significant advancement in image quality, prompt adherence, and detail preservation.

### Key Features

- **Superior Image Quality**: Enhanced detail and realism
- **Better Prompt Understanding**: More accurate interpretation
- **Improved Text Rendering**: Readable text in images
- **Flexible Architecture**: Multiple variants for different needs
- **Advanced Control**: Fine-grained control over generation
- **Fast Inference**: Optimized for speed

### Model Architecture

- **Flow Matching**: Advanced diffusion technique
- **Hybrid Architecture**: Combines transformer and diffusion
- **12B Parameters**: Larger than SD models
- **Parallel Attention**: Efficient processing
- **Rotation Position Embeddings (RoPE)**: Better spatial understanding

## Model Variants

### Flux.1 [pro]

**Commercial, API-only**

```python
import requests

API_URL = "https://api.bfl.ml/v1/flux-pro"
API_KEY = "your-api-key"

def generate_flux_pro(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 30
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    return response.json()

# Generate
result = generate_flux_pro(
    "a professional photograph of a modern office, natural lighting, detailed"
)
```

**Features:**
- Highest quality
- Best prompt adherence
- Commercial use allowed
- API access only
- Pay per generation

**Best for:**
- Professional work
- Commercial projects
- Maximum quality needs

### Flux.1 [dev]

**Non-commercial, open-weight**

```python
import torch
from diffusers import FluxPipeline

# Load model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Generate
prompt = "a majestic lion in the savanna at sunset, highly detailed"
image = pipe(
    prompt,
    guidance_scale=3.5,
    num_inference_steps=30,
    height=1024,
    width=1024,
).images[0]

image.save("flux_output.png")
```

**Features:**
- High quality
- Open weights
- Non-commercial license
- Requires Hugging Face auth
- Can run locally

**Requirements:**
- GPU: 24GB+ VRAM (recommended)
- RAM: 32GB+ system RAM
- Storage: ~30GB for model

**Best for:**
- Research and development
- Personal projects
- Learning and experimentation

### Flux.1 [schnell]

**Apache 2.0 license, fastest**

```python
from diffusers import FluxPipeline
import torch

# Load schnell variant
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Fast generation (1-4 steps)
prompt = "a portrait of a person, professional photography"
image = pipe(
    prompt,
    num_inference_steps=4,  # Very few steps needed
    guidance_scale=0.0,  # No guidance needed
    height=1024,
    width=1024,
).images[0]

image.save("schnell_output.png")
```

**Features:**
- Very fast (1-4 steps)
- Good quality
- Apache 2.0 license
- Commercial use allowed
- Lower VRAM requirements

**Best for:**
- Real-time applications
- High-volume generation
- Commercial projects
- Resource-constrained environments

## Installation & Setup

### Option 1: Diffusers (Recommended)

```bash
# Install dependencies
pip install diffusers transformers accelerate torch

# Install from latest
pip install git+https://github.com/huggingface/diffusers.git
```

```python
from diffusers import FluxPipeline
import torch

# Authenticate with Hugging Face
from huggingface_hub import login
login(token="your_hf_token")

# Load model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # Save VRAM

# Generate
image = pipe("a beautiful landscape").images[0]
```

### Option 2: ComfyUI

```bash
# Update ComfyUI
cd ComfyUI
git pull

# Download Flux models to:
# models/unet/flux1-dev.safetensors
# models/unet/flux1-schnell.safetensors

# Download CLIP and T5 encoders to:
# models/clip/clip_l.safetensors
# models/clip/t5xxl_fp16.safetensors

# Download VAE to:
# models/vae/ae.safetensors
```

### Option 3: AUTOMATIC1111 (via extension)

```bash
cd extensions
git clone https://github.com/XLabs-AI/x-flux-comfyui.git
# Restart WebUI
```

### Hardware Requirements

| Variant | Minimum VRAM | Recommended VRAM | Storage |
|---------|--------------|------------------|---------|
| Schnell | 12GB | 16GB | 30GB |
| Dev | 16GB | 24GB | 30GB |
| Pro | N/A (API) | N/A (API) | N/A |

**Optimizations:**
- bfloat16: Reduces VRAM by ~50%
- CPU offload: Reduces VRAM usage further
- Quantization: 8-bit or 4-bit for lower VRAM

## Usage

### Basic Generation

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Simple generation
prompt = "a serene mountain lake at sunrise"
image = pipe(prompt).images[0]
image.save("output.png")
```

### With Parameters

```python
image = pipe(
    prompt="a futuristic city with flying cars, neon lights, cyberpunk",
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=3.5,
    max_sequence_length=256,
).images[0]
```

### Batch Generation

```python
# Multiple images from one prompt
images = pipe(
    prompt="a cute cat",
    num_images_per_prompt=4,
    num_inference_steps=30,
).images

for i, img in enumerate(images):
    img.save(f"cat_{i}.png")
```

### Seed Control

```python
# Fixed seed for reproducibility
generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt="a magical forest",
    generator=generator,
    num_inference_steps=30,
).images[0]
```

### Memory-Efficient Generation

```python
# For lower VRAM
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Generate
image = pipe(
    prompt="a detailed landscape",
    height=1024,
    width=1024,
).images[0]
```

## Prompt Engineering

### Prompt Structure

Flux.1 has excellent prompt understanding. Use natural language:

```python
# Simple and effective
prompt = "a portrait of a woman with red hair, wearing a blue dress, in a garden"

# Detailed
prompt = """
a professional photograph of a young woman with flowing red hair, 
wearing an elegant blue silk dress, standing in a lush garden 
with blooming roses, soft natural lighting, golden hour, 
depth of field, bokeh background, shot on Canon EOS R5
"""

# With style
prompt = """
oil painting of a medieval knight in full armor, 
standing on a cliff overlooking the ocean at sunset,
dramatic lighting, renaissance art style, 
highly detailed, masterpiece
"""
```

### Natural Language

Flux excels with conversational prompts:

```python
prompts = [
    "Show me a cat wearing sunglasses at the beach",
    "Create an image of a steampunk airship flying over Victorian London",
    "Paint a serene Japanese garden in autumn with falling maple leaves",
    "Design a futuristic sports car that looks fast even when standing still"
]
```

### Text in Images

Flux.1 can render text (unlike most other models):

```python
# Text rendering
prompt = '''
a modern cafe storefront with a neon sign that says "COFFEE SHOP",
rainy evening, reflections on wet pavement, cinematic lighting
'''

# Book cover
prompt = '''
a fantasy book cover with the title "The Dragon's Tale" 
written in elegant golden letters at the top,
featuring a majestic dragon flying over mountains
'''

# Product mockup
prompt = '''
a white t-shirt with the text "FLUX.1" printed in bold black letters,
product photography, plain white background, professional lighting
'''
```

### Aspect Ratios

```python
# Portrait
image = pipe(prompt, height=1344, width=768).images[0]

# Landscape
image = pipe(prompt, height=768, width=1344).images[0]

# Square
image = pipe(prompt, height=1024, width=1024).images[0]

# Cinematic
image = pipe(prompt, height=576, width=1024).images[0]

# Ultra-wide
image = pipe(prompt, height=512, width=1536).images[0]
```

### Prompt Tips

1. **Be Specific**: More detail = better results
2. **Natural Language**: Write as you would describe to a person
3. **Quality Terms**: "professional", "detailed", "high quality"
4. **Style References**: "photograph", "oil painting", "digital art"
5. **Lighting**: "golden hour", "dramatic lighting", "soft light"
6. **Camera/Lens**: "50mm lens", "wide angle", "macro"

### Example Prompts

```python
# Photorealistic
prompt = """
a cinematic photograph of a lone astronaut standing on mars, 
red desert landscape, distant sun on horizon, 
dust particles in air, dramatic lighting, 
shot on ARRI Alexa, anamorphic lens
"""

# Artistic
prompt = """
watercolor painting of a coastal village, 
Mediterranean architecture, boats in harbor, 
soft pastel colors, impressionist style, 
painted by Claude Monet
"""

# Product
prompt = """
professional product photography of a luxury watch, 
silver metal band, blue dial face, 
on marble surface with dramatic side lighting, 
reflections, 8k resolution, advertising quality
"""

# Character
prompt = """
character design of a cyberpunk hacker, 
purple mohawk, neon goggles, leather jacket with patches, 
detailed facial features, full body illustration, 
concept art style, trending on artstation
"""

# Architecture
prompt = """
modern minimalist house in forest setting, 
large glass windows, wooden exterior, 
surrounded by tall pine trees, morning mist, 
architectural photography, professional real estate photo
"""
```

## Parameters

### num_inference_steps

Number of denoising steps:

```python
# Schnell: 1-4 steps (optimized for speed)
image = pipe(prompt, num_inference_steps=4).images[0]

# Dev: 20-50 steps (balance)
image = pipe(prompt, num_inference_steps=30).images[0]

# Pro: API manages automatically
```

**Recommendations:**
- Schnell: 1-4 (4 recommended)
- Dev: 20-30 (30 recommended)
- More steps = better quality but slower

### guidance_scale

How closely to follow the prompt:

```python
# Schnell: 0.0 (no guidance needed)
image = pipe(prompt, guidance_scale=0.0).images[0]

# Dev: 3.0-5.0 (3.5 recommended)
image = pipe(prompt, guidance_scale=3.5).images[0]
```

**Flux uses lower guidance than SD:**
- SD typical: 7-10
- Flux typical: 3-5

### max_sequence_length

Token limit for prompt:

```python
# Standard
image = pipe(prompt, max_sequence_length=256).images[0]

# Long prompts
image = pipe(prompt, max_sequence_length=512).images[0]
```

### Resolution

```python
# Standard resolutions (in pixels)
resolutions = {
    "square": (1024, 1024),
    "portrait": (768, 1344),
    "landscape": (1344, 768),
    "wide": (1536, 640),
    "tall": (640, 1536),
}

# Use
image = pipe(
    prompt,
    height=resolutions["landscape"][0],
    width=resolutions["landscape"][1]
).images[0]
```

**Notes:**
- Keep dimensions divisible by 16
- Total pixels should be ~1MP for best results
- Higher resolutions need more VRAM

## Comparison with Other Models

### Flux.1 vs Stable Diffusion

| Feature | Flux.1 | SD 1.5 | SDXL |
|---------|--------|--------|------|
| Image Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Prompt Adherence | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Text Rendering | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| Speed (Dev) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed (Schnell) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| VRAM Usage | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Ecosystem | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| License | Varies | Open | Open |

### Quality Comparison

```python
# Same prompt across models
prompt = "a detailed portrait of a person with glasses"

# Flux.1 Dev
flux_image = flux_pipe(prompt, num_inference_steps=30).images[0]
# Result: High detail, accurate glasses, natural lighting

# SDXL
sdxl_image = sdxl_pipe(prompt, num_inference_steps=30).images[0]
# Result: Good quality, some artifacts

# SD 1.5
sd15_image = sd15_pipe(prompt, num_inference_steps=30).images[0]
# Result: Lower quality, potential distortions
```

### Strengths of Flux.1

1. **Superior Detail**: Finer details in textures, faces, objects
2. **Better Composition**: More coherent scene layouts
3. **Text Rendering**: Can actually render readable text
4. **Prompt Understanding**: Better interpretation of complex prompts
5. **Natural Images**: More photorealistic when requested

### Strengths of Stable Diffusion

1. **Ecosystem**: Vast library of models, LoRAs, tools
2. **VRAM Efficiency**: Runs on lower-end hardware
3. **Community**: Large community, extensive documentation
4. **Extensions**: ControlNet, regional prompting, etc.
5. **Customization**: Easy to fine-tune and merge

### When to Use Each

**Use Flux.1 when:**
- Maximum quality is priority
- Need text in images
- Want natural, detailed results
- Have adequate hardware
- Creating professional content

**Use Stable Diffusion when:**
- Need specific styles (anime, etc.)
- Want to use LoRAs/embeddings
- Limited VRAM (<12GB)
- Need extensive control (ControlNet)
- Large existing workflow

## Advanced Techniques

### Image-to-Image (via Diffusers)

```python
from diffusers import FluxImg2ImgPipeline
from PIL import Image

# Load pipeline
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load input image
init_image = Image.open("input.jpg").convert("RGB")

# Transform
prompt = "transform into an oil painting, artistic style"
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=3.5
).images[0]
```

### ControlNet (via third-party)

```python
# Note: Official ControlNet not yet released
# Community implementations available

# Example with X-Labs implementation
from flux_control import FluxControlNetPipeline

pipe = FluxControlNetPipeline.from_pretrained(
    "XLabs-AI/flux-controlnet-canny",
    torch_dtype=torch.bfloat16
)

# Use canny edge detection
control_image = generate_canny(input_image)
output = pipe(prompt, control_image=control_image).images[0]
```

### LoRA Fine-tuning

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# Load LoRA (when available)
pipe.load_lora_weights("path/to/flux-lora.safetensors")

# Generate with LoRA style
prompt = "a portrait in the custom style"
image = pipe(prompt).images[0]
```

### Batching for Efficiency

```python
# Generate multiple variations
prompts = [
    "a red car",
    "a blue car",
    "a green car",
    "a yellow car"
]

images = []
for prompt in prompts:
    image = pipe(prompt, num_inference_steps=30).images[0]
    images.append(image)
    
# Or use batch processing if memory allows
```

## Optimization

### Memory Optimization

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# Enable CPU offloading
pipe.enable_model_cpu_offload()

# Enable VAE optimizations
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# For extreme memory savings
pipe.enable_sequential_cpu_offload()
```

### Speed Optimization

```python
# Use Schnell for speed
pipe_schnell = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# Compile for faster inference (PyTorch 2.0+)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# Use fewer steps
image = pipe_schnell(prompt, num_inference_steps=4).images[0]
```

### Quantization

```python
# 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)
```

### Multi-GPU

```python
# Distribute across GPUs
from accelerate import PartialState

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

distributed_state = PartialState()
pipe.to(distributed_state.device)
```

## API Usage (Flux Pro)

### REST API

```python
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "https://api.bfl.ml/v1/flux-pro"
API_KEY = "your-api-key"

def generate_flux_pro(
    prompt,
    width=1024,
    height=1024,
    steps=30,
    guidance=3.5
):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance
    }
    
    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        image_data = response.json()["image"]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        return image
    else:
        raise Exception(f"API Error: {response.text}")

# Generate
image = generate_flux_pro(
    "a beautiful sunset over mountains",
    width=1344,
    height=768
)
image.save("pro_output.png")
```

### Async API

```python
import asyncio
import aiohttp

async def generate_async(prompt):
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        payload = {"prompt": prompt}
        
        async with session.post(API_URL, json=payload, headers=headers) as resp:
            return await resp.json()

# Use
image_data = asyncio.run(generate_async("a futuristic city"))
```

## Tips & Best Practices

### 1. Prompt Quality

```python
# Good prompts for Flux
good_prompts = [
    "a cinematic photograph of [subject], [details], [lighting], [camera]",
    "an oil painting of [scene], [style], by [artist]",
    "product photography of [item], [background], professional lighting",
    "character design of [character], [details], concept art"
]
```

### 2. Iteration Strategy

```python
# Start with Schnell for quick iterations
quick_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# Iterate quickly
for variation in range(5):
    gen = torch.Generator("cuda").manual_seed(variation)
    preview = quick_pipe(
        prompt,
        num_inference_steps=4,
        generator=gen
    ).images[0]
    preview.save(f"preview_{variation}.png")

# Refine winner with Dev
final = dev_pipe(
    final_prompt,
    num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(winning_seed)
).images[0]
```

### 3. VRAM Management

```python
# Monitor VRAM
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache between generations
torch.cuda.empty_cache()
```

### 4. Prompt Templates

```python
templates = {
    "portrait": "{subject}, {expression}, {clothing}, {background}, portrait photography, {lighting}",
    "landscape": "{location}, {time_of_day}, {weather}, {style}, landscape photography",
    "product": "product photography of {product}, {surface}, {lighting}, professional, commercial",
    "artistic": "{style} of {subject}, {details}, by {artist}, masterpiece"
}

# Use
prompt = templates["portrait"].format(
    subject="a young woman",
    expression="slight smile",
    clothing="elegant dress",
    background="bokeh lights",
    lighting="soft natural light"
)
```

## Resources

### Official
- [Black Forest Labs Website](https://blackforestlabs.ai/)
- [Hugging Face Models](https://huggingface.co/black-forest-labs)
- [API Documentation](https://docs.bfl.ml/)

### Community
- r/FluxAI
- Hugging Face Discussions
- Discord communities

### Tools
- [ComfyUI Flux Nodes](https://github.com/XLabs-AI/x-flux-comfyui)
- [Diffusers Integration](https://github.com/huggingface/diffusers)

### Learning
- [Flux.1 Paper](https://blackforestlabs.ai/announcing-flux-1/)
- Comparison benchmarks
- Community prompts

## Conclusion

Flux.1 represents a significant leap in image generation quality. While it requires more resources than Stable Diffusion, the results are often worth it for professional applications. The Schnell variant offers excellent speed-to-quality ratio, while Dev provides maximum quality for local generation.

Key takeaways:
- **Schnell**: Fast, commercial-friendly, good quality
- **Dev**: Best local quality, non-commercial
- **Pro**: Highest quality, API-only, commercial

Choose based on your needs, hardware, and use case. Experiment with natural language prompts and leverage Flux's superior understanding for best results.
