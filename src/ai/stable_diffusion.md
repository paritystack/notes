# Stable Diffusion

Complete guide to Stable Diffusion for image generation, from setup to advanced techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation & Setup](#installation--setup)
- [Model Versions](#model-versions)
- [Prompt Engineering](#prompt-engineering)
- [Parameters](#parameters)
- [Advanced Techniques](#advanced-techniques)
- [Extensions & Tools](#extensions--tools)
- [Optimization](#optimization)
- [Common Issues](#common-issues)

## Introduction

Stable Diffusion is an open-source text-to-image diffusion model capable of generating high-quality images from text descriptions. Unlike proprietary alternatives, it can run locally on consumer hardware.

### Key Features
- **Open Source**: Free to use and modify
- **Local Execution**: Run on your own hardware
- **Extensible**: ControlNet, LoRA, extensions
- **Fast**: Optimized inference with various schedulers
- **Flexible**: Text-to-image, image-to-image, inpainting

## Installation & Setup

### Option 1: AUTOMATIC1111 WebUI (Most Popular)

```bash
# Clone repository
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Install (Linux/Mac)
./webui.sh

# Install (Windows)
# Double-click webui-user.bat

# With custom arguments
# Edit webui-user.sh or webui-user.bat:
export COMMANDLINE_ARGS="--xformers --medvram --api"
```

**System Requirements:**
- GPU: NVIDIA (8GB+ VRAM recommended)
- RAM: 16GB+ system RAM
- Storage: 10GB+ for models

### Option 2: ComfyUI (Node-Based)

```bash
# Clone repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run
python main.py

# With arguments
python main.py --listen --port 8188
```

### Option 3: Python Library (Diffusers)

```bash
pip install diffusers transformers accelerate torch torchvision
```

```python
from diffusers import StableDiffusionPipeline
import torch

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Enable optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Generate
prompt = "a beautiful landscape"
image = pipe(prompt).images[0]
image.save("output.png")
```

### Option 4: Invoke AI

```bash
pip install invokeai
invokeai-configure
invokeai --web
```

## Model Versions

### Stable Diffusion 1.x

**SD 1.4**
```bash
# Download location
models/Stable-diffusion/sd-v1-4.ckpt
```
- Resolution: 512x512
- Training: LAION-2B subset
- Good for: General use

**SD 1.5**
```bash
# Most popular 1.x version
wget https://huggingface.co/runwayml/stable-diffusion-v1-5
```
- Improved over 1.4
- Massive ecosystem of fine-tunes
- Best model support

### Stable Diffusion 2.x

**SD 2.0**
- Resolution: 768x768
- New text encoder (OpenCLIP)
- Better quality but different style

**SD 2.1**
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
```
- Improvements over 2.0
- Recommended 2.x version

### Stable Diffusion XL (SDXL)

```python
from diffusers import StableDiffusionXLPipeline

# Base model
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
base.to("cuda")

# Refiner (optional, improves quality)
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
refiner.to("cuda")

# Generate
image = base(prompt="a futuristic city").images[0]
image = refiner(prompt="a futuristic city", image=image).images[0]
```

**Features:**
- Resolution: 1024x1024
- Higher quality
- Better text rendering
- Dual text encoders
- Requires more VRAM (8GB+)

### Stable Diffusion 3

Latest version with improved architecture:
- Multimodal diffusion transformer
- Better prompt understanding
- Improved composition

## Prompt Engineering

### Basic Structure

```
[Subject] [Action/Scene] [Environment] [Lighting] [Style] [Quality]
```

### Effective Prompts

```
Basic:
"a cat"

Better:
"a fluffy orange cat sitting on a windowsill"

Best:
"a fluffy orange tabby cat sitting on a wooden windowsill, looking outside at falling snow, soft natural lighting, cozy atmosphere, detailed fur texture, photorealistic, 4k, highly detailed"
```

### Prompt Components

#### 1. Subject

```
"portrait of a young woman"
"a medieval castle"
"a steampunk airship"
"cyberpunk street scene"
```

#### 2. Action/Pose

```
"running through a field"
"sitting in contemplation"
"dancing under moonlight"
"reading a book by firelight"
```

#### 3. Environment

```
"in a mystical forest"
"on a alien planet"
"in a Victorian library"
"at a bustling marketplace"
```

#### 4. Lighting

```
"golden hour lighting"
"dramatic rim lighting"
"soft diffused light"
"neon lights reflecting on wet streets"
"volumetric fog with god rays"
```

#### 5. Style

```
"oil painting style"
"anime art style"
"photorealistic"
"watercolor painting"
"digital art, trending on artstation"
"in the style of Greg Rutkowski"
```

#### 6. Quality Boosters

```
"highly detailed"
"8k resolution"
"masterpiece"
"professional photography"
"award-winning"
"intricate details"
"sharp focus"
```

### Negative Prompts

What to avoid in generation:

```
Negative Prompt:
"ugly, blurry, low quality, distorted, deformed, bad anatomy, poorly drawn, low resolution, watermark, signature, text, cropped, worst quality, jpeg artifacts"
```

**Common Negative Terms:**
- Quality: `blurry, low quality, pixelated, grainy`
- Anatomy: `bad anatomy, extra limbs, malformed hands`
- Artifacts: `watermark, text, signature, logo`
- Style: `cartoon (for photorealistic), realistic (for artistic)`

### Prompt Weighting

Emphasize or de-emphasize parts:

```
# AUTOMATIC1111 syntax
(keyword)       # 1.1x weight
((keyword))     # 1.21x weight
(keyword:1.5)   # 1.5x weight
[keyword]       # 0.9x weight

Example:
"a (beautiful:1.3) landscape with (mountains:1.2) and [trees:0.8]"
```

### Prompt Editing

Change prompts during generation:

```
# AUTOMATIC1111 syntax
[keyword1:keyword2:step]

Example:
"a [dog:cat:0.5]"
# Generates dog for first 50% of steps, then cat

"photo of a woman [smiling:serious:10]"
# Smiling for first 10 steps, then serious
```

### Artist Styles

Reference famous artists:

```
"in the style of Van Gogh"
"by Greg Rutkowski"
"by Alphonse Mucha"
"by Simon Stalenhag"
"by Artgerm"
"by Ilya Kuvshinov"
```

## Parameters

### Core Parameters

#### Steps (num_inference_steps)

```python
# Fewer steps = faster, less refined
image = pipe(prompt, num_inference_steps=20)

# More steps = slower, more refined
image = pipe(prompt, num_inference_steps=50)
```

**Recommendations:**
- Quick preview: 15-20 steps
- Standard quality: 25-35 steps
- High quality: 40-60 steps
- Diminishing returns after 60

#### CFG Scale (guidance_scale)

How closely to follow the prompt:

```python
# Low CFG = creative, less adherence
image = pipe(prompt, guidance_scale=3.5)

# Medium CFG = balanced
image = pipe(prompt, guidance_scale=7.5)

# High CFG = strict adherence, may oversaturate
image = pipe(prompt, guidance_scale=15)
```

**Recommendations:**
- Creative/artistic: 5-7
- Balanced: 7-10
- Strict/detailed: 10-15
- Avoid: >20 (over-saturated)

#### Seed

Reproducible results:

```python
# Random seed
image = pipe(prompt)

# Fixed seed for reproducibility
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(prompt, generator=generator)
```

#### Sampler/Scheduler

Different algorithms for denoising:

```python
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)

# Fast and high quality (recommended)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# More creative, varied
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

# Stable, predictable
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)
```

**Popular Samplers:**
- **DPM++ 2M Karras**: Fast, high quality (recommended)
- **Euler a**: Creative, varied results
- **DDIM**: Stable, reproducible
- **UniPC**: Very fast, good quality
- **DPM++ SDE Karras**: High quality, slower

#### Resolution

```python
# SD 1.5 native: 512x512
image = pipe(prompt, width=512, height=512)

# SD 2.1 native: 768x768
image = pipe(prompt, width=768, height=768)

# SDXL native: 1024x1024
image = pipe(prompt, width=1024, height=1024)

# Portrait
image = pipe(prompt, width=512, height=768)

# Landscape
image = pipe(prompt, width=768, height=512)
```

**Tips:**
- Stick to multiples of 8 or 64
- Native resolution gives best results
- Higher resolution needs more VRAM
- Use upscaling for ultra-high resolution

### Batch Settings

```python
# Generate multiple images
images = pipe(
    prompt,
    num_images_per_prompt=4,
    guidance_scale=7.5
).images

# Save all
for i, img in enumerate(images):
    img.save(f"output_{i}.png")
```

## Advanced Techniques

### Image-to-Image

Transform existing images:

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Load image
init_image = Image.open("input.jpg").convert("RGB")
init_image = init_image.resize((768, 768))

# Transform
prompt = "a fantasy castle, magical, highly detailed"
images = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # 0=no change, 1=complete regeneration
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

images.save("transformed.png")
```

**Strength Parameter:**
- 0.1-0.3: Minor adjustments, preserve structure
- 0.4-0.6: Moderate changes, guided by original
- 0.7-0.9: Major changes, loose interpretation
- 1.0: Complete regeneration

### Inpainting

Edit specific parts of images:

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("photo.png")
mask = Image.open("mask.png")  # White = inpaint, Black = keep

prompt = "a red vintage car"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

result.save("inpainted.png")
```

### ControlNet

Precise control over generation:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load image and create canny edge map
image = Image.open("input.jpg")
image = np.array(image)
canny_edges = cv2.Canny(image, 100, 200)
canny_edges = Image.fromarray(canny_edges)

# Generate with control
prompt = "a professional architectural photograph"
output = pipe(
    prompt=prompt,
    image=canny_edges,
    num_inference_steps=30
).images[0]
```

**ControlNet Models:**
- **Canny**: Edge detection
- **Depth**: Depth map
- **OpenPose**: Human pose
- **Scribble**: Hand-drawn sketches
- **Normal**: Normal maps
- **Segmentation**: Semantic segmentation
- **MLSD**: Line detection (architecture)

### LoRA (Low-Rank Adaptation)

Fine-tuned models with small file size:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA
pipe.load_lora_weights("path/to/lora.safetensors")

# Generate with LoRA style
prompt = "a portrait in the style of <lora-trigger-word>"
image = pipe(prompt).images[0]

# Unload LoRA
pipe.unload_lora_weights()
```

**Popular LoRA Types:**
- Character/celebrity faces
- Art styles
- Concepts
- Objects/clothing

### Textual Inversion

Custom concepts/embeddings:

```python
# Load embedding
pipe.load_textual_inversion("path/to/embedding.pt", token="<special-token>")

# Use in prompt
prompt = "a photo of <special-token> in a forest"
image = pipe(prompt).images[0]
```

### Upscaling

Increase resolution with detail:

```python
from diffusers import StableDiffusionUpscalePipeline

# Load upscaler
upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16
).to("cuda")

# Load low-res image
low_res = Image.open("output_512.png")

# Upscale
prompt = "highly detailed, sharp, professional"
upscaled = upscaler(
    prompt=prompt,
    image=low_res,
    num_inference_steps=50
).images[0]

upscaled.save("output_2048.png")
```

**Upscaling Options:**
1. **SD Upscale**: Built-in SD upscaler
2. **Real-ESRGAN**: Traditional upscaler
3. **Ultimate SD Upscale**: Tiled upscaling
4. **ControlNet Tile**: Detail-preserving upscale

## Extensions & Tools

### AUTOMATIC1111 Extensions

Install via Extensions tab or:

```bash
cd extensions
git clone [extension-repo-url]
```

#### Essential Extensions

**ControlNet**
```bash
git clone https://github.com/Mikubill/sd-webui-controlnet.git
```

**Dynamic Prompts**
```bash
git clone https://github.com/adieyal/sd-dynamic-prompts.git
```
- Wildcard support: `{red|blue|green} car`
- Combinatorial generation

**Image Browser**
```bash
git clone https://github.com/AlUlkesh/stable-diffusion-webui-images-browser.git
```
- Browse generated images
- Search by metadata

**Cutoff**
```bash
git clone https://github.com/hnmr293/sd-webui-cutoff.git
```
- Prevent color bleeding between subjects

**Regional Prompter**
```bash
git clone https://github.com/hako-mikan/sd-webui-regional-prompter.git
```
- Different prompts for image regions

### Checkpoint Merging

Combine models:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load two models
pipe1 = StableDiffusionPipeline.from_pretrained("model1")
pipe2 = StableDiffusionPipeline.from_pretrained("model2")

# Merge (0.5 = 50/50 blend)
alpha = 0.5
for key in pipe1.unet.state_dict():
    pipe1.unet.state_dict()[key] = (
        alpha * pipe1.unet.state_dict()[key] +
        (1 - alpha) * pipe2.unet.state_dict()[key]
    )

# Save merged model
pipe1.save_pretrained("merged_model")
```

### Prompt Matrix

Test multiple prompts:

```
# In AUTOMATIC1111
Prompt: a |red, blue, green| |car, house| in a forest

Generates:
- a red car in a forest
- a red house in a forest
- a blue car in a forest
- a blue house in a forest
- a green car in a forest
- a green house in a forest
```

## Optimization

### Memory Optimization

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # Half precision
).to("cuda")

# Enable memory optimizations
pipe.enable_attention_slicing()  # Reduce memory
pipe.enable_vae_slicing()  # Reduce VAE memory
pipe.enable_xformers_memory_efficient_attention()  # Faster attention

# For very low VRAM (4GB)
pipe.enable_sequential_cpu_offload()
```

### Speed Optimization

```python
# Use faster scheduler
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# Compile model (PyTorch 2.0+)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# Reduce steps with quality scheduler
image = pipe(prompt, num_inference_steps=20)  # vs 50 with others
```

### VRAM Requirements

| Configuration | Minimum VRAM |
|--------------|--------------|
| SD 1.5 (512x512) | 4GB |
| SD 1.5 (512x512, optimized) | 2GB |
| SD 2.1 (768x768) | 6GB |
| SDXL (1024x1024) | 8GB |
| SDXL (1024x1024, optimized) | 6GB |
| ControlNet + SD | +2GB |
| Batch size 2 | +2GB per image |

### Launch Arguments (AUTOMATIC1111)

```bash
# Basic optimization
--xformers              # Memory-efficient attention
--medvram               # Medium VRAM optimization
--lowvram               # Low VRAM optimization
--no-half-vae           # Fix black images on some GPUs

# API
--api                   # Enable API
--listen                # Allow network connections

# Performance
--opt-sdp-attention     # Scaled dot product attention
--no-gradio-queue       # Disable queue

# Example combination
./webui.sh --xformers --medvram --api --no-half-vae
```

## Common Issues

### Black Images

```bash
# Solution: Disable half precision for VAE
--no-half-vae
```

Or in Python:
```python
pipe.vae.to(torch.float32)
```

### Out of Memory (OOM)

```python
# Enable all optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()

# Reduce resolution
image = pipe(prompt, width=512, height=512)

# Reduce batch size
image = pipe(prompt, num_images_per_prompt=1)
```

### Bad Hands/Anatomy

```
Negative prompt: "bad hands, bad anatomy, extra fingers, missing fingers, deformed hands, poorly drawn hands"

# Or use inpainting to fix
# Or use ControlNet OpenPose for guidance
```

### Inconsistent Results

```python
# Use fixed seed
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(prompt, generator=generator)

# Use lower temperature sampler (DDIM instead of Euler a)
```

### Prompt Not Working

1. Check prompt weighting: `(keyword:1.3)`
2. Use negative prompt to exclude unwanted elements
3. Increase CFG scale
4. Try different sampler
5. Add quality boosters: "highly detailed, 8k"

## Best Practices

### 1. Prompt Structure

```
[Quality] [Style] [Subject] [Action] [Environment] [Lighting] [Details]

Example:
"masterpiece, best quality, photorealistic, portrait of a young woman, smiling, in a sunlit garden, golden hour lighting, detailed facial features, professional photography, 8k uhd"
```

### 2. Iterative Refinement

```python
# Start with low steps for preview
preview = pipe(prompt, num_inference_steps=15).images[0]

# Refine with more steps
final = pipe(prompt, num_inference_steps=50).images[0]

# Upscale for details
upscaled = upscale(final)
```

### 3. Seed Management

```python
# Save seeds for good results
good_seeds = []

for seed in range(100):
    gen = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt, generator=gen).images[0]
    
    if is_good(image):
        good_seeds.append(seed)
        image.save(f"good_{seed}.png")
```

### 4. Negative Prompts Library

```python
negative_prompts = {
    'photorealistic': "anime, cartoon, drawing, painting, low quality",
    'artistic': "photorealistic, photo, realistic, low quality",
    'quality': "ugly, blurry, low quality, low resolution, pixelated",
    'anatomy': "bad anatomy, extra limbs, poorly drawn, deformed",
    'artifacts': "watermark, signature, text, logo, copyright"
}

# Combine as needed
negative = ", ".join([
    negative_prompts['quality'],
    negative_prompts['anatomy'],
    negative_prompts['artifacts']
])
```

## Resources

### Models
- [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Civitai](https://civitai.com/) - Community models, LoRAs
- [Stability AI](https://huggingface.co/stabilityai)

### Tools
- [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)

### Learning
- [Stable Diffusion Art](https://stable-diffusion-art.com/)
- [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/)
- [OpenArt Prompt Book](https://openart.ai/promptbook)

### Communities
- Discord: Stable Diffusion
- Reddit: r/StableDiffusion
- Twitter/X: #StableDiffusion

## Conclusion

Stable Diffusion offers incredible flexibility and power for image generation. Success comes from understanding the fundamentals, experimenting with parameters, and iterating on prompts. Start simple, learn the basics, then explore advanced techniques like ControlNet and LoRA for professional results.
