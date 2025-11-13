# Generative AI

A comprehensive guide to generative AI models, applications, and practical implementations.

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Text Generation](#text-generation)
- [Image Generation](#image-generation)
- [Audio Generation](#audio-generation)
- [Video Generation](#video-generation)
- [Multimodal Models](#multimodal-models)
- [Applications](#applications)
- [Implementation Examples](#implementation-examples)

## Introduction

Generative AI refers to artificial intelligence systems that can create new content—text, images, audio, video, code, and more. Unlike discriminative models that classify or predict, generative models learn to produce novel outputs that resemble their training data.

### Key Characteristics

- **Content Creation**: Generate new, original content
- **Pattern Learning**: Understand and replicate complex patterns
- **Conditional Generation**: Create outputs based on specific inputs/prompts
- **Iterative Refinement**: Improve outputs through multiple passes

## Core Concepts

### 1. Generative Models

#### Autoregressive Models
Generate sequences one token at a time, using previous tokens as context:

```
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
```

Examples: GPT series, LLaMA

#### Diffusion Models
Learn to denoise data through iterative refinement:

```
Forward process: x₀ → x₁ → ... → xₜ (add noise)
Reverse process: xₜ → xₜ₋₁ → ... → x₀ (remove noise)
```

Examples: Stable Diffusion, DALL-E 3, Midjourney

#### Variational Autoencoders (VAE)
Learn compressed representations in latent space:

```
Encoder: x → z (data to latent space)
Decoder: z → x' (latent space to reconstruction)
```

#### Generative Adversarial Networks (GAN)
Two networks compete—generator creates, discriminator evaluates:

```
Generator: z → x (noise to data)
Discriminator: x → [0,1] (real vs fake)
```

Examples: StyleGAN, BigGAN

### 2. Foundation Models

Large-scale models trained on vast datasets, adaptable to many tasks:

- **Scale**: Billions to trillions of parameters
- **Transfer Learning**: Fine-tune for specific tasks
- **Few-Shot Learning**: Adapt with minimal examples
- **Emergent Abilities**: Capabilities not explicitly trained

## Text Generation

### Large Language Models (LLMs)

#### GPT Family (OpenAI)

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

# GPT-4 Turbo - Most capable
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a short sci-fi story about AI."}
    ],
    temperature=0.8,
    max_tokens=500
)

print(response.choices[0].message.content)
```

**Models:**
- `gpt-4-turbo`: Most capable, best for complex tasks
- `gpt-4`: High capability, slower and more expensive
- `gpt-3.5-turbo`: Fast, cost-effective for simple tasks

#### Claude (Anthropic)

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

# Claude Sonnet 4.5 - Latest model
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {
            "role": "user", 
            "content": "Analyze this code and suggest improvements: [code]"
        }
    ]
)

print(message.content[0].text)
```

**Models:**
- `claude-sonnet-4-5`: Balanced performance and capability
- `claude-opus-4`: Most capable, deep analysis
- `claude-haiku-4`: Fastest, most cost-effective

#### Llama (Meta)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Chat format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Mistral

```python
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

client = MistralClient(api_key="your-key")

messages = [
    ChatMessage(role="user", content="What is machine learning?")
]

# Mistral Large - Most capable
response = client.chat(
    model="mistral-large-latest",
    messages=messages
)

print(response.choices[0].message.content)
```

### Use Cases for Text Generation

#### 1. Content Creation

```python
# Blog post generation
prompt = """
Write a 500-word blog post about sustainable living.

Include:
- Engaging introduction
- 3 practical tips
- Statistics or facts
- Call to action

Tone: Informative but conversational
"""
```

#### 2. Code Generation

```python
# Function generation
prompt = """
Create a Python function that:
- Takes a list of dictionaries
- Filters by a key-value pair
- Sorts by another key
- Returns top N results

Include type hints and docstring.
"""
```

#### 3. Data Analysis

```python
# Analysis prompt
prompt = """
Analyze this sales data and provide:
1. Key trends
2. Anomalies
3. Predictions
4. Recommendations

Data: [CSV or JSON data]
"""
```

#### 4. Translation

```python
# Contextual translation
prompt = """
Translate this technical documentation from English to Spanish:

[text]

Maintain:
- Technical terminology accuracy
- Professional tone
- Code examples unchanged
"""
```

## Image Generation

### Stable Diffusion

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe = pipe.to("cuda")

# Generate image
prompt = "a serene japanese garden with cherry blossoms, koi pond, stone lanterns, soft morning light, highly detailed, 4k"
negative_prompt = "blurry, distorted, low quality, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=768,
    height=768
).images[0]

image.save("japanese_garden.png")
```

### DALL-E 3 (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city with flying cars and neon lights, cyberpunk style, detailed, high quality",
    size="1024x1024",
    quality="hd",
    n=1
)

image_url = response.data[0].url
print(f"Generated image: {image_url}")
```

### Midjourney

Accessed through Discord bot:

```
/imagine prompt: a mystical forest with glowing mushrooms, ethereal lighting, fantasy art style, intricate details --v 6 --ar 16:9 --q 2
```

Parameters:
- `--v`: Version (6 is latest)
- `--ar`: Aspect ratio
- `--q`: Quality (0.25, 0.5, 1, 2)
- `--s`: Stylization (0-1000)

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Load initial image
init_image = Image.open("sketch.png").convert("RGB")
init_image = init_image.resize((768, 768))

# Transform image
prompt = "a professional photograph of a modern building, architectural photography"
images = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # How much to transform (0=no change, 1=complete regeneration)
    guidance_scale=7.5,
    num_inference_steps=50
).images

images[0].save("transformed.png")
```

### Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("photo.png")
mask = Image.open("mask.png")  # White areas will be regenerated

prompt = "a red sports car"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]

result.save("inpainted.png")
```

## Audio Generation

### Text-to-Speech

#### OpenAI TTS

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI()

speech_file_path = Path("output.mp3")

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",  # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! This is a generated voice. AI can now speak naturally."
)

response.stream_to_file(speech_file_path)
```

#### ElevenLabs

```python
from elevenlabs import generate, play, set_api_key

set_api_key("your-api-key")

audio = generate(
    text="Welcome to the future of voice synthesis.",
    voice="Bella",
    model="eleven_monolingual_v1"
)

play(audio)
```

### Music Generation

#### MusicGen (Meta)

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Generate music
descriptions = ['upbeat electronic dance music with strong bass']
duration = 30  # seconds

model.set_generation_params(duration=duration)
wav = model.generate(descriptions)

# Save
for idx, one_wav in enumerate(wav):
    torchaudio.save(f'generated_{idx}.wav', one_wav.cpu(), model.sample_rate)
```

## Video Generation

### Stable Video Diffusion

```python
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Load initial image
image = Image.open("first_frame.png")

# Generate video frames
frames = pipe(image, decode_chunk_size=8, num_frames=25).frames[0]

# Save as video
from diffusers.utils import export_to_video
export_to_video(frames, "output_video.mp4", fps=7)
```

### RunwayML Gen-2

API-based video generation:

```python
import runwayml

client = runwayml.RunwayML(api_key="your-key")

# Text to video
task = client.image_generation.create(
    prompt="a serene ocean at sunset with waves gently crashing",
    model="gen2",
    duration=4
)

# Wait for completion and download
video_url = task.get_output_url()
```

## Multimodal Models

### GPT-4 Vision

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)
```

### Claude Vision

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Read and encode image
with open("image.jpg", "rb") as image_file:
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ],
        }
    ],
)

print(message.content[0].text)
```

### LLaVA (Open Source)

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from PIL import Image

model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# Load and process image
image = Image.open("photo.jpg")
image_tensor = process_images([image], image_processor, model.config)

# Generate description
prompt = "Describe this image in detail."
outputs = model.generate(
    image_tensor,
    prompt,
    max_new_tokens=512
)
```

## Applications

### 1. Content Creation

```python
# Automated blog writing pipeline
def generate_blog_post(topic):
    # Research
    outline_prompt = f"Create a detailed outline for a blog post about {topic}"
    outline = llm.generate(outline_prompt)
    
    # Write sections
    sections = []
    for section in outline.sections:
        content = llm.generate(f"Write about: {section}")
        sections.append(content)
    
    # Generate image
    image_prompt = f"blog header image for {topic}, professional, modern"
    image = image_generator.generate(image_prompt)
    
    return {
        'outline': outline,
        'content': sections,
        'image': image
    }
```

### 2. Education & Training

```python
# Personalized tutoring
def create_lesson(topic, student_level, learning_style):
    prompt = f"""
    Create a {student_level}-level lesson on {topic} for a {learning_style} learner.
    
    Include:
    - Clear explanations with analogies
    - 3 practice problems
    - Visual aids descriptions
    """
    
    lesson = llm.generate(prompt)
    
    # Generate visual aids
    visuals = [
        image_gen.generate(desc) 
        for desc in lesson.visual_descriptions
    ]
    
    return lesson, visuals
```

### 3. Software Development

```python
# AI-assisted coding
def code_assistant(task_description, language="python"):
    # Generate code
    code_prompt = f"Write {language} code for: {task_description}"
    code = llm.generate(code_prompt)
    
    # Generate tests
    test_prompt = f"Write unit tests for this code:\n{code}"
    tests = llm.generate(test_prompt)
    
    # Generate documentation
    doc_prompt = f"Write comprehensive documentation for:\n{code}"
    docs = llm.generate(doc_prompt)
    
    return {
        'code': code,
        'tests': tests,
        'docs': docs
    }
```

### 4. Marketing & Advertising

```python
# Campaign generation
def create_marketing_campaign(product, target_audience):
    # Generate copy variations
    copy_prompt = f"""
    Create 5 ad copy variations for {product} targeting {target_audience}.
    Each should be:
    - Under 100 characters
    - Compelling call-to-action
    - Different emotional angle
    """
    copies = llm.generate(copy_prompt)
    
    # Generate visuals
    for copy in copies:
        visual_prompt = f"advertising image for: {copy}, {product}, professional photography"
        image = image_gen.generate(visual_prompt)
        
    return campaign
```

### 5. Data Augmentation

```python
# Expand training dataset
def augment_dataset(original_data):
    augmented = []
    
    for item in original_data:
        # Text augmentation
        variations = llm.generate(
            f"Create 5 paraphrases of: {item.text}"
        )
        augmented.extend(variations)
        
        # Image augmentation (if applicable)
        if item.image:
            synthetic_images = image_gen.generate(
                f"similar to: {item.image_description}"
            )
            augmented.extend(synthetic_images)
    
    return augmented
```

### 6. Accessibility

```python
# Multi-modal accessibility
def make_accessible(content):
    if content.is_text():
        # Text to speech
        audio = tts.generate(content.text)
        
        # Generate descriptive images
        image = image_gen.generate(f"illustration of: {content.text}")
        
    elif content.is_image():
        # Image to text description
        description = vision_model.describe(content.image)
        
        # Text to speech
        audio = tts.generate(description)
    
    return {
        'text': description,
        'audio': audio,
        'image': image
    }
```

## Best Practices

### 1. Prompt Engineering

```python
# Good prompt structure
prompt = """
Role: You are an expert {domain} specialist

Task: {specific_task}

Context: {relevant_background}

Requirements:
- {requirement_1}
- {requirement_2}
- {requirement_3}

Format: {output_format}
"""
```

### 2. Temperature & Sampling

```python
# Creative tasks: High temperature
creative_config = {
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50
}

# Factual tasks: Low temperature
factual_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40
}
```

### 3. Error Handling

```python
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = llm.generate(prompt)
            
            # Validate response
            if validate(response):
                return response
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            continue
    
    return fallback_response
```

### 4. Cost Optimization

```python
# Cache responses
from functools import lru_cache

@lru_cache(maxsize=1000)
def generate_cached(prompt):
    return llm.generate(prompt)

# Batch requests
def generate_batch(prompts):
    return llm.batch_generate(prompts)

# Use appropriate model
def select_model(task_complexity):
    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # Cheaper
    else:
        return "gpt-4"  # More capable
```

## Ethical Considerations

### 1. Content Authenticity

```python
# Add watermarks to generated content
def generate_with_watermark(prompt):
    content = llm.generate(prompt)
    metadata = {
        'generated_by': 'AI',
        'model': 'gpt-4',
        'timestamp': datetime.now(),
        'watermark': True
    }
    return content, metadata
```

### 2. Bias Detection

```python
# Check for biased outputs
def check_bias(generated_content):
    bias_check_prompt = f"""
    Analyze this content for potential bias:
    {generated_content}
    
    Check for:
    - Gender bias
    - Racial bias
    - Cultural bias
    - Age bias
    """
    
    analysis = llm.generate(bias_check_prompt)
    return analysis
```

### 3. Safety Filters

```python
# Content filtering
def safe_generate(prompt):
    # Check input
    if contains_unsafe_content(prompt):
        return "Request rejected: unsafe content"
    
    # Generate
    output = llm.generate(prompt)
    
    # Check output
    if contains_unsafe_content(output):
        return "Generation failed: unsafe output"
    
    return output
```

## Future Trends

### 1. Multimodal Foundation Models
- Unified models handling text, image, audio, video
- Seamless cross-modal generation

### 2. Personalization
- Models adapting to individual user preferences
- Context-aware generation

### 3. Efficiency
- Smaller, faster models with comparable quality
- Edge deployment of generative models

### 4. Controllability
- Fine-grained control over generation
- Steering models toward specific outputs

### 5. Collaboration
- Human-AI co-creation workflows
- Interactive refinement systems

## Resources

### Learning
- [Hugging Face Diffusers Course](https://huggingface.co/docs/diffusers)
- [DeepLearning.AI Generative AI Courses](https://www.deeplearning.ai/)
- [Stability AI Documentation](https://platform.stability.ai/docs)

### Tools
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Replicate](https://replicate.com/)
- [Gradio](https://gradio.app/)

### Communities
- r/StableDiffusion
- r/LocalLLaMA
- Discord: Stable Diffusion, Midjourney
- Twitter/X: AI researchers and practitioners

## Conclusion

Generative AI is rapidly evolving, with new models and capabilities emerging constantly. Success comes from understanding the fundamentals, choosing appropriate tools, and applying ethical practices. Experiment, iterate, and stay updated with the latest developments.
