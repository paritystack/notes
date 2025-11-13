# Deep Generative Models

Advanced architectures and techniques for generating high-quality data.

## Table of Contents

1. [Transformer-based Generative Models](#transformer-based-generative-models)
2. [Diffusion Models](#diffusion-models)
3. [Vector Quantized Models](#vector-quantized-models)
4. [NeRF and 3D Generation](#nerf-and-3d-generation)
5. [Multimodal Generative Models](#multimodal-generative-models)

## Transformer-based Generative Models

### GPT (Generative Pre-trained Transformer)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_length=1024, dropout=0.1):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_length = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device))
        mask = mask.view(1, 1, seq_length, seq_length)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_length else input_ids[:, -self.max_seq_length:]
            
            # Forward pass
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# Training example
model = GPTModel(vocab_size=50257, d_model=768, num_heads=12, num_layers=12)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))

def train_step(input_ids, targets):
    optimizer.zero_grad()
    logits, loss = model(input_ids, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
```

### Vision Transformer for Generation (ViT-VQGAN)

```python
class VisionTransformerGenerator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, num_heads=12, depth=12):
        super(VisionTransformerGenerator, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(depth)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Reshape for decoder
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        
        # Decode
        x = self.decoder(x)
        
        return x
```

## Diffusion Models

### Improved DDPM

```python
class ImprovedDDPM(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, timesteps=1000):
        super(ImprovedDDPM, self).__init__()
        
        self.timesteps = timesteps
        
        # Variance schedule (cosine)
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # U-Net architecture
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # Encoder
        self.down1 = self._make_down_block(img_channels, base_channels)
        self.down2 = self._make_down_block(base_channels, base_channels * 2)
        self.down3 = self._make_down_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.mid = self._make_res_block(base_channels * 4)
        
        # Decoder
        self.up3 = self._make_up_block(base_channels * 4, base_channels * 2)
        self.up2 = self._make_up_block(base_channels * 2, base_channels)
        self.up1 = self._make_up_block(base_channels, img_channels)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule for betas"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _make_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
    
    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
    
    def forward(self, x, t):
        """Predict noise"""
        # Time embedding
        t_emb = self._get_timestep_embedding(t, x.device)
        t_emb = self.time_embed(t_emb)
        
        # U-Net forward
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        
        h = self.mid(h3)
        
        h = self.up3(h + h3)
        h = self.up2(h + h2)
        h = self.up1(h + h1)
        
        return h
    
    def _get_timestep_embedding(self, timesteps, device, dim=128):
        """Sinusoidal positional embedding"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    @torch.no_grad()
    def sample(self, batch_size, img_size, device):
        """DDPM sampling"""
        # Start from random noise
        img = torch.randn(batch_size, 3, img_size, img_size, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(img, t_batch)
            
            # Compute x_{t-1}
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
            
            img = (1 / alpha_t.sqrt()) * (img - ((1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()) * predicted_noise)
            img = img + beta_t.sqrt() * noise
        
        return img
```

### Latent Diffusion Models (Stable Diffusion)

```python
class LatentDiffusion(nn.Module):
    def __init__(self, vae, unet, text_encoder):
        super(LatentDiffusion, self).__init__()
        
        self.vae = vae  # VAE for encoding/decoding images
        self.unet = unet  # U-Net for denoising in latent space
        self.text_encoder = text_encoder  # CLIP text encoder
        
        self.timesteps = 1000
        self.betas = torch.linspace(0.0001, 0.02, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, images, text_embeddings, t):
        """Training forward pass"""
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images)
        
        # Add noise
        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, t)
        
        # Predict noise conditioned on text
        predicted_noise = self.unet(noisy_latents, t, text_embeddings)
        
        # Loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def _add_noise(self, latents, noise, t):
        """Add noise according to schedule"""
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5
        
        return sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
    
    @torch.no_grad()
    def generate(self, text, batch_size=1, guidance_scale=7.5):
        """Text-to-image generation"""
        # Encode text
        text_embeddings = self.text_encoder(text)
        
        # Start from random noise in latent space
        latents = torch.randn(batch_size, 4, 64, 64)
        
        # Denoising loop
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t)
            
            # Predict noise with and without conditioning (classifier-free guidance)
            noise_pred_text = self.unet(latents, t_batch, text_embeddings)
            noise_pred_uncond = self.unet(latents, t_batch, None)
            
            # Apply guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latents
            latents = self._denoise_step(latents, noise_pred, t)
        
        # Decode latents to images
        images = self.vae.decode(latents)
        
        return images
    
    def _denoise_step(self, latents, noise, t):
        """Single denoising step"""
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        pred_original = (latents - ((1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()) * noise) / alpha_t.sqrt()
        
        if t > 0:
            noise = torch.randn_like(latents)
            latents = pred_original * alpha_t.sqrt() + (1 - alpha_t).sqrt() * noise
        else:
            latents = pred_original
        
        return latents
```

## Vector Quantized Models

### VQ-VAE

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view_as(inputs)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 3, 1, 1)
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        recon = self.decoder(quantized)
        
        recon_loss = F.mse_loss(recon, x)
        
        return recon, recon_loss + vq_loss
```

## NeRF and 3D Generation

### Neural Radiance Fields

```python
class NeRF(nn.Module):
    def __init__(self, pos_dim=3, dir_dim=3, hidden_dim=256):
        super(NeRF, self).__init__()
        
        # Position encoding
        self.pos_encoder = self._positional_encoding
        self.dir_encoder = self._positional_encoding
        
        # MLP for density and features
        self.density_net = nn.Sequential(
            nn.Linear(pos_dim * 2 * 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # density + features
        )
        
        # MLP for color
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim * 2 * 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
    
    def _positional_encoding(self, x, L=10):
        """Positional encoding for coordinates"""
        encoding = []
        for l in range(L):
            encoding.append(torch.sin(2**l * math.pi * x))
            encoding.append(torch.cos(2**l * math.pi * x))
        return torch.cat(encoding, dim=-1)
    
    def forward(self, positions, directions):
        # Encode positions and directions
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(directions)
        
        # Get density and features
        density_features = self.density_net(pos_enc)
        density = F.relu(density_features[:, :1])
        features = density_features[:, 1:]
        
        # Get color
        color_input = torch.cat([features, dir_enc], dim=-1)
        color = self.color_net(color_input)
        
        return density, color
    
    def render_rays(self, ray_origins, ray_directions, near=2.0, far=6.0, n_samples=64):
        """Volume rendering along rays"""
        # Sample points along rays
        t_vals = torch.linspace(near, far, n_samples, device=ray_origins.device)
        points = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[None, :, None]
        
        # Flatten for network
        points_flat = points.reshape(-1, 3)
        dirs_flat = ray_directions[:, None, :].expand_as(points).reshape(-1, 3)
        
        # Get density and color
        density, color = self.forward(points_flat, dirs_flat)
        
        # Reshape
        density = density.reshape(points.shape[0], n_samples)
        color = color.reshape(points.shape[0], n_samples, 3)
        
        # Volume rendering
        dists = t_vals[1:] - t_vals[:-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device)])
        
        alpha = 1.0 - torch.exp(-density * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=-1)
        
        weights = alpha * transmittance
        rgb = torch.sum(weights[:, :, None] * color, dim=1)
        
        return rgb
```

## Multimodal Generative Models

### CLIP-guided Generation

```python
class CLIPGuidedGenerator:
    def __init__(self, generator, clip_model):
        self.generator = generator
        self.clip_model = clip_model
    
    def generate(self, text_prompt, num_steps=100, lr=0.1):
        """Generate image guided by CLIP text embedding"""
        # Encode text
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_prompt)
        
        # Start with random latent
        latent = torch.randn(1, 512, requires_grad=True)
        optimizer = torch.optim.Adam([latent], lr=lr)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Generate image
            image = self.generator(latent)
            
            # Encode image with CLIP
            image_features = self.clip_model.encode_image(image)
            
            # CLIP loss (maximize similarity)
            loss = -torch.cosine_similarity(text_features, image_features).mean()
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        
        # Generate final image
        with torch.no_grad():
            final_image = self.generator(latent)
        
        return final_image
```

## Practical Tips

1. **Model Size**: Start small, scale up gradually
2. **Training Stability**: Use gradient clipping, EMA
3. **Quality Metrics**: FID, IS, LPIPS for evaluation
4. **Computational Efficiency**: Use mixed precision, model parallelism
5. **Fine-tuning**: Transfer from pre-trained models

## Resources

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- DALL-E 2 paper: https://arxiv.org/abs/2204.06125
- Imagen paper: https://arxiv.org/abs/2205.11487
- NeRF: https://www.matthewtancik.com/nerf

