# Generative Models

Generative models learn to create new data samples that resemble the training data distribution.

## Table of Contents

1. [Introduction](#introduction)
2. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks)
3. [Variational Autoencoders (VAEs)](#variational-autoencoders)
4. [Normalizing Flows](#normalizing-flows)
5. [Autoregressive Models](#autoregressive-models)
6. [Energy-Based Models](#energy-based-models)
7. [Diffusion Models](#diffusion-models)

## Introduction

**Types of Generative Models:**
- **Explicit Density**: Models that define explicit probability distribution (VAE, Flow models)
- **Implicit Density**: Models that can sample without explicit density (GANs)
- **Tractable**: Can compute exact likelihoods (Autoregressive, Flow models)
- **Approximate**: Use approximate inference (VAEs)

## Generative Adversarial Networks

GANs use two networks competing against each other: Generator and Discriminator.

### Basic GAN

**Objective Function:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training GAN
class GANTrainer:
    def __init__(self, generator, discriminator, latent_dim=100, 
                 lr=0.0002, betas=(0.5, 0.999)):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
    
    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Loss for real images
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        
        # Loss for fake images
        z = torch.randn(batch_size, self.latent_dim)
        fake_imgs = self.generator(z)
        fake_loss = self.adversarial_loss(self.discriminator(fake_imgs.detach()), fake)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim)
        gen_imgs = self.generator(z)
        
        # Generator loss (fool discriminator)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def train(self, dataloader, num_epochs=100):
        """Train GAN"""
        for epoch in range(num_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                d_loss, g_loss = self.train_step(imgs)
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] "
                          f"[D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")
            
            # Sample images
            if epoch % 10 == 0:
                self.sample_images(epoch)
    
    def sample_images(self, epoch, n_row=10):
        """Generate and save sample images"""
        z = torch.randn(n_row**2, self.latent_dim)
        gen_imgs = self.generator(z)
        
        import torchvision.utils as vutils
        vutils.save_image(gen_imgs.data, f"images/epoch_{epoch}.png", 
                         nrow=n_row, normalize=True)

# Example usage
img_shape = (1, 28, 28)
generator = Generator(latent_dim=100, img_shape=img_shape)
discriminator = Discriminator(img_shape=img_shape)

trainer = GANTrainer(generator, discriminator)
# trainer.train(dataloader, num_epochs=100)
```

### Deep Convolutional GAN (DCGAN)

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(DCGANGenerator, self).__init__()
        
        self.init_size = 4
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 4x4 -> 8x8
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(DCGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 32x32 -> 16x16
            *discriminator_block(16, 32),                   # 16x16 -> 8x8
            *discriminator_block(32, 64),                   # 8x8 -> 4x4
            *discriminator_block(64, 128),                  # 4x4 -> 2x2
        )
        
        # Output layer
        ds_size = 2
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
```

### Conditional GAN (cGAN)

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 28, 28)):
        super(ConditionalGenerator, self).__init__()
        self.img_shape = img_shape
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate label embedding and noise
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_classes=10, img_shape=(1, 28, 28)):
        super(ConditionalDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Concatenate label embedding and image
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
```

### Wasserstein GAN (WGAN)

```python
class WGANTrainer:
    def __init__(self, generator, discriminator, latent_dim=100, 
                 lr=0.00005, n_critic=5, clip_value=0.01):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.clip_value = clip_value
        
        # RMSprop optimizers
        self.optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
        self.optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)
    
    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim)
        fake_imgs = self.generator(z).detach()
        
        # Wasserstein loss
        loss_D = -torch.mean(self.discriminator(real_imgs)) + \
                  torch.mean(self.discriminator(fake_imgs))
        
        loss_D.backward()
        self.optimizer_D.step()
        
        # Clip weights
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
        
        # Train generator every n_critic iterations
        if self.n_critic > 0:
            self.n_critic -= 1
            return loss_D.item(), None
        
        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim)
        gen_imgs = self.generator(z)
        
        # Generator loss
        loss_G = -torch.mean(self.discriminator(gen_imgs))
        
        loss_G.backward()
        self.optimizer_G.step()
        
        self.n_critic = 5  # Reset
        
        return loss_D.item(), loss_G.item()
```

### StyleGAN Concepts

```python
class StyleGANGenerator(nn.Module):
    """Simplified StyleGAN architecture"""
    def __init__(self, latent_dim=512, style_dim=512, n_mlp=8):
        super(StyleGANGenerator, self).__init__()
        
        # Mapping network (converts z to w)
        layers = []
        for i in range(n_mlp):
            layers.append(nn.Linear(latent_dim if i == 0 else style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        
        # Synthesis network (generates image from w)
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Progressive layers with AdaIN
        self.prog_blocks = nn.ModuleList()
        self.style_blocks = nn.ModuleList()
        
        channels = [512, 512, 512, 256, 128, 64, 32]
        for i in range(len(channels) - 1):
            self.prog_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                    nn.LeakyReLU(0.2)
                )
            )
            self.style_blocks.append(
                nn.Linear(style_dim, channels[i+1] * 2)  # For AdaIN
            )
        
        self.to_rgb = nn.Conv2d(channels[-1], 3, 1)
    
    def forward(self, z):
        # Map to style space
        w = self.mapping(z)
        
        # Start with constant
        x = self.const_input.repeat(z.size(0), 1, 1, 1)
        
        # Apply progressive blocks with style modulation
        for prog_block, style_block in zip(self.prog_blocks, self.style_blocks):
            x = prog_block(x)
            
            # AdaIN (Adaptive Instance Normalization)
            style = style_block(w).unsqueeze(2).unsqueeze(3)
            style_mean, style_std = style.chunk(2, 1)
            
            x = F.instance_norm(x)
            x = x * (style_std + 1) + style_mean
        
        # Convert to RGB
        img = self.to_rgb(x)
        return torch.tanh(img)
```

## Variational Autoencoders

VAEs learn a latent representation by maximizing a variational lower bound on the data likelihood.

**Objective (ELBO):**
```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

### Basic VAE

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to output"""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function"""
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Training
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_vae(model, dataloader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)}] '
                      f'Loss: {loss.item() / len(data):.4f}')
        
        print(f'Epoch {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')
```

### Convolutional VAE

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, channels=3):
        super(ConvVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),        # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),       # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),      # 4x4 -> 2x2
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### Beta-VAE

```python
def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    """Beta-VAE loss with adjustable KL weight"""
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence with beta weight
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD
```

## Normalizing Flows

Flow models use invertible transformations to model complex distributions.

### Simple Flow

```python
class CouplingLayer(nn.Module):
    """Affine coupling layer"""
    def __init__(self, dim, hidden_dim=256):
        super(CouplingLayer, self).__init__()
        self.dim = dim
        self.split = dim // 2
        
        # Scale and translate networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split)
        )
    
    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        
        if not reverse:
            # Forward pass
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
            log_det = torch.sum(s, dim=1)
        else:
            # Inverse pass
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], dim=1)
            log_det = -torch.sum(s, dim=1)
        
        return y, log_det

class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_layers=8):
        super(NormalizingFlow, self).__init__()
        
        self.layers = nn.ModuleList([
            CouplingLayer(dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, reverse=False):
        log_det_sum = 0
        
        layers = reversed(self.layers) if reverse else self.layers
        
        for layer in layers:
            x, log_det = layer(x, reverse=reverse)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_prob(self, x):
        """Compute log probability"""
        z, log_det = self.forward(x, reverse=False)
        
        # Base distribution (standard normal)
        log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        
        return log_prob_z + log_det
```

## Autoregressive Models

Generate data sequentially, one element at a time.

### PixelCNN

```python
class MaskedConv2d(nn.Conv2d):
    """Masked convolution for autoregressive generation"""
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        
        self.mask[:, :, :self.kernel_size[0] // 2] = 1
        self.mask[:, :, self.kernel_size[0] // 2, :self.kernel_size[1] // 2] = 1
        
        if mask_type == 'A':
            # Mask type A: exclude center pixel
            self.mask[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    def __init__(self, n_channels=1, n_filters=64, n_layers=7):
        super(PixelCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer (mask type A)
        self.layers.append(
            nn.Sequential(
                MaskedConv2d('A', n_channels, n_filters, 7, padding=3),
                nn.BatchNorm2d(n_filters),
                nn.ReLU()
            )
        )
        
        # Hidden layers (mask type B)
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    MaskedConv2d('B', n_filters, n_filters, 7, padding=3),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU()
                )
            )
        
        # Output layer
        self.output = nn.Conv2d(n_filters, n_channels * 256, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.output(x)
        
        # Reshape for pixel-wise softmax
        b, _, h, w = x.size()
        x = x.view(b, 256, -1, h, w)
        
        return x
```

## Energy-Based Models

Model probability as energy function: p(x) ∝ exp(-E(x))

```python
class EnergyBasedModel(nn.Module):
    def __init__(self, input_dim):
        super(EnergyBasedModel, self).__init__()
        
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def energy(self, x):
        """Compute energy E(x)"""
        return self.energy_net(x)
    
    def sample_langevin(self, x, n_steps=100, step_size=0.01):
        """Sample using Langevin dynamics"""
        x = x.clone().detach().requires_grad_(True)
        
        for _ in range(n_steps):
            energy = self.energy(x).sum()
            grad = torch.autograd.grad(energy, x)[0]
            
            noise = torch.randn_like(x) * np.sqrt(step_size * 2)
            x = x - step_size * grad + noise
        
        return x.detach()
```

## Diffusion Models

Gradually add noise then learn to denoise.

### DDPM (Denoising Diffusion Probabilistic Models)

```python
class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Noise prediction network (U-Net)
        self.noise_predictor = self._build_unet()
    
    def _build_unet(self):
        """Simple U-Net for noise prediction"""
        # Simplified version
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, xt, t):
        """Reverse diffusion: denoise xt"""
        # Predict noise
        predicted_noise = self.noise_predictor(xt)
        
        # Compute x_{t-1}
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        x0_pred = (xt - ((1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()) * predicted_noise) / alpha_t.sqrt()
        
        if t > 0:
            noise = torch.randn_like(xt)
            x_prev = x0_pred * alpha_t.sqrt() + (1 - alpha_t).sqrt() * noise
        else:
            x_prev = x0_pred
        
        return x_prev
    
    def sample(self, shape):
        """Generate samples"""
        device = next(self.parameters()).device
        
        # Start from random noise
        x = torch.randn(shape).to(device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        
        return x
```

## Evaluation Metrics

```python
# Inception Score (IS)
def inception_score(imgs, splits=10):
    """Higher is better"""
    from torchvision.models import inception_v3
    
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    
    # Get predictions
    with torch.no_grad():
        preds = inception_model(imgs)
        preds = F.softmax(preds, dim=1)
    
    # Compute IS
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]
        py = part.mean(dim=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append((pyx * (torch.log(pyx) - torch.log(py))).sum())
        split_scores.append(torch.exp(torch.mean(torch.stack(scores))))
    
    return torch.mean(torch.stack(split_scores)), torch.std(torch.stack(split_scores))

# Fréchet Inception Distance (FID)
def calculate_fid(real_imgs, fake_imgs):
    """Lower is better"""
    # Extract features using Inception network
    # Calculate mean and covariance
    # Compute FID score
    pass
```

## Practical Tips

1. **GAN Training**: Balance G and D, use label smoothing, add noise to inputs
2. **VAE**: Choose appropriate beta value, use warm-up for KL term
3. **Stability**: Monitor losses, use spectral normalization
4. **Architecture**: Start simple, gradually add complexity
5. **Evaluation**: Use multiple metrics (IS, FID, visual inspection)

## Resources

- "Generative Deep Learning" by David Foster
- OpenAI papers: https://openai.com/research/
- Distill.pub: https://distill.pub/
- Papers with Code: https://paperswithcode.com/

