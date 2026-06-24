# Diffusion Models

The generative family behind modern image, video, and audio synthesis. A diffusion model
learns to **reverse a gradual noising process**: take data, destroy it step by step into pure
noise, and train a network to undo one step at a time. Sampling then starts from noise and
denoises into a fresh sample. This page covers the mechanics; for the applied Stable Diffusion
stack and tooling see [Stable Diffusion](../ai/stable_diffusion.md).

## Table of Contents

1. [Overview](#overview)
2. [The Forward (Noising) Process](#the-forward-noising-process)
3. [The Reverse (Denoising) Process](#the-reverse-denoising-process)
4. [The Training Objective](#the-training-objective)
5. [Sampling: DDPM vs DDIM](#sampling-ddpm-vs-ddim)
6. [Conditioning & Classifier-Free Guidance](#conditioning--classifier-free-guidance)
7. [Latent Diffusion](#latent-diffusion)
8. [Architectures: U-Net & DiT](#architectures-u-net--dit)
9. [Score-Based & Flow View](#score-based--flow-view)
10. [Where this connects](#where-this-connects)
11. [Pitfalls](#pitfalls)

## Overview

Earlier generative models trade off in known ways: [GANs](generative_models.md) give sharp
samples but train unstably (the adversarial min-max game), while VAEs train stably but produce
blurry samples. **Diffusion models** sidestep both — a stable, simple regression loss that
produces high-fidelity, diverse samples — at the cost of **slow, iterative sampling** (many
forward passes per sample).

```
   forward  q:   x₀ ──► x₁ ──► … ──► x_T        add a little Gaussian noise each step
            (data)                  (≈ pure noise)     fixed, no learning

   reverse  pθ:  x_T ──► … ──► x₁ ──► x₀         learned network removes noise step by step
            (noise)                 (sample)
```

The whole model is the **reverse** process; the forward process is a fixed, hand-designed
schedule with no parameters. Training teaches the network to invert one noising step, which is
a far easier target than modeling the data distribution directly.

## The Forward (Noising) Process

The forward process `q` adds Gaussian noise over `T` timesteps according to a **variance
schedule** `β₁…β_T`. A convenient property: you can jump to *any* timestep in closed form,
without simulating the intermediate steps:

```
   x_t = √(ᾱ_t) · x₀  +  √(1 − ᾱ_t) · ε ,     ε ~ N(0, I)

   α_t = 1 − β_t ,   ᾱ_t = Π_{s≤t} α_s        (cumulative product)
```

As `t → T`, `ᾱ_t → 0`, so `x_T` is essentially standard Gaussian noise regardless of `x₀`.
The schedule (linear, **cosine**) controls how fast information is destroyed; a cosine schedule
keeps more signal in the middle steps and tends to train better.

## The Reverse (Denoising) Process

The reverse process is also Gaussian (for small steps), and the network parameterizes its mean.
In practice the network doesn't predict the clean image or the mean directly — it predicts the
**noise `ε`** that was added:

```
   network  ε_θ(x_t, t)  ≈  the noise ε that produced x_t from x₀

   one reverse step then "subtracts" the predicted noise (scaled by the schedule)
   and adds a little fresh noise, to go from x_t → x_{t-1}
```

Predicting noise (the **ε-prediction** parameterization from DDPM) is what reduces the whole
thing to a simple regression. Variants predict `x₀` directly or a velocity `v` — equivalent
targets that differ in numerical conditioning across the schedule.

## The Training Objective

The remarkable result (Ho et al., DDPM 2020) is that the full variational bound simplifies to a
plain **MSE between true and predicted noise**:

```
   L = E_{x₀, t, ε} ‖ ε − ε_θ(x_t, t) ‖²

   per step:  sample x₀ from data
              sample a random timestep t  and noise ε
              form x_t = √(ᾱ_t)x₀ + √(1−ᾱ_t)ε
              regress the network to predict ε
```

No adversary, no posterior collapse — just a stable [loss](loss_functions.md) (the diffusion
bullet there points back here). The timestep `t` is fed to the network via a sinusoidal
[positional-style embedding](positional_encoding.md), so one network handles all noise levels.

## Sampling: DDPM vs DDIM

Generation runs the reverse chain from `x_T ~ N(0, I)` down to `x₀`. The naive **DDPM** sampler
takes one stochastic step per training timestep — typically **1000 forward passes**, which is
slow.

```
  DDPM   stochastic, ~1000 steps, high quality but expensive
  DDIM   deterministic, non-Markovian → skip steps → 20–50 steps for similar quality
```

**DDIM** (Song et al., 2021) reinterprets the process as deterministic, letting you take far
fewer, larger steps and even get a deterministic latent→image map (useful for interpolation and
inversion). Faster ODE/SDE solvers (DPM-Solver, etc.) push high-quality sampling down to ~10–20
steps — the main lever for inference cost, which connects to
[inference optimization](../ai/inference_optimization.md) on the serving side.

## Conditioning & Classifier-Free Guidance

To generate *a specific thing* (a text prompt, a class), the network is **conditioned** on `c`:
`ε_θ(x_t, t, c)`. The dominant technique to make conditioning strong is **classifier-free
guidance (CFG)**:

```
   train one network for BOTH conditional and unconditional (drop c with some probability)

   at sampling, extrapolate away from the unconditional prediction:
       ε̂ = ε_θ(x_t, t, ∅) + w · ( ε_θ(x_t, t, c) − ε_θ(x_t, t, ∅) )
                                  ╰── push harder toward the condition ──╯
   guidance scale w: higher → more prompt-faithful, less diverse (and can over-saturate)
```

CFG needs no separate classifier and is why the **guidance scale** is the knob you tune for
prompt adherence vs. variety. Text conditioning typically comes from a [CLIP](self_supervised_learning.md)
or T5 text encoder, injected via cross-[attention](attention.md).

## Latent Diffusion

Running diffusion directly on pixels is expensive (a 512×512 image is ~786k values). **Latent
Diffusion Models** (Rombach et al., 2022 — the basis of Stable Diffusion) first compress images
into a small latent space with a [VAE](generative_models.md), then run the entire diffusion
process **in that latent space**:

```
   image ──[VAE encoder]──► latent z  (e.g. 64×64×4, ~48× smaller)
                              │
                       diffusion happens HERE  (cheap)
                              │
   latent ẑ ──[VAE decoder]──► image
```

This is the single change that made high-resolution text-to-image practical on consumer GPUs.
The applied stack (schedulers, LoRA, ControlNet, ecosystems) lives in
[Stable Diffusion](../ai/stable_diffusion.md); [quantization](quantization.md) further shrinks
the UNet/transformer for deployment.

## Architectures: U-Net & DiT

The denoiser `ε_θ` needs to map a noisy input to a same-shaped noise prediction:

```
  U-Net   conv encoder–decoder with skip connections + cross-attention for conditioning
          (the original and still-common backbone; uses GroupNorm, SiLU, residual blocks)

  DiT     Diffusion Transformer — replace the U-Net with a transformer over latent patches
          (Peebles & Xie, 2023); scales cleanly like other transformers, used in modern
          large image/video models
```

DiT shows the field converging on [transformers](transformers.md): patchify the latent, add
timestep/condition tokens, run a standard transformer stack. This reuses everything from
[normalization](normalization.md) (adaLN) to [attention](attention.md) kernels.

## Score-Based & Flow View

Diffusion has an equivalent **score-based** formulation: the network learns the **score**
`∇ₓ log p(x)` (the gradient of log-density), and sampling solves a reverse-time stochastic
differential equation (Song et al.). The ε-prediction and score views are the same model up to
scaling. A related, increasingly popular framing is **flow matching / rectified flow**, which
learns a straight-line velocity field between noise and data and often samples in fewer steps.
These unify diffusion with continuous [normalizing flows](deep_generative_models.md).

## Where this connects

- [Generative models](generative_models.md) and [deep generative models](deep_generative_models.md)
  — diffusion alongside GANs, VAEs, and flows; the code-level DDPM/latent-diffusion implementations
- [Stable Diffusion](../ai/stable_diffusion.md) — the applied text-to-image stack, schedulers,
  LoRA, ControlNet, and tooling built on latent diffusion
- [Loss functions](loss_functions.md) — the noise-prediction MSE objective
- [Self-supervised learning](self_supervised_learning.md) — CLIP text/image encoders that condition
  generation
- [Transformers](transformers.md) and [attention](attention.md) — DiT backbones and cross-attention
  conditioning
- [Positional encodings](positional_encoding.md) — sinusoidal timestep embeddings
- [Quantization](quantization.md) and [inference optimization](../ai/inference_optimization.md) —
  making sampling cheap enough to serve
- [Normalization](normalization.md) — GroupNorm in U-Nets, adaLN in DiT

## Pitfalls

- **Too few sampling steps** — aggressive step-skipping degrades quality; match the sampler
  (DDIM/DPM-Solver) to the step budget rather than just truncating DDPM.
- **Guidance scale too high** — over-saturated, low-diversity, artifact-prone images; tune `w`.
- **Schedule mismatch** — training and sampling must use a consistent noise schedule; a cosine
  schedule usually beats linear for images.
- **Pixel-space diffusion at high resolution** — prohibitively expensive; use latent diffusion.
- **Ignoring the ε vs x₀ vs v parameterization** — the target choice affects stability across
  noise levels, especially near `t=0` and `t=T`.
- **VAE bottleneck artifacts (latent diffusion)** — a weak autoencoder caps final image quality
  no matter how good the diffusion model is.
- **Expecting GAN-speed sampling** — diffusion is inherently iterative; budget for many forward
  passes or adopt distillation/few-step methods.
