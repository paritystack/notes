# Distributed Training & Parallelism

How to train a model that no longer fits — in memory or in time — on a single GPU, by
splitting the work across many devices along different axes.

## Table of Contents

1. [Overview](#overview)
2. [The Memory Budget](#the-memory-budget)
3. [Data Parallelism (DDP)](#data-parallelism-ddp)
4. [Sharded Data Parallelism: ZeRO and FSDP](#sharded-data-parallelism-zero-and-fsdp)
5. [Tensor Parallelism](#tensor-parallelism)
6. [Pipeline Parallelism](#pipeline-parallelism)
7. [Sequence and Context Parallelism](#sequence-and-context-parallelism)
8. [3D Parallelism: Composing the Axes](#3d-parallelism-composing-the-axes)
9. [Supporting Techniques](#supporting-techniques)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Where this connects](#where-this-connects)

## Overview

A modern model hits two walls. The first is **memory**: the weights, their gradients, and
the [optimizer](optimizers.md) state for a multi-billion-parameter [transformer](transformers.md)
do not fit in one GPU's HBM. The second is **time**: even if it fit, one device would take
months to see enough tokens. Distributed training breaks through both walls by spreading a
single logical training step across many GPUs connected by fast interconnects (NVLink within
a node, InfiniBand/Ethernet across nodes).

There are two fundamental things you can split, and almost every technique is a combination
of the two:

```
  DATA PARALLELISM                      MODEL PARALLELISM
  split the BATCH across GPUs           split the MODEL across GPUs
  every GPU has a full model copy       every GPU holds a slice of the weights
  GPUs sync GRADIENTS                   GPUs sync ACTIVATIONS
  scales throughput                     scales the model size that fits
```

Data parallelism is the workhorse — it is how you go faster. Model parallelism (in its
tensor, pipeline, and sharded forms) is how you fit a model that is simply too big for one
device. Real large-model training stacks compose all of them. This page builds up from the
[neural network](neural_networks.md) training loop you already know — forward, backward via
autograd in [PyTorch](pytorch.md), optimizer step — and shows where the communication gets
inserted. Everything ultimately rides on [CUDA](cuda.md) kernels and collective communication
primitives (NCCL).

## The Memory Budget

Before choosing a strategy, you have to know what is eating the memory. For a model trained
with **mixed precision + Adam**, per parameter you store roughly:

```
  per parameter (Adam + mixed precision, bytes):
  ┌────────────────────────────┬───────┐
  │ fp16/bf16 weight           │   2   │
  │ fp16/bf16 gradient         │   2   │
  │ fp32 master weight copy    │   4   │   ← optimizer state
  │ fp32 Adam momentum  (m)    │   4   │   ← optimizer state
  │ fp32 Adam variance  (v)    │   4   │   ← optimizer state
  ├────────────────────────────┼───────┤
  │ TOTAL                      │  16   │  ≈ 16 bytes / param
  └────────────────────────────┴───────┘

  → a 7B model needs ~112 GB JUST for weights+grads+optimizer state,
    before a single activation. That already exceeds an 80 GB GPU.
```

On top of that sit **activations** — the intermediate tensors saved during the forward pass
so the backward pass can compute gradients. Activation memory scales with batch size,
sequence length, and depth, and for long sequences it often dominates everything else. The
two big levers are therefore: shard the fixed `16×params` state (ZeRO/FSDP), and shrink or
recompute activations (gradient checkpointing). [Quantization](quantization.md) attacks the
same budget from the numeric-precision side; here we attack it by *distributing* it.

## Data Parallelism (DDP)

The simplest scaling strategy. Replicate the full model on every GPU, give each GPU a
different shard of the batch, run forward and backward independently, then **all-reduce** the
gradients so every replica applies an identical update and the copies stay in lockstep.

```
  GPU0: model copy ── fwd/bwd on batch shard 0 ─┐
  GPU1: model copy ── fwd/bwd on batch shard 1 ─┤ all-reduce
  GPU2: model copy ── fwd/bwd on batch shard 2 ─┤ gradients  → identical step
  GPU3: model copy ── fwd/bwd on batch shard 3 ─┘   (mean)
```

The synchronization is a **ring all-reduce**: gradients are bucketed and each GPU sends/
receives chunks around a ring so bandwidth cost is independent of the number of GPUs. The key
performance trick in `torch.nn.parallel.DistributedDataParallel` is that it **overlaps**
communication with computation — gradient buckets are all-reduced as soon as they are ready
during the backward pass, hiding the network latency behind the still-running backprop.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")               # one process per GPU
torch.cuda.set_device(local_rank)
model = DDP(model.cuda(), device_ids=[local_rank])
# DistributedSampler gives each rank a disjoint shard of the data
for batch in loader:
    loss = model(batch).loss
    loss.backward()                            # grads all-reduced here, overlapped
    optimizer.step(); optimizer.zero_grad()
```

DDP makes you *faster* (bigger effective batch, more tokens/sec) but does **not** let you fit
a bigger model — every GPU still holds a full copy. That is the limitation ZeRO removes.

## Sharded Data Parallelism: ZeRO and FSDP

DDP wastes memory: the `16×params` of optimizer state is replicated identically on every GPU.
**ZeRO** (Zero Redundancy Optimizer) observes that this redundancy is unnecessary and shards
the state across the data-parallel group, reconstructing pieces on demand. It comes in three
cumulative stages:

```
  ZeRO stage   shards...                        memory/GPU (N data-parallel ranks)
  ──────────   ──────────────────────────────   ──────────────────────────────────
  stage 1      optimizer state                   weights + grads + (opt-state / N)
  stage 2      + gradients                        weights + (grads + opt-state) / N
  stage 3      + parameters                        (weights + grads + opt-state) / N
```

Stage 3 shards *everything*, so per-GPU memory drops nearly linearly with the number of
ranks. The cost is communication: each GPU holds only a slice of the weights, so before a
layer runs it must **all-gather** the full parameters for that layer, use them, then free
them again. Gradients are reduced with **reduce-scatter** so each rank keeps only its shard.

**FSDP** (`FullyShardedDataParallel`) is PyTorch's native implementation of this idea
(equivalent to ZeRO-3). It wraps the model in units; for each unit it all-gathers params just
in time for the forward/backward, runs the computation, then discards the gathered copy:

```
  per FSDP unit, during forward:
    all-gather params → compute → free params
  during backward:
    all-gather params → compute grads → reduce-scatter grads → free
```

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=transformer_wrap_policy)
```

FSDP/ZeRO-3 is the default way to train models that are too big for one GPU but whose single
*layers* still fit comfortably. When even a single layer's matmul is too large, you need
tensor parallelism.

## Tensor Parallelism

Tensor parallelism (the Megatron-LM style) splits an **individual operation** — a matrix
multiply — across GPUs, rather than splitting batches or layers. A weight matrix is
partitioned and each GPU computes part of the output.

For an MLP `Y = GeLU(X·A)·B`, you split `A` **column-wise** and `B` **row-wise** so that the
two halves compose with a single all-reduce at the end:

```
            ┌──────── GPU0: X·A₀ → GeLU → ·B₀ ─┐
   X  ──────┤                                   ├─(all-reduce)──► Y
            └──────── GPU1: X·A₁ → GeLU → ·B₁ ─┘
   A = [A₀ | A₁]  (split columns)   B = [B₀ ; B₁]  (split rows)
```

The same pattern applies to [attention](attention.md): the heads are partitioned across GPUs
(each GPU computes a subset of heads), with an all-reduce after the output projection. Tensor
parallelism communicates **twice per transformer layer** (once in attention, once in the MLP)
and the messages are large, so it is bandwidth-hungry — it is almost always confined to GPUs
**within a single node** connected by NVLink. This is also exactly how
[mixture-of-experts](moe.md) layers shard their experts (expert parallelism is a cousin).

## Pipeline Parallelism

Pipeline parallelism splits the model **by layers** (by depth). Stage 0 holds the first
chunk of layers, stage 1 the next, and so on. The naive version is terrible — only one stage
works at a time while the rest idle. The fix is to split the batch into **micro-batches** and
flow them through the stages like an assembly line:

```
  time ───────────────────────────────────────────►
  stage0  F1 F2 F3 F4 .. .. .. B4 B3 B2 B1
  stage1     F1 F2 F3 F4 .. .. B4 B3 B2 B1
  stage2        F1 F2 F3 F4 B4 B3 B2 B1
  stage3           F1 F2 F3 F4 B4 B3 B2 B1
                 ↑ fill        ↑ drain
                 (the "bubble" = idle time at start/end)
```

The idle triangles at fill and drain are the **pipeline bubble**; more micro-batches amortize
it (bubble fraction ≈ `(stages − 1) / micro_batches`). Schedules like **GPipe** (all forwards,
then all backwards) and **1F1B** (one-forward-one-backward, interleaved) trade bubble size
against activation-memory pressure. Pipeline communication is small (just the activations
crossing stage boundaries), so unlike tensor parallelism it tolerates slower cross-node links.

## Sequence and Context Parallelism

For very long contexts, the activations and attention computation along the **sequence
dimension** become the bottleneck. Sequence/context parallelism shards the sequence itself
across GPUs, with each device responsible for a slice of tokens and exchanging the partial
key/value information needed to complete [attention](attention.md) (e.g. Ring Attention passes
KV blocks around a ring). This is the training-time analog of why
[state space models](state_space_models.md) chase sub-quadratic sequence scaling — it makes
million-token context lengths tractable without any single GPU holding the whole sequence.

## 3D Parallelism: Composing the Axes

The axes are orthogonal, so frameworks compose them. "3D parallelism" means **data ×
tensor × pipeline**, often with ZeRO sharding layered on the data axis and expert parallelism
for [MoE](moe.md) models as a further dimension:

```
  a 64-GPU job, arranged as 4 × 2 × 8:
    tensor-parallel = 2   ── within a node (NVLink), splits each matmul
    pipeline-parallel = 8 ── across nodes, splits layers into stages
    data-parallel = 4     ── replicas of the whole TP×PP block, sync grads
```

The guiding rule is to **match each axis to the interconnect it needs**: tensor parallelism
on the fastest intra-node links (it talks the most), pipeline and data parallelism across
slower inter-node links (they talk less). Frameworks that orchestrate this include
**DeepSpeed** (ZeRO + pipeline), **Megatron-LM** (tensor + pipeline), and native **PyTorch
FSDP** combined with tensor/pipeline APIs.

## Supporting Techniques

These are not parallelism strategies themselves but are essential companions:

- **Mixed precision** — compute in `bf16`/`fp16` for speed and memory, keep an `fp32` master
  copy for the optimizer update. `bf16` has the same exponent range as `fp32` and avoids the
  **loss-scaling** gymnastics that `fp16` needs to keep small gradients from underflowing.
- **Gradient checkpointing** (activation recomputation) — don't store every activation; store
  a few and **recompute** the rest during backward. Trades extra compute (~30%) for a large
  drop in activation memory, often the single biggest memory win for long sequences.
- **Gradient accumulation** — run several micro-batches and sum their gradients before
  stepping, simulating a larger batch than fits in memory. (Covered in more depth in
  [Optimizers](optimizers.md).)
- **CPU / NVMe offload** — ZeRO-Infinity pushes optimizer state and even parameters to CPU
  RAM or SSD, trading bandwidth for the ability to train enormous models on modest hardware.

## Best Practices

- **Use the least parallelism that fits.** Pure DDP (or FSDP) is simplest and most efficient;
  reach for tensor/pipeline parallelism only when the model or its layers won't fit otherwise.
- **Map axes to interconnects:** tensor parallelism *inside* a node, pipeline and data
  parallelism *across* nodes.
- **Prefer `bf16`** over `fp16` on hardware that supports it — no loss scaling, fewer
  divergence headaches.
- **Turn on gradient checkpointing** before adding more model parallelism; recomputation is
  often cheaper than the communication a new parallel axis introduces.
- **Scale the learning rate with the global batch size** (and use [warmup](optimizers.md)) —
  the effective batch is `per_gpu_batch × data_parallel_size × grad_accum_steps`.
- **Profile communication.** If GPUs sit idle waiting on all-reduce, you are network-bound,
  not compute-bound — fix overlap, bucket sizes, or topology before buying more GPUs.

## Common Pitfalls

- **Forgetting to scale the LR** when you scale the global batch — training silently
  underfits or diverges.
- **Treating all-reduce as free.** At large scale gradient synchronization can dominate step
  time; without computation/communication overlap, DDP stalls.
- **Pipeline bubble** eating your speedup — too few micro-batches relative to the number of
  stages leaves GPUs idle at fill and drain.
- **Uneven sharding** — a batch or sequence length not divisible by the parallel degree
  causes hangs or wasted padding; collectives deadlock if one rank takes a different code path.
- **Checkpoint/resume mismatch** — sharded (FSDP/ZeRO) checkpoints are saved per-rank; loading
  them under a different parallel layout without consolidation corrupts or fails the restore.
- **`fp16` overflow/underflow** — without loss scaling, small gradients vanish; this is the
  whole reason `bf16` is preferred when available.

## Where this connects

- [Optimizers](optimizers.md) — the optimizer state is what ZeRO shards, and gradient
  accumulation / LR warmup live here.
- [Neural Networks](neural_networks.md) and [PyTorch](pytorch.md) — the single-device training
  loop these techniques wrap; DDP/FSDP are `torch.distributed` modules.
- [JAX](jax.md) — expresses the same ideas declaratively via `pmap`/`shard_map` and `pjit`
  sharding annotations instead of explicit collectives.
- [CUDA](cuda.md) — the collective primitives (NCCL all-reduce, all-gather, reduce-scatter)
  and the kernels everything runs on.
- [Quantization](quantization.md) — an orthogonal way to shrink the same memory budget by
  lowering numeric precision.
- [Mixture-of-Experts](moe.md) — expert parallelism is a fourth parallel axis; sparse models
  are a primary reason to scale past one node.
- [Attention](attention.md) and [State Space Models](state_space_models.md) — sequence/context
  parallelism and sub-quadratic architectures both target long-context scaling.
- For the **inference** side of the same coin — tensor/pipeline parallelism applied to
  *serving* rather than training — see [`../ai/inference_optimization.md`](../ai/inference_optimization.md)
  and [`../ai/vllm.md`](../ai/vllm.md).
