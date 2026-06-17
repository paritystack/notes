# Machine Learning — deep per-page correctness review

Per-page review of all 25 pages in `src/machine_learning/`. Each page's code, example
outputs, formulas, dates/attributions, and prose were checked. Severity: **ERR** = wrong,
**SUS** = questionable/misleading, **NIT** = minor/style.

## Summary

The corpus is high quality — concepts, formulas, algorithm implementations, paper
dates/authors, and hardware figures are almost all correct. As in the algorithms review, the
bugs found were concentrated in **computed example outputs** (comments asserting what a snippet
prints) plus a few attribution/notation slips.

**Errors fixed inline (10):**
1. `convolution.md` — FLOPs example `112×112×7×7×3×64×2` labeled `≈118M`; that product is
   **236M FLOPs** (118M is the MAC count, i.e. without the ×2). → annotated `≈236M (≈118M MACs)`.
2. `lora.md` — A/B initialization reversed: said "B random Gaussian, A zero". The LoRA paper
   (and this file's own code + LoRA+ note) use **A = random Gaussian, B = 0**. → fixed.
3. `lora.md` — Pattern 1 `print_trainable_parameters()` showed `4,194,304 / 0.06%` (the r=8
   count) under an `r=16` config. For r=16 on q_proj+v_proj across Llama-2-7B's 32 layers it's
   `16×(4096+4096)×2×32 = 8,388,608` (~0.12%). → fixed.
4. `moe.md` — "Soft MoE (Meta, 2023)" — Soft MoE (Puigcerver, Riquelme et al.) is **Google
   DeepMind**, not Meta. → fixed.
5. `metrics.md` — MSE temperature example: `0.7240` → **0.7420** (verified 3.71/5).
6. `metrics.md` — RMSE stock example: `$0.52` → **$0.48** (verified √(1.17/5)).
7. `metrics.md` — R² exam example: `0.8869` → **0.9281** (verified 1 − 23/320).
8. `metrics.md` — MAPE sales example: `~6.40%` → **~6.60%**.
9. `metrics.md` — Perplexity formula `= 2^(-entropy)` is a sign error; perplexity ≥ 1 so the
   exponent must be positive. → `= 2^(entropy)` with `entropy = -1/N Σ log₂ P`.
10. `thinking.md` — Tree-of-Thoughts example labeled `(3x-1)(x+2)=0` as "Wrong, pruned", but
    that *is* the correct factorization of `3x²+5x-2`. → changed to `(3x+1)(x-2)=0` (genuinely
    wrong, expands to `3x²-5x-2`).

**Verified clean (no changes):** README, gradient_boosting, supervised_learning,
neural_networks, transformers, interesting_papers, deep_learning, reinforcement_learning,
deep_reinforcement_learning, generative_models, deep_generative_models, unsupervised_learning,
transfer_learning, quantization, numpy, jax, pytorch, cuda, hugging_face.

---

## Per-page notes

- **README.md** — OK. Index/overview; pipeline, bias-variance, regularization formulas correct.
- **gradient_boosting.md** — OK. Bagging/boosting, XGBoost/LightGBM/CatBoost innovations correct.
- **supervised_learning.md** — OK. All algorithm descriptions and sklearn usage correct.
- **neural_networks.md** — OK. Activation table, optimizer table, ZeRO stages, Mamba/S4 claims,
  mixed-precision speedups all sound.
- **transformers.md** — OK. Scaled-dot-product/multi-head/positional-encoding math correct;
  worked-example scaling (√8≈2.83) self-consistent.
- **interesting_papers.md** — OK. **All ~45 dates and author lists verified** (AlexNet 2012
  26%→15.3%, ResNet-152 2015, GPT-1/2/3/4 sizes, DDPM, etc.).
- **deep_learning.md** — OK. Backprop, losses, ResNet/Inception/SE, ViT, attention all correct.
- **reinforcement_learning.md** — OK. MDP/Bellman, DP, MC, TD(λ), Q-learning, Double-Q, SARSA,
  REINFORCE, A2C, bandits/UCB all correct.
- **deep_reinforcement_learning.md** — OK. DQN/Double/Dueling/PER, REINFORCE+baseline, A2C/A3C,
  PPO (clip+GAE), DDPG, SAC, TD3 all correct.
- **convolution.md** — FLOPs example fixed (#1). Rest verified: 1D/2D conv arithmetic, output-size
  formulas, depthwise-separable reduction (~8.7×), Winograd 2.25×, receptive-field formula.
- **quantization.md** — OK. FP32/FP16/BF16/INT tables, A100 throughput (19.5/312/624), Horowitz
  energy figures, NF4 levels, GGUF/K-quant/I-quant, paper dates all correct.
- **lora.md** — Two fixes (#2, #3). QLoRA memory table, 10,000× claim, double-quant 0.37 bits OK.
- **moe.md** — Soft MoE attribution fixed (#4). Switch 1.6T, Mixtral 8x7B (47B/13B active),
  GShard 600B, GLaM 1.2T, DeepSeek-V2 236B/21B, load-balance/z-loss math all correct.
- **generative_models.md** — OK (DDPM `p_sample` is loose but flagged "simplified").
- **deep_generative_models.md** — OK. GPT/ViT-VQGAN, improved-DDPM (cosine schedule, correct
  posterior-mean sampling), latent diffusion (4×64×64, guidance 7.5), VQ-VAE, NeRF rendering.
- **unsupervised_learning.md** — OK. K-means/DBSCAN/GMM/spectral, PCA/t-SNE/UMAP/LDA/NMF, anomaly
  detection, KDE, Apriori, clustering metrics.
- **transfer_learning.md** — OK. SimCLR NT-Xent, BERT MLM 80/10/10, DANN, MMD, ProtoNet, MAML,
  distillation (T² scaling); LoRA section uses correct A-random/B-zero init.
- **metrics.md** — Five fixes (#5–#9). All other formulas (F1 harmonic, MCC, Kappa, ROC/PR, BLEU,
  ROUGE, NDCG, MRR, IoU, mAP) and paper citations verified correct.
- **numpy.md** — OK. Verified arange/linspace, fancy-index, broadcasting, Strassen [[19,22],[43,50]],
  einsum attention, `np.choose`→[1,20,300,40,5], `add.at`, meshgrid outputs.
  - NIT: `np.logspace(0,2,5)` comment shows `[1, 10, 100]` (3 of the 5 points); loose but labeled.
- **jax.md** — OK (jit/grad/vmap/pmap/pytrees reference).
- **pytorch.md** — OK. autograd `dz/dx = 2x+y = [5,8]`, tensor ops, nn.Module all correct.
- **cuda.md** — OK. Memory-hierarchy bandwidth/latency table and coalescing GB/s figures are
  reasonable order-of-magnitude values for modern (A100-class) GPUs.
- **hugging_face.md** — OK (transformers/datasets API reference).
