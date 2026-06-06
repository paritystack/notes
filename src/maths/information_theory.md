# Information Theory

## Overview

Information theory, founded by Claude Shannon in 1948, quantifies *information*,
*uncertainty*, and the limits of compression and communication. It is built on
[probability](probability.md) — every quantity here is an expectation of a log-
probability — and it underpins [machine learning](../machine_learning/index.html) loss
functions (cross-entropy is *the* classification loss), data compression, error-correcting
codes, and the security intuition behind [hashing](../security/hashing.md) and
[encryption](../security/encryption.md).

```
Core idea: a rare event carries more information than a common one.
           "the sun rose"     → ~0 bits (certain)
           "it snowed in July" → many bits (surprising)

Information content of an outcome:  I(x) = −log₂ p(x)   (bits)
```

## Entropy — the amount of uncertainty

Entropy is the *expected* information content of a random variable — the average number
of bits needed to describe its outcomes, and the hard floor on lossless compression.

```
H(X) = − Σ p(x) · log₂ p(x)        (bits, when log base 2)
```

```
Fair coin   p=½,½        H = 1 bit          (maximum uncertainty for 2 outcomes)
Biased coin p=0.9,0.1    H ≈ 0.47 bits      (more predictable → less info)
Certain     p=1,0        H = 0 bits         (no surprise, nothing to encode)

Entropy is MAXIMIZED by the uniform distribution and MINIMIZED (0) by a certainty.
```

**Shannon's source coding theorem**: you cannot losslessly compress a source below `H(X)`
bits/symbol on average. Huffman coding and arithmetic coding approach this bound; this is
why already-random data (encrypted, or already-compressed) can't be compressed further —
it's already near maximum entropy.

## Cross-entropy and KL divergence

These are *the* link between information theory and machine learning.

```
Cross-entropy  H(p, q) = − Σ p(x) · log q(x)
   = expected bits to encode samples from the TRUE p using a code built for MODEL q.

KL divergence  D_KL(p ‖ q) = Σ p(x) · log( p(x) / q(x) ) = H(p,q) − H(p)
   = the EXTRA bits paid for using the wrong distribution q instead of p.
```

Properties that matter:

```
D_KL(p ‖ q) ≥ 0,   = 0  iff  p = q          (Gibbs' inequality)
D_KL is NOT symmetric:  D_KL(p‖q) ≠ D_KL(q‖p)   → it is not a true distance.
```

In a classifier the labels `p` are fixed, so minimizing cross-entropy `H(p,q)` is exactly
minimizing `D_KL(p‖q)` — driving the model `q` toward the truth `p`. That is why
**cross-entropy loss = maximum likelihood** for classification. See
[ML metrics](../machine_learning/metrics.md). KL also appears as the regularizer in
variational autoencoders and as the trust-region term in RLHF/PPO.

## Mutual information — shared information

How much knowing one variable reduces uncertainty about another:

```
I(X; Y) = H(X) − H(X | Y) = H(Y) − H(Y | X)
        = Σ p(x,y) · log( p(x,y) / (p(x)·p(y)) )
        = D_KL( p(x,y) ‖ p(x)·p(y) )

I(X; Y) = 0   ⟺   X and Y are independent.
```

Mutual information is a *general* dependence measure — unlike correlation, it captures
non-linear relationships. Uses: feature selection, the information-bottleneck view of deep
networks, registration in imaging, and clustering evaluation.

```
Relationships (a Venn diagram of bits):

   H(X)            H(Y)
 ┌────────┬──────────────┐
 │ H(X|Y) │  I(X;Y)  │ H(Y|X) │
 └────────┴──────────────┘
       H(X,Y) = H(X) + H(Y) − I(X;Y)
```

## Conditional entropy and chain rule

```
H(X | Y) = H(X, Y) − H(Y)        uncertainty in X once Y is known
H(X, Y)  = H(X) + H(Y | X)        chain rule
```

## Channel capacity & coding

Shannon's **noisy-channel coding theorem** says every channel has a maximum reliable
rate — the **capacity** `C` — and *any* rate below `C` can be achieved with arbitrarily
low error using a good enough code:

```
C = max_{p(x)} I(X; Y)         bits per channel use
```

This split modern communications into two solvable halves:

```
Source coding   : remove redundancy   → compress to ~H bits      (gzip, JPEG, H.264)
Channel coding  : add structured redundancy → survive noise      (Hamming, Reed–Solomon,
                                                                   LDPC, turbo codes)
```

Error-correcting codes (Reed–Solomon, LDPC) are how CDs, QR codes, deep-space links, RAID,
and 5G tolerate corruption — directly relevant to [embedded](../embedded/index.html) and
storage.

## Where this shows up

- **ML** — cross-entropy loss, KL regularization (VAEs, PPO), decision-tree information
  gain (`= mutual information`), perplexity (`= 2^H`) for language models.
- **Compression** — entropy is the lower bound; Huffman/arithmetic coding the practice.
- **Security** — a key with `H` bits of entropy needs `~2^H` guesses; password and RNG
  strength are entropy statements. See [hashing](../security/hashing.md),
  [key management](../security/key_management.md).
- **Communications & storage** — capacity and ECC bound throughput and reliability.

## Pitfalls

- **Bits vs nats** — `log₂` gives bits, `ln` gives nats; ML code usually uses nats.
- **KL asymmetry** — `D_KL(p‖q)` ≠ `D_KL(q‖p)`; "forward" vs "reverse" KL give different
  mode-covering vs mode-seeking behaviour in ML.
- **Entropy ≠ value** — high entropy means unpredictable, not useful or meaningful.
- **Estimating entropy from few samples** — biased low; needs care for large alphabets.
