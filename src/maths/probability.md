# Probability

## Overview

Probability is the mathematics of uncertainty — it assigns numbers between 0 and 1 to
events and gives consistent rules for combining them. It is the engine underneath
[Statistics](statistics.md) (which reasons *backwards* from data to the process that
produced it), all of [machine learning](../machine_learning/index.html) (likelihoods,
loss functions, sampling), [information theory](information_theory.md) (entropy is an
expectation), and randomized [algorithms](../algorithms/index.html). Where statistics
asks "what does this data tell me about the world?", probability asks "given a model of
the world, what data should I expect?"

```
Probability  : model ──► data        (forward, deductive)
Statistics   : data  ──► model        (inverse, inductive)
```

## Sample spaces and events

```
Sample space Ω : set of all possible outcomes      (a die: {1,2,3,4,5,6})
Event A        : a subset of Ω                      ("even": {2,4,6})
P(A)           : probability of A,  0 ≤ P(A) ≤ 1
```

**Kolmogorov axioms** — everything else is derived from these three:

```
1. P(A) ≥ 0
2. P(Ω) = 1
3. If A, B disjoint:  P(A ∪ B) = P(A) + P(B)
```

Useful consequences:

```
P(Aᶜ)      = 1 − P(A)                    (complement)
P(A ∪ B)   = P(A) + P(B) − P(A ∩ B)      (inclusion–exclusion)
P(∅)       = 0
```

## Conditional probability and independence

Conditioning is *updating* — restricting the world to the part where B happened and
renormalizing:

```
P(A | B) = P(A ∩ B) / P(B)        (defined when P(B) > 0)

Chain rule:  P(A ∩ B) = P(A | B) · P(B) = P(B | A) · P(A)
```

**Independence** means knowing B tells you nothing about A:

```
A ⟂ B   ⟺   P(A ∩ B) = P(A) · P(B)   ⟺   P(A | B) = P(A)
```

Watch the trap: *independent* ≠ *mutually exclusive*. Mutually exclusive events
(P(A∩B)=0) are maximally **dependent** — if one happens the other cannot.

## Bayes' theorem

The single most important formula for inference — it flips a conditional:

```
              P(B | A) · P(A)
P(A | B) = ─────────────────────
                  P(B)

         posterior = likelihood × prior / evidence
```

The classic gotcha is base rates. Disease affects 1% of people; a test is 99% accurate
both ways. You test positive — what's P(disease)?

```
P(D)=0.01   P(+|D)=0.99   P(+|¬D)=0.01
P(+) = 0.99·0.01 + 0.01·0.99 = 0.0198
P(D|+) = 0.99·0.01 / 0.0198 ≈ 0.50      ← only 50%, not 99%
```

Because the healthy 99% generate as many false positives as the sick 1% generate true
positives. This same machinery is naive Bayes classifiers and Bayesian inference.

## Random variables

A **random variable** maps outcomes to numbers, letting us do arithmetic on chance.

```
Discrete : countable values   → PMF  p(x) = P(X = x),    Σ p(x) = 1
Continuous: a range of values  → PDF  f(x),  P(X=x)=0,    ∫ f(x)dx = 1

CDF (both):  F(x) = P(X ≤ x)   — monotone, 0 → 1
```

For continuous variables only intervals have probability: `P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx`.

## Expectation, variance, moments

```
E[X]   = Σ x·p(x)   or   ∫ x·f(x) dx         the "center of mass"
Var(X) = E[(X − E[X])²] = E[X²] − E[X]²       spread
SD(X)  = √Var(X)
```

Key identities (memorize these — they save constant re-derivation):

```
E[aX + b]   = a·E[X] + b
Var(aX + b) = a²·Var(X)                 (shift doesn't change spread)
E[X + Y]    = E[X] + E[Y]               ALWAYS (even if dependent)
Var(X + Y)  = Var(X) + Var(Y) + 2·Cov(X,Y)
            = Var(X) + Var(Y)           iff independent
Cov(X,Y)    = E[XY] − E[X]E[Y]
Corr(X,Y)   = Cov(X,Y) / (SD(X)·SD(Y))  ∈ [−1, 1]
```

Linearity of expectation holds *without* independence — this is what makes many
probabilistic-algorithm proofs short.

## Common distributions

```
DISCRETE
  Bernoulli(p)        one trial, success/fail        E=p,        Var=p(1−p)
  Binomial(n,p)       # successes in n trials         E=np,       Var=np(1−p)
  Geometric(p)        # trials until first success    E=1/p,      Var=(1−p)/p²
  Poisson(λ)          # events in fixed interval      E=λ,        Var=λ

CONTINUOUS
  Uniform(a,b)        flat over [a,b]                 E=(a+b)/2
  Normal(μ,σ²)        bell curve                      E=μ,        Var=σ²
  Exponential(λ)      waiting time, memoryless        E=1/λ,      Var=1/λ²
  Beta(α,β)           values in [0,1]; prior on a probability
  Gamma(k,θ)          sum of exponentials; waiting times
```

The **Normal** dominates because of the next section. The **Poisson** approximates a
Binomial with large n and small p (rare events: server requests, decays). The
**Exponential** is the unique memoryless continuous distribution — the time already
waited tells you nothing about the time remaining.

## Law of Large Numbers & Central Limit Theorem

These two theorems are *why* statistics works.

```
Law of Large Numbers (LLN)
  Sample mean → true mean as n → ∞.
  "Averages stabilize." Justifies estimating E[X] by averaging samples.

Central Limit Theorem (CLT)
  The sum/mean of many independent variables is approximately Normal,
  regardless of the original distribution.

      X̄ₙ ≈ Normal(μ, σ²/n)   for large n

  This is why the bell curve is everywhere, and why standard error
  shrinks like 1/√n (quadruple the data → halve the error).
```

## Joint, marginal, conditional

```
Joint        p(x, y)              probability of both
Marginal     p(x) = Σ_y p(x,y)    "sum out" the other variable
Conditional  p(y | x) = p(x,y)/p(x)
```

This is the substrate of probabilistic graphical models and the "sum out a variable"
operation behind Bayesian networks.

## Markov chains

A process where the next state depends only on the current state, not the full history
(**the Markov property**). Encoded as a transition matrix `P` where `P[i][j] = P(next=j | now=i)`.

```
state distribution after one step:   πₜ₊₁ = πₜ · P
stationary distribution π:           π = π·P     (the long-run fraction of time per state)
```

Markov chains drive PageRank, MCMC sampling, hidden Markov models, and queueing models.
The discrete-time chain here is the probabilistic cousin of the continuous dynamics in
[differential equations](calculus.md).

## Where this shows up

- **ML** — maximum likelihood = maximizing P(data | params); cross-entropy loss is a
  log-likelihood; softmax outputs a distribution. See [metrics](../machine_learning/metrics.md).
- **Statistics** — every estimator, confidence interval, and p-value in
  [statistics](statistics.md) is a probability statement.
- **Information theory** — entropy `H(X) = −Σ p log p` is just an expectation; see
  [information theory](information_theory.md).
- **Systems** — tail-latency reasoning, Bloom filters, hashing, load balancing, and
  retry/backoff all rest on these distributions.

## Pitfalls

- **Base-rate neglect** — ignoring priors (the disease example above).
- **Gambler's fallacy** — independent trials have no memory; past coin flips don't change
  the next.
- **Confusing P(A|B) with P(B|A)** — the prosecutor's fallacy.
- **Assuming independence** to multiply probabilities when variables are correlated —
  the cause of underestimated tail risk in [finance](../finance/risk_management.md).
