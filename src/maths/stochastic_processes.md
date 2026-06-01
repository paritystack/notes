# Stochastic Processes

## Overview

A stochastic process is a **collection of random variables indexed by time** — a system
that evolves randomly, so instead of one number you study a whole *trajectory*. It is the
time-extended sibling of [probability](probability.md): where probability describes a
single random outcome, a stochastic process describes a random *path*. The Markov chains
introduced in [probability](probability.md) are the simplest case; this page extends them
to Poisson processes, random walks, martingales, and Brownian motion. These models drive
[option pricing and risk](../finance/risk_management.md) in finance,
[reinforcement learning](../machine_learning/reinforcement_learning.md), queueing and
tail-latency analysis in [operating systems](../misc/operating_systems.md), and MCMC
sampling.

```
A single random variable :   X            one draw
A stochastic process     :   {X_t}_{t≥0}  a whole timeline of draws

  X_t │      •          •
      │   •     •  •        •      one realization
      │ •          •    •     •    (sample path)
      └────────────────────────► t
```

## Classifying processes

```
                discrete time              continuous time
 discrete   │  Markov chain               Poisson / birth-death
 state      │  (steps n = 0,1,2…)          (counts of events)
 continuous │  time series / AR            Brownian motion / diffusions
 state      │                              (Wiener process)
```

Two structural properties recur:

- **Markov property** — the future depends only on the present state, not the full past.
- **Stationarity** — the statistics (mean, covariance) don't shift over time.

## Random walks

The canonical discrete process: start at 0, each step `+1` or `−1` with probability ½.

```
S_n = X₁ + X₂ + … + Xₙ ,   Xᵢ = ±1

E[Sₙ] = 0           (no drift)
Var(Sₙ) = n         spread grows ∝ n,  typical distance ∝ √n
```

The `√n` growth is the discrete shadow of the [CLT](probability.md) and of diffusion.
Random walks model gambler's ruin, diffusion, and the price-change intuition behind
Brownian motion. Adding a bias gives drift.

## Poisson process

The model for **events arriving randomly over continuous time** at rate `λ` — page
requests, radioactive decays, customer arrivals.

```
 • count N(t) of events in [0,t]  ~  Poisson(λt)        E[N(t)] = λt
 • inter-arrival times            ~  Exponential(λ)     memoryless
 • independent, non-overlapping intervals
```

The memoryless gap (waiting longer tells you nothing about the next arrival) is what
makes it both tractable and the default arrival model. Contrast carefully with the
**Poisson distribution**, which is a single random variable — the process is the
time-indexed family.

## Birth-death processes and queues

A continuous-time Markov chain on counts `0,1,2,…` with arrivals (birth rate λ) and
departures (death rate μ). The workhorse for **queueing**:

```
M/M/1 queue: Poisson arrivals (λ), exponential service (μ), one server
  utilization      ρ = λ/μ          (must be < 1 or the queue explodes)
  avg # in system  L = ρ/(1−ρ)
  avg wait         W = L/λ          (Little's law: L = λW)
```

As ρ → 1 the queue length blows up nonlinearly — the maths behind tail latency and why
servers degrade sharply near saturation; see [operating systems](../misc/operating_systems.md).

## Martingales

A process that is, on average, "fair" — the expected next value equals the current value:

```
E[X_{n+1} | X₀ … Xₙ] = Xₙ          a fair game; no expected drift
```

A symmetric random walk and a fair bet's bankroll are martingales. The **optional
stopping theorem** says you can't beat a fair game with a clever stopping rule — the
formal death of "double-your-bet" systems. Martingales also pin down arbitrage-free
pricing: under the risk-neutral measure, discounted asset prices are martingales.

## Brownian motion (Wiener process)

The continuous-time limit of a random walk as steps shrink — the foundation of continuous
stochastic modelling.

```
 • W(0) = 0
 • independent increments
 • W(t) − W(s) ~ Normal(0, t−s)        variance grows with time
 • continuous but nowhere differentiable paths
```

**Geometric Brownian motion** `dS = μS dt + σS dW` models asset prices (always positive,
proportional shocks) and is the engine inside the Black–Scholes
[option-pricing](../finance/derivatives.md) model. Manipulating these requires Itô
calculus, where `(dW)² = dt` — a twist on ordinary [calculus](calculus.md).

## Where this shows up

- **Finance** — [risk](../finance/risk_management.md) and
  [derivatives](../finance/derivatives.md) pricing rest on Brownian motion and
  martingales.
- **ML** — [reinforcement learning](../machine_learning/reinforcement_learning.md) is a
  Markov decision process; MCMC samplers (Metropolis, Gibbs) are engineered Markov chains;
  diffusion generative models reverse a noising stochastic process.
- **Systems** — queueing theory and Poisson arrivals model load, throughput, and tail
  latency; see [operating systems](../misc/operating_systems.md).
- **Foundations** — extends the Markov chains and distributions in
  [probability](probability.md).

## Pitfalls

- **Poisson process vs Poisson distribution** — the first is a time-indexed family, the
  second a single random variable; don't conflate them.
- **Assuming stationarity** — real arrival rates and volatilities drift; models calibrated
  on calm periods fail in regime changes.
- **Gambler's-ruin intuition** — a fair random walk hits any fixed loss with probability 1
  given enough time; "I'm due for a win" is the gambler's fallacy.
- **Treating Brownian paths as smooth** — they're nowhere differentiable; ordinary
  calculus chain rules don't apply (hence Itô's lemma).
