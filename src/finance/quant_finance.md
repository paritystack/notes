# Quantitative Finance

## Overview

Quantitative finance is the mathematical layer underneath the trading pages: the stochastic
models, PDEs, and numerical methods that turn "what's this contract worth?" into something
computable. The applied pages state results — [options](options.md) gives the Black-Scholes
*formula* and the Greeks, [derivatives](derivatives.md) asserts *risk-neutral pricing* and
shows a toy Monte Carlo, [volatility trading](volatility_trading.md) trades the vol surface,
and [interest rates](interest_rates.md) and [risk management](risk_management.md) lean on
term-structure and tail models. This page is where those results come from.

It is the finance-flavoured application of the pure-math pages: build on
[stochastic processes](../maths/stochastic_processes.md),
[probability](../maths/probability.md), and [numerical methods](../maths/numerical_methods.md).
The arc is always the same:

```
model the price (SDE)  →  Itô's lemma  →  no-arbitrage PDE  →  price
        │                                      │
        └──────────  risk-neutral expectation ─┘   ←── same answer, two views
                              │
                     numerical methods (trees / FD / Monte Carlo) when no closed form
```

## Modeling prices: random walks → Brownian motion → GBM

Start with a discrete random walk: each step the price ticks up or down. Shrink the step
size and the central limit theorem pushes the limit toward **Brownian motion** (a Wiener
process `W`), the continuous-time engine of almost every pricing model.

A Wiener process `W(t)` is defined by: `W(0)=0`; independent increments; `W(t)-W(s) ~
N(0, t-s)` (variance grows *linearly* in time, so std-dev grows like `√t`); and continuous
but nowhere-differentiable paths.

```
price
  │            .-.        Brownian path: jagged, continuous,
  │       .-. /   \  .-.  std-dev of displacement ∝ √t
  │   .-./   v     \/   \.
  │ ./                    \.-
  └────────────────────────────► time
       └─ "diffusion cone" widens like ±√t
```

We don't model the *price* as Brownian motion directly (it could go negative, and a $1 move
matters more on a $5 stock than a $500 one). We model **log-returns** as normal, which makes
the price **Geometric Brownian Motion (GBM)**:

```
dS = μ S dt + σ S dW
     └────┘   └────┘
      drift    diffusion (volatility)
```

`μ` is the expected return, `σ` the volatility. Because the percentage change is normal, `S`
stays positive and is **lognormally** distributed. GBM is the assumption behind Black-Scholes
(see the assumptions list in [options.md](options.md)).

## Itô's lemma

Ordinary calculus' chain rule breaks for stochastic processes because `(dW)² = dt` is *not*
negligible — Brownian motion accumulates quadratic variation. **Itô's lemma** is the
corrected chain rule. For `f(S, t)` with `S` following an SDE `dS = a dt + b dW`:

```
df = ( ∂f/∂t + a ∂f/∂S + ½ b² ∂²f/∂S² ) dt  +  b ∂f/∂S dW
                          └──────────┘
                   the extra "Itô term" — the whole point
```

Apply it to `f = ln S` under GBM (`a = μS`, `b = σS`):

```
d(ln S) = (μ − ½σ²) dt + σ dW
                └──┘
   volatility drag: log-growth is LOWER than μ
```

This single result explains *volatility drag* (why a volatile asset compounds slower than
its average return) and is the first step in deriving Black-Scholes. Itô's lemma is the one
tool everything below rests on.

## The Black-Scholes PDE

Build a **replicating portfolio**: hold one option and short `Δ = ∂V/∂S` shares. Choosing
`Δ` to cancel the random `dW` term makes the portfolio instantaneously *riskless*, so by
no-arbitrage it must earn the risk-free rate `r`. Substituting the GBM dynamics and Itô's
lemma yields the **Black-Scholes PDE**:

```
∂V/∂t + ½ σ² S² ∂²V/∂S² + r S ∂V/∂S − r V = 0
```

Notice `μ` has vanished — the expected return doesn't appear, only `r` and `σ`. Solving this
PDE with the payoff `max(S−K, 0)` as a terminal condition reproduces the closed-form
Black-Scholes *formula* already given in [options.md](options.md) (this page is the
derivation that page points to). The Greeks are literally the partial derivatives in this
equation: `Θ = ∂V/∂t`, `Δ = ∂V/∂S`, `Γ = ∂²V/∂S²`.

## Risk-neutral pricing

The replication argument has a dual, often-easier form. Under the **risk-neutral measure**
`ℚ` — a change of probability (Girsanov's theorem) that replaces the real drift `μ` with the
risk-free rate `r` — the discounted price of any tradable is a **martingale**. So any
derivative's value is just a discounted expected payoff:

```
V₀ = e^(−rT) · E^ℚ[ payoff(S_T) ]
```

**Feynman–Kac** is the bridge: it says the solution of the Black-Scholes PDE *equals* this
expectation. Same answer, two views — PDE (solve/grid it) or expectation (simulate it). This
is the rigorous "why" behind the risk-neutral pricing that [derivatives.md](derivatives.md)
states as a principle. The practical upshot: when a payoff is path-dependent or
high-dimensional and no PDE closed form exists, you price by simulating under `ℚ`.

## Numerical methods

Most real contracts have no closed form. Three workhorses:

- **Binomial / trinomial trees** — discretize `S` into an up/down lattice; price by backward
  induction. Naturally handles American early exercise (compare continuation vs. intrinsic at
  each node). Converges to Black-Scholes as steps → ∞.
- **Finite-difference** — grid the Black-Scholes PDE in `(S, t)` and step backward.
  *Explicit* schemes are simple but conditionally unstable; *implicit* / Crank–Nicolson are
  stable. Good for low-dimension PDEs with boundaries (barriers). See
  [numerical methods](../maths/numerical_methods.md).
- **Monte Carlo** — simulate many `ℚ`-paths, average the discounted payoff. The only viable
  route in high dimensions or for heavy path-dependence (Asians, autocallables). Error
  shrinks like `1/√N`, so use **variance reduction** (antithetic variates, control variates).
  This generalizes the toy loop in [derivatives.md](derivatives.md).

```python
import numpy as np

def mc_call_price(S0, K, r, sigma, T, n_paths=200_000, antithetic=True):
    """Risk-neutral Monte Carlo for a European call (GBM)."""
    z = np.random.standard_normal(n_paths)
    if antithetic:                       # pair z with -z to cut variance
        z = np.concatenate([z, -z])
    # exact GBM step to T: S_T = S0 * exp((r - ½σ²)T + σ√T z)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(S_T - K, 0.0)
    return np.exp(-r * T) * payoff.mean()

# For path-dependent payoffs, step through time with Euler–Maruyama instead:
# S[t+dt] = S[t] * (1 + r*dt + sigma*sqrt(dt)*randn())
```

## Beyond Black-Scholes

Constant-volatility GBM is wrong in ways the market prices in — hence the **volatility skew /
surface** (see [options.md](options.md) and [volatility trading](volatility_trading.md)).
Richer models:

- **Local volatility (Dupire)** — make `σ = σ(S, t)` a deterministic function fit to exactly
  reproduce today's surface. Great for consistency, poor at forward-smile dynamics.
- **Stochastic volatility (Heston, SABR)** — give volatility its *own* SDE (mean-reverting in
  Heston; SABR is the rates/FX-desk standard for smiles). Captures vol-of-vol and skew
  dynamics; needs calibration.
- **Jump-diffusion (Merton)** — add Poisson jumps to GBM for crash risk and fat tails; the
  extra parameters help fit short-dated skew that pure diffusion can't.

## Interest-rate models

Rates need their own dynamics because the whole *curve* moves and prices feed back into the
discount factor. **Short-rate models** specify the SDE of the instantaneous rate `r(t)`:

- **Vasicek** — mean-reverting, Gaussian; tractable but allows negative rates.
- **CIR (Cox–Ingersoll–Ross)** — mean-reverting with a `√r` diffusion that keeps rates ≥ 0.
- **Hull–White** — Vasicek extended with time-dependent parameters so it fits the *observed*
  curve exactly; the desk standard for vanilla rate derivatives.

These price bonds, swaptions, caps/floors and underpin the curve work in
[interest rates](interest_rates.md) and [bonds](bonds.md). (Forward-rate / LIBOR-market
models like HJM/BGM extend this to model the whole curve at once.)

## Calibration & model risk

A model is only as good as its fit to *market* quotes. **Calibration** inverts observed
prices (the vol surface, swaption grid) to recover parameters (`σ`, Heston's `κ, θ, ρ, ξ`,
Hull–White's mean-reversion). Two traps: an **overfit** model reprices today's quotes
perfectly but misprices anything off-grid, and an **unstable** calibration jumps parameters
day to day, making hedges noisy. Always sanity-check against simpler benchmarks — this is the
quantitative side of the "Model Risk" bullet in [derivatives.md](derivatives.md) and the tail
caveats in [risk management](risk_management.md).

## Where this connects

- [Options](options.md) — this page derives the Black-Scholes PDE/formula and shows the
  Greeks *are* the PDE's partial derivatives.
- [Derivatives](derivatives.md) — supplies the rigorous basis for risk-neutral pricing,
  replication, and the Monte Carlo it sketches.
- [Volatility trading](volatility_trading.md) — local/stochastic-vol models are the math
  behind the skew and surface that vol traders trade.
- [Interest rates](interest_rates.md) & [Bonds](bonds.md) — short-rate models price the curve
  and rate derivatives.
- [Risk management](risk_management.md) — fat-tailed and jump models drive realistic VaR and
  tail estimates.
- [Algorithmic trading](algorithmic_trading.md) — pricing and calibration engines sit under
  systematic strategies.
- [Stochastic processes](../maths/stochastic_processes.md),
  [probability](../maths/probability.md),
  [numerical methods](../maths/numerical_methods.md) — the pure-math foundations applied here.

## Pitfalls

- **Normality understates tails** — GBM's lognormal tails are far too thin; real returns have
  fat tails and crashes, so naive models underprice deep-OTM options and downside risk.
- **Constant vol is a fiction** — the market's skew exists precisely because `σ` isn't
  constant; pricing exotics off a single ATM vol is a classic error.
- **Discretization error** — too-coarse trees/grids or too-few Monte Carlo paths bias prices;
  explicit finite-difference schemes can blow up if the step ratio violates stability.
- **Calibration instability** — over-parameterized models fit today and break tomorrow;
  prefer the simplest model that captures the risk you're hedging.
- **Garbage in, garbage out** — an elegant model fed stale or illiquid quotes produces
  confident, wrong prices; model risk is mostly *input* risk.
- **Forgetting the measure** — discounting real-world (`ℙ`) expected payoffs instead of
  risk-neutral (`ℚ`) ones double-counts the risk premium and misprices everything.
