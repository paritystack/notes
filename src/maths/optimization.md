# Optimization

## Overview

Optimization is the search for the input that makes an objective as small (or large) as
possible, possibly subject to constraints. It is the mathematics under model training —
every neural net is a minimization of a loss, every SVM a constrained quadratic program —
and it leans directly on [calculus](calculus.md) (gradients), [linear algebra](linear_algebra.md)
(the Hessian, quadratic forms), and shows up across [machine learning](../machine_learning/deep_learning.md),
[operations research](../finance/portfolio_management.md), and control.

```
minimize    f(x)          ← objective / loss / cost
subject to  gᵢ(x) ≤ 0     ← inequality constraints
            hⱼ(x) = 0     ← equality constraints
x ∈ ℝⁿ                     ← decision variables
```

Maximizing `f` is just minimizing `−f`, so theory is written for minimization.

## Convex vs non-convex

The single most important distinction. **Convexity** is what separates "solved" from
"hard".

```
Convex set:       the line between any two points stays inside the set.
Convex function:  the chord between any two points lies above the graph.
                  f(λx + (1−λ)y) ≤ λf(x) + (1−λ)f(y)

  convex  ∪ ───  one bowl, every local min is THE global min
  non-convex ╱╲╱  many valleys, local minima ≠ global minimum
```

```
Convex problem  → any local minimum is global; reliable, fast solvers.
Non-convex      → gradient descent finds *a* local min; no global guarantee
                  (this is deep learning — it works anyway in practice).
```

Convexity tests: a twice-differentiable `f` is convex iff its Hessian `∇²f ⪰ 0`
(positive semidefinite) everywhere. Sums of convex functions, max of convex functions,
and affine compositions stay convex.

## Optimality conditions

```
Unconstrained, smooth f:
  Necessary:   ∇f(x*) = 0          (stationary point)
  Sufficient:  ∇f(x*) = 0  AND  ∇²f(x*) ≻ 0   (positive definite → local min)

  ∇²f indefinite → saddle point (a min in some directions, max in others)
```

Saddle points — not bad local minima — are the dominant obstacle in high-dimensional
deep learning, because in many dimensions it's unlikely *every* direction curves up.

## Gradient descent

The workhorse. Step downhill, proportional to the negative gradient:

```
xₜ₊₁ = xₜ − η · ∇f(xₜ)        η = learning rate (step size)
```

```
η too small → crawls, slow convergence
η too large → overshoots, oscillates or diverges
just right  → steady descent
```

Variants you'll meet in [ML training](../machine_learning/deep_learning.md):

```
Batch GD        gradient over the whole dataset    — exact, expensive
Stochastic GD   gradient from one sample           — noisy, cheap, escapes shallow mins
Mini-batch GD   gradient over a small batch         — the practical default
Momentum        accumulate a velocity to roll through small bumps & ravines
Adam / RMSProp  per-parameter adaptive step sizes   — the deep-learning standard
```

**Newton's method** uses curvature for quadratic convergence near the optimum, but needs
the Hessian inverse (`O(n³)`), so large-scale ML uses first-order methods and
quasi-Newton approximations (L-BFGS).

```
Newton step:  xₜ₊₁ = xₜ − [∇²f(xₜ)]⁻¹ ∇f(xₜ)
```

Conditioning matters: gradient descent crawls along long thin valleys (high
[condition number](numerical_methods.md)); this is why we normalize features and use
adaptive optimizers.

## Constrained optimization

### Lagrange multipliers (equality constraints)

To minimize `f` subject to `h(x) = 0`, the gradient of `f` must be parallel to the
gradient of the constraint at the optimum — you can't improve without leaving the
constraint surface.

```
Lagrangian:  L(x, λ) = f(x) + λ·h(x)
Solve:       ∇ₓL = 0  and  h(x) = 0

  λ (the multiplier) = sensitivity of the optimum to relaxing the constraint
                        — the "shadow price" in economics.
```

### KKT conditions (inequality constraints)

The Karush–Kuhn–Tucker conditions generalize Lagrange to `gᵢ(x) ≤ 0`. For convex
problems they are necessary *and* sufficient for global optimality.

```
Stationarity        ∇f + Σ μᵢ∇gᵢ + Σ λⱼ∇hⱼ = 0
Primal feasibility  gᵢ(x) ≤ 0,  hⱼ(x) = 0
Dual feasibility    μᵢ ≥ 0
Complementary       μᵢ · gᵢ(x) = 0   ← either constraint is tight (gᵢ=0) or μᵢ=0
```

Complementary slackness is the key idea: a constraint either binds (active, `gᵢ=0`) or
is irrelevant (`μᵢ=0`). This is exactly the "support vectors" in an SVM — only the
binding constraints matter.

## Duality

Every minimization (the **primal**) has a paired **dual** maximization. The dual lower-
bounds the primal (**weak duality**); for convex problems they meet (**strong duality**),
and the dual is often easier to solve or gives a certificate of optimality.

```
Duality gap = primal* − dual* ≥ 0    ;  = 0 for convex problems (typically)
```

## Problem classes

```
LP   Linear Program        linear objective + linear constraints
                           → simplex, interior-point. Scheduling, flows, blending.
QP   Quadratic Program     quadratic objective + linear constraints
                           → SVMs, portfolio optimization (mean–variance).
SOCP/SDP                   second-order-cone / semidefinite — robust & control problems.
Convex (general)           → CVXPY, interior-point methods.
Integer / MILP             variables must be integers → NP-hard; branch & bound.
Non-convex / non-smooth    → gradient methods (DL), evolutionary, simulated annealing.
```

## Beyond gradients (derivative-free / global)

When the objective is non-differentiable, noisy, or a black box:

```
Grid / random search     hyperparameter tuning baselines
Bayesian optimization     model f with a surrogate (Gaussian process), sample smartly
Evolutionary / GA         population + mutation + selection
Simulated annealing       accept worse moves early to escape local minima
```

These connect to [reinforcement learning](../machine_learning/reinforcement_learning.md)
and [heuristic search](../algorithms/heuristic_search.md).

## Where this shows up

- **ML training** — loss minimization by SGD/Adam; regularization (L1/L2) is a
  constraint on weights; see [deep learning](../machine_learning/deep_learning.md).
- **Finance** — Markowitz mean–variance is a QP; see [portfolio management](../finance/portfolio_management.md).
- **Systems** — scheduling, bin-packing, routing, and resource allocation are
  LP/MILP problems.
- **Control & robotics** — model-predictive control solves a QP every timestep.

## Pitfalls

- **Bad learning rate** — the number-one cause of training divergence or stalling.
- **Treating a non-convex result as global** — restart from multiple inits.
- **Ignoring conditioning** — unscaled features make first-order methods crawl.
- **Local minima vs saddle points** — in high dimensions, saddles dominate; momentum and
  noise help escape them.
