# Numerical Methods

## Overview

Numerical methods are the algorithms that compute *approximate* answers to continuous
[mathematics](calculus.md) on finite, discrete hardware. The gap between the clean math
and the machine — where real numbers become finite floats, and exact operations become
rounded ones — is where correctness silently breaks. This note covers how computers
represent numbers, how error grows, and the core algorithms for roots, linear systems,
integrals, and ODEs. It underpins [optimization](optimization.md),
[machine learning](../machine_learning/README.md) (every training step is numerical
linear algebra), [scientific computing](../misc/computer_graphics.md), and DSP on
[embedded](../embedded/README.md) hardware.

## Floating point (IEEE 754)

Real numbers are stored as `sign × mantissa × 2^exponent` — a finite set of values with
gaps that grow with magnitude.

```
float32 : 1 sign + 8 exponent + 23 mantissa  ≈ 7 decimal digits
float64 : 1 sign + 11 exponent + 52 mantissa ≈ 16 decimal digits

Machine epsilon ε = smallest x such that 1 + x ≠ 1
  float32 ε ≈ 1.2e−7      float64 ε ≈ 2.2e−16
```

Consequences that bite in practice:

```
0.1 + 0.2 ≠ 0.3            (0.1 has no exact binary representation)
NEVER test floats with ==  → use |a − b| ≤ tol
Special values: +0, −0, +∞, −∞, NaN     (NaN ≠ NaN; it poisons every operation)
```

```
Catastrophic cancellation: subtracting two nearly equal numbers destroys
significant digits.
  (1 + 1e−16) − 1  →  may yield 0 instead of 1e−16
  Fix: reformulate. e.g. the quadratic formula loses precision when b² ≫ 4ac;
       compute the stable root then use x₁x₂ = c/a for the other.
```

## Error, stability, conditioning

Two independent things can go wrong — keep them separate:

```
Conditioning  — a property of the PROBLEM: how much the output changes when the
                input is perturbed. An ill-conditioned problem amplifies any error.
Stability     — a property of the ALGORITHM: does it avoid introducing extra error?

  Well-conditioned problem + stable algorithm = trustworthy answer.
  Ill-conditioned problem  = no algorithm can save you (the problem is sensitive).
```

```
Absolute error = |x̂ − x|        Relative error = |x̂ − x| / |x|
Condition number κ(A) = ‖A‖·‖A⁻¹‖   (for linear systems)
  κ ≈ 1      well-conditioned
  κ huge     ill-conditioned → lose ~log₁₀κ digits of accuracy
```

A high condition number is exactly why unscaled features make
[gradient descent](optimization.md) crawl, and why we normalize inputs.

## Root finding — solve f(x) = 0

```
Bisection    bracket a root [a,b] with f(a)·f(b)<0, halve repeatedly.
             Linear convergence, but GUARANTEED. ~1 digit / 3.3 iterations.

Newton–Raphson   xₙ₊₁ = xₙ − f(xₙ)/f'(xₙ)
             Quadratic convergence (digits DOUBLE each step) — but needs the
             derivative and a good start, and can diverge.

Secant       Newton with a finite-difference derivative (no f' needed).

Practice: hybrid (Brent's method) = bisection's safety + secant's speed.
```

```
Newton picture: follow the tangent line down to the x-axis, repeat.

   f(x)\
        \___ tangent
            \      ← xₙ₊₁ where tangent hits 0
   ─────────●──────── x
```

## Numerical linear algebra

The computational core of ML and scientific computing — almost never invert a matrix
explicitly; *factor and solve*.

```
Solve Ax = b:
  LU decomposition (with partial pivoting)  — general square systems
  Cholesky (A = LLᵀ)                         — symmetric positive-definite, 2× faster
  QR decomposition                           — least squares, stable
  SVD (A = UΣVᵀ)                              — the swiss-army knife: rank, pseudoinverse,
                                               PCA, low-rank approximation

Iterative (huge / sparse A):
  Conjugate gradient (SPD), GMRES — used when A is too big to factor (FEM, PDEs).

Rule: solving Ax=b via factorization is more accurate AND faster than x = A⁻¹b.
      Never compute an explicit inverse to solve a system.
```

See [linear algebra](linear_algebra.md) for the decompositions themselves and their ML
uses (PCA, SVD-based dimensionality reduction).

## Numerical integration (quadrature)

Approximate `∫ₐᵇ f(x)dx` by sampling — for when there's no closed-form antiderivative.

```
Trapezoidal rule   sum of trapezoids        error ~ O(h²)
Simpson's rule     fit parabolas            error ~ O(h⁴)   (much better for smooth f)
Gaussian quadrature  optimal sample points  exact for polynomials up to degree 2n−1
Monte Carlo        average f at random points; error ~ O(1/√N)
                   — slow in 1-D but DIMENSION-INDEPENDENT → wins in high dimensions
                     (Bayesian inference, finance, rendering).
```

## Solving ODEs

Integrate `dy/dx = f(x, y)` forward in time — physics, control, simulation.

```
Euler          yₙ₊₁ = yₙ + h·f(xₙ, yₙ)            simplest, O(h) error, often unstable
RK4            4 weighted slope evaluations        O(h⁴) error — the standard workhorse
Adaptive (RKF45, Dormand–Prince)  vary step size h to hit an error tolerance

Stiff equations (fast + slow dynamics together) → explicit methods need tiny h;
   use IMPLICIT methods (backward Euler, BDF) for stability.
```

Step size `h` trades accuracy against cost and stability — too large diverges, too small
wastes work and accumulates rounding error.

## Practical guidance

```
- Use float64 by default; drop to float32/bfloat16 only when speed/memory forces it
  (ML inference, GPUs) and you've checked the accuracy cost.
- Prefer library routines (LAPACK, BLAS, NumPy/SciPy) — they encode decades of
  stability work. Don't hand-roll a matrix solver.
- Scale/normalize inputs to improve conditioning.
- Reformulate to avoid subtracting nearly-equal quantities.
- Check residuals (‖Ax̂ − b‖), not just that the code ran.
```

## Where this shows up

- **ML** — training is iterative numerical optimization over float matrices;
  mixed-precision and numerical stability (log-sum-exp, gradient clipping) are everyday
  concerns. See [optimization](optimization.md), [quantization](../machine_learning/quantization.md).
- **Graphics & simulation** — ODE/PDE solvers, linear systems.
- **Embedded/DSP** — fixed-point vs floating-point trade-offs on
  [microcontrollers](../embedded/README.md).
- **Finance** — Monte Carlo pricing, numerical PDE solvers for derivatives.

## Pitfalls

- **`==` on floats** — almost always a bug; compare with a tolerance.
- **Catastrophic cancellation** — reformulate before it eats your precision.
- **Inverting matrices** — factor and solve instead.
- **NaN propagation** — one NaN silently corrupts an entire computation; guard inputs
  (e.g. `log(0)`, `0/0`, `sqrt(−x)`).
- **Ignoring conditioning** — a "correct" algorithm still returns garbage on an
  ill-conditioned problem.
