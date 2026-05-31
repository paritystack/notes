# Maths

Mathematical foundations for computer science, machine learning, and engineering. Each
note favours intuition first, then the machinery, then where it shows up in practice.

## Foundations

- **[Discrete Mathematics](discrete_math.md)** — sets, logic, combinatorics, number
  theory, recurrences, and graph theory. The language under [algorithms](../algorithms/README.md),
  [data structures](../data_structures/README.md), and [cryptography](../security/encryption.md).
- **[Probability](probability.md)** — random variables, distributions, expectation,
  Bayes' theorem, and Markov chains. The basis of [statistics](statistics.md) and all of
  [machine learning](../machine_learning/README.md).
- **[Statistics](statistics.md)** — descriptive statistics, distributions, inference,
  hypothesis testing, correlation, and regression.

## Analysis

- **[Calculus](calculus.md)** — limits, derivatives, integrals, sequences/series,
  multivariable calculus, and differential equations.
- **[Linear Algebra](linear_algebra.md)** — vectors, matrices, decompositions
  (eigen/SVD), and their use in ML, graphics, and scientific computing.

## Applied

- **[Optimization](optimization.md)** — convex optimization, gradient descent, Lagrange
  multipliers, and linear/quadratic programming — the math under model training.
- **[Information Theory](information_theory.md)** — entropy, KL divergence, mutual
  information, and source/channel coding. Underlies ML loss functions and compression.
- **[Numerical Methods](numerical_methods.md)** — floating point, conditioning &
  stability, root finding, numerical linear algebra, and ODE solvers.

## Reading guide

```
Discrete Math ─┐
               ├─► Algorithms / Data Structures / Crypto
Probability ───┼─► Statistics ─► ML
               │
Calculus ──────┼─► Optimization ─► ML training
Linear Algebra ┘        │
                        └─► Numerical Methods (how it runs on real hardware)
Probability ─► Information Theory ─► ML losses / compression
```
