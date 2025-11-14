# Fundamental Mathematical Concepts

A comprehensive guide to calculus and essential mathematical concepts.

## Table of Contents
1. [Limits and Continuity](#limits-and-continuity)
2. [Derivatives](#derivatives)
3. [Differentiation Techniques](#differentiation-techniques)
4. [Applications of Derivatives](#applications-of-derivatives)
5. [Integration](#integration)
6. [Integration Techniques](#integration-techniques)
7. [Applications of Integration](#applications-of-integration)
8. [Sequences and Series](#sequences-and-series)
9. [Multivariable Calculus](#multivariable-calculus)
10. [Differential Equations](#differential-equations)

---

## Limits and Continuity

### Definition of a Limit

The limit of a function f(x) as x approaches a is L, written as:

```
lim(xía) f(x) = L
```

**Formal (µ-¥) Definition**: For every µ > 0, there exists a ¥ > 0 such that if 0 < |x - a| < ¥, then |f(x) - L| < µ.

**Intuitive Definition**: As x gets arbitrarily close to a (but not equal to a), f(x) gets arbitrarily close to L.

### Limit Laws

If lim(xía) f(x) = L and lim(xía) g(x) = M, then:

1. **Sum Rule**: lim(xía) [f(x) + g(x)] = L + M
2. **Difference Rule**: lim(xía) [f(x) - g(x)] = L - M
3. **Product Rule**: lim(xía) [f(x) ∑ g(x)] = L ∑ M
4. **Quotient Rule**: lim(xía) [f(x) / g(x)] = L / M (if M ` 0)
5. **Constant Multiple**: lim(xía) [c ∑ f(x)] = c ∑ L
6. **Power Rule**: lim(xía) [f(x)]^n = L^n

### Types of Limits

**One-Sided Limits**:
- Right-hand limit: lim(xíaz) f(x)
- Left-hand limit: lim(xía{) f(x)
- A limit exists if and only if both one-sided limits exist and are equal

**Infinite Limits**:
- lim(xía) f(x) =  (function grows without bound)
- lim(xía) f(x) = - (function decreases without bound)

**Limits at Infinity**:
- lim(xí) f(x) = L
- lim(xí-) f(x) = L

### Continuity

A function f is **continuous at x = a** if:
1. f(a) is defined
2. lim(xía) f(x) exists
3. lim(xía) f(x) = f(a)

**Types of Discontinuity**:
- **Removable**: Limit exists but f(a) is undefined or different
- **Jump**: Left and right limits exist but are unequal
- **Infinite**: Function approaches ±

**Important Theorems**:
- **Intermediate Value Theorem (IVT)**: If f is continuous on [a,b] and k is between f(a) and f(b), then there exists c in (a,b) such that f(c) = k
- **Extreme Value Theorem (EVT)**: A continuous function on a closed interval [a,b] attains both a maximum and minimum value

---

## Derivatives

### Definition

The **derivative** of f(x) at x = a is:

```
f'(a) = lim(hí0) [f(a+h) - f(a)] / h
```

Alternative form:
```
f'(a) = lim(xía) [f(x) - f(a)] / (x - a)
```

### Interpretations

**Geometric**: The derivative represents the slope of the tangent line to the curve at a point.

**Physical**: The derivative represents the instantaneous rate of change.
- If s(t) is position, then s'(t) is velocity
- If v(t) is velocity, then v'(t) is acceleration

### Notation

Multiple notations for derivatives:
- **Lagrange**: f'(x), f''(x), f'''(x), f}~(x)
- **Leibniz**: dy/dx, d≤y/dx≤, dy/dx
- **Newton**: è, ˇ (for time derivatives)
- **Euler**: D_x f, D≤_x f

### Basic Derivative Rules

1. **Constant Rule**: d/dx[c] = 0
2. **Power Rule**: d/dx[x^n] = n∑x^(n-1)
3. **Constant Multiple**: d/dx[c∑f(x)] = c∑f'(x)
4. **Sum Rule**: d/dx[f(x) + g(x)] = f'(x) + g'(x)
5. **Difference Rule**: d/dx[f(x) - g(x)] = f'(x) - g'(x)

### Higher-Order Derivatives

- **First derivative**: f'(x) or dy/dx - rate of change
- **Second derivative**: f''(x) or d≤y/dx≤ - rate of change of rate of change (concavity)
- **Third derivative**: f'''(x) or d≥y/dx≥ - jerk (in physics)
- **nth derivative**: f}~(x) or dy/dx

**Concavity**:
- f''(x) > 0 í concave up (curve opens upward)
- f''(x) < 0 í concave down (curve opens downward)
- f''(x) = 0 í possible inflection point

---

## Differentiation Techniques

### Product Rule

If u and v are differentiable functions:
```
d/dx[u∑v] = u'∑v + u∑v'
```

**Example**: d/dx[x≤∑sin(x)] = 2x∑sin(x) + x≤∑cos(x)

### Quotient Rule

```
d/dx[u/v] = (u'∑v - u∑v') / v≤
```

**Example**: d/dx[sin(x)/x] = [x∑cos(x) - sin(x)] / x≤

### Chain Rule

For composite functions f(g(x)):
```
d/dx[f(g(x))] = f'(g(x))∑g'(x)
```

Or in Leibniz notation:
```
dy/dx = (dy/du)∑(du/dx)
```

**Example**: d/dx[sin(x≤)] = cos(x≤)∑2x = 2x∑cos(x≤)

### Implicit Differentiation

When a relation is given implicitly (not solved for y):

**Steps**:
1. Differentiate both sides with respect to x
2. Apply chain rule to terms with y (multiply by dy/dx)
3. Solve for dy/dx

**Example**: x≤ + y≤ = 25
```
2x + 2y∑(dy/dx) = 0
dy/dx = -x/y
```

### Logarithmic Differentiation

Useful for products, quotients, and powers of functions:

**Steps**:
1. Take ln of both sides
2. Use logarithm properties to simplify
3. Differentiate implicitly
4. Solve for dy/dx

**Example**: y = x^x
```
ln(y) = x∑ln(x)
(1/y)∑(dy/dx) = ln(x) + 1
dy/dx = y∑(ln(x) + 1) = x^x∑(ln(x) + 1)
```

### Parametric Differentiation

For curves defined parametrically: x = f(t), y = g(t)
```
dy/dx = (dy/dt) / (dx/dt)
```

**Second derivative**:
```
d≤y/dx≤ = d/dx[dy/dx] = [d/dt(dy/dx)] / (dx/dt)
```

### Common Derivatives

**Trigonometric Functions**:
- d/dx[sin(x)] = cos(x)
- d/dx[cos(x)] = -sin(x)
- d/dx[tan(x)] = sec≤(x)
- d/dx[cot(x)] = -csc≤(x)
- d/dx[sec(x)] = sec(x)∑tan(x)
- d/dx[csc(x)] = -csc(x)∑cot(x)

**Inverse Trigonometric Functions**:
- d/dx[arcsin(x)] = 1/(1-x≤)
- d/dx[arccos(x)] = -1/(1-x≤)
- d/dx[arctan(x)] = 1/(1+x≤)

**Exponential and Logarithmic Functions**:
- d/dx[e^x] = e^x
- d/dx[a^x] = a^x∑ln(a)
- d/dx[ln(x)] = 1/x
- d/dx[log_a(x)] = 1/(x∑ln(a))

**Hyperbolic Functions**:
- d/dx[sinh(x)] = cosh(x)
- d/dx[cosh(x)] = sinh(x)
- d/dx[tanh(x)] = sech≤(x)

---

## Applications of Derivatives

### Critical Points and Extrema

**Critical Point**: x = c where f'(c) = 0 or f'(c) does not exist

**Finding Extrema**:
1. Find all critical points
2. Use First Derivative Test or Second Derivative Test
3. Check endpoints (for closed intervals)

**First Derivative Test**:
- If f' changes from + to - at c, then f has a local maximum at c
- If f' changes from - to + at c, then f has a local minimum at c

**Second Derivative Test**:
- If f'(c) = 0 and f''(c) > 0, then f has a local minimum at c
- If f'(c) = 0 and f''(c) < 0, then f has a local maximum at c
- If f''(c) = 0, test is inconclusive

### Optimization Problems

**General Strategy**:
1. Identify the quantity to optimize (write as a function)
2. Identify constraints
3. Use constraints to express the quantity as a function of one variable
4. Find critical points
5. Determine which critical point gives the optimal value

### Related Rates

For quantities that change with respect to time:

**Strategy**:
1. Draw a diagram and label variables
2. Write an equation relating the variables
3. Differentiate both sides with respect to time t
4. Substitute known values
5. Solve for the desired rate

**Example**: A ladder sliding down a wall
```
x≤ + y≤ = L≤
2x∑(dx/dt) + 2y∑(dy/dt) = 0
```

### Mean Value Theorem (MVT)

If f is continuous on [a,b] and differentiable on (a,b), then there exists c in (a,b) such that:
```
f'(c) = [f(b) - f(a)] / (b - a)
```

**Interpretation**: There exists a point where the instantaneous rate equals the average rate.

### Linear Approximation

The tangent line approximation at x = a:
```
L(x) = f(a) + f'(a)∑(x - a)
```

For small îx:
```
f(a + îx) H f(a) + f'(a)∑îx
```

**Differentials**:
- dx = îx (change in x)
- dy = f'(x)∑dx (change in tangent line)
- îy = f(x + dx) - f(x) (actual change in f)

### L'HÙpital's Rule

For indeterminate forms 0/0 or /:
```
lim(xía) [f(x)/g(x)] = lim(xía) [f'(x)/g'(x)]
```

Can be applied repeatedly if result is still indeterminate.

**Other indeterminate forms** (0∑, -, 0p, 1^, p) can be converted to 0/0 or / form.

### Curve Sketching

**Complete Analysis**:
1. Domain and range
2. Intercepts (x and y)
3. Symmetry (even, odd, periodic)
4. Asymptotes (vertical, horizontal, oblique)
5. First derivative (increasing/decreasing, local extrema)
6. Second derivative (concavity, inflection points)
7. Plot key points and sketch

---

## Integration

### Antiderivatives

An **antiderivative** of f(x) is a function F(x) such that F'(x) = f(x).

**General Antiderivative**: F(x) + C, where C is an arbitrary constant.

### Indefinite Integrals

The **indefinite integral** represents the family of all antiderivatives:
```
+ f(x) dx = F(x) + C
```

### Definite Integrals

The **definite integral** from a to b:
```
+[a to b] f(x) dx
```

**Geometric Interpretation**: The signed area between the curve and the x-axis from a to b.

**Properties**:
1. +[a to b] c∑f(x) dx = c∑+[a to b] f(x) dx
2. +[a to b] [f(x) ± g(x)] dx = +[a to b] f(x) dx ± +[a to b] g(x) dx
3. +[a to b] f(x) dx = -+[b to a] f(x) dx
4. +[a to a] f(x) dx = 0
5. +[a to b] f(x) dx + +[b to c] f(x) dx = +[a to c] f(x) dx

### Fundamental Theorem of Calculus

**Part 1**: If f is continuous on [a,b] and F(x) = +[a to x] f(t) dt, then F'(x) = f(x).

This establishes that integration and differentiation are inverse operations.

**Part 2**: If f is continuous on [a,b] and F is any antiderivative of f, then:
```
+[a to b] f(x) dx = F(b) - F(a)
```

This provides a practical method for evaluating definite integrals.

### Basic Integration Formulas

1. + k dx = kx + C
2. + x^n dx = x^(n+1)/(n+1) + C (n ` -1)
3. + (1/x) dx = ln|x| + C
4. + e^x dx = e^x + C
5. + a^x dx = a^x/ln(a) + C
6. + sin(x) dx = -cos(x) + C
7. + cos(x) dx = sin(x) + C
8. + sec≤(x) dx = tan(x) + C
9. + csc≤(x) dx = -cot(x) + C
10. + sec(x)tan(x) dx = sec(x) + C
11. + csc(x)cot(x) dx = -csc(x) + C
12. + 1/(1-x≤) dx = arcsin(x) + C
13. + 1/(1+x≤) dx = arctan(x) + C

### Riemann Sums

The definite integral is the limit of Riemann sums:
```
+[a to b] f(x) dx = lim(ní) £[i=1 to n] f(x_i*)∑îx
```

where îx = (b-a)/n and x_i* is a sample point in the ith subinterval.

**Types**:
- **Left Riemann Sum**: Use left endpoints
- **Right Riemann Sum**: Use right endpoints
- **Midpoint Rule**: Use midpoints
- **Trapezoidal Rule**: Average of left and right
- **Simpson's Rule**: Uses parabolic approximation

---

## Integration Techniques

### Substitution (u-Substitution)

**Method**: Let u = g(x), then du = g'(x)dx

**Steps**:
1. Choose substitution u = g(x)
2. Calculate du = g'(x)dx
3. Rewrite integral in terms of u
4. Integrate with respect to u
5. Substitute back to get result in terms of x

**Example**:
```
+ 2x∑cos(x≤) dx
Let u = x≤, du = 2x dx
= + cos(u) du
= sin(u) + C
= sin(x≤) + C
```

**For definite integrals**, also change the limits:
- If u = g(x), new limits are u = g(a) and u = g(b)

### Integration by Parts

**Formula**:
```
+ u dv = uv - + v du
```

**Choosing u and dv (LIATE rule)**:
- **L**ogarithmic
- **I**nverse trigonometric
- **A**lgebraic
- **T**rigonometric
- **E**xponential

Choose u in this order of preference; dv is what remains.

**Example**:
```
+ x∑e^x dx
u = x, dv = e^x dx
du = dx, v = e^x
= x∑e^x - + e^x dx
= x∑e^x - e^x + C
= e^x(x - 1) + C
```

**Tabular Integration**: Efficient for repeated integration by parts.

### Trigonometric Integrals

**Strategies for + sin^m(x)cos^n(x) dx**:

1. **If n is odd**: Save one cos(x), convert rest to sin(x) using cos≤(x) = 1 - sin≤(x), then substitute u = sin(x)
2. **If m is odd**: Save one sin(x), convert rest to cos(x) using sin≤(x) = 1 - cos≤(x), then substitute u = cos(x)
3. **If both are even**: Use power-reducing formulas
   - sin≤(x) = (1 - cos(2x))/2
   - cos≤(x) = (1 + cos(2x))/2

**Powers of tan and sec**:
- + tan^m(x)sec^n(x) dx
- Use tan≤(x) = sec≤(x) - 1 and sec≤(x) derivative of tan(x)

### Trigonometric Substitution

For integrals involving (a≤ - x≤), (a≤ + x≤), or (x≤ - a≤):

1. **(a≤ - x≤)**: Let x = a∑sin(∏), dx = a∑cos(∏)d∏
   - (a≤ - x≤) = a∑cos(∏)

2. **(a≤ + x≤)**: Let x = a∑tan(∏), dx = a∑sec≤(∏)d∏
   - (a≤ + x≤) = a∑sec(∏)

3. **(x≤ - a≤)**: Let x = a∑sec(∏), dx = a∑sec(∏)tan(∏)d∏
   - (x≤ - a≤) = a∑tan(∏)

**Example**:
```
+ (1 - x≤) dx
Let x = sin(∏), dx = cos(∏)d∏
= + cos(∏)∑cos(∏) d∏
= + cos≤(∏) d∏
= + (1 + cos(2∏))/2 d∏
= ∏/2 + sin(2∏)/4 + C
= arcsin(x)/2 + x(1-x≤)/2 + C
```

### Partial Fractions

For rational functions P(x)/Q(x) where degree(P) < degree(Q):

**Steps**:
1. Factor the denominator Q(x)
2. Decompose into partial fractions
3. Solve for coefficients (equate coefficients or plug in values)
4. Integrate each term

**Forms**:
1. **Linear factors**: (x - a) í A/(x - a)
2. **Repeated linear**: (x - a)^n í AÅ/(x-a) + AÇ/(x-a)≤ + ... + Aô/(x-a)^n
3. **Quadratic factors**: (x≤ + bx + c) í (Ax + B)/(x≤ + bx + c)
4. **Repeated quadratic**: Similar to repeated linear

**Example**:
```
+ 1/(x≤ - 1) dx = + 1/[(x-1)(x+1)] dx
1/(x≤ - 1) = A/(x-1) + B/(x+1)
1 = A(x+1) + B(x-1)
Solving: A = 1/2, B = -1/2
= (1/2)+ 1/(x-1) dx - (1/2)+ 1/(x+1) dx
= (1/2)ln|x-1| - (1/2)ln|x+1| + C
= (1/2)ln|(x-1)/(x+1)| + C
```

### Improper Integrals

**Type 1**: Infinite interval
```
+[a to ] f(x) dx = lim(tí) +[a to t] f(x) dx
```

**Type 2**: Discontinuous integrand
```
+[a to b] f(x) dx = lim(tíb{) +[a to t] f(x) dx  (if f is discontinuous at b)
```

**Convergence**: The improper integral converges if the limit exists and is finite; otherwise it diverges.

**Comparison Test**: If 0 d f(x) d g(x) for x e a:
- If + g(x) dx converges, then + f(x) dx converges
- If + f(x) dx diverges, then + g(x) dx diverges

---

## Applications of Integration

### Area Between Curves

**Vertical slicing** (integrate with respect to x):
```
A = +[a to b] [f(x) - g(x)] dx
```
where f(x) e g(x) on [a,b]

**Horizontal slicing** (integrate with respect to y):
```
A = +[c to d] [f(y) - g(y)] dy
```

### Volume

**Disk Method** (revolving around x-axis):
```
V = ¿∑+[a to b] [f(x)]≤ dx
```

**Washer Method** (hollow solid):
```
V = ¿∑+[a to b] [R(x)]≤ - [r(x)]≤ dx
```
where R(x) is outer radius, r(x) is inner radius

**Shell Method** (cylindrical shells):
```
V = 2¿∑+[a to b] x∑f(x) dx
```
or
```
V = 2¿∑+[c to d] y∑g(y) dy
```

**Cross-Sectional Method**:
```
V = +[a to b] A(x) dx
```
where A(x) is the area of cross-section at x

### Arc Length

**For y = f(x)** on [a,b]:
```
L = +[a to b] (1 + [f'(x)]≤) dx
```

**For parametric curves** x = f(t), y = g(t) on [±,≤]:
```
L = +[± to ≤] ([dx/dt]≤ + [dy/dt]≤) dt
```

**For polar curves** r = f(∏):
```
L = +[± to ≤] (r≤ + [dr/d∏]≤) d∏
```

### Surface Area

**Revolution around x-axis**:
```
S = 2¿∑+[a to b] f(x)∑(1 + [f'(x)]≤) dx
```

**Revolution around y-axis**:
```
S = 2¿∑+[a to b] x∑(1 + [f'(x)]≤) dx
```

### Work

**Constant force**: W = F∑d

**Variable force**:
```
W = +[a to b] F(x) dx
```

**Examples**:
- **Spring**: W = + kx dx = (1/2)kx≤ (Hooke's Law)
- **Lifting liquid**: W = + ¡∑g∑A(y)∑y dy
- **Pumping**: Account for distance each layer must be moved

### Center of Mass

**For a thin plate** (lamina) with density ¡(x,y):

**Mass**:
```
m = ++_R ¡(x,y) dA
```

**Moments**:
```
M_x = ++_R y∑¡(x,y) dA
M_y = ++_R x∑¡(x,y) dA
```

**Center of mass**:
```
x = M_y / m
3 = M_x / m
```

**For uniform density** (¡ = constant), center of mass = centroid.

---

## Sequences and Series

### Sequences

A **sequence** is an ordered list: {aÅ, aÇ, aÉ, ...} or {aô}

**Convergence**: lim(ní) aô = L means the sequence converges to L.

**Properties**:
- **Monotonic**: Always increasing or always decreasing
- **Bounded**: |aô| d M for all n
- **Monotone Convergence Theorem**: A bounded, monotonic sequence converges

### Series

An **infinite series** is the sum of a sequence:
```
£[n=1 to ] aô = aÅ + aÇ + aÉ + ...
```

**Partial sums**: Sô = £[k=1 to n] añ

**Convergence**: The series converges to S if lim(ní) Sô = S.

### Geometric Series

```
£[n=0 to ] ar^n = a + ar + ar≤ + ar≥ + ...
```

**Convergence**:
- If |r| < 1, series converges to a/(1-r)
- If |r| e 1, series diverges

### Tests for Convergence

**nth-Term Test (Divergence Test)**:
- If lim(ní) aô ` 0, then £aô diverges
- If lim(ní) aô = 0, test is inconclusive

**Integral Test**:
If f is continuous, positive, decreasing for x e 1:
- £[n=1 to ] aô and +[1 to ] f(x) dx both converge or both diverge

**p-Series**:
```
£[n=1 to ] 1/n^p
```
Converges if p > 1, diverges if p d 1

**Comparison Test**:
If 0 d aô d bô for all n:
- If £bô converges, then £aô converges
- If £aô diverges, then £bô diverges

**Limit Comparison Test**:
If aô, bô > 0 and lim(ní) aô/bô = c > 0:
- Both series converge or both diverge

**Ratio Test**:
```
L = lim(ní) |aôäÅ / aô|
```
- If L < 1, series converges absolutely
- If L > 1 (or L = ), series diverges
- If L = 1, test is inconclusive

**Root Test**:
```
L = lim(ní) |aô|
```
- If L < 1, series converges absolutely
- If L > 1 (or L = ), series diverges
- If L = 1, test is inconclusive

**Alternating Series Test**:
For alternating series £(-1)^n∑bô where bô > 0:
- If bô is decreasing and lim(ní) bô = 0, series converges

### Absolute and Conditional Convergence

- **Absolutely convergent**: £|aô| converges
- **Conditionally convergent**: £aô converges but £|aô| diverges

If a series converges absolutely, it converges.

### Power Series

A **power series** centered at a:
```
£[n=0 to ] cô(x - a)^n
```

**Radius of Convergence (R)**:
- Series converges for |x - a| < R
- Series diverges for |x - a| > R
- At endpoints x = a ± R, must test separately

**Finding R**:
```
R = lim(ní) |cô / côäÅ|
```
or
```
1/R = lim(ní) |côäÅ / cô|
```

**Interval of Convergence**: (a - R, a + R) plus possibly the endpoints

### Taylor and Maclaurin Series

**Taylor Series** of f(x) centered at x = a:
```
f(x) = £[n=0 to ] [f}~(a) / n!]∑(x - a)^n
     = f(a) + f'(a)(x-a) + [f''(a)/2!](x-a)≤ + [f'''(a)/3!](x-a)≥ + ...
```

**Maclaurin Series** (special case where a = 0):
```
f(x) = £[n=0 to ] [f}~(0) / n!]∑x^n
```

**Common Maclaurin Series**:

1. e^x = £[n=0 to ] x^n/n! = 1 + x + x≤/2! + x≥/3! + ...

2. sin(x) = £[n=0 to ] (-1)^n∑x^(2n+1)/(2n+1)! = x - x≥/3! + xu/5! - ...

3. cos(x) = £[n=0 to ] (-1)^n∑x^(2n)/(2n)! = 1 - x≤/2! + xt/4! - ...

4. 1/(1-x) = £[n=0 to ] x^n = 1 + x + x≤ + x≥ + ... (|x| < 1)

5. ln(1+x) = £[n=1 to ] (-1)^(n+1)∑x^n/n = x - x≤/2 + x≥/3 - ... (|x| < 1)

6. arctan(x) = £[n=0 to ] (-1)^n∑x^(2n+1)/(2n+1) = x - x≥/3 + xu/5 - ... (|x| d 1)

**Taylor's Remainder**:
```
Rô(x) = f(x) - Tô(x) = [f}zπ~(c) / (n+1)!]∑(x - a)^(n+1)
```
where c is between a and x.

---

## Multivariable Calculus

### Partial Derivatives

For a function f(x,y):

**Partial derivative with respect to x**:
```
f/x = lim(hí0) [f(x+h, y) - f(x, y)] / h
```

**Notation**:
- f/x, f_x, _x f

**Computing**: Treat other variables as constants and differentiate normally.

**Example**: f(x,y) = x≤y + y≥
- f/x = 2xy
- f/y = x≤ + 3y≤

**Higher-order partial derivatives**:
- f_xx = ≤f/x≤
- f_yy = ≤f/y≤
- f_xy = ≤f/xy (mixed partial)
- f_yx = ≤f/yx (mixed partial)

**Clairaut's Theorem**: If f_xy and f_yx are continuous, then f_xy = f_yx.

### Gradient

The **gradient** of f is a vector of partial derivatives:
```
f = <f/x, f/y, f/z> = f_x∑i + f_y∑j + f_z∑k
```

**Properties**:
- Points in direction of maximum rate of increase
- Perpendicular to level curves/surfaces
- Magnitude is the maximum rate of change

### Directional Derivatives

The **directional derivative** of f at point P in direction of unit vector u:
```
D_u f = f ∑ u
```

**Maximum rate of change** occurs in direction of f with magnitude |f|.

### Chain Rule (Multivariable)

**Case 1**: z = f(x,y), x = g(t), y = h(t)
```
dz/dt = (z/x)∑(dx/dt) + (z/y)∑(dy/dt)
```

**Case 2**: z = f(x,y), x = g(s,t), y = h(s,t)
```
z/s = (z/x)∑(x/s) + (z/y)∑(y/s)
z/t = (z/x)∑(x/t) + (z/y)∑(y/t)
```

### Extrema of Multivariable Functions

**Critical points**: Where f = 0 or f does not exist

**Second Derivative Test**: At critical point (a,b):
```
D = f_xx(a,b)∑f_yy(a,b) - [f_xy(a,b)]≤
```

- If D > 0 and f_xx(a,b) > 0: local minimum
- If D > 0 and f_xx(a,b) < 0: local maximum
- If D < 0: saddle point
- If D = 0: test is inconclusive

### Multiple Integrals

**Double Integral** over region R:
```
++_R f(x,y) dA
```

**Fubini's Theorem**: If R = [a,b] ◊ [c,d]:
```
++_R f(x,y) dA = +[a to b] +[c to d] f(x,y) dy dx
                = +[c to d] +[a to b] f(x,y) dx dy
```

**Applications**:
- Volume under surface: V = ++_R f(x,y) dA
- Area of region: A = ++_R 1 dA
- Mass: m = ++_R ¡(x,y) dA

**Triple Integral**:
```
+++_E f(x,y,z) dV
```

### Coordinate Systems

**Polar Coordinates** (x = r∑cos(∏), y = r∑sin(∏)):
```
++_R f(x,y) dA = ++ f(r∑cos(∏), r∑sin(∏))∑r dr d∏
```

**Cylindrical Coordinates** (x = r∑cos(∏), y = r∑sin(∏), z = z):
```
+++_E f(x,y,z) dV = +++ f(r∑cos(∏), r∑sin(∏), z)∑r dz dr d∏
```

**Spherical Coordinates** (x = ¡∑sin(∆)∑cos(∏), y = ¡∑sin(∆)∑sin(∏), z = ¡∑cos(∆)):
```
+++_E f(x,y,z) dV = +++ f(¡,∏,∆)∑¡≤∑sin(∆) d¡ d∏ d∆
```

### Vector Calculus

**Line Integrals**:
```
+_C f(x,y) ds = +[a to b] f(r(t))∑|r'(t)| dt
+_C F ∑ dr = +[a to b] F(r(t)) ∑ r'(t) dt
```

**Green's Theorem** (relates line integral to double integral):
```
._C P dx + Q dy = ++_D (Q/x - P/y) dA
```

**Conservative Vector Fields**:
- F = f for some scalar function f (potential function)
- Line integral is path-independent
- ._C F ∑ dr = 0 for any closed curve C

**Test**: F = <P, Q> is conservative if P/y = Q/x

---

## Differential Equations

### First-Order ODEs

**General form**: dy/dx = f(x,y) or M(x,y)dx + N(x,y)dy = 0

### Separable Equations

**Form**: dy/dx = g(x)∑h(y)

**Method**:
1. Separate variables: [1/h(y)]dy = g(x)dx
2. Integrate both sides
3. Solve for y if possible

**Example**: dy/dx = xy
```
dy/y = x dx
ln|y| = x≤/2 + C
y = Ae^(x≤/2)
```

### Linear First-Order ODEs

**Standard form**: dy/dx + P(x)∑y = Q(x)

**Method** (Integrating Factor):
1. Compute º(x) = e^(+P(x)dx)
2. Multiply equation by º(x)
3. Left side becomes d/dx[º(x)∑y]
4. Integrate: º(x)∑y = +º(x)∑Q(x)dx
5. Solve for y

**Example**: dy/dx + y = e^x
```
º(x) = e^+1 dx = e^x
e^x∑dy/dx + e^x∑y = e^(2x)
d/dx[e^x∑y] = e^(2x)
e^x∑y = (1/2)e^(2x) + C
y = (1/2)e^x + Ce^(-x)
```

### Exact Equations

**Form**: M(x,y)dx + N(x,y)dy = 0 is **exact** if M/y = N/x

**Solution**: Find function f(x,y) such that:
- f/x = M
- f/y = N

Then f(x,y) = C is the solution.

### Second-Order Linear ODEs

**Homogeneous**: ay'' + by' + cy = 0

**Characteristic equation**: ar≤ + br + c = 0

**Solutions**:
1. **Two distinct real roots** rÅ, rÇ: y = CÅe^(rÅx) + CÇe^(rÇx)
2. **Repeated root** r: y = (CÅ + CÇx)e^(rx)
3. **Complex roots** r = ± ± ≤i: y = e^(±x)[CÅcos(≤x) + CÇsin(≤x)]

**Non-homogeneous**: ay'' + by' + cy = g(x)

**General solution**: y = y_h + y_p
- y_h: homogeneous solution
- y_p: particular solution (use method of undetermined coefficients or variation of parameters)

### Applications

**Population growth**: dP/dt = kP (exponential growth)

**Newton's law of cooling**: dT/dt = -k(T - T_ambient)

**Spring-mass system**: my'' + cy' + ky = F(t)
- m: mass
- c: damping coefficient
- k: spring constant
- F(t): external force

**RC circuits**: RC∑dV/dt + V = V_source

---

## Summary

This document covers the fundamental concepts of calculus:

- **Limits and Continuity**: Foundation for understanding change
- **Derivatives**: Instantaneous rates of change and tangent slopes
- **Differentiation Techniques**: Tools for computing derivatives
- **Integration**: Accumulation and area under curves
- **Integration Techniques**: Methods for evaluating integrals
- **Applications**: Real-world uses of calculus
- **Sequences and Series**: Infinite processes and approximations
- **Multivariable Calculus**: Extension to higher dimensions
- **Differential Equations**: Modeling change and dynamics

These concepts form the backbone of mathematical analysis and are essential tools in physics, engineering, economics, and many other fields.
