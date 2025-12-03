# Calculus

## Overview

Calculus is the mathematical study of continuous change, consisting of two complementary branches: **differential calculus** (concerning rates of change and slopes) and **integral calculus** (concerning accumulation of quantities and areas). Together, they form one of the most powerful frameworks in mathematics, with applications spanning physics, engineering, economics, biology, and beyond.

## Historical Context

Calculus was independently developed in the late 17th century by:
- **Isaac Newton** (1642-1727): Developed "fluxions" for physics and mechanics
- **Gottfried Wilhelm Leibniz** (1646-1716): Created the notation we use today (dx, dy, ∫)

The fundamental insight was that many natural phenomena involve instantaneous rates of change and cumulative effects, requiring mathematics beyond algebra and geometry.

## Part I: Limits and Continuity

### The Concept of a Limit

The limit is the foundational concept of calculus. It describes the behavior of a function as the input approaches a particular value.

**Formal Definition:**
```
lim[x→a] f(x) = L
```
means that f(x) can be made arbitrarily close to L by making x sufficiently close to a (but not equal to a).

**Epsilon-Delta Definition (Rigorous):**
For every ε > 0, there exists a δ > 0 such that:
```
if 0 < |x - a| < δ, then |f(x) - L| < ε
```

### Types of Limits

1. **One-sided limits:**
   - Right-hand limit: `lim[x→a⁺] f(x)`
   - Left-hand limit: `lim[x→a⁻] f(x)`

2. **Infinite limits:**
   ```
   lim[x→a] f(x) = ∞  (vertical asymptote)
   lim[x→∞] f(x) = L  (horizontal asymptote)
   ```

3. **Indeterminate forms:**
   - 0/0, ∞/∞, 0·∞, ∞-∞, 0⁰, 1^∞, ∞⁰

### Limit Laws

If `lim[x→a] f(x) = L` and `lim[x→a] g(x) = M`, then:

1. **Sum/Difference:** `lim[x→a] [f(x) ± g(x)] = L ± M`
2. **Product:** `lim[x→a] [f(x)·g(x)] = L·M`
3. **Quotient:** `lim[x→a] [f(x)/g(x)] = L/M` (if M ≠ 0)
4. **Power:** `lim[x→a] [f(x)]ⁿ = Lⁿ`
5. **Composition:** `lim[x→a] g(f(x)) = g(lim[x→a] f(x))`

### Continuity

A function f is **continuous at a** if:
1. f(a) is defined
2. `lim[x→a] f(x)` exists
3. `lim[x→a] f(x) = f(a)`

**Properties of continuous functions:**
- Polynomials are continuous everywhere
- Rational functions are continuous where denominator ≠ 0
- Trigonometric functions are continuous on their domains
- Exponential and logarithmic functions are continuous on their domains

**Intermediate Value Theorem (IVT):**
If f is continuous on [a,b] and k is between f(a) and f(b), then there exists c ∈ (a,b) such that f(c) = k.

## Part II: Differential Calculus

### The Derivative

The derivative measures the instantaneous rate of change of a function.

**Definition (Limit definition):**
```
f'(x) = lim[h→0] [f(x+h) - f(x)] / h
```

**Interpretations:**
1. **Geometric:** Slope of the tangent line at a point
2. **Physical:** Instantaneous velocity (if f is position)
3. **Rate of change:** How quickly f changes with respect to x

### Notation

Multiple equivalent notations exist:
```
f'(x) = y' = dy/dx = df/dx = d/dx[f(x)] = Df(x)
```

### Basic Differentiation Rules

1. **Constant Rule:** `d/dx[c] = 0`

2. **Power Rule:** `d/dx[xⁿ] = n·xⁿ⁻¹`

3. **Constant Multiple:** `d/dx[c·f(x)] = c·f'(x)`

4. **Sum/Difference:** `d/dx[f(x) ± g(x)] = f'(x) ± g'(x)`

5. **Product Rule:** `d/dx[f(x)·g(x)] = f'(x)g(x) + f(x)g'(x)`

6. **Quotient Rule:** `d/dx[f(x)/g(x)] = [f'(x)g(x) - f(x)g'(x)] / [g(x)]²`

7. **Chain Rule:** `d/dx[f(g(x))] = f'(g(x))·g'(x)`
   - In Leibniz notation: `dy/dx = dy/du · du/dx`

### The Chain Rule in Depth

The chain rule is arguably the most important and widely-used differentiation technique. It appears in nearly every application of calculus, often implicitly. A deep understanding of the chain rule is essential for mastery of differential calculus.

#### Theoretical Foundation

**Formal Statement:**
If y = f(u) and u = g(x), then the composite function y = f(g(x)) has derivative:
```
dy/dx = f'(g(x))·g'(x)
```
Or equivalently in Leibniz notation:
```
dy/dx = (dy/du)·(du/dx)
```

**Intuitive Explanation: "Rates of Change Multiply"**

The chain rule states that when quantities are linked in a chain, their rates of change multiply. If:
- x changes at some rate
- u changes at rate g'(x) per unit change in x
- y changes at rate f'(u) per unit change in u

Then y changes at rate f'(u)·g'(x) per unit change in x.

**Geometric Interpretation:**

Function composition can be viewed as a sequence of transformations. If g transforms x to u, and f transforms u to y, then:
- The slope at x in the g-transformation is g'(x)
- The slope at u in the f-transformation is f'(u)
- These slopes "compound" multiplicatively

**Proof Sketch (Limit Definition):**

Starting from the definition of derivative:
```
d/dx[f(g(x))] = lim[h→0] [f(g(x+h)) - f(g(x))] / h
```

Let Δu = g(x+h) - g(x). Then:
```
= lim[h→0] [f(g(x)+Δu) - f(g(x))] / h
= lim[h→0] {[f(g(x)+Δu) - f(g(x))] / Δu} · {Δu / h}
= lim[h→0] [f(g(x)+Δu) - f(g(x))] / Δu · lim[h→0] Δu/h
= f'(g(x)) · g'(x)
```

(Technical note: This assumes Δu ≠ 0 for small h, which holds when g'(x) ≠ 0)

**The Infinitesimal Perspective:**

In Leibniz's differential notation:
- If u changes by du, then y changes by dy = f'(u)·du
- If x changes by dx, then u changes by du = g'(x)·dx
- Substituting: dy = f'(u)·[g'(x)·dx] = f'(g(x))·g'(x)·dx
- Therefore: dy/dx = f'(g(x))·g'(x)

This heuristic reasoning, while not rigorous, provides powerful intuition.

#### Multiple Notations and Forms

**1. Leibniz Notation (most intuitive):**
```
dy/dx = (dy/du)·(du/dx)
```
The "cancellation" of du makes the formula memorable (though this isn't rigorous cancellation).

**2. Prime Notation:**
If h(x) = f(g(x)), then:
```
h'(x) = f'(g(x))·g'(x)
```

**3. Operator Notation:**
```
D[f(g(x))] = Df(g(x))·Dg(x)
```
where D denotes the differentiation operator.

**4. Verbal Form:**
"The derivative of the outer function, evaluated at the inner function, times the derivative of the inner function"

Or more concisely: "Derivative of outer at inner, times derivative of inner"

#### Pattern Recognition Framework

Recognizing chain rule patterns is crucial for efficiency. Here are the most common forms:

**1. General Form:**
```
d/dx[f(g(x))] = f'(g(x))·g'(x)
```

**2. Power Chain Pattern:**
```
d/dx[(g(x))ⁿ] = n·(g(x))ⁿ⁻¹·g'(x)
```
Example: `d/dx[(x²+1)³] = 3(x²+1)²·(2x) = 6x(x²+1)²`

**3. Exponential Chain Pattern:**
```
d/dx[e^(g(x))] = e^(g(x))·g'(x)
```
Example: `d/dx[e^(x²)] = e^(x²)·(2x) = 2x·e^(x²)`

**4. Trigonometric Chain Patterns:**
```
d/dx[sin(g(x))] = cos(g(x))·g'(x)
d/dx[cos(g(x))] = -sin(g(x))·g'(x)
d/dx[tan(g(x))] = sec²(g(x))·g'(x)
```
Example: `d/dx[sin(3x)] = cos(3x)·3 = 3cos(3x)`

**5. Logarithmic Chain Pattern:**
```
d/dx[ln(g(x))] = g'(x)/g(x)
```
Example: `d/dx[ln(x²+1)] = 2x/(x²+1)`

**Recognition Strategy:**
1. Identify whether you have a composition f(g(x))
2. Identify the outer function f and inner function g
3. Find f'(u) where u = g(x)
4. Find g'(x)
5. Multiply: f'(g(x))·g'(x)

#### Progressive Examples: Simple to Complex

##### Level 1: Simple Compositions (Linear or Simple Inner Functions)

**Example 1:** Find `d/dx[sin(3x)]`
```
Outer function: f(u) = sin(u), so f'(u) = cos(u)
Inner function: g(x) = 3x, so g'(x) = 3

d/dx[sin(3x)] = cos(3x)·3 = 3cos(3x)
```

**Example 2:** Find `d/dx[(x²+1)⁵]`
```
Outer: f(u) = u⁵, so f'(u) = 5u⁴
Inner: g(x) = x²+1, so g'(x) = 2x

d/dx[(x²+1)⁵] = 5(x²+1)⁴·(2x) = 10x(x²+1)⁴
```

**Example 3:** Find `d/dx[e^(2x)]`
```
Outer: f(u) = eᵘ, so f'(u) = eᵘ
Inner: g(x) = 2x, so g'(x) = 2

d/dx[e^(2x)] = e^(2x)·2 = 2e^(2x)
```

**Example 4:** Find `d/dx[ln(x²)]`
```
Outer: f(u) = ln(u), so f'(u) = 1/u
Inner: g(x) = x², so g'(x) = 2x

d/dx[ln(x²)] = (1/x²)·(2x) = 2x/x² = 2/x
```
(Alternative: ln(x²) = 2ln(x), so derivative is 2/x directly)

**Example 5:** Find `d/dx[cos(x³)]`
```
Outer: f(u) = cos(u), so f'(u) = -sin(u)
Inner: g(x) = x³, so g'(x) = 3x²

d/dx[cos(x³)] = -sin(x³)·(3x²) = -3x²sin(x³)
```

##### Level 2: Moderate Complexity

**Example 6:** Find `d/dx[sin²(x)]` (power of trig function)
```
Rewrite: sin²(x) = [sin(x)]²

Outer: f(u) = u², so f'(u) = 2u
Inner: g(x) = sin(x), so g'(x) = cos(x)

d/dx[sin²(x)] = 2sin(x)·cos(x) = sin(2x)
```
(using double-angle identity in last step)

**Example 7:** Find `d/dx[e^(x²+3x)]`
```
Outer: f(u) = eᵘ, so f'(u) = eᵘ
Inner: g(x) = x²+3x, so g'(x) = 2x+3

d/dx[e^(x²+3x)] = e^(x²+3x)·(2x+3) = (2x+3)e^(x²+3x)
```

**Example 8:** Find `d/dx[√(x²+1)]`
```
Rewrite: √(x²+1) = (x²+1)^(1/2)

Outer: f(u) = u^(1/2), so f'(u) = (1/2)u^(-1/2)
Inner: g(x) = x²+1, so g'(x) = 2x

d/dx[(x²+1)^(1/2)] = (1/2)(x²+1)^(-1/2)·(2x) = x/√(x²+1)
```

**Example 9:** Find `d/dx[tan⁻¹(2x)]`
```
Outer: f(u) = tan⁻¹(u), so f'(u) = 1/(1+u²)
Inner: g(x) = 2x, so g'(x) = 2

d/dx[tan⁻¹(2x)] = [1/(1+(2x)²)]·2 = 2/(1+4x²)
```

**Example 10:** Find `d/dx[(3x-1)⁴/(2x+5)³]`
```
This requires quotient rule combined with chain rule.

Let u = (3x-1)⁴ and v = (2x+5)³

u' = 4(3x-1)³·3 = 12(3x-1)³
v' = 3(2x+5)²·2 = 6(2x+5)²

Using quotient rule:
d/dx[u/v] = (u'v - uv')/v²
= [12(3x-1)³(2x+5)³ - (3x-1)⁴·6(2x+5)²] / [(2x+5)⁶]
= [6(3x-1)³(2x+5)²(2(2x+5) - (3x-1))] / [(2x+5)⁶]
= [6(3x-1)³(2(2x+5) - (3x-1))] / [(2x+5)⁴]
= [6(3x-1)³(4x+10-3x+1)] / [(2x+5)⁴]
= [6(3x-1)³(x+11)] / [(2x+5)⁴]
```

##### Level 3: Multiple Compositions (Chain within Chain)

**Example 11:** Find `d/dx[sin(cos(x))]` (double composition)
```
Work from outside in:
Outermost: f(u) = sin(u), so f'(u) = cos(u)
Middle: g(v) = cos(v), so g'(v) = -sin(v)
Innermost: v = x

Chain rule twice:
d/dx[sin(cos(x))] = cos(cos(x))·d/dx[cos(x)]
                  = cos(cos(x))·(-sin(x))
                  = -sin(x)cos(cos(x))
```

**Example 12:** Find `d/dx[e^(sin(x²))]` (triple composition)
```
Outermost: f = eᵘ, so f' = eᵘ
Middle: u = sin(v), so du/dv = cos(v)
Inner: v = x², so dv/dx = 2x

Chain them:
d/dx[e^(sin(x²))] = e^(sin(x²))·d/dx[sin(x²)]
                  = e^(sin(x²))·cos(x²)·d/dx[x²]
                  = e^(sin(x²))·cos(x²)·2x
                  = 2x·cos(x²)·e^(sin(x²))
```

**Example 13:** Find `d/dx[ln(√(x²+1))]`
```
Rewrite: ln(√(x²+1)) = ln((x²+1)^(1/2)) = (1/2)ln(x²+1)

Method 1 (using log properties first):
d/dx[(1/2)ln(x²+1)] = (1/2)·d/dx[ln(x²+1)]
                     = (1/2)·[2x/(x²+1)]
                     = x/(x²+1)

Method 2 (chain rule directly):
Outer: f(u) = ln(u), so f'(u) = 1/u
Middle: u = √v, so du/dv = 1/(2√v)
Inner: v = x²+1, so dv/dx = 2x

d/dx[ln(√(x²+1))] = [1/√(x²+1)]·[1/(2√(x²+1))]·(2x)
                  = 2x / [2(x²+1)]
                  = x/(x²+1)
```

**Example 14:** Find `d/dx[(sin(e^x))⁵]`
```
Outermost: f(u) = u⁵, so f'(u) = 5u⁴
Middle: u = sin(v), so du/dv = cos(v)
Inner: v = e^x, so dv/dx = e^x

d/dx[(sin(e^x))⁵] = 5(sin(e^x))⁴·d/dx[sin(e^x)]
                  = 5(sin(e^x))⁴·cos(e^x)·d/dx[e^x]
                  = 5(sin(e^x))⁴·cos(e^x)·e^x
                  = 5e^x·cos(e^x)·(sin(e^x))⁴
```

**Example 15:** Find `d/dx[tan(ln(x²+1))]`
```
Outer: f(u) = tan(u), so f'(u) = sec²(u)
Middle: u = ln(v), so du/dv = 1/v
Inner: v = x²+1, so dv/dx = 2x

d/dx[tan(ln(x²+1))] = sec²(ln(x²+1))·d/dx[ln(x²+1)]
                    = sec²(ln(x²+1))·[2x/(x²+1)]
                    = [2x·sec²(ln(x²+1))] / (x²+1)
```

##### Level 4: Combined with Other Rules

**Example 16:** Product rule + chain rule: Find `d/dx[x·sin(x²)]`
```
Using product rule: d/dx[u·v] = u'v + uv'
u = x, so u' = 1
v = sin(x²), so v' = cos(x²)·2x (chain rule)

d/dx[x·sin(x²)] = 1·sin(x²) + x·cos(x²)·2x
                = sin(x²) + 2x²cos(x²)
```

**Example 17:** Quotient rule + chain rule: Find `d/dx[e^x/√(x²+1)]`
```
u = e^x, so u' = e^x
v = √(x²+1) = (x²+1)^(1/2), so v' = (1/2)(x²+1)^(-1/2)·2x = x/√(x²+1)

d/dx[e^x/√(x²+1)] = [e^x·√(x²+1) - e^x·x/√(x²+1)] / (x²+1)
                  = [e^x·√(x²+1)·√(x²+1) - e^x·x] / [(x²+1)√(x²+1)]
                  = e^x[(x²+1) - x] / [(x²+1)^(3/2)]
                  = e^x(x²-x+1) / (x²+1)^(3/2)
```

**Example 18:** All rules combined: Find `d/dx[x²·e^(sin(x))·ln(x+1)]`
```
This is a product of three functions. Use product rule iteratively.
Let u = x², v = e^(sin(x)), w = ln(x+1)

d/dx[uvw] = u'vw + uv'w + uvw'

u' = 2x
v' = e^(sin(x))·cos(x)  (chain rule)
w' = 1/(x+1)            (chain rule)

Result:
= 2x·e^(sin(x))·ln(x+1) + x²·e^(sin(x))·cos(x)·ln(x+1) + x²·e^(sin(x))·1/(x+1)

= x·e^(sin(x))[2ln(x+1) + x·cos(x)·ln(x+1) + x/(x+1)]
```

#### Special Applications of the Chain Rule

##### 1. Implicit Differentiation

The chain rule is **fundamental** to implicit differentiation. When differentiating an equation involving y with respect to x, we treat y as an implicit function of x and apply the chain rule.

**Example:** Find dy/dx for the circle `x² + y² = 25`

Differentiate both sides with respect to x:
```
d/dx[x²] + d/dx[y²] = d/dx[25]
2x + 2y·(dy/dx) = 0    ← chain rule applied to y²
```
The term `d/dx[y²]` requires the chain rule: if u = y², then by chain rule, `du/dx = 2y·(dy/dx)`.

Solving for dy/dx:
```
2y·(dy/dx) = -2x
dy/dx = -x/y
```

**Another example:** Find dy/dx for `e^y + xy = e`
```
d/dx[e^y] + d/dx[xy] = d/dx[e]
e^y·(dy/dx) + y + x·(dy/dx) = 0    ← chain rule on e^y
(e^y + x)·(dy/dx) = -y
dy/dx = -y/(e^y + x)
```

##### 2. Related Rates

Related rates problems involve multiple variables changing with respect to time. The chain rule connects these rates.

**Example:** A spherical balloon is being inflated. If the radius increases at 2 cm/s, how fast is the volume increasing when r = 5 cm?

Given: dr/dt = 2 cm/s, find dV/dt when r = 5

Volume of sphere: V = (4/3)πr³

Using chain rule:
```
dV/dt = dV/dr · dr/dt = 4πr² · dr/dt
```

When r = 5 and dr/dt = 2:
```
dV/dt = 4π(5)²·2 = 200π cm³/s
```

**Example:** A ladder 10 m long leans against a wall. If the bottom slides away at 1 m/s, how fast is the top sliding down when the bottom is 6 m from the wall?

Let x = distance from wall to bottom, y = height of top on wall
Given: x² + y² = 100 (Pythagorean theorem), dx/dt = 1 m/s

Differentiate with respect to time:
```
2x·(dx/dt) + 2y·(dy/dt) = 0
```

When x = 6: y = √(100-36) = 8

```
2(6)(1) + 2(8)·(dy/dt) = 0
12 + 16·(dy/dt) = 0
dy/dt = -12/16 = -3/4 m/s
```

The negative sign indicates the top is sliding down.

##### 3. Parametric Equations

For parametric curves where x = f(t) and y = g(t), the chain rule gives us dy/dx:

```
dy/dx = (dy/dt) / (dx/dt)
```

This comes from the chain rule: `dy/dx = (dy/dt)·(dt/dx) = (dy/dt) / (dx/dt)`

**Example:** Find dy/dx for the parametric curve x = t², y = t³

```
dx/dt = 2t
dy/dt = 3t²

dy/dx = (dy/dt)/(dx/dt) = 3t²/(2t) = 3t/2
```

We can also express this in terms of x: since t = √x (for t > 0), we have:
```
dy/dx = 3√x/2
```

##### 4. Inverse Functions

The derivative of an inverse function can be found using the chain rule. If y = f⁻¹(x), then x = f(y).

Differentiating both sides with respect to x:
```
1 = f'(y)·(dy/dx)
```

Therefore:
```
(f⁻¹)'(x) = dy/dx = 1/f'(y) = 1/f'(f⁻¹(x))
```

**Example:** Find the derivative of y = ln(x) using the fact that it's the inverse of e^x.

If y = ln(x), then x = e^y.

Differentiating: 1 = e^y·(dy/dx)

Therefore: dy/dx = 1/e^y = 1/e^(ln(x)) = 1/x

This confirms that `d/dx[ln(x)] = 1/x`.

#### Higher-Order and Generalized Chain Rules

##### Multiple Variables (Preview of Multivariable Calculus)

If z = f(x,y) where x = g(t) and y = h(t), then z depends on t through both x and y:

```
dz/dt = (∂z/∂x)·(dx/dt) + (∂z/∂y)·(dy/dt)
```

This is the **multivariable chain rule** or **total derivative**.

**Example:** If z = x²y where x = cos(t) and y = sin(t), find dz/dt.

```
∂z/∂x = 2xy
∂z/∂y = x²
dx/dt = -sin(t)
dy/dt = cos(t)

dz/dt = 2xy·(-sin(t)) + x²·cos(t)
      = 2cos(t)sin(t)·(-sin(t)) + cos²(t)·cos(t)
      = -2cos(t)sin²(t) + cos³(t)
      = cos(t)[cos²(t) - 2sin²(t)]
```

##### Generalized Chain Rule for Multiple Compositions

For a composition of three or more functions, f(g(h(x))), the pattern is:

```
d/dx[f(g(h(x)))] = f'(g(h(x)))·g'(h(x))·h'(x)
```

**General pattern:** Work from the outside in, multiplying derivatives at each level.

For f₁(f₂(f₃(...fₙ(x)...))):
```
d/dx = f₁'(f₂(...))·f₂'(f₃(...))·f₃'(f₄(...))···fₙ'(x)
```

**Example:** Find d/dx[sin(cos(tan(x)))]
```
= cos(cos(tan(x)))·d/dx[cos(tan(x))]
= cos(cos(tan(x)))·(-sin(tan(x)))·d/dx[tan(x)]
= cos(cos(tan(x)))·(-sin(tan(x)))·sec²(x)
= -sec²(x)·sin(tan(x))·cos(cos(tan(x)))
```

#### Common Mistakes and How to Avoid Them

##### Mistake 1: Forgetting the Inner Derivative

**WRONG:**
```
d/dx[sin(x²)] = cos(x²)  ✗
```

**CORRECT:**
```
d/dx[sin(x²)] = cos(x²)·2x  ✓
```

**How to avoid:** Always ask "is this a composition?" If yes, multiply by the inner derivative.

##### Mistake 2: Misidentifying Inner and Outer Functions

**Example:** Differentiate e^(x²) vs (e^x)²

These are **different** functions:
- e^(x²): Outer is e^u, inner is x² → derivative is e^(x²)·2x
- (e^x)²: Outer is u², inner is e^x → derivative is 2e^x·e^x = 2e^(2x)

**How to avoid:** Write out f(u) and u = g(x) explicitly before differentiating.

##### Mistake 3: Using Chain Rule When Not Needed

**Example:** d/dx[sin(x)]

This is NOT a composition (x is not a function of anything else), so:
```
d/dx[sin(x)] = cos(x)  ✓
```

NOT cos(x)·1 (though technically correct, it's unnecessarily complicated).

**How to avoid:** Only use chain rule when you have a genuine composition f(g(x)) where g(x) ≠ x.

##### Mistake 4: Stopping Too Early in Nested Compositions

**WRONG:**
```
d/dx[e^(sin(x²))] = e^(sin(x²))·cos(x²)  ✗  (forgot to differentiate x²)
```

**CORRECT:**
```
d/dx[e^(sin(x²))] = e^(sin(x²))·cos(x²)·2x  ✓
```

**How to avoid:** Continue applying the chain rule until you reach x (or the independent variable).

##### Mistake 5: Sign Errors in Multi-Step Chains

**Example:** d/dx[cos(sin(x))]
```
= -sin(sin(x))·cos(x)  ✓  (note the negative from d/dx[cos(u)] = -sin(u))
```

NOT sin(sin(x))·cos(x) ✗

**How to avoid:** Write each derivative carefully, including signs.

#### Visual and Geometric Understanding

##### The "Slopes Multiply" Interpretation

Imagine a multi-stage process:
1. Input x produces intermediate result u = g(x)
2. Intermediate u produces final output y = f(u)

If x changes by a small amount Δx:
- u changes by approximately g'(x)·Δx
- y then changes by approximately f'(u)·[g'(x)·Δx] = f'(u)·g'(x)·Δx

Therefore, the rate of change of y with respect to x is f'(u)·g'(x).

**Analogy:** If a car travels at 60 mph, and 1 mile equals 1.6 km, how fast is the car traveling in km/h?
```
d(km)/d(hour) = [d(km)/d(mile)]·[d(mile)/d(hour)] = 1.6 · 60 = 96 km/h
```

This is exactly the chain rule structure!

##### Tree Diagrams for Complex Chains

For complex dependencies, tree diagrams help visualize which derivatives to multiply.

Example: z depends on u and v, which both depend on x:

```
        z
       / \
      u   v
       \ /
        x
```

Then: dz/dx = (∂z/∂u)·(du/dx) + (∂z/∂v)·(dv/dx)

Each path from z to x represents a term in the derivative, formed by multiplying the derivatives along that path.

#### Connection to Integration: The Reverse Chain Rule

The chain rule in reverse gives us **u-substitution** in integration:

If we know that:
```
d/dx[F(g(x))] = F'(g(x))·g'(x) = f(g(x))·g'(x)
```

Then by the Fundamental Theorem of Calculus:
```
∫ f(g(x))·g'(x) dx = F(g(x)) + C
```

Or with u-substitution (u = g(x), du = g'(x)dx):
```
∫ f(u) du = F(u) + C
```

**Example:** Evaluate ∫ 2x·e^(x²) dx

Recognize this as having the form f(g(x))·g'(x):
- g(x) = x², so g'(x) = 2x ✓
- f(u) = e^u

Let u = x², du = 2x dx:
```
∫ 2x·e^(x²) dx = ∫ e^u du = e^u + C = e^(x²) + C
```

We can verify by differentiation (using chain rule):
```
d/dx[e^(x²)] = e^(x²)·2x ✓
```

**Recognition of chain rule patterns in integrands is essential for integration!**

### Common Derivatives

| Function | Derivative |
|----------|------------|
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `aˣ` | `aˣ·ln(a)` |
| `ln(x)` | `1/x` |
| `log_a(x)` | `1/(x·ln(a))` |
| `sin(x)` | `cos(x)` |
| `cos(x)` | `-sin(x)` |
| `tan(x)` | `sec²(x)` |
| `cot(x)` | `-csc²(x)` |
| `sec(x)` | `sec(x)tan(x)` |
| `csc(x)` | `-csc(x)cot(x)` |
| `sin⁻¹(x)` | `1/√(1-x²)` |
| `cos⁻¹(x)` | `-1/√(1-x²)` |
| `tan⁻¹(x)` | `1/(1+x²)` |

### Implicit Differentiation

**Note:** This technique fundamentally relies on the chain rule. See "The Chain Rule in Depth" section above for detailed explanation and examples.

For equations not explicitly solved for y (e.g., `x² + y² = 1`):
1. Differentiate both sides with respect to x
2. Treat y as a function of x (apply chain rule)
3. Solve for dy/dx

**Example:** Circle `x² + y² = r²`
```
d/dx[x²] + d/dx[y²] = d/dx[r²]
2x + 2y(dy/dx) = 0
dy/dx = -x/y
```

### Logarithmic Differentiation

**Note:** This technique combines logarithmic properties with implicit differentiation and the chain rule.

For products, quotients, or powers of functions:
1. Take ln of both sides
2. Use log properties to simplify
3. Differentiate implicitly (applying chain rule)
4. Solve for y'

**Example:** `y = x^x`
```
ln(y) = x·ln(x)
(1/y)·y' = ln(x) + x·(1/x) = ln(x) + 1
y' = y·(ln(x) + 1) = x^x·(ln(x) + 1)
```

### Higher-Order Derivatives

- **Second derivative:** `f''(x) = d²y/dx²` (concavity, acceleration)
- **Third derivative:** `f'''(x) = d³y/dx³` (jerk in physics)
- **nth derivative:** `f⁽ⁿ⁾(x) = dⁿy/dxⁿ`

### Related Rates

**Note:** Related rates problems are applications of the chain rule with respect to time. See "The Chain Rule in Depth → Special Applications → Related Rates" for detailed examples.

Problems involving multiple variables changing with respect to time:
1. Identify variables and what's given/unknown
2. Find equation relating the variables
3. Differentiate with respect to time (using the chain rule)
4. Substitute known values and solve

### Applications of Derivatives

#### 1. Critical Points and Extrema

**Critical points** occur where `f'(x) = 0` or `f'(x)` is undefined.

**First Derivative Test:**
- If f' changes from + to −, local maximum
- If f' changes from − to +, local minimum
- If f' doesn't change sign, neither

**Second Derivative Test:**
- If `f'(c) = 0` and `f''(c) > 0`, local minimum at c
- If `f'(c) = 0` and `f''(c) < 0`, local maximum at c
- If `f''(c) = 0`, test is inconclusive

#### 2. Curve Sketching

Analyze:
1. **Domain** and **intercepts**
2. **Symmetry** (even, odd, periodic)
3. **Asymptotes** (vertical, horizontal, oblique)
4. **First derivative:** Increasing/decreasing intervals, local extrema
5. **Second derivative:** Concavity, inflection points

**Concavity:**
- `f''(x) > 0`: Concave up (∪)
- `f''(x) < 0`: Concave down (∩)
- **Inflection point:** Where concavity changes

#### 3. Optimization

To find absolute extrema on [a,b]:
1. Find critical points in (a,b)
2. Evaluate f at critical points and endpoints
3. Largest value is absolute max, smallest is absolute min

#### 4. Linear Approximation

Near x = a:
```
L(x) = f(a) + f'(a)(x - a)
```
This is the tangent line approximation.

#### 5. Differentials

Small change approximation:
```
dy = f'(x)dx
Δy ≈ dy for small Δx
```

### Mean Value Theorem (MVT)

If f is continuous on [a,b] and differentiable on (a,b), then there exists c ∈ (a,b) such that:
```
f'(c) = [f(b) - f(a)] / (b - a)
```

**Interpretation:** At some point, the instantaneous rate equals the average rate.

**Rolle's Theorem** (special case): If f(a) = f(b), then f'(c) = 0 for some c.

### L'Hôpital's Rule

For indeterminate forms 0/0 or ∞/∞:
```
lim[x→a] f(x)/g(x) = lim[x→a] f'(x)/g'(x)
```
(if the limit on the right exists)

Can be applied repeatedly. Also works for other indeterminate forms after algebraic manipulation.

## Part III: Integral Calculus

### The Definite Integral

The definite integral represents the signed area under a curve from a to b.

**Definition (Riemann Sum):**
```
∫[a to b] f(x)dx = lim[n→∞] Σ[i=1 to n] f(xᵢ*)Δx
```
where `Δx = (b-a)/n` and `xᵢ*` is a sample point in the ith subinterval.

**Properties:**
1. `∫[a to b] [f(x) + g(x)]dx = ∫[a to b] f(x)dx + ∫[a to b] g(x)dx`
2. `∫[a to b] c·f(x)dx = c·∫[a to b] f(x)dx`
3. `∫[a to b] f(x)dx = -∫[b to a] f(x)dx`
4. `∫[a to b] f(x)dx + ∫[b to c] f(x)dx = ∫[a to c] f(x)dx`
5. If f(x) ≥ 0 on [a,b], then `∫[a to b] f(x)dx ≥ 0`

### The Indefinite Integral (Antiderivative)

If F'(x) = f(x), then F is an **antiderivative** of f.

**Notation:**
```
∫ f(x)dx = F(x) + C
```
where C is the constant of integration.

### Fundamental Theorem of Calculus

**Part 1:** If f is continuous on [a,b] and `F(x) = ∫[a to x] f(t)dt`, then:
```
F'(x) = f(x)
```

**Part 2:** If F is an antiderivative of f on [a,b], then:
```
∫[a to b] f(x)dx = F(b) - F(a) = F(x)|[a to b]
```

**Significance:** Connects differentiation and integration as inverse operations.

### Basic Integration Rules

1. **Power Rule:** `∫ xⁿ dx = xⁿ⁺¹/(n+1) + C` (n ≠ -1)

2. **Natural Log:** `∫ (1/x)dx = ln|x| + C`

3. **Exponential:**
   - `∫ eˣ dx = eˣ + C`
   - `∫ aˣ dx = aˣ/ln(a) + C`

4. **Trigonometric:**
   - `∫ sin(x)dx = -cos(x) + C`
   - `∫ cos(x)dx = sin(x) + C`
   - `∫ sec²(x)dx = tan(x) + C`
   - `∫ csc²(x)dx = -cot(x) + C`
   - `∫ sec(x)tan(x)dx = sec(x) + C`
   - `∫ csc(x)cot(x)dx = -csc(x) + C`

5. **Inverse Trigonometric:**
   - `∫ 1/√(1-x²) dx = sin⁻¹(x) + C`
   - `∫ 1/(1+x²) dx = tan⁻¹(x) + C`
   - `∫ 1/(x√(x²-1)) dx = sec⁻¹(|x|) + C`

### Integration Techniques

#### 1. Substitution (u-substitution)

**Note:** U-substitution is the reverse of the chain rule. See "The Chain Rule in Depth → Connection to Integration" for the relationship between differentiation and integration via the chain rule.

Reverse chain rule:
```
∫ f(g(x))g'(x)dx = ∫ f(u)du  where u = g(x)
```

**Steps:**
1. Choose u = g(x)
2. Find du = g'(x)dx
3. Substitute and integrate
4. Replace u with g(x)

**Example:** `∫ 2x·eˣ² dx`
```
Let u = x², du = 2x dx
∫ eᵘ du = eᵘ + C = eˣ² + C
```

#### 2. Integration by Parts

Reverse product rule:
```
∫ u dv = uv - ∫ v du
```

**LIATE priority for choosing u:**
- **L**ogarithmic
- **I**nverse trigonometric
- **A**lgebraic (polynomials)
- **T**rigonometric
- **E**xponential

**Example:** `∫ x·eˣ dx`
```
u = x,    dv = eˣ dx
du = dx,  v = eˣ

∫ x·eˣ dx = x·eˣ - ∫ eˣ dx = x·eˣ - eˣ + C = eˣ(x-1) + C
```

#### 3. Trigonometric Substitution

For integrals involving:
- `√(a² - x²)`: use `x = a·sin(θ)`
- `√(a² + x²)`: use `x = a·tan(θ)`
- `√(x² - a²)`: use `x = a·sec(θ)`

#### 4. Partial Fractions

For rational functions `P(x)/Q(x)` where deg(P) < deg(Q):
1. Factor Q(x)
2. Decompose into partial fractions
3. Integrate term by term

**Forms:**
- `A/(x-a)` → `A·ln|x-a| + C`
- `A/(x-a)ⁿ` → `A/(1-n)(x-a)ⁿ⁻¹ + C` (n > 1)
- `(Ax+B)/(x²+bx+c)` → combination of ln and arctan

#### 5. Trigonometric Integrals

**Powers of sin and cos:**
- If odd power: save one factor, convert rest using `sin²x + cos²x = 1`
- If both even: use half-angle formulas

**Products:** Use product-to-sum formulas

#### 6. Numerical Integration

When antiderivative is difficult/impossible:
- **Midpoint Rule:** `∫[a to b] f(x)dx ≈ Δx·Σf(midpoints)`
- **Trapezoidal Rule:** `∫[a to b] f(x)dx ≈ (Δx/2)·[f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]`
- **Simpson's Rule:** `∫[a to b] f(x)dx ≈ (Δx/3)·[f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]`

### Applications of Integrals

#### 1. Area Between Curves

Area between y = f(x) and y = g(x) from a to b:
```
A = ∫[a to b] |f(x) - g(x)| dx
```

#### 2. Volume

**Disk Method** (revolving around x-axis):
```
V = π∫[a to b] [R(x)]² dx
```

**Washer Method** (hollow solid):
```
V = π∫[a to b] ([R(x)]² - [r(x)]²) dx
```

**Shell Method** (revolving around y-axis):
```
V = 2π∫[a to b] x·f(x) dx
```

**Cross-sectional area A(x):**
```
V = ∫[a to b] A(x) dx
```

#### 3. Arc Length

For y = f(x) from a to b:
```
L = ∫[a to b] √(1 + [f'(x)]²) dx
```

For parametric curve x = f(t), y = g(t):
```
L = ∫[α to β] √([dx/dt]² + [dy/dt]²) dt
```

#### 4. Surface Area

Revolving y = f(x) around x-axis:
```
S = 2π∫[a to b] f(x)√(1 + [f'(x)]²) dx
```

#### 5. Work

```
W = ∫[a to b] F(x) dx
```

Examples: pumping liquid, spring compression (Hooke's Law: F = kx)

#### 6. Average Value

Average value of f on [a,b]:
```
f_avg = (1/(b-a))·∫[a to b] f(x) dx
```

#### 7. Center of Mass

For region with density ρ(x):
```
x̄ = (1/M)·∫ x·ρ(x) dx
```
where M is total mass.

### Improper Integrals

**Type 1:** Infinite interval
```
∫[a to ∞] f(x)dx = lim[t→∞] ∫[a to t] f(x)dx
```

**Type 2:** Discontinuous integrand
```
∫[a to b] f(x)dx = lim[t→b⁻] ∫[a to t] f(x)dx
```
(if f has discontinuity at b)

**Convergence tests:**
- Direct evaluation
- Comparison test
- Limit comparison test
- p-test: `∫[1 to ∞] 1/xᵖ dx` converges if p > 1

## Part IV: Advanced Topics

### Sequences and Series

**Sequence:** Ordered list {aₙ} = a₁, a₂, a₃, ...

**Limit:** `lim[n→∞] aₙ = L` if sequence approaches L

**Series:** Sum of sequence terms
```
Σ[n=1 to ∞] aₙ = a₁ + a₂ + a₃ + ...
```

**Convergence tests:**
1. **nth term test:** If `lim aₙ ≠ 0`, series diverges
2. **Geometric series:** `Σ arⁿ` converges if |r| < 1, sum = a/(1-r)
3. **p-series:** `Σ 1/nᵖ` converges if p > 1
4. **Integral test**
5. **Comparison test**
6. **Ratio test:** `lim |aₙ₊₁/aₙ| = L`: converges if L < 1
7. **Root test:** `lim ⁿ√|aₙ| = L`: converges if L < 1
8. **Alternating series test**

### Power Series

```
Σ[n=0 to ∞] cₙ(x-a)ⁿ = c₀ + c₁(x-a) + c₂(x-a)² + ...
```

**Radius of convergence R:**
- Converges for |x-a| < R
- Diverges for |x-a| > R

**Taylor Series** (centered at a):
```
f(x) = Σ[n=0 to ∞] [f⁽ⁿ⁾(a)/n!]·(x-a)ⁿ
```

**Maclaurin Series** (centered at 0):
```
f(x) = Σ[n=0 to ∞] [f⁽ⁿ⁾(0)/n!]·xⁿ
```

**Common series:**
- `eˣ = Σ xⁿ/n!`
- `sin(x) = Σ (-1)ⁿx^(2n+1)/(2n+1)!`
- `cos(x) = Σ (-1)ⁿx^(2n)/(2n)!`
- `1/(1-x) = Σ xⁿ` for |x| < 1

### Multivariable Calculus (Brief Introduction)

**Partial derivatives:**
```
∂f/∂x = lim[h→0] [f(x+h,y) - f(x,y)] / h
```

**Gradient:** `∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)`

**Directional derivative:** Rate of change in direction u

**Multiple integrals:**
- Double integral: `∬[R] f(x,y) dA` (volume under surface)
- Triple integral: `∭[E] f(x,y,z) dV` (mass, 4D volume)

**Line integrals:** `∫[C] f(x,y) ds` along curve C

**Green's Theorem, Stokes' Theorem, Divergence Theorem:** Connect integrals over regions with integrals over boundaries

### Differential Equations

An equation involving derivatives: `F(x, y, y', y'', ...) = 0`

**Separable:** `dy/dx = g(x)h(y)` → `∫ dy/h(y) = ∫ g(x)dx`

**First-order linear:** `dy/dx + P(x)y = Q(x)`
Solution using integrating factor `μ(x) = e^(∫P(x)dx)`:
```
y = (1/μ)·∫ μQ(x)dx
```

**Second-order linear with constant coefficients:**
```
ay'' + by' + cy = 0
```
Characteristic equation: `ar² + br + c = 0`
- Real distinct roots: `y = c₁e^(r₁x) + c₂e^(r₂x)`
- Repeated root: `y = (c₁ + c₂x)e^(rx)`
- Complex roots r = α ± βi: `y = e^(αx)(c₁cos(βx) + c₂sin(βx))`

## Part V: Theoretical Foundations

### Continuity and Differentiability

**Key theorem:** If f is differentiable at a, then f is continuous at a.
(Converse is false: |x| is continuous but not differentiable at 0)

**Weierstrass function:** Continuous everywhere but differentiable nowhere (fractal-like)

### Riemann Integrability

A function is Riemann integrable on [a,b] if upper and lower Riemann sums converge to the same limit.

**Conditions for integrability:**
- Continuous functions are integrable
- Functions with finitely many discontinuities are integrable
- Monotonic functions are integrable

### Convergence

**Pointwise convergence:** Sequence of functions fₙ converges pointwise to f if `lim fₙ(x) = f(x)` for each x

**Uniform convergence:** `lim sup|fₙ(x) - f(x)| = 0`

Uniform convergence preserves continuity and allows interchange of limit and integral.

## Part VI: Applications Across Disciplines

### Physics

1. **Kinematics:**
   - Position: s(t)
   - Velocity: v(t) = s'(t)
   - Acceleration: a(t) = v'(t) = s''(t)

2. **Work and Energy:**
   - Work: `W = ∫ F·ds`
   - Kinetic energy: KE = ½mv²
   - Potential energy and conservative forces

3. **Electromagnetism:**
   - Maxwell's equations (differential form)
   - Electric flux: `Φ = ∬ E·dA`

4. **Thermodynamics:**
   - Heat transfer: dQ = mc dT
   - Entropy changes

### Engineering

1. **Optimization:** Minimize cost, maximize efficiency
2. **Control theory:** Differential equations for system dynamics
3. **Signal processing:** Fourier series and transforms
4. **Fluid dynamics:** Navier-Stokes equations

### Economics

1. **Marginal analysis:**
   - Marginal cost: C'(x)
   - Marginal revenue: R'(x)
   - Marginal profit: P'(x) = R'(x) - C'(x)

2. **Optimization:** Maximize profit, minimize cost
3. **Consumer surplus:** `∫[0 to q] D(x)dx - p·q`
4. **Elasticity of demand**

### Biology

1. **Population dynamics:** Logistic equation
   ```
   dP/dt = rP(1 - P/K)
   ```

2. **Pharmacokinetics:** Drug concentration over time
3. **Enzyme kinetics:** Michaelis-Menten equation
4. **Epidemic models:** SIR model

### Computer Science

1. **Algorithm analysis:** Growth rates, Big-O notation
2. **Machine learning:** Gradient descent
   ```
   θ := θ - α·∇J(θ)
   ```
3. **Computer graphics:** Bezier curves, ray tracing
4. **Numerical methods:** Approximation algorithms

## Part VII: Problem-Solving Strategies

### General Approach

1. **Understand the problem:**
   - What is given? What is unknown?
   - Draw a diagram if applicable
   - Identify the type of problem

2. **Devise a plan:**
   - What technique applies?
   - Can you solve a simpler related problem?
   - Work backwards from the goal

3. **Execute the plan:**
   - Work carefully through calculations
   - Check each step
   - Show your work

4. **Review/Check:**
   - Does the answer make sense?
   - Check dimensions/units
   - Verify with alternative method

### Common Pitfalls

1. **Forgetting absolute value:** ln|x|, not ln(x)
2. **Forgetting +C:** In indefinite integrals
3. **Chain rule errors:** Forgetting inner derivative
4. **Sign errors:** Especially with trig derivatives
5. **Domain issues:** Division by zero, negative square roots
6. **Misapplying rules:** Power rule when base is not constant

## Conclusion

Calculus provides a unified framework for understanding change and accumulation. Its two fundamental operations—differentiation and integration—are inverses connected by the Fundamental Theorem of Calculus.

**The essence of calculus:**
- **Local to global:** Use infinitesimal changes to understand global behavior
- **Discrete to continuous:** Extend finite sums/differences to continuous domains
- **Approximation to exactness:** Limits formalize intuitive notions

Mastery requires:
1. **Conceptual understanding:** Why techniques work
2. **Computational skill:** Executing techniques accurately
3. **Application:** Translating real problems into mathematical form
4. **Intuition:** Geometric and physical interpretation

Modern extensions include complex analysis, differential geometry, functional analysis, and abstract measure theory—but the core ideas remain rooted in the calculus of Newton and Leibniz.

## Further Study

- **Real Analysis:** Rigorous foundations (ε-δ proofs, topology)
- **Complex Analysis:** Calculus of complex functions
- **Vector Calculus:** Multivariable extensions, vector fields
- **Differential Geometry:** Calculus on manifolds
- **Functional Analysis:** Infinite-dimensional calculus
- **Calculus of Variations:** Optimizing functionals
- **Stochastic Calculus:** Calculus with randomness (Ito calculus)

---

*"The calculus is the greatest aid we have to the application of physical truth in the broadest sense of the word."* - W. F. Osgood
