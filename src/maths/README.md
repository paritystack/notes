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

### Intuition: What Limits Really Mean

**The Core Idea**: A limit is about *prediction*, not *arrival*. It answers: "If I get arbitrarily close to a point, where is my function heading?" You care about the journey, not the destination.

**Why Limits Matter**: Real-world processes approach values without reaching them. A ball rolling toward a stop, a population approaching carrying capacity, an asymptote you'll never touch—limits capture this "tendency toward" behavior.

**The Key Insight**: The limit at a point can exist even if:
- The function isn't defined there (removable discontinuity)
- The function value is different from the limit (jump)
- You can never actually reach that point (approaching infinity)

**Mental Model**: Imagine walking toward a door. You can get 1 meter away, then 0.5m, then 0.25m, then 0.125m... You keep halving the distance. The limit is the door itself, even though in this infinite process you never quite touch it. That's the essence of a limit—where you're heading, not where you are.

### Definition of a Limit

The limit of a function f(x) as x approaches a is L, written as:

```
lim(x�a) f(x) = L
```

**Formal (�-�) Definition**: For every � > 0, there exists a � > 0 such that if 0 < |x - a| < �, then |f(x) - L| < �.

**Intuitive Definition**: As x gets arbitrarily close to a (but not equal to a), f(x) gets arbitrarily close to L.

### Limit Laws

If lim(x�a) f(x) = L and lim(x�a) g(x) = M, then:

1. **Sum Rule**: lim(x�a) [f(x) + g(x)] = L + M
2. **Difference Rule**: lim(x�a) [f(x) - g(x)] = L - M
3. **Product Rule**: lim(x�a) [f(x) � g(x)] = L � M
4. **Quotient Rule**: lim(x�a) [f(x) / g(x)] = L / M (if M ` 0)
5. **Constant Multiple**: lim(x�a) [c � f(x)] = c � L
6. **Power Rule**: lim(x�a) [f(x)]^n = L^n

### Types of Limits

**One-Sided Limits**:
- Right-hand limit: lim(x�az) f(x)
- Left-hand limit: lim(x�a{) f(x)
- A limit exists if and only if both one-sided limits exist and are equal

**Infinite Limits**:
- lim(x�a) f(x) =  (function grows without bound)
- lim(x�a) f(x) = - (function decreases without bound)

**Limits at Infinity**:
- lim(x�) f(x) = L
- lim(x�-) f(x) = L

### Continuity

A function f is **continuous at x = a** if:
1. f(a) is defined
2. lim(x�a) f(x) exists
3. lim(x�a) f(x) = f(a)

**Intuition: The Pencil Test**: A function is continuous if you can draw its graph without lifting your pencil. No jumps, no holes, no breaks. Continuity means "no surprises"—small changes in input give small changes in output.

**Why Three Conditions?**
1. **Function must be defined**: You need a value at the point (no hole)
2. **Limit must exist**: Left and right approaches agree (no jump)
3. **They must match**: Where you're going equals where you are (no removable discontinuity)

**Real-World Connection**: Temperature changes continuously through the day. You don't instantly jump from 20°C to 25°C. But a light switch has a discontinuity—it's OFF then suddenly ON, no in-between.

**Types of Discontinuity**:
- **Removable**: Limit exists but f(a) is undefined or different
- **Jump**: Left and right limits exist but are unequal
- **Infinite**: Function approaches �

**Important Theorems**:
- **Intermediate Value Theorem (IVT)**: If f is continuous on [a,b] and k is between f(a) and f(b), then there exists c in (a,b) such that f(c) = k

  *Intuition*: If you walk up a mountain continuously from elevation 100m to 300m, you must cross through 200m at some point. Continuous functions can't "skip" values. This is why roots exist—if f(a) < 0 and f(b) > 0, the function must cross zero somewhere between.

- **Extreme Value Theorem (EVT)**: A continuous function on a closed interval [a,b] attains both a maximum and minimum value

  *Intuition*: On a closed, bounded hike, there's a highest point and lowest point. You can't have a highest point if the path goes to infinity (unbounded) or if there's a discontinuous jump (function not continuous). Both continuity and closed interval are essential.

---

## Derivatives

### Intuition: Measuring Instantaneous Change

**The Central Question**: How fast is something changing *right now*?

**The Problem**: We can easily calculate *average* change (rise over run), but how do we measure change at a single instant? There's no "run" at a point—it's just one location.

**The Brilliant Solution**: Get closer and closer to the instant. Make the time interval smaller and smaller. The derivative is what that average rate approaches as the interval shrinks to zero.

**Why the Limit Definition?**
```
f'(a) = lim(h�0) [f(a+h) - f(a)] / h
```
- **f(a+h) - f(a)**: Change in output (rise)
- **h**: Change in input (run)
- **Ratio**: Average rate of change
- **As h→0**: Average becomes instantaneous

**Visual Intuition**: Draw a curve. Put two points close together and connect them with a line (secant). Now move the second point closer... closer... closer. That secant line becomes the tangent line. Its slope is the derivative.

**Three Ways to Think About Derivatives**:
1. **Geometric**: Slope of the tangent line (best linear approximation)
2. **Physical**: Instantaneous rate of change (velocity from position)
3. **Algebraic**: Ratio of infinitesimal changes (dy/dx)

**Real-World Power**: The derivative lets us answer:
- How fast is the rocket accelerating *right now*?
- At what rate is the population growing *at this instant*?
- How sensitive is profit to a price change *at this price point*?

**The Magic**: Even though we can't divide by zero, limits let us see what "would happen" if we could. That's the derivative—the impossible made possible.

### Definition

The **derivative** of f(x) at x = a is:

```
f'(a) = lim(h�0) [f(a+h) - f(a)] / h
```

Alternative form:
```
f'(a) = lim(x�a) [f(x) - f(a)] / (x - a)
```

### Interpretations

**Geometric**: The derivative represents the slope of the tangent line to the curve at a point.

**Physical**: The derivative represents the instantaneous rate of change.
- If s(t) is position, then s'(t) is velocity
- If v(t) is velocity, then v'(t) is acceleration

### Notation

Multiple notations for derivatives:
- **Lagrange**: f'(x), f''(x), f'''(x), f}~(x)
- **Leibniz**: dy/dx, d�y/dx�, dy/dx
- **Newton**: �, � (for time derivatives)
- **Euler**: D_x f, D�_x f

**Why So Many Notations?**
- **Lagrange's f'(x)**: Compact, emphasizes function
- **Leibniz's dy/dx**: Shows it's a ratio of changes, makes chain rule intuitive, great for manipulation
- **Newton's ẋ**: Perfect for physics where time is the variable
- **Euler's D_x**: Emphasizes the operator view (differentiation is an operation)

Each notation highlights a different aspect. Leibniz notation (dy/dx) is especially powerful because it reminds us that derivatives are ratios—even though dy and dx aren't real numbers, they behave algebraically like fractions in many contexts.

### Basic Derivative Rules

1. **Constant Rule**: d/dx[c] = 0
   - *Intuition*: Constants don't change. Derivative measures change, so zero change means zero derivative.

2. **Power Rule**: d/dx[x^n] = n�x^(n-1)
   - *Intuition*: The power comes down as a multiplier, and the degree drops by one. Why? When you increase x slightly, x^n grows proportionally to n times the previous value. This is the pattern of exponential-like growth encoded in powers.

3. **Constant Multiple**: d/dx[c�f(x)] = c�f'(x)
   - *Intuition*: Scaling doesn't change the rate pattern, just its magnitude. If f doubles, c·f doubles—same rate, scaled up.

4. **Sum Rule**: d/dx[f(x) + g(x)] = f'(x) + g'(x)
   - *Intuition*: Changes add. If position is f+g, then velocity is f'+g'. Independent contributions to change sum linearly.

5. **Difference Rule**: d/dx[f(x) - g(x)] = f'(x) - g'(x)
   - *Intuition*: Same as sum rule, but subtracting. The rate of change of a difference is the difference of rates.

### Higher-Order Derivatives

- **First derivative**: f'(x) or dy/dx - rate of change
- **Second derivative**: f''(x) or d�y/dx� - rate of change of rate of change (concavity)
- **Third derivative**: f'''(x) or d�y/dx� - jerk (in physics)
- **nth derivative**: f}~(x) or dy/dx

**Intuition for Higher Derivatives**:
- **First derivative (f')**: The speedometer—how fast you're going
- **Second derivative (f'')**: The accelerometer—how fast your speed is changing
- **Third derivative (f''')**: The "jerk meter"—how fast your acceleration is changing (why sudden braking feels jarring)

**Why Second Derivatives Matter**: They measure the *curvature* of change:
- f' tells you the slope
- f'' tells you if the slope is increasing or decreasing
- This reveals the shape of the curve

**Concavity**:
- f''(x) > 0 � concave up (curve opens upward) - "holds water" - smiling face ∪
  *Meaning*: Slope is increasing. The function is accelerating upward.

- f''(x) < 0 � concave down (curve opens downward) - "spills water" - frowning face ∩
  *Meaning*: Slope is decreasing. The function is accelerating downward.

- f''(x) = 0 � possible inflection point
  *Meaning*: The curvature changes. Like the middle of an S-curve where the turn reverses.

**Physical Intuition**:
- Position → Velocity → Acceleration
- Cost → Marginal Cost → Rate of change of marginal cost
- Each derivative is "one level deeper" into understanding change

---

## Differentiation Techniques

### Product Rule

If u and v are differentiable functions:
```
d/dx[u�v] = u'�v + u�v'
```

**Intuition**: When two things multiply and both are changing, you get contributions from each:
- **u'·v**: Change in u, holding v constant
- **u·v'**: Change in v, holding u constant

Think of area of a rectangle with changing width u and height v. The area changes in two ways: width changes (u' times v), and height changes (u times v'). Both contribute to how the total area changes.

**Memory trick**: "First times derivative of second, plus second times derivative of first"

**Example**: d/dx[x��sin(x)] = 2x�sin(x) + x��cos(x)

### Quotient Rule

```
d/dx[u/v] = (u'�v - u�v') / v�
```

**Intuition**: A fraction changes when:
- **Numerator increases**: Fraction goes up → positive contribution (u'·v)
- **Denominator increases**: Fraction goes down → negative contribution (-u·v')
- **Divide by v²**: Normalize by the square of denominator

**Why the minus sign?** When the bottom gets bigger, the fraction gets smaller. That's the opposite (negative) effect.

**Memory trick**: "Low dee-high minus high dee-low, over the square of what's below"
- Low (v) × derivative of high (u')
- Minus high (u) × derivative of low (v')
- Over low squared (v²)

**Pro tip**: Often easier to rewrite as u·v⁻¹ and use product rule + chain rule!

**Example**: d/dx[sin(x)/x] = [x�cos(x) - sin(x)] / x�

### Chain Rule

For composite functions f(g(x)):
```
d/dx[f(g(x))] = f'(g(x))�g'(x)
```

Or in Leibniz notation:
```
dy/dx = (dy/du)�(du/dx)
```

**Intuition: Nested Change**

The chain rule captures how change propagates through nested functions. It's the mathematical expression of cause-and-effect chains.

**The Principle**: If A affects B, and B affects C, then A's effect on C is the product of:
- How much B changes when A changes (inner derivative)
- How much C changes when B changes (outer derivative)

**Why Multiply?** Changes compound multiplicatively through composition:
- If x changes by small amount dx
- Then g(x) changes by approximately g'(x)·dx
- Then f(g(x)) changes by approximately f'(g(x))·[g'(x)·dx]
- So the total rate is f'(g(x))·g'(x)

**Leibniz notation magic**: dy/dx = (dy/du)·(du/dx) looks like fractions canceling! While not rigorous, it's a powerful mnemonic and often works algebraically.

**Visual**: Imagine zooming through nested magnifications. Each layer magnifies by its derivative. Total magnification is the product of all layers.

**Real-World Example**:
- Distance depends on time: d = f(t)
- Time depends on temperature: t = g(T)
- How does distance change with temperature? dd/dT = (dd/dt)·(dt/dT)
- Chain rule connects indirect relationships!

**Example**: d/dx[sin(x�)] = cos(x�)�2x = 2x�cos(x�)
- Outer function: sin(u) → derivative is cos(u)
- Inner function: u = x² → derivative is 2x
- Evaluate outer derivative at inner function: cos(x²)
- Multiply by inner derivative: cos(x²)·2x

### Implicit Differentiation

When a relation is given implicitly (not solved for y):

**Steps**:
1. Differentiate both sides with respect to x
2. Apply chain rule to terms with y (multiply by dy/dx)
3. Solve for dy/dx

**Intuition**: Sometimes you can't (or don't want to) solve for y explicitly. No problem! Differentiate the relationship itself.

**Key Insight**: y is a function of x, even if we haven't written y = f(x). So when differentiating y terms, use the chain rule—y's derivative with respect to x is dy/dx (which we're solving for).

**Why It Works**: The equation defines a relationship. Differentiation preserves that relationship. Both sides must change at the same rate to maintain the equation.

**Mental Model**: Think of x and y as linked by a constraint. When x changes, y must change in a specific way to keep the constraint satisfied. Implicit differentiation finds that required rate.

**Example**: x� + y� = 25 (circle equation)
```
2x + 2y�(dy/dx) = 0
dy/dx = -x/y
```
*Interpretation*: At any point on the circle, the slope is -x/y. This is the tangent to the circle!

### Logarithmic Differentiation

Useful for products, quotients, and powers of functions:

**Steps**:
1. Take ln of both sides
2. Use logarithm properties to simplify
3. Differentiate implicitly
4. Solve for dy/dx

**Intuition**: Logarithms convert multiplication to addition, division to subtraction, and powers to multiplication. This transforms messy products/quotients/powers into simple sums/differences.

**Why Take ln?** Logarithms are the perfect tool for:
- **Products**: ln(ab) = ln(a) + ln(b) → sum rule instead of product rule
- **Quotients**: ln(a/b) = ln(a) - ln(b) → difference instead of quotient rule
- **Powers**: ln(a^b) = b·ln(a) → brings exponents down as multipliers

**When to Use**:
- Variable in both base and exponent (x^x)
- Complicated products of many functions
- Complicated quotients
- Functions raised to function powers

**The Magic**: ln converts complex derivative rules into simple arithmetic!

**Example**: y = x^x (variable base and exponent!)
```
ln(y) = x�ln(x)
(1/y)�(dy/dx) = ln(x) + 1
dy/dx = y�(ln(x) + 1) = x^x�(ln(x) + 1)
```
*Why it works*: Without ln, we'd struggle with x^x (power rule needs constant exponent, exponential rule needs constant base). Logarithm untangles it!

### Parametric Differentiation

For curves defined parametrically: x = f(t), y = g(t)
```
dy/dx = (dy/dt) / (dx/dt)
```

**Second derivative**:
```
d�y/dx� = d/dx[dy/dx] = [d/dt(dy/dx)] / (dx/dt)
```

### Common Derivatives

**Trigonometric Functions**:
- d/dx[sin(x)] = cos(x)
- d/dx[cos(x)] = -sin(x)
- d/dx[tan(x)] = sec�(x)
- d/dx[cot(x)] = -csc�(x)
- d/dx[sec(x)] = sec(x)�tan(x)
- d/dx[csc(x)] = -csc(x)�cot(x)

**Inverse Trigonometric Functions**:
- d/dx[arcsin(x)] = 1/(1-x�)
- d/dx[arccos(x)] = -1/(1-x�)
- d/dx[arctan(x)] = 1/(1+x�)

**Exponential and Logarithmic Functions**:
- d/dx[e^x] = e^x
- d/dx[a^x] = a^x�ln(a)
- d/dx[ln(x)] = 1/x
- d/dx[log_a(x)] = 1/(x�ln(a))

**Hyperbolic Functions**:
- d/dx[sinh(x)] = cosh(x)
- d/dx[cosh(x)] = sinh(x)
- d/dx[tanh(x)] = sech�(x)

---

## Applications of Derivatives

### Critical Points and Extrema

**Critical Point**: x = c where f'(c) = 0 or f'(c) does not exist

**Intuition: Finding the Best**

**Why Derivative = 0?** At a peak or valley, the slope is horizontal (neither going up nor down). That's where f'(x) = 0. It's a moment of transition—the function stops increasing and starts decreasing (or vice versa).

**The Physical Picture**:
- Imagine hiking on a mountain path
- At the top of a hill: you stop going up and start going down → slope = 0 → local max
- At the bottom of a valley: you stop going down and start going up → slope = 0 → local min
- Critical points are potential peaks and valleys

**Why Also Check Where f' Doesn't Exist?** Sharp corners and cusps can be extrema even without f' = 0. Think of a spike—it's a maximum even though there's no horizontal tangent.

**Finding Extrema**:
1. Find all critical points
2. Use First Derivative Test or Second Derivative Test
3. Check endpoints (for closed intervals)

**First Derivative Test** (Sign Analysis):
- If f' changes from + to - at c, then f has a local maximum at c
  *Intuition*: Function rises then falls → peak
- If f' changes from - to + at c, then f has a local minimum at c
  *Intuition*: Function falls then rises → valley

**Second Derivative Test** (Concavity):
- If f'(c) = 0 and f''(c) > 0, then f has a local minimum at c
  *Intuition*: Concave up (∪ shape) + horizontal tangent → bottom of bowl
- If f'(c) = 0 and f''(c) < 0, then f has a local maximum at c
  *Intuition*: Concave down (∩ shape) + horizontal tangent → top of dome
- If f''(c) = 0, test is inconclusive
  *Intuition*: Could be inflection point, not extremum

### Optimization Problems

**Intuition: Finding the Best in Real Life**

Optimization is about making the best choice given constraints. Maximum profit, minimum cost, shortest distance, largest area—these are all optimization problems.

**The Key Insight**: "Best" happens where you can't improve by making small changes. That's exactly where the derivative is zero—tiny changes don't help (first-order improvement is zero).

**Real-World Examples**:
- Farmer: What dimensions maximize area with fixed fence length?
- Company: What price maximizes profit?
- Engineer: What design minimizes material while meeting strength requirements?

**Why Constraints Matter**: They reduce freedom. With constraints, you can eliminate variables and reduce to a one-variable optimization problem that calculus can solve.

**General Strategy**:
1. Identify the quantity to optimize (write as a function)
2. Identify constraints
3. Use constraints to express the quantity as a function of one variable
4. Find critical points
5. Determine which critical point gives the optimal value

**Pro Tip**: Always check endpoints and boundaries. Sometimes the best solution is at an extreme constraint, not at a critical point.

### Related Rates

For quantities that change with respect to time:

**Intuition: Everything is Connected**

Related rates problems capture how changes in one quantity affect another when they're linked by a relationship. It's the mathematics of interconnected change.

**The Core Idea**: If two variables are related by an equation, their rates of change are also related. Differentiate the relationship to find how rates connect.

**Why "Related"?** When x and y satisfy an equation, they're not independent. As x changes, y must change in a compatible way. Their rates of change (dx/dt and dy/dt) are thus related through the same equation structure.

**Real-World Examples**:
- Balloon inflating: radius grows → volume grows (but at what rate?)
- Shadow lengthening: person walks → shadow extends (how fast?)
- Water draining: height drops → volume drops (connection?)
- Ladder sliding: bottom slides out → top slides down (how are these rates related?)

**The Process**:
1. Identify the relationship between variables (geometric or physical)
2. Differentiate the entire relationship with respect to time
3. The result links the rates of change

**Strategy**:
1. Draw a diagram and label variables
2. Write an equation relating the variables
3. Differentiate both sides with respect to time t
4. Substitute known values
5. Solve for the desired rate

**Example**: A ladder sliding down a wall
```
x� + y� = L�
2x�(dx/dt) + 2y�(dy/dt) = 0
```
*Interpretation*: As bottom moves out (dx/dt), top must move down (dy/dt) to maintain constant ladder length L. The rates are inversely related through the geometry.

### Mean Value Theorem (MVT)

If f is continuous on [a,b] and differentiable on (a,b), then there exists c in (a,b) such that:
```
f'(c) = [f(b) - f(a)] / (b - a)
```

**Interpretation**: There exists a point where the instantaneous rate equals the average rate.

### Linear Approximation

The tangent line approximation at x = a:
```
L(x) = f(a) + f'(a)�(x - a)
```

For small �x:
```
f(a + �x) H f(a) + f'(a)��x
```

**Differentials**:
- dx = �x (change in x)
- dy = f'(x)�dx (change in tangent line)
- �y = f(x + dx) - f(x) (actual change in f)

### L'H�pital's Rule

For indeterminate forms 0/0 or /:
```
lim(x�a) [f(x)/g(x)] = lim(x�a) [f'(x)/g'(x)]
```

Can be applied repeatedly if result is still indeterminate.

**Other indeterminate forms** (0�, -, 0p, 1^, p) can be converted to 0/0 or / form.

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

### Intuition: Accumulation and Reverse Engineering

**The Big Picture**: Integration is about accumulation—adding up infinitely many infinitesimally small pieces. It's also the reverse of differentiation.

**Two Perspectives on Integration**:

1. **Geometric (Area/Accumulation)**:
   - Slice a region into infinitely thin rectangles
   - Add up their areas: height f(x) times width dx
   - As rectangles get infinitesimally thin, sum becomes integral
   - Result: area under curve

2. **Algebraic (Antiderivative)**:
   - Derivative breaks things apart (rate of change)
   - Integral builds things back up (accumulation from rate)
   - If F'(x) = f(x), then ∫f(x)dx = F(x) + C
   - Integration "undoes" differentiation

**Why Integration Matters**: Whenever you know a rate and want the total:
- Know velocity → find displacement
- Know flow rate → find total volume
- Know marginal cost → find total cost
- Know rate of growth → find population

**The Fundamental Question**: Given how fast something is changing (derivative), what is the thing itself (original function)?

**Why the dx?** It's not just notation—it represents an infinitesimal width. The integral is literally a sum: ∫f(x)dx = "sum of f(x) times infinitesimal dx pieces". Think of it as lim(Δx→0) Σf(x)Δx.

**The "+ C" Mystery**: When you differentiate, constants vanish (derivative of constant = 0). So when you integrate (reverse), you can't know what constant was there. Could be any C!

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
1. +[a to b] c�f(x) dx = c�+[a to b] f(x) dx
2. +[a to b] [f(x) � g(x)] dx = +[a to b] f(x) dx � +[a to b] g(x) dx
3. +[a to b] f(x) dx = -+[b to a] f(x) dx
4. +[a to a] f(x) dx = 0
5. +[a to b] f(x) dx + +[b to c] f(x) dx = +[a to c] f(x) dx

### Fundamental Theorem of Calculus

**The Most Important Theorem in Calculus**

This theorem is the bridge connecting derivatives and integrals—two concepts that seem completely different but are actually inverse operations.

**Part 1**: If f is continuous on [a,b] and F(x) = +[a to x] f(t) dt, then F'(x) = f(x).

**Intuition for Part 1**:
- F(x) = accumulated area from a to x
- When you increase x slightly to x + dx, you add a thin rectangle of area ≈ f(x)·dx
- Rate of change of accumulated area = height of function
- **Profound Insight**: Accumulating f gives you something whose rate of change is f. Integration and differentiation are inverses!

**Analogy**: If f(t) is your speedometer reading and F(x) is your odometer, then:
- Odometer accumulates distance: F(x) = ∫ speed
- Speedometer is rate of distance change: f(x) = F'(x)
- They're inverses of each other!

**Part 2**: If f is continuous on [a,b] and F is any antiderivative of f, then:
```
+[a to b] f(x) dx = F(b) - F(a)
```

**Intuition for Part 2**:
- Want to find area under curve from a to b
- Instead of summing infinitely many rectangles (hard!)
- Just find ANY function F whose derivative is f
- Evaluate F at endpoints and subtract: F(b) - F(a)
- **This is miraculous**: Infinite sum reduced to two function evaluations!

**Why It Works**:
- F tracks cumulative change
- F(b) = total accumulated from start to b
- F(a) = total accumulated from start to a
- F(b) - F(a) = accumulated from a to b
- That's exactly the integral!

**The Power**: This theorem transforms an infinitely complex problem (summing infinite pieces) into simple algebra (evaluate, subtract). It's why calculus is so powerful!

**Historical Note**: Newton and Leibniz's great insight wasn't derivatives or integrals separately—many knew about those. The breakthrough was realizing they're inverses (this theorem). That unified calculus and unlocked its power.

### Basic Integration Formulas

1. + k dx = kx + C
2. + x^n dx = x^(n+1)/(n+1) + C (n ` -1)
3. + (1/x) dx = ln|x| + C
4. + e^x dx = e^x + C
5. + a^x dx = a^x/ln(a) + C
6. + sin(x) dx = -cos(x) + C
7. + cos(x) dx = sin(x) + C
8. + sec�(x) dx = tan(x) + C
9. + csc�(x) dx = -cot(x) + C
10. + sec(x)tan(x) dx = sec(x) + C
11. + csc(x)cot(x) dx = -csc(x) + C
12. + 1/(1-x�) dx = arcsin(x) + C
13. + 1/(1+x�) dx = arctan(x) + C

### Riemann Sums

The definite integral is the limit of Riemann sums:
```
+[a to b] f(x) dx = lim(n�) �[i=1 to n] f(x_i*)��x
```

where �x = (b-a)/n and x_i* is a sample point in the ith subinterval.

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

**Intuition: Reverse Chain Rule**

u-substitution is the integration version of the chain rule. It recognizes that your integrand came from a chain rule differentiation, and "undoes" it.

**The Key Insight**: If you see f(g(x))·g'(x), this came from differentiating F(g(x)) via chain rule:
- d/dx[F(g(x))] = F'(g(x))·g'(x) = f(g(x))·g'(x)
- So ∫f(g(x))·g'(x)dx = F(g(x)) + C

**When to Use**: Look for:
- A composite function f(g(x))
- Whose "inside function's" derivative g'(x) appears as a factor
- Pattern: ∫[stuff]'·[function of stuff] → substitute u = stuff

**Why It Works**: The du = g'(x)dx substitution absorbs the chain rule's g'(x) term, reducing the composite function to a simple function of u.

**Mental Model**: You're "peeling off" the outer layer of composition. The integral becomes simpler in terms of the inner function.

**The Art**: Choosing the right u. Look for the "inner function" whose derivative (or a multiple) appears elsewhere in the integrand.

**Steps**:
1. Choose substitution u = g(x)
2. Calculate du = g'(x)dx
3. Rewrite integral in terms of u
4. Integrate with respect to u
5. Substitute back to get result in terms of x

**Example**:
```
+ 2x�cos(x�) dx
Let u = x�, du = 2x dx
= + cos(u) du
= sin(u) + C
= sin(x�) + C
```

**For definite integrals**, also change the limits:
- If u = g(x), new limits are u = g(a) and u = g(b)

### Integration by Parts

**Formula**:
```
+ u dv = uv - + v du
```

**Intuition: Reverse Product Rule**

Integration by parts is the integration version of the product rule. It trades one integral for another (hopefully simpler) integral.

**The Core Idea**:
- Product rule: (uv)' = u'v + uv'
- Rearrange: uv' = (uv)' - u'v
- Integrate both sides: ∫u(dv/dx)dx = uv - ∫v(du/dx)dx
- Or simply: ∫u dv = uv - ∫v du

**When to Use**: When integrand is a product of two different "types" of functions (polynomial × exponential, polynomial × trig, etc.)

**The Strategy**: Split the integrand into two parts:
- **u**: The part that gets simpler when differentiated
- **dv**: The part you can easily integrate

**Why LIATE?** This priority list ensures u gets simpler when you differentiate:
- **L**ogarithmic → derivative is algebraic (simpler!)
- **I**nverse trig → derivative is algebraic (simpler!)
- **A**lgebraic → derivative reduces power (simpler!)
- **T**rigonometric → derivative stays trig (no simpler)
- **E**xponential → derivative stays exponential (no simpler)

**The Trade-Off**: You're converting ∫u dv into uv - ∫v du. The goal is making the new integral ∫v du easier than the original.

**Mental Model**: You're "sacrificing" one factor (u) by differentiating it (hopefully simplifying it) while integrating the other (dv), then dealing with the resulting integral.

**Pro Tip**: Sometimes you need to apply integration by parts multiple times, or even in a cycle that allows you to solve for the original integral algebraically!

**Choosing u and dv (LIATE rule)**:
- **L**ogarithmic
- **I**nverse trigonometric
- **A**lgebraic
- **T**rigonometric
- **E**xponential

Choose u in this order of preference; dv is what remains.

**Example**:
```
+ x�e^x dx
u = x, dv = e^x dx
du = dx, v = e^x
= x�e^x - + e^x dx
= x�e^x - e^x + C
= e^x(x - 1) + C
```

**Tabular Integration**: Efficient for repeated integration by parts.

### Trigonometric Integrals

**Strategies for + sin^m(x)cos^n(x) dx**:

1. **If n is odd**: Save one cos(x), convert rest to sin(x) using cos�(x) = 1 - sin�(x), then substitute u = sin(x)
2. **If m is odd**: Save one sin(x), convert rest to cos(x) using sin�(x) = 1 - cos�(x), then substitute u = cos(x)
3. **If both are even**: Use power-reducing formulas
   - sin�(x) = (1 - cos(2x))/2
   - cos�(x) = (1 + cos(2x))/2

**Powers of tan and sec**:
- + tan^m(x)sec^n(x) dx
- Use tan�(x) = sec�(x) - 1 and sec�(x) derivative of tan(x)

### Trigonometric Substitution

For integrals involving (a� - x�), (a� + x�), or (x� - a�):

1. **(a� - x�)**: Let x = a�sin(�), dx = a�cos(�)d�
   - (a� - x�) = a�cos(�)

2. **(a� + x�)**: Let x = a�tan(�), dx = a�sec�(�)d�
   - (a� + x�) = a�sec(�)

3. **(x� - a�)**: Let x = a�sec(�), dx = a�sec(�)tan(�)d�
   - (x� - a�) = a�tan(�)

**Example**:
```
+ (1 - x�) dx
Let x = sin(�), dx = cos(�)d�
= + cos(�)�cos(�) d�
= + cos�(�) d�
= + (1 + cos(2�))/2 d�
= �/2 + sin(2�)/4 + C
= arcsin(x)/2 + x(1-x�)/2 + C
```

### Partial Fractions

For rational functions P(x)/Q(x) where degree(P) < degree(Q):

**Steps**:
1. Factor the denominator Q(x)
2. Decompose into partial fractions
3. Solve for coefficients (equate coefficients or plug in values)
4. Integrate each term

**Forms**:
1. **Linear factors**: (x - a) � A/(x - a)
2. **Repeated linear**: (x - a)^n � A�/(x-a) + A�/(x-a)� + ... + A�/(x-a)^n
3. **Quadratic factors**: (x� + bx + c) � (Ax + B)/(x� + bx + c)
4. **Repeated quadratic**: Similar to repeated linear

**Example**:
```
+ 1/(x� - 1) dx = + 1/[(x-1)(x+1)] dx
1/(x� - 1) = A/(x-1) + B/(x+1)
1 = A(x+1) + B(x-1)
Solving: A = 1/2, B = -1/2
= (1/2)+ 1/(x-1) dx - (1/2)+ 1/(x+1) dx
= (1/2)ln|x-1| - (1/2)ln|x+1| + C
= (1/2)ln|(x-1)/(x+1)| + C
```

### Improper Integrals

**Type 1**: Infinite interval
```
+[a to ] f(x) dx = lim(t�) +[a to t] f(x) dx
```

**Type 2**: Discontinuous integrand
```
+[a to b] f(x) dx = lim(t�b{) +[a to t] f(x) dx  (if f is discontinuous at b)
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
V = ��+[a to b] [f(x)]� dx
```

**Washer Method** (hollow solid):
```
V = ��+[a to b] [R(x)]� - [r(x)]� dx
```
where R(x) is outer radius, r(x) is inner radius

**Shell Method** (cylindrical shells):
```
V = 2��+[a to b] x�f(x) dx
```
or
```
V = 2��+[c to d] y�g(y) dy
```

**Cross-Sectional Method**:
```
V = +[a to b] A(x) dx
```
where A(x) is the area of cross-section at x

### Arc Length

**For y = f(x)** on [a,b]:
```
L = +[a to b] (1 + [f'(x)]�) dx
```

**For parametric curves** x = f(t), y = g(t) on [�,�]:
```
L = +[� to �] ([dx/dt]� + [dy/dt]�) dt
```

**For polar curves** r = f(�):
```
L = +[� to �] (r� + [dr/d�]�) d�
```

### Surface Area

**Revolution around x-axis**:
```
S = 2��+[a to b] f(x)�(1 + [f'(x)]�) dx
```

**Revolution around y-axis**:
```
S = 2��+[a to b] x�(1 + [f'(x)]�) dx
```

### Work

**Constant force**: W = F�d

**Variable force**:
```
W = +[a to b] F(x) dx
```

**Examples**:
- **Spring**: W = + kx dx = (1/2)kx� (Hooke's Law)
- **Lifting liquid**: W = + ��g�A(y)�y dy
- **Pumping**: Account for distance each layer must be moved

### Center of Mass

**For a thin plate** (lamina) with density �(x,y):

**Mass**:
```
m = ++_R �(x,y) dA
```

**Moments**:
```
M_x = ++_R y��(x,y) dA
M_y = ++_R x��(x,y) dA
```

**Center of mass**:
```
x = M_y / m
3 = M_x / m
```

**For uniform density** (� = constant), center of mass = centroid.

---

## Sequences and Series

### Intuition: The Mathematics of Infinity

**The Fundamental Questions**:
1. **Sequences**: Where is this infinite list heading?
2. **Series**: Can we add infinitely many numbers and get a finite answer?

These questions connect discrete (countable steps) with continuous (limits), and finite with infinite.

### Sequences

A **sequence** is an ordered list: {a�, a�, a�, ...} or {a�}

**Intuition**: A sequence is a pattern that continues forever. Convergence asks: "Does this pattern settle down to a specific value, or does it keep wandering?"

**Examples**:
- {1, 1/2, 1/3, 1/4, ...} → converges to 0 (gets arbitrarily close)
- {1, -1, 1, -1, ...} → diverges (oscillates forever)
- {1, 2, 3, 4, ...} → diverges (grows without bound)

**Convergence**: lim(n�) a� = L means the sequence converges to L.

**Properties**:
- **Monotonic**: Always increasing or always decreasing
- **Bounded**: |a�| d M for all n
- **Monotone Convergence Theorem**: A bounded, monotonic sequence converges

### Series

An **infinite series** is the sum of a sequence:
```
�[n=1 to ] a� = a� + a� + a� + ...
```

**Partial sums**: S� = �[k=1 to n] a�

**Convergence**: The series converges to S if lim(n�) S� = S.

### Geometric Series

```
�[n=0 to ] ar^n = a + ar + ar� + ar� + ...
```

**Convergence**:
- If |r| < 1, series converges to a/(1-r)
- If |r| e 1, series diverges

### Tests for Convergence

**nth-Term Test (Divergence Test)**:
- If lim(n�) a� ` 0, then �a� diverges
- If lim(n�) a� = 0, test is inconclusive

**Integral Test**:
If f is continuous, positive, decreasing for x e 1:
- �[n=1 to ] a� and +[1 to ] f(x) dx both converge or both diverge

**p-Series**:
```
�[n=1 to ] 1/n^p
```
Converges if p > 1, diverges if p d 1

**Comparison Test**:
If 0 d a� d b� for all n:
- If �b� converges, then �a� converges
- If �a� diverges, then �b� diverges

**Limit Comparison Test**:
If a�, b� > 0 and lim(n�) a�/b� = c > 0:
- Both series converge or both diverge

**Ratio Test**:
```
L = lim(n�) |a��� / a�|
```
- If L < 1, series converges absolutely
- If L > 1 (or L = ), series diverges
- If L = 1, test is inconclusive

**Root Test**:
```
L = lim(n�) |a�|
```
- If L < 1, series converges absolutely
- If L > 1 (or L = ), series diverges
- If L = 1, test is inconclusive

**Alternating Series Test**:
For alternating series �(-1)^n�b� where b� > 0:
- If b� is decreasing and lim(n�) b� = 0, series converges

### Absolute and Conditional Convergence

- **Absolutely convergent**: �|a�| converges
- **Conditionally convergent**: �a� converges but �|a�| diverges

If a series converges absolutely, it converges.

### Power Series

A **power series** centered at a:
```
�[n=0 to ] c�(x - a)^n
```

**Radius of Convergence (R)**:
- Series converges for |x - a| < R
- Series diverges for |x - a| > R
- At endpoints x = a � R, must test separately

**Finding R**:
```
R = lim(n�) |c� / c���|
```
or
```
1/R = lim(n�) |c��� / c�|
```

**Interval of Convergence**: (a - R, a + R) plus possibly the endpoints

### Taylor and Maclaurin Series

**Taylor Series** of f(x) centered at x = a:
```
f(x) = �[n=0 to ] [f}~(a) / n!]�(x - a)^n
     = f(a) + f'(a)(x-a) + [f''(a)/2!](x-a)� + [f'''(a)/3!](x-a)� + ...
```

**Maclaurin Series** (special case where a = 0):
```
f(x) = �[n=0 to ] [f}~(0) / n!]�x^n
```

**Common Maclaurin Series**:

1. e^x = �[n=0 to ] x^n/n! = 1 + x + x�/2! + x�/3! + ...

2. sin(x) = �[n=0 to ] (-1)^n�x^(2n+1)/(2n+1)! = x - x�/3! + xu/5! - ...

3. cos(x) = �[n=0 to ] (-1)^n�x^(2n)/(2n)! = 1 - x�/2! + xt/4! - ...

4. 1/(1-x) = �[n=0 to ] x^n = 1 + x + x� + x� + ... (|x| < 1)

5. ln(1+x) = �[n=1 to ] (-1)^(n+1)�x^n/n = x - x�/2 + x�/3 - ... (|x| < 1)

6. arctan(x) = �[n=0 to ] (-1)^n�x^(2n+1)/(2n+1) = x - x�/3 + xu/5 - ... (|x| d 1)

**Taylor's Remainder**:
```
R�(x) = f(x) - T�(x) = [f}z�~(c) / (n+1)!]�(x - a)^(n+1)
```
where c is between a and x.

---

## Multivariable Calculus

### Intuition: Calculus in Higher Dimensions

**The Big Picture**: Everything we learned for single-variable calculus extends to functions of multiple variables. But now we have richer geometry and more directions to consider.

**Key Difference**: With one variable, there's only one direction—left or right. With multiple variables, there are infinitely many directions. How does the function change in each direction?

**New Challenges**:
- Rate of change depends on direction
- Surfaces instead of curves
- Volumes instead of areas

**Core Concepts**:
- **Partial derivatives**: Rate of change along coordinate axes
- **Gradient**: The vector pointing toward steepest increase
- **Directional derivatives**: Rate of change in any direction
- **Multiple integrals**: Volume under surfaces, mass of 3D objects

### Partial Derivatives

For a function f(x,y):

**Intuition**: How does f change if I wiggle just ONE input variable, holding all others constant?

**Mental Model**: Imagine a mountain surface f(x,y) = height. Partial derivative ∂f/∂x is the slope if you walk in the pure x-direction (east-west). Partial derivative ∂f/∂y is the slope if you walk in the pure y-direction (north-south).

**Why "Partial"?** You're only looking at part of the story—change in one direction, ignoring others.

**Practical Meaning**:
- ∂Cost/∂Labor: How does cost change with more workers (holding materials constant)?
- ∂Temperature/∂x: How does temp change moving east (holding north-south position constant)?

**Partial derivative with respect to x**:
```
f/x = lim(h�0) [f(x+h, y) - f(x, y)] / h
```

**Notation**:
- f/x, f_x, _x f

**Computing**: Treat other variables as constants and differentiate normally.

**Example**: f(x,y) = x�y + y�
- f/x = 2xy
- f/y = x� + 3y�

**Higher-order partial derivatives**:
- f_xx = �f/x�
- f_yy = �f/y�
- f_xy = �f/xy (mixed partial)
- f_yx = �f/yx (mixed partial)

**Clairaut's Theorem**: If f_xy and f_yx are continuous, then f_xy = f_yx.

### Gradient

The **gradient** of f is a vector of partial derivatives:
```
f = <f/x, f/y, f/z> = f_x�i + f_y�j + f_z�k
```

**Intuition: The Direction of Steepest Ascent**

The gradient is the most important concept in multivariable calculus. It's a vector that answers: "Which way should I go to increase f the fastest?"

**Mountain Analogy**:
- Standing on a mountain, gradient points uphill in the steepest direction
- Magnitude of gradient = how steep that direction is
- Negative gradient points downhill (steepest descent)
- This is why gradient descent in machine learning works—it finds minimums!

**Why a Vector?** In multiple dimensions, "direction" needs multiple components. The gradient packs all directional information into one vector.

**Properties**:
- **Points in direction of maximum rate of increase**
  *Why?* It's constructed from rates in all coordinate directions, combines them optimally

- **Perpendicular to level curves/surfaces**
  *Why?* Along a level curve, f doesn't change (tangent to curve means no change). Gradient points where change is maximal, which is perpendicular.

- **Magnitude is the maximum rate of change**
  *Why?* |∇f| is how much f increases per unit distance in the optimal direction

**Applications**:
- Optimization: Follow gradient to find maxima
- Physics: Force = -∇(potential energy)
- Machine Learning: Gradient descent for training neural networks
- Computer Graphics: Surface normals for lighting

### Directional Derivatives

The **directional derivative** of f at point P in direction of unit vector u:
```
D_u f = f � u
```

**Maximum rate of change** occurs in direction of f with magnitude |f|.

### Chain Rule (Multivariable)

**Case 1**: z = f(x,y), x = g(t), y = h(t)
```
dz/dt = (z/x)�(dx/dt) + (z/y)�(dy/dt)
```

**Case 2**: z = f(x,y), x = g(s,t), y = h(s,t)
```
z/s = (z/x)�(x/s) + (z/y)�(y/s)
z/t = (z/x)�(x/t) + (z/y)�(y/t)
```

### Extrema of Multivariable Functions

**Critical points**: Where f = 0 or f does not exist

**Second Derivative Test**: At critical point (a,b):
```
D = f_xx(a,b)�f_yy(a,b) - [f_xy(a,b)]�
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

**Fubini's Theorem**: If R = [a,b] � [c,d]:
```
++_R f(x,y) dA = +[a to b] +[c to d] f(x,y) dy dx
                = +[c to d] +[a to b] f(x,y) dx dy
```

**Applications**:
- Volume under surface: V = ++_R f(x,y) dA
- Area of region: A = ++_R 1 dA
- Mass: m = ++_R �(x,y) dA

**Triple Integral**:
```
+++_E f(x,y,z) dV
```

### Coordinate Systems

**Polar Coordinates** (x = r�cos(�), y = r�sin(�)):
```
++_R f(x,y) dA = ++ f(r�cos(�), r�sin(�))�r dr d�
```

**Cylindrical Coordinates** (x = r�cos(�), y = r�sin(�), z = z):
```
+++_E f(x,y,z) dV = +++ f(r�cos(�), r�sin(�), z)�r dz dr d�
```

**Spherical Coordinates** (x = ��sin(�)�cos(�), y = ��sin(�)�sin(�), z = ��cos(�)):
```
+++_E f(x,y,z) dV = +++ f(�,�,�)����sin(�) d� d� d�
```

### Vector Calculus

**Line Integrals**:
```
+_C f(x,y) ds = +[a to b] f(r(t))�|r'(t)| dt
+_C F � dr = +[a to b] F(r(t)) � r'(t) dt
```

**Green's Theorem** (relates line integral to double integral):
```
._C P dx + Q dy = ++_D (Q/x - P/y) dA
```

**Conservative Vector Fields**:
- F = f for some scalar function f (potential function)
- Line integral is path-independent
- ._C F � dr = 0 for any closed curve C

**Test**: F = <P, Q> is conservative if P/y = Q/x

---

## Differential Equations

### Intuition: Equations of Change

**The Paradigm Shift**: Normal equations tell you WHAT something is. Differential equations tell you HOW it CHANGES. The solution is a function, not a number.

**The Core Idea**: Many real-world phenomena are easier to describe in terms of rates of change rather than explicit formulas:
- Population grows proportionally to current population: dP/dt = kP
- Temperature approaches ambient temp: dT/dt = -k(T - T_ambient)
- Velocity changes due to forces: ma = F (Newton's 2nd law)

**Why They're Powerful**: Most natural laws are differential equations. Newton's laws, Maxwell's equations, Schrödinger equation—all DEs. Nature speaks the language of rates of change.

**The Challenge**: Given a rule for how something changes, find what it actually IS. This is harder than it sounds—you're essentially "integrating" but with more complex relationships.

**Types of Solutions**:
- **General solution**: Contains arbitrary constants (family of functions)
- **Particular solution**: Specific function satisfying initial conditions
- **Explicit vs Implicit**: Sometimes we can't solve for y explicitly

**Mental Model**: Imagine a vector field showing velocities at each point. A solution curve follows those velocity vectors. The differential equation defines the field; you find the curves.

**Real-World Applications**:
- Physics: Motion, heat, waves, quantum mechanics
- Biology: Population dynamics, disease spread, neural activity
- Economics: Growth models, market dynamics
- Engineering: Control systems, circuits, fluid flow

### First-Order ODEs

**General form**: dy/dx = f(x,y) or M(x,y)dx + N(x,y)dy = 0

**Intuition**: First-order means only first derivatives (rate of change), no acceleration or higher rates. These are the simplest DEs and model basic change processes.

### Separable Equations

**Form**: dy/dx = g(x)�h(y)

**Method**:
1. Separate variables: [1/h(y)]dy = g(x)dx
2. Integrate both sides
3. Solve for y if possible

**Example**: dy/dx = xy
```
dy/y = x dx
ln|y| = x�/2 + C
y = Ae^(x�/2)
```

### Linear First-Order ODEs

**Standard form**: dy/dx + P(x)�y = Q(x)

**Method** (Integrating Factor):
1. Compute �(x) = e^(+P(x)dx)
2. Multiply equation by �(x)
3. Left side becomes d/dx[�(x)�y]
4. Integrate: �(x)�y = +�(x)�Q(x)dx
5. Solve for y

**Example**: dy/dx + y = e^x
```
�(x) = e^+1 dx = e^x
e^x�dy/dx + e^x�y = e^(2x)
d/dx[e^x�y] = e^(2x)
e^x�y = (1/2)e^(2x) + C
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

**Characteristic equation**: ar� + br + c = 0

**Solutions**:
1. **Two distinct real roots** r�, r�: y = C�e^(r�x) + C�e^(r�x)
2. **Repeated root** r: y = (C� + C�x)e^(rx)
3. **Complex roots** r = � � �i: y = e^(�x)[C�cos(�x) + C�sin(�x)]

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

**RC circuits**: RC�dV/dt + V = V_source

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
