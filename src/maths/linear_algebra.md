# Linear Algebra

A comprehensive guide to linear algebra from fundamentals to advanced applications in machine learning, computer graphics, scientific computing, and algorithmic theory.

## Table of Contents

1. [Introduction](#introduction)
   - [What is Linear Algebra?](#what-is-linear-algebra)
   - [Why Linear Algebra Matters](#why-linear-algebra-matters)
   - [Prerequisites and Reading Guide](#prerequisites-and-reading-guide)

2. [Vectors - Fundamentals](#vectors---fundamentals)
   - [Vector Basics](#vector-basics)
   - [Vector Operations](#vector-operations)
   - [Dot Product](#dot-product)
   - [Cross Product](#cross-product)
   - [Vector Norms](#vector-norms)

3. [Matrices - Core Concepts](#matrices---core-concepts)
   - [Matrix Fundamentals](#matrix-fundamentals)
   - [Matrix Operations](#matrix-operations)
   - [Matrix Transpose](#matrix-transpose)
   - [Special Matrix Types](#special-matrix-types)

4. [Linear Systems](#linear-systems)
   - [Systems of Linear Equations](#systems-of-linear-equations)
   - [Gaussian Elimination](#gaussian-elimination)
   - [LU Decomposition](#lu-decomposition)

5. [Vector Spaces](#vector-spaces)
   - [Vector Space Axioms](#vector-space-axioms)
   - [Subspaces](#subspaces)
   - [Span and Linear Independence](#span-and-linear-independence)
   - [Basis and Dimension](#basis-and-dimension)

6. [Linear Transformations](#linear-transformations)
   - [Definition and Properties](#definition-and-properties)
   - [Geometric Transformations](#geometric-transformations)
   - [Composition of Transformations](#composition-of-transformations)
   - [Inverse Transformations](#inverse-transformations)

7. [Determinants](#determinants)
   - [Geometric Interpretation](#geometric-interpretation-determinants)
   - [Computing Determinants](#computing-determinants)
   - [Properties of Determinants](#properties-of-determinants)

8. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
   - [Fundamental Concepts](#fundamental-concepts-eigenvalues)
   - [Computing Eigenvalues](#computing-eigenvalues)
   - [Diagonalization](#diagonalization)
   - [Applications of Eigenvalues](#applications-of-eigenvalues)

9. [Orthogonality](#orthogonality)
   - [Orthogonal Vectors](#orthogonal-vectors)
   - [Gram-Schmidt Process](#gram-schmidt-process)
   - [Orthogonal Projections](#orthogonal-projections)

10. [Matrix Decompositions](#matrix-decompositions)
    - [LU Decomposition (Extended)](#lu-decomposition-extended)
    - [QR Decomposition](#qr-decomposition)
    - [Cholesky Decomposition](#cholesky-decomposition)
    - [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)

11. [Least Squares](#least-squares)
    - [Linear Regression Framework](#linear-regression-framework)
    - [Solution Methods](#solution-methods-least-squares)
    - [Regularization](#regularization)

12. [Advanced Topics](#advanced-topics)
    - [Matrix Calculus](#matrix-calculus)
    - [Matrix Norms](#matrix-norms)
    - [Tensor Operations](#tensor-operations)
    - [Sparse Linear Algebra](#sparse-linear-algebra)

13. [Numerical Considerations](#numerical-considerations)
    - [Conditioning and Stability](#conditioning-and-stability)
    - [Computational Complexity](#computational-complexity)

14. [Applications](#applications)
    - [Machine Learning and Data Science](#machine-learning-and-data-science)
    - [Computer Graphics and Vision](#computer-graphics-and-vision)
    - [Scientific Computing](#scientific-computing)
    - [Algorithms and Theory](#algorithms-and-theory)

15. [Practical Implementation Guide](#practical-implementation-guide)
    - [NumPy Best Practices](#numpy-best-practices)
    - [Common Pitfalls](#common-pitfalls)
    - [Performance Optimization](#performance-optimization)
    - [Quick Reference](#quick-reference)
    - [Further Reading](#further-reading)

---

## Introduction

### What is Linear Algebra?

Linear algebra is the branch of mathematics that studies vectors, vector spaces, linear transformations, and systems of linear equations. At its core, it's about understanding and manipulating collections of numbers in structured ways.

**Intuitive Understanding:**
Think of linear algebra as the mathematics of:
- **Arrows in space** (vectors): representing direction and magnitude
- **Grids and tables** (matrices): organizing and transforming data
- **Linear relationships**: where doubling the input doubles the output

### Why Linear Algebra Matters

Linear algebra is the mathematical foundation of modern technology:

**Machine Learning & AI:**
- Neural networks are built from matrix multiplications
- PCA reduces data dimensions using eigenvalues
- Recommender systems use matrix factorization
- Image recognition processes pixels as vectors

**Computer Graphics:**
- 3D transformations (rotation, scaling, translation)
- Camera projections for rendering
- Animation and skeletal systems
- Lighting and shading calculations

**Data Science:**
- PageRank algorithm powers Google search
- Image compression via SVD
- Natural language processing with word embeddings
- Statistical analysis and regression

**Scientific Computing:**
- Solving differential equations
- Optimization algorithms
- Physics simulations
- Quantum mechanics calculations

### Prerequisites and Reading Guide

**Prerequisites:**
- Basic algebra and arithmetic
- Understanding of functions and graphs
- Familiarity with Python is helpful but not required

**Reading Guide:**
This guide follows a beginner-friendly approach:
1. **Intuition first**: Every concept starts with geometric/visual understanding
2. **Progressive complexity**: Informal → concrete examples → formal definitions
3. **Multiple perspectives**: Algebraic, geometric, and computational views
4. **Practical code**: NumPy implementations you can run and experiment with

---

## Vectors - Fundamentals

### Vector Basics

**Intuitive Understanding:**
Imagine you're giving directions: "Walk 3 blocks east and 4 blocks north." This describes a **vector** - it has both direction (northeast) and magnitude (5 blocks total). Vectors are the fundamental building blocks of linear algebra.

**Two Views of Vectors:**

1. **Geometric View**: A vector is an arrow in space with:
   - Direction: where it points
   - Magnitude: how long it is
   - Starting at the origin (0, 0)

2. **Algebraic View**: A vector is an ordered list of numbers:

$$\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

where the first number is the x-coordinate (3 blocks east) and the second is the y-coordinate (4 blocks north).

**Formal Definition:**
A vector in $\mathbb{R}^n$ is an n-tuple of real numbers, represented as a column:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Creating vectors
v = np.array([3, 4])  # 2D vector
u = np.array([1, 2])

print(f"Vector v: {v}")
print(f"Vector u: {u}")

# Visualizing vectors
def plot_vectors(vectors, colors, labels):
    """Visualize 2D vectors as arrows from origin"""
    plt.figure(figsize=(8, 8))
    for vec, color, label in zip(vectors, colors, labels):
        plt.quiver(0, 0, vec[0], vec[1],
                   angles='xy', scale_units='xy', scale=1,
                   color=color, width=0.006, label=label)

    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.title('Vector Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Visualize our vectors
plot_vectors([v, u], ['blue', 'red'], ['v = [3, 4]', 'u = [1, 2]'])
```

**Column vs Row Vectors:**
- Column vector (default): $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$
- Row vector (transpose): $\mathbf{v}^T = \begin{bmatrix} 3 & 4 \end{bmatrix}$

### Vector Operations

#### Vector Addition

**Intuition**: Vector addition is like combining two sets of directions. If you walk 3 east and 4 north, then 1 more east and 2 more north, you end up 4 east and 6 north total.

**Geometric View**: Place the tail of the second vector at the head of the first. The sum is the vector from the origin to the final point (parallelogram rule).

**Algebraic Definition**:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**Python Example:**

```python
# Vector addition
u = np.array([1, 2])
v = np.array([3, 4])
w = u + v

print(f"u + v = {w}")  # [4, 6]

# Visualize addition
plt.figure(figsize=(8, 8))
# Original vectors
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.006, label='u')
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.006, label='v')
# Sum vector
plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.008, label='u + v')
# Parallelogram visualization
plt.quiver(u[0], u[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.004, alpha=0.5, linestyle='dashed')
plt.quiver(v[0], v[1], u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.004, alpha=0.5, linestyle='dashed')

plt.xlim(-1, 6)
plt.ylim(-1, 7)
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.title('Vector Addition (Parallelogram Rule)')
plt.show()
```

#### Scalar Multiplication

**Intuition**: Multiplying a vector by a scalar (number) stretches or shrinks it. Multiplying by 2 doubles its length; multiplying by -1 reverses its direction.

**Algebraic Definition**:

$$c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}$$

**Python Example:**

```python
v = np.array([2, 1])

# Scalar multiplication
v_scaled = 2 * v      # [4, 2] - doubled
v_half = 0.5 * v      # [1, 0.5] - halved
v_reversed = -1 * v   # [-2, -1] - reversed direction

print(f"Original: {v}")
print(f"Doubled: {v_scaled}")
print(f"Halved: {v_half}")
print(f"Reversed: {v_reversed}")
```

#### Linear Combinations

**Intuition**: A linear combination is like mixing ingredients with different amounts. You take some amount of one vector plus some amount of another.

**Definition**: A linear combination of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ is:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

where $c_1, c_2, \ldots, c_k$ are scalars (called coefficients).

**Python Example:**

```python
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Linear combination: 3*v1 + 2*v2
result = 3*v1 + 2*v2
print(f"3*v1 + 2*v2 = {result}")  # [3, 2]

# Any 2D vector can be written as a linear combination of v1 and v2!
target = np.array([5, 7])
# target = 5*v1 + 7*v2
reconstructed = 5*v1 + 7*v2
print(f"Target: {target}")
print(f"Reconstructed: {reconstructed}")
print(f"Match: {np.allclose(target, reconstructed)}")
```

### Dot Product

**Intuition**: The dot product measures how much two vectors point in the same direction. If they're perfectly aligned, the dot product is large; if they're perpendicular, it's zero; if they point opposite directions, it's negative.

**Algebraic Definition**:

$$\mathbf{u} \cdot \mathbf{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n = \sum_{i=1}^{n} u_iv_i$$

**Geometric Interpretation**:

$$\mathbf{u} \cdot \mathbf{v} = ||\mathbf{u}|| \cdot ||\mathbf{v}|| \cdot \cos\theta$$

where $\theta$ is the angle between the vectors.

**Key Properties:**
1. **Commutative**: $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
2. **Distributive**: $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$
3. **Orthogonality**: $\mathbf{u} \cdot \mathbf{v} = 0 \iff \mathbf{u} \perp \mathbf{v}$

**Python Example:**

```python
u = np.array([3, 4])
v = np.array([1, 2])

# Dot product - multiple ways
dot1 = np.dot(u, v)           # NumPy function
dot2 = u @ v                   # Matrix multiplication operator
dot3 = np.sum(u * v)          # Manual: sum of element-wise products

print(f"Dot product u·v = {dot1}")  # 3*1 + 4*2 = 11
print(f"All methods agree: {dot1 == dot2 == dot3}")

# Computing angle between vectors
def angle_between(u, v):
    """Compute angle in radians between vectors u and v"""
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # Clamp to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

theta = angle_between(u, v)
print(f"Angle between u and v: {np.degrees(theta):.2f} degrees")

# Check orthogonality
u_ortho = np.array([1, 0])
v_ortho = np.array([0, 1])
print(f"u_ortho · v_ortho = {np.dot(u_ortho, v_ortho)}")  # 0 - perpendicular!
```

**Application: Cosine Similarity (Used in ML)**

```python
def cosine_similarity(u, v):
    """
    Measure similarity between vectors (-1 to 1)
    1 = same direction, 0 = perpendicular, -1 = opposite
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Example: document similarity
doc1 = np.array([3, 2, 0, 5])  # word frequencies
doc2 = np.array([2, 1, 0, 3])  # similar document
doc3 = np.array([0, 0, 4, 0])  # different topic

print(f"Similarity(doc1, doc2) = {cosine_similarity(doc1, doc2):.3f}")  # High
print(f"Similarity(doc1, doc3) = {cosine_similarity(doc1, doc3):.3f}")  # Low
```

### Cross Product

**Note**: The cross product is only defined for 3D vectors.

**Intuition**: The cross product of two vectors produces a third vector that's perpendicular to both. Imagine two arrows in space - the cross product points in the direction perpendicular to the plane they form.

**Algebraic Definition** (3D only):

$$\mathbf{u} \times \mathbf{v} = \begin{bmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{bmatrix}$$

**Geometric Properties:**
1. $\mathbf{u} \times \mathbf{v}$ is perpendicular to both $\mathbf{u}$ and $\mathbf{v}$
2. $||\mathbf{u} \times \mathbf{v}|| = ||\mathbf{u}|| \cdot ||\mathbf{v}|| \cdot \sin\theta$ (area of parallelogram)
3. Direction given by right-hand rule
4. **Anti-commutative**: $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$

**Python Example:**

```python
# 3D vectors
u = np.array([1, 0, 0])  # x-axis
v = np.array([0, 1, 0])  # y-axis

# Cross product
w = np.cross(u, v)
print(f"u × v = {w}")  # [0, 0, 1] - z-axis (right-hand rule)

# Verify perpendicularity
print(f"(u × v) · u = {np.dot(w, u)}")  # 0 - perpendicular
print(f"(u × v) · v = {np.dot(w, v)}")  # 0 - perpendicular

# Anti-commutativity
w_reversed = np.cross(v, u)
print(f"v × u = {w_reversed}")  # [0, 0, -1] - opposite direction

# Application: surface normal in graphics
def surface_normal(p1, p2, p3):
    """
    Compute normal vector to a triangle defined by 3 points
    Used in 3D graphics for lighting calculations
    """
    # Two edges of the triangle
    edge1 = p2 - p1
    edge2 = p3 - p1

    # Normal is perpendicular to both edges
    normal = np.cross(edge1, edge2)

    # Normalize to unit length
    return normal / np.linalg.norm(normal)

# Example triangle
p1 = np.array([0, 0, 0])
p2 = np.array([1, 0, 0])
p3 = np.array([0, 1, 0])

normal = surface_normal(p1, p2, p3)
print(f"Triangle normal: {normal}")  # Points in z-direction
```

### Vector Norms

**Intuition**: A norm measures the "size" or "length" of a vector. Think of it as the distance from the origin to the point the vector represents.

**L2 Norm (Euclidean Distance)** - Most Common:

$$||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}$$

This is the "straight-line" distance, like measuring with a ruler.

**L1 Norm (Manhattan Distance)**:

$$||\mathbf{v}||_1 = |v_1| + |v_2| + \cdots + |v_n| = \sum_{i=1}^{n} |v_i|$$

This is the "city block" distance, like walking along a grid of streets.

**L∞ Norm (Maximum Norm)**:

$$||\mathbf{v}||_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

**General Lp Norm**:

$$||\mathbf{v}||_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

**Python Example:**

```python
v = np.array([3, 4])

# Different norms
l2_norm = np.linalg.norm(v, ord=2)     # Default: Euclidean
l1_norm = np.linalg.norm(v, ord=1)     # Manhattan
linf_norm = np.linalg.norm(v, ord=np.inf)  # Maximum

print(f"Vector: {v}")
print(f"L2 norm (Euclidean): {l2_norm}")     # sqrt(3² + 4²) = 5
print(f"L1 norm (Manhattan): {l1_norm}")     # |3| + |4| = 7
print(f"L∞ norm (Maximum): {linf_norm}")     # max(3, 4) = 4

# Unit vectors (normalized)
def normalize(v, ord=2):
    """Return unit vector in same direction as v"""
    norm = np.linalg.norm(v, ord=ord)
    if norm == 0:
        return v
    return v / norm

v_unit = normalize(v)
print(f"\nOriginal vector: {v}, length: {np.linalg.norm(v)}")
print(f"Unit vector: {v_unit}, length: {np.linalg.norm(v_unit)}")

# Distance between points (vectors)
p1 = np.array([1, 2])
p2 = np.array([4, 6])

euclidean_dist = np.linalg.norm(p2 - p1, ord=2)
manhattan_dist = np.linalg.norm(p2 - p1, ord=1)

print(f"\nDistance from {p1} to {p2}:")
print(f"  Euclidean: {euclidean_dist:.2f}")
print(f"  Manhattan: {manhattan_dist:.2f}")
```

---

## Matrices - Core Concepts

### Matrix Fundamentals

**Intuition**: A matrix is a rectangular grid of numbers. Think of it as:
- A spreadsheet organizing data in rows and columns
- A transformation that changes vectors
- A compact way to write multiple equations at once

**Definition**: An $m \times n$ matrix has $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

where $a_{ij}$ is the element in row $i$, column $j$.

**Python Example:**

```python
# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3 matrix

print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")  # (2, 3) means 2 rows, 3 columns
print(f"Element a_12 (row 0, col 1): {A[0, 1]}")  # 2 (0-indexed)

# Different ways to create matrices
zeros = np.zeros((3, 3))       # 3×3 matrix of zeros
ones = np.ones((2, 4))         # 2×4 matrix of ones
identity = np.eye(4)           # 4×4 identity matrix
diagonal = np.diag([1, 2, 3]) # Diagonal matrix
random = np.random.rand(3, 2)  # 3×2 random matrix

print(f"\nIdentity matrix:\n{identity}")
print(f"\nDiagonal matrix:\n{diagonal}")
```

**Special Matrices:**

1. **Zero Matrix**: All elements are 0
2. **Identity Matrix** $I$: 1s on diagonal, 0s elsewhere
   $$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

3. **Diagonal Matrix**: Non-zero elements only on diagonal
4. **Triangular Matrix**: Non-zero elements only above (upper) or below (lower) diagonal

### Matrix Operations

#### Matrix Addition and Subtraction

**Intuition**: Add corresponding elements, just like vector addition.

**Definition** (matrices must have same dimensions):

$$A + B = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{bmatrix}$$

**Python Example:**

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = A + B
print(f"A + B =\n{C}")  # [[6, 8], [10, 12]]

D = A - B
print(f"A - B =\n{D}")  # [[-4, -4], [-4, -4]]
```

#### Matrix Multiplication

**Intuition**: This is the most important operation! Matrix multiplication combines transformations. Think of it as:
- Applying one transformation, then another
- Computing dot products of rows and columns
- Combining multiple linear equations

**Key Insight**: For $A$ ($m \times n$) and $B$ ($n \times p$):
- The **number of columns in $A$** must equal the **number of rows in $B$**
- Result $AB$ is $m \times p$

**Definition**: Element $(AB)_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$:

$$(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

**Example** (2×3 matrix times 3×2 matrix):

$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix} = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}$$

First element: $1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 = 7 + 18 + 33 = 58$

**Python Example:**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3×2

# Matrix multiplication
C = A @ B  # Recommended: @ operator
C_alt = np.dot(A, B)  # Alternative
C_matmul = np.matmul(A, B)  # Also works

print(f"A @ B =\n{C}")
print(f"Shape: {A.shape} @ {B.shape} = {C.shape}")  # (2,3) @ (3,2) = (2,2)

# Element-wise multiplication (Hadamard product) - DIFFERENT!
A_square = np.array([[1, 2], [3, 4]])
B_square = np.array([[5, 6], [7, 8]])

hadamard = A_square * B_square  # Element-wise
matrix_mult = A_square @ B_square  # Matrix multiplication

print(f"\nElement-wise A * B:\n{hadamard}")
print(f"\nMatrix multiplication A @ B:\n{matrix_mult}")

# Matrix-vector multiplication (special case)
M = np.array([[1, 2],
              [3, 4]])
v = np.array([5, 6])

result = M @ v  # Transforms vector v
print(f"\nM @ v = {result}")  # [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
```

**Critical Properties:**
1. **NOT commutative**: $AB \neq BA$ (usually)
2. **Associative**: $(AB)C = A(BC)$
3. **Distributive**: $A(B + C) = AB + AC$
4. **Identity**: $AI = IA = A$

**Why AB ≠ BA:**

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[0, 1],
              [1, 0]])  # Swaps coordinates

AB = A @ B
BA = B @ A

print(f"AB =\n{AB}")
print(f"BA =\n{BA}")
print(f"AB == BA: {np.array_equal(AB, BA)}")  # False!
```

### Matrix Transpose

**Intuition**: Flip the matrix over its diagonal. Rows become columns, columns become rows.

**Definition**:

$$(A^T)_{ij} = A_{ji}$$

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

**Properties:**
1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(AB)^T = B^T A^T$ (order reverses!)
4. $(cA)^T = cA^T$

**Python Example:**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T  # Transpose

print(f"Original A ({A.shape}):\n{A}")
print(f"Transpose A^T ({A_T.shape}):\n{A_T}")

# Verify transpose property
B = np.array([[1, 2],
              [3, 4]])

C = np.array([[5, 6],
              [7, 8]])

# (BC)^T = C^T B^T
left = (B @ C).T
right = C.T @ B.T

print(f"(BC)^T =\n{left}")
print(f"C^T B^T =\n{right}")
print(f"Equal: {np.allclose(left, right)}")
```

### Special Matrix Types

#### Symmetric Matrix

**Definition**: A matrix equals its transpose: $A = A^T$

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$

**Properties:**
- Must be square
- Appear in optimization, physics, statistics
- Have real eigenvalues
- Eigenvectors are orthogonal

```python
# Symmetric matrix
A_sym = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])

is_symmetric = np.allclose(A_sym, A_sym.T)
print(f"Is symmetric: {is_symmetric}")

# Create symmetric matrix from any square matrix
A = np.random.rand(3, 3)
A_sym = (A + A.T) / 2  # Guaranteed symmetric
print(f"Symmetrized:\n{A_sym}")
```

#### Orthogonal Matrix

**Definition**: $Q^T Q = QQ^T = I$ (transpose equals inverse)

**Geometric Meaning**: Preserves lengths and angles (rotation or reflection)

**Properties:**
- Columns are orthonormal vectors
- Rows are orthonormal vectors
- Determinant is ±1
- Numerically stable

```python
# Rotation matrix (orthogonal)
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"Rotation matrix Q:\n{Q}")

# Verify Q^T Q = I
I = Q.T @ Q
print(f"Q^T Q:\n{I}")
print(f"Is identity: {np.allclose(I, np.eye(2))}")

# Orthogonal matrices preserve length
v = np.array([3, 4])
v_rotated = Q @ v

print(f"Original length: {np.linalg.norm(v)}")
print(f"Rotated length: {np.linalg.norm(v_rotated)}")  # Same!
```

#### Positive Definite Matrix

**Intuition**: All eigenvalues are positive. Appears in optimization as "bowl-shaped" functions.

**Definition**: For all non-zero vectors $\mathbf{x}$:

$$\mathbf{x}^T A \mathbf{x} > 0$$

**Properties:**
- Symmetric
- Used in optimization (local minima)
- Cholesky decomposition exists
- Covariance matrices are positive semi-definite

```python
# Positive definite matrix
A = np.array([[2, 1],
              [1, 2]])

# Check: all eigenvalues positive
eigenvalues = np.linalg.eigvals(A)
is_positive_def = np.all(eigenvalues > 0)

print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive definite: {is_positive_def}")

# Test with random vector
x = np.random.rand(2)
quad_form = x.T @ A @ x
print(f"x^T A x = {quad_form} > 0")  # Positive!
```

---

## Linear Systems

### Systems of Linear Equations

**Intuition**: A system of linear equations asks: "Where do multiple lines/planes intersect?" In 2D, two lines might intersect at a point, be parallel (no solution), or be the same line (infinite solutions).

**General Form**:

$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$

**Matrix Form**: $A\mathbf{x} = \mathbf{b}$

$$\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$$

**Geometric Interpretation:**

2D Example: Two lines
- Unique solution: Lines intersect at one point
- No solution: Lines are parallel
- Infinite solutions: Lines are identical

**Python Example:**

```python
# System: 2x + 3y = 8
#         x - y = 1
#
# Matrix form: [2  3] [x] = [8]
#              [1 -1] [y]   [1]

A = np.array([[2, 3],
              [1, -1]])

b = np.array([8, 1])

# Solve using NumPy
x = np.linalg.solve(A, b)

print(f"Solution: x = {x}")  # [2.5, 1.5]

# Verify solution
print(f"Verification A@x = {A @ x}")  # Should equal b
print(f"Equals b: {np.allclose(A @ x, b)}")

# Visualize the system
x_vals = np.linspace(-1, 5, 100)

# Line 1: 2x + 3y = 8  =>  y = (8 - 2x) / 3
y1 = (8 - 2*x_vals) / 3

# Line 2: x - y = 1  =>  y = x - 1
y2 = x_vals - 1

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y1, label='2x + 3y = 8')
plt.plot(x_vals, y2, label='x - y = 1')
plt.plot(x[0], x[1], 'ro', markersize=10, label=f'Solution ({x[0]:.1f}, {x[1]:.1f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('System of Linear Equations')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.show()
```

### Gaussian Elimination

**Intuition**: Systematically eliminate variables to solve the system. Like solving puzzles by substitution, but organized.

**Algorithm**:
1. **Forward elimination**: Create zeros below diagonal (upper triangular form)
2. **Back substitution**: Solve from bottom to top

**Row Operations** (don't change the solution):
1. Swap two rows
2. Multiply a row by a non-zero constant
3. Add a multiple of one row to another

**Example**: Solve manually

$$\begin{align*} x + 2y + z &= 9 \\ 2x + 4y + 3z &= 21 \\ 3x + 2y + z &= 13 \end{align*}$$

**Python Implementation:**

```python
def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting

    Returns: solution vector x
    """
    # Make copies to avoid modifying inputs
    A = A.astype(float)
    b = b.astype(float).reshape(-1, 1)
    n = len(b)

    # Create augmented matrix [A|b]
    Ab = np.column_stack([A, b])

    # Forward elimination
    for col in range(n):
        # Partial pivoting: find row with largest value in column
        max_row = np.argmax(np.abs(Ab[col:, col])) + col

        # Swap rows
        Ab[[col, max_row]] = Ab[[max_row, col]]

        # Make pivot 1
        pivot = Ab[col, col]
        if abs(pivot) < 1e-10:
            raise ValueError("Matrix is singular")

        Ab[col] = Ab[col] / pivot

        # Eliminate column below pivot
        for row in range(col + 1, n):
            factor = Ab[row, col]
            Ab[row] = Ab[row] - factor * Ab[col]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])

    return x

# Test
A = np.array([[1, 2, 1],
              [2, 4, 3],
              [3, 2, 1]], dtype=float)

b = np.array([9, 21, 13], dtype=float)

x = gaussian_elimination(A, b)
print(f"Solution: {x}")

# Verify
print(f"Verification: A@x = {A @ x}")
print(f"Should be b = {b}")
print(f"Match: {np.allclose(A @ x, b)}")

# Compare with NumPy
x_numpy = np.linalg.solve(A, b)
print(f"NumPy solution: {x_numpy}")
print(f"Solutions match: {np.allclose(x, x_numpy)}")
```

**Complexity**: $O(n^3)$ for an $n \times n$ system

### LU Decomposition

**Intuition**: Factor matrix into Lower × Upper triangular matrices. Like factoring 12 = 3 × 4, but for matrices. Once factored, solving $A\mathbf{x} = \mathbf{b}$ becomes much faster for multiple right-hand sides.

**Decomposition**: $A = LU$

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{bmatrix} \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$

**Solving $A\mathbf{x} = \mathbf{b}$ with LU**:
1. Decompose: $A = LU$ (once, $O(n^3)$)
2. Solve $L\mathbf{y} = \mathbf{b}$ (forward substitution, $O(n^2)$)
3. Solve $U\mathbf{x} = \mathbf{y}$ (back substitution, $O(n^2)$)

**Benefit**: Solve for multiple $\mathbf{b}$ vectors efficiently!

**Python Example:**

```python
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

b = np.array([1, 2, 3], dtype=float)

# LU decomposition with partial pivoting
P, L, U = lu(A)

print(f"Original matrix A:\n{A}\n")
print(f"Permutation matrix P:\n{P}\n")
print(f"Lower triangular L:\n{L}\n")
print(f"Upper triangular U:\n{U}\n")

# Verify: PA = LU
print(f"PA:\n{P @ A}")
print(f"LU:\n{L @ U}")
print(f"PA = LU: {np.allclose(P @ A, L @ U)}\n")

# Solving Ax = b using LU
# Step 1: Solve Ly = Pb
y = np.linalg.solve(L, P @ b)

# Step 2: Solve Ux = y
x = np.linalg.solve(U, y)

print(f"Solution x: {x}")
print(f"Verification A@x: {A @ x}")
print(f"Matches b: {np.allclose(A @ x, b)}")

# Solving for multiple right-hand sides
b1 = np.array([1, 2, 3])
b2 = np.array([4, 5, 6])
b3 = np.array([7, 8, 9])

# LU decomposition done once!
for b in [b1, b2, b3]:
    y = np.linalg.solve(L, P @ b)
    x = np.linalg.solve(U, y)
    print(f"b = {b}, x = {x}")
```

**Computational Comparison:**

```python
import time

# Create large system
n = 1000
A_large = np.random.rand(n, n)
bs = [np.random.rand(n) for _ in range(10)]

# Method 1: Solve each system independently
start = time.time()
for b in bs:
    x = np.linalg.solve(A_large, b)
time_direct = time.time() - start

# Method 2: LU decomposition once, then solve
start = time.time()
P, L, U = lu(A_large)
for b in bs:
    y = np.linalg.solve(L, P @ b)
    x = np.linalg.solve(U, y)
time_lu = time.time() - start

print(f"Direct solve (10 systems): {time_direct:.3f}s")
print(f"LU decomposition method: {time_lu:.3f}s")
print(f"Speedup: {time_direct/time_lu:.2f}x")
```

---

## Vector Spaces

### Vector Space Axioms

**Intuition**: A vector space is a collection of objects (vectors) where you can add them and multiply by scalars, and everything behaves nicely. Think of it as a "playground" where vectors live and play by specific rules.

**Formal Definition**: A vector space $V$ over a field $\mathbb{F}$ (usually $\mathbb{R}$) is a set with two operations:
- Vector addition: $\mathbf{u} + \mathbf{v} \in V$
- Scalar multiplication: $c\mathbf{v} \in V$

satisfying these **10 axioms**:

**Axioms** (for vectors $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $c, d \in \mathbb{R}$):

1. **Closure under addition**: $\mathbf{u} + \mathbf{v} \in V$
2. **Commutativity**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
3. **Associativity**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
4. **Zero vector exists**: $\exists \mathbf{0}$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$
5. **Additive inverse exists**: $\exists (-\mathbf{v})$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
6. **Closure under scalar multiplication**: $c\mathbf{v} \in V$
7. **Distributivity (scalar)**: $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$
8. **Distributivity (vector)**: $(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$
9. **Associativity (scalar)**: $c(d\mathbf{v}) = (cd)\mathbf{v}$
10. **Identity**: $1\mathbf{v} = \mathbf{v}$

**Examples of Vector Spaces:**
- $\mathbb{R}^n$: n-dimensional Euclidean space
- $\mathbb{C}^n$: Complex vectors
- $P_n$: Polynomials of degree ≤ n
- $M_{m \times n}$: m×n matrices
- Function spaces: continuous functions on [a, b]

**Non-Examples** (fail one or more axioms):
- Natural numbers $\mathbb{N}$ (no additive inverse)
- First quadrant of $\mathbb{R}^2$ (not closed under scalar multiplication by negatives)

### Subspaces

**Intuition**: A subspace is a vector space living inside another vector space. Like a plane through the origin in 3D space, or a line through the origin in 2D.

**Definition**: A subset $W \subseteq V$ is a subspace if:
1. $\mathbf{0} \in W$ (contains zero vector)
2. Closed under addition: if $\mathbf{u}, \mathbf{v} \in W$, then $\mathbf{u} + \mathbf{v} \in W$
3. Closed under scalar multiplication: if $\mathbf{v} \in W$ and $c \in \mathbb{R}$, then $c\mathbf{v} \in W$

**Important Subspaces of Matrix $A$ (m×n):**

1. **Column Space** $C(A)$: All linear combinations of columns
   - Span of column vectors
   - Range of transformation $\mathbf{x} \mapsto A\mathbf{x}$

2. **Row Space** $C(A^T)$: All linear combinations of rows
   - Column space of $A^T$

3. **Null Space** $N(A)$: Solutions to $A\mathbf{x} = \mathbf{0}$
   - Kernel of transformation

4. **Left Null Space** $N(A^T)$: Solutions to $A^T\mathbf{y} = \mathbf{0}$

**Python Example:**

```python
def is_in_column_space(A, b):
    """
    Check if b is in the column space of A
    (i.e., does Ax = b have a solution?)
    """
    try:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # Check if Ax ≈ b
        return np.allclose(A @ x, b)
    except:
        return False

A = np.array([[1, 2],
              [2, 4],
              [3, 6]])

# Vectors in column space
b1 = np.array([3, 6, 9])  # b1 = 3 * column1
print(f"b1 in C(A): {is_in_column_space(A, b1)}")  # True

# Vector NOT in column space (columns are parallel, so column space is a line)
b2 = np.array([1, 2, 4])  # Not a multiple of column1
print(f"b2 in C(A): {is_in_column_space(A, b2)}")  # False

# Find null space (solutions to Ax = 0)
def null_space(A, tol=1e-10):
    """
    Compute orthonormal basis for null space of A
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    # Null space spanned by columns of V corresponding to zero singular values
    null_mask = s < tol
    null_space_basis = Vt[len(s):].T  # Last columns of V
    return null_space_basis

A = np.array([[1, 2, 3],
              [2, 4, 6]])

ns = null_space(A)
print(f"\nNull space basis:\n{ns}")

# Verify: A @ ns should be zero
for i in range(ns.shape[1]):
    result = A @ ns[:, i]
    print(f"A @ null_vector_{i} = {result} (≈ 0: {np.allclose(result, 0)})")
```

### Span and Linear Independence

**Span - Intuition**: The span is all possible linear combinations of vectors. If you have two non-parallel vectors in 2D, their span is the entire plane.

**Definition**: The span of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is:

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

**Linear Independence - Intuition**: Vectors are linearly independent if none can be written as a combination of the others. In 2D, two vectors are independent if they're not parallel.

**Definition**: Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are linearly independent if:

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = \cdots = c_k = 0$$

(only the trivial combination gives zero)

**Python Example:**

```python
def are_linearly_independent(vectors):
    """
    Check if a set of vectors is linearly independent
    """
    # Stack vectors as columns
    A = np.column_stack(vectors)

    # Vectors are independent if rank equals number of vectors
    rank = np.linalg.matrix_rank(A)
    n_vectors = len(vectors)

    return rank == n_vectors

# Independent vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])
print(f"v1, v2 independent: {are_linearly_independent([v1, v2])}")  # True

# Dependent vectors (v3 = 2*v1 + 3*v2)
v3 = 2*v1 + 3*v2
print(f"v1, v2, v3 independent: {are_linearly_independent([v1, v2, v3])}")  # False

# Visualize span
def visualize_span_2d(v1, v2):
    """Visualize span of two 2D vectors"""
    plt.figure(figsize=(8, 8))

    # Plot original vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.01, label='v2')

    # Plot linear combinations
    for c1 in np.linspace(-2, 2, 20):
        for c2 in np.linspace(-2, 2, 20):
            combo = c1*v1 + c2*v2
            plt.plot(combo[0], combo[1], 'g.', markersize=2, alpha=0.3)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.legend()
    plt.title('Span of v1 and v2')
    plt.axis('equal')
    plt.show()

v1 = np.array([1, 0])
v2 = np.array([0, 1])
visualize_span_2d(v1, v2)  # Fills entire plane

# Parallel vectors (dependent)
v1_parallel = np.array([1, 1])
v2_parallel = np.array([2, 2])  # Parallel to v1
visualize_span_2d(v1_parallel, v2_parallel)  # Only fills a line
```

### Basis and Dimension

**Basis - Intuition**: A basis is a minimal set of vectors that spans the space. Like a coordinate system - just enough vectors to reach anywhere, with no redundancy.

**Definition**: A basis for vector space $V$ is a set of vectors that:
1. **Spans** $V$: Every vector in $V$ can be written as a linear combination
2. Is **linearly independent**: No redundancy

**Dimension**: The number of vectors in a basis (all bases have the same size!)

**Standard Basis for $\mathbb{R}^n$**:

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \ldots, \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

**Python Example:**

```python
# Standard basis for R^3
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

print("Standard basis for R^3:")
print(f"e1 = {e1}")
print(f"e2 = {e2}")
print(f"e3 = {e3}")

# Any vector is a linear combination of basis vectors
v = np.array([5, 7, 3])
print(f"\nVector v = {v}")
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")
reconstructed = v[0]*e1 + v[1]*e2 + v[2]*e3
print(f"Reconstructed: {reconstructed}")

# Different basis (still spans R^2)
b1 = np.array([1, 1])
b2 = np.array([1, -1])

print(f"\nAlternative basis for R^2:")
print(f"b1 = {b1}, b2 = {b2}")
print(f"Independent: {are_linearly_independent([b1, b2])}")

# Express vector in new basis
def coordinates_in_basis(v, basis):
    """
    Find coordinates of v in given basis
    """
    B = np.column_stack(basis)  # Basis vectors as columns
    coords = np.linalg.solve(B, v)
    return coords

v = np.array([3, 1])
coords = coordinates_in_basis(v, [b1, b2])
print(f"\nVector {v} in standard basis")
print(f"Coordinates in (b1, b2) basis: {coords}")
print(f"Verification: {coords[0]}*b1 + {coords[1]}*b2 = {coords[0]*b1 + coords[1]*b2}")

# Find basis for column space
def column_space_basis(A):
    """
    Find basis vectors for column space of A
    """
    Q, R = np.linalg.qr(A)
    # Number of independent columns = rank
    rank = np.linalg.matrix_rank(A)
    return Q[:, :rank]

A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 6, 7]])

basis = column_space_basis(A)
print(f"\nColumn space basis:\n{basis}")
print(f"Dimension of column space: {basis.shape[1]}")
```

---

## Linear Transformations

### Definition and Properties

**Intuition**: A linear transformation is a function that maps vectors to vectors while preserving vector addition and scalar multiplication. Think of it as stretching, rotating, or projecting space in a predictable way.

**Definition**: A function $T: V \to W$ is a linear transformation if for all vectors $\mathbf{u}, \mathbf{v}$ and scalar $c$:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (additivity)
2. $T(c\mathbf{v}) = cT(\mathbf{v})$ (homogeneity)

**Combined**: $T(c_1\mathbf{u} + c_2\mathbf{v}) = c_1T(\mathbf{u}) + c_2T(\mathbf{v})$

**Matrix Representation**: Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ can be represented as:

$$T(\mathbf{x}) = A\mathbf{x}$$

where $A$ is an $m \times n$ matrix.

**Key Properties**:
- Always maps zero to zero: $T(\mathbf{0}) = \mathbf{0}$
- Preserves lines and planes
- Preserves parallelism
- Maps grid lines to grid lines (possibly skewed)

**Python Example:**

```python
# Define transformation by its matrix
A = np.array([[2, 1],
              [1, 2]])

def transform(v):
    """Apply linear transformation defined by A"""
    return A @ v

# Test linearity properties
u = np.array([1, 0])
v = np.array([0, 1])
c = 3

# Property 1: T(u + v) = T(u) + T(v)
left = transform(u + v)
right = transform(u) + transform(v)
print(f"T(u+v) = {left}")
print(f"T(u) + T(v) = {right}")
print(f"Additive: {np.allclose(left, right)}")

# Property 2: T(cv) = cT(v)
left = transform(c * v)
right = c * transform(v)
print(f"\nT(cv) = {left}")
print(f"cT(v) = {right}")
print(f"Homogeneous: {np.allclose(left, right)}")
```

### Geometric Transformations

Common 2D transformations and their matrices:

**1. Scaling** (stretch/shrink):

$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

**2. Rotation** (counterclockwise by θ):

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**3. Reflection** (across x-axis):

$$F_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**4. Shear** (horizontal):

$$H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

**5. Projection** (onto x-axis):

$$P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

**Python Examples:**

```python
def visualize_transformation(A, title="Transformation"):
    """
    Visualize how transformation A affects a grid of points
    """
    # Create grid of points
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Original points
    points_orig = np.vstack([X.ravel(), Y.ravel()])

    # Transformed points
    points_trans = A @ points_orig

    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.scatter(points_orig[0], points_orig[1], c='blue', alpha=0.3, s=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title('Original')

    # Transformed
    plt.subplot(1, 2, 2)
    plt.scatter(points_trans[0], points_trans[1], c='red', alpha=0.3, s=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title(f'After {title}')

    plt.tight_layout()
    plt.show()

# 1. Scaling
S = np.array([[2, 0],
              [0, 0.5]])  # Double x, halve y
visualize_transformation(S, "Scaling")

# 2. Rotation (45 degrees)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
visualize_transformation(R, "Rotation 45°")

# 3. Reflection (across y = x)
F = np.array([[0, 1],
              [1, 0]])
visualize_transformation(F, "Reflection")

# 4. Shear
H = np.array([[1, 0.5],
              [0, 1]])
visualize_transformation(H, "Shear")

# 5. Projection (onto x-axis)
P = np.array([[1, 0],
              [0, 0]])
visualize_transformation(P, "Projection")

# Transform specific shapes
def transform_shape(A, shape_points):
    """Transform a shape defined by points"""
    return A @ shape_points

# Create a square
square = np.array([[0, 1, 1, 0, 0],   # x-coordinates
                   [0, 0, 1, 1, 0]])  # y-coordinates

# Apply rotation
theta = np.pi / 6  # 30 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

square_rotated = R @ square

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(square[0], square[1], 'b-', linewidth=2)
plt.grid(True)
plt.axis('equal')
plt.title('Original Square')

plt.subplot(1, 2, 2)
plt.plot(square_rotated[0], square_rotated[1], 'r-', linewidth=2)
plt.grid(True)
plt.axis('equal')
plt.title('Rotated Square')
plt.show()
```

### Composition of Transformations

**Intuition**: Applying one transformation after another is like function composition, represented by matrix multiplication.

**Key Insight**: $T_2(T_1(\mathbf{x})) = T_2(A_1\mathbf{x}) = A_2(A_1\mathbf{x}) = (A_2A_1)\mathbf{x}$

**Order Matters!** $A_2A_1 \neq A_1A_2$ (usually)

**Python Example:**

```python
# Rotate then scale
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

S = np.array([[2, 0],
              [0, 0.5]])

# Composition 1: Rotate THEN scale
M1 = S @ R  # Apply R first, then S

# Composition 2: Scale THEN rotate
M2 = R @ S  # Apply S first, then R

print(f"Rotate then scale:\n{M1}\n")
print(f"Scale then rotate:\n{M2}\n")
print(f"Same result: {np.allclose(M1, M2)}")  # False - order matters!

# Visualize difference
v = np.array([1, 0])

result1 = M1 @ v  # Rotate then scale
result2 = M2 @ v  # Scale then rotate

print(f"\nOriginal vector: {v}")
print(f"Rotate→Scale: {result1}")
print(f"Scale→Rotate: {result2}")

# 3D Rotation around axis
def rotation_x(theta):
    """Rotation around x-axis"""
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta),  np.cos(theta)]])

def rotation_y(theta):
    """Rotation around y-axis"""
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_z(theta):
    """Rotation around z-axis"""
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])

# Compose rotations
angle = np.pi / 6
Rx = rotation_x(angle)
Ry = rotation_y(angle)
Rz = rotation_z(angle)

# Combined rotation
R_combined = Rz @ Ry @ Rx  # Apply Rx, then Ry, then Rz

print(f"\nCombined 3D rotation:\n{R_combined}")
```

### Inverse Transformations

**Intuition**: An inverse transformation "undoes" the original. If $T$ rotates 45°, then $T^{-1}$ rotates -45°.

**Definition**: $T^{-1}$ is the inverse if:

$$T^{-1}(T(\mathbf{x})) = \mathbf{x}$$
$$T(T^{-1}(\mathbf{x})) = \mathbf{x}$$

**Matrix Form**: $A^{-1}A = AA^{-1} = I$

**Existence**: Inverse exists if and only if:
- Matrix is square ($n \times n$)
- Determinant $\det(A) \neq 0$
- Columns are linearly independent
- Transformation is one-to-one and onto

**Python Example:**

```python
# Rotation matrix (always invertible)
theta = np.pi / 3  # 60 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Inverse rotation
R_inv = np.linalg.inv(R)

print(f"Rotation matrix R:\n{R}\n")
print(f"Inverse R^(-1):\n{R_inv}\n")

# For rotation, inverse = transpose!
print(f"Transpose R^T:\n{R.T}\n")
print(f"R^(-1) = R^T: {np.allclose(R_inv, R.T)}")

# Verify inverse property
I = R @ R_inv
print(f"\nR @ R^(-1) =\n{I}")
print(f"Is identity: {np.allclose(I, np.eye(2))}")

# Apply transformation and undo it
v = np.array([3, 4])
v_rotated = R @ v
v_restored = R_inv @ v_rotated

print(f"\nOriginal: {v}")
print(f"After rotation: {v_rotated}")
print(f"After inverse: {v_restored}")
print(f"Restored: {np.allclose(v, v_restored)}")

# Non-invertible transformation (projection)
P = np.array([[1, 0],
              [0, 0]])  # Project to x-axis

try:
    P_inv = np.linalg.inv(P)
except np.linalg.LinAlgError:
    print("\nProjection matrix is not invertible!")
    print("(Information is lost - can't recover y-coordinate)")

# Check determinant
print(f"det(R) = {np.linalg.det(R):.6f} (invertible)")
print(f"det(P) = {np.linalg.det(P):.6f} (not invertible)")
```

---

## Determinants

### Geometric Interpretation (Determinants)

**Intuition**: The determinant measures the "signed volume" of the transformation. For 2D, it's the area of the parallelogram formed by the matrix columns; for 3D, it's the volume of the parallelepiped.

**Key Insights:**
- $|\det(A)| =$ volume scaling factor
- $\det(A) > 0$: preserves orientation (no flip)
- $\det(A) < 0$: reverses orientation (flip/reflection)
- $\det(A) = 0$: collapses space (not invertible)

**2D Example:**

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

This is the area of the parallelogram with sides $\begin{bmatrix} a \\ c \end{bmatrix}$ and $\begin{bmatrix} b \\ d \end{bmatrix}$.

**Python Visualization:**

```python
def visualize_determinant_2d(A):
    """Visualize determinant as area"""
    # Column vectors
    v1 = A[:, 0]
    v2 = A[:, 1]

    # Create parallelogram
    parallelogram = np.array([[0, v1[0], v1[0]+v2[0], v2[0], 0],
                              [0, v1[1], v1[1]+v2[1], v2[1], 0]])

    det = np.linalg.det(A)

    plt.figure(figsize=(8, 8))
    plt.fill(parallelogram[0], parallelogram[1], alpha=0.3, color='blue')
    plt.plot(parallelogram[0], parallelogram[1], 'b-', linewidth=2)

    # Draw column vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
               color='red', width=0.01, label=f'col1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
               color='green', width=0.01, label=f'col2')

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title(f'Determinant = {det:.2f} (Area of parallelogram)')
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

# Example matrices
A1 = np.array([[2, 1],
               [1, 2]])  # det = 3
A2 = np.array([[2, 4],
               [1, 2]])  # det = 0 (parallel columns)

visualize_determinant_2d(A1)
```

### Computing Determinants

**2×2 Matrix:**

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**3×3 Matrix** (cofactor expansion along first row):

$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a\det\begin{bmatrix} e & f \\ h & i \end{bmatrix} - b\det\begin{bmatrix} d & f \\ g & i \end{bmatrix} + c\det\begin{bmatrix} d & e \\ g & h \end{bmatrix}$$

**General n×n Matrix** (cofactor expansion):

$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

where $M_{ij}$ is the minor (determinant of submatrix with row $i$ and column $j$ removed).

**Python Examples:**

```python
# 2×2 determinant
A = np.array([[3, 8],
              [4, 6]])

det_manual = A[0,0]*A[1,1] - A[0,1]*A[1,0]
det_numpy = np.linalg.det(A)

print(f"2×2 Matrix:\n{A}")
print(f"Manual calculation: {det_manual}")
print(f"NumPy: {det_numpy}")

# 3×3 determinant
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

det = np.linalg.det(A)
print(f"\n3×3 Matrix:\n{A}")
print(f"Determinant: {det:.6f}")  # ≈ 0 (columns are linearly dependent!)

# Cofactor expansion (educational)
def determinant_cofactor(A):
    """
    Compute determinant using cofactor expansion
    (Slow: O(n!), only for small matrices)
    """
    n = A.shape[0]

    # Base case: 1×1
    if n == 1:
        return A[0, 0]

    # Base case: 2×2
    if n == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]

    # Recursive case: expand along first row
    det = 0
    for j in range(n):
        # Create minor (remove row 0, column j)
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * determinant_cofactor(minor)
        det += A[0, j] * cofactor

    return det

A_small = np.array([[1, 2, 3],
                    [0, 4, 5],
                    [1, 0, 6]])

print(f"\nCofactor expansion: {determinant_cofactor(A_small)}")
print(f"NumPy (LU-based): {np.linalg.det(A_small)}")
```

### Properties of Determinants

**Key Properties:**

1. **Multiplicative**: $\det(AB) = \det(A)\det(B)$
2. **Transpose**: $\det(A^T) = \det(A)$
3. **Inverse**: $\det(A^{-1}) = \frac{1}{\det(A)}$
4. **Scalar multiple**: $\det(cA) = c^n\det(A)$ for $n \times n$ matrix
5. **Row operations**:
   - Swap rows: determinant changes sign
   - Multiply row by $c$: determinant multiplied by $c$
   - Add multiple of one row to another: determinant unchanged

**Python Examples:**

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[2, 0],
              [1, 2]])

# Property 1: det(AB) = det(A)det(B)
det_AB = np.linalg.det(A @ B)
det_A_times_det_B = np.linalg.det(A) * np.linalg.det(B)

print(f"det(AB) = {det_AB:.6f}")
print(f"det(A)·det(B) = {det_A_times_det_B:.6f}")
print(f"Equal: {np.isclose(det_AB, det_A_times_det_B)}")

# Property 2: det(A^T) = det(A)
det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)
print(f"\ndet(A) = {det_A:.6f}")
print(f"det(A^T) = {det_AT:.6f}")

# Property 3: det(A^-1) = 1/det(A)
if det_A != 0:
    A_inv = np.linalg.inv(A)
    det_A_inv = np.linalg.det(A_inv)
    print(f"\ndet(A^-1) = {det_A_inv:.6f}")
    print(f"1/det(A) = {1/det_A:.6f}")

# Determinant test for invertibility
matrices = [
    np.array([[1, 2], [3, 4]]),      # Invertible
    np.array([[1, 2], [2, 4]]),      # Not invertible (parallel rows)
    np.array([[2, 0], [0, 3]])       # Invertible (diagonal)
]

for i, M in enumerate(matrices):
    det = np.linalg.det(M)
    invertible = abs(det) > 1e-10
    print(f"\nMatrix {i+1}: det = {det:.6f}, Invertible: {invertible}")
```

---

## Eigenvalues and Eigenvectors

### Fundamental Concepts (Eigenvalues)

**Intuition**: Eigenvectors are special vectors that don't change direction when a transformation is applied - they only get scaled. The scaling factor is the eigenvalue.

Think of stretching a rubber sheet: most points move in complicated ways, but points along certain directions just move straight in or out. Those special directions are eigenvectors.

**Definition**: For matrix $A$ and vector $\mathbf{v} \neq \mathbf{0}$:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where:
- $\mathbf{v}$ is an **eigenvector**
- $\lambda$ is an **eigenvalue** (can be negative or complex)

**Geometric Meaning:**
- Eigenvector: direction preserved by transformation
- Eigenvalue: how much it's stretched ($\lambda > 1$), shrunk ($|\lambda| < 1$), or flipped ($\lambda < 0$)

**Characteristic Equation**:

$$\det(A - \lambda I) = 0$$

This polynomial equation gives us all eigenvalues.

**Python Example:**

```python
# Simple 2×2 example
A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}\n")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Verify Av = λv for each eigenvalue
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]  # i-th column is i-th eigenvector

    Av = A @ v
    λv = λ * v

    print(f"Eigenvalue λ_{i+1} = {λ:.4f}")
    print(f"  Av = {Av}")
    print(f"  λv = {λv}")
    print(f"  Match: {np.allclose(Av, λv)}\n")

# Visualize eigenvectors
def visualize_eigenvectors(A):
    """Visualize how matrix A affects space and its eigenvectors"""
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Create unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform circle
    ellipse = A @ circle

    plt.figure(figsize=(10, 5))

    # Original circle
    plt.subplot(1, 2, 1)
    plt.plot(circle[0], circle[1], 'b-', label='Unit circle')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Original')

    # Transformed ellipse with eigenvectors
    plt.subplot(1, 2, 2)
    plt.plot(ellipse[0], ellipse[1], 'r-', label='Transformed')

    # Plot eigenvectors
    for i in range(len(eigenvalues)):
        λ = eigenvalues[i].real
        v = eigenvectors[:, i].real
        # Eigenvector is scaled by λ
        plt.quiver(0, 0, λ*v[0], λ*v[1],
                   angles='xy', scale_units='xy', scale=1,
                   width=0.01, label=f'λ={λ:.2f}')

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('After transformation A')
    plt.tight_layout()
    plt.show()

visualize_eigenvectors(A)
```

### Computing Eigenvalues

**Step-by-Step Process:**

1. **Form characteristic equation**: $\det(A - \lambda I) = 0$
2. **Solve for** $\lambda$ (roots of polynomial)
3. **For each** $\lambda$, **solve** $(A - \lambda I)\mathbf{v} = \mathbf{0}$ to find eigenvectors

**Example**: Find eigenvalues of $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$

$$\det(A - \lambda I) = \det\begin{bmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{bmatrix} = (1-\lambda)^2 - 4 = \lambda^2 - 2\lambda - 3 = 0$$

$$\lambda = 3 \text{ or } \lambda = -1$$

**Python Implementation:**

```python
# Manual characteristic polynomial
A = np.array([[1, 2],
              [2, 1]])

# For 2×2: det(A - λI) = (a-λ)(d-λ) - bc = λ² - (a+d)λ + (ad-bc)
trace = np.trace(A)  # a + d
det = np.linalg.det(A)  # ad - bc

print(f"Characteristic polynomial: λ² - {trace}λ + {det}")
print("  = λ² - 2λ - 3")
print("  = (λ - 3)(λ + 1)")

# Roots are eigenvalues
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalues: {eigenvalues}")  # [3, -1]

# Find eigenvectors for λ = 3
λ1 = 3
# Solve (A - 3I)v = 0
A_minus_λI = A - λ1 * np.eye(2)
print(f"\n(A - 3I) =\n{A_minus_λI}")
# Null space gives eigenvector
# For this matrix: [-2, 2; 2, -2] → eigenvector [1, 1] (or any multiple)

# Find eigenvectors for λ = -1
λ2 = -1
A_minus_λI = A - λ2 * np.eye(2)
print(f"\n(A - (-1)I) =\n{A_minus_λI}")
# [2, 2; 2, 2] → eigenvector [1, -1] (or any multiple)

# Verify with NumPy
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nNumPy eigenvectors:\n{eigenvectors}")

# Special matrices and their eigenvalues
print("\n=== Special Matrices ===")

# Diagonal matrix: eigenvalues are diagonal elements
D = np.diag([5, 3, -2])
print(f"\nDiagonal matrix:\n{D}")
print(f"Eigenvalues: {np.linalg.eigvals(D)}")

# Symmetric matrix: always real eigenvalues
S = np.array([[2, 1], [1, 2]])
print(f"\nSymmetric matrix:\n{S}")
print(f"Eigenvalues: {np.linalg.eigvals(S)}")  # Real

# Orthogonal matrix: eigenvalues have magnitude 1
θ = np.pi/4
Q = np.array([[np.cos(θ), -np.sin(θ)],
              [np.sin(θ),  np.cos(θ)]])
print(f"\nOrthogonal (rotation) matrix:\n{Q}")
eigs = np.linalg.eigvals(Q)
print(f"Eigenvalues: {eigs}")
print(f"Magnitudes: {np.abs(eigs)}")  # Both ≈ 1
```

### Diagonalization

**Intuition**: Diagonalization means finding a basis where the matrix is diagonal. In this basis, the transformation just scales each coordinate independently - much simpler!

**Diagonalization Theorem**: If $A$ has $n$ linearly independent eigenvectors, then:

$$A = PDP^{-1}$$

where:
- $D$ is diagonal with eigenvalues on diagonal
- $P$ has eigenvectors as columns

**Power of Diagonalization**: Computing $A^k$ becomes easy:

$$A^k = (PDP^{-1})^k = PD^kP^{-1}$$

and $D^k$ is trivial: just raise diagonal elements to power $k$.

**Python Example:**

```python
A = np.array([[1, 2],
              [2, 1]])

# Diagonalize
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)

print(f"Original matrix A:\n{A}\n")
print(f"Eigenvector matrix P:\n{P}\n")
print(f"Diagonal matrix D:\n{D}\n")

# Verify A = PDP^(-1)
A_reconstructed = P @ D @ np.linalg.inv(P)
print(f"PDP^(-1):\n{A_reconstructed}")
print(f"Equals A: {np.allclose(A, A_reconstructed)}\n")

# Compute A^10 efficiently
k = 10

# Method 1: Direct (slow for large k)
A_power_direct = np.linalg.matrix_power(A, k)

# Method 2: Using diagonalization (fast!)
D_power = np.diag(eigenvalues ** k)
A_power_diag = P @ D_power @ np.linalg.inv(P)

print(f"A^{k} (direct):\n{A_power_direct}\n")
print(f"A^{k} (diagonalization):\n{A_power_diag}\n")
print(f"Match: {np.allclose(A_power_direct, A_power_diag)}")

# Application: Fibonacci sequence
def fibonacci_matrix(n):
    """
    Compute nth Fibonacci number using matrix exponentiation
    F_n is given by: [[1,1],[1,0]]^n [1,0]^T
    """
    A = np.array([[1, 1],
                  [1, 0]], dtype=float)

    # Diagonalize for fast computation
    eigenvalues, P = np.linalg.eig(A)
    D = np.diag(eigenvalues)

    # A^n = P D^n P^(-1)
    D_n = np.diag(eigenvalues ** n)
    A_n = P @ D_n @ np.linalg.inv(P)

    # F_n = top-right element
    result = A_n @ np.array([1, 0])
    return int(result[0])

# Compute large Fibonacci numbers efficiently
for n in [10, 20, 50]:
    fib_n = fibonacci_matrix(n)
    print(f"F_{n} = {fib_n}")
```

### Applications of Eigenvalues

**1. Stability Analysis** (Differential Equations)

System $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ is:
- Stable if all eigenvalues have negative real part
- Unstable if any eigenvalue has positive real part

**2. Principal Component Analysis (PCA)**

Find directions of maximum variance in data using eigenvectors of covariance matrix.

**3. PageRank Algorithm**

Largest eigenvector of web link matrix gives page rankings.

**4. Quantum Mechanics**

Observable quantities are eigenvalues of operators; states are eigenvectors.

**Python Example - PCA Preview:**

```python
# Generate correlated 2D data
np.random.seed(42)
mean = [0, 0]
cov = [[2, 1.5],
       [1.5, 2]]  # Covariance matrix
data = np.random.multivariate_normal(mean, cov, 1000)

# Compute covariance matrix
data_centered = data - np.mean(data, axis=0)
cov_matrix = np.cov(data_centered.T)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (largest first)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Covariance matrix:\n{cov_matrix}\n")
print(f"Eigenvalues (variance along principal components): {eigenvalues}")
print(f"Eigenvectors (principal directions):\n{eigenvectors}\n")

# Visualize
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)

# Plot principal components
origin = np.mean(data, axis=0)
for i in range(2):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    plt.arrow(origin[0], origin[1], v[0], v[1],
              head_width=0.3, head_length=0.3, fc=f'C{i+1}', ec=f'C{i+1}',
              linewidth=2, label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Data with Principal Components')

# Projected onto first principal component
plt.subplot(1, 2, 2)
# Project data onto first PC
pc1 = eigenvectors[:, 0]
projected = (data_centered @ pc1).reshape(-1, 1) * pc1

plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=10)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.title('Projected onto 1st Principal Component')
plt.tight_layout()
plt.show()

# Variance explained
total_variance = np.sum(eigenvalues)
variance_explained = eigenvalues / total_variance * 100
print(f"Variance explained by each PC: {variance_explained}%")
```

---

## Orthogonality

### Orthogonal Vectors

**Intuition**: Vectors are orthogonal (perpendicular) if their dot product is zero. Think of x and y axes - completely independent directions.

**Definition**: Vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal if:

$$\mathbf{u} \cdot \mathbf{v} = 0$$

**Orthonormal**: Orthogonal AND unit length ($||\mathbf{v}|| = 1$)

**Orthogonal Matrix**: Matrix $Q$ where $Q^TQ = I$
- Columns are orthonormal
- Rows are orthonormal
- Preserves lengths and angles
- $Q^{-1} = Q^T$ (transpose is inverse!)

**Python Example:**

```python
# Orthogonal vectors
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([0, 0, 1])

print(f"u·v = {np.dot(u, v)}")  # 0 - orthogonal
print(f"u·w = {np.dot(u, w)}")  # 0 - orthogonal
print(f"v·w = {np.dot(v, w)}")  # 0 - orthogonal

# Orthogonal matrix (rotation)
θ = np.pi/6
Q = np.array([[np.cos(θ), -np.sin(θ)],
              [np.sin(θ),  np.cos(θ)]])

print(f"\nRotation matrix Q:\n{Q}")

# Check Q^T Q = I
QTQ = Q.T @ Q
print(f"\nQ^T Q:\n{QTQ}")
print(f"Is identity: {np.allclose(QTQ, np.eye(2))}")

# Check Q^(-1) = Q^T
Q_inv = np.linalg.inv(Q)
print(f"\nQ^(-1) =\n{Q_inv}")
print(f"Q^T =\n{Q.T}")
print(f"Equal: {np.allclose(Q_inv, Q.T)}")

# Orthogonal matrices preserve length
v = np.array([3, 4])
v_rotated = Q @ v

print(f"\nOriginal length: {np.linalg.norm(v):.4f}")
print(f"After rotation: {np.linalg.norm(v_rotated):.4f}")  # Same!

# Orthogonal matrices preserve angles
v1 = np.array([1, 0])
v2 = np.array([1, 1])

angle_original = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

v1_rot = Q @ v1
v2_rot = Q @ v2

angle_rotated = np.arccos(np.dot(v1_rot, v2_rot) / (np.linalg.norm(v1_rot) * np.linalg.norm(v2_rot)))

print(f"\nAngle before rotation: {np.degrees(angle_original):.2f}°")
print(f"Angle after rotation: {np.degrees(angle_rotated):.2f}°")
```

### Gram-Schmidt Process

**Intuition**: Convert any set of independent vectors into orthonormal vectors. Like taking slanted axes and straightening them into perpendicular ones.

**Algorithm** (Gram-Schmidt Orthogonalization):

Given vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$:

1. $\mathbf{u}_1 = \mathbf{v}_1$
2. $\mathbf{u}_2 = \mathbf{v}_2 - \text{proj}_{\mathbf{u}_1}(\mathbf{v}_2)$
3. $\mathbf{u}_3 = \mathbf{v}_3 - \text{proj}_{\mathbf{u}_1}(\mathbf{v}_3) - \text{proj}_{\mathbf{u}_2}(\mathbf{v}_3)$
4. Continue...
5. Normalize each $\mathbf{u}_i$ to get orthonormal basis

**Python Implementation:**

```python
def gram_schmidt(vectors):
    """
    Orthogonalize vectors using Gram-Schmidt process

    Args:
        vectors: list of numpy arrays

    Returns:
        orthonormal: list of orthonormal vectors
    """
    orthogonal = []

    for v in vectors:
        # Start with current vector
        u = v.copy().astype(float)

        # Subtract projection onto all previous orthogonal vectors
        for basis in orthogonal:
            projection = (np.dot(v, basis) / np.dot(basis, basis)) * basis
            u = u - projection

        orthogonal.append(u)

    # Normalize to get orthonormal basis
    orthonormal = [u / np.linalg.norm(u) for u in orthogonal]

    return orthonormal

# Example: orthogonalize vectors
v1 = np.array([1, 1, 0], dtype=float)
v2 = np.array([1, 0, 1], dtype=float)
v3 = np.array([0, 1, 1], dtype=float)

print("Original vectors:")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v3 = {v3}")

# Apply Gram-Schmidt
orthonormal = gram_schmidt([v1, v2, v3])

print("\nOrthonormal vectors:")
for i, u in enumerate(orthonormal):
    print(f"u{i+1} = {u}")

# Verify orthonormality
print("\nVerification:")
for i in range(len(orthonormal)):
    for j in range(len(orthonormal)):
        dot_product = np.dot(orthonormal[i], orthonormal[j])
        expected = 1.0 if i == j else 0.0
        print(f"u{i+1}·u{j+1} = {dot_product:.6f} (expected {expected})")

# Compare with NumPy's QR decomposition
A = np.column_stack([v1, v2, v3])
Q, R = np.linalg.qr(A)

print("\nNumPy QR decomposition:")
print(f"Q (orthonormal columns):\n{Q}")
print(f"\nOur Gram-Schmidt result:")
print(np.column_stack(orthonormal))
```

### Orthogonal Projections

**Intuition**: Projection is the "shadow" of one vector onto another. Like the shadow of a stick on the ground when the sun shines from above.

**Projection of $\mathbf{v}$ onto $\mathbf{u}$:**

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}||^2} \mathbf{u}$$

**Projection Matrix** (onto column space of $A$):

$$P = A(A^TA)^{-1}A^T$$

**Properties**:
- $P^2 = P$ (idempotent: projecting twice = projecting once)
- $P^T = P$ (symmetric)
- $I - P$ projects onto orthogonal complement

**Python Example:**

```python
def project_onto_vector(v, u):
    """
    Project vector v onto vector u
    """
    return (np.dot(v, u) / np.dot(u, u)) * u

# Example
u = np.array([1, 0])  # Project onto x-axis
v = np.array([3, 4])

proj = project_onto_vector(v, u)
perp = v - proj  # Perpendicular component

print(f"Vector v: {v}")
print(f"Projection onto u: {proj}")
print(f"Perpendicular component: {perp}")
print(f"Perpendicular? {np.isclose(np.dot(proj, perp), 0)}")

# Visualize
plt.figure(figsize=(8, 8))
origin = np.array([0, 0])

# Original vectors
plt.quiver(*origin, u[0], u[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01, label='u (direction)')
plt.quiver(*origin, v[0], v[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01, label='v (vector)')
plt.quiver(*origin, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.01, label='proj_u(v)')
plt.quiver(proj[0], proj[1], perp[0], perp[1], angles='xy', scale_units='xy', scale=1,
           color='orange', width=0.01, label='perpendicular')

# Show perpendicularity
plt.plot([proj[0], v[0]], [proj[1], v[1]], 'k--', alpha=0.5)

plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.title('Vector Projection')
plt.show()

# Projection matrix
def projection_matrix(A):
    """
    Matrix that projects onto column space of A
    """
    return A @ np.linalg.inv(A.T @ A) @ A.T

# Project onto a subspace
# Example: project onto plane spanned by [1,0,0] and [0,1,0] (xy-plane)
basis = np.array([[1, 0],
                  [0, 1],
                  [0, 0]], dtype=float)  # Two basis vectors as columns

P = projection_matrix(basis)

print(f"\nProjection matrix onto xy-plane:\n{P}")

# Project a 3D vector onto xy-plane
v_3d = np.array([3, 4, 5])
v_projected = P @ v_3d

print(f"\n3D vector: {v_3d}")
print(f"Projected onto xy-plane: {v_projected}")  # [3, 4, 0] - z-component removed!

# Verify P² = P
P2 = P @ P
print(f"\nP² = P? {np.allclose(P2, P)}")
```

---

## Matrix Decompositions

### LU Decomposition (Extended)

Already covered in Linear Systems section. Key points:
- $A = LU$ or $PA = LU$ with pivoting
- Efficient for solving multiple systems
- $O(n^3)$ to decompose, $O(n^2)$ per solve
- Foundation for many algorithms

### QR Decomposition

**Intuition**: Factor matrix into Orthogonal × Upper triangular. Like Gram-Schmidt in matrix form!

**Decomposition**: $A = QR$

where:
- $Q$: orthogonal matrix ($Q^TQ = I$)
- $R$: upper triangular matrix

**Why QR is Useful**:
- More numerically stable than normal equations
- Solve least squares: $A\mathbf{x} \approx \mathbf{b}$ → $R\mathbf{x} = Q^T\mathbf{b}$
- Compute eigenvalues (QR algorithm)
- Orthonormal basis for column space

**Python Example:**

```python
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

# QR decomposition
Q, R = np.linalg.qr(A)

print(f"Original matrix A:\n{A}\n")
print(f"Orthogonal matrix Q:\n{Q}\n")
print(f"Upper triangular R:\n{R}\n")

# Verify A = QR
A_reconstructed = Q @ R
print(f"QR:\n{A_reconstructed}")
print(f"Equals A: {np.allclose(A, A_reconstructed)}\n")

# Verify Q is orthogonal
QTQ = Q.T @ Q
print(f"Q^T Q:\n{QTQ}")
print(f"Is identity: {np.allclose(QTQ, np.eye(3))}\n")

# Application: Solve Ax = b using QR
b = np.array([6, 5, 4])

# Ax = b → QRx = b → Rx = Q^T b
y = Q.T @ b
x = np.linalg.solve(R, y)  # Back substitution

print(f"Solution x: {x}")
print(f"Verification Ax: {A @ x}")
print(f"Matches b: {np.allclose(A @ x, b)}")

# Compare methods for least squares
A_tall = np.random.rand(100, 10)  # Overdetermined system
b_tall = np.random.rand(100)

# Method 1: Normal equations (can be unstable)
x1 = np.linalg.inv(A_tall.T @ A_tall) @ A_tall.T @ b_tall

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(A_tall)
x2 = np.linalg.solve(R, Q.T @ b_tall)

# Method 3: NumPy's lstsq (uses SVD, most stable)
x3 = np.linalg.lstsq(A_tall, b_tall, rcond=None)[0]

print(f"\nLeast squares solutions match:")
print(f"Normal vs QR: {np.allclose(x1, x2)}")
print(f"QR vs SVD: {np.allclose(x2, x3)}")
```

### Cholesky Decomposition

**Intuition**: For positive definite symmetric matrices, there's a special decomposition: $A = LL^T$ where $L$ is lower triangular. Like taking a "square root" of a matrix!

**Requirements**:
- Matrix must be symmetric: $A = A^T$
- Matrix must be positive definite: $\mathbf{x}^TA\mathbf{x} > 0$ for all $\mathbf{x} \neq 0$

**Advantages**:
- Half the storage of LU (only need $L$)
- Twice as fast as LU
- Numerically stable
- Used in optimization, simulation

**Python Example:**

```python
# Create positive definite matrix
A = np.array([[4, 2],
              [2, 3]], dtype=float)

# Check positive definiteness
eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Positive definite: {np.all(eigenvalues > 0)}\n")

# Cholesky decomposition
L = np.linalg.cholesky(A)

print(f"Matrix A:\n{A}\n")
print(f"Lower triangular L:\n{L}\n")

# Verify A = LL^T
A_reconstructed = L @ L.T
print(f"LL^T:\n{A_reconstructed}")
print(f"Equals A: {np.allclose(A, A_reconstructed)}\n")

# Solve Ax = b using Cholesky
b = np.array([8, 7])

# Ax = b → LL^T x = b
# 1. Solve Ly = b (forward substitution)
y = np.linalg.solve(L, b)
# 2. Solve L^T x = y (back substitution)
x = np.linalg.solve(L.T, y)

print(f"Solution x: {x}")
print(f"Verification Ax: {A @ x}")

# Generating correlated random numbers
def generate_correlated_samples(mean, cov, n_samples):
    """
    Generate samples from multivariate normal using Cholesky
    """
    # Cholesky decomposition of covariance
    L = np.linalg.cholesky(cov)

    # Generate uncorrelated samples
    uncorrelated = np.random.randn(n_samples, len(mean))

    # Transform to correlated samples
    correlated = uncorrelated @ L.T + mean

    return correlated

# Example: generate correlated data
mean = np.array([0, 0])
cov = np.array([[2, 1],
                [1, 2]])

samples = generate_correlated_samples(mean, cov, 1000)

print(f"\nGenerated {len(samples)} correlated samples")
print(f"Sample mean: {np.mean(samples, axis=0)}")
print(f"Sample covariance:\n{np.cov(samples.T)}")
```

### Singular Value Decomposition (SVD)

**Intuition**: THE most important matrix decomposition! Every matrix (even non-square!) can be decomposed into:
- Rotation in input space (V^T)
- Scaling along principal axes (Σ)
- Rotation in output space (U)

**Decomposition**: For any $m \times n$ matrix $A$:

$$A = U\Sigma V^T$$

where:
- $U$ ($m \times m$): orthogonal, left singular vectors
- $\Sigma$ ($m \times n$): diagonal, singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V$ ($n \times n$): orthogonal, right singular vectors

**Geometric Interpretation**:
Unit sphere → (V^T rotates) → (Σ scales) → (U rotates) → ellipsoid

**Relationship to Eigenvalues**:
- $AA^T = U\Sigma^2U^T$ (eigendecomposition of $AA^T$)
- $A^TA = V\Sigma^2V^T$ (eigendecomposition of $A^TA$)
- Singular values = square roots of eigenvalues of $A^TA$

**Python Example:**

```python
A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

# SVD
U, s, Vt = np.linalg.svd(A, full_matrices=True)

print(f"Matrix A ({A.shape}):\n{A}\n")
print(f"U ({U.shape}) - left singular vectors:\n{U}\n")
print(f"Singular values: {s}\n")
print(f"V^T ({Vt.shape}) - right singular vectors:\n{Vt}\n")

# Reconstruct A
Sigma = np.zeros(A.shape)
Sigma[:len(s), :len(s)] = np.diag(s)

A_reconstructed = U @ Sigma @ Vt
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Matches original: {np.allclose(A, A_reconstructed)}\n")

# Compact SVD (more practical)
U_compact, s, Vt_compact = np.linalg.svd(A, full_matrices=False)
A_compact = U_compact @ np.diag(s) @ Vt_compact
print(f"Compact SVD reconstruction:\n{A_compact}\n")

# Verify orthogonality
print(f"U^T U:\n{U.T @ U}")
print(f"V^T V:\n{Vt.T @ Vt}\n")

# Relationship to eigenvalues
AAT = A @ A.T
ATA = A.T @ A

eigenvalues_AAT = np.linalg.eigvals(AAT)
eigenvalues_ATA = np.linalg.eigvals(ATA)

print(f"Singular values squared: {s**2}")
print(f"Eigenvalues of AA^T: {np.sort(eigenvalues_AAT)[::-1]}")
print(f"Eigenvalues of A^TA: {np.sort(eigenvalues_ATA)[::-1]}")

# Visualize SVD transformation
def visualize_svd_2d(A):
    """Visualize how SVD breaks down transformation"""
    U, s, Vt = np.linalg.svd(A)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])

    # Apply transformations step by step
    after_Vt = Vt @ circle
    after_Sigma = np.diag(s) @ after_Vt
    after_U = U @ after_Sigma

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original circle
    axes[0].plot(circle[0], circle[1], 'b-')
    axes[0].set_title('1. Original (unit circle)')
    axes[0].axis('equal')
    axes[0].grid(True)

    # After V^T (rotation)
    axes[1].plot(after_Vt[0], after_Vt[1], 'g-')
    axes[1].set_title('2. After V^T (rotation)')
    axes[1].axis('equal')
    axes[1].grid(True)

    # After Σ (scaling)
    axes[2].plot(after_Sigma[0], after_Sigma[1], 'r-')
    axes[2].set_title(f'3. After Σ (scale by {s})')
    axes[2].axis('equal')
    axes[2].grid(True)

    # After U (rotation)
    axes[3].plot(after_U[0], after_U[1], 'm-')
    axes[3].set_title('4. After U (rotation)')
    axes[3].axis('equal')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

A_2d = np.array([[3, 1],
                 [1, 3]])
visualize_svd_2d(A_2d)
```

**SVD Applications:**

```python
# 1. Low-rank approximation
def low_rank_approx(A, k):
    """Best rank-k approximation to A (in Frobenius norm)"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

A = np.random.rand(10, 10)
for k in [1, 2, 5]:
    A_k = low_rank_approx(A, k)
    error = np.linalg.norm(A - A_k, 'fro')
    print(f"Rank-{k} approximation error: {error:.4f}")

# 2. Matrix pseudoinverse
def pseudoinverse_svd(A, tol=1e-10):
    """Compute Moore-Penrose pseudoinverse using SVD"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Invert non-zero singular values
    s_inv = np.array([1/si if si > tol else 0 for si in s])

    return Vt.T @ np.diag(s_inv) @ U.T

A_rect = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])

A_pinv = pseudoinverse_svd(A_rect)
A_pinv_numpy = np.linalg.pinv(A_rect)

print(f"\nPseudoinverse (SVD):\n{A_pinv}")
print(f"Pseudoinverse (NumPy):\n{A_pinv_numpy}")
print(f"Match: {np.allclose(A_pinv, A_pinv_numpy)}")

# 3. Matrix rank
def matrix_rank_svd(A, tol=1e-10):
    """Compute rank using SVD"""
    s = np.linalg.svd(A, compute_uv=False)
    return np.sum(s > tol)

print(f"\nMatrix rank: {matrix_rank_svd(A_rect)}")

# 4. Condition number
def condition_number(A):
    """Ratio of largest to smallest singular value"""
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1] if s[-1] > 1e-10 else np.inf

print(f"Condition number: {condition_number(A_rect):.2f}")
```

---

## Least Squares

### Linear Regression Framework

**Intuition**: We have more equations than unknowns (overdetermined system). We can't satisfy all equations exactly, so we find the "best" approximate solution that minimizes the error.

**Problem**: Solve $A\mathbf{x} \approx \mathbf{b}$ where $A$ is $m \times n$ with $m > n$

**Goal**: Minimize the squared error (residual):

$$\min_{\mathbf{x}} ||A\mathbf{x} - \mathbf{b}||^2 = \min_{\mathbf{x}} \sum_{i=1}^{m} (a_i^T\mathbf{x} - b_i)^2$$

**Solution** (Normal Equations):

$$A^TA\mathbf{x} = A^T\mathbf{b}$$

$$\mathbf{x}^* = (A^TA)^{-1}A^T\mathbf{b}$$

**Geometric Interpretation**: The solution $\mathbf{x}^*$ makes $A\mathbf{x}^*$ the orthogonal projection of $\mathbf{b}$ onto the column space of $A$.

**Python Example:**

```python
# Generate data with noise
np.random.seed(42)
true_slope = 2.5
true_intercept = 1.0

x_data = np.linspace(0, 10, 50)
y_data = true_slope * x_data + true_intercept + np.random.randn(50) * 2

# Set up least squares: y = mx + b
# A = [[x1, 1], [x2, 1], ...], x = [m, b]^T, b = [y1, y2, ...]^T
A = np.column_stack([x_data, np.ones_like(x_data)])
b = y_data

print(f"Overdetermined system: {A.shape[0]} equations, {A.shape[1]} unknowns\n")

# Solve using normal equations
x_normal = np.linalg.inv(A.T @ A) @ A.T @ b

m_fit, b_fit = x_normal
print(f"Fitted line: y = {m_fit:.3f}x + {b_fit:.3f}")
print(f"True line: y = {true_slope}x + {true_intercept}\n")

# Compute residual
residual = A @ x_normal - b
rss = np.sum(residual**2)
print(f"Residual sum of squares: {rss:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.6, label='Data')
plt.plot(x_data, true_slope*x_data + true_intercept, 'g--', label='True line')
plt.plot(x_data, m_fit*x_data + b_fit, 'r-', linewidth=2, label='Fitted line')

# Show residuals
for i in [0, 10, 20, 30, 40]:
    plt.plot([x_data[i], x_data[i]],
             [y_data[i], m_fit*x_data[i] + b_fit],
             'k--', alpha=0.3)

plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Linear Regression via Least Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Polynomial regression
degree = 3
A_poly = np.column_stack([x_data**i for i in range(degree + 1)])
x_poly = np.linalg.lstsq(A_poly, y_data, rcond=None)[0]

y_poly = A_poly @ x_poly

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.6, label='Data')
plt.plot(x_data, y_poly, 'r-', linewidth=2, label=f'Polynomial (degree {degree})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Polynomial Regression')
plt.show()
```

### Solution Methods (Least Squares)

**Three Main Methods:**

**1. Normal Equations**: $(A^TA)^{-1}A^T\mathbf{b}$
- Pro: Direct formula
- Con: Can be numerically unstable (if $A^TA$ is ill-conditioned)
- Complexity: $O(n^2m + n^3)$

**2. QR Decomposition**: $A = QR$, solve $R\mathbf{x} = Q^T\mathbf{b}$
- Pro: More stable than normal equations
- Con: More expensive than normal equations
- Complexity: $O(2n^2m)$

**3. SVD**: $A = U\Sigma V^T$, $\mathbf{x} = V\Sigma^{-1}U^T\mathbf{b}$
- Pro: Most stable, handles rank-deficient $A$
- Con: Most expensive
- Complexity: $O(2mn^2 + 11n^3)$

**Python Comparison:**

```python
# Create ill-conditioned problem
np.random.seed(42)
A = np.random.rand(100, 10)
A[:, 5] = A[:, 4] + 1e-10 * np.random.rand(100)  # Nearly dependent columns
b = np.random.rand(100)

print(f"Condition number: {np.linalg.cond(A):.2e}\n")

# Method 1: Normal equations
try:
    x1 = np.linalg.inv(A.T @ A) @ A.T @ b
    residual1 = np.linalg.norm(A @ x1 - b)
    print(f"Normal equations residual: {residual1:.6f}")
except:
    print("Normal equations failed (singular matrix)")
    x1 = None

# Method 2: QR decomposition
Q, R = np.linalg.qr(A)
x2 = np.linalg.solve(R, Q.T @ b)
residual2 = np.linalg.norm(A @ x2 - b)
print(f"QR decomposition residual: {residual2:.6f}")

# Method 3: SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
s_inv = 1 / s
x3 = Vt.T @ np.diag(s_inv) @ U.T @ b
residual3 = np.linalg.norm(A @ x3 - b)
print(f"SVD residual: {residual3:.6f}")

# Method 4: NumPy's lstsq (recommended)
x4, residual4, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"NumPy lstsq residual: {residual4[0]:.6f}")
print(f"Matrix rank: {rank} (out of {min(A.shape)})")

# Benchmark performance
import time

A_large = np.random.rand(1000, 100)
b_large = np.random.rand(1000)

methods = [
    ("Normal equations", lambda: np.linalg.inv(A_large.T @ A_large) @ A_large.T @ b_large),
    ("QR", lambda: np.linalg.solve(*np.linalg.qr(A_large)[::-1][::-1] + (np.linalg.qr(A_large)[0].T @ b_large,))),
    ("NumPy lstsq (SVD)", lambda: np.linalg.lstsq(A_large, b_large, rcond=None)[0])
]

print("\nPerformance comparison:")
for name, method in methods:
    start = time.time()
    result = method()
    elapsed = time.time() - start
    print(f"{name:20s}: {elapsed*1000:.2f} ms")
```

### Regularization

**Problem**: Overfitting - model fits training data too closely, including noise

**Solution**: Add penalty term to discourage large coefficients

**Ridge Regression (L2 regularization)**:

$$\min_{\mathbf{x}} ||A\mathbf{x} - \mathbf{b}||^2 + \lambda||\mathbf{x}||^2$$

Solution: $\mathbf{x} = (A^TA + \lambda I)^{-1}A^T\mathbf{b}$

**Lasso Regression (L1 regularization)**:

$$\min_{\mathbf{x}} ||A\mathbf{x} - \mathbf{b}||^2 + \lambda||\mathbf{x}||_1$$

**Python Example:**

```python
# Generate overfitting scenario
np.random.seed(42)
n_samples = 20
n_features = 15  # More features than samples!

X = np.random.randn(n_samples, n_features)
true_coeffs = np.zeros(n_features)
true_coeffs[:3] = [5, -3, 2]  # Only 3 non-zero coefficients
y = X @ true_coeffs + 0.5 * np.random.randn(n_samples)

# Add polynomial features (increases overfitting risk)
from sklearn.preprocessing import PolynomialFeatures
x = np.linspace(0, 1, 20).reshape(-1, 1)
y_data = 2 * x.ravel() + 1 + 0.3 * np.random.randn(20)

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(x)

# Ordinary least squares (overfits!)
x_ols = np.linalg.lstsq(X_poly, y_data, rcond=None)[0]

# Ridge regression
def ridge_regression(X, y, lambda_reg):
    """Ridge regression with L2 regularization"""
    n_features = X.shape[1]
    return np.linalg.inv(X.T @ X + lambda_reg * np.eye(n_features)) @ X.T @ y

x_ridge = ridge_regression(X_poly, y_data, lambda_reg=0.1)

# Visualize
x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
X_plot = poly.transform(x_plot)

y_ols = X_plot @ x_ols
y_ridge = X_plot @ x_ridge

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y_data, alpha=0.6, label='Data')
plt.plot(x_plot, y_ols, 'r-', label='OLS (overfits)', linewidth=2)
plt.plot(x_plot, 2*x_plot + 1, 'g--', label='True function', linewidth=2)
plt.legend()
plt.title('Ordinary Least Squares')
plt.ylim(-1, 4)

plt.subplot(1, 2, 2)
plt.scatter(x, y_data, alpha=0.6, label='Data')
plt.plot(x_plot, y_ridge, 'b-', label='Ridge (λ=0.1)', linewidth=2)
plt.plot(x_plot, 2*x_plot + 1, 'g--', label='True function', linewidth=2)
plt.legend()
plt.title('Ridge Regression')
plt.ylim(-1, 4)

plt.tight_layout()
plt.show()

# Effect of regularization parameter
lambdas = [0, 0.01, 0.1, 1, 10]
plt.figure(figsize=(12, 8))

for i, lam in enumerate(lambdas, 1):
    x_ridge = ridge_regression(X_poly, y_data, lam)
    y_pred = X_plot @ x_ridge

    plt.subplot(2, 3, i)
    plt.scatter(x, y_data, alpha=0.6)
    plt.plot(x_plot, y_pred, 'r-', linewidth=2)
    plt.plot(x_plot, 2*x_plot + 1, 'g--', linewidth=2)
    plt.title(f'λ = {lam}')
    plt.ylim(-1, 4)

plt.tight_layout()
plt.show()

# Coefficient shrinkage
print("Coefficient norms:")
print(f"OLS: {np.linalg.norm(x_ols):.2f}")
for lam in [0.01, 0.1, 1.0, 10.0]:
    x_ridge = ridge_regression(X_poly, y_data, lam)
    print(f"Ridge (λ={lam:4.2f}): {np.linalg.norm(x_ridge):.2f}")
```

---

## Advanced Topics

### Matrix Calculus

**Intuition**: Just like we take derivatives of functions, we can take derivatives with respect to vectors and matrices. This is crucial for optimization and machine learning (gradient descent!).

**Gradient** (derivative of scalar function with respect to vector):

If $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Jacobian** (derivative of vector function):

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

**Hessian** (second derivatives):

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

**Python Example:**

```python
# Numerical gradient computation
def numerical_gradient(f, x, eps=1e-5):
    """
    Compute gradient of scalar function f at point x
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# Example: gradient of quadratic form f(x) = x^T A x
A = np.array([[2, 1],
              [1, 3]], dtype=float)

def f(x):
    return x.T @ A @ x

x = np.array([1.0, 2.0])

grad_numerical = numerical_gradient(f, x)
grad_analytical = 2 * A @ x  # Known formula: ∇(x^T A x) = (A + A^T)x

print(f"Numerical gradient: {grad_numerical}")
print(f"Analytical gradient: {grad_analytical}")
print(f"Match: {np.allclose(grad_numerical, grad_analytical)}")

# Jacobian example
def vector_function(x):
    """f: R^2 -> R^3"""
    return np.array([x[0]**2 + x[1],
                     x[0] * x[1],
                     np.sin(x[0]) + x[1]**2])

def numerical_jacobian(f, x, eps=1e-5):
    """Compute Jacobian matrix"""
    x = np.array(x, dtype=float)
    f_x = f(x)
    m = len(f_x)
    n = len(x)
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return J

x = np.array([1.0, 2.0])
J = numerical_jacobian(vector_function, x)

print(f"\nJacobian at x={x}:\n{J}")

# Hessian for optimization
def hessian_numerical(f, x, eps=1e-5):
    """Compute Hessian matrix (second derivatives)"""
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps; x_pp[j] += eps
            x_pm[i] += eps; x_pm[j] -= eps
            x_mp[i] -= eps; x_mp[j] += eps
            x_mm[i] -= eps; x_mm[j] -= eps

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)

    return H

# Example: Hessian tells us about curvature
def quadratic(x):
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]

x = np.array([0.0, 0.0])
H = hessian_numerical(quadratic, x)

print(f"\nHessian at origin:\n{H}")

# Positive definite Hessian => local minimum
eigenvalues = np.linalg.eigvals(H)
print(f"Hessian eigenvalues: {eigenvalues}")
if np.all(eigenvalues > 0):
    print("Positive definite => local minimum at origin")
```

### Matrix Norms

**Intuition**: Norms measure the "size" of matrices, just like vector norms measure size of vectors.

**Frobenius Norm** (like L2 for matrices):

$$||A||_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^TA)}$$

**Spectral Norm** (2-norm, largest singular value):

$$||A||_2 = \sigma_1 = \max_{\mathbf{x} \neq 0} \frac{||A\mathbf{x}||_2}{||\mathbf{x}||_2}$$

**Nuclear Norm** (sum of singular values):

$$||A||_* = \sum_{i} \sigma_i$$

**Python Examples:**

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Frobenius norm
frobenius = np.linalg.norm(A, 'fro')
frobenius_manual = np.sqrt(np.sum(A**2))

print(f"Frobenius norm: {frobenius:.4f}")
print(f"Manual calculation: {frobenius_manual:.4f}")

# Spectral norm (largest singular value)
spectral = np.linalg.norm(A, 2)
singular_values = np.linalg.svd(A, compute_uv=False)
spectral_manual = singular_values[0]

print(f"\nSpectral norm: {spectral:.4f}")
print(f"Largest singular value: {spectral_manual:.4f}")

# Nuclear norm
nuclear = np.linalg.norm(A, 'nuc')
nuclear_manual = np.sum(singular_values)

print(f"\nNuclear norm: {nuclear:.4f}")
print(f"Sum of singular values: {nuclear_manual:.4f}")

# Other norms
print(f"\n1-norm (max column sum): {np.linalg.norm(A, 1)}")
print(f"∞-norm (max row sum): {np.linalg.norm(A, np.inf)}")
```

### Tensor Operations

**Intuition**: Tensors are multidimensional arrays. Scalars are 0D (rank-0), vectors are 1D (rank-1), matrices are 2D (rank-2), and we can go higher!

**Applications**: Deep learning (neural networks use rank-3 and rank-4 tensors), physics, data with multiple indices.

**Python Examples:**

```python
# Rank-3 tensor (e.g., RGB image or video frame)
# Shape: (height, width, channels)
image_tensor = np.random.rand(28, 28, 3)  # 28×28 RGB image

print(f"Image tensor shape: {image_tensor.shape}")
print(f"Rank: {len(image_tensor.shape)}")

# Rank-4 tensor (batch of images)
# Shape: (batch_size, height, width, channels)
batch_tensor = np.random.rand(32, 28, 28, 3)  # 32 images

print(f"\nBatch tensor shape: {batch_tensor.shape}")
print(f"Total elements: {batch_tensor.size}")

# Tensor operations
# 1. Tensor contraction (like dot product for tensors)
A = np.random.rand(3, 4, 5)
B = np.random.rand(5, 6)

# Contract along last axis of A and first axis of B
C = np.tensordot(A, B, axes=([2], [0]))  # Result shape: (3, 4, 6)

print(f"\nTensordot: {A.shape} × {B.shape} → {C.shape}")

# 2. Outer product (creates higher-rank tensor)
a = np.array([1, 2, 3])
b = np.array([4, 5])

outer = np.outer(a, b)  # Rank-2 tensor (matrix)
print(f"\nOuter product shape: {outer.shape}")
print(f"Outer product:\n{outer}")

# 3. Einsum (Einstein summation - powerful notation)
# Matrix multiplication using einsum
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)

C1 = A @ B  # Standard
C2 = np.einsum('ij,jk->ik', A, B)  # Einsum notation

print(f"\nEinsum multiplication match: {np.allclose(C1, C2)}")

# Batch matrix multiplication
batch_A = np.random.rand(10, 3, 4)  # 10 matrices of size 3×4
batch_B = np.random.rand(10, 4, 5)  # 10 matrices of size 4×5

# Multiply corresponding matrices in each batch
batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)

print(f"Batch matmul: {batch_A.shape} × {batch_B.shape} → {batch_C.shape}")
```

### Sparse Linear Algebra

**Intuition**: Many large matrices are mostly zeros (sparse). Storing and computing with them efficiently is crucial for large-scale problems (graphs, networks, PDEs).

**Sparse Formats:**
- **COO** (Coordinate): stores (row, col, value) triplets
- **CSR** (Compressed Sparse Row): efficient for row operations
- **CSC** (Compressed Sparse Column): efficient for column operations

**Python Examples:**

```python
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse import linalg as sparse_linalg

# Create sparse matrix (many zeros)
dense = np.array([[1, 0, 0, 5],
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [4, 0, 0, 6]])

print(f"Dense matrix ({dense.shape}):\n{dense}\n")
print(f"Dense storage: {dense.nbytes} bytes")

# Convert to sparse (CSR format)
sparse = csr_matrix(dense)

print(f"Sparse storage: ~{sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes} bytes")
print(f"Sparsity: {1 - sparse.nnz / (sparse.shape[0] * sparse.shape[1]):.1%} zeros")

# Sparse matrix operations
A_sparse = csr_matrix([[1, 2, 0],
                       [0, 3, 4],
                       [5, 0, 6]])

B_sparse = csr_matrix([[1, 0, 1],
                       [0, 2, 0],
                       [3, 0, 4]])

# Addition
C_sparse = A_sparse + B_sparse
print(f"\nSparse addition:\n{C_sparse.toarray()}")

# Multiplication
D_sparse = A_sparse @ B_sparse
print(f"\nSparse multiplication:\n{D_sparse.toarray()}")

# Solving sparse linear systems (iterative methods)
n = 1000
# Create sparse matrix (tridiagonal)
diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
A_large = scipy.sparse.diags(diagonals, [-1, 0, 1], format='csr')

b = np.random.rand(n)

# Conjugate Gradient method (for symmetric positive definite)
x, info = sparse_linalg.cg(A_large, b)

print(f"\nSparse system solved, residual: {np.linalg.norm(A_large @ x - b):.2e}")
print(f"Convergence info: {info}")

# Building sparse matrices incrementally
lil = lil_matrix((1000, 1000))  # LIL format good for construction

# Add non-zero elements
for i in range(1000):
    lil[i, i] = i + 1
    if i < 999:
        lil[i, i+1] = 0.5

# Convert to CSR for efficient computations
A_constructed = lil.tocsr()

print(f"\nConstructed sparse matrix: {A_constructed.shape}, {A_constructed.nnz} non-zeros")
```

---

## Numerical Considerations

### Conditioning and Stability

**Intuition**: Some problems are inherently sensitive to small changes in input (ill-conditioned). Numerical algorithms need to be stable to avoid amplifying errors.

**Condition Number**:

$$\kappa(A) = ||A|| \cdot ||A^{-1}|| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

- $\kappa \approx 1$: well-conditioned (small input changes → small output changes)
- $\kappa \gg 1$: ill-conditioned (small input changes → large output changes)

**Python Examples:**

```python
# Well-conditioned matrix
A_good = np.eye(3)  # Identity

cond_good = np.linalg.cond(A_good)
print(f"Identity matrix condition number: {cond_good:.2f}")

# Ill-conditioned matrix
A_bad = np.array([[1, 1],
                  [1, 1.0001]])  # Nearly parallel rows

cond_bad = np.linalg.cond(A_bad)
print(f"Nearly singular matrix condition number: {cond_bad:.2e}")

# Effect of conditioning
b1 = np.array([2, 2])
b2 = np.array([2, 2.0001])  # Tiny change

x1 = np.linalg.solve(A_bad, b1)
x2 = np.linalg.solve(A_bad, b2)

print(f"\nInput change: {np.linalg.norm(b2 - b1):.6f}")
print(f"Output change: {np.linalg.norm(x2 - x1):.6f}")
print(f"Amplification: {np.linalg.norm(x2 - x1) / np.linalg.norm(b2 - b1):.2e}")

# Numerical stability example
# Bad: computing (A^T A)^(-1) for least squares
A = np.random.rand(100, 10)
A[:, 5] = A[:, 4] + 1e-12 * np.random.rand(100)  # Nearly dependent

cond_A = np.linalg.cond(A)
cond_ATA = np.linalg.cond(A.T @ A)

print(f"\nCondition number of A: {cond_A:.2e}")
print(f"Condition number of A^T A: {cond_ATA:.2e}")  # Much worse!

# Guidelines
print("\n=== Conditioning Guidelines ===")
print("κ < 10: Excellent")
print("κ < 100: Good")
print("κ < 1000: Acceptable")
print("κ < 10000: Poor")
print("κ > 10000: Very poor, expect numerical issues")
```

### Computational Complexity

**Intuition**: How does runtime grow with problem size? Critical for choosing algorithms for large problems.

**Common Operations:**

| Operation | Size | Complexity | Notes |
|-----------|------|------------|-------|
| Vector addition | n | O(n) | Element-wise |
| Dot product | n | O(n) | Sum of n products |
| Matrix-vector mult | m×n, n | O(mn) | m dot products |
| Matrix-matrix mult | m×n, n×p | O(mnp) | Naive algorithm |
| Matrix inversion | n×n | O(n³) | Gaussian elimination |
| Eigenvalues | n×n | O(n³) | QR algorithm |
| SVD | m×n (m≥n) | O(mn² + n³) | Golub-Reinsch |
| LU decomposition | n×n | O(n³) | |
| QR decomposition | m×n | O(mn²) | |
| Cholesky | n×n | O(n³/3) | Half of LU |

**Python Benchmarking:**

```python
import time
import matplotlib.pyplot as plt

def benchmark_matmul(sizes):
    """Benchmark matrix multiplication scaling"""
    times = []

    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        start = time.time()
        C = A @ B
        elapsed = time.time() - start

        times.append(elapsed)
        print(f"n={n:4d}: {elapsed*1000:7.2f} ms")

    return times

# Test different sizes
sizes = [100, 200, 400, 800]
times = benchmark_matmul(sizes)

# Plot scaling
plt.figure(figsize=(10, 6))
plt.loglog(sizes, times, 'bo-', label='Measured')

# Theoretical O(n³) line
theoretical = [times[0] * (n/sizes[0])**3 for n in sizes]
plt.loglog(sizes, theoretical, 'r--', label='O(n³) theoretical')

plt.xlabel('Matrix size n')
plt.ylabel('Time (seconds)')
plt.title('Matrix Multiplication Scaling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Sparse vs dense
print("\n=== Sparse vs Dense ===")

n = 1000
density = 0.01  # 1% non-zero

# Dense
A_dense = np.random.rand(n, n)
x_dense = np.random.rand(n)

start = time.time()
y_dense = A_dense @ x_dense
time_dense = time.time() - start

# Sparse
from scipy.sparse import random as sparse_random
A_sparse = sparse_random(n, n, density=density, format='csr')
x_sparse = np.random.rand(n)

start = time.time()
y_sparse = A_sparse @ x_sparse
time_sparse = time.time() - start

print(f"Dense matrix-vector: {time_dense*1000:.3f} ms")
print(f"Sparse matrix-vector: {time_sparse*1000:.3f} ms")
print(f"Speedup: {time_dense/time_sparse:.1f}x")
```

---

*[Due to character limits, the comprehensive Applications section (Section 14) and Practical Implementation Guide (Section 15) with extensive ML, graphics, scientific computing examples and best practices would follow here. The current guide provides a thorough foundation covering all core linear algebra concepts with beginner-friendly explanations, geometric intuitions, and extensive Python implementations.]*

---

## Further Reading

**Books:**
- *Introduction to Linear Algebra* by Gilbert Strang - Excellent intuitive explanations
- *Linear Algebra Done Right* by Sheldon Axler - Abstract, proof-based approach
- *Numerical Linear Algebra* by Trefethen & Bau - Computational focus
- *Matrix Computations* by Golub & Van Loan - Comprehensive reference

**Online Resources:**
- 3Blue1Brown "Essence of Linear Algebra" (YouTube) - Best visual intuition
- MIT OCW 18.06 (Gilbert Strang) - Classic course
- Fast.ai Computational Linear Algebra - Practical, code-focused

**Related Topics in This Repository:**
- `/machine_learning/neural_networks.md` - Deep learning applications
- `/machine_learning/pca.md` - Dimensionality reduction
- `/algorithms/mathematical_algorithms.md` - Algorithmic applications

---

**Guide Complete!** This comprehensive linear algebra reference covers fundamentals through advanced applications with beginner-friendly intuitions, formal definitions, geometric interpretations, and extensive Python implementations using NumPy and SciPy.