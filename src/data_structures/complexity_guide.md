# Complexity Analysis Deep Dive

## Overview

Complexity analysis is the study of the resources (time and space) required by an algorithm as the input size grows. This guide provides a comprehensive understanding of asymptotic notation, amortized analysis, space-time tradeoffs, and mathematical tools for analyzing algorithms.

## Table of Contents

1. [Asymptotic Notation](#asymptotic-notation)
2. [Big-O, Big-Θ, Big-Ω Explained](#big-o-big-θ-big-ω-explained)
3. [Common Time Complexities](#common-time-complexities)
4. [Space Complexity](#space-complexity)
5. [Amortized Analysis](#amortized-analysis)
6. [Space-Time Tradeoffs](#space-time-tradeoffs)
7. [Recurrence Relations](#recurrence-relations)
8. [Master Theorem](#master-theorem)
9. [Practical Analysis Tips](#practical-analysis-tips)

## Asymptotic Notation

Asymptotic notation describes how an algorithm's runtime or space requirements grow as the input size approaches infinity. It focuses on the dominant term and ignores constants.

### Why Asymptotic Notation?

```python
# Algorithm 1: 5n + 10 operations
def algorithm1(n):
    for i in range(n):
        for j in range(5):
            print(i, j)  # 5n operations
    for i in range(10):
        print(i)  # 10 operations

# Algorithm 2: 100n operations
def algorithm2(n):
    for i in range(100 * n):
        print(i)

# For small n: algorithm1 might be faster
# For large n: Both are O(n), constants don't matter
```

**Key insight**: Constants and lower-order terms become insignificant as $n$ grows large.

## Big-O, Big-Θ, Big-Ω Explained

### Big-O (Upper Bound)

$f(n) = O(g(n))$ means $f(n)$ grows **no faster** than $g(n)$ for large $n$.

**Formal definition**: There exist constants $c > 0$ and $n_0$ such that:
$$f(n) \leq c \cdot g(n) \text{ for all } n \geq n_0$$

```python
def linear_search(arr, target):
    """
    Best case: O(1) - found at first position
    Worst case: O(n) - found at last position or not found
    Big-O: O(n) - upper bound
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

**Usage**: Big-O describes the **worst-case** scenario or an upper bound on growth rate.

### Big-Ω (Lower Bound)

$f(n) = \Omega(g(n))$ means $f(n)$ grows **at least as fast** as $g(n)$ for large $n$.

**Formal definition**: There exist constants $c > 0$ and $n_0$ such that:
$$f(n) \geq c \cdot g(n) \text{ for all } n \geq n_0$$

```python
def find_all_pairs(arr):
    """
    Must examine all pairs: Ω(n²)
    Even in best case, must check all combinations
    """
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairs.append((arr[i], arr[j]))
    return pairs
```

**Usage**: Big-Ω describes the **best-case** scenario or a lower bound on growth rate.

### Big-Θ (Tight Bound)

$f(n) = \Theta(g(n))$ means $f(n)$ grows **at the same rate** as $g(n)$ for large $n$.

**Formal definition**: $f(n) = O(g(n))$ AND $f(n) = \Omega(g(n))$

```python
def sum_array(arr):
    """
    Always loops n times
    Best, average, worst case: Θ(n)
    """
    total = 0
    for num in arr:
        total += num
    return total

def bubble_sort(arr):
    """
    Best case: Ω(n) with optimization
    Worst case: O(n²)
    Average case: Θ(n²)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
```

**Usage**: Big-Θ describes the **exact** growth rate when best and worst cases have the same complexity.

### Comparison Summary

| Notation | Meaning | Analogy |
|----------|---------|---------|
| $O(g(n))$ | Upper bound (≤) | "At most" / "No worse than" |
| $\Omega(g(n))$ | Lower bound (≥) | "At least" / "No better than" |
| $\Theta(g(n))$ | Tight bound (=) | "Exactly" / "On the order of" |

```python
# Example: Binary search
def binary_search(arr, target):
    """
    Best case: Ω(1) - target at middle
    Worst case: O(log n) - target not found
    Average case: Θ(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # Ω(1)
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # O(log n)
```

## Common Time Complexities

### Complexity Hierarchy

From fastest to slowest:

$$O(1) < O(\log n) < O(\sqrt{n}) < O(n) < O(n \log n) < O(n^2) < O(n^3) < O(2^n) < O(n!)$$

### Visual Growth Rates

```python
import math

def demonstrate_growth(n=20):
    """Compare growth rates"""
    complexities = {
        'O(1)': lambda x: 1,
        'O(log n)': lambda x: math.log2(x) if x > 0 else 0,
        'O(n)': lambda x: x,
        'O(n log n)': lambda x: x * math.log2(x) if x > 0 else 0,
        'O(n²)': lambda x: x * x,
        'O(2ⁿ)': lambda x: 2 ** x if x < 30 else float('inf'),
    }

    for name, func in complexities.items():
        print(f"{name:12} | n=10: {func(10):10.0f} | n=20: {func(20):15.0f}")

# Output:
# O(1)         | n=10:          1 | n=20:               1
# O(log n)     | n=10:          3 | n=20:               4
# O(n)         | n=10:         10 | n=20:              20
# O(n log n)   | n=10:         33 | n=20:              86
# O(n²)        | n=10:        100 | n=20:             400
# O(2ⁿ)        | n=10:       1024 | n=20:         1048576
```

### Common Complexities with Examples

#### $O(1)$ - Constant Time

```python
def get_first_element(arr):
    """Always takes same time regardless of input size"""
    return arr[0] if arr else None

def hash_lookup(hash_table, key):
    """Hash table lookup (average case)"""
    return hash_table.get(key)
```

#### $O(\log n)$ - Logarithmic Time

```python
def binary_search(arr, target):
    """Divides problem in half each iteration"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Tree height in balanced BST
def tree_height(n_nodes):
    """Height of balanced tree with n nodes"""
    import math
    return math.ceil(math.log2(n_nodes + 1))
```

#### $O(n)$ - Linear Time

```python
def find_max(arr):
    """Must examine every element once"""
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

def sum_linked_list(head):
    """Traverse entire linked list"""
    total = 0
    current = head
    while current:
        total += current.val
        current = current.next
    return total
```

#### $O(n \log n)$ - Linearithmic Time

```python
def merge_sort(arr):
    """Divide and conquer with merge - O(n log n)"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # T(n/2)
    right = merge_sort(arr[mid:])   # T(n/2)

    return merge(left, right)       # O(n)

def heap_sort(arr):
    """Build heap + n extractions - O(n log n)"""
    import heapq
    heapq.heapify(arr)  # O(n)
    return [heapq.heappop(arr) for _ in range(len(arr))]  # n * O(log n)
```

#### $O(n^2)$ - Quadratic Time

```python
def bubble_sort(arr):
    """Nested loops over array"""
    n = len(arr)
    for i in range(n):           # n iterations
        for j in range(n - 1):   # n iterations
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def find_all_pairs(arr):
    """All combinations of pairs"""
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairs.append((arr[i], arr[j]))
    return pairs
```

#### $O(2^n)$ - Exponential Time

```python
def fibonacci_recursive(n):
    """Naive recursive Fibonacci - O(2ⁿ)"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def generate_subsets(arr):
    """All subsets of array - O(2ⁿ)"""
    result = []

    def backtrack(index, current):
        if index == len(arr):
            result.append(current[:])
            return

        # Don't include arr[index]
        backtrack(index + 1, current)

        # Include arr[index]
        current.append(arr[index])
        backtrack(index + 1, current)
        current.pop()

    backtrack(0, [])
    return result  # 2ⁿ subsets
```

#### $O(n!)$ - Factorial Time

```python
def generate_permutations(arr):
    """All permutations - O(n!)"""
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()

    backtrack([], arr)
    return result  # n! permutations
```

## Space Complexity

Space complexity measures the **memory** required by an algorithm as a function of input size.

### Types of Space

1. **Input space**: Space for input data (usually excluded from analysis)
2. **Auxiliary space**: Extra space used by algorithm
3. **Output space**: Space for output (sometimes excluded)

```python
def reverse_array_in_place(arr):
    """
    Time: O(n)
    Space: O(1) - only uses a few variables
    """
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

def reverse_array_new(arr):
    """
    Time: O(n)
    Space: O(n) - creates new array
    """
    return arr[::-1]

def fibonacci_memoized(n, memo=None):
    """
    Time: O(n)
    Space: O(n) - recursion stack + memo dictionary
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]
```

### Space Complexity Examples

```python
# O(1) space - Constant
def find_sum(arr):
    total = 0
    for num in arr:
        total += num
    return total

# O(log n) space - Recursion depth in binary search
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# O(n) space - Additional array
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # Creates new arrays
    right = merge_sort(arr[mid:])  # O(n) space total

    return merge(left, right)

# O(n²) space - 2D matrix
def create_adjacency_matrix(n):
    return [[0] * n for _ in range(n)]
```

## Amortized Analysis

Amortized analysis gives the **average performance** of each operation in a sequence of operations, even if individual operations are expensive.

### Why Amortized Analysis?

Some operations are occasionally expensive but infrequent enough that their cost is "amortized" over many cheap operations.

### Dynamic Array Example

```python
class DynamicArray:
    """
    Array that grows automatically
    Individual append can be O(n) when resizing
    But amortized cost is O(1)
    """
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.array = [None] * self.capacity

    def append(self, item):
        """
        Worst case: O(n) - when resize needed
        Amortized: O(1) - most appends are O(1)
        """
        if self.size == self.capacity:
            self._resize()  # O(n) operation

        self.array[self.size] = item
        self.size += 1

    def _resize(self):
        """Double capacity - O(n)"""
        self.capacity *= 2
        new_array = [None] * self.capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array

# Analysis of n appends:
# - Most appends: O(1) - just store value
# - Resize happens at sizes: 1, 2, 4, 8, 16, ..., n
# - Resize costs: 1 + 2 + 4 + 8 + ... + n = 2n - 1 = O(n)
# - Total cost: O(n) for resizes + O(n) for appends = O(2n)
# - Amortized cost per append: O(2n) / n = O(1)
```

### Three Methods of Amortized Analysis

#### 1. Aggregate Method

Calculate total cost of $n$ operations, divide by $n$.

```python
def demonstrate_aggregate():
    """
    Analyze sequence of push/pop operations on stack
    """
    # Consider n operations: push and multi-pop
    # push: O(1)
    # multi-pop(k): O(k) - pop k elements

    # Key insight: Each element pushed once, popped once
    # Total pushes: ≤ n
    # Total pops: ≤ n (can't pop more than pushed)
    # Total cost: O(2n) = O(n)
    # Amortized cost per operation: O(n) / n = O(1)
    pass
```

#### 2. Accounting Method

Assign different charges to operations such that the charged amount covers the actual cost.

```python
class DynamicArrayAccounting:
    """
    Charge $3 for each append:
    - $1 for immediate insertion
    - $1 saved for copying this element during resize
    - $1 saved for copying previously inserted element

    When resize happens, we've saved enough to cover O(n) cost
    """
    def append(self, item):
        # Charge: $3 (amortized O(1))
        # Actual work: $1 for insertion
        # Bank: $2 for future resize
        pass
```

#### 3. Potential Method

Define a potential function $\Phi$ representing stored energy in data structure.

**Amortized cost** = Actual cost + Change in potential

```python
def potential_analysis_dynamic_array():
    """
    Potential function: Φ(h) = 2 * size - capacity

    After resize:
    - size = n, capacity = 2n
    - Φ = 2n - 2n = 0

    Before next resize:
    - size = 2n, capacity = 2n
    - Φ = 4n - 2n = 2n

    Resize operation:
    - Actual cost: n (copy n elements)
    - Potential change: 0 - 2n = -2n
    - Amortized cost: n + (-2n) = -n... wait, need better analysis

    Corrected: Φ(h) = 2 * (size - capacity/2)
    This gives amortized O(1) for append
    """
    pass
```

### Real-World Amortized Analysis Examples

```python
# Union-Find with path compression
class UnionFind:
    """
    Single operation: O(log n) worst case
    Amortized over m operations: O(α(n)) where α is inverse Ackermann
    α(n) < 5 for all practical n
    """
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

# Splay tree
class SplayTree:
    """
    Single operation: O(n) worst case
    Amortized: O(log n) per operation
    """
    def splay(self, node):
        # Rotate node to root
        # Frequently accessed nodes move closer to root
        pass
```

## Space-Time Tradeoffs

Often, we can trade space for time or vice versa. Understanding these tradeoffs is crucial for optimization.

### Common Tradeoff Patterns

#### 1. Memoization (Space for Time)

```python
# Without memoization: O(2ⁿ) time, O(n) space
def fib_slow(n):
    if n <= 1:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

# With memoization: O(n) time, O(n) space
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

# Iterative (optimal): O(n) time, O(1) space
def fib_iterative(n):
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr
```

#### 2. Precomputation (Space for Time)

```python
# Without precomputation
def is_prime(n):
    """O(√n) per query"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# With precomputation (Sieve of Eratosthenes)
def sieve_of_eratosthenes(limit):
    """
    Precompute: O(n log log n) time, O(n) space
    Query: O(1) time
    """
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    return is_prime

# Usage
primes = sieve_of_eratosthenes(1000000)  # One-time cost
print(primes[999983])  # O(1) query
```

#### 3. Hash Tables vs Arrays

```python
# Array: O(1) space overhead, O(n) search
def find_in_array(arr, target):
    return target in arr  # O(n)

# Hash set: O(n) space, O(1) search (average)
def find_in_set(s, target):
    return target in s  # O(1) average

# Two-sum problem comparison
def two_sum_brute_force(arr, target):
    """O(n²) time, O(1) space"""
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
    return None

def two_sum_hash(arr, target):
    """O(n) time, O(n) space"""
    seen = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None
```

#### 4. Compression vs Speed

```python
class CompressedStorage:
    """
    More space-efficient but slower access
    """
    def __init__(self, data):
        import zlib
        self.compressed = zlib.compress(data.encode())

    def get_data(self):
        import zlib
        return zlib.decompress(self.compressed).decode()

class FastStorage:
    """
    Uses more space but faster access
    """
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data
```

#### 5. Caching (Space for Time)

```python
class URLShortener:
    """Space-time tradeoff in caching"""

    def __init__(self):
        self.url_to_short = {}  # O(n) space
        self.short_to_url = {}  # O(n) space
        self.counter = 0

    def shorten(self, url):
        """O(1) time with O(n) space"""
        if url in self.url_to_short:
            return self.url_to_short[url]

        short = self._encode(self.counter)
        self.counter += 1

        self.url_to_short[url] = short
        self.short_to_url[short] = url
        return short

    def expand(self, short):
        """O(1) time"""
        return self.short_to_url.get(short)

    def _encode(self, num):
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        while num:
            result.append(chars[num % 62])
            num //= 62
        return ''.join(reversed(result)) or '0'
```

### Decision Framework

When choosing between time and space:

1. **If queries >> updates**: Precompute (space for time)
2. **If space is limited**: Use time-intensive algorithms
3. **If real-time response critical**: Cache/precompute
4. **If processing batch data**: Optimize for time, use more space

## Recurrence Relations

Recurrence relations express the runtime of recursive algorithms.

### Common Forms

#### 1. Linear Recurrence

$$T(n) = T(n-1) + O(1)$$

**Solution**: $T(n) = O(n)$

```python
def factorial(n):
    """T(n) = T(n-1) + O(1) = O(n)"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def linear_search_recursive(arr, target, index=0):
    """T(n) = T(n-1) + O(1) = O(n)"""
    if index >= len(arr):
        return -1
    if arr[index] == target:
        return index
    return linear_search_recursive(arr, target, index + 1)
```

#### 2. Binary Recurrence (Divide by 2)

$$T(n) = T(n/2) + O(1)$$

**Solution**: $T(n) = O(\log n)$

```python
def binary_search(arr, target, left, right):
    """T(n) = T(n/2) + O(1) = O(log n)"""
    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)

def find_power_of_two(n):
    """T(n) = T(n/2) + O(1) = O(log n)"""
    if n <= 1:
        return 0
    return 1 + find_power_of_two(n // 2)
```

#### 3. Tree Recurrence

$$T(n) = 2T(n/2) + O(n)$$

**Solution**: $T(n) = O(n \log n)$

```python
def merge_sort(arr):
    """
    T(n) = 2T(n/2) + O(n)
    Solves to O(n log n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])      # T(n/2)
    right = merge_sort(arr[mid:])     # T(n/2)

    return merge(left, right)         # O(n)

def merge(left, right):
    """O(n) - merge two sorted arrays"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

#### 4. Multiple Branches

$$T(n) = T(n-1) + T(n-2) + O(1)$$

**Solution**: $T(n) = O(2^n)$ (more precisely $O(\phi^n)$ where $\phi = \frac{1+\sqrt{5}}{2}$)

```python
def fibonacci(n):
    """
    T(n) = T(n-1) + T(n-2) + O(1)
    Solves to O(2ⁿ)
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### Solving Recurrences

#### Method 1: Substitution Method

Guess the solution and prove by induction.

```python
# Example: T(n) = 2T(n/2) + n
# Guess: T(n) = O(n log n)

# Proof:
# Base case: T(1) = 1 ≤ c * 1 * log(1) = 0... adjust base
# T(1) = 1 ≤ c (constant)
#
# Inductive step: Assume T(k) ≤ ck log k for k < n
# T(n) = 2T(n/2) + n
#      ≤ 2(c(n/2)log(n/2)) + n
#      = cn log(n/2) + n
#      = cn(log n - 1) + n
#      = cn log n - cn + n
#      = cn log n - (c-1)n
#      ≤ cn log n  (for c ≥ 1)
```

#### Method 2: Recursion Tree Method

```python
# T(n) = 2T(n/2) + n
#
# Tree visualization:
#                    n             = n
#                  /   \
#               n/2     n/2        = n
#              /  \    /  \
#           n/4  n/4 n/4  n/4      = n
#           ...
#
# Height: log₂ n
# Work per level: n
# Total: n * log n = O(n log n)

def visualize_recursion_tree():
    """
    Level 0: n                    (1 node)
    Level 1: n/2 + n/2           (2 nodes)
    Level 2: n/4 + n/4 + n/4 + n/4  (4 nodes)
    ...
    Level h: 2^h nodes of size n/2^h

    Total levels: log₂ n
    Work per level: 2^h * (n/2^h) = n
    Total work: n * log₂ n
    """
    pass
```

#### Method 3: Master Theorem (see next section)

## Master Theorem

The Master Theorem provides a cookbook method for solving recurrences of the form:

$$T(n) = aT(n/b) + f(n)$$

where:
- $a \geq 1$: number of subproblems
- $b > 1$: factor by which subproblem size decreases
- $f(n)$: cost of work done outside recursive calls

### Three Cases

#### Case 1: Work dominated by leaves

If $f(n) = O(n^c)$ where $c < \log_b a$:

$$T(n) = \Theta(n^{\log_b a})$$

```python
def example_case1(n):
    """
    T(n) = 8T(n/2) + n²

    a = 8, b = 2, f(n) = n²
    log_b a = log₂ 8 = 3
    f(n) = n² = O(n²) where 2 < 3

    By Case 1: T(n) = Θ(n³)
    """
    if n <= 1:
        return 1

    # 8 recursive calls on n/2
    result = 0
    for _ in range(8):
        result += example_case1(n // 2)

    # n² work outside recursion
    for i in range(n):
        for j in range(n):
            result += 1

    return result
```

#### Case 2: Work balanced at all levels

If $f(n) = \Theta(n^c)$ where $c = \log_b a$:

$$T(n) = \Theta(n^c \log n) = \Theta(n^{\log_b a} \log n)$$

```python
def merge_sort(arr):
    """
    T(n) = 2T(n/2) + n

    a = 2, b = 2, f(n) = n
    log_b a = log₂ 2 = 1
    f(n) = n = Θ(n¹) where 1 = 1

    By Case 2: T(n) = Θ(n log n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def binary_tree_traversal(root):
    """
    T(n) = 2T(n/2) + O(1)

    a = 2, b = 2, f(n) = 1
    log_b a = 1
    f(n) = O(1) = O(n⁰) where 0 < 1

    Actually Case 1: T(n) = Θ(n)
    """
    if not root:
        return

    binary_tree_traversal(root.left)
    binary_tree_traversal(root.right)
```

#### Case 3: Work dominated by root

If $f(n) = \Omega(n^c)$ where $c > \log_b a$ AND $af(n/b) \leq kf(n)$ for some $k < 1$:

$$T(n) = \Theta(f(n))$$

```python
def example_case3(n):
    """
    T(n) = 2T(n/2) + n²

    a = 2, b = 2, f(n) = n²
    log_b a = log₂ 2 = 1
    f(n) = n² = Ω(n²) where 2 > 1

    Regularity: 2(n/2)² = n²/2 ≤ kn² for k = 1/2 < 1 ✓

    By Case 3: T(n) = Θ(n²)
    """
    if n <= 1:
        return 1

    result = example_case3(n // 2) + example_case3(n // 2)

    # n² work outside recursion
    for i in range(n):
        for j in range(n):
            result += 1

    return result
```

### Master Theorem Examples

```python
# Example 1: Binary search
# T(n) = T(n/2) + O(1)
# a=1, b=2, f(n)=1, log₂1 = 0, f(n)=Θ(n⁰)
# Case 2: T(n) = Θ(log n)

# Example 2: Merge sort
# T(n) = 2T(n/2) + O(n)
# a=2, b=2, f(n)=n, log₂2 = 1, f(n)=Θ(n¹)
# Case 2: T(n) = Θ(n log n)

# Example 3: Karatsuba multiplication
# T(n) = 3T(n/2) + O(n)
# a=3, b=2, f(n)=n, log₂3 ≈ 1.585, f(n)=O(n¹) where 1 < 1.585
# Case 1: T(n) = Θ(n^1.585) = Θ(n^(log₂3))

# Example 4: Strassen's matrix multiplication
# T(n) = 7T(n/2) + O(n²)
# a=7, b=2, f(n)=n², log₂7 ≈ 2.807, f(n)=O(n²) where 2 < 2.807
# Case 1: T(n) = Θ(n^2.807) = Θ(n^(log₂7))

# Example 5: Linear time splitting
# T(n) = T(n/2) + O(n)
# a=1, b=2, f(n)=n, log₂1 = 0, f(n)=Ω(n¹) where 1 > 0
# Regularity: 1 * (n/2) = n/2 ≤ (1/2)n ✓
# Case 3: T(n) = Θ(n)
```

### When Master Theorem Doesn't Apply

```python
# T(n) = 2T(n/2) + n log n
# a=2, b=2, f(n)=n log n, log₂2 = 1
# f(n) = n log n is neither O(n^c) for c<1 nor Ω(n^c) for c>1
# Master theorem doesn't apply directly
# Solution by recursion tree: Θ(n log² n)

# T(n) = T(n-1) + n
# Not in the form T(n) = aT(n/b) + f(n)
# Master theorem doesn't apply
# Solution by substitution: Θ(n²)

# T(n) = 2ⁿT(n/2) + nⁿ
# 'a' is not constant
# Master theorem doesn't apply
```

## Practical Analysis Tips

### 1. Identify the Input Size

```python
# Array problems: n = len(array)
def find_max(arr):  # O(n) where n = len(arr)
    return max(arr)

# Matrix problems: n = rows, m = cols
def matrix_search(matrix, target):  # O(n*m)
    for row in matrix:
        if target in row:
            return True
    return False

# Graph problems: V = vertices, E = edges
def bfs(graph, start):  # O(V + E)
    visited = set()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex])
```

### 2. Count Basic Operations

```python
def analyze_operations(n):
    # O(1) - constant operations
    x = 5
    y = 10
    z = x + y

    # O(n) - single loop
    for i in range(n):
        print(i)

    # O(n²) - nested loops
    for i in range(n):
        for j in range(n):
            print(i, j)

    # O(n) - sequential loops add
    for i in range(n):
        print(i)
    for j in range(n):
        print(j)
    # Total: O(n) + O(n) = O(n)

    # O(log n) - dividing problem
    while n > 1:
        n //= 2
```

### 3. Watch for Hidden Complexity

```python
# Looks O(n) but is O(n²)!
def join_strings(strings):
    result = ""
    for s in strings:  # n iterations
        result += s    # O(n) string concatenation
    return result      # Total: O(n²)

# Correct O(n) version
def join_strings_efficient(strings):
    return ''.join(strings)  # O(n)

# Looks O(n) but is O(n²)!
def check_membership(arr, items):
    for item in items:     # n iterations
        if item in arr:    # O(n) list search
            return True
    return False           # Total: O(n²)

# Correct O(n) version
def check_membership_efficient(arr, items):
    arr_set = set(arr)     # O(n)
    for item in items:     # n iterations
        if item in arr_set:  # O(1) set search
            return True
    return False           # Total: O(n)
```

### 4. Best, Average, Worst Cases

```python
def quick_sort(arr):
    """
    Best case: O(n log n) - balanced partitions
    Average case: O(n log n) - random pivots
    Worst case: O(n²) - already sorted with bad pivot
    Space: O(log n) - recursion depth
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def linear_search(arr, target):
    """
    Best case: O(1) - found at first position
    Average case: O(n/2) = O(n)
    Worst case: O(n) - not found or at last position
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

### 5. Drop Constants and Lower-Order Terms

```python
# f(n) = 3n² + 5n + 10
# Drop constants: n² + n + 1
# Drop lower terms: n²
# Big-O: O(n²)

def example(n):
    # 3n operations
    for i in range(n):
        for _ in range(3):
            print(i)

    # 5n operations
    for i in range(n):
        for _ in range(5):
            print(i)

    # 10 operations
    for _ in range(10):
        print("hello")

    # Total: 3n + 5n + 10 = 8n + 10 = O(n)
```

### 6. Analyze Recursive Code

```python
def analyze_recursive(n):
    """
    1. Find recurrence relation
    2. Determine base case
    3. Solve recurrence
    """
    # Base case
    if n <= 1:
        return 1  # O(1)

    # Recursive case
    return (
        analyze_recursive(n - 1) +  # T(n-1)
        analyze_recursive(n - 1) +  # T(n-1)
        n                           # O(n)
    )

    # Recurrence: T(n) = 2T(n-1) + O(n)
    # Solution: O(2ⁿ)
```

### 7. Practical Tips

1. **Start with brute force**: Understand the problem first
2. **Identify bottlenecks**: What operation is slowest?
3. **Look for patterns**: Two pointers, sliding window, etc.
4. **Consider data structures**: Can hash table help? Tree? Graph?
5. **Trade space for time**: Memoization, precomputation
6. **Optimize gradually**: Don't optimize prematurely
7. **Test edge cases**: Empty input, single element, large input
8. **Measure in practice**: Asymptotic analysis isn't everything

## Summary

### Key Takeaways

1. **Big-O** (O): Upper bound - worst case
2. **Big-Ω** (Ω): Lower bound - best case
3. **Big-Θ** (Θ): Tight bound - exact growth rate

4. **Common complexities**: $O(1) < O(\log n) < O(n) < O(n \log n) < O(n^2) < O(2^n) < O(n!)$

5. **Amortized analysis**: Average cost over sequence of operations

6. **Space-time tradeoffs**: Often can trade one for the other

7. **Recurrence relations**: Express recursive algorithm complexity

8. **Master theorem**: Solve divide-and-conquer recurrences

### Quick Reference

```python
# Time Complexity Cheat Sheet
O(1)        # Constant      - Array access, hash lookup
O(log n)    # Logarithmic   - Binary search, balanced tree ops
O(n)        # Linear        - Array scan, linear search
O(n log n)  # Linearithmic  - Merge sort, heap sort
O(n²)       # Quadratic     - Nested loops, bubble sort
O(n³)       # Cubic         - Triple nested loops
O(2ⁿ)       # Exponential   - Recursive Fibonacci, subsets
O(n!)       # Factorial     - Permutations, TSP brute force

# Space Complexity Cheat Sheet
O(1)        # Constant      - Few variables
O(log n)    # Logarithmic   - Recursion depth in binary search
O(n)        # Linear        - Additional array, hash table
O(n²)       # Quadratic     - 2D matrix
```

## Related Topics

- [Data Structures README](README.md) - Overview of all structures
- [Implementation Patterns](implementation_patterns.md) - Common coding patterns
- [Arrays](arrays.md), [Trees](trees.md), [Graphs](graphs.md) - Specific data structures

## Practice Problems

1. Analyze time complexity of nested loops with dependencies
2. Calculate amortized cost of operations on dynamic array
3. Solve recurrence relations for custom recursive algorithms
4. Apply Master theorem to divide-and-conquer algorithms
5. Identify space-time tradeoff opportunities in given code
6. Compare best/average/worst case for various algorithms
7. Optimize algorithms by choosing better data structures

Remember: Complexity analysis is a tool for understanding algorithm behavior. In practice, consider constants, cache effects, and real-world constraints alongside asymptotic analysis.
