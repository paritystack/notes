# Mathematical Algorithms

## Overview
Mathematical algorithms are essential for competitive programming and solving number theory problems. This guide covers fundamental mathematical techniques used in algorithm design.

## 1. Prime Algorithms

### Sieve of Eratosthenes
Efficiently finds all prime numbers up to n.

**Time Complexity**: O(n log log n)
**Space Complexity**: O(n)

```python
def sieve_of_eratosthenes(n):
    """Find all primes up to n"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark all multiples as not prime
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]

# Example: Find all primes up to 30
primes = sieve_of_eratosthenes(30)
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

### Segmented Sieve
For finding primes in a range [L, R] where R - L is small but R can be large.

```python
def segmented_sieve(L, R):
    """Find all primes in range [L, R]"""
    limit = int(R**0.5) + 1
    # First get all primes up to sqrt(R)
    base_primes = sieve_of_eratosthenes(limit)

    # Create array for [L, R]
    size = R - L + 1
    is_prime = [True] * size

    for prime in base_primes:
        # Find first multiple of prime in [L, R]
        start = max(prime * prime, ((L + prime - 1) // prime) * prime)

        for j in range(start, R + 1, prime):
            is_prime[j - L] = False

    # Handle edge cases
    if L == 1:
        is_prime[0] = False

    return [L + i for i in range(size) if is_prime[i]]
```

### Primality Testing

**Trial Division** - O(√n)
```python
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

**Miller-Rabin (Probabilistic)** - O(k log³n)
```python
def miller_rabin(n, k=5):
    """Miller-Rabin primality test"""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Test k times
    import random
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True
```

## 2. GCD and LCM

### Euclidean Algorithm
**Time Complexity**: O(log min(a, b))

```python
def gcd(a, b):
    """Greatest Common Divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple"""
    return (a * b) // gcd(a, b)

# Extended Euclidean Algorithm
def extended_gcd(a, b):
    """Returns (gcd, x, y) where ax + by = gcd(a, b)"""
    if b == 0:
        return a, 1, 0

    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1

    return gcd, x, y

# Example: Find coefficients such that 30x + 20y = gcd(30, 20)
gcd, x, y = extended_gcd(30, 20)
# gcd = 10, 30*1 + 20*(-1) = 10
```

### Applications

**Finding GCD of array**
```python
from functools import reduce

def gcd_array(arr):
    return reduce(gcd, arr)
```

**Coprime numbers**
```python
def are_coprime(a, b):
    return gcd(a, b) == 1
```

## 3. Modular Arithmetic

### Basic Operations

```python
MOD = 10**9 + 7

def mod_add(a, b, mod=MOD):
    return (a + b) % mod

def mod_sub(a, b, mod=MOD):
    return (a - b + mod) % mod

def mod_mul(a, b, mod=MOD):
    return (a * b) % mod
```

### Modular Inverse

**Using Extended GCD**
```python
def mod_inverse(a, mod):
    """Find modular inverse of a under modulo mod"""
    gcd, x, y = extended_gcd(a, mod)
    if gcd != 1:
        return None  # Inverse doesn't exist
    return (x % mod + mod) % mod
```

**Using Fermat's Little Theorem** (when mod is prime)
```python
def mod_inverse_fermat(a, mod):
    """Modular inverse using Fermat's little theorem (mod must be prime)"""
    return pow(a, mod - 2, mod)
```

**Modular Division**
```python
def mod_div(a, b, mod=MOD):
    """Compute (a / b) % mod"""
    return mod_mul(a, mod_inverse_fermat(b, mod), mod)
```

### Modular Exponentiation

```python
def mod_pow(base, exp, mod):
    """Compute (base^exp) % mod efficiently"""
    result = 1
    base = base % mod

    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod

    return result

# Python built-in is optimized
result = pow(base, exp, mod)
```

## 4. Fast Exponentiation

### Binary Exponentiation
**Time Complexity**: O(log n)

```python
def fast_power(base, exp):
    """Calculate base^exp efficiently"""
    if exp == 0:
        return 1

    half = fast_power(base, exp // 2)

    if exp % 2 == 0:
        return half * half
    else:
        return half * half * base

# Iterative version
def fast_power_iterative(base, exp):
    result = 1
    while exp > 0:
        if exp & 1:  # If exp is odd
            result *= base
        base *= base
        exp >>= 1
    return result
```

### Matrix Exponentiation
Used for solving recurrence relations.

```python
def matrix_multiply(A, B, mod=None):
    """Multiply two matrices"""
    n = len(A)
    C = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
                if mod:
                    C[i][j] %= mod
    return C

def matrix_power(matrix, n, mod=None):
    """Calculate matrix^n using binary exponentiation"""
    size = len(matrix)
    # Identity matrix
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    while n > 0:
        if n & 1:
            result = matrix_multiply(result, matrix, mod)
        matrix = matrix_multiply(matrix, matrix, mod)
        n >>= 1

    return result

# Example: Calculate Fibonacci using matrix exponentiation
def fibonacci_matrix(n, mod=None):
    """Calculate nth Fibonacci number in O(log n)"""
    if n <= 1:
        return n

    # Transformation matrix [[1, 1], [1, 0]]
    matrix = [[1, 1], [1, 0]]
    result = matrix_power(matrix, n - 1, mod)

    return result[0][0]
```

## 5. Number Theory Essentials

### Chinese Remainder Theorem

```python
def chinese_remainder_theorem(remainders, moduli):
    """
    Find x such that:
    x ≡ remainders[0] (mod moduli[0])
    x ≡ remainders[1] (mod moduli[1])
    ...
    Moduli must be pairwise coprime
    """
    total = 0
    prod = 1
    for m in moduli:
        prod *= m

    for r, m in zip(remainders, moduli):
        p = prod // m
        total += r * mod_inverse(p, m) * p

    return total % prod
```

### Euler's Totient Function

```python
def euler_totient(n):
    """Count numbers from 1 to n that are coprime with n"""
    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            # Remove all factors of p
            while n % p == 0:
                n //= p
            # Apply formula: φ(n) = n * (1 - 1/p)
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result

# Compute totient for all numbers up to n
def euler_totient_sieve(n):
    phi = list(range(n + 1))  # phi[i] = i initially

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    return phi
```

### Prime Factorization

```python
def prime_factors(n):
    """Return list of prime factors with their powers"""
    factors = []
    d = 2

    while d * d <= n:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append((d, count))
        d += 1

    if n > 1:
        factors.append((n, 1))

    return factors

def count_divisors(n):
    """Count number of divisors of n"""
    factors = prime_factors(n)
    count = 1
    for prime, power in factors:
        count *= (power + 1)
    return count

def sum_of_divisors(n):
    """Calculate sum of all divisors of n"""
    factors = prime_factors(n)
    result = 1
    for prime, power in factors:
        # Sum of geometric series: (p^(k+1) - 1) / (p - 1)
        result *= (pow(prime, power + 1) - 1) // (prime - 1)
    return result
```

### Combinatorics

```python
def factorial_mod(n, mod):
    """Calculate n! % mod"""
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result

def nCr_mod(n, r, mod):
    """Calculate C(n, r) % mod using modular inverse"""
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1

    # Calculate n! / (r! * (n-r)!)
    numerator = factorial_mod(n, mod)
    denominator = (factorial_mod(r, mod) * factorial_mod(n - r, mod)) % mod

    return (numerator * mod_inverse_fermat(denominator, mod)) % mod

# Pascal's Triangle approach for small values
def nCr_pascal(n, r):
    """Calculate C(n, r) using Pascal's triangle"""
    if r > n - r:
        r = n - r

    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)

    return result
```

## Common Problem Patterns

### 1. Counting Problems with Modular Arithmetic
```python
def count_ways(n):
    """Example: Count ways modulo 10^9+7"""
    MOD = 10**9 + 7
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        dp[i] = (dp[i-1] * 2) % MOD

    return dp[n]
```

### 2. Fast Fibonacci
```python
def fibonacci_fast(n):
    """Calculate nth Fibonacci in O(log n)"""
    if n <= 1:
        return n

    def fib_pair(n):
        """Returns (F(n), F(n+1))"""
        if n == 0:
            return (0, 1)

        m = n // 2
        f_m, f_m1 = fib_pair(m)

        c = f_m * (2 * f_m1 - f_m)
        d = f_m * f_m + f_m1 * f_m1

        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

    return fib_pair(n)[0]
```

### 3. Check if Number is Power of 2
```python
def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0
```

### 4. Finding All Divisors
```python
def get_divisors(n):
    """Find all divisors in O(√n)"""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)
```

## Competitive Programming Tips

1. **Use built-in pow(base, exp, mod)** - It's optimized for modular exponentiation
2. **Precompute factorials** - For multiple nCr calculations
3. **Sieve once** - Precompute primes if needed multiple times
4. **Watch for overflow** - Use modular arithmetic throughout
5. **Fermat's theorem** - Works only when modulus is prime
6. **GCD optimization** - Use built-in math.gcd in Python

## Practice Problems

- LeetCode 204: Count Primes
- LeetCode 50: Pow(x, n)
- LeetCode 372: Super Pow
- LeetCode 509: Fibonacci Number
- Codeforces: Number Theory problems
- Project Euler: Mathematical challenges

## Resources

- [CP-Algorithms: Algebra](https://cp-algorithms.com/algebra/)
- [OEIS](https://oeis.org/) - Encyclopedia of Integer Sequences
- [Prime Number Theorem](https://en.wikipedia.org/wiki/Prime_number_theorem)
