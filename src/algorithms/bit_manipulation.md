# Bit Manipulation

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Fundamentals](#fundamentals)
  - [Binary Representation](#binary-representation)
  - [Bitwise Operators](#bitwise-operators)
- [Common Bit Tricks](#common-bit-tricks)
  - [Check if Power of 2](#check-if-power-of-2)
  - [Count Set Bits](#count-set-bits)
  - [Get, Set, Clear, Toggle Bits](#get-set-clear-toggle-bits)
  - [XOR Properties](#xor-properties)
  - [Isolate Rightmost Set Bit](#isolate-rightmost-set-bit)
  - [Remove Rightmost Set Bit](#remove-rightmost-set-bit)
- [Advanced Techniques](#advanced-techniques)
  - [Bit Masking](#bit-masking)
  - [Gray Code](#gray-code)
  - [Bit Packing](#bit-packing)
- [Common Patterns](#common-patterns)
- [Interview Problems](#interview-problems)
- [Complexity Analysis](#complexity-analysis)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

Bit manipulation involves working directly with bits (0s and 1s) to perform operations efficiently. It's a fundamental technique used in:
- Low-level programming (device drivers, embedded systems)
- Performance optimization
- Cryptography and hashing
- Compression algorithms
- Competitive programming
- Interview problems

Bit manipulation can often solve problems with **O(1) space** and very fast execution times.

## ELI10 Explanation

Imagine you have a row of light switches, and each switch can be either ON (1) or OFF (0).

```
Position: 7 6 5 4 3 2 1 0
Switch:   0 1 0 1 1 0 1 0  (This is the number 90 in binary!)
```

Bit manipulation is like having special tools to work with these switches:
- **AND** - Both switches must be ON for the result to be ON
- **OR** - At least one switch must be ON
- **XOR** - Exactly one switch must be ON (not both, not neither)
- **NOT** - Flip all switches
- **Shift** - Slide all switches left or right

These simple operations let us do amazing things super fast!

## Fundamentals

### Binary Representation

Every integer can be represented in binary (base 2):

```python
# Decimal to Binary examples
5  = 0b00000101  # 4 + 1
10 = 0b00001010  # 8 + 2
15 = 0b00001111  # 8 + 4 + 2 + 1
16 = 0b00010000  # 16

def decimal_to_binary(n: int) -> str:
    """Convert decimal to binary string."""
    if n == 0:
        return "0"

    binary = ""
    while n > 0:
        binary = str(n & 1) + binary
        n >>= 1
    return binary

# Python built-in
print(bin(10))  # '0b1010'
print(bin(10)[2:])  # '1010' (remove '0b' prefix)
```

### Bitwise Operators

```python
# Basic operators (using 8-bit representation for clarity)
a = 0b00001100  # 12
b = 0b00001010  # 10

# AND (&) - Both bits must be 1
print(bin(a & b))  # 0b00001000 = 8
# 1100
# 1010 &
# ----
# 1000

# OR (|) - At least one bit must be 1
print(bin(a | b))  # 0b00001110 = 14
# 1100
# 1010 |
# ----
# 1110

# XOR (^) - Exactly one bit must be 1
print(bin(a ^ b))  # 0b00000110 = 6
# 1100
# 1010 ^
# ----
# 0110

# NOT (~) - Flip all bits (careful: Python uses signed integers)
print(bin(~a & 0xFF))  # 0b11110011 = 243 (for 8-bit)

# Left Shift (<<) - Multiply by 2^n
print(bin(a << 2))  # 0b00110000 = 48 (12 * 4)

# Right Shift (>>) - Divide by 2^n
print(bin(a >> 2))  # 0b00000011 = 3 (12 // 4)
```

## Common Bit Tricks

### Check if Power of 2

A number is a power of 2 if it has exactly one bit set.

```python
def is_power_of_two(n: int) -> bool:
    """
    Check if n is a power of 2.

    Power of 2: exactly one bit is set
    Examples:
        8  = 1000
        7  = 0111
        8&7 = 0000

    Time: O(1), Space: O(1)
    """
    return n > 0 and (n & (n - 1)) == 0

# Examples
print(is_power_of_two(8))   # True  (2^3)
print(is_power_of_two(16))  # True  (2^4)
print(is_power_of_two(18))  # False
print(is_power_of_two(1))   # True  (2^0)

# Why it works:
# Power of 2:     1000 (8)
# n - 1:          0111 (7)
# n & (n-1):      0000
#
# Not power of 2: 1010 (10)
# n - 1:          1001 (9)
# n & (n-1):      1000 (not zero!)
```

### Count Set Bits

Count the number of 1s in binary representation (Hamming Weight).

```python
def count_set_bits_naive(n: int) -> int:
    """
    Count set bits by checking each bit.
    Time: O(log n) - number of bits
    """
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

def count_set_bits_kernighan(n: int) -> int:
    """
    Brian Kernighan's Algorithm - faster for sparse bits.
    Each operation removes the rightmost set bit.

    Time: O(k) where k = number of set bits
    """
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost set bit
        count += 1
    return count

# Example trace:
# n = 10 = 1010
# Iteration 1: n = 1010 & 1001 = 1000, count = 1
# Iteration 2: n = 1000 & 0111 = 0000, count = 2

def count_set_bits_builtin(n: int) -> int:
    """Use Python's built-in bin() and count()."""
    return bin(n).count('1')

# Lookup table method (fastest for repeated calls)
def count_set_bits_lookup(n: int) -> int:
    """
    Use precomputed table for 8-bit chunks.
    Time: O(1) for 32-bit numbers
    """
    # Precompute for 0-255
    table = [bin(i).count('1') for i in range(256)]

    count = 0
    while n:
        count += table[n & 0xFF]  # Check last 8 bits
        n >>= 8  # Move to next 8 bits
    return count

# Examples
print(count_set_bits_kernighan(15))  # 4 (1111)
print(count_set_bits_kernighan(7))   # 3 (0111)
print(count_set_bits_kernighan(8))   # 1 (1000)
```

### Get, Set, Clear, Toggle Bits

```python
def get_bit(num: int, i: int) -> int:
    """
    Get the bit at position i (0-indexed from right).
    Returns 0 or 1.
    """
    return (num >> i) & 1

def set_bit(num: int, i: int) -> int:
    """Set the bit at position i to 1."""
    return num | (1 << i)

def clear_bit(num: int, i: int) -> int:
    """Set the bit at position i to 0."""
    return num & ~(1 << i)

def toggle_bit(num: int, i: int) -> int:
    """Flip the bit at position i."""
    return num ^ (1 << i)

def update_bit(num: int, i: int, value: int) -> int:
    """
    Update bit at position i to given value (0 or 1).
    """
    # Clear the bit, then OR with shifted value
    mask = ~(1 << i)
    return (num & mask) | (value << i)

# Example usage
n = 0b01010110  # 86

print(bin(get_bit(n, 2)))        # 1
print(bin(set_bit(n, 0)))        # 0b01010111 (87)
print(bin(clear_bit(n, 1)))      # 0b01010100 (84)
print(bin(toggle_bit(n, 0)))     # 0b01010111 (87)
print(bin(update_bit(n, 3, 0)))  # 0b01010110 -> 0b01010110

# Visual example:
# n =           01010110
# Position:     76543210
#
# get_bit(n, 2):    Check bit 2 -> 1
# set_bit(n, 0):    01010111 (set rightmost to 1)
# clear_bit(n, 1):  01010100 (clear bit 1)
# toggle_bit(n, 0): 01010111 (flip bit 0)
```

### XOR Properties

XOR has unique mathematical properties that make it extremely useful.

```python
"""
XOR Properties:
1. x ^ 0 = x           (Identity)
2. x ^ x = 0           (Self-inverse)
3. x ^ y = y ^ x       (Commutative)
4. (x ^ y) ^ z = x ^ (y ^ z)  (Associative)
5. If a ^ b = c, then a ^ c = b and b ^ c = a (Reversible)
"""

def find_unique_number(nums: list[int]) -> int:
    """
    Find the unique number when all others appear twice.
    Uses XOR property: x ^ x = 0

    Example: [4, 1, 2, 1, 2] -> 4
    4 ^ 1 ^ 2 ^ 1 ^ 2 = 4 ^ (1^1) ^ (2^2) = 4 ^ 0 ^ 0 = 4

    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def swap_numbers(a: int, b: int) -> tuple[int, int]:
    """
    Swap two numbers without temporary variable.
    Uses XOR property.
    """
    print(f"Before: a={a}, b={b}")
    a = a ^ b
    b = a ^ b  # b = (a^b)^b = a
    a = a ^ b  # a = (a^b)^a = b
    print(f"After: a={a}, b={b}")
    return a, b

def find_two_unique_numbers(nums: list[int]) -> list[int]:
    """
    Find two unique numbers when all others appear twice.

    Example: [1, 2, 1, 3, 2, 5] -> [3, 5]

    Strategy:
    1. XOR all numbers: result = 3 ^ 5
    2. Find any set bit in result (diff between 3 and 5)
    3. Partition numbers by that bit
    4. XOR each partition separately

    Time: O(n), Space: O(1)
    """
    # Step 1: XOR all numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num
    # xor_all now contains a ^ b (the two unique numbers)

    # Step 2: Find rightmost set bit (where a and b differ)
    rightmost_bit = xor_all & (-xor_all)

    # Step 3: Partition and XOR
    num1, num2 = 0, 0
    for num in nums:
        if num & rightmost_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]

# Examples
print(find_unique_number([4, 1, 2, 1, 2]))  # 4
print(find_two_unique_numbers([1, 2, 1, 3, 2, 5]))  # [3, 5]
swap_numbers(5, 10)  # Swaps to (10, 5)
```

### Isolate Rightmost Set Bit

```python
def isolate_rightmost_set_bit(n: int) -> int:
    """
    Isolate the rightmost (least significant) set bit.

    Uses: n & (-n) or n & (~n + 1)

    Why it works:
    -n is the two's complement (invert bits and add 1)

    Example: n = 12 = 1100
    -n = -(1100) = 0011 + 1 = 0100
    n & -n = 1100 & 0100 = 0100 = 4
    """
    return n & (-n)

# Examples with visualization
def visualize_isolate(n: int):
    result = n & (-n)
    print(f"n     = {n:4d} = {bin(n)}")
    print(f"-n    = {-n:4d} = {bin(-n & 0xFF)}")
    print(f"n&-n  = {result:4d} = {bin(result)}")
    print()

visualize_isolate(12)  # Isolates bit at position 2 (value 4)
visualize_isolate(10)  # Isolates bit at position 1 (value 2)
visualize_isolate(7)   # Isolates bit at position 0 (value 1)

# Application: Find position of rightmost set bit
def rightmost_set_bit_position(n: int) -> int:
    """
    Find position (0-indexed) of rightmost set bit.
    Returns -1 if no bits are set.
    """
    if n == 0:
        return -1

    isolated = n & (-n)
    position = 0
    while isolated > 1:
        isolated >>= 1
        position += 1
    return position

print(rightmost_set_bit_position(12))  # 2
print(rightmost_set_bit_position(10))  # 1
```

### Remove Rightmost Set Bit

```python
def remove_rightmost_set_bit(n: int) -> int:
    """
    Remove the rightmost (least significant) set bit.

    Uses: n & (n - 1)

    This is the key operation in Brian Kernighan's algorithm!
    """
    return n & (n - 1)

# Examples with visualization
def visualize_remove(n: int):
    result = n & (n - 1)
    print(f"n     = {n:4d} = {bin(n)}")
    print(f"n-1   = {n-1:4d} = {bin(n-1)}")
    print(f"n&n-1 = {result:4d} = {bin(result)}")
    print()

visualize_remove(12)  # 1100 -> 1000 (removes rightmost 1)
visualize_remove(10)  # 1010 -> 1000 (removes rightmost 1)
visualize_remove(7)   # 0111 -> 0110 (removes rightmost 1)

# Application: Check if n is power of 4
def is_power_of_four(n: int) -> bool:
    """
    Check if n is a power of 4.

    Strategy:
    1. Must be power of 2: n & (n-1) == 0
    2. The set bit must be at even position (0, 2, 4, 6...)
    3. Use mask 0x55555555 = 0b01010101010101010101010101010101
    """
    if n <= 0:
        return False

    # Check power of 2
    if n & (n - 1) != 0:
        return False

    # Check bit at even position
    return (n & 0x55555555) != 0

print(is_power_of_four(16))  # True (2^4)
print(is_power_of_four(8))   # False (2^3)
print(is_power_of_four(64))  # True (2^6)
```

## Advanced Techniques

### Bit Masking

Using bits to represent sets and perform set operations efficiently.

```python
class BitMask:
    """
    Represent a set using bits. Each bit position represents an element.
    Perfect for small universes (0-63 with int64).

    Example: Set {0, 2, 5} = 0b00100101 = 37
    """

    @staticmethod
    def add_element(mask: int, elem: int) -> int:
        """Add element to set."""
        return mask | (1 << elem)

    @staticmethod
    def remove_element(mask: int, elem: int) -> int:
        """Remove element from set."""
        return mask & ~(1 << elem)

    @staticmethod
    def contains(mask: int, elem: int) -> bool:
        """Check if element is in set."""
        return (mask & (1 << elem)) != 0

    @staticmethod
    def union(mask1: int, mask2: int) -> int:
        """Union of two sets."""
        return mask1 | mask2

    @staticmethod
    def intersection(mask1: int, mask2: int) -> int:
        """Intersection of two sets."""
        return mask1 & mask2

    @staticmethod
    def difference(mask1: int, mask2: int) -> int:
        """Set difference (elements in mask1 but not mask2)."""
        return mask1 & ~mask2

    @staticmethod
    def subset(mask1: int, mask2: int) -> bool:
        """Check if mask1 is subset of mask2."""
        return (mask1 & mask2) == mask1

    @staticmethod
    def size(mask: int) -> int:
        """Count elements in set."""
        count = 0
        while mask:
            mask &= mask - 1
            count += 1
        return count

    @staticmethod
    def iterate_subsets(mask: int) -> list[int]:
        """
        Generate all subsets of the given mask.
        Uses the trick: subset = (subset - 1) & mask
        """
        subsets = []
        subset = mask
        while True:
            subsets.append(subset)
            if subset == 0:
                break
            subset = (subset - 1) & mask
        return subsets

# Example usage
mask = 0  # Empty set
mask = BitMask.add_element(mask, 0)  # {0}
mask = BitMask.add_element(mask, 2)  # {0, 2}
mask = BitMask.add_element(mask, 5)  # {0, 2, 5}

print(f"Set: {bin(mask)}")  # 0b100101
print(f"Contains 2: {BitMask.contains(mask, 2)}")  # True
print(f"Contains 3: {BitMask.contains(mask, 3)}")  # False
print(f"Size: {BitMask.size(mask)}")  # 3

# Subsets of {0, 2}
subsets = BitMask.iterate_subsets(0b101)
print(f"Subsets: {[bin(s) for s in subsets]}")
# [0b101, 0b100, 0b1, 0b0] = [{0,2}, {2}, {0}, {}]
```

### Gray Code

A sequence where consecutive numbers differ by exactly one bit.

```python
def generate_gray_code(n: int) -> list[int]:
    """
    Generate n-bit Gray code sequence.

    Gray code property: consecutive numbers differ by 1 bit.
    Formula: gray(i) = i ^ (i >> 1)

    Time: O(2^n), Space: O(2^n)
    """
    result = []
    for i in range(1 << n):  # 2^n numbers
        gray = i ^ (i >> 1)
        result.append(gray)
    return result

def gray_to_binary(gray: int) -> int:
    """Convert Gray code to binary."""
    binary = gray
    while gray > 0:
        gray >>= 1
        binary ^= gray
    return binary

# Example: 3-bit Gray code
gray_codes = generate_gray_code(3)
print("Gray Code Sequence:")
for i, gray in enumerate(gray_codes):
    print(f"{i}: {bin(gray)[2:].zfill(3)} (binary: {bin(i)[2:].zfill(3)})")

# Output:
# 0: 000 (binary: 000)
# 1: 001 (binary: 001) - differs by 1 bit from previous
# 2: 011 (binary: 010) - differs by 1 bit from previous
# 3: 010 (binary: 011) - differs by 1 bit from previous
# 4: 110 (binary: 100)
# 5: 111 (binary: 101)
# 6: 101 (binary: 110)
# 7: 100 (binary: 111)
```

### Bit Packing

Store multiple small values in a single integer.

```python
class BitPacker:
    """
    Pack multiple values into a single integer.
    Useful for state compression in DP problems.

    Example: Store RGB color (each 0-255) in one int
    """

    @staticmethod
    def pack_rgb(r: int, g: int, b: int) -> int:
        """
        Pack 3 color values (0-255) into one integer.
        Format: 0xRRGGBB
        """
        return (r << 16) | (g << 8) | b

    @staticmethod
    def unpack_rgb(color: int) -> tuple[int, int, int]:
        """Extract RGB values from packed integer."""
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return (r, g, b)

    @staticmethod
    def pack_coordinates(x: int, y: int) -> int:
        """
        Pack 2D coordinates (assuming 16-bit each).
        Useful for memoization in grid problems.
        """
        return (x << 16) | y

    @staticmethod
    def unpack_coordinates(packed: int) -> tuple[int, int]:
        """Extract coordinates from packed integer."""
        x = (packed >> 16) & 0xFFFF
        y = packed & 0xFFFF
        return (x, y)

# Example usage
color = BitPacker.pack_rgb(255, 128, 64)
print(f"Packed color: {hex(color)}")  # 0xff8040
r, g, b = BitPacker.unpack_rgb(color)
print(f"RGB: ({r}, {g}, {b})")  # (255, 128, 64)

# State compression for DP
def unique_paths_bitmask(grid: list[list[int]]) -> int:
    """
    Example: Use bitmask to represent visited cells.
    For a 4x4 grid, we can use 16 bits to track visits.
    """
    m, n = len(grid), len(grid[0])

    # Pack (x, y, visited_mask) into memoization key
    memo = {}

    def dfs(x: int, y: int, visited: int) -> int:
        if x == m - 1 and y == n - 1:
            return 1

        # Pack state for memoization
        state = (x << 20) | (y << 10) | visited
        if state in memo:
            return memo[state]

        paths = 0
        for dx, dy in [(0, 1), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n:
                cell_bit = nx * n + ny
                if not (visited & (1 << cell_bit)):
                    new_visited = visited | (1 << cell_bit)
                    paths += dfs(nx, ny, new_visited)

        memo[state] = paths
        return paths

    return dfs(0, 0, 1)  # Start with (0,0) visited
```

## Common Patterns

### Pattern 1: Toggle/Flip Bits in Range

```python
def flip_bits_in_range(n: int, left: int, right: int) -> int:
    """
    Flip bits from position left to right (0-indexed).

    Example: n = 0b10101 (21), flip bits 1-3
    Result:  0b11011 (27)
    """
    # Create mask: 1s in range [left, right]
    mask = ((1 << (right - left + 1)) - 1) << left
    return n ^ mask

print(bin(flip_bits_in_range(0b10101, 1, 3)))  # 0b11011
```

### Pattern 2: Find Missing Number

```python
def find_missing_number(nums: list[int]) -> int:
    """
    Find missing number in array [0, n].
    Uses XOR: x ^ x = 0

    Example: [3, 0, 1] -> 2 is missing

    Time: O(n), Space: O(1)
    """
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

# Alternative: XOR all numbers 0 to n with array elements
def find_missing_number_v2(nums: list[int]) -> int:
    """Same problem, clearer logic."""
    xor_all = 0
    xor_array = 0

    for i in range(len(nums) + 1):
        xor_all ^= i

    for num in nums:
        xor_array ^= num

    return xor_all ^ xor_array

print(find_missing_number([3, 0, 1]))  # 2
print(find_missing_number([9,6,4,2,3,5,7,0,1]))  # 8
```

### Pattern 3: Reverse Bits

```python
def reverse_bits(n: int) -> int:
    """
    Reverse bits of a 32-bit unsigned integer.

    Example: 00000010100101000001111010011100
          -> 00111001011110000010100101000000

    Time: O(1), Space: O(1)
    """
    result = 0
    for i in range(32):
        # Get bit from right, add to left
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

# Optimized version using divide and conquer
def reverse_bits_optimized(n: int) -> int:
    """
    Reverse bits in O(log(bits)) operations.
    Uses divide and conquer approach.
    """
    # Swap adjacent bits
    n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1)
    # Swap adjacent pairs
    n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2)
    # Swap adjacent nibbles
    n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4)
    # Swap adjacent bytes
    n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8)
    # Swap adjacent 2-bytes
    n = (n >> 16) | (n << 16)

    return n & 0xFFFFFFFF  # Ensure 32-bit

print(bin(reverse_bits(0b00000010100101000001111010011100)))
```

### Pattern 4: Generate All Subsets

```python
def generate_subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets using bit manipulation.
    For n elements, there are 2^n subsets.

    Each subset corresponds to a binary number:
    - If bit i is set, include nums[i]

    Time: O(n * 2^n), Space: O(n * 2^n)
    """
    n = len(nums)
    subsets = []

    # Iterate through all possible bit patterns
    for mask in range(1 << n):  # 2^n patterns
        subset = []
        for i in range(n):
            # Check if i-th bit is set
            if mask & (1 << i):
                subset.append(nums[i])
        subsets.append(subset)

    return subsets

# Example with trace
def generate_subsets_traced(nums: list[int]) -> list[list[int]]:
    """Generate subsets with detailed trace."""
    n = len(nums)
    subsets = []

    print(f"Generating subsets for {nums}")
    print(f"Total subsets: {1 << n}")
    print()

    for mask in range(1 << n):
        subset = []
        print(f"Mask: {mask:03b}", end=" -> ")

        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
                print(f"nums[{i}]={nums[i]}", end=" ")

        subsets.append(subset)
        print(f"= {subset}")

    return subsets

# Example
result = generate_subsets([1, 2, 3])
print(f"\nAll subsets of [1,2,3]: {result}")
# [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

# With trace for [1, 2]
generate_subsets_traced([1, 2])
```

## Interview Problems

### Problem 1: Single Number

```python
def single_number(nums: list[int]) -> int:
    """
    LeetCode 136: Every element appears twice except one.
    Find the single one.

    Solution: XOR all numbers (duplicates cancel out).
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result
```

### Problem 2: Single Number III

```python
def single_number_iii(nums: list[int]) -> list[int]:
    """
    LeetCode 260: Every element appears twice except two.
    Find both single numbers.

    Time: O(n), Space: O(1)
    """
    # Already implemented above in XOR Properties section
    return find_two_unique_numbers(nums)
```

### Problem 3: Maximum XOR of Two Numbers

```python
class TrieNode:
    def __init__(self):
        self.children = {}

def find_maximum_xor(nums: list[int]) -> int:
    """
    LeetCode 421: Find maximum XOR of two numbers in array.

    Strategy: Build Trie of binary representations, then for each
    number, try to find the path that maximizes XOR (choose opposite bits).

    Time: O(n * 32), Space: O(n * 32)
    """
    root = TrieNode()

    # Build Trie (32 bits per number)
    for num in nums:
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]

    max_xor = 0

    # For each number, find maximum XOR
    for num in nums:
        node = root
        current_xor = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for maximum XOR
            toggled_bit = 1 - bit

            if toggled_bit in node.children:
                current_xor |= (1 << i)
                node = node.children[toggled_bit]
            else:
                node = node.children[bit]

        max_xor = max(max_xor, current_xor)

    return max_xor

print(find_maximum_xor([3, 10, 5, 25, 2, 8]))  # 28 (5 ^ 25)
```

### Problem 4: Counting Bits

```python
def counting_bits(n: int) -> list[int]:
    """
    LeetCode 338: Count set bits for all numbers 0 to n.

    DP approach: bits[i] = bits[i >> 1] + (i & 1)
    - i >> 1: remove rightmost bit
    - i & 1: check if rightmost bit is 1

    Time: O(n), Space: O(n)
    """
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits[i] = bits[i >> 1] + (i & 1)
    return bits

# Alternative: bits[i] = bits[i & (i-1)] + 1
def counting_bits_v2(n: int) -> list[int]:
    """Using Brian Kernighan's insight."""
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits[i] = bits[i & (i - 1)] + 1
    return bits

print(counting_bits(5))  # [0, 1, 1, 2, 1, 2]
```

### Problem 5: UTF-8 Validation

```python
def valid_utf8(data: list[int]) -> bool:
    """
    LeetCode 393: Validate UTF-8 encoding.

    UTF-8 rules:
    - 1-byte: 0xxxxxxx
    - 2-byte: 110xxxxx 10xxxxxx
    - 3-byte: 1110xxxx 10xxxxxx 10xxxxxx
    - 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

    Time: O(n), Space: O(1)
    """
    def get_byte_count(byte: int) -> int:
        """Get number of bytes in this UTF-8 character."""
        if (byte >> 7) == 0b0:
            return 1
        elif (byte >> 5) == 0b110:
            return 2
        elif (byte >> 4) == 0b1110:
            return 3
        elif (byte >> 3) == 0b11110:
            return 4
        else:
            return 0  # Invalid

    i = 0
    while i < len(data):
        byte_count = get_byte_count(data[i])

        if byte_count == 0 or i + byte_count > len(data):
            return False

        # Check continuation bytes (must be 10xxxxxx)
        for j in range(1, byte_count):
            if (data[i + j] >> 6) != 0b10:
                return False

        i += byte_count

    return True

print(valid_utf8([197, 130, 1]))  # True (2-byte + 1-byte)
print(valid_utf8([235, 140, 4]))  # False
```

### Problem 6: Bitwise AND of Range

```python
def range_bitwise_and(left: int, right: int) -> int:
    """
    LeetCode 201: Bitwise AND of all numbers in range [left, right].

    Key insight: Result keeps only the common prefix bits.
    All other bits will have both 0 and 1 in the range.

    Strategy: Find common prefix by right-shifting both until equal.

    Time: O(log n), Space: O(1)
    """
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1

    return left << shift

# Example: [5, 7]
# 5 = 101
# 6 = 110
# 7 = 111
# AND = 100 (only first bit is common)

print(range_bitwise_and(5, 7))  # 4
print(range_bitwise_and(1, 2147483647))  # 0
```

## Complexity Analysis

Most bit manipulation operations are **O(1)** or **O(k)** where k is the number of bits:

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| AND, OR, XOR, NOT | O(1) | O(1) |
| Shift left/right | O(1) | O(1) |
| Check/Set/Clear bit | O(1) | O(1) |
| Count set bits (naive) | O(log n) | O(1) |
| Count set bits (Kernighan) | O(k) where k = set bits | O(1) |
| Check power of 2 | O(1) | O(1) |
| Generate subsets | O(n * 2^n) | O(2^n) |
| Find unique number | O(n) | O(1) |
| Maximum XOR (Trie) | O(n * 32) | O(n * 32) |

**Key advantages:**
- Often replace O(n) space with O(1) using bitmasks
- Very fast in practice (hardware-level operations)
- Useful for state compression in DP

## When to Use

**Use bit manipulation when:**

1. **Set operations on small universe** (0-63 elements)
   - Faster than HashSet for small sets
   - O(1) operations vs O(1) amortized

2. **Finding duplicates/unique elements**
   - XOR properties eliminate duplicates
   - O(1) space vs O(n) with hash map

3. **State compression in DP**
   - Represent visited states as bitmask
   - Reduce dimensions in DP table

4. **Performance-critical code**
   - Multiply/divide by powers of 2 with shifts
   - Check even/odd with & 1

5. **Space optimization**
   - Pack multiple boolean flags in one integer
   - Store small values together

6. **Mathematical properties**
   - GCD algorithms
   - Prime checking optimizations
   - Number theory problems

**Don't use when:**
- Readability is more important than performance
- Working with large sets (use proper data structures)
- Debugging is difficult (bit operations are opaque)
- Team unfamiliar with techniques

## Common Pitfalls

### 1. Signed vs Unsigned Integers

```python
# Python has arbitrary precision integers
# Be careful with right shift on negative numbers
n = -8
print(n >> 1)  # -4 (arithmetic shift, preserves sign)

# Use unsigned shift equivalent:
def unsigned_right_shift(n: int, shift: int) -> int:
    """Simulate unsigned right shift."""
    return (n & 0xFFFFFFFF) >> shift if n < 0 else n >> shift
```

### 2. Operator Precedence

```python
# Bitwise operators have lower precedence than comparison
n = 5
# Wrong: if n & 1 == 1:  # Parsed as: n & (1 == 1)
# Correct:
if (n & 1) == 1:  # Check if odd
    print("Odd")

# Always use parentheses for clarity!
```

### 3. Integer Overflow (in other languages)

```python
# Not an issue in Python, but in Java/C++:
# int x = 1 << 31;  // Overflow! Becomes negative
# Use: long x = 1L << 31;

# Python handles this automatically:
print(1 << 100)  # Works fine!
```

### 4. Off-by-One Errors

```python
# Creating mask for n bits
# Wrong: (1 << n)     # This has n+1 bits!
# Correct: (1 << n) - 1  # This has n bits

print(bin(1 << 3))      # 0b1000 (4 bits)
print(bin((1 << 3) - 1))  # 0b111 (3 bits of 1s)
```

### 5. Forgetting Edge Cases

```python
def is_power_of_two(n: int) -> bool:
    # Wrong: return (n & (n - 1)) == 0
    # Correct: must check n > 0 first!
    return n > 0 and (n & (n - 1)) == 0

# Edge cases:
print(is_power_of_two(0))   # False (not power of 2)
print(is_power_of_two(-8))  # False (negative)
print(is_power_of_two(1))   # True (2^0)
```

## Practice Problems

### Easy
1. **Single Number** (LeetCode 136)
2. **Number of 1 Bits** (LeetCode 191)
3. **Reverse Bits** (LeetCode 190)
4. **Power of Two** (LeetCode 231)
5. **Power of Four** (LeetCode 342)
6. **Missing Number** (LeetCode 268)
7. **Hamming Distance** (LeetCode 461)
8. **Binary Number with Alternating Bits** (LeetCode 693)
9. **Prime Number of Set Bits** (LeetCode 762)

### Medium
10. **Single Number II** (LeetCode 137) - element appears 3 times
11. **Single Number III** (LeetCode 260) - two unique elements
12. **Bitwise AND of Numbers Range** (LeetCode 201)
13. **Counting Bits** (LeetCode 338)
14. **Maximum XOR of Two Numbers** (LeetCode 421)
15. **UTF-8 Validation** (LeetCode 393)
16. **Sum of Two Integers** (LeetCode 371) - without +/-
17. **Repeated DNA Sequences** (LeetCode 187)
18. **Gray Code** (LeetCode 89)
19. **Subsets** (LeetCode 78)
20. **Find the Duplicate Number** (LeetCode 287)

### Hard
21. **Maximum XOR with Element from Array** (LeetCode 1707)
22. **Minimum XOR Sum of Two Arrays** (LeetCode 1879)
23. **Find XOR Sum of All Pairs Bitwise AND** (LeetCode 1835)
24. **Maximize XOR for Each Query** (LeetCode 1829)

### Bonus Challenges
25. Implement bitset operations (union, intersection, etc.)
26. Solve Sudoku using bitmasks
27. Traveling Salesman with bitmask DP
28. Count valid permutations using bitmasks

## Additional Resources

### Online Tools
- **Bit Visualizer**: https://visualgo.net/en/bitmask
- **Binary Calculator**: https://www.rapidtables.com/convert/number/binary-calculator.html

### Tutorials
- **HackerEarth**: Bit Manipulation Tutorial
- **TopCoder**: A Bit of Fun: Fun with Bits
- **GeeksforGeeks**: Bit Manipulation

### Books
- **Hacker's Delight** by Henry S. Warren Jr. (the bible of bit manipulation)
- **Programming Pearls** by Jon Bentley
- **The Art of Computer Programming, Vol 4A** by Donald Knuth

### Practice Platforms
- LeetCode Tag: Bit Manipulation
- Codeforces: Bitmask DP problems
- HackerRank: Bit Manipulation track

---

**Key Takeaways:**
1. Bit manipulation provides O(1) time and space for many operations
2. XOR has unique properties (self-inverse, associative) useful for finding unique elements
3. Common patterns: check/set/clear bits, counting bits, generating subsets
4. Great for interviews: shows strong CS fundamentals
5. Trade-off: code clarity vs performance

Master these techniques and you'll have a powerful tool for optimization and problem-solving!
