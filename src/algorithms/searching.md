# Searching Algorithms

## Overview

Searching algorithms help find elements in data structures. The choice depends on whether data is sorted and the size of the data.

## Linear Search

**Time**: $O(n)$ | **Space**: $O(1)$ | **Works on**: Unsorted arrays

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**When to use**: Small arrays, unsorted data, or when you need to find all occurrences

**Pros**: Simple, works on unsorted data, stable
**Cons**: Slow for large datasets

## Binary Search

**Time**: $O(\log n)$ | **Space**: $O(1)$ iterative, $O(\log n)$ recursive | **Requires**: Sorted array

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found
```

### Recursive Implementation

```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### Variations

```python
# Find first occurrence
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Keep searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# Find last occurrence
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Keep searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# Find insertion position (lower bound)
def lower_bound(arr, target):
    """Find first position where target can be inserted"""
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left

# Find upper bound
def upper_bound(arr, target):
    """Find last position where target can be inserted"""
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left
```

**When to use**: Large sorted datasets, need $O(\log n)$ performance

## Two Pointer Technique

**Time**: $O(n)$ | **Space**: $O(1)$

```python
def two_sum(arr, target):
    """Find two numbers that sum to target in sorted array"""
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# Three sum
def three_sum_closest(arr, target):
    """Find three numbers whose sum is closest to target"""
    arr.sort()
    closest = float('inf')
    result = 0

    for i in range(len(arr) - 2):
        left, right = i + 1, len(arr) - 1

        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]

            if abs(target - current_sum) < closest:
                closest = abs(target - current_sum)
                result = current_sum

            if current_sum < target:
                left += 1
            elif current_sum > target:
                right -= 1
            else:
                return current_sum

    return result
```

**When to use**: Sorted arrays, finding pairs/triplets with specific properties

## Jump Search

**Time**: $O(\sqrt{n})$ | **Space**: $O(1)$ | **Requires**: Sorted array

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0

    # Find block where target is present
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1

    # Linear search in block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1

    # Check if target found
    if arr[prev] == target:
        return prev

    return -1
```

**When to use**: Large sorted arrays where binary search overhead is a concern, systems with costly comparisons

**Pros**: Better than linear, simpler than binary, good for jumping through data
**Cons**: Slower than binary search

## Interpolation Search

**Time**: $O(\log \log n)$ average, $O(n)$ worst | **Requires**: Sorted uniformly distributed data

```python
def interpolation_search(arr, target):
    left, right = 0, len(arr) - 1

    while (left <= right and
           target >= arr[left] and
           target <= arr[right]):

        # Avoid division by zero
        if arr[left] == arr[right]:
            if arr[left] == target:
                return left
            return -1

        # Estimate position using interpolation formula
        pos = left + int((right - left) / (arr[right] - arr[left]) *
                        (target - arr[left]))

        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1

    return -1
```

**When to use**: Uniformly distributed sorted data (e.g., dictionary, phone book)

**Pros**: Faster than binary for uniform data
**Cons**: Worst case $O(n)$, doesn't work well with non-uniform distribution

## Exponential Search

**Time**: $O(\log n)$ | **Space**: $O(1)$ | **Requires**: Sorted array

```python
def exponential_search(arr, target):
    n = len(arr)

    # If target at first position
    if arr[0] == target:
        return 0

    # Find range for binary search
    i = 1
    while i < n and arr[i] <= target:
        i *= 2

    # Binary search in range [i//2, min(i, n-1)]
    left = i // 2
    right = min(i, n - 1)

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**When to use**: Unbounded/infinite arrays, when target is likely near the beginning

**Pros**: Better than binary for unbounded searches
**Cons**: Slightly more complex than binary search

## Fibonacci Search

**Time**: $O(\log n)$ | **Space**: $O(1)$ | **Requires**: Sorted array

```python
def fibonacci_search(arr, target):
    n = len(arr)

    # Generate Fibonacci numbers
    fib2 = 0  # (m-2)'th Fibonacci
    fib1 = 1  # (m-1)'th Fibonacci
    fib = fib2 + fib1  # m'th Fibonacci

    # Find smallest Fibonacci >= n
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib2 + fib1

    offset = -1

    while fib > 1:
        # Check if fib2 is valid location
        i = min(offset + fib2, n - 1)

        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return i

    # Check last element
    if fib1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1

    return -1
```

**When to use**: Memory-constrained systems, data on sequential access devices

**Pros**: Divides using addition instead of division, good for tape/disk
**Cons**: More complex to implement

## Ternary Search

**Time**: $O(\log_3 n)$ | **Space**: $O(1)$ | **Requires**: Sorted array or unimodal function

```python
def ternary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        # Divide into three parts
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2

        if target < arr[mid1]:
            right = mid1 - 1
        elif target > arr[mid2]:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1

    return -1

# Finding maximum in unimodal function
def ternary_search_max(f, left, right, epsilon=1e-9):
    """Find maximum of unimodal function f"""
    while right - left > epsilon:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3

        if f(mid1) < f(mid2):
            left = mid1
        else:
            right = mid2

    return (left + right) / 2
```

**When to use**: Finding extrema in unimodal functions, optimization problems

**Pros**: Can find max/min of functions
**Cons**: More comparisons than binary search for standard searching

## Hash Table Search

**Time**: $O(1)$ average, $O(n)$ worst | **Space**: $O(n)$

```python
def hash_table_search():
    """Using Python's built-in dictionary"""
    # Build hash table
    hash_table = {}
    arr = [4, 2, 7, 1, 9, 3]

    for i, val in enumerate(arr):
        hash_table[val] = i

    # Search in O(1)
    target = 7
    if target in hash_table:
        return hash_table[target]
    return -1

# Custom hash table with collision handling
class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_index = self.hash_function(key)
        # Linear probing for collisions
        for item in self.table[hash_index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[hash_index].append([key, value])

    def search(self, key):
        hash_index = self.hash_function(key)
        for item in self.table[hash_index]:
            if item[0] == key:
                return item[1]
        return None
```

**When to use**: Need $O(1)$ lookups, have extra memory, data doesn't need to be sorted

**Pros**: Fastest average case
**Cons**: Extra space, no ordering, worst case $O(n)$

## Sentinel Search

**Time**: $O(n)$ | **Space**: $O(1)$ | **Works on**: Unsorted arrays

Optimize linear search by eliminating boundary check:

```python
def sentinel_search(arr, target):
    n = len(arr)
    last = arr[n - 1]
    arr[n - 1] = target

    i = 0
    while arr[i] != target:
        i += 1

    arr[n - 1] = last  # Restore

    if i < n - 1 or last == target:
        return i
    return -1
```

**When to use**: Optimization of linear search when boundary checks are expensive

## Comparison

| Algorithm | Time (Avg) | Time (Worst) | Space | Requires Sorted | Best For |
|-----------|-----------|-------------|-------|-----------------|----------|
| Linear | $O(n)$ | $O(n)$ | $O(1)$ | No | Small/unsorted data |
| Binary | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Yes | Large sorted data |
| Jump | $O(\sqrt{n})$ | $O(\sqrt{n})$ | $O(1)$ | Yes | Large arrays, costly comparisons |
| Interpolation | $O(\log \log n)$ | $O(n)$ | $O(1)$ | Yes (uniform) | Uniform distribution |
| Exponential | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Yes | Unbounded arrays |
| Fibonacci | $O(\log n)$ | $O(\log n)$ | $O(1)$ | Yes | Sequential access media |
| Ternary | $O(\log_3 n)$ | $O(\log_3 n)$ | $O(1)$ | Yes | Finding extrema |
| Hash Table | $O(1)$ | $O(n)$ | $O(n)$ | No | Fast lookups with space |

## Decision Tree

```
Do you need O(1) lookup and have extra space?
├─ Yes → Hash Table
└─ No → Is data sorted?
    ├─ No → Linear Search (or sort first)
    └─ Yes → What's the data distribution?
        ├─ Uniform → Interpolation Search
        ├─ Unknown → Binary Search
        ├─ Near beginning → Exponential Search
        └─ Sequential access → Fibonacci Search
```

## Common Patterns and Edge Cases

### Pattern: Search Space Reduction

```python
# Find peak element
def find_peak(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid

    return left

# Search in rotated sorted array
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

### Edge Cases to Handle

```python
def robust_binary_search(arr, target):
    # Handle empty array
    if not arr:
        return -1

    # Handle single element
    if len(arr) == 1:
        return 0 if arr[0] == target else -1

    # Handle target out of bounds
    if target < arr[0] or target > arr[-1]:
        return -1

    # Standard binary search
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

## Key Takeaways

1. **Unsorted data?** Use Linear Search ($O(n)$) or Hash Table ($O(1)$ average)
2. **Sorted data?** Use Binary Search for $O(\log n)$ - it's the default choice
3. **Uniformly distributed?** Try Interpolation Search for $O(\log \log n)$
4. **Need fastest lookup?** Build a Hash Table for $O(1)$ average lookup
5. **Unbounded data?** Use Exponential Search
6. **Finding extrema?** Use Ternary Search on unimodal functions
7. **Space constrained?** Stick with in-place algorithms like Binary or Jump Search

## Performance Tips

1. **Use `mid = left + (right - left) // 2`** to avoid integer overflow
2. **Check bounds** before accessing array elements
3. **For multiple queries**, preprocess data (sort or build hash table)
4. **Binary search variants** are powerful - master lower_bound and upper_bound
5. **Consider data characteristics** - uniform distribution favors interpolation search

## ELI10

Imagine finding a word in a dictionary:

- **Linear Search**: Check every word from start (slow!)
- **Binary Search**: Open middle, go left or right, repeat (fast!)
- **Interpolation Search**: Estimate where word should be based on first letter (smarter!)
- **Hash Table**: Like an index that tells you exactly which page (fastest!)

Binary search is like playing "guess the number" - each guess eliminates half the possibilities!

## LeetCode Problems

- [704. Binary Search](https://leetcode.com/problems/binary-search/) - Basic
- [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/) - Lower bound
- [34. Find First and Last Position](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) - Binary search variations
- [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) - Modified binary search
- [162. Find Peak Element](https://leetcode.com/problems/find-peak-element/) - Search space reduction
- [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) - 2D binary search

## Further Resources

- [LeetCode Binary Search](https://leetcode.com/tag/binary-search/)
- [Searching Algorithms Visualization](https://visualgo.net/en/search)
- [Binary Search Tutorial](https://leetcode.com/explore/learn/card/binary-search/)
