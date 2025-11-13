# Big O Notation

Big O notation is a mathematical concept used to describe the performance or complexity of an algorithm. Specifically, it characterizes algorithms in terms of their time or space requirements in relation to the size of the input data. Understanding Big O notation is crucial for evaluating the efficiency of algorithms and making informed decisions about which algorithm to use in a given situation.

## Key Concepts

- **Time Complexity**: This refers to the amount of time an algorithm takes to complete as a function of the length of the input. It helps in understanding how the execution time increases with the size of the input.

- **Space Complexity**: This refers to the amount of memory an algorithm uses in relation to the input size. It is important to consider both time and space complexity when analyzing an algorithm.

## Common Big O Notations

### $O(1)$ - Constant Time

The execution time does not change regardless of the input size.

```python
def get_first_element(arr):
    return arr[0]  # Always one operation

def hash_lookup(dictionary, key):
    return dictionary[key]  # Constant time hash table lookup

# Example
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(get_first_element(arr))  # $O(1)$
```

**Examples**: Array access, hash table operations, simple arithmetic

### $O(\log n)$ - Logarithmic Time

The execution time grows logarithmically as the input size increases.

```python
def binary_search(arr, target):
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

# Example: With 1000 elements, only ~10 comparisons needed
arr = list(range(1000))
print(binary_search(arr, 742))  # $O(\log n)$
```

**Examples**: Binary search, balanced binary tree operations

### $O(n)$ - Linear Time

The execution time grows linearly with the input size.

```python
def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1

def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

# Example
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(linear_search(arr, 9))  # $O(n)$
print(find_max(arr))          # $O(n)$
```

**Examples**: Linear search, array traversal, finding min/max

### $O(n \log n)$ - Linearithmic Time

Common in efficient sorting algorithms.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
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

# Example
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # $O(n \log n)$
```

**Examples**: Merge sort, heap sort, quick sort (average case)

### $O(n^2)$ - Quadratic Time

The execution time grows quadratically with the input size.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def find_duplicates_naive(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates

# Example
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr.copy()))  # $O(n^2)$
```

**Examples**: Bubble sort, selection sort, insertion sort, nested loops

### $O(2^n)$ - Exponential Time

The execution time doubles with each additional element.

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def power_set(s):
    if not s:
        return [[]]
    
    subsets = power_set(s[1:])
    return subsets + [[s[0]] + subset for subset in subsets]

# Example (slow for large n!)
print(fibonacci_recursive(10))  # $O(2^n)$
print(power_set([1, 2, 3]))     # $O(2^n)$
```

**Examples**: Recursive Fibonacci, generating all subsets

### $O(n!)$ - Factorial Time

The execution time grows factorially with the input size.

```python
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for p in permutations(rest):
            result.append([arr[i]] + p)
    return result

# Example (very slow!)
print(permutations([1, 2, 3]))  # $O(n!)$
# For n=10, this would generate 3,628,800 permutations!
```

**Examples**: Generating all permutations, traveling salesman (brute force)

## Complexity Comparison

```python
import time
import random

def compare_complexities(n):
    # $O(1)$
    start = time.time()
    _ = n
    o1_time = time.time() - start

    # $O(\log n)$
    start = time.time()
    _ = n.bit_length()
    olog_time = time.time() - start

    # $O(n)$
    start = time.time()
    _ = sum(range(n))
    on_time = time.time() - start

    # $O(n \log n)$
    start = time.time()
    arr = list(range(n))
    random.shuffle(arr)
    _ = sorted(arr)
    onlogn_time = time.time() - start

    # $O(n^2)$
    start = time.time()
    for i in range(min(n, 1000)):  # Limited to avoid long wait
        for j in range(min(n, 1000)):
            pass
    on2_time = time.time() - start
    
    print(f"n = {n}:")
    print(f"  $O(1)$:      {o1_time:.6f}s")
    print(f"  $O(\log n)$:  {olog_time:.6f}s")
    print(f"  $O(n)$:      {on_time:.6f}s")
    print(f"  $O(n \log n)$:{onlogn_time:.6f}s")
    print(f"  $O(n^2)$:     {on2_time:.6f}s (limited)")

# Example
compare_complexities(10000)
```

## Space Complexity Examples

```python
# $O(1)$ space - In-place
def reverse_array_inplace(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# $O(n)$ space - Additional array
def reverse_array_new(arr):
    return arr[::-1]

# $O(n)$ space - Recursion stack
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# $O(n^2)$ space - 2D array
def create_matrix(n):
    return [[0 for _ in range(n)] for _ in range(n)]
```

## Best, Average, and Worst Case

Different scenarios can have different complexities:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Quick Sort:
# Best case:    $O(n \log n)$ - balanced partitions
# Average case: $O(n \log n)$
# Worst case:   $O(n^2)$ - already sorted array
```

## Analyzing Algorithm Complexity

```python
def example_algorithm(arr):
    n = len(arr)

    # $O(1)$ - constant operations
    first = arr[0]
    last = arr[-1]

    # $O(n)$ - single loop
    total = sum(arr)

    # $O(n^2)$ - nested loops
    for i in range(n):
        for j in range(n):
            pass

    # $O(n \log n)$ - sorting
    sorted_arr = sorted(arr)

    # Overall: $O(1) + O(n) + O(n^2) + O(n \log n) = O(n^2)$
    # (Dominant term is $n^2$)
```

## Big O Rules

1. **Drop constants**: $O(2n) \to O(n)$
2. **Drop non-dominant terms**: $O(n^2 + n) \to O(n^2)$
3. **Different inputs use different variables**: $O(a + b)$ for two arrays
4. **Multiplication for nested**: $O(a \times b)$ for nested loops over different arrays

```python
# Rule 1: Drop constants
def example1(arr):
    for item in arr:        # $O(n)$
        print(item)
    for item in arr:        # $O(n)$
        print(item)
    # Total: $O(2n) = O(n)$

# Rule 2: Drop non-dominant terms
def example2(arr):
    for i in range(len(arr)):           # $O(n)$
        for j in range(len(arr)):       # $O(n^2)$
            print(i, j)
    for item in arr:                    # $O(n)$
        print(item)
    # Total: $O(n^2 + n) = O(n^2)$

# Rule 3: Different inputs
def example3(arr1, arr2):
    for item in arr1:       # $O(a)$
        print(item)
    for item in arr2:       # $O(b)$
        print(item)
    # Total: $O(a + b)$

# Rule 4: Multiplication for nested
def example4(arr1, arr2):
    for item1 in arr1:              # $O(a)$
        for item2 in arr2:          # $O(b)$
            print(item1, item2)
    # Total: $O(a \times b)$
```

## Complexity Cheat Sheet

| Complexity | Name | Example Operations |
|------------|------|-------------------|
| $O(1)$ | Constant | Array access, hash lookup |
| $O(\log n)$ | Logarithmic | Binary search |
| $O(n)$ | Linear | Loop through array |
| $O(n \log n)$ | Linearithmic | Efficient sorting |
| $O(n^2)$ | Quadratic | Nested loops |
| $O(n^3)$ | Cubic | Triple nested loops |
| $O(2^n)$ | Exponential | Recursive Fibonacci |
| $O(n!)$ | Factorial | All permutations |

## Growth Rates Visualization

```
For n = 100:
$O(1)$:      1 operation
$O(\log n)$:  7 operations
$O(n)$:      100 operations
$O(n \log n)$:700 operations
$O(n^2)$:     10,000 operations
$O(n^3)$:     1,000,000 operations
$O(2^n)$:    $1.27 \times 10^{30}$ operations (intractable!)
$O(n!)$:     $9.33 \times 10^{157}$ operations (impossible!)
```

## Practical Tips

1. **Optimize bottlenecks**: Focus on the most time-consuming parts
2. **Trade-offs**: Sometimes $O(n)$ space can give $O(1)$ time (caching)
3. **Real-world considerations**: Constants matter for small n
4. **Amortized analysis**: Some operations are cheaper on average
5. **Choose appropriately**: Don't over-optimize; $O(n^2)$ is fine for small n

## Conclusion

Big O notation provides a high-level understanding of the efficiency of algorithms, allowing developers to compare and choose the most suitable algorithm for their needs. By analyzing both time and space complexity, one can make informed decisions that lead to better performance in software applications.
