# Divide and Conquer

Divide and conquer is a fundamental algorithmic technique that involves breaking a problem down into smaller subproblems, solving each subproblem independently, and then combining their solutions to solve the original problem. This approach is particularly effective for problems that can be recursively divided into similar subproblems.

## Key Concepts

- **Divide**: The problem is divided into smaller subproblems that are similar to the original problem but smaller in size. This step often involves identifying a base case for the recursion.

- **Conquer**: Each subproblem is solved independently, often using the same divide and conquer strategy recursively. If the subproblems are small enough, they may be solved directly.

- **Combine**: The solutions to the subproblems are combined to form a solution to the original problem. This step is crucial as it integrates the results of the smaller problems into a coherent solution.

## Merge Sort

Efficient sorting algorithm using divide and conquer.

```python
def merge_sort(arr):
    # Base case: array with 0 or 1 element
    if len(arr) <= 1:
        return arr
    
    # Divide: split array in half
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer and Combine: merge sorted halves
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # Output: [3, 9, 10, 27, 38, 43, 82]
```

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

## Quick Sort

Efficient in-place sorting algorithm.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Choose pivot (middle element)
    pivot = arr[len(arr) // 2]
    
    # Divide: partition around pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Conquer and Combine
    return quick_sort(left) + middle + quick_sort(right)

# In-place version
def quick_sort_inplace(arr, low, high):
    if low < high:
        # Partition and get pivot index
        pi = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Example usage
arr = [10, 7, 8, 9, 1, 5]
quick_sort_inplace(arr, 0, len(arr) - 1)
print(arr)  # Output: [1, 5, 7, 8, 9, 10]
```

**Time Complexity**: O(n log n) average, O(n²) worst
**Space Complexity**: O(log n) for recursion stack

## Binary Search

Classic divide and conquer search algorithm.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half
    
    return -1  # Not found

# Recursive version
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17]
print(binary_search(arr, 7))  # Output: 3
print(binary_search_recursive(arr, 13, 0, len(arr) - 1))  # Output: 6
```

**Time Complexity**: O(log n)
**Space Complexity**: O(1) iterative, O(log n) recursive

## Maximum Subarray (Kadane's Algorithm)

Find the contiguous subarray with the largest sum.

```python
def max_subarray_divide_conquer(arr, left, right):
    # Base case: single element
    if left == right:
        return arr[left]
    
    # Divide: find middle
    mid = (left + right) // 2
    
    # Conquer: recursively find max in left and right halves
    left_max = max_subarray_divide_conquer(arr, left, mid)
    right_max = max_subarray_divide_conquer(arr, mid + 1, right)
    
    # Combine: find max crossing the middle
    cross_max = max_crossing_sum(arr, left, mid, right)
    
    return max(left_max, right_max, cross_max)

def max_crossing_sum(arr, left, mid, right):
    # Sum from mid to left
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, left - 1, -1):
        current_sum += arr[i]
        left_sum = max(left_sum, current_sum)
    
    # Sum from mid+1 to right
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, right + 1):
        current_sum += arr[i]
        right_sum = max(right_sum, current_sum)
    
    return left_sum + right_sum

# Example usage
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = max_subarray_divide_conquer(arr, 0, len(arr) - 1)
print(f"Maximum subarray sum: {max_sum}")  # Output: 6 ([4,-1,2,1])
```

**Time Complexity**: O(n log n)

## Count Inversions

Count how many pairs are out of order in an array.

```python
def merge_count_inversions(arr):
    if len(arr) <= 1:
        return arr, 0
    
    mid = len(arr) // 2
    left, left_inv = merge_count_inversions(arr[:mid])
    right, right_inv = merge_count_inversions(arr[mid:])
    
    merged, split_inv = merge_and_count(left, right)
    
    return merged, left_inv + right_inv + split_inv

def merge_and_count(left, right):
    result = []
    inversions = 0
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            inversions += len(left) - i  # All remaining in left are inversions
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inversions

# Example usage
arr = [2, 4, 1, 3, 5]
sorted_arr, inversions = merge_count_inversions(arr)
print(f"Inversions: {inversions}")  # Output: 3
```

## Closest Pair of Points

Find the two closest points in a 2D plane.

```python
import math

def closest_pair(points):
    # Sort points by x-coordinate
    px = sorted(points, key=lambda p: p[0])
    # Sort points by y-coordinate
    py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_recursive(px, py)

def closest_pair_recursive(px, py):
    n = len(px)
    
    # Base case: few points, use brute force
    if n <= 3:
        return brute_force_closest(px)
    
    # Divide: split by vertical line
    mid = n // 2
    midpoint = px[mid]
    
    pyl = [p for p in py if p[0] <= midpoint[0]]
    pyr = [p for p in py if p[0] > midpoint[0]]
    
    # Conquer: find closest in each half
    dl = closest_pair_recursive(px[:mid], pyl)
    dr = closest_pair_recursive(px[mid:], pyr)
    
    # Find minimum
    d = min(dl, dr)
    
    # Combine: check points near dividing line
    strip = [p for p in py if abs(p[0] - midpoint[0]) < d]
    strip_min = strip_closest(strip, d)
    
    return min(d, strip_min)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def brute_force_closest(points):
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            min_dist = min(min_dist, distance(points[i], points[j]))
    return min_dist

def strip_closest(strip, d):
    min_dist = d
    for i in range(len(strip)):
        for j in range(i + 1, min(i + 7, len(strip))):
            min_dist = min(min_dist, distance(strip[i], strip[j]))
    return min_dist

# Example usage
points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
min_distance = closest_pair(points)
print(f"Smallest distance: {min_distance:.2f}")
```

## Matrix Multiplication (Strassen's Algorithm)

Faster matrix multiplication algorithm.

```python
import numpy as np

def strassen_matrix_multiply(A, B):
    n = len(A)
    
    # Base case: 1x1 matrix
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Divide matrices into quadrants
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]
    
    # Compute 7 products (Strassen's method)
    M1 = strassen_matrix_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    M2 = strassen_matrix_multiply(matrix_add(A21, A22), B11)
    M3 = strassen_matrix_multiply(A11, matrix_sub(B12, B22))
    M4 = strassen_matrix_multiply(A22, matrix_sub(B21, B11))
    M5 = strassen_matrix_multiply(matrix_add(A11, A12), B22)
    M6 = strassen_matrix_multiply(matrix_sub(A21, A11), matrix_add(B11, B12))
    M7 = strassen_matrix_multiply(matrix_sub(A12, A22), matrix_add(B21, B22))
    
    # Combine
    C11 = matrix_add(matrix_sub(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_sub(matrix_add(M1, M3), M2), M6)
    
    # Construct result
    result = []
    for i in range(mid):
        result.append(C11[i] + C12[i])
    for i in range(mid):
        result.append(C21[i] + C22[i])
    
    return result

def matrix_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
```

**Time Complexity**: O(n^2.807) vs O(n³) for standard multiplication

## Divide and Conquer Template

```python
def divide_and_conquer(problem):
    # Base case
    if is_simple(problem):
        return solve_directly(problem)
    
    # Divide
    subproblems = divide(problem)
    
    # Conquer
    subsolutions = [divide_and_conquer(subproblem) for subproblem in subproblems]
    
    # Combine
    solution = combine(subsolutions)
    
    return solution
```

## Applications

Divide and conquer is widely used in various algorithms and applications, including:

- **Sorting Algorithms**: Algorithms like Merge Sort and Quick Sort utilize the divide and conquer approach to sort elements efficiently.

- **Searching Algorithms**: Binary Search is a classic example of a divide and conquer algorithm that efficiently finds an element in a sorted array.

- **Matrix Multiplication**: Strassen's algorithm for matrix multiplication is another example where the divide and conquer technique is applied to reduce the complexity of the operation.

- **Computational Geometry**: Problems like finding the closest pair of points or convex hull.

- **Fast Fourier Transform**: FFT uses divide and conquer for efficient signal processing.

## Advantages

1. **Efficiency**: Often achieves better time complexity than brute force
2. **Parallelization**: Subproblems can be solved independently
3. **Cache-friendly**: Works well with memory hierarchy
4. **Elegant solutions**: Natural recursive structure

## Disadvantages

1. **Overhead**: Recursive calls add overhead
2. **Space complexity**: Requires stack space for recursion
3. **Not always optimal**: Some problems have better iterative solutions

## Conclusion

The divide and conquer strategy is a powerful tool in algorithm design, enabling efficient solutions to complex problems by breaking them down into manageable parts. Understanding this technique is essential for developing efficient algorithms in computer science and software engineering.
