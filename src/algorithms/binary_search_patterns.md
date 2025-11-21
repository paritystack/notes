# Binary Search Patterns

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use Binary Search](#when-to-use-binary-search)
  - [Binary Search Properties](#binary-search-properties)
- [Classic Binary Search](#classic-binary-search)
- [Pattern 1: Find Exact Match](#pattern-1-find-exact-match)
- [Pattern 2: Find First/Last Occurrence](#pattern-2-find-firstlast-occurrence)
- [Pattern 3: Find Insert Position](#pattern-3-find-insert-position)
- [Pattern 4: Search in Rotated Array](#pattern-4-search-in-rotated-array)
- [Pattern 5: Binary Search on Answer](#pattern-5-binary-search-on-answer)
- [Pattern 6: Peak Finding](#pattern-6-peak-finding)
- [Pattern 7: Matrix Binary Search](#pattern-7-matrix-binary-search)
- [Advanced Patterns](#advanced-patterns)
- [Common Templates](#common-templates)
- [Interview Problems](#interview-problems)
- [Complexity Analysis](#complexity-analysis)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Binary Search** is a divide-and-conquer algorithm that efficiently searches for a target in a sorted sequence. Beyond the classic "find target in sorted array," binary search is a powerful technique applicable to many optimization and search problems.

**Key characteristics:**
- Works on **sorted** (or partially sorted) data
- Divides search space in half each iteration
- Time complexity: **O(log n)**
- Can be applied to abstract search spaces (not just arrays)
- Useful for finding "minimum X such that condition holds"

## ELI10 Explanation

Imagine you have 100 numbered cards arranged in order (1, 2, 3, ..., 100) and need to find card number 67.

**Bad way (Linear Search):**
Check card 1, then 2, then 3... until you find 67. Could take 67 tries!

**Smart way (Binary Search):**
1. Look at the MIDDLE card (card 50)
2. 67 > 50, so you know 67 must be in the RIGHT half
3. Throw away the left half! Now search cards 51-100
4. Check middle of this range (card 75)
5. 67 < 75, so search cards 51-74
6. Keep cutting in half...
7. Found it in just 7 tries instead of 67!

This is like the "guess my number" game where someone says "higher" or "lower"!

## Core Concepts

### When to Use Binary Search

Binary search works when you can:

1. **Sorted array** - obvious case
2. **Rotated sorted array** - partially sorted
3. **Search space has ordering** - can eliminate half each time
4. **Answer space is bounded** - search for optimal value
5. **Monotonic function** - if `f(x)` is true, then `f(x+1)` might be true/false but doesn't flip back

**Key question:** Can you determine which half contains the answer?

### Binary Search Properties

```python
"""
SEARCH SPACE REDUCTION:
Iteration 1: Search space = n
Iteration 2: Search space = n/2
Iteration 3: Search space = n/4
...
Iteration k: Search space = n/(2^k)

When n/(2^k) = 1, we're done.
So k = log₂(n) iterations!

Example: n = 1,000,000
log₂(1,000,000) ≈ 20 iterations
Linear search: up to 1,000,000 iterations!
"""

# INVARIANT (most important concept!):
# After each iteration, the answer is ALWAYS in [left, right]
```

## Classic Binary Search

### Basic Template

```python
def binary_search(nums: list[int], target: int) -> int:
    """
    Classic binary search: find index of target.
    Returns -1 if not found.

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        # Avoid overflow: mid = (left + right) // 2
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1  # Search right half
        else:
            right = mid - 1  # Search left half

    return -1  # Not found

# Example with trace
def binary_search_traced(nums: list[int], target: int) -> int:
    """Binary search with detailed trace."""
    print(f"Array: {nums}")
    print(f"Target: {target}\n")

    left, right = 0, len(nums) - 1
    iteration = 1

    while left <= right:
        mid = left + (right - left) // 2
        print(f"Iteration {iteration}:")
        print(f"  left={left}, mid={mid}, right={right}")
        print(f"  nums[mid]={nums[mid]}")

        if nums[mid] == target:
            print(f"  Found at index {mid}!")
            return mid
        elif nums[mid] < target:
            print(f"  {nums[mid]} < {target}, search right half")
            left = mid + 1
        else:
            print(f"  {nums[mid]} > {target}, search left half")
            right = mid - 1

        iteration += 1
        print()

    print("  Not found!")
    return -1

# Test
print(binary_search([1, 3, 5, 7, 9, 11, 13, 15], 7))  # 3
print()
binary_search_traced([1, 3, 5, 7, 9, 11, 13, 15], 7)
```

## Pattern 1: Find Exact Match

Already shown above. Additional variations:

### Search in Unknown Size Array

```python
class ArrayReader:
    """Interface for array with unknown size."""
    def get(self, index: int) -> int:
        """Returns value at index or 2^31 - 1 if out of bounds."""
        pass

def search(reader: ArrayReader, target: int) -> int:
    """
    LeetCode 702: Search in sorted array of unknown size.

    Strategy:
    1. Find upper bound using exponential search
    2. Binary search in [0, bound]

    Time: O(log n), Space: O(1)
    """
    # Find upper bound
    left, right = 0, 1
    while reader.get(right) < target:
        left = right
        right *= 2  # Exponential expansion

    # Binary search
    while left <= right:
        mid = left + (right - left) // 2
        val = reader.get(mid)

        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

## Pattern 2: Find First/Last Occurrence

Find leftmost or rightmost position of target.

### Find First Occurrence

```python
def find_first(nums: list[int], target: int) -> int:
    """
    Find leftmost (first) occurrence of target.
    Returns -1 if not found.

    Key: Don't return immediately when found!
    Continue searching left half.

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left!
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# Example
print(find_first([1, 2, 2, 2, 3, 4, 5], 2))  # 1 (first occurrence)
```

### Find Last Occurrence

```python
def find_last(nums: list[int], target: int) -> int:
    """
    Find rightmost (last) occurrence of target.
    Returns -1 if not found.

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right!
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# Example
print(find_last([1, 2, 2, 2, 3, 4, 5], 2))  # 3 (last occurrence)
```

### Find Range

```python
def search_range(nums: list[int], target: int) -> list[int]:
    """
    LeetCode 34: Find first and last position of target.

    Time: O(log n), Space: O(1)
    """
    return [find_first(nums, target), find_last(nums, target)]

print(search_range([5, 7, 7, 8, 8, 10], 8))  # [3, 4]
print(search_range([5, 7, 7, 8, 8, 10], 6))  # [-1, -1]
```

### Count Occurrences

```python
def count_occurrences(nums: list[int], target: int) -> int:
    """
    Count how many times target appears.

    Use first and last occurrence.
    Time: O(log n), Space: O(1)
    """
    first = find_first(nums, target)
    if first == -1:
        return 0

    last = find_last(nums, target)
    return last - first + 1

print(count_occurrences([1, 2, 2, 2, 3, 4, 5], 2))  # 3
```

## Pattern 3: Find Insert Position

Find where target should be inserted to maintain sorted order.

### Search Insert Position

```python
def search_insert(nums: list[int], target: int) -> int:
    """
    LeetCode 35: Find index where target should be inserted.

    Returns:
    - Index of target if found
    - Index where target should be inserted otherwise

    Key: When loop ends, left is the insert position!

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # Not found: left is the insert position
    return left

# Examples
print(search_insert([1, 3, 5, 6], 5))  # 2 (found)
print(search_insert([1, 3, 5, 6], 2))  # 1 (insert here)
print(search_insert([1, 3, 5, 6], 7))  # 4 (insert at end)
```

### Find Closest Value

```python
def find_closest_value(nums: list[int], target: int) -> int:
    """
    Find value in array closest to target.

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    closest = nums[0]

    while left <= right:
        mid = left + (right - left) // 2

        # Update closest
        if abs(nums[mid] - target) < abs(closest - target):
            closest = nums[mid]

        if nums[mid] == target:
            return nums[mid]
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return closest

print(find_closest_value([1, 3, 5, 7, 9], 6))  # 5 or 7 (both distance 1)
```

## Pattern 4: Search in Rotated Array

Array is sorted but rotated at some pivot.

### Search in Rotated Sorted Array

```python
def search_rotated(nums: list[int], target: int) -> int:
    """
    LeetCode 33: Search in rotated sorted array (no duplicates).

    Example: [4,5,6,7,0,1,2] (rotated at index 4)

    Strategy:
    1. Find which half is sorted
    2. Check if target is in sorted half
    3. Search appropriate half

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1   # Target in right half
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Target in right half
            else:
                right = mid - 1  # Target in left half

    return -1

# Example
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 0))  # 4
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 3))  # -1
```

### Find Minimum in Rotated Array

```python
def find_min_rotated(nums: list[int]) -> int:
    """
    LeetCode 153: Find minimum in rotated sorted array.

    Key insight: minimum is at the "break point" where
    nums[i] > nums[i+1]

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        # If mid > right, minimum is in right half
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid

    return nums[left]

# Examples
print(find_min_rotated([3, 4, 5, 1, 2]))  # 1
print(find_min_rotated([4, 5, 6, 7, 0, 1, 2]))  # 0
print(find_min_rotated([11, 13, 15, 17]))  # 11 (not rotated)
```

### Find Rotation Point

```python
def find_rotation_count(nums: list[int]) -> int:
    """
    Find how many times array was rotated.
    (Index of minimum element)

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return left

print(find_rotation_count([4, 5, 6, 7, 0, 1, 2]))  # 4 (rotated 4 times)
```

## Pattern 5: Binary Search on Answer

Search for answer in a range, not in array!

### Square Root

```python
def my_sqrt(x: int) -> int:
    """
    LeetCode 69: Integer square root (floor).

    Binary search on answer space [0, x].

    Time: O(log x), Space: O(1)
    """
    if x < 2:
        return x

    left, right = 0, x
    result = 0

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == x:
            return mid
        elif square < x:
            result = mid  # Save this as potential answer
            left = mid + 1
        else:
            right = mid - 1

    return result

# Examples
print(my_sqrt(4))   # 2
print(my_sqrt(8))   # 2 (floor of 2.828...)
print(my_sqrt(16))  # 4
```

### Koko Eating Bananas

```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    """
    LeetCode 875: Minimum speed to eat all bananas in h hours.

    Binary search on speed [1, max(piles)].
    For each speed, check if can finish in h hours.

    Time: O(n log m) where m = max(piles), Space: O(1)
    """
    def can_finish(speed: int) -> bool:
        """Check if can eat all bananas at this speed in h hours."""
        hours = 0
        for pile in piles:
            hours += math.ceil(pile / speed)
        return hours <= h

    left, right = 1, max(piles)
    result = right

    while left <= right:
        mid = left + (right - left) // 2

        if can_finish(mid):
            result = mid
            right = mid - 1  # Try slower speed
        else:
            left = mid + 1   # Need faster speed

    return result

# Example: piles = [3,6,7,11], h = 8
# At speed 4: 1 + 2 + 2 + 3 = 8 hours ✓
# At speed 3: 1 + 2 + 3 + 4 = 10 hours ✗
print(min_eating_speed([3, 6, 7, 11], 8))  # 4
```

### Capacity to Ship Packages

```python
def ship_within_days(weights: list[int], days: int) -> int:
    """
    LeetCode 1011: Minimum ship capacity to ship within D days.

    Binary search on capacity [max(weights), sum(weights)].

    Time: O(n log S) where S = sum(weights), Space: O(1)
    """
    def can_ship(capacity: int) -> bool:
        """Check if can ship all packages in 'days' days."""
        days_needed = 1
        current_load = 0

        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight

        return days_needed <= days

    left, right = max(weights), sum(weights)
    result = right

    while left <= right:
        mid = left + (right - left) // 2

        if can_ship(mid):
            result = mid
            right = mid - 1  # Try smaller capacity
        else:
            left = mid + 1   # Need larger capacity

    return result

print(ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5))  # 15
```

### Split Array Largest Sum

```python
def split_array(nums: list[int], k: int) -> int:
    """
    LeetCode 410: Split array into k subarrays to minimize largest sum.

    Binary search on answer [max(nums), sum(nums)].

    Time: O(n log S) where S = sum(nums), Space: O(1)
    """
    def can_split(max_sum: int) -> bool:
        """Check if can split into k subarrays with each sum <= max_sum."""
        subarrays = 1
        current_sum = 0

        for num in nums:
            if current_sum + num > max_sum:
                subarrays += 1
                current_sum = num
            else:
                current_sum += num

        return subarrays <= k

    left, right = max(nums), sum(nums)
    result = right

    while left <= right:
        mid = left + (right - left) // 2

        if can_split(mid):
            result = mid
            right = mid - 1  # Try smaller max sum
        else:
            left = mid + 1   # Need larger max sum

    return result

print(split_array([7, 2, 5, 10, 8], 2))  # 18 ([7,2,5], [10,8])
```

## Pattern 6: Peak Finding

### Find Peak Element

```python
def find_peak_element(nums: list[int]) -> int:
    """
    LeetCode 162: Find any peak element (nums[i] > neighbors).

    nums[-1] and nums[n] are considered -infinity.

    Key: If nums[mid] < nums[mid+1], peak must be on right!

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] < nums[mid + 1]:
            left = mid + 1  # Peak is on right
        else:
            right = mid     # Peak is on left (or mid itself)

    return left

# Examples
print(find_peak_element([1, 2, 3, 1]))  # 2 (value 3)
print(find_peak_element([1, 2, 1, 3, 5, 6, 4]))  # 1 or 5
```

### Peak Index in Mountain Array

```python
def peak_index_in_mountain_array(arr: list[int]) -> int:
    """
    LeetCode 852: Find peak in mountain array.
    Mountain: increases then decreases, no duplicates.

    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2

        if arr[mid] < arr[mid + 1]:
            left = mid + 1  # Ascending, peak is right
        else:
            right = mid     # Descending or peak

    return left

print(peak_index_in_mountain_array([0, 1, 0]))  # 1
print(peak_index_in_mountain_array([0, 2, 1, 0]))  # 1
```

## Pattern 7: Matrix Binary Search

### Search 2D Matrix (Fully Sorted)

```python
def search_matrix(matrix: list[list[int]], target: int) -> bool:
    """
    LeetCode 74: Search in matrix where:
    - Each row is sorted
    - First element of row > last element of previous row

    Treat as 1D sorted array!

    Time: O(log(m*n)), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = left + (right - left) // 2

        # Convert 1D index to 2D
        row, col = mid // n, mid % n
        mid_value = matrix[row][col]

        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

# Example
matrix = [
    [1,  3,  5,  7],
    [10, 11, 16, 20],
    [23, 30, 34, 60]
]
print(search_matrix(matrix, 3))   # True
print(search_matrix(matrix, 13))  # False
```

### Search 2D Matrix II (Row/Col Sorted)

```python
def search_matrix_ii(matrix: list[list[int]], target: int) -> bool:
    """
    LeetCode 240: Search in matrix where:
    - Each row is sorted left to right
    - Each column is sorted top to bottom

    Start from top-right (or bottom-left).

    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False

    row, col = 0, len(matrix[0]) - 1

    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            row += 1  # Move down
        else:
            col -= 1  # Move left

    return False

# Example
matrix = [
    [1,  4,  7,  11, 15],
    [2,  5,  8,  12, 19],
    [3,  6,  9,  16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
]
print(search_matrix_ii(matrix, 5))   # True
print(search_matrix_ii(matrix, 20))  # False
```

## Advanced Patterns

### Find K Closest Elements

```python
def find_closest_elements(arr: list[int], k: int, x: int) -> list[int]:
    """
    LeetCode 658: Find k closest elements to x.

    Binary search to find best starting position for window of size k.

    Time: O(log n + k), Space: O(1)
    """
    # Search for best left boundary of k-sized window
    left, right = 0, len(arr) - k

    while left < right:
        mid = left + (right - left) // 2

        # Compare: is window starting at mid better than mid+1?
        # Choose based on which gives closer elements
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid

    return arr[left:left + k]

# Example
print(find_closest_elements([1, 2, 3, 4, 5], 4, 3))  # [1,2,3,4]
print(find_closest_elements([1, 2, 3, 4, 5], 4, -1))  # [1,2,3,4]
```

### Median of Two Sorted Arrays

```python
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    LeetCode 4: Find median of two sorted arrays.

    Use binary search to partition arrays correctly.

    Time: O(log(min(m,n))), Space: O(1)
    """
    # Ensure nums1 is shorter
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        partition1 = left + (right - left) // 2
        partition2 = (m + n + 1) // 2 - partition1

        # Handle edge cases
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]

        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]

        # Check if correct partition
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found correct partition
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1

    raise ValueError("Input arrays are not sorted")

# Example
print(find_median_sorted_arrays([1, 3], [2]))  # 2.0
print(find_median_sorted_arrays([1, 2], [3, 4]))  # 2.5
```

## Common Templates

### Template 1: Classic Binary Search

```python
def binary_search_template(nums: list[int], target: int) -> int:
    """Find exact match or -1."""
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### Template 2: Find Boundary (Left/Right)

```python
def find_boundary_template(nums: list[int], target: int, find_left: bool = True) -> int:
    """Find leftmost or rightmost occurrence."""
    left, right = 0, len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            result = mid
            if find_left:
                right = mid - 1  # Continue left
            else:
                left = mid + 1   # Continue right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

### Template 3: Binary Search on Answer

```python
def binary_search_answer_template(check_function, left: int, right: int,
                                   find_minimum: bool = True) -> int:
    """
    Binary search on answer space.

    check_function(mid): returns True if mid is valid answer.
    find_minimum: True to find minimum valid, False for maximum.
    """
    result = right if find_minimum else left

    while left <= right:
        mid = left + (right - left) // 2

        if check_function(mid):
            result = mid
            if find_minimum:
                right = mid - 1  # Try smaller
            else:
                left = mid + 1   # Try larger
        else:
            if find_minimum:
                left = mid + 1   # Need larger
            else:
                right = mid - 1  # Need smaller

    return result
```

## Complexity Analysis

| Pattern | Time Complexity | Space Complexity | Notes |
|---------|----------------|------------------|-------|
| Classic search | O(log n) | O(1) | Each iteration halves space |
| First/Last occurrence | O(log n) | O(1) | May need full log n iterations |
| Rotated array | O(log n) | O(1) | One binary search |
| Binary search on answer | O(n log m) | O(1) | n = validation time, m = range |
| Peak finding | O(log n) | O(1) | Similar to classic |
| 2D matrix (fully sorted) | O(log(mn)) | O(1) | Treat as 1D array |
| 2D matrix (row/col sorted) | O(m + n) | O(1) | Not true binary search |
| Median of two sorted | O(log(min(m,n))) | O(1) | Binary search on shorter |

## When to Use

**Use binary search when:**

1. **Array is sorted** (or partially sorted like rotated array)
2. **Search space is ordered** and can be halved
3. **Looking for optimal value** in bounded range
4. **Checking condition** is faster than iteration
5. **Need O(log n)** instead of O(n)

**Common keywords:**
- "sorted array"
- "find minimum/maximum X such that..."
- "in a certain range"
- "log n time"

**Don't use when:**
- Array is unsorted and can't be sorted
- Need to examine every element anyway
- Simple linear scan is clearer and fast enough

## Common Pitfalls

### 1. Integer Overflow in Mid Calculation

```python
# Wrong (can overflow in some languages):
mid = (left + right) // 2

# Correct:
mid = left + (right - left) // 2
```

### 2. Infinite Loop with Wrong Bounds

```python
# Wrong: left = mid (without +1) can cause infinite loop
while left < right:
    mid = left + (right - left) // 2
    if condition:
        left = mid  # If mid never changes, infinite loop!
    else:
        right = mid - 1

# Correct:
while left < right:
    mid = left + (right - left) // 2
    if condition:
        left = mid + 1  # Always make progress
    else:
        right = mid
```

### 3. Off-by-One in Loop Condition

```python
# Use left <= right when you need to check when left == right
while left <= right:
    # ...

# Use left < right when you want to stop before they meet
while left < right:
    # ...
```

### 4. Not Handling Edge Cases

```python
def binary_search(nums: list[int], target: int) -> int:
    # Wrong: crashes on empty array
    # left, right = 0, len(nums) - 1

    # Correct: check empty first
    if not nums:
        return -1

    left, right = 0, len(nums) - 1
    # ...
```

### 5. Wrong Comparison in Rotated Array

```python
# Wrong: comparing with middle instead of boundaries
if nums[mid] < nums[left]:  # Incorrect logic

# Correct:
if nums[left] <= nums[mid]:
    # Left half is sorted
```

## Practice Problems

### Easy
1. **Binary Search** (LeetCode 704)
2. **Search Insert Position** (LeetCode 35)
3. **Sqrt(x)** (LeetCode 69)
4. **First Bad Version** (LeetCode 278)
5. **Valid Perfect Square** (LeetCode 367)
6. **Guess Number Higher or Lower** (LeetCode 374)
7. **Arranging Coins** (LeetCode 441)
8. **Two Sum II - Input Array Is Sorted** (LeetCode 167) - two pointers

### Medium
9. **Find First and Last Position of Element** (LeetCode 34)
10. **Search in Rotated Sorted Array** (LeetCode 33)
11. **Find Minimum in Rotated Sorted Array** (LeetCode 153)
12. **Find Peak Element** (LeetCode 162)
13. **Search a 2D Matrix** (LeetCode 74)
14. **Search a 2D Matrix II** (LeetCode 240)
15. **Koko Eating Bananas** (LeetCode 875)
16. **Capacity To Ship Packages Within D Days** (LeetCode 1011)
17. **Find K Closest Elements** (LeetCode 658)
18. **Time Based Key-Value Store** (LeetCode 981)
19. **Single Element in a Sorted Array** (LeetCode 540)
20. **Peak Index in a Mountain Array** (LeetCode 852)

### Hard
21. **Median of Two Sorted Arrays** (LeetCode 4)
22. **Split Array Largest Sum** (LeetCode 410)
23. **Find Minimum in Rotated Sorted Array II** (LeetCode 154) - with duplicates
24. **Aggressive Cows** (SPOJ)
25. **Painter's Partition Problem**

## Additional Resources

### Visualizations
- **VisuAlgo**: Binary Search visualization
- **Algorithm Visualizer**: Interactive binary search

### Tutorials
- **LeetCode Explore**: Binary Search card
- **GeeksforGeeks**: Binary Search variations
- **TopCoder**: Binary Search tutorial

### Articles
- "Powerful Ultimate Binary Search Template" - LeetCode Discuss
- "Binary Search 101" - The Ultimate Binary Search Handbook
- "How to Binary Search on Answer" - CP-Algorithms

### Practice Platforms
- LeetCode Tag: Binary Search (100+ problems)
- Codeforces: Binary Search problems
- AtCoder: Binary Search practice

---

**Key Takeaways:**
1. Binary search is O(log n) - extremely efficient
2. Works on sorted or monotonic search spaces
3. Three main patterns: find exact, find boundary, search on answer
4. Always maintain invariant: answer is in [left, right]
5. Watch for off-by-one errors and infinite loops
6. Applicable beyond arrays: any searchable space

Master binary search patterns and you'll efficiently solve optimization and search problems that would otherwise require expensive linear scans!
