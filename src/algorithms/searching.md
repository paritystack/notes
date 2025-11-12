# Searching Algorithms

## Overview

Searching algorithms help find elements in data structures. The choice depends on whether data is sorted and the size of the data.

## Linear Search

**Time**: O(n) | **Space**: O(1) | **Works on**: Unsorted arrays

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**When to use**: Small arrays, unsorted data

## Binary Search

**Time**: O(log n) | **Space**: O(1) | **Requires**: Sorted array

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found
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
```

## Two Pointer Technique

**Time**: O(n) | **Space**: O(1)

```python
def two_sum(arr, target):
    """Find two numbers that sum to target"""
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
```

## Jump Search

**Time**: O(n) | **Space**: O(1) | **Requires**: Sorted array

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

## Interpolation Search

**Time**: O(log log n) average, O(n) worst | **Requires**: Sorted uniformly distributed data

```python
def interpolation_search(arr, target):
    left, right = 0, len(arr) - 1

    while (left <= right and
           target >= arr[left] and
           target <= arr[right]):

        # Estimate position
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

## Exponential Search

**Time**: O(log n) | **Space**: O(1) | **Requires**: Sorted array

```python
def exponential_search(arr, target):
    n = len(arr)

    # Find range
    i = 1
    while i < n and arr[i] < target:
        i *= 2

    # Binary search in range
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

## Sentinel Search

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

## Comparison

| Algorithm | Time (Avg) | Time (Worst) | Space | Requires Sorted |
|-----------|-----------|-------------|-------|-----------------|
| Linear | O(n) | O(n) | O(1) | No |
| Binary | O(log n) | O(log n) | O(1) | Yes |
| Jump | O(n) | O(n) | O(1) | Yes |
| Interpolation | O(log log n) | O(n) | O(1) | Yes |
| Exponential | O(log n) | O(log n) | O(1) | Yes |

## Key Takeaways

1. **Unsorted data?** Use Linear Search or Hash Table
2. **Sorted data?** Use Binary Search for O(log n)
3. **Uniformly distributed?** Try Interpolation Search
4. **Need flexibility?** Build a Hash Table for O(1) lookup

## ELI10

Imagine finding a word in a dictionary:
- **Linear Search**: Check every word from start (slow!)
- **Binary Search**: Open middle, go left or right, repeat (fast!)
- **Interpolation**: Estimate where word should be based on first letter

Binary search is fastest for sorted data!

## Further Resources

- [LeetCode Binary Search](https://leetcode.com/tag/binary-search/)
- [Searching Algorithms Visualization](https://visualgo.net/en/search)
