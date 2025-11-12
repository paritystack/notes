# Sorting Algorithms

## Overview

Sorting arranges elements in order. Different algorithms have different trade-offs in speed, memory, and stability.

## Common Algorithms

### Bubble Sort

**Time**: O(n²) | **Space**: O(1)

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

### Selection Sort

**Time**: O(n²) | **Space**: O(1)

```python
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

### Insertion Sort

**Time**: O(n²) | **Space**: O(1) | **Best**: O(n)

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

### Merge Sort

**Time**: O(n log n) | **Space**: O(n) | **Stable**: ✓

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
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Quick Sort

**Time**: O(n log n) avg, O(n²) worst | **Space**: O(log n)

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
```

### Heap Sort

**Time**: O(n log n) | **Space**: O(1)

```python
def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr
```

## Comparison

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| **Bubble** | O(n) | O(n²) | O(n²) | O(1) | ✓ |
| **Selection** | O(n²) | O(n²) | O(n²) | O(1) | ✗ |
| **Insertion** | O(n) | O(n²) | O(n²) | O(1) | ✓ |
| **Merge** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✓ |
| **Quick** | O(n log n) | O(n log n) | O(n²) | O(log n) | ✗ |
| **Heap** | O(n log n) | O(n log n) | O(n log n) | O(1) | ✗ |

## When to Use

- **Insertion Sort**: Small arrays, nearly sorted
- **Merge Sort**: Need stability, external sorting
- **Quick Sort**: General purpose, good cache
- **Heap Sort**: Guaranteed O(n log n), no extra space

## Python Built-in

```python
# Best for most cases
arr.sort()  # In-place, O(n log n)
sorted(arr)  # Returns new list

# Custom comparator
arr.sort(key=lambda x: x['age'])
```

## ELI10

Different sorting strategies:
- **Bubble**: Compare neighbors (slow)
- **Quick**: Pick pivot, divide and conquer (fast)
- **Merge**: Split in half, merge back (reliable)

Use built-in sorts unless learning!

## Further Resources

- [Sorting Algorithms Visualized](https://www.sortvisualizer.com/)
- [Big-O Cheatsheet](https://www.bigocheatsheet.com/)
