# Sorting Algorithms

## Overview

Sorting arranges elements in order (ascending or descending). Different algorithms have different trade-offs in speed, memory usage, and stability. Understanding these trade-offs helps you choose the right algorithm for your specific use case.

## Algorithm Characteristics

Before diving into specific algorithms, it's important to understand key characteristics:

### Stability
A sorting algorithm is **stable** if it preserves the relative order of elements with equal keys. For example, if you sort student records by grade, stable algorithms keep students with the same grade in their original order.

**Stable**: Bubble Sort, Insertion Sort, Merge Sort, Tim Sort, Counting Sort, Radix Sort
**Unstable**: Selection Sort, Quick Sort, Heap Sort

### In-Place vs Out-of-Place
**In-place** algorithms sort with O(1) or O(log n) extra space, modifying the input array directly.
**Out-of-place** algorithms require O(n) or more additional space.

### Adaptiveness
**Adaptive** algorithms perform better on partially sorted data. They can achieve better than their worst-case complexity when input has some order.

**Adaptive**: Bubble Sort, Insertion Sort, Tim Sort
**Non-adaptive**: Selection Sort, Merge Sort, Heap Sort, Quick Sort (standard)

### Comparison-Based vs Non-Comparison
**Comparison-based** algorithms can only compare elements. They have a theoretical lower bound of O(n log n) for average complexity.
**Non-comparison** algorithms use properties like integer values to achieve linear time in certain cases.

## Common Algorithms

### Bubble Sort

**Time**: $O(n^2)$ average/worst, $O(n)$ best | **Space**: $O(1)$ | **Stable**: ✓ | **Adaptive**: ✓

#### How It Works
Bubble Sort repeatedly steps through the list, comparing adjacent elements and swapping them if they're in the wrong order. The algorithm gets its name because smaller elements "bubble" to the top (beginning) of the list. After each pass, the largest unsorted element "bubbles" to its correct position at the end.

The algorithm continues until no more swaps are needed, indicating the array is sorted. An optimization adds a flag to detect when the array becomes sorted early, giving O(n) best-case performance on already-sorted data.

**Why it's stable**: Equal elements are never swapped (we only swap when `arr[j] > arr[j+1]`), preserving their original order.

**When to use**: Educational purposes, very small datasets (< 10 elements), or when you need a simple implementation and performance isn't critical.

**When NOT to use**: Any practical application with more than a handful of elements. It's one of the slowest sorting algorithms.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: stop if no swaps occurred
            break
    return arr
```

#### Step-by-Step Example
Sorting `[64, 34, 25, 12, 22]`:

**Pass 1:**
- [**64, 34**, 25, 12, 22] → [34, 64, 25, 12, 22] (swap)
- [34, **64, 25**, 12, 22] → [34, 25, 64, 12, 22] (swap)
- [34, 25, **64, 12**, 22] → [34, 25, 12, 64, 22] (swap)
- [34, 25, 12, **64, 22**] → [34, 25, 12, 22, 64] (swap)
- Result: [34, 25, 12, 22, **64**] (64 is now in final position)

**Pass 2:**
- [**34, 25**, 12, 22, 64] → [25, 34, 12, 22, 64] (swap)
- [25, **34, 12**, 22, 64] → [25, 12, 34, 22, 64] (swap)
- [25, 12, **34, 22**, 64] → [25, 12, 22, 34, 64] (swap)
- Result: [25, 12, 22, **34, 64**] (34 is now in final position)

**Pass 3:**
- [**25, 12**, 22, 34, 64] → [12, 25, 22, 34, 64] (swap)
- [12, **25, 22**, 34, 64] → [12, 22, 25, 34, 64] (swap)
- Result: [12, 22, **25, 34, 64**]

**Pass 4:**
- [**12, 22**, 25, 34, 64] → [12, 22, 25, 34, 64] (no swap)
- Result: [**12, 22, 25, 34, 64**] (sorted!)

### Selection Sort

**Time**: $O(n^2)$ all cases | **Space**: $O(1)$ | **Stable**: ✗ | **Adaptive**: ✗

#### How It Works
Selection Sort divides the array into sorted and unsorted regions. It repeatedly finds the minimum element from the unsorted region and places it at the beginning of that region. Unlike Bubble Sort, Selection Sort performs exactly n-1 passes regardless of input, making it non-adaptive.

The algorithm maintains two subarrays: the sorted portion (at the start) and the unsorted portion (the rest). In each iteration, it selects the smallest element from the unsorted portion and swaps it with the first unsorted element.

**Why it's unstable**: The long-distance swaps can change the relative order of equal elements. For example, [5a, 5b, 2] becomes [2, 5b, 5a] after the first pass.

**When to use**: When write operations are expensive (like EEPROM) since it minimizes the number of swaps (at most n-1). Also useful when you need predictable performance regardless of input order.

**When NOT to use**: When stability matters, or when you have a better algorithm available (which is almost always).

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

#### Step-by-Step Example
Sorting `[64, 25, 12, 22, 11]`:

**Pass 1:** Find minimum in [64, 25, 12, 22, 11]
- Minimum is **11** at index 4
- Swap with position 0: [**11**, 25, 12, 22, 64]
- Sorted: [11] | Unsorted: [25, 12, 22, 64]

**Pass 2:** Find minimum in [25, 12, 22, 64]
- Minimum is **12** at index 2
- Swap with position 1: [11, **12**, 25, 22, 64]
- Sorted: [11, 12] | Unsorted: [25, 22, 64]

**Pass 3:** Find minimum in [25, 22, 64]
- Minimum is **22** at index 3
- Swap with position 2: [11, 12, **22**, 25, 64]
- Sorted: [11, 12, 22] | Unsorted: [25, 64]

**Pass 4:** Find minimum in [25, 64]
- Minimum is **25** at index 3 (already in place)
- Swap with itself: [11, 12, 22, **25**, 64]
- Sorted: [11, 12, 22, 25] | Unsorted: [64]

**Final:** [11, 12, 22, 25, 64] (sorted!)

### Insertion Sort

**Time**: $O(n^2)$ average/worst, $O(n)$ best | **Space**: $O(1)$ | **Stable**: ✓ | **Adaptive**: ✓

#### How It Works
Insertion Sort builds the final sorted array one element at a time. It works similarly to how you might sort playing cards in your hands. The algorithm maintains a sorted subarray and repeatedly picks the next element from the unsorted portion, inserting it into the correct position in the sorted portion.

For each element, the algorithm shifts all larger elements in the sorted portion one position to the right, then inserts the current element into the gap created. This makes it very efficient for nearly sorted data (O(n) when array is already sorted or has few inversions).

**Why it's stable**: Elements are only moved when they're strictly greater than the key, never when equal. This preserves the original order of equal elements.

**Why it's adaptive**: On nearly sorted data, the inner while loop performs very few iterations, approaching O(n) total time.

**When to use**: Small arrays (< 50 elements), nearly sorted data, online sorting (elements arrive one at a time), or as the base case in hybrid algorithms like Tim Sort.

**When NOT to use**: Large arrays with random data (use Quick Sort or Merge Sort instead).

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

#### Step-by-Step Example
Sorting `[12, 11, 13, 5, 6]`:

**Initial:** [**12**] | [11, 13, 5, 6]
- Sorted: [12] (first element is trivially sorted)

**Step 1:** Insert 11
- [**12**, 11, 13, 5, 6] → key = 11
- 12 > 11, shift 12 right: [_, **12**, 13, 5, 6]
- Insert 11: [**11, 12**] | [13, 5, 6]

**Step 2:** Insert 13
- [11, 12, **13**, 5, 6] → key = 13
- 12 < 13, no shift needed
- Result: [**11, 12, 13**] | [5, 6]

**Step 3:** Insert 5
- [11, 12, 13, **5**, 6] → key = 5
- 13 > 5, shift right: [11, 12, _, **13**, 6]
- 12 > 5, shift right: [11, _, **12**, 13, 6]
- 11 > 5, shift right: [_, **11**, 12, 13, 6]
- Insert 5: [**5, 11, 12, 13**] | [6]

**Step 4:** Insert 6
- [5, 11, 12, 13, **6**] → key = 6
- 13 > 6, shift right: [5, 11, 12, _, **13**]
- 12 > 6, shift right: [5, 11, _, **12**, 13]
- 11 > 6, shift right: [5, _, **11**, 12, 13]
- 5 < 6, stop
- Insert 6: [**5, 6, 11, 12, 13**] (sorted!)

### Merge Sort

**Time**: $O(n \log n)$ all cases | **Space**: $O(n)$ | **Stable**: ✓ | **Adaptive**: ✗

#### How It Works
Merge Sort is a divide-and-conquer algorithm that divides the array into two halves, recursively sorts them, and then merges the two sorted halves. The recursion continues until subarrays of size 1 are reached (which are trivially sorted).

The magic happens in the merge step: two sorted arrays are combined by repeatedly taking the smaller of the two front elements. This guarantees O(n log n) time in all cases because:
- Dividing takes O(log n) levels (binary division)
- Merging at each level processes all n elements
- Total: O(n) × O(log n) = O(n log n)

**Why it's stable**: When merging, if elements are equal, we take from the left array first, preserving original order.

**Why it uses O(n) space**: The merge operation needs temporary arrays to store the merged result. In-place merge sort exists but is complex and slower.

**When to use**: When you need guaranteed O(n log n) time, stability is required, working with linked lists (can be done in O(1) extra space), external sorting (data doesn't fit in memory), or parallelization is needed.

**When NOT to use**: Space is extremely constrained (use Heap Sort), or you need the fastest average-case performance (use Quick Sort).

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
        if left[i] <= right[j]:  # <= ensures stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

#### Step-by-Step Example
Sorting `[38, 27, 43, 3]`:

**Divide Phase:**
```
                [38, 27, 43, 3]
                /              \
        [38, 27]                [43, 3]
        /      \                /      \
     [38]     [27]           [43]     [3]
```

**Conquer Phase (Merge):**

**Level 1:** Merge single elements
- Merge [38] and [27]:
  - Compare 38 vs 27 → take 27: [27]
  - Take remaining 38: [**27, 38**]

- Merge [43] and [3]:
  - Compare 43 vs 3 → take 3: [3]
  - Take remaining 43: [**3, 43**]

**Level 2:** Merge sorted pairs
- Merge [27, 38] and [3, 43]:
  - Compare 27 vs 3 → take 3: [3]
  - Compare 27 vs 43 → take 27: [3, 27]
  - Compare 38 vs 43 → take 38: [3, 27, 38]
  - Take remaining 43: [**3, 27, 38, 43**]

**Result:** [3, 27, 38, 43] (sorted!)

**Detailed merge of [27, 38] and [3, 43]:**
```
Left:  [27, 38]    Right: [3, 43]    Result: []
       ^                   ^
       i                   j
Compare 27 vs 3 → take 3

Left:  [27, 38]    Right: [3, 43]    Result: [3]
       ^                      ^
       i                      j
Compare 27 vs 43 → take 27

Left:  [27, 38]    Right: [3, 43]    Result: [3, 27]
           ^                   ^
           i                   j
Compare 38 vs 43 → take 38

Left:  [27, 38]    Right: [3, 43]    Result: [3, 27, 38]
               ^               ^
               i               j
Left exhausted, take remaining 43: [3, 27, 38, 43]
```

### Quick Sort

**Time**: $O(n \log n)$ average, $O(n^2)$ worst | **Space**: $O(\log n)$ | **Stable**: ✗ | **Adaptive**: ✗

#### How It Works
Quick Sort is a divide-and-conquer algorithm that selects a 'pivot' element and partitions the array around it. All elements smaller than the pivot go to its left, all larger elements go to its right. The algorithm then recursively sorts the left and right partitions.

The key insight is that after partitioning, the pivot is in its final sorted position. The efficiency comes from this in-place partitioning and the fact that average-case behavior divides the array roughly in half each time.

**Pivot selection strategies:**
- First/last element: Simple but O(n²) on sorted data
- Middle element: Better than first/last
- Random: Good average case, avoids worst-case patterns
- Median-of-three: Takes median of first, middle, last elements

**Why it's unstable**: The partitioning process makes long-distance swaps that can change relative order of equal elements.

**Why O(n²) worst case**: When pivot is always the smallest/largest element (e.g., sorted array with poor pivot choice), we get unbalanced partitions: n + (n-1) + (n-2) + ... = O(n²).

**Why O(log n) space**: Recursion depth is log n on average (balanced partitions), but can be O(n) worst case.

**When to use**: General-purpose sorting, large datasets, when average-case performance matters most, or when you need in-place sorting with good cache locality.

**When NOT to use**: Worst-case guarantees are critical (use Heap Sort or Merge Sort), stability is required, or data is already sorted/nearly sorted without randomization.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# In-place version (more efficient)
def quick_sort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

#### Step-by-Step Example
Sorting `[10, 7, 8, 9, 1, 5]` with last element as pivot:

**Initial:** [10, 7, 8, 9, 1, 5]
- Pivot = 5 (last element)

**Partition around 5:**
- i = -1 (tracks position of smaller elements)
- j scans from left:
  - j=0: 10 > 5, skip
  - j=1: 7 > 5, skip
  - j=2: 8 > 5, skip
  - j=3: 9 > 5, skip
  - j=4: 1 ≤ 5, i=0, swap arr[0] and arr[4]: [**1**, 7, 8, 9, 10, 5]
- Place pivot: swap arr[i+1] and arr[high]: [1, **5**, 8, 9, 10, 7]
- Pivot 5 is now at index 1 in final position

**Recursion tree:**
```
[1, 5, 8, 9, 10, 7]
   ↓      ↓
  [1]   [8, 9, 10, 7]
```

**Sort [8, 9, 10, 7]:**
- Pivot = 7
- Partition: i=-1
  - j=0: 8 > 7, skip
  - j=1: 9 > 7, skip
  - j=2: 10 > 7, skip
- Place pivot: [**7**, 9, 10, 8] (pivot at index 0)
- Sort [] and [9, 10, 8]

**Sort [9, 10, 8]:**
- Pivot = 8
- Partition: [**8**, 10, 9]
- Sort [] and [10, 9]

**Sort [10, 9]:**
- Pivot = 9
- Partition: [**9**, 10]

**Final result:** [1, 5, 7, 8, 9, 10]

### Heap Sort

**Time**: $O(n \log n)$ all cases | **Space**: $O(1)$ | **Stable**: ✗ | **Adaptive**: ✗

#### How It Works
Heap Sort uses a binary heap data structure (specifically a max-heap) to sort elements. The algorithm has two main phases:

1. **Build Max-Heap** (O(n)): Transform the array into a max-heap where each parent node is greater than its children. This ensures the largest element is at the root (index 0).

2. **Extract Max and Heapify** (O(n log n)): Repeatedly swap the root (maximum) with the last element, reduce heap size by 1, and restore the heap property by "heapifying" down from the root.

**Binary heap properties:**
- For element at index i:
  - Left child at 2i + 1
  - Right child at 2i + 2
  - Parent at (i - 1) / 2
- Max-heap: Every parent ≥ its children

**Why it's unstable**: Elements are moved long distances during heap operations, losing their relative order.

**Why it's not adaptive**: The algorithm always builds the heap and performs n heap extractions regardless of input order.

**When to use**: Guaranteed O(n log n) time with O(1) space, embedded systems with memory constraints, or when you need predictable performance without recursion overhead.

**When NOT to use**: Stability matters, you need the fastest average case (Quick Sort is faster), or you're implementing a priority queue (use heap data structure directly, not heap sort).

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

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Swap max to end
        heapify(arr, i, 0)  # Heapify reduced heap

    return arr
```

#### Step-by-Step Example
Sorting `[4, 10, 3, 5, 1]`:

**Phase 1: Build Max-Heap**

Initial array: [4, 10, 3, 5, 1]
```
       4
      / \
    10   3
   / \
  5   1
```

Start from last non-leaf node (index n//2 - 1 = 1):

**Heapify at index 1 (value 10):**
- Children: 5 (left), 1 (right)
- 10 > 5 and 10 > 1, no change
- [4, **10**, 3, 5, 1]

**Heapify at index 0 (value 4):**
- Children: 10 (left), 3 (right)
- 10 > 4, swap with 10: [**10**, 4, 3, 5, 1]
- Now heapify at index 1 (value 4):
  - Children: 5 (left), 1 (right)
  - 5 > 4, swap with 5: [10, **5**, 3, 4, 1]

Max-heap built: [10, 5, 3, 4, 1]
```
       10
      /  \
     5    3
    / \
   4   1
```

**Phase 2: Sort by extracting max**

**Iteration 1:**
- Swap 10 and 1: [1, 5, 3, 4, | **10**]
- Heapify first 4 elements:
  - 1 < max(5, 3), swap with 5: [5, 1, 3, 4, | 10]
  - 1 < 4, swap with 4: [5, 4, 3, 1, | 10]

**Iteration 2:**
- Swap 5 and 1: [1, 4, 3, | **5, 10**]
- Heapify first 3 elements:
  - 1 < max(4, 3), swap with 4: [4, 1, 3, | 5, 10]

**Iteration 3:**
- Swap 4 and 3: [3, 1, | **4, 5, 10**]
- Heapify first 2 elements:
  - 3 > 1, no change: [3, 1, | 4, 5, 10]

**Iteration 4:**
- Swap 3 and 1: [1, | **3, 4, 5, 10**]

**Final sorted array:** [1, 3, 4, 5, 10]

### Counting Sort

**Time**: $O(n + k)$ where k = range | **Space**: $O(k)$ | **Stable**: ✓ | **Comparison**: ✗

#### How It Works
Counting Sort is a non-comparison sorting algorithm that works by counting the occurrences of each distinct element. It then uses arithmetic to determine the positions of elements in the output array. This algorithm exploits the fact that if you know there are 3 elements less than x, then x belongs at position 4.

**Algorithm steps:**
1. Find the range: determine min and max values
2. Count occurrences: create a count array where count[i] = number of times (min + i) appears
3. Compute cumulative counts: convert counts to positions by making each count[i] = sum of all previous counts
4. Place elements: iterate through input array backwards (for stability), place each element at its position, and decrement position counter

**Why it's stable**: By iterating backwards and carefully managing positions, equal elements maintain their relative order.

**Why O(n + k)**: We scan the array O(n), create and process count array O(k), and place elements O(n).

**When to use**: Integers within a small range, frequency counting, as a subroutine in Radix Sort, or when you need linear time sorting and know the range is reasonable.

**When NOT to use**: Large range (k >> n) wastes memory and time, floating-point numbers, strings (use Radix Sort), or when you don't know the range in advance.

```python
def counting_sort(arr):
    if not arr:
        return arr

    # Find range
    min_val = min(arr)
    max_val = max(arr)
    range_size = max_val - min_val + 1

    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1

    # Compute cumulative counts (positions)
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Place elements in output (iterate backwards for stability)
    output = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        num = arr[i]
        pos = count[num - min_val] - 1
        output[pos] = num
        count[num - min_val] -= 1

    return output

# Simplified version (not stable, but easier to understand)
def counting_sort_simple(arr):
    if not arr:
        return arr

    min_val = min(arr)
    max_val = max(arr)
    count = [0] * (max_val - min_val + 1)

    # Count occurrences
    for num in arr:
        count[num - min_val] += 1

    # Reconstruct array
    result = []
    for i, freq in enumerate(count):
        result.extend([i + min_val] * freq)

    return result
```

#### Step-by-Step Example
Sorting `[4, 2, 2, 8, 3, 3, 1]`:

**Step 1: Find range**
- min = 1, max = 8
- range = 8 - 1 + 1 = 8

**Step 2: Count occurrences**
```
Original: [4, 2, 2, 8, 3, 3, 1]
Value:     1  2  3  4  5  6  7  8
Count:    [1, 2, 2, 1, 0, 0, 0, 1]
           ↑  ↑  ↑  ↑           ↑
          (1)(2)(3)(4)         (8)
```

**Step 3: Compute cumulative counts (positions)**
```
Before: [1, 2, 2, 1, 0, 0, 0, 1]
After:  [1, 3, 5, 6, 6, 6, 6, 7]
         ↑  ↑  ↑  ↑           ↑
    1 goes  2 goes  3 goes  4 goes  8 goes
    at pos 0 at 1-2  at 3-4  at 5   at 6
```
Interpretation: count[i] = "how many elements are ≤ value i"

**Step 4: Place elements (backwards for stability)**

Initialize output: [_, _, _, _, _, _, _]

Process from right to left:
- `arr[6] = 1`: count[0] = 1, place at pos 0: [**1**, _, _, _, _, _, _], count[0] = 0
- `arr[5] = 3`: count[2] = 5, place at pos 4: [1, _, _, _, **3**, _, _], count[2] = 4
- `arr[4] = 3`: count[2] = 4, place at pos 3: [1, _, _, **3**, 3, _, _], count[2] = 3
- `arr[3] = 8`: count[7] = 7, place at pos 6: [1, _, _, 3, 3, _, **8**], count[7] = 6
- `arr[2] = 2`: count[1] = 3, place at pos 2: [1, _, **2**, 3, 3, _, 8], count[1] = 2
- `arr[1] = 2`: count[1] = 2, place at pos 1: [1, **2**, 2, 3, 3, _, 8], count[1] = 1
- `arr[0] = 4`: count[3] = 6, place at pos 5: [1, 2, 2, 3, 3, **4**, 8], count[3] = 5

**Final sorted array:** [1, 2, 2, 3, 3, 4, 8]

### Radix Sort

**Time**: $O(d \cdot (n + k))$ where d = digits, k = base | **Space**: $O(n + k)$ | **Stable**: ✓ | **Comparison**: ✗

#### How It Works
Radix Sort sorts numbers by processing individual digits. It uses a stable sorting algorithm (typically Counting Sort) as a subroutine to sort by each digit position, starting from the least significant digit (LSD) to the most significant digit (MSD).

**Why it works:** Stable sorting is crucial. When we sort by a less significant digit, elements with equal values in that position maintain their relative order (which was determined by even less significant digits). By the time we finish processing all digits, the array is fully sorted.

**Two variants:**
- **LSD (Least Significant Digit)**: Start from rightmost digit, more common, simpler
- **MSD (Most Significant Digit)**: Start from leftmost digit, can short-circuit on strings

**Choosing the base (k):**
- Binary (base 2): Many passes but simple
- Decimal (base 10): Natural for humans
- Base 256: Often optimal for computer implementations (process byte by byte)

**Time complexity:** For n numbers with d digits in base k:
- Each pass: O(n + k) using Counting Sort
- Total passes: d
- Overall: O(d × (n + k))

For fixed-length integers (d constant), this is effectively O(n).

**When to use**: Integers with fixed or known maximum digits, strings of similar length, when all values fit in memory, or when you need linear time for fixed-range integers.

**When NOT to use**: Variable-length data with large variations, floating-point numbers (need special handling), small datasets (overhead not worth it), or when memory for counting arrays is limited.

```python
def radix_sort(arr):
    if not arr:
        return arr

    # Find maximum to determine number of digits
    max_val = max(arr)

    # Do counting sort for every digit
    exp = 1  # 10^0, 10^1, 10^2, ...
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9

    # Count occurrences of digits
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1

    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build output array (backwards for stability)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    # Copy output to original array
    for i in range(n):
        arr[i] = output[i]
```

#### Step-by-Step Example
Sorting `[170, 45, 75, 90, 802, 24, 2, 66]`:

**Pass 1: Sort by ones digit (exp = 1)**
```
Numbers:  170  45  75  90  802  24   2  66
Ones:       0   5   5   0    2   4   2   6
           ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓
Result:   170  90 802   2  24  45  75  66
```
Grouped by ones digit: 170,90 (0) → 802,2 (2) → 24 (4) → 45,75 (5) → 66 (6)

**Pass 2: Sort by tens digit (exp = 10)**
```
Numbers:  170  90  802   2  24  45  75  66
Tens:       7   9    0   0   2   4   7   6
           ↓   ↓    ↓   ↓   ↓   ↓   ↓   ↓
Result:   802   2  24  45  66 170  75  90
```
Grouped by tens digit: 802,2 (0) → 24 (2) → 45 (4) → 66 (6) → 170,75 (7) → 90 (9)

**Pass 3: Sort by hundreds digit (exp = 100)**
```
Numbers:  802   2  24  45  66 170  75  90
Hundreds:   8   0   0   0   0   1   0   0
           ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
Result:     2  24  45  66  75  90 170 802
```
Grouped by hundreds digit: 2,24,45,66,75,90 (0) → 170 (1) → 802 (8)

**Final sorted array:** [2, 24, 45, 66, 75, 90, 170, 802]

**Detailed Pass 1 using Counting Sort:**
```
Input: [170, 45, 75, 90, 802, 24, 2, 66]
Extract ones digits: [0, 5, 5, 0, 2, 4, 2, 6]

Count array (digit frequency):
Digit: 0  1  2  3  4  5  6  7  8  9
Count: 2  0  2  0  1  2  1  0  0  0

Cumulative count:
       2  2  4  4  5  7  8  8  8  8

Place elements (backwards):
66 → digit 6 → position 7 → output[7] = 66
2 → digit 2 → position 3 → output[3] = 2
24 → digit 4 → position 4 → output[4] = 24
802 → digit 2 → position 2 → output[2] = 802
90 → digit 0 → position 1 → output[1] = 90
75 → digit 5 → position 6 → output[6] = 75
45 → digit 5 → position 5 → output[5] = 45
170 → digit 0 → position 0 → output[0] = 170

Output: [170, 90, 802, 2, 24, 45, 75, 66]
```

### Bucket Sort

**Time**: $O(n + k)$ average, $O(n^2)$ worst | **Space**: $O(n + k)$ | **Stable**: ✓ | **Comparison**: ✗

#### How It Works
Bucket Sort distributes elements into several "buckets" (usually arrays or lists), sorts each bucket individually (often with insertion sort), and then concatenates the buckets. It works best when input is uniformly distributed across a range.

**Algorithm steps:**
1. Create k empty buckets
2. Distribute elements into buckets based on a mapping function (e.g., floor(n * value / max_value))
3. Sort each bucket individually (often using insertion sort since buckets are small)
4. Concatenate all buckets in order

**Why it's fast on uniform data:** If n elements are evenly distributed into k buckets, each bucket has ~n/k elements. Sorting each with insertion sort (O(m²)) gives O((n/k)²) per bucket, total O(k × (n/k)²) = O(n²/k). When k = n, this becomes O(n).

**Why worst case is O(n²):** If all elements go into one bucket, we're just doing insertion sort on n elements.

**Choosing bucket count:**
- k = n gives best average case
- k too small: buckets become large, slow to sort
- k too large: overhead from many empty buckets

**When to use**: Data is uniformly distributed (e.g., random floats between 0 and 1), external sorting, parallel processing (buckets can be sorted independently), or when you need stable linear-time sorting for uniform data.

**When NOT to use**: Data has skewed distribution (many elements in few buckets), data range unknown, or small datasets (overhead not worth it).

```python
def bucket_sort(arr):
    if not arr:
        return arr

    # Create buckets
    bucket_count = len(arr)
    max_val = max(arr)
    min_val = min(arr)

    # Handle edge case of all same values
    if max_val == min_val:
        return arr

    bucket_range = (max_val - min_val) / bucket_count
    buckets = [[] for _ in range(bucket_count)]

    # Distribute elements into buckets
    for num in arr:
        # Calculate bucket index
        index = int((num - min_val) / bucket_range)
        # Handle edge case where num == max_val
        if index == bucket_count:
            index -= 1
        buckets[index].append(num)

    # Sort individual buckets and concatenate
    result = []
    for bucket in buckets:
        # Use insertion sort for small buckets
        result.extend(insertion_sort(bucket))

    return result

# For floating-point numbers in [0, 1)
def bucket_sort_normalized(arr):
    if not arr:
        return arr

    n = len(arr)
    buckets = [[] for _ in range(n)]

    # Distribute into buckets
    for num in arr:
        index = int(n * num)
        if index == n:  # Handle 1.0
            index = n - 1
        buckets[index].append(num)

    # Sort and concatenate
    result = []
    for bucket in buckets:
        result.extend(sorted(bucket))  # Can use insertion_sort

    return result
```

#### Step-by-Step Example
Sorting `[0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]` (normalized floats):

Using 5 buckets for n=10:

**Step 1: Create buckets**
```
Bucket 0: [0.0, 0.2) → []
Bucket 1: [0.2, 0.4) → []
Bucket 2: [0.4, 0.6) → []
Bucket 3: [0.6, 0.8) → []
Bucket 4: [0.8, 1.0] → []
```

**Step 2: Distribute elements**
- 0.78 → index = 10 * 0.78 = 7 → bucket[3]
- 0.17 → index = 10 * 0.17 = 1 → bucket[0]
- 0.39 → index = 10 * 0.39 = 3 → bucket[1]
- 0.26 → index = 10 * 0.26 = 2 → bucket[1]
- 0.72 → index = 10 * 0.72 = 7 → bucket[3]
- 0.94 → index = 10 * 0.94 = 9 → bucket[4]
- 0.21 → index = 10 * 0.21 = 2 → bucket[1]
- 0.12 → index = 10 * 0.12 = 1 → bucket[0]
- 0.23 → index = 10 * 0.23 = 2 → bucket[1]
- 0.68 → index = 10 * 0.68 = 6 → bucket[3]

```
Bucket 0: [0.17, 0.12]
Bucket 1: [0.39, 0.26, 0.21, 0.23]
Bucket 2: []
Bucket 3: [0.78, 0.72, 0.68]
Bucket 4: [0.94]
```

**Step 3: Sort each bucket (using insertion sort)**

Bucket 0: [0.17, 0.12] → [0.12, 0.17]
Bucket 1: [0.39, 0.26, 0.21, 0.23]
- Insert 0.26: [0.26, 0.39, 0.21, 0.23]
- Insert 0.21: [0.21, 0.26, 0.39, 0.23]
- Insert 0.23: [0.21, 0.23, 0.26, 0.39]

Bucket 2: [] → []
Bucket 3: [0.78, 0.72, 0.68]
- Insert 0.72: [0.72, 0.78, 0.68]
- Insert 0.68: [0.68, 0.72, 0.78]

Bucket 4: [0.94] → [0.94]

**Step 4: Concatenate**
```
[0.12, 0.17] + [0.21, 0.23, 0.26, 0.39] + [] + [0.68, 0.72, 0.78] + [0.94]
= [0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94]
```

**Final sorted array:** [0.12, 0.17, 0.21, 0.23, 0.26, 0.39, 0.68, 0.72, 0.78, 0.94]

### Tim Sort

**Time**: $O(n \log n)$ worst, $O(n)$ best | **Space**: $O(n)$ | **Stable**: ✓ | **Adaptive**: ✓

#### How It Works
Tim Sort is a hybrid stable sorting algorithm derived from Merge Sort and Insertion Sort. It's the default sorting algorithm in Python and Java because it performs exceptionally well on real-world data, which often has some inherent order.

**Key concepts:**
1. **Runs**: Natural runs (already sorted subsequences) are identified in the data. Minimum run length is calculated based on array size (typically 32-64).
2. **Insertion Sort**: Short runs are extended to minimum length using insertion sort (very efficient for small arrays).
3. **Merge**: Runs are merged using an optimized merge sort strategy, using galloping mode when one run consistently "wins" comparisons.
4. **Stack invariants**: Maintains rules about run sizes on a stack to ensure balanced merges.

**Why it's adaptive:** It takes advantage of existing order in the data. Already sorted data requires O(n) time because no merging is needed.

**Why it's complex but effective:** The algorithm has many optimizations:
- Galloping mode: Binary search when merging highly imbalanced runs
- Minimum run size prevents worst-case behavior
- Merge only when beneficial based on stack invariants

**When to use**: You're using Python/Java (it's the default!), real-world data with partial order, need stability with good performance, or when you want the best general-purpose sort.

**When NOT to use**: You need a simple algorithm for educational purposes, working with uniformly random data (Quick Sort might be slightly faster), or space is extremely constrained.

```python
def tim_sort(arr):
    min_run = 32
    n = len(arr)

    # Sort individual runs using insertion sort
    for start in range(0, n, min_run):
        end = min(start + min_run, n)
        insertion_sort_range(arr, start, end)

    # Merge sorted runs
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size
            end = min(start + size * 2, n)
            if mid < end:
                merge_tim(arr, start, mid, end)
        size *= 2

    return arr

def insertion_sort_range(arr, left, right):
    for i in range(left + 1, right):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge_tim(arr, left, mid, right):
    left_part = arr[left:mid]
    right_part = arr[mid:right]

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1

# Note: This is a simplified version. Real Tim Sort has additional
# optimizations like galloping mode and sophisticated run merging strategies.
```

#### Step-by-Step Example
Sorting `[5, 21, 7, 23, 19, 3, 15, 28, 11, 17]` with min_run = 4:

**Phase 1: Create and sort initial runs**

**Run 1 [0:4]:** [5, 21, 7, 23]
- Insertion sort:
  - Insert 21: [5, 21, 7, 23]
  - Insert 7: [5, 7, 21, 23]
  - Insert 23: [5, 7, 21, 23]
- Result: [**5, 7, 21, 23**, 19, 3, 15, 28, 11, 17]

**Run 2 [4:8]:** [19, 3, 15, 28]
- Insertion sort:
  - Insert 3: [3, 19, 15, 28]
  - Insert 15: [3, 15, 19, 28]
  - Insert 28: [3, 15, 19, 28]
- Result: [5, 7, 21, 23, **3, 15, 19, 28**, 11, 17]

**Run 3 [8:10]:** [11, 17]
- Insertion sort:
  - Insert 17: [11, 17]
- Result: [5, 7, 21, 23, 3, 15, 19, 28, **11, 17**]

**Phase 2: Merge runs**

**Merge Run 1 and Run 2:**
```
Left:  [5, 7, 21, 23]    Right: [3, 15, 19, 28]

Compare 5 vs 3 → take 3:  [3]
Compare 5 vs 15 → take 5: [3, 5]
Compare 7 vs 15 → take 7: [3, 5, 7]
Compare 21 vs 15 → take 15: [3, 5, 7, 15]
Compare 21 vs 19 → take 19: [3, 5, 7, 15, 19]
Compare 21 vs 28 → take 21: [3, 5, 7, 15, 19, 21]
Compare 23 vs 28 → take 23: [3, 5, 7, 15, 19, 21, 23]
Take remaining 28: [3, 5, 7, 15, 19, 21, 23, 28]
```

Result: [**3, 5, 7, 15, 19, 21, 23, 28**, 11, 17]

**Merge combined run with Run 3:**
```
Left:  [3, 5, 7, 15, 19, 21, 23, 28]    Right: [11, 17]

Compare 3 vs 11 → take 3: [3]
Compare 5 vs 11 → take 5: [3, 5]
Compare 7 vs 11 → take 7: [3, 5, 7]
Compare 15 vs 11 → take 11: [3, 5, 7, 11]
Compare 15 vs 17 → take 15: [3, 5, 7, 11, 15]
Compare 19 vs 17 → take 17: [3, 5, 7, 11, 15, 17]
Take remaining: [3, 5, 7, 11, 15, 17, 19, 21, 23, 28]
```

**Final sorted array:** [3, 5, 7, 11, 15, 17, 19, 21, 23, 28]

**Why Tim Sort excels here:**
- If the input had natural runs (e.g., [5, 7, 21, 23, 3, 15, 19, 28, 11, 17]), it would detect them directly
- For partially sorted data like [3, 5, 7, 11, 19, 15, 17, 21, 23, 28], it would need fewer operations
- The algorithm adapts to the existing order in the data

## Comparison

| Algorithm | Best | Average | Worst | Space | Stable | Adaptive |
|-----------|------|---------|-------|-------|--------|----------|
| **Bubble** | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ✓ | ✓ |
| **Selection** | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ✗ | ✗ |
| **Insertion** | $O(n)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ✓ | ✓ |
| **Merge** | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | ✓ | ✗ |
| **Quick** | $O(n \log n)$ | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | ✗ | ✗ |
| **Heap** | $O(n \log n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | ✗ | ✗ |
| **Counting** | $O(n + k)$ | $O(n + k)$ | $O(n + k)$ | $O(k)$ | ✓ | N/A |
| **Radix** | $O(d(n + k))$ | $O(d(n + k))$ | $O(d(n + k))$ | $O(n + k)$ | ✓ | N/A |
| **Bucket** | $O(n + k)$ | $O(n + k)$ | $O(n^2)$ | $O(n + k)$ | ✓ | ✗ |
| **Tim** | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | ✓ | ✓ |

**Legend:**
- k = range of input values (for counting/radix) or number of buckets
- d = number of digits/characters
- N/A = concept doesn't apply to non-comparison algorithms

## When to Use Each Algorithm

### By Use Case

**Small arrays (n < 50):**
- **Insertion Sort**: Best choice, simple and efficient for small n

**Nearly sorted data:**
- **Insertion Sort**: O(n) on nearly sorted data
- **Tim Sort**: Designed for this, O(n) best case
- **Bubble Sort**: Works but slower than insertion sort

**General purpose (unknown data patterns):**
- **Tim Sort**: Best overall (Python/Java default)
- **Quick Sort**: Fastest average case, but needs randomization
- **Merge Sort**: Consistent performance, use if stability matters

**Guaranteed O(n log n), O(1) space:**
- **Heap Sort**: Only option meeting both constraints

**Need stability:**
- **Merge Sort**: Classic stable O(n log n)
- **Tim Sort**: Better in practice
- **Insertion Sort**: For small n
- Avoid: Quick Sort, Selection Sort, Heap Sort

**Integers in known small range:**
- **Counting Sort**: O(n + k) when k is small
- **Radix Sort**: When k is large but digits d is small

**Floating-point uniformly distributed:**
- **Bucket Sort**: Near-linear time with uniform distribution

**External sorting (data doesn't fit in memory):**
- **Merge Sort**: Natural fit for tape/disk access patterns
- **Tim Sort**: Modern choice with optimizations

**Parallel processing:**
- **Merge Sort**: Easy to parallelize
- **Quick Sort**: Can be parallelized
- **Bucket Sort**: Buckets can be sorted independently

**Embedded systems (memory constrained):**
- **Heap Sort**: O(1) space, guaranteed O(n log n)
- **Insertion Sort**: If n is small
- **Quick Sort**: With iterative implementation to avoid recursion stack

### By Data Characteristics

**Data Type:**
- **Integers (small range)**: Counting Sort → Radix Sort → Tim Sort
- **Integers (large range)**: Tim Sort → Quick Sort
- **Floats (uniform)**: Bucket Sort → Tim Sort
- **Floats (arbitrary)**: Tim Sort → Quick Sort
- **Strings (variable length)**: Tim Sort → Merge Sort → MSD Radix
- **Records (complex objects)**: Tim Sort → Merge Sort (if stable needed)

**Data Pattern:**
- **Random**: Quick Sort → Tim Sort → Heap Sort
- **Sorted**: Insertion Sort → Tim Sort → Bubble Sort
- **Reverse sorted**: Tim Sort → Merge Sort
- **Nearly sorted**: Insertion Sort → Tim Sort
- **Many duplicates**: Quick Sort (3-way partition) → Tim Sort

**Dataset Size:**
- **Tiny (< 10)**: Insertion Sort
- **Small (10-50)**: Insertion Sort → Quick Sort
- **Medium (50-10K)**: Quick Sort → Tim Sort
- **Large (10K-1M)**: Quick Sort → Tim Sort → Merge Sort
- **Very large (1M+)**: Tim Sort → Quick Sort → External Merge Sort

### When NOT to Use

**Don't use Bubble Sort when:** n > 10 or performance matters (use anything else)

**Don't use Selection Sort when:** You need stability or have a better option (almost always)

**Don't use Insertion Sort when:** n > 100 and data isn't nearly sorted (use O(n log n) algorithm)

**Don't use Merge Sort when:** Space is constrained (use Heap Sort) or you need fastest average case (use Quick Sort)

**Don't use Quick Sort when:** Worst-case guarantees matter (use Heap Sort), you need stability (use Merge/Tim Sort), or data is already sorted without randomization

**Don't use Heap Sort when:** Stability matters, you want fastest average case, or you're using a modern language (use Tim Sort)

**Don't use Counting Sort when:** Range k >> n (wastes space/time) or data isn't integers

**Don't use Radix Sort when:** Variable-length data with huge variations or floating-point without preprocessing

**Don't use Bucket Sort when:** Distribution is skewed (many elements in few buckets)

**Don't use Tim Sort when:** You're learning algorithms (too complex) or implementing in resource-constrained environments

## Practical Considerations

### Cache Performance
Modern CPUs have multiple levels of cache. Algorithms that access memory sequentially have better cache performance:
- **Best**: Insertion Sort, Heap Sort (works on array in-place)
- **Good**: Quick Sort (good locality within partitions)
- **Poor**: Merge Sort (jumps between arrays), Bucket Sort (scattered buckets)

### Real-World Performance
Big-O notation doesn't tell the whole story. Real performance depends on:
- **Constants**: Quick Sort has smaller constants than Merge Sort
- **Memory access patterns**: Cache-friendly algorithms are faster
- **Branch prediction**: Modern CPUs predict branches; predictable patterns help
- **Hardware**: Quick Sort benefits from good caching, Radix Sort benefits from parallel processing

### Hybrid Approaches
Many practical implementations use multiple algorithms:
- **Intro Sort** (C++ std::sort): Quick Sort → Heap Sort if recursion too deep
- **Tim Sort** (Python/Java): Merge Sort + Insertion Sort for runs
- **Quick Sort variants**: Use Insertion Sort for subarrays below threshold (often 10-20 elements)

## Python Built-in

Python uses Tim Sort for its built-in sorting, which is optimal for most cases:

```python
# In-place sort (modifies original list)
arr.sort()  # O(n log n) average and worst case

# Return new sorted list (original unchanged)
sorted_arr = sorted(arr)

# Reverse sort
arr.sort(reverse=True)
sorted_arr = sorted(arr, reverse=True)

# Custom comparator using key function
students = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 20}]
students.sort(key=lambda x: x['age'])  # Sort by age

# Multiple sort keys (age, then name)
students.sort(key=lambda x: (x['age'], x['name']))

# Sort with custom comparison (descending by absolute value)
numbers = [-4, 2, -8, 3, 1]
numbers.sort(key=abs, reverse=True)  # [8, -4, 3, 2, 1]

# Stability matters: sort by multiple fields
# Sort by grade (descending), then name (ascending) for ties
students.sort(key=lambda x: x['name'])  # Secondary sort first
students.sort(key=lambda x: x['grade'], reverse=True)  # Primary sort second
```

**Performance tips:**
- `list.sort()` is faster than `sorted(list)` if you don't need the original
- Key functions are called once per element (efficient)
- Use `functools.cmp_to_key` only if you really need comparison function
- For large datasets, consider if you actually need full sort (maybe `heapq.nlargest(k)` is enough)

## ELI10 (Explain Like I'm 10)

Imagine sorting a deck of cards. Different strategies work better in different situations:

### Simple Methods (Slow but Easy)

**Bubble Sort** - Like bubbles rising in soda:
- Compare two cards next to each other
- If the left one is bigger, swap them
- Keep doing this until no more swaps needed
- The biggest card "bubbles" to the end each time
- Very slow for lots of cards!

**Selection Sort** - Like picking the smallest card repeatedly:
- Look through all cards to find the smallest
- Put it in first position
- Look through remaining cards for the next smallest
- Put it in second position
- Continue until done

**Insertion Sort** - Like organizing cards in your hand:
- Take cards one by one
- Insert each new card in the right spot among sorted cards
- Shift other cards to make room
- Great when cards are almost sorted already!

### Fast Methods (Complex but Quick)

**Merge Sort** - Divide and conquer:
- Split the deck in half repeatedly until you have single cards
- Merge pairs of single cards into sorted pairs
- Merge sorted pairs into sorted groups of 4
- Keep merging until all cards are back together
- Like organizing two pre-sorted decks into one

**Quick Sort** - Pick a pivot and partition:
- Pick a "pivot" card (say, the 7)
- Put all smaller cards to the left, bigger cards to the right
- Now the 7 is in its final position!
- Do the same thing for the left and right piles
- Fast and commonly used

**Heap Sort** - Using a special tree structure:
- Imagine arranging cards in a tree where parents are bigger than children
- Keep taking the biggest card (always at top)
- Reorganize the tree after each take
- Guaranteed to be fast, uses no extra space

### Special Methods (Super Fast for Specific Cases)

**Counting Sort** - For numbered cards only:
- Make piles for each possible number (Ace, 2, 3, ...)
- Put each card in its number's pile
- Collect piles in order
- Lightning fast if you don't have too many different numbers!

**Bucket Sort** - For evenly spread numbers:
- Make several buckets for ranges (0-10, 11-20, 21-30...)
- Toss each card in the right bucket
- Sort each small bucket
- Collect buckets in order
- Great when cards are spread out evenly

**Tim Sort** (Python's choice) - Smart combination:
- Look for parts that are already sorted
- Use insertion sort for small groups
- Use merge sort to combine groups
- Adapts to partially sorted data
- This is what Python uses because real data is often partially sorted!

### Which Should You Use?

**For homework/learning:** Start with Bubble or Insertion Sort (easy to understand)

**For real programs:** Use your language's built-in sort (like Python's `sorted()`) - it's Tim Sort and super optimized!

**For fun/interviews:** Learn Quick Sort and Merge Sort - they're fast and commonly asked about

**For special cases:** If you're sorting millions of ages (numbers 0-120), Counting Sort is perfect!

Remember: Don't write your own sort unless you have a good reason. Built-in sorts are faster and bug-free!

## Further Resources

- [Sorting Algorithms Visualized](https://www.sortvisualizer.com/) - Watch algorithms in action
- [Big-O Cheatsheet](https://www.bigocheatsheet.com/) - Quick complexity reference
- [VisuAlgo](https://visualgo.net/en/sorting) - Interactive visualizations with step-by-step execution
- [Sorting Algorithm Animations](https://www.toptal.com/developers/sorting-algorithms) - Compare algorithm performance
- [Tim Sort Explained](https://github.com/python/cpython/blob/main/Objects/listsort.txt) - Original documentation by Tim Peters
