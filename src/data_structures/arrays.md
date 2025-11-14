# Arrays

## Overview

An array is a fundamental data structure that stores elements of the same type in contiguous memory locations. Arrays provide fast, constant-time access to elements using an index, making them one of the most commonly used data structures in programming.

## Key Concepts

### Characteristics

- **Fixed Size**: Most arrays have a fixed size determined at creation
- **Contiguous Memory**: Elements stored sequentially in memory
- **Index-Based**: Access elements using zero-based indexing
- **Homogeneous**: All elements must be of the same type
- **Fast Access**: $O(1)$ time complexity for accessing any element

### Memory Layout

```
Index:     0    1    2    3    4
Array:   | 10 | 20 | 30 | 40 | 50 |
Address: 1000 1004 1008 1012 1016  (for 4-byte integers)
```

## Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| **Access** | $O(1)$ |
| **Search** | $O(n)$ |
| **Insert (at end)** | $O(1)$ amortized* |
| **Insert (at position)** | $O(n)$ |
| **Delete (at end)** | $O(1)$ |
| **Delete (at position)** | $O(n)$ |

*For dynamic arrays like Python lists or C++ vectors

## Code Examples

### Python

```python
# Creating arrays
arr = [1, 2, 3, 4, 5]
arr_zeros = [0] * 10  # [0, 0, 0, ..., 0] (10 elements)

# Accessing elements
first = arr[0]        # 1
last = arr[-1]        # 5 (negative indexing)

# Modifying elements
arr[2] = 100          # [1, 2, 100, 4, 5]

# Slicing
sub = arr[1:4]        # [2, 100, 4]
reversed_arr = arr[::-1]  # [5, 4, 100, 2, 1]

# Common operations
arr.append(6)         # Add to end: [1, 2, 100, 4, 5, 6]
arr.insert(2, 99)     # Insert at index: [1, 2, 99, 100, 4, 5, 6]
arr.pop()             # Remove last: returns 6
arr.remove(99)        # Remove first occurrence of 99
length = len(arr)     # Get length

# Iteration
for element in arr:
    print(element)

for index, element in enumerate(arr):
    print(f"Index {index}: {element}")

# List comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, ..., 81]
evens = [x for x in arr if x % 2 == 0]  # Filter even numbers

# 2D Arrays (Matrix)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
element = matrix[1][2]  # Access: 6
```

### JavaScript

```javascript
// Creating arrays
let arr = [1, 2, 3, 4, 5];
let arr2 = new Array(10);  // Array with 10 undefined elements
let arr3 = Array.from({length: 5}, (_, i) => i);  // [0, 1, 2, 3, 4]

// Accessing and modifying
arr[0] = 100;
let last = arr[arr.length - 1];  // 5

// Common methods
arr.push(6);              // Add to end
arr.pop();                // Remove from end
arr.unshift(0);           // Add to beginning
arr.shift();              // Remove from beginning
arr.splice(2, 1, 99);     // Remove 1 element at index 2, insert 99

// Iteration
arr.forEach((element, index) => {
    console.log(index, element);
});

// Map, filter, reduce
let doubled = arr.map(x => x * 2);
let evens = arr.filter(x => x % 2 === 0);
let sum = arr.reduce((acc, x) => acc + x, 0);

// Find elements
let found = arr.find(x => x > 3);     // First element > 3
let index = arr.findIndex(x => x > 3);  // Index of first element > 3
let includes = arr.includes(3);       // true if 3 exists

// Sorting
arr.sort((a, b) => a - b);  // Ascending
arr.sort((a, b) => b - a);  // Descending

// Spread operator
let combined = [...arr, ...arr2];
let copy = [...arr];
```

### C++

```cpp
#include <iostream>
#include <vector>
#include <array>
using namespace std;

int main() {
    // Static array
    int arr[5] = {1, 2, 3, 4, 5};
    int size = sizeof(arr) / sizeof(arr[0]);  // 5

    // Access and modify
    arr[0] = 100;
    int last = arr[size - 1];

    // std::array (fixed size, safer)
    array<int, 5> std_arr = {1, 2, 3, 4, 5};
    std_arr[0] = 100;
    int sz = std_arr.size();

    // std::vector (dynamic array)
    vector<int> vec = {1, 2, 3, 4, 5};
    vec.push_back(6);         // Add to end
    vec.pop_back();           // Remove from end
    vec.insert(vec.begin() + 2, 99);  // Insert at index 2
    vec.erase(vec.begin() + 2);       // Remove at index 2

    // Iteration
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << " ";
    }

    // Range-based for loop
    for (int x : vec) {
        cout << x << " ";
    }

    // 2D vector
    vector<vector<int>> matrix(3, vector<int>(4, 0));  // 3x4 matrix of zeros
    matrix[1][2] = 99;

    return 0;
}
```

### Java

```java
import java.util.ArrayList;
import java.util.Arrays;

public class ArrayExamples {
    public static void main(String[] args) {
        // Static array
        int[] arr = {1, 2, 3, 4, 5};
        int[] arr2 = new int[10];  // 10 elements, initialized to 0

        // Access and modify
        arr[0] = 100;
        int length = arr.length;

        // ArrayList (dynamic array)
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(2, 99);  // Insert at index 2
        list.remove(2);   // Remove at index 2
        int element = list.get(1);  // Access index 1
        list.set(1, 100);           // Modify index 1

        // Iteration
        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i));
        }

        for (int x : list) {
            System.out.println(x);
        }

        // Useful methods
        boolean contains = list.contains(3);
        int idx = list.indexOf(3);
        list.sort((a, b) -> a - b);  // Sort

        // Arrays utility
        int[] arr3 = {3, 1, 4, 1, 5};
        Arrays.sort(arr3);
        int index = Arrays.binarySearch(arr3, 4);  // Binary search (sorted array)
    }
}
```

## Common Algorithms

### Linear Search

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1  # Not found

# Time: $O(n)$, Space: $O(1)$
```

### Binary Search (Sorted Array)

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

# Time: $O(\log n)$, Space: $O(1)$
```

### Two Pointers Technique

```python
def two_sum_sorted(arr, target):
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

    return []  # Not found
```

### Sliding Window

```python
def max_sum_subarray(arr, k):
    """Maximum sum of k consecutive elements"""
    if len(arr) < k:
        return None

    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Time: $O(n)$, Space: $O(1)$
```

### Kadane's Algorithm (Maximum Subarray)

```python
def max_subarray_sum(arr):
    """Find maximum sum of any contiguous subarray"""
    max_so_far = arr[0]
    max_ending_here = arr[0]

    for i in range(1, len(arr)):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

# Time: $O(n)$, Space: $O(1)$
# Example: [-2, 1, -3, 4, -1, 2, 1, -5, 4] -> 6 (subarray [4, -1, 2, 1])
```

## Common Problems

### Reverse an Array

```python
def reverse(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr
```

### Rotate Array

```python
def rotate_right(arr, k):
    """Rotate array to the right by k positions"""
    n = len(arr)
    k = k % n  # Handle k > n

    # Reverse entire array
    arr.reverse()
    # Reverse first k elements
    arr[:k] = reversed(arr[:k])
    # Reverse remaining elements
    arr[k:] = reversed(arr[k:])

    return arr

# Example: [1, 2, 3, 4, 5], k=2 -> [4, 5, 1, 2, 3]
```

### Remove Duplicates (Sorted Array)

```python
def remove_duplicates(arr):
    """Remove duplicates in-place, return new length"""
    if not arr:
        return 0

    write_index = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            arr[write_index] = arr[i]
            write_index += 1

    return write_index
```

## Best Practices

### 1. Bounds Checking

```python
# Bad
if arr[i] > 0:  # May cause IndexError

# Good
if i < len(arr) and arr[i] > 0:
```

### 2. Avoid Modifying While Iterating

```python
# Bad
for x in arr:
    if x % 2 == 0:
        arr.remove(x)  # Causes skipping

# Good
arr = [x for x in arr if x % 2 != 0]  # Create new list
```

### 3. Use Appropriate Data Structure

- Need frequent insertions/deletions? Consider linked list
- Need fast lookups? Consider hash table
- Working with numerical data? Use NumPy arrays

## ELI10

Think of an array like a row of mailboxes in an apartment building:

- Each mailbox has a number (index): 0, 1, 2, 3, ...
- Each mailbox can hold one item (element)
- To get mail from mailbox #3, you go directly to it - very fast!
- All mailboxes are right next to each other in a line
- You know exactly how many mailboxes there are

**The cool part**: You can instantly go to any mailbox by its number, no need to check all the other mailboxes first!

**The tricky part**: If you want to add a new mailbox in the middle, you have to shift all the mailboxes after it to make room - that takes time!

## Further Resources

- [Arrays in Python - Official Docs](https://docs.python.org/3/tutorial/datastructures.html)
- [MDN JavaScript Arrays](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array)
- [C++ Vector Documentation](https://en.cppreference.com/w/cpp/container/vector)
- [LeetCode Array Problems](https://leetcode.com/tag/array/)
