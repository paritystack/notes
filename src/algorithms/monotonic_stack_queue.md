# Monotonic Stack and Queue

## Overview
Monotonic stacks and queues are powerful data structures that maintain elements in sorted order (either increasing or decreasing). They're essential for solving a wide range of problems involving finding next/previous greater/smaller elements and range queries.

## Core Concepts

### Monotonic Stack
A stack that maintains elements in monotonic order:
- **Monotonic Increasing**: Elements increase from bottom to top
- **Monotonic Decreasing**: Elements decrease from bottom to top

**Key Property**: When adding an element, pop all elements that violate the monotonic property.

### Monotonic Queue
A queue (deque) that maintains monotonic order, typically used for sliding window problems.

**Time Complexity**: O(n) for processing n elements (each element pushed/popped once)
**Space Complexity**: O(n)

## 1. Next Greater Element

### Next Greater Element to the Right

```python
def next_greater_elements(arr):
    """
    Find next greater element for each element
    Returns array where result[i] = next greater element or -1
    """
    n = len(arr)
    result = [-1] * n
    stack = []  # Store indices

    for i in range(n):
        # Pop elements smaller than current
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]

        stack.append(i)

    return result

# Example
arr = [4, 5, 2, 10, 8]
print(next_greater_elements(arr))
# Output: [5, 10, 10, -1, -1]
```

### Next Greater Element in Circular Array

```python
def next_greater_circular(arr):
    """LeetCode 503: Next Greater Element II"""
    n = len(arr)
    result = [-1] * n
    stack = []

    # Iterate twice to simulate circular array
    for i in range(2 * n):
        idx = i % n

        while stack and arr[stack[-1]] < arr[idx]:
            result[stack.pop()] = arr[idx]

        # Only push in first iteration
        if i < n:
            stack.append(idx)

    return result

# Example
arr = [1, 2, 1]
print(next_greater_circular(arr))
# Output: [2, -1, 2]
```

### Next Greater Element with Mapping

```python
def next_greater_element(nums1, nums2):
    """
    LeetCode 496: Next Greater Element I
    Find next greater element for nums1 elements in nums2
    """
    # Build next greater map for nums2
    next_greater = {}
    stack = []

    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)

    # For remaining elements in stack
    while stack:
        next_greater[stack.pop()] = -1

    # Build result for nums1
    return [next_greater[num] for num in nums1]
```

## 2. Next Smaller Element

### Next Smaller Element to the Right

```python
def next_smaller_elements(arr):
    """Find next smaller element for each element"""
    n = len(arr)
    result = [-1] * n
    stack = []  # Monotonic increasing stack

    for i in range(n):
        # Pop elements greater than current
        while stack and arr[stack[-1]] > arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]

        stack.append(i)

    return result

# Example
arr = [4, 2, 1, 5, 3]
print(next_smaller_elements(arr))
# Output: [2, 1, -1, 3, -1]
```

### Previous Smaller Element

```python
def previous_smaller_elements(arr):
    """Find previous smaller element for each element"""
    n = len(arr)
    result = [-1] * n
    stack = []  # Store indices

    for i in range(n):
        # Maintain monotonic increasing stack
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()

        if stack:
            result[i] = arr[stack[-1]]

        stack.append(i)

    return result

# Example
arr = [4, 5, 2, 10, 8]
print(previous_smaller_elements(arr))
# Output: [-1, 4, -1, 2, 2]
```

## 3. Histogram Problems

### Largest Rectangle in Histogram

Classic problem using monotonic stack.

```python
def largest_rectangle_area(heights):
    """
    LeetCode 84: Largest Rectangle in Histogram
    Time: O(n), Space: O(n)
    """
    stack = []  # Store indices
    max_area = 0
    heights = heights + [0]  # Add sentinel to clear stack

    for i, h in enumerate(heights):
        # Pop heights greater than current
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            height = heights[height_idx]

            # Width = current index - previous index - 1
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area

# Example
heights = [2, 1, 5, 6, 2, 3]
print(largest_rectangle_area(heights))
# Output: 10 (rectangle of height 5-6)
```

**Explanation**:
- For each bar, find left and right boundaries where height >= current height
- Use monotonic increasing stack to efficiently find these boundaries
- Area = height × (right_boundary - left_boundary - 1)

### Maximal Rectangle in Binary Matrix

```python
def maximal_rectangle(matrix):
    """
    LeetCode 85: Maximal Rectangle
    Convert to histogram problem for each row
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # Update heights
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # Find largest rectangle in current histogram
        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area

# Example
matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]
print(maximal_rectangle(matrix))
# Output: 6
```

### Trapping Rain Water

```python
def trap_rain_water(height):
    """
    LeetCode 42: Trapping Rain Water
    Using monotonic decreasing stack
    """
    stack = []
    water = 0

    for i, h in enumerate(height):
        # Calculate water trapped between bars
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()

            if not stack:
                break

            # Height of trapped water
            bounded_height = min(height[stack[-1]], h) - height[bottom]
            # Width between bars
            width = i - stack[-1] - 1
            water += bounded_height * width

        stack.append(i)

    return water

# Example
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap_rain_water(height))
# Output: 6
```

## 4. Sliding Window Maximum

Using monotonic decreasing deque to efficiently find maximum in each window.

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    LeetCode 239: Sliding Window Maximum
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Store indices
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove elements smaller than current (maintain decreasing order)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Add to result once we have a complete window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Example
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(max_sliding_window(nums, k))
# Output: [3, 3, 5, 5, 6, 7]
```

### Sliding Window Minimum

```python
def min_sliding_window(nums, k):
    """Find minimum in each sliding window"""
    if not nums or k == 0:
        return []

    dq = deque()  # Monotonic increasing deque
    result = []

    for i, num in enumerate(nums):
        # Remove out of window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain increasing order (opposite of maximum)
        while dq and nums[dq[-1]] > num:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### Constrained Subsequence Sum

```python
def constrained_subset_sum(nums, k):
    """
    LeetCode 1425: Find max sum of subsequence
    where any two adjacent elements are at most k apart
    """
    n = len(nums)
    dp = nums[:]  # dp[i] = max sum ending at i
    dq = deque()  # Monotonic decreasing deque of dp values

    for i in range(n):
        # Remove elements outside window
        while dq and dq[0] < i - k:
            dq.popleft()

        # Update dp[i]
        if dq:
            dp[i] = max(dp[i], dp[dq[0]] + nums[i])

        # Maintain monotonic decreasing deque
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()

        dq.append(i)

    return max(dp)
```

## 5. Stock Span Patterns

### Stock Span Problem

```python
class StockSpanner:
    """
    LeetCode 901: Online Stock Span
    Calculate span = consecutive days with price <= today's price
    """
    def __init__(self):
        self.stack = []  # (price, span)

    def next(self, price):
        span = 1

        # Pop smaller prices and accumulate their spans
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]

        self.stack.append((price, span))
        return span

# Example
spanner = StockSpanner()
print(spanner.next(100))  # 1
print(spanner.next(80))   # 1
print(spanner.next(60))   # 1
print(spanner.next(70))   # 2
print(spanner.next(60))   # 1
print(spanner.next(75))   # 4
print(spanner.next(85))   # 6
```

### Sum of Subarray Minimums

```python
def sum_subarray_mins(arr):
    """
    LeetCode 907: Sum of Subarray Minimums
    For each element, find how many subarrays have it as minimum
    """
    MOD = 10**9 + 7
    n = len(arr)

    # Find previous less element (PLE) and next less element (NLE)
    left = [0] * n   # Distance to PLE
    right = [0] * n  # Distance to NLE

    # Calculate left distances (PLE)
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()

        left[i] = i - stack[-1] if stack else i + 1
        stack.append(i)

    # Calculate right distances (NLE)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()

        right[i] = stack[-1] - i if stack else n - i
        stack.append(i)

    # Calculate sum
    result = 0
    for i in range(n):
        # Number of subarrays with arr[i] as minimum
        result += arr[i] * left[i] * right[i]
        result %= MOD

    return result

# Example
arr = [3, 1, 2, 4]
print(sum_subarray_mins(arr))
# Output: 17
```

### Remove K Digits

```python
def remove_k_digits(num, k):
    """
    LeetCode 402: Remove K Digits
    Remove k digits to make smallest possible number
    """
    stack = []

    for digit in num:
        # Remove larger digits while we can
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1

        stack.append(digit)

    # Remove remaining k digits from end
    if k > 0:
        stack = stack[:-k]

    # Build result, removing leading zeros
    result = ''.join(stack).lstrip('0')

    return result if result else '0'

# Example
print(remove_k_digits("1432219", 3))
# Output: "1219"
```

## 6. Advanced Patterns

### Daily Temperatures

```python
def daily_temperatures(temperatures):
    """
    LeetCode 739: Daily Temperatures
    Find how many days until warmer temperature
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx

        stack.append(i)

    return result

# Example
temps = [73,74,75,71,69,72,76,73]
print(daily_temperatures(temps))
# Output: [1,1,4,2,1,1,0,0]
```

### Car Fleet

```python
def car_fleet(target, position, speed):
    """
    LeetCode 853: Car Fleet
    Count number of car fleets reaching target
    """
    cars = sorted(zip(position, speed), reverse=True)
    stack = []

    for pos, spd in cars:
        # Time to reach target
        time = (target - pos) / spd

        # If this car takes longer, it forms new fleet
        if not stack or time > stack[-1]:
            stack.append(time)
        # Otherwise it catches up to previous fleet

    return len(stack)
```

### Minimum Cost Tree From Leaf Values

```python
def mct_from_leaf_values(arr):
    """
    LeetCode 1130: Minimum Cost Tree From Leaf Values
    Use monotonic decreasing stack
    """
    stack = [float('inf')]
    result = 0

    for num in arr:
        while stack[-1] <= num:
            mid = stack.pop()
            # Cost = mid * min(left, right)
            result += mid * min(stack[-1], num)

        stack.append(num)

    # Process remaining elements
    while len(stack) > 2:
        result += stack.pop() * stack[-1]

    return result
```

## Template Patterns

### Template 1: Next Greater/Smaller

```python
def next_element_template(arr, compare_func, default=-1):
    """
    Generic template for next element problems
    compare_func: lambda to determine when to pop
    """
    n = len(arr)
    result = [default] * n
    stack = []

    for i in range(n):
        while stack and compare_func(arr[stack[-1]], arr[i]):
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)

    return result

# Next Greater
next_greater = lambda a, b: a < b
result = next_element_template(arr, next_greater)

# Next Smaller
next_smaller = lambda a, b: a > b
result = next_element_template(arr, next_smaller)
```

### Template 2: Sliding Window with Deque

```python
def sliding_window_template(nums, k, operation='max'):
    """
    Generic sliding window with monotonic deque
    operation: 'max' or 'min'
    """
    from collections import deque

    dq = deque()
    result = []
    compare = (lambda a, b: a < b) if operation == 'max' else (lambda a, b: a > b)

    for i, num in enumerate(nums):
        # Remove out of window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain monotonic property
        while dq and compare(nums[dq[-1]], num):
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

## Common Patterns Recognition

1. **Find Next/Previous Greater/Smaller** → Monotonic Stack
2. **Sliding Window Min/Max** → Monotonic Deque
3. **Rectangle in Histogram** → Monotonic Increasing Stack
4. **Calculate Spans/Ranges** → Monotonic Stack with counts
5. **Remove Elements to Optimize** → Monotonic Stack

## Problem-Solving Tips

1. **Stack stores indices**, not values (usually)
2. **Monotonic increasing** for next smaller element
3. **Monotonic decreasing** for next greater element
4. **Use deque** for sliding window problems
5. **Add sentinel** values to simplify edge cases
6. **Each element pushed/popped once** → O(n) time

## Practice Problems

### Easy
- LeetCode 496: Next Greater Element I
- LeetCode 739: Daily Temperatures
- LeetCode 1475: Final Prices With a Special Discount

### Medium
- LeetCode 503: Next Greater Element II
- LeetCode 901: Online Stock Span
- LeetCode 239: Sliding Window Maximum
- LeetCode 84: Largest Rectangle in Histogram
- LeetCode 42: Trapping Rain Water

### Hard
- LeetCode 85: Maximal Rectangle
- LeetCode 907: Sum of Subarray Minimums
- LeetCode 1425: Constrained Subsequence Sum
- LeetCode 1425: Shortest Subarray with Sum at Least K

## Resources

- [Monotonic Stack Explanation](https://leetcode.com/tag/monotonic-stack/)
- [CP-Algorithms: Stack](https://cp-algorithms.com/)
