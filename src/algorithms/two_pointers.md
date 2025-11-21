# Two Pointers

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use Two Pointers](#when-to-use-two-pointers)
  - [Types of Two Pointer Patterns](#types-of-two-pointer-patterns)
- [Pattern 1: Opposite Direction](#pattern-1-opposite-direction)
- [Pattern 2: Same Direction (Fast and Slow)](#pattern-2-same-direction-fast-and-slow)
- [Pattern 3: Sliding Window (Variable Size)](#pattern-3-sliding-window-variable-size)
- [Pattern 4: Partition/Sort Related](#pattern-4-partitionsort-related)
- [Pattern 5: Merge Two Sorted Arrays](#pattern-5-merge-two-sorted-arrays)
- [Advanced Patterns](#advanced-patterns)
- [Common Interview Problems](#common-interview-problems)
- [Complexity Analysis](#complexity-analysis)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

The **Two Pointers** technique is a fundamental algorithmic pattern that uses two pointers to traverse a data structure (typically an array or linked list) efficiently. Instead of using nested loops (O(n²)), two pointers can often solve problems in O(n) time.

**Key characteristics:**
- Two indices/references that move through the data
- Can move in same or opposite directions
- Reduces time complexity from O(n²) to O(n)
- Often used with sorted arrays
- No extra space needed (O(1) space)

## ELI10 Explanation

Imagine you have a long line of toy blocks arranged in order.

**Opposite Direction Pointers:**
You stand at one end and your friend stands at the other end. You both walk toward each other, looking for two blocks that add up to a specific number. When you find them, you're done!

**Same Direction Pointers (Fast/Slow):**
You and your friend both start at the beginning. Your friend walks faster (2 blocks at a time) while you walk normally (1 block at a time). If there's a loop in the blocks, your faster friend will eventually catch up to you!

This is way faster than checking every possible pair of blocks (which would take forever)!

## Core Concepts

### When to Use Two Pointers

Two pointers work best when:

1. **Array is sorted** (or can be sorted)
2. Looking for **pairs** or **triplets** with certain properties
3. **Removing duplicates** in-place
4. **Detecting cycles** in linked lists
5. **Merging** sorted arrays
6. **Partitioning** arrays based on conditions
7. Problem can be solved by **comparing elements from both ends**

### Types of Two Pointer Patterns

```python
"""
1. OPPOSITE DIRECTION
   left -->        <-- right
   [1, 2, 3, 4, 5, 6, 7, 8]

2. SAME DIRECTION (FAST/SLOW)
   slow -->    fast -->
   [1, 2, 3, 4, 5, 6, 7, 8]

3. SLIDING WINDOW (covered in sliding_window.md)
   [start ... end]
   [1, 2, 3, 4, 5, 6, 7, 8]

4. MULTIPLE ARRAYS
   i -->
   [1, 3, 5, 7]
   j -->
   [2, 4, 6, 8]
"""
```

## Pattern 1: Opposite Direction

Pointers start at opposite ends and move toward each other.

### Problem: Two Sum (Sorted Array)

```python
def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """
    Find two numbers that sum to target in sorted array.

    Strategy:
    - Left pointer at start, right at end
    - If sum too small, move left right (increase sum)
    - If sum too large, move right left (decrease sum)
    - If equal, found answer

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return []  # No solution

# Example with trace
def two_sum_traced(nums: list[int], target: int) -> list[int]:
    """Two sum with detailed trace."""
    left, right = 0, len(nums) - 1
    print(f"Array: {nums}, Target: {target}\n")

    step = 1
    while left < right:
        current_sum = nums[left] + nums[right]
        print(f"Step {step}: left={left}, right={right}")
        print(f"  nums[{left}] + nums[{right}] = {nums[left]} + {nums[right]} = {current_sum}")

        if current_sum == target:
            print(f"  ✓ Found! Indices: [{left}, {right}]")
            return [left, right]
        elif current_sum < target:
            print(f"  Sum too small, move left pointer right")
            left += 1
        else:
            print(f"  Sum too large, move right pointer left")
            right -= 1

        step += 1
        print()

    return []

# Test
print(two_sum_sorted([1, 2, 3, 4, 6], 6))  # [1, 3] -> 2 + 4 = 6
two_sum_traced([1, 2, 3, 4, 6], 6)
```

### Problem: Container With Most Water

```python
def max_area(height: list[int]) -> int:
    """
    LeetCode 11: Find container that holds most water.

    Strategy:
    - Start with widest container (left=0, right=n-1)
    - Move pointer at shorter height inward (might find taller)
    - Track maximum area seen

    Intuition: Moving the taller line inward can't improve area,
    so always move the shorter one.

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height

        max_water = max(max_water, current_area)

        # Move pointer at shorter height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

# Example
print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # 49
# Explanation: Lines at index 1 (height=8) and 8 (height=7)
# Area = min(8,7) * (8-1) = 7 * 7 = 49
```

### Problem: Valid Palindrome

```python
def is_palindrome(s: str) -> bool:
    """
    LeetCode 125: Check if string is palindrome (ignore non-alphanumeric).

    Strategy: Two pointers from both ends moving inward.

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

# Examples
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

### Problem: Reverse String

```python
def reverse_string(s: list[str]) -> None:
    """
    LeetCode 344: Reverse string in-place.

    Strategy: Swap characters from both ends moving inward.

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(s) - 1

    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

# Example
chars = ['h', 'e', 'l', 'l', 'o']
reverse_string(chars)
print(chars)  # ['o', 'l', 'l', 'e', 'h']
```

## Pattern 2: Same Direction (Fast and Slow)

Both pointers start at the same end and move in the same direction at different speeds.

### Problem: Remove Duplicates from Sorted Array

```python
def remove_duplicates(nums: list[int]) -> int:
    """
    LeetCode 26: Remove duplicates in-place from sorted array.
    Return length of array with unique elements.

    Strategy:
    - Slow pointer: position for next unique element
    - Fast pointer: scan through array
    - When fast finds new element, copy to slow position

    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0

    slow = 0  # Next position for unique element

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1  # Length of unique elements

# Example with trace
def remove_duplicates_traced(nums: list[int]) -> int:
    """Remove duplicates with visualization."""
    if not nums:
        return 0

    print(f"Original: {nums}\n")
    slow = 0

    for fast in range(1, len(nums)):
        print(f"slow={slow}, fast={fast}")
        print(f"  nums[slow]={nums[slow]}, nums[fast]={nums[fast]}")

        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
            print(f"  Different! Copy nums[{fast}] to position {slow}")
            print(f"  Array now: {nums[:slow+1]} + ...")
        else:
            print(f"  Same, skip")

        print()

    print(f"Final: {nums[:slow+1]}")
    return slow + 1

# Test
nums = [1, 1, 2, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(nums)
print(f"Unique elements: {nums[:length]}")  # [1, 2, 3, 4, 5]

nums2 = [1, 1, 2, 2, 2, 3, 4, 4, 5]
remove_duplicates_traced(nums2)
```

### Problem: Move Zeroes

```python
def move_zeroes(nums: list[int]) -> None:
    """
    LeetCode 283: Move all zeros to end while maintaining relative order.

    Strategy:
    - Slow pointer: position for next non-zero
    - Fast pointer: find non-zeros
    - Swap when fast finds non-zero

    Time: O(n), Space: O(1)
    """
    slow = 0  # Next position for non-zero

    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

# Example
nums = [0, 1, 0, 3, 12]
move_zeroes(nums)
print(nums)  # [1, 3, 12, 0, 0]
```

### Problem: Linked List Cycle (Floyd's Algorithm)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode) -> bool:
    """
    LeetCode 141: Detect cycle in linked list.

    Floyd's Cycle Detection (Tortoise and Hare):
    - Slow moves 1 step at a time
    - Fast moves 2 steps at a time
    - If there's a cycle, they'll meet
    - If no cycle, fast reaches end

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False  # Reached end, no cycle

        slow = slow.next
        fast = fast.next.next

    return True  # Pointers met, cycle exists

def detect_cycle_start(head: ListNode) -> ListNode:
    """
    LeetCode 142: Find where cycle begins.

    After slow and fast meet:
    1. Move one pointer to head
    2. Move both one step at a time
    3. Where they meet is the cycle start

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None

    # Phase 1: Detect cycle
    slow = fast = head
    has_cycle = False

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            has_cycle = True
            break

    if not has_cycle:
        return None

    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow  # Cycle start node
```

### Problem: Middle of Linked List

```python
def find_middle(head: ListNode) -> ListNode:
    """
    LeetCode 876: Find middle node of linked list.

    Strategy:
    - Slow moves 1 step, fast moves 2 steps
    - When fast reaches end, slow is at middle

    Time: O(n), Space: O(1)
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # Middle node (or second middle if even length)
```

### Problem: Remove Nth Node From End

```python
def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    LeetCode 19: Remove nth node from end of linked list.

    Strategy:
    - Fast pointer moves n steps ahead
    - Then both move together
    - When fast reaches end, slow is before nth from end

    Time: O(n), Space: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    slow = fast = dummy

    # Move fast n steps ahead
    for _ in range(n):
        fast = fast.next

    # Move both until fast reaches end
    while fast.next:
        slow = slow.next
        fast = fast.next

    # Remove node
    slow.next = slow.next.next

    return dummy.next
```

## Pattern 3: Sliding Window (Variable Size)

See `sliding_window.md` for detailed coverage. Brief overview here:

```python
def length_of_longest_substring(s: str) -> int:
    """
    LeetCode 3: Longest substring without repeating characters.

    Two pointers with hashmap for variable-size window.

    Time: O(n), Space: O(min(n, m)) where m = charset size
    """
    char_index = {}
    max_length = 0
    start = 0

    for end in range(len(s)):
        # If character seen before, move start
        if s[end] in char_index and char_index[s[end]] >= start:
            start = char_index[s[end]] + 1

        char_index[s[end]] = end
        max_length = max(max_length, end - start + 1)

    return max_length

print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
```

## Pattern 4: Partition/Sort Related

### Problem: Dutch National Flag (Sort Colors)

```python
def sort_colors(nums: list[int]) -> None:
    """
    LeetCode 75: Sort array of 0s, 1s, and 2s in-place.

    Dutch National Flag algorithm (3-way partition):
    - Low pointer: boundary for 0s
    - Mid pointer: current element
    - High pointer: boundary for 2s

    Invariant:
    - [0, low): all 0s
    - [low, mid): all 1s
    - [mid, high]: unexplored
    - (high, n): all 2s

    Time: O(n), Space: O(1)
    """
    low, mid, high = 0, 0, len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid! Need to check swapped element

# Example
colors = [2, 0, 2, 1, 1, 0]
sort_colors(colors)
print(colors)  # [0, 0, 1, 1, 2, 2]
```

### Problem: Partition Array

```python
def partition_array(nums: list[int], pivot: int) -> int:
    """
    Partition array around pivot (like QuickSort partition).
    Return index where pivot should be.

    Strategy:
    - Left: boundary for elements < pivot
    - Right: scan through array
    - Swap when find element < pivot

    Time: O(n), Space: O(1)
    """
    left = 0

    for right in range(len(nums)):
        if nums[right] < pivot:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

    return left  # All elements [0, left) are < pivot

# Example
nums = [3, 7, 1, 9, 5, 2, 8]
pivot_index = partition_array(nums, 5)
print(nums)  # Elements < 5 on left, >= 5 on right
print(f"Pivot should be at index: {pivot_index}")
```

## Pattern 5: Merge Two Sorted Arrays

### Problem: Merge Sorted Array

```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    LeetCode 88: Merge nums2 into nums1 (nums1 has size m+n).

    Strategy: Merge from the END to avoid overwriting!
    - p1: last element of nums1
    - p2: last element of nums2
    - p: last position in merged array

    Time: O(m + n), Space: O(1)
    """
    p1, p2 = m - 1, n - 1
    p = m + n - 1

    # Merge from end
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

# Example
nums1 = [1, 2, 3, 0, 0, 0]
nums2 = [2, 5, 6]
merge(nums1, 3, nums2, 3)
print(nums1)  # [1, 2, 2, 3, 5, 6]
```

### Problem: Intersection of Two Arrays

```python
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    LeetCode 350: Find intersection (with duplicates).

    If arrays are sorted (or we sort them):
    - Use two pointers to find common elements

    Time: O(n log n + m log m), Space: O(1) excluding output
    """
    nums1.sort()
    nums2.sort()

    i, j = 0, 0
    result = []

    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            result.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return result

print(intersect([1, 2, 2, 1], [2, 2]))  # [2, 2]
print(intersect([4, 9, 5], [9, 4, 9, 8, 4]))  # [4, 9]
```

## Advanced Patterns

### Problem: 3Sum

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    LeetCode 15: Find all triplets that sum to zero.

    Strategy:
    1. Sort array
    2. Fix one element (i)
    3. Use two pointers for remaining two elements
    4. Skip duplicates to avoid duplicate triplets

    Time: O(n²), Space: O(1) excluding output
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Two pointers for remaining elements
        left, right = i + 1, len(nums) - 1
        target = -nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1

            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result

# Example
print(three_sum([-1, 0, 1, 2, -1, -4]))
# [[-1, -1, 2], [-1, 0, 1]]
```

### Problem: 4Sum

```python
def four_sum(nums: list[int], target: int) -> list[list[int]]:
    """
    LeetCode 18: Find all quadruplets that sum to target.

    Strategy: Extend 3Sum with another loop.

    Time: O(n³), Space: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        for j in range(i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            # Two pointers for remaining elements
            left, right = j + 1, n - 1

            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]

                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

    return result

print(four_sum([1, 0, -1, 0, -2, 2], 0))
# [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
```

### Problem: Trapping Rain Water

```python
def trap(height: list[int]) -> int:
    """
    LeetCode 42: Calculate trapped rainwater.

    Strategy: Two pointers tracking max height from both sides.
    - Water at position i = min(max_left, max_right) - height[i]
    - Move pointer with smaller max (that side determines water level)

    Time: O(n), Space: O(1)
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0

    while left < right:
        if left_max < right_max:
            # Left side determines water level
            left += 1
            left_max = max(left_max, height[left])
            water += max(0, left_max - height[left])
        else:
            # Right side determines water level
            right -= 1
            right_max = max(right_max, height[right])
            water += max(0, right_max - height[right])

    return water

print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # 6
```

### Problem: Remove Duplicates II

```python
def remove_duplicates_ii(nums: list[int]) -> int:
    """
    LeetCode 80: Allow each element to appear at most twice.

    Strategy: Keep slow pointer at next write position.
    Write if element appears < 2 times (check nums[slow-2]).

    Time: O(n), Space: O(1)
    """
    if len(nums) <= 2:
        return len(nums)

    slow = 2  # First two elements can stay

    for fast in range(2, len(nums)):
        # If current != element 2 positions back, it's ok to add
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1

    return slow

nums = [1, 1, 1, 2, 2, 3]
length = remove_duplicates_ii(nums)
print(nums[:length])  # [1, 1, 2, 2, 3]
```

## Common Interview Problems

### Problem: Squares of Sorted Array

```python
def sorted_squares(nums: list[int]) -> list[int]:
    """
    LeetCode 977: Square elements of sorted array, return sorted.

    Input can have negative numbers!
    Strategy: Two pointers from both ends (largest squares are at ends).

    Time: O(n), Space: O(n)
    """
    n = len(nums)
    result = [0] * n
    left, right = 0, n - 1
    pos = n - 1  # Fill result from end

    while left <= right:
        left_square = nums[left] ** 2
        right_square = nums[right] ** 2

        if left_square > right_square:
            result[pos] = left_square
            left += 1
        else:
            result[pos] = right_square
            right -= 1

        pos -= 1

    return result

print(sorted_squares([-4, -1, 0, 3, 10]))  # [0, 1, 9, 16, 100]
```

### Problem: Backspace String Compare

```python
def backspace_compare(s: str, t: str) -> bool:
    """
    LeetCode 844: Compare strings with backspaces (#).

    Strategy: Process from end using two pointers.
    Skip characters based on backspace count.

    Time: O(n + m), Space: O(1)
    """
    def next_valid_char(string: str, index: int) -> int:
        """Find next valid character index (after processing backspaces)."""
        backspace_count = 0

        while index >= 0:
            if string[index] == '#':
                backspace_count += 1
            elif backspace_count > 0:
                backspace_count -= 1
            else:
                break  # Found valid character
            index -= 1

        return index

    i, j = len(s) - 1, len(t) - 1

    while i >= 0 or j >= 0:
        i = next_valid_char(s, i)
        j = next_valid_char(t, j)

        # Check characters
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            return False  # One string has more characters

        i -= 1
        j -= 1

    return True

print(backspace_compare("ab#c", "ad#c"))  # True (both "ac")
print(backspace_compare("ab##", "c#d#"))  # True (both "")
```

## Complexity Analysis

| Pattern | Time Complexity | Space Complexity | Notes |
|---------|----------------|------------------|-------|
| Opposite direction | O(n) | O(1) | Single pass |
| Fast/Slow (array) | O(n) | O(1) | Single pass |
| Fast/Slow (linked list) | O(n) | O(1) | Cycle detection |
| 3Sum | O(n²) | O(1) | Fix one, two pointers for rest |
| 4Sum | O(n³) | O(1) | Fix two, two pointers for rest |
| Merge sorted arrays | O(n + m) | O(1) | Linear merge |
| Partition | O(n) | O(1) | Single pass |

**Key advantages:**
- Reduces O(n²) nested loops to O(n)
- In-place operations (O(1) space)
- Simple and intuitive
- No complex data structures needed

## When to Use

**Use two pointers when:**

1. **Array/String is sorted** or can be sorted
   - Two Sum in sorted array
   - 3Sum, 4Sum problems

2. **Looking for pairs/triplets** with specific properties
   - Sum equals target
   - Product within range
   - Difference requirements

3. **In-place array modification**
   - Remove duplicates
   - Move elements
   - Partition arrays

4. **Linked list problems**
   - Find middle
   - Detect cycles
   - Remove nth from end

5. **Merging sorted sequences**
   - Merge sorted arrays
   - Find intersection

6. **Optimization over brute force**
   - When nested loop seems necessary
   - Looking for O(n) instead of O(n²)

**Don't use when:**
- Need to track multiple elements (use hash map)
- Array is unsorted and can't be sorted
- Random access pattern required
- Need to preserve original order and can't use stable sort

## Common Pitfalls

### 1. Infinite Loops

```python
# Wrong: Pointers might never meet or cross
def wrong_approach(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        # Forgot to update pointers!
        if some_condition:
            return left

# Correct: Always update at least one pointer
def correct_approach(nums: list[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        if some_condition:
            left += 1
        else:
            right -= 1
```

### 2. Off-by-One Errors

```python
# Careful with loop condition: < vs <=
left, right = 0, len(nums) - 1

# Use < when pointers shouldn't overlap
while left < right:  # Stops when left == right
    pass

# Use <= when you need to process when pointers meet
while left <= right:  # Processes left == right case
    pass
```

### 3. Not Handling Edge Cases

```python
def remove_duplicates(nums: list[int]) -> int:
    # Wrong: Empty array crashes
    # slow = 0
    # for fast in range(1, len(nums)):
    #     ...

    # Correct: Check empty case
    if not nums:
        return 0

    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

### 4. Forgetting to Skip Duplicates

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Must skip duplicates for first element!
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            # ... two pointer logic ...

            if found_triplet:
                # Must skip duplicates for left and right too!
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

    return result
```

### 5. Wrong Pointer Update in Fast/Slow

```python
# Dutch National Flag problem
def sort_colors(nums: list[int]) -> None:
    low, mid, high = 0, 0, len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1  # Safe to increment (we know what we swapped)
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # DON'T increment mid! Need to check swapped element
```

## Practice Problems

### Easy
1. **Two Sum II** (LeetCode 167) - Sorted array
2. **Valid Palindrome** (LeetCode 125)
3. **Remove Duplicates from Sorted Array** (LeetCode 26)
4. **Move Zeroes** (LeetCode 283)
5. **Reverse String** (LeetCode 344)
6. **Squares of a Sorted Array** (LeetCode 977)
7. **Merge Sorted Array** (LeetCode 88)
8. **Intersection of Two Arrays II** (LeetCode 350)
9. **Linked List Cycle** (LeetCode 141)
10. **Middle of the Linked List** (LeetCode 876)

### Medium
11. **3Sum** (LeetCode 15)
12. **Container With Most Water** (LeetCode 11)
13. **Sort Colors** (LeetCode 75)
14. **Remove Duplicates from Sorted Array II** (LeetCode 80)
15. **3Sum Closest** (LeetCode 16)
16. **Remove Nth Node From End of List** (LeetCode 19)
17. **Linked List Cycle II** (LeetCode 142)
18. **Find the Duplicate Number** (LeetCode 287)
19. **Partition Labels** (LeetCode 763)
20. **Backspace String Compare** (LeetCode 844)
21. **Interval List Intersections** (LeetCode 986)
22. **Subarray Product Less Than K** (LeetCode 713)

### Hard
23. **Trapping Rain Water** (LeetCode 42)
24. **4Sum** (LeetCode 18)
25. **Minimum Window Substring** (LeetCode 76) - combines with sliding window
26. **Substring with Concatenation of All Words** (LeetCode 30)

## Additional Resources

### Visualizations
- **VisuAlgo**: https://visualgo.net/en/sorting (for partition visualization)
- **Algorithm Visualizer**: https://algorithm-visualizer.org/

### Tutorials
- **LeetCode Explore**: Two Pointers technique card
- **GeeksforGeeks**: Two Pointer Technique
- **InterviewBit**: Two Pointers problems

### Articles
- "Two Pointers: One of the Most Common Coding Patterns" - Medium
- "Master the Two Pointer Technique" - LeetCode Discuss

### Practice Platforms
- LeetCode Tag: Two Pointers (100+ problems)
- HackerRank: Arrays section
- AlgoExpert: Two Pointers category

---

**Key Takeaways:**
1. Two pointers reduce O(n²) to O(n) for many problems
2. Works best with sorted data or when order doesn't matter
3. Multiple patterns: opposite direction, same direction, fast/slow
4. Essential for linked list problems (cycle detection, middle finding)
5. Combine with sorting for problems like 3Sum, 4Sum
6. Always consider edge cases and pointer update logic

Master two pointers and you'll solve a huge category of interview problems efficiently!
