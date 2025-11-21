# Implementation Patterns Guide

## Overview

This guide covers common implementation patterns and techniques used across data structures and algorithms. These patterns are cross-cutting concerns that appear repeatedly in different contexts.

## Table of Contents

1. [Iterative vs Recursive Approaches](#iterative-vs-recursive-approaches)
2. [Sentinel Nodes](#sentinel-nodes)
3. [Dummy Head Pattern](#dummy-head-pattern)
4. [Two-Pointer Techniques](#two-pointer-techniques)
5. [Common Pitfalls](#common-pitfalls)
6. [Debugging Strategies](#debugging-strategies)
7. [Edge Cases Checklist](#edge-cases-checklist)

## Iterative vs Recursive Approaches

Many algorithms can be implemented either iteratively or recursively. Understanding the tradeoffs helps choose the right approach.

### When to Use Recursion

**Advantages:**
- More elegant and readable for naturally recursive problems
- Simpler code for tree/graph traversal
- Natural fit for divide-and-conquer algorithms
- Easier to reason about for backtracking problems

**Disadvantages:**
- Stack overflow risk for deep recursion
- Higher memory overhead (call stack)
- Slower due to function call overhead
- Harder to debug

```python
# Recursive: Elegant for tree problems
def max_depth_recursive(root):
    """Find maximum depth of binary tree - Recursive"""
    if not root:
        return 0
    return 1 + max(max_depth_recursive(root.left),
                   max_depth_recursive(root.right))

# Recursive: Natural for divide-and-conquer
def binary_search_recursive(arr, target, left, right):
    """Binary search - Recursive"""
    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Recursive: Clear for backtracking
def generate_subsets(nums, index=0, current=None, result=None):
    """Generate all subsets - Recursive"""
    if current is None:
        current = []
    if result is None:
        result = []

    if index == len(nums):
        result.append(current[:])
        return result

    # Don't include nums[index]
    generate_subsets(nums, index + 1, current, result)

    # Include nums[index]
    current.append(nums[index])
    generate_subsets(nums, index + 1, current, result)
    current.pop()

    return result
```

### When to Use Iteration

**Advantages:**
- No stack overflow risk
- Lower memory usage (no call stack)
- Often faster (no function call overhead)
- Easier to control execution flow

**Disadvantages:**
- Can be less intuitive for some problems
- May require explicit stack/queue data structure
- More complex state management

```python
# Iterative: Better for large inputs
def max_depth_iterative(root):
    """Find maximum depth of binary tree - Iterative BFS"""
    if not root:
        return 0

    queue = [(root, 1)]
    max_depth = 0

    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return max_depth

# Iterative: No risk of stack overflow
def binary_search_iterative(arr, target):
    """Binary search - Iterative"""
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

# Iterative: Using explicit stack
def inorder_traversal_iterative(root):
    """Inorder traversal - Iterative with stack"""
    result = []
    stack = []
    current = root

    while current or stack:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left

        # Process node
        current = stack.pop()
        result.append(current.val)

        # Move to right subtree
        current = current.right

    return result
```

### Converting Recursion to Iteration

**Pattern 1: Tail Recursion → Simple Loop**

```python
# Recursive tail call
def sum_recursive(n):
    """Tail recursive sum"""
    def helper(n, acc):
        if n == 0:
            return acc
        return helper(n - 1, acc + n)  # Tail call
    return helper(n, 0)

# Iterative equivalent
def sum_iterative(n):
    """Iterative sum"""
    acc = 0
    while n > 0:
        acc += n
        n -= 1
    return acc
```

**Pattern 2: Tree Recursion → Explicit Stack**

```python
# Recursive DFS
def dfs_recursive(node, visited=None):
    """Recursive depth-first search"""
    if visited is None:
        visited = set()

    if node in visited:
        return

    visited.add(node)
    print(node.val)

    for neighbor in node.neighbors:
        dfs_recursive(neighbor, visited)

# Iterative DFS
def dfs_iterative(start):
    """Iterative depth-first search"""
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        print(node.val)

        for neighbor in node.neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
```

**Pattern 3: Multiple Recursion → Queue/Stack**

```python
# Recursive level-order traversal
def level_order_recursive(root):
    """BFS using recursion (less intuitive)"""
    result = []

    def helper(node, level):
        if not node:
            return

        if len(result) <= level:
            result.append([])

        result[level].append(node.val)
        helper(node.left, level + 1)
        helper(node.right, level + 1)

    helper(root, 0)
    return result

# Iterative level-order traversal (clearer!)
def level_order_iterative(root):
    """BFS using queue"""
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Decision Guide

| Factor | Choose Recursion | Choose Iteration |
|--------|-----------------|------------------|
| Problem type | Trees, graphs, divide-and-conquer | Array traversal, simple loops |
| Input size | Small to moderate | Large (avoid stack overflow) |
| Code clarity | Natural recursive structure | Complex state management |
| Performance | Not critical | Critical (tight loops) |
| Stack space | Available | Limited |
| Debugging | Can handle complexity | Need simple flow |

## Sentinel Nodes

Sentinel nodes are dummy nodes that simplify boundary conditions by eliminating special cases.

### Concept

Instead of checking for null/None repeatedly, use a sentinel node that:
- Always exists
- Simplifies edge case handling
- Reduces conditional logic

### Linked List with Sentinel

```python
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Without sentinel
class LinkedListNoSentinel:
    def __init__(self):
        self.head = None  # Can be None!

    def insert_at_beginning(self, val):
        """Need to handle empty list specially"""
        new_node = Node(val)
        if self.head is None:  # Special case!
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node

    def delete(self, val):
        """Complex with special cases"""
        if not self.head:  # Empty list
            return

        if self.head.val == val:  # Delete head - special case!
            self.head = self.head.next
            return

        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next

# With sentinel
class LinkedListWithSentinel:
    def __init__(self):
        self.sentinel = Node(0)  # Always exists!
        self.head = self.sentinel

    def insert_at_beginning(self, val):
        """Simpler - no special cases"""
        new_node = Node(val)
        new_node.next = self.sentinel.next
        self.sentinel.next = new_node

    def delete(self, val):
        """Simpler - no head special case"""
        current = self.sentinel
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next

    def get_first_real_node(self):
        """Actual first node after sentinel"""
        return self.sentinel.next
```

### Doubly Linked List with Sentinel

```python
class DNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedListWithSentinel:
    """Using sentinel eliminates all null checks!"""

    def __init__(self):
        # Sentinel points to itself when empty
        self.sentinel = DNode(0)
        self.sentinel.prev = self.sentinel
        self.sentinel.next = self.sentinel

    def insert_after(self, node, val):
        """Insert val after node - no null checks needed!"""
        new_node = DNode(val)
        new_node.prev = node
        new_node.next = node.next
        node.next.prev = new_node
        node.next = new_node

    def insert_at_beginning(self, val):
        """Insert at beginning"""
        self.insert_after(self.sentinel, val)

    def insert_at_end(self, val):
        """Insert at end"""
        self.insert_after(self.sentinel.prev, val)

    def delete(self, node):
        """Delete node - works for any node!"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def is_empty(self):
        """Check if list is empty"""
        return self.sentinel.next == self.sentinel
```

### Binary Search Tree with Sentinel

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTWithSentinel:
    """Using sentinel NIL node"""

    def __init__(self):
        # Sentinel NIL node - represents null
        self.NIL = TreeNode(None)
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.root = self.NIL

    def insert(self, val):
        """Insert value - no null checks!"""
        new_node = TreeNode(val, self.NIL, self.NIL)

        parent = self.NIL
        current = self.root

        while current != self.NIL:
            parent = current
            if val < current.val:
                current = current.left
            else:
                current = current.right

        if parent == self.NIL:
            self.root = new_node
        elif val < parent.val:
            parent.left = new_node
        else:
            parent.right = new_node

    def search(self, val):
        """Search - cleaner without null checks"""
        current = self.root
        while current != self.NIL and current.val != val:
            if val < current.val:
                current = current.left
            else:
                current = current.right
        return current if current != self.NIL else None
```

### When to Use Sentinels

**Use sentinels when:**
- Many null/None checks clutter code
- Implementing doubly linked lists
- Building complex tree structures (Red-Black trees)
- Boundary conditions are repetitive

**Don't use sentinels when:**
- Simple data structure with few operations
- Memory is extremely constrained
- Sentinel adds more complexity than it removes

## Dummy Head Pattern

The dummy head (or dummy node) pattern is a special case of sentinel nodes, specifically for linked lists.

### Concept

Create a temporary dummy node at the beginning to simplify operations that modify the head.

### Use Cases

#### 1. List Merging

```python
def merge_sorted_lists(l1, l2):
    """
    Merge two sorted lists
    Dummy head eliminates head special case
    """
    dummy = Node(0)  # Dummy head
    current = dummy

    # Merge without worrying about head
    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # Attach remaining
    current.next = l1 if l1 else l2

    return dummy.next  # Skip dummy head
```

#### 2. List Partitioning

```python
def partition_list(head, x):
    """
    Partition list around value x
    Elements < x before elements >= x
    """
    # Two dummy heads for two partitions
    less_dummy = Node(0)
    greater_dummy = Node(0)

    less = less_dummy
    greater = greater_dummy

    current = head
    while current:
        if current.val < x:
            less.next = current
            less = less.next
        else:
            greater.next = current
            greater = greater.next
        current = current.next

    # Connect partitions
    greater.next = None  # Important: terminate list
    less.next = greater_dummy.next

    return less_dummy.next  # Skip dummy
```

#### 3. List Reversal

```python
def reverse_list(head):
    """Reverse linked list using dummy head"""
    dummy = Node(0)
    dummy.next = head

    # Reverse by moving nodes to front
    current = head
    while current and current.next:
        # Move current.next to front
        temp = current.next
        current.next = temp.next
        temp.next = dummy.next
        dummy.next = temp

    return dummy.next
```

#### 4. Removing Elements

```python
def remove_elements(head, val):
    """
    Remove all nodes with given value
    Dummy head handles removal of head nodes
    """
    dummy = Node(0)
    dummy.next = head

    current = dummy
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next

    return dummy.next  # New head
```

#### 5. Insertion Sort on List

```python
def insertion_sort_list(head):
    """Insertion sort on linked list using dummy"""
    dummy = Node(0)

    current = head
    while current:
        # Save next before inserting
        next_node = current.next

        # Find insertion position
        prev = dummy
        while prev.next and prev.next.val < current.val:
            prev = prev.next

        # Insert current
        current.next = prev.next
        prev.next = current

        current = next_node

    return dummy.next
```

### Pattern Template

```python
def list_operation(head):
    """Generic template for dummy head pattern"""

    # 1. Create dummy head
    dummy = Node(0)
    dummy.next = head

    # 2. Use dummy.next to work with list
    current = dummy
    while current.next:
        # Perform operations on current.next
        # Can safely delete/modify nodes
        current = current.next

    # 3. Return dummy.next (actual head)
    return dummy.next
```

### Why Dummy Head Works

```python
# Without dummy head - complex!
def remove_head_if_zero(head):
    """Remove head if it's zero"""
    while head and head.val == 0:  # Special case for head!
        head = head.next

    current = head
    while current and current.next:
        if current.next.val == 0:
            current.next = current.next.next
        else:
            current = current.next

    return head

# With dummy head - simpler!
def remove_zeros(head):
    """Remove all zeros"""
    dummy = Node(-1)
    dummy.next = head

    current = dummy
    while current.next:
        if current.next.val == 0:
            current.next = current.next.next
        else:
            current = current.next

    return dummy.next  # Automatically handles removed head
```

## Two-Pointer Techniques

Two-pointer patterns are powerful techniques for solving array and linked list problems efficiently.

### Pattern 1: Fast and Slow Pointers

**Use cases**: Cycle detection, finding middle, finding nth from end

```python
# Detect cycle in linked list
def has_cycle(head):
    """
    Floyd's Cycle Detection
    Fast pointer moves 2x speed of slow pointer
    If cycle exists, they meet
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next          # Move 1 step
        fast = fast.next.next     # Move 2 steps

        if slow == fast:
            return True

    return False

# Find cycle start
def detect_cycle_start(head):
    """Find where cycle begins"""
    slow = fast = head

    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow

# Find middle of linked list
def find_middle(head):
    """
    When fast reaches end, slow is at middle
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow

# Find nth node from end
def nth_from_end(head, n):
    """
    Fast pointer moves n steps ahead
    Then both move together until fast reaches end
    """
    fast = slow = head

    # Move fast n steps ahead
    for _ in range(n):
        if not fast:
            return None
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    return slow

# Check if linked list is palindrome
def is_palindrome(head):
    """Find middle, reverse second half, compare"""
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    prev = None
    while slow:
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node

    # Compare
    left, right = head, prev
    while right:  # right is shorter if odd length
        if left.val != right.val:
            return False
        left = left.next
        right = right.next

    return True
```

### Pattern 2: Two Pointers from Both Ends

**Use cases**: Two sum, container with most water, valid palindrome

```python
# Two sum on sorted array
def two_sum_sorted(arr, target):
    """
    O(n) time, O(1) space
    Move pointers based on sum
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return None

# Three sum
def three_sum(nums):
    """Find all unique triplets that sum to zero"""
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates
        if i > 0 and nums[i] == nums[i-1]:
            continue

        # Two sum on remaining array
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result

# Container with most water
def max_area(heights):
    """
    Move pointer with smaller height
    Maximize area = height * width
    """
    left, right = 0, len(heights) - 1
    max_area = 0

    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_area = max(max_area, width * height)

        # Move pointer with smaller height
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_area

# Valid palindrome
def is_valid_palindrome(s):
    """Check if string is palindrome (ignoring non-alphanumeric)"""
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

# Reverse vowels
def reverse_vowels(s):
    """Reverse only vowels in string"""
    vowels = set('aeiouAEIOU')
    s = list(s)
    left, right = 0, len(s) - 1

    while left < right:
        # Find vowels from both ends
        while left < right and s[left] not in vowels:
            left += 1
        while left < right and s[right] not in vowels:
            right -= 1

        # Swap
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

    return ''.join(s)
```

### Pattern 3: Sliding Window (Two Pointers, Same Direction)

**Use cases**: Subarray sum, longest substring, minimum window

```python
# Maximum sum subarray of size k
def max_sum_subarray(arr, k):
    """Fixed size sliding window"""
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        # Slide window: remove left, add right
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Longest substring without repeating characters
def length_of_longest_substring(s):
    """Variable size sliding window"""
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Shrink window while duplicate exists
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        # Add current character
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length

# Minimum window substring
def min_window(s, t):
    """
    Find minimum window in s containing all characters of t
    """
    if not t or not s:
        return ""

    # Character counts needed
    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)  # Unique characters needed
    formed = 0  # Unique characters formed so far

    window_counts = {}
    left = 0
    min_len = float('inf')
    min_left = 0

    for right in range(len(s)):
        # Add character from right
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if frequency matches
        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        # Try to shrink window
        while left <= right and formed == required:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            # Remove from left
            char = s[left]
            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

    return "" if min_len == float('inf') else s[min_left:min_left + min_len]

# Subarray sum equals K
def subarray_sum(nums, k):
    """Count subarrays with sum = k"""
    count = 0
    current_sum = 0
    sum_counts = {0: 1}  # Handle sum from start

    for num in nums:
        current_sum += num

        # Check if (current_sum - k) exists
        if current_sum - k in sum_counts:
            count += sum_counts[current_sum - k]

        # Add current sum to map
        sum_counts[current_sum] = sum_counts.get(current_sum, 0) + 1

    return count
```

### Pattern 4: Partition/Quickselect Pointers

**Use cases**: Quick sort partition, Dutch national flag, move zeros

```python
# Move zeros to end
def move_zeros(nums):
    """
    Maintain: [non-zero elements | zeros | unprocessed]
    left pointer: position for next non-zero
    """
    left = 0  # Position for next non-zero

    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1

# Dutch national flag (3-way partition)
def sort_colors(nums):
    """
    Sort array of 0s, 1s, 2s
    Maintain: [0s | 1s | unprocessed | 2s]
    """
    left = 0        # Position for next 0
    right = len(nums) - 1  # Position for next 2
    current = 0

    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 2:
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
            # Don't increment current (need to process swapped element)
        else:  # nums[current] == 1
            current += 1

# Partition around pivot (QuickSort)
def partition(arr, low, high):
    """
    Partition array around pivot
    Elements < pivot on left, >= pivot on right
    """
    pivot = arr[high]
    i = low - 1  # Position for next element < pivot

    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

## Common Pitfalls

### 1. Off-by-One Errors

```python
# WRONG: Misses last element
def binary_search_wrong(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:  # Should be left <= right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# CORRECT
def binary_search_correct(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:  # Include case where left == right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# WRONG: Infinite loop
def find_insert_position_wrong(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid  # Should be mid + 1 - can cause infinite loop!
        else:
            right = mid

    return left

# CORRECT
def find_insert_position_correct(arr, target):
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1  # Move past mid
        else:
            right = mid

    return left
```

### 2. Integer Overflow in Mid Calculation

```python
# WRONG: Can overflow with large indices
def binary_search_overflow(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2  # Can overflow if left + right > MAX_INT
        # ... rest of code

# CORRECT: Avoid overflow
def binary_search_safe(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Safe from overflow
        # ... rest of code
```

### 3. Modifying Collection While Iterating

```python
# WRONG: Modifying list while iterating
def remove_evens_wrong(nums):
    for num in nums:
        if num % 2 == 0:
            nums.remove(num)  # Skips elements!

# CORRECT: Iterate backwards
def remove_evens_correct(nums):
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] % 2 == 0:
            nums.pop(i)

# CORRECT: List comprehension
def remove_evens_best(nums):
    return [num for num in nums if num % 2 != 0]
```

### 4. Not Handling Empty Input

```python
# WRONG: Crashes on empty input
def find_max_wrong(arr):
    max_val = arr[0]  # IndexError if arr is empty!
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val

# CORRECT: Handle empty case
def find_max_correct(arr):
    if not arr:
        return None  # Or raise exception, or return float('-inf')

    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val
```

### 5. Incorrect Boundary Conditions in Sliding Window

```python
# WRONG: Window size calculation error
def max_sum_subarray_wrong(arr, k):
    if len(arr) < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr) + 1):  # Off by one!
        window_sum = window_sum - arr[i-k] + arr[i]  # IndexError
        max_sum = max(max_sum, window_sum)

    return max_sum

# CORRECT
def max_sum_subarray_correct(arr, k):
    if len(arr) < k:
        return 0

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):  # Correct range
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

### 6. Not Considering Negative Numbers

```python
# WRONG: Assumes positive numbers
def max_subarray_wrong(arr):
    max_sum = 0  # Wrong for all-negative array!
    current_sum = 0

    for num in arr:
        current_sum = max(0, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Example: [-5, -2, -3] returns 0, but should return -2

# CORRECT: Kadane's algorithm
def max_subarray_correct(arr):
    if not arr:
        return 0

    max_sum = current_sum = arr[0]  # Start with first element

    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### 7. Shallow Copy vs Deep Copy

```python
# WRONG: Shallow copy creates reference
def modify_2d_array_wrong(n):
    row = [0] * n
    matrix = [row] * n  # All rows reference same list!

    matrix[0][0] = 1
    print(matrix)  # [[1,0,0], [1,0,0], [1,0,0]] - all rows changed!

# CORRECT: Create separate lists
def modify_2d_array_correct(n):
    matrix = [[0] * n for _ in range(n)]  # Each row is different list

    matrix[0][0] = 1
    print(matrix)  # [[1,0,0], [0,0,0], [0,0,0]] - only first row changed
```

### 8. Reference vs Value in Tree/Graph Problems

```python
# WRONG: Losing reference to head
def delete_node_wrong(head, val):
    if head.val == val:
        head = head.next  # Doesn't affect caller's head!
        return

    current = head
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
            return
        current = current.next

# CORRECT: Return new head
def delete_node_correct(head, val):
    if head.val == val:
        return head.next  # Return new head

    current = head
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
            return head
        current = current.next

    return head
```

## Debugging Strategies

### 1. Print Debugging

```python
def debug_binary_search(arr, target):
    """Add strategic print statements"""
    left, right = 0, len(arr) - 1

    iteration = 0
    while left <= right:
        mid = (left + right) // 2

        print(f"Iteration {iteration}:")
        print(f"  left={left}, right={right}, mid={mid}")
        print(f"  arr[mid]={arr[mid]}, target={target}")
        print(f"  Current window: {arr[left:right+1]}")

        if arr[mid] == target:
            print(f"Found at index {mid}")
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

        iteration += 1

    print("Not found")
    return -1
```

### 2. Visualize Data Structures

```python
def visualize_linked_list(head):
    """Print linked list structure"""
    values = []
    current = head
    visited = set()

    while current:
        if id(current) in visited:
            values.append(f"[CYCLE to {current.val}]")
            break
        visited.add(id(current))
        values.append(str(current.val))
        current = current.next

    print(" -> ".join(values))

def visualize_tree(root, level=0, prefix="Root: "):
    """Print tree structure"""
    if root:
        print(" " * (level * 4) + prefix + str(root.val))
        if root.left or root.right:
            if root.left:
                visualize_tree(root.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")
            if root.right:
                visualize_tree(root.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")
```

### 3. Assertions and Invariants

```python
def binary_search_with_assertions(arr, target):
    """Use assertions to verify invariants"""
    # Precondition
    assert arr == sorted(arr), "Array must be sorted"
    assert len(arr) > 0, "Array must not be empty"

    left, right = 0, len(arr) - 1

    while left <= right:
        # Invariant: if target exists, it's in arr[left:right+1]
        mid = (left + right) // 2

        # Invariant checks
        assert 0 <= left <= len(arr), "left out of bounds"
        assert 0 <= right < len(arr), "right out of bounds"
        assert left <= mid <= right, "mid not between left and right"

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### 4. Test with Edge Cases

```python
def test_function_thoroughly():
    """Comprehensive test cases"""
    # Empty input
    assert my_function([]) == expected_empty

    # Single element
    assert my_function([1]) == expected_single

    # Two elements
    assert my_function([1, 2]) == expected_two

    # All same elements
    assert my_function([5, 5, 5, 5]) == expected_same

    # Sorted array
    assert my_function([1, 2, 3, 4, 5]) == expected_sorted

    # Reverse sorted
    assert my_function([5, 4, 3, 2, 1]) == expected_reverse

    # With duplicates
    assert my_function([1, 2, 2, 3, 3, 3]) == expected_dups

    # Large input
    assert my_function(list(range(10000))) == expected_large

    # Negative numbers
    assert my_function([-5, -2, 0, 3, 7]) == expected_negative

    print("All tests passed!")
```

### 5. Rubber Duck Debugging

```python
# Explain your code line by line to someone (or a rubber duck!)

def explain_algorithm():
    """
    Explaining forces you to clarify your thinking:

    1. What is the input?
    2. What is the expected output?
    3. What is the algorithm doing at each step?
    4. Why is this step necessary?
    5. What are the invariants?
    6. What are the edge cases?
    """
    pass
```

## Edge Cases Checklist

### Array/List Edge Cases

```python
def test_edge_cases():
    # 1. Empty array/list
    arr = []

    # 2. Single element
    arr = [1]

    # 3. Two elements
    arr = [1, 2]

    # 4. All elements same
    arr = [5, 5, 5, 5, 5]

    # 5. Already sorted
    arr = [1, 2, 3, 4, 5]

    # 6. Reverse sorted
    arr = [5, 4, 3, 2, 1]

    # 7. With duplicates
    arr = [1, 2, 2, 3, 3, 3, 4]

    # 8. Negative numbers
    arr = [-5, -3, -1, 0, 2, 4]

    # 9. Mixed positive/negative
    arr = [-2, -1, 0, 1, 2]

    # 10. Very large/small values
    arr = [-10**9, 0, 10**9]

    # 11. Overflow potential
    arr = [2**31 - 1, 2**31 - 1]  # Sum overflows 32-bit int
```

### Linked List Edge Cases

```python
def test_linked_list_edges():
    # 1. None/null head
    head = None

    # 2. Single node
    head = Node(1)

    # 3. Two nodes
    head = Node(1, Node(2))

    # 4. Circular list
    node1 = Node(1)
    node2 = Node(2)
    node1.next = node2
    node2.next = node1

    # 5. Cycle in middle
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node1.next = node2
    node2.next = node3
    node3.next = node2  # Cycle

    # 6. All nodes same value
    head = Node(5, Node(5, Node(5)))
```

### Tree Edge Cases

```python
def test_tree_edges():
    # 1. Empty tree
    root = None

    # 2. Single node
    root = TreeNode(1)

    # 3. Only left children (skewed)
    root = TreeNode(1, TreeNode(2, TreeNode(3)))

    # 4. Only right children (skewed)
    root = TreeNode(1, None, TreeNode(2, None, TreeNode(3)))

    # 5. Perfect binary tree
    root = TreeNode(1,
                    TreeNode(2, TreeNode(4), TreeNode(5)),
                    TreeNode(3, TreeNode(6), TreeNode(7)))

    # 6. Complete binary tree
    root = TreeNode(1,
                    TreeNode(2, TreeNode(4), TreeNode(5)),
                    TreeNode(3, TreeNode(6)))

    # 7. All values same
    # 8. Negative values
    # 9. Very deep tree (potential stack overflow)
```

### String Edge Cases

```python
def test_string_edges():
    # 1. Empty string
    s = ""

    # 2. Single character
    s = "a"

    # 3. All same character
    s = "aaaaaaa"

    # 4. Palindrome
    s = "racecar"

    # 5. With spaces
    s = "hello world"

    # 6. Special characters
    s = "!@#$%^&*()"

    # 7. Unicode/emoji
    s = "Hello 👋 World 🌍"

    # 8. Very long string
    s = "a" * 10**6
```

### Numeric Edge Cases

```python
def test_numeric_edges():
    # 1. Zero
    n = 0

    # 2. Negative
    n = -5

    # 3. Maximum int
    n = 2**31 - 1

    # 4. Minimum int
    n = -2**31

    # 5. Float precision
    x = 0.1 + 0.2  # != 0.3 exactly!

    # 6. Division by zero
    try:
        result = 10 / 0
    except ZeroDivisionError:
        pass

    # 7. Overflow
    # 8. Underflow
```

## Summary

### Key Patterns

1. **Iterative vs Recursive**: Choose based on problem type, input size, and stack constraints
2. **Sentinel Nodes**: Eliminate null checks and simplify boundary conditions
3. **Dummy Head**: Simplify linked list operations that modify the head
4. **Two Pointers**: Efficient for arrays, linked lists, and strings
5. **Common Pitfalls**: Watch for off-by-one, overflow, shallow copy, edge cases

### Best Practices

- Always handle edge cases: empty input, single element, null/None
- Use assertions to verify invariants
- Test with diverse inputs: sorted, reverse sorted, duplicates, negatives
- Visualize data structures when debugging
- Explain your algorithm step by step
- Consider time and space complexity tradeoffs
- Write clean, readable code with meaningful variable names

### When in Doubt

1. Start with brute force
2. Identify bottlenecks
3. Consider appropriate data structures
4. Look for patterns (two pointers, sliding window, etc.)
5. Test with edge cases
6. Optimize gradually

## Related Topics

- [Complexity Guide](complexity_guide.md) - Understanding algorithm complexity
- [Data Structures README](README.md) - Overview of all data structures
- [Linked Lists](linked_lists.md), [Trees](trees.md), [Arrays](arrays.md) - Specific implementations

Remember: Patterns are tools, not rules. Understand why they work, and you'll know when to use them!
