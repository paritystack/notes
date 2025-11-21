# Interview Patterns - Problem Categorization Guide

## Overview
This guide categorizes common coding interview problems by recognizable patterns. Learning to identify patterns helps you quickly map new problems to known solutions.

## Table of Contents
- [Two Pointers](#two-pointers)
- [Sliding Window](#sliding-window)
- [Fast & Slow Pointers](#fast--slow-pointers)
- [Merge Intervals](#merge-intervals)
- [Cyclic Sort](#cyclic-sort)
- [In-place Reversal of Linked List](#in-place-reversal-of-linked-list)
- [Tree BFS](#tree-bfs)
- [Tree DFS](#tree-dfs)
- [Two Heaps](#two-heaps)
- [Subsets](#subsets)
- [Modified Binary Search](#modified-binary-search)
- [Top K Elements](#top-k-elements)
- [K-way Merge](#k-way-merge)
- [Topological Sort](#topological-sort)
- [Binary Search on Answer](#binary-search-on-answer)
- [Backtracking](#backtracking)
- [Dynamic Programming Patterns](#dynamic-programming-patterns)
- [Graph Patterns](#graph-patterns)
- [String Patterns](#string-patterns)
- [Bit Manipulation](#bit-manipulation)

---

## Two Pointers

### When to Use
- Array/string is sorted
- Need to find pair/triplet with specific property
- Need to compare elements from both ends
- Removing duplicates in-place

### Pattern Recognition
- Keywords: "pair", "triplet", "sorted array", "two sum in sorted array"
- Input: Sorted array or linked list
- Output: Usually indices or count

### Template
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process current pair
        current_sum = arr[left] + arr[right]

        if condition_met:
            return result
        elif need_larger_value:
            left += 1
        else:
            right -= 1

    return default_result
```

### Common Problems
1. **Two Sum II** (sorted array)
2. **Three Sum** - find triplets that sum to zero
3. **Remove Duplicates** - remove duplicates in-place
4. **Container With Most Water** - maximize area
5. **Trapping Rain Water** - calculate trapped water
6. **Valid Palindrome** - check if palindrome
7. **Reverse String/Array** - in-place reversal
8. **Sorted Squares** - square and sort array

### Variations

**Opposite Direction (Most Common)**
```python
left, right = 0, len(arr) - 1
while left < right:
    # Process and move pointers
```

**Same Direction**
```python
slow = fast = 0
for fast in range(len(arr)):
    # Process with slow and fast
    # Move slow conditionally
```

**Three Pointers**
```python
# For three sum
for i in range(len(arr)):
    left, right = i + 1, len(arr) - 1
    while left < right:
        # Find pairs that sum with arr[i]
```

---

## Sliding Window

### When to Use
- Finding subarray/substring with specific property
- Maximum/minimum subarray of size K
- Longest/shortest substring with condition
- Contiguous sequence problems

### Pattern Recognition
- Keywords: "subarray", "substring", "contiguous", "window"
- Need to track elements in a range
- Optimize from O(n²) to O(n)

### Template

**Fixed Size Window**
```python
def fixed_window(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        # Slide window: remove left, add right
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Variable Size Window**
```python
def variable_window(arr, target):
    left = 0
    window_sum = 0
    result = float('inf')

    for right in range(len(arr)):
        # Expand window
        window_sum += arr[right]

        # Shrink window while condition met
        while window_sum >= target:
            result = min(result, right - left + 1)
            window_sum -= arr[left]
            left += 1

    return result
```

### Common Problems
1. **Maximum Sum Subarray of Size K**
2. **Longest Substring with K Distinct Characters**
3. **Fruits Into Baskets** - at most 2 distinct
4. **Minimum Window Substring** - contains all characters
5. **Longest Substring Without Repeating Characters**
6. **Permutation in String** - check if permutation exists
7. **Longest Repeating Character Replacement** - with K replacements
8. **Max Consecutive Ones III** - with K flips

### Window Tracking Patterns

**Using Hash Map**
```python
from collections import defaultdict

window = defaultdict(int)
for right in range(len(arr)):
    window[arr[right]] += 1

    while len(window) > k:  # Shrink condition
        window[arr[left]] -= 1
        if window[arr[left]] == 0:
            del window[arr[left]]
        left += 1
```

**Using Deque (for min/max)**
```python
from collections import deque

dq = deque()  # Store indices
for right in range(len(arr)):
    # Maintain monotonic property
    while dq and arr[dq[-1]] < arr[right]:
        dq.pop()
    dq.append(right)

    # Remove out of window
    while dq[0] < right - k + 1:
        dq.popleft()
```

---

## Fast & Slow Pointers

### When to Use
- Cycle detection in linked list
- Finding middle of linked list
- Palindrome linked list
- Happy number problem

### Pattern Recognition
- Keywords: "cycle", "middle", "linked list"
- Floyd's algorithm
- Usually involves linked structures

### Template
```python
def has_cycle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True  # Cycle detected

    return False
```

### Common Problems
1. **Linked List Cycle** - detect cycle
2. **Linked List Cycle II** - find cycle start
3. **Happy Number** - detect cycle in sequence
4. **Middle of Linked List** - find middle node
5. **Palindrome Linked List** - check palindrome
6. **Reorder List** - rearrange list
7. **Circular Array Loop** - detect cycle in array

### Finding Cycle Start
```python
def find_cycle_start(head):
    # Find meeting point
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break

    if not fast or not fast.next:
        return None  # No cycle

    # Find start of cycle
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow
```

---

## Merge Intervals

### When to Use
- Overlapping intervals
- Meeting rooms
- Scheduling problems
- Range merging

### Pattern Recognition
- Keywords: "intervals", "overlapping", "merge", "schedule"
- Input: Array of intervals
- Often requires sorting first

### Template
```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        # Check overlap
        if current[0] <= last[1]:
            # Merge intervals
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # No overlap
            merged.append(current)

    return merged
```

### Common Problems
1. **Merge Intervals** - merge overlapping intervals
2. **Insert Interval** - insert and merge
3. **Interval List Intersections** - find intersections
4. **Meeting Rooms** - can attend all meetings
5. **Meeting Rooms II** - minimum rooms needed
6. **Non-overlapping Intervals** - minimum removals
7. **Employee Free Time** - common free time

### Interval Intersection
```python
def interval_intersection(A, B):
    result = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Check if intervals overlap
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:
            result.append([start, end])

        # Move pointer of interval that ends first
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result
```

---

## Cyclic Sort

### When to Use
- Array contains numbers from 1 to n
- Finding missing/duplicate numbers
- Sorting in O(n) with limited range

### Pattern Recognition
- Keywords: "1 to n", "missing number", "duplicate"
- Array indices can represent values
- In-place sorting possible

### Template
```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_index = nums[i] - 1

        # If number is not at correct position
        if nums[i] != nums[correct_index]:
            # Swap to correct position
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1

    return nums
```

### Common Problems
1. **Missing Number** - find missing number 0 to n
2. **Find All Missing Numbers** - find all missing
3. **Find Duplicate Number** - find single duplicate
4. **Find All Duplicates** - find all duplicates
5. **First Missing Positive** - smallest positive missing
6. **Find Corrupt Pair** - duplicate and missing

### Finding Missing Number
```python
def find_missing(nums):
    i = 0
    while i < len(nums):
        correct_index = nums[i]
        if nums[i] < len(nums) and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1

    # Find first missing
    for i in range(len(nums)):
        if nums[i] != i:
            return i

    return len(nums)
```

---

## In-place Reversal of Linked List

### When to Use
- Reverse entire linked list
- Reverse sublist
- Reverse every k nodes
- Reverse alternating k nodes

### Pattern Recognition
- Keywords: "reverse", "linked list", "in-place"
- Pointer manipulation required
- Usually O(1) space

### Template
```python
def reverse_list(head):
    prev = None
    current = head

    while current:
        # Save next
        next_node = current.next

        # Reverse link
        current.next = prev

        # Move pointers
        prev = current
        current = next_node

    return prev  # New head
```

### Common Problems
1. **Reverse Linked List** - reverse entire list
2. **Reverse Linked List II** - reverse between positions
3. **Reverse Nodes in k-Group** - reverse k at a time
4. **Swap Nodes in Pairs** - swap every two nodes
5. **Rotate List** - rotate by k positions
6. **Palindrome Linked List** - check if palindrome

### Reverse Sublist
```python
def reverse_between(head, left, right):
    if left == right:
        return head

    # Skip to position before left
    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    # Reverse from left to right
    reverse_start = prev.next
    current = reverse_start.next

    for _ in range(right - left):
        # Move current to after prev
        reverse_start.next = current.next
        current.next = prev.next
        prev.next = current
        current = reverse_start.next

    return dummy.next
```

---

## Tree BFS

### When to Use
- Level-order traversal
- Minimum depth problems
- Zigzag traversal
- Level averages/sums

### Pattern Recognition
- Keywords: "level", "breadth-first", "minimum depth"
- Process tree level by level
- Use queue

### Template
```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

### Common Problems
1. **Binary Tree Level Order Traversal**
2. **Binary Tree Zigzag Level Order**
3. **Minimum Depth of Binary Tree**
4. **Level Order Successor** - find next node
5. **Connect Level Order Siblings** - populate next pointer
6. **Average of Levels**
7. **Right Side View** - rightmost nodes

### Level Processing Variants

**Single Value per Level**
```python
level_max = max(node.val for node in level_nodes)
```

**Zigzag Order**
```python
if level % 2 == 1:
    current_level.reverse()
```

**Right Side View**
```python
if i == level_size - 1:  # Last node in level
    result.append(node.val)
```

---

## Tree DFS

### When to Use
- Path sum problems
- All paths from root to leaf
- Diameter/height calculations
- Lowest common ancestor

### Pattern Recognition
- Keywords: "path", "sum", "root to leaf", "depth"
- Recursive exploration
- Backtracking often needed

### Template

**Preorder (Root → Left → Right)**
```python
def preorder_dfs(root, path=[]):
    if not root:
        return

    # Process current node
    path.append(root.val)

    # Recurse
    preorder_dfs(root.left, path)
    preorder_dfs(root.right, path)

    # Backtrack if needed
    path.pop()
```

**Postorder (Left → Right → Root)**
```python
def postorder_dfs(root):
    if not root:
        return 0

    left_result = postorder_dfs(root.left)
    right_result = postorder_dfs(root.right)

    # Process current node with children results
    return process(root.val, left_result, right_result)
```

### Common Problems
1. **Path Sum** - check if path with sum exists
2. **Path Sum II** - all paths with sum
3. **Sum Root to Leaf Numbers** - sum all paths
4. **Path Sum III** - paths not starting from root
5. **Diameter of Binary Tree** - longest path
6. **Maximum Path Sum** - maximum sum path
7. **Lowest Common Ancestor** - find LCA
8. **Validate BST** - check if valid BST

### Pattern Variants

**Path Tracking**
```python
def find_paths(root, target, current_path=[], all_paths=[]):
    if not root:
        return

    current_path.append(root.val)

    if not root.left and not root.right and sum(current_path) == target:
        all_paths.append(list(current_path))

    find_paths(root.left, target, current_path, all_paths)
    find_paths(root.right, target, current_path, all_paths)

    current_path.pop()  # Backtrack
```

**Global Maximum (Diameter, Max Path Sum)**
```python
def helper(root):
    nonlocal max_value

    if not root:
        return 0

    left = max(0, helper(root.left))
    right = max(0, helper(root.right))

    max_value = max(max_value, left + right + root.val)

    return max(left, right) + root.val
```

---

## Two Heaps

### When to Use
- Median from data stream
- Sliding window median
- Scheduling with priorities

### Pattern Recognition
- Keywords: "median", "balance", "priority"
- Need quick access to middle elements
- Continuous stream of data

### Template
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negative values)
        self.large = []  # Min heap

    def add_num(self, num):
        # Add to max heap (small)
        heapq.heappush(self.small, -num)

        # Balance: move largest from small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Ensure small has equal or one more element
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

### Common Problems
1. **Find Median from Data Stream**
2. **Sliding Window Median**
3. **IPO** - maximize capital
4. **Next Interval** - find next interval start

---

## Subsets

### When to Use
- Generate all combinations
- Generate all permutations
- Generate all subsets
- Power set problems

### Pattern Recognition
- Keywords: "all combinations", "all permutations", "subsets", "power set"
- Exponential time complexity O(2ⁿ) or O(n!)
- Backtracking approach

### Template

**Subsets (Iterative)**
```python
def subsets(nums):
    result = [[]]

    for num in nums:
        # Add num to all existing subsets
        result += [curr + [num] for curr in result]

    return result
```

**Subsets (Backtracking)**
```python
def subsets_backtrack(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

**Permutations**
```python
def permutations(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:])

    backtrack([], nums)
    return result
```

### Common Problems
1. **Subsets** - all subsets
2. **Subsets II** - with duplicates
3. **Permutations** - all permutations
4. **Permutations II** - with duplicates
5. **Combinations** - all combinations of size k
6. **Combination Sum** - combinations that sum to target
7. **Letter Case Permutation** - toggle letter cases
8. **Generate Parentheses** - valid combinations

---

## Modified Binary Search

### When to Use
- Searching in rotated sorted array
- Finding peak element
- Search in infinite sorted array
- Finding boundary (first/last occurrence)

### Pattern Recognition
- Keywords: "sorted", "rotated", "search", "find first/last"
- Can eliminate half of search space
- O(log n) time complexity

### Template
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

    return -1
```

### Common Problems
1. **Binary Search** - standard search
2. **Search in Rotated Sorted Array** - rotated array
3. **Find Minimum in Rotated Array** - find minimum
4. **Search in 2D Matrix** - sorted matrix
5. **Find Peak Element** - peak in array
6. **First and Last Position** - find boundaries
7. **Search Insert Position** - insertion point

### Variant Templates

**Find First Occurrence**
```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**Rotated Array Search**
```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

---

## Top K Elements

### When to Use
- K largest/smallest elements
- K most frequent elements
- K closest points

### Pattern Recognition
- Keywords: "top k", "k largest", "k smallest", "k most frequent"
- Heap-based solutions
- O(n log k) time complexity

### Template

**Using Min Heap (for K largest)**
```python
import heapq

def find_k_largest(nums, k):
    min_heap = []

    for num in nums:
        heapq.heappush(min_heap, num)

        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return list(min_heap)
```

**Using Max Heap (for K smallest)**
```python
def find_k_smallest(nums, k):
    max_heap = []

    for num in nums:
        heapq.heappush(max_heap, -num)  # Negate for max heap

        if len(max_heap) > k:
            heapq.heappop(max_heap)

    return [-x for x in max_heap]
```

### Common Problems
1. **Kth Largest Element** - find kth largest
2. **Top K Frequent Elements** - k most frequent
3. **K Closest Points to Origin** - k closest points
4. **Kth Smallest in Sorted Matrix** - matrix problem
5. **Sort Characters by Frequency** - sort by count
6. **Reorganize String** - no adjacent same characters

### QuickSelect Alternative
```python
def quickselect(nums, k):
    """Find kth largest in O(n) average"""
    k = len(nums) - k  # Convert to kth smallest

    def partition(left, right):
        pivot = nums[right]
        i = left

        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1

        nums[i], nums[right] = nums[right], nums[i]
        return i

    left, right = 0, len(nums) - 1

    while left <= right:
        pivot_index = partition(left, right)

        if pivot_index == k:
            return nums[k]
        elif pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index - 1
```

---

## K-way Merge

### When to Use
- Merge k sorted arrays/lists
- Smallest range from k lists
- K pairs with smallest sums

### Pattern Recognition
- Keywords: "k sorted", "merge k", "k lists"
- Min heap with k elements
- Process smallest element repeatedly

### Template
```python
import heapq

def merge_k_sorted(lists):
    min_heap = []
    result = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result
```

### Common Problems
1. **Merge K Sorted Lists** - merge linked lists
2. **Merge K Sorted Arrays** - merge arrays
3. **Smallest Range from K Lists** - find smallest range
4. **K Pairs with Smallest Sums** - find k pairs
5. **Find Median from Data Stream** - related pattern

---

## Topological Sort

### When to Use
- Course schedule problems
- Build order problems
- Dependency resolution
- DAG ordering

### Pattern Recognition
- Keywords: "prerequisites", "dependencies", "order", "schedule"
- Directed acyclic graph (DAG)
- Two algorithms: DFS and Kahn's (BFS)

### Template

**Kahn's Algorithm (BFS)**
```python
from collections import deque, defaultdict

def topological_sort(num_courses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    # Build graph
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with nodes having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Reduce in-degree of neighbors
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all nodes processed (no cycle)
    return result if len(result) == num_courses else []
```

**DFS-based**
```python
def topological_sort_dfs(num_courses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    visited = [0] * num_courses  # 0: unvisited, 1: visiting, 2: visited
    result = []

    def dfs(node):
        if visited[node] == 1:  # Cycle detected
            return False
        if visited[node] == 2:  # Already processed
            return True

        visited[node] = 1  # Mark as visiting

        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False

        visited[node] = 2  # Mark as visited
        result.append(node)
        return True

    for i in range(num_courses):
        if not dfs(i):
            return []

    return result[::-1]  # Reverse for correct order
```

### Common Problems
1. **Course Schedule** - can finish all courses
2. **Course Schedule II** - find valid order
3. **Alien Dictionary** - determine letter order
4. **Minimum Height Trees** - find tree centers
5. **Sequence Reconstruction** - verify unique order

---

## Binary Search on Answer

### When to Use
- Minimize/maximize something with verification function
- Search in large answer space
- "Minimum capacity", "least time" problems

### Pattern Recognition
- Keywords: "minimum capacity", "minimum time", "least", "at most"
- Can verify if answer X works in O(n)
- Answer space is sorted/monotonic

### Template
```python
def binary_search_on_answer(arr, target):
    def is_valid(mid):
        """Check if mid is a valid answer"""
        # Implement verification logic
        return True  # or False

    left, right = min_value, max_value

    while left < right:
        mid = left + (right - left) // 2

        if is_valid(mid):
            right = mid  # Try smaller value
        else:
            left = mid + 1  # Need larger value

    return left
```

### Common Problems
1. **Split Array Largest Sum** - minimize largest sum
2. **Capacity To Ship Packages** - minimum capacity
3. **Koko Eating Bananas** - minimum eating speed
4. **Minimize Max Distance** - gas stations problem
5. **Magnetic Force Between Balls** - maximize min distance
6. **Find K-th Smallest Pair Distance** - kth distance

### Example: Ship Packages
```python
def ship_within_days(weights, days):
    def can_ship(capacity):
        current_weight = 0
        days_needed = 1

        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight

        return days_needed <= days

    left = max(weights)  # Minimum capacity needed
    right = sum(weights)  # Maximum capacity

    while left < right:
        mid = left + (right - left) // 2

        if can_ship(mid):
            right = mid
        else:
            left = mid + 1

    return left
```

---

## Backtracking

### When to Use
- Generate all solutions
- Constraint satisfaction
- Combinatorial search
- Game solving (Sudoku, N-Queens)

### Pattern Recognition
- Keywords: "all solutions", "combinations", "permutations"
- Build solution incrementally
- Undo choices (backtrack)

### Template
```python
def backtrack_template():
    result = []

    def backtrack(path, choices):
        # Base case: solution found
        if is_solution(path):
            result.append(path[:])  # Make copy
            return

        # Try each choice
        for choice in choices:
            # Make choice
            path.append(choice)

            # Recurse with updated choices
            backtrack(path, get_next_choices(choice))

            # Undo choice (backtrack)
            path.pop()

    backtrack([], initial_choices)
    return result
```

### Common Problems
1. **N-Queens** - place n queens
2. **Sudoku Solver** - solve sudoku
3. **Word Search** - find word in grid
4. **Palindrome Partitioning** - partition into palindromes
5. **Combination Sum** - combinations summing to target
6. **Letter Combinations of Phone** - generate combinations
7. **Generate Parentheses** - valid parentheses

### Problem-Specific Templates

**Grid Backtracking (Word Search)**
```python
def word_search(board, word):
    def backtrack(i, j, k):
        if k == len(word):
            return True

        if (i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or
            board[i][j] != word[k]):
            return False

        temp = board[i][j]
        board[i][j] = '#'  # Mark visited

        found = (backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or
                 backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1))

        board[i][j] = temp  # Restore
        return found

    for i in range(len(board)):
        for j in range(len(board[0])):
            if backtrack(i, j, 0):
                return True
    return False
```

**Constraint Checking (N-Queens)**
```python
def solve_n_queens(n):
    def backtrack(row, cols, diag1, diag2):
        if row == n:
            result.append(construct_board())
            return

        for col in range(n):
            if col in cols or (row + col) in diag1 or (row - col) in diag2:
                continue

            # Make choice
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row + col)
            diag2.add(row - col)

            backtrack(row + 1, cols, diag1, diag2)

            # Undo choice
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row + col)
            diag2.remove(row - col)

    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0, set(), set(), set())
    return result
```

---

## Dynamic Programming Patterns

### 1. Linear DP (1D)

**When**: Sequence problems, single dimension state

```python
# House Robber pattern
dp[i] = max(dp[i-1], dp[i-2] + nums[i])

# Climbing Stairs pattern
dp[i] = dp[i-1] + dp[i-2]

# Coin Change pattern (unbounded knapsack)
for coin in coins:
    for amount in range(coin, target + 1):
        dp[amount] = min(dp[amount], dp[amount - coin] + 1)
```

**Problems**: House Robber, Climbing Stairs, Min Cost Climbing Stairs, Decode Ways

### 2. 0/1 Knapsack (2D)

**When**: Include/exclude decisions, limited items

```python
dp = [[0] * (W + 1) for _ in range(n + 1)]

for i in range(1, n + 1):
    for w in range(1, W + 1):
        if weights[i-1] <= w:
            dp[i][w] = max(dp[i-1][w],  # Exclude
                          dp[i-1][w - weights[i-1]] + values[i-1])  # Include
        else:
            dp[i][w] = dp[i-1][w]
```

**Space Optimized (1D)**:
```python
dp = [0] * (W + 1)
for i in range(n):
    for w in range(W, weights[i] - 1, -1):  # Reverse order
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

**Problems**: Partition Equal Subset Sum, Target Sum, Last Stone Weight II

### 3. Unbounded Knapsack

**When**: Items can be used multiple times

```python
dp = [0] * (W + 1)
for i in range(n):
    for w in range(weights[i], W + 1):  # Forward order
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

**Problems**: Coin Change, Coin Change 2, Rod Cutting, Minimum Cost for Tickets

### 4. Longest Common Subsequence (LCS)

**When**: Two sequences, finding common patterns

```python
dp = [[0] * (n + 1) for _ in range(m + 1)]

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**Problems**: LCS, Edit Distance, Longest Palindromic Subsequence, Distinct Subsequences

### 5. Longest Increasing Subsequence (LIS)

**When**: Finding increasing subsequence

**O(n²) DP**:
```python
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], dp[j] + 1)
```

**O(n log n) Binary Search**:
```python
import bisect
tails = []
for num in nums:
    pos = bisect.bisect_left(tails, num)
    if pos == len(tails):
        tails.append(num)
    else:
        tails[pos] = num
return len(tails)
```

**Problems**: LIS, Number of LIS, Russian Doll Envelopes, Maximum Height by Stacking Cuboids

### 6. Palindrome DP

**When**: Palindrome-related problems

```python
# Check if s[i:j+1] is palindrome
dp = [[False] * n for _ in range(n)]

# All single characters are palindromes
for i in range(n):
    dp[i][i] = True

# Check for length 2+
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        if s[i] == s[j]:
            dp[i][j] = (length == 2) or dp[i+1][j-1]
```

**Problems**: Longest Palindromic Substring, Palindrome Partitioning, Palindrome Partitioning II

### 7. State Machine DP

**When**: Different states with transitions

```python
# Buy/Sell Stock with states
hold = -prices[0]  # State: holding stock
sold = 0           # State: just sold
rest = 0           # State: resting

for price in prices[1:]:
    prev_sold = sold
    sold = hold + price
    hold = max(hold, rest - price)
    rest = max(rest, prev_sold)
```

**Problems**: Best Time to Buy/Sell Stock series, Paint House series

### 8. Digit DP

**When**: Counting numbers with digit constraints

```python
@lru_cache(None)
def dp(pos, tight, started):
    if pos == len(num):
        return 1 if started else 0

    limit = int(num[pos]) if tight else 9
    result = 0

    for digit in range(0, limit + 1):
        if not started and digit == 0:
            result += dp(pos + 1, False, False)
        else:
            result += dp(pos + 1, tight and digit == limit, True)

    return result
```

**Problems**: Count Numbers with Unique Digits, Numbers At Most N Given Digit Set

---

## Graph Patterns

### 1. Union-Find (Disjoint Set)

**When**: Connectivity, grouping, cycle detection

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True
```

**Problems**: Number of Connected Components, Graph Valid Tree, Redundant Connection, Accounts Merge

### 2. Matrix DFS/BFS

**When**: Grid problems, island counting, flood fill

```python
def dfs_grid(grid, i, j, visited):
    if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or
        visited[i][j] or grid[i][j] == 0):
        return

    visited[i][j] = True

    # Explore 4 directions
    for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
        dfs_grid(grid, i + di, j + dj, visited)
```

**Problems**: Number of Islands, Max Area of Island, Surrounded Regions, Pacific Atlantic Water Flow

### 3. Shortest Path in Grid

**When**: Finding shortest path in matrix/grid

```python
from collections import deque

def shortest_path_grid(grid, start, end):
    queue = deque([(start[0], start[1], 0)])  # (row, col, dist)
    visited = set([start])

    while queue:
        i, j, dist = queue.popleft()

        if (i, j) == end:
            return dist

        for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
            ni, nj = i + di, j + dj

            if (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and
                (ni, nj) not in visited and grid[ni][nj] != 1):

                visited.add((ni, nj))
                queue.append((ni, nj, dist + 1))

    return -1
```

**Problems**: Shortest Path in Binary Matrix, Rotting Oranges, Walls and Gates, 01 Matrix

---

## String Patterns

### 1. Anagram Detection

```python
from collections import Counter

# Using Counter
Counter(s1) == Counter(s2)

# Using sorted
sorted(s1) == sorted(s2)
```

### 2. Palindrome Check

```python
# Two pointers
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### 3. String Matching (KMP)

```python
def kmp_search(text, pattern):
    # Build LPS array
    lps = [0] * len(pattern)
    j = 0

    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j

    # Search
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - j + 1  # Found at index

    return -1
```

---

## Bit Manipulation

### Common Operations

```python
# Check if i-th bit is set
(num >> i) & 1

# Set i-th bit
num | (1 << i)

# Clear i-th bit
num & ~(1 << i)

# Toggle i-th bit
num ^ (1 << i)

# Clear lowest set bit
num & (num - 1)

# Get lowest set bit
num & -num

# Count set bits
bin(num).count('1')

# Check power of 2
num > 0 and (num & (num - 1)) == 0
```

### Common Problems
1. **Single Number** - XOR all numbers
2. **Number of 1 Bits** - count set bits
3. **Reverse Bits** - reverse bit pattern
4. **Power of Two** - check power
5. **Sum of Two Integers** - add without + operator
6. **Missing Number** - XOR to find missing

---

## Pattern Recognition Checklist

When you see a problem, ask:

1. **Array/String sorted?** → Binary Search, Two Pointers
2. **Subarray/substring?** → Sliding Window
3. **Linked list cycle?** → Fast & Slow Pointers
4. **Tree traversal?** → BFS (level-order) or DFS (path)
5. **Graph connectivity?** → Union-Find, DFS/BFS
6. **Shortest path?** → BFS (unweighted), Dijkstra (weighted)
7. **Overlapping intervals?** → Merge Intervals
8. **All combinations?** → Backtracking
9. **Top k elements?** → Heap
10. **Optimal substructure?** → Dynamic Programming
11. **Dependencies/order?** → Topological Sort
12. **Range [1...n]?** → Cyclic Sort
13. **Median/balance?** → Two Heaps
14. **Merge k lists?** → K-way Merge
15. **Verify if X works?** → Binary Search on Answer

## Study Plan

**Week 1-2**: Master 5 patterns
- Two Pointers
- Sliding Window
- Fast & Slow Pointers
- Tree BFS/DFS
- Binary Search

**Week 3-4**: Intermediate patterns
- Merge Intervals
- Top K Elements
- Subsets
- Modified Binary Search
- Backtracking basics

**Week 5-6**: Advanced patterns
- Dynamic Programming
- Graph algorithms
- Two Heaps
- K-way Merge
- Topological Sort

**Week 7-8**: Practice mixed problems and pattern recognition

Solve **10-15 problems per pattern** before moving to the next.
