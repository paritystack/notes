# Interview Questions

A comprehensive guide to technical interview preparation covering algorithmic patterns, data structures, problem-solving strategies, and common interview questions.

## Table of Contents

- [Problem-Solving Approach](#problem-solving-approach)
- [Data Structures Fundamentals](#data-structures-fundamentals)
- [Algorithm Patterns](#algorithm-patterns)
- [Dynamic Programming](#dynamic-programming)
- [System Design Basics](#system-design-basics)
- [Behavioral Questions](#behavioral-questions)
- [Language-Specific Tips](#language-specific-tips)
- [Interview Strategy](#interview-strategy)

## Problem-Solving Approach

### The UMPIRE Method

1. **Understand** - Clarify the problem
   - What are the inputs and outputs?
   - What are the constraints?
   - What are the edge cases?
   - Can I restate the problem in my own words?

2. **Match** - Pattern recognition
   - Does this problem match a known pattern?
   - What data structure would be most appropriate?
   - Have I solved a similar problem before?

3. **Plan** - Design the algorithm
   - What's the brute force approach?
   - Can I optimize it?
   - What's the time/space complexity?
   - Walk through examples

4. **Implement** - Write the code
   - Start with clear variable names
   - Handle edge cases
   - Keep it readable

5. **Review** - Test and validate
   - Test with example inputs
   - Test edge cases
   - Check for off-by-one errors

6. **Evaluate** - Analyze complexity
   - Time complexity
   - Space complexity
   - Can it be optimized further?

### Complexity Analysis Quick Reference

| Complexity | Name | Example |
|------------|------|---------|
| O(1) | Constant | Hash table lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Array traversal |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Nested loops |
| O(n³) | Cubic | Triple nested loops |
| O(2ⁿ) | Exponential | Recursive fibonacci |
| O(n!) | Factorial | Permutations |

## Data Structures Fundamentals

### Arrays and Strings

**Key Operations:**
- Access: O(1)
- Search: O(n)
- Insert: O(n)
- Delete: O(n)

**Common Techniques:**
```python
# Two pointers
def reverse_string(s):
    left, right = 0, len(s) - 1
    s = list(s)
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return ''.join(s)

# Sliding window
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Prefix sum
def range_sum_query(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i+1] = prefix[i] + arr[i]

    def query(left, right):
        return prefix[right+1] - prefix[left]

    return query
```

**Java:**
```java
// Two pointers - Remove duplicates from sorted array
public int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    int slow = 0;
    for (int fast = 1; fast < nums.length; fast++) {
        if (nums[fast] != nums[slow]) {
            slow++;
            nums[slow] = nums[fast];
        }
    }
    return slow + 1;
}
```

**C++:**
```cpp
// Sliding window - Longest substring without repeating chars
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> chars;
    int left = 0, maxLen = 0;

    for (int right = 0; right < s.length(); right++) {
        if (chars.find(s[right]) != chars.end()) {
            left = max(left, chars[s[right]] + 1);
        }
        chars[s[right]] = right;
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

### Linked Lists

**Key Operations:**
- Access: O(n)
- Search: O(n)
- Insert: O(1) with pointer
- Delete: O(1) with pointer

**Common Patterns:**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Reverse linked list (iterative)
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev

# Reverse linked list (recursive)
def reverse_list_recursive(head):
    if not head or not head.next:
        return head

    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head

# Detect cycle (Floyd's algorithm)
def has_cycle(head):
    if not head:
        return False

    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False

# Find middle node
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Merge two sorted lists
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 if l1 else l2
    return dummy.next
```

### Stacks and Queues

**Stack - LIFO (Last In First Out)**
- Push: O(1)
- Pop: O(1)
- Peek: O(1)

```python
# Valid parentheses
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)

    return not stack

# Daily temperatures (monotonic stack)
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []  # stores indices

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)

    return result
```

**Java - Stack Applications:**
```java
// Evaluate Reverse Polish Notation
public int evalRPN(String[] tokens) {
    Stack<Integer> stack = new Stack<>();

    for (String token : tokens) {
        if (token.equals("+")) {
            stack.push(stack.pop() + stack.pop());
        } else if (token.equals("-")) {
            int b = stack.pop();
            int a = stack.pop();
            stack.push(a - b);
        } else if (token.equals("*")) {
            stack.push(stack.pop() * stack.pop());
        } else if (token.equals("/")) {
            int b = stack.pop();
            int a = stack.pop();
            stack.push(a / b);
        } else {
            stack.push(Integer.parseInt(token));
        }
    }

    return stack.pop();
}
```

**Queue - FIFO (First In First Out)**
```python
from collections import deque

# Moving average from data stream
class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0

    def next(self, val):
        self.queue.append(val)
        self.sum += val

        if len(self.queue) > self.size:
            self.sum -= self.queue.popleft()

        return self.sum / len(self.queue)
```

### Hash Tables

**Key Operations:**
- Insert: O(1) average
- Delete: O(1) average
- Search: O(1) average

```python
# Two sum
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Group anagrams
def group_anagrams(strs):
    anagrams = {}
    for s in strs:
        key = ''.join(sorted(s))
        if key not in anagrams:
            anagrams[key] = []
        anagrams[key].append(s)
    return list(anagrams.values())

# Longest consecutive sequence
def longest_consecutive(nums):
    num_set = set(nums)
    longest = 0

    for num in num_set:
        if num - 1 not in num_set:  # Start of sequence
            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            longest = max(longest, length)

    return longest
```

**C++ - Unordered Map:**
```cpp
// Subarray sum equals K
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixSum;
    prefixSum[0] = 1;
    int sum = 0, count = 0;

    for (int num : nums) {
        sum += num;
        if (prefixSum.find(sum - k) != prefixSum.end()) {
            count += prefixSum[sum - k];
        }
        prefixSum[sum]++;
    }

    return count;
}
```

### Trees

**Binary Tree Traversals:**

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Inorder (Left, Root, Right) - for BST gives sorted order
def inorder(root):
    result = []
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)
    traverse(root)
    return result

# Preorder (Root, Left, Right)
def preorder(root):
    result = []
    def traverse(node):
        if not node:
            return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)
    traverse(root)
    return result

# Postorder (Left, Right, Root)
def postorder(root):
    result = []
    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node.val)
    traverse(root)
    return result

# Level order (BFS)
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result
```

**Binary Search Tree Operations:**
```python
# Search in BST
def search_bst(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst(root.left, val)
    return search_bst(root.right, val)

# Insert into BST
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

# Validate BST
def is_valid_bst(root):
    def validate(node, low=float('-inf'), high=float('inf')):
        if not node:
            return True
        if node.val <= low or node.val >= high:
            return False
        return (validate(node.left, low, node.val) and
                validate(node.right, node.val, high))
    return validate(root)

# Lowest common ancestor in BST
def lowest_common_ancestor_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

### Heaps (Priority Queue)

**Min Heap and Max Heap:**
```python
import heapq

# Kth largest element
def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]

# Alternative: Min heap of size k
def find_kth_largest_heap(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)

    return heap[0]

# Top K frequent elements
def top_k_frequent(nums, k):
    from collections import Counter
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Merge K sorted lists
def merge_k_lists(lists):
    heap = []
    dummy = ListNode(0)
    current = dummy

    # Initialize heap with first node from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**Java - Priority Queue:**
```java
// Find median from data stream
class MedianFinder {
    PriorityQueue<Integer> maxHeap; // Lower half
    PriorityQueue<Integer> minHeap; // Upper half

    public MedianFinder() {
        maxHeap = new PriorityQueue<>((a, b) -> b - a);
        minHeap = new PriorityQueue<>();
    }

    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());

        if (minHeap.size() > maxHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }

    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.peek();
        }
        return (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}
```

### Graphs

**Graph Representations:**
```python
# Adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Adjacency matrix
n = 6
matrix = [[0] * n for _ in range(n)]
```

**Graph Traversals:**
```python
# DFS (recursive)
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    visited.add(node)
    print(node)

    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

    return visited

# DFS (iterative)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return visited

# BFS
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

**Common Graph Algorithms:**
```python
# Number of islands (DFS)
def num_islands(grid):
    if not grid:
        return 0

    def dfs(i, j):
        if (i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or
            grid[i][j] == '0'):
            return

        grid[i][j] = '0'  # Mark as visited
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1

    return count

# Clone graph
def clone_graph(node):
    if not node:
        return None

    clones = {}

    def dfs(node):
        if node in clones:
            return clones[node]

        clone = Node(node.val)
        clones[node] = clone

        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)

# Course schedule (cycle detection)
def can_finish(numCourses, prerequisites):
    graph = {i: [] for i in range(numCourses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    visited = [0] * numCourses  # 0: unvisited, 1: visiting, 2: visited

    def has_cycle(course):
        if visited[course] == 1:  # Currently visiting
            return True
        if visited[course] == 2:  # Already visited
            return False

        visited[course] = 1
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        visited[course] = 2
        return False

    for course in range(numCourses):
        if has_cycle(course):
            return False

    return True
```

## Algorithm Patterns

### 1. Sliding Window

**Pattern:** Use two pointers to create a window that slides through the array/string.

**When to use:**
- Contiguous subarray/substring problems
- Finding max/min in subarrays of size k
- Longest/shortest substring with conditions

**Time Complexity:** O(n)

```python
# Maximum sum subarray of size k (fixed window)
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Longest substring with at most K distinct characters (dynamic window)
def length_of_longest_substring_k_distinct(s, k):
    char_count = {}
    left = 0
    max_len = 0

    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len

# Minimum window substring
def min_window(s, t):
    from collections import Counter

    if not s or not t:
        return ""

    need = Counter(t)
    have = {}

    required = len(need)
    formed = 0

    left = 0
    min_len = float('inf')
    min_left = 0

    for right in range(len(s)):
        char = s[right]
        have[char] = have.get(char, 0) + 1

        if char in need and have[char] == need[char]:
            formed += 1

        while formed == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            char = s[left]
            have[char] -= 1
            if char in need and have[char] < need[char]:
                formed -= 1
            left += 1

    return "" if min_len == float('inf') else s[min_left:min_left + min_len]
```

**Java:**
```java
// Longest substring without repeating characters
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> chars = new HashMap<>();
    int left = 0, maxLen = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (chars.containsKey(c)) {
            left = Math.max(left, chars.get(c) + 1);
        }
        chars.put(c, right);
        maxLen = Math.max(maxLen, right - left + 1);
    }

    return maxLen;
}
```

### 2. Two Pointers

**Pattern:** Use two pointers moving towards/away from each other or at different speeds.

**When to use:**
- Sorted array problems
- Pair finding problems
- Palindrome checks
- Partition problems

```python
# Two sum in sorted array
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# Three sum
def three_sum(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
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
def max_area(height):
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        max_water = max(max_water, min(height[left], height[right]) * width)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

# Remove duplicates from sorted array
def remove_duplicates(nums):
    if not nums:
        return 0

    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

**C++:**
```cpp
// Trapping rain water
int trap(vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int leftMax = 0, rightMax = 0;
    int water = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }

    return water;
}
```

### 3. Fast and Slow Pointers

**Pattern:** Two pointers moving at different speeds (usually slow: +1, fast: +2).

**When to use:**
- Cycle detection
- Finding middle element
- Finding nth element from end

```python
# Linked list cycle detection
def has_cycle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False

# Find cycle start
def detect_cycle(head):
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

# Happy number
def is_happy(n):
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    slow = n
    fast = get_next(n)

    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))

    return fast == 1

# Find middle of linked list
def find_middle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow

# Palindrome linked list
def is_palindrome(head):
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
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next

    return True
```

### 4. Merge Intervals

**Pattern:** Sort intervals and merge overlapping ones.

**When to use:**
- Overlapping intervals
- Meeting rooms
- Insert intervals

```python
# Merge intervals
def merge(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged

# Insert interval
def insert(intervals, newInterval):
    result = []
    i = 0

    # Add all intervals before newInterval
    while i < len(intervals) and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    result.append(newInterval)

    # Add remaining intervals
    while i < len(intervals):
        result.append(intervals[i])
        i += 1

    return result

# Meeting rooms II (minimum rooms needed)
def min_meeting_rooms(intervals):
    if not intervals:
        return 0

    start_times = sorted([i[0] for i in intervals])
    end_times = sorted([i[1] for i in intervals])

    rooms = 0
    max_rooms = 0
    s = e = 0

    while s < len(start_times):
        if start_times[s] < end_times[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1

    return max_rooms
```

**Java:**
```java
// Non-overlapping intervals (min removals)
public int eraseOverlapIntervals(int[][] intervals) {
    if (intervals.length == 0) return 0;

    Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
    int end = intervals[0][1];
    int count = 0;

    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] < end) {
            count++;
        } else {
            end = intervals[i][1];
        }
    }

    return count;
}
```

### 5. Cyclic Sort

**Pattern:** Use array indices to place elements in their correct position.

**When to use:**
- Arrays with elements in range [1, n]
- Finding missing/duplicate numbers

```python
# Cyclic sort
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_index = nums[i] - 1
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    return nums

# Find missing number
def find_missing_number(nums):
    i = 0
    n = len(nums)

    while i < n:
        correct_index = nums[i]
        if correct_index < n and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1

    for i in range(n):
        if nums[i] != i:
            return i

    return n

# Find all duplicates
def find_duplicates(nums):
    i = 0
    while i < len(nums):
        correct_index = nums[i] - 1
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1

    duplicates = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            duplicates.append(nums[i])

    return duplicates

# First missing positive
def first_missing_positive(nums):
    n = len(nums)

    # Place each number in its right place
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]

    # Find first missing
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1
```

### 6. In-place Reversal of Linked List

**Pattern:** Reverse links between nodes without using extra space.

```python
# Reverse linked list
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev

# Reverse sublist from position m to n
def reverse_between(head, m, n):
    if m == n:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    # Move to position m-1
    for _ in range(m - 1):
        prev = prev.next

    # Reverse from m to n
    current = prev.next
    for _ in range(n - m):
        next_node = current.next
        current.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next

# Reverse nodes in k-group
def reverse_k_group(head, k):
    def reverse(head, k):
        prev = None
        current = head
        for _ in range(k):
            if not current:
                return head  # Not enough nodes
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        return prev

    # Check if k nodes exist
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1

    if count < k:
        return head

    # Reverse first k nodes
    new_head = reverse(head, k)
    # Recursively reverse remaining
    head.next = reverse_k_group(node, k)

    return new_head
```

### 7. Tree BFS (Breadth-First Search)

**Pattern:** Level-order traversal using a queue.

**When to use:**
- Level order traversal
- Finding minimum depth
- Level-wise processing

```python
from collections import deque

# Level order traversal
def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

# Zigzag level order
def zigzag_level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right

    return result

# Right side view
def right_side_view(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # Last node in level
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result

# Minimum depth
def min_depth(root):
    if not root:
        return 0

    queue = deque([(root, 1)])

    while queue:
        node, depth = queue.popleft()

        if not node.left and not node.right:
            return depth

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return 0
```

### 8. Tree DFS (Depth-First Search)

**Pattern:** Recursive or stack-based traversal.

**When to use:**
- Path problems
- Sum problems
- Tree structure validation

```python
# Maximum depth
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Path sum
def has_path_sum(root, targetSum):
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == targetSum

    return (has_path_sum(root.left, targetSum - root.val) or
            has_path_sum(root.right, targetSum - root.val))

# All paths from root to leaf
def binary_tree_paths(root):
    if not root:
        return []

    paths = []

    def dfs(node, path):
        if not node.left and not node.right:
            paths.append(path + str(node.val))
            return

        if node.left:
            dfs(node.left, path + str(node.val) + "->")
        if node.right:
            dfs(node.right, path + str(node.val) + "->")

    dfs(root, "")
    return paths

# Path sum II (all paths)
def path_sum(root, targetSum):
    result = []

    def dfs(node, remaining, path):
        if not node:
            return

        path.append(node.val)

        if not node.left and not node.right and remaining == node.val:
            result.append(list(path))
        else:
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)

        path.pop()

    dfs(root, targetSum, [])
    return result

# Diameter of binary tree
def diameter_of_binary_tree(root):
    diameter = 0

    def height(node):
        nonlocal diameter
        if not node:
            return 0

        left = height(node.left)
        right = height(node.right)
        diameter = max(diameter, left + right)

        return 1 + max(left, right)

    height(root)
    return diameter
```

### 9. Two Heaps

**Pattern:** Use max heap and min heap to maintain median or balance.

**When to use:**
- Finding median in stream
- Sliding window median

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated values)
        self.large = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)

        # Balance: largest in small <= smallest in large
        if self.small and self.large and -self.small[0] > self.large[0]:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

# Sliding window median
def median_sliding_window(nums, k):
    from sortedcontainers import SortedList

    window = SortedList(nums[:k])
    medians = []

    for i in range(k, len(nums) + 1):
        if k % 2 == 0:
            medians.append((window[k//2-1] + window[k//2]) / 2.0)
        else:
            medians.append(float(window[k//2]))

        if i < len(nums):
            window.remove(nums[i-k])
            window.add(nums[i])

    return medians
```

### 10. Subsets and Backtracking

**Pattern:** Explore all possibilities through recursion.

**When to use:**
- Combinations
- Permutations
- Subsets

```python
# Subsets
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(list(path))
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Subsets II (with duplicates)
def subsets_with_dup(nums):
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(list(path))
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Permutations
def permute(nums):
    result = []

    def backtrack(path):
        if len(path) == len(nums):
            result.append(list(path))
            return

        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path)
            path.pop()

    backtrack([])
    return result

# Combinations
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(list(path))
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result

# Letter combinations of phone number
def letter_combinations(digits):
    if not digits:
        return []

    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index, path):
        if index == len(digits):
            result.append(path)
            return

        for letter in phone[digits[index]]:
            backtrack(index + 1, path + letter)

    backtrack(0, "")
    return result

# N-Queens
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check anti-diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return result
```

### 11. Modified Binary Search

**Pattern:** Binary search with modifications for specific problems.

**When to use:**
- Rotated sorted arrays
- Finding boundaries
- Search in 2D matrix

```python
# Binary search
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

# Search in rotated sorted array
def search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# Find minimum in rotated sorted array
def find_min(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]

# Search 2D matrix
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = left + (right - left) // 2
        num = matrix[mid // n][mid % n]

        if num == target:
            return True
        elif num < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

# Find peak element
def find_peak_element(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left
```

### 12. Topological Sort

**Pattern:** Order nodes in directed acyclic graph by dependencies.

**When to use:**
- Course schedule
- Build dependencies
- Task scheduling

```python
from collections import deque, defaultdict

# Topological sort (Kahn's algorithm - BFS)
def topological_sort_bfs(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == n else []

# Topological sort (DFS)
def topological_sort_dfs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited
    result = []

    def dfs(node):
        if visited[node] == 1:  # Cycle detected
            return False
        if visited[node] == 2:
            return True

        visited[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False

        visited[node] = 2
        result.append(node)
        return True

    for i in range(n):
        if visited[i] == 0:
            if not dfs(i):
                return []

    return result[::-1]

# Course schedule II
def find_order(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == numCourses else []
```

### 13. K-way Merge

**Pattern:** Merge K sorted lists using a heap.

```python
import heapq

# Merge K sorted lists
def merge_k_sorted_lists(lists):
    heap = []
    dummy = ListNode(0)
    current = dummy

    # Initialize heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next

# Merge K sorted arrays
def merge_k_sorted_arrays(arrays):
    heap = []
    result = []

    # Initialize heap with first element from each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    while heap:
        val, array_idx, element_idx = heapq.heappop(heap)
        result.append(val)

        if element_idx + 1 < len(arrays[array_idx]):
            next_val = arrays[array_idx][element_idx + 1]
            heapq.heappush(heap, (next_val, array_idx, element_idx + 1))

    return result

# Kth smallest in sorted matrix
def kth_smallest(matrix, k):
    n = len(matrix)
    heap = []

    # Add first element from each row
    for r in range(min(k, n)):
        heapq.heappush(heap, (matrix[r][0], r, 0))

    count = 0
    while heap:
        val, r, c = heapq.heappop(heap)
        count += 1

        if count == k:
            return val

        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c+1], r, c+1))

    return -1
```

### 14. Monotonic Stack/Queue

**Pattern:** Maintain elements in monotonic order.

**When to use:**
- Next greater/smaller element
- Stock span
- Sliding window maximum

```python
from collections import deque

# Next greater element
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []

    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

# Daily temperatures
def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx
        stack.append(i)

    return result

# Largest rectangle in histogram
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    heights.append(0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area

# Sliding window maximum
def max_sliding_window(nums, k):
    result = []
    dq = deque()  # stores indices

    for i, num in enumerate(nums):
        # Remove elements outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### 15. Union-Find (Disjoint Set)

**Pattern:** Track connected components and perform union operations.

**When to use:**
- Connected components
- Detect cycles in undirected graph
- Account merging

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

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

        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Number of connected components
def count_components(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count

# Redundant connection
def find_redundant_connection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    return []

# Accounts merge
def accounts_merge(accounts):
    from collections import defaultdict

    uf = UnionFind(len(accounts))
    email_to_id = {}

    # Build union-find
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            else:
                email_to_id[email] = i

    # Group emails by component
    components = defaultdict(set)
    for email, idx in email_to_id.items():
        components[uf.find(idx)].add(email)

    # Build result
    result = []
    for idx, emails in components.items():
        result.append([accounts[idx][0]] + sorted(emails))

    return result
```

## Dynamic Programming

### 1D DP

```python
# Climbing stairs
def climb_stairs(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1

# House robber
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2, prev1 = 0, 0
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2, prev1 = prev1, current

    return prev1

# Longest increasing subsequence
def length_of_lis(nums):
    if not nums:
        return 0

    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Word break
def word_break(s, wordDict):
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[len(s)]

# Decode ways
def num_decodings(s):
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1

    for i in range(2, n + 1):
        # One digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]

        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[n]
```

### 2D DP

```python
# Unique paths
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# Minimum path sum
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                grid[i][j] += grid[i][j-1]
            elif j == 0:
                grid[i][j] += grid[i-1][j]
            else:
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])

    return grid[m-1][n-1]

# Longest common subsequence
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Edit distance
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )

    return dp[m][n]

# Regular expression matching
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle patterns like a*, a*b*, etc.
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == s[i-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # 0 occurrence
                if p[j-2] == s[i-1] or p[j-2] == '.':
                    dp[i][j] |= dp[i-1][j]  # 1+ occurrence

    return dp[m][n]
```

### Knapsack DP

```python
# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    values[i-1] + dp[i-1][w - weights[i-1]]
                )
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]

# Partition equal subset sum
def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]

# Coin change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Coin change II (number of ways)
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]
```

## System Design Basics

### Key Concepts

**1. Scalability**
- Vertical scaling (scale up): Increase resources on single machine
- Horizontal scaling (scale out): Add more machines

**2. Load Balancing**
- Distribute requests across multiple servers
- Algorithms: Round robin, least connections, IP hash

**3. Caching**
- Reduce database load
- Types: Client-side, CDN, server-side, database
- Strategies: Cache-aside, write-through, write-back

**4. Database Design**
- SQL vs NoSQL
- Sharding and partitioning
- Replication (master-slave, master-master)
- Indexing

**5. Message Queues**
- Decouple components
- Examples: RabbitMQ, Apache Kafka
- Patterns: Pub-sub, point-to-point

**6. Microservices**
- Service-oriented architecture
- API Gateway
- Service discovery

**7. CAP Theorem**
- Consistency: All nodes see same data
- Availability: Every request gets response
- Partition tolerance: System works despite network partitions
- Can only guarantee 2 of 3

### Common System Design Questions

**Design URL Shortener**
- Requirements: Shorten URL, redirect, analytics
- Database: Key-value store (short → long URL)
- Encoding: Base62 encoding
- Scale: Caching, load balancing

**Design Twitter**
- Core features: Tweet, follow, timeline
- Fan-out: Push model (write heavy) vs pull model (read heavy)
- Timeline: Cache recent tweets
- Scale: Sharding users, replication

**Design Rate Limiter**
- Algorithms: Token bucket, leaky bucket, fixed/sliding window
- Storage: Redis
- Implementation: Middleware/API gateway

## Behavioral Questions

### STAR Method

**S**ituation: Set the context
**T**ask: Describe the challenge
**A**ction: Explain what you did
**R**esult: Share the outcome

### Common Questions

**1. Tell me about yourself**
- Brief professional background
- Current role and responsibilities
- Why interested in this position

**2. Why do you want to work here?**
- Company mission alignment
- Technology stack interest
- Growth opportunities

**3. Tell me about a challenging project**
- Use STAR method
- Focus on problem-solving
- Highlight technical decisions

**4. Conflict with team member**
- Stay professional
- Focus on resolution
- What you learned

**5. Failure/mistake**
- Be honest
- Focus on learning
- How you improved

**6. Questions for interviewer**
- Team structure and collaboration
- Technology stack and tools
- Growth and learning opportunities
- Project lifecycle and development process

## Language-Specific Tips

### Python

```python
# List comprehension
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}

# Enumerate
for i, val in enumerate(['a', 'b', 'c']):
    print(f"{i}: {val}")

# Zip
for a, b in zip([1, 2, 3], ['a', 'b', 'c']):
    print(f"{a}: {b}")

# Lambda
squared = list(map(lambda x: x**2, [1, 2, 3]))
evens = list(filter(lambda x: x % 2 == 0, range(10)))

# Collections
from collections import defaultdict, Counter, deque
dd = defaultdict(int)
counter = Counter([1, 2, 2, 3, 3, 3])
queue = deque([1, 2, 3])

# Heapq
import heapq
heap = [3, 1, 4, 1, 5]
heapq.heapify(heap)

# Sorting
nums.sort()  # in-place
sorted_nums = sorted(nums)  # returns new list
nums.sort(key=lambda x: (x[0], -x[1]))  # custom key
```

### Java

```java
// ArrayList vs LinkedList
List<Integer> arrayList = new ArrayList<>();  // Fast random access
List<Integer> linkedList = new LinkedList<>();  // Fast insertion/deletion

// HashMap
Map<String, Integer> map = new HashMap<>();
map.put("key", 1);
map.getOrDefault("key", 0);
map.containsKey("key");

// HashSet
Set<Integer> set = new HashSet<>();
set.add(1);
set.contains(1);

// PriorityQueue
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

// Sorting
Collections.sort(list);
Collections.sort(list, (a, b) -> a - b);
Arrays.sort(array);

// StringBuilder
StringBuilder sb = new StringBuilder();
sb.append("text");
String result = sb.toString();
```

### C++

```cpp
// Vector
vector<int> vec = {1, 2, 3};
vec.push_back(4);
vec.pop_back();

// Unordered map
unordered_map<string, int> map;
map["key"] = 1;
map.count("key");

// Set
set<int> s = {1, 2, 3};
s.insert(4);
s.erase(1);

// Priority queue
priority_queue<int> maxHeap;
priority_queue<int, vector<int>, greater<int>> minHeap;

// Sorting
sort(vec.begin(), vec.end());
sort(vec.begin(), vec.end(), greater<int>());
sort(vec.begin(), vec.end(), [](int a, int b) { return a > b; });

// Lambda
auto sum = [](int a, int b) { return a + b; };
```

## Interview Strategy

### Before the Interview

1. **Practice coding problems** (LeetCode, HackerRank)
2. **Review data structures and algorithms**
3. **Prepare questions for interviewer**
4. **Research the company**
5. **Test your setup** (camera, mic, internet)

### During the Interview

1. **Clarify the problem**
   - Ask questions
   - Verify assumptions
   - Discuss constraints

2. **Think out loud**
   - Explain your thought process
   - Discuss trade-offs
   - Mention alternative approaches

3. **Start with brute force**
   - State the obvious solution
   - Analyze complexity
   - Optimize iteratively

4. **Write clean code**
   - Use meaningful variable names
   - Modularize with functions
   - Handle edge cases

5. **Test your solution**
   - Walk through examples
   - Consider edge cases
   - Check for bugs

6. **Optimize**
   - Analyze time/space complexity
   - Discuss improvements
   - Refactor if time permits

### Common Mistakes to Avoid

1. **Jumping into code too quickly**
2. **Not asking clarifying questions**
3. **Poor communication**
4. **Ignoring edge cases**
5. **Not testing the code**
6. **Getting stuck on one approach**
7. **Poor time management**
8. **Not discussing trade-offs**

### Time Management

For a 45-minute coding interview:
- 5 min: Understand problem
- 5-10 min: Plan approach
- 20-25 min: Implement
- 5-10 min: Test and debug
- 5 min: Discussion and questions

## Additional Resources

### Practice Platforms
- LeetCode
- HackerRank
- CodeSignal
- AlgoExpert
- Pramp (mock interviews)

### Books
- "Cracking the Coding Interview" by Gayle Laakmann McDowell
- "Elements of Programming Interviews"
- "Algorithm Design Manual" by Steven Skiena

### Online Courses
- Coursera: Algorithms Specialization
- MIT OpenCourseWare: Introduction to Algorithms
- Educative.io: Grokking the Coding Interview

### YouTube Channels
- NeetCode
- Back To Back SWE
- Tech Dummies
- Errichto (competitive programming)

## Summary

Successful interview preparation requires:

1. **Strong fundamentals** in data structures and algorithms
2. **Pattern recognition** to identify problem types
3. **Practice** with diverse problems
4. **Communication skills** to explain your thinking
5. **Problem-solving approach** using frameworks like UMPIRE
6. **Time management** during interviews
7. **Continuous learning** and improvement

Remember: Interviews are a skill that improves with practice. Don't get discouraged by initial failures. Each interview is a learning opportunity.

Good luck with your interviews!
