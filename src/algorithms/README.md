# Algorithms

## Overview

An algorithm is a step-by-step procedure or formula for solving a problem. In computer science, algorithms are fundamental to writing efficient and effective code. Understanding algorithms helps you choose the right approach for solving computational problems and optimize performance.

## What is an Algorithm?

An algorithm must have these characteristics:

1. **Input**: Zero or more inputs
2. **Output**: At least one output
3. **Definiteness**: Clear and unambiguous steps
4. **Finiteness**: Must terminate after a finite number of steps
5. **Effectiveness**: Steps must be basic enough to be executed

## Algorithm Analysis

### Time Complexity

Time complexity measures how the runtime of an algorithm grows with input size.

```python
# O(1) - Constant time
def get_first_element(arr):
    return arr[0] if arr else None

# O(n) - Linear time
def find_element(arr, target):
    for elem in arr:
        if elem == target:
            return True
    return False

# O(n²) - Quadratic time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# O(log n) - Logarithmic time
def binary_search(arr, target):
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

# O(n log n) - Linearithmic time
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

### Space Complexity

Space complexity measures the amount of memory an algorithm uses.

```python
# O(1) space - In-place
def reverse_array_inplace(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# O(n) space - Additional array
def reverse_array_new(arr):
    return arr[::-1]

# O(n) space - Recursion stack
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

## Algorithm Categories

### 1. Sorting Algorithms

Transform data into a specific order (ascending/descending).

**Common Sorting Algorithms:**
- Bubble Sort - O(n²)
- Selection Sort - O(n²)
- Insertion Sort - O(n²)
- Merge Sort - O(n log n)
- Quick Sort - O(n log n) average
- Heap Sort - O(n log n)

See: [Sorting Algorithms](sorting.md)

### 2. Searching Algorithms

Find specific elements in data structures.

**Common Searching Algorithms:**
- Linear Search - O(n)
- Binary Search - O(log n)
- Jump Search - O(√n)
- Interpolation Search - O(log log n) average

See: [Searching Algorithms](searching.md)

### 3. Graph Algorithms

Solve problems related to graph structures.

**Common Graph Algorithms:**
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Dijkstra's Algorithm
- Bellman-Ford Algorithm
- Floyd-Warshall Algorithm
- Kruskal's Algorithm
- Prim's Algorithm

See: [Graph Algorithms](../data_structures/graphs.md)

### 4. Tree Algorithms

Operations on tree data structures.

**Common Tree Algorithms:**
- Tree Traversals (Inorder, Preorder, Postorder, Level-order)
- Binary Search Tree Operations
- AVL Tree Balancing
- Red-Black Tree Operations
- Trie Operations

See: [Tree Algorithms](../data_structures/trees.md)

### 5. Dynamic Programming

Break complex problems into simpler subproblems and store results.

**Classic DP Problems:**
- Fibonacci Sequence
- Longest Common Subsequence
- Knapsack Problem
- Matrix Chain Multiplication
- Edit Distance

See: [Dynamic Programming](dynamic_programming.md)

### 6. Greedy Algorithms

Make locally optimal choices at each step.

**Common Greedy Problems:**
- Activity Selection
- Huffman Coding
- Fractional Knapsack
- Coin Change (greedy variant)
- Job Sequencing

See: [Greedy Algorithms](greedy_algorithms.md)

### 7. Divide and Conquer

Divide problem into subproblems, solve recursively, combine results.

**Examples:**
- Merge Sort
- Quick Sort
- Binary Search
- Strassen's Matrix Multiplication
- Closest Pair of Points

See: [Divide and Conquer](divide_and_conquer.md)

### 8. Backtracking

Try all possibilities and backtrack when stuck.

**Classic Problems:**
- N-Queens Problem
- Sudoku Solver
- Permutations and Combinations
- Graph Coloring
- Hamiltonian Path

See: [Backtracking](backtracking.md)

### 9. Recursion

Function calls itself to solve problems.

**Examples:**
- Factorial
- Fibonacci
- Tower of Hanoi
- Tree Traversals
- Divide and Conquer algorithms

See: [Recursion](recursion.md)

### 10. String Algorithms

Pattern matching and string manipulation techniques.

**Common String Algorithms:**
- KMP Pattern Matching
- Rabin-Karp Algorithm
- Z Algorithm
- Manacher's Algorithm
- Trie Operations

See: [String Algorithms](string_algorithms.md)

### 11. Bit Manipulation

Efficient operations using bitwise operators.

**Common Techniques:**
- XOR properties for finding unique elements
- Counting set bits
- Power of 2 checks
- Bit masking for sets
- Gray code generation

See: [Bit Manipulation](bit_manipulation.md)

## Common Algorithm Patterns

### Two Pointers

Use two pointers to traverse data structures efficiently. Reduces O(n²) to O(n) for many problems.

**Common patterns:**
- Opposite direction (two sum in sorted array)
- Same direction fast/slow (remove duplicates, cycle detection)
- Multiple arrays (merge sorted arrays)

```python
# Find pair with given sum in sorted array
def find_pair_with_sum(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return (arr[left], arr[right])
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return None

# Remove duplicates from sorted array
def remove_duplicates(arr):
    if not arr:
        return 0

    write_index = 1
    for read_index in range(1, len(arr)):
        if arr[read_index] != arr[read_index - 1]:
            arr[write_index] = arr[read_index]
            write_index += 1

    return write_index
```

See: [Two Pointers](two_pointers.md) for comprehensive guide

### Sliding Window

Maintain a window of elements that slides through array/string. Optimizes from O(n²) to O(n).

**Types:**
- Fixed-size window (max sum of k elements)
- Variable-size window (longest substring with k distinct chars)

```python
# Maximum sum subarray of size k
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return None

    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Longest substring without repeating characters
def longest_unique_substring(s):
    char_index = {}
    max_length = 0
    start = 0

    for end in range(len(s)):
        if s[end] in char_index and char_index[s[end]] >= start:
            start = char_index[s[end]] + 1

        char_index[s[end]] = end
        max_length = max(max_length, end - start + 1)

    return max_length
```

See: [Sliding Window](sliding_window.md) for comprehensive guide

### Fast and Slow Pointers

```python
# Detect cycle in linked list
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

# Find middle of linked list
def find_middle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

### Merge Intervals

```python
# Merge overlapping intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlapping - merge
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # Non-overlapping - add
            merged.append(current)
    
    return merged
```

### Binary Search Patterns

Efficient O(log n) search on sorted data or monotonic search spaces.

**Common patterns:**
- Find exact match in sorted array
- Find first/last occurrence
- Search in rotated sorted array
- Binary search on answer (find minimum/maximum value)
- Peak finding

```python
# Find first occurrence
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# Find peak element
def find_peak(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return left
```

See: [Binary Search Patterns](binary_search_patterns.md) for comprehensive guide

### Union-Find (Disjoint Set)

Track and merge disjoint sets with near-constant time operations.

**Use cases:**
- Connected components in graphs
- Cycle detection in undirected graphs
- Kruskal's MST algorithm
- Dynamic connectivity problems

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
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True
```

See: [Union-Find](union_find.md) for comprehensive guide

## Problem-Solving Approach

### 1. Understand the Problem
- Read carefully
- Identify inputs and outputs
- Clarify constraints
- Consider edge cases

### 2. Plan Your Approach
- Think of similar problems
- Consider multiple solutions
- Analyze time/space complexity
- Choose appropriate data structures

### 3. Implement
- Write clean, readable code
- Use meaningful variable names
- Add comments for complex logic
- Handle edge cases

### 4. Test
- Test with sample inputs
- Test edge cases (empty, single element, large input)
- Test boundary conditions
- Verify correctness

### 5. Optimize
- Analyze bottlenecks
- Consider trade-offs
- Improve time/space complexity
- Refactor for clarity

## Time Complexity Cheat Sheet

| Complexity | Name | Example |
|------------|------|---------|
| O(1) | Constant | Array access, hash table lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Linear search, array traversal |
| O(n log n) | Linearithmic | Merge sort, quick sort (average) |
| O(n²) | Quadratic | Bubble sort, nested loops |
| O(n³) | Cubic | Triple nested loops |
| O(2ⁿ) | Exponential | Recursive fibonacci |
| O(n!) | Factorial | Permutations |

## Space Complexity Considerations

1. **In-place algorithms**: O(1) space - modify input directly
2. **Recursion**: O(n) space for call stack
3. **Memoization**: Trade space for time
4. **Auxiliary data structures**: Arrays, hash tables, etc.

## Interview Tips

### Common Algorithm Questions

1. **Arrays and Strings**
   - Two Sum
   - Reverse String
   - Longest Substring
   - Array Rotation

2. **Linked Lists**
   - Reverse Linked List
   - Detect Cycle
   - Merge Two Lists
   - Find Middle

3. **Trees and Graphs**
   - Tree Traversals
   - Validate BST
   - Lowest Common Ancestor
   - Graph BFS/DFS

4. **Dynamic Programming**
   - Fibonacci
   - Climbing Stairs
   - Coin Change
   - Longest Increasing Subsequence

5. **Sorting and Searching**
   - Binary Search variants
   - Merge K Sorted Lists
   - Find Kth Largest
   - Quick Select

### Best Practices

1. **Communication**: Think aloud
2. **Clarification**: Ask questions
3. **Examples**: Work through examples
4. **Optimization**: Discuss trade-offs
5. **Testing**: Verify with test cases
6. **Edge Cases**: Consider all scenarios
7. **Clean Code**: Write readable code
8. **Time Management**: Don't get stuck

## Practice Resources

### Online Platforms
- LeetCode
- HackerRank
- CodeSignal
- Project Euler
- Codeforces
- AtCoder

### Books
- "Introduction to Algorithms" (CLRS)
- "Algorithm Design Manual" (Skiena)
- "Cracking the Coding Interview"
- "Elements of Programming Interviews"

## Available Topics

Explore detailed guides for specific algorithm types:

### Fundamentals
1. [Big O Notation](big_o.md) - Understanding algorithm complexity
2. [Recursion](recursion.md) - Recursive problem solving

### Sorting & Searching
3. [Sorting Algorithms](sorting.md) - Comprehensive sorting guide
4. [Searching Algorithms](searching.md) - Various search techniques
5. [Binary Search Patterns](binary_search_patterns.md) - Advanced binary search techniques

### Algorithm Paradigms
6. [Dynamic Programming](dynamic_programming.md) - DP patterns and problems
7. [Greedy Algorithms](greedy_algorithms.md) - Greedy approach and examples
8. [Divide and Conquer](divide_and_conquer.md) - Divide and conquer strategy
9. [Backtracking](backtracking.md) - Backtracking techniques

### Common Patterns & Techniques
10. [Two Pointers](two_pointers.md) - Two pointer technique patterns
11. [Sliding Window](sliding_window.md) - Sliding window optimization
12. [Bit Manipulation](bit_manipulation.md) - Bitwise operations and tricks
13. [Union-Find](union_find.md) - Disjoint set union data structure

### Graph & Tree Algorithms
14. [Graph Algorithms](graph_algorithms.md) - Graph traversal and algorithms
15. [Tree Algorithms](../data_structures/trees.md) - Tree operations and traversals

### String & Specialized
16. [String Algorithms](string_algorithms.md) - Pattern matching and string manipulation
17. [Heaps](../data_structures/heaps.md) - Heap data structure and algorithms
18. [Tries](../data_structures/tries.md) - Trie data structure and applications
19. [Raft Consensus](raft.md) - Distributed consensus algorithm

## Quick Reference

### Most Important Algorithms to Know

**Sorting:**
- Quick Sort
- Merge Sort
- Heap Sort

**Searching:**
- Binary Search
- Depth-First Search (DFS)
- Breadth-First Search (BFS)

**Graph:**
- Dijkstra's Algorithm
- Topological Sort
- Union-Find

**Dynamic Programming:**
- 0/1 Knapsack
- Longest Common Subsequence
- Edit Distance

**String:**
- KMP Pattern Matching
- Rabin-Karp
- Trie Operations

## Next Steps

1. Review [Big O Notation](big_o.md) for complexity analysis
2. Practice with [Sorting](sorting.md) and [Searching](searching.md)
3. Master [Recursion](recursion.md) fundamentals
4. Explore [Dynamic Programming](dynamic_programming.md)
5. Study [Graph Algorithms](../data_structures/graphs.md) and [Trees](../data_structures/trees.md)
6. Practice on coding platforms
7. Participate in coding contests
8. Review and optimize solutions

Remember: The key to mastering algorithms is consistent practice and understanding the underlying patterns. Start with fundamentals and gradually tackle more complex problems.
