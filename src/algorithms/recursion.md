# Recursion

Recursion is a programming technique where a function calls itself in order to solve a problem. It is often used to break down complex problems into simpler subproblems.

## Key Concepts

- **Base Case**: The condition under which the recursion ends. It prevents infinite loops and allows the function to return a result.

- **Recursive Case**: The part of the function where the recursion occurs, typically involving a call to the same function with modified arguments.

## Factorial

The factorial of a non-negative integer n is the product of all positive integers less than or equal to n.

```python
def factorial(n):
    # Base case
    if n == 0 or n == 1:
        return 1
    # Recursive case
    return n * factorial(n - 1)

# Example usage
print(factorial(5))  # Output: 120
```

**Complexity Analysis:**
- **Time Complexity:** O(n) - Makes n recursive calls
- **Space Complexity:** O(n) - Call stack depth is n

## Fibonacci Sequence

The Fibonacci sequence where each number is the sum of the two preceding ones.

```python
# Simple recursion (exponential time)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# With memoization (linear time)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Example usage
print(fibonacci(10))       # Output: 55
print(fibonacci_memo(100)) # Much faster for large n
```

**Complexity Analysis:**

*Simple Recursion:*
- **Time Complexity:** O(2^n) - Each call spawns two more calls, creating exponential growth
- **Space Complexity:** O(n) - Maximum depth of recursion tree

*With Memoization:*
- **Time Complexity:** O(n) - Each value computed once and cached
- **Space Complexity:** O(n) - Cache storage + call stack

## Binary Search

Recursive implementation of binary search.

```python
def binary_search(arr, target, left, right):
    # Base case: element not found
    if left > right:
        return -1

    mid = left + (right - left) // 2

    # Base case: element found
    if arr[mid] == target:
        return mid

    # Recursive cases
    if arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13]
result = binary_search(arr, 7, 0, len(arr) - 1)
print(f"Element found at index: {result}")  # Output: 3
```

**Complexity Analysis:**
- **Time Complexity:** O(log n) - Search space halves with each recursive call
- **Space Complexity:** O(log n) - Depth of recursion is logarithmic

## Sum of Array

Calculate the sum of all elements in an array recursively.

```python
def array_sum(arr):
    # Base case: empty array
    if not arr:
        return 0
    # Recursive case: first element + sum of rest
    return arr[0] + array_sum(arr[1:])

# Optimized with index
def array_sum_optimized(arr, index=0):
    if index == len(arr):
        return 0
    return arr[index] + array_sum_optimized(arr, index + 1)

# Example usage
numbers = [1, 2, 3, 4, 5]
print(array_sum(numbers))  # Output: 15
```

**Complexity Analysis:**

*Array Slicing Version:*
- **Time Complexity:** O(n²) - Array slicing creates copies at each level
- **Space Complexity:** O(n²) - Due to array copies + call stack

*Index Version:*
- **Time Complexity:** O(n) - Single pass through array
- **Space Complexity:** O(n) - Call stack only

## Power Function

Calculate x raised to the power n.

```python
# Simple recursion
def power(x, n):
    if n == 0:
        return 1
    return x * power(x, n - 1)

# Optimized (divide and conquer)
def power_optimized(x, n):
    if n == 0:
        return 1

    half = power_optimized(x, n // 2)

    if n % 2 == 0:
        return half * half
    else:
        return x * half * half

# Example usage
print(power(2, 10))           # Output: 1024
print(power_optimized(2, 10)) # Faster for large n
```

**Complexity Analysis:**

*Simple Recursion:*
- **Time Complexity:** O(n) - Linear recursive calls
- **Space Complexity:** O(n) - Call stack depth

*Optimized (Exponentiation by Squaring):*
- **Time Complexity:** O(log n) - Halves the problem size each time
- **Space Complexity:** O(log n) - Call stack depth is logarithmic

## String Reversal

Reverse a string using recursion.

```python
def reverse_string(s):
    # Base case: empty or single character
    if len(s) <= 1:
        return s
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])

# Alternative implementation
def reverse_string_alt(s):
    if len(s) == 0:
        return s
    return reverse_string_alt(s[1:]) + s[0]

# Example usage
print(reverse_string("hello"))  # Output: "olleh"
```

**Complexity Analysis:**
- **Time Complexity:** O(n²) - String slicing and concatenation create copies
- **Space Complexity:** O(n²) - New strings created at each level + call stack

## Palindrome Check

Check if a string is a palindrome recursively.

```python
def is_palindrome(s, left=0, right=None):
    if right is None:
        right = len(s) - 1

    # Base cases
    if left >= right:
        return True

    if s[left] != s[right]:
        return False

    # Recursive case
    return is_palindrome(s, left + 1, right - 1)

# Example usage
print(is_palindrome("racecar"))  # Output: True
print(is_palindrome("hello"))    # Output: False
```

**Complexity Analysis:**
- **Time Complexity:** O(n) - Checks at most n/2 character pairs
- **Space Complexity:** O(n) - Call stack depth is n/2

## Tree Traversals

Recursive tree traversal algorithms.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Inorder traversal (left, root, right)
def inorder(root):
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# Preorder traversal (root, left, right)
def preorder(root):
    if root is None:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# Postorder traversal (left, right, root)
def postorder(root):
    if root is None:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Tree height
def tree_height(root):
    if root is None:
        return 0
    return 1 + max(tree_height(root.left), tree_height(root.right))

# Example usage
#       1
#      / \
#     2   3
#    / \
#   4   5
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print("Inorder:", inorder(root))    # [4, 2, 5, 1, 3]
print("Preorder:", preorder(root))  # [1, 2, 4, 5, 3]
print("Postorder:", postorder(root))# [4, 5, 2, 3, 1]
print("Height:", tree_height(root)) # 3
```

**Complexity Analysis:**

*All Traversals (Inorder, Preorder, Postorder):*
- **Time Complexity:** O(n) - Visits each node exactly once
- **Space Complexity:** O(h) - Call stack depth equals tree height (O(log n) for balanced, O(n) for skewed)

*Tree Height:*
- **Time Complexity:** O(n) - Must visit all nodes
- **Space Complexity:** O(h) - Recursion depth equals tree height

## Greatest Common Divisor (GCD)

Find GCD using Euclidean algorithm.

```python
def gcd(a, b):
    # Base case
    if b == 0:
        return a
    # Recursive case
    return gcd(b, a % b)

# Example usage
print(gcd(48, 18))  # Output: 6
```

**Complexity Analysis:**
- **Time Complexity:** O(log min(a, b)) - Fibonacci sequence determines worst case
- **Space Complexity:** O(log min(a, b)) - Call stack depth

## Tower of Hanoi

Classic puzzle solved recursively.

```python
def tower_of_hanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return

    # Move n-1 disks from source to auxiliary
    tower_of_hanoi(n - 1, source, auxiliary, destination)

    # Move nth disk from source to destination
    print(f"Move disk {n} from {source} to {destination}")

    # Move n-1 disks from auxiliary to destination
    tower_of_hanoi(n - 1, auxiliary, destination, source)

# Example usage
tower_of_hanoi(3, 'A', 'C', 'B')
```

**Complexity Analysis:**
- **Time Complexity:** O(2^n) - Makes 2^n - 1 moves
- **Space Complexity:** O(n) - Recursion depth equals number of disks

## Flatten Nested List

Flatten a nested list structure.

```python
def flatten(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Example usage
nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(flatten(nested))  # Output: [1, 2, 3, 4, 5, 6, 7, 8]
```

**Complexity Analysis:**
- **Time Complexity:** O(n) - Where n is total number of elements (including nested)
- **Space Complexity:** O(d) - Where d is maximum nesting depth

## Permutations

Generate all permutations of a list.

```python
def permutations(arr):
    # Base case: single element or empty
    if len(arr) <= 1:
        return [arr]

    result = []
    for i in range(len(arr)):
        # Fix first element
        current = arr[i]
        # Get remaining elements
        remaining = arr[:i] + arr[i+1:]
        # Generate permutations of remaining
        for perm in permutations(remaining):
            result.append([current] + perm)

    return result

# Alternative using backtracking
def permute_backtrack(arr):
    result = []

    def backtrack(start):
        if start == len(arr):
            result.append(arr[:])
            return

        for i in range(start, len(arr)):
            # Swap
            arr[start], arr[i] = arr[i], arr[start]
            # Recurse
            backtrack(start + 1)
            # Backtrack (undo swap)
            arr[start], arr[i] = arr[i], arr[start]

    backtrack(0)
    return result

# Example usage
print(permutations([1, 2, 3]))
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

**Complexity Analysis:**
- **Time Complexity:** O(n! × n) - n! permutations, each taking O(n) to construct
- **Space Complexity:** O(n! × n) - Storing all permutations

## Combinations

Generate all k-sized combinations from n elements.

```python
def combinations(arr, k):
    # Base cases
    if k == 0:
        return [[]]
    if len(arr) == 0:
        return []

    # Include first element
    with_first = [[arr[0]] + combo for combo in combinations(arr[1:], k-1)]
    # Exclude first element
    without_first = combinations(arr[1:], k)

    return with_first + without_first

# Alternative implementation
def combine(n, k):
    result = []

    def backtrack(start, current):
        if len(current) == k:
            result.append(current[:])
            return

        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result

# Example usage
print(combinations([1, 2, 3, 4], 2))
# Output: [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
```

**Complexity Analysis:**
- **Time Complexity:** O(C(n,k) × k) - C(n,k) combinations, each taking O(k) to construct
- **Space Complexity:** O(C(n,k) × k) - Storing all combinations

## Subsets (Power Set)

Generate all possible subsets of a set.

```python
def subsets(arr):
    # Base case
    if len(arr) == 0:
        return [[]]

    # Recursive case
    first = arr[0]
    rest_subsets = subsets(arr[1:])

    # Subsets without first element + subsets with first element
    with_first = [[first] + subset for subset in rest_subsets]

    return rest_subsets + with_first

# Alternative using backtracking
def subsets_backtrack(arr):
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(arr)):
            current.append(arr[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result

# Example usage
print(subsets([1, 2, 3]))
# Output: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
```

**Complexity Analysis:**
- **Time Complexity:** O(2^n × n) - 2^n subsets, each taking O(n) to construct
- **Space Complexity:** O(2^n × n) - Storing all subsets

## Letter Combinations of Phone Number

Generate all possible letter combinations for a phone number.

```python
def letter_combinations(digits):
    if not digits:
        return []

    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    def backtrack(index, current):
        # Base case: complete combination
        if index == len(digits):
            result.append(current)
            return

        # Get letters for current digit
        letters = phone_map[digits[index]]

        # Try each letter
        for letter in letters:
            backtrack(index + 1, current + letter)

    result = []
    backtrack(0, "")
    return result

# Example usage
print(letter_combinations("23"))
# Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

**Complexity Analysis:**
- **Time Complexity:** O(4^n × n) - Up to 4 letters per digit, n digits
- **Space Complexity:** O(n) - Recursion depth

## Generate Parentheses

Generate all combinations of well-formed parentheses.

```python
def generate_parentheses(n):
    result = []

    def backtrack(current, open_count, close_count):
        # Base case: valid combination complete
        if len(current) == 2 * n:
            result.append(current)
            return

        # Add opening parenthesis if we can
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        # Add closing parenthesis if valid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack("", 0, 0)
    return result

# Example usage
print(generate_parentheses(3))
# Output: ['((()))', '(()())', '(())()', '()(())', '()()()']
```

**Complexity Analysis:**
- **Time Complexity:** O(4^n / √n) - Catalan number
- **Space Complexity:** O(n) - Recursion depth

## Graph Traversal (DFS)

Depth-First Search using recursion.

```python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    # Mark current node as visited
    visited.add(node)
    print(node, end=' ')

    # Recurse for all adjacent nodes
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

    return visited

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

dfs_recursive(graph, 'A')  # Output: A B D E F C
```

**Complexity Analysis:**
- **Time Complexity:** O(V + E) - Visits each vertex and edge once
- **Space Complexity:** O(V) - Recursion stack and visited set

## Find All Paths in Graph

Find all paths between two nodes.

```python
def find_all_paths(graph, start, end, path=None):
    if path is None:
        path = []

    path = path + [start]

    # Base case: reached destination
    if start == end:
        return [path]

    # Base case: no such node
    if start not in graph:
        return []

    paths = []
    for node in graph[start]:
        if node not in path:  # Avoid cycles
            new_paths = find_all_paths(graph, node, end, path)
            paths.extend(new_paths)

    return paths

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(find_all_paths(graph, 'A', 'F'))
# Output: [['A', 'B', 'E', 'F'], ['A', 'C', 'F']]
```

**Complexity Analysis:**
- **Time Complexity:** O(V!) - In worst case (complete graph), explores all permutations
- **Space Complexity:** O(V) - Path length and recursion depth

## Has Path (Graph Connectivity)

Check if a path exists between two nodes.

```python
def has_path(graph, start, end, visited=None):
    if visited is None:
        visited = set()

    # Base case: reached destination
    if start == end:
        return True

    # Mark as visited
    visited.add(start)

    # Check all neighbors
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            if has_path(graph, neighbor, end, visited):
                return True

    return False

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['E'],
    'D': ['F'],
    'E': [],
    'F': []
}

print(has_path(graph, 'A', 'F'))  # True
print(has_path(graph, 'A', 'E'))  # True
print(has_path(graph, 'D', 'E'))  # False
```

**Complexity Analysis:**
- **Time Complexity:** O(V + E) - May visit all vertices and edges
- **Space Complexity:** O(V) - Visited set and recursion stack

## Count Islands (Grid DFS)

Count number of islands in a 2D grid.

```python
def count_islands(grid):
    if not grid:
        return 0

    def dfs(i, j):
        # Base cases: out of bounds or water
        if (i < 0 or i >= len(grid) or
            j < 0 or j >= len(grid[0]) or
            grid[i][j] == '0'):
            return

        # Mark as visited by changing to water
        grid[i][j] = '0'

        # Explore all 4 directions
        dfs(i + 1, j)  # down
        dfs(i - 1, j)  # up
        dfs(i, j + 1)  # right
        dfs(i, j - 1)  # left

    islands = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                islands += 1
                dfs(i, j)

    return islands

# Example usage
grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1']
]

print(count_islands(grid))  # Output: 3
```

**Complexity Analysis:**
- **Time Complexity:** O(m × n) - Visit each cell at most once
- **Space Complexity:** O(m × n) - Worst case recursion depth for a grid filled with land

## Divide and Conquer Algorithms

### Merge Sort

Efficiently sort an array using divide and conquer.

```python
def merge_sort(arr):
    # Base case: array of size 1 or 0
    if len(arr) <= 1:
        return arr

    # Divide: split array in half
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Conquer: merge sorted halves
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    # Merge two sorted arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # Output: [3, 9, 10, 27, 38, 43, 82]
```

**Complexity Analysis:**
- **Time Complexity:** O(n log n) - Divides array log n times, merges in O(n) at each level
- **Space Complexity:** O(n) - Temporary arrays for merging

### Quick Sort

Sort using pivot-based partitioning.

```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    # Base case
    if low < high:
        # Partition and get pivot index
        pivot_index = partition(arr, low, high)

        # Recursively sort elements before and after partition
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

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

# Example usage
arr = [10, 7, 8, 9, 1, 5]
print(quick_sort(arr))  # Output: [1, 5, 7, 8, 9, 10]
```

**Complexity Analysis:**
- **Time Complexity:** O(n log n) average, O(n²) worst case
- **Space Complexity:** O(log n) - Recursion stack for balanced partitions

### Maximum Subarray (Divide and Conquer)

Find maximum sum subarray using divide and conquer.

```python
def max_subarray(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    # Base case: single element
    if low == high:
        return arr[low]

    # Find middle point
    mid = (low + high) // 2

    # Maximum in left half
    left_max = max_subarray(arr, low, mid)

    # Maximum in right half
    right_max = max_subarray(arr, mid + 1, high)

    # Maximum crossing the middle
    cross_max = max_crossing_sum(arr, low, mid, high)

    # Return maximum of three
    return max(left_max, right_max, cross_max)

def max_crossing_sum(arr, low, mid, high):
    # Include elements on left of mid
    left_sum = float('-inf')
    current_sum = 0
    for i in range(mid, low - 1, -1):
        current_sum += arr[i]
        left_sum = max(left_sum, current_sum)

    # Include elements on right of mid
    right_sum = float('-inf')
    current_sum = 0
    for i in range(mid + 1, high + 1):
        current_sum += arr[i]
        right_sum = max(right_sum, current_sum)

    return left_sum + right_sum

# Example usage
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray(arr))  # Output: 6 (subarray [4, -1, 2, 1])
```

**Complexity Analysis:**
- **Time Complexity:** O(n log n) - Divides log n times, linear work at each level
- **Space Complexity:** O(log n) - Recursion stack depth

### Count Inversions

Count pairs where arr[i] > arr[j] and i < j.

```python
def count_inversions(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])

    merged, split_inv = merge_count(left, right)

    total_inv = left_inv + right_inv + split_inv
    return merged, total_inv

def merge_count(left, right):
    result = []
    inversions = 0
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            # All remaining elements in left are inversions
            inversions += len(left) - i
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result, inversions

# Example usage
arr = [8, 4, 2, 1]
sorted_arr, inv_count = count_inversions(arr)
print(f"Inversions: {inv_count}")  # Output: 6
```

**Complexity Analysis:**
- **Time Complexity:** O(n log n) - Similar to merge sort
- **Space Complexity:** O(n) - Temporary arrays for merging

## Backtracking Algorithms

### N-Queens Problem

Place N queens on an N×N chessboard so no two queens attack each other.

```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check upper-right diagonal
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
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'  # Backtrack

    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0)
    return result

# Example usage
solutions = solve_n_queens(4)
for solution in solutions:
    for row in solution:
        print(row)
    print()
```

**Complexity Analysis:**
- **Time Complexity:** O(n!) - Explores all possible placements
- **Space Complexity:** O(n²) - Board storage + recursion stack

### Sudoku Solver

Solve a 9×9 Sudoku puzzle using backtracking.

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if backtrack():
                                return True
                            board[row][col] = '.'  # Backtrack
                    return False
        return True

    backtrack()
    return board

# Example usage (partial board)
board = [
    ['5','3','.','.','7','.','.','.','.'],
    ['6','.','.','1','9','5','.','.','.'],
    ['.','9','8','.','.','.','.','6','.'],
    ['8','.','.','.','6','.','.','.','3'],
    ['4','.','.','8','.','3','.','.','1'],
    ['7','.','.','.','2','.','.','.','6'],
    ['.','6','.','.','.','.','2','8','.'],
    ['.','.','.','4','1','9','.','.','5'],
    ['.','.','.','.','8','.','.','7','9']
]
solve_sudoku(board)
```

**Complexity Analysis:**
- **Time Complexity:** O(9^m) - Where m is number of empty cells
- **Space Complexity:** O(m) - Recursion depth

### Word Search

Find if a word exists in a 2D grid.

```python
def word_search(board, word):
    def backtrack(row, col, index):
        # Base case: found the word
        if index == len(word):
            return True

        # Out of bounds or wrong character
        if (row < 0 or row >= len(board) or
            col < 0 or col >= len(board[0]) or
            board[row][col] != word[index]):
            return False

        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'

        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                 backtrack(row - 1, col, index + 1) or
                 backtrack(row, col + 1, index + 1) or
                 backtrack(row, col - 1, index + 1))

        # Backtrack: restore cell
        board[row][col] = temp

        return found

    for i in range(len(board)):
        for j in range(len(board[0])):
            if backtrack(i, j, 0):
                return True

    return False

# Example usage
board = [
    ['A','B','C','E'],
    ['S','F','C','S'],
    ['A','D','E','E']
]
print(word_search(board, "ABCCED"))  # True
print(word_search(board, "SEE"))     # True
print(word_search(board, "ABCB"))    # False
```

**Complexity Analysis:**
- **Time Complexity:** O(m × n × 4^L) - m×n starting points, 4 directions, L word length
- **Space Complexity:** O(L) - Recursion depth equals word length

### Rat in a Maze

Find path from top-left to bottom-right in a maze.

```python
def rat_in_maze(maze):
    n = len(maze)
    solution = [[0 for _ in range(n)] for _ in range(n)]

    def is_safe(x, y):
        return (0 <= x < n and 0 <= y < n and maze[x][y] == 1)

    def backtrack(x, y):
        # Base case: reached destination
        if x == n - 1 and y == n - 1:
            solution[x][y] = 1
            return True

        if is_safe(x, y):
            # Mark as part of solution
            solution[x][y] = 1

            # Move right
            if backtrack(x, y + 1):
                return True

            # Move down
            if backtrack(x + 1, y):
                return True

            # Backtrack: not part of solution
            solution[x][y] = 0
            return False

        return False

    if backtrack(0, 0):
        return solution
    return None

# Example usage
maze = [
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
]
result = rat_in_maze(maze)
if result:
    for row in result:
        print(row)
```

**Complexity Analysis:**
- **Time Complexity:** O(2^(n²)) - Worst case explores all paths
- **Space Complexity:** O(n²) - Solution matrix + recursion stack

### Subset Sum

Find if there's a subset with a given sum.

```python
def subset_sum(arr, target):
    def backtrack(index, current_sum):
        # Base case: found target sum
        if current_sum == target:
            return True

        # Base case: exceeded target or no more elements
        if current_sum > target or index >= len(arr):
            return False

        # Include current element
        if backtrack(index + 1, current_sum + arr[index]):
            return True

        # Exclude current element
        if backtrack(index + 1, current_sum):
            return True

        return False

    return backtrack(0, 0)

# Example usage
arr = [3, 34, 4, 12, 5, 2]
print(subset_sum(arr, 9))   # True (4 + 5)
print(subset_sum(arr, 30))  # False
```

**Complexity Analysis:**
- **Time Complexity:** O(2^n) - Explores all subsets
- **Space Complexity:** O(n) - Recursion depth

## Recursive Patterns

Common recursive patterns to recognize:

### 1. Linear Recursion
```python
def linear_recursion(n):
    if n == 0:
        return 0
    return n + linear_recursion(n - 1)
```

### 2. Binary Recursion
```python
def binary_recursion(n):
    if n <= 1:
        return n
    return binary_recursion(n - 1) + binary_recursion(n - 2)
```

### 3. Tail Recursion
```python
def tail_recursion(n, accumulator=0):
    if n == 0:
        return accumulator
    return tail_recursion(n - 1, accumulator + n)
```

## Recursion vs Iteration

```python
# Recursive factorial
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# Iterative factorial
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

## Tips for Recursion

1. **Always define a base case**: Prevents infinite recursion
2. **Make progress toward base case**: Each recursive call should move closer to the base case
3. **Trust the recursion**: Assume the recursive call works correctly for smaller inputs
4. **Consider stack depth**: Deep recursion can cause stack overflow
5. **Use memoization**: Cache results to avoid redundant calculations
6. **Know when to use iteration**: Sometimes iteration is clearer and more efficient

## Common Pitfalls

```python
# BAD: No base case (infinite recursion)
def bad_recursion(n):
    return 1 + bad_recursion(n - 1)  # Never stops!

# BAD: Doesn't make progress
def bad_recursion2(n):
    if n == 0:
        return 0
    return bad_recursion2(n)  # n never changes!

# GOOD: Proper base case and progress
def good_recursion(n):
    if n == 0:
        return 0
    return 1 + good_recursion(n - 1)  # n decreases
```

## Recursion Depth and Optimization

### Understanding Stack Depth

Every recursive call adds a frame to the call stack. Deep recursion can cause stack overflow.

```python
import sys

# Check current recursion limit
print(sys.getrecursionlimit())  # Default: 1000 (Python)

# Increase limit (use with caution)
sys.setrecursionlimit(5000)

# Example: deep recursion
def deep_recursion(n):
    if n == 0:
        return 0
    return 1 + deep_recursion(n - 1)

# This will fail with default limit
try:
    print(deep_recursion(2000))
except RecursionError:
    print("Stack overflow!")
```

### Tail Call Optimization

A tail call is when the recursive call is the last operation in the function.

```python
# NOT tail recursive (multiplication happens after recursive call)
def factorial_not_tail(n):
    if n <= 1:
        return 1
    return n * factorial_not_tail(n - 1)

# Tail recursive version with accumulator
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)

# Note: Python doesn't optimize tail calls, but other languages do
```

### Converting Recursion to Iteration

Many recursive algorithms can be rewritten iteratively to save stack space.

```python
# Recursive Fibonacci - exponential time, linear space
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# Iterative Fibonacci - linear time, constant space
def fib_iterative(n):
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr

# Using explicit stack (simulates recursion)
def tree_traversal_iterative(root):
    if not root:
        return []

    stack = [root]
    result = []

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Add children to stack (right first for left-first traversal)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result
```

### Advanced Memoization Techniques

```python
# Using functools.lru_cache decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

# Manual memoization with decorator
def memoize(func):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper

@memoize
def expensive_recursive_function(n):
    if n <= 1:
        return 1
    return expensive_recursive_function(n - 1) + expensive_recursive_function(n - 2)

# Bottom-up dynamic programming (no recursion)
def fib_dp(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
```

### Optimizing Recursive Algorithms

```python
# Bad: Creating new lists in each call
def sum_bad(arr):
    if not arr:
        return 0
    return arr[0] + sum_bad(arr[1:])  # O(n²) time, O(n²) space

# Good: Using indices
def sum_good(arr, index=0):
    if index >= len(arr):
        return 0
    return arr[index] + sum_good(arr, index + 1)  # O(n) time, O(n) space

# Better: Tail recursive with accumulator
def sum_tail(arr, index=0, acc=0):
    if index >= len(arr):
        return acc
    return sum_tail(arr, index + 1, acc + arr[index])  # O(n) time, O(n) space

# Best: Iterative
def sum_iterative(arr):
    total = 0
    for num in arr:
        total += num
    return total  # O(n) time, O(1) space
```

### Handling Deep Recursion

```python
# Technique 1: Increase recursion limit
import sys
sys.setrecursionlimit(10000)

# Technique 2: Use iteration with explicit stack
def dfs_with_stack(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)
            stack.extend(reversed(graph[node]))

    return visited

# Technique 3: Trampolining (advanced)
class Trampoline:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

def trampoline_factorial(n, acc=1):
    if n <= 1:
        return acc
    return Trampoline(trampoline_factorial, n - 1, n * acc)

def execute_trampoline(bouncer):
    while isinstance(bouncer, Trampoline):
        bouncer = bouncer.func(*bouncer.args)
    return bouncer
```

### When to Use Recursion vs Iteration

**Use Recursion When:**
- Problem has natural recursive structure (trees, graphs)
- Code clarity is more important than performance
- Working with recursive data structures
- Implementing divide and conquer algorithms
- Backtracking is required

**Use Iteration When:**
- Performance is critical
- Risk of stack overflow
- Simple sequential processing
- Space complexity must be minimized
- Tail recursive but language doesn't optimize it

```python
# Example: Tree problems are naturally recursive
def tree_depth(node):
    if not node:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))

# Example: Simple loops are better iterative
def count_to_n(n):
    # Don't do this recursively!
    for i in range(1, n + 1):
        print(i)
```

## Applications

Recursion is widely used in various applications across computer science:

### Data Structures
- **Tree Operations**: Traversals (inorder, preorder, postorder), height calculation, searching
- **Graph Algorithms**: DFS, path finding, cycle detection, connectivity checks
- **Linked Lists**: Reversing, searching, merging

### Algorithm Paradigms
- **Divide and Conquer**: Merge sort, quick sort, binary search, maximum subarray
- **Backtracking**: N-Queens, Sudoku solver, maze solving, word search, subset generation
- **Dynamic Programming**: Fibonacci with memoization, longest common subsequence, knapsack problems

### Combinatorics
- **Permutations and Combinations**: Generating all arrangements and selections
- **Power Sets**: Generating all subsets of a set
- **Catalan Numbers**: Valid parentheses combinations, binary tree structures

### String Processing
- **Pattern Matching**: Regular expressions, wildcard matching
- **Palindrome Checking**: Recursive character comparison
- **String Transformations**: Reversals, subsequence generation

### Mathematical Problems
- **Number Theory**: GCD, factorial, exponentiation by squaring
- **Sequences**: Fibonacci, Lucas numbers, recurrence relations
- **Tower of Hanoi**: Classic recursive puzzle

### Real-World Applications
- **Compilers and Interpreters**: Parsing expressions, syntax tree traversal
- **File Systems**: Directory traversal, file searching
- **AI and Game Development**: Minimax algorithm, game tree exploration
- **Computer Graphics**: Fractal generation, ray tracing
- **Network Routing**: Finding paths in network topologies

## Performance Considerations

### Memory Usage
- Each recursive call consumes stack space
- Default stack limits vary by language (Python: ~1000 frames)
- Deep recursion risks stack overflow

### Time Complexity
- Naive recursion can be exponential (e.g., naive Fibonacci)
- Memoization converts exponential to linear in many cases
- Iterative versions often have better constant factors

### Best Practices
1. **Always define clear base cases** to prevent infinite recursion
2. **Ensure progress toward base case** in each recursive call
3. **Use memoization** for overlapping subproblems
4. **Consider iterative alternatives** for deep recursion
5. **Profile your code** to identify performance bottlenecks
6. **Document recursive logic** clearly for maintainability

## Conclusion

Recursion is a fundamental programming technique that enables elegant solutions to problems with recursive structure. While it excels at expressing algorithms for trees, graphs, and divide-and-conquer approaches, it requires careful consideration of:

- **Stack depth limitations** and potential for stack overflow
- **Performance trade-offs** between clarity and efficiency
- **Optimization opportunities** through memoization and tail recursion
- **When to use iteration** instead of recursion

Mastering recursion involves understanding both its power and its limitations. The key is recognizing when recursive thinking simplifies problem-solving and when iterative approaches are more practical. With practice, you'll develop intuition for choosing the right tool for each problem.

Remember: The best recursive solution is one that is clear, correct, and doesn't blow the stack!
