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

## Applications

Recursion is widely used in various applications, including:

- **Tree Traversals**: Navigating through tree data structures using recursive methods
- **Backtracking Algorithms**: Solving problems incrementally by trying partial solutions
- **Dynamic Programming**: Many DP problems can be solved using recursive approaches with memoization
- **Divide and Conquer**: Breaking problems into smaller subproblems
- **Mathematical Computations**: Factorials, Fibonacci, GCD, etc.

## Conclusion

Recursion is a powerful tool in programming that allows for elegant solutions to complex problems. Understanding how to effectively use recursion is essential for developing efficient algorithms in computer science and software engineering.
