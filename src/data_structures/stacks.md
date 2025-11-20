# Stacks

## Overview

A stack is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end, called the top. Think of it like a stack of dinner plates - you put plates on top and take them from the top.

### When to Use Stacks

- **Reversing**: Need to process elements in reverse order
- **Backtracking**: Undo operations, browser history, text editor undo
- **Parsing**: Expression evaluation, syntax checking, matching brackets
- **Function Calls**: Call stack in programming languages
- **DFS Traversal**: Depth-first search in graphs and trees
- **Maintaining State**: When you need to remember previous states

## Key Concepts

### LIFO Principle

Last-In-First-Out means the most recently added element is the first one to be removed.

**Step-by-Step Example:**
```
Initial: Empty Stack
         []

Push(1): Add 1 to top
         [1]  <- Top

Push(2): Add 2 to top
         [2]  <- Top
         [1]

Push(3): Add 3 to top
         [3]  <- Top (Last In)
         [2]
         [1]  <- Bottom (First In)

Pop(): Remove and return 3
       Returns: 3
         [2]  <- Top
         [1]

Pop(): Remove and return 2
       Returns: 2
         [1]  <- Top

Peek(): View top without removing
        Returns: 1
         [1]  <- Top (unchanged)
```

## Operations & Time Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Push | $O(1)$ | $O(n)$ |
| Pop | $O(1)$ | - |
| Peek | $O(1)$ | - |
| Is Empty | $O(1)$ | - |
| Search | $O(n)$ | - |

## Implementation

### Python Implementation

```python
from typing import Optional, List
from collections import deque

class Stack:
    """
    Stack implementation using Python list.
    All operations are O(1) amortized time.
    """

    def __init__(self):
        self.items: List = []

    def push(self, data) -> None:
        """Add element to top. Time: O(1) amortized"""
        self.items.append(data)

    def pop(self) -> Optional[int]:
        """Remove and return top element. Time: O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self) -> Optional[int]:
        """Return top element without removing. Time: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self) -> bool:
        """Check if stack is empty. Time: O(1)"""
        return len(self.items) == 0

    def size(self) -> int:
        """Return number of elements. Time: O(1)"""
        return len(self.items)

# Using deque for guaranteed O(1) operations
stack = deque()
stack.append(1)    # Push O(1)
stack.pop()        # Pop O(1)
```

### Java Implementation

```java
import java.util.EmptyStackException;
import java.util.ArrayList;
import java.util.Stack; // Built-in Stack class

/**
 * Generic Stack implementation using ArrayList
 * All operations are O(1) amortized time
 */
public class StackImpl<T> {
    private ArrayList<T> items;

    public StackImpl() {
        items = new ArrayList<>();
    }

    /**
     * Add element to top
     * Time: O(1) amortized
     */
    public void push(T data) {
        items.add(data);
    }

    /**
     * Remove and return top element
     * Time: O(1)
     * @throws EmptyStackException if stack is empty
     */
    public T pop() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return items.remove(items.size() - 1);
    }

    /**
     * Return top element without removing
     * Time: O(1)
     */
    public T peek() {
        if (isEmpty()) {
            throw new EmptyStackException();
        }
        return items.get(items.size() - 1);
    }

    /**
     * Check if stack is empty
     * Time: O(1)
     */
    public boolean isEmpty() {
        return items.isEmpty();
    }

    /**
     * Return number of elements
     * Time: O(1)
     */
    public int size() {
        return items.size();
    }
}

// Using built-in Stack class
Stack<Integer> stack = new Stack<>();
stack.push(1);     // Push O(1)
stack.pop();       // Pop O(1)
stack.peek();      // Peek O(1)
```

### Linked List Implementation (Python)

```python
class Node:
    """Node for linked list stack"""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    """
    Stack using linked list
    O(1) for all operations, no amortization
    Better for frequent push/pop, worse for memory
    """

    def __init__(self):
        self.head: Optional[Node] = None
        self._size: int = 0

    def push(self, data) -> None:
        """Time: O(1), Space: O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def pop(self):
        """Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        data = self.head.data
        self.head = self.head.next
        self._size -= 1
        return data

    def peek(self):
        """Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.head.data

    def is_empty(self) -> bool:
        """Time: O(1), Space: O(1)"""
        return self.head is None

    def size(self) -> int:
        """Time: O(1), Space: O(1)"""
        return self._size
```

## Common Problems

### Valid Parentheses
```python
def is_valid(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}

    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False

    return len(stack) == 0
```

### Next Greater Element
```python
def next_greater(arr):
    stack = []
    result = [-1] * len(arr)

    for i in range(len(arr) - 1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(arr[i])

    return result
```

### Infix to Postfix Conversion
Convert infix expressions (like `3 + 4 * 2`) to postfix notation (like `3 4 2 * +`).

```python
def infix_to_postfix(expression):
    """
    Convert infix to postfix using operator precedence
    Example: "3 + 4 * 2" -> "3 4 2 * +"
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    output = []
    stack = []

    for token in expression.split():
        if token.isdigit():
            # Operand: add to output
            output.append(token)
        elif token == '(':
            # Left paren: push to stack
            stack.append(token)
        elif token == ')':
            # Right paren: pop until matching '('
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        else:
            # Operator: pop higher/equal precedence operators
            while (stack and stack[-1] != '(' and
                   stack[-1] in precedence and
                   precedence[stack[-1]] >= precedence[token]):
                output.append(stack.pop())
            stack.append(token)

    # Pop remaining operators
    while stack:
        output.append(stack.pop())

    return ' '.join(output)

# Example: "( 3 + 4 ) * 2" -> "3 4 + 2 *"
```

### Postfix Evaluation
```python
def evaluate_postfix(expr):
    """
    Evaluate postfix expression
    Example: "3 4 2 * +" -> 11
    """
    stack = []
    ops = {'+', '-', '*', '/'}

    for token in expr.split():
        if token not in ops:
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            else: stack.append(a // b)

    return stack[0]
```

## Monotonic Stacks

A monotonic stack maintains elements in either increasing or decreasing order. Used for finding next/previous greater/smaller elements efficiently.

### Monotonic Decreasing Stack
Elements decrease from bottom to top. Used to find next greater element.

```python
def next_greater_element(nums):
    """Find next greater element for each number"""
    n = len(nums)
    result = [-1] * n
    stack = []  # Stores indices

    for i in range(n):
        # Pop smaller elements - they found their next greater
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

# Example: [2, 1, 2, 4, 3] -> [4, 2, 4, -1, -1]
```

### Monotonic Increasing Stack
Elements increase from bottom to top. Used to find next smaller element.

```python
def next_smaller_element(nums):
    """Find next smaller element for each number"""
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        # Pop larger elements - they found their next smaller
        while stack and nums[stack[-1]] > nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

# Example: [4, 2, 1, 5, 3] -> [2, 1, -1, 3, -1]
```

### Common Monotonic Stack Problems
- **Next Greater Element**: Use decreasing stack
- **Next Smaller Element**: Use increasing stack
- **Largest Rectangle in Histogram**: Use increasing stack
- **Trapping Rain Water**: Use stacks to track water boundaries
- **Daily Temperatures**: Find next warmer day

## Advanced Stack Problems

### Daily Temperatures
Find how many days until a warmer temperature.

```python
def daily_temperatures(temps):
    """
    Given [73,74,75,71,69,72,76,73]
    Return [1,1,4,2,1,1,0,0]
    Time: O(n), Space: O(n)
    """
    n = len(temps)
    result = [0] * n
    stack = []  # Stores indices

    for i in range(n):
        # Pop all colder days - they found a warmer day
        while stack and temps[stack[-1]] < temps[i]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)

    return result
```

### Largest Rectangle in Histogram
Find the largest rectangle area in a histogram.

```python
def largest_rectangle(heights):
    """
    Given heights [2,1,5,6,2,3]
    Return 10 (rectangle from index 2-3 with height 5)
    Time: O(n), Space: O(n)
    """
    stack = []  # Monotonic increasing stack
    max_area = 0
    heights.append(0)  # Add sentinel

    for i, h in enumerate(heights):
        # Pop higher bars and calculate area
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            height = heights[height_idx]
            # Width: from after previous bar to before current
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    heights.pop()  # Remove sentinel
    return max_area
```

### Trapping Rain Water
Calculate water trapped between bars after raining.

```python
def trap_rain_water(height):
    """
    Given [0,1,0,2,1,0,1,3,2,1,2,1]
    Return 6 units of water
    Time: O(n), Space: O(n)
    """
    stack = []
    water = 0

    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            # Water bounded by left wall (stack[-1]) and right wall (i)
            width = i - stack[-1] - 1
            bounded_height = min(height[stack[-1]], h) - height[bottom]
            water += width * bounded_height
        stack.append(i)

    return water
```

### Basic Calculator
Implement a calculator for expressions with +, -, and parentheses.

```python
def calculate(s):
    """
    Evaluate "1 + (2 - (3 + 4))" = -4
    Time: O(n), Space: O(n)
    """
    stack = []
    num = 0
    sign = 1  # 1 for +, -1 for -
    result = 0

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            # Push current result and sign to stack
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            # Pop sign and previous result
            result *= stack.pop()  # Sign before '('
            result += stack.pop()  # Result before '('

    return result + sign * num
```

### Decode String
Decode string with pattern "k[encoded_string]".

```python
def decode_string(s):
    """
    "3[a2[c]]" -> "accaccacc"
    Time: O(n), Space: O(n)
    """
    stack = []
    current_str = ""
    current_num = 0

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append(current_str)
            stack.append(current_num)
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
            num = stack.pop()
            prev_str = stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char

    return current_str
```

### Asteroid Collision
Asteroids moving left (-) and right (+) collide.

```python
def asteroid_collision(asteroids):
    """
    [5, 10, -5] -> [5, 10] (10 destroys -5)
    [8, -8] -> [] (both destroy each other)
    Time: O(n), Space: O(n)
    """
    stack = []

    for ast in asteroids:
        while stack and ast < 0 < stack[-1]:
            # Collision: positive moving right meets negative moving left
            if stack[-1] < -ast:
                stack.pop()  # Right asteroid destroyed
                continue
            elif stack[-1] == -ast:
                stack.pop()  # Both destroyed
            break  # Left asteroid destroyed or both destroyed
        else:
            stack.append(ast)  # No collision

    return stack
```

## Min Stack (Design Problem)

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

### Approach 1: Two Stacks
```python
class MinStack:
    """Stack with O(1) getMin() using auxiliary stack"""

    def __init__(self):
        self.stack = []      # Main stack
        self.min_stack = []  # Tracks minimum values

    def push(self, val: int) -> None:
        """Time: O(1), Space: O(1)"""
        self.stack.append(val)
        # Push current min to min_stack
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        """Time: O(1), Space: O(1)"""
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        """Time: O(1), Space: O(1)"""
        return self.stack[-1]

    def getMin(self) -> int:
        """Time: O(1), Space: O(1)"""
        return self.min_stack[-1]
```

### Approach 2: Single Stack with Tuples
```python
class MinStack:
    """Space-optimized: store (value, current_min) pairs"""

    def __init__(self):
        self.stack = []  # Stores (val, min_so_far) tuples

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

### Key Insight
Both approaches maintain the invariant: **at any point, we can access the minimum of all elements currently in the stack in O(1) time**.

## Stack vs Recursion

Every recursive algorithm can be converted to an iterative one using an explicit stack. The call stack IS a stack!

### Recursive Tree Traversal
```python
def inorder_recursive(root):
    """Recursive inorder traversal"""
    if not root:
        return
    inorder_recursive(root.left)    # Implicit stack: push left
    print(root.val)                  # Process
    inorder_recursive(root.right)   # Implicit stack: push right
```

### Iterative with Explicit Stack
```python
def inorder_iterative(root):
    """Iterative inorder using explicit stack"""
    stack = []
    current = root

    while current or stack:
        # Go left as far as possible
        while current:
            stack.append(current)
            current = current.left

        # Process node
        current = stack.pop()
        print(current.val)

        # Move to right subtree
        current = current.right
```

### When to Use Each

| Recursive | Iterative + Stack |
|-----------|-------------------|
| Cleaner, more readable | More control over execution |
| Risk of stack overflow | Explicit memory management |
| Hidden complexity | Visible complexity |
| Good for small inputs | Better for large inputs |

### Converting Recursion to Stack
1. Create an explicit stack
2. Push initial state onto stack
3. Loop while stack is not empty:
   - Pop from stack
   - Process current state
   - Push next states onto stack

## Common Stack Patterns

Recognizing these patterns helps identify when to use a stack:

### 1. Matching/Balancing Pattern
**Use Case**: Parentheses, brackets, tags, quotes
**Key**: Push opening symbols, pop when closing found
**Examples**: Valid Parentheses, HTML/XML validation

```python
# Template
stack = []
pairs = {'(': ')', '[': ']', '{': '}'}

for char in string:
    if char in pairs:  # Opening
        stack.append(char)
    else:  # Closing
        if not stack or pairs[stack.pop()] != char:
            return False
return len(stack) == 0
```

### 2. Monotonic Stack Pattern
**Use Case**: Next/Previous greater/smaller element
**Key**: Maintain increasing or decreasing order
**Examples**: Daily Temperatures, Stock Span, Largest Rectangle

```python
# Template for Next Greater
stack = []  # Stores indices
result = [-1] * len(arr)

for i in range(len(arr)):
    while stack and arr[stack[-1]] < arr[i]:
        idx = stack.pop()
        result[idx] = arr[i]
    stack.append(i)
```

### 3. State Saving Pattern
**Use Case**: Backtracking, undo operations, nested contexts
**Key**: Push state before change, pop to restore
**Examples**: Calculator with parentheses, Decode String, Text editor undo

```python
# Template
stack = []

# Before entering new context
stack.append(current_state)
current_state = new_state

# When exiting context
current_state = stack.pop()
```

### 4. Reverse/Backtrack Pattern
**Use Case**: Process in reverse order, path tracking
**Key**: Push during forward pass, pop for reverse
**Examples**: Reverse Polish Notation, DFS path tracking

```python
# Template
stack = []

# Forward pass
for item in items:
    stack.append(process(item))

# Reverse pass
while stack:
    result = combine(result, stack.pop())
```

### 5. Two-Stack Pattern
**Use Case**: Min/Max tracking, queue implementation
**Key**: One stack for data, another for metadata
**Examples**: Min Stack, Implement Queue using Stacks

```python
# Template
main_stack = []
aux_stack = []  # Tracks min/max or other property

def push(val):
    main_stack.append(val)
    # Update auxiliary stack
    if not aux_stack or val <= aux_stack[-1]:
        aux_stack.append(val)
```

### Pattern Recognition Checklist
- ✅ **Need to match pairs?** → Matching Pattern
- ✅ **Find next greater/smaller?** → Monotonic Stack
- ✅ **Nested contexts/scopes?** → State Saving Pattern
- ✅ **Process in reverse?** → Reverse/Backtrack Pattern
- ✅ **Track min/max?** → Two-Stack Pattern
- ✅ **Replace recursion?** → Explicit Stack Pattern

## Real-World Uses

- **Browser Navigation**: Back/forward buttons using two stacks
- **Function Call Stack**: Every program execution uses a stack
- **Undo/Redo Operations**: Text editors, image editors, games
- **Expression Parsing**: Compilers, calculators, formula evaluation
- **DFS Traversal**: Graph algorithms, maze solving, tree traversal
- **Memory Management**: Stack allocation for local variables
- **Backtracking Algorithms**: Chess engines, puzzle solvers, constraint satisfaction
- **Syntax Checking**: IDEs checking matching brackets, code formatters

## Interview Tips

### Recognition
- Keywords: "balanced", "matching", "nested", "nearest", "next greater/smaller"
- Recursion can often be converted to iterative with a stack
- Multiple levels of nesting suggest a stack

### Common Mistakes
- ❌ Forgetting to check if stack is empty before pop/peek
- ❌ Not handling edge cases (empty input, single element)
- ❌ Using index when value is needed (or vice versa)
- ❌ Wrong order in monotonic stack (increasing vs decreasing)

### Optimization Tips
- Use indices instead of values when you need position
- Deque in Python is faster than list for stack operations
- Consider space-time tradeoffs (two stacks vs tuples)
- Monotonic stacks reduce O(n²) to O(n) for many problems

### Time Complexity Analysis
Most stack problems are **O(n)** even with nested loops:
- Each element is pushed once: O(n)
- Each element is popped once: O(n)
- Total: O(n) + O(n) = O(n)

## ELI10

Imagine a stack of dinner plates - you:
- Add new plates on **top**
- Take plates from **top**
- Can't grab from the middle without removing top plates

That's LIFO! Last In = First Out. The last plate you put on is the first one you take off.

## Further Resources

- [LeetCode Stack Problems](https://leetcode.com/tag/stack/)
- [Stack Wikipedia](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))
