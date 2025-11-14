# Stacks

## Overview

A stack is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end, called the top. Think of it like a stack of dinner plates - you put plates on top and take them from the top.

## Key Concepts

### LIFO Principle
```
Push: 1 -> 2 -> 3

     3     Top (Last In)

     2

     1     First In


Pop: Returns 3 (First Out)
```

## Operations & Time Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Push | $O(1)$ | $O(n)$ |
| Pop | $O(1)$ | - |
| Peek | $O(1)$ | - |
| Is Empty | $O(1)$ | - |
| Search | $O(n)$ | - |

## Implementation (Python)

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, data):
        self.items.append(data)

    def pop(self):
        return self.items.pop() if not self.is_empty() else None

    def peek(self):
        return self.items[-1] if not self.is_empty() else None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Using deque for $O(1)$ operations
from collections import deque
stack = deque()
stack.append(1)  # Push $O(1)$
stack.pop()      # Pop $O(1)$
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

### Postfix Evaluation
```python
def evaluate_postfix(expr):
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

## Real-World Uses

- **Browser Back Button**: Last visited page is first to go back to
- **Function Call Stack**: Each function call pushed, returns pop
- **Undo/Redo**: Last action undone first
- **Expression Parsing**: Manage operator precedence
- **DFS (Depth-First Search)**: Graph traversal

## ELI10

Imagine a stack of dinner plates - you:
- Add new plates on **top**
- Take plates from **top**
- Can't grab from the middle without removing top plates

That's LIFO! Last In = First Out. The last plate you put on is the first one you take off.

## Further Resources

- [LeetCode Stack Problems](https://leetcode.com/tag/stack/)
- [Stack Wikipedia](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))
