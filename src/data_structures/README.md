# Data Structures

## Overview

A data structure is a specialized format for organizing, processing, retrieving, and storing data. Different data structures are suited for different kinds of applications, and some are highly specialized for specific tasks. Understanding data structures is fundamental to writing efficient algorithms and building scalable software systems.

## Why Data Structures Matter

1. **Efficiency**: Right data structure can dramatically improve performance
2. **Organization**: Logical way to organize and manage data
3. **Reusability**: Common patterns for solving problems
4. **Abstraction**: Hide implementation details
5. **Optimization**: Trade-offs between time and space complexity

## Classification of Data Structures

### Linear Data Structures
Elements are arranged in sequential order:
- Arrays
- Linked Lists
- Stacks
- Queues

### Non-Linear Data Structures
Elements are arranged hierarchically or in a network:
- Trees
- Graphs
- Tries
- Hash Tables

### Static vs Dynamic
- **Static**: Fixed size (arrays)
- **Dynamic**: Size can change (linked lists, dynamic arrays)

## Core Data Structures

### 1. Arrays

Contiguous memory locations storing elements of the same type.

```python
# Array operations
arr = [1, 2, 3, 4, 5]

# Access - O(1)
element = arr[2]  # 3

# Insert at end - O(1) amortized
arr.append(6)

# Insert at position - O(n)
arr.insert(2, 10)

# Delete - O(n)
arr.remove(10)

# Search - O(n)
if 4 in arr:
    print("Found")

# 2D Array
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Access element
value = matrix[1][2]  # 6
```

**Time Complexity:**
- Access: O(1)
- Search: O(n)
- Insert: O(n)
- Delete: O(n)

**Space Complexity:** O(n)

See: [Arrays](arrays.md)

### 2. Linked Lists

Nodes connected via pointers, allowing efficient insertion/deletion.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, data):
        new_node = Node(data)
        
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def delete(self, key):
        current = self.head
        
        # Delete head
        if current and current.data == key:
            self.head = current.next
            return
        
        # Delete other node
        prev = None
        while current and current.data != key:
            prev = current
            current = current.next
        
        if current:
            prev.next = current.next
    
    def search(self, key):
        current = self.head
        while current:
            if current.data == key:
                return True
            current = current.next
        return False
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements
```

**Types:**
- Singly Linked List
- Doubly Linked List
- Circular Linked List

**Time Complexity:**
- Access: O(n)
- Search: O(n)
- Insert at beginning: O(1)
- Insert at end: O(n) or O(1) with tail pointer
- Delete: O(n)

See: [Linked Lists](linked_lists.md)

### 3. Stacks

LIFO (Last In, First Out) structure.

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top - O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item - O(1)"""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        """Return top item without removing - O(1)"""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)

# Usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())    # 3
print(stack.peek())   # 2

# Applications
def is_balanced(expression):
    """Check if parentheses are balanced"""
    stack = []
    opening = "([{"
    closing = ")]}"
    pairs = {"(": ")", "[": "]", "{": "}"}
    
    for char in expression:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0

# Reverse string using stack
def reverse_string(s):
    stack = list(s)
    return ''.join(stack[::-1])
```

**Applications:**
- Function call stack
- Undo/Redo operations
- Expression evaluation
- Backtracking algorithms
- Browser history

See: [Stacks](stacks.md)

### 4. Queues

FIFO (First In, First Out) structure.

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item - O(1)"""
        if not self.is_empty():
            return self.items.popleft()
        raise IndexError("Queue is empty")
    
    def front(self):
        """Return front item - O(1)"""
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    
    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)

# Priority Queue
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
    
    def push(self, item, priority):
        """Add item with priority - O(log n)"""
        heapq.heappush(self.heap, (priority, item))
    
    def pop(self):
        """Remove and return highest priority item - O(log n)"""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        raise IndexError("Queue is empty")

# Circular Queue
class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = self.rear = -1
    
    def enqueue(self, item):
        if (self.rear + 1) % self.size == self.front:
            raise Exception("Queue is full")
        
        if self.front == -1:
            self.front = 0
        
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = item
    
    def dequeue(self):
        if self.front == -1:
            raise Exception("Queue is empty")
        
        item = self.queue[self.front]
        
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.size
        
        return item
```

**Types:**
- Simple Queue
- Circular Queue
- Priority Queue
- Double-Ended Queue (Deque)

**Applications:**
- Task scheduling
- BFS traversal
- Print queue
- Buffer management
- Async processing

See: [Queues](queues.md)

### 5. Hash Tables

Key-value pairs with O(1) average-case operations.

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Hash function - O(1)"""
        return hash(key) % self.size
    
    def insert(self, key, value):
        """Insert key-value pair - O(1) average"""
        index = self._hash(key)
        
        # Update if key exists
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        
        # Insert new key-value
        self.table[index].append((key, value))
    
    def get(self, key):
        """Get value by key - O(1) average"""
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        """Delete key-value pair - O(1) average"""
        index = self._hash(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                return
        
        raise KeyError(f"Key '{key}' not found")
    
    def contains(self, key):
        """Check if key exists - O(1) average"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False

# Python dict is a hash table
hash_map = {}
hash_map["name"] = "John"
hash_map["age"] = 30

# Counter using hash table
from collections import Counter
text = "hello world"
char_count = Counter(text)
```

**Collision Resolution:**
- Chaining (linked lists)
- Open addressing (linear probing, quadratic probing, double hashing)

**Time Complexity:**
- Average: O(1) for insert, delete, search
- Worst: O(n) with many collisions

See: [Hash Tables](hash_tables.md)

## Advanced Data Structures

### 6. Trees

Hierarchical structure with nodes connected by edges.

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        """Insert value - O(log n) average, O(n) worst"""
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        """Search for value - O(log n) average"""
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None or node.value == value:
            return node
        
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)
    
    def inorder_traversal(self, node, result=None):
        """Inorder: Left -> Root -> Right"""
        if result is None:
            result = []
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.value)
            self.inorder_traversal(node.right, result)
        
        return result
```

**Types:**
- Binary Tree
- Binary Search Tree
- AVL Tree (self-balancing)
- Red-Black Tree
- B-Tree
- Heap

See: [Trees documentation in algorithms](../algorithms/trees.md)

### 7. Graphs

Network of nodes (vertices) connected by edges.

```python
# Adjacency List representation
class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_vertex(self, vertex):
        """Add vertex - O(1)"""
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, v1, v2):
        """Add edge - O(1)"""
        if v1 in self.graph and v2 in self.graph:
            self.graph[v1].append(v2)
            self.graph[v2].append(v1)  # For undirected graph
    
    def bfs(self, start):
        """Breadth-First Search - O(V + E)"""
        visited = set()
        queue = [start]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend(self.graph[vertex])
        
        return result
    
    def dfs(self, start, visited=None):
        """Depth-First Search - O(V + E)"""
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result

# Adjacency Matrix representation
class GraphMatrix:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    def add_edge(self, v1, v2, weight=1):
        """Add edge with optional weight"""
        self.matrix[v1][v2] = weight
        self.matrix[v2][v1] = weight  # For undirected graph
```

**Types:**
- Directed/Undirected
- Weighted/Unweighted
- Cyclic/Acyclic
- Connected/Disconnected

See: [Graph algorithms](../algorithms/graphs.md)

## Choosing the Right Data Structure

### Array vs Linked List

**Use Array when:**
- Need random access
- Size is known and fixed
- Memory is contiguous
- Cache performance matters

**Use Linked List when:**
- Frequent insertions/deletions
- Size is unknown
- Don't need random access
- Memory fragmentation is acceptable

### Stack vs Queue

**Use Stack for:**
- LIFO operations
- Recursion simulation
- Undo/redo functionality
- Expression evaluation

**Use Queue for:**
- FIFO operations
- Scheduling
- BFS traversal
- Resource sharing

### Hash Table vs Tree

**Use Hash Table when:**
- Need O(1) lookup
- Order doesn't matter
- No range queries needed
- Keys are hashable

**Use Tree when:**
- Need sorted order
- Range queries required
- Prefix searches (Trie)
- Hierarchical data

## Performance Comparison

| Operation | Array | Linked List | Stack | Queue | Hash Table | BST |
|-----------|-------|-------------|-------|-------|------------|-----|
| Access | O(1) | O(n) | O(n) | O(n) | - | O(log n) |
| Search | O(n) | O(n) | O(n) | O(n) | O(1)* | O(log n) |
| Insert | O(n) | O(1)** | O(1) | O(1) | O(1)* | O(log n) |
| Delete | O(n) | O(1)** | O(1) | O(1) | O(1)* | O(log n) |

\* Average case, \*\* At beginning/with reference

## Common Operations

### Traversal Patterns

```python
# Array traversal
for i in range(len(arr)):
    process(arr[i])

# Linked list traversal
current = head
while current:
    process(current.data)
    current = current.next

# Tree traversal (recursion)
def traverse_tree(node):
    if node:
        traverse_tree(node.left)
        process(node.value)
        traverse_tree(node.right)

# Graph traversal (BFS)
def bfs(graph, start):
    visited = set()
    queue = [start]
    
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            process(vertex)
            queue.extend(graph[vertex])
```

### Searching Patterns

```python
# Linear search - O(n)
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# Binary search - O(log n)
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

# Hash table search - O(1)
def hash_search(hash_table, key):
    return hash_table.get(key)
```

## Real-World Applications

### Arrays
- Database tables
- Image processing (pixel arrays)
- Dynamic programming tables
- Buffer implementation

### Linked Lists
- Music playlists
- Browser history (doubly linked)
- Undo functionality
- Memory management (free lists)

### Stacks
- Function call management
- Expression evaluation
- Backtracking (maze solving)
- Browser back button

### Queues
- Print spooling
- CPU scheduling
- BFS traversal
- Message queues (async)

### Hash Tables
- Database indexing
- Caching
- Symbol tables (compilers)
- Spell checkers

### Trees
- File systems
- DOM (HTML)
- Decision trees (AI)
- Database indexing (B-trees)

### Graphs
- Social networks
- Maps and navigation
- Network routing
- Recommendation systems

## Interview Preparation

### Essential Topics

1. **Arrays and Strings**
   - Two pointers
   - Sliding window
   - Prefix sums

2. **Linked Lists**
   - Reverse list
   - Detect cycle
   - Merge lists

3. **Stacks and Queues**
   - Valid parentheses
   - Min/max stack
   - Implement queue with stacks

4. **Trees**
   - Traversals
   - Height/depth
   - Lowest common ancestor

5. **Graphs**
   - BFS/DFS
   - Cycle detection
   - Shortest path

6. **Hash Tables**
   - Two sum
   - Group anagrams
   - LRU cache

### Common Patterns

- **Two Pointers**: Array problems
- **Fast/Slow Pointers**: Linked list cycles
- **Sliding Window**: Subarray problems
- **BFS/DFS**: Tree/graph traversal
- **Backtracking**: Combinatorial problems
- **Dynamic Programming**: Optimization problems

## Available Resources

Explore detailed guides for specific data structures:

1. [Arrays](arrays.md) - Array operations and techniques
2. [Linked Lists](linked_lists.md) - Singly, doubly, circular lists
3. [Stacks](stacks.md) - Stack implementation and applications
4. [Queues](queues.md) - Queue types and use cases
5. [Hash Tables](hash_tables.md) - Hashing and collision resolution

Related algorithm topics:
- [Sorting Algorithms](../algorithms/sorting.md)
- [Searching Algorithms](../algorithms/searching.md)
- [Tree Algorithms](../algorithms/trees.md)
- [Graph Algorithms](../algorithms/graphs.md)

## Best Practices

1. **Choose appropriately**: Match data structure to problem
2. **Consider trade-offs**: Time vs space complexity
3. **Test edge cases**: Empty, single element, duplicates
4. **Optimize**: Start simple, then optimize
5. **Document**: Comment complex logic
6. **Practice**: Regular coding practice
7. **Learn patterns**: Recognize common patterns
8. **Understand internals**: Know how they work

## Next Steps

1. Master the fundamental structures (array, linked list, stack, queue)
2. Practice implementing each structure from scratch
3. Solve problems using each data structure
4. Learn when to use each structure
5. Study advanced structures (trees, graphs, tries)
6. Practice on coding platforms (LeetCode, HackerRank)
7. Review time/space complexity for all operations
8. Work on real-world projects using these structures

Remember: Understanding data structures is essential for writing efficient code and succeeding in technical interviews. Focus on understanding the concepts, not just memorizing implementations.
