# Linked Lists

## Overview

A linked list is a linear data structure where elements (nodes) are connected via pointers/references rather than stored in contiguous memory. Each node contains data and a reference to the next node, creating a chain-like structure.

## Key Concepts

### Structure

```
    Head
     |
    Data -> Next -> Data -> Next -> Data -> Next -> None
```

### Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Singly Linked List** | Each node points to next node only | Standard, memory efficient |
| **Doubly Linked List** | Each node points to next and previous | Need bidirectional traversal |
| **Circular Linked List** | Last node points back to first | Round-robin scheduling |

### Advantages vs Arrays

| Feature | Linked List | Array |
|---------|------------|-------|
| **Access** | $O(n)$ | $O(1)$ |
| **Insert/Delete at start** | $O(1)$ | $O(n)$ |
| **Insert/Delete in middle** | $O(n)$ to find, $O(1)$ to insert | $O(n)$ |
| **Memory** | Flexible, dynamic | Fixed or expensive to resize |
| **Cache Efficiency** | Poor | Excellent |

## Implementation

### Python - Singly Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """Add element to end"""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data):
        """Add element to beginning"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_after(self, prev_data, data):
        """Insert after specific value"""
        current = self.head
        while current and current.data != prev_data:
            current = current.next

        if current:
            new_node = Node(data)
            new_node.next = current.next
            current.next = new_node

    def delete(self, data):
        """Remove first occurrence"""
        if not self.head:
            return

        # If head needs to be deleted
        if self.head.data == data:
            self.head = self.head.next
            return

        current = self.head
        while current.next and current.next.data != data:
            current = current.next

        if current.next:
            current.next = current.next.next

    def search(self, data):
        """Find element"""
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def display(self):
        """Print all elements"""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements) + " -> None")

    def __len__(self):
        """Get length"""
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)
ll.display()  # 0 -> 1 -> 2 -> 3 -> None
ll.delete(2)
ll.display()  # 0 -> 1 -> 3 -> None
```

### Python - Doubly Linked List

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """Add to end"""
        new_node = DNode(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def reverse_display(self):
        """Print in reverse"""
        if not self.head:
            return

        current = self.head
        while current.next:
            current = current.next

        elements = []
        while current:
            elements.append(str(current.data))
            current = current.prev
        print(" -> ".join(elements) + " -> None")
```

### JavaScript

```javascript
class Node {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class LinkedList {
    constructor() {
        this.head = null;
    }

    append(data) {
        const newNode = new Node(data);
        if (!this.head) {
            this.head = newNode;
            return;
        }

        let current = this.head;
        while (current.next) {
            current = current.next;
        }
        current.next = newNode;
    }

    prepend(data) {
        const newNode = new Node(data);
        newNode.next = this.head;
        this.head = newNode;
    }

    delete(data) {
        if (!this.head) return;

        if (this.head.data === data) {
            this.head = this.head.next;
            return;
        }

        let current = this.head;
        while (current.next && current.next.data !== data) {
            current = current.next;
        }

        if (current.next) {
            current.next = current.next.next;
        }
    }

    display() {
        let current = this.head;
        let result = [];
        while (current) {
            result.push(current.data);
            current = current.next;
        }
        console.log(result.join(" -> ") + " -> null");
    }
}

// Usage
const ll = new LinkedList();
ll.append(1);
ll.append(2);
ll.prepend(0);
ll.display();  // 0 -> 1 -> 2 -> null
```

### C++

```cpp
#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;

    Node(int data) : data(data), next(nullptr) {}
};

class LinkedList {
private:
    Node* head;

public:
    LinkedList() : head(nullptr) {}

    void append(int data) {
        Node* newNode = new Node(data);
        if (!head) {
            head = newNode;
            return;
        }

        Node* current = head;
        while (current->next) {
            current = current->next;
        }
        current->next = newNode;
    }

    void prepend(int data) {
        Node* newNode = new Node(data);
        newNode->next = head;
        head = newNode;
    }

    void deleteNode(int data) {
        if (!head) return;

        if (head->data == data) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return;
        }

        Node* current = head;
        while (current->next && current->next->data != data) {
            current = current->next;
        }

        if (current->next) {
            Node* temp = current->next;
            current->next = current->next->next;
            delete temp;
        }
    }

    void display() {
        Node* current = head;
        while (current) {
            cout << current->data << " -> ";
            current = current->next;
        }
        cout << "null\n";
    }

    ~LinkedList() {
        Node* current = head;
        while (current) {
            Node* temp = current;
            current = current->next;
            delete temp;
        }
    }
};
```

## Common Problems

### Reverse a Linked List

```python
def reverse(head):
    """Reverse entire linked list"""
    prev = None
    current = head

    while current:
        next_temp = current.next  # Save next
        current.next = prev       # Reverse link
        prev = current            # Move prev forward
        current = next_temp       # Move current forward

    return prev  # New head
```

### Find Middle

```python
def find_middle(head):
    """Find middle node using slow/fast pointers"""
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # Slow pointer at middle
```

### Detect Cycle

```python
def has_cycle(head):
    """Detect if linked list has cycle"""
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:  # Cycle detected
            return True

    return False
```

### Merge Two Sorted Lists

```python
def merge_sorted(l1, l2):
    """Merge two sorted linked lists"""
    dummy = Node(0)
    current = dummy

    while l1 and l2:
        if l1.data < l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # Attach remaining
    current.next = l1 if l1 else l2

    return dummy.next
```

### Remove Nth Node from End

```python
def remove_nth_from_end(head, n):
    """Remove nth node from end"""
    dummy = Node(0)
    dummy.next = head
    first = second = dummy

    # Move first pointer n+1 steps ahead
    for i in range(n + 1):
        if not first:
            return head
        first = first.next

    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next

    # Remove node
    second.next = second.next.next

    return dummy.next
```

## Time Complexity Summary

| Operation | Singly | Doubly |
|-----------|--------|--------|
| **Access** | $O(n)$ | $O(n)$ |
| **Search** | $O(n)$ | $O(n)$ |
| **Insert at head** | $O(1)$ | $O(1)$ |
| **Insert at tail** | $O(n)$ | $O(1)$* |
| **Delete from head** | $O(1)$ | $O(1)$ |
| **Delete from tail** | $O(n)$ | $O(1)$* |
| **Reverse** | $O(n)$ | $O(n)$ |

*With tail pointer

## Best Practices

### 1. Use Sentinel Nodes

```python
# Bad: Check for None multiple times
if head and head.next and head.next.next:
    ...

# Good: Use dummy node
dummy = Node(0)
dummy.next = head
current = dummy
# Now no need to check if current exists
```

### 2. Avoid Memory Leaks (C++)

```cpp
// Always delete removed nodes
Node* temp = current->next;
current->next = current->next->next;
delete temp;  // Free memory
```

### 3. Two-Pointer Technique

```python
# Many problems solved with slow/fast pointers:
# - Find middle
# - Detect cycle
# - Remove nth from end
```

## ELI10

Imagine a treasure hunt with clues:

- Each clue card (node) has treasure info and points to the next clue
- You start at the first clue (head)
- To find a specific clue, you must follow the chain - you can't jump!
- To add a clue in the middle, you just change what one card points to
- You don't need a big board to write all clues - they can be anywhere!

The tricky part: You can only look at clues in order, you can't jump to the middle one directly like you could with an array.

## Further Resources

- [Linked List Wikipedia](https://en.wikipedia.org/wiki/Linked_list)
- [LeetCode Linked List Problems](https://leetcode.com/tag/linked-list/)
- [Visual Algo - Linked List Visualizations](https://visualgo.net/en/list)
