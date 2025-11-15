# Queues

## Table of Contents
- [Introduction](#introduction)
- [Queue Fundamentals](#queue-fundamentals)
- [Basic Queue Implementation](#basic-queue-implementation)
- [Circular Queue](#circular-queue)
- [Priority Queue](#priority-queue)
- [Deque (Double-Ended Queue)](#deque-double-ended-queue)
- [Thread-Safe Queues](#thread-safe-queues)
- [Real-World Applications](#real-world-applications)
- [Complexity Analysis](#complexity-analysis)
- [Comparison with Other Data Structures](#comparison-with-other-data-structures)
- [Interview Problems and Patterns](#interview-problems-and-patterns)

## Introduction

A queue is a fundamental linear data structure that follows the **First In First Out (FIFO)** principle. This means that the first element added to the queue will be the first one to be removed, much like a line of people waiting for service.

Queues are ubiquitous in computer science and are used in:
- Operating systems for process scheduling
- Network packet management
- Printer job management
- Breadth-first search algorithms
- Asynchronous data handling
- Event-driven programming

## Queue Fundamentals

### FIFO Principle

The FIFO (First In First Out) principle is the defining characteristic of a queue. Elements are:
- **Enqueued** (added) at the **rear** (back/tail) of the queue
- **Dequeued** (removed) from the **front** (head) of the queue

```
Front                                    Rear
  |                                        |
  v                                        v
[10] -> [20] -> [30] -> [40] -> [50]
  ^                                        ^
  |                                        |
Dequeue here                         Enqueue here
```

### Core Operations

1. **Enqueue(item)**: Add an element to the rear of the queue
2. **Dequeue()**: Remove and return the element at the front
3. **Peek() / Front()**: View the front element without removing it
4. **IsEmpty()**: Check if the queue is empty
5. **Size()**: Return the number of elements in the queue
6. **Clear()**: Remove all elements from the queue

### Queue Properties

- **Dynamic Size**: Queues can grow or shrink as elements are added or removed
- **Order Preservation**: Elements maintain their insertion order
- **Access Restriction**: Only the front and rear are accessible (no random access)
- **Memory Efficiency**: Can be implemented using arrays or linked lists

## Basic Queue Implementation

### Python Implementation

#### Using List (Simple but Inefficient for Dequeue)

```python
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """Add item to rear of queue - O(1)"""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item - O(n) due to list shift"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop(0)

    def peek(self):
        """Return front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty queue")
        return self.items[0]

    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0

    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)

    def clear(self):
        """Remove all items - O(1)"""
        self.items = []

    def __str__(self):
        return f"Queue({self.items})"


# Usage example
queue = Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)
print(queue)  # Queue([10, 20, 30])
print(queue.dequeue())  # 10
print(queue.peek())  # 20
print(queue.size())  # 2
```

#### Using collections.deque (Efficient)

```python
from collections import deque

class EfficientQueue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()

    def peek(self):
        """Return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty queue")
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def clear(self):
        self.items.clear()


# Usage
q = EfficientQueue()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q.dequeue())  # 1
print(q.dequeue())  # 2
```

#### Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def enqueue(self, item):
        """Add item to rear - O(1)"""
        new_node = Node(item)

        if self.rear is None:
            # Queue is empty
            self.front = self.rear = new_node
        else:
            # Add to rear and update rear pointer
            self.rear.next = new_node
            self.rear = new_node

        self._size += 1

    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")

        item = self.front.data
        self.front = self.front.next
        self._size -= 1

        # If queue becomes empty, update rear pointer
        if self.front is None:
            self.rear = None

        return item

    def peek(self):
        """Return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty queue")
        return self.front.data

    def is_empty(self):
        return self.front is None

    def size(self):
        return self._size

    def clear(self):
        self.front = None
        self.rear = None
        self._size = 0

    def __str__(self):
        items = []
        current = self.front
        while current:
            items.append(str(current.data))
            current = current.next
        return f"Queue([{' -> '.join(items)}])"


# Usage
lq = LinkedQueue()
lq.enqueue(100)
lq.enqueue(200)
lq.enqueue(300)
print(lq)  # Queue([100 -> 200 -> 300])
print(lq.dequeue())  # 100
print(lq)  # Queue([200 -> 300])
```

### JavaScript/TypeScript Implementation

#### Basic Queue Class

```javascript
class Queue {
    constructor() {
        this.items = [];
    }

    enqueue(item) {
        // Add to rear - O(1)
        this.items.push(item);
    }

    dequeue() {
        // Remove from front - O(n) due to array shift
        if (this.isEmpty()) {
            throw new Error("Dequeue from empty queue");
        }
        return this.items.shift();
    }

    peek() {
        if (this.isEmpty()) {
            throw new Error("Peek from empty queue");
        }
        return this.items[0];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }

    clear() {
        this.items = [];
    }

    toString() {
        return `Queue([${this.items.join(', ')}])`;
    }
}

// Usage
const queue = new Queue();
queue.enqueue(10);
queue.enqueue(20);
queue.enqueue(30);
console.log(queue.toString());  // Queue([10, 20, 30])
console.log(queue.dequeue());  // 10
console.log(queue.peek());  // 20
```

#### Efficient Queue Using Object

```javascript
class EfficientQueue {
    constructor() {
        this.items = {};
        this.frontIndex = 0;
        this.rearIndex = 0;
    }

    enqueue(item) {
        // Add to rear - O(1)
        this.items[this.rearIndex] = item;
        this.rearIndex++;
    }

    dequeue() {
        // Remove from front - O(1)
        if (this.isEmpty()) {
            throw new Error("Dequeue from empty queue");
        }

        const item = this.items[this.frontIndex];
        delete this.items[this.frontIndex];
        this.frontIndex++;

        return item;
    }

    peek() {
        if (this.isEmpty()) {
            throw new Error("Peek from empty queue");
        }
        return this.items[this.frontIndex];
    }

    isEmpty() {
        return this.frontIndex === this.rearIndex;
    }

    size() {
        return this.rearIndex - this.frontIndex;
    }

    clear() {
        this.items = {};
        this.frontIndex = 0;
        this.rearIndex = 0;
    }
}

// Usage
const eq = new EfficientQueue();
eq.enqueue(1);
eq.enqueue(2);
eq.enqueue(3);
console.log(eq.dequeue());  // 1
console.log(eq.size());  // 2
```

#### TypeScript Implementation

```typescript
interface QueueInterface<T> {
    enqueue(item: T): void;
    dequeue(): T;
    peek(): T;
    isEmpty(): boolean;
    size(): number;
    clear(): void;
}

class TypedQueue<T> implements QueueInterface<T> {
    private items: Map<number, T>;
    private frontIndex: number;
    private rearIndex: number;

    constructor() {
        this.items = new Map<number, T>();
        this.frontIndex = 0;
        this.rearIndex = 0;
    }

    enqueue(item: T): void {
        this.items.set(this.rearIndex, item);
        this.rearIndex++;
    }

    dequeue(): T {
        if (this.isEmpty()) {
            throw new Error("Dequeue from empty queue");
        }

        const item = this.items.get(this.frontIndex)!;
        this.items.delete(this.frontIndex);
        this.frontIndex++;

        return item;
    }

    peek(): T {
        if (this.isEmpty()) {
            throw new Error("Peek from empty queue");
        }
        return this.items.get(this.frontIndex)!;
    }

    isEmpty(): boolean {
        return this.frontIndex === this.rearIndex;
    }

    size(): number {
        return this.rearIndex - this.frontIndex;
    }

    clear(): void {
        this.items.clear();
        this.frontIndex = 0;
        this.rearIndex = 0;
    }
}

// Usage
const tq = new TypedQueue<number>();
tq.enqueue(10);
tq.enqueue(20);
console.log(tq.dequeue());  // 10
```

### C++ Implementation

#### Using std::queue

```cpp
#include <iostream>
#include <queue>
#include <string>

void basicQueueExample() {
    std::queue<int> q;

    // Enqueue
    q.push(10);
    q.push(20);
    q.push(30);

    std::cout << "Front: " << q.front() << std::endl;  // 10
    std::cout << "Size: " << q.size() << std::endl;    // 3

    // Dequeue
    q.pop();
    std::cout << "Front after pop: " << q.front() << std::endl;  // 20

    // Check empty
    std::cout << "Is empty: " << (q.empty() ? "Yes" : "No") << std::endl;
}
```

#### Custom Queue Implementation Using Array

```cpp
#include <iostream>
#include <stdexcept>

template<typename T>
class ArrayQueue {
private:
    T* items;
    int capacity;
    int frontIndex;
    int rearIndex;
    int count;

    void resize() {
        int newCapacity = capacity * 2;
        T* newItems = new T[newCapacity];

        // Copy elements
        for (int i = 0; i < count; i++) {
            newItems[i] = items[(frontIndex + i) % capacity];
        }

        delete[] items;
        items = newItems;
        capacity = newCapacity;
        frontIndex = 0;
        rearIndex = count;
    }

public:
    ArrayQueue(int initialCapacity = 10)
        : capacity(initialCapacity), frontIndex(0), rearIndex(0), count(0) {
        items = new T[capacity];
    }

    ~ArrayQueue() {
        delete[] items;
    }

    void enqueue(const T& item) {
        if (count == capacity) {
            resize();
        }

        items[rearIndex] = item;
        rearIndex = (rearIndex + 1) % capacity;
        count++;
    }

    T dequeue() {
        if (isEmpty()) {
            throw std::runtime_error("Dequeue from empty queue");
        }

        T item = items[frontIndex];
        frontIndex = (frontIndex + 1) % capacity;
        count--;

        return item;
    }

    T& peek() {
        if (isEmpty()) {
            throw std::runtime_error("Peek from empty queue");
        }
        return items[frontIndex];
    }

    bool isEmpty() const {
        return count == 0;
    }

    int size() const {
        return count;
    }

    void clear() {
        frontIndex = 0;
        rearIndex = 0;
        count = 0;
    }
};

// Usage
int main() {
    ArrayQueue<int> q;
    q.enqueue(10);
    q.enqueue(20);
    q.enqueue(30);

    std::cout << "Dequeue: " << q.dequeue() << std::endl;  // 10
    std::cout << "Peek: " << q.peek() << std::endl;  // 20
    std::cout << "Size: " << q.size() << std::endl;  // 2

    return 0;
}
```

#### Custom Queue Using Linked List

```cpp
#include <iostream>
#include <stdexcept>

template<typename T>
class LinkedQueue {
private:
    struct Node {
        T data;
        Node* next;

        Node(const T& value) : data(value), next(nullptr) {}
    };

    Node* front;
    Node* rear;
    int count;

public:
    LinkedQueue() : front(nullptr), rear(nullptr), count(0) {}

    ~LinkedQueue() {
        clear();
    }

    void enqueue(const T& item) {
        Node* newNode = new Node(item);

        if (rear == nullptr) {
            // Queue is empty
            front = rear = newNode;
        } else {
            rear->next = newNode;
            rear = newNode;
        }

        count++;
    }

    T dequeue() {
        if (isEmpty()) {
            throw std::runtime_error("Dequeue from empty queue");
        }

        Node* temp = front;
        T item = front->data;

        front = front->next;

        if (front == nullptr) {
            rear = nullptr;
        }

        delete temp;
        count--;

        return item;
    }

    T& peek() {
        if (isEmpty()) {
            throw std::runtime_error("Peek from empty queue");
        }
        return front->data;
    }

    bool isEmpty() const {
        return front == nullptr;
    }

    int size() const {
        return count;
    }

    void clear() {
        while (!isEmpty()) {
            dequeue();
        }
    }
};

// Usage
int main() {
    LinkedQueue<std::string> q;
    q.enqueue("First");
    q.enqueue("Second");
    q.enqueue("Third");

    std::cout << q.dequeue() << std::endl;  // First
    std::cout << q.peek() << std::endl;     // Second

    return 0;
}
```

## Circular Queue

A circular queue is a linear data structure that connects the end position back to the beginning, forming a circle. This design overcomes the limitation of a regular queue where space at the beginning cannot be reused after dequeue operations.

### Advantages of Circular Queue

1. **Efficient Memory Usage**: Reuses freed space after dequeue operations
2. **Fixed Size**: Useful when maximum size is known beforehand
3. **No Shifting Required**: Unlike linear queues, no need to shift elements
4. **Cache-Friendly**: Better locality of reference

### Visual Representation

```
Initial State (capacity = 5):
Front = 0, Rear = 0
[_][_][_][_][_]
 0  1  2  3  4

After enqueue(10, 20, 30):
Front = 0, Rear = 3
[10][20][30][_][_]
 F            R

After dequeue() twice:
Front = 2, Rear = 3
[_][_][30][_][_]
       F   R

After enqueue(40, 50, 60):
Front = 2, Rear = 0 (wrapped around)
[60][_][30][40][50]
 R       F

```

### Python Implementation

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = [None] * capacity
        self.front = -1
        self.rear = -1
        self.count = 0

    def enqueue(self, item):
        """Add item to rear - O(1)"""
        if self.is_full():
            raise OverflowError("Queue is full")

        if self.is_empty():
            self.front = 0
            self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.capacity

        self.items[self.rear] = item
        self.count += 1

    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")

        item = self.items[self.front]

        if self.front == self.rear:
            # Queue becomes empty
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity

        self.count -= 1
        return item

    def peek(self):
        """Return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[self.front]

    def is_empty(self):
        return self.count == 0

    def is_full(self):
        return self.count == self.capacity

    def size(self):
        return self.count

    def __str__(self):
        if self.is_empty():
            return "CircularQueue([])"

        result = []
        i = self.front
        for _ in range(self.count):
            result.append(str(self.items[i]))
            i = (i + 1) % self.capacity

        return f"CircularQueue([{', '.join(result)}])"


# Usage
cq = CircularQueue(5)
cq.enqueue(10)
cq.enqueue(20)
cq.enqueue(30)
print(cq)  # CircularQueue([10, 20, 30])

cq.dequeue()
cq.dequeue()
print(cq)  # CircularQueue([30])

cq.enqueue(40)
cq.enqueue(50)
cq.enqueue(60)
cq.enqueue(70)
print(cq)  # CircularQueue([30, 40, 50, 60, 70])
```

### C++ Implementation

```cpp
#include <iostream>
#include <stdexcept>

template<typename T>
class CircularQueue {
private:
    T* items;
    int capacity;
    int front;
    int rear;
    int count;

public:
    CircularQueue(int cap) : capacity(cap), front(-1), rear(-1), count(0) {
        items = new T[capacity];
    }

    ~CircularQueue() {
        delete[] items;
    }

    void enqueue(const T& item) {
        if (isFull()) {
            throw std::overflow_error("Queue is full");
        }

        if (isEmpty()) {
            front = 0;
            rear = 0;
        } else {
            rear = (rear + 1) % capacity;
        }

        items[rear] = item;
        count++;
    }

    T dequeue() {
        if (isEmpty()) {
            throw std::underflow_error("Queue is empty");
        }

        T item = items[front];

        if (front == rear) {
            // Queue becomes empty
            front = -1;
            rear = -1;
        } else {
            front = (front + 1) % capacity;
        }

        count--;
        return item;
    }

    T& peek() {
        if (isEmpty()) {
            throw std::runtime_error("Queue is empty");
        }
        return items[front];
    }

    bool isEmpty() const {
        return count == 0;
    }

    bool isFull() const {
        return count == capacity;
    }

    int size() const {
        return count;
    }

    void display() const {
        if (isEmpty()) {
            std::cout << "CircularQueue([])" << std::endl;
            return;
        }

        std::cout << "CircularQueue([";
        int i = front;
        for (int c = 0; c < count; c++) {
            std::cout << items[i];
            if (c < count - 1) std::cout << ", ";
            i = (i + 1) % capacity;
        }
        std::cout << "])" << std::endl;
    }
};
```

### Use Cases for Circular Queue

1. **CPU Scheduling**: Round-robin scheduling algorithm
2. **Memory Management**: Buffer management in operating systems
3. **Traffic Systems**: Traffic light control systems
4. **Audio/Video Streaming**: Ring buffers for streaming data
5. **Keyboard Buffers**: Storing keystrokes in a fixed-size buffer

## Priority Queue

A priority queue is an abstract data type where each element has an associated priority. Elements with higher priority are dequeued before elements with lower priority, regardless of insertion order.

### Properties

- Elements are served based on priority, not FIFO order
- Can be implemented using heaps (binary heap most common)
- Supports efficient insertion and removal of the highest priority element
- Can be min-priority (smallest first) or max-priority (largest first)

### Python Implementation Using heapq

```python
import heapq

class PriorityQueue:
    def __init__(self, max_heap=False):
        self.heap = []
        self.max_heap = max_heap
        self.counter = 0  # For tie-breaking (FIFO for same priority)

    def enqueue(self, item, priority):
        """Add item with priority - O(log n)"""
        # For max heap, negate priority
        if self.max_heap:
            priority = -priority

        # Use counter for tie-breaking to maintain FIFO order
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1

    def dequeue(self):
        """Remove and return highest priority item - O(log n)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty priority queue")

        priority, _, item = heapq.heappop(self.heap)
        return item

    def peek(self):
        """Return highest priority item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty priority queue")

        return self.heap[0][2]

    def is_empty(self):
        return len(self.heap) == 0

    def size(self):
        return len(self.heap)

    def clear(self):
        self.heap = []
        self.counter = 0


# Usage - Min Priority Queue (lower value = higher priority)
pq = PriorityQueue()
pq.enqueue("Low priority task", priority=5)
pq.enqueue("High priority task", priority=1)
pq.enqueue("Medium priority task", priority=3)

print(pq.dequeue())  # High priority task (priority=1)
print(pq.dequeue())  # Medium priority task (priority=3)
print(pq.dequeue())  # Low priority task (priority=5)

# Max Priority Queue (higher value = higher priority)
max_pq = PriorityQueue(max_heap=True)
max_pq.enqueue("Task A", priority=10)
max_pq.enqueue("Task B", priority=50)
max_pq.enqueue("Task C", priority=30)

print(max_pq.dequeue())  # Task B (priority=50)
print(max_pq.dequeue())  # Task C (priority=30)
```

### Custom Priority Queue with Heap

```python
class HeapPriorityQueue:
    def __init__(self):
        self.heap = []

    def _parent(self, i):
        return (i - 1) // 2

    def _left_child(self, i):
        return 2 * i + 1

    def _right_child(self, i):
        return 2 * i + 2

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, i):
        """Move element up to maintain heap property"""
        while i > 0 and self.heap[i][0] < self.heap[self._parent(i)][0]:
            parent = self._parent(i)
            self._swap(i, parent)
            i = parent

    def _heapify_down(self, i):
        """Move element down to maintain heap property"""
        min_index = i
        left = self._left_child(i)
        right = self._right_child(i)

        if left < len(self.heap) and self.heap[left][0] < self.heap[min_index][0]:
            min_index = left

        if right < len(self.heap) and self.heap[right][0] < self.heap[min_index][0]:
            min_index = right

        if i != min_index:
            self._swap(i, min_index)
            self._heapify_down(min_index)

    def enqueue(self, item, priority):
        """Add item with priority - O(log n)"""
        self.heap.append((priority, item))
        self._heapify_up(len(self.heap) - 1)

    def dequeue(self):
        """Remove and return min priority item - O(log n)"""
        if self.is_empty():
            raise IndexError("Dequeue from empty priority queue")

        if len(self.heap) == 1:
            return self.heap.pop()[1]

        # Swap root with last element
        self._swap(0, len(self.heap) - 1)
        priority, item = self.heap.pop()

        # Restore heap property
        if len(self.heap) > 0:
            self._heapify_down(0)

        return item

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty priority queue")
        return self.heap[0][1]

    def is_empty(self):
        return len(self.heap) == 0

    def size(self):
        return len(self.heap)


# Usage
hpq = HeapPriorityQueue()
hpq.enqueue("Emergency", 1)
hpq.enqueue("Normal", 5)
hpq.enqueue("Urgent", 2)

print(hpq.dequeue())  # Emergency
print(hpq.dequeue())  # Urgent
print(hpq.dequeue())  # Normal
```

### C++ Implementation

```cpp
#include <iostream>
#include <queue>
#include <vector>
#include <string>

// Using std::priority_queue (max heap by default)
void basicPriorityQueue() {
    // Max heap (largest value has highest priority)
    std::priority_queue<int> maxHeap;

    maxHeap.push(30);
    maxHeap.push(10);
    maxHeap.push(50);
    maxHeap.push(20);

    while (!maxHeap.empty()) {
        std::cout << maxHeap.top() << " ";  // 50 30 20 10
        maxHeap.pop();
    }
    std::cout << std::endl;

    // Min heap (smallest value has highest priority)
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;

    minHeap.push(30);
    minHeap.push(10);
    minHeap.push(50);
    minHeap.push(20);

    while (!minHeap.empty()) {
        std::cout << minHeap.top() << " ";  // 10 20 30 50
        minHeap.pop();
    }
    std::cout << std::endl;
}

// Custom priority queue with objects
struct Task {
    std::string name;
    int priority;

    Task(const std::string& n, int p) : name(n), priority(p) {}

    // Comparison operator for max heap (higher priority value = higher priority)
    bool operator<(const Task& other) const {
        return priority < other.priority;  // Lower priority goes to bottom
    }
};

void customPriorityQueue() {
    std::priority_queue<Task> taskQueue;

    taskQueue.push(Task("Low priority", 1));
    taskQueue.push(Task("High priority", 10));
    taskQueue.push(Task("Medium priority", 5));

    while (!taskQueue.empty()) {
        Task t = taskQueue.top();
        std::cout << t.name << " (priority: " << t.priority << ")" << std::endl;
        taskQueue.pop();
    }
}
```

### JavaScript Implementation

```javascript
class PriorityQueue {
    constructor(comparator = (a, b) => a.priority - b.priority) {
        this.heap = [];
        this.comparator = comparator;
    }

    parent(i) {
        return Math.floor((i - 1) / 2);
    }

    leftChild(i) {
        return 2 * i + 1;
    }

    rightChild(i) {
        return 2 * i + 2;
    }

    swap(i, j) {
        [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
    }

    heapifyUp(i) {
        while (i > 0 && this.comparator(this.heap[i], this.heap[this.parent(i)]) < 0) {
            const parent = this.parent(i);
            this.swap(i, parent);
            i = parent;
        }
    }

    heapifyDown(i) {
        let minIndex = i;
        const left = this.leftChild(i);
        const right = this.rightChild(i);

        if (left < this.heap.length &&
            this.comparator(this.heap[left], this.heap[minIndex]) < 0) {
            minIndex = left;
        }

        if (right < this.heap.length &&
            this.comparator(this.heap[right], this.heap[minIndex]) < 0) {
            minIndex = right;
        }

        if (i !== minIndex) {
            this.swap(i, minIndex);
            this.heapifyDown(minIndex);
        }
    }

    enqueue(item, priority) {
        this.heap.push({ item, priority });
        this.heapifyUp(this.heap.length - 1);
    }

    dequeue() {
        if (this.isEmpty()) {
            throw new Error("Dequeue from empty priority queue");
        }

        if (this.heap.length === 1) {
            return this.heap.pop().item;
        }

        this.swap(0, this.heap.length - 1);
        const item = this.heap.pop().item;
        this.heapifyDown(0);

        return item;
    }

    peek() {
        if (this.isEmpty()) {
            throw new Error("Peek from empty priority queue");
        }
        return this.heap[0].item;
    }

    isEmpty() {
        return this.heap.length === 0;
    }

    size() {
        return this.heap.length;
    }
}

// Usage
const pq = new PriorityQueue();
pq.enqueue("Low priority", 5);
pq.enqueue("High priority", 1);
pq.enqueue("Medium priority", 3);

console.log(pq.dequeue());  // High priority
console.log(pq.dequeue());  // Medium priority
console.log(pq.dequeue());  // Low priority
```

### Priority Queue Use Cases

1. **Dijkstra's Shortest Path Algorithm**: Finding shortest paths in graphs
2. **A* Search Algorithm**: Pathfinding with heuristics
3. **Huffman Coding**: Data compression
4. **Event-Driven Simulation**: Processing events by time
5. **Task Scheduling**: Operating system task scheduling
6. **Median Finding**: Maintaining running median with two heaps
7. **Load Balancing**: Distributing tasks based on priority

## Deque (Double-Ended Queue)

A deque (pronounced "deck") is a double-ended queue that allows insertion and deletion at both ends. It's more flexible than a standard queue and can be used as both a queue and a stack.

### Operations

- **addFront(item)**: Add to the front - O(1)
- **addRear(item)**: Add to the rear - O(1)
- **removeFront()**: Remove from front - O(1)
- **removeRear()**: Remove from rear - O(1)
- **peekFront()**: View front element - O(1)
- **peekRear()**: View rear element - O(1)

### Python Implementation Using collections.deque

```python
from collections import deque

class Deque:
    def __init__(self):
        self.items = deque()

    def add_front(self, item):
        """Add item to front - O(1)"""
        self.items.appendleft(item)

    def add_rear(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)

    def remove_front(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.popleft()

    def remove_rear(self):
        """Remove and return rear item - O(1)"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop()

    def peek_front(self):
        """View front item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty deque")
        return self.items[0]

    def peek_rear(self):
        """View rear item - O(1)"""
        if self.is_empty():
            raise IndexError("Peek from empty deque")
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def clear(self):
        self.items.clear()

    def __str__(self):
        return f"Deque({list(self.items)})"


# Usage
dq = Deque()
dq.add_rear(10)
dq.add_rear(20)
dq.add_front(5)
print(dq)  # Deque([5, 10, 20])

print(dq.remove_front())  # 5
print(dq.remove_rear())   # 20
print(dq)  # Deque([10])
```

### Custom Deque Using Doubly Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedDeque:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def add_front(self, item):
        """Add item to front - O(1)"""
        new_node = Node(item)

        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node

        self._size += 1

    def add_rear(self, item):
        """Add item to rear - O(1)"""
        new_node = Node(item)

        if self.is_empty():
            self.front = self.rear = new_node
        else:
            new_node.prev = self.rear
            self.rear.next = new_node
            self.rear = new_node

        self._size += 1

    def remove_front(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")

        item = self.front.data
        self.front = self.front.next

        if self.front is None:
            self.rear = None
        else:
            self.front.prev = None

        self._size -= 1
        return item

    def remove_rear(self):
        """Remove and return rear item - O(1)"""
        if self.is_empty():
            raise IndexError("Remove from empty deque")

        item = self.rear.data
        self.rear = self.rear.prev

        if self.rear is None:
            self.front = None
        else:
            self.rear.next = None

        self._size -= 1
        return item

    def peek_front(self):
        if self.is_empty():
            raise IndexError("Peek from empty deque")
        return self.front.data

    def peek_rear(self):
        if self.is_empty():
            raise IndexError("Peek from empty deque")
        return self.rear.data

    def is_empty(self):
        return self.front is None

    def size(self):
        return self._size


# Usage
dll_deque = DoublyLinkedDeque()
dll_deque.add_rear(1)
dll_deque.add_rear(2)
dll_deque.add_front(0)
print(dll_deque.remove_front())  # 0
print(dll_deque.remove_rear())   # 2
```

### C++ Implementation

```cpp
#include <iostream>
#include <deque>

void basicDequeExample() {
    std::deque<int> dq;

    // Add to rear
    dq.push_back(10);
    dq.push_back(20);

    // Add to front
    dq.push_front(5);
    dq.push_front(1);

    // dq: [1, 5, 10, 20]

    std::cout << "Front: " << dq.front() << std::endl;  // 1
    std::cout << "Back: " << dq.back() << std::endl;    // 20

    // Remove from front
    dq.pop_front();  // dq: [5, 10, 20]

    // Remove from rear
    dq.pop_back();  // dq: [5, 10]

    // Access by index (like vector)
    std::cout << "dq[0]: " << dq[0] << std::endl;  // 5
    std::cout << "dq[1]: " << dq[1] << std::endl;  // 10
}
```

### Deque Common Patterns

#### Sliding Window Maximum

```python
from collections import deque

def sliding_window_max(nums, k):
    """
    Find maximum in each sliding window of size k.
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Store indices
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove elements smaller than current
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Add to result when window is full
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# Usage
print(sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3))
# Output: [3, 3, 5, 5, 6, 7]
```

#### Palindrome Checker

```python
def is_palindrome(text):
    """Check if text is palindrome using deque"""
    from collections import deque

    # Remove non-alphanumeric and convert to lowercase
    dq = deque(c.lower() for c in text if c.isalnum())

    while len(dq) > 1:
        if dq.popleft() != dq.pop():
            return False

    return True


# Usage
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("race a car"))  # False
```

### Deque Use Cases

1. **Sliding Window Problems**: Maximum/minimum in sliding windows
2. **Palindrome Checking**: Compare from both ends
3. **Undo/Redo Operations**: Browser history navigation
4. **Task Stealing**: Work-stealing algorithms in parallel processing
5. **Cache Implementation**: LRU cache with quick access at both ends

## Thread-Safe Queues

Thread-safe queues are essential for concurrent programming, allowing multiple threads to safely enqueue and dequeue elements without data corruption or race conditions.

### Python Thread-Safe Queue

```python
import queue
import threading
import time

# Using queue.Queue (thread-safe by default)
def producer(q, items):
    """Producer thread adds items to queue"""
    for item in items:
        print(f"Producing: {item}")
        q.put(item)
        time.sleep(0.1)

    # Signal completion
    q.put(None)


def consumer(q):
    """Consumer thread removes items from queue"""
    while True:
        item = q.get()

        if item is None:
            # Poison pill - exit
            q.task_done()
            break

        print(f"Consuming: {item}")
        time.sleep(0.2)
        q.task_done()


# Usage
q = queue.Queue()

# Create threads
producer_thread = threading.Thread(target=producer, args=(q, range(10)))
consumer_thread = threading.Thread(target=consumer, args=(q,))

# Start threads
producer_thread.start()
consumer_thread.start()

# Wait for completion
producer_thread.join()
consumer_thread.join()
q.join()

print("All tasks completed")
```

### Priority Queue with Threading

```python
import queue
import threading
import time
import random

def task_producer(pq, num_tasks):
    """Produce tasks with random priorities"""
    for i in range(num_tasks):
        priority = random.randint(1, 10)
        task = f"Task-{i}"
        pq.put((priority, task))
        print(f"Added: {task} with priority {priority}")
        time.sleep(0.1)


def task_consumer(pq, consumer_id):
    """Consume tasks based on priority"""
    while True:
        try:
            # Wait for 1 second, then exit if no items
            priority, task = pq.get(timeout=1)
            print(f"Consumer-{consumer_id} processing: {task} (priority {priority})")
            time.sleep(0.3)
            pq.task_done()
        except queue.Empty:
            print(f"Consumer-{consumer_id} finished")
            break


# Usage
pq = queue.PriorityQueue()

# Create producer
producer = threading.Thread(target=task_producer, args=(pq, 10))

# Create multiple consumers
consumers = [
    threading.Thread(target=task_consumer, args=(pq, i))
    for i in range(3)
]

# Start all threads
producer.start()
for consumer in consumers:
    consumer.start()

# Wait for completion
producer.join()
pq.join()
for consumer in consumers:
    consumer.join()

print("All tasks processed")
```

### Custom Thread-Safe Queue

```python
import threading

class ThreadSafeQueue:
    def __init__(self, max_size=None):
        self.items = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def enqueue(self, item, block=True, timeout=None):
        """Add item to queue - thread-safe"""
        with self.not_full:
            # Wait if queue is full
            if self.max_size is not None:
                while len(self.items) >= self.max_size:
                    if not block:
                        raise queue.Full("Queue is full")
                    self.not_full.wait(timeout)

            self.items.append(item)
            self.not_empty.notify()

    def dequeue(self, block=True, timeout=None):
        """Remove item from queue - thread-safe"""
        with self.not_empty:
            # Wait if queue is empty
            while len(self.items) == 0:
                if not block:
                    raise queue.Empty("Queue is empty")
                self.not_empty.wait(timeout)

            item = self.items.pop(0)
            self.not_full.notify()
            return item

    def size(self):
        with self.lock:
            return len(self.items)

    def is_empty(self):
        with self.lock:
            return len(self.items) == 0
```

### C++ Thread-Safe Queue

```cpp
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond_var;

public:
    void enqueue(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(item);
        }
        cond_var.notify_one();
    }

    bool dequeue(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex);

        if (timeout_ms < 0) {
            // Wait indefinitely
            cond_var.wait(lock, [this] { return !queue.empty(); });
        } else {
            // Wait with timeout
            auto timeout = std::chrono::milliseconds(timeout_ms);
            if (!cond_var.wait_for(lock, timeout, [this] { return !queue.empty(); })) {
                return false;  // Timeout
            }
        }

        item = queue.front();
        queue.pop();
        return true;
    }

    bool tryDequeue(T& item) {
        std::lock_guard<std::mutex> lock(mutex);

        if (queue.empty()) {
            return false;
        }

        item = queue.front();
        queue.pop();
        return true;
    }

    bool isEmpty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

// Usage example
void producer(ThreadSafeQueue<int>& q) {
    for (int i = 0; i < 10; i++) {
        q.enqueue(i);
        std::cout << "Produced: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(ThreadSafeQueue<int>& q, int id) {
    while (true) {
        int item;
        if (q.dequeue(item, 1000)) {
            std::cout << "Consumer " << id << " consumed: " << item << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } else {
            break;  // Timeout - no more items
        }
    }
}

int main() {
    ThreadSafeQueue<int> q;

    std::thread prod(producer, std::ref(q));
    std::thread cons1(consumer, std::ref(q), 1);
    std::thread cons2(consumer, std::ref(q), 2);

    prod.join();
    cons1.join();
    cons2.join();

    return 0;
}
```

## Real-World Applications

### 1. Task Scheduling

```python
from collections import deque
import time

class TaskScheduler:
    def __init__(self):
        self.task_queue = deque()

    def add_task(self, task_name, priority=0):
        """Add task to queue"""
        self.task_queue.append({
            'name': task_name,
            'priority': priority,
            'timestamp': time.time()
        })

    def execute_tasks(self):
        """Execute all tasks in FIFO order"""
        while self.task_queue:
            task = self.task_queue.popleft()
            print(f"Executing: {task['name']} (queued at {task['timestamp']:.2f})")
            # Simulate task execution
            time.sleep(0.1)


# Usage
scheduler = TaskScheduler()
scheduler.add_task("Send email")
scheduler.add_task("Generate report")
scheduler.add_task("Backup database")
scheduler.execute_tasks()
```

### 2. Breadth-First Search (BFS) Traversal

```python
from collections import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def bfs_traversal(root):
    """
    Breadth-first traversal of binary tree using queue.
    Time: O(n), Space: O(w) where w is max width
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.value)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result


def level_order_traversal(root):
    """Get nodes level by level"""
    if not root:
        return []

    levels = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.value)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        levels.append(current_level)

    return levels


# Usage
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print(bfs_traversal(root))  # [1, 2, 3, 4, 5]
print(level_order_traversal(root))  # [[1], [2, 3], [4, 5]]
```

### 3. Message Queue System

```python
from collections import deque
from datetime import datetime

class Message:
    def __init__(self, sender, recipient, content):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.timestamp = datetime.now()

    def __str__(self):
        return f"[{self.timestamp}] {self.sender} -> {self.recipient}: {self.content}"


class MessageQueue:
    def __init__(self):
        self.queue = deque()

    def send_message(self, sender, recipient, content):
        """Add message to queue"""
        message = Message(sender, recipient, content)
        self.queue.append(message)
        print(f"Message queued: {message}")

    def process_messages(self, batch_size=10):
        """Process messages in batches"""
        processed = 0

        while self.queue and processed < batch_size:
            message = self.queue.popleft()
            self._deliver_message(message)
            processed += 1

        return processed

    def _deliver_message(self, message):
        """Simulate message delivery"""
        print(f"Delivering: {message}")

    def pending_count(self):
        return len(self.queue)


# Usage
mq = MessageQueue()
mq.send_message("Alice", "Bob", "Hello!")
mq.send_message("Bob", "Alice", "Hi there!")
mq.send_message("Charlie", "Alice", "Meeting at 3pm")

print(f"\nProcessing {mq.pending_count()} messages...")
mq.process_messages()
```

### 4. Print Queue Management

```python
from collections import deque
import time

class PrintJob:
    def __init__(self, document_name, pages, priority=0):
        self.document_name = document_name
        self.pages = pages
        self.priority = priority
        self.submitted_at = time.time()

    def __str__(self):
        return f"{self.document_name} ({self.pages} pages)"


class PrintQueue:
    def __init__(self):
        self.queue = deque()

    def submit_job(self, document_name, pages, priority=0):
        """Submit print job"""
        job = PrintJob(document_name, pages, priority)
        self.queue.append(job)
        print(f"Job submitted: {job}")

    def print_next(self):
        """Print next job in queue"""
        if not self.queue:
            print("No jobs in queue")
            return

        job = self.queue.popleft()
        print(f"Printing: {job}")

        # Simulate printing time (1 second per page)
        time.sleep(job.pages * 0.1)
        print(f"Completed: {job}")

    def print_all(self):
        """Process all print jobs"""
        while self.queue:
            self.print_next()

    def cancel_job(self, document_name):
        """Cancel a specific job"""
        for i, job in enumerate(self.queue):
            if job.document_name == document_name:
                del self.queue[i]
                print(f"Cancelled: {job}")
                return True
        return False

    def jobs_pending(self):
        return len(self.queue)


# Usage
printer = PrintQueue()
printer.submit_job("Report.pdf", 5)
printer.submit_job("Presentation.pptx", 10)
printer.submit_job("Invoice.pdf", 2)

print(f"\nJobs pending: {printer.jobs_pending()}")
printer.print_all()
```

### 5. Buffer Management

```python
from collections import deque

class CircularBuffer:
    """Fixed-size buffer for streaming data"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def write(self, data):
        """Write data to buffer (oldest data removed if full)"""
        self.buffer.append(data)

    def read(self, n=1):
        """Read n items from buffer"""
        if n > len(self.buffer):
            raise ValueError(f"Only {len(self.buffer)} items available")

        result = []
        for _ in range(n):
            result.append(self.buffer.popleft())
        return result

    def peek(self, n=1):
        """View n items without removing"""
        if n > len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[:n]

    def size(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.capacity

    def is_empty(self):
        return len(self.buffer) == 0


# Usage - Audio streaming buffer
audio_buffer = CircularBuffer(capacity=1000)

# Simulate audio streaming
for i in range(1500):
    audio_buffer.write(f"sample_{i}")

print(f"Buffer size: {audio_buffer.size()}")  # 1000
print(f"First samples: {audio_buffer.peek(5)}")  # Last 5 of first 1000
```

## Complexity Analysis

### Time Complexity

| Operation | Array-based Queue | Linked Queue | Circular Queue | Priority Queue (Heap) | Deque |
|-----------|------------------|--------------|----------------|----------------------|-------|
| Enqueue | O(1) amortized* | O(1) | O(1) | O(log n) | O(1) |
| Dequeue | O(n)** | O(1) | O(1) | O(log n) | O(1) |
| Peek | O(1) | O(1) | O(1) | O(1) | O(1) |
| Search | O(n) | O(n) | O(n) | O(n) | O(n) |
| IsEmpty | O(1) | O(1) | O(1) | O(1) | O(1) |
| Size | O(1) | O(1) | O(1) | O(1) | O(1) |

\* Amortized O(1) for dynamic array resizing
\** O(n) if using array.pop(0); O(1) with proper implementation (object-based or circular)

### Space Complexity

- **Array-based Queue**: O(n) where n is the number of elements
- **Linked Queue**: O(n) with additional overhead for node pointers
- **Circular Queue**: O(capacity) - fixed size
- **Priority Queue**: O(n)
- **Deque**: O(n)

### Detailed Complexity Analysis

#### Enqueue Operation

```python
# Array-based (dynamic resizing)
# Most operations: O(1)
# When resizing: O(n) - copy all elements to new array
# Amortized: O(1)

# Linked list
# Always: O(1) - just update rear pointer
```

#### Dequeue Operation

```python
# Array-based with pop(0)
# O(n) - shift all remaining elements

# Array-based with index tracking
# O(1) - just increment front index

# Linked list
# O(1) - update front pointer
```

## Comparison with Other Data Structures

### Queue vs Stack

| Feature | Queue | Stack |
|---------|-------|-------|
| Order | FIFO (First In First Out) | LIFO (Last In First Out) |
| Access | Front and rear only | Top only |
| Operations | enqueue, dequeue | push, pop |
| Use Cases | BFS, scheduling, buffering | DFS, undo/redo, expression evaluation |
| Real-world analogy | Line at store | Stack of plates |

### Queue vs Array/List

| Feature | Queue | Array/List |
|---------|-------|------------|
| Access pattern | Sequential (FIFO) | Random access by index |
| Insertion | O(1) at rear | O(1) at end, O(n) elsewhere |
| Deletion | O(1) at front | O(1) at end, O(n) elsewhere |
| Use when | Order matters, process sequentially | Need random access |

### Queue vs Priority Queue

| Feature | Queue | Priority Queue |
|---------|-------|----------------|
| Order | Insertion order (FIFO) | Priority order |
| Dequeue | First element | Highest priority |
| Implementation | Array or linked list | Heap |
| Complexity | O(1) enqueue/dequeue | O(log n) enqueue/dequeue |
| Use when | Order by time | Order by importance |

### When to Use Each

**Use Regular Queue when:**
- Processing items in order received
- FIFO behavior is required
- Simple task scheduling
- BFS traversal
- Request handling

**Use Priority Queue when:**
- Items have different priorities
- Need to process most important items first
- Implementing Dijkstra's algorithm
- Event-driven simulation
- Task scheduling with priorities

**Use Deque when:**
- Need insertion/deletion at both ends
- Implementing sliding window problems
- Palindrome checking
- Undo/redo functionality
- Both stack and queue operations needed

**Use Circular Queue when:**
- Fixed maximum size is known
- Need to reuse memory efficiently
- Implementing buffers
- Round-robin scheduling

## Interview Problems and Patterns

### Pattern 1: Queue Using Stacks

**Problem**: Implement a queue using two stacks.

```python
class QueueUsingStacks:
    def __init__(self):
        self.stack_in = []   # For enqueue
        self.stack_out = []  # For dequeue

    def enqueue(self, item):
        """O(1)"""
        self.stack_in.append(item)

    def dequeue(self):
        """Amortized O(1)"""
        if not self.stack_out:
            # Transfer all from stack_in to stack_out
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())

        if not self.stack_out:
            raise IndexError("Dequeue from empty queue")

        return self.stack_out.pop()

    def peek(self):
        """Amortized O(1)"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())

        if not self.stack_out:
            raise IndexError("Peek from empty queue")

        return self.stack_out[-1]

    def is_empty(self):
        return len(self.stack_in) == 0 and len(self.stack_out) == 0


# Test
q = QueueUsingStacks()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
print(q.dequeue())  # 1
print(q.peek())     # 2
q.enqueue(4)
print(q.dequeue())  # 2
print(q.dequeue())  # 3
```

### Pattern 2: Stack Using Queues

**Problem**: Implement a stack using two queues.

```python
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, item):
        """O(n) - make push expensive"""
        # Add to q2
        self.q2.append(item)

        # Move all from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())

        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        """O(1)"""
        if not self.q1:
            raise IndexError("Pop from empty stack")
        return self.q1.popleft()

    def top(self):
        """O(1)"""
        if not self.q1:
            raise IndexError("Top from empty stack")
        return self.q1[0]

    def is_empty(self):
        return len(self.q1) == 0


# Test
s = StackUsingQueues()
s.push(1)
s.push(2)
s.push(3)
print(s.pop())  # 3
print(s.top())  # 2
```

### Pattern 3: First Unique Number

**Problem**: Find the first non-repeating element in a stream.

```python
from collections import deque

class FirstUnique:
    def __init__(self):
        self.queue = deque()
        self.count = {}

    def add(self, num):
        """Add number to stream"""
        if num in self.count:
            self.count[num] += 1
        else:
            self.count[num] = 1
            self.queue.append(num)

    def get_first_unique(self):
        """Get first unique number - O(1) amortized"""
        # Remove non-unique from front
        while self.queue and self.count[self.queue[0]] > 1:
            self.queue.popleft()

        if self.queue:
            return self.queue[0]
        return -1


# Usage
fu = FirstUnique()
for num in [1, 2, 1, 3, 2, 4]:
    fu.add(num)
    print(f"Added {num}, first unique: {fu.get_first_unique()}")
# Output: 1, 2, 2, 3, 3, 3
```

### Pattern 4: Generate Binary Numbers

**Problem**: Generate binary numbers from 1 to n using a queue.

```python
from collections import deque

def generate_binary_numbers(n):
    """
    Generate binary numbers 1 to n using queue.
    Time: O(n), Space: O(n)
    """
    result = []
    queue = deque(['1'])

    for _ in range(n):
        # Get front binary number
        binary = queue.popleft()
        result.append(binary)

        # Generate next numbers by appending 0 and 1
        queue.append(binary + '0')
        queue.append(binary + '1')

    return result


# Usage
print(generate_binary_numbers(10))
# ['1', '10', '11', '100', '101', '110', '111', '1000', '1001', '1010']
```

### Pattern 5: Moving Average from Data Stream

**Problem**: Calculate moving average from a stream of integers.

```python
from collections import deque

class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0

    def next(self, val):
        """Add value and return moving average - O(1)"""
        self.queue.append(val)
        self.sum += val

        if len(self.queue) > self.size:
            removed = self.queue.popleft()
            self.sum -= removed

        return self.sum / len(self.queue)


# Usage
ma = MovingAverage(3)
print(ma.next(1))   # 1.0
print(ma.next(10))  # 5.5
print(ma.next(3))   # 4.666...
print(ma.next(5))   # 6.0
```

### Pattern 6: Recent Counter

**Problem**: Count requests in the last 3000 milliseconds.

```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.requests = deque()

    def ping(self, t):
        """
        Add request at time t, return count in [t-3000, t].
        Time: O(1) amortized
        """
        self.requests.append(t)

        # Remove requests older than t-3000
        while self.requests and self.requests[0] < t - 3000:
            self.requests.popleft()

        return len(self.requests)


# Usage
rc = RecentCounter()
print(rc.ping(1))     # 1
print(rc.ping(100))   # 2
print(rc.ping(3001))  # 3
print(rc.ping(3002))  # 3 (request at t=1 is now outside window)
```

### Pattern 7: Design Hit Counter

**Problem**: Design a hit counter that counts hits in the past 5 minutes.

```python
from collections import deque

class HitCounter:
    def __init__(self):
        self.hits = deque()

    def hit(self, timestamp):
        """Record a hit at timestamp - O(1)"""
        self.hits.append(timestamp)

    def get_hits(self, timestamp):
        """
        Return hits in past 5 minutes (300 seconds).
        Time: O(n) worst case, O(1) amortized
        """
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()

        return len(self.hits)


# Usage
hc = HitCounter()
hc.hit(1)
hc.hit(2)
hc.hit(3)
print(hc.get_hits(4))    # 3
hc.hit(300)
print(hc.get_hits(300))  # 4
print(hc.get_hits(301))  # 3 (hit at t=1 expired)
```

### Pattern 8: Number of Recent Calls (Sliding Window)

**Problem**: Solve sliding window problems using queue.

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []

    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove elements outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they won't be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# Usage
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# Output: [3, 3, 5, 5, 6, 7]
```

### Pattern 9: Perfect Squares (BFS)

**Problem**: Find minimum number of perfect squares that sum to n.

```python
from collections import deque
import math

def num_squares(n):
    """
    Find min perfect squares that sum to n using BFS.
    Time: O(n * sqrt(n)), Space: O(n)
    """
    if n <= 0:
        return 0

    # Generate perfect squares up to n
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1

    # BFS
    queue = deque([(n, 0)])  # (remaining, steps)
    visited = {n}

    while queue:
        remaining, steps = queue.popleft()

        for square in squares:
            next_remaining = remaining - square

            if next_remaining == 0:
                return steps + 1

            if next_remaining > 0 and next_remaining not in visited:
                visited.add(next_remaining)
                queue.append((next_remaining, steps + 1))

    return -1


# Usage
print(num_squares(12))  # 3 (4+4+4)
print(num_squares(13))  # 2 (4+9)
```

### Pattern 10: Rotting Oranges (Multi-source BFS)

**Problem**: Find minimum time to rot all oranges.

```python
from collections import deque

def oranges_rotting(grid):
    """
    Multi-source BFS to find time to rot all oranges.
    Time: O(m*n), Space: O(m*n)
    """
    if not grid:
        return -1

    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0

    # Find all initial rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh_count += 1

    if fresh_count == 0:
        return 0

    # BFS
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    max_time = 0

    while queue:
        r, c, time = queue.popleft()
        max_time = max(max_time, time)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # Mark as rotten
                fresh_count -= 1
                queue.append((nr, nc, time + 1))

    return max_time if fresh_count == 0 else -1


# Usage
grid = [
    [2, 1, 1],
    [1, 1, 0],
    [0, 1, 1]
]
print(oranges_rotting(grid))  # 4
```

## Summary

Queues are fundamental data structures with wide-ranging applications in computer science:

### Key Takeaways

1. **FIFO Principle**: First In First Out - core defining characteristic
2. **Multiple Variants**: Regular, circular, priority, and deque serve different needs
3. **Efficient Operations**: O(1) for enqueue/dequeue with proper implementation
4. **Essential for Algorithms**: BFS, scheduling, buffering all rely on queues
5. **Thread-Safe Options**: Critical for concurrent programming
6. **Interview Frequency**: Common in coding interviews with various patterns

### Best Practices

- Use `collections.deque` in Python for efficient queue operations
- Use circular queues when maximum size is known
- Use priority queues when order depends on priority, not arrival time
- Consider thread-safety requirements in concurrent environments
- Choose implementation based on specific use case requirements

### Further Study

- Advanced priority queue operations (decrease-key, merge)
- Lock-free concurrent queues
- Distributed message queues (RabbitMQ, Kafka)
- Queue-based load balancing algorithms
- Advanced graph algorithms using queues (bidirectional BFS, 0-1 BFS)
