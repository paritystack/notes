# Heaps

Heaps are a special tree-based data structure that satisfies the heap property. In a max heap, for any given node, the value of the node is greater than or equal to the values of its children, while in a min heap, the value of the node is less than or equal to the values of its children. Heaps are commonly used to implement priority queues and for efficient sorting algorithms.

## Key Concepts

- **Heap Property**: The key property that defines a heap, ensuring that the parent node is either greater than (max heap) or less than (min heap) its children.

- **Complete Binary Tree**: Heaps are typically implemented as complete binary trees, where all levels are fully filled except possibly for the last level, which is filled from left to right.

- **Array Representation**: Heaps are efficiently stored in arrays where for any element at index `i`:
  - Parent: `(i - 1) / 2`
  - Left Child: `2 * i + 1`
  - Right Child: `2 * i + 2`

- **Height**: A heap with `n` elements has height $O(\log n)$, which makes many operations logarithmic.

## Types of Heaps

### Binary Heap
- **Min Heap**: Parent is smaller than or equal to children. Root contains minimum element.
- **Max Heap**: Parent is greater than or equal to children. Root contains maximum element.
- Most common and straightforward implementation.

### D-ary Heap
- Each node has `d` children instead of 2.
- Better cache performance for large datasets.
- Trade-off: Faster insertion, slower deletion.

### Fibonacci Heap
- Collection of trees satisfying min/max heap property.
- Supports amortized $O(1)$ for insert, decrease-key, and merge operations.
- Used in advanced graph algorithms (Dijkstra, Prim).

### Binomial Heap
- Collection of binomial trees.
- Supports efficient merge operation in $O(\log n)$.
- Each binomial tree satisfies heap property.

## Implementation Details

### Array Representation

A heap can be efficiently represented as an array:

```
Array:  [1, 3, 6, 5, 9, 8]
Tree:       1
           / \
          3   6
         / \  /
        5  9 8
```

**Index Formulas (0-indexed):**
- Parent of node at index `i`: `⌊(i-1)/2⌋`
- Left child of node at index `i`: `2i + 1`
- Right child of node at index `i`: `2i + 2`

**Index Formulas (1-indexed):**
- Parent of node at index `i`: `⌊i/2⌋`
- Left child of node at index `i`: `2i`
- Right child of node at index `i`: `2i + 1`

## Detailed Operations

### 1. Insertion (Push)

**Process:**
1. Add the new element at the end of the array (next available position)
2. "Bubble up" (percolate up): Compare with parent and swap if heap property is violated
3. Continue until heap property is restored or reach root

**Time Complexity:** $O(\log n)$
**Space Complexity:** $O(1)$

**Pseudocode:**
```
insert(heap, value):
    heap.append(value)
    index = heap.size - 1

    while index > 0:
        parent = (index - 1) / 2
        if heap[parent] > heap[index]:  // for min heap
            swap(heap[parent], heap[index])
            index = parent
        else:
            break
```

### 2. Extract Min/Max (Pop)

**Process:**
1. Save the root element (min/max) to return
2. Replace root with the last element in the heap
3. Remove the last element
4. "Bubble down" (percolate down): Compare with children and swap with smaller/larger child if heap property is violated
5. Continue until heap property is restored or reach leaf

**Time Complexity:** $O(\log n)$
**Space Complexity:** $O(1)$

**Pseudocode:**
```
extractMin(heap):
    if heap.isEmpty():
        return null

    minValue = heap[0]
    heap[0] = heap[heap.size - 1]
    heap.removeLast()

    index = 0
    while true:
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < heap.size and heap[left] < heap[smallest]:
            smallest = left
        if right < heap.size and heap[right] < heap[smallest]:
            smallest = right

        if smallest != index:
            swap(heap[index], heap[smallest])
            index = smallest
        else:
            break

    return minValue
```

### 3. Peek (Get Min/Max)

**Process:**
- Simply return the root element without removing it

**Time Complexity:** $O(1)$
**Space Complexity:** $O(1)$

### 4. Heapify

**Process:** Convert an arbitrary array into a heap.

**Bottom-Up Approach (Optimal):**
1. Start from the last non-leaf node: `⌊n/2⌋ - 1`
2. Apply "bubble down" operation on each node moving towards root
3. This ensures all subtrees satisfy heap property before processing parent

**Time Complexity:** $O(n)$ - Though it seems $O(n \log n)$, mathematical analysis proves it's linear
**Space Complexity:** $O(1)$ for iterative, $O(\log n)$ for recursive (call stack)

**Pseudocode:**
```
heapify(array):
    n = array.length
    // Start from last non-leaf node
    for i from (n/2 - 1) down to 0:
        bubbleDown(array, i, n)

bubbleDown(array, index, heapSize):
    while true:
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < heapSize and array[left] < array[smallest]:
            smallest = left
        if right < heapSize and array[right] < array[smallest]:
            smallest = right

        if smallest != index:
            swap(array[index], array[smallest])
            index = smallest
        else:
            break
```

### 5. Decrease Key (Min Heap) / Increase Key (Max Heap)

**Process:**
1. Decrease the value at given index
2. Bubble up to restore heap property

**Time Complexity:** $O(\log n)$
**Space Complexity:** $O(1)$

**Use Cases:** Dijkstra's algorithm, Prim's algorithm

### 6. Delete Arbitrary Element

**Process:**
1. Replace element with the last element
2. Remove last element
3. Bubble up or bubble down as needed

**Time Complexity:** $O(\log n)$
**Space Complexity:** $O(1)$

### 7. Merge Two Heaps

**Process:**
- **Simple approach**: Combine arrays and heapify: $O(n + m)$
- **For Fibonacci/Binomial heaps**: More efficient merge operations

**Time Complexity:** $O(n + m)$ for binary heaps
**Space Complexity:** $O(n + m)$ or $O(1)$ if in-place

## Common Patterns and Use Cases

### 1. K Largest/Smallest Elements

**Pattern:** Use a min heap of size K to find K largest elements (or max heap for K smallest).

**Approach:**
- Maintain a min heap of size K
- For each element, if it's larger than heap's minimum, remove min and add new element
- Final heap contains K largest elements

**Time Complexity:** $O(n \log k)$
**Space Complexity:** $O(k)$

**Example Problem:** Find the Kth largest element in an array.

```python
def findKthLargest(nums, k):
    heap = []
    for num in nums:
        heappush(heap, num)
        if len(heap) > k:
            heappop(heap)
    return heap[0]  # root of min heap
```

**Variations:**
- K largest elements in a stream
- K closest points to origin
- K most frequent elements

### 2. Merge K Sorted Lists/Arrays

**Pattern:** Use a min heap to efficiently merge K sorted sequences.

**Approach:**
- Add the first element from each list to a min heap (with list index tracking)
- Repeatedly extract min, add to result, and insert next element from same list
- Continue until all elements are processed

**Time Complexity:** $O(N \log k)$ where N is total elements
**Space Complexity:** $O(k)$ for the heap

**Example Problem:** Merge K sorted linked lists.

```python
def mergeKLists(lists):
    heap = []
    result = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heappush(heap, (lst[0], i, 0))  # (value, list_index, element_index)

    while heap:
        val, list_idx, elem_idx = heappop(heap)
        result.append(val)

        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
```

**Variations:**
- Smallest range covering elements from K lists
- Merge K sorted arrays

### 3. Median Maintenance (Two Heaps)

**Pattern:** Use two heaps to maintain running median in a data stream.

**Approach:**
- **Max heap (left)**: Stores smaller half of numbers
- **Min heap (right)**: Stores larger half of numbers
- Balance: Ensure size difference is at most 1
- Median: If equal size, average of two tops; otherwise, top of larger heap

**Time Complexity:** $O(\log n)$ per insertion
**Space Complexity:** $O(n)$

**Example Problem:** Find median from data stream.

```python
class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negate values)
        self.large = []  # min heap

    def addNum(self, num):
        # Add to max heap (small)
        heappush(self.small, -num)

        # Balance: move largest from small to large
        heappush(self.large, -heappop(self.small))

        # Maintain size property
        if len(self.small) < len(self.large):
            heappush(self.small, -heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

**Variations:**
- Sliding window median
- Find median in specific range

### 4. Sliding Window Maximum/Minimum

**Pattern:** Use heap with lazy deletion or monotonic deque.

**Approach with Heap:**
- Maintain a max heap with (value, index) pairs
- For each window, add new element
- Remove elements outside window (lazy deletion - check index)
- Top of heap is maximum for current window

**Time Complexity:** $O(n \log n)$
**Space Complexity:** $O(n)$

**Example Problem:** Sliding window maximum.

```python
def maxSlidingWindow(nums, k):
    heap = []
    result = []

    for i, num in enumerate(nums):
        heappush(heap, (-num, i))  # max heap

        if i >= k - 1:
            # Remove elements outside window
            while heap and heap[0][1] <= i - k:
                heappop(heap)
            result.append(-heap[0][0])

    return result
```

**Note:** Monotonic deque is more optimal $O(n)$ for this specific pattern.

### 5. Top K Frequent Elements

**Pattern:** Use heap to find most/least frequent elements.

**Approach:**
- Count frequency using hash map
- Use min heap of size K to track K most frequent
- Or use bucket sort for $O(n)$ solution

**Time Complexity:** $O(n \log k)$
**Space Complexity:** $O(n)$

**Example Problem:** Top K frequent words.

```python
def topKFrequent(words, k):
    from collections import Counter
    count = Counter(words)

    # Min heap of size k
    heap = []
    for word, freq in count.items():
        heappush(heap, (freq, word))
        if len(heap) > k:
            heappop(heap)

    # Extract and reverse for descending order
    result = []
    while heap:
        result.append(heappop(heap)[1])
    return result[::-1]
```

### 6. Task Scheduling / Meeting Rooms

**Pattern:** Use heap to track ongoing tasks/meetings and their end times.

**Approach:**
- Sort intervals by start time
- Use min heap to track end times of ongoing intervals
- For each new interval, remove finished ones from heap
- Heap size represents minimum resources needed

**Time Complexity:** $O(n \log n)$
**Space Complexity:** $O(n)$

**Example Problem:** Meeting Rooms II (minimum conference rooms needed).

```python
def minMeetingRooms(intervals):
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])  # sort by start time
    heap = []

    for interval in intervals:
        # If earliest ending meeting finishes before current starts
        if heap and heap[0] <= interval[0]:
            heappop(heap)
        heappush(heap, interval[1])  # add end time

    return len(heap)  # heap size = rooms needed
```

**Variations:**
- CPU task scheduling with cooldown
- Car pooling (capacity constraints)
- Maximum CPU load

### 7. Dijkstra's Shortest Path

**Pattern:** Use min heap to always process nearest unvisited vertex.

**Approach:**
- Initialize heap with (distance, node) starting from source
- Extract minimum distance node
- Update distances to neighbors
- Continue until destination reached or heap empty

**Time Complexity:** $O((V + E) \log V)$ with binary heap
**Space Complexity:** $O(V)$

**Example Problem:** Single source shortest path.

```python
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    visited = set()

    while heap:
        current_dist, node = heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in graph[node]:
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(heap, (distance, neighbor))

    return distances
```

### 8. Prim's Minimum Spanning Tree

**Pattern:** Use min heap to select minimum weight edge connecting tree to non-tree vertex.

**Approach:**
- Start with arbitrary vertex
- Maintain heap of edges from tree to non-tree vertices
- Repeatedly add minimum weight edge that extends tree
- Continue until all vertices included

**Time Complexity:** $O(E \log V)$
**Space Complexity:** $O(V)$

### 9. Huffman Coding

**Pattern:** Use min heap to build optimal prefix-free encoding tree.

**Approach:**
- Create leaf node for each character with frequency
- Build min heap of all nodes
- Repeatedly extract two minimum nodes, create parent with combined frequency
- Continue until one node remains (root)

**Time Complexity:** $O(n \log n)$
**Space Complexity:** $O(n)$

### 10. Continuous Median / Running Statistics

**Pattern:** Two heaps for dynamic median, can extend to percentiles.

**Use Case:**
- Real-time analytics
- Monitoring systems
- Streaming data processing

**Example Problem:** Find 95th percentile in stream (use two heaps with 95:5 ratio).

### 11. Reorganize String / Task Scheduler

**Pattern:** Use max heap to greedily select most frequent character/task.

**Approach:**
- Count frequencies
- Use max heap to always select most frequent available item
- Track cooldown or previously used item
- Build result by alternating selections

**Time Complexity:** $O(n \log k)$ where k is unique items
**Space Complexity:** $O(k)$

**Example Problem:** Reorganize string (no two adjacent characters same).

```python
def reorganizeString(s):
    from collections import Counter
    count = Counter(s)
    heap = [(-freq, char) for char, freq in count.items()]
    heapify(heap)

    result = []
    prev_freq, prev_char = 0, ''

    while heap:
        freq, char = heappop(heap)
        result.append(char)

        if prev_freq < 0:
            heappush(heap, (prev_freq, prev_char))

        prev_freq, prev_char = freq + 1, char

    result_str = ''.join(result)
    return result_str if len(result_str) == len(s) else ""
```

### 12. Stock Price Fluctuation / Maximum in Window

**Pattern:** Track maximum/minimum with ability to update past values.

**Approach:**
- Use heap with timestamps or indices
- Support update operation
- Lazy deletion for outdated entries

### 13. Trapping Rain Water II (2D)

**Pattern:** Use min heap to process cells from outside to inside.

**Approach:**
- Start with all boundary cells in min heap
- Process cells in order of height
- Track water level as maximum height seen
- Calculate trapped water as difference

**Time Complexity:** $O(mn \log(mn))$
**Space Complexity:** $O(mn)$

## Time & Space Complexity Reference

| Operation | Binary Heap | Fibonacci Heap | Binomial Heap |
|-----------|-------------|----------------|---------------|
| Insert | $O(\log n)$ | $O(1)$* | $O(\log n)$ |
| Extract-Min/Max | $O(\log n)$ | $O(\log n)$* | $O(\log n)$ |
| Peek | $O(1)$ | $O(1)$ | $O(1)$ |
| Decrease-Key | $O(\log n)$ | $O(1)$* | $O(\log n)$ |
| Delete | $O(\log n)$ | $O(\log n)$* | $O(\log n)$ |
| Merge | $O(n)$ | $O(1)$ | $O(\log n)$ |
| Build Heap | $O(n)$ | $O(n)$ | $O(n)$ |

\* Amortized time complexity

**Space Complexity:** $O(n)$ for storing n elements in all heap types.

## Problem-Solving Strategies

### When to Use Heaps

1. **Need repeated access to minimum/maximum element**
   - Priority queues
   - Scheduling problems

2. **K-way problems**
   - K largest/smallest elements
   - Merge K sorted sequences
   - Top K frequent items

3. **Streaming/online algorithms**
   - Running median
   - Top K in real-time
   - Continuous statistics

4. **Greedy algorithms**
   - Always need next best choice
   - Dijkstra's, Prim's algorithms
   - Huffman coding

5. **Partial sorting**
   - Don't need full sort, just top/bottom K elements
   - More efficient than full sort: $O(n \log k)$ vs $O(n \log n)$

### When NOT to Use Heaps

1. **Need to access arbitrary elements** - Use hash map or array
2. **Need to maintain sorted order of all elements** - Use balanced BST
3. **Need to search for specific element** - $O(n)$ in heap, use hash map for $O(1)$
4. **Small K relative to N** - For K=1 or K=2, simple variables might be faster

### Common Pitfalls

1. **Forgetting heap property during custom comparisons**
   - When using tuples, ensure primary sort key is correct
   - Python uses lexicographic order for tuples

2. **Max heap in languages with only min heap**
   - Python's heapq only provides min heap
   - Solution: Negate values for max heap behavior

3. **Not handling empty heap**
   - Always check `heap.isEmpty()` before `peek()` or `pop()`

4. **Index calculation errors**
   - Remember: 0-indexed vs 1-indexed affects formulas
   - Parent: `(i-1)/2` for 0-indexed, `i/2` for 1-indexed

5. **Inefficient heap building**
   - Use bottom-up heapify $O(n)$ instead of repeated insertion $O(n \log n)$

6. **Memory issues with large datasets**
   - Heap of size K uses $O(k)$ space, not $O(n)$
   - Better than sorting entire array for top-K problems

### Optimization Techniques

1. **Lazy Deletion**
   - Mark elements as deleted instead of removing
   - Clean up when encountered during extract operations
   - Useful for sliding window problems

2. **Custom Comparators**
   - Define comparison based on specific problem needs
   - Can store complex objects in heap

3. **Heap + Hash Map**
   - Hash map for $O(1)$ lookups
   - Heap for $O(\log n)$ priority operations
   - Useful for LRU/LFU caches with priority

4. **Two Heaps Pattern**
   - Separate min and max heaps
   - Powerful for median, percentile problems
   - Can generalize to multiple heaps for different priorities

## Code Examples

### Python Implementation

```python
import heapq
from typing import List

class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        heapq.heappush(self.heap, val)

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0] if self.heap else None

    def size(self):
        return len(self.heap)

# Max Heap (negate values)
class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        heapq.heappush(self.heap, -val)

    def pop(self):
        return -heapq.heappop(self.heap)

    def peek(self):
        return -self.heap[0] if self.heap else None

# Build heap from array
def build_heap(arr: List[int]) -> List[int]:
    heapq.heapify(arr)  # O(n) operation
    return arr

# Heap sort
def heap_sort(arr: List[int]) -> List[int]:
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]
```

### Java Implementation

```java
import java.util.*;

public class HeapExamples {
    // Min Heap
    public static void minHeapExample() {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        minHeap.offer(5);
        minHeap.offer(3);
        minHeap.offer(7);
        System.out.println(minHeap.poll()); // 3
    }

    // Max Heap
    public static void maxHeapExample() {
        PriorityQueue<Integer> maxHeap =
            new PriorityQueue<>(Collections.reverseOrder());
        maxHeap.offer(5);
        maxHeap.offer(3);
        maxHeap.offer(7);
        System.out.println(maxHeap.poll()); // 7
    }

    // Custom Comparator
    public static void customComparator() {
        PriorityQueue<int[]> pq = new PriorityQueue<>(
            (a, b) -> a[1] - b[1]  // compare by second element
        );
        pq.offer(new int[]{1, 5});
        pq.offer(new int[]{2, 3});
        System.out.println(Arrays.toString(pq.poll())); // [2, 3]
    }
}
```

### C++ Implementation

```cpp
#include <queue>
#include <vector>
#include <iostream>
using namespace std;

int main() {
    // Min Heap (default)
    priority_queue<int, vector<int>, greater<int>> minHeap;
    minHeap.push(5);
    minHeap.push(3);
    minHeap.push(7);
    cout << minHeap.top() << endl; // 3

    // Max Heap
    priority_queue<int> maxHeap;
    maxHeap.push(5);
    maxHeap.push(3);
    maxHeap.push(7);
    cout << maxHeap.top() << endl; // 7

    // Custom Comparator
    auto cmp = [](pair<int,int> a, pair<int,int> b) {
        return a.second > b.second;
    };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
    pq.push({1, 5});
    pq.push({2, 3});
    cout << pq.top().first << endl; // 2

    return 0;
}
```

## Applications

Heaps are widely used in various applications across computer science and software engineering:

### Core Applications

- **Priority Queues**: Heaps provide an efficient way to implement priority queues, allowing for quick access to the highest (or lowest) priority element. Used in task scheduling, event-driven simulation, and job scheduling systems.

- **Heap Sort**: A comparison-based sorting algorithm that uses the heap data structure to sort elements in $O(n \log n)$ time. While not as cache-friendly as quicksort, it guarantees worst-case $O(n \log n)$ performance.

- **Graph Algorithms**:
  - **Dijkstra's Algorithm**: Shortest path finding using min heap to select nearest vertex
  - **Prim's Algorithm**: Minimum spanning tree construction
  - **A\* Search**: Pathfinding with heuristic priority

### Data Processing

- **Stream Processing**: Finding top-K elements, medians, or percentiles in streaming data without storing entire dataset.

- **Data Compression**: Huffman coding uses heaps to build optimal encoding trees for compression algorithms.

- **External Sorting**: K-way merge operations when sorting data that doesn't fit in memory.

### System Design

- **Operating Systems**:
  - Process scheduling (priority-based scheduling)
  - Memory management (best-fit allocation)
  - Event handling in real-time systems

- **Database Systems**:
  - Query optimization (join order selection)
  - Buffer management
  - Index construction

- **Network Systems**:
  - Bandwidth management
  - Packet scheduling in routers
  - Connection pooling

### Real-world Use Cases

- **E-commerce**: Order processing by priority, customer service queue management
- **Gaming**: AI decision-making, pathfinding, event processing
- **Finance**: High-frequency trading (processing orders by price-time priority)
- **Healthcare**: Emergency room triage systems, appointment scheduling
- **Transportation**: Route optimization, ride-sharing algorithms

## Conclusion

Heaps are a fundamental and versatile data structure that provides efficient solutions for a wide range of problems, particularly those involving priority management, partial sorting, and dynamic datasets. Their ability to maintain the min/max element in $O(1)$ time while supporting $O(\log n)$ insertions and deletions makes them indispensable in modern software systems.

**Key Takeaways:**

1. **Efficiency**: Heaps provide optimal time complexity for priority queue operations and enable $O(n)$ heap construction.

2. **Versatility**: From simple top-K problems to complex graph algorithms, heaps solve diverse computational challenges.

3. **Patterns**: Mastering common heap patterns (two heaps for median, heap for K-way problems, heap with hash map) enables elegant solutions to many algorithm problems.

4. **Trade-offs**: Understanding when to use heaps versus other data structures (BST, hash maps, arrays) is crucial for optimal algorithm design.

5. **Implementation**: While conceptually a tree, heaps are efficiently implemented as arrays, providing excellent cache performance.

Whether you're implementing a task scheduler, optimizing graph traversal, or processing streaming data, heaps offer a powerful tool in your algorithmic toolkit. Mastery of heap operations and patterns is essential for technical interviews, competitive programming, and building high-performance systems.
