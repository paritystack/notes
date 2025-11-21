# Greedy Algorithms

Greedy algorithms are a class of algorithms that make locally optimal choices at each stage with the hope of finding a global optimum. They are often used for optimization problems where a solution can be built incrementally.

## Key Concepts

### Greedy Choice Property

A problem exhibits the **greedy choice property** if we can make locally optimal choices at each step and still arrive at a globally optimal solution. This means:
- At each step, we make the choice that seems best at the moment
- We never reconsider or backtrack from previous choices
- The sequence of greedy choices leads to an optimal solution

**Proving Greedy Choice Property**: To prove a greedy algorithm is correct, we typically use one of these techniques:
1. **Exchange Argument**: Show that any optimal solution can be transformed into the greedy solution without making it worse
2. **Greedy Stays Ahead**: Prove that the greedy solution is always at least as good as any other solution at every step
3. **Structural Induction**: Show that if the greedy choice is optimal for smaller subproblems, it remains optimal when extended

### Optimal Substructure

A problem has **optimal substructure** if an optimal solution to the problem contains optimal solutions to its subproblems. This property is shared with dynamic programming, but the key difference is:
- **Greedy**: Make an irrevocable choice, then solve the remaining subproblem
- **Dynamic Programming**: Examine all choices, solve all resulting subproblems, then choose the best

### Greedy vs. Dynamic Programming

| Aspect | Greedy | Dynamic Programming |
|--------|--------|---------------------|
| Decision | Makes one choice at each step | Considers all choices at each step |
| Backtracking | Never backtracks | May reconsider previous choices |
| Efficiency | Usually O(n log n) or O(n) | Often O(n²) or worse |
| Correctness | Doesn't always work | Always finds optimal if applicable |
| Use when | Greedy choice property holds | Overlapping subproblems exist |

## Activity Selection Problem

**Problem**: Select the maximum number of activities that don't overlap in time.

**Intuition**:
- Greedy strategy: Always pick the activity that finishes earliest among remaining activities
- Why? Activities that finish early leave more room for subsequent activities
- Sorting by finish time ensures we can process activities in the optimal order

**Greedy Choice Property**: If we sort by finish time, selecting the first activity is always part of some optimal solution.

```python
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities.

    Args:
        activities: List of tuples (start_time, end_time)

    Returns:
        List of selected activities that don't overlap

    Time: O(n log n) - dominated by sorting
    Space: O(n) - for storing selected activities
    """
    if not activities:
        return []

    # Sort by finish time - greedy choice: pick earliest finisher
    activities.sort(key=lambda x: x[1])

    # Always select first activity (earliest to finish)
    selected = [activities[0]]
    last_finish = activities[0][1]

    # For remaining activities, select if compatible with last selected
    for start, finish in activities[1:]:
        if start >= last_finish:  # No overlap
            selected.append((start, finish))
            last_finish = finish

    return selected

# Example usage
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9),
              (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
result = activity_selection(activities)
print(f"Selected {len(result)} activities:")
for activity in result:
    print(f"  Start: {activity[0]}, Finish: {activity[1]}")
```

**Complexity Analysis**:
- **Time**: $O(n \log n)$ - dominated by sorting; iteration is $O(n)$
- **Space**: $O(n)$ - for storing selected activities list

## Fractional Knapsack

**Problem**: Maximize value in knapsack by taking fractions of items (items are divisible).

**Intuition**:
- Greedy strategy: Always take items with the highest value-to-weight ratio first
- Unlike 0/1 knapsack, we can take fractions, so greedy works perfectly
- Sort by value/weight ratio and greedily fill the knapsack

**Greedy Choice Property**: Taking the item with the highest value-per-unit-weight first is always optimal.

```python
def fractional_knapsack(items, capacity):
    """
    Maximize value in knapsack by taking fractions of items.

    Args:
        items: List of tuples (value, weight)
        capacity: Maximum weight capacity of knapsack

    Returns:
        Tuple of (max_value, items_taken)
        items_taken is list of (value, weight, fraction_taken)

    Time: O(n log n) - dominated by sorting
    Space: O(n) - for items_with_ratio and taken lists
    """
    if not items or capacity <= 0:
        return 0, []

    # Calculate value per weight and sort by it (descending)
    items_with_ratio = [(value, weight, value/weight) for value, weight in items]
    items_with_ratio.sort(key=lambda x: x[2], reverse=True)

    total_value = 0
    remaining_capacity = capacity
    taken = []

    for value, weight, ratio in items_with_ratio:
        if remaining_capacity <= 0:
            break

        if remaining_capacity >= weight:
            # Take full item
            total_value += value
            remaining_capacity -= weight
            taken.append((value, weight, 1.0))
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            taken.append((value, weight, fraction))
            remaining_capacity = 0  # Knapsack is full

    return total_value, taken

# Example usage
items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
capacity = 50
max_value, taken = fractional_knapsack(items, capacity)
print(f"Maximum value: {max_value}")
print("Items taken:")
for value, weight, fraction in taken:
    print(f"  Value={value}, Weight={weight}, Fraction={fraction:.2f}")
```

**Complexity Analysis**:
- **Time**: $O(n \log n)$ - sorting dominates; iteration is $O(n)$
- **Space**: $O(n)$ - for storing items with ratios and taken items

## Coin Change (Greedy - doesn't always work!)

**Problem**: Make change using minimum number of coins.

**Intuition**:
- Greedy strategy: Always use the largest coin possible
- Works for **canonical** coin systems (like US: 1, 5, 10, 25)
- Does NOT work for arbitrary coin systems!

**When Greedy Fails**: For coins [1, 3, 4] and amount 6:
- Greedy: 4 + 1 + 1 = 3 coins
- Optimal: 3 + 3 = 2 coins

```python
def coin_change_greedy(coins, amount):
    """
    Make change using minimum coins (greedy approach).
    WARNING: Only works for canonical coin systems!

    Args:
        coins: List of coin denominations
        amount: Target amount to make change for

    Returns:
        Tuple of (coin_count, coins_used)
        Returns (-1, []) if exact change impossible

    Time: O(n log n + amount/min_coin) - sorting + greedy selection
    Space: O(n) - for result list in worst case
    """
    coins.sort(reverse=True)  # Largest first
    count = 0
    result = []

    for coin in coins:
        # Use as many of this coin as possible
        num_coins = amount // coin
        if num_coins > 0:
            count += num_coins
            result.extend([coin] * num_coins)
            amount -= coin * num_coins

    if amount > 0:
        return -1, []  # Cannot make exact change

    return count, result

# Example 1: US coins (greedy works!)
coins = [25, 10, 5, 1]
amount = 63
count, result = coin_change_greedy(coins, amount)
print(f"Amount {amount}: {count} coins")
print(f"Coins used: {result}")  # [25, 25, 10, 1, 1, 1]

# Example 2: Non-canonical system (greedy fails!)
coins = [1, 3, 4]
amount = 6
count, result = coin_change_greedy(coins, amount)
print(f"\nAmount {amount}: {count} coins (greedy)")
print(f"Coins: {result}")  # [4, 1, 1] - 3 coins
print("Optimal would be [3, 3] - 2 coins!")
```

**Complexity Analysis**:
- **Time**: $O(n \log n + \frac{\text{amount}}{\text{min\_coin}})$ - sorting + greedy selection
- **Space**: $O(n)$ - for result list (worst case with many small coins)

**Note**: For arbitrary coin systems, use dynamic programming instead!

## Huffman Coding

**Problem**: Generate optimal prefix-free binary encoding for data compression.

**Intuition**:
- Greedy strategy: Build tree by repeatedly merging two lowest-frequency nodes
- More frequent characters get shorter codes, less frequent get longer codes
- Result is a prefix-free code (no code is prefix of another)

**Greedy Choice Property**: Merging the two lowest-frequency nodes first is always optimal.

```python
import heapq
from collections import defaultdict

class HuffmanNode:
    """Node in Huffman tree."""
    def __init__(self, char, freq):
        self.char = char    # Character (None for internal nodes)
        self.freq = freq    # Frequency of character(s)
        self.left = None    # Left child
        self.right = None   # Right child

    def __lt__(self, other):
        """Compare nodes by frequency for heap ordering."""
        return self.freq < other.freq

def huffman_encoding(text):
    """
    Generate Huffman encoding for given text.

    Args:
        text: String to encode

    Returns:
        Tuple of (encoded_string, character_codes, huffman_tree_root)

    Time: O(n + k log k) where n=text length, k=unique characters
    Space: O(k) for tree and codes
    """
    if not text:
        return "", {}, None

    # Count frequency of each character - O(n)
    freq = defaultdict(int)
    for char in text:
        freq[char] += 1

    # Special case: single unique character
    if len(freq) == 1:
        char = list(freq.keys())[0]
        codes = {char: '0'}
        encoded = '0' * len(text)
        return encoded, codes, HuffmanNode(char, freq[char])

    # Create priority queue with leaf nodes - O(k)
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)  # O(k)

    # Build Huffman tree - O(k log k)
    while len(heap) > 1:
        # Extract two nodes with minimum frequency
        left = heapq.heappop(heap)   # O(log k)
        right = heapq.heappop(heap)  # O(log k)

        # Create internal node with combined frequency
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)  # O(log k)

    # Generate codes by traversing tree - O(k)
    root = heap[0]
    codes = {}

    def generate_codes(node, code):
        """DFS to generate binary codes."""
        if node.char is not None:  # Leaf node
            codes[node.char] = code if code else '0'
            return
        if node.left:
            generate_codes(node.left, code + '0')
        if node.right:
            generate_codes(node.right, code + '1')

    generate_codes(root, '')

    # Encode text using generated codes - O(n)
    encoded = ''.join(codes[char] for char in text)

    return encoded, codes, root

# Example usage
text = "huffman coding example"
encoded, codes, tree = huffman_encoding(text)
print("Character codes:")
for char, code in sorted(codes.items()):
    print(f"  '{char}': {code}")
print(f"\nOriginal size: {len(text) * 8} bits")
print(f"Encoded size: {len(encoded)} bits")
print(f"Compression ratio: {len(encoded) / (len(text) * 8):.2%}")
```

**Complexity Analysis**:
- **Time**: $O(n + k \log k)$ where $n$ = text length, $k$ = unique characters
  - Frequency counting: $O(n)$
  - Building heap: $O(k)$
  - Tree construction: $O(k \log k)$
  - Encoding: $O(n)$
- **Space**: $O(k)$ - for tree nodes and code dictionary

## Job Sequencing

**Problem**: Schedule jobs with deadlines to maximize profit (each job takes 1 unit time).

**Intuition**:
- Greedy strategy: Sort jobs by profit (highest first), schedule each in latest possible slot
- Scheduling in latest slot leaves more options for other jobs
- Only profitable if we can schedule before deadline

**Greedy Choice Property**: Considering jobs in decreasing order of profit and scheduling in latest available slot is optimal.

```python
def job_sequencing(jobs):
    """
    Schedule jobs to maximize profit.

    Args:
        jobs: List of tuples (job_id, deadline, profit)
              Each job takes 1 unit time

    Returns:
        Tuple of (total_profit, scheduled_jobs)

    Time: O(n^2) - for each job, search for free slot
    Space: O(max_deadline) - for slots array
    """
    if not jobs:
        return 0, []

    # Sort by profit (descending) - greedy choice
    jobs.sort(key=lambda x: x[2], reverse=True)

    # Find maximum deadline to determine slot array size
    max_deadline = max(job[1] for job in jobs)

    # Create slot array to track which job is scheduled when
    slots = [-1] * max_deadline
    total_profit = 0
    scheduled_jobs = []

    # For each job (in profit order), try to schedule it
    for job_id, deadline, profit in jobs:
        # Find latest available slot before deadline
        for slot in range(min(max_deadline, deadline) - 1, -1, -1):
            if slots[slot] == -1:  # Slot is free
                slots[slot] = job_id
                total_profit += profit
                scheduled_jobs.append((job_id, profit))
                break  # Job scheduled, move to next

    return total_profit, scheduled_jobs

# Example usage
# Jobs: (job_id, deadline, profit)
jobs = [
    ('a', 2, 100),
    ('b', 1, 19),
    ('c', 2, 27),
    ('d', 1, 25),
    ('e', 3, 15)
]
profit, scheduled = job_sequencing(jobs)
print(f"Maximum profit: {profit}")
print("Scheduled jobs:")
for job_id, profit in scheduled:
    print(f"  Job {job_id}: ${profit}")
```

**Complexity Analysis**:
- **Time**: $O(n^2)$ - sorting is $O(n \log n)$, but slot search is $O(n \times \text{max\_deadline})$ which can be $O(n^2)$
- **Space**: $O(d)$ where $d$ = max deadline - for slots array

**Optimization**: Can be improved to $O(n \log n)$ using Disjoint Set Union (Union-Find)

## Minimum Spanning Tree - Prim's Algorithm

**Problem**: Find minimum spanning tree (MST) of a weighted connected graph.

**Intuition**:
- Greedy strategy: Start from any vertex, repeatedly add the minimum-weight edge that connects a new vertex
- Grows the MST one vertex at a time by always choosing the cheapest connection
- Uses priority queue to efficiently find minimum edge

**Greedy Choice Property**: The minimum-weight edge connecting MST to a non-MST vertex is always in some MST.

```python
import heapq

def prim_mst(graph, start=0):
    """
    Find minimum spanning tree using Prim's algorithm.

    Args:
        graph: Adjacency list where graph[u] = [(v, weight), ...]
        start: Starting vertex (default 0)

    Returns:
        Tuple of (mst_edges, total_cost)
        mst_edges is list of (from, to, weight)

    Time: O(E log V) with binary heap
    Space: O(V + E) for visited set and priority queue
    """
    n = len(graph)
    visited = set([start])
    # Priority queue: (cost, from_vertex, to_vertex)
    edges = [(cost, start, to) for to, cost in graph[start]]
    heapq.heapify(edges)

    mst = []
    total_cost = 0

    # Continue until all vertices visited or no more edges
    while edges and len(visited) < n:
        cost, frm, to = heapq.heappop(edges)

        # Skip if vertex already in MST
        if to in visited:
            continue

        # Add vertex to MST
        visited.add(to)
        mst.append((frm, to, cost))
        total_cost += cost

        # Add all edges from newly added vertex
        for next_to, next_cost in graph[to]:
            if next_to not in visited:
                heapq.heappush(edges, (next_cost, to, next_to))

    return mst, total_cost

# Example usage
# Graph as adjacency list: graph[node] = [(neighbor, weight), ...]
graph = [
    [(1, 2), (3, 6)],                      # Node 0
    [(0, 2), (2, 3), (3, 8), (4, 5)],      # Node 1
    [(1, 3), (4, 7)],                      # Node 2
    [(0, 6), (1, 8)],                      # Node 3
    [(1, 5), (2, 7)]                       # Node 4
]
mst, cost = prim_mst(graph)
print(f"Minimum spanning tree cost: {cost}")
print("Edges in MST:")
for frm, to, weight in mst:
    print(f"  {frm} -- {to} (weight: {weight})")
```

**Complexity Analysis**:
- **Time**: $O(E \log V)$ with binary heap
  - Each edge processed at most once: $O(E)$
  - Each heap operation: $O(\log V)$
- **Space**: $O(V + E)$ - visited set $O(V)$ + priority queue $O(E)$

## Minimum Spanning Tree - Kruskal's Algorithm

**Problem**: Find MST by considering edges in order of increasing weight.

**Intuition**:
- Greedy strategy: Sort all edges by weight, add edges if they don't create a cycle
- Uses Union-Find (Disjoint Set Union) to efficiently detect cycles
- Considers edges globally (unlike Prim's which grows from a single vertex)

**Greedy Choice Property**: Adding the minimum-weight edge that doesn't create a cycle is always safe.

```python
class UnionFind:
    """
    Disjoint Set Union (DSU) with path compression and union by rank.
    """
    def __init__(self, n):
        self.parent = list(range(n))  # Each node is its own parent initially
        self.rank = [0] * n           # Rank for union by rank optimization

    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Union two sets by rank.
        Returns True if sets were different (no cycle), False otherwise.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already in same set (would create cycle)

        # Union by rank: attach smaller tree under larger tree
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

def kruskal_mst(n, edges):
    """
    Find minimum spanning tree using Kruskal's algorithm.

    Args:
        n: Number of vertices
        edges: List of tuples (u, v, weight)

    Returns:
        Tuple of (mst_edges, total_cost)

    Time: O(E log E) for sorting edges
    Space: O(V) for Union-Find structure
    """
    # Sort edges by weight - greedy choice
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst = []
    total_cost = 0

    # Process edges in sorted order
    for u, v, weight in edges:
        # Add edge if it doesn't create a cycle
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_cost += weight
            # MST has exactly n-1 edges
            if len(mst) == n - 1:
                break

    return mst, total_cost

# Example usage
n = 5  # Number of vertices
edges = [
    (0, 1, 2), (0, 3, 6), (1, 2, 3),
    (1, 3, 8), (1, 4, 5), (2, 4, 7)
]
mst, cost = kruskal_mst(n, edges)
print(f"Minimum spanning tree cost: {cost}")
print("Edges in MST:")
for u, v, weight in mst:
    print(f"  {u} -- {v} (weight: {weight})")
```

**Complexity Analysis**:
- **Time**: $O(E \log E)$ or equivalently $O(E \log V)$ since $E \leq V^2$
  - Sorting edges: $O(E \log E)$
  - Union-Find operations: $O(E \cdot \alpha(V))$ where $\alpha$ is inverse Ackermann (nearly constant)
- **Space**: $O(V)$ - for Union-Find parent and rank arrays

## Dijkstra's Shortest Path

**Problem**: Find shortest paths from a source vertex to all other vertices (non-negative weights).

**Intuition**:
- Greedy strategy: Always explore the nearest unvisited vertex first
- Once we visit a vertex, we've found the shortest path to it
- Uses priority queue to efficiently get nearest vertex

**Greedy Choice Property**: The nearest unvisited vertex has its final shortest distance determined.

**Note**: Does NOT work with negative edge weights (use Bellman-Ford instead).

```python
import heapq

def dijkstra(graph, start):
    """
    Find shortest paths from start to all vertices using Dijkstra's algorithm.

    Args:
        graph: Adjacency list where graph[u] = [(v, weight), ...]
        start: Source vertex

    Returns:
        List of shortest distances from start to each vertex

    Time: O((V + E) log V) with binary heap
    Space: O(V) for distance array and visited set
    """
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)

        # Skip if already processed (may have duplicate entries)
        if u in visited:
            continue
        visited.add(u)

        # Relax all edges from u
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist

# Example usage
graph = [
    [(1, 4), (2, 1)],               # Node 0
    [(3, 1)],                       # Node 1
    [(1, 2), (3, 5)],               # Node 2
    [(4, 3)],                       # Node 3
    []                              # Node 4
]
distances = dijkstra(graph, 0)
print("Shortest distances from node 0:")
for i, d in enumerate(distances):
    if d == float('inf'):
        print(f"  To node {i}: unreachable")
    else:
        print(f"  To node {i}: {d}")
```

**Complexity Analysis**:
- **Time**: $O((V + E) \log V)$ with binary heap
  - Each vertex added to queue once: $O(V \log V)$
  - Each edge relaxed once: $O(E \log V)$
- **Space**: $O(V)$ - for distance array, visited set, and priority queue

**Optimization**: Can achieve $O(V \log V + E)$ with Fibonacci heap

## Gas Station Problem

**Problem**: Find starting gas station to complete a circular route, or determine if impossible.

**Intuition**:
- If total gas < total cost, impossible to complete circuit
- If possible, there exists exactly one valid starting station
- Greedy strategy: If we can't reach station `i+1` from start, then no station between start and `i` can work either (so try from `i+1`)

**Greedy Choice Property**: If sum(gas) ≥ sum(cost), the first station where we can't continue must be after the optimal start.

```python
def can_complete_circuit(gas, cost):
    """
    Find starting gas station to complete circular route.

    Args:
        gas: List of gas available at each station
        cost: List of gas needed to reach next station

    Returns:
        Starting station index, or -1 if impossible

    Time: O(n) - single pass
    Space: O(1) - only using constant extra space
    """
    n = len(gas)
    total_gas = sum(gas)
    total_cost = sum(cost)

    # If total gas < total cost, impossible to complete circuit
    if total_gas < total_cost:
        return -1

    # If we can complete circuit, find the starting point
    start = 0
    tank = 0

    for i in range(n):
        tank += gas[i] - cost[i]

        # If tank goes negative, we can't reach next station
        if tank < 0:
            # Key insight: none of the stations from start to i can work
            # So try starting from i+1
            start = i + 1
            tank = 0

    # If total_gas >= total_cost, 'start' is guaranteed to work
    return start

# Example usage
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
start = can_complete_circuit(gas, cost)
if start != -1:
    print(f"Start at station: {start}")  # Output: 3
else:
    print("No solution possible")
```

**Complexity Analysis**:
- **Time**: $O(n)$ - single pass through all stations
- **Space**: $O(1)$ - only constant extra space

**Why This Works**: If we can't reach station `j` starting from station `i`, then we also can't reach `j` starting from any station between `i` and `j-1`. This is because we would have less gas starting from a middle station than starting from `i`.

## Jump Game I

**Problem**: Determine if you can reach the last index, where each element represents maximum jump length.

**Intuition**:
- Greedy strategy: Track the farthest position we can reach
- If at any point our current position exceeds the farthest reachable, return false
- No need to try all possible jumps - just track maximum reach

**Greedy Choice Property**: We only need to know if the last index is reachable, not the actual path.

```python
def can_jump(nums):
    """
    Determine if we can reach the last index.

    Args:
        nums: List where nums[i] is max jump length from position i

    Returns:
        True if last index is reachable, False otherwise

    Time: O(n) - single pass
    Space: O(1) - constant space
    """
    if not nums:
        return False

    max_reach = 0  # Farthest index we can reach

    for i in range(len(nums)):
        # If current position is beyond our reach, we're stuck
        if i > max_reach:
            return False

        # Update farthest position we can reach from here
        max_reach = max(max_reach, i + nums[i])

        # Early exit if we can already reach the end
        if max_reach >= len(nums) - 1:
            return True

    return True

# Example usage
nums1 = [2, 3, 1, 1, 4]
print(f"Can reach end of {nums1}: {can_jump(nums1)}")  # True

nums2 = [3, 2, 1, 0, 4]
print(f"Can reach end of {nums2}: {can_jump(nums2)}")  # False
```

**Complexity Analysis**:
- **Time**: $O(n)$ - single pass through array
- **Space**: $O(1)$ - only tracking max_reach

## Jump Game II

**Problem**: Find minimum number of jumps to reach the last index (guaranteed reachable).

**Intuition**:
- Greedy strategy: Make jumps as late as possible to maximize options
- Track the farthest position reachable with current number of jumps
- When we reach the end of current jump range, increment jump count

**Greedy Choice Property**: Always jumping to the position that maximizes our next range is optimal.

```python
def jump(nums):
    """
    Find minimum number of jumps to reach last index.

    Args:
        nums: List where nums[i] is max jump length from position i

    Returns:
        Minimum number of jumps needed

    Time: O(n) - single pass
    Space: O(1) - constant space
    """
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0      # End of range for current jump
    farthest = 0         # Farthest position reachable

    # Don't need to check last index (we start there)
    for i in range(len(nums) - 1):
        # Update farthest position we can reach
        farthest = max(farthest, i + nums[i])

        # If we've reached the end of current jump range
        if i == current_end:
            jumps += 1
            current_end = farthest  # Start new jump range

            # Early exit if we can reach the end
            if current_end >= len(nums) - 1:
                break

    return jumps

# Example usage
nums = [2, 3, 1, 1, 4]
print(f"Minimum jumps for {nums}: {jump(nums)}")  # 2
# Explanation: Jump 1 step from index 0 to 1, then 3 steps to last index

nums2 = [2, 3, 0, 1, 4]
print(f"Minimum jumps for {nums2}: {jump(nums2)}")  # 2
```

**Complexity Analysis**:
- **Time**: $O(n)$ - single pass through array
- **Space**: $O(1)$ - only tracking a few variables

## Meeting Rooms II

**Problem**: Find minimum number of conference rooms needed to hold all meetings.

**Intuition**:
- Greedy strategy: Track how many meetings are active at any time
- Sort start times and end times separately
- When a meeting starts, check if a room is free (if a meeting has ended)

**Greedy Choice Property**: The minimum rooms needed equals the maximum number of overlapping meetings at any point.

```python
def min_meeting_rooms(intervals):
    """
    Find minimum number of conference rooms needed.

    Args:
        intervals: List of tuples (start_time, end_time)

    Returns:
        Minimum number of rooms needed

    Time: O(n log n) - sorting start and end times
    Space: O(n) - for start and end time arrays
    """
    if not intervals:
        return 0

    # Separate start and end times
    start_times = sorted([interval[0] for interval in intervals])
    end_times = sorted([interval[1] for interval in intervals])

    rooms_needed = 0
    rooms_available = 0
    start_ptr = 0
    end_ptr = 0

    # Process all events
    while start_ptr < len(intervals):
        # If a meeting starts before earliest ending meeting
        if start_times[start_ptr] < end_times[end_ptr]:
            # Need a new room
            rooms_needed += 1
            start_ptr += 1
        else:
            # A room becomes available
            rooms_available += 1
            end_ptr += 1

    return rooms_needed - rooms_available

# Alternative implementation using heap
import heapq

def min_meeting_rooms_heap(intervals):
    """
    Find minimum conference rooms using min heap.

    Time: O(n log n) - sorting + heap operations
    Space: O(n) - heap size
    """
    if not intervals:
        return 0

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Min heap to track end times of ongoing meetings
    heap = []

    for start, end in intervals:
        # If earliest ending meeting has ended, remove it
        if heap and heap[0] <= start:
            heapq.heappop(heap)

        # Add current meeting's end time
        heapq.heappush(heap, end)

    # Heap size = number of rooms needed
    return len(heap)

# Example usage
meetings = [(0, 30), (5, 10), (15, 20)]
print(f"Meetings: {meetings}")
print(f"Rooms needed: {min_meeting_rooms(meetings)}")  # 2
print(f"Rooms needed (heap): {min_meeting_rooms_heap(meetings)}")  # 2

meetings2 = [(7, 10), (2, 4)]
print(f"\nMeetings: {meetings2}")
print(f"Rooms needed: {min_meeting_rooms_heap(meetings2)}")  # 1
```

**Complexity Analysis**:
- **Time**: $O(n \log n)$ - sorting dominates
- **Space**: $O(n)$ - for sorted arrays or heap

## Minimum Arrows to Burst Balloons

**Problem**: Find minimum arrows needed to burst all balloons (given as intervals).

**Intuition**:
- Similar to Activity Selection but we want minimum resources to cover all intervals
- Greedy strategy: Sort by end position, shoot arrow at end of each balloon
- One arrow can burst all balloons that overlap at that position

**Greedy Choice Property**: Shooting at the earliest end position covers maximum balloons.

```python
def find_min_arrows(points):
    """
    Find minimum arrows to burst all balloons.

    Args:
        points: List of [start, end] representing balloon diameters

    Returns:
        Minimum number of arrows needed

    Time: O(n log n) - sorting
    Space: O(1) - constant extra space
    """
    if not points:
        return 0

    # Sort by end position - greedy choice
    points.sort(key=lambda x: x[1])

    arrows = 1
    current_arrow_pos = points[0][1]  # Shoot at end of first balloon

    for start, end in points[1:]:
        # If current balloon starts after our arrow position
        if start > current_arrow_pos:
            # Need a new arrow
            arrows += 1
            current_arrow_pos = end  # Shoot at end of this balloon

    return arrows

# Example usage
balloons = [[10, 16], [2, 8], [1, 6], [7, 12]]
print(f"Balloons: {balloons}")
print(f"Minimum arrows: {find_min_arrows(balloons)}")  # 2
# Shoot at position 6 (bursts [2,8] and [1,6])
# Shoot at position 12 (bursts [10,16] and [7,12])

balloons2 = [[1, 2], [3, 4], [5, 6], [7, 8]]
print(f"\nBalloons: {balloons2}")
print(f"Minimum arrows: {find_min_arrows(balloons2)}")  # 4
```

**Complexity Analysis**:
- **Time**: $O(n \log n)$ - sorting dominates
- **Space**: $O(1)$ - only tracking arrows count and position

## Task Scheduler

**Problem**: Schedule tasks with cooldown period `n` between same tasks. Find minimum time units needed.

**Intuition**:
- Greedy strategy: Schedule most frequent tasks first to minimize idle time
- Create "frames" of size `n+1` where each task appears at most once per frame
- Fill frames with most frequent tasks first, then less frequent ones

**Greedy Choice Property**: Scheduling most frequent tasks first and spacing them optimally minimizes total time.

```python
from collections import Counter

def least_interval(tasks, n):
    """
    Find minimum time units needed to complete all tasks.

    Args:
        tasks: List of task names (e.g., ['A', 'A', 'B', 'B', 'C'])
        n: Cooldown period - must wait n intervals between same tasks

    Returns:
        Minimum number of time units needed

    Time: O(k log k) where k = number of unique tasks (max 26)
          In practice O(n) since k is bounded
    Space: O(k) for frequency counter
    """
    if n == 0:
        return len(tasks)

    # Count frequency of each task
    freq = Counter(tasks)
    max_freq = max(freq.values())

    # Count how many tasks have the maximum frequency
    max_freq_count = sum(1 for f in freq.values() if f == max_freq)

    # Calculate minimum time needed
    # Formula: (max_freq - 1) * (n + 1) + max_freq_count
    # Explanation:
    # - We create (max_freq - 1) complete frames of size (n + 1)
    # - Plus one final slot for each max-frequency task
    min_time = (max_freq - 1) * (n + 1) + max_freq_count

    # Answer is max of calculated time or total tasks
    # (if cooldown allows, we can execute continuously)
    return max(min_time, len(tasks))

# Example usage
tasks1 = ['A', 'A', 'A', 'B', 'B', 'B']
n1 = 2
print(f"Tasks: {tasks1}, n={n1}")
print(f"Minimum time: {least_interval(tasks1, n1)}")  # 8
# Explanation: A -> B -> idle -> A -> B -> idle -> A -> B

tasks2 = ['A', 'A', 'A', 'B', 'B', 'B']
n2 = 0
print(f"\nTasks: {tasks2}, n={n2}")
print(f"Minimum time: {least_interval(tasks2, n2)}")  # 6

tasks3 = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
n3 = 2
print(f"\nTasks: {tasks3}, n={n3}")
print(f"Minimum time: {least_interval(tasks3, n3)}")  # 16
```

**Complexity Analysis**:
- **Time**: $O(n)$ where $n$ is number of tasks (counter operations + constant work for 26 letters max)
- **Space**: $O(1)$ - at most 26 different tasks (can be considered constant)

## Partition Labels

**Problem**: Partition a string into as many parts as possible such that each letter appears in at most one part.

**Intuition**:
- Greedy strategy: Track the last occurrence of each character
- Extend current partition until we've included all characters' last occurrences
- Start new partition when current partition is complete

**Greedy Choice Property**: Making a partition as soon as all characters are complete is optimal.

```python
def partition_labels(s):
    """
    Partition string into maximum parts where each char appears in one part.

    Args:
        s: Input string

    Returns:
        List of partition lengths

    Time: O(n) - two passes through string
    Space: O(1) - at most 26 characters (constant space)
    """
    # Record last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}

    partitions = []
    start = 0
    end = 0

    for i, char in enumerate(s):
        # Extend partition to include this character's last occurrence
        end = max(end, last_occurrence[char])

        # If we've reached the end of current partition
        if i == end:
            partitions.append(end - start + 1)
            start = i + 1  # Start new partition

    return partitions

# Example usage
s1 = "ababcbacadefegdehijhklij"
print(f"String: {s1}")
print(f"Partitions: {partition_labels(s1)}")  # [9, 7, 8]
# Explanation: "ababcbaca", "defegde", "hijhklij"

s2 = "eccbbbbdec"
print(f"\nString: {s2}")
print(f"Partitions: {partition_labels(s2)}")  # [10]
# Explanation: entire string is one partition
```

**Complexity Analysis**:
- **Time**: $O(n)$ - one pass to find last occurrences, one pass to partition
- **Space**: $O(1)$ - at most 26 characters in dictionary (constant)

## Greedy vs Dynamic Programming

Some problems can be solved by both approaches:

```python
# Greedy (doesn't always work)
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1

# Dynamic Programming (always correct)
def coin_change_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

## When to Use Greedy

### Problems Where Greedy Works

| Problem Type | Key Characteristic | Example |
|--------------|-------------------|---------|
| **Interval Scheduling** | Maximize non-overlapping intervals | Activity Selection, Meeting Rooms |
| **Shortest Path** | Non-negative weights | Dijkstra's Algorithm |
| **Spanning Trees** | Minimum weight connections | Prim's, Kruskal's |
| **Huffman Coding** | Optimal prefix-free codes | Data Compression |
| **Fractional Problems** | Can take fractions | Fractional Knapsack |
| **Array Problems** | Maximize/minimize with greedy choice | Jump Game, Gas Station |

### Problems Where Greedy Fails

| Problem Type | Why Greedy Fails | Correct Approach |
|--------------|------------------|------------------|
| **0/1 Knapsack** | Can't take fractions; greedy by ratio fails | Dynamic Programming |
| **Coin Change (arbitrary)** | Greedy doesn't consider all combinations | Dynamic Programming |
| **Longest Path** | Negative weights or need max path | Dynamic Programming |
| **Subset Sum** | Need exact sum, not greedy accumulation | Dynamic Programming |
| **Edit Distance** | Many overlapping subproblems | Dynamic Programming |

### Checklist: Should I Use Greedy?

1. **Greedy Choice Property**: Can I make a locally optimal choice?
2. **Optimal Substructure**: Does optimal solution contain optimal subsolutions?
3. **No Backtracking Needed**: Once I make a choice, do I never reconsider it?
4. **Proof**: Can I prove greedy works via exchange argument or greedy-stays-ahead?

If YES to all → Try Greedy
If NO → Consider DP, Backtracking, or other approaches

## Common Greedy Patterns

1. **Sorting first**: Many greedy algorithms start by sorting
2. **Priority queue**: Use heap for best choice at each step
3. **Intervals**: Scheduling problems often use greedy
4. **Graph traversal**: MST, shortest path

## Applications

### Real-World Applications of Greedy Algorithms

| Domain | Application | Algorithm Used | Impact |
|--------|-------------|----------------|--------|
| **Networking** | Internet routing (OSPF, RIP) | Dijkstra's Shortest Path | Fast packet routing |
| **Data Compression** | ZIP, JPEG, MP3 | Huffman Coding | File size reduction |
| **Operating Systems** | Process scheduling | Greedy scheduling | CPU utilization |
| **Cloud Computing** | Resource allocation | Greedy bin packing | Cost optimization |
| **Transportation** | GPS navigation | Dijkstra's/A* | Route planning |
| **Telecommunications** | Network design | MST (Prim's/Kruskal's) | Minimize cable cost |
| **Finance** | Portfolio optimization | Greedy selection | Maximize returns |
| **Manufacturing** | Job scheduling on machines | Greedy job sequencing | Minimize completion time |
| **Game Development** | AI pathfinding | Greedy best-first search | Real-time decisions |

### Specific Examples

1. **Dijkstra's in GPS Systems**
   - Google Maps, Waze use variants of Dijkstra's algorithm
   - Finds shortest routes between locations
   - Modified for real-time traffic data

2. **Huffman Coding in Compression**
   - Used in ZIP, GZIP, JPEG, MP3
   - Assigns shorter codes to frequent characters
   - Achieves optimal prefix-free compression

3. **Greedy Scheduling in Operating Systems**
   - CPU scheduling algorithms (Shortest Job First)
   - Memory management (Best Fit, First Fit)
   - Disk scheduling (SCAN, C-SCAN)

4. **MST in Network Design**
   - Telecommunications network layout
   - Electrical grid design
   - Minimizes total cable/wire length

5. **Activity Selection in Resource Management**
   - Conference room scheduling
   - Meeting room allocation
   - Classroom assignment

## Conclusion

Greedy algorithms provide an elegant and efficient approach to solving optimization problems. Their key advantages are:

- **Simplicity**: Easy to understand and implement
- **Efficiency**: Often $O(n \log n)$ or better
- **Practicality**: Work well for many real-world problems

However, remember that greedy algorithms don't always guarantee optimal solutions. Always verify the greedy choice property and optimal substructure before applying greedy approach. When greedy fails, consider:
- **Dynamic Programming**: For overlapping subproblems
- **Backtracking**: When you need to explore all possibilities
- **Branch and Bound**: For optimization with constraints

**Key Takeaway**: Greedy algorithms are powerful when applicable, but require careful analysis to ensure correctness.
