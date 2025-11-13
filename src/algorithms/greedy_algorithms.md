# Greedy Algorithms

Greedy algorithms are a class of algorithms that make locally optimal choices at each stage with the hope of finding a global optimum. They are often used for optimization problems where a solution can be built incrementally.

## Key Concepts

- **Greedy Choice Property**: A global optimum can be reached by selecting a local optimum. This property is essential for the effectiveness of greedy algorithms.

- **Optimal Substructure**: A problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems.

## Activity Selection Problem

Select the maximum number of activities that don't overlap in time.

```python
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_finish = activities[0][1]
    
    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    
    return selected

# Example usage
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
result = activity_selection(activities)
print(f"Selected {len(result)} activities:")
for activity in result:
    print(f"  Start: {activity[0]}, Finish: {activity[1]}")
```

**Time Complexity**: $O(n \log n)$ for sorting
**Space Complexity**: $O(n)$

## Fractional Knapsack

Maximize value in knapsack by taking fractions of items.

```python
def fractional_knapsack(items, capacity):
    # Calculate value per weight and sort by it
    items_with_ratio = [(value, weight, value/weight) for value, weight in items]
    items_with_ratio.sort(key=lambda x: x[2], reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    taken = []
    
    for value, weight, ratio in items_with_ratio:
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
            break
    
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

**Time Complexity**: $O(n \log n)$

## Coin Change (Greedy - doesn't always work!)

Make change using minimum number of coins (works for standard coin systems).

```python
def coin_change_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    result = []
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
            result.append(coin)
    
    if amount > 0:
        return -1, []  # Cannot make exact change
    
    return count, result

# Example usage (US coins)
coins = [25, 10, 5, 1]
amount = 63
count, result = coin_change_greedy(coins, amount)
print(f"Minimum coins: {count}")
print(f"Coins used: {result}")  # [25, 25, 10, 1, 1, 1]
```

**Note**: Greedy doesn't always give optimal solution for arbitrary coin systems.
For example, with coins [1, 3, 4] and amount 6, greedy gives [4, 1, 1] (3 coins) 
but optimal is [3, 3] (2 coins).

## Huffman Coding

Optimal prefix-free encoding for data compression.

```python
import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text):
    # Count frequency
    freq = defaultdict(int)
    for char in text:
        freq[char] += 1
    
    # Create priority queue
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0]
    codes = {}
    
    def generate_codes(node, code):
        if node.char is not None:
            codes[node.char] = code
            return
        if node.left:
            generate_codes(node.left, code + '0')
        if node.right:
            generate_codes(node.right, code + '1')
    
    generate_codes(root, '')
    
    # Encode text
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

**Time Complexity**: $O(n \log n)$

## Job Sequencing

Maximize profit by scheduling jobs with deadlines.

```python
def job_sequencing(jobs):
    # Sort by profit (descending)
    jobs.sort(key=lambda x: x[2], reverse=True)
    
    # Find maximum deadline
    max_deadline = max(job[1] for job in jobs)
    
    # Create slot array
    slots = [-1] * max_deadline
    total_profit = 0
    scheduled_jobs = []
    
    # For each job, try to schedule it
    for job_id, deadline, profit in jobs:
        # Find a free slot before deadline
        for slot in range(min(max_deadline, deadline) - 1, -1, -1):
            if slots[slot] == -1:
                slots[slot] = job_id
                total_profit += profit
                scheduled_jobs.append((job_id, profit))
                break
    
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

**Time Complexity**: $O(n^2)$

## Minimum Spanning Tree - Prim's Algorithm

Find minimum spanning tree of a weighted graph.

```python
import heapq

def prim_mst(graph, start=0):
    n = len(graph)
    visited = set([start])
    edges = [(cost, start, to) for to, cost in graph[start]]
    heapq.heapify(edges)
    
    mst = []
    total_cost = 0
    
    while edges and len(visited) < n:
        cost, frm, to = heapq.heappop(edges)
        
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, cost))
            total_cost += cost
            
            for next_to, next_cost in graph[to]:
                if next_to not in visited:
                    heapq.heappush(edges, (next_cost, to, next_to))
    
    return mst, total_cost

# Example usage
# Graph as adjacency list: graph[node] = [(neighbor, weight), ...]
graph = [
    [(1, 2), (3, 6)],           # Node 0
    [(0, 2), (2, 3), (3, 8), (4, 5)],  # Node 1
    [(1, 3), (4, 7)],           # Node 2
    [(0, 6), (1, 8)],           # Node 3
    [(1, 5), (2, 7)]            # Node 4
]
mst, cost = prim_mst(graph)
print(f"Minimum spanning tree cost: {cost}")
print("Edges in MST:")
for frm, to, weight in mst:
    print(f"  {frm} -- {to} (weight: {weight})")
```

**Time Complexity**: $O(E \log V)$ with binary heap

## Minimum Spanning Tree - Kruskal's Algorithm

Another MST algorithm using Union-Find.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal_mst(n, edges):
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst = []
    total_cost = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_cost += weight
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

**Time Complexity**: $O(E \log E)$ or $O(E \log V)$

## Dijkstra's Shortest Path

Find shortest path from source to all other vertices.

```python
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        visited.add(u)
        
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
    print(f"  To node {i}: {d}")
```

**Time Complexity**: $O((V + E) \log V)$ with binary heap

## Gas Station Problem

Find starting station to complete circular route.

```python
def can_complete_circuit(gas, cost):
    n = len(gas)
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    # If total gas < total cost, impossible
    if total_gas < total_cost:
        return -1
    
    start = 0
    tank = 0
    
    for i in range(n):
        tank += gas[i] - cost[i]
        if tank < 0:
            # Can't reach next station from current start
            start = i + 1
            tank = 0
    
    return start

# Example usage
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
start = can_complete_circuit(gas, cost)
print(f"Start at station: {start}")  # Output: 3
```

**Time Complexity**: $O(n)$

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

Use greedy when:
1. Problem has greedy choice property
2. Problem has optimal substructure
3. Local optimum leads to global optimum

## Common Greedy Patterns

1. **Sorting first**: Many greedy algorithms start by sorting
2. **Priority queue**: Use heap for best choice at each step
3. **Intervals**: Scheduling problems often use greedy
4. **Graph traversal**: MST, shortest path

## Applications

Greedy algorithms are widely used in various applications, including:

- **Network Routing**: Finding the shortest path in a network (Dijkstra's algorithm)
- **Resource Allocation**: Distributing resources in a way that maximizes efficiency
- **Job Scheduling**: Scheduling jobs on machines to minimize completion time
- **Data Compression**: Huffman coding for optimal compression
- **Minimum Spanning Trees**: Network design problems

## Conclusion

Greedy algorithms provide a straightforward and efficient approach to solving optimization problems. While they do not always yield the optimal solution, they are often easier to implement and can be very effective for certain types of problems.
