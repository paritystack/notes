# Graphs

Graphs are a fundamental data structure used to represent relationships between pairs of objects. They consist of vertices (or nodes) and edges (connections between the nodes). Graphs can be directed or undirected, weighted or unweighted, and are widely used in various applications such as social networks, transportation systems, and computer networks.

## Graph Fundamentals

### Core Components

- **Vertices (Nodes)**: The individual elements in a graph. Example: cities, people, web pages
- **Edges**: The connections between vertices representing relationships or paths
- **Adjacent Vertices**: Two vertices connected by an edge are adjacent (neighbors)
- **Degree**: Number of edges connected to a vertex
  - **In-degree**: Number of incoming edges (directed graphs)
  - **Out-degree**: Number of outgoing edges (directed graphs)
- **Path**: A sequence of vertices connected by edges
- **Simple Path**: A path with no repeated vertices
- **Cycle**: A path that starts and ends at the same vertex
- **Connected Graph**: A graph where there's a path between every pair of vertices
- **Weighted Edge**: An edge with an associated numerical value (weight/cost)

### Graph Types

**Directed vs Undirected**:
```
Directed (Digraph):         Undirected:
    A → B → C                  A — B — C
    ↓   ↓                      |   |
    D → E                      D — E
```

**Weighted vs Unweighted**:
```
Weighted:                   Unweighted:
    A -5→ B                    A → B
    ↓2    ↓3                   ↓   ↓
    C -1→ D                    C → D
```

**Graph Classifications**:
- **Directed Acyclic Graph (DAG)**: Directed graph with no cycles (e.g., task dependencies)
- **Complete Graph**: Every pair of vertices is connected
- **Bipartite Graph**: Vertices can be divided into two disjoint sets with no edges within sets
- **Sparse Graph**: Few edges relative to vertices (E << V²)
- **Dense Graph**: Many edges relative to vertices (E ≈ V²)
- **Multigraph**: Multiple edges between same pair of vertices
- **Self-loop**: An edge from a vertex to itself

### Graph vs Tree

| Property | Tree | General Graph |
|----------|------|---------------|
| Edges | V - 1 | Any number |
| Cycles | No | May have |
| Root | Yes | No |
| Parent-Child | Yes | No |
| Connected | Always | Maybe |

## Graph Representations

### 1. Adjacency Matrix

A 2D matrix where `matrix[i][j]` indicates edge from vertex i to vertex j.

```python
class GraphAdjacencyMatrix:
    def __init__(self, num_vertices, directed=False):
        """
        Initialize graph with adjacency matrix.
        Space: $O(V^2)$
        """
        self.num_vertices = num_vertices
        self.directed = directed
        # Use float('inf') for no edge in weighted graphs, 0 for unweighted
        self.matrix = [[0] * num_vertices for _ in range(num_vertices)]

    def add_edge(self, u, v, weight=1):
        """
        Add edge from u to v.
        Time: $O(1)$
        """
        self.matrix[u][v] = weight
        if not self.directed:
            self.matrix[v][u] = weight

    def remove_edge(self, u, v):
        """
        Remove edge from u to v.
        Time: $O(1)$
        """
        self.matrix[u][v] = 0
        if not self.directed:
            self.matrix[v][u] = 0

    def has_edge(self, u, v):
        """
        Check if edge exists.
        Time: $O(1)$
        """
        return self.matrix[u][v] != 0

    def get_neighbors(self, v):
        """
        Get all neighbors of vertex v.
        Time: $O(V)$
        """
        neighbors = []
        for i in range(self.num_vertices):
            if self.matrix[v][i] != 0:
                neighbors.append((i, self.matrix[v][i]))
        return neighbors

    def display(self):
        """Display the adjacency matrix."""
        for row in self.matrix:
            print(row)

# Example
graph = GraphAdjacencyMatrix(4, directed=False)
graph.add_edge(0, 1, 1)
graph.add_edge(0, 2, 1)
graph.add_edge(1, 2, 1)
graph.add_edge(2, 3, 1)

# Matrix representation:
# [[0, 1, 1, 0],
#  [1, 0, 1, 0],
#  [1, 1, 0, 1],
#  [0, 0, 1, 0]]
```

**When to use Adjacency Matrix**:
- Dense graphs (many edges)
- Need $O(1)$ edge lookup
- Need to quickly check if edge exists
- Graph size is small (memory intensive for large graphs)

### 2. Adjacency List

A collection of lists where each vertex has a list of its neighbors.

```python
from collections import defaultdict

class GraphAdjacencyList:
    def __init__(self, directed=False):
        """
        Initialize graph with adjacency list.
        Space: $O(V + E)$
        """
        self.graph = defaultdict(list)
        self.directed = directed

    def add_vertex(self, v):
        """
        Add a vertex to the graph.
        Time: $O(1)$
        """
        if v not in self.graph:
            self.graph[v] = []

    def add_edge(self, u, v, weight=1):
        """
        Add edge from u to v with optional weight.
        Time: $O(1)$
        """
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

    def remove_edge(self, u, v):
        """
        Remove edge from u to v.
        Time: $O(V)$ - need to find and remove
        """
        self.graph[u] = [(node, weight) for node, weight in self.graph[u] if node != v]
        if not self.directed:
            self.graph[v] = [(node, weight) for node, weight in self.graph[v] if node != u]

    def has_edge(self, u, v):
        """
        Check if edge exists from u to v.
        Time: $O(degree(u))$
        """
        return any(node == v for node, _ in self.graph[u])

    def get_neighbors(self, v):
        """
        Get all neighbors of vertex v.
        Time: $O(1)$ to access, $O(degree(v))$ to iterate
        """
        return self.graph[v]

    def get_all_vertices(self):
        """Get all vertices in the graph."""
        return list(self.graph.keys())

    def display(self):
        """Display the adjacency list."""
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")

# Example
graph = GraphAdjacencyList(directed=False)
graph.add_edge('A', 'B', 5)
graph.add_edge('A', 'C', 3)
graph.add_edge('B', 'C', 2)
graph.add_edge('C', 'D', 1)

# Adjacency list representation:
# A: [(B, 5), (C, 3)]
# B: [(A, 5), (C, 2)]
# C: [(A, 3), (B, 2), (D, 1)]
# D: [(C, 1)]
```

**When to use Adjacency List**:
- Sparse graphs (most common case)
- Need to iterate through neighbors efficiently
- Space efficiency is important
- Most graph algorithms (DFS, BFS, Dijkstra, etc.)

### 3. Edge List

A simple list of all edges in the graph.

```python
class GraphEdgeList:
    def __init__(self, directed=False):
        """
        Initialize graph with edge list.
        Space: $O(E)$
        """
        self.edges = []  # List of (u, v, weight) tuples
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        """
        Add edge to the list.
        Time: $O(1)$
        """
        self.edges.append((u, v, weight))
        if not self.directed:
            self.edges.append((v, u, weight))

    def get_edges(self):
        """Return all edges."""
        return self.edges

    def sort_by_weight(self):
        """
        Sort edges by weight (useful for Kruskal's algorithm).
        Time: $O(E \log E)$
        """
        self.edges.sort(key=lambda x: x[2])

    def display(self):
        """Display all edges."""
        for u, v, weight in self.edges:
            print(f"{u} -> {v} (weight: {weight})")

# Example
graph = GraphEdgeList(directed=True)
graph.add_edge('A', 'B', 5)
graph.add_edge('A', 'C', 3)
graph.add_edge('B', 'D', 2)

# Edge list:
# [('A', 'B', 5), ('A', 'C', 3), ('B', 'D', 2)]
```

**When to use Edge List**:
- Kruskal's MST algorithm
- Simple edge processing
- When you primarily work with edges rather than vertices

### Representation Comparison

| Operation | Adjacency Matrix | Adjacency List | Edge List |
|-----------|-----------------|----------------|-----------|
| **Space** | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| **Add edge** | $O(1)$ | $O(1)$ | $O(1)$ |
| **Remove edge** | $O(1)$ | $O(V)$ | $O(E)$ |
| **Has edge** | $O(1)$ | $O(degree)$ | $O(E)$ |
| **Get neighbors** | $O(V)$ | $O(degree)$ | $O(E)$ |
| **Best for** | Dense graphs | Sparse graphs | Edge operations |

## Graph Traversal Algorithms

### 1. Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking. Uses a stack (or recursion).

#### Recursive DFS

```python
def dfs_recursive(graph, start, visited=None):
    """
    DFS recursive traversal.
    Time: $O(V + E)$, Space: $O(V)$ for visited set + $O(h)$ recursion stack

    Args:
        graph: Dictionary where graph[v] = list of neighbors
        start: Starting vertex
        visited: Set of visited vertices

    Returns:
        List of vertices in DFS order
    """
    if visited is None:
        visited = set()

    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        # Extract vertex if stored as (vertex, weight) tuple
        next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
        if next_vertex not in visited:
            result.extend(dfs_recursive(graph, next_vertex, visited))

    return result

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(dfs_recursive(graph, 'A'))  # ['A', 'B', 'D', 'E', 'F', 'C']
```

**Visual Example**:
```
Graph:       A
           /   \
          B     C
         / \     \
        D   E     F
             \   /
              [F already visited]

DFS Order: A → B → D → E → F → C
Stack trace:
  dfs(A) → dfs(B) → dfs(D) → [return] → dfs(E) → dfs(F) → [return] → [return] → dfs(C) → [F visited] → [return]
```

#### Iterative DFS

```python
def dfs_iterative(graph, start):
    """
    DFS iterative using explicit stack.
    Time: $O(V + E)$, Space: $O(V)$

    More control than recursion, avoids stack overflow for deep graphs.
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            # Add neighbors to stack (in reverse order to match recursive DFS)
            neighbors = graph[vertex]
            for neighbor in reversed(neighbors):
                next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if next_vertex not in visited:
                    stack.append(next_vertex)

    return result
```

#### DFS with Path Tracking

```python
def dfs_find_path(graph, start, end, path=None):
    """
    Find a path from start to end using DFS.
    Time: $O(V + E)$, Space: $O(V)$
    """
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return path

    for neighbor in graph[start]:
        next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
        if next_vertex not in path:  # Avoid cycles
            new_path = dfs_find_path(graph, next_vertex, end, path)
            if new_path:
                return new_path

    return None  # No path found

def dfs_all_paths(graph, start, end, path=None):
    """
    Find all paths from start to end using DFS.
    Time: $O(V!)$ worst case (exponential)
    """
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return [path]

    paths = []
    for neighbor in graph[start]:
        next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
        if next_vertex not in path:  # Avoid cycles
            new_paths = dfs_all_paths(graph, next_vertex, end, path)
            paths.extend(new_paths)

    return paths

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
    'E': []
}

print(dfs_find_path(graph, 'A', 'E'))     # ['A', 'B', 'D', 'E']
print(dfs_all_paths(graph, 'A', 'E'))    # [['A', 'B', 'D', 'E'], ['A', 'C', 'D', 'E']]
```

#### DFS Applications

1. **Cycle Detection** in directed/undirected graphs
2. **Topological Sort** for DAGs
3. **Finding Connected Components**
4. **Path Finding** (not guaranteed shortest)
5. **Maze Solving**
6. **Strongly Connected Components** (Tarjan's, Kosaraju's)
7. **Backtracking Problems** (puzzles, games)

### 2. Breadth-First Search (BFS)

BFS explores all neighbors at the current depth before moving to the next level. Uses a queue.

#### Standard BFS

```python
from collections import deque

def bfs(graph, start):
    """
    BFS traversal.
    Time: $O(V + E)$, Space: $O(V)$

    Args:
        graph: Dictionary where graph[v] = list of neighbors
        start: Starting vertex

    Returns:
        List of vertices in BFS order
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph[vertex]:
            next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if next_vertex not in visited:
                visited.add(next_vertex)
                queue.append(next_vertex)

    return result

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

print(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']
```

**Visual Example**:
```
Graph:       A
           /   \
          B     C
         / \     \
        D   E     F

BFS by Level:
Level 0: [A]
Level 1: [B, C]
Level 2: [D, E, F]

BFS Order: A → B → C → D → E → F

Queue trace:
[A] → process A, add B,C → [B,C]
[B,C] → process B, add D,E → [C,D,E]
[C,D,E] → process C, add F → [D,E,F]
[D,E,F] → process D → [E,F]
[E,F] → process E (F already visited) → [F]
[F] → process F → []
```

#### BFS with Level Tracking

```python
def bfs_by_level(graph, start):
    """
    BFS that returns nodes grouped by level (distance from start).
    Time: $O(V + E)$, Space: $O(V)$

    Useful for:
    - Finding all nodes at distance k
    - Level-order processing
    - Visualizing graph structure
    """
    visited = set([start])
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []

        # Process all nodes at current level
        for _ in range(level_size):
            vertex = queue.popleft()
            current_level.append(vertex)

            for neighbor in graph[vertex]:
                next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if next_vertex not in visited:
                    visited.add(next_vertex)
                    queue.append(next_vertex)

        levels.append(current_level)

    return levels

# Example
print(bfs_by_level(graph, 'A'))
# [['A'], ['B', 'C'], ['D', 'E', 'F']]
```

#### BFS Shortest Path

```python
def bfs_shortest_path(graph, start, end):
    """
    Find shortest path in unweighted graph using BFS.
    Time: $O(V + E)$, Space: $O(V)$

    BFS guarantees shortest path in unweighted graphs!
    """
    if start == end:
        return [start]

    visited = {start}
    queue = deque([(start, [start])])  # (vertex, path)

    while queue:
        vertex, path = queue.popleft()

        for neighbor in graph[vertex]:
            next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor

            if next_vertex == end:
                return path + [next_vertex]

            if next_vertex not in visited:
                visited.add(next_vertex)
                queue.append((next_vertex, path + [next_vertex]))

    return None  # No path found

def bfs_shortest_distance(graph, start):
    """
    Find shortest distance from start to all other vertices.
    Time: $O(V + E)$, Space: $O(V)$

    Returns dictionary: {vertex: distance}
    """
    distances = {start: 0}
    visited = {start}
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        current_dist = distances[vertex]

        for neighbor in graph[vertex]:
            next_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if next_vertex not in visited:
                visited.add(next_vertex)
                distances[next_vertex] = current_dist + 1
                queue.append(next_vertex)

    return distances

# Example
print(bfs_shortest_path(graph, 'A', 'F'))  # ['A', 'C', 'F']
print(bfs_shortest_distance(graph, 'A'))   # {'A': 0, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2}
```

#### BFS Applications

1. **Shortest Path** in unweighted graphs
2. **Level-order Processing**
3. **Finding Connected Components**
4. **Testing Bipartiteness** (2-coloring)
5. **Finding all nodes within k distance**
6. **Web Crawling** (page rank)
7. **Social Network Analysis** (degrees of separation)

### DFS vs BFS Comparison

| Aspect | DFS | BFS |
|--------|-----|-----|
| **Data Structure** | Stack (recursion or explicit) | Queue |
| **Space Complexity** | $O(h)$ where h = height/depth | $O(w)$ where w = max width |
| **Path Found** | May not be shortest | **Shortest** in unweighted graphs |
| **When to Use** | Cycle detection, topological sort, exhaustive search | Shortest path, level-order, minimum steps |
| **Implementation** | Usually simpler (recursive) | Requires queue |
| **Memory** | Can be deep (stack overflow risk) | Can be wide (more memory for wide graphs) |

**Rule of Thumb**:
- Use **BFS** when you need the shortest path or want to explore nearby nodes first
- Use **DFS** when you need to explore all possibilities or detect cycles

## Shortest Path Algorithms

### 1. Dijkstra's Algorithm

Finds shortest paths from a source vertex to all other vertices in a graph with **non-negative** edge weights.

```python
import heapq

def dijkstra(graph, start):
    """
    Dijkstra's algorithm for shortest paths from start to all vertices.

    Time: $O((V + E) \log V)$ with binary heap
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [(v, weight), ...] (adjacency list with weights)
        start: Starting vertex

    Returns:
        Dictionary {vertex: shortest_distance}

    Requirements:
        - All edge weights must be non-negative
        - Graph can be directed or undirected
    """
    # Initialize distances to infinity
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        # Skip if already visited (outdated entry in pq)
        if current in visited:
            continue

        visited.add(current)

        # Relax edges
        for neighbor, weight in graph[current]:
            distance = current_dist + weight

            # If found shorter path, update
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances


def dijkstra_with_path(graph, start, end):
    """
    Dijkstra's algorithm with path reconstruction.

    Returns: (shortest_distance, path)
    """
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    previous = {vertex: None for vertex in graph}

    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        # Early termination if reached destination
        if current == end:
            break

        if current in visited:
            continue

        visited.add(current)

        for neighbor, weight in graph[current]:
            distance = current_dist + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    # Return None if no path exists
    if path[0] != start:
        return float('inf'), None

    return distances[end], path


# Example
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 5)],
    'C': [('B', 1), ('D', 8)],
    'D': [('E', 3)],
    'E': []
}

print(dijkstra(graph, 'A'))
# {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 11}

print(dijkstra_with_path(graph, 'A', 'E'))
# (11, ['A', 'C', 'B', 'D', 'E'])
```

**Visual Example**:
```
Graph:
    A --4--> B --5--> D --3--> E
    |       ^         ^
    2       1         8
    |       |         |
    └-----> C --------┘

Step-by-step execution:
Initial: distances = {A:0, B:∞, C:∞, D:∞, E:∞}, pq = [(0,A)]

1. Visit A (dist=0):
   - Update B: 0+4=4
   - Update C: 0+2=2
   distances = {A:0, B:4, C:2, D:∞, E:∞}
   pq = [(2,C), (4,B)]

2. Visit C (dist=2):
   - Update B: 2+1=3 (better than 4!)
   - Update D: 2+8=10
   distances = {A:0, B:3, C:2, D:10, E:∞}
   pq = [(3,B), (4,B), (10,D)]

3. Visit B (dist=3):
   - Update D: 3+5=8 (better than 10!)
   distances = {A:0, B:3, C:2, D:8, E:∞}
   pq = [(4,B), (8,D), (10,D)]

4. Skip B (dist=4): already visited

5. Visit D (dist=8):
   - Update E: 8+3=11
   distances = {A:0, B:3, C:2, D:8, E:11}
   pq = [(10,D), (11,E)]

6. Skip D (dist=10): already visited

7. Visit E (dist=11):
   distances = {A:0, B:3, C:2, D:8, E:11}

Final shortest paths from A:
A→A: 0
A→B: 3 (path: A→C→B)
A→C: 2 (path: A→C)
A→D: 8 (path: A→C→B→D)
A→E: 11 (path: A→C→B→D→E)
```

**Why Dijkstra Doesn't Work with Negative Weights**:
```
    A --1--> B
    |       ↓
    5     -10
    |       ↓
    └-----> C

Dijkstra would visit in order: A, B, C
- Visit A: distances = {A:0, B:1, C:5}
- Visit B (already finalized B=1): updates C to -9
- Visit C: Too late! B already visited with wrong distance

Correct: A→B→C = 1+(-10) = -9 is shorter than A→C = 5
But Dijkstra found A→B = 1 and marked it final before seeing the negative edge!
```

### 2. Bellman-Ford Algorithm

Finds shortest paths from a source vertex, works with **negative edge weights**, and detects negative cycles.

```python
def bellman_ford(graph, start):
    """
    Bellman-Ford algorithm for shortest paths.

    Time: $O(VE)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [(v, weight), ...]
        start: Starting vertex

    Returns:
        Dictionary {vertex: shortest_distance}

    Raises:
        ValueError: If negative cycle is detected

    Advantages over Dijkstra:
        - Works with negative edge weights
        - Detects negative cycles

    Disadvantages:
        - Slower: $O(VE)$ vs Dijkstra's $O((V+E) \log V)$
    """
    # Get all vertices
    vertices = set(graph.keys())
    for u in graph:
        for v, _ in graph[u]:
            vertices.add(v)

    # Initialize distances
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0

    # Get all edges as list of (u, v, weight)
    edges = []
    for u in graph:
        for v, weight in graph[u]:
            edges.append((u, v, weight))

    # Relax all edges V-1 times
    for _ in range(len(vertices) - 1):
        updated = False
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True

        # Early termination if no updates
        if not updated:
            break

    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            raise ValueError("Graph contains negative cycle")

    return distances


# Example with negative weights
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('D', 2)],
    'C': [('B', -3), ('D', 5)],
    'D': []
}

print(bellman_ford(graph, 'A'))
# {'A': 0, 'B': -1, 'C': 2, 'D': 1}
# Path A→C→B has weight 2+(-3)=-1, shorter than A→B=4!

# Example with negative cycle
graph_with_cycle = {
    'A': [('B', 1)],
    'B': [('C', -3)],
    'C': [('A', 1)]
}
# Cycle A→B→C→A has total weight 1+(-3)+1=-1
# Can loop infinitely to decrease distance!

try:
    bellman_ford(graph_with_cycle, 'A')
except ValueError as e:
    print(e)  # "Graph contains negative cycle"
```

**Algorithm Explanation**:
```
Why V-1 iterations?
- Maximum shortest path has at most V-1 edges (no cycles)
- Each iteration relaxes at least one edge on the shortest path
- After V-1 iterations, all shortest paths are found

Example:
    A --1--> B --1--> C --1--> D

Iteration 1: A=0, B=1, C=∞, D=∞
Iteration 2: A=0, B=1, C=2, D=∞
Iteration 3: A=0, B=1, C=2, D=3
Done! (V-1 = 3 iterations)

Negative Cycle Detection:
- If we can still relax edges after V-1 iterations,
  there must be a negative cycle
- Because legitimate shortest paths can't have more than V-1 edges
```

### 3. Floyd-Warshall Algorithm

Finds shortest paths between **all pairs** of vertices.

```python
def floyd_warshall(graph):
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.

    Time: $O(V^3)$
    Space: $O(V^2)$

    Args:
        graph: Dict where graph[u] = [(v, weight), ...]

    Returns:
        2D list dist where dist[i][j] = shortest distance from vertex i to j
        List of vertices (for indexing)

    Works with:
        - Negative edge weights
        - Detects negative cycles

    Use when:
        - Need shortest paths between all pairs
        - Graph is small/medium (V² space, V³ time)
        - Simpler implementation than running Dijkstra V times
    """
    # Get all vertices
    vertices = sorted(set(graph.keys()))
    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]

    # Distance from vertex to itself is 0
    for i in range(n):
        dist[i][i] = 0

    # Add edges from graph
    for u in graph:
        i = vertex_to_idx[u]
        for v, weight in graph[u]:
            j = vertex_to_idx[v]
            dist[i][j] = weight

    # Floyd-Warshall algorithm
    # Try all vertices as intermediate points
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Is path i→k→j shorter than current i→j?
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("Graph contains negative cycle")

    return dist, vertices


def floyd_warshall_with_path(graph):
    """
    Floyd-Warshall with path reconstruction.

    Returns: (distance matrix, next matrix for path reconstruction, vertices)
    """
    vertices = sorted(set(graph.keys()))
    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    dist = [[float('inf')] * n for _ in range(n)]
    next_vertex = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u in graph:
        i = vertex_to_idx[u]
        for v, weight in graph[u]:
            j = vertex_to_idx[v]
            dist[i][j] = weight
            next_vertex[i][j] = j

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]

    return dist, next_vertex, vertices


def reconstruct_path(i, j, next_vertex):
    """Reconstruct path from i to j using next matrix."""
    if next_vertex[i][j] is None:
        return None

    path = [i]
    while i != j:
        i = next_vertex[i][j]
        path.append(i)

    return path


# Example
graph = {
    'A': [('B', 3), ('C', 8)],
    'B': [('C', 1), ('D', 2)],
    'C': [('D', 4)],
    'D': []
}

dist, vertices = floyd_warshall(graph)

print("Shortest distances between all pairs:")
for i, u in enumerate(vertices):
    for j, v in enumerate(vertices):
        if dist[i][j] == float('inf'):
            print(f"{u}→{v}: ∞", end="  ")
        else:
            print(f"{u}→{v}: {dist[i][j]}", end="  ")
    print()

# Output:
# A→A: 0  A→B: 3  A→C: 4  A→D: 5
# B→B: 0  B→C: 1  B→D: 2
# C→C: 0  C→D: 4
# D→D: 0
```

**Algorithm Visualization**:
```
Idea: Try all possible intermediate vertices

Path from i to j can either:
1. Go directly: i → j
2. Go through k: i → k → j

For each pair (i,j), try all possible intermediate k:
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

Example:
    A --3--> B --1--> C
    |                 ^
    8                 |
    └-----------------┘

Initial:
    A  B  C
A   0  3  8
B   ∞  0  1
C   ∞  ∞  0

Try k=B:
A→C through B: dist[A][B] + dist[B][C] = 3+1=4 < 8
Update A→C to 4!

Final:
    A  B  C
A   0  3  4  (A→C improved via B)
B   ∞  0  1
C   ∞  ∞  0
```

### 4. A* Search Algorithm

A* is an informed search algorithm that uses heuristics to find the shortest path more efficiently.

```python
import heapq

def a_star(graph, start, goal, heuristic):
    """
    A* pathfinding algorithm.

    Time: $O(E \log V)$ typically, depends on heuristic quality
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [(v, weight), ...]
        start: Starting vertex
        goal: Goal vertex
        heuristic: Function h(node, goal) estimating cost to goal

    Returns:
        Shortest path from start to goal, or None if no path exists

    Key concepts:
        - g(n): Actual cost from start to n
        - h(n): Heuristic estimated cost from n to goal
        - f(n) = g(n) + h(n): Total estimated cost

    Heuristic requirements:
        - Admissible: Never overestimates actual cost (h(n) ≤ true cost)
        - Consistent: h(n) ≤ cost(n,n') + h(n') for all neighbors n'

    If heuristic is admissible, A* finds optimal path!
    """
    # Priority queue: (f_score, vertex)
    open_set = [(0, start)]
    came_from = {}

    # g_score: cost from start to vertex
    g_score = {vertex: float('inf') for vertex in graph}
    g_score[start] = 0

    # f_score: g_score + heuristic
    f_score = {vertex: float('inf') for vertex in graph}
    f_score[start] = heuristic(start, goal)

    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        # Reached goal!
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in closed_set:
            continue

        closed_set.add(current)

        # Check all neighbors
        for neighbor, weight in graph[current]:
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score
            tentative_g = g_score[current] + weight

            # Found better path to neighbor
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


# Example: Grid pathfinding with Manhattan distance heuristic
def manhattan_distance(pos1, pos2):
    """
    Manhattan distance heuristic for grid graphs.
    Admissible for grids with 4-directional movement.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """
    Euclidean distance heuristic.
    Admissible for grids with 8-directional movement.
    """
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


# Grid graph example
grid_graph = {
    (0,0): [((0,1), 1), ((1,0), 1)],
    (0,1): [((0,0), 1), ((0,2), 1), ((1,1), 1)],
    (0,2): [((0,1), 1), ((1,2), 1)],
    (1,0): [((0,0), 1), ((1,1), 1), ((2,0), 1)],
    (1,1): [((0,1), 1), ((1,0), 1), ((1,2), 1), ((2,1), 1)],
    (1,2): [((0,2), 1), ((1,1), 1), ((2,2), 1)],
    (2,0): [((1,0), 1), ((2,1), 1)],
    (2,1): [((2,0), 1), ((1,1), 1), ((2,2), 1)],
    (2,2): [((1,2), 1), ((2,1), 1)]
}

path = a_star(grid_graph, (0,0), (2,2), manhattan_distance)
print(path)  # [(0, 0), (1, 1), (2, 2)]
```

**A* vs Dijkstra**:
```
Dijkstra: Explores uniformly in all directions
A*: Explores preferentially toward goal using heuristic

Grid Example (start=S, goal=G):

Dijkstra exploration:        A* exploration (good heuristic):
    3 2 3 4 5                    . . 4 5 6
    2 1 2 3 4                    . 3 3 4 5
    1 S 1 2 3                    2 S 2 3 4
    2 1 2 3 4                    . . 2 3 G
    3 2 3 4 G                    . . . . .

Dijkstra explores ~25 nodes     A* explores ~10 nodes

If heuristic = 0:
    A* becomes Dijkstra!
If heuristic = true cost:
    A* goes directly to goal!
```

### Shortest Path Algorithm Comparison

| Algorithm | Negative Weights | All-Pairs | Time Complexity | Space | Best Use Case |
|-----------|-----------------|-----------|-----------------|-------|---------------|
| **BFS** | No (unweighted) | No | $O(V + E)$ | $O(V)$ | Unweighted graphs |
| **Dijkstra** | No | No | $O((V+E) \log V)$ | $O(V)$ | Single-source, non-negative weights |
| **Bellman-Ford** | Yes | No | $O(VE)$ | $O(V)$ | Negative weights, cycle detection |
| **Floyd-Warshall** | Yes | Yes | $O(V^3)$ | $O(V^2)$ | All-pairs, small graphs |
| **A\*** | No | No | $O(E \log V)$ (typical) | $O(V)$ | Pathfinding with good heuristic |

**Decision Tree**:
```
Need shortest path?
├─ Unweighted graph? → Use BFS
├─ Single source?
│  ├─ Non-negative weights? → Use Dijkstra
│  ├─ Negative weights? → Use Bellman-Ford
│  └─ Have good heuristic? → Use A*
└─ All pairs?
   ├─ Small graph? → Use Floyd-Warshall
   └─ Large graph? → Run Dijkstra V times
```

## Minimum Spanning Tree (MST)

A **spanning tree** of a graph is a subgraph that includes all vertices and is a tree (connected, acyclic). A **minimum spanning tree** is a spanning tree with minimum total edge weight.

**Properties**:
- Has exactly V-1 edges (V = number of vertices)
- Connects all vertices
- No cycles
- Minimum total weight among all spanning trees
- Not unique if multiple edges have same weight

**Applications**:
- Network design (minimum cable/pipe length)
- Approximation algorithms for TSP
- Cluster analysis
- Image segmentation

### Union-Find (Disjoint Set Union)

Union-Find is a data structure used in Kruskal's algorithm to efficiently detect cycles.

```python
class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.

    Supports two operations:
    - find(x): Find the representative (root) of x's set
    - union(x, y): Merge the sets containing x and y

    Optimizations:
    - Path compression: Make tree flatter during find()
    - Union by rank: Attach smaller tree under larger tree

    Time Complexity (amortized):
    - find(): $O(\alpha(n))$ ≈ $O(1)$ where α is inverse Ackermann
    - union(): $O(\alpha(n))$ ≈ $O(1)$

    Space: $O(n)$
    """

    def __init__(self, n):
        """
        Initialize with n elements (0 to n-1).
        Each element starts in its own set.
        """
        self.parent = list(range(n))
        self.rank = [0] * n  # Rank is approximate tree height

    def find(self, x):
        """
        Find the root of x's set with path compression.

        Path compression: Make all nodes on path point directly to root.
        This flattens the tree structure for faster future finds.
        """
        if self.parent[x] != x:
            # Recursively find root and compress path
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """
        Merge the sets containing x and y.

        Union by rank: Attach shorter tree under taller tree.
        This keeps the tree balanced.

        Returns:
            True if union was performed (x and y in different sets)
            False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        # Already in same set
        if root_x == root_y:
            return False

        # Union by rank: attach smaller tree under larger
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Same rank: choose one as root, increase its rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)


# Example usage
uf = UnionFind(5)  # Elements 0,1,2,3,4

uf.union(0, 1)  # {0,1} {2} {3} {4}
uf.union(2, 3)  # {0,1} {2,3} {4}
uf.union(0, 4)  # {0,1,4} {2,3}

print(uf.connected(1, 4))  # True (same set)
print(uf.connected(1, 3))  # False (different sets)
```

### 1. Kruskal's Algorithm

Builds MST by adding edges in order of increasing weight, skipping edges that create cycles.

```python
def kruskal(edges, num_vertices):
    """
    Kruskal's algorithm for finding Minimum Spanning Tree.

    Time: $O(E \log E)$ dominated by edge sorting
    Space: $O(V)$ for Union-Find

    Args:
        edges: List of (weight, u, v) tuples
        num_vertices: Number of vertices (0 to num_vertices-1)

    Returns:
        List of edges in MST, total weight

    Algorithm:
    1. Sort all edges by weight
    2. For each edge (u,v):
       - If u and v in different components: add edge, union components
       - Else: skip edge (would create cycle)
    3. Stop when MST has V-1 edges
    """
    # Sort edges by weight
    edges = sorted(edges, key=lambda x: x[0])

    uf = UnionFind(num_vertices)
    mst = []
    total_weight = 0

    for weight, u, v in edges:
        # If u and v are not connected, add this edge
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

            # MST complete when we have V-1 edges
            if len(mst) == num_vertices - 1:
                break

    return mst, total_weight


# Example
edges = [
    (1, 0, 1),  # (weight, u, v)
    (2, 0, 2),
    (3, 1, 2),
    (4, 1, 3),
    (5, 2, 3),
    (6, 2, 4),
    (7, 3, 4)
]

mst, weight = kruskal(edges, 5)
print("MST edges:", mst)
print("Total weight:", weight)
# MST edges: [(0, 1, 1), (0, 2, 2), (1, 3, 4), (2, 4, 6)]
# Total weight: 13
```

**Visual Example**:
```
Original Graph:
        1
    0 ----- 1
    |     / | \
   2|   3/  |4 \7
    |   /   |   \
    2 ----- 3 --- 4
       5  /   \
         6     7

Edges sorted: [(1,0,1), (2,0,2), (3,1,2), (4,1,3), (5,2,3), (6,2,4), (7,3,4)]

Step-by-step:
1. Add (0,1,1): {0,1} {2} {3} {4}          Weight: 1
2. Add (0,2,2): {0,1,2} {3} {4}            Weight: 3
3. Skip (1,2,3): 1 and 2 already connected (would form cycle)
4. Add (1,3,4): {0,1,2,3} {4}              Weight: 7
5. Skip (2,3,5): 2 and 3 already connected
6. Add (2,4,6): {0,1,2,3,4}                Weight: 13
Done! Have 4 edges (V-1 = 5-1)

Final MST:
    0 ----- 1
    |       |
   2|       |4
    |       |
    2 ----- 3
       6
         \
          4

Total weight: 1+2+4+6 = 13
```

### 2. Prim's Algorithm

Builds MST by growing a tree from a starting vertex, always adding the minimum weight edge that connects a new vertex.

```python
import heapq

def prim(graph, start):
    """
    Prim's algorithm for finding Minimum Spanning Tree.

    Time: $O((V + E) \log V)$ with binary heap
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [(v, weight), ...]
        start: Starting vertex (can be any vertex)

    Returns:
        List of edges in MST, total weight

    Algorithm:
    1. Start with single vertex
    2. Repeat:
       - Find minimum weight edge connecting tree to non-tree vertex
       - Add that edge and vertex to tree
    3. Stop when all vertices in tree
    """
    mst = []
    visited = {start}

    # Priority queue: (weight, from_vertex, to_vertex)
    edges = [(weight, start, neighbor) for neighbor, weight in graph[start]]
    heapq.heapify(edges)

    total_weight = 0

    while edges and len(visited) < len(graph):
        weight, u, v = heapq.heappop(edges)

        # Skip if v already in MST
        if v in visited:
            continue

        # Add edge to MST
        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight

        # Add edges from newly added vertex
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush(edges, (w, v, neighbor))

    return mst, total_weight


# Example
graph = {
    0: [(1, 1), (2, 2)],
    1: [(0, 1), (2, 3), (3, 4)],
    2: [(0, 2), (1, 3), (3, 5), (4, 6)],
    3: [(1, 4), (2, 5), (4, 7)],
    4: [(2, 6), (3, 7)]
}

mst, weight = prim(graph, 0)
print("MST edges:", mst)
print("Total weight:", weight)
# MST edges: [(0, 1, 1), (0, 2, 2), (1, 3, 4), (2, 4, 6)]
# Total weight: 13
```

**Visual Example**:
```
Same graph as Kruskal example:
        1
    0 ----- 1
    |     / | \
   2|   3/  |4 \7
    |   /   |   \
    2 ----- 3 --- 4
       5  /   \
         6     7

Starting from vertex 0:

Step 1: MST={0}, Edges from 0: [(1,0,1), (2,0,2)]
        Choose (0,1,1) - minimum
        MST={0,1}, Added: 0-1 (weight 1)

Step 2: Edges: [(2,0,2), (3,1,2), (4,1,3)]
        Choose (0,2,2) - minimum
        MST={0,1,2}, Added: 0-2 (weight 2)

Step 3: Edges: [(3,1,2), (4,1,3), (5,2,3), (6,2,4)]
        Choose (1,2,3) - skip, 2 already in MST
        Choose (1,3,4) - minimum unvisited
        MST={0,1,2,3}, Added: 1-3 (weight 4)

Step 4: Edges: [(5,2,3), (6,2,4), (7,3,4)]
        Choose (2,3,5) - skip, 3 already in MST
        Choose (2,4,6) - minimum unvisited
        MST={0,1,2,3,4}, Added: 2-4 (weight 6)

Done! All vertices in MST.
Total weight: 1+2+4+6 = 13
```

### Kruskal vs Prim

| Aspect | Kruskal | Prim |
|--------|---------|------|
| **Approach** | Edge-centric (greedy on edges) | Vertex-centric (grow tree) |
| **Data Structure** | Union-Find + sorted edges | Priority queue (heap) |
| **Time Complexity** | $O(E \log E)$ | $O((V+E) \log V)$ |
| **Best For** | Sparse graphs | Dense graphs |
| **Implementation** | Simpler with Union-Find | More complex |
| **Graph Type** | Works on disconnected | Needs connected component |
| **Edge Selection** | Global minimum | Local minimum from tree |

**When to use which**:
- **Kruskal**: Sparse graphs, already have sorted edges, simpler to implement
- **Prim**: Dense graphs, need to find MST for specific component

Both produce same total weight (MST is usually not unique if multiple edges have same weight, but total weight is always the same).

## Topological Sort

**Topological Sort** of a Directed Acyclic Graph (DAG) is a linear ordering of vertices such that for every directed edge (u,v), vertex u comes before v in the ordering.

**Applications**:
- Task scheduling with dependencies
- Build systems (compile order)
- Course prerequisites
- Dependency resolution (package managers)

**Note**: Only possible for DAGs (no cycles). If graph has cycles, topological sort doesn't exist.

### 1. DFS-based Topological Sort

```python
def topological_sort_dfs(graph):
    """
    Topological sort using DFS.

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...] (directed graph)

    Returns:
        List of vertices in topological order

    Algorithm:
    1. Perform DFS from each unvisited vertex
    2. Add vertex to stack after visiting all descendants
    3. Stack gives reverse topological order

    Intuition: Vertices with no dependencies end up at bottom of stack,
               so when reversed, they come last.
    """
    visited = set()
    stack = []

    def dfs(vertex):
        visited.add(vertex)

        # Visit all neighbors first
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)

        # Add to stack after visiting all descendants
        stack.append(vertex)

    # Visit all vertices
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)

    # Reverse stack to get topological order
    return stack[::-1]


# Example: Course prerequisites
graph = {
    'Math': ['Physics', 'CS'],
    'Physics': ['Engineering'],
    'CS': ['AI', 'Engineering'],
    'AI': [],
    'Engineering': []
}

print(topological_sort_dfs(graph))
# Possible output: ['Math', 'CS', 'Physics', 'AI', 'Engineering']
# or: ['Math', 'Physics', 'CS', 'Engineering', 'AI']
# Multiple valid orderings!
```

**Visual Example**:
```
Graph (course prerequisites):
    Math
    / \
  CS   Physics
  / \    |
AI  Engineering
     (CS → Engineering)

DFS traversal:
1. Visit Math
   2. Visit CS
      3. Visit AI → add AI to stack: [AI]
      4. Visit Engineering → add Engineering: [AI, Engineering]
   5. Add CS: [AI, Engineering, CS]
   6. Visit Physics
      7. Engineering already visited
   8. Add Physics: [AI, Engineering, CS, Physics]
9. Add Math: [AI, Engineering, CS, Physics, Math]

Reversed: [Math, Physics, CS, Engineering, AI]

This ensures:
- Math before Physics and CS
- CS before AI and Engineering
- Physics before Engineering
```

### 2. Kahn's Algorithm (BFS-based)

```python
from collections import deque

def topological_sort_kahn(graph):
    """
    Topological sort using Kahn's algorithm (BFS-based).

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...]

    Returns:
        List of vertices in topological order

    Raises:
        ValueError: If graph contains a cycle

    Algorithm:
    1. Calculate in-degree for all vertices
    2. Add vertices with in-degree 0 to queue
    3. Process queue:
       - Remove vertex, add to result
       - Decrease in-degree of neighbors
       - If neighbor's in-degree becomes 0, add to queue
    4. If processed all vertices: success
       Else: graph has cycle

    Advantage over DFS: Detects cycles naturally
    """
    # Calculate in-degree for all vertices
    in_degree = {v: 0 for v in graph}

    # Build in-degree map
    for u in graph:
        for v in graph[u]:
            if v not in in_degree:
                in_degree[v] = 0
            in_degree[v] += 1

    # Queue of vertices with no incoming edges
    queue = deque([v for v in in_degree if in_degree[v] == 0])
    result = []

    while queue:
        # Remove vertex with no incoming edges
        vertex = queue.popleft()
        result.append(vertex)

        # Reduce in-degree of neighbors
        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1

            # If in-degree becomes 0, add to queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all vertices were processed
    if len(result) != len(in_degree):
        raise ValueError("Graph contains a cycle - topological sort not possible")

    return result


# Example
graph = {
    'A': ['C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

print(topological_sort_kahn(graph))
# Possible output: ['A', 'B', 'C', 'D', 'E', 'F']
# or: ['B', 'A', 'C', 'D', 'E', 'F']

# Example with cycle
graph_with_cycle = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']  # Cycle: A→B→C→A
}

try:
    topological_sort_kahn(graph_with_cycle)
except ValueError as e:
    print(e)  # "Graph contains a cycle"
```

**Visual Example**:
```
Graph:
    A → C → E → F
    B → D ↗     ↑
                |

In-degrees:
A: 0, B: 0, C: 2, D: 1, E: 1, F: 2

Step 1: Queue = [A, B] (in-degree 0)
        Process A: result = [A]
        Decrease in-degree: C (2→1)

Step 2: Queue = [B]
        Process B: result = [A, B]
        Decrease in-degree: C (1→0), D (1→0)

Step 3: Queue = [C, D]
        Process C: result = [A, B, C]
        Decrease in-degree: E (1→0)

Step 4: Queue = [D, E]
        Process D: result = [A, B, C, D]
        Decrease in-degree: F (2→1)

Step 5: Queue = [E]
        Process E: result = [A, B, C, D, E]
        Decrease in-degree: F (1→0)

Step 6: Queue = [F]
        Process F: result = [A, B, C, D, E, F]

All vertices processed! Valid topological order.
```

### DFS vs Kahn's Algorithm

| Aspect | DFS-based | Kahn's (BFS-based) |
|--------|-----------|-------------------|
| **Approach** | Post-order DFS | Remove vertices with in-degree 0 |
| **Cycle Detection** | Requires additional code | Built-in (check if all processed) |
| **Implementation** | Simpler with recursion | Requires in-degree calculation |
| **Intuition** | Dependencies finish first | Process nodes with no dependencies |
| **Space** | $O(h)$ recursion + $O(V)$ | $O(V)$ for queue and in-degree |

Both are $O(V + E)$ time. Choose based on preference or if cycle detection is needed (Kahn's is cleaner for this).

## Cycle Detection

Detecting cycles is crucial for many graph algorithms and validations.

### 1. Cycle Detection in Undirected Graphs

```python
def has_cycle_undirected(graph):
    """
    Detect cycle in undirected graph using DFS.

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...] (undirected)

    Returns:
        True if cycle exists, False otherwise

    Key idea:
    - In DFS, if we reach a visited vertex that's not the parent,
      we found a back edge → cycle exists
    - Parent exception: In undirected graph, edge u-v appears as
      both u→v and v→u, so we must ignore the edge we came from
    """
    visited = set()

    def dfs(vertex, parent):
        visited.add(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                if dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                # Found back edge to visited vertex (not parent) = cycle!
                return True

        return False

    # Check all components
    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True

    return False


# Example without cycle
graph1 = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B']
}
print(has_cycle_undirected(graph1))  # False

# Example with cycle
graph2 = {
    'A': ['B', 'C'],
    'B': ['A', 'C'],
    'C': ['A', 'B']  # A-B-C-A forms cycle
}
print(has_cycle_undirected(graph2))  # True
```

**Visual Example**:
```
No cycle:        Has cycle:
    A                A
   / \              / \
  B   C            B---C
  |
  D

DFS from A in cyclic graph:
1. Visit A
2. Visit B (from A)
3. See C (from B)
   - C is visited
   - C is not parent (parent is A)
   - CYCLE FOUND!
```

### 2. Cycle Detection in Directed Graphs

```python
def has_cycle_directed(graph):
    """
    Detect cycle in directed graph using DFS with colors.

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...] (directed)

    Returns:
        True if cycle exists, False otherwise

    Three-color approach:
    - WHITE (0): Not visited yet
    - GRAY (1): Currently being explored (in DFS stack)
    - BLACK (2): Completely explored

    Key idea:
    - If we encounter a GRAY vertex during DFS, we found a back edge
      (edge to an ancestor in DFS tree) → cycle exists
    - GRAY vertices form the current DFS path
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}

    def dfs(vertex):
        color[vertex] = GRAY

        for neighbor in graph.get(vertex, []):
            if color.get(neighbor, WHITE) == GRAY:
                # Back edge to vertex currently being explored = cycle!
                return True

            if color.get(neighbor, WHITE) == WHITE:
                if dfs(neighbor):
                    return True

        color[vertex] = BLACK
        return False

    # Check all components
    for vertex in graph:
        if color[vertex] == WHITE:
            if dfs(vertex):
                return True

    return False


# Example without cycle (DAG)
dag = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}
print(has_cycle_directed(dag))  # False

# Example with cycle
cyclic = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']  # A→B→C→A
}
print(has_cycle_directed(cyclic))  # True
```

**Visual Example**:
```
Directed cycle:
    A → B → C
    ↑       |
    └-------┘

DFS traversal:
1. Visit A (color: GRAY)
2. Visit B from A (color: GRAY)
3. Visit C from B (color: GRAY)
4. See A from C
   - A is GRAY (currently being explored)
   - Back edge found: C→A
   - CYCLE DETECTED!

Current DFS path: [A, B, C]
A is ancestor of C in this path!
```

### 3. Finding the Cycle

```python
def find_cycle_directed(graph):
    """
    Find and return the actual cycle in a directed graph.

    Returns:
        List of vertices forming the cycle, or None if no cycle
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}
    parent = {}

    def dfs(vertex, path):
        color[vertex] = GRAY
        path.append(vertex)

        for neighbor in graph.get(vertex, []):
            if color.get(neighbor, WHITE) == GRAY:
                # Found cycle! Extract it from path
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]

            if color.get(neighbor, WHITE) == WHITE:
                result = dfs(neighbor, path)
                if result:
                    return result

        color[vertex] = BLACK
        path.pop()
        return None

    for vertex in graph:
        if color[vertex] == WHITE:
            cycle = dfs(vertex, [])
            if cycle:
                return cycle

    return None


# Example
graph = {
    'A': ['B'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['E'],
    'E': ['B']  # Cycle: B→D→E→B
}

cycle = find_cycle_directed(graph)
print(cycle)  # ['B', 'D', 'E', 'B']
```

## Connected Components

### 1. Connected Components (Undirected Graphs)

```python
def find_connected_components(graph):
    """
    Find all connected components in an undirected graph.

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...] (undirected)

    Returns:
        List of components, where each component is a list of vertices

    A connected component is a maximal set of vertices such that
    there's a path between any two vertices in the set.
    """
    visited = set()
    components = []

    def dfs(vertex, component):
        visited.add(vertex)
        component.append(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor, component)

    # Find each connected component
    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs(vertex, component)
            components.append(component)

    return components


# Example
graph = {
    'A': ['B', 'C'],
    'B': ['A'],
    'C': ['A'],
    'D': ['E'],
    'E': ['D'],
    'F': []
}

components = find_connected_components(graph)
print(components)
# [['A', 'B', 'C'], ['D', 'E'], ['F']]
# Three separate components!
```

### 2. Strongly Connected Components (Directed Graphs)

A **strongly connected component** (SCC) is a maximal set of vertices where there's a directed path between every pair of vertices.

#### Kosaraju's Algorithm

```python
def kosaraju_scc(graph):
    """
    Find strongly connected components using Kosaraju's algorithm.

    Time: $O(V + E)$
    Space: $O(V + E)$

    Args:
        graph: Dict where graph[u] = [v, ...] (directed)

    Returns:
        List of SCCs, each SCC is a list of vertices

    Algorithm:
    1. Perform DFS on original graph, record finish times
    2. Create transpose graph (reverse all edges)
    3. Perform DFS on transpose in decreasing finish time order
    4. Each DFS tree in step 3 is an SCC

    Why it works:
    - DFS from highest finish time explores one SCC
    - In transpose, can't escape SCC (reversed edges)
    - Process SCCs in reverse topological order
    """
    # Step 1: DFS on original graph to get finish order
    visited = set()
    finish_stack = []

    def dfs1(vertex):
        visited.add(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_stack.append(vertex)

    for vertex in graph:
        if vertex not in visited:
            dfs1(vertex)

    # Step 2: Create transpose graph
    transpose = {v: [] for v in graph}
    for u in graph:
        for v in graph[u]:
            transpose[v].append(u)

    # Step 3: DFS on transpose in reverse finish order
    visited = set()
    sccs = []

    def dfs2(vertex, scc):
        visited.add(vertex)
        scc.append(vertex)
        for neighbor in transpose.get(vertex, []):
            if neighbor not in visited:
                dfs2(neighbor, scc)

    while finish_stack:
        vertex = finish_stack.pop()
        if vertex not in visited:
            scc = []
            dfs2(vertex, scc)
            sccs.append(scc)

    return sccs


# Example
graph = {
    'A': ['B'],
    'B': ['C', 'E'],
    'C': ['A', 'D'],
    'D': [],
    'E': ['F'],
    'F': ['E']
}

sccs = kosaraju_scc(graph)
print(sccs)
# [['D'], ['E', 'F'], ['A', 'B', 'C']]
# Three SCCs: {D}, {E,F}, {A,B,C}
```

**Visual Example**:
```
Graph:
    A → B → E ⇄ F
    ↑   ↓
    C ← ┘
    ↓
    D

SCCs:
1. {A, B, C} - cycle A→B→C→A
2. {E, F} - cycle E→F→E
3. {D} - single node, no cycle

Why {A,B,C} is SCC:
- Can reach B from A: A→B
- Can reach C from A: A→B→C
- Can reach A from B: B→C→A
- Can reach A from C: C→A
- Can reach C from B: B→C
- Can reach B from C: C→A→B
All pairs reachable!

Why D is separate:
- Can reach D from others, but can't reach others from D
```

#### Tarjan's Algorithm

```python
def tarjan_scc(graph):
    """
    Find strongly connected components using Tarjan's algorithm.

    Time: $O(V + E)$ - single DFS!
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...]

    Returns:
        List of SCCs

    Advantage over Kosaraju: Only one DFS pass (no transpose needed)

    Key concepts:
    - index: Order in which vertices are visited
    - lowlink: Smallest index reachable from vertex via DFS
    - Stack: Current path in DFS
    - SCC found when vertex.lowlink == vertex.index
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = set()
    sccs = []

    def strongconnect(vertex):
        # Set depth index for vertex
        index[vertex] = index_counter[0]
        lowlinks[vertex] = index_counter[0]
        index_counter[0] += 1
        stack.append(vertex)
        on_stack.add(vertex)

        # Consider successors
        for neighbor in graph.get(vertex, []):
            if neighbor not in index:
                # Neighbor not yet visited, recurse
                strongconnect(neighbor)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbor])
            elif neighbor in on_stack:
                # Neighbor in current SCC
                lowlinks[vertex] = min(lowlinks[vertex], index[neighbor])

        # If vertex is root of SCC
        if lowlinks[vertex] == index[vertex]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == vertex:
                    break
            sccs.append(scc)

    for vertex in graph:
        if vertex not in index:
            strongconnect(vertex)

    return sccs


# Example (same as Kosaraju)
graph = {
    'A': ['B'],
    'B': ['C', 'E'],
    'C': ['A', 'D'],
    'D': [],
    'E': ['F'],
    'F': ['E']
}

sccs = tarjan_scc(graph)
print(sccs)
# [['F', 'E'], ['D'], ['C', 'B', 'A']]
# Same SCCs, possibly different order
```

### Connected Components Summary

| Algorithm | Graph Type | Time | Space | Passes |
|-----------|-----------|------|-------|--------|
| **DFS/BFS** | Undirected | $O(V+E)$ | $O(V)$ | 1 |
| **Kosaraju** | Directed (SCC) | $O(V+E)$ | $O(V+E)$ | 2 (+ transpose) |
| **Tarjan** | Directed (SCC) | $O(V+E)$ | $O(V)$ | 1 |

Tarjan's algorithm is more efficient (single DFS) but Kosaraju's is simpler to understand.

## Bipartite Graphs

A **bipartite graph** is a graph whose vertices can be divided into two disjoint sets such that every edge connects a vertex from one set to the other set (no edges within sets).

**Properties**:
- A graph is bipartite ⟺ it contains no odd-length cycles
- Can be 2-colored (vertices in set A = color 1, set B = color 2)
- Trees are always bipartite

**Applications**:
- Matching problems (jobs-candidates, students-projects)
- Scheduling (tasks-workers)
- Network flow problems

### 1. Check if Bipartite

```python
from collections import deque

def is_bipartite(graph):
    """
    Check if graph is bipartite using BFS coloring.

    Time: $O(V + E)$
    Space: $O(V)$

    Args:
        graph: Dict where graph[u] = [v, ...]

    Returns:
        True if bipartite, False otherwise

    Algorithm:
    - Try to 2-color the graph using BFS
    - Start with any vertex, color it 0
    - Color all neighbors with opposite color (1)
    - If neighbor already has same color → not bipartite
    - If successfully colored all vertices → bipartite
    """
    color = {}

    def bfs(start):
        queue = deque([start])
        color[start] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in graph.get(vertex, []):
                if neighbor not in color:
                    # Color neighbor with opposite color
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    # Neighbor has same color → not bipartite!
                    return False

        return True

    # Check all components (graph might be disconnected)
    for vertex in graph:
        if vertex not in color:
            if not bfs(vertex):
                return False

    return True


# Example: Bipartite graph
bipartite = {
    'A': ['C', 'D'],
    'B': ['C', 'D'],
    'C': ['A', 'B'],
    'D': ['A', 'B']
}
print(is_bipartite(bipartite))  # True
# Sets: {A, B} and {C, D}

# Example: Not bipartite (triangle = odd cycle)
not_bipartite = {
    'A': ['B', 'C'],
    'B': ['A', 'C'],
    'C': ['A', 'B']
}
print(is_bipartite(not_bipartite))  # False
```

**Visual Example**:
```
Bipartite (rectangle):
    A --- C
    |     |
    |     |
    B --- D

Two sets: {A, B} and {C, D}
Coloring: A=0, B=0, C=1, D=1 ✓

Not Bipartite (triangle):
    A --- B
     \   /
      \ /
       C

Trying to color:
A=0, B=1, C=0
But C is neighbor of both A(0) and B(1)!
Need C to be both 1 and 0 → impossible!

Odd cycle: A-B-C-A (length 3)
```

### 2. Find Bipartite Partitions

```python
def find_bipartite_sets(graph):
    """
    Find the two disjoint sets if graph is bipartite.

    Returns:
        (set_0, set_1) if bipartite, None if not bipartite
    """
    color = {}

    def bfs(start):
        queue = deque([start])
        color[start] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in graph.get(vertex, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return False

        return True

    # Check all components
    for vertex in graph:
        if vertex not in color:
            if not bfs(vertex):
                return None  # Not bipartite

    # Separate vertices by color
    set_0 = [v for v, c in color.items() if c == 0]
    set_1 = [v for v, c in color.items() if c == 1]

    return set_0, set_1


# Example
graph = {
    'A': ['C', 'D'],
    'B': ['C', 'D'],
    'C': ['A', 'B', 'E'],
    'D': ['A', 'B'],
    'E': ['C']
}

sets = find_bipartite_sets(graph)
print(sets)
# (['A', 'B', 'E'], ['C', 'D'])
# or (['C', 'D'], ['A', 'B', 'E'])
```

### 3. DFS-based Bipartite Check

```python
def is_bipartite_dfs(graph):
    """
    Check if graph is bipartite using DFS.

    Alternative to BFS approach, same complexity.
    """
    color = {}

    def dfs(vertex, c):
        color[vertex] = c

        for neighbor in graph.get(vertex, []):
            if neighbor not in color:
                # Color neighbor with opposite color
                if not dfs(neighbor, 1 - c):
                    return False
            elif color[neighbor] == color[vertex]:
                # Same color → not bipartite
                return False

        return True

    # Check all components
    for vertex in graph:
        if vertex not in color:
            if not dfs(vertex, 0):
                return False

    return True
```

## Common Graph Problems

### 1. Clone Graph

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    """
    Deep clone a graph (LeetCode 133).

    Time: $O(V + E)$
    Space: $O(V)$

    Returns: Clone of the input node
    """
    if not node:
        return None

    # Map original nodes to clones
    clones = {}

    def dfs(node):
        if node in clones:
            return clones[node]

        # Create clone
        clone = Node(node.val)
        clones[node] = clone

        # Clone all neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)
```

### 2. Number of Islands

```python
def num_islands(grid):
    """
    Count number of islands in 2D grid (LeetCode 200).

    Time: $O(m \times n)$ where m=rows, n=cols
    Space: $O(m \times n)$ worst case (all land)

    An island is surrounded by water and formed by connecting
    adjacent lands horizontally or vertically.
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        # Boundary check and water check
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0'):
            return

        # Mark as visited by changing to water
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left

    # Find all islands
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                islands += 1

    return islands


# Example
grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1']
]
print(num_islands(grid))  # 3 islands
```

### 3. Course Schedule (Cycle Detection)

```python
def can_finish(num_courses, prerequisites):
    """
    Determine if you can finish all courses (LeetCode 207).

    Time: $O(V + E)$
    Space: $O(V + E)$

    prerequisites[i] = [a, b] means must take course b before a.
    Return True if can finish all courses (no cycle in dependency graph).
    """
    # Build graph
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    # Detect cycle using DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_courses

    def has_cycle(course):
        color[course] = GRAY

        for prereq in graph[course]:
            if color[prereq] == GRAY:
                return True  # Back edge = cycle
            if color[prereq] == WHITE:
                if has_cycle(prereq):
                    return True

        color[course] = BLACK
        return False

    # Check all components
    for course in range(num_courses):
        if color[course] == WHITE:
            if has_cycle(course):
                return False  # Cycle found, can't finish

    return True  # No cycle, can finish all


# Example
print(can_finish(2, [[1,0]]))  # True: 0 → 1
print(can_finish(2, [[1,0],[0,1]]))  # False: 0 ⇄ 1 (cycle)
```

### 4. Course Schedule II (Topological Sort)

```python
def find_order(num_courses, prerequisites):
    """
    Return course order to finish all courses (LeetCode 210).

    Returns: List of course order, or [] if impossible
    """
    # Build graph
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq → course

    # Topological sort using DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_courses
    order = []
    has_cycle = [False]

    def dfs(course):
        if has_cycle[0]:
            return

        color[course] = GRAY

        for next_course in graph[course]:
            if color[next_course] == GRAY:
                has_cycle[0] = True
                return
            if color[next_course] == WHITE:
                dfs(next_course)

        color[course] = BLACK
        order.append(course)

    for course in range(num_courses):
        if color[course] == WHITE:
            dfs(course)
            if has_cycle[0]:
                return []

    return order[::-1]  # Reverse for correct order


# Example
print(find_order(4, [[1,0],[2,0],[3,1],[3,2]]))
# [0, 1, 2, 3] or [0, 2, 1, 3]
```

### 5. Word Ladder

```python
def ladder_length(begin_word, end_word, word_list):
    """
    Find shortest transformation sequence length (LeetCode 127).

    Time: $O(M^2 \times N)$ where M=word length, N=word list size
    Space: $O(M \times N)$

    Each transformation changes exactly one letter.
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, length = queue.popleft()

        if word == end_word:
            return length

        # Try all possible one-letter changes
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]

                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, length + 1))

    return 0  # No transformation sequence found


# Example
begin = "hit"
end = "cog"
word_list = ["hot","dot","dog","lot","log","cog"]
print(ladder_length(begin, end, word_list))
# 5: "hit" → "hot" → "dot" → "dog" → "cog"
```

### 6. Network Delay Time

```python
def network_delay_time(times, n, k):
    """
    Find time for signal to reach all nodes (LeetCode 743).

    Time: $O((V + E) \log V)$ using Dijkstra
    Space: $O(V + E)$

    times[i] = [u, v, w] means signal from u to v takes w time.
    Return minimum time for all nodes to receive signal from k.
    """
    # Build graph
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra's algorithm
    import heapq
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0
    pq = [(0, k)]
    visited = set()

    while pq:
        curr_dist, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in graph[node]:
            dist = curr_dist + weight
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(pq, (dist, neighbor))

    # Max distance to any node
    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1


# Example
times = [[2,1,1],[2,3,1],[3,4,1]]
n = 4
k = 2
print(network_delay_time(times, n, k))
# 2: node 2 → node 3 → node 4 (takes 2 time)
```

### 7. Alien Dictionary

```python
def alien_order(words):
    """
    Derive order of letters in alien language (LeetCode 269).

    Time: $O(C)$ where C = total characters in all words
    Space: $O(1)$ - at most 26 letters

    Given sorted dictionary in alien language, return order of letters.
    """
    # Build graph
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for word in words for c in word}

    # Compare adjacent words to find character order
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))

        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j]
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
        else:
            # word1 is prefix of word2, or same → check lengths
            if len(word1) > len(word2):
                return ""  # Invalid: ["abc", "ab"]

    # Topological sort (Kahn's algorithm)
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    order = []

    while queue:
        c = queue.popleft()
        order.append(c)

        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all characters processed (no cycle)
    if len(order) != len(in_degree):
        return ""  # Cycle exists, invalid

    return "".join(order)


# Example
words = ["wrt", "wrf", "er", "ett", "rftt"]
print(alien_order(words))
# "wertf" - one possible order
# w before e (wrt vs er)
# t before f (wrt vs wrf)
# r before t (er vs ett)
# e before r (wrf vs er)
```

## Graph Patterns and Problem-Solving Strategies

### Common Graph Patterns

1. **Matrix as Graph**
   - Treat 2D grid cells as vertices
   - Adjacent cells (up/down/left/right) are neighbors
   - Problems: Number of Islands, Word Search, Maze solving

2. **Implicit Graph**
   - Graph not explicitly given, build from problem
   - Problems: Word Ladder (words as nodes), Sliding Puzzle

3. **State Space Graph**
   - Each state is a vertex
   - Transitions between states are edges
   - Problems: BFS for minimum steps, Game solvers

4. **Tree as Graph**
   - Trees are special graphs (connected, acyclic)
   - Can use graph algorithms on trees
   - But tree algorithms often simpler

5. **Union-Find Pattern**
   - Dynamic connectivity queries
   - Problems: Number of Connected Components, Redundant Connection

6. **Two-Pointer/Multi-Source**
   - Start BFS from multiple sources simultaneously
   - Problems: Rotting Oranges, Walls and Gates

7. **Backtracking on Graphs**
   - DFS with path tracking
   - Problems: All Paths, Hamiltonian Path

### Pattern Recognition Guide

| Problem Type | Algorithm | Key Indicator |
|--------------|-----------|---------------|
| Shortest path (unweighted) | BFS | "Minimum steps", "shortest path" |
| Shortest path (weighted) | Dijkstra/Bellman-Ford | Edge weights given |
| Pathfinding with heuristic | A* | Have distance estimate to goal |
| Connectivity | DFS/BFS/Union-Find | "Connected", "reachable" |
| Cycle detection | DFS (colors) | "Circular dependency", "deadlock" |
| Ordering with dependencies | Topological Sort | "Prerequisites", "order" |
| Minimum cost tree | MST (Kruskal/Prim) | "Connect all", "minimum cost" |
| All-pairs distances | Floyd-Warshall | Need distances between all pairs |
| 2-coloring | Bipartite check | "Divide into two groups" |
| Components | DFS/BFS | "Number of groups", "clusters" |

## Complexity Summary

### Graph Representation Complexities

| Operation | Adj Matrix | Adj List | Edge List |
|-----------|-----------|----------|-----------|
| Space | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| Add vertex | $O(V^2)$ | $O(1)$ | $O(1)$ |
| Add edge | $O(1)$ | $O(1)$ | $O(1)$ |
| Remove vertex | $O(V^2)$ | $O(E)$ | $O(E)$ |
| Remove edge | $O(1)$ | $O(V)$ | $O(E)$ |
| Query edge | $O(1)$ | $O(degree)$ | $O(E)$ |
| Iterate neighbors | $O(V)$ | $O(degree)$ | $O(E)$ |

### Algorithm Complexities

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| **DFS** | $O(V + E)$ | $O(V)$ | Traverse, cycle detection, topological sort |
| **BFS** | $O(V + E)$ | $O(V)$ | Shortest path (unweighted), level-order |
| **Dijkstra** | $O((V+E) \log V)$ | $O(V)$ | Shortest path (non-negative weights) |
| **Bellman-Ford** | $O(VE)$ | $O(V)$ | Shortest path (negative weights) |
| **Floyd-Warshall** | $O(V^3)$ | $O(V^2)$ | All-pairs shortest paths |
| **A\*** | $O(E \log V)$ | $O(V)$ | Pathfinding with heuristic |
| **Kruskal's MST** | $O(E \log E)$ | $O(V)$ | Minimum spanning tree |
| **Prim's MST** | $O((V+E) \log V)$ | $O(V)$ | Minimum spanning tree |
| **Topological Sort** | $O(V + E)$ | $O(V)$ | Ordering DAG |
| **Kosaraju's SCC** | $O(V + E)$ | $O(V)$ | Strongly connected components |
| **Tarjan's SCC** | $O(V + E)$ | $O(V)$ | Strongly connected components |
| **Union-Find** | $O(\alpha(n))$ | $O(n)$ | Dynamic connectivity |

## Tips and Best Practices

### When to Use Each Algorithm

**For Shortest Paths**:
- Unweighted graph → **BFS**
- Non-negative weights, single source → **Dijkstra**
- Negative weights allowed → **Bellman-Ford**
- All-pairs, small graph → **Floyd-Warshall**
- Have good heuristic → **A***

**For Traversal**:
- Shortest path, level-order → **BFS**
- Explore all possibilities, backtracking → **DFS**
- Need to track depth/distance → **BFS with levels**

**For Connectivity**:
- Find connected components → **DFS/BFS**
- Dynamic connectivity → **Union-Find**
- Strongly connected (directed) → **Kosaraju/Tarjan**

**For Ordering**:
- Dependencies, prerequisites → **Topological Sort**
- Need to detect cycle → **Kahn's algorithm**

### Common Pitfalls

1. **Forgetting to Mark Visited**
   - Always track visited vertices to avoid infinite loops
   - For undirected graphs in DFS, remember parent to avoid false cycles

2. **Directed vs Undirected**
   - Clarify with interviewer
   - Undirected: add edge both ways
   - Directed: cycle detection different

3. **Weighted vs Unweighted**
   - BFS only finds shortest path in unweighted graphs
   - For weighted, need Dijkstra/Bellman-Ford

4. **Disconnected Graphs**
   - Many algorithms need to loop over all vertices
   - DFS/BFS from one vertex might not reach all

5. **Negative Weight Cycles**
   - Dijkstra fails with negative weights
   - Bellman-Ford needed, or Floyd-Warshall

6. **Modifying Graph During Traversal**
   - Be careful when marking cells in grid as visited
   - Consider using separate visited set vs modifying input

### Interview Tips

**Always Ask**:
1. Is the graph directed or undirected?
2. Is it weighted or unweighted?
3. Can there be multiple edges between same vertices?
4. Can there be self-loops?
5. Is the graph connected?
6. What's the size of the graph? (sparse vs dense)
7. Are there negative weights/cycles?

**Problem-Solving Approach**:
1. **Identify the graph**: Explicit or implicit? How to model?
2. **Choose representation**: Matrix or list based on density
3. **Pick algorithm**: Based on problem requirements
4. **Handle edge cases**: Empty graph, single node, disconnected
5. **Optimize if needed**: Consider space/time tradeoffs

**Common Optimizations**:
- Early termination when target found
- Bidirectional BFS for shortest path
- Use visited set to avoid reprocessing
- For dense graphs, matrix might be faster
- For sparse graphs, adjacency list is better

### Debugging Graph Code

1. **Start with small examples**: Draw graph, trace manually
2. **Print/log**: Current vertex, visited set, path
3. **Check base cases**: Empty graph, single node
4. **Verify graph construction**: Print adjacency list
5. **Test disconnected graphs**: Ensure all components handled

## Conclusion

Graphs are one of the most versatile and powerful data structures in computer science. Mastering graph algorithms opens doors to solving a wide variety of complex problems:

**Key Takeaways**:

1. **Understand graph types**: Directed/undirected, weighted/unweighted, cyclic/acyclic
2. **Master core traversals**: DFS and BFS are foundation for most graph algorithms
3. **Know your shortest paths**: BFS for unweighted, Dijkstra for non-negative, Bellman-Ford for negative
4. **Recognize patterns**: Many problems reduce to standard graph problems
5. **Ask clarifying questions**: Graph problems have many variations
6. **Practice, practice, practice**: Graph problems are common in interviews

**Graph Algorithm Hierarchy**:
```
Graph Algorithms
├── Traversal
│   ├── DFS (Stack/Recursion)
│   └── BFS (Queue)
├── Shortest Path
│   ├── Unweighted: BFS
│   ├── Single-source: Dijkstra, Bellman-Ford
│   ├── All-pairs: Floyd-Warshall
│   └── Heuristic: A*
├── Minimum Spanning Tree
│   ├── Kruskal (Union-Find)
│   └── Prim (Priority Queue)
├── Topological Sort
│   ├── DFS-based
│   └── Kahn's (BFS-based)
├── Connectivity
│   ├── Connected Components (DFS/BFS)
│   ├── Strongly Connected (Kosaraju/Tarjan)
│   └── Dynamic Connectivity (Union-Find)
└── Special Properties
    ├── Cycle Detection
    ├── Bipartite Check
    └── Bridges/Articulation Points
```

With a solid understanding of graph theory and these algorithms, you'll be well-equipped to tackle complex problems in algorithm design, system design, and real-world applications!

## Applications

Graphs are used extensively in:

- **Social Networks**: Friend connections, influence analysis
- **Web**: PageRank, web crawling, link analysis
- **Navigation**: GPS routing, shortest paths
- **Networks**: Internet routing, load balancing
- **Compilers**: Dependency analysis, optimization
- **Databases**: Query optimization, relationship modeling
- **AI**: State space search, planning
- **Biology**: Protein interactions, phylogenetic trees
- **Recommendation Systems**: User-item relationships
- **Game Development**: Pathfinding, AI behavior

Graphs truly are everywhere in computer science and software engineering!
