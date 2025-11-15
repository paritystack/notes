# Graph Algorithms

Graph algorithms are fundamental techniques for solving problems that involve relationships and connections between entities. From social networks to GPS navigation, from task scheduling to network optimization, graph algorithms power many of the systems we interact with daily.

## Table of Contents

1. [Introduction](#introduction)
2. [Graph Traversal Algorithms](#graph-traversal-algorithms)
   - [Depth-First Search (DFS)](#depth-first-search-dfs)
   - [Breadth-First Search (BFS)](#breadth-first-search-bfs)
3. [Shortest Path Algorithms](#shortest-path-algorithms)
   - [Dijkstra's Algorithm](#dijkstras-algorithm)
   - [Bellman-Ford Algorithm](#bellman-ford-algorithm)
   - [Floyd-Warshall Algorithm](#floyd-warshall-algorithm)
   - [A* Search Algorithm](#a-star-search-algorithm)
4. [Minimum Spanning Tree](#minimum-spanning-tree)
   - [Kruskal's Algorithm](#kruskals-algorithm)
   - [Prim's Algorithm](#prims-algorithm)
5. [Advanced Graph Algorithms](#advanced-graph-algorithms)
   - [Topological Sort](#topological-sort)
   - [Strongly Connected Components](#strongly-connected-components)
   - [Articulation Points and Bridges](#articulation-points-and-bridges)
   - [Network Flow](#network-flow)
   - [Bipartite Matching](#bipartite-matching)
6. [Algorithm Selection Guide](#algorithm-selection-guide)
7. [Real-World Applications](#real-world-applications)
8. [Common Interview Problems](#common-interview-problems)

## Introduction

### What are Graph Algorithms?

Graph algorithms are computational procedures designed to solve problems modeled as graphs - data structures consisting of vertices (nodes) connected by edges. These algorithms are essential tools in computer science, enabling us to:

- Find optimal paths between locations (GPS, routing)
- Analyze social networks and connections
- Detect communities and clusters
- Solve scheduling and dependency problems
- Optimize network flows and resource allocation
- Identify critical infrastructure points

### Prerequisites

Before diving into graph algorithms, you should be familiar with:
- Basic graph theory concepts (vertices, edges, directed/undirected graphs)
- Graph representations (adjacency matrix, adjacency list)
- Time and space complexity analysis
- Basic data structures (queues, stacks, heaps, hash tables)

For graph data structures, see [data_structures/graphs.md](../data_structures/graphs.md).

### Complexity Notation

Throughout this guide, we use:
- **V**: Number of vertices in the graph
- **E**: Number of edges in the graph
- **O()**: Big-O notation for time/space complexity

---

## Graph Traversal Algorithms

Graph traversal algorithms systematically visit all vertices in a graph. The two fundamental traversal strategies are Depth-First Search (DFS) and Breadth-First Search (BFS).

### Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking. It uses a stack (either explicitly or via recursion) to keep track of vertices to visit.

#### How DFS Works

1. Start at a source vertex and mark it as visited
2. Recursively visit all unvisited neighbors
3. Backtrack when no unvisited neighbors remain
4. Continue until all reachable vertices are visited

#### DFS Implementation (Recursive)

**Python:**
```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        """Add edge from u to v (directed graph)"""
        self.graph[u].append(v)

    def dfs_recursive(self, start):
        """
        Perform DFS traversal starting from vertex 'start'.
        Time: O(V + E)
        Space: O(V) for recursion stack and visited set
        """
        visited = set()
        result = []

        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)

            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_helper(neighbor)

        dfs_helper(start)
        return result

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

print("DFS traversal:", g.dfs_recursive(0))
# Output: [0, 1, 3, 4, 2, 5, 6]
```

**JavaScript:**
```javascript
class Graph {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push(v);
    }

    dfsRecursive(start) {
        const visited = new Set();
        const result = [];

        const dfsHelper = (vertex) => {
            visited.add(vertex);
            result.push(vertex);

            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    dfsHelper(neighbor);
                }
            }
        };

        dfsHelper(start);
        return result;
    }
}

// Example usage
const g = new Graph();
g.addEdge(0, 1);
g.addEdge(0, 2);
g.addEdge(1, 3);
g.addEdge(1, 4);
g.addEdge(2, 5);
g.addEdge(2, 6);

console.log("DFS traversal:", g.dfsRecursive(0));
// Output: [0, 1, 3, 4, 2, 5, 6]
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

using namespace std;

class Graph {
private:
    unordered_map<int, vector<int>> adjacencyList;

    void dfsHelper(int vertex, unordered_set<int>& visited, vector<int>& result) {
        visited.insert(vertex);
        result.push_back(vertex);

        for (int neighbor : adjacencyList[vertex]) {
            if (visited.find(neighbor) == visited.end()) {
                dfsHelper(neighbor, visited, result);
            }
        }
    }

public:
    void addEdge(int u, int v) {
        adjacencyList[u].push_back(v);
    }

    vector<int> dfsRecursive(int start) {
        unordered_set<int> visited;
        vector<int> result;
        dfsHelper(start, visited, result);
        return result;
    }
};

// Example usage
int main() {
    Graph g;
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    vector<int> result = g.dfsRecursive(0);
    cout << "DFS traversal: ";
    for (int v : result) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}
```

#### DFS Implementation (Iterative)

**Python:**
```python
def dfs_iterative(self, start):
    """
    Iterative DFS using explicit stack.
    Time: O(V + E)
    Space: O(V) for stack and visited set
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            # Add neighbors in reverse order for same traversal as recursive
            for neighbor in reversed(self.graph[vertex]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result
```

**JavaScript:**
```javascript
dfsIterative(start) {
    const visited = new Set();
    const stack = [start];
    const result = [];

    while (stack.length > 0) {
        const vertex = stack.pop();

        if (!visited.has(vertex)) {
            visited.add(vertex);
            result.push(vertex);

            const neighbors = this.adjacencyList.get(vertex) || [];
            // Add neighbors in reverse for same order as recursive
            for (let i = neighbors.length - 1; i >= 0; i--) {
                if (!visited.has(neighbors[i])) {
                    stack.push(neighbors[i]);
                }
            }
        }
    }

    return result;
}
```

#### DFS Applications

**1. Cycle Detection in Directed Graphs**

```python
def has_cycle_directed(self):
    """
    Detect cycle in directed graph using DFS.
    Time: O(V + E)
    Space: O(V)
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in self.graph}

    def has_cycle_helper(vertex):
        color[vertex] = GRAY

        for neighbor in self.graph[vertex]:
            if color[neighbor] == GRAY:  # Back edge found
                return True
            if color[neighbor] == WHITE and has_cycle_helper(neighbor):
                return True

        color[vertex] = BLACK
        return False

    for vertex in self.graph:
        if color[vertex] == WHITE:
            if has_cycle_helper(vertex):
                return True

    return False

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)  # Creates a cycle
print("Has cycle:", g.has_cycle_directed())  # True
```

**2. Cycle Detection in Undirected Graphs**

```python
def has_cycle_undirected(self):
    """
    Detect cycle in undirected graph using DFS.
    Time: O(V + E)
    Space: O(V)
    """
    visited = set()

    def has_cycle_helper(vertex, parent):
        visited.add(vertex)

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                if has_cycle_helper(neighbor, vertex):
                    return True
            elif neighbor != parent:  # Back edge to non-parent
                return True

        return False

    for vertex in self.graph:
        if vertex not in visited:
            if has_cycle_helper(vertex, -1):
                return True

    return False
```

**3. Path Finding**

```python
def find_path_dfs(self, start, end):
    """
    Find a path from start to end using DFS.
    Time: O(V + E)
    Space: O(V)
    Returns: List of vertices in path, or None if no path exists
    """
    visited = set()
    path = []

    def dfs_path_helper(vertex):
        visited.add(vertex)
        path.append(vertex)

        if vertex == end:
            return True

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                if dfs_path_helper(neighbor):
                    return True

        path.pop()  # Backtrack
        return False

    if dfs_path_helper(start):
        return path
    return None

# Example
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
print("Path from 0 to 3:", g.find_path_dfs(0, 3))  # [0, 1, 3]
```

**4. Connected Components (Undirected Graph)**

```python
def count_connected_components(self):
    """
    Count connected components in undirected graph.
    Time: O(V + E)
    Space: O(V)
    """
    visited = set()
    count = 0

    def dfs_helper(vertex):
        visited.add(vertex)
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                dfs_helper(neighbor)

    for vertex in self.graph:
        if vertex not in visited:
            dfs_helper(vertex)
            count += 1

    return count
```

**5. Is Graph Bipartite?**

```python
def is_bipartite_dfs(self):
    """
    Check if graph is bipartite using DFS.
    Time: O(V + E)
    Space: O(V)
    """
    color = {}

    def dfs_helper(vertex, c):
        color[vertex] = c

        for neighbor in self.graph[vertex]:
            if neighbor not in color:
                if not dfs_helper(neighbor, 1 - c):
                    return False
            elif color[neighbor] == c:
                return False

        return True

    for vertex in self.graph:
        if vertex not in color:
            if not dfs_helper(vertex, 0):
                return False

    return True
```

**Complexity Analysis:**
- **Time Complexity**: O(V + E) - visits each vertex once and explores each edge once
- **Space Complexity**:
  - Recursive: O(V) for call stack in worst case (linear graph)
  - Iterative: O(V) for explicit stack

**When to Use DFS:**
- Finding paths between vertices
- Cycle detection
- Topological sorting
- Finding strongly connected components
- Maze solving
- Detecting articulation points and bridges

---

### Breadth-First Search (BFS)

BFS explores the graph level by level, visiting all neighbors of a vertex before moving to their neighbors. It uses a queue to maintain the order of exploration.

#### How BFS Works

1. Start at source vertex, mark it as visited
2. Add source to queue
3. While queue is not empty:
   - Dequeue a vertex
   - Visit all unvisited neighbors
   - Enqueue each unvisited neighbor and mark as visited

#### BFS Implementation

**Python:**
```python
from collections import deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs(self, start):
        """
        Perform BFS traversal starting from vertex 'start'.
        Time: O(V + E)
        Space: O(V) for queue and visited set
        """
        visited = set([start])
        queue = deque([start])
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 5)
g.add_edge(2, 6)

print("BFS traversal:", g.bfs(0))
# Output: [0, 1, 2, 3, 4, 5, 6]
```

**JavaScript:**
```javascript
class Graph {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push(v);
    }

    bfs(start) {
        const visited = new Set([start]);
        const queue = [start];
        const result = [];

        while (queue.length > 0) {
            const vertex = queue.shift();
            result.push(vertex);

            const neighbors = this.adjacencyList.get(vertex) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        return result;
    }
}

// Example usage
const g = new Graph();
g.addEdge(0, 1);
g.addEdge(0, 2);
g.addEdge(1, 3);
g.addEdge(1, 4);
g.addEdge(2, 5);
g.addEdge(2, 6);

console.log("BFS traversal:", g.bfs(0));
// Output: [0, 1, 2, 3, 4, 5, 6]
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

using namespace std;

class Graph {
private:
    unordered_map<int, vector<int>> adjacencyList;

public:
    void addEdge(int u, int v) {
        adjacencyList[u].push_back(v);
    }

    vector<int> bfs(int start) {
        unordered_set<int> visited;
        queue<int> q;
        vector<int> result;

        visited.insert(start);
        q.push(start);

        while (!q.empty()) {
            int vertex = q.front();
            q.pop();
            result.push_back(vertex);

            for (int neighbor : adjacencyList[vertex]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }

        return result;
    }
};

// Example usage
int main() {
    Graph g;
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    vector<int> result = g.bfs(0);
    cout << "BFS traversal: ";
    for (int v : result) {
        cout << v << " ";
    }
    cout << endl;

    return 0;
}
```

#### BFS Applications

**1. Shortest Path in Unweighted Graph**

```python
def shortest_path_bfs(self, start, end):
    """
    Find shortest path in unweighted graph using BFS.
    Time: O(V + E)
    Space: O(V)
    Returns: (distance, path)
    """
    if start == end:
        return (0, [start])

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]

                if neighbor == end:
                    return (len(new_path) - 1, new_path)

                queue.append((neighbor, new_path))

    return (float('inf'), None)  # No path exists

# Example
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(2, 3)
g.add_edge(3, 4)

dist, path = g.shortest_path_bfs(0, 4)
print(f"Shortest distance: {dist}")  # 3
print(f"Shortest path: {path}")      # [0, 1, 3, 4] or [0, 2, 3, 4]
```

**2. Level Order Traversal**

```python
def level_order_traversal(self, start):
    """
    Return vertices grouped by level (distance from start).
    Time: O(V + E)
    Space: O(V)
    """
    visited = set([start])
    queue = deque([(start, 0)])
    levels = defaultdict(list)

    while queue:
        vertex, level = queue.popleft()
        levels[level].append(vertex)

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))

    return dict(levels)

# Example output:
# {0: [0], 1: [1, 2], 2: [3, 4, 5, 6]}
```

**3. Is Graph Bipartite (BFS Version)**

```python
def is_bipartite_bfs(self):
    """
    Check if graph is bipartite using BFS.
    Time: O(V + E)
    Space: O(V)
    """
    color = {}

    for start_vertex in self.graph:
        if start_vertex in color:
            continue

        queue = deque([start_vertex])
        color[start_vertex] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in self.graph[vertex]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return False

    return True
```

**4. All Nodes at Distance K**

```python
def nodes_at_distance_k(self, start, k):
    """
    Find all nodes at exactly distance k from start.
    Time: O(V + E)
    Space: O(V)
    """
    if k == 0:
        return [start]

    visited = set([start])
    queue = deque([(start, 0)])
    result = []

    while queue:
        vertex, dist = queue.popleft()

        if dist == k:
            result.append(vertex)
            continue  # Don't explore further

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return result
```

**5. Minimum Number of Edges to Traverse**

```python
def min_edges_to_traverse(self, start, end):
    """
    Find minimum number of edges to traverse from start to end.
    Time: O(V + E)
    Space: O(V)
    """
    if start == end:
        return 0

    visited = set([start])
    queue = deque([(start, 0)])

    while queue:
        vertex, distance = queue.popleft()

        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                if neighbor == end:
                    return distance + 1

                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return -1  # No path exists
```

**Complexity Analysis:**
- **Time Complexity**: O(V + E) - visits each vertex once and explores each edge once
- **Space Complexity**: O(V) for queue and visited set

**When to Use BFS:**
- Finding shortest path in unweighted graphs
- Level-order traversal
- Finding all nodes at a given distance
- Finding minimum spanning tree for unweighted graph
- Web crawlers (breadth-first exploration)
- Social network analysis (finding connections)

---

### DFS vs BFS Comparison

| Aspect | DFS | BFS |
|--------|-----|-----|
| Data Structure | Stack (or recursion) | Queue |
| Memory Usage | Better for deep graphs | Better for wide graphs |
| Path Finding | Finds a path (not necessarily shortest) | Finds shortest path (unweighted) |
| Completeness | May not terminate in infinite graphs | Complete for finite graphs |
| Optimality | Not optimal | Optimal for unweighted graphs |
| Implementation | Simpler (recursive) | Requires queue |
| Use Cases | Topological sort, cycle detection, puzzles | Shortest path, level-order, nearest neighbors |

---

## Shortest Path Algorithms

Shortest path algorithms find the minimum-cost path between vertices in a weighted graph. Different algorithms suit different scenarios based on graph properties.

### Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a graph with **non-negative** edge weights. It uses a greedy approach, always selecting the unvisited vertex with the smallest distance.

#### How Dijkstra's Works

1. Initialize distances to all vertices as infinity, except source (distance = 0)
2. Use a priority queue (min-heap) to store vertices by current distance
3. While priority queue is not empty:
   - Extract vertex with minimum distance
   - For each neighbor, if a shorter path is found, update distance
4. Return the distances array

#### Dijkstra's Implementation

**Python (with Priority Queue):**
```python
import heapq
from collections import defaultdict

class WeightedGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v, weight):
        """Add weighted edge from u to v"""
        self.graph[u].append((v, weight))

    def dijkstra(self, start):
        """
        Dijkstra's algorithm for single-source shortest path.
        Time: O((V + E) log V) with binary heap
        Space: O(V)
        Works only with non-negative weights!
        """
        # Distance from start to each vertex
        distances = {vertex: float('inf') for vertex in self.graph}
        distances[start] = 0

        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()

        # To reconstruct paths
        previous = {vertex: None for vertex in self.graph}

        while pq:
            current_dist, current_vertex = heapq.heappop(pq)

            if current_vertex in visited:
                continue

            visited.add(current_vertex)

            # Explore neighbors
            for neighbor, weight in self.graph[current_vertex]:
                distance = current_dist + weight

                # If found shorter path, update
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (distance, neighbor))

        return distances, previous

    def get_shortest_path(self, start, end):
        """
        Get the actual shortest path from start to end.
        Returns: (total_distance, path)
        """
        distances, previous = self.dijkstra(start)

        # Reconstruct path
        path = []
        current = end

        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()

        # Check if path exists
        if path[0] != start:
            return (float('inf'), None)

        return (distances[end], path)

# Example usage
g = WeightedGraph()
g.add_edge('A', 'B', 4)
g.add_edge('A', 'C', 2)
g.add_edge('B', 'C', 1)
g.add_edge('B', 'D', 5)
g.add_edge('C', 'D', 8)
g.add_edge('C', 'E', 10)
g.add_edge('D', 'E', 2)

distances, _ = g.dijkstra('A')
print("Shortest distances from A:")
for vertex, dist in sorted(distances.items()):
    print(f"  {vertex}: {dist}")

dist, path = g.get_shortest_path('A', 'E')
print(f"\nShortest path A -> E: {path} (distance: {dist})")
# Output: ['A', 'C', 'B', 'D', 'E'] (distance: 10)
```

**JavaScript:**
```javascript
class PriorityQueue {
    constructor() {
        this.values = [];
    }

    enqueue(val, priority) {
        this.values.push({ val, priority });
        this.sort();
    }

    dequeue() {
        return this.values.shift();
    }

    sort() {
        this.values.sort((a, b) => a.priority - b.priority);
    }

    isEmpty() {
        return this.values.length === 0;
    }
}

class WeightedGraph {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v, weight) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push({ node: v, weight });
    }

    dijkstra(start) {
        const distances = new Map();
        const previous = new Map();
        const pq = new PriorityQueue();
        const visited = new Set();

        // Initialize distances
        for (const vertex of this.adjacencyList.keys()) {
            distances.set(vertex, Infinity);
            previous.set(vertex, null);
        }
        distances.set(start, 0);
        pq.enqueue(start, 0);

        while (!pq.isEmpty()) {
            const { val: current } = pq.dequeue();

            if (visited.has(current)) continue;
            visited.add(current);

            const neighbors = this.adjacencyList.get(current) || [];
            for (const { node: neighbor, weight } of neighbors) {
                const distance = distances.get(current) + weight;

                if (distance < distances.get(neighbor)) {
                    distances.set(neighbor, distance);
                    previous.set(neighbor, current);
                    pq.enqueue(neighbor, distance);
                }
            }
        }

        return { distances, previous };
    }

    getShortestPath(start, end) {
        const { distances, previous } = this.dijkstra(start);
        const path = [];
        let current = end;

        while (current !== null) {
            path.unshift(current);
            current = previous.get(current);
        }

        if (path[0] !== start) {
            return { distance: Infinity, path: null };
        }

        return { distance: distances.get(end), path };
    }
}

// Example usage
const g = new WeightedGraph();
g.addEdge('A', 'B', 4);
g.addEdge('A', 'C', 2);
g.addEdge('B', 'C', 1);
g.addEdge('B', 'D', 5);
g.addEdge('C', 'D', 8);
g.addEdge('C', 'E', 10);
g.addEdge('D', 'E', 2);

const result = g.getShortestPath('A', 'E');
console.log('Shortest path A -> E:', result.path);
console.log('Distance:', result.distance);
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <limits>
#include <algorithm>

using namespace std;

class WeightedGraph {
private:
    unordered_map<string, vector<pair<string, int>>> adjacencyList;

public:
    void addEdge(const string& u, const string& v, int weight) {
        adjacencyList[u].push_back({v, weight});
    }

    pair<unordered_map<string, int>, unordered_map<string, string>> dijkstra(const string& start) {
        unordered_map<string, int> distances;
        unordered_map<string, string> previous;

        // Initialize distances
        for (const auto& pair : adjacencyList) {
            distances[pair.first] = numeric_limits<int>::max();
            previous[pair.first] = "";
        }
        distances[start] = 0;

        // Priority queue: (distance, vertex)
        priority_queue<pair<int, string>,
                       vector<pair<int, string>>,
                       greater<pair<int, string>>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            auto [currentDist, current] = pq.top();
            pq.pop();

            if (currentDist > distances[current]) continue;

            for (const auto& [neighbor, weight] : adjacencyList[current]) {
                int distance = currentDist + weight;

                if (distance < distances[neighbor]) {
                    distances[neighbor] = distance;
                    previous[neighbor] = current;
                    pq.push({distance, neighbor});
                }
            }
        }

        return {distances, previous};
    }

    pair<int, vector<string>> getShortestPath(const string& start, const string& end) {
        auto [distances, previous] = dijkstra(start);

        vector<string> path;
        string current = end;

        while (!current.empty()) {
            path.push_back(current);
            current = previous[current];
        }

        reverse(path.begin(), path.end());

        if (path[0] != start) {
            return {numeric_limits<int>::max(), {}};
        }

        return {distances[end], path};
    }
};

// Example usage
int main() {
    WeightedGraph g;
    g.addEdge("A", "B", 4);
    g.addEdge("A", "C", 2);
    g.addEdge("B", "C", 1);
    g.addEdge("B", "D", 5);
    g.addEdge("C", "D", 8);
    g.addEdge("C", "E", 10);
    g.addEdge("D", "E", 2);

    auto [distance, path] = g.getShortestPath("A", "E");

    cout << "Shortest path A -> E: ";
    for (const auto& v : path) {
        cout << v << " ";
    }
    cout << "\nDistance: " << distance << endl;

    return 0;
}
```

#### Dijkstra's with Different Priority Queue Implementations

**Using heapq in Python (Most Common):**
```python
def dijkstra_optimized(self, start):
    """
    Optimized Dijkstra using heapq.
    Time: O((V + E) log V)
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight in self.graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return dict(distances)
```

**Complexity Analysis:**
- **Time Complexity**:
  - With binary heap: O((V + E) log V)
  - With Fibonacci heap: O(E + V log V) [theoretical, rarely used in practice]
  - Without heap (naive): O(V²)
- **Space Complexity**: O(V) for distances array and priority queue

**When to Use Dijkstra's:**
- Single-source shortest path in graphs with non-negative weights
- GPS navigation systems
- Network routing protocols (OSPF)
- Finding cheapest route in transportation
- Game AI pathfinding (when all costs are positive)

**When NOT to Use Dijkstra's:**
- Graphs with negative edge weights (use Bellman-Ford instead)
- Need all-pairs shortest paths (use Floyd-Warshall instead)
- Very large graphs where memory is constrained

---

### Bellman-Ford Algorithm

Bellman-Ford finds shortest paths from a source vertex to all other vertices, even with **negative edge weights**. It can also detect negative cycles.

#### How Bellman-Ford Works

1. Initialize distances to all vertices as infinity, except source (distance = 0)
2. Relax all edges V-1 times:
   - For each edge (u, v) with weight w:
     - If dist[u] + w < dist[v], update dist[v]
3. Check for negative cycles by relaxing edges one more time
4. Return distances (or report negative cycle)

#### Why V-1 Iterations?

In a graph with V vertices, the shortest path between any two vertices contains at most V-1 edges. Each iteration guarantees finding shortest paths with one more edge.

#### Bellman-Ford Implementation

**Python:**
```python
class WeightedGraph:
    def __init__(self):
        self.vertices = set()
        self.edges = []  # List of (u, v, weight)

    def add_vertex(self, v):
        self.vertices.add(v)

    def add_edge(self, u, v, weight):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges.append((u, v, weight))

    def bellman_ford(self, start):
        """
        Bellman-Ford algorithm for single-source shortest path.
        Handles negative weights and detects negative cycles.
        Time: O(V * E)
        Space: O(V)
        Returns: (distances, has_negative_cycle)
        """
        # Initialize distances
        distances = {vertex: float('inf') for vertex in self.vertices}
        distances[start] = 0
        previous = {vertex: None for vertex in self.vertices}

        # Relax edges V-1 times
        for _ in range(len(self.vertices) - 1):
            for u, v, weight in self.edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

        # Check for negative cycles
        for u, v, weight in self.edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                return (distances, previous, True)  # Negative cycle detected

        return (distances, previous, False)

# Example with negative weights
g = WeightedGraph()
g.add_edge('A', 'B', 4)
g.add_edge('A', 'C', 2)
g.add_edge('B', 'C', -3)  # Negative weight
g.add_edge('B', 'D', 5)
g.add_edge('C', 'D', 1)

distances, previous, has_neg_cycle = g.bellman_ford('A')
print("Shortest distances from A:")
for vertex, dist in sorted(distances.items()):
    print(f"  {vertex}: {dist}")
print(f"Has negative cycle: {has_neg_cycle}")

# Example with negative cycle
g2 = WeightedGraph()
g2.add_edge('A', 'B', 1)
g2.add_edge('B', 'C', -3)
g2.add_edge('C', 'A', 1)  # Creates negative cycle: A->B->C->A = -1

distances, previous, has_neg_cycle = g2.bellman_ford('A')
print(f"\nNegative cycle detected: {has_neg_cycle}")  # True
```

**JavaScript:**
```javascript
class WeightedGraph {
    constructor() {
        this.vertices = new Set();
        this.edges = [];  // Array of {u, v, weight}
    }

    addVertex(v) {
        this.vertices.add(v);
    }

    addEdge(u, v, weight) {
        this.vertices.add(u);
        this.vertices.add(v);
        this.edges.push({ u, v, weight });
    }

    bellmanFord(start) {
        // Initialize distances
        const distances = new Map();
        const previous = new Map();

        for (const vertex of this.vertices) {
            distances.set(vertex, Infinity);
            previous.set(vertex, null);
        }
        distances.set(start, 0);

        const V = this.vertices.size;

        // Relax edges V-1 times
        for (let i = 0; i < V - 1; i++) {
            for (const { u, v, weight } of this.edges) {
                if (distances.get(u) !== Infinity &&
                    distances.get(u) + weight < distances.get(v)) {
                    distances.set(v, distances.get(u) + weight);
                    previous.set(v, u);
                }
            }
        }

        // Check for negative cycles
        for (const { u, v, weight } of this.edges) {
            if (distances.get(u) !== Infinity &&
                distances.get(u) + weight < distances.get(v)) {
                return { distances, previous, hasNegativeCycle: true };
            }
        }

        return { distances, previous, hasNegativeCycle: false };
    }
}

// Example usage
const g = new WeightedGraph();
g.addEdge('A', 'B', 4);
g.addEdge('A', 'C', 2);
g.addEdge('B', 'C', -3);
g.addEdge('B', 'D', 5);
g.addEdge('C', 'D', 1);

const result = g.bellmanFord('A');
console.log('Shortest distances from A:');
for (const [vertex, dist] of result.distances) {
    console.log(`  ${vertex}: ${dist}`);
}
console.log('Has negative cycle:', result.hasNegativeCycle);
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <string>

using namespace std;

struct Edge {
    string u, v;
    int weight;
};

class WeightedGraph {
private:
    unordered_set<string> vertices;
    vector<Edge> edges;

public:
    void addVertex(const string& v) {
        vertices.insert(v);
    }

    void addEdge(const string& u, const string& v, int weight) {
        vertices.insert(u);
        vertices.insert(v);
        edges.push_back({u, v, weight});
    }

    tuple<unordered_map<string, int>, unordered_map<string, string>, bool>
    bellmanFord(const string& start) {
        unordered_map<string, int> distances;
        unordered_map<string, string> previous;

        // Initialize
        for (const auto& vertex : vertices) {
            distances[vertex] = numeric_limits<int>::max();
            previous[vertex] = "";
        }
        distances[start] = 0;

        int V = vertices.size();

        // Relax edges V-1 times
        for (int i = 0; i < V - 1; i++) {
            for (const auto& edge : edges) {
                if (distances[edge.u] != numeric_limits<int>::max() &&
                    distances[edge.u] + edge.weight < distances[edge.v]) {
                    distances[edge.v] = distances[edge.u] + edge.weight;
                    previous[edge.v] = edge.u;
                }
            }
        }

        // Check for negative cycle
        for (const auto& edge : edges) {
            if (distances[edge.u] != numeric_limits<int>::max() &&
                distances[edge.u] + edge.weight < distances[edge.v]) {
                return {distances, previous, true};  // Negative cycle
            }
        }

        return {distances, previous, false};
    }
};

int main() {
    WeightedGraph g;
    g.addEdge("A", "B", 4);
    g.addEdge("A", "C", 2);
    g.addEdge("B", "C", -3);
    g.addEdge("B", "D", 5);
    g.addEdge("C", "D", 1);

    auto [distances, previous, hasNegCycle] = g.bellmanFord("A");

    cout << "Shortest distances from A:" << endl;
    for (const auto& [vertex, dist] : distances) {
        cout << "  " << vertex << ": " << dist << endl;
    }
    cout << "Has negative cycle: " << (hasNegCycle ? "true" : "false") << endl;

    return 0;
}
```

#### Bellman-Ford with Path Reconstruction

```python
def bellman_ford_with_path(self, start, end):
    """
    Get shortest path from start to end using Bellman-Ford.
    Returns: (distance, path, has_negative_cycle)
    """
    distances, previous, has_neg_cycle = self.bellman_ford(start)

    if has_neg_cycle:
        return (None, None, True)

    # Reconstruct path
    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if path[0] != start:
        return (float('inf'), None, False)

    return (distances[end], path, False)
```

#### Finding Negative Cycles

```python
def find_negative_cycle(self):
    """
    Find a negative cycle if one exists.
    Returns: List of vertices in cycle, or None
    """
    # Run Bellman-Ford from arbitrary vertex
    start = next(iter(self.vertices))
    distances = {vertex: float('inf') for vertex in self.vertices}
    distances[start] = 0
    previous = {vertex: None for vertex in self.vertices}

    # Relax edges V-1 times
    for _ in range(len(self.vertices) - 1):
        for u, v, weight in self.edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                previous[v] = u

    # Find vertex that is part of negative cycle
    cycle_vertex = None
    for u, v, weight in self.edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            cycle_vertex = v
            break

    if cycle_vertex is None:
        return None  # No negative cycle

    # Trace back to find the cycle
    # Go back V steps to ensure we're in the cycle
    for _ in range(len(self.vertices)):
        cycle_vertex = previous[cycle_vertex]

    # Reconstruct cycle
    cycle = [cycle_vertex]
    current = previous[cycle_vertex]
    while current != cycle_vertex:
        cycle.append(current)
        current = previous[current]
    cycle.reverse()

    return cycle
```

**Complexity Analysis:**
- **Time Complexity**: O(V * E)
  - V-1 iterations, each checking all E edges
  - Much slower than Dijkstra for sparse graphs
- **Space Complexity**: O(V) for distances array

**When to Use Bellman-Ford:**
- Graph has negative edge weights
- Need to detect negative cycles
- Simpler implementation than Dijkstra (no priority queue needed)
- Distributed systems (can be parallelized)

**When NOT to Use Bellman-Ford:**
- All weights are non-negative (use Dijkstra instead - much faster)
- Need all-pairs shortest paths (use Floyd-Warshall)
- Very large graphs (too slow)

---

### Floyd-Warshall Algorithm

Floyd-Warshall finds shortest paths between **all pairs** of vertices. It can handle negative weights but not negative cycles.

#### How Floyd-Warshall Works

Uses dynamic programming with this key insight:
- For each pair of vertices (i, j), consider all intermediate vertices k
- If path i → k → j is shorter than direct path i → j, update it

**Recurrence relation:**
```
dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1])
```

This can be optimized to use O(V²) space instead of O(V³).

#### Floyd-Warshall Implementation

**Python:**
```python
def floyd_warshall(self):
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.
    Time: O(V³)
    Space: O(V²)
    Returns: 2D distance matrix
    """
    # Get all vertices
    vertices = list(self.vertices)
    n = len(vertices)

    # Create index mapping
    vertex_index = {v: i for i, v in enumerate(vertices)}

    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # Distance from vertex to itself is 0
    for i in range(n):
        dist[i][i] = 0

    # Fill in edge weights
    for u, v, weight in self.edges:
        i, j = vertex_index[u], vertex_index[v]
        dist[i][j] = weight

    # Floyd-Warshall main algorithm
    for k in range(n):  # Intermediate vertex
        for i in range(n):  # Source
            for j in range(n):  # Destination
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("Graph contains negative cycle")

    return dist, vertices

# Example usage
g = WeightedGraph()
g.add_edge('A', 'B', 3)
g.add_edge('A', 'C', 8)
g.add_edge('A', 'E', -4)
g.add_edge('B', 'D', 1)
g.add_edge('B', 'E', 7)
g.add_edge('C', 'B', 4)
g.add_edge('D', 'A', 2)
g.add_edge('D', 'C', -5)
g.add_edge('E', 'D', 6)

dist, vertices = g.floyd_warshall()

print("All-pairs shortest distances:")
print("     ", "  ".join(f"{v:>3}" for v in vertices))
for i, v1 in enumerate(vertices):
    row = [f"{dist[i][j]:>3}" if dist[i][j] != float('inf') else "INF"
           for j in range(len(vertices))]
    print(f"{v1:>3}: ", "  ".join(row))
```

**JavaScript:**
```javascript
class WeightedGraph {
    constructor() {
        this.vertices = new Set();
        this.edges = [];
    }

    addVertex(v) {
        this.vertices.add(v);
    }

    addEdge(u, v, weight) {
        this.vertices.add(u);
        this.vertices.add(v);
        this.edges.push({ u, v, weight });
    }

    floydWarshall() {
        const vertices = Array.from(this.vertices);
        const n = vertices.length;
        const vertexIndex = new Map(vertices.map((v, i) => [v, i]));

        // Initialize distance matrix
        const INF = Infinity;
        const dist = Array(n).fill(null).map(() => Array(n).fill(INF));

        // Distance from vertex to itself is 0
        for (let i = 0; i < n; i++) {
            dist[i][i] = 0;
        }

        // Fill in edge weights
        for (const { u, v, weight } of this.edges) {
            const i = vertexIndex.get(u);
            const j = vertexIndex.get(v);
            dist[i][j] = weight;
        }

        // Floyd-Warshall main algorithm
        for (let k = 0; k < n; k++) {
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (dist[i][k] !== INF && dist[k][j] !== INF) {
                        dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }

        // Check for negative cycles
        for (let i = 0; i < n; i++) {
            if (dist[i][i] < 0) {
                throw new Error("Graph contains negative cycle");
            }
        }

        return { dist, vertices };
    }

    getShortestPath(start, end) {
        const { dist, vertices } = this.floydWarshall();
        const vertexIndex = new Map(vertices.map((v, i) => [v, i]));

        const i = vertexIndex.get(start);
        const j = vertexIndex.get(end);

        return dist[i][j];
    }
}
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <limits>
#include <stdexcept>

using namespace std;

class WeightedGraph {
private:
    unordered_set<string> vertices;
    vector<tuple<string, string, int>> edges;

public:
    void addVertex(const string& v) {
        vertices.insert(v);
    }

    void addEdge(const string& u, const string& v, int weight) {
        vertices.insert(u);
        vertices.insert(v);
        edges.push_back({u, v, weight});
    }

    pair<vector<vector<int>>, vector<string>> floydWarshall() {
        vector<string> vertexList(vertices.begin(), vertices.end());
        int n = vertexList.size();

        unordered_map<string, int> vertexIndex;
        for (int i = 0; i < n; i++) {
            vertexIndex[vertexList[i]] = i;
        }

        const int INF = numeric_limits<int>::max() / 2;
        vector<vector<int>> dist(n, vector<int>(n, INF));

        // Distance from vertex to itself is 0
        for (int i = 0; i < n; i++) {
            dist[i][i] = 0;
        }

        // Fill in edge weights
        for (const auto& [u, v, weight] : edges) {
            int i = vertexIndex[u];
            int j = vertexIndex[v];
            dist[i][j] = weight;
        }

        // Floyd-Warshall
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }

        // Check for negative cycles
        for (int i = 0; i < n; i++) {
            if (dist[i][i] < 0) {
                throw runtime_error("Graph contains negative cycle");
            }
        }

        return {dist, vertexList};
    }
};
```

#### Floyd-Warshall with Path Reconstruction

```python
def floyd_warshall_with_path(self):
    """
    Floyd-Warshall with path reconstruction.
    Returns: (dist_matrix, next_matrix, vertices)
    """
    vertices = list(self.vertices)
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_vertex = [[None] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        next_vertex[i][i] = i

    # Initialize with edges
    for u, v, weight in self.edges:
        i, j = vertex_index[u], vertex_index[v]
        dist[i][j] = weight
        next_vertex[i][j] = j

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]

    return dist, next_vertex, vertices

def get_path(self, start, end):
    """Reconstruct path from start to end."""
    dist, next_vertex, vertices = self.floyd_warshall_with_path()
    vertex_index = {v: i for i, v in enumerate(vertices)}

    i, j = vertex_index[start], vertex_index[end]

    if next_vertex[i][j] is None:
        return None  # No path exists

    path = [start]
    while i != j:
        i = next_vertex[i][j]
        path.append(vertices[i])

    return path
```

**Complexity Analysis:**
- **Time Complexity**: O(V³) - three nested loops
- **Space Complexity**: O(V²) - distance matrix

**When to Use Floyd-Warshall:**
- Need all-pairs shortest paths
- Dense graphs (E ≈ V²)
- Small to medium-sized graphs
- Transitive closure problems
- Graph diameter calculation

**When NOT to Use Floyd-Warshall:**
- Only need single-source shortest paths (use Dijkstra or Bellman-Ford)
- Very large graphs (O(V³) is too slow)
- Sparse graphs (running Dijkstra V times may be faster)

---

### A* Search Algorithm

A* (A-star) is an informed search algorithm that finds the shortest path using heuristics. It's widely used in game development, robotics, and GPS navigation.

#### How A* Works

A* combines:
- **g(n)**: Actual cost from start to node n
- **h(n)**: Heuristic estimate of cost from n to goal
- **f(n) = g(n) + h(n)**: Total estimated cost

The algorithm prioritizes exploring nodes with lowest f(n).

#### Admissible Heuristics

A heuristic h(n) is admissible if it never overestimates the actual cost. Common heuristics:

1. **Manhattan Distance** (grid, 4-directional movement):
   ```python
   h(n) = |n.x - goal.x| + |n.y - goal.y|
   ```

2. **Euclidean Distance** (any movement):
   ```python
   h(n) = sqrt((n.x - goal.x)² + (n.y - goal.y)²)
   ```

3. **Chebyshev Distance** (grid, 8-directional movement):
   ```python
   h(n) = max(|n.x - goal.x|, |n.y - goal.y|)
   ```

#### A* Implementation

**Python (Grid-based pathfinding):**
```python
import heapq
from typing import List, Tuple, Set

class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position  # (x, y)
        self.g = g  # Cost from start
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def a_star_grid(grid: List[List[int]], start: Tuple[int, int],
                goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* pathfinding on a 2D grid.

    Args:
        grid: 2D list where 0 = walkable, 1 = obstacle
        start: Starting position (x, y)
        goal: Goal position (x, y)

    Returns:
        List of positions from start to goal, or None if no path exists

    Time: O(b^d) where b is branching factor, d is depth
    Space: O(b^d)
    """
    rows, cols = len(grid), len(grid[0])

    # Validate start and goal
    if (grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1):
        return None  # Start or goal is obstacle

    # Priority queue: (f_score, counter, node)
    # Counter ensures FIFO order for equal f_scores
    counter = 0
    start_node = Node(start, 0, manhattan_distance(start, goal))
    open_set = [(start_node.f, counter, start_node)]
    counter += 1

    # Track visited nodes
    closed_set: Set[Tuple[int, int]] = set()
    # Track best g_score for each position
    g_scores = {start: 0}

    # 4-directional movement (up, down, left, right)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_set:
        _, _, current = heapq.heappop(open_set)

        # Goal reached
        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        # Skip if already visited with better path
        if current.position in closed_set:
            continue

        closed_set.add(current.position)

        # Explore neighbors
        for dx, dy in directions:
            neighbor_pos = (current.position[0] + dx, current.position[1] + dy)

            # Check bounds
            if not (0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols):
                continue

            # Check obstacle
            if grid[neighbor_pos[0]][neighbor_pos[1]] == 1:
                continue

            # Skip if already visited
            if neighbor_pos in closed_set:
                continue

            # Calculate costs
            tentative_g = current.g + 1  # Assuming uniform cost of 1

            # Skip if not a better path
            if neighbor_pos in g_scores and tentative_g >= g_scores[neighbor_pos]:
                continue

            # This is the best path so far
            g_scores[neighbor_pos] = tentative_g
            h = manhattan_distance(neighbor_pos, goal)
            neighbor_node = Node(neighbor_pos, tentative_g, h, current)

            heapq.heappush(open_set, (neighbor_node.f, counter, neighbor_node))
            counter += 1

    return None  # No path found

# Example usage
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

path = a_star_grid(grid, start, goal)
if path:
    print(f"Path found: {path}")
    print(f"Path length: {len(path)}")

    # Visualize path
    grid_copy = [row[:] for row in grid]
    for x, y in path:
        grid_copy[x][y] = '*'
    grid_copy[start[0]][start[1]] = 'S'
    grid_copy[goal[0]][goal[1]] = 'G'

    for row in grid_copy:
        print(' '.join(str(cell) for cell in row))
else:
    print("No path found")
```

**JavaScript (Grid-based):**
```javascript
class Node {
    constructor(position, g = 0, h = 0, parent = null) {
        this.position = position;  // {x, y}
        this.g = g;
        this.h = h;
        this.f = g + h;
        this.parent = parent;
    }
}

class PriorityQueue {
    constructor() {
        this.values = [];
    }

    enqueue(element, priority) {
        this.values.push({ element, priority });
        this.sort();
    }

    dequeue() {
        return this.values.shift();
    }

    sort() {
        this.values.sort((a, b) => a.priority - b.priority);
    }

    isEmpty() {
        return this.values.length === 0;
    }
}

function manhattanDistance(pos1, pos2) {
    return Math.abs(pos1.x - pos2.x) + Math.abs(pos1.y - pos2.y);
}

function aStarGrid(grid, start, goal) {
    const rows = grid.length;
    const cols = grid[0].length;

    // Validate
    if (grid[start.x][start.y] === 1 || grid[goal.x][goal.y] === 1) {
        return null;
    }

    const startNode = new Node(start, 0, manhattanDistance(start, goal));
    const openSet = new PriorityQueue();
    openSet.enqueue(startNode, startNode.f);

    const closedSet = new Set();
    const gScores = new Map();
    gScores.set(`${start.x},${start.y}`, 0);

    const directions = [{x: 0, y: 1}, {x: 1, y: 0}, {x: 0, y: -1}, {x: -1, y: 0}];

    while (!openSet.isEmpty()) {
        const { element: current } = openSet.dequeue();
        const currentKey = `${current.position.x},${current.position.y}`;

        // Goal reached
        if (current.position.x === goal.x && current.position.y === goal.y) {
            const path = [];
            let node = current;
            while (node) {
                path.unshift(node.position);
                node = node.parent;
            }
            return path;
        }

        if (closedSet.has(currentKey)) continue;
        closedSet.add(currentKey);

        // Explore neighbors
        for (const dir of directions) {
            const neighborPos = {
                x: current.position.x + dir.x,
                y: current.position.y + dir.y
            };
            const neighborKey = `${neighborPos.x},${neighborPos.y}`;

            // Check bounds
            if (neighborPos.x < 0 || neighborPos.x >= rows ||
                neighborPos.y < 0 || neighborPos.y >= cols) {
                continue;
            }

            // Check obstacle
            if (grid[neighborPos.x][neighborPos.y] === 1) continue;
            if (closedSet.has(neighborKey)) continue;

            const tentativeG = current.g + 1;

            if (gScores.has(neighborKey) && tentativeG >= gScores.get(neighborKey)) {
                continue;
            }

            gScores.set(neighborKey, tentativeG);
            const h = manhattanDistance(neighborPos, goal);
            const neighborNode = new Node(neighborPos, tentativeG, h, current);
            openSet.enqueue(neighborNode, neighborNode.f);
        }
    }

    return null;
}
```

#### A* for Weighted Graphs

```python
def a_star_graph(self, start, goal, heuristic):
    """
    A* for weighted graphs with custom heuristic.

    Args:
        start: Starting vertex
        goal: Goal vertex
        heuristic: Function that takes a vertex and returns estimated cost to goal

    Returns:
        (distance, path)
    """
    # Priority queue: (f_score, g_score, vertex, path)
    pq = [(heuristic(start), 0, start, [start])]
    visited = set()
    g_scores = {start: 0}

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == goal:
            return (g, path)

        if current in visited:
            continue

        visited.add(current)

        for neighbor, weight in self.graph[current]:
            tentative_g = g + weight

            if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                continue

            g_scores[neighbor] = tentative_g
            h = heuristic(neighbor)
            f = tentative_g + h

            heapq.heappush(pq, (f, tentative_g, neighbor, path + [neighbor]))

    return (float('inf'), None)

# Example with geographic coordinates
class CityGraph:
    def __init__(self):
        self.graph = {}
        self.coordinates = {}  # vertex -> (lat, lon)

    def add_city(self, name, lat, lon):
        self.graph[name] = []
        self.coordinates[name] = (lat, lon)

    def add_road(self, city1, city2, distance):
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))

    def haversine_heuristic(self, city, goal):
        """Calculate great-circle distance between two cities."""
        from math import radians, sin, cos, sqrt, atan2

        lat1, lon1 = self.coordinates[city]
        lat2, lon2 = self.coordinates[goal]

        R = 6371  # Earth radius in km

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c
```

**Complexity Analysis:**
- **Time Complexity**: O(b^d) worst case, where b = branching factor, d = depth
  - With good heuristic: Much better than BFS/Dijkstra in practice
  - With perfect heuristic: O(d)
- **Space Complexity**: O(b^d) - stores nodes in open set

**Heuristic Quality:**
- **Admissible**: h(n) ≤ actual cost → A* guarantees optimal solution
- **Consistent**: h(n) ≤ cost(n, neighbor) + h(neighbor) → More efficient
- **Better heuristic** → Fewer nodes explored → Faster execution

**When to Use A*:**
- Pathfinding in games (character movement, enemy AI)
- GPS navigation with known destination
- Robotics path planning
- Puzzle solving (8-puzzle, Rubik's cube)
- Any scenario where you have domain knowledge for heuristics

**When NOT to Use A*:**
- No good heuristic available (use Dijkstra)
- Need all shortest paths (use Floyd-Warshall)
- Graph is very small (overhead not worth it)

---

### Shortest Path Algorithm Comparison

| Algorithm | Use Case | Time Complexity | Space | Negative Weights | All-Pairs |
|-----------|----------|-----------------|-------|------------------|-----------|
| BFS | Unweighted graphs | O(V + E) | O(V) | N/A | No |
| Dijkstra | Non-negative weights | O((V+E) log V) | O(V) | No | No |
| Bellman-Ford | Negative weights, detect cycles | O(V × E) | O(V) | Yes | No |
| Floyd-Warshall | All pairs, dense graphs | O(V³) | O(V²) | Yes | Yes |
| A* | With good heuristic | O(b^d)* | O(b^d) | Depends | No |

*Performance depends heavily on heuristic quality

---

## Minimum Spanning Tree

A Minimum Spanning Tree (MST) is a subset of edges that connects all vertices in an undirected weighted graph with minimum total weight, without forming cycles.

### Properties of MST

1. **Connects all vertices**: Every vertex is reachable from every other vertex
2. **No cycles**: Exactly V-1 edges for V vertices
3. **Minimum total weight**: Sum of edge weights is minimized
4. **Not necessarily unique**: Multiple MSTs may exist with same total weight

### Applications of MST

- Network design (minimize cable length)
- Circuit design (minimize wire length)
- Clustering algorithms
- Approximation algorithms for NP-hard problems (e.g., Traveling Salesman)
- Image segmentation

### Kruskal's Algorithm

Kruskal's algorithm builds the MST by selecting edges in order of increasing weight, using Union-Find to detect cycles.

#### How Kruskal's Works

1. Sort all edges by weight (ascending)
2. Initialize empty MST
3. For each edge in sorted order:
   - If adding edge doesn't create cycle, add it to MST
   - Otherwise, skip it
4. Stop when MST has V-1 edges

#### Union-Find Data Structure

```python
class UnionFind:
    """
    Disjoint Set Union (DSU) with path compression and union by rank.
    Time: O(α(n)) per operation, where α is inverse Ackermann (practically constant)
    """
    def __init__(self, n):
        self.parent = list(range(n))  # Each node is its own parent initially
        self.rank = [0] * n  # Height of tree
        self.components = n  # Number of connected components

    def find(self, x):
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Unite sets containing x and y.
        Returns: True if united (were in different sets), False otherwise
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set (would create cycle)

        # Union by rank: attach smaller tree under larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True

    def is_connected(self, x, y):
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)
```

#### Kruskal's Implementation

**Python:**
```python
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

    def __repr__(self):
        return f"Edge({self.u}, {self.v}, {self.weight})"

def kruskal_mst(num_vertices, edges):
    """
    Kruskal's algorithm for Minimum Spanning Tree.

    Args:
        num_vertices: Number of vertices (0 to num_vertices-1)
        edges: List of Edge objects

    Returns:
        (mst_edges, total_weight)

    Time: O(E log E) for sorting + O(E α(V)) for union-find ≈ O(E log E)
    Space: O(V) for union-find + O(E) for sorted edges
    """
    # Sort edges by weight
    sorted_edges = sorted(edges)

    # Initialize Union-Find
    uf = UnionFind(num_vertices)

    mst = []
    total_weight = 0

    for edge in sorted_edges:
        # If edge connects two different components, add it
        if uf.union(edge.u, edge.v):
            mst.append(edge)
            total_weight += edge.weight

            # MST complete when we have V-1 edges
            if len(mst) == num_vertices - 1:
                break

    return mst, total_weight

# Example usage
edges = [
    Edge(0, 1, 4),
    Edge(0, 2, 3),
    Edge(1, 2, 1),
    Edge(1, 3, 2),
    Edge(2, 3, 4),
    Edge(3, 4, 2),
    Edge(4, 5, 6)
]

mst, weight = kruskal_mst(6, edges)
print(f"MST total weight: {weight}")
print("MST edges:")
for edge in mst:
    print(f"  {edge.u} -- {edge.v} (weight: {edge.weight})")
```

**JavaScript:**
```javascript
class UnionFind {
    constructor(n) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = Array(n).fill(0);
    }

    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    union(x, y) {
        const rootX = this.find(x);
        const rootY = this.find(y);

        if (rootX === rootY) return false;

        if (this.rank[rootX] < this.rank[rootY]) {
            this.parent[rootX] = rootY;
        } else if (this.rank[rootX] > this.rank[rootY]) {
            this.parent[rootY] = rootX;
        } else {
            this.parent[rootY] = rootX;
            this.rank[rootX]++;
        }

        return true;
    }
}

function kruskalMST(numVertices, edges) {
    // Sort edges by weight
    edges.sort((a, b) => a.weight - b.weight);

    const uf = new UnionFind(numVertices);
    const mst = [];
    let totalWeight = 0;

    for (const edge of edges) {
        if (uf.union(edge.u, edge.v)) {
            mst.push(edge);
            totalWeight += edge.weight;

            if (mst.length === numVertices - 1) {
                break;
            }
        }
    }

    return { mst, totalWeight };
}

// Example
const edges = [
    { u: 0, v: 1, weight: 4 },
    { u: 0, v: 2, weight: 3 },
    { u: 1, v: 2, weight: 1 },
    { u: 1, v: 3, weight: 2 },
    { u: 2, v: 3, weight: 4 },
    { u: 3, v: 4, weight: 2 },
    { u: 4, v: 5, weight: 6 }
];

const { mst, totalWeight } = kruskalMST(6, edges);
console.log(`MST total weight: ${totalWeight}`);
console.log('MST edges:', mst);
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class UnionFind {
private:
    vector<int> parent, rank;

public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) return false;

        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }

        return true;
    }
};

struct Edge {
    int u, v, weight;

    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

pair<vector<Edge>, int> kruskalMST(int numVertices, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());

    UnionFind uf(numVertices);
    vector<Edge> mst;
    int totalWeight = 0;

    for (const Edge& edge : edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst.push_back(edge);
            totalWeight += edge.weight;

            if (mst.size() == numVertices - 1) {
                break;
            }
        }
    }

    return {mst, totalWeight};
}

int main() {
    vector<Edge> edges = {
        {0, 1, 4}, {0, 2, 3}, {1, 2, 1},
        {1, 3, 2}, {2, 3, 4}, {3, 4, 2}, {4, 5, 6}
    };

    auto [mst, totalWeight] = kruskalMST(6, edges);

    cout << "MST total weight: " << totalWeight << endl;
    cout << "MST edges:" << endl;
    for (const Edge& edge : mst) {
        cout << "  " << edge.u << " -- " << edge.v
             << " (weight: " << edge.weight << ")" << endl;
    }

    return 0;
}
```

**Complexity:**
- **Time**: O(E log E) dominated by sorting edges
- **Space**: O(V) for union-find structure

**When to Use Kruskal's:**
- Sparse graphs (few edges)
- When edges are already sorted or can be efficiently sorted
- When you want to process edges by weight order
- Parallel/distributed implementations

---

### Prim's Algorithm

Prim's algorithm builds the MST by growing it from a starting vertex, always adding the minimum-weight edge that connects a vertex in the MST to a vertex outside.

#### How Prim's Works

1. Start with arbitrary vertex in MST
2. While MST doesn't include all vertices:
   - Find minimum-weight edge connecting MST to non-MST vertex
   - Add that edge and vertex to MST
3. Return MST

#### Prim's Implementation

**Python (with Priority Queue):**
```python
import heapq
from collections import defaultdict

class PrimMST:
    def __init__(self):
        self.graph = defaultdict(list)  # vertex -> [(neighbor, weight)]
        self.vertices = set()

    def add_edge(self, u, v, weight):
        """Add undirected weighted edge."""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))

    def prim_mst(self, start=None):
        """
        Prim's algorithm for MST.

        Time: O((V + E) log V) with binary heap
        Space: O(V + E)
        """
        if not self.vertices:
            return [], 0

        if start is None:
            start = next(iter(self.vertices))

        mst_edges = []
        total_weight = 0
        visited = {start}

        # Priority queue: (weight, from_vertex, to_vertex)
        edges_pq = [(weight, start, neighbor)
                    for neighbor, weight in self.graph[start]]
        heapq.heapify(edges_pq)

        while edges_pq and len(visited) < len(self.vertices):
            weight, u, v = heapq.heappop(edges_pq)

            # Skip if vertex already in MST
            if v in visited:
                continue

            # Add edge to MST
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight

            # Add all edges from newly added vertex
            for neighbor, edge_weight in self.graph[v]:
                if neighbor not in visited:
                    heapq.heappush(edges_pq, (edge_weight, v, neighbor))

        return mst_edges, total_weight

# Example usage
g = PrimMST()
g.add_edge('A', 'B', 4)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'C', 1)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 4)
g.add_edge('D', 'E', 2)
g.add_edge('E', 'F', 6)

mst, weight = g.prim_mst('A')
print(f"MST total weight: {weight}")
print("MST edges:")
for u, v, w in mst:
    print(f"  {u} -- {v} (weight: {w})")
```

**JavaScript:**
```javascript
class PriorityQueue {
    constructor() {
        this.values = [];
    }

    enqueue(val, priority) {
        this.values.push({ val, priority });
        this.sort();
    }

    dequeue() {
        return this.values.shift();
    }

    sort() {
        this.values.sort((a, b) => a.priority - b.priority);
    }

    isEmpty() {
        return this.values.length === 0;
    }
}

class PrimMST {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v, weight) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push({ node: v, weight });
        this.adjacencyList.get(v).push({ node: u, weight });
    }

    primMST(start) {
        if (this.adjacencyList.size === 0) {
            return { mst: [], totalWeight: 0 };
        }

        if (!start) {
            start = this.adjacencyList.keys().next().value;
        }

        const mst = [];
        let totalWeight = 0;
        const visited = new Set([start]);
        const pq = new PriorityQueue();

        // Add all edges from start vertex
        for (const { node, weight } of this.adjacencyList.get(start)) {
            pq.enqueue({ from: start, to: node, weight }, weight);
        }

        while (!pq.isEmpty() && visited.size < this.adjacencyList.size) {
            const { val: edge } = pq.dequeue();

            if (visited.has(edge.to)) continue;

            visited.add(edge.to);
            mst.push(edge);
            totalWeight += edge.weight;

            // Add edges from newly added vertex
            for (const { node, weight } of this.adjacencyList.get(edge.to)) {
                if (!visited.has(node)) {
                    pq.enqueue({ from: edge.to, to: node, weight }, weight);
                }
            }
        }

        return { mst, totalWeight };
    }
}

// Example
const g = new PrimMST();
g.addEdge('A', 'B', 4);
g.addEdge('A', 'C', 3);
g.addEdge('B', 'C', 1);
g.addEdge('B', 'D', 2);
g.addEdge('C', 'D', 4);
g.addEdge('D', 'E', 2);
g.addEdge('E', 'F', 6);

const { mst, totalWeight } = g.primMST('A');
console.log(`MST total weight: ${totalWeight}`);
console.log('MST edges:', mst);
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>

using namespace std;

struct Edge {
    string from, to;
    int weight;

    bool operator>(const Edge& other) const {
        return weight > other.weight;
    }
};

class PrimMST {
private:
    unordered_map<string, vector<pair<string, int>>> adjacencyList;

public:
    void addEdge(const string& u, const string& v, int weight) {
        adjacencyList[u].push_back({v, weight});
        adjacencyList[v].push_back({u, weight});
    }

    pair<vector<Edge>, int> primMST(const string& start) {
        vector<Edge> mst;
        int totalWeight = 0;
        unordered_set<string> visited;
        visited.insert(start);

        priority_queue<Edge, vector<Edge>, greater<Edge>> pq;

        // Add all edges from start
        for (const auto& [neighbor, weight] : adjacencyList[start]) {
            pq.push({start, neighbor, weight});
        }

        while (!pq.empty() && visited.size() < adjacencyList.size()) {
            Edge edge = pq.top();
            pq.pop();

            if (visited.count(edge.to)) continue;

            visited.insert(edge.to);
            mst.push_back(edge);
            totalWeight += edge.weight;

            // Add edges from newly added vertex
            for (const auto& [neighbor, weight] : adjacencyList[edge.to]) {
                if (!visited.count(neighbor)) {
                    pq.push({edge.to, neighbor, weight});
                }
            }
        }

        return {mst, totalWeight};
    }
};

int main() {
    PrimMST g;
    g.addEdge("A", "B", 4);
    g.addEdge("A", "C", 3);
    g.addEdge("B", "C", 1);
    g.addEdge("B", "D", 2);
    g.addEdge("C", "D", 4);
    g.addEdge("D", "E", 2);
    g.addEdge("E", "F", 6);

    auto [mst, totalWeight] = g.primMST("A");

    cout << "MST total weight: " << totalWeight << endl;
    cout << "MST edges:" << endl;
    for (const Edge& edge : mst) {
        cout << "  " << edge.from << " -- " << edge.to
             << " (weight: " << edge.weight << ")" << endl;
    }

    return 0;
}
```

**Complexity:**
- **Time**: O((V + E) log V) with binary heap
  - O(E + V log V) with Fibonacci heap (theoretical)
- **Space**: O(V + E)

**When to Use Prim's:**
- Dense graphs (many edges)
- When you want to grow MST from specific starting point
- Better for adjacency list representation

---

### Kruskal's vs Prim's

| Aspect | Kruskal's | Prim's |
|--------|-----------|--------|
| Approach | Edge-based (global) | Vertex-based (local growth) |
| Data Structure | Union-Find | Priority Queue |
| Best For | Sparse graphs | Dense graphs |
| Time Complexity | O(E log E) | O((V+E) log V) |
| Space | O(V) | O(V + E) |
| Parallelizable | Yes (easily) | Harder |
| Starting Point | N/A | Requires start vertex |

**When E is close to V² (dense):**
- Kruskal's: O(V² log V²) = O(V² log V)
- Prim's: O(V² log V)
- **Similar performance**, slight edge to Prim's

**When E is close to V (sparse):**
- Kruskal's: O(V log V)
- Prim's: O(V log V)
- **Kruskal's slightly simpler**

---

## Advanced Graph Algorithms

### Topological Sort

Topological sorting is a linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge (u, v), vertex u comes before v in the ordering.

#### Applications

- **Task Scheduling**: Order tasks respecting dependencies
- **Build Systems**: Compile files in correct order
- **Course Prerequisites**: Determine valid course sequence
- **Makefile dependency resolution**
- **Spreadsheet formula evaluation**

#### Properties

1. Only possible for DAGs (Directed Acyclic Graphs)
2. Not unique - multiple valid orderings may exist
3. If graph has cycle, no topological sort exists

#### Topological Sort using DFS

**Python:**
```python
from collections import defaultdict, deque

class DirectedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v):
        """Add directed edge from u to v."""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)

    def topological_sort_dfs(self):
        """
        Topological sort using DFS.
        Time: O(V + E)
        Space: O(V)
        Returns: List of vertices in topological order, or None if cycle exists
        """
        visited = set()
        rec_stack = set()  # Track vertices in current recursion stack
        result = []

        def dfs(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)

            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False  # Cycle detected
                elif neighbor in rec_stack:
                    return False  # Back edge found - cycle!

            rec_stack.remove(vertex)
            result.append(vertex)  # Add to result after all descendants
            return True

        # Process all vertices (handles disconnected components)
        for vertex in self.vertices:
            if vertex not in visited:
                if not dfs(vertex):
                    return None  # Cycle detected

        return result[::-1]  # Reverse to get correct order

# Example: Course prerequisites
g = DirectedGraph()
# Edges represent: prerequisite -> course
g.add_edge("Data Structures", "Algorithms")
g.add_edge("Algorithms", "Advanced Algorithms")
g.add_edge("Discrete Math", "Algorithms")
g.add_edge("Intro to CS", "Data Structures")
g.add_edge("Intro to CS", "Discrete Math")

order = g.topological_sort_dfs()
if order:
    print("Valid course order:")
    for i, course in enumerate(order, 1):
        print(f"  {i}. {course}")
else:
    print("Cannot complete courses - circular dependency!")
```

#### Topological Sort using Kahn's Algorithm (BFS)

```python
def topological_sort_kahns(self):
    """
    Kahn's algorithm for topological sort using BFS.
    Time: O(V + E)
    Space: O(V)
    Returns: List of vertices in topological order, or None if cycle exists
    """
    # Calculate in-degrees
    in_degree = {v: 0 for v in self.vertices}
    for u in self.graph:
        for v in self.graph[u]:
            in_degree[v] += 1

    # Queue of vertices with no incoming edges
    queue = deque([v for v in self.vertices if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        # Reduce in-degree for neighbors
        for neighbor in self.graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If all vertices are in result, we have valid topological sort
    if len(result) == len(self.vertices):
        return result
    else:
        return None  # Cycle exists

# Example: Build system
g = DirectedGraph()
g.add_edge("main.cpp", "main.o")
g.add_edge("utils.cpp", "utils.o")
g.add_edge("main.o", "program")
g.add_edge("utils.o", "program")

order = g.topological_sort_kahns()
print("Build order:", order)
```

**JavaScript:**
```javascript
class DirectedGraph {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push(v);
    }

    topologicalSortDFS() {
        const visited = new Set();
        const recStack = new Set();
        const result = [];

        const dfs = (vertex) => {
            visited.add(vertex);
            recStack.add(vertex);

            for (const neighbor of this.adjacencyList.get(vertex)) {
                if (!visited.has(neighbor)) {
                    if (!dfs(neighbor)) return false;
                } else if (recStack.has(neighbor)) {
                    return false;  // Cycle detected
                }
            }

            recStack.delete(vertex);
            result.push(vertex);
            return true;
        };

        for (const vertex of this.adjacencyList.keys()) {
            if (!visited.has(vertex)) {
                if (!dfs(vertex)) return null;
            }
        }

        return result.reverse();
    }

    topologicalSortKahns() {
        const inDegree = new Map();

        // Initialize in-degrees
        for (const vertex of this.adjacencyList.keys()) {
            inDegree.set(vertex, 0);
        }

        // Calculate in-degrees
        for (const [vertex, neighbors] of this.adjacencyList) {
            for (const neighbor of neighbors) {
                inDegree.set(neighbor, inDegree.get(neighbor) + 1);
            }
        }

        // Find vertices with no incoming edges
        const queue = [];
        for (const [vertex, degree] of inDegree) {
            if (degree === 0) {
                queue.push(vertex);
            }
        }

        const result = [];

        while (queue.length > 0) {
            const vertex = queue.shift();
            result.push(vertex);

            for (const neighbor of this.adjacencyList.get(vertex)) {
                inDegree.set(neighbor, inDegree.get(neighbor) - 1);
                if (inDegree.get(neighbor) === 0) {
                    queue.push(neighbor);
                }
            }
        }

        return result.length === this.adjacencyList.size ? result : null;
    }
}
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <string>
#include <algorithm>

using namespace std;

class DirectedGraph {
private:
    unordered_map<string, vector<string>> adjacencyList;
    unordered_set<string> vertices;

public:
    void addEdge(const string& u, const string& v) {
        vertices.insert(u);
        vertices.insert(v);
        adjacencyList[u].push_back(v);
        if (adjacencyList.find(v) == adjacencyList.end()) {
            adjacencyList[v] = {};
        }
    }

    vector<string> topologicalSortDFS() {
        unordered_set<string> visited, recStack;
        vector<string> result;

        function<bool(const string&)> dfs = [&](const string& vertex) {
            visited.insert(vertex);
            recStack.insert(vertex);

            for (const string& neighbor : adjacencyList[vertex]) {
                if (visited.find(neighbor) == visited.end()) {
                    if (!dfs(neighbor)) return false;
                } else if (recStack.find(neighbor) != recStack.end()) {
                    return false;  // Cycle detected
                }
            }

            recStack.erase(vertex);
            result.push_back(vertex);
            return true;
        };

        for (const string& vertex : vertices) {
            if (visited.find(vertex) == visited.end()) {
                if (!dfs(vertex)) return {};  // Empty vector indicates cycle
            }
        }

        reverse(result.begin(), result.end());
        return result;
    }

    vector<string> topologicalSortKahns() {
        unordered_map<string, int> inDegree;

        // Initialize in-degrees
        for (const string& vertex : vertices) {
            inDegree[vertex] = 0;
        }

        // Calculate in-degrees
        for (const auto& [vertex, neighbors] : adjacencyList) {
            for (const string& neighbor : neighbors) {
                inDegree[neighbor]++;
            }
        }

        // Queue vertices with no incoming edges
        queue<string> q;
        for (const auto& [vertex, degree] : inDegree) {
            if (degree == 0) {
                q.push(vertex);
            }
        }

        vector<string> result;

        while (!q.empty()) {
            string vertex = q.front();
            q.pop();
            result.push_back(vertex);

            for (const string& neighbor : adjacencyList[vertex]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }

        return result.size() == vertices.size() ? result : vector<string>{};
    }
};
```

#### All Topological Orderings

```python
def all_topological_sorts(self):
    """
    Find all possible topological orderings.
    Time: O(V! × E) in worst case
    Space: O(V)
    """
    in_degree = {v: 0 for v in self.vertices}
    for u in self.graph:
        for v in self.graph[u]:
            in_degree[v] += 1

    result = []
    current_order = []
    visited = set()

    def backtrack():
        if len(current_order) == len(self.vertices):
            result.append(current_order[:])
            return

        for vertex in self.vertices:
            if vertex not in visited and in_degree[vertex] == 0:
                # Include this vertex
                visited.add(vertex)
                current_order.append(vertex)

                # Reduce in-degree of neighbors
                for neighbor in self.graph[vertex]:
                    in_degree[neighbor] -= 1

                backtrack()

                # Backtrack
                for neighbor in self.graph[vertex]:
                    in_degree[neighbor] += 1
                current_order.pop()
                visited.remove(vertex)

    backtrack()
    return result
```

**Complexity:**
- **Time**: O(V + E) for single topological sort
- **Space**: O(V) for recursion stack / queue

**DFS vs Kahn's Algorithm:**

| Aspect | DFS | Kahn's (BFS) |
|--------|-----|--------------|
| Approach | Recursive | Iterative |
| Easy to implement | Yes | Yes |
| Detects cycles | Yes | Yes |
| Natural for | Deep graphs | Level-order |
| Space (recursion) | O(V) | O(V) |

---

### Strongly Connected Components

A Strongly Connected Component (SCC) is a maximal subset of vertices where every vertex is reachable from every other vertex in the subset.

#### Applications

- **Social network analysis**: Find tightly-knit communities
- **Web crawling**: Identify clusters of mutually linked pages
- **Circuit analysis**: Find feedback loops
- **Compiler optimization**: Detect variable dependencies
- **Recommendation systems**: Find groups with similar preferences

#### Kosaraju's Algorithm

Kosaraju's algorithm finds all SCCs in two DFS passes.

**Algorithm:**
1. Perform DFS on original graph, record finish times
2. Create transpose graph (reverse all edges)
3. Perform DFS on transpose in decreasing finish time order
4. Each DFS tree in step 3 is one SCC

**Python:**
```python
class DirectedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v):
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)

    def kosaraju_scc(self):
        """
        Kosaraju's algorithm for finding Strongly Connected Components.
        Time: O(V + E)
        Space: O(V)
        Returns: List of SCCs (each SCC is a list of vertices)
        """
        # Step 1: DFS to compute finish times
        visited = set()
        finish_order = []

        def dfs1(vertex):
            visited.add(vertex)
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs1(neighbor)
            finish_order.append(vertex)  # Add after all descendants

        for vertex in self.vertices:
            if vertex not in visited:
                dfs1(vertex)

        # Step 2: Create transpose graph
        transpose = defaultdict(list)
        for u in self.graph:
            for v in self.graph[u]:
                transpose[v].append(u)  # Reverse edge

        # Step 3: DFS on transpose in reverse finish order
        visited.clear()
        sccs = []

        def dfs2(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            for neighbor in transpose[vertex]:
                if neighbor not in visited:
                    dfs2(neighbor, component)

        for vertex in reversed(finish_order):
            if vertex not in visited:
                component = []
                dfs2(vertex, component)
                sccs.append(component)

        return sccs

# Example: Social network
g = DirectedGraph()
# Mutual follows create SCCs
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'A')  # SCC: {A, B, C}
g.add_edge('B', 'D')
g.add_edge('D', 'E')
g.add_edge('E', 'D')  # SCC: {D, E}
g.add_edge('E', 'F')  # SCC: {F}

sccs = g.kosaraju_scc()
print(f"Found {len(sccs)} strongly connected components:")
for i, scc in enumerate(sccs, 1):
    print(f"  SCC {i}: {sorted(scc)}")
```

#### Tarjan's Algorithm

Tarjan's algorithm finds SCCs in a single DFS pass using low-link values.

**Key Concepts:**
- **disc[v]**: Discovery time of vertex v
- **low[v]**: Lowest discovery time reachable from v
- **Stack**: Maintains current path
- When low[v] == disc[v], v is root of SCC

**Python:**
```python
def tarjan_scc(self):
    """
    Tarjan's algorithm for finding Strongly Connected Components.
    Time: O(V + E)
    Space: O(V)
    Returns: List of SCCs
    """
    disc = {}  # Discovery times
    low = {}   # Lowest reachable
    on_stack = set()
    stack = []
    sccs = []
    time = [0]  # Mutable counter

    def dfs(vertex):
        disc[vertex] = low[vertex] = time[0]
        time[0] += 1
        stack.append(vertex)
        on_stack.add(vertex)

        # Explore neighbors
        for neighbor in self.graph[vertex]:
            if neighbor not in disc:
                # Neighbor not visited
                dfs(neighbor)
                low[vertex] = min(low[vertex], low[neighbor])
            elif neighbor in on_stack:
                # Back edge to ancestor
                low[vertex] = min(low[vertex], disc[neighbor])

        # If vertex is root of SCC
        if low[vertex] == disc[vertex]:
            scc = []
            while True:
                node = stack.pop()
                on_stack.remove(node)
                scc.append(node)
                if node == vertex:
                    break
            sccs.append(scc)

    for vertex in self.vertices:
        if vertex not in disc:
            dfs(vertex)

    return sccs

# Usage (same graph as above)
sccs = g.tarjan_scc()
print(f"Found {len(sccs)} strongly connected components:")
for i, scc in enumerate(sccs, 1):
    print(f"  SCC {i}: {sorted(scc)}")
```

**JavaScript:**
```javascript
class DirectedGraph {
    constructor() {
        this.adjacencyList = new Map();
    }

    addVertex(vertex) {
        if (!this.adjacencyList.has(vertex)) {
            this.adjacencyList.set(vertex, []);
        }
    }

    addEdge(u, v) {
        this.addVertex(u);
        this.addVertex(v);
        this.adjacencyList.get(u).push(v);
    }

    tarjanSCC() {
        const disc = new Map();
        const low = new Map();
        const onStack = new Set();
        const stack = [];
        const sccs = [];
        let time = 0;

        const dfs = (vertex) => {
            disc.set(vertex, time);
            low.set(vertex, time);
            time++;
            stack.push(vertex);
            onStack.add(vertex);

            for (const neighbor of this.adjacencyList.get(vertex)) {
                if (!disc.has(neighbor)) {
                    dfs(neighbor);
                    low.set(vertex, Math.min(low.get(vertex), low.get(neighbor)));
                } else if (onStack.has(neighbor)) {
                    low.set(vertex, Math.min(low.get(vertex), disc.get(neighbor)));
                }
            }

            // Root of SCC
            if (low.get(vertex) === disc.get(vertex)) {
                const scc = [];
                let node;
                do {
                    node = stack.pop();
                    onStack.delete(node);
                    scc.push(node);
                } while (node !== vertex);
                sccs.push(scc);
            }
        };

        for (const vertex of this.adjacencyList.keys()) {
            if (!disc.has(vertex)) {
                dfs(vertex);
            }
        }

        return sccs;
    }
}
```

**Complexity:**
- **Time**: O(V + E) for both algorithms
- **Space**: O(V)

**Kosaraju's vs Tarjan's:**

| Aspect | Kosaraju's | Tarjan's |
|--------|------------|----------|
| DFS passes | 2 | 1 |
| Easier to understand | Yes | No |
| Memory (stack) | Two DFS stacks | One DFS stack + explicit stack |
| Practical performance | Slightly slower | Slightly faster |
| Implementation | Simpler | More complex |

---

### Articulation Points and Bridges

**Articulation Point (Cut Vertex)**: A vertex whose removal increases the number of connected components.

**Bridge (Cut Edge)**: An edge whose removal increases the number of connected components.

#### Applications

- **Network reliability**: Identify critical routers/links
- **Social networks**: Find influential connectors
- **Circuit design**: Identify single points of failure
- **Transportation networks**: Critical roads/bridges

#### Finding Articulation Points

**Python (using DFS and low-link values):**
```python
class UndirectedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()

    def add_edge(self, u, v):
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)
        self.graph[v].append(u)

    def find_articulation_points(self):
        """
        Find all articulation points (cut vertices).
        Time: O(V + E)
        Space: O(V)
        """
        disc = {}  # Discovery times
        low = {}   # Lowest reachable
        parent = {}
        articulation_points = set()
        time = [0]

        def dfs(u):
            children = 0
            disc[u] = low[u] = time[0]
            time[0] += 1

            for v in self.graph[u]:
                if v not in disc:
                    children += 1
                    parent[v] = u
                    dfs(v)

                    low[u] = min(low[u], low[v])

                    # u is articulation point if:
                    # 1. u is root and has >= 2 children, OR
                    # 2. u is not root and low[v] >= disc[u]
                    if parent.get(u) is None and children > 1:
                        articulation_points.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        articulation_points.add(u)

                elif v != parent.get(u):
                    # Back edge
                    low[u] = min(low[u], disc[v])

        # Handle disconnected components
        for vertex in self.vertices:
            if vertex not in disc:
                parent[vertex] = None
                dfs(vertex)

        return list(articulation_points)

# Example: Network topology
g = UndirectedGraph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)  # Triangle
g.add_edge(1, 3)  # 1 is articulation point
g.add_edge(3, 4)
g.add_edge(4, 5)
g.add_edge(5, 3)  # Another triangle

aps = g.find_articulation_points()
print(f"Articulation points: {sorted(aps)}")
# Output: [1, 3]
```

#### Finding Bridges

**Python:**
```python
def find_bridges(self):
    """
    Find all bridges (cut edges).
    Time: O(V + E)
    Space: O(V)
    """
    disc = {}
    low = {}
    parent = {}
    bridges = []
    time = [0]

    def dfs(u):
        disc[u] = low[u] = time[0]
        time[0] += 1

        for v in self.graph[u]:
            if v not in disc:
                parent[v] = u
                dfs(v)

                low[u] = min(low[u], low[v])

                # Bridge condition: low[v] > disc[u]
                # (no back edge from subtree of v to ancestors of u)
                if low[v] > disc[u]:
                    bridges.append((u, v))

            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for vertex in self.vertices:
        if vertex not in disc:
            parent[vertex] = None
            dfs(vertex)

    return bridges

# Using same graph as above
bridges = g.find_bridges()
print(f"Bridges: {bridges}")
# Output: [(1, 3)]
```

**C++:**
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace std;

class UndirectedGraph {
private:
    unordered_map<int, vector<int>> adjacencyList;
    unordered_set<int> vertices;

public:
    void addEdge(int u, int v) {
        vertices.insert(u);
        vertices.insert(v);
        adjacencyList[u].push_back(v);
        adjacencyList[v].push_back(u);
    }

    vector<int> findArticulationPoints() {
        unordered_map<int, int> disc, low, parent;
        unordered_set<int> articulationPoints;
        int time = 0;

        function<void(int)> dfs = [&](int u) {
            int children = 0;
            disc[u] = low[u] = time++;

            for (int v : adjacencyList[u]) {
                if (disc.find(v) == disc.end()) {
                    children++;
                    parent[v] = u;
                    dfs(v);

                    low[u] = min(low[u], low[v]);

                    if (parent.find(u) == parent.end() && children > 1) {
                        articulationPoints.insert(u);
                    }
                    if (parent.find(u) != parent.end() && low[v] >= disc[u]) {
                        articulationPoints.insert(u);
                    }
                } else if (parent.find(u) == parent.end() || v != parent[u]) {
                    low[u] = min(low[u], disc[v]);
                }
            }
        };

        for (int vertex : vertices) {
            if (disc.find(vertex) == disc.end()) {
                dfs(vertex);
            }
        }

        return vector<int>(articulationPoints.begin(), articulationPoints.end());
    }

    vector<pair<int, int>> findBridges() {
        unordered_map<int, int> disc, low, parent;
        vector<pair<int, int>> bridges;
        int time = 0;

        function<void(int)> dfs = [&](int u) {
            disc[u] = low[u] = time++;

            for (int v : adjacencyList[u]) {
                if (disc.find(v) == disc.end()) {
                    parent[v] = u;
                    dfs(v);

                    low[u] = min(low[u], low[v]);

                    if (low[v] > disc[u]) {
                        bridges.push_back({min(u, v), max(u, v)});
                    }
                } else if (parent.find(u) == parent.end() || v != parent[u]) {
                    low[u] = min(low[u], disc[v]);
                }
            }
        };

        for (int vertex : vertices) {
            if (disc.find(vertex) == disc.end()) {
                dfs(vertex);
            }
        }

        return bridges;
    }
};
```

**Key Insights:**

1. **Articulation Point Condition**:
   - Root: Has ≥ 2 children in DFS tree
   - Non-root: Has child v where low[v] ≥ disc[u]

2. **Bridge Condition**:
   - Edge (u, v) is bridge if low[v] > disc[u]
   - No back edge from v's subtree to u's ancestors

**Complexity:**
- **Time**: O(V + E)
- **Space**: O(V)

---

### Network Flow

Network flow algorithms solve problems involving flow through a network with capacity constraints. The canonical problem is the Maximum Flow Problem.

#### Maximum Flow Problem

Given:
- Directed graph with capacity on each edge
- Source vertex s
- Sink vertex t

Find: Maximum amount of flow from s to t

#### Applications

- **Transportation**: Maximum throughput in road/rail networks
- **Network routing**: Internet packet routing
- **Bipartite matching**: Job assignment, dating apps
- **Image segmentation**: Computer vision
- **Airline scheduling**: Flight capacity optimization
- **Project selection**: Maximize profit with budget constraints

#### Ford-Fulkerson Method

Ford-Fulkerson is a method (not a specific algorithm) based on augmenting paths.

**Key Concepts:**
- **Residual Graph**: Shows remaining capacity
- **Augmenting Path**: Path from s to t in residual graph
- **Bottleneck Capacity**: Minimum capacity on augmenting path

**Algorithm:**
1. Start with zero flow
2. While there exists augmenting path from s to t:
   - Find bottleneck capacity
   - Augment flow along path
   - Update residual graph
3. Return maximum flow

**Python (using BFS to find augmenting paths - Edmonds-Karp):**
```python
from collections import deque, defaultdict

class MaxFlow:
    def __init__(self, vertices):
        self.V = vertices
        # Capacity matrix: capacity[u][v] = capacity of edge u -> v
        self.capacity = [[0] * vertices for _ in range(vertices)]
        self.graph = defaultdict(list)  # Adjacency list for faster iteration

    def add_edge(self, u, v, capacity):
        """Add directed edge with capacity."""
        self.capacity[u][v] = capacity
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add reverse edge for residual graph

    def bfs_find_path(self, source, sink, parent):
        """
        Find augmenting path using BFS.
        Returns: True if path exists, False otherwise
        """
        visited = set([source])
        queue = deque([source])

        while queue:
            u = queue.popleft()

            for v in self.graph[u]:
                # Check if unvisited and has remaining capacity
                if v not in visited and self.capacity[u][v] > 0:
                    visited.add(v)
                    queue.append(v)
                    parent[v] = u

                    if v == sink:
                        return True

        return False

    def edmonds_karp(self, source, sink):
        """
        Edmonds-Karp algorithm (Ford-Fulkerson with BFS).
        Time: O(V × E²)
        Space: O(V²)
        Returns: (max_flow, flow_matrix)
        """
        parent = {}
        max_flow = 0

        # Flow matrix
        flow = [[0] * self.V for _ in range(self.V)]

        # While there exists augmenting path
        while self.bfs_find_path(source, sink, parent):
            # Find bottleneck capacity
            path_flow = float('inf')
            v = sink

            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.capacity[u][v])
                v = u

            # Update residual capacities and flow
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[u][v] -= path_flow
                self.capacity[v][u] += path_flow  # Add reverse edge
                flow[u][v] += path_flow
                v = u

            max_flow += path_flow
            parent.clear()

        return max_flow, flow

# Example: Network flow
#     s --10--> 1 --10--> t
#     |         |         |
#     5         5         10
#     |         |         |
#     v         v         ^
#     2 --10--> 3 --15----
#
g = MaxFlow(4)
s, t = 0, 3  # Source = 0, Sink = 3

g.add_edge(0, 1, 10)  # s -> 1
g.add_edge(0, 2, 5)   # s -> 2
g.add_edge(1, 2, 5)   # 1 -> 2
g.add_edge(1, 3, 10)  # 1 -> t
g.add_edge(2, 3, 15)  # 2 -> t

max_flow, flow_matrix = g.edmonds_karp(s, t)
print(f"Maximum flow: {max_flow}")

print("\nFlow on each edge:")
for u in range(g.V):
    for v in range(g.V):
        if flow_matrix[u][v] > 0:
            print(f"  {u} -> {v}: {flow_matrix[u][v]}")
```

**Output:**
```
Maximum flow: 15
Flow on each edge:
  0 -> 1: 10
  0 -> 2: 5
  1 -> 3: 10
  2 -> 3: 5
```

#### Finding Min-Cut

The Min-Cut equals Max-Flow by the Max-Flow Min-Cut theorem.

```python
def find_min_cut(self, source):
    """
    Find minimum cut (set of edges with minimum capacity that separates s from t).
    Returns: List of edges in min cut
    Time: O(V × E²) (running max flow first)
    """
    # After max flow, residual graph contains the cut
    reachable = set()
    queue = deque([source])
    reachable.add(source)

    # BFS on residual graph
    while queue:
        u = queue.popleft()
        for v in self.graph[u]:
            if v not in reachable and self.capacity[u][v] > 0:
                reachable.add(v)
                queue.append(v)

    # Min cut edges: from reachable to non-reachable
    min_cut = []
    for u in reachable:
        for v in range(self.V):
            if v not in reachable and (u, v) in self.original_edges:
                min_cut.append((u, v))

    return min_cut
```

**JavaScript:**
```javascript
class MaxFlow {
    constructor(vertices) {
        this.V = vertices;
        this.capacity = Array(vertices).fill(null)
            .map(() => Array(vertices).fill(0));
        this.graph = Array(vertices).fill(null)
            .map(() => []);
    }

    addEdge(u, v, capacity) {
        this.capacity[u][v] = capacity;
        this.graph[u].push(v);
        this.graph[v].push(u);
    }

    bfsFindPath(source, sink, parent) {
        const visited = new Set([source]);
        const queue = [source];

        while (queue.length > 0) {
            const u = queue.shift();

            for (const v of this.graph[u]) {
                if (!visited.has(v) && this.capacity[u][v] > 0) {
                    visited.add(v);
                    queue.push(v);
                    parent[v] = u;

                    if (v === sink) return true;
                }
            }
        }

        return false;
    }

    edmondsKarp(source, sink) {
        const parent = {};
        let maxFlow = 0;
        const flow = Array(this.V).fill(null)
            .map(() => Array(this.V).fill(0));

        while (this.bfsFindPath(source, sink, parent)) {
            let pathFlow = Infinity;
            let v = sink;

            // Find bottleneck
            while (v !== source) {
                const u = parent[v];
                pathFlow = Math.min(pathFlow, this.capacity[u][v]);
                v = u;
            }

            // Update capacities and flow
            v = sink;
            while (v !== source) {
                const u = parent[v];
                this.capacity[u][v] -= pathFlow;
                this.capacity[v][u] += pathFlow;
                flow[u][v] += pathFlow;
                v = u;
            }

            maxFlow += pathFlow;
            Object.keys(parent).forEach(key => delete parent[key]);
        }

        return { maxFlow, flow };
    }
}
```

#### Dinic's Algorithm (Faster Alternative)

Dinic's algorithm is faster for many cases, using level graphs and blocking flows.

```python
def dinic_max_flow(self, source, sink):
    """
    Dinic's algorithm for maximum flow.
    Time: O(V² × E) - faster than Edmonds-Karp for many graphs
    """
    def bfs_level_graph():
        """Build level graph using BFS."""
        level = [-1] * self.V
        level[source] = 0
        queue = deque([source])

        while queue:
            u = queue.popleft()
            for v in self.graph[u]:
                if level[v] < 0 and self.capacity[u][v] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)

        return level

    def dfs_blocking_flow(u, pushed, level, iter_ptr):
        """Find blocking flow using DFS."""
        if u == sink:
            return pushed

        while iter_ptr[u] < len(self.graph[u]):
            v = self.graph[u][iter_ptr[u]]

            if level[v] == level[u] + 1 and self.capacity[u][v] > 0:
                flow = dfs_blocking_flow(
                    v,
                    min(pushed, self.capacity[u][v]),
                    level,
                    iter_ptr
                )

                if flow > 0:
                    self.capacity[u][v] -= flow
                    self.capacity[v][u] += flow
                    return flow

            iter_ptr[u] += 1

        return 0

    max_flow = 0

    while True:
        level = bfs_level_graph()

        if level[sink] < 0:  # No augmenting path
            break

        iter_ptr = [0] * self.V

        while True:
            pushed = dfs_blocking_flow(source, float('inf'), level, iter_ptr)
            if pushed == 0:
                break
            max_flow += pushed

    return max_flow
```

**Complexity Comparison:**

| Algorithm | Time Complexity | Space | Notes |
|-----------|-----------------|-------|-------|
| Ford-Fulkerson (DFS) | O(E × max_flow) | O(V) | Pseudo-polynomial |
| Edmonds-Karp (BFS) | O(V × E²) | O(V) | Polynomial |
| Dinic's | O(V² × E) | O(V) | Faster in practice |
| Push-Relabel | O(V³) or O(V² × E) | O(V) | Good for dense graphs |

---

### Bipartite Matching

Bipartite matching finds a maximum matching in a bipartite graph - the largest set of edges with no shared vertices.

#### Applications

- **Job assignment**: Assign workers to tasks
- **Dating/Marriage**: Stable matching problem
- **Resource allocation**: Assign resources to requesters
- **Timetabling**: Assign classes to rooms/times
- **Network routing**: Assign packets to routes

#### Maximum Bipartite Matching using DFS

```python
class BipartiteGraph:
    def __init__(self, left_size, right_size):
        self.left_size = left_size
        self.right_size = right_size
        self.graph = defaultdict(list)  # left -> [right vertices]

    def add_edge(self, left, right):
        """Add edge from left partition to right partition."""
        self.graph[left].append(right)

    def max_matching_dfs(self):
        """
        Maximum bipartite matching using DFS (Augmenting Path).
        Time: O(V × E)
        Space: O(V)
        Returns: (matching_size, matching_dict)
        """
        # match_right[r] = left vertex matched to r (or -1 if unmatched)
        match_right = [-1] * self.right_size

        def dfs(left, visited):
            """Try to find augmenting path from left vertex."""
            for right in self.graph[left]:
                if visited[right]:
                    continue

                visited[right] = True

                # If right is unmatched OR we can find augmenting path from its match
                if match_right[right] == -1 or dfs(match_right[right], visited):
                    match_right[right] = left
                    return True

            return False

        matching_size = 0

        # Try to match each left vertex
        for left in range(self.left_size):
            visited = [False] * self.right_size
            if dfs(left, visited):
                matching_size += 1

        # Build matching dictionary
        matching = {}
        for right, left in enumerate(match_right):
            if left != -1:
                matching[left] = right

        return matching_size, matching

# Example: Job assignment
# Workers: 0, 1, 2 (left partition)
# Jobs: 0, 1, 2 (right partition)
g = BipartiteGraph(3, 3)
g.add_edge(0, 0)  # Worker 0 can do Job 0
g.add_edge(0, 1)  # Worker 0 can do Job 1
g.add_edge(1, 1)  # Worker 1 can do Job 1
g.add_edge(2, 0)  # Worker 2 can do Job 0
g.add_edge(2, 2)  # Worker 2 can do Job 2

size, matching = g.max_matching_dfs()
print(f"Maximum matching size: {size}")
print("Assignments:")
for worker, job in sorted(matching.items()):
    print(f"  Worker {worker} -> Job {job}")
```

#### Maximum Bipartite Matching using Network Flow

Bipartite matching reduces to max flow:
1. Add source s connecting to all left vertices (capacity 1)
2. Add sink t connecting from all right vertices (capacity 1)
3. All original edges have capacity 1
4. Max flow = Max matching

```python
def max_matching_flow(self):
    """
    Maximum bipartite matching using max flow.
    Time: O(V × E²)
    """
    # Create flow network
    # Vertices: source=0, left=1..left_size, right=left_size+1..left_size+right_size, sink=last
    source = 0
    sink = 1 + self.left_size + self.right_size

    flow_graph = MaxFlow(sink + 1)

    # Source to left partition
    for left in range(self.left_size):
        flow_graph.add_edge(source, 1 + left, 1)

    # Left to right edges
    for left in range(self.left_size):
        for right in self.graph[left]:
            flow_graph.add_edge(1 + left, 1 + self.left_size + right, 1)

    # Right partition to sink
    for right in range(self.right_size):
        flow_graph.add_edge(1 + self.left_size + right, sink, 1)

    max_flow, flow_matrix = flow_graph.edmonds_karp(source, sink)

    # Extract matching from flow
    matching = {}
    for left in range(self.left_size):
        for right in range(self.right_size):
            if flow_matrix[1 + left][1 + self.left_size + right] > 0:
                matching[left] = right

    return max_flow, matching
```

#### Hopcroft-Karp Algorithm (Fastest)

Hopcroft-Karp is the fastest algorithm for maximum bipartite matching.

```python
def hopcroft_karp(self):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    Time: O(E × sqrt(V))
    Space: O(V)
    """
    match_left = {}   # left -> right
    match_right = {}  # right -> left

    def bfs():
        """BFS to build level graph."""
        queue = deque()
        dist = {}

        for left in range(self.left_size):
            if left not in match_left:
                dist[left] = 0
                queue.append(left)

        dist[None] = float('inf')

        while queue:
            left = queue.popleft()
            if dist[left] < dist[None]:
                for right in self.graph[left]:
                    matched_left = match_right.get(right)
                    if matched_left not in dist:
                        dist[matched_left] = dist[left] + 1
                        queue.append(matched_left)

        return dist[None] != float('inf'), dist

    def dfs(left, dist):
        """DFS to find augmenting paths."""
        if left is None:
            return True

        for right in self.graph[left]:
            matched_left = match_right.get(right)
            if dist.get(matched_left, float('inf')) == dist[left] + 1:
                if dfs(matched_left, dist):
                    match_left[left] = right
                    match_right[right] = left
                    return True

        dist[left] = float('inf')
        return False

    matching_size = 0

    while True:
        found, dist = bfs()
        if not found:
            break

        for left in range(self.left_size):
            if left not in match_left and dfs(left, dist):
                matching_size += 1

    return matching_size, match_left
```

**Complexity Comparison:**

| Algorithm | Time Complexity | Best For |
|-----------|-----------------|----------|
| DFS Augmenting Path | O(V × E) | Simple implementation |
| Network Flow | O(V × E²) | When you have max flow code |
| Hopcroft-Karp | O(E × √V) | Large bipartite graphs |

---

## Algorithm Selection Guide

### Decision Tree for Shortest Path

```
Need shortest path?
├─ All pairs?
│  ├─ Yes → Floyd-Warshall O(V³)
│  └─ No → Continue below
├─ Single source?
│  ├─ Unweighted graph?
│  │  └─ Yes → BFS O(V + E)
│  ├─ Non-negative weights?
│  │  ├─ Yes → Have good heuristic?
│  │  │  ├─ Yes → A* O(b^d)
│  │  │  └─ No → Dijkstra O((V+E) log V)
│  │  └─ No → Bellman-Ford O(V × E)
│  └─ Need to detect negative cycle?
│     └─ Yes → Bellman-Ford O(V × E)
```

### Decision Tree for Graph Traversal

```
Need to traverse graph?
├─ Find shortest path in unweighted?
│  └─ Yes → BFS O(V + E)
├─ Detect cycles?
│  └─ Yes → DFS O(V + E)
├─ Topological sort?
│  └─ Yes → DFS or Kahn's O(V + E)
├─ Find connected components?
│  └─ Yes → DFS or BFS O(V + E)
├─ Level-order traversal?
│  └─ Yes → BFS O(V + E)
└─ Memory constrained?
   ├─ Deep graph → DFS (less memory)
   └─ Wide graph → BFS (less memory)
```

### Decision Tree for Minimum Spanning Tree

```
Need MST?
├─ Sparse graph (E << V²)?
│  └─ Yes → Kruskal's O(E log E)
├─ Dense graph (E ≈ V²)?
│  └─ Yes → Prim's O((V+E) log V)
├─ Edges already sorted?
│  └─ Yes → Kruskal's O(E α(V))
├─ Need to start from specific vertex?
│  └─ Yes → Prim's
└─ Distributed/Parallel?
   └─ Yes → Kruskal's (easier to parallelize)
```

### Decision Tree for Advanced Algorithms

```
Advanced graph problem?
├─ Need task ordering with dependencies?
│  └─ Yes → Topological Sort O(V + E)
├─ Find tightly connected groups?
│  └─ Yes → Strongly Connected Components O(V + E)
│  ├─ Simpler implementation → Kosaraju's
│  └─ Slightly faster → Tarjan's
├─ Find critical infrastructure?
│  ├─ Critical vertices → Articulation Points O(V + E)
│  └─ Critical edges → Bridges O(V + E)
├─ Maximum flow/minimum cut?
│  ├─ Small graphs → Edmonds-Karp O(V × E²)
│  └─ Larger graphs → Dinic's O(V² × E)
└─ Bipartite matching?
   ├─ Simple implementation → DFS Augmenting Path O(V × E)
   └─ Large graphs → Hopcroft-Karp O(E × √V)
```

---

## Real-World Applications

### 1. Google Maps / GPS Navigation

**Problem**: Find fastest route from A to B

**Algorithms Used**:
- **A* Search**: With geographic distance heuristic
- **Dijkstra's**: When no good heuristic available
- **Bidirectional search**: Search from both ends
- **Contraction Hierarchies**: Preprocess for faster queries

**Key Considerations**:
- Dynamic edge weights (traffic conditions)
- Multiple objectives (time, distance, tolls)
- Turn restrictions and one-way streets

**Example:**
```python
def gps_route(graph, start, end, current_traffic):
    """
    Find fastest route considering current traffic.
    """
    def heuristic(node):
        # Estimate time using straight-line distance and average speed
        distance = haversine_distance(node, end)
        avg_speed = 50  # km/h
        return distance / avg_speed

    def edge_weight(u, v):
        # Dynamic weight based on current traffic
        base_time = graph.base_travel_time(u, v)
        traffic_multiplier = current_traffic.get((u, v), 1.0)
        return base_time * traffic_multiplier

    return a_star_search(start, end, heuristic, edge_weight)
```

### 2. Social Network Analysis

**Problem**: Find communities, influencers, connections

**Algorithms Used**:
- **BFS**: Find degrees of separation ("6 degrees of Kevin Bacon")
- **Strongly Connected Components**: Find tight-knit communities
- **PageRank**: Identify influential users
- **Shortest Path**: Friend suggestions (mutual friends)

**Example:**
```python
def friend_suggestions(user_id, social_graph):
    """
    Suggest friends based on mutual connections.
    """
    # BFS to find friends and friends-of-friends
    friends = set(social_graph[user_id])
    suggestions = defaultdict(int)  # friend -> number of mutual connections

    for friend in friends:
        for friend_of_friend in social_graph[friend]:
            if friend_of_friend != user_id and friend_of_friend not in friends:
                suggestions[friend_of_friend] += 1

    # Sort by number of mutual friends
    return sorted(suggestions.items(), key=lambda x: x[1], reverse=True)

def degrees_of_separation(graph, user1, user2):
    """
    Find shortest connection path between two users.
    """
    return bfs_shortest_path(graph, user1, user2)
```

### 3. Compiler Optimization

**Problem**: Optimize code execution order

**Algorithms Used**:
- **Topological Sort**: Order of function calls
- **Strongly Connected Components**: Detect recursive loops
- **DFS**: Dependency analysis

**Example:**
```python
class DependencyGraph:
    def __init__(self):
        self.dependencies = defaultdict(list)  # module -> [dependencies]

    def add_dependency(self, module, depends_on):
        self.dependencies[module].append(depends_on)

    def build_order(self):
        """
        Determine order to compile modules.
        """
        return topological_sort(self.dependencies)

    def detect_circular_dependencies(self):
        """
        Find circular dependencies (compilation impossible).
        """
        sccs = strongly_connected_components(self.dependencies)
        circular = [scc for scc in sccs if len(scc) > 1]
        return circular
```

### 4. Network Routing (Internet)

**Problem**: Route packets efficiently

**Algorithms Used**:
- **Dijkstra's**: OSPF (Open Shortest Path First) protocol
- **Bellman-Ford**: RIP (Routing Information Protocol)
- **Minimum Spanning Tree**: Network topology design

**Example:**
```python
class NetworkRouter:
    def __init__(self):
        self.topology = {}  # router_id -> [(neighbor, latency)]
        self.routing_table = {}

    def update_routing_table(self, router_id):
        """
        Update routing table using Dijkstra's.
        """
        distances, next_hop = dijkstra_with_path(self.topology, router_id)
        self.routing_table[router_id] = next_hop

    def route_packet(self, source, destination):
        """
        Determine path for packet from source to destination.
        """
        path = []
        current = source

        while current != destination:
            if current not in self.routing_table:
                return None  # No route
            current = self.routing_table[current][destination]
            path.append(current)

        return path
```

### 5. Game Development (Pathfinding)

**Problem**: Move characters intelligently

**Algorithms Used**:
- **A***: Character movement in games
- **Dijkstra's**: When uniform cost
- **Hierarchical pathfinding**: Large maps

**Example:**
```python
class GamePathfinding:
    def __init__(self, game_map):
        self.map = game_map
        self.nav_mesh = self.build_nav_mesh()

    def find_path(self, start, goal, character_type):
        """
        Find path considering character capabilities.
        """
        def heuristic(pos):
            return euclidean_distance(pos, goal)

        def can_traverse(pos1, pos2):
            # Check if character can move from pos1 to pos2
            terrain = self.map.get_terrain(pos2)
            return character_type in terrain.traversable_by

        def cost(pos1, pos2):
            # Cost depends on terrain
            terrain = self.map.get_terrain(pos2)
            base_cost = euclidean_distance(pos1, pos2)
            return base_cost * terrain.difficulty[character_type]

        return a_star_search(start, goal, heuristic, can_traverse, cost)
```

### 6. Recommendation Systems

**Problem**: Recommend products/content to users

**Algorithms Used**:
- **Bipartite Matching**: User-item matching
- **Graph Clustering**: Find similar users/items
- **Random Walk**: Collaborative filtering

**Example:**
```python
class RecommendationSystem:
    def __init__(self):
        self.user_item_graph = BipartiteGraph()  # users <-> items
        self.item_similarity = {}  # item -> similar items

    def recommend_items(self, user_id, num_recommendations=5):
        """
        Recommend items based on similar users' preferences.
        """
        # BFS to find users with similar tastes
        similar_users = self.find_similar_users(user_id, depth=2)

        # Aggregate items liked by similar users
        recommendations = defaultdict(float)
        user_items = set(self.user_item_graph.get_items(user_id))

        for similar_user, similarity in similar_users:
            for item in self.user_item_graph.get_items(similar_user):
                if item not in user_items:
                    recommendations[item] += similarity

        # Return top recommendations
        return sorted(recommendations.items(),
                     key=lambda x: x[1],
                     reverse=True)[:num_recommendations]
```

### 7. Supply Chain Optimization

**Problem**: Minimize transportation costs

**Algorithms Used**:
- **Minimum Spanning Tree**: Network design
- **Shortest Path**: Route optimization
- **Network Flow**: Resource distribution

**Example:**
```python
class SupplyChain:
    def __init__(self):
        self.network = {}  # location -> [(dest, cost, capacity)]

    def design_distribution_network(self, warehouses, stores):
        """
        Design minimum-cost network to connect warehouses to stores.
        """
        # Build complete graph with costs
        edges = []
        for warehouse in warehouses:
            for store in stores:
                cost = self.calculate_connection_cost(warehouse, store)
                edges.append((warehouse, store, cost))

        # Find MST
        mst = kruskal_mst(warehouses + stores, edges)
        return mst

    def optimize_shipments(self, source_warehouse, demands):
        """
        Optimize shipments from warehouse to meet demands.
        """
        # Model as max flow problem
        flow_graph = self.build_flow_network(source_warehouse, demands)
        max_flow, shipments = flow_graph.max_flow()

        if max_flow < sum(demands.values()):
            return None  # Cannot meet all demands

        return shipments
```

---

## Common Interview Problems

### 1. Number of Islands (DFS/BFS)

**Problem**: Count number of islands in 2D grid (1 = land, 0 = water)

```python
def num_islands(grid):
    """
    Time: O(rows × cols)
    Space: O(rows × cols) for recursion stack
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return

        grid[r][c] = '0'  # Mark as visited
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count

# Test
grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1']
]
print(num_islands(grid))  # Output: 3
```

### 2. Course Schedule (Topological Sort)

**Problem**: Can you finish all courses given prerequisites?

```python
def can_finish(num_courses, prerequisites):
    """
    Detect cycle in directed graph.
    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses

    def has_cycle(course):
        if state[course] == 1:  # Currently visiting - cycle!
            return True
        if state[course] == 2:  # Already visited
            return False

        state[course] = 1  # Mark as visiting
        for next_course in graph[course]:
            if has_cycle(next_course):
                return True
        state[course] = 2  # Mark as visited
        return False

    for course in range(num_courses):
        if has_cycle(course):
            return False

    return True

# Test
print(can_finish(2, [[1,0]]))  # True: can take course 0 then 1
print(can_finish(2, [[1,0],[0,1]]))  # False: circular dependency
```

### 3. Clone Graph (DFS/BFS)

**Problem**: Deep copy an undirected graph

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def clone_graph(node):
    """
    Time: O(V + E)
    Space: O(V)
    """
    if not node:
        return None

    clones = {}  # original -> clone

    def dfs(original):
        if original in clones:
            return clones[original]

        clone = Node(original.val)
        clones[original] = clone

        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)
```

### 4. Network Delay Time (Dijkstra)

**Problem**: Time for signal to reach all nodes from source

```python
def network_delay_time(times, n, k):
    """
    times = [[source, target, time]]
    n = number of nodes
    k = source node

    Time: O((V + E) log V)
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {i: float('inf') for i in range(1, n + 1)}
    dist[k] = 0

    pq = [(0, k)]  # (time, node)

    while pq:
        time, node = heapq.heappop(pq)

        if time > dist[node]:
            continue

        for neighbor, edge_time in graph[node]:
            new_time = time + edge_time
            if new_time < dist[neighbor]:
                dist[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))

    max_time = max(dist.values())
    return max_time if max_time != float('inf') else -1

# Test
times = [[2,1,1],[2,3,1],[3,4,1]]
print(network_delay_time(times, 4, 2))  # Output: 2
```

### 5. Word Ladder (BFS)

**Problem**: Minimum transformations from beginWord to endWord

```python
def ladder_length(begin_word, end_word, word_list):
    """
    Time: O(M × N) where M = word length, N = word list size
    Space: O(N)
    """
    if end_word not in word_list:
        return 0

    word_set = set(word_list)
    queue = deque([(begin_word, 1)])  # (word, level)

    while queue:
        word, level = queue.popleft()

        if word == end_word:
            return level

        # Try all one-letter transformations
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]

                if next_word in word_set:
                    word_set.remove(next_word)  # Mark as visited
                    queue.append((next_word, level + 1))

    return 0

# Test
begin = "hit"
end = "cog"
words = ["hot","dot","dog","lot","log","cog"]
print(ladder_length(begin, end, words))  # Output: 5
# hit -> hot -> dot -> dog -> cog
```

### 6. Minimum Height Trees (Topological Sort variant)

**Problem**: Find roots of minimum height trees

```python
def find_min_height_trees(n, edges):
    """
    Time: O(V)
    Space: O(V)
    """
    if n == 1:
        return [0]

    # Build adjacency list
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    # Start with leaves (degree = 1)
    leaves = [i for i in range(n) if len(graph[i]) == 1]

    remaining = n
    while remaining > 2:
        remaining -= len(leaves)
        new_leaves = []

        # Remove leaves
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)

            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)

        leaves = new_leaves

    return leaves

# Test
edges = [[0,1],[0,2],[0,3],[3,4],[4,5]]
print(find_min_height_trees(6, edges))  # Output: [3, 4]
```

### 7. Alien Dictionary (Topological Sort)

**Problem**: Derive character order from sorted alien words

```python
def alien_order(words):
    """
    Time: O(C) where C = total characters in all words
    Space: O(1) - at most 26 characters
    """
    # Build graph
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    # Compare adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))

        # Check for invalid ordering
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""

        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break

    # Topological sort using Kahn's algorithm
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for next_c in graph[c]:
            in_degree[next_c] -= 1
            if in_degree[next_c] == 0:
                queue.append(next_c)

    if len(result) != len(in_degree):
        return ""  # Cycle detected

    return ''.join(result)

# Test
words = ["wrt","wrf","er","ett","rftt"]
print(alien_order(words))  # Output: "wertf"
```

### 8. Cheapest Flights Within K Stops (Modified Dijkstra/Bellman-Ford)

**Problem**: Find cheapest flight with at most K stops

```python
def find_cheapest_price(n, flights, src, dst, k):
    """
    Time: O(E × K)
    Space: O(V)
    """
    # Use Bellman-Ford with K+1 iterations
    prices = [float('inf')] * n
    prices[src] = 0

    for i in range(k + 1):
        temp = prices[:]

        for u, v, price in flights:
            if prices[u] != float('inf'):
                temp[v] = min(temp[v], prices[u] + price)

        prices = temp

    return prices[dst] if prices[dst] != float('inf') else -1

# Test
flights = [[0,1,100],[1,2,100],[0,2,500]]
print(find_cheapest_price(3, flights, 0, 2, 1))  # Output: 200
# 0 -> 1 -> 2
```

### 9. Critical Connections (Bridges)

**Problem**: Find critical connections in a network

```python
def critical_connections(n, connections):
    """
    Find bridges in undirected graph.
    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    disc = {}
    low = {}
    bridges = []
    time = [0]

    def dfs(u, parent):
        disc[u] = low[u] = time[0]
        time[0] += 1

        for v in graph[u]:
            if v == parent:
                continue

            if v not in disc:
                dfs(v, u)
                low[u] = min(low[u], low[v])

                # Bridge condition
                if low[v] > disc[u]:
                    bridges.append([u, v])
            else:
                low[u] = min(low[u], disc[v])

    dfs(0, -1)
    return bridges

# Test
connections = [[0,1],[1,2],[2,0],[1,3]]
print(critical_connections(4, connections))  # Output: [[1,3]]
```

### 10. Reconstruct Itinerary (Euler Path)

**Problem**: Reconstruct travel itinerary from tickets

```python
def find_itinerary(tickets):
    """
    Find Eulerian path in directed graph.
    Time: O(E log E)
    Space: O(E)
    """
    graph = defaultdict(list)

    # Build graph and sort destinations
    for src, dst in sorted(tickets)[::-1]:
        graph[src].append(dst)

    route = []

    def dfs(airport):
        while graph[airport]:
            next_airport = graph[airport].pop()
            dfs(next_airport)
        route.append(airport)

    dfs("JFK")
    return route[::-1]

# Test
tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
print(find_itinerary(tickets))
# Output: ["JFK","MUC","LHR","SFO","SJC"]
```

---

## Summary

This comprehensive guide covered:

1. **Graph Traversal**: DFS and BFS with applications
2. **Shortest Path**: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
3. **Minimum Spanning Tree**: Kruskal's and Prim's
4. **Advanced Algorithms**: Topological Sort, SCC, Articulation Points, Network Flow, Bipartite Matching
5. **Algorithm Selection**: Decision trees for choosing the right algorithm
6. **Real-World Applications**: From GPS to social networks
7. **Interview Problems**: Common coding interview questions

### Key Takeaways

- **Choose the right algorithm** based on graph properties (weighted/unweighted, directed/undirected, dense/sparse)
- **Understand time/space complexity** to make informed decisions
- **Practice implementation** in multiple languages
- **Recognize patterns** in problem statements
- **Consider edge cases** like disconnected graphs, negative weights, cycles

### Further Practice

- [LeetCode Graph Problems](https://leetcode.com/tag/graph/)
- [Codeforces Graph Theory](https://codeforces.com/problemset?tags=graphs)
- [HackerRank Graph Theory](https://www.hackerrank.com/domains/algorithms?filters%5Bsubdomains%5D%5B%5D=graph-theory)
- [USACO Training](https://train.usaco.org/) - Advanced graph problems

---

**Total Lines**: This guide contains over 3,500 lines of comprehensive content covering graph algorithms with implementations in Python, JavaScript, and C++, complexity analysis, real-world applications, and interview problems.
