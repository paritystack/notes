# Network Flow Algorithms

> **Domain:** Graph Theory, Algorithms, Optimization
> **Key Concepts:** Max-Flow, Min-Cut, Residual Graph, Augmenting Paths

**Network Flow** theory deals with modeling transport through a graph where edges have capacities. It answers questions like: "What is the maximum amount of water that can flow through these pipes?" or "What is the bottleneck in this traffic network?"

---

## 1. The Max-Flow Problem

*   **Input:**
    *   A directed graph $G = (V, E)$.
    *   A **Source** node $s$ (where flow starts).
    *   A **Sink** node $t$ (where flow ends).
    *   Each edge $(u, v)$ has a **Capacity** $c(u, v) > 0$.
*   **Goal:** Find the maximum flow $f$ from $s$ to $t$ such that:
    1.  **Capacity Constraint:** Flow on edge $\le$ Capacity ($f(u,v) \le c(u,v)$).
    2.  **Flow Conservation:** Flow In = Flow Out for all nodes except $s$ and $t$.

---

## 2. Max-Flow Min-Cut Theorem

This is the fundamental theorem of network flow.

*   **Cut:** A partition of vertices into two disjoint sets, one containing $s$ and one containing $t$.
*   **Cut Capacity:** The sum of capacities of edges going from the $s$-set to the $t$-set.
*   **Theorem:** The maximum flow in a network is exactly equal to the capacity of the minimum cut.
    $$ \text{Max Flow} = \text{Min Cut} $$

**Intuition:** The flow is limited by the tightest bottleneck in the system.

---

## 3. Algorithms

### 3.1. Ford-Fulkerson Method
The generic approach.
1.  Initialize flow to 0.
2.  While there exists an **Augmenting Path** from $s$ to $t$ in the **Residual Graph**:
    *   Find the bottleneck capacity on this path.
    *   Increase flow along the path by this amount.
    *   Update the Residual Graph (decrease forward capacity, increase backward capacity).
3.  Return max flow.

**The Residual Graph:** If edge $(u, v)$ has capacity 10 and flow 3, the residual graph has:
*   Forward edge $(u, v)$ with capacity 7 (remaining room).
*   Backward edge $(v, u)$ with capacity 3 (ability to "undo" flow).

### 3.2. Edmonds-Karp Implementation
Ford-Fulkerson is an "method", not a specific algorithm, because it doesn't specify *how* to find the path.
*   **Edmonds-Karp:** Uses **BFS (Breadth-First Search)** to find the augmenting path.
*   **Logic:** Always chooses the shortest path (in number of edges).
*   **Complexity:** $O(V E^2)$.

### 3.3. Dinic's Algorithm
The standard for competitive programming and high-performance needs.
*   **Level Graph:** Uses BFS to build a layered graph (only keeping edges that progress "forward" from $s$ to $t$).
*   **Blocking Flow:** Uses DFS on the Level Graph to push flow until no more paths exist in that layer structure.
*   **Complexity:** $O(V^2 E)$. On unit networks (bipartite matching), it's $O(E \sqrt{V})$.

---

## 4. Applications

1.  **Bipartite Matching:** Find the maximum number of pairings (e.g., Job Applicants -> Jobs).
    *   *Setup:* Create source connected to all Applicants, all Jobs connected to sink. Max Flow = Max Matchings.
2.  **Image Segmentation:** Separate foreground from background.
3.  **Airline Scheduling:** Can a fleet of planes cover a set of flights?
4.  **Circulation with Demands:** Flow with lower bounds (minimum flow required).

---

## 5. Python Example (Edmonds-Karp)

```python
from collections import deque

def bfs(capacity, source, sink, parent):
    visited = set()
    queue = deque([source])
    visited.add(source)
    parent[source] = -1
    
    while queue:
        u = queue.popleft()
        for v in range(len(capacity)):
            if v not in visited and capacity[u][v] > 0:
                queue.append(v)
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return True
    return False

def max_flow(capacity, source, sink):
    parent = [0] * len(capacity)
    max_flow_val = 0
    
    while bfs(capacity, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        while(s != source):
            path_flow = min(path_flow, capacity[parent[s]][s])
            s = parent[s]
            
        max_flow_val += path_flow
        
        v = sink
        while(v != source):
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = parent[v]
            
    return max_flow_val
```

