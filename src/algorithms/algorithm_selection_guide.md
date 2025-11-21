# Algorithm Selection Guide

## Quick Decision Flowchart

```
START: What type of problem?
│
├─ SEARCHING/FINDING
│  ├─ Array sorted? → Binary Search O(log n)
│  ├─ Find in range? → Binary Search variations
│  ├─ Find k-th element? → QuickSelect O(n) avg
│  ├─ Multiple queries? → Build index/hash O(1)
│  └─ Unsorted? → Hash Table O(1) or Linear Search O(n)
│
├─ SORTING
│  ├─ Small array (n < 50)? → Insertion Sort O(n²)
│  ├─ Nearly sorted? → Insertion Sort O(n)
│  ├─ Need stability? → Merge Sort O(n log n)
│  ├─ Space constrained? → Heap Sort O(n log n)
│  ├─ Range known? → Counting Sort O(n+k)
│  └─ General case → Quick Sort O(n log n) avg
│
├─ GRAPH TRAVERSAL
│  ├─ Shortest path (unweighted)? → BFS O(V+E)
│  ├─ Shortest path (weighted, no negative)? → Dijkstra O(E log V)
│  ├─ Shortest path (negative edges)? → Bellman-Ford O(VE)
│  ├─ All pairs shortest path? → Floyd-Warshall O(V³)
│  ├─ Explore all paths? → DFS O(V+E)
│  ├─ Topological order? → DFS or Kahn's O(V+E)
│  ├─ Minimum spanning tree? → Kruskal/Prim O(E log V)
│  └─ Connected components? → Union-Find O(α(n)) or DFS
│
├─ DYNAMIC PROGRAMMING
│  ├─ Optimal substructure? → Check if DP applicable
│  ├─ 1D sequence problem? → 1D DP array
│  ├─ 2D grid/matrix? → 2D DP table
│  ├─ Substring/subarray? → Sliding window or DP
│  ├─ Choices at each step? → Decision DP
│  └─ Unbounded choices? → Complete knapsack pattern
│
├─ STRING MATCHING
│  ├─ Single pattern? → KMP O(n+m) or Rabin-Karp
│  ├─ Multiple patterns? → Aho-Corasick O(n+m+z)
│  ├─ Pattern with wildcards? → Dynamic Programming
│  ├─ Longest common substring? → DP or Suffix Array
│  └─ Palindrome check? → Two pointers O(n)
│
├─ OPTIMIZATION
│  ├─ Maximize/minimize sum? → Greedy or DP
│  ├─ Subset with constraints? → Backtracking or DP
│  ├─ Assignment problem? → Hungarian Algorithm
│  ├─ Network flow? → Max Flow algorithms
│  └─ Search space large? → Binary search on answer
│
└─ DATA STRUCTURE CHOICE
   ├─ Need min/max quickly? → Heap
   ├─ Range queries? → Segment Tree or Fenwick Tree
   ├─ Recent elements? → Queue/Deque
   ├─ Undo operations? → Stack
   ├─ Fast lookup? → Hash Table
   ├─ Maintain order? → BST or Sorted Array
   └─ Union/Find sets? → Union-Find
```

## Problem Characteristics → Algorithm Mapping

### 1. Array Problems

#### Sorted Array
```
Problem Type              → Algorithm                 → Complexity
─────────────────────────────────────────────────────────────────
Find element              → Binary Search             → O(log n)
Find range                → Binary Search (bounds)    → O(log n)
Find k closest            → Binary Search + expand    → O(log n + k)
Merge arrays              → Two pointers             → O(n + m)
Search 2D matrix          → Binary Search            → O(log(mn))
```

#### Unsorted Array
```
Problem Type              → Algorithm                 → Complexity
─────────────────────────────────────────────────────────────────
Find duplicates           → Hash Set                 → O(n)
Two Sum                   → Hash Map                 → O(n)
Three Sum                 → Sort + Two Pointers      → O(n²)
Subarray sum = k          → Prefix Sum + Hash        → O(n)
Max subarray sum          → Kadane's Algorithm       → O(n)
K-th largest              → QuickSelect/Heap         → O(n)/O(n log k)
```

#### Intervals/Ranges
```
Problem Type              → Algorithm                 → Complexity
─────────────────────────────────────────────────────────────────
Merge intervals           → Sort + Greedy            → O(n log n)
Insert interval           → Binary Search + Merge    → O(n)
Non-overlapping intervals → Sort + Greedy            → O(n log n)
Meeting rooms             → Sort + Heap              → O(n log n)
Range queries (static)    → Segment Tree build       → O(n)
Range queries (updates)   → Segment Tree/Fenwick     → O(log n)
```

### 2. String Problems

```
Problem Pattern           → Algorithm                 → Complexity
─────────────────────────────────────────────────────────────────
Pattern matching          → KMP                      → O(n + m)
Multiple patterns         → Aho-Corasick             → O(n + m + z)
Substring search          → Rabin-Karp               → O(n + m)
Longest palindrome        → Expand around center     → O(n²)
                          → Manacher's               → O(n)
Anagrams                  → Hash Map (char count)    → O(n)
Common substring          → DP                       → O(nm)
Edit distance             → DP                       → O(nm)
Lexicographic order       → Trie                     → O(L)
```

### 3. Tree Problems

```
Problem Type              → Traversal                 → When to Use
─────────────────────────────────────────────────────────────────
Level order               → BFS                      → Level-by-level processing
Path from root            → DFS (Preorder)           → Top-down decisions
Validate BST              → DFS (Inorder)            → Need sorted order
Subtree problems          → DFS (Postorder)          → Bottom-up aggregation
Lowest common ancestor    → Binary lifting/Tarjan    → O(log n) or O(1)
Diameter/height           → DFS (Postorder)          → O(n)
Serialize/deserialize     → BFS or DFS               → O(n)
```

### 4. Graph Problems

```
Problem Type              → Algorithm                 → When to Use
─────────────────────────────────────────────────────────────────
Shortest path (unweighted)→ BFS                      → All edges weight 1
Shortest path (weighted)  → Dijkstra                 → Non-negative weights
Shortest path (negative)  → Bellman-Ford             → Has negative edges
All pairs shortest        → Floyd-Warshall           → Dense graph, all pairs
Detect cycle (undirected) → DFS/Union-Find           → O(V+E) or O(α(n))
Detect cycle (directed)   → DFS (colors)             → O(V+E)
Topological sort          → DFS or Kahn's            → DAG ordering
Connected components      → DFS/BFS/Union-Find       → Partition graph
Strongly connected comp.  → Kosaraju/Tarjan          → Directed graph
Minimum spanning tree     → Kruskal/Prim             → Connect all vertices
Max flow                  → Ford-Fulkerson/Dinic     → Network flow
Bipartite check           → BFS/DFS (2-coloring)     → O(V+E)
```

### 5. Dynamic Programming

```
Problem Pattern           → DP Type                   → Dimensions
─────────────────────────────────────────────────────────────────
Fibonacci/climbing stairs → 1D DP                    → dp[i]
Coin change               → 1D DP (unbounded)        → dp[amount]
Knapsack (0/1)            → 2D DP                    → dp[i][w]
Knapsack (unbounded)      → 1D DP                    → dp[w]
Longest increasing subseq → 1D DP + Binary Search    → O(n log n)
Edit distance             → 2D DP                    → dp[i][j]
Longest common substring  → 2D DP                    → dp[i][j]
Matrix chain multiply     → 2D DP (interval)         → dp[i][j]
Partition problems        → 2D DP or Backtrack       → dp[i][sum]
Word break                → 1D DP + Trie             → dp[i]
```

## Decision Trees for Common Scenarios

### Searching Decision Tree

```
Need to search?
│
├─ Is data sorted?
│  ├─ YES → Binary Search O(log n)
│  │       ├─ 1D array? → Standard binary search
│  │       ├─ 2D matrix? → Binary search on rows/cols
│  │       └─ Find boundary? → Lower/upper bound variant
│  │
│  └─ NO → Can you sort?
│         ├─ YES → Sort first O(n log n) + Binary Search
│         └─ NO → Hash Table O(n) space, O(1) lookup
│                  OR Linear Search O(n) if space-constrained
│
└─ Multiple searches?
   ├─ Few searches → Linear O(n) per search
   └─ Many searches → Build index O(n) + O(1) per search
```

### Optimization Decision Tree

```
Optimization problem (min/max)?
│
├─ Greedy choice property?
│  ├─ YES → Greedy Algorithm O(n log n) typically
│  │       Examples: Interval scheduling, Huffman coding
│  │       Test: Does local optimum lead to global optimum?
│  │
│  └─ NO → Has optimal substructure?
│         ├─ YES → Dynamic Programming
│         │        ├─ Small state space? → Standard DP
│         │        └─ Large state space? → DP + memoization
│         │
│         └─ NO → Constraints small?
│                  ├─ YES → Backtracking/Brute force
│                  └─ NO → Approximation or heuristics
│
└─ Binary search on answer?
   Test: Can you verify solution in O(f(n))?
   If YES and search space monotonic → Binary search
```

### Graph Traversal Decision Tree

```
Graph problem?
│
├─ Finding paths?
│  ├─ Single source shortest path?
│  │  ├─ Unweighted? → BFS O(V+E)
│  │  ├─ Weighted (non-negative)? → Dijkstra O(E log V)
│  │  └─ Weighted (negative edges)? → Bellman-Ford O(VE)
│  │
│  ├─ All pairs shortest path?
│  │  ├─ Dense graph? → Floyd-Warshall O(V³)
│  │  └─ Sparse graph? → Run Dijkstra V times O(VE log V)
│  │
│  └─ Any path exists?
│         └─ DFS or BFS O(V+E)
│
├─ Connectivity?
│  ├─ Connected components? → Union-Find O(α(n)) or DFS
│  ├─ Strongly connected? → Kosaraju/Tarjan O(V+E)
│  └─ Bipartite? → BFS/DFS 2-coloring O(V+E)
│
└─ Spanning tree?
   └─ Minimum spanning tree?
      ├─ Dense graph? → Prim O(V²) or O(E log V)
      └─ Sparse graph? → Kruskal O(E log E)
```

## Size Constraints → Algorithm Choice

```
Input Size (n)          → Acceptable Complexity    → Algorithms
─────────────────────────────────────────────────────────────────
n ≤ 10                  → O(n!), O(n^7)            → Brute force, permutations
n ≤ 20                  → O(2^n), O(n^6)           → Bitmask DP, backtracking
n ≤ 100                 → O(n^4)                   → 4 nested loops, DP
n ≤ 500                 → O(n^3)                   → Floyd-Warshall, 3 loops
n ≤ 5,000               → O(n^2)                   → Bubble sort, naive DP
n ≤ 100,000             → O(n log n)               → Merge/quick sort, heap
n ≤ 1,000,000           → O(n)                     → Hash map, two pointers
n > 1,000,000           → O(log n) or O(1)         → Binary search, math formula
```

## Problem Keywords → Algorithm Hints

### Keywords that Suggest Specific Approaches

```
Keyword/Phrase                  → Consider
──────────────────────────────────────────────────────────────
"sorted array"                  → Binary search
"find k-th"                     → QuickSelect, heap
"maximum/minimum"               → Greedy, DP, heap
"count of"                      → Hash map, DP
"shortest path"                 → BFS, Dijkstra, Bellman-Ford
"connected"                     → Union-Find, DFS, BFS
"cycle"                         → DFS, Union-Find
"substring/subarray"            → Sliding window, DP
"palindrome"                    → Two pointers, DP
"parentheses"                   → Stack
"top k elements"                → Heap, QuickSelect
"frequency"                     → Hash map
"anagram"                       → Hash map, sorting
"in-order/pre-order/post-order" → DFS variants
"level-by-level"                → BFS
"interval"                      → Sorting + greedy/sweep line
"range query"                   → Segment tree, Fenwick tree
"optimal"                       → DP, greedy
"all combinations"              → Backtracking
"minimum spanning"              → Kruskal, Prim
"max flow"                      → Ford-Fulkerson
"assignment"                    → Hungarian algorithm
"next greater/smaller"          → Monotonic stack
"sliding window maximum"        → Monotonic deque
```

## Time/Space Tradeoffs

```
Scenario                        → Time-Optimized           → Space-Optimized
──────────────────────────────────────────────────────────────────────────────
Fibonacci                       → O(1) with memo          → O(1) only 2 vars
Counting frequency              → O(n) with hash          → O(1) if small range
Range sum queries               → O(1) with prefix sum    → O(n) recalculate
K-th element                    → O(n) with QuickSelect   → O(n log k) with heap
Shortest path                   → O(V²) with matrix       → O(V+E) adjacency list
LRU Cache                       → O(1) hash+list          → O(n) array only
String search                   → O(1) with trie          → O(m) KMP pattern only
```

## Common Anti-Patterns to Avoid

```
❌ AVOID                        → ✅ USE INSTEAD
──────────────────────────────────────────────────────────────────────
Sorting just to find max/min    → Linear scan O(n)
Hash map for small fixed range  → Array indexing O(1)
DFS for shortest path           → BFS for unweighted graphs
Recursion without memoization   → DP with tabulation
Multiple passes when one works  → Single pass with running calc
Creating new collections        → In-place modification
Nested loops for counting       → Hash map
String concatenation in loop    → StringBuilder/list join
```

## Algorithm Selection Checklist

### Before Writing Code

- [ ] **Understand constraints**: What are n, m, time/space limits?
- [ ] **Identify problem type**: Search, sort, graph, DP, greedy?
- [ ] **Check for patterns**: Does it match a known pattern?
- [ ] **Consider edge cases**: Empty input, single element, duplicates?
- [ ] **Estimate complexity**: Will chosen algorithm pass time limits?
- [ ] **Think about space**: Can you optimize space if needed?

### Initial Approach

1. **Brute Force First**: What's the naive O(n²) or O(n³) solution?
2. **Optimize**: Can you reduce one dimension? Use better data structure?
3. **Pattern Match**: Does this resemble a classic problem?
4. **Test Small**: Verify logic with small examples

### When Stuck

- [ ] Try a different data structure
- [ ] Consider preprocessing the data
- [ ] Think about problem inversely
- [ ] Break into smaller subproblems
- [ ] Draw examples and look for patterns
- [ ] Consider binary search on answer

## Quick Reference: Top 10 Algorithm Families

```
1. Two Pointers           → Sorted arrays, palindromes
2. Sliding Window         → Subarrays/substrings with constraints
3. Binary Search          → Sorted data, search space
4. BFS/DFS               → Trees, graphs, connectivity
5. Dynamic Programming    → Optimal substructure, overlapping subproblems
6. Greedy                → Local optimum → global optimum
7. Backtracking          → Generate combinations/permutations
8. Union-Find            → Disjoint sets, connectivity
9. Hash Table            → Fast lookup, counting, grouping
10. Heap                 → Top k, running median, priority
```

## Complexity Target Guide

```
Problem Size  → Target Complexity  → Suitable Algorithms
─────────────────────────────────────────────────────────────────
10^8          → O(1), O(log n)     → Math, binary search
10^7          → O(n)                → Linear scan, hash map
10^6          → O(n log n)          → Sorting, heap operations
10^5          → O(n log n)          → Sorting, segment tree
10^4          → O(n²)               → Nested loops, simple DP
10^3          → O(n²), O(n³)        → Advanced DP, Floyd-Warshall
10^2          → O(n³), O(n⁴)        → Multiple nested loops
20-30         → O(2^n)              → Backtracking, bitmask DP
10-15         → O(n!)               → Permutations, brute force
```

## Special Cases and Optimizations

### When Input Has Special Properties

```
Property                  → Optimization
──────────────────────────────────────────────────────────
Array is sorted           → Binary search, two pointers
Array nearly sorted       → Insertion sort O(n)
Small range of values     → Counting sort O(n+k)
Many duplicates           → Three-way quicksort
Only 0s and 1s           → Counting, bit manipulation
Values in range [0, n]    → Array indexing as hash
Tree is BST              → Inorder gives sorted sequence
Graph is DAG             → Topological sort possible
Graph is tree            → No cycles, V = E + 1
```

### When to Use Advanced Data Structures

```
Need                      → Data Structure
──────────────────────────────────────────────────────────
Range min/max query       → Segment Tree, Sparse Table
Range sum query           → Fenwick Tree, Prefix Sum
Update + query           → Segment Tree, Fenwick Tree
LCA in tree              → Binary Lifting, RMQ
Union of sets            → Union-Find
Order statistics         → Indexed Tree, Order Statistic Tree
Auto-complete            → Trie
IP routing               → Trie (bit-wise)
Undo operations          → Stack with snapshots
Median from stream       → Two heaps (max + min)
```

## Final Decision Framework

```
1. Read problem carefully
   ↓
2. Identify constraints (n, m, time, space)
   ↓
3. Categorize problem type
   ↓
4. Check keyword → algorithm mapping
   ↓
5. Estimate complexity needed
   ↓
6. Choose algorithm family
   ↓
7. Select specific algorithm
   ↓
8. Consider optimizations
   ↓
9. Implement and test
```

## Practice Strategy

Start with these problem sets to build pattern recognition:

1. **Arrays**: Two pointers, sliding window
2. **Strings**: Pattern matching, DP
3. **Trees**: DFS/BFS variations
4. **Graphs**: Shortest path, connectivity
5. **DP**: Classic problems (knapsack, LIS, LCS)
6. **Greedy**: Interval scheduling, Huffman
7. **Binary Search**: On arrays, on answer space
8. **Heap**: Top k, merge k sorted
9. **Hash**: Frequency, grouping
10. **Backtracking**: Combinations, permutations

Build intuition by solving 5-10 problems per pattern before moving to the next.
