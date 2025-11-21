# Algorithm Complexity Cheatsheet

## Table of Contents
- [Data Structures](#data-structures)
- [Sorting Algorithms](#sorting-algorithms)
- [Searching Algorithms](#searching-algorithms)
- [Graph Algorithms](#graph-algorithms)
- [String Algorithms](#string-algorithms)
- [Tree Algorithms](#tree-algorithms)
- [Dynamic Programming](#dynamic-programming)
- [Mathematical Algorithms](#mathematical-algorithms)
- [Complexity Classes](#complexity-classes)

## Data Structures

### Basic Data Structures

| Data Structure | Access | Search | Insertion | Deletion | Space |
|----------------|--------|--------|-----------|----------|-------|
| **Array** | O(1) | O(n) | O(n) | O(n) | O(n) |
| **Dynamic Array** | O(1) | O(n) | O(1)* | O(n) | O(n) |
| **Linked List** | O(n) | O(n) | O(1) | O(1) | O(n) |
| **Doubly Linked List** | O(n) | O(n) | O(1) | O(1) | O(n) |
| **Stack** | O(n) | O(n) | O(1) | O(1) | O(n) |
| **Queue** | O(n) | O(n) | O(1) | O(1) | O(n) |
| **Deque** | O(1) | O(n) | O(1) | O(1) | O(n) |
| **Hash Table** | N/A | O(1)* | O(1)* | O(1)* | O(n) |

*Amortized or average case

### Trees

| Data Structure | Access | Search | Insert | Delete | Space | Notes |
|----------------|--------|--------|--------|--------|-------|-------|
| **Binary Search Tree** | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n) | O(n) worst case |
| **AVL Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Self-balancing |
| **Red-Black Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Self-balancing |
| **Splay Tree** | O(log n)* | O(log n)* | O(log n)* | O(log n)* | O(n) | Amortized |
| **B-Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Good for disk |
| **B+ Tree** | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Range queries |
| **Segment Tree** | O(log n) | O(log n) | O(n) | N/A | O(n) | Range queries |
| **Fenwick Tree** | O(log n) | O(log n) | O(n) | N/A | O(n) | Prefix sums |
| **Trie** | O(L) | O(L) | O(L) | O(L) | O(ALPHABET*N*L) | L = key length |
| **Suffix Tree** | O(L) | O(L) | O(n) | N/A | O(n²) | String matching |
| **Suffix Array** | O(log n) | O(log n) | O(n log n) | N/A | O(n) | Space-efficient |

### Heaps

| Data Structure | Find-Min | Extract-Min | Insert | Decrease-Key | Merge | Space |
|----------------|----------|-------------|--------|--------------|-------|-------|
| **Binary Heap** | O(1) | O(log n) | O(log n) | O(log n) | O(n) | O(n) |
| **Binomial Heap** | O(1) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) |
| **Fibonacci Heap** | O(1) | O(log n)* | O(1) | O(1)* | O(1) | O(n) |
| **Pairing Heap** | O(1) | O(log n)* | O(1) | O(log n)* | O(1) | O(n) |

*Amortized

### Advanced Data Structures

| Data Structure | Operation | Time Complexity | Space | Use Case |
|----------------|-----------|-----------------|-------|----------|
| **Disjoint Set (Union-Find)** | Union | O(α(n))* | O(n) | Connectivity |
| | Find | O(α(n))* | | |
| **Bloom Filter** | Insert | O(k) | O(m) | Membership test |
| | Search | O(k) | | |
| **Skip List** | Search | O(log n)* | O(n log n)* | Sorted data |
| | Insert | O(log n)* | | |
| **Treap** | Search | O(log n)* | O(n) | BST + Heap |
| | Insert | O(log n)* | | |
| **Interval Tree** | Search | O(log n + k) | O(n) | Overlapping intervals |
| **Range Tree** | Range Query | O(log^d n + k) | O(n log^(d-1) n) | d-dimensional |
| **KD-Tree** | Nearest Neighbor | O(log n)* | O(n) | Spatial queries |

*Average case; α(n) is inverse Ackermann function (practically constant)

## Sorting Algorithms

### Comparison-Based Sorts

| Algorithm | Best | Average | Worst | Space | Stable | Notes |
|-----------|------|---------|-------|-------|--------|-------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes | Simple, rarely used |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | No | Minimal swaps |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | Yes | Good for small/nearly sorted |
| **Shell Sort** | O(n log n) | O(n^1.5) | O(n²) | O(1) | No | Gap sequence dependent |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | Guaranteed O(n log n) |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | No* | Usually fastest |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | No | In-place, guaranteed |
| **Tree Sort** | O(n log n) | O(n log n) | O(n²) | O(n) | Yes | Using BST |
| **Tim Sort** | O(n) | O(n log n) | O(n log n) | O(n) | Yes | Python/Java default |
| **Intro Sort** | O(n log n) | O(n log n) | O(n log n) | O(log n) | No | Hybrid (C++ default) |

*Can be made stable

### Non-Comparison Sorts

| Algorithm | Best | Average | Worst | Space | Stable | Constraints |
|-----------|------|---------|-------|-------|--------|-------------|
| **Counting Sort** | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes | Range k known |
| **Radix Sort** | O(d(n+k)) | O(d(n+k)) | O(d(n+k)) | O(n+k) | Yes | d digits, base k |
| **Bucket Sort** | O(n+k) | O(n+k) | O(n²) | O(n+k) | Yes | Uniform distribution |

## Searching Algorithms

| Algorithm | Data Structure | Time Complexity | Space | Notes |
|-----------|----------------|-----------------|-------|-------|
| **Linear Search** | Array | O(n) | O(1) | Unsorted data |
| **Binary Search** | Sorted Array | O(log n) | O(1) | Iterative |
| **Binary Search** | Sorted Array | O(log n) | O(log n) | Recursive |
| **Jump Search** | Sorted Array | O(√n) | O(1) | Block-based |
| **Interpolation Search** | Uniformly Sorted | O(log log n)* | O(1) | Average case |
| **Exponential Search** | Sorted Array | O(log n) | O(1) | Unbounded search |
| **Fibonacci Search** | Sorted Array | O(log n) | O(1) | Division-free |
| **Ternary Search** | Unimodal Function | O(log₃ n) | O(1) | Find max/min |

## Graph Algorithms

### Traversal

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|-----------------|------------------|-------|
| **BFS** | O(V + E) | O(V) | Queue-based, shortest path |
| **DFS** | O(V + E) | O(V) | Stack/recursion, exploration |
| **Iterative Deepening DFS** | O(V + E) | O(d) | d = depth, space-efficient |

### Shortest Path

| Algorithm | Time Complexity | Space | Graph Type | Notes |
|-----------|-----------------|-------|------------|-------|
| **Dijkstra** | O(E log V) | O(V) | Weighted, non-negative | Min heap |
| **Dijkstra (array)** | O(V²) | O(V) | Dense graphs | Array implementation |
| **Bellman-Ford** | O(VE) | O(V) | Weighted, negative edges | Detects negative cycles |
| **Floyd-Warshall** | O(V³) | O(V²) | All pairs | Dense graphs |
| **Johnson's** | O(V² log V + VE) | O(V²) | All pairs, sparse | Uses Bellman-Ford + Dijkstra |
| **A\*** | O(E) | O(V) | Weighted | With good heuristic |
| **BFS (unweighted)** | O(V + E) | O(V) | Unweighted | Shortest path |

### Minimum Spanning Tree

| Algorithm | Time Complexity | Space | Implementation |
|-----------|-----------------|-------|----------------|
| **Kruskal** | O(E log E) | O(V) | Sort edges, Union-Find |
| **Kruskal (sorted)** | O(E α(V)) | O(V) | Pre-sorted edges |
| **Prim (binary heap)** | O(E log V) | O(V) | Priority queue |
| **Prim (fibonacci heap)** | O(E + V log V) | O(V) | Advanced heap |
| **Prim (array)** | O(V²) | O(V) | Dense graphs |
| **Borůvka** | O(E log V) | O(V) | Parallel-friendly |

### Graph Properties

| Algorithm | Time Complexity | Space | Purpose |
|-----------|-----------------|-------|---------|
| **Topological Sort (DFS)** | O(V + E) | O(V) | DAG ordering |
| **Topological Sort (Kahn)** | O(V + E) | O(V) | BFS-based |
| **Cycle Detection (Undirected)** | O(V + E) | O(V) | DFS or Union-Find |
| **Cycle Detection (Directed)** | O(V + E) | O(V) | DFS with colors |
| **Strongly Connected Components (Kosaraju)** | O(V + E) | O(V) | Two DFS passes |
| **Strongly Connected Components (Tarjan)** | O(V + E) | O(V) | Single DFS |
| **Articulation Points** | O(V + E) | O(V) | DFS with low/disc |
| **Bridges** | O(V + E) | O(V) | DFS with low/disc |
| **Bipartite Check** | O(V + E) | O(V) | BFS/DFS 2-coloring |
| **Eulerian Path** | O(E) | O(E) | Hierholzer's algorithm |

### Network Flow

| Algorithm | Time Complexity | Space | Notes |
|-----------|-----------------|-------|-------|
| **Ford-Fulkerson** | O(E * max_flow) | O(V) | Augmenting paths |
| **Edmonds-Karp** | O(VE²) | O(V) | BFS for augmenting path |
| **Dinic** | O(V²E) | O(V) | Level graph + blocking flow |
| **Push-Relabel** | O(V³) | O(V) | Preflow-push |
| **Min-Cost Max-Flow** | O(V²E²) | O(V²) | Cost-aware flow |

## String Algorithms

| Algorithm | Preprocessing | Matching | Space | Purpose |
|-----------|---------------|----------|-------|---------|
| **Naive Search** | O(1) | O(nm) | O(1) | Simple pattern match |
| **KMP** | O(m) | O(n) | O(m) | Pattern matching |
| **Boyer-Moore** | O(m + σ) | O(n/m) best, O(nm) worst | O(σ) | Skip characters |
| **Rabin-Karp** | O(m) | O(n+m) avg, O(nm) worst | O(1) | Rolling hash |
| **Aho-Corasick** | O(Σm) | O(n + z) | O(Σm) | Multiple patterns |
| **Z-Algorithm** | O(n+m) | O(n+m) | O(n+m) | Pattern matching |
| **Suffix Array** | O(n log n) | O(m log n) | O(n) | Multiple queries |
| **Suffix Tree** | O(n) | O(m) | O(n) | Ukkonen's algorithm |
| **Manacher** | O(n) | O(n) | O(n) | Longest palindrome |
| **LCS (DP)** | O(nm) | O(nm) | O(nm) or O(min(n,m)) | Longest common subseq |
| **Edit Distance** | O(nm) | O(nm) | O(nm) or O(min(n,m)) | Levenshtein distance |
| **Trie Build** | O(Σ L) | O(L) | O(ALPHABET*N*L) | Insert/search |

n = text length, m = pattern length, σ = alphabet size, z = total matches

## Tree Algorithms

| Algorithm | Time Complexity | Space | Purpose |
|-----------|-----------------|-------|---------|
| **Inorder Traversal** | O(n) | O(h) | Sorted order (BST) |
| **Preorder Traversal** | O(n) | O(h) | Copy tree, prefix expr |
| **Postorder Traversal** | O(n) | O(h) | Delete tree, postfix expr |
| **Level Order (BFS)** | O(n) | O(w) | Level-by-level |
| **Morris Traversal** | O(n) | O(1) | Space-efficient inorder |
| **LCA (Binary Lifting)** | O(log n) | O(n log n) | After O(n log n) preprocessing |
| **LCA (Tarjan)** | O(n α(n)) | O(n) | Offline algorithm |
| **LCA (RMQ)** | O(1) | O(n) | After O(n) preprocessing |
| **Diameter** | O(n) | O(h) | Longest path |
| **Height** | O(n) | O(h) | Tree height |
| **Serialization** | O(n) | O(n) | Convert to string |

h = height, w = max width

## Dynamic Programming

### Classic Problems

| Problem | Time | Space | Optimized Space | Pattern |
|---------|------|-------|-----------------|---------|
| **Fibonacci** | O(n) | O(n) | O(1) | Linear DP |
| **Climbing Stairs** | O(n) | O(n) | O(1) | Linear DP |
| **Coin Change** | O(nS) | O(S) | O(S) | Unbounded knapsack |
| **0/1 Knapsack** | O(nW) | O(nW) | O(W) | Knapsack |
| **Unbounded Knapsack** | O(nW) | O(W) | O(W) | Knapsack |
| **Longest Increasing Subsequence** | O(n²) | O(n) | O(n) | Sequence DP |
| **LIS (optimized)** | O(n log n) | O(n) | O(n) | Binary search |
| **Longest Common Subsequence** | O(nm) | O(nm) | O(min(n,m)) | 2D DP |
| **Edit Distance** | O(nm) | O(nm) | O(min(n,m)) | 2D DP |
| **Matrix Chain Multiplication** | O(n³) | O(n²) | O(n²) | Interval DP |
| **Palindrome Partitioning** | O(n²) | O(n²) | O(n²) | Interval DP |
| **Word Break** | O(n² + m) | O(n + m) | O(n) | Linear DP + Trie |
| **Egg Drop** | O(nk²) | O(nk) | O(nk) | 2D DP |
| **Egg Drop (optimized)** | O(n log k) | O(nk) | O(k) | Binary search |
| **Subset Sum** | O(nS) | O(nS) | O(S) | Knapsack variant |
| **Partition** | O(nS) | O(S) | O(S) | Subset sum |
| **Rod Cutting** | O(n²) | O(n) | O(n) | Unbounded knapsack |
| **Longest Palindromic Substring** | O(n²) | O(n²) | O(n²) | 2D DP |
| **Longest Palindromic Substring (Manacher)** | O(n) | O(n) | O(n) | Linear |

n, m = input sizes; S = sum/target; W = weight capacity; k = eggs/other parameter

### DP on Trees

| Problem | Time | Space | Pattern |
|---------|------|-------|---------|
| **Tree DP (max path sum)** | O(n) | O(h) | Postorder |
| **Diameter of Tree** | O(n) | O(h) | Postorder |
| **Tree Matching** | O(n) | O(n) | State DP |
| **Vertex Cover** | O(n) | O(n) | State DP |

### DP on Graphs

| Problem | Time | Space | Pattern |
|---------|------|-------|---------|
| **Shortest Path (DP)** | O(VE) | O(V) | Bellman-Ford |
| **Traveling Salesman** | O(n² 2ⁿ) | O(n 2ⁿ) | Bitmask DP |
| **Hamiltonian Path** | O(n² 2ⁿ) | O(n 2ⁿ) | Bitmask DP |

## Mathematical Algorithms

| Algorithm | Time Complexity | Space | Notes |
|-----------|-----------------|-------|-------|
| **Sieve of Eratosthenes** | O(n log log n) | O(n) | Primes up to n |
| **Segmented Sieve** | O(n log log n) | O(√n) | Primes in range |
| **GCD (Euclidean)** | O(log min(a,b)) | O(1) | Iterative |
| **GCD (Recursive)** | O(log min(a,b)) | O(log min(a,b)) | Recursive |
| **Extended GCD** | O(log min(a,b)) | O(log min(a,b)) | ax + by = gcd(a,b) |
| **LCM** | O(log min(a,b)) | O(1) | Using GCD |
| **Modular Exponentiation** | O(log n) | O(1) | Fast power |
| **Modular Inverse** | O(log n) | O(1) | Extended GCD or Fermat |
| **Matrix Multiplication** | O(n³) | O(n²) | Naive |
| **Matrix Multiplication (Strassen)** | O(n^2.807) | O(n²) | Divide & conquer |
| **Matrix Exponentiation** | O(n³ log k) | O(n²) | For recurrences |
| **Fibonacci (Matrix)** | O(log n) | O(1) | Using matrix power |
| **Fast Fourier Transform** | O(n log n) | O(n) | Polynomial multiply |
| **Number Theoretic Transform** | O(n log n) | O(n) | Modular FFT |
| **Chinese Remainder Theorem** | O(n log m) | O(n) | System of congruences |
| **Euler's Totient** | O(√n) | O(1) | Single number |
| **Totient Sieve** | O(n log log n) | O(n) | All numbers up to n |
| **Prime Factorization** | O(√n) | O(log n) | Trial division |
| **Pollard's Rho** | O(n^(1/4)) | O(1) | Probabilistic factoring |
| **Miller-Rabin** | O(k log³ n) | O(1) | Primality test |
| **Primality (AKS)** | O(log⁶ n) | O(log n) | Deterministic |

## Bitwise Operations

| Operation | Time | Space | Purpose |
|-----------|------|-------|---------|
| **Set bit** | O(1) | O(1) | x \| (1 << i) |
| **Clear bit** | O(1) | O(1) | x & ~(1 << i) |
| **Toggle bit** | O(1) | O(1) | x ^ (1 << i) |
| **Check bit** | O(1) | O(1) | x & (1 << i) |
| **Count set bits** | O(log n) | O(1) | Brian Kernighan |
| **Count set bits (lookup)** | O(1) | O(2^k) | Precompute k-bit table |
| **Power of 2 check** | O(1) | O(1) | x & (x-1) == 0 |
| **Next power of 2** | O(log n) | O(1) | Bit manipulation |

## Greedy Algorithms

| Problem | Time Complexity | Space | Notes |
|---------|-----------------|-------|-------|
| **Activity Selection** | O(n log n) | O(1) | Sort by end time |
| **Huffman Coding** | O(n log n) | O(n) | Min heap |
| **Fractional Knapsack** | O(n log n) | O(1) | Sort by value/weight |
| **Job Sequencing** | O(n² ) or O(n log n) | O(n) | Sort by profit |
| **Minimum Platforms** | O(n log n) | O(n) | Sort arrivals/departures |

## Divide and Conquer

| Algorithm | Time Complexity | Space | Recurrence |
|-----------|-----------------|-------|------------|
| **Binary Search** | O(log n) | O(1) or O(log n) | T(n) = T(n/2) + O(1) |
| **Merge Sort** | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| **Quick Sort** | O(n log n) avg | O(log n) | T(n) = 2T(n/2) + O(n) |
| **Strassen's Matrix** | O(n^2.807) | O(n²) | T(n) = 7T(n/2) + O(n²) |
| **Closest Pair** | O(n log n) | O(n) | T(n) = 2T(n/2) + O(n) |
| **QuickSelect** | O(n) avg | O(log n) | T(n) = T(n/2) + O(n) |

## Backtracking

| Problem | Time Complexity | Space | Notes |
|---------|-----------------|-------|-------|
| **N-Queens** | O(n!) | O(n²) | Place queens |
| **Sudoku Solver** | O(9^m) | O(1) | m = empty cells |
| **Permutations** | O(n!) | O(n) | Generate all |
| **Combinations** | O(2ⁿ) | O(n) | Generate all |
| **Subsets** | O(2ⁿ) | O(n) | Generate all |
| **Graph Coloring** | O(m^V) | O(V) | m colors, V vertices |
| **Hamiltonian Path** | O(n!) | O(n) | Brute force |
| **Knight's Tour** | O(8ⁿ) | O(n²) | Warnsdorff's heuristic better |

## Complexity Classes

### Time Complexity Hierarchy (from fastest to slowest)

```
O(1)           < O(log log n)  < O(log n)      < O(√n)
< O(n)         < O(n log n)    < O(n²)         < O(n³)
< O(n^k)       < O(2ⁿ)         < O(n!)         < O(n^n)
```

### Space Complexity Categories

- **O(1)**: Constant - Few variables
- **O(log n)**: Logarithmic - Recursive call stack (balanced)
- **O(n)**: Linear - Single array/hash table
- **O(n log n)**: Linearithmic - Recursive merge sort
- **O(n²)**: Quadratic - 2D matrix
- **O(2ⁿ)**: Exponential - Subset generation

## Amortized Analysis

| Data Structure | Operation | Worst Case | Amortized |
|----------------|-----------|------------|-----------|
| **Dynamic Array** | Insert | O(n) | O(1) |
| **Stack (with doubling)** | Push | O(n) | O(1) |
| **Queue (circular)** | Enqueue | O(n) | O(1) |
| **Splay Tree** | Search/Insert | O(n) | O(log n) |
| **Fibonacci Heap** | Extract-Min | O(n) | O(log n) |
| **Fibonacci Heap** | Decrease-Key | O(log n) | O(1) |
| **Disjoint Set** | Union/Find | O(n) | O(α(n)) |

## Master Theorem Quick Reference

For recurrences of form: T(n) = aT(n/b) + f(n)

Let c = log_b(a):

| Case | Condition | Complexity |
|------|-----------|------------|
| 1 | f(n) = O(n^(c-ε)), ε > 0 | T(n) = Θ(n^c) |
| 2 | f(n) = Θ(n^c log^k n), k ≥ 0 | T(n) = Θ(n^c log^(k+1) n) |
| 3 | f(n) = Ω(n^(c+ε)), ε > 0 | T(n) = Θ(f(n)) |

### Examples

```
T(n) = 2T(n/2) + O(n)         → O(n log n)    [Merge Sort]
T(n) = T(n/2) + O(1)          → O(log n)      [Binary Search]
T(n) = 2T(n/2) + O(1)         → O(n)          [Tree traversal]
T(n) = T(n-1) + O(1)          → O(n)          [Linear recursion]
T(n) = 2T(n-1) + O(1)         → O(2ⁿ)         [Fibonacci naive]
T(n) = 8T(n/2) + O(n²)        → O(n³)         [Naive matrix mult]
T(n) = 7T(n/2) + O(n²)        → O(n^2.807)    [Strassen]
```

## Growth Rates Comparison

For n = 1,000,000:

| Complexity | Operations | Example |
|------------|------------|---------|
| O(1) | 1 | Hash lookup |
| O(log n) | 20 | Binary search |
| O(√n) | 1,000 | Prime check |
| O(n) | 1,000,000 | Linear scan |
| O(n log n) | 20,000,000 | Merge sort |
| O(n²) | 1,000,000,000,000 | Bubble sort |
| O(n³) | 10^18 | Floyd-Warshall (1M nodes) |
| O(2ⁿ) | 2^1000000 | Subset enumeration |

## Performance Tips

### When to Optimize

```
n ≤ 12       → O(n!) acceptable (brute force permutations)
n ≤ 25       → O(2ⁿ) acceptable (subset enumeration)
n ≤ 100      → O(n⁴) acceptable
n ≤ 500      → O(n³) acceptable
n ≤ 5,000    → O(n²) acceptable
n ≤ 1,000,000 → O(n log n) or better needed
n > 1,000,000 → O(n) or O(log n) needed
```

### Common Optimizations

| From | To | Technique |
|------|-----|-----------|
| O(n²) | O(n log n) | Sort + binary search |
| O(n²) | O(n) | Hash table |
| O(n log n) | O(n) | Counting/radix sort |
| O(2ⁿ) | O(n²) | Dynamic programming |
| O(n!) | O(2ⁿ) | Bitmask DP |
| O(log n) | O(1) | Precomputation/memoization |

## Quick Reference: Choose by Input Size

| n | Max Complexity | Algorithms |
|---|---------------|------------|
| n ≤ 10 | O(n!), O(2ⁿ) | Permutations, brute force |
| n ≤ 20 | O(2ⁿ) | Bitmask DP, backtracking |
| n ≤ 500 | O(n³) | Floyd-Warshall, DP |
| n ≤ 5,000 | O(n²) | Nested loops, simple DP |
| n ≤ 10^6 | O(n log n) | Sort, heap, segment tree |
| n > 10^6 | O(n), O(log n) | Hash map, binary search |

## Common Pitfalls

❌ **Avoid**:
- Using O(n²) when O(n log n) exists
- Sorting when hash table would work
- Recomputing values (use memoization)
- Creating copies of large structures
- Using recursion without tail call optimization for O(n) depth

✅ **Prefer**:
- In-place algorithms when possible
- Iterative over recursive for space
- Hash tables for O(1) lookup
- Appropriate data structures for the problem
- Lazy evaluation when applicable
