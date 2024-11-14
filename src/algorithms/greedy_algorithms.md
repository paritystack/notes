# Greedy Algorithms

Greedy algorithms are a class of algorithms that make locally optimal choices at each stage with the hope of finding a global optimum. They are often used for optimization problems where a solution can be built incrementally.

## Key Concepts

- **Greedy Choice Property**: A global optimum can be reached by selecting a local optimum. This property is essential for the effectiveness of greedy algorithms.

- **Optimal Substructure**: A problem exhibits optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems.

## Common Greedy Algorithms

1. **Activity Selection Problem**: This problem involves selecting the maximum number of activities that don't overlap in time. The greedy choice is to always select the next activity that finishes the earliest.

2. **Huffman Coding**: A compression algorithm that uses a greedy approach to assign variable-length codes to input characters, based on their frequencies.

3. **Kruskal's Algorithm**: An algorithm for finding the minimum spanning tree of a graph by adding edges in increasing order of weight, ensuring no cycles are formed.

4. **Prim's Algorithm**: Another algorithm for finding the minimum spanning tree, which grows the spanning tree one vertex at a time, always choosing the smallest edge that connects a vertex in the tree to a vertex outside the tree.

## Applications

Greedy algorithms are widely used in various applications, including:

- **Network Routing**: Finding the shortest path in a network.
- **Resource Allocation**: Distributing resources in a way that maximizes efficiency.
- **Job Scheduling**: Scheduling jobs on machines to minimize completion time.

## Conclusion

Greedy algorithms provide a straightforward and efficient approach to solving optimization problems. While they do not always yield the optimal solution, they are often easier to implement and can be very effective for certain types of problems.
