# Heuristic Search Algorithms

## Overview

Heuristic search algorithms use domain knowledge and estimation functions to guide the search process, making them more efficient than blind search methods (BFS, DFS) for large state spaces. These algorithms are essential in AI, pathfinding, optimization, and machine learning applications where exhaustive search is impractical.

**Key Concepts:**
- **Heuristic Function** $h(n)$: Estimates cost from node $n$ to goal
- **Cost Function** $g(n)$: Actual cost from start to node $n$
- **Evaluation Function** $f(n)$: Combines $g(n)$ and $h(n)$ to prioritize nodes
- **Admissibility**: A heuristic is admissible if it never overestimates the true cost
- **Consistency**: A heuristic is consistent if $h(n) \leq c(n,n') + h(n')$ for all neighbors

## Beam Search

**Time**: $O(b \cdot w \cdot d)$ | **Space**: $O(w \cdot d)$ | **Use Case**: NLP, sequence generation, constrained optimization

Where $b$ is branching factor, $w$ is beam width, $d$ is solution depth.

Beam search is a heuristic search algorithm that explores a graph by expanding the most promising nodes in a limited set (the "beam"). It's a greedy algorithm that maintains only the top-k most promising candidates at each level, trading completeness for memory efficiency.

### Core Implementation

```python
from typing import List, Callable, Tuple, Any
import heapq

def beam_search(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    max_depth: int = 100
) -> List:
    """
    Beam search algorithm with configurable beam width.

    Args:
        initial_state: Starting state
        goal_test: Function that returns True if state is goal
        get_neighbors: Function that returns list of (next_state, cost) tuples
        heuristic: Function that estimates cost to goal from a state
        beam_width: Number of candidates to keep at each level
        max_depth: Maximum search depth

    Returns:
        List of states representing the path to goal, or empty list if not found
    """
    # Each candidate is (score, path)
    beam = [(heuristic(initial_state), [initial_state])]

    for depth in range(max_depth):
        # Check if any candidate reached goal
        for score, path in beam:
            if goal_test(path[-1]):
                return path

        # Generate all successors from current beam
        candidates = []
        for score, path in beam:
            current_state = path[-1]

            # Expand current state
            for next_state, cost in get_neighbors(current_state):
                if next_state not in path:  # Avoid cycles
                    new_path = path + [next_state]
                    # Score is the heuristic value (greedy)
                    new_score = heuristic(next_state)
                    candidates.append((new_score, new_path))

        # If no candidates, search failed
        if not candidates:
            return []

        # Keep only top beam_width candidates
        # Use negative score for min-heap to get best scores
        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    # Return best path found if goal not reached
    return min(beam, key=lambda x: x[0])[1] if beam else []


# Example: Finding path in a grid
def grid_beam_search(grid, start, goal, beam_width=5):
    """Beam search for pathfinding in a 2D grid."""

    def goal_test(state):
        return state == goal

    def get_neighbors(state):
        x, y = state
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(grid) and
                0 <= ny < len(grid[0]) and
                grid[nx][ny] != '#'):  # Not a wall
                neighbors.append(((nx, ny), 1))  # cost = 1
        return neighbors

    def heuristic(state):
        # Manhattan distance
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    return beam_search(start, goal_test, get_neighbors, heuristic, beam_width)
```

### Beam Search with Cost Function

```python
def beam_search_with_cost(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    max_depth: int = 100
) -> Tuple[List, float]:
    """
    Beam search that tracks both path and cumulative cost.
    Evaluation function: f(n) = g(n) + h(n)
    """
    # Each candidate is (f_score, g_cost, path)
    beam = [(heuristic(initial_state), 0, [initial_state])]

    for depth in range(max_depth):
        for f_score, g_cost, path in beam:
            if goal_test(path[-1]):
                return path, g_cost

        candidates = []
        for f_score, g_cost, path in beam:
            current_state = path[-1]

            for next_state, step_cost in get_neighbors(current_state):
                if next_state not in path:
                    new_path = path + [next_state]
                    new_g_cost = g_cost + step_cost
                    new_h_cost = heuristic(next_state)
                    new_f_score = new_g_cost + new_h_cost
                    candidates.append((new_f_score, new_g_cost, new_path))

        if not candidates:
            break

        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    if beam:
        best = min(beam, key=lambda x: x[0])
        return best[2], best[1]
    return [], float('inf')
```

### Variations

#### 1. Diverse Beam Search

Encourages diversity among beam candidates to avoid local optima:

```python
def diverse_beam_search(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    num_groups: int = 2,
    diversity_penalty: float = 0.5,
    max_depth: int = 100
) -> List:
    """
    Diverse beam search splits candidates into groups and penalizes
    similarity within groups to encourage exploration.
    """
    group_size = beam_width // num_groups
    beam = [(heuristic(initial_state), [initial_state], 0)]  # (score, path, group)

    for depth in range(max_depth):
        for score, path, group in beam:
            if goal_test(path[-1]):
                return path

        # Generate candidates for each group
        all_candidates = [[] for _ in range(num_groups)]

        for score, path, group in beam:
            current_state = path[-1]

            for next_state, cost in get_neighbors(current_state):
                if next_state not in path:
                    new_path = path + [next_state]
                    base_score = heuristic(next_state)

                    # Add diversity penalty based on similarity to other groups
                    diversity_score = 0
                    for other_group in range(num_groups):
                        if other_group != group:
                            # Simple diversity: penalize if states are similar
                            # (implementation depends on state representation)
                            similarity = compute_similarity(next_state,
                                [p[-1] for _, p, g in beam if g == other_group])
                            diversity_score += similarity * diversity_penalty

                    final_score = base_score + diversity_score
                    all_candidates[group].append((final_score, new_path, group))

        # Select top candidates from each group
        beam = []
        for group_candidates in all_candidates:
            if group_candidates:
                beam.extend(heapq.nsmallest(group_size, group_candidates,
                                           key=lambda x: x[0]))

        if not beam:
            return []

    return min(beam, key=lambda x: x[0])[1] if beam else []

def compute_similarity(state, other_states):
    """Compute similarity score (application-specific)."""
    if not other_states:
        return 0
    # Example: For sequence generation, could be token overlap
    # For pathfinding, could be distance
    return len(other_states) * 0.1  # Placeholder
```

#### 2. Stochastic Beam Search

Adds randomness to candidate selection for better exploration:

```python
import random

def stochastic_beam_search(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    temperature: float = 1.0,
    max_depth: int = 100
) -> List:
    """
    Stochastic beam search samples candidates probabilistically
    based on their scores rather than always taking the top-k.

    temperature: Controls randomness (lower = more greedy)
    """
    beam = [(heuristic(initial_state), [initial_state])]

    for depth in range(max_depth):
        for score, path in beam:
            if goal_test(path[-1]):
                return path

        candidates = []
        for score, path in beam:
            current_state = path[-1]

            for next_state, cost in get_neighbors(current_state):
                if next_state not in path:
                    new_path = path + [next_state]
                    new_score = heuristic(next_state)
                    candidates.append((new_score, new_path))

        if not candidates:
            return []

        # Convert scores to probabilities using softmax
        scores = [score for score, _ in candidates]
        # Apply temperature and compute probabilities
        exp_scores = [math.exp(-score / temperature) for score in scores]
        total = sum(exp_scores)
        probabilities = [exp_score / total for exp_score in exp_scores]

        # Sample beam_width candidates
        selected_indices = random.choices(
            range(len(candidates)),
            weights=probabilities,
            k=min(beam_width, len(candidates))
        )
        beam = [candidates[i] for i in selected_indices]

    return min(beam, key=lambda x: x[0])[1] if beam else []
```

#### 3. Beam Search with Pruning

Prunes unpromising candidates based on threshold:

```python
def beam_search_with_pruning(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    pruning_threshold: float = 2.0,
    max_depth: int = 100
) -> List:
    """
    Beam search with pruning removes candidates that are significantly
    worse than the best candidate.

    pruning_threshold: Candidates with score > best_score * threshold are pruned
    """
    beam = [(heuristic(initial_state), [initial_state])]

    for depth in range(max_depth):
        for score, path in beam:
            if goal_test(path[-1]):
                return path

        candidates = []
        for score, path in beam:
            current_state = path[-1]

            for next_state, cost in get_neighbors(current_state):
                if next_state not in path:
                    new_path = path + [next_state]
                    new_score = heuristic(next_state)
                    candidates.append((new_score, new_path))

        if not candidates:
            return []

        # Find best score
        best_score = min(score for score, _ in candidates)

        # Prune candidates worse than threshold
        pruned = [(score, path) for score, path in candidates
                  if score <= best_score * pruning_threshold]

        # Keep top beam_width from pruned candidates
        beam = heapq.nsmallest(beam_width, pruned, key=lambda x: x[0])

    return min(beam, key=lambda x: x[0])[1] if beam else []
```

### Practical Applications

#### Sequence Generation (NLP)

```python
def beam_search_sequence_generation(
    model,
    start_token: str,
    end_token: str,
    vocab: List[str],
    beam_width: int = 5,
    max_length: int = 50
) -> List[str]:
    """
    Beam search for sequence generation in NLP.
    Commonly used in machine translation, text generation, etc.
    """
    # Each candidate: (log_prob, sequence)
    beam = [(0.0, [start_token])]

    for _ in range(max_length):
        candidates = []

        for log_prob, sequence in beam:
            # If sequence ended, keep it as is
            if sequence[-1] == end_token:
                candidates.append((log_prob, sequence))
                continue

            # Get probability distribution for next token
            probs = model.predict_next(sequence)  # Returns dict {token: prob}

            # Expand with each possible next token
            for token, prob in probs.items():
                new_sequence = sequence + [token]
                # Use log probabilities to avoid numerical underflow
                new_log_prob = log_prob + math.log(prob + 1e-10)
                candidates.append((new_log_prob, new_sequence))

        # Keep top beam_width candidates
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

        # Check if all beams ended
        if all(seq[-1] == end_token for _, seq in beam):
            break

    # Return best sequence
    return max(beam, key=lambda x: x[0])[1]


# Example usage with simple n-gram model
class SimpleNGramModel:
    def __init__(self):
        # Simplified model for demonstration
        self.vocab = ["the", "cat", "sat", "on", "mat", "<END>"]

    def predict_next(self, sequence):
        """Returns probability distribution over next tokens."""
        last_word = sequence[-1]
        # Simplified probabilities based on last word
        if last_word == "the":
            return {"cat": 0.7, "mat": 0.2, "sat": 0.1}
        elif last_word == "cat":
            return {"sat": 0.8, "on": 0.2}
        elif last_word == "sat":
            return {"on": 0.9, "<END>": 0.1}
        elif last_word == "on":
            return {"the": 0.7, "mat": 0.3}
        elif last_word == "mat":
            return {"<END>": 1.0}
        return {"<END>": 1.0}

# Usage
model = SimpleNGramModel()
result = beam_search_sequence_generation(
    model,
    start_token="<START>",
    end_token="<END>",
    vocab=model.vocab,
    beam_width=3
)
```

**When to use**:
- Large search spaces where complete search is impossible
- NLP tasks: machine translation, text generation, speech recognition
- When memory is constrained (vs. keeping all possibilities)
- Problems where good approximate solutions are acceptable
- Sequence generation with probabilistic models

**Pros**:
- Memory efficient compared to breadth-first search
- Often finds good solutions quickly
- Configurable trade-off between speed and quality (beam width)
- Works well with neural networks and probabilistic models

**Cons**:
- Not complete (may miss optimal solution)
- Not optimal (no guarantee of best solution)
- Sensitive to beam width parameter
- Can get stuck in local optima
- May suffer from lack of diversity

### Beam Width Selection Guidelines

```python
def adaptive_beam_width(depth: int, base_width: int = 3, max_width: int = 10) -> int:
    """
    Dynamically adjust beam width based on search depth.
    Increase width at deeper levels to improve exploration.
    """
    return min(base_width + depth // 10, max_width)

# Empirical guidelines:
# - Small problems (< 100 states): beam_width = 2-5
# - Medium problems: beam_width = 5-20
# - Large problems (NLP): beam_width = 10-100
# - Trade-off: Larger width = better quality but slower
```

## A* Search

**Time**: $O(b^d)$ worst case | **Space**: $O(b^d)$ | **Use Case**: Pathfinding, puzzle solving

A* is an informed search algorithm that finds the optimal path using both actual cost $g(n)$ and heuristic estimate $h(n)$. With an admissible heuristic, A* is guaranteed to find the optimal solution.

### Core Implementation

```python
import heapq
from typing import Dict, List, Tuple, Callable

def a_star_search(
    start,
    goal,
    get_neighbors: Callable,
    heuristic: Callable,
    cost_function: Callable = lambda x, y: 1
) -> Tuple[List, float]:
    """
    A* search algorithm.

    Args:
        start: Starting state
        goal: Goal state
        get_neighbors: Function returning neighbors of a state
        heuristic: Admissible heuristic function h(state)
        cost_function: Cost between two adjacent states

    Returns:
        Tuple of (path, total_cost)
    """
    # Priority queue: (f_score, state, path, g_score)
    frontier = [(heuristic(start), start, [start], 0)]
    explored = set()

    # Best g_score for each state
    g_scores = {start: 0}

    while frontier:
        f_score, current, path, g_score = heapq.heappop(frontier)

        if current == goal:
            return path, g_score

        if current in explored:
            continue

        explored.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in explored:
                continue

            # Calculate tentative g_score
            tentative_g = g_score + cost_function(current, neighbor)

            # Skip if we've found a better path to neighbor
            if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                continue

            # This is the best path to neighbor so far
            g_scores[neighbor] = tentative_g
            h_score = heuristic(neighbor)
            f_score = tentative_g + h_score

            new_path = path + [neighbor]
            heapq.heappush(frontier, (f_score, neighbor, new_path, tentative_g))

    return [], float('inf')  # No path found


# Example: Grid pathfinding with obstacles
def grid_a_star(grid: List[List[int]], start: Tuple, goal: Tuple):
    """A* for grid pathfinding."""

    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(grid) and
                0 <= ny < len(grid[0]) and
                grid[nx][ny] == 0):  # 0 = walkable
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(pos):
        # Manhattan distance
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    return a_star_search(start, goal, get_neighbors, heuristic)
```

### Common Heuristics

```python
# Manhattan distance (4-directional movement)
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Euclidean distance (any direction)
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Chebyshev distance (8-directional movement)
def chebyshev_distance(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

# Diagonal distance (8-directional with different costs)
def diagonal_distance(pos1, pos2, D=1, D2=1.414):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
```

**When to use**:
- Need optimal path with performance better than Dijkstra
- Have good admissible heuristic
- Complete search is required
- Memory is available

**Pros**:
- Optimal (with admissible heuristic)
- Complete
- Faster than Dijkstra with good heuristic

**Cons**:
- High memory usage
- Requires admissible heuristic for optimality
- Can be slow for large state spaces

## Greedy Best-First Search

**Time**: $O(b^m)$ | **Space**: $O(b^m)$ | **Use Case**: Fast approximate solutions

Greedy best-first search expands nodes with the best heuristic value, ignoring path cost. It's fast but not optimal or complete.

```python
def greedy_best_first_search(
    start,
    goal,
    get_neighbors: Callable,
    heuristic: Callable
) -> List:
    """
    Greedy best-first search using only heuristic function.
    Fast but not guaranteed to find optimal solution.
    """
    # Priority queue: (h_score, state, path)
    frontier = [(heuristic(start), start, [start])]
    explored = set()

    while frontier:
        h_score, current, path = heapq.heappop(frontier)

        if current == goal:
            return path

        if current in explored:
            continue

        explored.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in explored:
                new_path = path + [neighbor]
                heapq.heappush(frontier,
                              (heuristic(neighbor), neighbor, new_path))

    return []  # No path found
```

**When to use**: Speed matters more than optimality, good heuristic available

**Pros**: Fast, simple, low memory
**Cons**: Not optimal, not complete, can get stuck

## IDA* (Iterative Deepening A*)

**Time**: $O(b^d)$ | **Space**: $O(d)$ | **Use Case**: Memory-constrained optimal search

IDA* combines benefits of A* (optimality) with iterative deepening (low memory).

```python
def ida_star_search(
    start,
    goal,
    get_neighbors: Callable,
    heuristic: Callable,
    cost_function: Callable = lambda x, y: 1
) -> Tuple[List, float]:
    """
    Iterative Deepening A* - memory-efficient optimal search.
    """

    def search(path, g_cost, threshold):
        """Recursive DFS with f-cost threshold."""
        current = path[-1]
        f_cost = g_cost + heuristic(current)

        if f_cost > threshold:
            return f_cost, []

        if current == goal:
            return 0, path

        min_threshold = float('inf')

        for neighbor in get_neighbors(current):
            if neighbor not in path:  # Avoid cycles
                new_cost = g_cost + cost_function(current, neighbor)
                new_path = path + [neighbor]

                result_threshold, result_path = search(new_path, new_cost, threshold)

                if result_path:  # Found goal
                    return result_threshold, result_path

                min_threshold = min(min_threshold, result_threshold)

        return min_threshold, []

    # Start with heuristic value as threshold
    threshold = heuristic(start)

    while threshold < float('inf'):
        threshold, path = search([start], 0, threshold)
        if path:
            # Calculate total cost
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += cost_function(path[i], path[i+1])
            return path, total_cost

    return [], float('inf')
```

**When to use**: Need optimal solution with limited memory

**Pros**: Optimal, complete, low memory
**Cons**: Can revisit states, slower than A*

## Hill Climbing

**Time**: $O(n \cdot m)$ where $n$ = iterations, $m$ = neighbors | **Space**: $O(1)$ | **Use Case**: Local optimization

Hill climbing is a local search algorithm that continually moves toward increasing value (or decreasing cost).

```python
def hill_climbing(
    initial_state,
    get_neighbors: Callable,
    evaluate: Callable,  # Higher is better
    max_iterations: int = 1000
):
    """
    Simple hill climbing - moves to best neighbor.
    Can get stuck in local maxima.
    """
    current = initial_state
    current_value = evaluate(current)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)

        if not neighbors:
            break

        # Find best neighbor
        best_neighbor = max(neighbors, key=evaluate)
        best_value = evaluate(best_neighbor)

        # If no improvement, stop (local maximum reached)
        if best_value <= current_value:
            break

        current = best_neighbor
        current_value = best_value

    return current, current_value


def stochastic_hill_climbing(
    initial_state,
    get_neighbors: Callable,
    evaluate: Callable,
    max_iterations: int = 1000
):
    """
    Stochastic hill climbing - randomly selects uphill move.
    Can escape some local maxima.
    """
    current = initial_state
    current_value = evaluate(current)

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)

        # Filter neighbors that improve the solution
        better_neighbors = [n for n in neighbors if evaluate(n) > current_value]

        if not better_neighbors:
            break

        # Randomly select an improving neighbor
        current = random.choice(better_neighbors)
        current_value = evaluate(current)

    return current, current_value


def random_restart_hill_climbing(
    generate_random_state: Callable,
    get_neighbors: Callable,
    evaluate: Callable,
    num_restarts: int = 10,
    max_iterations: int = 1000
):
    """
    Hill climbing with random restarts to escape local maxima.
    """
    best_state = None
    best_value = float('-inf')

    for _ in range(num_restarts):
        initial = generate_random_state()
        state, value = hill_climbing(initial, get_neighbors, evaluate, max_iterations)

        if value > best_value:
            best_state = state
            best_value = value

    return best_state, best_value
```

**When to use**: Local optimization, continuous search spaces, quick approximate solutions

**Pros**: Simple, fast, low memory
**Cons**: Gets stuck in local maxima, not complete, not optimal

## Simulated Annealing

**Time**: $O(n \cdot m)$ | **Space**: $O(1)$ | **Use Case**: Global optimization with acceptance of worse moves

Simulated annealing allows occasional moves to worse states to escape local optima, with probability decreasing over time.

```python
import math
import random

def simulated_annealing(
    initial_state,
    get_neighbor: Callable,  # Generate one random neighbor
    evaluate: Callable,  # Lower is better (cost function)
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    min_temp: float = 0.01,
    max_iterations: int = 10000
):
    """
    Simulated annealing for optimization.
    Accepts worse solutions with probability that decreases over time.

    Args:
        initial_temp: Starting temperature
        cooling_rate: Temperature multiplier each iteration (0 < rate < 1)
        min_temp: Stop when temperature reaches this value
    """
    current = initial_state
    current_cost = evaluate(current)
    best = current
    best_cost = current_cost

    temp = initial_temp
    iteration = 0

    while temp > min_temp and iteration < max_iterations:
        # Generate random neighbor
        neighbor = get_neighbor(current)
        neighbor_cost = evaluate(neighbor)

        # Calculate cost difference
        delta = neighbor_cost - current_cost

        # Accept if better, or with probability based on temperature
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost

            # Update best solution found
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        # Cool down
        temp *= cooling_rate
        iteration += 1

    return best, best_cost


# Example: Traveling Salesman Problem
def tsp_simulated_annealing(cities, distances):
    """
    Solve TSP using simulated annealing.
    cities: List of city indices
    distances: 2D matrix of distances
    """
    def evaluate(route):
        """Calculate total distance of route."""
        total = 0
        for i in range(len(route)):
            total += distances[route[i]][route[(i+1) % len(route)]]
        return total

    def get_neighbor(route):
        """Generate neighbor by swapping two cities."""
        neighbor = route[:]
        i, j = random.sample(range(len(route)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    initial_route = list(range(len(cities)))
    random.shuffle(initial_route)

    return simulated_annealing(
        initial_route,
        get_neighbor,
        evaluate,
        initial_temp=1000.0,
        cooling_rate=0.995,
        min_temp=1.0
    )
```

**When to use**: Combinatorial optimization, avoiding local optima, when near-optimal solutions are acceptable

**Pros**: Can escape local optima, probabilistically complete, simple to implement
**Cons**: Sensitive to parameters, no guarantee of optimality, slower than greedy methods

## Comparison

| Algorithm | Time | Space | Optimal | Complete | Memory Use | Best For |
|-----------|------|-------|---------|----------|------------|----------|
| Beam Search | $O(w \cdot b \cdot d)$ | $O(w \cdot d)$ | ❌ | ❌ | Low | NLP, constrained search |
| A* | $O(b^d)$ | $O(b^d)$ | ✅* | ✅* | High | Optimal pathfinding |
| Greedy Best-First | $O(b^m)$ | $O(b^m)$ | ❌ | ❌ | Medium | Fast approximate paths |
| IDA* | $O(b^d)$ | $O(d)$ | ✅* | ✅* | Very Low | Memory-constrained optimal |
| Hill Climbing | $O(n \cdot m)$ | $O(1)$ | ❌ | ❌ | Minimal | Local optimization |
| Simulated Annealing | $O(n \cdot m)$ | $O(1)$ | ❌ | ~✅ | Minimal | Global optimization |

\* With admissible heuristic and finite branching factor

**Legend:**
- $w$ = beam width, $b$ = branching factor, $d$ = depth, $m$ = maximum depth, $n$ = iterations

## Decision Tree

```
What's your primary constraint?
├─ Memory → Is optimal solution required?
│   ├─ Yes → IDA*
│   └─ No → Beam Search or Hill Climbing
├─ Time (need fast solution) → Is optimality important?
│   ├─ Yes → A* with good heuristic
│   └─ No → Greedy Best-First or Beam Search
├─ Solution Quality → Need guaranteed optimal?
│   ├─ Yes → A* (if memory allows) or IDA*
│   └─ No → Do you have discrete states or continuous?
│       ├─ Discrete → Beam Search
│       └─ Continuous → Simulated Annealing or Hill Climbing
└─ Problem Type?
    ├─ Sequence Generation (NLP) → Beam Search
    ├─ Pathfinding → A* or IDA*
    ├─ Combinatorial Optimization → Simulated Annealing
    └─ Local Optimization → Hill Climbing
```

## Common Patterns and Applications

### Pattern: Early Stopping

```python
def beam_search_with_early_stopping(
    initial_state,
    goal_test: Callable,
    get_neighbors: Callable,
    heuristic: Callable,
    beam_width: int = 3,
    patience: int = 5,  # Stop if no improvement for N iterations
    max_depth: int = 100
) -> List:
    """Beam search with early stopping when no improvement."""
    beam = [(heuristic(initial_state), [initial_state])]
    best_score = float('inf')
    no_improvement_count = 0

    for depth in range(max_depth):
        for score, path in beam:
            if goal_test(path[-1]):
                return path

        current_best = min(score for score, _ in beam)

        if current_best < best_score:
            best_score = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                # No improvement, return best found
                return min(beam, key=lambda x: x[0])[1]

        # Standard beam search expansion...
        candidates = []
        for score, path in beam:
            for next_state, _ in get_neighbors(path[-1]):
                if next_state not in path:
                    new_path = path + [next_state]
                    candidates.append((heuristic(next_state), new_path))

        if not candidates:
            break

        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    return min(beam, key=lambda x: x[0])[1] if beam else []
```

### Real-World Applications

**1. Neural Machine Translation**
```python
# Beam search for translating sentences
def translate_with_beam_search(source_sentence, model, beam_width=5):
    """
    Translate using beam search over decoder outputs.
    model: Neural translation model
    """
    # Encode source
    encoder_output = model.encode(source_sentence)

    # Beam search through decoder
    beam = [(0.0, ["<START>"])]

    for step in range(model.max_length):
        candidates = []

        for log_prob, sequence in beam:
            if sequence[-1] == "<END>":
                candidates.append((log_prob, sequence))
                continue

            # Get next word probabilities from decoder
            decoder_output = model.decode(sequence, encoder_output)

            for word, prob in decoder_output.items():
                new_seq = sequence + [word]
                new_log_prob = log_prob + math.log(prob + 1e-10)
                candidates.append((new_log_prob, new_seq))

        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

        if all(seq[-1] == "<END>" for _, seq in beam):
            break

    return max(beam, key=lambda x: x[0])[1]
```

**2. Game AI - Pathfinding**
```python
# A* for game character navigation
def game_pathfinding(game_map, character_pos, target_pos, character_abilities):
    """
    Find path considering character abilities (jumping, swimming, etc.)
    """
    def get_neighbors(pos):
        neighbors = []
        x, y = pos

        # Normal movement
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_pos = (x+dx, y+dy)
            if is_walkable(game_map, new_pos, character_abilities):
                neighbors.append(new_pos)

        # Special moves (jumping over obstacles, etc.)
        if character_abilities.can_jump:
            for dx, dy in [(0,2), (2,0), (0,-2), (-2,0)]:
                new_pos = (x+dx, y+dy)
                if is_valid_jump(game_map, pos, new_pos):
                    neighbors.append(new_pos)

        return neighbors

    def heuristic(pos):
        return manhattan_distance(pos, target_pos)

    path, cost = a_star_search(character_pos, target_pos, get_neighbors, heuristic)
    return path
```

**3. Resource Allocation**
```python
# Simulated annealing for resource allocation
def optimize_resource_allocation(resources, demands, constraints):
    """
    Optimize allocation of limited resources to maximize satisfaction.
    """
    def evaluate(allocation):
        """Calculate satisfaction score (higher is better)."""
        satisfaction = 0
        for i, alloc in enumerate(allocation):
            satisfaction += utility_function(demands[i], alloc)

        # Penalize constraint violations
        if sum(allocation) > resources:
            satisfaction -= 1000 * (sum(allocation) - resources)

        return -satisfaction  # Negate for minimization

    def get_neighbor(allocation):
        """Randomly adjust allocation."""
        neighbor = allocation[:]
        i, j = random.sample(range(len(allocation)), 2)
        transfer = random.uniform(0, neighbor[i])
        neighbor[i] -= transfer
        neighbor[j] += transfer
        return neighbor

    # Initial uniform allocation
    initial = [resources / len(demands)] * len(demands)

    best_allocation, _ = simulated_annealing(
        initial, get_neighbor, evaluate,
        initial_temp=100, cooling_rate=0.99
    )

    return best_allocation
```

## Key Takeaways

1. **Beam Search**: Memory-efficient but incomplete - ideal for NLP and constrained search spaces
2. **A***: Optimal and complete with good heuristic - best for pathfinding when memory allows
3. **IDA***: Optimal with minimal memory - use when memory is critically constrained
4. **Greedy Best-First**: Fastest but least reliable - quick approximate solutions only
5. **Hill Climbing**: Simple local search - good for continuous optimization
6. **Simulated Annealing**: Better global search - escapes local optima with probability

**Critical Parameters:**
- **Beam Width**: Larger = better quality but slower and more memory
- **Heuristic Quality**: Better heuristic = faster search and better results
- **Temperature Schedule**: Controls exploration vs. exploitation in simulated annealing

## Performance Tips

1. **Heuristic Design**: Ensure heuristic is admissible for A*/IDA* optimality
2. **Beam Width Tuning**: Start with width = 3-5, increase if quality insufficient
3. **Diversity**: Use diverse beam search when beam candidates are too similar
4. **Early Stopping**: Monitor improvement and stop when convergence detected
5. **Hybrid Approaches**: Combine methods (e.g., beam search with A* heuristic)
6. **State Caching**: Cache computed states to avoid redundant evaluations
7. **Pruning**: Remove clearly suboptimal candidates early

```python
# Example: Combining beam search with A* evaluation
def hybrid_beam_a_star(initial, goal, neighbors, heuristic, beam_width=5):
    """Use A* evaluation (g + h) in beam search."""
    beam = [(heuristic(initial), 0, [initial])]

    for depth in range(100):
        candidates = []
        for f_score, g_score, path in beam:
            if path[-1] == goal:
                return path

            for next_state, cost in neighbors(path[-1]):
                if next_state not in path:
                    new_g = g_score + cost
                    new_h = heuristic(next_state)
                    new_f = new_g + new_h
                    candidates.append((new_f, new_g, path + [next_state]))

        if not candidates:
            break

        beam = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])

    return []
```

## ELI10

Imagine you're trying to find the best path through a maze:

- **Beam Search**: You can only remember your 3 favorite paths. At each step, you try extending each path and keep only the 3 best new paths. Fast but might miss the perfect path!

- **A\* Search**: You remember all paths and always extend the one that seems best (considering both how far you've gone and how far is left). Finds the perfect path but remembers everything!

- **Greedy Best-First**: Always go toward the goal, ignoring how far you've walked. Fast but can walk into dead ends!

- **Hill Climbing**: Always walk uphill. Great if you're on the right mountain, but you might be climbing a small hill while the big mountain is next door!

- **Simulated Annealing**: Like hill climbing, but sometimes you randomly walk downhill to explore. You do this less over time until you only walk uphill. Finds bigger mountains!

## Further Resources

- [A* Pathfinding for Beginners](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html)
- [Beam Search Explained](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24)
- [Heuristic Search - Stanford CS221](https://stanford-cs221.github.io/autumn2019/modules/)
- [Red Blob Games - Pathfinding](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [Simulated Annealing Tutorial](https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7)
- [LeetCode Heuristic Search Problems](https://leetcode.com/tag/heuristic/)

### Related Algorithm Files
- See `graph_algorithms.md` for BFS, DFS, Dijkstra
- See `dynamic_programming.md` for optimal substructure problems
- See `greedy_algorithms.md` for greedy strategies
