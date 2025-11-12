# Dynamic Programming

## Overview

Dynamic Programming (DP) solves complex problems by breaking them into simpler subproblems, solving each once, and storing results to avoid recomputation. Essential for optimization problems.

## Key Concepts

**Optimal Substructure**: Optimal solution built from optimal solutions of subproblems

**Overlapping Subproblems**: Same subproblem computed multiple times

## Approaches

### Memoization (Top-Down)
```python
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

### Tabulation (Bottom-Up)
```python
def fib(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

## Classic Problems

### Coin Change
```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Knapsack
```python
def knapsack(weights, values, capacity):
    dp = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    for i in range(1, len(weights) + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w - weights[i-1]],
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
    return dp[len(weights)][capacity]
```

### Longest Common Subsequence
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### Edit Distance
```python
def editDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

## Patterns

1D DP: `dp[i] = f(dp[i-1], dp[i-2], ...)`
2D DP: `dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)`

## Complexity

| Problem | Naive | DP |
|---------|-------|-----|
| Fibonacci | O(2^n) | O(n) |
| Coin Change | O(n^m) | O(n*m) |
| Knapsack | O(2^n) | O(n*w) |

## ELI10

DP is like climbing stairs - memorize steps you've already calculated instead of redoing them!

## Further Resources

- [LeetCode DP Problems](https://leetcode.com/tag/dynamic-programming/)
- [Key Concepts

- **Overlapping Subproblems**: Dynamic programming is applicable when the problem can be broken down into smaller, overlapping subproblems that can be solved independently. The results of these subproblems are stored to avoid redundant calculations.

- **Optimal Substructure**: A problem exhibits optimal substructure if an optimal solution to the problem can be constructed from optimal solutions to its subproblems. This property allows dynamic programming to build up solutions incrementally.

## Techniques

1. **Top-Down Approach (Memoization)**: This approach involves solving the problem recursively and storing the results of subproblems in a table (or cache) to avoid redundant calculations. When a subproblem is encountered again, the stored result is used instead of recalculating it.

2. **Bottom-Up Approach (Tabulation)**: In this approach, the problem is solved iteratively by filling up a table based on previously computed values. This method typically starts with the smallest subproblems and builds up to the solution of the original problem.

## Applications

Dynamic programming is widely used in various applications, including:

- **Fibonacci Sequence**: Calculating Fibonacci numbers can be optimized using dynamic programming to avoid exponential time complexity.

- **Knapsack Problem**: The 0/1 knapsack problem can be efficiently solved using dynamic programming techniques to maximize the total value of items that can be carried.

- **Longest Common Subsequence**: Finding the longest common subsequence between two strings can be accomplished using dynamic programming to build a solution based on previously computed subsequences.

## Conclusion

Dynamic programming is a crucial technique in algorithm design that enables efficient solutions to problems with overlapping subproblems and optimal substructure. By leveraging memoization or tabulation, developers can significantly improve the performance of their algorithms, making dynamic programming an essential tool in computer science and software engineering.
