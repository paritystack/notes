# Dynamic Programming

## Overview

Dynamic Programming (DP) is a powerful algorithmic paradigm that solves complex optimization problems by breaking them down into simpler overlapping subproblems. Instead of solving the same subproblem multiple times, DP stores the results of subproblems and reuses them, dramatically improving efficiency from exponential to polynomial time complexity.

DP is essential for solving optimization problems where you need to find the best solution among many possibilities, counting problems where you need to count all possible ways to do something, and decision-making problems with multiple stages.

## Core Principles

### Optimal Substructure

A problem exhibits **optimal substructure** if an optimal solution can be constructed from optimal solutions of its subproblems. This is the foundation that allows DP to work.

**Example**: In the shortest path problem, if the shortest path from A to C goes through B, then the path from A to B must also be the shortest path between those two points.

**How to identify**:
1. Try to express the solution in terms of solutions to smaller instances
2. Verify that combining optimal solutions to subproblems gives an optimal solution to the original problem
3. If greedy choices work, you might not need DP (use greedy algorithm instead)

### Overlapping Subproblems

A problem has **overlapping subproblems** if the same subproblems are solved multiple times during the computation.

**Example**: Computing Fibonacci(5) recursively computes Fibonacci(3) twice, Fibonacci(2) three times, and Fibonacci(1) five times.

**Key insight**: Without DP, exponential time complexity. With DP, polynomial time complexity.

## DP Approaches

### 1. Memoization (Top-Down)

Memoization uses recursion with caching. Start with the original problem and recursively solve subproblems, storing results in a cache (usually a hash map or array).

**Advantages**:
- Intuitive and easier to implement for complex problems
- Only computes subproblems that are actually needed
- Natural fit for problems with recursive structure

**Disadvantages**:
- Recursion overhead (stack space)
- Potential stack overflow for deep recursion

```python
def fib_memo(n, memo=None):
    """Calculate nth Fibonacci number using memoization"""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    # Base cases
    if n <= 1:
        return n

    # Recursive case with memoization
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Time: O(n), Space: O(n)
```

### 2. Tabulation (Bottom-Up)

Tabulation builds up the solution iteratively, starting from the smallest subproblems and working up to the final solution.

**Advantages**:
- No recursion overhead
- Usually faster in practice
- Easier to optimize space complexity

**Disadvantages**:
- May compute unnecessary subproblems
- Can be less intuitive for complex problems

```python
def fib_tab(n):
    """Calculate nth Fibonacci number using tabulation"""
    if n <= 1:
        return n

    # Build table bottom-up
    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Time: O(n), Space: O(n)
```

### 3. Space-Optimized Tabulation

For many DP problems, you only need the last few states, not the entire table.

```python
def fib_optimized(n):
    """Space-optimized Fibonacci calculation"""
    if n <= 1:
        return n

    prev2, prev1 = 0, 1

    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1

# Time: O(n), Space: O(1)
```

## Problem Classification by Pattern

### Pattern 1: Linear Sequence DP

**Characteristics**: `dp[i]` depends on previous states `dp[i-1]`, `dp[i-2]`, etc.

**Template**:
```python
dp[i] = f(dp[i-1], dp[i-2], ..., dp[i-k])
```

#### Climbing Stairs

**Problem**: Count ways to climb n stairs (1 or 2 steps at a time).

```python
def climb_stairs(n):
    """Count distinct ways to climb n stairs"""
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Time: O(n), Space: O(n)
# Can be optimized to O(1) space
```

#### House Robber

**Problem**: Rob houses to maximize money without robbing adjacent houses.

```python
def rob(nums):
    """Maximum money you can rob without adjacent houses"""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    # dp[i] = max money robbing houses 0..i
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        # Either rob current house + dp[i-2], or skip it
        dp[i] = max(dp[i-1], nums[i] + dp[i-2])

    return dp[-1]

# Time: O(n), Space: O(n)
```

**Space-optimized version**:
```python
def rob_optimized(nums):
    if not nums:
        return 0

    prev2, prev1 = 0, 0

    for num in nums:
        current = max(prev1, num + prev2)
        prev2, prev1 = prev1, current

    return prev1

# Time: O(n), Space: O(1)
```

#### Longest Increasing Subsequence

**Problem**: Find length of longest strictly increasing subsequence.

```python
def length_of_LIS(nums):
    """Length of longest increasing subsequence"""
    if not nums:
        return 0

    n = len(nums)
    # dp[i] = length of LIS ending at index i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Time: O(n²), Space: O(n)
# Can be optimized to O(n log n) using binary search
```

### Pattern 2: Grid/Matrix DP

**Characteristics**: `dp[i][j]` depends on neighbors in 2D space.

**Template**:
```python
dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1], ...)
```

#### Unique Paths

**Problem**: Count paths from top-left to bottom-right (only right/down moves).

```python
def unique_paths(m, n):
    """Count unique paths in m×n grid"""
    # dp[i][j] = number of paths to reach cell (i,j)
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# Time: O(m×n), Space: O(m×n)
```

**Space-optimized**:
```python
def unique_paths_optimized(m, n):
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]

    return dp[n-1]

# Time: O(m×n), Space: O(n)
```

#### Minimum Path Sum

**Problem**: Find minimum sum path from top-left to bottom-right.

```python
def min_path_sum(grid):
    """Minimum path sum in grid"""
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # Initialize first cell
    dp[0][0] = grid[0][0]

    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    # Fill rest of table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]

# Time: O(m×n), Space: O(m×n)
```

#### Maximal Square

**Problem**: Find largest square containing only 1s in binary matrix.

```python
def maximal_square(matrix):
    """Find area of largest square of 1s"""
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    # Square side length at (i,j)
                    dp[i][j] = min(
                        dp[i-1][j],      # top
                        dp[i][j-1],      # left
                        dp[i-1][j-1]     # diagonal
                    ) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side

# Time: O(m×n), Space: O(m×n)
```

### Pattern 3: String DP

**Characteristics**: Problems involving sequences, subsequences, or substring operations.

#### Longest Common Subsequence (LCS)

**Problem**: Find length of longest subsequence common to both strings.

```python
def longest_common_subsequence(text1, text2):
    """Length of longest common subsequence"""
    m, n = len(text1), len(text2)
    # dp[i][j] = LCS length of text1[0..i-1] and text2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # Characters match: extend LCS
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # Take max of excluding one character
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

**Reconstructing the LCS**:
```python
def get_lcs(text1, text2):
    """Get actual LCS string"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))
```

#### Edit Distance (Levenshtein Distance)

**Problem**: Minimum operations to convert word1 to word2 (insert, delete, replace).

```python
def min_distance(word1, word2):
    """Minimum edit distance between two words"""
    m, n = len(word1), len(word2)
    # dp[i][j] = min operations to convert word1[0..i-1] to word2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: converting to/from empty string
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                # No operation needed
                dp[i][j] = dp[i-1][j-1]
            else:
                # Min of: replace, delete, insert
                dp[i][j] = 1 + min(
                    dp[i-1][j-1],  # Replace
                    dp[i-1][j],    # Delete from word1
                    dp[i][j-1]     # Insert to word1
                )

    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

#### Longest Palindromic Subsequence

**Problem**: Find length of longest palindromic subsequence.

```python
def longest_palindrome_subseq(s):
    """Length of longest palindromic subsequence"""
    n = len(s)
    # dp[i][j] = length of LPS in s[i..j]
    dp = [[0] * n for _ in range(n)]

    # Every single character is a palindrome of length 1
    for i in range(n):
        dp[i][i] = 1

    # Build table for substrings of increasing length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                # Characters match: add 2 to inner subsequence
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                # Take max of excluding one end
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]

# Time: O(n²), Space: O(n²)
```

#### Word Break

**Problem**: Check if string can be segmented into dictionary words.

```python
def word_break(s, wordDict):
    """Check if string can be segmented into words"""
    word_set = set(wordDict)
    n = len(s)
    # dp[i] = True if s[0..i-1] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string

    for i in range(1, n + 1):
        for j in range(i):
            # If s[0..j-1] can be segmented and s[j..i-1] is a word
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]

# Time: O(n² × m) where m is average word length
# Space: O(n)
```

### Pattern 4: Knapsack Problems

**Characteristics**: Selection problems with capacity constraints.

#### 0/1 Knapsack

**Problem**: Maximize value with weight constraint (each item used at most once).

```python
def knapsack_01(weights, values, capacity):
    """0/1 Knapsack: maximize value within capacity"""
    n = len(weights)
    # dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't include item i-1
            dp[i][w] = dp[i-1][w]

            # Include item i-1 if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    values[i-1] + dp[i-1][w - weights[i-1]]
                )

    return dp[n][capacity]

# Time: O(n×W), Space: O(n×W)
```

**Space-optimized 0/1 Knapsack**:
```python
def knapsack_01_optimized(weights, values, capacity):
    """Space-optimized 0/1 knapsack"""
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

    return dp[capacity]

# Time: O(n×W), Space: O(W)
```

#### Unbounded Knapsack

**Problem**: Same as 0/1 but can use each item unlimited times.

```python
def knapsack_unbounded(weights, values, capacity):
    """Unbounded knapsack: items can be used multiple times"""
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], values[i] + dp[w - weights[i]])

    return dp[capacity]

# Time: O(n×W), Space: O(W)
```

#### Coin Change (Minimum Coins)

**Problem**: Minimum coins needed to make amount.

```python
def coin_change(coins, amount):
    """Minimum coins to make amount"""
    # dp[i] = min coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Time: O(amount × len(coins)), Space: O(amount)
```

#### Coin Change (Count Ways)

**Problem**: Count ways to make amount with given coins.

```python
def coin_change_ways(coins, amount):
    """Count ways to make amount"""
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

# Time: O(amount × len(coins)), Space: O(amount)
```

### Pattern 5: Partition DP

**Characteristics**: Dividing array/string into parts to optimize some property.

#### Partition Equal Subset Sum

**Problem**: Check if array can be partitioned into two equal-sum subsets.

```python
def can_partition(nums):
    """Check if array can be partitioned into equal sum subsets"""
    total = sum(nums)
    if total % 2:
        return False

    target = total // 2
    # dp[i] = True if sum i is achievable
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        # Traverse backwards (0/1 knapsack pattern)
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]

    return dp[target]

# Time: O(n × sum/2), Space: O(sum/2)
```

#### Palindrome Partitioning II

**Problem**: Minimum cuts needed to partition string into palindromes.

```python
def min_cut(s):
    """Minimum cuts for palindrome partitioning"""
    n = len(s)

    # Precompute palindrome check
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palindrome[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_palindrome[i][j] = (length == 2 or is_palindrome[i+1][j-1])

    # dp[i] = min cuts for s[0..i]
    dp = [0] * n
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            dp[i] = i  # Max cuts
            for j in range(i):
                if is_palindrome[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n-1]

# Time: O(n²), Space: O(n²)
```

### Pattern 6: State Machine DP

**Characteristics**: Problems with multiple states and transitions.

#### Best Time to Buy and Sell Stock with Cooldown

**Problem**: Max profit with cooldown after selling.

```python
def max_profit_cooldown(prices):
    """Max profit with cooldown"""
    if not prices:
        return 0

    n = len(prices)
    # States: hold stock, sold today, cooldown
    hold = [0] * n
    sold = [0] * n
    cooldown = [0] * n

    hold[0] = -prices[0]

    for i in range(1, n):
        # Hold: either already holding or buy today
        hold[i] = max(hold[i-1], cooldown[i-1] - prices[i])
        # Sold: sell today
        sold[i] = hold[i-1] + prices[i]
        # Cooldown: either already in cooldown or just sold
        cooldown[i] = max(cooldown[i-1], sold[i-1])

    return max(sold[-1], cooldown[-1])

# Time: O(n), Space: O(n)
```

### Pattern 7: Interval DP

**Characteristics**: Problems on ranges/intervals `[i, j]`.

#### Burst Balloons

**Problem**: Maximize coins from bursting balloons.

```python
def max_coins(nums):
    """Maximum coins from bursting balloons"""
    # Add 1s at boundaries
    nums = [1] + nums + [1]
    n = len(nums)
    # dp[i][j] = max coins bursting balloons (i, j) (exclusive)
    dp = [[0] * n for _ in range(n)]

    # Build for increasing interval lengths
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            # Try bursting each balloon k last in (left, right)
            for k in range(left + 1, right):
                coins = nums[left] * nums[k] * nums[right]
                coins += dp[left][k] + dp[k][right]
                dp[left][right] = max(dp[left][right], coins)

    return dp[0][n-1]

# Time: O(n³), Space: O(n²)
```

## Advanced Techniques

### State Space Reduction

Many DP problems can be optimized by reducing the state space:

1. **Eliminate redundant dimensions**: If a dimension can be computed from others, remove it
2. **Use modulo arithmetic**: For counting problems with large numbers
3. **Compress coordinates**: Map large ranges to smaller ones
4. **Rolling array**: Keep only last k rows/columns instead of entire table

### Bitmask DP

Use bitmasks to represent subsets when state involves combinations.

**Example: Traveling Salesman Problem**
```python
def tsp(dist):
    """Minimum cost to visit all cities (TSP)"""
    n = len(dist)
    # dp[mask][i] = min cost to visit cities in mask, ending at i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(
                    dp[new_mask][v],
                    dp[mask][u] + dist[u][v]
                )

    # Return to start
    return min(dp[(1 << n) - 1][i] + dist[i][0] for i in range(n))

# Time: O(2ⁿ × n²), Space: O(2ⁿ × n)
```

### Digit DP

Solve problems on ranges of numbers by processing digits.

**Example: Count numbers with property in range [L, R]**
```python
def count_digit_dp(n):
    """Count numbers up to n with some property"""
    s = str(n)
    memo = {}

    def dp(pos, tight, started):
        """
        pos: current digit position
        tight: whether we're bounded by n
        started: whether number has started (handle leading zeros)
        """
        if pos == len(s):
            return 1 if started else 0

        if (pos, tight, started) in memo:
            return memo[(pos, tight, started)]

        limit = int(s[pos]) if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            # Check property here
            new_tight = tight and (digit == limit)
            new_started = started or (digit != 0)
            result += dp(pos + 1, new_tight, new_started)

        memo[(pos, tight, started)] = result
        return result

    return dp(0, True, False)
```

### Tree DP

DP on trees, usually processing from leaves up.

**Example: Maximum independent set in tree**
```python
def tree_dp(graph, root):
    """Maximum independent set in tree"""
    # include[v] = max value including v
    # exclude[v] = max value excluding v
    include = {}
    exclude = {}

    def dfs(node, parent):
        include[node] = 1  # Value of node
        exclude[node] = 0

        for child in graph[node]:
            if child == parent:
                continue
            dfs(child, node)
            # If we include node, can't include children
            include[node] += exclude[child]
            # If we exclude node, take max of children
            exclude[node] += max(include[child], exclude[child])

    dfs(root, -1)
    return max(include[root], exclude[root])
```

## Complexity Analysis

### Time Complexity Patterns

| Pattern | Typical Complexity | Example |
|---------|-------------------|---------|
| 1D DP | O(n) to O(n²) | Fibonacci, House Robber |
| 2D DP | O(n²) to O(n³) | LCS, Edit Distance |
| Knapsack | O(n × W) | 0/1 Knapsack, Coin Change |
| Substring | O(n²) to O(n³) | Palindrome problems |
| Interval DP | O(n³) | Matrix chain, Burst balloons |
| Bitmask DP | O(2ⁿ × n) to O(2ⁿ × n²) | TSP, Subset problems |

### Space Complexity Optimization

1. **Rolling array**: O(n × m) → O(m) for many 2D DP problems
2. **In-place modification**: Use input array as DP table
3. **State elimination**: Remove redundant state dimensions

### Comparison: Naive vs DP

| Problem | Naive | DP | Improvement |
|---------|-------|-----|-------------|
| Fibonacci | O(2ⁿ) | O(n) | Exponential → Linear |
| LCS | O(2^(m+n)) | O(m×n) | Exponential → Polynomial |
| Knapsack | O(2ⁿ) | O(n×W) | Exponential → Pseudo-polynomial |
| Coin Change | O(S^n) | O(n×amount) | Exponential → Polynomial |

## Implementation Tips

### 1. Define the State

**Questions to ask**:
- What information is needed to solve subproblems?
- What's the minimum state needed (avoid redundancy)?
- Can I solve for state X using smaller states?

**Example**: For LCS, state is `(i, j)` representing position in both strings.

### 2. Define the Recurrence Relation

Express the solution in terms of smaller subproblems.

**Template**:
```
dp[current_state] = optimal_choice(
    dp[smaller_state_1],
    dp[smaller_state_2],
    ...
)
```

### 3. Identify Base Cases

What are the smallest subproblems you can solve directly?

**Example**:
- Empty string: `dp[0][...] = 0`
- Single element: `dp[i][i] = ...`

### 4. Determine Iteration Order

Ensure you compute smaller subproblems before larger ones.

**Patterns**:
- 1D: Iterate i from small to large
- 2D: Iterate i, then j (or by diagonal/length)
- Intervals: Iterate by increasing interval length

### 5. Initialize the DP Table

Set base cases and default values (0, infinity, false, etc.)

### 6. Implement and Optimize

Start with clear memoization, then optimize to tabulation and space reduction.

## Common Pitfalls

### 1. Wrong Base Cases

```python
# Wrong: doesn't handle n=0
def fib(n):
    dp = [0] * n
    # IndexError when n=0!

# Correct
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
```

### 2. Wrong Iteration Order

```python
# Wrong: using updated values in 0/1 knapsack
for item in items:
    for w in range(W + 1):  # Forward iteration
        dp[w] = max(dp[w], dp[w - weight] + value)

# Correct: backward iteration prevents using updated values
for item in items:
    for w in range(W, weight - 1, -1):
        dp[w] = max(dp[w], dp[w - weight] + value)
```

### 3. Off-by-One Errors

Be careful with array indices vs. problem indices.

```python
# dp[i] represents first i elements (0-indexed array)
# So element at index i-1 in array
dp[i] = f(array[i-1], dp[i-1])
```

### 4. Integer Overflow

For counting problems with large results:

```python
MOD = 10**9 + 7

def count_ways(n):
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = (dp[i-1] + dp[i-2]) % MOD  # Take modulo
    return dp[n]
```

### 5. Not Considering All Transitions

Ensure your recurrence considers all possible ways to reach a state.

### 6. Mutable Default Arguments

```python
# Wrong: memo is shared across calls!
def dp(n, memo={}):
    ...

# Correct
def dp(n, memo=None):
    if memo is None:
        memo = {}
    ...
```

## Problem-Solving Framework

### Step-by-Step Approach

1. **Understand the problem**
   - What are we optimizing/counting/deciding?
   - What are the constraints?

2. **Check if DP is applicable**
   - Optimal substructure?
   - Overlapping subproblems?
   - Can you identify a recurrence?

3. **Define the state**
   - What parameters uniquely identify a subproblem?
   - Minimize dimensions if possible

4. **Write the recurrence**
   - How does `dp[current]` relate to previous states?
   - Consider all transitions

5. **Identify base cases**
   - What are the simplest subproblems?

6. **Choose implementation**
   - Start with memoization (easier to implement)
   - Optimize to tabulation if needed
   - Consider space optimization

7. **Code and test**
   - Test base cases
   - Test small examples
   - Verify time/space complexity

## Real-World Applications

### 1. Text Processing
- Spell checkers (edit distance)
- Diff tools (LCS)
- Plagiarism detection (longest common substring)

### 2. Computational Biology
- DNA sequence alignment
- Protein folding prediction
- Gene prediction

### 3. Resource Allocation
- Memory management
- Cache algorithms
- Budget optimization

### 4. Graphics and Image Processing
- Seam carving (content-aware image resizing)
- Image segmentation
- Path finding in graphics

### 5. Compiler Optimization
- Register allocation
- Code generation
- Instruction scheduling

### 6. Network Routing
- Shortest paths with constraints
- Network flow optimization
- Bandwidth allocation

### 7. Game Theory
- Optimal game playing strategies
- Move prediction
- Score maximization

### 8. Finance
- Portfolio optimization
- Option pricing
- Risk management

## Practice Problems by Difficulty

### Beginner

1. Climbing Stairs (LeetCode 70)
2. Min Cost Climbing Stairs (LeetCode 746)
3. House Robber (LeetCode 198)
4. Maximum Subarray (LeetCode 53)
5. Best Time to Buy and Sell Stock (LeetCode 121)

### Intermediate

1. Longest Increasing Subsequence (LeetCode 300)
2. Coin Change (LeetCode 322)
3. Word Break (LeetCode 139)
4. Unique Paths (LeetCode 62)
5. Longest Common Subsequence (LeetCode 1143)
6. Edit Distance (LeetCode 72)
7. Partition Equal Subset Sum (LeetCode 416)
8. Decode Ways (LeetCode 91)

### Advanced

1. Burst Balloons (LeetCode 312)
2. Regular Expression Matching (LeetCode 10)
3. Wildcard Matching (LeetCode 44)
4. Distinct Subsequences (LeetCode 115)
5. Interleaving String (LeetCode 97)
6. Palindrome Partitioning II (LeetCode 132)
7. Best Time to Buy and Sell Stock IV (LeetCode 188)
8. Cherry Pickup (LeetCode 741)

### Expert

1. Minimum Window Subsequence (LeetCode 727)
2. Count Different Palindromic Subsequences (LeetCode 730)
3. Strange Printer (LeetCode 664)
4. Frog Jump (LeetCode 403)
5. Number of Music Playlists (LeetCode 920)

## Quick Reference

### When to Use DP

✅ **Use DP when**:
- Problem asks for optimum (max/min) or count
- Decisions lead to subproblems with similar structure
- Same subproblems appear multiple times
- Problem has optimal substructure

❌ **Don't use DP when**:
- Problem needs actual combinations/permutations (use backtracking)
- Greedy approach works
- Problem is NP-complete without special structure
- State space is too large

### DP vs Other Paradigms

| Paradigm | When to Use | Example |
|----------|-------------|---------|
| **DP** | Overlapping subproblems, optimal substructure | LCS, Knapsack |
| **Greedy** | Optimal substructure, greedy choice property | Huffman coding, Activity selection |
| **Divide & Conquer** | Non-overlapping subproblems | Merge sort, Quick sort |
| **Backtracking** | Need all solutions, not just optimal | N-Queens, Sudoku |

### State Transition Patterns

```python
# 1. Take or skip
dp[i] = max(skip, take)

# 2. Extend or reset
dp[i] = max(dp[i-1] + arr[i], arr[i])

# 3. Minimum of choices
dp[i] = min(choice1, choice2, ...)

# 4. Sum of ways
dp[i] = sum(dp[j] for j in valid_previous_states)

# 5. 2D combination
dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
```

## ELI10 (Explain Like I'm 10)

Imagine you're climbing a staircase and you can either take 1 step or 2 steps at a time. How many different ways can you reach the top?

You could try every single path (slow!), or you could be smart:

"To reach step 5, I either came from step 4 (one 1-step) or step 3 (one 2-step). So: ways(5) = ways(4) + ways(3)"

That's DP! Instead of redoing all the work, you remember answers to smaller problems and build up to the big answer. Like remembering your times tables instead of counting on your fingers every time!

## Further Resources

### Online Judges
- [LeetCode DP Problems](https://leetcode.com/tag/dynamic-programming/) - 500+ problems
- [Codeforces DP Tag](https://codeforces.com/problemset/tags/dp) - Competitive programming
- [AtCoder DP Contest](https://atcoder.jp/contests/dp) - Educational DP problems

### Books
- "Introduction to Algorithms" (CLRS) - Chapter 15
- "Algorithm Design" by Kleinberg & Tardos - Chapter 6
- "Dynamic Programming for Coding Interviews" by Meenakshi & Kamal Rawat

### Tutorials
- [GeeksforGeeks DP Tutorial](https://www.geeksforgeeks.org/dynamic-programming/)
- [TopCoder DP Tutorial](https://www.topcoder.com/community/competitive-programming/tutorials/dynamic-programming-from-novice-to-advanced/)
- [Tushar Roy's YouTube DP Playlist](https://www.youtube.com/playlist?list=PLrmLmBdmIlpsHaNTPP_jHHDx_os9ItYXr)

### Visualizations
- [VisuAlgo Dynamic Programming](https://visualgo.net/en/recursion) - Interactive visualizations
- [Algorithm Visualizer](https://algorithm-visualizer.org/)

### Practice Platforms
- [NeetCode DP Roadmap](https://neetcode.io/roadmap) - Curated problem list
- [Blind 75](https://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU) - Essential interview problems
