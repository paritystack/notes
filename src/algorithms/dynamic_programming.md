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

### DP with Data Structures

Combining DP with advanced data structures can optimize solutions.

#### Monotonic Queue Optimization

**Problem**: Sliding window maximum with DP.

```python
from collections import deque

def max_sliding_window_dp(nums, k):
    """DP with monotonic deque for sliding window maximum"""
    if not nums:
        return []

    dq = deque()  # Stores indices
    result = []

    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (maintain decreasing order)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result when window is full
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Time: O(n), Space: O(k)
```

#### Segment Tree DP

**Problem**: Range maximum query with updates in DP.

```python
class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (4 * n)

    def update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(2*node, start, mid, idx, val)
            else:
                self.update(2*node+1, mid+1, end, idx, val)
            self.tree[node] = max(self.tree[2*node], self.tree[2*node+1])

    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('-inf')
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return max(
            self.query(2*node, start, mid, l, r),
            self.query(2*node+1, mid+1, end, l, r)
        )

def dp_with_segment_tree(arr):
    """DP optimization using segment tree"""
    n = len(arr)
    st = SegmentTree(n)
    dp = [0] * n

    for i in range(n):
        # Query best previous state in range
        if i > 0:
            best = st.query(1, 0, n-1, 0, i-1)
            dp[i] = best + arr[i]
        else:
            dp[i] = arr[i]
        # Update segment tree
        st.update(1, 0, n-1, i, dp[i])

    return max(dp)

# Time: O(n log n), Space: O(n)
```

### Convex Hull Trick (CHT)

Optimize DP transitions with linear functions.

**Problem**: Minimum cost with linear transitions.

```python
from collections import deque

def convex_hull_trick(arr):
    """DP optimization using convex hull trick"""
    n = len(arr)
    dp = [0] * n

    # Line represented as (m, c) for y = mx + c
    hull = deque()

    def bad(l1, l2, l3):
        """Check if l2 is redundant"""
        m1, c1 = l1
        m2, c2 = l2
        m3, c3 = l3
        # Cross product comparison
        return (c3 - c1) * (m1 - m2) <= (c2 - c1) * (m1 - m3)

    def query(hull, x):
        """Find minimum value at x"""
        # Binary search for best line
        left, right = 0, len(hull) - 1
        while left < right:
            mid = (left + right) // 2
            m1, c1 = hull[mid]
            m2, c2 = hull[mid + 1]
            if m1 * x + c1 >= m2 * x + c2:
                left = mid + 1
            else:
                right = mid
        m, c = hull[left]
        return m * x + c

    dp[0] = 0
    hull.append((0, 0))  # Initial line

    for i in range(1, n):
        # Query best previous state
        dp[i] = query(hull, arr[i])

        # Add new line to hull
        new_line = (i, dp[i])
        while len(hull) >= 2 and bad(hull[-2], hull[-1], new_line):
            hull.pop()
        hull.append(new_line)

    return dp[n-1]

# Time: O(n log n) with binary search, O(n) if queries are monotonic
```

### Divide and Conquer Optimization

For DP with special monotonicity property.

**Condition**: If `dp[i][j] = min(dp[i-1][k] + cost[k][j])` for `k < j`, and the optimal `k` is monotonic.

```python
def divide_and_conquer_dp(cost, m):
    """
    Divide and conquer DP optimization
    dp[i][j] = min cost to partition arr[0..j] into i groups
    """
    n = len(cost)
    dp = [[float('inf')] * n for _ in range(m + 1)]

    # Base case
    for j in range(n):
        dp[1][j] = cost[0][j]

    def solve(i, l, r, opt_l, opt_r):
        """
        Compute dp[i][l..r] knowing optimal k is in [opt_l, opt_r]
        """
        if l > r:
            return

        mid = (l + r) // 2
        best_k = -1

        # Find optimal k for dp[i][mid]
        for k in range(opt_l, min(mid, opt_r) + 1):
            val = dp[i-1][k] + cost[k+1][mid]
            if val < dp[i][mid]:
                dp[i][mid] = val
                best_k = k

        # Recursively solve left and right
        solve(i, l, mid - 1, opt_l, best_k)
        solve(i, mid + 1, r, best_k, opt_r)

    for i in range(2, m + 1):
        solve(i, 0, n - 1, 0, n - 1)

    return dp[m][n-1]

# Time: O(m × n log n), Space: O(m × n)
# Without optimization: O(m × n²)
```

### Knuth's Optimization

For interval DP with quadrangle inequality.

**Condition**: If `cost[i][j]` satisfies quadrangle inequality.

```python
def knuth_optimization(arr):
    """
    Optimal binary search tree using Knuth's optimization
    """
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]  # Stores optimal split point

    # Base case: single elements
    for i in range(n):
        opt[i][i] = i

    # Build for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')

            # Search only between opt[i][j-1] and opt[i+1][j]
            for k in range(opt[i][j-1], min(opt[i+1][j], j) + 1):
                cost = dp[i][k-1] if k > i else 0
                cost += dp[k+1][j] if k < j else 0
                cost += sum(arr[i:j+1])  # Additional cost

                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n-1]

# Time: O(n²), Space: O(n²)
# Without optimization: O(n³)
```

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

#### Pattern 1: Maximum Independent Set in Tree

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

#### Pattern 2: Tree Distance DP

**Problem**: Find maximum distance from each node.

```python
def tree_distance_dp(graph, n):
    """Maximum distance from each node in tree"""
    # dp_down[v] = max distance going down from v
    # dp_up[v] = max distance going up from v
    dp_down = [0] * n
    dp_up = [0] * n

    def dfs_down(node, parent):
        """Calculate max distance going down"""
        max_dist = 0
        for child in graph[node]:
            if child != parent:
                dfs_down(child, node)
                max_dist = max(max_dist, 1 + dp_down[child])
        dp_down[node] = max_dist

    def dfs_up(node, parent):
        """Calculate max distance going up or to siblings"""
        # Find two largest child distances
        distances = []
        for child in graph[node]:
            if child != parent:
                distances.append(dp_down[child])
        distances.sort(reverse=True)

        for child in graph[node]:
            if child != parent:
                # Distance going up through parent
                up_dist = dp_up[node] + 1

                # Distance to sibling through parent
                if distances and dp_down[child] == distances[0]:
                    # This child has max distance, use second max
                    sibling_dist = (distances[1] + 2) if len(distances) > 1 else 0
                else:
                    sibling_dist = distances[0] + 2 if distances else 0

                dp_up[child] = max(up_dist, sibling_dist)
                dfs_up(child, node)

    dfs_down(0, -1)
    dfs_up(0, -1)

    # Answer for each node
    return [max(dp_down[i], dp_up[i]) for i in range(n)]

# Time: O(n), Space: O(n)
```

#### Pattern 3: Rerooting Technique

**Problem**: Compute answer for each node as root.

```python
def tree_rerooting(graph, n):
    """Compute DP for each node as root using rerooting"""
    dp = [0] * n
    ans = [0] * n

    def dfs1(node, parent):
        """First DFS: compute subtree answers"""
        result = 0
        for child in graph[node]:
            if child != parent:
                result += dfs1(child, node) + 1
        dp[node] = result
        return result

    def dfs2(node, parent, parent_contribution):
        """Second DFS: reroot and compute answers"""
        ans[node] = dp[node] + parent_contribution

        for child in graph[node]:
            if child != parent:
                # Remove child's contribution
                without_child = ans[node] - (dp[child] + 1)
                # Reroot to child
                dfs2(child, node, without_child + 1)

    dfs1(0, -1)
    dfs2(0, -1, 0)
    return ans

# Time: O(n), Space: O(n)
```

### Probabilistic DP

Handle problems involving probabilities and expected values.

#### Expected Value DP

**Problem**: Expected number of dice rolls to reach target.

```python
def expected_dice_rolls(target):
    """Expected rolls to reach target with fair die (1-6)"""
    # dp[i] = expected rolls to reach target from i
    dp = [0] * (target + 7)

    for i in range(target - 1, -1, -1):
        # From position i, roll die
        expected = 0
        for dice in range(1, 7):
            next_pos = min(i + dice, target)
            if next_pos == target:
                expected += 1  # Reached target in 1 roll
            else:
                expected += 1 + dp[next_pos]  # 1 roll + expected from next
        dp[i] = expected / 6  # Average over all outcomes

    return dp[0]

# Time: O(target), Space: O(target)
```

#### Probability DP

**Problem**: Probability of reaching target score.

```python
def probability_target(n, k, target):
    """
    Probability of reaching exactly target with n dice, k faces each
    """
    # dp[i][j] = probability of sum j using i dice
    dp = [[0.0] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = 1.0  # Base: 0 dice, 0 sum

    for i in range(1, n + 1):
        for j in range(i, min(target + 1, i * k + 1)):
            # Roll current die
            for face in range(1, k + 1):
                if j - face >= 0:
                    dp[i][j] += dp[i-1][j-face] / k

    return dp[n][target]

# Time: O(n × target × k), Space: O(n × target)
```

#### Expected Value with Decisions

**Problem**: Expected maximum value with optimal strategy.

```python
def expected_maximum_value(prices):
    """
    Expected value selling stock optimally
    Each day: know future is randomly up/down
    """
    n = len(prices)
    # dp[i] = expected value starting from day i
    dp = [0] * (n + 1)

    for i in range(n - 1, -1, -1):
        # Option 1: Sell now
        sell_now = prices[i]

        # Option 2: Wait (assume 50% up, 50% down)
        if i < n - 1:
            wait = (dp[i+1] * 1.1 + dp[i+1] * 0.9) / 2  # Expected next value
        else:
            wait = 0

        dp[i] = max(sell_now, wait)

    return dp[0]

# Time: O(n), Space: O(n)
```

### Profile DP

For grid problems where you need to track column state.

**Problem**: Tiling a board with dominoes.

```python
def domino_tiling(n, m):
    """Count ways to tile n×m board with 1×2 dominoes"""
    # dp[col][mask] = ways to reach col with profile mask
    # mask[i] = 1 if cell (i, col) is filled from previous column

    def fits(mask, i, n):
        """Check if we can place tiles starting from row i"""
        if i == n:
            return mask == 0  # All cells must be filled

        if mask & (1 << i):  # Already filled
            return fits(mask, i + 1, n)

        # Try vertical tile (fills current column)
        result = fits(mask | (1 << i), i + 1, n)

        # Try horizontal tile (extends to next column)
        if i + 1 < n and not (mask & (1 << (i + 1))):
            new_mask = mask | (1 << i) | (1 << (i + 1))
            result += fits(new_mask, i + 2, n)

        return result

    # dp[col][mask]
    dp = [{} for _ in range(m + 1)]
    dp[0][0] = 1

    for col in range(m):
        for mask, ways in dp[col].items():
            # Try all next profiles
            def fill_column(row, curr_mask, next_mask):
                if row == n:
                    dp[col + 1][next_mask] = dp[col + 1].get(next_mask, 0) + ways
                    return

                if curr_mask & (1 << row):  # Already filled
                    fill_column(row + 1, curr_mask, next_mask)
                else:
                    # Place vertical tile
                    fill_column(row + 1, curr_mask | (1 << row), next_mask)

                    # Place horizontal tile
                    if row + 1 < n and not (curr_mask & (1 << (row + 1))):
                        new_curr = curr_mask | (1 << row) | (1 << (row + 1))
                        new_next = next_mask | (1 << row) | (1 << (row + 1))
                        fill_column(row + 2, new_curr, new_next)

            fill_column(0, mask, 0)

    return dp[m].get(0, 0)

# Time: O(m × 2^n × n), Space: O(2^n)
```

### SOS (Sum over Subsets) DP

Efficiently compute sum over all subsets.

**Problem**: For each mask, compute sum over all its submasks.

```python
def sum_over_subsets(arr):
    """
    For each mask i, compute sum of arr[j] for all j that are submasks of i
    """
    n = len(arr)
    max_mask = n  # Assuming arr indexed by mask values
    log_n = max_mask.bit_length()

    dp = arr[:]

    # Iterate over bits
    for i in range(log_n):
        # Iterate over masks
        for mask in range(max_mask):
            if mask & (1 << i):
                # Add contribution from mask without bit i
                dp[mask] += dp[mask ^ (1 << i)]

    return dp

# Time: O(n × log n) where n = 2^k
# Without SOS DP: O(3^k) for k bits
```

**Example: Count AND pairs**

```python
def count_and_pairs(arr, target):
    """Count pairs where arr[i] & arr[j] == target"""
    max_val = max(arr)
    freq = [0] * (max_val + 1)

    # Count frequency
    for num in arr:
        freq[num] += 1

    # SOS DP
    dp = freq[:]
    for i in range(20):  # Assuming 20-bit numbers
        for mask in range(max_val + 1):
            if mask & (1 << i):
                dp[mask] += dp[mask ^ (1 << i)]

    # Count pairs
    count = 0
    for num in arr:
        # Find supermasks that AND with num gives target
        supermask = num | target
        if supermask <= max_val:
            count += dp[supermask]

    return count

# Time: O(n + M log M) where M is max value
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

### Matrix Exponentiation with DP

Optimize linear recurrences using matrix exponentiation.

**Problem**: Compute nth Fibonacci number in O(log n).

```python
def matrix_mult(A, B):
    """Multiply two 2x2 matrices"""
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ]

def matrix_pow(M, n):
    """Compute M^n using binary exponentiation"""
    if n == 1:
        return M
    if n % 2 == 0:
        half = matrix_pow(M, n // 2)
        return matrix_mult(half, half)
    else:
        return matrix_mult(M, matrix_pow(M, n - 1))

def fibonacci_fast(n):
    """Compute nth Fibonacci in O(log n)"""
    if n <= 1:
        return n

    # Transformation matrix for Fibonacci
    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n)
    return result[0][1]

# Time: O(log n), Space: O(log n)
# Standard DP: O(n) time
```

**General linear recurrence**:

```python
def linear_recurrence_fast(coeffs, init, n):
    """
    Solve f(n) = c1*f(n-1) + c2*f(n-2) + ... + ck*f(n-k)
    coeffs = [c1, c2, ..., ck]
    init = [f(0), f(1), ..., f(k-1)]
    """
    k = len(coeffs)
    if n < k:
        return init[n]

    # Build transformation matrix
    M = [[0] * k for _ in range(k)]
    M[0] = coeffs
    for i in range(1, k):
        M[i][i-1] = 1

    def mat_mult(A, B):
        size = len(A)
        C = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    def mat_pow(mat, exp):
        if exp == 1:
            return mat
        if exp % 2 == 0:
            half = mat_pow(mat, exp // 2)
            return mat_mult(half, half)
        return mat_mult(mat, mat_pow(mat, exp - 1))

    # Apply transformation n - k + 1 times
    result_mat = mat_pow(M, n - k + 1)

    # Compute result from initial values
    result = 0
    for i in range(k):
        result += result_mat[0][i] * init[k - 1 - i]

    return result

# Time: O(k³ log n), Space: O(k²)
```

### DP with Number Theory

Combine DP with mathematical properties.

#### Counting with Modular Arithmetic

```python
MOD = 10**9 + 7

def count_ways_mod(n, k):
    """Count ways to reach n using steps 1 to k, modulo MOD"""
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for step in range(1, min(i, k) + 1):
            dp[i] = (dp[i] + dp[i - step]) % MOD

    return dp[n]

# Time: O(n × k), Space: O(n)
```

#### DP with GCD/LCM

```python
import math

def max_gcd_path(grid):
    """Maximum GCD along path from top-left to bottom-right"""
    m, n = len(grid), len(grid[0])
    # dp[i][j] = set of possible GCDs reaching (i, j)
    dp = [[set() for _ in range(n)] for _ in range(m)]

    dp[0][0].add(grid[0][0])

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue

            # From top
            if i > 0:
                for g in dp[i-1][j]:
                    dp[i][j].add(math.gcd(g, grid[i][j]))

            # From left
            if j > 0:
                for g in dp[i][j-1]:
                    dp[i][j].add(math.gcd(g, grid[i][j]))

    return max(dp[m-1][n-1])

# Time: O(m × n × G × log V) where G is number of unique GCDs, V is max value
```

#### Digit DP with Constraints

```python
def count_numbers_with_digit_sum(n, target_sum):
    """Count numbers from 1 to n with digit sum equal to target_sum"""
    s = str(n)
    memo = {}

    def dp(pos, sum_so_far, tight, started):
        """
        pos: current position
        sum_so_far: sum of digits chosen
        tight: whether we're still bounded by n
        started: whether we've placed a non-zero digit
        """
        if pos == len(s):
            return 1 if (started and sum_so_far == target_sum) else 0

        state = (pos, sum_so_far, tight, started)
        if state in memo:
            return memo[state]

        limit = int(s[pos]) if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            if not started and digit == 0:
                # Leading zero
                result += dp(pos + 1, sum_so_far, False, False)
            else:
                new_sum = sum_so_far + digit
                if new_sum <= target_sum:  # Prune
                    new_tight = tight and (digit == limit)
                    result += dp(pos + 1, new_sum, new_tight, True)

        memo[state] = result
        return result

    return dp(0, 0, True, False)

# Time: O(len(n) × target_sum × 2 × 2 × 10)
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

### Debugging DP Solutions

#### Common Debugging Strategies

1. **Print the DP table**
```python
def debug_dp_table(dp):
    """Visualize DP table"""
    for i, row in enumerate(dp):
        print(f"dp[{i}] = {row}")
```

2. **Verify base cases**
```python
def verify_base_cases():
    """Test smallest inputs"""
    assert climb_stairs(1) == 1
    assert climb_stairs(2) == 2
    assert climb_stairs(3) == 3
```

3. **Check recurrence manually**
```python
def manual_check():
    """Manually verify recurrence for small n"""
    # For climbing stairs: dp[3] should equal dp[2] + dp[1]
    assert dp[3] == dp[2] + dp[1]
```

4. **Compare with brute force**
```python
def brute_force(n):
    """Exponential but correct solution"""
    if n <= 1:
        return 1
    return brute_force(n-1) + brute_force(n-2)

def test_against_brute_force():
    """Verify DP against brute force for small inputs"""
    for n in range(1, 15):
        assert climb_stairs(n) == brute_force(n)
```

5. **Trace execution**
```python
def dp_with_trace(n, memo=None):
    """Add tracing to see execution flow"""
    if memo is None:
        memo = {}

    print(f"Computing dp({n})")

    if n in memo:
        print(f"  -> Found in memo: {memo[n]}")
        return memo[n]

    if n <= 1:
        print(f"  -> Base case: {n}")
        return n

    result = dp_with_trace(n-1, memo) + dp_with_trace(n-2, memo)
    memo[n] = result
    print(f"  -> Computed dp({n}) = {result}")
    return result
```

#### Performance Testing

```python
import time
import functools

def benchmark_dp_solutions():
    """Compare different DP approaches"""
    n = 30

    # Memoization
    start = time.time()
    @functools.lru_cache(None)
    def fib_memo(n):
        return n if n <= 1 else fib_memo(n-1) + fib_memo(n-2)
    result1 = fib_memo(n)
    time1 = time.time() - start

    # Tabulation
    start = time.time()
    def fib_tab(n):
        if n <= 1: return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    result2 = fib_tab(n)
    time2 = time.time() - start

    # Space-optimized
    start = time.time()
    def fib_opt(n):
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    result3 = fib_opt(n)
    time3 = time.time() - start

    print(f"Memoization: {time1:.6f}s")
    print(f"Tabulation:  {time2:.6f}s")
    print(f"Optimized:   {time3:.6f}s")
```

### Optimization Checklist

Before submitting your DP solution, verify:

- [ ] **State is minimal**: No redundant dimensions
- [ ] **Base cases are correct**: Handle edge cases (n=0, empty array, etc.)
- [ ] **Recurrence is complete**: All transitions considered
- [ ] **Iteration order is correct**: Smaller subproblems computed first
- [ ] **Space can be optimized**: Check if rolling array applies
- [ ] **Integer overflow handled**: Use modulo if needed
- [ ] **Time complexity is acceptable**: Ensure it fits constraints
- [ ] **Tested on examples**: Small inputs, edge cases, large inputs

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

### 9. Machine Learning
- Sequence alignment in NLP
- Hidden Markov Models (Viterbi algorithm)
- Reinforcement learning (value iteration, policy iteration)

## Advanced Case Studies

### Case Study 1: Autocomplete System

**Problem**: Design an autocomplete system that suggests top k sentences based on input.

**DP Application**: Trie + DP for ranking.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.sentences = []  # (sentence, frequency) pairs

class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.current = self.root
        self.prefix = ""

        # Build trie with DP for top-k at each node
        for sentence, freq in zip(sentences, times):
            self._add_sentence(sentence, freq)

    def _add_sentence(self, sentence, freq):
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # DP: maintain top k sentences at each node
            node.sentences.append((sentence, freq))
            node.sentences.sort(key=lambda x: (-x[1], x[0]))
            node.sentences = node.sentences[:3]  # Keep top 3

    def input(self, c):
        if c == '#':
            # Save sentence
            self._add_sentence(self.prefix, 1)
            self.prefix = ""
            self.current = self.root
            return []

        self.prefix += c
        if self.current and c in self.current.children:
            self.current = self.current.children[c]
            return [s for s, _ in self.current.sentences]
        else:
            self.current = None
            return []

# Time: O(k × L) per input, where L is sentence length
# Space: O(T) where T is total characters in trie
```

### Case Study 2: Video Encoding Optimization

**Problem**: Optimize video encoding by selecting keyframes to minimize file size while maintaining quality.

**DP Application**: Interval DP with quality constraints.

```python
def optimize_video_encoding(frames, max_distance):
    """
    Select keyframes to minimize encoding cost
    frames[i] = quality score of frame i
    max_distance = maximum frames between keyframes
    """
    n = len(frames)
    # dp[i] = min cost to encode frames[0..i]
    dp = [float('inf')] * n
    keyframes = [[] for _ in range(n)]

    # Cost function: more distance between keyframes = lower quality
    def encoding_cost(start, end):
        distance = end - start
        if distance > max_distance:
            return float('inf')
        # Cost increases with distance
        base_cost = distance * 10
        quality_loss = sum(frames[start+1:end+1]) * distance
        return base_cost + quality_loss

    # Base case
    dp[0] = 0
    keyframes[0] = [0]

    for i in range(1, n):
        # Try each possible previous keyframe
        for prev_keyframe in range(max(0, i - max_distance), i + 1):
            cost = encoding_cost(prev_keyframe, i)
            total_cost = (dp[prev_keyframe] if prev_keyframe > 0 else 0) + cost

            if total_cost < dp[i]:
                dp[i] = total_cost
                keyframes[i] = keyframes[prev_keyframe - 1] + [i] if prev_keyframe > 0 else [i]

    return dp[n-1], keyframes[n-1]

# Time: O(n × max_distance), Space: O(n)
```

### Case Study 3: Supply Chain Optimization

**Problem**: Minimize cost of ordering and storing inventory over time.

**DP Application**: Inventory management with holding costs.

```python
def optimize_inventory(demand, order_cost, holding_cost, capacity):
    """
    Optimize inventory orders over time
    demand[i] = demand in period i
    order_cost = fixed cost per order
    holding_cost = cost per unit per period
    capacity = warehouse capacity
    """
    n = len(demand)
    # dp[i] = min cost to satisfy demand for periods 0..i
    dp = [float('inf')] * n
    orders = [None] * n

    for i in range(n):
        # Try ordering for periods j to i in one order
        total_demand = 0
        for j in range(i, -1, -1):
            total_demand += demand[j]

            if total_demand > capacity:
                break

            # Calculate holding cost for this order
            hold_cost = 0
            cumulative = 0
            for k in range(j, i + 1):
                cumulative += demand[k]
                # Hold cumulative units for (i - k) periods
                hold_cost += cumulative * holding_cost * (i - k)

            # Total cost
            prev_cost = dp[j-1] if j > 0 else 0
            total = prev_cost + order_cost + hold_cost

            if total < dp[i]:
                dp[i] = total
                orders[i] = (j, i, total_demand)

    # Reconstruct ordering strategy
    strategy = []
    i = n - 1
    while i >= 0:
        strategy.append(orders[i])
        i = orders[i][0] - 1

    return dp[n-1], list(reversed(strategy))

# Time: O(n²), Space: O(n)
```

### Case Study 4: Route Planning with Time Windows

**Problem**: Find optimal delivery route with time window constraints.

**DP Application**: State includes time, making this a 2D DP problem.

```python
def delivery_route_dp(locations, time_windows, travel_time):
    """
    Find optimal delivery sequence
    locations = list of delivery points
    time_windows[i] = (earliest, latest) time for location i
    travel_time[i][j] = time from location i to j
    """
    n = len(locations)
    # dp[mask][last][time] = min cost to visit locations in mask, ending at last, at time
    # Use dictionary for sparse storage
    dp = {}

    def solve(visited, last, current_time):
        state = (visited, last, current_time)
        if state in dp:
            return dp[state]

        # All locations visited
        if visited == (1 << n) - 1:
            return 0

        min_cost = float('inf')

        # Try visiting each unvisited location
        for next_loc in range(n):
            if visited & (1 << next_loc):
                continue

            # Travel to next location
            arrival_time = current_time + travel_time[last][next_loc]
            earliest, latest = time_windows[next_loc]

            # Check if we can make the time window
            if arrival_time <= latest:
                # Wait if we arrive early
                service_time = max(arrival_time, earliest)
                wait_cost = max(0, earliest - arrival_time)

                # Recurse
                future_cost = solve(
                    visited | (1 << next_loc),
                    next_loc,
                    service_time + 1  # Service takes 1 unit
                )

                total_cost = wait_cost + future_cost
                min_cost = min(min_cost, total_cost)

        dp[state] = min_cost
        return min_cost

    # Start from depot (location 0) at time 0
    return solve(1, 0, 0)

# Time: O(n² × 2^n × T) where T is time range
# Space: O(2^n × T)
```

### Case Study 5: Natural Language Processing - Text Segmentation

**Problem**: Segment text into words using a dictionary (Chinese word segmentation).

**DP Application**: String DP with dictionary lookup.

```python
def segment_text(text, dictionary, language_model):
    """
    Segment text into words optimally
    text = unsegmented text
    dictionary = set of valid words
    language_model = function giving probability of word sequence
    """
    n = len(text)
    # dp[i] = (max_prob, segmentation) for text[0..i]
    dp = [(0, [])] * (n + 1)
    dp[0] = (1.0, [])

    for i in range(1, n + 1):
        best_prob = 0
        best_seg = []

        # Try all possible last words
        for j in range(i):
            word = text[j:i]
            if word in dictionary:
                prev_prob, prev_seg = dp[j]
                # Use language model for word probability
                word_prob = language_model(prev_seg, word)
                total_prob = prev_prob * word_prob

                if total_prob > best_prob:
                    best_prob = total_prob
                    best_seg = prev_seg + [word]

        dp[i] = (best_prob, best_seg)

    return dp[n][1]

# Example with simple language model
def simple_language_model(prev_words, new_word):
    """Simple unigram model"""
    # In practice, use bigram/trigram probabilities
    freq = {
        'hello': 0.01,
        'world': 0.008,
        'the': 0.05,
        # ... more word frequencies
    }
    return freq.get(new_word, 0.0001)

# Time: O(n² × D) where D is dictionary lookup time
# Space: O(n × W) where W is average segmentation length
```

### Case Study 6: Database Query Optimization

**Problem**: Optimize join order for multiple database tables.

**DP Application**: Bitmask DP for subset enumeration.

```python
def optimize_join_order(tables, join_costs):
    """
    Find optimal order to join database tables
    tables = list of table names
    join_costs[i][j] = cost to join tables i and j
    """
    n = len(tables)
    # dp[mask] = (min_cost, join_order) for tables in mask
    dp = {}
    dp[0] = (0, [])

    # Initialize single tables
    for i in range(n):
        dp[1 << i] = (0, [tables[i]])

    # Try all subsets
    for mask in range(1, 1 << n):
        if mask not in dp:
            continue

        current_cost, current_order = dp[mask]

        # Try joining with each table not in mask
        for i in range(n):
            if mask & (1 << i):
                continue

            new_mask = mask | (1 << i)

            # Calculate cost of joining table i
            join_cost = 0
            for j in range(n):
                if mask & (1 << j):
                    join_cost += join_costs[j][i]

            total_cost = current_cost + join_cost
            new_order = current_order + [tables[i]]

            if new_mask not in dp or total_cost < dp[new_mask][0]:
                dp[new_mask] = (total_cost, new_order)

    full_mask = (1 << n) - 1
    return dp[full_mask]

# Time: O(n² × 2^n), Space: O(2^n)
```

### Case Study 7: Image Seam Carving (Content-Aware Resizing)

**Problem**: Resize image by removing least important seams.

**DP Application**: Grid DP with energy minimization.

```python
def seam_carving(image, energy_function):
    """
    Find minimum energy vertical seam for content-aware resizing
    image = 2D array of pixels
    energy_function = function to compute pixel importance
    """
    m, n = len(image), len(image[0])

    # Compute energy for each pixel
    energy = [[energy_function(image, i, j) for j in range(n)] for i in range(m)]

    # dp[i][j] = min energy to reach pixel (i, j)
    dp = [[float('inf')] * n for _ in range(m)]
    parent = [[None] * n for _ in range(m)]

    # Base case: first row
    for j in range(n):
        dp[0][j] = energy[0][j]

    # Fill DP table
    for i in range(1, m):
        for j in range(n):
            # Try coming from three possible parents
            for pj in range(max(0, j-1), min(n, j+2)):
                if dp[i-1][pj] + energy[i][j] < dp[i][j]:
                    dp[i][j] = dp[i-1][pj] + energy[i][j]
                    parent[i][j] = pj

    # Find minimum in last row
    min_col = min(range(n), key=lambda j: dp[m-1][j])

    # Backtrack to find seam
    seam = []
    col = min_col
    for i in range(m-1, -1, -1):
        seam.append((i, col))
        if parent[i][col] is not None:
            col = parent[i][col]

    return list(reversed(seam)), dp[m-1][min_col]

def simple_energy(image, i, j):
    """Simple gradient-based energy function"""
    m, n = len(image), len(image[0])
    energy = 0

    # Horizontal gradient
    if j > 0 and j < n - 1:
        energy += abs(image[i][j+1] - image[i][j-1])

    # Vertical gradient
    if i > 0 and i < m - 1:
        energy += abs(image[i+1][j] - image[i-1][j])

    return energy

# Time: O(m × n), Space: O(m × n)
```

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
