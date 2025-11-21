# Backtracking

Backtracking is a general algorithmic technique that incrementally builds candidates for solutions and abandons a candidate as soon as it is determined that it cannot lead to a valid solution. It is often used for solving constraint satisfaction problems, such as puzzles, combinatorial problems, and optimization problems.

## Key Concepts

- **Recursive Approach**: Backtracking is typically implemented using recursion. The algorithm explores each possible option and recursively attempts to build a solution. At each step, the algorithm makes a choice, explores the consequences of that choice, and if it doesn't lead to a solution, it "backs up" by undoing the choice and trying another option. This pattern of "choose → explore → unchoose" is the fundamental building block of all backtracking algorithms.

- **State Space Tree**: The process of backtracking can be visualized as a tree where each node represents a partial state of the solution. The root node represents the initial (empty) state, and each branch represents a decision or choice made. As we traverse down the tree, we build up a candidate solution incrementally. The leaves of the tree represent either complete solutions (if valid) or dead ends (if constraints are violated). Backtracking performs a depth-first traversal of this conceptual tree.

- **Pruning**: One of the key advantages of backtracking is its ability to prune the search space through early constraint checking. Instead of generating all possible candidates and then checking validity, backtracking checks constraints as soon as possible. If a partial solution violates any constraint, the entire subtree rooted at that node is eliminated from consideration, saving potentially exponential time. Effective pruning is often the difference between a practical algorithm and one that's too slow to use.

- **Decision Variables**: In backtracking problems, we identify decision variables that need to be assigned values. For example, in N-Queens, the decision is "which column should the queen in row i be placed?" For each variable, we try all possible values (the domain) that haven't been ruled out by constraints.

- **Constraint Propagation**: Good backtracking implementations check constraints immediately after making each choice. This allows the algorithm to fail fast and backtrack earlier, rather than continuing down an invalid path. The constraints can be explicit (like "no two queens on the same diagonal") or implicit (like "the current sum exceeds the target").

- **Solution Recovery**: Backtracking maintains the current partial solution and modifies it incrementally. When a complete valid solution is found, it's typically copied to a results collection. The algorithm then backtracks to find additional solutions if needed.

## When to Use Backtracking

Backtracking is most appropriate when:

1. **The problem requires exploring all possible solutions**: When you need to find all valid configurations, not just one optimal solution (e.g., generating all permutations, finding all paths).

2. **There are clear constraints that can be checked incrementally**: If you can determine early whether a partial solution can lead to a valid complete solution, pruning becomes effective.

3. **The problem has a recursive structure**: Problems that can be broken down into making a choice, solving the remaining subproblem, and potentially undoing the choice fit the backtracking paradigm naturally.

4. **The search space is too large for brute force but pruning is effective**: While backtracking is still exponential in worst case, good pruning can make many problems tractable.

**Good candidates for backtracking:**
- Constraint satisfaction problems (Sudoku, N-Queens, graph coloring)
- Combinatorial enumeration (permutations, combinations, subsets)
- Puzzle solving (crosswords, mazes, knight's tour)
- Parsing and pattern matching problems

**When NOT to use backtracking:**
- When dynamic programming can be applied (overlapping subproblems with optimal substructure)
- When greedy algorithms work (problems with optimal substructure where local choices lead to global optimum)
- When the problem requires finding shortest paths or optimal solutions (use BFS, Dijkstra, or A* instead)
- When the search space is small enough for simple iteration

## N-Queens Problem

Place N queens on an N×N chessboard such that no two queens threaten each other.

```python
from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem using backtracking.

    Find all distinct solutions to place n queens on an n×n chessboard
    such that no two queens attack each other (same row, column, or diagonal).

    Args:
        n: Size of the chessboard (n×n) and number of queens to place

    Returns:
        List of all valid board configurations, where each configuration
        is a list of strings representing the board rows ('Q' for queen,
        '.' for empty)

    Time Complexity: O(n!) - we try n positions in first row, at most n-1
                     in second row, etc., with pruning
    Space Complexity: O(n²) - for the board, plus O(n) recursion depth

    Example:
        >>> solve_n_queens(4)
        [['.Q..', '...Q', 'Q...', '..Q.'],
         ['..Q.', 'Q...', '...Q', '.Q..']]
    """
    def is_valid(board: List[List[str]], row: int, col: int) -> bool:
        """
        Check if placing a queen at (row, col) is valid.

        Only need to check previous rows since we place queens row by row.
        """
        # Check column for any queens above
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check top-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check top-right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(board: List[List[str]], row: int) -> None:
        """
        Recursively try to place queens starting from the given row.
        """
        # Base case: all queens placed successfully
        if row == n:
            # Convert board to required format and store solution
            result.append([''.join(row) for row in board])
            return

        # Try placing queen in each column of current row
        for col in range(n):
            if is_valid(board, row, col):
                # Make choice: place queen
                board[row][col] = 'Q'

                # Explore: recurse to next row
                backtrack(board, row + 1)

                # Unchoose: remove queen (backtrack)
                board[row][col] = '.'

    result = []
    # Initialize empty board
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(board, 0)
    return result

# Example usage
solutions = solve_n_queens(4)
print(f"Found {len(solutions)} solutions for 4-Queens")
for i, solution in enumerate(solutions, 1):
    print(f"Solution {i}:")
    for row in solution:
        print(row)
    print()
```

## Sudoku Solver

Solve a 9×9 Sudoku puzzle.

```python
from typing import List

def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Solve a 9×9 Sudoku puzzle using backtracking.

    Modifies the input board in-place to fill empty cells (marked with '.')
    with digits 1-9 such that each row, column, and 3×3 box contains all
    digits exactly once.

    Args:
        board: 9×9 grid where empty cells are marked with '.'

    Returns:
        True if the puzzle was solved, False if no solution exists

    Time Complexity: O(9^m) where m is the number of empty cells
                     In worst case, we try up to 9 values per empty cell
    Space Complexity: O(m) for recursion depth where m is number of empty cells

    Example:
        >>> board = [["5","3",".",".","7",".",".",".","."]]  # + 8 more rows
        >>> solve_sudoku(board)
        True
        >>> board[0][2]  # Previously empty cell now filled
        '4'
    """
    def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
        """
        Check if placing num at (row, col) violates Sudoku constraints.
        """
        # Check if num already exists in the row
        if num in board[row]:
            return False

        # Check if num already exists in the column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check if num already exists in the 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def backtrack() -> bool:
        """
        Find next empty cell and try all possible values.
        Returns True if puzzle is solved, False otherwise.
        """
        # Find next empty cell
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    # Try each digit 1-9
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            # Make choice: place the number
                            board[row][col] = num

                            # Explore: recurse to fill remaining cells
                            if backtrack():
                                return True

                            # Unchoose: backtrack if this choice didn't work
                            board[row][col] = '.'

                    # No valid number for this cell, backtrack
                    return False

        # All cells filled successfully
        return True

    backtrack()
    return board

# Example usage
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
if solve_sudoku(board):
    print("Sudoku solved successfully:")
    for row in board:
        print(' '.join(row))
else:
    print("No solution exists")
```

## Generate Subsets

Generate all subsets (power set) of a given set.

```python
from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets (power set) of a given set using backtracking.

    The power set of a set is the set of all possible subsets, including
    the empty set and the set itself.

    Args:
        nums: List of unique integers

    Returns:
        List containing all possible subsets

    Time Complexity: O(2^n) - there are 2^n possible subsets
    Space Complexity: O(n) for recursion depth (not counting output)

    Example:
        >>> subsets([1, 2, 3])
        [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
    """
    result = []

    def backtrack(start: int, path: List[int]) -> None:
        """
        Build subsets by either including or excluding each element.

        Args:
            start: Index to start considering elements from
            path: Current subset being built
        """
        # Add current subset to result (base case implicit)
        # Every path represents a valid subset
        result.append(path[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            # Make choice: include nums[i]
            path.append(nums[i])

            # Explore: recurse with next index
            backtrack(i + 1, path)

            # Unchoose: backtrack by removing the element
            path.pop()

    backtrack(0, [])
    return result

# Example usage
nums = [1, 2, 3]
result = subsets(nums)
print(f"All subsets of {nums}:")
print(result)
# Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

## Generate Permutations

Generate all permutations of a given list.

```python
from typing import List

def permute(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations of a list using backtracking.

    A permutation is an arrangement of all elements in a specific order.
    For n elements, there are n! permutations.

    Args:
        nums: List of unique integers to permute

    Returns:
        List of all possible permutations

    Time Complexity: O(n! * n) - n! permutations, each takes O(n) to build
    Space Complexity: O(n) for recursion depth and path storage

    Example:
        >>> permute([1, 2, 3])
        [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    """
    result = []

    def backtrack(path: List[int], remaining: List[int]) -> None:
        """
        Build permutations by choosing from remaining elements.

        Args:
            path: Current permutation being built
            remaining: Elements not yet chosen
        """
        # Base case: no more elements to choose
        if not remaining:
            result.append(path[:])
            return

        # Try each remaining element as the next choice
        for i in range(len(remaining)):
            # Make choice: add remaining[i] to path
            path.append(remaining[i])

            # Explore: recurse with remaining elements
            # (all except the one we just chose)
            backtrack(path, remaining[:i] + remaining[i+1:])

            # Unchoose: backtrack by removing the element
            path.pop()

    backtrack([], nums)
    return result

# Alternative implementation using in-place swaps (more efficient)
def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations using in-place swapping.

    This approach is more space-efficient as it modifies the input array
    in-place rather than creating new lists for remaining elements.

    Args:
        nums: List of unique integers to permute

    Returns:
        List of all possible permutations

    Time Complexity: O(n! * n)
    Space Complexity: O(n) for recursion only (more efficient than permute)
    """
    result = []

    def backtrack(first: int) -> None:
        """
        Generate permutations by swapping elements.

        Args:
            first: Index of the first position we're currently filling
        """
        # Base case: filled all positions
        if first == len(nums):
            result.append(nums[:])  # Make a copy
            return

        # Try each element from 'first' onwards in this position
        for i in range(first, len(nums)):
            # Make choice: swap nums[first] with nums[i]
            nums[first], nums[i] = nums[i], nums[first]

            # Explore: fix this position and permute the rest
            backtrack(first + 1)

            # Unchoose: swap back to restore original state
            nums[first], nums[i] = nums[i], nums[first]

    backtrack(0)
    return result

# Example usage
nums = [1, 2, 3]
print("Method 1 (using remaining list):")
print(permute(nums))

print("\nMethod 2 (using swaps):")
print(permute_swap(nums))
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

## Combination Sum

Find all combinations that sum to a target value.

```python
from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations of candidates that sum to target.

    Each number in candidates may be used unlimited times. The same
    combination should not be repeated (e.g., [2,2,3] and [2,3,2] are
    considered the same).

    Args:
        candidates: List of distinct positive integers
        target: Target sum to achieve

    Returns:
        List of all unique combinations that sum to target

    Time Complexity: O(n^(t/m)) where n is # of candidates, t is target,
                     m is minimal value in candidates. In worst case, we
                     explore a tree of depth t/m with branching factor n.
    Space Complexity: O(t/m) for recursion depth

    Example:
        >>> combination_sum([2, 3, 6, 7], 7)
        [[2, 2, 3], [7]]
    """
    result = []

    def backtrack(start: int, path: List[int], current_sum: int) -> None:
        """
        Find combinations starting from index 'start'.

        Args:
            start: Index to start considering candidates from (prevents duplicates)
            path: Current combination being built
            current_sum: Sum of numbers in current path
        """
        # Base case: found a valid combination
        if current_sum == target:
            result.append(path[:])
            return

        # Pruning: if sum exceeds target, no point continuing
        if current_sum > target:
            return

        # Try each candidate starting from 'start' index
        for i in range(start, len(candidates)):
            # Make choice: add candidates[i] to combination
            path.append(candidates[i])

            # Explore: can reuse same element, so pass i (not i+1)
            backtrack(i, path, current_sum + candidates[i])

            # Unchoose: backtrack by removing the element
            path.pop()

    backtrack(0, [], 0)
    return result

# Example usage
candidates = [2, 3, 6, 7]
target = 7
result = combination_sum(candidates, target)
print(f"Combinations that sum to {target}: {result}")
# Output: [[2, 2, 3], [7]]

# Another example
candidates2 = [2, 3, 5]
target2 = 8
result2 = combination_sum(candidates2, target2)
print(f"Combinations that sum to {target2}: {result2}")
# Output: [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```

## Word Search

Find if a word exists in a 2D board.

```python
from typing import List

def word_search(board: List[List[str]], word: str) -> bool:
    """
    Determine if a word exists in a 2D character board.

    The word can be constructed from letters of sequentially adjacent cells,
    where adjacent cells are horizontally or vertically neighboring. The same
    cell may not be used more than once in a single word.

    Args:
        board: 2D grid of characters
        word: Word to search for

    Returns:
        True if word exists in the board, False otherwise

    Time Complexity: O(m * n * 4^L) where m,n are board dimensions and L is
                     word length. We try each cell as a starting point (m*n)
                     and explore 4 directions for each of L characters.
    Space Complexity: O(L) for recursion depth

    Example:
        >>> board = [['A','B','C','E'], ['S','F','C','S'], ['A','D','E','E']]
        >>> word_search(board, "ABCCED")
        True
        >>> word_search(board, "SEE")
        True
        >>> word_search(board, "ABCB")
        False
    """
    rows, cols = len(board), len(board[0])

    def backtrack(row: int, col: int, index: int) -> bool:
        """
        Search for word[index:] starting at board[row][col].

        Args:
            row: Current row position
            col: Current column position
            index: Current position in the word we're matching

        Returns:
            True if we can match the remaining word from this position
        """
        # Base case: found the complete word
        if index == len(word):
            return True

        # Check boundaries and character match
        if (row < 0 or row >= rows or
            col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False

        # Make choice: mark current cell as visited
        temp = board[row][col]
        board[row][col] = '#'  # Use '#' as visited marker

        # Explore: try all four directions (down, up, right, left)
        found = (backtrack(row + 1, col, index + 1) or  # Down
                backtrack(row - 1, col, index + 1) or   # Up
                backtrack(row, col + 1, index + 1) or   # Right
                backtrack(row, col - 1, index + 1))     # Left

        # Unchoose: restore the cell (backtrack)
        board[row][col] = temp

        return found

    # Try starting the search from each cell in the board
    for row in range(rows):
        for col in range(cols):
            if backtrack(row, col, 0):
                return True

    return False

# Example usage
board = [
    ['A','B','C','E'],
    ['S','F','C','S'],
    ['A','D','E','E']
]

test_words = ["ABCCED", "SEE", "ABCB"]
for word in test_words:
    result = word_search(board, word)
    print(f"Word '{word}' exists: {result}")
# Output:
# Word 'ABCCED' exists: True
# Word 'SEE' exists: True
# Word 'ABCB' exists: False
```

## Palindrome Partitioning

Partition a string into all possible palindrome substrings.

```python
from typing import List

def partition(s: str) -> List[List[str]]:
    """
    Partition a string into all possible palindrome substrings.

    Find all possible ways to partition the string such that every
    substring in the partition is a palindrome.

    Args:
        s: Input string to partition

    Returns:
        List of all possible palindrome partitions

    Time Complexity: O(n * 2^n) where n is the length of the string.
                     There are 2^n possible partitions (we can cut or not
                     cut between each pair of characters), and for each
                     partition we spend O(n) checking palindromes.
    Space Complexity: O(n) for recursion depth and path storage

    Example:
        >>> partition("aab")
        [["a","a","b"], ["aa","b"]]
        >>> partition("a")
        [["a"]]
    """
    def is_palindrome(s: str, left: int, right: int) -> bool:
        """
        Check if substring s[left:right+1] is a palindrome.

        Args:
            s: The string to check
            left: Left index of substring
            right: Right index of substring

        Returns:
            True if substring is a palindrome, False otherwise
        """
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    result = []

    def backtrack(start: int, path: List[str]) -> None:
        """
        Find all palindrome partitions starting from index 'start'.

        Args:
            start: Current starting index in the string
            path: Current partition being built
        """
        # Base case: reached end of string, found valid partition
        if start == len(s):
            result.append(path[:])
            return

        # Try all possible end positions for the next palindrome
        for end in range(start, len(s)):
            # Only proceed if s[start:end+1] is a palindrome
            if is_palindrome(s, start, end):
                # Make choice: add this palindrome to current partition
                path.append(s[start:end+1])

                # Explore: continue partitioning the rest of the string
                backtrack(end + 1, path)

                # Unchoose: backtrack by removing the palindrome
                path.pop()

    backtrack(0, [])
    return result

# Example usage
test_strings = ["aab", "a", "aabb"]
for s in test_strings:
    result = partition(s)
    print(f"Palindrome partitions of '{s}':")
    for p in result:
        print(f"  {p}")
    print()
# Output:
# Palindrome partitions of 'aab':
#   ['a', 'a', 'b']
#   ['aa', 'b']
#
# Palindrome partitions of 'a':
#   ['a']
#
# Palindrome partitions of 'aabb':
#   ['a', 'a', 'b', 'b']
#   ['a', 'a', 'bb']
#   ['aa', 'b', 'b']
#   ['aa', 'bb']
```

## Graph Coloring

Color the vertices of a graph such that no two adjacent vertices have the same color, using minimum number of colors.

```python
from typing import List, Dict, Set

def graph_coloring(graph: Dict[int, List[int]], num_colors: int) -> List[int]:
    """
    Solve the graph coloring problem using backtracking.

    Assign colors to vertices such that no two adjacent vertices have the
    same color, using at most num_colors colors.

    Args:
        graph: Adjacency list representation {vertex: [neighbors]}
        num_colors: Maximum number of colors available

    Returns:
        List where index is vertex and value is assigned color (0 to num_colors-1)
        Empty list if no valid coloring exists

    Time Complexity: O(m^n) where n is number of vertices, m is number of colors
                     In worst case, we try all color combinations
    Space Complexity: O(n) for recursion depth and color assignment

    Example:
        >>> graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
        >>> graph_coloring(graph, 3)
        [0, 1, 2, 0]  # Valid 3-coloring
    """
    n = len(graph)
    colors = [-1] * n  # -1 means uncolored

    def is_valid(vertex: int, color: int) -> bool:
        """
        Check if assigning 'color' to 'vertex' is valid.

        Args:
            vertex: The vertex to color
            color: The color to try

        Returns:
            True if no adjacent vertex has this color
        """
        # Check all neighbors of this vertex
        for neighbor in graph[vertex]:
            if colors[neighbor] == color:
                return False
        return True

    def backtrack(vertex: int) -> bool:
        """
        Try to color vertices starting from 'vertex'.

        Args:
            vertex: Current vertex to color

        Returns:
            True if successfully colored all vertices
        """
        # Base case: all vertices colored successfully
        if vertex == n:
            return True

        # Try each color for current vertex
        for color in range(num_colors):
            if is_valid(vertex, color):
                # Make choice: assign this color
                colors[vertex] = color

                # Explore: try to color remaining vertices
                if backtrack(vertex + 1):
                    return True

                # Unchoose: backtrack
                colors[vertex] = -1

        # No valid coloring found with current choices
        return False

    # Start coloring from vertex 0
    if backtrack(0):
        return colors
    else:
        return []  # No valid coloring exists

# Example usage
# Graph: 0 -- 1
#        |  X |
#        2 -- 3
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2]
}

for num_colors in [2, 3, 4]:
    result = graph_coloring(graph, num_colors)
    if result:
        print(f"Valid {num_colors}-coloring: {result}")
    else:
        print(f"No valid {num_colors}-coloring exists")
# Output:
# No valid 2-coloring exists
# Valid 3-coloring: [0, 1, 2, 0]
# Valid 4-coloring: [0, 1, 2, 0]
```

## Knight's Tour

Find a sequence of moves for a knight on a chessboard to visit every square exactly once.

```python
from typing import List, Tuple

def knights_tour(n: int, start_row: int = 0, start_col: int = 0) -> List[List[int]]:
    """
    Solve the Knight's Tour problem using backtracking.

    Find a sequence of knight moves that visits every square on an n×n
    chessboard exactly once, starting from position (start_row, start_col).

    Args:
        n: Size of the chessboard (n×n)
        start_row: Starting row position (0-indexed)
        start_col: Starting column position (0-indexed)

    Returns:
        n×n board where each cell contains the move number (1 to n²)
        Empty list if no solution exists

    Time Complexity: O(8^(n²)) - at each position we try up to 8 moves
    Space Complexity: O(n²) for the board and recursion depth

    Example:
        >>> knights_tour(5)
        Board with numbers 1-25 showing the knight's path
    """
    board = [[-1 for _ in range(n)] for _ in range(n)]

    # Knight can move in 8 possible directions
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    def is_valid(row: int, col: int) -> bool:
        """
        Check if position is valid and unvisited.

        Args:
            row: Row position
            col: Column position

        Returns:
            True if position is on board and not yet visited
        """
        return (0 <= row < n and 0 <= col < n and board[row][col] == -1)

    def backtrack(row: int, col: int, move_count: int) -> bool:
        """
        Try to complete the tour from current position.

        Args:
            row: Current row
            col: Current column
            move_count: Number of moves made so far (1 to n²)

        Returns:
            True if tour can be completed from this position
        """
        # Make choice: mark current square with move number
        board[row][col] = move_count

        # Base case: all squares visited
        if move_count == n * n:
            return True

        # Try all 8 possible knight moves
        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc

            if is_valid(next_row, next_col):
                # Explore: try this move
                if backtrack(next_row, next_col, move_count + 1):
                    return True

        # Unchoose: backtrack
        board[row][col] = -1
        return False

    # Start the tour
    if backtrack(start_row, start_col, 1):
        return board
    else:
        return []

# Example usage (note: may take a while for n > 5)
n = 5
result = knights_tour(n)

if result:
    print(f"Knight's Tour on {n}×{n} board:")
    for row in result:
        print(' '.join(f'{cell:3}' for cell in row))
else:
    print(f"No solution exists for {n}×{n} board")

# For a smaller board (faster to compute):
n = 5
result = knights_tour(n, 0, 0)
if result:
    print(f"\nSolution found for {n}×{n} board")
```

## Rat in a Maze

Find a path from start to end in a maze where 1 represents walkable cells and 0 represents walls.

```python
from typing import List

def rat_in_maze(maze: List[List[int]]) -> List[List[int]]:
    """
    Find a path from top-left to bottom-right in a maze using backtracking.

    The rat can only move right or down. A cell with value 1 is walkable,
    and 0 represents a wall.

    Args:
        maze: n×n binary matrix where 1 = walkable, 0 = wall

    Returns:
        n×n matrix where 1 indicates cells in the solution path
        Empty list if no path exists

    Time Complexity: O(2^(n²)) - at each cell we have 2 choices (right/down)
    Space Complexity: O(n²) for solution matrix and recursion depth

    Example:
        >>> maze = [[1,0,0,0],
        ...         [1,1,0,1],
        ...         [0,1,0,0],
        ...         [1,1,1,1]]
        >>> rat_in_maze(maze)
        [[1,0,0,0],
         [1,1,0,0],
         [0,1,0,0],
         [0,1,1,1]]
    """
    n = len(maze)
    solution = [[0 for _ in range(n)] for _ in range(n)]

    def is_valid(row: int, col: int) -> bool:
        """
        Check if cell is valid to move to.

        Args:
            row: Row position
            col: Column position

        Returns:
            True if cell is in bounds, walkable, and not yet visited
        """
        return (0 <= row < n and
                0 <= col < n and
                maze[row][col] == 1 and
                solution[row][col] == 0)

    def backtrack(row: int, col: int) -> bool:
        """
        Try to find path from (row, col) to (n-1, n-1).

        Args:
            row: Current row
            col: Current column

        Returns:
            True if path exists from this position to destination
        """
        # Base case: reached destination
        if row == n - 1 and col == n - 1:
            solution[row][col] = 1
            return True

        # Check if current position is valid
        if is_valid(row, col):
            # Make choice: include this cell in path
            solution[row][col] = 1

            # Explore: try moving right
            if backtrack(row, col + 1):
                return True

            # Explore: try moving down
            if backtrack(row + 1, col):
                return True

            # Unchoose: backtrack
            solution[row][col] = 0
            return False

        return False

    # Start from top-left corner
    if backtrack(0, 0):
        return solution
    else:
        return []

# Example usage
maze = [
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 1, 1]
]

result = rat_in_maze(maze)
if result:
    print("Path found:")
    for row in result:
        print(row)
else:
    print("No path exists")
# Output:
# Path found:
# [1, 0, 0, 0]
# [1, 1, 0, 0]
# [0, 1, 0, 0]
# [0, 1, 1, 1]

# Version with all 4 directions (up, down, left, right)
def rat_in_maze_4dir(maze: List[List[int]]) -> List[List[int]]:
    """
    Find path allowing movement in all 4 directions (up, down, left, right).
    """
    n = len(maze)
    solution = [[0 for _ in range(n)] for _ in range(n)]

    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_valid(row: int, col: int) -> bool:
        return (0 <= row < n and 0 <= col < n and
                maze[row][col] == 1 and solution[row][col] == 0)

    def backtrack(row: int, col: int) -> bool:
        if row == n - 1 and col == n - 1:
            solution[row][col] = 1
            return True

        if is_valid(row, col):
            solution[row][col] = 1

            # Try all 4 directions
            for dr, dc in directions:
                if backtrack(row + dr, col + dc):
                    return True

            solution[row][col] = 0

        return False

    if backtrack(0, 0):
        return solution
    return []
```

## Letter Combinations of Phone Number

Generate all possible letter combinations that a phone number could represent (like old phone keypads).

```python
from typing import List

def letter_combinations(digits: str) -> List[str]:
    """
    Generate all letter combinations from a phone number string.

    Given a string containing digits from 2-9, return all possible letter
    combinations that the number could represent, based on phone keypad mapping.

    Args:
        digits: String of digits from '2' to '9'

    Returns:
        List of all possible letter combinations

    Time Complexity: O(4^n) where n is length of digits. Each digit maps to
                     3-4 letters, so we explore up to 4^n combinations.
    Space Complexity: O(n) for recursion depth and current combination

    Example:
        >>> letter_combinations("23")
        ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
        >>> letter_combinations("2")
        ['a', 'b', 'c']
    """
    if not digits:
        return []

    # Phone keypad mapping
    phone_map = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }

    result = []

    def backtrack(index: int, path: str) -> None:
        """
        Build letter combinations recursively.

        Args:
            index: Current position in the digits string
            path: Current combination being built
        """
        # Base case: processed all digits
        if index == len(digits):
            result.append(path)
            return

        # Get letters for current digit
        current_digit = digits[index]
        letters = phone_map[current_digit]

        # Try each possible letter for current digit
        for letter in letters:
            # Make choice: append this letter
            # Explore: move to next digit
            backtrack(index + 1, path + letter)
            # No explicit unchoose needed since we pass path + letter
            # (not modifying path in place)

    backtrack(0, "")
    return result

# Example usage
test_cases = ["23", "2", "234", ""]
for digits in test_cases:
    result = letter_combinations(digits)
    print(f"Combinations for '{digits}': {result}")
    print(f"  Total: {len(result)} combinations\n")

# Output:
# Combinations for '23': ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
#   Total: 9 combinations
#
# Combinations for '2': ['a', 'b', 'c']
#   Total: 3 combinations
#
# Combinations for '234': ['adg', 'adh', 'adi', ..., 'cfg', 'cfh', 'cfi']
#   Total: 27 combinations
#
# Combinations for '': []
#   Total: 0 combinations
```

## Backtracking Template

General template for backtracking problems:

```python
from typing import List, Any

def backtrack_template(input_data: Any) -> List[Any]:
    """
    Generic backtracking template for solving constraint satisfaction problems.

    This template follows the "choose → explore → unchoose" paradigm that's
    central to all backtracking algorithms.

    Args:
        input_data: Problem-specific input

    Returns:
        List of all valid solutions
    """
    result = []  # Store all valid solutions

    def is_valid_solution(state: Any) -> bool:
        """
        Check if current state represents a complete valid solution.

        For example:
        - Subsets: always True (every state is valid)
        - N-Queens: True when all N queens are placed
        - Sudoku: True when all cells are filled
        """
        pass  # Implement based on problem

    def construct_solution(state: Any) -> Any:
        """
        Convert current state to solution format.

        Often this is just copying the state, but sometimes requires
        transformation (e.g., converting 2D board to list of strings).
        """
        return state.copy()  # Or appropriate transformation

    def get_choices(state: Any) -> List[Any]:
        """
        Get all valid choices/options from current state.

        For example:
        - N-Queens: columns in current row
        - Sudoku: digits 1-9 for current empty cell
        - Permutations: remaining unused elements
        """
        pass  # Implement based on problem

    def is_valid_choice(state: Any, choice: Any) -> bool:
        """
        Check if making this choice violates any constraints.

        This is the pruning step that makes backtracking efficient.
        Return False to prune branches that can't lead to solutions.
        """
        pass  # Implement constraint checks

    def make_choice(state: Any, choice: Any) -> None:
        """
        Modify state to reflect the choice being made.

        Examples:
        - Add element to current path
        - Mark cell as used/visited
        - Assign value to variable
        """
        pass  # Implement state modification

    def undo_choice(state: Any, choice: Any) -> None:
        """
        Revert state to before the choice was made (backtrack).

        This must exactly reverse what make_choice() did.
        Critical for ensuring algorithm correctness.
        """
        pass  # Implement state restoration

    def backtrack(state: Any) -> None:
        """
        Core recursive backtracking function.

        Explores the solution space using depth-first search with pruning.
        """
        # Base case: check if we've found a complete solution
        if is_valid_solution(state):
            result.append(construct_solution(state))
            return  # Or continue if you want all solutions

        # Recursive case: try all possible choices
        for choice in get_choices(state):
            # Pruning: skip choices that violate constraints
            if not is_valid_choice(state, choice):
                continue

            # CHOOSE: make the choice and update state
            make_choice(state, choice)

            # EXPLORE: recurse with the new state
            backtrack(state)

            # UNCHOOSE: revert the choice (backtrack)
            undo_choice(state, choice)

    # Initialize state and start the search
    initial_state = initialize_state()
    backtrack(initial_state)

    return result

def initialize_state() -> Any:
    """
    Create and return the initial state for the problem.
    """
    pass  # Implement based on problem


# ===== EXAMPLE: Applying template to generate subsets =====

def subsets_template_example(nums: List[int]) -> List[List[int]]:
    """
    Example of using the template for the subsets problem.
    """
    result = []

    def backtrack(start: int, current: List[int]) -> None:
        # Every state is a valid solution for subsets
        result.append(current[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            # Make choice: include nums[i]
            current.append(nums[i])

            # Explore: continue with next elements
            backtrack(i + 1, current)

            # Unchoose: remove nums[i]
            current.pop()

    backtrack(0, [])
    return result
```

## Complexity Analysis

Backtracking algorithms typically have exponential time complexity, but the exact complexity depends on the problem structure and effectiveness of pruning.

### Time Complexity by Problem

| Problem | Time Complexity | Explanation |
|---------|----------------|-------------|
| **Subsets** | $O(n \cdot 2^n)$ | 2^n subsets, each taking O(n) to copy |
| **Permutations** | $O(n \cdot n!)$ | n! permutations, O(n) to build each |
| **N-Queens** | $O(n!)$ | Place queen in n positions row 1, ≤n-1 in row 2, etc. Pruning reduces actual runtime significantly |
| **Sudoku** | $O(9^m)$ | Try up to 9 values for each of m empty cells |
| **Combination Sum** | $O(n^{t/m})$ | n candidates, depth t/m where t=target, m=min value |
| **Word Search** | $O(m \cdot n \cdot 4^L)$ | Try each of m×n cells, explore 4 directions for L chars |
| **Palindrome Partition** | $O(n \cdot 2^n)$ | 2^n ways to partition, O(n) to check each palindrome |
| **Graph Coloring** | $O(m^n)$ | m colors, n vertices |
| **Knight's Tour** | $O(8^{n^2})$ | 8 moves per cell, n² cells (heavily pruned in practice) |
| **Letter Combinations** | $O(4^n)$ | Up to 4 letters per digit, n digits |

### Space Complexity Considerations

Space complexity for backtracking algorithms comes from:

1. **Recursion Stack**: Depth of recursion determines stack space
   - Subsets: O(n) depth
   - Permutations: O(n) depth
   - N-Queens: O(n) depth (one per row)
   - Sudoku: O(m) where m is number of empty cells
   - Word Search: O(L) where L is word length

2. **State Storage**: Space to maintain current partial solution
   - N-Queens: O(n²) for board
   - Sudoku: O(1) if modifying in-place, O(n²) if copying
   - Graph Coloring: O(n) for color assignments
   - Rat in Maze: O(n²) for solution matrix

3. **Output Storage**: Not usually counted in space complexity analysis
   - Subsets: O(n × 2^n) to store all subsets
   - Permutations: O(n × n!) to store all permutations

### Best, Average, and Worst Cases

**Best Case**: Early pruning finds solution quickly
- Example: Sudoku with many constraints may find solution early

**Average Case**: Depends heavily on problem structure and input
- Random inputs often closer to worst case
- Real-world instances may have structure that enables pruning

**Worst Case**: Explore most/all of the search space
- Happens when pruning is ineffective
- Example: N-Queens on small boards, some Sudoku configurations

### Impact of Pruning

Effective constraint checking dramatically improves performance:
- **Without pruning**: Generate all candidates, then filter (e.g., O(n!) for N-Queens generates all permutations)
- **With pruning**: Fail fast and skip entire subtrees (e.g., O(n!) but with much smaller constant factor)

The difference can be orders of magnitude in practice, making intractable problems solvable.

## Applications

Backtracking is widely used in various applications, including:

- **Puzzle Solving**: Problems like Sudoku, N-Queens, and mazes can be efficiently solved using backtracking techniques.

- **Combinatorial Problems**: Generating permutations, combinations, and subsets of a set can be accomplished through backtracking.

- **Graph Problems**: Backtracking can be applied to find Hamiltonian paths, Eulerian paths, and other graph-related problems.

- **Constraint Satisfaction**: Solving problems with constraints like graph coloring, map coloring, and scheduling.

## Common Patterns in Backtracking

Recognizing these patterns helps identify when and how to apply backtracking:

### 1. Generate All Combinations/Permutations

**Pattern**: Need to generate all possible arrangements or selections
- **Problems**: Subsets, Permutations, Combinations, Letter Combinations
- **Key characteristic**: No strong constraints, just enumerate all possibilities
- **Template**: Track current path and remaining choices

```python
def generate_all(choices):
    result = []
    def backtrack(path, remaining):
        if <complete>:
            result.append(path.copy())
        for choice in remaining:
            path.append(choice)
            backtrack(path, new_remaining)
            path.pop()
```

### 2. Constraint Satisfaction (Find One Solution)

**Pattern**: Find any valid assignment that satisfies all constraints
- **Problems**: Sudoku, N-Queens, Graph Coloring
- **Key characteristic**: Return as soon as one solution found
- **Template**: Return True/False to propagate success upward

```python
def solve(state):
    if <complete>:
        return True
    for choice in get_choices():
        if is_valid(choice):
            make_choice(state, choice)
            if solve(state):  # Found solution!
                return True
            undo_choice(state, choice)
    return False  # No solution from this state
```

### 3. Find All Valid Solutions

**Pattern**: Find all configurations that satisfy constraints
- **Problems**: N-Queens (all solutions), Palindrome Partitioning
- **Key characteristic**: Don't stop at first solution, collect all
- **Template**: Continue search after finding solutions

```python
def find_all(state):
    if <complete> and is_valid(state):
        result.append(copy(state))
        return  # Don't return True, keep searching
    for choice in get_choices():
        if can_lead_to_solution(choice):
            make_choice(state, choice)
            find_all(state)
            undo_choice(state, choice)
```

### 4. Path Finding in Grid/Graph

**Pattern**: Find path(s) through a grid or graph
- **Problems**: Word Search, Rat in Maze, Knight's Tour
- **Key characteristic**: Mark cells as visited, unmark on backtrack
- **Template**: Use temporary marking for visited cells

```python
def find_path(row, col, target):
    if reached_target():
        return True
    mark_visited(row, col)
    for next_row, next_col in get_neighbors():
        if is_valid_and_unvisited(next_row, next_col):
            if find_path(next_row, next_col, target):
                return True
    unmark_visited(row, col)  # Backtrack
    return False
```

### 5. Optimization with Pruning

**Pattern**: Early termination when partial solution can't improve
- **Problems**: Combination Sum (prune when sum > target)
- **Key characteristic**: Prune branches that exceed bounds
- **Template**: Check bounds before recursing

```python
def backtrack(current_sum, path):
    if current_sum == target:
        result.append(path.copy())
    if current_sum > target:
        return  # Prune: no point continuing
    for choice in choices:
        backtrack(current_sum + choice, path + [choice])
```

### 6. Partition Problems

**Pattern**: Divide input into valid segments
- **Problems**: Palindrome Partitioning
- **Key characteristic**: Try all possible split points
- **Template**: Loop over possible partition positions

```python
def partition(start, path):
    if start == len(input):
        result.append(path.copy())
    for end in range(start, len(input)):
        if is_valid_segment(start, end):
            path.append(segment(start, end))
            partition(end + 1, path)
            path.pop()
```

### Pattern Selection Guide

| If you need to... | Use Pattern |
|------------------|-------------|
| Generate all combinations/subsets | Pattern 1: Generate All |
| Find one valid assignment | Pattern 2: Constraint Satisfaction |
| Find all valid assignments | Pattern 3: Find All Solutions |
| Navigate through grid/graph | Pattern 4: Path Finding |
| Optimize with early termination | Pattern 5: Pruning |
| Split input into valid parts | Pattern 6: Partition |

## Tips for Backtracking

### Strategy and Design

1. **Identify the decision space**: What choices can be made at each step?
   - For N-Queens: which column in the current row?
   - For Subsets: include or exclude current element?
   - For Permutations: which unused element next?

2. **Define constraints clearly**: What makes a solution valid or invalid?
   - Write down all constraints before coding
   - Separate hard constraints (must satisfy) from soft constraints (optimization)
   - Identify which constraints can be checked incrementally

3. **Design your state representation**: How do you track the current partial solution?
   - Implicit state (function parameters): cleaner but may copy data
   - Explicit state (modified in-place): faster but requires careful backtracking
   - Choose based on whether state is complex or simple

### Implementation

4. **Implement aggressive pruning**: Check constraints as early as possible
   - Don't wait until a complete solution to check validity
   - The earlier you prune, the more branches you eliminate
   - Example: In N-Queens, check diagonal conflicts immediately, not after placing all queens

5. **Use proper state management**: Ensure state is correctly restored when backtracking
   - If modifying state in-place: explicitly undo in backtrack step
   - If passing copies: no explicit undo needed but may be slower
   - Common bug: forgetting to restore state leads to incorrect results

6. **Order your choices wisely**: Try most promising choices first
   - In Sudoku: try cells with fewest possibilities first (most constrained)
   - In Graph Coloring: color high-degree vertices first
   - Can dramatically reduce search time in practice

7. **Consider iterative deepening**: For problems with unknown solution depth
   - Try depth 1, then 2, then 3, etc.
   - Useful when solutions exist at shallow depths
   - Avoids getting stuck in deep, fruitless branches

### Optimization

8. **Optimize with memoization**: Cache results of repeated subproblems when possible
   - Only works when subproblems overlap
   - Transitions backtracking toward dynamic programming
   - Example: Cache palindrome checks in Palindrome Partitioning

9. **Use bit manipulation for sets**: When tracking used elements
   - Faster than hash sets for small domains
   - Example: Track used numbers in Sudoku row/col/box with bitmasks
   - Operations like checking membership and toggling are O(1)

10. **Pre-process input when possible**: Set up data structures that speed up constraint checks
    - N-Queens: Track occupied columns and diagonals in sets
    - Sudoku: Maintain sets of available values per row/col/box
    - Graph Coloring: Pre-compute vertex degrees

### Debugging

11. **Visualize the search tree**: Draw out small examples by hand
    - Helps understand the order of exploration
    - Identifies where pruning should occur
    - Reveals patterns in the recursion

12. **Add instrumentation**: Count nodes visited, branches pruned
    - Helps measure effectiveness of pruning
    - Identifies performance bottlenecks
    - Useful for comparing different approaches

13. **Test incrementally**: Start with small inputs
    - Easier to trace and debug
    - Build confidence before scaling up
    - Example: Test N-Queens on 4×4 before 8×8

### Common Pitfalls

14. **Avoid infinite loops**: Ensure progress toward base case
    - In grid traversal: mark cells as visited
    - In choice enumeration: advance the index
    - In permutations: reduce the remaining set

15. **Watch out for shallow vs deep copies**:
    - `result.append(path)` stores reference (wrong!)
    - `result.append(path[:])` or `path.copy()` stores a copy (correct!)
    - Critical when path is modified after appending

16. **Don't confuse finding one vs all solutions**:
    - Finding one: return True as soon as found
    - Finding all: don't return early, collect all solutions
    - Mixing these up is a common error

## Conclusion

Backtracking is a powerful algorithmic technique that provides a systematic way to explore all possible solutions to a problem. By leveraging recursion and pruning, it can efficiently solve complex problems that would otherwise require exhaustive search methods.
