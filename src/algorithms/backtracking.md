# Backtracking

Backtracking is a general algorithmic technique that incrementally builds candidates for solutions and abandons a candidate as soon as it is determined that it cannot lead to a valid solution. It is often used for solving constraint satisfaction problems, such as puzzles, combinatorial problems, and optimization problems.

## Key Concepts

- **Recursive Approach**: Backtracking is typically implemented using recursion. The algorithm explores each possible option and recursively attempts to build a solution. If a solution is found, it is returned; if not, the algorithm backtracks to try the next option.

- **State Space Tree**: The process of backtracking can be visualized as a tree where each node represents a state of the solution. The root node represents the initial state, and each branch represents a choice made. The leaves of the tree represent complete solutions or dead ends.

- **Pruning**: One of the key advantages of backtracking is its ability to prune the search space. If a partial solution cannot lead to a valid complete solution, the algorithm can abandon that path early, thus saving time and resources.

## N-Queens Problem

Place N queens on an N×N chessboard such that no two queens threaten each other.

```python
def solve_n_queens(n):
    def is_valid(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal (top-left)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check diagonal (top-right)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(board, row):
        if row == n:
            result.append([''.join(row) for row in board])
            return

        for col in range(n):
            if is_valid(board, row, col):
                board[row][col] = 'Q'
                backtrack(board, row + 1)
                board[row][col] = '.'  # Backtrack

    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(board, 0)
    return result

# Example usage
solutions = solve_n_queens(4)
print(f"Found {len(solutions)} solutions for 4-Queens")
for solution in solutions:
    for row in solution:
        print(row)
    print()
```

## Sudoku Solver

Solve a 9×9 Sudoku puzzle.

```python
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    for num in '123456789':
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if backtrack():
                                return True
                            board[row][col] = '.'  # Backtrack
                    return False
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
solve_sudoku(board)
```

## Generate Subsets

Generate all subsets (power set) of a given set.

```python
def subsets(nums):
    result = []

    def backtrack(start, path):
        # Add current subset to result
        result.append(path[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()  # Backtrack

    backtrack(0, [])
    return result

# Example usage
nums = [1, 2, 3]
print(subsets(nums))
# Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

## Generate Permutations

Generate all permutations of a given list.

```python
def permute(nums):
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return

        for i in range(len(remaining)):
            # Choose
            path.append(remaining[i])
            # Explore
            backtrack(path, remaining[:i] + remaining[i+1:])
            # Unchoose (backtrack)
            path.pop()

    backtrack([], nums)
    return result

# Alternative implementation using swap
def permute_swap(nums):
    result = []

    def backtrack(first):
        if first == len(nums):
            result.append(nums[:])
            return

        for i in range(first, len(nums)):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first + 1)
            nums[first], nums[i] = nums[i], nums[first]  # Backtrack

    backtrack(0)
    return result

# Example usage
nums = [1, 2, 3]
print(permute(nums))
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

## Combination Sum

Find all combinations that sum to a target value.

```python
def combination_sum(candidates, target):
    result = []

    def backtrack(start, path, current_sum):
        if current_sum == target:
            result.append(path[:])
            return

        if current_sum > target:
            return  # Prune this branch

        for i in range(start, len(candidates)):
            path.append(candidates[i])
            # Can reuse same element, so pass i (not i+1)
            backtrack(i, path, current_sum + candidates[i])
            path.pop()  # Backtrack

    backtrack(0, [], 0)
    return result

# Example usage
candidates = [2, 3, 6, 7]
target = 7
print(combination_sum(candidates, target))
# Output: [[2,2,3], [7]]
```

## Word Search

Find if a word exists in a 2D board.

```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(row, col, index):
        # Found the word
        if index == len(word):
            return True

        # Out of bounds or wrong character
        if (row < 0 or row >= rows or
            col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False

        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'

        # Explore all directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))

        # Backtrack
        board[row][col] = temp

        return found

    # Try starting from each cell
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
print(word_search(board, "ABCCED"))  # True
print(word_search(board, "SEE"))     # True
print(word_search(board, "ABCB"))    # False
```

## Palindrome Partitioning

Partition a string into all possible palindrome substrings.

```python
def partition(s):
    def is_palindrome(s, left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    result = []

    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return

        for end in range(start, len(s)):
            if is_palindrome(s, start, end):
                path.append(s[start:end+1])
                backtrack(end + 1, path)
                path.pop()  # Backtrack

    backtrack(0, [])
    return result

# Example usage
s = "aab"
print(partition(s))
# Output: [["a","a","b"], ["aa","b"]]
```

## Backtracking Template

General template for backtracking problems:

```python
def backtrack_template(input_data):
    result = []

    def backtrack(state, ...):
        # Base case: valid solution found
        if is_valid_solution(state):
            result.append(construct_solution(state))
            return

        # Try all possible choices
        for choice in get_choices(state):
            # Make choice
            make_choice(state, choice)

            # Recurse with updated state
            backtrack(state, ...)

            # Undo choice (backtrack)
            undo_choice(state, choice)

    # Initialize and start backtracking
    initial_state = initialize()
    backtrack(initial_state)
    return result
```

## Time Complexity

Most backtracking algorithms have exponential time complexity:
- **Subsets**: $O(2^n)$ - each element can be included or excluded
- **Permutations**: $O(n!)$ - n choices for first, n-1 for second, etc.
- **N-Queens**: $O(n!)$ - approximately, with pruning
- **Sudoku**: $O(9^m)$ where m is number of empty cells

## Applications

Backtracking is widely used in various applications, including:

- **Puzzle Solving**: Problems like Sudoku, N-Queens, and mazes can be efficiently solved using backtracking techniques.

- **Combinatorial Problems**: Generating permutations, combinations, and subsets of a set can be accomplished through backtracking.

- **Graph Problems**: Backtracking can be applied to find Hamiltonian paths, Eulerian paths, and other graph-related problems.

- **Constraint Satisfaction**: Solving problems with constraints like graph coloring, map coloring, and scheduling.

## Tips for Backtracking

1. **Identify the decision space**: What choices can be made at each step?
2. **Define constraints**: What makes a solution valid or invalid?
3. **Implement pruning**: Abandon paths early when constraints are violated
4. **Use proper state management**: Ensure state is correctly restored when backtracking
5. **Optimize with memoization**: Cache results of repeated subproblems when possible

## Conclusion

Backtracking is a powerful algorithmic technique that provides a systematic way to explore all possible solutions to a problem. By leveraging recursion and pruning, it can efficiently solve complex problems that would otherwise require exhaustive search methods.
