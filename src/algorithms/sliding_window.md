# Sliding Window

## Table of Contents
- [Introduction](#introduction)
- [ELI10 Explanation](#eli10-explanation)
- [Core Concepts](#core-concepts)
  - [When to Use Sliding Window](#when-to-use-sliding-window)
  - [Types of Sliding Windows](#types-of-sliding-windows)
- [Fixed-Size Window](#fixed-size-window)
- [Variable-Size Window](#variable-size-window)
- [Window with HashMap/Set](#window-with-hashmapset)
- [Window with Multiple Conditions](#window-with-multiple-conditions)
- [Advanced Patterns](#advanced-patterns)
- [Common Templates](#common-templates)
- [Interview Problems](#interview-problems)
- [Complexity Analysis](#complexity-analysis)
- [When to Use](#when-to-use)
- [Common Pitfalls](#common-pitfalls)
- [Practice Problems](#practice-problems)
- [Additional Resources](#additional-resources)

## Introduction

**Sliding Window** is a powerful technique for solving problems involving sequences (arrays, strings) where you need to track a contiguous subarray/substring. Instead of recalculating from scratch for each position (O(n²) or worse), the window "slides" across the data, maintaining state as it moves (O(n)).

**Key characteristics:**
- Works on contiguous subarrays/substrings
- Maintains a "window" of elements
- Window expands and/or contracts as needed
- Reduces time complexity from O(n²) or O(n³) to O(n)
- Often uses two pointers (start/end of window)

## ELI10 Explanation

Imagine you're looking through a window at a row of toy cars. You want to find the best group of cars (maybe the most colorful group of 3 cars, or the longest group with no duplicates).

**Fixed-size window:**
Instead of checking every possible group of 3 cars separately (which takes forever!), you:
1. Look at the first 3 cars
2. Slide your window one car to the right
3. Remove the leftmost car, add the new rightmost car
4. Repeat until you've seen all groups

**Variable-size window:**
Your window can grow and shrink!
- If you see a duplicate car, shrink the window from the left
- If all cars are unique, grow the window to the right
- Keep track of the biggest valid window you've seen

This is MUCH faster than checking every possible group!

## Core Concepts

### When to Use Sliding Window

Sliding window works when:

1. Problem involves **contiguous subarrays/substrings**
2. Looking for **maximum/minimum** size or value
3. Asking for **longest/shortest** sequence with a property
4. Need to track **running sum/product/count**
5. Can incrementally **update window state** (add/remove elements efficiently)

**Keywords to look for:**
- "contiguous subarray"
- "longest substring"
- "maximum sum of size k"
- "minimum window"
- "all valid windows"

### Types of Sliding Windows

```python
"""
1. FIXED SIZE WINDOW
   [1, 2, 3, 4, 5, 6, 7, 8]
    ^-----^        (size = 3)
       ^-----^     (slide right)
          ^-----^

2. VARIABLE SIZE WINDOW (EXPANDING)
   [1, 2, 3, 4, 5, 6, 7, 8]
    ^                (start small)
    ^-----^          (expand)
    ^---------^      (keep expanding)

3. VARIABLE SIZE WINDOW (SHRINKING)
   [1, 2, 3, 4, 5, 6, 7, 8]
    ^---------^      (start valid)
       ^------^      (shrink left)
       ^---------^   (expand right)
"""
```

## Fixed-Size Window

Window size `k` stays constant. Slide one position at a time.

### Problem: Maximum Sum Subarray of Size K

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    """
    Find maximum sum of any contiguous subarray of size k.

    Brute force: O(n*k) - recalculate sum for each position
    Sliding window: O(n) - subtract left, add right

    Time: O(n), Space: O(1)
    """
    if len(nums) < k:
        return 0

    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide window: remove left, add right
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example with trace
def max_sum_subarray_traced(nums: list[int], k: int) -> int:
    """Maximum sum with detailed trace."""
    if len(nums) < k:
        return 0

    print(f"Array: {nums}, k={k}\n")

    # First window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    print(f"Window [0:{k}] = {nums[:k]}, sum = {window_sum}")

    # Slide window
    for i in range(k, len(nums)):
        old_sum = window_sum
        removed = nums[i - k]
        added = nums[i]
        window_sum = window_sum - removed + added

        print(f"Window [{i-k+1}:{i+1}] = {nums[i-k+1:i+1]}")
        print(f"  Remove {removed}, Add {added}")
        print(f"  {old_sum} - {removed} + {added} = {window_sum}")

        max_sum = max(max_sum, window_sum)
        print(f"  Max so far: {max_sum}\n")

    return max_sum

# Test
print(max_sum_subarray([2, 1, 5, 1, 3, 2], 3))  # 9 ([5,1,3])
print()
max_sum_subarray_traced([2, 1, 5, 1, 3, 2], 3)
```

### Problem: Average of Subarrays of Size K

```python
def find_averages(nums: list[int], k: int) -> list[float]:
    """
    Calculate average of all contiguous subarrays of size k.

    Time: O(n), Space: O(n) for result
    """
    if len(nums) < k:
        return []

    result = []
    window_sum = sum(nums[:k])
    result.append(window_sum / k)

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        result.append(window_sum / k)

    return result

print(find_averages([1, 3, 2, 6, -1, 4, 1, 8, 2], 5))
# [2.2, 2.8, 2.4, 3.6, 2.8]
```

### Problem: Max of All Subarrays of Size K

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    LeetCode 239: Return max of each subarray of size k.

    Use deque to maintain potential maximums in decreasing order.
    Deque stores INDICES, not values.

    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []

    result = []
    dq = deque()  # Stores indices

    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result once we have a complete window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Example
print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
# [3, 3, 5, 5, 6, 7]
# Window: [1,3,-1] -> max=3
# Window: [3,-1,-3] -> max=3
# Window: [-1,-3,5] -> max=5
# etc.
```

### Problem: First Negative in Every Window

```python
def first_negative_in_windows(nums: list[int], k: int) -> list[int]:
    """
    Find first negative number in every window of size k.

    Use deque to track negative numbers in current window.

    Time: O(n), Space: O(k)
    """
    result = []
    negatives = deque()  # Indices of negative numbers

    for i in range(len(nums)):
        # Remove indices outside window
        while negatives and negatives[0] <= i - k:
            negatives.popleft()

        # Add current if negative
        if nums[i] < 0:
            negatives.append(i)

        # Once window is complete, get result
        if i >= k - 1:
            if negatives:
                result.append(nums[negatives[0]])
            else:
                result.append(0)  # No negative in window

    return result

print(first_negative_in_windows([12, -1, -7, 8, -15, 30, 16, 28], 3))
# [-1, -1, -7, -15, -15, 0]
```

## Variable-Size Window

Window size changes based on condition. Typically use two pointers (left/right or start/end).

### Problem: Longest Substring Without Repeating Characters

```python
def length_of_longest_substring(s: str) -> int:
    """
    LeetCode 3: Longest substring without repeating characters.

    Strategy:
    - Expand window by moving right
    - If duplicate found, shrink from left until valid
    - Track maximum window size

    Time: O(n), Space: O(min(n, m)) where m = charset size
    """
    char_index = {}  # char -> last seen index
    max_length = 0
    start = 0

    for end in range(len(s)):
        # If character seen before and inside current window
        if s[end] in char_index and char_index[s[end]] >= start:
            # Shrink window: move start past duplicate
            start = char_index[s[end]] + 1

        # Update last seen index
        char_index[s[end]] = end

        # Update max length
        max_length = max(max_length, end - start + 1)

    return max_length

# Example with trace
def length_of_longest_substring_traced(s: str) -> int:
    """Longest substring with detailed trace."""
    char_index = {}
    max_length = 0
    start = 0

    print(f"String: '{s}'\n")

    for end in range(len(s)):
        char = s[end]
        print(f"Step {end + 1}: char='{char}', end={end}")

        if char in char_index and char_index[char] >= start:
            old_start = start
            start = char_index[char] + 1
            print(f"  Duplicate! Move start from {old_start} to {start}")

        char_index[char] = end
        current_length = end - start + 1

        print(f"  Window: s[{start}:{end+1}] = '{s[start:end+1]}'")
        print(f"  Length: {current_length}")

        max_length = max(max_length, current_length)
        print(f"  Max length: {max_length}\n")

    return max_length

print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print()
length_of_longest_substring_traced("pwwkew")  # 3 ("wke")
```

### Problem: Longest Substring with At Most K Distinct Characters

```python
def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    LeetCode 340: Longest substring with at most k distinct characters.

    Strategy:
    - Expand window by adding characters
    - Track character frequencies
    - When distinct > k, shrink from left

    Time: O(n), Space: O(k)
    """
    char_count = {}
    max_length = 0
    start = 0

    for end in range(len(s)):
        # Add character to window
        char_count[s[end]] = char_count.get(s[end], 0) + 1

        # Shrink window if too many distinct characters
        while len(char_count) > k:
            char_count[s[start]] -= 1
            if char_count[s[start]] == 0:
                del char_count[s[start]]
            start += 1

        # Update max length
        max_length = max(max_length, end - start + 1)

    return max_length

print(length_of_longest_substring_k_distinct("eceba", 2))  # 3 ("ece")
print(length_of_longest_substring_k_distinct("aa", 1))     # 2 ("aa")
```

### Problem: Minimum Window Substring

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    """
    LeetCode 76: Find minimum window in s containing all chars of t.

    Strategy:
    - Expand window until valid (contains all chars of t)
    - Shrink window while maintaining validity
    - Track minimum valid window

    Time: O(|s| + |t|), Space: O(|t|)
    """
    if not s or not t or len(s) < len(t):
        return ""

    # Count characters needed
    t_count = Counter(t)
    required = len(t_count)  # Unique characters needed

    # Track characters in current window
    window_count = {}
    formed = 0  # Unique characters in window with correct frequency

    # Result: (window_length, left, right)
    result = (float('inf'), 0, 0)

    left = 0
    for right in range(len(s)):
        # Add character to window
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        # Check if this character's frequency matches requirement
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1

        # Try to shrink window while valid
        while formed == required:
            # Update result if current window is smaller
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)

            # Remove leftmost character
            char = s[left]
            window_count[char] -= 1
            if char in t_count and window_count[char] < t_count[char]:
                formed -= 1

            left += 1

    # Return empty string if no valid window found
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]

# Examples
print(min_window("ADOBECODEBANC", "ABC"))  # "BANC"
print(min_window("a", "a"))                # "a"
print(min_window("a", "aa"))               # ""
```

### Problem: Longest Substring with At Most 2 Distinct Characters

```python
def length_of_longest_substring_two_distinct(s: str) -> int:
    """
    LeetCode 159: Longest substring with at most 2 distinct chars.

    Specialized version of k distinct (k=2).

    Time: O(n), Space: O(1) - at most 2 characters stored
    """
    char_count = {}
    max_length = 0
    start = 0

    for end in range(len(s)):
        char_count[s[end]] = char_count.get(s[end], 0) + 1

        # Shrink if more than 2 distinct
        while len(char_count) > 2:
            char_count[s[start]] -= 1
            if char_count[s[start]] == 0:
                del char_count[s[start]]
            start += 1

        max_length = max(max_length, end - start + 1)

    return max_length

print(length_of_longest_substring_two_distinct("eceba"))  # 3 ("ece")
print(length_of_longest_substring_two_distinct("ccaabbb"))  # 5 ("aabbb")
```

## Window with HashMap/Set

Use additional data structures to track window state efficiently.

### Problem: Permutation in String

```python
def check_inclusion(s1: str, s2: str) -> bool:
    """
    LeetCode 567: Check if s2 contains permutation of s1.

    Strategy: Fixed window of size len(s1), check if character
    frequencies match.

    Time: O(|s1| + |s2|), Space: O(1) - at most 26 characters
    """
    if len(s1) > len(s2):
        return False

    # Count frequencies in s1
    s1_count = Counter(s1)

    # Sliding window in s2
    window_count = Counter(s2[:len(s1)])

    if window_count == s1_count:
        return True

    # Slide window
    for i in range(len(s1), len(s2)):
        # Add new character
        window_count[s2[i]] += 1

        # Remove old character
        old_char = s2[i - len(s1)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check if permutation
        if window_count == s1_count:
            return True

    return False

print(check_inclusion("ab", "eidbaooo"))  # True ("ba" is permutation)
print(check_inclusion("ab", "eidboaoo"))  # False
```

### Problem: Find All Anagrams

```python
def find_anagrams(s: str, p: str) -> list[int]:
    """
    LeetCode 438: Find all start indices of p's anagrams in s.

    Similar to permutation in string, but return all positions.

    Time: O(|s| + |p|), Space: O(1)
    """
    if len(p) > len(s):
        return []

    result = []
    p_count = Counter(p)
    window_count = Counter(s[:len(p)])

    if window_count == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        # Add new character
        window_count[s[i]] += 1

        # Remove old character
        old_char = s[i - len(p)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check if anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)

    return result

print(find_anagrams("cbaebabacd", "abc"))  # [0, 6]
# "cba" at index 0, "bac" at index 6
```

### Problem: Longest Repeating Character Replacement

```python
def character_replacement(s: str, k: int) -> int:
    """
    LeetCode 424: Longest substring with same char after k replacements.

    Strategy:
    - Track frequency of most common character in window
    - If (window_size - max_frequency) > k, shrink window
    - This means we need more than k replacements

    Time: O(n), Space: O(1) - at most 26 characters
    """
    char_count = {}
    max_length = 0
    max_count = 0  # Frequency of most common char in window
    start = 0

    for end in range(len(s)):
        # Add character to window
        char_count[s[end]] = char_count.get(s[end], 0) + 1
        max_count = max(max_count, char_count[s[end]])

        # Current window size
        window_size = end - start + 1

        # If we need more than k replacements, shrink window
        if window_size - max_count > k:
            char_count[s[start]] -= 1
            start += 1
        else:
            # Valid window, update max
            max_length = max(max_length, window_size)

    return max_length

print(character_replacement("ABAB", 2))   # 4 (replace any 2 -> "AAAA")
print(character_replacement("AABABBA", 1))  # 4 ("AABA" or "ABBA")
```

## Window with Multiple Conditions

### Problem: Subarrays with K Different Integers

```python
def subarrays_with_k_distinct(nums: list[int], k: int) -> int:
    """
    LeetCode 992: Count subarrays with exactly k distinct integers.

    Trick: at_most(k) - at_most(k-1) = exactly(k)

    Time: O(n), Space: O(k)
    """
    def at_most_k_distinct(k: int) -> int:
        """Count subarrays with at most k distinct integers."""
        count = 0
        num_count = {}
        start = 0

        for end in range(len(nums)):
            num_count[nums[end]] = num_count.get(nums[end], 0) + 1

            while len(num_count) > k:
                num_count[nums[start]] -= 1
                if num_count[nums[start]] == 0:
                    del num_count[nums[start]]
                start += 1

            # All subarrays ending at 'end' with start in [start, end]
            count += end - start + 1

        return count

    return at_most_k_distinct(k) - at_most_k_distinct(k - 1)

print(subarrays_with_k_distinct([1, 2, 1, 2, 3], 2))  # 7
# [1,2], [2,1], [1,2], [2,1], [1,2,1], [2,1,2], [1,2,1,2]
```

### Problem: Fruit Into Baskets

```python
def total_fruit(fruits: list[int]) -> int:
    """
    LeetCode 904: Maximum fruits you can collect (at most 2 types).

    This is actually: longest subarray with at most 2 distinct integers!

    Time: O(n), Space: O(1)
    """
    fruit_count = {}
    max_fruits = 0
    start = 0

    for end in range(len(fruits)):
        fruit_count[fruits[end]] = fruit_count.get(fruits[end], 0) + 1

        # More than 2 types, shrink window
        while len(fruit_count) > 2:
            fruit_count[fruits[start]] -= 1
            if fruit_count[fruits[start]] == 0:
                del fruit_count[fruits[start]]
            start += 1

        max_fruits = max(max_fruits, end - start + 1)

    return max_fruits

print(total_fruit([1, 2, 1]))        # 3
print(total_fruit([0, 1, 2, 2]))     # 3 ([1,2,2] or [2,2])
print(total_fruit([1, 2, 3, 2, 2]))  # 4 ([2,3,2,2])
```

## Advanced Patterns

### Problem: Minimum Size Subarray Sum

```python
def min_subarray_len(target: int, nums: list[int]) -> int:
    """
    LeetCode 209: Minimum length subarray with sum >= target.

    Variable window: expand until valid, then shrink to find minimum.

    Time: O(n), Space: O(1)
    """
    min_length = float('inf')
    current_sum = 0
    start = 0

    for end in range(len(nums)):
        current_sum += nums[end]

        # Shrink window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, end - start + 1)
            current_sum -= nums[start]
            start += 1

    return min_length if min_length != float('inf') else 0

print(min_subarray_len(7, [2, 3, 1, 2, 4, 3]))  # 2 ([4,3])
print(min_subarray_len(11, [1, 1, 1, 1, 1, 1, 1, 1]))  # 0
```

### Problem: Max Consecutive Ones III

```python
def longest_ones(nums: list[int], k: int) -> int:
    """
    LeetCode 1004: Longest subarray of 1s after flipping at most k 0s.

    Strategy: Track number of 0s in window. If > k, shrink.

    Time: O(n), Space: O(1)
    """
    max_length = 0
    zero_count = 0
    start = 0

    for end in range(len(nums)):
        if nums[end] == 0:
            zero_count += 1

        # Too many zeros, shrink window
        while zero_count > k:
            if nums[start] == 0:
                zero_count -= 1
            start += 1

        max_length = max(max_length, end - start + 1)

    return max_length

print(longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2))  # 6
# Flip two 0s: [1,1,1,0,0,0,1,1,1,1] -> [1,1,1,1,1,0,1,1,1,1]
#                                          ^---------^
```

### Problem: Substring with Concatenation of All Words

```python
def find_substring(s: str, words: list[str]) -> list[int]:
    """
    LeetCode 30: Find all starting indices where concatenation of all
    words appears as substring.

    All words have same length. Use sliding window with word-level steps.

    Time: O(n * m) where n = len(s), m = word_length
    Space: O(k) where k = number of words
    """
    if not s or not words:
        return []

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count

    if len(s) < total_len:
        return []

    words_freq = Counter(words)
    result = []

    # Try each possible starting position (0 to word_len-1)
    for i in range(word_len):
        start = i
        window_freq = {}
        count = 0  # Words matched in current window

        for j in range(i, len(s) - word_len + 1, word_len):
            word = s[j:j + word_len]

            # Add word to window
            if word in words_freq:
                window_freq[word] = window_freq.get(word, 0) + 1
                count += 1

                # Too many of this word, shrink window
                while window_freq[word] > words_freq[word]:
                    left_word = s[start:start + word_len]
                    window_freq[left_word] -= 1
                    count -= 1
                    start += word_len

                # Valid window found
                if count == word_count:
                    result.append(start)

            else:
                # Invalid word, reset window
                window_freq.clear()
                count = 0
                start = j + word_len

    return result

print(find_substring("barfoothefoobarman", ["foo", "bar"]))  # [0, 9]
print(find_substring("wordgoodgoodgoodbestword", ["word","good","best","word"]))  # []
```

## Common Templates

### Template 1: Fixed-Size Window

```python
def fixed_window_template(arr: list, k: int):
    """Template for fixed-size sliding window."""
    if len(arr) < k:
        return None

    # Initialize window
    window_state = initialize_state(arr[:k])
    result = window_state

    # Slide window
    for i in range(k, len(arr)):
        # Remove element leaving window
        update_state_remove(window_state, arr[i - k])

        # Add element entering window
        update_state_add(window_state, arr[i])

        # Update result
        result = update_result(result, window_state)

    return result
```

### Template 2: Variable-Size Window (Shrinkable)

```python
def variable_window_template(arr: list, condition):
    """
    Template for variable-size window.
    Expand until valid, shrink while valid.
    """
    start = 0
    best_result = initialize_result()

    for end in range(len(arr)):
        # Expand window: add arr[end]
        update_state_add(state, arr[end])

        # Shrink window while condition met
        while is_valid(state):
            best_result = update_result(best_result, start, end)

            # Shrink: remove arr[start]
            update_state_remove(state, arr[start])
            start += 1

    return best_result
```

### Template 3: Variable-Size Window (At Most K)

```python
def at_most_k_template(arr: list, k: int):
    """Template for 'at most k distinct/count' problems."""
    count_map = {}
    start = 0
    result = 0

    for end in range(len(arr)):
        # Add element
        count_map[arr[end]] = count_map.get(arr[end], 0) + 1

        # Shrink while condition violated (> k)
        while len(count_map) > k:  # or other condition
            count_map[arr[start]] -= 1
            if count_map[arr[start]] == 0:
                del count_map[arr[start]]
            start += 1

        # Update result (all windows ending at 'end')
        result += end - start + 1

    return result
```

## Complexity Analysis

| Problem Type | Time Complexity | Space Complexity | Notes |
|--------------|----------------|------------------|-------|
| Fixed window (sum/avg) | O(n) | O(1) | Each element visited once |
| Fixed window (max/min) | O(n) | O(k) | With deque |
| Variable window (distinct) | O(n) | O(k) | k = max distinct |
| Variable window (frequency) | O(n) | O(m) | m = charset size |
| At most K pattern | O(n) | O(k) | HashMap with k keys |
| Word concatenation | O(n * m) | O(k) | m = word length |

**Key advantages:**
- Reduces O(n²) or O(nk) to O(n)
- Often uses O(1) or O(k) space
- Single pass through data
- Can handle streaming data

## When to Use

**Use sliding window when:**

1. **Contiguous subarray/substring** requirement
   - "Find subarray/substring..."
   - "All windows of size k..."

2. **Optimization problems**
   - Maximum/minimum length
   - Maximum/minimum sum/product
   - Count of valid windows

3. **Can update incrementally**
   - Adding/removing element doesn't require full recalculation
   - State can be maintained efficiently

4. **Fixed or flexible window size**
   - "Windows of size k" → fixed
   - "Longest substring with..." → variable
   - "At most k distinct..." → variable

**Don't use when:**
- Need non-contiguous elements (use DP or other techniques)
- Can't incrementally update state
- Window doesn't make sense (e.g., finding median requires sorting)
- Better algorithms exist (e.g., binary search for certain problems)

## Common Pitfalls

### 1. Off-by-One Errors in Window Size

```python
# Wrong: Window size calculation
window_size = right - left  # Off by one!

# Correct:
window_size = right - left + 1  # Inclusive of both ends
```

### 2. Not Handling Empty Inputs

```python
# Wrong:
def max_sum(nums: list[int], k: int) -> int:
    window_sum = sum(nums[:k])  # Crashes if len(nums) < k
    # ...

# Correct:
def max_sum(nums: list[int], k: int) -> int:
    if not nums or len(nums) < k:
        return 0  # or appropriate default
    window_sum = sum(nums[:k])
    # ...
```

### 3. Forgetting to Shrink Window

```python
# Wrong: Window only expands, never shrinks
for end in range(len(s)):
    add_to_window(s[end])
    # Forgot to shrink when condition violated!

# Correct:
for end in range(len(s)):
    add_to_window(s[end])
    while not is_valid():  # Shrink when needed
        remove_from_window(s[start])
        start += 1
```

### 4. Using Wrong Data Structure for State

```python
# Inefficient: Array for character count
char_count = [0] * 26
# Need to iterate all 26 every time to check distinct count

# Better: HashMap
char_count = {}  # Only store characters present
# len(char_count) gives distinct count in O(1)
```

### 5. Comparing Collections Inefficiently

```python
# Inefficient: Comparing Counter objects repeatedly
s1_count = Counter(s1)
for i in range(len(s2)):
    window_count = Counter(s2[i:i+len(s1)])  # Recalculates!
    if window_count == s1_count:
        # ...

# Efficient: Update window incrementally
window_count = Counter(s2[:len(s1)])
for i in range(len(s1), len(s2)):
    window_count[s2[i]] += 1  # Add new
    window_count[s2[i - len(s1)]] -= 1  # Remove old
    if window_count == s1_count:
        # ...
```

## Practice Problems

### Easy
1. **Maximum Average Subarray I** (LeetCode 643)
2. **Defanging an IP Address** (LeetCode 1108) - not typical sliding window
3. **Contains Duplicate II** (LeetCode 219) - fixed window with set

### Medium
4. **Longest Substring Without Repeating Characters** (LeetCode 3)
5. **Minimum Size Subarray Sum** (LeetCode 209)
6. **Maximum Sliding Window** (LeetCode 239)
7. **Permutation in String** (LeetCode 567)
8. **Find All Anagrams in a String** (LeetCode 438)
9. **Longest Repeating Character Replacement** (LeetCode 424)
10. **Max Consecutive Ones III** (LeetCode 1004)
11. **Fruit Into Baskets** (LeetCode 904)
12. **Subarray Product Less Than K** (LeetCode 713)
13. **Grumpy Bookstore Owner** (LeetCode 1052)
14. **Get Equal Substrings Within Budget** (LeetCode 1208)
15. **Count Number of Nice Subarrays** (LeetCode 1248)

### Hard
16. **Minimum Window Substring** (LeetCode 76)
17. **Substring with Concatenation of All Words** (LeetCode 30)
18. **Subarrays with K Different Integers** (LeetCode 992)
19. **Sliding Window Maximum** (LeetCode 239)
20. **Smallest Range Covering Elements from K Lists** (LeetCode 632)

### Premium
21. **Longest Substring with At Most Two Distinct Characters** (LeetCode 159)
22. **Longest Substring with At Most K Distinct Characters** (LeetCode 340)

## Additional Resources

### Visualizations
- **VisuAlgo**: Sliding window animation
- **Algorithm Visualizer**: String matching with sliding window

### Tutorials
- **LeetCode Explore**: Sliding Window card
- **GeeksforGeeks**: Sliding Window Technique
- **AlgoDaily**: Sliding Window Pattern Guide

### Articles
- "Sliding Window Technique: A Powerful Tool" - Medium
- "Master the Sliding Window Pattern" - LeetCode Discuss
- "Two Pointers + Sliding Window Patterns" - Interview Prep Guide

### Practice Platforms
- LeetCode Tag: Sliding Window (80+ problems)
- AlgoExpert: Sliding Window section
- NeetCode: Sliding Window roadmap

---

**Key Takeaways:**
1. Sliding window optimizes from O(n²) or O(nk) to O(n)
2. Two main types: fixed-size and variable-size windows
3. Variable windows use two pointers with expand/shrink logic
4. Often combined with HashMap/Set for state tracking
5. "At most k" pattern: use difference of two sliding windows
6. Essential for substring/subarray optimization problems

Master sliding window patterns and you'll efficiently solve a huge class of optimization problems!
