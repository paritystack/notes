# Suffix Arrays and Suffix Trees

## Table of Contents
- [Overview](#overview)
- [Suffix Arrays](#suffix-arrays)
  - [What is a Suffix Array](#what-is-a-suffix-array)
  - [Construction Algorithms](#construction-algorithms)
  - [LCP Array](#lcp-array)
  - [Applications](#suffix-array-applications)
- [Suffix Trees](#suffix-trees)
  - [What is a Suffix Tree](#what-is-a-suffix-tree)
  - [Construction (Ukkonen's Algorithm)](#construction-ukkonens-algorithm)
  - [Applications](#suffix-tree-applications)
- [Implementation](#implementation)
- [Comparison](#comparison)
- [Common Problems](#common-problems)
- [Advanced Topics](#advanced-topics)

## Overview

**Suffix arrays** and **suffix trees** are powerful data structures for string processing. They enable efficient solutions to problems like pattern matching, longest repeated substring, and string compression.

### Key Characteristics

**Suffix Array:**
- Array of integers representing sorted suffixes
- Space: O(n) integers
- Construction: O(n log n) or O(n)
- Queries: O(m log n) for pattern matching

**Suffix Tree:**
- Tree structure storing all suffixes
- Space: O(n) nodes
- Construction: O(n) with Ukkonen's algorithm
- Queries: O(m) for pattern matching
- More versatile but more complex

### Why These Structures?

**Problems they solve efficiently:**
- Pattern matching in O(m + log n) or O(m)
- Find longest repeated substring
- Find longest common substring of two strings
- Count distinct substrings
- String compression (LZ77, Burrows-Wheeler Transform)
- Bioinformatics (DNA sequence analysis)

## Suffix Arrays

### What is a Suffix Array

A **suffix array** for string S is an array of integers representing the starting positions of all suffixes of S, sorted in lexicographical order.

**Example:**

```
String: "banana"
Index:   012345

Suffixes:
i  Suffix      Sorted
0  banana      5  a
1  anana       3  ana
2  nana        1  anana
3  ana         0  banana
4  na          4  na
5  a           2  nana

Suffix Array: [5, 3, 1, 0, 4, 2]
```

**Definition:**
```
For string S[0..n-1]:
SA[i] = starting index of the i-th smallest suffix
```

**Key Property**: Binary search on suffix array enables fast pattern matching!

### Visual Representation

```
String S = "banana$" ($ = sentinel, lexicographically smallest)

Suffixes in sorted order:
SA[0] = 6  →  $
SA[1] = 5  →  a$
SA[2] = 3  →  ana$
SA[3] = 1  →  anana$
SA[4] = 0  →  banana$
SA[5] = 4  →  na$
SA[6] = 2  →  nana$
```

### Construction Algorithms

#### Naive Construction - O(n² log n)

**Algorithm:**
1. Generate all suffixes
2. Sort them lexicographically
3. Store starting positions

```python
def build_suffix_array_naive(s):
    """
    Naive O(n² log n) construction.

    Args:
        s (str): Input string

    Returns:
        list: Suffix array

    Time: O(n² log n) - O(n log n) sorts, O(n) comparisons each
    """
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()  # O(n² log n) due to string comparisons
    return [suffix[1] for suffix in suffixes]

# Example
s = "banana"
sa = build_suffix_array_naive(s)
print(sa)  # [5, 3, 1, 0, 4, 2]
```

**Problem**: O(n) string comparisons are expensive!

#### Prefix Doubling - O(n log² n)

**Idea**: Build suffix array by doubling sorted prefix length each iteration.

**Algorithm:**
1. Sort by first character: O(n log n)
2. Sort by first 2 characters using first-char ranks: O(n log n)
3. Sort by first 4 characters using 2-char ranks: O(n log n)
4. Continue: log n iterations

```python
def build_suffix_array_doubling(s):
    """
    Prefix doubling O(n log² n) construction.

    Key idea:
    - Rank suffixes by first k characters
    - Use ranks to sort by first 2k characters
    - Double k each iteration until k ≥ n

    Time: O(n log² n)
    """
    n = len(s)
    s = s + '$'  # Add sentinel
    n += 1

    # Initial ranking by first character
    rank = [ord(c) for c in s]
    sa = list(range(n))
    k = 1

    while k < n:
        # Sort by (rank[i], rank[i+k]) pairs
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        # Compute new ranks
        new_rank = [0] * n
        new_rank[sa[0]] = 0

        for i in range(1, n):
            prev = sa[i - 1]
            curr = sa[i]

            prev_pair = (rank[prev], rank[prev + k] if prev + k < n else -1)
            curr_pair = (rank[curr], rank[curr + k] if curr + k < n else -1)

            if prev_pair == curr_pair:
                new_rank[curr] = new_rank[prev]
            else:
                new_rank[curr] = new_rank[prev] + 1

        rank = new_rank
        k *= 2

    return sa[1:]  # Remove sentinel position


# Example
s = "banana"
sa = build_suffix_array_doubling(s)
print(sa)  # [5, 3, 1, 0, 4, 2]
```

**How it works:**

```
String: "banana"

Iteration 1 (k=1): Sort by first 1 character
Ranks: [b:1, a:0, n:2, a:0, n:2, a:0]
SA after sorting: [5, 3, 1, 0, 4, 2] (positions of 'a', then 'b', then 'n')

Iteration 2 (k=2): Sort by first 2 characters
Use ranks: position i sorted by (rank[i], rank[i+1])
...

Iteration 3 (k=4): Sort by first 4 characters
...

After log(n) iterations: Full suffix array built!
```

#### DC3 / Skew Algorithm - O(n)

**Best algorithm for suffix array construction in O(n) time.**

**Key ideas:**
1. Recursively sort 2/3 of suffixes
2. Use sorted 2/3 to sort remaining 1/3
3. Merge sorted portions

**Not commonly implemented in interviews** due to complexity.

### LCP Array

**LCP (Longest Common Prefix) Array**: Stores length of longest common prefix between consecutive suffixes in sorted order.

**Definition:**
```
LCP[i] = length of longest common prefix of SA[i-1] and SA[i]
LCP[0] = 0 (by convention)
```

**Example:**

```
String: "banana"

SA:  [5, 3, 1, 0, 4, 2]
     Suffixes:
     a          (SA[0] = 5)
     ana        (SA[1] = 3)
     anana      (SA[2] = 1)
     banana     (SA[3] = 0)
     na         (SA[4] = 4)
     nana       (SA[5] = 2)

LCP: [0, 1, 3, 0, 0, 2]
     ^  ^  ^  ^  ^  ^
     |  |  |  |  |  └─ "na" vs "nana": LCP = 2
     |  |  |  |  └──── "banana" vs "na": LCP = 0
     |  |  |  └─────── "anana" vs "banana": LCP = 0
     |  |  └────────── "ana" vs "anana": LCP = 3
     |  └───────────── "a" vs "ana": LCP = 1
     └──────────────── convention
```

#### Kasai's Algorithm - O(n)

**Efficient construction of LCP array from suffix array.**

**Key insight**: If LCP of suffixes starting at positions i and j is k, then LCP of suffixes starting at i+1 and j+1 is at least k-1.

```python
def build_lcp_array(s, sa):
    """
    Build LCP array using Kasai's algorithm.

    Args:
        s (str): Input string
        sa (list): Suffix array

    Returns:
        list: LCP array

    Time: O(n)
    """
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    # Compute rank array (inverse of suffix array)
    for i in range(n):
        rank[sa[i]] = i

    k = 0  # Length of current LCP
    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue

        # Compare suffix starting at i with next suffix in sorted order
        j = sa[rank[i] + 1]

        # Extend LCP
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1

        lcp[rank[i]] = k

        # Key insight: LCP can decrease by at most 1
        if k > 0:
            k -= 1

    return lcp


# Example
s = "banana"
sa = [5, 3, 1, 0, 4, 2]
lcp = build_lcp_array(s, sa)
print(lcp)  # [0, 1, 3, 0, 0, 2]
```

**Why O(n)?**
- k can increase at most n times total (bounded by string length)
- k can decrease at most n times total
- Each character compared at most twice
- Total: O(n)

### Suffix Array Applications

#### 1. Pattern Matching

**Problem**: Find all occurrences of pattern P in text T

**Solution**: Binary search on suffix array

```python
def pattern_matching(text, pattern, sa):
    """
    Find all occurrences of pattern in text using suffix array.

    Args:
        text (str): Text to search in
        pattern (str): Pattern to find
        sa (list): Suffix array of text

    Returns:
        list: Starting positions of pattern occurrences

    Time: O(m log n) where m = len(pattern), n = len(text)
    """
    def compare_with_pattern(suffix_start):
        """Compare suffix with pattern"""
        for i in range(len(pattern)):
            if suffix_start + i >= len(text):
                return -1  # Pattern longer than suffix
            if text[suffix_start + i] < pattern[i]:
                return -1
            if text[suffix_start + i] > pattern[i]:
                return 1
        return 0

    # Binary search for leftmost occurrence
    left, right = 0, len(sa)
    while left < right:
        mid = (left + right) // 2
        if compare_with_pattern(sa[mid]) < 0:
            left = mid + 1
        else:
            right = mid

    start = left

    # Binary search for rightmost occurrence
    left, right = 0, len(sa)
    while left < right:
        mid = (left + right) // 2
        if compare_with_pattern(sa[mid]) <= 0:
            left = mid + 1
        else:
            right = mid

    end = right

    # Extract all occurrences
    return [sa[i] for i in range(start, end)]


# Example
text = "banana"
pattern = "ana"
sa = [5, 3, 1, 0, 4, 2]
positions = pattern_matching(text, pattern, sa)
print(f"Pattern '{pattern}' found at positions: {positions}")  # [1, 3]
```

#### 2. Longest Repeated Substring

**Problem**: Find longest substring that appears at least twice in string

**Solution**: Maximum value in LCP array

```python
def longest_repeated_substring(s):
    """
    Find longest repeated substring.

    Args:
        s (str): Input string

    Returns:
        str: Longest repeated substring

    Time: O(n log n) for SA construction + O(n) for LCP
    """
    sa = build_suffix_array_doubling(s)
    lcp = build_lcp_array(s, sa)

    # Find maximum LCP value
    max_lcp = max(lcp)
    max_idx = lcp.index(max_lcp)

    # Extract substring
    start = sa[max_idx]
    return s[start:start + max_lcp]


# Example
s = "banana"
result = longest_repeated_substring(s)
print(f"Longest repeated substring: '{result}'")  # "ana"
```

**Why it works:**
- Repeated substring appears as prefix of multiple suffixes
- These suffixes are adjacent in sorted suffix array
- LCP of adjacent suffixes = length of common prefix = repeated substring

#### 3. Count Distinct Substrings

**Problem**: Count number of distinct substrings

**Solution**: Use formula involving LCP array

```python
def count_distinct_substrings(s):
    """
    Count number of distinct substrings.

    Formula: Total substrings - Duplicate substrings
           = n*(n+1)/2 - sum(LCP)

    Args:
        s (str): Input string

    Returns:
        int: Number of distinct substrings

    Time: O(n log n)
    """
    n = len(s)
    sa = build_suffix_array_doubling(s)
    lcp = build_lcp_array(s, sa)

    # Total possible substrings
    total = n * (n + 1) // 2

    # Subtract duplicates (sum of LCP values)
    duplicates = sum(lcp)

    return total - duplicates


# Example
s = "banana"
count = count_distinct_substrings(s)
print(f"Distinct substrings: {count}")  # 15
```

**Explanation:**
- Each suffix contributes len(suffix) substrings
- But substrings matching LCP with previous suffix are duplicates
- Formula accounts for all unique substrings

#### 4. Longest Common Substring of Two Strings

**Problem**: Find longest substring common to both strings

**Solution**: Build suffix array of concatenated string

```python
def longest_common_substring(s1, s2):
    """
    Find longest common substring of two strings.

    Args:
        s1, s2 (str): Input strings

    Returns:
        str: Longest common substring

    Time: O((n+m) log(n+m))
    """
    # Concatenate with separator
    separator = '#'  # Character not in either string
    combined = s1 + separator + s2
    n1, n2 = len(s1), len(s2)

    # Build suffix array and LCP
    sa = build_suffix_array_doubling(combined)
    lcp = build_lcp_array(combined, sa)

    # Find max LCP where suffixes come from different strings
    max_lcp = 0
    max_idx = 0

    for i in range(1, len(lcp)):
        # Check if adjacent suffixes from different strings
        suffix1_from_s1 = sa[i - 1] < n1
        suffix2_from_s1 = sa[i] < n1

        if suffix1_from_s1 != suffix2_from_s1:  # Different strings
            if lcp[i] > max_lcp:
                max_lcp = lcp[i]
                max_idx = i

    # Extract substring
    start = sa[max_idx] if sa[max_idx] < n1 else sa[max_idx] - n1 - 1
    original_string = s1 if sa[max_idx] < n1 else s2
    return original_string[start:start + max_lcp]


# Example
s1 = "banana"
s2 = "ananas"
result = longest_common_substring(s1, s2)
print(f"Longest common substring: '{result}'")  # "anana"
```

## Suffix Trees

### What is a Suffix Tree

A **suffix tree** for string S is a compressed trie (prefix tree) of all suffixes of S.

**Properties:**
- Each edge labeled with substring of S
- Each path from root represents a suffix
- Leaves correspond to suffixes
- Internal nodes have at least 2 children

**Example:**

```
String: "banana$"

Suffix Tree:
                    root
            /       |      \
           $        a       banana$
                 /     \
               na$      $
               /
              na$

More detailed:
                        root
                    /    |    \
                   $     a     banana$
                        / \
                   na$-3  $-5
                      /
                na$-1,4
```

### Space Optimization: Edge Compression

**Key idea**: Store edge labels as (start, end) indices into original string instead of actual substrings.

```python
class SuffixTreeNode:
    def __init__(self):
        self.children = {}  # char -> child node
        self.start = -1     # Edge label start index
        self.end = None     # Edge label end index (pointer for leaves)
        self.suffix_link = None  # For Ukkonen's algorithm
        self.suffix_index = -1   # Leaf: suffix starting position
```

**Space**: O(n) nodes, O(n) total edge label storage

### Construction (Ukkonen's Algorithm)

**Ukkonen's algorithm** builds suffix tree in O(n) time using online construction.

**Key ideas:**
1. **Build incrementally**: Add one character at a time
2. **Implicit representation**: Lazy evaluation of leaves
3. **Suffix links**: Fast navigation between related suffixes
4. **Active point**: Track construction state

**Algorithm outline:**

```
For each character c in string:
    1. Extend all existing leaves (implicit via end pointer)
    2. Add new leaves for suffixes ending at c
    3. Use suffix links to efficiently navigate
    4. Update active point
```

**Implementation** (simplified):

```python
class SuffixTree:
    """
    Suffix tree using Ukkonen's algorithm.
    Simplified implementation for understanding.
    """

    class Node:
        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.children = {}
            self.suffix_link = None
            self.suffix_index = -1

    def __init__(self, text):
        self.text = text + '$'  # Add terminator
        self.root = self.Node(-1, -1)
        self.root.suffix_link = self.root
        self._build()

    def _build(self):
        """
        Build suffix tree using Ukkonen's algorithm.
        Simplified version - full implementation is complex.
        """
        # Active point tracks where to insert next
        active_node = self.root
        active_edge = -1
        active_length = 0

        # Remaining suffix count
        remaining = 0

        # Build tree character by character
        for i, char in enumerate(self.text):
            remaining += 1

            # ... complex logic for:
            # - Extending existing suffixes
            # - Adding new suffixes
            # - Following suffix links
            # - Rule 1: Leaf extension (implicit)
            # - Rule 2: New branch creation
            # - Rule 3: Suffix already exists

            # Simplified: Just show structure
            pass

    def search(self, pattern):
        """Search for pattern in suffix tree. Time: O(m)"""
        node = self.root
        i = 0

        while i < len(pattern):
            char = pattern[i]
            if char not in node.children:
                return False

            child = node.children[char]
            edge_length = child.end - child.start + 1

            # Match along edge
            for j in range(edge_length):
                if i >= len(pattern):
                    return True
                if pattern[i] != self.text[child.start + j]:
                    return False
                i += 1

            node = child

        return True
```

**Full Ukkonen's algorithm is complex** - typically 200+ lines. For interviews, understanding the concept is usually sufficient.

### Suffix Tree Applications

#### 1. Fast Pattern Matching

**Time**: O(m) where m = pattern length

```python
def pattern_match_suffix_tree(suffix_tree, pattern):
    """
    Search pattern in O(m) time.
    Much faster than suffix array's O(m log n).
    """
    return suffix_tree.search(pattern)
```

#### 2. Longest Repeated Substring

**Solution**: Find deepest internal node

```python
def longest_repeated_substring_tree(suffix_tree):
    """
    Find longest repeated substring using suffix tree.
    = Deepest internal node
    """
    def dfs(node, depth):
        if not node.children:  # Leaf
            return "", 0

        max_substring = ""
        max_depth = depth

        for child in node.children.values():
            edge_len = child.end - child.start + 1
            substr, d = dfs(child, depth + edge_len)

            if d > max_depth:
                max_depth = d
                max_substring = suffix_tree.text[child.start:child.end+1] + substr

        return max_substring, max_depth

    result, _ = dfs(suffix_tree.root, 0)
    return result
```

#### 3. All Occurrences in O(m + occ)

**occ** = number of occurrences

```python
def find_all_occurrences(suffix_tree, pattern):
    """
    Find all occurrences of pattern.
    Time: O(m + occ)
    """
    # Step 1: Find pattern node (O(m))
    node = suffix_tree.traverse(pattern)

    if not node:
        return []

    # Step 2: Collect all leaf nodes under this node (O(occ))
    def collect_leaves(node):
        if node.suffix_index != -1:  # Leaf
            return [node.suffix_index]

        indices = []
        for child in node.children.values():
            indices.extend(collect_leaves(child))
        return indices

    return collect_leaves(node)
```

## Implementation

### Complete Suffix Array Implementation

```python
def build_suffix_array(s):
    """
    Build suffix array using prefix doubling.
    Time: O(n log² n)
    Space: O(n)
    """
    n = len(s)
    s = s + chr(0)  # Add smallest character
    n += 1

    rank = [ord(c) for c in s]
    sa = list(range(n))
    k = 1

    while k < n:
        # Sort by (rank[i], rank[i+k])
        sa.sort(key=lambda i: (rank[i], rank[i+k] if i+k < n else -1))

        # Update ranks
        new_rank = [0] * n
        for i in range(1, n):
            prev_pair = (rank[sa[i-1]], rank[sa[i-1]+k] if sa[i-1]+k < n else -1)
            curr_pair = (rank[sa[i]], rank[sa[i]+k] if sa[i]+k < n else -1)
            new_rank[sa[i]] = new_rank[sa[i-1]] + (prev_pair != curr_pair)

        rank = new_rank
        k *= 2

    return sa[1:]  # Remove sentinel


def build_lcp(s, sa):
    """Build LCP array using Kasai's algorithm"""
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    for i in range(n):
        rank[sa[i]] = i

    k = 0
    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue

        j = sa[rank[i] + 1]
        while i+k < n and j+k < n and s[i+k] == s[j+k]:
            k += 1

        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    return lcp


# Complete example usage
class SuffixArrayDS:
    """Complete suffix array data structure"""

    def __init__(self, text):
        self.text = text
        self.sa = build_suffix_array(text)
        self.lcp = build_lcp(text, self.sa)

    def search(self, pattern):
        """Find all occurrences of pattern"""
        def compare(idx):
            for i in range(len(pattern)):
                if idx + i >= len(self.text):
                    return -1
                if self.text[idx + i] < pattern[i]:
                    return -1
                if self.text[idx + i] > pattern[i]:
                    return 1
            return 0

        # Binary search for range
        left = 0
        right = len(self.sa)

        while left < right:
            mid = (left + right) // 2
            if compare(self.sa[mid]) < 0:
                left = mid + 1
            else:
                right = mid

        start = left

        left = 0
        right = len(self.sa)
        while left < right:
            mid = (left + right) // 2
            if compare(self.sa[mid]) <= 0:
                left = mid + 1
            else:
                right = mid

        end = right

        return [self.sa[i] for i in range(start, end)]

    def longest_repeated_substring(self):
        """Find longest repeated substring"""
        if not self.lcp:
            return ""

        max_lcp = max(self.lcp)
        idx = self.lcp.index(max_lcp)
        start = self.sa[idx]
        return self.text[start:start + max_lcp]

    def count_distinct_substrings(self):
        """Count distinct substrings"""
        n = len(self.text)
        total = n * (n + 1) // 2
        return total - sum(self.lcp)


# Usage
text = "banana"
sa_ds = SuffixArrayDS(text)

print("Suffix Array:", sa_ds.sa)
print("LCP Array:", sa_ds.lcp)
print("Search 'ana':", sa_ds.search("ana"))
print("Longest repeated:", sa_ds.longest_repeated_substring())
print("Distinct substrings:", sa_ds.count_distinct_substrings())
```

### JavaScript Implementation

```javascript
function buildSuffixArray(s) {
    const n = s.length;
    s = s + String.fromCharCode(0);
    let rank = Array.from(s, c => c.charCodeAt(0));
    let sa = Array.from({length: n + 1}, (_, i) => i);
    let k = 1;

    while (k < n + 1) {
        sa.sort((a, b) => {
            const rankA = [rank[a], a + k < n + 1 ? rank[a + k] : -1];
            const rankB = [rank[b], b + k < n + 1 ? rank[b + k] : -1];
            return rankA[0] - rankB[0] || rankA[1] - rankB[1];
        });

        const newRank = new Array(n + 1);
        newRank[sa[0]] = 0;

        for (let i = 1; i < n + 1; i++) {
            const prev = sa[i - 1], curr = sa[i];
            const prevPair = [rank[prev], prev + k < n + 1 ? rank[prev + k] : -1];
            const currPair = [rank[curr], curr + k < n + 1 ? rank[curr + k] : -1];

            newRank[curr] = newRank[prev] +
                (prevPair[0] !== currPair[0] || prevPair[1] !== currPair[1] ? 1 : 0);
        }

        rank = newRank;
        k *= 2;
    }

    return sa.slice(1);
}

function buildLCP(s, sa) {
    const n = s.length;
    const rank = new Array(n);
    const lcp = new Array(n);

    for (let i = 0; i < n; i++) {
        rank[sa[i]] = i;
    }

    let k = 0;
    for (let i = 0; i < n; i++) {
        if (rank[i] === n - 1) {
            k = 0;
            continue;
        }

        const j = sa[rank[i] + 1];
        while (i + k < n && j + k < n && s[i + k] === s[j + k]) {
            k++;
        }

        lcp[rank[i]] = k;
        if (k > 0) k--;
    }

    return lcp;
}

// Usage
const text = "banana";
const sa = buildSuffixArray(text);
const lcp = buildLCP(text, sa);
console.log("Suffix Array:", sa);
console.log("LCP Array:", lcp);
```

## Comparison

### Suffix Array vs Suffix Tree

| Feature | Suffix Array | Suffix Tree |
|---------|--------------|-------------|
| **Space** | O(n) integers (~4n bytes) | O(n) nodes (~20n bytes) |
| **Construction** | O(n log n) simple, O(n) complex | O(n) Ukkonen |
| **Pattern match** | O(m log n) | O(m) |
| **Implementation** | Simpler | Complex |
| **Memory access** | Better cache locality | More pointer chasing |
| **Queries** | Less versatile | More versatile |
| **Practical use** | Preferred in most cases | When O(m) query needed |

**When to use Suffix Array:**
- Space is limited
- O(m log n) pattern matching acceptable
- Simpler implementation preferred
- Better cache performance desired

**When to use Suffix Tree:**
- Need O(m) pattern matching
- Require complex string queries
- Space is not a concern
- Need rich structure (LCA, etc.)

### Both vs Alternatives

| Problem | Suffix Array/Tree | Alternative |
|---------|------------------|-------------|
| Pattern matching | O(m log n) / O(m) | KMP: O(n + m) |
| Repeated substring | O(n log n) | DP: O(n²) |
| Distinct substrings | O(n log n) | Trie: O(n²) space |
| LCS of 2 strings | O(n log n) | DP: O(n²) |

**Advantage**: Suffix structures solve multiple problems efficiently once built.

## Common Problems

### LeetCode Problems

| Problem | Difficulty | Technique |
|---------|-----------|-----------|
| [28. Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/) | Easy | Pattern matching |
| [1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/) | Hard | Suffix array + LCP |
| [1698. Number of Distinct Substrings in a String](https://leetcode.com/problems/number-of-distinct-substrings-in-a-string/) | Hard | Suffix array formula |
| [1923. Longest Common Subpath](https://leetcode.com/problems/longest-common-subpath/) | Hard | Suffix array + binary search |

### Competitive Programming

Common patterns:
1. **Longest repeated substring with at least k occurrences**
2. **Number of substrings matching pattern**
3. **Lexicographically kth substring**
4. **Burrows-Wheeler Transform** (compression)

## Advanced Topics

### Generalized Suffix Array/Tree

**Problem**: Build suffix structure for multiple strings

**Solution**: Concatenate strings with unique separators

```
String 1: "banana"
String 2: "ananas"
Combined: "banana#ananas$"

Build suffix array/tree on combined string
Query which string each suffix comes from
```

**Applications:**
- Multiple pattern matching
- Longest common substring of k strings
- Document searching

### Compressed Suffix Arrays

**Space optimization** techniques:
1. **ψ-function**: Compress SA to O(n log log n) bits
2. **Wavelet trees**: Compressed pattern matching
3. **FM-index**: Used in Burrows-Wheeler Transform

**Trade-off**: Less space, slightly slower queries

### Applications in Bioinformatics

**DNA sequence analysis:**
- Find repeated sequences
- Identify mutations
- Sequence alignment
- Genome assembly

**Example**: Find all tandem repeats (repeated adjacent patterns)

### Burrows-Wheeler Transform (BWT)

**Uses suffix array** for compression:

```python
def burrows_wheeler_transform(s):
    """
    Apply BWT using suffix array.
    Used in bzip2 compression.
    """
    s = s + '$'
    sa = build_suffix_array(s)

    # BWT is last column of sorted rotation matrix
    bwt = ''.join(s[(sa[i] - 1) % len(s)] for i in range(len(sa)))
    return bwt
```

### Suffix Array vs Hashing

**Comparison with rolling hash methods:**

| Method | Pattern Match | Space | Collisions |
|--------|--------------|-------|------------|
| Suffix Array | O(m log n) | O(n) | No |
| Rolling Hash | O(n + m) | O(1) | Possible |

**Use suffix array** when:
- Need multiple queries on same text
- Want guaranteed correctness
- Building once, querying many times

## Key Takeaways

1. **Suffix arrays** = sorted array of suffix starting positions
2. **Suffix trees** = compressed trie of all suffixes
3. **LCP array** = longest common prefixes, enables many algorithms
4. **Construction**: O(n log² n) easy, O(n) possible
5. **Pattern matching**: O(m log n) with SA, O(m) with ST
6. **Applications**: String matching, bioinformatics, compression
7. **Practical**: Suffix array preferred (space, simplicity)
8. **LCP formula**: Distinct substrings = n(n+1)/2 - Σ LCP

## When to Use

✅ **Use Suffix Array when:**
- Multiple pattern matching queries
- Finding longest repeated/common substrings
- Counting distinct substrings
- Space efficiency matters
- Simpler implementation preferred

✅ **Use Suffix Tree when:**
- Need O(m) pattern matching
- Complex tree queries (LCA, etc.)
- Rich structure needed
- Space is not critical

❌ **Don't use when:**
- Single pattern match (use KMP/Boyer-Moore)
- Text changes frequently (rebuild expensive)
- Very short strings (overhead not worth it)

---

**Time to Implement**:
- Suffix Array: 30-45 minutes (with LCP)
- Suffix Tree: 2-3 hours (Ukkonen's)

**Most Common Interview Uses**:
- Longest repeated substring
- Pattern matching in text
- Distinct substring counting
- Longest common substring

**Pro Tip**: For interviews, master suffix array construction and LCP array. Suffix trees are often too complex to implement fully but good to understand conceptually. Focus on applications and problem patterns!
