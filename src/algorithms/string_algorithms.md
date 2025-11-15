# String Algorithms

## Overview

String algorithms are fundamental in computer science, powering everything from text editors to search engines. They solve problems related to pattern matching, string comparison, and text processing with varying time and space complexities.

## Table of Contents

1. [Pattern Matching](#pattern-matching)
   - [Naive Algorithm](#naive-pattern-matching)
   - [KMP (Knuth-Morris-Pratt)](#kmp-algorithm)
   - [Rabin-Karp](#rabin-karp-algorithm)
   - [Boyer-Moore](#boyer-moore-algorithm)
   - [Z-Algorithm](#z-algorithm)
2. [String Search Structures](#string-search-structures)
   - [Trie Applications](#trie-applications)
   - [Suffix Arrays](#suffix-arrays)
   - [Suffix Trees](#suffix-trees)
3. [String Comparison](#string-comparison)
   - [Longest Common Substring](#longest-common-substring)
   - [Longest Common Subsequence](#longest-common-subsequence)
   - [Edit Distance](#edit-distance)
   - [Hamming Distance](#hamming-distance)
4. [Advanced Topics](#advanced-topics)
   - [Manacher's Algorithm](#manachers-algorithm)
   - [Aho-Corasick](#aho-corasick-algorithm)
   - [Regular Expression Matching](#regular-expression-matching)
5. [Applications](#applications)
6. [Interview Patterns](#interview-patterns)

---

## Pattern Matching

Pattern matching finds occurrences of a pattern string within a text string. Different algorithms optimize for different scenarios.

### Naive Pattern Matching

**Time**: $O(n \times m)$ | **Space**: $O(1)$

The simplest approach: slide the pattern over the text and check character by character.

```python
def naive_search(text, pattern):
    """
    Find all occurrences of pattern in text using naive approach.

    Args:
        text: The text to search in
        pattern: The pattern to search for

    Returns:
        List of starting indices where pattern is found
    """
    n = len(text)
    m = len(pattern)
    result = []

    # Slide pattern over text one by one
    for i in range(n - m + 1):
        # Check if pattern matches at current position
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1

        # Pattern found at index i
        if j == m:
            result.append(i)

    return result

# Example usage
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))  # Output: [0, 9, 12]
```

**Pros:**
- Simple to implement
- No preprocessing required
- Works well for small patterns/texts

**Cons:**
- Poor performance on large texts
- Many unnecessary comparisons
- Worst case: checking every position

**Use Cases:**
- Small strings
- Educational purposes
- Quick prototypes

---

### KMP Algorithm

**Time**: $O(n + m)$ | **Space**: $O(m)$

Knuth-Morris-Pratt avoids re-checking characters by using information from previous matches.

**Key Insight**: When a mismatch occurs, the pattern itself contains information about where the next match could begin.

```python
def compute_lps(pattern):
    """
    Compute Longest Proper Prefix which is also Suffix array.

    LPS[i] = length of longest proper prefix of pattern[0..i]
              which is also a suffix of pattern[0..i]

    Args:
        pattern: Pattern string

    Returns:
        LPS array
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # Length of previous longest prefix suffix
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                # Don't increment i, try with shorter prefix
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text, pattern):
    """
    KMP pattern matching algorithm.

    Args:
        text: Text to search in
        pattern: Pattern to search for

    Returns:
        List of starting indices where pattern is found
    """
    n = len(text)
    m = len(pattern)

    # Preprocess pattern
    lps = compute_lps(pattern)

    result = []
    i = 0  # Index for text
    j = 0  # Index for pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            result.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return result

# Example usage
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
print(kmp_search(text, pattern))  # Output: [10]

# Understanding LPS array
pattern = "ABABCABAB"
lps = compute_lps(pattern)
print(f"Pattern: {pattern}")
print(f"LPS:     {lps}")  # [0, 0, 1, 2, 0, 1, 2, 3, 4]
```

**How LPS Works:**

```
Pattern: A B A B C A B A B
Index:   0 1 2 3 4 5 6 7 8
LPS:     0 0 1 2 0 1 2 3 4

At index 8: "ABABCABAB"
  - Longest proper prefix that is also suffix: "ABAB" (length 4)

At index 3: "ABAB"
  - Longest proper prefix that is also suffix: "AB" (length 2)
```

**Complexity Analysis:**
- Preprocessing: $O(m)$ to compute LPS
- Searching: $O(n)$ - each character examined at most twice
- Total: $O(n + m)$

**Pros:**
- Linear time complexity
- No backtracking in text
- Efficient for streaming data

**Cons:**
- Requires preprocessing
- Extra space for LPS array
- More complex than naive

**Use Cases:**
- Large text searches
- Real-time text processing
- When pattern is reused multiple times

---

### Rabin-Karp Algorithm

**Time**: $O(n + m)$ average, $O(n \times m)$ worst | **Space**: $O(1)$

Uses hashing to find pattern matches. Compares hash values instead of character-by-character comparison.

**Key Insight**: Use rolling hash to compute next hash in $O(1)$ time.

```python
def rabin_karp(text, pattern, prime=101):
    """
    Rabin-Karp algorithm using rolling hash.

    Args:
        text: Text to search in
        pattern: Pattern to search for
        prime: Prime number for hashing

    Returns:
        List of starting indices where pattern is found
    """
    n = len(text)
    m = len(pattern)
    d = 256  # Number of characters in input alphabet

    pattern_hash = 0  # Hash value for pattern
    text_hash = 0     # Hash value for current window of text
    h = 1             # Hash multiplier
    result = []

    # Calculate h = d^(m-1) % prime
    for i in range(m - 1):
        h = (h * d) % prime

    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime

    # Slide pattern over text
    for i in range(n - m + 1):
        # Check if hash values match
        if pattern_hash == text_hash:
            # Verify character by character (handle hash collisions)
            if text[i:i + m] == pattern:
                result.append(i)

        # Calculate hash for next window
        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime

            # Handle negative hash
            if text_hash < 0:
                text_hash += prime

    return result

# Example usage
text = "GEEKS FOR GEEKS"
pattern = "GEEK"
print(rabin_karp(text, pattern))  # Output: [0, 10]
```

**Rolling Hash Explained:**

```
Text: "ABCDE", Pattern: "BC"
Window size m = 2, d = 256, prime = 101

Initial hash for "AB":
  hash = (256 * ord('A') + ord('B')) % 101

Rolling to "BC":
  Remove 'A': hash = hash - ord('A') * 256^1
  Shift:      hash = hash * 256
  Add 'C':    hash = hash + ord('C')
  Modulo:     hash = hash % 101
```

**Advanced Rabin-Karp with Multiple Patterns:**

```python
def rabin_karp_multiple(text, patterns, prime=101):
    """
    Search for multiple patterns simultaneously.

    Args:
        text: Text to search in
        patterns: List of patterns to search for
        prime: Prime number for hashing

    Returns:
        Dictionary mapping pattern to list of indices
    """
    d = 256
    result = {pattern: [] for pattern in patterns}

    # Group patterns by length for efficiency
    patterns_by_length = {}
    for pattern in patterns:
        m = len(pattern)
        if m not in patterns_by_length:
            patterns_by_length[m] = []
        patterns_by_length[m].append(pattern)

    # Search for each group
    for m, pattern_group in patterns_by_length.items():
        # Compute hashes for all patterns of this length
        pattern_hashes = {}
        for pattern in pattern_group:
            p_hash = 0
            for char in pattern:
                p_hash = (d * p_hash + ord(char)) % prime
            pattern_hashes[p_hash] = pattern

        # Search in text
        text_hash = 0
        h = pow(d, m - 1, prime)

        # Initial hash
        for i in range(m):
            text_hash = (d * text_hash + ord(text[i])) % prime

        # Slide window
        for i in range(len(text) - m + 1):
            if text_hash in pattern_hashes:
                pattern = pattern_hashes[text_hash]
                if text[i:i + m] == pattern:
                    result[pattern].append(i)

            # Rolling hash
            if i < len(text) - m:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
                if text_hash < 0:
                    text_hash += prime

    return result

# Example
text = "AABAACAADAABAAABAA"
patterns = ["AABA", "AAC", "ABA"]
print(rabin_karp_multiple(text, patterns))
# Output: {'AABA': [0, 9, 13], 'AAC': [3], 'ABA': [1, 10, 14]}
```

**Complexity Analysis:**
- Average case: $O(n + m)$
- Worst case: $O(n \times m)$ (many hash collisions)
- Space: $O(1)$ (excluding result)

**Pros:**
- Simple to implement
- Excellent for multiple pattern search
- Good average-case performance

**Cons:**
- Hash collisions require character comparison
- Performance depends on hash function
- Worst case same as naive

**Use Cases:**
- Plagiarism detection
- Multiple pattern matching
- When patterns change frequently

---

### Boyer-Moore Algorithm

**Time**: $O(n/m)$ best, $O(n \times m)$ worst | **Space**: $O(k)$ where k is alphabet size

Searches from right to left in the pattern, allowing larger jumps when mismatches occur.

**Key Insights:**
1. **Bad Character Rule**: Skip alignments based on mismatched character
2. **Good Suffix Rule**: Skip based on matched suffix

```python
def bad_character_heuristic(pattern):
    """
    Preprocess pattern for bad character heuristic.

    Returns:
        Dictionary mapping character to its rightmost position
    """
    m = len(pattern)
    bad_char = {}

    # Fill with rightmost occurrence of each character
    for i in range(m):
        bad_char[pattern[i]] = i

    return bad_char

def boyer_moore_simple(text, pattern):
    """
    Simplified Boyer-Moore using only bad character rule.

    Args:
        text: Text to search in
        pattern: Pattern to search for

    Returns:
        List of starting indices where pattern is found
    """
    n = len(text)
    m = len(pattern)

    bad_char = bad_character_heuristic(pattern)
    result = []

    s = 0  # Shift of pattern with respect to text

    while s <= n - m:
        j = m - 1

        # Reduce j while characters match (right to left)
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # Pattern found
            result.append(s)
            # Shift pattern to align with next character
            s += (m - bad_char.get(text[s + m], -1) - 1) if s + m < n else 1
        else:
            # Shift pattern based on bad character
            s += max(1, j - bad_char.get(text[s + j], -1))

    return result

# Full Boyer-Moore with good suffix rule
def good_suffix_heuristic(pattern):
    """
    Preprocess pattern for good suffix rule.

    Returns:
        shift array for good suffix rule
    """
    m = len(pattern)
    shift = [0] * (m + 1)
    border_pos = [0] * (m + 1)

    # Initialize
    i = m
    j = m + 1
    border_pos[i] = j

    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = border_pos[j]

        i -= 1
        j -= 1
        border_pos[i] = j

    j = border_pos[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = border_pos[j]

    return shift

def boyer_moore(text, pattern):
    """
    Full Boyer-Moore algorithm with both heuristics.

    Args:
        text: Text to search in
        pattern: Pattern to search for

    Returns:
        List of starting indices where pattern is found
    """
    n = len(text)
    m = len(pattern)

    bad_char = bad_character_heuristic(pattern)
    good_suffix = good_suffix_heuristic(pattern)
    result = []

    s = 0

    while s <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            result.append(s)
            s += good_suffix[0]
        else:
            # Use maximum shift from both heuristics
            bad_char_shift = j - bad_char.get(text[s + j], -1)
            good_suffix_shift = good_suffix[j + 1]
            s += max(bad_char_shift, good_suffix_shift)

    return result

# Example usage
text = "ABAAABCDABCDABCDE"
pattern = "ABCD"
print(boyer_moore(text, pattern))  # Output: [5, 9]
```

**Bad Character Rule Example:**

```
Text:    T H I S  I S  A  T E S T
Pattern:     T E S T
              ↑
              Mismatch at 'I'

'I' not in pattern, skip entire pattern:

Text:    T H I S  I S  A  T E S T
Pattern:             T E S T
```

**Good Suffix Rule Example:**

```
Text:    A B C A B C A B D
Pattern: C A B C A B
              ↑   ↑ ↑
              Matched suffix: "CAB"

Shift to next occurrence of "CAB" with different preceding character:

Text:    A B C A B C A B D
Pattern:       C A B C A B
```

**Complexity Analysis:**
- Best case: $O(n/m)$ - can skip large sections
- Average: $O(n)$
- Worst: $O(n \times m)$ - rare in practice
- Preprocessing: $O(m + k)$ where k is alphabet size

**Pros:**
- Often fastest in practice
- Excellent for large alphabets
- Sublinear expected time

**Cons:**
- Complex implementation
- Requires significant preprocessing
- Poor for small alphabets

**Use Cases:**
- Text editors (find/replace)
- Large alphabet searches (e.g., Chinese text)
- When pattern searched repeatedly

---

### Z-Algorithm

**Time**: $O(n + m)$ | **Space**: $O(n + m)$

Computes Z-array where Z[i] is the length of longest substring starting from i which is also a prefix.

```python
def compute_z_array(s):
    """
    Compute Z-array for string s.

    Z[i] = length of longest substring starting from s[i]
           which is also a prefix of s

    Args:
        s: Input string

    Returns:
        Z-array
    """
    n = len(s)
    z = [0] * n

    # [l, r] is the rightmost segment that matches prefix
    l = r = 0

    for i in range(1, n):
        if i > r:
            # Outside current window, compute Z[i] from scratch
            l = r = i
            while r < n and s[r - l] == s[r]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            # Inside current window
            k = i - l

            if z[k] < r - i + 1:
                # Z[k] is entirely within window
                z[i] = z[k]
            else:
                # Need to check beyond window
                l = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1

    return z

def z_algorithm_search(text, pattern):
    """
    Pattern matching using Z-algorithm.

    Args:
        text: Text to search in
        pattern: Pattern to search for

    Returns:
        List of starting indices where pattern is found
    """
    # Concatenate pattern and text with separator
    concat = pattern + "$" + text
    n = len(concat)
    m = len(pattern)

    z = compute_z_array(concat)
    result = []

    # Find positions where Z-value equals pattern length
    for i in range(m + 1, n):
        if z[i] == m:
            result.append(i - m - 1)

    return result

# Example usage
text = "AABAACAADAABAAABAA"
pattern = "AABA"
print(z_algorithm_search(text, pattern))  # Output: [0, 9, 13]

# Understanding Z-array
s = "aabcaabxaaz"
z = compute_z_array(s)
print(f"String: {s}")
print(f"Z-array: {z}")
# Output: [0, 1, 0, 0, 3, 1, 0, 0, 2, 1, 0]
#
# Explanation:
# Index 0: Not computed (convention)
# Index 1: "a" matches prefix of length 1
# Index 4: "aab" matches prefix of length 3
# Index 8: "aa" matches prefix of length 2
```

**Z-Array Visualization:**

```
String:  a a b c a a b x a a z
Index:   0 1 2 3 4 5 6 7 8 9 10
Z-array: 0 1 0 0 3 1 0 0 2 1 0

At index 4:
  "aabcaab..."
   ^^^
  Matches "aab" at start (length 3)

At index 8:
  "aa..."
   ^^
  Matches "aa" at start (length 2)
```

**Applications of Z-Algorithm:**

```python
def find_all_occurrences_with_context(text, pattern):
    """
    Find pattern with surrounding context.
    """
    concat = pattern + "$" + text
    z = compute_z_array(concat)
    m = len(pattern)

    results = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            pos = i - m - 1
            # Get context (5 chars before and after)
            start = max(0, pos - 5)
            end = min(len(text), pos + m + 5)
            context = text[start:end]
            results.append({
                'position': pos,
                'context': context,
                'highlight': (pos - start, pos - start + m)
            })

    return results

# Example
text = "The quick brown fox jumps over the lazy dog"
pattern = "the"
results = find_all_occurrences_with_context(text.lower(), pattern)
for r in results:
    print(f"Position {r['position']}: ...{r['context']}...")
```

**Complexity Analysis:**
- Time: $O(n + m)$ - linear in concatenated string length
- Space: $O(n + m)$ for Z-array
- Preprocessing and searching combined in single pass

**Pros:**
- Simple to implement
- Linear time guarantee
- Useful beyond pattern matching

**Cons:**
- Requires concatenation (extra space)
- Not cache-friendly
- Less known than KMP

**Use Cases:**
- When simplicity is valued
- Finding repeating patterns
- String compression
- Periodic string detection

---

## Pattern Matching Comparison

| Algorithm | Best | Average | Worst | Space | Preprocessing | Best For |
|-----------|------|---------|-------|-------|---------------|----------|
| **Naive** | $O(n \times m)$ | $O(n \times m)$ | $O(n \times m)$ | $O(1)$ | None | Small strings |
| **KMP** | $O(n + m)$ | $O(n + m)$ | $O(n + m)$ | $O(m)$ | $O(m)$ | Streaming data |
| **Rabin-Karp** | $O(n + m)$ | $O(n + m)$ | $O(n \times m)$ | $O(1)$ | $O(m)$ | Multiple patterns |
| **Boyer-Moore** | $O(n/m)$ | $O(n)$ | $O(n \times m)$ | $O(k)$ | $O(m + k)$ | Large alphabets |
| **Z-Algorithm** | $O(n + m)$ | $O(n + m)$ | $O(n + m)$ | $O(n + m)$ | Combined | General purpose |

---

## String Search Structures

### Trie Applications

**Time**: $O(m)$ per operation | **Space**: $O(ALPHABET\_SIZE \times N \times M)$

Tries (prefix trees) excel at prefix-based operations and multiple pattern matching.

```python
class TrieNode:
    """Node in a Trie."""

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0  # Number of words ending here

class Trie:
    """
    Trie data structure for efficient string operations.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Insert word into trie. O(m) where m is word length."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.word_count += 1

    def search(self, word):
        """Search for exact word. O(m)"""
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end_of_word

    def starts_with(self, prefix):
        """Check if any word starts with prefix. O(m)"""
        node = self.root

        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]

        return True

    def find_all_with_prefix(self, prefix):
        """Find all words with given prefix. O(n) where n is total chars."""
        node = self.root

        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS to collect all words
        words = []
        self._dfs_collect(node, prefix, words)
        return words

    def _dfs_collect(self, node, current_word, words):
        """Helper for DFS word collection."""
        if node.is_end_of_word:
            words.append(current_word)

        for char, child in node.children.items():
            self._dfs_collect(child, current_word + char, words)

    def delete(self, word):
        """Delete word from trie. O(m)"""
        def _delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                node.is_end_of_word = False
                return len(node.children) == 0

            char = word[index]
            if char not in node.children:
                return False

            child = node.children[char]
            should_delete = _delete_helper(child, word, index + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end_of_word

            return False

        _delete_helper(self.root, word, 0)

# Example usage
trie = Trie()
words = ["apple", "app", "apricot", "banana", "band"]
for word in words:
    trie.insert(word)

print(trie.search("app"))  # True
print(trie.search("appl"))  # False
print(trie.starts_with("app"))  # True
print(trie.find_all_with_prefix("ap"))  # ['app', 'apple', 'apricot']
```

**Advanced Trie Applications:**

```python
class AutocompleteSystem:
    """
    Autocomplete system using Trie with frequency tracking.
    """

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
            self.frequency = 0
            self.sentence = ""

    def __init__(self, sentences, times):
        """
        Initialize with historical data.

        Args:
            sentences: List of historical sentences
            times: List of frequencies for each sentence
        """
        self.root = self.TrieNode()
        self.current_node = self.root
        self.current_sentence = ""

        # Build trie from historical data
        for sentence, freq in zip(sentences, times):
            self._insert(sentence, freq)

    def _insert(self, sentence, frequency):
        """Insert sentence with frequency."""
        node = self.root

        for char in sentence:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]

        node.is_end = True
        node.frequency += frequency
        node.sentence = sentence

    def input(self, c):
        """
        Process input character.
        Returns top 3 suggestions sorted by frequency.
        """
        if c == '#':
            # End of sentence
            self._insert(self.current_sentence, 1)
            self.current_sentence = ""
            self.current_node = self.root
            return []

        self.current_sentence += c

        # Navigate trie
        if c not in self.current_node.children:
            # No matches, create new path
            self.current_node.children[c] = self.TrieNode()

        self.current_node = self.current_node.children[c]

        # Get all sentences from this node
        sentences = []
        self._dfs_sentences(self.current_node, sentences)

        # Sort by frequency (desc) then lexicographically
        sentences.sort(key=lambda x: (-x[1], x[0]))

        # Return top 3
        return [s[0] for s in sentences[:3]]

    def _dfs_sentences(self, node, sentences):
        """Collect all sentences from node."""
        if node.is_end:
            sentences.append((node.sentence, node.frequency))

        for child in node.children.values():
            self._dfs_sentences(child, sentences)

# Example
ac = AutocompleteSystem(
    ["i love you", "island", "ironman", "i love leetcode"],
    [5, 3, 2, 2]
)

print(ac.input('i'))  # ["i love you", "island", "i love leetcode"]
print(ac.input(' '))  # ["i love you", "i love leetcode"]
print(ac.input('a'))  # []
print(ac.input('#'))  # []
```

**Trie for Pattern Matching:**

```python
def match_wildcard_trie(trie_node, pattern, index=0):
    """
    Match pattern with wildcards using trie.
    '.' matches any single character
    '*' matches any sequence of characters

    Args:
        trie_node: Current trie node
        pattern: Pattern with wildcards
        index: Current position in pattern

    Returns:
        True if pattern matches any word in trie
    """
    if index == len(pattern):
        return trie_node.is_end_of_word

    char = pattern[index]

    if char == '.':
        # Try all children
        for child in trie_node.children.values():
            if match_wildcard_trie(child, pattern, index + 1):
                return True
        return False

    elif char == '*':
        # Try matching 0 or more characters
        # Match 0 characters
        if match_wildcard_trie(trie_node, pattern, index + 1):
            return True

        # Match 1+ characters
        for child in trie_node.children.values():
            if match_wildcard_trie(child, pattern, index):
                return True
        return False

    else:
        # Regular character
        if char not in trie_node.children:
            return False
        return match_wildcard_trie(trie_node.children[char], pattern, index + 1)
```

**Complexity Analysis:**
- Insert: $O(m)$ where m is word length
- Search: $O(m)$
- Prefix search: $O(m + n)$ where n is number of results
- Space: $O(ALPHABET\_SIZE \times N \times M)$ - can be large

**Pros:**
- Fast prefix operations
- Efficient autocomplete
- Natural for dictionaries

**Cons:**
- High space complexity
- Cache-unfriendly
- Overhead for small datasets

**Use Cases:**
- Autocomplete systems
- Spell checkers
- IP routing
- Dictionary implementations

---

### Suffix Arrays

**Time**: $O(n \log n)$ construction | **Space**: $O(n)$

Suffix array is a sorted array of all suffixes of a string. Enables fast substring search.

```python
def build_suffix_array_naive(text):
    """
    Build suffix array using naive sorting.
    O(n^2 log n) due to string comparisons.

    Args:
        text: Input string

    Returns:
        Suffix array (array of starting indices)
    """
    n = len(text)
    suffixes = [(text[i:], i) for i in range(n)]
    suffixes.sort()
    return [suffix[1] for suffix in suffixes]

def build_suffix_array_efficient(text):
    """
    Build suffix array efficiently using counting sort and ranking.
    O(n log n)

    Args:
        text: Input string

    Returns:
        Suffix array
    """
    n = len(text)

    # Initial rank based on first character
    rank = [ord(c) for c in text]
    sa = list(range(n))

    k = 1
    while k < n:
        # Sort by (rank[i], rank[i+k])
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        # Recompute ranks
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

    return sa

def search_suffix_array(text, sa, pattern):
    """
    Search for pattern using suffix array (binary search).
    O(m log n) where m is pattern length, n is text length.

    Args:
        text: Original text
        sa: Suffix array
        pattern: Pattern to search

    Returns:
        List of positions where pattern occurs
    """
    n = len(text)
    m = len(pattern)

    # Binary search for first occurrence
    left, right = 0, n - 1
    start = -1

    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:sa[mid] + m]

        if suffix >= pattern:
            if suffix == pattern:
                start = mid
            right = mid - 1
        else:
            left = mid + 1

    if start == -1:
        return []

    # Binary search for last occurrence
    left, right = 0, n - 1
    end = -1

    while left <= right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:sa[mid] + m]

        if suffix <= pattern:
            if suffix == pattern:
                end = mid
            left = mid + 1
        else:
            right = mid - 1

    # Return all positions
    return [sa[i] for i in range(start, end + 1)]

# Example usage
text = "banana"
sa = build_suffix_array_efficient(text)

print("Text:", text)
print("Suffix Array:", sa)
print("\nSuffixes in sorted order:")
for i, idx in enumerate(sa):
    print(f"{i}: {text[idx:]}")

# Search for pattern
pattern = "ana"
positions = search_suffix_array(text, sa, pattern)
print(f"\nPattern '{pattern}' found at positions: {positions}")
```

**Suffix Array Output:**

```
Text: banana
Suffix Array: [5, 3, 1, 0, 4, 2]

Suffixes in sorted order:
0: a         (from index 5)
1: ana       (from index 3)
2: anana     (from index 1)
3: banana    (from index 0)
4: na        (from index 4)
5: nana      (from index 2)
```

**LCP (Longest Common Prefix) Array:**

```python
def build_lcp_array(text, sa):
    """
    Build LCP array from suffix array.
    LCP[i] = longest common prefix between sa[i] and sa[i-1]

    Time: O(n)

    Args:
        text: Original text
        sa: Suffix array

    Returns:
        LCP array
    """
    n = len(text)
    rank = [0] * n
    lcp = [0] * n

    # Compute rank (inverse of suffix array)
    for i in range(n):
        rank[sa[i]] = i

    k = 0
    for i in range(n):
        if rank[i] == n - 1:
            k = 0
            continue

        j = sa[rank[i] + 1]

        # Compute LCP between suffix i and suffix j
        while i + k < n and j + k < n and text[i + k] == text[j + k]:
            k += 1

        lcp[rank[i]] = k

        if k > 0:
            k -= 1

    return lcp

# Example
text = "banana"
sa = build_suffix_array_efficient(text)
lcp = build_lcp_array(text, sa)

print("Suffix Array:", sa)
print("LCP Array:", lcp)
print("\nSuffixes with LCP:")
for i in range(len(sa)):
    print(f"LCP={lcp[i] if i > 0 else '-'}: {text[sa[i]:]}")
```

**Applications of Suffix Arrays:**

```python
def find_longest_repeated_substring(text):
    """
    Find longest substring that appears at least twice.
    Uses suffix array + LCP array.

    Time: O(n log n)
    """
    sa = build_suffix_array_efficient(text)
    lcp = build_lcp_array(text, sa)

    # Maximum LCP value gives longest repeated substring
    max_lcp = max(lcp)
    max_idx = lcp.index(max_lcp)

    return text[sa[max_idx]:sa[max_idx] + max_lcp]

def count_distinct_substrings(text):
    """
    Count number of distinct substrings.
    Uses: total substrings - repeated substrings

    Time: O(n log n)
    """
    n = len(text)
    sa = build_suffix_array_efficient(text)
    lcp = build_lcp_array(text, sa)

    # Total possible substrings
    total = n * (n + 1) // 2

    # Subtract repeated (counted by LCP)
    repeated = sum(lcp)

    return total - repeated

# Examples
text = "banana"
print(f"Longest repeated substring: '{find_longest_repeated_substring(text)}'")
print(f"Distinct substrings: {count_distinct_substrings(text)}")
```

**Complexity Analysis:**
- Construction (naive): $O(n^2 \log n)$
- Construction (efficient): $O(n \log n)$
- Search: $O(m \log n)$ where m is pattern length
- Space: $O(n)$

**Pros:**
- Space-efficient compared to suffix trees
- Fast pattern matching
- Enables complex string algorithms

**Cons:**
- Slower construction than some alternatives
- Less intuitive than tries
- Requires sorting

**Use Cases:**
- Finding repeated substrings
- Pattern matching in DNA sequences
- Data compression
- Text indexing

---

### Suffix Trees

**Time**: $O(n)$ construction (Ukkonen's algorithm) | **Space**: $O(n)$

Suffix tree is a compressed trie of all suffixes. Enables linear-time string operations.

```python
class SuffixTreeNode:
    """Node in suffix tree."""

    def __init__(self, start, end):
        self.children = {}
        self.start = start  # Start index of edge label
        self.end = end      # End index of edge label (reference for efficiency)
        self.suffix_link = None
        self.suffix_index = -1  # For leaf nodes

class SuffixTree:
    """
    Simplified suffix tree implementation.
    (Full Ukkonen's algorithm is complex - this shows the concept)
    """

    def __init__(self, text):
        """
        Build suffix tree for text.

        Args:
            text: Input text (should end with unique character like $)
        """
        self.text = text
        self.root = SuffixTreeNode(-1, -1)
        self._build_naive()

    def _build_naive(self):
        """
        Naive construction O(n^2).
        For production, use Ukkonen's algorithm O(n).
        """
        n = len(self.text)

        for i in range(n):
            self._insert_suffix(i)

    def _insert_suffix(self, suffix_start):
        """Insert suffix starting at suffix_start."""
        node = self.root
        i = suffix_start

        while i < len(self.text):
            char = self.text[i]

            if char in node.children:
                child = node.children[char]
                # Match as much as possible
                j = child.start

                while j <= child.end.value and i < len(self.text) and \
                      self.text[j] == self.text[i]:
                    i += 1
                    j += 1

                if j <= child.end.value:
                    # Need to split edge
                    split_node = SuffixTreeNode(child.start, End(j - 1))
                    node.children[char] = split_node

                    # Update old child
                    child.start = j
                    split_node.children[self.text[j]] = child

                    # Add new leaf for remainder
                    new_leaf = SuffixTreeNode(i, End(len(self.text) - 1))
                    new_leaf.suffix_index = suffix_start
                    split_node.children[self.text[i]] = new_leaf
                    return
                else:
                    # Continue from child
                    node = child
            else:
                # Create new leaf
                leaf = SuffixTreeNode(i, End(len(self.text) - 1))
                leaf.suffix_index = suffix_start
                node.children[char] = leaf
                return

    def search(self, pattern):
        """
        Search for pattern in suffix tree.
        O(m) where m is pattern length.

        Returns:
            True if pattern exists
        """
        node = self.root
        i = 0

        while i < len(pattern):
            char = pattern[i]

            if char not in node.children:
                return False

            child = node.children[char]
            j = child.start

            # Match edge label
            while j <= child.end.value and i < len(pattern):
                if self.text[j] != pattern[i]:
                    return False
                i += 1
                j += 1

            if i < len(pattern):
                node = child

        return True

    def find_all_occurrences(self, pattern):
        """
        Find all occurrences of pattern.
        O(m + k) where k is number of occurrences.
        """
        node = self.root
        i = 0

        # Navigate to pattern node
        while i < len(pattern):
            char = pattern[i]
            if char not in node.children:
                return []

            child = node.children[char]
            j = child.start

            while j <= child.end.value and i < len(pattern):
                if self.text[j] != pattern[i]:
                    return []
                i += 1
                j += 1

            if i < len(pattern):
                node = child

        # Collect all leaf indices under this node
        occurrences = []
        self._collect_leaves(node, occurrences)
        return sorted(occurrences)

    def _collect_leaves(self, node, occurrences):
        """DFS to collect all leaf nodes."""
        if node.suffix_index != -1:
            occurrences.append(node.suffix_index)

        for child in node.children.values():
            self._collect_leaves(child, occurrences)

class End:
    """Helper class for end pointer (allows O(1) extension in Ukkonen's)."""
    def __init__(self, value):
        self.value = value

# Example usage (simplified)
text = "banana$"
st = SuffixTree(text)

print(st.search("ana"))  # True
print(st.search("nan"))  # True
print(st.search("xyz"))  # False
```

**Suffix Tree Applications:**

```python
def longest_common_substring_multiple(strings):
    """
    Find longest common substring among multiple strings.
    Uses generalized suffix tree.

    Args:
        strings: List of strings

    Returns:
        Longest common substring
    """
    # Create concatenated string with unique separators
    separators = ['#', '$', '%', '@', '&']
    concat = ""
    boundaries = []

    for i, s in enumerate(strings):
        boundaries.append(len(concat))
        concat += s + separators[i]

    # Build suffix tree (simplified - real implementation more complex)
    # For each internal node, check if it has suffixes from all strings

    # This is a conceptual implementation
    # Real implementation requires tracking which string each suffix belongs to

    return "Conceptual implementation - see full Ukkonen's algorithm"

def find_longest_palindrome_substring(text):
    """
    Find longest palindrome using suffix tree.

    Approach:
    1. Create text$reverse(text)
    2. Build suffix tree
    3. Find longest common substring that is a palindrome
    """
    reversed_text = text[::-1]
    combined = text + "$" + reversed_text + "#"

    # Build suffix tree and find LCS
    # Check if LCS is centered properly (is a palindrome)

    # Simplified - actual implementation needs careful index tracking
    pass
```

**Complexity Analysis:**
- Construction (Ukkonen's): $O(n)$
- Construction (naive): $O(n^2)$
- Search: $O(m)$ where m is pattern length
- Space: $O(n)$ but with larger constants than suffix array

**Pros:**
- Linear time construction (Ukkonen's)
- Linear time search
- Enables many linear-time string algorithms
- Intuitive structure

**Cons:**
- Complex to implement correctly
- Higher space overhead than suffix arrays
- Large constant factors

**Use Cases:**
- Longest common substring
- Longest repeated substring
- Finding all occurrences
- Bioinformatics (DNA analysis)

---

## String Comparison

### Longest Common Substring

**Time**: $O(n \times m)$ | **Space**: $O(n \times m)$ or $O(min(n,m))$ optimized

Find the longest substring that appears in both strings.

```python
def longest_common_substring(text1, text2):
    """
    Find longest common substring using dynamic programming.

    Args:
        text1, text2: Input strings

    Returns:
        Tuple of (length, substring)
    """
    n, m = len(text1), len(text2)

    # dp[i][j] = length of common substring ending at text1[i-1] and text2[j-1]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    max_length = 0
    end_pos = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    substring = text1[end_pos - max_length:end_pos]
    return max_length, substring

# Space-optimized version
def longest_common_substring_optimized(text1, text2):
    """
    Space-optimized version using only O(min(n,m)) space.
    """
    # Ensure text1 is shorter
    if len(text1) > len(text2):
        text1, text2 = text2, text1

    n, m = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    max_length = 0
    end_pos = 0

    for j in range(1, m + 1):
        for i in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[i] = prev[i - 1] + 1

                if curr[i] > max_length:
                    max_length = curr[i]
                    end_pos = i
            else:
                curr[i] = 0

        prev, curr = curr, prev

    substring = text1[end_pos - max_length:end_pos]
    return max_length, substring

# Example usage
text1 = "ABABC"
text2 = "BABCA"
length, substring = longest_common_substring(text1, text2)
print(f"LCS length: {length}, substring: '{substring}'")  # 4, "BABC"
```

**DP Table Visualization:**

```
text1 = "ABABC"
text2 = "BABCA"

    ""  B  A  B  C  A
""   0  0  0  0  0  0
A    0  0  1  0  0  1
B    0  1  0  2  0  0
A    0  0  2  0  0  1
B    0  1  0  3  0  0
C    0  0  0  0  4  0

Maximum value: 4 at position (4,3)
Substring: "BABC" (but there's an error above - let me recalculate)

Actually for "ABABC" and "BABCA":
text1 = "ABABC"
text2 = "BABCA"

    ""  B  A  B  C  A
""   0  0  0  0  0  0
A    0  0  1  0  0  1
B    0  1  0  2  0  0
A    0  0  2  0  0  1
B    0  1  0  3  0  0
C    0  0  0  0  4  0

Max = 4? Let me trace:
At (4,3): text1[3]='B', text2[2]='B' match, dp[4][3] = dp[3][2] + 1
dp[3][2]: text1[2]='A', text2[1]='A' match, dp[3][2] = dp[2][1] + 1
dp[2][1]: text1[1]='B', text2[0]='B' match, dp[2][1] = dp[1][0] + 1
dp[1][0] = 0 (base)

So dp[4][3] = 3, giving "BAB"

Let me recalculate the whole table:
```

**All Common Substrings:**

```python
def all_common_substrings(text1, text2):
    """
    Find all common substrings (not just longest).

    Returns:
        Set of all common substrings
    """
    n, m = len(text1), len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    common = set()

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                # Add substring of this length
                length = dp[i][j]
                substring = text1[i - length:i]
                common.add(substring)
            else:
                dp[i][j] = 0

    return common

# Example
text1 = "ABABC"
text2 = "BABCA"
print(all_common_substrings(text1, text2))
# {'A', 'B', 'AB', 'BA', 'ABC', 'BAB'}
```

**Complexity Analysis:**
- Time: $O(n \times m)$
- Space: $O(n \times m)$ or $O(min(n,m))$ optimized

**Use Cases:**
- Diff tools
- Plagiarism detection
- DNA sequence alignment
- File comparison

---

### Longest Common Subsequence

**Time**: $O(n \times m)$ | **Space**: $O(n \times m)$ or $O(min(n,m))$ optimized

Find the longest subsequence present in both strings (not necessarily contiguous).

```python
def longest_common_subsequence(text1, text2):
    """
    Find LCS using dynamic programming.

    Args:
        text1, text2: Input strings

    Returns:
        Tuple of (length, subsequence)
    """
    n, m = len(text1), len(text2)

    # dp[i][j] = LCS length of text1[0:i] and text2[0:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find actual subsequence
    lcs = []
    i, j = n, m

    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()
    return dp[n][m], ''.join(lcs)

# Example
text1 = "ABCDGH"
text2 = "AEDFHR"
length, lcs = longest_common_subsequence(text1, text2)
print(f"LCS length: {length}, subsequence: '{lcs}'")  # 3, "ADH"
```

**DP Table with Backtracking:**

```
text1 = "ABCDGH"
text2 = "AEDFHR"

      ""  A  E  D  F  H  R
 ""    0  0  0  0  0  0  0
 A     0  1→ 1  1  1  1  1
 B     0  1  1  1  1  1  1
 C     0  1  1  1  1  1  1
 D     0  1  1  2→ 2  2  2
 G     0  1  1  2  2  2  2
 H     0  1  1  2  2  3→ 3

Backtrack from (6,6):
- text1[5]='H' == text2[4]='H': take it, go to (5,4)
- text1[4]='G' != text2[3]='F': dp[4][4] > dp[5][3], go to (4,4)
- text1[3]='D' == text2[2]='D': take it, go to (3,2)
- text1[2]='C' != text2[1]='E': dp[2][2] = dp[3][1], go to (2,2)
- text1[1]='B' != text2[1]='E': dp[1][2] = dp[2][1], go to (1,1)
- text1[0]='A' == text2[0]='A': take it, go to (0,0)

LCS = "ADH"
```

**Space-Optimized LCS:**

```python
def lcs_length_optimized(text1, text2):
    """
    Get LCS length using O(min(n,m)) space.
    (Cannot reconstruct actual LCS with this optimization)
    """
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m = len(text2)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for char1 in text1:
        for j in range(1, m + 1):
            if char1 == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])

        prev, curr = curr, prev

    return prev[m]
```

**LCS Variants:**

```python
def lcs_of_three(text1, text2, text3):
    """
    LCS of three strings.
    Time: O(n * m * p)
    """
    n, m, p = len(text1), len(text2), len(text3)

    # 3D DP table
    dp = [[[0] * (p + 1) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, p + 1):
                if text1[i-1] == text2[j-1] == text3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(
                        dp[i-1][j][k],
                        dp[i][j-1][k],
                        dp[i][j][k-1]
                    )

    return dp[n][m][p]

def shortest_common_supersequence(text1, text2):
    """
    Find shortest string that has both text1 and text2 as subsequences.

    Length = len(text1) + len(text2) - LCS_length
    """
    lcs_len, lcs = longest_common_subsequence(text1, text2)

    # Reconstruct SCS
    result = []
    i = j = k = 0

    while k < lcs_len:
        # Add characters from text1 until we hit LCS char
        while i < len(text1) and text1[i] != lcs[k]:
            result.append(text1[i])
            i += 1

        # Add characters from text2 until we hit LCS char
        while j < len(text2) and text2[j] != lcs[k]:
            result.append(text2[j])
            j += 1

        # Add the LCS character
        result.append(lcs[k])
        i += 1
        j += 1
        k += 1

    # Add remaining characters
    result.extend(text1[i:])
    result.extend(text2[j:])

    return ''.join(result)

# Examples
print(lcs_of_three("ABCD", "ACBD", "ABAD"))  # 2 ("AB" or "AD")
print(shortest_common_supersequence("abac", "cab"))  # "cabac"
```

**Complexity Analysis:**
- Time: $O(n \times m)$
- Space: $O(n \times m)$ or $O(min(n,m))$ for length only

**Use Cases:**
- Diff algorithms (git diff)
- Version control
- Sequence alignment in bioinformatics
- Merge tools

---

### Edit Distance

**Time**: $O(n \times m)$ | **Space**: $O(n \times m)$ or $O(min(n,m))$ optimized

Minimum number of operations (insert, delete, replace) to transform one string to another. Also known as Levenshtein distance.

```python
def edit_distance(word1, word2):
    """
    Calculate minimum edit distance between two words.
    Operations: insert, delete, replace

    Args:
        word1, word2: Input strings

    Returns:
        Tuple of (distance, operations)
    """
    n, m = len(word1), len(word2)

    # dp[i][j] = edit distance between word1[0:i] and word2[0:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(n + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(m + 1):
        dp[0][j] = j  # Insert all characters

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from word1
                    dp[i][j - 1],      # Insert to word1
                    dp[i - 1][j - 1]   # Replace in word1
                )

    # Backtrack to find operations
    operations = []
    i, j = n, m

    while i > 0 or j > 0:
        if i == 0:
            operations.append(f"Insert '{word2[j-1]}'")
            j -= 1
        elif j == 0:
            operations.append(f"Delete '{word1[i-1]}'")
            i -= 1
        elif word1[i-1] == word2[j-1]:
            i -= 1
            j -= 1
        else:
            # Find which operation was used
            if dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                operations.append(f"Delete '{word1[i-1]}'")
                i -= 1
            else:
                operations.append(f"Insert '{word2[j-1]}'")
                j -= 1

    operations.reverse()
    return dp[n][m], operations

# Example
word1 = "horse"
word2 = "ros"
distance, ops = edit_distance(word1, word2)
print(f"Edit distance: {distance}")
print("Operations:")
for op in ops:
    print(f"  {op}")
```

**DP Table Visualization:**

```
word1 = "horse"
word2 = "ros"

      ""  r  o  s
 ""    0  1  2  3
 h     1  1  2  3
 o     2  2  1  2
 r     3  2  2  2
 s     4  3  3  2
 e     5  4  4  3

Operations to transform "horse" → "ros":
1. Replace 'h' with 'r': "rorse"
2. Delete 'r': "rose"
3. Delete 'e': "ros"

Total: 3 operations
```

**Edit Distance Variants:**

```python
def edit_distance_with_costs(word1, word2, insert_cost=1, delete_cost=1, replace_cost=1):
    """
    Edit distance with custom operation costs.
    """
    n, m = len(word1), len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * delete_cost
    for j in range(m + 1):
        dp[0][j] = j * insert_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + delete_cost,
                    dp[i][j - 1] + insert_cost,
                    dp[i - 1][j - 1] + replace_cost
                )

    return dp[n][m]

def damerau_levenshtein_distance(word1, word2):
    """
    Edit distance allowing transposition (swap adjacent chars).
    """
    n, m = len(word1), len(word2)

    # Create dictionary for all characters
    da = {}
    for char in word1 + word2:
        da[char] = 0

    max_dist = n + m
    H = [[max_dist] * (m + 2) for _ in range(n + 2)]
    H[0][0] = max_dist

    for i in range(0, n + 1):
        H[i + 1][0] = max_dist
        H[i + 1][1] = i
    for j in range(0, m + 1):
        H[0][j + 1] = max_dist
        H[1][j + 1] = j

    for i in range(1, n + 1):
        db = 0
        for j in range(1, m + 1):
            k = da[word2[j - 1]]
            l = db
            cost = 1

            if word1[i - 1] == word2[j - 1]:
                cost = 0
                db = j

            H[i + 1][j + 1] = min(
                H[i][j] + cost,           # Substitution
                H[i + 1][j] + 1,          # Insertion
                H[i][j + 1] + 1,          # Deletion
                H[k][l] + (i - k - 1) + 1 + (j - l - 1)  # Transposition
            )

        da[word1[i - 1]] = i

    return H[n + 1][m + 1]

# Examples
print(edit_distance_with_costs("kitten", "sitting", insert_cost=2, delete_cost=2, replace_cost=1))
print(damerau_levenshtein_distance("ca", "abc"))  # 2 (can transpose)
```

**Fuzzy String Matching:**

```python
def fuzzy_match(text, pattern, max_distance):
    """
    Find all approximate matches of pattern in text.

    Args:
        text: Text to search in
        pattern: Pattern to search for
        max_distance: Maximum allowed edit distance

    Returns:
        List of (position, distance) tuples
    """
    n = len(text)
    m = len(pattern)
    matches = []

    for i in range(n - m + 1):
        substring = text[i:i + m]
        dist, _ = edit_distance(pattern, substring)
        if dist <= max_distance:
            matches.append((i, dist))

    return matches

# Example
text = "The quick brown fox jumps"
pattern = "quack"
print(fuzzy_match(text, pattern, max_distance=2))
# [(4, 2)] - "quick" matches with distance 2
```

**Complexity Analysis:**
- Time: $O(n \times m)$
- Space: $O(n \times m)$ or $O(min(n,m))$ for distance only

**Use Cases:**
- Spell checkers
- DNA sequence alignment
- Natural language processing
- Autocorrect systems
- Fuzzy search

---

### Hamming Distance

**Time**: $O(n)$ | **Space**: $O(1)$

Number of positions at which corresponding characters differ. Only defined for equal-length strings.

```python
def hamming_distance(str1, str2):
    """
    Calculate Hamming distance between two strings.
    Strings must be of equal length.

    Args:
        str1, str2: Input strings of equal length

    Returns:
        Number of differing positions
    """
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")

    distance = 0
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            distance += 1

    return distance

# One-liner version
def hamming_distance_oneliner(str1, str2):
    """Hamming distance in one line."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Example
str1 = "karolin"
str2 = "kathrin"
print(hamming_distance(str1, str2))  # 3 (positions 1, 4, 5 differ)
```

**Hamming Distance for Binary Strings:**

```python
def hamming_distance_binary(x, y):
    """
    Hamming distance for integers (count differing bits).

    Args:
        x, y: Integers

    Returns:
        Number of bit positions where x and y differ
    """
    # XOR gives 1 where bits differ
    xor = x ^ y

    # Count number of 1s
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1

    return count

# Using built-in bit_count (Python 3.10+)
def hamming_distance_binary_fast(x, y):
    """Fast version using built-in bit count."""
    return (x ^ y).bit_count()

# Example
print(hamming_distance_binary(1, 4))  # 2 (0001 vs 0100)
print(hamming_distance_binary(3, 1))  # 1 (0011 vs 0001)
```

**Applications:**

```python
def find_similar_words(word, dictionary, max_distance):
    """
    Find words in dictionary within Hamming distance.

    Args:
        word: Target word
        dictionary: List of words (all same length as word)
        max_distance: Maximum allowed Hamming distance

    Returns:
        List of similar words
    """
    similar = []

    for dict_word in dictionary:
        if len(dict_word) == len(word):
            dist = hamming_distance(word, dict_word)
            if dist <= max_distance:
                similar.append((dict_word, dist))

    return sorted(similar, key=lambda x: x[1])

def detect_errors(transmitted, received):
    """
    Detect transmission errors using Hamming distance.

    Args:
        transmitted: Original message
        received: Received message

    Returns:
        Number of bit errors
    """
    if len(transmitted) != len(received):
        return -1  # Invalid comparison

    errors = hamming_distance(transmitted, received)
    error_positions = [i for i, (c1, c2) in enumerate(zip(transmitted, received)) if c1 != c2]

    return errors, error_positions

# Examples
dictionary = ["cat", "hat", "rat", "bat", "car", "bar"]
print(find_similar_words("cat", dictionary, max_distance=1))
# [('cat', 0), ('hat', 1), ('rat', 1), ('bat', 1), ('car', 1)]

transmitted = "10101010"
received = "10111010"
errors, positions = detect_errors(transmitted, received)
print(f"Errors: {errors} at positions {positions}")  # Errors: 1 at positions [3]
```

**Total Hamming Distance:**

```python
def total_hamming_distance(nums):
    """
    Calculate sum of Hamming distances between all pairs.

    For array of integers, sum of Hamming distances for all pairs.
    Efficient approach: O(n * k) where k is number of bits.

    Args:
        nums: List of integers

    Returns:
        Total Hamming distance
    """
    n = len(nums)
    total = 0

    # Check each bit position
    for i in range(32):  # Assuming 32-bit integers
        count_ones = 0

        # Count numbers with 1 at position i
        for num in nums:
            if num & (1 << i):
                count_ones += 1

        # Numbers with 0 at position i
        count_zeros = n - count_ones

        # Add contribution of this bit position
        total += count_ones * count_zeros

    return total

# Example
nums = [4, 14, 2]
print(total_hamming_distance(nums))  # 6
# Pairs: (4,14)=2, (4,2)=2, (14,2)=2, total=6
```

**Complexity Analysis:**
- Time: $O(n)$ for strings of length n
- Time: $O(k)$ for integers where k is number of bits
- Space: $O(1)$

**Use Cases:**
- Error detection/correction codes
- Bioinformatics (DNA sequences)
- Network transmission error detection
- Information theory
- Finding similar strings

---

## Advanced Topics

### Manacher's Algorithm

**Time**: $O(n)$ | **Space**: $O(n)$

Finds the longest palindromic substring in linear time.

**Key Insight**: Use previously computed palindrome information to avoid redundant checks.

```python
def longest_palindrome_manacher(s):
    """
    Find longest palindromic substring using Manacher's algorithm.
    O(n) time complexity.

    Args:
        s: Input string

    Returns:
        Longest palindromic substring
    """
    # Preprocess: insert '#' between characters
    # This handles even and odd length palindromes uniformly
    t = '#'.join('^{}$'.format(s))
    n = len(t)

    # P[i] = length of palindrome centered at i
    P = [0] * n
    center = right = 0

    for i in range(1, n - 1):
        # Mirror of i with respect to center
        mirror = 2 * center - i

        if i < right:
            # Use previously computed values
            P[i] = min(right - i, P[mirror])

        # Attempt to expand palindrome centered at i
        try:
            while t[i + P[i] + 1] == t[i - P[i] - 1]:
                P[i] += 1
        except IndexError:
            pass

        # If palindrome centered at i extends past right,
        # adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]

    # Find maximum element in P
    max_len = max(P)
    center_index = P.index(max_len)

    # Extract palindrome from original string
    start = (center_index - max_len) // 2
    return s[start:start + max_len]

def all_palindrome_lengths(s):
    """
    Get length of longest palindrome centered at each position.

    Returns:
        List where result[i] is length of longest palindrome centered at i
    """
    t = '#'.join('^{}$'.format(s))
    n = len(t)
    P = [0] * n
    center = right = 0

    for i in range(1, n - 1):
        mirror = 2 * center - i

        if i < right:
            P[i] = min(right - i, P[mirror])

        try:
            while t[i + P[i] + 1] == t[i - P[i] - 1]:
                P[i] += 1
        except IndexError:
            pass

        if i + P[i] > right:
            center, right = i, i + P[i]

    # Convert back to original string positions
    result = []
    for i in range(1, n - 1):
        if t[i] != '#':
            result.append(P[i])

    return result

# Example usage
s = "babad"
print(longest_palindrome_manacher(s))  # "bab" or "aba"

s = "cbbd"
print(longest_palindrome_manacher(s))  # "bb"

# All palindrome lengths
s = "abacabad"
lengths = all_palindrome_lengths(s)
print(f"String: {s}")
print(f"Palindrome lengths: {lengths}")
```

**How Manacher's Works:**

```
Original: "babad"
Processed: "^#b#a#b#a#d#$"
           0 1 2 3 4 5 6 7 8 9 10 11

P array:   0 0 1 0 3 0 1 0 1 0 0  0
           ^   b   a   b   a   d  $

At index 4 (character 'b'):
  P[4] = 3 means palindrome of length 3 on each side
  "#a#b#a#" is a palindrome
  In original string: "aba"

At index 2 (character 'b'):
  P[2] = 1 means palindrome of length 1 on each side
  "#b#" is a palindrome
  In original string: "b"
```

**Count All Palindromes:**

```python
def count_palindromic_substrings(s):
    """
    Count all palindromic substrings using Manacher's.

    Returns:
        Number of palindromic substrings
    """
    t = '#'.join('^{}$'.format(s))
    n = len(t)
    P = [0] * n
    center = right = 0

    for i in range(1, n - 1):
        mirror = 2 * center - i

        if i < right:
            P[i] = min(right - i, P[mirror])

        try:
            while t[i + P[i] + 1] == t[i - P[i] - 1]:
                P[i] += 1
        except IndexError:
            pass

        if i + P[i] > right:
            center, right = i, i + P[i]

    # Count palindromes
    # Each P[i] value contributes (P[i] + 1) // 2 palindromes
    count = 0
    for i in range(1, n - 1):
        count += (P[i] + 1) // 2

    return count

# Example
s = "aaa"
print(count_palindromic_substrings(s))  # 6: "a", "a", "a", "aa", "aa", "aaa"
```

**Complexity Analysis:**
- Time: $O(n)$ - each character expanded at most once
- Space: $O(n)$ for P array and processed string

**Pros:**
- Optimal time complexity
- Handles all palindrome queries efficiently
- Elegant algorithm

**Cons:**
- More complex than naive approach
- Requires preprocessing
- Not intuitive initially

**Use Cases:**
- Finding longest palindrome
- Counting palindromic substrings
- Competitive programming
- Interview questions

---

### Aho-Corasick Algorithm

**Time**: $O(n + m + z)$ where n is text length, m is total pattern length, z is number of matches | **Space**: $O(m)$

Efficiently finds all occurrences of multiple patterns simultaneously using trie + failure links.

```python
from collections import deque, defaultdict

class AhoCorasick:
    """
    Aho-Corasick algorithm for multiple pattern matching.
    """

    class Node:
        def __init__(self):
            self.children = {}
            self.fail = None  # Failure link
            self.output = []  # Patterns ending at this node

    def __init__(self, patterns):
        """
        Build Aho-Corasick automaton.

        Args:
            patterns: List of patterns to search for
        """
        self.root = self.Node()
        self.patterns = patterns
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self):
        """Build trie from patterns. O(m)"""
        for pattern_idx, pattern in enumerate(self.patterns):
            node = self.root

            for char in pattern:
                if char not in node.children:
                    node.children[char] = self.Node()
                node = node.children[char]

            node.output.append(pattern_idx)

    def _build_failure_links(self):
        """Build failure links using BFS. O(m)"""
        queue = deque()

        # Initialize root's children
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        # BFS to set failure links
        while queue:
            current = queue.popleft()

            for char, child in current.children.items():
                queue.append(child)

                # Find failure link
                fail_node = current.fail

                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail

                if fail_node is not None:
                    child.fail = fail_node.children[char]
                else:
                    child.fail = self.root

                # Inherit output from failure link
                child.output.extend(child.fail.output)

    def search(self, text):
        """
        Search for all patterns in text.

        Args:
            text: Text to search in

        Returns:
            Dictionary mapping pattern index to list of positions
        """
        results = defaultdict(list)
        node = self.root

        for i, char in enumerate(text):
            # Follow failure links until we find a match or reach root
            while node is not None and char not in node.children:
                node = node.fail

            if node is None:
                node = self.root
                continue

            node = node.children[char]

            # Report all patterns ending at this position
            for pattern_idx in node.output:
                pattern_len = len(self.patterns[pattern_idx])
                start_pos = i - pattern_len + 1
                results[pattern_idx].append(start_pos)

        return results

# Example usage
patterns = ["he", "she", "his", "hers"]
text = "she sells hershells by the seashore"

ac = AhoCorasick(patterns)
results = ac.search(text)

print("Pattern matches:")
for pattern_idx, positions in results.items():
    pattern = patterns[pattern_idx]
    print(f"'{pattern}': {positions}")

# Output:
# 'she': [0, 14]
# 'he': [1, 15, 27]
# 'hers': [10]
```

**Failure Link Visualization:**

```
Patterns: ["he", "she", "his", "hers"]

Trie structure:
       (root)
       /  |  \
      h   s   (others)
     /|    \
    e i     h
    |  |     \
   rs  s      e
              |
              rs

Failure links (shown with -->):
- 's' at level 1 --> root
- 'h' at level 1 --> root
- 'h' (from 's') --> 'h' at level 1
- 'e' (from 'h') --> root
- 'e' (from 'sh') --> 'e' (from 'h')

When searching "she":
1. Match 's' from root
2. Match 'h' from 's'
3. Match 'e' from 'sh'
   - Output: "she" (ending at 'e' from 'sh')
   - Follow failure link from 'she''s 'e' to 'he''s 'e'
   - Output: "he" (ending at 'e' from 'h')
```

**Applications:**

```python
def find_all_word_occurrences(text, words):
    """
    Find all occurrences of words in text (case-insensitive).

    Args:
        text: Text to search
        words: List of words to find

    Returns:
        Dictionary with results
    """
    text_lower = text.lower()
    words_lower = [w.lower() for w in words]

    ac = AhoCorasick(words_lower)
    results = ac.search(text_lower)

    # Convert back to original words
    output = {}
    for pattern_idx, positions in results.items():
        word = words[pattern_idx]
        output[word] = positions

    return output

def censor_words(text, banned_words, replacement='*'):
    """
    Censor all banned words in text.

    Args:
        text: Original text
        banned_words: List of words to censor
        replacement: Character to replace with

    Returns:
        Censored text
    """
    ac = AhoCorasick([word.lower() for word in banned_words])
    results = ac.search(text.lower())

    # Convert to list for in-place modification
    censored = list(text)

    # Censor all matches
    for pattern_idx, positions in results.items():
        word_len = len(banned_words[pattern_idx])
        for pos in positions:
            for i in range(pos, pos + word_len):
                censored[i] = replacement

    return ''.join(censored)

# Examples
text = "She sells seashells by the seashore"
words = ["she", "sea", "shore"]
print(find_all_word_occurrences(text, words))
# {'She': [0], 'sea': [10, 27], 'shore': [30]}

text = "This is a badword and another badword here"
banned = ["badword", "another"]
print(censor_words(text, banned))
# "This is a ******* and ******* ******* here"
```

**Complexity Analysis:**
- Build trie: $O(m)$ where m is sum of pattern lengths
- Build failure links: $O(m)$
- Search: $O(n + z)$ where n is text length, z is number of matches
- Total: $O(n + m + z)$
- Space: $O(m)$

**Pros:**
- Optimal for multiple pattern matching
- Linear time in text length
- Finds all patterns simultaneously

**Cons:**
- Complex implementation
- Requires preprocessing
- Higher memory than single-pattern algorithms

**Use Cases:**
- Spam filters
- Content moderation
- Virus scanners
- Intrusion detection systems
- Text analysis tools

---

### Regular Expression Matching

**Time**: $O(n \times m)$ DP approach | **Space**: $O(n \times m)$

Match text against pattern with special characters.

```python
def is_match_recursive(text, pattern):
    """
    Regular expression matching using recursion.
    Supports '.' (any character) and '*' (zero or more of previous char).

    Args:
        text: Text to match
        pattern: Pattern with wildcards

    Returns:
        True if text matches pattern
    """
    # Base case: empty pattern
    if not pattern:
        return not text

    # Check if first character matches
    first_match = bool(text) and pattern[0] in {text[0], '.'}

    # Handle '*'
    if len(pattern) >= 2 and pattern[1] == '*':
        # Two options:
        # 1. '*' matches zero occurrences
        # 2. '*' matches one or more occurrences
        return (is_match_recursive(text, pattern[2:]) or
                (first_match and is_match_recursive(text[1:], pattern)))
    else:
        # No '*', must match current character and continue
        return first_match and is_match_recursive(text[1:], pattern[1:])

def is_match_dp(text, pattern):
    """
    Regular expression matching using dynamic programming.
    More efficient than recursion.

    Time: O(n * m)
    Space: O(n * m)
    """
    n, m = len(text), len(pattern)

    # dp[i][j] = whether text[0:i] matches pattern[0:j]
    dp = [[False] * (m + 1) for _ in range(n + 1)]

    # Base case: empty text and empty pattern
    dp[0][0] = True

    # Handle patterns like a*, a*b*, etc. that can match empty string
    for j in range(2, m + 1):
        if pattern[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pattern[j - 1] == '*':
                # Two cases:
                # 1. '*' matches zero of previous character
                dp[i][j] = dp[i][j - 2]

                # 2. '*' matches one or more of previous character
                if pattern[j - 2] == text[i - 1] or pattern[j - 2] == '.':
                    dp[i][j] = dp[i][j] or dp[i - 1][j]

            elif pattern[j - 1] == '.' or pattern[j - 1] == text[i - 1]:
                # Characters match
                dp[i][j] = dp[i - 1][j - 1]

    return dp[n][m]

# Examples
print(is_match_dp("aa", "a"))       # False
print(is_match_dp("aa", "a*"))      # True
print(is_match_dp("ab", ".*"))      # True
print(is_match_dp("aab", "c*a*b"))  # True
print(is_match_dp("mississippi", "mis*is*p*."))  # False
```

**DP Table Visualization:**

```
text = "aab"
pattern = "c*a*b"

       ""  c  *  a  *  b
 ""    T   F  T  F  T  F
 a     F   F  F  T  T  F
 a     F   F  F  F  T  F
 b     F   F  F  F  F  T

Explanation:
- dp[0][0] = True (empty matches empty)
- dp[0][2] = True (c* matches empty)
- dp[0][4] = True (c*a* matches empty)
- dp[1][3] = True ("a" matches "c*a")
- dp[2][4] = True ("aa" matches "c*a*")
- dp[3][5] = True ("aab" matches "c*a*b")
```

**Wildcard Pattern Matching:**

```python
def is_match_wildcard(text, pattern):
    """
    Wildcard pattern matching.
    '?' matches any single character
    '*' matches any sequence (including empty)

    Args:
        text: Text to match
        pattern: Pattern with wildcards

    Returns:
        True if text matches pattern
    """
    n, m = len(text), len(pattern)
    dp = [[False] * (m + 1) for _ in range(n + 1)]

    # Base case
    dp[0][0] = True

    # Handle leading '*' in pattern
    for j in range(1, m + 1):
        if pattern[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pattern[j - 1] == '*':
                # '*' matches empty or one or more characters
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif pattern[j - 1] == '?' or pattern[j - 1] == text[i - 1]:
                # '?' or exact match
                dp[i][j] = dp[i - 1][j - 1]

    return dp[n][m]

# Examples
print(is_match_wildcard("aa", "a"))      # False
print(is_match_wildcard("aa", "*"))      # True
print(is_match_wildcard("cb", "?a"))     # False
print(is_match_wildcard("adceb", "*a*b"))  # True
print(is_match_wildcard("acdcb", "a*c?b"))  # False
```

**Extended Regex Features:**

```python
def compile_regex(pattern):
    """
    Compile regex pattern to NFA (Nondeterministic Finite Automaton).
    Supports: literals, '.', '*', '+', '?', '|', '(', ')'

    This is a simplified version - real regex engines are much more complex.
    """
    class NFA:
        def __init__(self):
            self.states = []
            self.start = None
            self.end = None

    # This would require Thompson's construction algorithm
    # Simplified for demonstration
    pass

def regex_search(text, pattern):
    """
    Find first match of pattern in text.

    Returns:
        (start, end) positions of match, or None
    """
    n = len(text)
    m = len(pattern)

    # Try matching at each position
    for i in range(n - m + 1):
        if is_match_dp(text[i:], pattern):
            # Find exact end position
            for j in range(m, n - i + 1):
                if is_match_dp(text[i:i+j], pattern):
                    end = i + j
                else:
                    break
            return (i, end)

    return None

def regex_findall(text, pattern):
    """
    Find all non-overlapping matches.

    Returns:
        List of (start, end) tuples
    """
    matches = []
    i = 0

    while i < len(text):
        match = regex_search(text[i:], pattern)
        if match:
            start, end = match
            matches.append((i + start, i + end))
            i += end
        else:
            break

    return matches

# Note: For real regex, use Python's re module
import re

# Example with real regex
text = "The quick brown fox jumps over the lazy dog"
pattern = r'\b\w{5}\b'  # 5-letter words
matches = re.findall(pattern, text)
print(matches)  # ['quick', 'brown', 'jumps']
```

**Complexity Analysis:**
- Time: $O(n \times m)$ for DP approach
- Space: $O(n \times m)$ or $O(m)$ optimized
- Recursive: Exponential without memoization

**Use Cases:**
- Text validation
- Search and replace
- Input parsing
- Lexical analysis
- Data extraction

---

## Applications

### Text Editors

String algorithms power find/replace, undo/redo, and syntax highlighting.

```python
class TextEditor:
    """
    Simple text editor with string algorithm applications.
    """

    def __init__(self):
        self.text = ""
        self.history = []
        self.history_index = -1

    def insert(self, pos, string):
        """Insert string at position."""
        self.text = self.text[:pos] + string + self.text[pos:]
        self._save_state()

    def delete(self, start, end):
        """Delete characters from start to end."""
        self.text = self.text[:start] + self.text[end:]
        self._save_state()

    def find(self, pattern):
        """Find all occurrences using KMP."""
        return kmp_search(self.text, pattern)

    def replace(self, old, new):
        """Replace all occurrences."""
        positions = self.find(old)

        # Replace from right to left to maintain positions
        for pos in reversed(positions):
            self.text = self.text[:pos] + new + self.text[pos + len(old):]

        self._save_state()

    def fuzzy_find(self, pattern, max_distance=2):
        """Find approximate matches."""
        matches = []
        n = len(self.text)
        m = len(pattern)

        for i in range(n - m + 1):
            substr = self.text[i:i + m]
            dist, _ = edit_distance(pattern, substr)
            if dist <= max_distance:
                matches.append((i, dist))

        return matches

    def undo(self):
        """Undo last operation."""
        if self.history_index > 0:
            self.history_index -= 1
            self.text = self.history[self.history_index]

    def redo(self):
        """Redo last undone operation."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.text = self.history[self.history_index]

    def _save_state(self):
        """Save current state to history."""
        # Remove any redo states
        self.history = self.history[:self.history_index + 1]
        self.history.append(self.text)
        self.history_index += 1

# Example usage
editor = TextEditor()
editor.insert(0, "Hello World")
editor.insert(5, " Beautiful")
print(editor.text)  # "Hello Beautiful World"

positions = editor.find("World")
print(f"'World' found at: {positions}")

editor.replace("World", "Universe")
print(editor.text)  # "Hello Beautiful Universe"
```

### Spell Checkers

Use edit distance and tries for suggestions.

```python
class SpellChecker:
    """
    Spell checker using trie and edit distance.
    """

    def __init__(self, dictionary):
        """
        Initialize with dictionary of valid words.

        Args:
            dictionary: List of valid words
        """
        self.trie = Trie()
        for word in dictionary:
            self.trie.insert(word.lower())

    def is_correct(self, word):
        """Check if word is spelled correctly."""
        return self.trie.search(word.lower())

    def suggestions(self, word, max_distance=2, max_suggestions=5):
        """
        Get spelling suggestions for misspelled word.

        Args:
            word: Misspelled word
            max_distance: Maximum edit distance for suggestions
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (suggestion, distance) sorted by distance
        """
        word_lower = word.lower()
        suggestions = []

        # BFS through trie to find similar words
        def dfs(node, current_word, distance):
            if distance > max_distance:
                return

            if node.is_end_of_word:
                if current_word != word_lower:
                    dist, _ = edit_distance(word_lower, current_word)
                    if dist <= max_distance:
                        suggestions.append((current_word, dist))

            for char, child in node.children.items():
                dfs(child, current_word + char, distance + 1)

        dfs(self.trie.root, "", 0)

        # Sort by distance, then alphabetically
        suggestions.sort(key=lambda x: (x[1], x[0]))

        return suggestions[:max_suggestions]

    def autocomplete(self, prefix, max_suggestions=5):
        """Get autocomplete suggestions."""
        words = self.trie.find_all_with_prefix(prefix.lower())
        return words[:max_suggestions]

# Example
dictionary = ["hello", "help", "helping", "world", "word", "work"]
checker = SpellChecker(dictionary)

print(checker.is_correct("hello"))  # True
print(checker.is_correct("helo"))   # False

suggestions = checker.suggestions("helo")
print(f"Suggestions for 'helo': {suggestions}")
# [('hello', 1), ('help', 1)]

completions = checker.autocomplete("hel")
print(f"Autocomplete for 'hel': {completions}")
# ['hello', 'help', 'helping']
```

### DNA Sequence Analysis

Pattern matching in genomic data.

```python
class DNAAnalyzer:
    """
    DNA sequence analysis using string algorithms.
    """

    def __init__(self, sequence):
        """
        Initialize with DNA sequence.

        Args:
            sequence: DNA string (A, T, G, C)
        """
        self.sequence = sequence.upper()
        self.complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

    def find_gene(self, gene_sequence):
        """Find all occurrences of gene sequence."""
        return kmp_search(self.sequence, gene_sequence.upper())

    def complement(self):
        """Get complementary DNA strand."""
        return ''.join(self.complement_map[base] for base in self.sequence)

    def reverse_complement(self):
        """Get reverse complement (important in DNA analysis)."""
        return self.complement()[::-1]

    def find_palindromes(self, min_length=4):
        """
        Find palindromic sequences (important in restriction sites).

        Returns:
            List of (position, length) tuples
        """
        palindromes = []
        n = len(self.sequence)

        # Use Manacher's algorithm concept
        for center in range(n):
            # Odd length palindromes
            left = right = center
            while left >= 0 and right < n and \
                  self.sequence[left] == self.complement_map[self.sequence[right]]:
                length = right - left + 1
                if length >= min_length:
                    palindromes.append((left, length))
                left -= 1
                right += 1

        return palindromes

    def find_repeats(self, min_length=10):
        """
        Find repeated sequences using suffix array.

        Returns:
            List of repeated sequences
        """
        sa = build_suffix_array_efficient(self.sequence)
        lcp = build_lcp_array(self.sequence, sa)

        repeats = []
        for i in range(1, len(lcp)):
            if lcp[i] >= min_length:
                repeat = self.sequence[sa[i]:sa[i] + lcp[i]]
                repeats.append(repeat)

        return list(set(repeats))

    def similarity(self, other_sequence):
        """
        Calculate similarity with another sequence using LCS.

        Returns:
            Similarity percentage
        """
        lcs_len, _ = longest_common_subsequence(self.sequence, other_sequence.upper())
        max_len = max(len(self.sequence), len(other_sequence))
        return (lcs_len / max_len) * 100

# Example
dna = DNAAnalyzer("ATCGATCGATCG")
print(f"Sequence: {dna.sequence}")
print(f"Complement: {dna.complement()}")
print(f"Reverse complement: {dna.reverse_complement()}")

positions = dna.find_gene("ATCG")
print(f"Gene 'ATCG' found at positions: {positions}")

dna2 = DNAAnalyzer("ATCGGGGATCG")
similarity = dna.similarity("ATCGGGGATCG")
print(f"Similarity: {similarity:.1f}%")
```

### Search Engines

Inverted index with pattern matching.

```python
class SimpleSearchEngine:
    """
    Simple search engine using string algorithms.
    """

    def __init__(self):
        self.documents = {}  # doc_id -> content
        self.inverted_index = {}  # word -> set of doc_ids
        self.doc_counter = 0

    def add_document(self, content):
        """
        Add document to search engine.

        Returns:
            Document ID
        """
        doc_id = self.doc_counter
        self.documents[doc_id] = content
        self.doc_counter += 1

        # Index words
        words = content.lower().split()
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(doc_id)

        return doc_id

    def search_exact(self, query):
        """
        Search for exact query match.

        Returns:
            List of (doc_id, positions) tuples
        """
        query_lower = query.lower()
        results = []

        for doc_id, content in self.documents.items():
            positions = kmp_search(content.lower(), query_lower)
            if positions:
                results.append((doc_id, positions))

        return results

    def search_fuzzy(self, query, max_distance=2):
        """
        Fuzzy search allowing typos.

        Returns:
            List of (doc_id, relevance_score) tuples
        """
        query_words = query.lower().split()
        results = {}

        for doc_id, content in self.documents.items():
            score = 0
            content_words = content.lower().split()

            for query_word in query_words:
                for content_word in content_words:
                    dist, _ = edit_distance(query_word, content_word)
                    if dist <= max_distance:
                        # Closer matches get higher scores
                        score += (max_distance - dist + 1)

            if score > 0:
                results[doc_id] = score

        # Sort by relevance
        return sorted(results.items(), key=lambda x: x[1], reverse=True)

    def search_boolean(self, *query_words):
        """
        Boolean AND search (all words must be present).

        Returns:
            Set of matching document IDs
        """
        if not query_words:
            return set()

        # Start with documents containing first word
        result = self.inverted_index.get(query_words[0].lower(), set()).copy()

        # Intersect with documents containing other words
        for word in query_words[1:]:
            result &= self.inverted_index.get(word.lower(), set())

        return result

# Example usage
search_engine = SimpleSearchEngine()

# Add documents
search_engine.add_document("The quick brown fox jumps over the lazy dog")
search_engine.add_document("A quick brown dog runs in the park")
search_engine.add_document("The lazy cat sleeps all day")

# Exact search
results = search_engine.search_exact("quick brown")
print(f"Exact matches: {results}")

# Fuzzy search (handles typos)
results = search_engine.search_fuzzy("quik brwn", max_distance=2)
print(f"Fuzzy matches: {results}")

# Boolean search
results = search_engine.search_boolean("quick", "dog")
print(f"Documents with 'quick' AND 'dog': {results}")
```

---

## Interview Patterns

### Common Problem Types

**1. Palindrome Problems**
```python
def is_palindrome(s):
    """Check if string is palindrome. O(n)"""
    return s == s[::-1]

def longest_palindrome_dp(s):
    """Longest palindromic substring using DP. O(n^2)"""
    n = len(s)
    if n == 0:
        return ""

    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1

    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check two-character palindromes
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    # Check palindromes of length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length

    return s[start:start + max_len]

def min_insertions_to_make_palindrome(s):
    """Minimum insertions to make string a palindrome."""
    n = len(s)
    # LCS of s and reverse of s
    lcs_len, _ = longest_common_subsequence(s, s[::-1])
    return n - lcs_len
```

**2. Anagram Problems**
```python
def are_anagrams(s1, s2):
    """Check if two strings are anagrams. O(n)"""
    from collections import Counter
    return Counter(s1) == Counter(s2)

def group_anagrams(words):
    """
    Group words that are anagrams.

    Returns:
        List of lists of anagrams
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for word in words:
        # Sort characters as key
        key = ''.join(sorted(word))
        groups[key].append(word)

    return list(groups.values())

def find_all_anagrams(text, pattern):
    """
    Find all anagram substrings in text.
    Uses sliding window. O(n)
    """
    from collections import Counter

    n, m = len(text), len(pattern)
    if n < m:
        return []

    pattern_count = Counter(pattern)
    window_count = Counter(text[:m])

    result = []
    if window_count == pattern_count:
        result.append(0)

    # Slide window
    for i in range(m, n):
        # Add new character
        window_count[text[i]] += 1

        # Remove old character
        old_char = text[i - m]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check if anagram
        if window_count == pattern_count:
            result.append(i - m + 1)

    return result

# Examples
print(are_anagrams("listen", "silent"))  # True
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
print(find_all_anagrams("cbaebabacd", "abc"))  # [0, 6]
```

**3. Substring Problems**
```python
def length_of_longest_substring(s):
    """
    Longest substring without repeating characters.
    Uses sliding window. O(n)
    """
    char_index = {}
    max_length = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            # Move start past previous occurrence
            start = char_index[char] + 1

        char_index[char] = end
        max_length = max(max_length, end - start + 1)

    return max_length

def min_window_substring(s, t):
    """
    Minimum window substring containing all characters of t.
    Uses sliding window. O(n)
    """
    from collections import Counter

    if not s or not t:
        return ""

    # Count characters in t
    dict_t = Counter(t)
    required = len(dict_t)

    # Sliding window
    left = right = 0
    formed = 0  # Number of unique chars in window with desired frequency
    window_counts = {}

    # (window length, left, right)
    ans = float("inf"), None, None

    while right < len(s):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        # Try to shrink window
        while left <= right and formed == required:
            char = s[left]

            # Save smallest window
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

        right += 1

    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]

# Examples
print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print(min_window_substring("ADOBECODEBANC", "ABC"))  # "BANC"
```

**4. String Transformation**
```python
def one_edit_distance(s1, s2):
    """
    Check if strings are one edit apart.
    Edits: insert, delete, replace
    """
    n1, n2 = len(s1), len(s2)

    # Ensure s1 is shorter
    if n1 > n2:
        return one_edit_distance(s2, s1)

    if n2 - n1 > 1:
        return False

    for i in range(n1):
        if s1[i] != s2[i]:
            if n1 == n2:
                # Replace: rest must match
                return s1[i + 1:] == s2[i + 1:]
            else:
                # Delete from s2: s1 must match s2 after deletion
                return s1[i:] == s2[i + 1:]

    # All characters match, strings differ by one if lengths differ
    return n1 + 1 == n2

def word_ladder_length(begin_word, end_word, word_list):
    """
    Shortest transformation sequence from begin_word to end_word.
    Each step changes one letter, intermediate words must be in word_list.

    Uses BFS. O(M * N) where M is word length, N is number of words.
    """
    from collections import deque

    if end_word not in word_list:
        return 0

    word_set = set(word_list)
    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, steps = queue.popleft()

        if word == end_word:
            return steps

        # Try all possible one-letter transformations
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i + 1:]

                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))

    return 0

# Examples
print(one_edit_distance("ab", "acb"))  # True
print(word_ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
# 5: hit -> hot -> dot -> dog -> cog
```

**5. Pattern Matching**
```python
def strStr(haystack, needle):
    """
    Find first occurrence of needle in haystack (implement indexOf).
    Returns index or -1.
    """
    if not needle:
        return 0

    positions = kmp_search(haystack, needle)
    return positions[0] if positions else -1

def repeated_substring_pattern(s):
    """
    Check if string can be constructed by repeating a substring.
    Uses Z-algorithm or KMP failure function.
    """
    n = len(s)
    lps = compute_lps(s)

    # If lps[n-1] > 0 and n % (n - lps[n-1]) == 0, then repeating
    if lps[n - 1] > 0 and n % (n - lps[n - 1]) == 0:
        return True
    return False

def is_subsequence(s, t):
    """
    Check if s is subsequence of t.
    Two pointers. O(n)
    """
    i = 0
    for char in t:
        if i < len(s) and s[i] == char:
            i += 1
    return i == len(s)

# Examples
print(strStr("hello", "ll"))  # 2
print(repeated_substring_pattern("abcabcabc"))  # True
print(is_subsequence("abc", "ahbgdc"))  # True
```

### Time Complexity Quick Reference

| Problem Type | Naive | Optimized | Algorithm |
|--------------|-------|-----------|-----------|
| Pattern matching | $O(nm)$ | $O(n+m)$ | KMP, Z-algorithm |
| Multiple patterns | $O(nm \times k)$ | $O(n+m \times k)$ | Aho-Corasick |
| Longest palindrome | $O(n^3)$ | $O(n)$ | Manacher's |
| Edit distance | Exponential | $O(nm)$ | DP |
| LCS | Exponential | $O(nm)$ | DP |
| Anagrams | $O(n \log n)$ | $O(n)$ | Hash table |
| Substring search | $O(nm)$ | $O(n+m)$ | KMP, Rabin-Karp |

### Interview Tips

1. **Start Simple**: Clarify requirements, discuss naive approach first
2. **Pattern Recognition**: Identify problem type (palindrome, anagram, substring, etc.)
3. **Choose Right Algorithm**:
   - Single pattern → KMP or Rabin-Karp
   - Multiple patterns → Aho-Corasick or Rabin-Karp
   - Palindrome → Manacher's or DP
   - Edit operations → DP
4. **Optimize Space**: Many DP solutions can use O(n) instead of O(n²)
5. **Edge Cases**: Empty strings, single character, all same characters
6. **Test Cases**: Normal case, edge cases, large input

---

## Further Resources

**Books:**
- "Algorithms on Strings, Trees, and Sequences" by Dan Gusfield
- "String Searching Algorithms" by Graham A. Stephen
- "Flexible Pattern Matching in Strings" by Gonzalo Navarro, Mathieu Raffinot

**Online Resources:**
- [CP-Algorithms String Algorithms](https://cp-algorithms.com/string/)
- [GeeksforGeeks String Algorithms](https://www.geeksforgeeks.org/string-data-structure/)
- [LeetCode String Problems](https://leetcode.com/tag/string/)

**Practice Platforms:**
- LeetCode (String tag)
- Codeforces
- HackerRank
- CSES Problem Set

**Visualizations:**
- [VisuAlgo String Algorithms](https://visualgo.net/en/suffixarray)
- [String Algorithm Animations](https://www.cs.usfca.edu/~galles/visualization/)

---

## ELI10

**String algorithms help computers find and compare text super fast!**

**Pattern Matching** is like playing "Where's Waldo?" in a book - you're looking for a specific pattern in a lot of text:
- **Naive**: Check every spot (slow!)
- **KMP**: Learn from mistakes, don't recheck (smart!)
- **Boyer-Moore**: Start from the end, skip big chunks (fastest for big alphabets!)

**String Comparison** is like figuring out how similar two words are:
- **LCS**: Find the longest common subsequence (like finding what's same in "kitten" and "sitting")
- **Edit Distance**: Count how many changes to transform one word to another (useful for spell check!)

**Advanced Topics**:
- **Tries**: Special trees for storing words (like a dictionary!)
- **Suffix Arrays**: Sort all word endings (helps find patterns super fast!)
- **Manacher's**: Find palindromes in linear time (racecar!)

Remember: Choose the right algorithm for your problem - faster isn't always better if it's too complex!
