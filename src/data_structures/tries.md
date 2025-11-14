# Tries (Prefix Trees)

A **trie**, also known as a **prefix tree** or **digital tree**, is a specialized tree-based data structure used for efficient storage and retrieval of strings. The name "trie" comes from the word "retrieval", though it's pronounced "try" to distinguish it from "tree". Tries excel at prefix-based operations and are commonly used in autocomplete systems, spell checkers, IP routing, and dictionary implementations.

## Visual Example

Here's a simple trie storing the words ["cat", "car", "card", "dog", "dodge", "door"]:

```
                    (root)
                    /    \
                   c      d
                   |      |
                   a      o
                  / \    /|\
                 t   r  d g |
                     |  | |  \
                     d  g e  r
                        |
                        e
```

Each path from root to a marked node represents a complete word. Notice how words sharing common prefixes (like "car", "card") share the same nodes for those prefixes.

---

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [How Tries Work](#how-tries-work)
3. [Operations & Time Complexity](#operations--time-complexity)
4. [Implementation](#implementation)
5. [Trie Variations](#trie-variations)
6. [Common Patterns & Techniques](#common-patterns--techniques)
7. [Common Problems](#common-problems)
8. [Applications](#applications)
9. [Optimizations & Tricks](#optimizations--tricks)
10. [Advantages & Disadvantages](#advantages--disadvantages)
11. [Comparison with Other Data Structures](#comparison-with-other-data-structures)
12. [Complexity Analysis](#complexity-analysis)
13. [Interview Tips & Patterns](#interview-tips--patterns)
14. [Real-World Implementation Considerations](#real-world-implementation-considerations)
15. [Advanced Code Examples](#advanced-code-examples)
16. [Explain Like I'm 10](#explain-like-im-10)
17. [Further Resources](#further-resources)
18. [Conclusion](#conclusion)

---

## Key Concepts

### Node Structure

Each **TrieNode** contains:
- **children**: A collection (array, hash map, or dictionary) mapping characters to child nodes
- **is_end_of_word**: A boolean flag indicating if this node marks the end of a valid word
- **Optional fields**: word count, frequency, actual word string, etc.

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Can also be an array[26] for lowercase letters
        self.is_end_of_word = False
        # Optional: store additional data
        # self.word = None
        # self.frequency = 0
```

### Prefix Property

The fundamental property of tries: **all descendants of a node share a common prefix**. This makes prefix-based operations extremely efficient.

Example: In a trie containing ["tea", "ted", "ten", "inn", "in"]
```
        root
        / \
       t   i
       |   |
       e   n
      /|\  |
     a d n n
```
- All words starting with "te" share the path root → t → e
- This shared structure saves space and enables fast prefix queries

### Character Set Considerations

**Alphabet size** affects implementation choices:
- **Lowercase letters only (a-z)**: Use array of size 26 for O(1) lookup
- **Mixed case (a-z, A-Z)**: Array of size 52 or normalize to lowercase
- **Alphanumeric**: Array of size 62 or hash map
- **Unicode/Any character**: Hash map or dictionary is necessary

### Memory Representation

**Array-based (fixed alphabet)**:
```python
children = [None] * 26  # For 'a' to 'z'
index = ord(char) - ord('a')
```
- **Pros**: O(1) access, cache-friendly
- **Cons**: Wastes space for sparse data

**Hash map-based (dynamic)**:
```python
children = {}  # Dictionary
children[char] = TrieNode()
```
- **Pros**: Space-efficient for sparse data, supports any character
- **Cons**: Slightly slower lookup, hash overhead

---

## How Tries Work

### Insertion Process

**Inserting "CAR" into an empty trie:**

**Step 1:** Start at root
```
root
```

**Step 2:** Add 'C' as child of root
```
root
 |
 C
```

**Step 3:** Add 'A' as child of 'C'
```
root
 |
 C
 |
 A
```

**Step 4:** Add 'R' as child of 'A', mark as end of word
```
root
 |
 C
 |
 A
 |
 R*  (* = end of word)
```

**Inserting "CAT" into the existing trie:**

Since 'C' and 'A' already exist, we traverse to 'A' and add only 'T':
```
root
 |
 C
 |
 A
/ \
R* T*
```

### Search Process

**Searching for "CAT":**

1. Start at root, look for 'C' → Found
2. Move to 'C', look for 'A' → Found
3. Move to 'A', look for 'T' → Found
4. Check if 'T' is marked as end of word → Yes
5. **Result**: Word exists ✓

**Searching for "CA":**

1. Start at root, look for 'C' → Found
2. Move to 'C', look for 'A' → Found
3. Check if 'A' is marked as end of word → No
4. **Result**: Word doesn't exist (though it's a valid prefix) ✗

### Prefix Search Process

**Finding all words with prefix "CA":**

1. Navigate to the node representing "CA"
2. Perform DFS/BFS from that node
3. Collect all words (nodes marked as end of word)
4. **Result**: ["CAR", "CAT"]

### Deletion Process

**Three cases for deletion:**

**Case 1: Leaf Node (no children)**
- Delete "CAT" where T has no children
- Simply remove the node and clean up unused parents

```
Before:        After:
root           root
 |              |
 C              C
 |              |
 A              A
/ \             |
R* T*  →       R*
```

**Case 2: Middle Node with Children**
- Delete "CA" where A has children
- Just unmark the end-of-word flag

```
Before:        After:
root           root
 |              |
 C              C
 |              |
 A*             A      (unmarked)
/ \            / \
R* T*         R* T*
```

**Case 3: Node Part of Other Words**
- Delete "CAR" where path is shared
- Unmark R, remove only if no children

---

## Operations & Time Complexity

### Complexity Summary Table

| Operation | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| **Insert** | O(m) | O(m) | m = length of word |
| **Search** | O(m) | O(1) | Exact word search |
| **Delete** | O(m) | O(1) | May need to clean up nodes |
| **StartsWith** | O(p) | O(1) | Check if prefix exists, p = prefix length |
| **AutoComplete** | O(p + n*k) | O(n*k) | p = prefix length, n = results, k = avg word length |
| **Get All Words** | O(N*M) | O(N*M) | N = total words, M = avg length |
| **Longest Prefix** | O(m) | O(1) | Find longest matching prefix |
| **Count Words** | O(1) | O(1) | If counter maintained |
| **Count Prefixes** | O(m) | O(1) | Count words with prefix |

### Detailed Operation Explanations

#### 1. Insert Operation - O(m)

Insert adds a new word to the trie by creating nodes for each character.

**Three scenarios:**

**a) Completely new word:**
```python
# Insert "DOG" into trie containing only "CAT"
# Creates entirely new branch from root
```

**b) New word is prefix of existing:**
```python
# Insert "CAR" into trie containing "CARD"
# Navigate to 'R' and mark it as end of word
```

**c) New word extends existing prefix:**
```python
# Insert "CARD" into trie containing "CAR"
# Navigate to 'R' and add 'D' as child
```

**Why O(m)?** We iterate through each character once.

#### 2. Search Operation - O(m)

Search checks if an exact word exists in the trie.

**Key distinction:** Must verify `is_end_of_word` flag!

```python
# Searching "CAR" in trie containing "CARD"
# Navigate: root → C → A → R
# Check: is_end_of_word at R?
# If True: word exists
# If False: only a prefix exists
```

**Why O(m)?** We traverse one character at a time.

#### 3. Delete Operation - O(m)

Delete removes a word and cleans up unused nodes.

**Algorithm:**
1. Find the word (if it doesn't exist, return)
2. Unmark `is_end_of_word` flag
3. Recursively remove nodes that are no longer needed

**Node can be deleted if:**
- It's not marked as end of word
- It has no children

**Why O(m)?** Traverse to find word (O(m)) + potential cleanup (O(m))

#### 4. Prefix Search (StartsWith) - O(p)

Check if any word starts with given prefix.

```python
# StartsWith("CA") in trie with ["CAR", "CAT"]
# Navigate: root → C → A
# If we successfully navigate entire prefix: return True
```

**Difference from Search:**
- Search requires exact match + `is_end_of_word` flag
- StartsWith only requires path to exist

#### 5. AutoComplete - O(p + n*k)

Find all words with a given prefix.

**Steps:**
1. Navigate to prefix node - O(p)
2. DFS/BFS from that node - O(n*k) where n = number of results, k = avg length
3. Collect all complete words

```python
# AutoComplete("CA") returns ["CAR", "CAT", "CARD"]
```

---

## Implementation

### Basic Trie Implementation (Python)

```python
class TrieNode:
    """
    Node in a Trie. Each node represents a character.
    """
    def __init__(self):
        # Dictionary mapping characters to TrieNode objects
        self.children = {}
        # Flag indicating if this node marks the end of a valid word
        self.is_end_of_word = False
        # Optional: store the actual word at leaf nodes for convenience
        self.word = None


class Trie:
    """
    Trie (Prefix Tree) data structure for efficient string operations.
    """

    def __init__(self):
        """Initialize trie with empty root node."""
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.

        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(m) in worst case (all new nodes)

        Args:
            word: String to insert
        """
        if not word:
            return

        node = self.root

        # Traverse or create nodes for each character
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        # Mark the last node as end of word
        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.word = word
            self.word_count += 1

    def search(self, word: str) -> bool:
        """
        Search for an exact word in the trie.

        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(1)

        Args:
            word: String to search for

        Returns:
            True if word exists in trie, False otherwise
        """
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in trie starts with given prefix.

        Time Complexity: O(p) where p is the length of prefix
        Space Complexity: O(1)

        Args:
            prefix: Prefix to search for

        Returns:
            True if prefix exists, False otherwise
        """
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """
        Helper method to find the node representing a prefix.

        Args:
            prefix: String to find

        Returns:
            TrieNode if prefix exists, None otherwise
        """
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node

    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie.

        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(m) due to recursion stack

        Args:
            word: Word to delete

        Returns:
            True if word was deleted, False if word didn't exist
        """
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            """
            Recursive helper for deletion.

            Returns:
                True if the node should be deleted, False otherwise
            """
            # Base case: reached end of word
            if index == len(word):
                # Word doesn't exist
                if not node.is_end_of_word:
                    return False

                # Unmark as end of word
                node.is_end_of_word = False
                node.word = None

                # Delete node if it has no children
                return len(node.children) == 0

            char = word[index]

            # Character not found
            if char not in node.children:
                return False

            child_node = node.children[char]
            should_delete_child = _delete_helper(child_node, word, index + 1)

            # Delete child node if necessary
            if should_delete_child:
                del node.children[char]

                # Delete current node if:
                # 1. It's not end of another word
                # 2. It has no other children
                return not node.is_end_of_word and len(node.children) == 0

            return False

        if _delete_helper(self.root, word, 0):
            self.word_count -= 1
            return True
        return False

    def get_all_words_with_prefix(self, prefix: str) -> list:
        """
        Get all words that start with the given prefix (autocomplete).

        Time Complexity: O(p + n*k) where p = prefix length,
                         n = number of results, k = avg word length
        Space Complexity: O(n*k) for storing results

        Args:
            prefix: Prefix to search for

        Returns:
            List of all words starting with prefix
        """
        results = []
        node = self._find_node(prefix)

        if node is None:
            return results

        # DFS to find all words from this node
        self._dfs_words(node, prefix, results)
        return results

    def _dfs_words(self, node: TrieNode, current_word: str, results: list) -> None:
        """
        DFS helper to collect all words from a given node.

        Args:
            node: Current node
            current_word: Word formed so far
            results: List to store found words
        """
        if node.is_end_of_word:
            results.append(current_word)

        for char, child_node in node.children.items():
            self._dfs_words(child_node, current_word + char, results)

    def get_all_words(self) -> list:
        """
        Get all words stored in the trie.

        Time Complexity: O(N*M) where N = number of words, M = avg length
        Space Complexity: O(N*M)

        Returns:
            List of all words in trie
        """
        return self.get_all_words_with_prefix("")

    def longest_prefix(self, word: str) -> str:
        """
        Find the longest prefix of word that exists in trie.

        Time Complexity: O(m) where m is length of word
        Space Complexity: O(1)

        Args:
            word: Word to find prefix for

        Returns:
            Longest matching prefix
        """
        node = self.root
        prefix = ""

        for char in word:
            if char not in node.children:
                break
            prefix += char
            node = node.children[char]

        return prefix

    def count_words_with_prefix(self, prefix: str) -> int:
        """
        Count how many words start with given prefix.

        Time Complexity: O(p + n) where p = prefix length, n = words with prefix
        Space Complexity: O(1) excluding recursion

        Args:
            prefix: Prefix to count

        Returns:
            Number of words with prefix
        """
        node = self._find_node(prefix)
        if node is None:
            return 0

        return self._count_words(node)

    def _count_words(self, node: TrieNode) -> int:
        """
        Count all words in subtree rooted at node.

        Args:
            node: Root of subtree

        Returns:
            Number of words in subtree
        """
        count = 1 if node.is_end_of_word else 0

        for child in node.children.values():
            count += self._count_words(child)

        return count

    def __len__(self) -> int:
        """Return number of words in trie."""
        return self.word_count

    def __contains__(self, word: str) -> bool:
        """Support 'in' operator."""
        return self.search(word)

    def __repr__(self) -> str:
        """String representation of trie."""
        return f"Trie(words={self.word_count})"


# Example Usage
if __name__ == "__main__":
    # Create trie and add words
    trie = Trie()
    words = ["cat", "car", "card", "dog", "dodge", "door", "cat"]

    print("Inserting words:", words)
    for word in words:
        trie.insert(word)

    print(f"\nTrie contains {len(trie)} unique words")

    # Search operations
    print("\n=== Search Operations ===")
    print(f"'cat' in trie: {trie.search('cat')}")  # True
    print(f"'ca' in trie: {trie.search('ca')}")    # False (prefix only)
    print(f"'card' in trie: {trie.search('card')}") # True

    # Prefix operations
    print("\n=== Prefix Operations ===")
    print(f"Starts with 'ca': {trie.starts_with('ca')}")  # True
    print(f"Starts with 'bat': {trie.starts_with('bat')}")  # False

    # Autocomplete
    print("\n=== Autocomplete ===")
    print(f"Words with prefix 'ca': {trie.get_all_words_with_prefix('ca')}")
    print(f"Words with prefix 'do': {trie.get_all_words_with_prefix('do')}")

    # Count operations
    print("\n=== Count Operations ===")
    print(f"Words starting with 'car': {trie.count_words_with_prefix('car')}")
    print(f"Words starting with 'do': {trie.count_words_with_prefix('do')}")

    # Longest prefix
    print("\n=== Longest Prefix ===")
    print(f"Longest prefix of 'cardinal': {trie.longest_prefix('cardinal')}")
    print(f"Longest prefix of 'catch': {trie.longest_prefix('catch')}")

    # Delete operations
    print("\n=== Delete Operations ===")
    print(f"Deleting 'cat': {trie.delete('cat')}")
    print(f"'cat' in trie after deletion: {trie.search('cat')}")
    print(f"'car' still in trie: {trie.search('car')}")

    # Get all words
    print("\n=== All Words ===")
    print(f"All words in trie: {trie.get_all_words()}")
```

### Array-Based Trie Node (Fixed Alphabet)

Optimized for lowercase English letters only:

```python
class TrieNodeArray:
    """
    Array-based trie node for lowercase letters (a-z).
    More memory per node but O(1) lookup.
    """

    def __init__(self):
        # Array of 26 pointers (one for each letter)
        self.children = [None] * 26
        self.is_end_of_word = False

    def get_child(self, char: str) -> 'TrieNodeArray':
        """Get child node for character."""
        index = ord(char) - ord('a')
        return self.children[index]

    def set_child(self, char: str, node: 'TrieNodeArray') -> None:
        """Set child node for character."""
        index = ord(char) - ord('a')
        self.children[index] = node

    def has_child(self, char: str) -> bool:
        """Check if child exists for character."""
        index = ord(char) - ord('a')
        return self.children[index] is not None


class TrieArray:
    """Trie using array-based nodes."""

    def __init__(self):
        self.root = TrieNodeArray()

    def insert(self, word: str) -> None:
        """Insert word into trie."""
        node = self.root

        for char in word.lower():
            if not node.has_child(char):
                node.set_child(char, TrieNodeArray())
            node = node.get_child(char)

        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Search for word in trie."""
        node = self.root

        for char in word.lower():
            if not node.has_child(char):
                return False
            node = node.get_child(char)

        return node.is_end_of_word


# Comparison: Array vs HashMap
# Array-based:
#   - Pros: O(1) access, cache-friendly
#   - Cons: 26 pointers per node (26 * 8 = 208 bytes on 64-bit)
#
# HashMap-based:
#   - Pros: Space-efficient for sparse data
#   - Cons: Hash overhead, slightly slower lookup
```

### Test Cases

```python
def test_trie():
    """Comprehensive test suite for Trie."""

    trie = Trie()

    # Test 1: Empty trie
    assert len(trie) == 0
    assert not trie.search("hello")
    assert not trie.starts_with("h")

    # Test 2: Single word insertion
    trie.insert("hello")
    assert len(trie) == 1
    assert trie.search("hello")
    assert not trie.search("hell")  # Prefix, not a word
    assert trie.starts_with("hell")

    # Test 3: Multiple words with shared prefix
    trie.insert("hell")
    trie.insert("help")
    assert len(trie) == 3
    assert trie.search("hell")
    assert trie.search("help")
    assert trie.count_words_with_prefix("hel") == 3

    # Test 4: Duplicate insertion
    trie.insert("hello")
    assert len(trie) == 3  # Should not increase

    # Test 5: Autocomplete
    words = trie.get_all_words_with_prefix("hel")
    assert set(words) == {"hello", "hell", "help"}

    # Test 6: Deletion
    assert trie.delete("hell")
    assert not trie.search("hell")
    assert trie.search("hello")  # Other words intact
    assert len(trie) == 2

    # Test 7: Delete non-existent word
    assert not trie.delete("world")

    # Test 8: Empty string
    trie.insert("")
    assert trie.search("")

    # Test 9: Case sensitivity
    trie.insert("Hello")
    assert trie.search("Hello")
    assert not trie.search("hello") == trie.search("Hello")  # Different words

    print("All tests passed!")


if __name__ == "__main__":
    test_trie()
```

---

## Trie Variations

### 1. Compressed Trie (Radix Tree / Patricia Trie)

A **compressed trie** merges nodes with single children to save space. Also called **radix tree** or **Patricia trie**.

**Example:**

Standard trie for ["test", "testing", "tester"]:
```
    t
    |
    e
    |
    s
    |
    t*
   / \
  i   e
  |   |
  n   r*
  |
  g*
```

Compressed trie:
```
    test*
    /    \
  ing*  er*
```

**Implementation:**

```python
class RadixNode:
    """Node in a Radix Tree (Compressed Trie)."""

    def __init__(self, label=""):
        self.label = label  # Edge label (can be multi-character)
        self.children = {}
        self.is_end_of_word = False
        self.value = None  # Optional: store associated value


class RadixTree:
    """
    Radix Tree (Compressed Trie) - space-optimized trie.
    Merges chains of single-child nodes.
    """

    def __init__(self):
        self.root = RadixNode()

    def insert(self, word: str, value=None) -> None:
        """
        Insert word into radix tree.

        Time Complexity: O(m) where m is word length
        """
        if not word:
            return

        node = self.root
        i = 0

        while i < len(word):
            char = word[i]

            # Find child with matching first character
            if char not in node.children:
                # No match - create new node with remaining string
                new_node = RadixNode(word[i:])
                new_node.is_end_of_word = True
                new_node.value = value
                node.children[char] = new_node
                return

            child = node.children[char]
            label = child.label

            # Find common prefix length
            j = 0
            while j < len(label) and i + j < len(word) and label[j] == word[i + j]:
                j += 1

            if j == len(label):
                # Entire label matches, continue with this child
                node = child
                i += j
            else:
                # Partial match - need to split
                # Create intermediate node
                common_prefix = label[:j]
                remaining_label = label[j:]
                remaining_word = word[i + j:]

                # Split existing node
                intermediate = RadixNode(common_prefix)
                node.children[char] = intermediate

                # Original child becomes child of intermediate
                child.label = remaining_label
                intermediate.children[remaining_label[0]] = child

                if remaining_word:
                    # Create new node for remaining word
                    new_node = RadixNode(remaining_word)
                    new_node.is_end_of_word = True
                    new_node.value = value
                    intermediate.children[remaining_word[0]] = new_node
                else:
                    # Current node is end of word
                    intermediate.is_end_of_word = True
                    intermediate.value = value

                return

        # Word completely consumed
        node.is_end_of_word = True
        node.value = value

    def search(self, word: str) -> bool:
        """Search for exact word."""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def _find_node(self, word: str) -> RadixNode:
        """Find node representing word/prefix."""
        node = self.root
        i = 0

        while i < len(word):
            char = word[i]

            if char not in node.children:
                return None

            child = node.children[char]
            label = child.label

            # Check if word matches label
            j = 0
            while j < len(label) and i + j < len(word):
                if label[j] != word[i + j]:
                    return None
                j += 1

            if j < len(label):
                # Word ended mid-label
                return None

            node = child
            i += j

        return node

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._find_node(prefix) is not None


# Example usage
radix = RadixTree()
words = ["test", "testing", "tester", "team", "toast"]

for word in words:
    radix.insert(word)

print(radix.search("test"))     # True
print(radix.search("testing"))  # True
print(radix.search("tes"))      # False
print(radix.starts_with("tes")) # True
```

**Use Cases:**
- IP routing tables (longest prefix matching)
- Memory-efficient string storage with few common prefixes
- String matching algorithms
- File system paths

**Complexity:**
- Space: Better than standard trie (fewer nodes)
- Time: Same as standard trie O(m)

---

### 2. Suffix Trie

A **suffix trie** stores all suffixes of a string, enabling efficient pattern matching.

**Example:** Suffix trie for "BANANA"

Suffixes: "BANANA", "ANANA", "NANA", "ANA", "NA", "A"

```
class SuffixTrie:
    """
    Suffix Trie for pattern matching in strings.
    Stores all suffixes of a text.
    """

    def __init__(self, text: str):
        """
        Build suffix trie for text.

        Time Complexity: O(n²) where n is length of text
        Space Complexity: O(n²) in worst case
        """
        self.root = TrieNode()
        self.text = text

        # Insert all suffixes
        for i in range(len(text)):
            self._insert_suffix(text[i:], i)

    def _insert_suffix(self, suffix: str, start_index: int) -> None:
        """Insert a suffix starting at start_index."""
        node = self.root

        for char in suffix:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.start_index = start_index

    def contains_pattern(self, pattern: str) -> bool:
        """
        Check if pattern exists in original text.

        Time Complexity: O(m) where m is pattern length
        """
        node = self.root

        for char in pattern:
            if char not in node.children:
                return False
            node = node.children[char]

        return True

    def find_all_occurrences(self, pattern: str) -> list:
        """
        Find all starting positions where pattern occurs.

        Time Complexity: O(m + k) where m = pattern length, k = occurrences
        """
        node = self.root

        # Navigate to pattern
        for char in pattern:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all start indices
        indices = []
        self._collect_indices(node, indices)
        return sorted(indices)

    def _collect_indices(self, node: TrieNode, indices: list) -> None:
        """DFS to collect all start indices."""
        if node.is_end_of_word:
            indices.append(node.start_index)

        for child in node.children.values():
            self._collect_indices(child, indices)

    def longest_repeated_substring(self) -> str:
        """
        Find longest substring that appears at least twice.

        Time Complexity: O(n²)
        """
        longest = ""

        def dfs(node, path):
            nonlocal longest

            # Count how many suffixes pass through this node
            count = self._count_leaves(node)

            if count >= 2 and len(path) > len(longest):
                longest = path

            for char, child in node.children.items():
                dfs(child, path + char)

        dfs(self.root, "")
        return longest

    def _count_leaves(self, node: TrieNode) -> int:
        """Count leaf nodes in subtree."""
        if node.is_end_of_word:
            return 1
        return sum(self._count_leaves(child) for child in node.children.values())


# Example usage
text = "BANANA"
suffix_trie = SuffixTrie(text)

print(f"Text: {text}")
print(f"Contains 'ANA': {suffix_trie.contains_pattern('ANA')}")  # True
print(f"Contains 'BAN': {suffix_trie.contains_pattern('BAN')}")  # True
print(f"Contains 'XYZ': {suffix_trie.contains_pattern('XYZ')}")  # False

print(f"\nOccurrences of 'ANA': {suffix_trie.find_all_occurrences('ANA')}")  # [1, 3]
print(f"Longest repeated substring: {suffix_trie.longest_repeated_substring()}")  # "ANA"
```

**Applications:**
- String pattern matching
- Finding longest repeated substring
- DNA sequence analysis
- Text compression

**Note:** Suffix trees are more space-efficient than suffix tries (O(n) vs O(n²)).

---

### 3. Ternary Search Trie (TST)

A **ternary search trie** is a space-efficient alternative where each node has **three children**: less than, equal to, and greater than current character.

**Structure:**
```
Each node has:
- char: current character
- left: chars < current
- mid: chars = current (next char in word)
- right: chars > current
- is_end_of_word: marks end
```

**Example:** TST for ["cat", "bat", "can"]
```
       c
      /|\
     b   a
        /|\
       t   n
```

**Implementation:**

```python
class TSTNode:
    """Node in Ternary Search Trie."""

    def __init__(self, char):
        self.char = char
        self.left = None   # Less than
        self.mid = None    # Equal (next character)
        self.right = None  # Greater than
        self.is_end_of_word = False
        self.value = None


class TernarySearchTrie:
    """
    Ternary Search Trie - combines benefits of BST and Trie.
    More space-efficient than standard trie.
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, word: str, value=None) -> None:
        """
        Insert word into TST.

        Time Complexity: O(m) where m is word length
        Space Complexity: O(m) in worst case
        """
        if not word:
            return

        self.root = self._insert_helper(self.root, word, 0, value)

    def _insert_helper(self, node: TSTNode, word: str, index: int, value) -> TSTNode:
        """Recursive insertion helper."""
        char = word[index]

        # Create new node if necessary
        if node is None:
            node = TSTNode(char)

        if char < node.char:
            node.left = self._insert_helper(node.left, word, index, value)
        elif char > node.char:
            node.right = self._insert_helper(node.right, word, index, value)
        else:
            # char == node.char
            if index + 1 < len(word):
                # More characters to process
                node.mid = self._insert_helper(node.mid, word, index + 1, value)
            else:
                # End of word
                if not node.is_end_of_word:
                    self.size += 1
                node.is_end_of_word = True
                node.value = value

        return node

    def search(self, word: str) -> bool:
        """
        Search for exact word.

        Time Complexity: O(m + log n) where m = word length, n = words
        """
        node = self._find_node(self.root, word, 0)
        return node is not None and node.is_end_of_word

    def _find_node(self, node: TSTNode, word: str, index: int) -> TSTNode:
        """Find node representing word."""
        if node is None or index >= len(word):
            return node

        char = word[index]

        if char < node.char:
            return self._find_node(node.left, word, index)
        elif char > node.char:
            return self._find_node(node.right, word, index)
        else:
            if index + 1 == len(word):
                return node
            return self._find_node(node.mid, word, index + 1)

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._find_node(self.root, prefix, 0) is not None

    def get_all_words_with_prefix(self, prefix: str) -> list:
        """Get all words starting with prefix."""
        results = []
        node = self._find_node(self.root, prefix, 0)

        if node is not None:
            if node.is_end_of_word:
                results.append(prefix)
            self._collect_words(node.mid, prefix, results)

        return results

    def _collect_words(self, node: TSTNode, prefix: str, results: list) -> None:
        """DFS to collect words."""
        if node is None:
            return

        self._collect_words(node.left, prefix, results)

        current = prefix + node.char
        if node.is_end_of_word:
            results.append(current)
        self._collect_words(node.mid, current, results)

        self._collect_words(node.right, prefix, results)

    def __len__(self) -> int:
        return self.size


# Example usage
tst = TernarySearchTrie()
words = ["cat", "cats", "dog", "dodge", "card", "care"]

for word in words:
    tst.insert(word)

print(f"Size: {len(tst)}")  # 6
print(f"Search 'cat': {tst.search('cat')}")  # True
print(f"Search 'ca': {tst.search('ca')}")    # False
print(f"Starts with 'ca': {tst.starts_with('ca')}")  # True
print(f"Words with 'ca': {tst.get_all_words_with_prefix('ca')}")  # ['cat', 'cats', 'card', 'care']
```

**Comparison: TST vs Standard Trie:**

| Aspect | Standard Trie | TST |
|--------|--------------|-----|
| **Space** | O(ALPHABET_SIZE * n * m) | O(3n) nodes |
| **Search** | O(m) | O(m + log n) |
| **Insertion** | O(m) | O(m + log n) |
| **When to use** | Fast lookups, large alphabet | Space-constrained, good balance |

**TST Advantages:**
- Much less memory than standard trie (3 pointers vs 26-256)
- Faster than hash table for prefix operations
- Natural alphabetic ordering

**TST Disadvantages:**
- Slightly slower than standard trie
- More complex implementation
- Not as cache-friendly

---

### Comparison of Trie Variations

| Variation | Space Complexity | Best Use Case | Key Feature |
|-----------|-----------------|---------------|-------------|
| **Standard Trie** | O(ALPHABET_SIZE * n * m) | Fast prefix ops, large datasets | Simple, fast |
| **Compressed Trie (Radix)** | O(n) nodes | Few common prefixes | Compressed paths |
| **Suffix Trie** | O(n²) | Pattern matching | All suffixes stored |
| **TST** | O(3n) | Space-constrained | 3 pointers per node |

---

## Common Patterns & Techniques

### Pattern 1: Dictionary / Word Search Problems

**Classic Problem:** Implement a dictionary with add and search functionality, supporting wildcards.

```python
class WordDictionary:
    """
    Add and search words - supports '.' wildcard.
    LeetCode 211: Design Add and Search Words Data Structure
    """

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        """Add word to dictionary. O(m) time."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Search word with wildcard support.
        '.' matches any single character.
        O(m * 26^k) where k is number of wildcards
        """
        return self._search_helper(self.root, word, 0)

    def _search_helper(self, node: TrieNode, word: str, index: int) -> bool:
        """Recursive search with wildcard support."""
        if index == len(word):
            return node.is_end_of_word

        char = word[index]

        if char == '.':
            # Wildcard - try all children
            for child in node.children.values():
                if self._search_helper(child, word, index + 1):
                    return True
            return False
        else:
            # Regular character
            if char not in node.children:
                return False
            return self._search_helper(node.children[char], word, index + 1)


# Example usage
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")

print(wd.search("pad"))  # False
print(wd.search("bad"))  # True
print(wd.search(".ad"))  # True (matches bad, dad, mad)
print(wd.search("b.."))  # True (matches bad)
```

**Key Technique:** Use DFS/backtracking when wildcards are involved.

---

### Pattern 2: Word Search II (Board Game)

**Problem:** Find all words from dictionary that can be formed on a 2D board (Boggle).

```python
class Solution:
    """
    LeetCode 212: Word Search II
    Given an m x n board and a list of words, find all words on the board.
    """

    def findWords(self, board: list[list[str]], words: list[str]) -> list[str]:
        """
        Time Complexity: O(m * n * 4^L) where L is max word length
        Space Complexity: O(W * L) for trie, W = number of words
        """
        # Build trie from words
        trie = Trie()
        for word in words:
            trie.insert(word)

        rows, cols = len(board), len(board[0])
        result = set()

        def dfs(r, c, node, path):
            """DFS on board with trie traversal."""
            # Bounds check
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return

            char = board[r][c]

            # Already visited or not in trie
            if char == '#' or char not in node.children:
                return

            node = node.children[char]
            path += char

            # Found a word
            if node.is_end_of_word:
                result.add(path)
                # Don't return - there might be longer words

            # Mark as visited
            board[r][c] = '#'

            # Explore neighbors
            dfs(r + 1, c, node, path)
            dfs(r - 1, c, node, path)
            dfs(r, c + 1, node, path)
            dfs(r, c - 1, node, path)

            # Restore
            board[r][c] = char

        # Start DFS from each cell
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, trie.root, "")

        return list(result)


# Example
board = [
    ['o', 'a', 'a', 'n'],
    ['e', 't', 'a', 'e'],
    ['i', 'h', 'k', 'r'],
    ['i', 'f', 'l', 'v']
]
words = ["oath", "pea", "eat", "rain"]

solution = Solution()
print(solution.findWords(board, words))  # ["oath", "eat"]
```

**Key Technique:** Build trie from dictionary, then DFS on board while traversing trie simultaneously.

**Optimization:** Prune trie nodes after finding words to avoid redundant searches.

---

### Pattern 3: Autocomplete / Top K Frequent

**Problem:** Implement autocomplete that returns top K most frequent words with given prefix.

```python
class AutocompleteSystem:
    """
    LeetCode 642: Design Search Autocomplete System
    Returns top 3 historical hot sentences with given prefix.
    """

    def __init__(self, sentences: list[str], times: list[int]):
        """
        Initialize with historical sentences and their frequencies.
        """
        self.trie = Trie()
        self.current_input = ""

        # Build trie with frequencies
        for sentence, count in zip(sentences, times):
            self._insert_with_frequency(sentence, count)

    def _insert_with_frequency(self, sentence: str, count: int) -> None:
        """Insert sentence with frequency tracking."""
        node = self.trie.root

        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.word = sentence
        node.frequency = getattr(node, 'frequency', 0) + count

    def input(self, c: str) -> list[str]:
        """
        Process input character and return top 3 suggestions.
        '#' marks end of input.
        """
        if c == '#':
            # Save current input
            self._insert_with_frequency(self.current_input, 1)
            self.current_input = ""
            return []

        self.current_input += c

        # Find all words with current prefix
        node = self.trie.root
        for char in self.current_input:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all words from this node with frequencies
        candidates = []
        self._collect_with_frequency(node, candidates)

        # Sort by frequency (desc) then lexicographically (asc)
        candidates.sort(key=lambda x: (-x[1], x[0]))

        # Return top 3
        return [word for word, freq in candidates[:3]]

    def _collect_with_frequency(self, node: TrieNode, results: list) -> None:
        """DFS to collect words with frequencies."""
        if node.is_end_of_word:
            results.append((node.word, node.frequency))

        for child in node.children.values():
            self._collect_with_frequency(child, results)


# Example usage
sentences = ["i love you", "island", "iroman", "i love leetcode"]
times = [5, 3, 2, 2]
system = AutocompleteSystem(sentences, times)

print(system.input('i'))  # ["i love you", "island", "i love leetcode"]
print(system.input(' '))  # ["i love you", "i love leetcode"]
print(system.input('a'))  # []
print(system.input('#'))  # []
```

**Key Technique:** Store frequency at nodes, collect candidates, and sort by frequency + lexicographic order.

---

### Pattern 4: Replace Words (Shortest Prefix)

**Problem:** Replace words with their shortest root form from dictionary.

```python
class Solution:
    """
    LeetCode 648: Replace Words
    Replace words with their shortest dictionary root.
    """

    def replaceWords(self, dictionary: list[str], sentence: str) -> str:
        """
        Time Complexity: O(D + S) where D = total chars in dictionary,
                         S = total chars in sentence
        """
        # Build trie from dictionary
        trie = Trie()
        for root in dictionary:
            trie.insert(root)

        def find_shortest_root(word):
            """Find shortest root of word in trie."""
            node = trie.root
            prefix = ""

            for char in word:
                if char not in node.children:
                    # No prefix found, return original word
                    return word

                prefix += char
                node = node.children[char]

                # Found a root
                if node.is_end_of_word:
                    return prefix

            # No root found
            return word

        # Process each word in sentence
        words = sentence.split()
        return ' '.join(find_shortest_root(word) for word in words)


# Example
solution = Solution()
dictionary = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
print(solution.replaceWords(dictionary, sentence))
# Output: "the cat was rat by the bat"
```

**Key Technique:** Traverse trie while building prefix; return immediately when hitting `is_end_of_word`.

---

### Pattern 5: Longest Word in Dictionary

**Problem:** Find longest word that can be built one character at a time.

```python
class Solution:
    """
    LeetCode 720: Longest Word in Dictionary
    Find longest word that can be built one character at a time.
    """

    def longestWord(self, words: list[str]) -> str:
        """
        Time Complexity: O(n * m) where n = words, m = avg length
        """
        trie = Trie()

        # Insert all words
        for word in words:
            trie.insert(word)

        # DFS to find longest word where all prefixes exist
        longest = ""

        def dfs(node, path):
            nonlocal longest

            # Update longest if current path is longer
            # (or same length but lexicographically smaller)
            if len(path) > len(longest) or \
               (len(path) == len(longest) and path < longest):
                longest = path

            # Only continue if all prefixes exist (all nodes are end of word)
            for char, child in sorted(node.children.items()):
                if child.is_end_of_word:
                    dfs(child, path + char)

        dfs(trie.root, "")
        return longest


# Example
solution = Solution()
words = ["w", "wo", "wor", "worl", "world"]
print(solution.longestWord(words))  # "world"

words2 = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
print(solution.longestWord(words2))  # "apple"
```

**Key Technique:** DFS through trie, only traverse paths where all intermediate nodes are end of word.

---

### Pattern 6: Maximum XOR of Two Numbers

**Problem:** Find maximum XOR of any two numbers in array using binary trie.

```python
class Solution:
    """
    LeetCode 421: Maximum XOR of Two Numbers in an Array
    Uses binary trie (bits as characters).
    """

    def findMaximumXOR(self, nums: list[int]) -> int:
        """
        Time Complexity: O(n * 32) = O(n)
        Space Complexity: O(n * 32) = O(n)
        """
        class BitTrie:
            def __init__(self):
                self.root = {}

            def insert(self, num):
                """Insert number as 32-bit binary."""
                node = self.root
                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    if bit not in node:
                        node[bit] = {}
                    node = node[bit]

            def find_max_xor(self, num):
                """Find number that gives max XOR with num."""
                node = self.root
                max_xor = 0

                for i in range(31, -1, -1):
                    bit = (num >> i) & 1
                    # Try to go opposite direction for max XOR
                    toggle_bit = 1 - bit

                    if toggle_bit in node:
                        max_xor |= (1 << i)
                        node = node[toggle_bit]
                    else:
                        node = node[bit]

                return max_xor

        trie = BitTrie()
        max_xor = 0

        # Insert all numbers and find max XOR
        for num in nums:
            trie.insert(num)
            max_xor = max(max_xor, trie.find_max_xor(num))

        return max_xor


# Example
solution = Solution()
print(solution.findMaximumXOR([3, 10, 5, 25, 2, 8]))  # 28 (5 XOR 25)
```

**Key Technique:** Use binary representation as trie path; greedily choose opposite bits for maximum XOR.

---

### Summary of Common Patterns

| Pattern | Key Technique | Complexity | Example Problem |
|---------|--------------|------------|-----------------|
| **Dictionary with Wildcards** | DFS with backtracking | O(m * 26^k) | LeetCode 211 |
| **Board Word Search** | Trie + DFS on grid | O(m*n*4^L) | LeetCode 212 |
| **Autocomplete Top K** | Frequency tracking + sorting | O(p + n log n) | LeetCode 642 |
| **Shortest Prefix** | Early termination on match | O(m) | LeetCode 648 |
| **Prefix Chain** | Check all prefixes exist | O(n*m) | LeetCode 720 |
| **Binary Trie** | Bit manipulation | O(n*log MAX) | LeetCode 421 |

---

## Common Problems

### Problem 1: Implement Trie (Prefix Tree)

**LeetCode 208: Implement Trie (Prefix Tree)**

```python
class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """

    def __init__(self):
        """Initialize your data structure here."""
        self.root = {}

    def insert(self, word: str) -> None:
        """Inserts a word into the trie. O(m) time."""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True  # End of word marker

    def search(self, word: str) -> bool:
        """Returns if the word is in the trie. O(m) time."""
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '#' in node

    def startsWith(self, prefix: str) -> bool:
        """Returns if there is any word in the trie that starts with the given prefix. O(p) time."""
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True


# Test
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    # True
print(trie.search("app"))      # False
print(trie.startsWith("app"))  # True
trie.insert("app")
print(trie.search("app"))      # True
```

**Key Points:**
- Use `'#'` or special marker for end of word
- Search requires end marker, startsWith doesn't
- Can use nested dictionaries for compact implementation

---

### Problem 2: Add and Search Word

**LeetCode 211: Design Add and Search Words Data Structure**

```python
class WordDictionary:
    """Support wildcard '.' that matches any character."""

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        """Add word. O(m) time."""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True

    def search(self, word: str) -> bool:
        """Search with wildcard support. O(m * 26^w) where w = wildcards."""
        def dfs(node, i):
            if i == len(word):
                return '#' in node

            char = word[i]
            if char == '.':
                # Try all possible characters
                for key in node:
                    if key != '#' and dfs(node[key], i + 1):
                        return True
                return False
            else:
                if char not in node:
                    return False
                return dfs(node[char], i + 1)

        return dfs(self.root, 0)


# Test
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")
print(wd.search("pad"))  # False
print(wd.search("bad"))  # True
print(wd.search(".ad"))  # True
print(wd.search("b.."))  # True
```

**Key Points:**
- Use DFS/recursion for wildcard handling
- Try all children when encountering '.'
- Backtracking naturally handled by recursion

---

### Problem 3: Word Search II

**LeetCode 212: Word Search II** (Already covered in patterns section)

---

### Problem 4: Replace Words

**LeetCode 648: Replace Words** (Already covered in patterns section)

---

### Problem 5: Longest Word with All Prefixes

**LeetCode 720: Longest Word in Dictionary** (Already covered in patterns section)

---

### Problem 6: Palindrome Pairs

**LeetCode 336: Palindrome Pairs**

```python
class Solution:
    """
    Find all pairs of distinct indices (i, j) where words[i] + words[j] is a palindrome.
    """

    def palindromePairs(self, words: list[str]) -> list[list[int]]:
        """
        Time Complexity: O(n * m²) where n = words, m = avg length
        """
        def is_palindrome(s):
            return s == s[::-1]

        # Build trie with reversed words
        trie = {}
        for idx, word in enumerate(words):
            node = trie
            for char in reversed(word):
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['#'] = idx

        result = []

        for i, word in enumerate(words):
            node = trie

            # Case 1: word + reversed(other_word) is palindrome
            for j, char in enumerate(word):
                # Check if remaining part of word is palindrome
                # and we've reached end of a reversed word in trie
                if '#' in node and node['#'] != i:
                    if is_palindrome(word[j:]):
                        result.append([i, node['#']])

                if char not in node:
                    break
                node = node[char]
            else:
                # Case 2: Reached end of word, check trie suffixes
                def dfs(n, path):
                    if '#' in n and n['#'] != i:
                        if is_palindrome(path):
                            result.append([i, n['#']])
                    for c, child in n.items():
                        if c != '#':
                            dfs(child, path + c)

                dfs(node, "")

        return result


# Example
solution = Solution()
words = ["abcd", "dcba", "lls", "s", "sssll"]
print(solution.palindromePairs(words))
# Output: [[0,1], [1,0], [3,2], [2,4]]
# "abcd" + "dcba" = "abcddcba"
# "lls" + "sssll" = "llssssll"
```

**Key Points:**
- Store reversed words in trie
- Check palindrome at each step of traversal
- Handle both prefix and suffix cases

---

### Problem 7: Map Sum Pairs

**LeetCode 677: Map Sum Pairs**

```python
class MapSum:
    """
    Implement a map with string keys and integer values.
    Support sum of values for all keys with a given prefix.
    """

    def __init__(self):
        self.trie = {}
        self.key_values = {}  # Store actual key-value pairs

    def insert(self, key: str, val: int) -> None:
        """Insert or update key-value pair. O(m) time."""
        # Calculate delta for updating trie sums
        delta = val - self.key_values.get(key, 0)
        self.key_values[key] = val

        # Update trie with delta
        node = self.trie
        for char in key:
            if char not in node:
                node[char] = {'sum': 0}
            node = node[char]
            node['sum'] = node.get('sum', 0) + delta

    def sum(self, prefix: str) -> int:
        """Return sum of all values with given prefix. O(p) time."""
        node = self.trie
        for char in prefix:
            if char not in node:
                return 0
            node = node[char]
        return node.get('sum', 0)


# Test
ms = MapSum()
ms.insert("apple", 3)
print(ms.sum("ap"))  # 3
ms.insert("app", 2)
print(ms.sum("ap"))  # 5
ms.insert("apple", 5)  # Update
print(ms.sum("ap"))  # 7
```

**Key Points:**
- Store cumulative sums at each node
- Handle updates by calculating delta
- Track actual key-values separately

---

## Applications

### 1. Autocomplete Systems

**Real-world example: Google Search Suggestions**

```python
class AutocompleteSystemAdvanced:
    """
    Production-grade autocomplete with ranking, caching, and personalization.
    """

    def __init__(self):
        self.trie = Trie()
        self.query_frequency = {}  # Track search frequencies
        self.cache = {}  # Cache popular prefix results
        self.max_suggestions = 10

    def index_documents(self, documents: list[str]) -> None:
        """Index documents for search."""
        for doc in documents:
            # Index full document and significant phrases
            self.trie.insert(doc.lower())

            # Also index individual words
            for word in doc.split():
                if len(word) >= 3:  # Minimum word length
                    self.trie.insert(word.lower())

    def search(self, prefix: str) -> list[tuple[str, int]]:
        """
        Get autocomplete suggestions with ranking.
        Returns: List of (suggestion, score) tuples
        """
        prefix = prefix.lower()

        # Check cache first
        if prefix in self.cache:
            return self.cache[prefix]

        # Get all matching words
        candidates = self.trie.get_all_words_with_prefix(prefix)

        # Score and rank candidates
        scored_candidates = []
        for word in candidates:
            score = self._calculate_score(word, prefix)
            scored_candidates.append((word, score))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: -x[1])

        # Take top suggestions
        results = scored_candidates[:self.max_suggestions]

        # Cache results
        self.cache[prefix] = results

        return results

    def _calculate_score(self, word: str, prefix: str) -> int:
        """
        Calculate relevance score for a word.
        Factors: frequency, length, exact prefix match
        """
        score = 0

        # Frequency score (from past searches)
        score += self.query_frequency.get(word, 0) * 100

        # Shorter words score higher (more specific)
        score += (100 - len(word))

        # Exact word match gets bonus
        if word == prefix:
            score += 1000

        # Words starting with prefix get bonus
        if word.startswith(prefix):
            score += 500

        return score

    def record_search(self, query: str) -> None:
        """Record that user searched for this query."""
        query = query.lower()
        self.query_frequency[query] = self.query_frequency.get(query, 0) + 1

        # Invalidate cache entries affected by this update
        for i in range(1, len(query) + 1):
            prefix = query[:i]
            if prefix in self.cache:
                del self.cache[prefix]

    def clear_cache(self) -> None:
        """Clear suggestion cache."""
        self.cache.clear()


# Example usage
autocomplete = AutocompleteSystemAdvanced()

# Index some documents
documents = [
    "Python programming tutorial",
    "Python data structures",
    "Python for beginners",
    "Java programming",
    "JavaScript frameworks"
]
autocomplete.index_documents(documents)

# Search
results = autocomplete.search("pyt")
print("Suggestions for 'pyt':")
for word, score in results:
    print(f"  {word} (score: {score})")

# Record user selection
autocomplete.record_search("python")

# Search again - "python" should rank higher now
results = autocomplete.search("pyt")
print("\nAfter recording search:")
for word, score in results:
    print(f"  {word} (score: {score})")
```

**Key Features:**
- Frequency-based ranking
- Caching for performance
- Scoring algorithm considering multiple factors
- Real-time updates

---

### 2. Spell Checkers

**Edit distance-based spell checking:**

```python
class SpellChecker:
    """
    Spell checker with suggestions based on edit distance.
    """

    def __init__(self, dictionary: list[str]):
        self.trie = Trie()
        for word in dictionary:
            self.trie.insert(word.lower())

    def is_correct(self, word: str) -> bool:
        """Check if word is spelled correctly."""
        return self.trie.search(word.lower())

    def get_suggestions(self, word: str, max_distance: int = 2) -> list[str]:
        """
        Get spelling suggestions within max_distance edits.
        Uses trie traversal with dynamic programming.
        """
        word = word.lower()
        suggestions = []

        def dfs(node, current_word, prev_row):
            """
            DFS with edit distance calculation.
            prev_row: DP array from previous character
            """
            cols = len(word) + 1
            current_row = [prev_row[0] + 1]  # Deletion

            # Calculate edit distance for current character
            for col in range(1, cols):
                insert_cost = current_row[col - 1] + 1
                delete_cost = prev_row[col] + 1
                replace_cost = prev_row[col - 1]

                if word[col - 1] != current_word[-1]:
                    replace_cost += 1

                current_row.append(min(insert_cost, delete_cost, replace_cost))

            # If edit distance is within threshold and word is complete
            if current_row[-1] <= max_distance and node.is_end_of_word:
                suggestions.append((current_word, current_row[-1]))

            # Only continue if there's potential for valid words
            if min(current_row) <= max_distance:
                for char, child_node in node.children.items():
                    dfs(child_node, current_word + char, current_row)

        # Start DFS from root
        first_row = list(range(len(word) + 1))
        dfs(self.trie.root, "", first_row)

        # Sort by edit distance, then alphabetically
        suggestions.sort(key=lambda x: (x[1], x[0]))

        return [word for word, _ in suggestions]

    def correct(self, text: str) -> str:
        """Auto-correct text."""
        words = text.split()
        corrected = []

        for word in words:
            if self.is_correct(word):
                corrected.append(word)
            else:
                # Get best suggestion
                suggestions = self.get_suggestions(word, max_distance=2)
                if suggestions:
                    corrected.append(suggestions[0])
                else:
                    corrected.append(word)  # No correction found

        return ' '.join(corrected)


# Example usage
dictionary = [
    "hello", "world", "python", "programming",
    "spell", "checker", "correct", "algorithm"
]

spell_checker = SpellChecker(dictionary)

# Check spelling
print(spell_checker.is_correct("hello"))  # True
print(spell_checker.is_correct("helo"))   # False

# Get suggestions
print(spell_checker.get_suggestions("helo"))  # ['hello']
print(spell_checker.get_suggestions("wrld"))  # ['world']
print(spell_checker.get_suggestions("spel"))  # ['spell']

# Auto-correct
text = "helo wrld, this is a pythom progam"
print(spell_checker.correct(text))
# Output: "hello world, this is a python programming"
```

**Key Techniques:**
- Edit distance (Levenshtein distance) with dynamic programming
- Trie traversal to find similar words efficiently
- Threshold-based suggestions

---

### 3. IP Routing (Longest Prefix Matching)

**Network routers use tries for IP address lookups:**

```python
class IPRouter:
    """
    IP router using trie for longest prefix matching.
    Stores IP addresses in binary trie.
    """

    def __init__(self):
        self.root = {}

    def ip_to_binary(self, ip: str, prefix_length: int = 32) -> str:
        """Convert IP address to binary string."""
        octets = ip.split('.')
        binary = ''.join(format(int(octet), '08b') for octet in octets)
        return binary[:prefix_length]

    def add_route(self, cidr: str, next_hop: str) -> None:
        """
        Add routing entry.
        cidr: IP in CIDR notation (e.g., "192.168.1.0/24")
        next_hop: Gateway address
        """
        ip, prefix_len = cidr.split('/')
        prefix_len = int(prefix_len)
        binary_ip = self.ip_to_binary(ip, prefix_len)

        node = self.root
        for bit in binary_ip:
            if bit not in node:
                node[bit] = {}
            node = node[bit]

        node['next_hop'] = next_hop
        node['cidr'] = cidr

    def lookup(self, ip: str) -> str:
        """
        Find next hop for IP address using longest prefix match.
        Returns next_hop address or None if no route found.
        """
        binary_ip = self.ip_to_binary(ip)

        node = self.root
        last_next_hop = None
        last_cidr = None

        # Traverse as far as possible, keeping track of last next_hop
        for bit in binary_ip:
            if 'next_hop' in node:
                last_next_hop = node['next_hop']
                last_cidr = node['cidr']

            if bit not in node:
                break

            node = node[bit]

        # Check final node
        if 'next_hop' in node:
            last_next_hop = node['next_hop']
            last_cidr = node['cidr']

        return last_next_hop, last_cidr

    def delete_route(self, cidr: str) -> bool:
        """Delete routing entry."""
        ip, prefix_len = cidr.split('/')
        prefix_len = int(prefix_len)
        binary_ip = self.ip_to_binary(ip, prefix_len)

        node = self.root
        for bit in binary_ip:
            if bit not in node:
                return False
            node = node[bit]

        if 'next_hop' in node:
            del node['next_hop']
            del node['cidr']
            return True

        return False


# Example usage
router = IPRouter()

# Add routes
router.add_route("192.168.1.0/24", "gateway1")
router.add_route("192.168.0.0/16", "gateway2")
router.add_route("10.0.0.0/8", "gateway3")
router.add_route("0.0.0.0/0", "default_gateway")  # Default route

# Lookup IPs
test_ips = [
    "192.168.1.100",  # Matches /24
    "192.168.5.50",   # Matches /16
    "10.5.10.20",     # Matches /8
    "8.8.8.8"         # Matches default
]

print("IP Routing Table Lookups:")
for ip in test_ips:
    next_hop, cidr = router.lookup(ip)
    print(f"{ip:20} -> {next_hop:20} (matched {cidr})")
```

**Output:**
```
192.168.1.100        -> gateway1             (matched 192.168.1.0/24)
192.168.5.50         -> gateway2             (matched 192.168.0.0/16)
10.5.10.20           -> gateway3             (matched 10.0.0.0/8)
8.8.8.8              -> default_gateway      (matched 0.0.0.0/0)
```

**Key Technique:** Longest prefix matching naturally handled by trie structure.

---

### 4. Word Games (Boggle Solver)

```python
class BoggleSolver:
    """
    Solve Boggle game - find all valid words on board.
    """

    def __init__(self, dictionary: list[str]):
        self.trie = Trie()
        for word in dictionary:
            if len(word) >= 3:  # Boggle minimum word length
                self.trie.insert(word.upper())

    def solve(self, board: list[list[str]]) -> set[str]:
        """
        Find all valid words on Boggle board.

        Time Complexity: O(m * n * 4^L) where L is max word length
        """
        rows, cols = len(board), len(board[0])
        found_words = set()

        def dfs(r, c, node, path, visited):
            """DFS with trie traversal."""
            # Bounds and visited check
            if (r < 0 or r >= rows or c < 0 or c >= cols or
                (r, c) in visited):
                return

            char = board[r][c]
            if char not in node.children:
                return

            node = node.children[char]
            path += char
            visited.add((r, c))

            # Found valid word
            if node.is_end_of_word and len(path) >= 3:
                found_words.add(path)

            # Explore all 8 neighbors
            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1),
                          (0,1), (1,-1), (1,0), (1,1)]:
                dfs(r + dr, c + dc, node, path, visited)

            visited.remove((r, c))

        # Start from each cell
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, self.trie.root, "", set())

        return found_words


# Example
dictionary = [
    "OATH", "PEAS", "EAT", "RAIN", "OATS", "TEA", "ETA"
]

board = [
    ['O', 'A', 'T', 'H'],
    ['E', 'T', 'A', 'E'],
    ['I', 'H', 'K', 'R'],
    ['I', 'F', 'L', 'V']
]

solver = BoggleSolver(dictionary)
words = solver.solve(board)
print(f"Found {len(words)} words:")
for word in sorted(words):
    print(f"  {word}")
```

---

### 5. DNA Sequence Analysis

```python
class DNAAnalyzer:
    """
    Analyze DNA sequences using suffix trie.
    """

    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.suffix_trie = SuffixTrie(self.sequence)

    def find_pattern(self, pattern: str) -> list[int]:
        """Find all occurrences of pattern in DNA sequence."""
        return self.suffix_trie.find_all_occurrences(pattern.upper())

    def longest_repeat(self) -> str:
        """Find longest repeated subsequence."""
        return self.suffix_trie.longest_repeated_substring()

    def find_motifs(self, min_length: int = 3, min_occurrences: int = 2) -> list[tuple[str, int]]:
        """
        Find repeated motifs (patterns) in DNA.
        Returns list of (motif, count) tuples.
        """
        motifs = {}

        # Check all substrings
        for i in range(len(self.sequence)):
            for j in range(i + min_length, len(self.sequence) + 1):
                motif = self.sequence[i:j]
                occurrences = len(self.find_pattern(motif))

                if occurrences >= min_occurrences:
                    if motif not in motifs or occurrences > motifs[motif]:
                        motifs[motif] = occurrences

        # Sort by count (descending)
        return sorted(motifs.items(), key=lambda x: -x[1])


# Example
dna = "ATCGATCGAATCGAATCG"
analyzer = DNAAnalyzer(dna)

print(f"DNA Sequence: {dna}")
print(f"\nFind 'ATCG': {analyzer.find_pattern('ATCG')}")
print(f"Longest repeat: {analyzer.longest_repeat()}")
print(f"\nCommon motifs:")
for motif, count in analyzer.find_motifs(min_length=3, min_occurrences=2)[:5]:
    print(f"  {motif}: {count} occurrences")
```

---

### 6. T9 Predictive Text

```python
class T9Dictionary:
    """
    T9 predictive text system (like old cell phones).
    Maps number sequences to words.
    """

    def __init__(self):
        self.trie = Trie()
        # T9 keyboard mapping
        self.t9_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        # Reverse mapping
        self.char_to_digit = {}
        for digit, chars in self.t9_map.items():
            for char in chars:
                self.char_to_digit[char] = digit

    def add_word(self, word: str) -> None:
        """Add word to dictionary."""
        self.trie.insert(word.lower())

    def word_to_digits(self, word: str) -> str:
        """Convert word to T9 digit sequence."""
        return ''.join(self.char_to_digit.get(c, '') for c in word.lower())

    def predict(self, digits: str) -> list[str]:
        """
        Get word predictions for digit sequence.
        """
        suggestions = []

        def dfs(node, path, digit_idx):
            """DFS to find matching words."""
            if digit_idx == len(digits):
                if node.is_end_of_word:
                    suggestions.append(path)
                return

            digit = digits[digit_idx]
            possible_chars = self.t9_map.get(digit, '')

            for char in possible_chars:
                if char in node.children:
                    dfs(node.children[char], path + char, digit_idx + 1)

        dfs(self.trie.root, "", 0)
        return suggestions


# Example
t9 = T9Dictionary()

# Add dictionary words
words = ["hello", "world", "help", "good", "home", "gone"]
for word in words:
    t9.add_word(word)

# Predict words
print("T9 Predictions:")
print(f"4663: {t9.predict('4663')}")  # "good", "gone", "home"
print(f"43556: {t9.predict('43556')}")  # "hello"
print(f"96753: {t9.predict('96753')}")  # "world"
```

---

## Optimizations & Tricks

### 1. Space Optimizations

#### a) Array vs Hash Map Choice

```python
class AdaptiveTrie:
    """
    Trie that chooses node implementation based on density.
    """

    class ArrayNode:
        """Use for dense children (many characters present)."""
        def __init__(self):
            self.children = [None] * 26
            self.is_end = False

        def density(self):
            return sum(1 for c in self.children if c) / 26

    class HashNode:
        """Use for sparse children (few characters present)."""
        def __init__(self):
            self.children = {}
            self.is_end = False

        def density(self):
            return len(self.children) / 26 if self.children else 0

    DENSITY_THRESHOLD = 0.5

    def choose_node_type(self, density: float):
        """Choose node type based on expected density."""
        if density > self.DENSITY_THRESHOLD:
            return self.ArrayNode()
        return self.HashNode()
```

**Rule of thumb:**
- **Array nodes**: Alphabet ≤ 26, dense children (>50% slots filled)
- **Hash nodes**: Large alphabet, sparse children (<50% slots filled)

#### b) Compressed Tries / Radix Trees

Already covered in Trie Variations section. Key benefit: O(n) nodes instead of O(n*m).

#### c) Lazy Deletion

Instead of physically removing nodes, mark them as deleted:

```python
class LazyDeleteTrie:
    """Trie with lazy deletion for better performance."""

    def __init__(self):
        self.root = TrieNode()
        self.deleted = set()  # Set of deleted words

    def delete(self, word: str) -> None:
        """Lazy deletion - O(1) time."""
        self.deleted.add(word)

    def search(self, word: str) -> bool:
        """Check if word exists and not deleted."""
        if word in self.deleted:
            return False
        # Normal trie search
        return self._standard_search(word)

    def garbage_collect(self) -> None:
        """Periodically rebuild trie without deleted words."""
        all_words = self._get_all_words()
        valid_words = [w for w in all_words if w not in self.deleted]

        # Rebuild trie
        self.root = TrieNode()
        for word in valid_words:
            self.insert(word)

        self.deleted.clear()
```

---

### 2. Time Optimizations

#### a) Caching Frequent Prefixes

```python
class CachedTrie(Trie):
    """Trie with LRU cache for frequent prefix queries."""

    def __init__(self, cache_size=1000):
        super().__init__()
        from functools import lru_cache

        # Cache autocomplete results
        @lru_cache(maxsize=cache_size)
        def cached_autocomplete(prefix):
            return tuple(self.get_all_words_with_prefix(prefix))

        self.cached_autocomplete = cached_autocomplete

    def insert(self, word):
        super().insert(word)
        # Invalidate cache on modifications
        self.cached_autocomplete.cache_clear()
```

#### b) Early Termination

For operations that don't need complete traversal:

```python
def exists_prefix(self, prefix: str) -> bool:
    """
    Check if ANY word starts with prefix.
    Returns immediately on finding first match.
    """
    node = self._find_node(prefix)
    return node is not None  # Don't need to search further

def find_first_word_with_prefix(self, prefix: str) -> str:
    """
    Find first word (not all words) with prefix.
    Much faster than finding all words.
    """
    node = self._find_node(prefix)
    if not node:
        return None

    # DFS until hitting first complete word
    path = prefix
    while not node.is_end_of_word:
        if not node.children:
            return None
        # Take any child
        char, node = next(iter(node.children.items()))
        path += char

    return path
```

#### c) Batch Operations

Process multiple operations together:

```python
def batch_insert(self, words: list[str]) -> None:
    """
    Insert multiple words efficiently.
    Can optimize by sorting words first (better cache locality).
    """
    # Sort for better cache performance
    words_sorted = sorted(words)

    for word in words_sorted:
        self.insert(word)

def batch_search(self, words: list[str]) -> dict[str, bool]:
    """Search multiple words, return results as dict."""
    return {word: self.search(word) for word in words}
```

---

### 3. Hybrid Approaches

#### a) Trie + Hash Table

For small datasets, use hash table; for large, use trie:

```python
class HybridDictionary:
    """
    Automatically choose between hash table and trie.
    """

    SIZE_THRESHOLD = 1000

    def __init__(self):
        self.size = 0
        self.hash_set = set()
        self.trie = None

    def insert(self, word: str) -> None:
        if self.size < self.SIZE_THRESHOLD:
            self.hash_set.add(word)
        else:
            # Convert to trie
            if self.trie is None:
                self.trie = Trie()
                for w in self.hash_set:
                    self.trie.insert(w)
                self.hash_set.clear()

            self.trie.insert(word)

        self.size += 1

    def search(self, word: str) -> bool:
        if self.trie:
            return self.trie.search(word)
        return word in self.hash_set
```

#### b) Trie + Bloom Filter

Use Bloom filter for fast negative lookups:

```python
class BloomTrie:
    """
    Trie with Bloom filter for fast negative answers.
    """

    def __init__(self):
        self.trie = Trie()
        self.bloom = BloomFilter(size=10000, hash_count=3)

    def insert(self, word: str) -> None:
        self.trie.insert(word)
        self.bloom.add(word)

    def search(self, word: str) -> bool:
        # Fast negative check
        if word not in self.bloom:
            return False  # Definitely not present

        # Might be present, check trie
        return self.trie.search(word)
```

---

### 4. Memory Management

#### a) Reference Counting

Track references to safely delete nodes:

```python
class RefCountedTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.ref_count = 0  # Number of words using this node

    def increment_ref(self):
        self.ref_count += 1

    def decrement_ref(self):
        self.ref_count -= 1
        return self.ref_count == 0  # Can be deleted if no refs
```

#### b) Node Pooling

Reuse deleted nodes instead of allocating new ones:

```python
class PooledTrie:
    """Trie with node pooling for memory efficiency."""

    def __init__(self):
        self.root = TrieNode()
        self.node_pool = []  # Pool of reusable nodes

    def _get_node(self):
        """Get node from pool or create new."""
        if self.node_pool:
            node = self.node_pool.pop()
            node.children.clear()
            node.is_end_of_word = False
            return node
        return TrieNode()

    def _return_node(self, node):
        """Return node to pool for reuse."""
        self.node_pool.append(node)
```

---

## Advantages & Disadvantages

### Advantages

1. **Predictable O(m) Performance**
   - Lookup time depends only on key length, not dataset size
   - No worst-case degradation like hash tables with collisions
   - Consistent performance regardless of n (number of keys)

2. **Efficient Prefix Operations**
   - Find all words with prefix in O(p + n) time
   - Autocomplete naturally supported
   - Longest common prefix queries
   - Prefix counting

3. **No Hash Collisions**
   - Unlike hash tables, no collision resolution needed
   - No rehashing required
   - Deterministic behavior

4. **Space-Efficient for Common Prefixes**
   - Shared prefixes stored once
   - Example: 1000 words starting with "inter" share first 5 characters
   - Can compress further with radix trees

5. **Alphabetically Sorted Iteration**
   - In-order traversal gives sorted results
   - Useful for sorted autocomplete suggestions
   - Range queries possible

6. **Pattern Matching**
   - Wildcard searches supported
   - Regular expression matching possible
   - Edit distance queries feasible

7. **No Rebalancing**
   - Unlike BSTs, no rebalancing needed
   - Simpler implementation than AVL/Red-Black trees
   - Predictable structure

### Disadvantages

1. **High Memory Overhead**
   - Each node needs pointers for alphabet size
   - 26 pointers per node for lowercase English = 208 bytes on 64-bit
   - Sparse tries waste space (many NULL pointers)
   - Worse than hash tables for random strings

2. **Cache-Unfriendly**
   - Pointer chasing hurts CPU cache performance
   - Non-contiguous memory layout
   - Multiple cache misses per lookup
   - Arrays or hash tables more cache-friendly

3. **Not Suitable for Dense Numeric Keys**
   - Huge alphabet size for integers
   - Better to use direct addressing or hash table
   - Binary trie possible but often not optimal

4. **Complex Implementation**
   - More complex than hash tables
   - Deletion is tricky (need to clean up nodes)
   - More code, more bugs
   - Edge cases (empty string, single char)

5. **Poor for Random Access**
   - No direct access to arbitrary key
   - Must traverse from root every time
   - Hash tables provide O(1) average access

6. **Space Overhead for Unique Strings**
   - No benefit if all strings are completely different
   - Each character needs full node
   - Hash table more efficient in this case

7. **Limited to String-like Keys**
   - Naturally suited for strings
   - Awkward for other data types
   - Requires serialization for complex keys

### When to Use Tries

**✓ Use Tries when:**
- Many strings with **common prefixes** (autocomplete, dictionaries)
- **Prefix-based queries** are frequent
- Need **sorted string iteration**
- **Wildcards or pattern matching** required
- **Predictable performance** more important than average-case speed
- Dataset size is large (prefix sharing benefits)
- Memory allows for pointer overhead

**Examples:**
- Autocomplete systems
- Spell checkers
- IP routing tables
- Dictionary implementations
- Word games (Boggle, Scrabble)
- DNA sequence analysis

### When NOT to Use Tries

**✗ Avoid Tries when:**
- **Small datasets** (< 100 items) - hash table better
- **No common prefixes** - wasted space
- **Memory severely constrained** - hash table more compact
- **Random string access only** - hash table faster
- **Numeric keys** - direct addressing or hash table better
- **Need average O(1) lookup** - hash table wins

**Examples:**
- Configuration key-value pairs
- User ID lookups
- Small word lists
- Random UUID storage
- Simple existence checks

### Comparison Summary

| Criterion | Trie | Hash Table | BST |
|-----------|------|------------|-----|
| **Lookup** | O(m) | O(1) avg | O(log n) |
| **Prefix Search** | O(p) | O(n) | O(log n + k) |
| **Space** | High | Medium | Low |
| **Sorted Order** | Yes | No | Yes |
| **Collisions** | No | Yes | No |
| **Implementation** | Complex | Simple | Medium |
| **Cache Friendly** | No | Yes | No |

---

## Comparison with Other Data Structures

### 1. Trie vs Hash Table

| Aspect | Trie | Hash Table |
|--------|------|------------|
| **Average Lookup** | O(m) | O(1) |
| **Worst Lookup** | O(m) | O(n) with collisions |
| **Prefix Search** | O(p + k) | O(n) - must check all |
| **Space Complexity** | O(ALPHABET * n * m) | O(n * m) |
| **Collision Handling** | Not needed | Required |
| **Sorted Iteration** | Natural | Requires sorting |
| **Memory Overhead** | High (pointers) | Lower |
| **Cache Performance** | Poor | Better |
| **Wildcards** | Efficient | Inefficient |

**Example comparison:**

```python
# Scenario: Store 10,000 words, frequently search by prefix

# Hash Table approach
hash_table = set(words)
# Pros: Fast exact lookup O(1)
# Cons: Prefix search requires checking all 10,000 words

# Trie approach
trie = Trie()
for word in words:
    trie.insert(word)
# Pros: Prefix search only traverses relevant branch
# Cons: More memory, O(m) lookup instead of O(1)

# Verdict: Use Trie if prefix operations common, else Hash Table
```

---

### 2. Trie vs Binary Search Tree (BST)

| Aspect | Trie | Balanced BST |
|--------|------|--------------|
| **Search** | O(m) | O(log n) |
| **Insert** | O(m) | O(log n) |
| **Delete** | O(m) | O(log n) |
| **Prefix Search** | O(p + k) | O(log n + k) |
| **Space** | Higher | Lower |
| **Balancing** | Not needed | Required (AVL, RB) |
| **Key Type** | Strings | Any comparable |
| **Sorted Iteration** | Yes | Yes |

**When Trie is better than BST:**
- String keys with common prefixes
- Prefix operations are frequent
- Want O(m) instead of O(log n) where n >> m

**When BST is better than Trie:**
- Non-string keys
- No prefix operations needed
- Memory constrained
- Need range queries on non-prefix ranges

```python
# Example: English dictionary (50,000 words, avg length 8 chars)

# Trie: O(8) = 8 operations regardless of dictionary size
trie.search("computer")  # Always 8 character checks

# BST: O(log 50000) ≈ 16 comparisons
bst.search("computer")  # Up to 16 string comparisons

# Verdict: Trie slightly faster, but uses more memory
```

---

### 3. Trie vs Suffix Array

| Aspect | Trie | Suffix Array |
|--------|------|--------------|
| **Build Time** | O(n²) for suffix trie | O(n log n) |
| **Space** | O(n²) worst case | O(n) |
| **Search** | O(m + occ) | O(m log n + occ) |
| **Pattern Matching** | Excellent | Very Good |
| **Implementation** | Complex | Moderate |

**Suffix Trie vs Suffix Array:**

For text: "banana"

```python
# Suffix Trie: Stores all suffixes in trie
# Space: O(n²) = 36 nodes worst case
# Search pattern: O(m)

# Suffix Array: Sorted array of suffix positions
# Space: O(n) = 6 integers
# Search pattern: O(m log n) with binary search

# Verdict: Suffix arrays more space-efficient
# Suffix trees (compressed tries) competitive
```

---

### 4. Trie vs Ternary Search Tree (TST)

| Aspect | Standard Trie | TST |
|--------|--------------|-----|
| **Space per Node** | ALPHABET_SIZE pointers | 3 pointers |
| **Search** | O(m) | O(m + log n) |
| **Memory** | High | Much lower |
| **Prefix Ops** | Fast | Slightly slower |
| **Large Alphabet** | Very expensive | Manageable |

**Example: Unicode strings (alphabet size = 65,536)**

```python
# Standard Trie node
# Memory: 65,536 * 8 bytes = 524 KB per node!
# Impractical for large alphabets

# TST node
# Memory: 3 * 8 bytes = 24 bytes per node
# Practical for any alphabet size

# Verdict: TST better for large alphabets
```

---

### 5. Trie vs Set (for membership testing)

| Operation | Trie | Set (Hash) |
|-----------|------|------------|
| **Add** | O(m) | O(1) avg |
| **Contains** | O(m) | O(1) avg |
| **Prefix Search** | O(p + k) | O(n) |
| **Memory** | High | Lower |

**When to choose:**

```python
# Just checking if words exist → Use Set
words = {"apple", "banana", "cherry"}
"apple" in words  # O(1)

# Need prefix operations → Use Trie
trie = Trie(words)
trie.get_all_words_with_prefix("app")  # O(p + k)

# Need both → Use Both!
word_set = set(words)  # Fast membership
word_trie = Trie(words)  # Fast prefix search
```

---

### 6. General Decision Tree

```
Do you need prefix operations (autocomplete, search suggestions)?
│
├─ YES → Consider Trie
│   │
│   ├─ Memory constrained? → Compressed Trie/Radix Tree
│   ├─ Large alphabet? → TST
│   └─ Small dataset (<100 items)? → Hash Table might still be better
│
└─ NO → Don't use Trie
    │
    ├─ Need sorted order? → BST (TreeMap)
    ├─ Just membership testing? → Hash Set
    ├─ Need fast lookup? → Hash Table
    └─ Pattern matching in text? → Suffix Array/Tree
```

---

## Complexity Analysis

### Time Complexity Deep Dive

#### Insert: O(m)

**Why?** Must visit each character once.

```python
def insert(word):  # word length = m
    node = root
    for char in word:  # m iterations
        if char not in node.children:  # O(1) with hash map
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end = True  # O(1)
# Total: O(m)
```

**Breakdown:**
- Loop m times: O(m)
- Each iteration: O(1) hash map access
- Total: O(m)

**Best case = Average case = Worst case = O(m)**

#### Search: O(m)

**Why?** Must check each character.

```python
def search(word):  # word length = m
    node = root
    for char in word:  # m iterations
        if char not in node.children:  # O(1)
            return False
        node = node.children[char]  # O(1)
    return node.is_end  # O(1)
# Total: O(m)
```

**Best case:** O(1) if first char doesn't exist
**Average/Worst case:** O(m)

#### Delete: O(m)

**Why?** Must traverse to word (O(m)) + cleanup (O(m))

```python
def delete(word):
    # Phase 1: Traverse to word - O(m)
    # Phase 2: Recursively cleanup - O(m)
# Total: O(m)
```

#### Prefix Search: O(p + n*k)

**Why?** Navigate to prefix (O(p)) + collect all words (O(n*k))

```python
def get_all_words_with_prefix(prefix):
    node = find_node(prefix)  # O(p)
    results = []
    dfs_collect(node, prefix, results)  # O(n*k)
    return results
# Total: O(p + n*k)
# where n = number of results, k = avg length
```

### Space Complexity Analysis

#### Space per Node

**Hash map implementation:**
```python
class TrieNode:
    children = {}              # Dict overhead: 240 bytes (Python)
    is_end_of_word = False     # Bool: 28 bytes
    # Total: ~280 bytes per node (Python 3.10)
```

**Array implementation (26 children):**
```python
class TrieNode:
    children = [None] * 26     # 26 * 8 = 208 bytes (pointers)
    is_end_of_word = False     # 1 byte
    # Total: ~210 bytes per node (64-bit system)
```

#### Total Space Complexity

**Worst case:** O(ALPHABET_SIZE × n × m)
- n = number of words
- m = average word length
- ALPHABET_SIZE = 26 for lowercase

**Example:** 1000 words, avg length 10, alphabet 26

```python
# Worst case (no shared prefixes)
nodes = 1000 * 10 = 10,000 nodes
memory_per_node = 26 * 8 bytes = 208 bytes
total = 10,000 * 208 = 2,080,000 bytes ≈ 2 MB

# Best case (maximum sharing)
# E.g., words are: "a", "aa", "aaa", "aaaa", ...
nodes = m = 10 nodes only
total = 10 * 208 = 2,080 bytes ≈ 2 KB
```

**Actual space depends on prefix sharing!**

#### Comparison: Trie vs Hash Table

```python
# Hash Table
# Space: O(n * m) = total characters
# 1000 words * 10 chars = 10,000 chars = ~10 KB

# Trie (worst case, no sharing)
# Space: O(n * m * ALPHABET_SIZE) = 2 MB (from above)

# Trie (with 50% prefix sharing)
# Space: ~1 MB

# Verdict: Hash table much more space-efficient
# unless prefix sharing is significant
```

### Practical Performance Considerations

#### 1. Cache Effects

**Tries are cache-unfriendly:**

```python
# Trie traversal
node = root
for char in "hello":
    node = node.children[char]  # Pointer dereference
    # Each dereference might be cache miss!

# Cache misses: Up to 5 (one per character)
```

**Array/Hash Table is cache-friendly:**

```python
# Array access
words = ["hello", "world", ...]
if "hello" in set(words):  # Likely 1 cache miss
    ...
```

**Impact:** Tries can be 2-3x slower than expected due to cache misses.

#### 2. Memory Allocation Overhead

Each node allocation has overhead:
- Heap allocation: ~16-32 bytes overhead
- Memory alignment: wasted bytes
- Fragmentation: non-contiguous memory

**Example:**
```python
# Logical node size: 208 bytes
# Actual allocation: 240 bytes (due to overhead)
# Wasted: 15% of memory!
```

#### 3. Pointer Size Impact

**32-bit vs 64-bit systems:**

```python
# 32-bit: pointers are 4 bytes
array_node = [None] * 26  # 26 * 4 = 104 bytes

# 64-bit: pointers are 8 bytes
array_node = [None] * 26  # 26 * 8 = 208 bytes

# 64-bit uses 2x memory for pointers!
```

#### 4. Alphabet Size Impact

```python
# Lowercase only (26)
memory_per_node = 26 * 8 = 208 bytes

# Alphanumeric (62)
memory_per_node = 62 * 8 = 496 bytes

# ASCII printable (95)
memory_per_node = 95 * 8 = 760 bytes

# Unicode (65,536)
memory_per_node = 65,536 * 8 = 524,288 bytes = 512 KB per node!
# → Must use hash map or TST for Unicode
```

### Amortized Analysis

**Insert with dynamic resizing (hash map children):**

```python
# Hash map resizes when load factor exceeds threshold
# Resize cost: O(current_size)
# Frequency: After O(current_size) insertions
# Amortized: O(1) per insertion

# Therefore:
# Insert word of length m
# Hash map operations: O(1) amortized
# Total insert: O(m) amortized
```

### Space-Time Tradeoffs

**Technique 1: Lazy Deletion**
- Time: O(1) delete (just mark)
- Space: Wasted nodes remain
- Tradeoff: Fast delete, more memory

**Technique 2: Immediate Cleanup**
- Time: O(m) delete (cleanup recursion)
- Space: Minimal waste
- Tradeoff: Slower delete, less memory

**Technique 3: Compressed Trie**
- Time: Slightly slower (string comparisons)
- Space: Much better (fewer nodes)
- Tradeoff: Complexity for space savings

### Big-O Summary Table

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Insert | O(m) | O(m) worst | m = word length |
| Search | O(m) | O(1) | |
| Delete | O(m) | O(1) | Assuming cleanup |
| Prefix | O(p) | O(1) | Just check existence |
| Autocomplete | O(p+n*k) | O(n*k) | n results, k avg length |
| Total Space | O(A*n*m) | - | A = alphabet size |

---

## Interview Tips & Patterns

### Common Interview Question Signals

**"Trie" Red Flags - When interviewer likely wants a trie:**

1. ✓ "Find all words with prefix..."
2. ✓ "Autocomplete system..."
3. ✓ "Dictionary with wildcards..."
4. ✓ "Spell checker suggestions..."
5. ✓ "Group anagrams" → Wait, no! Hash table better
6. ✓ "Search in 2D board for words..." (Word Search II)
7. ✓ "Multiple string matching..."
8. ✓ "IP routing / longest prefix match..."

**Keywords to listen for:**
- Prefix, autocomplete, dictionary, words
- Multiple strings to search
- Pattern matching with wildcards
- Search suggestions

### Implementation Checklist

When implementing a trie in an interview:

```python
# ✓ Step 1: Define TrieNode
class TrieNode:
    def __init__(self):
        self.children = {}           # ← Choose dict vs array
        self.is_end_of_word = False  # ← Don't forget this!
        # self.word = None           # ← Optional: store word here

# ✓ Step 2: Define Trie class
class Trie:
    def __init__(self):
        self.root = TrieNode()       # ← Initialize root

    # ✓ Step 3: Implement core operations
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True   # ← Critical!

    def search(self, word):
        node = self._find_node(word)
        return node and node.is_end_of_word  # ← Check flag!

    def _find_node(self, prefix):    # ← Helper function
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

**Interview checklist:**
- [ ] Define TrieNode with children dict/array
- [ ] Include `is_end_of_word` flag
- [ ] Initialize root in Trie constructor
- [ ] Implement insert, search, startsWith
- [ ] Handle empty strings
- [ ] Consider case sensitivity
- [ ] Test with examples

### Common Mistakes to Avoid

#### Mistake 1: Forgetting `is_end_of_word`

```python
# ❌ WRONG
def search(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return True  # ← BUG: Returns True for prefixes too!

# ✓ CORRECT
def search(self, word):
    node = self._find_node(word)
    return node is not None and node.is_end_of_word  # ← Check flag!
```

#### Mistake 2: Not Handling Empty String

```python
# ❌ WRONG
def insert(self, word):
    node = self.root
    for char in word:  # ← Breaks on empty string
        ...

# ✓ CORRECT
def insert(self, word):
    if not word:  # ← Handle empty string
        self.root.is_end_of_word = True
        return
    node = self.root
    for char in word:
        ...
```

#### Mistake 3: Memory Leaks in Deletion

```python
# ❌ WRONG
def delete(self, word):
    node = self._find_node(word)
    node.is_end_of_word = False  # ← Nodes not cleaned up!

# ✓ CORRECT
def delete(self, word):
    def helper(node, word, index):
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            return len(node.children) == 0  # ← Can delete if no children

        char = word[index]
        if char not in node.children:
            return False

        child = node.children[char]
        should_delete = helper(child, word, index + 1)

        if should_delete:
            del node.children[char]
            return not node.is_end_of_word and len(node.children) == 0

        return False

    helper(self.root, word, 0)
```

#### Mistake 4: Inefficient Wildcard Handling

```python
# ❌ WRONG: Try to handle wildcards in regular search
def search(self, word):
    if '.' in word:
        # ... complex logic mixed with regular search
        pass

# ✓ CORRECT: Separate methods
def search(self, word):
    # Regular search
    ...

def search_with_wildcards(self, pattern):
    # DFS with wildcard handling
    def dfs(node, i):
        if i == len(pattern):
            return node.is_end_of_word

        if pattern[i] == '.':
            return any(dfs(child, i+1) for child in node.children.values())
        else:
            if pattern[i] not in node.children:
                return False
            return dfs(node.children[pattern[i]], i+1)

    return dfs(self.root, 0)
```

#### Mistake 5: Wrong Alphabet Size Choice

```python
# ❌ WRONG: Using array for variable characters
class TrieNode:
    def __init__(self):
        self.children = [None] * 26  # ← What about uppercase? Numbers?

# ✓ CORRECT: Use hash map for flexibility
class TrieNode:
    def __init__(self):
        self.children = {}  # ← Works for any character
```

### Problem-Solving Template

**Standard trie problem approach:**

```python
# Step 1: Build the trie
trie = Trie()
for word in dictionary:
    trie.insert(word)

# Step 2: Query/Traverse the trie
# Pattern A: Simple query
result = trie.search(query_word)

# Pattern B: DFS traversal
def dfs(node, path, results):
    if node.is_end_of_word:
        results.append(path)
    for char, child in node.children.items():
        dfs(child, path + char, results)

# Pattern C: Simultaneous traversal (e.g., Word Search II)
def dfs_grid_with_trie(r, c, trie_node, path):
    char = board[r][c]
    if char not in trie_node.children:
        return

    next_node = trie_node.children[char]
    if next_node.is_end_of_word:
        found_words.add(path + char)

    # Continue DFS...

# Step 3: Collect and return results
return results
```

### Time Complexity Analysis in Interviews

**Always analyze complexity:**

```python
# Interviewer: "What's the time complexity?"

# Your answer should be structured:
"The time complexity is O(m) where m is the length of the word.

Breaking it down:
- We iterate through each character exactly once: O(m)
- At each character, we do a hash map lookup: O(1)
- Total: O(m)

The space complexity is O(1) for the search operation itself,
not counting the space used by the trie structure, which is
O(ALPHABET_SIZE * N * M) in the worst case, where N is the
number of words and M is the average word length."
```

### Optimization Discussion Points

**When interviewer asks "Can you optimize?":**

1. **Space optimization:**
   - "We could use a compressed trie (radix tree) to reduce nodes"
   - "For fixed alphabet, array-based nodes are more cache-friendly"
   - "Lazy deletion saves time at cost of space"

2. **Time optimization:**
   - "Cache frequent prefix queries with LRU cache"
   - "Early termination if we don't need all results"
   - "Batch operations for better cache locality"

3. **Trade-offs:**
   - "Hash map children: flexible but slower lookup"
   - "Array children: faster but wastes space"
   - "TST: balanced space-time compromise"

### Code Interview Best Practices

1. **Start with clarifying questions:**
   - "What's the alphabet size? Just lowercase?"
   - "Do I need to handle Unicode?"
   - "Should search be case-sensitive?"
   - "Can words be empty strings?"

2. **Explain your approach:**
   - "I'll use a trie because we need efficient prefix operations"
   - "Each node represents a character position"
   - "I'll mark end-of-word with a boolean flag"

3. **Walk through an example:**
   ```
   "Let me trace through inserting 'cat':
   - Start at root
   - Add 'c' as child
   - Add 'a' as child of 'c'
   - Add 't' as child of 'a'
   - Mark 't' as end of word"
   ```

4. **Test your code:**
   - Test normal case: "apple"
   - Test edge cases: "", "a", same prefix words
   - Test wildcards if applicable

5. **Discuss follow-ups:**
   - "We could add word frequency for ranking"
   - "Could optimize with Bloom filter for negative queries"
   - "For production, would add persistence layer"

---

## Real-World Implementation Considerations

### 1. Thread Safety

**Problem:** Multiple threads inserting/searching concurrently

**Solution A: Coarse-grained locking**

```python
import threading

class ThreadSafeTrie:
    """Simple thread-safe trie with global lock."""

    def __init__(self):
        self.root = TrieNode()
        self.lock = threading.RLock()  # Reentrant lock

    def insert(self, word):
        with self.lock:
            # Normal insert logic
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

    def search(self, word):
        with self.lock:
            # Normal search logic
            ...
```

**Pros:** Simple, correct
**Cons:** Poor concurrency (global lock bottleneck)

**Solution B: Fine-grained locking**

```python
class FineLockTrieNode:
    """Node with its own lock."""

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.lock = threading.RLock()


class FineLockTrie:
    """Trie with per-node locking."""

    def __init__(self):
        self.root = FineLockTrieNode()

    def insert(self, word):
        node = self.root

        for char in word:
            node.lock.acquire()

            if char not in node.children:
                node.children[char] = FineLockTrieNode()

            next_node = node.children[char]
            node.lock.release()
            node = next_node

        node.lock.acquire()
        node.is_end_of_word = True
        node.lock.release()
```

**Pros:** Better concurrency
**Cons:** Complex, risk of deadlocks, overhead

**Solution C: Read-write locks**

```python
from threading import RLock
from readerwriterlock import rwlock

class RWLockTrie:
    """Trie with read-write lock (many readers, few writers)."""

    def __init__(self):
        self.root = TrieNode()
        self.rwlock = rwlock.RWLockFair()

    def insert(self, word):
        with self.rwlock.gen_wlock():  # Write lock
            # Insert logic
            ...

    def search(self, word):
        with self.rwlock.gen_rlock():  # Read lock (concurrent reads OK)
            # Search logic
            ...
```

**Pros:** Optimizes for read-heavy workloads
**Cons:** Still serializes writes

---

### 2. Persistence / Serialization

**Serialize trie to disk:**

```python
import json
import pickle

class PersistentTrie(Trie):
    """Trie with save/load capabilities."""

    def save_to_file(self, filename):
        """Save trie to file."""

        # Option 1: JSON (human-readable)
        def serialize_node(node):
            return {
                'children': {char: serialize_node(child)
                            for char, child in node.children.items()},
                'is_end': node.is_end_of_word
            }

        data = serialize_node(self.root)

        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, filename):
        """Load trie from file."""

        def deserialize_node(data):
            node = TrieNode()
            node.is_end_of_word = data['is_end']
            node.children = {char: deserialize_node(child_data)
                           for char, child_data in data['children'].items()}
            return node

        with open(filename, 'r') as f:
            data = json.load(f)

        self.root = deserialize_node(data)

    def save_binary(self, filename):
        """Save using pickle (faster, smaller)."""
        with open(filename, 'wb') as f:
            pickle.dump(self.root, f)

    def load_binary(self, filename):
        """Load from pickle file."""
        with open(filename, 'rb') as f:
            self.root = pickle.load(f)


# Usage
trie = PersistentTrie()
# ... insert words ...
trie.save_to_file('dictionary.json')

# Later...
trie2 = PersistentTrie()
trie2.load_from_file('dictionary.json')
```

**Database storage:**

```python
class DatabaseTrie:
    """Store trie in database (SQL)."""

    def __init__(self, db_connection):
        self.db = db_connection
        self._create_tables()

    def _create_tables(self):
        """Create tables for trie nodes."""
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS trie_nodes (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                char TEXT,
                is_end_of_word BOOLEAN,
                FOREIGN KEY (parent_id) REFERENCES trie_nodes(id)
            )
        ''')

    def insert(self, word):
        """Insert word into database."""
        parent_id = None  # Root

        for char in word:
            # Find or create node
            cursor = self.db.execute('''
                SELECT id FROM trie_nodes
                WHERE parent_id = ? AND char = ?
            ''', (parent_id, char))

            row = cursor.fetchone()
            if row:
                parent_id = row[0]
            else:
                cursor = self.db.execute('''
                    INSERT INTO trie_nodes (parent_id, char, is_end_of_word)
                    VALUES (?, ?, ?)
                ''', (parent_id, char, False))
                parent_id = cursor.lastrowid

        # Mark as end of word
        self.db.execute('''
            UPDATE trie_nodes SET is_end_of_word = ? WHERE id = ?
        ''', (True, parent_id))

        self.db.commit()
```

---

### 3. Scalability

**Problem:** Trie too large for single machine memory

**Solution A: Sharding by prefix**

```python
class ShardedTrie:
    """Distribute trie across multiple shards based on first character."""

    def __init__(self, num_shards=26):
        self.shards = [Trie() for _ in range(num_shards)]
        self.num_shards = num_shards

    def _get_shard(self, word):
        """Determine which shard to use."""
        if not word:
            return 0
        # Simple: hash first character
        return ord(word[0].lower()) % self.num_shards

    def insert(self, word):
        shard = self._get_shard(word)
        self.shards[shard].insert(word)

    def search(self, word):
        shard = self._get_shard(word)
        return self.shards[shard].search(word)

    def get_all_words_with_prefix(self, prefix):
        shard = self._get_shard(prefix)
        return self.shards[shard].get_all_words_with_prefix(prefix)
```

**Solution B: Distributed trie (multiple machines)**

```python
class DistributedTrie:
    """Trie distributed across multiple machines."""

    def __init__(self, shard_urls):
        """
        shard_urls: List of URLs to trie shard servers
        e.g., ['http://shard1:8000', 'http://shard2:8000']
        """
        self.shards = shard_urls

    def _get_shard_url(self, word):
        """Consistent hashing to determine shard."""
        shard_idx = hash(word[0]) % len(self.shards)
        return self.shards[shard_idx]

    def insert(self, word):
        """Send insert request to appropriate shard."""
        url = self._get_shard_url(word)
        response = requests.post(f'{url}/insert', json={'word': word})
        return response.json()

    def search(self, word):
        """Send search request to appropriate shard."""
        url = self._get_shard_url(word)
        response = requests.get(f'{url}/search', params={'word': word})
        return response.json()['found']
```

---

### 4. Testing Strategies

**Unit tests:**

```python
import unittest

class TestTrie(unittest.TestCase):
    """Comprehensive trie test suite."""

    def setUp(self):
        self.trie = Trie()

    def test_empty_trie(self):
        """Test operations on empty trie."""
        self.assertFalse(self.trie.search("hello"))
        self.assertFalse(self.trie.starts_with("h"))
        self.assertEqual(len(self.trie), 0)

    def test_insert_and_search(self):
        """Test basic insert and search."""
        self.trie.insert("hello")
        self.assertTrue(self.trie.search("hello"))
        self.assertFalse(self.trie.search("hell"))  # Prefix only
        self.assertTrue(self.trie.starts_with("hell"))

    def test_duplicate_insert(self):
        """Test inserting same word twice."""
        self.trie.insert("hello")
        self.trie.insert("hello")
        self.assertEqual(len(self.trie), 1)  # Should not duplicate

    def test_prefix_sharing(self):
        """Test words with shared prefixes."""
        words = ["cat", "cats", "caterpillar", "dog"]
        for word in words:
            self.trie.insert(word)

        self.assertEqual(len(self.trie), 4)
        self.assertTrue(all(self.trie.search(w) for w in words))
        self.assertEqual(len(self.trie.get_all_words_with_prefix("cat")), 3)

    def test_deletion(self):
        """Test word deletion."""
        self.trie.insert("hello")
        self.trie.insert("hell")

        self.assertTrue(self.trie.delete("hello"))
        self.assertFalse(self.trie.search("hello"))
        self.assertTrue(self.trie.search("hell"))  # Should remain

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty string
        self.trie.insert("")
        self.assertTrue(self.trie.search(""))

        # Single character
        self.trie.insert("a")
        self.assertTrue(self.trie.search("a"))

        # Very long word
        long_word = "a" * 1000
        self.trie.insert(long_word)
        self.assertTrue(self.trie.search(long_word))

    def test_case_sensitivity(self):
        """Test case handling."""
        self.trie.insert("Hello")
        self.trie.insert("hello")
        self.assertTrue(self.trie.search("Hello"))
        self.assertTrue(self.trie.search("hello"))
        # Different words if case-sensitive

    def test_special_characters(self):
        """Test special characters."""
        words = ["hello-world", "test_case", "foo.bar"]
        for word in words:
            self.trie.insert(word)
        self.assertTrue(all(self.trie.search(w) for w in words))


# Performance tests
class TestTriePerformance(unittest.TestCase):
    """Performance benchmarks."""

    def test_large_dataset(self):
        """Test with large number of words."""
        import time

        trie = Trie()
        words = [f"word{i}" for i in range(10000)]

        # Benchmark insert
        start = time.time()
        for word in words:
            trie.insert(word)
        insert_time = time.time() - start

        # Benchmark search
        start = time.time()
        for word in words:
            trie.search(word)
        search_time = time.time() - start

        print(f"Insert 10k words: {insert_time:.3f}s")
        print(f"Search 10k words: {search_time:.3f}s")

        # Assert reasonable performance
        self.assertLess(insert_time, 1.0)  # Should be fast
        self.assertLess(search_time, 0.5)


if __name__ == '__main__':
    unittest.main()
```

---

## Advanced Code Examples

### 1. Trie with Frequency Tracking

**Use case:** Autocomplete with ranking by frequency

```python
class FrequencyTrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0  # How many times this word was inserted
        self.word = None


class FrequencyTrie:
    """Trie that tracks word frequencies for ranking."""

    def __init__(self):
        self.root = FrequencyTrieNode()

    def insert(self, word, frequency=1):
        """Insert word with frequency (or increment by frequency)."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = FrequencyTrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.frequency += frequency
        node.word = word

    def search(self, word):
        """Search returns (found, frequency) tuple."""
        node = self.root

        for char in word:
            if char not in node.children:
                return (False, 0)
            node = node.children[char]

        if node.is_end_of_word:
            return (True, node.frequency)
        return (False, 0)

    def top_k_with_prefix(self, prefix, k=10):
        """
        Get top K most frequent words with given prefix.

        Returns: List of (word, frequency) tuples, sorted by frequency (desc)
        """
        import heapq

        # Find prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all words with frequencies
        candidates = []

        def dfs(n):
            if n.is_end_of_word:
                candidates.append((n.word, n.frequency))
            for child in n.children.values():
                dfs(child)

        dfs(node)

        # Return top K by frequency
        return heapq.nlargest(k, candidates, key=lambda x: x[1])

    def increment_frequency(self, word):
        """Increment frequency when user selects this word."""
        self.insert(word, frequency=1)


# Example usage
freq_trie = FrequencyTrie()

# Simulate search history
searches = ["apple", "application", "apple", "apply", "apple", "appetite"]
for search in searches:
    freq_trie.insert(search)

print("Top 3 suggestions for 'app':")
for word, freq in freq_trie.top_k_with_prefix("app", k=3):
    print(f"  {word}: {freq} times")

# Output:
# apple: 3 times
# application: 1 times
# apply: 1 times
```

---

### 2. Trie with Wildcards (Advanced)

**Supports multiple wildcard types:**

```python
class WildcardTrie(Trie):
    """
    Trie supporting wildcards:
    - '.' matches any single character
    - '*' matches zero or more characters
    """

    def search_with_wildcards(self, pattern):
        """
        Search with wildcard support.

        Examples:
        - "a.c" matches "abc", "adc", but not "abbc"
        - "a*c" matches "ac", "abc", "abbc", etc.
        """
        results = []

        def dfs(node, pat_idx, current_word):
            if pat_idx == len(pattern):
                if node.is_end_of_word:
                    results.append(current_word)
                return

            char = pattern[pat_idx]

            if char == '.':
                # Match any single character
                for c, child in node.children.items():
                    dfs(child, pat_idx + 1, current_word + c)

            elif char == '*':
                # Match zero or more characters
                # Case 1: Match zero chars (skip *)
                dfs(node, pat_idx + 1, current_word)

                # Case 2: Match one or more chars
                for c, child in node.children.items():
                    dfs(child, pat_idx, current_word + c)  # Keep * active

            else:
                # Regular character
                if char in node.children:
                    dfs(node.children[char], pat_idx + 1, current_word + char)

        dfs(self.root, 0, "")
        return results


# Example
wc_trie = WildcardTrie()
words = ["cat", "car", "card", "cart", "dog", "dodge"]
for word in words:
    wc_trie.insert(word)

print("Matches for 'ca.':", wc_trie.search_with_wildcards("ca."))
# Output: ['cat', 'car']

print("Matches for 'ca*':", wc_trie.search_with_wildcards("ca*"))
# Output: ['cat', 'car', 'card', 'cart']

print("Matches for '.*g':", wc_trie.search_with_wildcards(".*g"))
# Output: ['dog']
```

---

### 3. Trie with Edit Distance (Fuzzy Search)

**Find words within edit distance k:**

```python
class FuzzyTrie(Trie):
    """Trie with fuzzy search (edit distance)."""

    def search_fuzzy(self, word, max_distance=2):
        """
        Find all words within max_distance edits of word.

        Uses dynamic programming during DFS.
        Returns: List of (word, distance) tuples
        """
        results = []

        def dfs(node, current_word, prev_row):
            """
            DFS with dynamic edit distance calculation.
            prev_row: DP array from previous level
            """
            cols = len(word) + 1
            current_row = [prev_row[0] + 1]  # First column (deletions)

            # Calculate edit distance for current character
            for col in range(1, cols):
                insert_cost = current_row[col - 1] + 1
                delete_cost = prev_row[col] + 1
                replace_cost = prev_row[col - 1]

                if word[col - 1] != current_word[-1]:
                    replace_cost += 1

                current_row.append(min(insert_cost, delete_cost, replace_cost))

            # If edit distance is within threshold and word is complete
            if current_row[-1] <= max_distance and node.is_end_of_word:
                results.append((current_word, current_row[-1]))

            # Only continue if there's potential
            if min(current_row) <= max_distance:
                for char, child in node.children.items():
                    dfs(child, current_word + char, current_row)

        # Initialize first row (distance from empty string)
        first_row = list(range(len(word) + 1))
        dfs(self.root, "", first_row)

        # Sort by distance, then alphabetically
        results.sort(key=lambda x: (x[1], x[0]))
        return results


# Example
fuzzy_trie = FuzzyTrie()
dictionary = ["hello", "hallo", "hillo", "yellow", "jello", "help"]
for word in dictionary:
    fuzzy_trie.insert(word)

print("Fuzzy search for 'hello' (distance ≤ 1):")
for word, distance in fuzzy_trie.search_fuzzy("hello", max_distance=1):
    print(f"  {word} (distance: {distance})")

# Output:
# hello (distance: 0)
# hallo (distance: 1)
# hillo (distance: 1)
# jello (distance: 1)
```

---

### 4. Complete Compressed Trie (Radix Tree)

Full production-ready implementation:

```python
class RadixNode:
    """Node in radix tree with full features."""

    def __init__(self, label=""):
        self.label = label
        self.children = {}
        self.is_end_of_word = False
        self.value = None
        self.count = 0  # Number of words in subtree


class RadixTree:
    """Full-featured radix tree (compressed trie)."""

    def __init__(self):
        self.root = RadixNode()
        self.size = 0

    def insert(self, word, value=None):
        """Insert with full compression."""
        if not word:
            return

        node = self.root
        i = 0

        while i < len(word):
            char = word[i]

            if char not in node.children:
                # No matching child - create new node
                new_node = RadixNode(word[i:])
                new_node.is_end_of_word = True
                new_node.value = value
                node.children[char] = new_node
                self._update_counts(node)
                self.size += 1
                return

            child = node.children[char]
            label = child.label

            # Find length of common prefix
            j = 0
            while (j < len(label) and i + j < len(word) and
                   label[j] == word[i + j]):
                j += 1

            if j == len(label):
                # Full label matches - continue deeper
                node = child
                i += j
            else:
                # Partial match - need to split node
                self._split_node(node, child, char, j, word[i:], value)
                self.size += 1
                return

        # Word fully consumed at existing node
        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.value = value
            self._update_counts(node)
            self.size += 1

    def _split_node(self, parent, child, first_char, split_pos,
                    remaining_word, value):
        """Split a node when partial match occurs."""
        label = child.label
        common_prefix = label[:split_pos]
        child_suffix = label[split_pos:]
        word_suffix = remaining_word[split_pos:]

        # Create intermediate node with common prefix
        intermediate = RadixNode(common_prefix)
        parent.children[first_char] = intermediate

        # Original child gets remaining label
        child.label = child_suffix
        intermediate.children[child_suffix[0]] = child

        if word_suffix:
            # Create new node for word's suffix
            new_node = RadixNode(word_suffix)
            new_node.is_end_of_word = True
            new_node.value = value
            intermediate.children[word_suffix[0]] = new_node
        else:
            # Intermediate node is end of word
            intermediate.is_end_of_word = True
            intermediate.value = value

        self._update_counts(intermediate)

    def _update_counts(self, node):
        """Update word count for node."""
        count = 1 if node.is_end_of_word else 0
        for child in node.children.values():
            count += child.count
        node.count = count

    def search(self, word):
        """Search for exact word."""
        node, remaining = self._find_node(word)
        return (node is not None and not remaining and
                node.is_end_of_word)

    def _find_node(self, word):
        """
        Find node and remaining unmatched portion.
        Returns: (node, remaining_word)
        """
        node = self.root
        i = 0

        while i < len(word):
            char = word[i]

            if char not in node.children:
                return (None, word[i:])

            child = node.children[char]
            label = child.label

            # Check if word matches label
            j = 0
            while j < len(label) and i + j < len(word):
                if label[j] != word[i + j]:
                    return (None, word[i:])
                j += 1

            if j < len(label):
                # Word ended mid-label
                return (child, "")  # Prefix match

            node = child
            i += j

        return (node, "")

    def get_all_words(self):
        """Get all words in radix tree."""
        results = []

        def dfs(node, prefix):
            current = prefix + node.label
            if node.is_end_of_word:
                results.append(current)
            for child in node.children.values():
                dfs(child, current)

        for child in self.root.children.values():
            dfs(child, "")

        return results

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"RadixTree(size={self.size})"


# Example
radix = RadixTree()
words = ["test", "testing", "tester", "team", "toast", "toaster"]

print("Inserting:", words)
for word in words:
    radix.insert(word)

print(f"\nRadix tree size: {len(radix)}")
print("All words:", radix.get_all_words())
print(f"Search 'test': {radix.search('test')}")
print(f"Search 'testing': {radix.search('testing')}")
print(f"Search 'tes': {radix.search('tes')}")
```

---

## Explain Like I'm 10

**Question:** "What's a trie and why do we need it?"

**Answer:**

Imagine you have a **giant dictionary** with millions of words, and you want to play a game where you type letters one by one, and the computer shows you all words that start with those letters (like when you search on Google).

### The Slow Way (List)

If you store all words in a list:
```
["cat", "car", "card", "dog", "dodge"]
```

When you type "ca", the computer has to check EVERY SINGLE WORD to see if it starts with "ca". With a million words, that's a million checks! Super slow! 😰

### The Smart Way (Trie - sounds like "try")

A **trie is like a word family tree**:

```
                  (start)
                  /    \
                 c      d
                 |      |
                 a      o
                / \     |\
               t   r    d g
                   |    | |
                   d    g e
```

Each letter is a stepping stone. When you type "ca", you just walk down:
- Start → c → a

Now you're at the "a" after "c", and you can see ALL the words below:
- Go down to "t" = "cat"
- Go down to "r" = "car"
  - Keep going to "d" = "card"

**Why it's awesome:**
1. **Fast!** You only walk through the letters you typed (like 2-3 steps), not a million words
2. **Saves space!** Words like "car" and "card" share the letters "c-a-r", so we store those letters only once
3. **Smart search!** It knows instantly there are no words starting with "zx" without checking every word

### Real Life Example

Think of a **library organized by topic**:

**Bad way:**
- All books thrown in one giant pile
- To find books about "cats", check every single book 😫

**Trie way:**
- First floor: Animals, Plants, Space
- Go to Animals floor
- Section: Mammals, Birds, Fish
- Go to Mammals section
- Shelf: Cats, Dogs, Horses
- Go to Cats shelf
- Found all cat books quickly! 🎉

Each level narrows down your search, just like each letter in a trie narrows down possible words!

### When You'd Use It

- **Google search** - showing suggestions as you type
- **Spell checker** - "Did you mean...?"
- **Phone contacts** - finding names as you type
- **Word games** - like Boggle or Scrabble word checking

It's basically a super-organized way to store words so finding them is lightning fast! ⚡

---

## Further Resources

### Practice Problems (LeetCode)

**Easy:**
- [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/) ⭐ Start here
- [720. Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/)

**Medium:**
- [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/) ⭐ Wildcards
- [648. Replace Words](https://leetcode.com/problems/replace-words/)
- [677. Map Sum Pairs](https://leetcode.com/problems/map-sum-pairs/)
- [421. Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) ⭐ Binary trie
- [820. Short Encoding of Words](https://leetcode.com/problems/short-encoding-of-words/)
- [1268. Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/) ⭐ Autocomplete

**Hard:**
- [212. Word Search II](https://leetcode.com/problems/word-search-ii/) ⭐⭐ Very important
- [336. Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/)
- [472. Concatenated Words](https://leetcode.com/problems/concatenated-words/)
- [642. Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/) 🔒 Premium
- [1032. Stream of Characters](https://leetcode.com/problems/stream-of-characters/)

### Other Practice Platforms

**HackerRank:**
- [Contacts](https://www.hackerrank.com/challenges/contacts)
- [No Prefix Set](https://www.hackerrank.com/challenges/no-prefix-set)

**Codeforces:**
- [Subset of Strings](https://codeforces.com/problemset/problem/923/A) (Trie application)

**CSES:**
- [String Matching](https://cses.fi/problemset/task/1753) (Can use trie)

### Interactive Visualizations

- [VisuAlgo - Trie](https://visualgo.net/en/trie) - Excellent step-by-step visualization
- [University of San Francisco - Trie](https://www.cs.usfca.edu/~galles/visualization/Trie.html) - Interactive animations
- [Trie Visualization](https://www.cs.usfca.edu/~galles/visualization/RadixTree.html) - Radix tree comparison

### Video Tutorials

- [William Fiset - Trie Data Structure](https://www.youtube.com/watch?v=AXjmTQ8LEoI) - Comprehensive overview
- [Back To Back SWE - Tries](https://www.youtube.com/watch?v=AXjmTQ8LEoI) - Interview-focused
- [Tushar Roy - Trie Implementation](https://www.youtube.com/watch?v=-urNrIAQnNo) - Code walkthrough

### Articles & Tutorials

- [GeeksforGeeks - Trie Data Structure](https://www.geeksforgeeks.org/trie-insert-and-search/)
- [TopCoder - Using Tries](https://www.topcoder.com/community/competitive-programming/tutorials/using-tries/)
- [Stanford CS166 - Tries and String Matching](http://web.stanford.edu/class/cs166/lectures/16/Small16.pdf) - Academic

### Books

- **"Introduction to Algorithms" (CLRS)** - Chapter on String Matching (Section 32.1)
- **"The Algorithm Design Manual" by Skiena** - Chapter 12.3 on Tries
- **"Algorithms" by Sedgewick and Wayne** - Section 5.2 on Tries
- **"Programming Pearls" by Jon Bentley** - Column 13 discusses tries

### Research Papers (Classic)

- **Morrison (1968)** - "PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric" (Original compressed trie)
- **Aho-Corasick Algorithm (1975)** - Multiple pattern matching with tries
- **Ukkonen (1995)** - "On-line construction of suffix trees" (Advanced suffix trie)

### Related Topics to Explore

1. **Suffix Trees** - More space-efficient than suffix tries (O(n) space)
2. **Aho-Corasick Algorithm** - Multiple pattern matching using trie
3. **Burrows-Wheeler Transform** - Text compression using suffix arrays
4. **Directed Acyclic Word Graphs (DAWG)** - Space-optimized trie variant
5. **Double-Array Trie** - Cache-friendly trie implementation
6. **Hat-Trie** - Hybrid hash table + trie for best of both worlds

### GitHub Repositories

- [trie-search](https://github.com/joshuatvernon/trie-search) - JavaScript implementation
- [pytrie](https://github.com/gsakkis/pytrie) - Python trie library
- [radix](https://github.com/armon/radix) - Go radix tree implementation

### Real-World Codebases Using Tries

- **Linux Kernel** - Radix trees for memory management
- **Nginx** - HTTP routing with radix trees
- **Redis** - Sorted sets use specialized tries
- **Lucene/Elasticsearch** - Term dictionary with FST (similar to tries)

---

## Conclusion

Tries are a **powerful and elegant data structure** for string manipulation, offering unique advantages for prefix-based operations that no other data structure can match. While they come with higher memory overhead, their predictable O(m) performance and natural support for autocomplete, spell checking, and pattern matching make them indispensable for many applications.

### Key Takeaways

1. **When tries shine:**
   - Autocomplete and search suggestions
   - Dictionary implementations with prefix queries
   - IP routing (longest prefix matching)
   - Spell checkers and word games
   - Any scenario with common prefixes and frequent prefix queries

2. **Critical implementation details:**
   - Always include `is_end_of_word` flag to distinguish words from prefixes
   - Choose children representation (array vs hash map) based on alphabet size
   - Consider variations (radix tree, TST) for space optimization
   - Handle edge cases: empty strings, case sensitivity, special characters

3. **Performance characteristics:**
   - Time: O(m) for basic operations (m = word length)
   - Space: O(ALPHABET_SIZE × n × m) worst case
   - Trade-off: More space for faster prefix operations
   - Cache-unfriendly but algorithmically efficient

4. **Interview preparation:**
   - Recognize trie problems by keywords: prefix, autocomplete, dictionary, words
   - Practice core problems: LC 208, 211, 212, 421
   - Master both recursive and iterative implementations
   - Be ready to discuss trade-offs vs hash tables and BSTs

5. **Production considerations:**
   - Thread safety: use appropriate locking strategies
   - Persistence: implement serialization for saving/loading
   - Scalability: consider sharding for very large datasets
   - Testing: comprehensive test suites including edge cases

### The Big Picture

Tries exemplify the classic computer science trade-off between **time and space**. By investing more memory in a structured hierarchy, we gain incredibly efficient prefix operations that would be prohibitively expensive with other data structures. This makes tries a perfect example of **paying upfront costs for long-term benefits**—a principle that extends far beyond data structures into software architecture and system design.

As you continue your journey in mastering data structures, remember that **understanding when NOT to use a trie is just as important as knowing when to use one**. For random-access string lookups without prefix operations, a hash table is simpler and more efficient. For small datasets, the complexity of a trie isn't justified. The mark of a skilled engineer is choosing the right tool for the job.

### Next Steps

1. **Practice:** Solve 10-15 trie problems on LeetCode, focusing on medium and hard difficulty
2. **Implement:** Build a complete trie library with all methods discussed
3. **Experiment:** Try different optimizations and measure performance
4. **Apply:** Use tries in a real project (autocomplete feature, text search, etc.)
5. **Explore:** Study related structures (suffix trees, Aho-Corasick, DAWG)

### Final Thoughts

Tries are more than just a data structure—they're a **fundamental concept in computer science** that appears in various forms across systems programming, databases, networking, and bioinformatics. Understanding tries deeply will not only help you ace interviews but also give you insights into how modern search engines, routers, and text processing tools work under the hood.

The journey from understanding to mastery involves:
- **Learning** the theory and implementation details ✓
- **Practicing** with real problems 🎯
- **Applying** in projects 🚀
- **Teaching** others to solidify understanding 🎓

Keep exploring, keep coding, and remember: every great software engineer started exactly where you are now. The trie structure may seem complex at first, but with practice, it becomes second nature.

**Happy coding!** 🌟

---

*"The best way to learn is to build."* — Keep implementing, keep improving.

---

**Document Statistics:**
- Total Sections: 18
- Code Examples: 50+
- Problems Covered: 20+
- Lines: ~2,400
- Reading Time: ~45-60 minutes
- Skill Level: Beginner to Advanced

**Last Updated:** 2025

---
