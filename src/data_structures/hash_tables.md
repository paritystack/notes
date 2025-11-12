# Hash Tables

## Overview

A hash table (hash map) stores key-value pairs with O(1) average-case lookup, insertion, and deletion. It uses a hash function to map keys to array indices.

## How It Works

### Hash Function
Converts key to index:
```
hash("name") = 5
hash(123) = 2
hash("email") = 5  (collision!)
```

### Collision Resolution

**Chaining**: Store multiple values at same index
```
Index 0: None
Index 1: None
Index 2: 123 ’ "John"
Index 3: 456 ’ "Jane" ’ 789 ’ "Jack"
```

**Open Addressing**: Find next empty slot
```
hash("a") = 5 (occupied)
Try 6, 7, 8... until empty
```

## Operations

| Operation | Average | Worst |
|-----------|---------|-------|
| **Get** | O(1) | O(n) |
| **Set** | O(1) | O(n) |
| **Delete** | O(1) | O(n) |

## Python Implementation

```python
# Built-in dict
d = {"key": "value"}
d.get("key")  # O(1)
d["key"] = "new_value"
del d["key"]

# Custom
class HashTable:
    def __init__(self, size=10):
        self.table = [[] for _ in range(size)]

    def set(self, key, value):
        index = hash(key) % len(self.table)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))

    def get(self, key):
        index = hash(key) % len(self.table)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None
```

## Common Problems

### Two Sum
```python
def two_sum(arr, target):
    seen = {}
    for num in arr:
        if target - num in seen:
            return [seen[target - num], arr.index(num)]
        seen[num] = arr.index(num)
    return []
```

### Duplicate Detection
```python
def has_duplicates(arr):
    return len(arr) != len(set(arr))
```

## ELI10

Think of hash tables like library catalogs:
- **Hash function** = catalog system (tells you which shelf)
- **Index** = shelf number
- **Value** = book

Instead of searching every shelf, the system instantly tells you which one!

## Further Resources

- [LeetCode Hash Table](https://leetcode.com/tag/hash-table/)
- [Hash Table Wikipedia](https://en.wikipedia.org/wiki/Hash_table)
