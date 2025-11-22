# Conflict-Free Replicated Data Types (CRDTs)

## Overview

**Conflict-Free Replicated Data Types (CRDTs)** are data structures designed for distributed systems that guarantee **strong eventual consistency** without requiring coordination or consensus protocols. CRDTs enable replicas to be updated independently and concurrently, while ensuring that all replicas converge to the same state when they have received the same set of updates.

CRDTs are the foundation for many modern distributed systems including collaborative editors (Google Docs, Figma), distributed databases (Riak, Redis), and synchronization systems (TomTom GPS, Apple Notes).

### Why CRDTs?

In distributed systems, achieving consistency traditionally requires:
- **Locks/Coordination**: Expensive and reduces availability
- **Consensus Protocols** (Paxos, Raft): Complex and have latency overhead
- **Conflict Resolution**: Manual or application-specific logic

CRDTs eliminate these requirements by ensuring that **all concurrent operations commute**, meaning they can be applied in any order and still produce the same result.

### Core Principles

1. **Strong Eventual Consistency**: If all replicas receive the same updates, they converge to the same state
2. **No Coordination**: Replicas can update independently without waiting for others
3. **Automatic Conflict Resolution**: Built into the data structure semantics
4. **Mathematically Provable**: Based on semilattice theory and order theory

## Key Concepts

### State-based CRDTs (CvRDT)

**Convergent Replicated Data Types** merge entire states using a join operation.

**Requirements:**
- States form a **semilattice** (join-semilattice)
- Merge operation is **commutative**, **associative**, and **idempotent**
- Updates increase state monotonically

**Characteristics:**
- Send entire state (or delta-state) during synchronization
- Simple to implement
- Higher bandwidth requirements
- Resilient to message loss and reordering

**Example**: Two replicas of a counter merge by taking the maximum value.

```
Replica A: {value: 5}
Replica B: {value: 3}
Merged:    {value: max(5, 3) = 5}
```

### Operation-based CRDTs (CmRDT)

**Commutative Replicated Data Types** transmit operations that commute.

**Requirements:**
- Operations are **commutative** when concurrent
- Delivery guarantees: **exactly-once**, **causal order** delivery
- Operations can be applied immediately

**Characteristics:**
- Send operations (smaller messages)
- Lower bandwidth
- Requires reliable causal broadcast
- More complex delivery infrastructure

**Example**: Two concurrent increment operations commute regardless of order.

```
State: 0
Op1: increment(+3)
Op2: increment(+2)
Result: 5 (regardless of order)
```

### Semilattice Structure

A **join-semilattice** is a partially ordered set with a least upper bound (LUB) for any two elements.

**Properties:**
- **Commutativity**: `a ⊔ b = b ⊔ a`
- **Associativity**: `(a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)`
- **Idempotence**: `a ⊔ a = a`

Where `⊔` represents the join operation (merge).

### Causality and Vector Clocks

Many CRDTs use **vector clocks** or **version vectors** to track causality between operations:

```
VectorClock = {replica_id → counter}
```

**Happens-before relation**: Event `a` happened before `b` if `VC(a) < VC(b)`
**Concurrent events**: `a` and `b` are concurrent if neither `VC(a) < VC(b)` nor `VC(b) < VC(a)`

## CRDT Types

### 1. G-Counter (Grow-only Counter)

A counter that only increments. Each replica maintains its own counter, and the global value is the sum of all replica counters.

**State**: `{replica_id → count}`

**Operations:**
- `increment(replica_id)`: Increase replica's counter
- `value()`: Sum all counters
- `merge(other)`: Take pairwise maximum

**Properties:**
- Monotonically increasing
- Eventually consistent
- Simple and efficient

### 2. PN-Counter (Positive-Negative Counter)

A counter supporting both increments and decrements using two G-Counters.

**State**: `{P: G-Counter, N: G-Counter}`

**Operations:**
- `increment(replica_id)`: Increment P counter
- `decrement(replica_id)`: Increment N counter
- `value()`: P.value() - N.value()
- `merge(other)`: Merge both P and N

### 3. G-Set (Grow-only Set)

A set that only supports additions (no removals).

**State**: `Set of elements`

**Operations:**
- `add(element)`: Add element to set
- `contains(element)`: Check membership
- `merge(other)`: Set union

**Merge**: `S1 ⊔ S2 = S1 ∪ S2`

### 4. 2P-Set (Two-Phase Set)

A set supporting both additions and removals, but elements can only be added once and removed once.

**State**: `{A: G-Set (added), R: G-Set (removed)}`

**Operations:**
- `add(element)`: Add to A set
- `remove(element)`: Add to R set (if in A)
- `contains(element)`: `element ∈ A ∧ element ∉ R`
- `merge(other)`: Merge both A and R

**Limitation**: Cannot re-add removed elements (tombstone problem).

### 5. LWW-Element-Set (Last-Write-Wins Element Set)

A set where each element is tagged with a timestamp, and the most recent operation wins.

**State**: `{A: {element → timestamp}, R: {element → timestamp}}`

**Operations:**
- `add(element, timestamp)`: Add element with timestamp
- `remove(element, timestamp)`: Mark removed with timestamp
- `contains(element)`: Element in A with timestamp > R timestamp (or not in R)
- `merge(other)`: Keep element with maximum timestamp for each element

**Bias**: Configurable (add-bias or remove-bias) for concurrent operations with same timestamp.

### 6. OR-Set (Observed-Remove Set)

A set where removes only affect elements that have been observed (adds always win over concurrent removes).

**State**: `{element → Set of unique_tags}`

**Operations:**
- `add(element)`: Add element with new unique tag
- `remove(element)`: Remove all currently observed tags for element
- `contains(element)`: Element has at least one tag
- `merge(other)`: Union of all tags for each element

**Semantics**: Add-wins (concurrent add beats remove).

### 7. LWW-Register (Last-Write-Wins Register)

A register (single value holder) where the value with the latest timestamp wins.

**State**: `{value, timestamp}`

**Operations:**
- `set(value, timestamp)`: Update if timestamp is greater
- `get()`: Return current value
- `merge(other)`: Keep value with maximum timestamp

### 8. MV-Register (Multi-Value Register)

A register that preserves all concurrent values (supports multiple simultaneous values).

**State**: `{value → vector_clock}`

**Operations:**
- `set(value, vector_clock)`: Add value with its vector clock
- `get()`: Return all concurrent values
- `merge(other)`: Keep values that are concurrent or maximal

## Time and Space Complexity

| CRDT Type | Add/Update | Remove | Query | Merge | Space |
|-----------|------------|---------|-------|-------|-------|
| G-Counter | $O(1)$ | N/A | $O(n)$ | $O(n)$ | $O(n)$ replicas |
| PN-Counter | $O(1)$ | $O(1)$ | $O(n)$ | $O(n)$ | $O(n)$ replicas |
| G-Set | $O(1)$ | N/A | $O(1)$ | $O(m)$ | $O(m)$ elements |
| 2P-Set | $O(1)$ | $O(1)$ | $O(1)$ | $O(m)$ | $O(m)$ (with tombstones) |
| LWW-Element-Set | $O(1)$ | $O(1)$ | $O(1)$ | $O(m)$ | $O(m)$ elements |
| OR-Set | $O(1)$ | $O(k)$ | $O(1)$ | $O(m \cdot k)$ | $O(m \cdot k)$ tags |
| LWW-Register | $O(1)$ | N/A | $O(1)$ | $O(1)$ | $O(1)$ |
| MV-Register | $O(1)$ | N/A | $O(k)$ | $O(k)$ | $O(k)$ concurrent values |

Where:
- `n` = number of replicas
- `m` = number of elements
- `k` = number of unique tags/concurrent values per element

## Implementation

### G-Counter (State-based)

```python
class GCounter:
    """
    Grow-only Counter (State-based CRDT)
    Supports increment and merge operations
    """
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.counters = {}  # {replica_id: count}

    def increment(self, amount=1):
        """Increment this replica's counter - $O(1)$"""
        if self.replica_id not in self.counters:
            self.counters[self.replica_id] = 0
        self.counters[self.replica_id] += amount

    def value(self):
        """Get total count across all replicas - $O(n)$"""
        return sum(self.counters.values())

    def merge(self, other):
        """Merge with another GCounter - $O(n)$"""
        result = GCounter(self.replica_id)
        all_replicas = set(self.counters.keys()) | set(other.counters.keys())

        for replica in all_replicas:
            # Take maximum for each replica (semilattice join)
            result.counters[replica] = max(
                self.counters.get(replica, 0),
                other.counters.get(replica, 0)
            )
        return result

    def __repr__(self):
        return f"GCounter(replica={self.replica_id}, counters={self.counters}, value={self.value()})"


# Usage Example
replica_a = GCounter("A")
replica_b = GCounter("B")

replica_a.increment(5)
replica_b.increment(3)

# Merge replicas
merged = replica_a.merge(replica_b)
print(merged.value())  # Output: 8
```

### PN-Counter (State-based)

```python
class PNCounter:
    """
    Positive-Negative Counter (State-based CRDT)
    Supports increment, decrement, and merge
    """
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.positive = GCounter(replica_id)  # Increments
        self.negative = GCounter(replica_id)  # Decrements

    def increment(self, amount=1):
        """Increment counter - $O(1)$"""
        self.positive.increment(amount)

    def decrement(self, amount=1):
        """Decrement counter - $O(1)$"""
        self.negative.increment(amount)

    def value(self):
        """Get net count - $O(n)$"""
        return self.positive.value() - self.negative.value()

    def merge(self, other):
        """Merge with another PNCounter - $O(n)$"""
        result = PNCounter(self.replica_id)
        result.positive = self.positive.merge(other.positive)
        result.negative = self.negative.merge(other.negative)
        return result

    def __repr__(self):
        return f"PNCounter(value={self.value()}, +{self.positive.value()}, -{self.negative.value()})"


# Usage Example
counter_a = PNCounter("A")
counter_b = PNCounter("B")

counter_a.increment(10)
counter_a.decrement(3)

counter_b.increment(5)
counter_b.decrement(2)

merged = counter_a.merge(counter_b)
print(merged.value())  # Output: (10 + 5) - (3 + 2) = 10
```

### OR-Set (Observed-Remove Set)

```python
import uuid

class ORSet:
    """
    Observed-Remove Set (State-based CRDT)
    Add-wins semantics: concurrent adds beat removes
    """
    def __init__(self):
        self.elements = {}  # {element: {unique_tag, ...}}

    def add(self, element):
        """Add element with unique tag - $O(1)$"""
        if element not in self.elements:
            self.elements[element] = set()
        # Generate unique tag for this add operation
        unique_tag = str(uuid.uuid4())
        self.elements[element].add(unique_tag)
        return unique_tag

    def remove(self, element):
        """Remove element by removing all observed tags - $O(k)$"""
        if element in self.elements:
            # Remove all currently observed tags
            self.elements[element].clear()

    def contains(self, element):
        """Check if element exists - $O(1)$"""
        return element in self.elements and len(self.elements[element]) > 0

    def merge(self, other):
        """Merge with another ORSet - $O(m \cdot k)$"""
        result = ORSet()
        all_elements = set(self.elements.keys()) | set(other.elements.keys())

        for element in all_elements:
            # Union of all tags for each element
            tags = self.elements.get(element, set()) | other.elements.get(element, set())
            if tags:
                result.elements[element] = tags

        return result

    def get_elements(self):
        """Get all elements in set - $O(m)$"""
        return {elem for elem, tags in self.elements.items() if tags}

    def __repr__(self):
        return f"ORSet({self.get_elements()})"


# Usage Example
set_a = ORSet()
set_b = ORSet()

# Concurrent operations
set_a.add("apple")
set_a.add("banana")

set_b.add("apple")  # Concurrent add
set_b.remove("banana")  # Remove without observing the add

# Merge
merged = set_a.merge(set_b)
print(merged.get_elements())  # {'apple', 'banana'} - add wins!
```

### LWW-Register (Last-Write-Wins Register)

```python
import time

class LWWRegister:
    """
    Last-Write-Wins Register (State-based CRDT)
    The value with the highest timestamp wins
    """
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.value = None
        self.timestamp = 0

    def set(self, value, timestamp=None):
        """Set value with timestamp - $O(1)$"""
        if timestamp is None:
            timestamp = time.time()

        # Only update if new timestamp is greater
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp

    def get(self):
        """Get current value - $O(1)$"""
        return self.value

    def merge(self, other):
        """Merge with another register - $O(1)$"""
        result = LWWRegister(self.replica_id)

        # Keep value with maximum timestamp
        if self.timestamp > other.timestamp:
            result.value = self.value
            result.timestamp = self.timestamp
        elif other.timestamp > self.timestamp:
            result.value = other.value
            result.timestamp = other.timestamp
        else:
            # Tie-break: use replica_id or bias (here: keep self)
            result.value = self.value
            result.timestamp = self.timestamp

        return result

    def __repr__(self):
        return f"LWWRegister(value={self.value}, ts={self.timestamp})"


# Usage Example
reg_a = LWWRegister("A")
reg_b = LWWRegister("B")

reg_a.set("first", timestamp=100)
reg_b.set("second", timestamp=200)

merged = reg_a.merge(reg_b)
print(merged.get())  # Output: "second" (higher timestamp)
```

### JavaScript Implementation: G-Counter

```javascript
class GCounter {
  /**
   * Grow-only Counter (State-based CRDT)
   * @param {string} replicaId - Unique identifier for this replica
   */
  constructor(replicaId) {
    this.replicaId = replicaId;
    this.counters = new Map(); // replica_id → count
  }

  /**
   * Increment this replica's counter - O(1)
   * @param {number} amount - Amount to increment
   */
  increment(amount = 1) {
    const current = this.counters.get(this.replicaId) || 0;
    this.counters.set(this.replicaId, current + amount);
  }

  /**
   * Get total count across all replicas - O(n)
   * @returns {number} Sum of all counters
   */
  value() {
    let sum = 0;
    for (const count of this.counters.values()) {
      sum += count;
    }
    return sum;
  }

  /**
   * Merge with another GCounter - O(n)
   * @param {GCounter} other - Another GCounter to merge with
   * @returns {GCounter} Merged result
   */
  merge(other) {
    const result = new GCounter(this.replicaId);

    // Get all replica IDs from both counters
    const allReplicas = new Set([
      ...this.counters.keys(),
      ...other.counters.keys()
    ]);

    // Take maximum for each replica
    for (const replica of allReplicas) {
      const thisCount = this.counters.get(replica) || 0;
      const otherCount = other.counters.get(replica) || 0;
      result.counters.set(replica, Math.max(thisCount, otherCount));
    }

    return result;
  }

  toString() {
    return `GCounter(replica=${this.replicaId}, value=${this.value()})`;
  }
}

// Usage Example
const replicaA = new GCounter('A');
const replicaB = new GCounter('B');

replicaA.increment(5);
replicaB.increment(3);

const merged = replicaA.merge(replicaB);
console.log(merged.value()); // Output: 8
```

### CRDT Text Editing (Simplified)

```python
class CharWithMetadata:
    """Character with position and causality information"""
    def __init__(self, char, position, replica_id, counter):
        self.char = char
        self.position = position  # Fractional index between neighbors
        self.replica_id = replica_id
        self.counter = counter  # Lamport timestamp
        self.deleted = False

    def __lt__(self, other):
        """Compare by position for ordering"""
        return self.position < other.position


class CRDTText:
    """
    Simplified CRDT for collaborative text editing
    Uses fractional indexing for position
    """
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.chars = []  # List of CharWithMetadata
        self.counter = 0

    def insert(self, char, index):
        """Insert character at index - $O(\log n + n)$"""
        self.counter += 1

        # Calculate position between neighbors
        if index == 0:
            pos = 0.0 if not self.chars else self.chars[0].position - 1.0
        elif index >= len(self.chars):
            pos = self.chars[-1].position + 1.0 if self.chars else 0.0
        else:
            # Position between index-1 and index
            prev_pos = self.chars[index - 1].position if index > 0 else 0.0
            next_pos = self.chars[index].position
            pos = (prev_pos + next_pos) / 2.0

        char_meta = CharWithMetadata(char, pos, self.replica_id, self.counter)
        self.chars.append(char_meta)
        self.chars.sort()  # Maintain order by position

    def delete(self, index):
        """Delete character at index (tombstone) - $O(1)$"""
        visible_chars = [c for c in self.chars if not c.deleted]
        if 0 <= index < len(visible_chars):
            visible_chars[index].deleted = True

    def merge(self, other):
        """Merge with another CRDT text - $O(m + n)$"""
        result = CRDTText(self.replica_id)
        result.counter = max(self.counter, other.counter)

        # Merge character lists
        merged_chars = {}
        for char in self.chars + other.chars:
            key = (char.position, char.replica_id, char.counter)
            if key not in merged_chars:
                merged_chars[key] = char
            else:
                # Keep the one marked as deleted if either is deleted
                if char.deleted or merged_chars[key].deleted:
                    merged_chars[key].deleted = True

        result.chars = sorted(merged_chars.values())
        return result

    def to_string(self):
        """Get visible text - $O(n)$"""
        return ''.join(c.char for c in self.chars if not c.deleted)

    def __repr__(self):
        return f"CRDTText('{self.to_string()}')"


# Usage Example: Collaborative editing
doc_a = CRDTText("Alice")
doc_b = CRDTText("Bob")

# Alice types "Hello"
for i, char in enumerate("Hello"):
    doc_a.insert(char, i)

# Bob concurrently types "World"
for i, char in enumerate("World"):
    doc_b.insert(char, i)

# Merge documents
merged = doc_a.merge(doc_b)
print(merged.to_string())  # Both "Hello" and "World" characters preserved
```

## Common Algorithms and Patterns

### Delta-State CRDTs

To reduce bandwidth, send only the **delta** (changes since last sync) instead of full state.

```python
class DeltaGCounter(GCounter):
    """G-Counter with delta-state optimization"""

    def __init__(self, replica_id):
        super().__init__(replica_id)
        self.delta = {}  # Track changes since last sync

    def increment(self, amount=1):
        """Increment and record delta - $O(1)$"""
        super().increment(amount)
        if self.replica_id not in self.delta:
            self.delta[self.replica_id] = 0
        self.delta[self.replica_id] += amount

    def get_delta(self):
        """Get changes since last sync - $O(k)$ where k = changed replicas"""
        delta_counter = GCounter(self.replica_id)
        delta_counter.counters = self.delta.copy()
        self.delta.clear()  # Reset delta
        return delta_counter

    def merge_delta(self, delta):
        """Merge delta state - $O(k)$"""
        return self.merge(delta)
```

### Garbage Collection for Tombstones

CRDTs with remove operations (2P-Set, OR-Set) accumulate tombstones. Garbage collection removes obsolete metadata.

```python
class ORSetWithGC(ORSet):
    """OR-Set with garbage collection support"""

    def garbage_collect(self, cutoff_timestamp):
        """
        Remove tombstones older than cutoff - $O(m)$
        Requires causal stability: all replicas have seen operations before cutoff
        """
        for element in list(self.elements.keys()):
            if not self.elements[element]:  # Empty tag set
                del self.elements[element]
```

### Vector Clock Implementation

```python
class VectorClock:
    """Vector clock for tracking causality"""

    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.clock = {}  # {replica_id: counter}

    def increment(self):
        """Increment local counter - $O(1)$"""
        if self.replica_id not in self.clock:
            self.clock[self.replica_id] = 0
        self.clock[self.replica_id] += 1

    def update(self, other):
        """Update clock with received clock - $O(n)$"""
        for replica, count in other.clock.items():
            self.clock[replica] = max(self.clock.get(replica, 0), count)
        self.increment()

    def happens_before(self, other):
        """Check if this clock happens before other - $O(n)$"""
        # self < other iff: self[i] <= other[i] for all i, and exists j: self[j] < other[j]
        all_replicas = set(self.clock.keys()) | set(other.clock.keys())

        less_than_or_equal = True
        strictly_less = False

        for replica in all_replicas:
            self_count = self.clock.get(replica, 0)
            other_count = other.clock.get(replica, 0)

            if self_count > other_count:
                less_than_or_equal = False
                break
            if self_count < other_count:
                strictly_less = True

        return less_than_or_equal and strictly_less

    def concurrent(self, other):
        """Check if clocks are concurrent - $O(n)$"""
        return not self.happens_before(other) and not other.happens_before(self)
```

### Causal Broadcast for Op-based CRDTs

```python
from collections import deque

class CausalBroadcast:
    """
    Ensures causal delivery order for operation-based CRDTs
    Uses vector clocks
    """
    def __init__(self, replica_id, num_replicas):
        self.replica_id = replica_id
        self.vector_clock = VectorClock(replica_id)
        self.buffer = deque()  # Operations waiting for causal dependencies

    def send_operation(self, operation):
        """Send operation with vector clock - $O(1)$"""
        self.vector_clock.increment()
        return {
            'op': operation,
            'timestamp': self.vector_clock.clock.copy(),
            'sender': self.replica_id
        }

    def receive_operation(self, message):
        """
        Receive and buffer operation until causally ready - $O(n)$
        Returns operations ready to apply
        """
        self.buffer.append(message)
        ready_ops = []

        # Try to deliver buffered operations in causal order
        i = 0
        while i < len(self.buffer):
            msg = self.buffer[i]
            sender = msg['sender']
            timestamp = msg['timestamp']

            # Check if causally ready
            can_deliver = True
            for replica, count in timestamp.items():
                local_count = self.vector_clock.clock.get(replica, 0)

                if replica == sender:
                    # Must be exactly next operation from sender
                    if count != local_count + 1:
                        can_deliver = False
                        break
                else:
                    # Must have seen all prior operations from other replicas
                    if count > local_count:
                        can_deliver = False
                        break

            if can_deliver:
                # Deliver operation
                self.vector_clock.update(VectorClock(sender))
                self.vector_clock.clock[sender] = timestamp[sender]
                ready_ops.append(msg['op'])
                self.buffer.remove(msg)
            else:
                i += 1

        return ready_ops
```

## Practical Applications

### 1. Collaborative Text Editing

**Systems**: Google Docs, Figma, Atom Teletype, Apple Notes

**CRDT Used**:
- **WOOT** (Without Operational Transformation)
- **RGA** (Replicated Growable Array)
- **Logoot**: Fractional indexing with unique identifiers
- **Yjs**: High-performance CRDT framework

**Example**: Two users editing the same document concurrently:
```
User A types: "Hello"
User B types: "World" at the same position
Result after merge: "HelloWorld" or "WorldHello" (deterministic based on position/timestamp)
```

### 2. Distributed Databases

**Systems**:
- **Riak**: Uses CRDTs for counters, sets, maps, and registers
- **Redis Enterprise**: CRDT-based active-active geo-distribution
- **AntidoteDB**: Highly available transactional database with CRDTs

**Use Case**: Shopping cart that works offline
```python
# Shopping cart as OR-Set CRDT
cart = ORSet()

# User adds items while offline
cart.add("laptop")
cart.add("mouse")

# Later removes item
cart.remove("mouse")

# Syncs with server - changes merge automatically
```

### 3. Distributed Caching

**Pattern**: Multi-master cache with LWW-Register or MV-Register

```python
# Cache value across data centers
cache_us = LWWRegister("US")
cache_eu = LWWRegister("EU")

# Update in US
cache_us.set("user_123_profile", {"name": "Alice", "age": 30}, timestamp=1000)

# Concurrent update in EU
cache_eu.set("user_123_profile", {"name": "Alice", "age": 31}, timestamp=1001)

# Merge - EU value wins (higher timestamp)
merged = cache_us.merge(cache_eu)
```

### 4. Real-time Analytics

**Pattern**: PN-Counter for metrics aggregation

```python
# Page view counter across edge servers
views_edge1 = PNCounter("edge1")
views_edge2 = PNCounter("edge2")

# Count views independently
views_edge1.increment(1523)
views_edge2.increment(2841)

# Aggregate globally
total_views = views_edge1.merge(views_edge2)
print(f"Total views: {total_views.value()}")  # 4364
```

### 5. Offline-First Mobile Apps

**Pattern**: Local-first software with CRDTs

**Example**: Todo list app
```python
class TodoList:
    """Todo list using OR-Set CRDT"""
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.items = ORSet()  # Todo items
        self.completed = ORSet()  # Completed items

    def add_todo(self, item):
        self.items.add(item)

    def complete_todo(self, item):
        if self.items.contains(item):
            self.completed.add(item)

    def remove_todo(self, item):
        self.items.remove(item)
        self.completed.remove(item)

    def sync(self, other_list):
        """Sync with another device"""
        self.items = self.items.merge(other_list.items)
        self.completed = self.completed.merge(other_list.completed)
```

### 6. Configuration Management

**Pattern**: Multi-value register for feature flags

```python
class FeatureFlags:
    """Feature flags using MV-Register"""
    def __init__(self):
        self.flags = {}  # {flag_name: MV-Register}

    def set_flag(self, name, enabled, replica_id):
        if name not in self.flags:
            self.flags[name] = MVRegister()
        self.flags[name].set(enabled, VectorClock(replica_id))

    def is_enabled(self, name):
        """Return True if any replica has it enabled"""
        if name not in self.flags:
            return False
        values = self.flags[name].get()
        return any(values)  # Optimistic: enable if any says yes
```

## Trade-offs and Best Practices

### When to Use CRDTs

✅ **Good Use Cases:**
- High availability required (offline-first, geo-distributed)
- Concurrent updates are common
- Low-latency local operations needed
- Network partitions expected
- Eventual consistency is acceptable

❌ **Not Ideal When:**
- Strong consistency required (banking transactions)
- Total ordering of operations critical
- Storage constraints are severe (CRDTs can have metadata overhead)
- Complex invariants must be maintained (e.g., uniqueness constraints)

### Performance Considerations

1. **Metadata Overhead**
   - OR-Set: Each element carries unique tags
   - Vector Clocks: Grows with number of replicas
   - **Solution**: Use delta-state CRDTs, garbage collection

2. **Merge Complexity**
   - Full-state merge can be expensive for large CRDTs
   - **Solution**: Delta-state synchronization, incremental merge

3. **Tombstone Accumulation**
   - Removed elements leave tombstones
   - **Solution**: Periodic garbage collection with causal stability

4. **Memory Usage**
   - CRDTs retain metadata for conflict resolution
   - **Solution**: Use compact representations, prune old data

### Best Practices

#### 1. Choose the Right CRDT

```
Need counter? → G-Counter (increment only) or PN-Counter (inc/dec)
Need set with add/remove? → OR-Set (add-wins) or LWW-Set (timestamp-based)
Need single value? → LWW-Register or MV-Register
Need ordered collection? → RGA or Logoot (specialized text CRDTs)
```

#### 2. Handle Semantic Conflicts

CRDTs prevent *technical* conflicts but not *semantic* ones:

```python
# Bad: Balance can go negative
account = PNCounter("account_123")
account.increment(100)  # Deposit $100
account.decrement(150)  # Withdraw $150 (allowed but semantically wrong!)

# Good: Add application-level validation
class BankAccount:
    def __init__(self, account_id):
        self.balance = PNCounter(account_id)
        self.pending_withdrawals = ORSet()

    def withdraw(self, amount):
        if self.balance.value() >= amount:
            self.balance.decrement(amount)
            return True
        else:
            # Queue for later or reject
            return False
```

#### 3. Implement Garbage Collection

```python
class ManagedORSet(ORSet):
    """OR-Set with automatic GC"""
    def __init__(self, max_tombstones=10000):
        super().__init__()
        self.max_tombstones = max_tombstones
        self.tombstone_count = 0

    def remove(self, element):
        super().remove(element)
        self.tombstone_count += 1

        if self.tombstone_count > self.max_tombstones:
            self._gc()

    def _gc(self):
        # Remove elements with no tags
        self.elements = {k: v for k, v in self.elements.items() if v}
        self.tombstone_count = 0
```

#### 4. Use Delta-State for Efficiency

```python
# Bad: Send full state every sync (expensive for large CRDTs)
def sync_full_state(crdt):
    full_state = crdt.to_dict()  # Potentially huge
    send_to_replicas(full_state)

# Good: Send only changes
def sync_delta_state(delta_crdt):
    delta = delta_crdt.get_delta()  # Only recent changes
    send_to_replicas(delta)
```

#### 5. Timestamp Hygiene

```python
# Bad: System clocks can drift or go backwards
timestamp = time.time()

# Good: Use logical clocks or hybrid logical clocks
class HybridLogicalClock:
    """Combines physical and logical time"""
    def __init__(self):
        self.wall_clock = 0
        self.logical = 0

    def tick(self):
        new_wall = time.time_ns()
        if new_wall > self.wall_clock:
            self.wall_clock = new_wall
            self.logical = 0
        else:
            self.logical += 1
        return (self.wall_clock, self.logical)
```

#### 6. Testing for Convergence

```python
def test_convergence():
    """Test that replicas converge after concurrent operations"""
    replica_a = ORSet()
    replica_b = ORSet()
    replica_c = ORSet()

    # Concurrent operations
    replica_a.add("x")
    replica_b.add("y")
    replica_c.add("z")
    replica_a.remove("y")

    # Merge in different orders
    result1 = replica_a.merge(replica_b).merge(replica_c)
    result2 = replica_c.merge(replica_a).merge(replica_b)
    result3 = replica_b.merge(replica_c).merge(replica_a)

    # All should converge to same state
    assert result1.get_elements() == result2.get_elements() == result3.get_elements()
```

### Common Pitfalls

❌ **Don't rely on operation order**
```python
# Bad: Assuming operations apply in sent order
crdt.add("a")
crdt.add("b")
# Don't assume "a" appears before "b" on other replicas
```

✅ **Use CRDTs designed for ordering**
```python
# Good: Use RGA or other sequence CRDTs for ordered data
sequence = RGA()
sequence.insert("a", position=0)
sequence.insert("b", position=1)
```

❌ **Don't use LWW for critical data**
```python
# Bad: LWW can lose concurrent updates
register.set("Alice's edit", timestamp=100)
register.set("Bob's edit", timestamp=100)  # One edit lost!
```

✅ **Use MV-Register or application-level merge**
```python
# Good: Preserve concurrent values
mv_register = MVRegister()
mv_register.set("Alice's edit", vc_alice)
mv_register.set("Bob's edit", vc_bob)
all_values = mv_register.get()  # Both preserved, app decides how to merge
```

## Advanced Topics

### Combining CRDTs

CRDTs can be composed to create more complex data structures:

```python
class CRDTMap:
    """Map CRDT: keys → CRDT values"""
    def __init__(self):
        self.keys = ORSet()  # Keys as OR-Set
        self.values = {}  # {key: CRDT_value}

    def set(self, key, value_crdt):
        """Set key to a CRDT value"""
        self.keys.add(key)
        if key not in self.values:
            self.values[key] = value_crdt
        else:
            self.values[key] = self.values[key].merge(value_crdt)

    def remove(self, key):
        """Remove key"""
        self.keys.remove(key)

    def merge(self, other):
        """Merge maps"""
        result = CRDTMap()
        result.keys = self.keys.merge(other.keys)

        all_keys = result.keys.get_elements()
        for key in all_keys:
            if key in self.values and key in other.values:
                result.values[key] = self.values[key].merge(other.values[key])
            elif key in self.values:
                result.values[key] = self.values[key]
            else:
                result.values[key] = other.values[key]

        return result
```

### Pure Operation-based CRDT Example

```python
class OpBasedGCounter:
    """
    Operation-based G-Counter
    Requires reliable causal broadcast
    """
    def __init__(self, replica_id, broadcast_fn):
        self.replica_id = replica_id
        self.broadcast = broadcast_fn
        self.count = 0

    def increment(self, amount=1):
        """Broadcast increment operation - $O(1)$"""
        operation = {'type': 'increment', 'amount': amount, 'replica': self.replica_id}
        self.broadcast(operation)
        self.apply_increment(amount)

    def apply_increment(self, amount):
        """Apply increment operation - $O(1)$"""
        self.count += amount

    def value(self):
        """Get current value - $O(1)$"""
        return self.count
```

### Strong Eventual Consistency Proof (Intuition)

For a state-based CRDT to guarantee **Strong Eventual Consistency (SEC)**:

1. **Eventual Delivery**: All updates are eventually delivered to all replicas
2. **Convergence**: Replicas that have received the same updates have equivalent state

**Proof sketch**:
- State space forms a join-semilattice with merge operation `⊔`
- Updates move state monotonically upward: `s ⊑ s'` after update
- Merge is commutative, associative, idempotent
- For any set of updates `U`, all permutations of merging reach the same state (LUB)

## ELI10

Imagine you and your friend both have a **shared notebook** where you write down your favorite movies. But there's a twist: you each have your own copy of the notebook, and sometimes you can't see what the other person wrote right away (like when you're not connected to the internet).

### The Problem

If you both write "Toy Story" at the same time in your separate notebooks, how do you make sure both notebooks end up with the same list when you finally compare them? What if one of you deletes a movie while the other adds it?

### The CRDT Solution

**CRDTs are like magic rules** for your notebooks that guarantee you'll **always end up with the same list**, no matter what order you compare your changes!

#### Example 1: The Counter

Imagine counting how many times you've watched your favorite movie:
- You're at home: click, click, click (3 times)
- Your friend at school: click, click (2 times)
- When you meet up, you add your counts: 3 + 2 = **5 times total**

This works because adding numbers doesn't care about order: 3+2 = 2+3!

#### Example 2: The Set

You both write movie names:
- You write: "Frozen", "Moana"
- Friend writes: "Frozen", "Encanto"
- When you merge: you keep all unique movies: **"Frozen", "Moana", "Encanto"**

Even if you both wrote "Frozen", it only appears once (no duplicates).

#### Example 3: The Timestamp Trick

What if you both change the same thing?
- You write: "Best movie: Frozen" at 3:00 PM
- Friend writes: "Best movie: Moana" at 3:15 PM
- Rule: **The most recent one wins** → "Best movie: Moana"

### Why This is Cool

1. **You can work offline**: Write in your notebook anytime, sync later
2. **No arguments**: The rules automatically decide what to keep
3. **Nothing gets lost**: If you both add different things, both are kept
4. **Always the same result**: Doesn't matter who syncs first

### Real-World Magic

- **Google Docs**: You and your classmate edit the same document at the same time
- **Multiplayer games**: Everyone sees the same game state eventually
- **Your phone's contacts**: Changes sync across your phone, tablet, and computer

CRDTs are the secret sauce that makes all these work without everyone having to wait for each other! 🎉

## Further Resources

### Academic Papers

1. **Shapiro et al. (2011)** - "A comprehensive study of Convergent and Commutative Replicated Data Types"
   - Original CRDT paper defining state-based and operation-based CRDTs

2. **Shapiro et al. (2011)** - "Conflict-free Replicated Data Types" (Technical Report)
   - Formal specifications and proofs for major CRDT types

3. **Almeida et al. (2018)** - "Delta State Replicated Data Types"
   - Optimization for reducing synchronization bandwidth

4. **Roh et al. (2011)** - "Replicated abstract data types: Building blocks for collaborative applications"
   - RGA (Replicated Growable Array) for sequences

5. **Kleppmann & Beresford (2017)** - "A Conflict-Free Replicated JSON Datatype"
   - JSON CRDTs for structured data

### Libraries and Frameworks

**JavaScript/TypeScript:**
- [Yjs](https://github.com/yjs/yjs) - High-performance CRDT framework
- [Automerge](https://github.com/automerge/automerge) - JSON-like CRDT
- [ShareDB](https://github.com/share/sharedb) - Real-time database with CRDTs

**Python:**
- [pycrdt](https://github.com/jupyter-server/pycrdt) - Python bindings for Yjs
- [concordant](https://github.com/coast-team/concordant) - CRDT library

**Rust:**
- [automerge-rs](https://github.com/automerge/automerge-rs) - Rust implementation of Automerge
- [yrs](https://github.com/y-crdt/y-crdt) - Rust port of Yjs

**Go:**
- [go-crdt](https://github.com/neurodyne/go-crdt) - CRDT implementations in Go

### Databases with CRDT Support

- **Riak** - Distributed database with built-in CRDTs
- **Redis Enterprise** - Active-active geo-distribution with CRDTs
- **AntidoteDB** - Transactional database with CRDT support
- **OrbitDB** - Peer-to-peer database using CRDTs
- **Datomic** - Immutable database with CRDT-like properties

### Articles and Tutorials

- [CRDT.tech](https://crdt.tech/) - Community hub for CRDT resources
- [Local-First Software](https://www.inkandswitch.com/local-first/) - Ink & Switch research on CRDTs
- [Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/) - Chapter on replication and CRDTs

### Related Topics

- [Vector Clocks](./vector_clocks.md) - Causality tracking
- [Distributed Systems](../systems/distributed_systems.md) - Context for CRDT usage
- [Hash Tables](./hash_tables.md) - Often used in CRDT implementations
- [Merkle Trees](./trees.md) - Efficient state synchronization

### Video Resources

- [CRDTs and the Quest for Distributed Consistency](https://www.youtube.com/watch?v=B5NULPSiOGw) - Martin Kleppmann
- [Conflict Resolution for Eventual Consistency](https://www.youtube.com/watch?v=yCcWpzY8dIA) - Marc Shapiro

---

**Last Updated**: 2025-01-22
**Related**: [Distributed Systems](../systems/distributed_systems.md), [Consensus Algorithms](../algorithms/consensus.md), [Vector Clocks](./vector_clocks.md)
