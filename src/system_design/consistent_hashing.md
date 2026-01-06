# Consistent Hashing

> **Domain:** Distributed Systems, Caching, Databases
> **Key Concepts:** Hash Ring, Virtual Nodes, Partitioning, Replication

**Consistent Hashing** is a distributed hashing scheme that operates independently of the number of servers or objects in a distributed hash table. It powers systems like Amazon Dynamo, Apache Cassandra, and Discord's ring.

---

## 1. The Problem: Modulo Hashing

The naive way to distribute keys across $N$ servers is:
`server_index = hash(key) % N`

*   **Scenario:** You have 100 cache servers ($N=100$).
*   **Event:** Server 50 crashes ($N=99$).
*   **Result:** `hash(key) % 99` produces a completely different result than `hash(key) % 100`.
*   **Catastrophe:** **Almost 100% of keys are remapped.** The cache is flushed, the database is hammered, and the system goes down.

---

## 2. The Solution: The Hash Ring

Consistent hashing maps both **Servers** and **Keys** to the same circular keyspace (e.g., $0$ to $2^{32}-1$).

### 2.1. The Mechanism
1.  **Placement:** Hash every Server IP to a point on the circle.
2.  **Lookup:** Hash the Key to a point on the circle.
3.  **Routing:** Walk **clockwise** from the Key's position until you find a Server. That server owns the key.

### 2.2. Adding/Removing Nodes
*   **Add Node:** When Server X joins, it places itself on the ring. It "steals" only the keys that fall between it and its counter-clockwise neighbor.
    *   *Impact:* Only $1/N$ keys are remapped.
*   **Remove Node:** When Server Y leaves, its keys are picked up by its clockwise neighbor.
    *   *Impact:* Only $1/N$ keys are moved.

---

## 3. The Refinement: Virtual Nodes (VNodes)

The basic ring has a flaw: **Data Skew**.
If Server A is at 12 o'clock and Server B is at 1 o'clock, Server A might own 90% of the ring (from 1:01 to 12:00).

*   **Solution:** Virtual Nodes.
*   **Concept:** Server A doesn't appear once. It appears 1000 times (Virtual Nodes) at random positions on the ring.
*   **Benefit:**
    1.  **Uniform Distribution:** The load is statistically likely to be even.
    2.  **Heterogeneity:** A powerful server can have 2000 VNodes, while a weak one has 500.

---

## 4. Replication

To ensure durability, we don't just store data on the *first* server we find.
*   **Strategy:** Walk clockwise and store data on the first $R$ distinct physical servers (Replication Factor, usually 3).
*   **Preference Lists:** The list of nodes responsible for a key.

---

## 5. Implementation (Python Concept)

```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas
        self.ring = dict()
        self.sorted_keys = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def get_node(self, key_string):
        if not self.ring:
            return None
        
        hash_val = self._hash(key_string)
        
        # Find the first node to the right (clockwise)
        idx = bisect.bisect(self.sorted_keys, hash_val)
        
        # Wrap around to 0 if we hit the end
        if idx == len(self.sorted_keys):
            idx = 0
            
        return self.ring[self.sorted_keys[idx]]
```
