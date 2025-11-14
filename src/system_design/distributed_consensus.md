# Distributed Consensus

## Overview

Distributed consensus is the challenge of getting multiple nodes in a distributed system to agree on a shared state, even in the presence of failures.

## The CAP Theorem

You can only guarantee 2 out of 3:

```
       Consistency
          /  \
         /    \
        /  ??  \
       /________\
Availability   Partition
              Tolerance
```

### Consistency (C)

All nodes see the same data at the same time:

```
Write X=5 to Node A
  ↓
Node A, B, C all show X=5 immediately
```

### Availability (A)

Every request gets a response (success or failure):

```
Request → Node → Response (always)
(even if stale data)
```

### Partition Tolerance (P)

System continues operating despite network failures:

```
Network Split:
  [Node A, B]  |  [Node C, D]
     ↑                  ↑
  Can't communicate but both keep running
```

## CAP Trade-offs

### CP (Consistency + Partition Tolerance)

Sacrifice availability for consistency:

```
Network partition detected
  ↓
System becomes unavailable (refuse requests)
  ↓
Maintain consistency (no stale reads)
```

**Examples**: HBase, MongoDB (with majority writes), Redis (with wait)

**Use Case**: Financial systems, inventory management

### AP (Availability + Partition Tolerance)

Sacrifice consistency for availability:

```
Network partition detected
  ↓
Both partitions keep serving requests
  ↓
Data diverges (eventually consistent)
```

**Examples**: Cassandra, DynamoDB, Riak

**Use Case**: Social media, shopping carts, session storage

### CA (Consistency + Availability)

Only works without network partitions:

```
Single datacenter, reliable network
  ↓
All nodes always agree
  ↓
(Not realistic for distributed systems)
```

**Examples**: PostgreSQL (single instance), MySQL (single instance)

**Reality**: Network partitions always happen, must choose CP or AP

## Consistency Models

### Strong Consistency

Reads always return latest write:

```
Time 0: Write X=5 to Node A
Time 1: Read from Node B → Returns X=5 (latest)
Time 2: Write X=10 to Node A
Time 3: Read from Node B → Returns X=10 (latest)
```

**Implementation**: Wait for all replicas to acknowledge

```python
def write(key, value):
    # Write to all nodes
    for node in nodes:
        node.write(key, value)
        wait_for_ack(node)  # Block until confirmed

    return success

def read(key):
    # Read from majority
    values = []
    for node in majority_nodes:
        values.append(node.read(key))

    # Return latest version
    return max(values, key=lambda v: v.timestamp)
```

### Eventual Consistency

Reads may return stale data, but eventually converge:

```
Time 0: Write X=5 to Node A
Time 1: Read from Node B → Returns X=1 (stale, not replicated yet)
Time 2: Replication completes
Time 3: Read from Node B → Returns X=5 (consistent)
```

**Implementation**: Asynchronous replication

```python
def write(key, value):
    # Write to primary
    primary.write(key, value)

    # Async replicate to others
    async_replicate(key, value)

    return success  # Don't wait

def async_replicate(key, value):
    for node in replica_nodes:
        background_task.queue({
            'action': 'replicate',
            'node': node,
            'key': key,
            'value': value
        })
```

### Causal Consistency

Causally related writes are seen in order:

```
User A writes Post → User B reads Post → User B writes Comment
  ↓                      ↓                       ↓
All nodes see Post before Comment
```

```python
# Vector clocks for causality
class VectorClock:
    def __init__(self):
        self.clocks = {}  # {node_id: counter}

    def increment(self, node_id):
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other):
        for node_id, counter in other.clocks.items():
            self.clocks[node_id] = max(
                self.clocks.get(node_id, 0),
                counter
            )

    def happens_before(self, other):
        # True if self happened before other
        return all(
            self.clocks.get(k, 0) <= other.clocks.get(k, 0)
            for k in set(self.clocks) | set(other.clocks)
        )
```

### Read-After-Write Consistency

Your writes are immediately visible to you:

```
User writes X=5
  ↓
Same user reads → Gets X=5 (consistent)
  ↓
Other users read → May get old value (eventual)
```

```python
def write(key, value, user_id):
    version = primary.write(key, value)

    # Cache user's write version
    user_versions[user_id][key] = version

    return success

def read(key, user_id):
    # Check if user has written this key
    if key in user_versions.get(user_id, {}):
        # Read from primary to get latest
        return primary.read(key)
    else:
        # Can read from any replica
        return random.choice(replicas).read(key)
```

### Monotonic Reads

Once you read a value, you never read older values:

```
Time 1: Read X=5
Time 2: Read X=10 (newer) ✓
Time 3: Read X=5 (older) ✗ Not allowed
```

```python
# Sticky sessions - always read from same replica
def read(key, session_id):
    replica = get_sticky_replica(session_id)
    return replica.read(key)

def get_sticky_replica(session_id):
    # Hash session to consistent replica
    replica_id = hash(session_id) % num_replicas
    return replicas[replica_id]
```

## Consensus Algorithms

### Two-Phase Commit (2PC)

Distributed transaction protocol:

```
Phase 1: PREPARE
Coordinator → All Participants: "Can you commit?"
Participants → Coordinator: "Yes" or "No"

Phase 2: COMMIT
If all "Yes":
  Coordinator → All: "Commit!"
  Participants commit
Else:
  Coordinator → All: "Abort!"
  Participants rollback
```

```python
class TwoPhaseCommit:
    def __init__(self, coordinator, participants):
        self.coordinator = coordinator
        self.participants = participants

    def execute_transaction(self, transaction):
        # Phase 1: Prepare
        votes = []
        for participant in self.participants:
            vote = participant.prepare(transaction)
            votes.append(vote)

        # Phase 2: Commit or Abort
        if all(votes):
            # All voted yes - commit
            for participant in self.participants:
                participant.commit(transaction)
            return "committed"
        else:
            # Someone voted no - abort
            for participant in self.participants:
                participant.abort(transaction)
            return "aborted"
```

**Pros**: Strong consistency
**Cons**: Blocking (if coordinator fails), not partition-tolerant

### Three-Phase Commit (3PC)

Non-blocking version of 2PC:

```
Phase 1: PREPARE (can commit?)
Phase 2: PRE-COMMIT (will commit)
Phase 3: COMMIT (do commit)
```

**Advantage**: Can timeout and continue if coordinator fails

### Paxos

Consensus algorithm for distributed systems:

```
Proposer → Prepare(N)
  ↓
Acceptors → Promise(N, LastValue)
  ↓
Proposer → Accept(N, Value)
  ↓
Acceptors → Accepted(N, Value)
  ↓
Learners learn chosen value
```

**Simplified**:
1. Proposer suggests value with sequence number
2. Acceptors promise to not accept older proposals
3. If majority accepts, value is chosen

**Pros**: Proven correct, fault-tolerant
**Cons**: Complex, hard to implement

### Raft

Easier-to-understand consensus algorithm:

```
1. Leader Election
   Nodes elect a leader via voting

2. Log Replication
   Leader receives writes
   Leader replicates to followers
   Commits once majority acknowledges

3. Safety
   Only nodes with up-to-date logs can be leader
```

```python
class RaftNode:
    def __init__(self):
        self.state = "follower"  # follower, candidate, leader
        self.current_term = 0
        self.voted_for = None
        self.log = []

    def start_election(self):
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.id

        votes = 1  # Vote for self
        for peer in self.peers:
            if peer.request_vote(self.current_term, self.id):
                votes += 1

        if votes > len(self.peers) / 2:
            self.become_leader()

    def become_leader(self):
        self.state = "leader"
        # Send heartbeats to maintain leadership

    def append_entry(self, entry):
        if self.state != "leader":
            return redirect_to_leader()

        self.log.append(entry)

        # Replicate to majority
        acks = 1  # Self
        for peer in self.peers:
            if peer.append_entries(self.current_term, entry):
                acks += 1

        if acks > len(self.peers) / 2:
            # Committed
            return success
        else:
            return failure
```

**Pros**: Understandable, proven correct, widely used
**Cons**: Requires majority for writes (not available during partition)

**Used by**: etcd, Consul, TiKV

## Quorum Reads/Writes

Read and write from majority:

```
N = 5 nodes
W = 3 (write quorum - must write to 3)
R = 3 (read quorum - must read from 3)

W + R > N ensures read sees latest write
```

```python
class QuorumStore:
    def __init__(self, nodes, write_quorum, read_quorum):
        self.nodes = nodes
        self.W = write_quorum
        self.R = read_quorum

    def write(self, key, value):
        version = get_next_version()

        acks = 0
        for node in self.nodes:
            if node.write(key, value, version):
                acks += 1
                if acks >= self.W:
                    return success

        return failure  # Couldn't reach quorum

    def read(self, key):
        values = []

        for node in self.nodes:
            result = node.read(key)
            if result:
                values.append(result)
                if len(values) >= self.R:
                    break

        if len(values) < self.R:
            return failure

        # Return value with highest version
        return max(values, key=lambda v: v.version)
```

**Examples**: Cassandra (configurable), DynamoDB

## Conflict Resolution

### Last-Write-Wins (LWW)

Keep value with latest timestamp:

```
Node A: Write X=5 at time=100
Node B: Write X=10 at time=105
  ↓
Merge: Keep X=10 (latest timestamp)
```

**Pros**: Simple
**Cons**: Requires synchronized clocks, data loss

### Vector Clocks

Track causality to detect conflicts:

```
Node A writes: X=5, Clock={A:1}
Node B writes: X=10, Clock={B:1}
  ↓
Conflict detected (concurrent writes)
  ↓
Application resolves (merge, last-write-wins, etc.)
```

```python
# Detect conflict with vector clocks
def is_concurrent(clock1, clock2):
    not_before = any(
        clock1.get(k, 0) > clock2.get(k, 0)
        for k in clock1
    )
    not_after = any(
        clock2.get(k, 0) > clock1.get(k, 0)
        for k in clock2
    )
    return not_before and not_after

# Example
v1 = {A: 1, B: 2}
v2 = {A: 1, B: 3}
is_concurrent(v1, v2)  # False (v2 after v1)

v1 = {A: 2, B: 1}
v2 = {A: 1, B: 2}
is_concurrent(v1, v2)  # True (concurrent)
```

### CRDTs (Conflict-free Replicated Data Types)

Data structures that automatically resolve conflicts:

**G-Counter (Grow-only Counter)**:
```python
class GCounter:
    def __init__(self, node_id):
        self.node_id = node_id
        self.counts = {}  # {node_id: count}

    def increment(self):
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + 1

    def value(self):
        return sum(self.counts.values())

    def merge(self, other):
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(
                self.counts.get(node_id, 0),
                count
            )
```

**PN-Counter (Positive-Negative Counter)**:
```python
class PNCounter:
    def __init__(self, node_id):
        self.increments = GCounter(node_id)
        self.decrements = GCounter(node_id)

    def increment(self):
        self.increments.increment()

    def decrement(self):
        self.decrements.increment()

    def value(self):
        return self.increments.value() - self.decrements.value()

    def merge(self, other):
        self.increments.merge(other.increments)
        self.decrements.merge(other.decrements)
```

**OR-Set (Observed-Remove Set)**:
```python
class ORSet:
    def __init__(self):
        self.elements = {}  # {element: {unique_tags}}

    def add(self, element, unique_tag):
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(unique_tag)

    def remove(self, element):
        if element in self.elements:
            # Remember tags to remove during merge
            self.removed_tags = self.elements[element].copy()
            del self.elements[element]

    def contains(self, element):
        return element in self.elements

    def merge(self, other):
        for element, tags in other.elements.items():
            if element not in self.elements:
                self.elements[element] = set()
            self.elements[element] |= tags
```

## Distributed Locks

### Simple Lock with TTL

```python
import redis

def acquire_lock(lock_name, timeout=10):
    lock_key = f"lock:{lock_name}"
    acquired = redis.set(
        lock_key,
        "locked",
        nx=True,  # Only if not exists
        ex=timeout  # Expires in timeout seconds
    )
    return acquired

def release_lock(lock_name):
    redis.delete(f"lock:{lock_name}")

# Usage
if acquire_lock("process-order-123"):
    try:
        process_order(123)
    finally:
        release_lock("process-order-123")
```

**Problem**: Lock holder crashes, lock expires, another process acquires

### Redlock Algorithm

Acquire locks from majority of independent Redis instances:

```python
def acquire_redlock(lock_name, redis_instances):
    token = random_token()
    start_time = time.time()

    acquired = 0
    for redis in redis_instances:
        if redis.set(lock_name, token, nx=True, px=10000):
            acquired += 1

    elapsed = time.time() - start_time
    validity_time = 10000 - elapsed

    if acquired >= len(redis_instances) // 2 + 1 and validity_time > 0:
        return token, validity_time
    else:
        # Couldn't acquire majority, release all
        for redis in redis_instances:
            if redis.get(lock_name) == token:
                redis.delete(lock_name)
        return None
```

### Fencing Tokens

Prevent old lock holders from causing issues:

```
Process A acquires lock with token=1
Process A pauses (GC, network delay)
Lock expires
Process B acquires lock with token=2
Process B writes to resource
Process A wakes up, tries to write
Resource rejects (token 1 < current token 2)
```

```python
current_token = 0

def write_with_token(data, token):
    global current_token

    if token > current_token:
        current_token = token
        do_write(data)
        return success
    else:
        return failure  # Stale token
```

## Split-Brain Problem

Network partition causes multiple leaders:

```
Before partition:
[Leader A] --- [Follower B] --- [Follower C]

After partition:
[Leader A]  |  [Leader B, Follower C]
             |
        Network split
```

**Solutions**:

**Quorum**: Require majority for leader election
```python
# Side with 2 nodes can elect leader
# Side with 1 node cannot (no majority)
[Leader A]  |  [Leader B, Follower C] ← Can elect new leader
```

**Fencing**: Isolate old leader
```python
# Tell storage to reject writes from old leader
storage.fence(old_leader_id)
```

## Consistency Guarantees Comparison

| Model | Staleness | Performance | Use Case |
|-------|-----------|-------------|----------|
| **Strong** | None | Slow | Banking, inventory |
| **Eventual** | Temporary | Fast | Social media, caching |
| **Causal** | Related events ordered | Medium | Collaborative editing |
| **Read-after-write** | Own writes visible | Medium | User profiles |
| **Monotonic** | No going back in time | Medium | Feeds, timelines |

## ELI10

Distributed consensus is like a group of friends deciding where to eat:

- **Strong consistency**: Everyone must agree before choosing (slow but fair)
- **Eventual consistency**: Everyone decides separately, sort it out later (fast but messy)
- **CAP theorem**: Can't have fast decisions (A), everyone agrees (C), and handle people not responding (P)
- **Quorum**: Majority decides (more than half agree = decision made)
- **Split-brain**: Group splits, both sides think they're in charge (need quorum to prevent)

**Paxos/Raft**: Formal voting systems that guarantee everyone eventually agrees

Getting distributed systems to agree is hard!

## Further Resources

- [Designing Data-Intensive Applications (DDIA)](https://dataintensive.app/)
- [Raft Consensus Algorithm](https://raft.github.io/)
- [Jepsen: Distributed Systems Safety Research](https://jepsen.io/)
- [CAP Theorem Explained](https://www.ibm.com/topics/cap-theorem)
- [CRDTs: Conflict-free Replicated Data Types](https://crdt.tech/)
