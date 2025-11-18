# Distributed Systems

## Overview

A distributed system is a collection of independent computers that appear to its users as a single coherent system. These systems work together to achieve a common goal by communicating and coordinating their actions through message passing.

## Distributed Systems Fundamentals

### Core Challenges

#### 1. Network Failures
- **Packet loss**: Messages can be dropped in transit
- **Network partitions**: Parts of the system become isolated from each other
- **Asymmetric failures**: Node A can reach B, but B cannot reach A
- **Message reordering**: Messages may arrive out of order
- **Byzantine failures**: Nodes may behave arbitrarily or maliciously

#### 2. Latency and Performance
- **Variable latency**: Network delays are unpredictable
- **Geographic distribution**: Physical distance increases communication time
- **Bandwidth limitations**: Network capacity constraints
- **Synchronous vs asynchronous**: Trade-offs between consistency and performance

#### 3. Partial Failures
- **Individual component failures**: Single nodes can fail while others continue
- **Cascading failures**: One failure triggers others
- **Gray failures**: Components function partially, making detection difficult
- **Fail-stop vs fail-slow**: Different failure modes require different handling

### The Eight Fallacies of Distributed Computing

Originally identified by L. Peter Deutsch and others at Sun Microsystems:

1. **The network is reliable**: Networks fail, packets get lost, connections drop
2. **Latency is zero**: Communication takes time, and that time varies
3. **Bandwidth is infinite**: Network capacity is limited and shared
4. **The network is secure**: Security must be designed in, not assumed
5. **Topology doesn't change**: Network paths and configurations change dynamically
6. **There is one administrator**: Multiple organizations and teams manage different parts
7. **Transport cost is zero**: Serialization, network usage, and infrastructure have real costs
8. **The network is homogeneous**: Different protocols, formats, and systems must interoperate

### Time and Ordering

#### Physical Clocks
- **Clock drift**: Hardware clocks run at slightly different rates
- **Clock skew**: Difference between clock values at a point in time
- **NTP (Network Time Protocol)**: Synchronizes clocks across network
  - Accuracy: typically 1-50ms on internet, <1ms on LAN
  - Cannot guarantee perfect synchronization

#### Logical Clocks

##### Lamport Timestamps
- Provides partial ordering of events
- Each process maintains a counter
- Algorithm:
  1. Increment counter before each event
  2. When sending message, include timestamp
  3. On receiving message: counter = max(local_counter, message_timestamp) + 1
- **Limitation**: Cannot distinguish concurrent events

##### Vector Clocks
- Provides causal ordering
- Each process maintains vector of counters (one per process)
- Can determine if events are concurrent or causally related
- Algorithm:
  1. Increment own position in vector before event
  2. Send entire vector with message
  3. On receive: merge vectors element-wise (take max) and increment own position
- **Use cases**: Conflict detection in replicated systems (Riak, Voldemort)

##### Hybrid Logical Clocks (HLC)
- Combines physical and logical clocks
- Maintains causality like vector clocks
- Bounded by physical time
- More compact than vector clocks
- Used in: CockroachDB, MongoDB

#### Happens-Before Relationship
- Event A happens-before event B if:
  1. A and B occur on same process and A occurs before B, OR
  2. A is sending a message and B is receiving that message, OR
  3. Transitive: A → C and C → B, then A → B
- Events are concurrent if neither happens-before the other

## CAP Theorem

### The Theorem

Proven by Seth Gilbert and Nancy Lynch (2002), based on Eric Brewer's conjecture (2000):

A distributed system can only guarantee **two out of three** properties simultaneously:

- **Consistency (C)**: All nodes see the same data at the same time (linearizability)
- **Availability (A)**: Every request receives a response (success or failure), without guarantee of most recent write
- **Partition Tolerance (P)**: System continues to operate despite network partitions

### Understanding the Trade-offs

Since **network partitions are inevitable** in real-world systems, the practical choice is:

#### CP Systems (Consistency + Partition Tolerance)
**Choose consistency over availability during partitions**

- System refuses to respond or returns errors during partition
- Ensures all clients see the same data
- May sacrifice uptime

**Real-world CP systems:**
- **HBase**: Returns errors if cannot reach required replicas
- **MongoDB** (with strong consistency settings): Primary election during partition causes unavailability
- **Redis** (with wait command): Blocks until replication confirmed
- **ZooKeeper**: Refuses writes if cannot reach quorum
- **Consul**: CP for service configuration
- **Google Spanner**: Sacrifices availability for strong consistency across regions
- **etcd**: Raft-based consensus, unavailable during leader election

**Use cases:**
- Financial transactions
- Inventory management
- Systems requiring strong guarantees

#### AP Systems (Availability + Partition Tolerance)
**Choose availability over consistency during partitions**

- System always responds, even if data might be stale
- Different nodes may return different values temporarily
- Eventual consistency when partition heals

**Real-world AP systems:**
- **Cassandra**: Always available, tunable consistency
- **DynamoDB**: Eventually consistent reads by default
- **Riak**: Highly available, uses vector clocks for conflict resolution
- **CouchDB**: Multi-master replication, conflict resolution
- **Voldemort**: Shopping cart always writable (Amazon design)
- **DNS**: Availability critical, stale data acceptable

**Use cases:**
- Social media feeds
- Product catalogs
- User profiles
- Shopping carts

### PACELC Theorem

Extension by Daniel Abadi - describes behavior both during and without partitions:

- **If Partition (P)**: choose between Availability (A) and Consistency (C)
- **Else (E)**: choose between Latency (L) and Consistency (C)

**Examples:**
- **PA/EL systems**: Cassandra, Riak (Available during partition, Low latency otherwise)
- **PC/EC systems**: HBase, MongoDB (Consistent during partition, Consistent otherwise)
- **PA/EC systems**: DynamoDB (Available during partition, Consistent for normal ops)
- **PC/EL systems**: Rare, but some MySQL cluster configurations

## Consistency Models

Consistency models define guarantees about when and how updates become visible.

### 1. Strong Consistency (Linearizability)

**Guarantee**: All operations appear to occur atomically in some total order consistent with real-time ordering

- Strongest consistency model
- After write completes, all subsequent reads see that value or newer
- Operations appear instantaneous
- Expensive: requires coordination

**Implementation approaches:**
- Consensus algorithms (Paxos, Raft)
- Two-phase commit
- Primary-copy replication with synchronous replication

**Examples:**
- Google Spanner
- CockroachDB
- etcd
- ZooKeeper

### 2. Sequential Consistency

**Guarantee**: Operations appear to take effect in some sequential order, consistent with program order on each process

- Weaker than linearizability (no real-time constraint)
- All processes see operations in same order
- Each process's operations stay in order

**Use cases:**
- Multi-processor memory models
- Some distributed databases

### 3. Causal Consistency

**Guarantee**: Causally related operations are seen in the same order by all processes

- Concurrent (non-causal) operations may be seen in different orders
- Preserves happens-before relationships
- More available than sequential consistency

**Implementation:**
- Vector clocks
- Dependency tracking

**Examples:**
- COPS (Clusters of Order-Preserving Servers)
- Bolt-on Causal Consistency (Facebook)

### 4. Eventual Consistency

**Guarantee**: If no new updates, all replicas eventually converge to the same value

- Most available model
- No guarantees about intermediate states
- Convergence time unbounded (in theory)

**Variants:**

#### 4a. Read-Your-Writes Consistency
- Process always sees its own writes
- Other processes may see stale data
- Implementation: read from same replica you wrote to, or track write version

#### 4b. Monotonic Reads
- If process reads value X, subsequent reads never return older values
- Prevents "going back in time"
- Implementation: sticky sessions, or track last-read version

#### 4c. Monotonic Writes
- Process's writes are applied in order they were submitted
- Implementation: serialize writes from same client

#### 4d. Writes-Follow-Reads
- Write after reading value is guaranteed to see that read value or newer
- Implementation: include read version with write

**Examples:**
- DynamoDB (default mode)
- Cassandra (with eventual consistency level)
- Riak
- DNS

### 5. Session Consistency

**Guarantee**: Strong consistency within a session, eventual consistency across sessions

- Combines read-your-writes, monotonic reads, and writes-follow-reads
- Common in practice

**Examples:**
- Azure CosmosDB session consistency
- Many web applications with sticky sessions

## Consensus Algorithms

Consensus allows multiple nodes to agree on a single value or sequence of values, even in the presence of failures.

### Paxos

Developed by Leslie Lamport (1989), published 1998.

#### Roles
1. **Proposers**: Propose values
2. **Acceptors**: Vote on proposals (typically 2f+1 to tolerate f failures)
3. **Learners**: Learn chosen value

#### Algorithm Phases

**Phase 1: Prepare**
1. Proposer selects proposal number n, sends PREPARE(n) to acceptors
2. Acceptor receives PREPARE(n):
   - If n > any previous proposal, promise not to accept proposals < n
   - Return highest-numbered proposal already accepted (if any)

**Phase 2: Accept**
1. If proposer receives responses from majority:
   - If any acceptor already accepted value, use highest-numbered one
   - Otherwise, use own value
   - Send ACCEPT(n, value) to acceptors
2. Acceptor receives ACCEPT(n, value):
   - If n ≥ any promised proposal number, accept it
   - Notify learners

#### Challenges
- **Livelock**: Competing proposers can prevent progress
  - Solution: Use leader election, or randomized backoff
- **Complex**: Difficult to understand and implement correctly
- **Multi-Paxos**: Extension for agreeing on sequence of values (log)

#### Usage
- Google Chubby lock service
- Apache ZooKeeper (variant: ZAB - ZooKeeper Atomic Broadcast)
- Cassandra (for lightweight transactions)

### Raft - Deep Dive

Designed by Diego Ongaro and John Ousterhout (2014) for understandability.

#### Core Principles
- **Strong leader**: Log entries only flow from leader to followers
- **Decomposed problem**: Separate leader election, log replication, safety
- **Simplicity**: Easier to understand and implement than Paxos

#### Server States
1. **Leader**: Handles all client requests, sends heartbeats
2. **Follower**: Passive, responds to RPCs from leader and candidates
3. **Candidate**: Used to elect new leader

#### Terms
- Logical clock numbered with consecutive integers
- Each term has at most one leader
- Servers maintain current term number
- Term advances when:
  - Follower times out and becomes candidate
  - Server discovers higher term

#### Leader Election

**Trigger**: Follower doesn't receive heartbeat within election timeout (randomized: 150-300ms)

**Process**:
1. Follower increments term, transitions to candidate
2. Votes for self
3. Sends RequestVote RPCs to all servers
4. Outcomes:
   - **Wins election**: Receives votes from majority → becomes leader
   - **Another server wins**: Receives heartbeat with ≥ term → becomes follower
   - **Timeout**: Split vote, nobody wins → start new election (increment term, retry)

**Vote granting**:
- One vote per term, first-come-first-served
- Candidate's log must be at least as up-to-date:
  - Last log entry has higher term, OR
  - Same term but log is at least as long

**Election timeout randomization**: Prevents split votes

#### Log Replication

**Normal operation**:
1. Client sends command to leader
2. Leader appends entry to local log
3. Leader sends AppendEntries RPCs to followers
4. Once replicated on majority: entry is **committed**
5. Leader applies entry to state machine, returns result to client
6. Leader includes commit index in heartbeats
7. Followers apply committed entries to their state machines

**Log matching property**:
- If two logs contain entry with same index and term:
  - They store the same command
  - All preceding entries are identical

**Consistency check**:
- AppendEntries includes index and term of immediately preceding entry
- Follower rejects if it doesn't have matching entry
- Leader decrements nextIndex and retries
- Eventually finds point where logs match, overwrites follower's inconsistent entries

#### Safety Properties

**Election restriction**: Leader must contain all committed entries
- Ensured by vote granting rule (candidate's log must be up-to-date)

**Commitment rule**: Leader never overwrites or deletes entries in its log
- Only appends new entries

**State machine safety**: If server has applied log entry at index i, no other server applies different entry at index i

#### Log Compaction (Snapshotting)
- Snapshot includes:
  - State machine state
  - Last included index and term
- Discard log entries before snapshot
- Send InstallSnapshot RPC to slow followers

#### Cluster Membership Changes
- **Joint consensus**: Two configurations overlap during transition
- Prevents split-brain during reconfiguration

#### Key Advantages
1. **Understandability**: Clear separation of concerns
2. **Strong leader**: Simplifies log replication
3. **Randomized timeouts**: Solves split vote problem elegantly
4. **Membership changes**: Safe reconfiguration protocol

**Implementations**:
- etcd (Kubernetes)
- Consul (HashiCorp)
- CockroachDB
- TiKV (TiDB)

### Two-Phase Commit (2PC)

Atomic commitment protocol for distributed transactions.

#### Roles
- **Coordinator**: Orchestrates the commit
- **Participants**: Resources being committed (databases, services)

#### Phases

**Phase 1: Prepare (Voting)**
1. Coordinator sends PREPARE message to all participants
2. Each participant:
   - Prepares transaction (write to redo log, acquire locks)
   - Votes YES (can commit) or NO (abort)
   - If YES, enters prepared state (cannot unilaterally abort)

**Phase 2: Commit/Abort**
1. Coordinator collects votes:
   - If all YES: sends COMMIT to all participants
   - If any NO or timeout: sends ABORT to all participants
2. Participants execute command and acknowledge
3. Coordinator completes when all acknowledgments received

#### Problems

**Blocking protocol**:
- If coordinator crashes after PREPARE, participants are blocked
- Cannot commit or abort without coordinator decision
- Locks held until coordinator recovers

**No progress guarantee**:
- Single point of failure (coordinator)
- Participant failures also block progress

**Performance**:
- Multiple round-trips
- Synchronous blocking
- High latency

**Usage**: Traditional distributed databases (Oracle, DB2, MySQL XA)

### Three-Phase Commit (3PC)

Non-blocking extension of 2PC.

#### Additional Phase: Pre-Commit

**Phase 1: CanCommit**
- Like 2PC prepare phase

**Phase 2: PreCommit**
- Coordinator sends PRECOMMIT if all voted YES
- Participants acknowledge
- **Key property**: If participant receives PRECOMMIT, it knows all voted YES

**Phase 3: DoCommit**
- Coordinator sends COMMIT
- Participants commit and acknowledge

#### Advantages
- **Non-blocking**: Participants can make progress using timeout + state machine
- If participant times out in pre-commit state, it knows all voted YES → can commit

#### Disadvantages
- **Network partitions**: Can lead to inconsistency if partition occurs between phases
- **More latency**: Additional round-trip
- **Rarely used in practice**: Complexity outweighs benefits; partition tolerance is critical

## Distributed Transactions

### ACID in Distributed Systems

Traditional ACID properties are challenging in distributed environments:

#### Atomicity
- **Challenge**: Partial failures across multiple nodes
- **Solutions**:
  - Two-phase commit (2PC)
  - Saga pattern with compensating transactions
  - Consensus-based approaches (Raft, Paxos)

#### Consistency
- **Challenge**: Maintaining invariants across distributed data
- **Solutions**:
  - Application-level validation
  - Distributed constraints checking
  - Eventual consistency with conflict resolution

#### Isolation
- **Challenge**: Coordinating concurrent access across nodes
- **Solutions**:
  - Distributed locking (pessimistic)
  - Optimistic concurrency control
  - Snapshot isolation (Google Spanner)
  - Serializable Snapshot Isolation (SSI)

#### Durability
- **Challenge**: Ensuring writes survive failures
- **Solutions**:
  - Replication (synchronous or asynchronous)
  - Write-ahead logging
  - Quorum-based writes

### Saga Pattern

Long-lived transactions broken into sequence of local transactions, each with compensating action.

#### Choreography

**Decentralized coordination**: Each service produces and listens to events

**Example**: Order placement
```
1. Order Service: Create order → Emit OrderCreated event
2. Inventory Service: Reserve items → Emit ItemsReserved (or ReservationFailed)
3. Payment Service: Charge customer → Emit PaymentSucceeded (or PaymentFailed)
4. Shipping Service: Schedule shipment → Emit ShipmentScheduled

If any step fails:
- Emit failure event
- Previous services listen and execute compensating transactions
```

**Advantages**:
- No central coordination
- Loose coupling
- Good for simple workflows

**Disadvantages**:
- Hard to understand and debug
- Difficult to track overall state
- Complex error handling
- Cyclic dependencies possible

#### Orchestration

**Centralized coordination**: Orchestrator tells services what to do

**Example**: Same order placement
```
Orchestrator:
1. Call Order Service: Create order
2. Call Inventory Service: Reserve items
   - If fails: Call Order Service: Cancel order → END
3. Call Payment Service: Charge customer
   - If fails: Call Inventory: Release items → Call Order: Cancel → END
4. Call Shipping Service: Schedule shipment
   - If fails: Call Payment: Refund → Call Inventory: Release → Call Order: Cancel → END
```

**Advantages**:
- Clear workflow logic in one place
- Easier to understand and debug
- Centralized monitoring
- Timeout management simplified

**Disadvantages**:
- Orchestrator is potential bottleneck
- Additional infrastructure required
- Tighter coupling to orchestrator

### Compensating Transactions

**Semantic undo**: Logically reverse a transaction (not physical undo)

**Examples**:
- Order placement → Order cancellation
- Money debit → Money credit
- Item reservation → Item release
- Email sent → Correction email (cannot "unsend")

**Key properties**:
- **Idempotent**: Safe to retry
- **Commutative** (ideally): Order shouldn't matter
- **Semantically correct**: Achieves business goal of reversal

**Challenges**:
- Some actions cannot be compensated (sent email, published data)
- Timing issues (compensate before user sees original effect?)
- Partial compensations
- Compensation failures (need retries, dead letter queues)

**Best practices**:
1. Design compensating actions upfront
2. Make them idempotent
3. Log all actions for audit trail
4. Monitor saga execution
5. Alert on compensation failures
6. Consider time windows for compensation

## Event Sourcing

### Core Concept

**Event log as source of truth**: Store all changes as immutable sequence of events, rather than storing current state.

**Traditional approach**:
```
User account table: { id: 1, name: "Alice", email: "alice@example.com", balance: 1000 }
Update balance → Overwrite value
```

**Event sourcing**:
```
Events:
1. UserCreated(id=1, name="Alice", email="alice@example.com")
2. DepositMade(id=1, amount=1000)
3. WithdrawalMade(id=1, amount=200)
4. DepositMade(id=1, amount=200)

Current state = replay all events
Balance = 0 + 1000 - 200 + 200 = 1000
```

### Key Benefits

1. **Audit trail**: Complete history of what happened
2. **Time travel**: Reconstruct state at any point in time
3. **Event replay**: Fix bugs by replaying with corrected logic
4. **Multiple projections**: Build different views from same events
5. **Debug and analysis**: Understand how system reached current state
6. **Event notifications**: Other systems subscribe to events

### Event Store

**Append-only log** of events:
- Events are immutable
- Only append new events, never modify or delete
- Events ordered (typically per aggregate)

**Operations**:
- **Append**: Add new event
- **Read**: Get events for aggregate or time range
- **Subscribe**: Listen for new events

**Implementations**:
- Event Store DB
- Apache Kafka
- Custom database tables
- AWS DynamoDB Streams

### Event Replay

**Rebuild state** by replaying events:

```
Initial state: empty
Apply UserCreated → { id: 1, name: "Alice", email: "alice@example.com", balance: 0 }
Apply DepositMade → { balance: 1000 }
Apply WithdrawalMade → { balance: 800 }
Apply DepositMade → { balance: 1000 }
```

**Use cases**:
- Rebuild read models after schema change
- Fix bugs in event handlers
- Create new projections
- Audit and compliance

**Challenges**:
- Slow for large event streams
- Schema evolution (old events with old format)
- Solution: Snapshots

### Snapshotting

**Periodic state snapshots** to avoid replaying all events.

**Process**:
1. Replay events up to snapshot point
2. Save snapshot with version/event number
3. To rebuild: Load snapshot + replay subsequent events

**Example**:
```
Events 1-1000: Snapshot at event 1000 (balance = 5000)
Events 1001-1500: Current state
To get current state: Load snapshot + replay events 1001-1500
```

**Snapshot strategies**:
- **Periodic**: Every N events or time interval
- **On-demand**: When loading latest snapshot + replaying is still fast enough
- **Per aggregate**: Different aggregates snapshot independently

**Storage**:
- Same event store
- Separate snapshot store
- Cache (Redis, Memcached)

### CQRS Integration

**Command Query Responsibility Segregation**: Separate models for reads and writes.

**Event sourcing + CQRS**:

**Write side (Command)**:
- Commands validate and generate events
- Events appended to event store
- No read operations on write model

**Read side (Query)**:
- Event handlers build projections (read models)
- Optimized for queries (denormalized, indexed)
- Can have multiple projections for different use cases

**Example**:
```
Commands (Write):
- CreateOrder
- AddOrderItem
- PlaceOrder
→ Generate events: OrderCreated, ItemAdded, OrderPlaced
→ Store in event log

Events published →

Read Models (Query):
1. Order details view: Relational table with current order state
2. Order history view: Timeline of order changes
3. Analytics view: Aggregated sales data
4. Search index: Elasticsearch for order search
```

**Benefits**:
- Independent scaling of reads and writes
- Optimize each side for its purpose
- Multiple specialized read models
- Eventual consistency acceptable

**Challenges**:
- Eventual consistency between write and read
- More complex architecture
- Data duplication across projections
- Need to handle projection rebuilds

### Event Schema Evolution

**Problem**: Old events with old schema, new code expects new schema

**Strategies**:

1. **Upcasting**: Convert old events to new format when reading
2. **Versioned events**: Include version number, handle each version
3. **Weak schema**: Use flexible formats (JSON) with optional fields
4. **Event migration**: Background process to rewrite old events (rare)

### Best Practices

1. **Events are facts**: Past tense (UserRegistered, OrderPlaced, not RegisterUser)
2. **Events are immutable**: Never change or delete events
3. **Domain events**: Model business events, not CRUD operations
4. **Idempotency**: Handle duplicate events gracefully
5. **Event size**: Keep events small and focused
6. **Correlation IDs**: Track related events across aggregates
7. **Metadata**: Timestamp, user, causation ID, correlation ID
8. **Testing**: Verify state transitions via event replay

## Replication

Keeping copies of data on multiple nodes for fault tolerance and performance.

### Leader-Follower Replication (Master-Slave)

**One leader** accepts writes, **followers** replicate and serve reads.

#### Synchronous Replication
- Leader waits for follower acknowledgment before confirming write
- **Pros**: Follower guaranteed to have up-to-date copy
- **Cons**: Write latency increases, unavailable if follower down
- **Semi-synchronous**: Wait for one follower, others async

#### Asynchronous Replication
- Leader confirms write immediately, replicates in background
- **Pros**: Low latency, high availability
- **Cons**: Data loss if leader fails before replication
- **Most common** in practice

#### Follower Failure and Catch-up
- Follower keeps log of processed transactions
- On reconnect, requests all changes since last processed
- Applies changes to catch up

#### Leader Failure (Failover)

**Detection**: Heartbeat timeout (typically 30s)

**New leader election**:
1. Promote follower (often most up-to-date)
2. Reconfigure clients to send writes to new leader
3. Old leader becomes follower when it recovers

**Challenges**:
- **Data loss**: If async replication, some writes lost
- **Split brain**: Two nodes think they're leader
- **Timeout tuning**: Too short → unnecessary failovers, too long → longer downtime

#### Replication Log Implementation

**Statement-based**: Ship SQL statements
- **Problem**: Non-deterministic functions (NOW(), RAND())

**Write-ahead log (WAL) shipping**: Ship low-level disk writes
- **Problem**: Tightly coupled to storage engine

**Logical (row-based) log**: Ship logical row changes
- **Most common**: Decoupled from storage, supports different versions

**Trigger-based**: Application-level triggers
- **Flexibility**: Custom logic, but higher overhead

### Multi-Leader Replication (Multi-Master)

**Multiple nodes** accept writes, replicate to each other.

#### Use Cases
- **Multi-datacenter**: Leader in each datacenter
- **Offline clients**: Each device is a leader (mobile apps)
- **Collaborative editing**: Each user's edits are writes

#### Advantages
- **Performance**: Lower latency (write to nearest leader)
- **Fault tolerance**: Continue operating if datacenter fails
- **Availability**: Each datacenter operates independently

#### Conflict Resolution

**Conflicts inevitable**: Same key modified concurrently at different leaders

**Example**:
```
User A (DC1): Update title = "Distributed Systems"
User B (DC2): Update title = "Distributed Computing"
Both writes succeed locally, then replicate to each other
→ Conflict!
```

**Resolution strategies**:

1. **Last-write-wins (LWW)**:
   - Use timestamp or version number
   - **Problem**: Data loss, timestamp synchronization issues
   - **Use**: Cassandra, Riak (with client-side timestamps)

2. **Application-level resolution**:
   - Application provides conflict handler
   - **Example**: Merge function for collaborative editing
   - **Use**: CouchDB

3. **Multi-value (version vectors)**:
   - Keep all conflicting versions
   - Application reads all versions and resolves
   - **Use**: Riak, Voldemort

4. **CRDT (Conflict-free Replicated Data Types)**:
   - Data structures with built-in conflict resolution
   - Mathematically proven to converge
   - **Examples**: Counters, sets, maps
   - **Use**: Riak (maps), Redis (CRDTs)

5. **Operational Transform**:
   - Transform concurrent operations so they can be applied in any order
   - **Use**: Google Docs, collaborative editing

**Custom topologies**:
- **Circular**: Each leader replicates to next in ring
- **Star**: One designated root, others replicate through it
- **All-to-all**: Every leader replicates to every other (most common)

### Leaderless Replication (Dynamo-style)

**No leader**: Client writes to multiple replicas directly.

#### Key Concepts

**Quorum reads and writes**:
- N = total replicas
- W = write quorum (replicas that must acknowledge write)
- R = read quorum (replicas that must respond to read)
- **Rule**: W + R > N ensures reads see recent writes

**Example**: N=3, W=2, R=2
- Write succeeds when 2 of 3 replicas acknowledge
- Read queries 2 of 3 replicas, takes newest value

#### Read Repair
- Read queries multiple replicas
- If stale data detected, write newer value back
- Ensures eventually all replicas converge

#### Anti-Entropy Process
- Background process compares replicas
- Synchronizes differences
- Uses Merkle trees for efficient comparison

#### Sloppy Quorums and Hinted Handoff

**Problem**: W replicas unavailable, write would fail

**Sloppy quorum**: Accept writes to any W available nodes, even if not "home" replicas

**Hinted handoff**: When home replica recovers, temporary replica forwards writes

**Trade-off**: Higher availability, but W + R > N doesn't guarantee latest value

#### Conflict Resolution
- Same strategies as multi-leader (LWW, version vectors, CRDTs)
- **Siblings**: Multiple conflicting values returned to client
- **Application resolves**: Client merges conflicts

**Examples**:
- Amazon DynamoDB
- Apache Cassandra
- Riak
- Voldemort

### Conflict Resolution Strategies (Detailed)

#### 1. Version Vectors (Vector Clocks)

Track causality to detect conflicts:

```
Initial: {}
Write A: {A:1} value="Alice"
Write B: {B:1} value="Bob"
Replicate A→B: {A:1, B:1} (conflict detected!)
Replicate B→A: {A:1, B:1} (conflict detected!)
→ Application resolves: {A:1, B:1} value="Alice, Bob"
```

#### 2. CRDTs (Conflict-free Replicated Data Types)

**Grow-only Counter (G-Counter)**:
- Each replica maintains counter per node
- Increment local counter
- Merge: take max of each position
- Value = sum of all counters

**PN-Counter (Positive-Negative Counter)**:
- Two G-Counters: increments and decrements
- Value = increments - decrements

**G-Set (Grow-only Set)**:
- Add-only set
- Merge: union

**OR-Set (Observed-Remove Set)**:
- Add includes unique tag
- Remove based on observed tags
- Merge: union adds, remove only if tag in removed set

**LWW-Register (Last-Write-Wins Register)**:
- Each write includes timestamp
- Merge: keep value with latest timestamp

#### 3. Operational Transform

Transform concurrent operations to maintain consistency:

```
Initial: "Hello"
Op1: Insert("World", position=5)  → "HelloWorld"
Op2: Delete(position=0, length=1) → "ello"

Transform Op1 for Op2: Insert("World", position=4)  → "elloWorld"
Both paths converge to same result
```

## Partitioning (Sharding)

Splitting data across multiple nodes to scale beyond single machine capacity.

### Horizontal vs Vertical Partitioning

#### Vertical Partitioning
- Split **columns** into separate tables/databases
- **Example**: User table → (UserProfile, UserActivity, UserSettings)
- **Use case**: Different access patterns, separate hot/cold data
- **Limit**: Still limited by single-entity scale

#### Horizontal Partitioning (Sharding)
- Split **rows** across multiple nodes
- **Example**: Users 1-1000 → Node A, Users 1001-2000 → Node B
- **Use case**: True scalability, no single-node bottleneck

### Sharding Strategies

#### 1. Range-Based Sharding

**Partition by key ranges**:
```
A-F → Shard 1
G-M → Shard 2
N-Z → Shard 3
```

**Advantages**:
- Range queries efficient
- Easy to understand

**Disadvantages**:
- **Hotspots**: Uneven distribution (many names start with S, few with Q)
- Load imbalance
- Requires understanding of data distribution

**Example**: HBase, MongoDB (with range-based shard keys)

#### 2. Hash-Based Sharding

**Hash key to determine partition**:
```
hash(user_id) % num_shards → shard_id
```

**Advantages**:
- Even distribution
- No hotspots (if good hash function)

**Disadvantages**:
- Range queries require querying all shards
- Rebalancing requires moving data

**Example**: Cassandra, Redis Cluster

#### 3. Directory-Based Sharding

**Lookup table** maps keys to shards:
```
Lookup table:
user_id=1 → Shard A
user_id=2 → Shard A
user_id=3 → Shard B
```

**Advantages**:
- Flexible placement
- Easy to rebalance (update directory)
- Can use any partitioning logic

**Disadvantages**:
- Lookup table is bottleneck and single point of failure
- Additional latency

**Example**: Some MySQL sharding solutions

### Consistent Hashing

Minimizes data movement when nodes added/removed.

**Algorithm**:
1. Hash nodes and keys to same hash space (e.g., 0-2^32)
2. Arrange nodes on hash ring
3. Key belongs to first node clockwise from key position

**Example**:
```
Ring: [Node A at 0, Node B at 1000, Node C at 2000]
Key X hashes to 1500 → belongs to Node C
```

**Adding node D at 500**:
- Only keys between Node A (0) and Node D (500) move to Node D
- ~1/4 of keys move (not all keys like in modulo hashing)

**Virtual nodes**:
- Each physical node represented by multiple virtual nodes
- Better load distribution
- Smoother scaling

**Usage**:
- Cassandra
- DynamoDB
- Riak
- Chord DHT
- Memcached (client-side)

### Rebalancing

**Goal**: Move data when adding/removing nodes while minimizing disruption

#### Strategies

**1. Don't use hash % num_nodes**:
- Problem: Changing num_nodes moves almost all keys
- Solution: Use consistent hashing or fixed number of partitions

**2. Fixed number of partitions**:
- Create many partitions upfront (e.g., 1000)
- Assign partitions to nodes
- When adding node, move some partitions to new node
- **Example**: Riak, Elasticsearch, Couchbase

**3. Dynamic partitioning**:
- Split partitions when they grow too large
- Merge when too small
- **Example**: HBase, MongoDB

**4. Proportional to nodes**:
- Fixed number of partitions per node
- When node added, steal partitions from existing nodes
- **Example**: Cassandra (virtual nodes)

#### Rebalancing Process

**Manual vs Automatic**:
- **Manual**: Administrator triggers rebalancing
  - More control, prevents cascading failures
- **Automatic**: System rebalances automatically
  - Convenient, but can cause issues during partial failures

**Challenges**:
- **Network load**: Rebalancing moves lots of data
- **Performance impact**: Resources diverted from serving requests
- **Consistency**: Ensure availability during rebalancing

### Partitioning and Secondary Indexes

**Problem**: How to handle queries by non-partition key?

#### Document-based Partitioning (Local Index)

Each partition maintains index for its own data only.

**Query process**: Scatter-gather across all partitions

**Example**:
```
Partition 1: Users A-M, index on age for users A-M
Partition 2: Users N-Z, index on age for users N-Z

Query "age=25": Query both partitions, merge results
```

**Pros**: Writes only affect one partition
**Cons**: Reads are expensive (query all partitions)

**Use**: MongoDB, Cassandra

#### Term-based Partitioning (Global Index)

Index itself is partitioned separately from data.

**Example**:
```
Data partitions: by user_id
Index partition 1: age 0-25
Index partition 2: age 26-50
Index partition 3: age 51+

Query "age=25": Query index partition 1 only, then fetch data
```

**Pros**: Reads are efficient
**Cons**: Writes slower (update data partition and index partition), eventual consistency

**Use**: DynamoDB (Global Secondary Indexes), Riak Search

## Distributed Caching

### Cache Invalidation Strategies

#### 1. Time-to-Live (TTL)
- Entry expires after fixed duration
- **Pros**: Simple, prevents stale data
- **Cons**: May serve stale data before TTL, cache miss on expiry

#### 2. Write-Through
- Write to cache and database simultaneously
- **Pros**: Cache always consistent with database
- **Cons**: Higher write latency, wasted cache space for rarely-read data

#### 3. Write-Behind (Write-Back)
- Write to cache, asynchronously write to database
- **Pros**: Low write latency
- **Cons**: Risk of data loss, complexity

#### 4. Cache-Aside (Lazy Loading)
```
1. Check cache
2. If miss: Read from database, write to cache, return data
3. If hit: Return data from cache
On write: Invalidate cache (or update)
```
- **Pros**: Only caches requested data
- **Cons**: Cache miss penalty, potential for stale data

#### 5. Refresh-Ahead
- Automatically refresh hot entries before expiration
- **Pros**: Reduces cache misses for popular items
- **Cons**: Difficult to predict what to refresh

### Cache Coherence

**Problem**: Keeping multiple cache copies consistent

#### Strategies

**1. Invalidation-based**:
- When data changes, invalidate all cached copies
- Next access fetches fresh data
- **Use**: Most distributed caches (Redis, Memcached)

**2. Update-based**:
- When data changes, push updates to all caches
- **Pros**: No stale reads
- **Cons**: More network traffic

**3. Lease-based**:
- Cache entries have leases (time-limited)
- Source can revoke leases to invalidate
- **Use**: Some CDNs

**4. Version-based**:
- Include version with cached data
- Check version on read
- **Use**: HTTP ETags

#### Thundering Herd Problem

**Problem**: Cache expires, many requests simultaneously query database

**Solutions**:

1. **Request coalescing**: Only one request fetches, others wait
2. **Probabilistic early expiration**: Refresh before TTL with probability
3. **Lock-based**: First request acquires lock, others wait or use stale data
4. **Sentinel values**: Placeholder while refreshing

### Distributed Cache Architectures

#### 1. Client-Side Caching
- Each client has local cache
- **Pros**: Lowest latency
- **Cons**: Coherence challenges, memory usage

#### 2. Server-Side Caching
- Cache layer between clients and database
- **Pros**: Centralized control
- **Cons**: Network hop

#### 3. CDN (Content Delivery Network)
- Geographically distributed caches
- **Use**: Static assets, media
- **Examples**: Cloudflare, Akamai, CloudFront

### Cache Replacement Policies

- **LRU (Least Recently Used)**: Evict least recently accessed
- **LFU (Least Frequently Used)**: Evict least frequently accessed
- **FIFO (First In First Out)**: Evict oldest
- **Random**: Evict random entry
- **ARC (Adaptive Replacement Cache)**: Balances recency and frequency

## Real-World Distributed Systems

### Google

#### Bigtable
- **Type**: Wide-column store (column-family database)
- **Architecture**:
  - Data stored in tablets (row ranges)
  - Tablet servers serve read/write requests
  - Master assigns tablets to servers
  - GFS (Google File System) for storage
  - Chubby for coordination and master election
- **Data model**: (row key, column key, timestamp) → value
- **Features**:
  - Sorted by row key
  - Strong consistency for single-row transactions
  - Atomic row operations
  - Bloom filters for efficient lookups
- **Use cases**: Google Search, Maps, Gmail
- **Inspired**: HBase, Cassandra, Hypertable

#### Spanner
- **Type**: Globally distributed SQL database
- **Architecture**:
  - Paxos groups for replication
  - TrueTime API for global consistency
  - Two-phase commit for distributed transactions
- **TrueTime**:
  - GPS and atomic clocks in each datacenter
  - Returns time interval with guaranteed bounds
  - Enables serializable transactions globally
- **Features**:
  - Linearizability across data centers
  - ACID transactions
  - SQL queries
  - Schema changes without downtime
- **Trade-offs**:
  - Write latency (cross-datacenter commits)
  - Requires specialized hardware (TrueTime)
- **Use cases**: Google AdWords, Play
- **Inspired**: CockroachDB, YugabyteDB

#### Other Google Systems
- **GFS/Colossus**: Distributed file system
- **MapReduce/Dataflow**: Distributed computation
- **Chubby**: Distributed lock service
- **Megastore**: Semi-relational database (predecessor to Spanner)

### Amazon

#### DynamoDB
- **Type**: Key-value and document database
- **Architecture**:
  - Consistent hashing for partitioning
  - Leaderless replication (Dynamo-style)
  - Multi-datacenter replication
- **Consistency models**:
  - Eventually consistent reads (default)
  - Strongly consistent reads (optional)
  - Transactions (ACID for multiple items)
- **Features**:
  - Automatic partitioning and rebalancing
  - Global tables (multi-region)
  - Streams (change data capture)
  - On-demand and provisioned capacity
- **Quorums**: W=2, R=2, N=3 (configurable via read consistency)
- **Conflict resolution**: Last-write-wins (LWW) by default
- **Use cases**: Amazon.com, Alexa, gaming leaderboards
- **Inspired**: Cassandra, Riak, Voldemort

#### Other Amazon Systems
- **S3**: Object storage (eventual consistency → strong consistency as of 2020)
- **Aurora**: MySQL/PostgreSQL-compatible relational database
  - Replicates storage across 3 AZs (6 copies)
  - Quorum: W=4, R=3, N=6
- **EBS**: Block storage with replication

### Facebook (Meta)

#### Cassandra
- **Origin**: Developed at Facebook, open-sourced 2008
- **Type**: Wide-column store
- **Architecture**:
  - Dynamo-style partitioning (consistent hashing)
  - Bigtable-style data model
  - Leaderless replication
  - Gossip protocol for cluster membership
- **Consistency levels**: ONE, QUORUM, ALL (per-query tunable)
- **Features**:
  - Linear scalability
  - Multi-datacenter replication
  - Lightweight transactions (Paxos-based)
  - CQL (Cassandra Query Language)
- **Write path**: MemTable → SSTable
- **Read path**: Bloom filters → SSTables → compaction
- **Use cases**: Originally for Facebook inbox search, now widely used (Netflix, Apple, Instagram)

#### TAO (The Associations and Objects)
- **Type**: Distributed data store for social graph
- **Architecture**:
  - Graph database on top of MySQL
  - Read-optimized, heavily cached
  - Write-through cache
- **Features**:
  - Optimized for social graph queries (friends, likes, comments)
  - Eventually consistent reads
  - Asynchronous replication across datacenters
- **Scale**: Billions of nodes, trillions of edges

#### Other Facebook Systems
- **Haystack**: Photo storage
- **Memcache**: Massive distributed cache layer
- **RocksDB**: Embedded key-value store (based on LevelDB)
- **Presto**: Distributed SQL query engine

### Other Notable Systems

#### Apache Kafka (LinkedIn)
- **Type**: Distributed event streaming platform
- **Architecture**:
  - Topics partitioned across brokers
  - ZooKeeper for coordination (moving to KRaft)
  - Replication with leader-follower
- **Features**:
  - High throughput (millions msgs/sec)
  - Persistent log
  - At-least-once, exactly-once semantics
  - Consumer groups for parallel processing

#### Redis
- **Type**: In-memory data structure store
- **Features**:
  - Replication (leader-follower)
  - Sentinel for high availability
  - Cluster mode for partitioning
  - Persistence (RDB snapshots, AOF log)
  - CRDTs support
- **Use cases**: Caching, session store, leaderboards, pub/sub

#### Elasticsearch
- **Type**: Distributed search and analytics
- **Architecture**:
  - Built on Lucene
  - Sharding and replication
  - Master-eligible nodes elect leader
- **Features**:
  - Full-text search
  - Real-time indexing
  - Aggregations for analytics
  - RESTful API

## Patterns and Anti-Patterns

### Distributed System Patterns

#### 1. Circuit Breaker
- **Purpose**: Prevent cascading failures
- **How**: Track failure rate, open circuit if threshold exceeded
- **States**: Closed (normal), Open (failing), Half-Open (testing)
- **Example**: Hystrix, Resilience4j

#### 2. Bulkhead
- **Purpose**: Isolate resources to limit blast radius
- **How**: Separate thread pools/connection pools per service
- **Example**: 100 threads total → 30 for service A, 30 for B, 40 for C

#### 3. Retry with Exponential Backoff
- **Purpose**: Handle transient failures
- **How**: Retry with increasing delays (1s, 2s, 4s, 8s)
- **Enhancement**: Add jitter to prevent thundering herd

#### 4. Idempotency
- **Purpose**: Safe retry of operations
- **How**: Same request produces same result, no side effects on retry
- **Implementation**: Idempotency keys, deterministic UUIDs

#### 5. Timeout
- **Purpose**: Prevent indefinite waiting
- **How**: Set maximum wait time for operations
- **Challenge**: Choosing right timeout value

#### 6. Rate Limiting / Throttling
- **Purpose**: Protect system from overload
- **Algorithms**: Token bucket, leaky bucket, fixed/sliding window
- **Example**: Max 100 requests/second per user

#### 7. Load Shedding
- **Purpose**: Gracefully degrade under extreme load
- **How**: Reject low-priority requests, serve high-priority only
- **Example**: Serve logged-in users, reject anonymous

#### 8. Health Checks
- **Purpose**: Detect unhealthy instances
- **Types**: Liveness (is it running?), Readiness (can it serve traffic?)
- **Implementation**: HTTP endpoint (/health), regular probing

#### 9. Service Discovery
- **Purpose**: Dynamic service location
- **Patterns**: Client-side (Eureka), Server-side (Consul), DNS-based
- **Example**: Service registers with Consul, client queries Consul

#### 10. API Gateway
- **Purpose**: Single entry point for clients
- **Functions**: Routing, authentication, rate limiting, load balancing
- **Example**: Kong, Ambassador, AWS API Gateway

#### 11. Sidecar Pattern
- **Purpose**: Augment service with additional capabilities
- **How**: Deploy helper container alongside main container
- **Use cases**: Logging, monitoring, service mesh proxy (Envoy)

#### 12. Strangler Fig Pattern
- **Purpose**: Incrementally migrate legacy system
- **How**: Route requests to new system, fall back to legacy
- **Process**: Gradually replace pieces until legacy retired

### Anti-Patterns

#### 1. Distributed Monolith
- **Problem**: Microservices with tight coupling
- **Symptoms**: Must deploy all services together, shared database
- **Solution**: Proper service boundaries, loose coupling

#### 2. Chatty Services
- **Problem**: Excessive inter-service communication
- **Symptoms**: N+1 queries, high latency, network saturation
- **Solution**: Batch requests, caching, coarser-grained APIs

#### 3. Mega Service
- **Problem**: Service doing too much
- **Symptoms**: Hard to scale, deploy, understand
- **Solution**: Split into smaller services with clear boundaries

#### 4. Shared Database
- **Problem**: Multiple services accessing same database
- **Symptoms**: Tight coupling, hard to evolve schema
- **Solution**: Database per service, async replication

#### 5. Ignoring Network Failures
- **Problem**: Not handling network issues
- **Symptoms**: Hangs, cascading failures, poor UX
- **Solution**: Timeouts, retries, circuit breakers, fallbacks

#### 6. Synchronous Coupling
- **Problem**: Over-reliance on synchronous calls
- **Symptoms**: Tight coupling, cascading failures, high latency
- **Solution**: Async messaging, event-driven architecture

#### 7. Missing Observability
- **Problem**: Can't understand system behavior
- **Symptoms**: Hard to debug, slow to detect issues
- **Solution**: Logging, metrics, distributed tracing

#### 8. No Idempotency
- **Problem**: Retries cause duplicate side effects
- **Symptoms**: Double charges, duplicate records
- **Solution**: Idempotency keys, idempotent operations

#### 9. Single Point of Failure
- **Problem**: One component failure brings down entire system
- **Symptoms**: System outages from single failure
- **Solution**: Redundancy, replication, failover

#### 10. Ignoring CAP Theorem
- **Problem**: Expecting strong consistency AND high availability during partitions
- **Symptoms**: Surprised by eventual consistency, data loss
- **Solution**: Understand trade-offs, choose appropriate model

#### 11. Premature Optimization
- **Problem**: Over-engineering for scale not yet needed
- **Symptoms**: Complex architecture, high costs, slow development
- **Solution**: Start simple, scale when needed

#### 12. Death by a Thousand Microservices
- **Problem**: Too many small services
- **Symptoms**: High operational overhead, complex deployments, hard to trace
- **Solution**: Right-size services, group related functionality

## Service Mesh

### Overview

**Service mesh**: Infrastructure layer for service-to-service communication, handling cross-cutting concerns.

#### Core Capabilities

1. **Traffic Management**: Load balancing, routing, failover
2. **Security**: mTLS, authentication, authorization
3. **Observability**: Metrics, logs, traces
4. **Resilience**: Retries, timeouts, circuit breakers

### Architecture

#### Data Plane
- **Sidecar proxies**: Deployed alongside each service instance
- **Intercept traffic**: All service communication flows through proxy
- **Popular proxies**: Envoy, Linkerd-proxy, NGINX

#### Control Plane
- **Configuration**: Push config to data plane proxies
- **Service discovery**: Track service instances
- **Certificate management**: Issue and rotate certificates
- **Telemetry aggregation**: Collect metrics and traces

### Popular Service Meshes

#### Istio
- **Data plane**: Envoy proxy
- **Control plane**: istiod (unified control plane)
- **Features**:
  - Rich traffic management (canary, A/B testing)
  - Strong security (mutual TLS by default)
  - Extensive observability
  - Multi-cluster support
- **Complexity**: Feature-rich but complex to operate
- **Use cases**: Large enterprises, complex traffic patterns

#### Linkerd
- **Data plane**: Custom Rust-based proxy (linkerd2-proxy)
- **Control plane**: Simplified architecture
- **Features**:
  - Lightweight and fast
  - Automatic mTLS
  - Service profiles for per-route metrics
  - Multi-cluster support
- **Simplicity**: Easier to adopt and operate than Istio
- **Use cases**: Teams wanting simplicity with core features

#### Consul Connect
- **Data plane**: Envoy or built-in proxy
- **Control plane**: HashiCorp Consul
- **Features**:
  - Integrated service discovery
  - Multi-datacenter support
  - Intention-based security
  - Works with VMs and Kubernetes
- **Use cases**: Hybrid cloud, VM + container environments

### Traffic Management Patterns

#### Canary Deployments
```
Traffic split:
- 95% to v1 (stable)
- 5% to v2 (canary)

Monitor metrics, gradually increase v2 traffic
```

#### Blue-Green Deployments
```
- Blue: Current production version
- Green: New version
- Switch traffic instantly: 100% Blue → 100% Green
- Quick rollback if issues
```

#### A/B Testing
```
Route based on user attributes:
- Premium users → v2 (new features)
- Regular users → v1 (stable)
```

#### Traffic Mirroring (Shadowing)
```
Send production traffic to:
- Primary: v1 (serves responses)
- Shadow: v2 (responses discarded)

Test v2 with real traffic, no user impact
```

### Security Features

#### Mutual TLS (mTLS)
- **Automatic**: Service mesh handles certificate issuance and rotation
- **Strong identity**: Each service has cryptographic identity
- **Encryption**: All service-to-service traffic encrypted
- **Zero-trust**: Verify identity on every request

#### Authorization Policies
```
Example (Istio):
- Allow service A to call service B on /api/data
- Deny all other access to service B
- Require JWT token for external traffic
```

### Observability Integration

**Automatic metrics**:
- Request rate, latency (p50, p95, p99)
- Success rate, error rate
- Connection pools, retries

**Distributed tracing**:
- Automatic span creation
- Context propagation
- Integration with Jaeger, Zipkin

**Topology visualization**:
- Service dependency graphs
- Traffic flow visualization
- Error tracking

### Trade-offs

**Advantages**:
- Uniform traffic management across services
- Security without application changes
- Rich observability out of the box
- Multi-language support

**Disadvantages**:
- **Complexity**: Additional infrastructure to manage
- **Performance**: Proxy adds latency (typically 1-5ms)
- **Resource overhead**: Sidecar per pod increases resource usage
- **Learning curve**: New concepts and tools

## Message Queuing and Stream Processing

### Message Queue Patterns

#### Point-to-Point (Queue)
- **Model**: One message, one consumer
- **Delivery**: Message removed after consumption
- **Use cases**: Task distribution, job queues
- **Example**: Worker pool processing jobs

#### Publish-Subscribe (Topic)
- **Model**: One message, multiple consumers
- **Delivery**: All subscribers receive copy
- **Use cases**: Event broadcasting, notifications
- **Example**: Order placed → notify inventory, shipping, analytics

#### Request-Reply
- **Model**: Synchronous-like communication over async messaging
- **How**: Sender includes reply queue, waits for response
- **Use cases**: RPC over messaging, distributed API calls

### Message Delivery Guarantees

#### At-Most-Once
- **Guarantee**: Message delivered 0 or 1 times
- **Mechanism**: Send and forget, no acknowledgment
- **Use cases**: Metrics, logs (where loss acceptable)
- **Pros**: Highest performance
- **Cons**: Possible message loss

#### At-Least-Once
- **Guarantee**: Message delivered 1 or more times
- **Mechanism**: Retry until acknowledged
- **Use cases**: Most common, when duplicates tolerable
- **Pros**: No message loss
- **Cons**: Possible duplicates
- **Requirement**: Idempotent consumers

#### Exactly-Once
- **Guarantee**: Message processed exactly once
- **Mechanism**: Deduplication + transactional processing
- **Use cases**: Financial transactions, critical updates
- **Complexity**: Hard to achieve, requires coordination
- **Approaches**:
  - Idempotency keys
  - Transactional outbox pattern
  - Two-phase commit
  - Kafka transactions (producer-consumer)

### Apache Kafka - Deep Dive

#### Architecture

**Topics**: Logical channels for messages
- Partitioned for parallelism
- Replicated for fault tolerance

**Partitions**: Ordered, immutable sequence of messages
- Messages appended to end (log)
- Each message has offset (position)
- Distributed across brokers

**Brokers**: Kafka servers storing partitions
- Leader: Handles reads/writes for partition
- Followers: Replicate leader's data

**Producers**: Write messages to topics
- Choose partition (round-robin, key-based, custom)
- Batching for efficiency

**Consumers**: Read messages from topics
- Consumer groups for parallel processing
- Each partition consumed by one consumer in group

**ZooKeeper** (legacy) / **KRaft** (new): Cluster coordination
- Leader election
- Configuration management
- KRaft removes ZooKeeper dependency

#### Key Features

**1. Persistence**
- All messages written to disk
- Retention configurable (time or size-based)
- Enables replay and multiple consumers

**2. Ordering Guarantees**
- Total order within partition
- Key-based partitioning for related messages
- No global ordering across partitions

**3. Scalability**
- Add brokers to scale storage and throughput
- Add partitions to scale parallel processing
- Consumer groups for load distribution

**4. Fault Tolerance**
- Replication factor (typically 3)
- In-sync replicas (ISR) for durability
- Automatic leader election on failure

**5. Performance**
- Sequential disk I/O (fast)
- Zero-copy transfer to consumers
- Batching and compression
- Millions of messages per second

#### Producer Configuration

**Acknowledgment levels**:
- `acks=0`: No acknowledgment (fire and forget)
- `acks=1`: Leader acknowledges (fast, some risk)
- `acks=all`: All in-sync replicas acknowledge (durable, slower)

**Idempotent producer**:
```
enable.idempotence=true
- Prevents duplicate messages on retry
- Maintains ordering per partition
```

**Transactions**:
```
Atomic writes to multiple partitions
Exactly-once semantics (with transactional consumers)
```

#### Consumer Configuration

**Offset management**:
- `auto.offset.reset`: What to do when no offset (earliest, latest)
- `enable.auto.commit`: Automatic vs manual commit
- Manual commit provides more control

**Consumer groups**:
- Partition assignment strategies (range, round-robin, sticky)
- Rebalancing when consumers added/removed

**Exactly-once consumption**:
```
1. Read messages
2. Process messages
3. Save results + commit offsets in transaction
```

#### Use Cases

**1. Event Sourcing**: Kafka as event store
**2. Stream Processing**: Kafka Streams, ksqlDB
**3. Log Aggregation**: Centralized logging
**4. Metrics Collection**: Time-series data
**5. CDC (Change Data Capture)**: Database change events
**6. Microservices Communication**: Event-driven architecture

### RabbitMQ

#### Key Features

**1. Flexible Routing**
- Direct exchange: Route by routing key
- Fanout exchange: Broadcast to all queues
- Topic exchange: Pattern matching (e.g., `orders.*.created`)
- Headers exchange: Route by message headers

**2. Message Priority**: Prioritize urgent messages

**3. Dead Letter Queues**: Failed messages routed to DLQ

**4. Message TTL**: Automatic expiration

**5. Federation**: Connect multiple RabbitMQ clusters

#### vs Kafka

| Feature | Kafka | RabbitMQ |
|---------|-------|----------|
| **Model** | Log-based, persistent | Traditional message queue |
| **Throughput** | Very high (millions/sec) | High (tens of thousands/sec) |
| **Retention** | Long-term (days/weeks) | Short-term (until consumed) |
| **Ordering** | Per-partition | Per-queue (with single consumer) |
| **Replay** | Yes (messages persisted) | No (consumed messages deleted) |
| **Use case** | Event streaming, logs, analytics | Task queues, RPC, complex routing |

### Stream Processing

#### Apache Kafka Streams

**Library**: Embedded in your application (no separate cluster)

**Features**:
- Stateful processing (joins, aggregations, windowing)
- Exactly-once semantics
- Interactive queries (query local state)
- Fault-tolerant state stores (RocksDB)

**Example use cases**:
- Real-time analytics
- Fraud detection
- Anomaly detection
- Stream enrichment

#### Apache Flink

**Framework**: Separate cluster for stream processing

**Features**:
- True streaming (not micro-batching)
- Event time processing (handle late data)
- Exactly-once state consistency
- Complex event processing (CEP)
- SQL support

**Advantages over Spark Streaming**:
- Lower latency (milliseconds vs seconds)
- Better for event-time processing
- Native streaming (not batching)

#### Apache Spark Streaming

**Model**: Micro-batching (Structured Streaming)

**Features**:
- Unified batch and streaming
- Integration with Spark ecosystem (MLlib, SQL)
- Scalable and fault-tolerant

**Use cases**:
- ETL pipelines
- Real-time analytics
- ML model serving

## Cloud-Native Patterns

### Twelve-Factor App Principles

1. **Codebase**: One codebase tracked in version control, many deploys
2. **Dependencies**: Explicitly declare and isolate dependencies
3. **Config**: Store config in environment variables
4. **Backing Services**: Treat backing services as attached resources
5. **Build, Release, Run**: Strictly separate build and run stages
6. **Processes**: Execute app as one or more stateless processes
7. **Port Binding**: Export services via port binding
8. **Concurrency**: Scale out via the process model
9. **Disposability**: Maximize robustness with fast startup and graceful shutdown
10. **Dev/Prod Parity**: Keep development, staging, and production similar
11. **Logs**: Treat logs as event streams
12. **Admin Processes**: Run admin/management tasks as one-off processes

### Container Orchestration with Kubernetes

#### Core Concepts

**Pod**: Smallest deployable unit
- One or more containers
- Shared network namespace
- Shared storage volumes
- Co-located, co-scheduled

**ReplicaSet**: Maintains desired number of pod replicas
- Self-healing (replaces failed pods)
- Scaling (horizontal pod autoscaler)

**Deployment**: Declarative updates for pods
- Rolling updates
- Rollback capability
- Version history

**Service**: Stable network endpoint for pods
- Load balancing across pod replicas
- Service discovery (DNS)
- Types: ClusterIP, NodePort, LoadBalancer

**ConfigMap**: Configuration data
- Decoupled from container images
- Injected as environment variables or volumes

**Secret**: Sensitive data (passwords, tokens)
- Base64 encoded
- Can be encrypted at rest
- Mounted as volumes or env vars

#### Kubernetes Patterns

##### 1. Sidecar Pattern
```
Pod:
- Main container: Application
- Sidecar: Log collector, metrics exporter, proxy
```
**Examples**: Istio proxy, Fluentd log shipper, Consul agent

##### 2. Ambassador Pattern
```
Sidecar acts as proxy for external services
- Main container: Connects to localhost
- Ambassador: Handles connection pooling, retry logic, circuit breaking
```

##### 3. Adapter Pattern
```
Sidecar standardizes and normalizes output
- Main container: Legacy app with custom log format
- Adapter: Converts logs to standard format
```

##### 4. Init Container
```
Runs before main containers
- Setup tasks (download files, wait for dependencies)
- Security (set permissions, scan for vulnerabilities)
```

##### 5. Jobs and CronJobs
```
Job: Run to completion (batch processing)
CronJob: Scheduled execution (backups, reports)
```

#### Scaling Patterns

**Horizontal Pod Autoscaler (HPA)**:
- Scale based on CPU, memory, or custom metrics
- Min/max replica bounds
```
If CPU > 80%: Add pods
If CPU < 20%: Remove pods
```

**Vertical Pod Autoscaler (VPA)**:
- Adjust resource requests/limits
- Rightsizing for efficiency

**Cluster Autoscaler**:
- Add/remove nodes based on pod resource requests
- Integrates with cloud providers (AWS, GCP, Azure)

#### Advanced Scheduling

**Node Affinity**: Schedule pods on specific nodes
```
Example: GPU workloads on GPU nodes
```

**Pod Affinity/Anti-Affinity**: Co-locate or separate pods
```
Anti-affinity: Spread replicas across availability zones
Affinity: Place cache pods near compute pods
```

**Taints and Tolerations**: Prevent pods from scheduling on certain nodes
```
Taint node for dedicated workloads
Only pods with matching toleration can schedule
```

## Security in Distributed Systems

### Authentication

#### Mutual TLS (mTLS)
- Both client and server present certificates
- Cryptographic identity verification
- Prevents impersonation
- **Implementation**: Service mesh, application-level

#### OAuth 2.0 / OpenID Connect
- **OAuth 2.0**: Authorization framework
- **OpenID Connect**: Authentication layer on OAuth 2.0
- **Flow**: Client → Authorization Server → Resource Server
- **Tokens**: Access tokens, refresh tokens, ID tokens (JWT)
- **Use cases**: User authentication, API authorization

#### JSON Web Tokens (JWT)
- **Structure**: Header.Payload.Signature
- **Stateless**: No server-side session storage
- **Claims**: User info, permissions, expiration
- **Verification**: Signature validates authenticity
- **Challenges**: Token revocation (use short expiration + refresh tokens)

### Authorization

#### Role-Based Access Control (RBAC)
- Users assigned to roles
- Roles have permissions
- Check user's roles for access decision
- **Example**: Admin, Editor, Viewer

#### Attribute-Based Access Control (ABAC)
- Policies based on attributes (user, resource, environment)
- More flexible than RBAC
- **Example**: "Allow if user.department == resource.department AND time.hour >= 9 AND time.hour <= 17"

#### Policy Engines
- **Open Policy Agent (OPA)**: General-purpose policy engine
  - Rego policy language
  - Decoupled authorization
  - Used in Kubernetes, microservices
- **Casbin**: Authorization library
  - Multiple models (ACL, RBAC, ABAC)
  - Multiple languages

### Data Security

#### Encryption at Rest
- Encrypt data stored on disk
- **Methods**: Full-disk encryption, database-level encryption, application-level
- **Key management**: KMS (AWS KMS, Google Cloud KMS, Azure Key Vault)

#### Encryption in Transit
- TLS/SSL for network communication
- Certificate management (Let's Encrypt, cert-manager)
- Perfect Forward Secrecy (PFS)

#### Secrets Management
- **HashiCorp Vault**: Dynamic secrets, encryption as a service, lease management
- **AWS Secrets Manager**: Rotation, access control
- **Kubernetes Secrets**: Base64 encoding, encryption with KMS
- **Sealed Secrets**: Encrypted secrets in Git (GitOps)

### Network Security

#### Zero Trust Architecture
- **Principle**: Never trust, always verify
- **Implementation**:
  - Verify every request (even internal)
  - Micro-segmentation
  - Least privilege access
  - Continuous monitoring

#### Network Policies (Kubernetes)
- Control traffic flow between pods
- Default deny, explicit allow
```
Example:
- Allow frontend pods to call backend pods on port 8080
- Deny all other traffic to backend
```

#### API Gateway Security
- **Rate limiting**: Prevent abuse
- **Authentication**: Verify client identity
- **Authorization**: Check permissions
- **Input validation**: Prevent injection attacks
- **DDoS protection**: Throttling, IP blocking

### Security Best Practices

1. **Principle of Least Privilege**: Minimal permissions necessary
2. **Defense in Depth**: Multiple layers of security
3. **Secure by Default**: Security enabled out of the box
4. **Immutable Infrastructure**: Replace, don't patch
5. **Audit Logging**: Track all access and changes
6. **Vulnerability Scanning**: Regular image and dependency scans
7. **Secret Rotation**: Regularly rotate credentials
8. **Network Segmentation**: Isolate services and data
9. **Input Validation**: Sanitize all inputs
10. **Security Testing**: Penetration testing, chaos engineering

## Disaster Recovery and Multi-Region

### Recovery Objectives

**RTO (Recovery Time Objective)**: Maximum acceptable downtime
- Example: RTO = 1 hour (system must be restored within 1 hour)

**RPO (Recovery Point Objective)**: Maximum acceptable data loss
- Example: RPO = 15 minutes (can lose max 15 minutes of data)

### Multi-Region Architectures

#### Active-Passive (Disaster Recovery)

**Setup**:
- **Active region**: Serves all traffic
- **Passive region**: Standby, ready to take over
- Data replication: Active → Passive

**Failover**:
1. Detect failure in active region
2. Promote passive region to active
3. Redirect traffic (DNS update, load balancer)
4. RPO: Replication lag (seconds to minutes)
5. RTO: Failover time (minutes to hours)

**Use cases**: Cost-conscious DR, acceptable downtime

#### Active-Active (Multi-Region)

**Setup**:
- Multiple regions serve traffic simultaneously
- Data replicated between regions
- Global load balancer distributes traffic

**Advantages**:
- Lower latency (users routed to nearest region)
- Higher availability (region failure transparent)
- Better resource utilization

**Challenges**:
- Data consistency (cross-region writes)
- Conflict resolution
- Increased cost

**Patterns**:

**1. Read-Local, Write-Global**:
- Reads from nearest region
- Writes to primary region, replicated globally
- **Trade-off**: Write latency, but consistent

**2. Write-Local, Async Replication**:
- Writes to local region, async replication
- **Trade-off**: Low latency, eventual consistency, conflicts

**3. Multi-Master with CRDT**:
- Writes to any region
- CRDTs ensure convergence
- **Trade-off**: Complex, but no conflicts

### Database Replication Strategies

#### Cross-Region Replication

**Synchronous**:
- Wait for remote region acknowledgment
- **Pros**: No data loss (RPO = 0)
- **Cons**: High latency (limited by speed of light)

**Asynchronous**:
- Replicate in background
- **Pros**: Low latency
- **Cons**: Data loss on failure (RPO > 0)

**Semi-synchronous**:
- Wait for one local replica, async to remote
- **Balance**: Durability + performance

#### Global Databases

**Google Spanner**:
- Globally distributed, strongly consistent
- TrueTime for global ordering
- Multi-region ACID transactions

**CockroachDB**:
- Distributed SQL, Spanner-inspired
- Raft consensus per range
- Geo-partitioning for data locality

**AWS Aurora Global Database**:
- Primary region + up to 5 secondary regions
- < 1 second replication lag
- Cross-region failover

**DynamoDB Global Tables**:
- Multi-region, multi-master
- Last-write-wins conflict resolution
- Active-active replication

### Backup Strategies

#### Backup Types

**Full Backup**: Complete copy of all data
- **Pros**: Simple restore
- **Cons**: Large storage, slow

**Incremental Backup**: Only changes since last backup
- **Pros**: Fast, efficient storage
- **Cons**: Complex restore (need full + all incrementals)

**Differential Backup**: Changes since last full backup
- **Pros**: Faster restore than incremental
- **Cons**: Larger than incremental

#### Backup Best Practices

1. **3-2-1 Rule**: 3 copies, 2 different media, 1 offsite
2. **Automated Backups**: Scheduled, no manual intervention
3. **Test Restores**: Regularly verify backups work
4. **Encryption**: Encrypt backups at rest and in transit
5. **Retention Policy**: Balance cost and compliance
6. **Immutable Backups**: Prevent ransomware deletion
7. **Cross-Region**: Store backups in different region

### Chaos Engineering

**Purpose**: Proactively find weaknesses before they cause outages

#### Principles
1. **Define steady state**: Normal behavior metrics
2. **Hypothesize**: Predict impact of failure
3. **Inject failure**: Controlled experiments
4. **Observe**: Monitor impact on steady state
5. **Learn and improve**: Fix weaknesses

#### Failure Scenarios
- **Network**: Latency injection, packet loss, partition
- **Compute**: Kill instances, CPU/memory pressure
- **Storage**: Disk failures, corruption
- **Dependencies**: Service failures, degraded performance

#### Tools
- **Chaos Monkey** (Netflix): Randomly kills instances
- **Chaos Toolkit**: Generic chaos engineering platform
- **Gremlin**: Chaos engineering as a service
- **Litmus** (Kubernetes): Chaos experiments for K8s

## Edge Computing and CDN

### Edge Computing

**Definition**: Computation and data storage closer to users/devices

#### Use Cases

**1. Low Latency Applications**:
- Gaming (real-time multiplayer)
- AR/VR (motion-to-photon latency)
- Video streaming (adaptive bitrate)

**2. Bandwidth Optimization**:
- Process data locally, send only results
- IoT devices (process sensor data at edge)

**3. Privacy and Compliance**:
- Keep data within geographic boundaries
- Process sensitive data locally

**4. Offline Capability**:
- Continue operation without cloud connectivity
- Sync when connection restored

#### Edge Architectures

**1. CDN with Edge Computing** (Cloudflare Workers, AWS Lambda@Edge):
- Run code at CDN edge locations
- Modify requests/responses
- A/B testing, personalization, auth

**2. Mobile Edge Computing (MEC)**:
- Compute at cellular network edge (5G)
- Ultra-low latency (<10ms)
- Use cases: Autonomous vehicles, smart cities

**3. IoT Edge**:
- Gateways aggregate and process IoT data
- Machine learning inference at edge
- Examples: AWS IoT Greengrass, Azure IoT Edge

### Content Delivery Networks (CDN)

#### How CDNs Work

1. **Origin server**: Original content source
2. **Edge servers**: Cached content near users
3. **Request flow**:
   - User requests content
   - DNS routes to nearest edge server
   - Edge server serves from cache (cache hit)
   - Or fetches from origin (cache miss), caches, serves

#### Cache Strategies

**1. Cache-Control Headers**:
```
Cache-Control: public, max-age=3600
- Public: Can be cached by CDN
- max-age: Cache for 1 hour
```

**2. Cache Invalidation**:
- **Purge**: Remove from all edge servers
- **Time-based**: Expire after TTL
- **Version-based**: Include version in URL (e.g., /app.v2.js)

**3. Cache Key**:
- Default: URL
- Custom: URL + headers (User-Agent, Accept-Language)

#### CDN Features

**1. Geographic Distribution**: Servers in multiple regions
**2. DDoS Protection**: Absorb attack traffic
**3. SSL/TLS Termination**: Offload encryption from origin
**4. Compression**: Gzip, Brotli
**5. Image Optimization**: Resize, format conversion (WebP)
**6. Streaming**: HLS, DASH for video

#### Popular CDNs

- **Cloudflare**: Global network, DDoS protection, Workers (edge compute)
- **Akamai**: Largest CDN, enterprise focus
- **Fastly**: Real-time purging, edge compute (Compute@Edge)
- **AWS CloudFront**: Integrated with AWS, Lambda@Edge
- **Google Cloud CDN**: Integrated with GCP

## Serverless Architectures

### Function as a Service (FaaS)

#### Characteristics

1. **Event-driven**: Functions triggered by events
2. **Stateless**: No persistent state between invocations
3. **Ephemeral**: Short-lived execution (seconds to minutes)
4. **Auto-scaling**: Scale to zero, scale to thousands
5. **Pay-per-use**: Charged for execution time, not idle time

#### Popular FaaS Platforms

**AWS Lambda**:
- Multiple runtimes (Node.js, Python, Java, Go, .NET, Ruby)
- 15-minute max execution
- Event sources: S3, DynamoDB, API Gateway, SQS, etc.
- Provisioned concurrency for low latency

**Google Cloud Functions**:
- HTTP and event-driven
- Auto-scaling
- Integration with GCP services

**Azure Functions**:
- Multiple triggers (HTTP, timer, queue, blob)
- Durable Functions (stateful workflows)
- Integration with Azure services

**Cloudflare Workers**:
- Edge compute (runs at CDN edge)
- V8 isolates (not containers)
- Sub-millisecond startup
- JavaScript/WebAssembly

#### Serverless Patterns

##### 1. API Backend
```
API Gateway → Lambda → DynamoDB
- Scalable REST API
- No server management
```

##### 2. Stream Processing
```
Kinesis/Kafka → Lambda → S3/Database
- Real-time data processing
- Auto-scaling with stream shards
```

##### 3. Scheduled Jobs
```
CloudWatch Events (cron) → Lambda
- Periodic tasks (cleanup, reports)
```

##### 4. File Processing
```
S3 upload → Lambda (resize image) → S3
- Event-driven processing
```

##### 5. Webhooks
```
External service → API Gateway → Lambda
- Handle incoming webhooks
```

### Serverless Databases

**AWS DynamoDB**:
- Serverless NoSQL
- On-demand or provisioned capacity
- Auto-scaling

**Google Firestore**:
- Serverless document database
- Real-time synchronization
- Offline support

**Azure Cosmos DB** (serverless):
- Multi-model database
- Global distribution
- Multiple consistency levels

**FaunaDB**:
- Serverless transactional database
- GraphQL, FQL query languages
- Multi-region, ACID

### Serverless Challenges

#### 1. Cold Starts
- **Problem**: First invocation slow (100ms-10s)
- **Solutions**:
  - Provisioned concurrency (keep warm)
  - Minimize function size
  - Use faster runtimes (Go, Rust)
  - Edge compute (Cloudflare Workers)

#### 2. Statelessness
- **Problem**: No persistent memory between invocations
- **Solutions**:
  - External state stores (Redis, DynamoDB)
  - Step Functions for workflows
  - Durable Functions (Azure)

#### 3. Vendor Lock-in
- **Problem**: Tied to specific cloud provider
- **Solutions**:
  - Abstraction layers (Serverless Framework)
  - Multi-cloud deployment
  - Containers (Cloud Run, Fargate)

#### 4. Debugging and Monitoring
- **Problem**: Distributed, ephemeral environment
- **Solutions**:
  - Distributed tracing (AWS X-Ray, Datadog)
  - Structured logging
  - Local emulators (SAM, LocalStack)

#### 5. Timeouts and Limits
- **Problem**: Execution time limits (e.g., 15 min for Lambda)
- **Solutions**:
  - Break into smaller functions
  - Use Step Functions for orchestration
  - Hybrid approach (long tasks on containers)

### Serverless vs Containers

| Aspect | Serverless (FaaS) | Containers |
|--------|-------------------|------------|
| **Abstraction** | High (no infra) | Medium (manage containers) |
| **Scaling** | Automatic, instant | Auto-scaling with delay |
| **Cold start** | Yes (100ms-10s) | Minimal (if running) |
| **Cost** | Pay per execution | Pay for running time |
| **State** | Stateless | Can be stateful |
| **Execution limit** | 15 min (Lambda) | No limit |
| **Flexibility** | Limited runtimes | Any language/runtime |
| **Best for** | Event-driven, bursty | Long-running, stateful |

## GraphQL Federation

### Overview

**GraphQL Federation**: Compose multiple GraphQL services into single unified graph

#### Traditional Approach
```
Single GraphQL server
- Monolithic schema
- All resolvers in one codebase
- Doesn't scale for large teams
```

#### Federated Approach
```
Multiple GraphQL services (subgraphs)
- Each owns part of schema
- Gateway composes and routes queries
- Teams work independently
```

### Architecture

**Subgraphs**: Individual GraphQL services
- Own domain-specific types and fields
- Extend types from other subgraphs
- Independent deployment

**Gateway**: Composes and executes federated queries
- Schema composition
- Query planning
- Request routing

### Apollo Federation

#### Key Concepts

**1. Entities**: Types shared across subgraphs
```graphql
# Products subgraph
type Product @key(fields: "id") {
  id: ID!
  name: String!
  price: Float!
}

# Reviews subgraph (extends Product)
extend type Product @key(fields: "id") {
  id: ID! @external
  reviews: [Review!]!
}
```

**2. @key Directive**: Identifies entity
- Tells gateway how to uniquely identify object
- Enables cross-service joins

**3. @external**: Field defined in another subgraph

**4. @requires**: Field requires other fields to resolve

**5. @provides**: Field can provide additional fields

#### Query Planning

**Example**:
```graphql
query {
  product(id: "123") {
    name       # Products subgraph
    price      # Products subgraph
    reviews {  # Reviews subgraph
      rating
      comment
    }
  }
}
```

**Execution**:
1. Gateway queries Products subgraph for product(id: "123")
2. Returns: { id: "123", name: "Widget", price: 29.99, __typename: "Product" }
3. Gateway queries Reviews subgraph with Product entity reference
4. Returns reviews
5. Gateway merges results

### Schema Stitching vs Federation

| Aspect | Schema Stitching | Federation |
|--------|------------------|------------|
| **Ownership** | Gateway owns schema | Subgraphs own schema |
| **Composition** | Manual stitching | Automatic composition |
| **Type extension** | Limited | Native support |
| **Performance** | More round trips | Optimized query plans |
| **Best for** | Combining 3rd party APIs | Microservices architecture |

### Benefits

1. **Team Autonomy**: Teams own their subgraphs
2. **Independent Deployment**: Deploy subgraphs separately
3. **Incremental Adoption**: Gradually migrate to federation
4. **Type Safety**: Shared types across services
5. **Unified API**: Single GraphQL endpoint for clients

### Challenges

1. **Complexity**: More moving parts
2. **Debugging**: Distributed query execution
3. **Schema Coordination**: Avoid breaking changes
4. **Gateway Performance**: Single point of failure

## Best Practices

1. **Design for Failure**: Assume everything will fail
2. **Loose Coupling**: Services should be independent
3. **Idempotency**: Make operations safe to retry
4. **Asynchronous Communication**: Use message queues when possible
5. **Graceful Degradation**: Partial functionality over complete failure
6. **Monitoring and Alerting**: Comprehensive observability
7. **Automation**: Auto-scaling, self-healing systems
8. **Testing**: Chaos engineering, fault injection
9. **Documentation**: Clear service contracts and APIs
10. **Security**: Authentication, authorization, encryption
11. **Backward Compatibility**: Versioning, graceful upgrades
12. **Distributed Tracing**: Track requests across services
13. **Bulkheads**: Isolate failures
14. **Rate Limiting**: Protect from overload
15. **Caching**: Reduce load, improve performance
16. **Immutable Infrastructure**: Treat servers as disposable
17. **Infrastructure as Code**: Version control infra changes
18. **Zero Trust Security**: Never trust, always verify
19. **Multi-Region**: Plan for regional failures
20. **Cost Optimization**: Right-size resources, use spot instances

## Observability

### The Three Pillars

#### 1. Metrics
- **Definition**: Numeric measurements over time
- **Examples**: Request rate, error rate, latency, CPU usage
- **Tools**: Prometheus, Grafana, Datadog, CloudWatch
- **Patterns**: RED (Rate, Errors, Duration), USE (Utilization, Saturation, Errors)

#### 2. Logs
- **Definition**: Discrete event records
- **Structure**: Structured (JSON) vs unstructured (text)
- **Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), Splunk, Loki
- **Best practices**: Include correlation IDs, timestamps, context

#### 3. Traces
- **Definition**: End-to-end request path through system
- **Components**: Spans (single operation), traces (collection of spans)
- **Tools**: Jaeger, Zipkin, Datadog APM, AWS X-Ray
- **Context propagation**: Trace ID passed in headers

### Key Metrics

- **Latency**: Time to process requests (p50, p95, p99)
- **Throughput**: Requests per second
- **Error Rate**: Failed requests percentage
- **Saturation**: Resource utilization (CPU, memory, disk, network)
- **Availability**: Uptime percentage (SLA)

## Further Reading

### Books
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Distributed Systems" by Maarten van Steen and Andrew S. Tanenbaum
- "Building Microservices" by Sam Newman
- "Release It!" by Michael Nygard
- "Site Reliability Engineering" by Google

### Papers
- **Consensus**: "Paxos Made Simple" (Lamport), "In Search of an Understandable Consensus Algorithm" (Raft)
- **Storage**: "Bigtable: A Distributed Storage System for Structured Data" (Google), "Dynamo: Amazon's Highly Available Key-value Store" (Amazon)
- **Databases**: "Spanner: Google's Globally-Distributed Database", "TAO: Facebook's Distributed Data Store for the Social Graph"
- **Theory**: "CAP Twelve Years Later: How the Rules Have Changed" (Brewer), "Impossibility of Distributed Consensus with One Faulty Process" (FLP)
- **Time**: "Time, Clocks, and the Ordering of Events" (Lamport)

### Online Resources
- AWS Architecture Blog
- Google Cloud Architecture Center
- Martin Fowler's blog
- The Morning Paper (paper summaries)
- Papers We Love

## Common Trade-offs

| Aspect | Trade-off |
|--------|-----------|
| Consistency vs Availability | Stronger consistency reduces availability during partitions |
| Latency vs Consistency | Lower latency may sacrifice consistency |
| Complexity vs Performance | More complex systems may be more performant but harder to operate |
| Cost vs Reliability | Higher reliability requires more resources (replication, redundancy) |
| Scalability vs Simplicity | Horizontal scaling increases complexity |
| Strong Consistency vs Throughput | Coordination for consistency reduces throughput |
| Normalization vs Denormalization | Normalized reduces storage, denormalized improves read performance |
| Sync vs Async | Synchronous simpler but couples services, async more complex but decouples |
| Monolith vs Microservices | Monolith simpler initially, microservices better for scale and teams |

---

*Note: Distributed systems require careful consideration of requirements, constraints, and trade-offs. There is no one-size-fits-all solution. Choose the right tool and pattern for your specific use case.*

