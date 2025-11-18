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

