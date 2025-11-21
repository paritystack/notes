# Raft Consensus Algorithm

Raft is a consensus algorithm designed to be easy to understand. It's used for managing a replicated log in distributed systems.

## Overview

Raft ensures that a cluster of servers agrees on a sequence of values, even in the presence of failures.

**Key Properties:**
- Leader election
- Log replication
- Safety
- Membership changes

## Quick Reference

### State Variables

**Persistent state (must survive crashes):**
- `currentTerm`: Latest term server has seen (initialized to 0)
- `votedFor`: CandidateId that received vote in current term (null if none)
- `log[]`: Log entries; each entry contains command and term when entry was received by leader

**Volatile state (all servers):**
- `commitIndex`: Index of highest log entry known to be committed (initialized to 0)
- `lastApplied`: Index of highest log entry applied to state machine (initialized to 0)

**Volatile state (leaders only, reinitialized after election):**
- `nextIndex[]`: For each server, index of next log entry to send (initialized to leader's last log index + 1)
- `matchIndex[]`: For each server, index of highest log entry known to be replicated (initialized to 0)

### Key Parameters

- **Election timeout**: 150-300ms (randomized)
- **Heartbeat interval**: < 150ms (typically 50ms)
- **RPC timeout**: Varies by implementation (typically 100-500ms)
- **Minimum election timeout**: Should be >> network round-trip time

### Core Guarantees

- **At most one leader per term**
- **Leader never deletes/overwrites log entries**
- **If two logs contain entry with same index and term, all preceding entries are identical**
- **Committed entries are durable and will eventually be executed by all state machines**

## Server States

```
┌─────────┐    times out, starts election    ┌───────────┐
│Follower │───────────────────────────────>│ Candidate │
└─────────┘                                  └───────────┘
     │                                            │
     │discovers current leader or new term         │receives votes from
     │                                            │majority of servers
     │                                            │
     │                                            ▼
     │                                       ┌────────┐
     └───────────────────────────────────────│ Leader │
                 discovers server with       └────────┘
                 higher term
```

### Timing and Network Assumptions

Raft's correctness depends on safety properties that hold regardless of timing, but **liveness** (making progress) requires specific timing constraints:

**Key timing requirement:**
```
broadcastTime ≪ electionTimeout ≪ MTBF
```

Where:
- **broadcastTime**: Average time for server to send RPCs to every server and receive responses (0.5ms to 20ms)
- **electionTimeout**: Time before follower becomes candidate (typically 150-300ms)
- **MTBF**: Mean Time Between Failures for single server (typically months or years)

**Why these relationships matter:**

1. **broadcastTime ≪ electionTimeout**
   - Leader must send heartbeats reliably before followers timeout
   - Prevents unnecessary elections during normal operation
   - Typical: broadcastTime = 10ms, electionTimeout = 200ms (20x margin)

2. **electionTimeout ≪ MTBF**
   - Cluster remains available even when leader crashes
   - Brief unavailability during election acceptable
   - System makes progress most of the time

**Network assumptions:**
- Messages can be lost, reordered, or duplicated
- Messages are not corrupted (or corruption is detected via checksums)
- Network partitions can occur (split brain scenarios)
- Eventually, messages will be delivered (asynchronous model)

**Handling network partitions:**
- Minority partition cannot commit entries (lacks majority)
- Majority partition continues operating normally
- When partition heals, minority rejoins and updates its log
- Old leader in minority steps down when it discovers higher term

## Leader Election

### Election Process

1. **Follower times out** (randomized 150-300ms)
   - Each server has randomized election timeout to prevent split votes
   - Timeout resets on receiving AppendEntries from valid leader
2. **Becomes candidate**, increments current term
   - Transitions from follower → candidate state
   - Increments `currentTerm` by 1
3. **Votes for itself**
   - Sets `votedFor` to its own server ID
4. **Requests votes** from other servers (RequestVote RPC)
5. **Election outcomes:**
   - **Wins**: Receives votes from majority → becomes leader
   - **Loses**: Receives AppendEntries from valid leader → becomes follower
   - **Times out**: No winner (split vote) → increment term, retry

### RequestVote RPC

**Arguments:**
- `term`: Candidate's term number
- `candidateId`: Candidate requesting vote
- `lastLogIndex`: Index of candidate's last log entry
- `lastLogTerm`: Term of candidate's last log entry

**Response:**
- `term`: Current term (for candidate to update itself)
- `voteGranted`: true if candidate received vote

**Receiver Implementation:**
1. Reply false if `term < currentTerm`
2. If `votedFor` is null or candidateId, and candidate's log is at least as up-to-date as receiver's log, grant vote

**Up-to-date log check:**
- If logs have different term numbers for last entry, log with later term is more up-to-date
- If logs end with same term, whichever log is longer is more up-to-date

### Split Vote Handling

When no candidate receives majority:
- Multiple candidates split votes
- Each candidate times out independently
- Randomized timeouts (150-300ms) make it unlikely same split occurs twice
- Failed candidates increment term and retry election

## Log Replication

### Replication Flow

```
Leader receives command from client
  ↓
Append to local log
  ↓
Send AppendEntries RPCs to followers
  ↓
Wait for majority to acknowledge
  ↓
Apply to state machine
  ↓
Return result to client
```

### AppendEntries RPC

Used for log replication and as heartbeat (empty entries).

**Arguments:**
- `term`: Leader's term number
- `leaderId`: So followers can redirect clients
- `prevLogIndex`: Index of log entry immediately preceding new ones
- `prevLogTerm`: Term of prevLogIndex entry
- `entries[]`: Log entries to store (empty for heartbeat)
- `leaderCommit`: Leader's commitIndex

**Response:**
- `term`: Current term (for leader to update itself)
- `success`: true if follower contained entry matching prevLogIndex and prevLogTerm

**Receiver Implementation:**
1. Reply false if `term < currentTerm`
2. Reply false if log doesn't contain entry at `prevLogIndex` whose term matches `prevLogTerm` (consistency check)
3. If existing entry conflicts with new one (same index, different terms), delete existing entry and all that follow
4. Append any new entries not already in log
5. If `leaderCommit > commitIndex`, set `commitIndex = min(leaderCommit, index of last new entry)`

### Consistency Checking

The **Log Matching Property** ensures:
- If two entries in different logs have same index and term, they store same command
- If two entries in different logs have same index and term, logs are identical in all preceding entries

**How it's enforced:**
1. Leader sends `prevLogIndex` and `prevLogTerm` with each AppendEntries
2. Follower checks if it has entry at `prevLogIndex` with term `prevLogTerm`
3. If check fails, follower rejects AppendEntries
4. Leader decrements `nextIndex` for that follower and retries
5. Eventually finds point where logs match, then follower's log matches leader's from that point forward

### Log State Examples

**Example 1: Normal replication (all followers in sync)**
```
Leader (S1):  [1,x] [1,y] [2,z] [3,a] [3,b]  (term 3)
Follower (S2):[1,x] [1,y] [2,z] [3,a] [3,b]  commitIndex=3
Follower (S3):[1,x] [1,y] [2,z] [3,a]        commitIndex=3
              ─────────────────committed─────
```
- Entry [3,b] is replicated to S1 and S2 (majority)
- Once S3 acknowledges, [3,b] can be committed

**Example 2: Follower lagging behind leader**
```
Leader (S1):  [1,x] [1,y] [2,z] [3,a] [3,b] [3,c]  (term 3)
Follower (S2):[1,x] [1,y] [2,z]
              ─────────────committed──────

Leader sends: AppendEntries(prevLogIndex=3, prevLogTerm=2, entries=[[3,a],[3,b],[3,c]])
S2 has entry at index 3 with term 2 → accepts and appends [3,a], [3,b], [3,c]
```

**Example 3: Conflicting entries (follower has extra uncommitted entries)**
```
Leader (S1):  [1,x] [1,y] [3,z] [3,a]  (term 3, was follower in term 2)
Follower (S2):[1,x] [1,y] [2,z] [2,a] [2,b]  (was following old leader in term 2)
              ─────────────committed─────

Leader sends: AppendEntries(prevLogIndex=2, prevLogTerm=1, entries=[[3,z],[3,a]])
S2 receives [3,z] at index 3, but has [2,z] → conflict!
S2 deletes [2,z], [2,a], [2,b] and appends [3,z], [3,a]

Result:
Follower (S2):[1,x] [1,y] [3,z] [3,a]  ✓ now matches leader
```

**Example 4: Follower missing entries in middle of log**
```
Leader (S1):  [1,x] [1,y] [1,z] [2,a] [2,b] [3,c]  (term 3)
Follower (S2):[1,x] [1,y]
              ─────────────committed──────

Leader sends: AppendEntries(prevLogIndex=6, prevLogTerm=3, entries=[])  # heartbeat
S2 doesn't have entry at index 6 → rejects

Leader decrements nextIndex[S2] = 5
Leader sends: AppendEntries(prevLogIndex=5, prevLogTerm=2, entries=[[3,c]])
S2 doesn't have entry at index 5 → rejects

(continues decrementing until...)

Leader sends: AppendEntries(prevLogIndex=2, prevLogTerm=1, entries=[[1,z],[2,a],[2,b],[3,c]])
S2 has entry at index 2 with term 1 → accepts and appends all entries
```

### Conflict Resolution

When follower's log conflicts with leader's:
1. **Leader never overwrites** its own log entries
2. **Follower's conflicting entries are deleted**
3. Leader maintains `nextIndex[]` for each follower (initially set to leader's last log index + 1)
4. On AppendEntries rejection:
   - Leader decrements `nextIndex` for that follower
   - Retries AppendEntries with earlier entries
5. When AppendEntries succeeds:
   - Follower's log now matches leader's up to that point
   - Leader updates `nextIndex` and `matchIndex` for follower

### Commitment Rules

An entry is **committed** when:
1. Leader has stored it on majority of servers
2. At least one entry from current term is also stored on majority (prevents committing entries from previous terms directly)

**Why rule #2 matters: The commitment restriction**

Raft never commits log entries from previous terms by counting replicas. Only entries from the leader's current term can be committed by counting replicas. Once an entry from the current term is committed, all prior entries are committed indirectly (due to Log Matching Property).

**Example scenario showing why this is necessary:**

```
Time  S1 (Leader)  S2          S3          S4          S5
────────────────────────────────────────────────────────────
t0    [1,x] [2,y]  [1,x] [2,y]  [1,x]       -           -
      Leader T2

t1    [1,x] [2,y]  [1,x] [2,y]  [1,x] [2,y] -           -
      Crashed!

t2    [1,x] [2,y]  [1,x] [2,y]  [1,x] [2,y] [1,x]       [1,x]
                                 Leader T3

t3    [1,x] [2,y]  [1,x] [2,y]  [1,x] [2,y] [1,x] [3,z] [1,x] [3,z]
                                 [3,z]

Without the restriction: S3 could commit [2,y] in term 3 because it's on majority (S1,S2,S3)
But if S3 crashes and S5 becomes leader in term 4, [2,y] would be overwritten!

With restriction: S3 must commit [3,z] first, which commits [2,y] indirectly
If S3 crashes before committing [3,z], S5 can become leader and [2,y] is safely lost
```

**Commitment process:**
1. Leader tracks highest committed entry in `commitIndex`
2. Once entry committed, leader applies it to state machine
3. Leader includes `commitIndex` in AppendEntries RPCs
4. Followers apply committed entries to their state machines
5. Entries are applied in log order to ensure state machine consistency

**Formal commitment rule:**
```
commitIndex = max index N where:
  - N ≤ leader's last log index
  - matchIndex[i] ≥ N for majority of i
  - log[N].term == currentTerm
```

## Safety Rules

Raft guarantees the following properties hold at all times:

### 1. Election Safety
**Property:** At most one leader can be elected in a given term.

**How it's enforced:**
- Each server votes for at most one candidate per term
- Server stores `votedFor` and persists it to stable storage
- Candidate needs majority of votes to win
- Two different candidates cannot both get majority in same term

### 2. Leader Append-Only
**Property:** A leader never overwrites or deletes entries in its log; it only appends new entries.

**Why it matters:**
- Simplifies reasoning about log consistency
- Once leader commits entry, it remains in leader's log forever
- Leader's log is always "truth" for its term

### 3. Log Matching Property
**Property:** If two logs contain an entry with the same index and term, then:
- The logs are identical in all entries up through that index
- The entries store the same command

**How it's enforced:**
- Leader creates at most one entry per log index in a given term
- Log entries never change position
- AppendEntries consistency check verifies log matching before accepting new entries

**Implications:**
- When AppendEntries returns success, leader knows follower's log is identical to its own through new entries
- Transitive property: if A matches B and B matches C, then A matches C

### 4. Leader Completeness Property
**Property:** If a log entry is committed in a given term, then that entry will be present in the logs of leaders for all higher-numbered terms.

**How it's enforced:**
- Voting restriction: candidate cannot win election unless its log contains all committed entries
- RequestVote RPC includes `lastLogIndex` and `lastLogTerm`
- Voter denies vote if its own log is "more up-to-date" than candidate's
- "More up-to-date" defined as: later term number or same term but longer log

**Why it matters:**
- Leaders never need to look at previous terms to determine which entries are committed
- Committed entries flow forward through leaders
- Ensures linearizable consistency

### 5. State Machine Safety
**Property:** If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index.

**How it's enforced:**
- Servers only apply committed entries
- Leader Completeness ensures committed entries present in all future leaders
- Log Matching ensures all servers apply same sequence of commands
- Entries applied to state machine in log order

**Result:** All servers execute same sequence of commands in same order, maintaining identical state machines (assuming deterministic commands).

### Key Invariants

Throughout normal operation, Raft maintains:
- **Leader has most complete log**: Among servers in its term, leader's committed entries form superset
- **Committed entries are durable**: Once committed, entry present in majority; will survive into future leaders
- **Applied entries are consistent**: All servers apply same entries at each index
- **Terms increase monotonically**: Servers never decrease currentTerm

### Formal Safety Proof Sketch

**Theorem: State Machine Safety holds**

Proof by contradiction:
1. Assume State Machine Safety is violated
2. Then some server applies entry at index i, another applies different entry at same index i
3. Let T be smallest term where this occurs
4. Both entries must have been committed (servers only apply committed entries)
5. By Leader Completeness, if entry committed in term ≤ T, it appears in leader's log for term T
6. By Election Safety, only one leader in term T
7. By Leader Append-Only, leader has single log
8. By Log Matching, if two entries have same index and term, they're identical
9. Therefore, both servers must have applied same entry at index i
10. Contradiction! State Machine Safety cannot be violated.

**Key lemma: Leader Completeness**

If entry is committed in term T, then present in leader's log for all terms > T.

Proof intuition:
- Entry committed → replicated to majority in term T
- Candidate needs majority to win election in term T+1
- These majorities must overlap (pigeonhole principle)
- Voting restriction ensures candidate's log has all committed entries from servers that vote for it
- By induction, committed entries flow forward through all future leaders

## Performance Optimizations

While Raft prioritizes understandability, several optimizations improve performance in production systems:

### 1. Batching and Pipelining

**Batching:**
- Accumulate multiple client requests before creating log entries
- Send multiple log entries in single AppendEntries RPC
- Reduces RPC overhead and improves throughput
- Trade-off: Slight increase in latency vs. much higher throughput

**Pipelining:**
- Don't wait for AppendEntries response before sending next batch
- Leader maintains window of in-flight AppendEntries RPCs
- Significantly improves throughput on high-latency networks
- Must track responses to update nextIndex/matchIndex correctly

### 2. Fast Log Backtracking

**Problem:** When follower's log diverges significantly, decrementing nextIndex one entry at a time is slow.

**Optimization:** Include additional info in AppendEntries rejection:
- `conflictTerm`: Term of conflicting entry (or -1 if log too short)
- `conflictIndex`: First index with conflictTerm

**Leader's response strategy:**
1. If follower's log too short: set nextIndex = conflictIndex
2. If follower has conflicting term:
   - If leader has entries from conflictTerm: set nextIndex to leader's last entry for that term
   - Otherwise: set nextIndex = conflictIndex

**Result:** Often repairs log in single round-trip instead of many.

**Example:**
```
Leader (T5):  [1,a] [1,b] [1,c] [4,d] [4,e] [5,f]
Follower:     [1,a] [1,b] [2,c] [2,d] [2,e] [3,f] [3,g]

Old approach: 6 round-trips (decrement from index 7→6→5→4→3→2)

Fast backtracking:
- Follower rejects, returns conflictTerm=3, conflictIndex=6
- Leader has no entries from term 3, sets nextIndex=6
- Next rejection: conflictTerm=2, conflictIndex=3
- Leader has no entries from term 2, sets nextIndex=3
- Next AppendEntries succeeds: 2 round-trips total ✓
```

### 3. Lease-Based Reads (Linearizable Reads)

**Problem:** Reading from leader requires heartbeat to all servers to ensure leadership (read might be stale if leader was partitioned).

**Optimization - Leader Leases:**
- Leader maintains lease while receiving heartbeat responses from majority
- Lease duration < election timeout
- While lease valid, leader can serve reads locally without contacting followers
- Reduces read latency significantly (no network round-trip)

**Implementation considerations:**
- Requires synchronized clocks (or conservative lease durations)
- Must account for clock drift
- Leader cannot serve reads immediately after election (must commit entry from current term first)

**Alternative - Read Index:**
- Leader records commitIndex when read request arrives
- Sends heartbeat to majority to confirm leadership
- Once confirmed and state machine has applied up to readIndex, serve read
- Doesn't require synchronized clocks
- One round-trip instead of full Raft replication

### 4. Asynchronous Log Application

**Optimization:** Decouple log replication from state machine application:
- Leader commits entries as soon as replicated to majority
- Apply entries to state machine asynchronously in background
- Return result to client once applied
- Improves throughput by parallelizing replication and application

### 5. Snapshot Compaction

**Problem:** Log grows unbounded, consuming memory and slowing catch-up for new/recovering servers.

**Solution:** Periodically compact log by taking state machine snapshot:
- Snapshot includes: last included index, last included term, state machine state
- Discard all log entries up to last included index
- When follower too far behind, send snapshot instead of log entries

**Trade-offs:**
- Reduces memory usage and speeds up catch-up
- Snapshot creation can be expensive (copy-on-write helps)
- Must not block normal operations during snapshot

### 6. Parallel Log Replication

**Optimization:** Send AppendEntries to all followers in parallel
- Don't wait for responses sequentially
- Track responses and update commitIndex when majority responds
- Standard approach in production implementations

## Cluster Membership Changes

Raft supports changing cluster membership (adding/removing servers) without taking the cluster offline.

### The Problem

Directly switching from old configuration to new configuration is unsafe:
- Different servers may switch at different times
- Could have two independent majorities during transition
- Violates election safety (two leaders in same term)

**Example of unsafe direct switch:**
```
Old config: Server1, Server2, Server3 (majority = 2)
New config: Server1, Server2, Server3, Server4, Server5 (majority = 3)

During transition, could have:
- Old majority: Server1, Server2 elect LeaderA
- New majority: Server3, Server4, Server5 elect LeaderB
→ Two leaders in same term! ✗
```

### Joint Consensus Approach

Raft uses a two-phase approach with **joint consensus** configuration (C-old,new):

**Phase 1: Enter joint consensus**
1. Leader receives configuration change request
2. Creates C-old,new configuration (includes both old and new servers)
3. Replicates C-old,new as log entry
4. Once C-old,new committed, cluster operates under joint consensus rules

**Phase 2: Move to new configuration**
1. Leader creates C-new configuration
2. Replicates C-new as log entry
3. Once C-new committed, cluster operates under new configuration

### Joint Consensus Rules

While in C-old,new state:
- **Log entries must be replicated to majority of BOTH old and new configs**
- **Elections require majority of BOTH old and new configs**
- Any server from either configuration can serve as leader
- Ensures safety: impossible to have two leaders

**Why it works:**
- Cannot make decisions without majority of old configuration
- Cannot make decisions without majority of new configuration
- Any two majorities must overlap → consensus maintained

### Configuration Change Protocol

**Detailed steps:**

1. **Client requests membership change** (add/remove servers)

2. **Leader creates C-old,new entry**
   - Log entry containing both configurations
   - Leader applies C-old,new immediately when creating it

3. **C-old,new is replicated**
   - Sent to all servers in both old and new configurations
   - Servers apply C-old,new as soon as they receive it (before commitment)
   - System now requires dual majorities for all decisions

4. **C-old,new is committed**
   - Once replicated to majority of both old and new configurations
   - Leader knows it's safe to proceed to C-new

5. **Leader creates C-new entry**
   - Contains only new configuration
   - Replicates to all servers in new configuration

6. **C-new is committed**
   - Configuration change complete
   - Servers not in C-new can shut down

### Single-Server Changes

**Simplified approach:** Change only one server at a time.

**Why it's safe:**
- Majorities of any two consecutive configurations always overlap
- No possibility of disjoint majorities
- Simpler to implement than joint consensus

**Limitations:**
- Slower for adding/removing multiple servers
- May not maintain desired replication level during changes
- Still needs to handle special cases (see below)

### Special Considerations

**Adding new servers:**
- New servers start with empty logs
- Would take time to catch up
- During catch-up, availability could be impacted
- **Solution:** Add servers in non-voting mode first
  - Leader replicates log entries to them
  - Once caught up (within threshold), promote to voting member
  - Ensures availability maintained

**Removing current leader:**
- Leader could be removed from new configuration
- Leader must step down after committing C-new
- **Leader's steps:**
  1. Commit C-new (in which leader is not present)
  2. Step down to follower state
  3. Stop sending heartbeats
  4. New election occurs among remaining servers

**Disruptive servers:**
- Removed servers don't receive heartbeats
- Will timeout and start elections
- Can disrupt cluster with higher term numbers
- **Solution:** Servers ignore RequestVote RPCs when they believe current leader exists
  - Specifically: if received AppendEntries within minimum election timeout
  - Prevents disruption from removed servers

## Common Pitfalls and Debugging

### 1. Not Persisting State Correctly

**Pitfall:** Failing to persist `currentTerm`, `votedFor`, or `log[]` before responding to RPCs.

**Consequence:**
- Server could vote for multiple candidates in same term after crash
- Could lose committed entries after crash
- Violates safety properties

**Solution:** Always flush to disk before sending RPC responses:
```python
def handle_request_vote(term, candidate_id):
    if term > self.current_term:
        self.current_term = term
        self.voted_for = None
        self.persist()  # Must persist before continuing!

    if self.voted_for is None and candidate_log_is_up_to_date():
        self.voted_for = candidate_id
        self.persist()  # Must persist before responding!
        return True
```

### 2. Committing Entries from Previous Terms Directly

**Pitfall:** Trying to commit old entries by counting replicas, ignoring the current-term requirement.

**Consequence:** Can lead to committed entries being overwritten (violates State Machine Safety).

**Solution:** Always check `log[N].term == currentTerm` before committing. See "Commitment Rules" section for detailed example.

### 3. Applying Entries Out of Order

**Pitfall:** Applying log entries to state machine in wrong order or multiple times.

**Consequence:** State machines diverge across servers.

**Solution:**
- Track `lastApplied` index
- Apply entries sequentially: `for i in (lastApplied+1)..commitIndex`
- Make state machine operations idempotent where possible
- Never apply entries beyond commitIndex

### 4. Incorrect Election Timeout Handling

**Pitfall:** Not resetting election timeout when receiving valid AppendEntries from current leader.

**Consequence:** Unnecessary elections, reduced availability, term inflation.

**Solution:**
```python
def handle_append_entries(term, leader_id, ...):
    if term >= self.current_term:
        self.reset_election_timeout()  # Reset timer!
        self.current_term = term
        self.state = "follower"
```

### 5. Split Vote Loops

**Pitfall:** Multiple servers timing out simultaneously, causing perpetual split votes.

**Consequence:** Cluster unavailable until randomization breaks symmetry.

**Solution:**
- Use **randomized** election timeouts (e.g., 150-300ms)
- Randomize timeout for each election attempt
- Ensure randomization range is significant (not just ±1ms)

### 6. Ignoring Terms in RPC Responses

**Pitfall:** Leader receives RPC response with higher term but doesn't step down.

**Consequence:** Multiple leaders in different terms, split brain.

**Solution:** Always check response terms:
```python
response = send_append_entries(server, ...)
if response.term > self.current_term:
    self.current_term = response.term
    self.state = "follower"
    self.voted_for = None
    self.persist()
    return  # Stop being leader!
```

### 7. Not Handling RPC Duplicates/Reordering

**Pitfall:** Assuming RPCs arrive in order and exactly once.

**Consequence:** Log inconsistencies, incorrect state transitions.

**Solution:**
- Use term numbers to detect stale RPCs
- Reject requests with `term < currentTerm`
- Make RPC handlers idempotent where possible
- Include unique identifiers in client requests

### 8. Incorrect Log Matching Check

**Pitfall:** Only checking `prevLogTerm` without checking if entry exists at `prevLogIndex`.

**Consequence:** Accepting entries when follower's log is too short, causing inconsistencies.

**Solution:**
```python
def append_entries(prev_log_index, prev_log_term, entries):
    # Check 1: Log must be long enough
    if len(self.log) <= prev_log_index:
        return False

    # Check 2: Terms must match
    if self.log[prev_log_index].term != prev_log_term:
        return False

    # Both checks pass → accept entries
```

### Debugging Tips

**Check these invariants during testing:**
- `commitIndex ≤ lastApplied ≤ len(log)` at all times
- At most one server thinks it's leader in a given term
- All servers' `currentTerm` values are non-decreasing
- If entry is committed, it appears in all future leaders' logs

**Common debugging techniques:**
- Log all RPC requests/responses with timestamps and terms
- Verify persistent state is actually persisted (check after simulated crashes)
- Add assertions for safety invariants
- Use property-based testing (e.g., Jepsen, TLA+)
- Simulate network partitions, crashes, message delays/losses

## Implementation Considerations

### Persistence Layer

**Critical operations requiring persistence:**
1. Before responding to RequestVote
2. Before responding to AppendEntries
3. Before starting new election (increment term)

**Implementation options:**
- **Synchronous writes**: Call `fsync()` before responding (safest, slower)
- **Write-ahead log**: Batch multiple operations, fsync periodically (faster, more complex)
- **Battery-backed RAM**: Non-volatile memory for state (fastest, expensive)

**Trade-offs:**
- More frequent fsync → better safety, worse performance
- Batched fsync → better performance, slight safety risk during batching window

### Threading Model

**Common approaches:**

1. **Single-threaded with event loop:**
   - Simplest, no concurrency bugs
   - All operations serialized
   - Good for small clusters or low load

2. **Thread per RPC:**
   - Better throughput for large clusters
   - Requires locks around shared state
   - More complex, potential for deadlocks

3. **Actor model:**
   - Each server is an actor
   - Message passing between actors
   - Good balance of performance and correctness

**Shared state requiring protection:**
- currentTerm, votedFor, log[]
- commitIndex, lastApplied
- nextIndex[], matchIndex[]

### State Machine Interface

**Design considerations:**

```python
class StateMachine:
    def apply(self, command) -> result:
        """Apply command to state machine. Must be deterministic!"""
        pass

    def snapshot(self) -> bytes:
        """Create snapshot of current state."""
        pass

    def restore(self, snapshot: bytes):
        """Restore state from snapshot."""
        pass
```

**Requirements:**
- Operations must be deterministic (same input → same output)
- Must support snapshotting for log compaction
- Should be idempotent if possible (helps with at-least-once semantics)

### Client Interaction

**Linearizable semantics:**
- Client sends command to leader
- Leader waits for commitment and application
- Returns result to client
- If leader crashes, client retries with new leader

**Handling retries:**
- Client includes unique request ID
- Leader tracks recently processed requests
- If duplicate detected, return cached result (don't reapply)

### Testing Strategies

**Unit tests:**
- Test each RPC handler in isolation
- Verify state transitions (follower → candidate → leader)
- Test persistence and recovery

**Integration tests:**
- Multi-server clusters (3, 5, 7 servers)
- Inject failures: crashes, network partitions, message loss
- Verify safety properties maintained

**Property-based testing:**
- Use tools like Jepsen, Maelstrom, or custom frameworks
- Generate random failure scenarios
- Check invariants: no data loss, linearizability, etc.

**Formal verification:**
- TLA+ specification of Raft protocol
- Model checking to verify safety properties
- Helps catch subtle edge cases

## Example Implementation (Python-like pseudocode)

```python
class RaftNode:
    def __init__(self, node_id, peers):
        # Persistent state (must be saved to disk before responding to RPCs)
        self.current_term = 0
        self.voted_for = None
        self.log = []  # List of (term, command) tuples

        # Volatile state on all servers
        self.commit_index = 0
        self.last_applied = 0

        # Volatile state on leaders (reinitialized after election)
        self.next_index = {}   # For each server, index of next log entry to send
        self.match_index = {}  # For each server, index of highest log entry known to be replicated

        # Server metadata
        self.node_id = node_id
        self.peers = peers
        self.state = "follower"  # "follower", "candidate", or "leader"
        self.leader_id = None
        self.election_timeout = random_timeout(150, 300)

    def persist(self):
        """Write current_term, voted_for, and log to stable storage."""
        # In real implementation: fsync() to disk
        pass

    # === RPC Handlers ===

    def handle_request_vote(self, term, candidate_id, last_log_index, last_log_term):
        """RequestVote RPC handler."""
        # 1. Update term if needed
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.state = "follower"
            self.persist()

        # 2. Reject if term is old
        if term < self.current_term:
            return {"term": self.current_term, "vote_granted": False}

        # 3. Check if already voted for someone else
        if self.voted_for is not None and self.voted_for != candidate_id:
            return {"term": self.current_term, "vote_granted": False}

        # 4. Check if candidate's log is at least as up-to-date
        my_last_log_index = len(self.log) - 1
        my_last_log_term = self.log[-1][0] if self.log else 0

        log_is_up_to_date = (
            last_log_term > my_last_log_term or
            (last_log_term == my_last_log_term and last_log_index >= my_last_log_index)
        )

        if log_is_up_to_date:
            self.voted_for = candidate_id
            self.persist()
            self.reset_election_timeout()
            return {"term": self.current_term, "vote_granted": True}

        return {"term": self.current_term, "vote_granted": False}

    def handle_append_entries(self, term, leader_id, prev_log_index, prev_log_term,
                             entries, leader_commit):
        """AppendEntries RPC handler (also serves as heartbeat)."""
        # 1. Update term if needed
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.persist()

        # 2. Reject if term is old
        if term < self.current_term:
            return {"term": self.current_term, "success": False}

        # 3. Valid leader for this term
        self.state = "follower"
        self.leader_id = leader_id
        self.reset_election_timeout()

        # 4. Consistency check: log must contain entry at prev_log_index with prev_log_term
        if prev_log_index >= 0:
            if prev_log_index >= len(self.log):
                # Log too short
                return {"term": self.current_term, "success": False}
            if self.log[prev_log_index][0] != prev_log_term:
                # Term mismatch
                return {"term": self.current_term, "success": False}

        # 5. Delete conflicting entries and append new ones
        log_index = prev_log_index + 1
        for i, entry in enumerate(entries):
            if log_index + i < len(self.log):
                if self.log[log_index + i][0] != entry[0]:
                    # Conflict: delete this entry and all that follow
                    self.log = self.log[:log_index + i]
                    self.log.append(entry)
            else:
                # Append new entry
                self.log.append(entry)

        self.persist()

        # 6. Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
            self.apply_committed_entries()

        return {"term": self.current_term, "success": True}

    # === Leader Election ===

    def start_election(self):
        """Transition to candidate and start election."""
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.persist()

        votes_received = 1  # Vote for self
        last_log_index = len(self.log) - 1
        last_log_term = self.log[-1][0] if self.log else 0

        # Request votes from all peers
        for peer in self.peers:
            response = peer.request_vote(
                self.current_term, self.node_id, last_log_index, last_log_term
            )

            if response["term"] > self.current_term:
                # Discovered higher term, become follower
                self.current_term = response["term"]
                self.state = "follower"
                self.voted_for = None
                self.persist()
                return

            if response["vote_granted"]:
                votes_received += 1

        # Check if won election (majority)
        if votes_received > len(self.peers) // 2:
            self.become_leader()

    def become_leader(self):
        """Transition to leader state."""
        self.state = "leader"
        self.leader_id = self.node_id

        # Initialize leader-specific state
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = -1

        # Send initial heartbeat
        self.send_heartbeat()

    def send_heartbeat(self):
        """Send AppendEntries (heartbeat) to all followers."""
        for peer in self.peers:
            self.replicate_to_follower(peer)

    # === Log Replication ===

    def replicate_to_follower(self, peer):
        """Send AppendEntries RPC to specific follower."""
        next_idx = self.next_index[peer]
        prev_log_index = next_idx - 1
        prev_log_term = self.log[prev_log_index][0] if prev_log_index >= 0 else 0

        entries = self.log[next_idx:]  # Entries to send

        response = peer.append_entries(
            self.current_term, self.node_id, prev_log_index,
            prev_log_term, entries, self.commit_index
        )

        if response["term"] > self.current_term:
            # Discovered higher term, step down
            self.current_term = response["term"]
            self.state = "follower"
            self.voted_for = None
            self.persist()
            return

        if response["success"]:
            # Update next_index and match_index
            self.next_index[peer] = next_idx + len(entries)
            self.match_index[peer] = self.next_index[peer] - 1
            self.update_commit_index()
        else:
            # Decrement next_index and retry
            self.next_index[peer] = max(0, self.next_index[peer] - 1)

    def update_commit_index(self):
        """Update commit index based on majority replication."""
        # Find highest N where majority of match_index[i] >= N
        # and log[N].term == current_term
        for n in range(len(self.log) - 1, self.commit_index, -1):
            if self.log[n][0] == self.current_term:
                replicated_count = sum(
                    1 for peer in self.peers if self.match_index.get(peer, -1) >= n
                )
                if replicated_count + 1 > len(self.peers) // 2:  # +1 for leader itself
                    self.commit_index = n
                    self.apply_committed_entries()
                    break

    # === State Machine ===

    def apply_committed_entries(self):
        """Apply committed entries to state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied]
            self.apply_to_state_machine(entry[1])  # entry[1] is the command

    def apply_to_state_machine(self, command):
        """Apply a single command to the state machine."""
        # In real implementation: execute the command
        pass

    # === Utilities ===

    def reset_election_timeout(self):
        """Reset election timeout with randomization."""
        self.election_timeout = random_timeout(150, 300)

def random_timeout(min_ms, max_ms):
    """Generate random timeout between min_ms and max_ms."""
    import random
    return random.uniform(min_ms, max_ms) / 1000.0
```

## Real-World Implementations

Raft is used in production by many distributed systems:

- **etcd**: Kubernetes' distributed key-value store
- **Consul**: Service mesh and service discovery (HashiCorp)
- **CockroachDB**: Distributed SQL database
- **TiKV**: Distributed transactional key-value database
- **LogCabin**: Replicated state machine for datacenter coordination

## Further Reading

- **Original paper**: "In Search of an Understandable Consensus Algorithm" by Diego Ongaro and John Ousterhout
- **Interactive visualization**: https://raft.github.io/
- **TLA+ specification**: Formal specification for verification
- **Raft PhD dissertation**: Extended version with more details and proofs
