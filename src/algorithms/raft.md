# Raft Consensus Algorithm

Raft is a consensus algorithm designed to be easy to understand. It's used for managing a replicated log in distributed systems.

## Overview

Raft ensures that a cluster of servers agrees on a sequence of values, even in the presence of failures.

**Key Properties:**
- Leader election
- Log replication
- Safety
- Membership changes

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

**Commitment process:**
1. Leader tracks highest committed entry in `commitIndex`
2. Once entry committed, leader applies it to state machine
3. Leader includes `commitIndex` in AppendEntries RPCs
4. Followers apply committed entries to their state machines
5. Entries are applied in log order to ensure state machine consistency

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

## Example (Python-like pseudocode)

```python
class RaftNode:
    def __init__(self):
        self.state = "follower"
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0

    def request_vote(self, term, candidate_id):
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None

        if self.voted_for is None:
            self.voted_for = candidate_id
            return True
        return False

    def append_entries(self, term, leader_id, entries):
        if term >= self.current_term:
            self.state = "follower"
            self.current_term = term
            self.log.extend(entries)
            return True
        return False
```

Raft provides understandable consensus for building reliable distributed systems like etcd, Consul, and CockroachDB.
