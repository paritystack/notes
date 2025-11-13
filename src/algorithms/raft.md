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

1. Follower times out (150-300ms)
2. Becomes candidate, increments term
3. Votes for itself
4. Requests votes from other servers
5. If majority votes: becomes leader
6. If another leader found: becomes follower

## Log Replication

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

## Safety Rules

1. **Election Safety**: At most one leader per term
2. **Leader Append-Only**: Leader never overwrites entries
3. **Log Matching**: If two logs contain entry with same index/term, entries are identical
4. **Leader Completeness**: If entry committed in term, it's in leader's log
5. **State Machine Safety**: If server applies entry at index, no other server applies different entry at that index

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
