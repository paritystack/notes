# ACID vs. BASE

> **Domain:** Databases, Distributed Systems
> **Key Concepts:** Consistency, Availability, CAP Theorem, Eventual Consistency

This is the fundamental trade-off in database design. **ACID** is the gold standard for traditional Relational Databases (SQL). **BASE** is the model for NoSQL/Distributed systems that prioritize scale.

---

## 1. The CAP Theorem

You cannot understand ACID/BASE without CAP. In a distributed data store, you can only guarantee **two** of the following three:

1.  **Consistency (C):** Every read receives the most recent write or an error. (All nodes see the same data).
2.  **Availability (A):** Every request receives a (non-error) response, without the guarantee that it contains the most recent write.
3.  **Partition Tolerance (P):** The system continues to operate despite an arbitrary number of messages being dropped/delayed by the network.

**The Reality:** In a distributed system, network partitions (P) *will* happen. So you essentially choose between **CP** (Consistency) and **AP** (Availability).

---

## 2. ACID (The CP Choice)

Used by: PostgreSQL, MySQL, Oracle, SQL Server.
Focus: **Correctness**.

1.  **Atomicity:** All or nothing. If a transaction has 10 steps and step 10 fails, steps 1-9 are rolled back.
    *   *Example:* Money transfer. Debit A, Credit B. Both happen, or neither happens.
2.  **Consistency:** The database moves from one valid state to another valid state. Constraints (Foreign Keys, Unique) are enforced immediately.
3.  **Isolation:** Concurrent transactions don't interfere. (e.g., Transaction A doesn't see B's uncommitted data).
4.  **Durability:** Once committed, data is saved even if power fails (Write-Ahead Logs).

---

## 3. BASE (The AP Choice)

Used by: Cassandra, DynamoDB, MongoDB (in certain modes).
Focus: **Availability & Scale**.

1.  **Basically Available:** The system guarantees availability. If a node is down, another node answers (potentially with stale data).
2.  **Soft state:** The state of the system may change over time, even without input (due to replication syncing).
3.  **Eventual consistency:** The system will *eventually* become consistent once it stops receiving input.
    *   *Example:* You update your profile picture. Your friend might see the old one for 5 seconds. That is acceptable.

---

## 4. Tuning Consistency

Modern databases aren't just "ACID" or "BASE". They are tunable.

### 4.1. Quorums (Cassandra/Dynamo)
*   $N$: Number of replicas (e.g., 3).
*   $W$: Write quorum (how many must confirm write).
*   $R$: Read quorum (how many must be read to confirm data).

*   **Strong Consistency:** $R + W > N$. (e.g., Write to 2, Read from 2. You are guaranteed to see the write).
*   **Eventual Consistency:** $R + W \le N$. (Fast, but risky).

### 4.2. Isolation Levels (SQL)
Even ACID DBs trade consistency for speed.
*   **Serializable:** Perfect isolation. Slowest.
*   **Repeatable Read:** Default in MySQL.
*   **Read Committed:** Default in Postgres.
*   **Read Uncommitted:** Dirty reads allowed. Fastest/Dangerous.

---

## 5. When to use what?

| Use Case | Choice | Why? |
| :--- | :--- | :--- |
| **Financial Ledger** | **ACID** | Money cannot disappear. Atomicity is non-negotiable. |
| **Social Media Feed** | **BASE** | If a post appears 2 seconds late, nobody cares. Availability is king. |
| **E-Commerce Cart** | **ACID-ish** | You don't want to sell the last item twice (Inventory count). |
| **Sensor Logs** | **BASE** | Massive write volume. Occasional lost packet is fine. |
