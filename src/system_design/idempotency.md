# Idempotency & Deduplication

A request is **idempotent** if applying it once or many times produces the same result. Networks fail; retries are mandatory; therefore idempotency is mandatory.

## Why It Matters

In any distributed system, you cannot tell the difference between:
- "request succeeded, response lost" and
- "request failed".

Without idempotency, the client retries → the action happens twice. Double-charge, double-post, double-ship. Idempotency is what makes "at-least-once" delivery survivable.

## HTTP Method Semantics

| Method | Idempotent? | Safe? |
|---|---|---|
| GET | ✅ | ✅ |
| HEAD | ✅ | ✅ |
| PUT | ✅ (replace) | ❌ |
| DELETE | ✅ | ❌ |
| POST | ❌ (by default) | ❌ |
| PATCH | ❌ (depends) | ❌ |

POST is the dangerous one — and the most common write. Idempotency keys fix it.

## Idempotency Keys (the Standard Pattern)

Client generates a unique key per logical request and sends it; server records "I handled this key" and short-circuits duplicates.

```
POST /v1/payments
Idempotency-Key: 8f7a3e2c-...    ← client-generated UUID
{ "amount": 1000, "currency": "USD", "to": "acct_42" }
```

### Server-side flow
```
1. Receive request with key K.
2. Look up key K in store:
     ├── Not found → acquire lock(K); proceed; on success persist (K, response).
     ├── Found, in-flight → return 409 Conflict or wait.
     ├── Found, completed → return cached response (same body, same status).
     └── Found, completed but body differs → return 422 mismatch.
4. Release lock; return response.
```

### Implementation sketch (Redis)
```
SET idem:K "in_flight" NX EX 600    ← atomic acquire, 10 min TTL
if NX failed:
    GET idem:K → if "in_flight": 409 / poll
                 if {response}:    return cached
do work
SET idem:K "{response}" EX 86400    ← cache result 24h
```

**Lock TTL** must exceed worst-case operation time. **Cache TTL** must exceed client retry window (Stripe uses 24h).

### Key generation
- Client generates UUID v4. Stored locally so retries reuse same key.
- Don't derive key from payload (request body fingerprint) — defeats deliberate retries that change keys per logical op.
- Don't reuse keys across logically different operations.

## Body Mismatch Detection

What if a client retries with the same idempotency key but **different body**? Two camps:

1. **Strict (Stripe)**: hash the request body; mismatch → 422. Forces deliberate clients to use a new key.
2. **First wins**: return cached response regardless. Risk: client *intended* a different operation.

Strict is safer. Always hash and compare.

## Database-level Idempotency

### Unique constraint
The simplest dedup: a unique index on (idempotency_key) in the table. INSERT fails on duplicate → catch, return existing row.

```sql
CREATE TABLE payments (
  id UUID PRIMARY KEY,
  idempotency_key TEXT UNIQUE NOT NULL,
  amount NUMERIC,
  status TEXT,
  created_at TIMESTAMPTZ
);

INSERT INTO payments(id, idempotency_key, ...)
VALUES (gen_random_uuid(), $key, ...)
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING id;
```

If `RETURNING` is empty, fetch existing.

### Conditional updates with version
```sql
UPDATE accounts
   SET balance = balance - 100, version = version + 1
 WHERE id = $id AND version = $expected;
```

0 rows updated → already applied (or stale). Re-read and decide.

## Idempotency in Message Queues

"At-least-once" delivery is the default. Consumers must dedup.

### Consumer-side dedup
```
on_message(msg):
    key = msg.idempotency_key or hash(msg.id + topic)
    if dedup_store.add_if_absent(key, ttl=24h):
        process(msg)
        commit_offset()
    else:
        commit_offset()   # already processed
```

`add_if_absent` must be atomic (Redis SET NX, DynamoDB conditional put).

### Outbox Pattern (publisher-side, for transactional emit)
Avoid the "DB committed but message not sent" hole:
```
BEGIN
  INSERT order ...
  INSERT outbox (event_id, payload, status='pending')
COMMIT

separate poller: SELECT outbox WHERE status='pending'
                 publish to Kafka
                 UPDATE status='sent'
```

Combined with idempotency keys on consumers, this gives effectively-once.

## Exactly-Once Illusions

True end-to-end exactly-once delivery is impossible in a distributed system. What's actually achievable:

| Layer | Guarantee | How |
|---|---|---|
| Producer → broker | exactly-once write | Idempotent producer (Kafka) |
| Within broker | persisted exactly | Replication + flush |
| Broker → consumer | at-least-once delivery | Network retries |
| Consumer state | exactly-once effect | Idempotent processing |

The trick is **idempotent effects**, not exactly-once messages. Kafka EOS, Flink checkpointing, Spark streaming all rely on this.

## Common Idempotent Operation Patterns

### CRDT-style aggregations
Counts as **set additions** rather than increments:
```
- BAD: counter += 1
- GOOD: SET.add(event_id);   counter = SET.size()
```

Re-applying is a no-op because sets dedup.

### "Setter" instead of "delta"
- BAD: `withdraw(100)` (relative)
- GOOD: `set_balance(900, expected_was=1000)` (absolute + precondition)

The conditional setter is idempotent and detects conflicts.

### State-machine transitions
```
move(order, from='paid', to='shipped')
```

If order is already `shipped`, transition is a no-op (or 200 with current state). Avoid actions that re-fire side effects (don't re-ship).

## Dedup Windows

Storage for idempotency keys is bounded. Pick a **dedup window** — the max retry delay you'll honor.

| Use case | Window |
|---|---|
| Synchronous API (web request) | 24 h |
| Mobile (poor connectivity) | 24–72 h |
| Message bus consumer | 7 d |
| Financial / regulatory | 90 d to permanent (DB index) |

After the window, dedup state expires. Late retries may double-apply — accept this risk or use a permanent unique constraint.

## Tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| Idempotency keys (Redis) | Fast, simple | TTL-bounded, requires client cooperation |
| Unique index in DB | Permanent, transactional | Slower (DB write per key) |
| Outbox + dedup at consumer | Atomic emit | Adds DB pressure |
| Pure functional state (CRDT) | Naturally idempotent | Limited operation set |
| Versioned conditional updates | Detects conflicts | Requires version field everywhere |

## Pitfalls

- **Side effects outside the key check.** Make sure dedup happens *before* charging the card. Race: lock acquired → crash before persist → next retry double-charges.
- **Storing the key but not the response.** Second retry succeeds with a different result. Always return *the original response body*.
- **Time-of-check / time-of-use (TOCTOU).** "Check then act" without a lock allows two requests to both find "not yet processed" and both proceed. Use atomic ops or row locks.
- **Different keys for the same logical op.** Mobile retries with new keys defeat dedup. Persist the key client-side.
- **Idempotency across services.** A "create order" that fans out to inventory + payment must propagate the key. Each downstream service dedupes on a derived key (e.g., `order_id:inventory_reserve`).

## When Idempotency Is Hard

- **External actions** (send email, push to third-party API). Wrap them with idempotency where possible (most providers support Idempotency-Key); else dedup before calling.
- **Time-dependent ops** (e.g., "charge $X at current rate"). Snapshot inputs into the request so replay produces same result.
- **Side-effectful reads** (e.g., "claim next job from queue"). Convert to "claim job X" — promote the ID into the request.

## Interview Cheat

Whenever you design any write API, mention:
1. **Idempotency-Key header** support.
2. **Storage choice** (Redis short, DB unique index for permanent).
3. **Body-hash mismatch handling**.
4. **Retry behavior** (client backoff with jitter, reuse same key).
5. **Dedup window**.

Without these, a payment system is incorrect, not just inefficient.

## Related

- `design_patterns.md` — saga, retry-with-backoff patterns
- `message_queues.md` — at-least-once delivery semantics
- `distributed_systems.md` — why "exactly-once" is an illusion
- `databases.md` — unique constraints, transactional outbox
