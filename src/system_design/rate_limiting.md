# Rate Limiting

> **Domain:** System Design, Distributed Systems, API Security
> **Key Concepts:** Token Bucket, Leaky Bucket, Throttling, Redis, Sliding Window

**Rate Limiting** is a defensive strategy used to control the amount of traffic sent or received by a network interface or controller. It is critical for preventing Denial of Service (DoS) attacks, managing resource usage (CPU/Memory), and enforcing tiered pricing models (e.g., "Free Tier users get 100 req/min").

---

## 1. Why Rate Limit?
1.  **Prevent Starvation:** Stop a single "noisy neighbor" from consuming all database connections.
2.  **Security:** Mitigate brute-force login attempts or credential stuffing.
3.  **Cost Management:** Prevent auto-scaling groups from spinning up infinite servers due to a bot spike.
4.  **Revenue:** Monetize API access by selling higher limits.

---

## 2. Algorithms

Choosing the right algorithm is a trade-off between "burstiness" allowance and implementation complexity.

### 2.1. Token Bucket
*   **Concept:** A bucket holds tokens. Tokens are added at a fixed rate (e.g., 10/sec). Each request consumes a token. If the bucket is empty, the request is rejected.
*   **Behavior:** Allows **bursts**. If the bucket is full (e.g., capacity 100), the user can fire 100 requests instantly, then gets throttled to the refill rate.
*   **Use Case:** User-facing APIs where short bursts of activity are normal (e.g., loading a dashboard).

### 2.2. Leaky Bucket
*   **Concept:** Requests enter a queue (bucket). The queue is processed (leaks) at a constant, fixed rate. If the queue is full, new requests are dropped.
*   **Behavior:** Smooths out traffic. It enforces a strict, uniform output rate regardless of input bursts.
*   **Use Case:** Writing to a database or message queue (Kafka) where a steady ingestion rate is required to prevent overload.

### 2.3. Fixed Window Counter
*   **Concept:** Divide time into windows (e.g., 1 minute). Counter resets at the start of each window.
*   **Flaw:** **The Edge Case Spike**. If a user sends 100 requests at 10:00:59 and 100 more at 10:01:01, they sent 200 requests in 2 seconds, violating the "100 per minute" intent.

### 2.4. Sliding Window Log
*   **Concept:** Store a timestamp for every request. On a new request, count how many timestamps exist in the last $T$ seconds.
*   **Pros:** Perfectly accurate.
*   **Cons:** Expensive memory usage (stores data per request).

### 2.5. Sliding Window Counter (Hybrid)
*   **Concept:** Weighted average of the previous window's count and current window's count.
*   **Formula:** `Count = Current_Window_Reqs + (Previous_Window_Reqs * (1 - %_time_passed_in_current))`
*   **Pros:** Accurate enough, memory efficient.

---

## 3. Distributed Rate Limiting

In a microservices architecture, you can't store counters in local memory (the user might hit Server A then Server B). You need a shared store, usually **Redis**.

### 3.1. Race Conditions
*   *Naive Approach:* `GET key`, `INCREMENT`, `SET key`.
*   *Problem:* Two servers read "99" simultaneously, both increment to "100", and both allow the request. Real count is 101.
*   *Solution:* **Lua Scripts**. Redis Lua scripts execute atomically.

**Sample Lua Script (Fixed Window):**
```lua
local current = redis.call("INCR", KEYS[1])
if tonumber(current) == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return current
```

### 3.2. Synchronization Issues
*   **Thundering Herd:** If millions of keys expire at the exact same second, Redis CPU spikes. **Jitter** your expiration times.
*   **Latency:** Calling Redis for every API request adds latency.
    *   *Solution:* **Local Memory Caching**. Allow a small buffer (e.g., 10 tokens) in local memory, sync with Redis asynchronously. "Eventually Consistent" rate limiting.

---

## 4. HTTP Headers

Standardize how you communicate limits to clients.

*   `X-RateLimit-Limit`: The ceiling for this time window.
*   `X-RateLimit-Remaining`: Requests left.
*   `X-RateLimit-Reset`: UTC timestamp when the window resets.
*   `Retry-After`: (On 429 Error) Seconds to wait before retrying.

---

## 5. Implementation Strategies

### 5.1. API Gateway Level (The Standard)
Implement limiting at the entry point (Nginx, Kong, AWS API Gateway).
*   *Pros:* Offloads logic from services. Blocks bad traffic early.
*   *Cons:* Less granular context (Gateway might not know "User ID" deep inside the payload).

### 5.2. Application Level (Middleware)
Implement inside the code (e.g., Express.js middleware, Django decorators).
*   *Pros:* Full access to business logic ("Limit Premium users differently").
*   *Cons:* Consumes app resources to reject requests.

## 6. Conclusion

Rate limiting is not just about blocking bots; it's a fundamental part of system reliability. For most distributed systems, the **Token Bucket** algorithm backed by **Redis Lua scripts** is the industry standard balance of performance and accuracy.
