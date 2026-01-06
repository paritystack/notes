# Probabilistic Data Structures

> **Domain:** Big Data, High Performance Computing, Algorithms
> **Key Concepts:** Bloom Filter, HyperLogLog, Count-Min Sketch, Approximation

**Probabilistic Data Structures** are clever algorithms that use hashing and approximation to answer queries about massive datasets using extremely small amounts of memory. They trade **accuracy** for **space**.

**The Guarantee:** They never lie in a way that hurts you (e.g., False Positives are possible, False Negatives are impossible, or error is bounded).

---

## 1. Bloom Filter (Membership)
*   **Question:** "Have I seen this item before?"
*   **Memory:** Bits (not bytes).
*   **Operations:** `add(item)`, `contains(item)`.
*   **Accuracy:**
    *   **False Positive:** Possible. (Says "Yes" when it's actually "No").
    *   **False Negative:** Impossible. (If it says "No", it definitely means "No").
*   **Mechanism:** $K$ hash functions map an item to $K$ bits in an array. To check, see if all $K$ bits are set to 1.
*   **Use Case:** Avoiding expensive disk lookups. "Is this URL malicious?" (Check Bloom Filter -> If Yes, check Database).

---

## 2. HyperLogLog (Cardinality)
*   **Question:** "How many *unique* items have I seen?" (e.g., Daily Active Users).
*   **The Problem:** Storing 100M User IDs requires 400MB+ RAM.
*   **The Solution:** HyperLogLog needs ~1.5KB to count up to $10^9$ items with 2% error.
*   **Mechanism (The Flajolet-Martin idea):**
    1.  Hash the item to a binary string.
    2.  Count the number of leading zeros.
    3.  If you see a hash with 10 leading zeros, you've likely seen $2^{10}$ items.
    4.  Use harmonic averaging (stochastic averaging) across many buckets to reduce variance.
*   **Use Case:** Redis `PFADD` / `PFCOUNT`. counting unique IP addresses visiting a site.

---

## 3. Count-Min Sketch (Frequency)
*   **Question:** "How many times did I see this item?" (Frequency Table).
*   **The Problem:** A Hash Map `{"IP": count}` explodes in memory if you have billions of IPs.
*   **The Solution:** A 2D array of counters (Rows = Hash Functions, Columns = Buckets).
*   **Mechanism:**
    1.  **Add:** Hash item $x$ with function $h_1, h_2... h_d$. Increment the counter at each position.
    2.  **Query:** Hash item $x$. Look at the counters. Return the **minimum** value among them.
*   **Why Minimum?** Because collisions only *add* to the count (overestimation). The true count cannot be higher than the minimum observed counter.
*   **Error:** Always overestimates, never underestimates.
*   **Use Case:** "Top K" problems. "What are the trending hashtags right now?"

---

## 4. T-Digest (Quantiles)
*   **Question:** "What is the 99th percentile latency?"
*   **The Problem:** You can't calculate P99 without sorting all values (expensive) or storing all of them.
*   **The Solution:** T-Digest clusters points into "centroids" with a mean and a weight. It keeps centroids small near the tails (0% and 100%) for high accuracy where it matters (P99, P99.9) and merges them in the middle (P50).
*   **Use Case:** Monitoring systems (Prometheus histograms, DataDog).

---

## 5. Summary Table

| Structure | Answers... | Error Type | Space |
| :--- | :--- | :--- | :--- |
| **Bloom Filter** | Is X in the set? | False Positives | 10 bits / item |
| **HyperLogLog** | How many unique X? | Standard Error (~1%) | < 12KB |
| **Count-Min Sketch** | Frequency of X? | Overestimation | Fixed (e.g., 2MB) |
| **T-Digest** | Percentile of X? | Approx Error | Fixed (small) |

---

## 6. Implementation Example (Python Application)
*Don't implement these from scratch in production. Use libraries.*

```python
import mmh3 # MurmurHash
from bitarray import bitarray

class SimpleBloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, string):
        for seed in range(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            self.bit_array[result] = 1

    def lookup(self, string):
        for seed in range(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            if self.bit_array[result] == 0:
                return False
        return True
```
