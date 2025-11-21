# Circular Buffer (Ring Buffer)

## Overview

A **circular buffer** (also known as a **ring buffer** or **cyclic buffer**) is a fixed-size buffer that uses a single, continuous block of memory and wraps around when it reaches the end. It operates as if the memory is connected end-to-end in a circular fashion, making it ideal for implementing fixed-size queues where old data is automatically overwritten by new data.

**Key Characteristics:**
- **Fixed Size**: Memory is allocated once and size doesn't change
- **O(1) Operations**: Enqueue and dequeue in constant time
- **Wrap-Around**: When reaching the end, operations continue from the beginning
- **No Shifting**: Elements stay in place, only head/tail pointers move
- **Cache-Friendly**: Contiguous memory improves cache performance
- **Lock-Free**: Can be implemented without locks for single producer/consumer

**Visual Representation:**

```
Initial State (capacity = 5):
[_, _, _, _, _]
 ↑
head/tail

After enqueueing 3 elements (A, B, C):
[A, B, C, _, _]
 ↑     ↑
head  tail

After dequeueing 2 elements:
[_, _, C, _, _]
       ↑  ↑
      head tail

After enqueueing 4 more elements (D, E, F, G):
[F, G, C, D, E]  ← Buffer wraps around!
    ↑  ↑
   tail head
```

## Implementation

### Basic Circular Buffer (Python)

```python
class CircularBuffer:
    """
    A fixed-size circular buffer with automatic overwriting.
    """
    def __init__(self, capacity):
        """
        Initialize circular buffer with given capacity.
        Time: O(n), Space: O(n)
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # Points to oldest element
        self.tail = 0  # Points to next write position
        self.size = 0  # Current number of elements
        self.is_full = False

    def enqueue(self, item):
        """
        Add item to buffer. Overwrites oldest if full.
        Time: O(1), Space: O(1)
        """
        self.buffer[self.tail] = item

        if self.is_full:
            # Overwrite mode: move head forward too
            self.head = (self.head + 1) % self.capacity

        self.tail = (self.tail + 1) % self.capacity

        if self.tail == self.head:
            self.is_full = True

        self.size = min(self.size + 1, self.capacity)

    def dequeue(self):
        """
        Remove and return oldest item.
        Time: O(1), Space: O(1)
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty buffer")

        item = self.buffer[self.head]
        self.buffer[self.head] = None  # Optional: clear reference
        self.head = (self.head + 1) % self.capacity
        self.is_full = False
        self.size -= 1

        return item

    def peek(self):
        """
        Return oldest item without removing.
        Time: O(1), Space: O(1)
        """
        if self.is_empty():
            raise IndexError("Peek from empty buffer")
        return self.buffer[self.head]

    def is_empty(self):
        """Check if buffer is empty. Time: O(1)"""
        return self.size == 0

    def is_full_buffer(self):
        """Check if buffer is full. Time: O(1)"""
        return self.is_full

    def __len__(self):
        """Return current size. Time: O(1)"""
        return self.size

    def __str__(self):
        """String representation showing buffer state."""
        if self.is_empty():
            return "[]"

        items = []
        idx = self.head
        for _ in range(self.size):
            items.append(str(self.buffer[idx]))
            idx = (idx + 1) % self.capacity

        return f"[{', '.join(items)}]"


# Example usage
if __name__ == "__main__":
    # Create buffer with capacity 3
    cb = CircularBuffer(3)

    # Enqueue elements
    cb.enqueue('A')
    cb.enqueue('B')
    cb.enqueue('C')
    print(f"After enqueuing A,B,C: {cb}")  # [A, B, C]

    # Dequeue one element
    print(f"Dequeued: {cb.dequeue()}")  # A
    print(f"After dequeue: {cb}")  # [B, C]

    # Enqueue more (wraps around)
    cb.enqueue('D')
    cb.enqueue('E')  # Overwrites B
    print(f"After enqueuing D,E: {cb}")  # [C, D, E]
```

### Non-Overwriting Version (Blocking)

```python
class BlockingCircularBuffer:
    """
    Circular buffer that blocks/raises error when full instead of overwriting.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item):
        """
        Add item to buffer. Raises exception if full.
        Time: O(1), Space: O(1)
        """
        if self.is_full():
            raise OverflowError("Buffer is full")

        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        """
        Remove and return oldest item.
        Time: O(1), Space: O(1)
        """
        if self.is_empty():
            raise IndexError("Buffer is empty")

        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1

        return item

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity
```

### Thread-Safe Circular Buffer (Single Producer/Single Consumer)

```python
import threading

class ThreadSafeCircularBuffer:
    """
    Lock-free circular buffer for single producer and single consumer.
    Uses atomic operations for thread safety without locks.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # Only modified by consumer
        self.tail = 0  # Only modified by producer

    def try_enqueue(self, item):
        """
        Try to add item. Returns True if successful, False if full.
        Time: O(1), Space: O(1)
        Thread-safe for single producer.
        """
        next_tail = (self.tail + 1) % self.capacity

        # Check if buffer is full
        if next_tail == self.head:
            return False

        self.buffer[self.tail] = item
        self.tail = next_tail  # Atomic write

        return True

    def try_dequeue(self):
        """
        Try to remove item. Returns (True, item) if successful,
        (False, None) if empty.
        Time: O(1), Space: O(1)
        Thread-safe for single consumer.
        """
        # Check if buffer is empty
        if self.head == self.tail:
            return False, None

        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity  # Atomic write

        return True, item

    def size(self):
        """
        Approximate size (may not be exact in concurrent scenario).
        Time: O(1)
        """
        head = self.head
        tail = self.tail

        if tail >= head:
            return tail - head
        else:
            return self.capacity - head + tail
```

### JavaScript Implementation

```javascript
class CircularBuffer {
    constructor(capacity) {
        this.capacity = capacity;
        this.buffer = new Array(capacity);
        this.head = 0;
        this.tail = 0;
        this.size = 0;
        this.isFull = false;
    }

    /**
     * Add item to buffer. Overwrites oldest if full.
     * Time: O(1), Space: O(1)
     */
    enqueue(item) {
        this.buffer[this.tail] = item;

        if (this.isFull) {
            this.head = (this.head + 1) % this.capacity;
        }

        this.tail = (this.tail + 1) % this.capacity;

        if (this.tail === this.head) {
            this.isFull = true;
        }

        this.size = Math.min(this.size + 1, this.capacity);
    }

    /**
     * Remove and return oldest item.
     * Time: O(1), Space: O(1)
     */
    dequeue() {
        if (this.isEmpty()) {
            throw new Error("Dequeue from empty buffer");
        }

        const item = this.buffer[this.head];
        this.buffer[this.head] = undefined;
        this.head = (this.head + 1) % this.capacity;
        this.isFull = false;
        this.size--;

        return item;
    }

    /**
     * Return oldest item without removing.
     * Time: O(1), Space: O(1)
     */
    peek() {
        if (this.isEmpty()) {
            throw new Error("Peek from empty buffer");
        }
        return this.buffer[this.head];
    }

    isEmpty() {
        return this.size === 0;
    }

    isFullBuffer() {
        return this.isFull;
    }

    length() {
        return this.size;
    }

    toString() {
        if (this.isEmpty()) return "[]";

        const items = [];
        let idx = this.head;
        for (let i = 0; i < this.size; i++) {
            items.push(this.buffer[idx]);
            idx = (idx + 1) % this.capacity;
        }

        return `[${items.join(', ')}]`;
    }
}

// Example usage
const cb = new CircularBuffer(3);
cb.enqueue('A');
cb.enqueue('B');
cb.enqueue('C');
console.log(cb.toString());  // [A, B, C]

console.log(cb.dequeue());  // A
console.log(cb.toString());  // [B, C]

cb.enqueue('D');
cb.enqueue('E');  // Overwrites B
console.log(cb.toString());  // [C, D, E]
```

## Advanced Patterns

### 1. Generic Circular Buffer with Batch Operations

```python
from typing import TypeVar, Generic, List, Optional

T = TypeVar('T')

class GenericCircularBuffer(Generic[T]):
    """
    Generic circular buffer with batch operations.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Optional[T]] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue_batch(self, items: List[T]) -> int:
        """
        Enqueue multiple items. Returns number of items actually added.
        Time: O(k) where k = min(len(items), capacity)
        """
        count = 0
        for item in items:
            if self.size >= self.capacity:
                break
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            count += 1
        return count

    def dequeue_batch(self, n: int) -> List[T]:
        """
        Dequeue up to n items.
        Time: O(min(n, size))
        """
        result = []
        for _ in range(min(n, self.size)):
            item = self.buffer[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            result.append(item)
        return result

    def peek_batch(self, n: int) -> List[T]:
        """
        Peek at up to n oldest items without removing.
        Time: O(min(n, size))
        """
        result = []
        idx = self.head
        for _ in range(min(n, self.size)):
            result.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
        return result
```

### 2. Resizable Circular Buffer

```python
class ResizableCircularBuffer:
    """
    Circular buffer that can grow when needed.
    """
    def __init__(self, initial_capacity: int = 8):
        self.buffer = [None] * initial_capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    @property
    def capacity(self):
        return len(self.buffer)

    def enqueue(self, item):
        """
        Add item, growing buffer if necessary.
        Time: O(1) amortized (O(n) when resizing)
        """
        if self.size == self.capacity:
            self._resize(self.capacity * 2)

        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        """
        Remove oldest item, shrinking if necessary.
        Time: O(1) amortized (O(n) when resizing)
        """
        if self.size == 0:
            raise IndexError("Dequeue from empty buffer")

        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1

        # Shrink if size is 1/4 of capacity
        if self.size > 0 and self.size == self.capacity // 4:
            self._resize(self.capacity // 2)

        return item

    def _resize(self, new_capacity):
        """
        Resize buffer to new capacity.
        Time: O(n), Space: O(n)
        """
        new_buffer = [None] * new_capacity

        # Copy elements in order
        for i in range(self.size):
            new_buffer[i] = self.buffer[(self.head + i) % self.capacity]

        self.buffer = new_buffer
        self.head = 0
        self.tail = self.size
```

### 3. Circular Buffer with Statistics

```python
from collections import deque
import statistics

class StatisticalCircularBuffer:
    """
    Circular buffer that maintains running statistics.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = CircularBuffer(capacity)
        self.sum = 0
        self.values = deque(maxlen=capacity)

    def add(self, value: float):
        """
        Add value and update statistics.
        Time: O(1)
        """
        # Remove old value if buffer is full
        if len(self.values) == self.capacity:
            old_value = self.values[0]
            self.sum -= old_value

        self.values.append(value)
        self.sum += value
        self.buffer.enqueue(value)

    def mean(self) -> float:
        """Calculate mean. Time: O(1)"""
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    def median(self) -> float:
        """Calculate median. Time: O(n log n)"""
        if not self.values:
            return 0.0
        return statistics.median(self.values)

    def std_dev(self) -> float:
        """Calculate standard deviation. Time: O(n)"""
        if len(self.values) < 2:
            return 0.0
        return statistics.stdev(self.values)

    def min(self) -> float:
        """Get minimum value. Time: O(n)"""
        if not self.values:
            raise ValueError("Buffer is empty")
        return min(self.values)

    def max(self) -> float:
        """Get maximum value. Time: O(n)"""
        if not self.values:
            raise ValueError("Buffer is empty")
        return max(self.values)
```

## Time & Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| **Initialization** | O(n) | O(n) | Allocate buffer once |
| **Enqueue** | O(1) | O(1) | Always constant time |
| **Dequeue** | O(1) | O(1) | Always constant time |
| **Peek** | O(1) | O(1) | Just read, no write |
| **Is Empty/Full** | O(1) | O(1) | Check size/pointers |
| **Batch Operations** | O(k) | O(1) | k = number of items |
| **Resize (if resizable)** | O(n) | O(n) | Amortized O(1) |

**Memory Characteristics:**
- **Fixed Size**: No dynamic allocation after initialization
- **No Fragmentation**: Contiguous memory block
- **Cache-Friendly**: Sequential access pattern within buffer
- **Space Overhead**: Just 2-3 integers (head, tail, size)

## Real-World Applications

### 1. **Audio/Video Streaming**
```python
class AudioBuffer:
    """
    Circular buffer for audio streaming with jitter handling.
    """
    def __init__(self, sample_rate: int, duration_ms: int):
        samples = (sample_rate * duration_ms) // 1000
        self.buffer = CircularBuffer(samples)
        self.underrun_count = 0
        self.overrun_count = 0

    def write_samples(self, samples):
        """Producer writes audio samples."""
        for sample in samples:
            if self.buffer.is_full_buffer():
                self.overrun_count += 1
            self.buffer.enqueue(sample)

    def read_samples(self, count):
        """Consumer reads samples for playback."""
        samples = []
        for _ in range(count):
            if self.buffer.is_empty():
                self.underrun_count += 1
                samples.append(0)  # Silence
            else:
                samples.append(self.buffer.dequeue())
        return samples
```

### 2. **Producer-Consumer Pattern**
```python
import time
from threading import Thread

def producer(buffer, items):
    """Producer thread adds items to buffer."""
    for item in items:
        while not buffer.try_enqueue(item):
            time.sleep(0.001)  # Wait if buffer full
        print(f"Produced: {item}")
        time.sleep(0.1)

def consumer(buffer, count):
    """Consumer thread removes items from buffer."""
    consumed = 0
    while consumed < count:
        success, item = buffer.try_dequeue()
        if success:
            print(f"Consumed: {item}")
            consumed += 1
        else:
            time.sleep(0.001)  # Wait if buffer empty

# Example usage
buffer = ThreadSafeCircularBuffer(5)
items = list(range(10))

prod_thread = Thread(target=producer, args=(buffer, items))
cons_thread = Thread(target=consumer, args=(buffer, len(items)))

prod_thread.start()
cons_thread.start()
prod_thread.join()
cons_thread.join()
```

### 3. **Network Packet Buffer**
```python
class PacketBuffer:
    """
    Circular buffer for network packet processing.
    """
    def __init__(self, capacity: int):
        self.buffer = BlockingCircularBuffer(capacity)
        self.dropped_packets = 0
        self.total_packets = 0

    def receive_packet(self, packet):
        """Receive incoming packet."""
        self.total_packets += 1
        try:
            self.buffer.enqueue(packet)
        except OverflowError:
            self.dropped_packets += 1
            print(f"Packet dropped! Drop rate: "
                  f"{self.dropped_packets/self.total_packets:.2%}")

    def process_packets(self, max_count: int):
        """Process up to max_count packets."""
        processed = 0
        while processed < max_count and not self.buffer.is_empty():
            packet = self.buffer.dequeue()
            self._process_packet(packet)
            processed += 1
        return processed

    def _process_packet(self, packet):
        """Process single packet (implementation specific)."""
        pass
```

### 4. **Embedded Systems: Sensor Data**
```python
class SensorDataBuffer:
    """
    Memory-efficient circular buffer for embedded sensor data.
    """
    def __init__(self, capacity: int):
        self.buffer = CircularBuffer(capacity)
        self.sample_count = 0

    def add_reading(self, value: float):
        """Add sensor reading."""
        self.buffer.enqueue((self.sample_count, value))
        self.sample_count += 1

    def get_recent_readings(self, n: int):
        """Get n most recent readings."""
        readings = []
        temp_buffer = []

        # Extract readings
        count = min(n, len(self.buffer))
        for _ in range(count):
            reading = self.buffer.dequeue()
            readings.append(reading)
            temp_buffer.append(reading)

        # Restore buffer
        for reading in temp_buffer:
            self.buffer.enqueue(reading)

        return readings

    def calculate_moving_average(self, window: int) -> float:
        """Calculate moving average over window."""
        readings = self.get_recent_readings(window)
        if not readings:
            return 0.0
        return sum(r[1] for r in readings) / len(readings)
```

### 5. **Log Buffer with Rotation**
```python
import datetime

class RotatingLogBuffer:
    """
    Circular buffer for log messages with automatic rotation.
    """
    def __init__(self, capacity: int):
        self.buffer = CircularBuffer(capacity)

    def log(self, level: str, message: str):
        """Add log entry with timestamp."""
        timestamp = datetime.datetime.now()
        entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.buffer.enqueue(entry)

    def get_logs(self, level: str = None):
        """Get all logs, optionally filtered by level."""
        logs = []
        temp = []

        while not self.buffer.is_empty():
            entry = self.buffer.dequeue()
            temp.append(entry)
            if level is None or entry['level'] == level:
                logs.append(entry)

        # Restore buffer
        for entry in temp:
            self.buffer.enqueue(entry)

        return logs

    def info(self, message: str):
        self.log('INFO', message)

    def warning(self, message: str):
        self.log('WARNING', message)

    def error(self, message: str):
        self.log('ERROR', message)
```

## LeetCode Problems

### Direct Applications

1. **[622. Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)** - Medium
   - **Pattern**: Basic circular buffer implementation
   - **Key**: Track head, tail, and size with modulo arithmetic

2. **[641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/)** - Medium
   - **Pattern**: Bidirectional circular buffer
   - **Key**: Support insertions/deletions at both ends

3. **[346. Moving Average from Data Stream](https://leetcode.com/problems/moving-average-from-data-stream/)** - Easy
   - **Pattern**: Fixed-size window statistics
   - **Key**: Use circular buffer to maintain window

### Related Problems

4. **[933. Number of Recent Calls](https://leetcode.com/problems/number-of-recent-calls/)** - Easy
   - **Pattern**: Time-based windowing
   - **Key**: Remove old entries, count recent ones

5. **[1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)** - Medium
   - **Pattern**: Sliding window with min/max tracking
   - **Key**: Circular buffer-like deque operations

6. **[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)** - Hard
   - **Pattern**: Fixed-size window queries
   - **Key**: Efficient removal of oldest elements

## Interview Patterns & Tips

### Common Interview Questions

1. **"Implement a circular buffer with fixed capacity"**
   - Focus on modulo arithmetic for wrap-around
   - Discuss overflow strategies (overwrite vs. block)
   - Mention thread-safety considerations

2. **"Design a rate limiter using a circular buffer"**
   - Use buffer to track recent requests
   - Remove expired timestamps
   - Count valid entries in window

3. **"How would you implement a lock-free queue?"**
   - Single producer/single consumer circular buffer
   - Atomic operations on head/tail
   - Memory barriers for visibility

### Key Concepts to Explain

**1. Wrap-Around Logic:**
```python
# Always use modulo for circular indexing
next_index = (current_index + 1) % capacity

# Alternative: conditional
next_index = current_index + 1
if next_index >= capacity:
    next_index = 0
```

**2. Full vs. Empty Detection:**
```python
# Method 1: Track size separately
is_empty = (size == 0)
is_full = (size == capacity)

# Method 2: Use flag
is_full = (tail == head) and flag
is_empty = (tail == head) and not flag

# Method 3: Waste one slot
is_full = ((tail + 1) % capacity == head)
is_empty = (tail == head)
```

**3. Overwrite vs. Block Behavior:**
- **Overwrite**: Move head forward when full (data loss acceptable)
- **Block**: Raise exception or return false (data preservation required)
- **Choice depends on use case**: Logs can overwrite, queues should block

### Common Pitfalls

1. **Off-by-One Errors**: Carefully handle wrap-around at capacity boundary
2. **Full/Empty Confusion**: When head == tail, buffer could be full or empty
3. **Integer Overflow**: In long-running systems, counters may overflow
4. **Thread Safety**: Without proper synchronization, concurrent access breaks buffer
5. **Memory Leaks**: Clear references when dequeuing (especially in garbage-collected languages)

### Performance Considerations

**Advantages:**
- O(1) enqueue/dequeue operations
- No dynamic memory allocation
- Cache-friendly due to contiguous memory
- Predictable performance (no garbage collection pauses)

**When to Use:**
- Fixed maximum size is acceptable
- Predictable memory usage required
- High-performance producer-consumer scenarios
- Embedded systems with limited memory
- Real-time systems requiring deterministic behavior

**When Not to Use:**
- Need unbounded growth
- Random access to all elements required
- Priority-based processing needed
- Complex reordering operations

## Comparison with Alternatives

| Feature | Circular Buffer | Dynamic Queue | Linked List | Array Deque |
|---------|----------------|---------------|-------------|-------------|
| **Enqueue Time** | O(1) | O(1) amortized | O(1) | O(1) amortized |
| **Dequeue Time** | O(1) | O(1) | O(1) | O(1) |
| **Memory** | Fixed O(n) | Variable O(n) | O(n) + pointers | Variable O(n) |
| **Cache Efficiency** | Excellent | Good | Poor | Good |
| **Resizable** | No* | Yes | Yes | Yes |
| **Memory Overhead** | Minimal | Moderate | High | Moderate |
| **Predictability** | Perfect | Good | Poor | Good |
| **Thread-Safe** | Can be lock-free | Needs locks | Needs locks | Needs locks |

*Unless implemented as resizable variant

## Advanced Topics

### 1. Lock-Free Multi-Producer Multi-Consumer

For multiple producers/consumers, use atomic compare-and-swap (CAS):

```python
import threading

class MPMCCircularBuffer:
    """
    Lock-free circular buffer for multiple producers/consumers.
    Note: This is a simplified version. Production code needs memory barriers.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.enqueue_pos = 0
        self.dequeue_pos = 0
        self.lock = threading.Lock()  # Simplified version uses lock

    def try_enqueue(self, item):
        """Thread-safe enqueue using lock."""
        with self.lock:
            next_pos = (self.enqueue_pos + 1) % self.capacity
            if next_pos == self.dequeue_pos:
                return False  # Full

            self.buffer[self.enqueue_pos] = item
            self.enqueue_pos = next_pos
            return True

    def try_dequeue(self):
        """Thread-safe dequeue using lock."""
        with self.lock:
            if self.enqueue_pos == self.dequeue_pos:
                return False, None  # Empty

            item = self.buffer[self.dequeue_pos]
            self.dequeue_pos = (self.dequeue_pos + 1) % self.capacity
            return True, item
```

### 2. Memory-Mapped Circular Buffer

For IPC (inter-process communication):

```python
import mmap
import struct

class MemoryMappedCircularBuffer:
    """
    Circular buffer using memory-mapped file for IPC.
    """
    def __init__(self, filename: str, capacity: int):
        self.capacity = capacity
        self.item_size = 8  # 8 bytes per item (long long)

        # Header: head (8 bytes) + tail (8 bytes)
        header_size = 16
        total_size = header_size + (capacity * self.item_size)

        # Create memory-mapped file
        with open(filename, 'wb') as f:
            f.write(b'\x00' * total_size)

        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), total_size)

    def enqueue(self, value: int):
        """Write integer to shared buffer."""
        head = struct.unpack('Q', self.mmap[0:8])[0]
        tail = struct.unpack('Q', self.mmap[8:16])[0]

        next_tail = (tail + 1) % self.capacity
        if next_tail == head:
            raise OverflowError("Buffer full")

        offset = 16 + (tail * self.item_size)
        self.mmap[offset:offset+8] = struct.pack('Q', value)
        self.mmap[8:16] = struct.pack('Q', next_tail)

    def dequeue(self) -> int:
        """Read integer from shared buffer."""
        head = struct.unpack('Q', self.mmap[0:8])[0]
        tail = struct.unpack('Q', self.mmap[8:16])[0]

        if head == tail:
            raise IndexError("Buffer empty")

        offset = 16 + (head * self.item_size)
        value = struct.unpack('Q', self.mmap[offset:offset+8])[0]

        next_head = (head + 1) % self.capacity
        self.mmap[0:8] = struct.pack('Q', next_head)

        return value

    def close(self):
        """Close memory-mapped file."""
        self.mmap.close()
        self.file.close()
```

### 3. Power-of-Two Capacity Optimization

Using power-of-two capacity allows bitwise AND instead of modulo:

```python
class OptimizedCircularBuffer:
    """
    Circular buffer optimized with power-of-2 capacity.
    """
    def __init__(self, capacity_bits: int):
        """
        Initialize with capacity = 2^capacity_bits.
        Example: capacity_bits=10 gives capacity=1024
        """
        self.capacity = 1 << capacity_bits  # 2^capacity_bits
        self.mask = self.capacity - 1  # For bitwise AND
        self.buffer = [None] * self.capacity
        self.head = 0
        self.tail = 0

    def enqueue(self, item):
        """Enqueue using bitwise AND instead of modulo."""
        next_tail = (self.tail + 1) & self.mask  # Same as % capacity
        if next_tail == self.head:
            raise OverflowError("Buffer full")

        self.buffer[self.tail] = item
        self.tail = next_tail

    def dequeue(self):
        """Dequeue using bitwise AND."""
        if self.head == self.tail:
            raise IndexError("Buffer empty")

        item = self.buffer[self.head]
        self.head = (self.head + 1) & self.mask
        return item
```

## Summary

**Circular Buffer is ideal when:**
- Fixed maximum size is acceptable
- Need O(1) enqueue/dequeue operations
- Predictable memory usage is critical
- Building producer-consumer systems
- Working in embedded/real-time environments
- Old data can be overwritten (in overwrite mode)
- High cache efficiency is important

**Avoid Circular Buffer when:**
- Need dynamic, unbounded growth
- Require random access to all elements
- Need complex reordering or priority queues
- Can't tolerate data loss (and buffer size is uncertain)

**Key Takeaway**: Circular buffers are the go-to choice for fixed-size FIFO queues where performance and memory predictability matter. They shine in streaming applications, embedded systems, and high-performance producer-consumer scenarios.
