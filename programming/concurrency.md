# Concurrency Programming Guide

A comprehensive guide to concurrent programming, covering fundamentals, synchronization primitives, patterns, and practical implementations across multiple programming languages.

## Table of Contents

1. [Fundamentals](#fundamentals)
2. [Synchronization Primitives](#synchronization-primitives)
3. [Concurrency Patterns](#concurrency-patterns)
4. [Language-Specific Implementations](#language-specific-implementations)
5. [Deadlock Prevention](#deadlock-prevention)
6. [Performance Considerations](#performance-considerations)
7. [Real-World Applications](#real-world-applications)
8. [Best Practices](#best-practices)
9. [Anti-Patterns](#anti-patterns)
10. [Debugging Concurrent Programs](#debugging-concurrent-programs)
11. [Testing Concurrent Code](#testing-concurrent-code)

---

## Fundamentals

### Concurrency vs Parallelism

**Concurrency** and **parallelism** are related but distinct concepts:

**Concurrency:**
- Dealing with multiple tasks at once
- Tasks make progress by interleaving execution
- Can occur on a single-core processor through time-slicing
- About the *structure* of the program
- Example: A single person juggling multiple tasks by switching between them

**Parallelism:**
- Executing multiple tasks simultaneously
- Requires multiple processing units (cores)
- Tasks execute at the exact same time
- About the *execution* of the program
- Example: Multiple people each working on different tasks simultaneously

```
Concurrency:        |----Task A----|----Task A----|
                         |----Task B----|----Task B----|
                    (Interleaved on single core)

Parallelism:        |----Task A----|----Task A----|
                    |----Task B----|----Task B----|
                    (Simultaneous on multiple cores)
```

**Key Insight:** A program can be concurrent but not parallel (one core, multiple tasks interleaved), parallel but not concurrent (two independent single-threaded programs), or both concurrent and parallel (multi-threaded program on multi-core system).

### Processes vs Threads

#### Processes

**Definition:** A process is an instance of a running program with its own memory space.

**Characteristics:**
- **Isolated memory:** Each process has its own address space
- **Heavy-weight:** High overhead to create and context switch
- **Independent:** Crash in one process doesn't affect others
- **Communication:** Inter-Process Communication (IPC) required (pipes, sockets, shared memory)
- **Security:** Strong isolation boundaries

**When to use:**
- Need strong isolation
- Running untrusted code
- Want to leverage multiple cores without shared memory complexity
- Fault tolerance is critical

```python
# Python multiprocessing example
import multiprocessing
import os

def worker(num):
    print(f"Worker {num}, PID: {os.getpid()}")
    return num * num

if __name__ == '__main__':
    # Create separate processes
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(worker, range(10))
    print(results)
```

#### Threads

**Definition:** A thread is a lightweight execution unit within a process that shares the process's memory.

**Characteristics:**
- **Shared memory:** All threads share the same address space
- **Light-weight:** Lower overhead to create and context switch
- **Dependent:** Crash in one thread can crash the entire process
- **Communication:** Direct memory sharing (requires synchronization)
- **Speed:** Faster context switching than processes

**When to use:**
- Need fast communication between concurrent tasks
- Sharing large amounts of data
- I/O-bound operations
- GUI applications (event handling)

```python
# Python threading example
import threading

def worker(num):
    print(f"Worker {num}, Thread: {threading.current_thread().name}")
    return num * num

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

**Process vs Thread Comparison:**

| Aspect | Process | Thread |
|--------|---------|--------|
| Memory | Separate address space | Shared address space |
| Creation overhead | High (~1-10ms) | Low (~10-100μs) |
| Context switch | Expensive | Cheap |
| Communication | IPC (slow) | Direct (fast) |
| Isolation | Strong | Weak |
| Crash impact | Isolated | Affects all threads |

### Context Switching

**Definition:** Context switching is the process of storing and restoring the state of a thread or process so execution can resume from the same point later.

**What gets saved/restored:**
- Program counter (PC)
- CPU registers
- Stack pointer
- Memory management information
- I/O status

**Cost of context switching:**
1. **Direct costs:**
   - Saving/restoring registers
   - Updating kernel data structures
   - Time: ~1-10 microseconds

2. **Indirect costs:**
   - Cache pollution (cold cache after switch)
   - TLB (Translation Lookaside Buffer) misses
   - Pipeline stalls
   - Can be 10-100x the direct cost

**Example scenario:**
```
Thread A running -> Interrupt/yield -> Save Thread A state
                                    -> Load Thread B state
Thread B running -> Interrupt/yield -> Save Thread B state
                                    -> Load Thread A state
Thread A resumes
```

**Minimizing context switches:**
- Reduce number of threads (use thread pools)
- Minimize lock contention
- Use asynchronous I/O
- Batch operations
- Set appropriate thread affinity

### Race Conditions

**Definition:** A race condition occurs when the program's behavior depends on the relative timing or interleaving of multiple threads or processes.

**Classic example - Bank account:**

```python
# UNSAFE: Race condition
class BankAccount:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        # This is NOT atomic!
        current = self.balance  # Read
        current += amount       # Modify
        self.balance = current  # Write

# Two threads depositing simultaneously
account = BankAccount()

# Thread 1: deposit(100)
# Thread 2: deposit(50)

# Possible execution:
# T1: current = balance  (reads 0)
# T2: current = balance  (reads 0)
# T1: current += 100     (current = 100)
# T2: current += 50      (current = 50)
# T1: balance = current  (balance = 100)
# T2: balance = current  (balance = 50)
# Final balance: 50 (WRONG! Should be 150)
```

**Types of race conditions:**

1. **Data race:** Multiple threads access shared data, at least one writes, without synchronization
2. **Read-modify-write:** Classic race (shown above)
3. **Check-then-act:** Checking a condition then acting on it

```python
# Check-then-act race condition
if file_exists("config.txt"):  # Check
    data = read_file("config.txt")  # Act (file might be deleted between check and read)
```

**Detecting race conditions:**
- Dynamic analysis tools (ThreadSanitizer, Helgrind, Intel Inspector)
- Static analysis
- Stress testing with many threads
- Code review focusing on shared mutable state

**Fixing race conditions:**
- Use synchronization primitives (locks, atomics)
- Eliminate shared mutable state
- Use immutable data structures
- Message passing instead of shared memory

### Deadlocks

**Definition:** A deadlock is a situation where two or more threads are blocked forever, each waiting for resources held by the other.

**Classic example - Dining Philosophers:**

```python
import threading

fork1 = threading.Lock()
fork2 = threading.Lock()

def philosopher1():
    fork1.acquire()  # Got fork 1
    # Context switch!
    fork2.acquire()  # Waiting for fork 2... (held by philosopher2)
    print("Philosopher 1 eating")
    fork2.release()
    fork1.release()

def philosopher2():
    fork2.acquire()  # Got fork 2
    # Context switch!
    fork1.acquire()  # Waiting for fork 1... (held by philosopher1)
    print("Philosopher 2 eating")
    fork1.release()
    fork2.release()

# DEADLOCK: philosopher1 has fork1, wants fork2
#           philosopher2 has fork2, wants fork1
#           Both wait forever
```

**Resource deadlock example:**

```
Thread A:          Thread B:
lock(mutex1)       lock(mutex2)
  lock(mutex2)       lock(mutex1)  <-- DEADLOCK
    ...                ...
  unlock(mutex2)     unlock(mutex1)
unlock(mutex1)     unlock(mutex2)
```

**Four necessary conditions for deadlock (Coffman conditions):**

1. **Mutual Exclusion:** Resources cannot be shared
2. **Hold and Wait:** Thread holds resources while waiting for others
3. **No Preemption:** Resources cannot be forcibly taken away
4. **Circular Wait:** Circular chain of threads, each waiting for a resource held by the next

**All four conditions must be present for deadlock to occur.**

**Prevention strategies:**
- Break one of the four conditions
- Lock ordering (always acquire locks in same order)
- Lock timeout (try-lock with timeout)
- Deadlock detection and recovery

**Visualizing deadlock:**

```
    Thread A              Thread B
       |                     |
   Lock(R1) ✓               |
       |                 Lock(R2) ✓
       |                     |
   Lock(R2) ⏸              |
   (waiting...)             |
       |                 Lock(R1) ⏸
       |                 (waiting...)
       ↓                     ↓
     DEADLOCK - Both threads waiting forever
```

---

## Synchronization Primitives

Synchronization primitives are low-level constructs used to control access to shared resources and coordinate thread execution.

### Mutexes and Locks

**Mutex (Mutual Exclusion):** Ensures that only one thread can access a critical section at a time.

**Basic operations:**
- `lock()` / `acquire()`: Acquire the lock (block if held by another thread)
- `unlock()` / `release()`: Release the lock
- `trylock()`: Try to acquire without blocking (returns success/failure)

**Python example:**

```python
import threading

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:  # Acquire lock
            # Critical section
            current = self.value
            current += 1
            self.value = current
        # Lock released automatically

    def increment_manual(self):
        self.lock.acquire()
        try:
            self.value += 1
        finally:
            self.lock.release()  # Always release, even if exception

# Usage
counter = Counter()
threads = [threading.Thread(target=counter.increment) for _ in range(1000)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Final value: {counter.value}")  # Correctly prints 1000
```

**C++ example:**

```cpp
#include <mutex>
#include <thread>

class Counter {
private:
    int value = 0;
    std::mutex mtx;

public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);  // RAII
        value++;
    }  // Lock released automatically

    void increment_manual() {
        mtx.lock();
        value++;
        mtx.unlock();
    }

    bool try_increment() {
        if (mtx.try_lock()) {
            value++;
            mtx.unlock();
            return true;
        }
        return false;
    }

    int get_value() {
        std::lock_guard<std::mutex> lock(mtx);
        return value;
    }
};
```

**Rust example (using Mutex from std::sync):**

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        }); // Lock released when `num` goes out of scope
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

**Types of locks:**

1. **Spinlock:** Busy-waits in a loop checking the lock
   - Low latency for short critical sections
   - Wastes CPU cycles
   - Good for kernel-level code or when lock is held very briefly

2. **Mutex (blocking lock):** Puts thread to sleep when waiting
   - Higher latency (context switch overhead)
   - Doesn't waste CPU
   - Good for longer critical sections

3. **Recursive lock:** Can be locked multiple times by the same thread
   - Useful but can hide design issues
   - Higher overhead

```python
import threading

# Recursive lock example
lock = threading.RLock()  # Recursive lock

def recursive_function(n):
    with lock:
        if n > 0:
            print(n)
            recursive_function(n - 1)  # Can re-acquire same lock

recursive_function(5)
```

### Semaphores

**Semaphore:** A synchronization primitive that maintains a count, allowing a fixed number of threads to access a resource.

**Operations:**
- `wait()` / `P()` / `acquire()`: Decrement count (block if zero)
- `signal()` / `V()` / `release()`: Increment count

**Types:**

1. **Binary semaphore:** Count of 0 or 1 (similar to mutex)
2. **Counting semaphore:** Count can be any non-negative integer

**Python example:**

```python
import threading
import time

# Limit to 3 concurrent database connections
db_semaphore = threading.Semaphore(3)

def access_database(thread_id):
    print(f"Thread {thread_id} waiting for DB access")
    with db_semaphore:
        print(f"Thread {thread_id} accessing database")
        time.sleep(2)  # Simulate database operation
        print(f"Thread {thread_id} done with database")

threads = [threading.Thread(target=access_database, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

**C++ example:**

```cpp
#include <semaphore>
#include <thread>
#include <iostream>

std::counting_semaphore<3> db_semaphore(3);  // Max 3 concurrent accesses

void access_database(int id) {
    db_semaphore.acquire();
    std::cout << "Thread " << id << " accessing database\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Thread " << id << " done\n";
    db_semaphore.release();
}
```

**Classic use case - Producer-Consumer:**

```python
import threading
import queue
import time

# Using semaphores to implement producer-consumer
MAX_SIZE = 5
buffer = []
mutex = threading.Lock()
empty_slots = threading.Semaphore(MAX_SIZE)  # Initially MAX_SIZE
full_slots = threading.Semaphore(0)          # Initially 0

def producer(id):
    for i in range(10):
        item = f"Item-{id}-{i}"
        empty_slots.acquire()  # Wait for empty slot
        with mutex:
            buffer.append(item)
            print(f"Producer {id} produced {item}, buffer size: {len(buffer)}")
        full_slots.release()  # Signal item available
        time.sleep(0.1)

def consumer(id):
    for i in range(10):
        full_slots.acquire()  # Wait for item
        with mutex:
            item = buffer.pop(0)
            print(f"Consumer {id} consumed {item}, buffer size: {len(buffer)}")
        empty_slots.release()  # Signal empty slot
        time.sleep(0.15)

# Create producers and consumers
producers = [threading.Thread(target=producer, args=(i,)) for i in range(2)]
consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(2)]

for t in producers + consumers:
    t.start()
for t in producers + consumers:
    t.join()
```

### Condition Variables

**Condition Variable:** Allows threads to wait for a specific condition to become true, avoiding busy-waiting.

**Operations:**
- `wait()`: Release lock and sleep until signaled
- `notify()` / `signal()`: Wake up one waiting thread
- `notify_all()` / `broadcast()`: Wake up all waiting threads

**Python example:**

```python
import threading
import time

class BoundedQueue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:  # Acquires lock
            while len(self.queue) >= self.max_size:
                self.not_full.wait()  # Release lock and wait
            self.queue.append(item)
            self.not_empty.notify()  # Wake up a consumer

    def get(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()
            item = self.queue.pop(0)
            self.not_full.notify()  # Wake up a producer
            return item

# Usage
queue = BoundedQueue(5)

def producer(id):
    for i in range(10):
        item = f"P{id}-Item{i}"
        queue.put(item)
        print(f"Produced: {item}")
        time.sleep(0.1)

def consumer(id):
    for i in range(10):
        item = queue.get()
        print(f"Consumer {id} consumed: {item}")
        time.sleep(0.15)

threads = [
    threading.Thread(target=producer, args=(1,)),
    threading.Thread(target=consumer, args=(1,)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

**C++ example:**

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

template<typename T>
class BoundedQueue {
private:
    std::queue<T> queue;
    size_t max_size;
    std::mutex mtx;
    std::condition_variable not_empty;
    std::condition_variable not_full;

public:
    BoundedQueue(size_t size) : max_size(size) {}

    void put(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        not_full.wait(lock, [this] { return queue.size() < max_size; });
        queue.push(item);
        not_empty.notify_one();
    }

    T get() {
        std::unique_lock<std::mutex> lock(mtx);
        not_empty.wait(lock, [this] { return !queue.empty(); });
        T item = queue.front();
        queue.pop();
        not_full.notify_one();
        return item;
    }
};
```

**Important pattern - Wait in a loop:**

```python
# WRONG: Don't do this
with condition:
    if not predicate():
        condition.wait()
    # Proceed

# RIGHT: Always wait in a loop
with condition:
    while not predicate():
        condition.wait()
    # Proceed
```

**Why?** Spurious wakeups can occur (thread wakes up without being signaled), and multiple threads might be waiting.

### Read-Write Locks

**RWLock:** Allows multiple readers OR one writer (but not both simultaneously).

**Benefits:**
- Better concurrency for read-heavy workloads
- Multiple threads can read simultaneously
- Writes still get exclusive access

**Python example:**

```python
import threading

class RWLock:
    def __init__(self):
        self.readers = 0
        self.writer = False
        self.lock = threading.Lock()
        self.can_read = threading.Condition(self.lock)
        self.can_write = threading.Condition(self.lock)

    def acquire_read(self):
        with self.lock:
            while self.writer:
                self.can_read.wait()
            self.readers += 1

    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.can_write.notify()

    def acquire_write(self):
        with self.lock:
            while self.writer or self.readers > 0:
                self.can_write.wait()
            self.writer = True

    def release_write(self):
        with self.lock:
            self.writer = False
            self.can_write.notify()
            self.can_read.notify_all()

# Usage
class CachedData:
    def __init__(self):
        self.data = {}
        self.rwlock = RWLock()

    def read(self, key):
        self.rwlock.acquire_read()
        try:
            return self.data.get(key)
        finally:
            self.rwlock.release_read()

    def write(self, key, value):
        self.rwlock.acquire_write()
        try:
            self.data[key] = value
        finally:
            self.rwlock.release_write()
```

**Rust example (using std::sync::RwLock):**

```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];

    // Spawn readers
    for i in 0..5 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let r = data.read().unwrap();
            println!("Reader {}: {:?}", i, *r);
        });
        handles.push(handle);
    }

    // Spawn writer
    let data_clone = Arc::clone(&data);
    handles.push(thread::spawn(move || {
        let mut w = data_clone.write().unwrap();
        w.push(4);
        println!("Writer added element");
    }));

    for handle in handles {
        handle.join().unwrap();
    }
}
```

**C++ example:**

```cpp
#include <shared_mutex>
#include <thread>
#include <vector>

class CachedData {
private:
    std::map<std::string, int> data;
    mutable std::shared_mutex rwlock;

public:
    int read(const std::string& key) const {
        std::shared_lock lock(rwlock);  // Multiple readers OK
        auto it = data.find(key);
        return it != data.end() ? it->second : 0;
    }

    void write(const std::string& key, int value) {
        std::unique_lock lock(rwlock);  // Exclusive access
        data[key] = value;
    }
};
```

**Performance characteristics:**
- **Uncontended read:** Very fast (just increment counter)
- **Contended read:** Still fast (multiple readers allowed)
- **Write:** More expensive (must wait for all readers to finish)

**Use when:**
- Read operations are much more frequent than writes
- Read operations take significant time
- Data is large enough that read-only access is valuable

### Spinlocks

**Spinlock:** A lock that causes threads to busy-wait (spin) in a loop checking if the lock is available.

**Characteristics:**
- No context switching overhead
- Wastes CPU cycles while waiting
- Only suitable for very short critical sections
- Often used in kernel code

**Python example (conceptual - not recommended for Python):**

```python
import threading
import time

class Spinlock:
    def __init__(self):
        self.locked = False

    def acquire(self):
        while True:
            # Atomic test-and-set
            if not self.locked:
                self.locked = True
                break
            # Spin (busy-wait)

    def release(self):
        self.locked = False

# Note: Python's GIL makes spinlocks inefficient
# This is just for demonstration
```

**C++ example:**

```cpp
#include <atomic>

class Spinlock {
private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // Spin
        }
    }

    void unlock() {
        flag.clear(std::memory_order_release);
    }
};

// Usage
Spinlock spinlock;
int counter = 0;

void increment() {
    spinlock.lock();
    counter++;
    spinlock.unlock();
}
```

**Optimized spinlock with backoff:**

```cpp
#include <atomic>
#include <thread>

class BackoffSpinlock {
private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

public:
    void lock() {
        int backoff = 1;
        while (flag.test_and_set(std::memory_order_acquire)) {
            for (int i = 0; i < backoff; i++) {
                // Pause instruction (hint to CPU)
                std::this_thread::yield();
            }
            backoff = std::min(backoff * 2, 1024);  // Exponential backoff
        }
    }

    void unlock() {
        flag.clear(std::memory_order_release);
    }
};
```

**When to use spinlocks:**
- Critical section is very short (< 100 nanoseconds)
- Number of threads ≤ number of cores
- Real-time systems where latency is critical
- Kernel-level code where sleeping is not allowed

**When NOT to use spinlocks:**
- Critical section is long
- More threads than cores (causes CPU waste)
- User-space applications (use mutexes instead)

### Atomic Operations

**Atomic operation:** An operation that completes without interruption, appearing instantaneous to other threads.

**Common atomic operations:**
- Load
- Store
- Exchange (swap)
- Compare-and-swap (CAS)
- Fetch-and-add
- Fetch-and-subtract

**Python example:**

```python
import threading

# Python's += is NOT atomic (even for integers)
counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # NOT ATOMIC!

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter: {counter}")  # Will be < 1000000 due to race conditions

# To fix: use threading.Lock or atomic operations
```

**C++ atomic example:**

```cpp
#include <atomic>
#include <thread>
#include <vector>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; i++) {
        counter.fetch_add(1, std::memory_order_relaxed);
        // Or simply: counter++;  (atomic increment)
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment);
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Counter: " << counter << std::endl;  // Correctly prints 1000000
}
```

**Compare-and-swap (CAS) example:**

```cpp
#include <atomic>

std::atomic<int> value(0);

void increment_cas() {
    int expected = value.load();
    int desired;
    do {
        desired = expected + 1;
    } while (!value.compare_exchange_weak(expected, desired));
    // If value == expected, set value to desired and return true
    // Otherwise, load current value into expected and return false
}
```

**Rust atomic example:**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..100000 {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", counter.load(Ordering::Relaxed));
}
```

**Memory ordering (important for atomics):**

1. **Relaxed:** No synchronization, only atomicity guaranteed
2. **Acquire:** Prevents reordering of subsequent reads/writes before this operation
3. **Release:** Prevents reordering of prior reads/writes after this operation
4. **AcqRel:** Both acquire and release
5. **SeqCst:** Sequential consistency (strongest, most expensive)

```cpp
// Example: Producer-consumer with atomics
std::atomic<bool> data_ready(false);
int data;

// Producer thread
void produce() {
    data = 42;  // Write data
    data_ready.store(true, std::memory_order_release);  // Signal
}

// Consumer thread
void consume() {
    while (!data_ready.load(std::memory_order_acquire)) {
        // Wait
    }
    // Now safe to read data
    std::cout << data << std::endl;
}
```

**Lock-free counter using atomics:**

```cpp
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
    };
    std::atomic<Node*> head;

public:
    LockFreeStack() : head(nullptr) {}

    void push(T value) {
        Node* new_node = new Node{value, nullptr};
        new_node->next = head.load();
        while (!head.compare_exchange_weak(new_node->next, new_node)) {
            // Retry if another thread modified head
        }
    }

    bool pop(T& result) {
        Node* old_head = head.load();
        while (old_head &&
               !head.compare_exchange_weak(old_head, old_head->next)) {
            // Retry
        }
        if (old_head) {
            result = old_head->data;
            delete old_head;  // Note: ABA problem in production code!
            return true;
        }
        return false;
    }
};
```

---

## Concurrency Patterns

Common patterns for structuring concurrent programs.

### Producer-Consumer Pattern

**Problem:** Decouple production of data from its consumption.

**Components:**
- Producers: Generate data
- Consumers: Process data
- Buffer: Queue between producers and consumers

**Python implementation:**

```python
import threading
import queue
import time
import random

def producer(q, producer_id):
    for i in range(5):
        item = f"Item-{producer_id}-{i}"
        time.sleep(random.uniform(0.1, 0.5))
        q.put(item)
        print(f"Producer {producer_id} produced {item}")
    # Signal completion
    q.put(None)

def consumer(q, consumer_id):
    while True:
        item = q.get()
        if item is None:
            q.put(None)  # Pass signal to other consumers
            break
        print(f"Consumer {consumer_id} consumed {item}")
        time.sleep(random.uniform(0.1, 0.3))
        q.task_done()

# Create queue with max size
buffer = queue.Queue(maxsize=10)

# Create and start threads
producers = [threading.Thread(target=producer, args=(buffer, i)) for i in range(2)]
consumers = [threading.Thread(target=consumer, args=(buffer, i)) for i in range(3)]

for t in producers + consumers:
    t.start()

for t in producers + consumers:
    t.join()

print("All done!")
```

**Go implementation:**

```go
package main

import (
    "fmt"
    "math/rand"
    "sync"
    "time"
)

func producer(ch chan<- int, id int, wg *sync.WaitGroup) {
    defer wg.Done()
    for i := 0; i < 5; i++ {
        item := id*100 + i
        time.Sleep(time.Duration(rand.Intn(500)) * time.Millisecond)
        ch <- item
        fmt.Printf("Producer %d produced %d\n", id, item)
    }
}

func consumer(ch <-chan int, id int, wg *sync.WaitGroup) {
    defer wg.Done()
    for item := range ch {
        fmt.Printf("Consumer %d consumed %d\n", id, item)
        time.Sleep(time.Duration(rand.Intn(300)) * time.Millisecond)
    }
}

func main() {
    buffer := make(chan int, 10)  // Buffered channel
    var producerWg, consumerWg sync.WaitGroup

    // Start producers
    for i := 0; i < 2; i++ {
        producerWg.Add(1)
        go producer(buffer, i, &producerWg)
    }

    // Start consumers
    for i := 0; i < 3; i++ {
        consumerWg.Add(1)
        go consumer(buffer, i, &consumerWg)
    }

    // Wait for producers to finish, then close channel
    go func() {
        producerWg.Wait()
        close(buffer)
    }()

    // Wait for consumers
    consumerWg.Wait()
    fmt.Println("All done!")
}
```

### Reader-Writer Pattern

**Problem:** Multiple readers can access data simultaneously, but writers need exclusive access.

**Implementation using RWLock:**

```python
import threading
import time

class SharedResource:
    def __init__(self):
        self.data = []
        self.rwlock = threading.Lock()  # Simple version
        # In production, use a proper RWLock implementation

    def read_data(self, reader_id):
        # Multiple readers can hold this
        print(f"Reader {reader_id} reading: {self.data}")
        time.sleep(0.1)

    def write_data(self, writer_id, value):
        with self.rwlock:
            print(f"Writer {writer_id} writing {value}")
            self.data.append(value)
            time.sleep(0.2)

# Usage
resource = SharedResource()

def reader(resource, id):
    for _ in range(3):
        resource.read_data(id)
        time.sleep(0.05)

def writer(resource, id):
    for i in range(2):
        resource.write_data(id, f"Data-{id}-{i}")
        time.sleep(0.1)

threads = []
threads.extend([threading.Thread(target=reader, args=(resource, i)) for i in range(5)])
threads.extend([threading.Thread(target=writer, args=(resource, i)) for i in range(2)])

for t in threads:
    t.start()
for t in threads:
    t.join()
```

### Thread Pool Pattern

**Problem:** Creating threads is expensive; reuse a fixed pool of threads for tasks.

**Python implementation:**

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    print(f"Processing task {n}")
    time.sleep(1)
    return n * n

# Create thread pool with 4 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks
    futures = [executor.submit(task, i) for i in range(10)]

    # Get results as they complete
    for future in futures:
        result = future.result()
        print(f"Result: {result}")

# Alternative: map operation
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(task, range(10))
    for result in results:
        print(f"Result: {result}")
```

**Custom thread pool implementation:**

```python
import threading
import queue

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _worker(self):
        while True:
            func, args, kwargs = self.tasks.get()
            if func is None:
                break
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in task: {e}")
            finally:
                self.tasks.task_done()

    def submit(self, func, *args, **kwargs):
        self.tasks.put((func, args, kwargs))

    def wait_completion(self):
        self.tasks.join()

    def shutdown(self):
        for _ in self.threads:
            self.tasks.put((None, None, None))
        for t in self.threads:
            t.join()

# Usage
pool = ThreadPool(4)
for i in range(10):
    pool.submit(task, i)
pool.wait_completion()
pool.shutdown()
```

**Java ExecutorService:**

```java
import java.util.concurrent.*;

public class ThreadPoolExample {
    public static void main(String[] args) throws InterruptedException {
        // Create thread pool
        ExecutorService executor = Executors.newFixedThreadPool(4);

        // Submit tasks
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("Task " + taskId + " running");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                return taskId * taskId;
            });
        }

        // Shutdown
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
    }
}
```

### Future/Promise Pattern

**Problem:** Represent a value that will be available in the future, allowing asynchronous computation.

**Python Future example:**

```python
from concurrent.futures import ThreadPoolExecutor
import time

def slow_computation(n):
    time.sleep(2)
    return n * n

executor = ThreadPoolExecutor(max_workers=4)

# Submit computation, get Future object immediately
future = executor.submit(slow_computation, 5)

print("Computation started, doing other work...")
time.sleep(1)
print("Still doing other work...")

# Block until result is ready
result = future.result()  # Blocks here
print(f"Result: {result}")

# Check if done without blocking
future2 = executor.submit(slow_computation, 10)
if future2.done():
    print("Already done!")
else:
    print("Still computing...")
    future2.add_done_callback(lambda f: print(f"Result: {f.result()}"))

executor.shutdown()
```

**JavaScript Promise:**

```javascript
// Creating a Promise
function slowComputation(n) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (n < 0) {
                reject(new Error("Negative number"));
            } else {
                resolve(n * n);
            }
        }, 2000);
    });
}

// Using Promise
slowComputation(5)
    .then(result => {
        console.log("Result:", result);
        return slowComputation(result);
    })
    .then(result => {
        console.log("Second result:", result);
    })
    .catch(error => {
        console.error("Error:", error);
    });

// Multiple Promises
Promise.all([
    slowComputation(2),
    slowComputation(3),
    slowComputation(4)
]).then(results => {
    console.log("All results:", results);
});
```

**Rust Future:**

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

struct SlowComputation {
    value: i32,
}

impl Future for SlowComputation {
    type Output = i32;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Simulate async work
        Poll::Ready(self.value * self.value)
    }
}

// Using async/await (built on Futures)
async fn compute(n: i32) -> i32 {
    // Simulate slow computation
    n * n
}

#[tokio::main]
async fn main() {
    let result = compute(5).await;
    println!("Result: {}", result);

    // Multiple concurrent futures
    let (r1, r2, r3) = tokio::join!(
        compute(2),
        compute(3),
        compute(4)
    );
    println!("Results: {}, {}, {}", r1, r2, r3);
}
```

### Async/Await Pattern

**Problem:** Write asynchronous code that looks synchronous, avoiding callback hell.

**Python asyncio:**

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    print(f"Fetching {url}")
    async with session.get(url) as response:
        data = await response.text()
        print(f"Got {len(data)} bytes from {url}")
        return data

async def main():
    urls = [
        'http://example.com',
        'http://example.org',
        'http://example.net'
    ]

    async with aiohttp.ClientSession() as session:
        # Concurrent execution
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    print(f"Fetched {len(results)} URLs")

# Run
asyncio.run(main())
```

**JavaScript async/await:**

```javascript
// Async function
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const user = await response.json();

        // Sequential awaits
        const posts = await fetch(`/api/users/${userId}/posts`);
        const postsData = await posts.json();

        return { user, posts: postsData };
    } catch (error) {
        console.error("Error fetching user data:", error);
        throw error;
    }
}

// Concurrent execution
async function fetchMultipleUsers(userIds) {
    const promises = userIds.map(id => fetchUserData(id));
    const results = await Promise.all(promises);
    return results;
}

// Usage
fetchMultipleUsers([1, 2, 3])
    .then(users => console.log("Users:", users))
    .catch(error => console.error("Error:", error));
```

**C# async/await:**

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

class Program {
    static async Task<string> FetchUrlAsync(string url) {
        using (HttpClient client = new HttpClient()) {
            Console.WriteLine($"Fetching {url}");
            string content = await client.GetStringAsync(url);
            Console.WriteLine($"Got {content.Length} bytes");
            return content;
        }
    }

    static async Task Main(string[] args) {
        var urls = new[] {
            "http://example.com",
            "http://example.org",
            "http://example.net"
        };

        // Concurrent execution
        var tasks = Array.ConvertAll(urls, url => FetchUrlAsync(url));
        var results = await Task.WhenAll(tasks);

        Console.WriteLine($"Fetched {results.Length} URLs");
    }
}
```

### Pipeline Pattern

**Problem:** Process data through a series of stages, each running concurrently.

**Go pipeline:**

```go
package main

import "fmt"

// Stage 1: Generate numbers
func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

// Stage 2: Square numbers
func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// Stage 3: Filter even numbers
func filterEven(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if n%2 == 0 {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

func main() {
    // Build pipeline
    c := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    c = square(c)
    c = filterEven(c)

    // Consume results
    for result := range c {
        fmt.Println(result)
    }
}
```

**Python pipeline:**

```python
import queue
import threading

def pipeline_stage(input_queue, output_queue, transform):
    while True:
        item = input_queue.get()
        if item is None:
            output_queue.put(None)
            break
        result = transform(item)
        if result is not None:
            output_queue.put(result)
        input_queue.task_done()

# Create queues
q1 = queue.Queue()
q2 = queue.Queue()
q3 = queue.Queue()

# Create stages
stage1 = threading.Thread(target=pipeline_stage, args=(q1, q2, lambda x: x * x))
stage2 = threading.Thread(target=pipeline_stage, args=(q2, q3, lambda x: x if x % 2 == 0 else None))

stage1.start()
stage2.start()

# Feed input
for i in range(1, 11):
    q1.put(i)
q1.put(None)

# Consume output
while True:
    item = q3.get()
    if item is None:
        break
    print(item)

stage1.join()
stage2.join()
```

---

## Language-Specific Implementations

### Python

Python's concurrency model is unique due to the **Global Interpreter Lock (GIL)**.

#### Global Interpreter Lock (GIL)

**What is it?** A mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

**Implications:**
- **CPU-bound tasks:** Multithreading doesn't help (only one thread executes at a time)
- **I/O-bound tasks:** Multithreading works well (threads release GIL during I/O)
- **Multiprocessing:** Required for true parallelism in CPU-bound tasks

**Example showing GIL impact:**

```python
import threading
import time

def cpu_bound():
    count = 0
    for i in range(50_000_000):
        count += 1
    return count

# Single-threaded
start = time.time()
cpu_bound()
cpu_bound()
print(f"Single-threaded: {time.time() - start:.2f}s")

# Multi-threaded (doesn't help due to GIL)
start = time.time()
t1 = threading.Thread(target=cpu_bound)
t2 = threading.Thread(target=cpu_bound)
t1.start()
t2.start()
t1.join()
t2.join()
print(f"Multi-threaded: {time.time() - start:.2f}s")  # Similar time!

# Multi-processing (true parallelism)
import multiprocessing
start = time.time()
p1 = multiprocessing.Process(target=cpu_bound)
p2 = multiprocessing.Process(target=cpu_bound)
p1.start()
p2.start()
p1.join()
p2.join()
print(f"Multi-processing: {time.time() - start:.2f}s")  # Faster!
```

#### Threading Module

```python
import threading
import time

# Basic thread creation
def worker(name, delay):
    print(f"{name} starting")
    time.sleep(delay)
    print(f"{name} finished")

t = threading.Thread(target=worker, args=("Thread-1", 2))
t.start()
t.join()

# Thread with return value
from concurrent.futures import ThreadPoolExecutor

def compute(x):
    return x * x

with ThreadPoolExecutor() as executor:
    future = executor.submit(compute, 5)
    result = future.result()
    print(f"Result: {result}")

# Thread-local storage
thread_local = threading.local()

def process():
    if not hasattr(thread_local, 'value'):
        thread_local.value = threading.current_thread().name
    print(f"Thread {thread_local.value} processing")

threads = [threading.Thread(target=process) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

#### Multiprocessing Module

```python
import multiprocessing
import os

def worker(num):
    print(f"Worker {num}, PID: {os.getpid()}")
    return num * num

if __name__ == '__main__':
    # Process pool
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(worker, range(10))
        print(results)

    # Shared memory
    shared_value = multiprocessing.Value('i', 0)
    shared_array = multiprocessing.Array('d', [1.0, 2.0, 3.0])

    def increment(val):
        with val.get_lock():
            val.value += 1

    processes = [multiprocessing.Process(target=increment, args=(shared_value,)) for _ in range(10)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"Final value: {shared_value.value}")

    # Queue for communication
    queue = multiprocessing.Queue()

    def producer(q):
        for i in range(5):
            q.put(i)

    def consumer(q):
        while True:
            item = q.get()
            if item is None:
                break
            print(f"Consumed: {item}")

    p1 = multiprocessing.Process(target=producer, args=(queue,))
    p2 = multiprocessing.Process(target=consumer, args=(queue,))

    p1.start()
    p2.start()
    p1.join()
    queue.put(None)
    p2.join()
```

#### AsyncIO

```python
import asyncio
import time

# Basic async function
async def say_hello(name, delay):
    await asyncio.sleep(delay)
    print(f"Hello, {name}!")

# Run async function
asyncio.run(say_hello("World", 1))

# Multiple concurrent tasks
async def main():
    await asyncio.gather(
        say_hello("Alice", 1),
        say_hello("Bob", 2),
        say_hello("Charlie", 1.5)
    )

asyncio.run(main())

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)

async def use_resource():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(use_resource())

# Async generator
async def async_range(count):
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i

async def consume():
    async for i in async_range(5):
        print(i)

asyncio.run(consume())

# Running blocking code in executor
import concurrent.futures

def blocking_io():
    time.sleep(1)
    return "Done"

async def main():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_io)
    print(result)

asyncio.run(main())
```

### JavaScript

JavaScript uses an **event loop** for concurrency, running on a single thread.

#### Event Loop

```javascript
console.log("1");

setTimeout(() => {
    console.log("2");
}, 0);

Promise.resolve().then(() => {
    console.log("3");
});

console.log("4");

// Output: 1, 4, 3, 2
// Explanation:
// - Synchronous code runs first: 1, 4
// - Microtasks (Promises) run next: 3
// - Macrotasks (setTimeout) run last: 2
```

**Event loop phases:**
1. **Call stack:** Synchronous code
2. **Microtask queue:** Promises, process.nextTick (Node.js)
3. **Macrotask queue:** setTimeout, setInterval, I/O

#### Async/Await

```javascript
// Async function always returns a Promise
async function fetchData() {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
}

// Error handling
async function fetchWithErrorHandling() {
    try {
        const data = await fetchData();
        console.log(data);
    } catch (error) {
        console.error("Error:", error);
    }
}

// Parallel execution
async function fetchMultiple() {
    const [user, posts, comments] = await Promise.all([
        fetch('/api/user').then(r => r.json()),
        fetch('/api/posts').then(r => r.json()),
        fetch('/api/comments').then(r => r.json())
    ]);

    return { user, posts, comments };
}

// Race condition
async function fetchWithTimeout(url, timeout) {
    const fetchPromise = fetch(url);
    const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Timeout')), timeout)
    );

    return Promise.race([fetchPromise, timeoutPromise]);
}
```

#### Web Workers

```javascript
// main.js - Main thread
const worker = new Worker('worker.js');

// Send message to worker
worker.postMessage({ data: [1, 2, 3, 4, 5] });

// Receive message from worker
worker.onmessage = function(event) {
    console.log("Result from worker:", event.data);
};

worker.onerror = function(error) {
    console.error("Worker error:", error);
};

// Terminate worker
// worker.terminate();

// worker.js - Worker thread
self.onmessage = function(event) {
    const data = event.data.data;

    // Perform heavy computation
    const result = data.map(x => x * x);

    // Send result back
    self.postMessage(result);
};
```

**SharedArrayBuffer (advanced):**

```javascript
// main.js
const shared = new SharedArrayBuffer(16);
const view = new Int32Array(shared);

const worker = new Worker('worker.js');
worker.postMessage(shared);

// Atomic operations
Atomics.store(view, 0, 123);
console.log(Atomics.load(view, 0));

// Wait/notify
Atomics.wait(view, 0, 123);  // Wait until value at index 0 is not 123

// worker.js
self.onmessage = function(event) {
    const shared = event.data;
    const view = new Int32Array(shared);

    Atomics.store(view, 0, 456);
    Atomics.notify(view, 0, 1);  // Wake up one waiter
};
```

### Go

Go's concurrency is based on **goroutines** and **channels**.

#### Goroutines

```go
package main

import (
    "fmt"
    "time"
)

func say(s string) {
    for i := 0; i < 3; i++ {
        time.Sleep(100 * time.Millisecond)
        fmt.Println(s)
    }
}

func main() {
    // Start goroutine
    go say("world")
    say("hello")
}

// Anonymous goroutine
go func() {
    fmt.Println("Anonymous goroutine")
}()

// Goroutines are very lightweight (~2KB stack)
for i := 0; i < 1000; i++ {
    go func(id int) {
        fmt.Println("Goroutine", id)
    }(i)
}
```

#### Channels

```go
// Unbuffered channel
ch := make(chan int)

// Send (blocks until received)
go func() {
    ch <- 42
}()

// Receive (blocks until sent)
value := <-ch
fmt.Println(value)

// Buffered channel
ch := make(chan int, 3)
ch <- 1  // Doesn't block
ch <- 2
ch <- 3
// ch <- 4  // Would block (buffer full)

// Close channel
close(ch)

// Range over channel
ch := make(chan int, 5)
go func() {
    for i := 0; i < 5; i++ {
        ch <- i
    }
    close(ch)
}()

for value := range ch {
    fmt.Println(value)
}

// Check if closed
value, ok := <-ch
if !ok {
    fmt.Println("Channel closed")
}
```

#### Select Statement

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "one"
    }()

    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "two"
    }()

    // Wait for both
    for i := 0; i < 2; i++ {
        select {
        case msg1 := <-ch1:
            fmt.Println("Received", msg1)
        case msg2 := <-ch2:
            fmt.Println("Received", msg2)
        case <-time.After(3 * time.Second):
            fmt.Println("Timeout")
        }
    }

    // Non-blocking select
    select {
    case msg := <-ch1:
        fmt.Println(msg)
    default:
        fmt.Println("No message ready")
    }
}
```

#### Sync Package

```go
package main

import (
    "fmt"
    "sync"
)

// Mutex
var (
    counter int
    mutex   sync.Mutex
)

func increment() {
    mutex.Lock()
    counter++
    mutex.Unlock()
}

// WaitGroup
func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d starting\n", id)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup

    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }

    wg.Wait()
    fmt.Println("All workers done")
}

// Once (execute exactly once)
var once sync.Once

func initialize() {
    fmt.Println("Initializing...")
}

func main() {
    for i := 0; i < 10; i++ {
        once.Do(initialize)  // Only prints once
    }
}

// Atomic operations
import "sync/atomic"

var counter int64

func increment() {
    atomic.AddInt64(&counter, 1)
}

func get() int64 {
    return atomic.LoadInt64(&counter)
}
```

### Rust

Rust's ownership system ensures memory safety and eliminates data races at compile time.

#### Send and Sync Traits

**Send:** Type can be transferred between threads
**Sync:** Type can be accessed from multiple threads simultaneously (T is Sync if &T is Send)

```rust
// Most types are Send and Sync
// Exceptions: Rc, RefCell (not thread-safe)

use std::sync::Arc;
use std::thread;

fn main() {
    let data = Arc::new(vec![1, 2, 3]);  // Arc is Send + Sync

    let mut handles = vec![];
    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            println!("Thread {} sees: {:?}", i, data);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

#### Threads

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // Spawn thread
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("Thread: {}", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("Main: {}", i);
        thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();

    // Thread with return value
    let handle = thread::spawn(|| {
        42
    });
    let result = handle.join().unwrap();
    println!("Result: {}", result);

    // Moving data into thread
    let v = vec![1, 2, 3];
    let handle = thread::spawn(move || {
        println!("Vector: {:?}", v);
    });
    // v is moved, can't use here
    handle.join().unwrap();
}
```

#### Mutex and Arc

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

#### Channels

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let vals = vec![
            String::from("hi"),
            String::from("from"),
            String::from("thread"),
        ];

        for val in vals {
            tx.send(val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    for received in rx {
        println!("Got: {}", received);
    }

    // Multiple producers
    let (tx, rx) = mpsc::channel();
    let tx1 = tx.clone();

    thread::spawn(move || {
        tx.send("message from first").unwrap();
    });

    thread::spawn(move || {
        tx1.send("message from second").unwrap();
    });

    for _ in 0..2 {
        println!("{}", rx.recv().unwrap());
    }
}
```

### Java

#### Threads

```java
// Extending Thread class
class MyThread extends Thread {
    public void run() {
        System.out.println("Thread running: " + getName());
    }
}

// Implementing Runnable
class MyRunnable implements Runnable {
    public void run() {
        System.out.println("Runnable running");
    }
}

public class Main {
    public static void main(String[] args) {
        // Start thread
        MyThread thread = new MyThread();
        thread.start();

        // Using Runnable
        Thread thread2 = new Thread(new MyRunnable());
        thread2.start();

        // Lambda expression
        Thread thread3 = new Thread(() -> {
            System.out.println("Lambda thread");
        });
        thread3.start();

        // Join
        try {
            thread.join();
            thread2.join();
            thread3.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

#### Synchronized

```java
class Counter {
    private int count = 0;

    // Synchronized method
    public synchronized void increment() {
        count++;
    }

    // Synchronized block
    public void increment2() {
        synchronized(this) {
            count++;
        }
    }

    public synchronized int getCount() {
        return count;
    }
}

// Static synchronized (class-level lock)
class MyClass {
    private static int count = 0;

    public static synchronized void increment() {
        count++;
    }
}
```

#### ExecutorService

```java
import java.util.concurrent.*;

public class ExecutorExample {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        // Fixed thread pool
        ExecutorService executor = Executors.newFixedThreadPool(4);

        // Submit Runnable
        executor.submit(() -> {
            System.out.println("Task running");
        });

        // Submit Callable (returns value)
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(1000);
            return 42;
        });

        System.out.println("Result: " + future.get());  // Blocks

        // Execute multiple tasks
        List<Callable<Integer>> tasks = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            tasks.add(() -> taskId * taskId);
        }

        List<Future<Integer>> results = executor.invokeAll(tasks);
        for (Future<Integer> result : results) {
            System.out.println(result.get());
        }

        // Shutdown
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
    }
}
```

#### Concurrent Collections

```java
import java.util.concurrent.*;

// ConcurrentHashMap
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
map.put("key", 1);
map.putIfAbsent("key", 2);  // Atomic
int value = map.get("key");

// CopyOnWriteArrayList (good for read-heavy workloads)
CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
list.add("item");

// BlockingQueue
BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(10);

// Producer
new Thread(() -> {
    try {
        for (int i = 0; i < 10; i++) {
            queue.put(i);  // Blocks if full
        }
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
}).start();

// Consumer
new Thread(() -> {
    try {
        while (true) {
            Integer item = queue.take();  // Blocks if empty
            System.out.println(item);
        }
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
}).start();
```

### C++

#### std::thread

```cpp
#include <iostream>
#include <thread>
#include <vector>

void hello() {
    std::cout << "Hello from thread\n";
}

void count(int n) {
    for (int i = 0; i < n; i++) {
        std::cout << i << " ";
    }
}

int main() {
    // Basic thread
    std::thread t1(hello);
    t1.join();

    // Thread with arguments
    std::thread t2(count, 10);
    t2.join();

    // Lambda
    std::thread t3([]() {
        std::cout << "Lambda thread\n";
    });
    t3.join();

    // Multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back([i]() {
            std::cout << "Thread " << i << "\n";
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
```

#### Mutex

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>

std::mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);  // RAII
        counter++;
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Counter: " << counter << "\n";
    return 0;
}
```

#### std::async

```cpp
#include <iostream>
#include <future>
#include <chrono>

int compute(int n) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return n * n;
}

int main() {
    // Launch async task
    std::future<int> result = std::async(std::launch::async, compute, 5);

    std::cout << "Doing other work...\n";

    // Get result (blocks if not ready)
    std::cout << "Result: " << result.get() << "\n";

    // Multiple async tasks
    auto f1 = std::async(std::launch::async, compute, 2);
    auto f2 = std::async(std::launch::async, compute, 3);
    auto f3 = std::async(std::launch::async, compute, 4);

    std::cout << f1.get() + f2.get() + f3.get() << "\n";

    return 0;
}
```

---

## Deadlock Prevention

Four necessary conditions for deadlock (Coffman conditions):
1. Mutual Exclusion
2. Hold and Wait
3. No Preemption
4. Circular Wait

**Prevent deadlock by breaking at least one condition.**

### Lock Ordering

Always acquire locks in the same global order.

```python
import threading

class BankAccount:
    def __init__(self, id, balance):
        self.id = id
        self.balance = balance
        self.lock = threading.Lock()

def transfer(from_account, to_account, amount):
    # WRONG: Can deadlock
    # with from_account.lock:
    #     with to_account.lock:
    #         from_account.balance -= amount
    #         to_account.balance += amount

    # RIGHT: Lock in consistent order (by ID)
    first, second = (from_account, to_account) if from_account.id < to_account.id else (to_account, from_account)

    with first.lock:
        with second.lock:
            from_account.balance -= amount
            to_account.balance += amount

# Now safe regardless of call order
account1 = BankAccount(1, 1000)
account2 = BankAccount(2, 1000)

# Both threads acquire locks in same order (lock1, then lock2)
t1 = threading.Thread(target=transfer, args=(account1, account2, 100))
t2 = threading.Thread(target=transfer, args=(account2, account1, 50))
t1.start()
t2.start()
t1.join()
t2.join()
```

**C++ example:**

```cpp
#include <mutex>
#include <algorithm>

class BankAccount {
public:
    int id;
    int balance;
    std::mutex mtx;

    BankAccount(int id, int bal) : id(id), balance(bal) {}
};

void transfer(BankAccount& from, BankAccount& to, int amount) {
    // Lock in consistent order
    BankAccount* first = &from;
    BankAccount* second = &to;

    if (from.id > to.id) {
        std::swap(first, second);
    }

    std::lock_guard<std::mutex> lock1(first->mtx);
    std::lock_guard<std::mutex> lock2(second->mtx);

    from.balance -= amount;
    to.balance += amount;
}

// Or use std::lock to acquire multiple locks atomically
void transfer_v2(BankAccount& from, BankAccount& to, int amount) {
    std::unique_lock<std::mutex> lock1(from.mtx, std::defer_lock);
    std::unique_lock<std::mutex> lock2(to.mtx, std::defer_lock);

    std::lock(lock1, lock2);  // Acquire both atomically

    from.balance -= amount;
    to.balance += amount;
}
```

### Lock Timeout

Try to acquire lock with timeout; if timeout, release all locks and retry.

```python
import threading
import time

class TimedLock:
    def __init__(self):
        self.lock = threading.Lock()

    def acquire_with_timeout(self, timeout):
        end_time = time.time() + timeout
        while True:
            if self.lock.acquire(blocking=False):
                return True
            if time.time() >= end_time:
                return False
            time.sleep(0.001)

    def release(self):
        self.lock.release()

lock1 = TimedLock()
lock2 = TimedLock()

def worker1():
    while True:
        if lock1.acquire_with_timeout(1):
            try:
                time.sleep(0.1)
                if lock2.acquire_with_timeout(1):
                    try:
                        print("Worker1 has both locks")
                        break
                    finally:
                        lock2.release()
                else:
                    print("Worker1 timeout on lock2, retrying")
            finally:
                lock1.release()
        else:
            print("Worker1 timeout on lock1, retrying")
        time.sleep(0.01)  # Backoff

def worker2():
    while True:
        if lock2.acquire_with_timeout(1):
            try:
                time.sleep(0.1)
                if lock1.acquire_with_timeout(1):
                    try:
                        print("Worker2 has both locks")
                        break
                    finally:
                        lock1.release()
                else:
                    print("Worker2 timeout on lock1, retrying")
            finally:
                lock2.release()
        else:
            print("Worker2 timeout on lock2, retrying")
        time.sleep(0.01)

t1 = threading.Thread(target=worker1)
t2 = threading.Thread(target=worker2)
t1.start()
t2.start()
t1.join()
t2.join()
```

### Try-Lock

Attempt to acquire lock without blocking.

```cpp
#include <mutex>
#include <thread>
#include <chrono>

std::mutex mtx1, mtx2;

void worker() {
    while (true) {
        if (mtx1.try_lock()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (mtx2.try_lock()) {
                // Got both locks
                std::cout << "Worker has both locks\n";
                mtx2.unlock();
                mtx1.unlock();
                break;
            } else {
                // Couldn't get second lock, release first
                mtx1.unlock();
            }
        }
        // Backoff before retry
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
```

### Deadlock Detection

Build resource allocation graph and detect cycles.

```python
class DeadlockDetector:
    def __init__(self):
        self.waiting_for = {}  # thread -> resource
        self.held_by = {}      # resource -> thread
        self.lock = threading.Lock()

    def acquire_intent(self, thread_id, resource_id):
        with self.lock:
            self.waiting_for[thread_id] = resource_id

            # Check for cycle
            if self._has_cycle(thread_id):
                del self.waiting_for[thread_id]
                raise Exception(f"Deadlock detected! Thread {thread_id} waiting for {resource_id}")

    def acquire_complete(self, thread_id, resource_id):
        with self.lock:
            if thread_id in self.waiting_for:
                del self.waiting_for[thread_id]
            self.held_by[resource_id] = thread_id

    def release(self, thread_id, resource_id):
        with self.lock:
            if resource_id in self.held_by:
                del self.held_by[resource_id]

    def _has_cycle(self, start_thread):
        visited = set()
        thread = start_thread

        while thread not in visited:
            visited.add(thread)

            if thread not in self.waiting_for:
                return False

            resource = self.waiting_for[thread]

            if resource not in self.held_by:
                return False

            thread = self.held_by[resource]

            if thread == start_thread:
                return True

        return False
```

### Resource Hierarchy

Assign hierarchy levels to resources; always acquire in increasing order.

```python
# Define resource hierarchy
RESOURCE_LEVELS = {
    'database': 1,
    'cache': 2,
    'network': 3,
    'file': 4
}

class HierarchicalLock:
    def __init__(self, name):
        self.name = name
        self.level = RESOURCE_LEVELS[name]
        self.lock = threading.Lock()

thread_local = threading.local()

def acquire_hierarchical(lock):
    if not hasattr(thread_local, 'max_level'):
        thread_local.max_level = 0

    if lock.level <= thread_local.max_level:
        raise Exception(f"Lock hierarchy violation! Trying to acquire {lock.name} (level {lock.level}) after level {thread_local.max_level}")

    lock.lock.acquire()
    thread_local.max_level = lock.level

def release_hierarchical(lock):
    lock.lock.release()
    thread_local.max_level = lock.level - 1

# Usage
db_lock = HierarchicalLock('database')
cache_lock = HierarchicalLock('cache')

# This is OK
acquire_hierarchical(db_lock)
acquire_hierarchical(cache_lock)
release_hierarchical(cache_lock)
release_hierarchical(db_lock)

# This would raise exception (wrong order)
# acquire_hierarchical(cache_lock)
# acquire_hierarchical(db_lock)  # Exception!
```

---

## Performance Considerations

### Lock Contention

**Problem:** Many threads competing for the same lock, causing serialization.

**Solutions:**

1. **Reduce critical section size:**

```python
# BAD: Large critical section
with lock:
    data = read_from_database()
    result = expensive_computation(data)
    write_to_cache(result)

# GOOD: Minimize critical section
data = read_from_database()
result = expensive_computation(data)
with lock:
    write_to_cache(result)
```

2. **Lock striping:** Use multiple locks for different parts of data structure

```python
class StripedHashMap:
    def __init__(self, num_stripes=16):
        self.num_stripes = num_stripes
        self.stripes = [{'lock': threading.Lock(), 'data': {}} for _ in range(num_stripes)]

    def _get_stripe(self, key):
        return hash(key) % self.num_stripes

    def get(self, key):
        stripe = self.stripes[self._get_stripe(key)]
        with stripe['lock']:
            return stripe['data'].get(key)

    def put(self, key, value):
        stripe = self.stripes[self._get_stripe(key)]
        with stripe['lock']:
            stripe['data'][key] = value
```

3. **Read-write locks:** Allow concurrent readers

```cpp
#include <shared_mutex>

std::shared_mutex rwlock;
std::map<std::string, int> data;

int read(const std::string& key) {
    std::shared_lock lock(rwlock);  // Concurrent reads
    return data[key];
}

void write(const std::string& key, int value) {
    std::unique_lock lock(rwlock);  // Exclusive write
    data[key] = value;
}
```

### False Sharing

**Problem:** Different threads access different variables on the same cache line, causing unnecessary cache invalidation.

**Cache line:** Typically 64 bytes; when one thread modifies a byte, the entire cache line is invalidated in other cores.

**Example of false sharing:**

```cpp
// BAD: False sharing
struct Counters {
    int counter1;  // Likely on same cache line
    int counter2;  // as counter1
};

Counters counters;

// Thread 1
void increment1() {
    for (int i = 0; i < 1000000; i++) {
        counters.counter1++;  // Invalidates cache line
    }
}

// Thread 2
void increment2() {
    for (int i = 0; i < 1000000; i++) {
        counters.counter2++;  // Invalidates cache line
    }
}

// GOOD: Padding to separate cache lines
struct alignas(64) PaddedCounters {
    int counter1;
    char padding1[60];  // Fill rest of cache line
    int counter2;
    char padding2[60];
};

// Or use C++17 hardware_destructive_interference_size
struct Counters {
    alignas(std::hardware_destructive_interference_size) int counter1;
    alignas(std::hardware_destructive_interference_size) int counter2;
};
```

**Java example:**

```java
// Using @Contended annotation (requires -XX:-RestrictContended)
public class Counters {
    @jdk.internal.vm.annotation.Contended
    volatile long counter1;

    @jdk.internal.vm.annotation.Contended
    volatile long counter2;
}
```

### Lock-Free Data Structures

Use atomic operations instead of locks for better performance.

**Lock-free queue (simplified):**

```cpp
#include <atomic>

template<typename T>
class LockFreeQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next;
        Node(T val) : data(val), next(nullptr) {}
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;

public:
    LockFreeQueue() {
        Node* dummy = new Node(T());
        head.store(dummy);
        tail.store(dummy);
    }

    void enqueue(T value) {
        Node* node = new Node(value);
        Node* prev_tail;

        while (true) {
            prev_tail = tail.load();
            Node* next = prev_tail->next.load();

            if (prev_tail == tail.load()) {
                if (next == nullptr) {
                    if (prev_tail->next.compare_exchange_weak(next, node)) {
                        break;
                    }
                } else {
                    tail.compare_exchange_weak(prev_tail, next);
                }
            }
        }
        tail.compare_exchange_weak(prev_tail, node);
    }

    bool dequeue(T& result) {
        while (true) {
            Node* first = head.load();
            Node* last = tail.load();
            Node* next = first->next.load();

            if (first == head.load()) {
                if (first == last) {
                    if (next == nullptr) {
                        return false;  // Empty
                    }
                    tail.compare_exchange_weak(last, next);
                } else {
                    result = next->data;
                    if (head.compare_exchange_weak(first, next)) {
                        delete first;  // Caution: ABA problem
                        return true;
                    }
                }
            }
        }
    }
};
```

**Benefits:**
- No lock contention
- No deadlocks
- Better scalability

**Drawbacks:**
- Complex to implement correctly
- ABA problem
- Memory reclamation challenges

### Memory Ordering

**Sequential consistency (strongest):**
```cpp
std::atomic<int> x(0);
x.store(1, std::memory_order_seq_cst);  // Default
```

**Relaxed (weakest, fastest):**
```cpp
x.store(1, std::memory_order_relaxed);  // Only atomicity, no ordering
```

**Acquire-Release:**
```cpp
// Producer
data = 42;
flag.store(true, std::memory_order_release);

// Consumer
while (!flag.load(std::memory_order_acquire));
assert(data == 42);  // Guaranteed
```

**Performance impact:**
- **SeqCst:** Full memory fence (slowest)
- **AcqRel:** Partial fence
- **Relaxed:** No fence (fastest)

---

## Real-World Applications

### Web Servers

**Problem:** Handle thousands of concurrent requests.

**Solutions:**

1. **Thread-per-request (traditional):**

```python
import socket
import threading

def handle_client(client_socket):
    request = client_socket.recv(1024)
    # Process request
    response = b"HTTP/1.1 200 OK\r\n\r\nHello World"
    client_socket.send(response)
    client_socket.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen(5)

while True:
    client, addr = server.accept()
    thread = threading.Thread(target=handle_client, args=(client,))
    thread.start()
```

2. **Thread pool:**

```python
from concurrent.futures import ThreadPoolExecutor
import socket

def handle_client(client_socket):
    # ... same as above

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen(5)

with ThreadPoolExecutor(max_workers=50) as executor:
    while True:
        client, addr = server.accept()
        executor.submit(handle_client, client)
```

3. **Async I/O (most scalable):**

```python
import asyncio

async def handle_client(reader, writer):
    data = await reader.read(1024)
    # Process request
    response = b"HTTP/1.1 200 OK\r\n\r\nHello World"
    writer.write(response)
    await writer.drain()
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, '0.0.0.0', 8080)
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

### Database Connection Pools

**Problem:** Database connections are expensive to create; reuse a pool.

```python
import threading
import queue
import time

class ConnectionPool:
    def __init__(self, create_connection, max_connections=10):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.current_connections = 0
        self.lock = threading.Lock()

    def acquire(self, timeout=None):
        try:
            # Try to get from pool
            return self.pool.get(block=False)
        except queue.Empty:
            # Pool empty, maybe create new connection
            with self.lock:
                if self.current_connections < self.max_connections:
                    self.current_connections += 1
                    return self.create_connection()

            # Wait for available connection
            return self.pool.get(timeout=timeout)

    def release(self, connection):
        try:
            self.pool.put(connection, block=False)
        except queue.Full:
            # Pool full, close connection
            connection.close()
            with self.lock:
                self.current_connections -= 1

    def __enter__(self):
        self.connection = self.acquire()
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release(self.connection)

# Usage
def create_db_connection():
    # Simulate creating connection
    print("Creating new connection")
    return {"connection": "db"}

pool = ConnectionPool(create_db_connection, max_connections=5)

def worker(id):
    with pool as conn:
        print(f"Worker {id} using {conn}")
        time.sleep(1)
    print(f"Worker {id} released connection")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### GUI Event Handling

**Problem:** Keep UI responsive while doing background work.

```python
import tkinter as tk
import threading
import time

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Concurrent GUI")

        self.label = tk.Label(self, text="Ready")
        self.label.pack()

        self.button = tk.Button(self, text="Start Task", command=self.start_task)
        self.button.pack()

    def start_task(self):
        self.label.config(text="Working...")
        self.button.config(state='disabled')

        # Run in background thread
        thread = threading.Thread(target=self.background_task)
        thread.start()

    def background_task(self):
        # Simulate long-running task
        for i in range(5):
            time.sleep(1)
            # Update UI from background thread (use after)
            self.after(0, self.update_progress, i + 1)

        self.after(0, self.task_complete)

    def update_progress(self, count):
        self.label.config(text=f"Progress: {count}/5")

    def task_complete(self):
        self.label.config(text="Done!")
        self.button.config(state='normal')

app = Application()
app.mainloop()
```

### Background Task Processing

**Task queue with workers:**

```python
import threading
import queue
import time

class TaskQueue:
    def __init__(self, num_workers=4):
        self.tasks = queue.Queue()
        self.workers = []
        self.shutdown_flag = False

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.start()
            self.workers.append(worker)

    def _worker(self, worker_id):
        while not self.shutdown_flag:
            try:
                task, callback = self.tasks.get(timeout=1)
                print(f"Worker {worker_id} processing {task}")
                result = self._process_task(task)
                if callback:
                    callback(result)
                self.tasks.task_done()
            except queue.Empty:
                continue

    def _process_task(self, task):
        # Simulate task processing
        time.sleep(2)
        return f"Result of {task}"

    def submit(self, task, callback=None):
        self.tasks.put((task, callback))

    def shutdown(self):
        self.shutdown_flag = True
        for worker in self.workers:
            worker.join()

# Usage
def on_complete(result):
    print(f"Task completed: {result}")

task_queue = TaskQueue(num_workers=4)

for i in range(10):
    task_queue.submit(f"Task-{i}", on_complete)

task_queue.tasks.join()  # Wait for all tasks
task_queue.shutdown()
```

---

## Best Practices

### 1. Prefer Immutability

Immutable data can be shared without synchronization.

```python
# BAD: Mutable shared state
class Counter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1

# GOOD: Immutable data
from dataclasses import dataclass

@dataclass(frozen=True)
class CounterState:
    count: int

def increment(state):
    return CounterState(state.count + 1)

# Use atomic reference for updates
import threading

class AtomicReference:
    def __init__(self, value):
        self.value = value
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.value

    def set(self, new_value):
        with self.lock:
            self.value = new_value

    def update(self, func):
        with self.lock:
            self.value = func(self.value)

state = AtomicReference(CounterState(0))
state.update(increment)
```

### 2. Minimize Shared State

Reduce the amount of data shared between threads.

```python
# BAD: Everything shared
shared_data = {'counter': 0, 'results': [], 'status': 'running'}
lock = threading.Lock()

def worker():
    with lock:
        shared_data['counter'] += 1
        shared_data['results'].append(compute())

# GOOD: Minimize sharing, use message passing
result_queue = queue.Queue()

def worker(task_id):
    result = compute(task_id)  # No shared state
    result_queue.put(result)   # Communicate via queue
```

### 3. Use Thread-Safe Data Structures

```python
from queue import Queue
from collections import deque
import threading

# Thread-safe queue
q = Queue()

# Thread-safe deque (for most operations)
d = deque()

# NOT thread-safe without synchronization
lst = []
dct = {}
```

### 4. Always Release Locks

Use RAII, context managers, or try-finally.

```python
# BAD: Can leak lock on exception
lock.acquire()
do_something()  # Exception here leaks lock!
lock.release()

# GOOD: Context manager
with lock:
    do_something()

# GOOD: Try-finally
lock.acquire()
try:
    do_something()
finally:
    lock.release()
```

### 5. Avoid Nested Locks When Possible

```python
# BAD: Nested locks increase deadlock risk
with lock1:
    with lock2:
        do_something()

# GOOD: Single lock or lock-free
with combined_lock:
    do_something()

# GOOD: Lock ordering if nested necessary
locks = sorted([lock1, lock2], key=id)
with locks[0]:
    with locks[1]:
        do_something()
```

### 6. Document Thread Safety

```python
class BankAccount:
    """
    Thread-safe bank account.

    All methods are thread-safe and can be called concurrently.
    """
    def __init__(self):
        self._balance = 0
        self._lock = threading.Lock()

    def deposit(self, amount):
        """Thread-safe deposit."""
        with self._lock:
            self._balance += amount
```

### 7. Use Appropriate Concurrency Model

- **CPU-bound:** Use multiprocessing (Python), or threads in languages without GIL
- **I/O-bound:** Use async/await or threading
- **Mixed:** Combine approaches

### 8. Set Thread Names for Debugging

```python
thread = threading.Thread(target=worker, name="Worker-1")
thread.start()

# In worker
print(f"Running in {threading.current_thread().name}")
```

### 9. Handle Exceptions in Threads

```python
def worker():
    try:
        do_work()
    except Exception as e:
        logging.error(f"Error in thread: {e}", exc_info=True)
        # Don't let exception kill thread silently
```

### 10. Use Daemon Threads Carefully

```python
# Daemon threads die when main thread exits
thread = threading.Thread(target=background_task, daemon=True)
thread.start()

# Non-daemon threads keep program running
thread = threading.Thread(target=important_task, daemon=False)
thread.start()
```

---

## Anti-Patterns

### 1. Sleeping Instead of Synchronization

```python
# BAD: Race condition masked by sleep
def worker1():
    write_data()
    time.sleep(0.1)  # Hope worker2 is ready...
    read_shared_data()

# GOOD: Proper synchronization
event = threading.Event()

def worker1():
    write_data()
    event.set()

def worker2():
    event.wait()
    read_shared_data()
```

### 2. Busy-Waiting

```python
# BAD: Wastes CPU
while not data_ready:
    pass  # Spin!

# GOOD: Use condition variable
condition = threading.Condition()

def producer():
    with condition:
        prepare_data()
        data_ready = True
        condition.notify()

def consumer():
    with condition:
        while not data_ready:
            condition.wait()  # Sleeps, doesn't waste CPU
        process_data()
```

### 3. Lock Hogging

```python
# BAD: Hold lock too long
with lock:
    data = read_database()  # Long I/O
    result = expensive_compute(data)  # Long CPU
    write_cache(result)

# GOOD: Minimize critical section
data = read_database()
result = expensive_compute(data)
with lock:
    write_cache(result)  # Only lock needed part
```

### 4. Forgetting to Join Threads

```python
# BAD: Main exits before thread finishes
def main():
    thread = threading.Thread(target=important_work)
    thread.start()
    # Main exits, thread might be killed!

# GOOD: Wait for completion
def main():
    thread = threading.Thread(target=important_work)
    thread.start()
    thread.join()
```

### 5. Using Mutable Default Arguments

```python
# BAD: Default list shared between threads!
def worker(results=[]):
    results.append(compute())  # Race condition!
    return results

# GOOD: Immutable default
def worker(results=None):
    if results is None:
        results = []
    results.append(compute())
    return results
```

### 6. Double-Checked Locking (Without Proper Memory Barriers)

```python
# BAD: Broken double-checked locking
singleton = None

def get_singleton():
    global singleton
    if singleton is None:  # Check 1 (unlocked)
        with lock:
            if singleton is None:  # Check 2 (locked)
                singleton = Singleton()  # Can be partially visible!
    return singleton

# GOOD: Use proper synchronization or module-level initialization
_singleton = None
_lock = threading.Lock()

def get_singleton():
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = Singleton()
    return _singleton

# BETTER: Module-level (thread-safe in Python)
_singleton = Singleton()

def get_singleton():
    return _singleton
```

### 7. Not Considering Thread Count vs Core Count

```python
# BAD: Creating too many threads
threads = [threading.Thread(target=cpu_work) for _ in range(1000)]

# GOOD: Use thread pool with appropriate size
import multiprocessing
num_cores = multiprocessing.cpu_count()
with ThreadPoolExecutor(max_workers=num_cores) as executor:
    executor.map(cpu_work, range(1000))
```

---

## Debugging Concurrent Programs

### 1. Logging with Thread Information

```python
import logging
import threading

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s'
)

def worker(n):
    logging.info(f"Starting work on {n}")
    # ... work ...
    logging.info(f"Finished work on {n}")

thread = threading.Thread(target=worker, args=(42,), name="Worker-1")
thread.start()
```

### 2. Deadlock Detection

**Python: Use faulthandler**

```python
import faulthandler
import signal

# Dump all thread stacks on SIGUSR1
faulthandler.register(signal.SIGUSR1)

# Or dump after timeout
faulthandler.dump_traceback_later(10, repeat=True)
```

**Tools:**
- **Python:** `threading.enumerate()`, stack traces
- **Java:** jstack, VisualVM
- **C++:** gdb, lldb
- **Helgrind (Valgrind):** Detects race conditions and deadlocks

### 3. Race Condition Detection

**ThreadSanitizer (C/C++):**

```bash
# Compile with -fsanitize=thread
g++ -fsanitize=thread -g program.cpp -o program
./program
```

**Python: Use threading debug mode**

```python
import sys
import threading

# Enable thread debugging
threading.settrace(lambda *args: print(args))
```

### 4. Reproducible Debugging

Add determinism for debugging:

```python
import random
import threading

# Seed random for reproducibility
random.seed(42)

# Add random sleeps to expose race conditions
def worker():
    # Add jitter to expose timing issues
    time.sleep(random.random() * 0.01)
    critical_section()
```

### 5. Visualization

Visualize thread execution:

```python
import time
import threading

class ExecutionTracer:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def log(self, event):
        with self.lock:
            self.events.append({
                'time': time.time(),
                'thread': threading.current_thread().name,
                'event': event
            })

    def print_trace(self):
        for e in sorted(self.events, key=lambda x: x['time']):
            print(f"{e['time']:.4f} [{e['thread']:15s}] {e['event']}")

tracer = ExecutionTracer()

def worker(n):
    tracer.log(f"Start {n}")
    time.sleep(0.1)
    tracer.log(f"End {n}")

threads = [threading.Thread(target=worker, args=(i,), name=f"Worker-{i}") for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

tracer.print_trace()
```

---

## Testing Concurrent Code

### 1. Stress Testing

Run many iterations to expose race conditions:

```python
import threading

def test_concurrent_counter():
    counter = Counter()  # Your concurrent counter

    def increment_many():
        for _ in range(10000):
            counter.increment()

    threads = [threading.Thread(target=increment_many) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.get() == 100000, f"Expected 100000, got {counter.get()}"

# Run many times
for _ in range(100):
    test_concurrent_counter()
```

### 2. Property-Based Testing

Use libraries like `hypothesis`:

```python
from hypothesis import given, strategies as st
import threading

@given(st.lists(st.integers()))
def test_concurrent_list_operations(items):
    thread_safe_list = ThreadSafeList()

    def add_items():
        for item in items:
            thread_safe_list.append(item)

    threads = [threading.Thread(target=add_items) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(thread_safe_list) == len(items) * 4
```

### 3. Deterministic Testing with Barriers

```python
def test_race_condition():
    barrier = threading.Barrier(2)
    result = []

    def thread1():
        barrier.wait()  # Synchronize start
        result.append(1)

    def thread2():
        barrier.wait()  # Synchronize start
        result.append(2)

    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both 1 and 2 should be present
    assert set(result) == {1, 2}
```

### 4. Mock Synchronization Primitives

Inject failures for testing:

```python
class FailingLock:
    def __init__(self, fail_on=None):
        self.lock = threading.Lock()
        self.acquire_count = 0
        self.fail_on = fail_on or set()

    def acquire(self):
        self.acquire_count += 1
        if self.acquire_count in self.fail_on:
            raise Exception("Lock acquisition failed")
        return self.lock.acquire()

    def release(self):
        return self.lock.release()

def test_error_handling():
    lock = FailingLock(fail_on={2})

    with pytest.raises(Exception):
        for i in range(3):
            lock.acquire()
            # ... do work ...
            lock.release()
```

### 5. Timeout Testing

Ensure no deadlocks:

```python
import pytest
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def test_no_deadlock():
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout

    try:
        # Code that might deadlock
        run_concurrent_operation()
    except TimeoutException:
        pytest.fail("Operation timed out (possible deadlock)")
    finally:
        signal.alarm(0)  # Cancel alarm
```

---

## Summary

Concurrency is a powerful tool but comes with complexity:

**Key Takeaways:**
1. **Understand the difference** between concurrency and parallelism
2. **Choose the right primitive:** Mutex, semaphore, condition variable, atomic, etc.
3. **Follow patterns:** Producer-consumer, thread pool, async/await
4. **Prevent deadlocks:** Lock ordering, timeout, detection
5. **Optimize performance:** Minimize contention, avoid false sharing, use lock-free when appropriate
6. **Apply to real problems:** Web servers, connection pools, background processing
7. **Follow best practices:** Immutability, minimal shared state, proper error handling
8. **Avoid anti-patterns:** No busy-waiting, no lock hogging, proper thread management
9. **Debug effectively:** Logging, sanitizers, visualization
10. **Test thoroughly:** Stress testing, deterministic testing, timeout guards

**Remember:** The best concurrent program is one that minimizes shared mutable state and uses the simplest synchronization mechanism that works.

**Further Reading:**
- "The Art of Multiprocessor Programming" by Herlihy & Shavit
- "Java Concurrency in Practice" by Goetz et al.
- "Seven Concurrency Models in Seven Weeks" by Butcher
- "Programming Rust" (Chapter on Concurrency) by Blandy & Orendorff
