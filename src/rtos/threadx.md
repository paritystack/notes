# ThreadX

ThreadX is a real-time operating system (RTOS) designed for deeply embedded applications. It's known for its small footprint and fast execution.

## Core Concepts

**Threads**: Execution contexts
**Message Queues**: Inter-thread communication  
**Semaphores**: Synchronization
**Mutexes**: Resource protection
**Event Flags**: Thread synchronization
**Memory Pools**: Dynamic memory management

## Thread Creation

```c
#include "tx_api.h"

TX_THREAD my_thread;
UCHAR thread_stack[1024];

void my_thread_entry(ULONG thread_input) {
    while(1) {
        // Thread logic
        tx_thread_sleep(100);  // Sleep 100 ticks
    }
}

void tx_application_define(void *first_unused_memory) {
    tx_thread_create(
        &my_thread,              // Thread control block
        "My Thread",             // Name
        my_thread_entry,         // Entry function
        0,                       // Input
        thread_stack,            // Stack start
        sizeof(thread_stack),    // Stack size
        16,                      // Priority (0-31)
        16,                      // Preemption threshold
        TX_NO_TIME_SLICE,        // Time slice
        TX_AUTO_START            // Auto start
    );
}
```

## Message Queues

```c
TX_QUEUE my_queue;
UCHAR queue_area[100 * sizeof(ULONG)];

// Create queue
tx_queue_create(
    &my_queue,
    "My Queue",
    TX_1_ULONG,              // Message size
    queue_area,
    sizeof(queue_area)
);

// Send message
ULONG message = 0x12345678;
tx_queue_send(&my_queue, &message, TX_WAIT_FOREVER);

// Receive message
ULONG received;
tx_queue_receive(&my_queue, &received, TX_WAIT_FOREVER);
```

## Semaphores

```c
TX_SEMAPHORE my_semaphore;

// Create counting semaphore
tx_semaphore_create(&my_semaphore, "My Semaphore", 1);

// Get semaphore
tx_semaphore_get(&my_semaphore, TX_WAIT_FOREVER);

// Put semaphore
tx_semaphore_put(&my_semaphore);
```

## Mutex

```c
TX_MUTEX my_mutex;

// Create mutex
tx_mutex_create(&my_mutex, "My Mutex", TX_NO_INHERIT);

// Get mutex
tx_mutex_get(&my_mutex, TX_WAIT_FOREVER);

// Release mutex
tx_mutex_put(&my_mutex);
```

ThreadX is widely used in embedded systems, particularly in IoT and industrial applications, offering deterministic real-time performance.
