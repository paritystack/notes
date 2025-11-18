# ThreadX (Azure RTOS)

ThreadX is a professional-grade real-time operating system (RTOS) designed for deeply embedded, real-time, and IoT applications. Now part of Azure RTOS (open-sourced by Microsoft), it's known for its small footprint, fast execution, and deterministic real-time performance.

## Key Features

- **Ultra-small footprint**: Kernel as small as 2KB
- **Deterministic**: Bounded execution times
- **Priority-based preemptive scheduling**: 32 or 1024 priority levels
- **Safety certifications**: IEC 61508, IEC 62304, ISO 26262, EN 50128
- **Picokernel architecture**: Fast context switching (~200 clock cycles)
- **No royalties or licensing fees**: Open-source under MIT license
- **Thread-Metrics test suite**: Built-in performance analysis

## Core Concepts

**Threads**: Independent execution contexts with individual stacks and priorities
**Message Queues**: FIFO-based inter-thread communication
**Semaphores**: Counting and binary synchronization primitives
**Mutexes**: Resource protection with priority inheritance
**Event Flags**: Flexible thread synchronization mechanism
**Memory Pools**: Block and byte pool allocation
**Timers**: Application timers with expiration routines
**Interrupt Management**: Fast ISR processing with deferred work

## Thread Management

### Thread Creation and Configuration

```c
#include "tx_api.h"

TX_THREAD my_thread;
UCHAR thread_stack[1024];

void my_thread_entry(ULONG thread_input) {
    UINT status;

    while(1) {
        // Thread logic
        tx_thread_sleep(100);  // Sleep 100 ticks
    }
}

void tx_application_define(void *first_unused_memory) {
    UINT status = tx_thread_create(
        &my_thread,              // Thread control block
        "My Thread",             // Name
        my_thread_entry,         // Entry function
        0x1234,                  // Input parameter
        thread_stack,            // Stack start
        sizeof(thread_stack),    // Stack size
        16,                      // Priority (0-31, 0 highest)
        16,                      // Preemption threshold
        TX_NO_TIME_SLICE,        // Time slice (disabled)
        TX_AUTO_START            // Auto start
    );

    if (status != TX_SUCCESS) {
        // Handle error
    }
}
```

### Thread Control

```c
// Suspend and resume
tx_thread_suspend(&my_thread);
tx_thread_resume(&my_thread);

// Priority management
tx_thread_priority_change(&my_thread, 10, &old_priority);

// Preemption threshold (prevents priority inversion)
tx_thread_preemption_change(&my_thread, 5, &old_threshold);

// Time slicing
tx_thread_time_slice_change(&my_thread, 10, &old_time_slice);

// Terminate and delete
tx_thread_terminate(&my_thread);
tx_thread_delete(&my_thread);

// Relinquish CPU to other threads at same priority
tx_thread_relinquish();
```

### Thread States

- **READY**: Ready to execute
- **COMPLETED**: Execution finished
- **TERMINATED**: Terminated by another thread
- **SUSPENDED**: Suspended by application
- **SLEEP**: Sleeping for specific time
- **QUEUE**: Waiting on queue
- **SEMAPHORE**: Waiting on semaphore
- **EVENT**: Waiting on event flags
- **MEMORY**: Waiting on memory
- **MUTEX**: Waiting on mutex

## Message Queues

### Queue Operations

```c
TX_QUEUE my_queue;
UCHAR queue_area[100 * sizeof(ULONG)];

// Create queue (supports 1-16 ULONGs per message)
tx_queue_create(
    &my_queue,
    "My Queue",
    TX_1_ULONG,              // Message size (TX_1_ULONG to TX_16_ULONG)
    queue_area,
    sizeof(queue_area)
);

// Send message (to back of queue)
ULONG message[4] = {0x12345678, 0xABCDEF00, 0x11111111, 0x22222222};
tx_queue_send(&my_queue, message, TX_WAIT_FOREVER);

// Send to front (priority message)
tx_queue_front_send(&my_queue, message, TX_NO_WAIT);

// Receive message
ULONG received[4];
tx_queue_receive(&my_queue, received, TX_WAIT_FOREVER);

// Query queue information
ULONG enqueued, available;
TX_THREAD *first_suspended;
tx_queue_info_get(&my_queue, TX_NULL, &enqueued,
                  &available, &first_suspended, TX_NULL, TX_NULL);

// Flush all messages
tx_queue_flush(&my_queue);

// Delete queue
tx_queue_delete(&my_queue);
```

## Semaphores

### Counting Semaphores

```c
TX_SEMAPHORE my_semaphore;

// Create counting semaphore (initial count = 3)
tx_semaphore_create(&my_semaphore, "My Semaphore", 3);

// Get semaphore (decrements count)
UINT status = tx_semaphore_get(&my_semaphore, TX_WAIT_FOREVER);

if (status == TX_SUCCESS) {
    // Access shared resource
}

// Put semaphore (increments count)
tx_semaphore_put(&my_semaphore);

// Prioritized put (wakes highest priority thread)
tx_semaphore_prioritize(&my_semaphore);

// Ceiling put (sets maximum count)
tx_semaphore_ceiling_put(&my_semaphore, 10);

// Delete semaphore
tx_semaphore_delete(&my_semaphore);
```

### Binary Semaphore Pattern

```c
TX_SEMAPHORE binary_sem;

// Binary semaphore: initial count = 0
tx_semaphore_create(&binary_sem, "Binary Sem", 0);

// Signal (producer)
tx_semaphore_put(&binary_sem);

// Wait (consumer)
tx_semaphore_get(&binary_sem, TX_WAIT_FOREVER);
```

## Mutex

### Mutex with Priority Inheritance

```c
TX_MUTEX my_mutex;

// Create mutex with priority inheritance
tx_mutex_create(&my_mutex, "My Mutex", TX_INHERIT);

// Alternatives:
// TX_NO_INHERIT - no priority inheritance
// TX_INHERIT - priority inheritance enabled

// Get mutex (with deadlock detection)
UINT status = tx_mutex_get(&my_mutex, TX_WAIT_FOREVER);

if (status == TX_SUCCESS) {
    // Critical section - access shared resource

    // Release mutex
    tx_mutex_put(&my_mutex);
}

// Priority inheritance prevents priority inversion
// Lower priority thread inherits priority of waiting higher priority thread
```

### Recursive Mutex Locks

```c
// ThreadX mutexes support recursive locking
tx_mutex_get(&my_mutex, TX_WAIT_FOREVER);
tx_mutex_get(&my_mutex, TX_WAIT_FOREVER);  // Same thread, succeeds

// Must release same number of times
tx_mutex_put(&my_mutex);
tx_mutex_put(&my_mutex);
```

## Event Flags

Event flags provide flexible thread synchronization based on boolean conditions.

```c
TX_EVENT_FLAGS_GROUP my_events;

// Create event flags group (32 flags per group)
tx_event_flags_create(&my_events, "My Events");

// Set event flags (OR operation)
#define EVENT_FLAG_1    0x00000001
#define EVENT_FLAG_2    0x00000002
#define EVENT_FLAG_3    0x00000004

tx_event_flags_set(&my_events, EVENT_FLAG_1 | EVENT_FLAG_2, TX_OR);

// Clear event flags (AND operation)
tx_event_flags_set(&my_events, ~EVENT_FLAG_1, TX_AND);

// Wait for event flags - AND condition (all flags must be set)
ULONG actual_flags;
tx_event_flags_get(
    &my_events,
    EVENT_FLAG_1 | EVENT_FLAG_2,    // Requested flags
    TX_AND,                          // Wait for ALL flags
    &actual_flags,                   // Actual flags returned
    TX_WAIT_FOREVER
);

// Wait for event flags - OR condition (any flag can be set)
tx_event_flags_get(
    &my_events,
    EVENT_FLAG_1 | EVENT_FLAG_2,    // Requested flags
    TX_OR,                           // Wait for ANY flag
    &actual_flags,
    100                              // Timeout in ticks
);

// Clear flags after getting them
tx_event_flags_get(
    &my_events,
    EVENT_FLAG_1,
    TX_OR_CLEAR,                     // Clear after getting
    &actual_flags,
    TX_WAIT_FOREVER
);

// Delete event flags group
tx_event_flags_delete(&my_events);
```

## Memory Management

### Block Memory Pools

Fixed-size block allocation for deterministic performance.

```c
TX_BLOCK_POOL my_pool;
UCHAR pool_area[10000];

// Create block pool (64-byte blocks)
tx_block_pool_create(
    &my_pool,
    "My Block Pool",
    64,                      // Block size
    pool_area,
    sizeof(pool_area)
);

// Allocate block
VOID *block_ptr;
UINT status = tx_block_allocate(&my_pool, &block_ptr, TX_WAIT_FOREVER);

if (status == TX_SUCCESS) {
    // Use block

    // Release block
    tx_block_release(block_ptr);
}

// Query pool information
ULONG available_blocks, total_blocks;
tx_block_pool_info_get(&my_pool, TX_NULL, &available_blocks,
                       &total_blocks, TX_NULL, TX_NULL, TX_NULL);

// Delete pool
tx_block_pool_delete(&my_pool);
```

### Byte Memory Pools

Variable-size allocation (like malloc, but RTOS-aware).

```c
TX_BYTE_POOL my_byte_pool;
UCHAR pool_area[10000];

// Create byte pool
tx_byte_pool_create(
    &my_byte_pool,
    "My Byte Pool",
    pool_area,
    sizeof(pool_area)
);

// Allocate memory
VOID *memory_ptr;
tx_byte_allocate(&my_byte_pool, &memory_ptr, 256, TX_WAIT_FOREVER);

// Use memory
if (memory_ptr != TX_NULL) {
    // Release memory
    tx_byte_release(memory_ptr);
}

// Prioritize suspended threads
tx_byte_pool_prioritize(&my_byte_pool);

// Delete pool
tx_byte_pool_delete(&my_byte_pool);
```

## Application Timers

```c
TX_TIMER my_timer;

void timer_expiration_function(ULONG timer_input) {
    // Timer expired - this runs in timer thread context
    // Keep this function short and non-blocking
}

// Create one-shot timer
tx_timer_create(
    &my_timer,
    "My Timer",
    timer_expiration_function,
    0x1234,                  // Input to expiration function
    100,                     // Initial ticks (delay before first expiration)
    0,                       // Reschedule ticks (0 = one-shot)
    TX_NO_ACTIVATE           // Don't activate yet
);

// Create periodic timer
tx_timer_create(
    &my_timer,
    "Periodic Timer",
    timer_expiration_function,
    0,
    100,                     // Initial ticks
    100,                     // Reschedule ticks (periodic)
    TX_AUTO_ACTIVATE         // Activate immediately
);

// Activate timer
tx_timer_activate(&my_timer);

// Deactivate timer
tx_timer_deactivate(&my_timer);

// Change timer settings
tx_timer_change(&my_timer, 200, 200);  // New initial and reschedule ticks

// Delete timer
tx_timer_delete(&my_timer);
```

## Interrupt Management

### Interrupt Service Routines

```c
void my_isr(void) {
    // Notify ThreadX we're in ISR context
    // (Some ports do this automatically)

    // ISR processing - keep minimal

    // Wake up thread using semaphore
    tx_semaphore_put(&my_isr_semaphore);

    // Or set event flag
    tx_event_flags_set(&my_events, ISR_EVENT, TX_OR);
}

// Deferred processing thread
void isr_processing_thread(ULONG input) {
    while(1) {
        // Wait for ISR signal
        tx_semaphore_get(&my_isr_semaphore, TX_WAIT_FOREVER);

        // Do time-consuming processing here
        // (not in ISR context)
    }
}
```

### Nested Interrupt Support

```c
// ThreadX automatically handles nested interrupts
// Save/restore is handled by context switching mechanism

// Disable interrupts when needed
TX_INTERRUPT_SAVE_AREA
TX_DISABLE  // Save interrupt state and disable

// Critical section code

TX_RESTORE  // Restore previous interrupt state
```

## Time Management

```c
// Get current time (ticks since system start)
ULONG current_time = tx_time_get();

// Set current time
tx_time_set(1000);

// Sleep for specific ticks
tx_thread_sleep(100);

// Configure system timer
// Timer tick usually occurs every 10ms (100Hz)
// Configured in tx_initialize_low_level.s or similar
```

## System Information

```c
// Get ThreadX version
ULONG version = tx_kernel_version_get();
// Returns format: 0x06010000 for version 6.1

// Get system state
TX_THREAD *thread;
ULONG state = tx_thread_identify();  // Returns current thread

// Performance information
ULONG resumptions, suspensions, solicited_preemptions;
ULONG interrupt_preemptions, priority_inversions, time_slices;
ULONG relinquishes, timeouts, waits;

tx_thread_performance_info_get(
    &my_thread,
    &resumptions,
    &suspensions,
    &solicited_preemptions,
    &interrupt_preemptions,
    &priority_inversions,
    &time_slices,
    &relinquishes,
    &timeouts,
    &waits,
    TX_NULL
);
```

## Priority Inversion Prevention

ThreadX provides multiple mechanisms to prevent priority inversion:

### 1. Preemption Threshold

```c
// Priority = 10, Preemption Threshold = 5
// Thread runs at priority 10, but can only be preempted by priorities 0-4
tx_thread_create(&my_thread, "Thread", entry_func, 0,
                 stack, 1024,
                 10,                    // Priority
                 5,                     // Preemption threshold
                 TX_NO_TIME_SLICE, TX_AUTO_START);
```

### 2. Priority Inheritance (Mutexes)

```c
// Create mutex with priority inheritance
tx_mutex_create(&my_mutex, "Mutex", TX_INHERIT);

// Low priority thread gets mutex
// High priority thread waits
// Low priority thread inherits high priority temporarily
```

## Error Handling

All ThreadX services return status codes:

```c
// Common return values
TX_SUCCESS              // Successful completion
TX_DELETED              // Object was deleted
TX_POOL_ERROR          // Invalid memory pool
TX_PTR_ERROR           // Invalid pointer
TX_WAIT_ERROR          // Invalid wait option
TX_SIZE_ERROR          // Invalid size
TX_GROUP_ERROR         // Invalid group pointer
TX_NO_EVENTS           // No events satisfied request
TX_OPTION_ERROR        // Invalid option
TX_QUEUE_ERROR         // Invalid queue pointer
TX_QUEUE_EMPTY         // Queue is empty
TX_QUEUE_FULL          // Queue is full
TX_SEMAPHORE_ERROR     // Invalid semaphore pointer
TX_NO_INSTANCE         // No instance available
TX_THREAD_ERROR        // Invalid thread pointer
TX_PRIORITY_ERROR      // Invalid priority
TX_START_ERROR         // Invalid auto-start
TX_DELETE_ERROR        // Thread not terminated
TX_RESUME_ERROR        // Thread not suspended
TX_CALLER_ERROR        // Invalid caller
TX_SUSPEND_ERROR       // Thread already suspended
TX_TIMER_ERROR         // Invalid timer pointer
TX_TICK_ERROR          // Invalid tick value
TX_ACTIVATE_ERROR      // Timer already active
TX_THRESH_ERROR        // Invalid preemption threshold
TX_SUSPEND_LIFTED      // Delayed suspension lifted
TX_WAIT_ABORTED        // Wait aborted
TX_MUTEX_ERROR         // Invalid mutex pointer
TX_NOT_AVAILABLE       // Service not available
TX_NOT_OWNED           // Mutex not owned
TX_INHERIT_ERROR       // Invalid priority inheritance
TX_NOT_DONE            // Service not completed
```

### Example Error Handling

```c
UINT status = tx_queue_send(&my_queue, &message, 100);

switch(status) {
    case TX_SUCCESS:
        // Message sent successfully
        break;
    case TX_QUEUE_FULL:
        // Queue full, handle overflow
        break;
    case TX_WAIT_ABORTED:
        // Wait was aborted by another thread
        break;
    case TX_QUEUE_ERROR:
        // Invalid queue pointer
        break;
    default:
        // Unexpected error
        break;
}
```

## Best Practices

1. **Stack Sizing**: Use `tx_thread_stack_error_notify()` to detect stack overflow
2. **Priority Assignment**: Reserve highest priorities (0-2) for critical ISR threads
3. **Avoid Busy Waiting**: Use semaphores/events instead of polling
4. **Timer Callbacks**: Keep timer expiration functions short and non-blocking
5. **Memory Pools**: Prefer block pools over byte pools for predictable timing
6. **Mutex vs Semaphore**: Use mutexes for resource protection, semaphores for signaling
7. **Event Flags**: Use for complex synchronization conditions
8. **Preemption Threshold**: Set threshold = priority for most threads
9. **Debug Support**: Enable `TX_ENABLE_STACK_CHECKING` during development
10. **TraceX Integration**: Use Azure RTOS TraceX for system analysis

## Configuration Options

ThreadX behavior is configured in `tx_user.h`:

```c
// Maximum priority levels (default 32)
#define TX_MAX_PRIORITIES               32

// Minimum stack size
#define TX_MINIMUM_STACK               200

// Enable stack checking
#define TX_ENABLE_STACK_CHECKING

// Disable time-slice
#define TX_DISABLE_TIME_SLICE

// Timer thread priority
#define TX_TIMER_THREAD_PRIORITY        0

// Timer thread stack size
#define TX_TIMER_THREAD_STACK_SIZE   1024

// Enable event trace
#define TX_ENABLE_EVENT_TRACE

// Disable error checking (for production)
#define TX_DISABLE_ERROR_CHECKING

// Inline ThreadX services for performance
#define TX_INLINE_THREAD_RESUME_SUSPEND
```

## Performance Characteristics

- **Context Switch**: ~200 CPU cycles (Cortex-M4)
- **Interrupt Latency**: Minimal (< 10 cycles to ISR)
- **Service Call Overhead**: 50-150 cycles depending on operation
- **Memory Footprint**: 2-20 KB depending on configuration
- **RAM Usage**: ~1 KB for kernel + per-thread overhead (~350 bytes/thread)
- **Deterministic**: All operations have bounded execution time

## Supported Architectures

ThreadX supports 50+ processor families:
- ARM: Cortex-M, Cortex-R, Cortex-A, ARM7/9/11
- RISC-V: RV32, RV64
- x86/x64
- MIPS
- PowerPC
- Renesas: RX, SH, Synergy
- Microchip: PIC32, AVR32
- And many more...

## Azure RTOS Integration

ThreadX is part of the Azure RTOS family:

- **FileX**: Embedded file system (FAT compatible)
- **NetX Duo**: TCP/IP stack (IPv4/IPv6)
- **USBX**: USB host/device stack
- **GUIX**: Embedded GUI framework
- **LevelX**: NAND/NOR flash wear leveling
- **TraceX**: System analysis tool

All components are designed to work seamlessly together.

## Safety Certifications

ThreadX is pre-certified for:
- **IEC 61508** SIL 4 (Industrial)
- **IEC 62304** Class C (Medical)
- **ISO 26262** ASIL D (Automotive)
- **EN 50128** SW-SIL 4 (Railway)
- **UL/IEC 60730** Class B (Appliances)

Certification packages include safety manuals, test reports, and compliance documentation.

## License and Availability

- **License**: MIT License (open source)
- **Repository**: https://github.com/azure-rtos/threadx
- **Documentation**: https://docs.microsoft.com/azure/rtos/threadx
- **No royalties**: Free for commercial use
- **Community support**: GitHub issues and discussions
- **Commercial support**: Available through Microsoft and partners

ThreadX is ideal for resource-constrained embedded systems requiring deterministic real-time performance, safety certification, or long-term support.
