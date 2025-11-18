# FreeRTOS

FreeRTOS is a real-time operating system kernel for embedded devices. It's designed to be small, simple, and easy to use, providing deterministic real-time behavior with minimal resource overhead.

## Core Concepts

**Tasks**: Independent threads of execution with their own stack and priority level
**Queues**: FIFO buffers for inter-task communication and data passing
**Semaphores**: Synchronization primitives for signaling and resource counting
**Mutexes**: Mutual exclusion locks with priority inheritance to prevent priority inversion
**Timers**: Software timers that execute callbacks in timer daemon task context
**Event Groups**: Synchronization mechanism for managing multiple event flags
**Task Notifications**: Lightweight alternative to semaphores and queues for task signaling
**Stream Buffers**: Efficient byte stream passing between tasks or interrupts

## Task Management

### Task Creation

```c
#include "FreeRTOS.h"
#include "task.h"

void vTaskFunction(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000);

    for(;;) {
        // Task code executes every 1000ms
        printf("Task running\n");

        // Delay until next cycle (more precise than vTaskDelay)
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

void main(void) {
    TaskHandle_t xHandle = NULL;

    BaseType_t xReturned = xTaskCreate(
        vTaskFunction,      // Function pointer
        "TaskName",         // Descriptive name
        configMINIMAL_STACK_SIZE,  // Stack size in words
        NULL,               // Task parameters
        tskIDLE_PRIORITY + 1,  // Priority
        &xHandle            // Task handle
    );

    if(xReturned == pdPASS) {
        vTaskStartScheduler();  // Start scheduler
    }

    for(;;);  // Should never reach here if scheduler starts
}
```

### Task States

Tasks can be in one of four states:

- **Running**: Currently executing on the CPU
- **Ready**: Ready to run but not currently executing
- **Blocked**: Waiting for an event (timeout, semaphore, queue, etc.)
- **Suspended**: Not available to scheduler (explicitly suspended)

### Task Priority and Scheduling

```c
// Priority levels
#define PRIORITY_IDLE       0
#define PRIORITY_LOW        1
#define PRIORITY_NORMAL     2
#define PRIORITY_HIGH       3
#define PRIORITY_REALTIME   4

// Change task priority at runtime
vTaskPrioritySet(xHandle, PRIORITY_HIGH);
UBaseType_t uxPriority = uxTaskPriorityGet(xHandle);

// Task suspension and resumption
vTaskSuspend(xHandle);    // Suspend task
vTaskResume(xHandle);     // Resume task
xTaskResumeFromISR(xHandle);  // Resume from ISR

// Task deletion
vTaskDelete(xHandle);     // Delete specified task
vTaskDelete(NULL);        // Delete current task
```

## Queues

Queues provide thread-safe FIFO communication between tasks and interrupts.

```c
#include "queue.h"

typedef struct {
    int sensor_id;
    float value;
    TickType_t timestamp;
} SensorData_t;

QueueHandle_t xDataQueue;

void vProducerTask(void *pvParameters) {
    SensorData_t data;

    // Create queue for 10 items
    xDataQueue = xQueueCreate(10, sizeof(SensorData_t));

    if(xDataQueue != NULL) {
        for(;;) {
            data.sensor_id = 1;
            data.value = read_sensor();
            data.timestamp = xTaskGetTickCount();

            // Send to queue (wait up to 100ms if full)
            if(xQueueSend(xDataQueue, &data, pdMS_TO_TICKS(100)) != pdPASS) {
                // Queue full, handle error
            }

            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
}

void vConsumerTask(void *pvParameters) {
    SensorData_t received;

    for(;;) {
        // Wait indefinitely for data
        if(xQueueReceive(xDataQueue, &received, portMAX_DELAY) == pdPASS) {
            printf("Sensor %d: %.2f at tick %lu\n",
                   received.sensor_id,
                   received.value,
                   received.timestamp);
        }
    }
}

// Queue utility functions
UBaseType_t uxMessagesWaiting = uxQueueMessagesWaiting(xDataQueue);
UBaseType_t uxSpacesAvailable = uxQueueSpacesAvailable(xDataQueue);
xQueueReset(xDataQueue);  // Empty the queue
```

### Queue Operations from ISR

```c
void vExampleISR(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    SensorData_t data = {0};

    // Send from ISR
    xQueueSendFromISR(xDataQueue, &data, &xHigherPriorityTaskWoken);

    // Context switch if higher priority task was woken
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

## Semaphores

### Binary Semaphores

Used for synchronization and signaling between tasks.

```c
#include "semphr.h"

SemaphoreHandle_t xBinarySemaphore;

void vTaskSignaler(void *pvParameters) {
    xBinarySemaphore = xSemaphoreCreateBinary();

    for(;;) {
        // Perform work
        process_data();

        // Signal completion
        xSemaphoreGive(xBinarySemaphore);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void vTaskWaiter(void *pvParameters) {
    for(;;) {
        // Wait for signal
        if(xSemaphoreTake(xBinarySemaphore, portMAX_DELAY) == pdTRUE) {
            // Semaphore acquired, respond to event
            handle_completion();
        }
    }
}
```

### Counting Semaphores

Used for resource management and event counting.

```c
// Create semaphore with max count of 5, initial count of 5
SemaphoreHandle_t xCountingSemaphore = xSemaphoreCreateCounting(5, 5);

void vResourceUser(void *pvParameters) {
    for(;;) {
        // Wait for resource (decrements count)
        if(xSemaphoreTake(xCountingSemaphore, pdMS_TO_TICKS(100)) == pdTRUE) {
            // Use resource
            use_limited_resource();

            // Release resource (increments count)
            xSemaphoreGive(xCountingSemaphore);
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
```

## Mutexes

Mutexes provide mutual exclusion with priority inheritance to prevent priority inversion.

```c
SemaphoreHandle_t xMutex;

void vCriticalTask(void *pvParameters) {
    xMutex = xSemaphoreCreateMutex();

    for(;;) {
        // Acquire mutex (blocks if already taken)
        if(xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE) {
            // Critical section - protected resource access
            shared_resource++;
            update_display(shared_resource);

            // Release mutex
            xSemaphoreGive(xMutex);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
```

### Recursive Mutexes

Allow the same task to take a mutex multiple times.

```c
SemaphoreHandle_t xRecursiveMutex;

void vRecursiveFunction(int depth) {
    if(depth > 0) {
        xSemaphoreTakeRecursive(xRecursiveMutex, portMAX_DELAY);
        vRecursiveFunction(depth - 1);
        xSemaphoreGiveRecursive(xRecursiveMutex);
    }
}
```

## Software Timers

Software timers execute callbacks in the timer daemon task context.

```c
#include "timers.h"

TimerHandle_t xAutoReloadTimer;
TimerHandle_t xOneShotTimer;

void vTimerCallback(TimerHandle_t xTimer) {
    // Called when timer expires
    uint32_t ulCount = (uint32_t)pvTimerGetTimerID(xTimer);
    printf("Timer expired, count: %lu\n", ulCount);
}

void vTimerSetup(void) {
    // Auto-reload timer (periodic, 1000ms)
    xAutoReloadTimer = xTimerCreate(
        "AutoReload",               // Name
        pdMS_TO_TICKS(1000),       // Period
        pdTRUE,                     // Auto-reload
        (void *)0,                  // Timer ID
        vTimerCallback              // Callback
    );

    // One-shot timer (single execution, 5000ms)
    xOneShotTimer = xTimerCreate(
        "OneShot",
        pdMS_TO_TICKS(5000),
        pdFALSE,                    // One-shot
        (void *)1,
        vTimerCallback
    );

    // Start timers
    if(xAutoReloadTimer != NULL) {
        xTimerStart(xAutoReloadTimer, 0);
    }

    if(xOneShotTimer != NULL) {
        xTimerStart(xOneShotTimer, 0);
    }
}

// Timer control
xTimerStop(xAutoReloadTimer, 0);
xTimerChangePeriod(xAutoReloadTimer, pdMS_TO_TICKS(2000), 0);
xTimerReset(xOneShotTimer, 0);  // Restart timer
```

## Event Groups

Event groups allow tasks to wait for multiple conditions (event bits).

```c
#include "event_groups.h"

EventGroupHandle_t xEventGroup;

// Define event bits
#define BIT_0  (1 << 0)  // Sensor ready
#define BIT_1  (1 << 1)  // Data valid
#define BIT_2  (1 << 2)  // Calibration complete
#define ALL_SYNC_BITS (BIT_0 | BIT_1 | BIT_2)

void vEventSetter(void *pvParameters) {
    xEventGroup = xEventGroupCreate();

    for(;;) {
        // Set individual bits
        xEventGroupSetBits(xEventGroup, BIT_0);
        vTaskDelay(pdMS_TO_TICKS(100));

        xEventGroupSetBits(xEventGroup, BIT_1);
        vTaskDelay(pdMS_TO_TICKS(100));

        xEventGroupSetBits(xEventGroup, BIT_2);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void vEventWaiter(void *pvParameters) {
    EventBits_t uxBits;

    for(;;) {
        // Wait for all bits to be set
        uxBits = xEventGroupWaitBits(
            xEventGroup,
            ALL_SYNC_BITS,          // Bits to wait for
            pdTRUE,                 // Clear on exit
            pdTRUE,                 // Wait for all bits
            portMAX_DELAY           // Wait indefinitely
        );

        if((uxBits & ALL_SYNC_BITS) == ALL_SYNC_BITS) {
            printf("All events occurred\n");
        }
    }
}

// Event group utilities
EventBits_t uxCurrentBits = xEventGroupGetBits(xEventGroup);
xEventGroupClearBits(xEventGroup, BIT_0);
```

## Task Notifications

Lightweight alternative to semaphores and queues for direct task-to-task communication.

```c
TaskHandle_t xHandleToNotify;

void vNotifyingTask(void *pvParameters) {
    for(;;) {
        // Perform work
        uint32_t ulValueToSend = 42;

        // Send notification
        xTaskNotify(
            xHandleToNotify,
            ulValueToSend,
            eSetValueWithOverwrite
        );

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void vNotifiedTask(void *pvParameters) {
    uint32_t ulNotificationValue;

    for(;;) {
        // Wait for notification
        if(xTaskNotifyWait(
            0x00,                   // Don't clear on entry
            0xFFFFFFFF,             // Clear all on exit
            &ulNotificationValue,
            portMAX_DELAY) == pdTRUE) {

            printf("Received: %lu\n", ulNotificationValue);
        }
    }
}

// Notification from ISR
void vExampleISR(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    vTaskNotifyGiveFromISR(xHandleToNotify, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

## Memory Management

FreeRTOS provides multiple heap allocation schemes.

### Heap Schemes

**heap_1**: Simplest, no freeing (deterministic)
**heap_2**: Permits freeing, no coalescence
**heap_3**: Wraps malloc/free (thread-safe)
**heap_4**: Coalescence, suitable for fragmentation-prone apps
**heap_5**: Like heap_4, supports multiple memory regions

### Memory API

```c
// Allocate memory
void *pvBuffer = pvPortMalloc(100);

// Free memory
vPortFree(pvBuffer);

// Get heap statistics
HeapStats_t xHeapStats;
vPortGetHeapStats(&xHeapStats);

printf("Available heap: %zu bytes\n", xHeapStats.xAvailableHeapSpaceInBytes);
printf("Largest free block: %zu bytes\n", xHeapStats.xSizeOfLargestFreeBlockInBytes);
printf("Minimum ever free: %zu bytes\n", xHeapStats.xMinimumEverFreeBytesRemaining);

// Get free heap size
size_t xFreeHeapSize = xPortGetFreeHeapSize();
size_t xMinimumEverFreeHeapSize = xPortGetMinimumEverFreeHeapSize();
```

### Static Allocation

Allocate memory at compile-time instead of runtime.

```c
// Enable static allocation in FreeRTOSConfig.h
#define configSUPPORT_STATIC_ALLOCATION 1

StaticTask_t xTaskBuffer;
StackType_t xStack[128];

TaskHandle_t xHandle = xTaskCreateStatic(
    vTaskFunction,
    "StaticTask",
    128,
    NULL,
    1,
    xStack,
    &xTaskBuffer
);
```

## Interrupt Handling

### ISR-Safe API Functions

FreeRTOS provides FromISR variants for interrupt context.

```c
void vUART_ISR(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    char cReceivedChar;

    // Read character from UART
    cReceivedChar = UART_ReadByte();

    // Send to queue from ISR
    xQueueSendFromISR(xRxQueue, &cReceivedChar, &xHigherPriorityTaskWoken);

    // Yield if necessary
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

### Critical Sections

```c
// Disable interrupts (short critical sections only)
taskENTER_CRITICAL();
// Critical code
taskEXIT_CRITICAL();

// From ISR (saves/restores interrupt state)
UBaseType_t uxSavedInterruptStatus;
uxSavedInterruptStatus = taskENTER_CRITICAL_FROM_ISR();
// Critical code
taskEXIT_CRITICAL_FROM_ISR(uxSavedInterruptStatus);
```

### Interrupt Priority

```c
// Configure interrupt priorities
// (implementation specific to hardware)

// IMPORTANT: FreeRTOS API can only be called from interrupts
// with priority at or below configMAX_SYSCALL_INTERRUPT_PRIORITY

// On Cortex-M, lower numeric value = higher priority
// Set priorities appropriately in FreeRTOSConfig.h
#define configMAX_SYSCALL_INTERRUPT_PRIORITY  5

// High priority interrupt (cannot use FreeRTOS API)
void vHighPriorityISR(void) {
    // No FreeRTOS calls allowed
    toggle_gpio_fast();
}

// Lower priority interrupt (can use FreeRTOS API)
void vLowPriorityISR(void) {
    // Can safely call FromISR functions
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(xSemaphore, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

## Configuration

Key settings in `FreeRTOSConfig.h`:

```c
// Scheduling
#define configUSE_PREEMPTION              1
#define configUSE_TIME_SLICING            1
#define configUSE_IDLE_HOOK               0
#define configUSE_TICK_HOOK               0

// Tick rate (Hz)
#define configTICK_RATE_HZ                1000

// CPU clock (Hz)
#define configCPU_CLOCK_HZ                80000000

// Priorities
#define configMAX_PRIORITIES              5
#define configMINIMAL_STACK_SIZE          128

// Heap size (bytes)
#define configTOTAL_HEAP_SIZE             10240

// Features
#define configUSE_MUTEXES                 1
#define configUSE_RECURSIVE_MUTEXES       1
#define configUSE_COUNTING_SEMAPHORES     1
#define configUSE_QUEUE_SETS              1
#define configUSE_TIMERS                  1
#define configUSE_TASK_NOTIFICATIONS      1

// Memory allocation
#define configSUPPORT_STATIC_ALLOCATION   1
#define configSUPPORT_DYNAMIC_ALLOCATION  1

// Runtime statistics
#define configGENERATE_RUN_TIME_STATS     1
#define configUSE_TRACE_FACILITY          1
#define configUSE_STATS_FORMATTING_FUNCTIONS 1

// Stack overflow detection
#define configCHECK_FOR_STACK_OVERFLOW    2

// Assert
#define configASSERT(x)  if(!(x)) { taskDISABLE_INTERRUPTS(); for(;;); }
```

## Runtime Statistics

Monitor task execution and performance.

```c
void vTaskList_Display(void) {
    char pcWriteBuffer[512];

    // Task list (name, state, priority, stack, task number)
    vTaskList(pcWriteBuffer);
    printf("%s\n", pcWriteBuffer);

    // Runtime statistics (name, runtime, percentage)
    vTaskGetRunTimeStats(pcWriteBuffer);
    printf("%s\n", pcWriteBuffer);
}

// Get individual task information
TaskStatus_t xTaskDetails;
vTaskGetInfo(
    xHandle,
    &xTaskDetails,
    pdTRUE,         // Include stack high water mark
    eInvalid        // Get current state
);

printf("Stack high water mark: %u\n", xTaskDetails.usStackHighWaterMark);
printf("Task state: %d\n", xTaskDetails.eCurrentState);
printf("Task priority: %u\n", xTaskDetails.uxCurrentPriority);
```

## Best Practices

### Task Design

1. **Keep ISRs short**: Defer processing to tasks using queues/semaphores
2. **Use appropriate delays**: `vTaskDelayUntil()` for periodic tasks
3. **Avoid polling**: Use blocking calls with timeouts
4. **One responsibility per task**: Follow single responsibility principle
5. **Minimize critical sections**: Keep interrupt-disabled time minimal

### Priority Assignment

```c
// Example priority scheme
#define PRIORITY_IDLE           0  // Idle task (system)
#define PRIORITY_BACKGROUND     1  // Background processing
#define PRIORITY_NORMAL         2  // Standard tasks
#define PRIORITY_UI             3  // User interface
#define PRIORITY_COMMS          4  // Time-critical communication
#define PRIORITY_CONTROL        5  // Real-time control loops
#define PRIORITY_SAFETY         6  // Safety-critical tasks
```

### Stack Size Optimization

```c
// Monitor stack usage
UBaseType_t uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
printf("Unused stack: %u words\n", uxHighWaterMark);

// Start with generous size, then reduce based on measurements
// Words, not bytes (multiply by sizeof(StackType_t) for bytes)
```

### Resource Management

```c
// RAII pattern for mutex
typedef struct {
    SemaphoreHandle_t mutex;
    BaseType_t locked;
} MutexGuard_t;

MutexGuard_t MutexGuard_Lock(SemaphoreHandle_t mutex) {
    MutexGuard_t guard = {mutex, pdFALSE};
    if(xSemaphoreTake(mutex, portMAX_DELAY) == pdTRUE) {
        guard.locked = pdTRUE;
    }
    return guard;
}

void MutexGuard_Unlock(MutexGuard_t *guard) {
    if(guard->locked) {
        xSemaphoreGive(guard->mutex);
        guard->locked = pdFALSE;
    }
}
```

## Common Pitfalls

### Priority Inversion

```c
// Problem: Low priority task holds mutex, high priority task blocks
// Solution: Use mutexes (not binary semaphores) for priority inheritance

// BAD: Using binary semaphore for mutual exclusion
SemaphoreHandle_t xBadLock = xSemaphoreCreateBinary();
xSemaphoreGive(xBadLock);  // Initialize

// GOOD: Using mutex with priority inheritance
SemaphoreHandle_t xGoodLock = xSemaphoreCreateMutex();
```

### Stack Overflow

```c
// Enable stack overflow detection
#define configCHECK_FOR_STACK_OVERFLOW 2

// Implement hook
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    // Log error and halt
    printf("Stack overflow in task: %s\n", pcTaskName);
    for(;;);
}
```

### Deadlock

```c
// Problem: Circular wait for resources
// Solution: Always acquire mutexes in same order

SemaphoreHandle_t xMutexA, xMutexB;

void vSafeTask1(void *pvParameters) {
    // Always acquire A then B
    xSemaphoreTake(xMutexA, portMAX_DELAY);
    xSemaphoreTake(xMutexB, portMAX_DELAY);

    // Critical section

    xSemaphoreGive(xMutexB);
    xSemaphoreGive(xMutexA);
}

void vSafeTask2(void *pvParameters) {
    // Always acquire A then B (same order)
    xSemaphoreTake(xMutexA, portMAX_DELAY);
    xSemaphoreTake(xMutexB, portMAX_DELAY);

    // Critical section

    xSemaphoreGive(xMutexB);
    xSemaphoreGive(xMutexA);
}
```

### Inappropriate API Usage

```c
// WRONG: Calling non-ISR function from ISR
void vBadISR(void) {
    xQueueSend(xQueue, &data, 0);  // DON'T DO THIS
}

// CORRECT: Use FromISR variant
void vGoodISR(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(xQueue, &data, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

## Debugging Techniques

### Trace Hooks

```c
#define configUSE_TRACE_FACILITY 1

// Task switched in
void vApplicationTaskSwitchedInHook(void) {
    TaskHandle_t xHandle = xTaskGetCurrentTaskHandle();
    char *pcTaskName = pcTaskGetName(xHandle);
    printf("Switched to: %s\n", pcTaskName);
}

// Tick hook (called every tick)
void vApplicationTickHook(void) {
    // Lightweight monitoring only
}

// Idle hook (called by idle task)
void vApplicationIdleHook(void) {
    // Background processing, watchdog feeding
}
```

### Assertions

```c
// Define assert macro
#define configASSERT(x) \
    if(!(x)) { \
        printf("Assert failed: %s:%d\n", __FILE__, __LINE__); \
        taskDISABLE_INTERRUPTS(); \
        for(;;); \
    }

// Use in code
configASSERT(xQueue != NULL);
configASSERT(xSemaphoreTake(xMutex, portMAX_DELAY) == pdTRUE);
```

### Queue Set Monitoring

```c
// Monitor multiple queues
QueueSetHandle_t xQueueSet = xQueueCreateSet(20);

xQueueAddToSet(xQueue1, xQueueSet);
xQueueAddToSet(xQueue2, xQueueSet);

QueueSetMemberHandle_t xActivatedMember;
xActivatedMember = xQueueSelectFromSet(xQueueSet, portMAX_DELAY);

if(xActivatedMember == xQueue1) {
    // Data available on queue 1
} else if(xActivatedMember == xQueue2) {
    // Data available on queue 2
}
```

## Real-World Example

Complete embedded system with multiple tasks and IPC mechanisms.

```c
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#include "timers.h"

// Shared resources
QueueHandle_t xSensorQueue;
SemaphoreHandle_t xDisplayMutex;
TimerHandle_t xWatchdogTimer;

typedef struct {
    uint8_t sensor_id;
    float temperature;
    float humidity;
} SensorReading_t;

// Sensor reading task (high priority)
void vSensorTask(void *pvParameters) {
    SensorReading_t reading;
    TickType_t xLastWakeTime = xTaskGetTickCount();

    for(;;) {
        // Read sensors
        reading.sensor_id = 1;
        reading.temperature = read_temperature();
        reading.humidity = read_humidity();

        // Send to processing queue
        if(xQueueSend(xSensorQueue, &reading, 0) != pdPASS) {
            // Queue full, log error
        }

        // Periodic execution every 500ms
        vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(500));
    }
}

// Data processing task (medium priority)
void vProcessingTask(void *pvParameters) {
    SensorReading_t reading;

    for(;;) {
        // Wait for sensor data
        if(xQueueReceive(xSensorQueue, &reading, portMAX_DELAY) == pdPASS) {
            // Process data
            if(reading.temperature > 30.0f) {
                activate_cooling();
            }

            // Update display (mutex protected)
            if(xSemaphoreTake(xDisplayMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                update_display(reading.temperature, reading.humidity);
                xSemaphoreGive(xDisplayMutex);
            }
        }
    }
}

// Communication task (medium priority)
void vCommTask(void *pvParameters) {
    char txBuffer[64];

    for(;;) {
        // Wait for command from UART
        if(uart_data_available()) {
            process_command();
        }

        // Periodic status transmission
        snprintf(txBuffer, sizeof(txBuffer), "Status: OK\n");
        uart_send(txBuffer, strlen(txBuffer));

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// Watchdog timer callback
void vWatchdogCallback(TimerHandle_t xTimer) {
    feed_hardware_watchdog();
}

// Idle hook for power management
void vApplicationIdleHook(void) {
    enter_low_power_mode();
}

// Stack overflow hook
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    printf("Stack overflow: %s\n", pcTaskName);
    for(;;);
}

// Main application
int main(void) {
    // Hardware initialization
    hardware_init();

    // Create synchronization primitives
    xSensorQueue = xQueueCreate(10, sizeof(SensorReading_t));
    xDisplayMutex = xSemaphoreCreateMutex();

    // Create watchdog timer (auto-reload, 1000ms)
    xWatchdogTimer = xTimerCreate(
        "Watchdog",
        pdMS_TO_TICKS(1000),
        pdTRUE,
        NULL,
        vWatchdogCallback
    );

    // Create tasks
    xTaskCreate(vSensorTask, "Sensor", 256, NULL, 3, NULL);
    xTaskCreate(vProcessingTask, "Process", 256, NULL, 2, NULL);
    xTaskCreate(vCommTask, "Comm", 256, NULL, 2, NULL);

    // Start watchdog timer
    xTimerStart(xWatchdogTimer, 0);

    // Start scheduler
    vTaskStartScheduler();

    // Should never reach here
    for(;;);

    return 0;
}
```

## Performance Optimization

### Context Switch Overhead

- Minimize task switching frequency
- Use task notifications instead of queues where possible
- Batch process queue messages
- Adjust tick rate based on timing requirements

### Memory Footprint

```c
// Reduce RAM usage
#define configMINIMAL_STACK_SIZE          64   // Tune per task
#define configTOTAL_HEAP_SIZE             4096 // Based on actual needs
#define configMAX_PRIORITIES              4    // Only what's needed

// Use static allocation for predictability
#define configSUPPORT_STATIC_ALLOCATION   1
#define configSUPPORT_DYNAMIC_ALLOCATION  0
```

### CPU Utilization

```c
// Monitor CPU usage
void vCPUUtilization(void) {
    TaskStatus_t *pxTaskStatusArray;
    UBaseType_t uxArraySize, x;
    uint32_t ulTotalRunTime, ulStatsAsPercentage;

    uxArraySize = uxTaskGetNumberOfTasks();
    pxTaskStatusArray = pvPortMalloc(uxArraySize * sizeof(TaskStatus_t));

    if(pxTaskStatusArray != NULL) {
        uxArraySize = uxTaskGetSystemState(pxTaskStatusArray,
                                           uxArraySize,
                                           &ulTotalRunTime);

        for(x = 0; x < uxArraySize; x++) {
            ulStatsAsPercentage = pxTaskStatusArray[x].ulRunTimeCounter /
                                  (ulTotalRunTime / 100);

            printf("%s\t\t%u%%\n",
                   pxTaskStatusArray[x].pcTaskName,
                   ulStatsAsPercentage);
        }

        vPortFree(pxTaskStatusArray);
    }
}
```

## Platform Support

FreeRTOS supports numerous architectures:

- **ARM Cortex-M** (M0, M0+, M3, M4, M7, M23, M33)
- **ARM Cortex-A** (A9, A53, A72)
- **ARM Cortex-R** (R4, R5)
- **RISC-V** (RV32, RV64)
- **x86** (IA-32, x86-64)
- **Xtensa** (ESP32, ESP8266)
- **AVR** (ATmega)
- **PIC** (PIC24, PIC32)
- **MSP430**
- **Renesas RX**

## FreeRTOS provides essential RTOS functionality in a small footprint, ideal for resource-constrained embedded systems with deterministic real-time requirements.
