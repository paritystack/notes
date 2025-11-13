# FreeRTOS

FreeRTOS is a real-time operating system kernel for embedded devices. It's designed to be small, simple, and easy to use.

## Core Concepts

**Tasks**: Independent threads of execution
**Queues**: Inter-task communication
**Semaphores**: Synchronization
**Mutexes**: Mutual exclusion
**Timers**: Software timers
**Event Groups**: Synchronization

## Task Creation

```c
#include "FreeRTOS.h"
#include "task.h"

void vTaskFunction(void *pvParameters) {
    for(;;) {
        // Task code
        vTaskDelay(pdMS_TO_TICKS(1000));  // Delay 1 second
    }
}

void main(void) {
    xTaskCreate(
        vTaskFunction,      // Function
        "TaskName",         // Name
        128,                // Stack size
        NULL,               // Parameters
        1,                  // Priority
        NULL                // Task handle
    );

    vTaskStartScheduler();  // Start scheduler

    for(;;);  // Should never reach here
}
```

## Queues

```c
#include "queue.h"

QueueHandle_t xQueue;

void vSenderTask(void *pvParameters) {
    int value = 42;
    xQueue = xQueueCreate(10, sizeof(int));

    for(;;) {
        xQueueSend(xQueue, &value, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void vReceiverTask(void *pvParameters) {
    int received;

    for(;;) {
        if(xQueueReceive(xQueue, &received, portMAX_DELAY)) {
            printf("Received: %d\n", received);
        }
    }
}
```

## Semaphores

```c
#include "semphr.h"

SemaphoreHandle_t xSemaphore;

void vTask1(void *pvParameters) {
    for(;;) {
        if(xSemaphoreTake(xSemaphore, portMAX_DELAY)) {
            // Critical section
            xSemaphoreGive(xSemaphore);
        }
    }
}
```

## Priority Levels

- Higher number = higher priority
- Idle task = priority 0
- Typical range: 0-31
- Preemptive scheduling (default)

FreeRTOS provides essential RTOS functionality in a small footprint, ideal for resource-constrained embedded systems.
