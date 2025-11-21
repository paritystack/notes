# Watchdog Timers

A Watchdog Timer (WDT) is a hardware or software timer that is used to detect and recover from computer malfunctions. During normal operation, the system regularly resets the watchdog timer to prevent it from elapsing, or "timing out." If the system fails to reset the watchdog timer, it is assumed to be malfunctioning, and corrective actions are taken, such as resetting the system.

## Key Concepts

- **Timeout Period**: The duration for which the watchdog timer runs before it times out. If the timer is not reset within this period, it triggers a system reset or other corrective actions.
- **Reset Mechanism**: The action taken when the watchdog timer times out. This is typically a system reset, but it can also include other actions like logging an error or entering a safe state.
- **Feeding the Watchdog**: The process of regularly resetting the watchdog timer to prevent it from timing out. This is also known as "kicking" or "patting" the watchdog.

## Hardware Implementation

### Architecture

A typical hardware watchdog timer consists of:

1. **Counter/Timer**: A down-counter that decrements at a fixed rate
2. **Prescaler**: Divides the system clock to create longer timeout periods
3. **Control Registers**: Configure timeout period, enable/disable, and operating modes
4. **Reset Circuit**: Generates reset signal when counter reaches zero

### Common Hardware Configurations

**STM32 Independent Watchdog (IWDG)**
- Free-running down-counter clocked by independent LSI oscillator
- Prescaler range: 4 to 256
- 12-bit reload register (timeout from ~100μs to ~32s)
- Once started, cannot be stopped except by reset
- Key register: IWDG_KR (0xCCCC to start, 0xAAAA to reload)

**AVR Watchdog Timer**
- Built-in oscillator running at approximately 128 kHz
- Timeout periods: 15ms to 8s (configurable via WDTCSR register)
- Supports both interrupt and reset modes
- Can be disabled by fuse bits (WDTON fuse)

**ESP32 Watchdog**
- Two-level watchdog: Task WDT and Interrupt WDT
- Task WDT monitors FreeRTOS tasks
- Interrupt WDT monitors ISRs and critical sections
- Configurable through esp_task_wdt_* APIs

### Register Operations

Typical register operations include:

1. **Enable/Start**: Write specific key value to start counter
2. **Reload/Feed**: Write reload value to reset counter to maximum
3. **Configure Timeout**: Set prescaler and reload values
4. **Status Check**: Read flags to determine if reset was WDT-triggered

## Software Implementation Examples

### Bare-Metal Microcontroller (STM32)

```c
#include "stm32f4xx.h"

void IWDG_Init(uint32_t timeout_ms) {
    // Enable write access to IWDG_PR and IWDG_RLR registers
    IWDG->KR = 0x5555;

    // Set prescaler to 32 (LSI/32)
    IWDG->PR = IWDG_PR_PR_2;

    // Set reload value (timeout_ms * 32000 / 32 / 1000)
    IWDG->RLR = (timeout_ms * 1000) / 1000;

    // Reload counter
    IWDG->KR = 0xAAAA;

    // Enable IWDG
    IWDG->KR = 0xCCCC;
}

void IWDG_Feed(void) {
    IWDG->KR = 0xAAAA;  // Reload watchdog counter
}

int main(void) {
    IWDG_Init(1000);  // 1 second timeout

    while(1) {
        // Perform application tasks
        do_work();

        // Feed watchdog periodically
        IWDG_Feed();

        HAL_Delay(500);  // Delay less than timeout
    }
}
```

### ESP32 with FreeRTOS

```c
#include "esp_task_wdt.h"

#define WDT_TIMEOUT_S 5

void critical_task(void *pvParameters) {
    // Subscribe this task to the watchdog
    esp_task_wdt_add(NULL);

    while(1) {
        // Perform critical operations
        process_data();
        read_sensors();

        // Reset watchdog for this task
        esp_task_wdt_reset();

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void) {
    // Initialize Task Watchdog Timer
    esp_task_wdt_init(WDT_TIMEOUT_S, true);

    xTaskCreate(critical_task, "critical", 2048, NULL, 5, NULL);
}
```

### Linux Userspace

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/watchdog.h>

int main(void) {
    int fd;
    int timeout = 10;  // seconds

    // Open watchdog device
    fd = open("/dev/watchdog", O_WRONLY);
    if (fd < 0) {
        perror("Failed to open watchdog");
        exit(1);
    }

    // Set timeout
    ioctl(fd, WDIOC_SETTIMEOUT, &timeout);

    while(1) {
        // Perform application tasks
        do_application_work();

        // Feed the watchdog
        ioctl(fd, WDIOC_KEEPALIVE, 0);

        sleep(5);  // Sleep less than timeout
    }

    // Close watchdog (triggers reset on some systems)
    close(fd);

    return 0;
}
```

### FreeRTOS Task Monitor

```c
#include "FreeRTOS.h"
#include "task.h"

TaskHandle_t xTask1Handle = NULL;
TaskHandle_t xTask2Handle = NULL;

void vTask1(void *pvParameters) {
    while(1) {
        // Task 1 work
        perform_task1();

        // Notify watchdog task that we're alive
        xTaskNotifyGive(xWatchdogTaskHandle);

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void vWatchdogTask(void *pvParameters) {
    uint32_t ulNotificationValue;
    const TickType_t xMaxExpectedBlockTime = pdMS_TO_TICKS(500);

    while(1) {
        // Wait for notifications from monitored tasks
        ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxExpectedBlockTime);

        if(ulNotificationValue > 0) {
            // Tasks are running, feed hardware watchdog
            HAL_IWDG_Refresh(&hiwdg);
        } else {
            // Timeout - tasks are stuck, allow WDT reset
            // Do not feed the watchdog
        }
    }
}
```

## Advanced Concepts

### Window Watchdog Timer

A window watchdog enforces both minimum and maximum refresh intervals, preventing both hung systems and runaway code that refreshes too quickly.

**Characteristics:**
- Has both upper and lower time bounds
- Generates reset if refreshed too early or too late
- Useful for detecting timing violations in periodic tasks

**STM32 WWDG Example:**
```c
void WWDG_Init(void) {
    __HAL_RCC_WWDG_CLK_ENABLE();

    hwwdg.Instance = WWDG;
    hwwdg.Init.Prescaler = WWDG_PRESCALER_8;
    hwwdg.Init.Window = 127;    // Upper bound
    hwwdg.Init.Counter = 127;   // Start value
    hwwdg.Init.EWIMode = WWDG_EWI_DISABLE;

    HAL_WWDG_Init(&hwwdg);
}

void WWDG_Refresh(void) {
    // Refresh only when counter falls below window
    HAL_WWDG_Refresh(&hwwdg);
}
```

### Cascaded Watchdog Systems

Multiple watchdog layers provide defense in depth:

1. **Hardware WDT**: Final safety net, cannot be disabled by software
2. **Software WDT**: Monitors task health, can trigger controlled recovery
3. **Application Monitor**: Checks data validity and logical correctness

```
Application Tasks → Software Monitor → Hardware WDT → Reset
                         ↓
                   Graceful Recovery
```

### Interrupt-Driven Watchdog

Some watchdogs can generate interrupts before reset, allowing cleanup:

```c
void WWDG_IRQHandler(void) {
    if(__HAL_WWDG_GET_FLAG(&hwwdg, WWDG_FLAG_EWIF)) {
        // Early warning interrupt
        // Last chance to save critical data
        save_critical_state();
        log_error("Watchdog about to reset");

        __HAL_WWDG_CLEAR_FLAG(&hwwdg, WWDG_FLAG_EWIF);

        // System will reset shortly if not refreshed
    }
}
```

### Watchdog in Low-Power Modes

Watchdog behavior during sleep/standby modes:

- **Independent WDT (STM32 IWDG)**: Continues running in all power modes
- **Window WDT (STM32 WWDG)**: Stops in Stop/Standby modes
- **ESP32**: Task WDT can be suspended during light sleep

**Considerations:**
- Choose appropriate watchdog type for power profile
- Account for sleep duration in timeout calculations
- May need to disable/reconfigure WDT before entering deep sleep

## Troubleshooting & Best Practices

### Common Pitfalls

1. **Timeout Too Short**
   - Symptom: System resets during legitimate long operations
   - Solution: Profile worst-case execution time, add margin

2. **Timeout Too Long**
   - Symptom: System hangs go undetected too long
   - Solution: Break long operations into chunks with WDT refreshes

3. **Feeding Too Frequently**
   - Symptom: Stuck in tight loop but still feeding watchdog
   - Solution: Use window watchdog or structured feeding points

4. **Feeding in Interrupt**
   - Symptom: Interrupt runs, main code hung, but WDT still fed
   - Solution: Feed only from main application flow, not ISRs

5. **Forgetting Clock Source**
   - Symptom: Timeout period incorrect or unstable
   - Solution: Account for RC oscillator accuracy (±10-40%)

### Debugging Strategies

**Detect Watchdog Resets:**
```c
void check_reset_source(void) {
    if(__HAL_RCC_GET_FLAG(RCC_FLAG_IWDGRST)) {
        // System was reset by Independent Watchdog
        log_error("WDT Reset detected");
        __HAL_RCC_CLEAR_RESET_FLAGS();
    }
}
```

**Add Instrumentation:**
```c
volatile uint32_t wdt_feed_counter = 0;
uint32_t last_feed_timestamp = 0;

void IWDG_Feed_Debug(void) {
    wdt_feed_counter++;
    last_feed_timestamp = HAL_GetTick();
    IWDG_Feed();
}
```

**Simulate Failures:**
```c
#ifdef DEBUG_WDT
void test_watchdog_reset(void) {
    // Deliberately stop feeding to test WDT
    while(1) {
        // Wait for reset
    }
}
#endif
```

### Design Patterns

**Centralized Feeding:**
```c
void system_health_monitor(void) {
    if(check_all_tasks_healthy()) {
        IWDG_Feed();
    }
    // If unhealthy, don't feed - allow reset
}
```

**Distributed Feeding with Health Tokens:**
```c
typedef struct {
    bool task1_alive;
    bool task2_alive;
    bool task3_alive;
} health_status_t;

health_status_t system_health = {0};

void watchdog_manager_task(void) {
    while(1) {
        if(system_health.task1_alive &&
           system_health.task2_alive &&
           system_health.task3_alive) {
            IWDG_Feed();
        }

        // Reset health flags
        system_health.task1_alive = false;
        system_health.task2_alive = false;
        system_health.task3_alive = false;

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
```

**Graceful Degradation:**
```c
typedef enum {
    MODE_NORMAL,
    MODE_SAFE,
    MODE_EMERGENCY
} system_mode_t;

system_mode_t current_mode = MODE_NORMAL;

void main_loop(void) {
    switch(current_mode) {
        case MODE_NORMAL:
            execute_all_features();
            break;
        case MODE_SAFE:
            execute_critical_features_only();
            break;
        case MODE_EMERGENCY:
            minimal_operation();
            break;
    }

    if(system_healthy()) {
        IWDG_Feed();
    }
}
```

### Testing Strategies

1. **Unit Test Timeout Calculations**
   - Verify timeout math accounts for clock accuracy
   - Test prescaler and reload value calculations

2. **Integration Test Reset Recovery**
   - Deliberately trigger WDT reset
   - Verify system recovers to known safe state

3. **Stress Test Under Load**
   - Run worst-case scenarios
   - Measure maximum time between feeds

4. **Power Cycle Testing**
   - Verify WDT behavior across power transitions
   - Test brownout conditions

5. **Long-Term Reliability Testing**
   - Run for extended periods (days/weeks)
   - Monitor for unexpected resets

## Example Usage

1. **Embedded Systems**: Watchdog timers are commonly used in embedded systems to ensure that the system can recover from unexpected failures. For example, if a microcontroller stops responding, the watchdog timer can reset it to restore normal operation.
2. **Safety-Critical Applications**: In applications where safety is paramount, such as automotive or medical devices, watchdog timers help ensure that the system can recover from faults and continue to operate safely.
3. **Industrial Control Systems**: PLCs and SCADA systems use watchdogs to maintain continuous operation and automatically recover from transient faults.
4. **IoT Devices**: Remote devices use watchdogs to recover from network issues or software hangs without manual intervention.

## Conclusion

Watchdog timers are essential components in many systems, providing a mechanism to detect and recover from malfunctions. Proper implementation requires understanding hardware architecture, choosing appropriate timeout periods, implementing robust feeding strategies, and thorough testing. A well-designed watchdog system provides reliable fault detection while avoiding false resets, balancing safety with system stability. Whether implementing a simple independent watchdog or a sophisticated multi-level monitoring system, the key is to match the watchdog strategy to the specific reliability requirements and failure modes of your application.
