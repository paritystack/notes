# Zephyr

## Overview

Zephyr is a scalable, vendor-neutral RTOS from the Linux Foundation. Compare [FreeRTOS](freertos.md) (widely used on bare-metal MCUs, simpler kernel), [ThreadX](threadx.md) (safety-certified picokernel), and [RT-Linux](rt_linux.md) for real-time on full Linux. Zephyr reuses Linux concepts — [Devicetree](../linux/device_tree.md) for hardware description and Kconfig for configuration — applied to bare-metal microcontrollers.

Zephyr is a scalable, vendor-neutral real-time operating system hosted by the Linux Foundation. It targets resource-constrained microcontrollers as well as more capable application processors, offering a small, configurable kernel alongside a large ecosystem of drivers, networking stacks, and subsystems. Configuration is driven by Kconfig and hardware is described with Devicetree, giving a single source tree that builds for hundreds of boards.

## Core Concepts

**Threads**: Independent execution contexts with their own stack and priority
**Scheduler**: Priority-based, supporting both cooperative and preemptible threads
**Kernel Objects**: Semaphores, mutexes, message queues, FIFOs, mailboxes, pipes
**Workqueues**: Deferred work executed in a dedicated thread context
**Timers**: Kernel timers firing callbacks in system clock ISR context
**Devicetree**: Static description of hardware, resolved at build time
**Kconfig**: Symbolic configuration of kernel and subsystem features (`prj.conf`)
**west**: Meta-tool for fetching modules, building, flashing, and debugging

## Architecture & Build System

Zephyr applications are built with the `west` meta-tool, which wraps CMake, Kconfig, and Devicetree.

```bash
# Initialize a workspace and fetch sources/modules
west init ~/zephyrproject
cd ~/zephyrproject
west update

# Build an application for a specific board
west build -b nrf52840dk_nrf52840 samples/hello_world

# Flash and debug
west flash
west debug

# Override Kconfig/Devicetree at build time
west build -b qemu_cortex_m3 -- -DCONFIG_DEBUG=y -DEXTRA_DTC_OVERLAY_FILE=app.overlay
```

A minimal application has three files:

```
my_app/
├── CMakeLists.txt    # find_package(Zephyr); target_sources(app ...)
├── prj.conf          # Kconfig options for this app
└── src/main.c        # Application entry point
```

```c
/* src/main.c */
#include <zephyr/kernel.h>

int main(void)
{
    printk("Hello from Zephyr %s\n", KERNEL_VERSION_STRING);
    return 0;
}
```

## Threads

A thread is a function with its own stack and a priority. Threads can be defined statically at compile time or created at runtime.

### Static Thread Definition

```c
#include <zephyr/kernel.h>

#define STACK_SIZE 1024
#define PRIORITY   5

void blink_thread(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1);
    ARG_UNUSED(p2);
    ARG_UNUSED(p3);

    for (;;) {
        printk("blink\n");
        k_msleep(500);     /* sleep 500 ms, yielding the CPU */
    }
}

/* Defines stack, control block, and starts the thread automatically */
K_THREAD_DEFINE(blink_id, STACK_SIZE,
                blink_thread, NULL, NULL, NULL,
                PRIORITY, 0, 0);
```

### Runtime Thread Creation

```c
K_THREAD_STACK_DEFINE(worker_stack, STACK_SIZE);
static struct k_thread worker_data;

void start_worker(void)
{
    k_tid_t tid = k_thread_create(
        &worker_data, worker_stack,
        K_THREAD_STACK_SIZEOF(worker_stack),
        worker_entry,
        NULL, NULL, NULL,
        PRIORITY, 0, K_NO_WAIT);   /* K_NO_WAIT starts immediately */

    k_thread_name_set(tid, "worker");
}
```

### Thread Control

```c
k_thread_start(tid);       /* Start a thread created with K_FOREVER delay */
k_thread_suspend(tid);     /* Remove from scheduling */
k_thread_resume(tid);      /* Return to ready state */
k_thread_abort(tid);       /* Terminate a thread */
k_thread_priority_set(tid, 3);
```

## Scheduling

Zephyr uses a priority-based scheduler. **Lower numbers are higher priority.**

- **Cooperative threads** have negative priorities (`-CONFIG_NUM_COOP_PRIORITIES` .. `-1`). They run until they yield or block — they are never preempted by the scheduler.
- **Preemptible threads** have non-negative priorities (`0` .. `CONFIG_NUM_PREEMPT_PRIORITIES - 1`). They can be preempted by any higher-priority ready thread.

```c
k_yield();                 /* Yield to equal/higher priority ready threads */
k_sleep(K_MSEC(100));      /* Block this thread for 100 ms */
k_msleep(100);             /* Convenience wrapper for k_sleep(K_MSEC(100)) */
k_busy_wait(50);           /* Spin for 50 microseconds (no reschedule) */

/* Time slicing: round-robin among equal-priority preemptible threads */
k_sched_time_slice_set(10, 0);  /* 10 ms slices for priority >= 0 */
```

## Semaphores

Counting semaphores for signaling and resource counting.

```c
#include <zephyr/kernel.h>

K_SEM_DEFINE(data_sem, 0, 1);   /* initial 0, limit 1 (binary) */

void producer(void)
{
    /* ... produce data ... */
    k_sem_give(&data_sem);      /* signal */
}

void consumer(void)
{
    /* Wait up to 100 ms for the signal */
    if (k_sem_take(&data_sem, K_MSEC(100)) == 0) {
        /* semaphore obtained */
    } else {
        /* timed out */
    }
}
```

`k_sem_give()` is ISR-safe, making semaphores a common way to hand off work from an interrupt to a thread.

## Mutexes

Mutexes provide mutual exclusion with priority inheritance to avoid priority inversion. They are recursive — the owning thread may lock multiple times.

```c
K_MUTEX_DEFINE(my_mutex);

void update_shared(void)
{
    k_mutex_lock(&my_mutex, K_FOREVER);
    /* critical section */
    shared_value++;
    k_mutex_unlock(&my_mutex);
}
```

Unlike semaphores, mutexes may **not** be used from ISR context.

## Message Queues

Fixed-size message passing with internal copy semantics.

```c
struct sensor_msg {
    uint8_t id;
    float   value;
};

/* 10 messages, 4-byte aligned */
K_MSGQ_DEFINE(sensor_q, sizeof(struct sensor_msg), 10, 4);

void producer(void)
{
    struct sensor_msg m = { .id = 1, .value = 23.5f };

    /* Block if full, up to 50 ms */
    if (k_msgq_put(&sensor_q, &m, K_MSEC(50)) != 0) {
        k_msgq_purge(&sensor_q);   /* drop stale data on overflow */
    }
}

void consumer(void)
{
    struct sensor_msg m;

    if (k_msgq_get(&sensor_q, &m, K_FOREVER) == 0) {
        printk("sensor %u = %.2f\n", m.id, (double)m.value);
    }
}
```

## FIFOs and LIFOs

Linked-list queues that pass pointers (zero-copy). The first word of each item is reserved for the kernel's link pointer.

```c
struct data_item {
    void *fifo_reserved;   /* first word reserved for kernel use */
    int   payload;
};

K_FIFO_DEFINE(my_fifo);

void enqueue(struct data_item *item)
{
    k_fifo_put(&my_fifo, item);    /* append; ISR-safe */
}

void dequeue(void)
{
    struct data_item *item = k_fifo_get(&my_fifo, K_FOREVER);
    /* process item */
}
```

`k_lifo_put` / `k_lifo_get` provide the same API with last-in-first-out ordering.

## Mailboxes and Pipes

**Mailboxes** (`k_mbox`) support targeted, synchronous thread-to-thread messages with optional sender/receiver matching. **Pipes** (`k_pipe`) move arbitrary byte streams between threads with partial-transfer support.

```c
K_PIPE_DEFINE(my_pipe, 256, 4);   /* 256-byte ring buffer, 4-byte aligned */

void writer(void)
{
    uint8_t buf[32];
    size_t  written;

    k_pipe_put(&my_pipe, buf, sizeof(buf), &written,
               sizeof(buf) /* min */, K_FOREVER);
}

void reader(void)
{
    uint8_t buf[32];
    size_t  read;

    k_pipe_get(&my_pipe, buf, sizeof(buf), &read,
               1 /* min */, K_MSEC(100));
}
```

## Events

Event objects let a thread wait on a combination of bits set by other threads or ISRs.

```c
K_EVENT_DEFINE(my_event);

#define EVT_SENSOR_READY  BIT(0)
#define EVT_CONFIG_DONE   BIT(1)

void signaler(void)
{
    k_event_post(&my_event, EVT_SENSOR_READY);
}

void waiter(void)
{
    /* Wait until ALL of the requested bits are set */
    uint32_t bits = k_event_wait_all(&my_event,
                                     EVT_SENSOR_READY | EVT_CONFIG_DONE,
                                     false, K_FOREVER);
    if (bits) {
        /* all events occurred */
    }
}
```

## Polling

`k_poll` waits on several kernel objects (semaphores, FIFOs, signals) at once, similar to `select()`.

```c
struct k_poll_event events[2] = {
    K_POLL_EVENT_STATIC_INITIALIZER(K_POLL_TYPE_SEM_AVAILABLE,
        K_POLL_MODE_NOTIFY_ONLY, &data_sem, 0),
    K_POLL_EVENT_STATIC_INITIALIZER(K_POLL_TYPE_FIFO_DATA_AVAILABLE,
        K_POLL_MODE_NOTIFY_ONLY, &my_fifo, 0),
};

void multiplex(void)
{
    k_poll(events, ARRAY_SIZE(events), K_FOREVER);

    if (events[0].state == K_POLL_STATE_SEM_AVAILABLE) {
        k_sem_take(events[0].sem, K_NO_WAIT);
    }
    if (events[1].state == K_POLL_STATE_FIFO_DATA_AVAILABLE) {
        void *item = k_fifo_get(events[1].fifo, K_NO_WAIT);
        (void)item;
    }
}
```

## Workqueues

Workqueues run submitted work items in a dedicated thread, a clean way to defer processing out of ISR context. The kernel provides a system workqueue, and applications can define their own.

```c
struct k_work my_work;

void work_handler(struct k_work *work)
{
    /* Runs in workqueue thread context — full kernel API available */
    process_deferred_data();
}

void init(void)
{
    k_work_init(&my_work, work_handler);
}

/* Submit from an ISR to defer heavy processing to thread context */
void my_isr(void)
{
    k_work_submit(&my_work);   /* queued to the system workqueue */
}

/* Delayed (timed) work */
struct k_work_delayable poll_work;
k_work_init_delayable(&poll_work, work_handler);
k_work_schedule(&poll_work, K_MSEC(250));
```

## Timers

Kernel timers fire an expiry callback in system clock interrupt context.

```c
struct k_timer my_timer;

void timer_expiry(struct k_timer *timer)
{
    /* ISR context: keep short, defer heavy work to a workqueue */
    k_work_submit(&my_work);
}

void setup_timer(void)
{
    k_timer_init(&my_timer, timer_expiry, NULL);

    /* duration 1 s, then period 1 s (auto-reload) */
    k_timer_start(&my_timer, K_SECONDS(1), K_SECONDS(1));
}

/* One-shot: zero period */
k_timer_start(&my_timer, K_MSEC(500), K_NO_WAIT);

k_timer_stop(&my_timer);
uint32_t missed = k_timer_status_get(&my_timer);  /* expirations since last read */
```

## Memory Management

Zephyr favors static allocation but provides several dynamic schemes.

### Memory Slabs

Fixed-size block allocator — deterministic, no fragmentation.

```c
/* 16 blocks of 64 bytes, 4-byte aligned */
K_MEM_SLAB_DEFINE(my_slab, 64, 16, 4);

void *block;
if (k_mem_slab_alloc(&my_slab, &block, K_NO_WAIT) == 0) {
    /* use block ... */
    k_mem_slab_free(&my_slab, block);
}
```

### Heaps

Variable-size allocation from a bounded region.

```c
K_HEAP_DEFINE(my_heap, 2048);

void *p = k_heap_alloc(&my_heap, 128, K_MSEC(10));
if (p) {
    /* ... */
    k_heap_free(&my_heap, p);
}

/* System heap (backs k_malloc/k_free), sized by CONFIG_HEAP_MEM_POOL_SIZE */
char *buf = k_malloc(256);
k_free(buf);
```

## Interrupts

Interrupts are connected statically with `IRQ_CONNECT`; handlers should stay short and hand off work to threads.

```c
#define MY_IRQ        27
#define MY_IRQ_PRIO   2

void my_isr(const void *arg)
{
    ARG_UNUSED(arg);
    /* Acknowledge hardware, then defer processing */
    k_sem_give(&data_sem);     /* ISR-safe */
}

void irq_setup(void)
{
    IRQ_CONNECT(MY_IRQ, MY_IRQ_PRIO, my_isr, NULL, 0);
    irq_enable(MY_IRQ);
}

/* Short locks where a mutex/sem is overkill */
unsigned int key = irq_lock();
/* very short critical section */
irq_unlock(key);
```

## Devicetree & Drivers

Hardware is described in Devicetree; drivers are obtained via generated macros rather than hard-coded addresses.

```c
#include <zephyr/drivers/gpio.h>

/* Reference the "led0" alias from the board's Devicetree */
static const struct gpio_dt_spec led =
    GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

int blink_setup(void)
{
    if (!gpio_is_ready_dt(&led)) {
        return -ENODEV;
    }
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);

    for (;;) {
        gpio_pin_toggle_dt(&led);
        k_msleep(500);
    }
}
```

Board-specific pin assignments live in an overlay file (`app.overlay`) so the C code stays portable across boards.

## Configuration

Kernel and subsystem features are selected in `prj.conf`.

```ini
# prj.conf

# Kernel
CONFIG_MAIN_STACK_SIZE=2048
CONFIG_NUM_PREEMPT_PRIORITIES=15
CONFIG_NUM_COOP_PRIORITIES=16
CONFIG_TIMESLICING=y
CONFIG_TIMESLICE_SIZE=10

# Memory
CONFIG_HEAP_MEM_POOL_SIZE=4096

# Debug / observability
CONFIG_THREAD_NAME=y
CONFIG_THREAD_ANALYZER=y
CONFIG_THREAD_ANALYZER_AUTO=y
CONFIG_STACK_SENTINEL=y
CONFIG_ASSERT=y

# Logging
CONFIG_LOG=y
CONFIG_LOG_DEFAULT_LEVEL=3
```

## Best Practices

1. **Keep ISRs short**: signal a semaphore or submit a `k_work` item and return.
2. **Prefer static definitions** (`K_THREAD_DEFINE`, `K_MSGQ_DEFINE`) for predictable RAM usage.
3. **Size stacks from measurements**: enable `CONFIG_THREAD_ANALYZER` and watch unused-stack reports.
4. **Use mutexes, not semaphores, for mutual exclusion** to get priority inheritance.
5. **Reserve the first struct word** (`*fifo_reserved`) when placing items on a FIFO/LIFO.
6. **Use Devicetree specs** (`GPIO_DT_SPEC_GET`) instead of hard-coded peripheral addresses.

## Common Pitfalls

### Cooperative vs. Preemptible Priority Confusion

```c
/* Negative priority = cooperative: never preempted, must yield/block
 * explicitly or it will starve every other thread. */
#define COOP_PRIO  -1
#define PREEMPT_PRIO 5

/* A cooperative thread that never sleeps will hang the system */
void bad_coop(void *a, void *b, void *c)
{
    for (;;) { /* no k_sleep/k_yield — starves the scheduler */ }
}
```

### Calling Thread-Only APIs from an ISR

```c
void bad_isr(const void *arg)
{
    k_mutex_lock(&m, K_FOREVER);   /* WRONG: mutexes/sleeps are illegal in ISRs */
}

void good_isr(const void *arg)
{
    k_sem_give(&data_sem);         /* OK: give is ISR-safe */
}
```

### Blocking with a Timeout in ISR Context

Any kernel call from an ISR must use `K_NO_WAIT`; passing `K_FOREVER` or a timeout from interrupt context is a fatal error.

## Real-World Example

A small system: an interrupt-driven sensor signals a worker thread through a message queue; the worker updates shared state under a mutex.

```c
#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

struct reading {
    uint8_t id;
    float   temperature;
};

K_MSGQ_DEFINE(reading_q, sizeof(struct reading), 8, 4);
K_MUTEX_DEFINE(state_lock);
K_SEM_DEFINE(sample_ready, 0, 1);

static float g_last_temp;

/* ISR: triggered by the sensor's data-ready line */
static void sensor_isr(const void *arg)
{
    ARG_UNUSED(arg);
    k_sem_give(&sample_ready);     /* defer the read to a thread */
}

/* Sampling thread: reads hardware, posts to the queue */
static void sampler(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    for (;;) {
        k_sem_take(&sample_ready, K_FOREVER);

        struct reading r = {
            .id          = 1,
            .temperature = read_temperature_sensor(),
        };

        if (k_msgq_put(&reading_q, &r, K_NO_WAIT) != 0) {
            k_msgq_purge(&reading_q);   /* shed load on overflow */
        }
    }
}

/* Worker thread: consumes readings, updates shared state */
static void worker(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    struct reading r;

    for (;;) {
        if (k_msgq_get(&reading_q, &r, K_FOREVER) == 0) {
            k_mutex_lock(&state_lock, K_FOREVER);
            g_last_temp = r.temperature;
            k_mutex_unlock(&state_lock);

            if (r.temperature > 30.0f) {
                activate_cooling();
            }
        }
    }
}

K_THREAD_DEFINE(sampler_id, 1024, sampler, NULL, NULL, NULL, 4, 0, 0);
K_THREAD_DEFINE(worker_id,  1024, worker,  NULL, NULL, NULL, 5, 0, 0);

int main(void)
{
    IRQ_CONNECT(SENSOR_IRQ, 2, sensor_isr, NULL, 0);
    irq_enable(SENSOR_IRQ);

    printk("system started\n");
    return 0;
}
```

## Platform Support

Zephyr runs on a wide range of architectures:

- **ARM Cortex-M** (M0/M0+/M3/M4/M7/M23/M33)
- **ARM Cortex-A / Cortex-R** (32- and 64-bit)
- **RISC-V** (RV32, RV64)
- **Xtensa** (ESP32 family)
- **ARC** (Synopsys DesignWare)
- **x86** (IA-32, x86-64)
- **SPARC**, **MIPS**, **POSIX/native_sim** (host-based testing)

## Zephyr combines a small, configurable real-time kernel with Kconfig/Devicetree-driven portability and a broad subsystem ecosystem, scaling from tiny MCUs to multicore application processors.

## Where this connects

- [FreeRTOS](freertos.md) — simpler alternative for bare-metal MCUs; larger community, fewer built-in subsystems
- [ThreadX](threadx.md) — picokernel RTOS with IEC/ISO safety certifications; similar thread/queue/mutex primitives
- [RT-Linux](rt_linux.md) — PREEMPT_RT when you need real-time guarantees on a full Linux system rather than a bare-metal RTOS
- [Linux device tree](../linux/device_tree.md) — Zephyr reuses the Devicetree format from Linux to describe hardware at build time
- [Interrupts](../embedded/interrupts.md) — Zephyr ISR API and interrupt routing are central to driver and workqueue design
