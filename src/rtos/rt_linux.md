# RT-Linux (PREEMPT_RT)

## Overview

"Real-time Linux" today almost always means the **PREEMPT_RT** preemption model — a set of changes that make the mainline Linux kernel fully preemptible, so high-priority tasks meet tight, bounded latency deadlines. Long maintained as an out-of-tree patch set, PREEMPT_RT has been progressively merged upstream, with `CONFIG_PREEMPT_RT` now selectable in mainline. It delivers *soft/firm* real-time on a full-featured OS — not the hard guarantees of a bare-metal RTOS like [FreeRTOS](freertos.md), [Zephyr](zephyr.md), or [ThreadX](threadx.md). [Interrupts](../embedded/interrupts.md) become threaded IRQs under PREEMPT_RT; understanding [Linux kernel patterns](../linux/kernel_patterns.md) (locking, RCU) is necessary to write RT-safe drivers.

> **Historical note:** the original *RTLinux* (and relatives RTAI and Xenomai) used a **dual-kernel** approach: a small hard-real-time microkernel ran the RT tasks and scheduled stock Linux as its lowest-priority thread. PREEMPT_RT is fundamentally different — there is one kernel, and it is made preemptible from within. This page covers the PREEMPT_RT, single-kernel approach.

## Core Concepts

**PREEMPT_RT**: Kernel config making nearly all kernel code preemptible
**Threaded IRQs**: Interrupt handlers run in schedulable kernel threads
**Sleeping spinlocks**: Most spinlocks become priority-inheriting `rt_mutex`es
**Priority inheritance**: In-kernel locks boost holders to avoid priority inversion
**SCHED_FIFO / SCHED_RR / SCHED_DEADLINE**: POSIX real-time scheduling policies
**Latency**: Time between an event becoming due and the task actually running
**cyclictest**: Standard tool for measuring scheduling latency

## What PREEMPT_RT Changes

Standard Linux is optimized for throughput, leaving sections where the kernel cannot be preempted — the source of latency spikes. PREEMPT_RT shrinks these to near zero by:

- **Threaded interrupt handlers** — hard IRQ work is minimal; the bulk runs in a kernel thread that can itself be scheduled and prioritized (`irq/<n>-<name>`).
- **Sleeping spinlocks** — most `spinlock_t` become preemptible `rt_mutex`es with priority inheritance, so holding a lock no longer blocks the scheduler. (`raw_spinlock_t` remains truly non-preemptible for the few places that need it.)
- **Priority inheritance** for in-kernel mutexes, preventing unbounded priority inversion.
- **High-resolution timers** (`hrtimers`) for precise wakeups independent of the tick.
- **Preemptible RCU** and reworked softirq handling so background kernel work yields to RT tasks.

The preemption model is chosen at build time:

```
CONFIG_PREEMPT_NONE       # server: throughput, worst latency
CONFIG_PREEMPT_VOLUNTARY  # desktop: some preemption points
CONFIG_PREEMPT            # low-latency desktop: preemptible kernel
CONFIG_PREEMPT_RT         # real-time: fully preemptible kernel
```

## Building / Enabling

```bash
# On a distro RT kernel, confirm PREEMPT_RT is active:
uname -v
#   #1 SMP PREEMPT_RT  ...      <- look for "PREEMPT_RT"
uname -a

# Or check the running config:
zcat /proc/config.gz | grep PREEMPT_RT     # if CONFIG_IKCONFIG_PROC=y
cat /sys/kernel/realtime 2>/dev/null        # prints 1 on an RT kernel
```

Many distributions ship ready-made RT kernels (e.g. `linux-image-rt-*` on Debian/Ubuntu, the `kernel-rt` package on RHEL/Fedora). To build your own, enable `CONFIG_PREEMPT_RT=y` (mainline) or apply the matching `patch-*-rtN` series for older trees, then configure and build as usual.

## Real-Time Scheduling Policies

A task only gets real-time behavior if it runs under an RT scheduling policy. Linux exposes the POSIX policies plus `SCHED_DEADLINE`.

- **SCHED_FIFO** — fixed priority (1–99), runs until it blocks or is preempted by a higher-priority RT task. No time slicing among equal priorities.
- **SCHED_RR** — like FIFO but round-robins equal-priority tasks with a time quantum.
- **SCHED_DEADLINE** — Earliest-Deadline-First with a Constant Bandwidth Server; each task declares `runtime`, `deadline`, and `period`. Outranks FIFO/RR.

```c
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <string.h>

/* Set the calling thread to SCHED_FIFO at a given priority */
int make_realtime(int priority)
{
    struct sched_param param;
    memset(&param, 0, sizeof(param));
    param.sched_priority = priority;   /* 1 (low) .. 99 (high) */

    /* pid 0 == calling thread */
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        return -1;   /* needs CAP_SYS_NICE / RLIMIT_RTPRIO */
    }
    return 0;
}

/* Same, applied to a pthread at creation time */
void spawn_rt_thread(void *(*fn)(void *))
{
    pthread_attr_t attr;
    struct sched_param param = { .sched_priority = 80 };

    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &param);

    pthread_t t;
    pthread_create(&t, &attr, fn, NULL);
    pthread_attr_destroy(&attr);
}
```

```bash
# Inspect or change a running task's policy/priority without code:
chrt -p <pid>                 # show policy and priority
chrt -f -p 80 <pid>           # set SCHED_FIFO priority 80
chrt -f 80 ./my_rt_app        # launch directly under SCHED_FIFO
```

## Determinism Techniques

Choosing an RT policy is not enough; the rest of the system must avoid introducing latency.

### Lock Memory and Pre-Fault the Stack

A page fault in the RT path can cost milliseconds. Lock all memory and touch the stack up front so nothing faults later.

```c
#include <sys/mman.h>
#include <string.h>

#define MAX_SAFE_STACK (512 * 1024)   /* bytes we promise never to exceed */

void prepare_realtime_memory(void)
{
    /* Keep all current and future pages resident in RAM */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall");
    }

    /* Pre-fault the stack: write to it so pages are mapped now */
    unsigned char dummy[MAX_SAFE_STACK];
    memset(dummy, 0, sizeof(dummy));
    /* 'dummy' deliberately unused afterward */
}
```

Also avoid `malloc`/`free`, file I/O, and `printf` inside the time-critical loop — allocate and open everything during startup.

### CPU Isolation

Dedicate cores to RT work and keep the kernel's housekeeping and other IRQs off them. Kernel command line:

```
isolcpus=2,3        # remove CPUs 2,3 from the general scheduler balance
nohz_full=2,3       # stop the periodic tick on those CPUs when one task runs
rcu_nocbs=2,3       # offload RCU callback processing away from RT CPUs
irqaffinity=0,1     # route device IRQs to the housekeeping CPUs
```

```bash
# Pin an RT process to an isolated CPU:
taskset -c 2 chrt -f 80 ./my_rt_app

# Steer a device interrupt away from the RT core (IRQ 45 -> CPU 0):
echo 1 > /proc/irq/45/smp_affinity
```

### Tame Power Management and Throttling

```bash
# Disable frequency scaling — run isolated cores at a fixed frequency
cpupower frequency-set -g performance

# RT throttling caps RT tasks at 95% by default to prevent lockups.
# A runaway SCHED_FIFO task can still starve the system; tune deliberately.
cat /proc/sys/kernel/sched_rt_runtime_us    # 950000 of 1000000 us
# Set to -1 to disable (only with a watchdog/escape hatch in place)
```

## Priority Inheritance Mutexes

User-space mutexes can also inherit priority, mirroring the kernel's protection against priority inversion.

```c
#include <pthread.h>

pthread_mutex_t lock;

void init_pi_mutex(void)
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);

    /* Boost the holder to the highest waiter's priority */
    pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);

    pthread_mutex_init(&lock, &attr);
    pthread_mutexattr_destroy(&attr);
}
```

## Measuring Latency

### cyclictest

The canonical tool (from the `rt-tests` package) measures wakeup latency of a periodic high-priority thread.

```bash
# 1 thread per CPU, SCHED_FIFO prio 80, 200 us interval, run forever,
# print min/act/avg/max; add load in another terminal (e.g. hackbench)
cyclictest --smp -p 80 -i 200 -m -q

# Typical summary line:
# T: 0 (1234) P:80 I:200 C: 500000 Min:  2 Act:  3 Avg:  4 Max:  18
#   Min/Avg/Max are microseconds. Max under load is the number that matters.
```

A well-tuned PREEMPT_RT box keeps the **maximum** latency in the low tens of microseconds even under heavy load; a non-RT kernel can spike into the milliseconds.

### Other Tools

```bash
hwlatdetect --duration=60     # detect hardware/firmware (SMI) induced stalls
trace-cmd record -p function_graph ./app   # ftrace-based latency tracing
perf sched record -- sleep 10; perf sched latency   # scheduler latency report
```

## Best Practices

1. **Lock memory** with `mlockall` and pre-fault stacks before the RT loop.
2. **Isolate CPUs** (`isolcpus`/`nohz_full`) and pin RT tasks there; keep IRQs off those cores.
3. **Never allocate or do I/O** in the time-critical path — do it all at init.
4. **Use priority-inheritance mutexes** in user space, just as the kernel does internally.
5. **Leave headroom below priority 99** — kernel threads (e.g. threaded IRQs, `migration`) run at high RT priorities; don't outrank what your task depends on.
6. **Always measure** with `cyclictest` *under representative load*; idle numbers are meaningless.

## Common Pitfalls

### Starving the System with SCHED_FIFO

```c
/* A SCHED_FIFO task that never blocks monopolizes its CPU.
 * Without RT throttling or CPU isolation it can lock up the machine. */
for (;;) {
    do_work();      /* no sleep, no blocking -> nothing else runs */
}
```

Always yield via a timed wait (see the example below), and keep `sched_rt_runtime_us` as a safety net during development.

### Page Faults and Hidden Allocations

`printf`, `malloc`, growing a `std::vector`, or first-touch of a memory-mapped file can each fault and blow your deadline. Keep the hot loop allocation-free.

### Priority Inversion Without Inheritance

A plain mutex shared with a lower-priority thread can block your RT task indefinitely while a mid-priority task runs. Use `PTHREAD_PRIO_INHERIT` (or `PTHREAD_PRIO_PROTECT`).

### Ignoring Hardware Latency

System Management Interrupts (SMIs) and aggressive C-states preempt *everything*, invisibly to the OS. Use `hwlatdetect` and disable deep idle states (`cpu_dma_latency`/`/dev/cpu_dma_latency`) for the worst offenders.

## Real-World Example

The canonical PREEMPT_RT cyclic task: a `SCHED_FIFO` thread that wakes on an absolute deadline with `clock_nanosleep`, with memory locked and stack pre-faulted.

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>

#define PERIOD_NS   (1000 * 1000)      /* 1 ms cycle */
#define NSEC_PER_SEC 1000000000L

static void timespec_add_ns(struct timespec *t, long ns)
{
    t->tv_nsec += ns;
    while (t->tv_nsec >= NSEC_PER_SEC) {
        t->tv_nsec -= NSEC_PER_SEC;
        t->tv_sec  += 1;
    }
}

int main(void)
{
    struct sched_param param = { .sched_priority = 80 };
    struct timespec next;

    /* 1. Become a real-time task */
    if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
        perror("sched_setscheduler");
        return EXIT_FAILURE;
    }

    /* 2. Lock memory so nothing pages out during the loop */
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        perror("mlockall");
        return EXIT_FAILURE;
    }

    /* 3. Pre-fault the stack */
    {
        unsigned char stack[512 * 1024];
        memset(stack, 0, sizeof(stack));
    }

    /* 4. Anchor the first deadline to the monotonic clock */
    clock_gettime(CLOCK_MONOTONIC, &next);

    for (;;) {
        timespec_add_ns(&next, PERIOD_NS);

        /* Sleep until the ABSOLUTE next deadline — no drift accumulation */
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);

        /* --- deterministic work goes here; keep it bounded --- */
        do_control_step();
    }

    return EXIT_SUCCESS;   /* never reached */
}
```

Compile and run on an RT kernel:

```bash
gcc -O2 -Wall rt_cyclic.c -o rt_cyclic -lrt
sudo taskset -c 2 ./rt_cyclic      # pin to isolated CPU 2
```

## PREEMPT_RT turns mainline Linux into a fully preemptible, low-latency real-time OS — pairing standard POSIX scheduling APIs with careful system tuning (memory locking, CPU isolation, priority inheritance) and `cyclictest`-driven measurement to achieve bounded, deterministic response.

## Where this connects

- [FreeRTOS](freertos.md) — bare-metal alternative for hard real-time on resource-constrained MCUs; no Linux userspace
- [Zephyr](zephyr.md) — another bare-metal RTOS option with Devicetree and networking stacks
- [ThreadX](threadx.md) — safety-certified picokernel for deeply embedded targets without OS overhead
- [Linux kernel patterns](../linux/kernel_patterns.md) — RT-safe locking (sleeping spinlocks, rt_mutex) requires understanding kernel concurrency patterns
- [Interrupts](../embedded/interrupts.md) — PREEMPT_RT converts interrupt handlers to threaded IRQs, making them schedulable and prioritizable
