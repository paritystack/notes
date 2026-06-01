# Real-Time Operating Systems (RTOS)

This directory contains guides for real-time operating systems used in embedded development.

## Contents

**Kernels**

- **[FreeRTOS](freertos.md)** - Popular open-source RTOS
- **[ThreadX](threadx.md)** - Commercial RTOS (now open-source)
- **[Zephyr](zephyr.md)** - Scalable open-source RTOS (Linux Foundation)
- **[RT-Linux](rt_linux.md)** - Real-time Linux via PREEMPT_RT

**Concepts**

- **[Real-Time Scheduling](scheduling.md)** - Priority-preemptive, round-robin, RMS, EDF, schedulability analysis
- **[Priority Inversion](priority_inversion.md)** - Inheritance and ceiling protocols (the Mars Pathfinder bug)
- **[Synchronization](synchronization.md)** - Queues, semaphores, mutexes, event flags
- **[Context Switching](context_switching.md)** - Save/restore internals, PendSV on Cortex-M
- **[RTOS vs Bare-Metal](rtos_vs_bare_metal.md)** - Super-loop vs cooperative vs preemptive

**Safety-critical standards**

- **[OSEK/AUTOSAR OS](osek_autosar.md)** - Statically-configured automotive RTOS (ISO 26262)
- **[ARINC 653](arinc653.md)** - Time/space partitioned avionics RTOS (DO-178C / IMA)

**Operational**

- **[Tickless Idle](tickless_idle.md)** - Suppressing the tick for low-power sleep
- **[RTOS Tracing](rtos_tracing.md)** - Tracealyzer/SystemView, runtime stats, stack-overflow detection

## RTOS Concepts

**Real-Time**: Guarantees task execution within specified time constraints — see
[Real-Time Scheduling](scheduling.md) for hard/soft/firm timing and deadline analysis.

**Deterministic**: Predictable behavior and timing

**Scheduling**: Priority-based task execution — [Real-Time Scheduling](scheduling.md)

**Inter-Task Communication** (see [Synchronization](synchronization.md)):
- Queues: Message passing
- Semaphores: Synchronization
- Mutexes: Mutual exclusion (with [priority inheritance](priority_inversion.md))
- Event flags: Thread synchronization

## Comparison

| Feature | FreeRTOS | ThreadX | Zephyr | RT-Linux (PREEMPT_RT) |
|---------|----------|---------|--------|-----------------------|
| License | MIT | MIT (since 2019) | Apache 2.0 | GPLv2 |
| Footprint | Very small | Small | Small–medium | Full OS |
| Scheduling | Preemptive | Preemptive | Preemptive + cooperative | Preemptive (SCHED_FIFO/RR/DEADLINE) |
| Priority levels | Configurable | 32 levels | Configurable (coop + preempt) | 1–99 (RT) |
| Use cases | IoT, embedded | Industrial, IoT | MCU to app processors | Industrial control, robotics, audio |

RTOS systems provide deterministic task scheduling essential for time-critical embedded applications.
