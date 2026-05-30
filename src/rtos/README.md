# Real-Time Operating Systems (RTOS)

This directory contains guides for real-time operating systems used in embedded development.

## Contents

- **[FreeRTOS](freertos.md)** - Popular open-source RTOS
- **[ThreadX](threadx.md)** - Commercial RTOS (now open-source)
- **[Zephyr](zephyr.md)** - Scalable open-source RTOS (Linux Foundation)
- **[RT-Linux](rt_linux.md)** - Real-time Linux via PREEMPT_RT

## RTOS Concepts

**Real-Time**: Guarantees task execution within specified time constraints

**Deterministic**: Predictable behavior and timing

**Scheduling**: Priority-based task execution

**Inter-Task Communication**:
- Queues: Message passing
- Semaphores: Synchronization
- Mutexes: Mutual exclusion
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
