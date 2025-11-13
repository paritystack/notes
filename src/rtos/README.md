# Real-Time Operating Systems (RTOS)

This directory contains guides for real-time operating systems used in embedded development.

## Contents

- **[FreeRTOS](freertos.md)** - Popular open-source RTOS
- **[ThreadX](threadx.md)** - Commercial RTOS (now open-source)

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

| Feature | FreeRTOS | ThreadX |
|---------|----------|---------|
| License | MIT | MIT (since 2019) |
| Footprint | Very small | Small |
| Scheduling | Preemptive | Preemptive |
| Priority levels | Configurable | 32 levels |
| Use cases | IoT, embedded | Industrial, IoT |

RTOS systems provide deterministic task scheduling essential for time-critical embedded applications.
