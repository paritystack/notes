# RTOS Tracing & Runtime Analysis

## Overview

Debugging an RTOS is harder than debugging bare-metal code: bugs are *timing* bugs —
priority inversions, missed deadlines, a task that starves another, a stack that overflows
only under load. A breakpoint freezes the very timing you're trying to observe. The answer
is **tracing**: instrument the kernel to record every scheduling event (task switch, ISR
entry, queue send, semaphore give) with a timestamp, then visualize the timeline offline.
This page covers the tooling — Percepio Tracealyzer, SEGGER SystemView — plus runtime
stats, stack-overflow detection, and RTOS-aware debugger views, and how trace data gets
off the chip over [RTT](../embedded/rtt_semihosting.md).

It complements general [embedded debugging](../embedded/debugging.md) and
[GDB on embedded](../embedded/gdb_embedded.md), and is how you *see*
[scheduling](scheduling.md), [priority inversion](priority_inversion.md), and
[context switches](context_switching.md) actually happening.

## What a trace records

The kernel emits an event at every state change; the host reconstructs the timeline:

```
   ISR┤      ┌─┐                    ┌─┐
      │      │ │ gives sem          │ │
   T_H┤      └─┴──preempts──┌───────┘ └────
      │                     │ runs
   T_L┤──holds mutex──┐     └ blocked? inversion shows here as T_H waiting on T_L
      │               └ ...
      └──────────────────────────────────────────────► time (µs-resolution)

   Visible at a glance: who ran when, for how long, what unblocked whom,
   ISR vs task time, deadline misses, and inversion intervals.
```

This makes the otherwise-invisible behaviour from the [scheduling](scheduling.md) and
[priority inversion](priority_inversion.md) pages directly observable.

## Tooling

```
Percepio Tracealyzer   rich offline timeline + statistics; integrates with FreeRTOS,
                       Zephyr, ThreadX (via the kernel's trace hook macros). Snapshot
                       mode (dump a RAM buffer) or streaming mode (continuous, over RTT).
SEGGER SystemView      lightweight live recorder + viewer; streams events over RTT with
                       very low overhead; great for "what is the system doing right now".
Built-in kernel hooks  FreeRTOS traceTASK_SWITCHED_IN/OUT etc.; Zephyr's tracing
                       subsystem; ThreadX TraceX — you can also log to your own buffer.
```

All three hang off the RTOS's **trace hook macros**, so enabling them is mostly a config +
recorder-library step, not code changes throughout the app.

## Getting trace data off-chip

```
RTT (Real-Time Transfer)  J-Link reads/writes a RAM ring buffer while the CPU runs —
                          high bandwidth, microsecond-cheap, no UART pins. The usual
                          transport for SystemView/Tracealyzer streaming.
ITM/SWO                   Cortex-M trace pin; can stream timestamped events with low cost.
Snapshot dump            fill a RAM ring buffer, halt, dump via debugger — no live link
                          needed but you only see the last N events.
```

See [RTT & semihosting](../embedded/rtt_semihosting.md) for the transport mechanics.

## Runtime statistics

Beyond event traces, the kernel exposes cheap aggregate metrics you can poll or log:

```
CPU load per task    runtime counters (FreeRTOS vTaskGetRunTimeStats) → % CPU each task
                     uses. Spots a runaway or starved task.
Stack high-water     the deepest each task's stack has ever been used
                     (uxTaskGetStackHighWaterMark) → right-size stacks, catch near-misses.
Queue/heap usage     fill levels and free heap → spot leaks and undersized queues.
```

These feed back into the [scheduling](scheduling.md) budget (is utilization where you
predicted?) and [context-switch](context_switching.md) stack sizing.

## Stack-overflow detection

Each task has its own stack, and overflow is a top cause of mysterious
[HardFaults](../embedded/hardfault_debugging.md). Common detectors:

```
Painted stack    fill the stack with a pattern at creation; on each switch check the
                 last bytes (FreeRTOS configCHECK_FOR_STACK_OVERFLOW = 2). Cheap, slightly
                 late (detects after a small overrun).
SP bounds check  on switch-out, verify the task's saved SP is within its stack region
                 (config = 1). Fast, catches gross overruns.
MPU guard region  put an unmapped MPU region just past each stack → instant fault on
                 touch (precise, hardware-enforced). See the MPU page.
```

## RTOS-aware debugging

A plain debugger sees one stack and one set of registers. An **RTOS-aware** debugger
(GDB thread support, or IDE plugins) understands the kernel's task list:

```
- Lists all tasks with state (Running / Ready / Blocked) and priority.
- Lets you inspect the call stack of a BLOCKED task, not just the running one.
- Shows what each task is blocked on (which queue/semaphore).
```

Combine with [GDB on embedded](../embedded/gdb_embedded.md) for post-mortem inspection of a
halted system.

## Where this connects

- [Scheduling](scheduling.md) — traces let you verify utilization and deadlines against the
  analysis; runtime stats give measured CPU load
- [Priority inversion](priority_inversion.md) — inversion intervals are directly visible in
  a timeline trace
- [Context switching](context_switching.md) — switch events are the trace's backbone;
  stack high-water guides stack sizing
- [RTT & semihosting](../embedded/rtt_semihosting.md) — the low-overhead transport for live
  streaming traces
- [Embedded debugging](../embedded/debugging.md) / [GDB on embedded](../embedded/gdb_embedded.md)
  — the broader debug toolchain
- [HardFault debugging](../embedded/hardfault_debugging.md) — stack overflow is a leading
  fault cause that these detectors catch
- [MPU](../embedded/mpu.md) — hardware stack guard regions for precise overflow traps

## Pitfalls

```
- Heisenbugs from heavy instrumentation: a chatty trace adds overhead and shifts timing.
  Prefer low-overhead transports (RTT/ITM) and trace selectively.
- Snapshot buffer too small: you only capture the last N events and may miss the cause.
  Size the ring buffer for the window you need, or stream.
- Trusting stack sizes without high-water monitoring — overflow is silent until it
  corrupts a neighbour. Always check high-water in development; consider MPU guards in
  production.
- Painted-stack detection is slightly late and can miss a single large frame that jumps
  past the canary. Combine with SP-bounds or MPU guards for safety-critical stacks.
- Leaving full tracing enabled in production: overhead and buffer RAM cost. Gate it behind
  a build flag.
- Timestamp resolution too coarse to see short ISRs/critical sections — use a high-rate
  trace clock if you're chasing microsecond effects.
```
