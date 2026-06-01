# Context Switching

## Overview

A context switch is the act of **saving one task's CPU state and restoring another's** —
the mechanism that actually carries out a [scheduler](scheduling.md) decision. Everything
else in an RTOS (priorities, [synchronization](synchronization.md) primitives,
[priority inheritance](priority_inversion.md)) ultimately resolves to "switch from task A
to task B," and the *cost* of that switch is part of every task's timing budget. This page
covers what a context is, how the tick and yields trigger switches, and the
[ARM Cortex-M](../embedded/cmsis.md) specifics — PendSV, SVC, the dual stack pointers, and
the exception stack frame — that [FreeRTOS](freertos.md), [ThreadX](threadx.md), and
[Zephyr](zephyr.md) ports rely on.

## What is a "context"?

A task's context is everything it needs to resume exactly where it stopped:

```
- General-purpose registers (r0–r12 on Cortex-M)
- Stack pointer (each task has its OWN stack)
- Program counter + status register (xPSR)
- Link register (return address)
- FPU registers (s0–s31, FPSCR) IF the task uses floating point
```

Each task owns a separate stack; the switch is really just *swapping stack pointers* plus
saving/restoring registers. Sizing those stacks is a recurring source of bugs — see
[Pitfalls](#pitfalls) and [RTOS tracing](rtos_tracing.md) for stack-overflow detection.

## What triggers a switch

```
Preemption   a higher-priority task becomes ready (ISR gives a semaphore, a timeout
             expires) → switch immediately.
Blocking     the running task waits on a queue/semaphore/delay → switch to next ready.
Yield        cooperative taskYIELD(), or end of a round-robin time slice.
Tick         the periodic timer interrupt wakes delayed tasks and may time-slice.
```

The **tick** is the RTOS heartbeat — a hardware timer (SysTick on Cortex-M) firing at a
fixed rate (often 1 kHz). Each tick the kernel checks for expired delays and, if a higher
priority task is now ready, requests a switch. (Tickless variants suppress this when
idle — see [tickless idle](tickless_idle.md).)

## Cortex-M mechanics: PendSV, SVC, and the two stacks

On ARM Cortex-M the context switch is built from architectural features designed for
exactly this:

```
MSP  Main Stack Pointer      → used by the kernel and all ISRs (handler mode)
PSP  Process Stack Pointer   → used by tasks (thread mode)         ← swapped per task

SVC    supervisor call       → used to start the FIRST task
PendSV pendable service IRQ  → the actual context switch, run at LOWEST priority
SysTick timer                → the tick; may pend PendSV if a switch is needed
```

The trick is **PendSV runs at the lowest exception priority**. The kernel *pends* PendSV;
it executes only after all real ISRs finish, so a context switch never preempts an
interrupt handler. This keeps [interrupt](../embedded/interrupts.md) latency low and the
switch logic in one place.

```
   IRQ fires ──► ISR runs (high prio) ──► gives semaphore, pends PendSV
                                              │
   ...ISR returns, no other IRQ pending...    ▼
                                          PendSV (lowest prio) runs:
                                          save task A's r4–r11 + PSP,
                                          pick task B, restore B's r4–r11 + PSP,
                                          return into task B
```

### The exception stack frame

When any exception is taken, hardware **automatically stacks** eight registers
(r0–r3, r12, LR, PC, xPSR) onto the current stack. So PendSV only has to manually save
the *callee-saved* registers (r4–r11) — the hardware already saved the rest:

```
   high addr ┌──────────┐
             │  xPSR    │ ◄── auto-stacked by hardware on exception entry
             │  PC      │
             │  LR      │
             │  r12     │
             │  r3..r0  │
             ├──────────┤
             │ r11..r4  │ ◄── manually saved by PendSV (the RTOS port asm)
   low addr  └──────────┘ ◄── saved PSP for this task
```

**Lazy FP stacking:** if the FPU is enabled, Cortex-M reserves space for s0–s15 but only
actually stacks them when a handler first touches the FPU — saving cycles on tasks that
don't use floating point. The RTOS port must save s16–s31 itself for FP tasks.

## The cost

A switch is not free — it's overhead charged to real-time budgets:

```
- Register save/restore (~dozens of cycles; more with FPU context)
- Cache/pipeline effects: the new task's code/data may be cold (see cache & TCM)
- Kernel bookkeeping: choosing the next task, list updates
Typical Cortex-M switch: low hundreds of cycles. Count it in your WCET.
```

Locking hot code/data into [TCM or cache](../embedded/cache_tcm.md) reduces the cold-start
penalty and tightens [WCET](scheduling.md).

## Where this connects

- [Scheduling](scheduling.md) — the switch enacts the scheduler's choice; its cost is per
  task overhead in response-time analysis
- [Interrupts](../embedded/interrupts.md) — PendSV/SysTick are exceptions; switch design
  keeps ISR latency low (deferred via lowest-priority PendSV)
- [Synchronization](synchronization.md) — an ISR `give` that unblocks a higher task pends
  a switch (`portYIELD_FROM_ISR`)
- [CMSIS](../embedded/cmsis.md) — defines SysTick, NVIC, and the core registers the port
  manipulates
- [HardFault debugging](../embedded/hardfault_debugging.md) — corrupt PSP/stack frames are
  a top cause of faults; the stacked frame is what you decode in a fault
- [Cache & TCM](../embedded/cache_tcm.md) — cold caches inflate post-switch timing
- [Tickless idle](tickless_idle.md) — suppressing the periodic tick that drives switches

## Pitfalls

```
- Stack overflow: each task stack must fit its deepest call chain PLUS the exception
  frame(s) pushed on top. Size with high-water-mark checks; enable stack-overflow
  detection in development.
- Forgetting FPU context: a task that uses float without the port saving s16–s31 will
  silently corrupt another task's FP state. Enable FP context in the port config.
- Doing real work in PendSV/SysTick beyond the switch: it runs every tick and inflates
  latency. Keep it minimal.
- Calling blocking/kernel APIs from a context where switching is illegal (inside a
  critical section, from a high-priority ISR above the kernel's max syscall priority).
- Mis-set NVIC priorities: PendSV/SysTick must be lowest, or a switch can preempt an ISR
  and corrupt state. This is a classic port-config bug.
- Ignoring switch cost in tight control loops — at high tick rates the overhead becomes a
  measurable fraction of the CPU.
```
