# RTOS vs Bare-Metal

## Overview

Not every embedded system needs an RTOS. The default starting point is **bare-metal** — a
`main()` super-loop plus [interrupts](../embedded/interrupts.md), no kernel. An RTOS like
[FreeRTOS](freertos.md), [ThreadX](threadx.md), or [Zephyr](zephyr.md) adds preemptive
[scheduling](scheduling.md), tasks, and [synchronization](synchronization.md) primitives —
but also RAM, latency, and complexity. This page lays out the architectures on the
spectrum (super-loop → cooperative scheduler → preemptive RTOS), what each costs, and a
checklist for deciding when the jump to an RTOS actually pays off.

## The super-loop (bare-metal)

The simplest firmware architecture: initialize, then loop forever doing each job in turn,
with interrupts handling time-critical events.

```c
int main(void) {
    init();
    for (;;) {
        read_sensors();
        update_control();
        service_comms();
        // ISRs fire asynchronously throughout
    }
}
```

```
+ Tiny, fully deterministic, no kernel RAM/flash, trivial to reason about.
+ No scheduling overhead, no context-switch cost, no priority-inversion class of bugs.
- Every job runs at "loop priority"; a slow job delays all the others (no preemption).
- Timing of any one job depends on the worst case of every other job in the loop.
- Hard to add independent activities at different rates without it becoming spaghetti.
```

Best when the work is a small set of fast, periodic jobs — many sensor nodes, simple
controllers, and [Arduino](../embedded/arduino.md)-style sketches live here happily.

## Structured bare-metal: state machines + a timing tick

You can scale the super-loop a long way before reaching for an RTOS by structuring jobs as
[state machines](../embedded/state_machines.md) driven by a timer tick, so no job blocks:

```
   timer tick (e.g. 1 ms) sets flags / advances state
   loop:  if (flag_10ms)  task_a_step();   // each step is short, non-blocking
          if (flag_100ms) task_b_step();
```

This gives cooperative multitasking without a kernel. The discipline — *never block,
always return quickly* — is the price.

## Cooperative scheduler

A thin scheduler runs registered tasks, each of which runs to completion or voluntarily
yields. No preemption, so shared data is race-free without locks, but one long task stalls
everyone. This is the model of protothreads, many "RTOS-lite" libraries, and Arduino
schedulers.

## Preemptive RTOS

Independent tasks with priorities; the kernel preempts lower-priority tasks the instant a
higher-priority one is ready. This is where you get true [response-time](scheduling.md)
guarantees for mixed-criticality work — but you take on context-switch cost, per-task
stacks, and the [priority-inversion](priority_inversion.md) class of bugs.

```
super-loop        ─►  cooperative        ─►  preemptive RTOS
no kernel             tiny scheduler          full kernel
no preemption         voluntary yield         priority preemption
race-free shared      race-free shared        needs mutexes/queues
data                  data                    (and their pitfalls)
simplest              moderate                most capable, most overhead
```

## When to move to an RTOS

```
Reach for an RTOS when…                          Stay bare-metal when…
─────────────────────────────────────────────   ─────────────────────────────────────
Several activities at different rates with        A handful of fast periodic jobs that
hard deadlines that interfere                     comfortably fit one loop
A long/blocking job must coexist with a           No job blocks; everything returns fast
latency-sensitive one                             
You need a TCP/IP, USB, BLE or filesystem         No heavy middleware, or it's polled
stack that assumes threads                        
Team wants modular, independently-scheduled        One developer, small codebase
components                                        
You have the RAM (kernel + per-task stacks)       Severely RAM-constrained (a few KB)
and can afford switch latency                     and every cycle counts
```

A useful rule of thumb: if you find yourself hand-building a scheduler with timers and
flags to juggle competing deadlines, you're reinventing an RTOS — adopt one. If a single
super-loop with a couple of ISRs meets timing with margin, an RTOS is pure overhead.

## Costs to budget for

```
RAM      kernel data + a separate stack per task (often the dominant cost)
Flash    kernel code (FreeRTOS ~6–12 KB; Zephyr more with subsystems)
Latency  context-switch time + tick handling added to every task's budget
Bugs     priority inversion, deadlock, stack overflow, ISR-safety — absent in a
         single-threaded super-loop
```

## Where this connects

- [Scheduling](scheduling.md) — what an RTOS buys you: provable deadlines for competing
  tasks
- [State machines](../embedded/state_machines.md) — the bare-metal alternative for
  structuring non-blocking jobs
- [Interrupts](../embedded/interrupts.md) — the time-critical layer in *both* models;
  bare-metal leans on them harder
- [Synchronization](synchronization.md) / [priority inversion](priority_inversion.md) —
  the new bug classes an RTOS introduces
- [Context switching](context_switching.md) — the overhead the preemptive model adds
- [FreeRTOS](freertos.md) / [ThreadX](threadx.md) / [Zephyr](zephyr.md) — the kernels you
  graduate to
- [Power management](../embedded/power_management.md) — both models idle the CPU; RTOSes
  add [tickless idle](tickless_idle.md)

## Pitfalls

```
- Adopting an RTOS reflexively: for a few periodic jobs it adds RAM, latency, and a whole
  bug class for no benefit. Match the architecture to the problem.
- Blocking calls (delay loops, busy-wait) inside a super-loop or cooperative task — they
  stall everything. Keep jobs non-blocking.
- Under-budgeting RAM: per-task stacks add up fast and stack overflow is silent. Size and
  monitor them (see context switching, RTOS tracing).
- Porting blocking driver code straight into a super-loop: it works on the bench, then a
  slow peripheral hangs the loop in the field.
- Assuming an RTOS makes timing deterministic for free — it only does if you assign
  priorities and verify schedulability.
```
