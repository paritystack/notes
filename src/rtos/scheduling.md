# Real-Time Scheduling

## Overview

Scheduling is the part of an RTOS that decides *which task runs next* and *for how
long*. In a general-purpose OS the scheduler optimizes for throughput and fairness; in a
real-time system it optimizes for **meeting deadlines**. That difference is the whole
reason RTOSes like [FreeRTOS](freertos.md), [ThreadX](threadx.md), and
[Zephyr](zephyr.md) exist, and why [PREEMPT_RT Linux](rt_linux.md) reworks the stock
Linux scheduler. This page covers the policies (priority-preemptive, round-robin, RMS,
EDF) and the timing analysis (utilization bounds, response-time analysis, WCET) that let
you *prove* a task set will always hit its deadlines rather than hoping it does.

It builds on the general scheduling material in
[Operating Systems](../misc/operating_systems.md) and connects tightly to
[priority inversion](priority_inversion.md), [synchronization](synchronization.md), and
the [context switch](context_switching.md) mechanism that actually carries out the
scheduler's decisions.

## Hard, soft, and firm real-time

"Real-time" means *timely*, not *fast*. What matters is the consequence of missing a
deadline.

```
Hard real-time : a missed deadline is a system failure.
                 (airbag squib firing, motor commutation, flight control)
Firm real-time : a late result is useless but not catastrophic — drop it.
                 (a video frame that arrives after its display slot)
Soft real-time : late results lose value gradually.
                 (UI responsiveness, telemetry logging)
```

A correct hard real-time system is one where you can *guarantee* every deadline under
worst-case conditions — which is what the analysis below is for.

## Task model

Most real-time theory models each task as a recurring job:

```
  T  period      — how often the task is released (e.g. every 10 ms)
  C  WCET        — worst-case execution time of one job
  D  deadline    — time after release by which the job must finish (often D = T)
  U  utilization — C / T  (fraction of the CPU this task needs)

  release        release        release
    │              │              │
    ▼              ▼              ▼
    ├──C──┐        ├──C──┐        ├──C──┐
    │     │        │     │        │     │
    0     │◄── D ──►        T              2T
```

Total CPU utilization is the sum of `C/T` over all tasks. You can never schedule a set
whose total utilization exceeds 1.0 (100% of one core) — but as the bounds below show,
you usually need headroom well below 1.0.

## Scheduling policies

### Priority-preemptive (the RTOS default)

Every task has a fixed priority; the scheduler always runs the highest-priority *ready*
task, preempting a lower-priority one the instant a higher-priority task becomes ready
(e.g. from an ISR posting to a [queue](synchronization.md)). This is what FreeRTOS,
ThreadX, and Zephyr do by default and what `SCHED_FIFO` provides on
[PREEMPT_RT Linux](rt_linux.md).

```
prio
 high │      ┌────┐            ┌────┐
   B  │      │ B  │            │ B  │     B preempts A on release
      │ ┌────┘    └────┐  ┌────┘    └──
   A  │ │ A           A│  │A         A   A resumes where it left off
  low │─┘              └──┘
      └────────────────────────────────► time
```

### Round-robin / time-slicing

Tasks *of equal priority* share the CPU in time slices (one tick each, typically). This
prevents one busy equal-priority task from starving its peers. It does **not** override
priority — a higher-priority task still preempts. `SCHED_RR` is the POSIX equivalent.

### Cooperative

Tasks run until they voluntarily yield (`taskYIELD()`, a blocking call). Simple and
race-free for shared data, but one misbehaving task hangs the system. See
[RTOS vs bare-metal](rtos_vs_bare_metal.md) for where cooperative scheduling fits.

## Fixed-priority assignment: Rate-Monotonic (RMS)

How do you *assign* the priorities? Rate-Monotonic Scheduling gives the rule: **shorter
period → higher priority**. It is the optimal fixed-priority assignment — if any
fixed-priority ordering can schedule the set, the rate-monotonic ordering can too.

Liu & Layland (1973) proved a sufficient schedulability bound for `n` independent
periodic tasks with `D = T`:

```
        n
       ───
       \    Ci          1/n
        >   ──   ≤  n (2     − 1)
       /    Ti
       ───
       i=1

   n=1 → 1.00     n=3 → 0.780     n→∞ → ln 2 ≈ 0.693
   n=2 → 0.828    n=4 → 0.757
```

If total utilization is **below** this bound, the set is *guaranteed* schedulable under
RMS. Above it, the set *might* still be schedulable — you fall back to exact
response-time analysis. The practical takeaway: with fixed priorities, plan to leave
~30% of the CPU idle as a safety margin.

## Dynamic priority: Earliest-Deadline-First (EDF)

EDF assigns priority dynamically: the ready job with the **nearest absolute deadline**
runs. It is optimal on a single core and schedulable whenever total utilization `≤ 1.0`
— it reclaims the headroom RMS leaves on the table.

```
RMS  : fixed priority by period      → simple, predictable, ~69–82% usable
EDF  : dynamic priority by deadline  → up to 100% usable, but more overhead
                                        and worse overload behavior (domino effect)
```

Most MCU RTOSes are fixed-priority (RMS-friendly); EDF appears in Linux's
`SCHED_DEADLINE` (see [RT-Linux](rt_linux.md)) and some safety RTOSes. Under transient
overload, RMS degrades predictably (low-priority tasks miss first) while EDF can
*cascade* into everyone missing — a key reason hard real-time systems often prefer fixed
priorities.

## Response-time analysis

When utilization exceeds the RMS bound, test each task exactly. The worst-case response
time `R` of task `i` is its own WCET plus interference from all higher-priority tasks
that preempt it, solved by iteration:

```
            ───        ┌  Rᵢ  ┐
  Rᵢ = Cᵢ +  >    ────  │ ──── │ · Cⱼ      iterate until Rᵢ stops changing
            ───        │  Tⱼ  │
          j∈hp(i)      

  Schedulable ⇔ Rᵢ ≤ Dᵢ for every task i.
```

This accounts for the actual periods, so it's exact (necessary and sufficient) for
fixed-priority preemptive scheduling — but it needs a trustworthy `Cᵢ` (WCET) and it
must also add **blocking** from lower-priority tasks holding shared resources (see
[priority inversion](priority_inversion.md)).

## WCET, jitter, and latency

The analysis is only as good as its inputs. Three timing quantities matter:

```
WCET     longest time a job's code can take. Inflated by cache misses, branch
         mispredicts, DMA contention, flash wait-states, variable-length loops.
Jitter   variation in release or completion time. Bad for control loops and I/O.
Latency  delay from an event (interrupt) to the responding task running:
           ISR latency + ISR run time + scheduler + context switch.
```

WCET is notoriously hard to measure — average-case timing tells you almost nothing about
the worst case. Determinism features like a [TCM/cache lockdown](../embedded/cache_tcm.md)
and an [MPU](../embedded/mpu.md) exist largely to make WCET tighter and provable. Disable
interrupts as little as possible, since [interrupt](../embedded/interrupts.md) latency
directly inflates the response time of every task.

## Where this connects

- [Priority inversion](priority_inversion.md) — blocking term that response-time
  analysis must include; inheritance/ceiling protocols bound it
- [Synchronization](synchronization.md) — blocking on queues/semaphores/mutexes is what
  makes a task transition between ready and blocked states
- [Context switching](context_switching.md) — the mechanism that enacts a scheduling
  decision; its cost is part of every task's overhead
- [RTOS vs bare-metal](rtos_vs_bare_metal.md) — when a super-loop is enough and when you
  need a real scheduler
- [FreeRTOS](freertos.md) / [ThreadX](threadx.md) / [Zephyr](zephyr.md) — fixed-priority
  preemptive schedulers with optional time-slicing
- [RT-Linux](rt_linux.md) — `SCHED_FIFO`/`SCHED_RR` (fixed) and `SCHED_DEADLINE` (EDF)
- [Operating Systems](../misc/operating_systems.md) — general scheduling background
- [Functional safety](../embedded/functional_safety.md) — deadline guarantees are part
  of safety arguments (ISO 26262, DO-178C)

## Pitfalls

```
- Confusing "fast" with "real-time": a high average throughput can still miss deadlines
  if the worst case is bad. Design and measure the worst case, not the average.
- Running fixed-priority systems near 100% utilization — leave RMS headroom (~30%).
- Forgetting the blocking term: a low-priority task holding a mutex can delay a
  high-priority one. Add it to response-time analysis (priority inversion).
- Trusting measured WCET from typical inputs; cache/flash/DMA effects make the real
  worst case much larger. Account for them or lock memory down.
- Too many priority levels collapsed onto a few hardware levels (some RTOSes round) —
  check your port's actual distinct priorities.
- Long interrupt-disabled or critical sections silently raise every task's latency.
- Assuming EDF is strictly better: its overload behavior (domino effect) makes fixed
  priority safer for hard deadlines.
```
