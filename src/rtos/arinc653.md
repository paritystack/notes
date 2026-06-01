# ARINC 653 — Partitioned Avionics RTOS

## Overview

ARINC 653 is the standard that lets multiple applications of **different safety
criticality** share one processor in an aircraft — the foundation of Integrated Modular
Avionics (IMA). Where [OSEK/AUTOSAR](osek_autosar.md) groups tasks on a shared automotive
ECU, ARINC 653 enforces hard **time and space partitioning**: each application runs in an
isolated partition with a guaranteed CPU window and protected memory, so a fault in one
partition cannot affect another. This "robust partitioning" is what allows a
[DO-178C](../embedded/functional_safety.md) Level-A flight-control function and a lower-
criticality display function to be certified on the same box. This page covers the
partition model, the major/minor frame schedule, the APEX API, and health monitoring.

It's the avionics sibling of the [scheduling](scheduling.md) and
[functional safety](../embedded/functional_safety.md) material, and contrasts sharply with
the [OSEK/AUTOSAR](osek_autosar.md) automotive model.

## Robust partitioning: space and time

```
Space partitioning   each partition gets its own memory, enforced by the MPU/MMU.
                     A partition cannot read or corrupt another's memory or the kernel's.
Time partitioning    each partition gets fixed CPU time WINDOWS on a repeating schedule.
                     A partition that overruns or hangs cannot steal another's slice.
```

The guarantee is **freedom from interference** in both space and time — the same goal as
AUTOSAR's memory/timing protection, but elevated to a strict, pre-planned partition
schedule rather than priority-based sharing.

## Two-level scheduling: partitions then processes

ARINC 653 schedules in two tiers:

```
Tier 1 — partition scheduler (deterministic, time-driven):
   a fixed, cyclic MAJOR FRAME of partition windows, repeated forever.

   │◄────────────────  major frame (e.g. 100 ms)  ────────────────►│
   ├────────┬──────┬────────────┬──────┬────────┬──────────────────┤
   │  P_A   │ P_B  │   P_A      │ P_C  │  P_A   │      idle         │
   └────────┴──────┴────────────┴──────┴────────┴──────────────────┘
     minor frames; each partition's windows sum to its guaranteed budget

Tier 2 — process scheduler (within a partition):
   priority-preemptive scheduling of the partition's own processes (≈ tasks),
   just like a conventional RTOS, but only during that partition's windows.
```

The major frame is **statically defined offline** by the system integrator. It is purely
time-triggered — no partition can run outside its window, which is what makes the timing
deterministic and certifiable.

## The APEX API

Partitions talk to the OS and to each other through the standardized APEX (APplication
EXecutive) services:

```
Partition mgmt   GET_PARTITION_STATUS, SET_PARTITION_MODE (init → normal)
Process mgmt     CREATE_PROCESS, START, STOP, SUSPEND, set priority
Time             TIMED_WAIT, PERIODIC_WAIT, GET_TIME (no busy-waiting)
Intra-partition  buffers, blackboards, semaphores, events (between processes)
Inter-partition  SAMPLING ports (latest value) and QUEUING ports (FIFO messages),
                 routed by config — partitions never share memory directly
Health monitor   RAISE_APPLICATION_ERROR, error handler hooks
```

Inter-partition communication is deliberately restricted to **ports** with statically
configured routing — there is no shared-memory backdoor between partitions.

## Health monitoring

A defined error-handling hierarchy isolates faults to the smallest scope possible:

```
Process level     handled by the partition's own error handler.
Partition level   e.g. restart or stop just that partition.
Module level      the whole core module — last resort.

Triggers: deadline overrun, memory violation (MPU trap), illegal APEX call,
          hardware fault. Configured responses keep a fault from propagating.
```

## ARINC 653 vs OSEK/AUTOSAR

```
                 ARINC 653 (avionics)            OSEK/AUTOSAR (automotive)
isolation        strict space + time partitions  OS-Applications + timing protection
scheduling       time-triggered major frame,      fixed-priority preemptive
                 then priority within partition
goal             mixed-criticality on one CPU      tiny, fast, certifiable ECU OS
standard         DO-178C / DO-297 (IMA)            ISO 26262
comms            sampling/queuing ports            COM over CAN/FlexRay
```

Both exist to make [functional safety](../embedded/functional_safety.md) arguments
tractable; they pick different mechanisms suited to their domain's certification regime.

## Where this connects

- [Functional safety](../embedded/functional_safety.md) — DO-178C design assurance levels
  are the reason for robust partitioning
- [OSEK/AUTOSAR](osek_autosar.md) — the automotive counterpart; different isolation model
- [Scheduling](scheduling.md) — the major frame is time-triggered scheduling; processes
  within a partition use fixed-priority preemption
- [MPU](../embedded/mpu.md) — hardware that enforces space partitioning
- [Synchronization](synchronization.md) — intra-partition buffers/blackboards/semaphores
  mirror the primitives covered there

## Pitfalls

```
- Designing as if partitions share memory: they don't. All cross-partition data goes
  through sampling/queuing ports with statically configured routing.
- Over-budgeting a partition window: total windows in a major frame are fixed; one
  partition's generosity starves another. Size from WCET, not guesswork.
- Blocking/busy-waiting that runs past a partition window — work is simply suspended until
  the next window, adding latency. Design processes to fit their slice.
- Treating sampling and queuing ports interchangeably: sampling keeps only the latest
  value (overwrite); queuing preserves a FIFO (and can overflow). Pick per data semantics.
- Ignoring health-monitor configuration: an unhandled error escalates to module level
  (whole box) instead of being contained to one partition.
- Confusing ARINC 653 partition scheduling with priority preemption — the major frame is
  fixed and time-driven; priority only acts *inside* a window.
```
