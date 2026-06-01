# OSEK/VDX & AUTOSAR Classic OS

## Overview

OSEK/VDX and its successor AUTOSAR Classic are the **automotive** RTOS standards — the
operating systems running in the engine, brake, and body-control ECUs of nearly every
modern car. Unlike general-purpose kernels such as [FreeRTOS](freertos.md) or
[Zephyr](zephyr.md), an OSEK OS is **statically configured**: every task, resource, and
alarm is declared at build time, nothing is created at runtime, and the whole system is
sized to fit a tiny, safety-certified footprint. This page covers the OSEK conformance
classes, its task/alarm/resource model, the mandatory
[priority ceiling protocol](priority_inversion.md), and how AUTOSAR Classic wraps it for
[ISO 26262](../embedded/functional_safety.md) functional safety.

It sits at the intersection of [scheduling](scheduling.md), the automotive buses
[CAN](../embedded/can.md) and [LIN/FlexRay](../embedded/lin_flexray.md), and the
[MISRA C](../embedded/coding_standards.md) coding discipline these ECUs are written under.

## Why static configuration

Automotive ECUs must be deterministic, certifiable, and run in a few KB of RAM. OSEK
forbids dynamic allocation entirely:

```
- All tasks, alarms, events, resources, counters declared in OIL (OSEK Implementation
  Language) and generated into ROM tables at build time.
- No malloc, no task creation at runtime → memory use is exactly known, statically.
- The whole kernel + config is analysable for WCET and stack usage ahead of time.
```

This makes the system amenable to the [schedulability and WCET analysis](scheduling.md)
that a safety case demands.

## Tasks: basic vs extended

```
Basic task      runs to completion; can only be preempted, never self-suspend.
                Lighter: tasks can share one stack.
Extended task   may WAIT on events (block mid-execution); needs its own stack.

States: SUSPENDED ─activate─► READY ─dispatch─► RUNNING ─terminate─► SUSPENDED
                                  ▲                   │
                                  └──── (extended: WAITING on event ◄─┘)
```

Scheduling is fixed-priority preemptive (see [scheduling](scheduling.md)), but each task
can be declared **preemptable** or **non-preemptable**, giving coarse control over
critical sections without disabling interrupts.

## Conformance classes

OSEK defines four classes by whether tasks can wait and whether multiple activations are
allowed:

```
        │ basic tasks only │ extended tasks
────────┼──────────────────┼────────────────
single  │      BCC1        │     ECC1
multiple│      BCC2        │     ECC2

BCC1 = simplest (one task per priority, no waiting); ECC2 = full (waiting + queued
activations). You pick the smallest class your application needs.
```

## Resources and the priority ceiling protocol

OSEK mutual exclusion uses **resources**, and the standard *mandates* the
[Priority Ceiling Protocol (OSEK PCP)](priority_inversion.md): on `GetResource`, the task
is immediately raised to the resource's ceiling priority. This guarantees:

```
- Bounded priority inversion (one critical section at most).
- No deadlock among resources.
- No need for the priority-inheritance bookkeeping of dynamic RTOSes.
```

It's the same protocol described in [priority inversion](priority_inversion.md), chosen
here precisely because the static task set makes ceilings computable at build time.

## Alarms, counters, and the schedule table

Time-driven activation is built from **counters** (ticked by a hardware timer or an angle
sensor) and **alarms** that fire actions when a counter reaches a value:

```
   counter (e.g. 1 ms tick, or crankshaft degrees)
        │ increments
        ▼
   alarm ──expires──► activate a task / set an event / call a callback
```

AUTOSAR adds **schedule tables** — a statically defined, offset-based sequence of
expiry points — giving deterministic, synchronizable time-triggered scheduling (useful for
angle-synchronous engine control).

## AUTOSAR Classic: OS in the basic software

AUTOSAR Classic Platform standardizes the whole ECU software stack; the OS is one module
of the Basic Software (BSW), layered under the Runtime Environment (RTE):

```
   ┌─────────────── Application SW components ───────────────┐
   │            (port-mapped, hardware-independent)          │
   ├──────────────────── RTE (generated glue) ──────────────┤
   ├─────────────── Basic Software (BSW) ───────────────────┤
   │   OS (OSEK-based)  │  COM / CAN  │  Diagnostics  │ ...  │
   ├──────────────── Microcontroller Abstraction (MCAL) ────┤
   └──────────────────────── hardware ──────────────────────┘
```

AUTOSAR OS extends OSEK with **memory protection** (via the [MPU](../embedded/mpu.md)),
**timing protection** (execution-budget and inter-arrival monitoring to catch overruns),
and **OS-Applications** that group tasks into isolated trust domains — features driven by
[ISO 26262 / ASIL](../embedded/functional_safety.md) freedom-from-interference
requirements.

## Where this connects

- [Functional safety](../embedded/functional_safety.md) — ISO 26262/ASIL is the reason for
  static config, memory/timing protection, and OS-Applications
- [Priority inversion](priority_inversion.md) — OSEK mandates the priority ceiling protocol
- [Scheduling](scheduling.md) — fixed-priority preemptive with build-time WCET/stack
  analysis; schedule tables are time-triggered scheduling
- [ARINC 653](arinc653.md) — the avionics counterpart: partitioning instead of OSEK's
  shared-stack task model
- [CAN](../embedded/can.md) / [LIN & FlexRay](../embedded/lin_flexray.md) — the in-vehicle
  networks AUTOSAR COM rides on
- [MPU](../embedded/mpu.md) — enforces AUTOSAR memory protection between OS-Applications
- [MISRA / coding standards](../embedded/coding_standards.md) — the C subset these ECUs
  are written in

## Pitfalls

```
- Treating OSEK like a dynamic RTOS: there is no runtime task creation or heap. Everything
  is declared in OIL and generated.
- Using extended tasks where basic tasks suffice: extended tasks each need their own stack,
  inflating RAM. Prefer basic (stack-sharing) tasks.
- Ignoring the resource ceiling configuration — a wrong ceiling silently breaks the PCP
  guarantees (bounded blocking, no deadlock).
- Long non-preemptable tasks blocking higher-priority activations; balance preemptability
  against shared-data safety.
- Forgetting timing protection budgets in AUTOSAR: an overrunning task can starve others;
  configure execution-time and arrival monitoring.
- Confusing AUTOSAR Classic (this static, OSEK-based platform) with AUTOSAR Adaptive
  (a POSIX/C++ platform for high-performance ECUs) — different OS model entirely.
```
