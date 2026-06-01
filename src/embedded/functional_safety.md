# Functional Safety

## Overview

Functional safety is the discipline of building embedded systems that fail *safely* — where
a hardware fault or software bug cannot cause unacceptable harm to people. It is what
separates a hobby project from firmware that drives a car's brakes, an insulin pump, an
industrial robot, or a power inverter. Where [MISRA C & Defensive Firmware](coding_standards.md)
gives you coding rules and [Embedded Unit Testing](embedded_testing.md) gives you
confidence the code does what you intended, functional safety adds a system-level question:
**what happens when something breaks anyway** — a bit flips in [RAM](memory_management.md),
a sensor lies, the CPU mis-executes, the clock stops? The answer is a mix of standards
(ISO 26262, IEC 61508), redundancy, and on-line self-tests, leaning on hardware you've met
elsewhere: the [watchdog](watchdog.md), the [MPU](mpu.md), [ECC](cache_tcm.md) memory, and
lockstep cores.

```
   NORMAL FIRMWARE                 SAFETY FIRMWARE
   "make it work"                  "make it FAIL SAFE"
        │                               │
        ▼                               ▼
   detect happy path             detect FAULTS (RAM, flash, CPU, clock,
                                  sensor, stack) ──► enter a known SAFE STATE
                                  within a bounded FAULT TOLERANT TIME
```

## The Standards: SIL and ASIL

Two standards dominate, both descended from IEC 61508:

- **IEC 61508** — the generic industrial functional-safety standard. Defines **Safety
  Integrity Levels SIL 1–4** (4 = most stringent), based on the required risk reduction.
- **ISO 26262** — the automotive adaptation. Defines **ASIL A–D** (D = most stringent),
  derived from a hazard analysis combining **Severity × Exposure × Controllability**.
- Domain cousins: **IEC 62304** (medical device software), **DO-178C** (avionics),
  **EN 50128** (rail), **IEC 60730** (appliances).

```
   Risk assessment ──► required integrity level ──► mandated rigor
                                                     ┌──────────────────────────┐
   ASIL A  (lowest)  ─────────────────────────────► │ more reviews, more tests, │
   ASIL B                                            │ redundancy, diagnostic    │
   ASIL C                                            │ coverage, documentation,  │
   ASIL D  (highest) ─────────────────────────────► │ independent assessment    │
                                                     └──────────────────────────┘
```

The level doesn't change *what* the device does; it dictates *how rigorously* you must
develop, verify, and document it — and how much fault detection (**diagnostic coverage**)
the running system must provide. Higher level → process, traceability, and hardware
metrics all scale up.

## Key Concepts

- **Safe state** — the condition the system enters on a detected fault (motor de-energized,
  valve closed, output disabled). Reaching it must itself be reliable — often a
  *de-energize to safe* design so loss of power = safe.
- **Fault Tolerant Time Interval (FTTI)** — the maximum time from a fault occurring to a
  hazard, *minus* margin; all detection + reaction must complete inside it. Drives how fast
  your self-tests and [watchdog](watchdog.md) must act.
- **Single-point fault** — one failure that defeats the safety function with no detection;
  the thing safety architecture exists to eliminate (via redundancy or diagnostics).
- **Diagnostic coverage (DC)** — fraction of dangerous faults the system *detects*. ASIL D
  demands very high DC, which is why on-line self-tests pervade safety firmware.
- **Freedom From Interference (FFI)** — a lower-criticality task must not corrupt a
  higher-criticality one. Enforced with the [MPU](mpu.md) (memory partitioning), time
  budgets, and separate stacks — closely related to [TrustZone-M](trustzone_m.md) isolation.

## Redundancy & Architecture Patterns

When you can't make a single channel reliable enough, you duplicate and compare:

```
   1oo1 (single)        1oo2 / 2oo2 (compare)        2oo3 (vote)
   ┌────────┐           ┌────────┐ ┌────────┐        ┌───┐┌───┐┌───┐
   │ channel│           │ chan A │ │ chan B │        │ A ││ B ││ C │
   └────────┘           └───┬────┘ └───┬────┘        └─┬─┘└─┬─┘└─┬─┘
   no detection             └── compare ─┘              └─ majority ─┘
                            mismatch → safe state       outvote a faulty one,
                                                        keep running (TMR)
```

- **Homogeneous redundancy** — two identical channels; catches random hardware faults but
  not a shared design/software bug.
- **Diverse (heterogeneous) redundancy** — two *different* implementations (different MCUs,
  different teams, different algorithms); catches systematic faults too. Required at the
  highest levels.
- **2oo3 / TMR (triple modular redundancy)** — three channels vote; the system *keeps
  operating* through a single fault (fault-tolerant, not just fail-safe).
- **Monitor / safety companion** — an asymmetric pattern: a small independent monitor MCU
  (or safety MCU) watches the main controller and forces the safe state if it misbehaves.

## Lockstep Cores

Many safety MCUs (TI Hercules, Infineon AURIX, some STM32, NXP) run **two CPU cores in
lockstep**: both execute the *same* instruction stream, and hardware compares their outputs
every cycle. A divergence means one core hit a random fault — the comparator immediately
flags it and the chip enters a safe state. The second core is *delayed* by a few cycles and
sometimes physically/temporally offset so a single transient (a voltage glitch, a radiation
upset) doesn't hit both identically.

```
   instruction stream
        │
        ├──► Core 1 ──┐
        │             ├──► COMPARATOR ──► mismatch? → fault signal → safe state
        └──► Core 2 ──┘     (every cycle)
          (delayed N cycles)
```

Lockstep gives very high diagnostic coverage of the CPU itself for "free" at runtime — but
the second core does no useful extra work (no performance gain), which is the cost.

## Runtime Self-Tests

Safety firmware continuously checks its own substrate; representative on-line tests:

- **RAM test (March)** — periodically run a March pattern over RAM (or rely on **ECC** RAM)
  to catch stuck bits and coupling faults; relates to [memory management](memory_management.md).
- **Flash CRC** — checksum/CRC the program [flash](linker_scripts.md) at startup and
  periodically to detect bit rot or corruption before it's executed.
- **CPU register / ALU test** — a self-test routine (often vendor-supplied, e.g. STL —
  Self-Test Library) that exercises CPU registers and ALU paths for stuck faults.
- **Clock monitor** — an independent oscillator/CSS detects a stopped or out-of-range main
  [clock](clock_systems.md); a frozen clock is a classic dangerous fault.
- **Windowed watchdog** — the [watchdog](watchdog.md) must be kicked within a *window* (not
  too early, not too late), so both a hung CPU *and* a runaway loop are caught; often an
  external watchdog so an internal failure can't disable it.
- **Stack monitoring** — an [MPU](mpu.md) guard region or stack canary detects overflow
  before it corrupts adjacent data.
- **Program-flow / control-flow monitoring** — checkpoints verify the code executed its
  intended sequence, catching a corrupted PC or skipped step.
- **Plausibility / range checks** — reject sensor values outside physically possible bounds;
  cross-check redundant or diverse [sensors](sensors.md) against each other.

## Where this connects

- [MISRA C & Defensive Firmware](coding_standards.md) — the coding-standard and language-subset foundation that safety processes mandate.
- [Embedded Unit Testing](embedded_testing.md) — safety integrity levels demand high test/structural coverage and traceability to requirements.
- [Watchdog](watchdog.md) — the windowed watchdog is the cornerstone runtime fault detector; often external for independence.
- [MPU](mpu.md) / [TrustZone-M](trustzone_m.md) — memory partitioning enforces freedom-from-interference between mixed-criticality tasks.
- [Cache & TCM](cache_tcm.md) — ECC-protected RAM/TCM provides single-bit-error correction and double-bit detection.
- [Memory Management](memory_management.md) — static allocation and stack sizing are near-mandatory; dynamic allocation is typically banned.
- [Sensors & Sensor Fusion](sensors.md) — redundant/diverse sensing and plausibility checks detect a lying sensor.
- [Clock Systems](clock_systems.md) — clock-security circuitry detects a stopped/erratic clock.

## Pitfalls

1. **Confusing "safe" with "reliable."** A system that fails *often but always safely* can
   meet safety goals; one that rarely fails but fails *dangerously* does not. Design the
   failure mode, not just MTBF.
2. **Homogeneous redundancy against software bugs.** Two identical channels share the same
   bug and fail together. Systematic faults need *diverse* implementations or independent monitors.
3. **Watchdog kicked from a timer ISR.** If a periodic interrupt refreshes the watchdog, a
   hung main loop still gets petted — the watchdog proves nothing. Kick it from the main flow,
   ideally a windowed watchdog tied to a flow check.
4. **No defined safe state (or an unreachable one).** Detection is useless if the reaction
   path can't be trusted; prefer de-energize-to-safe so loss of power is inherently safe.
5. **Self-tests that don't fit inside the FTTI.** A RAM test that takes longer than the
   fault-tolerant time leaves a window of undetected hazard. Budget detection + reaction
   against the FTTI.
6. **Dynamic allocation / recursion / unbounded loops.** Banned or heavily restricted in
   safety code because they make worst-case timing and memory non-deterministic.
7. **Treating certification as a final step.** Functional safety is a *process* (the
   "safety lifecycle") with traceability from hazard → requirement → design → test; you
   cannot bolt it on at the end.
8. **Ignoring common-cause failures.** Shared power, clock, or ground defeats redundancy —
   a single brownout takes out "independent" channels at once. Separate the shared resources.

## See Also

- [MISRA C & Defensive Firmware](coding_standards.md) — the coding-rule foundation
- [Watchdog](watchdog.md) — the primary runtime fault detector
- [MPU](mpu.md) — freedom-from-interference between tasks
- [Embedded Unit Testing](embedded_testing.md) — the verification rigor safety levels require
