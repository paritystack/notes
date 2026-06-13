# Circuits

## Overview

A circuit is a closed loop through which [current](charge_current.md) can flow. "Closed" is the key word — break the loop anywhere and current stops. Once you understand [voltage](voltage.md), [current](charge_current.md), and [resistance](resistance.md) as isolated ideas, circuits are where those ideas combine. This page covers how components connect (series vs parallel), what determines voltage and current at every point, and the two fundamental laws — **Kirchhoff's Laws** — that let you analyse any circuit no matter how complex.

```
The simplest closed circuit:

      +9V ──────────────┐
                         │
                        [R]  ← resistor (load)
                         │
       0V (GND) ─────────┘

  Current flows: battery(+) → wire → R → wire → battery(−)
  Break any wire = circuit "opens" = current stops
```

## Key Concepts

### Series Circuits

Components in series share the **same current**. There is only one path for current to take.

```
    +V ──[R1]──[R2]──[R3]── 0V

  Current through R1 = current through R2 = current through R3 = I

  Total resistance:  R_total = R1 + R2 + R3
  Total current:     I = V / R_total
  Voltage drop across each:
      V1 = I × R1
      V2 = I × R2
      V3 = I × R3
      V1 + V2 + V3 = V  (the voltage drops add up to the supply)
```

Analogy: Single-lane road — every car (electron) goes through every traffic light (resistor).

### Parallel Circuits

Components in parallel share the **same voltage**. Current splits among multiple paths.

```
        +V
         │
    ┌────┴────┬────────┐
   [R1]      [R2]     [R3]
    │         │        │
    └────┬────┴────────┘
         │
        0V (GND)

  Voltage across each = V  (same for all)
  Current through each:
      I1 = V / R1
      I2 = V / R2
      I3 = V / R3
  Total current from supply: I_total = I1 + I2 + I3
  Total resistance: 1/R_total = 1/R1 + 1/R2 + 1/R3
```

Analogy: Multi-lane highway — cars (electrons) spread out; total throughput is higher.

### Kirchhoff's Current Law (KCL)

> **The sum of all currents entering a node equals the sum of currents leaving it.**

Nothing is created or destroyed. Electrons are conserved.

```
          I1 = 3A →
                    ┌─── I2 = 1A ──→
     ───────────────┤
                    └─── I3 = 2A ──→

  KCL: I1 = I2 + I3  →  3 = 1 + 2  ✓
```

### Kirchhoff's Voltage Law (KVL)

> **The sum of all voltage rises and drops around any closed loop equals zero.**

All the energy pumped in by sources must be used up by the loads.

```
  Around this loop (clockwise):

  +9V ─[R1=6Ω]─[R2=3Ω]─ 0V

  Source gives +9 V
  R1 drops −6 V  (I×R = 1A × 6Ω)
  R2 drops −3 V  (I×R = 1A × 3Ω)
  Sum: +9 − 6 − 3 = 0  ✓
```

KVL is the formal way to express the intuition that "all the voltage is used up".

### Mixed Series-Parallel Circuits

Real circuits are combinations. The approach:

1. Identify purely parallel groups → collapse to equivalent single R
2. Now some series elements appear → collapse those
3. Repeat until one R remains
4. Compute total current, then work backwards to find branch currents and voltages

```
  Example:
      +12V ──[R1=2Ω]──┬──[R2=6Ω]──┐
                       │            ├── 0V
                       └──[R3=3Ω]──┘

  Step 1: R2 ∥ R3 = (6×3)/(6+3) = 2 Ω
  Step 2: R1 + 2Ω = 4 Ω total
  Step 3: I_total = 12V / 4Ω = 3 A
  Step 4: V across R1 = 3A × 2Ω = 6V
          V across R2∥R3 = 12 − 6 = 6V
          I through R2 = 6V/6Ω = 1A
          I through R3 = 6V/3Ω = 2A  ✓ (1+2=3 total)
```

### Short Circuit and Open Circuit

```
Short circuit:  two points connected by zero resistance
                → I = V / 0 = ∞  (limited only by source resistance)
                → potentially destructive current, heat, fire

Open circuit:   broken path, infinite resistance
                → I = V / ∞ = 0
                → no current flows at all

These are the two failure modes of most circuit faults.
```

## How It Works

Real circuits involve many nodes (junctions) and meshes (loops). KCL applied to every node and KVL applied to every loop produces a system of equations that can be solved simultaneously. This is the foundation of computer circuit simulators (SPICE and its descendants) — they do exactly this algebraically for circuits with millions of nodes.

For hand analysis, two systematic techniques are:
- **Node voltage method** — apply KCL at each node, express currents in terms of node voltages
- **Mesh current method** — apply KVL around each loop, solve for mesh currents

## Pitfalls

- **Leaving a circuit open** — forgetting to connect GND or leaving a wire disconnected means no current flows and the circuit doesn't work. Always verify the loop is closed.
- **Creating an accidental short** — a stray wire or solder bridge between supply and ground discharges the supply instantly, often destroying something.
- **Applying series formula to parallel (or vice versa)** — identify the topology first; don't guess.
- **Forgetting KVL when tracing voltage** — in a series chain, each component "eats" some of the supply voltage. The sum must equal the supply, or you've made an arithmetic error.

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — Ohm's law is the engine; Kirchhoff's laws are the framework
- [Capacitors](capacitors.md) — RC circuits; capacitor voltage follows an exponential as the loop charges
- [Inductors](inductors.md) — RL circuits; inductor current follows an exponential as the loop energises
- [Filters](filters.md) — filter circuits are analysed with KVL/KCL in frequency domain
- [Power & Energy](power.md) — power is summed across all branches using node voltages
