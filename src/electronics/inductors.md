# Inductors

## Overview

An inductor is a coil of wire that resists *changes in current*. Where a [capacitor](capacitors.md) resists sudden changes in voltage, an inductor resists sudden changes in current. It stores energy in a magnetic field while current flows through it, and releases that energy (sometimes as a large voltage spike) when the current tries to change. This makes inductors essential in [power supplies](power_supplies.md) (where they transfer energy efficiently), filters, and radio circuits.

```
Flywheel analogy:

  Trying to spin up a flywheel:        Trying to stop a flywheel:
  → hard to get moving (resists start) → hard to stop (resists stopping)
  → once spinning, keeps going         → keeps going even after force removed

  Electric current in an inductor:
  → hard to start (back-EMF opposes increase)
  → once flowing, keeps flowing (back-EMF opposes decrease)
  → inductor "wants" to keep current steady
```

## Key Concepts

### How an Inductor Works

A coil of wire creates a magnetic field when current flows through it. By Faraday's Law, any change in this magnetic field induces a voltage that *opposes* the change (Lenz's Law). So:

```
  Current increasing → magnetic field growing
                     → induced voltage opposes the increase (slows it down)

  Current decreasing → magnetic field collapsing
                     → induced voltage opposes the decrease (tries to maintain it)
                     → this can produce a HIGH voltage spike (flyback)
```

The fundamental equation:

```
V = L × dI/dt

V = voltage across the inductor (Volts)
L = inductance in Henries (H)
dI/dt = rate of change of current (Amperes per second)
```

A large L or a fast change in current → large voltage across the inductor.

### Inductance

| Unit | Symbol | Typical use |
|------|--------|------------|
| Henry | H | Large power inductors |
| Millihenry | mH | Audio, small power inductors |
| Microhenry | µH | RF, buck/boost converters |
| Nanohenry | nH | Very high frequency / RF |

Inductance depends on:
- Number of turns in the coil (more turns = more inductance)
- Core material (air, ferrite, iron — ferrite greatly increases inductance)
- Cross-sectional area and length of the coil

### RL Time Constant

Like RC circuits, RL circuits have a time constant governing how fast current rises:

```
I(t) = (V / R) × (1 − e^(−t/τ))

τ = L / R   (seconds)

At t = τ:   current reaches 63% of final value
At t = 5τ:  current reaches ~99% (considered steady-state)
```

```
  Current rise in RL circuit:

  I_final = V/R ─ ─ ─ ─ ─ ─ ─ ─ ─
                    .....─────────────
                 ...
              ...
            ..
           .
  0 ────────────────────────────── time
          τ   2τ   3τ   4τ   5τ
```

### Passing DC, Blocking AC

The dual of the capacitor:
- **DC** — once current reaches steady-state, dI/dt = 0, so V across inductor = 0. Inductor looks like a short circuit to DC.
- **AC** — continuously changing current means continuous V opposition. High frequency AC sees high opposition.

**Inductive reactance** (resistance to AC):
```
XL = 2π × f × L

High frequency → high reactance (hard for AC to pass)
Low frequency  → low reactance (easy for AC to pass)
DC (f=0)       → zero reactance (no opposition)
```

This is opposite to capacitors — inductors block high frequencies, pass low ones.

### Energy Stored

```
E = ½ × L × I²

Energy in Joules, L in Henries, I in Amperes.
```

When current is interrupted suddenly (switch opens), this stored energy has nowhere to go. The inductor produces a large voltage spike to maintain the current — the **flyback voltage**. This can be hundreds of volts even from a small inductor with a low supply voltage, and it destroys transistors and other components if not handled.

A **flyback diode** (freewheeling diode) is placed across an inductive load (motor, relay coil) to give this energy a safe path to circulate.

```
  Flyback protection:

  +V ────┬──── [inductor / motor / relay coil] ──── 0V
         │                                     |
         │          [diode] (across load, reverse polarity) ←──┘

  When switch opens: inductor voltage spike forward-biases diode,
  energy circulates safely instead of destroying the switch.
```

### Common Inductor Types

| Type | Core | Typical use |
|------|------|------------|
| Air core | None | RF, very high frequency |
| Ferrite core | Ferrite | Power converters, EMI filters |
| Toroid | Ferrite/iron ring | Low EMI, efficient coupling |
| Coupled (transformer) | Ferrite/iron | [Transformers](power_supplies.md), isolated supplies |

## Pitfalls

- **Switching inductive loads without flyback protection** — always add a diode across any relay coil, motor winding, or solenoid driven by a transistor. Without it, the voltage spike kills the transistor.
- **Ignoring DC resistance (DCR)** — the wire in an inductor has resistance. A 10 µH inductor might have 100 mΩ DCR, which causes power loss and voltage drop in power circuits.
- **Exceeding saturation current** — at a certain current, the core material saturates: inductance drops sharply. Always check the inductor's saturation current rating in power supply designs.
- **EMI from large inductors** — inductor coils radiate magnetic fields. Poor placement on a PCB couples noise into nearby sensitive circuits (ADC inputs, RF traces).

## Where this connects

- [Capacitors](capacitors.md) — dual of the inductor; RC and LC circuits form complementary pairs
- [Circuits](circuits.md) — KVL loop analysis with L includes the V = L dI/dt term
- [Filters](filters.md) — LC filters combine inductors and capacitors for sharper frequency response
- [Power Supplies](power_supplies.md) — buck and boost converters use inductors to transfer energy efficiently
- [Diodes](diodes.md) — flyback diodes protect switches from inductive voltage spikes
- [Motor Control](../embedded/motor_control.md) — motors are inductive loads; back-EMF is the flyback phenomenon
