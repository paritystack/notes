# Transformers

## Overview

A transformer is two (or more) [inductor](inductors.md) coils wound on a shared
magnetic core. An alternating current in the first coil (the **primary**) creates
a changing magnetic field; that field passes through the core and induces a
voltage in the second coil (the **secondary**). Because the two coils share no
electrical connection, a transformer does two priceless jobs at once: it
**changes the voltage** up or down by the ratio of turns, and it provides
**galvanic isolation** between primary and secondary. Transformers only respond to
*changing* current, so they work on [AC](ac_signals.md), not DC — which is exactly
why mains [power supplies](power_supplies.md) start with one.

```
  Two coils, one core:

   Primary            Secondary
   Np turns           Ns turns
   ║)))))      ┌──┐      (((((║
   ║)))))      │##│      (((((║   ## = iron / ferrite core
   ║)))))      └──┘      (((((║      carries the magnetic field
     │                        │
   AC in                   AC out (isolated, scaled by Ns/Np)

   No wire crosses from left to right — only magnetism does.
```

## Key Concepts

### The Turns Ratio

The voltage scales with the ratio of turns; the current scales inversely (an
ideal transformer conserves power, V·I in = V·I out):

```
  Vs / Vp = Ns / Np          (voltage follows the turns ratio)
  Is / Ip = Np / Ns          (current goes the opposite way)
  Vp · Ip = Vs · Is          (power in = power out, ideal)

  n = Ns / Np  is the turns ratio.
```

| n = Ns/Np | Name | Effect |
|-----------|------|--------|
| < 1 | Step-down | Lower voltage, higher current (mains → 12 V) |
| > 1 | Step-up | Higher voltage, lower current (inverter, CRT, microwave) |
| = 1 | Isolation | Same voltage, isolation only (safety, ground-loop break) |

**Worked example** — a mains transformer steps 230 V down to drive a 12 V
secondary at 2 A:

```
  Turns ratio n = Vs / Vp = 12 / 230 = 0.052   (≈ 1 : 19)
  Primary current Ip = Is · n = 2 A × 0.052 = 104 mA
  Power transferred = 12 V × 2 A = 24 W (≈ 230 V × 0.104 A, ideal)
```

So a thin primary winding (100 mA) supports a fat secondary winding (2 A) — the
step-down trades volts for amps.

### Isolation

Primary and secondary share no copper, only flux. That means the secondary's
"ground" can float relative to the primary's. Uses:
- **Safety** — a 1:1 isolation transformer breaks the path from mains live to a
  device chassis, so touching the output and earth doesn't shock you.
- **Ground loops** — isolating audio/data lines breaks hum-causing return loops.
- **Switching supplies** — a small high-frequency transformer isolates the output
  of a flyback or forward converter from the mains.

### Why It Needs AC

Induction depends on a *changing* field (V = N·dΦ/dt). Apply steady DC and after
the initial switch-on transient the field is constant, dΦ/dt = 0, and the
secondary voltage is zero — while the primary, now just a low-resistance coil,
draws a large current and overheats. Transformers are inherently AC devices; to
"transform" DC you must first chop it into AC (what a switching converter does).

### Core and Losses

| Loss | Cause | Mitigation |
|------|-------|------------|
| Copper (I²R) | Winding resistance | Thicker wire |
| Eddy current | Currents induced in the core | Laminated / ferrite core |
| Hysteresis | Re-magnetising the core each cycle | Soft magnetic material |
| Leakage | Flux that misses the secondary | Tight coupling, good core |

Mains transformers use **laminated iron** (good at 50/60 Hz); switching supplies
use **ferrite** cores (low loss at tens of kHz to MHz), which is why a 100 W phone
charger transformer is tiny while a 100 W mains transformer is a brick.

### Common Configurations

- **Centre-tapped secondary** — a tap at the winding's midpoint gives two
  equal voltages, used for full-wave rectification with just two diodes, or for
  ± supplies.
- **Multiple secondaries** — one primary, several isolated outputs at different
  voltages.
- **Autotransformer** — a single tapped winding (no isolation) for a small
  step up/down at lower cost (e.g. a Variac).

## Pitfalls

- **Applying DC to a transformer** — saturates the core and draws huge current.
  Transformers only work on AC.
- **Assuming it changes power** — a transformer trades voltage for current; it
  can't create energy. Stepping voltage up steps current down by the same ratio
  (minus losses).
- **Core saturation** — too high a voltage or too low a frequency drives the core
  past saturation; inductance collapses and primary current spikes. Honour the
  rated voltage *and* frequency (a 60 Hz transformer run at 50 Hz sees more flux).
- **Ignoring isolation when it matters** — a switching supply without a proper
  isolation transformer (or with its barrier bridged) can put mains potential on
  the output. Safety-critical.
- **No load ≠ no current** — an unloaded transformer still draws magnetising
  current and dissipates core loss; a warm idle transformer is normal.

## Where this connects

- [Inductors](inductors.md) — a transformer is two coupled inductors; same physics, shared core
- [AC Signals & Impedance](ac_signals.md) — transformers only pass changing signals; turns ratio also transforms impedance (×n²)
- [Diodes](diodes.md) — the secondary feeds a rectifier bridge to make DC
- [Power Supplies](power_supplies.md) — the first stage of a linear supply; the isolation element of a switcher
- [Power & Energy](power.md) — V·I is conserved across the transformer (minus losses)
```
