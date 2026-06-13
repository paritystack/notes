# 20 · Read one real datasheet end-to-end

## Overview

You've used a [sensor](17_i2c_sensor.md), a [MOSFET](08_mosfet_fan.md), and a
[regulator](21_ldo_regulator.md)-to-be by following tutorials. To *design* — which Phase 6
demands — you must size and wire a part from its [datasheet](../electronics/datasheets.md)
alone. This rung is reading, not building: take one component you already own and extract
everything you'd need to use it in a design you've never seen a tutorial for. It's the skill
that makes you independent of tutorials.

```
   A datasheet's anatomy (where to look first):

   1. First page    → what it is, key specs, block diagram
   2. Pinout        → which pin is which (and package variants)
   3. Abs. max      → the "do not exceed or it dies" table
   4. Recommended   → the "operate here" conditions
   5. Electrical    → typical/min/max for every parameter
   6. Timing        → setup/hold, clock rates, rise times
   7. App circuits  → the manufacturer's reference wiring
```

## What you'll need

No new parts. Pick one component you own with a real datasheet — ideal choices: the
**IRLZ44N** MOSFET, the **AMS1117/MCP1700** [regulator](21_ldo_regulator.md), or the
**ATmega328P** itself (your capstone brain). Download the PDF.

## The exercise

Work top to bottom and write down the answers — for, say, the **IRLZ44N**:

```
  Absolute maximum ratings:   Vds max? Id max? gate Vgs max?
  Threshold Vgs(th):          min/typ/max — will 5 V fully turn it on?
  On-resistance Rds(on):      at Vgs = 4.5 V vs 10 V — how much heat at your current?
  Power & thermal:            Pd max, junction-to-ambient θ — needs a heatsink?
  Gate charge Qg:             how hard is it to switch fast (PWM)?
  Reference circuit:          how does the maker say to wire it?
```

Then compute one real number, e.g. heat in your [fan](08_mosfet_fan.md) driver:
`P = I² × Rds(on)`. For 0.5 A through ~0.04 Ω → 0.01 W — cold. Now you *know*, instead of
hoping. Repeat the absolute-max check: confirm your supply never exceeds any rating.

## It works when…

- [ ] You can state the part's absolute-max ratings and your design's margin to each.
- [ ] You found one spec that surprised you (e.g. Vgs(th) range, dropout, or quiescent current).
- [ ] You computed at least one real number (power, current, or timing) from the tables.

## What's happening

A [datasheet](../electronics/datasheets.md) is a contract: **absolute maximum** ratings are
the "instant death" limits (never operate there); **recommended operating conditions** are
where the part behaves as specified; the **electrical characteristics** give min/typ/max
because every part varies — good designs work across the whole range, not just "typ". The
**application circuits** encode the manufacturer's hard-won wiring (decoupling caps,
pull-ups, reference values). Learning to pull numbers from these tables is precisely what
lets you choose a [regulator](21_ldo_regulator.md), size a [resistor](../electronics/resistance.md),
or wire the [ATmega328](27_kicad_schematic_capstone.md) on your own board without a tutorial.

## Pitfalls

- **Reading "typical" as "guaranteed"** — design to min/max, not typ, or parts at the edge of tolerance will fail in the field.
- **Confusing absolute-max with operating** — abs-max is the cliff edge, not a target. Leave margin (e.g. ≤80% of a rating).
- **Ignoring conditions on a spec** — Rds(on) "0.022 Ω" might be *at Vgs = 10 V*; at 5 V it's higher. Always read the test condition next to the number.
- **Skipping the thermal section** — a part within its electrical limits can still cook if you ignore power dissipation and θ(JA).

## Where this connects

- [Reading a Datasheet](../electronics/datasheets.md) — the full reference this rung practises
- [Circuit Design](../electronics/circuit_design.md) — where datasheet numbers become design choices
- [27 · KiCad schematic](27_kicad_schematic_capstone.md) — you'll apply this to the ATmega328 and friends
- **Previous:** [19 · Logic analyzer](19_logic_analyzer_i2c.md) · **Next:** [21 · LDO regulator](21_ldo_regulator.md)
