# 03 · Series & parallel — measure the voltage drops

## Overview

You sized one resistor in [rung 02](02_light_an_led.md). Now you'll wire resistors in
**series** and **parallel** and watch [Kirchhoff's laws](../electronics/circuits.md) hold
true on the [multimeter](../electronics/prototyping.md): in series the voltages add up to
the supply; in parallel the currents split. This is the rung that turns "I memorised the
formulas" into "I can see them." No new parts — just resistors, a breadboard, and your
meter.

```
  Series (same current, voltages add):     Parallel (same voltage, currents split):

   5V ─[R1]─┬─[R2]─ GND                      5V ─┬─[R1]─┬─ GND
            │                                     ├─[R2]─┤
        measure here                              currents add up
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- Breadboard, jumpers, multimeter, 5 V source (Nano 5V pin)
- Two resistors of known, different value — e.g. 1 kΩ and 2.2 kΩ

## The build — series

1. Wire **5 V → R1 → R2 → GND** in a single line (a [series circuit](../electronics/circuits.md)).
2. Predict first: total R = R1 + R2, so I = 5 V / (R1 + R2). Each resistor drops `I × R`.
3. Measure the voltage across R1, then across R2.

```
  Predict for R1 = 1 kΩ, R2 = 2.2 kΩ on 5 V:
   I       = 5 V / 3.2 kΩ        = 1.56 mA   (same through both)
   V(R1)   = 1.56 mA × 1 kΩ      = 1.56 V
   V(R2)   = 1.56 mA × 2.2 kΩ    = 3.44 V
   V(R1)+V(R2)                   = 5.0 V  ← Kirchhoff's voltage law
```

The bigger resistor drops more voltage. The two drops *always* add back to the supply.

## The build — parallel

1. Wire **R1 and R2 both from 5 V to GND**, side by side (a [parallel circuit](../electronics/circuits.md)).
2. Predict: each branch sees the full 5 V, so each carries its own `5 V / R`. Total
   current is the sum; combined resistance is *less than the smaller* resistor.

```
  Predict for R1 = 1 kΩ, R2 = 2.2 kΩ on 5 V:
   I(R1)   = 5 V / 1 kΩ   = 5.0 mA
   I(R2)   = 5 V / 2.2 kΩ = 2.27 mA
   I(total)= 7.27 mA      R_eq = 5 V / 7.27 mA ≈ 688 Ω  (< 1 kΩ)
```

To measure a branch current, [break that branch and put the meter in series](../electronics/prototyping.md)
(red where current enters). Start on the mA range, red lead in the mA socket — then move it
back to VΩ when done.

## It works when…

- [ ] In series, your two measured voltage drops add up to ≈ 5 V.
- [ ] The larger resistor drops the larger voltage.
- [ ] In parallel, each measured branch current matches `5 V / R`, and the total exceeds either branch.

## What's happening

Series components share one current path, so charge that flows through R1 must flow through
R2 — same current, and the energy each ohm "spends" (its voltage drop) adds up to the total
the supply provides. Parallel branches share two nodes, so each sees the same voltage and
draws current independently; the supply has to provide all of them. These two facts —
[Kirchhoff's voltage and current laws](../electronics/circuits.md) — are the bedrock of
every circuit you'll ever analyse, including the [voltage divider](../electronics/resistance.md)
you'll exploit in the next rung.

## Pitfalls

- **Measuring current in parallel with the source** — putting the meter (in current mode) straight across 5 V shorts the supply through the meter and blows its fuse. Current is *always* measured in series, voltage in parallel.
- **Leaving the red lead in the mA socket** — after a current measurement, move it back to VΩ before measuring voltage again (the [rung 01](01_bench_and_multimeter.md) habit).
- **Expecting parallel resistance to go up** — it always goes *down*; adding a branch adds a path for current.

## Where this connects

- [Circuits](../electronics/circuits.md) — series/parallel rules and Kirchhoff's laws in full
- [Resistance & Ohm's Law](../electronics/resistance.md) — the per-resistor `V = IR` you applied to each branch
- [Power & Energy](../electronics/power.md) — each resistor dissipates `I²R`; sum equals supply power
- **Previous:** [02 · Light an LED](02_light_an_led.md) · **Next:** [04 · Potentiometer divider](04_pot_divider.md)
