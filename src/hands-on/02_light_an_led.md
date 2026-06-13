# 02 · Light an LED (calculate the resistor)

## Overview

Your first real circuit — and the first time you'll *predict* a number with
[Ohm's law](../electronics/resistance.md) and then confirm it with the
[multimeter](../electronics/prototyping.md). An [LED](../electronics/diodes.md) is a
[diode](../electronics/diodes.md) that emits light, and like every diode it will pull as
much [current](../electronics/charge_current.md) as the supply allows — so it needs a
series resistor to survive. You'll calculate that resistor from scratch, build the circuit
on a breadboard, and measure that reality matches your math. This is the core loop of all
hardware work: *predict, build, measure, reconcile.*

```
The whole circuit:

   +5V ───[ R ]───►|─── GND
                    LED
                  (long leg = +, anode)

  The resistor sets the current; the LED sets the colour.
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- Breadboard + jumper wires, multimeter
- An LED (any colour — start with red)
- A handful of resistors (you'll calculate the value; have ~220 Ω and ~330 Ω on hand)
- A 5 V source — easiest is the **5V pin of your Arduino Nano** powered over USB, or a battery

## First, do the math

An LED has a roughly fixed **forward voltage** (`Vf`) — it "uses up" that much voltage and
the resistor must drop the rest. Typical `Vf`:

```
  Red / yellow   ≈ 1.8–2.2 V
  Green          ≈ 2.0–2.4 V
  Blue / white   ≈ 3.0–3.4 V

  Target current: ~10 mA (comfortably bright, well within a 5 mm LED's ~20 mA max)
```

Apply [Ohm's law](../electronics/resistance.md) to the resistor, which sees the
*leftover* voltage:

```
  R = (Vsupply − Vf) / I

  For a red LED on 5 V at 10 mA:
  R = (5 V − 2 V) / 0.010 A = 300 Ω

  No 300 Ω in the kit? Use the next value UP (330 Ω) → slightly less current, always safe.
```

> **Rule of thumb:** when in doubt, **330 Ω on 5 V** is a safe LED resistor for any colour.
> Higher R = dimmer but safer; never go without one.

## The build

```
  Breadboard wiring (LED straddles two rows):

   5V rail ──[ 330Ω ]── row A ─────┐
                                    │ LED long leg (anode) in row A
                                    ▼ LED short leg (cathode) in row B
   GND rail ───────────── row B ───┘
```

1. Put the **resistor** from the +5 V rail to an empty row (call it A). Resistors have no polarity — either way round is fine.
2. Put the **LED** with its **long leg (anode)** in row A (with the resistor) and its **short leg (cathode)** in another row (B).
3. Jump row B to the **GND rail**.
4. Connect 5 V and GND from your Nano (USB plugged in) to the breadboard rails.

It should light immediately. If not, see Pitfalls — 90% of the time it's the LED in
backwards.

## Measure it

Now close the predict→measure loop:

```
  1. Voltage across the LED (Vf):
     Meter on V⎓, RED on anode row, BLACK on cathode row
     → should read ≈ your assumed Vf (e.g. 1.9 V for red)

  2. Voltage across the RESISTOR:
     RED on the 5V side, BLACK on the row-A side
     → should read ≈ (5 V − Vf)

  3. Current, computed:  I = V_resistor / R
     e.g. 3.1 V / 330 Ω = 9.4 mA   ← matches your ~10 mA target!
```

Notice the two LED + resistor voltages add up to your 5 V supply — that's
[Kirchhoff's voltage law](../electronics/circuits.md) in the flesh.

## It works when…

- [ ] The LED lights at a comfortable brightness.
- [ ] The measured `Vf` is close to the value you assumed.
- [ ] `V_resistor / R` comes out near your 10 mA target.
- [ ] You can explain why removing the resistor would destroy the LED.

## What's happening

A resistor obeys [Ohm's law](../electronics/resistance.md) — its current rises smoothly
with voltage. An LED does **not**: below `Vf` almost no current flows, and above it the
current shoots up almost vertically for a tiny voltage increase. With no resistor, even
the 0.1 V of "wiggle room" between *barely on* and *way too much* is uncontrollable, so the
LED draws whatever the supply can deliver and burns out. The series resistor turns that
knife-edge into a gentle slope: pick the resistor, and you've picked the current. This is
why a current-limiting resistor sits next to nearly every indicator LED you'll ever see —
including the ones on your future [PCB](../electronics/kicad_pcb.md).

## Pitfalls

- **LED in backwards** — an LED only conducts one way. Long leg = anode (toward +), short leg = cathode (toward −, also the flat side of the rim). Backwards = no light (and that's fine, it won't be harmed at these voltages). Just flip it.
- **No resistor "just to test"** — don't. An LED across 5 V with no resistor can die in under a second. Always have the resistor in place *before* you apply power.
- **Resistor too small** — using, say, 47 Ω pushes ~65 mA through a 20 mA LED: it'll be very bright, then dead. Bigger resistor is always the safe direction.
- **Forgetting the LED's own voltage drop** — sizing the resistor as `5 V / I` instead of `(5 V − Vf) / I` overestimates the resistor slightly; harmless here, but the habit matters for tighter designs.

## Where this connects

- [Resistance & Ohm's Law](../electronics/resistance.md) — the `R = V/I` you just applied, plus the colour-band code to read your resistor
- [Diodes](../electronics/diodes.md) — why an LED is one-way and has a forward voltage
- [Circuits](../electronics/circuits.md) — the series voltages you measured are Kirchhoff's voltage law
- [Power & Energy](../electronics/power.md) — check `P = I²R` on your resistor to confirm it's well under its ¼ W rating
- **Previous:** [01 · Bench & Multimeter](01_bench_and_multimeter.md) · **Next:** *03 · Series & parallel — measure the voltage drops*
