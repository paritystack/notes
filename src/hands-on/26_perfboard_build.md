# 26 · Move the sensor-node circuit to perfboard

## Overview

Now apply your new [soldering](25_learn_to_solder.md) skill to a real circuit: rebuild the
[battery sensor node](24_deep_sleep.md) — MCU, sensor, display, power — permanently on
**perfboard**. It's the intermediate step between a [breadboard](../electronics/prototyping.md)
and a [custom PCB](27_kicad_schematic_capstone.md): you commit to a layout and make it
durable, but you can still rework it by hand. You'll plan where parts go, solder them down,
and create connections with wire — which teaches the *layout thinking* the PCB phase formalises.

```
   Breadboard ──► Perfboard ──► Custom PCB
   (loose, fast)   (permanent,   (permanent,
                    hand-wired)   manufactured)

   Perfboard: a grid of holes; you place parts and join pads with
   solder bridges or wire on the back.
```

## What you'll need

From **Stage C**: perfboard (plain or strip/“veroboard”), the [sensor node](24_deep_sleep.md)
parts (an Arduino Nano can solder straight onto perfboard via header pins to start),
hook-up/enamelled wire, your soldering kit, and the multimeter for continuity checks.

## The build

1. **Plan the layout on paper first.** Group by function: power in (LiPo/[LDO](21_ldo_regulator.md))
   on one side, the Nano in the middle, the [I²C](17_i2c_sensor.md) sensor/OLED on the other.
   Keep [power](../electronics/power.md) and ground runs short and direct.
2. **Place and solder** the parts (or female headers, so modules stay removable).
3. **Make the connections** on the back — solder bridges between adjacent pads, or short wires
   for longer runs. SDA→A4, SCL→A5, power rails to each part, common ground everywhere.
4. **Verify with continuity before powering** ([rung 01](01_bench_and_multimeter.md)): buzz
   out every net, and check for **shorts** between power and ground. This catch-before-smoke
   step is non-negotiable.
5. Power up — first from the [bench supply](21_ldo_regulator.md) with current limit set low,
   then the battery. It should behave exactly like the breadboard version.

## It works when…

- [ ] The node runs the same firmware and behaves like the breadboard build.
- [ ] Continuity confirms every intended connection and *no* power-to-ground short.
- [ ] It survives being picked up, moved, and gently shaken (the point of permanence).

## What's happening

Perfboard forces the decisions a breadboard let you dodge: *where* each part sits and *how*
power and signals route between them. Short, direct power and ground paths reduce noise and
voltage drop — a first taste of the [signal-integrity](../embedded/signal_integrity.md) and
layout concerns that dominate [PCB design](../electronics/circuit_design.md). Checking
continuity and shorts before applying power is the discipline that saves boards (and LiPo
cells). When you lay out the [KiCad PCB](27_kicad_schematic_capstone.md) next, you'll be
formalising exactly the placement-and-routing thinking you just did by hand — but the CAD
tool will check it for you.

## Pitfalls

- **No layout plan** — improvising on perfboard yields a tangled, un-debuggable mess. Sketch placement and routing first.
- **Solder bridges where you don't want them** — easy on dense perfboard; inspect and buzz every adjacent pad for unintended shorts.
- **Powering before checking** — a power-to-ground short can damage parts or the battery instantly. Continuity-check first, then current-limit the first power-up.
- **Long, looping power/ground wires** — add noise and resistance; keep them short and consider a ground "bus" wire.

## Where this connects

- [25 · Learn to solder](25_learn_to_solder.md) — the skill you're now applying for real
- [Prototyping & Test Equipment](../electronics/prototyping.md) — perfboard vs breadboard vs PCB
- [Circuit Design](../electronics/circuit_design.md) / [Signal Integrity](../embedded/signal_integrity.md) — the layout thinking this previews
- **Previous:** [25 · Learn to solder](25_learn_to_solder.md) · **Next:** [27 · KiCad schematic](27_kicad_schematic_capstone.md)
