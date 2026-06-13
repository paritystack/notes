# 01 · Set up the bench & meet the multimeter

## Overview

Before you build anything, you need to be able to *measure* it. This first rung has no
circuit to design — its whole job is to make the [multimeter](../electronics/prototyping.md)
feel familiar, because every later build is debugged with it. You'll measure a known
[voltage](../electronics/voltage.md) (a battery), buzz out a connection
([continuity](../electronics/prototyping.md)), and learn the one habit that prevents most
beginner accidents: *think before you touch the dial*. This is the hardware equivalent of
getting your debugger and print statements working before writing real code.

```
Tonight's bench:

   ┌───────────────┐     ┌──────────────┐
   │  BREADBOARD   │     │  MULTIMETER  │      9V battery
   │ ○○○○○○○○○○○○ │     │  ┌────────┐  │       ┌────┐
   │ ○○○○○○○○○○○○ │     │  │  9.41  │  │       │ +  │
   │ ○○○○○○○○○○○○ │     │  └────────┘  │       │ -  │
   └───────────────┘     │  red ● ● blk │       └────┘
                         └──────────────┘
   Nothing to wire yet — just get comfortable measuring.
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- Digital multimeter (with two probes — red and black)
- A 9 V battery (or any battery: AA, coin cell, USB power bank)
- A breadboard and a couple of jumper wires (just to practice continuity)

## The build

There's no circuit to assemble — this rung is about the instrument.

### 1. Identify the probes and sockets

```
  Multimeter sockets (typical):

   ┌──────────────────────────────┐
   │   [10A]   [mAμA]      [COM]   [VΩ]   │
   │     ▲        ▲          ▲       ▲     │
   │  high current  small   BLACK  RED for │
   │  (own socket)  current always volts/Ω │
   └──────────────────────────────┘

  BLACK probe → COM  (always, never moves)
  RED probe   → VΩ   (for voltage, resistance, continuity)
```

The red probe only moves to a current socket when you deliberately measure current
(a later rung). For now it lives in **VΩ**.

### 2. Measure a known voltage

```
  Dial → V⎓  (DC volts; the straight/dashed line, NOT the wavy ~ which is AC)
  If your meter is not auto-ranging, pick the 20 V range.

  RED probe  → battery +
  BLACK probe→ battery −

  Reading: ~9 V for a fresh 9 V battery (1.5 V for an AA, ~3 V coin cell)
```

A "tired" 9 V battery might read 8.2 V; a dead one well under 7 V. Swapping the probes
just flips the sign (`-9.0`) — harmless on DC, and a useful way to confirm polarity.

### 3. Buzz out a connection (continuity)

```
  Dial → continuity  (the ·)))  speaker / sound-wave symbol)

  Touch the two probes together   → BEEP   (a closed path)
  Hold them apart                 → silence (an open path)

  Now poke both ends of the SAME breadboard column   → BEEP
  Poke two DIFFERENT columns / across the centre gap → silence
```

This is how you'll later confirm a wire is really connected, find an accidental
short, or check that a switch actually opens and closes — see
[Prototyping](../electronics/prototyping.md) for the full breadboard internal map.

## It works when…

- [ ] You read your battery's voltage and it's in the expected range.
- [ ] Touching the probes together beeps; holding them apart is silent.
- [ ] You can predict which breadboard holes are connected and confirm it with the beep.

If all three are true, you can now *measure* — which means you can debug everything that
follows.

## What's happening

The DC-volts mode puts a very high resistance (megohms) across the two probes and reports
the [voltage](../electronics/voltage.md) it sees — high resistance so the meter barely
disturbs the circuit it's measuring. Continuity mode does the opposite: it pushes a tiny
test current out the red probe and beeps if it finds a low-[resistance](../electronics/resistance.md)
path back. Understanding that a voltmeter is "high resistance, measure across" while an
ammeter (later) is "low resistance, measure in series" is the single idea that keeps you
from blowing the meter's fuse.

## Pitfalls

- **Leaving the red probe in the current socket** — the most common way to destroy a meter (or its fuse). Measuring *voltage* with the red lead in the 10 A socket creates a near-short across whatever you probe. After any current measurement, move the red lead straight back to VΩ.
- **AC vs DC dial position** — batteries and breadboards are DC (`V⎓`). The wavy `V~` is for mains/AC and will read garbage on a battery.
- **Measuring resistance or continuity on a powered circuit** — both modes inject the meter's own test current; an external voltage on top of that gives wrong readings and can damage the meter. Power off first.
- **Trusting a reading without checking the range** — on a manual-range meter, a `1` or `OL` alone on the display means *over-range*: bump to a higher range.

## Where this connects

- [Prototyping & Test Equipment](../electronics/prototyping.md) — the full reference for the multimeter, breadboard, soldering, and scope
- [Voltage](../electronics/voltage.md) — what you just measured, and why it's always relative to a reference point
- [Resistance & Ohm's Law](../electronics/resistance.md) — needed for the next rung, where you'll calculate an LED's resistor
- **Next:** *02 · Light an LED (calculate the resistor)* — your first actual circuit
