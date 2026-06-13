# 09 · Flyback diode on an inductive load

## Overview

Switching a resistor off is harmless. Switching an **inductive** load — a relay coil, a
motor, a solenoid — is not: an [inductor](../electronics/inductors.md) resists sudden
current change and slams back with a large reverse voltage spike (hundreds of volts) the
instant you open the circuit. That spike destroys transistors and arcs across switch
contacts. A **flyback (freewheeling) diode** gives that energy a safe path to dissipate.
This rung makes the spike visible and tames it — protecting the
[MOSFET](../electronics/transistors_mosfet.md)/[BJT](../electronics/transistors_bjt.md)
drivers from the last two rungs.

```
  Diode across the coil, reverse-biased in normal operation:

   V+ ──┬───────────┐
        │           │ coil (relay/motor)
       ▲▼ diode     │
        │ (cathode  │
        │  to V+)   │
        └───────────┤ D
                  [ MOSFET ]
   GND ────────────┘

  Switch ON  → diode blocks, coil runs normally.
  Switch OFF → coil's current keeps flowing, now LOOPING through the diode,
               decaying safely instead of spiking.
```

## What you'll need

From **Stage B** of the [shopping list](README.md):

- A relay module or small DC motor (an inductive load)
- A flyback diode (1N4007 for general use; a fast Schottky like 1N5819 is even better)
- The MOSFET driver from [rung 08](08_mosfet_fan.md), breadboard, multimeter (scope ideal if you have one)

## The build

1. Build the [MOSFET driver](08_mosfet_fan.md) with a **relay coil or motor** as the load.
2. **First, no diode:** switch it on and off. If you have an [oscilloscope](../electronics/prototyping.md),
   probe the drain — you'll see a sharp spike well above the supply at turn-off. On just a
   multimeter you may see erratic behaviour, clicks, or the MOSFET running warm.
3. **Add the flyback diode** *across the coil*, **cathode (striped end) to V+**, anode to the
   drain/switch side.
4. Switch again: the spike is gone (clamped to ~0.7 V above supply). Smooth, repeatable.

```
  Diode orientation is everything:
   Cathode (stripe) → the POSITIVE supply rail.
   Backwards = a dead short across your supply when powered. Double-check.
```

## It works when…

- [ ] Without the diode you observe a turn-off spike (scope) or erratic switching/clicks.
- [ ] With the diode correctly oriented, switching is clean and the spike is clamped.
- [ ] You can state which way the diode's stripe faces and why backwards is catastrophic.

## What's happening

Current through an [inductor](../electronics/inductors.md) can't change instantly —
`V = L·di/dt`, so forcing `di/dt` toward infinity (opening the switch) produces a huge
voltage. The coil will find *some* path for its stored energy; without a diode that path is
an arc or a punch-through of your transistor. The flyback [diode](../electronics/diodes.md),
reverse-biased during normal operation, becomes forward-biased the moment the coil's voltage
reverses, letting the current freewheel in a loop and decay through the coil's own
resistance. Every relay, motor, and solenoid driver you ever build needs this — including
your capstone if it drives anything that moves.

## Pitfalls

- **Diode backwards** — cathode must face the positive rail. Reversed, it shorts the supply through the diode the instant you power up. The single most important check here.
- **Omitting it "because it worked once"** — the spike is intermittent and cumulative; the MOSFET may survive for a while, then fail. Always fit it on inductive loads.
- **Too-slow diode for fast PWM** — a 1N4007 is fine for on/off relays but slow; for [PWM](../embedded/pwm.md)-driven motors use a fast/Schottky diode.

## Where this connects

- [Inductors](../electronics/inductors.md) — why current can't change instantly and energy is stored in the field
- [Diodes](../electronics/diodes.md) — forward/reverse bias and the freewheeling action
- [Switches, Relays & Electromechanical](../electronics/switches_relays.md) — relay coils are the classic inductive load
- **Previous:** [08 · MOSFET drives a fan](08_mosfet_fan.md) · **Next:** [10 · 555 timer blink](10_555_blink.md)
