# Diodes

## Overview

A diode is the simplest semiconductor component. It allows [current](charge_current.md) to flow in one direction only — like a one-way check valve in a water pipe. Forward biased (connected the right way), it conducts; reverse biased (backwards), it blocks. This simple property is used everywhere: converting AC to DC, protecting circuits from reverse voltage, emitting light (LEDs), and clamping voltage spikes. Understanding diodes is also the foundation for understanding [transistors](transistors_bjt.md), which are built from two back-to-back junctions.

```
Check valve analogy:

  Forward (conducting):             Reverse (blocking):
  Water pressure →                  ← Water pressure
  >>>  [valve open] >>>             <<< [valve closed] <<<

  In electronics:
  Anode (+) ──►|── Cathode (−)     Current flows →  (forward biased)
  Anode (−) ──►|── Cathode (+)     No current      (reverse biased)

  ►| is the schematic symbol (triangle pointing in direction of conventional current)
```

## Key Concepts

### The P-N Junction

A diode is made by joining two types of semiconductor:
- **P-type** — has excess "holes" (missing electrons, behave as positive charge carriers)
- **N-type** — has excess electrons

```
  P region  |  N region
  (holes)   |  (electrons)
  + + + +   |   − − − −
  + + + +   |   − − − −
            |
         junction
```

At the junction, electrons and holes recombine, forming a **depletion region** — a thin zone with no free carriers that acts as a barrier. Applying a forward voltage overcomes this barrier and allows current to flow.

### Forward Voltage Drop

When conducting, a diode has a fixed voltage drop across it regardless of current (approximately):

| Type | Forward voltage (V_f) | Typical use |
|------|-----------------------|------------|
| Silicon (1N4007) | ~0.7 V | General rectifier |
| Schottky (1N5819) | ~0.3 V | Low-drop rectifier, fast switching |
| Germanium | ~0.2 V | Old radios, some detectors |
| LED (red) | ~1.8–2.0 V | Indicator light |
| LED (green) | ~2.0–2.2 V | Indicator light |
| LED (blue/white) | ~3.0–3.5 V | Modern LEDs |

```
  Voltage across a conducting silicon diode:

  Anode: +5 V                 Cathode: 5 − 0.7 = 4.3 V
      +5V ──────────[D]─────── 4.3V
                    ↑
               0.7 V drop
```

### I-V Curve

```
  Current (mA)
   ↑
   │     ← forward bias →
   │              ....──────
   │          ....
   │        ..
   │       .
   │      .
───┼──────────────────────────→ Voltage (V)
   │← −    0    +0.7V   +
   │
   │         ← reverse bias, blocking until:
   │
   │
   │__
  ─────  ← reverse breakdown (Zener effect)
```

In reverse, a tiny leakage current flows, but essentially the diode blocks until the **reverse breakdown voltage** is reached — at that point it conducts suddenly. For regular diodes this destroys them. For **Zener diodes**, this is intentional and useful.

### Diode as a Rectifier

The primary use of a rectifier diode: convert AC to DC by blocking the negative half-cycles.

```
  Half-wave rectifier:

  AC input:     /\ /\ /\          after diode:   /\  /\  /\
               /  V  V  V         (negative half  │   │   │
  ────────────/──────────────     cycles blocked) │   │   │
                                  ───────────────────────────

  Full-wave bridge rectifier (4 diodes) passes both half-cycles:

        D1        D2
   +AC ─►├────────┤►─ +DC out
   −AC ─►├────────┤►─ GND
        D3        D4

  Both positive and negative AC swings converted to positive DC.
  A large [capacitor](capacitors.md) then smooths the ripple.
```

### Zener Diode

A Zener diode is designed to conduct in **reverse** at a precise, stable breakdown voltage (the **Zener voltage**, V_Z). Used as a simple voltage reference or clamp.

```
  5.1 V Zener clamp:

       Vin (varying) ─── [R] ─── Output (clamped to 5.1 V)
                                  │
                                 [Z] ← Zener, cathode up
                                  │
                                 GND

  If Vin tries to push output above 5.1 V, Zener conducts,
  clamping the output. Below 5.1 V, Zener blocks, Ohm's law applies.
```

### LED (Light-Emitting Diode)

An LED is a diode that emits light when current flows forward through it. The current — not the voltage — determines brightness. Always use a **series resistor** to limit current:

```
  R = (V_supply − V_f) / I_LED

  Example: 5 V supply, red LED (V_f = 2.0 V), want 20 mA:
  R = (5 − 2.0) / 0.020 = 150 Ω → use 150 Ω or 160 Ω

  +5V ──[150Ω]──[LED]── GND
```

Without the resistor, the LED will draw maximum current and burn out in seconds.

## Pitfalls

- **Connecting a diode backwards** — no current flows, circuit doesn't work. Double-check the cathode stripe (the marked end on the component body).
- **Omitting the current-limiting resistor for an LED** — LEDs have no internal resistance limit. Without a series resistor, they self-destruct.
- **Ignoring the forward voltage drop** — 0.7 V sounds small, but in a low-voltage circuit (3.3 V) it's 21% of the supply. Use a Schottky diode (0.3 V drop) when this matters.
- **Forgetting flyback diodes on inductive loads** — see [Inductors](inductors.md). Any relay, motor, or solenoid driven by a transistor needs a diode across it.
- **Reverse bias voltage rating** — every diode has a maximum reverse voltage (PIV). Exceed it and the diode conducts destructively. In mains rectifier circuits, this rating must account for the peak voltage, not just the RMS value.

## Where this connects

- [Circuits](circuits.md) — a forward-biased diode introduces a fixed 0.7 V drop in a loop; include it in KVL analysis
- [Inductors](inductors.md) — flyback diodes protect transistors from inductive spikes
- [Power Supplies](power_supplies.md) — rectifier bridges, Schottky diodes in switching supplies
- [BJT Transistors](transistors_bjt.md) — built from two P-N junctions; understanding diodes makes BJT operation clear
- [MOSFET Transistors](transistors_mosfet.md) — MOSFETs have a built-in body diode with important consequences
