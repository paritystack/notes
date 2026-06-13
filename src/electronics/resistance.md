# Resistance & Ohm's Law

## Overview

Resistance is what fights the flow of [current](charge_current.md). Every real material has some resistance — even a copper wire, though very little. Resistance converts electrical energy into heat. The relationship between [voltage](voltage.md), current, and resistance is captured in **Ohm's Law**: the single most used equation in electronics. Understanding this page is the prerequisite for [Circuits](circuits.md), [Capacitors](capacitors.md), [Filters](filters.md), and virtually everything else.

```
Narrow pipe analogy:

  Wide pipe (low R):          Narrow pipe (high R):
  ==============================    ====----====
  >>> lots of flow >>>              > little flow >
  ==============================    ====----====

  Same pressure (voltage), but the narrower pipe (higher resistance)
  restricts how much water (current) can flow.
```

## Key Concepts

### Ohm's Law

The cornerstone of circuit analysis:

```
V = I × R

V = voltage across the resistor (Volts)
I = current through the resistor (Amperes)
R = resistance (Ohms, symbol Ω)
```

Rearranged forms you'll use constantly:

```
I = V / R    (how much current flows for a given voltage and resistance)
R = V / I    (what resistance produces a given current from a given voltage)
```

**Example:** A 9 V battery connected to a 470 Ω resistor:
```
I = 9 V / 470 Ω = 0.019 A = 19 mA
```

### The Resistor

A resistor is a component designed to have a specific, stable resistance. Its job is to limit current, set voltages via voltage dividers, or convert current into a voltage signal.

```
Resistor symbol (schematic):

   ──[/\/\/\]──

Resistor body:

   ║ band1 band2 band3 band4 ║
   ══════════════════════════
   (color bands encode the resistance value)
```

**Color code** (4-band resistors, from left to right):

| Color | Digit | Multiplier |
|-------|-------|------------|
| Black | 0 | ×1 |
| Brown | 1 | ×10 |
| Red | 2 | ×100 |
| Orange | 3 | ×1,000 |
| Yellow | 4 | ×10,000 |
| Green | 5 | ×100,000 |
| Blue | 6 | ×1,000,000 |
| Violet | 7 | — |
| Gray | 8 | — |
| White | 9 | — |
| Gold | — | ×0.1 (tolerance ±5%) |
| Silver | — | ×0.01 (tolerance ±10%) |

Mnemonic: **B**lack **B**ears **R**ob **O**ur **Y**oung **G**irls **B**ut **V**iolets **G**row **W**ild

Example: Red–Red–Brown–Gold = 2–2–×10–±5% = **220 Ω ±5%**

### Common Resistance Values

| Value | Typical use |
|-------|------------|
| 10 Ω | Current sensing, inrush limiting |
| 100 Ω | Series protection resistor |
| 330 Ω | LED current limiting (5 V supply) |
| 1 kΩ | Pull-up/pull-down for logic signals (see [GPIO](../embedded/gpio.md)) |
| 10 kΩ | Pull-up/pull-down, voltage dividers |
| 1 MΩ | High-impedance input bias |

### Resistors in Series

When resistors are connected end-to-end (same current flows through all):

```
──[R1]──[R2]──[R3]──

R_total = R1 + R2 + R3
```

The total resistance *adds up*. Think of two narrow pipes in a row — it's even harder for water to flow.

### Resistors in Parallel

When resistors are connected side by side (same voltage across all):

```
    ┌──[R1]──┐
────┤        ├────
    ├──[R2]──┤
    └──[R3]──┘

1/R_total = 1/R1 + 1/R2 + 1/R3

For two resistors:  R_total = (R1 × R2) / (R1 + R2)
```

Parallel total is always *less than the smallest* individual resistor. Think of adding lanes to a road — more lanes, less total congestion.

### Voltage Divider

One of the most useful resistor circuits:

```
     Vin
      │
     [R1]
      │
      ├───── Vout = Vin × R2 / (R1 + R2)
      │
     [R2]
      │
     GND
```

Used to scale voltages down — for example, to measure a 12 V signal with a 3.3 V [ADC](../embedded/adc.md) input.

### Potentiometers & Variable Resistors

A **potentiometer** ("pot") is a resistor with a movable wiper that taps off any
point along its resistive track — a voltage divider you can turn with a knob.

```
  Full track between A and B (e.g. 10 kΩ);
  wiper W slides along it:

   A ●━━━━━━━━━━━━━━● B
            ▲
            W (wiper)

  Wiper at top    → Vout = Vin   (W = A)
  Wiper at middle → Vout = Vin/2
  Wiper at bottom → Vout = 0     (W = B)

  Used as a 3-terminal divider: volume knobs, set-points, calibration trims.
  Wire only 2 terminals (one end + wiper) and it becomes a 2-terminal
  variable resistor (a "rheostat") for adjusting current.
```

- **Linear taper** — resistance changes evenly with rotation (set-points, position sensing).
- **Logarithmic (audio) taper** — matches how ears perceive loudness, used for volume.
- **Trimmer** — a tiny screwdriver-adjusted pot set once during calibration.

A pot used for [position sensing](sensors.md) is just a divider whose ratio reports
the shaft angle.

## How It Works

Resistance arises from electrons colliding with the atoms of the material as they drift through. Each collision transfers kinetic energy from the electron to the atom, making it vibrate faster — that's heat. Materials with a tight, regular crystal lattice (like pure copper) have few collisions — low resistance. Materials with a disordered structure or many impurities have more collisions — high resistance.

```
Metal conductor (low R):            Resistive material (high R):
  Cu atoms spaced regularly           disordered lattice
  ● ● ● ● ● ● ●                       ●  ● ●  ●   ●
  e→  →  →  →  →  (few bumps)         e→ * → * → * (many bumps, → slower)
```

## Pitfalls

- **Exceeding power rating** — resistors have a maximum power they can handle (typically 0.25 W for small through-hole parts). Exceeding it causes overheating or fire. Always check P = I²R.
- **Using the wrong resistor for an LED** — LEDs need a series resistor to limit current. Without one, the LED draws as much current as the supply can give and burns out instantly.
- **Mixing up series and parallel formulas** — series total *adds*; parallel total uses the reciprocal formula. Easy to swap by accident.
- **Forgetting resistor tolerance** — a "10 kΩ ±5%" part could be anywhere from 9.5 kΩ to 10.5 kΩ. For precision circuits, use 1% tolerance (four-band or five-band) resistors.

## Where this connects

- [Voltage](voltage.md) — V = IR requires understanding voltage first
- [Power & Energy](power.md) — P = I²R and P = V²/R give heat dissipated in a resistor
- [Circuits](circuits.md) — series/parallel analysis is extended to whole circuits
- [Capacitors](capacitors.md) — RC circuits (resistor + capacitor) set timing constants
- [Filters](filters.md) — RC networks are the simplest filters
- [GPIO](../embedded/gpio.md) — pull-up and pull-down resistors set logic levels on digital pins
- [Sensors & Transducers](sensors.md) — potentiometers and resistive sensors are read with voltage dividers
