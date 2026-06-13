# Reading a Datasheet

## Overview

A datasheet is the manufacturer's contract for a component: every voltage,
current, timing, and temperature it promises to honour, and every limit beyond
which it makes no promises at all. Learning to read one is the skill that turns
the theory in every other page into a working, reliable circuit — it tells you
which [resistor](resistance.md) value to pick, whether a [MOSFET](transistors_mosfet.md)
will switch from your logic level, how much current a [regulator](power_supplies.md)
can deliver, and which pin is which. This page is the bridge from
[prototyping](prototyping.md) to confident part selection. You rarely read one
cover to cover; you learn where each answer lives and jump to it.

```
  Anatomy of a datasheet (typical order):

  ┌────────────────────────────────────────────┐
  │ 1. Title + features  "what it is, headline" │
  │ 2. Pinout / package   "which pin is which"  │
  │ 3. Absolute Maximum   "what destroys it"    │
  │ 4. Recommended Op.    "where to actually run"│
  │ 5. Electrical Char.   "the guaranteed specs"│
  │ 6. Typical curves     "behaviour vs temp/I" │
  │ 7. Application info   "reference circuits"   │
  └────────────────────────────────────────────┘
```

## Key Concepts

### Absolute Maximum vs Recommended Operating

The most important distinction in the whole document — and the most commonly
confused:

```
  Absolute Maximum Ratings     Recommended Operating Conditions
  ────────────────────────     ────────────────────────────────
  "Exceed these and the part   "Stay inside these and every other
   may be permanently damaged"  spec in the datasheet is guaranteed"

  Example (a 3.3 V MCU):
    Abs-max VDD     = 4.6 V     Recommended VDD = 3.0 – 3.6 V
    Abs-max on pin  = VDD+0.3   Recommended in  = 0 – VDD

  Running AT the abs-max is not "fine" — it is the edge of destruction.
  Design to the recommended numbers, with margin.
```

### Typical vs Min/Max

Specs come in three columns. **Min** and **Max** are *guaranteed* across the rated
temperature range and from part to part — design to these. **Typical** is what one
nominal part does at 25 °C — useful for intuition, never for worst-case design.

```
  Parameter          Min   Typ   Max   Unit
  ───────────────────────────────────────────
  Forward voltage    —     2.0   2.4   V     ← design for 2.4 V worst case
  Quiescent current  —     1.2   5.0   µA    ← budget battery life at 5 µA
  Output current     1     —     —     A     ← guaranteed to supply ≥ 1 A
```

A spec with a dash in Min/Max is **not guaranteed** — it is only a typical hint.

### Pinout and Package

The pinout maps logical pins to physical positions; the package tells you the
footprint, size, and how to solder it.

```
  Pin 1 marker (dot or notch) — orient everything from here:

   ┌───●───┐         Package names encode size:
   │1     8│           SOT-23   small 3-pin SMD
   │2     7│           SOIC-8   8-pin SMD, 1.27 mm pitch
   │3     6│           TO-220   through-hole power tab
   │4     5│           QFN-32   leadless, pad underneath
   └───────┘           0603     2-terminal chip (R/C)

  Match the datasheet package to a PCB footprint — see [Circuit Design](circuit_design.md).
```

### Reading a Power Part: the Numbers That Bite

For anything that carries real current, four numbers decide whether your design
survives:

```
  MOSFET example (logic-level N-channel):
    V_DS(max)    30 V    ← must exceed your supply + spikes
    V_GS(th)     1–2 V   ← must be well below your gate drive (3.3 V ✓)
    R_DS(on)     11 mΩ @ V_GS=4.5 V  ← heat = I² × R_DS(on)
    I_D          20 A    ← but only with adequate cooling (read the footnote)

  The current rating almost always assumes a heatsink and a case temperature
  you will never actually reach — derate it.
```

### Conditions Are Everything

Every number is quoted *under conditions*. `R_DS(on) = 11 mΩ` is meaningless
without `@ V_GS = 4.5 V`. A regulator's `1 A` may be `@ V_in − V_out ≤ 2 V`. Always
read the condition in the same row — a spec out of its stated conditions is just a
number.

### A Quick Walk-Through (LM7805 regulator)

```
  Question                    Where to look           Answer
  ──────────────────────────────────────────────────────────────
  Will it output 5 V?         Electrical Char.        5 V ±4%
  How much current?           Recommended Op.         up to 1 A (with heatsink)
  Minimum input voltage?      Dropout spec            V_out + ~2 V → ≥ 7 V
  Will it overheat?           Thermal + P = (Vin−5)·I check vs θJA
  Which pin is ground?        Pinout (TO-220)         pin 2 (and the tab)
  Caps needed?                Application circuit      0.33 µF in, 0.1 µF out
```

The "Application Information" section usually hands you a reference circuit — copy
it; it exists because the manufacturer knows how the part misbehaves without it.

## Pitfalls

- **Designing to typical values** — the part you solder may be the worst-case one.
  Budget current, timing, and battery life against Min/Max, not Typ.
- **Treating absolute-maximum as an operating point** — it is the failure
  boundary. Leave headroom (e.g. run a 30 V-rated part at ≤ 20 V).
- **Ignoring the conditions column** — R_DS(on), gain, and dropout all depend on
  voltage, current, and temperature. A spec without its condition is useless.
- **Skipping the application circuit** — the recommended input/output capacitors
  and layout notes are not optional garnish; omitting them causes instability.
- **Wrong package variant** — the same part number ships in TO-220, SOT-223, and
  D2PAK. Order and lay out the exact suffix you intend to use.

## Where this connects

- [Prototyping & Test Equipment](prototyping.md) — measure the real part against its datasheet claims
- [Circuit Design](circuit_design.md) — the datasheet package maps to a PCB footprint
- [MOSFET Transistors](transistors_mosfet.md) — V_GS(th) and R_DS(on) decide gate drive and heat
- [Power Supplies](power_supplies.md) — dropout, current, and thermal limits come straight from the datasheet
- [Resistance & Ohm's Law](resistance.md) — power and tolerance ratings are datasheet numbers too
- [Schematic Symbol Reference](symbols.md) — the symbol you draw represents the part you just specified
```
