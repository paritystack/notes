# Schematic Symbol Reference

## Overview

Every component in this section has a schematic symbol — a small standardised
drawing that says "this part connects here" without showing what it physically
looks like. The symbols are scattered across the individual component pages; this
page collects them into one quick lookup, the way a [datasheet](datasheets.md)
collects a part's numbers. Use it when reading a schematic in
[Circuit Design](circuit_design.md) or placing parts in
[KiCad](kicad_schematic.md) and you meet a symbol you don't recognise. Each entry
links to the page that explains the component in full.

```
  A schematic shows connectivity, not layout:

   +5V ──[R1]──┬──►|── GND        ← reads: 5 V through a resistor,
               │   D1  (LED)        then an LED, to ground.
              C1
               │
              GND
```

## Passive Components

| Component | Symbol | Notes |
|-----------|--------|-------|
| [Resistor](resistance.md) | `──[/\/\/\]──` or `──▭──` | Zig-zag (US) or box (IEC); non-polarised |
| [Potentiometer](resistance.md) | `──▭──` with `↓` arrow on top | Third terminal is the wiper |
| [Capacitor (non-pol.)](capacitors.md) | `──┤├──` | Two straight plates; ceramic, film |
| [Capacitor (polarised)](capacitors.md) | `──┤(──` | Curved plate = `−`; electrolytic, tantalum |
| [Inductor](inductors.md) | `──(((((──` | Coil/loops; ferrite adds a bar over it |
| [Transformer](transformers.md) | `(((((│#│)))))` | Two coils + core bars between them |
| [Crystal](oscillators.md) | `──┤├──` in a box | Quartz resonator, fixed frequency |

```
  Polarised vs non-polarised capacitor:

   non-polarised      polarised (watch + / −)
     │   │              │   │
     ┤   ├              ┤   (        ( = negative side
     │   │              │   │
```

## Semiconductors

| Component | Symbol | Notes |
|-----------|--------|-------|
| [Diode](diodes.md) | `──►\|──` | Triangle points with conventional current; bar = cathode |
| [LED](diodes.md) | `──►\|──` with two arrows out | Emits light; needs series resistor |
| [Zener](diodes.md) | `──►\|─` with bent cathode bar | Conducts in reverse at V_Z |
| [Schottky](diodes.md) | `──►\|─` with S-shaped bar | Low forward drop, fast |
| [NPN BJT](transistors_bjt.md) | circle, arrow **out** of emitter | Base controls C→E current |
| [PNP BJT](transistors_bjt.md) | circle, arrow **into** emitter | Emitter at the higher potential |
| [N-MOSFET](transistors_mosfet.md) | gate bar + arrow toward channel | Voltage-controlled; arrow in |
| [P-MOSFET](transistors_mosfet.md) | gate bar + arrow away | Arrow out; high-side switch |

```
  BJT emitter arrow tells you the type:

     NPN              PNP
      C                C
      │                │
  B ──┤            B ──┤
      │↘ (out)         │↖ (in)
      E                E
```

## Active & Integrated

| Component | Symbol | Notes |
|-----------|--------|-------|
| [Op-amp / comparator](op_amps.md) | triangle, `+` and `−` inputs, apex output | Amplifies (V+ − V−) |
| [Logic gate](logic_gates.md) | D-shape (AND), shield (OR), `○` = invert | See the logic gates page |
| Generic IC | labelled rectangle with numbered pins | Pin 1 marked by a dot/notch |

## Connections, Sources & Reference

| Element | Symbol | Notes |
|---------|--------|-------|
| Wire junction | `──┬──` with a dot | Dot = connected; no dot at a cross = not connected |
| [Switch (SPST)](switches_relays.md) | `──o  o──` | Open contact; SPDT adds a second throw |
| [Relay](switches_relays.md) | coil box + separate contact | Control and load drawn apart |
| DC voltage source | `──┤│──` (long line +) | Battery: long line positive |
| Ground (GND) | `─┴─` then `─┬─` shrinking lines | The 0 V reference |
| Analogue ground | same, labelled `AGND` | Kept separate from digital GND |
| Power rail | `+5V`, `+3V3` flag/arrow up | Shorthand net label — same name = connected |

```
  Ground and a connection dot — the two you'll meet most:

   junction (connected)      ground (0 V reference)
       │                          │
   ────●────                     ─┴─
       │                         ─┴─   (shrinking lines)
```

## Pitfalls

- **US vs IEC resistor symbol** — a zig-zag and a plain box mean the same thing.
  KiCad libraries mix both; don't read the box as something exotic.
- **Crossing wires** — a crossover *with* a junction dot is connected; *without* a
  dot it is just two wires passing over. See [Circuit Design](circuit_design.md).
- **Capacitor polarity** — the curved/filled plate is negative on a polarised cap.
  Reversing an electrolytic can make it vent or burst.
- **Diode/BJT/MOSFET arrow direction** — the arrow encodes the type and the
  current direction. Mirroring a symbol while drawing silently changes its meaning.
- **Pin 1 orientation** — an IC symbol's pin order must match the footprint's pin 1
  marker, or the part is soldered in rotated.

## Where this connects

- [Circuit Design](circuit_design.md) — reference designators, nets, and conventions that use these symbols
- [KiCad Schematic](kicad_schematic.md) — placing these symbols from the library in eeschema
- [Reading a Datasheet](datasheets.md) — the symbol represents a real part with real ratings
- [Diodes](diodes.md), [BJT](transistors_bjt.md), [MOSFET](transistors_mosfet.md), [Op-Amps](op_amps.md) — each symbol's component explained in depth
- [Logic Gates](logic_gates.md) — the full set of gate shapes and truth tables
```
