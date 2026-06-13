# Circuit Design

## Overview

Circuit design turns a working breadboard idea into a permanent, manufacturable PCB. It has two distinct stages: **schematic capture** (drawing what connects to what, tool-agnostically) and **PCB layout** (deciding where every component sits and how copper traces connect them physically). This page covers the concepts that apply regardless of which EDA tool you use. [KiCad Schematic](kicad_schematic.md) and [KiCad PCB](kicad_pcb.md) cover the hands-on workflow.

```
  The two-stage pipeline:

  Idea on breadboard
        │
        ▼
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Schematic  │────▶│  PCB Layout │────▶│  Gerber     │
  │  (logical)  │     │  (physical) │     │  (fab files)│
  └─────────────┘     └─────────────┘     └─────────────┘
   "what connects"     "where it sits"     "send to fab"
```

## Schematics

A schematic is a logical map of a circuit — it shows connectivity, not physical position.

### Reference Designators

Every component gets a unique tag: `R` for resistor, `C` for capacitor, `U` for IC, `D` for diode, `Q` for transistor, `J` for connector, `L` for inductor. Numbers start at 1 and increment: `R1`, `R2`, `C1`, `U1`.

```
  Component annotation:

  R1         C3         U2         D1
  330Ω       100nF      ATmega328  LED
  resistor   bypass cap microcontroller  indicator
```

### Nets and Net Labels

A net is a named electrical node — every pin connected to the same net is electrically equivalent. Two ways to show connectivity:

```
  Direct wire:                 Net label:

  VCC ───┬─── R1              VCC ──── R1
         │                             │
         C1                          SENS ◄── label
         │                             │
        GND                    Q1 ─── SENS ◄── same net, no wire drawn

  Labels are cleaner for nets that cross a schematic.
```

### Power Symbols

Power symbols (VCC, GND, +3V3, +5V) are shorthand net labels — every symbol with the same name is connected without drawing wires.

```
  +5V      +3V3      GND
   │         │        │
   ▼         ▼       ─┴─  (flat line)
```

Common rail names: `VCC`, `VDD`, `+5V`, `+3V3`, `+1V8`, `GND`, `AGND` (analogue ground).

### Schematic Conventions

- Components flow **left to right, top to bottom** — inputs on the left, outputs on the right
- Power rails at **top** (positive), ground at **bottom**
- Crossing wires without a dot are **not** connected; a dot means junction
- Every net must have exactly **one driver** (a source or output pin); multiple drivers short together

## PCB Concepts

### Layers

A PCB is a stack of layers. A simple 2-layer board has:

```
  Layer stack (2-layer):

  ┌───────────────────────────────────┐  ← F.SilkS   (component labels, top)
  ├───────────────────────────────────┤  ← F.Mask     (solder mask opening, top)
  ├═══════════════════════════════════╡  ← F.Cu       (front copper traces)
  │         FR4 substrate (1.6 mm)    │
  ├═══════════════════════════════════╡  ← B.Cu       (back copper traces)
  ├───────────────────────────────────┤  ← B.Mask     (solder mask opening, back)
  └───────────────────────────────────┘  ← B.SilkS   (labels, back)

  Edge.Cuts layer defines the board outline (what the router cuts).
```

### Symbols vs Footprints

The schematic uses **symbols** (logical representations). The PCB uses **footprints** (physical pad patterns matched to the actual component package).

```
  Symbol (schematic)     Footprint (PCB)

       R1                  ○ pad 1
      ─┤├─                 [   body   ]
                           ○ pad 2

  Same R1, two representations. Must be linked before layout.
```

Common package types: `0402`, `0603`, `0805` (SMD resistors/caps by size in hundredths of an inch); `SOT-23`, `SOIC-8`, `QFP-32` (IC packages); `THT` (through-hole, legs go through the board).

### Traces, Vias, and Pads

```
  Trace: copper wire on one layer
  ══════════════════════

  Via: hole connecting layers (drilled, plated)
  F.Cu ══════╗
             ║ ← via
  B.Cu ══════╝

  Pad: exposed copper for soldering a component pin
  ┌────┐
  │ P1 │ ← SMD pad (no hole)
  └────┘
  ╔════╗
  ║ P1 ║ ← THT pad (hole through board)
  ╚════╝
```

### Design Rules

Fab houses publish minimum constraints. JLCPCB standard rules (2-layer, as of 2025):

| Parameter | JLCPCB minimum | Safe default |
|-----------|---------------|--------------|
| Trace width | 0.1 mm | 0.2 mm |
| Trace clearance | 0.1 mm | 0.2 mm |
| Via drill | 0.3 mm | 0.4 mm |
| Via annular ring | 0.13 mm | 0.2 mm |

**Trace width for current** — a rough rule for 1 oz copper (35 µm):

| Current | Min trace width |
|---------|----------------|
| 0.5 A | 0.5 mm |
| 1 A | 1.0 mm |
| 2 A | 2.0 mm |
| > 3 A | use a trace width calculator |

Signal traces (GPIO, SPI, I2C) carry milliamps and 0.2 mm is fine. Power traces carrying amps need to be wider.

### Ground Planes

A ground plane is a copper pour covering most of one layer, connected to GND. It reduces impedance, provides a return path for every signal, and improves EMI behaviour.

```
  Without ground plane:       With ground plane:

  VCC ─────── R1 ──── U1      VCC ─────── R1 ──── U1
  GND ─────────────── U1      ████████████████████████ ← GND fills rest of layer
                               every component's GND pin drops straight down to it
```

Always use a ground plane on B.Cu of a 2-layer board — route signals on F.Cu, GND fills B.Cu.

## Pitfalls

- **Floating nets** — a net with no driver (or a pin left unconnected when it needs a pull-up/down) causes unpredictable behaviour. Run ERC before layout.
- **Missing decoupling caps** — every IC power pin needs a 100 nF ceramic cap as close as possible to the pin, on the same layer, with short traces to VCC and GND.
- **Symbol/footprint mismatch** — a 0603 symbol linked to an 0805 footprint means pads in the wrong place. Verify footprint dimensions against the component datasheet.
- **Thin power traces** — routing a 1 A supply trace at 0.2 mm will cause voltage drop and heat. Check with a trace width calculator.
- **Silkscreen over pads** — silk on a pad blocks solder and causes opens. Keep silkscreen clear of pads.
- **No board outline** — if Edge.Cuts is empty, the fab can't cut the board. Always draw a closed outline.

## Where this connects

- [Prototyping & Test Equipment](prototyping.md) — breadboarding and measurement come before committing to a PCB
- [KiCad Schematic](kicad_schematic.md) — hands-on walkthrough of schematic capture in KiCad
- [KiCad PCB](kicad_pcb.md) — hands-on walkthrough of PCB layout, Gerber export, and ordering
- [Power Supplies](power_supplies.md) — power rail design (decoupling, regulation) maps directly to PCB power plane decisions
- [Signal Integrity](../embedded/signal_integrity.md) — high-speed traces require controlled impedance, which starts with PCB stackup choices
