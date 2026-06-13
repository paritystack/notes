# 28 · PCB layout — route, DRC, export Gerbers

## Overview

Your [schematic](27_kicad_schematic_capstone.md) says *what* connects; now you decide *where*
everything physically goes and draw the copper that connects it. In [KiCad's PCB
editor](../electronics/kicad_pcb.md) you'll place footprints, route traces, pour a ground
plane, pass the **design rule check (DRC)**, and export **Gerber** files — the universal
format a fab uses to make your board. This is the formalised version of the layout thinking
you did on [perfboard](26_perfboard_build.md), now with the tool checking your work.

```
   Schematic (logical)  ──►  PCB layout (physical)  ──►  Gerbers (manufacturing)

   ratsnest (thin lines = unrouted) → traces (copper) → DRC clean → export
```

## What you'll need

The ERC-clean [schematic with footprints](27_kicad_schematic_capstone.md) from rung 27, and
KiCad's PCB editor. No hardware — still CAD.

## The build

1. **Import the netlist** (update PCB from schematic). All footprints appear with a
   **ratsnest** — thin lines showing every connection you must route.
2. **Place footprints** sensibly: the [ATmega328](../embedded/avr.md) central, decoupling
   [caps](27_kicad_schematic_capstone.md) hard against its supply pins, the
   [LDO](21_ldo_regulator.md)/power section at the input edge, connectors at the board edge,
   the [I²C](17_i2c_sensor.md) parts near the MCU's SDA/SCL pins.
3. **Set design rules** to your fab's limits (e.g. JLCPCB: 6 mil trace/space, 0.3 mm holes).
   Size **power traces wider** than signals.
4. **Route** the traces (start with power and ground), keep them short, avoid sharp detours.
   **Pour a ground plane** on the bottom layer to give a low-impedance return — a
   [signal-integrity](../embedded/signal_integrity.md) best practice.
5. **Run DRC** and fix every violation: clearance errors, unrouted nets, overlaps.
6. **Export Gerbers + drill files** (and a BOM/position file if you want assembly). Use KiCad's
   Gerber viewer to sanity-check the layers before sending.

```
   Routing priorities:
     1. Solid ground (plane/pour) — the most important net
     2. Power — wide traces, short paths, decoupling close to pins
     3. Signals — keep I2C short; everything else is forgiving at these speeds
```

## It works when…

- [ ] Every ratsnest line is routed (zero unrouted nets in DRC).
- [ ] DRC passes against your fab's rules with no errors.
- [ ] The Gerber viewer shows correct copper, silkscreen, soldermask, and drill layers.

## What's happening

A PCB layout maps the logical [schematic](27_kicad_schematic_capstone.md) onto real copper
geometry. Placement comes first because it determines how easy routing is — parts that talk to
each other should sit near each other, and decoupling [caps](../electronics/capacitors.md) must
be right at the chip's pins to do their job. A **ground plane/pour** gives currents a short,
low-impedance return path, cutting noise — increasingly important as speeds rise
([signal integrity](../embedded/signal_integrity.md)). **DRC** is the manufacturability
compiler: it enforces the fab's minimum trace width, spacing, and hole sizes so the board can
actually be made. **Gerbers** are the photographic per-layer description every fab understands —
the deliverable you'll [order](29_order_assemble.md) next. Full detail in
[KiCad PCB](../electronics/kicad_pcb.md).

## Pitfalls

- **Decoupling caps placed far from the chip** — defeats their purpose; keep each within a few mm of the supply pin it serves.
- **Power traces too thin** — sized like signal traces, they drop voltage and heat up. Widen them for current ([rung 20](20_read_a_datasheet.md) thinking).
- **No ground plane** — a star of thin ground traces is noisy and hard to route; pour a plane.
- **DRC rules not matching the fab** — designing to 4 mil when your fab does 6 mil means a board they can't make (or charge extra for). Set rules to the fab first.
- **Exporting the wrong layers** — missing the drill file or a copper layer yields an unbuildable order. Verify in the Gerber viewer.

## Where this connects

- [KiCad PCB](../electronics/kicad_pcb.md) — placement, routing, planes, DRC, Gerber export
- [Signal Integrity](../embedded/signal_integrity.md) — why ground planes and short returns matter
- [27 · KiCad schematic](27_kicad_schematic_capstone.md) — the source this layout realises
- **Previous:** [27 · KiCad schematic](27_kicad_schematic_capstone.md) · **Next:** [29 · Order & assemble](29_order_assemble.md)
