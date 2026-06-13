# KiCad PCB

## Overview

KiCad's PCB editor (pcbnew) is where the logical schematic becomes a physical board. You import the netlist from the schematic, assign positions to footprints, draw the board outline, route copper traces, fill a ground plane, run a Design Rules Check (DRC), and export Gerber files for fabrication. This page follows the USB-powered LED driver from [KiCad Schematic](kicad_schematic.md) through to ordering from JLCPCB. See [Circuit Design](circuit_design.md) for the underlying PCB concepts.

```
  PCB layout workflow:

  Update PCB from schematic
        │
        ▼
  Draw board outline ──▶ Place components ──▶ Route traces
                                                    │
                                                    ▼
                                          Add copper fills ──▶ DRC ──▶ Gerbers ──▶ Order
```

## Running Example

The LED driver: USB connector (J1), 330 Ω SMD resistor (R1 in 0603), and a 3 mm through-hole LED (D1). Target: a 30 × 20 mm 2-layer board.

```
  Target board (top view):

  ┌──────────────────────────────┐
  │                              │
  │   J1 (USB)    R1   D1 (LED) │
  │   [======]   [=]   ( )      │
  │                              │
  └──────────────────────────────┘
  30 mm × 20 mm, F.Cu signals, B.Cu GND plane
```

## Workflow

### 1. Open the PCB Editor

From the KiCad Project Manager, click the PCB editor icon, or open `project.kicad_pcb` directly.

### 2. Import Schematic (Update PCB from Schematic)

Tools → Update PCB from Schematic (`F8`). KiCad places all footprints in a cluster off-board with ratsnest lines showing unrouted connections.

```
  After import — ratsnest view:

  [J1]───────[R1]───[D1]
   └───────────────────┘ GND
   (dashed lines = unrouted nets)
```

If a footprint is missing, the error message names the component — go back to the schematic and assign the footprint, then re-run `F8`.

### 3. Draw Board Outline

Select layer `Edge.Cuts` in the layer dropdown (right side panel or the layer selector at top).

Place → Rectangle (or press `R` then draw), or Place → Line for irregular shapes. Draw a closed shape — the fab router cuts exactly along this line.

For the example: Draw a 30 × 20 mm rectangle. Verify it is closed (zoom in on corners).

```
  Board outline on Edge.Cuts:

  ┌──────────────────────────────┐  ← closed rectangle, yellow line
  │                              │
  │                              │
  └──────────────────────────────┘
```

### 4. Place Components

Drag footprints inside the board outline. Press `G` to move while keeping ratsnest connected, `R` to rotate.

Placement strategy:
- Keep components close to the connections they serve — short ratsnest lines mean short traces
- Orient connectors at board edges (J1 at the left edge so the USB plug overhangs)
- Group power components together (not relevant here, but important in larger designs)
- Verify component orientation: LED cathode (shorter leg) toward GND; check the footprint silkscreen for polarity markers

```
  After placement:

  ┌──────────────────────────────┐
  │                              │
  │  J1         R1       D1     │
  │  [USB]     [0603]    ( )    │
  │                              │
  └──────────────────────────────┘
  Ratsnest still shows dashed lines — not routed yet
```

### 5. Route Traces

Press `X` to start the interactive router. Click the start pad, route, click the end pad.

- Select the layer in the toolbar before routing (F.Cu for this example)
- Press `/` to switch layers mid-route (inserts a via automatically)
- Press `W` during routing to change trace width
- Press `Esc` to abandon an in-progress trace

Route the LED driver in this order:

```
  1. VBUS net:   J1.VBUS ──── R1.pad1   (F.Cu, 0.5 mm — carries USB 5V)
  2. LED_A net:  R1.pad2 ──── D1.anode  (F.Cu, 0.2 mm — signal current, ~9 mA)
  3. GND net:    J1.GND  ──── D1.cathode (F.Cu, or leave for ground plane)
```

Ratsnest lines disappear as each net is routed. The board is fully routed when no dashed lines remain.

### 6. Add Copper Fill (Ground Plane)

A copper fill on B.Cu connected to GND acts as a ground plane — it automatically connects every GND pad and reduces return-path impedance.

Place → Zone (`Ctrl+Shift+Z`):
1. Select layer `B.Cu`
2. Select net `GND`
3. Click around the entire board outline to define the fill area
4. Press `B` to fill all zones

```
  B.Cu after fill:

  ████████████████████████████████  ← copper everywhere
  █  J1.GND ██████ R1.GND ███████  ← GND pads connected automatically
  ████████████████████████████████
  ████████████████████████████████
```

Any GND pad now has a short thermal-relief connection directly to the plane.

### 7. Add Silkscreen Labels

On layer `F.SilkS`, add text for board identification (name, version, date). The component reference designators (R1, D1, J1) are already placed by the footprint — move them to avoid overlapping pads or each other.

Verify silkscreen does not overlap any pad (DRC will warn if it does).

### 8. Run DRC

Inspect → Design Rules Checker → Run DRC.

Fix all errors before exporting. Common DRC errors:

| Error | Cause | Fix |
|-------|-------|-----|
| Clearance violation | Two copper elements too close | Re-route or increase spacing |
| Unrouted net | A ratsnest line not converted to a trace | Route the missing connection |
| Silkscreen on pad | Silk overlaps a pad | Move the silk text |
| Board outline not closed | Gap in Edge.Cuts | Zoom in, close the gap |
| Annular ring too small | Via drill almost as large as pad | Use a larger pad or smaller drill |

Re-fill zones (`B`) after any edits — DRC checks the filled state.

### 9. Export Gerber Files

File → Fabrication Outputs → Gerbers.

Settings for JLCPCB (2025):
- **Layers**: F.Cu, B.Cu, F.SilkS, B.SilkS, F.Mask, B.Mask, Edge.Cuts
- **Format**: Gerber X2 (or RS-274X)
- **Precision**: 6 decimal places
- Enable "Use Protel filename extensions" for JLCPCB compatibility

Also export the drill file: Fabrication Outputs → Drill Files → Generate Drill File (Excellon format, metric).

```
  Output files:

  project-F_Cu.gtl       ← front copper
  project-B_Cu.gbl       ← back copper
  project-F_SilkS.gto    ← front silkscreen
  project-B_SilkS.gbo    ← back silkscreen
  project-F_Mask.gts     ← front solder mask
  project-B_Mask.gbs     ← back solder mask
  project-Edge_Cuts.gm1  ← board outline
  project.drl            ← drill file
```

Zip all files into one archive.

### 10. Order from JLCPCB

1. Go to jlcpcb.com → Quote Now
2. Upload the zip file — the Gerber viewer shows a 3D preview; verify it looks correct
3. Standard options for a basic 2-layer board:
   - Layers: 2
   - Base material: FR4
   - Thickness: 1.6 mm
   - Surface finish: HASL (cheapest) or ENIG (flatter pads, better for fine-pitch SMD)
   - Copper weight: 1 oz
   - Min track/spacing: 0.2 mm/0.2 mm
4. Add to cart; minimum order is 5 boards; typical cost $2–5 for a small board + shipping
5. Lead time: ~24 h production + 5–7 day shipping (DHL option available for ~1 week total)

## Key Shortcuts

| Key | Action |
|-----|--------|
| `X` | Route track |
| `G` | Grab (move, keeping connections) |
| `R` | Rotate |
| `E` | Edit properties |
| `/` | Switch layer during routing (inserts via) |
| `W` | Change track width during routing |
| `B` | Fill all copper zones |
| `U` | Select connected track |
| `D` | Interactive drag (re-routes around obstacles) |
| `Del` | Delete |
| `Ctrl+Z` | Undo |
| `F8` | Update PCB from schematic |

## Pitfalls

- **Re-filling zones after edits** — copper fills are not live. Press `B` again after any routing change; otherwise DRC checks stale copper.
- **Not verifying Gerbers** — always open the exported Gerbers in KiCad's Gerber viewer (or gerbv) before ordering. A missing layer is a wasted order.
- **Polarity-sensitive components** — D1 and electrolytic capacitors must match the footprint polarity markings. Check the silkscreen and footprint documentation.
- **SMD on both sides without reflow oven** — if you're hand-soldering, keep all SMD on F.Cu. SMD on B.Cu requires two reflow passes or careful hand-soldering while holding the top-side components.
- **Ignoring thermal relief** — a pad inside a ground plane without thermal relief is hard to solder by hand (the plane wicks heat away). KiCad adds thermal reliefs by default; do not disable them for hand-soldered boards.
- **Ordering before reviewing Gerbers** — the board house will make exactly what you send. A missing Edge.Cuts means no board; a mirrored silkscreen means unreadable labels. Review first.

## Where this connects

- [KiCad Schematic](kicad_schematic.md) — the schematic is the input to this workflow
- [Circuit Design](circuit_design.md) — trace width rules, layer conventions, design rules
- [Prototyping & Test Equipment](prototyping.md) — after the board arrives, use a multimeter for continuity and an oscilloscope to verify signals
- [Power Supplies](power_supplies.md) — power plane design (decoupling cap placement) maps directly to what is done in steps 5–6
- [Signal Integrity](../embedded/signal_integrity.md) — controlled impedance, differential pairs, and via stitching become relevant for high-speed designs
