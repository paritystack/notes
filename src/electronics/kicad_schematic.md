# KiCad Schematic

## Overview

KiCad's schematic editor (eeschema) is where you capture the logical design of your circuit before touching the PCB. You place symbols from the library, draw wires, assign net labels and power symbols, annotate reference designators, and run an Electrical Rules Check (ERC) to catch mistakes. This page walks through that workflow using a USB-powered LED driver as the running example. See [Circuit Design](circuit_design.md) for the underlying concepts and [KiCad PCB](kicad_pcb.md) for the layout stage.

```
  Schematic workflow:

  New project
      │
      ▼
  Place symbols ──▶ Wire nets ──▶ Add power symbols
                                        │
                                        ▼
                               Annotate (R1, D1…) ──▶ Assign footprints ──▶ ERC
                                                                               │
                                                                               ▼
                                                                        Export netlist
                                                                        (or update PCB)
```

## Running Example

A USB-powered LED driver: +5V from a USB connector drives a 330 Ω resistor into an LED, then to GND. Simple enough to fit on one schematic sheet; complex enough to show every workflow step.

```
  LED driver schematic:

  +5V
   │
  J1 (USB connector)
   │
   ├── VBUS ──── R1 (330Ω) ──── D1 (LED) ──── GND
   │
   └── GND ──────────────────────────────────── GND

  Net names: VBUS (J1 pin 1 to R1), LED_A (R1 to D1 anode)
  D1 cathode → GND symbol
```

## Workflow

### 1. Create a KiCad Project

File → New Project → choose a folder. KiCad creates `project.kicad_pro`, `project.kicad_sch`, and `project.kicad_pcb`.

Open the schematic: double-click `project.kicad_sch` in the Project Manager, or click the schematic editor icon.

### 2. Place Symbols

Press `A` (or Place → Add Symbol) to open the symbol chooser. Search by name or keyword.

| Component | Search term | Library |
|-----------|-------------|---------|
| USB Type-A connector | `USB_A` | `Connector` |
| Resistor | `R` | `Device` |
| LED | `LED` | `Device` |
| Generic GND | `GND` | `Power` |
| +5V rail | `+5V` | `Power` |

Click to place. Press `R` to rotate before placing.

```
  After placing symbols (unconnected):

  J1        R1          D1
  [USB]     [330Ω]      [LED]
  VBUS○     ○───○       ○───○
  GND○
```

### 3. Wire Nets

Press `W` to start a wire. Click the start pin, route, click the end pin. Press `Esc` to stop.

- Wires connect pins directly on the schematic page
- A **junction dot** (J) appears automatically where a wire meets a wire mid-segment — verify it's there at T-junctions
- Press `Esc` before clicking an empty space to avoid dangling wires

```
  Wired circuit:

  +5V ──── J1.VBUS ──── R1.1       R1.2 ──── D1.A       D1.K ──── GND
                │                                  │
               GND (power symbol)                 GND (power symbol)
```

### 4. Add Power Symbols

Press `P` (or Place → Add Power Port) to add `+5V`, `GND`, `+3V3`, etc. These are shorthand net labels — every `GND` symbol on the sheet is connected without drawing wires.

Place `+5V` above `J1.VBUS` and `GND` below `J1.GND` and below `D1.K`.

### 5. Add Net Labels

For nets that need a name but don't use a power symbol, press `L` (Place → Net Label). Type the name and place it on a wire end.

Net labels used in the example:
- `VBUS` on the wire from J1 pin 1 to R1 pin 1
- `LED_A` on the wire from R1 pin 2 to D1 anode

Two wires with the same label are electrically connected even if they don't touch — useful for keeping the schematic uncluttered.

### 6. Annotate Reference Designators

Tools → Annotate Schematic → Annotate. This assigns numbers to all placeholders (`R?` → `R1`, `D?` → `D1`).

Review the result — make sure no two components share the same reference.

### 7. Assign Footprints

Each symbol must be linked to a physical footprint before layout.

Tools → Assign Footprints (or click the footprint field on each component with `E`).

| Component | Footprint |
|-----------|-----------|
| R1 (330Ω) | `Resistor_SMD:R_0603_1608Metric` |
| D1 (LED) | `LED_THT:LED_D3.0mm` |
| J1 (USB-A) | `Connector_USB:USB_A_Molex_105057-0001` |

Pick SMD or THT to match the actual parts you have. The footprint name encodes the package dimensions.

### 8. Edit Component Properties

Double-click a symbol (or press `E` with it selected) to edit its properties:

- **Reference**: `R1`, `D1`, etc.
- **Value**: `330`, `RED`, `USB_A`
- **Footprint**: linked package (set in step 7)
- **Datasheet**: optional URL

### 9. Run ERC

Inspect → Electrical Rules Checker → Run. Fix all errors before moving to layout.

Common ERC errors:

| Error | Cause | Fix |
|-------|-------|-----|
| Pin unconnected | A pin has no wire | Connect it or add a "no-connect" flag (press `Q`) |
| Net has no driver | A net has no power source | Add a power symbol or mark an output pin |
| Duplicate reference | Two components share a ref | Re-annotate |
| Wire not connected | Dangling wire end | Delete or extend the wire |

"No-connect" flag (`Q`) explicitly marks unused pins as intentionally unconnected — silences ERC without hiding real problems.

### 10. Update PCB / Export Netlist

When ERC is clean:

- **KiCad 6+**: File → Export → Netlist (legacy), or directly: open the PCB editor and use Tools → Update PCB from Schematic (`F8`). This pushes all components and connections into pcbnew without a manual netlist file.

## Key Shortcuts

| Key | Action |
|-----|--------|
| `A` | Add symbol |
| `P` | Add power symbol |
| `W` | Draw wire |
| `L` | Add net label |
| `R` | Rotate |
| `E` | Edit properties |
| `Q` | Add no-connect flag |
| `G` | Grab (move keeping wires attached) |
| `Del` | Delete |
| `Ctrl+Z` | Undo |

## Pitfalls

- **Dangling wire ends** — a wire that ends in space rather than a pin creates a floating net. ERC catches this, but zoom in to verify manually before running ERC.
- **Wrong power net name** — `VCC` and `+5V` are different nets. Use consistent names across the whole schematic.
- **Skipping footprint assignment** — "Update PCB from Schematic" will fail or place blank outlines if footprints are missing.
- **Forgetting decoupling caps** — any IC (not in this example, but in real designs) needs a 100 nF cap on every VCC pin. Add them in the schematic; ERC won't remind you.
- **Junction missing at T-junction** — if two wires meet but there's no dot, they may not be connected. KiCad auto-adds junctions at T-junctions but not always at crosses. Verify.

## Where this connects

- [Circuit Design](circuit_design.md) — schematic conventions, net naming, reference designators
- [KiCad PCB](kicad_pcb.md) — the next stage: import the netlist, place footprints, route traces
- [Circuits](circuits.md) — series/parallel rules determine how you wire the LED driver
- [Diodes](diodes.md) — LED forward voltage and current calculations drive R1's value
- [Resistance & Ohm's Law](resistance.md) — R1 = (VCC − V_f) / I_f; the formula is here
- [Schematic Symbol Reference](symbols.md) — what each symbol you place from the library means
