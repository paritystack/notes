# Prototyping & Test Equipment

## Overview

Building and measuring real circuits requires hands-on tools. The breadboard lets you wire up circuits without soldering — perfect for testing ideas quickly. Soldering makes connections permanent on a PCB. The multimeter measures voltage, current, and resistance to verify what you've built actually works. An oscilloscope lets you *see* signals in time — invaluable once your circuits go beyond DC. This page is the practical bridge from the theory in every other page to real hardware.

```
The prototyping toolkit:

  ┌─────────────────┐   ┌──────────────┐   ┌───────────────┐
  │   BREADBOARD    │   │  MULTIMETER  │   │  OSCILLOSCOPE │
  │                 │   │              │   │               │
  │  ○○○○○○○○○○○○  │   │  ┌──────┐    │   │   ╱╲    ╱╲   │
  │  ○○○○○○○○○○○○  │   │  │ 3.31 │    │   │  ╱  ╲  ╱  ╲  │
  │  ○○○○○○○○○○○○  │   │  └──────┘    │   │ ╱    ╲╱    ╲ │
  │  ○○○○○○○○○○○○  │   │    V Ω mA    │   │               │
  └─────────────────┘   └──────────────┘   └───────────────┘
   No solder, fast       Verify DC values    Visualise waveforms
```

## Key Concepts

### Breadboard

A breadboard is a reusable board with holes connected internally — no soldering needed.

```
  Breadboard internal connections:

  Top rails (power rails — horizontal, full length):
  ════════════════════════════   ← + rail (red)
  ════════════════════════════   ← − rail (blue/black)

  Main area (vertical strips of 5):
  a b c d e   f g h i j
  ○ ○ ○ ○ ○   ○ ○ ○ ○ ○   ← row 1, connected: a1-b1-c1-d1-e1 and f1-g1-h1-i1-j1
  ○ ○ ○ ○ ○   ○ ○ ○ ○ ○   ← row 2, separate from row 1
  ...

  Middle gap (between e and f columns) is NOT connected.
  ICs straddle this gap so each pin has 4 holes to connect to.
```

```
  Wiring a simple LED circuit:

  +5V ─── red rail ─── [R 330Ω] ─── row 5 ─── [LED anode, row 5]
                                               [LED cathode, row 6]
  GND ─── black rail ──────────────────────── row 6

  (LED straddles rows 5 and 6, resistor connects rail to row 5)
```

**Tips:**
- Use short, colour-coded jumper wires (red = power, black = GND, other colours for signals)
- Don't crowd too many wires — it's hard to debug a messy breadboard
- Breadboards don't work well above ~1 MHz due to parasitic capacitance between holes

### Soldering Basics

Soldering permanently joins component leads to copper pads on a PCB using a low-melting-point metal alloy (solder).

```
  Good solder joint:

     ╲ component lead
      ╲
       ○ ← shiny, cone-shaped, smooth surface
      / ← copper pad
     /

  Bad joints:

  Cold joint: dull, grainy, crumbly → joint was moved while solder solidified
  Blob: too much solder → may bridge to adjacent pad (short circuit)
  No wetting: solder balled up, didn't flow → pad/lead not hot enough
```

**How to solder:**
1. Heat the joint (pad + lead together) with the iron for 2–3 seconds
2. Apply solder to the joint (not the iron) — it flows where things are hot
3. Remove solder, then iron
4. Let cool without moving (3–5 seconds)
5. Inspect: should be shiny and cone-shaped

**Tools:** soldering iron (60 W adjustable, 350 °C), flux-core solder (60/40 or lead-free), solder wick (for removing solder), flux pen.

### Multimeter

The multimeter measures three fundamental quantities. Always start with the correct mode.

#### Measuring Voltage (Voltmeter)

```
  Mode: V (DC) or V~ (AC)
  Connect: RED probe to point to measure, BLACK to reference (GND)
  Safe range: start high (200V range), reduce as needed

  ⚠ Never measure voltage with probes in mA/current sockets
```

#### Measuring Current (Ammeter)

```
  Mode: mA or A
  Connect: BREAK the circuit, insert meter in series
  RED probe where current enters, BLACK where it exits

          ┌──[METER]──┐
  + ──────┘           └────── [load] ──── −

  ⚠ Never connect in parallel (across a voltage source) → will short the supply
  ⚠ Most meters have a fuse that blows at 200 mA or 10 A — check your meter
```

#### Measuring Resistance (Ohmmeter)

```
  Mode: Ω
  Probe the component with power OFF and capacitors discharged
  Touch probes to both ends of the component

  ⚠ Never measure resistance in a live circuit → damages the meter
```

#### Continuity Test

```
  Mode: continuity (beep symbol or diode symbol)
  Beeps when resistance < ~30 Ω

  Use for:
  - Verifying a wire or trace is connected (expected beep)
  - Checking for short circuits (unexpected beep)
  - Confirming a switch opens and closes
```

#### Diode Test

```
  Mode: diode symbol
  Forward bias: RED to anode, BLACK to cathode → shows forward voltage (~0.7V)
  Reverse bias: RED to cathode, BLACK to anode → shows "OL" (open)
  Useful to identify anode/cathode if markings are unclear.
```

### Oscilloscope

A multimeter shows a *number*. An oscilloscope shows a *graph* of voltage vs time — essential for any signal that changes (PWM, UART, SPI, audio, anything AC).

```
  Screen layout:

    Volts/div  Time/div
    ↓            ↓
  │ 2V │────────────────────────────────│
  │    │     ___     ___     ___        │  ← waveform
  │    │    │   │   │   │   │   │       │
  │ 0V │────┘   └───┘   └───┘   └───── │
  │    │                                │
  │    │← one time division (e.g. 1ms) →│

  Key controls:
  Volts/div:  vertical scale (amplitude)
  Time/div:   horizontal scale (how fast)
  Trigger:    what makes the trace stable (usually rising edge of signal)
  Probe ×1/×10: ×10 probe reduces input capacitance but divides voltage by 10
```

**Entry-level options:**
- Rigol DS1054Z (~$350) — 4 channels, 50 MHz, widely recommended for beginners
- Fnirsi 1014D (~$80) — handheld, adequate for basic digital signals
- DSLogic Plus — USB logic analyser, for decoding UART/SPI/I2C without analog view

### Common Test Points to Check

| Signal | What to measure | Tool |
|--------|----------------|------|
| Power rail stable? | Voltage, ripple | Multimeter, oscilloscope |
| LED getting current? | Voltage across resistor ÷ R | Multimeter |
| GPIO logic level correct? | Voltage at pin | Multimeter |
| PWM signal correct? | Frequency, duty cycle | Oscilloscope |
| I2C/SPI working? | Signal shape and timing | Oscilloscope or logic analyser |
| Short circuit? | Continuity (off), current (on) | Multimeter |

## Pitfalls

- **Measuring current without breaking the circuit** — the meter must be in series. Placing it in parallel short-circuits the supply and blows the fuse in the meter (or worse).
- **Wrong range** — starting on a low range (e.g., 200 mV) to measure 12 V will give an error reading. Always start high and work down.
- **Soldering too long** — overheating a joint lifts the pad off the PCB (pad lifts). Keep heat application to 3–5 seconds.
- **Breadboard stray capacitance at high frequency** — above ~1 MHz, the capacitance between adjacent tracks distorts signals. Test RF or high-speed circuits on a proper PCB.
- **Assuming digital oscilloscope always triggers correctly** — if the waveform looks messy or scrolls, adjust the trigger level to the midpoint of the signal.

## Where this connects

- [Charge & Current](charge_current.md) — measuring current requires understanding what you're measuring and where in the circuit to do it
- [Voltage](voltage.md) — every measurement reference point is a voltage relative to GND
- [Resistance & Ohm's Law](resistance.md) — resistance measurement verifies component values
- [Circuits](circuits.md) — schematic reading is prerequisite to knowing where to probe
- [Filters](filters.md) — oscilloscope verifies filter cutoff frequency in practice
- [Power Supplies](power_supplies.md) — oscilloscope reveals switching ripple and load transients
- [Reading a Datasheet](datasheets.md) — verify a real part against its specified ratings
