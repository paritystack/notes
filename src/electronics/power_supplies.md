# Power Supplies

## Overview

A power supply converts one form of electrical power into another — typically turning noisy, varying input voltage into a clean, regulated output voltage. Everything else in a circuit depends on having a stable supply; if the supply voltage fluctuates, so does every measurement, logic threshold, and analog reference. Power supplies connect [diodes](diodes.md) (rectification), [capacitors](capacitors.md) (smoothing), [inductors](inductors.md) (energy storage in switching), [transistors](transistors_mosfet.md) (switching), and [op-amps](op_amps.md) (error amplification) into one system. The topic continues in depth at [Embedded Power Management](../embedded/power_management.md).

```
Dam and regulator valve analogy:

  River (mains AC / raw DC)
     │
  [DAM] ─── stores large head of water (big reservoir capacitor)
     │
  [VALVE] ─── regulator: opens/closes to keep output pressure steady
     │
  [LOAD] ─── gets steady, clean pressure regardless of river level
```

## Key Concepts

### The Goal: Regulation

A regulated supply keeps V_out constant even when:
- V_in varies (battery draining, mains fluctuating)
- Load current changes (microcontroller idle vs transmitting)

The measure of quality is **load regulation** (how much output changes with current) and **line regulation** (how much output changes with input).

### AC to DC Conversion (Rectification + Filtering)

Before any regulation, mains AC must become rough DC:

```
  Step 1: Transformer (steps down 230 V AC to 12 V AC)
  Step 2: Bridge rectifier (4 diodes, flips negative half-cycles to positive)
  Step 3: Bulk capacitor (smooths rectified pulses into rippling DC)

  230V AC → [Transformer] → 12V AC → [Bridge] → ~17V pulsing DC
           → [Cap 1000µF] → ~16V DC with small ripple
```

After these three steps, you have unregulated DC. Then apply a regulator.

### Linear Regulator

The simplest regulator. A transistor in series with the output, controlled by an error amplifier that compares V_out to a reference:

```
  Vin (12V) ──[pass transistor]── Vout (5V)
                    │
              [error amp] ← compares Vout to 3.3V reference
                    │
              [feedback resistors]

  If Vout rises: error amp reduces transistor conduction → Vout falls back
  If Vout falls: error amp increases transistor conduction → Vout rises
```

The excess voltage (12 − 5 = 7 V) is dropped across the transistor as heat:

```
  Power wasted = (Vin − Vout) × I_load
               = 7V × 1A = 7W  ← needs a large heatsink!

  Efficiency ≤ Vout / Vin = 5/12 = 42%
```

**LDO (Low Dropout) regulator** — same principle but works with Vin only slightly above Vout (as little as 100–300 mV dropout). Examples: AMS1117, LP2985, MCP1700.

Common fixed voltage LDOs:

| Part | Output | Max current |
|------|--------|------------|
| 7805 | 5.0 V | 1 A |
| LM3.3 / AMS1117-3.3 | 3.3 V | 1 A |
| MCP1700-3302 | 3.3 V | 250 mA |

**Use linear regulators when:** low noise matters (analog circuits, ADC supply), input-to-output voltage difference is small (< 2 V), current is low (< 500 mA).

### Switching Regulator (Buck / Boost)

Instead of wasting energy as heat, a switching regulator rapidly switches a transistor ON and OFF, storing and releasing energy in an inductor:

```
  Buck converter (step-down, e.g. 12V → 5V):

  Vin ──[MOSFET switch]──[Inductor]── Vout
              │                   │
         [MOSFET or diode]      [Cap]
              │                   │
             GND                 GND

  MOSFET switches at 100 kHz – 1 MHz:
  ON:  current builds in inductor, energy transferred to output
  OFF: inductor releases energy via diode, maintains output current
```

Efficiency: typically **85–95%** — much better than a linear regulator.

```
  Boost converter (step-up, e.g. 3.7V battery → 5V USB output):

  Vin ──[Inductor]──┬── Vout (higher)
                    │
                [MOSFET]   ← switches to ground
                    │
                   GND

  When MOSFET closes: inductor stores energy (current builds)
  When MOSFET opens:  inductor voltage spikes above Vin; diode conducts to Vout
```

**Use switching regulators when:** efficiency matters (battery-powered devices), large voltage difference between Vin and Vout, or you need to step voltage *up*.

### Buck vs Linear Comparison

| Feature | Linear | Buck/Boost |
|---------|--------|-----------|
| Efficiency | Low (Vout/Vin) | High (85–95%) |
| Noise | Very low | Higher (switching noise) |
| Complexity | Simple (3 pins) | More components |
| Heat | Yes (at high power) | Minimal |
| Cost | Very low | Slightly higher |
| Common ICs | AMS1117, LP2985 | MP2307, LM2596, TPS62x |

### Filtering the Output

All power supplies benefit from capacitors directly at the load:
- **Bulk cap** (10–100 µF electrolytic) — handles slow load transients
- **Decoupling cap** (100 nF ceramic, per IC) — filters high-frequency switching noise

See [Capacitors](capacitors.md) and [ADC](../embedded/adc.md) for why these are critical.

### Voltage Regulator ICs Pinout

```
  TO-220 package (7805 etc.):
   ___________
  |   7805    |
  | IN OUT ADJ|
     │   │   │
    Vin  Vout GND/ADJ

  SOT-23 (AMS1117):
   Pin 1: GND / ADJ
   Pin 2: Vout
   Pin 3: Vin
```

## Pitfalls

- **Input capacitor missing** — a switching regulator needs a ceramic cap (1–10 µF) close to its Vin pin. Without it, the switching spikes cause instability.
- **Exceeding power dissipation on a linear regulator** — a TO-92 (small plastic) linear regulator handling 2 W of dissipation will overheat and shut down. Always check (Vin − Vout) × I ≤ P_max (with derating).
- **Wrong inductor for a switcher** — the inductor must handle the peak current without saturating. Saturation causes a sudden loss of inductance, disrupting the control loop and potentially causing a loud "tick" or current spike.
- **Switching noise coupling into analog circuits** — the high-frequency edges of a switching supply radiate EMI. Keep switching components away from [ADC](../embedded/adc.md) inputs, RF circuits, and oscillator pins. If unavoidable, use an LC filter on the supply to that section.
- **Forgetting the enable pin** — many regulator ICs have an ENABLE pin that must be pulled HIGH (or LOW) to turn on. Leaving it floating means the supply may or may not work.

## Where this connects

- [Transformers](transformers.md) — step down mains and isolate the output before rectification
- [Diodes](diodes.md) — rectifier bridges, freewheeling diodes in buck/boost
- [Capacitors](capacitors.md) — bulk smoothing, output filtering, decoupling
- [Inductors](inductors.md) — energy storage element in every switching converter
- [MOSFET Transistors](transistors_mosfet.md) — the switching element in DC-DC converters
- [Op-Amps](op_amps.md) — the error amplifier inside linear regulators
- [Power Management](../embedded/power_management.md) — embedded system-level power budgeting, sleep modes
