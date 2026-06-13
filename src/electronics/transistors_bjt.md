# BJT Transistors

## Overview

A Bipolar Junction Transistor (BJT) is a three-terminal device where a small **base current** controls a large **collector current**. Think of it as a valve: a tiny trickle of water through a pilot tube opens a large main valve and lets a flood through. This current amplification property makes BJTs useful as amplifiers (audio, RF) and as switches (driving LEDs, motors, relays). The complementary voltage-controlled device is the [MOSFET](transistors_mosfet.md). Understanding [diodes](diodes.md) first is essential — a BJT is effectively two diodes back-to-back.

```
Pilot valve analogy:

  Small control pipe (Base)
          |
          ↓
   ┌──────────────┐
   │    valve     │  ← small current in = large current through
   └──────────────┘
          |
  Large main pipe (Collector → Emitter)

  NPN BJT:
     Collector (C)
          │
    B ────┤  NPN   (current into Base → current flows C to E)
          │
     Emitter (E)
```

## Key Concepts

### NPN vs PNP

| Type | Activates when | Current direction | Common use |
|------|---------------|-------------------|-----------|
| NPN | Base > Emitter (~0.7 V) | C → E | Low-side switch (between load and GND) |
| PNP | Emitter > Base (~0.7 V) | E → C | High-side switch (between supply and load) |

NPN is more common and easier to reason about. This page focuses on NPN.

```
  NPN symbol:          PNP symbol:
      C                    C
      │                    │
  B ──┤                B ──┤
      │                    │
      E (arrow out)        E (arrow in)

  Arrow points in direction of conventional current
  through the emitter (out for NPN, in for PNP)
```

### Current Gain (β or hFE)

The ratio of collector current to base current:

```
  β = IC / IB

  IC = collector current (large)
  IB = base current (small)
  IE = emitter current = IC + IB
```

Typical β values:
- Small signal BJTs: 100–500
- Power BJTs: 20–100
- Darlington pairs: 1000–100000

**Example:** β = 200, IB = 0.1 mA → IC = 200 × 0.1 = 20 mA

### Three Operating Regions

```
  ┌─────────────┬──────────────┬──────────────────────────┐
  │ Region      │ Condition    │ Behaviour                │
  ├─────────────┼──────────────┼──────────────────────────┤
  │ Cutoff      │ VBE < 0.6V   │ No current flows (OFF)   │
  │ Active      │ VBE ≈ 0.7V   │ IC = β × IB (amplifier) │
  │ Saturation  │ VBE ≥ 0.7V,  │ Fully ON, VCE ≈ 0.2V   │
  │             │ base driven  │ (saturated switch)       │
  │             │ hard         │                          │
  └─────────────┴──────────────┴──────────────────────────┘
```

For **switch mode** (on/off), operate in cutoff (off) or saturation (on).
For **amplifier mode**, operate in the active region.

### BJT as a Switch

The most common use in microcontroller circuits: using a logic signal (3.3 V or 5 V) to switch a load that draws more current than the GPIO pin can supply.

```
  Switching a 12 V / 200 mA motor with a 3.3 V GPIO:

       +12V
         │
       [Motor]
         │
         C ← NPN BJT
  GPIO──[Rb]──B
         E
         │
        GND

  GPIO HIGH (3.3V):
    VBE ≈ 0.7V, base current flows through Rb
    Rb = (3.3 − 0.7) / IB = 2.6 / IB
    IB needed = IC / β = 200mA / 100 = 2mA
    Rb = 2.6 / 0.002 = 1.3 kΩ → use 1 kΩ (slightly oversaturated is fine)
    Transistor saturates → motor gets current

  GPIO LOW (0V):
    VBE < 0.6V → cutoff → no IC → motor off

  Don't forget: flyback diode across the motor! (see [Inductors](inductors.md))
```

### BJT as an Amplifier

In the active region, small variations in base current produce large, proportional variations in collector current:

```
  Common-emitter amplifier:

        +Vcc
          │
         [RC]  ← collector resistor converts IC variation to voltage
          │
          C
  Vin ──[Rb]──B   BJT in active region
          E
          │
         [RE]  ← emitter resistor stabilises bias
          │
         GND

  Voltage gain ≈ −RC / RE   (negative = signal inverted)
```

## Pitfalls

- **Forgetting the base resistor** — connecting a GPIO directly to the base without a resistor means the GPIO must supply large base current. This damages the GPIO. Always put a resistor in series with the base.
- **Not saturating the switch hard enough** — if the base current is too small relative to IC, the transistor stays in active region: it partially conducts and dissipates a lot of heat instead of fully switching on. Use IB ≥ IC / (β/10) to ensure saturation.
- **Ignoring VCE(sat)** — a "fully on" BJT still has ~0.2 V across collector-emitter. For low-voltage circuits (3.3 V), this is a significant drop. MOSFETs have much lower ON voltage.
- **Thermal runaway** — in the active region, as the junction heats up, β increases, IC increases, more heat — a runaway cycle. Always use emitter resistors or feedback in amplifier designs.

## Where this connects

- [Diodes](diodes.md) — a BJT is two P-N junctions; VBE ≈ 0.7 V is the diode forward voltage
- [MOSFET Transistors](transistors_mosfet.md) — voltage-controlled alternative; preferred in most modern digital circuits
- [Power & Energy](power.md) — P = VCE × IC is heat dissipated in the transistor; must stay within ratings
- [GPIO](../embedded/gpio.md) — GPIO current limits make BJT switching necessary for larger loads
- [Motor Control](../embedded/motor_control.md) — H-bridges for bidirectional motor control use four transistors
- [PWM](../embedded/pwm.md) — PWM signals switch a BJT on and off rapidly to control average power to a load
