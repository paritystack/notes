# MOSFET Transistors

## Overview

A MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor) is a three-terminal switch controlled by **voltage**, not current. Unlike a [BJT](transistors_bjt.md) where a base *current* opens the valve, a MOSFET's gate is electrically isolated — it takes essentially zero current to hold the switch open. Think of it as a tap controlled by a voltage level on the handle: you set the voltage, the tap opens or closes, and once set, no energy is wasted holding it. MOSFETs are the dominant component in digital logic, power switching, motor drives, and battery management.

```
Voltage-controlled tap analogy:

  Tap handle voltage (Gate)
         |
         ↓
  ┌─────────────┐
  │   tap gate  │  ← voltage determines if tap is open or closed
  └─────────────┘
  Source ──────── Drain
  (inlet)          (outlet)

  N-channel MOSFET (most common):
     Drain (D)
         │
  G ─────┤  NMOS
         │
     Source (S)

  Apply voltage above threshold on Gate → current flows D to S
  Gate voltage below threshold → no current
```

## Key Concepts

### N-channel vs P-channel

| Type | Turns ON when | Conventional current | Common use |
|------|--------------|---------------------|-----------|
| N-channel (NMOS) | V_GS > V_th (positive) | D → S | Low-side switch |
| P-channel (PMOS) | V_GS < V_th (negative) | S → D | High-side switch |

N-channel MOSFETs have lower ON-resistance and are more common. P-channel is used for high-side switching (between supply and load) where the source is at supply voltage.

### Enhancement vs Depletion Mode

- **Enhancement mode** (most common): OFF by default, turns ON when V_GS reaches threshold. Like a normally-closed valve.
- **Depletion mode**: ON by default, turned OFF by applying gate voltage. Less common.

All MOSFETs in this page are enhancement mode.

### Gate Threshold Voltage (V_th)

The gate-to-source voltage at which the MOSFET starts to conduct significantly.

| Category | V_th range | Notes |
|----------|-----------|-------|
| Standard | 2–4 V | May not fully switch with 3.3 V logic |
| Logic-level | 1–2 V | Fully on at 3.3 V or 5 V; use with microcontrollers |
| Low-threshold | 0.5–1 V | Ultra-low power circuits |

**Always check V_th against your logic voltage.** A MOSFET rated for 10 V gate drive will only partially conduct at 3.3 V — it won't saturate and will dissipate a lot of heat.

### ON Resistance (R_DS(on))

When fully switched on, the MOSFET's drain-source resistance:

```
  Power dissipated when conducting:
  P = I_D² × R_DS(on)

  Example: 10 A load, R_DS(on) = 10 mΩ
  P = 100 × 0.01 = 1 W   (much less than a BJT in the same role)
```

Good power MOSFETs have R_DS(on) in the milliohm range. This is why they dominate high-current switching applications.

### Gate Capacitance

The gate is isolated by a thin oxide layer — electrically it's a capacitor. No DC current flows, but you must charge/discharge this capacitance to switch the MOSFET:

```
  Switching time depends on how fast gate capacitance charges:
  t_switch ∝ C_gate × R_drive

  At low frequencies (manual switching or slow PWM): gate capacitance irrelevant
  At high frequencies (fast PWM, DC-DC converters): gate driver circuit needed
  to supply the transient charging current quickly
```

### The Body Diode

Every MOSFET has an inherent **body diode** between drain and source (anode at source for NMOS):

```
  NMOS with body diode:
     D
     │
     ├──►|── (body diode, cathode at D)
     │
  G──┤
     │
     S
```

This diode:
- Allows synchronous rectification in power supplies (MOSFET replaces rectifier diode)
- Conducts when the MOSFET is OFF and reverse current tries to flow
- Has higher forward voltage (~0.7 V) and slower recovery than a Schottky diode — important in high-frequency switching

### MOSFET as a Switch (Low-side)

```
  Switching a 12 V / 5 A load with a 3.3 V GPIO:

        +12V
          │
        [Load]
          │
          D ← NMOS (IRF3708, logic-level, V_th ≈ 1V, R_DS(on) = 11mΩ)
  GPIO───[Rg=100Ω]── G
          S
          │
         GND

  GPIO HIGH (3.3V): V_GS = 3.3V > V_th → MOSFET on → load gets power
  GPIO LOW  (0V):   V_GS = 0V < V_th  → MOSFET off → load disconnected

  100Ω gate resistor: limits gate current transients, prevents ringing
  No high-side resistor on the gate path needed (it's voltage-controlled)
```

### MOSFET vs BJT

| Feature | MOSFET | BJT |
|---------|--------|-----|
| Control | Voltage (V_GS) | Current (I_B) |
| Gate/base current | ~0 (capacitive) | Must maintain I_B |
| ON-resistance | Very low (mΩ) | VCE(sat) ≈ 0.2 V |
| Speed | Faster | Slower |
| Temperature stability | Better | Can run away |
| Cost (for power) | Lower | Lower for BJT at low current |
| Common use | Digital logic, power | Audio amp, low-cost switch |

## Pitfalls

- **Using standard V_th MOSFET with 3.3 V logic** — a MOSFET rated "fully on at 10 V gate" will only be half-on at 3.3 V: it passes some current but with high R_DS(on) → lots of heat. Always use a "logic-level" MOSFET for direct GPIO drive.
- **Leaving the gate floating** — a floating gate picks up static or noise and can turn the MOSFET on unexpectedly. Always add a pull-down resistor (10–100 kΩ) from gate to source when the gate is not actively driven.
- **Forgetting the gate resistor in fast switching** — without a series gate resistor, the capacitive gate causes current spikes that cause EMI and ringing. 10–100 Ω in series is usually sufficient.
- **Ignoring the body diode** — in motor H-bridge designs, the body diode can conduct during dead-time between switching. Know when it will conduct and whether its reverse recovery speed matters.

## Where this connects

- [BJT Transistors](transistors_bjt.md) — current-controlled alternative; understand both to choose correctly
- [Power & Energy](power.md) — P = I² × R_DS(on) determines heat; heatsinking follows from this
- [Power Supplies](power_supplies.md) — MOSFET switches do the switching in buck/boost converters
- [Diodes](diodes.md) — body diode; also Schottky diodes used alongside MOSFETs
- [PWM](../embedded/pwm.md) — fast PWM needs low gate capacitance and a proper gate driver
- [Motor Control](../embedded/motor_control.md) — H-bridges for motors use four MOSFETs
- [GPIO](../embedded/gpio.md) — GPIO pin limits require MOSFET for any load above ~20 mA
