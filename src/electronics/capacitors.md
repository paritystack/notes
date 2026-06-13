# Capacitors

## Overview

A capacitor stores electric charge — it's like a tiny rechargeable battery that charges and discharges almost instantly. Unlike a [resistor](resistance.md) that simply opposes current, a capacitor *stores* energy and *resists sudden changes in voltage*. That last property makes capacitors essential for filtering noise, timing signals, coupling AC while blocking DC, and stabilising power rails. They appear in virtually every electronic circuit. [Inductors](inductors.md) are the complementary component that resist changes in current instead.

```
Water tank analogy:

  Filling a tank:                   Capacitor charging:

   Tap open → water                  Switch closed → current
   flows in → tank                   flows in → capacitor
   fills up → pressure               charges up → voltage
   rises → flow slows                rises → current slows

   Full tank = charged capacitor = energy stored

  Empty tank → open tap → water rushes out fast, then slows
  Charged capacitor → connect load → current rushes out, voltage drops
```

## Key Concepts

### How a Capacitor Is Built

Two conducting plates separated by an insulating material (the **dielectric**):

```
  Plate A (+) ──────────────────
                [dielectric gap]   ← insulator (air, ceramic, film, electrolyte)
  Plate B (−) ──────────────────

  When voltage is applied:
  (+) pulls electrons OFF plate A → plate A becomes positive
  (−) pushes electrons ONTO plate B → plate B becomes negative
  Charge is stored on the plates; current cannot cross the gap.
```

### Capacitance

Capacitance (C) measures how much charge is stored per volt of voltage:

```
Q = C × V

Q = charge in Coulombs (C)
C = capacitance in Farads (F)
V = voltage across the capacitor (V)
```

**One Farad** is enormous — practical values are:

| Unit | Symbol | Typical use |
|------|--------|------------|
| Microfarad | µF | Power supply filtering, electrolytic caps |
| Nanofarad | nF | Signal coupling, medium filters |
| Picofarad | pF | RF circuits, small ceramic caps |

### RC Time Constant

When a capacitor charges through a resistor, the voltage doesn't jump instantly — it rises on an exponential curve:

```
  Vc(t) = V_supply × (1 − e^(−t/τ))

  τ = R × C   (the time constant, in seconds)

  At t = τ:    Vc ≈ 63% of V_supply
  At t = 2τ:   Vc ≈ 86%
  At t = 5τ:   Vc ≈ 99.3%  (considered "fully charged")
```

```
  Charging curve:

  V_supply ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
              .....─────────────────
           ...
         ..
        .
       .
  0 ───────────────────────────────── time
        τ   2τ   3τ   4τ   5τ
```

**Example:** R = 10 kΩ, C = 100 µF → τ = 10,000 × 0.0001 = 1 second. Takes about 5 seconds to fully charge.

### Blocking DC, Passing AC

A capacitor in series with a signal:
- **DC (constant voltage)** — charges up once, then no more current flows. It *blocks* DC.
- **AC (alternating voltage)** — continuously charges and discharges as the voltage alternates. Current *appears* to flow through it. It *passes* AC.

This property is used to couple audio signals between circuit stages without passing any DC offset.

### Energy Stored

```
E = ½ × C × V²

Energy in Joules (J), capacitance in Farads, voltage in Volts.
```

A 1000 µF capacitor at 400 V (common in mains power supplies) stores:
E = 0.5 × 0.001 × 160,000 = **80 J** — enough to cause a serious electric shock. Capacitors can hold their charge long after the power is removed.

### Common Capacitor Types

| Type | Range | Polarised? | Notes |
|------|-------|-----------|-------|
| Ceramic (MLCC) | 1 pF – 100 µF | No | Small, cheap, general purpose |
| Electrolytic (aluminium) | 1 µF – 100,000 µF | Yes | Bulk storage, power supply filtering |
| Tantalum | 0.1 µF – 1000 µF | Yes | Smaller than electrolytic, stable |
| Film (polyester) | 1 nF – 100 µF | No | Low noise, audio/timing circuits |

**Polarised capacitors** (electrolytic, tantalum) have a + and − side. Connecting them backwards can cause them to fail violently — they can bulge, leak, or explode.

## How It Works

When voltage is applied, the electric field between the plates stores energy. No current actually crosses the insulating gap; instead, electrons pile up on one plate while being pulled away from the other. The electric field stores potential energy the same way a compressed spring does.

For AC signals, as the voltage alternates, the capacitor constantly charges and discharges. From the outside, this *looks* like current is flowing straight through — which is why capacitors are said to "pass AC". The higher the frequency, the easier this becomes, which is the basis of [filters](filters.md).

The resistance a capacitor presents to AC is called **capacitive reactance**:
```
Xc = 1 / (2π × f × C)

High frequency → low reactance (easy for AC to pass)
Low frequency → high reactance (hard for AC to pass)
DC (f=0) → infinite reactance (blocked completely)
```

## Pitfalls

- **Reversing a polarised capacitor** — always check the polarity markings. The stripe on an electrolytic cap marks the negative lead.
- **Touching a charged capacitor** — even after power is off, large capacitors (especially in mains equipment) can hold lethal charge for minutes or hours. Always discharge before working on the circuit.
- **Ignoring voltage rating** — every capacitor has a maximum working voltage. Exceed it and the dielectric breaks down; the capacitor fails, sometimes explosively.
- **Forgetting decoupling caps** — microcontrollers and ICs need small ceramic capacitors (100 nF) placed close to their power pins. Without them, switching noise on the supply causes glitches. This is one of the most common PCB design mistakes. See [ADC](../embedded/adc.md) and [Power Management](../embedded/power_management.md).

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — RC time constant τ = R × C
- [Circuits](circuits.md) — capacitors in series/parallel follow the *inverse* of resistor rules
- [Inductors](inductors.md) — complementary component; together they form LC resonant circuits
- [Filters](filters.md) — RC networks are the simplest low-pass and high-pass filters
- [Power Supplies](power_supplies.md) — large electrolytic caps smooth rectified AC into DC
- [ADC](../embedded/adc.md) — bypass capacitors suppress supply noise that contaminates analog measurements
