# Voltage

## Overview

If [current](charge_current.md) is the flow of charge, voltage is what causes that flow — the pressure, the push, the force behind it. Without a voltage difference between two points, charge has no reason to move and no current flows. Voltage is always a *difference* between two points, never an absolute number on its own. Understanding voltage is essential before [Resistance & Ohm's Law](resistance.md), [Capacitors](capacitors.md), and every circuit concept that follows.

```
Water tower analogy:

  High altitude (high potential)
       ___
      |   |
      |   |  <-- water stored at height
      |___|
        |
        |  <-- height difference = pressure = voltage
        |
       ~~~   <-- ground level (reference = 0 V)

  The greater the height difference,
  the more pressure, the more flow.

  In electronics:
    Battery (+) terminal  = high potential
    Battery (−) terminal  = low potential (= ground, 0 V)
    Voltage across battery = potential difference
```

## Key Concepts

### What is Voltage?

Voltage (also called *potential difference* or *electromotive force* when from a source) measures how much energy each unit of charge gains or loses moving between two points.

```
V = W / Q

V = voltage in Volts (V)
W = energy in Joules (J)
Q = charge in Coulombs (C)
```

**One Volt = one Joule of energy transferred per Coulomb of charge.**

If a battery is rated at 9 V, it gives 9 joules of energy to every coulomb of charge that travels through it.

### Ground (Reference Point)

Voltage is always *relative*. You must always specify "voltage at point A with respect to point B". In circuits, we pick one reference point and call it **ground** (0 V). All other voltages are measured relative to it.

```
Circuit with a 9 V battery:

  +9 V ─────────────┐
                     │
                    [R]  <-- resistor
                     │
   0 V ─────────────┘
  (GND)

  Voltage across resistor = 9 V − 0 V = 9 V
```

Ground is just a convention — it doesn't mean the circuit is connected to the earth (though sometimes it is). It's the chosen zero reference.

### Voltage Sources

| Source | Voltage | Notes |
|--------|---------|-------|
| AA battery | 1.5 V | Alkaline |
| 9V battery | 9 V | Alkaline |
| USB 2.0 / 3.0 | 5 V | |
| USB-C PD | 5–20 V | Negotiated |
| Li-ion cell | 3.7 V nominal | 4.2 V full, 3.0 V empty |
| Mains (Europe) | 230 V AC | ~325 V peak |
| Mains (USA) | 120 V AC | ~170 V peak |

### Voltage Drop

As current flows through a component, it "uses up" voltage. The voltage measured across a component (from one end to the other) is called the **voltage drop** across it.

```
9 V battery feeding two resistors in series:

  +9 V ──[R1 = 6Ω]──[R2 = 3Ω]── 0 V

  Current: I = 9 V / 9 Ω = 1 A
  Drop across R1: V1 = 1 A × 6 Ω = 6 V
  Drop across R2: V2 = 1 A × 3 Ω = 3 V
  Total drop: 6 + 3 = 9 V  ✓  (equals source voltage)
```

All the voltage supplied by the source is "dropped" across the components — nothing disappears, it's just converted to heat, light, or motion.

## How It Works

A battery maintains a voltage difference through a chemical reaction. The reaction at the negative terminal releases electrons; the reaction at the positive terminal absorbs them. This separation of charge creates an electric field between the terminals. The field exerts force on electrons in any conductor connected between the terminals, causing them to drift — that is, current flows.

```
Inside a battery:

  (-) terminal                (+) terminal
  [oxidation          cell           reduction]
  [releases e−]  ←chemical→   [absorbs e−]
       |                              |
       └──── wire ── load ────────────┘
              current flows this way →
```

When the chemical fuel is exhausted, the reaction can't maintain the separation and the voltage drops — the battery is "dead".

## Pitfalls

- **"Voltage flows"** — voltage doesn't flow, *current* flows. Voltage is a pressure that *drives* flow. Saying "voltage flows through a wire" is like saying "pressure flows through a pipe" — incorrect.
- **Forgetting to specify a reference** — "the voltage at this pin is 3.3 V" only makes sense if you know the reference. In most circuits it's GND (0 V), but always check.
- **Touching high-voltage DC thinking it's safer than AC** — the body's response to DC vs AC differs, but high-voltage DC (above ~50 V) is just as lethal as AC. The danger comes from current through the body; voltage is what drives that current.
- **Confusing voltage and energy** — a 9 V battery in a TV remote contains very little *energy* (9 V × tiny capacity = milliwatt-hours). A 9 V battery in a car jump-starter has vastly more energy. Same voltage, very different stored energy.

## Where this connects

- [Charge & Current](charge_current.md) — current is what voltage drives through a circuit
- [Resistance & Ohm's Law](resistance.md) — V = I × R relates voltage, current, and resistance
- [Power & Energy](power.md) — P = V × I; voltage times current equals power
- [Capacitors](capacitors.md) — a capacitor stores charge to maintain a voltage
- [Diodes](diodes.md) — diodes have a fixed forward voltage drop (~0.7 V for silicon)
- [Power Supplies](power_supplies.md) — circuits that produce a stable, regulated voltage
