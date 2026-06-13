# Charge & Current

## Overview

Everything in electronics starts with one question: what is electricity? The answer is **moving charge**. Atoms have electrons orbiting their nucleus. In metals, the outermost electrons are loosely held and can drift from atom to atom. When something pushes those electrons in the same direction, we call that flow **electric current**. Understanding current is step one; [Voltage](voltage.md) (what does the pushing) and [Resistance](resistance.md) (what fights the flow) come next.

```
Water pipe analogy:

  Water tank                  Pipe                   Drain
  (source of                (path for               (sink)
   pressure)                 flow)

  [ TANK ] ============================== [ DRAIN ]
              --> water molecules flow -->

  In electronics:
  [ BATTERY ] ============================ [ LOAD ]
               --> electrons flow -->

  Water molecules  =  Electrons
  Litres per second = Amperes (A)
```

## Key Concepts

### Electric Charge

Charge is a fundamental property of matter, like mass. It comes in two flavours:
- **Positive charge** — protons (fixed inside the nucleus, they don't move in wires)
- **Negative charge** — electrons (free to drift in metals)

The unit of charge is the **Coulomb (C)**. One Coulomb is approximately 6.24 × 10¹⁸ electrons. That sounds enormous, but electrons are tiny — even a small current involves enormous numbers of them.

### Electric Current

Current is the rate at which charge flows past a point.

```
I = Q / t

I = current in Amperes (A)
Q = charge in Coulombs (C)
t = time in seconds (s)
```

**One Ampere = one Coulomb of charge passing a point every second.**

Think of it like counting how many litres of water flow through a pipe per second. More electrons per second = higher current.

### Conventional vs Electron Current Direction

This is a famous historical confusion. When Benjamin Franklin defined current direction in the 1700s, nobody knew electrons existed. He picked the direction from positive to negative. Later, we discovered electrons actually flow from negative to positive — the opposite.

```
Battery:
  Negative terminal (-) -----> electrons flow -----> Positive terminal (+)
  Positive terminal (+) -----> conventional current --> Negative terminal (-)

Conventional current (used in all circuit diagrams): + to −
Electron flow (what physically happens): − to +
```

**The rule:** In all schematics and formulas, use **conventional current** (from + to −). It gives correct results even though the electrons physically move the other way.

### DC vs AC

| Type | What it means | Example |
|------|--------------|---------|
| **DC (Direct Current)** | Current flows in one constant direction | Battery, USB power |
| **AC (Alternating Current)** | Current direction reverses periodically | Wall outlet (50 Hz or 60 Hz) |

```
DC:        __________________
          |
 0 -------|----------------------------> time
          |
          (constant positive flow)

AC:       /\    /\    /\
         /  \  /  \  /  \
0 ------/----\/----\/----\-----------> time
              \  /  \  /
               \/    \/
          (flow reverses each half-cycle)
```

Household mains electricity is AC. Batteries, solar panels, and most electronics circuits internally use DC.

### Units and Prefixes

Current is measured in Amperes, but real circuits often deal with fractions:

| Prefix | Symbol | Value | Example |
|--------|--------|-------|---------|
| Ampere | A | 1 A | Electric motor |
| Milliampere | mA | 0.001 A | LED, small circuits |
| Microampere | µA | 0.000001 A | Sensor, low-power MCU sleep |

A human can feel currents as low as 1 mA. Currents above ~100 mA through the heart can be lethal.

## How It Works

In a metal wire, electrons are not truly stationary. They bounce around randomly (thermal motion) at high speeds. When a voltage is applied, it adds a small *drift* in one direction on top of all that random bouncing. This drift is surprisingly slow — about 1 mm/s in a typical copper wire — but because there are so many electrons and the electric field propagates at close to the speed of light, the effect of switching on a circuit is felt almost instantly everywhere.

```
Without voltage (random motion):
  → ← ↑ ↓ ← → ↑ ← → ↓   (net flow = zero)

With voltage applied (drift added):
  →→ ← →→ ↑→ →→ ←→ →→   (net flow = small drift to the right)
```

## Pitfalls

- **Confusing current direction** — schematics always show conventional current (+ to −). Don't fight it; just remember electrons physically go the other way.
- **Thinking electrons "carry" energy like trucks** — they don't. The energy is carried by the electric field around the wire. Electrons are more like a medium that the field pushes.
- **Assuming higher current = more dangerous automatically** — what matters for shock hazard is both current *through the body* and the voltage that drives it. Low voltage (like a 9V battery) rarely drives dangerous current through skin resistance.

## Where this connects

- [Voltage](voltage.md) — the "pressure" that drives current through a circuit
- [Resistance](resistance.md) — what limits how much current flows for a given voltage
- [Power & Energy](power.md) — current × voltage = power consumed
- [Circuits](circuits.md) — how current splits and combines in series/parallel paths
- [Diodes](diodes.md) — components that only allow current to flow one way
