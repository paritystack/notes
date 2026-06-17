# Op-Amps

## Overview

An operational amplifier (op-amp) is a high-gain differential amplifier: it amplifies the *difference* between two input voltages. Think of it as a very sensitive referee that watches two players and shouts the difference between them extremely loudly. By adding a **feedback resistor** from output back to one input, you can tame that extreme gain into a precise, useful amplifier. Op-amps are everywhere — audio circuits, sensor signal conditioning, [filters](filters.md), comparators, oscillators, and [power supply](power_supplies.md) regulators all use them.

```
Referee analogy:

  Two judges score a performance:
    Judge A (non-inverting input, +): 7.5/10
    Judge B (inverting input, −):     7.0/10
    Difference: 7.5 − 7.0 = 0.5

  Referee (op-amp) amplifies that 0.5 by 100,000×
  Output = 50,000  → but capped at max score (supply voltage)

  In practice, we add feedback to control the gain:
  Output feeds back to Judge B → self-corrects → stable amplification
```

## Key Concepts

### The Ideal Op-Amp

Real op-amps are close enough to ideal that the ideal model works for most designs:

| Property | Ideal value | Real value (LM741 example) |
|----------|-------------|---------------------------|
| Open-loop gain (A) | Infinite | ~200,000 |
| Input impedance | Infinite | ~2 MΩ |
| Output impedance | 0 | ~75 Ω |
| Bandwidth | Infinite | ~1 MHz |
| Input offset voltage | 0 V | ~1 mV |

### The Symbol

```
        V+ (non-inverting input)
         │
    +────┘
    │         ╲
    │    op-amp ╲─── Vout
    │         /
    −────┐  /
         │
        V− (inverting input)

  Vout = A × (V+ − V−)

  With A = 100,000, even a 0.1 mV difference → 10 V output
  (until it hits the supply rail and clips)
```

### The Two Golden Rules (with negative feedback)

When an op-amp is connected with negative feedback (output connects back to the − input), two rules apply and make circuit analysis very simple:

1. **The output does whatever it takes to make V+ = V−** (no voltage difference between inputs)
2. **No current flows into either input** (infinite input impedance)

These rules are the key to understanding every op-amp circuit below.

### Common Configurations

#### Non-Inverting Amplifier

Output is in phase with input. Gain > 1.

```
           Vout
            │
           [Rf]
            │
  Vin ──── +│          Vout = Vin × (1 + Rf/R1)
            │
   −────[R1]┘
            │
           GND

  Example: Rf = 9kΩ, R1 = 1kΩ → Gain = 1 + 9/1 = 10×
```

#### Inverting Amplifier

Output is opposite phase to input (inverted). Gain can be < 1.

```
  Vin ──[Rin]─── −
                  │──── Vout = −Vin × (Rf / Rin)
        +          │
        │         [Rf]
       GND         │
                  Vout

  Example: Rin = 1kΩ, Rf = 10kΩ → Gain = −10×  (output inverted)
```

#### Voltage Follower (Buffer)

Gain = 1. Output tracks input exactly. Used to isolate a high-impedance source from a low-impedance load without loading the source.

```
  Vin ──── +
            │──── Vout = Vin
   −────────┘

  (Rf is a direct wire; Rin = ∞)
  High input impedance, low output impedance. Perfect for sensor buffering.
```

#### Summing Amplifier

Adds multiple inputs:

```
  V1 ──[R1]──┐
  V2 ──[R2]──┼── − ──[Rf]── Vout = −(V1×Rf/R1 + V2×Rf/R2 + ...)
  V3 ──[R3]──┘
              +
              │
             GND
```

#### Comparator

No feedback. Output swings hard to + or − rail depending on which input is higher:

```
  V+ > V−  → Vout = +Vsupply (or high)
  V+ < V−  → Vout = −Vsupply (or low)

  Used to compare a signal against a threshold.
  Dedicated comparator ICs (LM393) are faster and avoid output saturation issues.
```

### Rail-to-Rail vs Standard

- **Standard op-amp** — output cannot reach the supply rails; stops ~1–2 V short.
- **Rail-to-rail** — output swings nearly to both supply rails, useful in single-supply (3.3 V, 5 V) battery systems.

### Supply Options

| Supply | How it works | Typical ICs |
|--------|-------------|-------------|
| Dual supply (±15V) | Allows output to swing through 0V | LM741, TL071 |
| Single supply (5V or 3.3V) | Must bias mid-rail; rail-to-rail preferred | MCP6001, LMV358 |

## Pitfalls

- **Forgetting feedback** — an op-amp without feedback operates open-loop (gain = 100,000). Any tiny noise on the inputs saturates the output to the supply rail. This is intentional for comparators, but not for amplifiers.
- **Driving a low-impedance load directly** — op-amp output current is limited (~10–30 mA for small ICs). Driving a speaker or motor directly overloads it. Add a buffer transistor stage.
- **Input common-mode range** — the input voltages must stay within the allowed range (usually 1–2 V from the supply rails for standard op-amps). Exceeding this causes unexpected output behaviour.
- **Phase margin / oscillation** — adding large capacitive loads or poorly chosen feedback components can cause the output to oscillate. This is one of the more subtle op-amp pitfalls.
- **Single supply virtual ground** — on a single 5 V supply, to amplify a signal that swings through zero, you need to bias the non-inverting input to Vcc/2 (2.5 V), making it the "AC ground" reference.

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — gain is set by resistor ratios
- [Filters](filters.md) — active filters use op-amps to achieve sharp roll-off without inductors
- [Capacitors](capacitors.md) — integrator and differentiator circuits use a capacitor as the feedback element
- [Power Supplies](power_supplies.md) — op-amps are the error amplifier inside linear regulators
- [ADC](../embedded/adc.md) — op-amp buffers and amplifiers condition sensor signals before digitising
