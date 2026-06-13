# Logic Gates

## Overview

Logic gates are the building blocks of all digital electronics. Each gate takes one or more binary inputs (HIGH or LOW, 1 or 0, true or false) and produces a single binary output according to a simple rule. A smartphone processor contains billions of these gates. Understanding gates bridges analog electronics (transistors, voltages) with the world of software and digital systems — every instruction your CPU executes traces back to combinations of gates made from [MOSFETs](transistors_mosfet.md).

```
Light switch analogy:

  AND gate: both switches must be ON for the light to turn on
            ──[SW_A]──[SW_B]──[Light]──
            Series wiring: both closed = light on

  OR gate:  either switch can turn the light on
            ──[SW_A]──┬──[Light]──
                      │
            ──[SW_B]──┘
            Parallel wiring: either closed = light on
```

## Key Concepts

### Binary Voltage Levels

| Logic level | Meaning | Typical voltage (3.3 V logic) |
|-------------|---------|-------------------------------|
| HIGH (1) | True | 2.0–3.3 V |
| LOW (0) | False | 0–0.8 V |

Anything in between is an illegal state — digital circuits must never linger there.

### The Basic Gates

#### NOT (Inverter)

One input. Output is the opposite.

```
  Symbol:  A ──○── Y     (circle = inversion)
  Truth table:
    A | Y
    0 | 1
    1 | 0
```

#### AND

Two (or more) inputs. Output is HIGH only if ALL inputs are HIGH.

```
  Symbol:   A ─┐
               ├D── Y
            B ─┘
  Truth table:
    A | B | Y
    0 | 0 | 0
    0 | 1 | 0
    1 | 0 | 0
    1 | 1 | 1
```

#### OR

Output is HIGH if ANY input is HIGH.

```
  Symbol:   A ─┐
               ├)── Y
            B ─┘
  Truth table:
    A | B | Y
    0 | 0 | 0
    0 | 1 | 1
    1 | 0 | 1
    1 | 1 | 1
```

#### NAND (NOT AND)

AND followed by inversion. Output LOW only if ALL inputs HIGH. **Universal gate** — any logic function can be built from NAND gates alone.

```
  Symbol:   A ─┐
               ├D○── Y
            B ─┘
  Truth table:
    A | B | Y
    0 | 0 | 1
    0 | 1 | 1
    1 | 0 | 1
    1 | 1 | 0   ← only case where output is 0
```

#### NOR (NOT OR)

OR followed by inversion. Also a **universal gate**.

```
  Truth table:
    A | B | Y
    0 | 0 | 1   ← only case where output is 1
    0 | 1 | 0
    1 | 0 | 0
    1 | 1 | 0
```

#### XOR (Exclusive OR)

Output HIGH when inputs are *different*. Used in adder circuits and parity checking.

```
  Symbol:   A ─┐
               ├)=── Y
            B ─┘
  Truth table:
    A | B | Y
    0 | 0 | 0
    0 | 1 | 1   ← different = 1
    1 | 0 | 1   ← different = 1
    1 | 1 | 0
```

### Gates From MOSFETs

Every logic gate in a CMOS chip is made from pairs of N-channel and P-channel [MOSFETs](transistors_mosfet.md):

```
  CMOS NOT (inverter) — simplest gate, 1 PMOS + 1 NMOS:

  Vdd (supply)
   │
  [PMOS]  ← ON when A=0 (gate low → PMOS turns on)
   │
   ├────── Y (output)
   │
  [NMOS]  ← ON when A=1 (gate high → NMOS turns on)
   │
  GND

  When A=0: PMOS on, NMOS off → Y = Vdd = 1
  When A=1: PMOS off, NMOS on → Y = GND = 0
```

This complementary push-pull arrangement (CMOS = Complementary MOS) draws nearly zero DC current — only during switching. That's why modern chips can pack billions of gates and still run on batteries.

### Universal Gates

Any Boolean function can be built from NAND gates alone (or NOR alone):

```
  NOT from NAND:    A ─┬─┐
                       └─┤D○── Y  (tie both inputs together)

  AND from NAND:    NAND → NOT output = AND
  OR from NAND:     De Morgan's theorem: A OR B = NOT(NOT_A AND NOT_B)

  Entire CPU can be (and historically was) built from a single gate type.
```

### Combining Gates — Half Adder

Adding two 1-bit numbers requires two gates:

```
  A ─┬── XOR ── Sum (the result bit)
  B ─┤
     └── AND ── Carry (overflow into next bit)

  A=1, B=1: Sum=0, Carry=1  (1+1=10 in binary = 2 in decimal)
```

Chain 8 of these together (with carry propagation) and you have an 8-bit adder — the heart of every ALU in every processor.

## Pitfalls

- **Floating inputs** — an unconnected input on a logic gate doesn't default to 0 or 1. It picks up noise and behaves unpredictably. Always tie unused inputs to Vcc or GND through a resistor, or connect to a defined signal.
- **Mixing logic families** — a 5 V TTL output driving a 3.3 V CMOS input can damage the input (5 V exceeds the 3.3 V rated max). Use level shifters when crossing voltage domains.
- **Propagation delay** — each gate takes a few nanoseconds to respond. At high clock speeds (GHz), this matters. Chains of many gates limit operating frequency.
- **Glitches (hazards)** — when multiple paths through combinational logic have different delays, the output can momentarily spike before settling. This matters in asynchronous designs.

## Where this connects

- [MOSFET Transistors](transistors_mosfet.md) — CMOS gates are built from NMOS and PMOS pairs
- [BJT Transistors](transistors_bjt.md) — older TTL logic used BJTs
- [Processor Design](../embedded/processor_design.md) — combining gates builds adders, multiplexers, and ultimately CPUs
- [ISA](../embedded/isa.md) — instruction set architecture describes what operations the gates implement
- [Switches, Relays & Electromechanical](switches_relays.md) — an SR latch from two NAND/NOR gates debounces a switch in hardware
