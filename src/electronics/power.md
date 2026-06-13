# Power & Energy

## Overview

Power is the rate at which electrical energy is converted into something else — heat, light, motion, or stored charge. You can have [high voltage](voltage.md) with tiny [current](charge_current.md) (safe to touch but useless for a motor), or low voltage with huge current (a car battery welding something). Power is what actually determines what a circuit *does*. Energy is how long it can keep doing it. These concepts are used constantly when sizing components, picking batteries, and preventing things from burning.

```
Water wheel analogy:

  Water pressure      Water flow rate
  (Voltage, V)   ×   (Current, I)   =   Power output of the wheel

  High pressure, trickle flow  → small wheel output
  Low pressure, flood flow     → small wheel output
  High pressure, high flow     → huge wheel output

  P = V × I   (Watts)
```

## Key Concepts

### Power Formula

```
P = V × I

P = power in Watts (W)
V = voltage in Volts (V)
I = current in Amperes (A)
```

Combined with [Ohm's Law](resistance.md) (V = I × R), two more useful forms:

```
P = I² × R    (useful when you know current and resistance)
P = V² / R    (useful when you know voltage and resistance)
```

**Example — LED current limiting resistor:**
A 5 V supply, 330 Ω resistor, 20 mA LED:
```
P = I² × R = (0.020)² × 330 = 0.132 W

A standard 0.25 W resistor is fine.
```

### Energy

Energy is power consumed over time:

```
E = P × t

E = energy in Joules (J) or Watt-hours (Wh)
P = power in Watts (W)
t = time in seconds (s) or hours (h)
```

**Battery capacity** is usually given in milliampere-hours (mAh):

```
Capacity (mAh) × Voltage (V) = Energy (mWh)

Example: 2000 mAh @ 3.7 V = 7400 mWh = 7.4 Wh
```

How long will it run a device drawing 50 mA at 3.7 V?
```
Time = Capacity / Current = 2000 mAh / 50 mA = 40 hours
(ideal; real batteries deliver less)
```

### Power Dissipation as Heat

When current flows through a [resistor](resistance.md), the power it consumes all becomes heat. This is **not optional** — every resistor in a circuit is a tiny space heater.

```
Resistor power rating:

  0.125 W (1/8 W)  — tiny SMD resistor, laptop circuits
  0.25 W (1/4 W)   — standard through-hole resistor
  0.5 W            — slightly beefier
  1 W, 2 W, 5 W    — power resistors (larger, need airflow)
  10 W+            — heatsink required
```

If you exceed a resistor's power rating, it overheats and can crack, smoke, or catch fire.

### Efficiency

Real circuits don't convert all input power to useful output — some is lost as heat:

```
Efficiency (η) = P_out / P_in × 100%

Example: A motor driver takes 12 V at 1 A (12 W in),
         motor shaft delivers 10 W of mechanical power:
         η = 10 / 12 = 83%
         2 W is wasted as heat in the driver circuit.
```

Switching [power supplies](power_supplies.md) are ~85–95% efficient. Linear regulators can be as low as 30% efficient when the voltage drop is large.

### Common Power Levels

| Load | Typical Power |
|------|--------------|
| LED | 0.02–0.1 W |
| Microcontroller (active) | 0.01–0.5 W |
| WiFi radio | 0.25–1 W |
| Raspberry Pi 4 | 3–7 W |
| USB phone charger output | 5–20 W |
| Desktop PC | 50–500 W |
| Electric kettle | 2000–3000 W |

## How It Works

Energy in an electrical circuit is stored in the electric and magnetic fields surrounding the wires and components. When current flows through a resistive material, electrons collide with atoms. Each collision transfers kinetic energy to the lattice as heat — this is Joule heating. The rate of those collisions (and thus heat generation) is exactly P = I²R.

In components that store energy — [capacitors](capacitors.md) store it in an electric field, [inductors](inductors.md) store it in a magnetic field — no heat is produced during ideal storage, only during the resistive parts of the circuit.

## Pitfalls

- **Underrating components** — always pick components with a power rating at least 2× the calculated worst-case dissipation. Components run cooler and last longer.
- **Forgetting the power budget** — summing all the component power consumptions is the first step in any battery-powered design. Many beginners discover the battery dies in minutes because they never checked.
- **Confusing energy and power** — a 100 W light bulb running for 10 hours uses 1 kWh (kilowatt-hour) of energy. The power (100 W) stays constant; the energy accumulates over time.
- **Ignoring efficiency** — if your [power supply](power_supplies.md) is 70% efficient and your load needs 5 W, the supply must handle 5/0.7 ≈ 7.1 W of input power. Size the supply and wiring for the input, not just the output.

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — P = I²R and P = V²/R come from V = IR
- [Circuits](circuits.md) — power consumed by each branch must be tracked in any real design
- [Capacitors](capacitors.md) — energy stored: E = ½CV²
- [Inductors](inductors.md) — energy stored: E = ½LI²
- [Power Supplies](power_supplies.md) — efficiency, input power, and heat all stem from here
- [Transistors (BJT)](transistors_bjt.md) / [MOSFET](transistors_mosfet.md) — switching transistors dissipate power; heatsinking is sized from P = I²×R_DS(on)
