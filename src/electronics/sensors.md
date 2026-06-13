# Sensors & Transducers

## Overview

A **sensor** (or **transducer**) turns a physical quantity — temperature, light,
pressure, force, position — into an electrical signal a circuit can read. This page
is about the *analog front end*: how a sensor's raw output is shaped into a clean
voltage before anything digital touches it. It leans on the [voltage divider](resistance.md)
from resistance, the buffering and amplifying tricks of [op-amps](op_amps.md), and
the noise-shaping of [filters](filters.md). It is the partner of the firmware-side
[Sensors & Sensor Fusion](../embedded/sensors.md) page: *that* one covers sampling,
calibration and fusion once the signal is digital; *this* one covers getting a good
analog signal in the first place.

```
  Physical world → transducer → conditioning → ADC → firmware

   light/heat/      sensor       amplify,      digitise   numbers
   force/...        element      filter,                  & fusion
                                 scale
```

## Key Concepts

### Sensor Output Types

Sensors differ mainly in *what electrical property* they vary. Each type needs a
different reading circuit.

| Output | How you read it | Examples |
|--------|----------------|----------|
| **Resistive** | Voltage divider — resistance changes → voltage changes | Thermistor (temp), photoresistor/LDR (light), strain gauge (force), potentiometer (position) |
| **Voltage** | Read/amplify directly | Thermocouple (temp), photodiode, Hall sensor (magnetic) |
| **Current** | Sense across a resistor; immune to wire-length drops | 4–20 mA industrial loop sensors |
| **Capacitive** | Measure changing capacitance (often via an [oscillator](oscillators.md)) | Touch pads, humidity, proximity |
| **Digital** | Read a bus directly — conditioning is on-chip | I2C/SPI temp, IMU, pressure modules |

### Reading a Resistive Sensor — the Voltage Divider

The workhorse circuit. Pair the sensor with a fixed resistor and read the midpoint:

```
   Vcc
    │
  [R_fixed]
    │
    ├──── Vout = Vcc × R_sensor / (R_fixed + R_sensor)
    │
  [R_sensor]   ← thermistor, LDR, etc.
    │
   GND

  As the sensor's resistance changes with temperature/light,
  Vout moves — feed Vout to an ADC.
```

Pick `R_fixed` near the sensor's mid-range resistance so the output swing is
largest where you care about it.

### The Wheatstone Bridge — Reading Tiny Changes

A strain gauge might change resistance by only 0.1%. A plain divider buries that in
a large offset voltage. A **Wheatstone bridge** is two dividers compared against
each other so the big common offset cancels and only the *difference* remains:

```
        Vcc
       ╱   ╲
    [R1]   [R3]
      │     │
     A●     ●B      Vout = V_A − V_B
      │     │
    [R2]   [Rs]     Balanced (R1/R2 = R3/Rs) → Vout = 0
       ╲   ╱        Sensor changes → small Vout, no big offset
        GND

  Feed Vout to a differential/instrumentation amp for a clean reading.
```

### Signal Conditioning

The raw signal is rarely ADC-ready. Conditioning shapes it:

- **Buffer** — a [voltage follower](op_amps.md) isolates a high-impedance sensor so
  the measuring circuit doesn't load it down (see Pitfalls).
- **Amplify** — a thermocouple outputs tens of microvolts; an op-amp (often an
  instrumentation amp) scales it to use the full ADC range.
- **Filter** — a [low-pass RC filter](filters.md) removes mains hum and high-frequency
  noise before sampling, preventing aliasing at the [ADC](../embedded/adc.md).
- **Level-shift / scale** — bias a bipolar signal into the ADC's 0–Vref window.

### Digital Sensors and Pull-ups

Many modern sensors do the conditioning internally and expose an [I2C](../embedded/i2c.md)
or [SPI](../embedded/spi.md) bus. Those open-drain bus lines need pull-up resistors
(typically 4.7 kΩ) to define the idle high level — the same [pull-up idea](resistance.md)
used on logic inputs.

## Pitfalls

- **Self-heating** — pushing too much current through a thermistor or strain gauge
  warms it, so it measures its own heat. Keep excitation current low.
- **Nonlinearity** — thermistors and LDRs are very nonlinear; a doubling of
  resistance is not a doubling of temperature/light. Linearise in firmware with a
  lookup table or the sensor's equation (e.g. Steinhart–Hart for thermistors).
- **Loading a high-impedance source** — connecting a 1 MΩ sensor straight to a
  lower-impedance ADC input pulls the reading down. Buffer it with an op-amp follower.
- **Noise pickup** — high-impedance, low-level sensor lines act like antennas for
  mains hum and switching noise. Keep them short, shielded, and filtered; amplify
  *close* to the sensor.
- **Ratiometric reference mismatch** — a divider's output scales with Vcc. If the
  ADC's reference is separate and drifts independently, readings drift too. Reference
  the ADC to the same supply that excites the sensor (ratiometric measurement).

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — voltage dividers and pull-ups read resistive sensors
- [Op-Amps](op_amps.md) — buffering, amplifying, and instrumentation amplifiers
- [Filters](filters.md) — anti-alias and noise filtering ahead of the ADC
- [AC Signals & Impedance](ac_signals.md) — sensor signals carry noise across a frequency band
- [Oscillators & the 555 Timer](oscillators.md) — capacitive sensors are read by frequency shift
- [ADC](../embedded/adc.md) — the conditioned signal is finally digitised here
- [Sensors & Sensor Fusion](../embedded/sensors.md) — the firmware side: sampling, calibration, fusion
- [I2C](../embedded/i2c.md) — digital sensor modules expose conditioned data over a bus
