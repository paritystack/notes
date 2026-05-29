# Signal Integrity & Hardware Fundamentals

## Overview

"My SPI bus works at 5 MHz but breaks at 20 MHz." "Sometimes I2C hangs on cold mornings." "ADC reads are noisy when the relay clicks." These are not software bugs. They are **signal integrity** problems — the physical layer of the board failing to deliver clean voltages and currents where the chip expects them.

Embedded software engineers don't need to be RF engineers, but they need enough to:
- Recognize SI issues by symptom.
- Not write off random failures as "software".
- Have informed conversations with hardware folks.
- Avoid common board-bring-up mistakes that look like firmware bugs.

```
What ideal looks like:        What you actually get:

   ┌──────┐                       ┌────╮          ╭────
   │      │ <- clean edge         │    \  ringing │
   │      │                       │     ╲     ╲   │
───┘      └───                ────┘      ╲___╱╲__│
                                          undershoot
```

## Decoupling Capacitors

Every digital chip needs decoupling caps near its power pins. When the chip switches outputs simultaneously, it pulls a sudden burst of current. Without local capacitance to supply that burst, V_DD dips, and the chip's internal logic glitches.

```
                 trace inductance L
   Power rail ──────~~~~~~──────┬───────► chip V_DD
                                │
                                C_decouple
                                │
                               GND
```

When the chip demands `ΔI` in `Δt`:

```
ΔV = L × (ΔI / Δt)
```

A 10 nH trace inductance + 100 mA in 1 ns = 1 V dip. The cap stabilizes V_DD by supplying current locally on that timescale.

### Rules of Thumb

- **One 100 nF cap per V_DD pin**, as close as physically possible (< 5 mm of trace).
- **One 4.7-10 µF bulk cap** per chip for slower transients.
- **Multiple values in parallel** (100 nF + 10 nF + 1 nF) to cover different frequency ranges — questionable benefit on modern MLCCs; mostly historical.
- **Short trace to GND**, not via a long ground path.

If you see "the chip works in some boards, not others", it's often a decoupling layout problem on the failing batch.

## Ground Planes

Current returns to the source it came from. If you don't give it a low-impedance path, it finds a high-impedance path and creates voltage drops that look like noise everywhere.

```
WITHOUT ground plane:                 WITH ground plane:

VCC ──> chip ──> signal               Solid copper underneath
            │
            └── must return            Return current naturally
            via convoluted             follows the signal trace
            ground trace               on the plane below.
```

A **solid ground plane** under the entire board is the cheap, effective default. Splits, slots, or via cuts disrupt return paths and create EMI.

**Critical rule**: **never route a high-speed signal across a gap in the ground plane**. The return current has to detour around the gap → big loop → antenna.

## Trace Length and Reflection

Above a few MHz, traces behave as **transmission lines**, not wires. If the source and load impedances don't match the trace impedance (typically 50 Ω for SMA-routed digital), edges reflect.

```
Source         line (50 Ω)         Load
   ──[source impedance]──~~~~~~──[load impedance]──
            R_s                              R_load
```

Reflection coefficient at load:

```
Γ = (R_load - Z_0) / (R_load + Z_0)
```

A 1 kΩ load on a 50 Ω line reflects ~91% of the signal back. The reflection bounces, summing with new edges, producing **ringing** that overshoots V_DD or undershoots GND. Modern CMOS inputs survive this for a while but ESD diodes inject current into rails and create wider system noise.

### Quick Sanity Check

```
T_propagation ≈ 6.7 ns / m (for FR-4)
λ/10 rule: if trace_length > (T_rise × c / 10), treat as transmission line
```

For 1 ns edges: anything longer than ~20 cm needs SI consideration. For 100 ps edges (DDR, etc.): anything longer than ~2 cm.

### Termination Techniques

- **Series resistor at source** (e.g., 22-33 Ω). Most common for digital, cheap.
- **Parallel resistor at load** (e.g., 50 Ω to ground or to a Vtt rail).
- **AC termination** (R + C in series).

For 8-50 MHz SPI on a short PCB, a 33 Ω series resistor near the master output usually solves overshoot.

## Crosstalk

Adjacent traces capacitively and magnetically couple. A switching aggressor induces voltage on a victim. Result: a quiet trace shows ghost-pulses synchronized with a noisy neighbor.

Mitigations:
- **Separate aggressors from victims** (clock/SPI vs ADC inputs, RF vs digital).
- **Ground guard traces** between sensitive and noisy lines.
- **Layer stack-up**: digital on layer 2 next to a ground plane, analog on layer 4 next to its own plane.
- **Increase separation** — crosstalk falls quickly with distance.

## Power Sequencing and Rail Settling

Multi-rail chips (MCU + Wi-Fi + analog frontend) often require **rails to come up in a specific order**, with bounded time skew. Bringing analog up first and digital second, or violating max V(diff) between rails, can:
- Latch up I/O cells (some chip designs).
- Inject reverse current via ESD diodes.
- Leave parts in an unrecoverable state until full power cycle.

Read the chip's "power-up sequence" diagram in the datasheet. Use **load-switch ICs** or **PMIC** chips with sequencing built in.

## Decoupling for Analog

Analog supplies (V_REF for ADC, AVDD for op-amps) need more care:
- **Ferrite bead + cap** between digital and analog rails to isolate digital noise.
- **Star-point ground** between AGND and DGND on mixed-signal designs (or a single solid plane with careful routing).
- **Quiet supply for V_REF**: low-noise LDO, additional bypass.

A 12-bit ADC's LSB is ~0.7 mV with 3 V V_REF. Digital trash on V_REF directly costs you bits.

## Pull-ups, Pull-downs, and Open-Drain

Many MCU pins start as **inputs floating** at reset. Floating CMOS inputs oscillate, drawing extra current. Tie unused pins to a known state.

**I2C** is open-drain: SDA and SCL are pulled high by an external resistor, devices only pull low. Standard pull-up: 4.7 kΩ at 3.3 V for 100 kHz, **smaller (1-2 kΩ) for 400 kHz fast mode**. Calculating:

```
t_rise = 0.85 × R_pullup × C_bus
```

For 400 kHz I2C the SDA rise must complete in ~300 ns. With 100 pF bus capacitance, R = 3.5 kΩ max.

**Symptom**: I2C works at 100 kHz, fails at 400 kHz with random NACKs. Cause: pull-ups too weak.

## ESD and Hot-Swap

- **TVS diodes** on any externally-exposed pin (USB, CAN, button inputs that leave the board).
- **Series resistors** (100 Ω) on inputs to limit fault current.
- **Inrush limiting** when hot-plugging boards into powered backplanes.

A buzzer wire that touches ground while running rips ESD into V_DD, brownouts, soft-resets, sometimes silicon damage.

## Brown-Out and Reset Supervision

When V_DD drops below the chip's minimum, behavior becomes undefined. MCUs include:
- **POR (Power-On Reset)**: triggers at low rising V_DD.
- **BOR (Brown-Out Reset)**: holds reset while V_DD is below a programmable threshold.

Enable BOR in production. Without it, a slow droop on V_DD can leave the chip in an indeterminate state (cache corrupt, register half-written) without resetting.

External supervisor ICs (TPS3839, MAX809) are even more reliable than on-chip BOR for safety-critical applications.

## Level Shifting

3.3 V MCU talking to 5 V sensor, or 1.8 V SoC talking to 3.3 V flash. Three classes of solution:

| Method | Best for |
|--------|----------|
| **Resistor divider** | One-way, low speed (< 1 MHz) input only |
| **MOSFET-based** (BSS138 + 2 resistors) | Bi-directional, I2C, slow SPI |
| **Dedicated level translator IC** (TXS0108, LSF0108, TXB0108) | High-speed, multi-line, push-pull |
| **Open-drain + matched pull-up** | I2C only, simplest |

The classic **BSS138 + pull-ups** circuit handles bi-directional 3.3 V ↔ 5 V I2C at 100-400 kHz with two parts per line. Above 1 MHz, switch to a dedicated translator.

## EMI vs EMC

- **EMI** (Electromagnetic Interference) = noise your device radiates / conducts to the world.
- **EMC** (Electromagnetic Compatibility) = your device tolerates noise from the world.

You will not solve EMI/EMC issues at the firmware layer. But you can avoid making them worse:

- **Slower switching edges** when full speed isn't needed (datasheet often offers "slew rate control" on GPIOs).
- **Spread-spectrum clocking** on PLLs (smears emissions over a wider band).
- **Avoid switching all GPIOs simultaneously**.
- **Don't toggle unused outputs**.
- **Filter PWM outputs going off-board** through ferrite + cap to limit harmonics.

Production firmware needs EMC/EMI testing in a chamber. If you fail at certification, the first fixes are hardware (more decoupling, better shielding, different cable shielding); firmware-level tweaks come later.

## Common Symptoms and Diagnosis

| Symptom | Likely cause |
|---------|--------------|
| Works at low clock, fails at high | Reflection, undertermination |
| ADC noisy when motor runs | Brownout, ground bounce, V_REF noise |
| I2C random NACKs in fast mode | Pull-ups too weak |
| SPI works near MCU, fails on long cable | Trace impedance / termination |
| Chip resets when load switches | Insufficient bulk decoupling |
| EMC failure at specific frequency | A specific clock or its harmonic radiating |
| Works in lab, fails in field | Brownout, ESD, temperature drift |
| Some boards work, others don't | Component tolerance, assembly issue, layout |
| Crash on hot/cold start | Crystal startup margin, voltage threshold |

## Bring-Up Sanity Checklist

When debugging a new board, before suspecting firmware:

1. **Scope V_DD at the MCU**. Is it clean? Within spec? Does it dip during activity?
2. **Scope the reset pin**. Held low somehow? Glitches?
3. **Scope the clock pin (MCO if you have one)**. Is the crystal oscillating?
4. **Scope the data lines**. Reasonable edges? Levels reach rails? Ringing?
5. **Check ground continuity**. Multimeter between any GND on the board and any other.
6. **Check decoupling caps installed**. Sometimes BOM mistakes lose them.

90% of "firmware doesn't work on new board" reduces to one of these.

## Layout-Adjacent Software Tricks

A few things firmware can do to dodge SI problems:

- **Slow down peripherals** during board bring-up. Confirm functional at 1 MHz before pushing to 50 MHz.
- **GPIO slew rate control** (STM32 OSPEEDR, ESP32 drive strength, etc.). Slower edges = less ringing.
- **Avoid simultaneous toggling of many GPIOs** in one cycle (ground bounce).
- **Stagger high-current peripheral start-ups** (don't enable all Wi-Fi + radios + actuators at once).
- **Filter ADC reads** in software (oversampling + averaging) to fight residual noise.

## Common Pitfalls

### Pitfall 1: Calling Noise "A Software Bug"

If your sensor reads are jittery, the easy fix is exponential smoothing in firmware. The right fix is finding the noise source. Smoothing hides slow-drift problems too.

### Pitfall 2: Long Cables to External Sensors

I2C over a 1 m cable. With 200 pF capacitance, rise time becomes microseconds. NACKs. Either move to differential (RS-485, CAN) or shorten the cable.

### Pitfall 3: Powering External Boards Off the MCU's 3.3 V Rail

The on-board LDO is sized for the MCU. A camera module draws 200 mA, brownouts the MCU. Use a separate regulator with its own decoupling.

### Pitfall 4: Probing With a 1 m Scope Lead Ground

A long scope ground wire is a 200 nH antenna. It picks up everything. Use the spring-tip ground accessory and probe within mm of the IC.

### Pitfall 5: Assuming the Crystal "Just Works"

Crystal that starts in 5 ms at 25 °C might not start at -40 °C, or in a board variant with slightly higher stray capacitance. Test cold/hot.

### Pitfall 6: Mixing Decoupling Cap Values

3 different cap values stacked on a single power pin can resonate at unexpected frequencies. Modern practice: a single 100 nF MLCC is fine; multiple identical caps are fine; mixed values rarely help.

### Pitfall 7: GPIOs Floating

After reset, GPIOs default to input. Floating CMOS inputs draw extra µA and can oscillate. Set them to a defined state in early init, even if unused.

### Pitfall 8: PWM Output Without Filtering Used as "Analog"

A 5 kHz PWM driving an analog input radiates the fundamental and harmonics everywhere. Add an RC filter on the PWM line if it's used as an analog control.

## Summary

1. **Decoupling caps near every V_DD pin** — 100 nF rule.
2. **Solid ground plane** under everything; never break return paths.
3. **Above a few MHz, traces are transmission lines** — match impedances or terminate.
4. **I2C pull-ups: smaller for faster, watch rise time.**
5. **BOR / external supervisor for production.**
6. **Level shift with the right method for the speed.**
7. **GPIO slew control and staggered switching** reduce edge noise.
8. **Diagnosis order**: V_DD → reset → clock → data → ground.
9. **Don't paper over SI bugs in software** — they reappear later.
10. **EMC/EMI is a hardware problem**; firmware can only avoid making it worse.

## See Also

- [Clock Systems](clock_systems.md) — crystals, layout, MCO probe
- [Power Management](power_management.md) — supply transients during sleep wake
- [I2C](i2c.md), [SPI](spi.md), [UART](uart.md) — protocols affected by SI
- [ADC](adc.md) — noisy reads from layout issues
