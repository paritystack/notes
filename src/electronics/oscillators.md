# Oscillators & the 555 Timer

## Overview

An **oscillator** is a circuit that produces a repeating signal all by itself — no
input required, just power in and a steady waveform out. It is the heartbeat of
electronics: every clock, blinking LED, tone generator, PWM source, and radio
carrier starts with an oscillator. This page builds on the [RC time constant](capacitors.md)
from capacitors, the [comparator](op_amps.md) behaviour of op-amps, and the
frequency ideas from [AC signals](ac_signals.md). The two big families are
*relaxation oscillators* (charge a cap, dump it, repeat) and *feedback oscillators*
(amplify a signal back into itself). The classic chip that everyone learns first is
the **555 timer**.

```
  Oscillator = power in → repeating waveform out

        ┌───────────────┐
   Vcc ─┤   oscillator  ├─►  ╱│╱│╱│   (square, triangle, or sine)
        └───────────────┘
              (no signal input needed)
```

## Key Concepts

### The RC Time Constant — the Engine of Timing

Almost every simple oscillator times itself by charging a capacitor through a
resistor. Recall from [capacitors](capacitors.md):

```
  τ = R × C          (seconds)

  Cap charges toward Vcc on an exponential curve;
  a threshold detector fires when it crosses a set level,
  resets the cap, and the cycle repeats.
  Bigger R or C → slower charge → lower frequency.
```

### The 555 Timer

The 555 is a comparator-plus-flip-flop chip built around exactly that idea. It has
two internal comparators set at ⅓ Vcc and ⅔ Vcc that watch the capacitor voltage.

#### Astable Mode (free-running oscillator)

The cap charges through R1+R2 and discharges through R2, ping-ponging between
⅓ and ⅔ Vcc forever:

```
  Vcc ──[R1]──┬──[R2]──┬── DISCHARGE(7)
              │        │
           THRESH(6)  TRIG(2)
              └───┬────┘
                 ─┴─ C
                  │
                 GND
                                 ⅔Vcc ─ ─ ╱│ ╲ ╱│ ╲
  Cap voltage swings:                    ╱   ╳   ╳
                                 ⅓Vcc ─ ╱  ╲ │╱  ╲│

  Output (pin 3):  ‾‾|__|‾‾|__|‾‾   square wave

  f       = 1.44 / ((R1 + 2·R2) · C)
  T_high  = 0.693 · (R1 + R2) · C
  T_low   = 0.693 · R2 · C
```

#### Monostable Mode (one-shot pulse)

A trigger produces a *single* output pulse of a fixed length, then the circuit
waits for the next trigger:

```
  Pulse width  t = 1.1 · R · C

  Trigger ─┐_┌──────────────  (brief low pulse on pin 2)
  Output  ____|‾‾‾‾‾‾‾|______  (one clean pulse of length t)
```

Used for debouncing buttons, timeouts, and turning a messy edge into one tidy pulse.

### Comparator / Schmitt-Trigger Relaxation Oscillator

You can build the same idea from an [op-amp](op_amps.md) comparator with positive
feedback (a Schmitt trigger). The output charges a cap through a resistor; when the
cap crosses the trigger's threshold the output flips, reversing the charge — and it
oscillates. The positive feedback gives clean, snap-action switching.

### Crystal & Resonator Oscillators

RC oscillators are cheap but drift with temperature and component tolerance — fine
for a blinking LED, useless for a clock. When you need accuracy, a **quartz
crystal** replaces the RC network. A crystal mechanically resonates at one precise
frequency (stable to a few parts per million).

```
  Crystal symbol:        ──┤├──   (two plates around a quartz slice)
                          ┌──┐
                       ──┤ XX ├──

  Typical: 32.768 kHz (real-time clocks — divides neatly to 1 Hz),
           8/16/25 MHz (MCU and Ethernet clocks).
```

This is why microcontrollers list an external crystal: it sets the exact CPU clock.
See [Timers](../embedded/timers.md) and the clock trees in MCU datasheets.

## Pitfalls

- **555 astable can't reach ≤50% duty cycle the simple way** — because the charge
  path (R1+R2) is always longer than the discharge path (R2), T_high > T_low. To
  get a square or shorter-high duty cycle, add a diode across R2 so charging
  bypasses it, or use a different topology.
- **No decoupling cap** — the 555's output stage draws a sharp current spike each
  transition, dumping noise on the supply. Always add a 100 nF cap across Vcc–GND
  (and the recommended 10 nF on the CONTROL pin). See [decoupling](capacitors.md).
- **RC oscillators drift** — frequency moves with temperature, supply voltage, and
  ±20% capacitor tolerance. Don't use one where timing accuracy matters; use a crystal.
- **Loading the timing node** — connecting a meter or load directly to the
  capacitor/threshold pin changes the RC constant and shifts the frequency. Buffer it.
- **CMOS vs bipolar 555** — the classic NE555 is bipolar and noisy; the 7555/TLC555
  CMOS versions run at lower current and higher frequency. Pick deliberately.

## Where this connects

- [Capacitors](capacitors.md) — the RC time constant sets the oscillation period
- [Resistance & Ohm's Law](resistance.md) — timing resistors and pull-ups
- [Op-Amps](op_amps.md) — comparator and Schmitt-trigger oscillators
- [AC Signals & Impedance](ac_signals.md) — frequency, period, and waveform shape
- [Filters](filters.md) — turning a square wave into a sine, or selecting a harmonic
- [Timers](../embedded/timers.md) — MCUs generate precise timing in hardware instead
- [PWM](../embedded/pwm.md) — a variable-duty square wave drives motors, LEDs, and DACs
