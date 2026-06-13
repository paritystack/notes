# AC Signals & Impedance

## Overview

So far every page has assumed **DC** — voltages and currents that hold steady. But most real signals *change over time*: the mains in your wall, the audio in a speaker cable, the radio waves around you, the clock inside a chip. These are **AC** (alternating current) signals. This page is the bridge between the DC world of [resistance](resistance.md) and the time-varying behaviour of [capacitors](capacitors.md) and [inductors](inductors.md), and it underpins every [filter](filters.md) and [power supply](power_supplies.md) you'll meet later. It builds directly on [charge & current](charge_current.md), [voltage](voltage.md), and [circuits](circuits.md).

```
DC vs AC:

  DC (battery):              AC (mains / signal):

  V ──────────────           V    ╱‾╲      ╱‾╲
    constant level                ╱    ╲  ╱    ╲
  0 ─────────────── t         0 ─╳──────╳──────╳── t
                                      ╲╱      ╲╱
  Always one polarity        Swings + then − repeatedly
```

## Key Concepts

### Anatomy of a Sine Wave

The sine wave is the fundamental AC shape — every other periodic signal is a sum of sines.

```
  V
  │      peak (Vp)
  │      ╱‾‾╲
  │    ╱      ╲           ┬
  0 ─╱──────────╲───────╱── t      peak-to-peak (Vpp) = 2 × Vp
  │              ╲    ╱   ┴
  │                ╲╱
  │      |←  period T  →|

  amplitude / peak Vp  : max distance from zero
  peak-to-peak  Vpp    : top to bottom = 2·Vp
  period  T            : time for one full cycle (seconds)
  frequency  f = 1/T   : cycles per second (Hertz, Hz)
  angular freq ω = 2πf : radians per second
```

A signal is written `v(t) = Vp · sin(2πf·t + φ)`, where `φ` is the **phase** (its
horizontal starting offset).

### RMS — the "DC-equivalent" Value

A sine spends most of its time below its peak, so its peak value overstates how
much *work* it does. **RMS** (root-mean-square) is the equivalent DC voltage that
would dissipate the same heat in a resistor:

```
  V_rms = Vp / √2  ≈ 0.707 × Vp        (for a sine wave only)

  Mains "230 V" is the RMS value:
    Vp  = 230 × √2 ≈ 325 V  (the actual peak)
    Vpp = 650 V             (top to bottom)
```

This is why [power](power.md) for AC uses RMS: `P = V_rms² / R`. When someone
quotes an AC voltage without qualification, they mean RMS.

### Phase

Two signals of the same frequency can be shifted in time relative to each other.
That shift, measured in degrees (360° = one full period), is the **phase difference**.

```
  In phase (0°):          90° out of phase:

   A ╱‾╲   ╱‾╲             A ╱‾╲   ╱‾╲
   B ╱‾╲   ╱‾╲             B   ╱‾╲   ╱‾╲   (B lags A by a quarter cycle)
```

Phase is the whole reason capacitors and inductors behave differently from
resistors: in a capacitor the current *leads* the voltage by 90°; in an inductor
it *lags* by 90°.

### Reactance — AC "Resistance" of Caps and Inductors

A capacitor or inductor opposes AC, but the opposition depends on **frequency**.
This frequency-dependent opposition is **reactance** (X), measured in ohms:

```
  Capacitive reactance:  Xc = 1 / (2π·f·C)
    high f → low Xc   (cap passes AC easily)
    f = 0 (DC)        → Xc = ∞ (blocks DC)

  Inductive reactance:   XL = 2π·f·L
    high f → high XL  (inductor blocks high-frequency AC)
    f = 0 (DC)        → XL = 0 (looks like a plain wire)
```

Caps and inductors are *opposites*: rising frequency lowers Xc but raises XL.

### Impedance — Resistance and Reactance Combined

Real circuits mix resistance (R, in-phase) with reactance (X, 90° out of phase).
Because they act 90° apart, you can't just add them — you combine them like the
two legs of a right triangle. The result is **impedance** (Z):

```
  Impedance triangle:

        |Z|
        ╱│
       ╱ │ X   (reactance)
      ╱  │
     ╱θ  │
    └─────
       R         (resistance)

  |Z| = √(R² + X²)        magnitude, in ohms
  θ   = arctan(X / R)     phase angle between V and I

  Ohm's law for AC:  V = I × Z
```

When X = 0 the impedance is purely resistive and AC behaves just like DC. This is
also the doorway to **resonance**: at one special frequency Xc and XL cancel
(Xc = XL), leaving only R — the basis of [LC filters](filters.md) and tuned radio
circuits.

## Pitfalls

- **Confusing peak with RMS** — a "5 V" sine on a scope might mean 5 V peak, 5 V
  peak-to-peak, or 5 V RMS. Always check which. The √2 factor between peak and RMS
  causes constant errors.
- **Treating reactance as fixed** — Xc and XL change with frequency. A coupling
  cap that's "transparent" at 1 kHz can be a serious obstacle at 1 Hz.
- **Adding R and X arithmetically** — they are 90° apart; you must combine them as
  `√(R² + X²)`, not `R + X`.
- **Ignoring phase when summing voltages** — in AC, two 5 V sources out of phase do
  not give 10 V. Phase must be accounted for (this is what phasors handle).

## Where this connects

- [Capacitors](capacitors.md) — capacitive reactance Xc and the charge/discharge basis of AC behaviour
- [Inductors](inductors.md) — inductive reactance XL, the complementary component
- [Filters](filters.md) — reactance vs frequency is exactly what makes a filter selective
- [Power & Energy](power.md) — AC power calculations use RMS values
- [Power Supplies](power_supplies.md) — rectifying and smoothing AC into DC
- [Op-Amps](op_amps.md) — active filters and AC amplifiers work in this domain
- [ADC](../embedded/adc.md) — sampling an AC signal requires understanding its frequency content
