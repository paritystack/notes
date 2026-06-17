# Filters

## Overview

A filter is a circuit that lets certain frequencies of a signal through and blocks others. You already know what filters do — a stereo equaliser that boosts bass and cuts treble is a filter. An anti-glare screen on a monitor is an optical filter. In electronics, filters remove noise, separate signals, and condition data before it reaches an [ADC](../embedded/adc.md) or after it leaves a [DAC](../embedded/dac.md). They rely on [capacitors](capacitors.md) (which pass high frequencies) and [inductors](inductors.md) (which pass low frequencies) and [op-amps](op_amps.md) (for active filters). They connect deeply to the [RF & Spectrum](../wifi/rf_spectrum.md) world.

```
Sound equaliser analogy:

  Audio signal (all frequencies mixed):
    ═══════════════════════════════════
    bass  mid  treble  high

  Low-pass filter (keeps bass, cuts treble):
    ████████░░░░░░░░░░░░░░░░░░░
    bass only

  High-pass filter (cuts bass, keeps treble):
    ░░░░░░░░░░░░░░░████████████
                   treble only

  Band-pass filter (keeps a middle range):
    ░░░░░░░████████░░░░░░░░░░░
              mid only
```

## Key Concepts

### Frequency and the Key Equation

Filters are characterised by their **cutoff frequency** (f_c) — the frequency at which the output power drops to half (−3 dB, or the amplitude drops to 1/√2 ≈ 0.707 of input):

```
  For an RC filter:
  f_c = 1 / (2π × R × C)

  For an LC filter:
  f_c = 1 / (2π × √(L × C))
```

### RC Low-Pass Filter

The simplest filter — one resistor and one capacitor:

```
       R
  Vin ─[///]─┬──── Vout
              │
             [C]
              │
             GND

  Low frequency: capacitor barely conducts → signal passes through
  High frequency: capacitor conducts heavily → signal shorted to GND

  Roll-off: −20 dB per decade above f_c (i.e. 10× frequency = ÷10 amplitude)
```

**Example:** R = 1 kΩ, C = 100 nF → f_c = 1 / (2π × 1000 × 0.0000001) = **1.59 kHz**

Frequencies well below 1.59 kHz pass; frequencies well above it are attenuated.

### RC High-Pass Filter

Swap R and C positions:

```
      C
  Vin ─||─┬──── Vout
           │
          [R]
           │
          GND

  Low frequency: capacitor blocks → signal attenuated
  High frequency: capacitor passes → signal reaches output

  Same f_c formula. Roll-off −20 dB/decade below f_c.
```

**Use:** Removing DC offset from audio signals (coupling capacitor) while passing AC.

### Frequency Response Curve (Bode Plot)

```
  Low-pass filter:

  Gain (dB)
   0 dB ────────────────────.
                             \   ← roll-off slope (−20 dB/dec for 1st order)
  −3 dB                      .
                               \
  −20 dB                        \
                                  \
  −40 dB                           \
           |_________________________\___________→ frequency (log scale)
                                   f_c
```

### First vs Second Order

| Order | Circuit | Roll-off | Typical use |
|-------|---------|----------|------------|
| 1st | One RC | −20 dB/decade | Simple noise rejection |
| 2nd | Two RC, or LC | −40 dB/decade | Sharper cutoff |
| Higher | Multiple stages | −20n dB/decade (n = order) | Anti-aliasing, audio |

For clean anti-aliasing before an ADC sampling at 10 kHz, you want the filter to strongly attenuate signals above 5 kHz — typically a 2nd or higher order filter.

### Active vs Passive Filters

| Type | Components | Advantages | Disadvantages |
|------|-----------|------------|---------------|
| Passive RC | R, C only | Simple, no power | Gain ≤ 1, loading effect |
| Passive LC | L, C | Higher order, efficient | Inductors are large, expensive |
| Active | RC + [op-amp](op_amps.md) | Gain > 1, no inductors, buffered | Needs power supply, limited bandwidth |

**Active low-pass (Sallen-Key, 2nd order):**

```
  Vin ─[R1]──[R2]─── + ─── Vout
                 │         │
                [C1]       │
                 │        [C2]
                GND        │
                          GND
        (op-amp provides buffering and sets gain)
```

### Band-Pass and Band-Stop

```
  Band-pass = high-pass + low-pass in series:
    f_low < signal < f_high → passes through

  Band-stop (notch) = removes a narrow frequency range:
    Used to eliminate 50/60 Hz mains hum from sensor signals
```

### Decibels (dB)

Filter gain is usually expressed in dB:

```
  Gain (dB) = 20 × log10(Vout / Vin)

  Vout = Vin:       0 dB   (no attenuation)
  Vout = 0.707×Vin: −3 dB  (the −3dB cutoff point)
  Vout = 0.1×Vin:  −20 dB  (10× attenuation)
  Vout = 0.01×Vin: −40 dB  (100× attenuation)
```

## Pitfalls

- **Loading effect** — connecting a low-resistance load after a passive RC filter changes the effective R and shifts f_c. Use an op-amp buffer (voltage follower) after a passive filter to prevent this.
- **Ignoring phase shift** — filters don't just change amplitude, they also shift the phase of signals. At f_c, a 1st-order filter shifts phase by 45°. In control loops and audio, this matters.
- **Anti-aliasing filter too weak** — if sampling at 10 kHz without adequately attenuating signals above 5 kHz, higher-frequency content "folds back" into the sampled signal as false noise (aliasing). The filter must roll off sharply enough before f_Nyquist = f_sample / 2.
- **Not accounting for component tolerances** — ±5% resistors and ±10% capacitors can shift f_c significantly. Use precise components (1% R, 1–5% C) in timing-critical filters.

## Where this connects

- [Resistance & Ohm's Law](resistance.md) — R sets the filter's time constant along with C
- [Capacitors](capacitors.md) — capacitive reactance is what makes RC filters frequency-selective
- [Inductors](inductors.md) — inductive reactance enables LC filters
- [Op-Amps](op_amps.md) — active filter stages use op-amps for gain and buffering
- [ADC](../embedded/adc.md) — anti-aliasing filter prevents frequency folding before digitisation
- [RF & Spectrum](../wifi/rf_spectrum.md) — RF front-ends use filters to select channels and reject interference
