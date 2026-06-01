# DSP & Fixed-Point Math

## Overview

Digital Signal Processing on a microcontroller means running filters, transforms, and math-heavy algorithms on streams of samples from an [ADC](adc.md), [I2S](i2s.md) codec, or [IMU](sensors.md) — fast enough to keep up in real time, on a chip that may have no floating-point unit at all. Two intertwined skills make this possible: **fixed-point arithmetic** (representing fractions with integers when there's no FPU, or when you want determinism and speed) and the **filter/transform toolbox** (FIR, IIR, FFT). ARM's **[CMSIS-DSP](cmsis.md)** library packages optimized, fixed-point and float versions of all of these. This page is the bridge between raw [sensor](sensors.md) samples and meaningful results, and the math engine behind [motor control](motor_control.md) FOC and [audio](i2s.md).

```
   ADC / I2S / IMU          ┌──────────────────────────┐
   samples  ──────────────▶ │  DSP pipeline            │ ──▶ result
   (Q15 / float)            │  ▸ scale / window         │     (filtered,
                            │  ▸ FIR / IIR filter       │      transformed,
                            │  ▸ FFT → spectrum         │      classified)
                            │  ▸ fixed-point throughout │
                            └──────────────────────────┘
```

## Fixed-Point: Fractions Without an FPU

A fixed-point number is just an integer with an *implied* binary point. **Q notation** `Qm.n` means `m` integer bits and `n` fractional bits; the stored integer equals `real_value × 2^n`.

| Format | Bits | Range | Resolution | Common use |
|--------|------|-------|------------|------------|
| **Q15** | 16 | [−1, +1) | 2⁻¹⁵ ≈ 0.00003 | Audio samples, normalized DSP |
| **Q31** | 32 | [−1, +1) | 2⁻³¹ | High-precision DSP |
| **Q16.16** | 32 | ±32768 | 2⁻¹⁶ | General fixed-point |

```
  Q15 value 0.5  →  stored as 0.5 × 32768 = 16384
  Q15 value -1.0 →  stored as -32768

  ADD/SUB:  same format, just add the integers
  MULTIPLY: Q15 × Q15 → Q30, then shift right 15 to get back to Q15
            int32_t p = ((int32_t)a * b) >> 15;   // keep intermediate in 32-bit!
```

The two rules that cause every fixed-point bug:

- **Multiplication grows the fractional bits** (`Qa × Qb = Q(a+b)`), so you must shift back and you must hold the intermediate product in a *wider* type (Q15×Q15 needs 32 bits before the shift).
- **Addition can overflow the range.** Summing many Q15 values can exceed ±1; use **saturating** arithmetic (clamp to the max/min instead of wrapping) — wrapping turns a loud sample into the opposite-polarity loud sample, an audible *crack*. CMSIS-DSP and the Cortex-M DSP extension provide saturating `QADD`/`QSUB` instructions.

### When to use what

| Have | Prefer |
|------|--------|
| No FPU (M0/M0+/M3) | Fixed-point (Q15/Q31), integer-only |
| Single-precision FPU (M4F/M7) | `float` — often as fast as fixed-point and far easier |
| Need hard determinism | Fixed-point (no FPU lazy-stacking jitter) |
| Double precision | Avoid on MCUs — usually soft-float, very slow |

The single-precision FPU on Cortex-M4F/M7 changed the calculus: for those parts, `float` is frequently the right default and fixed-point is reserved for the hottest loops or no-FPU targets.

## The Filter Toolbox

### FIR (Finite Impulse Response)

Output is a weighted sum of the last N inputs — a sliding dot product with a coefficient table:

```
  y[n] = Σ  h[k] · x[n−k]     (k = 0..N−1)

  + Always stable, linear phase (no waveform distortion)
  − Needs many taps for a sharp cutoff → more compute per sample
```

This is exactly what the Cortex-M **MAC (multiply-accumulate)** instruction and CMSIS-DSP's `arm_fir_q15` accelerate. A circular buffer of past samples + a coefficient array is the whole implementation.

### IIR (Infinite Impulse Response)

Feeds output back into itself, so few coefficients give a steep response:

```
  y[n] = Σ b[k]·x[n−k] − Σ a[k]·y[n−k]

  + Very cheap (a 2nd-order "biquad" rivals a 30-tap FIR)
  − Can be UNSTABLE; nonlinear phase; fixed-point coefficient
    quantization can move poles outside the unit circle
```

Build steep IIR filters as **cascaded biquads** (`arm_biquad_cascade_df1_q15`) rather than one high-order section — better numerical stability in fixed-point.

### FFT

Converts a block of samples from time domain to frequency domain in `O(N log N)` — the basis of spectrum analysis, tone detection, vibration monitoring, and audio effects. CMSIS-DSP provides `arm_rfft_fast_f32` and fixed-point variants. The gotcha is **windowing**: FFT assumes the block repeats periodically, so apply a window (Hann, Hamming) first to avoid spectral leakage smearing energy across bins.

## CMSIS-DSP

[CMSIS-DSP](cmsis.md) is the standard, free, ARM-optimized library — use it instead of writing your own:

```c
#include "arm_math.h"

// Q15 FIR low-pass over a block
static q15_t state[BLOCK + NUM_TAPS - 1];
arm_fir_instance_q15 fir;
arm_fir_init_q15(&fir, NUM_TAPS, coeffs_q15, state, BLOCK);
arm_fir_q15(&fir, in_block, out_block, BLOCK);   // uses SIMD/MAC under the hood
```

It auto-selects DSP-extension/SIMD/Helium instructions for your core, so the same call is fast on an M4 and faster on an M7/M55. It covers filters, transforms, matrix math (the [FOC](motor_control.md) Clarke/Park transforms), statistics, and the `q7/q15/q31/f32` type families.

## Where this connects

- [CMSIS](cmsis.md) — CMSIS-DSP ships as part of the CMSIS umbrella; same versioning and headers.
- [ADC](adc.md) — raw samples (often converted to Q15) feeding the pipeline.
- [I2S](i2s.md) — audio sample streams; PDM-mic decimation (CIC/FIR) is a DSP task.
- [Sensors & Fusion](sensors.md) — the filters here implement the smoothing and the fusion math.
- [Motor Control](motor_control.md) — Clarke/Park/SVPWM in FOC are CMSIS-DSP matrix/trig calls.
- [Cache & TCM](cache_tcm.md) — hot DSP loops and coefficient tables often live in DTCM for zero-jitter throughput.

## Pitfalls

1. **Overflow in fixed-point multiply.** `Q15 × Q15` overflows 16 bits before you shift; keep the intermediate in 32 bits, then shift.
2. **Wrap instead of saturate.** A sum exceeding ±1 that wraps flips polarity → audible crack / control glitch. Use saturating `QADD`/`QSUB`.
3. **Unstable fixed-point IIR.** Coefficient quantization moves poles outside the unit circle; the filter rings or blows up. Use cascaded biquads and check stability after quantizing.
4. **FFT without windowing.** Spectral leakage smears a clean tone across bins. Apply Hann/Hamming first.
5. **Double precision on an MCU.** `double` math is soft-float even on an FPU part (the FPU is single-precision) — cripplingly slow. Use `float`/fixed-point.
6. **Soft-float in a fast loop.** No-FPU parts emulate `float` in software; a per-sample `float` filter can't keep up. Switch to fixed-point/CMSIS-DSP.
7. **FPU context not saved in ISRs.** If an [ISR](interrupts.md) uses the FPU, the RTOS/port must preserve FPU registers (lazy stacking) — otherwise float state corrupts across context switches.
8. **Reinventing CMSIS-DSP.** Hand-rolled filters are slower and buggier than the vendor-tuned library. Reach for `arm_*` first.

## See Also

- [CMSIS](cmsis.md) — the CMSIS-DSP library home
- [Sensors & Sensor Fusion](sensors.md) — applied filtering and fusion
- [I2S](i2s.md) — audio sample streams
- [Motor Control](motor_control.md) — FOC transforms
