# Signal Processing & Fourier Analysis

## Overview

Signal processing is the maths of *signals* — quantities that vary over time or space —
and its central idea is that any signal can be viewed two ways: as a waveform in time, or
as a recipe of frequencies. The bridge between those views is the **Fourier transform**.
This duality underpins [convolution](../machine_learning/convolution.md) in neural nets,
audio/DSP on [embedded](../embedded/processor_design.md) hardware, compression and
bandwidth limits in [information theory](information_theory.md), and the stability
concerns of [numerical methods](numerical_methods.md). It leans on
[calculus](calculus.md) (integrals), [linear algebra](linear_algebra.md) (the transform
is a linear operator), and [complex numbers](discrete_math.md) (frequencies are phasors).

```
TIME DOMAIN                         FREQUENCY DOMAIN
amplitude vs time                   amplitude vs frequency
  /\    /\    /\                       │
 /  \  /  \  /  \        ⇄ Fourier ⇄   │   ▌            one spike =
/    \/    \/    \                     │   ▌            one pure tone
─────────────────► t                   └───┴──────────► f
```

## The frequency-domain idea

A pure sinusoid `A·sin(2πft + φ)` has one frequency `f`, amplitude `A`, phase `φ`.
**Fourier's theorem**: *any* reasonable signal is a sum (or integral) of sinusoids. A
square wave, for instance, is the sum of a fundamental plus odd harmonics:

```
square(t) ≈ sin(ωt) + (1/3)sin(3ωt) + (1/5)sin(5ωt) + …
```

Listing each frequency's amplitude and phase is the **spectrum**. The spectrum carries
exactly the same information as the waveform — no data is lost, the view just changes.

## Fourier series → Fourier transform

```
Fourier SERIES     periodic signal  → discrete set of harmonics (k·f₀)
Fourier TRANSFORM  any signal       → continuous spectrum  X(f)

           ∞
  X(f) =   ∫  x(t) · e^(−2πi f t) dt          (analysis: time → freq)
          −∞
           ∞
  x(t) =   ∫  X(f) · e^( 2πi f t) df          (synthesis: freq → time)
          −∞
```

The kernel `e^(−2πift) = cos(2πft) − i·sin(2πft)` is a rotating phasor; multiplying and
integrating measures "how much of frequency f is present." Magnitude `|X(f)|` is the
strength, angle `∠X(f)` is the phase.

## DFT and the FFT

Computers handle *sampled* signals of `N` points, so they use the **Discrete Fourier
Transform**:

```
        N−1
 X[k] =  Σ  x[n] · e^(−2πi·kn/N)          k = 0 … N−1
        n=0
```

Done naively this is `O(N²)`. The **Fast Fourier Transform** exploits symmetry to split
the sum into even/odd indices recursively — a classic
[divide and conquer](../algorithms/divide_and_conquer.md) recurrence — giving
`O(N log N)`:

```
DFT(x) = DFT(x_even) + twiddle · DFT(x_odd)     T(N) = 2T(N/2) + O(N)
```

This single speedup is what makes real-time audio, radio, MP3/JPEG, and OFDM (Wi‑Fi/LTE)
practical.

## Sampling and Nyquist

To digitize a continuous signal you sample it every `Ts = 1/fs` seconds.

```
Nyquist–Shannon theorem
  To reconstruct a signal perfectly, sample at  fs > 2·f_max.
  fs/2 is the "Nyquist frequency" — the highest representable frequency.
```

Sample too slowly and high frequencies **alias** — they masquerade as lower ones and
corrupt the signal irreversibly:

```
true 7 Hz tone sampled at 8 Hz  ──►  appears as a 1 Hz tone   (8 − 7)
```

The fix is an analog **anti-aliasing low-pass filter** before the sampler.

## Convolution and filtering

A linear time-invariant system is fully described by its **impulse response** `h`. Its
output is the input **convolved** with `h`:

```
              ∞
 (x * h)[n] = Σ  x[m]·h[n−m]        slide, multiply, sum
            m=−∞
```

The **convolution theorem** is the workhorse identity — convolution in time is
multiplication in frequency:

```
  x * h   ⇄   X · H
```

So filtering = shaping the spectrum. This is exactly the operation a CNN learns; see
[convolution](../machine_learning/convolution.md).

```
FILTER TYPES                  keeps
  Low-pass  (LPF)             frequencies below cutoff   (smoothing, anti-alias)
  High-pass (HPF)             above cutoff               (edge/detail)
  Band-pass                   a middle band              (radio tuning)
  Band-stop / notch           rejects a band             (kill 50/60 Hz hum)
```

## Windowing and spectrograms

The DFT assumes the `N`-sample block repeats forever. A signal that doesn't fit a whole
number of cycles shows **spectral leakage** — energy smears across bins. Multiplying by a
tapering **window** (Hann, Hamming, Blackman) before the DFT suppresses it.

Chopping a long signal into overlapping windowed blocks and stacking their spectra gives
a **spectrogram** — a time × frequency heat-map, the standard input representation for
speech and audio ML models.

```
freq │ ▓▓░░  ░░▓▓        a "chirp" sweeping
     │  ░▓▓░░▓▓░          up in frequency
     │   ░░▓▓░░          over time
     └──────────► time
```

## Where this shows up

- **ML** — [convolutional networks](../machine_learning/convolution.md) are learned
  filters; spectrograms feed audio/speech models; the FFT accelerates large convolutions.
- **Embedded / DSP** — real-time filtering, FFTs, and codecs on
  [DSP cores and accelerators](../embedded/processor_design.md).
- **Information theory** — bandwidth and channel capacity are frequency-domain limits;
  see [information theory](information_theory.md).
- **Numerical methods** — FFT-based methods and the conditioning of transforms relate to
  [numerical stability](numerical_methods.md).
- **Systems** — JPEG (DCT), MP3, and OFDM in Wi‑Fi/LTE all rest on these transforms.

## Pitfalls

- **Aliasing** — under-sampling folds high frequencies into low ones, irreversibly;
  always anti-alias filter before sampling.
- **Spectral leakage** — forgetting to window a non-periodic block smears the spectrum.
- **Off-by-one / Nyquist bin confusion** — DFT bin `k` maps to frequency `k·fs/N`; bins
  above `N/2` are the negative (mirror) frequencies of a real signal.
- **Assuming linearity/time-invariance** — convolution-based reasoning only holds for LTI
  systems; saturating or time-varying systems break it.
