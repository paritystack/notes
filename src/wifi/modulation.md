# Modulation

## Overview

Modulation is the process of encoding information onto a radio wave so it can travel through the air. This page builds the intuition from first principles — starting with analog AM/FM/PM, then following the thread to digital modulation, and finally to the QAM and OFDM schemes that underpin every modern Wi-Fi link. [RF & Spectrum](rf_spectrum.md) covers how modulated signals propagate and degrade; [OFDMA](ofdma.md) extends OFDM to multiple simultaneous users; [Standards](standards.md) maps each 802.11 generation to the modulation it introduced.

## The carrier wave — three things to wiggle

A radio transmitter generates a sinusoidal **carrier wave** at a fixed frequency (e.g., 2.4 GHz). On its own it carries no information. To send data, you vary — *modulate* — one or more of its three properties:

```
One cycle of a carrier wave:

  Amplitude
  (peak height)
       │
   +A ─┤     ╭──╮           ╭──╮
       │   ╭╯    ╰╮       ╭╯    ╰╮
   0 ──┼──╯        ╰──────╯        ╰──►  time
       │                                  →  frequency = cycles/second
   -A ─┘                                  →  phase = where in the cycle you are

  Three handles:
    Amplitude  — how tall the wave is
    Frequency  — how fast it oscillates
    Phase      — where the cycle starts relative to a reference
```

Any of the three can carry a signal. Analog schemes vary them continuously; digital schemes snap them to discrete values (symbols).

## Analog modulation

### AM — Amplitude Modulation

The audio signal (e.g., a voice) controls the *height* of the carrier. When the voice is loud, the carrier is tall; when quiet, it shrinks.

```
Audio signal:
  ╭──╮       ╭──╮
 ╯    ╰─────╯    ╰──

AM output (envelope follows audio):
  ╭╮  ╭╮         ╭╮  ╭╮
 ╯  ╰╯  ╰───────╯  ╰╯  ╰──
  ↑ tall when loud    ↑ shrunk when quiet
```

**Trade-off:** Noise and interference add to or subtract from amplitude. Static, lightning, and nearby electrical equipment punch straight through into the audio — the crackling you hear on AM radio. AM is spectrally efficient (narrow bandwidth) but fragile.

### FM — Frequency Modulation

The audio signal controls how *fast* the carrier oscillates. Louder signal → higher frequency; quieter → lower. The amplitude stays constant.

```
Audio signal:
  ╭──╮                   ╭──╮
 ╯    ╰───────────────── ╯    ╰──

FM output (cycles squeeze and stretch):
 ╭╮╭╮╭╮╭╮  ╭─╮  ╭─╮  ╭╮╭╮╭╮╭╮
╯  ╰╯  ╰╯  ╰╯ ╰╯  ╰╯  ╰╯  ╰╯  ╰──
  ↑ fast (loud)  ↑ slow (quiet)  ↑ fast again
```

**Trade-off:** Because information is in the frequency, not the amplitude, noise (which hits amplitude) is largely ignored by the FM receiver. This is why FM radio sounds much cleaner than AM. The cost: FM occupies more bandwidth than AM for the same audio quality (Carson's rule: bandwidth ≈ 2 × (peak deviation + audio bandwidth)).

### PM — Phase Modulation

The audio signal shifts the *phase* of the carrier — how far it is offset from a reference cycle. A positive voltage nudges the wave forward; negative nudges it back.

```
Reference carrier:   ╭──╮     ╭──╮
                    ╯    ╰───╯    ╰──

PM output (phase shifted forward):
                  ╭──╮     ╭──╮
                 ╯    ╰───╯    ╰──
                 ↑ whole waveform shifted left
```

PM and FM are closely related: FM is equivalent to PM applied to the *integral* of the modulating signal. This relationship becomes important in digital modulation, where PSK (phase-shift keying) and its variants dominate.

## Why digital?

Analog modulation has two fundamental problems:

1. **Noise accumulates.** Each relay or amplifier in a chain adds noise. After enough hops, the signal is degraded.
2. **No error correction.** A scratched vinyl or a noisy AM signal is permanently damaged — there is no way to reconstruct what was intended.

Digital modulation sidesteps both. Instead of a continuous voltage, data is encoded as discrete **symbols** — a finite set of possible states. The receiver only needs to decide *which symbol* was sent, not measure an exact value. A little noise just needs to be smaller than the gap between symbols. And once you have discrete symbols, you can add **forward error correction (FEC)** — redundant bits that let the receiver fix errors.

The three analog techniques map directly to digital equivalents:

| Analog | Digital | Principle |
|--------|---------|-----------|
| AM | ASK (Amplitude Shift Keying) | discrete amplitude levels |
| FM | FSK (Frequency Shift Keying) | discrete frequency hops |
| PM | PSK (Phase Shift Keying) | discrete phase values |

## Digital modulation

### ASK / OOK

**On-Off Keying (OOK)** is the simplest case: carrier on = 1, carrier off = 0. Infrared TV remotes use this. It's susceptible to interference (noise looks like a signal) and wastes power during 0s.

```
Bit stream:   1    0    1    1    0

OOK signal:  ╭──╮     ╭──╮╭──╮
             ╯    ╰───╯       ╰──
```

### FSK — Frequency Shift Keying

Two (or more) discrete frequencies represent different bit values. Bluetooth uses a variant (GFSK). Simple to implement but spectrally inefficient — each symbol only carries 1 bit.

```
Bit stream:   0         1         0

FSK:         ╭─╮  ╭─╮ ╭╮╭╮╭╮╭╮ ╭─╮  ╭─╮
            ╯   ╰╯   ╰╯         ╰╯   ╰╯   ╰──
             low freq (0)  high (1)  low (0)
```

### PSK — Phase Shift Keying

**BPSK (Binary PSK):** Two phases, 180° apart. Each symbol carries 1 bit.

```
BPSK constellation (I/Q plane):

  Q
  │
  X ─────── I ─────── X
  │
  0°                 180°
  (bit 1)            (bit 0)
```

**QPSK (Quadrature PSK):** Four phases, 90° apart. Each symbol carries 2 bits — doubling throughput with the same bandwidth.

```
QPSK constellation:

  Q
  │
  X   X       (11)  (01)
  ────┼──── I
  X   X       (10)  (00)
  │

Four points at 45°, 135°, 225°, 315°.
```

The receiver measures which quadrant the received signal falls in and decodes the 2-bit value. BPSK and QPSK are robust at low SNR — the symbols are far apart, so noise has to be large to cause an error. This is why they appear at MCS 0 for weak signals in Wi-Fi.

### QAM — combining amplitude and phase

QPSK only uses phase. **QAM (Quadrature Amplitude Modulation)** varies *both* amplitude and phase simultaneously, creating a grid of constellation points. More points = more bits per symbol.

```
16-QAM (4 bits/symbol):           64-QAM (6 bits/symbol):

  Q                                  Q
  │                                  │
  X X │ X X                        X X X │ X X X
  X X │ X X                        X X X │ X X X
  ────┼──── I                      X X X │ X X X
  X X │ X X                        ──────┼────── I
  X X │ X X                        X X X │ X X X
  │                                X X X │ X X X
                                   X X X │ X X X
  16 points → 4 bits/symbol         64 points → 6 bits/symbol
```

Each step up multiplies throughput, but the constellation points get closer together:

```
Packing more points into the same space:

  16-QAM:          64-QAM:         256-QAM:
  X . X . X        X.X.X.X.X       X·X·X·X·X·X·X·X
  . . . . .        .........       ···············
  X . X . X        X.X.X.X.X       X·X·X·X·X·X·X·X
  . . . . .        .........
  X . X . X

  ← farther apart              closer together →
  ← tolerates more noise       needs cleaner signal →
```

This is the fundamental SNR trade-off: higher-order QAM carries more bits per symbol but requires a higher signal-to-noise ratio for the receiver to distinguish adjacent points. Wi-Fi's MCS (Modulation and Coding Scheme) index automatically selects the highest QAM order the current SNR can support. See [RF & Spectrum](rf_spectrum.md) for the full MCS table and SNR thresholds.

## From single-carrier to OFDM

### Why single-carrier fails at high rates

If you push a high data rate through a single wide carrier, the symbol period gets very short. In a real environment, multipath echoes (reflections off walls) arrive at the receiver delayed by tens to hundreds of nanoseconds. When the delay is a significant fraction of a symbol period, the echo of symbol N smears into symbol N+1 — **inter-symbol interference (ISI)**. The receiver can no longer cleanly decode either symbol.

```
Direct path:     ─── symbol N ──── symbol N+1 ────►
Reflected path:       (delayed Δt)
                           ─── symbol N ──── symbol N+1 ────►
                                        ↑
                              Echo of N bleeds into N+1
                              (ISI — receiver gets confused)
```

### OFDM: many narrow sub-carriers

**OFDM (Orthogonal Frequency Division Multiplexing)** solves ISI by splitting the channel into many narrow sub-carriers. Each sub-carrier has a *long* symbol period (because it's narrow), so the multipath delay is a small fraction of the symbol — manageable with a guard interval.

```
Single-carrier (wide, fast, ISI-prone):

  |←─────────── 20 MHz ──────────────→|
  │                                   │
  └───────────── one carrier ─────────┘
     symbol period = short → ISI risk

OFDM (52 narrow sub-carriers, 20 MHz channel):

  |←─────────── 20 MHz ──────────────→|
  ||||||||||||||||||||||||||||||||||||
  each sub-carrier independently QAM-modulated
  symbol period per sub-carrier = ~3.2 µs → long → robust to multipath
```

Each sub-carrier carries its own QAM symbol independently. The sub-carriers are mathematically **orthogonal** — they can be packed tightly without interfering with each other (their peaks align with zeros of neighbors). An IFFT at the transmitter and FFT at the receiver handle the math efficiently.

The **guard interval (cyclic prefix)** — a silent or repeated prefix before each OFDM symbol — absorbs the multipath echo so it doesn't bleed into the next symbol.

For how OFDM is extended to serve multiple users simultaneously (assigning sub-carrier groups to different stations), see [OFDMA](ofdma.md).

## Where this connects

- [RF & Spectrum](rf_spectrum.md) — MCS table, SNR thresholds, OFDM sub-carrier details, and how modulation interacts with propagation and noise
- [OFDMA](ofdma.md) — multi-user OFDM: assigning resource units to different stations simultaneously
- [Standards](standards.md) — how each 802.11 generation raised the QAM ceiling (256-QAM in ac, 1024-QAM in ax/be)
- [Wifi Basics](basics.md) — channels and bands that carry the modulated signals

## Pitfalls

**Higher QAM order is not always better.** 1024-QAM requires ~35 dB SNR; if your link only has 20 dB, the radio drops to 64-QAM automatically. Forcing a higher MCS index via software won't improve throughput — it will wreck it.

**FM ≠ FSK.** FM is analog and continuous; FSK is digital and discrete. They use the same physical principle (frequency carries information) but are completely different in implementation and decoding.

**Modulation order ≠ throughput.** MCS bundles modulation *and* coding rate. MCS 5 (64-QAM 2/3) and MCS 7 (64-QAM 5/6) use the same modulation but different amounts of FEC redundancy. The coding rate matters almost as much as the modulation order.

**OFDM sub-carriers are not independent channels.** They share the same regulatory channel and must be transmitted and received together as one OFDM symbol. You cannot assign different OFDM sub-carriers to different Wi-Fi clients in basic OFDM — that requires OFDMA (Wi-Fi 6+).
