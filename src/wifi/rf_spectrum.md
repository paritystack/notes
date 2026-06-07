# RF & Spectrum

## Overview

This page explains the radio-frequency fundamentals that underpin every Wi-Fi link: how signals are represented, how they travel through the air, what degrades them, and what regulatory rules constrain them. [Modulation](modulation.md) builds the intuition from analog AM/FM up through QAM and OFDM before diving into the WiFi-specific detail here; [Wifi Basics](basics.md) covers bands and channels from an 802.11 perspective; [Standards](standards.md) maps each revision to its PHY capabilities; [OFDMA](ofdma.md) goes deep on the multi-carrier scheduler; and [Scanning](scanning.md) shows how devices use these concepts to discover networks.

## The electromagnetic spectrum

Radio frequency (RF) is the portion of the electromagnetic spectrum between roughly 3 kHz and 300 GHz. At those frequencies, electrical oscillations in an antenna couple into propagating waves that travel through free space at the speed of light.

```
Frequency (Hz)     3k      30k    300k    3M     30M    300M    3G     30G    300G
                   |        |      |       |       |       |      |      |      |
Band               ELF     VLF     LF     MF      HF     VHF    UHF    SHF   EHF
                                                          FM/TV  Cell   WiFi  mmWave

                                                                  ^------^
                                                            Wi-Fi lives here
                                                          (2.4 GHz, 5 GHz, 6 GHz)
```

Wavelength and frequency are inversely related:

```
wavelength (m) = speed_of_light (3×10⁸ m/s) / frequency (Hz)

2.4 GHz  →  ~12.5 cm
5.0 GHz  →  ~6.0 cm
6.0 GHz  →  ~5.0 cm
```

The key intuition: lower frequency = longer wavelength = travels farther and penetrates better but carries less information per unit time. Higher frequency = shorter wavelength = more bandwidth available but attenuates faster.

### Wi-Fi's three bands

| Band | Center range | Typical channels | Key trait |
|------|-------------|-----------------|-----------|
| 2.4 GHz | 2.400–2.4835 GHz | 1–13 (14 in JP) | Long range, crowded, 3 non-overlapping channels |
| 5 GHz | 5.150–5.850 GHz | 36–177 | More channels, shorter range, DFS in middle sub-bands |
| 6 GHz | 5.925–7.125 GHz | 1–233 (1.2 GHz of clean spectrum) | Wi-Fi 6E/7 only, virtually no legacy interference |

## Signal fundamentals

A radio signal is a sinusoidal wave characterized by three properties:

- **Frequency** (Hz) — how many complete cycles per second; determines which channel a transmission occupies.
- **Amplitude** — the peak displacement of the wave; translates to transmit power.
- **Phase** — the position of the wave within its cycle (0°–360°); exploited by modulation and MIMO.

A **carrier wave** is a high-frequency sinusoid at the channel's center frequency. Data is encoded by varying (modulating) one or more of its properties.

**Bandwidth** in RF means a range of frequencies, not a data rate. A 20 MHz channel occupies 20 MHz of spectrum. The achievable data rate through that channel depends on modulation, coding rate, and SNR — bandwidth is just the pipe width.

```
Carrier at 5.18 GHz (channel 36, center):

  Spectrum occupancy (20 MHz channel):
  |<-------- 20 MHz -------->|
  5.170                   5.190  GHz
         ^
      center = 5.180 GHz
```

## Modulation

Raw data (bits) cannot be transmitted directly as a baseband signal over RF. Modulation maps bit patterns onto changes in the carrier's amplitude and/or phase so they can ride the carrier across the air.

### Amplitude and phase modulation primitives

```
BPSK (1 bit/symbol):           QPSK (2 bits/symbol):
  Q                               Q
  |                               |
  X         X                  X  |  X
  |    ──────────── I           ──────────── I
            |                  X  |  X
                                   |

BPSK encodes 0 and 1 as         QPSK uses four phase points,
two opposite phases.            doubling bits per symbol.
```

### QAM — combining amplitude and phase

Higher-order QAM packs more bits into each symbol by using a larger constellation:

```
16-QAM constellation (4 bits/symbol):

   Q
   |
 X X | X X
 X X | X X
 ────+──── I
 X X | X X
 X X | X X

64 points → 64-QAM (6 bits/symbol)
256 points → 256-QAM (8 bits/symbol)   ← 802.11ac
1024 points → 1024-QAM (10 bits/symbol) ← 802.11ax/be
```

Higher-order QAM requires the constellation points to be closer together, which demands a higher SNR to distinguish them reliably. This is why MCS (Modulation and Coding Scheme) index drops automatically at distance.

### MCS progression

| MCS | Modulation | Coding rate | Bits/symbol (1 stream, 20 MHz) |
|-----|-----------|------------|-------------------------------|
| 0   | BPSK      | 1/2        | ~7 Mbps |
| 4   | 16-QAM    | 3/4        | ~39 Mbps |
| 7   | 64-QAM    | 5/6        | ~65 Mbps |
| 9   | 256-QAM   | 5/6        | ~87 Mbps |
| 11  | 1024-QAM  | 5/6        | ~108 Mbps |

### OFDM — splitting the channel into sub-carriers

Instead of modulating a single wide carrier, OFDM (Orthogonal Frequency Division Multiplexing) divides the channel into many narrow sub-carriers that are mathematically orthogonal — they can be packed tightly without mutual interference.

```
OFDM sub-carriers within a 20 MHz channel (simplified):

 Power
  |   | | | | | | | | | | | | | |
  |   | | | | | | | | | | | | | |
  |___| | | | | | | | | | | | |_|
        ↑                     ↑
   guard sub-carriers     guard sub-carriers
        (null)                 (null)

Each sub-carrier carries its own QAM symbol independently.
```

Key OFDM components:
- **Guard interval (GI)** — silent time between OFDM symbols, absorbs multipath echoes. 802.11ax adds a 0.8/1.6/3.2 µs choice; shorter GI = higher throughput but less multipath tolerance.
- **Cyclic prefix** — a copy of the tail of the symbol prepended to its front; lets the receiver discard multipath-corrupted samples.
- **Pilot sub-carriers** — known reference tones at fixed positions; the receiver uses them to track channel variations.

For multi-user OFDMA (assigning sub-carrier groups to different stations simultaneously), see [OFDMA](ofdma.md).

## Propagation & path loss

### Free-space path loss (FSPL)

In open air with no obstacles, signal power falls off with the square of distance:

```
FSPL (dB) ≈ 20·log10(d) + 20·log10(f) + 32.45
            (d in km, f in MHz)

At 2.4 GHz, doubling distance ≈ +6 dB loss.
At 5 GHz vs 2.4 GHz at the same distance ≈ +6 dB extra loss.
```

This means 5 GHz has ~6 dB higher path loss than 2.4 GHz at identical distances — that's the fundamental reason 5 GHz has shorter range.

### Real-world impairments

```
                    Wall            Ceiling
  Transmitter ───────┤──────────── ─┤─ ──────── Receiver
        │            │              │
        │    Reflection             │  Diffraction
        │    from surface           │  around edge
        └────────────────────────► ─┘
                 Reflected ray (multipath echo)
```

| Mechanism | What happens | Effect on signal |
|-----------|-------------|-----------------|
| **Reflection** | Signal bounces off flat surfaces (walls, floors, metal) | Creates delayed copies (multipath) |
| **Diffraction** | Signal bends around edges | Extends coverage past corners, reduces signal level |
| **Absorption** | Energy converted to heat in the material | Permanent power loss; water, concrete, and brick are strong absorbers |
| **Scattering** | Rough surfaces scatter signal in many directions | Energy spreads; useful for NLOS but weakens the main path |

### Multipath fading

When reflected copies arrive at the receiver slightly delayed, they can add constructively (stronger signal) or destructively (deep null) depending on their phase relationship. This causes **frequency-selective fading** — some sub-carriers are strong while others are weak. OFDM's narrow sub-carriers make this manageable: each sub-carrier experiences flat fading independently, and forward error correction recovers the weak ones.

```
Direct path:    ─────────────────────────────────────►
Reflected path: ──────────────────────────────────────────►
                                                   ↑
                                            delayed by Δt
                                       If Δt = half wavelength,
                                       the signals cancel at the receiver.
```

### 2.4 GHz vs 5 GHz vs 6 GHz propagation trade-offs

| | 2.4 GHz | 5 GHz | 6 GHz |
|---|---------|-------|-------|
| Free-space range | Long | Medium | Short |
| Through-wall loss | ~3–5 dB/wall | ~10–15 dB/wall | ~15–20 dB/wall |
| Multipath resilience | Good (long wavelength) | Moderate | Moderate |
| Available channels | 3 non-overlapping | 25+ (region-dependent) | 59 (20 MHz) / 7 (160 MHz) |
| Interference | Heavy (IoT, BT, microwaves) | Moderate | Minimal (Wi-Fi 6E+ only) |

## Noise & signal quality

### The noise floor

Thermal noise is unavoidable: random electron motion in any resistive element generates noise power. The minimum detectable signal is set by this floor:

```
Noise power (dBm) = −174 dBm/Hz + 10·log10(bandwidth Hz) + NF (dB)

For a 20 MHz channel, NF = 7 dB:
  −174 + 73 + 7 = −94 dBm  ← approximate noise floor

For 80 MHz: −174 + 79 + 7 = −88 dBm (6 dB higher floor)
```

A wider channel has a higher noise floor — which is one reason 80 MHz and 160 MHz channels require a stronger signal to operate reliably.

### SNR, RSSI, and MCS gating

```
Signal level:    −65 dBm  (RSSI — what the radio reports)
Noise floor:     −94 dBm
SNR:              29 dB   (signal − noise)

Typical MCS thresholds (SNR required):
  SNR < 5 dB   → no association
  SNR ~10 dB   → BPSK MCS 0
  SNR ~20 dB   → 64-QAM MCS 7
  SNR ~30 dB   → 256-QAM MCS 9
  SNR ~35 dB   → 1024-QAM MCS 11
```

RSSI alone is misleading. A strong RSSI of −60 dBm in a noisy environment (noise floor −65 dBm) gives only 5 dB SNR — terrible throughput. A −75 dBm RSSI with a −95 dBm noise floor (20 dB SNR) performs much better.

### Interference types

| Type | Description | Mitigation |
|------|-------------|-----------|
| **Co-channel (CCI)** | Another BSS on the same channel; both must contend with CSMA/CA | Choose a different channel; reduce AP density |
| **Adjacent-channel (ACI)** | Overlapping channel from nearby AP bleeds into your channel | Use non-overlapping channels (1/6/11 in 2.4 GHz) |
| **Non-Wi-Fi** | Microwave ovens (~2.45 GHz), Bluetooth, DECT phones, ZigBee, radar | Move to 5 GHz / 6 GHz; DFS avoids radar |

## Antenna basics

### Isotropic vs. real antennas

A theoretical **isotropic antenna** radiates equally in all directions — a perfect sphere. Real antennas concentrate energy in certain directions, giving **gain** in those directions at the expense of others.

Gain is measured in **dBi** (decibels relative to isotropic). A 6 dBi antenna doubles the effective radiated power in its preferred direction compared to a 3 dBi antenna.

```
Dipole radiation pattern (side view):

         | (vertical dipole)
    .-' '-.
   (       )   ← maximum radiation broadside
    '-.  .-'
       \/
    (nothing radiated off the tips)
```

### Radiation patterns

```
Omnidirectional (typical AP ceiling antenna):

  Top view:          Side view:

      N                   ---
   W─────E            (       )
      S               ---   ---
                      (       )
  Equal in all            ↑
  horizontal          Nulls at top/bottom
  directions
```

```
Directional (patch / Yagi):

  Top view:

       ──────────────────────────►
  AP ──────────────────────────►   narrow beam, high gain
       ──────────────────────────►

  Used for point-to-point links or stadium/corridor coverage.
```

### Polarization

Polarization describes the orientation of the electric field component. Most indoor antennas are vertically polarized. When transmit and receive antennas are cross-polarized (one vertical, one horizontal), 20–30 dB of additional attenuation occurs. MIMO exploits cross-polarization: different spatial streams use orthogonal polarizations, effectively doubling the spatial paths in a compact form factor.

### MIMO and spatial streams

MIMO (Multiple Input, Multiple Output) uses multiple antennas at both transmitter and receiver. With enough multipath richness, the receiver can mathematically separate signals that were transmitted simultaneously on the same channel — each is a **spatial stream**.

```
2×2 MIMO (2 transmit, 2 receive):

  TX1 ──── channel matrix H ────► RX1
  TX2 ──────────────────────────► RX2

  H is a 2×2 matrix; if its rows are linearly independent
  (enough multipath), two independent streams are recoverable.
```

| Standard | Max spatial streams | Notes |
|----------|--------------------|----|
| 802.11n  | 4 | First MIMO in Wi-Fi |
| 802.11ac | 8 | MU-MIMO downlink |
| 802.11ax | 8 | MU-MIMO uplink + downlink |
| 802.11be | 16 | Multi-link MIMO |

## Channel width & bonding

### Available widths

Wi-Fi channels can be 20, 40, 80, 160, or (Wi-Fi 7) 320 MHz wide. Wider channels increase throughput because more sub-carriers carry data simultaneously — but they also raise the noise floor and are harder to fit without interference.

```
5 GHz channel bonding example:

20 MHz:  [36]  [40]  [44]  [48]
40 MHz:  [  36+40  ]  [  44+48  ]
80 MHz:  [      36+40+44+48      ]

Primary channel: the 20 MHz slot that carries control frames and beacons.
Secondary channels: bonded for data only when the medium is idle on all of them.
```

Throughput scales roughly linearly with channel width (double width ≈ double PHY rate), but real-world gains depend on whether all bonded channels are clean simultaneously.

### 6 GHz: a clean slate

The 6 GHz band (Wi-Fi 6E and Wi-Fi 7) is restricted to Wi-Fi only — no legacy 802.11a/b/g/n/ac devices, no Bluetooth, no microwave oven harmonics. This makes 160 MHz and 320 MHz channels practical in ways they rarely are at 5 GHz, where overlapping devices fill the spectrum.

### Narrower is more reliable

In dense deployments (apartment buildings, stadiums), forcing 20 MHz channels at 5 GHz gives more non-overlapping channels and avoids CCI from neighbors who would otherwise share your 80 MHz block.

## Regulatory domains & DFS

### Transmit power limits

Regulators cap **EIRP** (Equivalent Isotropically Radiated Power = transmit power + antenna gain). The limit varies by region and sub-band:

| Region | Body | Max EIRP (typical indoor) |
|--------|------|--------------------------|
| USA    | FCC  | 30 dBm (1 W) at 2.4 GHz; 23–30 dBm at 5 GHz |
| Europe | ETSI | 20 dBm at 2.4 GHz; 23 dBm at 5 GHz UNII-2 |
| Japan  | MKK  | 10 mW/MHz PSD limit |

Exceeding EIRP limits is illegal. Consumer APs are certified for a specific regulatory domain and should not be forced to another.

### DFS — Dynamic Frequency Selection

The middle block of the 5 GHz band (UNII-2 and UNII-2 Extended, channels 52–144) is shared with radar systems (weather radar, military). Regulators require Wi-Fi devices in this range to:

1. **Listen for 60 seconds** (Channel Availability Check, CAC) before transmitting on a DFS channel.
2. **Vacate within 10 seconds** if radar is detected during operation (In-Service Monitoring, ISM).
3. Stay off the channel for 30 minutes after a radar event.

```
DFS channel lifecycle:

  [CAC: 60 s silent listen]
          │
          ▼ (no radar)
  [Normal operation]
          │
          ├── radar detected ──► [channel switch announcement]
          │                               │
          │                               ▼
          │                      [vacate within 10 s]
          │                               │
          │                               ▼
          │                      [30 min non-occupancy]
          │
          ▼ (client disconnects during CAC or switch)
```

A radar detection event causes all connected clients to roam or disconnect while the AP moves to a new channel. This is a real operational concern in enterprise environments — plan at least two non-DFS channels (e.g., channels 36–48) as fallback.

### TPC — Transmit Power Control

DFS-band regulations also require **TPC**: the ability to reduce transmit power by at least 6 dB from maximum when a lower power level is sufficient. TPC helps reduce interference with satellites and other incumbent users.

### Non-DFS 5 GHz channels (safe for time-sensitive applications)

```
UNII-1 (indoor only, most regions):   36, 40, 44, 48
UNII-3:                               149, 153, 157, 161, 165
```

These never require CAC and cannot be radar-vacated, making them preferable for voice/video infrastructure.

### 6 GHz regulatory status

The 6 GHz band uses **Automated Frequency Coordination (AFC)** for outdoor standard-power devices: they query a government-approved database to learn which channels are safe near incumbent users (fixed microwave links). Indoor low-power devices (like consumer APs) are typically AFC-exempt.

## Where this connects

- [Modulation](modulation.md) — builds the intuition from AM/FM/PM through PSK and QAM; read this first if the constellation diagrams above feel unmotivated
- [Wifi Basics](basics.md) — channels, bands, and 802.11 frame structure built on top of these RF concepts
- [Wifi Standards](standards.md) — how each generation of 802.11 pushed MCS, spatial streams, and channel width further
- [OFDMA](ofdma.md) — sub-carrier assignment across multiple users, extending OFDM described here
- [Scanning](scanning.md) — passive/active scan behavior depends on regulatory domain and DFS state
- [QoS Management](qos_management.md) — EDCA access categories interact with channel contention shaped by RF conditions
- [../networking/](../networking/) — layer 2/3 context above the PHY layer described here

## Pitfalls

**Confusing RSSI with SNR.** RSSI is just received signal power; without knowing the noise floor it tells you nothing about link quality. Always check both.

**5 GHz short-range surprise.** Moving an AP from 2.4 GHz to 5 GHz in a building with concrete walls can cut coverage range by 50% or more. Plan AP density accordingly.

**Co-channel vs. adjacent-channel interference.** Co-channel (same channel) is generally better than adjacent-channel (overlapping channel) because CSMA/CA lets co-channel devices yield to each other. Adjacent-channel interference is uncorrelated noise — far harder to manage.

**DFS radar kicks at the worst moment.** A radar detection during a voice call or video conference causes an immediate channel switch. Pre-configure at least one non-DFS fallback channel and test your client's 802.11v BSS Transition behavior.

**Regulatory domain mismatch.** Flashing a router with third-party firmware and setting the wrong regulatory domain can unlock channels or power levels illegal in your country, expose you to legal risk, and cause interference to incumbents.

**Wider channels don't always win.** In an apartment building, your 80 MHz channel at 5 GHz likely overlaps four or more neighboring networks. Dropping to 40 MHz and picking a clean primary channel can double real-world throughput by eliminating CCI.

**Antenna polarization matters in MIMO.** If all antennas are co-polarized and the environment is LOS with little scattering, MIMO degenerates — spatial streams collapse to one. Good multipath richness is required for MIMO gains.
