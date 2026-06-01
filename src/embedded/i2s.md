# I2S (Inter-IC Sound)

## Overview

I2S is a synchronous serial bus dedicated to carrying **digital audio** between chips — an MCU/SoC and an audio codec, DAC, ADC, amplifier, or MEMS microphone. Like [SPI](spi.md) it is clocked and full-duplex-capable, but unlike SPI it has no chip-select and no addressing: it is a continuous stream of audio samples framed by a left/right clock. If [UART](uart.md)/[SPI](spi.md)/[I2C](i2c.md) move *bytes*, I2S moves *samples* forever, in lockstep with a sample-rate clock. It almost always runs with [DMA](dma.md) double-buffering because audio cannot tolerate gaps.

```
        ┌──────────────┐                    ┌──────────────┐
        │     MCU      │  SCK / BCLK  ───▶  │    Codec     │
        │  (I2S        │  WS  / LRCLK ───▶  │  (DAC/ADC/   │
        │   peripheral)│  SD  / SDOUT ───▶  │   amp/mic)   │
        │              │  SD  / SDIN  ◀───  │              │
        │              │  MCLK (opt)  ───▶  │              │
        └──────────────┘                    └──────────────┘
```

## The Clocks

I2S needs several clocks that are all integer ratios of the audio sample rate `Fs`:

| Signal | Name | What it is |
|--------|------|------------|
| **SD** | Serial Data | The audio bits, MSB first |
| **SCK** | Bit Clock (BCLK) | One edge per data bit: `BCLK = Fs × bits_per_frame` |
| **WS** | Word Select (LRCLK / FS) | Channel selector: low = left, high = right. Toggles at `Fs` |
| **MCLK** | Master Clock (optional) | Codec oversampling clock, typically `256×Fs` or `384×Fs` |

For 48 kHz stereo, 32-bit slots: `BCLK = 48000 × 2 × 32 = 3.072 MHz`, `LRCLK = 48 kHz`, `MCLK = 256 × 48000 = 12.288 MHz`. The "master" is whichever side drives BCLK + WS — usually the MCU, but a codec with its own crystal can be master and clock the MCU as slave.

## Frame Formats

The catch with "I2S" is that it names a family. The data-vs-WS alignment differs:

```
Philips I2S (data delayed 1 BCLK after WS edge):
 WS  ──┐__________________┌──────────────────
 SD  ───┤MSB ........ LSB├┤MSB ........ LSB├
        └─ left sample ──┘ └─ right sample ┘
        (1 bit after edge)

Left-justified (MSB aligned to WS edge):
 WS  ──┐__________________┌──────────────────
 SD  ──┤MSB ........ LSB├──┤MSB ........ LSB├

Right-justified (LSB aligned to end of slot).
```

Get the format and the slot width wrong and you get noise, half-volume, or swapped channels. Match the codec datasheet's "audio interface format" exactly.

### TDM / PCM mode

For more than two channels (multi-mic arrays, multichannel amps) I2S extends to **TDM**: WS becomes a short frame-sync pulse and N slots ride one data line.

```
 FSYNC ─┐_____________________________________
 SD    ─┤slot0├slot1├slot2├slot3├ ... ├slot7├
        (8 channels in one frame at 8×stereo BCLK)
```

PDM (Pulse Density Modulation) is a *different* 1-bit interface used by cheap MEMS mics — not I2S, though MCUs often share the peripheral. PDM needs a decimation filter (CIC/FIR) to turn the 1-bit stream into PCM samples; see [DSP](dsp.md).

## Why DMA Is Mandatory

At 48 kHz stereo you must hand the peripheral a new 32-bit sample every ~10 µs, forever. Servicing that by interrupt per-sample wastes the CPU and risks underruns. The standard pattern is **ping-pong (double) buffer DMA**:

```
   ┌─ Buffer A ─┐   ┌─ Buffer B ─┐
   │ DMA fills  │   │ CPU process│   ← swap on half/full-complete IRQ
   └────────────┘   └────────────┘
```

The DMA "half-transfer" and "transfer-complete" interrupts tell the CPU which half is now free to fill (playback) or process (capture). Your audio callback runs once per half-buffer, never per sample. See [DMA](dma.md) for the circular-buffer mechanics and the [cache](cache_tcm.md) coherency caveat on Cortex-M7.

```c
// STM32 HAL, circular DMA, stereo 16-bit
int16_t i2s_buf[2 * BLOCK];   // [left,right, left,right, ...] x2 halves

HAL_I2S_Transmit_DMA(&hi2s, (uint16_t*)i2s_buf, 2 * BLOCK);

void HAL_I2S_TxHalfCpltCallback(I2S_HandleTypeDef *h) {
    fill_audio(&i2s_buf[0], BLOCK);          // first half free
}
void HAL_I2S_TxCpltCallback(I2S_HandleTypeDef *h) {
    fill_audio(&i2s_buf[BLOCK], BLOCK);      // second half free
}
```

## Sample Rates and Clock Accuracy

Audio sample rates come in two families that share no common integer MCLK:

| Family | Rates |
|--------|-------|
| 48 kHz family | 8, 16, 32, 48, 96, 192 kHz |
| 44.1 kHz family | 11.025, 22.05, 44.1, 88.2 kHz |

To hit 44.1 kHz cleanly you usually need a dedicated audio PLL or a fractional-N divider — a plain integer divide off a 168 MHz system clock yields ppm error that drifts over a long stream. STM32 audio parts include a separate PLLI2S/SAI clock for exactly this; see [Clock Systems](clock_systems.md). On USB-audio or networked-audio links, clock drift between source and sink forces **asynchronous sample-rate conversion** or a feedback endpoint.

## I2S vs SAI vs SPI-as-I2S

| Peripheral | Notes |
|-----------|-------|
| Dedicated **I2S** block | Fixed I2S/PCM, simple, on most MCUs |
| **SAI** (Serial Audio Interface) | STM32's flexible block: free-running TDM, independent TX/RX clocks, more slot control. Preferred for anything beyond stereo I2S |
| **SPI in I2S mode** | Some MCUs (older STM32) multiplex I2S onto the SPI block — you pick one |
| **PDM peripheral** (DFSDM/MDF) | For PDM mics; includes the decimation filter |

## Where this connects

- [SPI](spi.md) — shares the clocked-serial idea and sometimes the same hardware block.
- [DMA](dma.md) — ping-pong buffering is what makes glitch-free audio possible.
- [Cache & TCM](cache_tcm.md) — DMA audio buffers on Cortex-M7 must be in non-cacheable RAM or hand-maintained.
- [DSP & Fixed-Point](dsp.md) — sample processing (filters, mixing, PDM decimation) happens on the buffers I2S delivers.
- [Clock Systems](clock_systems.md) — a clean audio PLL is what gets you accurate 44.1/48 kHz.
- [I2C](i2c.md) — codecs are almost always *configured* over I2C while *streaming* over I2S (two buses, one chip).

## Pitfalls

1. **Configuring the codec on the wrong bus.** I2S carries audio only; the codec's registers (volume, format, power) are set over [I2C](i2c.md) or SPI. Forgetting the control bus means silence even with perfect I2S.
2. **Frame-format mismatch.** Philips vs left-justified vs slot width — get it wrong and you hear noise, hum, or swapped/half channels. Copy the codec datasheet's format verbatim.
3. **Integer-dividing for 44.1 kHz.** Without an audio PLL the rate is off by enough ppm to audibly drift; use the dedicated I2S/SAI clock.
4. **Per-sample interrupts.** Underruns/clicks under load. Always DMA in half-buffer blocks.
5. **Cache-stale DMA buffers (Cortex-M7).** Buffer in cacheable RAM without clean/invalidate → garbage audio. See [Cache & TCM](cache_tcm.md).
6. **MCLK forgotten.** Many codecs need the master clock to run their internal oversampling; no MCLK = dead codec even though BCLK/WS toggle.
7. **Confusing PDM with I2S.** MEMS mics are often PDM, not PCM I2S — you must decimate the 1-bit stream first.

## See Also

- [SPI](spi.md) — sibling synchronous serial bus
- [DMA](dma.md) — ping-pong buffering
- [DSP & Fixed-Point Math](dsp.md) — processing the audio stream
- [Clock Systems](clock_systems.md) — audio PLLs
