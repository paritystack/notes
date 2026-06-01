# Displays & Graphics

## Overview

Adding a screen to an embedded product opens a stack of decisions that the rest of this
book's peripheral pages only hint at: how the pixels get to the panel
([SPI](spi.md), parallel, or MIPI DSI), where the **framebuffer** lives (and
whether you can afford one), how to avoid tearing, and what graphics library renders the
UI. The spectrum runs from a tiny 128×64 monochrome OLED bit-banged over [I2C](i2c.md) on
an 8-bit MCU, up to a 1024×600 24-bit TFT driven by an LCD controller and
[DMA](dma.md) on a Cortex-M7. This page maps the interfaces, the memory math, and the
software stack (notably **LVGL**), and connects to [SPI](spi.md), [DMA](dma.md),
[QSPI](qspi.md) (for storing assets), and [cache/TCM](cache_tcm.md) (framebuffer coherency).

```
   SMALL  ─────────────────────────────────────────────►  LARGE
   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ I2C OLED │   │ SPI TFT      │   │ parallel RGB │   │ MIPI DSI     │
   │ 128×64   │   │ 240×320      │   │ + LCD ctrl   │   │ 720p+ panel  │
   │ no FB,   │   │ panel-RAM or │   │ framebuffer  │   │ framebuffer  │
   │ page RAM │   │ MCU FB       │   │ in SDRAM     │   │ + GPU        │
   └──────────┘   └──────────────┘   └──────────────┘   └──────────────┘
   bytes of RAM      KB                 100s of KB          MB + Linux
```

## Interfaces

How pixels physically reach the glass — the first and most constraining choice:

| Interface | Wires/speed | Where the framebuffer lives | Typical panel |
|-----------|-------------|------------------------------|---------------|
| **I2C** | 2, slow | none — write to controller RAM in pages | tiny mono OLED (SSD1306) |
| **SPI / QSPI** | 4–6, ≤~80 MHz | **inside the panel** (GRAM) or in MCU | 0.96″–3.5″ TFT (ST7789, ILI9341) |
| **Parallel RGB (DPI)** | 16–24 data + sync | **in the MCU** (SDRAM), continuously scanned out | 4″–7″ TFT via on-chip LCD controller |
| **MIPI DSI** | 1–4 differential lanes, fast | in MCU/SoC framebuffer | high-res phone-class panels |
| **LVDS / HDMI** | — | SoC framebuffer | large/[Linux](embedded_linux.md) displays |

The critical distinction is **where the pixels are stored**:

- **Panel-RAM displays (SPI/I2C "MCU interface" controllers like ST7789/SSD1306)** keep a
  framebuffer *inside the display controller*. The MCU just writes updated pixels over
  SPI; the controller refreshes the glass itself. Cheap on MCU RAM — you can drive a color
  TFT from a part with a few KB of SRAM by sending only changed regions.
- **RGB/DSI displays** have *no* memory of their own; the host must hold the full
  framebuffer and **continuously scan it out** in real time (driven by a dedicated LCD-TFT
  controller + [DMA](dma.md)). This needs external SDRAM and bandwidth, but enables smooth
  full-screen animation.

## Framebuffer Math

A framebuffer is `width × height × bytes-per-pixel`. The number decides everything:

```
   240×320  RGB565 (2 B/px) =  150 KB   ── fits in some MCU SRAM
   480×272  RGB565          =  255 KB   ── needs large SRAM or external RAM
   800×480  RGB565          =  750 KB   ── external SDRAM required
   800×480  ARGB8888 (4 B)  = 1500 KB   ── SDRAM; double-buffered = 3 MB
```

**Color formats** trade memory for fidelity: `RGB565` (16-bit, the embedded default,
no alpha), `RGB888` (24-bit), `ARGB8888` (32-bit with alpha for blending). Halving
bit-depth halves bandwidth and memory — often the difference between fitting on-chip and
needing SDRAM.

When the whole framebuffer won't fit, the standard escape is a **partial / dirty-region**
strategy: render only changed rectangles into a small buffer and flush those to a
panel-RAM display. LVGL is built around exactly this (you give it a render buffer far
smaller than the screen).

## Tearing, Double-Buffering, VSYNC

If you draw into the same buffer the display is currently scanning out, the panel shows a
half-old, half-new frame — **tearing**. The fixes:

```
   SINGLE BUFFER          DOUBLE BUFFER (ping-pong)
   draw ──► scanout       draw into B  while  scanout reads A
   (tears mid-frame)      then swap A↔B at VSYNC (between frames)

   VSYNC: the panel's "frame just ended" signal — swap buffers HERE
```

- **Double buffering** — render into a back buffer, swap to front during the **vertical
  blanking** interval so the swap is invisible. Costs 2× framebuffer memory.
- **VSYNC / TE (Tearing Effect) line** — panel-RAM displays expose a **TE** GPIO that
  pulses when the panel finishes a refresh; sync your SPI flush to it to avoid tearing
  even with a single buffer.
- **DMA flush** — push the buffer over [SPI](spi.md)/parallel via [DMA](dma.md) so the CPU
  renders the next frame while the last one transfers. On [cached](cache_tcm.md) Cortex-M7,
  **clean the cache** before the DMA reads the framebuffer or the panel shows stale pixels.

## Graphics Libraries

What renders widgets, text, and images above the raw pixels:

- **LVGL** (Light and Versatile Graphics Library) — the dominant open-source embedded GUI:
  widgets, layouts, anti-aliased fonts, animations, themes; designed for MCUs with a small
  render buffer and a `flush_cb` you wire to your panel driver. Runs from a few hundred KB
  of RAM up.
- **Vendor stacks** — ST **TouchGFX** (Cortex-M, leverages Chrom-ART/DMA2D 2D accelerator),
  Segger **emWin**, **Slint**, **Embedded Wizard**.
- **Direct drawing** — for a status OLED you just write pixels/text with the controller's
  command set (e.g. SSD1306 page addressing); no library needed.
- **Linux** — on application-class SoCs, the framebuffer/DRM stack plus Qt, LVGL, or a web
  runtime; see [Embedded Linux](embedded_linux.md).

Asset storage (fonts, images) often lives in external [QSPI](qspi.md) flash, read via
memory-mapped XIP so large bitmaps don't consume internal flash.

## 2D Acceleration & Touch

- **2D accelerators** — many MCUs include a blitter (ST **DMA2D**/Chrom-ART, others) that
  fills rectangles, blends with alpha, and converts color formats without the CPU — the
  difference between sluggish and smooth UI. LVGL/TouchGFX hook into it.
- **Touch** — capacitive controllers (FT5x06, GT911) report touch points over
  [I2C](i2c.md) with an [interrupt](interrupts.md) line; resistive panels are read via
  [ADC](adc.md). The touch input feeds the GUI library's event loop — a natural fit for the
  [event-driven](state_machines.md) model.

## Where this connects

- [SPI](spi.md) / [QSPI](qspi.md) — the most common small-display interface; QSPI also stores fonts/images for memory-mapped read.
- [DMA](dma.md) — flushing framebuffers/regions to the panel without tying up the CPU; double-buffering relies on it.
- [Cache & TCM](cache_tcm.md) — clean the cache before DMA reads a framebuffer on Cortex-M7, or place it in non-cacheable RAM.
- [I2C](i2c.md) — tiny OLED panels and capacitive touch controllers.
- [Embedded Linux](embedded_linux.md) — DRM/framebuffer and full GUI toolkits on application-class SoCs.
- [State Machines & Event-Driven Firmware](state_machines.md) — touch and UI events drive the GUI as an event loop.

## Pitfalls

1. **Underestimating framebuffer RAM.** An 800×480 ARGB8888 double buffer is ~3 MB — no
   on-chip SRAM holds it. Compute `w×h×bpp×buffers` *before* choosing the panel and MCU.
2. **Tearing from single-buffer drawing.** Writing the live buffer mid-scan shows split
   frames. Double-buffer and swap at VSYNC, or sync the flush to the panel's TE line.
3. **Cached framebuffer not cleaned before DMA.** The panel scans stale cache contents;
   clean (not invalidate) before the DMA read, or use non-cacheable/[TCM](cache_tcm.md) memory.
4. **Confusing panel-RAM and RGB displays.** An SPI controller-RAM panel needs no host
   framebuffer; an RGB/DSI panel needs the host to scan one out continuously. Picking wrong
   blows your memory budget or your bandwidth.
5. **Blitting pixel-by-pixel on the CPU.** Full-screen redraws in software are painfully
   slow; use the 2D accelerator (DMA2D/Chrom-ART) and partial/dirty-region updates.
6. **SPI clock too slow for the frame rate.** Frame time = pixels × bits ÷ SPI clock; a
   240×320 RGB565 full refresh at 20 MHz is ~60 ms (~16 fps). Use DMA, partial updates, or a faster bus.
7. **Forgetting gamma/orientation/offset quirks.** Controllers (ST7789 et al.) have column/row
   offsets and rotation registers; a mis-set offset shifts the image off-screen by a few pixels.

## See Also

- [SPI](spi.md) — the workhorse small-display bus
- [DMA](dma.md) — framebuffer transfers and double-buffering
- [Cache & TCM](cache_tcm.md) — framebuffer coherency on Cortex-M7
- [Embedded Linux](embedded_linux.md) — the big-display, full-GUI end of the spectrum
