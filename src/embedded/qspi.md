# QSPI & External Flash

## Overview

QSPI (Quad SPI) is an extension of [SPI](spi.md) that uses **2, 4, or 8 data lines** instead of one, primarily to talk to external NOR flash chips at high speed. Its killer feature is **memory-mapped execute-in-place (XIP)**: the QSPI controller makes an external flash chip appear as a region in the CPU address space, so code and constants can live off-chip and be fetched transparently — at the cost of latency that makes the [cache](cache_tcm.md) essential. Where plain [SPI](spi.md) flash gives you a byte-at-a-time command interface, QSPI gives you both a command interface *and* a window you can `memcpy` from or run code out of. It is the standard way to add tens of MB of [flash](flash_filesystems.md) to an MCU that only has 1–2 MB internal.

```
 Standard SPI (1 data line)        Quad SPI (4 data lines)
   CS ─────                          CS ─────
   CLK ────                          CLK ────
   MOSI ───  (1 bit/clk)            IO0 ───┐
   MISO ───                          IO1 ───┤ 4 bits/clk
                                     IO2 ───┤
                                     IO3 ───┘
   Octal (OPI): IO0..IO7 = 8 bits/clk, optionally DDR (both edges)
```

## Line Widths and Throughput

QSPI commands are described by a triple like **1-4-4** or **4-4-4**, meaning (instruction lines)-(address lines)-(data lines):

| Mode | Lines | Rel. read speed | Notes |
|------|-------|-----------------|-------|
| Single (1-1-1) | 1 | 1× | Plain SPI, always works after reset |
| Dual (1-1-2 / 1-2-2) | 2 | ~2× | Rare now |
| Quad (1-4-4 / 4-4-4) | 4 | ~4× | The common case |
| Octal (1-8-8 / 8-8-8) | 8 | ~8× | "OPI", high-end MCUs |
| DDR/DTR | ×2 edges | ×2 again | Data on both clock edges |

A 4-4-4 DDR octal flash at 100 MHz moves ~200 MB/s — fast enough that the on-chip [cache](cache_tcm.md) hides most of the latency for XIP code.

## A Read Transaction

Unlike SPI's "just clock bytes", a QSPI read is a structured frame the controller assembles from a descriptor:

```
┌──────────┬──────────────┬────────────┬─────────────────┐
│ Command  │   Address    │ Dummy clks │      Data       │
│  (8 bit) │  (24/32 bit) │ (latency)  │  (N bytes)      │
└──────────┴──────────────┴────────────┴─────────────────┘
   e.g.        e.g.          required so
   0xEB        3-byte        the flash can
  (Quad I/O)   addr          prefetch
```

The **dummy cycles** are mandatory latency the flash needs between sending the address and returning data; set too few and you read garbage, too many and you waste time. The number is part of the chip's datasheet and sometimes configurable in a flash status register.

## Common Flash Commands

NOR flash is read-fast, write-slow, erase-by-block:

| Op | Typical cmd | Granularity | Time |
|----|-------------|-------------|------|
| Read | 0x03 / 0xEB (quad) | byte | ns/byte |
| Page Program (write) | 0x02 / 0x32 | up to 256 B page | ~ms/page |
| Sector Erase | 0x20 | 4 KB | tens of ms |
| Block Erase | 0xD8 | 64 KB | ~0.1–1 s |
| Chip Erase | 0xC7 | whole chip | seconds |
| Read Status | 0x05 | — | poll WIP bit |
| Write Enable | 0x06 | — | required before write/erase |

The rules that bite everyone: **you can only clear bits (1→0) by programming; to set bits back to 1 you must erase a whole sector**, and **every program/erase must be preceded by Write Enable (0x06) and followed by polling the Write-In-Progress bit**. This erase-before-write asymmetry is why you layer a [flash filesystem](flash_filesystems.md) (LittleFS) on top rather than treating it like RAM.

## Memory-Mapped (XIP) Mode

The reason QSPI matters more than "fast SPI flash" is memory-mapping:

```
CPU address space
┌────────────────────────┐ 0x0800_0000  internal flash (vectors, boot)
│  Internal flash 1 MB   │
├────────────────────────┤ 0x9000_0000  QSPI memory-mapped window
│  External QSPI flash   │  ← CPU reads here trigger QSPI reads
│  (XIP: code + assets)  │     transparently; cache fronts it
└────────────────────────┘
```

In memory-mapped mode the controller auto-issues the read command whenever the CPU (or [DMA](dma.md), or the LCD controller) touches the window. You can:

- **Run code from external flash (XIP)** — `__attribute__((section(".extflash")))` placed by the [linker script](linker_scripts.md). The first byte after a cache miss is slow; the [I-cache](cache_tcm.md) hides the rest.
- **`memcpy` fonts, images, ML weights** straight out of the window.

You **cannot write** through the memory-mapped window — to program/erase you switch the controller back to indirect (command) mode, do the operation, then re-enter memory-mapped mode. Code currently executing from the same flash must not be running during its own erase.

## Where this connects

- [SPI](spi.md) — QSPI is SPI with extra data lines; single-line mode is plain SPI and always works at reset.
- [Flash Filesystems](flash_filesystems.md) — LittleFS/SPIFFS/FatFs sit on top to manage erase blocks and wear.
- [Linker Scripts](linker_scripts.md) — placing `.text`/`.rodata` sections into the external-flash region for XIP.
- [Cache & TCM](cache_tcm.md) — XIP is only fast because the cache absorbs QSPI latency; cache config is mandatory.
- [Bootloaders](bootloaders.md) — second-stage bootloaders often live in or copy from external QSPI flash.
- [DMA](dma.md) — bulk reads from the memory-mapped window can be DMA'd.

## Pitfalls

1. **Wrong dummy-cycle count.** The single most common QSPI bring-up bug: address is right, data is garbage. Match the datasheet, and remember some chips need a status-register write to set their dummy cycles.
2. **Erasing flash you're executing from.** XIP code cannot run during an erase of its own sector — the flash is busy and returns status, not instructions. Move the erase routine to internal RAM/[TCM](cache_tcm.md).
3. **Treating NOR like RAM.** You can't overwrite in place; you must erase a 4 KB+ sector first. Use a [flash filesystem](flash_filesystems.md).
4. **Forgetting Write Enable / not polling WIP.** Writes silently fail without 0x06, or you corrupt data by issuing the next command before the chip finishes.
5. **Cache not invalidated after reprogramming.** Update the flash via indirect mode, then read stale bytes from the window because the [cache](cache_tcm.md) still holds the old data. Invalidate.
6. **Assuming quad mode at reset.** Flash powers up in single-SPI 1-1-1; the QE (Quad Enable) status bit must be set before quad commands work.
7. **3-byte vs 4-byte addressing.** Chips over 16 MB need 4-byte address mode (or a bank register); using 3-byte addresses silently wraps.

## See Also

- [SPI](spi.md) — the base protocol
- [Flash Filesystems](flash_filesystems.md) — managing external flash
- [Linker Scripts](linker_scripts.md) — XIP section placement
- [Cache & TCM](cache_tcm.md) — hiding XIP latency
