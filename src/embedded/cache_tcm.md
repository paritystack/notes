# Cache & TCM

## Overview

Higher-end Cortex-M cores (M7, M55) and all application-class cores add two kinds of fast memory between the CPU and slow main memory/[external flash](qspi.md): **caches** (automatic, transparent copies of recently used code/data) and **TCM** (Tightly-Coupled Memory — small SRAM wired directly to the core for guaranteed single-cycle, deterministic access). Caches make average performance fast; TCM makes worst-case performance *predictable*. The catch — and the reason this topic bites embedded developers specifically — is that a cache assumes it's the only thing touching memory, which is false the moment a [DMA](dma.md) engine, another core, or a peripheral reads/writes the same RAM. Managing that **coherency** is the defining embedded cache problem.

```
        ┌──────┐   fast    ┌─────────┐  slow   ┌──────────────┐
        │ CPU  │◀────────▶ │ I/D     │◀──────▶ │ Main SRAM /  │
        │ core │           │ Cache   │         │ QSPI flash   │
        └──┬───┘           └─────────┘         └──────┬───────┘
           │ single-cycle, deterministic              │ DMA writes here
        ┌──▼───┐                                       │ behind cache's back
        │ TCM  │ (ITCM/DTCM, not cached, not on AHB)   ▼  ← COHERENCY HOLE
        └──────┘                                  cache holds stale copy
```

## Cache: What It Buys, What It Costs

A Cortex-M7 fetching code/data from QSPI or even internal flash at 200+ MHz would stall constantly without a cache. Two separate caches:

- **I-cache** (instructions) — pure win, no coherency issue unless you self-modify code or reprogram [flash](qspi.md) (then you must invalidate).
- **D-cache** (data) — big win for throughput, but introduces the coherency hole with DMA.

### Write policies

| Policy | Behavior | DMA implication |
|--------|----------|-----------------|
| **Write-through** | Every write goes to cache *and* memory immediately | Memory always current; DMA reads see fresh data. Slower writes |
| **Write-back** | Writes stay in cache, flushed to memory later | Faster, but memory is *stale* until flush — DMA can read old data |

Write-back is the default for performance, which is exactly why coherency must be managed.

## The DMA Coherency Problem

This is the bug nearly everyone hits once. Two directions, two fixes:

```
TX (CPU → memory → DMA → peripheral):
  CPU writes buffer → sits in D-cache (write-back) → DMA reads MEMORY
  → DMA sends STALE data.
  FIX: SCB_CleanDCache_by_Addr(buf, len)  BEFORE starting DMA
       (push cache contents out to memory)

RX (peripheral → DMA → memory → CPU):
  DMA writes fresh data to MEMORY → CPU reads from D-CACHE
  → CPU sees STALE old data.
  FIX: SCB_InvalidateDCache_by_Addr(buf, len)  AFTER DMA completes
       (drop cache copy so CPU re-reads memory)
```

```c
// Transmit path — flush before the engine reads memory
memcpy(tx_buf, payload, len);
SCB_CleanDCache_by_Addr((uint32_t*)tx_buf, len);
HAL_SPI_Transmit_DMA(&hspi, tx_buf, len);

// Receive path — invalidate before the CPU reads memory
HAL_SPI_Receive_DMA(&hspi, rx_buf, len);
// ... wait for transfer-complete ...
SCB_InvalidateDCache_by_Addr((uint32_t*)rx_buf, len);
process(rx_buf);
```

Mnemonic: **Clean before TX, Invalidate after RX.** This applies to every DMA-driven peripheral — [SPI](spi.md), [I2S](i2s.md) audio buffers, [Ethernet](ethernet.md) descriptors, ADC scans.

### Two cleaner alternatives

1. **Mark the buffer non-cacheable via the [MPU](mpu.md).** Give DMA buffers a region with "Normal, non-cacheable" attributes; then no clean/invalidate is ever needed (at the cost of slower CPU access to those buffers). Often the most robust choice.
2. **Put DMA buffers in a region that isn't cached at all** (e.g. a dedicated SRAM bank you configure as non-cacheable).

### The cache-line alignment trap

`InvalidateDCache_by_Addr` operates on whole **32-byte cache lines**. If your RX buffer shares a cache line with an unrelated variable, invalidating can either drop that variable's pending write or fail to fully cover the buffer. **Align DMA buffers to 32 bytes and pad their size to a multiple of 32**:

```c
__attribute__((aligned(32))) uint8_t rx_buf[ (LEN + 31) & ~31 ];
```

## TCM: Determinism, Not Speed-on-Average

TCM is SRAM bolted directly to the core — same speed as cache *hits*, but with **no misses, ever**. You trade the cache's "fast on average" for "fast always", which is what hard real-time needs.

| Memory | Latency | Determinism | Typical use |
|--------|---------|-------------|-------------|
| **ITCM** (instruction TCM) | 1 cycle | Guaranteed | Hot [ISRs](interrupts.md), control loops, flash-erase routines |
| **DTCM** (data TCM) | 1 cycle | Guaranteed | Stack, [DSP](dsp.md) working buffers, time-critical data |
| Cached SRAM | 1 cycle hit / many on miss | No | General code/data |
| QSPI XIP | many cycles (cache-fronted) | No | Bulk code/assets |

Two big embedded uses:

- **Place a critical [ISR](interrupts.md) or control loop in ITCM** so its execution time has zero jitter regardless of what the cache is doing — vital for motor [FOC](motor_control.md) loops and [DSP](dsp.md).
- **Run flash-erase/[QSPI](qspi.md)-reprogram code from ITCM/RAM** so the core isn't trying to fetch instructions from the very flash it's erasing.
- **DTCM is conveniently non-cacheable**, so it's a natural home for small DMA buffers and avoids the coherency dance entirely (if the DMA engine can reach it — some can't; check the bus matrix).

You place code/data in TCM via [linker-script](linker_scripts.md) sections and attributes:

```c
void __attribute__((section(".itcm"))) control_loop(void) { /* zero-jitter */ }
int  __attribute__((section(".dtcm"))) fast_buffer[256];
```

## Where this connects

- [DMA](dma.md) — the source of every cache-coherency bug; clean/invalidate around every DMA buffer.
- [MPU](mpu.md) — region attributes set cacheability; marking DMA buffers non-cacheable is the clean fix.
- [QSPI & External Flash](qspi.md) — XIP is only fast because the I-cache hides external-flash latency; reprogramming requires invalidation.
- [I2S](i2s.md) / [Ethernet](ethernet.md) / [SPI](spi.md) — DMA-driven peripherals whose buffers all need coherency handling.
- [Interrupts](interrupts.md) / [Motor Control](motor_control.md) / [DSP](dsp.md) — beneficiaries of ITCM/DTCM determinism for jitter-free loops.
- [Linker Scripts](linker_scripts.md) — where ITCM/DTCM and non-cacheable regions are assigned.

## Pitfalls

1. **Forgetting clean/invalidate around DMA.** The #1 Cortex-M7 bug: TX sends stale data, RX reads stale data. Clean before TX, invalidate after RX.
2. **Buffer not 32-byte aligned/padded.** Cache-line operations spill onto neighbours, corrupting unrelated variables or under-covering the buffer. Align and pad to 32.
3. **Invalidating with dirty neighbours.** Invalidating a half-used line discards a pending write of an adjacent variable sharing that line.
4. **Assuming DTCM is DMA-reachable.** Some DMA engines can't access TCM (it's not on the AHB matrix). Check before parking DMA buffers there.
5. **Self-modifying / reprogrammed code without I-cache invalidate.** After writing new code (bootloader, [QSPI](qspi.md) update) the I-cache still holds the old instructions.
6. **Enabling cache without configuring the MPU.** On Cortex-M7 cache behavior depends on memory attributes; an unconfigured map can leave peripherals cacheable (disastrous) or everything non-cacheable (slow).
7. **Expecting TCM to be "faster" than cached RAM.** It isn't faster than a cache *hit* — its value is *determinism* (no misses), not peak bandwidth.

## See Also

- [DMA](dma.md) — the coherency counterpart
- [MPU](mpu.md) — setting cacheability attributes
- [QSPI & External Flash](qspi.md) — XIP latency hidden by the cache
- [Linker Scripts](linker_scripts.md) — placing ITCM/DTCM/non-cacheable sections
