# DMA (Direct Memory Access)

## Overview

DMA lets peripherals move data to/from memory **without CPU involvement**. The CPU programs the DMA controller once, then is free to do other work — or sleep — while the controller shuttles bytes between an ADC, UART, SPI, memory buffer, etc.

```
Without DMA:                   With DMA:
                                                   
  Peripheral                     Peripheral        
      │                              │             
      │ IRQ per sample               │ burst transfer
      ▼                              ▼             
    CPU ──► RAM                    DMA ──► RAM     
                                                   
  CPU runs ISR                   CPU sleeps,       
  for every byte                 IRQ only when     
                                 buffer is full
```

**Why it matters:**
- High-throughput peripherals (SPI displays, audio I2S, ADC streams) would overwhelm the CPU with per-byte interrupts.
- Lower power: CPU sleeps while transfer runs.
- Deterministic timing: peripheral-to-memory transfer is hardware-paced, not subject to ISR jitter.

## Mental Model

A DMA **channel** (or **stream** on STM32 F4/F7/H7) is a configured data path. You tell it:

| Setting | Example |
|---------|---------|
| **Source address** | `&SPI1->DR`, or a RAM buffer |
| **Destination address** | RAM buffer, or `&USART2->DR` |
| **Transfer size** | 8 / 16 / 32 bits per item |
| **Item count** | how many transfers before "done" |
| **Increment mode** | step source/dest after each transfer (or hold for FIFO peripherals) |
| **Direction** | mem-to-periph, periph-to-mem, mem-to-mem |
| **Trigger source** | which peripheral request fires this channel |
| **Mode** | one-shot, circular, double-buffer |

When the peripheral signals "I have data" (or "I need data"), the DMA controller arbitrates for the bus, performs the transfer, decrements the count, and optionally fires an interrupt at half/full/error.

## Typical Transfer Types

### Peripheral-to-Memory: UART RX into a Buffer

```c
// STM32 HAL: 256-byte circular RX buffer fed by UART2 DMA
uint8_t rx_buf[256];

HAL_UART_Receive_DMA(&huart2, rx_buf, sizeof(rx_buf));
// returns immediately, DMA runs in background

// HAL fires HAL_UART_RxHalfCpltCallback at index 128
// and HAL_UART_RxCpltCallback at index 256, then wraps
```

Idiomatic pattern: pair with **UART idle-line interrupt** to know when a packet ends mid-buffer (useful for variable-length protocols).

### Memory-to-Peripheral: SPI TX of a Frame Buffer

```c
// Sending a 320x240 RGB565 frame to a display in one shot
HAL_SPI_Transmit_DMA(&hspi1, framebuffer, 320*240*2);
// CPU is free; rendering next frame can start
```

### Memory-to-Memory

DMA can copy RAM-to-RAM faster than a CPU memcpy on cores with low ALU/bus throughput. Common for double-buffered render targets. STM32 supports this on DMA2 (F4) or all channels (H7).

### Circular Mode

The DMA wraps automatically when the count reaches zero. Combined with **half-transfer** and **transfer-complete** interrupts, you get a producer/consumer ring without re-arming the DMA each time.

```
        ┌─ Half-IRQ ─┐  ┌─ Full-IRQ ─┐
        ▼            │  ▼            │
buffer  │░░░░░░░░░░░░│  │░░░░░░░░░░░░│
        0           N/2             N
        
CPU consumes the half that DMA is not currently writing.
```

This is the classic "ADC streaming" or "audio I2S" pattern.

### Double-Buffer (Ping-Pong)

The DMA controller has **two** memory pointers and switches between them on each transfer-complete. Cleaner than circular for "process whole buffer at a time" workloads.

```c
// STM32 F4 stream with double-buffer
DMA_StreamConfTypeDef cfg = { ... };
HAL_DMAEx_MultiBufferStart_IT(&hdma_adc,
    (uint32_t)&ADC1->DR,         // src
    (uint32_t)buf_a,             // dst 1
    (uint32_t)buf_b,             // dst 2
    BUFFER_LEN);
```

## Setting Up a DMA Channel (STM32 LL)

Concrete F4 example: ADC1 conversions streaming into a buffer, with half/full interrupts.

```c
#include "stm32f4xx_ll_dma.h"

#define ADC_BUF_LEN 1024
uint16_t adc_buf[ADC_BUF_LEN];

void adc_dma_init(void) {
    LL_AHB1_GRP1_EnableClock(LL_AHB1_GRP1_PERIPH_DMA2);

    // DMA2, Stream 0, Channel 0 → ADC1 (per RM0090 Table 43)
    LL_DMA_SetChannelSelection (DMA2, LL_DMA_STREAM_0, LL_DMA_CHANNEL_0);
    LL_DMA_SetDataTransferDirection(DMA2, LL_DMA_STREAM_0,
                                    LL_DMA_DIRECTION_PERIPH_TO_MEMORY);
    LL_DMA_SetMode             (DMA2, LL_DMA_STREAM_0, LL_DMA_MODE_CIRCULAR);
    LL_DMA_SetPeriphIncMode    (DMA2, LL_DMA_STREAM_0, LL_DMA_PERIPH_NOINCREMENT);
    LL_DMA_SetMemoryIncMode    (DMA2, LL_DMA_STREAM_0, LL_DMA_MEMORY_INCREMENT);
    LL_DMA_SetPeriphSize       (DMA2, LL_DMA_STREAM_0, LL_DMA_PDATAALIGN_HALFWORD);
    LL_DMA_SetMemorySize       (DMA2, LL_DMA_STREAM_0, LL_DMA_MDATAALIGN_HALFWORD);
    LL_DMA_SetStreamPriorityLevel(DMA2, LL_DMA_STREAM_0, LL_DMA_PRIORITY_HIGH);

    LL_DMA_ConfigAddresses(DMA2, LL_DMA_STREAM_0,
        (uint32_t)&ADC1->DR,
        (uint32_t)adc_buf,
        LL_DMA_DIRECTION_PERIPH_TO_MEMORY);
    LL_DMA_SetDataLength(DMA2, LL_DMA_STREAM_0, ADC_BUF_LEN);

    LL_DMA_EnableIT_HT(DMA2, LL_DMA_STREAM_0);
    LL_DMA_EnableIT_TC(DMA2, LL_DMA_STREAM_0);
    LL_DMA_EnableIT_TE(DMA2, LL_DMA_STREAM_0);

    NVIC_SetPriority(DMA2_Stream0_IRQn, 5);
    NVIC_EnableIRQ(DMA2_Stream0_IRQn);

    LL_DMA_EnableStream(DMA2, LL_DMA_STREAM_0);
}

void DMA2_Stream0_IRQHandler(void) {
    if (LL_DMA_IsActiveFlag_HT0(DMA2)) {
        LL_DMA_ClearFlag_HT0(DMA2);
        process_block(&adc_buf[0], ADC_BUF_LEN/2);
    }
    if (LL_DMA_IsActiveFlag_TC0(DMA2)) {
        LL_DMA_ClearFlag_TC0(DMA2);
        process_block(&adc_buf[ADC_BUF_LEN/2], ADC_BUF_LEN/2);
    }
    if (LL_DMA_IsActiveFlag_TE0(DMA2)) {
        LL_DMA_ClearFlag_TE0(DMA2);
        // log + recover
    }
}
```

The peripheral itself (ADC1 in this case) also needs `ADC_CR2_DMA = 1` to issue DMA requests.

## DMA on STM32H7 / Cortex-M7: Cache Coherency

M7 cores have an L1 data cache (D-cache). DMA writes to SRAM **bypass the cache**:

- CPU writes to a TX buffer → bytes sit in cache, not RAM.
- DMA reads RAM → gets stale data.

And the reverse:

- DMA writes RX buffer to RAM.
- CPU reads → may get cached old value.

Three solutions, in order of preference:

| Approach | What | When |
|----------|------|------|
| **Place DMA buffers in non-cacheable region** | Configure MPU region as device/non-cacheable, link buffers there | Best for high-throughput, set-and-forget |
| **Clean/invalidate cache around transfer** | `SCB_CleanDCache_by_Addr` before TX, `SCB_InvalidateDCache_by_Addr` after RX | When buffer count is bounded |
| **Disable D-cache entirely** | Don't | You lose 2-5× CPU throughput |

```c
// Before TX (memory → peripheral): push CPU writes out to RAM
SCB_CleanDCache_by_Addr((uint32_t*)tx_buf, sizeof(tx_buf));
HAL_SPI_Transmit_DMA(&hspi1, tx_buf, sizeof(tx_buf));

// After RX (peripheral → memory): drop stale cache lines
HAL_SPI_Receive_DMA(&hspi1, rx_buf, sizeof(rx_buf));
// ... wait for done ...
SCB_InvalidateDCache_by_Addr((uint32_t*)rx_buf, sizeof(rx_buf));
```

**Buffers must be cache-line aligned** (32 bytes on M7). Misalignment causes adjacent cached data to get invalidated too.

```c
__attribute__((aligned(32))) uint8_t rx_buf[256];
```

This is the #1 source of "my DMA works fine on F4 but is corrupted on H7" bugs.

## DMA on F4/F7/H7: FIFO Mode

These chips have a small per-stream FIFO (16 bytes). Two modes:

- **Direct mode**: FIFO disabled. Each peripheral request triggers one bus transaction.
- **FIFO mode**: DMA collects up to N items before bursting them. Lets the controller use AHB burst transactions → fewer bus cycles, less CPU contention.

FIFO mode also enables **data width conversion** (e.g., 8-bit peripheral → pack into 32-bit memory writes).

```c
LL_DMA_EnableFifoMode(DMA2, LL_DMA_STREAM_0);
LL_DMA_SetFIFOThreshold(DMA2, LL_DMA_STREAM_0, LL_DMA_FIFOTHRESHOLD_1_2);
LL_DMA_SetMemoryBurstxfer(DMA2, LL_DMA_STREAM_0, LL_DMA_MBURST_INC4);
LL_DMA_SetPeriphBurstxfer(DMA2, LL_DMA_STREAM_0, LL_DMA_PBURST_SINGLE);
```

Watch for **FIFO underrun/overrun errors** in the error flag — they indicate the burst settings don't match what the peripheral and memory can sustain.

## Scatter-Gather (Linked-List DMA)

Some controllers (STM32 H7 BDMA / MDMA, Nordic EasyDMA via PPI, NXP eDMA) support **transfer descriptors** in memory. Each descriptor points to the next, letting you queue many disjoint transfers as a single armed operation.

Use cases: building a complex frame from a header + payload + footer in separate buffers, or scrolling a partial framebuffer.

```c
// STM32 MDMA-style (conceptual)
typedef struct {
    uint32_t src;
    uint32_t dst;
    uint32_t count;
    void*    next;   // NULL = end of chain
} mdma_node_t;

mdma_node_t chain[3] = {
    { (uint32_t)header, (uint32_t)&SPI->DR, sizeof(header), &chain[1] },
    { (uint32_t)payload,(uint32_t)&SPI->DR, payload_len,    &chain[2] },
    { (uint32_t)footer, (uint32_t)&SPI->DR, sizeof(footer), NULL },
};
mdma_start(&chain[0]);
```

## Triggering DMA From Peripheral Events

The DMA channel doesn't run continuously — each "transfer" is gated by a request from the peripheral. The mapping of peripheral → channel/stream is fixed in hardware and lives in the reference manual.

| Chip family | Mapping |
|-------------|---------|
| STM32 F1 / L0 | One channel per request, fixed |
| STM32 F4 / F7 / H7 | Stream + Channel selects (DMA2 Stream 0 Channel 0 = ADC1) |
| STM32 G0 / G4 / WB | DMAMUX — *any* request to *any* channel |
| NXP RT / iMX | DMAMUX |
| Nordic nRF52 | EasyDMA + PPI shortcut, request is implicit |
| ESP32 | Per-peripheral DMA engines (SPI DMA, I2S DMA), little user mapping needed |

**DMAMUX** is increasingly common and makes life easier — you pick a channel and tell DMAMUX which request maps to it, instead of consulting a fixed table.

## DMA in RTOS Context

Two important interactions:

**1. Sleep during transfer.** Use a binary semaphore signalled from the TC interrupt to put the requesting task to sleep until the transfer finishes. Better than busy-waiting on a `done_flag`.

```c
SemaphoreHandle_t spi_done;

bool spi_transfer(uint8_t* tx, uint8_t* rx, size_t n) {
    HAL_SPI_TransmitReceive_DMA(&hspi1, tx, rx, n);
    return xSemaphoreTake(spi_done, pdMS_TO_TICKS(100)) == pdTRUE;
}

void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef* h) {
    BaseType_t hpw = pdFALSE;
    xSemaphoreGiveFromISR(spi_done, &hpw);
    portYIELD_FROM_ISR(hpw);
}
```

**2. Buffer ownership.** A buffer handed to DMA is **owned by hardware** until the transfer completes. Don't let other code touch it — and don't put it on the stack of a function that may return before DMA is done.

## Common Pitfalls

### Pitfall 1: Buffer on the Stack

```c
// BAD: rx_buf is freed when this function returns,
// but DMA is still writing to it
void read_sensor(void) {
    uint8_t rx_buf[64];
    HAL_SPI_Receive_DMA(&hspi1, rx_buf, sizeof(rx_buf));
    // returns immediately, rx_buf is now garbage memory
}

// GOOD: static or heap-allocated
void read_sensor(void) {
    static uint8_t rx_buf[64];
    HAL_SPI_Receive_DMA(&hspi1, rx_buf, sizeof(rx_buf));
}
```

### Pitfall 2: Forgetting to Enable the Peripheral's DMA Request

DMA controller is configured and armed but nothing happens. Cause: the peripheral itself doesn't have its DMA-request bit set (`USART_CR3_DMAR`, `SPI_CR2_TXDMAEN`, `ADC_CR2_DMA`, etc.).

### Pitfall 3: Wrong Channel/Stream Mapping

STM32 F4 has a fixed table: USART2_RX is DMA1 Stream 5 Channel 4. Picking any other combination silently does nothing — the request never reaches your channel.

### Pitfall 4: Memory-to-Memory With Increment Disabled on the Wrong Side

```c
// "Copy" that just hammers the first byte over and over
LL_DMA_SetMemoryIncMode(DMA1, ch, LL_DMA_MEMORY_NOINCREMENT);
```

For mem-to-mem you almost always want both source and destination to increment.

### Pitfall 5: Cache Coherency on M7

Already covered above, but worth repeating because it's so easy to miss. Symptom: works in debug build (cache disabled by some configs), fails at -O2.

### Pitfall 6: Mixing Synchronous and DMA on the Same Peripheral

```c
HAL_SPI_Transmit_DMA(&hspi1, big_buf, 4096);
HAL_SPI_Transmit(&hspi1, small_buf, 4);   // BAD: collides with ongoing DMA
```

The synchronous call sees the peripheral as busy and either blocks forever, returns error, or corrupts the in-flight transfer depending on HAL version.

### Pitfall 7: Burst Size Crossing a 1KB Boundary (F4)

STM32 F4 AHB has a quirk: burst transactions cannot cross a 1KB address boundary. The hardware enforces it by raising a transfer error. Align large buffers carefully.

### Pitfall 8: Re-arming Inside the TC ISR Without Clearing Flags

Forgetting to clear the TC flag → ISR re-enters immediately on exit. Same trap as any other peripheral.

## Performance Notes

- **DMA priority** matters when multiple channels compete for the bus. Set HIGH for time-sensitive streams (audio, ADC) and LOW for bulk copies.
- **Burst transfers** reduce per-transfer overhead. Always use FIFO + burst for high-throughput streams.
- **TCM (Tightly-Coupled Memory)** on M7: TCM is *not* on the AHB bus, so DMA from peripherals **cannot reach DTCM**. Place DMA buffers in regular SRAM, not DTCM.
- **Bus matrix contention**: CPU running from flash + DMA from RAM → low contention. CPU running from RAM + DMA to RAM → contention, CPU stalls.

## Summary

1. **DMA = peripheral ⇄ memory without CPU.**
2. **Configure once with src, dst, count, mode, trigger.**
3. **Circular + half/full IRQ** is the canonical streaming pattern.
4. **Double-buffer** for clean buffer-at-a-time processing.
5. **On M7, mind the D-cache** — non-cacheable region or clean/invalidate.
6. **Cache-line align DMA buffers** (32 bytes).
7. **Don't put DMA buffers on the stack** of a function that returns before completion.
8. **The peripheral must explicitly enable its DMA request** — DMA setup alone isn't enough.
9. **DMA buffers in DTCM don't work** — DMA controllers can't reach tightly-coupled memory.

## See Also

- [Interrupts](interrupts.md) — DMA completion handlers
- [ADC](adc.md), [SPI](spi.md), [UART](uart.md) — common DMA consumers
- [Power Management](power_management.md) — sleeping during DMA transfers
