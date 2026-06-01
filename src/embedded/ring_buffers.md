# Ring Buffers & Lock-Free Concurrency

## Overview

A ring buffer (circular buffer, FIFO) is the single most important data structure in
embedded firmware: a fixed-size array with a **head** (write) and **tail** (read) index
that wrap around, used to hand a stream of data from one context to another without
dynamic allocation. It is how a [UART](uart.md) RX [interrupt](interrupts.md) hands bytes
to your main loop, how a [DMA](dma.md) engine streams [ADC](adc.md) samples to a
processing task, and how the [event queue](state_machines.md) behind an event-driven
design is built. The reason it dominates is that with **one producer and one consumer**
it can be made **lock-free** — correct without disabling interrupts or taking a mutex —
which is exactly what you need when one side is an ISR that must not block.

```
   buffer[8], head=write, tail=read

   idx:   0    1    2    3    4    5    6    7
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
        │    │    │ D  │ A  │ T  │    │    │    │
        └────┴────┴────┴────┴────┴────┴────┴────┘
                   ▲              ▲
                  tail           head
                 (read)         (write)

   empty: head == tail
   push:  buffer[head] = x; head = (head + 1) & 7;
   pop:   x = buffer[tail]; tail = (tail + 1) & 7;
```

This page covers the mechanics, the lock-free single-producer/single-consumer (SPSC)
case, the memory-ordering rules that make it actually correct, and when you must fall
back to disabling interrupts or a [RTOS](state_machines.md)-style primitive. It pairs
with [Interrupts](interrupts.md), [DMA](dma.md), and the
[event-driven](state_machines.md) dispatcher.

## Full vs Empty: the Classic Ambiguity

With only `head` and `tail`, `head == tail` means both *empty* and *completely full* —
indistinguishable. Three standard fixes:

```
1. KEEP ONE SLOT OPEN          2. SEPARATE COUNT          3. FREE-RUNNING INDICES
   full when                      full when                  indices grow forever;
   (head+1)&mask == tail          count == size              size = head - tail
   wastes 1 slot, simplest        count shared by both       (unsigned wrap is fine);
                                  → needs protection         capacity must be pow2
```

The **keep-one-slot-open** scheme is the SPSC favorite: `count` would be written by both
sides (a shared mutable variable — a race), whereas head and tail each have a single
writer. The **free-running index** scheme (head/tail are `uint32_t` that only ever
increment, masked on array access) is elegant — `used = head - tail` works correctly even
across unsigned overflow — and is what Linux's `kfifo` and many RTOS queues use.

## Power-of-Two Masking

If the capacity is a power of two, wrap with a bitwise AND instead of a modulo or branch:

```c
#define SIZE 256                 // must be power of two
idx = (idx + 1) & (SIZE - 1);    // cheap: no divide, no compare-and-reset
```

`& (SIZE-1)` is a single-cycle instruction; `% SIZE` on a non-power-of-two may pull in a
software division routine on an MCU with no hardware divide. It also makes the
free-running-index trick work: a `uint32_t` index masked to the array size wraps exactly
at the `uint32_t` boundary because the buffer size divides 2³².

## SPSC Lock-Free Ring

The key fact: with **exactly one producer and one consumer**, each index has a single
writer. The producer owns `head`, the consumer owns `tail`; each only *reads* the other's
index. No two contexts write the same variable, so no lock is needed — *if* the
read/write ordering is enforced.

```c
typedef struct {
    uint8_t  buf[SIZE];
    volatile uint32_t head;   // written by producer only
    volatile uint32_t tail;   // written by consumer only
} ring_t;

// PRODUCER (e.g. UART ISR) — returns false if full
bool ring_push(ring_t *r, uint8_t x) {
    uint32_t h = r->head;
    uint32_t next = (h + 1) & (SIZE - 1);
    if (next == r->tail) return false;     // full
    r->buf[h] = x;                         // (1) write data
    __DMB();                               // (2) ensure data lands before index
    r->head = next;                        // (3) publish
    return true;
}

// CONSUMER (e.g. main loop)
bool ring_pop(ring_t *r, uint8_t *x) {
    uint32_t t = r->tail;
    if (t == r->head) return false;        // empty
    *x = r->buf[t];                        // read data
    __DMB();                               // data read before freeing slot
    r->tail = (t + 1) & (SIZE - 1);        // publish
    return true;
}
```

The producer must write the *payload* before publishing the new `head`; otherwise the
consumer can observe an advanced `head` and read a slot the producer hasn't filled yet.
That ordering is what the barrier guarantees.

## Memory Ordering: `volatile` Is Not Enough

`volatile` tells the *compiler* not to cache or reorder accesses to that variable — it
does **not** stop the *CPU or store buffer* from reordering, and it gives no guarantee
about the ordering of a `volatile` access relative to a *non-volatile* one (the data
write). On a single-core Cortex-M with a simple pipeline you often get away with it; on
[cache](cache_tcm.md)-equipped Cortex-M7, multi-core, or [DMA](dma.md)-shared memory you
do not.

```
   What you need:           data store  ──must-happen-before──►  index store
   `volatile` gives:        each access not optimized away  (says NOTHING about order)
   Barrier gives:           __DMB()  forces the ordering across the boundary
```

Use a **Data Memory Barrier** (`__DMB()`) between the payload access and the index
publish, or better, C11 atomics with explicit ordering:

```c
atomic_store_explicit(&r->head, next, memory_order_release);   // producer publish
uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);  // consumer read
```

`release`/`acquire` pair exactly to "everything the producer wrote before the release is
visible to the consumer after the acquire" — the precise contract a FIFO needs.

## When You Can't Stay Lock-Free

The lock-free property holds *only* for single-producer/single-consumer. The moment two
ISRs push, or two tasks pop, you have multiple writers to one index and must serialize:

```
SPSC (1 producer, 1 consumer)  ──►  lock-free, just barriers
MPSC / MPMC (multiple)         ──►  must protect head and/or tail
```

On a bare-metal MCU the cheapest protection is a **critical section** — briefly mask
interrupts around the index update:

```c
uint32_t primask = __get_PRIMASK();
__disable_irq();
/* ... update shared index ... */
__set_PRIMASK(primask);     // restore (don't blindly re-enable)
```

Keep it as short as humanly possible — masking interrupts inflates worst-case
[interrupt](interrupts.md) latency, which can wreck real-time guarantees. Under an RTOS,
prefer the kernel's queue/semaphore, which handles the multi-writer case and can block a
task instead of dropping data.

## DMA-Driven Rings

A [DMA](dma.md) controller in **circular mode** acts as a hardware producer writing
straight into the buffer with zero CPU involvement; the CPU is the consumer, learning the
current write position from the DMA's remaining-count register (`NDTR` on STM32). On a
[cached](cache_tcm.md) core you must **invalidate** the cache before reading DMA-filled
memory (or place the buffer in non-cacheable / TCM RAM), or the CPU reads stale lines —
the canonical DMA coherency bug.

## Where this connects

- [Interrupts](interrupts.md) — the producer is usually an ISR; the ring lets it hand data to the main loop without blocking.
- [State Machines & Event-Driven Firmware](state_machines.md) — the event queue is a ring buffer with a single dispatcher as consumer.
- [DMA](dma.md) — circular-mode DMA is a hardware producer; coherency rules apply on cached cores.
- [Cache & TCM](cache_tcm.md) — barriers and cache maintenance are what make shared rings correct on Cortex-M7.
- [UART](uart.md) — RX/TX FIFOs in software are the textbook ring-buffer use case.
- [Memory Management](memory_management.md) — fixed-size rings give bounded, fragmentation-free buffering without `malloc`.

## Pitfalls

1. **`volatile` mistaken for a barrier.** It stops compiler reordering only; the CPU/store
   buffer can still reorder. Use `__DMB()` or C11 acquire/release between data and index.
2. **Publishing the index before writing the data.** The consumer sees an advanced head
   and reads a not-yet-written slot. Always: write payload → barrier → advance index.
3. **Using a lock-free SPSC ring with two producers.** Two writers to `head` corrupt it.
   That case needs a critical section or a kernel queue.
4. **`count` variable shared by both sides.** A `count++`/`count--` from two contexts is a
   read-modify-write race. Derive fullness from head/tail, each single-writer.
5. **Non-power-of-two size with `&` masking.** `& (SIZE-1)` only wraps correctly when SIZE
   is a power of two; otherwise indices walk off the array.
6. **Critical section left too long.** Masking interrupts around heavy work blows interrupt
   latency. Protect only the few index instructions.
7. **Cached DMA buffer not invalidated.** CPU reads stale cache lines instead of fresh DMA
   data. Invalidate before reading, or use non-cacheable/[TCM](cache_tcm.md) memory.
8. **Overflow policy left undefined.** Decide explicitly: drop newest, overwrite oldest, or
   apply backpressure. Silent overwrite corrupts a stream consumer mid-message.

## See Also

- [Interrupts](interrupts.md) — the producer context
- [State Machines & Event-Driven Firmware](state_machines.md) — the event queue built on a ring
- [DMA](dma.md) — hardware-driven circular buffers and coherency
- [Cache & TCM](cache_tcm.md) — barriers and cache maintenance for shared memory
