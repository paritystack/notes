# Embedded Memory Management

## Overview

Memory management on a microcontroller is a different discipline from desktop programming: there is **no virtual memory, no swap, no OOM killer, and often no good reason to call `malloc` at all**. You have a few KB to a few hundred KB of SRAM that must hold the stack, the heap, and all your globals simultaneously, and running out doesn't print an error — it silently corrupts an adjacent variable or overflows into the [vector table](startup_code.md). This page is about the patterns that make limited RAM safe and deterministic: static allocation, memory pools, careful stack sizing, fragmentation avoidance, and using the [MPU](mpu.md) to catch overflows. It builds directly on the regions the [startup code](startup_code.md) and [linker script](linker_scripts.md) lay down.

```
   SRAM (grows from both ends toward the middle)
   ┌────────────────────┐ high address (e.g. 0x2002_0000)
   │   STACK            │ ← SP, grows DOWN
   │     │              │
   │     ▼              │
   │   (free gap)       │ ← collision here = corruption, no warning
   │     ▲              │
   │     │              │
   │   HEAP             │ ← grows UP (if you use malloc)
   ├────────────────────┤
   │   .bss  (zeroed)   │ uninitialized globals
   │   .data (copied)   │ initialized globals
   └────────────────────┘ low address (0x2000_0000)
```

## Why `malloc` Is Suspect Here

Dynamic allocation brings four problems that barely matter on a PC but are serious on an MCU:

| Problem | Consequence on an MCU |
|---------|------------------------|
| **Fragmentation** | Free memory exists but in pieces too small to satisfy a request → allocation fails even though "there's room" |
| **Non-determinism** | `malloc` time varies with heap state → breaks real-time guarantees |
| **No failure recovery** | What does a sensor node *do* when `malloc` returns NULL mid-flight? Usually nothing safe |
| **Silent heap/stack collision** | Heap grows up, stack grows down; nothing stops them meeting |

Safety-critical standards like [MISRA](coding_standards.md) and automotive/aerospace guidelines often **ban dynamic allocation after initialization** entirely. The mainstream embedded philosophy: allocate everything you'll ever need at startup, then never call `malloc` again.

## The Allocation Strategies

### Static allocation

The simplest and safest: everything is a global or `static`, sized at compile time. Memory use is known from the map file; nothing can fail at runtime.

```c
static uint8_t rx_buffer[256];       // exists for the whole program
static sensor_t sensors[MAX_NODES];  // fixed fleet size
```

### Memory pools (fixed-block allocators)

When you *do* need allocate/free dynamics (network buffers, message objects), use a **pool**: a pre-allocated array of fixed-size blocks with a free-list. Allocation is O(1), deterministic, and **cannot fragment** because every block is identical.

```c
typedef struct block { struct block *next; } block_t;
static uint8_t  pool[N][BLOCK_SIZE];
static block_t *free_list;

void pool_init(void) {
    free_list = NULL;
    for (int i = 0; i < N; i++) {          // thread all blocks onto free list
        ((block_t*)pool[i])->next = free_list;
        free_list = (block_t*)pool[i];
    }
}
void *pool_alloc(void) {                    // O(1), never fragments
    block_t *b = free_list;
    if (b) free_list = b->next;
    return b;                               // NULL if exhausted
}
void pool_free(void *p) {
    ((block_t*)p)->next = free_list;
    free_list = p;
}
```

[RTOS](../rtos/freertos.md) kernels provide exactly this (FreeRTOS heap_1..heap_5, message buffers, `xMessageBufferCreateStatic`). `heap_1` (allocate-only, no free) and pools are the deterministic choices; general `heap_4` coalesces but can still fragment.

### Arena / bump allocator

For "allocate a lot during init, free it all at once" phases: hand out memory by bumping a pointer, then reset the pointer to free everything in one shot. No per-object free, no fragmentation.

## Sizing the Stack

The stack is the silent killer. Each function call pushes locals, saved registers, and (on an interrupt) the exception frame; deep call chains, big local arrays, recursion, and [ISRs](interrupts.md) nesting all add up. The total must fit in the gap above the heap/globals.

How to size it honestly:
- **Fill-and-check (watermarking):** paint the stack region with a known pattern (`0xDEADBEEF`) at [startup](startup_code.md), run worst-case workloads, then measure how far the pattern was overwritten. The high-water mark is your real peak usage; size with margin.
- **Static analysis:** tools (`-fstack-usage`, GCC's `.su` files, worst-case stack analyzers) compute the deepest call chain — but can't see through function pointers or recursion.
- **Account for the worst ISR nesting** on top of the deepest task stack.

In an [RTOS](../rtos/freertos.md) each task has its *own* stack; `uxTaskGetStackHighWaterMark()` reports the watermark per task.

## Catching Overflow in Hardware

Software watermarking detects overflow *after* it happened. To catch it *at the moment of corruption*, use the [MPU](mpu.md):

```
   ┌──────────────┐  task stack top
   │  task stack  │
   │      │       │
   │      ▼       │
   ├──────────────┤
   │  MPU region: │  ← 32-byte No-access guard
   │  NO ACCESS   │     overflow push → instant MemManage fault
   ├──────────────┤
   │  next task   │
```

The overflowing push hits the guard region and traps to a [MemManage handler](hardfault_debugging.md) with the faulting address, instead of silently smashing the neighbouring task's data. This is the single best technique for turning "mysterious corruption that appears hours later" into an immediate, located fault.

## Where this connects

- [Linker Scripts](linker_scripts.md) — defines the heap and stack regions, and ideally an assertion that they don't overlap.
- [Startup Code](startup_code.md) — establishes `.data`/`.bss`/stack/heap before `main`; the place to paint the stack for watermarking.
- [MPU](mpu.md) — guard regions catch stack overflow and isolate pools per task at the moment of violation.
- [HardFault Debugging](hardfault_debugging.md) — overflow into the guard or into peripherals surfaces here; CFSR/MMFAR localizes it.
- [FreeRTOS](../rtos/freertos.md) — per-task stacks, static vs heap allocation schemes, message buffers and pools.
- [MISRA & Defensive Firmware](coding_standards.md) — the rules that restrict dynamic allocation in safety-critical code.

## Pitfalls

1. **Heap/stack collision with no detection.** They grow toward each other; nothing stops the meeting. Add a linker assertion and/or an MPU guard.
2. **`malloc` in long-running code.** Fragmentation eventually fails an allocation even with free RAM. Allocate at init, or use fixed-block pools.
3. **Under-sized stack found in the field.** Worst-case path (deep ISR nesting + big local array) only happens occasionally. Watermark under stress, don't guess.
4. **Big arrays/structs as locals.** A `uint8_t buf[2048]` local silently consumes 2 KB of stack per call. Make large buffers `static` or pool-allocated.
5. **Ignoring `malloc` return value.** NULL deref or, worse, writing to address 0. Always check; better, avoid `malloc`.
6. **Recursion on an MCU.** Unbounded stack growth. Convert to iteration with an explicit bounded stack.
7. **Forgetting ISR stack cost.** On many cores interrupts use the same stack (or MSP); their frames stack on top of yours. Budget for them.
8. **Assuming `.bss` is zero without startup zeroing it.** See [Startup Code](startup_code.md) — uninitialized statics are only zero if the runtime zeroed `.bss`.

## See Also

- [Linker Scripts](linker_scripts.md) — region layout and overlap checks
- [Startup Code](startup_code.md) — runtime memory model setup
- [MPU](mpu.md) — hardware overflow guards
- [FreeRTOS](../rtos/freertos.md) — per-task stacks and allocators
