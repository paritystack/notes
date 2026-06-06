# MISRA C & Defensive Firmware

## Overview

Firmware fails differently from application software: a null-pointer bug doesn't pop a stack trace, it bricks a device in someone's wall or a car's brake controller. **MISRA C** and the broader practice of **defensive firmware** exist to push errors as far "left" as possible — caught by the compiler, a static analyzer, or an assertion at the developer's desk, rather than as a [HardFault](hardfault_debugging.md) in the field. This page covers the coding-standard landscape (MISRA C and friends), the C-language pitfalls that bite embedded code specifically (undefined behavior, `volatile`, integer promotion), and the defensive patterns — assertions, parameter validation, fault handling — that make firmware survivable. It complements the [debugging](debugging.md) and [watchdog](watchdog.md) pages: those recover from failures, this prevents them.

```
   Where a bug can be caught (cheapest → most expensive)
   ┌──────────┬──────────┬───────────┬──────────┬───────────┐
   │ Compiler │ Static   │ Assertion │ Test /   │ FIELD     │
   │ -Wall    │ analyzer │ at desk   │ HIL rig  │ (brick)   │
   │ -Werror  │ (MISRA)  │           │          │           │
   └──────────┴──────────┴───────────┴──────────┴───────────┘
   $                                                      $$$$$$
   ← defensive firmware shifts detection leftward →
```

## Coding Standards

| Standard | Domain | Nature |
|----------|--------|--------|
| **MISRA C** | Automotive, then everywhere | Rules restricting C to a safer subset |
| **CERT C** | Security | Rules to avoid exploitable bugs |
| **MISRA C++ / AUTOSAR C++** | C++ in safety-critical | C++ subset rules |
| **Barr Group Embedded C** | General embedded | Style + safety |
| **DO-178C / IEC 61508 / ISO 26262** | Avionics / functional safety / automotive | Process standards that *require* a coding standard |

MISRA C is the dominant one. It classifies rules as **Mandatory** (never break), **Required** (break only with documented justification — a "deviation"), and **Advisory**. The point isn't bureaucracy; it's removing C's many footguns from the toolbox.

### Representative MISRA rules

- **No dynamic memory after init** (`malloc`/`free` banned) — see [Memory Management](memory_management.md).
- **No recursion** — unbounded stack growth.
- **Single exit / explicit returns**, no fall-through in `switch` without comment.
- **No implicit type conversions** that lose data; explicit casts only.
- **Braces on every `if`/`for`/`while`**, even single statements (the "goto fail" class of bug).
- **No `goto`** (with narrow exceptions).
- **Every `switch` has a `default`.**
- **Functions have a single, documented return type; check all return values.**

You don't memorize these — a **static analyzer** (PC-lint Plus, Polyspace, Coverity, cppcheck with the MISRA addon, clang-tidy) enforces them in CI.

## C Pitfalls That Bite Embedded Code

### `volatile`

The most misunderstood keyword in embedded C. The compiler assumes memory doesn't change unless *it* changes it — so it caches values in registers and optimizes away "redundant" reads. That assumption is **false** for hardware registers, [ISR](interrupts.md)-modified globals, and memory-mapped peripherals.

```c
// BUG: compiler reads STATUS once, may loop forever
while ((UART->STATUS & TXEMPTY) == 0) { }

// FIX: volatile forces a fresh read each iteration
volatile uint32_t *status = &UART->STATUS;
while ((*status & TXEMPTY) == 0) { }

// A flag set in an ISR MUST be volatile, or the main loop never sees the change
volatile bool data_ready = false;
```

But `volatile` is **not** atomicity and **not** a memory barrier — for multi-byte values shared with an ISR you still need to disable interrupts or use atomics. See [Interrupts](interrupts.md).

### Undefined behavior

UB lets the optimizer do anything — including deleting your safety checks. The embedded-relevant offenders:

- **Signed integer overflow** is UB; the compiler may assume `x + 1 > x` always and remove an overflow check.
- **Unaligned access** faults on Cortex-M0/many cores ([HardFault](hardfault_debugging.md)).
- **Reading uninitialized memory** — and trusting `.bss` is zero only if [startup](startup_code.md) zeroed it.
- **Strict-aliasing violations** (type-punning through incompatible pointers) — use `memcpy` or a `union`, not pointer casts.

### Integer promotion & width

```c
uint8_t a = 200, b = 100;
if (a + b > 255) { ... }   // a+b promotes to int = 300, condition TRUE
                            // — surprising if you expected uint8 wrap
uint16_t timeout = 0xFFFF;
timeout++;                  // wraps to 0 — was that intended?
```

Use the fixed-width types from `<stdint.h>` (`uint8_t`, `int32_t`) everywhere, and be explicit about where promotion happens.

## Defensive Patterns

### Assertions

Catch "impossible" states at the point they occur, with the file/line, instead of corrupting onward:

```c
#define ASSERT(x)  do { if (!(x)) assert_failed(__FILE__, __LINE__); } while (0)

void assert_failed(const char *file, int line) {
    __disable_irq();
    log_fault(file, line);     // to RTT / flash / UART — see ../embedded/rtt_semihosting.md
    while (1) { /* or controlled reset via watchdog */ }
}
```

Keep asserts enabled in development; in production decide deliberately whether to keep them (safer) or compile out the cheap ones — but **never put side effects inside `ASSERT()`** (they vanish when compiled out).

### Validate at boundaries

```c
int sensor_read(uint8_t channel, uint16_t *out) {
    if (channel >= NUM_CHANNELS) return -EINVAL;   // reject bad input
    if (out == NULL)             return -EINVAL;
    ...
    return 0;
}
```

Check every public function's parameters and **every return value** — especially from HAL calls, bus transactions, and [flash](flash_filesystems.md) operations that routinely fail in the field.

### Fail safe, and recover

- Drive outputs to a **safe state** on any detected fault (motors off, valves closed) before halting.
- Use the [watchdog](watchdog.md) so a hung task forces a clean reset rather than an indefinite freeze.
- Decode and log faults via the [HardFault handler](hardfault_debugging.md) so a field failure leaves evidence.
- Make state machines explicit with a `default`/else that traps unexpected states instead of silently continuing.

## Where this connects

- [Debugging](debugging.md) — defensive code makes bugs reproducible and located; this is the prevention half.
- [HardFault Debugging](hardfault_debugging.md) — assertions and fault handlers turn corruption into diagnosable traps.
- [Watchdog](watchdog.md) — the last-resort recovery when defensive checks still don't prevent a hang.
- [Memory Management](memory_management.md) — the "no dynamic allocation" MISRA rule and stack-overflow defenses.
- [Interrupts](interrupts.md) — `volatile`, atomicity, and ISR-shared-data rules.
- [Startup Code](startup_code.md) — the `.bss`-zeroing assumption defensive code relies on.
- [Testing](../testing/index.html) — unit tests, static analysis, and HIL rigs that enforce the standard in CI.

## Pitfalls

1. **Missing `volatile` on hardware/ISR variables.** The compiler caches the value; a poll loop hangs or an ISR flag is never seen. The classic embedded bug.
2. **Treating `volatile` as atomic.** It guarantees fresh reads, not indivisible access. Multi-byte ISR-shared data still needs interrupt masking/atomics.
3. **Relying on undefined behavior.** Signed overflow checks get optimized away; type-punning breaks under `-O2`. Stay in defined C.
4. **Side effects inside `assert`.** When asserts compile out, the side effect vanishes and behavior changes. Keep asserts pure.
5. **Ignoring return values.** Unchecked HAL/bus/flash failures propagate silently. Check every one.
6. **Mixed/implicit integer widths.** Promotion and truncation surprises. Use `<stdint.h>` types and explicit casts.
7. **Adopting MISRA without tooling.** Hand-checking a coding standard doesn't scale and misses cases. Enforce it with a static analyzer in CI.
8. **Halting without failing safe.** Spinning in a fault handler while outputs stay energized is dangerous; drive to a safe state first.

## See Also

- [Debugging](debugging.md) — finding bugs that slipped through
- [HardFault Debugging](hardfault_debugging.md) — fault handlers and crash logging
- [Watchdog](watchdog.md) — recovery of last resort
- [Testing](../testing/index.html) — CI enforcement of the standard
