# Embedded Unit Testing

## Overview

Testing firmware is harder than testing a web app for one reason: the code is entangled
with hardware that doesn't exist on your build machine. The breakthrough technique is to
**run most of your logic on the host PC**, compiled with a normal compiler, with the
hardware faked out — so tests run in milliseconds in CI instead of needing a board on
every developer's desk. What can't be faked (timing, real peripherals, electrical
behavior) is covered separately by **on-target** and **hardware-in-the-loop** tests. This
page covers the toolchain that dominates C firmware testing — **Unity / CMock / Ceedling**
— and the architectural discipline (dependency inversion at the [HAL](cmsis.md) boundary)
that makes host testing possible.

```
   TEST PYRAMID FOR FIRMWARE
        ▲ few    ┌─────────────────────────┐
        │        │  HIL / system  (real     │  slow, real I/O, hardware rig
        │        │  board + stimulus rig)   │
        │        ├─────────────────────────┤
        │        │  on-target integration   │  runs ON the MCU, real peripherals
        │        ├─────────────────────────┤
        ▼ many   │  host unit tests         │  run on PC, hardware MOCKED — fast
                 └─────────────────────────┘
```

It pairs tightly with [MISRA C & Defensive Firmware](coding_standards.md) (testable code
is well-structured code), [Build Systems](build_systems.md) (tests are a second build
target), and [Debugging](debugging.md) / [GDB for Embedded](gdb_embedded.md).

## Host vs Target

The core split, and when to use each:

| | **Host (off-target)** | **On-target** | **HIL** |
|---|---|---|---|
| Runs on | your PC (native compiler) | the MCU itself | board + external rig |
| Speed | milliseconds | seconds (flash + run) | slow, physical |
| What it proves | logic, algorithms, protocol parsing, [state machines](state_machines.md) | drivers against *real* peripherals, timing, ISR behavior | end-to-end with real sensors/signals |
| Hardware deps | none (mocked) | the chip | the chip + stimulus |
| In CI? | yes, every commit | with a board farm | nightly / pre-release |

The strategy: push **as much logic as possible** into hardware-independent modules tested
on the host, and keep the truly hardware-touching layer thin so the small on-target/HIL
suites can focus there. This is the [ISR](interrupts.md)-stays-thin rule applied to
testability.

## Unity, CMock, Ceedling

The standard open-source stack for C (from ThrowTheSwitch):

- **Unity** — a tiny C unit-test framework: assertion macros (`TEST_ASSERT_EQUAL`,
  `TEST_ASSERT_EQUAL_HEX8`, `TEST_ASSERT_EQUAL_MEMORY`) and a runner. Pure C, compiles
  anywhere, including *onto the target* for on-chip tests.
- **CMock** — auto-generates **mock** implementations from a header. Point it at
  `i2c_hal.h` and it emits a fake `i2c_hal.c` you can program per-test: set return values,
  assert it was called with expected arguments, enforce call order.
- **Ceedling** — the build/orchestration layer (Ruby/Rake) that wires Unity + CMock
  together, discovers tests, generates runners, and reports results.

```c
// Production code depends on an INTERFACE, not the chip:
//   sensor.c calls i2c_read() declared in i2c_hal.h

#include "unity.h"
#include "mock_i2c_hal.h"     // CMock-generated fake of i2c_hal.h
#include "sensor.h"

void test_sensor_reports_celsius_from_raw_register(void) {
    // Arrange: program the mock to return a known raw reading
    uint8_t raw[2] = { 0x19, 0x00 };       // 25.0 °C raw value
    i2c_read_ExpectAndReturn(0x48, NULL, 2, I2C_OK);
    i2c_read_IgnoreArg_buf();
    i2c_read_ReturnArrayThruPtr_buf(raw, 2);

    // Act
    float t = sensor_read_temperature();

    // Assert: pure logic, no hardware involved
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 25.0f, t);
}
```

The test runs on the host in microseconds and never touches an I2C bus — it verifies the
*conversion logic*, the part most likely to have bugs.

## Designing for Testability: Seams

You can only mock what you can substitute. The enabling pattern is a **seam** — an
interface boundary where the real implementation can be swapped for a fake. In C this is
usually a header of function prototypes (a thin [HAL](cmsis.md)) that production code
calls and tests replace:

```
   sensor.c  ──calls──►  i2c_hal.h  ◄──implemented by──┐
                          (the seam)                    ├─ i2c_hal_stm32.c   (on target)
                                                        └─ mock_i2c_hal.c    (in host test)
```

Two common seam styles:
- **Link-time substitution** — compile `sensor.c` against the mock object in the test
  build, against the real driver in the firmware build. Simple; what Ceedling does.
- **Function pointers / dependency injection** — pass the dependency in, swap at runtime.
  More flexible, slight overhead — weigh against [MISRA](coding_standards.md) preferences.

Code that reads a global register directly has *no seam* and can't be host-tested — which
is itself a design smell the test pressure helpfully exposes.

## Faking Registers and Time

For code that *does* poke registers, you can still host-test by backing the register
addresses with plain RAM in the test build:

```c
// Production: #define UART_DR  (*(volatile uint32_t*)0x40004400)
// Test:       a real uint32_t the test can read/write to observe driver behavior
uint32_t fake_uart_dr;
#define UART_DR  fake_uart_dr
```

**Time** is the other thing to fake: never let tests call a real delay or read a real
[timer](timers.md). Inject a clock/tick function so a test can advance time instantly and
deterministically — essential for testing timeouts in [state machines](state_machines.md)
and debounce logic without waiting in real life.

## On-Target & Hardware-in-the-Loop

Some things only the silicon can tell you: real [interrupt](interrupts.md) latency,
[DMA](dma.md) behavior, peripheral quirks, electrical timing. For these:

- **On-target unit tests** — the same Unity tests cross-compiled and run on the MCU,
  reporting results back over [UART](uart.md), [RTT/semihosting](rtt_semihosting.md), or a
  [GDB](gdb_embedded.md) script. Slower, but exercises the real driver.
- **Hardware-in-the-loop (HIL)** — the board under test is driven by a rig (signal
  generators, a second MCU, a logic analyzer, a CAN/LIN bus simulator) that injects
  stimulus and checks outputs end-to-end. Catches integration and timing faults host tests
  can't, at the cost of a physical setup. Often automated nightly with a small **board
  farm** in CI.

## CI for Firmware

```
   git push
      │
      ▼
   ┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐
   │ host tests   │──►│ build firmware    │──►│ (optional) flash to │
   │ (Ceedling):  │   │ image + static    │   │ board farm, run     │
   │ fast, gating │   │ analysis (MISRA,  │   │ on-target/HIL suite │
   │              │   │ cppcheck)         │   │ nightly             │
   └──────────────┘   └──────────────────┘   └─────────────────────┘
```

Host tests gate every commit because they're fast and need no hardware; the on-target/HIL
tier runs where boards are available. Combine with the static-analysis gate from
[MISRA C & Defensive Firmware](coding_standards.md) — testing and static analysis catch
different bug classes.

## Where this connects

- [MISRA C & Defensive Firmware](coding_standards.md) — testable, seam-based code is well-structured code; static analysis complements tests.
- [State Machines & Event-Driven Firmware](state_machines.md) — pure, hardware-independent FSMs are the easiest and highest-value things to host-test (inject fake time for timeouts).
- [Build Systems](build_systems.md) — tests are a second build target (native compiler) alongside the firmware image.
- [Debugging](debugging.md) / [GDB for Embedded](gdb_embedded.md) — what you fall back to when an on-target test fails.
- [RTT & Semihosting](rtt_semihosting.md) — how on-target test results get reported back to the host.
- [CMSIS](cmsis.md) — the HAL boundary is the natural seam to mock.

## Pitfalls

1. **No seam at the hardware boundary.** Code that reads registers/globals directly can't be
   substituted, so it can't be host-tested. Introduce a thin HAL interface to mock against.
2. **Testing the mock instead of the code.** Over-specified mocks (asserting every internal
   call) make tests brittle and tautological. Assert observable behavior and key interactions.
3. **Real delays in tests.** A `HAL_Delay()` in a unit test wastes wall-clock and makes
   timeout tests flaky. Inject a fake clock and advance it explicitly.
4. **Host pass ≠ target pass.** Type widths, endianness, alignment, and `volatile`/timing
   behavior differ; never skip the on-target tier for hardware-touching code.
5. **Mocking what you should fake (and vice-versa).** Use generated mocks for *interactions*
   (was I2C called correctly?), plain RAM-backed fakes for *registers* (what did the driver
   write?). Mixing them up makes tests confusing.
6. **No CI gate.** Tests that aren't run on every commit rot. Wire host tests into CI so they
   actually block regressions.
7. **Ignoring coverage of error paths.** The happy path is easy; the bus-error, timeout, and
   CRC-fail branches are where firmware bugs hide — mock the failures explicitly.

## See Also

- [MISRA C & Defensive Firmware](coding_standards.md) — the static-analysis half of quality
- [State Machines & Event-Driven Firmware](state_machines.md) — the most testable firmware pattern
- [Build Systems](build_systems.md) — adding a host test target
