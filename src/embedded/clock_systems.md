# Clock Systems

## Overview

Every digital block on an MCU needs a clock. The chip starts with a coarse internal oscillator at reset, then your code typically routes a higher-quality source through a **PLL** to generate the system clock, then derives bus clocks and peripheral clocks from there. Getting the clock tree right is usually the **first thing main() does** — peripherals that look broken often boil down to wrong clock config.

```
External crystal ──► HSE oscillator ──┐
                                       ├──► PLL ──► SYSCLK ──► AHB ──► APB1
Internal RC      ──► HSI oscillator ──┘                                ├──► APB2
                                                                       └──► peripherals
```

A clock bug usually manifests as: peripheral runs N× too fast/slow, UART bauds are wrong, timers count at unexpected rates, FLASH wait states wrong → random hardfaults.

## Sources

| Source | Typical | Accuracy | Use |
|--------|---------|----------|-----|
| **HSI** (High-Speed Internal RC) | 8–64 MHz | ±1–2% | Default at reset. Good enough for most peripherals, **not** for USB or CAN. |
| **HSE** (High-Speed External) | 4–48 MHz crystal | ±10–50 ppm | Required for USB FS (48 MHz multiple) and most precision needs. |
| **LSI** (Low-Speed Internal RC) | ~32 kHz | ±5–10% | Watchdog, low-power RTC when accuracy doesn't matter. |
| **LSE** (Low-Speed External) | 32.768 kHz crystal | ±20–100 ppm | RTC, real-time clock applications. |
| **MSI** (Multi-Speed Internal, STM32L) | 100 kHz – 48 MHz | ±0.5–3% | Low-power: trim frequency to minimum needed. |

**32.768 kHz** is the magic RTC frequency because it's 2^15 — divides cleanly to 1 Hz with a binary counter.

## Crystals 101

A crystal is a piezoelectric resonator. To turn it into an oscillator you pair it with the on-chip inverter amplifier and two **load capacitors** (`C_L`).

```
       MCU pin OSC_IN ────┬───────[ crystal ]────────┬──── OSC_OUT pin
                          │                          │
                          ├── C1 ──┐         ┌── C2 ─┤
                          │        │         │       │
                          │       GND       GND      │
                          │                          │
                  [ internal feedback resistor and amplifier ]
```

### Load Capacitance

The crystal's datasheet specifies `C_L`, e.g., 12 pF. The capacitors C1, C2 plus the PCB parasitics must sum to that:

```
C_L = (C1 × C2) / (C1 + C2)  +  C_stray
```

For matched C1 = C2 = C: `C = 2 × (C_L − C_stray)`. Stray capacitance from traces is typically 2–5 pF.

Wrong load cap → either won't start, starts unreliably, or oscillates off-frequency.

### ESR and Drive Level

- **ESR** (equivalent series resistance) — too high → oscillator won't start, especially at low temperatures. Pick crystals with ESR ≪ chip's max-recommended.
- **Drive level** — power dissipated in the crystal. Too high damages low-ESR crystals; the inverter has gain-control registers (`HSEDRV` on STM32G/H) to back off if needed.

### Layout Tips

- Keep traces short (< 1 cm if possible).
- Guard ring of GND around OSC pins.
- No high-speed signals nearby.
- Don't route under the crystal.

A flaky 32.768 kHz oscillator is almost always layout — these crystals have very low drive levels and pick up noise easily.

## PLL Basics

A PLL multiplies a reference clock by a (often fractional) ratio to produce a higher frequency. Conceptually:

```
                 ┌─ /M ─┐                 ┌── ÷P ──► SYSCLK
   ref ────────►│       ├──► PFD ─► VCO ──┤
                 └─ × N ┘                 ├── ÷Q ──► 48 MHz domain (USB)
                                          └── ÷R ──► other
```

- **M (or PLLM)**: pre-divider on the reference. PFD frequency = `f_ref / M`. Datasheet requires this to be in a specific range (typical 1–2 MHz).
- **N (or PLLN)**: VCO multiplier. VCO = `(f_ref / M) × N`. Must land within VCO range (typical 100–432 MHz).
- **P, Q, R**: post-dividers to derive different output rails.

### Example: STM32F4 to 168 MHz from 8 MHz HSE

```
HSE = 8 MHz
M = 8   → PFD = 1 MHz
N = 336 → VCO = 336 MHz
P = 2   → SYSCLK = 168 MHz
Q = 7   → 48 MHz for USB
```

```c
RCC->CR |= RCC_CR_HSEON;
while (!(RCC->CR & RCC_CR_HSERDY)) {}

// Configure PLL: source = HSE, M=8, N=336, P=2, Q=7
RCC->PLLCFGR = (8 << 0)            // M
             | (336 << 6)          // N
             | (0 << 16)           // P = 2  (encoded 0b00)
             | (RCC_PLLCFGR_PLLSRC_HSE)
             | (7 << 24);          // Q

RCC->CR |= RCC_CR_PLLON;
while (!(RCC->CR & RCC_CR_PLLRDY)) {}

// Bump flash latency BEFORE switching SYSCLK
FLASH->ACR = FLASH_ACR_LATENCY_5WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;

// Switch SYSCLK to PLL
RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
```

The **flash wait states** are critical: as SYSCLK rises, flash can't keep up at zero wait states. Wrong setting → hardfaults trying to fetch instructions. Always raise wait states **before** switching to a faster clock; lower them **after** switching to a slower one.

## The Clock Tree

After SYSCLK, prescalers feed:

```
SYSCLK ──► HPRE ──► HCLK (AHB, CPU, M3/M4 core)
                       │
                       ├──► PPRE1 ──► PCLK1 (APB1: USART2/3, I2C, TIM2-7)
                       │                  │
                       │                  └──► (if APB1 prescaler ≠ 1)
                       │                       TIMxCLK = PCLK1 × 2
                       │
                       └──► PPRE2 ──► PCLK2 (APB2: USART1/6, SPI1, TIM1/8)
                                          └──► same TIMx × 2 rule
```

A few things to remember:
- **Each peripheral has a clock-enable bit** in `RCC_AHBxENR` / `RCC_APBxENR`. Forgetting to enable it = peripheral registers read as zero, writes ignored. Most common embedded gotcha.
- **APB1 is usually slower than APB2** (peripheral I/O on slow bus). USART1/6 are on the fast bus, USART2/3 on the slow bus — affects max baud rates.
- **Timer clock doubling**: if APB prescaler ≠ 1, the timer block gets `PCLK × 2`. This compensates for the lower bus speed so timers can still hit nominal rates.

A common bug: computing a UART baud divider using SYSCLK when the peripheral actually clocks from PCLK1. Always read the reference manual's "Clock distribution" diagram.

## Peripheral Clock Selection (Mux)

Modern STM32 families (L4, G0, G4, H7, U5) let you pick the clock per peripheral via `RCC_CCIPR`. USART1 could be on PCLK2, SYSCLK, HSI16, or LSE — each gives different tradeoffs:

| Choice | Why |
|--------|-----|
| **PCLK** | Default, no extra setup |
| **SYSCLK** | Keep baud constant across APB prescaler changes |
| **HSI16** | Survive STOP mode (HSI keeps running) |
| **LSE** | Sub-9600 baud with extreme low power |

Use HSI/LSE source for peripherals you want to work in low-power modes.

## MCO: Outputting a Clock for Debug

The MCO (Microcontroller Clock Output) pin lets you route an internal clock to a pin to verify on a scope. Invaluable when debugging "my SYSCLK isn't what I think it is".

```c
// STM32F4: route SYSCLK / 4 to PA8 (MCO1)
RCC->CFGR |= (3 << 21);   // MCO1SEL = PLL/SYSCLK
RCC->CFGR |= (6 << 24);   // MCO1PRE = /4
GPIOA->MODER  |= (2 << 16);   // PA8 AF
GPIOA->AFR[1] |= (0 <<  0);   // AF0 = MCO
```

Probe PA8 → you should see SYSCLK / 4. If it's wrong, your clock config is wrong.

## Clock Security System (CSS)

CSS watches HSE and triggers an NMI if it fails (broken crystal, missing oscillation). On NMI, the firmware should switch back to HSI and degrade gracefully. Critical for safety-critical / automotive use.

```c
RCC->CR |= RCC_CR_CSSON;

void NMI_Handler(void) {
    if (RCC->CIR & RCC_CIR_CSSF) {
        RCC->CIR |= RCC_CIR_CSSC;
        // HSE died; fall back to HSI-only PLL
        switch_to_safe_clocks();
    }
}
```

## Low-Power Considerations

Active power scales linearly with clock frequency (CV²f). Strategies:

- **Run slower when you can.** Drop SYSCLK to 16 MHz between bursts.
- **Use HSI instead of HSE** between bursts (no crystal startup time, fast wake).
- **Stop the PLL** in sleep modes when not needed.
- **STM32L MSI** can be range-trimmed (100 kHz – 48 MHz) without re-stabilizing — ideal for adaptive clocking.
- **Peripheral clock gating** (`RCC_AHBxENR` bits) — disable clocks for unused peripherals to save µA.

```c
// Going to sleep: drop SYSCLK to HSI
RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_HSI;
RCC->CR  &= ~RCC_CR_PLLON;

// Wake up: spin PLL back up
RCC->CR  |=  RCC_CR_PLLON;
while (!(RCC->CR & RCC_CR_PLLRDY)) {}
RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
```

PLL lock time is typically 100–200 µs — not free, factor into wake latency budgets.

## SysTick: The Easy Tick

ARM Cortex-M provides a standard **SysTick** timer separate from MCU peripherals: 24-bit down-counter, ticks at HCLK (or HCLK/8).

```c
SysTick_Config(SystemCoreClock / 1000);   // 1 ms tick
```

`SystemCoreClock` is a global the CMSIS startup populates from your clock config; **keep it in sync** with what you actually set up, or `HAL_Delay` and friends drift.

```c
extern uint32_t SystemCoreClock;
SystemCoreClock = 168000000UL;   // after you switched SYSCLK
SystemCoreClockUpdate();         // CMSIS helper that recomputes from RCC regs
```

## Calibrating LSI / HSI

Internal RCs drift with temperature and supply voltage. Two ways to calibrate at runtime:

- **Trim against a known external reference.** STM32 lets you adjust HSITRIM to fine-tune HSI16 frequency (typically by counting HSI cycles against a known LSE crystal).
- **TIM5 channel 4 → LSI input** on STM32F4 lets you measure LSI frequency exactly. Useful when LSI drives the IWDG and you need an accurate timeout.

## Cortex-M Specifics

- **CYCCNT (DWT cycle counter)** ticks at the **core clock**. Use it for fine-grained profiling — but remember the units change if you change SYSCLK.
- **WFI/WFE** stop the CPU clock instantly; PLL keeps running unless you also clear `PLLON`.
- **Cortex-M7 D-cache** affects perceived performance more than clock speed for some workloads — clocks aren't the whole story.

## Common Pitfalls

### Pitfall 1: Forgetting Flash Wait States

```c
// BAD: switch to 168 MHz with 0 wait states
RCC->CFGR |= RCC_CFGR_SW_PLL;   // hardfault on next instruction fetch
```

Always set FLASH->ACR latency **before** raising SYSCLK.

### Pitfall 2: Forgetting to Enable the Peripheral Clock

```c
GPIOA->MODER = ...;   // does nothing; AHB1ENR_GPIOAEN not set
```

The most common embedded bug. Every peripheral has an RCC enable bit.

### Pitfall 3: Wrong Crystal Load Caps

Two 22 pF caps with a crystal speced 12 pF → oscillator drifts ~30 ppm low. Two 4.7 pF with the same crystal → may not start at all over temperature.

### Pitfall 4: Using HSI for USB

USB FS needs 48 MHz with ±0.25% accuracy for compliance. HSI's ±1% misses this. Use HSE + PLLQ, or HSI48 with CRS (clock recovery system, on chips that support it).

### Pitfall 5: PLL Source Switched While PLL Is On

Some chips require disabling PLL (`PLLON = 0`) before changing the source. Reading the wrong bit order from the reference manual makes the chip lock to the old source silently.

### Pitfall 6: SystemCoreClock Not Updated

CMSIS macros and HAL functions use `SystemCoreClock`. If you set up the PLL manually but never call `SystemCoreClockUpdate()`, `HAL_Delay(1000)` might actually wait 6 seconds.

### Pitfall 7: APB Prescaler Affects Timer Clock

```c
// Set APB1 = HCLK / 4
RCC->CFGR |= RCC_CFGR_PPRE1_DIV4;
// Then TIM2 clock = PCLK1 × 2, not PCLK1. Easy to mis-compute prescaler.
```

### Pitfall 8: LSE Pin Floating in Hardware

LSE crystal not populated, but `LSEON=1` and you wait for `LSERDY`. Code blocks forever in `while(!(RCC->BDCR & RCC_BDCR_LSERDY))`. Always have a timeout, even on "should-be-instant" ready bits.

```c
uint32_t timeout = 100000;
while (!(RCC->BDCR & RCC_BDCR_LSERDY) && --timeout) {}
if (!timeout) { fall_back_to_lsi(); }
```

## Summary

1. **Clock tree = sources → PLL → SYSCLK → bus prescalers → peripheral clocks.**
2. **Crystal needs correct load caps** (datasheet `C_L`) and clean layout.
3. **PLL needs reference (M), VCO range (N), output dividers (P/Q/R)** — all bounded by datasheet.
4. **Flash wait states must be raised before SYSCLK is raised.**
5. **Every peripheral needs its RCC enable bit set.**
6. **APB1 prescaler ≠ 1 → TIMx clock doubles** (idiosyncrasy worth memorizing).
7. **MCO pin lets you scope-verify clocks** when in doubt.
8. **HSI for fast wake, HSE for accuracy** (USB, CAN, baud-precise UART).
9. **Always timeout on ready-bit polls** — a missing crystal hangs forever otherwise.
10. **Update SystemCoreClock** after every clock change or HAL drifts.

## See Also

- [Power Management](power_management.md) — clock gating, sleep modes
- [Timers](timers.md) — derive PWM/IC/OC frequencies from PCLK
- [RTC](rtc.md) — LSE-based timekeeping
- [Linker Scripts](linker_scripts.md) — Reset_Handler / SystemInit ordering
