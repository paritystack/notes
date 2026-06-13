# 24 · Deep-sleep the AVR, measure microamps

## Overview

[Rung 23](23_battery_current.md) showed an always-on device drains a cell in a day. The fix
is **duty-cycling**: the [microcontroller](../embedded/avr.md) does its work in a few
milliseconds, then **sleeps** at microamp levels until it's needed again. You'll put the AVR
into its deepest sleep, wake it periodically, and watch the average current drop by orders of
magnitude on the [multimeter](23_battery_current.md). This single technique is the difference
between a day and a year of battery life — and the last skill before you make the
[capstone permanent](26_perfboard_build.md).

```
   Duty cycle: brief work, long sleep

   active ▌                ▌                ▌
   sleep  ─────────────────────────────────────►
          read+show        sleep ~8 s        read+show
   avg current ≈ (I_active·t_active + I_sleep·t_sleep) / period
```

## What you'll need

The battery-powered build, the multimeter (ideally one that reads µA), and the AVR
sleep/watchdog libraries (`avr/sleep.h`, `avr/wdt.h`). A low-Iq [LDO](21_ldo_regulator.md)
matters more than ever here.

## The build

1. Use the watchdog [timer](../embedded/timers.md) to wake the chip every ~8 s; sleep deeply
   in between:

```c
#include <avr/sleep.h>
#include <avr/wdt.h>
#include <avr/interrupt.h>

ISR(WDT_vect) {}                       // wake-up handler (does nothing but return)

void sleep8s() {
  MCUSR &= ~(1 << WDRF);
  WDTCSR = (1 << WDCE) | (1 << WDE);
  WDTCSR = (1 << WDIE) | (1 << WDP3) | (1 << WDP0);   // ~8 s, interrupt mode
  set_sleep_mode(SLEEP_MODE_PWR_DOWN);                // deepest sleep
  sleep_enable(); sleep_cpu(); sleep_disable();
}

void loop() {
  readSensorAndShow();                 // a few ms of real work
  sleep8s();                           // ~8 s at microamps
}
```

2. Measure current during sleep (should be **microamps**, not milliamps) and during the brief
   active burst.
3. Compute the new average and runtime — often a 100×+ improvement over [rung 23](23_battery_current.md).

```
   Before: 25 mA always-on  → ~20 h
   After:  10 mA for 50 ms every 8 s + ~10 µA asleep
           avg ≈ (10 mA·0.05 + 0.01 mA·8)/8.05 ≈ 0.072 mA → weeks–months
```

Power down the [OLED](18_i2c_oled.md) between updates too — a lit display dwarfs the sleeping
MCU.

## It works when…

- [ ] Sleep current reads in microamps (or low single-digit mA if peripherals stay on).
- [ ] The device still wakes, reads, and updates on schedule.
- [ ] Your computed average current — and runtime — is dramatically better than rung 23.

## What's happening

In `SLEEP_MODE_PWR_DOWN` the AVR stops its main clock and most peripherals, drawing
microamps; only the watchdog [timer](../embedded/timers.md) keeps running on its own low-power
oscillator and fires an [interrupt](../embedded/interrupts.md) to wake the CPU. Average
current becomes a weighted blend of a tiny active slice and a long sleep — so cutting either
the active *time* or the sleep *current* extends life enormously. The real enemies are
**always-on peripherals** (the OLED, a wasteful [LDO](21_ldo_regulator.md)'s quiescent
current, pull-ups sinking current), which is why low-power design is a whole-system exercise.
This is the headline skill of [Power Management](../embedded/power_management.md), and your
capstone firmware will be built around it.

## Pitfalls

- **Peripherals left powered** — the MCU sleeps at 10 µA but a lit OLED still pulls 10 mA, erasing the win. Switch off or power-gate everything you can.
- **Brown-out detector & ADC left on** — these keep drawing in sleep; disable BOD and the ADC before sleeping for the lowest current.
- **Floating input pins** — undriven inputs can sink current and waste power; set unused pins to a defined state.
- **No way to wake** — without the watchdog interrupt or an external [interrupt](../embedded/interrupts.md), `PWR_DOWN` sleeps forever. Always arm a wake source first.

## Where this connects

- [Power Management](../embedded/power_management.md) — sleep modes, clock gating, duty-cycling
- [Interrupts](../embedded/interrupts.md) / [Timers](../embedded/timers.md) — the wake mechanism
- [23 · Battery current](23_battery_current.md) — the baseline this dramatically improves
- **Previous:** [23 · Run on battery](23_battery_current.md) · **Next:** [25 · Learn to solder](25_learn_to_solder.md)
