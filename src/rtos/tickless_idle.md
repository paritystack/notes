# Tickless Idle

## Overview

The periodic RTOS tick — the timer interrupt that drives [context switches](context_switching.md)
and wakes delayed tasks — is convenient but power-hungry: it pulls the CPU out of sleep
hundreds or thousands of times a second even when nothing is due. **Tickless idle**
suppresses that tick whenever the system is idle, letting the MCU stay in a low-power sleep
state for as long as possible and waking only when real work is due. For battery-powered
nodes ([BLE](../embedded/ble.md) sensors, [LoRa](../embedded/lora.md) endpoints) it's
often the single biggest lever on battery life. This page explains the mechanism and how
[FreeRTOS](freertos.md) (`configUSE_TICKLESS_IDLE`) and [Zephyr](zephyr.md) implement it.

It's the [scheduling](scheduling.md) tick meeting [power management](../embedded/power_management.md)
and [clock systems](../embedded/clock_systems.md).

## Why the periodic tick costs power

```
Fixed 1 kHz tick, system otherwise idle:
   sleep ─┬─wake─┬─sleep ─┬─wake─┬─sleep ─┬─wake─┬─  (1000×/second)
          │ tick │        │ tick │        │ tick │
          handle delays   nothing to do   nothing to do
   → the CPU never reaches deep sleep long enough to matter; each wake costs
     energy (clock restart, regulator settle) for no useful work.
```

Each wake has fixed overhead (oscillator startup, regulator settling) — so 1000 pointless
wakes per second can dominate the average current of an otherwise-sleeping device.

## The tickless mechanism

When the scheduler finds no ready task, instead of sleeping for one tick it asks: *how long
until the next task actually needs to run?* — then sleeps for (close to) that whole
interval and corrects the tick count on wake.

```
1. Idle task entered, no ready tasks.
2. Compute "expected idle time" = ticks until the nearest timer/delay expires.
3. Reprogram a low-power wakeup timer for that interval (capped at its max range).
4. Enter a deep sleep state.
5. Wake on: the timer expiring, OR an interrupt (a real event arriving early).
6. On wake, read elapsed time and ADVANCE the tick count by the slept duration
   (so software time stays correct — a "catch-up", not 1000 separate ticks).
```

```
   tick-based: ││││││││││││││││││││││││  (wake every tick)
   tickless:   │________________________│  (one long sleep to next deadline)
                ▲ sleep N ticks at once   ▲ wake & fast-forward the tick count
```

The key requirement is a **timer that keeps running in the chosen sleep state** — typically
a low-power timer or RTC on a 32.768 kHz crystal, since the main clock is stopped. See
[clock systems](../embedded/clock_systems.md) and [RTC](../embedded/rtc.md).

## FreeRTOS and Zephyr

```
FreeRTOS   configUSE_TICKLESS_IDLE = 1 uses a built-in SysTick-based implementation;
           set to 2 to supply your own portSUPPRESS_TICKS_AND_SLEEP() for a
           deep-sleep timer. Hooks: configPRE_SLEEP_PROCESSING / POST_SLEEP_PROCESSING
           to gate peripherals and pick the sleep mode.
Zephyr     tickless kernel is the default; the kernel programs the system timer for the
           next deadline. The Power Management subsystem (CONFIG_PM) then selects a
           residency-appropriate low-power state for the idle duration.
```

Both pick the *deepest* sleep state whose wake latency and retention suit the expected idle
time — a short idle uses a light sleep; a long one uses deep sleep.

## Where this connects

- [Scheduling](scheduling.md) — the tick this suppresses is what wakes delayed tasks and
  drives time-slicing
- [Context switching](context_switching.md) — SysTick normally pends the switch; tickless
  reprograms it instead of letting it free-run
- [Power management](../embedded/power_management.md) — sleep states, wake latency vs depth
  trade-offs, peripheral gating
- [Clock systems](../embedded/clock_systems.md) / [RTC](../embedded/rtc.md) — the
  always-on low-power timer that must keep time while the main clock is stopped
- [BLE](../embedded/ble.md) / [LoRa](../embedded/lora.md) — duty-cycled radios whose battery
  life hinges on staying asleep between events

## Pitfalls

```
- No always-on timer in the chosen sleep state → time is lost and delays drift. Use an
  LPTIM/RTC that survives the sleep mode you enter.
- Wake-latency longer than the next deadline: too-deep a sleep makes you miss it. Match
  sleep depth to the expected idle interval and the task's timing slack.
- Forgetting to gate peripherals before sleep — a UART/clock left running keeps current
  high and may even prevent deep sleep. Use the pre/post-sleep hooks.
- Timer counter width limits the max single sleep; the port must cap and re-enter, not
  overflow. Verify long idles fast-forward correctly.
- Races on wake: an interrupt arriving between "decide to sleep" and "enter sleep" must not
  be lost — enter sleep with interrupts masked the architecturally-correct way (e.g. WFI
  with PRIMASK set), or you can sleep through the event.
- Jitter on wake from deep sleep (oscillator startup) adds latency to the first task —
  account for it in time-critical paths.
```
