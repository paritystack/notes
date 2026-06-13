# 23 · Run the whole thing on battery, measure draw

## Overview

Your [sensor + OLED](18_i2c_oled.md) now runs from a [charged LiPo](22_lipo_charging.md)
through the [LDO](21_ldo_regulator.md) — untethered. The question that defines a portable
product: **how long will it last?** That's just current draw vs. battery capacity. You'll
measure the system's actual current, compute runtime, and discover that an always-on
microcontroller + display is a battery hog — motivating the [sleep](24_deep_sleep.md) work
next. This is where [power & energy](../electronics/power.md) stops being theory.

```
   Measure current in series with the supply:

   BAT+ ──[ multimeter mA ]── system +      (meter BREAKS the + line)
   BAT− ───────────────────── system −

   Runtime ≈ battery capacity (mAh) / average current (mA)
```

## What you'll need

The battery-powered [sensor/OLED](18_i2c_oled.md) build, the multimeter (in current mode), and
the cell's rated capacity (mAh). A USB power meter or a dedicated current tester helps but
isn't required.

## The build

1. Put the multimeter **in series** with the battery's positive lead, on the **mA range**
   (red lead in the mA socket — the [rung 01](01_bench_and_multimeter.md) habit).
2. Read the average current with everything running (sensor polling, OLED on).
3. Compute runtime:

```
   Example: system draws ~25 mA continuous, cell = 500 mAh
   Runtime ≈ 500 mAh / 25 mA ≈ 20 hours   (best case; real is less)

   Where the current goes (typical, always-on):
     ATmega328 active      ~8 mA
     OLED display          ~10–15 mA
     LDO quiescent + sensor ~1–5 mA
```

4. Try switching the OLED off between updates, or slowing the loop — watch the average drop.
   You're now *budgeting power*, the core discipline of portable design.

## It works when…

- [ ] You measure the system's running current in mA.
- [ ] You compute a runtime estimate from capacity ÷ current.
- [ ] You can identify which part (likely the always-on MCU + OLED) dominates the draw.

## What's happening

Battery life is energy in (mAh) divided by the rate you spend it (mA) — nothing more, but the
average current hides a lot. An always-active [AVR](../embedded/avr.md) and a lit OLED draw
tens of mA *continuously*, so even a decent cell lasts only a day. Most of that time the
device is doing nothing useful between sensor reads — energy wasted staying awake. Measuring
first (rather than guessing) is the engineering habit: you can't optimise what you haven't
measured. The fix is to make the MCU [sleep](24_deep_sleep.md) and wake only to work, which
can cut average current by 100–1000×. See [Power Management](../embedded/power_management.md)
for the techniques.

## Pitfalls

- **Meter in parallel** — current is measured *in series*; across the battery it shorts the cell through the meter (and blows its fuse, or worse with a LiPo). Break the line.
- **Wrong range / blown fuse** — start on a higher current range; if the meter reads 0 with a working device, its mA fuse may be blown.
- **Ignoring peaks** — averages hide brief high-current spikes (e.g. radio TX later); a cheap meter may not catch them. Note them when sizing the battery.
- **Forgetting the LDO's own draw** — quiescent current counts 24/7; a wasteful regulator ruins a low-power design (back to [rung 21](21_ldo_regulator.md)).

## Where this connects

- [Power & Energy](../electronics/power.md) — current, capacity, and runtime
- [Power Management](../embedded/power_management.md) — the sleep/duty-cycling that extends runtime
- [22 · LiPo charging](22_lipo_charging.md) — the cell whose capacity sets your budget
- **Previous:** [22 · LiPo charging](22_lipo_charging.md) · **Next:** [24 · Deep-sleep the AVR](24_deep_sleep.md)
