# 22 · LiPo charging (TP4056) + protection

## Overview

To go portable you need a rechargeable cell and a *safe* way to charge it. Lithium cells are
energy-dense but unforgiving — overcharge, over-discharge, or short them and they can vent or
catch fire. The **TP4056** module handles correct lithium charging (constant-current then
constant-voltage), and a protection IC guards against the dangerous edges. You'll charge a
[LiPo](../electronics/power_supplies.md) from USB and feed it into your
[LDO](21_ldo_regulator.md) from the last rung. **Respect this rung — it's the only one with
real fire risk.**

```
   USB 5V ──► [ TP4056 + protection ] ──► LiPo cell
                        │
                        └──► BAT out ──► [LDO] ──► 3.3V system

   Charge: CC (constant current) → CV (constant voltage 4.2V) → done LED
```

## What you'll need

From **Stage B/D**: a **TP4056 module with protection** (the version with an extra DW01 +
FS8205 chip — *not* the bare charger), a single LiPo cell with a known capacity, the
[LDO](21_ldo_regulator.md) rail, and the multimeter.

## The build

1. Connect the **LiPo to the B+ / B−** pads of the TP4056 (observe polarity — reversing it
   can destroy the module or the cell).
2. Power the module from **USB**. The charge LED (red) lights; it goes green/blue at full.
3. Measure **B+ to B−**: it rises toward **4.2 V** and holds there (the CV phase). Charge
   current tapers as it fills.
4. Take system power from the **OUT pads** (protected), feed the [LDO](21_ldo_regulator.md),
   and run your [sensor/OLED](18_i2c_oled.md) from the battery.

```
   Lithium charge profile:
     CC: hold a fixed current, voltage rises   ──►
     CV: hold 4.2 V, current falls             ──► stop at ~C/10
   The TP4056 sets charge current with one resistor (Rprog) — see its datasheet.
```

## It works when…

- [ ] The cell charges and the module's status LED indicates full.
- [ ] Battery voltage settles near 4.2 V and doesn't exceed it.
- [ ] Your system runs from the protected OUT pads through the LDO.

## What's happening

Lithium chemistry needs **CC-CV** charging: constant current until ~4.2 V, then constant
voltage while current tapers — overshooting 4.2 V is what makes cells dangerous, so the
TP4056 enforces it precisely. The **protection** IC (the second chip) disconnects the cell on
over-charge, over-discharge (below ~2.5 V, which permanently damages LiPos), or over-current/
short. Always take your load from the *protected* output, not directly off the cell. The cell
voltage swings 4.2 V (full) → ~3.0 V (empty), which is exactly why the previous rung cared so
much about [LDO dropout](21_ldo_regulator.md): your regulator must still make 3.3 V near the
bottom of that swing. See [Power Supplies](../electronics/power_supplies.md) for the broader
picture.

## Pitfalls

- **Bare TP4056 without protection** — the cheapest modules omit the protection chip; a cell can then over-discharge and become a hazard. Buy the protected variant.
- **Reversed cell polarity** — can destroy the module or the cell instantly, and damaged LiPos are a fire risk. Triple-check B+/B−.
- **Wrong charge current** — the default Rprog is ~1 A; for a small cell that's too fast. Charge at ≤1C (e.g. a 500 mAh cell at ≤0.5 A); change Rprog per the datasheet.
- **Puffed/damaged cells** — a swollen LiPo is failing; stop using it. Never charge unattended or near flammables.

## Where this connects

- [Power Supplies](../electronics/power_supplies.md) — battery chemistry, charging, regulation
- [21 · LDO regulator](21_ldo_regulator.md) — the rail this battery feeds (dropout matters!)
- [23 · Run on battery](23_battery_current.md) — now measure how long it lasts
- **Previous:** [21 · LDO regulator](21_ldo_regulator.md) · **Next:** [23 · Battery current](23_battery_current.md)
