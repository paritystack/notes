# 07 · BJT as a switch — drive a buzzer

## Overview

A microcontroller pin can only source a few milliamps — not enough for a buzzer, motor, or
bright lamp. The fix is a **transistor**: a tiny control current switches a much larger
load current. This rung uses a [BJT](../electronics/transistors_bjt.md) (bipolar junction
transistor) as a low-side switch to drive a piezo buzzer, controlled here by a wire (a
[GPIO](../embedded/gpio.md) pin in Phase 2). It's your first taste of the
**signal-controls-power** pattern that every output device depends on.

```
  NPN low-side switch:

   5V ──[ buzzer ]──┐
                    │ C (collector)
   control ─[ 1k ]──┤ B (base)        base current ON  → C–E conducts → buzzer ON
                    │ E (emitter)      base current OFF → C–E open     → buzzer OFF
   GND ─────────────┘
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- An NPN BJT (2N2222 or BC547), a piezo buzzer (or an LED to start)
- A 1 kΩ **base resistor**, breadboard, multimeter, 5 V source

> Check the [pinout](../electronics/datasheets.md) for your exact part — the E-B-C order
> differs between a 2N2222 and a BC547.

## The build

1. Buzzer from **5 V** to the transistor's **collector**.
2. **Emitter** to **GND**.
3. **Base** through a **1 kΩ resistor** to your control wire.
4. Touch the control resistor to **5 V** → buzzer sounds. To **GND** (or leave open) → silent.

```
  Why the base resistor?
   Base–emitter is a diode (~0.7 V). Without a resistor, the base would
   draw unlimited current and cook the transistor — just like an LED.
   I_base ≈ (5 V − 0.7 V) / 1 kΩ ≈ 4.3 mA  → plenty to saturate the switch.
```

Measure **collector-to-emitter voltage** when ON: it should be small (~0.1–0.3 V,
"saturated"). That low drop is what makes it a good switch — most of the 5 V lands on the
buzzer, not the transistor.

## It works when…

- [ ] Control HIGH sounds the buzzer; control LOW/open silences it.
- [ ] Collector–emitter voltage is near zero when ON (saturated).
- [ ] You can explain why the base needs a series resistor.

## What's happening

A BJT is **current-controlled**: a small base current `I_B` lets a much larger collector
current `I_C` flow, roughly `I_C = β × I_B` (β ≈ 100). To use it as a *switch* you drive the
base hard enough to **saturate** it — fully on, low voltage drop — rather than operate in
its in-between amplifying region. The 1 kΩ base resistor limits `I_B` the way an LED's
resistor limits its current. A real [GPIO](../embedded/gpio.md) pin replaces your control
wire in Phase 2. For higher-current or efficiency-critical loads you'll prefer a
[MOSFET](../electronics/transistors_mosfet.md) — that's the very next rung.

## Pitfalls

- **No base resistor** — destroys the transistor; the base-emitter junction is a diode that needs current limiting.
- **Wrong pinout** — E/B/C order varies by part. Swapping collector and emitter usually means it won't switch. Check the [datasheet](../electronics/datasheets.md).
- **No flyback diode on an inductive load** — fine for a piezo buzzer, but a relay or motor needs the diode you'll add in [rung 09](09_flyback_diode.md).
- **Expecting to switch high currents** — a small-signal BJT like a 2N2222 handles ~few hundred mA. For a 12 V fan, use the MOSFET in [rung 08](08_mosfet_fan.md).

## Where this connects

- [BJT Transistors](../electronics/transistors_bjt.md) — regions of operation, β, saturation
- [MOSFET Transistors](../electronics/transistors_mosfet.md) — the voltage-controlled alternative, used next
- [GPIO](../embedded/gpio.md) — the pin that will drive the base in Phase 2
- **Previous:** [06 · Button + pull-up](06_button_pullup.md) · **Next:** [08 · MOSFET drives a fan](08_mosfet_fan.md)
