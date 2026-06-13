# 06 · Button input with a pull-up/pull-down

## Overview

A [switch](../electronics/switches_relays.md) seems trivial — until you ask "what voltage
is the pin when the button *isn't* pressed?" The answer, without help, is *nobody knows* —
a **floating** input that picks up noise. The fix is a **pull-up** or **pull-down**
resistor that defines the idle level. This rung is pure analog, read on the
[multimeter](../electronics/prototyping.md), but it's the single most important idea for
every [GPIO](../embedded/gpio.md) input you'll wire to a microcontroller from Phase 2 on.

```
  Pull-down (idle LOW, press = HIGH):     Pull-up (idle HIGH, press = LOW):

   5V ──[ button ]──┬── PIN                5V ──[ 10k ]──┬── PIN
                    │                                    │
                  [ 10k ]                            [ button ]
                    │                                    │
   GND ─────────────┘                      GND ─────────┘
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- A push button, a 10 kΩ resistor, an LED + 330 Ω (optional, to visualise)
- Breadboard, multimeter, 5 V source

## The build

**First, see the problem.** Wire a button from 5 V to a breadboard row ("PIN"), nothing
else. Measure PIN to GND: pressed → 5 V; released → it floats (drifts, picks up your hand).
That ambiguity is the bug.

**Now fix it with a pull-down:**

1. Button from **5 V** to **PIN**.
2. **10 kΩ from PIN to GND.**
3. Measure PIN: released → a solid **0 V** (the resistor gently ties it low); pressed →
   **5 V** (the button wins, pulling it high).

Swap to a **pull-up** (10 kΩ from PIN to 5 V, button from PIN to GND) and it inverts: idle
HIGH, press LOW. Both are correct — pull-ups are more common because most chips have them
built in.

```
            released   pressed
  pull-down    0 V       5 V     (active-high)
  pull-up      5 V       0 V     (active-low)
```

## It works when…

- [ ] The bare button's idle voltage is ambiguous/floating.
- [ ] With a pull-down, released reads a firm 0 V and pressed reads 5 V.
- [ ] You can wire it the other way (pull-up) and predict the inverted readings.

## What's happening

A microcontroller input is high-impedance — it barely draws current, so an unconnected pin
has no defined voltage and acts like a tiny antenna. The pull resistor provides a weak,
defined path to a rail (10 kΩ passes only 0.5 mA at 5 V — negligible) that sets the idle
level, while the button provides a strong path that overrides it when pressed. From
[rung 12](12_button_debounce.md) you'll use the AVR's *internal* pull-up (`INPUT_PULLUP`)
to do this with zero external parts — but you'll know exactly what it's doing because you
built it by hand here. See [GPIO](../embedded/gpio.md) for the input/output modes.

## Pitfalls

- **No pull resistor at all** — a floating input reads random HIGH/LOW and triggers phantom presses. Always pull an input.
- **Pull resistor too small** — a 100 Ω pull wastes current and fights the button; 10 kΩ is the standard compromise. Too large (>1 MΩ) and noise can still win.
- **Forgetting active-low logic** — with a pull-up, *pressed = LOW*. Your later code must test for the level you actually wired, or the logic is inverted.

## Where this connects

- [Switches, Relays & Electromechanical](../electronics/switches_relays.md) — switch types and contact bounce (which you'll tame in code next)
- [GPIO](../embedded/gpio.md) — input modes, internal pull-ups, and reading a pin
- [Resistance & Ohm's Law](../electronics/resistance.md) — why 10 kΩ is "weak but defined"
- **Previous:** [05 · RC charge & fade](05_rc_fade.md) · **Next:** [07 · BJT as a switch](07_bjt_switch.md)
