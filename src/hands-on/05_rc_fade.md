# 05 · RC circuit — charge & discharge, watch a fade

## Overview

A resistor sets *how much* current flows; add a [capacitor](../electronics/capacitors.md)
and you control *how it changes over time*. An **RC circuit** is the simplest way to see
time in electronics — the capacitor charges and discharges on a curve set by the
[time constant](../electronics/capacitors.md) τ = R × C. You'll watch an
[LED](../electronics/diodes.md) fade out on its own as a cap discharges, and measure the
curve on the [multimeter](../electronics/prototyping.md). This is the intuition behind
debounce timing, power-supply smoothing, and [filters](../electronics/filters.md).

```
  Charge then discharge:

   5V ──[ R ]──┬── Vcap        Vcap over time (charging):
               │                 5V ┤        ____------
              ===  C               │     _--
               │                  0V ┤__--
   GND ────────┘                     └── τ ── 2τ ── 3τ ──►
                                     (≈63% charged at 1τ)
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- A capacitor — a large electrolytic (e.g. 100 µF) makes the timing visible by eye
- A resistor (e.g. 10 kΩ to charge; the LED branch to discharge), an LED + 330 Ω
- Breadboard, multimeter, 5 V source

> Electrolytic caps are **polarised** — the longer leg / unmarked side is **+**; the stripe
> marks **−**. Backwards, they can pop. See [Capacitors](../electronics/capacitors.md).

## The build

1. **Charge:** wire **5 V → 10 kΩ → cap(+) → GND(cap −)**. Probe Vcap with the meter and watch it climb toward 5 V on a curve — fast at first, then slowing.
2. **Discharge through an LED:** disconnect 5 V and connect the charged cap across **LED + 330 Ω to GND**. The LED lights and **fades out** as the cap empties.

```
  τ = R × C.   For R = 10 kΩ, C = 100 µF:
    τ = 10 000 Ω × 0.0001 F = 1 second
  ≈63% charged after 1τ, ≈99% after 5τ (≈5 s here).
```

Try a bigger cap or resistor → slower fade. Smaller → faster. You're dialling time with two
parts.

## It works when…

- [ ] Vcap rises on a visible curve, not instantly.
- [ ] The LED fades out smoothly when the cap discharges through it.
- [ ] Changing R or C visibly changes how long the fade takes.

## What's happening

A capacitor stores charge; voltage across it can't change instantly because moving charge
takes current, and the resistor [limits](../electronics/resistance.md) that current. The
result is the exponential approach captured by τ = RC — the single most reused timing idea
in electronics. The same curve smooths a [power supply](../electronics/power_supplies.md)'s
ripple, sets a [filter](../electronics/filters.md)'s cutoff, and debounces a button. When
you later debounce in [code](12_button_debounce.md), you're doing in software what this RC
does in hardware.

## Pitfalls

- **Electrolytic in backwards** — polarised caps can vent or pop. Stripe = −. Double-check before powering.
- **Probing a "dead" circuit that isn't** — a charged cap holds voltage after power is removed. Discharge big caps through a resistor before handling (a [datasheet/prototyping](../electronics/prototyping.md) habit).
- **Expecting a straight-line fade** — it's exponential: quick at first, then a long tail. That's the RC curve, not a fault.

## Where this connects

- [Capacitors](../electronics/capacitors.md) — charge storage, polarity, and the RC time constant
- [Filters](../electronics/filters.md) — an RC network is the simplest low-/high-pass filter
- [Power Supplies](../electronics/power_supplies.md) — smoothing caps use this same charge/hold behaviour
- **Previous:** [04 · Potentiometer divider](04_pot_divider.md) · **Next:** [06 · Button + pull-up](06_button_pullup.md)
