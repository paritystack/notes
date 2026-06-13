# 04 · Potentiometer as a voltage divider, dim an LED

## Overview

A [potentiometer](../electronics/resistance.md) is a [voltage divider](../electronics/resistance.md)
you can turn with a knob — and it's the bridge from "fixed resistors" to "an input a human
(or later, an [ADC](../embedded/adc.md)) can read." In this rung you'll watch the wiper
voltage sweep from 0 V to 5 V on the [multimeter](../electronics/prototyping.md), then use
that variable voltage to smoothly dim an [LED](../electronics/diodes.md). It's the last
no-microcontroller rung in Phase 0, and it sets up every analog-input project that follows.

```
  Pot wired as a 3-terminal divider:

   5V ──● A
        │
        ▒  wiper (W) ──► Vout  (0 V … 5 V as you turn)
        │
  GND ──● B
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- A potentiometer (10 kΩ is ideal), breadboard, jumpers, multimeter
- An LED + its series resistor (330 Ω) from [rung 02](02_light_an_led.md)
- 5 V source (Nano 5V pin)

## The build — read the wiper

1. Connect the pot's two **outer** pins to **5 V** and **GND**.
2. Connect the **middle** pin (the wiper) to your meter's red probe, black to GND.
3. Turn the knob and watch Vout sweep.

```
  Meter on V⎓, RED → wiper, BLACK → GND:

   knob fully one way   → ~0 V
   knob centred         → ~2.5 V
   knob fully other way → ~5 V
```

You've made an adjustable voltage source. The pot is just R1/R2 of a
[divider](../electronics/resistance.md) where turning the knob trades resistance between
the two halves.

## The build — dim the LED

Now feed the wiper through the LED branch:

```
   wiper (W) ──[ 330Ω ]──►|── GND
                           LED
```

Turn the knob: as Vout rises above the LED's [forward voltage](../electronics/diodes.md),
it lights and brightens. This is a crude dimmer — fine for seeing the effect, but note the
brightness isn't linear and the pot wastes power. (A microcontroller doing
[PWM](../embedded/pwm.md) in [rung 13](13_pwm_fade.md) will dim far more cleanly — keep that
contrast in mind.)

## It works when…

- [ ] The wiper voltage sweeps smoothly from ~0 V to ~5 V as you turn the knob.
- [ ] The LED brightens and dims with the knob.
- [ ] You can explain why the LED stays off for the first part of the travel (below `Vf`).

## What's happening

Inside the pot is one resistive track with a wiper that taps off some fraction of the way
along it. The two outer pins plus the wiper form a [voltage divider](../electronics/resistance.md)
`Vout = 5 V × R_below / (R_above + R_below)` — and turning the knob continuously changes
that ratio. Crucially, the wiper is a *voltage* output: this is exactly the kind of
0–5 V analog signal an [ADC](../embedded/adc.md) digitises, which is why pots are the
"hello world" of analog input. You'll read this very setup with code in
[rung 14](14_adc_pot_serial.md).

## Pitfalls

- **Wiring the pot with only two pins for a divider** — you need all three (two ends + wiper). Two-pin wiring makes a [rheostat](../electronics/resistance.md) (variable resistor), which behaves differently.
- **Driving the LED straight from the wiper with no resistor** — still needs the current-limiting resistor; the pot does not protect the LED.
- **Expecting linear brightness** — the eye's response and the LED's sharp `Vf` knee make this dimmer non-linear. That's expected; PWM fixes it later.

## Where this connects

- [Resistance & Ohm's Law](../electronics/resistance.md) — potentiometers, dividers, and taper types
- [Sensors & Transducers](../electronics/sensors.md) — many sensors are just resistive dividers whose ratio reports a physical quantity
- [ADC](../embedded/adc.md) — the wiper voltage is what an analog input reads
- **Previous:** [03 · Series & parallel](03_series_parallel.md) · **Next:** [05 · RC charge & fade](05_rc_fade.md)
