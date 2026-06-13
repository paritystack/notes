# 13 · PWM fade

## Overview

Back in [rung 04](04_pot_divider.md) you dimmed an LED with a pot — wasteful and non-linear.
Now you'll dim it the *right* way: **PWM** (pulse-width modulation) switches the pin fully
on and off very fast, and the **ratio** of on-time to off-time sets the average brightness.
The LED is never half-on, so nothing is wasted as heat. This is the single most useful
output trick in embedded — the same technique drives motor speed, [DAC](../embedded/dac.md)-like
analog outputs, and dimmers. You'll make an LED "breathe."

```
  PWM: same frequency, varying duty cycle → varying average

   25% duty:  █___█___█___█___   dim
   50% duty:  ██__██__██__██__   medium
   75% duty:  ███_███_███_███_   bright
              (switches ~490 Hz on the Nano — too fast for the eye)
```

## What you'll need

From **Stage A**: Arduino Nano, an LED + 330 Ω on a **PWM-capable pin** (on the Nano:
D3, D5, D6, D9, D10, D11 — marked `~`).

## The build

1. LED + 330 Ω from **D9** (a `~` pin) to GND.
2. Flash a breathing fade:

```c
const int LED = 9;            // must be a ~ (PWM) pin

void setup() {
  pinMode(LED, OUTPUT);
}

void loop() {
  for (int b = 0; b <= 255; b++) { analogWrite(LED, b); delay(4); } // brighten
  for (int b = 255; b >= 0; b--) { analogWrite(LED, b); delay(4); } // dim
}
```

`analogWrite(pin, 0..255)` sets the duty cycle. Note it's *not* a real analog voltage — put
the [multimeter](../electronics/prototyping.md) on the pin and you'll read a fuzzy average;
on a [scope](../electronics/prototyping.md) you'd see the square wave changing width. Compare
the smoothness to the pot dimmer from rung 04.

## It works when…

- [ ] The LED smoothly breathes up and down — no visible steps or flicker.
- [ ] On a scope (if you have one) the pin shows a fixed-frequency square wave whose width changes.
- [ ] You can explain why PWM dimming wastes almost no power, unlike the rung-04 pot.

## What's happening

A hardware [timer](../embedded/timers.md) inside the AVR counts continuously and flips the
pin LOW when the count passes your duty value, HIGH when it wraps — generating the
[PWM](../embedded/pwm.md) waveform with zero CPU effort after setup. Because the LED only
ever sees full-on or full-off, the [transistor](../electronics/transistors_mosfet.md)-like
switching is efficient: average power tracks duty cycle. Your eye (and a smoothing
[capacitor](../electronics/capacitors.md), or a motor's inertia) integrates the pulses into
an apparent steady level. This exact `analogWrite` drives the fan's speed in
[rung 15](15_pwm_fan_speed.md).

## Pitfalls

- **Using a non-PWM pin** — `analogWrite` on a plain digital pin only gives on/off at 0/255. Use a `~` pin (D3/5/6/9/10/11).
- **Expecting a true analog voltage** — PWM is a fast square wave; a multimeter shows a misleading average. Add an [RC filter](05_rc_fade.md) if you genuinely need analog.
- **Audible whine on motors/coils** — the ~490 Hz default is in the audible range; for motors you may raise the PWM frequency (timer config) to push it above hearing.

## Where this connects

- [PWM](../embedded/pwm.md) — duty cycle, frequency, and timer-driven generation
- [04 · Potentiometer divider](04_pot_divider.md) — the wasteful analog dimmer this replaces
- [DAC](../embedded/dac.md) — PWM + filter as a poor man's analog output
- **Previous:** [12 · Button + debounce](12_button_debounce.md) · **Next:** [14 · ADC: read a pot](14_adc_pot_serial.md)
