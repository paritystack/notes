# 12 · Read a button + debounce in code

## Overview

Now combine input and output: read the [button](06_button_pullup.md) and toggle an LED on
each press. You'll meet **contact bounce** — a mechanical switch doesn't close cleanly, it
chatters for a few milliseconds — and fix it in software. This is the firmware-meets-physics
moment: the [pull-up](06_button_pullup.md) you wired by hand is now the AVR's *internal*
pull-up, and the [RC](05_rc_fade.md) settling you watched is now a timing check in code.

```
   Button between D2 and GND, internal pull-up enabled:

   D2 ──[ button ]── GND      idle: pin reads HIGH (pulled up)
                              press: pin reads LOW  (button shorts to GND)
```

## What you'll need

From **Stage A**: Arduino Nano, a push button, an LED + 330 Ω (or onboard D13). No external
resistor needed — we use the chip's internal pull-up.

## The build

1. Button from **D2** to **GND** (nothing else — the internal pull-up replaces the external one).
2. LED on **D13** (or onboard).
3. Flash a naive version first to *see* the bounce, then the debounced one:

```c
const int BTN = 2, LED = 13;
bool ledOn = false;
int lastReading = HIGH;
unsigned long lastChange = 0;
const unsigned long DEBOUNCE_MS = 20;

void setup() {
  pinMode(BTN, INPUT_PULLUP);   // internal pull-up: idle HIGH, press LOW
  pinMode(LED, OUTPUT);
}

void loop() {
  int reading = digitalRead(BTN);
  if (reading != lastReading) {
    lastChange = millis();      // the input just moved; start the timer
  }
  if (millis() - lastChange > DEBOUNCE_MS) {
    // stable for 20 ms — trust it
    static int stable = HIGH;
    if (reading != stable) {
      stable = reading;
      if (stable == LOW) {      // active-low: LOW means pressed
        ledOn = !ledOn;
        digitalWrite(LED, ledOn);
      }
    }
  }
  lastReading = reading;
}
```

Without debounce, one press sometimes toggles twice (or not at all). With the 20 ms window,
every press toggles exactly once.

## It works when…

- [ ] A naive `if (digitalRead==LOW) toggle;` version mis-toggles or double-fires.
- [ ] The debounced version toggles the LED exactly once per press.
- [ ] You understand why `INPUT_PULLUP` means *pressed = LOW* (active-low).

## What's happening

`INPUT_PULLUP` enables the [GPIO](../embedded/gpio.md)'s built-in pull-up resistor, so the
pin idles HIGH and the button pulls it LOW — the exact circuit of [rung 06](06_button_pullup.md),
now free of external parts. Contact bounce is real physics: the metal contacts literally
bounce, producing a burst of HIGH/LOW transitions over a few ms. The debounce logic ignores
changes until the reading has held steady past a threshold — software doing what an
[RC](05_rc_fade.md) filter or a hardware debouncer would. `millis()` is a free-running
[timer](../embedded/timers.md) counter, which is why this is non-blocking unlike
`delay()`.

## Pitfalls

- **Inverted logic** — with a pull-up, *pressed = LOW*. Testing for HIGH inverts everything.
- **Debouncing with `delay()`** — blocks the whole program; use the `millis()` timestamp pattern so the chip stays responsive.
- **Too-short debounce window** — under ~5 ms can let bounce through; ~20 ms is a safe default for tactile buttons.
- **Forgetting `INPUT_PULLUP`** — a plain `INPUT` floats, giving random presses (the [rung 06](06_button_pullup.md) lesson).

## Where this connects

- [06 · Button + pull-up](06_button_pullup.md) — the hardware version of this exact input
- [GPIO](../embedded/gpio.md) — input modes and internal pull-ups
- [Timers](../embedded/timers.md) — `millis()` and non-blocking timing
- [Interrupts](../embedded/interrupts.md) — the event-driven alternative to polling a button
- **Previous:** [11 · Firmware blink](11_firmware_blink.md) · **Next:** [13 · PWM fade](13_pwm_fade.md)
