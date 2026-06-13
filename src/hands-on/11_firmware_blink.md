# 11 · Firmware blink — the hardware "hello, world"

## Overview

The microcontroller joins the breadboard. "Blink" is the hardware equivalent of
`print("hello")` — it proves your toolchain, board, and a [GPIO](../embedded/gpio.md)
output all work end to end. With your software background the *code* is trivial; the point
is the new ritual: write → compile → flash → it physically changes the world. From here the
[Arduino Nano](../embedded/arduino.md) (an [AVR](../embedded/avr.md) ATmega328) is your
prototyping brain for the rest of the project.

```
   USB ── [Arduino Nano] ── D13 ──[330Ω]──►|── GND
                                            LED
   (the Nano also has an LED on D13 already)
```

## What you'll need

From **Stage A** of the [shopping list](README.md):

- Arduino Nano clone + USB cable
- An LED + 330 Ω (optional — D13 has a built-in LED)
- The Arduino IDE (or `arduino-cli`) installed on your computer

## The build

1. Install the [Arduino](../embedded/arduino.md) IDE. Select board **Arduino Nano** and the
   right serial port. (Clones often need the *"ATmega328P (Old Bootloader)"* processor
   option — try it if upload fails.)
2. Wire an LED + 330 Ω from **D13** to **GND** (long leg to D13), or just use the onboard LED.
3. Flash this:

```c
void setup() {
  pinMode(13, OUTPUT);        // D13 is an output (drives the pin)
}

void loop() {
  digitalWrite(13, HIGH);     // pin → 5V, LED on
  delay(500);                 // wait 500 ms
  digitalWrite(13, LOW);      // pin → 0V, LED off
  delay(500);
}
```

Hit upload. The LED blinks at 1 Hz. You just did in software what the
[555](10_555_blink.md) did in hardware — but now the rate is a number you can change and
re-flash in seconds.

## It works when…

- [ ] The code compiles and uploads without error.
- [ ] The LED blinks at ~1 Hz; changing `delay(500)` to `delay(100)` makes it faster.
- [ ] You can measure ~5 V on D13 when HIGH and ~0 V when LOW with the [multimeter](../electronics/prototyping.md).

## What's happening

`pinMode(13, OUTPUT)` configures the pin's [GPIO](../embedded/gpio.md) direction register so
the chip *drives* the pin instead of listening to it; `digitalWrite` flips it between the 5 V
and 0 V rails. The Nano can source ~20 mA per pin — enough for an LED, which is exactly why
bigger loads needed the [transistor](07_bjt_switch.md) drivers you built earlier. The
`delay()` calls are the AVR busy-waiting; later you'll replace blocking delays with
[timers](../embedded/timers.md) so the chip can do other work meanwhile.

## Pitfalls

- **Upload fails ("not in sync")** — wrong port, wrong board, or the bootloader option. Try the "Old Bootloader" processor; check the cable is data-capable (not charge-only).
- **LED in backwards** — onboard LED always works; an external one needs long-leg to the pin. No light ≠ no upload.
- **Treating `delay()` as free** — it blocks everything. Fine for blink, a problem once you add buttons and sensors; you'll move past it.

## Where this connects

- [Arduino](../embedded/arduino.md) / [AVR](../embedded/avr.md) — the platform and chip you're now driving
- [GPIO](../embedded/gpio.md) — pin direction, drive strength, reading vs writing
- [10 · 555 blink](10_555_blink.md) — the same behaviour in pure hardware, for contrast
- **Previous:** [10 · 555 timer blink](10_555_blink.md) · **Next:** [12 · Button + debounce](12_button_debounce.md)
