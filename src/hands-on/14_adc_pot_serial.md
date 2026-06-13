# 14 · ADC — read a pot, print over serial

## Overview

So far the microcontroller has only *driven* things. Now it **reads** the analog world. The
[ADC](../embedded/adc.md) (analog-to-digital converter) turns the
[pot](04_pot_divider.md)'s wiper voltage into a number your code can use, and you'll send
that number to your computer over [serial/UART](../embedded/uart.md) — your primary debugging
window for the rest of the project (the embedded equivalent of `printf`). Reading a pot is
the gateway to reading *any* analog [sensor](../electronics/sensors.md).

```
   5V ─[pot]─ GND, wiper → A0

   ADC: 0 V → 0,  5 V → 1023   (10-bit: 0..1023)
   Serial Monitor:  512 ... 763 ... 1004 ...
```

## What you'll need

From **Stage A**: Arduino Nano, a 10 kΩ pot, jumpers. (Optional: feed the reading into the
[PWM](13_pwm_fade.md) LED.)

## The build

1. Pot outer pins to **5 V** and **GND**, wiper to **A0** (an analog input).
2. Flash:

```c
const int POT = A0, LED = 9;

void setup() {
  Serial.begin(9600);         // open the UART back to your PC
  pinMode(LED, OUTPUT);
}

void loop() {
  int raw = analogRead(POT);            // 0..1023
  float volts = raw * 5.0 / 1023.0;     // convert to voltage
  Serial.print(raw);
  Serial.print("  ->  ");
  Serial.print(volts, 2);
  Serial.println(" V");
  analogWrite(LED, raw / 4);            // 1023/4 ≈ 255: pot now controls brightness!
  delay(100);
}
```

Open the **Serial Monitor** (9600 baud). Turn the knob: the number sweeps 0→1023 and the
printed voltage matches what you metered in [rung 04](04_pot_divider.md). The LED brightness
follows the knob — input driving output, all in code.

## It works when…

- [ ] The Serial Monitor prints a value that tracks the knob from ~0 to ~1023.
- [ ] The printed voltage matches a [multimeter](../electronics/prototyping.md) reading on the wiper.
- [ ] The LED brightness follows the pot via PWM.

## What's happening

The AVR's [ADC](../embedded/adc.md) compares the input against an internal reference
(here Vcc = 5 V) and reports a 10-bit number, so each count ≈ 4.9 mV. `analogRead` triggers
one conversion and returns it. `Serial`/[UART](../embedded/uart.md) shifts bytes out one pin
at a fixed baud rate to the USB-serial chip on the Nano, which your PC sees as a COM port —
this print-to-serial loop is how you'll debug every later rung. Pairing `analogRead` →
`analogWrite` is a complete sense-and-act cycle in miniature; swap the pot for a
[sensor](17_i2c_sensor.md) and you have the core of the capstone.

## Pitfalls

- **Wrong baud rate** — Serial Monitor must match `Serial.begin()` (9600 here) or you get garbage. Mismatched baud is the #1 "garbled output" cause.
- **Reading a digital-only pin** — only A0–A7 have the ADC. Analog read on a digital pin returns nonsense.
- **Assuming the reference is exact** — Vcc from USB sags under load, shifting your volts math. For accuracy use the internal 1.1 V reference or measure Vcc.
- **Noisy readings** — long wires or a floating input jitter; average a few samples or add a small [cap](05_rc_fade.md) on the input.

## Where this connects

- [ADC](../embedded/adc.md) — resolution, reference voltage, sampling
- [UART](../embedded/uart.md) — the serial link carrying your debug prints
- [Sensors & Transducers](../electronics/sensors.md) — analog sensors read exactly like this pot
- **Previous:** [13 · PWM fade](13_pwm_fade.md) · **Next:** [15 · PWM fan speed](15_pwm_fan_speed.md)
