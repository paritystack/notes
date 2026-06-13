# 16 · Reaction-timer game

## Overview

This rung is a reward and a checkpoint: no new components, just everything from Phase 2
woven into something that *feels like a product*. A reaction-timer game lights an LED after a
random delay; you hit the [button](12_button_debounce.md) as fast as you can; it reports your
time over [serial](14_adc_pot_serial.md) (and beeps with the [buzzer](07_bjt_switch.md)).
It exercises [GPIO](../embedded/gpio.md) in and out, [timing](../embedded/timers.md),
randomness, and state — the kind of small finite-state-machine logic every real
[device](../embedded/state_machines.md) runs on.

```
   States:  WAIT ──(random 2–5 s)──► GO ──(press)──► SHOW time ──(press)──► WAIT
              │                                          
   if pressed during WAIT → "too soon!"
```

## What you'll need

From **Stage A**: Arduino Nano, a button (D2), an LED (D13), optionally a buzzer on a
[BJT driver](07_bjt_switch.md) (D8). Nothing new to buy.

## The build

```c
const int BTN = 2, LED = 13;
enum State { WAIT, GO, SHOW };
State state = WAIT;
unsigned long goTime = 0, waitUntil = 0;

bool pressed() {                 // simple edge detect (debounce as in rung 12)
  static int last = HIGH;
  int now = digitalRead(BTN);
  bool edge = (last == HIGH && now == LOW);
  last = now;
  delay(5);                      // crude debounce; reuse rung-12 logic for real
  return edge;
}

void setup() {
  pinMode(BTN, INPUT_PULLUP); pinMode(LED, OUTPUT);
  Serial.begin(9600); randomSeed(analogRead(A0));
  waitUntil = millis() + random(2000, 5000);
}

void loop() {
  switch (state) {
    case WAIT:
      if (pressed()) { Serial.println("Too soon!"); waitUntil = millis() + random(2000,5000); }
      if (millis() > waitUntil) { digitalWrite(LED, HIGH); goTime = millis(); state = GO; }
      break;
    case GO:
      if (pressed()) {
        Serial.print("Reaction: "); Serial.print(millis() - goTime); Serial.println(" ms");
        digitalWrite(LED, LOW); state = SHOW;
      }
      break;
    case SHOW:
      if (pressed()) { waitUntil = millis() + random(2000,5000); state = WAIT; }
      break;
  }
}
```

Play it. Try to beat 200 ms. Press too early and it scolds you. That branching-on-state
structure is a [state machine](../embedded/state_machines.md) — the backbone of real firmware.

## It works when…

- [ ] After a random delay the LED lights and your reaction time prints in ms.
- [ ] Pressing before the LED lights reports "too soon" and restarts.
- [ ] The game loops cleanly without freezing (no blocking `delay()` in the wait).

## What's happening

The program is an explicit [state machine](../embedded/state_machines.md): `WAIT`, `GO`,
`SHOW`, with events (button edges, elapsed [time](../embedded/timers.md)) driving
transitions. `millis()` gives non-blocking timing so the loop stays responsive — the lesson
from [rung 12](12_button_debounce.md). `randomSeed(analogRead(A0))` exploits the noisy
floating [ADC](../embedded/adc.md) input as an entropy source. You're now combining input,
output, timing, and control flow — exactly the ingredients of the sensor-node capstone, just
arranged for fun.

## Pitfalls

- **Blocking the loop with `delay()`** — long delays make the button feel dead. Keep timing in `millis()` comparisons.
- **No debounce** — bounce double-fires the edge detector; fold in the [rung-12](12_button_debounce.md) debounce for a clean feel.
- **Fixed (non-random) delay** — players learn the rhythm. Re-seed and randomise each round.
- **Driving a buzzer straight from a pin** — use the [BJT driver](07_bjt_switch.md) for anything but a tiny piezo.

## Where this connects

- [State Machines](../embedded/state_machines.md) — the pattern this game is built on
- [Timers](../embedded/timers.md) / [Interrupts](../embedded/interrupts.md) — precise, non-blocking timing and event handling
- [12 · Button debounce](12_button_debounce.md) — reuse its debounce for crisp input
- **Previous:** [15 · PWM fan speed](15_pwm_fan_speed.md) · **Next:** [17 · I²C sensor](17_i2c_sensor.md)
