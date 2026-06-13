# 15 · PWM-drive the fan (speed control)

## Overview

Time to combine three rungs into one capability: the [pot read](14_adc_pot_serial.md) sets a
[PWM](13_pwm_fade.md) duty cycle that drives the [MOSFET](08_mosfet_fan.md) controlling the
12 V fan. A knob now sets fan **speed** — your first real *actuator under proportional
control*, and the template for any motor, pump, or heater in the capstone. It also surfaces
two real-world gotchas: a motor won't start at very low duty, and PWM whine.

```
   pot → A0 ──► [code: analogRead → analogWrite] ──► D9 (PWM)
                                                       │
                                              [220Ω]── G
   12V ─[fan]─ D(MOSFET) ─ S ─ GND  (+ flyback diode, shared GND)
```

## What you'll need

From **Stage B**: the [MOSFET fan driver](08_mosfet_fan.md) (IRLZ44N, 12 V fan, flyback
diode, gate resistor/pulldown), plus the pot and Nano. **Drive the gate from a PWM pin
(D9)** through the 220 Ω resistor.

## The build

1. Build the [rung-08 driver](08_mosfet_fan.md) but connect the **gate (via 220 Ω) to D9**.
2. Keep the **flyback diode** (rung 09) and the **shared ground** between 12 V supply and Nano.
3. Flash:

```c
const int POT = A0, GATE = 9;

void setup() { pinMode(GATE, OUTPUT); Serial.begin(9600); }

void loop() {
  int raw  = analogRead(POT);          // 0..1023
  int duty = map(raw, 0, 1023, 0, 255);
  // motors often stall below ~20% duty — clamp to a minimum once spinning if needed
  analogWrite(GATE, duty);
  Serial.println(duty);
  delay(50);
}
```

Turn the knob: the fan speeds up and slows down. Note the **dead zone** at the bottom —
below some duty the fan hums but won't spin (not enough torque to overcome static friction).

## It works when…

- [ ] The knob proportionally controls fan speed.
- [ ] You observe a low-end dead zone where the fan won't start.
- [ ] The MOSFET stays cool and switching is clean (flyback diode doing its job).

## What's happening


`analogWrite` on the gate makes the [MOSFET](../electronics/transistors_mosfet.md) switch the
12 V fan on and off fast; the motor's mechanical inertia and inductance average the pulses
into a smooth speed — the load itself is the low-pass filter. The dead zone is physics:
average torque at low duty can't beat static friction, so practical motor control often
clamps to a minimum duty or kick-starts at full power briefly. The audible whine is the
[PWM](../embedded/pwm.md) frequency exciting the coil; raising the PWM frequency above ~20 kHz
moves it out of hearing — a job for direct [timer](../embedded/timers.md) configuration. This
sense→compute→actuate loop is the heart of [motor control](../embedded/motor_control.md).

## Pitfalls

- **Driving the gate from a non-PWM pin** — only `~` pins do `analogWrite` properly. D9 is fine.
- **Dropping the flyback diode** — PWM switches the inductive motor thousands of times a second; without the diode the MOSFET dies fast. Use a *fast* diode for PWM (rung 09).
- **Expecting smooth control to 0** — the dead zone is normal; map your usable range or kick-start.
- **Audible whine** — default ~490 Hz is in-band; raise PWM frequency for silence (advanced timer setup).

## Where this connects

- [08 · MOSFET fan](08_mosfet_fan.md) + [13 · PWM fade](13_pwm_fade.md) + [14 · ADC pot](14_adc_pot_serial.md) — the three rungs this fuses
- [Motor Control](../embedded/motor_control.md) — proper speed/torque control, kick-start, current limiting
- [PWM](../embedded/pwm.md) — frequency selection and resolution trade-offs
- **Previous:** [14 · ADC: read a pot](14_adc_pot_serial.md) · **Next:** [16 · Reaction-timer game](16_reaction_game.md)
