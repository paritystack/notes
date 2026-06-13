# 08 · MOSFET drives a 12 V fan

## Overview

The [BJT](../electronics/transistors_bjt.md) of the last rung is current-controlled and
drops a little voltage. For switching real power efficiently you want a
[MOSFET](../electronics/transistors_mosfet.md): **voltage-controlled**, near-zero on-
resistance, and it draws almost no current from your control pin. Here you'll switch a
**12 V fan** from a logic-level signal — your first cross-voltage circuit (12 V load,
5 V/3.3 V control sharing a common ground). This is the exact driver stage your capstone
will use for any motor or high-current output.

```
  N-channel low-side switch:

   12V ──[ fan ]──┐
                  │ D (drain)
   control ─[ 220Ω ]─┬─ G (gate)     gate HIGH → D–S conducts → fan ON
                     │ S (source)     gate LOW  → off
              [ 100k ] (gate pulldown)
                     │
   GND ──────────────┴──────────── (shared with control's GND!)
```

## What you'll need

From **Stage B** of the [shopping list](README.md):

- A **logic-level** N-MOSFET (IRLZ44N — the "L" matters; an IRF540 won't fully turn on at 5 V)
- A 12 V DC fan (or any 12 V/low-current DC load), a 12 V supply (bench PSU or wall-wart)
- A 220 Ω gate resistor, a 100 kΩ gate pulldown, a flyback diode (1N4007)
- Breadboard, multimeter

## The build

1. Fan from **12 V** to the MOSFET **drain**.
2. **Source** to **GND**.
3. **Gate** through **220 Ω** to your control wire; **100 kΩ** from gate to GND (so the gate
   doesn't float when control is disconnected — same idea as [rung 06](06_button_pullup.md)).
4. **Critical:** tie the 12 V supply's GND and your control's GND **together** — they must
   share a common reference.
5. Add a **flyback diode** across the fan (see [rung 09](09_flyback_diode.md)) — a fan is a
   motor, an [inductive load](../electronics/inductors.md).

Touch control to 5 V → fan spins. To GND → stops. Measure **drain-to-source** when ON: a
good logic-level MOSFET reads just tens of millivolts — far less than the BJT's drop.

## It works when…

- [ ] Control HIGH spins the fan; control LOW stops it.
- [ ] Drain–source voltage is very low when ON (efficient switch).
- [ ] The fan runs on 12 V while the control signal is only 5 V, sharing one ground.

## What's happening

A MOSFET's gate is insulated — it's a tiny [capacitor](../electronics/capacitors.md), not a
diode, so in steady state it draws essentially **no current**. Raising the gate voltage
above the threshold `Vgs(th)` creates a conductive channel between drain and source. A
*logic-level* MOSFET is specified to be fully on at `Vgs = 5 V` (even 3.3 V); a standard one
needs ~10 V and would barely conduct from a logic pin — the classic beginner trap. The gate
resistor tames the inrush into that gate capacitance; the pulldown keeps it off when
undriven. This is the canonical [GPIO](../embedded/gpio.md)-to-power-load interface, and you
add [PWM](../embedded/pwm.md) speed control to it in [rung 15](15_pwm_fan_speed.md).

## Pitfalls

- **Using a non-logic-level MOSFET** — an IRF540 from a 5 V pin barely turns on, gets hot, and the fan crawls. Use an IRLZ44N or similar (check `Vgs(th)` in the [datasheet](../electronics/datasheets.md)).
- **No common ground** — the 12 V supply and your control board *must* share GND, or the gate has no reference and nothing works.
- **No flyback diode** — switching a motor without it produces a voltage spike that can destroy the MOSFET. Add it (next rung).
- **Floating gate** — without the pulldown, a disconnected gate can hold charge and leave the fan unpredictably on. Always provide a gate pulldown.

## Where this connects

- [MOSFET Transistors](../electronics/transistors_mosfet.md) — threshold, on-resistance, logic-level vs standard
- [Inductors](../electronics/inductors.md) — why a motor load kicks back when switched off
- [PWM](../embedded/pwm.md) — switching the gate fast to control speed (rung 15)
- **Previous:** [07 · BJT as a switch](07_bjt_switch.md) · **Next:** [09 · Flyback diode](09_flyback_diode.md)
