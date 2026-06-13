# Switches, Relays & Electromechanical

## Overview

A switch is the most intuitive component in electronics: it makes or breaks a
connection. A relay is a switch that another circuit operates *electrically* —
a small [current](charge_current.md) through a coil pulls a metal contact closed,
letting a low-power signal control a high-power load with full electrical
isolation. These electromechanical parts sit between the clean world of
[logic](logic_gates.md) and messy reality: their contacts *bounce*, their coils
are [inductive](inductors.md) and kick back voltage spikes, and switching them
from a microcontroller needs the same [transistor](transistors_bjt.md) /
[MOSFET](transistors_mosfet.md) drive and [flyback diode](diodes.md) you use for
any inductive load. This page is the home for all of it.

```
Relay = electrically-operated switch:

   Control side (low power)        Load side (high power)
   ┌──────────────┐
   │   coil       │  magnetic       ╱  ← contact pulled closed
   │  ((((((( ))) │═══ pull ═══▶   ╱      when coil energised
   └──────────────┘               ●────────● to load
        5 V, 70 mA                 240 V, 5 A   (fully isolated
                                                 from control side)
```

## Key Concepts

### Switch Anatomy: Poles and Throws

A switch is described by how many independent circuits it controls (**poles**) and
how many positions each can connect to (**throws**).

```
  SPST  ──o  o──        Single Pole Single Throw: simple on/off
  SPDT  ──o     o──     Single Pole Double Throw: selects A or B
            ╲ o──         (a changeover / "selector")
  DPDT  two SPDT switches ganged on one actuator (e.g. polarity reversal)
```

| Type | Meaning | Typical use |
|------|---------|-------------|
| SPST | 1 circuit, on/off | Power switch, jumper |
| SPDT | 1 circuit, A-or-B | Mode select, changeover |
| DPDT | 2 circuits, A-or-B | Motor reverse, bypass |
| Push (NO) | Closed only while pressed | Reset, momentary input |
| Push (NC) | Open only while pressed | E-stop, interlock |

### Momentary vs Latching

- **Momentary** — returns to its rest state when released (a pushbutton). The
  circuit must *remember* the press in software or a flip-flop.
- **Latching / maintained** — stays where you put it (a toggle or rocker). The
  position itself holds the state.

### Contact Bounce

A switch contact is springy metal. When it closes, it physically bounces several
times over ~1–10 ms before settling — so one press looks like many to a fast
digital input.

```
  Ideal press:        Real press (bounce):

  ──┐                 ──┐ ┌┐ ┌─┐
    │                   │ ││ │ │
    └────               └─┘└─┘ └────   ← settles after a few ms
```

**Debouncing** removes the bounce:
- **Hardware** — an RC low-pass ([R + C](capacitors.md), e.g. 10 kΩ + 100 nF)
  feeding a Schmitt-trigger input smooths the edge; or an SR latch with an SPDT
  switch.
- **Software** — sample the pin, then ignore further changes for ~20 ms, or
  require N consecutive identical reads. Cheaper and far more common. See
  [GPIO](../embedded/gpio.md).

### The Relay

A relay's coil is an [electromagnet](inductors.md). Energising it moves an
**armature** that opens or closes one or more contact sets:

```
  Relay contacts (SPDT form):

   COM ●───────┐
               ╲          coil de-energised → COM connects to NC
   NC  ●        ●         coil energised   → COM connects to NO
   NO  ●───────╯

   COM = common, NC = normally-closed, NO = normally-open
```

| Spec | Meaning |
|------|---------|
| Coil voltage | Drive level to energise (5 V, 12 V, 24 V…) |
| Coil resistance | Sets coil current: I = V_coil / R_coil |
| Contact rating | Max load the contacts switch (e.g. 10 A @ 250 VAC) |
| NO / NC | Contact state when the coil is *off* |

### Driving a Relay from a Microcontroller

A GPIO pin can't supply the ~50–100 mA a relay coil needs, and the coil is
inductive — so you drive it through a transistor with a **flyback diode**:

```
        +12V
          │
       [relay coil]──┐
          │          │
          ●───►|─────┘   ← flyback diode (cathode to +12V)
          │
          C            NPN BJT (or logic-level MOSFET)
  GPIO──[Rb]──B
          E
          │
         GND
```

**Worked example** — switching a 12 V relay (coil R = 240 Ω) from a 3.3 V GPIO
through an NPN BJT (β = 100):

```
  Coil current  I_C = 12 V / 240 Ω = 50 mA
  Base current  I_B = I_C / β = 50 / 100 = 0.5 mA  (over-drive ×3 for hard saturation → 1.5 mA)
  Base resistor Rb = (3.3 − 0.7) / 0.0015 = 1.7 kΩ → use 1 kΩ
  Flyback diode: 1N4007 across the coil, cathode to +12 V
```

Without the flyback diode the collapsing coil field produces a spike of hundreds
of volts that destroys the transistor — see [Inductors](inductors.md).

### Solid-State Relays (SSR)

An SSR replaces the coil and contacts with an opto-isolator driving a TRIAC or
MOSFET. No moving parts, silent, fast, millions of cycles — but with a small
on-state voltage drop and leakage current. Preferred for frequent switching of AC
loads (heaters, lamps); mechanical relays still win for the lowest contact
resistance and true galvanic isolation at low cost.

## Pitfalls

- **No flyback diode on the coil** — the single most common way to kill the
  driving transistor. Every relay, solenoid, and motor needs one. See
  [Diodes](diodes.md).
- **Ignoring contact bounce** — a counter or interrupt on a raw button will count
  one press as several. Always debounce.
- **Switching inductive AC loads at the contacts** — motors and transformers draw
  a large inrush and arc the contacts on opening, pitting them. Use a snubber
  (R+C across the contacts) or an SSR rated for inductive loads.
- **Exceeding contact rating** — a relay rated 10 A resistive may handle far less
  for an inductive or DC load (DC arcs don't self-extinguish). Check the DC vs AC
  rating separately.
- **Driving the coil directly from a GPIO** — the pin can't source the current and
  has no flyback path. Always use a transistor stage.

## Where this connects

- [Diodes](diodes.md) — the flyback/freewheeling diode that protects the driver
- [Inductors](inductors.md) — the coil is an inductor; the spike is its stored energy
- [BJT Transistors](transistors_bjt.md) — current-driven low-side switch for the coil
- [MOSFET Transistors](transistors_mosfet.md) — voltage-driven alternative for the coil
- [Logic Gates](logic_gates.md) — an SR latch debounces an SPDT switch in hardware
- [Oscillators & the 555 Timer](oscillators.md) — a monostable turns a noisy edge into one clean pulse
- [GPIO](../embedded/gpio.md) — reading buttons and debouncing in firmware
- [Motor Control](../embedded/motor_control.md) — relays and SSRs switch motor power; H-bridges reverse it
```
