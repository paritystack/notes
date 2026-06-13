# 29 · Order from JLCPCB, assemble & hand-solder

## Overview

You have [Gerbers](28_kicad_pcb_layout.md). Now turn files into a physical board: upload to a
fab, wait ~1–2 weeks, then [solder](25_learn_to_solder.md) your components onto the bare PCB
that arrives. This is the moment the project becomes real hardware you can hold. The risk is
that any mistake is now baked into copper — so you'll do final checks before ordering, then
assemble carefully and verify before applying power.

```
   Gerbers ──► upload ──► fab makes boards ──► ship ──► populate ──► verify ──► power
                (JLCPCB)    (~$2–5 for 5)              (solder parts)
```

## What you'll need

From **Stage E**: your DRC-clean [Gerbers](28_kicad_pcb_layout.md), a fab account (JLCPCB or
similar), the [ATmega328P](27_kicad_schematic_capstone.md) + all components, your
[soldering kit](25_learn_to_solder.md), and the multimeter.

## The build

1. **Pre-flight the Gerbers** in KiCad's viewer one last time — silkscreen readable,
   designators present, drill aligned, board outline closed. Mistakes here cost a reorder.
2. **Order:** zip the Gerbers, upload to [JLCPCB](../electronics/kicad_pcb.md), confirm the
   rendered preview matches your design, pick the cheapest options (5 boards, standard
   thickness/colour), and order. ~$2–5 + shipping.
3. **When boards arrive, assemble** in a sensible order: lowest/most-heat-sensitive parts and
   the [decoupling caps](27_kicad_schematic_capstone.md) first, then the
   [ATmega328](../embedded/avr.md), then headers and the [sensor](17_i2c_sensor.md), connectors
   last. Mind polarity on caps, diodes, the chip (pin-1 dot), and the LiPo input.
4. **Verify before power** ([rung 26](26_perfboard_build.md) discipline): continuity-check the
   power and ground nets, confirm **no short between VCC and GND**, check pin-1 orientation.
5. **First power-up current-limited:** feed it from the [bench supply](21_ldo_regulator.md)
   with a low current limit. If it pulls a sane current (not slamming the limit), measure that
   your [LDO](21_ldo_regulator.md) makes the right rail voltage at the chip's pins. *Then*
   proceed to [bring-up](30_bringup.md).

## It works when…

- [ ] The fabricated board matches your design (no obvious copper/silk errors).
- [ ] All parts are soldered with [good joints](25_learn_to_solder.md) and correct polarity.
- [ ] No VCC-to-GND short; first powered current is sane and the regulated rail reads correct.

## What's happening

A fab takes your [Gerbers](28_kicad_pcb_layout.md) and photolithographically etches the copper,
drills holes, applies soldermask and silkscreen — producing the bare board cheaply because
many designs are panelised together. Assembly is just [soldering](25_learn_to_solder.md)
applied to your own layout; doing it in a height/heat-sensible order and respecting polarity
prevents most failures. The pre-power checks exist because a VCC-GND short or a backwards
[regulator](21_ldo_regulator.md)/cap can destroy parts (or a [LiPo](22_lipo_charging.md))
instantly — current-limiting the first power-up turns a potential bang into a harmless "huh,
that's too much current." A clean rail at the chip's pins is the green light for
[programming](30_bringup.md).

## Pitfalls

- **Ordering before a final Gerber review** — the #1 way to waste two weeks and a reorder. Check the fab's rendered preview against your intent.
- **Polarity mistakes** — chip pin-1, electrolytic caps, diodes, LiPo input. Backwards parts can destroy the board on power-up. Verify each.
- **Cold joints / bridges** — re-inspect under light/magnification; buzz adjacent pins. A single bad joint can mean a dead board that's maddening to debug.
- **Powering at full current first** — skip the current-limited first power-up and a short becomes smoke. Always limit current on the maiden power-on.

## Where this connects

- [KiCad PCB](../electronics/kicad_pcb.md) — Gerber export and the JLCPCB ordering flow
- [25 · Learn to solder](25_learn_to_solder.md) — the assembly skill applied here
- [22 · LiPo charging](22_lipo_charging.md) — safe power-up of a battery board
- **Previous:** [28 · PCB layout](28_kicad_pcb_layout.md) · **Next:** [30 · Bring-up](30_bringup.md)
