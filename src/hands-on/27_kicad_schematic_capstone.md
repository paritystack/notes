# 27 · KiCad schematic: ATmega328 + power + sensor

## Overview

The capstone begins. You'll capture your [sensor node](26_perfboard_build.md) as a formal
**schematic** in [KiCad](../electronics/kicad_schematic.md) — but built around a *bare*
**ATmega328P**, not an Arduino board. This is where all the theory converges: you'll add the
support parts a chip needs to run (decoupling, crystal, reset, programming header) that the
Nano hid from you, sized using your [datasheet](20_read_a_datasheet.md) skills. The output is
an electrical blueprint, checked by the tool, ready to become a [PCB](28_kicad_pcb_layout.md).

```
   Minimum ATmega328P "Arduino on a chip":

   3V3 ─┬─[100nF]─ GND   (decoupling, next to VCC pins)
        ├── VCC, AVCC
        │
   reset ─[10k]─ 3V3,  + ISP header
   XTAL1/2 ─ 16MHz crystal + 2×22pF (or use internal 8MHz osc, no crystal)
        │
   PC4/PC5 ─ SDA/SCL ─ BME280 + OLED   (your I2C bus)
   header: RX/TX/DTR ─ for USB-serial programming
```

## What you'll need

[KiCad](../electronics/kicad_schematic.md) installed (free), the
[datasheets](20_read_a_datasheet.md) for the ATmega328P, your [LDO](21_ldo_regulator.md), and
the [sensor](17_i2c_sensor.md). No hardware this rung — it's all CAD.

## The build

1. In KiCad's **schematic editor (eeschema)**, place the [symbols](../electronics/symbols.md):
   ATmega328P, the LDO + its [caps](21_ldo_regulator.md), the [BME280](17_i2c_sensor.md), an
   OLED header, a [LiPo/TP4056](22_lipo_charging.md) input, a reset network, and a programming
   header.
2. **Add the chip's support circuitry the Nano gave you for free:**
   - A **100 nF decoupling cap** beside each VCC/AVCC pin (and one bulk cap on the rail).
   - A **16 MHz crystal + two 22 pF caps** on XTAL1/XTAL2 — *or* skip it and run the
     internal 8 MHz oscillator (fewer parts, less accuracy).
   - A **10 kΩ pull-up on RESET** ([rung 06](06_button_pullup.md) logic) plus a reset button.
   - A **programming header** (ISP, or RX/TX/DTR for the [bootloader](29_order_assemble.md) route).
3. **Wire the nets:** power rails, I²C ([PC4=SDA, PC5=SCL](17_i2c_sensor.md)), and label nets
   clearly. Add power-flag symbols so KiCad knows your sources.
4. **Run ERC** (electrical rules check) and fix every error — unconnected pins, missing power
   flags, conflicting outputs. A clean ERC is the gate to layout.
5. **Assign footprints** to every symbol (DIP-28 for the ATmega, 0805 or through-hole passives,
   the sensor/header footprints).

## It works when…

- [ ] Every part from your perfboard node is on the schematic, plus the chip's support circuitry.
- [ ] ERC passes with no errors (warnings understood).
- [ ] Every symbol has a footprint assigned, ready for the PCB step.

## What's happening

A schematic is the *electrical* description — what connects to what — independent of physical
layout. The reason a bare [ATmega328](../embedded/avr.md) needs parts the Nano didn't is that
the dev board already integrated them: decoupling [caps](../electronics/capacitors.md) steady
the supply at the chip's pins, the crystal gives an accurate clock (or you trade accuracy for
the internal RC oscillator), and the reset pull-up keeps the chip running. **ERC** is your
compiler for hardware — it catches floating pins, shorts, and missing power before they become
copper. This schematic, plus footprints, is everything the [PCB editor](28_kicad_pcb_layout.md)
needs. See [Circuit Design](../electronics/circuit_design.md) for the why behind each choice.

## Pitfalls

- **Skipping decoupling caps** — an undecoupled MCU is unreliable: resets, glitches, noise. One 100 nF per supply pin, placed close, is mandatory.
- **Forgetting power-flag symbols** — KiCad's ERC complains it can't tell what drives a net; add PWR_FLAG to your sources.
- **Crystal load caps wrong** — use the value the crystal datasheet implies (~22 pF typical); wrong caps mean the clock won't start. Or sidestep with the internal oscillator.
- **No programming header** — you must be able to get firmware *into* the bare chip; design in an ISP or serial-bootloader header now ([rung 30](30_bringup.md)).
- **Unassigned/wrong footprints** — a DIP symbol with an SMD footprint (or vice versa) yields an unbuildable board. Match deliberately.

## Where this connects

- [KiCad Schematic](../electronics/kicad_schematic.md) — eeschema workflow, ERC, footprint assignment
- [Circuit Design](../electronics/circuit_design.md) / [Symbols](../electronics/symbols.md) — the concepts and symbol reference
- [AVR](../embedded/avr.md) — what the bare ATmega328 needs to run
- **Previous:** [26 · Perfboard build](26_perfboard_build.md) · **Next:** [28 · PCB layout](28_kicad_pcb_layout.md)
