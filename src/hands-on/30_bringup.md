# 30 · Bring-up — program the bare chip, debug · **the capstone**

## Overview

A powered, correctly-built [board](29_order_assemble.md) that does nothing yet. **Bring-up**
is the methodical process of getting first firmware into your bare
[ATmega328](../embedded/avr.md) and proving each subsystem alive — power, clock, programming,
[I²C](17_i2c_sensor.md), sleep. When the sensor reading appears on the OLED and the board sips
microamps between updates, **the capstone is complete**: a device you specified, schematic-
captured, laid out, fabricated, assembled, and programmed — entirely yourself.

```
   Bring-up order (verify each before the next):

   power rail OK ─► chip responds to programmer ─► blink (clock OK)
     ─► serial prints ─► I2C devices found ─► sensor reads ─► sleep/wake ─► DONE
```

## What you'll need

The assembled [board](29_order_assemble.md) with a verified power rail, a
[USB-serial adapter](27_kicad_schematic_capstone.md) (or an ISP programmer — e.g. another
Arduino as ISP), and your firmware from the earlier rungs.

## The build

1. **Burn the bootloader / set fuses.** A factory-fresh ATmega328 runs its internal 8 MHz
   RC oscillator divided by 8 (the `CKDIV8` fuse) — so an effective **1 MHz** — and has no
   bootloader. Use *Arduino as ISP* (or a dedicated programmer) to set the
   [fuses](../embedded/avr.md) for your clock choice (8 MHz internal RC *or* an 8 MHz
   crystal on the 3.3 V rail — must match [rung 27](27_kicad_schematic_capstone.md);
   16 MHz is out of spec at 3.3 V) and burn a bootloader if you want serial uploads.
2. **Smallest possible first program — blink** an LED on a spare pin. If it blinks, power,
   clock, and programming all work. (If not, debug *those* before anything else.)
3. **Bring up serial:** print "hello" over [UART](../embedded/uart.md) via the USB-serial
   header. Confirms the [clock](../embedded/avr.md) speed (wrong fuses → garbled baud).
4. **Bring up I²C:** run the [scanner](17_i2c_sensor.md) — the [sensor](17_i2c_sensor.md) and
   [OLED](18_i2c_oled.md) addresses should appear. Then run the full sensor-read-and-display
   firmware.
5. **Bring up low power:** enable [deep sleep](24_deep_sleep.md) and confirm microamp sleep
   current on this board.
6. **Debug methodically** when something fails: isolate to the stage, check the
   [datasheet](20_read_a_datasheet.md), measure with the [multimeter](01_bench_and_multimeter.md)
   /[logic analyzer](19_logic_analyzer_i2c.md), and only change one thing at a time.

## It works when…

- [ ] The bare chip accepts firmware and blinks (power + clock + programming proven).
- [ ] Serial prints cleanly and the I²C scan finds the sensor and display.
- [ ] **The capstone runs: live sensor data on the OLED, from battery, sleeping at microamps between updates.**

## What's happening

Bring-up is **divide-and-conquer debugging** applied to hardware: rather than flash the full
app onto an unproven board and stare at a dead device, you verify the foundation
(power → clock → programmability) then add one subsystem at a time, so any failure points at
exactly one thing. A blank [AVR](../embedded/avr.md) needs its **fuses** set to pick the clock
source — a frequent first-board snag, since wrong fuses give a dead or wrongly-clocked chip
(garbled serial is the classic symptom). Each layer reuses a tool you already know — the
[scanner](17_i2c_sensor.md), the [logic analyzer](19_logic_analyzer_i2c.md), the
[multimeter](01_bench_and_multimeter.md) — which is why the whole roadmap built toward this.
Finishing it means you can take an idea to working custom hardware on your own.

## Pitfalls

- **Wrong fuses / clock mismatch** — selecting external-crystal fuses with no crystal fitted (or vice versa) bricks the chip until you reprogram it via ISP with the correct settings. Match fuses to [rung 27](27_kicad_schematic_capstone.md).
- **Flashing the full app first** — hides where the fault is. Start with blink; add one subsystem at a time.
- **Programmer wiring/orientation** — ISP/serial header reversed = no comms. Check the header pinout you designed.
- **Garbled serial** — almost always a clock/fuse/baud mismatch, not a code bug. Verify the clock first.
- **Changing many things at once** — the cardinal debugging sin; isolate and change one variable per test.

## Where this connects

- [AVR](../embedded/avr.md) — fuses, clock sources, ISP vs bootloader programming
- [Debugging](../embedded/debugging.md) — methodical bring-up and fault isolation
- [The roadmap](README.md) — every prior rung was a piece of this board; you've finished the capstone
- **Previous:** [29 · Order & assemble](29_order_assemble.md) · **Next (optional):** [31 · ESP32 WiFi](31_esp32_wifi.md)
