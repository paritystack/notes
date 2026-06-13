# Hands-On Electronics

## Overview

This section is the *doing* counterpart to the [Electronics](../electronics/README.md) theory pages. Where that section explains how [resistance](../electronics/resistance.md), [capacitors](../electronics/capacitors.md), [transistors](../electronics/transistors_mosfet.md), and [power supplies](../electronics/power_supplies.md) work, this one walks you through *building* real circuits — one small evening-sized win at a time — until you can design, route, and assemble your own [PCB](../electronics/kicad_pcb.md). It's written for someone comfortable with software/firmware but new to hardware: the **hardware is the new thing**, so the code stays deliberately simple and every step ends with something that visibly *works*.

Each rung links back to the theory page it puts into practice, and forward to the [Embedded](../embedded/README.md) concepts ([GPIO](../embedded/gpio.md), [PWM](../embedded/pwm.md), [ADC](../embedded/adc.md), [power management](../embedded/power_management.md)) you'll lean on once a microcontroller joins the breadboard.

```
The journey, end to end:

  multimeter + LED  ──►  passives & transistors  ──►  microcontroller
        │                                                    │
        ▼                                                    ▼
   "it lights up"                                    sensing + power + battery
                                                             │
                                                             ▼
                                              solder it down  ──►  CUSTOM PCB
                                                                  (sensor node)
```

## The capstone

Everything ladders toward one device: a **battery-powered sensor node** — it reads a
sensor over [I²C](../embedded/i2c.md), shows or logs the result, runs from a LiPo cell,
sleeps to save power, and lives on a PCB *you* designed around a bare
**ATmega328P**. Optionally it sends its readings over WiFi via an ESP32 module.
By the time you build it, every part of it will be something you've already done in
isolation.

## How to use this section

- **One rung per session.** Each project is sized for a single evening (~1–2 h) and ends with a clear *"it works when…"* check. Don't skip ahead — each build assumes the last.
- **Prototype on an Arduino Nano clone**, then graduate the capstone to a bare ATmega328P on your own board. Same chip, so nothing you learn is wasted.
- **Low voltage only.** Everything here stays ≤ 24 V DC (USB 5 V, LiPo 3.7 V, the odd 12 V fan). No mains. No exceptions — see [Switches & Relays](../electronics/switches_relays.md) for why.
- **Measure everything.** Get into the habit of confirming with the [multimeter](../electronics/prototyping.md) what you *think* is happening. This is the firmware-debugging instinct applied to hardware.

## The ladder

### Phase 0 — Bench & Ohm's law (no microcontroller yet)

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 01 | [Set up the bench & meet the multimeter](01_bench_and_multimeter.md) | You measure a battery and buzz out a wire | [Prototyping](../electronics/prototyping.md), [Voltage](../electronics/voltage.md) |
| 02 | [Light an LED (calculate the resistor)](02_light_an_led.md) | An LED glows, and you predicted its current | [Resistance](../electronics/resistance.md), [Diodes](../electronics/diodes.md) |
| 03 | [Series & parallel — measure the voltage drops](03_series_parallel.md) | Meter readings match your math | [Circuits](../electronics/circuits.md) |
| 04 | [Potentiometer as a voltage divider, dim an LED](04_pot_divider.md) | A knob smoothly dims the LED | [Resistance](../electronics/resistance.md) |

### Phase 1 — Passives & analog feel

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 05 | [RC circuit — charge & discharge, watch a fade](05_rc_fade.md) | An LED fades out on its own | [Capacitors](../electronics/capacitors.md) |
| 06 | [Button input with a pull-up/pull-down](06_button_pullup.md) | A button changes an LED, no flicker | [Switches & Relays](../electronics/switches_relays.md) |
| 07 | [BJT as a switch — drive a buzzer](07_bjt_switch.md) | A tiny current switches a big load | [BJT Transistors](../electronics/transistors_bjt.md) |
| 08 | [MOSFET drives a 12 V fan](08_mosfet_fan.md) | The fan spins from a logic-level signal | [MOSFET Transistors](../electronics/transistors_mosfet.md) |
| 09 | [Flyback diode on an inductive load](09_flyback_diode.md) | A relay/motor switches without spikes | [Diodes](../electronics/diodes.md), [Inductors](../electronics/inductors.md) |
| 10 | [555 timer blink (an oscillator, no code)](10_555_blink.md) | An LED blinks with zero firmware | [Oscillators](../electronics/oscillators.md) |

### Phase 2 — Enter the microcontroller (Arduino Nano)

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 11 | [Firmware blink — the hardware "hello, world"](11_firmware_blink.md) | An LED blinks under code control | [GPIO](../embedded/gpio.md) |
| 12 | [Read a button + debounce in code](12_button_debounce.md) | A clean press toggles an LED | [GPIO](../embedded/gpio.md) |
| 13 | [PWM fade](13_pwm_fade.md) | An LED breathes smoothly | [PWM](../embedded/pwm.md) |
| 14 | [ADC — read a pot, print over serial](14_adc_pot_serial.md) | A number changes as you turn the knob | [ADC](../embedded/adc.md), [Sensors](../electronics/sensors.md) |
| 15 | [PWM-drive the fan (speed control)](15_pwm_fan_speed.md) | A knob sets fan speed | [MOSFET Transistors](../electronics/transistors_mosfet.md), [PWM](../embedded/pwm.md) |
| 16 | [Reaction-timer game](16_reaction_game.md) | Your first thing that feels like a *device* | — |

### Phase 3 — Sensing & I/O

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 17 | [I²C scan + read a temp/humidity sensor](17_i2c_sensor.md) | Real temperature printed over serial | [Sensors](../electronics/sensors.md), [I²C](../embedded/i2c.md) |
| 18 | [Drive an I²C OLED display](18_i2c_oled.md) | The reading shows on a tiny screen | [Sensors](../electronics/sensors.md) |
| 19 | [Logic analyzer — capture & decode the I²C bus](19_logic_analyzer_i2c.md) | You *see* the bytes on the wire | [Prototyping](../electronics/prototyping.md) |
| 20 | [Read one real datasheet end-to-end](20_read_a_datasheet.md) | You can size & wire a part from its datasheet | [Reading a Datasheet](../electronics/datasheets.md) |

### Phase 4 — Power & battery

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 21 | [LDO regulator — measure dropout & heat](21_ldo_regulator.md) | A stable 3.3 V rail from 5–12 V in | [Power Supplies](../electronics/power_supplies.md) |
| 22 | [LiPo charging (TP4056) + protection](22_lipo_charging.md) | A cell charges safely from USB | [Power Supplies](../electronics/power_supplies.md) |
| 23 | [Run the whole thing on battery, measure draw](23_battery_current.md) | It runs untethered; you know the mA | [Power & Energy](../electronics/power.md) |
| 24 | [Deep-sleep the AVR, measure microamps](24_deep_sleep.md) | Current drops 1000× when idle | [Power Management](../embedded/power_management.md) |

### Phase 5 — Solder & permanence

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 25 | [Learn to solder (practice kit)](25_learn_to_solder.md) | Shiny joints, no bridges | [Prototyping](../electronics/prototyping.md) |
| 26 | [Move the sensor-node circuit to perfboard](26_perfboard_build.md) | Your first *permanent* build | [Prototyping](../electronics/prototyping.md) |

### Phase 6 — Custom PCB  ·  **the capstone**

| # | Build | Ends with | Theory |
|---|-------|-----------|--------|
| 27 | [KiCad schematic: ATmega328 + power + sensor](27_kicad_schematic_capstone.md) | A captured, ERC-clean schematic | [KiCad Schematic](../electronics/kicad_schematic.md), [Circuit Design](../electronics/circuit_design.md), [Symbols](../electronics/symbols.md) |
| 28 | [PCB layout — route, DRC, export Gerbers](28_kicad_pcb_layout.md) | Manufacturing files ready to send | [KiCad PCB](../electronics/kicad_pcb.md) |
| 29 | [Order from JLCPCB, assemble & hand-solder](29_order_assemble.md) | A populated board in your hand | [KiCad PCB](../electronics/kicad_pcb.md) |
| 30 | [Bring-up — program the bare chip, debug](30_bringup.md) | **The capstone runs on your own board** | [GPIO](../embedded/gpio.md) |
| 31 | [*(optional)* ESP32 module — send data over WiFi](31_esp32_wifi.md) | Readings appear on your phone/PC | [Sensors](../electronics/sensors.md) |

## Staged shopping list (~$480)

Buy each stage only when you reach it — there's no point owning a scope before you have a signal to look at. Prices are rough USD for clones/budget parts.

```
STAGE A · Start tonight                                    ~$110
  Digital multimeter (e.g. Aneng AN8008 or similar)         $40
  Breadboard + jumper wire kit                              $12
  Component starter kit                                     $25
    (resistors, LEDs, ceramic + electrolytic caps,
     2N2222/BC547 BJTs, push buttons, 10k pots, diodes)
  Arduino Nano clone  x2  (one to keep on the breadboard)   $12
  9V battery + holder / USB power                            $8
  555 timers, piezo buzzer, small 12 V fan                  $13

STAGE B · MCU & sensing                                     ~$70
  BME280 temp/humidity/pressure sensor (I2C)                 $8
  SSD1306 0.96" OLED (I2C)                                   $6
  8-channel USB logic analyzer (works with PulseView)       $12
  IRLZ44N logic-level N-MOSFETs + gate resistors             $8
  TP4056 charger, relay module, small DC motor              $20
  Breadboard power-supply module + extra jumpers            $16

STAGE C · Power & soldering                                ~$140
  Pinecil V2 soldering iron + USB-C PD power supply         $60
  Solder, flux, tip cleaner, perfboard, helping hands       $30
  Bench power supply, 30 V / 5 A adjustable (budget)        $50

STAGE D · Seeing signals                                   ~$130
  Oscilloscope:
    FNIRSI handheld 2-ch (~$120)  ← within budget
    (upgrade path: used Rigol DS1054Z ~$350 when ready)
  LiPo cell(s) + holder                                     $10

STAGE E · Capstone PCB                                      ~$30
  ATmega328P-PU (DIP-28)  x3  + 16 MHz crystals + caps      $15
  USB-to-serial adapter (CP2102 / FTDI)                      $6
  Custom PCB from JLCPCB (5 boards + shipping)               $7
  ESP32 dev module (optional WiFi)                           $6
```

A multimeter, breadboard, and a $40 starter kit are genuinely all you need to start
Phase 0 tonight. Everything else is just-in-time.

## Where this connects

- [Electronics](../electronics/README.md) — the theory every rung here puts into practice; read the matching page before (or right after) each build
- [Prototyping & Test Equipment](../electronics/prototyping.md) — your breadboard, multimeter, soldering, and scope reference
- [Embedded](../embedded/README.md) — once the microcontroller arrives in Phase 2, its peripherals ([GPIO](../embedded/gpio.md), [PWM](../embedded/pwm.md), [ADC](../embedded/adc.md), [I²C](../embedded/i2c.md)) are what your code drives
- [Circuit Design](../electronics/circuit_design.md) → [KiCad Schematic](../electronics/kicad_schematic.md) → [KiCad PCB](../electronics/kicad_pcb.md) — the toolchain for the Phase 6 capstone
