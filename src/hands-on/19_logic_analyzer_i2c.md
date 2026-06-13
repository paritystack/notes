# 19 · Logic analyzer — capture & decode the I²C bus

## Overview

So far you've *trusted* that [I²C](../embedded/i2c.md) works. Now you'll **see it** — every
start condition, address, ACK, and data byte on the wire. A cheap logic analyzer plus open-
source software ([sigrok/PulseView](../embedded/debugging.md)) turns an invisible protocol
into a labelled timing diagram. For someone from software this is the missing `tcpdump`/
debugger for hardware buses, and it's how you'll diagnose every flaky [SPI](../embedded/spi.md)/
I²C/[UART](../embedded/uart.md) link from here on.

```
   Logic analyzer taps the bus in parallel (it only listens):

   Nano A4(SDA) ──┬── sensor/OLED        LA CH0 ── SDA
   Nano A5(SCL) ──┴── sensor/OLED        LA CH1 ── SCL
                                         LA GND ── common GND
   PulseView I2C decoder:  S | 0x76 W | A | 0xFA | A | ... | P
```

## What you'll need

From **Stage B**: an 8-channel USB logic analyzer (the ~$12 clone works with PulseView), the
running [sensor/OLED](18_i2c_oled.md) setup, and [PulseView](https://sigrok.org/) installed.

## The build

1. Connect **LA CH0 → SDA (A4)**, **CH1 → SCL (A5)**, and **LA GND → Nano GND** (shared
   ground is mandatory — the analyzer measures voltages relative to it).
2. In PulseView: set ~1–4 MHz sample rate, add a trigger on SCL falling, and **Run** while
   the Nano polls the sensor.
3. Add the **I²C protocol decoder**, assign SDA/SCL. It annotates the capture:

```
   S  Start
   0x76+W   address + write bit
   A        ACK from sensor
   0xFA     register pointer
   ... repeated start, read bytes, NACK, Stop
```

You can now literally read the conversation your library is having with the sensor — and
compare it against the [datasheet](../electronics/datasheets.md)'s register map.

## It works when…

- [ ] You capture activity on SDA/SCL while the Nano talks to the sensor.
- [ ] The I²C decoder shows the start, the address you scanned, and ACK bits.
- [ ] You can point to a byte on screen and find its meaning in the sensor datasheet.

## What's happening

A logic analyzer samples each line as 1/0 many times per microsecond and stores the stream;
the protocol decoder applies [I²C](../embedded/i2c.md) rules (start = SDA falls while SCL
high, 8 bits MSB-first, 9th = ACK) to reconstruct bytes. Because it's a high-impedance
*listener* tapped in parallel, it doesn't disturb the bus — unlike a multimeter, it shows
*timing and sequence*, which is what protocol bugs live in. This is the tool that tells you
whether "no data" is a wiring fault, a wrong address, a missing [pull-up](06_button_pullup.md),
or a software bug — collapsing hours of guessing into one capture.

## Pitfalls

- **No common ground** — the analyzer needs Nano GND or every reading is meaningless. The most common first mistake.
- **Sample rate too low** — sample at least 4–10× the bus clock (≥1 MHz for 100 kHz I²C) or edges alias and the decoder fails.
- **Probing the wrong pins** — A4=SDA, A5=SCL; mixing them up makes the decoder spit nonsense.
- **Expecting analog detail** — a logic analyzer sees only HIGH/LOW, not ringing or levels; use a [scope](../electronics/prototyping.md) for analog signal-quality issues.

## Where this connects

- [I²C](../embedded/i2c.md) / [SPI](../embedded/spi.md) / [UART](../embedded/uart.md) — buses you'll decode the same way
- [Debugging](../embedded/debugging.md) — adding the logic analyzer to your debug arsenal
- [Prototyping & Test Equipment](../electronics/prototyping.md) — where the analyzer sits vs. the scope
- **Previous:** [18 · I²C OLED](18_i2c_oled.md) · **Next:** [20 · Read a datasheet](20_read_a_datasheet.md)
