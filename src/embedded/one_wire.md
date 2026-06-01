# 1-Wire

## Overview

1-Wire is a single-data-line bus (plus ground) for low-speed sensors and ID chips, made famous by the **DS18B20 temperature sensor** and Dallas/Maxim iButtons. It is the minimalist cousin of [I2C](i2c.md): where I2C needs two wires and 7-bit addresses, 1-Wire needs **one wire** and gives every device a unique 64-bit factory ROM ID, with an option to draw power from the data line itself ("parasitic power") so a sensor runs on just two wires total. The tradeoff is strict, microsecond-level **timing slots** bit-banged on a single [GPIO](gpio.md), and very low throughput (~16 kbps). It is the go-to bus for "I need a few temperature probes on a long cheap cable."

```
                 Rpull (4.7 kΩ)
   3V3 ───────────┳───────────────┳───────────────┳──
                  │               │               │
   GPIO ──────────┻──── DQ ───────┻──── DQ ───────┻── DQ
   (open-drain)        │               │               │
   GND ────────────────┴───────────────┴───────────────┴
                    Sensor 1       Sensor 2       Sensor 3
              (each with unique 64-bit ROM ID)
```

The master idles the line high through a pull-up; any device (or the master) pulls it low to signal. It is an **open-drain, single-master** bus — exactly the electrical model of [I2C](i2c.md), just with one wire doing both directions.

## Parasitic Power

The defining trick: a device can run with its VDD pin tied to GND, harvesting energy from DQ while the line is high and storing it in a small internal cap.

```
 Normal (3 wires)            Parasitic (2 wires)
   VDD ─── 3V3                VDD ─── GND  (!)
   DQ  ─── data               DQ  ─── data + power
   GND ─── GND                GND ─── GND
```

During power-hungry operations (a DS18B20 temperature conversion draws ~1.5 mA) the master must apply a **strong pull-up** — drive DQ hard high (push-pull) for the conversion time (up to 750 ms at 12-bit) instead of relying on the weak resistor. Get this wrong and parasitic devices brown out mid-conversion and return 85 °C (the power-on default).

## Timing: Reset, Slots

Everything is the master toggling DQ low for precise durations and sampling. Three primitives:

```
RESET / PRESENCE:
  Master:  ───┐                              ┌──────────
              └──── 480 µs low ─────────────┘
  Slave:   ──────────────┐        ┌──────────────────────
                         └ presence┘   (slave pulls low
                          pulse ~60µs    → "I'm here")

WRITE 1 slot:           WRITE 0 slot:
  ─┐    ┌────────         ─┐                  ┌────
   └6µs─┘                  └──── 60µs low ────┘
   (release fast)          (hold low whole slot)

READ slot:
  ─┐  ┌── slave drives this region ──
   └6µs┘  master samples ~15µs in:
          line still low = 0, released high = 1
```

A single bit takes ~60–70 µs, so a full 9-byte scratchpad read is on the order of a millisecond. Because the windows are tight, the bit-bang must run with [interrupts](interrupts.md) masked during a slot — an ISR firing mid-slot stretches a "write 1" into a "write 0".

## ROM Addressing and Search

Every device has a 64-bit ROM code: `8-bit family code | 48-bit serial | 8-bit CRC`. Transactions follow a fixed two-phase shape:

```
[ Reset+Presence ] → [ ROM command ] → [ Function command + data ]
```

| ROM command | Use |
|-------------|-----|
| **Read ROM** (0x33) | Read the 64-bit ID — only valid with *one* device on the bus |
| **Match ROM** (0x55) | Address one specific device by its 64-bit ID |
| **Skip ROM** (0xCC) | Broadcast to all (e.g. "all sensors: start conversion") |
| **Search ROM** (0xF0) | Discover all IDs on a multi-drop bus |

**Search ROM** is the clever part: it's a binary-tree walk of the 64-bit ID space. At each bit the master reads the bit and its complement from all devices ANDed onto the wire; `0,1`→all have 0 here, `1,0`→all have 1, `0,0`→a discrepancy (some 0, some 1) so the master picks a branch and revisits the other later. N devices are enumerated in N passes — this is how you find every probe on a shared cable without pre-knowing the IDs.

## DS18B20 Example Flow

```c
// Single-sensor read (Skip ROM works with one device)
ow_reset();                 // reset + check presence pulse
ow_write_byte(0xCC);        // Skip ROM
ow_write_byte(0x44);        // Convert T
//   ... wait up to 750ms (or poll the read slot until it reads 1) ...
ow_reset();
ow_write_byte(0xCC);        // Skip ROM
ow_write_byte(0xBE);        // Read Scratchpad
for (int i = 0; i < 9; i++) scratch[i] = ow_read_byte();
// CRC-check scratch[8], temp = (scratch[1]<<8 | scratch[0]) / 16.0
```

The 9th scratchpad byte is a **CRC-8** (Dallas/Maxim polynomial) over the first 8 — always verify it; a long noisy cable flips bits.

## Hardware vs Bit-Bang

| Approach | Notes |
|----------|-------|
| **GPIO bit-bang** | Most common. Open-drain output, weak pull-up; mask interrupts per slot |
| **UART trick** | Drive 1-Wire from a [UART](uart.md) at specific baud rates: one byte = one bit slot, timing handled by the UART hardware — robust under interrupt load |
| **Dedicated 1-Wire master** (DS2482 over [I2C](i2c.md)) | Offloads all timing; good when CPU can't guarantee µs timing |

## Where this connects

- [I2C](i2c.md) — same open-drain, single-master, pull-up electrical model; 1-Wire trades the second wire for tight timing.
- [GPIO](gpio.md) — the bus is usually one open-drain GPIO bit-banged in software.
- [Interrupts](interrupts.md) — must be masked during a timing slot, or a slot stretches and corrupts the bit.
- [UART](uart.md) — the classic robust implementation maps one bit-slot to one UART byte.
- [Sensors & Sensor Fusion](sensors.md) — DS18B20 and friends feed the sensor signal chain.

## Pitfalls

1. **Interrupt during a slot.** An ISR firing mid-write-1 holds the line low too long and the device reads a 0. Mask interrupts around each slot (or use the UART method).
2. **Parasitic power without strong pull-up.** The sensor browns out during conversion and returns 85 °C (power-on default). Drive DQ hard-high for the conversion time.
3. **Read ROM on a multi-drop bus.** With more than one device, Read ROM (0x33) returns the AND of all IDs — garbage. Use Search ROM to enumerate, then Match ROM.
4. **Skipping the CRC check.** Long cables flip bits; an unchecked scratchpad gives wildly wrong temperatures. Verify the CRC-8.
5. **Missing the presence pulse.** No presence pulse after reset means wiring/pull-up fault — abort instead of reading zeros.
6. **Pull-up too weak for cable length.** Tens of metres of capacitance slow the rising edge; lower the resistor or add active pull-up.
7. **Not waiting the full conversion time.** Reading the scratchpad before "Convert T" finishes returns the previous (or default) value.

## See Also

- [I2C](i2c.md) — the two-wire sensor bus to compare against
- [GPIO](gpio.md) — open-drain bit-banging
- [Sensors & Sensor Fusion](sensors.md) — what 1-Wire devices feed into
