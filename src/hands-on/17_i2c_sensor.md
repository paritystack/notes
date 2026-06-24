# 17 · I²C scan + read a temp/humidity sensor

## Overview

A pot was an analog input; most real [sensors](../electronics/sensors.md) are digital and
talk over a **bus**. [I²C](../embedded/i2c.md) is the workhorse: two wires (SDA + SCL) shared
by many devices, each at its own address. You'll scan the bus to discover your sensor's
address, then read real temperature and humidity from a **BME280** — the actual sensing
front-end of your capstone. This is also your first time using a vendor *library* and reading
a [datasheet](../electronics/datasheets.md) for register meaning.

```
   Two shared wires, pulled up:

   3V3 ─[4.7k]─┬─ SDA ── Nano A4 ── sensor SDA
   3V3 ─[4.7k]─┴─ SCL ── Nano A5 ── sensor SCL
   (most breakout boards include the pull-ups already)
```

## What you'll need

From **Stage B**: a BME280 breakout (temp/humidity/pressure, I²C), the Nano, jumpers. Most
breakouts run at 3.3 V logic but tolerate the Nano's 5 V I²C — check your board's
[datasheet](../electronics/datasheets.md).

## The build

1. Wire **SDA→A4, SCL→A5, VCC→3V3 (or 5V per board), GND→GND**.
2. **Scan first** — confirm the wiring and learn the address:

```c
#include <Wire.h>
void setup() {
  Wire.begin(); Serial.begin(9600);
  for (byte a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { Serial.print("Found 0x"); Serial.println(a, HEX); }
  }
}
void loop() {}
```

You should see `0x76` or `0x77` (the BME280). If nothing appears, it's wiring or pull-ups.

3. **Read it** with a library (install *Adafruit BME280* + *Adafruit Unified Sensor*):

```c
#include <Adafruit_BME280.h>
Adafruit_BME280 bme;
void setup() {
  Serial.begin(9600);
  if (!bme.begin(0x76)) Serial.println("not found");   // use the scanned address
}
void loop() {
  Serial.print(bme.readTemperature()); Serial.print(" C  ");
  Serial.print(bme.readHumidity());    Serial.println(" %");
  delay(1000);
}
```

Breathe on the sensor — humidity jumps, temperature drifts up. Real physical data in your
serial window.

## It works when…

- [ ] The scanner prints the sensor's address (0x76/0x77).
- [ ] Temperature and humidity print and respond to breathing on the sensor.
- [ ] You can find the I²C address in the sensor's [datasheet](../electronics/datasheets.md) and match it.

## What's happening

[I²C](../embedded/i2c.md) is a two-wire, addressed, master/slave bus: the Nano (master)
pulls SDA/SCL in a defined sequence to send an address + read/write to a register inside the
sensor. Both lines are **open-drain**, so they need [pull-up](06_button_pullup.md) resistors
(usually on the breakout) — devices can only pull low, never drive high. The library hides
the register-level handshake, but under it is exactly the protocol you'll *see on the wire*
in [rung 19](19_logic_analyzer_i2c.md). One bus can host the sensor *and* the
[OLED](18_i2c_oled.md) of the next rung, sharing the same two pins.

## Pitfalls

- **No pull-ups** — a bare sensor chip (not a breakout) needs external 4.7 kΩ pull-ups on SDA/SCL or the bus never works.
- **Wrong address** — `0x76` vs `0x77` depends on an address pin; scan to be sure, then pass the right one to `begin()`.
- **SDA/SCL swapped** — A4 = SDA, A5 = SCL on the Nano. Swapped = silent failure.
- **Logic-level mismatch** — a strict 3.3 V sensor on a 5 V bus may need a level shifter; many breakouts are tolerant — check the [datasheet](../electronics/datasheets.md).
- **A "BME280" that's really a BMP280** — cheap boards are often mislabelled. The BMP280 has *no humidity* sensor, so `readHumidity()` returns nonsense (often 0 or a constant). If humidity never moves when you breathe on it, check the marking — the BMP280's address is the same 0x76/0x77, so the scanner won't tell them apart.

## Where this connects

- [I²C](../embedded/i2c.md) — addressing, open-drain, clock stretching, the full protocol
- [Sensors & Transducers](../electronics/sensors.md) — sensor types and interfaces
- [Reading a Datasheet](../electronics/datasheets.md) — finding address, registers, and timing
- **Previous:** [16 · Reaction-timer game](16_reaction_game.md) · **Next:** [18 · I²C OLED](18_i2c_oled.md)
