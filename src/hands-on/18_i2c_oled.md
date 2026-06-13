# 18 · Drive an I²C OLED display

## Overview

Your sensor reading currently only exists in the [serial monitor](14_adc_pot_serial.md) —
tethered to a PC. An OLED display puts it on the device itself, the first step toward an
untethered gadget. The SSD1306 OLED shares the very same [I²C](../embedded/i2c.md) bus as the
[BME280](17_i2c_sensor.md) — two devices, two wires — which demonstrates the whole point of a
bus. You'll show live temperature and humidity on a tiny screen.

```
   Shared I²C bus (sensor + display together):

   A4 (SDA) ──┬── BME280 SDA ──┬── OLED SDA
   A5 (SCL) ──┴── BME280 SCL ──┴── OLED SCL
              sensor @ 0x76        display @ 0x3C
```

## What you'll need

From **Stage B**: an SSD1306 0.96" I²C OLED, plus the [BME280](17_i2c_sensor.md) still wired
from the last rung. Both hang off A4/A5.

## The build

1. Wire the **OLED to the same SDA/SCL/VCC/GND** as the sensor (it's a bus — just join them).
2. Re-run the [scanner](17_i2c_sensor.md): you should now see **two** addresses (e.g. `0x76`
   and `0x3C`).
3. Install *Adafruit SSD1306* + *Adafruit GFX*, then:

```c
#include <Adafruit_SSD1306.h>
#include <Adafruit_BME280.h>
Adafruit_SSD1306 oled(128, 64, &Wire);
Adafruit_BME280 bme;

void setup() {
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  bme.begin(0x76);
}
void loop() {
  oled.clearDisplay();
  oled.setTextSize(2); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0,0);
  oled.print(bme.readTemperature(), 1); oled.println(" C");
  oled.print(bme.readHumidity(), 0);    oled.println(" %");
  oled.display();
  delay(1000);
}
```

The readings now appear on the screen. Unplug the USB-serial cable and power from a battery —
it still works. You've made a thing that displays the world.

## It works when…

- [ ] The scanner shows both the sensor and the display addresses.
- [ ] Live temperature/humidity render on the OLED and update.
- [ ] Both devices coexist on the same two wires.

## What's happening

Because [I²C](../embedded/i2c.md) is *addressed*, many devices share SDA/SCL and the master
talks to one at a time by its address — so adding the display cost zero extra pins. The OLED
controller (SSD1306) has its own RAM framebuffer; the [GFX](../embedded/display_graphics.md)
library renders text/shapes into a local buffer and `display()` ships it over I²C in one
burst. This is the first time your device produces output a human reads without a computer —
a real milestone toward the portable [capstone](README.md). Next you'll *watch* this bus
traffic with a [logic analyzer](19_logic_analyzer_i2c.md).

## Pitfalls

- **Address clash** — two devices can't share one address. Sensor (0x76/0x77) and OLED (0x3C/0x3D) differ, so they're fine; watch this when adding more.
- **Wrong screen size or address in `begin()`** — a 128×32 panel with 128×64 code shows garbage; some OLEDs are 0x3D. Scan and match.
- **Forgetting `display()`** — drawing only updates the buffer; nothing shows until you push it. A classic "blank screen" cause.
- **Bus too long / weak pull-ups** — adding devices and wire length loads the bus; if it gets flaky, lower the I²C clock or strengthen [pull-ups](06_button_pullup.md).

## Where this connects

- [I²C](../embedded/i2c.md) — multiple devices on one addressed bus
- [Display & Graphics](../embedded/display_graphics.md) — framebuffers, fonts, drawing
- [17 · I²C sensor](17_i2c_sensor.md) — the device that shares this bus
- **Previous:** [17 · I²C sensor](17_i2c_sensor.md) · **Next:** [19 · Logic analyzer](19_logic_analyzer_i2c.md)
