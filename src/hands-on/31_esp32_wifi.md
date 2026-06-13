# 31 · ESP32 module — send data over WiFi (optional)

## Overview

The capstone is done, but a "sensor node" really wants to *report* somewhere. This optional
rung adds connectivity: an [ESP32](../embedded/esp32.md) module takes your
[sensor](17_i2c_sensor.md) readings and pushes them over WiFi to your phone or a dashboard.
Rather than design an RF PCB (hard — antennas, impedance, shielding), you use the ESP32 as a
*pre-certified module* — the pragmatic way to add wireless. It's also your on-ramp to the
[IoT](../embedded/ota_updates.md) side of embedded.

```
   Two ways to use the ESP32:

   A) ESP32 replaces the AVR entirely (it has WiFi + enough GPIO/I2C)
   B) AVR does the sensing, ESP32 is a WiFi "modem" over UART/I2C

   sensor ─ I2C ─► ESP32 ─ WiFi ─► MQTT broker / HTTP / phone app
```

## What you'll need

From **Stage E**: an [ESP32](../embedded/esp32.md) dev module, the [sensor](17_i2c_sensor.md),
the Arduino IDE with ESP32 board support (or ESP-IDF), and a WiFi network. Easiest path:
**let the ESP32 do everything** (option A) since it has [I²C](17_i2c_sensor.md) and far more
power than the AVR.

## The build

1. Install **ESP32 board support** in the Arduino IDE; select your module.
2. Wire the [BME280](17_i2c_sensor.md) to the ESP32's I²C pins (commonly GPIO21=SDA,
   GPIO22=SCL — check your board). Note the ESP32 is **3.3 V logic** — no 5 V on its pins.
3. Read the sensor (same [Adafruit library](17_i2c_sensor.md)), connect to WiFi, and publish:

```c
#include <WiFi.h>
#include <Adafruit_BME280.h>
Adafruit_BME280 bme;

void setup() {
  Serial.begin(115200);
  bme.begin(0x76);
  WiFi.begin("your-ssid", "your-pass");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println(WiFi.localIP());
}
void loop() {
  float t = bme.readTemperature();
  // POST to a server, publish to MQTT, or serve a tiny web page — your choice
  Serial.println(t);
  delay(5000);
}
```

4. Confirm the reading reaches its destination (a serial print on another machine, an HTTP
   endpoint, or an MQTT subscriber). Then add the ESP32's own [deep sleep](24_deep_sleep.md)
   between transmissions — radios are power-hungry, so duty-cycling matters even more here.

## It works when…

- [ ] The ESP32 connects to WiFi and reads the sensor.
- [ ] A reading arrives at your phone/PC/dashboard over the network.
- [ ] The node sleeps between transmissions to keep battery life reasonable.

## What's happening

The [ESP32](../embedded/esp32.md) bundles a microcontroller, WiFi/BLE radio, and a certified
antenna into a module, so you get wireless without RF design — the right abstraction at your
stage. Reusing the *exact* [I²C sensor code](17_i2c_sensor.md) shows how transferable the
skills are: only the transport changed. The big new cost is power — a WiFi radio can pull
100–250 mA while transmitting, dwarfing everything else, which is why
[deep-sleep duty-cycling](24_deep_sleep.md) (wake, read, transmit, sleep) is essential for a
battery node. From here the embedded book opens up: [BLE](../embedded/ble.md),
[OTA updates](../embedded/ota_updates.md), security, and cloud dashboards.

## Pitfalls

- **5 V on ESP32 pins** — it's a 3.3 V part; 5 V can damage GPIO. Level-shift if mixing with 5 V logic.
- **Underpowering the module** — WiFi current spikes brown out a weak [supply](21_ldo_regulator.md); give it a solid 3.3 V rail and bulk [capacitance](../electronics/capacitors.md).
- **Radio left on continuously** — destroys battery life; sleep between transmissions and batch readings.
- **Hardcoding credentials** — fine for a test, but plan for config (captive portal / provisioning) and keep secrets out of shared code.

## Where this connects

- [ESP32](../embedded/esp32.md) — the module, its peripherals, and toolchains
- [BLE](../embedded/ble.md) / [OTA Updates](../embedded/ota_updates.md) — where to go next with connectivity
- [24 · Deep-sleep](24_deep_sleep.md) — essential for a battery-powered radio node
- **Previous:** [30 · Bring-up (capstone)](30_bringup.md) · **You've finished the roadmap.** Back to [the index](README.md).
