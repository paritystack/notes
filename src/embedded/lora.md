# LoRa and LoRaWAN

## Overview

LoRa ("Long Range") is a sub-GHz radio modulation by Semtech. LoRaWAN is the network protocol layered on top, providing a star-of-stars architecture: end-devices ↔ gateways ↔ network server.

```
        End devices                Gateways              Network Server
        (battery, sensors)          (always-on, IP-backhauled)
        
        ┌──────────┐                                       
        │ Sensor 1 ├──── 868 MHz ────►┌────────┐           
        ├──────────┤                  │Gateway │           ┌─────────────┐
        │ Sensor 2 ├──── 915 MHz ────►│  A     ├── IP ────►│  Network    │
        ├──────────┤                  └────────┘            │  Server     │
        │ Sensor 3 ├──────────────────►┌────────┐           │  (TTN, AWS, │
        └──────────┘                   │Gateway │           │   ChirpStack)│
                                       │  B     ├── IP ────►│             │
                                       └────────┘            └─────────────┘
                                                                   │
                                                                   ▼
                                                              Application
```

LoRa = the **PHY** (modulation). LoRaWAN = the **MAC + network**. You can use LoRa without LoRaWAN for point-to-point links.

## Why LoRa

The combination people pay for:

- **Long range** — 2-15 km in suburban, up to 40+ km line-of-sight.
- **Low power** — sub-µA sleep, ~30-100 mA TX. Years on a small battery.
- **Sub-GHz** ISM bands — fewer reflections, better penetration than 2.4 GHz.
- **License-free**. EU 868 MHz, US 915 MHz, AS 433/470/923 MHz.
- **Low data rate** — kbps at best. Good for telemetry, not for video.

The tradeoff: **bandwidth is tiny**. A single LoRa packet (51-242 bytes) takes hundreds of ms to milliseconds in the air. You can only send a few packets per device per hour due to duty-cycle limits.

## Chirp Spread Spectrum (CSS)

LoRa modulates by sweeping ("chirping") across the channel bandwidth. The chirp's direction and rate encodes data. Key parameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Spreading Factor (SF)** | 7–12 | Higher SF = slower, longer range, more energy per bit. Each step ~6 dB more range and ~2× airtime. |
| **Bandwidth (BW)** | 125 / 250 / 500 kHz | Wider BW = faster, less sensitive. EU normally 125 kHz. |
| **Coding Rate (CR)** | 4/5, 4/6, 4/7, 4/8 | Forward error correction overhead. Higher = more resilient. |
| **TX Power** | 2–22 dBm | Higher = more range and current draw. Regional caps apply. |

### Data Rate Cheat Sheet

```
Data rate ≈ SF × (BW / 2^SF) × CR_eff
```

| SF | BW | DR (bps) | Sensitivity | Airtime (51 B) |
|----|------|-----------|-------------|-----------------|
| 7  | 125 kHz | ~5,470 | -123 dBm | ~100 ms |
| 8  | 125 kHz | ~3,125 | -126 dBm | ~180 ms |
| 9  | 125 kHz | ~1,760 | -129 dBm | ~330 ms |
| 10 | 125 kHz | ~980   | -132 dBm | ~620 ms |
| 11 | 125 kHz | ~440   | -135 dBm | ~1.2 s |
| 12 | 125 kHz | ~250   | -137 dBm | ~2.5 s |

SF12 reaches further but each packet hogs the channel for seconds, eating your duty-cycle budget fast.

## Duty Cycle and Fair Use

Regulators cap how much time devices can transmit on ISM bands.

| Region | Rule |
|--------|------|
| EU 868 MHz | 1% duty cycle on most sub-bands (36 s/hour); 10% on a few |
| US 915 MHz | 400 ms max dwell; FHSS required (LoRaWAN handles this) |
| AS 923 MHz | LBT (Listen Before Talk) |

In practice: a device at SF12 sending 2-second packets at 1% duty cycle can transmit ~18 messages per hour. At SF7, hundreds. **Always estimate airtime per message** before claiming "we'll send updates every minute".

## LoRa Chips

| Chip | Notes |
|------|-------|
| **SX1276 / SX1278 / SX1279** | First-gen, very common, 137 MHz – 1020 MHz, ~15 km range |
| **SX1262 / SX1268** | Second-gen, lower current, ~22 dBm, used in modern modules |
| **SX1280** | 2.4 GHz LoRa — same modulation but on 2.4 GHz band (lower range, more bandwidth) |
| **STM32WL** | STM32 + integrated SX126x radio in one package |
| **LR1110 / LR1120** | Semtech's newer chip with GNSS + Wi-Fi sniffing for geolocation |

Most modules (RFM95, RAK4200, Heltec WiFi LoRa) wrap one of these. SPI interface to the MCU, plus DIO pins for IRQs.

### Interface (SX1276 example)

```c
// Set frequency to 868.1 MHz
uint32_t freq = (uint32_t)(868100000.0 / 32000000.0 * (1<<19));
spi_write(REG_FRF_MSB, (freq >> 16) & 0xFF);
spi_write(REG_FRF_MID, (freq >>  8) & 0xFF);
spi_write(REG_FRF_LSB, (freq)       & 0xFF);

// Configure SF=7, BW=125 kHz, CR=4/5
spi_write(REG_MODEM_CONFIG_1, (0x07 << 4) | (0x01 << 1));  // BW + CR
spi_write(REG_MODEM_CONFIG_2, (7 << 4) | 0x04);            // SF + CRC

// TX: write payload to FIFO and switch mode
spi_write(REG_FIFO_TX_BASE_ADDR, 0);
spi_write(REG_FIFO_ADDR_PTR,    0);
spi_write_buf(REG_FIFO, payload, len);
spi_write(REG_PAYLOAD_LENGTH, len);
spi_write(REG_OP_MODE, MODE_LORA | MODE_TX);

// Wait for DIO0 (TX_DONE)
while (!dio0_high()) {}
spi_write(REG_IRQ_FLAGS, IRQ_TX_DONE_MASK);  // W1C
```

You usually use a vendor or community driver (Semtech LoRaMac-node, LMIC, MCCI LoRaWAN-LMIC) rather than poking registers directly.

## LoRaWAN Network Stack

A device joins a network either by:

- **OTAA (Over-the-Air Activation)** — DevEUI + JoinEUI + AppKey baked in firmware; device sends Join Request, server replies with session keys. Recommended.
- **ABP (Activation by Personalization)** — DevAddr + NwkSKey + AppSKey baked in firmware; no join exchange. Simpler, weaker security, no key rotation.

After join, packets carry application payload encrypted with `AppSKey`, MIC computed with `NwkSKey`.

### Device Classes

| Class | Behavior | Latency | Power |
|-------|----------|---------|-------|
| **A** | TX, then two short RX windows (RX1, RX2) 1-2 s later. Pure pull. | Up to next uplink | Lowest |
| **B** | Class A + scheduled RX windows triggered by gateway beacons | Seconds | Medium |
| **C** | Continuous RX except during TX | Near-real-time | High (mains-powered) |

Most battery sensors are Class A. Class C is for actuators or always-on hubs.

### Frame Structure (Application Layer)

```
┌────────┬──────────┬──────────────────────────┬──────┐
│ MHDR   │ FHDR     │ FPort + FRMPayload       │ MIC  │
└────────┴──────────┴──────────────────────────┴──────┘
   1 B     7-22 B     up to ~222 B               4 B
```

- **MHDR**: message type (join req/accept, confirmed/unconfirmed data, etc.)
- **FHDR**: DevAddr, FCnt, FCtrl, MAC commands.
- **FRMPayload**: AES-128 encrypted application data.
- **MIC**: CMAC over the whole frame, NwkSKey.

### ADR (Adaptive Data Rate)

The network server can tell a device to use a lower SF and TX power when signal is strong. Saves battery and frees airtime. Disable for moving devices (network can't adapt fast enough to changing RSSI).

## Application Layer Decoding

LoRaWAN payloads are encrypted; the network server decrypts and forwards to your application as bytes. You typically:

1. Pack telemetry compactly (binary, not JSON — bytes are precious).
2. Send the binary blob.
3. Decode in a server-side "payload decoder" (TTN/ChirpStack support JS decoders).

```c
// Pack a temperature + humidity reading
typedef struct __attribute__((packed)) {
    int16_t  temp_c10;    // tenths of °C
    uint16_t humidity_p1000; // 0-1000 = 0-100.0%
    uint8_t  battery_v50;  // 50 = 0 V, 250 = 4 V (×0.02)
} sensor_payload_t;

sensor_payload_t pkt = {
    .temp_c10 = (int16_t)(t * 10),
    .humidity_p1000 = (uint16_t)(rh * 10),
    .battery_v50 = (uint8_t)(vbat * 50),
};
lorawan_send(LORAWAN_PORT_TELEMETRY, &pkt, sizeof(pkt));
```

5 bytes carries the same information as a 60-byte JSON object. That's 12× more daily messages on the same duty cycle.

## Point-to-Point LoRa (No LoRaWAN)

For private links between two LoRa devices, skip LoRaWAN entirely and just use the radio's raw LoRa packet mode.

```
Device A (TX) ──────► Device B (RX)

   No gateway. No network server.
   No duty cycle enforcement (still respect regulations).
   No standard encryption — roll your own.
```

Useful for: rural sensors talking to a single base station, point-to-point control links, custom mesh experiments (Meshtastic, Reticulum). Be aware you're responsible for collision avoidance, encryption, and acknowledgment.

## Common Libraries / Stacks

| Library | Platform | Notes |
|---------|----------|-------|
| **LoRaMac-node** (Semtech) | Bare-metal C | Reference implementation, all classes, supports SX127x/SX126x |
| **LMIC** (IBM/MCCI fork) | Arduino / bare-metal | Class A only on SX127x, widely used |
| **TinyLoRa** | Arduino | Class A, ABP only, tiny footprint |
| **Zephyr LoRaWAN** | Zephyr | Class A/B/C, integrates LoRaMac-node |
| **RAK / Adafruit AT modem libs** | AT-command modules | If you have an AT-command-driven LoRa module |

For new projects on a bare-metal C MCU: LoRaMac-node + your SX126x driver. For Zephyr-based: built-in `subsys/net/lib/lorawan`.

## Reception and Backhaul

Gateways have **8-channel concentrators** (SX1301/SX1302) — they listen on 8 frequencies × 6 SFs simultaneously, decoding any LoRa packet in range. Then they push it over IP (LoRaWAN packet forwarder, UDP) to a network server.

You don't need to deploy gateways yourself in many cities — **The Things Network (TTN)** is a free community network. Or pay for Helium, Senet, Loriot, ChirpStack-as-a-service.

## Geolocation

Three techniques:

- **GNSS chip on device** + LoRa for backhaul → standard approach, draws power.
- **TDoA (Time Difference of Arrival)** at gateways → ~50-100 m accuracy outdoors if 3+ gateways in range. No device hardware needed beyond LoRa.
- **LR1110 / LR1120** chips that snapshot GNSS satellites and Wi-Fi MAC addresses, then ship the snapshot via LoRa for cloud-side geolocation → very low power.

The TDoA + LR1110 combination is what makes asset-tracking tags possible at coin-cell scales.

## Power Numbers

Typical battery-powered Class A sensor sending every 15 min at SF10:

| State | Current | Time |
|-------|---------|------|
| Sleep (RTC) | 2 µA | 99.9% |
| TX (~620 ms at 14 dBm) | ~45 mA | 0.07% |
| RX1+RX2 (~200 ms) | ~15 mA | 0.02% |
| Sensor read (10 ms) | 5 mA | trivial |

Average ~5-10 µA → years on a 2200 mAh AA. Push to SF7 + 1-hour intervals → months from a coin cell.

## Common Pitfalls

### Pitfall 1: Sending Too Often

"We'll send every 30 s at SF12" — that's 100% duty cycle, illegal in EU. Compute airtime × messages/hour and check it's <36 s/hour at 868 MHz.

### Pitfall 2: ADR Disabled on Stationary Device

A wall-mounted sensor without ADR sits at SF12 forever even when it's 10 m from a gateway. Enable ADR for stationary nodes.

### Pitfall 3: ABP With No Frame Counter Persistence

ABP frame counters must persist across reboots, or the network server treats packets as replays. Save FCnt to flash before sleep.

### Pitfall 4: Wrong Region Config

EU868 firmware in US915 = silence. Each region has different channel plans, default DRs, duty cycles. Configure correctly per locale.

### Pitfall 5: TX Without Antenna

Even a 0 dBm TX into nothing can damage the PA over time. Always have an antenna or 50 Ω load before transmitting.

### Pitfall 6: Forgetting Down-Link Window Timing

Class A spec: RX1 opens exactly `RECEIVE_DELAY1` (default 1 s) after TX end. If your MCU is busy doing 1.5 s of sensor processing, you miss the downlink. Schedule TX as last action, sleep until RX1.

### Pitfall 7: Confusing LoRa and LoRaWAN

"Our protocol uses LoRa" might mean raw LoRa or LoRaWAN. They're not interchangeable. LoRaWAN devices won't decode raw LoRa packets and vice versa.

### Pitfall 8: Bandwidth vs Latency Confusion

Customers expect "real-time updates" over LoRaWAN. With Class A you can only push from the device; downlink waits for the next uplink. Set expectations.

## Summary

1. **LoRa = chirp-spread-spectrum PHY; LoRaWAN = MAC + network on top.**
2. **SF7-12 tradeoff: range vs airtime.** SF7 fast & short, SF12 slow & far.
3. **Duty cycle**: 1% on EU 868 MHz — budget airtime carefully.
4. **OTAA preferred over ABP** for security and FCnt rollover.
5. **Class A is the default.** B for synchronized RX, C for always-on.
6. **Pack binary payloads tightly.** JSON wastes 90%+ of duty cycle.
7. **ADR on for stationary devices**; off for movers.
8. **Class A latency = next uplink.** Not real-time.
9. **TTN, ChirpStack, Helium, Senet** for network infrastructure.
10. **Years on coin cells achievable** at low duty cycles and reasonable SF.

## See Also

- [Power Management](power_management.md) — sleep between LoRa transmissions
- [SPI](spi.md) — typical LoRa chip interface
- [BLE](ble.md) — short-range alternative
- [IoT Protocols](../networking/iot_protocols.md) — MQTT often layered above LoRaWAN at the application

## Where this connects

- [BLE](ble.md) — the short-range low-power sibling protocol
- [IEEE 802.15.4](ieee_802154.md) — Zigbee/Thread, another low-power wireless option
- [Power management](power_management.md) — duty-cycling for multi-year battery life
- [Sensors](sensors.md) — typical LoRaWAN end-device payloads
- [Signal integrity](signal_integrity.md) — sub-GHz antenna and RF layout
- [Modbus](modbus.md) — industrial field data often bridged onto LoRaWAN
