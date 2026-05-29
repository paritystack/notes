# Bluetooth Low Energy (BLE)

## Overview

BLE is a low-power short-range wireless protocol designed for sending small amounts of data (sensor readings, button events, control commands) from coin-cell-powered devices. It is a **completely different protocol** from Classic Bluetooth — they share a brand and a 2.4 GHz radio but nothing else.

```
                       ┌─────────────────────┐
                       │   Application       │ your code
                       ├─────────────────────┤
                       │ Profiles (GATT)     │ services + characteristics
                       ├─────────────────────┤
                       │ GAP (connection)    │ scan, advertise, connect
                       ├─────────────────────┤
                       │     ATT             │ attribute protocol
                       ├─────────────────────┤
                       │  L2CAP              │ logical channels
                       ├─────────────────────┤
                       │  Link Layer         │ packets, encryption, scheduling
                       ├─────────────────────┤
                       │  PHY (1M/2M/Coded)  │ 2.4 GHz radio
                       └─────────────────────┘
```

The stack is heavy compared to a UART driver, so on MCUs you almost always use a **vendor-supplied stack** rather than rolling your own.

## Common BLE Stacks on MCUs

| Stack | Where | Notes |
|-------|-------|-------|
| **Nordic SoftDevice** | nRF51/52 | Closed-source binary blob, runs alongside your app, very stable. The reference BLE stack. |
| **Nordic + Zephyr** | nRF52/53/54 | Open-source, host + controller in Zephyr. Modern default for Nordic. |
| **Apache NimBLE** | Zephyr, RIOT, Mynewt, ESP32 | Open-source host + optional controller. Used by ESP-IDF for BLE on ESP32. |
| **ESP-IDF Bluedroid** | ESP32 (older) | Heavy, full-featured, legacy. NimBLE replaced it for most projects. |
| **STM32 BlueNRG / WB** | STM32WB / external chips | Vendor stack with host on M0 and app on M4. |
| **Cypress/Infineon PSoC 6** | PSoC | Vendor-supplied dual-core stack. |

For a new MCU project the choice is usually "whatever Zephyr or your vendor's SDK provides" — the wire-level protocol is the same; the API isn't.

## The Three Layers Worth Memorizing

### PHY

- **1M PHY**: classic BLE, ~1 Mbps over-the-air, ~270 kbps usable.
- **2M PHY** (BLE 5+): doubled throughput, slightly shorter range, lower energy per byte.
- **Coded PHY** (BLE 5+): long range, ~4× range vs 1M at 125 kbps or ~2× at 500 kbps. Used for asset tracking.

You usually let the stack pick. Long-range applications (tracking tags, asset tags) explicitly request Coded PHY.

### Link Layer

Handles:
- 40 channels at 2 MHz spacing: 37 data + 3 advertising (37, 38, 39).
- **Frequency hopping** between connection events (avoids Wi-Fi interference).
- Encryption (AES-CCM).
- Connection state machine: standby → advertising → scanning → initiating → connected.

You almost never touch the link layer directly. The stack exposes it via GAP.

### GAP (Generic Access Profile)

The "who talks to whom" layer. Defines roles:

| Role | What | Example |
|------|------|---------|
| **Broadcaster** | Advertises only; no connections | Beacon |
| **Observer** | Scans only; no connections | Beacon listener |
| **Peripheral** | Advertises and accepts connections | Sensor, fitness band |
| **Central** | Scans and initiates connections | Phone, gateway |

A device can be both Central and Peripheral simultaneously (most phones are).

## Advertising

A peripheral broadcasts on channels 37/38/39 every `advInterval` milliseconds. Adv packet is up to 31 bytes payload (BLE 4.x) or 255 bytes (BLE 5 extended advertising).

```
┌──────────┬────────────────────────────────────────┐
│  Header  │  AdvData (length, type, value triples) │
└──────────┴────────────────────────────────────────┘
```

Standard AD types include:

| Type | Meaning |
|------|---------|
| 0x01 | Flags (LE General Discoverable, BR/EDR Not Supported, etc.) |
| 0x02/0x03 | Incomplete/Complete list of 16-bit Service UUIDs |
| 0x06/0x07 | Incomplete/Complete list of 128-bit UUIDs |
| 0x09 | Complete Local Name |
| 0xFF | Manufacturer Specific Data (vendor ID + payload) |
| 0x16 | Service Data |

Typical advertising tradeoff:

| Interval | Discovery latency | Power |
|----------|-------------------|-------|
| 20 ms | <100 ms | High (mA) |
| 100 ms | ~300 ms | Medium |
| 500 ms | ~1.5 s | Low |
| 1 s | ~3 s | Very low (~10s of µA) |

iBeacon, Eddystone, and most "beacon" devices broadcast every 100-1000 ms.

### Connectable vs Non-connectable

- **ADV_IND**: connectable + scannable.
- **ADV_NONCONN_IND**: just broadcast, no connection possible. Cheaper. Beacons use this.
- **ADV_SCAN_IND**: not connectable but answers scan requests (lets a scanner get more payload).
- **ADV_DIRECT_IND**: connectable to a specific peer only.

## Connections

When a central connects, both sides agree on a **connection interval** (7.5 ms – 4 s). Every `connInterval`, both wake up, exchange any data, ack, sleep.

| Conn interval | Latency | Power |
|---------------|---------|-------|
| 7.5 ms | very low | high |
| 50 ms | low | medium |
| 500 ms | bad UX for events | low |

**Slave latency** lets the peripheral skip up to N consecutive connection events if it has nothing to send → save power without giving up responsiveness when the master sends.

`supervisionTimeout` ends the connection if no packets exchanged for that long (typically 2-6 seconds).

## GATT: Services and Characteristics

GATT (Generic Attribute Profile) is the **data model**. Everything is an attribute, identified by a 16-bit handle, with a UUID, value, and permissions.

```
Service: Heart Rate (UUID 0x180D)
  └── Characteristic: Heart Rate Measurement (UUID 0x2A37)
       ├── Properties: Notify
       └── Descriptor: CCCD (Client Characteristic Configuration)
                       — client writes 0x0001 to enable notifications
  └── Characteristic: Body Sensor Location (UUID 0x2A38)
       └── Properties: Read
```

Standard services (16-bit UUIDs) include Heart Rate, Battery Service, Device Information, Cycling Speed and Cadence, etc. Custom services use 128-bit UUIDs (vendor-generated).

### Characteristic Properties

| Flag | Meaning |
|------|---------|
| Read | Client can read value |
| Write | Client writes, server acks |
| Write Without Response | Client writes, no ack (faster) |
| Notify | Server pushes value; no client ack |
| Indicate | Server pushes value; client must ack |

The **CCCD** (descriptor 0x2902) controls whether the client wants notifications/indications enabled. Without writing 0x0001 to CCCD, your "notify" characteristic does nothing on the air.

### Server-Side Code (NimBLE-style)

```c
static uint8_t hr_value[2] = { 0, 60 };  // flags + BPM
static uint16_t hr_handle;

static int gatt_svr_chr_access(uint16_t conn_handle, uint16_t attr_handle,
                                struct ble_gatt_access_ctxt* ctxt, void* arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        os_mbuf_append(ctxt->om, hr_value, sizeof(hr_value));
        return 0;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

static const struct ble_gatt_svc_def gatt_svcs[] = {
    { .type = BLE_GATT_SVC_TYPE_PRIMARY,
      .uuid = BLE_UUID16_DECLARE(0x180D),    // Heart Rate
      .characteristics = (struct ble_gatt_chr_def[]) {
        { .uuid = BLE_UUID16_DECLARE(0x2A37),
          .access_cb = gatt_svr_chr_access,
          .val_handle = &hr_handle,
          .flags = BLE_GATT_CHR_F_NOTIFY }, { 0 }
    } }, { 0 }
};

// To send a notification later:
struct os_mbuf* om = ble_hs_mbuf_from_flat(hr_value, sizeof(hr_value));
ble_gatts_notify_custom(conn_handle, hr_handle, om);
```

### Client-Side: Discovery

A central connects, then walks GATT:

1. `Discover All Primary Services` → list of service UUIDs and ranges.
2. For each service of interest, `Discover All Characteristics`.
3. For each notify characteristic, write `0x0001` to its CCCD to subscribe.

This takes seconds on the first connect. Caching service handles per peer is a common optimization (BLE 5 has GATT Caching for this).

## Security: Pairing & Bonding

**Pairing** establishes encryption keys for the current session. **Bonding** = pairing + saving the keys for future sessions.

### Pairing Modes

| Mode | How |
|------|-----|
| **Just Works** | No user action. Vulnerable to MITM. Used for keyboard-less devices. |
| **Passkey Entry** | 6-digit code displayed on one side, entered on the other. |
| **Numeric Comparison** (LE Secure Connections) | Both display the same 6 digits; user confirms match. |
| **Out-of-Band (OOB)** | Pair via NFC tap, QR scan, etc. — strongest. |

**LE Secure Connections** (BLE 4.2+) uses ECDH and is MITM-resistant when not in Just Works mode. Old "LE Legacy Pairing" is broken; new code should use SC only.

### Encryption

Once paired, all link-layer packets are AES-CCM encrypted. The pairing exchange derives an LTK (Long Term Key) bonded devices reuse.

### Privacy

Devices can use **Resolvable Private Addresses (RPA)** that rotate periodically. Only bonded peers (who have the IRK) can identify the device across address changes. Critical for wearables that shouldn't be trackable.

## Throughput in Practice

Real-world BLE throughput (peripheral → central):

| Config | Achievable |
|--------|------------|
| BLE 4.0/4.1, 1M PHY, default MTU 23 | ~5-10 kB/s |
| BLE 4.2+, 1M PHY, MTU 247, DLE | ~50-80 kB/s |
| BLE 5, 2M PHY, MTU 247, DLE | ~150-200 kB/s |
| BLE 5, 2M PHY, "high throughput" tuned | ~750 kB/s |

Negotiate **MTU exchange** and **Data Length Extension (DLE)** early on connect to unlock big-packet transfers. Without them, every byte goes in 20-byte chunks.

## Power Budget

Typical nRF52 with coin cell:

| Activity | Current | Duration |
|----------|---------|----------|
| Sleep (system off, retained RAM) | ~1 µA | ~most of the time |
| Sleep (system on, RTC running) | ~3 µA | between events |
| Radio TX/RX | ~5-15 mA | <1 ms per event |

Connection-event power = `I_active × t_active + I_sleep × (interval - t_active)`. At 500 ms interval and 1 ms active, you're spending 0.2% of time on the radio.

Battery life for typical sensor: **months to years** on a CR2032.

## Beacons

Special non-connectable advertising with a known payload format.

- **iBeacon (Apple)**: 16-byte UUID + 2-byte major + 2-byte minor + 1-byte tx_power. In Manufacturer Specific Data with Apple's 0x004C company ID.
- **Eddystone (Google)**: Service Data with UID, URL, or TLM (telemetry) frames.
- **AltBeacon**: open alternative.

A beacon stays at standby + advertise loop forever. With 1-second intervals and a CR2032 battery, expect ~1 year of life.

## Mesh, LE Audio, Direction Finding (BLE 5+)

- **BLE Mesh**: many-to-many over BLE; nodes relay packets. Used in lighting (Philips Hue Bluetooth, Wiz). Different architecture from GATT.
- **LE Audio / LC3**: replaces Classic Bluetooth for headphones at lower power.
- **Direction Finding (BLE 5.1)**: Angle-of-Arrival/Departure for indoor positioning. Requires antenna arrays.

These are domain-specific; learn them when you need them.

## Practical Stack Comparisons

### Nordic + SoftDevice (Classic)

```c
// nRF5 SDK style
ble_advertising_init(...);
ble_gap_adv_start(...);
ble_advertising_start(...);

// Connect events come through a single event handler dispatcher
```

The SoftDevice is a binary blob at the bottom of flash; your app at the top. They talk via SVC (supervisor call). Stable, well-supported, but locked to Nordic.

### Zephyr + Nordic / NimBLE

```c
const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR),
    BT_DATA(BT_DATA_NAME_COMPLETE, "myname", 6),
};

err = bt_enable(NULL);
bt_le_adv_start(BT_LE_ADV_CONN_NAME, ad, ARRAY_SIZE(ad), NULL, 0);
```

Vendor-portable, open-source, integrates with Zephyr device tree and Kconfig.

### ESP32 + NimBLE (ESP-IDF)

```c
nimble_port_init();
ble_hs_cfg.reset_cb = on_reset;
ble_hs_cfg.sync_cb = on_sync;
ble_gatts_count_cfg(gatt_svcs);
ble_gatts_add_svcs(gatt_svcs);
nimble_port_freertos_init(host_task);
```

Same NimBLE API as Zephyr (mostly). Sufficient for most BLE use cases on ESP32.

## Common Pitfalls

### Pitfall 1: Forgetting to Enable CCCD

Server sends `ble_gatts_notify_custom`, returns success, but nothing arrives on the client side. Cause: client never wrote `0x0001` to the CCCD descriptor. Always check `ble_gatts_notify_custom`'s preconditions — the API requires an active subscription.

### Pitfall 2: MTU of 23 With Big Payloads

Default ATT MTU is 23 bytes (20 bytes of user data). Sending 100 bytes silently fragments — or fails. Negotiate higher MTU on connect.

### Pitfall 3: Connection Interval Too Low for Battery

7.5 ms gives buttery-smooth UX but kills coin-cell battery in days. Tune for the use case (advertising-only beacons are coin-cell-friendly; constant 7.5 ms connections are not).

### Pitfall 4: Advertising Payload Too Big

31-byte limit (BLE 4.x). Service UUID + name + manufacturer data overflows quickly. Either move data to scan response (another 31 bytes if peer sends scan req), use BLE 5 extended advertising, or trim.

### Pitfall 5: Just Works Pairing in Production

Easy to ship, vulnerable to MITM. If your device exchanges anything sensitive, use Numeric Comparison or OOB.

### Pitfall 6: Treating BLE Like a Socket

BLE characteristic ops are not byte streams. They're discrete notifications, may be reordered (rare but legal in some configs), and dropping happens. Layer your own framing/seq numbers if you need stream semantics.

### Pitfall 7: Long-Running Work in BLE Event Handlers

Most stacks deliver events from a single context (a task or a callback). Blocking inside an event handler stalls the entire stack. Defer work to your own thread.

### Pitfall 8: Bonding Records Filling NVS

After many test pair/unpair cycles, NVS fills up. Stack runs out of bond storage. Periodically erase old bonds or grow the bonding storage region.

## Quick Spec Cheat Sheet

| Item | Range / value |
|------|---------------|
| Advertising channels | 37, 38, 39 (2402, 2426, 2480 MHz) |
| Data channels | 0–36 (avoiding Wi-Fi) |
| Adv interval | 20 ms – 10.24 s |
| Conn interval | 7.5 ms – 4 s |
| Slave latency | 0–499 events |
| Supervision timeout | 100 ms – 32 s |
| ATT MTU (default) | 23 bytes (20 user) |
| ATT MTU (max) | 247 bytes (BLE 4.2+) |
| Adv payload (legacy) | 31 bytes |
| Adv payload (extended, BLE 5) | up to 255 bytes |
| PHYs | 1M, 2M, Coded (125k/500k) |

## Summary

1. **BLE is not Classic Bluetooth** — different stack, different use cases.
2. **GAP = who's talking; GATT = what they're saying.**
3. **Service → Characteristic → Descriptor (CCCD)** is the data model.
4. **Notifications need the CCCD enabled** by the client.
5. **Negotiate MTU + DLE** on connect for any throughput beyond toy.
6. **Pairing modes**: Just Works (insecure), Passkey, Numeric Comparison, OOB.
7. **Power: long advertising interval + idle sleep** → years on a coin cell.
8. **Stack choice: vendor SDK + NimBLE/Zephyr** for new code.
9. **Beacons = non-connectable ADV; iBeacon/Eddystone are payload formats.**
10. **BLE 5: 2M PHY, Coded PHY (long range), extended advertising, BLE Mesh, LE Audio, AoA/AoD.**

## See Also

- [WiFi](../wifi/README.md) — coexists with BLE on 2.4 GHz
- [Power Management](power_management.md) — sleep between connection events
- [Security](../security/README.md) — TLS analog: pairing + bonding
- [IoT Protocols](../networking/iot_protocols.md) — application-layer protocols often layered over BLE
