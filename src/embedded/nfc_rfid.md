# NFC & RFID

## Overview

RFID (Radio-Frequency Identification) and its short-range, smartphone-friendly subset **NFC** (Near-Field Communication) let a powered **reader** talk to a tag — often an unpowered chip that harvests all its energy from the reader's magnetic field. From an embedded standpoint they're a wireless cousin of contact smart cards: no battery in the tag, a few hundred bytes to a few KB of storage, and a coupling range from millimetres to a few metres depending on frequency. They sit alongside [BLE](ble.md) and [LoRa](lora.md) in the short-range wireless toolbox but occupy a unique niche — *passive*, *tap-to-act*, *cheap-per-tag* — used for access cards, contactless payment, inventory, pairing handoff, and tamper-evident product tags. When NFC carries credentials or keys, it leans on a [secure element](secure_boot.md).

```
   Reader (powered, MCU-driven)        Tag (often passive)
   ┌──────────────┐                    ┌──────────────┐
   │ MCU ─SPI/I2C─│  ((  magnetic  ))  │  antenna     │
   │  NFC/RFID    │═══   coupling   ═══│  coil +      │
   │  controller  │   field powers &   │  chip        │
   │  + antenna   │   clocks the tag   │  (no battery)│
   └──────────────┘                    └──────────────┘
```

## Frequency Bands

"RFID" spans wildly different physics depending on frequency:

| Band | Freq | Range | Coupling | Typical use |
|------|------|-------|----------|-------------|
| **LF** | 125–134 kHz | ~10 cm | Inductive | Animal ID, immobilizers, legacy access |
| **HF / NFC** | 13.56 MHz | ~10 cm | Inductive (near-field) | Payment, transit, access, smartphones |
| **UHF** | 860–960 MHz | 1–10 m | Backscatter (far-field) | Inventory, supply chain, toll tags |

**NFC is specifically the 13.56 MHz HF band** with extra standards that make a phone and a tag interoperate. LF/HF use *inductive coupling* — the reader's coil and tag's coil act like a loosely-coupled transformer, which both powers the tag and carries data over the same field. UHF uses *backscatter* (the tag reflects/modulates the reader's far-field wave), giving much longer range but no smartphone support.

## NFC Standards Stack

NFC layers several ISO standards plus the NFC Forum specs:

```
   ┌────────────────────────────────────────┐
   │ Applications: payment (EMV), NDEF data, │
   │ pairing handover, access control        │
   ├────────────────────────────────────────┤
   │ NDEF  (NFC Data Exchange Format)        │ ← the data container
   │ Tag Types 1–5 (mapping to chips below)  │
   ├────────────────────────────────────────┤
   │ ISO 14443 A/B (proximity, cards)        │
   │ ISO 15693    (vicinity, ~1m)            │
   │ FeliCa (Sony, transit in Asia)          │
   ├────────────────────────────────────────┤
   │ 13.56 MHz analog front-end / RF         │
   └────────────────────────────────────────┘
```

| Standard | What | Examples |
|----------|------|----------|
| **ISO 14443 A/B** | Proximity cards (~10 cm) | MIFARE, bank cards, NTAG, passports |
| **ISO 15693** | Vicinity (~1 m), slower | Library tags, industrial |
| **FeliCa** | Sony's fast variant | Japanese transit (Suica) |
| **NDEF** | Standard message format on a tag | URL, text, Wi-Fi/BLE pairing record |

### NFC operating modes

- **Reader/Writer** — your MCU reads/writes NDEF on a passive tag (most embedded use).
- **Card Emulation** — your device *pretends to be a tag/card* (phone payment, emulated access badge); credentials usually live in a [secure element](secure_boot.md).
- **Peer-to-Peer** — two active devices exchange data (largely superseded by BLE handover).

## NDEF: The Data Format

NDEF is the standard container that makes "tap to open a URL" or "tap to pair" work across vendors. A message is a sequence of typed records:

```
NDEF Message
 ├─ Record: Type=URI,  Payload="https://example.com"
 ├─ Record: Type=Text, Payload="en: Hello"
 └─ Record: Type=MIME, Payload=<Wi-Fi or BLE pairing blob>
```

A common embedded pattern is the **BLE/Wi-Fi pairing handover**: the device exposes an NDEF record with its [BLE](ble.md) address and pairing info; tapping a phone reads it and connects — "tap to pair" without typing anything.

## Embedded Integration

The MCU rarely does the 13.56 MHz RF itself — a dedicated **NFC controller IC** (ST25, PN532/PN5180, NXP CLRC663) handles the analog front-end and protocol, and you talk to it over [I2C](i2c.md) or [SPI](spi.md):

```c
// Typical flow with an NFC reader IC over I2C/SPI
nfc_init();
if (nfc_poll_for_tag(&uid)) {          // RF field on, look for a tag
    if (nfc_select(&uid) == OK) {       // anticollision + select
        nfc_read_ndef(buf, sizeof buf); // read the NDEF message
        parse_ndef(buf);
    }
}
```

A neat trick: some tags are **dual-interface** ("dynamic NFC tags", e.g. ST25DV) — an [I2C](i2c.md) side for the MCU and an RF side for the phone, sharing the same EEPROM. The MCU writes data over I2C; a phone taps to read it over NFC, even with the board unpowered. Great for configuration, diagnostics, and provisioning.

### Antenna basics

NFC range lives or dies on the antenna: a flat coil (a few turns of PCB trace) tuned with a capacitor to resonate at 13.56 MHz, impedance-matched to the controller. Detuning from nearby metal or a poor match collapses range — antenna layout is the #1 NFC hardware issue, closely tied to [signal integrity](signal_integrity.md).

## Security Notes

- **Plain tags are clonable.** A bare UID or open NDEF tag can be copied trivially — never use a tag's UID alone as a secret (the classic broken access-control design).
- **Crypto tags** (MIFARE DESFire, NTAG with crypto) do challenge-response authentication; payment uses EMV with a [secure element](secure_boot.md).
- **Eavesdropping & relay attacks** are real at close range; secure protocols, not the short range, provide the actual protection.

## Where this connects

- [BLE](ble.md) — "tap to pair" hands a phone the BLE connection details via an NDEF record.
- [Secure Boot / Secure Element](secure_boot.md) — credentials and payment keys for card emulation live in a tamper-resistant secure element.
- [I2C](i2c.md) / [SPI](spi.md) — the MCU-to-NFC-controller interface; also the contact side of dual-interface tags.
- [Power Management](power_management.md) — passive tags are battery-free; readers can field-detect to wake the MCU only on tap.
- [Signal Integrity](signal_integrity.md) — antenna tuning/matching and metal detuning dominate NFC reliability.
- [LoRa](lora.md) — the long-range counterpart in the short-vs-long wireless spectrum.

## Pitfalls

1. **Trusting the UID as a secret.** UIDs are readable and clonable; access control on UID alone is trivially defeated. Use crypto authentication.
2. **Antenna detuning.** Metal near the coil (battery, shield, enclosure) shifts resonance and kills range. Keep clearance and tune the matching network.
3. **Confusing NFC with UHF RFID.** They're different physics (near-field inductive vs far-field backscatter) and ranges; a phone reads NFC, not UHF inventory tags.
4. **Expecting long range from NFC.** It's ~centimetres by design (a security feature). If you need metres, that's UHF or [BLE](ble.md).
5. **Skipping NDEF formatting.** Writing raw bytes a phone can't parse — phones expect well-formed NDEF to auto-act on a tap.
6. **Ignoring tag write-cycle/EEPROM limits.** Tags are EEPROM with finite write endurance; don't hammer them in a loop.
7. **Field/power timing on passive tags.** The tag only lives while in the field; pulling the field too early mid-transaction corrupts a write.

## See Also

- [BLE](ble.md) — pairing handover and the short-range wireless sibling
- [Secure Boot](secure_boot.md) — secure elements for credentials
- [I2C](i2c.md) / [SPI](spi.md) — NFC controller interface
- [Signal Integrity](signal_integrity.md) — antenna and matching
