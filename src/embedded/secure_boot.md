# Secure Boot

## Overview

Secure boot is the chain of trust that ensures **only authorized firmware runs** on a device, from immutable hardware up to your application. If an attacker can swap in modified firmware, every other security mechanism collapses — they can disable signature checks, dump keys, intercept user data.

```
   Immutable ROM ──verifies──► Bootloader ──verifies──► App
   (with key/hash               (with key in flash         (now trusted to
    fused into silicon)          + signature in app)        verify FS, OTA, etc.)
   
   Root of Trust ──► Chain of Trust ──► Authenticated execution
```

The threat model: an attacker can extract the flash chip, modify it, reflash, and reboot. Without secure boot, they win. With secure boot, the chip refuses to execute anything not signed by you.

## Root of Trust

The base of the chain has to be **immutable** and **secret-keeping**, or the chain has no anchor. Options:

| Mechanism | Implementation |
|-----------|----------------|
| **Mask ROM** | Code burned into silicon at manufacture. Cannot be modified. |
| **eFuses / OTP** | One-time programmable memory cells in the chip. Burn keys/hashes at provisioning, can't be erased. |
| **PUF (Physical Unclonable Function)** | Per-chip secret derived from manufacturing variations. No key storage needed. |
| **Secure element** | Separate IC (ATECC608, OPTIGA, SE050) that stores keys and signs on request. |

Typical bare-metal MCU: **mask ROM verifies a first-stage bootloader against a hash burned into eFuses**.

## What Gets Verified

Two common image authentication schemes:

### Asymmetric Signature (RSA / ECDSA)

```
Build time:
   image → hash (SHA-256) → sign with private key → signature
   ship: image + signature

Boot time:
   image + signature → recompute hash, verify signature with public key
                       (public key in flash or eFuse)
```

- **Public key on device, private key offline.** Compromise of the device doesn't compromise signing.
- **Standard algorithms**: ECDSA P-256 with SHA-256 is the modern default. RSA-2048 / RSA-3072 still common in older fleets.
- **Cost**: ECDSA verify takes ~100-500 ms on Cortex-M without hardware acceleration; ms with crypto accelerator.

### Symmetric MAC (HMAC, CMAC)

```
Same key on device and at signing time:
   image → HMAC(key, image) → mac
   ship: image + mac
```

- **Faster verify** (~10× faster than ECDSA without HW accel).
- **Same key everywhere is dangerous**: if a device gets cloned, the secret leaks.
- **Use only with HW-protected keys** (TrustZone-isolated, in secure element).

For consumer/IoT, ECDSA is the default. HMAC appears when verify performance matters a lot.

## Image Format

A signed image is typically:

```
┌──────────────────┐
│  Header          │  magic, version, image size, type, flags
├──────────────────┤
│  Payload         │  vector table + .text + .rodata + .data
│                  │
├──────────────────┤
│  Signature       │  ECDSA(P-256, SHA-256, header+payload)
│  + KeyID         │  which signing key was used
└──────────────────┘
```

MCUboot's TLV (type-length-value) format is the de-facto standard now. Vendor-specific formats (STM32 STSAFE, Nordic MCUBoot variant, ESP32 secure boot v2) exist but are conceptually identical.

## Verification Flow

```
1. ROM reads boot vector
       │
       ▼
2. ROM hashes first-stage bootloader, compares against eFuse hash.
       │  (or: ROM verifies bootloader signature against public key in eFuse)
       ▼
3. Bootloader starts, reads app image header.
       │
       ▼
4. Bootloader hashes app, verifies signature with public key in its own flash.
       │
       ▼
5. If valid → bootloader sets VTOR, jumps to app.
   If invalid → fall back to known-good slot, or enter recovery / brick.
```

The first hop is the most critical — that's where the **immutable root of trust** is enforced. After that, each link verifies the next.

## Anti-Rollback (Version Counter)

Without rollback protection, an attacker can install a previous valid signed firmware that has a known vulnerability. Counter the attack by tracking **minimum image version** in OTP:

```
On successful boot of vN:
   if (image_version > otp_min_version) {
       eFuse_write(otp_min_version, image_version);
   }

On image verify:
   if (signature_valid && image_version >= otp_min_version) {
       boot();
   }
```

Each eFuse bit can only be written 1→0 (or 0→1, depending on encoding). Use a monotonic counter where the version is represented by N bits of "burn level" — N bits = N supported version bumps.

## Vendor Implementations

### STM32

- **Option bytes** include RDP (Read-Out Protection) levels 0/1/2.
  - **Level 0**: open.
  - **Level 1**: flash unreadable via debug; readable from running code. JTAG erase mass-erases flash on disable.
  - **Level 2**: permanently locked. No JTAG, no debug, ever. (Set in production only; not reversible.)
- **STM32 Trusted Firmware-M** (TF-M): full ARM PSA-compliant secure boot for STM32U5/L5/H5 (TrustZone-M parts).
- **Firewall** (older M3/M4 parts) and **SAU/IDAU** (M33 parts) isolate secure regions.

### ESP32

- **Secure Boot V2**: ECDSA / RSA-3072 signed second-stage bootloader and app. Keys burned to eFuses.
- **Flash Encryption**: AES-256 keys in eFuses; flash is encrypted at rest, decrypted on read by hardware.
- **Both should be enabled together** for production; secure boot alone leaves cleartext images.

### Nordic nRF52/53

- **BL_SETTINGS_PAGE + softdevice + app** with optional signing via MCUboot.
- **nRF5340 / nRF54** add ARM TrustZone-M; bootloader runs in Secure, app in Non-Secure.

### NXP / Renesas / Microchip

Each has analogous schemes (NXP's HABv4 / AHAB, Renesas SCE, Microchip CryptoAuthentication). The concepts above translate.

## Provisioning

The hardest part of secure boot is **getting the right keys into the right chips at the right time**, securely.

```
Factory floor:
   1. Test fixture connects to chip via JTAG.
   2. Test fixture writes:
      - Public key hash → eFuse
      - Per-device unique ID → eFuse
      - Signed factory firmware → flash
   3. Test fixture verifies boot succeeds.
   4. Lock JTAG via eFuse (sometimes irreversibly).
```

Each step is a chance to leak keys, install rogue firmware, or brick devices. Real factories use:

- **HSMs (Hardware Security Modules)** to sign per-device certificates without exposing the private key.
- **Provisioning tokens** so test stations can sign limited numbers of devices but can't sign arbitrary firmware.
- **Per-device unique IDs** so a leak of one device's data doesn't compromise others.

## Disabling Debug

Once secure boot is live, **leaving JTAG open lets attackers bypass everything**. Production must disable debug at the silicon level:

- STM32: RDP Level 1 (limited bypass via mass-erase) or Level 2 (permanent).
- ESP32: `DISABLE_DL_ENCRYPT`, `DISABLE_DL_DECRYPT`, `DISABLE_DL_CACHE` eFuses; `JTAG_DISABLE`.
- Nordic: APPROTECT register prevents debug access until next chip erase.

This is also why factory bring-up is painful — you need a way to recover from "we shipped 10k devices with the wrong firmware" while still locking devices in the field.

## Encryption at Rest vs Authentication

Two independent properties:

- **Authentication** (signed images): "Only firmware I authorized runs."
- **Encryption** (flash encrypted): "Firmware binary is not readable from the flash chip."

Encryption alone is **not** secure boot. An attacker can replace ciphertext with their own, and if it decrypts to something the CPU executes, you're owned. Always pair encryption with authentication.

Conversely, signed-but-unencrypted firmware is fine for many use cases (no IP to protect, just integrity to defend).

## Threats Outside the Software Stack

Secure boot defends against software/flash-level tampering. It does **not** defend against:

- **Physical decapsulation + reading silicon** with a focused ion beam.
- **Power glitching** (precisely-timed voltage drops during the signature check causing the CPU to skip the branch). Defenses: redundant checks, glitch detectors, random delays.
- **Side-channel attacks** (timing, power, EM) revealing keys during cryptographic operations.
- **Compromise of the signing key** (offline private key leak). The whole chain dies.

For high-assurance products: secure elements (SE050, ATECC608), tamper-detection enclosures, audited signing infrastructure.

## Practical Code

```c
// Verify-and-boot, conceptual
bool verify_and_boot(const image_t* img) {
    if (img->header.magic != IMAGE_MAGIC) return false;
    if (img->header.size > MAX_IMAGE_SIZE) return false;
    if (img->header.version < min_allowed_version()) return false;

    uint8_t hash[32];
    sha256(img->payload, img->header.size, hash);

    if (!ecdsa_verify(public_key, hash, sizeof(hash),
                      img->signature, sizeof(img->signature))) {
        return false;
    }

    update_anti_rollback(img->header.version);

    SCB->VTOR = (uint32_t)img->payload;
    __set_MSP(*(uint32_t*)img->payload);
    ((void(*)(void))(*(uint32_t*)(img->payload + 4)))();
    /* never returns */
}
```

In real code, swap `sha256` and `ecdsa_verify` for vendor crypto-accelerator calls (STM32 HASH peripheral, ESP32 mbedTLS-on-HW), and add glitch-attack defenses (double-check the verify result with a redundant computation).

## MCUboot

The open-source bootloader for Cortex-M. Implements signed images, A/B slots, rollback, encryption (image-level), and the entire image-format machinery. **Standard choice for new projects.** Used by Zephyr, Pebble, Particle, many Nordic products.

- Image format: TLV-based, with magic, hash, signature, encryption tags.
- Algorithms: ECDSA P-256, RSA-2048/3072, ED25519.
- Slots: primary + secondary, with confirmation flags for rollback.
- Build pipeline: `imgtool` signs your built `.bin` with your offline ECDSA key.

```bash
# Build signed image
imgtool sign --key signing-ecdsa.pem \
             --header-size 0x200 \
             --slot-size 0x40000 \
             --version 1.2.3 \
             --pad-header \
             firmware.bin firmware.signed.bin

# Flash signed image to slot 1 (over-the-air update)
ota_write(SLOT_1_BASE, firmware_signed_bin, len);
boot_set_pending(0);
NVIC_SystemReset();
```

## Common Pitfalls

### Pitfall 1: Signing the Wrong Region

Signature covers headers + payload, but you sign only payload. Attacker swaps the header (e.g., changes "image type" or "load address") and signature still verifies. Always include all fields that affect behavior.

### Pitfall 2: Forgetting Anti-Rollback

You ship firmware v1.0 with a vulnerability. Patch in v1.1. Without anti-rollback, attacker installs signed v1.0 again. Always burn the OTP counter on successful upgrade.

### Pitfall 3: Public Key in Mutable Flash

Public key stored in main flash that the app can rewrite → attacker who can write flash also swaps your key for theirs. Key must be in eFuses or an immutable region.

### Pitfall 4: Single Point of Verification

Only check signature in the bootloader, then never again. Trust path collapses on partial flash corruption. Periodically re-verify; verify before OTA writes; verify at suspicious events (rapid reboots).

### Pitfall 5: Leaving Debug Enabled in Production

Secure boot + open JTAG = no secure boot. Production firmware must finalize the disable.

### Pitfall 6: Insecure Provisioning

Factory leaks a master signing key once → adversary signs arbitrary firmware forever. Use HSMs, per-device signing certs, audit signing events.

### Pitfall 7: Forgetting Encryption Is Not Authentication

Encrypted but unsigned firmware is bypassable. The two go together.

### Pitfall 8: Glitch Susceptibility

A naive single-branch signature check `if (verify(...)) jump_to_app()` can be glitched. Defenses: double-check, use trampolines that require multiple matches, deliberately delay random amounts.

### Pitfall 9: Slow Verify Blocks Boot

ECDSA P-256 verify takes ~300 ms unaccelerated on M4. If you boot 1000 times to debug, that's painful. Use a hardware crypto accelerator (STM32H7, nRF52 CryptoCell, ESP32 Digital Signature) in production.

### Pitfall 10: Permanent Lock With No Recovery Path

Shipping RDP Level 2 with no escape hatch is a one-way door. If you ship buggy keys or buggy bootloader, the entire fleet is dead. Have a recovery story before locking.

## Summary

1. **Root of trust must be immutable** — mask ROM + eFuses.
2. **ECDSA P-256 + SHA-256** is the modern default for image signing.
3. **Public key on device, private key offline + in HSM.**
4. **Sign the whole image** including headers and any field that affects behavior.
5. **Anti-rollback via OTP counter** to block downgrade attacks.
6. **Encryption ≠ authentication** — use both for confidentiality + integrity.
7. **Disable debug on production** silicon (RDP / APPROTECT / fuses).
8. **MCUboot** is the open-source standard; learn its image format.
9. **Defend against glitching** with redundant checks.
10. **Factory provisioning** is half of secure-boot security — invest in HSMs.

## See Also

- [Bootloaders](bootloaders.md) — the bootloader that runs the verify
- [TrustZone-M](trustzone_m.md) — hardware isolation for the bootloader
- [OTA Updates](ota_updates.md) — delivering signed images
- [Security](../security/index.html) — TLS, certificates, key management
