# Verified Boot & OTA

## Overview

Two tightly related platform features keep Android devices **trustworthy** and
**up-to-date**:

- **Verified Boot (AVB / dm-verity)** establishes a cryptographic *chain of trust* from the
  hardware root of trust through every partition, so the device only runs unmodified, signed
  system software (and can detect/refuse tampering).
- **OTA (Over-The-Air) updates**, especially the **A/B (seamless) update** scheme, deliver new
  system software safely with automatic rollback if an update fails to boot.

See [Project Treble & HALs](treble_hal.md) for the partitions involved and
[APK/AAB Packaging & Signing](app_signing.md) for *app*-level signing (distinct from
*system* image signing covered here).

## Verified Boot

### Chain of trust

```text
Hardware Root of Trust (immutable boot ROM, fused keys)
      │ verifies signature of
      ▼
Bootloader  ──verifies──▶  boot/init partition (vbmeta + kernel/ramdisk)
                                  │ AVB descriptor + dm-verity hash tree
                                  ▼
                            system / vendor / product partitions
```

Each stage cryptographically verifies the next before handing off control. The root key is
burned into hardware, so the chain can't be re-rooted by software.

### AVB (Android Verified Boot 2.0)

AVB uses a **`vbmeta`** partition that contains signed descriptors and hashes for the other
partitions. The bootloader verifies `vbmeta` against the device's root key, then trusts the
partition hashes it references.

### dm-verity

For large read-only partitions (`/system`, `/vendor`), verifying the whole image up front
would be slow. **dm-verity** is a kernel device-mapper target that builds a **Merkle hash
tree** over the partition's blocks. Blocks are verified **lazily, on read**: any block whose
hash doesn't match the signed root hash triggers an I/O error (corruption detected).

```text
            root hash (signed, in vbmeta)
                  │
          ┌───────┴───────┐
       hash            hash         ← intermediate hash nodes
     ┌──┴──┐         ┌──┴──┐
   blk0  blk1  ...  blkN-1 blkN     ← actual data blocks (verified on read)
```

### Verified boot state & rollback protection

The bootloader reports a **boot state** (GREEN/YELLOW/ORANGE/RED) reflecting how trusted the
running software is, and uses **rollback indexes** stored in tamper-evident storage to prevent
downgrading to an older, vulnerable image.

| State | Meaning |
|-------|---------|
| GREEN | Locked device, verified with OEM key |
| YELLOW | Locked device, verified with user-supplied key (shows warning) |
| ORANGE | Unlocked bootloader — verification not enforced (shows warning) |
| RED | Verification failed / corruption — device won't boot normally |

```bash
adb shell getprop ro.boot.verifiedbootstate     # green / yellow / orange
adb shell getprop ro.boot.flash.locked           # 1 = bootloader locked
# Apps can check device integrity via the Play Integrity API (replaces SafetyNet Attestation)
```

## OTA Updates

### A/B (Seamless) Updates

Modern Android uses **two copies of each partition** — **slot A** and **slot B**. The system
runs from one slot while an update is written to the *other* slot in the background.

```text
Running on slot A (active)            Update written to slot B (inactive)
   /system_a  /vendor_a  ...    →     /system_b  /vendor_b  ...  (downloaded + verified)
                                         │ mark slot B "bootable", set as active
   Reboot ──────────────────────────────┘
   Boot slot B; if it fails N times → bootloader reverts to slot A (rollback)
```

Benefits:

- **No downtime / no recovery-mode "Android is updating…" screen** — the update applies while
  you keep using the device.
- **Safe rollback**: if the new slot fails to boot, the bootloader automatically falls back to
  the known-good slot.
- Combined with **streaming updates**, blocks are written directly without needing a full
  staging copy.

```bash
adb shell getprop ro.boot.slot_suffix      # _a or _b — current slot
adb shell bootctl get-current-slot         # (if bootctl HAL exposed)
adb shell dumpsys update_engine            # update_engine status / progress
```

`update_engine` is the daemon that applies A/B payloads; `update_verifier` confirms the new
slot's dm-verity blocks are readable before marking the update successful.

### Virtual A/B (Android 11+)

To avoid doubling storage for every partition, **Virtual A/B** keeps a single physical copy
plus **snapshots** (via `dm-snapshot` / dynamic partitions) for the inactive slot, getting A/B
safety with much less disk overhead. Dynamic partitions live in a **`super`** partition that
can be repartitioned with OTAs.

### Non-A/B (legacy) updates

Older/low-storage devices use a single set of partitions and apply updates from **recovery**
mode, applying a patch and showing the "installing system update" screen. No seamless
background apply and weaker rollback story.

### OTA package types

| Type | Contents | Use |
|------|----------|-----|
| **Full OTA** | Entire new images | First install / large jumps / recovery |
| **Incremental (delta) OTA** | Block diffs from a specific source build | Smaller monthly updates |

OTA packages are themselves **signed**; the device verifies the package signature before
applying, and AVB verifies the resulting images at boot.

```bash
# Sideload a (signed) OTA package manually
adb reboot recovery
adb sideload ota_package.zip
```

## How They Work Together

1. Device receives a **signed OTA** (full or delta).
2. `update_engine` writes/patches the **inactive slot** and verifies dm-verity hashes.
3. Bootloader is told the new slot is active with a limited number of boot attempts.
4. On reboot, **AVB** verifies `vbmeta`/partitions against the hardware root of trust and
   enforces **rollback protection**.
5. If boot succeeds, the slot is marked good; if it fails, the bootloader **rolls back**.

## Best Practices

1. **Keep the bootloader locked** in production; unlocking disables verification (ORANGE).
2. **Respect rollback indexes** — never ship an update that lowers security patch level.
3. **Use A/B (or Virtual A/B)** for resilient, no-downtime updates.
4. **Verify integrity in apps via the Play Integrity API**, not by reading props that root can spoof.
5. **Test OTA on both slots** and validate `update_verifier` passes before release.
6. **Sign OTA and images with securely-stored release keys**; rotate per AOSP guidance.

## Resources

- [Verified Boot — AOSP](https://source.android.com/docs/security/features/verifiedboot)
- [AVB (libavb)](https://android.googlesource.com/platform/external/avb/+/master/README.md)
- [dm-verity](https://source.android.com/docs/security/features/verifiedboot/dm-verity)
- [A/B (seamless) updates](https://source.android.com/docs/core/ota/ab)
- [Virtual A/B](https://source.android.com/docs/core/ota/virtual_ab)
- [Play Integrity API](https://developer.android.com/google/play/integrity)

### Related Files

- [Project Treble & HALs](treble_hal.md) — partitions and the system/vendor split
- [APK/AAB Packaging & Signing](app_signing.md) — app-level (not system) signing
- [SELinux on Android](selinux_android.md) — complementary runtime protection

## Where this connects

- [App signing](app_signing.md) — the signing concepts extended to whole images
- [Mainline & APEX](mainline_apex.md) — updatable signed modules
- [Treble & HALs](treble_hal.md) — A/B partitions and the vendor split
- [Platform development](platform_dev.md) — building and signing system images
- [Secure boot](../embedded/secure_boot.md) — the same chain-of-trust idea on MCUs
- [SELinux on Android](selinux_android.md) — runtime integrity after boot
