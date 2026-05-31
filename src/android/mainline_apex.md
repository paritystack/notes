# Project Mainline & APEX

## Overview

**Project Mainline** (Android 10+) lets Google update core OS components **directly through
Google Play**, without a full system OTA. Critical subsystems (media codecs, network stack
pieces, permission controller, ART, conscrypt, etc.) are packaged as **modules** that can be
updated independently of the device's full firmware — getting security fixes and improvements
to billions of devices quickly, even on older releases.

The packaging format that makes most of these modules possible is **APEX** (Android Pony
EXpress), a container format for **lower-level system components** that sit below the app
layer but need independent updatability.

See [Project Treble & HALs](treble_hal.md) for the framework/vendor split that Mainline builds
on and [APK/AAB Packaging & Signing](app_signing.md) for app-level packaging.

## Why Mainline Exists

Traditionally, fixing a bug in a core component (say, the media framework) required a full
system OTA — gated by OEMs and carriers, often taking months or never reaching older devices.
Mainline decouples these components so Google can ship them like apps.

```text
Pre-Mainline:  bug fix in media stack → full system OTA → OEM → carrier → (maybe) your device
Mainline:      bug fix in media module → Google Play system update → device, directly
```

Updates arrive as **Google Play system updates** (Settings → Security → Google Play system
update), separate from app updates and from full OS OTAs.

## Module Packaging: APEX vs APK

Mainline modules ship in one of two forms (or both, as an **APK-in-APEX**):

| Form | For | Notes |
|------|-----|-------|
| **APK** | App-layer modules (e.g. Permission Controller, some UI) | Standard app package, updated via Play |
| **APEX** | Lower-level native components, libraries, ART, HALs | Mounted at boot, before most apps start |

### Why not just use APKs for everything?

APKs are designed for apps and are only available **after** the system is largely up. Many core
components (native libraries, ART, the media C/C++ stack) are needed **very early in boot** and
must present a stable on-disk layout. APEX solves this.

## APEX Format

An `.apex` file is a container holding a **filesystem image** (typically ext4/erofs) plus
metadata and its own signing info.

```text
foo.apex
├── apex_manifest.json     (module name + version)
├── apex_payload.img       (ext4/erofs image: /lib, /bin, etc. for the module)
├── apex_pubkey            (public key for the payload)
└── META-INF/ (APK-style signature over the whole container)
```

### How APEX is activated at boot

APEX modules must be available **before** Zygote/system_server start, so activation happens
early via `apexd`:

```text
init → apexd
   ├─ verifies each .apex signature (and dm-verity over the payload image)
   ├─ loop-mounts apex_payload.img
   └─ bind-mounts it at /apex/<module-name>/   (e.g. /apex/com.android.art/)
        └─ now its libs/binaries are on the path before Zygote preloads
```

Each module appears under **`/apex/<name>/`** (and `/apex/<name>@<version>/`). Updates are
staged and activated on the next reboot (APEX updates generally require a reboot, unlike most
APK module updates).

```bash
# List active APEX modules and versions
adb shell ls -l /apex
adb shell cmd -w apexservice getActivePackages   # (or)
adb shell pm list packages --apex-only
adb shell cmd apexservice getStagedSessions
```

## Examples of Mainline Modules

| Module | Packaging | What it covers |
|--------|-----------|----------------|
| `com.android.art` | APEX | The ART runtime + core libraries (see [ART](art_runtime.md)) |
| `com.android.media` / `media.swcodec` | APEX | Media framework & software codecs (security-critical) |
| `com.android.conscrypt` | APEX | TLS/crypto provider |
| `com.android.tethering` | APEX | Connectivity/tethering stack |
| `com.android.permission` | APK | Permission Controller UI/logic |
| `com.android.adbd` | APEX | adb daemon |

The exact set grows each release. ART becoming a Mainline (APEX) module is notable: the runtime
itself can now be updated via Play.

## Updating & Rollback

- Modules are **signed**; `apexd`/PackageManager verify signatures (and dm-verity on the
  payload) before activation — same trust principles as [Verified Boot](verified_boot_ota.md).
- APEX updates are **staged then activated on reboot**; if a new module version fails to boot
  cleanly, the system **rolls back** to the factory/previous version.
- Modules carry **version codes**; rollback protection prevents downgrading to vulnerable
  versions.

```bash
adb shell dumpsys rollback           # rollback availability/history
```

## Implications for Developers

- **App developers** rarely build APEX modules, but should know that core behavior (codecs,
  TLS, runtime) can change *between* full OS releases via Mainline — test against current
  module versions, not just an API level.
- **Platform/OEM developers** must keep certain modules as Google-signed and updatable; OEMs
  may prebuild/preinstall a version but Play can update it.

## Best Practices

1. **Don't assume component behavior is frozen per API level** — Mainline modules update
   independently.
2. **For platform work, package early-boot native components as APEX**, app-layer logic as APK.
3. **Sign APEX with securely-managed keys** and honor rollback/version rules.
4. **Test staged APEX updates and reboot activation** in CI before shipping.
5. **Use `/apex/<name>/` paths**, never hardcode versioned mount points.

## Resources

- [Modular System Components / Mainline — AOSP](https://source.android.com/docs/core/ota/modular-system)
- [APEX file format](https://source.android.com/docs/core/ota/apex)
- [ART Mainline module](https://source.android.com/docs/core/runtime)
- [Google Play system updates](https://support.google.com/android/answer/7680439)

### Related Files

- [Project Treble & HALs](treble_hal.md) — the framework/vendor split Mainline builds on
- [Verified Boot & OTA](verified_boot_ota.md) — signing/rollback principles for system images
- [ART & Dalvik Runtime](art_runtime.md) — now itself an APEX (Mainline) module
- [APK/AAB Packaging & Signing](app_signing.md) — app-level packaging counterpart
