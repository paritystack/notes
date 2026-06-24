# APK/AAB Packaging & Signing

## Overview

Every Android app is distributed as a package that must be **cryptographically signed** before
it can be installed. Signing proves the app's authorship and guarantees that updates come from
the same author (the **same-signature update rule**). This document covers the APK and AAB
container formats, the v1–v4 signature schemes, Play App Signing, and the `bundletool`/split
APK delivery model.

This is **app-level** signing — distinct from *system image* signing in
[Verified Boot & OTA](verified_boot_ota.md). See [Project Mainline & APEX](mainline_apex.md)
for how modular system components are packaged.

## APK vs AAB

| Format | What it is | Where used |
|--------|------------|------------|
| **APK** (Android Package) | A ZIP containing DEX, resources, `AndroidManifest.xml`, native libs, `resources.arsc` | Installed directly on devices; final delivery artifact |
| **AAB** (Android App Bundle) | A publishing format (also a ZIP) holding all compiled code/resources **without** generating APKs | Uploaded to Google Play; Play generates optimized APKs per device |

Since August 2021, new apps on Google Play must be published as **AABs**. Play then uses the
bundle to generate and sign **split APKs** tailored to each device (density, ABI, language),
reducing download size.

```text
AAB (developer uploads)
   └─ Google Play (bundletool) generates per-device APK sets:
        base.apk + split_config.arm64_v8a.apk + split_config.xxhdpi.apk + split_config.en.apk
```

### APK contents

```text
my-app.apk (ZIP)
├── AndroidManifest.xml      (binary XML)
├── classes.dex              (+ classes2.dex … for multidex)
├── resources.arsc           (compiled resources table)
├── res/                     (compiled resources, drawables, layouts)
├── lib/<abi>/*.so           (native libraries per ABI)
├── assets/                  (raw assets, incl. baseline.prof)
└── META-INF/                (signature files / signing block)
```

## Signature Schemes

Android has four signing schemes; an APK is typically signed with **multiple** for backward
compatibility. The verifier uses the **highest** scheme the platform supports.

| Scheme | Since | What it protects | Notes |
|--------|-------|------------------|-------|
| **v1 (JAR signing)** | Android 1.0 | Per-file digests in `META-INF` (`MANIFEST.MF`, `CERT.SF`, `CERT.RSA`) | Doesn't protect ZIP metadata; slow verify; insecure alone on modern APIs |
| **v2 (APK Signature Scheme v2)** | Android 7.0 | The **entire APK** as a whole, via an **APK Signing Block** before the ZIP central directory | Faster, tamper-evident over the full file |
| **v3** | Android 9.0 | v2 + **key rotation** (signing certificate lineage) | Lets you change signing keys while proving continuity |
| **v3.1 / v4** | Android 11/12 | v4 = streaming-friendly signature (Merkle hash tree) stored in a separate `.apk.idsig` | Enables **incremental install** (ADB/Play); used *alongside* v2/v3 |

```text
APK layout with v2+ signing:
[ ZIP entries (DEX, res, ...) ][ APK Signing Block (v2/v3 sigs) ][ ZIP Central Directory ][ EoCD ]
```

### Signing with apksigner

`apksigner` (Android SDK build-tools) is the recommended CLI. `jarsigner` only does v1 and
should not be used alone for modern apps.

```bash
# Sign an APK with v1+v2+v3 enabled (default for recent build-tools)
apksigner sign \
  --ks my-release.keystore \
  --ks-key-alias myalias \
  --out app-release-signed.apk \
  app-release-unsigned.apk

# Verify and show which schemes are present
apksigner verify --verbose --print-certs app-release-signed.apk
# (Verified using v1 scheme: false) ... (Verified using v2 scheme: true) ...
```

### Creating a signing key

```bash
keytool -genkeypair -v \
  -keystore my-release.keystore \
  -alias myalias \
  -keyalg RSA -keysize 2048 -validity 10000
```

The **upload/release private key must be kept secret and backed up** — losing it (without Play
App Signing key rotation) means you can never update the app under the same package.

### Gradle signing config

```gradle
android {
    signingConfigs {
        release {
            storeFile file("my-release.keystore")
            storePassword System.getenv("KEYSTORE_PASSWORD")
            keyAlias "myalias"
            keyPassword System.getenv("KEY_PASSWORD")
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
        }
    }
}
```

Keep secrets out of source control — use environment variables, `keystore.properties`
(gitignored), or a CI secret store.

## Play App Signing

With **Play App Signing**, Google manages the **app signing key** (the key end-user devices
verify) in a secure key store. You sign your AAB with an **upload key** and Play re-signs the
generated APKs with the real app signing key.

```text
Developer:  AAB signed with UPLOAD key  ──▶  Google Play
Google Play: re-signs generated APKs with APP SIGNING key (held by Google)  ──▶  devices
```

Benefits:

- If the **upload key** is lost/compromised, you can **reset** it without losing the app
  (the app signing key is safe with Google).
- Enables optimized split APK delivery and smaller downloads.
- v3 **key rotation** can be managed through Play.

## Split APKs & Dynamic Delivery

From a single AAB, Play generates:

- **Base APK** — core code/resources, always installed.
- **Configuration splits** — per density / ABI / language.
- **Feature splits** (`<dist:module>` / Play Feature Delivery) — optional features installed
  on demand.

```bash
# Generate an APK set from an AAB locally and install matching splits to a device
bundletool build-apks --bundle=app.aab --output=app.apks \
  --ks=my-release.keystore --ks-key-alias=myalias
bundletool install-apks --apks=app.apks
```

## The Same-Signature Update Rule

An update can only install over an existing app if it's signed with the **same key** (or a
rotated key in the same v3 lineage). This also gates **`sharedUserId`** (legacy) and
**signature-level permissions** — two apps signed by the same key can be granted access to
each other's `signature`-protected components.

```bash
# See what key an installed app was signed with
adb shell dumpsys package com.example.app | grep -A2 -i signing
```

## Best Practices

1. **Publish AABs**; let Play generate and sign optimized APKs.
2. **Enable Play App Signing** so a lost upload key is recoverable.
3. **Back up and tightly restrict access to signing keys**; never commit them.
4. **Sign with v2+ (and v3 for rotation, v4 for incremental install)** — don't rely on v1.
5. **Use `apksigner`, not `jarsigner`**, and verify with `apksigner verify --print-certs`.
6. **Pin CI secrets** for keystore credentials; rotate via Play if compromised.
7. **Validate splits locally** with `bundletool` before release.

## Resources

- [Sign your app — Android Developers](https://developer.android.com/studio/publish/app-signing)
- [APK Signature Scheme v2/v3/v4 — AOSP](https://source.android.com/docs/security/features/apksigning)
- [Android App Bundle](https://developer.android.com/guide/app-bundle)
- [bundletool](https://developer.android.com/tools/bundletool)
- [Play App Signing](https://support.google.com/googleplay/android-developer/answer/9842756)

### Related Files

- [Verified Boot & OTA](verified_boot_ota.md) — system image signing (different from app signing)
- [Project Mainline & APEX](mainline_apex.md) — packaging of modular system components
- [ART & Dalvik Runtime](art_runtime.md) — DEX/baseline profile that ships inside the package

## Where this connects

- [App security](app_security.md) — why signing identity matters
- [Gradle deep dive](gradle_android.md) — signing config in the build
- [Verified boot & OTA](verified_boot_ota.md) — platform-level signature chains
- [Digital signatures](../security/digital_signatures.md), [Certificates](../security/certificates.md) — the crypto under APK signing
- [adb](adb.md) — installing debug-signed builds
- [Mainline & APEX](mainline_apex.md) — signed modular OS components
