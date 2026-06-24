# Project Treble & HALs

## Overview

**Project Treble** (Android 8.0, Oreo) was a major re-architecture that separated the Android
OS **framework** from the **vendor implementation** (SoC- and device-specific code). The goal:
let device makers update the Android framework without re-integrating every chip vendor's
low-level code, dramatically reducing the cost and time of OS updates.

The mechanism is a stable, versioned boundary between framework and vendor, implemented through
**Hardware Abstraction Layers (HALs)** defined in interface description languages (**HIDL**,
later **AIDL**) and a compatibility-checking system (**VINTF**).

See [Binder](binder.md) for the IPC that underpins AIDL HALs and
[Android Internals](internals.md) for where HALs sit in the stack.

## The Problem Treble Solved

Before Treble, the framework and vendor HAL code were tightly coupled and compiled together.
A new Android version meant every SoC vendor had to port and re-validate their code for each
device — a slow, expensive chain (Google → SoC vendor → OEM → carrier).

```text
Pre-Treble (monolithic):
   Framework  ⇄  tightly-coupled vendor HALs  ⇄  kernel/drivers
   → any framework update forces full vendor re-integration

Post-Treble (split):
   /system  (framework, OS) ──VINTF──┐
                                     │  stable HAL interfaces (HIDL/AIDL)
   /vendor  (HAL impls, drivers) ────┘
   → framework on /system can be updated independently of /vendor
```

## Partition Layout

Treble formalized a partition split so the OS and vendor pieces can be updated separately:

| Partition | Owner | Contents |
|-----------|-------|----------|
| `/system` | Google/OEM (OS) | Android framework, system apps, ART |
| `/vendor` | SoC/OEM | HAL implementations, vendor libs, SEPolicy for vendor |
| `/product` | OEM | OEM/carrier customizations |
| `/odm`     | ODM | Board-specific bits |
| `/system_ext` | OEM | Closely-coupled-to-system OEM additions |

A **Generic System Image (GSI)** — a pure-AOSP `/system` — can boot on any Treble-compliant
device, which is how Google runs **VTS** (Vendor Test Suite) to verify the framework/vendor
contract.

## HAL: Hardware Abstraction Layer

A HAL is a stable interface that the framework calls to talk to hardware (camera, audio,
sensors, wifi, etc.) without knowing the device-specific implementation behind it. The vendor
provides the implementation; the framework codes against the versioned interface.

### HAL flavors over time

| Type | Era | Notes |
|------|-----|-------|
| **Legacy / conventional HAL** | pre-8.0 | `.so` loaded into the framework process; no process isolation |
| **HIDL HAL** | 8.0–10 | HAL Interface Definition Language; runs in its own process, Binderized over `/dev/hwbinder` |
| **Stable AIDL HAL** | 11+ | AIDL replaces HIDL as the preferred HAL IDL; uses regular Binder with stability guarantees |

Google is **deprecating HIDL** in favor of **Stable AIDL** for new and migrated HALs.

### Binderized vs passthrough

- **Binderized**: the HAL runs in a separate vendor process; framework talks to it over
  (hw)Binder. Provides isolation and independent updates.
- **Passthrough**: the HAL `.so` is loaded directly into the client process (mainly a
  migration/compat mode).

```text
Framework process            Vendor HAL process
   client proxy  ──hwbinder/binder──▶  HAL implementation  ──▶  kernel driver
```

## HIDL Example

```cpp
// hardware/interfaces/foo/1.0/IFoo.hal  (HIDL)
package android.hardware.foo@1.0;

interface IFoo {
    doSomething(int32_t cookie) generates (int32_t result);
};
```

HIDL interfaces are **versioned** (`@1.0`, `@1.1`, …); a new minor version must extend, not
break, the previous one.

## Stable AIDL HAL Example (preferred today)

```java
// hardware/interfaces/foo/aidl/android/hardware/foo/IFoo.aidl
package android.hardware.foo;

@VintfStability          // marks this AIDL as a stable vendor/framework interface
interface IFoo {
    int doSomething(int cookie);
}
```

```cpp
// Vendor implementation registers the service so the framework can find it
auto foo = ndk::SharedRefBase::make<Foo>();
const std::string name = std::string(IFoo::descriptor) + "/default";
AServiceManager_addService(foo->asBinder().get(), name.c_str());
```

`@VintfStability` AIDL interfaces are frozen per version under `aidl_api/` and changes are
checked at build time so the framework/vendor contract can't silently break.

## VINTF — Vendor Interface Object

**VINTF** describes, and checks compatibility between, what the framework *requires* and what
the vendor *provides*:

- **Device Manifest** (`/vendor/etc/vintf/manifest.xml`) — HAL versions the vendor implements.
- **Framework Compatibility Matrix** (`/system/etc/vintf/compatibility_matrix.xml`) — HAL
  versions/kernel requirements the framework needs.

At boot (and at build/OTA time), these are matched. If the vendor doesn't provide a HAL
version the framework requires, the device is non-compatible. This is the contract that lets
a GSI boot on any Treble device.

```bash
# Inspect HAL manifests and matrices on a device
adb shell cat /vendor/etc/vintf/manifest.xml
adb shell lshal                      # list running HAL services and their clients
adb shell lshal --types=b            # only binderized HALs
adb shell vintf                      # dump VINTF objects / compatibility
```

## GKI — Generic Kernel Image

Complementing Treble on the kernel side, **GKI** (Android 11+) splits the kernel into a
Google-maintained **generic core** plus vendor modules loaded via a stable
**KMI** (Kernel Module Interface). This lets the core kernel be updated independently of vendor
drivers — the kernel analog of the framework/vendor split.

## Best Practices

1. **Use Stable AIDL, not HIDL**, for new HALs — HIDL is deprecated.
2. **Never break a frozen interface version** — add a new version and keep the old one.
3. **Keep HALs binderized** for isolation and independent updatability.
4. **Validate with VTS and a GSI** to prove framework/vendor compatibility.
5. **Mark stable interfaces `@VintfStability`** and freeze their `aidl_api/` snapshots.
6. **Put vendor SELinux policy on `/vendor`**, not `/system` (see [SELinux on Android](selinux_android.md)).

## Resources

- [Project Treble overview — AOSP](https://source.android.com/docs/core/architecture/treble)
- [HAL types & AIDL HALs](https://source.android.com/docs/core/architecture/hal)
- [Stable AIDL](https://source.android.com/docs/core/architecture/aidl/stable-aidl)
- [VINTF](https://source.android.com/docs/core/architecture/vintf)
- [Generic Kernel Image (GKI)](https://source.android.com/docs/core/architecture/kernel/generic-kernel-image)

### Related Files

- [Binder](binder.md) — Binder/AIDL IPC underlying HALs
- [Platform Dev](platform_dev.md) — building AOSP and the partition images
- [SELinux on Android](selinux_android.md) — policy for vendor services

## Where this connects

- [Mainline & APEX](mainline_apex.md) — modular updates Treble enabled
- [Platform development](platform_dev.md) — implementing HALs in AOSP
- [Binder](binder.md) — HIDL/AIDL HALs ride over Binder
- [Internals](internals.md) — the system/vendor architecture
- [SELinux on Android](selinux_android.md) — policy across the split
- [Verified boot & OTA](verified_boot_ota.md) — partition layout and updates
