# SELinux on Android

## Overview

Android uses **SELinux** (Security-Enhanced Linux) to enforce **Mandatory Access Control
(MAC)**: even root-privileged processes are confined to exactly what policy permits. Where the
classic Linux **Discretionary Access Control** (UID/GID permissions) lets an owner do anything
with their files, MAC adds a system-wide policy that processes *cannot* override — dramatically
shrinking the damage a compromised process (even a root one) can do.

SELinux has been **enforcing by default since Android 5.0**. This document expands on the
SELinux material in [Android Internals](internals.md): the label model, domains and types,
`.te` policy authoring, `neverallow` rules, and how policy ties into Treble and CTS/VTS.

## DAC vs MAC

```text
DAC (traditional):  "Can this UID access this file?"  — owner can change perms; root bypasses
MAC (SELinux):      "Does policy allow domain X to do action Y on type Z?"  — root cannot bypass
Both must pass.  SELinux is checked AFTER DAC; a denial blocks the operation regardless of UID.
```

## Security Contexts (Labels)

Every process and object (file, socket, property, service, etc.) carries a **security context**:

```text
u : r : untrusted_app : s0 : c512,c768
│   │   │               │    └─ MLS/MCS categories (per-app isolation)
│   │   │               └────── sensitivity level
│   │   └────────────────────── TYPE (for files) / DOMAIN (for processes)
│   └────────────────────────── role (almost always 'r')
└────────────────────────────── SELinux user (almost always 'u')
```

The meaningful part on Android is the **type/domain**. Policy rules are written in terms of
domains (process types) acting on object types.

```bash
adb shell getenforce                     # Enforcing / Permissive
adb shell ps -Z | head                   # process contexts (domains)
adb shell ls -Z /data/local/tmp          # file contexts (types)
adb shell id -Z                           # current shell's context
```

### MCS categories for app isolation

Each app instance gets unique **MCS categories** (`c512,c768` above) derived from its UID/user,
so even two `untrusted_app`-domain apps can't read each other's files — the categories must
match. This complements per-app UID sandboxing.

## Domains and Types

- **Domain**: the type assigned to a *process* (e.g. `untrusted_app`, `system_server`,
  `platform_app`, `priv_app`, `vendor_init`, `hal_camera_default`).
- **Type**: the label on *objects* (e.g. `app_data_file`, `system_file`, `proc_net`).
- **Class**: the kind of object (`file`, `dir`, `socket`, `binder`, `property_service`, …).
- **Permission**: the action on a class (`read`, `write`, `open`, `call`, `set`, …).

### Domain transition on app launch

When Zygote forks an app, the new process **transitions** into the appropriate app domain:

```text
zygote (zygote domain)
   └─ fork app → app_process executes app code
        └─ transitions to untrusted_app / priv_app / platform_app
           depending on the app's signing/privilege
```

## Writing Policy: Type Enforcement (.te)

Android policy lives mostly under `system/sepolicy/` (and vendor policy under the vendor tree,
post-[Treble](treble_hal.md)). The core rules are **Type Enforcement** `allow` statements in
`.te` files.

```text
# system/sepolicy/private/myservice.te

# Define a new domain for our daemon and make it a service domain
type myservice, domain;
type myservice_exec, exec_type, file_type;

# Transition into myservice domain when init runs the executable
init_daemon_domain(myservice)

# Allow myservice to use binder to talk to servicemanager and register
binder_use(myservice)
add_service(myservice, myservice_service)

# Allow reading a specific system property
get_prop(myservice, my_prop)

# Explicit allow rule form:  allow SOURCE TARGET:CLASS { PERMISSIONS };
allow myservice my_data_file:file { read open getattr };
```

### Labeling files

`file_contexts` maps paths to types so objects get the right label:

```text
# system/sepolicy/private/file_contexts
/system/bin/myservice    u:object_r:myservice_exec:s0
/data/misc/myservice(/.*)?   u:object_r:my_data_file:s0
```

### Defining properties / services

```text
# property_contexts
my.prop.   u:object_r:my_prop:s0

# service_contexts
myservice  u:object_r:myservice_service:s0
```

## neverallow Rules

`neverallow` rules are **compile-time assertions** that certain dangerous permissions are never
granted to any domain. They don't grant anything — they make the policy build **fail** if some
`allow` rule (anywhere, including vendor policy) would violate the invariant. Many are mandated
by AOSP and checked by **CTS/VTS**.

```text
# Example: nothing in an app domain may ever execute writable memory (W^X)
neverallow { appdomain } self:process execmem;

# Untrusted apps must never directly access the kernel log
neverallow untrusted_app kmsg_device:chr_file *;
```

If your `allow` rule trips a `neverallow`, you must rethink the design — not delete the
`neverallow`. These rules are the backbone of the platform's security guarantees and a key part
of passing **CTS** (Compatibility Test Suite) / **VTS** (Vendor Test Suite).

## Debugging Denials (avc)

When SELinux blocks something, the kernel logs an **AVC (Access Vector Cache) denial**:

```bash
# View denials
adb shell dmesg | grep -i avc
adb logcat -b events | grep -i avc
# Example:
#  avc: denied { read } for comm="myservice" name="foo" dev="..." ino=...
#       scontext=u:r:myservice:s0 tcontext=u:object_r:my_data_file:s0 tclass=file permissive=0
```

Read it as: domain `myservice` was denied `read` on a `file` of type `my_data_file`. The fix is
a targeted `allow myservice my_data_file:file read;` (if legitimate and not blocked by a
`neverallow`).

```bash
# audit2allow generates candidate allow rules from denials (review before using!)
adb shell dmesg | grep avc | audit2allow
```

> **Beware `dontaudit`** rules: they suppress logging of *expected* denials. A missing log line
> doesn't always mean the access succeeded.

### Permissive vs enforcing (development only)

```bash
adb root
adb shell setenforce 0        # Permissive: denials logged but ALLOWED (debug only)
adb shell setenforce 1        # back to Enforcing
```

Per-domain `permissive` (in `.te`) can be used during bring-up, but production builds must be
**fully enforcing** with **no permissive domains** to pass CTS.

## Treble & Policy Split

Post-Treble, SELinux policy is split so vendor and platform policy can evolve independently:

| Policy | Location | Owner |
|--------|----------|-------|
| **Platform (plat) policy** | `system/sepolicy/` → `/system` | Google/AOSP |
| **Vendor policy** | device/vendor tree → `/vendor` | SoC/OEM |

Interfaces between them go through a stable **`*_attribute`** layer so a `/system` update doesn't
require recompiling vendor policy (the policy analog of the [VINTF](treble_hal.md) contract).

## Best Practices

1. **Ship fully enforcing with zero permissive domains** — required for CTS and security.
2. **Grant the minimum** — write narrow `allow` rules for specific types/permissions, not broad ones.
3. **Never weaken or remove `neverallow` rules** to make a build pass; redesign instead.
4. **Define dedicated types/domains** for new services rather than reusing broad ones.
5. **Review `audit2allow` output by hand** — it can suggest over-broad or unsafe rules.
6. **Keep vendor policy in the vendor tree** and use stable attributes across the Treble boundary.
7. **Label new files/props/services** in the appropriate `*_contexts` files.

## Resources

- [SELinux for Android — AOSP](https://source.android.com/docs/security/features/selinux)
- [Implementing SELinux policy](https://source.android.com/docs/security/features/selinux/implement)
- [Validating SELinux (CTS/denials)](https://source.android.com/docs/security/features/selinux/validate)
- [Customizing SEPolicy & device policy](https://source.android.com/docs/security/features/selinux/device-policy)

### Related Files

- [Android Internals](internals.md) — broader security/permission model
- [Project Treble & HALs](treble_hal.md) — platform/vendor policy split
- [Platform Dev](platform_dev.md) — building AOSP and `system/sepolicy`
- [SystemServer & Core Services](system_server.md) — services whose domains policy confines
