# Permissions & Privacy

## Overview

Android brokers access to sensitive resources — location, camera, contacts, storage,
notifications — through a **permission** system, and has steadily shifted toward
**privacy-by-default**: runtime prompts, scoped storage, one-time grants, and auto-reset of unused
permissions. This page covers the permission model, the runtime request flow, scoped storage, and
the privacy surfaces an app must respect. It builds on the sandbox described in
[App Security](app_security.md) and [SELinux on Android](selinux_android.md), gates the storage in
[Data & Persistence](data_persistence.md), constrains [Background Execution](background_work.md),
and protects the components in [App Components & Lifecycle](app_components.md).

## Permission Model

Permissions are declared in the manifest and classified by **protection level**:

| Level | Granted | Examples |
|-------|---------|----------|
| **Normal** | Automatically at install | `INTERNET`, `VIBRATE`, `ACCESS_NETWORK_STATE` |
| **Dangerous** | User prompt at runtime | `CAMERA`, `ACCESS_FINE_LOCATION`, `READ_CONTACTS` |
| **Signature** | Only to apps signed with the same key (or platform) | privileged/system perms |

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

Enforcement sits on the [app-sandbox](app_security.md): each app is a distinct Linux UID, and
[SELinux](selinux_android.md) plus the framework's **AppOps** track and revoke per-op access.
Dangerous permissions are grouped (granting one in a group eased others historically, but treat
each as individually requested today).

## Runtime Permissions (Android 6+)

Dangerous permissions must be requested **in context**, when the user triggers the feature — not
at launch. Use the Activity Result API (see [App Components](app_components.md)):

```kotlin
val requestCamera = registerForActivityResult(RequestPermission()) { granted ->
    if (granted) openCamera() else showRationaleOrSettings()
}

fun onTakePhoto() {
    when {
        hasPermission(CAMERA) -> openCamera()
        shouldShowRequestPermissionRationale(CAMERA) -> showRationale { requestCamera.launch(CAMERA) }
        else -> requestCamera.launch(CAMERA)
    }
}
```

- **`shouldShowRequestPermissionRationale`** is true after one denial — explain *why* before
  re-asking.
- **Permanent denial** ("Don't ask again" / second denial) returns `false` with no dialog — route
  the user to app settings.
- **One-time grants** (Android 11+) — location/camera/mic can be granted "only this time"; the
  grant is revoked when the app leaves the foreground, so re-check before each use.
- **Auto-reset** (Android 11+) — permissions of unused apps are revoked automatically; never assume
  a past grant persists.
- **`RequestMultiplePermissions`** for batches.

## Scoped Storage (Android 10+)

Apps can no longer roam the raw filesystem. Access tiers:

| Need | Mechanism | Permission |
|------|-----------|------------|
| App's own files | `filesDir`/`cacheDir`/`getExternalFilesDir()` | none |
| Shared **media** | **MediaStore** (`content://` queries) | `READ_MEDIA_IMAGES/VIDEO/AUDIO` (13+) |
| Pick **media** | **Photo Picker** (`PickVisualMedia`) | **none** — system UI |
| Pick/open **documents** | **Storage Access Framework** (`ACTION_OPEN_DOCUMENT`) | none (user grants per-URI) |

`READ_EXTERNAL_STORAGE` is replaced by granular `READ_MEDIA_*` on Android 13+, and the **Photo
Picker** needs no permission at all — prefer it for image/video selection. See
[Data & Persistence](data_persistence.md) for the storage APIs themselves.

## Location

- **Foreground**: request `ACCESS_COARSE_LOCATION` and/or `ACCESS_FINE_LOCATION`. Android 12+ lets
  the user grant **approximate** (coarse) even when you ask for precise — handle both.
- **Background** (`ACCESS_BACKGROUND_LOCATION`) is a **separate** grant, requested *after*
  foreground is granted, and routes the user to settings. Play heavily scrutinizes its use — only
  request it with a clear, disclosed need. Tie background location work to
  [Background Execution](background_work.md).

## Notifications

Posting notifications requires the **`POST_NOTIFICATIONS`** runtime permission on Android 13+ —
including the ongoing notification for a foreground service (see
[App Components](app_components.md)). Request it in context the first time you'd notify. (A
dedicated notifications page is a candidate future addition; for now `background_work.md` covers
foreground-service notifications.)

## Privacy Surfaces

Beyond permissions, the platform exposes user-facing privacy signals an app must not fight:

- **Privacy dashboard** (Android 12+) — shows recent location/camera/mic access.
- **Mic & camera indicators** + the global toggles — assume access can be off.
- **Clipboard access** shows a toast (Android 12+); read clipboard only on explicit user action.
- **Package visibility** (Android 11+) — you can't see other installed apps unless you declare
  `<queries>` or hold a broad visibility permission.
- **Advertising ID** is gated behind a permission and user opt-out; don't use it as a stable
  device identifier.

## Pitfalls

- **Requesting permissions on launch** instead of in-context — high denial rates; ask when the
  feature is used.
- **Not handling permanent denial** — app appears broken; deep-link to settings with an
  explanation.
- **Assuming raw file paths** post-scoped-storage — use MediaStore/SAF/Photo Picker.
- **Over-broad `<uses-permission>`** — review at audit; unused dangerous perms hurt Play review and
  trust.
- **Forgetting `POST_NOTIFICATIONS`** on Android 13+ — notifications silently dropped.
- **Caching a grant** — one-time grants and auto-reset mean you must re-check before each use.
- **Requesting background location** without justification — common Play rejection.

## Where this connects

- [App Security](app_security.md) — the UID sandbox and signing behind permissions
- [SELinux on Android](selinux_android.md) — mandatory access control under the framework checks
- [Data & Persistence](data_persistence.md) — scoped storage, MediaStore, SAF
- [App Components & Lifecycle](app_components.md) — exported components, FileProvider, FGS perms
- [Background Execution](background_work.md) — background location, foreground-service notifications
