# SystemServer & Core Services

## Overview

`system_server` is the process that hosts the bulk of the Android framework's **system
services** — ActivityManager, PackageManager, WindowManager, PowerManager, and ~80 others.
It is the second process Zygote forks at boot (the first "real" Android process after Zygote
itself), and it runs with the privileged `system` UID. Almost every framework API an app calls
(`getSystemService(...)`, starting activities, querying packages) ends up as a **Binder** call
into a service living in `system_server`.

See [Zygote & App Startup](zygote_startup.md) for how it's launched, [Binder](binder.md) for the
IPC, and [Android Internals](internals.md) for the surrounding architecture.

## Where system_server Comes From

```text
init → zygote (ZygoteInit.main)
          └─ forkSystemServer()
                └─ SystemServer.main()
                     ├─ startBootstrapServices()   (ActivityManager, PackageManager, Power, ...)
                     ├─ startCoreServices()         (BatteryService, UsageStats, ...)
                     └─ startOtherServices()        (Window, Input, Network, Audio, ...)
```

`system_server` is forked from Zygote (so it shares the preloaded runtime), but runs as UID
`system` (1000) with extensive permissions. If `system_server` crashes, the entire Android
framework restarts (a "soft reboot" / runtime restart) — but the kernel and `init` stay up.

```bash
adb shell ps -A | grep system_server
adb shell dumpsys -l            # list every registered system service
adb shell service list          # list Binder services registered with servicemanager
```

## ServiceManager — The Binder Registry

System services register themselves by name with **`servicemanager`**, the well-known Binder
context manager (handle 0). Clients look services up by name and receive a Binder proxy.

```text
system_server: ServiceManager.addService("activity", AMS_binder)
app:           IBinder b = ServiceManager.getService("activity")
               IActivityManager am = IActivityManager.Stub.asInterface(b)
               am.startActivity(...)   // Binder transaction → system_server
```

App developers don't call `ServiceManager` directly; they use `Context.getSystemService()`,
which returns a **manager** object (e.g. `ActivityManager`) that wraps the Binder proxy.

```kotlin
val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
val wm = context.getSystemService(Context.WINDOW_SERVICE)   as WindowManager
```

## The Core Services

### ActivityManagerService (AMS) → ActivityTaskManagerService (ATMS)

The heart of app lifecycle and process management. (In Android 10+, activity/task
responsibilities were split out into **ActivityTaskManagerService**.)

- Starts activities/services/broadcasts and resolves which process should run them.
- Asks **Zygote** to fork new app processes and tracks their lifecycle.
- Maintains the **process priority / OOM adjustment** (`oom_adj`) used by the
  Low Memory Killer to decide what to kill under memory pressure.
- Drives ANR (Application Not Responding) detection.

```bash
adb shell dumpsys activity activities     # task/activity stack
adb shell dumpsys activity processes      # process records + oom_adj
adb shell dumpsys activity broadcasts     # pending/historical broadcasts
```

### PackageManagerService (PMS)

Owns everything about installed packages: parsing `AndroidManifest.xml`, granting permissions,
tracking components, resolving Intents, managing app data dirs and signatures.

```bash
adb shell dumpsys package com.example.app   # permissions, components, signatures, paths
adb shell pm list packages -3               # third-party packages
adb shell pm path com.example.app           # APK path(s)
```

### WindowManagerService (WMS) & InputManagerService

WMS manages the **z-ordered set of windows**, surfaces (handing buffers to SurfaceFlinger —
see [Graphics Stack](graphics_stack.md)), transitions, and focus. InputManagerService reads
input events from the kernel (`/dev/input`) and dispatches them to the focused window.

```bash
adb shell dumpsys window windows
adb shell dumpsys input
```

### Other notable services

| Service | Responsibility |
|---------|----------------|
| `PowerManagerService` | Wakelocks, screen state, Doze (see background docs) |
| `BatteryService` / `BatteryStats` | Power source + per-app battery accounting |
| `ConnectivityService` | Network selection, routing between transports |
| `NotificationManagerService` | Posting/ranking notifications, channels |
| `AlarmManagerService` | Scheduled wakeups |
| `JobSchedulerService` | Deferred/constrained background work |
| `display`/`SurfaceFlinger`* | Compositing (*SurfaceFlinger is its own native process) |

## A Binder Call End-to-End

```text
App: startActivity(intent)
  └─ Instrumentation → IActivityTaskManager proxy (Binder)
        └─ kernel binder driver copies the transaction
              └─ ATMS in system_server (system UID) handles it
                    ├─ permission/UID checks
                    ├─ resolve target via PackageManager
                    ├─ (if needed) ask Zygote to fork the target process
                    └─ schedule lifecycle callbacks back to the app via ApplicationThread
```

Because these calls cross UIDs, **permission enforcement happens in `system_server`** using
`Binder.getCallingUid()/getCallingPid()` — the kernel guarantees the caller identity, so a
malicious app can't spoof it.

## Stability & Watchdog

A dedicated **Watchdog** thread in `system_server` periodically pings critical service threads
(holding their locks). If a thread is stuck past a timeout (deadlock/livelock), Watchdog
**kills `system_server`** to force a clean framework restart rather than leaving the device
wedged.

```bash
adb logcat | grep -iE "watchdog|system_server.*crash|FATAL EXCEPTION in system"
```

## Best Practices (for platform developers)

1. **Never block a system service's main/handler thread** — Watchdog will reboot the framework.
2. **Always verify the caller** with `getCallingUid()` and permission checks before acting.
3. **Use `dumpsys <service>`** as the first diagnostic step for framework misbehavior.
4. **Add new system services via the SystemServiceManager lifecycle**, respecting boot phases
   (`PHASE_SYSTEM_SERVICES_READY`, `PHASE_BOOT_COMPLETED`).
5. **Keep Binder transactions small** — the transaction buffer is limited (~1MB shared);
   large payloads cause `TransactionTooLargeException`.

## Resources

- [System services overview — AOSP](https://source.android.com/docs/core/architecture)
- [Binder & servicemanager](https://source.android.com/docs/core/architecture/hidl/binder-ipc)
- [Activity lifecycle (app side)](https://developer.android.com/guide/components/activities/activity-lifecycle)

### Related Files

- [Binder](binder.md) — the IPC mechanism all services use
- [Zygote & App Startup](zygote_startup.md) — how processes are forked on AMS's request
- [Graphics Stack](graphics_stack.md) — WMS/SurfaceFlinger relationship
- [Android Internals](internals.md) — overall architecture
