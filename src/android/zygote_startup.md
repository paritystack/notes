# Zygote & App Startup

## Overview

Every Android app process is born from **Zygote** — a special process that boots once, loads
and initializes the ART runtime plus thousands of common framework classes and resources,
and then **forks** itself to create each new app. Because forking uses copy-on-write (COW)
memory, all apps share the preloaded pages until they write to them, which makes app launch
fast and memory-efficient.

This document covers the Zygote model, the app process startup sequence, and the cold/warm/hot
launch states that determine perceived startup latency. See [ART & Dalvik Runtime](art_runtime.md)
for the runtime itself and [SystemServer & Core Services](system_server.md) for the services
that orchestrate launches.

## The Zygote Process

### Boot sequence to Zygote

```text
Boot ROM → Bootloader → Linux Kernel → init (init.rc)
   └─ starts "zygote" service → app_process
        └─ ZygoteInit.main()
             ├─ preloadClasses()      (~ thousands of framework classes)
             ├─ preloadResources()    (drawables, shared resource caches)
             ├─ starts SystemServer (first fork)        → see system_server.md
             └─ runSelectLoop()        (waits on socket for fork requests)
```

`init` launches Zygote via the `app_process` binary. `ZygoteInit` preloads classes listed in
`/system/etc/preloaded-classes` and common resources, then opens a Unix domain socket
(`/dev/socket/zygote`) and blocks in a select loop waiting for fork requests.

```text
# Excerpt of an init .rc entry for the 64-bit zygote
service zygote /system/bin/app_process64 -Xzygote /system/bin --zygote --start-system-server
    class main
    socket zygote stream 660 root system
    ...
```

### 32/64-bit and dual Zygotes

On 64-bit devices there are typically **two** Zygotes — `zygote64` and `zygote` (32-bit) — so
the framework can fork either ABI depending on the app's native code. The
`app_process64` / `app_process32` binaries correspond to these.

### Why fork instead of exec?

Preloading is expensive (loading + verifying + initializing framework classes). Doing it once
in Zygote and **sharing via COW** means:

- Each new app starts with the runtime and framework already warm.
- Shared, read-only preloaded pages cost memory only once across all apps.
- Launch is just `fork()` + a bit of app-specific setup — no re-init of the framework.

```bash
# Observe shared (COW) vs private memory of an app vs zygote
adb shell dumpsys meminfo com.example.app
adb shell cat /proc/$(adb shell pidof zygote64)/smaps_rollup
```

### USAP — Unspecialized App Processes (Android 10+)

To shave latency further, ART can keep a small pool of **pre-forked, unspecialized** app
processes (USAP) ready. When a launch request arrives, an idle USAP is *specialized* (assigned
the app's UID, args, etc.) instead of paying the `fork()` cost on the critical path.

```bash
adb shell getprop persist.device_config.runtime_native.usap_pool_enabled
```

## App Process Startup Sequence

When you tap an icon or an `Intent` resolves to a new process, roughly this happens:

```text
1. Launcher → ActivityManagerService (AMS) via Binder: start activity
2. AMS finds the target process is not running
3. AMS asks Zygote (over its socket) to fork a new process with the app's UID/GID/SELinux ctx
4. Zygote forks → child execs into ActivityThread.main()
5. ActivityThread:
     - creates the main (UI) Looper / message queue
     - binds to AMS (attachApplication)
     - AMS calls back: bindApplication
        → instantiates Application, calls Application.onCreate()
        → instantiates the target Activity, calls onCreate/onStart/onResume
6. First frame drawn → app is interactive
```

Key classes: `ActivityThread` (the per-process main thread), `ApplicationThread` (Binder
stub AMS calls back into), `Instrumentation` (instantiates components), `LoadedApk` (per-APK
class loader + resources).

```kotlin
// Application.onCreate runs on the main thread BEFORE any Activity.
// Keep it lean — work here delays every cold start.
class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // BAD: heavy synchronous init blocks first frame
        // GOOD: defer non-critical init (see App Startup library / background threads)
    }
}
```

### Jetpack App Startup

The `androidx.startup` library lets libraries declare initializers in a single
`ContentProvider`, avoiding the cost of many auto-initializing providers (a common hidden
cold-start tax).

```kotlin
class LoggerInitializer : Initializer<Logger> {
    override fun create(context: Context): Logger = Logger.init(context)
    override fun dependencies(): List<Class<out Initializer<*>>> = emptyList()
}
```

## Launch States: Cold, Warm, Hot

| State | Process exists? | Activity exists? | Work required | Typical target |
|-------|-----------------|------------------|---------------|----------------|
| **Cold** | No | No | Fork + Application.onCreate + Activity create + first draw | < 500 ms good, > 5 s = ANR-ish bad |
| **Warm** | Yes | No (or destroyed) | Recreate Activity, re-inflate, first draw | Faster than cold |
| **Hot** | Yes | Yes (in memory) | Bring to foreground, redraw | Near-instant |

```bash
# Measure cold start time (forces stop first)
adb shell am force-stop com.example.app
adb shell am start -W -S com.example.app/.MainActivity
# Output includes:  TotalTime: 412   WaitTime: 430   (milliseconds)

# Report fully drawn (call reportFullyDrawn() in app for accurate metric)
adb shell am start -W com.example.app/.MainActivity | grep -i drawn
```

For deeper, frame-level analysis use **Perfetto** / system tracing and Jetpack
**Macrobenchmark** (`StartupTimingMetric`), which also pairs with
[Baseline Profiles](art_runtime.md) to speed up cold start.

## Best Practices

1. **Keep `Application.onCreate()` minimal** — it's on the cold-start critical path.
2. **Lazy-init** heavy SDKs; defer to background threads or first use.
3. **Audit auto-initializing `ContentProvider`s** (analytics/crash SDKs) — consolidate with
   `androidx.startup`.
4. **Avoid blocking the main thread** during startup (disk/network I/O, large JSON parsing).
5. **Call `reportFullyDrawn()`** so the platform/Play Vitals measures true time-to-content.
6. **Ship a Baseline Profile** to AOT-compile startup paths (see [ART](art_runtime.md)).
7. **Use a themed launch background** (windowBackground) to avoid a blank/white screen.

## Resources

- [Zygote / app startup — AOSP](https://source.android.com/docs/core/runtime)
- [App startup time — Android Developers](https://developer.android.com/topic/performance/vitals/launch-time)
- [Jetpack App Startup library](https://developer.android.com/topic/libraries/app-startup)
- [Macrobenchmark](https://developer.android.com/topic/performance/benchmarking/macrobenchmark-overview)

### Related Files

- [ART & Dalvik Runtime](art_runtime.md) — the runtime Zygote preloads
- [SystemServer & Core Services](system_server.md) — AMS orchestrates process starts
- [Android Internals](internals.md) — boot process and process model
