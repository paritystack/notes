# ART & Dalvik Runtime

## Overview

The Android Runtime (ART) is the managed runtime that executes application code on
Android. It replaced the older Dalvik VM as the default runtime in Android 5.0 (Lollipop).
Both run **DEX bytecode** (Dalvik Executable) rather than standard Java `.class` files,
but they differ fundamentally in *how* and *when* that bytecode is turned into native
machine code.

- **Dalvik** (≤ Android 4.4): a register-based VM that used **JIT** (Just-In-Time)
  compilation — bytecode was interpreted and hot paths compiled to native code at runtime,
  on every launch.
- **ART** (≥ Android 5.0): initially used **AOT** (Ahead-Of-Time) compilation at install
  time, then evolved (Android 7.0+) into a **hybrid** interpreter + JIT + AOT model driven
  by **profiles**.

See also [Android Internals](internals.md) for how ART fits into the boot process and
[Zygote & App Startup](zygote_startup.md) for how the runtime is shared across apps.

## DEX: The Dalvik Executable Format

Java/Kotlin source compiles to `.class` files, which the **d8/r8** toolchain converts into
a single `classes.dex`. DEX is **register-based** (Dalvik VM registers) rather than
stack-based like the JVM, which yields more compact, fewer instructions per operation.

```text
.java / .kt
   │  javac / kotlinc
   ▼
.class (JVM stack-based bytecode)
   │  d8 (dexer)  /  r8 (dexer + optimizer + shrinker)
   ▼
classes.dex (Dalvik register-based bytecode)
   │  packaged into APK, then optimized on-device
   ▼
.odex / .vdex / .art  (optimized artifacts in /data/dalvik-cache or app's oat dir)
```

### The 64K method reference limit & multidex

A single DEX file can reference at most **65,536 (64K) methods** (the `method_ids` index is
16-bit). Large apps exceed this and need **multidex** (`classes.dex`, `classes2.dex`, …).
On `minSdk >= 21` (ART) multidex is native; below that the `multidex` support library is
required.

```gradle
android {
    defaultConfig {
        multiDexEnabled true   // only needed for minSdk < 21
    }
}
```

## Compilation Strategies

### Dalvik JIT (legacy)

Bytecode was interpreted; a profiler identified "hot" traces and compiled them to native
code in memory. Fast install, but slower steady-state and the JIT work repeated every run.

### ART AOT (Android 5.0–6.0)

At **install time**, `dex2oat` compiled the entire app AOT into an `.oat` ELF file. Result:
fast, consistent runtime performance, but **slow installs/updates** and large storage use
(every method compiled, even rarely-used ones).

### Profile-Guided Hybrid Compilation (Android 7.0+)

ART now combines three execution modes:

| Mode | When | Cost |
|------|------|------|
| **Interpreter** | First time code runs | No compile cost, slowest execution |
| **JIT** | Hot methods detected at runtime | Compiled in-memory, fast; profile recorded |
| **AOT** | Methods in the profile, compiled in background (idle/charging) | Persisted to `.oat`, fastest |

The JIT records a **profile** (`/data/misc/profiles/...`) of which methods are hot. When the
device is idle and charging, `dex2oat` AOT-compiles only those profiled methods. This gives
fast installs *and* near-AOT steady-state performance.

```bash
# Force compile an installed app with a given filter
adb shell cmd package compile -m speed-profile -f com.example.app
adb shell cmd package compile -m speed        -f com.example.app   # full AOT
adb shell cmd package compile -m verify       -f com.example.app   # verify only (no AOT)

# Reset/clear compiled artifacts
adb shell cmd package compile --reset com.example.app
```

Common **compiler filters**: `verify`, `speed-profile` (default for most apps),
`speed` (full AOT), `everything`.

### On-device artifacts

| File | Purpose |
|------|---------|
| `.vdex` | Verified DEX — pre-verified bytecode, lets ART skip re-verification |
| `.odex` / `.oat` | AOT-compiled native code (ELF) |
| `.art` | Pre-initialized heap image of compiled classes/objects for fast startup |

## Baseline Profiles

**Baseline Profiles** (Android 9+, tooling via Jetpack Macrobenchmark) let *developers* ship
a profile of critical startup/interaction paths inside the APK/AAB
(`assets/dexopt/baseline.prof`). At install time ART AOT-compiles those methods immediately,
so the very first launch is fast — without waiting for runtime JIT profiling to accumulate.

```kotlin
// Generated with the Baseline Profile Gradle plugin + Macrobenchmark
// Captures the methods executed during a representative user journey.
@RunWith(AndroidJUnit4::class)
class StartupBaselineProfile {
    @get:Rule val rule = BaselineProfileRule()

    @Test fun generate() = rule.collect("com.example.app") {
        startActivityAndWait()
        // scroll, navigate to key screens...
    }
}
```

**Cloud Profiles**: Google Play also aggregates anonymized profiles from real users and
distributes them, so even apps without a baseline profile benefit over time.

## Garbage Collection

ART's GC has evolved significantly over Dalvik's stop-the-world mark-and-sweep:

- **Concurrent Copying (CC) collector** — default since Android 8.0. A generational,
  mostly-concurrent collector that copies live objects (compaction), reducing fragmentation
  and shortening pause times.
- **Generational hypothesis**: most objects die young, so ART collects a small young
  generation frequently and the older generation rarely.
- **Large Object Space (LOS)** for big allocations (e.g. bitmaps).
- Pause times target sub-millisecond for typical collections; long pauses cause **jank**.

```bash
# Watch GC activity for an app
adb shell logcat | grep -i "art .*GC"
# Example line:
#  I art: Background concurrent copying GC freed 12345(2MB) ... paused 0.5ms total 30ms
```

Tips: avoid allocations in hot paths (e.g. per-frame in `onDraw`), reuse objects/pools,
prefer primitive collections, and watch bitmap memory (a common source of LOS churn and
OOM). Use the Android Studio **Memory Profiler** and **LeakCanary** to find leaks.

## How ART Is Launched

ART is not started per-app. The **Zygote** process initializes the runtime once, preloads
common framework classes/resources, and **forks** for each new app — so all apps share
copy-on-write runtime memory. The binary `/system/bin/app_process` is the entry point.
See [Zygote & App Startup](zygote_startup.md).

```bash
# Inspect the runtime / ISA / compiler state of a device
adb shell getprop ro.dalvik.vm.native.bridge
adb shell getprop dalvik.vm.isa.arm64.variant
adb shell getprop pm.dexopt.bg-dexopt        # filter used by background dexopt
```

## Best Practices

1. **Provide a Baseline Profile** for any app where startup/scroll latency matters.
2. **Keep methods small and hot paths allocation-free** — helps both JIT and GC.
3. **Enable R8** (`minifyEnabled true`) to shrink/optimize DEX; fewer methods, smaller `.oat`.
4. **Avoid reflection** on hot paths — it defeats AOT optimization and inlining.
5. **Profile, don't guess** — use Macrobenchmark for startup and the Memory Profiler for GC.
6. **Don't hold large bitmaps** longer than needed; recycle/scale to display size.
7. **Test on `speed-profile`**, not just `speed`, to reflect real install behavior.

## Resources

- [ART and Dalvik — AOSP](https://source.android.com/docs/core/runtime)
- [Configure ART](https://source.android.com/docs/core/runtime/configure)
- [Baseline Profiles](https://developer.android.com/topic/performance/baselineprofiles/overview)
- [DEX format reference](https://source.android.com/docs/core/runtime/dex-format)

### Related Files

- [Android Internals](internals.md) — boot, memory, process model
- [Zygote & App Startup](zygote_startup.md) — how the runtime is shared and forked
- [SystemServer & Core Services](system_server.md) — the system_server runs on ART too

## Where this connects

- [Zygote & app startup](zygote_startup.md) — forks the preloaded ART runtime
- [NDK & JNI](ndk_jni.md) — the native/managed boundary ART crosses
- [System server](system_server.md) — a long-lived ART process hosting services
- [Performance & profiling](performance_profiling.md) — AOT/JIT and GC tuning
- [Internals](internals.md) — where ART sits in the platform
- [Mainline & APEX](mainline_apex.md) — ART ships as a Mainline module
