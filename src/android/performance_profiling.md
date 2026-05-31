# Performance & Profiling

## Overview

Android performance work targets a few user-visible metrics: **startup time**, **smoothness**
(no jank — every frame within its budget), **memory** (no leaks/OOM), **battery**, and **app
size**. The discipline is: **measure first** with the right tool, fix the biggest cost, and
re-measure. This doc covers the tooling (Perfetto/system tracing, Android Studio Profiler,
Macrobenchmark) and the common levers (baseline profiles, R8, layout/overdraw, allocations).

Ties closely to [Graphics Stack](graphics_stack.md) (jank), [ART](art_runtime.md) (GC,
compilation), and [Zygote & App Startup](zygote_startup.md) (launch states).

## Key Metrics & Targets

| Metric | Tool | Rough target |
|--------|------|--------------|
| Cold start | `am start -W`, Macrobenchmark | < 500 ms good |
| Frame time | `gfxinfo`, Perfetto | ≤ 16.6 ms @60 Hz / 8.3 ms @120 Hz |
| Jank rate | Play Vitals, `gfxinfo` | < 1% janky frames |
| Memory | Memory Profiler, LeakCanary | no leaks; stable after GC |
| App size | APK Analyzer | as small as feasible |

## Tools

### Perfetto / System Tracing

The system-wide tracer; visualizes CPU scheduling, the UI thread, RenderThread, SurfaceFlinger,
Binder calls, and your own trace sections. Open captures in **ui.perfetto.dev**.

```bash
# Record a 10s trace from the command line
adb shell perfetto -o /data/misc/perfetto-traces/trace.pftrace -t 10s \
    sched freq idle am wm gfx view binder_driver
adb pull /data/misc/perfetto-traces/trace.pftrace
```

Add custom trace sections to find your own hot spots:

```kotlin
import androidx.tracing.trace
trace("loadDashboard") {
    // work shows up as a labeled slice in Perfetto
}
```

### Android Studio Profiler

Interactive CPU, Memory, Energy, and Network profilers attached to a running app:

- **CPU**: method/sampling traces, flame charts — find hot methods and main-thread stalls.
- **Memory**: allocation tracking, heap dumps — find leaks and churn.
- **Energy**: wakelocks, alarms, jobs.

### `gfxinfo` for jank

```bash
adb shell dumpsys gfxinfo com.example.app
# Reports total frames, "Janky frames", percentiles (50th/90th/95th/99th),
# and a histogram of frame durations.
adb shell dumpsys gfxinfo com.example.app reset   # reset counters before a test
```

### Macrobenchmark & Microbenchmark

Jetpack **Macrobenchmark** measures startup and scrolling/jank of a real APK; **Microbenchmark**
measures small code snippets. Macrobenchmark also **generates [baseline profiles](art_runtime.md)**.

```kotlin
@RunWith(AndroidJUnit4::class)
class StartupBenchmark {
    @get:Rule val rule = MacrobenchmarkRule()

    @Test fun startup() = rule.measureRepeated(
        packageName = "com.example.app",
        metrics = listOf(StartupTimingMetric(), FrameTimingMetric()),
        iterations = 10,
        startupMode = StartupMode.COLD,
    ) {
        pressHome(); startActivityAndWait()
    }
}
```

## Startup Performance

- **Trim `Application.onCreate()`** and audit auto-init `ContentProvider`s (see
  [Zygote & App Startup](zygote_startup.md)).
- **Ship a Baseline Profile** so startup paths are AOT-compiled on first launch.
- **Lazy-init** SDKs; defer non-critical work to after first frame / idle.
- **Use a windowBackground theme** for an instant placeholder instead of a blank screen.
- **Call `reportFullyDrawn()`** to measure true time-to-content.

## Rendering / Jank

Most jank is **work on the UI thread** missing the frame deadline (see
[Graphics Stack](graphics_stack.md)):

- Move parsing/I/O off the main thread (coroutines + `Dispatchers.IO/Default`).
- **Avoid per-frame allocations** (they trigger [GC](art_runtime.md) pauses).
- **Reduce overdraw** (Developer Options → Debug GPU Overdraw); flatten view hierarchies.
- In Compose: minimize recomposition scope, use `remember`, stable keys/params, `derivedStateOf`,
  and avoid reading state too high in the tree.

```bash
# Debug overdraw / GPU rendering visually
adb shell setprop debug.hwui.overdraw show ; adb shell stop ; adb shell start
```

## Memory

- **Leaks**: an `Activity`/`Context` retained after destruction. Use **LeakCanary** in debug
  builds; it auto-detects and gives the retention path.
- **Churn**: frequent short-lived allocations cause GC pauses — pool/reuse in hot paths.
- **Bitmaps**: decode/scale to the display size; they're the top OOM cause.

```kotlin
// LeakCanary (debug only) — just add the dependency; it self-installs.
// debugImplementation "com.squareup.leakcanary:leakcanary-android:<ver>"
```

```bash
adb shell dumpsys meminfo com.example.app    # PSS breakdown (Java/native/graphics/code)
```

## App Size: R8 / ProGuard & APK Analyzer

**R8** (the default) shrinks (removes unused code), optimizes, and obfuscates; it also shrinks
resources. Smaller DEX → smaller `.oat`, faster install.

```gradle
buildTypes {
    release {
        minifyEnabled true            // R8 code shrinking + obfuscation
        shrinkResources true          // remove unused resources
        proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
    }
}
```

Keep rules for reflection/serialization targets:

```proguard
-keep class com.example.model.** { *; }
```

Use **APK Analyzer** (Build → Analyze APK) to see what's taking space (DEX method counts,
resources, native libs). Publish **AABs** so Play ships per-device splits (see
[APK/AAB Packaging & Signing](app_signing.md)).

## Battery

- Batch network/wakeups; respect Doze/App Standby (see [Background Work](background_work.md)).
- Use **WorkManager** for deferrable work instead of wakelocks/alarms.
- Profile with the **Energy Profiler** and **Battery Historian** (`bugreport`).

```bash
adb shell dumpsys batterystats --charged com.example.app > stats.txt
# Feed bugreport into Battery Historian for a timeline view.
```

## Best Practices

1. **Measure before optimizing** — Perfetto/Profiler/Macrobenchmark, not intuition.
2. **Automate regression detection** with Macrobenchmark in CI.
3. **Ship a Baseline Profile** and enable **R8** for release.
4. **Keep the UI thread clean**; offload to coroutines with the right dispatcher.
5. **Eliminate leaks with LeakCanary**; right-size bitmaps.
6. **Watch Play Vitals** (ANRs, slow/frozen frames, excessive wakeups) for field data.
7. **Track app size** with APK Analyzer; publish AABs.

## Resources

- [App performance — Android Developers](https://developer.android.com/topic/performance)
- [Perfetto](https://perfetto.dev/docs/)
- [Macrobenchmark](https://developer.android.com/topic/performance/benchmarking/macrobenchmark-overview)
- [Shrink, obfuscate, optimize (R8)](https://developer.android.com/build/shrink-code)
- [LeakCanary](https://square.github.io/leakcanary/)

### Related Files

- [Graphics Stack](graphics_stack.md) — frame pipeline and jank diagnosis
- [ART & Dalvik Runtime](art_runtime.md) — GC, AOT/baseline profiles
- [Zygote & App Startup](zygote_startup.md) — launch states and startup cost
- [Background Work](background_work.md) — battery-friendly background execution
