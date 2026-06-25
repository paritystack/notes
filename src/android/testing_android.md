# Android Testing

## Overview

Android testing spans fast **local unit tests** (run on the JVM, no device) and slower
**instrumented tests** (run on a device/emulator with the Android framework). A healthy suite
follows the **testing pyramid**: many fast unit tests, fewer integration tests, and a small set
of end-to-end UI tests. This page is the **hub** — it covers the pyramid, where each test type
runs, CI, and best practices. Each framework has its own deep-dive page (see
[Frameworks](#frameworks-deep-dives) below).

Builds on [Coroutines & Flow](coroutines_flow.md), [App Architecture](app_architecture.md)
(testable layers), and runs via tasks from [Android Gradle Deep Dive](gradle_android.md).

## Test Types & Where They Run

| Type | Location | Speed | Runs on |
|------|----------|-------|---------|
| **Local unit** | `src/test/` | Fast (ms) | JVM (no Android) |
| **Robolectric** | `src/test/` | Fast-ish | JVM (simulated Android) |
| **Instrumented** | `src/androidTest/` | Slow | Device/emulator |
| **UI / E2E** | `src/androidTest/` | Slowest | Device/emulator |

```bash
./gradlew testDebugUnitTest        # local + Robolectric
./gradlew connectedAndroidTest     # instrumented (needs device/emulator)
```

## Frameworks (deep dives)

Each framework has its own page; this hub links them by where they sit on the pyramid.

| Page | Covers | Runs on |
|------|--------|---------|
| [JUnit, MockK & Truth](testing_junit_mockk.md) | Unit-test toolkit: JUnit, MockK mocking, Truth assertions, MockWebServer | JVM |
| [Testing Coroutines & Flow](testing_coroutines.md) | `runTest`/virtual time, `TestDispatcher`, `MainDispatcherRule`, Turbine | JVM |
| [Robolectric](testing_robolectric.md) | Android framework (Context/resources) simulated on the JVM | JVM |
| [Espresso & UI Automator](testing_espresso.md) | View UI, Espresso-Intents, cross-app UI Automator, Hilt instrumented | Device/emulator |
| [Compose UI Testing](testing_compose.md) | Semantics-tree finders/actions/assertions, `testTag`, sync | Device/emulator |
| [Screenshot & Snapshot Testing](testing_screenshot.md) | Paparazzi / Roborazzi / Compose Preview Screenshot golden diffs | JVM |

For startup/jank performance tests, see **Macrobenchmark** on
[Performance & Profiling](performance_profiling.md).

## CI

Run `testDebugUnitTest` (+ Robolectric) on every push (fast), and instrumented/UI tests on
emulators less frequently (e.g. Gradle Managed Devices or Firebase Test Lab). See
[Android Gradle Deep Dive](gradle_android.md) for the tasks.

```kotlin
// Gradle Managed Devices: spin up emulators from the build
android {
    testOptions {
        managedDevices.devices {
            create<ManagedVirtualDevice>("pixel6api34") {
                device = "Pixel 6"; apiLevel = 34; systemImageSource = "aosp"
            }
        }
    }
}
// ./gradlew pixel6api34DebugAndroidTest
```

## Best Practices

1. **Follow the pyramid** — lots of fast unit tests, few slow UI tests.
2. **Keep logic in JVM-testable classes** (no Android types in ViewModels/use cases).
3. **Prefer fakes over mocks** for repositories/data sources; use MockK for interactions.
4. **Inject dispatchers** and use `runTest` + Turbine for coroutines/Flow.
5. **Use `testTag`/stable matchers** in UI tests; avoid brittle text/position selectors.
6. **Stub the network** with MockWebServer; don't hit real servers in tests.
7. **Run unit tests on every CI push**; gate instrumented tests with managed devices/Test Lab.
8. **Add screenshot tests** for design-system components to catch visual regressions.

## Resources

- [Test apps on Android](https://developer.android.com/training/testing)
- [Test coroutines](https://developer.android.com/kotlin/coroutines/test)
- [Testing Compose](https://developer.android.com/develop/ui/compose/testing)
- [Espresso](https://developer.android.com/training/testing/espresso)
- [Robolectric](https://robolectric.org/)
- [MockK](https://mockk.io/) · [Turbine](https://github.com/cashapp/turbine)

### Related Files

- [Coroutines & Flow](coroutines_flow.md) — what `runTest`/Turbine test
- [App Architecture](app_architecture.md) — designing layers for testability
- [Android Gradle Deep Dive](gradle_android.md) — test tasks & managed devices
- [Performance & Profiling](performance_profiling.md) — Macrobenchmark performance tests

## Where this connects

- [App architecture](app_architecture.md) — testable design
- [Coroutines & Flow](coroutines_flow.md) — testing async code
- [Jetpack](jetpack.md) — AndroidX test libraries
- [Gradle deep dive](gradle_android.md) — test tasks and variants
- [adb](adb.md) — running instrumented tests on devices
- [Performance & profiling](performance_profiling.md) — benchmark/macrobenchmark
