# Robolectric

## Overview

**Robolectric** runs Android-framework-dependent code (`Context`, resources,
`SharedPreferences`, `Looper`, `Handler`) on the **JVM** without an emulator. It sits between
fast [JUnit unit tests](testing_junit_mockk.md) and slow [Espresso instrumented tests](testing_espresso.md):
fast feedback for code that touches the framework lightly. It's also the engine behind
Roborazzi [screenshot tests](testing_screenshot.md). See the [testing hub](testing_android.md)
for where it fits on the pyramid.

Robolectric tests live in `src/test/` and run on the fast unit path:

```bash
./gradlew testDebugUnitTest        # includes Robolectric
```

## How It Works

Robolectric provides a **sandbox**: it loads the Android SDK classes and replaces their native
behaviour with **shadow** objects that emulate the framework in pure Java/Kotlin. Your code calls
the real Android APIs; shadows back them. This avoids the round-trip to a device while still
exercising resource loading, `Context`, view inflation, lifecycle, etc.

```kotlin
@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class FormatterTest {
    @Test fun usesResources() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        assertEquals("Hello", ctx.getString(R.string.greeting))
    }
}
```

- **`@RunWith(AndroidJUnit4::class)`** — the same unified runner used by instrumented tests, so
  a test can often run on JVM (Robolectric) *or* device with the same source.
- **`@Config`** — configure the simulated environment: `sdk = [..]` (API level(s)),
  `application = ..`, `qualifiers = "ru-port"` for locale/orientation, `manifest = ..`.
- **`ApplicationProvider.getApplicationContext()`** — the AndroidX way to get a `Context`
  (works on both Robolectric and device).

## Shadows

A **shadow** intercepts calls to a framework class. Most are built in; you can grab one to
inspect or drive state the real API hides:

```kotlin
val shadowApp = Shadows.shadowOf(context as Application)
// e.g. assert a started service / broadcast, advance the main Looper, etc.
ShadowLooper.idleMainLooper()         // run posted main-thread work
```

You can register **custom shadows** via `@Config(shadows = [MyShadow::class])` when you need to
fake a class Robolectric doesn't cover.

## When to Use It

| Prefer Robolectric | Prefer real instrumented ([Espresso](testing_espresso.md)) |
|--------------------|------------------------------------------------------------|
| Resource/`Context`/`SharedPreferences` logic | Real rendering, animations, GPU |
| Lifecycle unit tests, simple view inflation | Cross-app flows, system UI (UI Automator) |
| Fast CI feedback on every push | High-fidelity end-to-end confidence |

The trade-off is **fidelity**: shadows approximate the framework, so behaviour can diverge from
a real device. Use Robolectric for breadth and speed; keep a thin layer of true instrumented
tests for confidence.

## Pitfalls

- **Treating it as ground truth** — shadows are approximations; a Robolectric pass doesn't
  guarantee device behaviour. Back critical paths with instrumented tests.
- **Slow first run** — Robolectric downloads SDK jars on first execution; cache them in CI.
- **SDK-level surprises** — pin `@Config(sdk = [..])`; default may differ from your `targetSdk`.
- **Main-looper work not running** — drive it with `ShadowLooper`/`idleMainLooper()`.

## Where this connects

- [Android Testing](testing_android.md) — the testing hub
- [JUnit, MockK & Truth](testing_junit_mockk.md) — the JVM unit-test foundation it extends
- [Espresso & UI Automator](testing_espresso.md) — the real-device alternative
- [Screenshot & Snapshot Testing](testing_screenshot.md) — Roborazzi builds on Robolectric
- [Jetpack](jetpack.md) — AndroidX `test`/`core` libraries (`ApplicationProvider`)
