# Screenshot & Snapshot Testing

## Overview

**Screenshot tests** render a UI component and compare it pixel-for-pixel against a stored
**golden** image, catching unintended visual regressions (spacing, colour, theming, RTL) that
behavioural tests in [Espresso](testing_espresso.md)/[Compose](testing_compose.md) miss. The
three common tools differ by where they render: **Paparazzi** (pure JVM), **Roborazzi**
(Robolectric-based), and **Compose Preview Screenshot Testing** (AndroidX, drives `@Preview`s).
See the [testing hub](testing_android.md) for the pyramid; these are most valuable for
design-system components.

## Record / Verify Workflow

All three share the same loop:

```
1. record   → render component → save golden PNG (committed to VCS)
2. verify   → render again → diff against golden
3. diff > threshold → test fails, emits a side-by-side diff image
4. intentional change → re-record, review the new golden in code review
```

The golden images live in the repo, so a reviewer sees the **visual diff** as part of the PR.

## Paparazzi (JVM, No Device)

CashApp **Paparazzi** renders Views/Composables on the JVM using layoutlib (the same engine
Android Studio's preview uses) — no emulator, fast, runs in any CI.

```kotlin
class ButtonSnapshotTest {
    @get:Rule val paparazzi = Paparazzi(deviceConfig = DeviceConfig.PIXEL_5)

    @Test fun primaryButton() {
        paparazzi.snapshot { AppTheme { PrimaryButton(text = "Buy") } }
    }
}
```

```bash
./gradlew recordPaparazziDebug   # write goldens
./gradlew verifyPaparazziDebug   # compare (CI)
```

## Roborazzi (Robolectric-based)

**Roborazzi** captures screenshots from inside [Robolectric](testing_robolectric.md) tests, so
you can screenshot the same `src/test/` tests you already run — including ones that need a fuller
framework than Paparazzi's layoutlib provides.

```kotlin
@RunWith(AndroidJUnit4::class)
class HomeRoborazziTest {
    @get:Rule val composeRule = createComposeRule()

    @Test fun home() {
        composeRule.setContent { AppTheme { HomeScreen(previewState) } }
        composeRule.onRoot().captureRoboImage()   // diffs against golden
    }
}
```

```bash
./gradlew recordRoborazziDebug
./gradlew verifyRoborazziDebug
```

## Compose Preview Screenshot Testing

The AndroidX-official tool turns your existing **`@Preview`** composables into screenshot tests —
minimal extra code, reuses previews you already write for the IDE.

```kotlin
@Preview(showBackground = true)
@Composable fun PrimaryButtonPreview() { AppTheme { PrimaryButton("Buy") } }
```

Enabled via the screenshot-test Gradle plugin/source set; `./gradlew ...validateScreenshotTest`
records/verifies. Still maturing but attractive because previews double as test fixtures.

## Choosing

| Tool | Renders on | Best for |
|------|-----------|----------|
| **Paparazzi** | JVM (layoutlib) | Fast design-system snapshots, no device in CI |
| **Roborazzi** | JVM (Robolectric) | Screenshotting existing Robolectric tests; fuller framework |
| **Compose Preview Screenshot** | Gradle plugin | Reusing `@Preview`s as the test corpus |

> Performance regressions (startup, jank) are a different axis — covered by **Macrobenchmark**,
> which lives on the [Performance & Profiling](performance_profiling.md) page, not here.

## Pitfalls

- **Non-deterministic rendering** — fonts, animations, timestamps, random data cause flaky diffs;
  pin a device config, freeze the clock, and feed fixed fixture data.
- **Cross-platform goldens** — pixels can differ by host OS/JDK; record and verify on the same
  environment (pin it in CI).
- **Golden bloat** — large PNGs in git; keep snapshots focused on components, not whole screens.
- **Rubber-stamping re-records** — a re-record hides regressions if reviewers don't inspect the diff.

## Where this connects

- [Android Testing](testing_android.md) — the testing hub
- [Compose UI Testing](testing_compose.md) — behaviour tests that complement pixel diffs
- [Robolectric](testing_robolectric.md) — the engine Roborazzi builds on
- [Performance & Profiling](performance_profiling.md) — Macrobenchmark for performance regressions
- [Jetpack](jetpack.md) — Compose `@Preview` and theming
