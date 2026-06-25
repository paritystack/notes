# Compose UI Testing

## Overview

Jetpack Compose has its own testing API built on the **semantics tree** — the accessibility-like
description of the UI — rather than View IDs. You find nodes by their semantics (text, tag, role),
act on them, and assert. It's the Compose counterpart to [Espresso](testing_espresso.md) and,
like it, runs instrumented on a device/emulator (or on the JVM via Robolectric for some setups).
See the [testing hub](testing_android.md) for the pyramid, and [Jetpack](jetpack.md) for Compose
itself.

These tests typically live in `src/androidTest/`:

```bash
./gradlew connectedAndroidTest
```

## The Compose Test Rule

```kotlin
@RunWith(AndroidJUnit4::class)
class CounterTest {
    @get:Rule val composeRule = createComposeRule()

    @Test fun incrementsCount() {
        composeRule.setContent { Counter() }
        composeRule.onNodeWithText("Count: 0").assertIsDisplayed()
        composeRule.onNodeWithText("Increment").performClick()
        composeRule.onNodeWithText("Count: 1").assertIsDisplayed()
    }
}
```

- **`createComposeRule()`** hosts composables in a blank Activity — best for isolated component
  tests; you call `setContent { }` yourself.
- **`createAndroidComposeRule<MyActivity>()`** runs against a real Activity — use when you need
  the Activity's content, DI graph, navigation, or intent extras.

## Finding, Acting, Asserting

| Stage | Examples |
|-------|----------|
| **Finder** | `onNodeWithText`, `onNodeWithTag`, `onNodeWithContentDescription`, `onNode(matcher)`, `onAllNodesWithText(...)` |
| **Action** | `performClick`, `performTextInput`, `performScrollTo`, `performTouchInput { swipeUp() }` |
| **Assertion** | `assertIsDisplayed`, `assertExists`, `assertTextEquals`, `assertIsEnabled`, `assertDoesNotExist` |

Add a **`testTag`** modifier for stable selectors that survive copy/layout changes:

```kotlin
Button(onClick = onIncrement, modifier = Modifier.testTag("increment")) { Text("Increment") }

composeRule.onNodeWithTag("increment").performClick()
```

Use matcher combinators for precise queries: `hasText("Save") and hasClickAction()`,
`onNodeWithText("Item").onChildren()`, `hasTestTag(...)`.

## Synchronisation

The rule **auto-syncs** with Compose recomposition and the Espresso idling pipeline, so you
rarely sleep. When you drive virtual time (animations) or non-Compose async work:

```kotlin
composeRule.mainClock.autoAdvance = false
composeRule.mainClock.advanceTimeBy(500)        // step animations deterministically

composeRule.waitUntil(timeoutMillis = 5_000) {  // wait on a condition
    composeRule.onAllNodesWithText("Loaded").fetchSemanticsNodes().size == 1
}
```

- **`mainClock`** controls the Compose test clock — pause `autoAdvance` to assert mid-animation.
- **`waitForIdle()`** / **`waitUntil { }`** for explicit synchronisation points.
- **`printToLog("TAG")`** on a node dumps the semantics tree — invaluable when a finder matches
  nothing.

## Interop with Views & Espresso

Compose tests and Espresso coexist: in a hybrid screen, query Views with `onView(...)` and
composables with `composeRule.onNode(...)` in the same test. `createAndroidComposeRule` wires the
Espresso idling resource for the Compose hierarchy automatically.

## Pitfalls

- **Asserting on raw text that's localized/dynamic** — prefer `testTag` for stable selectors.
- **Ambiguous finders** — `onNodeWithText` throws if multiple match; use `onAllNodes...` or add
  a tag.
- **Animations never settling** — pause `mainClock.autoAdvance` and advance manually, or the rule
  may wait forever for idle.
- **Merged semantics** — parent nodes merge children's semantics; use
  `useUnmergedTree = true` on the finder to reach a specific child.

## Where this connects

- [Android Testing](testing_android.md) — the testing hub
- [Espresso & UI Automator](testing_espresso.md) — the View-based equivalent (interops in hybrid UIs)
- [Jetpack](jetpack.md) — Compose runtime and tooling
- [Screenshot & Snapshot Testing](testing_screenshot.md) — pixel-level Compose regression tests
- [Navigation](navigation.md) — testing Compose navigation graphs
