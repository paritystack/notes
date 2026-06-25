# Espresso & UI Automator (Instrumented UI)

## Overview

**Instrumented UI tests** run on a real device/emulator with the full Android framework.
**Espresso** drives a single app's View-based UI (find a view → act → assert); **UI Automator**
reaches across apps and into system UI; **Espresso-Intents** stubs/verifies outgoing `Intent`s;
and **Hilt** testing swaps DI modules for fakes. For Compose UIs use the dedicated
[Compose testing](testing_compose.md) page; for JVM-speed framework tests see
[Robolectric](testing_robolectric.md). See the [testing hub](testing_android.md) for the pyramid.

These tests live in `src/androidTest/` and need a device/emulator:

```bash
./gradlew connectedAndroidTest
```

## Espresso

Espresso auto-synchronises with the UI thread (it waits for the message queue to idle before
acting), so tests are less flaky than naive sleeps. The core grammar is **matcher → action →
assertion**:

```kotlin
@RunWith(AndroidJUnit4::class)
class LoginUiTest {
    @get:Rule val rule = ActivityScenarioRule(LoginActivity::class.java)

    @Test fun submitShowsError() {
        onView(withId(R.id.email)).perform(typeText("bad"), closeSoftKeyboard())
        onView(withId(R.id.submit)).perform(click())
        onView(withId(R.id.error)).check(matches(isDisplayed()))
    }
}
```

| Stage | Examples |
|-------|----------|
| **Matcher** (`onView`) | `withId`, `withText`, `withHint`, `hasSibling`, `isDescendantOfA` |
| **Action** (`.perform`) | `click`, `typeText`, `clearText`, `scrollTo`, `swipeLeft`, `closeSoftKeyboard` |
| **Assertion** (`.check`) | `matches(isDisplayed())`, `matches(withText("…"))`, `doesNotExist()` |

- **`ActivityScenarioRule`** launches/closes the Activity per test; use `ActivityScenario`
  directly to drive lifecycle (`moveToState(RESUMED)`) or `onActivity { }` to reach into it.
- **RecyclerView**: use `onView(withId(...)).perform(RecyclerViewActions.actionOnItemAtPosition(...))`.
- **Idling resources**: register an `IdlingResource` so Espresso waits on your async work
  (network, custom executors) instead of `Thread.sleep`.

### Espresso-Intents

Stub and verify outgoing `Intent`s without leaving the test:

```kotlin
@get:Rule val intentsRule = IntentsRule()   // or Intents.init()/release()

@Test fun opensDetail() {
    intending(hasComponent(DetailActivity::class.java.name))
        .respondWith(ActivityResult(Activity.RESULT_OK, null))
    onView(withId(R.id.row)).perform(click())
    intended(hasComponent(DetailActivity::class.java.name))
}
```

## UI Automator

When a flow leaves your app — system permission dialogs, notifications, the launcher, a second
app — Espresso can't see it. **UI Automator** drives the whole device by accessibility tree:

```kotlin
val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
device.pressHome()
device.findObject(UiSelector().text("Allow")).click()        // system permission dialog
device.wait(Until.hasObject(By.pkg("com.example.app")), 5_000)
```

Use Espresso inside your app, UI Automator for cross-app/system steps — they compose in one test.

## Hilt Instrumented Testing

For DI-based apps, **Hilt** testing replaces production modules with test fakes:

```kotlin
@HiltAndroidTest
@UninstallModules(NetworkModule::class)
class FeedTest {
    @get:Rule(order = 0) val hiltRule = HiltAndroidRule(this)
    @get:Rule(order = 1) val activityRule = ActivityScenarioRule(FeedActivity::class.java)

    @BindValue @JvmField val api: FeedApi = FakeFeedApi()   // injected everywhere NetworkModule provided it

    @Test fun showsCachedFeed() {
        onView(withText("Top story")).check(matches(isDisplayed()))
    }
}
```

- **`@UninstallModules`** removes a production module for this test.
- **`@BindValue`** binds a test instance into the graph; **`@TestInstallIn`** swaps a whole module
  for many tests. Rule **order** matters — `HiltAndroidRule` must run first.
- Requires a `HiltTestApplication` via a custom `AndroidJUnitRunner`.

## Pitfalls

- **`Thread.sleep` for async** — flaky; use `IdlingResource` or `ActivityScenario` callbacks.
- **Brittle selectors** — text/position matchers break with copy/layout changes; prefer stable
  IDs (and `testTag` in Compose).
- **Forgetting `Intents.init()`/release()`** (use `IntentsRule`) — `intended`/`intending` no-op.
- **Animations on by default** — disable system animations on test devices to reduce flakiness.
- **Hilt rule order** — `HiltAndroidRule` before the Activity rule, or injection fails.

## Where this connects

- [Android Testing](testing_android.md) — the testing hub
- [Compose UI Testing](testing_compose.md) — the Compose equivalent of Espresso
- [Robolectric](testing_robolectric.md) — JVM-speed framework tests when full fidelity isn't needed
- [adb](adb.md) — installing and running instrumented tests on a device
- [App Architecture](app_architecture.md) — Hilt DI and testable seams
