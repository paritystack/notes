# Android Testing

## Overview

Android testing spans fast **local unit tests** (run on the JVM, no device) and slower
**instrumented tests** (run on a device/emulator with the Android framework). A healthy suite
follows the **testing pyramid**: many fast unit tests, fewer integration tests, and a small set
of end-to-end UI tests. This doc covers JUnit/MockK unit tests, coroutine/Flow testing,
Espresso and Compose UI tests, Robolectric, and where each fits.

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

## Unit Tests (JUnit)

Pure logic — ViewModels, use cases, repositories with fakes. Keep Android types out of these
classes (see [App Architecture](app_architecture.md)) so they're JVM-testable.

```kotlin
class PriceCalculatorTest {
    private val calc = PriceCalculator()

    @Test fun appliesDiscount() {
        assertEquals(90.0, calc.withDiscount(100.0, 0.1), 0.001)
    }
}
```

### Test doubles: fakes vs mocks

- **Fake**: a working lightweight implementation (e.g. in-memory repository) — preferred for
  most tests; more robust to refactors.
- **Mock**: a stand-in with programmed responses/verifications — good for interaction
  verification and hard-to-fake collaborators.

**MockK** is the idiomatic Kotlin mocking library:

```kotlin
@Test fun loadsUser() = runTest {
    val repo = mockk<UserRepository>()
    coEvery { repo.getUser(1) } returns User(1, "Ada")

    val vm = UserViewModel(repo)
    vm.load(1)

    assertEquals("Ada", vm.state.value.name)
    coVerify { repo.getUser(1) }
}
```

## Testing Coroutines & Flow

Use `kotlinx-coroutines-test`: `runTest` provides a virtual-time scheduler so delays are skipped
deterministically. **Inject dispatchers** so tests can substitute a `TestDispatcher`.

```kotlin
@Test fun emitsLoadingThenSuccess() = runTest {
    val vm = SearchViewModel(FakeRepo(), testDispatcher)
    vm.search("a")
    advanceUntilIdle()
    assertTrue(vm.state.value is UiState.Success)
}
```

### Turbine for Flow assertions

```kotlin
@Test fun stateTransitions() = runTest {
    vm.state.test {                       // Turbine
        assertEquals(UiState.Loading, awaitItem())
        vm.search("a")
        assertTrue(awaitItem() is UiState.Success)
        cancelAndConsumeRemainingEvents()
    }
}
```

A common helper is a **MainDispatcherRule** that swaps `Dispatchers.Main` for a test dispatcher
(needed because `viewModelScope` uses `Dispatchers.Main`).

```kotlin
class MainDispatcherRule(private val d: TestDispatcher = StandardTestDispatcher()) : TestWatcher() {
    override fun starting(desc: Description) = Dispatchers.setMain(d)
    override fun finished(desc: Description) = Dispatchers.resetMain()
}
```

## Robolectric

Runs Android-dependent code (Context, resources, SharedPreferences) on the **JVM** without an
emulator — fast feedback for code that touches the framework lightly.

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

## Instrumented Tests (View UI) — Espresso

Espresso drives real UI on a device/emulator: find a view, perform an action, check state.

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

For navigating across apps/system UI, use **UI Automator**.

## Compose UI Testing

Compose has its own semantics-based testing API via a compose test rule.

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

Add `testTag(...)` modifiers and query with `onNodeWithTag` for stable selectors. Use
`createAndroidComposeRule<Activity>()` when you need a real Activity.

## Hilt & Test Doubles

For DI-based apps, **Hilt testing** lets you replace modules with fakes:

```kotlin
@HiltAndroidTest
@UninstallModules(NetworkModule::class)
class FeedTest {
    @get:Rule val hiltRule = HiltAndroidRule(this)
    // provide a fake NetworkModule via @TestInstallIn / @BindValue
}
```

## Other Useful Tools

| Tool | Purpose |
|------|---------|
| **Truth** | Fluent assertions (`assertThat(x).isEqualTo(y)`) |
| **Turbine** | Testing Kotlin Flows |
| **MockWebServer** | Stub HTTP responses for Retrofit/OkHttp tests |
| **Espresso-Intents** | Verify/stub outgoing Intents |
| **Macrobenchmark** | Startup/jank performance tests (see [Performance](performance_profiling.md)) |
| **Screenshot testing** (Paparazzi / Roborazzi / Compose Preview Screenshot) | Catch UI regressions |

```kotlin
// MockWebServer
val server = MockWebServer()
server.enqueue(MockResponse().setBody("""{"id":1,"name":"Ada"}"""))
server.start()
val api = retrofit(server.url("/")).create(Api::class.java)
```

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
