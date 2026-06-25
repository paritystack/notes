# Testing Coroutines & Flow

## Overview

Asynchronous code — `suspend` functions, `viewModelScope` launches, and `Flow` streams — needs
deterministic tests that don't actually wait on real time. The `kotlinx-coroutines-test` library
provides a **virtual-time scheduler**, and **Turbine** makes `Flow` assertions readable. This
page builds on [Kotlin Coroutines & Flow](coroutines_flow.md) and complements the
[JUnit/MockK unit-testing](testing_junit_mockk.md) page (MockK's `coEvery`/`coVerify` stub the
suspend collaborators here). See the [testing hub](testing_android.md) for where this fits.

These are local unit tests in `src/test/`, run with `./gradlew testDebugUnitTest`.

## runTest & Virtual Time

`runTest` runs the body on a `TestScope` whose scheduler **skips delays** deterministically —
a `delay(10_000)` returns instantly while preserving ordering. **Inject dispatchers** into your
classes so tests can substitute a `TestDispatcher`.

```kotlin
@Test fun emitsLoadingThenSuccess() = runTest {
    val vm = SearchViewModel(FakeRepo(), testDispatcher)
    vm.search("a")
    advanceUntilIdle()                       // run all pending coroutines
    assertTrue(vm.state.value is UiState.Success)
}
```

Scheduler controls inside `runTest`:

| API | Effect |
|-----|--------|
| `advanceUntilIdle()` | Run everything until no more scheduled work |
| `advanceTimeBy(ms)` | Advance virtual clock by `ms`, running due tasks |
| `runCurrent()` | Run tasks scheduled at the current virtual time only |
| `currentTime` | Read the virtual clock (assert on elapsed time) |

### StandardTestDispatcher vs UnconfinedTestDispatcher

- **`StandardTestDispatcher`** queues new coroutines — nothing runs until you advance. Gives
  precise control over interleaving; you must `advanceUntilIdle()`/`runCurrent()`.
- **`UnconfinedTestDispatcher`** starts coroutines **eagerly** until their first suspension —
  convenient when you just want emissions to happen without manual advancing.

## MainDispatcherRule

`viewModelScope` uses `Dispatchers.Main`, which isn't available in plain JVM tests. A reusable
rule swaps it for a test dispatcher around each test:

```kotlin
class MainDispatcherRule(
    val dispatcher: TestDispatcher = StandardTestDispatcher()
) : TestWatcher() {
    override fun starting(desc: Description) = Dispatchers.setMain(dispatcher)
    override fun finished(desc: Description) = Dispatchers.resetMain()
}

class SearchViewModelTest {
    @get:Rule val mainRule = MainDispatcherRule()

    @Test fun loads() = runTest {
        val vm = SearchViewModel(FakeRepo())   // viewModelScope now safe
        vm.search("a")
        advanceUntilIdle()
        assertThat(vm.state.value).isInstanceOf(UiState.Success::class.java)
    }
}
```

Pass the **same dispatcher** into `runTest(mainRule.dispatcher)` if you need the rule's scheduler
and the test body to share one virtual clock.

## Turbine for Flow Assertions

Collecting a `Flow` by hand (launch, collect into a list, cancel) is fiddly. **Turbine** turns a
`Flow` into a suspending channel you `awaitItem()` from:

```kotlin
@Test fun stateTransitions() = runTest {
    vm.state.test {                          // Turbine extension on Flow
        assertEquals(UiState.Loading, awaitItem())
        vm.search("a")
        assertTrue(awaitItem() is UiState.Success)
        cancelAndConsumeRemainingEvents()
    }
}
```

Turbine API highlights:

| API | Purpose |
|-----|---------|
| `awaitItem()` | Suspend until the next emission |
| `awaitComplete()` / `awaitError()` | Assert terminal events |
| `expectNoEvents()` | Assert nothing emitted (yet) |
| `skipItems(n)` | Drop `n` emissions |
| `cancelAndConsumeRemainingEvents()` | Stop collecting, drain the rest |

This is ideal for testing `StateFlow`/`SharedFlow` from a ViewModel — assert the exact sequence
of UI states.

## Pitfalls

- **Hardcoding `Dispatchers.Main`/`IO`** instead of injecting — untestable; inject a dispatcher
  or `CoroutineDispatcher` provider.
- **Forgetting to advance** with `StandardTestDispatcher` — the coroutine never runs and the
  assertion sees the initial state.
- **Unconsumed Turbine events** — leftover emissions fail the test; consume or cancel explicitly.
- **Mixing real and virtual time** — don't combine `runTest` with `Thread.sleep`/real `Dispatchers.IO`.

## Where this connects

- [Android Testing](testing_android.md) — the testing hub
- [Kotlin Coroutines & Flow](coroutines_flow.md) — what `runTest`/Turbine exercise
- [JUnit, MockK & Truth](testing_junit_mockk.md) — `coEvery`/`coVerify` for suspend stubs
- [App Architecture](app_architecture.md) — ViewModels and `viewModelScope`
