# JUnit, MockK & Truth (Unit Tests)

## Overview

The foundation of the [testing pyramid](testing_android.md) is fast **local unit tests** that
run on the JVM with no device — pure logic in ViewModels, use cases, and repositories. This page
covers the unit-testing toolkit: **JUnit** as the test runner, **MockK** for Kotlin-idiomatic
mocking, **Truth** for fluent assertions, and **MockWebServer** for stubbing HTTP in
repository/network tests. For async code use the dedicated
[coroutine/Flow testing](testing_coroutines.md) page; for code that touches the Android
framework see [Robolectric](testing_robolectric.md). Design layers to be JVM-testable per
[App Architecture](app_architecture.md).

These tests live in `src/test/` and run via:

```bash
./gradlew testDebugUnitTest
```

## JUnit Structure

Pure logic — keep Android types out of these classes (see
[App Architecture](app_architecture.md)) so they stay JVM-testable.

```kotlin
class PriceCalculatorTest {
    private val calc = PriceCalculator()

    @Test fun appliesDiscount() {
        assertEquals(90.0, calc.withDiscount(100.0, 0.1), 0.001)
    }
}
```

Common JUnit 4 building blocks:

| Annotation | Purpose |
|------------|---------|
| `@Test` | Marks a test method |
| `@Before` / `@After` | Per-test setup / teardown |
| `@BeforeClass` / `@AfterClass` | Once-per-class (static) setup / teardown |
| `@get:Rule` | Attach a `TestRule` (e.g. `MainDispatcherRule`, `TemporaryFolder`) |
| `@Ignore` | Skip a test (note why) |

> JUnit 5 (Jupiter) exists, but the Android tooling/AGP defaults target **JUnit 4**;
> instrumented runners (`AndroidJUnit4`) are JUnit 4 based. Most Android codebases stay on 4.

## Test Doubles: Fakes vs Mocks

- **Fake**: a working lightweight implementation (e.g. an in-memory repository) — preferred for
  most tests; more robust to refactors because it has real behaviour.
- **Mock**: a stand-in with programmed responses/verifications — good for interaction
  verification and hard-to-fake collaborators.

```kotlin
// Fake: real behaviour, in-memory
class FakeUserRepository : UserRepository {
    private val users = mutableMapOf<Int, User>()
    fun seed(user: User) { users[user.id] = user }
    override suspend fun getUser(id: Int): User = users.getValue(id)
}
```

Prefer fakes for repositories/data sources; reach for mocks when you need to **verify
interactions** or stub an awkward collaborator.

## MockK

**MockK** is the idiomatic Kotlin mocking library — it understands coroutines, final classes
(Kotlin classes are final by default), and `object`/extension functions.

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

Key MockK building blocks:

| API | Use |
|-----|-----|
| `mockk<T>()` | Strict mock — unstubbed calls throw |
| `mockk<T>(relaxed = true)` | Returns sensible defaults for unstubbed calls |
| `every { … } returns …` | Stub a regular function |
| `coEvery { … } returns …` | Stub a `suspend` function |
| `verify { … }` / `coVerify { … }` | Assert a (suspend) call happened |
| `verify(exactly = n) { … }` | Assert call count |
| `slot<T>()` + `capture(slot)` | Capture an argument for assertions |
| `spyk(obj)` | Partial mock wrapping a real object |

```kotlin
val slot = slot<User>()
every { repo.save(capture(slot)) } returns Unit
vm.saveProfile("Ada")
assertEquals("Ada", slot.captured.name)   // inspect captured arg
```

> Use `relaxed = true` sparingly — strict mocks catch unexpected calls. Prefer `confirmVerified`
> at the end of interaction-heavy tests to ensure no calls went unverified.

## Truth Assertions

Google's **Truth** gives fluent, readable failure messages:

```kotlin
import com.google.common.truth.Truth.assertThat

assertThat(user.name).isEqualTo("Ada")
assertThat(list).containsExactly("a", "b").inOrder()
assertThat(map).containsEntry("id", 1)
assertThat(result).isInstanceOf(UiState.Success::class.java)
```

It reads better than raw JUnit `assertEquals` and produces richer diffs for collections. Kotlin
projects sometimes use Kotest assertions or AssertJ instead — pick one and stay consistent.

## MockWebServer (Network Stubbing)

For repository tests that go through Retrofit/OkHttp, **MockWebServer** stands up a real local
HTTP server you enqueue canned responses on — no real network, deterministic.

```kotlin
val server = MockWebServer()
server.enqueue(MockResponse().setBody("""{"id":1,"name":"Ada"}"""))
server.start()

val api = retrofit(server.url("/")).create(Api::class.java)
val user = api.getUser(1)

assertThat(user.name).isEqualTo("Ada")
server.shutdown()
```

You can also assert the **outgoing request** (`server.takeRequest().path`), enqueue error codes
to test failure paths, and add delays to exercise timeouts. This keeps network-layer tests on the
fast JVM unit path.

## Pitfalls

- **Mocking what you own when a fake is easy** — fakes survive refactors; over-mocking couples
  tests to implementation detail.
- **Forgetting `coEvery`/`coVerify`** for suspend functions — `every`/`verify` won't match them.
- **Leaking MockWebServer** — always `shutdown()` (ideally in `@After`) or ports leak across tests.
- **Asserting on mock internals** instead of observable behaviour.

## Where this connects

- [Android Testing](testing_android.md) — the hub: pyramid, where tests run, CI
- [Testing Coroutines & Flow](testing_coroutines.md) — `runTest` for the suspend cases above
- [App Architecture](app_architecture.md) — designing JVM-testable layers
- [Robolectric](testing_robolectric.md) — when a unit test needs the Android framework
