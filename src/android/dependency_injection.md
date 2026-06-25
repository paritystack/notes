# Dependency Injection

## Overview

**Dependency injection (DI)** means objects receive their collaborators instead of constructing
them — making the [app architecture](app_architecture.md) layers testable (swap real
implementations for fakes) and decoupled. On Android the standard is **Hilt** (a compile-time
layer over **Dagger**), with **Koin** as a runtime/Kotlin-DSL alternative. This page covers the
concepts, Hilt's component/scope model, and how DI underpins testing — Hilt's test support is the
basis of the [Espresso/Hilt instrumented](testing_espresso.md) and
[JUnit/MockK](testing_junit_mockk.md) seams.

```
@Inject constructor(deps…)   ←  Hilt builds the object graph at compile time
        ▲                       binding (where each type comes from)
        │
   @Module / @Binds / @Provides
```

## Manual DI vs a Framework

You can do DI by hand — pass collaborators through constructors and wire them in one place
(`Application`). It's explicit and dependency-free, but wiring graphs and managing scopes/lifetimes
by hand gets verbose. **Hilt** generates that wiring from annotations and ties scopes to Android
lifecycles automatically.

```kotlin
// constructor injection — the foundation of all of this
class UserViewModel @Inject constructor(private val repo: UserRepository) : ViewModel()
```

## Hilt

Hilt is the recommended option. Setup: annotate the `Application`, then inject into Android
entry points.

```kotlin
@HiltAndroidApp
class App : Application()

@AndroidEntryPoint
class FeedActivity : AppCompatActivity() {
    private val vm: FeedViewModel by viewModels()   // @HiltViewModel resolved automatically
}

@HiltViewModel
class FeedViewModel @Inject constructor(private val repo: FeedRepository) : ViewModel()
```

### Providing bindings

Constructor injection (`@Inject constructor`) covers classes you own. For interfaces and types you
don't own (Retrofit, OkHttp, Room), declare a **module**:

```kotlin
@Module @InstallIn(SingletonComponent::class)
abstract class DataModule {
    @Binds abstract fun bindRepo(impl: UserRepositoryImpl): UserRepository   // interface → impl

    companion object {
        @Provides @Singleton
        fun db(@ApplicationContext ctx: Context): AppDb =                    // construct a 3rd-party type
            Room.databaseBuilder(ctx, AppDb::class.java, "app").build()
    }
}
```

- **`@Binds`** — tells Hilt which implementation satisfies an interface (no body, cheaper).
- **`@Provides`** — a factory function for types you can't constructor-inject.
- **Qualifiers** (`@Named`, custom `@Qualifier`) disambiguate two bindings of the same type (e.g.
  IO vs Default dispatcher).

### Components & Scopes

Hilt generates **components** tied to Android lifecycles; a scope annotation makes a binding a
singleton *within* that component's lifetime:

| Component | Scope | Lifetime |
|-----------|-------|----------|
| `SingletonComponent` | `@Singleton` | Whole app |
| `ActivityRetainedComponent` | `@ActivityRetainedScoped` | Survives config change (ViewModel-ish) |
| `ViewModelComponent` | `@ViewModelScoped` | One ViewModel |
| `ActivityComponent` | `@ActivityScoped` | One Activity |
| `FragmentComponent` | `@FragmentScoped` | One Fragment |

Unscoped bindings create a **new instance per injection**; scoped bindings are cached for that
component's life. Scope deliberately (over-scoping leaks; under-scoping rebuilds expensive objects).

## Koin (Alternative)

**Koin** is a pragmatic, runtime DI framework configured with a Kotlin DSL — no annotation
processing, faster builds, simpler mental model, but errors surface at **runtime** rather than
compile time (Hilt/Dagger validate the graph at build).

```kotlin
val appModule = module {
    single<UserRepository> { UserRepositoryImpl(get()) }   // singleton
    viewModel { FeedViewModel(get()) }
    factory { Analytics(get()) }                           // new each resolve
}
```

| | Hilt/Dagger | Koin |
|--|-------------|------|
| Wiring | Annotation processing, compile-time | Kotlin DSL, runtime |
| Errors | Compile-time graph validation | Runtime (`NoBeanDefFoundException`) |
| Build cost | Higher (KAPT/KSP) | Lower |
| Android lifecycle scopes | Built-in components | Manual/scope APIs |

Hilt is the default recommendation for app modules; Koin shines for smaller apps, libraries, or
KMP where annotation processors are awkward.

## DI & Testing

DI exists largely *for* testing — it lets a test substitute fakes for real dependencies:

- **Unit tests** just call the constructor with fakes/mocks — no framework needed (see
  [JUnit/MockK](testing_junit_mockk.md)). This is why constructor injection matters.
- **Instrumented tests** use Hilt's `@HiltAndroidTest` + `@UninstallModules`/`@BindValue`/
  `@TestInstallIn` to swap whole modules — covered in [Espresso & UI Automator](testing_espresso.md).

## Pitfalls

- **Over-scoping** (`@Singleton` everything) — leaks Activity/Context, retains memory.
- **Injecting `Context` carelessly** — use `@ApplicationContext` for long-lived holders, not an
  Activity context.
- **Field injection over constructor injection** — only Android entry points need field injection;
  prefer constructors elsewhere (testable, immutable).
- **Missing qualifiers** — two bindings of the same type → duplicate-binding build error.
- **Koin runtime surprises** — a missing definition fails at first resolve, not at build.

## Where this connects

- [App Architecture](app_architecture.md) — DI wires the layers and enables testable seams
- [Espresso & UI Automator](testing_espresso.md) — Hilt instrumented test support
- [JUnit, MockK & Truth](testing_junit_mockk.md) — constructor injection of fakes/mocks
- [Networking](networking.md) · [Data & Persistence](data_persistence.md) — typical provided dependencies
- [Coroutines & Flow](coroutines_flow.md) — injecting dispatchers via qualifiers
