# App Architecture

## Overview

App architecture is how you organize an Android app's code so it stays testable,
maintainable, and resilient to the platform's lifecycle and configuration changes (rotation,
process death). Google's recommended approach centers on **separation of concerns**, a
**unidirectional data flow (UDF)**, a **single source of truth**, and clear layers: **UI →
Domain (optional) → Data**.

This builds directly on [Coroutines & Flow](coroutines_flow.md) and the
[Jetpack](jetpack.md) components (ViewModel, Room, etc.).

## Recommended Layered Architecture

```text
┌────────────────────────────── UI Layer ──────────────────────────────┐
│  UI elements (Compose / Views)  ◀── state ──  ViewModel (state holder) │
└───────────────────────────────────┬───────────────────────────────────┘
                                     │ calls
┌──────────────────────── Domain Layer (optional) ───────────────────────┐
│             Use cases / interactors (reusable business logic)           │
└───────────────────────────────────┬───────────────────────────────────┘
                                     │ calls
┌────────────────────────────── Data Layer ─────────────────────────────┐
│   Repositories (single source of truth)  →  data sources (network, DB) │
└────────────────────────────────────────────────────────────────────────┘
```

- **UI layer**: renders state; forwards user events upward. Stateless as possible.
- **Domain layer** (optional): encapsulates reusable business logic in **use cases**.
- **Data layer**: **repositories** expose data and mediate between network/DB/cache; they are
  the single source of truth.

Dependencies point **downward only** (UI depends on Data, never the reverse).

## Unidirectional Data Flow (UDF)

State flows **down** (Data → ViewModel → UI); events flow **up** (UI → ViewModel). The UI is a
function of state.

```text
        state ▼                       ▲ events
   ┌──────────────┐  observe   ┌──────────────┐  user action  ┌──────────┐
   │  Data layer  │ ─────────▶ │  ViewModel   │ ◀──────────── │    UI    │
   └──────────────┘            │ (UI state)   │ ─── render ─▶ └──────────┘
                               └──────────────┘
```

## UI Layer & State Holders

The **ViewModel** survives configuration changes and holds **UI state** (typically a
`StateFlow<UiState>`). The UI observes it and renders; it never holds business logic.

```kotlin
data class FeedUiState(
    val isLoading: Boolean = false,
    val items: List<Post> = emptyList(),
    val error: String? = null,
)

class FeedViewModel(private val getFeed: GetFeedUseCase) : ViewModel() {
    private val _state = MutableStateFlow(FeedUiState(isLoading = true))
    val state: StateFlow<FeedUiState> = _state.asStateFlow()

    init { refresh() }

    fun refresh() = viewModelScope.launch {
        _state.update { it.copy(isLoading = true, error = null) }
        runCatching { getFeed() }
            .onSuccess { posts -> _state.update { it.copy(isLoading = false, items = posts) } }
            .onFailure { e -> _state.update { it.copy(isLoading = false, error = e.message) } }
    }
}
```

```kotlin
@Composable
fun FeedScreen(vm: FeedViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    when {
        state.isLoading -> LoadingSpinner()
        state.error != null -> ErrorView(state.error) { vm.refresh() }
        else -> FeedList(state.items)
    }
}
```

### State vs Events

- **State** (`StateFlow`): the current screen contents — survives, re-emitted to new collectors.
- **One-off events** (navigation, snackbars): model as part of state consumed-flags, or a
  `SharedFlow`/Channel — avoid losing or duplicating them across recompositions/rotations.

## Domain Layer: Use Cases

A **use case** is a single, reusable piece of business logic. Optional — introduce it when
logic is shared across ViewModels or a ViewModel grows too large.

```kotlin
class GetFeedUseCase(
    private val feedRepo: FeedRepository,
    private val userRepo: UserRepository,
) {
    suspend operator fun invoke(): List<Post> {
        val blocked = userRepo.blockedUserIds()
        return feedRepo.getFeed().filterNot { it.authorId in blocked }
    }
}
```

## Data Layer: Repositories

A **repository** exposes data to the rest of the app and is the **single source of truth**,
coordinating remote and local sources and caching.

```kotlin
class FeedRepository(
    private val api: FeedApi,
    private val dao: PostDao,
) {
    // Single source of truth: UI observes the DB; network refreshes it.
    fun observeFeed(): Flow<List<Post>> = dao.observeAll()

    suspend fun refresh() {
        val remote = api.getFeed()
        dao.upsertAll(remote)   // DB emits → UI updates reactively
    }
}
```

A common robust pattern (**offline-first**): UI always observes the local DB; network writes
to the DB; failures don't blank the screen.

## Architectural Patterns

| Pattern | Idea | Notes |
|---------|------|-------|
| **MVVM** | View ↔ ViewModel ↔ Model; ViewModel exposes observable state | Google's default; pairs with StateFlow/Compose |
| **MVI** | Single immutable **state** + **intents** (events) reduced into new state | Very predictable UDF; more boilerplate |
| **Clean Architecture** | Concentric layers; dependencies point inward to domain | Good for large apps; can be over-engineering for small ones |

MVVM with a `UiState` data class and UDF effectively *is* a light MVI; choose ceremony to match
app size.

## Dependency Injection

DI (Hilt/Dagger, or manual) wires repositories/use cases into ViewModels, keeping classes
testable (inject fakes). See [Jetpack](jetpack.md) for Hilt.

```kotlin
@HiltViewModel
class FeedViewModel @Inject constructor(
    private val getFeed: GetFeedUseCase,
) : ViewModel() { /* ... */ }
```

## Modularization

Large apps split into Gradle modules to improve build times and enforce boundaries:

```text
:app                 (assembles features, navigation host)
:feature:feed        (UI + ViewModels for one feature)
:feature:profile
:core:data           (repositories)
:core:network        (API clients)
:core:database       (Room)
:core:ui / :core:designsystem
```

Feature modules depend on `:core:*`, not on each other; communicate via navigation/interfaces.

## Best Practices

1. **Single source of truth** — one layer owns each piece of data (usually the DB/repository).
2. **Drive UI from immutable state**; expose `StateFlow<UiState>`, mutate via `update {}`.
3. **Keep ViewModels free of Android UI types** (no `View`, `Context` leaks) and framework I/O.
4. **Put business logic in use cases/repositories**, not in the UI or ViewModel glue.
5. **Inject dependencies** so layers are swappable and testable.
6. **Handle process death** — persist critical state (SavedStateHandle) so the app restores.
7. **Modularize by feature** as the app grows to speed builds and enforce boundaries.

## Resources

- [Guide to app architecture — Android Developers](https://developer.android.com/topic/architecture)
- [UI layer](https://developer.android.com/topic/architecture/ui-layer)
- [Data layer](https://developer.android.com/topic/architecture/data-layer)
- [Now in Android (reference app)](https://github.com/android/nowinandroid)

### Related Files

- [Coroutines & Flow](coroutines_flow.md) — the async backbone of these layers
- [Jetpack](jetpack.md) — ViewModel, Room, Hilt building blocks
- [Navigation](navigation.md) — moving between feature screens
- [Android Testing](testing_android.md) — testing each layer in isolation
