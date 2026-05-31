# Kotlin Coroutines & Flow

## Overview

**Coroutines** are Kotlin's solution for asynchronous, non-blocking code that reads like
sequential code. On Android they replace callback hell, `AsyncTask`, and manual thread
management for things like network calls, database access, and any long-running work that
must stay off the main (UI) thread. **Flow** builds on coroutines to model **streams** of
asynchronous values (e.g. UI state over time, database observations, location updates).

This pairs with [Jetpack](jetpack.md) (ViewModel/lifecycle scopes) and underpins modern
[App Architecture](app_architecture.md).

## Suspend Functions

A `suspend` function can pause without blocking the thread it runs on, and resume later. It can
only be called from a coroutine or another suspend function.

```kotlin
suspend fun fetchUser(id: Int): User {
    // suspends here; the thread is free to do other work while waiting
    return apiService.getUser(id)
}
```

## Structured Concurrency

Coroutines are launched in a **CoroutineScope** and form a parent/child hierarchy. Cancelling
a scope cancels all its children; a failure propagates predictably. This **structured
concurrency** prevents leaked coroutines — the bane of ad-hoc threading.

```kotlin
viewModelScope.launch {          // scope tied to ViewModel lifecycle
    val user = fetchUser(1)      // child coroutine work
    _state.value = user
}   // if the ViewModel is cleared, this coroutine is cancelled automatically
```

### Builders

| Builder | Returns | Use |
|---------|---------|-----|
| `launch` | `Job` | Fire-and-forget work; no result |
| `async` | `Deferred<T>` | Concurrent work that returns a value (`.await()`) |
| `runBlocking` | `T` | Blocks the thread — tests/`main()` only, never on UI thread |
| `coroutineScope { }` | `T` | Suspends until all children finish; propagates failures |
| `supervisorScope { }` | `T` | Like above but a child failure doesn't cancel siblings |

```kotlin
// Run two requests concurrently and combine
suspend fun loadDashboard(): Dashboard = coroutineScope {
    val user = async { fetchUser(1) }
    val feed = async { fetchFeed() }
    Dashboard(user.await(), feed.await())   // both run in parallel
}
```

## Dispatchers

A **Dispatcher** decides which thread(s) a coroutine runs on. Switch with `withContext`.

| Dispatcher | For |
|------------|-----|
| `Dispatchers.Main` | UI work, updating views/state (Android main thread) |
| `Dispatchers.IO` | Network, disk, database (offloaded to a large thread pool) |
| `Dispatchers.Default` | CPU-bound work (sorting, parsing, JSON, image processing) |
| `Dispatchers.Main.immediate` | Avoids re-dispatch if already on main |

```kotlin
suspend fun loadAndParse(): Report = withContext(Dispatchers.IO) {
    val raw = file.readText()                 // IO thread
    withContext(Dispatchers.Default) {        // CPU thread
        parseReport(raw)
    }
}
```

**Main-safety**: a suspend function should be safe to call from the main thread — it should
internally `withContext` to the right dispatcher rather than forcing callers to know.

## Cancellation & Exceptions

- Cancellation is **cooperative**: suspend functions check for cancellation; CPU loops should
  call `ensureActive()` or `yield()` so they can be cancelled.
- Clean up with `try/finally`; use `withContext(NonCancellable)` for must-run cleanup.
- Use a **`CoroutineExceptionHandler`** or `try/catch`; `supervisorScope` to isolate failures.

```kotlin
val job = scope.launch {
    try {
        while (isActive) { doChunk(); yield() }   // cancellable loop
    } finally {
        withContext(NonCancellable) { release() } // guaranteed cleanup
    }
}
job.cancel()
```

## Flow: Cold Asynchronous Streams

A `Flow<T>` emits a sequence of values over time. It is **cold** — the producer block runs
fresh for each collector and only when collected.

```kotlin
fun searchResults(query: String): Flow<List<Item>> = flow {
    emit(emptyList())                  // initial
    emit(repository.search(query))     // results
}.flowOn(Dispatchers.IO)               // upstream runs on IO

// Collect (suspends)
lifecycleScope.launch {
    searchResults("android").collect { items -> render(items) }
}
```

### Operators

```kotlin
flow
    .map { it.toUiModel() }
    .filter { it.visible }
    .debounce(300)              // ignore rapid emissions (e.g. search-as-you-type)
    .distinctUntilChanged()
    .catch { e -> emit(ErrorState) }   // handle upstream errors
    .collect { render(it) }
```

### StateFlow & SharedFlow (hot streams)

For UI state, you want **hot**, observable holders:

| Type | Semantics | Typical use |
|------|-----------|-------------|
| `StateFlow<T>` | Always has a current value; conflated; emits latest to new collectors | UI **state** (replaces `LiveData`) |
| `SharedFlow<T>` | Configurable replay/buffer; no required initial value | One-off **events** (navigation, snackbars) |

```kotlin
class SearchViewModel(private val repo: Repo) : ViewModel() {
    private val _state = MutableStateFlow(UiState.Loading)
    val state: StateFlow<UiState> = _state.asStateFlow()

    fun search(q: String) {
        viewModelScope.launch {
            _state.value = UiState.Loading
            _state.value = runCatching { repo.search(q) }
                .fold({ UiState.Success(it) }, { UiState.Error(it.message) })
        }
    }
}
```

## Lifecycle-Aware Collection

Collecting a Flow in the UI must stop when the UI is not visible to avoid wasted work and
crashes. Use **`repeatOnLifecycle`** (or `flowWithLifecycle`):

```kotlin
// In a Fragment/Activity
lifecycleScope.launch {
    repeatOnLifecycle(Lifecycle.State.STARTED) {
        viewModel.state.collect { state -> render(state) }   // only collects while STARTED
    }
}
```

In **Compose**, use `collectAsStateWithLifecycle()`:

```kotlin
@Composable
fun SearchScreen(vm: SearchViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    // recomposes on new state, lifecycle-aware
}
```

## Common Patterns

- **Room/DataStore return Flows** — observe DB changes reactively.
- **`stateIn` / `shareIn`** convert a cold Flow into a hot StateFlow/SharedFlow scoped to the
  ViewModel (with `SharingStarted.WhileSubscribed(5000)` to stop work when no collectors).
- **`callbackFlow`** wraps callback-based APIs (e.g. location listeners) into a Flow.

```kotlin
val items: StateFlow<List<Item>> = repo.observeItems()
    .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())
```

## Best Practices

1. **Use the right scope** (`viewModelScope`, `lifecycleScope`) — never `GlobalScope`.
2. **Keep suspend functions main-safe** with `withContext` internally.
3. **Expose `StateFlow` for state, `SharedFlow` for events**; keep mutable versions private.
4. **Collect with `repeatOnLifecycle` / `collectAsStateWithLifecycle`** in the UI.
5. **Make CPU loops cancellable** (`ensureActive`/`yield`).
6. **Pick dispatchers deliberately**: IO for I/O, Default for CPU, Main for UI.
7. **Inject dispatchers** (don't hardcode) so code is testable; use `runTest` + `TestDispatcher`.

## Resources

- [Coroutines on Android](https://developer.android.com/kotlin/coroutines)
- [Kotlin Flow](https://developer.android.com/kotlin/flow)
- [StateFlow and SharedFlow](https://developer.android.com/kotlin/flow/stateflow-and-sharedflow)
- [Testing coroutines](https://developer.android.com/kotlin/coroutines/test)

### Related Files

- [App Architecture](app_architecture.md) — coroutines/Flow in MVVM/MVI
- [Jetpack](jetpack.md) — ViewModel, lifecycle scopes
- [Android Testing](testing_android.md) — testing coroutines/Flow (Turbine, `runTest`)
- [Background Work](background_work.md) — coroutines vs WorkManager for deferred work
