# Jetpack Compose

## Overview

**Jetpack Compose** is Android's declarative UI toolkit: you describe UI as `@Composable`
functions of state, and the runtime re-runs (**recomposes**) the affected functions when that
state changes. It replaces the imperative View/XML system — instead of mutating widgets you hold
references to, you emit UI from data. This page covers the mental model (composition,
recomposition, state, side effects, layout) and where it connects to the rest of the app. It sits
alongside [Jetpack](jetpack.md) (the wider library suite), feeds the patterns in
[App Architecture](app_architecture.md), consumes [Coroutines & Flow](coroutines_flow.md) for
async state, renders through the [Graphics Stack](graphics_stack.md), and is verified by
[Compose UI Testing](testing_compose.md).

```kotlin
@Composable
fun Counter() {
    var count by remember { mutableStateOf(0) }
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally) {
        Text("Count: $count")
        Button(onClick = { count++ }) { Text("Increment") }
    }
}
```

## The Mental Model

```
   state ──────────────►  @Composable functions  ──────────►  UI tree
     ▲                         (composition)                     │
     │                                                           │
     └──────────────  events (onClick, onValueChange)  ◄─────────┘
```

UI is a **function of state**. An event updates state; the runtime recomposes the composables
that read that state; the screen reflects the new state. This **unidirectional data flow** (state
down, events up) is the core idiom — see [App Architecture](app_architecture.md).

## Composition & Recomposition

- **Composition** — the first run builds the tree of UI nodes from your composables.
- **Recomposition** — when a read state value changes, Compose re-invokes *only* the composables
  that read it, skipping the rest. It can run frequently, out of order, and in parallel, so:
  - Composables must be **side-effect-free** in their body and **idempotent**.
  - Don't rely on execution order or run-count; don't mutate external state directly in the body.
- **Stability & skipping** — Compose skips recomposing a composable if its inputs are *stable* and
  unchanged. Unstable params (e.g. a `List` vs `ImmutableList`, a lambda capturing changing state)
  defeat skipping and cause needless recompositions. Mark data `@Immutable`/`@Stable` or use
  immutable collections to keep recomposition tight.

## State

State that Compose **observes** must be a snapshot state object so reads subscribe the composable:

```kotlin
var name by remember { mutableStateOf("") }     // read → subscribe; write → recompose
```

| API | Use |
|-----|-----|
| `remember { }` | Cache a value across recompositions (not across config change/process death) |
| `rememberSaveable { }` | Survive config changes & process death (Bundle-backed) |
| `mutableStateOf(x)` | Observable holder; reads trigger recomposition |
| `derivedStateOf { }` | Computed state that only recomputes when inputs change |
| `collectAsStateWithLifecycle()` | Bridge a `StateFlow` from a ViewModel into Compose, lifecycle-aware |

**State hoisting**: keep composables stateless by lifting state to the caller (pass `value` +
`onValueChange`). Hoist app state to a [ViewModel](app_architecture.md); expose it as
`StateFlow` and collect it:

```kotlin
@Composable
fun SearchRoute(vm: SearchViewModel = hiltViewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    SearchScreen(state, onQuery = vm::search)        // stateless, testable
}
```

## Side Effects

Work that escapes the composition (start a coroutine, register/unregister, run once) goes through
**effect APIs** keyed so they restart only when their key changes:

| Effect | When |
|--------|------|
| `LaunchedEffect(key)` | Run a coroutine tied to composition; cancels on leave / key change |
| `DisposableEffect(key)` | Register + `onDispose { }` cleanup (listeners, callbacks) |
| `rememberCoroutineScope()` | A scope to launch from event callbacks (not the body) |
| `rememberUpdatedState(x)` | Capture the latest value inside a long-lived effect |
| `SideEffect { }` | Publish Compose state to non-Compose code after each successful recomposition |
| `produceState { }` | Convert non-Compose async source into observable `State` |

```kotlin
LaunchedEffect(userId) {            // restarts only when userId changes
    vm.load(userId)
}
```

Never launch coroutines or mutate external state directly in a composable body — use these.

## Layout & Modifiers

Compose layout is a single measure/place pass (no nested-measure blowups like some View layouts).
Primitives: **`Column`**, **`Row`**, **`Box`**, and **`Lazy*`** lists.

```kotlin
LazyColumn {                       // only composes visible items (RecyclerView equivalent)
    items(users, key = { it.id }) { user -> UserRow(user) }
}
```

**Modifiers** are an ordered chain — order matters (`padding().background()` differs from
`background().padding()`). They handle size, layout, drawing, and input. Add a `testTag(...)` for
stable [test](testing_compose.md) selectors. For custom layouts use the `Layout` composable or
`Modifier.layout { }`.

## Theming & Interop

- **Material 3** (`MaterialTheme`) supplies color/typography/shape via `CompositionLocal`; dynamic
  color pulls from the wallpaper on Android 12+.
- **Interop both ways**: host Compose in Views with `ComposeView`; embed a View in Compose with
  `AndroidView`. This enables incremental migration from XML.
- **Navigation**: Compose has its own nav graph — see [Navigation](navigation.md).

## Performance

Recomposition is cheap but not free; jank comes from over-recomposing or heavy work in the body.

- **Defer state reads** to the lowest composable / use lambda-based modifiers (e.g. `offset { }`)
  so animation reads don't recompose ancestors.
- **Key `Lazy` items** so the runtime reuses slots instead of recomposing the list.
- **Stable types** + `@Immutable` to keep skipping working; profile with the **Compose Compiler
  metrics/reports** and **Layout Inspector** (recomposition counts).
- **Baseline Profiles** precompile hot Compose paths — see [Performance & Profiling](performance_profiling.md)
  and [ART & Dalvik Runtime](art_runtime.md).

## Pitfalls

- **Side effects in the body** (launching coroutines, mutating vars) — use `LaunchedEffect`/event
  callbacks; the body can run many times.
- **Unstable parameters** silently disabling skipping → recomposition storms. Use immutable
  collections / `@Immutable`.
- **`remember` for persistent state** — it's lost on config change/process death; use
  `rememberSaveable` or a ViewModel.
- **Reading state too high** in the tree, recomposing large subtrees on every change.
- **Forgetting `key`** in `LazyColumn` — wrong item reuse and lost scroll/animation state.

## Where this connects

- [Jetpack](jetpack.md) — the broader AndroidX suite Compose ships in
- [App Architecture](app_architecture.md) — UDF, state hoisting, ViewModel/state holders
- [Coroutines & Flow](coroutines_flow.md) — `collectAsStateWithLifecycle`, effect coroutines
- [Navigation](navigation.md) — Compose navigation graphs
- [Compose UI Testing](testing_compose.md) — semantics-tree tests for the UI above
- [Graphics Stack](graphics_stack.md) — how composed frames reach the display
- [Performance & Profiling](performance_profiling.md) — recomposition profiling, baseline profiles
