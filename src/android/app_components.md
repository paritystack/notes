# App Components & Lifecycle

## Overview

An Android app isn't a single `main()` — it's a set of **components** the system can start
independently: **Activities** (screens), **Services** (background work), **BroadcastReceivers**
(event subscribers), and **ContentProviders** (shared data). They're declared in the manifest and
glued together by **Intents**. This page covers each component, the Activity lifecycle, and the
Intent system, and connects to [App Architecture](app_architecture.md) (how to keep logic *out* of
components), [Navigation](navigation.md), [Background Execution](background_work.md),
[Permissions & Privacy](permissions_privacy.md), and the [Binder](binder.md) IPC that the system
uses to start them.

```
              ┌──────────── Intent (explicit / implicit) ────────────┐
   System  ───┤  Activity      Service      Receiver      Provider    │
 (AMS in     └──────────────────────────────────────────────────────┘
  system_server)   each runs in the app process forked from zygote
```

## The Four Components

| Component | What it is | Entry point | Declared as |
|-----------|------------|-------------|-------------|
| **Activity** | A single screen / UI entry | `onCreate()` | `<activity>` |
| **Service** | Long-running / background work, no UI | `onStartCommand` / `onBind` | `<service>` |
| **BroadcastReceiver** | Responds to system/app events | `onReceive()` | `<receiver>` or registered in code |
| **ContentProvider** | Exposes structured data to other apps | `onCreate` + CRUD | `<provider>` |

Each is a separate entry point the OS (the ActivityManager in
[system_server](system_server.md)) can launch — even into a cold process it forks from
[zygote](zygote_startup.md).

## Activity Lifecycle

The system drives an Activity through callbacks as it becomes visible, focused, hidden, and
destroyed. Put setup/teardown in the matching pair; never block these callbacks (they run on the
main thread).

```
   onCreate ─► onStart ─► onResume ─►  [RUNNING]
      ▲           ▲          │ (lose focus)
      │           │          ▼
      │       onRestart   onPause ─► onStop ─► onDestroy
      │           ▲          │           │
      └───────────┴──────────┴───────────┘ (back to foreground / recreated)
```

- **Config changes** (rotation, locale) destroy & recreate the Activity by default; a
  [ViewModel](app_architecture.md) survives recreation, and `onSaveInstanceState`/
  `rememberSaveable` preserve transient UI state across process death.
- **`ComponentActivity`** (base of `AppCompatActivity` and `ComponentActivity` for Compose) hosts
  the lifecycle, the `ViewModelStore`, and the Activity Result registry. Add `@AndroidEntryPoint`
  for [DI](dependency_injection.md).
- Keep logic in ViewModels/use cases (see [App Architecture](app_architecture.md)) so the Activity
  is a thin lifecycle shell.

## Intents

An **Intent** is a message describing an operation. **Explicit** intents name a target class;
**implicit** intents declare an action and let the system resolve a handler via `<intent-filter>`.

```kotlin
// explicit
startActivity(Intent(this, DetailActivity::class.java).putExtra("id", 42))

// implicit — any app that can view a URL
startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://example.com")))
```

- **Activity Result API** is the modern way to start-for-result and request permissions — register
  a contract, launch, receive the result in a callback:

  ```kotlin
  val pickImage = registerForActivityResult(GetContent()) { uri -> /* … */ }
  pickImage.launch("image/*")
  ```

- **`PendingIntent`** hands a pre-built intent to another process (notifications, alarms) to fire
  *as your app*. On Android 12+ you must set `FLAG_IMMUTABLE` or `FLAG_MUTABLE` explicitly.
- For deep links / App Links into Activities, see [Navigation](navigation.md).

## Services

A Service runs without UI. Two modes (a service can be both):

- **Started** (`startService`/`onStartCommand`) — runs until it stops itself.
- **Bound** (`bindService`/`onBind`) — a client-server [Binder](binder.md) interface; lives while
  clients are bound.

**Foreground services** show an ongoing notification and are for user-visible ongoing work
(playback, navigation). Since Android 8 the OS heavily restricts **background** starts; Android
14 requires a declared **FGS type** (`location`, `mediaPlayback`, …). For deferrable/guaranteed
background work, prefer **WorkManager** — see [Background Execution](background_work.md). The
`POST_NOTIFICATIONS` runtime permission (Android 13+) gates the FGS notification — see
[Permissions & Privacy](permissions_privacy.md).

## BroadcastReceivers

Receivers subscribe to events (connectivity change, boot completed, custom app broadcasts).

- **Manifest-registered**: declared statically; but since Android 8 most **implicit** broadcasts
  can't be received this way (battery/background limits) — use an explicit component or
  context-registration.
- **Context-registered**: `registerReceiver(...)` tied to a lifecycle; **must** `unregisterReceiver`
  to avoid leaks. On Android 13+ set `RECEIVER_EXPORTED`/`RECEIVER_NOT_EXPORTED`.
- `LocalBroadcastManager` is deprecated — prefer in-process observers (a `SharedFlow`, see
  [Coroutines & Flow](coroutines_flow.md)) instead of broadcasts for app-internal events.

## ContentProviders

Providers expose data behind a `content://` URI with CRUD semantics — the basis of cross-app data
sharing and the framework's own **MediaStore**/Documents. You rarely write one for in-app storage
(use [Room/DataStore](data_persistence.md)); you *do* use one to **share files** safely:

- **`FileProvider`** vends temporary, permissioned `content://` URIs for files you share via Intent
  (camera capture, attachments) — avoids `FileUriExposedException`.
- Access is gated by URI permissions and provider-level permissions — see
  [Permissions & Privacy](permissions_privacy.md).

## Process & Startup

Components run inside the app's **process**, forked from [zygote](zygote_startup.md) on first
launch; the [system_server](system_server.md) ActivityManager orchestrates starts and lifecycle,
talking to the app over [Binder](binder.md). Several components can share one process (default) or
be split via `android:process`. See [Android Internals](internals.md) for the full picture.

## Pitfalls

- **Heavy work in lifecycle callbacks / `onReceive`** — blocks the main thread/ANRs; offload to
  coroutines or WorkManager.
- **Leaking context-registered receivers / bound services** — always unregister/unbind in the
  paired lifecycle callback.
- **Relying on implicit broadcasts** (Android 8+ restrictions) or **background starts** (use FGS
  with a type, or WorkManager).
- **`PendingIntent` without an explicit mutability flag** (Android 12+) — crashes at build/runtime.
- **Exported components without protection** — an `exported=true` Activity/Service/Receiver/
  Provider is an attack surface; require a permission or set `exported=false`. See
  [App Security](app_security.md).
- **Treating components as logic homes** — keep business logic in ViewModels/use cases
  ([App Architecture](app_architecture.md)).

## Where this connects

- [App Architecture](app_architecture.md) — keeping logic out of components
- [Navigation](navigation.md) — Activity/Fragment/Compose navigation, deep links
- [Background Execution](background_work.md) — WorkManager vs services for background work
- [Permissions & Privacy](permissions_privacy.md) — exported components, FileProvider, FGS perms
- [Binder](binder.md) · [SystemServer & Core Services](system_server.md) · [Zygote & App Startup](zygote_startup.md) — how the system starts components
- [App Security](app_security.md) — exported-component and `PendingIntent` hardening
