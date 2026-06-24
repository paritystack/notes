# Background Execution

## Overview

Running work when the app isn't in the foreground is heavily restricted on modern Android to
protect **battery** and **memory**. Over many releases, Google has tightened background
execution: **Doze** and **App Standby**, **background service limits** (Android 8.0),
**foreground service types** (Android 10+), and **exact-alarm** restrictions. The right tool
depends on whether the work is *deferrable*, *time-critical*, or *user-visible-ongoing*.

This builds on [Coroutines & Flow](coroutines_flow.md) (in-process async) and is orchestrated by
services in [SystemServer & Core Services](system_server.md) (AlarmManager, JobScheduler).

## Choosing the Right Tool

```text
Is the work tied to a visible, ongoing user task (music, navigation, active upload)?
   └─ YES → Foreground Service (with a notification + correct foregroundServiceType)

Must it run at an exact wall-clock time (alarm clock, calendar reminder)?
   └─ YES → AlarmManager (exact alarms — needs special permission)

Otherwise (deferrable, guaranteed eventually, survives reboot/process death)?
   └─ WorkManager   ← the default recommendation for most background work

Just async work while the app is alive (network call for current screen)?
   └─ Coroutines (viewModelScope) — not "background execution" in the restricted sense
```

## Power Management Restrictions

### Doze & App Standby

- **Doze**: when the device is unused, unplugged, and stationary, the system enters maintenance
  windows — deferring network access, alarms (except exact/allow-listed), jobs, and wakelocks
  to batched windows.
- **App Standby Buckets**: apps are bucketed (Active, Working set, Frequent, Rare, Restricted)
  by usage; rarer buckets get tighter limits on jobs/alarms/network.

```bash
# Simulate Doze / inspect buckets (debugging)
adb shell dumpsys deviceidle force-idle
adb shell dumpsys deviceidle unforce
adb shell am get-standby-bucket com.example.app
adb shell dumpsys usagestats | grep com.example.app
```

You generally **shouldn't fight Doze** — design work to be deferrable. WorkManager already
respects these constraints.

### Background service limits (Android 8.0+)

Apps can no longer freely start background services. Starting a service while in the background
must quickly become a **foreground service** (with a notification) or use WorkManager/JobScheduler.

## WorkManager (recommended default)

For **deferrable, guaranteed** background work — runs even after app exit/reboot, with
**constraints**, **retries/backoff**, and **chaining**. Built on JobScheduler under the hood.

```kotlin
class SyncWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result = try {
        repository.sync()       // suspend work runs off the main thread
        Result.success()
    } catch (e: IOException) {
        Result.retry()          // WorkManager applies backoff
    }
}

// Enqueue with constraints
val request = OneTimeWorkRequestBuilder<SyncWorker>()
    .setConstraints(
        Constraints.Builder()
            .setRequiredNetworkType(NetworkType.UNMETERED)   // Wi-Fi only
            .setRequiresCharging(true)
            .build()
    )
    .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 10, TimeUnit.SECONDS)
    .build()

WorkManager.getInstance(context).enqueueUniqueWork(
    "sync", ExistingWorkPolicy.KEEP, request,
)
```

```kotlin
// Periodic work (minimum interval 15 minutes)
val periodic = PeriodicWorkRequestBuilder<SyncWorker>(1, TimeUnit.HOURS).build()

// Chaining
WorkManager.getInstance(context)
    .beginWith(downloadWork)
    .then(processWork)
    .then(uploadWork)
    .enqueue()
```

WorkManager can also run **expedited** work (`setExpedited(...)`) for important, near-immediate
tasks, and **long-running**/foreground work via `setForeground()`.

## Foreground Services

For work the **user is actively aware of** and that must continue while ongoing (media playback,
navigation, active file upload, fitness tracking). Requires a persistent **notification** and,
since Android 10+/14, a declared **`foregroundServiceType`** and matching permission.

```xml
<uses-permission android:name="android.permission.FOREGROUND_SERVICE"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC"/>

<service
    android:name=".UploadService"
    android:foregroundServiceType="dataSync"
    android:exported="false"/>
```

```kotlin
class UploadService : Service() {
    override fun onStartCommand(i: Intent?, flags: Int, startId: Int): Int {
        val notification = buildNotification()
        startForeground(NOTIF_ID, notification)   // must call promptly to avoid ANR/crash
        // do the ongoing work (e.g. on a coroutine)
        return START_STICKY
    }
    override fun onBind(intent: Intent?) = null
}
```

Android 12+ blocks **starting** foreground services from the background in many cases — start
them from a foreground context or use WorkManager expedited work instead.

## AlarmManager

For **exact, wall-clock-time** triggers (alarm clocks, calendar events). Most apps should *not*
use exact alarms — they wake the device and harm battery.

```kotlin
val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
// Exact alarms require SCHEDULE_EXACT_ALARM / USE_EXACT_ALARM permission (Android 12+)
am.setExactAndAllowWhileIdle(
    AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent,
)
// Prefer inexact for non-critical timing:
am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent)
```

## Notifications

Background work often surfaces via notifications. Since Android 8.0 every notification needs a
**channel**; since Android 13 you must request the **`POST_NOTIFICATIONS`** runtime permission.

```kotlin
// Create a channel once
val channel = NotificationChannel("sync", "Sync", NotificationManager.IMPORTANCE_LOW)
notificationManager.createNotificationChannel(channel)

// Android 13+: request permission
requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
```

## Comparison

| Tool | Guaranteed? | Survives reboot | Exact timing | Needs notification | Use |
|------|-------------|-----------------|--------------|--------------------|-----|
| **Coroutines** | While app alive | No | No | No | UI-scoped async |
| **WorkManager** | Yes (deferred) | Yes | No | Only if foreground | Sync, upload, deferrable tasks |
| **Foreground Service** | While running | Restart policy | No | **Yes** | Ongoing user-visible work |
| **AlarmManager (exact)** | At time | Re-register on boot | **Yes** | No | Clocks/reminders only |

## Best Practices

1. **Default to WorkManager** for background work; let it respect Doze/Standby/constraints.
2. **Use a Foreground Service only for user-visible ongoing tasks** and declare the correct
   `foregroundServiceType`.
3. **Avoid exact alarms** unless the feature is literally a clock/reminder; request the special
   permission and provide a fallback.
4. **Don't start background services / FGS from the background** on Android 12+ — use expedited
   WorkManager.
5. **Batch network and wakeups**; set constraints (charging, unmetered) to be battery-friendly.
6. **Create notification channels** and request `POST_NOTIFICATIONS` on Android 13+.
7. **Test under Doze** (`dumpsys deviceidle force-idle`) and on restricted standby buckets.

## Resources

- [Background work — Android Developers](https://developer.android.com/develop/background-work)
- [WorkManager](https://developer.android.com/topic/libraries/architecture/workmanager)
- [Foreground services](https://developer.android.com/develop/background-work/services/foreground-services)
- [Optimize for Doze and App Standby](https://developer.android.com/training/monitoring-device-state/doze-standby)
- [Schedule alarms](https://developer.android.com/develop/background-work/services/alarms/schedule)

### Related Files

- [Coroutines & Flow](coroutines_flow.md) — in-process async vs background execution
- [Performance & Profiling](performance_profiling.md) — battery profiling
- [SystemServer & Core Services](system_server.md) — AlarmManager/JobScheduler services
- [Jetpack](jetpack.md) — WorkManager is part of Jetpack

## Where this connects

- [Coroutines & Flow](coroutines_flow.md) — structured concurrency for background tasks
- [App architecture](app_architecture.md) — where background work belongs
- [System server](system_server.md) — JobScheduler/AlarmManager live here
- [Performance & profiling](performance_profiling.md) — battery and wakeup costs
- [Zygote & app startup](zygote_startup.md) — process lifecycle and death
- [Jetpack](jetpack.md) — WorkManager as the recommended API
