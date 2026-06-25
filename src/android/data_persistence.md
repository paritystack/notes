# Data & Persistence

## Overview

Android apps persist data at several tiers — **key-value** (DataStore / SharedPreferences),
**structured/relational** (Room over SQLite), **files** (app-private and shared storage), and
**network-backed caches**. This page covers each, when to use it, and how it feeds the
**repository/data layer** of [App Architecture](app_architecture.md). It pairs with
[Networking](networking.md) (the remote source behind a repository),
[Coroutines & Flow](coroutines_flow.md) (async reads as `Flow`/`suspend`), and is exercised by
[Robolectric](testing_robolectric.md) and [JUnit/MockK](testing_junit_mockk.md) tests.

```
ViewModel ──► Repository ──┬──► local source (Room / DataStore / files)   ← single source of truth
                          └──► remote source (Retrofit/Ktor)             → write-through cache
```

## Key-Value: DataStore vs SharedPreferences

**SharedPreferences** is the legacy XML-backed store — synchronous, easy to misuse on the main
thread, no transactional guarantees. **DataStore** is the modern replacement: async (coroutines/
`Flow`), transactional, and type-safe.

| | SharedPreferences | Preferences DataStore | Proto DataStore |
|--|-------------------|-----------------------|-----------------|
| API | Synchronous (blocking) | `suspend`/`Flow` | `suspend`/`Flow` |
| Schema | Untyped keys | Untyped keys | Typed (protobuf schema) |
| Use for | Legacy code | Simple typed prefs | Structured settings with a schema |

```kotlin
val Context.dataStore by preferencesDataStore("settings")
val DARK = booleanPreferencesKey("dark_mode")

val darkMode: Flow<Boolean> = context.dataStore.data.map { it[DARK] ?: false }

suspend fun setDark(on: Boolean) {
    context.dataStore.edit { it[DARK] = on }
}
```

Prefer DataStore for new code; reach for **Proto DataStore** when settings have real structure
(so reads/writes are typed and validated).

## Relational: Room over SQLite

**Room** is the recommended ORM-ish layer over SQLite: compile-time-verified SQL, suspend/`Flow`
returns, and migrations. The three pieces are **Entity** (table), **DAO** (queries), and
**Database**.

```kotlin
@Entity(tableName = "users")
data class User(@PrimaryKey val id: Int, val name: String)

@Dao
interface UserDao {
    @Query("SELECT * FROM users") fun observeAll(): Flow<List<User>>   // reactive
    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun upsert(u: User)
    @Transaction @Query("…") suspend fun withRelations(): List<UserWithPosts>
}

@Database(entities = [User::class], version = 2)
abstract class AppDb : RoomDatabase() { abstract fun userDao(): UserDao }
```

- **Reactive reads**: a `Flow<…>` query re-emits whenever the underlying tables change — the basis
  of a reactive data layer.
- **Off the main thread**: `suspend`/`Flow` DAO methods run on Room's executors; never block the
  UI thread on a query.
- **Relations**: `@Relation`/`@Embedded` + `@Transaction` for joins across entities.
- **Type converters**: `@TypeConverter` to store non-primitive columns (dates, enums).
- **FTS / full-text**: `@Fts4` entities for search.

### Migrations

Bumping `version` requires a migration or Room throws at runtime. Provide a `Migration`, or
`fallbackToDestructiveMigration()` only for throwaway/dev builds (it wipes data).

```kotlin
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
    }
}
```

Enable `exportSchema = true` and commit the generated schema JSON so migrations are testable with
`MigrationTestHelper`.

## Files & Storage Scopes

Android storage is **scoped** — apps get private dirs freely; shared media needs the right API.

| Location | Access | Persists on uninstall? |
|----------|--------|------------------------|
| `filesDir` / `cacheDir` (internal) | App-private, no permission | No (cache may be cleared sooner) |
| `getExternalFilesDir()` | App-private on external storage | No |
| **MediaStore** (Photos/Video/Audio) | Shared media via content URIs | Yes |
| **Storage Access Framework** (SAF) | User-picked files/trees (`ACTION_OPEN_DOCUMENT`) | Yes |

**Scoped storage** (Android 10+) means you can't roam the raw filesystem: use **MediaStore** for
media and **SAF** / the **Photo Picker** for user documents — no broad storage permission. See
[App Security](app_security.md) and [SELinux on Android](selinux_android.md) for the enforcement
behind app-private dirs (each app's UID-isolated `/data/data/<pkg>`).

## The Single Source of Truth

A repository exposes data as a `Flow` backed by **local storage**, and refreshes it from the
network — the UI always renders the local copy (offline-first):

```kotlin
class UserRepository(private val dao: UserDao, private val api: Api) {
    val users: Flow<List<User>> = dao.observeAll()          // SSOT = Room

    suspend fun refresh() = dao.upsert(api.fetchUsers())     // write-through from network
}
```

This decouples UI from network latency/failures and is the testable seam for fakes (see
[App Architecture](app_architecture.md), [JUnit/MockK](testing_junit_mockk.md)).

## Pitfalls

- **Blocking the main thread** on SharedPreferences/SQLite — use DataStore / suspend DAOs.
- **Forgetting a migration** on `version` bump — crash on launch; never ship
  `fallbackToDestructiveMigration()` to production.
- **Storing large blobs in Room** — keep files on disk, store the path/URI.
- **Assuming raw file paths work** post-Android-10 — use MediaStore/SAF for shared storage.
- **Caching secrets in plaintext** — use EncryptedSharedPreferences/keystore (see [App Security](app_security.md)).

## Where this connects

- [App Architecture](app_architecture.md) — repository/data layer and single source of truth
- [Networking](networking.md) — the remote source behind a repository
- [Coroutines & Flow](coroutines_flow.md) — reactive `Flow` reads and suspend writes
- [App Security](app_security.md) — at-rest encryption, per-app data isolation
- [Robolectric](testing_robolectric.md) · [JUnit, MockK & Truth](testing_junit_mockk.md) — testing the data layer
