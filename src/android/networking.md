# Networking

## Overview

Most apps talk to a backend over HTTP. The Android stack is layered: **OkHttp** (the HTTP client —
connections, interceptors, caching), a typed API layer (**Retrofit** for REST, or **Ktor Client**
as a multiplatform alternative), and a **serialization** library (kotlinx.serialization / Moshi /
Gson) to map JSON to data classes. This page covers that stack and how it becomes the *remote
source* behind a repository in [App Architecture](app_architecture.md), feeding
[Data & Persistence](data_persistence.md) as the cache. It uses [Coroutines & Flow](coroutines_flow.md)
for async calls and is tested with **MockWebServer** (see [JUnit/MockK](testing_junit_mockk.md)).

```
Repository ──► Retrofit (typed API) ──► OkHttp (client, interceptors, cache) ──► network
                     │
                     └── converter (kotlinx.serialization / Moshi) ⇄ data classes
```

## Retrofit

Declare the API as an interface; Retrofit generates the implementation. Use `suspend` so calls
integrate with [coroutines](coroutines_flow.md).

```kotlin
interface UserApi {
    @GET("users/{id}")          suspend fun getUser(@Path("id") id: Int): User
    @GET("users")               suspend fun list(@Query("page") page: Int): List<User>
    @POST("users")              suspend fun create(@Body user: User): User
    @GET("feed")                suspend fun feed(): Response<Feed>      // access headers/code
}

val api = Retrofit.Builder()
    .baseUrl("https://api.example.com/")
    .client(okHttp)
    .addConverterFactory(Json.asConverterFactory("application/json".toMediaType()))
    .build()
    .create(UserApi::class.java)
```

- Return `T` for the happy path, **`Response<T>`** when you need status/headers, or a
  **`Result`/sealed type** wrapper for explicit error modeling.
- `@Path`, `@Query`, `@Header`, `@Body`, `@Multipart`/`@Part` cover the common shapes.

## OkHttp: Client, Interceptors, Caching

Retrofit sits on **OkHttp**. Configure it once and share the instance (connection pooling lives on
the client):

```kotlin
val okHttp = OkHttpClient.Builder()
    .addInterceptor(AuthInterceptor(tokenStore))        // application interceptor
    .addNetworkInterceptor(loggingInterceptor)          // network interceptor
    .cache(Cache(context.cacheDir.resolve("http"), 10L * 1024 * 1024))
    .connectTimeout(10, SECONDS).readTimeout(30, SECONDS)
    .build()
```

- **Application interceptor** — runs once per call; good for auth headers, retries, logging.
- **Network interceptor** — runs per network hop; sees redirects/cache hits, can rewrite
  request/response (e.g. force cache headers).
- **Auth refresh**: an `Authenticator` reactively responds to 401s by refreshing the token.
- **HTTP cache**: OkHttp honors `Cache-Control`; a disk cache enables conditional GETs (ETag/
  `If-None-Match`) — cheap offline-ish reads.

```kotlin
class AuthInterceptor(private val tokens: TokenStore) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val req = chain.request().newBuilder()
            .header("Authorization", "Bearer ${tokens.access()}").build()
        return chain.proceed(req)
    }
}
```

## Serialization

Map JSON ⇄ Kotlin data classes:

| Library | Notes |
|---------|-------|
| **kotlinx.serialization** | Kotlin-native, compile-time, multiplatform; `@Serializable` data classes |
| **Moshi** | Reflection or codegen (`@JsonClass`), great Kotlin support |
| **Gson** | Legacy; weaker Kotlin null/default handling — avoid for new code |

```kotlin
@Serializable
data class User(val id: Int, val name: String, @SerialName("avatar_url") val avatar: String? = null)
```

Prefer codegen (kotlinx.serialization / Moshi codegen) over reflection for speed and R8/ProGuard
safety (reflection-based parsers need keep rules — see [Gradle Deep Dive](gradle_android.md)).

## Ktor Client (Alternative)

**Ktor Client** is a coroutine-first, multiplatform HTTP client — useful for Kotlin Multiplatform
or when you want one client across targets. It bundles content negotiation, auth, logging, and
retry as pluggable features:

```kotlin
val client = HttpClient(OkHttp) {
    install(ContentNegotiation) { json() }
    install(Logging)
}
suspend fun user(id: Int): User = client.get("https://api.example.com/users/$id").body()
```

## Error Handling & Resilience

- **Model failures explicitly** — wrap calls in `runCatching`/a `Result` type and map
  `HttpException`/`IOException` to domain errors the UI can render.
- **Timeouts & retries** — set sane OkHttp timeouts; retry idempotent calls with backoff (don't
  blindly retry POSTs).
- **Offline-first** — let [Data & Persistence](data_persistence.md) be the source of truth; network
  fills the cache (see the repository pattern in [App Architecture](app_architecture.md)).
- **Cancellation** — `suspend` calls cancel with their coroutine scope (e.g. `viewModelScope`),
  freeing the connection.

## Security

- **HTTPS only**; block cleartext via the network security config (`cleartextTrafficPermitted=false`).
- **Certificate pinning** with OkHttp `CertificatePinner` for high-value endpoints (weigh rotation
  risk).
- **Never log tokens/PII** in interceptors in release builds. See [App Security](app_security.md).

## Pitfalls

- **Creating a client per request** — defeats connection pooling/caching; share one `OkHttpClient`.
- **Blocking calls on the main thread** — use `suspend` + a background dispatcher.
- **Reflection serializers without R8 keep rules** — release builds crash on missing fields.
- **Swallowing `Response` error bodies** — check `isSuccessful`; read `errorBody()` for diagnostics.
- **Retrying non-idempotent requests** — can double-submit; only retry safe methods.

## Where this connects

- [App Architecture](app_architecture.md) — remote source behind the repository
- [Data & Persistence](data_persistence.md) — local cache / single source of truth
- [Coroutines & Flow](coroutines_flow.md) — suspend calls, cancellation, dispatchers
- [App Security](app_security.md) — TLS, pinning, network security config
- [JUnit, MockK & Truth](testing_junit_mockk.md) — MockWebServer for network-layer tests
- [Android Gradle Deep Dive](gradle_android.md) — R8/ProGuard keep rules for serializers
