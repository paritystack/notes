# Android Gradle Deep Dive

## Overview

**Gradle** with the **Android Gradle Plugin (AGP)** is Android's build system. It compiles
sources, processes resources, runs annotation processors/KSP, merges manifests, dexes,
shrinks (R8), and packages/signs APKs/AABs. Understanding build **types**, product **flavors**,
**variants**, **version catalogs**, and build-speed levers pays off daily — slow or fragile
builds are a major productivity drain.

Connects to [APK/AAB Packaging & Signing](app_signing.md) (signing configs, R8) and
[Performance & Profiling](performance_profiling.md) (R8/app size).

## Project Structure

```text
settings.gradle(.kts)     ← declares modules + plugin/dependency repositories
build.gradle(.kts)        ← root: plugin versions (often via plugins {} + version catalog)
gradle/libs.versions.toml ← version catalog (centralized dependency versions)
app/build.gradle(.kts)    ← module config: android {}, dependencies {}
gradle/wrapper/...        ← Gradle wrapper (pinned Gradle version)
```

Two DSLs exist: **Groovy** (`build.gradle`) and **Kotlin** (`build.gradle.kts`). Kotlin DSL is
recommended for IDE completion and type safety.

## Build Types, Flavors & Variants

### Build types

Differ by *how* you build (debug vs release):

```kotlin
android {
    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            isDebuggable = true
        }
        release {
            isMinifyEnabled = true          // R8 shrink/obfuscate
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
        }
    }
}
```

### Product flavors

Differ by *what* you build (free vs paid, dev vs prod backend):

```kotlin
android {
    flavorDimensions += "tier"
    productFlavors {
        create("free")  { dimension = "tier"; applicationIdSuffix = ".free" }
        create("paid")  { dimension = "tier" }
    }
}
```

### Variants = build type × flavor

`free` + `paid` × `debug` + `release` → `freeDebug`, `freeRelease`, `paidDebug`, `paidRelease`.
Each variant can have its own source set (`src/free/`, `src/paidRelease/`) and resources.

```bash
./gradlew tasks --all | grep assemble    # see assembleFreeDebug, assemblePaidRelease, ...
./gradlew assembleFreeDebug
```

### BuildConfig & manifest placeholders

```kotlin
android {
    buildFeatures { buildConfig = true }
    defaultConfig {
        buildConfigField("String", "API_URL", "\"https://api.example.com\"")
        manifestPlaceholders["mapsKey"] = System.getenv("MAPS_KEY") ?: ""
    }
}
// usage: BuildConfig.API_URL
```

## Version Catalogs

The modern way to centralize dependency and plugin versions across modules:
`gradle/libs.versions.toml`.

```toml
[versions]
kotlin = "2.0.0"
coroutines = "1.8.1"
retrofit = "2.11.0"

[libraries]
kotlinx-coroutines = { module = "org.jetbrains.kotlinx:kotlinx-coroutines-android", version.ref = "coroutines" }
retrofit = { module = "com.squareup.retrofit2:retrofit", version.ref = "retrofit" }

[bundles]
networking = ["retrofit", "kotlinx-coroutines"]

[plugins]
android-app = { id = "com.android.application", version = "8.5.0" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
```

```kotlin
// app/build.gradle.kts
plugins {
    alias(libs.plugins.android.app)
    alias(libs.plugins.kotlin.android)
}
dependencies {
    implementation(libs.bundles.networking)
    implementation(libs.retrofit)
}
```

## Dependency Configurations

| Configuration | Visible to consumers? | Use |
|---------------|-----------------------|-----|
| `implementation` | No (not on consumers' compile classpath) | Default — faster incremental builds |
| `api` | Yes | Only when a library re-exposes another's types |
| `compileOnly` | Compile only | Annotations, provided-at-runtime libs |
| `runtimeOnly` | Runtime only | e.g. logging backends |
| `ksp` / `kapt` | Annotation processing | KSP preferred (faster) |
| `testImplementation` / `androidTestImplementation` | Tests | Unit vs instrumented |

Prefer **`implementation`** over `api` to keep module API surfaces small and incremental builds
fast (changing an `implementation` dep doesn't force recompilation of downstream modules).

## KSP vs KAPT

**KSP** (Kotlin Symbol Processing) is the modern, much faster replacement for **KAPT** (which
generates Java stubs). Use KSP for Room, Moshi, Hilt, etc. wherever supported.

```kotlin
plugins { id("com.google.devtools.ksp") }
dependencies {
    implementation(libs.room.runtime)
    ksp(libs.room.compiler)            // was kapt(...)
}
```

## Convention Plugins (multi-module)

In large/modularized apps ([App Architecture](app_architecture.md)), avoid copy-pasting config
across dozens of modules. **Convention plugins** in a `build-logic` included build centralize
shared setup (compile SDK, Kotlin options, common deps).

```kotlin
// build-logic/.../AndroidLibraryConventionPlugin.kt
class AndroidLibraryConventionPlugin : Plugin<Project> {
    override fun apply(target: Project) = with(target) {
        pluginManager.apply("com.android.library")
        pluginManager.apply("org.jetbrains.kotlin.android")
        // configure compileSdk, java/kotlin toolchain, common deps once...
    }
}
// then in a module: plugins { id("myapp.android.library") }
```

## Build Speed

Gradle build speed is often the #1 local dev pain. Key levers (in `gradle.properties`):

```properties
org.gradle.caching=true            # reuse outputs across builds (build cache)
org.gradle.parallel=true           # build independent modules in parallel
org.gradle.configuration-cache=true# cache the configuration phase
org.gradle.jvmargs=-Xmx4g -XX:+UseParallelGC
kotlin.incremental=true
```

Other wins:

- **Modularize** so changes recompile fewer modules; use `implementation` not `api`.
- **Prefer KSP over KAPT**.
- **Keep AGP/Gradle/Kotlin up to date** — each release improves build perf.
- **Profile builds** with `--scan` (Build Scan) or `--profile` to find bottlenecks.
- Disable unused `buildFeatures` (e.g. `buildConfig`, `aidl`, `viewBinding`) per module.

```bash
./gradlew assembleDebug --scan         # uploadable build scan with timing breakdown
./gradlew assembleDebug --profile      # local HTML profile report
./gradlew help --task assembleDebug    # inspect a task
```

## Common Tasks

```bash
./gradlew assembleRelease       # build release APK(s)
./gradlew bundleRelease         # build release AAB
./gradlew testDebugUnitTest     # local JVM unit tests
./gradlew connectedAndroidTest  # instrumented tests on device/emulator
./gradlew lint                  # Android Lint
./gradlew :app:dependencies     # dependency tree (resolve conflicts)
./gradlew clean                 # remove build outputs
```

## Best Practices

1. **Use the Kotlin DSL + a version catalog** to centralize versions and get type safety.
2. **Prefer `implementation` over `api`**; minimize exposed API surfaces.
3. **Use KSP, not KAPT**, wherever available.
4. **Centralize shared config in convention plugins** for multi-module projects.
5. **Turn on build/configuration cache + parallel**; keep AGP/Gradle/Kotlin current.
6. **Keep secrets out of build files**; inject signing creds via env vars (see [App Signing](app_signing.md)).
7. **Publish AABs** (`bundleRelease`) and let R8 shrink release builds.
8. **Profile builds** with `--scan`/`--profile` before guessing at slowness.

## Resources

- [Configure your build — Android Developers](https://developer.android.com/build)
- [Android Gradle plugin release notes](https://developer.android.com/build/releases/gradle-plugin)
- [Version catalogs — Gradle](https://docs.gradle.org/current/userguide/platforms.html)
- [KSP](https://kotlinlang.org/docs/ksp-overview.html)
- [Optimize build speed](https://developer.android.com/build/optimize-your-build)

### Related Files

- [APK/AAB Packaging & Signing](app_signing.md) — signing configs, R8, AAB output
- [Performance & Profiling](performance_profiling.md) — R8 shrinking & app size
- [App Architecture](app_architecture.md) — modularization that shapes the build graph
- [Android Testing](testing_android.md) — test tasks/configurations
