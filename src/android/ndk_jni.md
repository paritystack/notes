# NDK & JNI

## Overview

The **Android NDK** (Native Development Kit) lets you write parts of an app in **C/C++** (and
other native languages), compiled to shared libraries (`.so`) that run directly on the device's
CPU rather than on [ART](art_runtime.md). **JNI** (Java Native Interface) is the bridge that
lets Java/Kotlin code call into — and be called back from — that native code.

Use native code for: CPU-intensive work (signal/image processing, codecs, crypto), reusing
existing C/C++ libraries, game engines, and low-latency audio. For most app logic, Kotlin on
ART is simpler and fast enough — reach for the NDK deliberately.

See [ART & Dalvik Runtime](art_runtime.md) for the managed runtime side and
[Project Treble & HALs](treble_hal.md) for native HAL development.

## The Native Toolchain

| Piece | Role |
|-------|------|
| **NDK** | Toolchain (Clang/LLVM), sysroot, headers, build glue |
| **Bionic** | Android's lightweight C library (not glibc) + dynamic linker |
| **libc++** | The C++ STL shipped/used on Android |
| **CMake / ndk-build** | Build systems for native code (CMake is the default) |
| **ABI** | The binary interface per CPU architecture (`arm64-v8a`, `armeabi-v7a`, `x86_64`) |

**Bionic** is intentionally smaller than glibc and tuned for mobile; subtle behavioral
differences (locale, some POSIX corners) can surface when porting desktop C/C++ code.

## ABIs

Android devices use different CPU architectures, so native libs are built per **ABI** and the
matching variant is installed (Play split APKs deliver only the device's ABI — see
[APK/AAB Packaging & Signing](app_signing.md)).

| ABI | CPU |
|-----|-----|
| `arm64-v8a` | 64-bit ARM (the dominant modern target; required for Play) |
| `armeabi-v7a` | 32-bit ARM (older devices) |
| `x86_64` | 64-bit x86 (emulators, some Chromebooks) |

```gradle
android {
    defaultConfig {
        ndk {
            // limit which ABIs to build/ship
            abiFilters 'arm64-v8a', 'armeabi-v7a'
        }
        externalNativeBuild {
            cmake { cppFlags '-std=c++17' }
        }
    }
    externalNativeBuild {
        cmake { path 'src/main/cpp/CMakeLists.txt' }
    }
}
```

## JNI: Crossing the Boundary

### Native method declaration (Kotlin)

```kotlin
class NativeLib {
    // 'external' means the implementation is in a loaded native library
    external fun stringFromJNI(): String
    external fun add(a: Int, b: Int): Int

    companion object {
        init { System.loadLibrary("native-lib") }  // loads libnative-lib.so
    }
}
```

### Native implementation (C++)

JNI function names follow a strict convention:
`Java_<package_with_underscores>_<Class>_<method>`.

```cpp
#include <jni.h>
#include <string>

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_app_NativeLib_stringFromJNI(JNIEnv* env, jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_app_NativeLib_add(JNIEnv* env, jobject thiz, jint a, jint b) {
    return a + b;
}
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(nativelib)

add_library(native-lib SHARED native-lib.cpp)

find_library(log-lib log)            # Android's logging library
target_link_libraries(native-lib ${log-lib})
```

### The JNIEnv and dynamic registration

`JNIEnv*` is the per-thread interface to the JVM/ART (allocate objects, call methods, throw
exceptions). It is **not shareable across threads** — get it per thread via the `JavaVM`.

For better performance and to avoid name-mangling fragility, register methods explicitly in
`JNI_OnLoad` instead of relying on name discovery:

```cpp
static jint native_add(JNIEnv*, jclass, jint a, jint b) { return a + b; }

static const JNINativeMethod methods[] = {
    {"add", "(II)I", reinterpret_cast<void*>(native_add)},  // "(II)I" = JNI type signature
};

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass cls = env->FindClass("com/example/app/NativeLib");
    env->RegisterNatives(cls, methods, sizeof(methods)/sizeof(methods[0]));
    return JNI_VERSION_1_6;
}
```

JNI **type signatures**: `I`=int, `J`=long, `Z`=boolean, `D`=double, `Ljava/lang/String;`=String,
`[I`=int[]. A method `(II)I` takes two ints, returns int.

## References: Local, Global, Weak

A frequent source of crashes and leaks:

| Reference | Lifetime | Notes |
|-----------|----------|-------|
| **Local** | Until the native method returns | Default for objects from `JNIEnv`; don't cache across calls. Limited slots — `DeleteLocalRef` in loops |
| **Global** | Until you `DeleteGlobalRef` | Use to cache objects/classes across calls; you own the lifetime |
| **Weak global** | Until GC collects the referent | For caches that shouldn't keep objects alive |

```cpp
// Cache a class reference across calls (created once, e.g. in JNI_OnLoad)
jclass localCls = env->FindClass("com/example/app/Foo");
jclass gCls = static_cast<jclass>(env->NewGlobalRef(localCls));
env->DeleteLocalRef(localCls);
// ... later, on shutdown:  env->DeleteGlobalRef(gCls);
```

## Threads & Exceptions

- **Attach native threads** before using JNI: `vm->AttachCurrentThread(&env, nullptr)` and
  `DetachCurrentThread()` when done; never reuse a `JNIEnv*` from another thread.
- **Check/clear pending exceptions** after JNI calls that can throw — JNI does *not* unwind C++
  on a pending Java exception; you must call `env->ExceptionCheck()` / `ExceptionClear()`.

```cpp
env->CallVoidMethod(obj, mid);
if (env->ExceptionCheck()) {
    env->ExceptionDescribe();   // log it
    env->ExceptionClear();
    // handle/return
}
```

## Logging & Debugging Native Code

```cpp
#include <android/log.h>
#define LOG_TAG "MyNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
LOGI("value = %d", x);
```

```bash
# Native crashes produce a tombstone with a backtrace
adb logcat | grep -E "DEBUG|tombstone"
adb shell ls /data/tombstones/
# Symbolicate a native crash backtrace
ndk-stack -sym path/to/obj/local/arm64-v8a -dump crash.txt
```

Debug with **LLDB** via Android Studio (native/dual debugger), and analyze leaks/UB with
**AddressSanitizer (ASan)** / HWASan builds.

## Performance Notes

- **JNI calls have overhead** — minimize boundary crossings; batch work in native code rather
  than calling back into Java in tight loops.
- **Use direct `ByteBuffer`s** (`NewDirectByteBuffer` / `GetDirectBufferAddress`) or primitive
  array critical regions to pass bulk data without copying.
- **Cache `jmethodID`/`jfieldID`/`jclass`** (as global refs) — looking them up each call is slow.
- **Avoid creating many local refs** in loops; delete them or use `PushLocalFrame`.

## Best Practices

1. **Use native code only where it pays off** — keep the JNI surface small and well-defined.
2. **Build `arm64-v8a`** (required by Play) and limit ABIs with `abiFilters`.
3. **Manage references explicitly** — global refs for caches, delete local refs in loops.
4. **Always check JNI exceptions** after calls that can throw.
5. **Attach/detach threads** correctly; never share `JNIEnv*` across threads.
6. **Prefer explicit `RegisterNatives`** over name-based binding for speed and robustness.
7. **Symbolicate crashes with `ndk-stack`** and test with ASan.

## Resources

- [Android NDK guides](https://developer.android.com/ndk/guides)
- [JNI tips — Android Developers](https://developer.android.com/training/articles/perf-jni)
- [Add C/C++ to your project](https://developer.android.com/studio/projects/add-native-code)
- [Bionic (libc)](https://android.googlesource.com/platform/bionic/+/master/README.md)
- [JNI specification (Oracle)](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/)

### Related Files

- [ART & Dalvik Runtime](art_runtime.md) — the managed runtime JNI bridges to
- [Project Treble & HALs](treble_hal.md) — native HAL development
- [APK/AAB Packaging & Signing](app_signing.md) — per-ABI native lib packaging

## Where this connects

- [ART & Dalvik runtime](art_runtime.md) — the managed side JNI bridges to
- [Gradle deep dive](gradle_android.md) — building native code in the app
- [App security](app_security.md) — native attack surface and hardening
- [Performance & profiling](performance_profiling.md) — native profiling and ABIs
- [Graphics stack](graphics_stack.md) — native rendering via JNI
- [ISA](../embedded/isa.md) — the architectures/ABIs native code targets
