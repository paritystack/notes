# Android Internals

## Overview

Android is an open-source operating system primarily designed for mobile devices such as smartphones and tablets. It is based on the Linux kernel and developed by Google. Understanding Android internals is crucial for developers who want to create efficient and optimized applications or modify the operating system itself.

This document provides deep technical insights into Android's architecture, from low-level kernel operations to high-level application frameworks.

## System Architecture & Boot Process

### Boot Sequence

The Android boot process involves several stages, each building upon the previous one to create a fully functional operating system.

#### 1. Boot ROM

When a device is powered on, the CPU executes code from the Boot ROM (Read-Only Memory), which is hard-wired into the SoC (System on Chip). The Boot ROM code:
- Initializes basic hardware (CPU, RAM)
- Locates the bootloader in flash memory
- Loads and verifies the bootloader
- Transfers control to the bootloader

#### 2. Bootloader

The bootloader is device-specific and runs in two stages:

**Primary Bootloader**: Minimal code that initializes essential hardware and loads the secondary bootloader.

**Secondary Bootloader** (e.g., U-Boot, LK):
- Initializes additional hardware components
- Sets up memory and loads the Linux kernel
- Loads the device tree blob (DTB) which describes hardware configuration
- Verifies boot image integrity (in verified boot)
- Passes control to the kernel with boot parameters

#### 3. Linux Kernel Initialization

The kernel startup process:

```c
// Simplified kernel boot flow
start_kernel()
  ├─ setup_arch()          // Architecture-specific setup
  ├─ mm_init()             // Memory management initialization
  ├─ sched_init()          // Scheduler initialization
  ├─ init_IRQ()            // Interrupt handling
  ├─ time_init()           // Timekeeping
  └─ rest_init()
       └─ kernel_thread(kernel_init)  // Creates init process (PID 1)
```

The kernel:
- Initializes core subsystems (memory, scheduling, drivers)
- Mounts the root filesystem
- Executes `/init` (Android's init process)

#### 4. Init Process

The init process (`/system/bin/init`) is the first user-space process (PID 1) and is responsible for:

**Parsing Init Scripts**: Reads `init.rc` and device-specific `.rc` files using the Android Init Language (AIL).

```bash
# Example init.rc snippet
on boot
    # Set memory parameters
    write /proc/sys/vm/overcommit_memory 1
    write /proc/sys/vm/min_free_order_shift 4

    # Start essential services
    start servicemanager
    start surfaceflinger

service servicemanager /system/bin/servicemanager
    class core
    user system
    critical
    onrestart restart healthd
    onrestart restart zygote
```

**Device Node Creation**: Creates `/dev` nodes for hardware devices.

**Property Service**: Manages system properties (similar to environment variables) accessible via `getprop`/`setprop`.

**Service Management**: Starts, stops, and monitors system services defined in `.rc` files.

#### 5. Zygote Process

Zygote is a specialized daemon that serves as the parent process for all Android applications.

**Why Zygote?**
- Pre-loads common classes and resources
- Forks new processes quickly without reloading shared libraries
- Reduces memory footprint through copy-on-write (COW)

**Zygote Initialization**:

```java
// Simplified Zygote startup
public static void main(String argv[]) {
    // 1. Preload classes and resources
    preload();

    // 2. Start system server
    if (startSystemServer) {
        Runnable r = forkSystemServer(/* args */);
    }

    // 3. Listen for new app requests on socket
    caller = zygoteServer.runSelectLoop();
}

private static void preload() {
    // Preload ~4000 classes from /system/etc/preloaded-classes
    preloadClasses();

    // Preload drawable resources, color state lists
    preloadResources();

    // Load shared libraries (libjavacore, libopenjdk, etc.)
    preloadSharedLibraries();

    // Preload HAL implementations
    preloadHAL();
}
```

When a new app needs to launch:
1. ActivityManagerService sends request to Zygote via socket
2. Zygote forks itself using `fork()`
3. Child process specializes (sets UID, GID, SELinux context)
4. Child loads the application code
5. Application's `ActivityThread.main()` executes

#### 6. System Server

System Server is forked from Zygote and hosts most Android framework services:

```java
// System Server service categories
private void startBootstrapServices() {
    // Core services needed for other services
    ActivityManagerService
    PowerManagerService
    PackageManagerService
    UserManagerService
}

private void startCoreServices() {
    BatteryService
    UsageStatsService
    WebViewUpdateService
}

private void startOtherServices() {
    // 60+ services including:
    WindowManagerService
    InputManagerService
    NetworkManagementService
    ConnectivityService
    WifiService
    BluetoothService
    LocationManagerService
    NotificationManagerService
    // ... and many more
}
```

Each service:
- Runs in the System Server process
- Registers with ServiceManager
- Communicates via Binder IPC
- Has specific permissions and capabilities

### Complete Boot Timeline

```
Power On
  ↓ (~100ms)
Boot ROM → Bootloader
  ↓ (~2s)
Kernel Init → /init
  ↓ (~1s)
Init Process → Parse .rc files → Start Services
  ↓ (~500ms)
Zygote → Preload → Fork System Server
  ↓ (~3s)
System Server → Start Framework Services
  ↓ (~2s)
Boot Animation → Launcher → User Space
  ↓ (~1s)
Boot Complete (typically 8-10 seconds total)
```

## Binder IPC Mechanism

Binder is Android's custom Inter-Process Communication (IPC) mechanism, designed for efficient and secure communication between processes.

### Why Binder?

Traditional Linux IPC mechanisms (pipes, sockets, shared memory) have limitations:
- **Performance**: Multiple copy operations, context switches
- **Security**: No built-in object-level permissions
- **Object-oriented**: Not designed for remote method invocation

Binder provides:
- Single data copy (client → kernel → server)
- Object reference counting
- Thread pool management
- Death notifications
- Built-in security (UID/PID verification)

### Binder Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Client Process    │         │   Server Process    │
│                     │         │                     │
│  ┌───────────────┐  │         │  ┌───────────────┐  │
│  │  Proxy Object │  │         │  │ Native Object │  │
│  └───────┬───────┘  │         │  └───────▲───────┘  │
│          │          │         │          │          │
│  ┌───────▼───────┐  │         │  ┌───────┴───────┐  │
│  │ Binder Driver │◄─┼────────►│  │ Binder Driver │  │
│  │   (libbinder) │  │         │  │   (libbinder) │  │
│  └───────┬───────┘  │         │  └───────▲───────┘  │
└──────────┼──────────┘         └──────────┼──────────┘
           │                               │
           │    ┌─────────────────────┐   │
           └───►│  Binder Kernel      │◄──┘
                │  Driver (/dev/binder)│
                └─────────────────────┘
```

### Binder Components

#### 1. Binder Driver

A kernel module (`/dev/binder`) that:
- Manages IPC transactions
- Maps memory between processes
- Handles object references
- Manages thread pools
- Implements death notifications

**Key IOCTLs**:
```c
// Client/Server interaction with /dev/binder
BINDER_WRITE_READ      // Read/write transaction data
BINDER_SET_MAX_THREADS // Configure thread pool
BINDER_VERSION         // Get driver version
BINDER_SET_CONTEXT_MGR // Become ServiceManager
```

#### 2. Binder Protocol

**Transaction Structure**:
```c
struct binder_transaction_data {
    union {
        size_t handle;   // Binder handle (client side)
        void *ptr;       // Binder pointer (server side)
    } target;

    void *cookie;        // Object cookie
    unsigned int code;   // Method code

    unsigned int flags;  // Transaction flags
    pid_t sender_pid;    // Sender process ID
    uid_t sender_euid;   // Sender user ID

    size_t data_size;    // Data buffer size
    size_t offsets_size; // Offsets buffer size

    union {
        struct {
            const void *buffer;  // Data buffer
            const void *offsets; // Object offsets
        } ptr;
        uint8_t buf[8];
    } data;
};
```

**Transaction Flow**:
1. Client calls method on proxy object
2. Proxy marshals parameters into Parcel
3. Proxy issues `BINDER_WRITE_READ` ioctl
4. Kernel copies data to target process
5. Server thread reads transaction
6. Server unmarshals Parcel
7. Server executes method
8. Server marshals reply into Parcel
9. Kernel copies reply to client
10. Client unmarshals reply

#### 3. ServiceManager

ServiceManager is the Binder "name service" (context manager):
- Runs as a separate process
- First process to open `/dev/binder`
- Handle 0 is reserved for ServiceManager
- Maintains registry of system services

```cpp
// Service registration
int main() {
    sp<ProcessState> ps = ProcessState::self();
    ps->setThreadPoolMaxThreads(0);
    ps->setContextManager(true);  // Become handle 0
    ps->giveThreadPoolName();

    sp<ServiceManager> manager = new ServiceManager();
    sp<BnServiceManager> bnManager = new BnServiceManager(manager);

    // Wait for service registration/lookup requests
    IPCThreadState::self()->joinThreadPool();
}

// Service lookup
sp<IBinder> getService(const String16& name) {
    sp<IServiceManager> sm = defaultServiceManager();
    return sm->checkService(name);  // Returns IBinder handle
}
```

**Common System Services**:
```bash
# List all registered services
$ adb shell service list
0    package: [android.content.pm.IPackageManager]
1    activity: [android.app.IActivityManager]
2    window: [android.view.IWindowManager]
3    input: [android.hardware.input.IInputManager]
...
```

#### 4. AIDL (Android Interface Definition Language)

AIDL defines interfaces for Binder communication:

```java
// IMyService.aidl
package com.example;

interface IMyService {
    int add(int a, int b);
    String getName();
    void setCallback(IMyCallback callback);
}

// IMyCallback.aidl
package com.example;

interface IMyCallback {
    void onEvent(String event);
}
```

**Generated Code**:

```java
// Auto-generated IMyService.java
public interface IMyService extends android.os.IInterface {
    // Stub class (server side)
    public static abstract class Stub extends android.os.Binder
        implements IMyService {

        @Override
        public boolean onTransact(int code, Parcel data,
                                  Parcel reply, int flags) {
            switch (code) {
                case TRANSACTION_add: {
                    int a = data.readInt();
                    int b = data.readInt();
                    int result = this.add(a, b);
                    reply.writeInt(result);
                    return true;
                }
                // ... other methods
            }
        }

        // Proxy class (client side)
        private static class Proxy implements IMyService {
            private IBinder mRemote;

            @Override
            public int add(int a, int b) throws RemoteException {
                Parcel data = Parcel.obtain();
                Parcel reply = Parcel.obtain();
                try {
                    data.writeInt(a);
                    data.writeInt(b);
                    mRemote.transact(TRANSACTION_add, data, reply, 0);
                    reply.readException();
                    return reply.readInt();
                } finally {
                    data.recycle();
                    reply.recycle();
                }
            }
        }
    }
}
```

### Parcels and Serialization

Parcels are containers for marshaling data:

```java
// Writing data
Parcel p = Parcel.obtain();
p.writeInt(42);
p.writeString("Hello");
p.writeStrongBinder(binder);  // Binder object reference

byte[] data = p.marshall();

// Reading data
Parcel p2 = Parcel.obtain();
p2.unmarshall(data, 0, data.length);
p2.setDataPosition(0);
int value = p2.readInt();
String str = p2.readString();
IBinder b = p2.readStrongBinder();
```

**Supported Types**:
- Primitives: int, long, float, double, boolean, char
- Strings and CharSequences
- Parcelable objects (custom serialization)
- Serializable objects (Java serialization)
- Binder references
- File descriptors
- Arrays and Lists

### Binder Thread Pool

Each process maintains a thread pool for handling incoming Binder calls:

```cpp
// Configure thread pool
ProcessState::self()->setThreadPoolMaxThreads(15);

// Start thread pool
ProcessState::self()->startThreadPool();

// Join main thread to pool
IPCThreadState::self()->joinThreadPool();
```

**Thread Management**:
- Kernel spawns threads as needed
- Threads block waiting for transactions
- Nested transactions supported (re-entrant calls)
- Thread limits prevent DoS attacks

### Security Model

Binder enforces security at multiple levels:

```java
// Service can check caller identity
int callerUid = Binder.getCallingUid();
int callerPid = Binder.getCallingPid();

// Verify caller has permission
if (callerUid != Process.SYSTEM_UID) {
    context.enforceCallingPermission(
        android.Manifest.permission.BLUETOOTH_ADMIN,
        "Need BLUETOOTH_ADMIN permission");
}

// Clear/restore identity for nested calls
long token = Binder.clearCallingIdentity();
try {
    // Call as local UID, not caller's UID
    doPrivilegedOperation();
} finally {
    Binder.restoreCallingIdentity(token);
}
```

## Memory & Process Management

Android's memory management is designed for constrained mobile environments with limited RAM.

### Process Lifecycle States

Android categorizes processes by importance:

#### 1. Foreground Process (oom_adj: 0)
- Has visible activity with user interaction
- Has service bound by foreground activity
- Has `Service.startForeground()` running
- Has BroadcastReceiver executing `onReceive()`

#### 2. Visible Process (oom_adj: 100)
- Has visible but not foreground activity (e.g., behind dialog)
- Has service bound to visible activity

#### 3. Service Process (oom_adj: 300-500)
- Has started service via `startService()`
- Not directly visible but performing user-aware work

#### 4. Cached Process (oom_adj: 900-906)
- Not currently needed
- Kept for faster restart
- First to be killed when memory is low

**Process State Tracking**:
```java
// ActivityManager importance constants
IMPORTANCE_FOREGROUND = 100;
IMPORTANCE_FOREGROUND_SERVICE = 125;
IMPORTANCE_VISIBLE = 200;
IMPORTANCE_SERVICE = 300;
IMPORTANCE_CACHED = 400;
IMPORTANCE_GONE = 1000;
```

### OOM Adjuster

The OOM (Out of Memory) Adjuster dynamically calculates process importance:

**Score Calculation Algorithm**:
```java
// Simplified OOM score calculation
int computeOomAdjLocked(ProcessRecord app) {
    int adj = ProcessList.UNKNOWN_ADJ;

    // Check activities
    if (app.activities.size() > 0) {
        for (ActivityRecord ar : app.activities) {
            if (ar.visible) {
                adj = Math.min(adj, ProcessList.VISIBLE_APP_ADJ);
            }
            if (ar.state == ActivityState.RESUMED) {
                adj = Math.min(adj, ProcessList.FOREGROUND_APP_ADJ);
            }
        }
    }

    // Check services
    for (ServiceRecord sr : app.services) {
        if (sr.isForeground) {
            adj = Math.min(adj, ProcessList.PERCEPTIBLE_APP_ADJ);
        }
    }

    // Check providers
    if (app.hasClientActivities) {
        adj = Math.min(adj, ProcessList.FOREGROUND_APP_ADJ);
    }

    return adj;
}
```

**OOM Score in Kernel**:
```bash
# View process OOM scores
$ adb shell cat /proc/$(pidof system_server)/oom_score_adj
-900

$ adb shell cat /proc/$(pidof com.android.systemui)/oom_score_adj
-800

$ adb shell cat /proc/$(pidof com.android.launcher3)/oom_score_adj
100
```

### Low Memory Killer (LMK)

Android uses LMK to kill processes when memory is low.

**Traditional LMK (Kernel Driver)**:
```c
// Kernel LMK thresholds (in pages, 1 page = 4KB)
static short lowmem_adj[] = {
    0,      // Foreground
    100,    // Visible
    200,    // Secondary server
    300,    // Hidden
    900,    // Content provider
    906     // Empty
};

static int lowmem_minfree[] = {
    3 * 512,   // 6MB   - foreground
    2 * 1024,  // 8MB   - visible
    4 * 1024,  // 16MB  - secondary
    16 * 1024, // 64MB  - hidden
    18 * 1024, // 72MB  - content provider
    20 * 1024  // 80MB  - empty
};
```

**Modern LMKD (User-space Daemon)**:

Android 9+ moved LMK to user-space daemon (`lmkd`) for better control:

```c
// lmkd monitors pressure stall information (PSI)
// /proc/pressure/memory
some avg10=0.00 avg60=0.00 avg300=0.00 total=0
full avg10=0.00 avg60=0.00 avg300=0.00 total=0

// When pressure increases, lmkd kills cached processes
// Priority: highest oom_adj first
int find_and_kill_process(int min_score_adj) {
    for (int i = OOM_SCORE_ADJ_MAX; i >= min_score_adj; i--) {
        ProcessRecord proc = findProcessWithAdj(i);
        if (proc != null) {
            kill(proc.pid, SIGKILL);
            return proc.pid;
        }
    }
}
```

**Memory Pressure Levels**:
- **Low**: Kill cached apps (oom_adj >= 900)
- **Medium**: Kill background services (oom_adj >= 600)
- **Critical**: Kill visible apps (oom_adj >= 200)

### Memory Management Strategies

#### 1. Process Reclamation
```java
// Application.onTrimMemory() callbacks
TRIM_MEMORY_RUNNING_MODERATE     // Device running low, app running
TRIM_MEMORY_RUNNING_LOW          // Device very low, app running
TRIM_MEMORY_RUNNING_CRITICAL     // Device critical, app running

TRIM_MEMORY_UI_HIDDEN            // UI no longer visible

TRIM_MEMORY_BACKGROUND           // App in background, low memory
TRIM_MEMORY_MODERATE             // App in middle of LRU list
TRIM_MEMORY_COMPLETE             // App first to be killed

// Example handling
@Override
public void onTrimMemory(int level) {
    switch (level) {
        case TRIM_MEMORY_RUNNING_CRITICAL:
            // Release all possible memory
            clearCaches();
            releaseNonEssentialResources();
            break;
        case TRIM_MEMORY_UI_HIDDEN:
            // Release UI resources
            releaseBitmaps();
            break;
    }
}
```

#### 2. Zygote Memory Sharing

Zygote preloads classes and resources, then forks. Child processes share read-only memory:

```bash
# View memory maps
$ adb shell cat /proc/$(pidof system_server)/maps | grep framework
# Many regions marked as shared (copy-on-write)

# Check shared/private memory
$ adb shell dumpsys meminfo com.android.systemui
                   Pss  Private  Shared
                 ------  ------  ------
  Native Heap    10432   10384      48
  Dalvik Heap     8456    8420      36
  Dalvik Other    1234      800     434  # Shared from Zygote
```

#### 3. Kernel Memory Reclamation

Linux kernel mechanisms used by Android:

**Page Cache**: Caches file contents in RAM
```bash
$ adb shell cat /proc/meminfo
MemTotal:        3870720 kB
MemFree:          123456 kB
Cached:          1234567 kB  # Can be freed if needed
```

**KSM (Kernel Samepage Merging)**: Merges identical memory pages
```bash
$ adb shell cat /sys/kernel/mm/ksm/pages_sharing
12345  # Pages deduped
```

**zRAM**: Compressed swap in RAM
```bash
$ adb shell cat /proc/swaps
Filename        Type      Size      Used    Priority
/dev/block/zram0 partition 1048576  524288  32758
```

### Memory Optimization Techniques

#### 1. Bitmap Management
```java
// Use appropriate bitmap config
BitmapFactory.Options options = new BitmapFactory.Options();
options.inPreferredConfig = Bitmap.Config.RGB_565;  // 2 bytes/pixel vs 4

// Subsample large images
options.inSampleSize = 2;  // 1/4 memory usage

// Reuse bitmaps
options.inBitmap = reusableBitmap;
options.inMutable = true;
```

#### 2. Memory Leaks Prevention
```java
// Avoid static references to Context
private static Context sContext;  // LEAK!

// Use WeakReference for callbacks
private WeakReference<Activity> mActivityRef;

// Unregister listeners
@Override
protected void onDestroy() {
    locationManager.removeUpdates(listener);
    super.onDestroy();
}
```

#### 3. Native Memory Management
```cpp
// JNI memory is not tracked by Dalvik GC
extern "C" JNIEXPORT void JNICALL
Java_com_example_NativeLib_allocate(JNIEnv* env, jobject obj) {
    // Allocate native memory
    void* buffer = malloc(1024 * 1024);  // 1 MB

    // MUST manually free
    free(buffer);
}

// Use Android NDK memory allocator
#include <android/native_memory.h>
ANativeWindow_Buffer buffer;
```

## Security & Permissions

Android implements defense-in-depth security through multiple layers.

### Application Sandboxing

Each app runs in its own security sandbox:

#### 1. UID Isolation

Every app gets a unique Linux user ID:

```bash
# App UIDs start at 10000
$ adb shell ps -u | grep com.android.chrome
u0_a123   12345  ... com.android.chrome

# System apps use special UIDs
$ adb shell ps -u | grep system_server
system     1234  ... system_server
```

**UID Assignment**:
```java
// PackageManagerService assigns UIDs
private int acquireAndRegisterNewUserIdLPw(Package pkg) {
    // User apps: 10000 - 19999
    // Secondary users: 100000+ (userId * 100000 + appId)
    int uid = Process.FIRST_APPLICATION_UID + mNextAppId++;
    return uid;
}
```

#### 2. Filesystem Permissions

Each app has a private directory:

```bash
# App data directories
$ adb shell ls -l /data/data/
drwx------  5 u0_a123 u0_a123  4096 com.android.chrome
drwx------  4 u0_a124 u0_a124  4096 com.google.android.gms

# Only the app's UID can access its directory
$ adb shell cat /data/data/com.android.chrome/databases/webdata
cat: permission denied

# Shared storage accessible by all apps
$ adb shell ls -l /sdcard/
drwxrwx--x 2 root sdcard_rw 4096 DCIM
drwxrwx--x 3 root sdcard_rw 4096 Download
```

#### 3. Process Isolation

Each app runs in its own process:
- Separate memory space
- Cannot access other app's memory
- Communication only via Binder IPC
- Crash isolation (one app crash doesn't affect others)

### SELinux (Security-Enhanced Linux)

Android uses SELinux to enforce mandatory access control (MAC).

#### SELinux Modes

```bash
# Check SELinux mode
$ adb shell getenforce
Enforcing  # or Permissive

# SELinux can be:
# - Enforcing: Denials are blocked and logged
# - Permissive: Denials are logged but allowed (debugging)
```

#### Security Contexts

Every process, file, and object has an SELinux context:

```bash
# View process contexts
$ adb shell ps -Z
u:r:init:s0                root      1     0   /init
u:r:kernel:s0              root      2     0   kthreadd
u:r:system_server:s0       system    1234  567 system_server
u:r:untrusted_app:s0:c123  u0_a123   5678  567 com.example.app

# Context format: user:role:type:sensitivity[:categories]
# - user: SELinux user (usually 'u')
# - role: SELinux role (usually 'r')
# - type: Domain for processes, type for files
# - sensitivity: MLS/MCS level
# - categories: MLS/MCS categories (app isolation)
```

```bash
# View file contexts
$ adb shell ls -Z /system/bin/
u:object_r:system_file:s0           app_process64
u:object_r:surfaceflinger_exec:s0   surfaceflinger
u:object_r:zygote_exec:s0           app_process
```

#### Policy Rules

SELinux policies define what operations are allowed:

```
# Example policy (simplified)
# Allow system_server to use binder with apps
allow system_server untrusted_app:binder { call transfer };

# Allow apps to read system properties
allow untrusted_app property_type:file { read getattr };

# Deny apps from accessing kernel messages
neverallow untrusted_app kernel:system module_request;
```

**Domain Transitions**:
```
# When zygote forks app, domain transitions
# /system/bin/app_process (zygote context)
#   → executes app code
#   → transitions to untrusted_app context

type_transition zygote untrusted_app_exec:process untrusted_app;
```

#### SELinux Debugging

```bash
# View denials
$ adb shell dmesg | grep avc:
avc: denied { read write } for pid=1234 comm="app_process" \
  name="binder" dev="tmpfs" ino=1234 \
  scontext=u:r:untrusted_app:s0 \
  tcontext=u:object_r:device:s0 \
  tclass=chr_file permissive=0

# Analyze denials
$ adb shell audit2allow -i /dev/kmsg
# Generates policy to allow the operation
```

### Permission Model

Android has two permission types:

#### 1. Install-Time Permissions (Normal)

Automatically granted, low-risk permissions:
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.VIBRATE" />
```

#### 2. Runtime Permissions (Dangerous)

Require explicit user approval:

```java
// Check permission
if (ContextCompat.checkSelfPermission(this,
        Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

    // Request permission
    ActivityCompat.requestPermissions(this,
        new String[]{Manifest.permission.CAMERA},
        REQUEST_CAMERA_PERMISSION);
}

// Handle result
@Override
public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                       int[] grantResults) {
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        }
    }
}
```

**Permission Groups**:
```
CALENDAR      - READ_CALENDAR, WRITE_CALENDAR
CAMERA        - CAMERA
CONTACTS      - READ_CONTACTS, WRITE_CONTACTS, GET_ACCOUNTS
LOCATION      - ACCESS_FINE_LOCATION, ACCESS_COARSE_LOCATION
MICROPHONE    - RECORD_AUDIO
PHONE         - READ_PHONE_STATE, CALL_PHONE, READ_CALL_LOG, ...
SENSORS       - BODY_SENSORS
SMS           - SEND_SMS, RECEIVE_SMS, READ_SMS, ...
STORAGE       - READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
```

#### Protection Levels

```xml
<!-- normal: Low-risk, auto-granted -->
<permission android:name="com.example.permission.NORMAL"
    android:protectionLevel="normal" />

<!-- dangerous: Requires runtime request -->
<permission android:name="android.permission.CAMERA"
    android:protectionLevel="dangerous" />

<!-- signature: Only apps signed with same key -->
<permission android:name="com.example.permission.SIGNATURE"
    android:protectionLevel="signature" />

<!-- privileged: System apps in /system/priv-app -->
<permission android:name="android.permission.INSTALL_PACKAGES"
    android:protectionLevel="signature|privileged" />

<!-- development: Granted to development builds -->
<permission android:name="android.permission.SET_DEBUG_APP"
    android:protectionLevel="signature|development" />
```

#### Permission Enforcement

```java
// System service checks caller permission
public void performPrivilegedOperation() {
    // Throws SecurityException if caller lacks permission
    mContext.enforceCallingPermission(
        android.Manifest.permission.WRITE_SETTINGS,
        "Need WRITE_SETTINGS permission");

    // Or check manually
    if (mContext.checkCallingPermission(
            android.Manifest.permission.BLUETOOTH_ADMIN)
            != PackageManager.PERMISSION_GRANTED) {
        throw new SecurityException("Requires BLUETOOTH_ADMIN");
    }

    // Do privileged work
    doPrivilegedOperation();
}
```

### Verified Boot

Verified Boot ensures system integrity from bootloader to kernel to system partition.

#### Boot Chain of Trust

```
┌─────────────────────┐
│   Boot ROM          │ Verifies bootloader signature
│   (Hardware root)   │ with embedded public key
└──────────┬──────────┘
           ↓ (verified)
┌──────────┴──────────┐
│   Bootloader        │ Verifies boot/recovery partition
│   (Locked)          │ with embedded certificate
└──────────┬──────────┘
           ↓ (verified)
┌──────────┴──────────┐
│   Boot Image        │ Contains kernel + ramdisk
│   (Signed)          │ Verified by bootloader
└──────────┬──────────┘
           ↓ (verified)
┌──────────┴──────────┐
│   dm-verity         │ Verifies /system partition
│   (Kernel)          │ block-by-block at runtime
└─────────────────────┘
```

#### dm-verity

Device-mapper verity provides transparent block-level verification:

```bash
# dm-verity hash tree
$ adb shell cat /proc/mounts | grep dm
/dev/block/dm-0 /system ext4 ro,verity ...

# Verification in action
# 1. Each 4KB block has SHA-256 hash
# 2. Hashes stored in tree structure
# 3. Root hash signed and verified by bootloader
# 4. Any corruption detected → kernel panic or read error

# Verity metadata
$ adb shell cat /verity_key
<public key for /system verification>
```

**Hash Tree Structure**:
```
┌─────────────────────────┐
│   Root Hash (Signed)    │
└────────────┬────────────┘
             ↓
┌────────────┴────────────┐
│   Level 1 Hashes        │
│   [H1][H2][H3][H4]      │
└────┬───┬───┬───┬────────┘
     ↓   ↓   ↓   ↓
┌────┴───┴───┴───┴────────┐
│   Level 0 (Data Blocks) │
│   [B1][B2][B3][B4]...   │
└─────────────────────────┘
```

#### Verified Boot States

```bash
# Green: Fully verified, locked bootloader
# Yellow: Verified but bootloader unlocked
# Orange: Custom OS, unlocked bootloader
# Red: Verification failed

# Check bootloader status
$ adb shell getprop ro.boot.verifiedbootstate
green

$ adb shell getprop ro.boot.veritymode
enforcing
```

### KeyStore and Cryptography

Android KeyStore provides secure key storage and cryptographic operations.

#### Hardware-Backed KeyStore

```java
// Generate key in hardware-backed KeyStore
KeyGenerator keyGen = KeyGenerator.getInstance(
    KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");

KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(
        "my_key",
        KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
    .setUserAuthenticationRequired(true)  // Requires unlock
    .setUserAuthenticationValidityDurationSeconds(30)
    .setInvalidatedByBiometricEnrollment(true)  // Invalidate on fingerprint change
    .build();

keyGen.init(spec);
SecretKey key = keyGen.generateKey();

// Key never leaves secure hardware
// All crypto operations performed in TEE/SE
```

#### Trusted Execution Environment (TEE)

```
┌─────────────────────────────────────┐
│     Normal World (REE)              │
│  ┌──────────────────────────────┐   │
│  │  Android OS                  │   │
│  │  - Apps                      │   │
│  │  - Frameworks                │   │
│  │  - KeyStore API              │   │
│  └──────────┬───────────────────┘   │
└─────────────┼───────────────────────┘
              ↓ (Secure monitor call)
┌─────────────┴───────────────────────┐
│     Secure World (TEE)              │
│  ┌──────────────────────────────┐   │
│  │  Trusted OS (e.g., Trusty)   │   │
│  │  - KeyStore HAL              │   │
│  │  - Gatekeeper (PIN/pattern)  │   │
│  │  - Fingerprint HAL           │   │
│  │  - DRM                       │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  Secure Storage              │   │
│  │  - Encrypted keys            │   │
│  │  - Crypto operations         │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

**Key Features**:
- Keys generated and stored in TEE
- Private keys never exposed to Android OS
- Cryptographic operations in secure hardware
- Isolated from Android vulnerabilities

#### Attestation

Key attestation proves keys are hardware-backed:

```java
// Request attestation
KeyGenParameterSpec spec = new KeyGenParameterSpec.Builder(...)
    .setAttestationChallenge(challenge)  // Nonce from server
    .build();

// Get attestation certificate chain
KeyStore ks = KeyStore.getInstance("AndroidKeyStore");
ks.load(null);
Certificate[] chain = ks.getCertificateChain("my_key");

// Chain format:
// [0] Leaf cert - key attestation
// [1] Intermediate cert - TEE attestation
// [2] Root cert - OEM key

// Server verifies:
// 1. Chain validity
// 2. Challenge matches
// 3. Hardware attestation (vs software)
// 4. Device integrity
```

## Key Components (Enhanced)

### 1. Linux Kernel

The Linux kernel is the core of the Android operating system, providing essential system services.

**Key Subsystems**:

**Process Management**:
- CFS (Completely Fair Scheduler) for process scheduling
- Process groups (cgroups) for resource management
- Android-specific cpusets for big.LITTLE architectures

```bash
# View CPU sets
$ adb shell cat /dev/cpuset/foreground/cpus
0-7  # All cores for foreground apps

$ adb shell cat /dev/cpuset/background/cpus
0-3  # Only little cores for background
```

**Memory Management**:
- Page allocation and virtual memory
- Slab allocator for kernel objects
- ION allocator for shared buffers (camera, display)

**Hardware Drivers**:
- Display driver (framebuffer, DRM)
- Input driver (touchscreen, keyboard)
- Power management (suspend/resume)
- Thermal management
- GPU driver

**Android-Specific Kernel Features**:
```c
// Binder IPC driver
CONFIG_ANDROID_BINDER_IPC=y

// Ashmem (Anonymous Shared Memory)
CONFIG_ASHMEM=y

// Low Memory Killer
CONFIG_ANDROID_LOW_MEMORY_KILLER=y

// ION memory allocator
CONFIG_ION=y

// Process accounting
CONFIG_ANDROID_INTF_ALARM_DEV=y
```

### 2. Hardware Abstraction Layer (HAL)

The HAL provides a standard interface between Android framework and hardware-specific drivers.

**HAL Architecture**:

```
┌───────────────────────────────────┐
│  Android Framework (Java/Kotlin)  │
└───────────────┬───────────────────┘
                ↓ JNI
┌───────────────┴───────────────────┐
│  Framework Native (C++)           │
│  - CameraService                  │
│  - AudioFlinger                   │
└───────────────┬───────────────────┘
                ↓ dlopen
┌───────────────┴───────────────────┐
│  HAL Interface (HIDL/AIDL)        │
│  - Versioned interfaces           │
│  - Binder-based communication     │
└───────────────┬───────────────────┘
                ↓
┌───────────────┴───────────────────┐
│  HAL Implementation (.so)         │
│  - Vendor-specific code           │
│  - Hardware drivers interaction   │
└───────────────┬───────────────────┘
                ↓
┌───────────────┴───────────────────┐
│  Kernel Drivers                   │
└───────────────────────────────────┘
```

**Camera HAL Example**:

```cpp
// Camera HAL interface (HIDL)
// hardware/interfaces/camera/device/3.2/ICameraDevice.hal

interface ICameraDevice {
    open(ICameraDeviceCallback callback)
        generates (Status status, ICameraDeviceSession session);

    getCameraCharacteristics()
        generates (Status status, CameraMetadata chars);

    setTorchMode(TorchMode mode)
        generates (Status status);
};

// Vendor implementation
class CameraDeviceImpl : public ICameraDevice {
    Return<void> open(const sp<ICameraDeviceCallback>& callback,
                      open_cb _hidl_cb) override {
        // Open camera hardware
        camera3_device_t* device;
        int ret = mModule->open_device(mCameraId, &device);

        // Create session
        sp<ICameraDeviceSession> session =
            new CameraDeviceSession(device, callback);

        _hidl_cb(Status::OK, session);
        return Void();
    }
};
```

**Audio HAL Example**:

```cpp
// Audio HAL interface
// hardware/interfaces/audio/7.0/IDevice.hal

interface IDevice {
    openOutputStream(
        int32_t ioHandle,
        DeviceAddress device,
        AudioConfig config,
        AudioOutputFlags flags
    ) generates (
        Result result,
        IStreamOut stream,
        AudioConfig suggestedConfig
    );

    openInputStream(...) generates (...);
};

// Implementation handles hardware-specific audio routing
class AudioDeviceImpl : public IDevice {
    // Routes to ALSA, TinyALSA, or proprietary drivers
    // Handles DSP configuration, effects, routing
};
```

### 3. Android Runtime (ART)

ART is the managed runtime that executes app code.

**Compilation Pipeline**:

```
Java/Kotlin Source Code
        ↓ javac/kotlinc
Java Bytecode (.class)
        ↓ d8/r8 (dex compiler)
DEX Bytecode (.dex)
        ↓ dexopt/dex2oat (at install time)
┌──────────────────────────┐
│ OAT File (.odex)         │
│ - Native code (AOT)      │
│ - Quick code             │
│ - Metadata               │
└──────────────────────────┘
        ↓ (at runtime)
┌──────────────────────────┐
│ Execution                │
│ - AOT compiled code      │
│ - JIT for hot methods    │
│ - Interpreter (fallback) │
└──────────────────────────┘
```

**Compilation Modes**:

```java
// Speed: Full AOT compilation (largest code size, fastest)
$ pm compile -m speed com.example.app

// Speed-profile: AOT compile hot methods only (balanced)
$ pm compile -m speed-profile com.example.app

// Quicken: Optimized interpreter (fast install, slower execution)
$ pm compile -m quicken com.example.app

// Verify: No compilation, verification only
$ pm compile -m verify com.example.app

// Check compilation status
$ adb shell dumpsys package dexopt
```

**JIT (Just-In-Time) Compilation**:

```cpp
// ART JIT workflow
while (executing bytecode) {
    // Count method invocations
    if (method.hotness_count > JIT_THRESHOLD) {
        // Add to JIT queue
        jit_compiler.enqueue(method);
    }

    // Execute
    if (method.has_compiled_code()) {
        execute_native_code(method);
    } else {
        interpret(method);
    }
}

// Background JIT compilation thread
void JitCompiler::compile(ArtMethod* method) {
    // Compile to native code
    CompiledMethod* compiled = compiler.compile(method);

    // Add to profile for future AOT
    profile_saver.addHotMethod(method);

    // Install compiled code
    method->setEntryPoint(compiled->code);
}
```

**Garbage Collection**:

ART uses concurrent, generational GC:

```
┌──────────────────────────────────┐
│  Heap Organization               │
├──────────────────────────────────┤
│  Image Space (preloaded classes) │
│  - Read-only                     │
│  - Shared across processes       │
├──────────────────────────────────┤
│  Zygote Space (zygote objects)   │
│  - Shared until written (COW)    │
├──────────────────────────────────┤
│  Allocation Space (new objects)  │
│  - Young generation (nursery)    │
│  - Old generation                │
├──────────────────────────────────┤
│  Large Object Space (>12KB)      │
│  - Directly allocated            │
└──────────────────────────────────┘
```

**GC Types**:
```cpp
// Concurrent mark-sweep (background GC)
void ConcurrentMarkSweep() {
    // 1. Mark roots (pause)
    mark_roots();

    // 2. Mark live objects (concurrent)
    concurrent_mark();

    // 3. Remark (short pause)
    remark();

    // 4. Sweep dead objects (concurrent)
    concurrent_sweep();
}

// Generational GC (young generation)
void MinorGC() {
    // Fast collection of young objects
    // Most objects die young (generational hypothesis)
}

// Compacting GC (reduces fragmentation)
void CompactingGC() {
    // Move live objects together
    // Update all references
    // Reduces memory fragmentation
}
```

### 4. Native C/C++ Libraries

Enhanced details on key native libraries:

**Bionic**:
- Android's C library (libc), optimized for mobile
- Smaller than glibc (~200KB vs ~2MB)
- Fast property access via `__system_property_get()`
- DNS resolution with multi-network support

**SurfaceFlinger**:
```cpp
// Composites all window surfaces
class SurfaceFlinger : public BnSurfaceComposer {
    void onVsync() {
        // 1. Latch new buffers from clients
        for (Layer* layer : mLayers) {
            layer->latchBuffer();
        }

        // 2. Compute layer visibility and transforms
        computeVisibleRegions();

        // 3. Composite layers (GPU or HWC)
        if (hwc_available) {
            hwc->prepare(mLayers);
            hwc->commit();
        } else {
            gpu_composite(mLayers);
        }

        // 4. Post framebuffer
        postFramebuffer();
    }
};
```

**Media Framework**:
- Stagefright (media playback engine)
- MediaCodec API (hardware encoding/decoding)
- OpenMAX IL (codec interface)

```cpp
// MediaCodec architecture
App → MediaCodec (Java)
        ↓ JNI
      MediaCodec (C++)
        ↓ Binder
      MediaCodecService
        ↓
      Codec2 or OMX
        ↓
      Hardware Codec (HAL)
```

### 5. Application Framework

Enhanced details on framework services:

**Activity Manager Service**:
```java
// Manages app lifecycle
class ActivityManagerService {
    // Start activity
    int startActivity(Intent intent, String callingPackage) {
        // 1. Resolve intent
        ActivityInfo ai = resolveActivity(intent);

        // 2. Check permissions
        checkPermissions(ai);

        // 3. Start process if needed
        if (!isProcessRunning(ai.processName)) {
            startProcessLocked(ai.processName, ai.uid);
        }

        // 4. Create activity record
        ActivityRecord r = new ActivityRecord(ai, intent);

        // 5. Add to task stack
        mTaskStack.addActivity(r);

        // 6. Resume activity
        resumeActivity(r);
    }

    // Process lifecycle
    void updateOomAdjLocked() {
        // Calculate OOM scores for all processes
        for (ProcessRecord app : mProcessList) {
            int oomAdj = computeOomAdj(app);
            setOomAdj(app.pid, oomAdj);
        }
    }
}
```

**Window Manager Service**:
```java
// Manages window layout and focus
class WindowManagerService {
    void relayoutWindow(Session session, IWindow client,
                        WindowManager.LayoutParams attrs) {
        // 1. Find window state
        WindowState win = windowForClientLocked(client);

        // 2. Apply layout parameters
        win.mAttrs = attrs;

        // 3. Compute layout
        performLayoutAndPlaceSurfacesLocked();

        // 4. Update surface
        win.prepareSurfaceLocked();
    }

    // Input focus
    void setFocusedWindow(WindowState win) {
        if (mCurrentFocus != win) {
            mCurrentFocus = win;
            mInputMonitor.setInputFocusLw(win);
        }
    }
}
```

**Package Manager Service**:
```java
// Manages app installation and package info
class PackageManagerService {
    void installPackage(String apkPath, int uid) {
        // 1. Parse APK (AndroidManifest.xml)
        Package pkg = parsePackage(apkPath);

        // 2. Verify signatures
        verifySignatures(pkg);

        // 3. Check permissions
        grantPermissions(pkg);

        // 4. dexopt (compile to native code)
        mDexOptimizer.performDexOpt(pkg);

        // 5. Create app data directory
        createDataDirectory(pkg, uid);

        // 6. Register package
        mPackages.put(pkg.packageName, pkg);
    }
}
```

### 6. System Applications

System apps provide core functionality:

**Launcher**:
- Home screen
- App drawer
- Widgets
- Icon management

**SystemUI**:
- Status bar
- Quick settings
- Notifications
- Navigation bar
- Lock screen

**Settings**:
- Configuration UI
- Manages SharedPreferences
- Communicates with system services

## Conclusion

Android's architecture is a sophisticated stack built on the Linux kernel, with multiple layers providing abstraction, security, and performance. Understanding these internals—from the boot process and Binder IPC to memory management and security mechanisms—is essential for developers who want to:

- Build high-performance applications
- Debug complex issues
- Contribute to AOSP (Android Open Source Project)
- Develop system-level services
- Optimize for specific hardware

Key takeaways:
- **Boot process** involves multiple stages with verification at each level
- **Binder IPC** provides efficient, secure inter-process communication
- **Memory management** uses aggressive reclamation for constrained environments
- **Security** employs defense-in-depth: sandboxing, SELinux, permissions, verified boot
- **ART runtime** uses AOT, JIT, and advanced GC for performance
- **HAL** enables hardware abstraction while maintaining stability

This deep technical knowledge empowers developers to create optimized, secure, and efficient Android applications.
