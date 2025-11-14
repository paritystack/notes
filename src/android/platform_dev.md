# Android Platform Development

## Overview

Android platform development involves building, modifying, and customizing the Android operating system itself (AOSP - Android Open Source Project), as opposed to developing applications that run on Android. This includes working with framework code, system services, HAL implementations, and the Linux kernel.

### Platform Development vs App Development

| Aspect | App Development | Platform Development |
|--------|----------------|---------------------|
| **Scope** | Single application | Entire OS and framework |
| **Language** | Kotlin/Java | Java, C++, C, Go, Rust |
| **Build System** | Gradle | Soong/Blueprint (Android.bp) |
| **Output** | APK/AAB | System image (system.img, boot.img) |
| **Tools** | Android Studio | Repo, Soong, ADB, Fastboot |
| **Testing** | Emulator/Device | AOSP Emulator, Physical device flashing |
| **Distribution** | Google Play | Custom ROM, OEM builds |

### When You Need Platform Development

- Building custom ROMs (LineageOS, GrapheneOS, etc.)
- OEM device customization
- Adding system-level features
- Implementing hardware support (HAL)
- Security research and hardening
- Contributing to AOSP
- Embedded Android systems

## Environment Setup

### System Requirements

**Hardware:**
- 64-bit CPU
- 400GB+ free disk space (SSD recommended)
- 64GB+ RAM (128GB recommended for full builds)
- Fast internet connection

**Operating System:**
- Ubuntu 22.04 LTS (recommended)
- Debian 11+
- macOS (limited support)

### Installing Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    git-core gnupg flex bison build-essential zip curl zlib1g-dev \
    libc6-dev-i386 libncurses5 lib32ncurses5-dev x11proto-core-dev \
    libx11-dev lib32z1-dev libgl1-mesa-dev libxml2-utils xsltproc \
    unzip fontconfig python3 python3-pip rsync bc schedtool lzop \
    imagemagick libssl-dev repo ccache adb fastboot

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Install repo tool (alternative method)
mkdir -p ~/bin
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
export PATH=~/bin:$PATH
```

### Downloading AOSP Source

```bash
# Create working directory
mkdir -p ~/aosp
cd ~/aosp

# Initialize repo for specific Android version
# Android 14
repo init -u https://android.googlesource.com/platform/manifest -b android-14.0.0_r1

# Or latest master
repo init -u https://android.googlesource.com/platform/manifest -b master

# Sync source code (this takes hours)
repo sync -c -j$(nproc) --force-sync --no-clone-bundle --no-tags

# For faster subsequent syncs
repo sync -c -j$(nproc) --force-sync
```

### Setting Up Build Environment

```bash
# Source build environment
cd ~/aosp
source build/envsetup.sh

# This adds important commands:
# - lunch: Select build target
# - m: Build from top of tree
# - mm: Build current directory
# - mma: Build current directory and dependencies
# - mmm: Build specific directory
# - croot: Change to repo root
# - cgrep: Search C/C++ files
# - jgrep: Search Java files
```

## AOSP Directory Structure

Understanding the source tree layout is crucial:

```
aosp/
├── art/                    # Android Runtime (ART)
├── bionic/                 # C library, math, and dynamic linker
├── bootable/              # Boot and recovery related
│   └── recovery/          # Recovery mode implementation
├── build/                 # Build system
│   ├── soong/            # Soong build system (Go)
│   └── make/             # Legacy Make-based build
├── cts/                   # Compatibility Test Suite
├── dalvik/                # Dalvik VM (legacy)
├── developers/            # Sample apps and docs
├── development/           # Development tools
├── device/                # Device-specific configurations
│   ├── google/           # Google devices
│   └── [vendor]/[device]/ # Device configurations
├── external/              # External projects and libraries
│   ├── chromium-webview/
│   ├── sqlite/
│   └── ...
├── frameworks/            # Android framework
│   ├── base/             # Core framework (services, APIs)
│   ├── native/           # Native framework libraries
│   ├── av/               # Audio/Video framework
│   └── opt/              # Optional frameworks
├── hardware/              # HAL definitions and implementations
│   ├── interfaces/       # HIDL/AIDL interface definitions
│   ├── libhardware/      # Legacy HAL
│   └── [vendor]/         # Vendor HAL implementations
├── kernel/                # Kernel source (if included)
├── packages/              # System packages and apps
│   ├── apps/             # System apps (Settings, Dialer, etc.)
│   ├── services/         # System services
│   └── providers/        # Content providers
├── pdk/                   # Platform Development Kit
├── platform_testing/      # Platform tests
├── prebuilts/            # Prebuilt binaries and tools
│   ├── sdk/
│   └── gcc/
├── sdk/                   # SDK source
├── system/                # Core system components
│   ├── core/             # Init, toolbox, debuggerd
│   ├── bt/               # Bluetooth stack
│   ├── netd/             # Network daemon
│   └── vold/             # Volume daemon
├── toolchain/            # Toolchain utilities
├── tools/                # Development tools
├── vendor/               # Vendor-specific code
│   └── [vendor]/         # Vendor proprietary code
└── out/                  # Build output (created during build)
    └── target/
        └── product/
            └── [device]/  # Built images
```

## Building Android Platform

### Selecting Build Target

```bash
# Source environment
source build/envsetup.sh

# List available targets
lunch

# Common targets:
# aosp_arm64-eng          - Generic ARM64, engineering build
# aosp_x86_64-eng         - Generic x86_64, engineering build
# aosp_cf_x86_64_phone-userdebug  - Cuttlefish virtual device

# Select target
lunch aosp_x86_64-eng
```

### Build Variants

- **eng**: Engineering builds
  - Debug enabled
  - Root access
  - Additional debugging tools
  - Not optimized

- **userdebug**: User debug builds
  - Root access available via adb
  - Production-like but debuggable
  - Most common for development

- **user**: Production builds
  - No root access
  - Optimized for performance
  - Production release configuration

### Building the Platform

```bash
# Full build (clean build)
m -j$(nproc)

# This typically takes 1-6 hours depending on hardware

# Incremental build (after changes)
m -j$(nproc)

# Build specific module
m [module_name]

# Examples:
m SystemUI              # Build System UI
m framework            # Build framework
m services             # Build system services

# Build with verbose output
m showcommands

# Clean build
m clean                # Clean all
m installclean        # Clean installed files (faster)
```

### Build Output

After successful build, images are located in:
```
out/target/product/[device]/
├── system.img          # System partition
├── vendor.img          # Vendor partition
├── boot.img           # Boot image (kernel + ramdisk)
├── userdata.img       # User data partition
├── recovery.img       # Recovery image
├── vbmeta.img         # Verified boot metadata
└── [device]-img.zip   # Flashable images package
```

## Common Platform Development Patterns

### 1. Adding a System Service

System services are core platform components that provide functionality to apps.

#### Define Service Interface

```java
// frameworks/base/core/java/android/os/IMyService.aidl
package android.os;

/** {@hide} */
interface IMyService {
    String getMessage();
    void setMessage(String message);
}
```

#### Implement Service

```java
// frameworks/base/services/core/java/com/android/server/MyService.java
package com.android.server;

import android.content.Context;
import android.os.IMyService;

public class MyService extends IMyService.Stub {
    private final Context mContext;
    private String mMessage = "Default message";

    public MyService(Context context) {
        mContext = context;
    }

    @Override
    public String getMessage() {
        return mMessage;
    }

    @Override
    public void setMessage(String message) {
        mMessage = message;
        // You might want to persist this
    }

    public void onStart() {
        // Service initialization
    }
}
```

#### Register Service

```java
// frameworks/base/services/java/com/android/server/SystemServer.java

import com.android.server.MyService;

private void startOtherServices() {
    // ... existing services ...

    MyService myService = null;
    try {
        Slog.i(TAG, "My Service");
        myService = new MyService(context);
        ServiceManager.addService("myservice", myService);
    } catch (Throwable e) {
        reportWtf("starting My Service", e);
    }

    // ... more services ...
}
```

#### Create Client API

```java
// frameworks/base/core/java/android/os/MyManager.java
package android.os;

import android.annotation.SystemService;
import android.content.Context;

@SystemService(Context.MY_SERVICE)
public class MyManager {
    private final IMyService mService;

    /** {@hide} */
    public MyManager(Context context, IMyService service) {
        mService = service;
    }

    public String getMessage() {
        try {
            return mService.getMessage();
        } catch (RemoteException e) {
            throw e.rethrowFromSystemServer();
        }
    }

    public void setMessage(String message) {
        try {
            mService.setMessage(message);
        } catch (RemoteException e) {
            throw e.rethrowFromSystemServer();
        }
    }
}
```

#### Register in Context

```java
// frameworks/base/core/java/android/app/SystemServiceRegistry.java

import android.os.MyManager;

static {
    // ... existing service registrations ...

    registerService(Context.MY_SERVICE, MyManager.class,
        new CachedServiceFetcher<MyManager>() {
            @Override
            public MyManager createService(ContextImpl ctx) {
                IBinder b = ServiceManager.getService(Context.MY_SERVICE);
                IMyService service = IMyService.Stub.asInterface(b);
                return new MyManager(ctx, service);
            }
        });
}
```

### 2. Adding System API

System APIs are hidden APIs accessible only to system apps.

```java
// frameworks/base/core/java/android/os/SystemProperties.java

/**
 * Get system property value
 * @hide
 */
@SystemApi
public static String getSystemProperty(String key, String defaultValue) {
    return SystemProperties.get(key, defaultValue);
}
```

Mark in API definition:
```java
/**
 * @hide
 */
@SystemApi
@RequiresPermission(android.Manifest.permission.READ_PRIVILEGED_PHONE_STATE)
public void privilegedMethod() {
    // Implementation
}
```

### 3. Adding System Permissions

#### Define Permission

```xml
<!-- frameworks/base/core/res/AndroidManifest.xml -->

<permission
    android:name="android.permission.MY_CUSTOM_PERMISSION"
    android:protectionLevel="signature|privileged"
    android:label="@string/permlab_myCustomPermission"
    android:description="@string/permdesc_myCustomPermission" />
```

Protection levels:
- **normal**: Low-risk, granted automatically
- **dangerous**: Runtime permission required
- **signature**: Only apps signed with same key
- **privileged**: System apps in priv-app
- **signature|privileged**: Signature OR privileged

#### Check Permission in Service

```java
// frameworks/base/services/core/java/com/android/server/MyService.java

public void privilegedOperation() {
    mContext.enforceCallingOrSelfPermission(
        android.Manifest.permission.MY_CUSTOM_PERMISSION,
        "Requires MY_CUSTOM_PERMISSION");

    // Perform operation
}
```

### 4. Modifying Framework Behavior

Example: Modifying ActivityManagerService

```java
// frameworks/base/services/core/java/com/android/server/am/ActivityManagerService.java

public class ActivityManagerService extends IActivityManager.Stub {

    // Add custom behavior
    @Override
    public void startActivity(/* parameters */) {
        // Custom pre-processing
        Slog.d(TAG, "Custom: Starting activity");

        // Original logic
        super.startActivity(/* parameters */);

        // Custom post-processing
        notifyCustomListeners();
    }
}
```

### 5. Adding SELinux Policies

SELinux (Security-Enhanced Linux) controls access between processes.

#### Define SELinux Type

```te
# system/sepolicy/private/myservice.te

type myservice, domain;
type myservice_exec, exec_type, file_type;

# Allow myservice to be started by init
init_daemon_domain(myservice)

# Allow reading/writing specific files
allow myservice system_data_file:dir rw_dir_perms;
allow myservice system_data_file:file create_file_perms;

# Allow binder communication
binder_use(myservice)
binder_call(myservice, system_server)
```

#### File Context Labeling

```
# system/sepolicy/private/file_contexts

/system/bin/myservice    u:object_r:myservice_exec:s0
/data/system/myservice(/.*)? u:object_r:system_data_file:s0
```

### 6. Hardware Abstraction Layer (HAL)

Modern HALs use HIDL (Hardware Interface Definition Language) or AIDL.

#### Define HIDL Interface

```java
// hardware/interfaces/mydevice/1.0/IMyDevice.hal

package android.hardware.mydevice@1.0;

interface IMyDevice {
    /**
     * Initialize device
     * @return status Operation status
     */
    initialize() generates (Status status);

    /**
     * Read data from device
     * @return status Operation status
     * @return data Data read from device
     */
    readData() generates (Status status, vec<uint8_t> data);

    /**
     * Write data to device
     * @param data Data to write
     * @return status Operation status
     */
    writeData(vec<uint8_t> data) generates (Status status);
};
```

#### Implement HAL

```cpp
// hardware/interfaces/mydevice/1.0/default/MyDevice.cpp

#include <android/hardware/mydevice/1.0/IMyDevice.h>

namespace android::hardware::mydevice::V1_0::implementation {

class MyDevice : public IMyDevice {
public:
    Return<Status> initialize() override {
        // Initialize hardware
        return Status::OK;
    }

    Return<void> readData(readData_cb _hidl_cb) override {
        std::vector<uint8_t> data;
        // Read from hardware
        _hidl_cb(Status::OK, data);
        return Void();
    }

    Return<Status> writeData(const hidl_vec<uint8_t>& data) override {
        // Write to hardware
        return Status::OK;
    }
};

} // namespace
```

## Build System (Soong)

Android uses Soong build system defined in `Android.bp` files.

### Basic Android.bp Structure

```bp
// Android.bp

// Java library
java_library {
    name: "my-library",
    srcs: ["src/**/*.java"],
    sdk_version: "current",

    static_libs: [
        "androidx.core_core",
    ],

    libs: [
        "framework",
    ],
}

// System app
android_app {
    name: "MySystemApp",
    srcs: ["src/**/*.java"],
    resource_dirs: ["res"],
    manifest: "AndroidManifest.xml",

    platform_apis: true,
    certificate: "platform",
    privileged: true,

    static_libs: [
        "my-library",
    ],
}

// Native library
cc_library_shared {
    name: "libmynative",
    srcs: ["native/*.cpp"],

    shared_libs: [
        "liblog",
        "libutils",
    ],

    cflags: [
        "-Wall",
        "-Werror",
    ],
}

// Native binary
cc_binary {
    name: "myservice",
    srcs: ["service/*.cpp"],

    shared_libs: [
        "libmynative",
        "libbinder",
    ],

    init_rc: ["myservice.rc"],
}

// Prebuilt APK
android_app_import {
    name: "PrebuiltApp",
    apk: "prebuilt/app.apk",
    certificate: "PRESIGNED",
    privileged: true,
}
```

### Module Types

| Type | Purpose | Example |
|------|---------|---------|
| `java_library` | Java library | Framework libraries |
| `android_app` | Android application | System apps |
| `android_app_import` | Prebuilt APK | Vendor apps |
| `cc_library` | Native library | libutils |
| `cc_binary` | Native executable | surfaceflinger |
| `cc_library_shared` | Shared native library | .so files |
| `cc_library_static` | Static native library | .a files |
| `prebuilt_etc` | Config/data files | init scripts |
| `sh_binary` | Shell script | Utility scripts |

### Common Build Properties

```bp
android_app {
    name: "MyApp",

    // Source files
    srcs: ["src/**/*.java"],
    resource_dirs: ["res"],
    manifest: "AndroidManifest.xml",

    // SDK version
    platform_apis: true,  // or sdk_version: "current"
    min_sdk_version: "30",

    // Signing
    certificate: "platform",  // platform, shared, media, testkey

    // Privileges
    privileged: true,  // Install in /system/priv-app
    system_ext_specific: true,  // Install in /system_ext
    product_specific: true,  // Install in /product
    vendor: true,  // Install in /vendor

    // Dependencies
    static_libs: ["lib1", "lib2"],
    libs: ["framework"],

    // Optimization
    optimize: {
        enabled: true,
        shrink: true,
        obfuscate: false,
    },

    // Overrides (replaces existing app)
    overrides: ["OriginalApp"],
}
```

### Building Specific Modules

```bash
# Build specific module
m MySystemApp

# Build and install to device
m MySystemApp && adb sync

# Build all modules in current directory
mm

# Build module and dependencies
mma

# Build modules in specific directory
mmm frameworks/base/packages/SystemUI/
```

## Device Configuration

### Device Makefile Structure

```
device/manufacturer/codename/
├── Android.mk
├── AndroidProducts.mk
├── BoardConfig.mk
├── device.mk
├── [codename].mk
├── system.prop
├── vendor.prop
├── proprietary-files.txt
├── extract-files.sh
├── setup-makefiles.sh
├── overlay/           # Runtime resource overlays
├── sepolicy/         # Device-specific SELinux policies
├── init/            # Init scripts
│   ├── init.device.rc
│   └── ueventd.device.rc
├── configs/         # Hardware configs
│   ├── audio/
│   ├── wifi/
│   └── media/
└── kernel/          # Kernel build config
```

### device.mk Example

```makefile
# device/manufacturer/codename/device.mk

# Inherit from common device config
$(call inherit-product, $(SRC_TARGET_DIR)/product/core_64_bit.mk)
$(call inherit-product, $(SRC_TARGET_DIR)/product/full_base_telephony.mk)

# Device identifier
PRODUCT_NAME := aosp_codename
PRODUCT_DEVICE := codename
PRODUCT_BRAND := Manufacturer
PRODUCT_MODEL := Device Model
PRODUCT_MANUFACTURER := manufacturer

# Build properties
PRODUCT_PROPERTY_OVERRIDES += \
    ro.build.fingerprint=custom-fingerprint \
    ro.product.board=codename

# Packages to include
PRODUCT_PACKAGES += \
    SystemUI \
    Settings \
    Launcher3 \
    MyCustomApp

# Copy device-specific files
PRODUCT_COPY_FILES += \
    $(LOCAL_PATH)/configs/audio/audio_policy.conf:$(TARGET_COPY_OUT_VENDOR)/etc/audio_policy.conf \
    $(LOCAL_PATH)/init/init.device.rc:$(TARGET_COPY_OUT_VENDOR)/etc/init/init.device.rc

# Vendor partition
$(call inherit-product, vendor/manufacturer/codename/codename-vendor.mk)
```

### BoardConfig.mk Example

```makefile
# device/manufacturer/codename/BoardConfig.mk

# Architecture
TARGET_ARCH := arm64
TARGET_ARCH_VARIANT := armv8-a
TARGET_CPU_ABI := arm64-v8a
TARGET_CPU_VARIANT := generic

TARGET_2ND_ARCH := arm
TARGET_2ND_ARCH_VARIANT := armv7-a-neon
TARGET_2ND_CPU_ABI := armeabi-v7a
TARGET_2ND_CPU_VARIANT := generic

# Bootloader
TARGET_BOOTLOADER_BOARD_NAME := codename
TARGET_NO_BOOTLOADER := true

# Kernel
BOARD_KERNEL_CMDLINE := console=ttyMSM0,115200n8
BOARD_KERNEL_BASE := 0x80000000
BOARD_KERNEL_PAGESIZE := 4096
TARGET_PREBUILT_KERNEL := $(LOCAL_PATH)/kernel
# Or build from source:
# TARGET_KERNEL_SOURCE := kernel/manufacturer/codename
# TARGET_KERNEL_CONFIG := codename_defconfig

# Partitions
BOARD_FLASH_BLOCK_SIZE := 131072
BOARD_BOOTIMAGE_PARTITION_SIZE := 67108864
BOARD_SYSTEMIMAGE_PARTITION_SIZE := 3221225472
BOARD_USERDATAIMAGE_PARTITION_SIZE := 10737418240
BOARD_VENDORIMAGE_PARTITION_SIZE := 1073741824

# Filesystem
TARGET_USERIMAGES_USE_EXT4 := true
TARGET_USERIMAGES_USE_F2FS := true
BOARD_VENDORIMAGE_FILE_SYSTEM_TYPE := ext4

# SELinux
BOARD_SEPOLICY_DIRS += $(LOCAL_PATH)/sepolicy/vendor
SELINUX_IGNORE_NEVERALLOWS := true

# Verified Boot
BOARD_AVB_ENABLE := true
BOARD_AVB_MAKE_VBMETA_IMAGE_ARGS += --flag 2
```

## Testing and Debugging

### Running on Emulator

```bash
# Build emulator target
lunch aosp_x86_64-eng
m -j$(nproc)

# Run emulator
emulator

# Or with specific options
emulator -memory 4096 -cores 4 -gpu host
```

### Flashing Physical Device

```bash
# Boot into bootloader
adb reboot bootloader

# Unlock bootloader (if needed, wipes data)
fastboot flashing unlock

# Flash images
fastboot flashall -w

# Or flash individually
fastboot flash boot out/target/product/[device]/boot.img
fastboot flash system out/target/product/[device]/system.img
fastboot flash vendor out/target/product/[device]/vendor.img

# Reboot
fastboot reboot

# Lock bootloader (optional)
fastboot flashing lock
```

### Debugging Platform Code

```bash
# View logs
adb logcat

# Filter by tag
adb logcat -s MyService

# View system server logs
adb logcat | grep SystemServer

# View kernel logs
adb shell dmesg

# Interactive debugging with GDB
gdbclient -p [process-name]

# Trace system calls
adb shell strace -p [pid]
```

### Testing Changes

```bash
# Build and sync to device
m -j$(nproc) && adb sync

# Sync only system partition
adb sync system

# Restart specific service
adb shell stop [service]
adb shell start [service]

# Restart system server (careful - disruptive)
adb shell stop
adb shell start

# Reboot to recovery
adb reboot recovery

# Reboot to bootloader
adb reboot bootloader
```

## Common Operations

### Adding System Application

1. **Create app module**:
```
packages/apps/MySystemApp/
├── Android.bp
├── AndroidManifest.xml
├── src/
└── res/
```

2. **Define in Android.bp**:
```bp
android_app {
    name: "MySystemApp",
    srcs: ["src/**/*.java"],
    resource_dirs: ["res"],
    manifest: "AndroidManifest.xml",
    platform_apis: true,
    certificate: "platform",
    privileged: true,
}
```

3. **Add to device.mk**:
```makefile
PRODUCT_PACKAGES += MySystemApp
```

### Creating System Service

Follow the pattern in "Adding a System Service" section, then:

```bash
# Build framework
m framework services

# Sync to device
adb sync system

# Restart system
adb shell stop && adb shell start
```

### Modifying System Properties

```bash
# Edit system properties
vim device/manufacturer/codename/system.prop

# Add properties:
# ro.my.property=value
# persist.my.property=value

# Rebuild and flash
m -j$(nproc)
fastboot flashall
```

### Creating OTA Package

```bash
# Build target files
m target-files-package

# Generate OTA
./build/tools/releasetools/ota_from_target_files \
    out/target/product/[device]/obj/PACKAGING/target_files_intermediates/aosp_[device]-target_files.zip \
    ota-package.zip

# Apply OTA
adb sideload ota-package.zip
```

### Enabling Root in User Builds

```bash
# Edit device.mk
PRODUCT_PROPERTY_OVERRIDES += \
    ro.secure=0 \
    ro.debuggable=1 \
    ro.adb.secure=0

# Or use userdebug/eng variants
lunch aosp_[device]-userdebug
```

## Advanced Topics

### Project Treble

Treble separates vendor implementation from Android framework:

```
┌─────────────────────────────┐
│   Android Framework         │  /system
├─────────────────────────────┤
│   VNDK (Vendor NDK)         │  ABI stability
├─────────────────────────────┤
│   Vendor Implementation     │  /vendor
│   (HALs, Drivers)           │
└─────────────────────────────┘
```

**Key Concepts:**
- **VNDK**: Vendor NDK libraries with stable ABI
- **Vendor Interface**: HIDL/AIDL HALs
- **Partition Separation**: /system vs /vendor

### Generic Kernel Image (GKI)

Android 11+ uses GKI for vendor-independent kernel:

```bash
# GKI structure
kernel/
├── common/              # GKI kernel
└── common-modules/     # Loadable kernel modules

# Build GKI
cd kernel/common
BUILD_CONFIG=common/build.config.gki.aarch64 build/build.sh
```

### Mainline Modules (APEX)

Updatable system components delivered via Google Play:

```bp
// APEX module definition
apex {
    name: "com.android.mymodule",
    manifest: "manifest.json",
    file_contexts: "file_contexts",
    key: "com.android.mymodule.key",
    certificate: ":com.android.mymodule.certificate",

    java_libs: [
        "mymodule-lib",
    ],

    prebuilts: [
        "mymodule-config",
    ],
}
```

### Vendor APEX

```json
// manifest.json
{
  "name": "com.android.mymodule",
  "version": 1
}
```

### CTS (Compatibility Test Suite)

```bash
# Run CTS
cd android-cts/tools
./cts-tradefed

# Run specific test
run cts -m CtsPermissionTestCases

# Run VTS (Vendor Test Suite)
./vts-tradefed
run vts
```

## Development Workflow

### Making Changes

```bash
# 1. Create topic branch
repo start my-feature .

# 2. Make changes
vim frameworks/base/core/java/android/app/Activity.java

# 3. Build and test
m -j$(nproc)
adb sync
# Test changes

# 4. Commit
git add -A
git commit -m "Add feature X

Bug: 123456
Test: Manual testing on device
Change-Id: I..."
```

### Code Review (Gerrit)

```bash
# Upload for review
repo upload .

# Or with git
git push ssh://[user]@android-review.googlesource.com:29418/platform/frameworks/base HEAD:refs/for/master

# Amend and re-upload
git commit --amend
repo upload .
```

### Syncing Updates

```bash
# Sync all projects
repo sync -c -j$(nproc)

# Sync specific project
repo sync frameworks/base

# Rebase local changes
repo rebase
```

## Performance Optimization

### Build Performance

```bash
# Use ccache
export USE_CCACHE=1
export CCACHE_DIR=~/.ccache
ccache -M 100G

# Parallel builds
m -j$(nproc)

# Incremental builds
m installclean  # Instead of clean
```

### Runtime Performance

```java
// Optimize framework code
public class MyService {
    // Use object pools
    private final Pools.SynchronizedPool<Message> mPool
        = new Pools.SynchronizedPool<>(10);

    // Avoid allocations in hot paths
    private final ArrayList<Item> mReusableList = new ArrayList<>();

    public void processItems() {
        mReusableList.clear();
        // Reuse list instead of creating new
    }
}
```

### Memory Optimization

```java
// Use weak references for callbacks
private final WeakHashMap<Context, Callback> mCallbacks
    = new WeakHashMap<>();

// Clear large data structures
@Override
public void onDestroy() {
    mLargeCache.clear();
    mBitmapCache.clear();
}
```

## Best Practices

### Code Style

- Follow [AOSP Java Code Style](https://source.android.com/docs/setup/contribute/code-style)
- Use 4 spaces for indentation
- Line length: 100 characters
- Use meaningful variable names

### Security

```java
// Always validate input
public void handleData(String input) {
    if (input == null || input.isEmpty()) {
        throw new IllegalArgumentException("Invalid input");
    }

    // Sanitize before use
    String sanitized = input.replaceAll("[^a-zA-Z0-9]", "");
}

// Check permissions
mContext.enforceCallingPermission(
    android.Manifest.permission.MY_PERMISSION,
    "Requires MY_PERMISSION");

// Use SELinux policies
// Define in sepolicy/
```

### Logging

```java
import android.util.Slog;

// System server logging
Slog.d(TAG, "Debug message");
Slog.i(TAG, "Info message");
Slog.w(TAG, "Warning message");
Slog.e(TAG, "Error message");

// App logging
android.util.Log.d(TAG, "Message");

// Conditional logging
if (DEBUG) {
    Slog.v(TAG, "Verbose debug info");
}
```

### Error Handling

```java
try {
    riskyOperation();
} catch (Exception e) {
    Slog.e(TAG, "Operation failed", e);
    // Report to system
    if (mContext != null) {
        mContext.sendBroadcast(new Intent(ACTION_ERROR));
    }
}
```

## Troubleshooting

### Build Issues

```bash
# Clean build
make clean
m -j$(nproc)

# Check dependencies
m modules
m nothing

# Fix repo state
repo forall -c 'git reset --hard'
repo sync -j$(nproc)
```

### Boot Issues

```bash
# Check boot logs
adb wait-for-device
adb logcat -b all

# Kernel logs
adb shell dmesg

# Check SELinux denials
adb shell cat /sys/fs/selinux/enforce
adb logcat | grep avc:
```

### Runtime Issues

```bash
# Check system services
adb shell dumpsys

# Specific service
adb shell dumpsys activity
adb shell dumpsys package

# Check crashes
adb shell cat /data/tombstones/tombstone_*

# Memory info
adb shell dumpsys meminfo
```

## Resources

### Official Documentation
- [AOSP Source](https://source.android.com/)
- [Building Android](https://source.android.com/docs/setup/build/building)
- [Platform Architecture](https://source.android.com/docs/core/architecture)
- [Treble](https://source.android.com/docs/core/architecture/treble)
- [SELinux](https://source.android.com/docs/security/features/selinux)

### Tools
- [repo](https://source.android.com/docs/setup/download#installing-repo) - Version control
- [Soong](https://source.android.com/docs/setup/build) - Build system
- [Gerrit](https://android-review.googlesource.com/) - Code review

### Related Documentation
- [Android Internals](internals.md) - System architecture
- [ADB Commands](adb.md) - Debug bridge reference
- [Binder IPC](binder.md) - Inter-process communication

## Quick Reference

### Essential Commands

```bash
# Setup
repo init -u https://android.googlesource.com/platform/manifest -b master
repo sync -c -j$(nproc)
source build/envsetup.sh
lunch aosp_x86_64-eng

# Build
m -j$(nproc)                    # Full build
m [module]                      # Build module
mm                              # Build current directory
mmm path/to/module/            # Build specific path

# Flash
adb reboot bootloader
fastboot flashall -w

# Debug
adb logcat
adb shell dumpsys
adb sync

# Navigation
croot                          # Go to repo root
cgrep [pattern]               # Search C/C++ files
jgrep [pattern]               # Search Java files
```

### Key Directories

| Path | Description |
|------|-------------|
| `frameworks/base/` | Android framework |
| `frameworks/base/services/` | System services |
| `system/core/` | Core system components |
| `packages/apps/` | System applications |
| `hardware/interfaces/` | HAL definitions |
| `device/` | Device configurations |
| `vendor/` | Vendor-specific code |
| `out/target/product/[device]/` | Build output |

---

**Last Updated**: 2025-11-14
