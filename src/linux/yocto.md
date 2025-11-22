# Yocto Project

A comprehensive guide to the Yocto Project build system, covering BitBake, layers, recipes, image customization, BSP development, and advanced build system techniques.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
   - [System Requirements](#system-requirements)
   - [Installing Dependencies](#installing-dependencies)
   - [Downloading Poky](#downloading-poky)
   - [First Build](#first-build)
3. [Core Architecture](#core-architecture)
   - [Yocto Project Components](#yocto-project-components)
   - [BitBake Build Engine](#bitbake-build-engine)
   - [Metadata and Layers](#metadata-and-layers)
   - [OpenEmbedded Core](#openembedded-core)
4. [Build System Workflow](#build-system-workflow)
   - [Build Process Overview](#build-process-overview)
   - [Task Execution](#task-execution)
   - [Dependency Management](#dependency-management)
   - [Shared State Cache](#shared-state-cache)
5. [Recipes](#recipes)
   - [Recipe Basics](#recipe-basics)
   - [Recipe Syntax](#recipe-syntax)
   - [Common Variables](#common-variables)
   - [Fetchers and Source Retrieval](#fetchers-and-source-retrieval)
   - [Tasks in Recipes](#tasks-in-recipes)
   - [Recipe Examples](#recipe-examples)
6. [Recipe Development](#recipe-development)
   - [Creating Recipes with recipetool](#creating-recipes-with-recipetool)
   - [Writing Custom Recipes](#writing-custom-recipes)
   - [Using .bbappend Files](#using-bbappend-files)
   - [Applying Patches](#applying-patches)
   - [Recipe Inheritance](#recipe-inheritance)
7. [Layers](#layers)
   - [Layer Structure](#layer-structure)
   - [Layer Configuration](#layer-configuration)
   - [Creating Custom Layers](#creating-custom-layers)
   - [Layer Priorities](#layer-priorities)
   - [BBPATH and BBFILES](#bbpath-and-bbfiles)
   - [Layer Dependencies](#layer-dependencies)
8. [Images](#images)
   - [Image Recipes](#image-recipes)
   - [Image Features](#image-features)
   - [Package Groups](#package-groups)
   - [Customizing Images](#customizing-images)
   - [Root Filesystem Customization](#root-filesystem-customization)
9. [BSP Development](#bsp-development)
   - [Machine Configuration](#machine-configuration)
   - [Kernel Configuration](#kernel-configuration)
   - [Bootloader Integration](#bootloader-integration)
   - [Device Tree Integration](#device-tree-integration)
   - [Creating a BSP Layer](#creating-a-bsp-layer)
10. [Package Management](#package-management)
    - [Package Formats](#package-formats)
    - [Package Feeds](#package-feeds)
    - [Runtime Package Management](#runtime-package-management)
    - [Package Dependencies](#package-dependencies)
11. [Configuration System](#configuration-system)
    - [Configuration Files](#configuration-files)
    - [local.conf](#localconf)
    - [bblayers.conf](#bblayersconf)
    - [Distribution Configuration](#distribution-configuration)
    - [Machine Configuration Files](#machine-configuration-files)
    - [Variable Assignment](#variable-assignment)
    - [Important Variables](#important-variables)
12. [Advanced Features](#advanced-features)
    - [SDK Generation](#sdk-generation)
    - [eSDK (Extensible SDK)](#esdk-extensible-sdk)
    - [devtool](#devtool)
    - [Multiconfig Builds](#multiconfig-builds)
    - [SSTATE and Build Acceleration](#sstate-and-build-acceleration)
    - [Template Layer](#template-layer)
13. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
    - [BitBake Debug Options](#bitbake-debug-options)
    - [Build Failures](#build-failures)
    - [Dependency Issues](#dependency-issues)
    - [Task Debugging](#task-debugging)
    - [Common Problems](#common-problems)
14. [Best Practices](#best-practices)
    - [Layer Design](#layer-design)
    - [Recipe Patterns](#recipe-patterns)
    - [Version Management](#version-management)
    - [Build Reproducibility](#build-reproducibility)
15. [Command Reference](#command-reference)
    - [BitBake Commands](#bitbake-commands)
    - [devtool Commands](#devtool-commands)
    - [recipetool Commands](#recipetool-commands)
    - [Common Workflows](#common-workflows)
16. [Resources](#resources)

---

## Overview

The **Yocto Project** is an open-source collaboration project that provides templates, tools, and methods to create custom Linux-based systems for embedded products regardless of the hardware architecture. It's not a Linux distribution but a build system that creates custom Linux distributions.

### Key Benefits

- **Customization**: Build exactly what you need, nothing more
- **Hardware Independence**: Support for multiple architectures (ARM, x86, MIPS, PowerPC, RISC-V)
- **Flexibility**: Modular layer-based architecture
- **Industry Support**: Backed by Linux Foundation with extensive commercial support
- **Long-term Support**: Stable releases with extended maintenance
- **Reproducibility**: Deterministic builds with version control

### Yocto vs. Traditional Distributions

```
Traditional Distribution          Yocto Project
┌─────────────────────┐          ┌─────────────────────┐
│   Pre-built Packages │          │  Custom Build System │
│   ┌──────────────┐  │          │   ┌──────────────┐   │
│   │   apt/yum    │  │          │   │   BitBake    │   │
│   └──────────────┘  │          │   └──────────────┘   │
│   Install packages   │          │   Build from source  │
│   from repository    │          │   with custom config │
└─────────────────────┘          └─────────────────────┘
   Fixed configuration              Full customization
   Quick deployment                 Optimized for target
   Larger footprint                 Minimal footprint
```

## Getting Started

### System Requirements

**Minimum Hardware:**
- 50 GB free disk space (100+ GB recommended)
- 8 GB RAM minimum (16+ GB recommended)
- Multi-core processor (builds are CPU-intensive)

**Supported Host Distributions:**
- Ubuntu 20.04, 22.04, 24.04 (LTS versions recommended)
- Fedora 38, 39, 40
- Debian 11, 12
- OpenSUSE Leap 15.4, 15.5
- AlmaLinux 8, 9
- Rocky Linux 8, 9

### Installing Dependencies

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install -y gawk wget git diffstat unzip texinfo gcc build-essential \
  chrpath socat cpio python3 python3-pip python3-pexpect xz-utils \
  debianutils iputils-ping python3-git python3-jinja2 libegl1-mesa \
  libsdl1.2-dev python3-subunit mesa-common-dev zstd liblz4-tool \
  file locales libacl1
```

**Fedora:**

```bash
sudo dnf install -y gawk make wget tar bzip2 gzip python3 unzip perl patch \
  diffutils diffstat git cpp gcc gcc-c++ glibc-devel texinfo chrpath \
  ccache perl-Data-Dumper perl-Text-ParseWords perl-Thread-Queue \
  perl-bignum socat python3-pexpect findutils which file cpio python3 \
  python3-pip python3-pexpect python3-GitPython python3-jinja2 SDL-devel \
  xz lz4 zstd
```

### Downloading Poky

**Poky** is the reference distribution of Yocto Project.

```bash
# Clone Poky repository
git clone git://git.yoctoproject.org/poky
cd poky

# Checkout latest LTS release (Scarthgap - 5.0)
git checkout -b scarthgap origin/scarthgap

# Or for latest stable (Styhead - 5.1)
git checkout -b styhead origin/styhead
```

**Yocto Release Codenames:**

| Release | Version | Codename | Release Date | Status |
|---------|---------|----------|--------------|--------|
| 5.1     | Styhead | Latest   | Apr 2024     | Current |
| 5.0     | Scarthgap | LTS    | Apr 2024     | LTS |
| 4.3     | Nanbield |         | Oct 2023     | Stable |
| 4.2     | Mickledore | LTS   | Apr 2023     | LTS |
| 4.0     | Kirkstone | LTS    | May 2022     | LTS |

### First Build

```bash
# Source the build environment setup script
source oe-init-build-env

# This creates a 'build' directory and changes to it
# You'll see output like:
#   You had no conf/local.conf file. This configuration file has
#   therefore been created for you from .../local.conf.sample
```

**Directory structure after initialization:**

```
poky/
├── build/                    # Build directory (created)
│   ├── conf/
│   │   ├── local.conf       # Build configuration
│   │   └── bblayers.conf    # Layer configuration
│   ├── tmp/                 # Build artifacts (created during build)
│   ├── downloads/           # Downloaded sources
│   └── sstate-cache/        # Shared state cache
├── meta/                    # OE-Core metadata
├── meta-poky/               # Poky-specific metadata
├── meta-yocto-bsp/          # Reference BSPs
└── meta-skeleton/           # Example recipes and layers
```

**Build a minimal image:**

```bash
# Build core-image-minimal (basic console-only image)
bitbake core-image-minimal

# This will:
# 1. Parse all metadata
# 2. Determine dependencies
# 3. Download sources (~2-5 GB)
# 4. Build cross-compilation toolchain
# 5. Build all packages
# 6. Create root filesystem
# 7. Generate image file
#
# First build: 1-4 hours depending on hardware
# Subsequent builds: Much faster due to caching
```

**Build output location:**

```bash
# Image files are in:
ls tmp/deploy/images/<MACHINE>/

# For QEMU x86-64:
ls tmp/deploy/images/qemux86-64/
# core-image-minimal-qemux86-64.ext4
# core-image-minimal-qemux86-64.iso
# core-image-minimal-qemux86-64.tar.bz2
# core-image-minimal-qemux86-64.wic
```

**Test with QEMU:**

```bash
# Run the built image in QEMU
runqemu qemux86-64

# Or with graphics
runqemu qemux86-64 nographic

# Login as 'root' (no password by default)
```

## Core Architecture

### Yocto Project Components

```
┌────────────────────────────────────────────────────────────┐
│                    Yocto Project                            │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   BitBake    │  │     Poky     │  │  OpenEmbedded   │  │
│  │ Build Engine │  │  Reference   │  │   Build System  │  │
│  └──────────────┘  │ Distribution │  └─────────────────┘  │
│         │          └──────────────┘           │            │
│         │                 │                   │            │
│         └─────────────────┴───────────────────┘            │
│                           │                                │
│         ┌─────────────────┴─────────────────┐             │
│         │                                    │             │
│  ┌──────▼──────┐                    ┌───────▼────────┐    │
│  │  Metadata   │                    │  Build Output   │    │
│  │  (Layers)   │                    │  (Images, SDK)  │    │
│  │             │                    │                 │    │
│  │  - Recipes  │                    │  - Filesystem   │    │
│  │  - Classes  │                    │  - Packages     │    │
│  │  - Config   │                    │  - Toolchain    │    │
│  └─────────────┘                    └─────────────────┘    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### BitBake Build Engine

**BitBake** is the task scheduler and execution engine. It:
- Parses recipes and configuration files
- Builds dependency graphs
- Schedules and executes tasks
- Manages parallel execution
- Handles shared state caching

**Key concepts:**

```python
# BitBake processes metadata written in:
# 1. .bb files (recipes)
# 2. .bbappend files (recipe extensions)
# 3. .bbclass files (shared functionality)
# 4. .conf files (configuration)
# 5. .inc files (included files)
```

### Metadata and Layers

**Layers** are repositories of related metadata. The layer mechanism provides:
- Modular organization
- Easy sharing of metadata
- Separation of concerns
- Override mechanism

```
Layer Stack Example:
┌─────────────────────────┐
│  meta-custom-app        │  ← Your application
├─────────────────────────┤
│  meta-custom-bsp        │  ← Your BSP
├─────────────────────────┤
│  meta-openembedded      │  ← Additional recipes
├─────────────────────────┤
│  meta-poky              │  ← Poky distro config
├─────────────────────────┤
│  meta (OE-Core)         │  ← Core recipes
└─────────────────────────┘

Higher layers can override lower layers
```

### OpenEmbedded Core

**OE-Core** (meta layer) provides:
- Core recipes for fundamental packages
- Base classes for recipe inheritance
- Build system infrastructure
- Cross-compilation toolchain recipes
- Standard image recipes

## Build System Workflow

### Build Process Overview

```
BitBake Execution Flow:
┌──────────────────────────────────────────────────────────┐
│ 1. Parse Configuration                                    │
│    - Read local.conf, bblayers.conf                      │
│    - Load layer configurations                           │
│    - Set variables                                       │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│ 2. Parse Metadata                                         │
│    - Parse all .bb files                                 │
│    - Parse .bbappend files                               │
│    - Inherit classes                                     │
│    - Resolve variables                                   │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│ 3. Build Dependency Graph                                 │
│    - DEPENDS (build-time dependencies)                   │
│    - RDEPENDS (runtime dependencies)                     │
│    - Task dependencies                                   │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│ 4. Execute Tasks                                          │
│    - Fetch sources                                       │
│    - Unpack archives                                     │
│    - Patch sources                                       │
│    - Configure                                           │
│    - Compile                                             │
│    - Install                                             │
│    - Package                                             │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│ 5. Create Output                                          │
│    - Root filesystem                                     │
│    - Image file(s)                                       │
│    - SDK (if requested)                                  │
└──────────────────────────────────────────────────────────┘
```

### Task Execution

**Standard task sequence for a recipe:**

```
do_fetch      → Download source from SRC_URI
    ↓
do_unpack     → Extract archives
    ↓
do_patch      → Apply patches
    ↓
do_configure  → Run ./configure or equivalent
    ↓
do_compile    → Build the software
    ↓
do_install    → Install to staging area
    ↓
do_package    → Split into packages
    ↓
do_package_write_* → Create package files (rpm/deb/ipk)
```

**Task dependencies:**

```bash
# View task dependencies for a recipe
bitbake -g <recipe-name>
# Generates task-depends.dot

# View all tasks for a recipe
bitbake <recipe-name> -c listtasks

# Example output:
# do_build
# do_bundle_initramfs
# do_checkuri
# do_clean
# do_cleanall
# do_cleansstate
# do_compile
# do_configure
# ...
```

### Dependency Management

**Build-time dependencies (DEPENDS):**

```python
# Required for building the recipe
DEPENDS = "zlib openssl"

# These packages' do_populate_sysroot must complete
# before this recipe's do_configure starts
```

**Runtime dependencies (RDEPENDS):**

```python
# Required for running the package
RDEPENDS:${PN} = "bash python3 libssl"

# These packages must be installed on target
# when this package is installed
```

**Dependency graph visualization:**

```bash
# Generate dependency graph
bitbake -g core-image-minimal

# This creates:
# - pn-buildlist: Package build order
# - task-depends.dot: Task dependency graph
# - pn-depends.dot: Package dependency graph

# View with graphviz
dot -Tpng pn-depends.dot -o depends.png
```

### Shared State Cache

**SSTATE** (Shared State) enables build acceleration:

```
Without SSTATE:              With SSTATE:
┌────────────────┐          ┌────────────────┐
│ Full rebuild   │          │ Check SSTATE   │
│ every time     │          │ cache          │
│                │          │                │
│ Hours...       │          │ ┌──────────┐   │
└────────────────┘          │ │ Hit: Use │   │
                            │ │ cached   │   │
                            │ └──────────┘   │
                            │ ┌──────────┐   │
                            │ │ Miss:    │   │
                            │ │ Rebuild  │   │
                            │ └──────────┘   │
                            └────────────────┘
                            Minutes instead!
```

**SSTATE configuration:**

```python
# In local.conf
SSTATE_DIR ?= "${TOPDIR}/sstate-cache"

# Use shared SSTATE location for multiple builds
SSTATE_DIR = "/common/sstate-cache"

# Use mirror for SSTATE
SSTATE_MIRRORS = "\
    file://.* http://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH \
    file://.* http://example.com/sstate/PATH;downloadfilename=PATH \
"
```

## Recipes

### Recipe Basics

A **recipe** (.bb file) contains:
- Source location (SRC_URI)
- Dependencies (DEPENDS, RDEPENDS)
- License information
- Build configuration
- Installation instructions
- Package splitting rules

**Minimal recipe example:**

```python
# hello-world_1.0.bb

SUMMARY = "Simple hello world application"
DESCRIPTION = "A hello world program demonstrating basic recipe structure"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=..."

SRC_URI = "https://example.com/hello-1.0.tar.gz"
SRC_URI[sha256sum] = "abcd1234..."

inherit autotools

do_install:append() {
    install -d ${D}${docdir}/${PN}
    install -m 0644 README ${D}${docdir}/${PN}/
}
```

### Recipe Syntax

**Variable assignment operators:**

```python
# = : Simple assignment (evaluated at reference time)
VAR = "value"

# := : Immediate assignment (evaluated immediately)
VAR := "value"

# ?= : Weak default (only if not already set)
VAR ?= "default"

# ??= : Weaker default (lowest priority)
VAR ??= "weak default"

# += : Append with space
VAR += "append"

# =+ : Prepend with space
VAR =+ "prepend"

# .= : Append without space
VAR .= "append"

# =. : Prepend without space
VAR =. "prepend"

# :append : Append after all other operations
VAR:append = " appended"

# :prepend : Prepend before all other operations
VAR:prepend = "prepended "

# :remove : Remove items
VAR:remove = "item to remove"
```

**Package-specific variables (overrides):**

```python
# Apply to all packages
RDEPENDS = "bash"

# Apply only to main package
RDEPENDS:${PN} = "bash"

# Apply to specific package
RDEPENDS:${PN}-dev = "pkgconfig"

# Machine-specific
SRC_URI:append:x86 = " file://x86-specific.patch"

# Distro-specific
PACKAGE_ARCH:poky = "core2-64"
```

### Common Variables

**Source and destination:**

```python
# Source directory (after do_unpack)
S = "${WORKDIR}/${PN}-${PV}"

# Build directory
B = "${WORKDIR}/build"

# Destination (install root)
D = "${WORKDIR}/image"

# Staging directories
STAGING_DIR_HOST = "${TMPDIR}/sysroots/${MACHINE}"
STAGING_BINDIR = "${STAGING_DIR_HOST}${bindir}"
STAGING_INCDIR = "${STAGING_DIR_HOST}${includedir}"
STAGING_LIBDIR = "${STAGING_DIR_HOST}${libdir}"
```

**Standard directories:**

```python
# Filesystem paths (target)
prefix = "/usr"
exec_prefix = "/usr"
bindir = "/usr/bin"
sbindir = "/usr/sbin"
libdir = "/usr/lib"
includedir = "/usr/include"
datadir = "/usr/share"
sysconfdir = "/etc"
localstatedir = "/var"
```

**Package naming:**

```python
# Package name
PN = "packagename"

# Package version
PV = "1.0"

# Package revision
PR = "r0"

# Package epoch (for versioning)
PE = "1"

# Full package name with version
PF = "${PN}-${PV}-${PR}"

# Base package name for files
BP = "${PN}-${PV}"
```

### Fetchers and Source Retrieval

**SRC_URI supports multiple protocols:**

```python
# HTTP/HTTPS
SRC_URI = "https://example.com/project-${PV}.tar.gz"

# Git repository
SRC_URI = "git://github.com/user/repo.git;protocol=https;branch=master"

# Git with specific commit
SRC_URI = "git://github.com/user/repo.git;protocol=https"
SRCREV = "abc123def456..."

# Subversion
SRC_URI = "svn://svn.example.com/project;module=trunk;protocol=https"

# Local files
SRC_URI = "file://local-source.tar.gz \
           file://0001-fix-build.patch \
           file://custom-config.cfg \
          "

# Multiple sources
SRC_URI = "https://example.com/main-${PV}.tar.gz \
           git://github.com/user/addon.git;protocol=https;destsuffix=addon \
           file://init.d/myservice \
          "
```

**Checksum verification:**

```python
# SHA256 checksum for archives
SRC_URI[sha256sum] = "abcd1234ef567890..."

# MD5 checksum (legacy, avoid)
SRC_URI[md5sum] = "abc123..."

# Multiple files with checksums
SRC_URI = "https://example.com/main.tar.gz \
           https://example.com/addon.tar.gz;name=addon"

SRC_URI[sha256sum] = "main-file-checksum..."
SRC_URI[addon.sha256sum] = "addon-file-checksum..."
```

**Git-specific options:**

```python
# Specific branch
SRC_URI = "git://github.com/user/repo.git;protocol=https;branch=develop"

# Specific tag
SRC_URI = "git://github.com/user/repo.git;protocol=https;tag=v1.0"
SRCREV = "${AUTOREV}"  # Or specific commit

# Submodules
SRC_URI = "git://github.com/user/repo.git;protocol=https;branch=master;submodules=1"

# Custom destination directory
SRC_URI = "git://github.com/user/repo.git;protocol=https;destsuffix=custom-dir"

# Multiple git repos
SRC_URI = "git://github.com/user/main.git;protocol=https;name=main \
           git://github.com/user/lib.git;protocol=https;name=lib;destsuffix=lib"
SRCREV_main = "abc123..."
SRCREV_lib = "def456..."
```

### Tasks in Recipes

**Standard tasks:**

```python
# Fetch task (download sources)
do_fetch() {
    # Usually handled by fetchers automatically
}

# Unpack task (extract archives)
do_unpack() {
    # Usually handled automatically
}

# Patch task (apply patches)
do_patch() {
    # Automatically applies patches from SRC_URI
}

# Configure task
do_configure() {
    # For autotools:
    oe_runconf

    # For CMake (with cmake class):
    cmake_do_configure

    # Custom:
    ./custom-configure --prefix=${prefix}
}

# Compile task
do_compile() {
    # For make:
    oe_runmake

    # Custom:
    ${CC} ${CFLAGS} -o myapp main.c
}

# Install task
do_install() {
    # Create directories
    install -d ${D}${bindir}
    install -d ${D}${sysconfdir}

    # Install files
    install -m 0755 myapp ${D}${bindir}/
    install -m 0644 myapp.conf ${D}${sysconfdir}/
}
```

**Task modification:**

```python
# Append to existing task
do_install:append() {
    # Add extra install steps
    install -d ${D}${datadir}/myapp
    cp -r data/* ${D}${datadir}/myapp/
}

# Prepend to existing task
do_configure:prepend() {
    # Run before configure
    ./autogen.sh
}

# Completely override task
do_compile() {
    # Completely replace default compile
    ${CC} ${CFLAGS} -o special special.c
}
```

**Custom tasks:**

```python
# Define custom task
do_custom_task() {
    echo "Running custom task"
    # Custom operations
}

# Add to task chain
addtask custom_task after do_compile before do_install

# Or with dependencies
addtask custom_task after do_configure before do_compile
do_custom_task[depends] = "other-recipe:do_install"
```

### Recipe Examples

**Simple C application:**

```python
# myapp_1.0.bb

SUMMARY = "My custom application"
DESCRIPTION = "A custom C application for embedded Linux"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=..."

SRC_URI = "git://github.com/myuser/myapp.git;protocol=https;branch=main \
           file://0001-add-syslog.patch \
           file://myapp.service \
          "
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git"

DEPENDS = "zlib openssl"
RDEPENDS:${PN} = "bash"

inherit systemd

SYSTEMD_SERVICE:${PN} = "myapp.service"

do_compile() {
    ${CC} ${CFLAGS} ${LDFLAGS} -o myapp main.c -lz -lssl
}

do_install() {
    install -d ${D}${bindir}
    install -m 0755 myapp ${D}${bindir}/

    # Install systemd service
    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/myapp.service ${D}${systemd_system_unitdir}/
}
```

**Python package:**

```python
# python3-mymodule_1.0.bb

SUMMARY = "My Python module"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=..."

SRC_URI = "https://pypi.org/packages/source/m/mymodule/mymodule-${PV}.tar.gz"
SRC_URI[sha256sum] = "..."

inherit setuptools3

RDEPENDS:${PN} = "python3-core python3-requests"
```

**Kernel module:**

```python
# my-driver_1.0.bb

SUMMARY = "Custom kernel driver"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=..."

inherit module

SRC_URI = "file://my-driver.c \
           file://Makefile \
          "

S = "${WORKDIR}"

# Module is tied to kernel version
RPROVIDES:${PN} += "kernel-module-my-driver"
```

**Library with development package:**

```python
# libmylib_1.0.bb

SUMMARY = "My custom library"
LICENSE = "LGPL-2.1-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=..."

SRC_URI = "https://example.com/libmylib-${PV}.tar.gz"
SRC_URI[sha256sum] = "..."

inherit autotools pkgconfig

# Package splitting
PACKAGES = "${PN} ${PN}-dev ${PN}-dbg ${PN}-doc"

FILES:${PN} = "${libdir}/lib*.so.*"
FILES:${PN}-dev = "${includedir} ${libdir}/lib*.so ${libdir}/*.la \
                   ${libdir}/pkgconfig"
FILES:${PN}-dbg = "${libdir}/.debug"
FILES:${PN}-doc = "${datadir}/doc ${mandir}"

RDEPENDS:${PN}-dev = "${PN} (= ${EXTENDPKGV})"
```

## Recipe Development

### Creating Recipes with recipetool

**recipetool** automates recipe creation:

```bash
# Create recipe from source URL
recipetool create https://example.com/package-1.0.tar.gz

# Create recipe from Git repository
recipetool create git://github.com/user/project.git

# Specify output file
recipetool create -o myrecipe_1.0.bb https://example.com/src.tar.gz

# Create recipe for Python package from PyPI
recipetool create https://pypi.org/packages/source/p/package/package-1.0.tar.gz
```

**Example output:**

```python
# Auto-generated by recipetool
SUMMARY = "Package description from source"
HOMEPAGE = "https://example.com"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=..."

SRC_URI = "https://example.com/package-${PV}.tar.gz"
SRC_URI[sha256sum] = "..."

# Dependencies detected automatically
DEPENDS = "zlib openssl"

inherit autotools
```

### Writing Custom Recipes

**Step-by-step recipe creation:**

```bash
# 1. Create layer (if not exists)
bitbake-layers create-layer meta-custom
bitbake-layers add-layer meta-custom

# 2. Create recipe directory
mkdir -p meta-custom/recipes-apps/myapp

# 3. Create recipe file
vim meta-custom/recipes-apps/myapp/myapp_1.0.bb
```

**Recipe template:**

```python
# myapp_1.0.bb

SUMMARY = "Brief description"
DESCRIPTION = "Longer description of what this does"
HOMEPAGE = "https://example.com"
SECTION = "applications"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://LICENSE;md5=..."

DEPENDS = ""
RDEPENDS:${PN} = ""

SRC_URI = "https://example.com/${PN}-${PV}.tar.gz"
SRC_URI[sha256sum] = ""

S = "${WORKDIR}/${PN}-${PV}"

inherit autotools

# Override/append tasks as needed
do_install:append() {
    # Additional install steps
}
```

**Computing checksums:**

```bash
# Calculate SHA256
sha256sum package-1.0.tar.gz

# Calculate MD5 (for LICENSE files)
md5sum LICENSE
```

### Using .bbappend Files

**bbappend files** extend/modify existing recipes without editing them:

```bash
# Location must match recipe path
# Recipe: meta/recipes-core/busybox/busybox_1.36.1.bb
# Append: meta-custom/recipes-core/busybox/busybox_%.bbappend

# % matches any version
# Or use specific version: busybox_1.36.1.bbappend
```

**Example bbappend:**

```python
# busybox_%.bbappend

FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI += "file://custom.cfg \
            file://0001-add-feature.patch \
           "

do_install:append() {
    # Add custom configuration
    install -d ${D}${sysconfdir}
    install -m 0644 ${WORKDIR}/custom.cfg ${D}${sysconfdir}/busybox.conf
}
```

**Directory structure:**

```
meta-custom/recipes-core/busybox/
├── busybox_%.bbappend
└── busybox/
    ├── custom.cfg
    └── 0001-add-feature.patch
```

**FILESEXTRAPATHS** tells BitBake where to find additional files:

```python
# Prepend to search path
FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

# ${THISDIR} = directory containing .bbappend
# ${PN} = package name (e.g., "busybox")
```

### Applying Patches

**Patch format:**

```bash
# Create patch from git
cd source-repo
git format-patch -1 commit-hash
# Creates 0001-commit-message.patch

# Or create patch manually
diff -Naur original/ modified/ > my-changes.patch
```

**Add patch to recipe:**

```python
SRC_URI = "https://example.com/package-${PV}.tar.gz \
           file://0001-fix-build.patch \
           file://0002-add-feature.patch \
          "

# Patches applied in order during do_patch
```

**Patch with different strip level:**

```python
# Default: -p1
SRC_URI = "file://0001-fix.patch"

# Custom strip level
SRC_URI = "file://0001-fix.patch;striplevel=0"
```

**Conditional patches:**

```python
# Machine-specific patch
SRC_URI:append:raspberrypi4 = " file://rpi4-fix.patch"

# Version-specific patch
SRC_URI:append = "${@bb.utils.contains('PV', '1.0', ' file://v1-fix.patch', '', d)}"
```

**Patch directory structure:**

```
meta-custom/recipes-apps/myapp/
├── myapp_1.0.bb
└── myapp/
    ├── 0001-first-fix.patch
    ├── 0002-second-fix.patch
    └── files/
        └── config.h
```

### Recipe Inheritance

**Classes** provide shared functionality:

```python
# Inherit a class
inherit autotools

# Multiple classes
inherit systemd update-rc.d

# Conditional inheritance
inherit ${@bb.utils.contains('DISTRO_FEATURES', 'systemd', 'systemd', '', d)}
```

**Common classes:**

```python
# autotools: ./configure && make && make install
inherit autotools

# cmake: CMake-based projects
inherit cmake

# setuptools3: Python setuptools
inherit setuptools3

# systemd: systemd service integration
inherit systemd

# update-rc.d: SysV init scripts
inherit update-rc.d

# pkgconfig: pkg-config integration
inherit pkgconfig

# module: Kernel modules
inherit module

# native: Build for build host
inherit native

# cross: Cross-compilation tools
inherit cross
```

**Class example (autotools):**

```python
inherit autotools

# Provides:
# - do_configure: Runs ./configure with standard options
# - do_compile: Runs make
# - do_install: Runs make install
# - Standard variable handling

# Customize configure arguments
EXTRA_OECONF = "--enable-feature --with-lib=/path"

# Customize make arguments
EXTRA_OEMAKE = "CFLAGS='${CFLAGS}' LDFLAGS='${LDFLAGS}'"
```

## Layers

### Layer Structure

**Standard layer structure:**

```
meta-custom/
├── conf/
│   └── layer.conf              # Layer configuration
├── classes/
│   └── custom.bbclass          # Shared classes
├── recipes-*/                  # Recipe directories
│   ├── recipes-core/
│   │   └── images/
│   │       └── custom-image.bb
│   ├── recipes-apps/
│   │   └── myapp/
│   │       ├── myapp_1.0.bb
│   │       └── myapp/
│   │           └── files...
│   └── recipes-bsp/
│       └── u-boot/
│           └── u-boot_%.bbappend
├── recipes-kernel/
│   └── linux/
│       └── linux-%.bbappend
├── conf/
│   ├── machine/
│   │   └── mymachine.conf      # Machine definitions
│   └── distro/
│       └── mydistro.conf       # Distribution config
├── wic/
│   └── custom-image.wks        # WIC kickstart files
├── LICENSE
└── README
```

### Layer Configuration

**layer.conf:**

```python
# meta-custom/conf/layer.conf

# Layer identity
BBPATH .= ":${LAYERDIR}"

# Recipe paths
BBFILES += "${LAYERDIR}/recipes-*/*/*.bb \
            ${LAYERDIR}/recipes-*/*/*.bbappend"

# Layer name
BBFILE_COLLECTIONS += "custom-layer"

# Recipe priority (higher = higher priority)
BBFILE_PRIORITY_custom-layer = "6"

# Layer pattern matching
BBFILE_PATTERN_custom-layer = "^${LAYERDIR}/"

# Compatible versions
LAYERSERIES_COMPAT_custom-layer = "styhead scarthgap"

# Layer dependencies
LAYERDEPENDS_custom-layer = "core openembedded-layer"

# Additional class paths
BBFILES_DYNAMIC += ""
```

**Layer priority:**

```
Priority values:
0-4:   Reserved for OE-Core
5:     meta-poky (default)
6-9:   Custom layers (typical)
10+:   High priority overrides
```

### Creating Custom Layers

```bash
# Create new layer with bitbake-layers
bitbake-layers create-layer meta-custom

# Output:
# meta-custom/
# ├── conf/
# │   └── layer.conf
# ├── COPYING.MIT
# ├── README
# └── recipes-example/
#     └── example/
#         └── example_0.1.bb

# Add layer to build
bitbake-layers add-layer meta-custom

# Or manually edit conf/bblayers.conf
```

**Manual layer creation:**

```bash
# 1. Create directory structure
mkdir -p meta-custom/conf
mkdir -p meta-custom/recipes-apps

# 2. Create layer.conf
cat > meta-custom/conf/layer.conf << 'EOF'
BBPATH .= ":${LAYERDIR}"
BBFILES += "${LAYERDIR}/recipes-*/*/*.bb \
            ${LAYERDIR}/recipes-*/*/*.bbappend"
BBFILE_COLLECTIONS += "custom"
BBFILE_PRIORITY_custom = "6"
BBFILE_PATTERN_custom = "^${LAYERDIR}/"
LAYERSERIES_COMPAT_custom = "styhead scarthgap"
EOF

# 3. Add to bblayers.conf
bitbake-layers add-layer meta-custom
```

### Layer Priorities

**Override mechanism:**

```
Higher priority layers override lower priority:

meta-custom (priority 8)      # Highest
    ↓ overrides
meta-custom-bsp (priority 7)
    ↓ overrides
meta-openembedded (priority 6)
    ↓ overrides
meta-poky (priority 5)
    ↓ overrides
meta (priority 5)             # Lowest
```

**Example:**

```python
# meta/recipes-core/bash/bash_5.2.bb (priority 5)
SRC_URI = "https://ftp.gnu.org/gnu/bash/bash-${PV}.tar.gz"

# meta-custom/recipes-core/bash/bash_%.bbappend (priority 8)
# This takes precedence
SRC_URI:append = " file://custom-config.patch"
```

### BBPATH and BBFILES

**BBPATH:**

```python
# Search path for configuration files
BBPATH = "path1:path2:path3"

# Each layer adds to BBPATH
BBPATH .= ":${LAYERDIR}"

# Used to find files like:
# - conf/machine/*.conf
# - conf/distro/*.conf
# - classes/*.bbclass
```

**BBFILES:**

```python
# Recipes to include
BBFILES = "${LAYERDIR}/recipes-*/*/*.bb ${LAYERDIR}/recipes-*/*/*.bbappend"

# Wildcard patterns:
# recipes-*    : Any directory starting with "recipes-"
# /*           : Any subdirectory
# /*.bb        : Any .bb file
# /*.bbappend  : Any .bbappend file
```

**Dynamic layers:**

```python
# Conditional recipe inclusion
BBFILES_DYNAMIC += "\
    qt5-layer:${LAYERDIR}/dynamic-layers/qt5-layer/recipes-*/*/*.bb \
    meta-python:${LAYERDIR}/dynamic-layers/meta-python/recipes-*/*/*.bb \
"

# Directory structure:
# meta-custom/
# └── dynamic-layers/
#     ├── qt5-layer/
#     │   └── recipes-apps/
#     └── meta-python/
#         └── recipes-devtools/

# Recipes only included if dependency layer present
```

### Layer Dependencies

**LAYERDEPENDS:**

```python
# In layer.conf
LAYERDEPENDS_custom-layer = "core openembedded-layer qt5-layer"

# Ensures required layers are present
# Build fails if dependencies missing
```

**LAYERRECOMMENDS:**

```python
# Recommended but not required
LAYERRECOMMENDS_custom-layer = "meta-networking"

# Warning if missing, but build continues
```

**Version-specific dependencies:**

```python
# Require specific layer version
LAYERDEPENDS_custom-layer = "core:styhead openembedded-layer:styhead"
```

**Checking layer dependencies:**

```bash
# Show layer configuration
bitbake-layers show-layers

# Output:
# layer           path                                      priority
# ==========================================================================
# meta            /path/to/poky/meta                        5
# meta-poky       /path/to/poky/meta-poky                   5
# meta-custom     /path/to/meta-custom                      6

# Show dependencies
bitbake-layers layerindex-show-depends meta-custom
```

## Images

### Image Recipes

**Image recipe basics:**

```python
# custom-image-minimal.bb

SUMMARY = "A minimal custom image"
DESCRIPTION = "Minimal image with custom applications"

# Inherit image creation class
inherit core-image

# License (for the recipe itself)
LICENSE = "MIT"

# Base packages to include
IMAGE_INSTALL = "packagegroup-core-boot ${CORE_IMAGE_EXTRA_INSTALL}"

# Add custom packages
IMAGE_INSTALL += "myapp python3 openssh"

# Image features
IMAGE_FEATURES += "ssh-server-openssh"

# Root filesystem size (KB)
IMAGE_ROOTFS_SIZE ?= "8192"

# Extra space (KB)
IMAGE_ROOTFS_EXTRA_SPACE = "1024"
```

**Common base image classes:**

```python
# core-image: Basic image class
inherit core-image

# core-image-minimal: Minimal bootable image
require recipes-core/images/core-image-minimal.bb

# core-image-base: Small image with package manager
require recipes-core/images/core-image-base.bb

# core-image-full-cmdline: Console-only with full hardware support
require recipes-core/images/core-image-full-cmdline.bb
```

### Image Features

**IMAGE_FEATURES:**

```python
# Development tools
IMAGE_FEATURES += "tools-debug tools-profile tools-sdk"

# Package management
IMAGE_FEATURES += "package-management"

# SSH server
IMAGE_FEATURES += "ssh-server-openssh"
# Or dropbear (lighter)
IMAGE_FEATURES += "ssh-server-dropbear"

# X11 support
IMAGE_FEATURES += "x11 x11-base x11-sato"

# Documentation
IMAGE_FEATURES += "doc-pkgs"

# Development headers
IMAGE_FEATURES += "dev-pkgs"

# Debug symbols
IMAGE_FEATURES += "dbg-pkgs"

# Read-only root filesystem
IMAGE_FEATURES += "read-only-rootfs"

# Empty root password
IMAGE_FEATURES += "empty-root-password"

# Allow root login
IMAGE_FEATURES += "allow-root-login"

# Post-install logging
IMAGE_FEATURES += "post-install-logging"
```

**Feature to package mapping:**

```python
# In conf/bitbake.conf and image.bbclass
FEATURE_PACKAGES_ssh-server-openssh = "packagegroup-core-ssh-openssh"
FEATURE_PACKAGES_tools-debug = "packagegroup-core-tools-debug"
FEATURE_PACKAGES_package-management = "${ROOTFS_PKGMANAGE}"
```

### Package Groups

**Package groups** bundle related packages:

```python
# packagegroup-custom-apps.bb

SUMMARY = "Custom application package group"
DESCRIPTION = "Package group for custom applications"

inherit packagegroup

PACKAGES = "${PN}"

RDEPENDS:${PN} = "\
    myapp \
    python3 \
    python3-requests \
    nginx \
    postgresql \
"
```

**Using package groups in images:**

```python
# custom-image.bb
IMAGE_INSTALL += "packagegroup-custom-apps"
```

**Standard package groups:**

```python
# Core boot packages
packagegroup-core-boot

# SSH server (OpenSSH)
packagegroup-core-ssh-openssh

# Debug tools
packagegroup-core-tools-debug
# Includes: gdb, strace, ltrace, tcf-agent

# Profiling tools
packagegroup-core-tools-profile
# Includes: perf, oprofile, lttng

# Build essentials
packagegroup-core-buildessential
# Includes: gcc, make, binutils, headers
```

### Customizing Images

**Adding packages:**

```python
# Direct package installation
IMAGE_INSTALL += "vim htop tmux"

# From package group
IMAGE_INSTALL += "packagegroup-core-tools-debug"

# Conditional installation
IMAGE_INSTALL += "${@bb.utils.contains('MACHINE', 'raspberrypi4', 'raspi-gpio', '', d)}"
```

**Removing packages:**

```python
# Remove from inherited image
IMAGE_INSTALL:remove = "packagegroup-core-x11"

# Remove specific packages
PACKAGE_EXCLUDE = "package1 package2"
```

**Custom post-processing:**

```python
# Run after rootfs creation
ROOTFS_POSTPROCESS_COMMAND += "custom_postprocess; "

custom_postprocess() {
    # Modify rootfs
    echo "CustomHost" > ${IMAGE_ROOTFS}/etc/hostname

    # Create custom files
    install -d ${IMAGE_ROOTFS}/opt/custom
    echo "config data" > ${IMAGE_ROOTFS}/opt/custom/config

    # Modify permissions
    chmod 600 ${IMAGE_ROOTFS}/opt/custom/config
}
```

### Root Filesystem Customization

**Root filesystem types:**

```python
# Set filesystem types to generate
IMAGE_FSTYPES = "ext4 tar.bz2 wic"

# Common types:
# - ext4: ext4 filesystem
# - ext3: ext3 filesystem
# - tar.bz2: Compressed tarball
# - tar.gz: Gzip compressed tarball
# - cpio: cpio archive
# - cpio.gz: Compressed cpio
# - wic: Partitioned disk image
# - iso: ISO image
# - hddimg: Hybrid ISO/HDD image
# - ubi: UBI filesystem (for NAND flash)
# - ubifs: UBIFS filesystem
# - squashfs: Compressed read-only filesystem
```

**Filesystem configuration:**

```python
# Ext4 options
EXTRA_IMAGECMD:ext4 = "-i 4096 -L rootfs"

# Size constraints
IMAGE_ROOTFS_SIZE = "8192"          # Fixed size (KB)
IMAGE_ROOTFS_MAXSIZE = "10240"      # Maximum size (KB)
IMAGE_ROOTFS_EXTRA_SPACE = "1024"   # Extra space (KB)

# Overhead factor
IMAGE_OVERHEAD_FACTOR = "1.3"       # 30% overhead
```

**Read-only root filesystem:**

```python
# Enable read-only rootfs
IMAGE_FEATURES += "read-only-rootfs"

# Requires:
# - /var as tmpfs or separate writable partition
# - /etc overlayfs or tmpfs
# - Proper mount configuration

# Additional configuration
EXTRA_IMAGE_FEATURES += "read-only-rootfs"

# Writable directories
VOLATILE_LOG_DIR = "yes"
```

**User and group management:**

```python
# Add users in recipe
inherit useradd

USERADD_PACKAGES = "${PN}"
USERADD_PARAM:${PN} = "-u 1000 -d /home/myuser -s /bin/bash myuser"
GROUPADD_PARAM:${PN} = "-g 1000 mygroup"

# Or in image recipe
ROOTFS_POSTPROCESS_COMMAND += "add_custom_users; "

add_custom_users() {
    echo "myuser:x:1000:1000::/home/myuser:/bin/bash" >> ${IMAGE_ROOTFS}/etc/passwd
    echo "myuser:!:18000:0:99999:7:::" >> ${IMAGE_ROOTFS}/etc/shadow
    echo "mygroup:x:1000:" >> ${IMAGE_ROOTFS}/etc/group
}
```

**Hostname and networking:**

```python
# Set hostname
hostname:pn-base-files = "my-device"

# Or in postprocess
ROOTFS_POSTPROCESS_COMMAND += "set_hostname; "

set_hostname() {
    echo "my-device" > ${IMAGE_ROOTFS}/etc/hostname
}

# Network configuration
ROOTFS_POSTPROCESS_COMMAND += "setup_network; "

setup_network() {
    cat > ${IMAGE_ROOTFS}/etc/network/interfaces << EOF
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet dhcp
EOF
}
```

## BSP Development

### Machine Configuration

**Machine conf file:**

```python
# conf/machine/mymachine.conf

#@TYPE: Machine
#@NAME: My Custom Machine
#@DESCRIPTION: Machine configuration for my custom hardware

# Architecture
DEFAULTTUNE = "cortexa72"
require conf/machine/include/arm/armv8a/tune-cortexa72.inc

# Kernel
PREFERRED_PROVIDER_virtual/kernel = "linux-custom"
PREFERRED_VERSION_linux-custom = "6.1%"

# Bootloader
PREFERRED_PROVIDER_virtual/bootloader = "u-boot-custom"
PREFERRED_VERSION_u-boot-custom = "2024.01%"

# Machine features
MACHINE_FEATURES = "usbhost usbgadget alsa wifi bluetooth ext2 ext3 ext4 serial"

# Serial console
SERIAL_CONSOLES = "115200;ttyS0"

# Kernel configuration
KERNEL_IMAGETYPE = "Image"
KERNEL_DEVICETREE = "myvendor/mymachine.dtb"

# U-Boot configuration
UBOOT_MACHINE = "mymachine_defconfig"
UBOOT_ENTRYPOINT = "0x80008000"
UBOOT_LOADADDRESS = "0x80008000"

# Image format
IMAGE_FSTYPES = "ext4 tar.bz2 wic"
WKS_FILE = "mymachine.wks"

# Extra packages for this machine
MACHINE_EXTRA_RDEPENDS = "kernel-modules"
MACHINE_EXTRA_RRECOMMENDS = "firmware-custom"

# Root device
ROOT_DEVICE = "mmcblk0p2"
```

**Machine features:**

```python
# Common machine features:
MACHINE_FEATURES = "\
    alsa          # Audio ALSA support
    bluetooth     # Bluetooth
    ext2          # ext2 filesystem
    ext3          # ext3 filesystem
    ext4          # ext4 filesystem
    keyboard      # Keyboard
    pci           # PCI bus
    pcmcia        # PCMCIA/CardBus
    phone         # Phone
    qvga          # QVGA display (320x240)
    screen        # Screen/display
    serial        # Serial console
    touchscreen   # Touchscreen
    usbgadget     # USB gadget
    usbhost       # USB host
    vfat          # VFAT filesystem
    wifi          # WiFi
"
```

### Kernel Configuration

**Kernel recipe for custom machine:**

```python
# recipes-kernel/linux/linux-custom_6.1.bb

require recipes-kernel/linux/linux-yocto.inc

LINUX_VERSION = "6.1.50"
LINUX_VERSION_EXTENSION = "-custom"

SRCREV = "abc123..."
SRC_URI = "git://github.com/myorg/linux.git;protocol=https;branch=custom-6.1"

SRC_URI += "\
    file://defconfig \
    file://mymachine.scc \
    file://0001-custom-driver.patch \
"

COMPATIBLE_MACHINE = "mymachine"

# Kernel configuration fragments
KERNEL_FEATURES:append = " cfg/smp.scc"
```

**Kernel configuration fragment:**

```bash
# recipes-kernel/linux/linux-custom/mymachine.scc

define KFEATURE_DESCRIPTION "My Machine kernel configuration"
define KFEATURE_COMPATIBILITY machine

# Include kernel types
kconf hardware mymachine.cfg

# mymachine.cfg:
CONFIG_MYDRIVER=y
CONFIG_I2C=y
CONFIG_SPI=y
CONFIG_GPIO_SYSFS=y
```

**Using kernel config fragments with bbappend:**

```python
# recipes-kernel/linux/linux-yocto_%.bbappend

FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI += "\
    file://custom-drivers.cfg \
    file://custom-features.cfg \
"

# custom-drivers.cfg:
# CONFIG_MYDRIVER=y
# CONFIG_ANOTHER_DRIVER=m
```

### Bootloader Integration

**U-Boot recipe:**

```python
# recipes-bsp/u-boot/u-boot-custom_2024.01.bb

require recipes-bsp/u-boot/u-boot-common.inc
require recipes-bsp/u-boot/u-boot.inc

SUMMARY = "U-Boot for My Machine"
LICENSE = "GPL-2.0-or-later"
LIC_FILES_CHKSUM = "file://Licenses/README;md5=..."

SRCREV = "def456..."
SRC_URI = "git://github.com/myorg/u-boot.git;protocol=https;branch=custom"

SRC_URI += "\
    file://boot.cmd \
    file://0001-add-board-support.patch \
"

S = "${WORKDIR}/git"

COMPATIBLE_MACHINE = "mymachine"

# Compile boot script
do_compile:append() {
    ${B}/tools/mkimage -C none -A arm -T script -d ${WORKDIR}/boot.cmd ${B}/boot.scr
}

do_install:append() {
    install -d ${D}/boot
    install -m 0644 ${B}/boot.scr ${D}/boot/
}

FILES:${PN} += "/boot/boot.scr"
```

**U-Boot bbappend for environment:**

```python
# recipes-bsp/u-boot/u-boot_%.bbappend

FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI += "file://mymachine-env.txt"

do_compile:append() {
    ${B}/tools/mkenvimage -s 131072 -o ${B}/u-boot-env.bin ${WORKDIR}/mymachine-env.txt
}
```

### Device Tree Integration

**Device tree recipe:**

```python
# recipes-bsp/device-tree/device-tree-custom.bb

SUMMARY = "Custom device trees"
LICENSE = "GPL-2.0-only | BSD-2-Clause"
LIC_FILES_CHKSUM = "file://COPYING;md5=..."

inherit devicetree

SRC_URI = "\
    file://mymachine.dts \
    file://mymachine-overlay.dts \
"

S = "${WORKDIR}"

# Compiled DTBs installed to /boot/devicetree/
```

**Device tree in kernel recipe:**

```python
# Specify device tree in machine conf
KERNEL_DEVICETREE = "\
    myvendor/mymachine.dtb \
    myvendor/mymachine-v2.dtb \
    myvendor/mymachine-overlay.dtbo \
"

# Add device tree sources to kernel
# recipes-kernel/linux/linux-custom/mymachine.dts
# Automatically compiled by kernel build
```

**Device tree overlay:**

```dts
/* mymachine-overlay.dts */
/dts-v1/;
/plugin/;

/ {
    compatible = "myvendor,mymachine";

    fragment@0 {
        target-path = "/";
        __overlay__ {
            custom-device {
                compatible = "myvendor,custom";
                status = "okay";
            };
        };
    };
};
```

### Creating a BSP Layer

```bash
# Create BSP layer
bitbake-layers create-layer meta-mymachine
cd meta-mymachine

# Structure:
mkdir -p conf/machine
mkdir -p recipes-bsp/u-boot
mkdir -p recipes-kernel/linux
mkdir -p recipes-core/images
mkdir -p wic

# Create machine configuration
cat > conf/machine/mymachine.conf << 'EOF'
#@TYPE: Machine
#@NAME: My Machine
#@DESCRIPTION: Machine configuration for My Hardware

DEFAULTTUNE = "cortexa72"
require conf/machine/include/arm/armv8a/tune-cortexa72.inc

PREFERRED_PROVIDER_virtual/kernel = "linux-mymachine"
PREFERRED_PROVIDER_virtual/bootloader = "u-boot-mymachine"

MACHINE_FEATURES = "usbhost usbgadget ext4 serial"
SERIAL_CONSOLES = "115200;ttyS0"

KERNEL_IMAGETYPE = "Image"
KERNEL_DEVICETREE = "myvendor/mymachine.dtb"

UBOOT_MACHINE = "mymachine_defconfig"

IMAGE_FSTYPES = "wic ext4"
WKS_FILE = "mymachine.wks"
EOF

# Create WIC kickstart
cat > wic/mymachine.wks << 'EOF'
# Partitioning for My Machine
# Boot partition (FAT32)
part /boot --source bootimg-partition --ondisk mmcblk0 --fstype=vfat --label boot --align 4096 --size 64M
# Root partition (ext4)
part / --source rootfs --ondisk mmcblk0 --fstype=ext4 --label root --align 4096 --size 1024M

bootloader --ptable msdos
EOF

# Add layer to build
bitbake-layers add-layer meta-mymachine
```

**BSP image recipe:**

```python
# recipes-core/images/mymachine-image.bb

require recipes-core/images/core-image-minimal.bb

SUMMARY = "Image for My Machine"

IMAGE_INSTALL += "\
    kernel-modules \
    u-boot-fw-utils \
    packagegroup-mymachine \
"

IMAGE_FEATURES += "ssh-server-dropbear"

# Machine-specific post-processing
ROOTFS_POSTPROCESS_COMMAND += "mymachine_setup; "

mymachine_setup() {
    # Custom setup for this machine
    echo "mymachine" > ${IMAGE_ROOTFS}/etc/hostname
}
```

## Package Management

### Package Formats

Yocto supports three package formats:

```python
# Select package format in local.conf
PACKAGE_CLASSES = "package_rpm"      # RPM (default)
# PACKAGE_CLASSES = "package_deb"    # Debian
# PACKAGE_CLASSES = "package_ipk"    # IPK (opkg)

# Multiple formats (discouraged)
# PACKAGE_CLASSES = "package_rpm package_deb"
```

**RPM (default):**

```python
# RPM-specific configuration
PACKAGE_CLASSES = "package_rpm"

# RPM architecture
PACKAGE_ARCH_RPM = "noarch"

# Repository configuration
PACKAGE_FEED_URIS = "http://example.com/rpm"
```

**DEB (Debian/Ubuntu style):**

```python
PACKAGE_CLASSES = "package_deb"

# Debian-specific
PACKAGE_FEED_URIS = "http://example.com/deb"
PACKAGE_FEED_BASE_PATHS = "deb"
PACKAGE_FEED_ARCHS = "all cortexa72"
```

**IPK (lightweight):**

```python
PACKAGE_CLASSES = "package_ipk"

# Used by opkg package manager
PACKAGE_FEED_URIS = "http://example.com/ipk"
```

### Package Feeds

**Creating package feeds:**

```python
# In local.conf
PACKAGE_CLASSES = "package_rpm"
PACKAGE_FEED_URIS = "http://myserver.com/rpm-feed"

# Build packages (without creating image)
bitbake <recipe-name> -c package_write_rpm

# Or build everything and populate feed
bitbake <image-name>
bitbake package-index
```

**Package feed directory structure:**

```
tmp/deploy/rpm/
├── all/                    # Architecture-independent
│   └── packagegroup-*.rpm
├── cortexa72/              # Architecture-specific
│   ├── kernel-*.rpm
│   └── myapp-*.rpm
├── mymachine/              # Machine-specific
│   └── firmware-*.rpm
└── repodata/               # RPM repository metadata
    └── repomd.xml
```

**Setting up HTTP package feed:**

```bash
# Serve packages via HTTP
cd tmp/deploy/rpm
python3 -m http.server 8000

# Or with nginx
# Copy tmp/deploy/rpm to /var/www/html/rpm-feed
```

**Client configuration:**

```python
# On target device
# For RPM (dnf/yum)
cat > /etc/yum.repos.d/custom.repo << EOF
[custom]
name=Custom Repository
baseurl=http://myserver:8000/cortexa72
enabled=1
gpgcheck=0
EOF

# For DEB (apt)
echo "deb http://myserver:8000/deb cortexa72/" > /etc/apt/sources.list.d/custom.list

# For IPK (opkg)
echo "src/gz custom http://myserver:8000/ipk/cortexa72" > /etc/opkg/custom.conf
```

### Runtime Package Management

**Enable package management in image:**

```python
# In image recipe or local.conf
IMAGE_FEATURES += "package-management"

# This includes:
# - Package manager (rpm/apt/opkg)
# - Package database
# - Repository configuration
```

**Using package managers on target:**

```bash
# RPM (dnf)
dnf update
dnf install mypackage
dnf search keyword
dnf remove mypackage

# DEB (apt)
apt update
apt install mypackage
apt search keyword
apt remove mypackage

# IPK (opkg)
opkg update
opkg install mypackage
opkg list | grep keyword
opkg remove mypackage
```

### Package Dependencies

**Build dependencies (DEPENDS):**

```python
# Required to build this recipe
DEPENDS = "zlib openssl virtual/kernel"

# These must complete do_populate_sysroot before
# this recipe's do_configure runs
```

**Runtime dependencies (RDEPENDS):**

```python
# Main package dependencies
RDEPENDS:${PN} = "bash libssl libz"

# Development package dependencies
RDEPENDS:${PN}-dev = "${PN} (= ${EXTENDPKGV})"

# Conditional dependencies
RDEPENDS:${PN} += "${@bb.utils.contains('DISTRO_FEATURES', 'systemd', 'systemd', 'sysvinit', d)}"

# Optional dependencies (recommended)
RRECOMMENDS:${PN} = "optional-package"
```

**Package provides/conflicts:**

```python
# Virtual package provision
PROVIDES = "virtual/some-service"

# Conflicts with other packages
RCONFLICTS:${PN} = "other-package"

# Replaces other packages
RREPLACES:${PN} = "old-package"

# Same functionality as another package
RPROVIDES:${PN} = "alternative-name"
```

**Dependency debugging:**

```bash
# Show runtime dependencies
bitbake -g <recipe>
# Generates pn-depends.dot

# Show why a package is included
bitbake -g <image> -u depexp
# Opens dependency explorer GUI

# List runtime dependencies of a package
oe-pkgdata-util list-pkg-files <package>
oe-pkgdata-util read-value RDEPENDS <package>
```

## Configuration System

### Configuration Files

**Configuration file hierarchy:**

```
Build Configuration Loading Order:
┌────────────────────────────────────┐
│ 1. bitbake.conf                    │  ← OE-Core defaults
├────────────────────────────────────┤
│ 2. layer.conf (each layer)         │  ← Layer settings
├────────────────────────────────────┤
│ 3. machine/*.conf                  │  ← Machine config
├────────────────────────────────────┤
│ 4. distro/*.conf                   │  ← Distribution config
├────────────────────────────────────┤
│ 5. bblayers.conf                   │  ← Layer list
├────────────────────────────────────┤
│ 6. local.conf                      │  ← Local overrides
└────────────────────────────────────┘
       Later files override earlier
```

### local.conf

**Key settings in local.conf:**

```python
# Machine selection
MACHINE = "qemux86-64"
# MACHINE = "raspberrypi4"
# MACHINE = "mymachine"

# Distribution
DISTRO = "poky"

# Package format
PACKAGE_CLASSES = "package_rpm"

# Parallel build jobs
BB_NUMBER_THREADS = "8"      # BitBake parallel tasks
PARALLEL_MAKE = "-j 8"       # Make parallel jobs

# Disk space monitoring
BB_DISKMON_DIRS = "\
    STOPTASKS,${TMPDIR},1G,100K \
    STOPTASKS,${DL_DIR},1G,100K \
    STOPTASKS,${SSTATE_DIR},1G,100K \
    ABORT,${TMPDIR},100M,1K \
    ABORT,${DL_DIR},100M,1K \
    ABORT,${SSTATE_DIR},100M,1K"

# Shared state cache
SSTATE_DIR = "${TOPDIR}/sstate-cache"
# SSTATE_DIR = "/common/sstate-cache"  # Shared location

# Download directory
DL_DIR = "${TOPDIR}/downloads"
# DL_DIR = "/common/downloads"  # Shared location

# SSTATE mirrors
SSTATE_MIRRORS = "\
    file://.* http://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH \
"

# Premirrors (check before original)
PREMIRRORS:prepend = "\
    git://.*/.* http://mirror.example.com/git/MIRRORNAME \
    ftp://.*/.* http://mirror.example.com/sources/ \
    http://.*/.* http://mirror.example.com/sources/ \
    https://.*/.* http://mirror.example.com/sources/ \
"

# Mirrors (fallback if download fails)
MIRRORS:prepend = "\
    ftp://.*/.* http://backup.example.com/sources/ \
    http://.*/.* http://backup.example.com/sources/ \
    https://.*/.* http://backup.example.com/sources/ \
"

# Remove packages from image
PACKAGE_EXCLUDE = "package1 package2"

# Additional image features
EXTRA_IMAGE_FEATURES = "debug-tweaks tools-debug"

# Root password (debug only!)
EXTRA_USERS_PARAMS = "usermod -P rootpass root;"

# SDK configuration
SDKMACHINE = "x86_64"

# RM_OLD_IMAGE: Remove previous image when building new one
RM_OLD_IMAGE = "1"

# Image name customization
IMAGE_NAME = "${IMAGE_BASENAME}-${MACHINE}-${DATETIME}"
```

**Development settings:**

```python
# Disable shared state cache (for testing)
# SSTATE_DIR = ""

# Keep temporary files after build
# RM_WORK_EXCLUDE += "recipe-name"

# Enable buildhistory
INHERIT += "buildhistory"
BUILDHISTORY_COMMIT = "1"

# Hash equivalence server (for distributed builds)
BB_HASHSERVE = "auto"
BB_SIGNATURE_HANDLER = "OEEquivHash"

# Enable network during build (for specific recipes only!)
# BB_NO_NETWORK = "0"  # Not recommended globally
```

### bblayers.conf

```python
# conf/bblayers.conf

# Build layers directory
BBPATH = "${TOPDIR}"

# Poky directory
BBFILES ?= ""

BBLAYERS ?= " \
  /path/to/poky/meta \
  /path/to/poky/meta-poky \
  /path/to/poky/meta-yocto-bsp \
  /path/to/meta-openembedded/meta-oe \
  /path/to/meta-openembedded/meta-python \
  /path/to/meta-openembedded/meta-networking \
  /path/to/meta-custom \
  /path/to/meta-mymachine \
"
```

**Managing layers:**

```bash
# Show configured layers
bitbake-layers show-layers

# Add layer
bitbake-layers add-layer /path/to/meta-layer

# Remove layer
bitbake-layers remove-layer meta-layer

# Show recipes provided by layer
bitbake-layers show-recipes -i meta-layer

# Show appends in layer
bitbake-layers show-appends
```

### Distribution Configuration

**Creating custom distribution:**

```python
# meta-custom/conf/distro/mydistro.conf

require conf/distro/poky.conf

DISTRO = "mydistro"
DISTRO_NAME = "My Custom Distribution"
DISTRO_VERSION = "1.0"
DISTRO_CODENAME = "custom"

MAINTAINER = "Your Name <your@email.com>"

# Target OS
TARGET_VENDOR = "-custom"

# Distribution features
DISTRO_FEATURES:append = " systemd"
DISTRO_FEATURES:remove = "sysvinit"
DISTRO_FEATURES_BACKFILL_CONSIDERED = "sysvinit"

# Init manager
VIRTUAL-RUNTIME_init_manager = "systemd"
VIRTUAL-RUNTIME_initscripts = ""

# Package management
PACKAGE_CLASSES = "package_rpm"

# Preferred versions
PREFERRED_VERSION_linux-yocto = "6.1%"
PREFERRED_VERSION_u-boot = "2024.01%"

# SDK
SDK_VENDOR = "-customsdk"
SDK_VERSION = "${DISTRO_VERSION}"

# Security flags
SECURITY_CFLAGS = "-fstack-protector-strong -D_FORTIFY_SOURCE=2"
SECURITY_LDFLAGS = "-Wl,-z,relro,-z,now"

# Reproducible builds
INHERIT += "reproducible_build"
```

**Using custom distribution:**

```python
# In local.conf
DISTRO = "mydistro"
```

### Machine Configuration Files

See [BSP Development - Machine Configuration](#machine-configuration) section for details.

### Variable Assignment

**Assignment operators summary:**

```python
# Immediate expansion
VAR := "value_${OTHER_VAR}"        # Expanded now

# Delayed expansion
VAR = "value_${OTHER_VAR}"         # Expanded when referenced

# Default (weak assignment)
VAR ?= "default"                   # Only if not set

# Weak default (weaker)
VAR ??= "weak"                     # Lowest priority

# Append with space
VAR += "append"                    # VAR = "original append"

# Prepend with space
VAR =+ "prepend"                   # VAR = "prepend original"

# Append without space
VAR .= "append"                    # VAR = "originalappend"

# Prepend without space
VAR =. "prepend"                   # VAR = "prependoriginal"

# Append (after all operations)
VAR:append = " late"               # Applied last

# Prepend (before all operations)
VAR:prepend = "early "             # Applied first

# Remove items
VAR:remove = "item"                # Removes "item" from VAR

# Override (conditional)
VAR:arm = "arm-value"              # Only for ARM
VAR:mymachine = "machine-value"    # Only for mymachine
```

**Order of operations:**

```
1. :prepend
2. =.
3. =+
4. = or :=
5. += or ?= or ??=
6. .=
7. :append
8. :remove
```

**Example:**

```python
VAR = "base"
VAR:prepend = "pre "
VAR:append = " post"
VAR += "add"

# Result: "pre base add post"
```

### Important Variables

**Paths:**

```python
# Top-level directories
TOPDIR          # Build directory
TMPDIR          # tmp/ (build artifacts)
DL_DIR          # downloads/
SSTATE_DIR      # sstate-cache/

# Deploy directories
DEPLOY_DIR              # tmp/deploy
DEPLOY_DIR_IMAGE        # tmp/deploy/images/${MACHINE}
DEPLOY_DIR_RPM          # tmp/deploy/rpm
DEPLOY_DIR_TAR          # tmp/deploy/tar

# Work directories
WORKDIR         # Recipe work directory
S               # Source directory
B               # Build directory
D               # Destination directory (install root)

# Staging
STAGING_DIR_HOST        # Sysroot for target
STAGING_DIR_NATIVE      # Sysroot for build host
STAGING_DIR_TARGET      # Sysroot for target (legacy)
```

**Build configuration:**

```python
MACHINE                 # Target machine
DISTRO                  # Distribution
TARGET_ARCH             # Target architecture (arm, x86_64, etc.)
TARGET_OS               # Target OS (linux, etc.)
TARGET_VENDOR           # Vendor string
TARGET_SYS              # Complete target system (arm-poky-linux-gnueabi)

BUILD_ARCH              # Build host architecture
BUILD_OS                # Build host OS
BUILD_SYS               # Build host system

HOST_ARCH               # Same as TARGET_ARCH
HOST_OS                 # Same as TARGET_OS
HOST_SYS                # Same as TARGET_SYS

TUNE_FEATURES           # CPU tuning features
DEFAULTTUNE             # Default tune settings
```

**Recipe metadata:**

```python
PN                      # Package name
PV                      # Package version
PR                      # Package revision
PE                      # Package epoch
PF                      # Package full name (PN-PV-PR)
BP                      # Base package (PN-PV)

SUMMARY                 # Short description
DESCRIPTION             # Long description
HOMEPAGE                # Project homepage
SECTION                 # Package category
LICENSE                 # License(s)
LIC_FILES_CHKSUM        # License file checksums
```

**Dependencies:**

```python
DEPENDS                 # Build-time dependencies
RDEPENDS                # Runtime dependencies
RRECOMMENDS             # Recommended packages
RSUGGESTS               # Suggested packages
RPROVIDES               # Virtual packages provided
RCONFLICTS              # Conflicting packages
RREPLACES               # Replaced packages
```

**Features:**

```python
DISTRO_FEATURES         # Distribution features
MACHINE_FEATURES        # Machine features
IMAGE_FEATURES          # Image features
COMBINED_FEATURES       # Intersection of DISTRO and MACHINE
```

## Advanced Features

### SDK Generation

**Standard SDK:**

```bash
# Build SDK for an image
bitbake core-image-minimal -c populate_sdk

# SDK location
ls tmp/deploy/sdk/
# poky-glibc-x86_64-core-image-minimal-cortexa72-mymachine-toolchain-5.0.sh
```

**SDK contents:**

```
SDK Directory Structure:
/opt/poky/5.0/
├── sysroots/
│   ├── x86_64-pokysdk-linux/        # Host tools
│   │   ├── usr/
│   │   │   ├── bin/                 # Cross-compilation tools
│   │   │   │   ├── arm-poky-linux-gcc
│   │   │   │   ├── arm-poky-linux-g++
│   │   │   │   └── ...
│   │   │   └── lib/
│   │   └── environment-setup-*      # Setup script
│   └── cortexa72-poky-linux/        # Target sysroot
│       ├── lib/                     # Target libraries
│       ├── usr/
│       │   ├── include/             # Headers
│       │   └── lib/                 # Development libraries
│       └── etc/
└── site-config-*
```

**Installing and using SDK:**

```bash
# Install SDK
./poky-glibc-x86_64-core-image-minimal-cortexa72-mymachine-toolchain-5.0.sh

# Default location: /opt/poky/5.0
# Or specify: ./sdk.sh -d /custom/path

# Setup environment
source /opt/poky/5.0/environment-setup-cortexa72-poky-linux

# Cross-compile application
$CC hello.c -o hello
# Uses arm-poky-linux-gcc with proper sysroot and flags

# Check variables
echo $CC
# arm-poky-linux-gcc -mthumb -mfpu=neon -mfloat-abi=hard -mcpu=cortex-a72 ...

echo $CXX
echo $CFLAGS
echo $LDFLAGS
```

**Customizing SDK:**

```python
# In image recipe
TOOLCHAIN_HOST_TASK:append = " nativesdk-cmake nativesdk-make"
TOOLCHAIN_TARGET_TASK:append = " libssl-dev zlib-dev"

# SDK name
SDK_NAME = "${DISTRO}-${TCLIBC}-${SDK_ARCH}-${IMAGE_BASENAME}-${TUNE_PKGARCH}"

# SDK output format
SDKIMAGE_FEATURES = "dev-pkgs dbg-pkgs src-pkgs"

# Post-install script
SDK_POST_INSTALL_COMMAND = "echo 'SDK installed' > ${SDK_OUTPUT}/${SDKPATH}/version"
```

### eSDK (Extensible SDK)

**Build eSDK:**

```bash
# Build extensible SDK
bitbake core-image-minimal -c populate_sdk_ext

# Output
ls tmp/deploy/sdk/
# poky-glibc-x86_64-core-image-minimal-cortexa72-mymachine-toolchain-ext-5.0.sh
```

**eSDK features:**

- Contains full build system
- Can build recipes from source
- Includes devtool for development workflow
- Larger than standard SDK (~5-10 GB)
- Self-contained build environment

**Using eSDK:**

```bash
# Install eSDK
./poky-*-toolchain-ext-*.sh

# Setup environment
source /opt/poky/5.0/environment-setup-cortexa72-poky-linux

# Use devtool
devtool add myapp https://github.com/user/myapp.git
devtool build myapp
devtool deploy-target myapp root@192.168.1.100
```

### devtool

**devtool workflow:**

```bash
# Add new recipe (creates workspace)
devtool add myapp https://github.com/user/myapp.git

# Creates:
# workspace/sources/myapp/     - Source code
# workspace/recipes/myapp/     - Recipe
# workspace/appends/myapp.bbappend

# Modify source
cd workspace/sources/myapp
# Make changes...

# Build recipe
devtool build myapp

# Build and create package
devtool build myapp -c package

# Test on target device
devtool deploy-target myapp root@192.168.1.100

# Update recipe (after source changes)
devtool update-recipe myapp

# Create patch from changes
devtool finish myapp meta-custom

# This:
# - Creates final recipe in meta-custom
# - Generates patches for source changes
# - Removes from workspace
```

**devtool commands:**

```bash
# Create new recipe
devtool add <recipe> <source>

# Modify existing recipe
devtool modify <recipe>

# Build recipe
devtool build <recipe>

# Build SDK for recipe
devtool build-sdk <recipe>

# Deploy to target
devtool deploy-target <recipe> <target>

# Undeploy from target
devtool undeploy-target <recipe> <target>

# Update recipe with changes
devtool update-recipe <recipe>

# Finalize recipe
devtool finish <recipe> <layer>

# Reset workspace
devtool reset <recipe>

# Search for recipe
devtool search <pattern>

# Show recipe info
devtool latest-version <recipe>

# Configure devtool
devtool configure-help <recipe>
```

**devtool configuration:**

```python
# workspace/devtool.conf (auto-generated)
# Configure target device
DEVTOOL_TARGET = "root@192.168.1.100"

# Configure deployment directory
DEVTOOL_DEPLOY_DIR = "/opt/deployed"
```

### Multiconfig Builds

**Multiconfig** enables building for multiple configurations in one build:

```bash
# Create additional config
cp conf/local.conf conf/multiconfig/target2.conf

# Edit target2.conf
sed -i 's/MACHINE = "qemux86-64"/MACHINE = "raspberrypi4"/' conf/multiconfig/target2.conf
```

**Enable multiconfig:**

```python
# In local.conf
BBMULTICONFIG = "target2 target3"

# This loads:
# - conf/multiconfig/target2.conf
# - conf/multiconfig/target3.conf
```

**Build with multiconfig:**

```bash
# Build for default config
bitbake core-image-minimal

# Build for specific multiconfig
bitbake mc:target2:core-image-minimal

# Build for all configs
bitbake mc::core-image-minimal mc:target2:core-image-minimal

# Or use dependency
# In recipe:
# do_something[mcdepends] = "mc:target2:core-image-minimal:do_image_complete"
```

**Use cases:**

- Multiple machine targets
- Different distro configurations
- SDK and target in one build
- Different optimization levels

### SSTATE and Build Acceleration

**Shared state cache optimization:**

```python
# Share SSTATE between builds
SSTATE_DIR = "/common/sstate-cache"

# SSTATE mirrors (read-only)
SSTATE_MIRRORS = "\
    file://.* http://sstate-mirror.example.com/PATH;downloadfilename=PATH \
    file://.* file:///network/sstate/PATH;downloadfilename=PATH \
"

# Publish to SSTATE mirror
SSTATE_UPLOAD = "1"
SSTATE_UPLOAD_DIR = "/network/sstate"
```

**Hash equivalence:**

```python
# Enable hash equivalence server
BB_HASHSERVE = "auto"  # Start local server
# BB_HASHSERVE = "hashserv.example.com:8686"  # Remote server

BB_SIGNATURE_HANDLER = "OEEquivHash"

# Allows reusing binaries even if hashes differ
# but build inputs are equivalent
```

**Build history:**

```python
# Enable build history
INHERIT += "buildhistory"
BUILDHISTORY_COMMIT = "1"
BUILDHISTORY_DIR = "${TOPDIR}/buildhistory"

# Track:
# - Installed packages
# - Package sizes
# - License information
# - Recipe changes

# View history
git -C buildhistory log
git -C buildhistory diff HEAD~1
```

**RM_WORK class:**

```python
# Automatically remove temporary files after build
INHERIT += "rm_work"

# Exclude specific recipes
RM_WORK_EXCLUDE += "myapp linux-yocto"

# Saves disk space but requires rebuild if task re-runs
```

### Template Layer

**Using meta-skeleton as template:**

```bash
# Copy template layer
cp -r poky/meta-skeleton /path/to/meta-custom

# Contains examples:
# - recipes-kernel/hello-mod/        # Kernel module
# - recipes-core/hello/              # Autotools app
# - recipes-core/hello-makefile/     # Makefile app
# - conf/layer.conf.sample           # Layer config
```

## Debugging and Troubleshooting

### BitBake Debug Options

**Verbose output:**

```bash
# Verbose logging
bitbake -v <recipe>

# Debug logging
bitbake -D <recipe>

# Very verbose debug
bitbake -DD <recipe>

# Specific log domain
bitbake -l DEBUG <recipe>
```

**Dry run and dependency analysis:**

```bash
# Dry run (parse only)
bitbake -n <recipe>

# Show what would be built
bitbake -s

# Dependency graph
bitbake -g <recipe>
# Creates: pn-depends.dot, task-depends.dot, pn-buildlist

# Dependency tree
bitbake -g <recipe> -u taskexp

# Why is package included?
bitbake -g <image>
grep "package-name" pn-depends.dot
```

**Task execution:**

```bash
# Run specific task
bitbake <recipe> -c compile

# Force re-run task
bitbake <recipe> -c compile -f

# Continue after task
bitbake <recipe> -c compile --continue

# Clean tasks
bitbake <recipe> -c clean        # Clean work files
bitbake <recipe> -c cleansstate  # Clean sstate
bitbake <recipe> -c cleanall     # Clean everything including downloads
```

**Interactive development:**

```bash
# Run task and drop into shell
bitbake <recipe> -c devshell

# Build environment available:
# - $CC, $CXX, $CFLAGS, etc.
# - Source in $S
# - Build in $B

# Python shell for recipe
bitbake <recipe> -c devpyshell

# Available:
# - d (datastore)
# - d.getVar('VARNAME')
```

### Build Failures

**Common build failure patterns:**

**1. Fetch failures:**

```bash
# Error: Fetcher failure for URL...

# Causes:
# - Network issues
# - Incorrect URL
# - Missing checksum

# Solutions:
# Check SRC_URI
bitbake <recipe> -c fetch -f

# Verify checksum
bitbake <recipe> -c checksum

# Use mirror
PREMIRRORS:prepend = "git://.*/.* http://mirror.example.com/git/MIRRORNAME"
```

**2. Configure failures:**

```bash
# Error: configure failed

# Check log
cat tmp/work/.../temp/log.do_configure

# Common causes:
# - Missing dependencies
# - Wrong configure flags

# Solutions:
# Add missing dependencies to DEPENDS
DEPENDS += "missing-package"

# Adjust configure options
EXTRA_OECONF += "--disable-problematic-feature"

# Manual configure
bitbake <recipe> -c devshell
./configure ...
```

**3. Compile failures:**

```bash
# Error: compilation failed

# Check log
cat tmp/work/.../temp/log.do_compile

# Common causes:
# - Incompatible compiler flags
# - Missing headers

# Solutions:
# Adjust compiler flags
CFLAGS:append = " -Wno-error=specific-warning"

# Add include directory
CFLAGS:append = " -I${STAGING_INCDIR}/custom"

# Apply patches
SRC_URI += "file://fix-build.patch"
```

**4. QA failures:**

```bash
# Error: QA Issue: ...

# Common QA issues:
# - already-stripped: Binaries stripped during install
# - installed-vs-shipped: Files installed but not packaged
# - host-user-contaminated: Files owned by build user
# - textrel: Text relocations in shared libraries

# Solutions:
# Skip specific QA check (not recommended)
INSANE_SKIP:${PN} += "already-stripped"

# Fix packaging
FILES:${PN} += "/path/to/installed/file"

# Fix installation
do_install:append() {
    # Don't strip here, let OE handle it
}
```

### Dependency Issues

**Missing dependencies:**

```bash
# Error: Nothing PROVIDES 'package'

# Solutions:
# Add layer containing package
bitbake-layers add-layer meta-layer

# Check layer compatibility
bitbake-layers show-layers

# Search for recipe
bitbake-layers show-recipes package

# Add dependency
DEPENDS += "package"
```

**Circular dependencies:**

```bash
# Error: Circular dependency

# Find cycle
bitbake -g <recipe>
grep -E "(recipe1.*recipe2|recipe2.*recipe1)" pn-depends.dot

# Solutions:
# - Remove unnecessary dependency
# - Split recipe into multiple packages
# - Use DEPENDS vs RDEPENDS appropriately
```

**Version conflicts:**

```bash
# Error: Multiple versions of recipe

# Check available versions
bitbake-layers show-recipes <recipe>

# Select specific version
PREFERRED_VERSION_<recipe> = "1.2.3"

# Or in recipe
DEFAULT_PREFERENCE = "-1"  # Prefer other versions
```

### Task Debugging

**Examine task scripts:**

```bash
# View generated task script
cat tmp/work/.../temp/run.do_compile

# Contains actual shell commands executed
# Useful for understanding what BitBake does
```

**Environment dump:**

```bash
# Dump all variables for recipe
bitbake -e <recipe> > recipe-env.txt

# Search for specific variable
bitbake -e <recipe> | grep ^DEPENDS=

# Show variable history
bitbake -e <recipe> | grep -A 10 "^# DEPENDS"
```

**Debugging Python tasks:**

```python
# Add debug output in recipe
python do_custom_task() {
    bb.warn("Debug message")
    bb.note("Info message")
    bb.error("Error message")

    # Print variables
    var = d.getVar('SOME_VAR')
    bb.warn(f"SOME_VAR = {var}")
}
```

**Logging:**

```bash
# Build log location
tmp/work/<arch>/<recipe>/<version>/temp/log.do_<task>

# Example
tmp/work/cortexa72-poky-linux/myapp/1.0-r0/temp/log.do_compile

# View recent logs
ls -lt tmp/work/*/*/*/temp/log.do_* | head
```

### Common Problems

**Problem: Disk space issues**

```bash
# Clean old builds
bitbake <recipe> -c cleanall

# Remove tmp (rebuild from scratch)
rm -rf tmp

# Use rm_work class
INHERIT += "rm_work"

# Clean specific states
bitbake <recipe> -c cleansstate
```

**Problem: State corruption**

```bash
# Clean and rebuild
bitbake <recipe> -c cleansstate
bitbake <recipe>

# Force re-parse
bitbake -e  # Then Ctrl+C
bitbake <recipe>

# Complete rebuild
rm -rf tmp sstate-cache
bitbake <recipe>
```

**Problem: Parse errors**

```bash
# Error: ParseError at recipe.bb:10

# Check syntax
# - Missing quotes
# - Unclosed strings
# - Wrong operators

# Validate Python syntax
python3 -m py_compile recipe.bb
```

**Problem: Patch failures**

```bash
# Error: patch failed

# Check patch
cat recipes-*/recipe/recipe/0001-fix.patch

# Verify patch applies
cd tmp/work/.../recipe-version
patch -p1 --dry-run < path/to/patch

# Update patch
devtool modify recipe
# Make changes
devtool finish recipe meta-custom
```

**Problem: Missing files in package**

```bash
# QA Issue: Files installed but not shipped

# Check what's installed
oe-pkgdata-util list-pkg-files <package>

# Fix FILES variable
FILES:${PN} += "/path/to/file"

# Or create new package
PACKAGES =+ "${PN}-extra"
FILES:${PN}-extra = "/path/to/extra/files"
```

## Best Practices

### Layer Design

**Layer organization:**

```
meta-company/
├── conf/
│   └── layer.conf
├── COPYING.MIT
├── README.md
│
├── classes/                    # Shared classes
│   └── company.bbclass
│
├── recipes-core/
│   └── images/
│       └── company-image.bb
│
├── recipes-bsp/               # Board support
│   ├── u-boot/
│   └── firmware/
│
├── recipes-kernel/            # Kernel customization
│   └── linux/
│
├── recipes-support/           # Libraries, utilities
│
└── recipes-apps/              # Applications
    └── company-app/
```

**Layer best practices:**

1. **One purpose per layer:**
   - BSP layer: Hardware support only
   - Distro layer: Distribution config only
   - App layer: Applications only

2. **Clear dependencies:**
   - Document in README
   - Set LAYERDEPENDS
   - Specify LAYERSERIES_COMPAT

3. **Maintainability:**
   - Use meaningful names
   - Version control
   - Document changes

4. **Compatibility:**
   - Test with multiple Yocto releases
   - Update LAYERSERIES_COMPAT
   - Pin SRCREV for stability

### Recipe Patterns

**Recipe naming:**

```python
# Format: <name>_<version>.bb
myapp_1.0.bb          # Good
myapp.bb              # Bad: no version

# Multiple versions
myapp_1.0.bb
myapp_1.1.bb
myapp_git.bb          # Development version
```

**Source management:**

```python
# Prefer specific commits over AUTOREV
SRCREV = "abc123def456..."
# Not: SRCREV = "${AUTOREV}"  # Unreproducible

# Use PV from tag
PV = "1.0+git${SRCPV}"

# Separate local files
SRC_URI = "git://github.com/user/repo.git;protocol=https \
           file://0001-fix.patch \
           file://config.in \
          "
```

**Dependency specification:**

```python
# Build-time dependencies
DEPENDS = "zlib virtual/kernel"

# Runtime dependencies (package-specific)
RDEPENDS:${PN} = "bash libz"
RDEPENDS:${PN}-dev = "${PN} (= ${EXTENDPKGV})"

# Avoid:
# RDEPENDS:${PN} = "${PN}-dev"  # Wrong: circular
```

**Task organization:**

```python
# Prefer :append over complete override
do_install:append() {
    # Additional install steps
}

# Not (unless necessary):
do_install() {
    # Complete replacement - loses base functionality
}

# Use :prepend for setup
do_configure:prepend() {
    ./autogen.sh
}
```

### Version Management

**Version pinning:**

```python
# In distro config or local.conf
PREFERRED_VERSION_linux-yocto = "6.1%"
PREFERRED_VERSION_u-boot = "2024.01%"

# % allows any PR
# Without %: exact match required
```

**Recipe versioning:**

```python
# Use PV from source
PV = "1.0"

# For git recipes
PV = "1.0+git${SRCPV}"

# PR for recipe changes (usually not needed)
PR = "r0"

# Increment PR for recipe-only changes
# (changes that don't affect PV)
```

**SRCREV management:**

```python
# Development: use branch
SRC_URI = "git://github.com/user/repo.git;protocol=https;branch=develop"
SRCREV = "${AUTOREV}"

# Production: use specific commit
SRC_URI = "git://github.com/user/repo.git;protocol=https"
SRCREV = "abc123def456..."

# Tagged release
SRC_URI = "git://github.com/user/repo.git;protocol=https;tag=v1.0"
SRCREV = "${AUTOREV}"
```

### Build Reproducibility

**Reproducible builds:**

```python
# Enable reproducible build class
inherit reproducible_build

# This ensures:
# - Timestamps are normalized
# - Build paths are consistent
# - File ordering is deterministic
```

**Version control metadata:**

```python
# Don't use timestamps in versions
# Bad:
PV = "1.0+${DATE}"

# Good:
PV = "1.0+git${SRCPV}"

# Lock down sources
SRCREV = "specific-commit-hash"
SRC_URI[sha256sum] = "specific-checksum"
```

**Build isolation:**

```python
# Disable network access (enforced by default)
BB_NO_NETWORK = "1"

# Prevent host contamination
# - Don't reference /usr or /opt
# - Use STAGING_* variables
# - Don't hardcode paths

# Bad:
CFLAGS += "-I/usr/include/custom"

# Good:
DEPENDS = "custom"
CFLAGS += "-I${STAGING_INCDIR}"
```

**Shared state:**

```python
# Use consistent SSTATE_DIR
SSTATE_DIR = "/shared/sstate-cache"

# Configure SSTATE mirrors
SSTATE_MIRRORS = "file://.* http://sstate-server.local/PATH;downloadfilename=PATH"

# Enable hash equivalence
BB_HASHSERVE = "auto"
BB_SIGNATURE_HANDLER = "OEEquivHash"
```

## Command Reference

### BitBake Commands

```bash
# Basic build
bitbake <recipe>

# Build image
bitbake <image-recipe>

# Show available recipes
bitbake -s

# Show recipe info
bitbake-layers show-recipes <pattern>

# Parse only (dry run)
bitbake -n <recipe>

# Clean tasks
bitbake <recipe> -c clean       # Clean work directory
bitbake <recipe> -c cleansstate # Clean shared state
bitbake <recipe> -c cleanall    # Clean everything

# Run specific task
bitbake <recipe> -c <task>

# Force task execution
bitbake <recipe> -c <task> -f

# List tasks
bitbake <recipe> -c listtasks

# Dependency graph
bitbake -g <recipe>

# Environment dump
bitbake -e <recipe> > env.log

# Continue despite errors
bitbake -k <recipe>

# Development shell
bitbake <recipe> -c devshell

# Python shell
bitbake <recipe> -c devpyshell

# Build world (all recipes)
bitbake world

# Build SDK
bitbake <image> -c populate_sdk

# Build eSDK
bitbake <image> -c populate_sdk_ext

# Multiconfig build
bitbake mc:config:recipe
```

### devtool Commands

```bash
# Add new recipe
devtool add <name> <source-url>

# Modify existing recipe
devtool modify <recipe>

# Build recipe
devtool build <recipe>

# Deploy to target
devtool deploy-target <recipe> root@<ip>

# Undeploy from target
devtool undeploy-target <recipe> root@<ip>

# Update recipe
devtool update-recipe <recipe>

# Finish development
devtool finish <recipe> <layer>

# Reset workspace
devtool reset <recipe>

# Search recipes
devtool search <pattern>

# Show latest version
devtool latest-version <recipe>

# Create workspace
devtool create-workspace <path>

# Configure recipe
devtool configure <recipe>

# Build image with workspace
devtool build-image <image>

# Export recipe
devtool export <recipe>

# Import recipe
devtool import <exported-file>
```

### recipetool Commands

```bash
# Create recipe from source
recipetool create <source-url>

# Create with output file
recipetool create -o <recipe.bb> <source>

# Append to recipe
recipetool appendfile <recipe> <file> <source-file>

# Append source file
recipetool appendsrcfile <recipe> <file> <source-file>

# Append source files
recipetool appendsrcfiles <recipe> <files...>

# New appends
recipetool newappend <recipe>

# Set variable
recipetool setvar <recipe> <variable> <value>
```

### Common Workflows

**New recipe development:**

```bash
# 1. Create recipe
recipetool create -o myapp_1.0.bb https://github.com/user/myapp.git

# 2. Or use devtool
devtool add myapp https://github.com/user/myapp.git

# 3. Modify source
cd workspace/sources/myapp
# Make changes...

# 4. Build and test
devtool build myapp
devtool deploy-target myapp root@192.168.1.100

# 5. Update recipe
devtool update-recipe myapp

# 6. Finalize
devtool finish myapp meta-custom
```

**Modify existing recipe:**

```bash
# 1. Start modification
devtool modify linux-yocto

# 2. Make changes
cd workspace/sources/linux-yocto
# Edit source...

# 3. Build
devtool build linux-yocto

# 4. Generate patches
devtool update-recipe linux-yocto

# 5. Finish
devtool finish linux-yocto meta-custom
```

**Image customization:**

```bash
# 1. Create custom layer
bitbake-layers create-layer meta-custom
bitbake-layers add-layer meta-custom

# 2. Create image recipe
mkdir -p meta-custom/recipes-core/images
cat > meta-custom/recipes-core/images/custom-image.bb << 'EOF'
require recipes-core/images/core-image-minimal.bb
SUMMARY = "Custom image"
IMAGE_INSTALL += "myapp python3"
IMAGE_FEATURES += "ssh-server-openssh"
EOF

# 3. Build image
bitbake custom-image

# 4. Test
runqemu custom-image
```

**BSP creation:**

```bash
# 1. Create BSP layer
bitbake-layers create-layer meta-mybsp
cd meta-mybsp

# 2. Create machine config
mkdir -p conf/machine
cat > conf/machine/mymachine.conf << 'EOF'
#@TYPE: Machine
#@NAME: My Machine
DEFAULTTUNE = "cortexa72"
require conf/machine/include/arm/armv8a/tune-cortexa72.inc
PREFERRED_PROVIDER_virtual/kernel = "linux-yocto"
KERNEL_IMAGETYPE = "Image"
SERIAL_CONSOLES = "115200;ttyS0"
MACHINE_FEATURES = "serial usbhost"
IMAGE_FSTYPES = "ext4 wic"
EOF

# 3. Add layer and build
bitbake-layers add-layer meta-mybsp
MACHINE=mymachine bitbake core-image-minimal
```

**Debugging failed build:**

```bash
# 1. Check error
bitbake myapp

# 2. View log
cat tmp/work/.../temp/log.do_compile

# 3. Development shell
bitbake myapp -c devshell
# Try manual build...

# 4. Force rebuild
bitbake myapp -c compile -f

# 5. Check dependencies
bitbake -g myapp
grep myapp pn-depends.dot

# 6. Clean and rebuild
bitbake myapp -c cleansstate
bitbake myapp
```

## Resources

**Official Documentation:**
- Yocto Project: https://www.yoctoproject.org/
- Documentation: https://docs.yoctoproject.org/
- Reference Manual: https://docs.yoctoproject.org/ref-manual/
- Developer Manual: https://docs.yoctoproject.org/dev-manual/
- BitBake Manual: https://docs.yoctoproject.org/bitbake/

**Layer Index:**
- OpenEmbedded Layer Index: https://layers.openembedded.org/
- Find existing layers and recipes

**Mailing Lists:**
- yocto@lists.yoctoproject.org
- openembedded-core@lists.openembedded.org

**IRC Channels:**
- #yocto on libera.chat
- #openembedded on libera.chat

**Source Code:**
- Poky: https://git.yoctoproject.org/poky/
- OpenEmbedded-Core: https://git.openembedded.org/openembedded-core/
- BitBake: https://git.openembedded.org/bitbake/

**Training and Books:**
- Embedded Linux Development using Yocto Project
- Linux Foundation training courses
- Free Software Foundation training

**Related Documentation in This Repository:**
- See [Cross Compilation](cross_compilation.md) for basic Yocto workflow
- See [Device Tree](device_tree.md) for device tree integration
- See [Driver Development](driver_development.md) for kernel module recipes
- See [Kernel](kernel.md) for kernel concepts
