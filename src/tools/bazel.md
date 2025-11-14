# Bazel

Bazel is a fast, scalable, multi-language build system developed by Google. It provides reproducible and hermetic builds with strong support for remote execution and caching.

## Overview

Bazel is an open-source build and test tool that scales to large codebases across multiple repositories and languages. It only rebuilds what is necessary and leverages advanced caching for fast, incremental builds.

**Key Concepts:**
- **Workspace**: Root directory containing source code and BUILD files
- **Package**: Directory with a BUILD file containing related targets
- **Target**: Unit of build (file, rule, or package group)
- **Label**: Unique identifier for a target (e.g., `//path/to/package:target`)
- **Rule**: Function that defines how to build outputs from inputs
- **BUILD File**: Declares targets and their dependencies

**Why Bazel:**
- **Fast**: Only rebuilds what changed, uses advanced caching
- **Correct**: Hermetic builds ensure reproducibility
- **Scalable**: Handles large codebases and monorepos
- **Multi-language**: Single tool for C++, Java, Python, Go, and more
- **Remote Execution**: Distribute builds across multiple machines

## Installation

### Using Bazelisk (Recommended)

```bash
# Install Bazelisk (manages Bazel versions automatically)
# macOS
brew install bazelisk

# Linux (download binary)
wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel

# Verify installation
bazel --version
```

### Direct Installation

```bash
# Ubuntu/Debian
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel

# macOS
brew install bazel

# From source or binary
# https://github.com/bazelbuild/bazel/releases
```

### Version Management

```bash
# Use .bazelversion file to specify version
echo "7.0.0" > .bazelversion

# Bazelisk reads this file and downloads the correct version
bazel version
```

## Basic Concepts

### Workspace Structure

```
my_project/
├── WORKSPACE (or MODULE.bazel for Bzlmod)
├── .bazelrc
├── BUILD.bazel
├── src/
│   ├── BUILD.bazel
│   ├── main.cc
│   └── lib/
│       ├── BUILD.bazel
│       └── helper.cc
└── tests/
    ├── BUILD.bazel
    └── main_test.cc
```

### Labels and Targets

```python
# Label syntax
//package:target          # Target in another package
:target                  # Target in current package
//package                # Shorthand for //package:package
@repo//package:target    # Target in external repository

# Target patterns
//...                    # All targets in workspace
//path/to/package/...    # All targets under path
//path/to/package:*      # All targets in package
//path/to/package:all    # All targets in package
```

### Dependencies

```python
# Direct dependencies
deps = [
    ":local_target",
    "//other/package:target",
    "@external_repo//package:target",
]

# Data dependencies (runtime files)
data = [
    "config.json",
    "//data:test_files",
]
```

## Basic Commands

### Build Commands

```bash
# Build a target
bazel build //path/to/package:target
bazel build :target  # In current package
bazel build //...    # Build everything

# Build with flags
bazel build //src:main --compilation_mode=opt
bazel build //src:main -c opt  # Optimized build
bazel build //src:main -c dbg  # Debug build

# Build multiple targets
bazel build //src:main //tests:all

# Show build commands
bazel build //src:main --verbose_failures
bazel build //src:main -s  # Show all commands
```

### Test Commands

```bash
# Run tests
bazel test //tests:all
bazel test //...  # Run all tests

# Test with options
bazel test //tests:main_test --test_output=all
bazel test //tests:main_test --test_output=errors
bazel test //tests:main_test --test_output=streamed

# Run specific test
bazel test //tests:main_test --test_filter=TestName

# Test with coverage
bazel coverage //tests:all
```

### Run Commands

```bash
# Run a binary target
bazel run //src:main
bazel run //src:main -- arg1 arg2  # With arguments

# Run with configuration
bazel run -c opt //src:main
```

### Query Commands

```bash
# Query target information
bazel query //src:main
bazel query 'deps(//src:main)'  # Show all dependencies
bazel query 'rdeps(//..., //src:lib)'  # Reverse dependencies

# Find targets
bazel query 'kind("cc_.*", //...)'  # All C++ targets
bazel query 'attr(name, ".*test.*", //...)'  # Targets matching pattern

# Build graph
bazel query 'somepath(//src:main, //third_party:lib)'
```

### Info Commands

```bash
# Workspace information
bazel info
bazel info workspace
bazel info bazel-bin
bazel info output_path

# Build information
bazel info compilation_mode
bazel info cpu
```

### Clean Commands

```bash
# Clean build outputs (keeps cache)
bazel clean

# Deep clean (removes all cached artifacts)
bazel clean --expunge

# Async clean (non-blocking)
bazel clean --async
```

## BUILD Files

### Basic Syntax

```python
# src/BUILD.bazel

# Load rules from external files
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# C++ library
cc_library(
    name = "helper",
    srcs = ["helper.cc"],
    hdrs = ["helper.h"],
    visibility = ["//visibility:public"],
)

# C++ binary
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":helper"],
)

# Multiple targets in one file
cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@googletest//:gtest_main",
    ],
)
```

### Common Attributes

```python
cc_binary(
    name = "myapp",              # Target name (required)
    srcs = ["main.cc"],          # Source files
    hdrs = ["main.h"],           # Header files
    deps = [":lib"],             # Dependencies
    data = ["config.json"],      # Runtime data files
    copts = ["-std=c++17"],      # Compiler options
    linkopts = ["-lpthread"],    # Linker options
    defines = ["DEBUG=1"],       # Preprocessor defines
    includes = ["include/"],     # Include directories
    visibility = ["//visibility:public"],  # Who can depend on this
    testonly = False,            # Only for tests
    tags = ["manual"],           # Metadata tags
)
```

### Visibility

```python
# Public - anyone can depend on this
visibility = ["//visibility:public"]

# Private - only targets in same package
visibility = ["//visibility:private"]

# Package group
package_group(
    name = "friends",
    packages = [
        "//src/...",
        "//tests/...",
    ],
)

cc_library(
    name = "internal_lib",
    srcs = ["lib.cc"],
    visibility = [":friends"],
)

# Specific packages
visibility = [
    "//src:__pkg__",        # Only src package
    "//src:__subpackages__", # src and all subpackages
]
```

### Glob Patterns

```python
# Glob for source files
cc_library(
    name = "lib",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
)

# Glob with exclusions
srcs = glob(
    ["**/*.cc"],
    exclude = [
        "test/**",
        "**/*_test.cc",
    ],
)

# Recursive glob
srcs = glob(["**/*.cc"])  # All .cc files recursively

# Include generated files explicitly (glob doesn't include them)
srcs = glob(["*.cc"]) + [":generated_source"]
```

## Common Build Rules

### C/C++ Rules

```python
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

# C++ library
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = ["mylib.h"],
    deps = [":other_lib"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
)

# C++ binary
cc_binary(
    name = "myapp",
    srcs = ["main.cc"],
    deps = [":mylib"],
    linkopts = ["-lpthread"],
)

# C++ test
cc_test(
    name = "mylib_test",
    srcs = ["mylib_test.cc"],
    deps = [
        ":mylib",
        "@googletest//:gtest_main",
    ],
    size = "small",  # small, medium, large, enormous
)

# Static library
cc_library(
    name = "static_lib",
    srcs = ["lib.cc"],
    linkstatic = True,
)

# Shared library
cc_binary(
    name = "libshared.so",
    srcs = ["lib.cc"],
    linkshared = True,
)
```

### Java Rules

```python
load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")

# Java library
java_library(
    name = "mylib",
    srcs = glob(["*.java"]),
    deps = [
        "@maven//:com_google_guava_guava",
    ],
    resources = glob(["resources/**"]),
)

# Java binary
java_binary(
    name = "myapp",
    srcs = ["Main.java"],
    main_class = "com.example.Main",
    deps = [":mylib"],
)

# Java test
java_test(
    name = "mylib_test",
    srcs = ["MyLibTest.java"],
    test_class = "com.example.MyLibTest",
    deps = [
        ":mylib",
        "@maven//:junit_junit",
    ],
)
```

### Python Rules

```python
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# Python library
py_library(
    name = "mylib",
    srcs = ["mylib.py"],
    deps = [":other_lib"],
    data = ["data.json"],
)

# Python binary
py_binary(
    name = "myapp",
    srcs = ["main.py"],
    deps = [":mylib"],
    python_version = "PY3",
)

# Python test
py_test(
    name = "mylib_test",
    srcs = ["mylib_test.py"],
    deps = [
        ":mylib",
        "@pip//pytest",
    ],
)
```

### Go Rules

```python
load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

# Go library
go_library(
    name = "go_default_library",
    srcs = ["lib.go"],
    importpath = "github.com/user/project/lib",
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_golang_glog//:go_default_library",
    ],
)

# Go binary
go_binary(
    name = "myapp",
    srcs = ["main.go"],
    deps = [":go_default_library"],
)

# Go test
go_test(
    name = "go_default_test",
    srcs = ["lib_test.go"],
    embed = [":go_default_library"],
)
```

### Protocol Buffers

```python
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

# Proto definition
proto_library(
    name = "person_proto",
    srcs = ["person.proto"],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
    ],
)

# C++ proto library
cc_proto_library(
    name = "person_cc_proto",
    deps = [":person_proto"],
)

# Use in C++ target
cc_binary(
    name = "myapp",
    srcs = ["main.cc"],
    deps = [":person_cc_proto"],
)
```

### Genrule (Custom Build Commands)

```python
# Generate files with custom commands
genrule(
    name = "generate_version",
    srcs = ["version.txt.template"],
    outs = ["version.txt"],
    cmd = "sed 's/VERSION/$(VERSION)/g' $< > $@",
)

# Multiple outputs
genrule(
    name = "codegen",
    srcs = ["schema.json"],
    outs = [
        "generated.h",
        "generated.cc",
    ],
    cmd = "$(location :generator) --input $< --output $(RULEDIR)",
    tools = [":generator"],
)

# Use environment variables
genrule(
    name = "config",
    outs = ["config.h"],
    cmd = """
        echo '#define BUILD_TIME "$(DATE)"' > $@
        echo '#define BUILD_HOST "$(HOSTNAME)"' >> $@
    """,
)
```

### Shell Scripts

```python
# Shell binary
sh_binary(
    name = "deploy",
    srcs = ["deploy.sh"],
    data = [
        ":myapp",
        "config.yaml",
    ],
)

# Shell test
sh_test(
    name = "integration_test",
    srcs = ["test.sh"],
    data = [
        ":myapp",
        "//testdata:files",
    ],
)
```

## WORKSPACE and MODULE.bazel

### WORKSPACE (Legacy)

```python
# WORKSPACE
workspace(name = "my_project")

# Load HTTP archive rule
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# External dependency from archive
http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/release-1.12.1.tar.gz"],
    strip_prefix = "googletest-release-1.12.1",
    sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
)

# Git repository
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "my_dependency",
    remote = "https://github.com/user/repo.git",
    tag = "v1.0.0",
    # Or use commit
    # commit = "abc123",
)

# Local repository
local_repository(
    name = "local_lib",
    path = "../local_lib",
)

# New local repository (creates BUILD files)
new_local_repository(
    name = "external_lib",
    path = "/usr/local/lib/mylib",
    build_file = "external/mylib.BUILD",
)
```

### MODULE.bazel (Bzlmod - Modern)

```python
# MODULE.bazel
module(
    name = "my_project",
    version = "1.0.0",
)

# Bazel dependencies
bazel_dep(name = "rules_cc", version = "0.0.9")
bazel_dep(name = "rules_python", version = "0.27.0")
bazel_dep(name = "googletest", version = "1.14.0")

# Override dependency version
single_version_override(
    module_name = "googletest",
    version = "1.12.1",
)

# Archive override
archive_override(
    module_name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/main.zip"],
    strip_prefix = "rules_cc-main",
)

# Local path override
local_path_override(
    module_name = "my_lib",
    path = "../my_lib",
)

# Git override
git_override(
    module_name = "my_lib",
    remote = "https://github.com/user/repo.git",
    commit = "abc123",
)
```

### Common External Dependencies

```python
# C++ dependencies
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.0.tar.gz"],
    strip_prefix = "abseil-cpp-20230802.0",
)

# Python dependencies
http_archive(
    name = "rules_python",
    sha256 = "...",
    strip_prefix = "rules_python-0.27.0",
    url = "https://github.com/bazelbuild/rules_python/archive/0.27.0.tar.gz",
)

# Load Python rules and set up pip dependencies
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pip",
    requirements_lock = "//:requirements.txt",
)
load("@pip//:requirements.bzl", "install_deps")
install_deps()
```

## Query System

### Basic Query

```bash
# List all targets in package
bazel query //path/to/package:all

# Show dependencies
bazel query 'deps(//src:main)'

# Show direct dependencies only
bazel query 'deps(//src:main, 1)'

# Reverse dependencies (what depends on this)
bazel query 'rdeps(//..., //src:lib)'

# Path between targets
bazel query 'somepath(//src:main, //third_party:lib)'

# All paths between targets
bazel query 'allpaths(//src:main, //third_party:lib)'
```

### Advanced Query

```bash
# Filter by rule kind
bazel query 'kind("cc_library", //...)'
bazel query 'kind(".*test", //...)'

# Filter by attribute
bazel query 'attr(name, ".*test.*", //...)'
bazel query 'attr(visibility, "public", //...)'

# Set operations
bazel query 'deps(//src:main) except deps(//src:lib)'
bazel query 'deps(//src:main) intersect deps(//src:other)'

# Targets that match pattern
bazel query 'filter(".*test", //...)'

# Build files
bazel query 'buildfiles(//src/...)'

# Tests for targets
bazel query 'tests(//src:main)'
```

### Configured Query (cquery)

```bash
# Query with configuration
bazel cquery //src:main

# Show configuration
bazel cquery //src:main --output=build

# Dependencies with configuration
bazel cquery 'deps(//src:main)' --output=graph

# Different configurations
bazel cquery //src:main --cpu=k8
bazel cquery //src:main -c opt
```

### Action Query (aquery)

```bash
# Show actions
bazel aquery //src:main

# Filter by type
bazel aquery 'mnemonic("CppCompile", //src:main)'

# Show inputs/outputs
bazel aquery //src:main --output=text

# Action graph
bazel aquery 'deps(//src:main)' --output=textproto
```

### Query Output Formats

```bash
# Different output formats
bazel query //... --output=label
bazel query //... --output=label_kind
bazel query //... --output=build
bazel query //... --output=xml
bazel query //... --output=proto
bazel query //... --output=graph  # Graphviz format

# Generate dependency graph
bazel query 'deps(//src:main)' --output=graph > graph.dot
dot -Tpng graph.dot -o graph.png
```

## Configuration

### .bazelrc File

```bash
# .bazelrc - Project-wide Bazel configuration

# Build configuration
build --cxxopt='-std=c++17'
build --host_cxxopt='-std=c++17'
build --javacopt='-source 11 -target 11'

# Optimization settings
build:opt -c opt
build:opt --copt=-O3

# Debug settings
build:dbg -c dbg
build:dbg --copt=-g
build:dbg --strip=never

# Test configuration
test --test_output=errors
test --test_summary=detailed

# Remote cache
build --remote_cache=https://cache.example.com
build --remote_upload_local_results=true

# Performance
build --jobs=auto
build --local_cpu_resources=HOST_CPUS*0.8

# Output
build --color=yes
build --show_timestamps
build --verbose_failures

# Platform-specific
build:linux --copt=-fPIC
build:macos --macos_minimum_os=10.15

# User-specific (import from ~/.bazelrc)
try-import %workspace%/.bazelrc.user
```

### Command-line Configuration

```bash
# Use configuration
bazel build --config=opt //src:main
bazel build --config=dbg //src:main

# Override options
bazel build --copt=-O2 //src:main
bazel build --cxxopt='-std=c++20' //src:main

# Set compilation mode
bazel build -c opt //src:main  # optimized
bazel build -c dbg //src:main  # debug
bazel build -c fastbuild //src:main  # fast compilation

# Set CPU architecture
bazel build --cpu=k8 //src:main
bazel build --cpu=arm64 //src:main

# Jobs and resources
bazel build --jobs=4 //src:main
bazel build --local_cpu_resources=8 //src:main
```

### Platform Configuration

```python
# BUILD.bazel
platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

# Use platform
# bazel build --platforms=:linux_x86_64 //src:main
```

## Advanced Features

### Custom Rules (.bzl files)

```python
# rules.bzl - Custom rule definition

def _my_rule_impl(ctx):
    """Implementation of my_rule"""
    # Access inputs
    input_file = ctx.file.src

    # Declare outputs
    output_file = ctx.actions.declare_file(ctx.label.name + ".out")

    # Create action
    ctx.actions.run_shell(
        inputs = [input_file],
        outputs = [output_file],
        command = "cat {} > {}".format(input_file.path, output_file.path),
    )

    # Return providers
    return [DefaultInfo(files = depset([output_file]))]

my_rule = rule(
    implementation = _my_rule_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)
```

```python
# BUILD.bazel - Using custom rule

load(":rules.bzl", "my_rule")

my_rule(
    name = "generate",
    src = "input.txt",
)
```

### Macros

```python
# macros.bzl

def cc_library_with_test(name, srcs, hdrs, deps = [], **kwargs):
    """Macro that creates a library and its test"""

    # Library
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        **kwargs
    )

    # Test
    native.cc_test(
        name = name + "_test",
        srcs = [name + "_test.cc"],
        deps = [
            ":" + name,
            "@googletest//:gtest_main",
        ] + deps,
    )
```

### Aspects

```python
# aspects.bzl

def _print_deps_impl(target, ctx):
    """Aspect that prints all dependencies"""
    deps = []
    if hasattr(ctx.rule.attr, 'deps'):
        for dep in ctx.rule.attr.deps:
            deps.append(str(dep.label))

    print("Target {} has deps: {}".format(target.label, deps))

    return []

print_deps = aspect(
    implementation = _print_deps_impl,
    attr_aspects = ['deps'],
)

# Use aspect
# bazel build //src:main --aspects=:aspects.bzl%print_deps
```

### Providers

```python
# Custom provider
MyInfo = provider(
    fields = {
        "data": "Runtime data files",
        "metadata": "Additional metadata",
    },
)

def _my_rule_impl(ctx):
    # Return custom provider
    return [
        DefaultInfo(files = depset(ctx.files.srcs)),
        MyInfo(
            data = ctx.files.data,
            metadata = ctx.attr.metadata,
        ),
    ]

my_rule = rule(
    implementation = _my_rule_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "data": attr.label_list(allow_files = True),
        "metadata": attr.string_dict(),
    },
)
```

### Transitions

```python
# Configuration transitions

def _arm64_transition_impl(settings, attr):
    return {
        "//command_line_option:cpu": "arm64",
    }

arm64_transition = transition(
    implementation = _arm64_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:cpu"],
)

def _cross_compile_rule_impl(ctx):
    # Build dependencies for ARM64
    return [DefaultInfo(files = depset(ctx.files.deps))]

cross_compile_rule = rule(
    implementation = _cross_compile_rule_impl,
    attrs = {
        "deps": attr.label_list(cfg = arm64_transition),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
```

## Common Patterns

### Monorepo Structure

```
monorepo/
├── WORKSPACE
├── .bazelrc
├── BUILD.bazel
├── services/
│   ├── api/
│   │   ├── BUILD.bazel
│   │   └── main.go
│   └── worker/
│       ├── BUILD.bazel
│       └── main.py
├── libraries/
│   ├── common/
│   │   ├── BUILD.bazel
│   │   └── utils.cc
│   └── proto/
│       ├── BUILD.bazel
│       └── api.proto
└── tools/
    ├── BUILD.bazel
    └── codegen/
```

```python
# Root BUILD.bazel
package(default_visibility = ["//visibility:public"])

# services/api/BUILD.bazel
go_binary(
    name = "api_server",
    srcs = ["main.go"],
    deps = [
        "//libraries/common:utils",
        "//libraries/proto:api_go_proto",
    ],
)

# libraries/common/BUILD.bazel
cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
)
```

### Multi-language Project

```python
# BUILD.bazel - Project with C++, Python, and Go

# C++ library
cc_library(
    name = "core",
    srcs = ["core.cc"],
    hdrs = ["core.h"],
)

# Python bindings
py_library(
    name = "py_bindings",
    srcs = ["bindings.py"],
    data = [":core"],
)

# Go service using C++ library
go_binary(
    name = "service",
    srcs = ["main.go"],
    cgo = True,
    cdeps = [":core"],
)

# Protocol buffers for all languages
proto_library(
    name = "api_proto",
    srcs = ["api.proto"],
)

cc_proto_library(
    name = "api_cc_proto",
    deps = [":api_proto"],
)

py_proto_library(
    name = "api_py_proto",
    deps = [":api_proto"],
)

go_proto_library(
    name = "api_go_proto",
    importpath = "example.com/api",
    protos = [":api_proto"],
)
```

### Code Generation

```python
# Code generation pattern

# Generator tool
cc_binary(
    name = "generator",
    srcs = ["generator.cc"],
)

# Generate code
genrule(
    name = "generated_srcs",
    srcs = ["schema.json"],
    outs = [
        "generated.h",
        "generated.cc",
    ],
    cmd = "$(location :generator) --input $(SRCS) --output $(RULEDIR)",
    tools = [":generator"],
)

# Use generated code
cc_library(
    name = "mylib",
    srcs = [
        "mylib.cc",
        ":generated_srcs",
    ],
    hdrs = [
        "mylib.h",
        ":generated_srcs",
    ],
)
```

### Cross-Compilation

```python
# toolchain/BUILD.bazel

# Define toolchains for different platforms
toolchain(
    name = "linux_x86_64_toolchain",
    toolchain = ":cc_toolchain_linux_x86_64",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

toolchain(
    name = "linux_arm64_toolchain",
    toolchain = ":cc_toolchain_linux_arm64",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:arm64",
    ],
)

# Build for different platforms
# bazel build --platforms=//toolchain:linux_arm64_platform //src:main
```

### Remote Caching Setup

```bash
# .bazelrc - Remote caching configuration

# Google Cloud Storage
build --remote_cache=https://storage.googleapis.com/my-bucket
build --google_default_credentials

# Generic HTTP cache
build --remote_cache=https://cache.example.com
build --remote_header=Authorization=Bearer TOKEN

# Local disk cache
build --disk_cache=/tmp/bazel_cache

# Remote cache options
build --remote_upload_local_results=true
build --remote_accept_cached=true
build --remote_timeout=60
build --remote_max_connections=100
```

### Remote Execution

```bash
# .bazelrc - Remote execution configuration

# Remote execution endpoint
build --remote_executor=grpc://remote.build.example.com:8980

# Remote cache with execution
build --remote_cache=grpc://remote.build.example.com:8980
build --remote_executor=grpc://remote.build.example.com:8980

# Execution properties
build --remote_default_exec_properties=OSFamily=linux
build --remote_default_exec_properties=container-image=docker://my-image

# Local fallback
build --remote_local_fallback
build --remote_local_fallback_strategy=local
```

### Docker Integration

```python
# BUILD.bazel - Docker container builds

load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@io_bazel_rules_docker//cc:image.bzl", "cc_image")

# Build C++ binary in container
cc_image(
    name = "app_image",
    binary = ":main",
    base = "@cc_base//image",
)

# Custom container image
container_image(
    name = "custom_image",
    base = "@ubuntu//image",
    files = [
        ":main",
        "config.yaml",
    ],
    entrypoint = ["/main"],
)
```

## Performance Optimization

### Build Performance

```bash
# Parallel builds
bazel build --jobs=auto //...
bazel build -j 8 //...

# Limit resource usage
bazel build --local_cpu_resources=HOST_CPUS*0.8 //...
bazel build --local_ram_resources=HOST_RAM*0.8 //...

# Incremental builds
bazel build --nobuild  # Analyze only
bazel build --keep_going  # Continue on errors

# Profile build
bazel build --profile=profile.json //...
bazel analyze-profile profile.json

# Show slow targets
bazel build --experimental_profile_include_target_label //...
```

### Action Caching

```bash
# Enable action cache
bazel build --action_cache=/tmp/action_cache //...

# Repository cache
bazel build --repository_cache=/tmp/repo_cache //...

# Disk cache
bazel build --disk_cache=/tmp/bazel_cache //...

# Cache statistics
bazel info | grep cache
```

### Remote Caching Best Practices

```bash
# .bazelrc

# Enable remote cache for all operations
build --remote_cache=https://cache.example.com
test --remote_cache=https://cache.example.com

# Upload local results
build --remote_upload_local_results=true

# Download all outputs
build --remote_download_all

# Or download only outputs needed locally
build --remote_download_minimal

# Compress cache data
build --remote_grpc_compression=gzip

# Timeout settings
build --remote_timeout=60s
```

### Optimize BUILD Files

```python
# Use filegroups for common file sets
filegroup(
    name = "common_hdrs",
    srcs = glob(["include/**/*.h"]),
)

# Avoid unnecessary globs
# Bad: srcs = glob(["**/*.cc"])
# Good: srcs = glob(["*.cc"])

# Explicit dependencies (better than transitive)
deps = [
    "//lib:specific_lib",  # Good
    # "//lib:all",  # Avoid
]

# Use select for platform-specific code
srcs = select({
    "@platforms//os:linux": ["linux_impl.cc"],
    "@platforms//os:macos": ["macos_impl.cc"],
    "//conditions:default": ["generic_impl.cc"],
})
```

## Best Practices

### BUILD File Organization

```python
# Good BUILD file structure

# 1. Load statements at top
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load(":custom_rules.bzl", "my_rule")

# 2. Package-level configuration
package(default_visibility = ["//visibility:private"])

# 3. Filegroups and exports
filegroup(
    name = "headers",
    srcs = glob(["*.h"]),
)

exports_files(["config.yaml"])

# 4. Libraries first
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
    hdrs = [":headers"],
)

# 5. Binaries next
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [":lib"],
)

# 6. Tests last
cc_test(
    name = "lib_test",
    srcs = ["lib_test.cc"],
    deps = [":lib"],
)
```

### Dependency Management

```python
# Explicit and minimal dependencies
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = ["mylib.h"],
    # Only list direct dependencies
    deps = [
        ":direct_dep",
        "//other:lib",
    ],
    # Avoid transitive dependencies
)

# Use visibility to control access
cc_library(
    name = "internal_lib",
    srcs = ["internal.cc"],
    visibility = [
        "//src:__subpackages__",  # Only src subtree
    ],
)

# Group related targets
package_group(
    name = "internal",
    packages = [
        "//src/core/...",
        "//src/internal/...",
    ],
)
```

### Hermetic Builds

```python
# Ensure builds are hermetic

# 1. Declare all inputs explicitly
cc_binary(
    name = "app",
    srcs = ["main.cc"],
    data = [
        "config.yaml",  # Runtime files
        "//data:dataset",
    ],
)

# 2. Use toolchains instead of system tools
# Bad:
# cmd = "/usr/bin/python script.py"

# Good:
genrule(
    name = "generate",
    tools = ["@python_interpreter//python3"],
    cmd = "$(location @python_interpreter//python3) script.py",
)

# 3. Pin external dependencies
http_archive(
    name = "dependency",
    urls = ["https://example.com/lib-1.2.3.tar.gz"],
    sha256 = "abc123...",  # Always include hash
)

# 4. Avoid host-specific paths
# Bad: data = ["/tmp/file.txt"]
# Good: data = ["//testdata:file.txt"]
```

### Reproducible Builds

```bash
# .bazelrc

# Stamp builds with version info
build --stamp
build --workspace_status_command=./tools/workspace_status.sh

# Use hermetic sandbox
build --spawn_strategy=sandboxed

# Enforce strict action environment
build --incompatible_strict_action_env

# Fixed timestamp for reproducibility
build --define=TIMESTAMP=0
```

### Testing Best Practices

```python
# Organize tests by size
cc_test(
    name = "unit_test",
    srcs = ["unit_test.cc"],
    size = "small",  # < 1 minute, < 20MB RAM
    deps = [":lib"],
)

cc_test(
    name = "integration_test",
    srcs = ["integration_test.cc"],
    size = "medium",  # < 5 minutes, < 100MB RAM
    data = ["//testdata:files"],
)

# Tag tests appropriately
cc_test(
    name = "slow_test",
    srcs = ["slow_test.cc"],
    tags = [
        "slow",
        "manual",  # Don't run with //... pattern
        "requires_gpu",
    ],
)

# Use test suites
test_suite(
    name = "all_tests",
    tests = [
        ":unit_test",
        ":integration_test",
    ],
)

# Exclude slow tests from regular runs
# bazel test //... --test_tag_filters=-slow
```

## Troubleshooting

### Common Errors

```bash
# "Target not found" error
# Check if target exists
bazel query //path/to:target

# "Undeclared inclusion" error
# Add missing dependency or include directory
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
    hdrs = ["lib.h"],
    includes = ["include/"],  # Add this
    deps = [":missing_dep"],  # Or this
)

# "Action failed" error
# Show verbose output
bazel build --verbose_failures //...
bazel build -s //...  # Show all commands

# "External dependency failed"
# Clear external cache
bazel clean --expunge_async
bazel sync

# "Out of memory" error
# Reduce parallelism
bazel build --jobs=2 --local_ram_resources=8192 //...
```

### Debugging Builds

```bash
# Show build commands
bazel build -s //src:main

# Explain why target is rebuilt
bazel build --explain=explain.log //src:main
cat explain.log

# Detailed explanation
bazel build --verbose_explanations --explain=explain.log //src:main

# Show dependency chain
bazel query 'somepath(//src:main, //third_party:lib)'

# Find circular dependencies
bazel query 'allpaths(//src:main, //src:main)'

# Check for test failures
bazel test --test_output=all //tests:all
bazel test --test_output=streamed //tests:failing_test
```

### Cache Issues

```bash
# Clear all caches
bazel clean --expunge

# Clear specific cache
rm -rf ~/.cache/bazel

# Disable caching for debugging
bazel build --nocache_test_results //tests:all

# Check cache statistics
bazel info | grep cache

# Remote cache diagnostics
bazel build --remote_cache_print_upload_stats=true //...
```

### Performance Debugging

```bash
# Profile build
bazel build --profile=profile.json //...

# Analyze profile
bazel analyze-profile profile.json
bazel analyze-profile --html profile.json > profile.html

# Show critical path
bazel analyze-profile --dump=critpath profile.json

# Memory profiling
bazel build --heap_dump_on_oom //...
bazel dump --rules
bazel dump --skylark_memory
```

### Dependency Problems

```bash
# Show all dependencies
bazel query 'deps(//src:main)' --output=graph

# Find unused dependencies
bazel query 'kind("cc_library", deps(//src:main))' --output=label

# Check for diamond dependencies
bazel query 'allpaths(//src:main, //third_party:lib)'

# Verify dependency visibility
bazel build --check_visibility //src:main

# Fix dependency issues
# Use buildozer for bulk edits
buildozer 'add deps :new_dep' //src:*
```

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `bazel build //path:target` | Build a target |
| `bazel test //...` | Run all tests |
| `bazel run //path:binary` | Run a binary |
| `bazel query 'deps(//path:target)'` | Show dependencies |
| `bazel clean` | Clean build outputs |
| `bazel clean --expunge` | Deep clean |
| `bazel info` | Workspace information |

### Common Flags

| Flag | Description |
|------|-------------|
| `-c opt` | Optimized build |
| `-c dbg` | Debug build |
| `--jobs=N` | Parallel jobs |
| `-s` | Show commands |
| `--verbose_failures` | Show error details |
| `--test_output=all` | Show all test output |

### BUILD File Patterns

| Pattern | Description |
|---------|-------------|
| `glob(["*.cc"])` | Match files |
| `select({...})` | Platform-specific config |
| `//path:target` | Absolute label |
| `:target` | Local label |
| `@repo//path:target` | External repository |

Bazel provides a powerful, scalable build system that ensures fast, correct, and reproducible builds across multiple languages and platforms.
