# GCC

GCC (GNU Compiler Collection) is a comprehensive compiler system supporting various programming languages including C, C++, Objective-C, Fortran, Ada, and Go. It's the standard compiler for most Unix-like operating systems and provides powerful optimization, debugging, and cross-compilation capabilities.

## Overview

GCC transforms source code into executable programs through multiple stages: preprocessing, compilation, assembly, and linking. It offers extensive control over the compilation process through command-line options.

**Key Concepts:**
- **Preprocessing**: Expands macros, includes headers, handles directives
- **Compilation**: Converts source to assembly code
- **Assembly**: Transforms assembly to machine code (object files)
- **Linking**: Combines object files and libraries into executables
- **Optimization**: Code transformations for speed or size
- **Cross-compilation**: Build for different target architectures

## Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install gcc g++ build-essential

# RedHat/CentOS/Fedora
sudo yum groupinstall "Development Tools"
sudo dnf install gcc gcc-c++

# macOS (via Xcode Command Line Tools)
xcode-select --install

# Verify installation
gcc --version
g++ --version

# Check available targets
gcc -v
```

## Basic Compilation

### Simple C Program

```bash
# Compile single source file
gcc hello.c
gcc hello.c -o hello

# Run the program
./hello
./a.out  # Default output name

# Compile and run
gcc hello.c -o hello && ./hello
```

### Simple C++ Program

```bash
# Compile C++ source
g++ hello.cpp -o hello
g++ hello.cpp -o hello -std=c++17

# Alternative: use gcc with explicit C++
gcc hello.cpp -o hello -lstdc++ -std=c++17
```

### Common Workflow

```bash
# Development build (with debugging)
gcc -g -Wall -Wextra program.c -o program

# Production build (optimized)
gcc -O2 -Wall program.c -o program

# Verbose compilation
gcc -v program.c -o program

# Save compiler output
gcc program.c -o program 2> compile.log
```

## Compilation Stages

### Individual Stages

```bash
# 1. Preprocessing only (-E)
gcc -E source.c -o source.i
gcc -E source.c | less  # View preprocessed output

# 2. Compile to assembly (-S)
gcc -S source.c -o source.s
gcc -S -O2 source.c -o source.s  # Optimized assembly

# 3. Assemble to object file (-c)
gcc -c source.c -o source.o

# 4. Link object files
gcc main.o utils.o -o program

# Complete manual process
gcc -E main.c -o main.i      # Preprocess
gcc -S main.i -o main.s      # Compile
gcc -c main.s -o main.o      # Assemble
gcc main.o -o main           # Link
```

### Viewing Intermediate Files

```bash
# Keep intermediate files
gcc -save-temps program.c -o program
# Creates: program.i, program.s, program.o

# Specify temp directory
gcc -save-temps=obj program.c -o program

# View assembly with source interleaved
gcc -Wa,-adhln -g program.c > program.lst
```

## Compiler Options

### Output Control

```bash
# Specify output filename
gcc source.c -o myprogram

# Compile without linking
gcc -c file1.c file2.c file3.c

# Compile to assembly
gcc -S program.c

# Preprocess only
gcc -E program.c

# Generate dependency information
gcc -M source.c      # All dependencies
gcc -MM source.c     # User dependencies only
gcc -MMD -c source.c # Create .d file during compilation
```

### Warning Flags

```bash
# Essential warnings
gcc -Wall source.c               # Enable most warnings
gcc -Wextra source.c             # Additional warnings
gcc -Werror source.c             # Treat warnings as errors
gcc -Wall -Wextra -Werror source.c

# Specific warnings
gcc -Wpedantic source.c          # ISO C/C++ compliance
gcc -Wconversion source.c        # Type conversion warnings
gcc -Wshadow source.c            # Variable shadowing
gcc -Wcast-align source.c        # Pointer alignment issues
gcc -Wunused source.c            # Unused variables/functions
gcc -Wformat=2 source.c          # Format string checking
gcc -Wstrict-overflow=5 source.c # Overflow optimization warnings

# Disable specific warnings
gcc -Wall -Wno-unused-parameter source.c
gcc -Wall -Wno-format-truncation source.c

# Comprehensive warning set
gcc -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
    -Wcast-align -Wformat=2 source.c
```

### Optimization Levels

```bash
# No optimization (default, best for debugging)
gcc -O0 program.c

# Basic optimization (balanced)
gcc -O1 program.c

# Recommended optimization (production)
gcc -O2 program.c

# Aggressive optimization
gcc -O3 program.c

# Optimize for size
gcc -Os program.c

# Maximum optimization (may break standards compliance)
gcc -Ofast program.c

# Optimization for debugging
gcc -Og -g program.c

# Compare optimization levels
gcc -O2 program.c -o program_o2
gcc -O3 program.c -o program_o3
ls -lh program_*
time ./program_o2
time ./program_o3
```

### Debugging Options

```bash
# Basic debug symbols
gcc -g program.c

# GDB-specific debug info
gcc -ggdb program.c
gcc -ggdb3 program.c  # Maximum debug info

# Debug with optimization (careful!)
gcc -Og -g program.c

# Keep frame pointer for debugging
gcc -g -fno-omit-frame-pointer program.c

# Debug macros
gcc -g3 program.c  # Include macro definitions

# Split debug info
gcc -g -gsplit-dwarf program.c  # Creates .dwo files

# Compressed debug sections
gcc -g -gz program.c
```

### Standard Selection

```bash
# C Standards
gcc -std=c89 program.c    # ANSI C (C90)
gcc -std=c99 program.c    # C99
gcc -std=c11 program.c    # C11
gcc -std=c17 program.c    # C17
gcc -std=c2x program.c    # C23 (experimental)

# GNU extensions
gcc -std=gnu99 program.c  # C99 + GNU extensions
gcc -std=gnu11 program.c  # C11 + GNU extensions

# C++ Standards
g++ -std=c++98 program.cpp  # C++98
g++ -std=c++11 program.cpp  # C++11
g++ -std=c++14 program.cpp  # C++14
g++ -std=c++17 program.cpp  # C++17
g++ -std=c++20 program.cpp  # C++20
g++ -std=c++23 program.cpp  # C++23 (experimental)

# GNU C++ extensions
g++ -std=gnu++17 program.cpp
```

## Include Paths and Libraries

### Include Directories

```bash
# Add include directory
gcc -I/usr/local/include program.c
gcc -I./include program.c
gcc -I../common/include program.c

# Multiple include directories
gcc -I./include -I./external/include program.c

# System include directory (no warnings)
gcc -isystem /usr/local/include program.c

# View default include paths
gcc -E -v - < /dev/null 2>&1 | grep "include"
echo | gcc -E -Wp,-v - 2>&1 | grep "^ "
```

### Library Linking

```bash
# Link with library (-l)
gcc program.c -lm          # Link with math library (libm.so)
gcc program.c -lpthread    # Link with pthread library
gcc program.c -lm -lpthread -lrt

# Library search path (-L)
gcc program.c -L/usr/local/lib -lmylib
gcc program.c -L./lib -lmylib

# Link with specific library file
gcc program.c /usr/lib/libfoo.a
gcc program.c /usr/lib/libfoo.so

# Order matters for static libraries
gcc main.o -lB -lA  # If libB depends on libA

# Show linker commands
gcc -Wl,--verbose program.c

# Pass options to linker
gcc program.c -Wl,-rpath,/usr/local/lib
gcc program.c -Wl,--as-needed -lm
```

### Static vs Dynamic Linking

```bash
# Dynamic linking (default)
gcc program.c -lm

# Static linking of specific library
gcc program.c -static -lm

# Static linking of all libraries
gcc program.c -static

# Prefer static libraries
gcc program.c -Wl,-Bstatic -lmylib -Wl,-Bdynamic

# Check library dependencies
ldd ./program

# Show which libraries will be linked
gcc -Wl,--trace program.c -lm 2>&1 | grep succeeded
```

## Preprocessor Directives

### Macro Definitions

```bash
# Define macro from command line
gcc -DDEBUG program.c
gcc -DDEBUG=1 program.c
gcc -DVERSION=\"1.0.0\" program.c

# Multiple definitions
gcc -DDEBUG -DVERBOSE -DVERSION=2 program.c

# Undefine macro
gcc -UDEBUG program.c

# View predefined macros
gcc -dM -E - < /dev/null
gcc -dM -E program.c | grep __VERSION__

# Common predefined macros
gcc -E -dM - < /dev/null | grep -E '__(linux|GNUC|x86_64)__'
```

### Conditional Compilation Example

```c
// program.c
#ifdef DEBUG
    #define LOG(msg) printf("DEBUG: %s\n", msg)
#else
    #define LOG(msg)
#endif

#if VERSION >= 2
    // New API
#else
    // Old API
#endif
```

```bash
# Compile with DEBUG enabled
gcc -DDEBUG program.c

# Compile production version
gcc -DNDEBUG -O2 program.c
```

## Architecture and Platform Options

### Target Architecture

```bash
# 32-bit compilation on 64-bit system
gcc -m32 program.c

# 64-bit compilation
gcc -m64 program.c

# Architecture-specific optimization
gcc -march=native program.c      # Optimize for current CPU
gcc -march=x86-64 program.c      # Generic x86-64
gcc -march=haswell program.c     # Intel Haswell
gcc -march=znver2 program.c      # AMD Zen 2

# Tune for specific CPU (without requiring its features)
gcc -mtune=native program.c
gcc -mtune=generic program.c

# CPU feature flags
gcc -mavx2 program.c             # Enable AVX2 instructions
gcc -msse4.2 program.c           # Enable SSE4.2
gcc -mfma program.c              # Enable FMA instructions

# ARM architectures
gcc -march=armv7-a program.c
gcc -march=armv8-a program.c
gcc -mcpu=cortex-a72 program.c
```

### Position Independent Code

```bash
# Position-independent code (required for shared libraries)
gcc -fPIC -c mylib.c
gcc -fpic -c mylib.c  # Smaller, faster, but limited

# Position-independent executable
gcc -fPIE -pie program.c

# No position-independent code (static executables)
gcc -fno-PIC program.c
```

## Building Libraries

### Static Library (.a)

```bash
# Compile source files
gcc -c lib1.c lib2.c lib3.c

# Create static library
ar rcs libmylib.a lib1.o lib2.o lib3.o

# Alternative: create archive
ar -rc libmylib.a lib1.o lib2.o lib3.o
ranlib libmylib.a  # Create index

# Use static library
gcc main.c -L. -lmylib -o program
gcc main.c libmylib.a -o program

# List archive contents
ar -t libmylib.a
nm libmylib.a  # List symbols
```

### Shared Library (.so)

```bash
# Compile with position-independent code
gcc -fPIC -c lib1.c lib2.c lib3.c

# Create shared library
gcc -shared -o libmylib.so lib1.o lib2.o lib3.o

# With soname (version info)
gcc -shared -Wl,-soname,libmylib.so.1 -o libmylib.so.1.0.0 lib1.o lib2.o lib3.o

# Create versioned symlinks
ln -s libmylib.so.1.0.0 libmylib.so.1
ln -s libmylib.so.1 libmylib.so

# Single command compilation
gcc -fPIC -shared -o libmylib.so lib1.c lib2.c lib3.c

# Use shared library
gcc main.c -L. -lmylib -o program

# Set runtime library path
gcc main.c -L. -lmylib -Wl,-rpath,. -o program
gcc main.c -L. -lmylib -Wl,-rpath,'$ORIGIN' -o program

# Check library dependencies
ldd program
readelf -d program | grep RPATH
```

### Library Symbol Visibility

```bash
# Control symbol visibility
gcc -fPIC -fvisibility=hidden -c mylib.c

# Export specific symbols
# In code: __attribute__((visibility("default"))) void public_func();

# Strip symbols from shared library
gcc -shared -o libmylib.so lib.o -s
strip --strip-unneeded libmylib.so

# Version script for symbol control
gcc -shared -o libmylib.so lib.o -Wl,--version-script=exports.map
```

## Advanced Compilation Features

### Link-Time Optimization (LTO)

```bash
# Enable LTO
gcc -flto -O2 file1.c file2.c -o program

# LTO with multiple jobs
gcc -flto=4 -O2 file1.c file2.c -o program

# Separate compilation with LTO
gcc -flto -c -O2 file1.c
gcc -flto -c -O2 file2.c
gcc -flto -O2 file1.o file2.o -o program

# Fat LTO objects (useful for incremental builds)
gcc -flto -ffat-lto-objects -c file1.c
```

### Whole Program Optimization

```bash
# Interprocedural optimization
gcc -fwhole-program main.c utils.c -o program

# Combined with LTO
gcc -flto -fwhole-program -O3 main.c utils.c -o program

# Function inlining
gcc -finline-functions -O2 program.c
gcc -finline-limit=1000 -O2 program.c
```

### Code Coverage (gcov)

```bash
# Compile with coverage instrumentation
gcc -fprofile-arcs -ftest-coverage program.c -o program
gcc --coverage program.c -o program  # Shorthand

# Run program to generate coverage data
./program

# Generate coverage report
gcov program.c

# View coverage (creates program.c.gcov)
cat program.c.gcov

# Coverage with optimization
gcc -O2 --coverage program.c -o program
```

### Profiling (gprof)

```bash
# Compile with profiling
gcc -pg program.c -o program

# Run program (generates gmon.out)
./program

# Analyze profile
gprof program gmon.out > analysis.txt
gprof -b program gmon.out  # Brief output

# Call graph
gprof -q program gmon.out
```

### Sanitizers

```bash
# Address Sanitizer (memory errors)
gcc -fsanitize=address -g program.c -o program
gcc -fsanitize=address -fno-omit-frame-pointer -g program.c

# Undefined Behavior Sanitizer
gcc -fsanitize=undefined -g program.c -o program

# Thread Sanitizer (data races)
gcc -fsanitize=thread -g program.c -o program -lpthread

# Memory Sanitizer (uninitialized memory)
gcc -fsanitize=memory -g program.c -o program

# Leak Sanitizer
gcc -fsanitize=leak -g program.c -o program

# Multiple sanitizers
gcc -fsanitize=address,undefined -g program.c -o program

# Run with sanitizer options
ASAN_OPTIONS=detect_leaks=1 ./program
UBSAN_OPTIONS=print_stacktrace=1 ./program
```

## Cross-Compilation

### Basic Cross-Compilation

```bash
# ARM cross-compiler
arm-linux-gnueabihf-gcc program.c -o program
aarch64-linux-gnu-gcc program.c -o program

# Specify target explicitly
gcc -target arm-linux-gnueabihf program.c

# Check available targets
gcc -print-targets

# Cross-compile with sysroot
arm-linux-gnueabihf-gcc --sysroot=/path/to/sysroot program.c
```

### Multi-arch Setup

```bash
# Install cross-compilation toolchain
sudo apt-get install gcc-arm-linux-gnueabihf
sudo apt-get install gcc-aarch64-linux-gnu

# Cross-compile example
arm-linux-gnueabihf-gcc -march=armv7-a -mfpu=neon program.c

# Verify target architecture
file ./program
arm-linux-gnueabihf-readelf -h program
```

## Security Hardening

### Security Flags

```bash
# Stack protection
gcc -fstack-protector program.c          # Basic
gcc -fstack-protector-strong program.c   # Recommended
gcc -fstack-protector-all program.c      # All functions

# Stack clash protection
gcc -fstack-clash-protection program.c

# Position-independent executable (ASLR)
gcc -fPIE -pie program.c

# Read-only relocations
gcc -Wl,-z,relro program.c
gcc -Wl,-z,relro,-z,now program.c  # Full RELRO

# Format string protection
gcc -Wformat -Wformat-security program.c

# Fortify source (needs optimization)
gcc -O2 -D_FORTIFY_SOURCE=2 program.c

# No executable stack
gcc -z noexecstack program.c

# Control-flow protection (newer GCC)
gcc -fcf-protection program.c

# Comprehensive security flags
gcc -O2 -D_FORTIFY_SOURCE=2 \
    -fstack-protector-strong \
    -fstack-clash-protection \
    -fPIE -pie \
    -Wl,-z,relro,-z,now \
    -Wl,-z,noexecstack \
    program.c -o program
```

### Buffer Overflow Detection

```bash
# Mudflap (deprecated in newer GCC, use sanitizers)
gcc -fmudflap program.c -lmudflap

# Better: use Address Sanitizer
gcc -fsanitize=address -g program.c
```

## Optimization Strategies

### Performance Optimization

```bash
# Profile-guided optimization (PGO)
# Step 1: Compile with instrumentation
gcc -fprofile-generate -O2 program.c -o program

# Step 2: Run with typical workload
./program < typical_input.txt

# Step 3: Compile with profile data
gcc -fprofile-use -O2 program.c -o program

# Aggressive inlining
gcc -O3 -finline-functions -finline-limit=1000 program.c

# Vectorization
gcc -O3 -ftree-vectorize program.c
gcc -O3 -fopt-info-vec program.c  # Show vectorization info
gcc -O3 -fopt-info-vec-missed program.c  # Show what wasn't vectorized

# Loop optimizations
gcc -O3 -funroll-loops program.c
gcc -O3 -funroll-all-loops program.c

# Fast math (may violate IEEE standards)
gcc -O3 -ffast-math program.c

# Architecture-specific with LTO
gcc -O3 -march=native -flto program.c
```

### Size Optimization

```bash
# Optimize for size
gcc -Os program.c

# More aggressive size optimization
gcc -Os -s program.c  # Strip symbols

# Function sections (allows linker to remove unused code)
gcc -ffunction-sections -fdata-sections program.c
gcc -ffunction-sections -fdata-sections \
    -Wl,--gc-sections program.c

# Small executable
gcc -Os -s -ffunction-sections -fdata-sections \
    -Wl,--gc-sections program.c -o program

# Check size
size program
strip program
```

## Common Patterns

### Single File Program

```bash
# Basic compilation
gcc hello.c -o hello

# With warnings and debugging
gcc -Wall -Wextra -g hello.c -o hello

# Production build
gcc -O2 -Wall hello.c -o hello
```

### Multi-file C Project

```bash
# Compile separately
gcc -c main.c
gcc -c utils.c
gcc -c parser.c

# Link together
gcc main.o utils.o parser.o -o program

# One-step compilation
gcc main.c utils.c parser.c -o program

# With headers in separate directory
gcc -I./include -c main.c utils.c parser.c
gcc main.o utils.o parser.o -o program

# Complete build with flags
gcc -Wall -Wextra -O2 -I./include \
    main.c utils.c parser.c -o program
```

### C++ Project

```bash
# Basic C++ compilation
g++ main.cpp utils.cpp -o program

# With C++17 standard
g++ -std=c++17 -Wall -Wextra main.cpp utils.cpp -o program

# Template-heavy projects (faster compilation)
g++ -std=c++17 -O2 -c main.cpp
g++ -std=c++17 -O2 -c utils.cpp
g++ main.o utils.o -o program

# With external libraries
g++ -std=c++17 main.cpp -lboost_system -lpthread -o program
```

### Mixed C and C++ Project

```bash
# Compile C files
gcc -c utils.c file.c

# Compile C++ files
g++ -c main.cpp module.cpp

# Link with C++ compiler (includes C++ standard library)
g++ main.o module.o utils.o file.o -o program

# Alternative: use gcc and explicitly link C++ library
gcc main.o module.o utils.o file.o -lstdc++ -o program
```

### Creating and Using Static Library

```bash
# Create library
gcc -c mylib.c helper.c
ar rcs libmylib.a mylib.o helper.o

# Use library
gcc main.c -L. -lmylib -o program

# Or link directly
gcc main.c libmylib.a -o program
```

### Creating and Using Shared Library

```bash
# Create shared library
gcc -fPIC -c mylib.c helper.c
gcc -shared -o libmylib.so mylib.o helper.o

# Use shared library
gcc main.c -L. -lmylib -o program

# Set runtime path
gcc main.c -L. -lmylib -Wl,-rpath,'$ORIGIN' -o program

# Alternative: set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./program
```

## Best Practices

### Development Builds

```bash
# Recommended flags for development
gcc -Wall -Wextra -Wpedantic -Wshadow \
    -Wformat=2 -Wconversion \
    -g -Og \
    -fsanitize=address,undefined \
    program.c -o program

# C++ development
g++ -std=c++17 -Wall -Wextra -Wpedantic \
    -g -Og \
    -fsanitize=address,undefined \
    program.cpp -o program
```

### Production Builds

```bash
# Recommended flags for production
gcc -Wall -Wextra -O2 \
    -D_FORTIFY_SOURCE=2 \
    -fstack-protector-strong \
    -fPIE -pie \
    -Wl,-z,relro,-z,now \
    program.c -o program

# High-performance production
gcc -Wall -O3 -march=native -flto \
    -fstack-protector-strong \
    program.c -o program
```

### Reproducible Builds

```bash
# Ensure reproducible builds
gcc -O2 -ffile-prefix-map=$(pwd)=. \
    -Wl,--build-id=sha1 \
    program.c -o program

# No timestamps
gcc -O2 -Wl,--no-insert-timestamp program.c

# Consistent debug info
gcc -g -fdebug-prefix-map=$(pwd)=. program.c
```

### Code Quality Checks

```bash
# Maximum warnings
gcc -Wall -Wextra -Wpedantic -Werror \
    -Wshadow -Wformat=2 -Wconversion \
    -Wunused -Wcast-align -Wstrict-prototypes \
    -Wold-style-definition -Wmissing-prototypes \
    program.c -o program

# C++ specific warnings
g++ -Wall -Wextra -Wpedantic -Werror \
    -Wshadow -Wformat=2 -Wconversion \
    -Wnon-virtual-dtor -Woverloaded-virtual \
    -Wold-style-cast \
    program.cpp -o program
```

## Troubleshooting

### Common Compilation Errors

```bash
# Undefined reference (missing library)
gcc program.c -lm  # Add missing library

# Header file not found
gcc -I/path/to/headers program.c

# Wrong include path order
gcc -I./local-include -I/usr/include program.c

# Check system include paths
gcc -E -v - < /dev/null 2>&1 | grep include

# Symbol multiply defined
# Check for duplicate object files in link command
gcc main.o utils.o main.o -o program  # Wrong!
gcc main.o utils.o -o program         # Correct
```

### Linker Errors

```bash
# Show linker search paths
gcc -Wl,--verbose 2>&1 | grep SEARCH_DIR

# Undefined reference to library function
gcc program.c -lm -lpthread  # Ensure correct order

# Static library dependency order matters
gcc main.o -lhigh-level -llow-level  # High-level depends on low-level

# Cannot find library
gcc program.c -L/path/to/lib -lmylib

# Check what symbols are needed
nm -u program.o  # Undefined symbols
nm -D libmylib.so  # Dynamic symbols

# Show all symbols
nm program.o
objdump -t program.o
```

### Runtime Errors

```bash
# Shared library not found
ldd program  # Check dependencies
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# Wrong library version loaded
ldd program  # Check which version is loaded
ldconfig -p | grep libname  # Check system libraries

# ABI compatibility issues
nm -D libold.so > old_symbols.txt
nm -D libnew.so > new_symbols.txt
diff old_symbols.txt new_symbols.txt

# Check binary information
file program
readelf -h program
objdump -f program
```

### Debugging Compilation Issues

```bash
# Verbose output
gcc -v program.c

# Show all commands executed
gcc -v program.c 2>&1 | grep cc1

# Preprocessor output
gcc -E program.c | less
gcc -E -dM program.c  # Show macros

# Assembly output
gcc -S -fverbose-asm program.c
gcc -S -masm=intel program.c  # Intel syntax

# Keep intermediate files
gcc -save-temps program.c

# Show optimization passes
gcc -O2 -fopt-info program.c
gcc -O2 -fopt-info-all program.c
```

### Performance Issues

```bash
# Check optimization level
gcc -Q --help=optimizers

# Profile the code
gcc -pg program.c -o program
./program
gprof program gmon.out

# Check if vectorized
gcc -O3 -fopt-info-vec program.c

# View optimization details
gcc -O3 -fopt-info-all program.c 2>&1 | grep -i vectorized

# Compare optimization levels
gcc -O2 program.c -o prog_o2
gcc -O3 program.c -o prog_o3
ls -lh prog_*
time ./prog_o2
time ./prog_o3
```

## Complete Examples

### Simple C Program

```c
// hello.c
#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

```bash
gcc hello.c -o hello
./hello
```

### Multi-file C Project

```c
// main.c
#include <stdio.h>
#include "calc.h"

int main(void) {
    int result = add(5, 3);
    printf("Result: %d\n", result);
    return 0;
}
```

```c
// calc.h
#ifndef CALC_H
#define CALC_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

```c
// calc.c
#include "calc.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

```bash
# Method 1: All at once
gcc main.c calc.c -o program

# Method 2: Separate compilation
gcc -c main.c
gcc -c calc.c
gcc main.o calc.o -o program

# Method 3: With warnings and optimization
gcc -Wall -Wextra -O2 -c main.c
gcc -Wall -Wextra -O2 -c calc.c
gcc main.o calc.o -o program
```

### Static Library Example

```bash
# Create library source files
cat > mathlib.c << 'EOF'
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
EOF

cat > mathlib.h << 'EOF'
#ifndef MATHLIB_H
#define MATHLIB_H
int add(int a, int b);
int subtract(int a, int b);
#endif
EOF

# Create main program
cat > main.c << 'EOF'
#include <stdio.h>
#include "mathlib.h"

int main(void) {
    printf("5 + 3 = %d\n", add(5, 3));
    printf("5 - 3 = %d\n", subtract(5, 3));
    return 0;
}
EOF

# Build static library
gcc -c mathlib.c
ar rcs libmath.a mathlib.o

# Use library
gcc main.c -L. -lmath -o program
./program
```

### Shared Library Example

```bash
# Same source files as static library example above

# Build shared library
gcc -fPIC -c mathlib.c
gcc -shared -o libmath.so mathlib.o

# Use library
gcc main.c -L. -lmath -Wl,-rpath,'$ORIGIN' -o program
./program

# Alternative: use LD_LIBRARY_PATH
gcc main.c -L. -lmath -o program
LD_LIBRARY_PATH=. ./program
```

### Makefile Integration

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2 -I./include
LDFLAGS = -L./lib
LDLIBS = -lm -lpthread

SRCS = main.c utils.c parser.c
OBJS = $(SRCS:.c=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
```

## Quick Reference

### Essential Flags

| Flag | Description |
|------|-------------|
| `-o file` | Output filename |
| `-c` | Compile without linking |
| `-S` | Generate assembly |
| `-E` | Preprocess only |
| `-g` | Debug symbols |
| `-Wall` | Enable warnings |
| `-Werror` | Warnings as errors |
| `-O0` | No optimization |
| `-O2` | Standard optimization |
| `-O3` | Aggressive optimization |
| `-I dir` | Include directory |
| `-L dir` | Library directory |
| `-l name` | Link library |
| `-std=c11` | C standard version |
| `-march=native` | Optimize for current CPU |

### Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `-O0` | No optimization | Debugging |
| `-O1` | Basic optimization | Development |
| `-O2` | Recommended | Production |
| `-O3` | Aggressive | Performance-critical |
| `-Os` | Size optimization | Embedded systems |
| `-Og` | Debug-friendly | Development with optimization |
| `-Ofast` | Maximum speed | May break standards compliance |

### Standard Versions

| Flag | Standard | Year |
|------|----------|------|
| `-std=c89` | ANSI C / C90 | 1989/1990 |
| `-std=c99` | C99 | 1999 |
| `-std=c11` | C11 | 2011 |
| `-std=c17` | C17 | 2017 |
| `-std=c++11` | C++11 | 2011 |
| `-std=c++14` | C++14 | 2014 |
| `-std=c++17` | C++17 | 2017 |
| `-std=c++20` | C++20 | 2020 |

## Useful Tips

1. **Use `-Wall -Wextra`** for all development builds
2. **Enable optimization with debugging**: `-Og -g` for development, `-O2` for production
3. **Use sanitizers** during development: `-fsanitize=address,undefined`
4. **Generate dependencies** automatically: `-MMD -MP`
5. **Profile before optimizing**: Use `-pg` with gprof or `-fprofile-generate`
6. **Use LTO** for maximum performance: `-flto -O3`
7. **Security-harden** production builds with stack protection and PIE
8. **Keep intermediate files** for debugging: `-save-temps`
9. **Check what optimization does**: `-fopt-info` family of flags
10. **Use makefiles** for complex projects to manage dependencies

GCC is a powerful, flexible compiler that provides comprehensive control over the compilation process, enabling developers to optimize for performance, size, debugging, or security depending on their needs.
