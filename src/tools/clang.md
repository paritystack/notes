# Clang

Clang is a C/C++/Objective-C compiler frontend for the LLVM compiler infrastructure. It provides fast compilation, excellent diagnostics, and a modular architecture that powers various development tools including formatters, linters, static analyzers, and language servers.

## Overview

Clang is part of the LLVM project and offers a comprehensive suite of tools for C-family language development. It's designed to be compatible with GCC while providing better error messages, faster compilation, and lower memory usage.

**Key Features:**
- Fast compilation with low memory footprint
- Expressive diagnostics with fix-it hints
- GCC compatibility for most use cases
- Modular architecture enabling powerful tools
- Built-in static analyzer
- Sanitizers for runtime error detection
- Cross-compilation support
- Language Server Protocol implementation (clangd)

**Common Use Cases:**
- C/C++/Objective-C compilation
- Code formatting and style enforcement
- Static analysis and linting
- IDE language server backend
- Cross-platform development
- Embedded systems development
- Security-focused compilation with sanitizers

## Installation

### Ubuntu/Debian

```bash
# Latest stable version
sudo apt update
sudo apt install clang

# Specific version
sudo apt install clang-15

# Full LLVM toolchain
sudo apt install clang llvm lld

# Additional tools
sudo apt install clang-format clang-tidy clangd

# Install all clang tools
sudo apt install clang-tools

# Verify installation
clang --version
clang-format --version
clang-tidy --version
```

### macOS

```bash
# Xcode Command Line Tools (includes clang)
xcode-select --install

# Via Homebrew (LLVM version)
brew install llvm

# Add to PATH (Homebrew LLVM)
export PATH="/usr/local/opt/llvm/bin:$PATH"

# Verify installation
clang --version
```

### Windows

```bash
# Via Visual Studio (includes clang-cl)
# Install "C++ Clang tools for Windows" component

# Via MSYS2
pacman -S mingw-w64-x86_64-clang

# Via Chocolatey
choco install llvm

# Verify installation
clang --version
```

### From Source

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
  ../llvm

# Build (use -j for parallel builds)
make -j$(nproc)

# Install
sudo make install

# Verify
clang --version
```

## Clang Compiler

### Basic Compilation

```bash
# Compile C program
clang hello.c -o hello

# Compile C++ program
clang++ hello.cpp -o hello

# Compile with warnings
clang -Wall -Wextra main.c -o program

# Compile multiple files
clang main.c utils.c helper.c -o program

# Compile to object file
clang -c module.c -o module.o

# Link object files
clang main.o utils.o -o program

# Preprocess only
clang -E source.c -o source.i

# Compile to assembly
clang -S source.c -o source.s

# Show compilation stages
clang -### main.c
```

### Optimization Levels

```bash
# No optimization (default, fastest compile)
clang -O0 main.c -o program

# Basic optimization
clang -O1 main.c -o program

# Moderate optimization (recommended)
clang -O2 main.c -o program

# Aggressive optimization
clang -O3 main.c -o program

# Optimize for size
clang -Os main.c -o program

# Aggressive size optimization
clang -Oz main.c -o program

# Fast math optimizations (less precise)
clang -O3 -ffast-math main.c -o program

# Debug optimization (optimize but keep debug info)
clang -Og -g main.c -o program
```

### C/C++ Standards

```bash
# C standards
clang -std=c89 main.c      # ANSI C (C89/C90)
clang -std=c99 main.c      # C99
clang -std=c11 main.c      # C11
clang -std=c17 main.c      # C17
clang -std=c2x main.c      # C23 (draft)

# C++ standards
clang++ -std=c++98 main.cpp   # C++98
clang++ -std=c++11 main.cpp   # C++11
clang++ -std=c++14 main.cpp   # C++14
clang++ -std=c++17 main.cpp   # C++17
clang++ -std=c++20 main.cpp   # C++20
clang++ -std=c++2b main.cpp   # C++23 (draft)

# GNU extensions (default)
clang -std=gnu11 main.c
clang++ -std=gnu++17 main.cpp
```

### Warning Flags

```bash
# Essential warnings
clang -Wall main.c              # Common warnings
clang -Wextra main.c            # Extra warnings
clang -Wpedantic main.c         # Strict standard compliance

# All warnings
clang -Wall -Wextra -Wpedantic main.c

# Treat warnings as errors
clang -Werror main.c

# Specific warnings
clang -Wunused main.c           # Unused variables
clang -Wshadow main.c           # Variable shadowing
clang -Wconversion main.c       # Type conversions
clang -Wcast-align main.c       # Alignment issues
clang -Wformat=2 main.c         # Format string issues

# Disable specific warnings
clang -Wno-unused-parameter main.c

# Everything (very verbose)
clang -Weverything main.c

# Recommended flags
clang -Wall -Wextra -Wpedantic -Wshadow -Wconversion main.c
```

### Include Paths and Linking

```bash
# Add include directory
clang -I/usr/local/include main.c

# Multiple include paths
clang -I./include -I./external/include main.c

# System include path
clang -isystem /usr/local/include main.c

# Link library
clang main.c -lm -o program              # Link math library
clang main.c -lpthread -o program        # Link pthread

# Library search path
clang -L/usr/local/lib main.c -lmylib

# Static linking
clang -static main.c -o program

# Shared library creation
clang -shared -fPIC lib.c -o libmylib.so

# Runtime library path
clang -Wl,-rpath,/usr/local/lib main.c -lmylib
```

### Debug and Symbols

```bash
# Debug symbols
clang -g main.c -o program

# Debug with optimization
clang -g -O2 main.c -o program

# Debug symbols level
clang -g0 main.c    # No debug info
clang -g1 main.c    # Minimal debug info
clang -g2 main.c    # Default debug info
clang -g3 main.c    # Maximum debug info

# DWARF version
clang -gdwarf-4 main.c

# Split debug info
clang -gsplit-dwarf main.c

# Strip symbols
strip program
```

### Preprocessor Options

```bash
# Define macro
clang -DDEBUG main.c
clang -DVERSION=1.0 main.c

# Undefine macro
clang -UDEBUG main.c

# Show defined macros
clang -dM -E - < /dev/null

# Include file
clang -include config.h main.c

# Precompiled headers
clang -x c-header header.h -o header.pch
clang -include-pch header.pch main.c

# Show include paths
clang -E -v main.c
```

### Position Independent Code

```bash
# Position independent code (for shared libraries)
clang -fPIC -c lib.c -o lib.o

# Position independent executable
clang -fPIE -pie main.c -o program

# Create shared library
clang -shared -fPIC lib.c -o libmylib.so

# Link against shared library
clang main.c -L. -lmylib -o program
```

## clang-format

clang-format automatically formats C/C++/Objective-C code according to a specified style guide.

### Basic Usage

```bash
# Format file (output to stdout)
clang-format file.cpp

# Format and overwrite file
clang-format -i file.cpp

# Format multiple files
clang-format -i src/*.cpp include/*.h

# Format with specific style
clang-format -style=llvm file.cpp
clang-format -style=google file.cpp
clang-format -style=chromium file.cpp
clang-format -style=mozilla file.cpp
clang-format -style=webkit file.cpp

# Format specific lines
clang-format -lines=10:20 file.cpp

# Dry run (show what would change)
clang-format -output-replacements-xml file.cpp

# Check if formatting needed (exit code)
clang-format --dry-run -Werror file.cpp
```

### Configuration File (.clang-format)

```yaml
# .clang-format in project root
---
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 100
BreakBeforeBraces: Allman
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: Never
AllowShortLoopsOnASingleLine: false
IndentCaseLabels: true
SpaceBeforeParens: ControlStatements
PointerAlignment: Left
```

### Common Styles

```bash
# Generate .clang-format file
clang-format -style=llvm -dump-config > .clang-format

# LLVM style
clang-format -style=llvm -i file.cpp

# Google C++ Style
clang-format -style=google -i file.cpp

# Chromium style
clang-format -style=chromium -i file.cpp

# Mozilla style
clang-format -style=mozilla -i file.cpp

# WebKit style
clang-format -style=webkit -i file.cpp

# Custom inline style
clang-format -style="{BasedOnStyle: llvm, IndentWidth: 8}" file.cpp
```

### Editor Integration

```bash
# Vim integration (~/.vimrc)
# map <C-K> :py3f /usr/share/clang/clang-format.py<cr>
# imap <C-K> <c-o>:py3f /usr/share/clang/clang-format.py<cr>

# VS Code
# Install "C/C++" extension by Microsoft
# Settings: "C_Cpp.clang_format_style": "file"

# Emacs
# (load "/usr/share/clang/clang-format.el")
# (global-set-key [C-M-tab] 'clang-format-region)

# Sublime Text
# Install "Clang Format" package
```

### Git Integration

```bash
# Format staged files before commit
git diff -U0 --no-color --cached | clang-format-diff -i -p1

# Pre-commit hook
# .git/hooks/pre-commit
#!/bin/bash
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(c|cpp|h|hpp)$'); do
    clang-format -i "$file"
    git add "$file"
done
```

### CI/CD Integration

```bash
# Check formatting in CI
find src include -name '*.cpp' -o -name '*.h' | \
  xargs clang-format --dry-run -Werror

# Format all files
find . -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec clang-format -i {} \;

# Check for formatting changes
clang-format -i src/**/*.{cpp,h}
git diff --exit-code
```

## clang-tidy

clang-tidy is a clang-based C++ linter tool providing static analysis, style checking, and automated fixes.

### Basic Usage

```bash
# Run clang-tidy on file
clang-tidy file.cpp

# With compilation database
clang-tidy file.cpp -p build/

# Specify checks
clang-tidy -checks='*' file.cpp
clang-tidy -checks='readability-*' file.cpp
clang-tidy -checks='modernize-*,readability-*' file.cpp

# Exclude checks
clang-tidy -checks='*,-modernize-use-trailing-return-type' file.cpp

# Apply fixes automatically
clang-tidy -fix file.cpp

# Apply fixes for errors only
clang-tidy -fix-errors file.cpp

# Export fixes to file
clang-tidy -export-fixes=fixes.yaml file.cpp

# List available checks
clang-tidy -list-checks

# Explain check
clang-tidy -checks='readability-*' -explain-config
```

### Configuration File (.clang-tidy)

```yaml
# .clang-tidy in project root
---
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  portability-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers

WarningsAsErrors: ''
HeaderFilterRegex: '.*'
FormatStyle: file
CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.ConstantCase
    value: UPPER_CASE
```

### Check Categories

```bash
# Bugprone checks
clang-tidy -checks='bugprone-*' file.cpp

# Performance checks
clang-tidy -checks='performance-*' file.cpp

# Modernization (C++11/14/17/20)
clang-tidy -checks='modernize-*' file.cpp

# Readability improvements
clang-tidy -checks='readability-*' file.cpp

# C++ Core Guidelines
clang-tidy -checks='cppcoreguidelines-*' file.cpp

# Clang static analyzer
clang-tidy -checks='clang-analyzer-*' file.cpp

# CERT secure coding
clang-tidy -checks='cert-*' file.cpp

# Google style guide
clang-tidy -checks='google-*' file.cpp

# LLVM coding standards
clang-tidy -checks='llvm-*' file.cpp

# Multiple categories
clang-tidy -checks='bugprone-*,performance-*,modernize-*' file.cpp
```

### Common Checks

```bash
# Use auto where appropriate
clang-tidy -checks='modernize-use-auto' -fix file.cpp

# Use nullptr instead of NULL/0
clang-tidy -checks='modernize-use-nullptr' -fix file.cpp

# Use override keyword
clang-tidy -checks='modernize-use-override' -fix file.cpp

# Use range-based for loops
clang-tidy -checks='modernize-loop-convert' -fix file.cpp

# Avoid C-style casts
clang-tidy -checks='cppcoreguidelines-pro-type-cstyle-cast' file.cpp

# Check for memory leaks
clang-tidy -checks='clang-analyzer-cplusplus.NewDelete*' file.cpp

# Performance: unnecessary copies
clang-tidy -checks='performance-unnecessary-copy-initialization' file.cpp

# Readability: naming conventions
clang-tidy -checks='readability-identifier-naming' file.cpp
```

### Build System Integration

```bash
# With CMake (generate compile_commands.json)
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
clang-tidy -p build/ src/file.cpp

# Run on all files in compilation database
find src -name '*.cpp' | xargs clang-tidy -p build/

# Using run-clang-tidy.py (parallel execution)
run-clang-tidy.py -p build/ -checks='*' -fix

# Makefile integration
make clean
bear -- make
clang-tidy -p . src/file.cpp
```

### Suppressing Warnings

```cpp
// Suppress specific warning
// NOLINTNEXTLINE(check-name)
int bad_code = 0;

// Suppress for entire line
int bad_code = 0;  // NOLINT

// Suppress specific checks
// NOLINTNEXTLINE(readability-magic-numbers, cppcoreguidelines-avoid-magic-numbers)
int value = 42;

// Suppress in region
// NOLINTBEGIN(check-name)
int bad_code1 = 0;
int bad_code2 = 0;
// NOLINTEND(check-name)
```

## clangd (Language Server Protocol)

clangd is a language server that provides IDE features for C/C++ development.

### Installation

```bash
# Ubuntu/Debian
sudo apt install clangd

# macOS
brew install llvm
# clangd is in /usr/local/opt/llvm/bin/clangd

# Verify installation
clangd --version

# Update alternatives (if multiple versions)
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-15 100
```

### Configuration

```yaml
# ~/.config/clangd/config.yaml
CompileFlags:
  Add:
    - "-Wall"
    - "-Wextra"
    - "-std=c++17"
  Remove:
    - "-W*"
  CompilationDatabase: build/

Index:
  Background: Build

Diagnostics:
  ClangTidy:
    Add:
      - modernize*
      - bugprone*
    Remove:
      - modernize-use-trailing-return-type
  UnusedIncludes: Strict

Hover:
  ShowAKA: Yes

InlayHints:
  Enabled: Yes
  ParameterNames: Yes
  DeducedTypes: Yes
```

### VS Code Integration

```json
// settings.json
{
  "clangd.path": "/usr/bin/clangd",
  "clangd.arguments": [
    "--background-index",
    "--clang-tidy",
    "--header-insertion=iwyu",
    "--completion-style=detailed",
    "--function-arg-placeholders",
    "--fallback-style=llvm"
  ],
  "clangd.checkUpdates": true,
  "[cpp]": {
    "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd",
    "editor.formatOnSave": true
  }
}
```

### Vim/Neovim Integration

```vim
" Using coc.nvim
" Install: :CocInstall coc-clangd

" ~/.config/nvim/coc-settings.json
{
  "clangd.path": "/usr/bin/clangd",
  "clangd.arguments": [
    "--background-index",
    "--clang-tidy",
    "--header-insertion=iwyu"
  ]
}

" Using vim-lsp
Plug 'prabirshrestha/vim-lsp'
Plug 'mattn/vim-lsp-settings'

" Auto-install clangd
:LspInstallServer clangd
```

### Emacs Integration

```elisp
;; Using lsp-mode
(use-package lsp-mode
  :commands lsp
  :config
  (setq lsp-clients-clangd-args
    '("--background-index"
      "--clang-tidy"
      "--header-insertion=iwyu")))

(add-hook 'c-mode-hook 'lsp)
(add-hook 'c++-mode-hook 'lsp)
```

### Compilation Database

```bash
# Generate with CMake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
ln -s build/compile_commands.json .

# Generate with Bear
bear -- make

# Generate with compiledb (for Make)
compiledb make

# Manual compile_commands.json
cat > compile_commands.json << 'EOF'
[
  {
    "directory": "/home/user/project",
    "command": "clang++ -std=c++17 -Wall src/main.cpp",
    "file": "src/main.cpp"
  }
]
EOF
```

### Features

```bash
# Code completion
# Automatic as you type

# Go to definition
# Ctrl+Click or F12 (VS Code)
# gd in Vim with LSP

# Find references
# Shift+F12 (VS Code)
# gr in Vim

# Hover documentation
# Hover with mouse or K in Vim

# Code formatting
# Alt+Shift+F (VS Code)
# :Format in Vim

# Rename symbol
# F2 (VS Code)
# <leader>rn in Vim

# Diagnostics
# Automatic inline errors and warnings

# Fix-its
# Automatic code fixes suggested
```

## Other Clang Tools

### clang-check

Static analysis and AST dumping tool.

```bash
# Check syntax
clang-check file.cpp

# With compilation database
clang-check -p build/ file.cpp

# Dump AST
clang-check -ast-dump file.cpp

# Dump specific function
clang-check -ast-dump -ast-dump-filter=functionName file.cpp

# Run static analyzer
clang-check --analyze file.cpp
```

### clang-query

Interactive tool for querying the Clang AST.

```bash
# Start interactive mode
clang-query file.cpp

# Execute query from command line
clang-query -c "match functionDecl()" file.cpp

# Common queries
match functionDecl()                    # Find all functions
match functionDecl(isMain())            # Find main function
match varDecl()                         # Find all variables
match recordDecl(isClass())             # Find all classes
match callExpr()                        # Find all function calls

# With filters
match functionDecl(hasName("foo"))
match functionDecl(returns(asString("int")))
match varDecl(hasType(isInteger()))

# Set output style
set output detailed
set output dump
```

### scan-build

Static analyzer wrapper for build systems.

```bash
# Analyze with make
scan-build make

# Analyze with CMake
scan-build cmake ..
scan-build make

# Specify analyzer
scan-build --use-analyzer=/usr/bin/clang make

# View results in browser
scan-build -o analysis-results make

# Enable all checks
scan-build -enable-checker alpha make

# Verbose output
scan-build -v make

# Generate HTML report
scan-build -o ./scan-results make
```

### clang-apply-replacements

Apply fix-it hints and replacements.

```bash
# Apply fixes from clang-tidy
clang-tidy -export-fixes=fixes.yaml file.cpp
clang-apply-replacements .

# Apply fixes from directory
clang-apply-replacements /path/to/fixes/

# Format after applying
clang-apply-replacements -format .
```

## Sanitizers

Sanitizers are runtime error detection tools built into Clang.

### AddressSanitizer (ASan)

Detects memory errors like buffer overflows, use-after-free, memory leaks.

```bash
# Compile with ASan
clang -fsanitize=address -g program.c -o program

# C++ with ASan
clang++ -fsanitize=address -g program.cpp -o program

# Run the program
./program

# With additional options
ASAN_OPTIONS=detect_leaks=1:symbolize=1 ./program

# Check for memory leaks only
ASAN_OPTIONS=detect_leaks=1 ./program

# Detailed error messages
ASAN_OPTIONS=verbosity=1:malloc_context_size=20 ./program

# Halt on first error
ASAN_OPTIONS=halt_on_error=1 ./program

# ASan with optimization
clang -fsanitize=address -O1 -g -fno-omit-frame-pointer program.c
```

### MemorySanitizer (MSan)

Detects use of uninitialized memory.

```bash
# Compile with MSan
clang -fsanitize=memory -g program.c -o program

# Track origins of uninitialized values
clang -fsanitize=memory -fsanitize-memory-track-origins -g program.c

# Run with options
MSAN_OPTIONS=halt_on_error=0 ./program

# Must compile ALL code with MSan (including libraries)
clang -fsanitize=memory -g main.c lib.c -o program
```

### ThreadSanitizer (TSan)

Detects data races in multithreaded programs.

```bash
# Compile with TSan
clang -fsanitize=thread -g program.c -o program

# Link with pthread
clang -fsanitize=thread -g program.c -lpthread -o program

# Run with options
TSAN_OPTIONS=second_deadlock_stack=1 ./program

# Suppress specific warnings
# Create tsan.supp file
echo "race:^FunctionName$" > tsan.supp
TSAN_OPTIONS=suppressions=tsan.supp ./program
```

### UndefinedBehaviorSanitizer (UBSan)

Detects various undefined behaviors.

```bash
# Compile with UBSan
clang -fsanitize=undefined -g program.c -o program

# Specific checks
clang -fsanitize=null -g program.c              # Null pointer
clang -fsanitize=signed-integer-overflow -g     # Integer overflow
clang -fsanitize=shift -g                       # Invalid shifts
clang -fsanitize=bounds -g                      # Array bounds

# Multiple checks
clang -fsanitize=undefined,integer -g program.c

# Trap on error (no runtime library)
clang -fsanitize=undefined -fsanitize-trap=undefined program.c

# Print stack traces
UBSAN_OPTIONS=print_stacktrace=1 ./program

# Halt on first error
UBSAN_OPTIONS=halt_on_error=1 ./program
```

### LeakSanitizer (LSan)

Detects memory leaks (part of ASan).

```bash
# Use with ASan
clang -fsanitize=address -g program.c -o program

# Standalone leak detection
clang -fsanitize=leak -g program.c -o program

# Run with leak detection
LSAN_OPTIONS=verbosity=1:log_threads=1 ./program

# Suppress leaks
echo "leak:FunctionName" > lsan.supp
LSAN_OPTIONS=suppressions=lsan.supp ./program
```

### Combining Sanitizers

```bash
# ASan + UBSan
clang -fsanitize=address,undefined -g program.c -o program

# Multiple sanitizers (not all can be combined)
clang -fsanitize=address,undefined,integer -g program.c

# Cannot combine TSan with ASan/MSan
# Use separately

# Common combination for testing
clang -fsanitize=address,undefined,leak \
  -fno-omit-frame-pointer \
  -g -O1 program.c -o program
```

## Cross-Compilation

### Target Specification

```bash
# Specify target triple
clang --target=arm-linux-gnueabihf main.c -o program

# Common targets
clang --target=aarch64-linux-gnu          # ARM64 Linux
clang --target=arm-linux-gnueabihf        # ARM Linux (hard float)
clang --target=x86_64-w64-mingw32         # Windows 64-bit
clang --target=i686-w64-mingw32           # Windows 32-bit
clang --target=wasm32-wasi                # WebAssembly

# Show default target
clang -v

# List available targets
llc --version
```

### Sysroot Configuration

```bash
# Specify sysroot
clang --target=arm-linux-gnueabihf \
  --sysroot=/usr/arm-linux-gnueabihf \
  main.c -o program

# With GCC toolchain
clang --target=arm-linux-gnueabihf \
  --gcc-toolchain=/usr/arm-linux-gnueabihf \
  main.c -o program

# Multiple paths
clang --target=arm-linux-gnueabihf \
  --sysroot=/usr/arm-linux-gnueabihf \
  -I/opt/cross/include \
  -L/opt/cross/lib \
  main.c -o program
```

### Cross-Compilation Example

```bash
# Install ARM cross-compiler
sudo apt install gcc-arm-linux-gnueabihf

# Cross-compile for ARM
clang --target=arm-linux-gnueabihf \
  --sysroot=/usr/arm-linux-gnueabihf \
  -march=armv7-a \
  -mfpu=neon \
  main.c -o program-arm

# Verify target
file program-arm
# program-arm: ELF 32-bit LSB executable, ARM

# Cross-compile for Windows
sudo apt install mingw-w64
clang --target=x86_64-w64-mingw32 \
  -L/usr/x86_64-w64-mingw32/lib \
  main.c -o program.exe

# WebAssembly
clang --target=wasm32-wasi \
  --sysroot=/opt/wasi-sysroot \
  main.c -o program.wasm
```

### CMake Cross-Compilation

```cmake
# toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ARM)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)

set(CMAKE_SYSROOT /usr/arm-linux-gnueabihf)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

```bash
# Use toolchain file
cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake ..
make
```

## Build System Integration

### CMake

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Use Clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable clang-tidy
set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*)

# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# Executable
add_executable(myprogram main.cpp utils.cpp)
```

```bash
# Configure with Clang
CC=clang CXX=clang++ cmake ..

# Or with CMake option
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..

# Build
cmake --build .

# With sanitizers
cmake -DENABLE_ASAN=ON ..
```

### Makefile

```makefile
# Makefile
CC = clang
CXX = clang++
CFLAGS = -Wall -Wextra -std=c11 -O2
CXXFLAGS = -Wall -Wextra -std=c++17 -O2

# Sanitizers (optional)
SANITIZE = -fsanitize=address,undefined

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Files
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/program

# Targets
all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(SANITIZE) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR) $(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Format code
format:
	find $(SRCDIR) -name '*.cpp' -o -name '*.h' | xargs clang-format -i

# Run linter
lint:
	clang-tidy $(SRCS) -- $(CXXFLAGS)

.PHONY: all clean format lint
```

### Compilation Database

```bash
# Generate with CMake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
ln -s build/compile_commands.json .

# Generate with Bear (for Makefiles)
bear -- make

# Generate with compiledb
pip install compiledb
compiledb make

# Manual JSON format
cat > compile_commands.json << 'EOF'
[
  {
    "directory": "/home/user/project",
    "command": "clang++ -std=c++17 -Wall -Iinclude -c src/main.cpp -o obj/main.o",
    "file": "src/main.cpp"
  },
  {
    "directory": "/home/user/project",
    "command": "clang++ -std=c++17 -Wall -Iinclude -c src/utils.cpp -o obj/utils.o",
    "file": "src/utils.cpp"
  }
]
EOF

# Verify compilation database
clangd --check=/path/to/file.cpp
```

## Common Patterns and Workflows

### Development Workflow

```bash
# 1. Project setup
mkdir -p myproject/{src,include,build,tests}
cd myproject

# 2. Initialize Git
git init

# 3. Create .clang-format
clang-format -style=llvm -dump-config > .clang-format

# 4. Create .clang-tidy
cat > .clang-tidy << 'EOF'
---
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  modernize-*,
  performance-*,
  readability-*
EOF

# 5. Create CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(myapp src/main.cpp)
EOF

# 6. Configure and build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
ln -s build/compile_commands.json ../compile_commands.json

# 7. Develop with clangd LSP support
# (automatic completion, diagnostics, etc.)

# 8. Format before commit
clang-format -i src/*.cpp include/*.h

# 9. Run linter
clang-tidy src/*.cpp

# 10. Test with sanitizers
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" ..
make && ./myapp
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Format code
echo "Running clang-format..."
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(c|cpp|h|hpp)$'); do
    clang-format -i "$file"
    git add "$file"
done

# Run clang-tidy
echo "Running clang-tidy..."
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp)$'); do
    clang-tidy "$file" -- -std=c++17
    if [ $? -ne 0 ]; then
        echo "clang-tidy failed for $file"
        exit 1
    fi
done

echo "Pre-commit checks passed!"
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install -y clang clang-format clang-tidy cmake

    - name: Check formatting
      run: |
        find src include -name '*.cpp' -o -name '*.h' | \
          xargs clang-format --dry-run -Werror

    - name: Build
      run: |
        mkdir build && cd build
        cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
        make

    - name: Run clang-tidy
      run: |
        cd build
        run-clang-tidy.py -p . -checks='*' ../src

    - name: Test with sanitizers
      run: |
        mkdir build-asan && cd build-asan
        cmake -DCMAKE_C_COMPILER=clang \
              -DCMAKE_CXX_COMPILER=clang++ \
              -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" ..
        make
        ./myapp
```

### Makefile with All Tools

```makefile
CC = clang
CXX = clang++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
TARGET = myapp
SRCS = $(wildcard src/*.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean format lint analyze asan

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

format:
	find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i

lint:
	clang-tidy $(SRCS) -- $(CXXFLAGS)

lint-fix:
	clang-tidy -fix $(SRCS) -- $(CXXFLAGS)

analyze:
	scan-build make

asan:
	$(CXX) $(CXXFLAGS) -fsanitize=address,undefined -g $(SRCS) -o $(TARGET)
	./$(TARGET)

compdb:
	bear -- make
```

### Code Review Workflow

```bash
# 1. Format code
make format

# 2. Run static analysis
make lint

# 3. Fix automatic issues
clang-tidy -fix src/*.cpp

# 4. Run tests with sanitizers
make asan

# 5. Check for compilation database
clangd --check=src/main.cpp

# 6. Commit changes
git add .
git commit -m "Fix issues found by clang-tidy"

# 7. Push for review
git push origin feature-branch
```

## Best Practices

### Project Configuration

```bash
# Project structure
myproject/
├── .clang-format          # Code style
├── .clang-tidy            # Linter config
├── .clangd                # LSP config (optional)
├── compile_commands.json  # Compilation database
├── CMakeLists.txt         # Build config
├── include/               # Public headers
├── src/                   # Source files
├── tests/                 # Test files
└── build/                 # Build artifacts

# Essential files
.clang-format     # Consistent formatting
.clang-tidy       # Static analysis rules
.gitignore        # Exclude build artifacts
CMakeLists.txt    # Build system
```

### Compiler Flags

```bash
# Development build
clang++ -Wall -Wextra -Wpedantic -Wshadow \
  -g -O0 -std=c++17 \
  -fsanitize=address,undefined \
  main.cpp -o myapp

# Release build
clang++ -Wall -Wextra -O3 -DNDEBUG \
  -std=c++17 -flto \
  main.cpp -o myapp

# Security-hardened build
clang++ -Wall -Wextra -O2 \
  -D_FORTIFY_SOURCE=2 \
  -fstack-protector-strong \
  -fPIE -pie \
  -Wformat -Wformat-security \
  main.cpp -o myapp
```

### Code Quality Checks

```bash
# Comprehensive checking workflow
make clean
make format              # Format code
make lint                # Run clang-tidy
make analyze             # Static analysis with scan-build
make asan                # Test with sanitizers
make test                # Run unit tests

# Automated in CI/CD
clang-format --dry-run -Werror src/*.cpp
clang-tidy src/*.cpp
scan-build make
make test
```

### Performance Optimization

```bash
# Profile-guided optimization (PGO)
# Step 1: Build with profiling
clang++ -O2 -fprofile-generate program.cpp -o program

# Step 2: Run with typical workload
./program < typical_input.txt

# Step 3: Build with profile data
clang++ -O2 -fprofile-use program.cpp -o program

# Link-time optimization (LTO)
clang++ -O3 -flto main.cpp utils.cpp -o program

# Fast math (trade precision for speed)
clang++ -O3 -ffast-math program.cpp -o program

# CPU-specific optimization
clang++ -O3 -march=native program.cpp -o program
```

### IDE Setup Recommendations

```bash
# 1. Generate compilation database
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# 2. Symlink to project root
ln -s build/compile_commands.json .

# 3. Configure clangd
cat > .clangd << 'EOF'
CompileFlags:
  CompilationDatabase: build/
EOF

# 4. Install editor extension
# VS Code: Install "clangd" extension
# Vim: Install coc-clangd or vim-lsp
# Emacs: Use lsp-mode with clangd

# 5. Verify setup
clangd --check=src/main.cpp
```

## Troubleshooting

### Common Errors

```bash
# "undefined reference" errors
# Problem: Missing library or object file
# Solution: Add -l flag for libraries
clang main.c -lm -lpthread -o program

# "cannot find -lxxx" error
# Problem: Library not in search path
# Solution: Add library path with -L
clang main.c -L/usr/local/lib -lmylib -o program

# "fatal error: 'header.h' file not found"
# Problem: Include path not specified
# Solution: Add include path with -I
clang -I./include main.c -o program

# Multiple definition errors
# Problem: Symbol defined in multiple files
# Solution: Use static or inline, or fix header guards

# Sanitizer errors
# Problem: Real bugs in code
# Solution: Fix the code based on sanitizer output
```

### Performance Issues

```bash
# Slow compilation
# Use ccache
export CC="ccache clang"
export CXX="ccache clang++"

# Parallel compilation
make -j$(nproc)

# Precompiled headers
clang++ -x c++-header pch.h -o pch.h.pch
clang++ -include-pch pch.h.pch main.cpp

# clangd high memory usage
# Limit background indexing in .clangd config
cat > .clangd << 'EOF'
Index:
  Background: Skip
EOF

# Slow clang-tidy
# Run on changed files only
git diff --name-only | grep '\.cpp$' | xargs clang-tidy
```

### Debugging

```bash
# Show compilation commands
clang -v main.c

# Show detailed compilation stages
clang -### main.c

# Dump preprocessor output
clang -E main.c > main.i

# Dump AST
clang -Xclang -ast-dump main.c

# Show include tree
clang -H main.c

# Verbose linking
clang -Wl,--verbose main.c

# Debug clangd
clangd --log=verbose --check=src/main.cpp 2>&1 | tee clangd.log
```

### Clean Build

```bash
# Full clean build
make clean
rm -rf build/
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
make

# Clear ccache
ccache --clear

# Regenerate compilation database
rm compile_commands.json
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
ln -s build/compile_commands.json .
```

## Complete Example Project

```bash
# Project structure
mkdir -p myproject/{src,include,tests,build}
cd myproject

# Main source file
cat > src/main.cpp << 'EOF'
#include <iostream>
#include "utils.h"

int main() {
    std::cout << "Result: " << calculate(10, 20) << std::endl;
    return 0;
}
EOF

# Header file
cat > include/utils.h << 'EOF'
#ifndef UTILS_H
#define UTILS_H

int calculate(int a, int b);

#endif
EOF

# Implementation file
cat > src/utils.cpp << 'EOF'
#include "utils.h"

int calculate(int a, int b) {
    return a + b;
}
EOF

# CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(MyProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler options
add_compile_options(-Wall -Wextra -Wpedantic)

# Include directories
include_directories(include)

# Executable
add_executable(myapp
    src/main.cpp
    src/utils.cpp
)

# Optional: Enable sanitizers
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(myapp PRIVATE -fsanitize=address,undefined)
    target_link_options(myapp PRIVATE -fsanitize=address,undefined)
endif()
EOF

# .clang-format
cat > .clang-format << 'EOF'
---
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
EOF

# .clang-tidy
cat > .clang-tidy << 'EOF'
---
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  modernize-*,
  performance-*,
  readability-*
EOF

# Makefile wrapper
cat > Makefile << 'EOF'
BUILD_DIR = build

.PHONY: all configure build clean format lint test

all: build

configure:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
	ln -sf $(BUILD_DIR)/compile_commands.json .

build: configure
	cmake --build $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) compile_commands.json

format:
	find src include -name '*.cpp' -o -name '*.h' | xargs clang-format -i

lint:
	clang-tidy src/*.cpp -- -Iinclude

test:
	$(BUILD_DIR)/myapp

asan: configure
	cd $(BUILD_DIR) && cmake -DENABLE_ASAN=ON ..
	cmake --build $(BUILD_DIR)
	$(BUILD_DIR)/myapp
EOF

# Build and run
make
make test

# Format and lint
make format
make lint

# Test with sanitizers
make asan
```

## Useful Tips

1. **Always use compilation database** (`compile_commands.json`) for accurate IDE support
2. **Enable warnings** (`-Wall -Wextra -Wpedantic`) to catch potential bugs early
3. **Use sanitizers during development** to find memory errors and undefined behavior
4. **Format code automatically** with clang-format to maintain consistency
5. **Run clang-tidy regularly** to catch common mistakes and enforce best practices
6. **Configure clangd** for powerful IDE features in any editor
7. **Use link-time optimization** (`-flto`) for release builds
8. **Enable debug symbols** (`-g`) even with optimization for better debugging
9. **Create .clang-format and .clang-tidy** files in project root for consistency
10. **Test with multiple sanitizers** to catch different classes of bugs

Clang provides a comprehensive ecosystem of tools that improve code quality, catch bugs early, and enhance developer productivity through excellent IDE integration and static analysis capabilities.
