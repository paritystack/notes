# WebAssembly (Wasm)

WebAssembly (abbreviated Wasm) is a binary instruction format for a stack-based virtual machine. It's designed as a portable compilation target for programming languages, enabling deployment of high-performance applications on the web and beyond.

## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Language Support](#language-support)
- [JavaScript Interoperability](#javascript-interoperability)
- [Memory Management](#memory-management)
- [WASI (WebAssembly System Interface)](#wasi-webassembly-system-interface)
- [Performance](#performance)
- [Tools & Ecosystem](#tools--ecosystem)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Debugging](#debugging)
- [Security](#security)

---

## Overview

### What is WebAssembly?

WebAssembly is a **low-level bytecode format** that runs in modern web browsers alongside JavaScript. It provides near-native performance and allows code written in languages like C, C++, Rust, and Go to run on the web.

### Key Features

- **Fast**: Near-native execution speed
- **Safe**: Sandboxed execution environment
- **Portable**: Platform-independent bytecode
- **Compact**: Efficient binary format
- **Open**: Standardized by W3C

### Why WebAssembly?

```
┌─────────────────────────────────────────────────┐
│         Traditional Web Development             │
│  JavaScript (interpreted/JIT compiled)          │
│  - Limited performance                          │
│  - Single language choice                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         WebAssembly Era                         │
│  JavaScript + Wasm (pre-compiled binary)        │
│  - Near-native performance                      │
│  - Multiple language support                    │
│  - Reuse existing codebases                     │
└─────────────────────────────────────────────────┘
```

### ELI10 (Explain Like I'm 10)

Think of WebAssembly like a universal translator for computer programs. Just like you can compile a C++ game to run on Windows, Mac, or PlayStation, WebAssembly lets you compile programs to run in any web browser, super fast!

---

## Core Concepts

### 1. Module

A WebAssembly **module** is the compiled unit containing functions, memory, tables, and globals.

```javascript
// Loading a WebAssembly module
const response = await fetch('module.wasm');
const bytes = await response.arrayBuffer();
const { instance } = await WebAssembly.instantiate(bytes);

// Call exported function
const result = instance.exports.add(5, 3);
console.log(result); // 8
```

### 2. Memory

WebAssembly uses **linear memory** - a contiguous, expandable array of bytes.

```javascript
// Creating memory
const memory = new WebAssembly.Memory({
  initial: 1,  // 1 page = 64KB
  maximum: 10  // Max 10 pages = 640KB
});

// Accessing memory
const buffer = new Uint8Array(memory.buffer);
buffer[0] = 42;
```

### 3. Table

**Tables** store references to functions or other objects.

```javascript
const table = new WebAssembly.Table({
  initial: 2,
  element: 'anyfunc'
});
```

### 4. Globals

**Globals** are mutable or immutable values accessible across module boundaries.

```javascript
const global = new WebAssembly.Global({
  value: 'i32',
  mutable: true
}, 42);

console.log(global.value); // 42
global.value = 100;
```

---

## Architecture

### WebAssembly Execution Model

```
┌──────────────────────────────────────────┐
│    Source Code (C/C++/Rust/etc.)        │
└──────────────┬───────────────────────────┘
               │ Compile
               ▼
┌──────────────────────────────────────────┐
│    WebAssembly Binary (.wasm)            │
└──────────────┬───────────────────────────┘
               │ Load & Instantiate
               ▼
┌──────────────────────────────────────────┐
│    WebAssembly VM (in Browser)           │
│    - Stack-based execution               │
│    - JIT compilation                     │
│    - Sandboxed environment               │
└──────────────┬───────────────────────────┘
               │ Execute
               ▼
┌──────────────────────────────────────────┐
│    JavaScript Interop & DOM Access       │
└──────────────────────────────────────────┘
```

### Value Types

WebAssembly supports four basic value types:

| Type   | Description           | Size    |
|--------|-----------------------|---------|
| `i32`  | 32-bit integer        | 4 bytes |
| `i64`  | 64-bit integer        | 8 bytes |
| `f32`  | 32-bit float          | 4 bytes |
| `f64`  | 64-bit float          | 8 bytes |

**New in Wasm 2.0**:
- `v128` - 128-bit SIMD vector
- Reference types (externref, funcref)

---

## Use Cases

### 1. Performance-Critical Applications

- **Game Engines**: Unity, Unreal Engine
- **Video/Audio Processing**: FFmpeg, codecs
- **Image Manipulation**: Photoshop, Figma
- **Simulations**: Physics engines, scientific computing

### 2. Code Portability

- **Legacy Code**: Run existing C/C++ libraries in the browser
- **Cross-Platform**: Write once, run anywhere
- **Code Reuse**: Share logic between server and client

### 3. Cryptography

```javascript
// Using a Wasm crypto library
const wasmCrypto = await loadWasmCrypto();
const hash = wasmCrypto.sha256(data);
```

### 4. Compression/Decompression

```javascript
// Wasm-based compression
const compressed = wasmModule.compress(largeData);
const decompressed = wasmModule.decompress(compressed);
```

### 5. Machine Learning

- **TensorFlow.js with Wasm backend**
- **ONNX Runtime**
- **ML model inference**

---

## Getting Started

### Hello World Example

**C Code** (`hello.c`):
```c
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    printf("Hello from WebAssembly!\n");
    return 0;
}
```

**Compile with Emscripten**:
```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Compile to Wasm
emcc hello.c -o hello.html
```

**JavaScript Usage**:
```javascript
// Load and use the module
const Module = await createModule();
const result = Module._add(5, 3);
console.log(result); // 8
```

### WAT (WebAssembly Text Format)

WebAssembly has a human-readable text format:

**hello.wat**:
```wat
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)
```

**Compile WAT to Wasm**:
```bash
# Using wat2wasm (from WABT toolkit)
wat2wasm hello.wat -o hello.wasm
```

---

## Language Support

### 1. C/C++ (Emscripten)

**Installation**:
```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
```

**Example** (`math.cpp`):
```cpp
#include <emscripten.h>

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
```

**Compile**:
```bash
emcc math.cpp -o math.js \
  -s EXPORTED_FUNCTIONS='["_fibonacci"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]'
```

**JavaScript**:
```javascript
const Module = await createModule();
const fib = Module.cwrap('fibonacci', 'number', ['number']);
console.log(fib(10)); // 55
```

### 2. Rust

**Installation**:
```bash
# Add Wasm target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**Example** (`src/lib.rs`):
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
```

**Cargo.toml**:
```toml
[package]
name = "wasm-example"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
```

**Build**:
```bash
wasm-pack build --target web
```

**JavaScript**:
```javascript
import init, { add, greet } from './pkg/wasm_example.js';

await init();
console.log(add(5, 3));        // 8
console.log(greet("World"));   // "Hello, World!"
```

### 3. AssemblyScript

AssemblyScript is **TypeScript-like syntax** that compiles to WebAssembly.

**Installation**:
```bash
npm install -g assemblyscript
```

**Example** (`assembly/index.ts`):
```typescript
export function add(a: i32, b: i32): i32 {
  return a + b;
}

export function factorial(n: i32): i32 {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}
```

**Compile**:
```bash
asc assembly/index.ts --outFile build/optimized.wasm --optimize
```

**JavaScript**:
```javascript
const { add, factorial } = await WebAssembly.instantiateStreaming(
  fetch('build/optimized.wasm')
).then(obj => obj.instance.exports);

console.log(add(5, 3));      // 8
console.log(factorial(5));   // 120
```

### 4. Go

**Example** (`main.go`):
```go
package main

import (
    "syscall/js"
)

func add(this js.Value, args []js.Value) interface{} {
    return args[0].Int() + args[1].Int()
}

func main() {
    js.Global().Set("add", js.FuncOf(add))
    <-make(chan bool) // Keep the program running
}
```

**Compile**:
```bash
GOOS=js GOARCH=wasm go build -o main.wasm main.go
```

**JavaScript**:
```javascript
const go = new Go();
const result = await WebAssembly.instantiateStreaming(
  fetch('main.wasm'),
  go.importObject
);
go.run(result.instance);

// Call Go function
const sum = add(5, 3);
```

---

## JavaScript Interoperability

### Calling JavaScript from Wasm

**Rust with wasm-bindgen**:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Import JavaScript console.log
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // Import JavaScript alert
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    log(&format!("Hello, {}!", name));
    alert(&format!("Welcome, {}!", name));
}
```

### Calling Wasm from JavaScript

```javascript
// Instantiate with imports
const importObject = {
  env: {
    consoleLog: (arg) => console.log(arg),
    jsMultiply: (a, b) => a * b
  }
};

const { instance } = await WebAssembly.instantiateStreaming(
  fetch('module.wasm'),
  importObject
);

// Call exported function
instance.exports.wasmFunction();
```

### Passing Complex Data

**Passing Strings**:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn reverse_string(s: String) -> String {
    s.chars().rev().collect()
}
```

```javascript
import { reverse_string } from './pkg';
console.log(reverse_string("Hello")); // "olleH"
```

**Passing Arrays**:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn sum_array(arr: &[i32]) -> i32 {
    arr.iter().sum()
}
```

```javascript
import { sum_array } from './pkg';
const arr = new Int32Array([1, 2, 3, 4, 5]);
console.log(sum_array(arr)); // 15
```

---

## Memory Management

### Linear Memory

WebAssembly uses a **linear memory model** - a contiguous, resizable array of bytes.

```javascript
// Create memory (1 page = 64KB)
const memory = new WebAssembly.Memory({
  initial: 1,   // Initial size: 1 page
  maximum: 100  // Max size: 100 pages
});

// Access as typed array
const uint8View = new Uint8Array(memory.buffer);
const uint32View = new Uint32Array(memory.buffer);

// Write data
uint8View[0] = 42;
uint32View[1] = 0xDEADBEEF;

// Grow memory
memory.grow(1); // Add 1 page (64KB)
```

### Sharing Memory

**JavaScript Side**:
```javascript
const memory = new WebAssembly.Memory({ initial: 1 });

const importObject = {
  js: { mem: memory }
};

const { instance } = await WebAssembly.instantiateStreaming(
  fetch('module.wasm'),
  importObject
);

// Access shared memory
const buffer = new Uint8Array(memory.buffer);
```

**C Side**:
```c
#include <emscripten.h>

EMSCRIPTEN_KEEPALIVE
void writeToMemory(int offset, int value) {
    int* ptr = (int*)offset;
    *ptr = value;
}
```

### Memory Layout

```
┌────────────────────────────────────────┐
│  WebAssembly Linear Memory             │
├────────────────────────────────────────┤
│  Stack (grows downward)                │ ← SP (Stack Pointer)
│  ↓                                     │
│                                        │
│  Heap (grows upward)                   │ ← Managed by allocator
│  ↑                                     │
│  Globals & Static Data                 │
│  Code (if using dynamic linking)       │
└────────────────────────────────────────┘
```

---

## WASI (WebAssembly System Interface)

WASI allows WebAssembly to **run outside the browser** with standardized system interfaces.

### Use Cases

- **Server-side applications**
- **CLI tools**
- **Edge computing**
- **Serverless functions**

### Example with Rust

**Cargo.toml**:
```toml
[dependencies]
```

**src/main.rs**:
```rust
use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);

    let mut file = File::create("output.txt").unwrap();
    file.write_all(b"Hello from WASI!").unwrap();
}
```

**Build**:
```bash
rustc --target wasm32-wasi src/main.rs -o app.wasm
```

**Run with Wasmtime**:
```bash
wasmtime app.wasm arg1 arg2
```

### WASI APIs

```rust
// File I/O
use std::fs;
let contents = fs::read_to_string("file.txt")?;

// Environment variables
use std::env;
let path = env::var("PATH")?;

// Command-line arguments
let args: Vec<String> = env::args().collect();

// Random numbers
use std::time::SystemTime;
let now = SystemTime::now();
```

---

## Performance

### Benchmarking: JavaScript vs WebAssembly

```javascript
// JavaScript version
function fibJS(n) {
  if (n <= 1) return n;
  return fibJS(n - 1) + fibJS(n - 2);
}

// Benchmark
console.time('JS');
console.log(fibJS(40));
console.timeEnd('JS');
// JS: ~1200ms

console.time('Wasm');
console.log(wasmModule.fibonacci(40));
console.timeEnd('Wasm');
// Wasm: ~300ms (4x faster!)
```

### Optimization Techniques

#### 1. SIMD (Single Instruction, Multiple Data)

```c
#include <wasm_simd128.h>

void add_arrays_simd(float* a, float* b, float* result, int len) {
    for (int i = 0; i < len; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        v128_t vr = wasm_f32x4_add(va, vb);
        wasm_v128_store(&result[i], vr);
    }
}
```

#### 2. Multithreading

```javascript
// Create shared memory
const memory = new WebAssembly.Memory({
  initial: 1,
  maximum: 10,
  shared: true  // Enable sharing
});

// Use with Web Workers
const worker = new Worker('worker.js');
worker.postMessage({ memory });
```

#### 3. Compilation Flags

**Emscripten**:
```bash
emcc -O3 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 \
     -s SIMD=1 -s ASSERTIONS=0 \
     source.c -o output.js
```

**Rust**:
```bash
wasm-pack build --release -- \
  -Z build-std=std,panic_abort \
  -Z build-std-features=panic_immediate_abort
```

### Performance Best Practices

✅ **Do**:
- Minimize JavaScript ↔ Wasm calls
- Batch operations
- Use SIMD when possible
- Pre-compile modules
- Use streaming compilation

❌ **Don't**:
- Frequently marshal complex data structures
- Make many small function calls
- Ignore memory growth overhead
- Use unoptimized builds in production

---

## Tools & Ecosystem

### 1. Emscripten

C/C++ to WebAssembly compiler.

```bash
# Install
emsdk install latest
emsdk activate latest

# Compile
emcc source.c -o output.html

# Optimize
emcc -O3 source.c -o output.js
```

### 2. wasm-pack (Rust)

```bash
# Build for web
wasm-pack build --target web

# Build for Node.js
wasm-pack build --target nodejs

# Build with profiling
wasm-pack build --profiling
```

### 3. WABT (WebAssembly Binary Toolkit)

```bash
# Convert WAT to Wasm
wat2wasm module.wat -o module.wasm

# Convert Wasm to WAT
wasm2wat module.wasm -o module.wat

# Validate Wasm
wasm-validate module.wasm

# Decompile to C-like syntax
wasm-decompile module.wasm
```

### 4. Wasmtime

High-performance WebAssembly runtime.

```bash
# Run WASI module
wasmtime run program.wasm

# Run with arguments
wasmtime run program.wasm -- arg1 arg2

# Map directories
wasmtime run --dir=/host/path program.wasm
```

### 5. Wasmer

Universal WebAssembly runtime.

```bash
# Run module
wasmer run module.wasm

# Create executable
wasmer create-exe module.wasm -o app

# Use packages
wasmer run cowsay hello
```

---

## Best Practices

### 1. Module Loading

**❌ Bad**: Blocking fetch
```javascript
const response = fetch('module.wasm');
const bytes = response.arrayBuffer();
const module = WebAssembly.instantiate(bytes);
```

**✅ Good**: Streaming compilation
```javascript
const { instance } = await WebAssembly.instantiateStreaming(
  fetch('module.wasm'),
  importObject
);
```

### 2. Memory Management

**❌ Bad**: Leaking memory
```rust
#[wasm_bindgen]
pub fn process_data(data: &[u8]) -> *const u8 {
    let result = data.to_vec();
    result.as_ptr() // Memory leak!
}
```

**✅ Good**: Proper cleanup
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn process_data(data: &[u8]) -> Vec<u8> {
    data.to_vec() // wasm-bindgen handles cleanup
}
```

### 3. Error Handling

**❌ Bad**: Panics
```rust
#[wasm_bindgen]
pub fn divide(a: i32, b: i32) -> i32 {
    a / b // Panics on division by zero!
}
```

**✅ Good**: Result types
```rust
#[wasm_bindgen]
pub fn divide(a: i32, b: i32) -> Result<i32, JsValue> {
    if b == 0 {
        Err(JsValue::from_str("Division by zero"))
    } else {
        Ok(a / b)
    }
}
```

### 4. Code Size Optimization

```bash
# Rust: Use minimal features
cargo build --target wasm32-unknown-unknown --release

# Optimize with wasm-opt
wasm-opt -Oz -o output.wasm input.wasm

# Strip debug info
wasm-strip output.wasm
```

---

## Common Patterns

### 1. Image Processing

**Rust**:
```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::ImageData;

#[wasm_bindgen]
pub fn grayscale(data: &mut [u8]) {
    for chunk in data.chunks_mut(4) {
        let gray = (chunk[0] as f32 * 0.299
                  + chunk[1] as f32 * 0.587
                  + chunk[2] as f32 * 0.114) as u8;
        chunk[0] = gray;
        chunk[1] = gray;
        chunk[2] = gray;
    }
}
```

**JavaScript**:
```javascript
import { grayscale } from './pkg';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

grayscale(imageData.data);
ctx.putImageData(imageData, 0, 0);
```

### 2. Game Loop

```rust
use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[wasm_bindgen]
pub struct Game {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Game {
        Game { x: 0.0, y: 0.0, vx: 1.0, vy: 1.0 }
    }

    pub fn update(&mut self, dt: f64) {
        self.x += self.vx * dt;
        self.y += self.vy * dt;
    }

    pub fn get_position(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }
}
```

### 3. Data Processing Pipeline

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn process_pipeline(data: &[f64]) -> Vec<f64> {
    data.iter()
        .map(|x| x * 2.0)       // Transform
        .filter(|x| *x > 10.0)  // Filter
        .take(100)              // Limit
        .collect()
}
```

---

## Debugging

### 1. Browser DevTools

Modern browsers support WebAssembly debugging:

**Chrome DevTools**:
- View Wasm modules in Sources panel
- Set breakpoints in WAT code
- Inspect memory

**Enable source maps**:
```bash
# Emscripten
emcc -g source.c -o output.js

# Rust
wasm-pack build --dev
```

### 2. Console Logging

**From Rust**:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn debug_function() {
    log("Debug message from Wasm!");
}
```

### 3. wasm-objdump

```bash
# View module sections
wasm-objdump -h module.wasm

# Disassemble
wasm-objdump -d module.wasm

# View imports/exports
wasm-objdump -x module.wasm
```

### 4. Performance Profiling

```javascript
performance.mark('wasm-start');
wasmModule.expensiveFunction();
performance.mark('wasm-end');

performance.measure('wasm-execution', 'wasm-start', 'wasm-end');
const measures = performance.getEntriesByType('measure');
console.log(measures[0].duration);
```

---

## Security

### Sandboxing

WebAssembly runs in a **sandboxed environment**:
- No direct access to OS
- No direct DOM access
- Memory isolation
- Capability-based security

### Security Best Practices

✅ **Do**:
- Validate all inputs from JavaScript
- Use WASI capabilities model
- Implement bounds checking
- Use secure random number generation

❌ **Don't**:
- Trust user input without validation
- Expose internal memory pointers
- Use predictable RNG for security
- Disable security features

### Example: Input Validation

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn process_string(input: String) -> Result<String, JsValue> {
    // Validate input
    if input.len() > 1000 {
        return Err(JsValue::from_str("Input too long"));
    }

    if !input.is_ascii() {
        return Err(JsValue::from_str("Non-ASCII characters"));
    }

    Ok(input.to_uppercase())
}
```

---

## Real-World Examples

### 1. Figma

Figma uses WebAssembly for:
- Rendering engine (C++)
- Complex calculations
- Near-native performance in browser

### 2. Google Earth

- Ported massive C++ codebase to WebAssembly
- Runs desktop-quality 3D graphics in browser

### 3. Autodesk AutoCAD

- 35-year-old C++ codebase
- Compiled to WebAssembly
- Full AutoCAD in browser

### 4. Video Editing (Clipchamp)

```javascript
// Video encoding with FFmpeg.wasm
import { createFFmpeg } from '@ffmpeg/ffmpeg';

const ffmpeg = createFFmpeg({ log: true });
await ffmpeg.load();

ffmpeg.FS('writeFile', 'input.mp4', videoData);
await ffmpeg.run('-i', 'input.mp4', '-vcodec', 'libx264', 'output.mp4');
const output = ffmpeg.FS('readFile', 'output.mp4');
```

---

## Further Learning

### Official Resources
- [WebAssembly Official Site](https://webassembly.org/)
- [MDN WebAssembly Guide](https://developer.mozilla.org/en-US/docs/WebAssembly)
- [WebAssembly Specification](https://webassembly.github.io/spec/)

### Tools Documentation
- [Emscripten Documentation](https://emscripten.org/docs/)
- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [AssemblyScript Documentation](https://www.assemblyscript.org/)

### Tutorials
- [Rust and WebAssembly Book](https://rustwasm.github.io/book/)
- [WebAssembly on MDN](https://developer.mozilla.org/en-US/docs/WebAssembly)
- [Lin Clark's Cartoon Guides](https://hacks.mozilla.org/category/code-cartoons/)

### Community
- [WebAssembly Discord](https://discord.gg/webassembly)
- [Bytecode Alliance](https://bytecodealliance.org/)

---

## Quick Reference

### Common Commands

```bash
# Emscripten (C/C++)
emcc source.c -o output.html
emcc -O3 source.c -o output.js

# Rust
cargo build --target wasm32-unknown-unknown
wasm-pack build --target web

# AssemblyScript
asc assembly/index.ts -o build/optimized.wasm -O3

# WABT
wat2wasm module.wat -o module.wasm
wasm2wat module.wasm -o module.wat

# Runtime
wasmtime run module.wasm
wasmer run module.wasm
```

### Performance Checklist

- [ ] Use streaming compilation
- [ ] Enable optimizations (-O3)
- [ ] Minimize JS ↔ Wasm calls
- [ ] Use SIMD when applicable
- [ ] Profile and benchmark
- [ ] Strip debug symbols for production
- [ ] Use wasm-opt for size reduction
- [ ] Consider threading for parallel work

---

## Summary

WebAssembly is a **game-changing technology** that brings:

✅ **Near-native performance** in web browsers
✅ **Multi-language support** (C/C++, Rust, Go, etc.)
✅ **Code reusability** across platforms
✅ **Safe execution** through sandboxing
✅ **Growing ecosystem** and tooling

**Use WebAssembly when**:
- Performance is critical
- Porting existing code
- CPU-intensive tasks
- Cross-platform requirements

**Stick with JavaScript when**:
- DOM manipulation
- Simple logic
- Rapid prototyping
- Build complexity is a concern

The future of web development is **JavaScript + WebAssembly** working together! 🚀
